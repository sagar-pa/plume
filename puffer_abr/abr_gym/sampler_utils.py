import ray
from typing import Callable, Tuple
from torch import nn
import numpy as np
import torch as th
from abr_gym.utils import discount_n_step
from abr_gym.trace_utils import get_dist_weights
from abr_gym.trace_downloader import MIN_TRACE_LEN
from pathlib import Path
import json

BATCH_SIZE = 64
MAX_HISTORY_LEN = 50000
CLUSTER_HISTORY = 5000
OPTIMIZER_CLS = th.optim.Adam
OPTIMIZER_KWARGS = dict(lr=1e-3)
LOSS_FUNC_CLS = nn.HuberLoss
LOSS_FUNC_KWARGS = dict(delta=1)
RECOMPUTE_FREQ = 75000
PTS_CHECKPOINT = 150000
GAMMA = 0.999
N_STEP = MIN_TRACE_LEN - 6

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.001)

def create_model(feature_dim: int) -> Tuple[nn.Module, Callable]:
    """
    Setup and return the model for trace to return predictions.
    Args:
        feature_dim: the number of features of a trace
    Returns:
        (The model, Loss function)
    """
    model = nn.Sequential(
        nn.Linear(feature_dim, 32),
        nn.ReLU(),
        nn.Linear(32, 32),
        nn.ReLU(),
        nn.Linear(32, 1)
    )
    model = model.apply(init_weights)
    loss_func = LOSS_FUNC_CLS(**LOSS_FUNC_KWARGS)
    return (model, loss_func)

@ray.remote(num_cpus=0.25)
class SharedPtsData:
    def __init__(self, alpha: float, 
            trace_features: np.ndarray, trace_dists: np.ndarray, log_dir: Path):
        log_dir.mkdir(parents=True, exist_ok=True)
        self.train_input = []
        self.train_output = []
        self.alpha = alpha
        self.log_dir = log_dir
        self.trace_dists = trace_dists

        trace_feature_dim = trace_features.shape[1]
        n_clusters = trace_dists.shape[1]
        model, loss_func = create_model(feature_dim=trace_feature_dim)
        self.model = model
        self.loss_func = loss_func

        clustered_input = [[] for _ in range(n_clusters)]
        clustered_output = [[] for _ in range(n_clusters)]
        optimizer = OPTIMIZER_CLS(self.model.parameters(), **OPTIMIZER_KWARGS)

        self.optimizer = optimizer
        self.clustered_input = clustered_input
        self.clustered_output = clustered_output
        self.probabilities = None
        self.samples_trained_on = 0
        self.samples_since_recompute = 0
        self.currently_computing_weights = False
        self.currently_training = False

    def get_probs(self) -> np.ndarray:
        self.samples_trained_on += 1
        self.samples_since_recompute += 1
        return self.probabilities

    def check_should_compute(self) -> bool:
        should_compute = self.samples_trained_on >= PTS_CHECKPOINT and \
            self.samples_since_recompute >= RECOMPUTE_FREQ and \
                        not self.currently_computing_weights
        if should_compute:
            self.currently_computing_weights = True
        return should_compute

    def broadcast_finish_computing_weights(self):
        self.currently_computing_weights = False
        self.samples_since_recompute = 0
    
    def check_should_train(self) -> bool:
        should_train = len(self.train_input) >= MAX_HISTORY_LEN and \
                            not self.currently_training
        if should_train:
            self.currently_training = True
        return should_train

    def broadcast_finish_training(self):
        self.currently_training = False

    def add_sample(self, feature: np.ndarray, returns: float, cluster: int) -> None:
        self.train_input.append(feature)
        self.train_output.append(returns)
        self.clustered_input[cluster].append(feature)
        self.clustered_output[cluster].append(returns)

    def process_batch(self):
        """
        Train the model with the input provided, and clear the training data.
        """
        max_len = len(self.train_input)
        if len(self.train_input) != len(self.train_output):
            raise ValueError("History len mismatch found while processing batch.")
        model_input = np.array(self.train_input, dtype=np.float32)
        returns = np.array(self.train_output, dtype=np.float32)
        self.train_input.clear()
        self.train_output.clear()
        indices = np.array_split(np.arange(max_len, dtype=np.int64), min(int(max_len / BATCH_SIZE), 1))

        for batch_indices in indices:
            input_batch = th.as_tensor(model_input[batch_indices], dtype=th.float32)
            output_batch = th.as_tensor(returns[batch_indices], dtype=th.float32)
            preds = self.model(input_batch)
            loss = self.loss_func(preds, output_batch)

            #backprop
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def calculate_class_weights(self) -> np.ndarray:
        """
        Calculate the class weights for selection, according to the formula 
            weights = weight for cluster = 
                [alpha * avg return error + (1-alpha) * -avg returns]
        Returns:
            The calculated class weights, a numpy array of shape (n_clusters, )
        """
        from sklearn.preprocessing import minmax_scale

        n_clusters = len(self.clustered_input)
        error_weights = np.zeros(shape=(n_clusters, ), dtype=np.float64)
        return_weights = np.zeros(shape=(n_clusters, ), dtype=np.float64)
        for cluster in range(n_clusters):
            n_samples = len(self.clustered_input[cluster])
            if len(self.clustered_input[cluster]) != len(self.clustered_output[cluster]):
                raise ValueError("History len mismatch found while calculating weights.")
            if n_samples < 1:
                raise ValueError((f"Did not have a any samples of returns in "
                                    f"cluster: {cluster}. Please set a higher "
                                    "sampling checkpoint to ensure at "
                                    "least 1 sample in present."))
            elif n_samples > CLUSTER_HISTORY:
                len_to_del = n_samples - CLUSTER_HISTORY
                self.clustered_input[cluster] = self.clustered_input[cluster][len_to_del:]
                self.clustered_output[cluster] = self.clustered_output[cluster][len_to_del:]
            model_input = np.array(self.clustered_input[cluster], dtype=np.float32)
            model_input = th.as_tensor(model_input, dtype=th.float32)
            returns = np.array(self.clustered_output[cluster], dtype=np.float32)
            returns = th.as_tensor(returns, dtype=th.float32)
            mean_returns = th.mean(returns)
            with th.no_grad():
                preds = self.model(model_input)
            loss = self.loss_func(preds, returns)
            error_weights[cluster] = loss.item() # non-negative
            return_weights[cluster] = - mean_returns.item()
        error_weights /= error_weights.sum()
        return_weights = minmax_scale(return_weights, feature_range=(1, 10))
        return_weights /= return_weights.sum()
        weights = (self.alpha * error_weights) + ((1-self.alpha) * return_weights) 
        weights = weights / weights.sum()

        probs = get_dist_weights(dist=self.trace_dists, class_weights=weights)
        self.probabilities = probs
        with open(self.log_dir / f"{self.samples_trained_on}.json", "w") as f:
            json.dump(list(weights), f)
        
def add_sample(shared_pts_data: SharedPtsData,
        trace_feature: np.ndarray, 
        rewards: np.ndarray,
        cluster: int) -> None:
    """
    Add the return sample to the dictionary, processing the train batch if necessary
    Args:
        shared_pts_data: The shared sampling Actor
        trace_feature: The trace features (input)
        rewards: The array of rewards of the trace
        cluster: The cluster the trace belongs to
    """
    returns = discount_n_step(rewards=rewards, n_step=N_STEP, gamma=GAMMA)
    shared_pts_data.add_sample.remote(
        feature=trace_feature, returns=[returns], cluster=cluster)
    should_train = ray.get(shared_pts_data.check_should_train.remote())
    if should_train:
        shared_pts_data.process_batch.remote()
        shared_pts_data.broadcast_finish_training.remote()


def get_probabilities(shared_pts_data: SharedPtsData) -> np.ndarray:
    """
    Get the shared probabilities in the shared_pts_data, recomputing them if necessary
    Args:
        shared_pts_data: The shared sampling Actor
    Returns:
        the probabilities, an array of shape [n_tracees, ]
    """
    should_compute = ray.get(shared_pts_data.check_should_compute.remote())
    if should_compute:
        shared_pts_data.calculate_class_weights.remote()
        shared_pts_data.broadcast_finish_computing_weights.remote()
    return ray.get(shared_pts_data.get_probs.remote())