from toy_abr_gym.trace_loader import TraceFeatures
from toy_abr_gym.trace_utils import get_dist_weights
from typing import Tuple
from toy_abr_gym.trace_generator import MIN_TRACE_LEN, BITRATES, Trace
import numpy as np
from toy_abr_gym.utils import make_reward_norm_func
from toy_abr_gym.sampler_utils import SharedPtsData, add_sample, get_probabilities, N_STEP
from toy_abr_gym.utils import SharedTestingData
import ray


SAMPLING_FUNCS = ["random", "plume_static", "iterative", "plume_dynamic"]

class Sampler:
    def __init__(self,
            np_random: np.random, 
            dataset: dict, 
            sampling_func_cls: str,
            trace_features: TraceFeatures = None, 
            shared_pts_data_name: str = None,
            shared_testing_data_name: str = None
            ) -> None:
        """
        The trace sampling module that will select the next trace for the environment.

        Args:
            np_random: The seeded numpy random module to use
            dataset: The dataset to sample from (see toy_abr_gym/trace_generator/generate_dataset)
            sampling_func: The name of the sampling strategy to use. One of ("random", "plume_static")
            trace_features: The features of the traces, if needed by the sampling_func
            shared_reward_pts_dict: The shared dictionary for reward plume_dynamic. 
                This allows all environments to see the benchmarked reward data of every other environment
            shared_testing_queue: The queue given by shared testing dict to sample from while iteratively sampling
                (see toy_abr_gym/utils/make_shared_testing_dict)
        """

        self.np_random = np_random
        self.dataset = dataset
        self.all_traces = dataset["traces"]
        if shared_pts_data_name is None:
            self.shared_pts_data = None
        else:
            self.shared_pts_data = ray.get_actor(shared_pts_data_name, namespace="plume_dynamic")
        if shared_testing_data_name is None:
            self.shared_testing_data = None
        else:
            self.shared_testing_data = ray.get_actor(shared_testing_data_name, namespace="test")
        self.current_trace_rewards = []
        self.trace_features = trace_features
        self.sampling_func_cls = sampling_func_cls
        if self.sampling_func_cls not in SAMPLING_FUNCS:
            raise NotImplementedError(f"Sampling Function must be one of {SAMPLING_FUNCS}")
        _, self.reward_norm_func = make_reward_norm_func(
            max_reward=BITRATES[-1], min_reward=BITRATES[0])
        self.is_init = True
        self.idx = -1
        self.probs = None
        self.compute_probs()


    def compute_probs(self):
        """
        If using a sampling_func that requires setting self.probs in weighted_sampling, set it up here
        """
        if self.sampling_func_cls == "plume_static" and self.is_init:
            if self.trace_features is None:
                raise ValueError("Attempted to use plume_static without passing in trace features.")
            dist = self.trace_features.cluster_dists
            self.probs = get_dist_weights(dist)
            self.is_init = False

        if self.sampling_func_cls == "plume_dynamic" and self.is_init:
            if self.shared_pts_data is None:
                raise ValueError(("Attempted to use plume_dynamic without passing"
                                " in a shared sampling actor"))

            from sklearn.preprocessing import scale

            self.scaled_trace_features = scale(self.trace_features.features)
            self.is_init = False

    def record_action_reward(self, action: int, reward: float,
            quality_reward: float = None, rebuf_reward: float=None) -> None:
        """
        Record the given action and reward into current_trace_data. To be used by Reward Based plume_dynamic if necessary.
        """
        if self.sampling_func_cls != "plume_dynamic":
            return
        else:
            reward = self.reward_norm_func(reward)
            self.current_trace_rewards.append(reward)

    def record_episode(self, idx: int) -> None:
        """
        Log the reward data observed during the episode (for plume_dynamic sampling)
        """
        if self.sampling_func_cls == "plume_dynamic":
            if len(self.current_trace_rewards) >= N_STEP:
                trace_feature = self.scaled_trace_features[idx]
                cluster_label = self.trace_features.cluster_labels[idx]
                rewards = np.array(self.current_trace_rewards, dtype=np.float64)
                add_sample(shared_pts_data=self.shared_pts_data, 
                    trace_feature=trace_feature, rewards=rewards, cluster=cluster_label)
            self.current_trace_rewards = []
        

    def weighted_sample(self, epsilon: float = 0.10) -> Tuple[Trace, int, int]:
        """
        Sample the Traces in an eplison-weighted way; 
            following self.probs and randomly picking the traces with eplison probability.
            Uses the shared reward_based_pts probs if necessary and also records self.current_trace_data if necessary
        Args:
            epsilon: The probability with which to sample randomly
        Returns:
            a Tuple of (Trace, start idx within the trace, index of trace in the dataset)
        """

        if self.np_random.random() < epsilon:
            p = None
        else:
            if self.sampling_func_cls == "plume_dynamic":
                p = get_probabilities(shared_pts_data = self.shared_pts_data)
            else:
                p = self.probs
                
        # sample
        self.idx = self.np_random.choice(len(self.all_traces), p=p)
        trace = self.all_traces[self.idx]

        # random start
        init_t_idx = self.np_random.choice(
            max(trace.network_data.shape[0] - MIN_TRACE_LEN, 1))
        return (trace, init_t_idx, self.idx)

    def iterative_sample(self) -> Tuple[Trace, int, int]:
        """
        Sample the Traces in an iterative way, trying to cover all the traces. 
            If shared_testing_queue is provided, dequeue the indices from it.
            
        Returns:
            a Tuple of (Trace, start idx within the trace, index of trace in the dataset)
        """
        if self.shared_testing_data is None:
            self.idx += 1
            if not 0 <= self.idx < len(self.all_traces):
                self.idx = 0
        else:
            self.idx = ray.get(self.shared_testing_data.get_index_to_test.remote())
        trace = self.all_traces[self.idx]
                
        # start at 0
        init_t_idx = 0
        return (trace, init_t_idx, self.idx)