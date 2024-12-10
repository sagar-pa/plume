import ray
import numpy as np
from typing import List, Tuple
from pathlib import Path
from abr_gym.utils import (
    Trace, make_reward_norm_func)
from abr_gym.trace_loader import load_trace, load_trace_features
from abr_gym.trace_downloader import MIN_TRACE_LEN
from abr_gym.trace_utils import (
    get_cluster_weights)
from abr_gym.sampler_utils import (get_probabilities, add_sample, N_STEP)


class Sampler:
    def __init__(self, np_random: np.random, 
            all_traces: List[Path], 
            trace_dir: Path, 
            metrics_file: Path, 
            starting_trace_idx: int = -1,
            sampling_func_cls: str = "random",
            reward_norm_style: str = "symmetric_sqrt_clip",
            trace_cache_name: str = None,
            shared_pts_data_name: str = None,
            shared_testing_data_name: str = None
            ) -> None:


        self.np_random = np_random
        self.all_traces = all_traces
        self.sampling_func_cls = sampling_func_cls
        self.trace_dir = trace_dir
        self.metrics_file = metrics_file
        self.idx = starting_trace_idx
        if trace_cache_name is None:
            self.trace_cache = None
        else:
            self.trace_cache = ray.get_actor(trace_cache_name, "cache")
        self.is_init = True
        self.probabilities = None
        if shared_pts_data_name is None:
            self.shared_pts_data = None
        else:
            self.shared_pts_data = ray.get_actor(shared_pts_data_name, "plume_dynamic")
        if shared_testing_data_name is None:
            self.shared_testing_data = None
        else:
            self.shared_testing_data = ray.get_actor(shared_testing_data_name, "test")
        self.current_trace_rewards = []

        *_, self.reward_norm_func = make_reward_norm_func(
            reward_norm_style, max_reward=30)
        if self.sampling_func_cls not in ["random", 
                "naive_weighted", "plume_static", "plume_dynamic", "iterative"]:
            raise NotImplementedError("Sampling strategy not yet implemented")
        if len(self.all_traces) == 0:
            raise ValueError("Cannot simulate the environment with no traces.")

        self.compute_probs()

    def compute_probs(self) -> None:
        if self.sampling_func_cls == "naive_weighted" and self.is_init:
            labels = np.ones(shape=(len(self.all_traces),), dtype=np.int64)
            trace_features = load_trace_features(all_traces=self.all_traces, 
                                trace_dir=self.trace_dir,
                                metrics_file=self.metrics_file)
            delivery_rates = trace_features.delivery_rates
            labels[delivery_rates <= 0.9822872859924906] = 0
            weights = get_cluster_weights(labels, max_pool_inflation=None)
            self.probabilities = weights
            self.is_init = False

        elif self.sampling_func_cls == "plume_static" and self.is_init:
            trace_features = load_trace_features(all_traces=self.all_traces, 
                                            trace_dir=self.trace_dir,
                                            metrics_file=self.metrics_file)
            class_weights = None
            weights = get_cluster_weights(
                trace_features.cluster_labels, class_weights=class_weights,
                max_pool_inflation=None)
            self.probabilities = weights
            self.is_init = False

        elif self.sampling_func_cls == "plume_dynamic" and self.is_init:           
            if self.shared_pts_data is None:
                raise ValueError(("Attempted to use plume_dynamic without passing"
                                " in a shared sampling actor"))

            from sklearn.preprocessing import scale
            trace_features = load_trace_features(all_traces=self.all_traces, 
                            trace_dir=self.trace_dir,
                            metrics_file=self.metrics_file)
            self.trace_features = scale(trace_features.features)
            self.cluster_dists = trace_features.cluster_dists
            self.cluster_labels = trace_features.cluster_labels
            self.is_init = False
        
    def record_action_reward(self, action: int, reward: float,
            quality_reward: float = None, rebuf_reward: float=None,
            change_reward: float=None) -> None:
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
                trace_feature = self.trace_features[idx]
                cluster_label = self.cluster_labels[idx]
                rewards = np.array(self.current_trace_rewards, dtype=np.float64)
                add_sample(shared_pts_data=self.shared_pts_data, 
                    trace_feature=trace_feature, rewards=rewards, cluster=cluster_label)
            self.current_trace_rewards = []



    def load_with_caching(self, idx: int) -> Trace:
        trace_path = self.all_traces[idx]
        if self.trace_cache is not None:
            trace = ray.get(self.trace_cache.get.remote(trace_path))
            if trace is None:
                trace, *_ = load_trace(trace_path)
                self.trace_cache.store.remote(trace_path, trace)
        else:
            trace, *_ = load_trace(trace_path)
        return trace


    def weighted_sample(self, epsilon: float = 0.10) -> Tuple[Trace, int, int]:
        if self.np_random.random() < epsilon:
            p = None
        else:
            if self.sampling_func_cls == "plume_dynamic":
                p = get_probabilities(shared_pts_data = self.shared_pts_data)
            else:
                p = self.probabilities

        # sample
        trace = None
        while trace is None:
            try:
                idx = self.np_random.choice(len(self.all_traces), p=p)
                trace = self.load_with_caching(idx)
            except (AssertionError, IndexError, ValueError):
                raise ValueError((f"Invalid Trace Detected: {self.all_traces[idx]}. "
                            "Please clean traces before proceeding."))

        self.idx = idx

        # random start
        init_t_idx = self.np_random.choice(
            max(trace.network_data.shape[0] - MIN_TRACE_LEN, 1))
        return trace, init_t_idx, idx

    def iterative_sample(self) -> Tuple[Trace, int, int]:
        trace = None
        while trace is None:
            try:
                if self.shared_testing_data is not None:
                    self.idx = ray.get(self.shared_testing_data.get_index_to_test.remote())
                else:
                    self.idx += 1
                    if not 0 <= self.idx < len(self.all_traces):
                        self.idx = 0
                trace = self.load_with_caching(self.idx)
            except (AssertionError, IndexError, ValueError):
                raise ValueError((f"Invalid Trace Detected: {self.all_traces[self.idx]}. "
                            "Please clean traces before proceeding."))
        
        # start at 0
        init_t_idx = 0
        return trace, init_t_idx, self.idx