import numpy as np
from typing import (
    Callable, Tuple, Union, 
    Iterable, Hashable, NamedTuple, List, Optional, Any
    )
from collections import deque
from decimal import Decimal
import ray
from ray.util.queue import Queue, Empty

STALL_RATIO_THRESHOLD = 0.015
SAFETY_VIOLATION_MULTIPLIER = 500

class Trace(NamedTuple):
    network_data: np.ndarray
    video_sizes: np.ndarray
    ssim_dbs: np.ndarray
    delivery_rates: List[Decimal]


class TraceInfo(NamedTuple):
    actions: np.ndarray
    buffers: np.ndarray
    rebufs: np.ndarray

REWARD_NORM_STYLES = {
    "symmetric_clip", "zero_clip", "symmetric_sqrt_clip", "zero_sqrt_clip", 
    "shifted_sqrt_clip", "none", "div_clip", "asym_clip"
}

def make_reward_norm_func(style: str, 
        max_reward: float, min_reward: float = None,
        ) -> Tuple[Tuple[float, float], Callable]:
    """
    Make a reward normalization function 
        (a callable which takes a reward and returns the normalized version of it).
    Args:
        style: the 'style' of normalization to create. One of 
            ["symmetric_clip", "zero_clip", "symmetric_sqrt_clip", 
                "zero_sqrt_clip", "shifted_sqrt_clip", 
                "none", "div_clip", "asym_clip"]
        max_reward: The maximum expected value of reward (for clipping)
        min_reward: The minimum value of reward (for clipping)
    Returns:
        a Tuple of [[min normalized reward, max normalized reward], norm function]
    """
    if not isinstance(style, str):
        raise TypeError("Reward norm style is specified in a str")
    style = style.strip().lower()
    if style not in REWARD_NORM_STYLES:
        raise NotImplementedError("Reward norm style not implemented")
    if style == "none":
        def func(reward: float) -> float: # Avoid lambda functions to keep pickling enabled
            return reward
        return ((-max_reward, max_reward), func)
    if "sqrt" in style:
        if not "shifted" in style:
            support_size_max = int(np.sqrt(max_reward))
            support_size_min = 0 if "zero" in style else -support_size_max 
            def func(reward: float) -> float:
                reward =  np.sign(reward) * (
                    (np.sqrt(np.abs(reward) + 1) - 1) + 0.001 * reward)
                reward = np.clip(reward, support_size_min, support_size_max)
                return reward
            return ((support_size_min, support_size_max), func)
        else:
            support_size_max = int(np.sqrt(max_reward) / 2.)
            support_size_min = -support_size_max - 0.5
            if min_reward is None:
                min_reward = support_size_max / 2. 
            def func(reward: float) -> float:
                reward -= min_reward
                reward =  np.sign(reward) * (
                    (np.sqrt(np.abs(reward) + 1) - 1) + 0.001 * reward)
                reward /= 2.25
                reward = np.clip(reward, support_size_min, support_size_max)
                return reward
            return ((support_size_min, support_size_max), func) 
    elif "div" in style:
        support_size_max = int(max_reward / 5)
        support_size_min = int(-support_size_max + support_size_max/6)
        def func(reward: float) -> float:
            reward /= support_size_max
            reward = np.clip(reward, support_size_min, support_size_max)
            return reward
        return ((support_size_min, support_size_max), func)
    elif "asym" in style:
        support_size_max = int(max_reward / 5)
        support_size_min = int(-1.5*support_size_max)
        def func(reward: float) -> float:
            reward /= support_size_max
            reward = np.clip(reward, support_size_min, support_size_max)
            return reward
        return ((support_size_min, support_size_max), func)
    else:
        support_size_max = max_reward
        support_size_min = 0 if "zero" in style else -support_size_max 
        def func(reward: float) -> float:
            reward = np.clip(reward, support_size_min, support_size_max)
            return reward
        return ((support_size_min, support_size_max), func)

class LongtermFeatureExtractor:
    def __init__(self, vars: Iterable[Hashable], max_vals: Iterable[float], 
            output_dim: int, history_len: int=None, 
            function: Union[Callable, Iterable[Callable]]=None):
        """
        Create a Feature Extractor that keeps track of features over a long history len,
            reshapes it a multiple of output_dim, applies aggregate function (e.g mean, sum, max) to it,
            and returns it as output_dim.
            Example: 
                history_len: 9, output_dim: 3, vals: [1,2,3,4,5,6,7,8,9], 
                    function: max, max_vals = 10
                observe() output: [3/10, 6/10, 9/10]

        Args:
            vars: list of keys in info dict to keep track of
            max_vals: The corresponding list of maximum 
                expected values of each var (for normalization)
            output_dim: The output dim to produce in observing
            history_len: The total history length to keep track (must be divisible by output_dim)
            function: The aggregator function to apply to reshaped history

        """
        self.active = True
        self.output_dim = output_dim
        if len(vars) == 0:
            self.active = False
        if self.active:
            if isinstance(function, Iterable):
                if len(function) != len(vars):
                    raise ValueError("Number of functions not equal to values to keep track of")
            else:
                function = [function] * len(vars)
            if history_len % output_dim != 0:
                raise ValueError("History cannot be equally divided into output shape")
            if len(max_vals) != len(vars):
                raise ValueError("List of values to track and their max values do not match")
            self.max_vals = max_vals
            self.history = [deque(maxlen=history_len) for _ in range(len(vars))]
            self.vars = vars
            self.n_vars = len(vars)
            self.function = function
        self.reset()

    def normalize(self, raw: Union[float, np.ndarray],
            max: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        norm = raw / max
        norm = np.clip(norm, 0, 1)
        return norm

    def reset(self) -> None:
        if self.active:
            for history in self.history:
                for _ in range(history.maxlen):
                    history.append(0)
    
    def record(self, info: dict) -> None:
        """
        Take the info dict from environment and record it to history
        Args:
            info: the gym environment's info dictionary [it must have self.vars as keys]
        """
        if self.active:
            for var, history in zip(self.vars, self.history):
                history.append(info[var])
    
    def observe(self) -> np.ndarray:
        """
        Take the history of vars, reshape it, and aggregate it
        Returns:
            The reshape and aggregated history of vars, array of shape [len(self.vars), 1]
        """
        if self.active:
            output = np.zeros(shape=(self.output_dim, self.n_vars))
            for j, history in enumerate(self.history):
                data = []
                max_size = history.maxlen // self.output_dim
                i = 0
                for num in history:
                    data.append(num)
                    if len(data) == max_size:
                        output[i, j] = self.normalize(
                            raw=self.function[j](data), max=self.max_vals[j])
                        i += 1
                        data.clear()
        else:
            output = np.zeros(shape=(self.output_dim, 0))
        return output


def discount_n_step(rewards: np.ndarray, n_step: int, gamma: float) -> float:
    """
    Compute the truncated discounted return of the first state in the reward array.
    Args:
        rewards: the undiscounted rewards to process
        n_step: the maximum horizon to enforce. Rewards after n_step are assumed to be zero.
        gamma: the discount factor
    Returns:
        The discounted return
    """
    if rewards.ndim != 1:
        raise ValueError("Passed in rewards not in shape (n_step..)")
    gammas = gamma ** np.arange(n_step)
    returns = rewards[np.newaxis, :n_step] @ gammas
    return returns[0]


@ray.remote(num_cpus=0.01)
class SharedTestingData:
    def __init__(self, n_traces: int):
        self.train_progress = 0
        self.n_traces = n_traces
        self.shared_testing_indices = Queue(actor_options=dict(num_cpus=0.01))
        self.shared_testing_indices.put_nowait_batch(list(range(n_traces)))
    
    def get_index_to_test(self) -> int:
        try:
            idx = self.shared_testing_indices.get_nowait()
        except Empty:
            idx = np.random.randint(low=0, high=self.n_traces)
        return idx

    def get_train_progress(self) -> int:
        return self.train_progress

    def set_train_progress(self, train_progress: int):
        self.train_progress = int(train_progress)
        while not self.shared_testing_indices.empty():
            try:
                self.shared_testing_indices.get_nowait_batch(len(self.shared_testing_indices))
            except Empty:
                break
        self.shared_testing_indices.put_nowait_batch(list(range(self.n_traces)))


@ray.remote(num_cpus=0.01)
class Cache:
    def __init__(self, max_size: int = None):
        self.max_size = max_size
        self.cache = {}
    
    def store(self, key: Any, value: Any):
        """
        Store the key, value pair in the shared dictionary
        """
        if key not in self.cache:
            self.cache[key] = value
        if self.max_size is not None:
            while len(self.cache) > self.max_size:
                try:
                    last_added_key = next(iter(self.cache))
                    del self.cache[last_added_key]
                except KeyError:
                    pass

    def get(self, key: str, default = None) -> Optional[Any]:
        return self.cache.get(key, default)


def terminate_shared_data(actor_ids: List[Tuple[str, str]]) -> None:
    """
    Forcefully terminate actors by their name (useful when ray instances do not die)
    Args:
        actor_ids: a list of name, namespace tuples for actors to terminate
    """
    for name, namespace in actor_ids:
        actor = ray.get_actor(name, namespace)
        ray.kill(actor)