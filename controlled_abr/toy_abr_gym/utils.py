import numpy as np
from typing import Tuple, Callable, Union, List
import ray
from ray.util.queue import Queue, Empty

def make_reward_norm_func(max_reward: float, min_reward: float,
            ) -> Tuple[Tuple[float, float], Callable]:
        """
        Make and return a normalization function for rewards.

        Args:
            max_reward: The highest reward possible in the environment
            min_reward: The lowest reward that is a baseline (i.e reward should be 0 if the agent achieves this performance)
        Returns:
            Absolute minimum reward
            normalized max reward
            The normalization function
        """
        support_size_max = int(np.sqrt(max_reward))
        support_size_min = -support_size_max
        def func(reward: float) -> float:
            reward -= min_reward
            reward =  np.sign(reward) * (
                (np.sqrt(np.abs(reward) + 1) - 1) + 0.001 * reward)
            return reward
        return ((support_size_min, func(max_reward)), func) 

def normalize(num: Union[float, np.ndarray], max: Union[float, np.ndarray]
        ) -> Union[float, np.ndarray]:
    """
    Normalize the num to (0, 1), clipping it if necessary [assumes minimum is 0.]

    Args:
        num: The number/array to normalize
        max: the maximum the number can reach
    Returns:
        The normalized num
    """
    normalized = num / max
    normalized = np.clip(normalized, 0., 1.)
    return normalized

def discount_n_step(rewards: np.ndarray, n_step: int, gamma: float) -> float:
    if rewards.ndim != 1:
        raise ValueError("Passed in rewards not in shape (n_steps..)")
    gammas = gamma ** np.arange(n_step)
    returns = rewards[np.newaxis, :n_step] @ gammas
    print(f"Saw rewards of len {len(rewards)}")
    return returns[0]


def format_float(num: float, precision: int = 4) -> str:
    """
    Create a string with num with the given precision (by rounding)
    Args:
        num: the number to format
        precision: the precision of the final string
    Returns:
        the string
    """
    return str(round(num, precision))


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


def terminate_shared_data(actor_ids: List[Tuple[str, str]]) -> None:
    """
    Forcefully terminate actors by their name (useful when ray instances do not die)
    Args:
        actor_ids: a list of name, namespace tuples for actors to terminate
    """
    for name, namespace in actor_ids:
        actor = ray.get_actor(name, namespace)
        ray.kill(actor)