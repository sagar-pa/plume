import numpy as np
from typing import Union, Dict, Tuple, Optional
from mpc_c.mpc import take_action_py
from time import time_ns

NAX_BUFFER = 15
MAX_SIZE = 3
MAX_QUALITY = 25
N_ACTIONS = 10
ACTION_SPACE = np.arange(N_ACTIONS, dtype=np.int64)

class BatchedAgent:
    def __init__(self, policy_name: str, n_envs: int) -> None:
        self.policies = [get_policy(policy_name) for _ in range(n_envs)]
        self.max_obs = np.array([MAX_QUALITY, MAX_SIZE, 20, 
            15, 10, 3])
    
    def process_obses(self, obs: np.ndarray) -> np.ndarray:
        unnormalizzed_obs = np.array(obs, copy=True)
        unnormalizzed_obs[:, :, 0:6] *= self.max_obs
        unnormalizzed_obs[:, 0:5, 6:16] *= MAX_SIZE
        unnormalizzed_obs[:, 0:5, 16:26] *= MAX_QUALITY
        return unnormalizzed_obs
        
    def predict(self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        
        n_envs = observation.shape[0]
        unnormalized_obses = self.process_obses(observation)
        actions = np.zeros(shape=(n_envs), dtype=np.int64)
        for env_idx in range(n_envs):
            action = self.policies[env_idx](unnormalized_obses[env_idx])
            actions[env_idx] = action
        
        return actions, []


def get_policy(policy_name: str):
    if policy_name == "linear_bba":
        return LinearBBA()
    elif policy_name == "bola_basic_v1":
        return BolaBasic(version=1)
    elif policy_name == "bola_basic_v2":
        return BolaBasic(version=2)
    elif policy_name == "mpc":
        return MPC()
    elif policy_name == "random":
        return RandomAgent()
    elif policy_name == "optimistic":
        return OptimisticAgent()
    else:
        raise ValueError
    
    
class RandomAgent:
    def __init__(self) -> None:
        seed_generator = np.random.default_rng(seed=time_ns())
        seed = seed_generator.choice(1000000000)
        self.rand = np.random.default_rng(seed=seed)

    def __call__(self, state: np.ndarray) -> int:
        return self.rand.choice(N_ACTIONS)
    
class OptimisticAgent:
    def __init__(self) -> None:
        pass

    def __call__(self, state: np.ndarray) -> int:
        return N_ACTIONS - 1 
        

class LinearBBA:
    EPSILON = 0.05
    def __init__(self, lower=3, upper=13.5):
        self.lower = lower
        self.upper = upper

    def __call__(self, state: np.ndarray) -> int:
        buffer = state[-1, 3]
        next_chunk_sizes = state[0, 6:16]
        next_ssims = state[0, 16:26]

        if buffer < self.lower:
            action = ACTION_SPACE.min()
        elif buffer >= self.upper:
            action = ACTION_SPACE.max()
        else:
            min_size = next_chunk_sizes[ACTION_SPACE.min()]
            max_size = next_chunk_sizes[ACTION_SPACE.max()]
            slope = (max_size - min_size) / (self.upper - self.lower)
            max_valid_size = min_size + slope * (buffer - self.lower)
            mask = next_chunk_sizes <= max_valid_size
            action = next_ssims[mask].argmax()
        return action

class BolaBasic:
    SIZE_LADDER = [44319, 93355, 115601, 
                         142904, 196884, 263965,
                         353752, 494902, 632193, 889893]
    SSIM_LADDER = [0.91050748, 0.94062527, 0.94806355, 
                         0.95498943, 0.96214503, 0.96717277,
                         0.97273958, 0.97689813, 0.98004106, 0.98332605]
    MIN_BUF_S = 3
    MAX_BUF_S = 15
    MIN_SSIM = 0
    MAX_SSIM = 60
    CHUNK_LENGTH = 2.002

    def __init__(self, version: int, is_causal_sim_version: bool = False):
        self.version = version

        smallest = {'size': self.SIZE_LADDER[0],
                    'utility': self.utility(self.SSIM_LADDER[0])}
        second_smallest = {'size': self.SIZE_LADDER[1],
                           'utility': self.utility(self.SSIM_LADDER[1])}
        largest = {'size': self.SIZE_LADDER[-1],
                   'utility': self.utility(self.SSIM_LADDER[-1])}

        size_delta = self.SIZE_LADDER[1] - self.SIZE_LADDER[0]
        if version == 1:
            utility_high = largest['utility']
        else:
            utility_high = self.utility(1)
        
        if is_causal_sim_version:
            self.Vp = 0.80502
            self.gp = 1.06592
        else:
            size_utility_term = second_smallest['size'] * smallest['utility'] - smallest['size'] * \
                                second_smallest['utility']
            gp_nominator = self.MAX_BUF_S * size_utility_term - utility_high * self.MIN_BUF_S * size_delta
            gp_denominator = ((self.MIN_BUF_S - self.MAX_BUF_S) * size_delta)
            self.gp = gp_nominator / gp_denominator
            self.Vp = self.MAX_BUF_S / (utility_high + self.gp)

    def utility(self, ssim_index):
        """

        :param ssim_index:
        :type ssim_index: float or np.ndarray
        :return:
        :rtype: float or np.ndarray
        """

        if self.version == 1:
            return np.where(ssim_index == 1, self.MAX_SSIM, np.clip(-10 * np.log10(1 - ssim_index),
                                                                    a_min=self.MIN_SSIM,
                                                                    a_max=self.MAX_SSIM))
        else:
            return ssim_index

    def objective(self, utility, size, buffer_in_chunks):
        """

        :param utility:
        :type utility: float or np.ndarray
        :param size:
        :type size: float or np.ndarray
        :param buffer_in_chunks:
        :type buffer_in_chunks: float
        :return:
        :rtype: float or np.ndarray
        """
        return (self.Vp / self.CHUNK_LENGTH * (utility + self.gp) - buffer_in_chunks) / size

    def choose_max_objective(self, format_sizes, format_ssims, buffer_in_chunks):
        """

        :param format_sizes:
        :type format_sizes: np.ndarray
        :param format_ssims:
        :type format_ssims: np.ndarray
        :param buffer_in_chunks:
        :type buffer_in_chunks: float
        :return:
        :rtype: (int, float, float, float)
        """
        objs = self.objective(self.utility(format_ssims), format_sizes, buffer_in_chunks)
        chosen_index = np.argmax(objs)
        return chosen_index, format_sizes[chosen_index], format_ssims[chosen_index], objs[chosen_index]

    def choose_max_scaled_utility(self, format_sizes, format_ssims):
        """

        :param format_sizes:
        :type format_sizes: np.ndarray
        :param format_ssims:
        :type format_ssims: np.ndarray
        :return:
        :rtype: (int, float, float)
        """
        chosen_index = np.argmax(self.utility(format_ssims) + self.gp)
        return chosen_index, format_sizes[chosen_index], format_ssims[chosen_index]

    def __call__(self, state: np.ndarray) -> int:
        """

        :param index_curr:
        :type index_curr: int
        :param buffer:
        :type buffer: float
        :return: Return the action index, size and ssim
        :rtype: (int, float, float)
        """
        buffer = state[-1, 3]
        size_arr_valid = state[0, 6:16] * 1e6
        ssim_arr_valid = state[0, 16:26]
        buffer_in_chunks = buffer / self.CHUNK_LENGTH

        max_obj_index, _, __, max_obj = self.choose_max_objective(size_arr_valid,
                                                                    ssim_arr_valid,
                                                                    buffer_in_chunks)

        if self.version == 1 or max_obj >= 0:
            return ACTION_SPACE[max_obj_index]
        else:
            max_util_index, _, __ = self.choose_max_scaled_utility(size_arr_valid,
                                                                    ssim_arr_valid)
            return ACTION_SPACE[max_util_index]
        
def safe_divide(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.divide(a, b, out=np.zeros_like(b), where=b!=0)
        
class MPC:
    def __init__(self):
        self.mpc_lookback = 5
        self.mpc_lookahead = 5
        self.past_chunk_sizes = None
        self.act_len = 10
        self.mpc_lookback = 5
        self.eps = 1e-6
        self.vid_bit_rate = np.array([.2, .3, .45, .75, 1.2, 1.85, 2.85, 4.3, 6.0, 8.0])
        self.rebuf_penalty = 4.3
        
    
    def __call__(self, obs: np.ndarray) -> int:
        """
            observation_array[i + 1, 10] = next_buffer
            observation_array[i + 1, 12] = act
            observation_array[i + 1, 0:4] = observation_array[i, 1:5]
            observation_array[i + 1, 4] = selected_size / download_time
            observation_array[i + 1, 5:9] = observation_array[i, 6:10]
            observation_array[i + 1, 9] = download_time
        
        """
        past_chunk_download_times = obs[:, 2]
        past_chunk_sizes = obs[:, 1]
        past_chunk_throughputs = safe_divide(past_chunk_sizes, past_chunk_download_times)
        past_throughput = past_chunk_throughputs[-1]
        buffer = obs[-1, 3]
        
        future_chunk_sizes = obs[:5, 6:16]
        if self.past_chunk_sizes is None:
            self.past_chunk_sizes = future_chunk_sizes[0]
        
        past_action = np.argmin(np.abs(self.past_chunk_sizes - past_chunk_sizes[-1]))
        
        
        obs_arr = [past_chunk_throughputs[-i] for i in range(self.mpc_lookback, 0, -1)]
        obs_arr.extend([past_chunk_download_times[-i] for i in range(self.mpc_lookback, 0, -1)])
        obs_arr.extend([buffer, 10, past_action])
        
        for chunk_idx in range(self.mpc_lookahead):
            obs_arr.extend(future_chunk_sizes[chunk_idx, i] for i in range(N_ACTIONS))

        for i in range(N_ACTIONS):
            obs_arr.append(past_throughput)
            
        past_delays = safe_divide(1, safe_divide(past_throughput, self.past_chunk_sizes))
        for i in range(N_ACTIONS):
            obs_arr.append(past_delays[i])
        
        obs_arr = np.array(obs_arr)
            
        return take_action_py(obs_arr, self.act_len, self.vid_bit_rate, self.rebuf_penalty, self.mpc_lookback,
                              self.mpc_lookahead, self.eps)
         