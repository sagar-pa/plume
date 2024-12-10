import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
from pathlib import Path
from typing import Tuple, List, NamedTuple

from abr_gym.trace_loader import load_traces
from abr_gym.sampler import Sampler
from abr_gym.trace_downloader import MAX_BUFFER

DIR_PATH = Path.cwd()

class RewardWeights(NamedTuple):
    quality_weight: float
    rebuf_weight: float
    quality_change_weight: float

class ABRSimEnv(gym.Env):
    def __init__(self, seed=None, trace_dir: Path = None, 
            selection_file: Path = None, metrics_file: Path = None,
            use_ssim: bool=True, split: str="train", max_traces: int = 100000, 
            sampling_func_cls: str = "random", starting_trace_idx: int = -1, 
            reward_weights: Tuple[float, float, float] = (1, 100, 1), 
            sampler_kwargs: dict = None,
            max_trace_k: int = 1) -> None:
        if trace_dir is None:
            trace_dir = DIR_PATH / "traces"
        if selection_file is None:
            selection_file = DIR_PATH / "parameters" / "train_select.json"
        if metrics_file is None:
            metrics_file = DIR_PATH / "parameters" / "precomputed_trace_metrics.json"
        
        all_traces=load_traces(trace_dir, selection_file, split)
        all_traces = all_traces[:max_traces]

        self._test = split.upper() in ["TEST", "WHOLE"]
        if self._test: 
            sampling_func_cls = "iterative"
            max_trace_k = 1
        if sampler_kwargs is None:
            sampler_kwargs = {}
        self._sampler = Sampler(np_random=None, all_traces=all_traces, 
            trace_dir=trace_dir, metrics_file=metrics_file,
            starting_trace_idx=starting_trace_idx, 
            sampling_func_cls=sampling_func_cls, **sampler_kwargs)
        if sampling_func_cls == "iterative":
            self.sampling_function = self._sampler.iterative_sample
        else:
            self.sampling_function = self._sampler.weighted_sample

        super().reset(seed=seed)
        self._sampler.np_random = self.np_random

        # observation and action space
        self.setup_space()
        self.bitrate_map = [.2,.3,.45,.75,1.2,1.85,2.85,4.3,6.0,8.0]
        self.use_ssim = use_ssim
        self.max_trace_k = max(max_trace_k, 1)
        self.curr_trace_k = -1
        self.trace_idx = None
        self.setup_rewards(reward_weights=reward_weights)

    def setup_rewards(self, reward_weights: Tuple[float, float, float]) -> None:
        """
        Set the Reward Weights used for ABR.

        Args:
            reward_weights: in absolute value [Quality Weight, Change Weight, Rebuf Weight]
        """
        self.reward_weights = RewardWeights(*reward_weights)

    def setup_space(self) -> None:
        """
        Setup Gym Spaces to finish registering the environment.
        """
        self.obs_low = np.array([0] * 110, dtype=np.float32)
        self.obs_high = np.array([1e4] * 110, dtype=np.float32)
        self.observation_space = spaces.Box(
            low=self.obs_low, high=self.obs_high)
        self.action_space = spaces.Discrete(10)


    def circular_increment(self, num: int, max: int) -> int:
        """
        Circularly increment num, resetting it to 0 if the next number equals max.

        Args: 
            num: The number to be incremented by 1
            max: The cap to compare it against
        Returns:
            The incremented number
        """
        if num + 1 == max:
            return 0
        else:
            return num + 1
    
    def get_chunk_time(self, network_data: np.ndarray, t_idx: int) -> float:
        """
        Calculate how long the current t_idx will last according to the network data.
        The time is calculated according to when the next throughput is logged to set.

        Args:
            network_data: The network data array of throughputs and times loaded by load_trace
            t_idx: The index to calculate value for
        Returns:
            The time that current throughput lasts for.

        """
        if t_idx == network_data.shape[0] - 1:
            return 1  # bandwidth last for 1 second
        else:
            time = network_data[t_idx + 1, 0] - network_data[t_idx, 0]
            return np.clip(time, 0, 20)

    def observe(self) -> np.ndarray:
        """
        Create and return an observatio. The obervation is an array of the client
        data, with the next 5 video sizes and SSIMs flattened and concatenated at the end.
        (Note that SSIMs are returned even when the environment's reward function does not use them.)
        """        
        buffer_size =  self.buffer_size if self.buffer_size is not None else 0.

        obs_arr = np.array([self.past_quality,
                    self.past_video_size,
                    self.past_delay,
                    buffer_size,
                    self.past_reward])
        # ignore time and throughput
        network_data = self.trace.network_data[self.curr_t_idx, 2:]
        # next 5 video sizes and ssim dbs 
        video_sizes = self.trace.video_sizes[self.video_idx: self.video_idx + 5, :].flatten()
        ssim_dbs = self.trace.ssim_dbs[self.video_idx: self.video_idx + 5, :].flatten()
        return np.concatenate((obs_arr, network_data, video_sizes, ssim_dbs))

    def reset(self, seed: int = None, options: dict = None) -> Tuple[np.ndarray, dict]:
        """
        Reset the environment. Sample a new trace and starting index if needed,
        and set the buffer and elements for reward calculations.
        """
        super().reset(seed=seed)
        self._sampler.np_random = self.np_random
        if self.trace_idx is not None:
            self._sampler.record_episode(idx = self.trace_idx)
        if self.curr_trace_k <= 0:
            self.trace, self.start_t_idx, self.trace_idx  = self.sampling_function()
            self.curr_trace_k = self.max_trace_k
        self.curr_trace_k -= 1
        self.curr_t_idx = self.start_t_idx
        self.network_time_left = self.get_chunk_time(
            self.trace.network_data, self.curr_t_idx)
        self.video_idx = self.curr_t_idx
        self.total_video_chunks = self.trace.video_sizes.shape[0]
        self.total_network_ts = self.trace.network_data.shape[0]
        self.buffer_size = None  # initial download time not counted
        self.past_quality = 0.
        self.past_delay = 0.
        self.past_video_size = 0.
        self.past_reward = 0.
        self.curr_time_sec = 0.
        return self.observe(), {'curr_t_idx': self.curr_t_idx,
                                'trace_idx': self.trace_idx}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """
        The main transition function of the environment.
        It takes the current environment state (buffer, throughput, action, chunk size and SSIM),
        and calculates what the next step should be. The core of function calculates
        how long the current chunk will take to send in the network by iteratively subtracting
        the size of the chunk by how much we can send with the current throughput
        (video_size - (throughput * time this throughput is available for)).

        Args:
            action: The index of the chunk to next send
        Returns:
            The tuple (Obervation, Reward, end of episode or not, information)
        """

        assert self.action_space.contains(action)

        video_size = self.trace.video_sizes[self.video_idx, action]

        # compute chunk download time based on trace
        delay = 0  # in seconds
        time_sent = self.curr_time_sec

        while video_size > 1e-8:  # floating number business
            if self.network_time_left <= 0:
                self.curr_t_idx = self.circular_increment(
                    self.curr_t_idx, self.total_network_ts)
                self.network_time_left = self.get_chunk_time(
                    self.trace.network_data, self.curr_t_idx)
            
            throughput = self.trace.network_data[self.curr_t_idx, 1]

            network_time_used = min(self.network_time_left, video_size / throughput)

            video_size -= throughput * network_time_used
            self.network_time_left -= network_time_used
            delay += network_time_used

        #Add noise
        if not self._test:
            noise = np.clip(1 + self.np_random.normal(0,.03), 0.95, 1.05)
        else:
            noise = 1
        delay *= noise

        if self.buffer_size is None:
            rebuffer_time = 0
            self.buffer_size = 0
        else:
            # compute rebuffering
            rebuffer_time = np.clip(delay - self.buffer_size, 0, 20)

        # update buffer
        self.buffer_size = max(self.buffer_size - delay, 0)
        self.buffer_size += 2.002  # each chunk is 2 seconds of video

        # cap the buffer size and add noise
        if self.buffer_size > MAX_BUFFER:
            self.network_time_left = 0 # Move onto next time stamp
            if not self._test:
                noise = np.clip(1 + self.np_random.normal(0, 0.005), 0.98, 1.02)
            else:
                noise = 1
            self.buffer_size = MAX_BUFFER * noise

        if self.use_ssim:
            quality = self.trace.ssim_dbs[self.video_idx, action]
        else:
            quality= self.bitrate_map[action]

        # bitrate change penalty    
        if self.past_quality == 0:
            quality_change = 0
        else:
            quality_change = abs(self.past_quality - quality)

        quality_reward = self.reward_weights.quality_weight * quality
        rebuf_reward = self.reward_weights.rebuf_weight * rebuffer_time
        change_reward = self.reward_weights.quality_change_weight * quality_change
        reward =  quality_reward - rebuf_reward - change_reward

        previously_sent_quality = self.past_quality
        self.past_delay = delay
        self.past_quality = quality
        self.past_video_size = self.trace.video_sizes[self.video_idx, action]
        self.past_reward = reward
        self._sampler.record_action_reward(action=action, reward=reward,
            quality_reward=quality_reward, rebuf_reward=-rebuf_reward,
            change_reward=-change_reward)

        self.video_idx += 1
        done = (self.video_idx + 5 == self.total_video_chunks)

        time_received = time_sent + delay
        self.curr_time_sec = time_received + 1.00001e-8 # ensure next time sent > time of last chunk

        obs, info = self.observe(), \
               {'action': action,
                'buffer': self.buffer_size,
                'rebuf': rebuffer_time,
                'quality': self.trace.ssim_dbs[self.video_idx -1][action],
                'quality_change': quality_change,
                'past_quality': previously_sent_quality,
                'network_data': self.trace.network_data[self.curr_t_idx, 2:],
                'delivery_rate': self.trace.delivery_rates[self.video_idx -1],
                'video_sizes': self.trace.video_sizes[self.video_idx: self.video_idx + 5, :],
                'ssim_dbs': self.trace.ssim_dbs[self.video_idx: self.video_idx + 5, :],
                'n': self.total_video_chunks - (self.video_idx + 5),
                'curr_t_idx': self.curr_t_idx,
                'trace_idx': self.trace_idx,
                'time_sent': time_sent,
                'time_received': time_received,
                'TimeLimit.truncated': False}

        return obs, reward, done, False, info
    