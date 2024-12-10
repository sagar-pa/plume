import numpy as np
from gymnasium import Env
from gymnasium import spaces
from gymnasium.utils import seeding
from pathlib import Path
from typing import Tuple, List, NamedTuple

from toy_abr_gym.sampler import Sampler
from toy_abr_gym.trace_generator import MAX_BUFFER, BITRATES

DIR_PATH = Path(__file__)

class RewardWeights(NamedTuple):
    quality_weight: float = 1.
    rebuf_weight: float = 6.

class ToyABREnv(Env):
    def __init__(self, seed=None,
            test: bool = False, 
            sampler_kwargs: dict = None,
            reward_weights: RewardWeights = None) -> None:
        """
        Create an ABR environment. In this environment, 
            the task to maximize he QoE of the viewer in a given Trace, 
            where the QoE is measured as Quality - Rebuffering.
        
        Args:
            seed: The seed to use (the random module is used for sampling, 
                adding noise to delay calculations and buffer sizes)
            test: Whether or not to disable the weighted sampling and delay noise
            sampler_kwargs: The arguments to feed to the sampler
            reward_weights: The weights to use for the environment. By default, use the ones above
        """
        super().reset(seed=seed)
        self._test = test
        self.trace_idx = None
        if sampler_kwargs is None:
            sampler_kwargs = {}
        self._sampler = Sampler(self.np_random, **sampler_kwargs)
        if self._test:
            self.sampling_function = self._sampler.iterative_sample
        else:
            self.sampling_function = self._sampler.weighted_sample
        # set up seed
        # observation and action space
        self.bitrate_map = BITRATES
        self.setup_space()
        self.setup_rewards(reward_weights)

    def setup_rewards(self, reward_weights: Tuple[float, float] = None) -> None:
        """
        Set the Reward Weights used for ABR.

        Args:
            reward_weights: in absolute value [Quality Weight, Rebuf Weight]
        """
        if reward_weights is None:
            self.reward_weights = RewardWeights()
        else:
            self.reward_weights = RewardWeights(*reward_weights)

    def setup_space(self) -> None:
        """
        Setup Gym Spaces to finish registering the environment.
        """
        self.obs_low = np.array([0.] * 8, dtype=np.float32)
        self.obs_high = np.array([100.] * 8, dtype=np.float32)
        self.observation_space = spaces.Box(
            low=self.obs_low, high=self.obs_high)
        self.action_space = spaces.Discrete(len(self.bitrate_map))


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
    
    def get_chunk_time(self, t_idx: int) -> float:
        """
        Calculate how long the current t_idx will last according to the network data.
        The time is calculated according to when the next throughput is logged to set.

        Args:
            network_data: The network data array of throughputs and times loaded by load_trace
            t_idx: The index to calculate value for
        Returns:
            The time that current throughput lasts for.

        """
        network_data = self.trace.network_data
        if t_idx == network_data.shape[0] - 1:
            return 1  # bandwidth last for 1 second
        else:
            time = network_data[t_idx + 1, 0] - network_data[t_idx, 0]
            return time

    def observe(self) -> np.ndarray:
        """
        Create and return an observatio. The obervation is an array of the client
        data, with the next video sizes concatenated at the end.
        """        
        buffer_size =  self.buffer_size if self.buffer_size is not None else 0.

        obs_arr = np.array([self.past_quality,
                    self.past_video_size,
                    self.past_delay,
                    buffer_size,
                    self.past_reward])
        # next 5 video sizes and ssim dbs 
        video_sizes = self.trace.video_sizes[self.video_idx,:]
        return np.concatenate((obs_arr, video_sizes))

    def reset(self, seed: int = None, options = None) -> np.ndarray:
        """
        Reset the environment. Sample a new trace and starting index if needed,
        and set the buffer and elements for reward calculations.
        """
        super().reset(seed=seed)
        self._sampler.np_random = self.np_random
            
        if self.trace_idx is not None:
            self._sampler.record_episode(idx = self.trace_idx)
        self.trace, self.start_t_idx, self.trace_idx  = self.sampling_function()
        self.curr_t_idx = self.start_t_idx
        self.network_time_left = self.get_chunk_time(self.curr_t_idx)
        self.video_idx = self.curr_t_idx
        self.total_video_chunks = self.trace.video_sizes.shape[0]
        self.total_network_ts = self.trace.network_data.shape[0]
        self.buffer_size = None  # initial download time not counted
        self.past_quality = 0.
        self.past_delay = 0.
        self.past_video_size = 0.
        self.past_reward = 0.
        return self.observe(), {"trace_idx": self.trace_idx,
                                "start_t_idx": self.start_t_idx}

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

        while video_size > 1e-8:  # floating number business
            if self.network_time_left <= 0:
                self.curr_t_idx = self.circular_increment(
                    self.curr_t_idx, self.total_network_ts)
                self.network_time_left = self.get_chunk_time(self.curr_t_idx)
            
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
            rebuffer_time = np.clip(delay - self.buffer_size, 0, 100)

        # update buffer
        self.buffer_size = max(self.buffer_size - delay, 0)
        self.buffer_size += 2.0  # each chunk is 2 seconds of video

        # cap the buffer size and add noise
        if self.buffer_size > MAX_BUFFER:
            self.network_time_left -= self.buffer_size - MAX_BUFFER
            if not self._test:
                noise = np.clip(1 + self.np_random.normal(0, 0.005), 0.98, 1.02)
            else:
                noise = 1
            self.buffer_size = MAX_BUFFER * noise

        quality= self.bitrate_map[action]

        quality_reward = self.reward_weights.quality_weight * quality
        rebuf_reward = self.reward_weights.rebuf_weight * rebuffer_time
        reward =  quality_reward - rebuf_reward

        self.past_delay = delay
        self.past_quality = quality
        self.past_video_size = self.trace.video_sizes[self.video_idx, action]
        self.past_reward = reward
        self._sampler.record_action_reward(action=action, reward=reward,
            quality_reward=quality_reward, rebuf_reward=-rebuf_reward)

        self.video_idx += 1
        done = (self.video_idx + 1 == self.total_video_chunks)

        obs, info = self.observe(), \
               {'action': action,
                'buffer': self.buffer_size,
                'rebuf': rebuffer_time,
                'quality': self.bitrate_map[action],
                'video_sizes': self.trace.video_sizes[self.video_idx],
                'curr_t_idx': self.curr_t_idx,
                'trace_idx': self.trace_idx}

        return obs, reward, done, False, info
    