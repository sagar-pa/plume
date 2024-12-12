import gymnasium as gym
from gymnasium import spaces
import numpy as np
from pathlib import Path
from collections import deque
from typing import Union, Tuple, List, Optional
import pandas as pd
import ray

from abr_gym.abr import ABRSimEnv
from abr_gym.trace_loader import ssim_db_to_index
from abr_gym.trace_downloader import CHUNK_LEN, VIDEO_FORMATS
from abr_gym.utils import (
        make_reward_norm_func,
        STALL_RATIO_THRESHOLD,
        SAFETY_VIOLATION_MULTIPLIER
)

class PensieveWrapper(gym.Env):
    def __init__(self, look_ahead_horizon = 4, 
            history_len = 10, max_trace_len: int = 10000, 
            reward_weights: Tuple[float, float, float] = (1,4.3,1), 
            reward_norm_style: str = "symmetric_sqrt_clip", 
            log_dir: Union[str, Path] = "./logs", 
            enable_logging: bool = False, 
            abr_kwargs: Optional[dict] = None,
            shared_testing_data_name: str = None) -> None:

        self.log_dir = log_dir
        self.log = []
        self.enable_logging = enable_logging
        if shared_testing_data_name is None:
            self.shared_testing_data = None
        else:
            self.shared_testing_data = ray.get_actor(shared_testing_data_name, "test")

        if self.shared_testing_data is not None:
            if "sampler_kwargs" not in abr_kwargs:
                abr_kwargs["sampler_kwargs"] = {}
            abr_kwargs["sampler_kwargs"].update(dict(shared_testing_data_name=\
                shared_testing_data_name))

        self.max_trace_len = max_trace_len
        self._env = ABRSimEnv(**abr_kwargs)
        self.history_len = history_len
        obs_len = 1 + 1 + history_len + history_len + \
            ((look_ahead_horizon +1) * 10) + 1 + ((look_ahead_horizon +1) * 10)
        self.observation_space = spaces.Box(np.zeros(shape = obs_len, dtype=np.float32),
                                            np.ones(shape = obs_len, dtype=np.float32),
                                            dtype=np.float32)
        self.action_space = spaces.Discrete(VIDEO_FORMATS)
        self.past_quality = 0
        self.past_buffer = 0
        self.throughput_history = deque(maxlen=history_len)
        self.delay_history = deque(maxlen=history_len)
        self.video_future = deque(maxlen=look_ahead_horizon + 1)
        self.quality_future = deque(maxlen=look_ahead_horizon + 1)
        self.max_video_size = 3
        self.max_quality = 25 if self._env.use_ssim else 8
        self.max_delay = 20
        self.max_throughput = 25
        self.max_buffer = 15
        self.max_n = -1
        self.n = -1
        self.use_n = False
        self.setup_rewards(reward_norm_style, reward_weights)

    def setup_rewards(self, reward_norm_style: str, 
            reward_weights: Tuple[float, float, float]) -> None:
        self.reward_scale, self.reward_norm_func = make_reward_norm_func(
            reward_norm_style, self.max_quality)
        self._env.setup_rewards(reward_weights)

    def change_log_dir(self, log_dir: Union[str, Path], 
            discard_logged_data: bool = True) -> None:
        self.log_dir = log_dir
        if discard_logged_data:
            self.discard_log()

    def discard_log(self) -> None:
        self.log = []

    def setup_log(self, log_file: Union[str, Path], force: bool = False
            ) -> Path:
        log_file = Path(log_file)
        log_file.parent.mkdir(exist_ok=True, parents=True)
        if log_file.exists() and not force:
            raise ValueError("File exists, and force is set to False.")
        return log_file

    def log_episode(self) -> None:
        if len(self.log) > 0:
            if self.shared_testing_data is None:
                progress = 100
            else:
                progress = ray.get(self.shared_testing_data.get_train_progress.remote())
            log_dir = Path(self.log_dir).resolve() / str(progress)
            idx = self.log[-1][0]
            log_file = log_dir / f"test_{idx}.feather"
            columns = ["episode", "step", "action", "ssim", "rebuf", 
                    "delivery_rate", "reward", "buffer", "cost", "time_sent", 
                    "time_received", "cwnd", "in_flight", 
                  "min_rtt", "rtt"] + \
                    [f"video_size_{i}" for i in range(VIDEO_FORMATS)] + \
                        [f"ssim_db_{i}" for i in range(VIDEO_FORMATS)]
            df = pd.DataFrame(data=self.log, columns=columns)
            written = False
            while not written:
                try:
                    log_file = self.setup_log(log_file, force=False)
                    df.to_feather(log_file)
                    pd.read_feather(log_file)
                    written = True
                except ValueError:
                    break
                except OSError:
                    pass
            self.discard_log()


    def set_env_attr(self, attribute, value) -> None:
        setattr(self._env, attribute, value)

    def set_sampler_attr(self, attribute, value) -> None:
        setattr(self._env._sampler, attribute, value)

    def normalize(self, raw: Union[int, np.ndarray], 
            max: Union[int, np.ndarray]) -> Union[int, np.ndarray]:
        norm = raw / max
        norm = np.clip(norm, 0, 1)
        return norm

    def observe(self) -> np.ndarray:
        past_quality = np.array([self.past_quality])
        past_buffer = np.array([self.past_buffer])
        past_throughputs = np.array(self.throughput_history)
        past_delays =np.array(self.delay_history)
        video_size_arr = np.array(self.video_future).flatten()
        n_arr = np.array([self.n])
        ssim_arr = np.array(self.quality_future).flatten() 

        return np.concatenate((past_quality, past_buffer, past_throughputs, 
            past_delays, video_size_arr, n_arr, ssim_arr))


    def step(self, action) -> Tuple[np.ndarray, float, bool, dict]:
        obs, reward, done, truncated, info = self._env.step(action)
        if self.use_n:
            if self.max_n  < 0:
                self.max_n = min(info["n"] + 1, self.max_trace_len + 1)
                self.n = self.max_n
            self.n -= 1

        network_data = obs[0:10]
        video_sizes = obs[10:60].reshape((5, 10))
        ssim_dbs = obs[60:110].reshape((5, 10))
        reward = self.reward_norm_func(reward)

        self.past_buffer = self.normalize(network_data[3], max=self.max_buffer)
        self.past_quality = self.normalize(network_data[0], max=self.max_quality)
        self.delay_history.append(
            self.normalize(network_data[2], max=self.max_delay))
        self.throughput_history.append(
            self.normalize(network_data[1] / network_data[2], 
            max=self.max_throughput))
        self.video_future.append(
            self.normalize(video_sizes[self.video_future.maxlen -1], 
            max=self.max_video_size))
        self.quality_future.append(
            self.normalize(ssim_dbs[self.video_future.maxlen -1],
            max=self.max_quality))
        self.episode_len += 1
        done = done or self.episode_len >= self.max_trace_len
        
        self.cum_rebuf += info["rebuf"]
        stall_ratio = self.cum_rebuf / (self.episode_len * CHUNK_LEN)
        if stall_ratio >= STALL_RATIO_THRESHOLD:
            cost = SAFETY_VIOLATION_MULTIPLIER * stall_ratio
        else:
            cost = 0
        info["cost"] = cost

        if self.enable_logging:
            row = [info["trace_idx"], 
                    self.episode_len, 
                    info["action"], 
                    ssim_db_to_index(info["quality"]),
                    info["rebuf"],
                    float(info["delivery_rate"]), 
                    reward, 
                    info["buffer"],
                    info["cost"],
                    info["time_sent"],
                    info["time_received"]
            ]
            row += info["network_data"][:-1].tolist()
            row += info["video_sizes"][0].tolist()
            row += info["ssim_dbs"][0].tolist()
            if len(self.log) > 1:
                if self.log[-1][0] != row[0]:
                    raise ValueError("Trace Changed while recording logs!")
            self.log.append(row)    
            if done:
                self.log_episode()

        return self.observe(), reward, done, truncated, info

    def reset(self, seed: int = None, options: dict = None) -> np.ndarray:
        self.episode_len = 0
        self.cum_rebuf = 0
        self.discard_log()
        self.past_buffer = 0
        self.past_quality = 0
        self.max_n = -1
        self.n = -1
        for i in range(self.history_len):
            self.delay_history.append(0)
            self.throughput_history.append(0)
        obs, info = self._env.reset(seed=seed, options=options)
        video_sizes = obs[10:60].reshape((5, 10))
        ssim_dbs = obs[60:110].reshape((5, 10))
        for i in range(self.video_future.maxlen):
            self.video_future.append(self.normalize(video_sizes[i], 
                max=self.max_video_size))
            self.quality_future.append(self.normalize(ssim_dbs[i],
                max=self.max_quality))
        self.use_n = not self._env._test and self._env.np_random.random() < 0.5
        return self.observe(), info

    def close(self) -> None:
        self._env.close()
        



