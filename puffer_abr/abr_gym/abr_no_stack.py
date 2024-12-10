from gymnasium import spaces
import gymnasium as gym
import numpy as np
from pathlib import Path
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

class ABRNoFramestack(gym.Env):
    def __init__(self, max_trace_len: int = 10000, 
            reward_weights : Tuple[float, float, float] = (1, 100, 1), 
            reward_norm_style: str = "shifted_sqrt_clip", 
            log_dir: Union[str, Path] = "./logs", 
            enable_logging: bool = False, 
            abr_kwargs: Optional[dict] = None,
            shared_testing_data_name: str = None) -> None:

        self.enable_logging = enable_logging
        self.log_dir = log_dir
        self.log = []
        if shared_testing_data_name is None:
            self.shared_testing_data = None
        else:
            self.shared_testing_data = ray.get_actor(shared_testing_data_name, "test")
        if abr_kwargs is None:
            abr_kwargs = {}

        if self.shared_testing_data is not None:
            if "sampler_kwargs" not in abr_kwargs:
                abr_kwargs["sampler_kwargs"] = {}
            abr_kwargs["sampler_kwargs"].update(dict(shared_testing_data_name=\
                shared_testing_data_name))

        self.max_trace_len = max_trace_len
        self._env = ABRSimEnv(**abr_kwargs)
        if not self._env.use_ssim:
            raise NotImplementedError("This wrapper is not designed to not use ssim")
        self.obs_shape = (27, )
        self.past_obs = np.zeros(shape=(7,), dtype=np.float32)
        self.video_sizes = np.zeros(shape=(10,), dtype=np.float32)
        self.qualities = np.zeros(shape=(10,), dtype=np.float32)
        self.observation_space = spaces.Box(np.zeros(shape = self.obs_shape, dtype=np.float32),
                                            np.ones(shape = self.obs_shape, dtype=np.float32),
                                            dtype=np.float32)
        self.action_space = spaces.Discrete(VIDEO_FORMATS)
        self.max_video_size = 3
        self.max_quality = 25
        self.min_quality = 6.5
        self.setup_rewards(reward_norm_style, reward_weights)

    def setup_rewards(self, reward_norm_style: str,
            reward_weights: Tuple[float, float, float]) -> None:
        self.reward_scale, self.reward_norm_func = make_reward_norm_func(
            reward_norm_style, self.max_quality, self.min_quality)
        self.max_obs = np.array([self.max_quality, self.max_video_size, 20, 
            15, self.reward_scale[1] - self.reward_scale[0]]) #99.9 percentile values
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

    def normalize(self, raw: np.ndarray, max: Union[int, np.ndarray]) -> np.ndarray:
        norm = raw / max
        norm = np.clip(norm, 0, 1)
        return norm

    def calculate_change_factor(self, values: np.ndarray) -> float:
        weights = 1. / np.arange(2, values.shape[0]+1)
        avg = np.average(values[1:], weights=weights, axis=0)
        change = (avg - values[0]) / values[0]
        change = np.clip(change, -1, 1)
        change = (change + 1) / 2 # [-1, 1] -> [0,1]
        return np.mean(change)

    def observe(self) -> np.ndarray:
        output_arr = np.zeros(shape=self.obs_shape)
        output_arr[0:7] = self.past_obs
        output_arr[7:17] = self.video_sizes
        output_arr[17:27] = self.qualities
        return output_arr

    def step(self, action) ->  Tuple[np.ndarray, float, bool, dict]:
        obs, reward, done, truncated, info = self._env.step(action)
        network_data = obs[0:5]
        video_sizes = obs[10:60].reshape((5, 10))
        ssim_dbs = obs[60:110].reshape((5, 10))
        reward = self.reward_norm_func(reward)
        network_data[4] = reward - self.reward_scale[0]
        done = done or self.episode_len >= self.max_trace_len

        change_factors = np.array([self.calculate_change_factor(video_sizes),
            self.calculate_change_factor(ssim_dbs)])

        self.past_obs = self.normalize(network_data, max=self.max_obs)
        self.past_obs = np.concatenate((self.past_obs, change_factors), axis=0)
        self.video_sizes = self.normalize(video_sizes[0], max=self.max_video_size)
        self.qualities = self.normalize(ssim_dbs[0], max=self.max_quality)
        self.episode_len += 1
        
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
        obs, info = self._env.reset(seed=seed, options=options)
        network_data = obs[0:5]
        video_sizes = obs[10:60].reshape((5, 10))
        ssim_dbs = obs[60:110].reshape((5, 10))
        reward = self.reward_norm_func(network_data[4])
        network_data[4] = reward - self.reward_scale[0]

        change_factors = np.array([self.calculate_change_factor(video_sizes),
            self.calculate_change_factor(ssim_dbs)])

        self.past_obs = self.normalize(network_data, max=self.max_obs)
        self.past_obs = np.concatenate((self.past_obs, change_factors), axis=0)
        self.video_sizes = self.normalize(video_sizes[0], max=self.max_video_size)
        self.qualities = self.normalize(ssim_dbs[0], max=self.max_quality)
        return self.observe(), info


    def close(self) -> None:
        self._env.close()
        



