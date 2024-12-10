import ray
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from pathlib import Path
from collections import deque
from typing import Any, Union, Tuple, List, Optional
import pandas as pd

from abr_gym.abr import ABRSimEnv
from abr_gym.trace_loader import ssim_db_to_index
from abr_gym.trace_downloader import CHUNK_LEN, VIDEO_FORMATS
from abr_gym.utils import (
        make_reward_norm_func, 
        LongtermFeatureExtractor,
        STALL_RATIO_THRESHOLD,
        SAFETY_VIOLATION_MULTIPLIER
)

class ABRWrapper(gym.Env):
    def __init__(self, history_len = 10, max_trace_len: int = 10000, 
            reward_weights : Tuple[float, float, float] = (1, 100, 1), 
            reward_norm_style: str = "symmetric_sqrt_clip", 
            log_dir: Union[str, Path] = "./logs", 
            enable_logging: bool = False, abr_kwargs: Optional[dict] = None, 
            longterm_extractor_kwargs: Optional[dict] = None,
            shared_testing_data_name: str = None) -> None:

        self.log_dir = log_dir
        self.log = []
        self.enable_logging = enable_logging
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
        self.history_len = history_len
        if longterm_extractor_kwargs is None:
            longterm_extractor_kwargs = dict(
                vars=["rebuf"], max_vals=[3], output_dim=history_len, 
                history_len=history_len * 3, function=sum)
        self.longterm_extractor = LongtermFeatureExtractor(**longterm_extractor_kwargs)
        self.obs_shape = (self.history_len, 25 + self.longterm_extractor.n_vars)
        self.observation_space = spaces.Box(np.zeros(shape = self.obs_shape, dtype=np.float32),
                                            np.ones(shape = self.obs_shape, dtype=np.float32),
                                            dtype=np.float32)
        self.action_space = spaces.Discrete(VIDEO_FORMATS)
        self.obs_history = deque(maxlen=history_len)
        self.video_history = deque(maxlen=5)
        self.quality_history = deque(maxlen=5)
        self.max_video_size = 3
        self.max_quality = 25
        self.setup_rewards(reward_norm_style, reward_weights)

    def setup_rewards(self, reward_norm_style: str,
            reward_weights: Tuple[float, float, float]) -> None:
        self.reward_scale, self.reward_norm_func = make_reward_norm_func(
            reward_norm_style, self.max_quality)
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


    def set_env_attr(self, attribute: str, value: Any) -> None:
        setattr(self._env, attribute, value)

    def set_sampler_attr(self, attribute: str, value: Any) -> None:
        setattr(self._env._sampler, attribute, value)

    def normalize(self, raw: np.ndarray, max: Union[int, np.ndarray]) -> np.ndarray:
        norm = raw / max
        norm = np.clip(norm, 0, 1)
        return norm

    def observe(self) -> np.ndarray:
        output_arr = np.zeros(shape=self.obs_shape)
        network_arr = np.array(self.obs_history)
        video_size_arr = np.array(self.video_history)
        ssim_arr = np.array(self.quality_history)
        output_arr[0:self.history_len, 
            0:5] = network_arr
        output_arr[0:self.history_len, 
            5:5+self.longterm_extractor.n_vars] = self.longterm_extractor.observe()
        output_arr[0:5, 
            5+self.longterm_extractor.n_vars:15+self.longterm_extractor.n_vars] = video_size_arr
        output_arr[0:5, 
            15+self.longterm_extractor.n_vars:25+self.longterm_extractor.n_vars] = ssim_arr
        return output_arr

    def step(self, action: int) ->  Tuple[np.ndarray, float, bool, dict]:
        obs, reward, done, truncated, info = self._env.step(action)
        network_data = obs[0:5]
        video_sizes = obs[10:60].reshape((5, 10))
        ssim_dbs = obs[60:110].reshape((5, 10))
        reward = self.reward_norm_func(reward)
        network_data[4] = reward - self.reward_scale[0]
        done = done or self.episode_len >= self.max_trace_len

        self.obs_history.append(
            self.normalize(network_data, max=self.max_obs))
        self.video_history.append(
            self.normalize(video_sizes[-1], max=self.max_video_size))
        self.quality_history.append(
            self.normalize(ssim_dbs[-1], max=self.max_quality))
        self.longterm_extractor.record(info)
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

        for var, history in zip(self.longterm_extractor.vars, self.longterm_extractor.history):
            info[f"longterm_{var}_history"] = list(history)

        return self.observe(), reward, done, truncated, info

    def reset(self, seed: int = None, options: dict = None) -> np.ndarray:
        self.episode_len = 0
        self.cum_rebuf = 0
        self.discard_log()
        self.longterm_extractor.reset()
        for i in range(self.history_len):
            self.obs_history.append(np.zeros(shape=(5)))
        obs, info = self._env.reset(seed=seed, options=options)
        network_data = obs[0:5]
        video_sizes = obs[10:60].reshape((5, 10))
        ssim_dbs = obs[60:110].reshape((5, 10))
        for i in range(5):
            self.video_history.append(
                self.normalize(video_sizes[i], max=self.max_video_size))
            self.quality_history.append(
                self.normalize(ssim_dbs[i], max=self.max_quality))
        self.obs_history.append(self.normalize(network_data, max=self.max_obs))
        return self.observe(), info

    def render(self, mode='human', close=False) -> None:
        pass

    def close(self) -> None:
        self._env.close()
        



