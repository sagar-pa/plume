from gymnasium import spaces
import gymnasium as gym
import numpy as np
import csv
import ray
from pathlib import Path
from collections import deque
from typing import Any, Union, Tuple, List

from toy_abr_gym.abr import ToyABREnv
from toy_abr_gym.utils import make_reward_norm_func, normalize, format_float
from toy_abr_gym.trace_generator import BITRATES, MAX_BUFFER

class ToyABRWrapper(gym.Env):
    def __init__(self, env_kwargs: dict = None) -> None:

        """
        A Wrapper that implements Framestacking, reward normalization, and logging for ABR.
        
        Args:
            env_kwargs: a wrapper dictionary with the following arguments:
                abr_kwargs: The arguments to be passed to the wrapped ABR base environment
                history_len: The lenghth to stack the obervations 
                    for (does not stack video_sizes)
                max_trace_len: If trace is longer than this, 
                    ignore it and reset the environment (to ensure 1 trace does not dominate training)
                log_dir: The directory to log the test data into
                    (saved as log_dir/training_progress/traces_{}.csv)
                enable_logging: Whether or not to log the data at all
                shared_training_progress_getter: A callable that returns the current progress of the agent

        """
        abr_kwargs = env_kwargs.get("abr_kwargs", None) 
        history_len = env_kwargs.get("history_len", 10)
        max_trace_len = env_kwargs.get("max_trace_len", 1000)
        log_dir = env_kwargs.get("log_dir", "./logs") 
        enable_logging = env_kwargs.get("enable_logging", False)
        shared_testing_name = env_kwargs.get("shared_testing_data_name", None)
        if shared_testing_name is None:
            self.shared_testing_data = None
        else:
            self.shared_testing_data = ray.get_actor(shared_testing_name, namespace="test")

        self.log_dir = log_dir
        self.log = []
        self.enable_logging = enable_logging
        if abr_kwargs is None:
            raise ValueError(("abr_kwargs not provided. "
                        "Must be specified for sampling and trace generation."))

        self.trace_labels = abr_kwargs["sampler_kwargs"]["dataset"]["labels"]

        if shared_testing_name is not None:
            if "sampler_kwargs" not in abr_kwargs:
                abr_kwargs["sampler_kwargs"] = {}
            abr_kwargs["sampler_kwargs"].update(dict(shared_testing_data_name=\
                shared_testing_name))

        self.max_trace_len = max_trace_len
        self._env = ToyABREnv(**abr_kwargs)
        self.history_len = history_len
        self.obs_shape = (self.history_len*5 + 3, )
        self.observation_space = spaces.Box(np.zeros(shape = self.obs_shape, dtype=np.float32),
                                            np.ones(shape = self.obs_shape, dtype=np.float32),
                                            dtype=np.float32)
        self.action_space = spaces.Discrete(len(BITRATES))
        self.obs_history = deque(maxlen=history_len)
        self.next_sizes = np.zeros(shape=(len(BITRATES), ), dtype=np.float32)
        self.max_video_size = BITRATES[-1]
        self.max_quality = BITRATES[-1]
        self.setup_rewards()

    def setup_rewards(self) -> None:
        """
        Create a reward normalization function and 
            setup the normalization needed to scale obsevations to (0, 1)
        """
        self.reward_scale, self.reward_norm_func = make_reward_norm_func(
            max_reward=BITRATES[-1], min_reward=BITRATES[0])
        self.max_obs = np.array([self.max_quality, self.max_video_size, 20, 
            MAX_BUFFER, self.reward_scale[1] - self.reward_scale[0]]) #99.9 percentile values

    def change_log_dir(self, log_dir: Union[str, Path], 
            discard_logged_data: bool = True) -> None:
        """
        Change the logging directory of the environment.
        
        Args:
            log_dir: The directory to log to.
            discard_logged_data: whether or not to discard any left over data 
        """
        self.log_dir = log_dir
        if discard_logged_data:
            self.discard_log()

    def discard_log(self) -> None:
        """
        Discard the logged action data
        """
        self.log = []

    def setup_log(self, log_file: Union[str, Path], force: bool = False
            ) -> Path:
        """
        Setup the logging file to be saved in the log_dir with a header.

        Args:
            log_file: Path to the log file (includes parent directory information)
            force: Whether or not to override existing file if it exists
        Raises:
            ValueError if the file exists and force is set to False
        Returns:
            The Path to the log file, with the header written to.
        """
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
            label = self.trace_labels[idx]
            log_file = log_dir / f"{idx}_{label}.csv"
            columns = ["episode", "step", "action", "quality", "rebuf", 
                "reward", "buffer"]
            try:
                log_file = self.setup_log(log_file, force=False)
                with open(log_file, "w") as f:
                    writer = csv.writer(f)
                    writer.writerow(columns)
                    writer.writerows(self.log)
            except ValueError:
                pass
            self.discard_log()


    def set_env_attr(self, attribute: str, value: Any) -> None:
        """
        A wrapper function for setting any ABR env attribute.

        Args:
            atribute: The name of the attribute to set
            value: The value to set it to
        """
        setattr(self._env, attribute, value)

    def set_sampler_attr(self, attribute: str, value: Any) -> None:
        """
        A wrapper function to set any sampler attribute in the base ABR environment.

        Args:
            atribute: The name of the attribute to set
            value: The value to set it to
        """
        setattr(self._env._sampler, attribute, value)

    def observe(self) -> np.ndarray:
        """
        Return the normalized observation from the history currently recorded

        Returns:
            The numpy array in the observation space
        """
        output_arr = np.concatenate((np.array(self.obs_history).flatten(), self.next_sizes), axis=0)
        return output_arr

    def step(self, action: int) ->  Tuple[np.ndarray, float, bool, dict]:
        """
        A wrapper for the main step function in ABR. 
            Records the observations from the underlying environment, 
            normalizes it, and logs it if enable_logging is True.

        Args:
            action: The action (bitrate) to pass to the underlying environment
        Returns:
            The next observation
            The reward
            Whether or not the environment is done (needs to be reset)
            Extra information from the environment
        """
        obs, reward, done, truncated, info = self._env.step(action)
        network_data = obs[0:5]
        next_sizes = obs[5:8]
        self.next_sizes = normalize(next_sizes, max=self.max_video_size)
        reward = self.reward_norm_func(reward)
        network_data[4] = reward - self.reward_scale[0]
        done = done or self.episode_len >= self.max_trace_len

        self.obs_history.append(
            normalize(network_data, max=self.max_obs))
        self.episode_len += 1

        if self.enable_logging:
            row = [info["trace_idx"],
                self.episode_len, 
                info["action"], 
                info["quality"], 
                format_float(info["rebuf"]),
                format_float(reward, 6), 
                format_float(info["buffer"])]
            if len(self.log) > 1:
                if self.log[-1][0] != row[0]:
                    raise ValueError("Trace Changed while recording logs!")
            self.log.append(row)    
            if done:
                self.log_episode()

        return self.observe(), reward, done, truncated, info

    def reset(self, seed: int = None, options = None) -> np.ndarray:
        """
        A wrapper to reset the underlying environment. Sets the history to zeros. 
        """
        self.episode_len = 0
        self.discard_log()
        for i in range(self.history_len):
            self.obs_history.append(np.zeros(shape=(5)))
        obs, info = self._env.reset(seed=seed, options=options)
        network_data = obs[0:5]
        next_sizes = obs[5:8]
        self.next_sizes = normalize(next_sizes, max=self.max_video_size)
        self.obs_history.append(normalize(network_data, max=self.max_obs))
        return self.observe(), info

    def close(self) -> None:
        """
        Close the uderlying gym environment
        """
        self._env.close()
        



