from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import get_linear_fn
from typing import Union, List
from pathlib import Path
import numpy as np

class AnnealingCallBack(BaseCallback):
    def __init__(self, variable: str, start: float, end: float, 
            total_train_steps: Union[int, float], n_train_envs: int, 
            end_fraction: float = 0.5, update_freq: int = 20):
        super(AnnealingCallBack, self).__init__()
        self.variable = variable
        self.function = get_linear_fn(start, end, end_fraction)
        self.update_freq = update_freq
        self.total_calls = max(total_train_steps // n_train_envs, 1)

    def _on_step(self) -> bool:
        if self.n_calls % self.update_freq == 0:
            setattr(self.model, self.variable, 
                self.function(1 - self.n_calls / self.total_calls))
        return True


class CustomCheckpointCallback(BaseCallback):
    def __init__(self, save_freqs: List[int], save_dir: Path, 
            model_name: str, total_train_steps: Union[int, float],
            n_train_envs: int):
        super(CustomCheckpointCallback, self).__init__()
        self.save_dir = Path(save_dir)
        self.save_freqs = np.clip(save_freqs, 0, 99)
        self.model_name = model_name
        self.total_calls = max(total_train_steps // n_train_envs, 1)
        self.save_idx = 0

    def _on_step(self) -> bool:
        train_progress = self.n_calls / self.total_calls * 100
        if self.save_idx is not None and \
                train_progress >= self.save_freqs[self.save_idx]:
            self.model.save(self.save_dir / "{name}_{progress:03d}.zip".format(
                name=self.model_name, progress=self.save_freqs[self.save_idx]))
            self.save_idx += 1
            if self.save_idx >= len(self.save_freqs):
                self.save_idx = None
        return True