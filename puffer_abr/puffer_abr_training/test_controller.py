from stable_baselines3.a2c import A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from pathlib import Path
from typing import Union, List
import numpy as np

from abr_gym.abr_wrapper import ABRWrapper
from abr_gym.pensieve_wrapper import PensieveWrapper
from abr_gym.trace_loader import load_traces
import pickle as pk
from abr_gym.utils import SharedTestingData, terminate_shared_data
from puffer_abr_training.plot import make_puffer_metrics
from puffer_abr_training.global_constants import (
    TEST_ENV_KWARGS, TRACE_DIR, TEST_SPLIT, N_TEST_TRACES, MAX_TRACE_LEN, N_ENVS, N_EVAL_EPS,
    LOG_DIR, SUMMARY_DIR, TEST_ABR_KWARGS
)

import ray

if not ray.is_initialized():
    ray.init(address="auto", log_to_driver=False)

TEST_ENV = None

def close_test_env():
    global TEST_ENV
    if TEST_ENV is not None:
        TEST_ENV.close()
        TEST_ENV = None

def test_rl(model_path: Union[str, Path, List[str], List[Path]], 
        model_name: str, test_env_kwargs: dict=None,
        ignore_train_progress: set = None):

    global TEST_ENV, TEST_ENV_KWARGS
    if TEST_ENV is not None:
        close_test_env()

    if test_env_kwargs is None:
        test_env_kwargs = TEST_ENV_KWARGS
    test_traces = load_traces(trace_dir=TRACE_DIR, 
        criteria=TEST_ABR_KWARGS["selection_file"], split=TEST_SPLIT)
    n_traces = min(len(test_traces), N_TEST_TRACES)
    SharedTestingData.options(lifetime = "detached", name=model_name, namespace="test").remote(
        n_traces
    )
    test_env_kwargs["abr_kwargs"].update(dict(
        max_traces = n_traces, 
        split = TEST_SPLIT, 
        trace_dir = TRACE_DIR,
        selection_file = TEST_ABR_KWARGS["selection_file"]))
    test_env_kwargs.update(dict(
        max_trace_len = MAX_TRACE_LEN, 
        enable_logging=True, 
        shared_testing_data_name=model_name))

    if isinstance(model_path, list):
        model_path = [Path(model) for model in model_path if Path(model).is_file()]
    else:
        model_path = [Path(model_path)]

  
    log_dir = LOG_DIR / model_name
    log_dir.mkdir(parents=True, exist_ok=True)
    test_env_kwargs.update(dict(log_dir=log_dir))

    env_name = PensieveWrapper if "pensieve" in model_name else ABRWrapper
    TEST_ENV = make_vec_env(env_name, n_envs=N_ENVS, seed=23, env_kwargs=test_env_kwargs, 
        vec_env_cls=SubprocVecEnv, vec_env_kwargs=dict(start_method="forkserver"))

    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    for model_to_load in model_path:
        try:
            *_, train_progress = \
                str(model_to_load.stem).split("_")
            log_path = log_dir
            train_progress = int(train_progress)
        except (ValueError, IndexError, TypeError):
            train_progress = 100
            log_path = log_dir / str(model_to_load.stem)
        if ignore_train_progress is not None and \
                train_progress in ignore_train_progress:
            continue
        TEST_ENV.env_method("change_log_dir", log_path)
        actor = ray.get_actor(name=model_name, namespace="test")
        actor.set_train_progress.remote(train_progress)
        model = A2C.load(model_to_load)
        evaluate_policy(model, TEST_ENV, 
            n_eval_episodes=N_EVAL_EPS)
        all_metric, slow_metric = make_puffer_metrics(log_path / str(train_progress), 
            glob_str="test_*.feather", name=model_name,
            train_progress=train_progress)
        for file_ending, metric in [("all", all_metric), ("slow", slow_metric)]:
            summary_path = SUMMARY_DIR / f"{model_name}_{train_progress}_{file_ending}.pkl"
            with open(summary_path, "wb") as summary_pik:
                pk.dump(metric, summary_pik)
    
    close_test_env()

    terminate_shared_data([(model_name, "test")])
