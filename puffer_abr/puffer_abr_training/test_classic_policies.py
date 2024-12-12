# %%
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from pathlib import Path
from typing import Union, List, TypedDict
import numpy as np

from abr_gym.abr_wrapper import ABRWrapper
from abr_gym.trace_loader import load_traces
import pickle as pk
from abr_gym.utils import SharedTestingData, terminate_shared_data
from puffer_abr_training.plot import make_puffer_metrics
from abc import ABC
from puffer_abr_training.classic_policies import BatchedAgent
from puffer_abr_training.global_constants import (
    ENV_KWARGS, TRACE_DIR, TEST_SPLIT, N_TEST_TRACES, MAX_TRACE_LEN, N_ENVS, N_EVAL_EPS, 
    LOG_DIR, SUMMARY_DIR, ABR_KWARGS
)
import ray

if not ray.is_initialized():
    ray.init(address="auto", log_to_driver=False)



# %%
def main(policy: str):
    env_kwargs = ENV_KWARGS
    model_name= policy
    test_traces = load_traces(trace_dir=TRACE_DIR, 
        criteria=ABR_KWARGS["selection_file"], split=TEST_SPLIT)
    n_traces = min(len(test_traces), N_TEST_TRACES)
    SharedTestingData.options(lifetime = "detached", name=model_name, namespace="test").remote(
        n_traces
    )
    env_kwargs["abr_kwargs"].update(dict(
        max_traces = n_traces, 
        split = TEST_SPLIT, 
        trace_dir = TRACE_DIR,
        selection_file = ABR_KWARGS["selection_file"]))
    env_kwargs.update(dict(
        max_trace_len = MAX_TRACE_LEN, 
        enable_logging=True, 
        shared_testing_data_name=model_name))



    log_dir = LOG_DIR / model_name
    log_dir.mkdir(parents=True, exist_ok=True)
    env_kwargs.update(dict(log_dir=log_dir))

    env_name = ABRWrapper
    test_env = make_vec_env(env_name, n_envs=N_ENVS, seed=23, env_kwargs=env_kwargs, 
        vec_env_cls=SubprocVecEnv, vec_env_kwargs=dict(start_method="forkserver"))

    train_progress = 100
    test_env.env_method("change_log_dir", log_dir)
    actor = ray.get_actor(name=model_name, namespace="test")
    actor.set_train_progress.remote(train_progress)
    model = BatchedAgent(policy_name=policy, n_envs=N_ENVS)
    evaluate_policy(model, test_env, 
        n_eval_episodes=N_EVAL_EPS)
    all_metric, slow_metric = make_puffer_metrics(log_dir / str(train_progress), 
        glob_str="test_*.feather", name=model_name,
        train_progress=train_progress)
    for file_ending, metric in [("all", all_metric), ("slow", slow_metric)]:
        summary_path = SUMMARY_DIR / f"{model_name}_{train_progress}_{file_ending}.pkl"
        with open(summary_path, "wb") as summary_pik:
            pk.dump(metric, summary_pik)

    test_env.close()

    terminate_shared_data([(model_name, "test")])



# %%
if __name__ == "__main__":
    from argparse import ArgumentParser
    
    parser = ArgumentParser()
    parser.add_argument("--policy", type=str, required=True, 
                        help=("Policy name to test, one of ['linear_bba',"
                              " 'bola_basic_v1', 'bola_basic_v2', 'mpc', 'random',"
                              " 'optimistic']"))
    args = parser.parse_args()
    main(args.policy)



