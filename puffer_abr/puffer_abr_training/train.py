from stable_baselines3.a2c import A2C
from torch import nn
import torch as th
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from pathlib import Path
from argparse import ArgumentParser
from typing import Tuple
from copy import deepcopy

from puffer_abr_training.network_heads import (SmallCNN, PensieveExtractorOriginal)
from puffer_abr_training.train_utils import AnnealingCallBack, CustomCheckpointCallback
from puffer_abr_training.test_controller import test_rl
from puffer_abr_training.global_constants import (
    ENV_KWARGS, TRACE_DIR, TRAIN_LOG_DIR, MAX_TRACE_K, MAX_CACHE_SIZE,
    TRAIN_STEPS, SEEDS, SAVE_FREQS, TEST_FREQS, N_ENVS, ABR_KWARGS
)

from abr_gym.abr_wrapper import ABRWrapper
from abr_gym.pensieve_wrapper import PensieveWrapper
from abr_gym.sampler_utils import SharedPtsData
from abr_gym.utils import Cache, terminate_shared_data
from abr_gym.trace_loader import load_trace_features, load_traces

import ray

if not ray.is_initialized():
    ray.init(address="auto", log_to_driver=False)



def run_gelato(seed: int, 
        sampling_func_cls: str, idx: int = 0, 
        plume_alpha: float = None,
        ) -> Tuple[str, Path, dict]:
    
    sampling_name = sampling_func_cls

    if sampling_func_cls == "plume_dynamic":
        if plume_alpha is None:
            raise ValueError("Cannot use plume_dynamic without specifying alpha for weight calculation")
        sampling_name = f"{sampling_name}_{plume_alpha:.2f}"
    if sampling_func_cls != "random":
        ABR_KWARGS.update(dict(max_trace_k = MAX_TRACE_K))    

    model_name = f"gelato_{sampling_name}_{idx}"
    checkpoint_dir = Path(TRAIN_LOG_DIR) / f"{model_name}_checkpoints"

    if sampling_func_cls == "plume_dynamic":
        sampling_log_dir = checkpoint_dir / "sampling"
        all_traces = load_traces(TRACE_DIR, ABR_KWARGS["selection_file"], "train")
        all_traces = all_traces[:ABR_KWARGS["max_traces"]]
        trace_features = load_trace_features(all_traces=all_traces, 
                trace_dir=TRACE_DIR,
                metrics_file=ABR_KWARGS["metrics_file"])
        
        SharedPtsData.options(
            lifetime="detached", name=model_name, namespace="plume_dynamic").remote(
                alpha = plume_alpha, 
                trace_features = trace_features.features,
                trace_dists = trace_features.cluster_dists, 
                log_dir = sampling_log_dir
        )
        sampler_kwargs = dict(
            shared_pts_data_name = model_name,
        )
        ABR_KWARGS["sampler_kwargs"] = sampler_kwargs


    if "sampler_kwargs" not in ABR_KWARGS:
        ABR_KWARGS["sampler_kwargs"] = {}
    Cache.options(lifetime="detached", name=model_name, namespace="cache").remote(max_size = MAX_CACHE_SIZE)
    ABR_KWARGS["sampler_kwargs"].update(dict(
        trace_cache_name=model_name))

    ABR_KWARGS.update(dict(sampling_func_cls=sampling_func_cls))
    th.manual_seed(seed)
    
    train_env = make_vec_env(ABRWrapper, n_envs=N_ENVS, seed=seed, 
        env_kwargs=ENV_KWARGS, vec_env_cls=SubprocVecEnv, 
        vec_env_kwargs=dict(start_method="forkserver"))

    policy_kwargs = dict(net_arch=dict(pi=[256], vf=[256]),
                            activation_fn=nn.ReLU,
                            features_extractor_class=SmallCNN )
    ent_callback = AnnealingCallBack("ent_coef", start=5.75, end=.00025, 
        total_train_steps=TRAIN_STEPS, n_train_envs=N_ENVS)
    save_freqs = SAVE_FREQS
    checkpoint_callback = CustomCheckpointCallback(save_freqs=save_freqs, 
        save_dir=checkpoint_dir, 
        model_name=model_name, total_train_steps=TRAIN_STEPS, n_train_envs=N_ENVS)
    model =  A2C("CnnPolicy", env=train_env, n_steps=15, gamma=0.95, 
        gae_lambda=1.0, ent_coef=5.75, vf_coef=0.90, learning_rate=.0001, 
        max_grad_norm=0.4, policy_kwargs=policy_kwargs, verbose=1, seed=seed)
    model.learn(TRAIN_STEPS, callback=[ent_callback, checkpoint_callback])
    model.save(TRAIN_LOG_DIR / model_name)
    model.save(checkpoint_dir / f"{model_name}_100.zip")

    #clean up
    del ABR_KWARGS["sampler_kwargs"]
    train_env.close()
    actors_to_terminate = [(model_name, "cache")]
    if sampling_func_cls == "plume_dynamic":
        actors_to_terminate.append((model_name, "plume_dynamic"))
    terminate_shared_data(actors_to_terminate)

    return (model_name, checkpoint_dir, deepcopy(ENV_KWARGS))

def run_pensieve(seed: int, 
        sampling_func_cls: str, idx: int = 0, 
        plume_alpha: float = None,
        ) -> Tuple[str, Path, dict]:

    sampling_name = sampling_func_cls
    if sampling_func_cls == "plume_dynamic" and plume_alpha is not None:
        sampling_name = f"{sampling_name}_{plume_alpha:.2f}"
    if sampling_func_cls != "random":
        ABR_KWARGS.update(dict(max_trace_k = MAX_TRACE_K))    

    
    ABR_KWARGS.update(dict(use_ssim=False, reward_weights= (1,4.3,1)))
    ENV_KWARGS.update(dict(look_ahead_horizon=0))

    model_name = f"pensieve_{sampling_name}_{idx}"
    checkpoint_dir = Path(TRAIN_LOG_DIR) / f"{model_name}_checkpoints"

    if sampling_func_cls == "plume_dynamic":
        sampling_log_dir = checkpoint_dir / "sampling"
        all_traces = load_traces(TRACE_DIR, ABR_KWARGS["selection_file"], "train")
        all_traces = all_traces[:ABR_KWARGS["max_traces"]]
        trace_features = load_trace_features(all_traces=all_traces, 
                trace_dir=TRACE_DIR,
                metrics_file=ABR_KWARGS["metrics_file"])
        
        SharedPtsData.options(
            lifetime="detached", name=model_name, namespace="plume_dynamic").remote(
                alpha = plume_alpha, 
                trace_features = trace_features.features,
                trace_dists = trace_features.cluster_dists, 
                log_dir = sampling_log_dir
        )
        sampler_kwargs = dict(
            shared_pts_data_name = model_name,
        )
        ABR_KWARGS["sampler_kwargs"] = sampler_kwargs


    if "sampler_kwargs" not in ABR_KWARGS:
        ABR_KWARGS["sampler_kwargs"] = {}
    Cache.options(lifetime="detached", name=model_name, namespace="cache").remote(max_size = MAX_CACHE_SIZE)
    ABR_KWARGS["sampler_kwargs"].update(dict(
        trace_cache_name=model_name))

    ABR_KWARGS.update(dict(sampling_func_cls=sampling_func_cls))
    th.manual_seed(seed)

    train_env = make_vec_env(PensieveWrapper, n_envs=N_ENVS, seed=seed, 
        env_kwargs=ENV_KWARGS, vec_env_cls=SubprocVecEnv, 
        vec_env_kwargs=dict(start_method="forkserver"))

    ent_callback = AnnealingCallBack("ent_coef", start=5.75, end=.00025, 
        total_train_steps=TRAIN_STEPS, n_train_envs=N_ENVS)
    save_freqs = SAVE_FREQS
    checkpoint_callback = CustomCheckpointCallback(save_freqs=save_freqs, 
        save_dir=checkpoint_dir, 
        model_name=model_name, total_train_steps=TRAIN_STEPS, n_train_envs=N_ENVS)
    policy_kwargs = dict(net_arch=dict(pi=[128], vf=[128]),
                            activation_fn=nn.ReLU,
                            features_extractor_class=PensieveExtractorOriginal)
    model =  A2C("CnnPolicy", env=train_env, n_steps=15, gamma=0.95, 
        gae_lambda=1.0, ent_coef=5.75, vf_coef=0.90, learning_rate=.0001, 
        max_grad_norm=0.4, policy_kwargs=policy_kwargs, verbose=1, seed=seed)
    model.learn(TRAIN_STEPS, callback=[ent_callback, checkpoint_callback])
    model.save(TRAIN_LOG_DIR / model_name)
    model.save(checkpoint_dir / f"{model_name}_100.zip")

    #clean up
    del ABR_KWARGS["sampler_kwargs"]
    train_env.close()
    actors_to_terminate = [(model_name, "cache")]
    if sampling_func_cls == "plume_dynamic":
        actors_to_terminate.append((model_name, "plume_dynamic"))
    terminate_shared_data(actors_to_terminate)

    return (model_name, checkpoint_dir, deepcopy(ENV_KWARGS))


def main(args):
    test_args = []
    idx = args.idx
    seed = SEEDS[idx]
    if args.kind == "pensieve": 
        model_name, checkpont_dir, env_kwargs = run_pensieve(seed, args.sampling_func, idx, 
            args.plume_alpha)
    elif args.kind == "gelato":
        model_name, checkpont_dir, env_kwargs = run_gelato(seed, args.sampling_func, idx, 
            args.plume_alpha)
    else:
        raise ValueError("kind of controller unknown")
    test_args.append((model_name, checkpont_dir, env_kwargs))
    for model_name, checkpont_dir, env_kwargs in test_args:
        ignore_train_progress = set(SAVE_FREQS) - set(TEST_FREQS)
        test_rl(model_path=sorted(checkpont_dir.iterdir(), reverse=True), 
            model_name=model_name, test_env_kwargs=env_kwargs,
            ignore_train_progress=ignore_train_progress)

    

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--kind",  help="One of 'pensieve' or 'gelato'",
                        type=str, required=True)
    parser.add_argument("--sampling_func", type=str, required=False, 
        default="random", help="One of 'random', 'plume_static', \
            'plume_dynamic'.")
    parser.add_argument("--plume_alpha", type=float, required=False, default=0.5, 
        help="The dynamic sampling weight alpha to use in plume_dynamic (0,1)")
    parser.add_argument("--idx", type=int, required=False, default=0)
    args = parser.parse_args()
    main(args)
