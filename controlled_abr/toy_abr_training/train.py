from argparse import ArgumentParser, Namespace
from toy_abr_gym.trace_generator import generate_dataset
from toy_abr_gym.trace_loader import load_trace_features
from toy_abr_gym.abr_wrapper import ToyABRWrapper
from toy_abr_gym.utils import SharedTestingData, terminate_shared_data
from toy_abr_gym.sampler_utils import SharedPtsData
from ray.rllib.algorithms.apex_dqn import ApexDQNConfig
from ray.rllib.algorithms.apex_dqn import ApexDQN
from pathlib import Path
import ray
from ray.util.queue import Queue
from tqdm.auto import tqdm
from ray.tune.registry import register_env
import numpy as np
from global_constants import (
    NUM_ENVS, N_STEPS, LOG_DIR, SAMPLING_LOG_DIR, GAMMA, N_STEP_RETURN,
    LEARNING_RATE, CHECKPOINT_FREQS, STEPS_PER_ITER, SEEDS, NUM_DATASETS, N_EVAL_EPS
)
from ray.tune.logger import Logger
import shutil


class NullLogger(Logger):

    def __init__(self, config: dict, logdir: str, trial = None):
        super(NullLogger, self).__init__(config, logdir, trial)

    def on_result(self, result):
        pass

def null_logger_creator(config):
    return NullLogger(config, Path("."))



def print_metrics(metrics: dict, kind: str = "training", is_eval: bool = False) -> None:
    """
    Print the metrics, outputting only thr predefined keys.
    Args:
        metrics: The metrics returned by trainer to print
    """

    tqdm.write("-" * 10 + f"{kind} metrics" + "-" * 10)
    if not is_eval:
        for info_key, keys in [["info", ["num_env_steps_sampled", "num_agent_steps_trained"]],
                                ["sampler_results", ["episode_reward_mean"]],
                            ]:
            all_info = [f'{key}: {metrics[info_key][key]}' for key in keys]
            all_info = '\t'.join(all_info)
            tqdm.write(f"{all_info}", end="\n")
        # nested info
        all_info = []
        #all_info.append(f"mean_td_error: {metrics['info']['learner']['default_policy']['mean_td_error']}")
        all_info.append(f"cur_epsilon: {metrics['info']['exploration_infos']['default_policy']['cur_epsilon']}")
        all_info = '\t'.join(all_info)
        tqdm.write(f"{all_info}", end="\n")
    else:
        tqdm.write(f"episode_reward_mean: {metrics['evaluation']['episode_reward_mean']}")
        tqdm.write(f"episode_evaluated: {metrics['evaluation']['episodes_this_iter']}")
        tqdm.write(f"episode_len_mean: {metrics['evaluation']['episode_len_mean']}")
    tqdm.write("", end="\n")


def main(args: Namespace) -> None:
    """
    Main function to run the training process. 
    Args:
        args: A namespace with the followring attribures defined for:
            per: bool: Whether or not to use PER
            curiosity: bool: Whether or not to use Curiosity
            sampling_func: str: The name of the sampling function to use
            idx: int: The index (an indentifier of the agent in multiple agents)
            dataset: str: The kind of dataset to train and test on

    """
    if not ray.is_initialized():
        ray.init(log_to_driver = False)

    per_name = "per" if args.per else "-"
    curiosity_name = "curiosity" if args.curiosity else "-"
    sampling_name = args.sampling_func
    if args.sampling_func == "plume_dynamic":
        if args.plume_alpha is None:
            raise ValueError("Cannot use plume_dynamic without specifying sampling alpha weight")
        sampling_name = f"{sampling_name}_{args.plume_alpha:.2f}"
    name = f"{args.dataset}__{sampling_name}__{per_name}__{curiosity_name}__{args.idx}"

    train_dataset = generate_dataset(args.dataset, "train")
    train_trace_features = load_trace_features(args.dataset)
    train_env_kwargs = dict(
        abr_kwargs = dict(test = False, sampler_kwargs = dict(
            dataset = train_dataset,
            sampling_func_cls = args.sampling_func,
            trace_features = train_trace_features
        )))
    if args.sampling_func == "plume_dynamic":
        sampling_log_dir = SAMPLING_LOG_DIR / name /  "sampling"
        pts_unique_name = name
        SharedPtsData.options(lifetime="detached", name=pts_unique_name, namespace="plume_dynamic").remote(
                alpha = args.plume_alpha, 
                trace_features = train_trace_features.features,
                trace_dists = train_trace_features.cluster_dists, 
                log_dir = sampling_log_dir
        )
        train_env_kwargs["abr_kwargs"]["sampler_kwargs"]["shared_pts_data_name"] = pts_unique_name
    train_env_maker_kwargs = dict(env_kwargs = train_env_kwargs, test = False)

    test_env_kwargs_queue = Queue(actor_options=dict(num_cpus=0.01))
    shared_testing_data_names = []
    for dataset_name in ["majority_fast", "balanced", "majority_slow"]:
        test_dataset = generate_dataset(dataset_name=dataset_name, split="test")
        shared_testing_data_unique_name = f"{name}__{dataset_name}"
        SharedTestingData.options(lifetime="detached", 
            name=shared_testing_data_unique_name, namespace="test").remote(
                len(test_dataset["traces"])
        )
        shared_testing_data_names.append(shared_testing_data_unique_name)
        log_dir = LOG_DIR / f"{name}__{dataset_name}"
        test_env_kwargs = dict(
            abr_kwargs = dict(test = True, sampler_kwargs = dict(
                sampling_func_cls = "iterative", dataset=test_dataset,
            )),
            log_dir = log_dir,
            enable_logging = True,
            shared_testing_data_name = shared_testing_data_unique_name
        )
        for __ in range(NUM_ENVS):
            test_env_kwargs_queue.put(test_env_kwargs)

    def env_maker(maker_kwargs: dict) -> ToyABRWrapper:
        test = maker_kwargs["test"]
        if test:
            env_kwargs = test_env_kwargs_queue.get_nowait()
            env = ToyABRWrapper(env_kwargs = env_kwargs)
        else:
            env_kwargs = maker_kwargs["env_kwargs"] 
            env = ToyABRWrapper(env_kwargs = env_kwargs)
        return env
    register_env("toy_abr_maker", env_maker)

    test_env_maker_kwargs = dict(test = True)
    
    config = ApexDQNConfig().to_dict()
    replay_buffer_config = {
        "type": "MultiAgentPrioritizedReplayBuffer",
        "capacity": 250000, #adjust
        "replay_sequence_length": 1, 
        "learning_starts": 50000,
        "worker_side_prioritization": args.per,
        "prioritized_replay_alpha": 0.6 if args.per else 0.0
    }
    if args.curiosity:
        exploration_config = { #TODO: modify curiosity parameters
            "type": "Curiosity",  # <- Use the Curiosity module for exploring.
                "eta": 1.0,  # Weight for intrinsic rewards before being added to extrinsic ones.
                "lr": 0.0001,  # Learning rate of the curiosity (ICM) module.
                "feature_dim": 288,  # Dimensionality of the generated feature vectors.
                # Setup of the feature net (used to encode observations into feature (latent) vectors).
                "feature_net_config": {
                    "fcnet_hiddens": [],
                    "fcnet_activation": "relu",
                },
                "inverse_net_hiddens": [256],  # Hidden layers of the "inverse" model.
                "inverse_net_activation": "relu",  # Activation of the "inverse" model.
                "forward_net_hiddens": [256],  # Hidden layers of the "forward" model.
                "forward_net_activation": "relu",  # Activation of the "forward" model.
                "beta": 0.2,  # Weight for the "forward" loss (beta) over the "inverse" loss (1.0 - beta).
                # Specify, which exploration sub-type to use (usually, the algo's "default"
                # exploration, e.g. EpsilonGreedy for DQN, StochasticSampling for PG/SAC).
                "sub_exploration": {
                    "type": "StochasticSampling",
                }
        }
    else:
        exploration_config = {
            "type": "PerWorkerEpsilonGreedy",
            "initial_epsilon": 1.0, 
            "final_epsilon": 0.02, #adjust 
            "epsilon_timesteps": 1e5, #adjust #TODO: Epsilon is incorrectly reported? 
            #https://discuss.ray.io/t/how-to-get-the-current-epsilon-value-after-a-training-iteration/6910
            "warmup_timesteps": replay_buffer_config["learning_starts"]
        }
    model_config = {
        "fcnet_hiddens": [256, 256], 
        "fcnet_activation": "elu", 
        "framestack": False,
        "zero_mean": False,
    }
    eval_config = {
        "explore": False,
        "env_config": test_env_maker_kwargs,
        "num_envs_per_worker": NUM_DATASETS,
    }
    config.update(
        {
            "num_workers": NUM_ENVS,
            "gamma": GAMMA,
            "lr": LEARNING_RATE,
            "env": "toy_abr_maker",
            "env_config": train_env_maker_kwargs,
            "framework": "torch",
            "evaluation_duration": N_EVAL_EPS,
            "evaluation_num_workers": NUM_ENVS,
            "seed": SEEDS[args.idx],
            "num_gpus": 0,
            "v_min": -32,
            "v_max": 32,
            "dueling": True,
            "hiddens": [256],
            "double_q": True, 
            "n_step": N_STEP_RETURN,
            "timesteps_per_iteration": STEPS_PER_ITER    
        }
    )
    config["model"].update(model_config)
    config["evaluation_config"].update(eval_config)
    config["exploration_config"].update(exploration_config)
    config["replay_buffer_config"].update(replay_buffer_config)
    
    config = ApexDQNConfig.from_dict(config)

    trainer = ApexDQN(config=config, logger_creator= null_logger_creator)
    total_iters = int((N_STEPS - replay_buffer_config["learning_starts"]) // (STEPS_PER_ITER * NUM_ENVS))
    checkpoint_idx = 0
    trainer.train()
    for i in tqdm(range(total_iters), desc="Training"):
        metrics = trainer.train()
        progress = np.round((i / (total_iters - 1)) * 100, 2)
        print_metrics(metrics=metrics, kind = f"Training {progress}%")
        if progress >= CHECKPOINT_FREQS[checkpoint_idx]:
            for shared_testing_name in shared_testing_data_names:
                actor = ray.get_actor(shared_testing_name, namespace="test")
                actor.set_train_progress.remote(CHECKPOINT_FREQS[checkpoint_idx])
            metrics = trainer.evaluate()
            print_metrics(metrics=metrics, kind = f"Evaluation {progress}%", is_eval=True)
            checkpoint_idx += 1

    #clean up
    actors_to_terminate = [(shared_testing_name, "test") 
        for shared_testing_name in shared_testing_data_names]
    if args.sampling_func == "plume_dynamic":
        actors_to_terminate.append((name, "plume_dynamic"))
    terminate_shared_data(actors_to_terminate)
    ray.shutdown()
    shutil.rmtree("/tmp/ray")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--curiosity", action="store_true", default=False)
    parser.add_argument("--per", action="store_true", default=False)
    parser.add_argument("--sampling_func", type=str, 
        help = "sampling function: One of (random, plume_static, plume_dynamic)", required=True)
    parser.add_argument("--plume_alpha", default=None, type=float, required=False,
        help="The alpha to use for plume_dynamic weights. See toy_aby_gym/sampler_utils for details.")
    parser.add_argument("--idx", type = int, default=0)
    parser.add_argument("--dataset", type = str, required=True,
        help = ("The name of the dataset to generate, one of "
        "(majority_fast, balanced, majority_slow)"))
    args = parser.parse_args()
    main(args)
    
