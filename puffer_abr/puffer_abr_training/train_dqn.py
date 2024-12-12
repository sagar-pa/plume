from argparse import ArgumentParser, Namespace
from ray.rllib.algorithms.apex_dqn import ApexDQNConfig
from ray.rllib.algorithms.apex_dqn import ApexDQN
from pathlib import Path
import ray
import torch as th
from torch import nn
from tqdm.auto import tqdm
from ray.tune.registry import register_env
import numpy as np
from ray.tune.logger import Logger
import shutil
from abr_gym.utils import Cache, terminate_shared_data, SharedTestingData
from abr_gym.abr_wrapper import ABRWrapper
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from puffer_abr_training.plot import make_puffer_metrics
import pickle as pk
from abr_gym.trace_loader import load_traces

class NullLogger(Logger):

    def __init__(self, config: dict, logdir: str, trial = None):
        super(NullLogger, self).__init__(config, logdir, trial)

    def on_result(self, result):
        pass

def null_logger_creator(config):
    return NullLogger(config, Path("."))

NUM_CPUS = 16
N_ENV_PER_CPU = 4
NUM_ENVS = 64 # 16 * 4
N_STEPS = 1e9
LOG_DIR = Path("./puffer/test_logs")
GAMMA = 0.95
N_STEP_RETURN = 7
LEARNING_RATE = 7.5e-6 #adjust
CHECKPOINT_FREQS = list(range(10, 101, 10))
STEPS_PER_ITER = 20000
SEEDS = [13, 103, 223, 347, 463, 607, 743, 883, 919, 937]
N_TEST_TRACES = 15588
N_TRAIN_TRACES = 40000
N_EVAL_EPS = N_TEST_TRACES + 5 * NUM_ENVS # extra episodes for overlap


TRACE_DIR = Path("./puffer/traces_feather/")
LOG_DIR = Path("./puffer/test_logs/")
SUMMARY_DIR = Path("./puffer/test_summary")
TEST_ABR_KWARGS = dict(
    trace_dir = TRACE_DIR, 
    selection_file=Path("./train_params/train_select.json"), 
    metrics_file=Path("./train_params/precomputed_trace_metrics.json"), 
    use_ssim=True, max_traces = N_TEST_TRACES, split="test", reward_weights= (1,100,1))
TEST_ENV_KWARGS = dict(max_trace_len = np.inf, enable_logging=True, 
    abr_kwargs=TEST_ABR_KWARGS, reward_norm_style="symmetric_sqrt_clip")
TRAIN_ABR_KWARGS = dict(trace_dir = TRACE_DIR, 
    selection_file=Path("./train_params/train_select.json"), 
    metrics_file=Path("./train_params/precomputed_trace_metrics.json"), use_ssim=True, 
    max_traces = N_TRAIN_TRACES, sampling_func_cls="random",
    reward_weights= (1,100,1), sampler_kwargs = {})
TRAIN_ENV_KWARGS = dict(max_trace_len = 500, enable_logging=False,
    abr_kwargs=TRAIN_ABR_KWARGS, reward_norm_style="symmetric_sqrt_clip")

class SmallCNN(TorchModelV2,  nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name, features_dim: int = 256, history_len: int = 10, 
            network_features: int = 6):

        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config,
                         name)
        nn.Module.__init__(self)
        self.network_features = network_features
        self.history_len = history_len
        self.network_cnn = nn.Sequential(
            nn.Conv1d(self.history_len, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),   
            nn.Flatten())
        self.video_cnn = nn.Sequential(
            nn.Conv1d(5, 32, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.Flatten())
        self.quality_cnn = nn.Sequential(
            nn.Conv1d(5, 32, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.Flatten())


    # Implement your own forward logic, whose output will then be sent
    def forward(self, input_dict, state, seq_lens):
        observations = input_dict["obs"].float() # Tensor
        network_features = self.network_cnn(
            observations[..., 0:self.history_len, 0:self.network_features])
        video_features = self.video_cnn(
            observations[..., 0:5, self.network_features:self.network_features+10])
        quality_features = self.quality_cnn(
            observations[..., 0:5, self.network_features+10:self.network_features+20])
        features = th.cat((network_features, video_features, quality_features), dim=1)
        return features, []


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
                                ["timers", ["learner_overall_throughput"]]
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
    sampling_name = args.sampling_func
    name = f"apex__{sampling_name}__{per_name}__{args.idx}"


    Cache.options(lifetime="detached", name=name, namespace="cache").remote(max_size = None)
    TRAIN_ABR_KWARGS["sampler_kwargs"].update(dict(
        trace_cache_name=name))
    TRAIN_ABR_KWARGS["sampling_func_cls"] = args.sampling_func 
    
    log_dir = LOG_DIR / name
    log_dir.mkdir(parents=True, exist_ok=True)
    test_traces = load_traces(trace_dir=TRACE_DIR, 
        criteria=TEST_ABR_KWARGS["selection_file"], split="test")
    n_traces = min(len(test_traces), N_TEST_TRACES)
    SharedTestingData.options(lifetime = "detached", name=name, namespace="test").remote(
        n_traces
    )
    TEST_ENV_KWARGS.update(dict(
        shared_testing_data_name=name, log_dir=log_dir))

    def env_maker(maker_kwargs: dict) -> ABRWrapper:
        env = ABRWrapper(**maker_kwargs)
        return env
    register_env("abr_maker", env_maker)
    ModelCatalog.register_custom_model("small_cnn", SmallCNN)
    config = ApexDQNConfig().to_dict()
    buffer_type = "MultiAgentPrioritizedReplayBuffer" if args.per else "MultiAgentReplayBuffer"
    replay_buffer_config = {
        "type": buffer_type,
        "capacity": 2000000, #adjust
        "replay_sequence_length": 1, 
        "learning_starts": 50000,
    }
    exploration_config = {
        "type": "PerWorkerEpsilonGreedy",
        "initial_epsilon": 1.0, 
        "final_epsilon": 0.02, #adjust 
        "warmup_timesteps": replay_buffer_config["learning_starts"]
    }
    model_config = {
        "custom_model": "small_cnn",
        "custom_model_config": {},
        "no_final_linear": True,
        "framestack": False,
        "zero_mean": False,
    }
    eval_config = {
        "explore": False,
        "env_config": TEST_ENV_KWARGS,
        "num_envs_per_worker": N_ENV_PER_CPU,
    }
    config.update(
        {
            "num_workers": NUM_CPUS,
            "num_envs_per_worker": N_ENV_PER_CPU,
            "gamma": GAMMA,
            "lr": LEARNING_RATE,
            "env": "abr_maker",
            "env_config": TRAIN_ENV_KWARGS,
            "framework": "torch",
            "evaluation_duration": N_EVAL_EPS,
            "evaluation_num_workers": NUM_CPUS,
            "seed": SEEDS[args.idx],
            "num_gpus": 0,
            "v_min": -32,
            "v_max": 32,
            "dueling": True,
            "hiddens": [256],
            "double_q": True, 
            "n_step": N_STEP_RETURN,
            "timesteps_per_iteration": STEPS_PER_ITER,
            "train_batch_size": 128    
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
            actor = ray.get_actor(name=name, namespace="test")
            actor.set_train_progress.remote(CHECKPOINT_FREQS[checkpoint_idx])
            metrics = trainer.evaluate()
            print_metrics(metrics=metrics, kind = f"Evaluation {progress}%", is_eval=True)
            checkpoint_idx += 1

    for train_progress in CHECKPOINT_FREQS:
        all_metric, slow_metric = make_puffer_metrics(log_dir / str(train_progress), 
            glob_str="test_*.feather", name=name,
            train_progress=train_progress)
        for file_ending, metric in [("all", all_metric), ("slow", slow_metric)]:
            summary_path = SUMMARY_DIR / f"{name}_{train_progress}_{file_ending}.pkl"
            with open(summary_path, "wb") as summary_pik:
                pk.dump(metric, summary_pik)

    #clean up
    actors_to_terminate = [(name, "test"), (name, "cache")]
    terminate_shared_data(actors_to_terminate)
    ray.shutdown()
    shutil.rmtree("/tmp/ray")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--per", action="store_true", default=False)
    parser.add_argument("--sampling_func", type=str, 
        help = "sampling function: One of (random, naive_weighted)", required=True)
    parser.add_argument("--idx", type = int, default=0)
    args = parser.parse_args()
    main(args)
    
