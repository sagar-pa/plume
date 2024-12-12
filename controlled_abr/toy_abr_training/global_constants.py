from pathlib import Path

NUM_ENVS = 4
N_STEPS = 4e6
LOG_DIR = Path("./toy_abr/new_logs")
SAMPLING_LOG_DIR = Path("./toy_abr/new_sampling_logs")

GAMMA = 0.975
N_STEP_RETURN = 7
LEARNING_RATE = 7.5e-6 #adjust
CHECKPOINT_FREQS = list(range(5, 101, 5))
STEPS_PER_ITER = 20000
SEEDS = [13, 103, 223, 347, 463, 607, 743, 883, 919, 937]
NUM_DATASETS = 3
N_EVAL_EPS = (1000 * NUM_DATASETS) + (5 * NUM_DATASETS *  NUM_ENVS) # extra episodes for overlap