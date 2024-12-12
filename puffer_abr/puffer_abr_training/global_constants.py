import numpy as np
from pathlib import Path

N_ENVS = 64 # decrease to reduce memory
MAX_TRACE_K = 3
MAX_TRACE_LEN = 500
N_TRAIN_TRACES = 40000
MAX_CACHE_SIZE = None # set to reduce memory
TRAIN_STEPS = 4e8
TRACE_DIR = Path("./traces")
SELECTION_FILE = Path("./train_params/train_select.json")
METRICS_FILE = Path("./train_params/precomputed_trace_metrics.json")
TRAIN_LOG_DIR = Path("./data/experiments/")
LOG_DIR = Path("./data/test_logs/")
SUMMARY_DIR = Path("./data/test_summary")

SEEDS = [13, 103, 223, 347, 463, 607, 743, 883, 919, 937]
SAVE_FREQS = [i for i in range(5, 101, 5)]
TEST_FREQS = [i for i in range(10, 101, 10)]

TEST_MAX_TRACE_LEN = np.inf
N_TEST_TRACES = 15588
N_EVAL_EPS = N_TEST_TRACES + N_ENVS * 5
TEST_SPLIT = "test"

ABR_KWARGS = dict(trace_dir = TRACE_DIR, 
    selection_file=SELECTION_FILE, 
    metrics_file=METRICS_FILE, 
    use_ssim=True, 
    max_traces = N_TRAIN_TRACES, sampling_func_cls="random",
    reward_weights= (1,100,1))
ENV_KWARGS = dict(max_trace_len = MAX_TRACE_LEN, enable_logging=False,
                  reward_norm_style="symmetric_sqrt_clip", 
                  abr_kwargs=ABR_KWARGS)
TEST_ABR_KWARGS = dict(
    trace_dir = TRACE_DIR, 
    selection_file=SELECTION_FILE, 
    metrics_file=METRICS_FILE, 
    use_ssim=True, max_traces = N_TEST_TRACES, split=TEST_SPLIT)
TEST_ENV_KWARGS = dict(max_trace_len = TEST_MAX_TRACE_LEN, enable_logging=True, 
    abr_kwargs=TEST_ABR_KWARGS)