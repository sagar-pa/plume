import numpy as np
from gymnasium.utils import seeding
from typing import NamedTuple, List

TRACE_LEN = 100
MIN_TRACE_LEN = 90 # For random starts
MAX_BUFFER = 15.
TRACE_GRANULARITY = 1.0
CHUNK_LEN = TRACE_GRANULARITY
SEED = 9
VAR_MIX_W = {
    False: [.95, .05], # low var
    True: [.7, .3]   # high var
}
DATASETS = {
    "majority_fast": [750, 250],
    "balanced": [500, 500],
    "majority_slow": [250, 750]
}

VALID_THROUGHPUTS = [0.1, 20.] # [low, high]

BITRATES = [1.0, 3.0, 6.0] # Low, Med, High
TIME_STEP_CHUNK_NOISE_DIST = [0, .01] # 1 + normal(dist)
INTER_CHUNK_NOISE_DIST = [0, .0005] # size = birate * time step noise * (1 + inter chunk noise)

class FastTrace(NamedTuple):
    throughput_dists: List[List[float]] = [[9.75, 0.65], [5.75, 0.5]] # [high_throughput [mean, std], low_throughput [mean, std]]
    window_dists: dict[List[List[int]]] = {False: [[3, 0.0125], [1, 0.0125]], # Low var: [high_throughput [mean, std], low_throughput [mean, std]]
                                            True: [[1, 0.0125], [3, 0.0125]]}

class SlowTrace(NamedTuple):
    throughput_dists: List[List[float]] = [[3.25, 0.5], [1.35, 0.35]] # [high_throughput [mean, std], low_throughput [mean, std]]
    window_dists: dict[List[List[int]]] = {False: [[3, 0.0125], [1, 0.0125]], # Low var: [high_throughput [mean, std], low_throughput [mean, std]]
                                            True: [[1, 0.0125], [3, 0.0125]]}

class Trace(NamedTuple):
    network_data: np.ndarray
    video_sizes: np.ndarray

def generate_video(np_random: np.random) -> List:
    """
    Generate a video by adding noise to the bitrates at a time-step and then at a per-action level.

    Args:
        np_random: The seeded numpy.random module
    Returns:
        The video, a list of shape (TRACE_LEN, BITRATES)
    """
    video = []
    bitrates = np.array(BITRATES)
    for _ in np.linspace(0, TRACE_LEN, num=int(TRACE_LEN / CHUNK_LEN)):
        noise = 1 + np_random.normal(*TIME_STEP_CHUNK_NOISE_DIST)
        inter_chunk_noise = 1 + np_random.normal(*INTER_CHUNK_NOISE_DIST, size=len(BITRATES))
        noise = noise * inter_chunk_noise
        sizes = bitrates * noise
        video.append(sizes.tolist())
    return video


def generate_trace(np_random: np.random, is_slow: bool = False, is_high_var: bool = False) -> List:
    """
    Generate a throughput trace by sampling a throughput distribution with weights and using windows to follow it.
    
    Args:
        np_random: The seeded numpy.random module
        is_slow: Whether or not the trace should be Slow or not
        is_high_var: Whether or not the trace should 
            follow the window sizes and mixture weights of a high_var trace
    Returns:
        The generated trace in shape (TRACE_LEN, 2) where columns are [time, throughput]

    """
    trace = []
    window = 0
    mixture_weights = VAR_MIX_W[is_high_var]
    trace_cls = FastTrace if not is_slow else SlowTrace
    trace_info = trace_cls()
    throughput_kind = np_random.choice(2, p=mixture_weights)
    for time in np.linspace(0, TRACE_LEN, num=int(TRACE_LEN / TRACE_GRANULARITY)):
        if window <= 1:
            window = -1
            while window < 0:
                throughput_kind = np_random.choice(2, p=mixture_weights)
                window = np_random.normal(*trace_info.window_dists[is_high_var][throughput_kind])
                window = int(np.round(window))
        else:
            window -= 1
        throughput = np_random.normal(*trace_info.throughput_dists[throughput_kind])
        throughput = np.clip(throughput, *VALID_THROUGHPUTS)
        trace.append([time, throughput])
    return trace

def generate_dataset(dataset_name: str, split: str) -> dict:
    """
    Generate a dataset of traces by repeatedly calling generate_trace and generate_video. 
        It does so in a way that for a fixed seed, the same dataset is generated every time.
    
    Args:
        dataset_name: The name of the dataset to generate, one of (majority_fast, balanced, majority_slow)
        split: The name of the split, one of (train, test)
    Returns:
        The dataset, a dictionary with the keys "traces" and "labels"
    """
    if dataset_name not in DATASETS:
        raise ValueError(f"dataset_name not in {[*DATASETS.keys()]}")
    if split.lower() not in ["train", "test"]:
        raise ValueError("Split must be either 'train' or 'test'")
    np_random, _ = seeding.np_random(seed=SEED)
    datasets = {}
    # Generate all datasets to ensure same traces are generated every time
    for dataset_name_, (n_fast_traces, n_slow_traces) in DATASETS.items():
        for split_ in ["train", "test"]:
            dataset = []
            labels = []
            for speed_name, is_slow, n_traces in [
                                    ["fast", False, n_fast_traces], 
                                    ["slow", True, n_slow_traces]]:
                for _ in range(int(n_traces // 2)):
                    for var_name, is_high_var in [
                            ["high-var", True], 
                            ["low-var", False]]:
                        trace = generate_trace(np_random=np_random, 
                            is_slow=is_slow, is_high_var=is_high_var)
                        video = generate_video(np_random)
                        trace, video = np.array(trace), np.array(video)
                        dataset.append(Trace(network_data=trace, video_sizes=video))
                        labels.append(f"{speed_name}_{var_name}")
            if dataset_name_ not in datasets:
                datasets[dataset_name_] = {}
            if split_ not in datasets[dataset_name_]:
                datasets[dataset_name_][split_] = {}
            datasets[dataset_name_][split_] = dict(traces=dataset, labels=labels)
    return datasets[dataset_name][split]
