import numpy as np
from typing import List, NamedTuple, Tuple, Union
from pathlib import Path
from csv import DictReader
import random
from math import isinf
import json
from tqdm import tqdm
from abr_gym.trace_downloader import MAX_BUFFER, MIN_TRACE_LEN
from abr_gym.trace_utils import (
    extract_trace_features, chunk_traces, determine_n_clusters, filter_features)
from abr_gym.utils import Trace, TraceInfo
from decimal import Decimal
import pandas as pd


TEST_SPLIT = 0.2
SHUFFLE_SEED = "13"
N_ACTIONS = 10

def load_trace(trace: Path) -> Tuple[Trace, TraceInfo]:
    df = pd.read_feather(trace)
    video_sizes = df[[f"video_size_{i}" for i in range(N_ACTIONS)]].to_numpy()
    ssim_dbs = df[[f"ssim_db_{i}" for i in range(N_ACTIONS)]].to_numpy()
    file_size_sent = video_sizes[np.arange(video_sizes.shape[0]), 
                                    df["action"].to_numpy(dtype=np.int64)]
    throughputs = file_size_sent / (df["time_received"] - df["time_sent"])
    df = df.assign(throughput = throughputs)
    network_data = df[["time_sent", "throughput", "cwnd", "in_flight", 
                  "min_rtt", "rtt", "delivery_rate"]].to_numpy()
    delivery_rates = df["delivery_rate"].astype(object).apply(
        lambda x: Decimal(f"{x:.5f}")).to_numpy(dtype=object)
    actions  = df["action"].to_numpy(dtype=np.int64)
    buffers = df["buffer"].to_numpy().clip(0, MAX_BUFFER)
    rebufs = df["rebuf"].to_numpy().clip(0, 20)

    invalid_mask = np.full(shape=(video_sizes.shape[0],), dtype=bool, fill_value=False)
    for arr in [video_sizes, ssim_dbs]:
        invalid_mask = invalid_mask | np.any(arr <= 0, axis=1)
    for arr in [video_sizes, ssim_dbs, network_data]:
        invalid_mask = invalid_mask | np.any(np.isinf(arr), axis=1)
    mask = ~invalid_mask

    network_data = network_data[mask]
    video_sizes = video_sizes[mask]
    ssim_dbs = ssim_dbs[mask]
    delivery_rates = delivery_rates[mask]
    actions = actions[mask]
    buffers = buffers[mask]
    rebufs = rebufs[mask]

    if network_data.shape[0] < MIN_TRACE_LEN or\
            network_data.shape[1] != 7 or\
            video_sizes.shape[1] != N_ACTIONS or\
            ssim_dbs.shape[1] != N_ACTIONS:
        raise ValueError(f"Invalid shape of data in trace {trace.absolute()}")


    trace_data = Trace(network_data=network_data, video_sizes=video_sizes, 
        ssim_dbs=ssim_dbs, delivery_rates=delivery_rates)
    trace_info = TraceInfo(actions=actions, buffers=buffers, rebufs=rebufs)
    return (trace_data, trace_info)

def load_traces(trace_dir: Path, criteria: Path, 
        split: str= "train") -> List[Path]:
    with open(criteria, "r") as f:
        criteria = json.load(f)
        f.close()
    all_traces = []
    for year in list(trace_dir.iterdir()):
        if not year.is_dir():
            continue
        for month in year.iterdir():
            if not month.is_dir():
                continue
            for day in month.iterdir():
                if not day.is_dir():
                    continue
                for abr in day.iterdir():
                    if not abr.is_dir():
                        continue
                    for cc in abr.iterdir():
                        if not cc.is_dir():
                            continue
                        if abr.stem not in criteria or \
                                cc.stem not in criteria[abr.stem]["cc"]:
                            continue
                        for file in cc.iterdir():
                            if file.is_file():
                                all_traces.append(file)
    all_traces.sort()
    random.seed(SHUFFLE_SEED)
    random.shuffle(all_traces)
    n = int(len(all_traces) * TEST_SPLIT)
    if not isinstance(split, str) or \
            split.upper() not in ["TRAIN", "TEST", "WHOLE"]:
        raise ValueError("Split must either be 'train', 'test' or 'whole'")
    if split.upper() == "TEST":
        all_traces = all_traces[:n]
    elif split.upper() == "TRAIN":
        all_traces = all_traces[n:]
    return all_traces

class TraceFeatures(NamedTuple):
    features: np.ndarray
    cluster_labels: np.ndarray
    cluster_dists: np.ndarray
    lengths: np.ndarray
    delivery_rates: np.ndarray

def extract_and_save_trace_features(all_traces: List[Path], 
        trace_dir: Path,
        features_save_path: Path, 
        suppress_progress=False) -> None:
    """
    Extract and save the filtered trace features to trace_data/settings/trace_features/{dataset_name}.json
        Filters the features and clusters the data automatically, saving the cluster labels 
    
    Args:
        all_traces: The list with all of the path of all of the potential traces
        trace_dir: The path to the root directory of the traces (used to save relative paths of the traces)
        features_save_path: The path to save the resulting features to.
        suppress_progress: Whether or not to disable the tqdm progress bar of loading traces
    """
    loaded_traces = {}
    for trace in tqdm(all_traces, disable=suppress_progress):
        try:
            loaded_trace = load_trace(trace)
            loaded_traces[trace] = loaded_trace
        except (ValueError):
            raise ValueError("Found an invalid trace. Please Clean traces before continuing.")
        
    chunked_traces = chunk_traces(loaded_traces.values())
    settings_file = Path(__file__).parent / "potential_features.json"
    trace_features = extract_trace_features(chunked_traces, settings=settings_file)
    filtered_features = filter_features(trace_features, features_to_keep=0.4)
    _, cluster_labels, dist, __ = determine_n_clusters(filtered_features, return_labels=True)

    trace_features_lst = filtered_features.to_dict("records")
    to_save = {}
    for i, (trace_path, trace) in enumerate(loaded_traces.items()):
        relative_trace_path = [part for part in trace_path.absolute().parts 
            if part not in trace_dir.absolute().parts]
        relative_trace_path = "/".join(relative_trace_path)
        relative_trace_path = relative_trace_path.removesuffix(trace_path.suffix)
        to_save[relative_trace_path] = dict(cluster_label=int(cluster_labels[i]),
                            cluster_dist=dist[i].tolist(),
                            length=trace[0].network_data.shape[0],
                            delivery_rate = np.mean(trace[0].network_data[:, -1]),
                            **trace_features_lst[i])

    with open(features_save_path, "w") as f:
        json.dump(to_save, f, indent=2, sort_keys=True)


def load_trace_features(all_traces: List[Path], 
        trace_dir: Path,
        metrics_file: Path) -> TraceFeatures:
    """
    Load the saved features for the given dataset name. 
    Assumes extrace_and_save_trace_features was already called.

    Args:
        all_traces: The list with all of the path of all of the potential traces
        trace_dir: The path to the root directory of the traces (used to save relative paths of the traces)
        metrics_file: The Path to the file with precomputed metrics
    Returns:
        The trace features, a np.ndarray of shape [n_traces, n_features]
        The cluster labels, a np.ndarray of shape [n_traces, ]
        The cluster dists, a np.ndarray of shape [n_traces, n_clusters]
        The trace lengths, a np.ndarray of shape [n_trace, ]
    """
    with open(metrics_file, "r") as f:
        trace_features = json.load(f)
    features = [None] * len(all_traces)
    cluster_labels = [None] * len(all_traces)
    cluster_dists = [0] * len(all_traces)
    trace_lengths = [0] * len(all_traces)
    delivery_rates = [0] * len(all_traces)
    for i, trace_path in enumerate(all_traces):
        relative_trace_path = [part for part in trace_path.absolute().parts 
            if part not in trace_dir.absolute().parts]
        relative_trace_path = "/".join(relative_trace_path)
        relative_trace_path = relative_trace_path.removesuffix(trace_path.suffix)
        if relative_trace_path not in trace_features:
            raise ValueError(("Attempted to find features for a trace not seen"
                             "before. Please extract and save features"
                             " before continuing."))
        trace_feature = trace_features[relative_trace_path]
    
        feature = {k: trace_feature[k] for k in sorted(trace_feature.keys()) if k not in [
                                                                "cluster_label", 
                                                                "cluster_dist",
                                                                "length",
                                                                "delivery_rate"]}
        features[i] = list([feature[key] for key in sorted(feature.keys())])
        cluster_labels[i] = int(trace_feature["cluster_label"])
        cluster_dists[i] = list(trace_feature["cluster_dist"])
        trace_lengths[i] = int(trace_feature["length"])
        delivery_rates[i] = float(trace_feature["delivery_rate"])

    features = np.array(features, dtype=np.float64)
    cluster_labels = np.array(cluster_labels, dtype = np.int64)
    cluster_dists = np.array(cluster_dists, dtype = np.float64)
    trace_lengths = np.array(trace_lengths, dtype = np.int64)
    delivery_rates = np.array(delivery_rates, dtype = np.float64)

    return TraceFeatures(features, cluster_labels, cluster_dists, trace_lengths, delivery_rates)




def clean_traces(all_traces: List[Path],  verbose: bool = False) -> None:
    """Deletes traces found in all_traces that are corrupt and cannot be used.

    Args:
        all_traces: List of traces to check
    """

    for trace_path in tqdm(all_traces, desc="Traces to check"):
        to_unlink = False
        try:
            load_trace(trace_path)
        except (AssertionError, IndexError, ValueError):
            to_unlink = True
        if to_unlink:
            if verbose:
                print(f"Deleting trace: {trace_path}")
            trace_path.unlink()


def ssim_index_to_db(ssim_index: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    ssim_index = np.asarray(ssim_index)
    ssim_index[ssim_index < 0] = -0.259
    ssim_index[ssim_index > 1] = -0.259 # ssim db for -0.259 = -1
    return -10 * np.log10(1 - ssim_index)

def ssim_db_to_index(ssim_db: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    return 1 - 10 ** (ssim_db / -10)