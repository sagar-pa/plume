from toy_abr_gym.trace_generator import generate_dataset
from toy_abr_gym.trace_utils import (
    filter_features, 
    determine_n_clusters, 
    extract_trace_features, 
    chunk_traces
    )
from pathlib import Path
import json
import numpy as np
from typing import NamedTuple

class TraceFeatures(NamedTuple):
    features: np.ndarray
    cluster_labels: np.ndarray
    cluster_dists: np.ndarray
    lengths: np.ndarray

def extract_and_save_trace_features(dataset_name: str) -> None:
    """
    Extract and save the filtered trace features to trace_data/settings/trace_features/{dataset_name}.json
        Filters the features and clusters the data automatically, saving the cluster labels 
    
    Args:
        dataset_name: The name of the dataset to feed to generate_dataset
            (see toy_abr_gym/trace_generator/generate_dataset)
    """
    dataset = generate_dataset(dataset_name, split="train")
    chunked_traces = chunk_traces(dataset["traces"])
    settings_file = Path(__file__).parent / "trace_data"/ "settings" / "potential_features.json"
    trace_features = extract_trace_features(chunked_traces, settings=settings_file)
    filtered_features = filter_features(trace_features, features_to_keep=0.5)
    _, cluster_labels, dist, __ = determine_n_clusters(filtered_features, return_labels=True)

    trace_features_lst = filtered_features.to_dict("records")
    to_save = {}
    for i, (trace_label, trace) in enumerate(zip(dataset["labels"], dataset["traces"])):
        to_save[i] = dict(cluster_label=int(cluster_labels[i]),
                            cluster_dist=dist[i].tolist(),
                            trace_label=trace_label,
                            length=trace.video_sizes.shape[0],
                            **trace_features_lst[i])

    with open(Path(__file__).parent / "trace_data" / "trace_features" / f"{dataset_name}.json", "w") as f:
        json.dump(to_save, f, indent=2, sort_keys=True)
    

def load_trace_features(dataset_name: str) -> TraceFeatures:
    """
    Load the saved features for the given dataset name. 
    Assumes extrace_and_save_trace_features was already called.

    Args:
        dataset_name: The name (one of ["majority_slow", "balanced", "majority_fast"])
    Returns:
        The trace features, a np.ndarray of shape [n_traces, n_features]
        The cluster labels, a np.ndarray of shape [n_traces, ]
        The cluster dists, a np.ndarray of shape [n_traces, n_clusters]
        The trace lengths, a np.ndarray of shape [n_trace, ]
    """
    with open(Path(__file__).parent / "trace_data" / "trace_features" / f"{dataset_name}.json", "r") as f:
        trace_features = json.load(f)
    features = [None] * len(trace_features)
    cluster_labels = [None] * len(trace_features)
    cluster_dists = [None] * len(trace_features)
    trace_lengths = [None] * len(trace_features)
    for i, trace_feature in trace_features.items():
        feature = {k: trace_feature[k] for k in sorted(trace_feature.keys()) if k not in [
                                                                "cluster_label", 
                                                                "trace_label", 
                                                                "cluster_dist",
                                                                "length"]}
        i = int(i)
        features[i] = list([feature[key] for key in sorted(feature.keys())])
        cluster_labels[i] = int(trace_feature["cluster_label"])
        cluster_dists[i] = list(trace_feature["cluster_dist"])
        trace_lengths[i] = int(trace_feature["length"])

    features = np.array(features, dtype=np.float64)
    cluster_labels = np.array(cluster_labels, dtype = np.int64)
    cluster_dists = np.array(cluster_dists, dtype = np.float64)
    trace_lengths = np.array(trace_lengths, dtype = np.int64)

    return TraceFeatures(features, cluster_labels, cluster_dists, trace_lengths)
    
