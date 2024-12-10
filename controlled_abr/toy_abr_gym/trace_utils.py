import numpy as np
import pandas as pd
from typing import Callable, Union, Tuple, Optional, Iterable
from pathlib import Path
import json
from toy_abr_gym.trace_generator import Trace, TRACE_GRANULARITY
from warnings import warn



def round_step_size(a: Union[np.ndarray, float], 
        step_size: Optional[float] = TRACE_GRANULARITY) -> Union[np.ndarray, float]:
    """
    Round a to the nearest step_size, using the banker's round

    Args:
        a: The num/array to round
        step_size: The granularity to round to
    """
    scaled = a / step_size
    return np.where(scaled % 1 >= 0.5, np.ceil(scaled), np.floor(scaled))*step_size

def chunk_traces(traces: Iterable[Trace]) -> pd.DataFrame:
    """
    Chunk the traces (using round_step_size)
        and aggregate them into a DataFrame to be fed to tsfresh. 
        Assumes traces are tuples where trace.network_data is an array of [time, throughput] columns
    
    Args:
        traces: The traces to chunk
    Returns:
        The dataframe with the columns [id(trace_idx), time, throughput]
    """
    processed_traces = []
    for i, trace in enumerate(traces):
        trace = trace.network_data
        processed_trace = np.ones(shape=(trace.shape[0], trace.shape[1] + 1), dtype=np.float64)
        processed_trace[:,0] = i
        processed_trace[:,1] = round_step_size(trace[:, 0])
        processed_trace[:,2] = trace[:, 1]
        processed_trace = processed_trace.tolist()
        processed_traces.extend(processed_trace)
    return pd.DataFrame(processed_traces, columns = ["id", "time", "throughput"])


def truncated_mean(x: Union[pd.Series, np.ndarray], q: float) -> float:
    """
    Args:
        x: The timeseries to extract features of
        q: The percentage of points higher and lower than p to ignore (0, 0.5)
    Returns:
        The truncated mean with the outlier values ignored
    """
    if not 0 < q < 0.5:
        raise ValueError("Truncated mean is only defined for percentiles between (0,0.5)")
    lower_bound = np.quantile(x, q)
    upper_bound = np.quantile(x, 1-q)
    valid_indices =  np.where((x > lower_bound) & (x < upper_bound))
    return np.mean(x[valid_indices])

def extract_trace_features(timeseries: pd.DataFrame, settings: Optional[Path] = None) -> pd.DataFrame:
    """
    Extract the features of the given timeseries using tsfresh.
        Assumes the timeseries is the one output by chunk_traces.
    
    Args:
        timeseries: The timeseries to extract features for
        settings: The Path to the tsfresh settings file. If not given, all features are extracted.
    Returns:
        The features extracted by tsfresh
    """
    from tsfresh.utilities.dataframe_functions import impute
    from tsfresh import extract_features
    from tsfresh.feature_extraction import feature_calculators
    from tsfresh.feature_extraction import MinimalFCParameters
    
    fc_settings = MinimalFCParameters()
    if settings is not None:
        with open(settings, "r") as f:
            fc_settings = json.load(f)
    
    for f_or_function_name, args in list(fc_settings.items()):
        func = getattr(feature_calculators, f_or_function_name, None)
        if func is None:
            func = globals().get(f_or_function_name, None)
            if func is None:
                warn("Function {} not found. Ignored.".format(f_or_function_name))
            else:
                del fc_settings[f_or_function_name]
                fc_settings[func] = args
            
    extracted_features = extract_features(timeseries, column_id="id", 
        column_sort="time", default_fc_parameters=fc_settings, 
        disable_progressbar=True, impute_function=impute, n_jobs=0)
    extracted_features = extracted_features.sort_index()
    return extracted_features

def cluster(features: np.ndarray, n_clusters: int, max_starts: int = 75
        ) -> Tuple[Callable, np.ndarray, np.ndarray, float]:
    """
    Cluster the given features into n_clusters max_starts number of times, 
        and maximize the mean log expectation of the Gaussian Mixture Model
    
    Args:
        features: The features to cluster
        n_clusters: The number of clusters (mixture components to have)
        max_starts: The max number of random states to try
    Returns:
        A tuple of (The gmm model, 
            The predicted labels (array of shape (n_samples, )),
            The predicted scores of each sampler for each component (array of shape (n_samples, n_components)),
            The mean log expectation)
    """
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import scale

    scaled_features = scale(features, with_mean=True)
    scores = []

    for i in range(max_starts):
        gmm = GaussianMixture(random_state=i, 
            n_components=n_clusters, n_init=1, init_params="k-means++", 
            covariance_type="diag", max_iter=300)
        labels = gmm.fit_predict(scaled_features)
        score = gmm.score(scaled_features)
        scores.append([gmm, labels, gmm.predict_proba(scaled_features), score])

    scores.sort(key= lambda x:x[-1])
    return scores[-1]
    
def calculate_important_features(features: pd.DataFrame, n_clusters: int, n_seeds: int = 75) -> float:
    """
    Calculate the importance of each of the columns of features by 
        clustering the features and then using decision tree to measure the importance of each feature (by information gain)
    
    Args:
        features: The features returned by extrace_trace_features, a DataFrame of (n_traces, n_features)
        n_clusters: The number of clusters to cluster with
        n_seeds: The number of seeds of the decision tree to compute the mean of the scores over.
    Returns:
        The mean importance, a np.ndarray of shape (n_features)
    """
    from sklearn.preprocessing import scale
    from sklearn.tree import DecisionTreeClassifier

    scores = []
    scaled_features = scale(features.values, with_mean=True)
    _, labels, *___ = cluster(scaled_features, n_clusters) 
    for seed in range(n_seeds):
        dt = DecisionTreeClassifier(random_state=seed, class_weight="balanced").fit(scaled_features, labels)
        scores.append(dt.feature_importances_)
        
    return np.mean(np.array(scores), axis=0)

CENTRAL_FEATURES = ["mean", "truncated_mean", "quantile", "root_mean_square", "fft_aggregated"]

def filter_features(features: pd.DataFrame, features_to_keep: float = 0.5, steps: int = 7) -> pd.DataFrame:
    """
    Filter the extracted features by selecting the most important features while
        iteratively increasing the number of clusters and decreasing the number of clusters.
        This strategy allows us to keep a majority of the features when the clustering is poorer 
            (there are too many features to cluster over), and reduce them when the clustering is better.
        When filtering, the top features selected from 
            [features the describe the central tendancy, features that describe the spread] in a balanced way

    Args:
        features: The features to filter, a DataFrame returned by extrace_trace_features
        features_to_keep: The fraction (0, 1) of features to keep
        steps: The number of iterations to carry out
    Returns:
        The filtered version of the features
    """
    features_to_keep = int(features_to_keep * features.shape[1])
    features_to_keep = max(2, features_to_keep)

    n_clusters_counter = np.linspace(3, 7, 
        num=steps, endpoint=True, dtype=np.int64)
    n_features_counter = np.linspace(int(features.shape[1] * .975), features_to_keep, 
        num=steps, endpoint=True, dtype=np.int64)

    are_central_features = [feature.split("__")[1] in CENTRAL_FEATURES for feature in features.columns]
    are_central_features = np.array(are_central_features)
    central_features = features.columns[are_central_features]
    variance_features = features.columns[~are_central_features]
    
    filtered_features = features.copy()
    for n_clusters, n_features in zip(n_clusters_counter, n_features_counter):
        divided_n_features = np.array([(n_features + i) // 2 for i in range(2)], dtype=int) 
        all_scores = calculate_important_features(filtered_features, n_clusters = n_clusters)
        divided_scores = [[], []]
        for score, feature_name in zip(all_scores, filtered_features.columns):
            for facet_i, facet in enumerate([central_features, variance_features]):
                if feature_name in facet:
                    divided_scores[facet_i].append([feature_name, score])
        relevant_features = []
        for faceted_scores, faceted_n_features in zip(divided_scores, divided_n_features):
            faceted_scores.sort(key=lambda x: x[-1], reverse=True)
            relevant_features.extend(
                [feature_name for feature_name, _ in faceted_scores[:faceted_n_features]])
        filtered_features = filtered_features.loc[:, relevant_features]
    
    return filtered_features

def determine_n_clusters(features: pd.DataFrame,
        min_clusters: int = 3,
        max_clusters: int = 8, 
        min_cluster_size: float = 0.0001, 
        return_labels: bool = False,) -> Union[int, Tuple[int, np.ndarray]]:
    """
    Automatically determine the number of clusters 
        between min and max clusters using mahalanobis-distance silhoutte score as the metric
    
    Args:
        features: The features returned by tsfesh of the traces
        min_clusters: The minimum number of clusters to try
        max_clusters: The maximum number of clusters to try
        min_cluster_size: If the smallest cluster is smaller than 
            this fraction (proportional to the number of traces), this clustering is ignored.
        return_labels: Whether or not to return the ideal cluster labels found
    Returns:
        The ideal number of clusters, if return_labels is False
        (number of clusters, labels), otherwise 
    """    
    
    from sklearn.preprocessing import scale
    from sklearn.metrics import silhouette_score
    
    scaled_features = scale(features.values, with_mean=True)

    scores = []
    min_cluster_size = min_cluster_size* features.shape[0]
    for n_clusters in range(min_clusters, max_clusters+1):
        _, labels, dist, ___ = cluster(scaled_features, n_clusters=n_clusters)
        _, counts = np.unique(labels, return_counts=True)
        score = silhouette_score(scaled_features, labels, metric="euclidean")
        if np.amin(counts) <= min_cluster_size:
            score = -1
        scores.append([n_clusters, labels, dist, score])
        
    sorted_scores = sorted(scores, key= lambda x: x[-1])
    best_n_clusters, best_labels, dist, _ = sorted_scores[-1]
    
    if return_labels:
        return (best_n_clusters, best_labels, dist, scores)
    else:
        return best_n_clusters

def get_cluster_weights(labels: np.ndarray, 
        class_weights: np.ndarray = None,
        return_class_weights: Optional[bool] = False, 
        max_pool_inflation: float = None
        ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Calculates the weights of each sample by giving equal
        (or scaled by class weights) importance to all labels.

        Args:
            labels: The labels associated to each sample
            class_weights: The target weights for each class in sorted
                order of unique labels (array 0-1, of shape (unique labels))
            return_class_weights: Whether or not to also return 
                the effective class weights
            max_pool_inflation: if given, is the multiplicative factor 
                by which the weight change will be clipped to.

        Returns:
            The calculated weights of shape (labels) and
                class weights of shape (unique labels) if return_class_weights is True
    """
    if not np.issubdtype(labels[-1], np.integer):
        warn("Labels are not ints. Converting them to np.int64", RuntimeWarning)
        labels = labels.astype(np.int64)
    weights = np.zeros(shape=labels.shape[0], dtype=np.float64)
    unique_labels, class_proportions = np.unique(labels, return_counts=True)
    class_proportions = class_proportions / class_proportions.sum()
    cluster_pools = class_weights
    if cluster_pools is None:
        cluster_pools = np.ones(len(unique_labels), dtype=np.float64) / len(unique_labels)
    cluster_pools /= cluster_pools.sum()
    if max_pool_inflation is not None:
        cluster_pools = np.clip(
            cluster_pools,
            class_proportions / max_pool_inflation,
            class_proportions * max_pool_inflation)
    cluster_pools /= cluster_pools.sum()
    for label, cluster_pool in zip(unique_labels, cluster_pools):
        indices = labels == label
        cluster_pdf = np.ones(shape=indices.sum(), dtype=np.float64)
        cluster_pdf /= cluster_pdf.sum()
        weights[indices] = cluster_pdf * cluster_pool

    if return_class_weights:
        return (weights / weights.sum(), cluster_pools)
    else:
        return weights / weights.sum()

def get_dist_weights(dist: np.ndarray, class_weights: np.ndarray = None,
        return_class_weights: bool = False, max_pool_inflation: float = None) -> np.ndarray:
    """
    The distribution equivalent of get_cluster_weights. 
    Works by assuming every trace has a probability of being in every cluster, 
        and adjusting them to be at class_weights.

    Args:
        dist: 2-D array of shape [n_samples, n_classes] 
            describing the prob to each sampling belonging to each class
        class_weights: 1-D array of shape [n_classes, ] 
            describing the target class weights, if not given, equal weights are assumed.
        return_class_weights: whether or not to return the true class weights used
        max_pool_inflation: if given, is the multiplicative factor 
            by which the weight change will be clipped to.

    Returns:
        The calculated weights of shape (labels) and
            class weights of shape (unique labels) if return_class_weights is True
    """
    class_proportions = dist.sum(axis=0)
    class_proportions = class_proportions / class_proportions.sum()
    cluster_pools = class_weights
    if cluster_pools is None:
        cluster_pools = np.ones((dist.shape[1], ), dtype=np.float64) / dist.shape[1]
    cluster_pools /= cluster_pools.sum()
    if max_pool_inflation is not None:
        cluster_pools = np.clip(
            cluster_pools,
            class_proportions / max_pool_inflation,
            class_proportions * max_pool_inflation)
    cluster_pools /= cluster_pools.sum()

    adjustment = cluster_pools / class_proportions
    weights = dist @ adjustment

    if return_class_weights:
        return (weights / weights.sum(), cluster_pools)
    else:
        return weights / weights.sum()


