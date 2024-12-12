from pathlib import Path
from csv import DictReader
import numpy as np
from typing import List, Optional, Tuple, NamedTuple, Callable
from dataclasses import dataclass, astuple
from statsmodels.stats.weightstats import DescrStatsW
import warnings
from statistics import mean
from decimal import Decimal
import pandas as pd
from abr_gym.utils import make_reward_norm_func


from abr_gym.trace_loader import ssim_index_to_db
from copy import deepcopy

CHUNK_LEN = 2.002 #seconds
SLOW_THRESHOLD = Decimal("0.75")

@dataclass
class TraceData:
    idx: int
    is_slow: bool
    timestamps: int
    avg_ssim: float
    time_spent_stalled: float
    avg_action_change: float
    avg_buffer: float
    avg_reward: float = 0
    avg_raw_reward: float = 0
    avg_quality_reward: float = 0
    avg_rebuf_reward: float = 0
    avg_quality_change_reward: float = 0
    avg_cost: float = 0

    def __iter__(self):
        return iter(astuple(self))
@dataclass
class StreamData:
    name: str
    train_progress: int
    is_slow: bool
    avg_ssim: float = 0
    time_spent_stalled: float = 0
    avg_action_change: float = 0
    avg_buffer: float = 0
    avg_reward: float = 0
    avg_raw_reward: float = 0
    avg_cost: float = 0
    avg_ssim_low: float = 0
    avg_ssim_hi: float = 0
    time_spent_stalled_low: float = 0
    time_spent_stalled_hi: float = 0
    avg_action_change_low: float = 0
    avg_action_change_hi: float = 0
    avg_buffer_low: float = 0
    avg_buffer_hi: float = 0
    timestamps: int = 0
    avg_reward_low: float = 0
    avg_reward_hi: float = 0
    avg_raw_reward_low: float = 0
    avg_raw_reward_hi: float = 0
    avg_cost_low: float = 0
    avg_cost_hi: float = 0
    raw_trace_data: List[TraceData] = None

    def __iter__(self):
        return iter(astuple(self))

class RewardWeights(NamedTuple):
    quality_weight: float = 1
    rebuf_weight: float = 100
    quality_change_weight: float = 1

def make_reward_function(weights: RewardWeights, 
            norm_function: Optional[Callable] = None) -> Callable:
    def reward_function(quality, rebuf, quality_change) -> float:
        quality_rew = weights.quality_weight * quality
        rebuf_rew = - weights.rebuf_weight * rebuf
        quality_change_rew = - weights.quality_change_weight * quality_change
        reward = quality_rew + rebuf_rew + quality_change_rew
        if norm_function is not None:
            reward = norm_function(reward)
        return reward, quality_rew, rebuf_rew, quality_change_rew
    return reward_function


def add_stream_metrics(data: List, idx: int,
        all_stream_data: List, slow_stream_data: List) -> None:
    mean_delivary_rate = mean([row[2] for row in data]) # deterministic decimal calculation
    for i in range(len(data)):
        data[i][2] = float(data[i][2]) # convert to float
    data = np.array(data)
    mean_data = np.mean(data, axis=0)
    cum_rebuf = np.sum(data[:, 1])
    is_slow = False
    if mean_delivary_rate <= SLOW_THRESHOLD: #From puffer stream metrics; bytes -> megabytes
        is_slow = True
    
    trace = TraceData(
            idx = idx, 
            is_slow = is_slow, 
            timestamps = data.shape[0], 
            avg_ssim = mean_data[0], 
            time_spent_stalled = cum_rebuf,
            avg_action_change = mean_data[3], 
            avg_buffer = mean_data[4], 
            avg_reward = mean_data[5], 
            avg_raw_reward= mean_data[6],
            avg_quality_reward = mean_data[7],
            avg_rebuf_reward= mean_data[8], 
            avg_quality_change_reward= mean_data[9],
            avg_cost=mean_data[10])
    
    all_stream_data.append(TraceData(*deepcopy(astuple(trace))))
    if is_slow:
        slow_stream_data.append(TraceData(*deepcopy(astuple(trace))))
        
    

def make_puffer_metrics(directory: Path, glob_str: str,
        name: str = "", train_progress: int = 100) -> Tuple[StreamData, StreamData]:
    all_stream_data = []
    slow_stream_data = []
    reward_func = make_reward_function(RewardWeights())
    *_, reward_norm_func = make_reward_norm_func(style="symmetric_sqrt_clip", max_reward=30)
    for file in directory.glob(glob_str):
        _, trace_idx = file.stem.split("_")
        trace_idx = int(trace_idx)
        df = pd.read_feather(file)
        ep = 0
        data = []
        last_action = None
        last_ssim_db = None

        for _, row in df.iterrows():
            try:
                read_ep = row["episode"]
                if read_ep != ep:
                    if len(data) > 1:
                        add_stream_metrics(data=data,
                            idx = trace_idx, 
                            all_stream_data=all_stream_data,
                            slow_stream_data=slow_stream_data)
                    data = []
                    ep = read_ep
                    last_action = None
                    last_ssim_db = None
                if last_action is None:
                    last_action = row["action"]
                    last_ssim_db = ssim_index_to_db(row["ssim"])
                ssim_db = ssim_index_to_db(row["ssim"])
                ssim_db_change = abs(last_ssim_db - ssim_db)
                raw_reward, quality_rew, rebuf_rew, quality_change_rew = \
                    reward_func(ssim_db, row["rebuf"], ssim_db_change)
                reward = reward_norm_func(raw_reward)
                data.append([row["ssim"],
                    row["rebuf"], 
                    Decimal(f"{row['delivery_rate']:.5f}"), 
                    abs(last_action - row["action"]), 
                    row["buffer"], 
                    reward, 
                    raw_reward,
                    quality_rew, rebuf_rew, quality_change_rew,
                    row.get("cost", 0)])
                last_action = row["action"]
                last_ssim_db = ssim_db
            except Exception as e:
                warnings.warn(f"Caught {e.args}.", RuntimeWarning)
                continue

        if len(data) > 1: # last episode
            add_stream_metrics(data=data,
                idx = trace_idx,  
                all_stream_data=all_stream_data,
                slow_stream_data=slow_stream_data)

    output = []
    for stream_data, is_slow in [(all_stream_data, False), (slow_stream_data, True)]:
        summary = aggregate_puffer_data(stream_data=stream_data, is_slow=is_slow,
            name=name, train_progress=train_progress)
        output.append(summary)
    return output


def aggregate_puffer_data(stream_data: List[TraceData], 
        is_slow: bool,
        to_ignore_trace_indices: List[int] = None, 
        name: str = "", train_progress: int = 100) -> StreamData:
    if to_ignore_trace_indices is None:
        to_ignore_trace_indices = []
    stream_data_copy = [TraceData(*deepcopy(astuple(stream))) for stream \
                        in stream_data if stream.idx not in \
                            to_ignore_trace_indices]
    for idx in range(len(stream_data_copy)):
        stall_ratio = stream_data_copy[idx].time_spent_stalled /\
            (stream_data_copy[idx].time_spent_stalled  + (stream_data_copy[idx].timestamps * CHUNK_LEN))
        stall_ratio *= 100
        stream_data_copy[idx].time_spent_stalled = stall_ratio
    weights = np.array([stream.timestamps for stream in stream_data_copy])
    stream = StreamData(name=name, train_progress= train_progress, 
        is_slow=is_slow, raw_trace_data=stream_data)
    total_timestamps = sum(stream.timestamps for stream in stream_data_copy)
    stream.timestamps = total_timestamps
    for attribute in ["avg_ssim", "avg_action_change", 
            "time_spent_stalled", "avg_buffer", "avg_reward", "avg_raw_reward",
            "avg_cost"]:
        arr = np.array([getattr(trace_data, attribute) 
            for trace_data in stream_data_copy])
        weighted_stats = DescrStatsW(arr, weights=weights)
        mean = weighted_stats.mean
        low_ci, high_ci = weighted_stats.zconfint_mean(alpha=0.05)
        setattr(stream, attribute, mean)
        setattr(stream, f"{attribute}_low", low_ci)
        setattr(stream, f"{attribute}_hi", high_ci)
    for attribute, value in list(stream.__dict__.items()):
        if "ssim" in attribute:
            setattr(stream, attribute, ssim_index_to_db(value))
    return stream
    