import requests
from datetime import timedelta, datetime
from pathlib import Path
from tqdm import tqdm
import json
import csv
import numpy as np
from collections import defaultdict
from bisect import bisect_right
import re
import multiprocessing as mp
import pandas as pd

from typing import Optional, List, NamedTuple, Dict
from dataclasses import dataclass, field
from math import isinf
import warnings
import time

CHANNELS = ["abc", "nbc", "fox", "pbs", "cbs", "cw"]
VIDEO_FORMATS = 10
MIN_TRACE_LEN = 86 # time spent watching = TRACE_LEN * 2.002
CHUNK_LEN = 2.002
MAX_BUFFER = 15
MAX_SSIM = 0.99999
MIN_SSIM = 1 - MAX_SSIM

class PufferFiles(NamedTuple):
    video_sent: Path
    video_acked: Path
    client_buffer: Path
    video_size: Path
    ssim: Path
    expt: Path

@dataclass(init=True, repr=True, eq=True, order=True)
class BufferStamp:
    time: float
    buffer: float = field(compare=False)
    cum_rebuf: float = field(compare=False)
    event: str = field(compare=False)

#Everything time unit should be in seconds and every size unit in Mb
@dataclass
class Timestamp:
	time_sent: float
	cwnd: int
	in_flight: int
	min_rtt: float
	rtt: float
	delivery_rate: float
	action: int
	video_sizes: List[float]
	ssim_dbs: List[float]
	time_received: float = -1
	buffer: float = -1
	rebuf: float = -1

	def flatten(self):
		return [self.time_sent, self.cwnd, self.in_flight, self.min_rtt, 
                    self.rtt, self.delivery_rate, self.action,
                    self.time_received, self.buffer, self.rebuf
                    ]  + self.video_sizes + self.ssim_dbs


class Stream:
    def __init__(self, id: str, abr: str, cc: str):
        self.id = id
        self.abr = abr
        self.cc = cc
        self.data = {}
        self.sent_time_to_video_time = {}
        self.last_valid_sent_time = -1

    def __eq__(self, o: object) -> bool:
        return isinstance(o, self.__class__) and self.id == o.id

    def __hash__(self) -> int:
        return hash(self.id)

    def record_sent(self, video_time: int, time_sent: float, cwnd: int, 
            in_flight: int, min_rtt: float, rtt: float, 
            delivery_rate: float, action:int, 
            video_sizes: List[float], ssim_dbs: List[float]) -> None:
        self.sent_time_to_video_time[time_sent] = video_time
        self.data[video_time] = Timestamp(
            time_sent, cwnd, in_flight, min_rtt, rtt, delivery_rate, 
            action, video_sizes, ssim_dbs)

    def record_acked(self, video_time: int, time_received: float) -> None:
        if video_time in self.data:
            self.data[video_time].time_received = time_received

    def record_buffer(self, video_time: int, buffer: float, rebuf: float) -> None:
        if video_time in self.data:
            self.data[video_time].buffer = buffer
            self.data[video_time].rebuf = rebuf

    def save_valid(self, location: Path, file: Optional[Path] = None) -> None:
        if file is None:
            trace_name = "{id}".format(
                id=re.sub(r"[^A-Za-z0-9_]", "", self.id))
            file_name = f"{trace_name}.feather"
            file = location / self.abr / self.cc / file_name
            if file.exists():
                i = 0
                while file.exists():
                    i += 1
                    trace_name_check = f"{trace_name}_{i}"
                    file_name = f"{trace_name_check}.feather"
                    file = location / self.abr / self.cc / file_name

        file.parent.mkdir(exist_ok=True, parents=True)
        header = ["time_sent", "cwnd", "in_flight", "min_rtt", "rtt",
            "delivery_rate", "action", "time_received", "buffer", "rebuf"]
        header += [f"video_size_{i}" for i in range(VIDEO_FORMATS)]
        header += [f"ssim_db_{i}" for i in range(VIDEO_FORMATS)]
        if len(self.data) >= MIN_TRACE_LEN:
            rows = []
            start_time = None
            for video_time in sorted(self.data.keys()):  # sort by video time
                timestamp = self.data[video_time]
                if timestamp.time_sent > self.last_valid_sent_time:
                    break
                if start_time is None:
                    start_time = timestamp.time_sent  # normalize video time by start video time
                timestamp.time_sent -= start_time
                timestamp.time_received -= start_time
                row = timestamp.flatten()
                if not any(item <= 0 for item in timestamp.video_sizes + timestamp.ssim_dbs):
                    if not any(item < 0 or isinf(item) for item in row): # if row has invalid data, ignore it
                        rows.append(row)
            if len(rows) >= MIN_TRACE_LEN:
                df = pd.DataFrame(data=rows, columns = header, dtype=np.float64)
                df.to_feather(file)

def make_videoformat_list():
	return [-1 for _ in range(VIDEO_FORMATS)]

def ssim_index_to_db(ssim_index: float) -> float:
    ssim_index = np.clip(ssim_index, MIN_SSIM, MAX_SSIM)
    return -10 * np.log10(1 - ssim_index)

def ssim_db_to_index(ssim_db: float) -> float:
    return 1 - 10 ** (ssim_db / -10)

def parse_video_size(video_size_file: Path, action_lookup: dict) -> dict:
    video_sizes = {}
    for channel in CHANNELS:
        video_sizes[channel] = defaultdict(make_videoformat_list)
    with open(video_size_file, "r", newline="") as size_file:
        reader = csv.DictReader(size_file)
        for row in reader:
            if np.any([row.get(key, None) is None for key in ["channel", 
                    "video_ts", "format", "size"]]):
                continue #invalid row
            channel = row["channel"]
            video_time = int(row["video_ts"])
            format = action_lookup[channel][row["format"]]
            size = float(row["size"]) * 1e-6 # b -> Mb
            video_sizes[channel][video_time][format] = size
    return video_sizes

def parse_ssim(ssim_data_file: Path, action_lookup: dict) -> dict:
    ssim_dbs = {}
    for channel in CHANNELS:
        ssim_dbs[channel] = defaultdict(make_videoformat_list)
    with open(ssim_data_file, "r", newline="") as ssim_file:
        reader = csv.DictReader(ssim_file)
        for row in reader:
            if np.any([row.get(key, None) is None for key in ["channel", 
                    "video_ts", "format", "ssim_index"]]):
                continue #invalid row
            channel = row["channel"]
            video_time = int(row["video_ts"])
            format = action_lookup[channel][row["format"]]
            ssim = float(row["ssim_index"])
            ssim_dbs[channel][video_time][format] = ssim_index_to_db(ssim)  # raw ssim can be an invalid value
    return ssim_dbs

def parse_video_sent(video_sent_file: Path, streams: Dict[str, Stream], 
        expt_lookup: dict, action_lookup: dict,
        video_sizes: dict, ssim_dbs: dict) -> None:
    with open(video_sent_file, "r", newline="") as sent_file:
        reader = csv.DictReader(sent_file)
        for row in reader:
            if np.any([row.get(key, None) is None for key in ["session_id", 
                    "index", "expt_id", "video_ts", "time (ns GMT)",
                    "cwnd", "in_flight", "min_rtt", "rtt", "delivery_rate",
                    "channel", "format"]]):
                continue #invalid row
            id = row["session_id"] + row["index"]
            if id not in streams:
                expt_info = expt_lookup.get(row["expt_id"], {"abr": "puffer_ttp", "cc": "bbr"}) #If unknown expt id, assume it's a puffer_ttp variant; might change offline analytics
                if "abr_name" in expt_info: # for puffer_ttp variants
                    expt_info["abr"] = expt_info["abr_name"] 
                expt_info["abr"] = re.sub(r"[^A-Za-z0-9_]", "", expt_info["abr"])
                expt_info["cc"] = re.sub(r"[^A-Za-z0-9_]", "", expt_info["cc"])
                streams[id] = Stream(id, expt_info["abr"], expt_info["cc"])
            video_time = int(row["video_ts"])
            time_sent = float(row["time (ns GMT)"]) * 1e-9 # ns -> s
            cwnd = int(row["cwnd"])
            in_flight = int(row["in_flight"])
            min_rtt = float(row["min_rtt"]) * 1e-6 # micro s -> s
            rtt = float(row["rtt"]) * 1e-6 # micro s -> s
            delivery_rate = float(row["delivery_rate"]) * 1e-6 # b/s -> Mb/s
            channel = row["channel"]
            action = action_lookup[channel][row["format"]]
            video_sizes_t = video_sizes[channel][video_time]
            ssim_dbs_t = ssim_dbs[channel][video_time] 
            streams[id].record_sent(
                video_time, time_sent, cwnd, in_flight, 
                min_rtt, rtt, delivery_rate, action, video_sizes_t, ssim_dbs_t)

def parse_video_acked(video_acked_file: Path, streams: Dict[str, Stream]) -> None:
    with open(video_acked_file, "r", newline="") as acked_file:
        reader = csv.DictReader(acked_file)
        for row in reader:
            if np.any([row.get(key, None) is None for key in ["session_id", 
                    "index", "video_ts", "time (ns GMT)"]]):
                continue #invalid row
            id = row["session_id"] + row["index"]
            video_time = int(row["video_ts"])
            time_acked = float(row["time (ns GMT)"]) * 1e-9 # ns -> s
            if id in streams:
                streams[id].record_acked(video_time, time_acked)

def parse_buffer(buffer_data_file: Path, streams: Dict[str, Stream]) -> None: 
    streams_to_buffer = {}
    with open(buffer_data_file, "r", newline="") as buffer_file:
        reader = csv.DictReader(buffer_file)
        for row in reader:
            if np.any([row.get(key, None) is None for key in ["session_id", 
                    "index", "time (ns GMT)", "buffer", "cum_rebuf", "event"]]):
                continue #invalid row
            id = row["session_id"] + row["index"]
            time = float(row["time (ns GMT)"]) * 1e-9 # ns -> s
            buffer = np.clip(float(row["buffer"]), 0, MAX_BUFFER)
            cum_rebuf = float(row["cum_rebuf"])
            event = row["event"]
            if id not in streams_to_buffer:
                streams_to_buffer[id] = []
            streams_to_buffer[id].append(BufferStamp(time, buffer, cum_rebuf, event))
    for id in streams_to_buffer:
        if id not in streams:
            continue # unknown id seen in buffer file
        buffer_data = sorted(streams_to_buffer[id])  # sort by time
        last_valid_idx = time_low_buffer_started = last_buffer_data_t = None
        truncated = False
        for idx, buffer_data_t in enumerate(buffer_data):
            if last_buffer_data_t is not None:
                new_rebuf_added = buffer_data_t.cum_rebuf - last_buffer_data_t.cum_rebuf
                new_rebuf_added = max(0, new_rebuf_added) #floating point stuff
                new_time_added = buffer_data_t.time - last_buffer_data_t.time
                if (buffer_data_t.buffer > 5) and \
                        (last_buffer_data_t.buffer > 5) and \
                        (new_rebuf_added > 0.15):
                    last_valid_idx = None # invalid
                    break
                if last_buffer_data_t.time - buffer_data_t.time > 8.:
                    truncated = True
                    break
                if time_low_buffer_started is not None and \
                        ((buffer_data_t.time - time_low_buffer_started) > 20.):
                    truncated = True
                    break
                if (new_rebuf_added - new_time_added) > 0.3: # stalling more than physically possible
                    truncated = True
                    break

            last_valid_idx = idx
            last_buffer_data_t = buffer_data_t
            if buffer_data_t.buffer < 0.3:
                if time_low_buffer_started is None:
                    time_low_buffer_started = buffer_data_t.time
            else:
                    time_low_buffer_started = None

        if last_valid_idx is None or last_valid_idx < 1:
            continue
        if truncated:
            buffer_data = buffer_data[:last_valid_idx+1]
        keys = [buffer_stamp.time for buffer_stamp in buffer_data]
        sent_times = sorted(
            streams[id].sent_time_to_video_time.items(),
            key=lambda item: item[0]) # sort by time chunck sent
        prev_idx = buffer = 0
        last_cum_rebuf = cum_rebuf_at_startup = None
        for sent_time, video_time in sent_times:
            rebuf = 0
            idx = bisect_right(keys, sent_time)  # assumes time around sent time is in buffer data
            if truncated and sent_time > buffer_data[-1].time:
                break # no more data
            for i in range(prev_idx, idx):
                buffer_data_t = buffer_data[i]
                buffer = buffer_data_t.buffer
                event = buffer_data_t.event
                if last_cum_rebuf is not None:
                    rebuf += max(0, buffer_data_t.cum_rebuf - last_cum_rebuf)
                if event != "init":   # don't count start-up delay as rebuffering
                    last_cum_rebuf = buffer_data_t.cum_rebuf
                    if cum_rebuf_at_startup is None:
                        cum_rebuf_at_startup = buffer_data_t.cum_rebuf
            prev_idx = idx
            streams[id].record_buffer(video_time, buffer, rebuf)
            streams[id].last_valid_sent_time = sent_time
        if last_cum_rebuf is None or last_cum_rebuf < cum_rebuf_at_startup:
            streams[id].last_valid_sent_time = -1 # invalid

#Downloads all the files for the date given
def download_csvs(date: datetime, location: Path) -> PufferFiles:
    date.replace(hour=11, minute=0, second=0, microsecond=0)
    next_date = date + timedelta(days=1)
    formatted_date = "{current}_{next}".format(
        current= date.isoformat(timespec="hours"),
        next = next_date.isoformat(timespec="hours"),)
    skeleton = "https://storage.googleapis.com/puffer-data-release/{timespan}/{file}"
    blobs = []
    for filetype in ["video_sent", "video_acked", "client_buffer", "video_size", "ssim"]:
        file = f"{filetype}_{formatted_date}.csv"
        blobs.append((skeleton.format(timespan=formatted_date, file=file), file))
    #Also download the expt settings for the day
    blobs.append(
        (skeleton.format(
        timespan=formatted_date, file="logs/expt_settings"),
        f"expt_settings_{formatted_date}.txt"))
    with requests.Session() as s:
        for blob, file in blobs:
            save_path = location / file
            if save_path.exists(): # don't download a file if it exists (i.e if original is saved)
                continue
            response = s.get(blob)
            with open(save_path, "wb") as f:
                f.write(response.content)
    return PufferFiles(*[location / file for _, file in blobs])

def parse_csvs(files: PufferFiles, date: datetime, trace_dir: Path, 
        abr_select: dict = None) -> None:
    streams = {}
    with open(Path(__file__).parent / "action_lookup.json", "r") as act_file:
        action_lookup = json.load(act_file)
    with open(files.expt, "r") as expt_file:
        expt_lookup = {}
        for line in expt_file:
            idx, *info = line.split()
            expt_lookup[idx] = json.loads(" ".join(info))
    video_sizes = parse_video_size(files.video_size, action_lookup)
    ssim_dbs = parse_ssim(files.ssim, action_lookup)
    parse_video_sent(files.video_sent, streams, expt_lookup, action_lookup, video_sizes, ssim_dbs)
    parse_video_acked(files.video_acked, streams)
    parse_buffer(files.client_buffer, streams)         
    save_path = trace_dir / str(date.year) / str(date.month) / str(date.day)
    for id in streams:
        stream = streams[id]
        to_save = True
        if abr_select is not None:
            if (stream.abr not in abr_select) or (stream.cc not in abr_select[stream.abr]["cc"]):
                to_save = False
        if to_save:
            stream.save_valid(location = save_path)


def process_date(date: datetime, tmp_dir: Path, save_dir: Path, 
        save_original: bool = False, abr_select: dict = None, n_retries: int = 3) -> bool:
    attempt = 0
    while attempt <= n_retries:
        try:
            files = download_csvs(date, tmp_dir)
            parse_csvs(files, date, save_dir, abr_select = abr_select)
            if not save_original:
                for file in files:
                    file.unlink()
            return True
        except (requests.exceptions.RequestException, 
                requests.exceptions.ConnectionError, 
                requests.exceptions.HTTPError,
                requests.exceptions.Timeout):
            attempt += 1
            warnings.warn((f"Couldn't download data for date: {date.date()}."
                            f" Retry ({attempt}/{n_retries})."))
            time.sleep(5)

    warnings.warn(f"Couldn't download data for date: {date.date()}. Skipping.")
    return False #return a response for multiprocessing


def process_date_wrapper(args):
    return process_date(*args)


def generate_dataset(past_days: int, start_date: datetime = None,
        save_original: bool = False,
        save_dir: Path = None, abr_select: Path = None, 
        n_processes: int = 16) -> None:

    global MIN_TRACE_LEN
    if abr_select is not None:
        with open(abr_select, "r") as f:
            abr_select_dict = json.load(f)
    else:
        abr_select_dict = None

    tmp = Path() / "tmp"
    tmp.mkdir(exist_ok=True)
    if save_dir is None:
        save_dir = Path() / "traces"
    save_dir.mkdir(exist_ok=True)

    if start_date is None:
        date = datetime.now()
    else:
        date = start_date
    date = date.replace(hour=11, minute=0, second=0, microsecond=0)
    days_processed = 0
    with mp.Pool(processes=n_processes) as p:
        with tqdm(desc="Processed days", position=0, total=past_days) as pbar:
            while days_processed < past_days:
                dates_to_process = []
                for _ in range(past_days - days_processed):
                    dates_to_process.append([date, tmp, save_dir, save_original, abr_select_dict])
                    date = date - timedelta(days=1)
                iterable = p.imap_unordered(process_date_wrapper, dates_to_process)
                for result_success in iterable:
                    if result_success:
                        pbar.update(1)
                        days_processed += 1


