from typing import List, Tuple, Optional
from datetime import timedelta, datetime
import requests
import warnings
import time
from dataclasses import dataclass, astuple
from decimal import Decimal
from abr_gym.trace_downloader import ssim_db_to_index, ssim_index_to_db
import multiprocessing as mp
from collections import defaultdict
import pickle as pk
from tqdm.auto import tqdm
from pathlib import Path
import numpy as np
from statsmodels.stats.weightstats import DescrStatsW
from argparse import ArgumentParser
import json
from gymnaisum.utils.seeding import np_random
from scipy.stats import bootstrap

BOOTSTRAP_SEED = 19
N_SAMPLES = 10000
CHUNK_LEN = 2.002 #seconds
SLOW_THRESHOLD = Decimal("0.75")
BYTES_TO_MB = Decimal("1e-6")
PUFFER_STATS_GLOB = "https://storage.googleapis.com/puffer-data-release/{current_date}_{next_date}/stream_stats_{current_date}_{next_date}.txt"
SAVE_LOCATION = Path("./puffer/platform_plot_data")

@dataclass
class SessionData:
    name: str
    is_slow: bool = False
    timestamps: float = 0
    avg_ssim: float = 0
    time_spent_stalled: float = 0
    avg_ssim_deviation: float = 0

    def __iter__(self):
        return iter(astuple(self))    


@dataclass
class AggregateData:
    name: str
    is_slow: bool
    avg_ssim: float = 0
    time_spent_stalled: float = 0
    avg_ssim_deviation: float = 0
    avg_ssim_low: float = 0
    avg_ssim_hi: float = 0
    time_spent_stalled_low: float = 0
    time_spent_stalled_hi: float = 0
    avg_ssim_deviation_low: float = 0
    avg_ssim_deviation_hi: float = 0
    timestamps: int = 0
    raw_session_data: List[SessionData] = None

    def __iter__(self):
        return iter(astuple(self))
    
def parse_day_stats(file: Path) -> List[SessionData]:
    full_stats = []
    with open(file, "r") as f:
        lines = f.readlines()
        raw_streams = [line.split() for line in lines]
        raw_streams = [[stat.split("=") for stat in split_line] for split_line in raw_streams]
        for raw_stream in raw_streams:
            is_valid = True
            session = SessionData(name = None)
            for (stat_name, value) in raw_stream:
                if stat_name == "scheme":
                    abr, _ = value.split("/")
                    session.name = abr
                if stat_name == "valid":
                    if value != "good":
                        is_valid = False
                        break
                if stat_name == "mean_delivery_rate":
                    delivery_rate = Decimal(value) * BYTES_TO_MB
                    is_slow = delivery_rate <= SLOW_THRESHOLD
                    session.is_slow = is_slow
                if stat_name == "total_after_startup":
                    session.timestamps = float(value)
                if stat_name == "stall_after_startup":
                    session.time_spent_stalled = float(value)
                if stat_name == "mean_ssim":
                    session.avg_ssim = float(value)
                if stat_name == "ssim_variation_db":
                    session.avg_ssim_deviation = ssim_db_to_index(float(value))
            if session.timestamps <= 1e-3:
                is_valid = False
            else:
                session.time_spent_stalled = 100 * session.time_spent_stalled /\
                                                                    session.timestamps
            if np.any(np.isnan(np.array([session.timestamps, 
                                        session.avg_ssim, 
                                        session.avg_ssim_deviation, 
                                        session.time_spent_stalled], 
                                dtype=np.float64)
                                )
                        ):
                is_valid = False
            if is_valid:
                full_stats.append(session)
    return full_stats
                

def process_date(date: datetime, save_dir: Path,
        save_original: bool = True, n_retries: int = 3) -> Tuple[bool, Optional[List[SessionData]]]:
    attempt = 0
    while attempt <= n_retries:
        try:
            date = date.replace(hour=11, minute=0, second=0, microsecond=0)
            next_date = date + timedelta(days=1)
            save_dir.mkdir(parents=True, exist_ok=True)
            with requests.Session() as s:
                file_url = PUFFER_STATS_GLOB.format(
                    current_date=date.isoformat(timespec="hours"),
                    next_date=next_date.isoformat(timespec="hours"))
                save_file_name = f"puffer_day_stats__{date.date()}.txt"
                save_path = save_dir / save_file_name
                if not save_path.exists(): # don't download a file if it exists (i.e if original is saved)
                    response = s.get(file_url)
                    with open(save_path, "wb") as f:
                        f.write(response.content)
            stats = parse_day_stats(save_path)
            if not save_original:
                    save_path.unlink()
            return (True, stats)
        except (requests.exceptions.RequestException, 
                requests.exceptions.ConnectionError, 
                requests.exceptions.HTTPError,
                requests.exceptions.Timeout):
            attempt += 1
            warnings.warn((f"Couldn't download data for date: {date.date()}."
                            f" Retry ({attempt}/{n_retries})."))
            time.sleep(5)
        except ValueError:
            warnings.warn(f"Couldn't download data for date: {date.date()}. Skipping.")
            return (False, None)

    warnings.warn(f"Couldn't download data for date: {date.date()}. Skipping.")
    return (False, None) #return a response for multiprocessing

def process_date_wrapper(args):
    return process_date(*args)

def aggregate_stream_data(session_data: List[SessionData], 
        is_slow: bool,
        name: str = "") -> AggregateData:
    rand, _ = np_random(seed=BOOTSTRAP_SEED)
    weights = np.array([session.timestamps for session in session_data])
    sample_weights = weights / weights.sum()
    stream = AggregateData(name=name, is_slow=is_slow, raw_session_data=session_data)
    total_timestamps = sum(session.timestamps for session in session_data)
    stream.timestamps = total_timestamps
    for attribute in ["avg_ssim", 
            "time_spent_stalled", "avg_ssim_deviation"]:
        arr = np.array([getattr(session, attribute) 
            for session in session_data], dtype=np.float64)
        # bootstrap confidence interval
        indices = rand.choice(len(arr), size=(N_SAMPLES, ), replace=True, p=sample_weights)
        samples = arr[indices]
        mean = np.mean(samples)
        res = bootstrap((samples,), np.mean, confidence_level=0.95, 
            random_state=BOOTSTRAP_SEED)
        low_ci, high_ci = res.confidence_interval
        setattr(stream, attribute, mean)
        setattr(stream, f"{attribute}_low", low_ci)
        setattr(stream, f"{attribute}_hi", high_ci)
    for attribute, value in list(stream.__dict__.items()):
        if "ssim" in attribute:
            setattr(stream, attribute, ssim_index_to_db(value))
    return stream

def generate_platform_plot_summary(past_days: int, start_date: datetime = None,
        save_original: bool = True,
        save_dir: Path = None, 
        dates_to_ignore: List[str] = None,
        n_processes: int = 16) -> None:

    if save_dir is None:
        save_dir = Path() / "platform_plot_data"
    save_dir.mkdir(parents=True, exist_ok=True)
    tmp = save_dir / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)

    if start_date is None:
        date = datetime.now()
    else:
        date = start_date
    if dates_to_ignore is None:
        dates_to_ignore = []
    dates_to_ignore = [datetime(*map(int, ignore_date.split("/"))) for ignore_date in dates_to_ignore]
    dates_to_ignore = [ignore_date.replace(hour=11, minute=0, second=0, microsecond=0) for ignore_date in dates_to_ignore]
    date = date.replace(hour=11, minute=0, second=0, microsecond=0)
    days_processed = 0
    all_stats = defaultdict(list)
    with mp.Pool(processes=n_processes) as p:
        with tqdm(desc="Processed days", position=0, total=past_days) as pbar:
            while days_processed < past_days:
                dates_to_process = []
                for _ in range(past_days - days_processed):
                    if date not in dates_to_ignore:
                        dates_to_process.append([date, tmp, save_original])
                    date = date - timedelta(days=1)
                iterable = p.imap_unordered(process_date_wrapper, dates_to_process)
                for result_success, day_stats in iterable:
                    if result_success:
                        pbar.update(1)
                        days_processed += 1
                        for session in day_stats:
                            all_stats[(session.name, False)].append(session) # all sessions
                            if session.is_slow:
                                all_stats[(session.name, session.is_slow)].append(session)
    for (name, is_slow) in all_stats:
        aggregate_stats = aggregate_stream_data(all_stats[(name, is_slow)], 
            is_slow=is_slow, name=name)
        file_ending = "slow" if is_slow else "all"
        summary_path = save_dir / f"{name}__{file_ending}.pkl"
        with open(summary_path, "wb") as summary_pik:
            pk.dump(aggregate_stats, summary_pik)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--days_to_process", type=int, 
        help=("Number of days to process. The program starts at "
        "start_date and goes backwards until specified."), required=True)
    parser.add_argument("--dates_to_ignore_file", type=str, 
        help = ("Path to a json file with all the dates"
        " (strings of YYYY/MM/DD) to ignore specified"),
        default=None,  required=False)
    parser.add_argument("--start_date", type=str, 
        help="Date to start at, specified as YYYY/MM/DD", default= None, required=False)
    parser.add_argument("--save_dir", type=str, 
        help="Path to directory to save all files to", default=None, required=False)
    parser.add_argument("--clean_puffer_files_after", action="store_true", default=False,
        help="Whether or not to delete all the stream data files downloaded")
    args = parser.parse_args()
    save_dir = args.save_dir
    start_date = args.start_date
    dates_to_ignore = []
    if save_dir is None:
        save_dir = SAVE_LOCATION
    else:
        save_dir = Path(save_dir)
    if start_date is not None:
        start_date = datetime(*map(int, start_date.split("/")))
    if args.dates_to_ignore_file is not None:
        with open(args.dates_to_ignore_file, "r") as f:
            dates_to_ignore = json.load(f)
    
    generate_platform_plot_summary(past_days= args.days_to_process,
                                    start_date= start_date,
                                    save_original= not args.clean_puffer_files_after,
                                    save_dir= save_dir,
                                    dates_to_ignore=dates_to_ignore)