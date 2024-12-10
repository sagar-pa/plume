from argparse import ArgumentParser
from datetime import datetime
import json
from pathlib import Path

from abr_gym.trace_loader import (
    load_traces, extract_and_save_trace_features, clean_traces)
from abr_gym.trace_downloader import generate_dataset

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--start_date", type=str, required=False, default=None,
        help="Date to start downloading data from. Specified in YYYY/MM/DD")
    parser.add_argument("--days", type=int, required=True,
        help="How many days to download. Download from start date and backwards up to days.")
    parser.add_argument("--trace_dir", type=str, required=True)
    parser.add_argument("--abr_select", type=str, help=("Path to json file"
                " specifying which ABR+CC combination to keep traces for."),
                required=False, default=None)
    parser.add_argument("--save_puffer_files", action="store_true", default=False)
    parser.add_argument("--save_trace_features", action="store_true", default=False)
    parser.add_argument("--clean", action="store_true", default=False)
    args = parser.parse_args()
    start_date = traces = plot_dir = None
    parameter_file = Path(".") / "train_params" / "train_select.json"
    metric_file = Path(".") / "train_params" / "precomputed_trace_metrics.json"
    trace_dir = Path(args.trace_dir)
    if args.start_date is not None:
        start_date = datetime(*map(int, args.start_date.split("/")))
    generate_dataset(args.days, start_date=start_date, 
        save_original=args.save_puffer_files,
        save_dir=trace_dir, abr_select = args.abr_select)
    if args.clean:
        traces = load_traces(trace_dir, criteria=parameter_file, split="whole")
        clean_traces(traces, verbose=True)
    if args.save_trace_features:
        traces = load_traces(trace_dir, criteria=parameter_file, split="train")
        extract_and_save_trace_features(all_traces=traces, trace_dir=trace_dir,
                                        features_save_path=metric_file)