from argparse import ArgumentParser, Namespace
from pathlib import Path
import json
from csv import DictReader
from typing import Callable
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
from tqdm.auto import tqdm
from matplotlib.legend_handler import HandlerLine2D

font = "Verdana"
plt.rcParams["font.family"] = font
mpl.rcParams['font.family'] = font
mpl.rcParams['lines.markersize'] = 10
sns.set_theme(style="whitegrid", font_scale=1.5, font=font)


WEIGHTS = {
    "reward": 1.,
    "quality": 1.,
    "rebuf": -1.
}
KEY_TO_NAME = {
    "reward": "Reward"
}
TRACE_TO_NAME = {
    "slow": {
        "low-var": "Slow Low-Var",
        "high-var": "Slow High-Var",
    },
    "fast": {
        "low-var": "Fast Low-Var",
        "high-var": "Fast High-Var"
    }
}
SUMMARY_FILE = "plot_summary.parquet.gzip"
PROCESSED_AGENTS_FILE = "agents_in_summary.json"
COLUMNS = ["Prioritization", "Training Progress", "Trace", "Feature", "Value"]

def make_quantile_func(quantile: float) -> Callable:
    """
    Make a wrapper function that computes the quantile upon given the values
    """
    def func(a):
        return np.quantile(a, q=quantile)
    return func


def process_new_logs(log_dir: Path, summary_file: Path, processed_agents: Path,
        n_batches: int = 30) -> None:
    """
    Read and process all the csv files in log_directory, saving them to the summary file.
    Args:
        log_dir: path of the logs
        summary_file: path to the summary file
        processed_agents: path to the agents already processed
    """
    all_agents = np.array(list(log_dir.iterdir()), dtype=object)
    pbar =  tqdm(total=len(all_agents), desc="All Agents")
    splitted_arrs = np.array_split(all_agents, max(1,int(len(all_agents) / n_batches)))
    for batch_dirs in splitted_arrs:
        if summary_file.exists():
            summary_data = pd.read_parquet(summary_file)
        else:
            summary_data = pd.DataFrame(columns=COLUMNS)
        if processed_agents.exists():
            with open(processed_agents, "r") as f:
                processed_agents_set = set(json.load(f))
        else:
            processed_agents_set = set()

        read_data = []
        for agent_log_dir in batch_dirs:
            pbar.update(1)
            *_, agent_log_dir_stem = str(agent_log_dir.absolute()).split("/")
            if not agent_log_dir.is_dir():
                continue
            if agent_log_dir_stem in processed_agents_set:
                continue
            processed_agents_set.add(agent_log_dir_stem)
            *agent, agent_idx, dataset_name = agent_log_dir_stem.split("__")
            agent = "__".join(agent)
            agent = f"{agent}__{dataset_name}"
            for train_progress_dir in agent_log_dir.iterdir():
                if not train_progress_dir.is_dir():
                    continue
                try:
                    train_progress = int(float(train_progress_dir.stem))
                except ValueError:
                    raise RuntimeError(f"Couldn't process dir: {train_progress_dir.stem} in {agent_log_dir}")
                for trace_file in train_progress_dir.iterdir():
                    __, speed, var = trace_file.stem.split("_")
                    trace_name = TRACE_TO_NAME[speed][var]
                    with open(trace_file, "r", newline="") as f:
                        reader = DictReader(f)
                        for row in reader:
                            for feature_key, feature_name in KEY_TO_NAME.items():
                                value = float(row[feature_key])
                                value *= WEIGHTS[feature_key]
                                read_data.append([agent, train_progress, trace_name, feature_name, value])

        read_data = pd.DataFrame(read_data, columns=COLUMNS)
        summary_data = pd.concat([summary_data, read_data])

        summary_data.to_parquet(summary_file, compression="gzip")
        with open(processed_agents, "w") as f:
            processed_agents_set = json.dump(list(processed_agents_set), f)



def main(args: Namespace) -> None:
    """
    Plot detailed logs created while training, saving every data 
        to a summary file for use later (loading it this time if exists).
    Args:
        args: A namespace with the following information:
            log_dir: The directory of logs provided in train.py
            plot_args_file: The file to look up plotting arguments from
            save_dir: Where to save the resulting plots and summary files to.
    """
    log_dir = Path(args.log_dir)
    plot_args_file = Path(args.plot_args_file)
    save_dir = Path(args.save_dir)


    summary_file = save_dir / SUMMARY_FILE
    processed_file = save_dir / PROCESSED_AGENTS_FILE
    if not args.skip_processing:
        process_new_logs(log_dir=log_dir, 
            summary_file=summary_file, 
            processed_agents=processed_file)

    summary_data = pd.read_parquet(summary_file)    
    with open(plot_args_file) as f:
        plot_args = json.load(f)
    agents = set(plot_args["names"].keys())
    summary_data = summary_data.loc[summary_data["Prioritization"].isin(agents)]
    summary_data["Prioritization"] = summary_data["Prioritization"].map(plot_args["names"])
    summary_data["Trace"] = summary_data["Trace"].map({
        "Slow Low-Var": "Slow",
        "Slow High-Var": "Slow",
        "Fast Low-Var": "All",
        "Fast High-Var": "All",
    })


    def update(handle, orig):
        handle.update_from(orig)
        handle.set_linewidth(2.5)
    handler = {plt.Line2D: HandlerLine2D(update_func=update)}

    print("Creating the plots...")
    g = sns.relplot(data=summary_data, 
        kind="line", 
        x="Training Progress",
        y="Value", 
        markers=plot_args["markers"], 
        palette=plot_args["palette"],
        dashes=plot_args["dashes"],
        hue="Prioritization", 
        hue_order=plot_args["names"].values(),
        style="Prioritization", 
        errorbar=("ci", 95),
        ms=6.5,
        aspect=1.45,
        lw=2.25, facet_kws=dict(sharey=False))
    g.set(ylabel="Test Reward")
    sns.move_legend(g, title=None, loc="lower right",
        bbox_to_anchor=(0.65, 0.2), frameon=True, handler_map=handler)
    g.savefig(save_dir / f"mean_test_reward.png")


    slow_data = summary_data[summary_data["Trace"] == "Slow"].copy(deep=True)
    g = sns.relplot(data=slow_data, 
        kind="line", 
        x="Training Progress",
        y="Value", 
        markers=plot_args["markers"], 
        palette=plot_args["palette"],
        dashes=plot_args["dashes"],
        hue="Prioritization", 
        hue_order=plot_args["names"].values(),
        style="Prioritization", 
        errorbar=("ci", 95),
        ms=6.5,
        aspect=1.45,
        lw=2.25, facet_kws=dict(sharey=False))
    g.set(ylabel="Test Reward")
    sns.move_legend(g, title=None, loc="lower right",
        bbox_to_anchor=(0.65, 0.2), frameon=True, handler_map=handler)
    g.savefig(save_dir / f"slow_test_reward.png")




if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--log_dir", 
        help="Location of log directory used in train.py.", required=True)
    parser.add_argument("--skip_processing", action="store_true", 
        default=False, help="Whether or not to skip processing the log_dir")
    parser.add_argument("--plot_args_file", 
        help="Location of file with plotting arguments", 
        default="./plot_args.json"
    )
    parser.add_argument("--save_dir", help="Location to output plots to.", 
        default=".")
    args = parser.parse_args()
    main(args)