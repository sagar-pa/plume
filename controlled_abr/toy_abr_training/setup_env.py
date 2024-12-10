from toy_abr_gym.trace_loader import extract_and_save_trace_features
from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--save_trace_features", action="store_true", default=False)
    args = parser.parse_args()
    if args.save_trace_features:
        for dataset_name in ["majority_fast", "balanced", "majority_slow"]:
            extract_and_save_trace_features(dataset_name)