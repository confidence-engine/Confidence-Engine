#!/usr/bin/env python3
import argparse
import os
import shutil
from datetime import datetime

from backtester.ml_baseline import run_ml_baseline


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bars_dir", default="bars", help="Directory of input bars CSVs")
    p.add_argument("--out_root", default="eval_runs/ml", help="Where to write timestamped ML runs")
    p.add_argument(
        "--link_dir",
        default="eval_runs/ml/latest",
        help="Directory to place latest copies (model.pt, features.csv, metrics.json)",
    )
    args = p.parse_args()

    run_dir = run_ml_baseline(args.bars_dir, args.out_root)
    model_src = os.path.join(run_dir, "model.pt")
    feats_src = os.path.join(run_dir, "features.csv")
    metrics_src = os.path.join(run_dir, "metrics.json")

    os.makedirs(args.link_dir, exist_ok=True)
    shutil.copy2(model_src, os.path.join(args.link_dir, "model.pt"))
    if os.path.isfile(feats_src):
        shutil.copy2(feats_src, os.path.join(args.link_dir, "features.csv"))
    if os.path.isfile(metrics_src):
        shutil.copy2(metrics_src, os.path.join(args.link_dir, "metrics.json"))

    print(run_dir)


if __name__ == "__main__":
    main()
