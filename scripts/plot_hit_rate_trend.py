#!/usr/bin/env python3
"""
Plot hit_rate_trend.csv into a PNG for quick visual progress.

Inputs:
  - eval_runs/hit_rate_trend.csv (assumed to have at least date,today_hit_rate columns)
Outputs:
  - eval_runs/hit_rate_trend.png

Usage:
  python scripts/plot_hit_rate_trend.py --csv eval_runs/hit_rate_trend.csv --out eval_runs/hit_rate_trend.png

Notes:
  - Designed to be CI-friendly: exits 0 even if CSV missing or empty, with a warning.
  - Requires matplotlib; see requirements.txt.
"""
import argparse
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="eval_runs/hit_rate_trend.csv")
    parser.add_argument("--out", default="eval_runs/hit_rate_trend.png")
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"[plot_hit_rate_trend] WARNING: CSV not found: {args.csv}. Skipping plot.")
        return 0

    try:
        df = pd.read_csv(args.csv)
    except Exception as e:
        print(f"[plot_hit_rate_trend] WARNING: Failed to read CSV ({args.csv}): {e}. Skipping plot.")
        return 0

    # Heuristic: look for a column that represents overall hit rate
    # Try common names in descending priority
    candidates = [
        "overall_hit_rate",  # explicit
        "hit_rate",          # generic
        "today_hit_rate",    # trend builder may use this
        "overall",           # fallback generic
    ]
    y_col = None
    for c in candidates:
        if c in df.columns:
            y_col = c
            break

    # fallback: search for first float-like column except timestamp-like columns
    if y_col is None:
        for c in df.columns:
            if c.lower() in ("date", "timestamp", "ts", "run_id"):  # skip typical index cols
                continue
            if pd.api.types.is_numeric_dtype(df[c]):
                y_col = c
                break

    if y_col is None or df.empty:
        print(f"[plot_hit_rate_trend] WARNING: No suitable hit-rate column found or empty data. Columns: {list(df.columns)}. Skipping plot.")
        return 0

    # x-axis: prefer 'date' or 'ts' if present; else use index
    x_col = None
    for c in ("date", "ts", "timestamp"):
        if c in df.columns:
            x_col = c
            break

    plt.figure(figsize=(8, 4.5), dpi=150)
    if x_col:
        # attempt to parse date for nicer ticks
        x = pd.to_datetime(df[x_col], errors="coerce")
        plt.plot(x, df[y_col], marker="o", linewidth=1.5)
        plt.gcf().autofmt_xdate()
        plt.xlabel("Date")
    else:
        plt.plot(df[y_col].values, marker="o", linewidth=1.5)
        plt.xlabel("Run #")

    plt.title("Directional Hit-Rate Trend")
    plt.ylabel("Hit rate")
    plt.ylim(0.0, 1.0)
    plt.grid(True, linestyle=":", alpha=0.5)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.out)
    print(f"[plot_hit_rate_trend] Wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
