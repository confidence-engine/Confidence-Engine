#!/usr/bin/env python3
"""
Compare the latest hit-rate with the previous entry in eval_runs/hit_rate_trend.csv.
Logs a warning if drop exceeds TB_HITRATE_REG_THRESH (default 0.05 = 5 percentage points).
Does not fail with non-zero exit to keep nightly safe.
"""
import csv
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TREND_CSV = ROOT / "eval_runs" / "hit_rate_trend.csv"


def read_last_two(path: Path):
    if not path.exists():
        print(f"No trend file found at {path}")
        return None
    rows = []
    with path.open() as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    if len(rows) < 2:
        print("Trend has fewer than 2 rows; nothing to compare yet.")
        return None
    return rows[-2], rows[-1]


def main() -> int:
    pair = read_last_two(TREND_CSV)
    if not pair:
        return 0
    prev, cur = pair
    try:
        prev_hr = float(prev.get("overall_hit_rate") or 0)
        cur_hr = float(cur.get("overall_hit_rate") or 0)
    except Exception:
        print("Could not parse hit rates from trend CSV")
        return 0
    thresh = float(os.getenv("TB_HITRATE_REG_THRESH", "0.05"))
    delta = cur_hr - prev_hr
    if delta < -thresh:
        print(f"WARNING: Hit-rate regression detected: {prev_hr:.3f} -> {cur_hr:.3f} (Δ={delta:.3f} < -{thresh:.3f})")
    else:
        print(f"Hit-rate stable/improved: {prev_hr:.3f} -> {cur_hr:.3f} (Δ={delta:.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
