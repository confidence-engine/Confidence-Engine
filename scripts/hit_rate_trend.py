#!/usr/bin/env python3
"""
Append a one-line hit-rate summary to eval_runs/hit_rate_trend.csv for nightly tracking.
Safe: no external sends, local file writes only.
"""
import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "universe_runs"
BARS_DIR = ROOT / "bars"
MAP_DIR = ROOT / "runs"
TREND_CSV = ROOT / "eval_runs" / "hit_rate_trend.csv"


def run_asset_hit_rate() -> dict:
    cmd = [
        sys.executable,
        str(ROOT / "scripts/asset_hit_rate.py"),
        "--runs_dir",
        str(RUNS_DIR),
        "--bars_dir",
        str(BARS_DIR),
        "--runs_map_dir",
        str(MAP_DIR),
        "--debug",
        "--failures_csv",
        str(ROOT / "eval_runs" / "hit_rate_failures.csv"),
    ]
    p = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    out, _ = p.communicate()
    if p.returncode != 0:
        raise SystemExit(f"asset_hit_rate failed: {p.returncode}\n{out}")
    # Parse JSON block from output
    lines = out.strip().splitlines()
    json_start = next((i for i, ln in enumerate(lines) if ln.strip().startswith("{")), None)
    if json_start is None:
        raise SystemExit(f"No JSON found in output:\n{out}")
    return json.loads("\n".join(lines[json_start:]))


def append_trend(summary: dict) -> None:
    TREND_CSV.parent.mkdir(parents=True, exist_ok=True)
    exists = TREND_CSV.exists()
    with TREND_CSV.open("a", newline="") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow([
                "ts_utc",
                "files",
                "items_examined",
                "items_with_prediction",
                "items_with_outcome",
                "overall_n",
                "overall_hits",
                "overall_hit_rate",
            ])
        ov = summary.get("overall", {})
        w.writerow([
            datetime.now(timezone.utc).isoformat(),
            summary.get("files"),
            summary.get("items_examined"),
            summary.get("items_with_prediction"),
            summary.get("items_with_outcome"),
            ov.get("n"),
            ov.get("hits"),
            ov.get("hit_rate"),
        ])


def main() -> int:
    # Ensure safe env defaults
    os.environ.setdefault("TB_NO_TELEGRAM", "1")
    os.environ.setdefault("TB_NO_DISCORD", "1")
    summary = run_asset_hit_rate()
    append_trend(summary)
    print("Trend updated:", TREND_CSV)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
