"""
Evaluation runner for Polymarket/system predictions.
- Input: CSV files in eval_data/resolved/*.csv with columns:
  id,asset,title,closed_at,market_prob,internal_prob,outcome,cohort
  where outcome in {0,1}, probs in [0,1].
- Output: eval_runs/<timestamp>/metrics.json, calibration.csv, cohorts.csv

Safe-by-default: if no input files, prints a notice and exits 0.
"""
from __future__ import annotations
import csv
import json
import os
import sys
from datetime import datetime
from glob import glob
from typing import List, Dict

# Ensure repo root is importable when run as a plain script
HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

try:  # prefer absolute import when run as script
    from scripts.eval_metrics import brier_score, log_loss, calibration_curve, cohort_win_rates
except Exception:
    # fallback for package-style execution
    from .eval_metrics import brier_score, log_loss, calibration_curve, cohort_win_rates


def _read_rows(paths: List[str]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for p in paths:
        with open(p, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                rows.append(r)
    return rows


def _to_float(x: str, default: float | None = None) -> float | None:
    try:
        return float(x)
    except Exception:
        return default


def main() -> int:
    base = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(base, ".."))
    input_dir = os.path.join(repo_root, "eval_data", "resolved")
    paths = sorted(glob(os.path.join(input_dir, "*.csv")))

    if not paths:
        print("[eval_runner] No input CSVs found in eval_data/resolved/. Nothing to evaluate.")
        return 0

    rows = _read_rows(paths)
    if not rows:
        print("[eval_runner] No rows found. Nothing to evaluate.")
        return 0

    # Extract fields
    probs = []
    outs = []
    cohorts = []
    for r in rows:
        # prefer internal_prob if available; else fall back to market_prob
        ip = _to_float(r.get("internal_prob", ""))
        mp = _to_float(r.get("market_prob", ""))
        p = ip if ip is not None else mp
        y = _to_float(r.get("outcome", ""))
        c = r.get("cohort") or r.get("asset") or "unknown"
        if p is None or y is None:
            continue
        if not (0.0 <= p <= 1.0 and y in (0.0, 1.0)):
            continue
        probs.append(p)
        outs.append(int(y))
        cohorts.append(str(c))

    if len(probs) == 0:
        print("[eval_runner] No valid rows with probs/outcomes. Nothing to evaluate.")
        return 0

    # Compute metrics
    brier = brier_score(probs, outs)
    nll = log_loss(probs, outs)
    cal = calibration_curve(probs, outs, n_bins=10)
    cohorts_stats = cohort_win_rates(probs, outs, cohorts)

    # Prepare output folder
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(repo_root, "eval_runs", ts)
    os.makedirs(out_dir, exist_ok=True)

    # Write metrics.json
    metrics = {"count": len(probs), "brier": brier, "log_loss": nll}
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Write calibration.csv
    cal_path = os.path.join(out_dir, "calibration.csv")
    with open(cal_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["bin_low", "bin_high", "count", "avg_pred", "frac_pos"])
        w.writeheader()
        for b in cal:
            w.writerow(b)

    # Write cohorts.csv
    coh_path = os.path.join(out_dir, "cohorts.csv")
    with open(coh_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["cohort", "count", "wins", "win_rate", "avg_prob"])
        w.writeheader()
        for k, v in sorted(cohorts_stats.items()):
            row = {"cohort": k}
            row.update(v)
            w.writerow(row)

    print(f"[eval_runner] Wrote metrics to {out_dir}")

    # Optional: auto-commit/push artifacts
    auto = os.getenv("TB_EVAL_GIT_AUTOCOMMIT", "1") == "1"
    do_push = os.getenv("TB_EVAL_GIT_PUSH", "1") == "1"
    include_data = os.getenv("TB_EVAL_GIT_INCLUDE_DATA", "0") == "1"
    if auto:
        try:
            import autocommit as ac

            paths = [out_dir]
            if include_data:
                paths.append(os.path.join(repo_root, "eval_data", "resolved"))
            msg = f"eval: metrics {ts}"
            res = ac.auto_commit_and_push(paths, extra_message=msg, push_enabled=do_push)
            print(f"[eval_runner] auto-commit: {res}")
        except Exception as e:
            print(f"[eval_runner] auto-commit error: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
