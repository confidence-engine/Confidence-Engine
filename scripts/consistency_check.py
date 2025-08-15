#!/usr/bin/env python3
"""
Consistency gate:
- Runs the universe scan twice under deterministic mode and safe profile
- Normalizes results (ignoring timestamps)
- Compares payload summaries and ranking order
- Exits 0 on match; non-zero on drift

Usage:
  python3 scripts/consistency_check.py [--top N] [--max-symbols M] [--config CONFIG]

Env it enforces for both runs:
  TB_DETERMINISTIC=1
  TB_NO_TELEGRAM=1
  TB_UNIVERSE_GIT_AUTOCOMMIT=0
  TB_UNIVERSE_GIT_PUSH=0

Returns non-zero and prints differences if mismatch.
"""
from __future__ import annotations
import os
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Tuple

# Allow importing project modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.scan_universe import run_universe_scan  # type: ignore


def _normalize_payloads(payloads: List[Dict]) -> Tuple[List[Tuple], List[str]]:
    tuples = [
        (
            p.get("symbol"),
            p.get("divergence"),
            p.get("confidence"),
            p.get("volume_z"),
            p.get("action"),
        )
        for p in payloads
    ]
    order = [p.get("symbol") for p in payloads]
    return tuples, order


def main():
    ap = argparse.ArgumentParser(description="Consistency check for universe scan")
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--max-symbols", type=int, default=6)
    ap.add_argument("--config", default="config/universe.yaml")
    args = ap.parse_args()

    # Enforce safe deterministic profile
    os.environ["TB_DETERMINISTIC"] = "1"
    os.environ["TB_NO_TELEGRAM"] = "1"
    os.environ["TB_UNIVERSE_GIT_AUTOCOMMIT"] = "0"
    os.environ["TB_UNIVERSE_GIT_PUSH"] = "0"

    run1 = run_universe_scan(
        config_path=args.config,
        symbols=None,
        top_k=args.top,
        debug=False,
        no_telegram=True,
        version_tag="v3.1",
        fail_fast=True,
        max_symbols=args.max_symbols,
    )
    run2 = run_universe_scan(
        config_path=args.config,
        symbols=None,
        top_k=args.top,
        debug=False,
        no_telegram=True,
        version_tag="v3.1",
        fail_fast=True,
        max_symbols=args.max_symbols,
    )

    p1 = run1.get("payloads", [])
    p2 = run2.get("payloads", [])

    t1, o1 = _normalize_payloads(p1)
    t2, o2 = _normalize_payloads(p2)

    ok = True
    msgs: List[str] = []

    if len(t1) != len(t2):
        ok = False
        msgs.append(f"Payload count differs: {len(t1)} vs {len(t2)}")

    if t1 != t2:
        ok = False
        msgs.append("Payload summaries differ.")

    if o1 != o2:
        ok = False
        msgs.append("Ranking order differs.")

    if ok:
        print("Consistency OK: payloads and ranking match across runs.")
        print(f"Files: {run1.get('universe_file')} vs {run2.get('universe_file')}")
        sys.exit(0)
    else:
        print("Consistency FAILED:")
        for m in msgs:
            print("- ", m)
        # Print short diffs
        print("Top-10 summaries run1:")
        for x in t1[:10]:
            print("  ", x)
        print("Top-10 summaries run2:")
        for x in t2[:10]:
            print("  ", x)
        sys.exit(1)


if __name__ == "__main__":
    main()
