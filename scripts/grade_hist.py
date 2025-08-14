#!/usr/bin/env python3
"""
Grade histogram CLI for Tracer Bullet.

Usage:
  python3 scripts/grade_hist.py [--file PATH] [--per-tf] [--aggregate] [--symbols BTC,ETH]

- If --file is omitted, the latest universe_runs/*.json is used.
- Prints overall grade distribution and (optionally) per-timeframe and aggregate micro-grade distributions.
- Read-only; does not send messages or modify artifacts.
"""
import argparse
import glob
import json
import os
import sys
import pathlib
from typing import Dict, Any, List

try:
    # When executed inside the package context
    from .evidence_lines import compute_setup_grade, compute_setup_grade_for_tf  # type: ignore
except Exception:
    # When run as a standalone script: python3 scripts/grade_hist.py
    ROOT = pathlib.Path(__file__).resolve().parent.parent
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from scripts.evidence_lines import compute_setup_grade, compute_setup_grade_for_tf  # type: ignore


def _latest_universe() -> str:
    paths = sorted(glob.glob('universe_runs/*.json'), key=os.path.getmtime)
    if not paths:
        raise SystemExit('No universe_runs/*.json files found')
    return paths[-1]


def _load_assets(path: str) -> List[Dict[str, Any]]:
    with open(path, 'r') as f:
        data = json.load(f)
    if isinstance(data, dict):
        if isinstance(data.get('assets'), list):
            return data['assets']
        if isinstance(data.get('digest'), dict) and isinstance(data['digest'].get('assets'), list):
            return data['digest']['assets']
    return []


def _bump(d: Dict[str, int], k: str):
    d[k] = d.get(k, 0) + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--file', help='Universe JSON path (default: latest in universe_runs/)')
    ap.add_argument('--per-tf', action='store_true', help='Show per-timeframe micro-grade counts')
    ap.add_argument('--aggregate', action='store_true', help='Show aggregate micro-grade counts across TFs')
    ap.add_argument('--symbols', help='Comma-separated list of symbols to include (default: all)')
    args = ap.parse_args()

    path = args.file or _latest_universe()
    assets = _load_assets(path)
    if not assets:
        print(f'Using {path}\nAssets found: 0')
        return

    symbols = None
    if args.symbols:
        symbols = {s.strip().upper() for s in args.symbols.split(',') if s.strip()}

    overall: Dict[str, int] = {}
    per_tf: Dict[str, int] = {}

    for a in assets:
        sym = str(a.get('symbol') or '').upper()
        if symbols and sym not in symbols:
            continue
        try:
            g = compute_setup_grade(a)
        except Exception:
            g = 'ERR'
        _bump(overall, g)
        plan = a.get('plan') or {}
        for tf, p in plan.items():
            try:
                mg = compute_setup_grade_for_tf(a, tf, p)
            except Exception:
                mg = 'ERR'
            _bump(per_tf, f'{tf}:{mg}')

    print(f'Using {path}')
    print('\nOverall grade distribution:')
    for k in sorted(overall):
        print(f'  {k}: {overall[k]}')

    if args.per_tf:
        print('\nPer-timeframe micro-grade distribution:')
        for k in sorted(per_tf):
            print(f'  {k}: {per_tf[k]}')

    if args.aggregate:
        agg: Dict[str, int] = {}
        for k, v in per_tf.items():
            g = k.split(':', 1)[1]
            _bump(agg, g)
        print('\nAggregate by grade (all TFs combined):')
        for k in sorted(agg):
            print(f'  {k}: {agg[k]}')


if __name__ == '__main__':
    main()
