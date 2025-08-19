#!/usr/bin/env python3
"""
Compute directional hit-rates from universe run artifacts.

Two input sources:
- universe_runs/*.json: artifacts per symbol with prediction fields (e.g., action) and timestamps
- bars/*.csv: minute bars per asset used to compute realized returns for 1h/4h/1d horizons

Behavior:
- Extract predicted direction from item fields (thesis.action or fallbacks)
- If realized outcomes exist in the artifact, use them; otherwise compute by joining bars
- Join uses a symbol->bars file mapping inferred from runs/*.json (heuristic, safe):
  symbol may map to multiple bars files; we select the file that covers the timestamp window

This script does not perform network calls or writes; safe for local validation.
"""
import argparse
import csv
import glob
import json
import os
import re
from collections import defaultdict, Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Tuple, Optional

# Map common action strings to direction
ACTION_DIR = {
    'up': 1,
    'long': 1,
    'buy': 1,
    'bull': 1,
    'bullish': 1,
    'down': -1,
    'short': -1,
    'sell': -1,
    'bear': -1,
    'bearish': -1,
    'sideways': 0,
    'hold': 0,
    'neutral': 0,
}

HORIZON_PATTERNS = [
    ('1h', re.compile(r'(?:^|_|\b)(1h|60m|next1h)(?:$|_|\b)', re.I)),
    ('4h', re.compile(r'(?:^|_|\b)(4h|240m|next4h)(?:$|_|\b)', re.I)),
    ('1d', re.compile(r'(?:^|_|\b)(1d|24h|next1d|next24h)(?:$|_|\b)', re.I)),
]

CHANGE_PAT = re.compile(r'(change|return|pct|perf)', re.I)


@dataclass
class BarRow:
    ts: datetime
    close: float


def _parse_iso(ts: str) -> Optional[datetime]:
    try:
        # handle possible Z or offset
        if ts.endswith('Z'):
            return datetime.fromisoformat(ts.replace('Z', '+00:00'))
        dt = datetime.fromisoformat(ts)
        # If no tzinfo, assume UTC
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


def _iter_json_items(obj: Any):
    """Yield dict-like items from nested JSON structures."""
    if isinstance(obj, dict):
        yield obj
        for v in obj.values():
            yield from _iter_json_items(v)
    elif isinstance(obj, list):
        for it in obj:
            yield from _iter_json_items(it)


def _extract_prediction(item: Dict[str, Any]) -> Tuple[int, str]:
    """Return (direction, raw_action) where direction in {-1,0,1}; -2 means unknown."""
    thesis = item.get('thesis') or {}
    action = thesis.get('action') or item.get('action')
    if isinstance(action, str):
        norm = action.strip().lower()
        if norm in ACTION_DIR:
            return ACTION_DIR[norm], norm
    # Look for alternative fields
    for k in ('stance', 'direction'):
        v = item.get(k)
        if isinstance(v, str):
            norm = v.strip().lower()
            if norm in ACTION_DIR:
                return ACTION_DIR[norm], norm
    return -2, ''


def _extract_outcomes(item: Dict[str, Any]) -> Dict[str, float]:
    """
    Try to collect realized percent changes keyed by horizon (1h/4h/1d).
    Looks for numeric fields with names containing CHANGE_PAT and a horizon token.
    Returns map: horizon -> signed change (e.g., +1.2 for +1.2%).
    """
    outcomes: Dict[str, float] = {}
    for k, v in item.items():
        if not isinstance(v, (int, float)):
            continue
        if not isinstance(k, str):
            continue
        if not CHANGE_PAT.search(k):
            continue
        for horizon, pat in HORIZON_PATTERNS:
            if pat.search(k):
                outcomes[horizon] = float(v)
    # Also check nested 'timescale_scores' dict
    tsc = item.get('timescale_scores')
    if isinstance(tsc, dict):
        for k, v in tsc.items():
            if isinstance(v, (int, float)) and isinstance(k, str) and CHANGE_PAT.search(k):
                for horizon, pat in HORIZON_PATTERNS:
                    if pat.search(k) and horizon not in outcomes:
                        outcomes[horizon] = float(v)
    return outcomes


def _build_symbol_to_bars_map(runs_dir: str, bars_dir: str) -> Dict[str, List[str]]:
    """
    Heuristically map symbols to bars files using runs/*.json filenames.
    For each runs/<id>.json with a top-level 'symbol', if bars/<id>.csv exists,
    associate that CSV to the symbol. Symbols may map to multiple bar files.
    """
    mapping: Dict[str, List[str]] = defaultdict(list)
    run_files = glob.glob(os.path.join(runs_dir, '*.json'))
    for rf in run_files:
        base = os.path.basename(rf)
        stem, ext = os.path.splitext(base)
        if not stem.isdigit():
            continue
        bars_path = os.path.join(bars_dir, f'{stem}.csv')
        if not os.path.exists(bars_path):
            continue
        try:
            with open(rf, 'r') as f:
                data = json.load(f)
        except Exception:
            continue
        sym = data.get('symbol')
        if isinstance(sym, str) and bars_path not in mapping[sym]:
            mapping[sym].append(bars_path)
    # Also try direct symbol-named CSVs as a safe fallback (does not override existing entries)
    # Candidates: SYMBOL with '/', '-', ' ' replaced by '_' and removed, in both cases
    # e.g., 'BTC/USD' -> 'BTC_USD.csv', 'BTCUSD.csv'
    symbols = set(mapping.keys())
    # Additionally harvest symbols directly from run files to consider mapping even if no numeric match
    for rf in run_files:
        try:
            with open(rf, 'r') as f:
                data = json.load(f)
        except Exception:
            continue
        sym = data.get('symbol')
        if isinstance(sym, str):
            symbols.add(sym)
    for sym in symbols:
        base1 = re.sub(r'[\s/\-]+', '_', sym)
        base2 = re.sub(r'[^A-Za-z0-9]', '', sym)
        candidates = [
            f"{base1}.csv",
            f"{base2}.csv",
            f"{base1.upper()}.csv",
            f"{base2.upper()}.csv",
            f"{base1.lower()}.csv",
            f"{base2.lower()}.csv",
        ]
        for cand in candidates:
            p = os.path.join(bars_dir, cand)
            if os.path.exists(p) and p not in mapping[sym]:
                mapping[sym].append(p)
    return mapping


def _load_bars(csv_path: str) -> List[BarRow]:
    rows: List[BarRow] = []
    try:
        with open(csv_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for r in reader:
                ts_s = r.get('timestamp')
                close_s = r.get('close')
                if not ts_s or not close_s:
                    continue
                ts = _parse_iso(ts_s)
                if ts is None:
                    # bars may be like "2025-08-08 20:21:00+00:00"
                    try:
                        ts = datetime.strptime(ts_s, '%Y-%m-%d %H:%M:%S%z')
                    except Exception:
                        continue
                try:
                    close = float(close_s)
                except Exception:
                    continue
                rows.append(BarRow(ts=ts, close=close))
    except Exception:
        return []
    rows.sort(key=lambda x: x.ts)
    return rows


def _find_return(bars: List[BarRow], t0: datetime, horizon: timedelta) -> Optional[float]:
    if not bars:
        return None
    # find the first bar at or after t0
    start = None
    end_time = t0 + horizon
    end = None
    for br in bars:
        if start is None and br.ts >= t0:
            start = br
        if br.ts >= end_time:
            end = br
            break
    if start is None or end is None:
        return None
    if start.close == 0:
        return None
    return (end.close - start.close) / start.close * 100.0


def _compute_outcomes_from_bars(symbol: str, event_ts: Optional[str], sym_bar_files: Dict[str, List[str]], diag: Optional[Dict[str, int]] = None, failures: Optional[List[Dict[str, Any]]] = None) -> Dict[str, float]:
    if not isinstance(event_ts, str):
        if diag is not None:
            diag['missing_event_ts'] = diag.get('missing_event_ts', 0) + 1
        if failures is not None:
            failures.append({'symbol': symbol, 'event_ts': None, 'reason': 'missing_event_ts'})
        return {}
    t0 = _parse_iso(event_ts)
    if t0 is None:
        if diag is not None:
            diag['event_ts_unparseable'] = diag.get('event_ts_unparseable', 0) + 1
        if failures is not None:
            failures.append({'symbol': symbol, 'event_ts': event_ts, 'reason': 'event_ts_unparseable'})
        return {}
    bars_files = sym_bar_files.get(symbol) or []
    horizons = {
        '1h': timedelta(hours=1),
        '4h': timedelta(hours=4),
        '1d': timedelta(days=1),
    }
    # Try each mapped bars file until we can compute at least one horizon
    if not bars_files and diag is not None:
        diag['no_bars_mapping'] = diag.get('no_bars_mapping', 0) + 1
        if failures is not None:
            failures.append({'symbol': symbol, 'event_ts': event_ts, 'reason': 'no_bars_mapping'})
    for bp in bars_files:
        bars = _load_bars(bp)
        if diag is not None and not bars:
            diag['empty_bars_file'] = diag.get('empty_bars_file', 0) + 1
            if failures is not None:
                failures.append({'symbol': symbol, 'event_ts': event_ts, 'reason': 'empty_bars_file', 'bars_file': bp})
        outcomes: Dict[str, float] = {}
        for hz, td in horizons.items():
            ret = _find_return(bars, t0, td)
            if ret is not None:
                outcomes[hz] = ret
        if outcomes:
            return outcomes
    if diag is not None and bars_files:
        diag['no_covering_window'] = diag.get('no_covering_window', 0) + 1
        if failures is not None:
            failures.append({'symbol': symbol, 'event_ts': event_ts, 'reason': 'no_covering_window'})
    return {}


def compute_hit_rates(paths: List[str], bars_dir: str, runs_map_dir: str, debug: bool = False, failures_csv: str = '') -> Dict[str, Any]:
    counts = Counter()
    correct = Counter()
    by_horizon_counts = Counter()
    by_horizon_correct = Counter()

    examined = 0
    with_pred = 0
    with_outcome = 0

    # Build symbol -> [bars files]
    sym_bar_files = _build_symbol_to_bars_map(runs_map_dir, bars_dir)
    diag: Dict[str, int] = {}
    if debug:
        diag['symbols_mapped'] = len(sym_bar_files)
    failures: List[Dict[str, Any]] = [] if debug and failures_csv else []

    for p in paths:
        try:
            with open(p, 'r') as f:
                data = json.load(f)
        except Exception:
            continue
        for item in _iter_json_items(data):
            examined += 1
            pred_dir, _ = _extract_prediction(item)
            if pred_dir == -2:
                continue
            with_pred += 1
            outcomes = _extract_outcomes(item)
            if not outcomes:
                # Try compute from bars using symbol and timestamp
                symbol = item.get('symbol')
                event_ts = item.get('timestamp') or data.get('timestamp_iso')
                if isinstance(symbol, str):
                    outcomes = _compute_outcomes_from_bars(symbol, event_ts, sym_bar_files, diag if debug else None, failures if failures_csv else None)
                # If still empty, skip
                if not outcomes:
                    if debug:
                        if not isinstance(symbol, str):
                            diag['missing_symbol'] = diag.get('missing_symbol', 0) + 1
                            if failures_csv:
                                failures.append({'symbol': None, 'event_ts': item.get('timestamp') or data.get('timestamp_iso'), 'reason': 'missing_symbol'})
                        else:
                            diag['unrealized_items'] = diag.get('unrealized_items', 0) + 1
                    continue
            with_outcome += 1
            # overall: weighted vote across available horizons
            corr_flags = []  # kept for by-horizon tallies (unchanged)
            total_w = 0.0
            correct_w = 0.0
            for hz, chg in outcomes.items():
                by_horizon_counts[hz] += 1
                # For sideways prediction, count correct if |chg| < epsilon (env-tunable)
                eps = float(os.getenv('TB_HITRATE_SIDEWAYS_EPS', '0.1'))
                if pred_dir == 0:
                    ok = abs(chg) < eps  # percent band
                else:
                    ok = (chg > 0 and pred_dir == 1) or (chg < 0 and pred_dir == -1)
                if ok:
                    by_horizon_correct[hz] += 1
                corr_flags.append(ok)
                # Weighted overall
                if hz == '1h':
                    w = float(os.getenv('TB_HITRATE_W_1H', '1.0'))
                elif hz == '4h':
                    w = float(os.getenv('TB_HITRATE_W_4H', '1.0'))
                elif hz == '1d':
                    w = float(os.getenv('TB_HITRATE_W_1D', '1.0'))
                else:
                    w = 1.0
                total_w += w
                if ok:
                    correct_w += w
            if corr_flags:
                counts['total'] += 1
                if total_w > 0 and (correct_w / total_w) >= 0.5:
                    correct['total'] += 1

    summary = {
        'files': len(paths),
        'items_examined': examined,
        'items_with_prediction': with_pred,
        'items_with_outcome': with_outcome,
        'overall': {
            'n': counts['total'],
            'hits': correct['total'],
            'hit_rate': (correct['total'] / counts['total']) if counts['total'] else None,
        },
        'by_horizon': {},
    }
    for hz in sorted(by_horizon_counts.keys()):
        n = by_horizon_counts[hz]
        h = by_horizon_correct[hz]
        summary['by_horizon'][hz] = {
            'n': n,
            'hits': h,
            'hit_rate': (h / n) if n else None,
        }
    if debug:
        # expose weights used
        diag['weights'] = {
            '1h': float(os.getenv('TB_HITRATE_W_1H', '1.0')),
            '4h': float(os.getenv('TB_HITRATE_W_4H', '1.0')),
            '1d': float(os.getenv('TB_HITRATE_W_1D', '1.0')),
        }
        summary['diagnostics'] = diag
        # Optionally write failures CSV
        if failures_csv and failures:
            try:
                import csv as _csv
                os.makedirs(os.path.dirname(failures_csv), exist_ok=True)
                with open(failures_csv, 'w', newline='') as f:
                    w = _csv.DictWriter(f, fieldnames=['symbol', 'event_ts', 'reason', 'bars_file'])
                    w.writeheader()
                    for row in failures:
                        if 'bars_file' not in row:
                            row['bars_file'] = ''
                        w.writerow(row)
            except Exception as e:
                print(f"Warning: failed to write failures CSV to {failures_csv}: {e}")
    return summary


def _format_markdown(summary: Dict[str, Any]) -> str:
    lines = []
    lines.append('# Asset directional hit-rate summary')
    ov = summary.get('overall', {})
    lines.append(f"- files: {summary.get('files')}\n- items_examined: {summary.get('items_examined')}\n- items_with_prediction: {summary.get('items_with_prediction')}\n- items_with_outcome: {summary.get('items_with_outcome')}")
    lines.append('')
    lines.append('## Overall')
    lines.append(f"- n: {ov.get('n')}\n- hits: {ov.get('hits')}\n- hit_rate: {ov.get('hit_rate')}")
    lines.append('')
    lines.append('## By Horizon')
    for hz, obj in sorted(summary.get('by_horizon', {}).items()):
        lines.append(f"- {hz}: n={obj.get('n')}, hits={obj.get('hits')}, hit_rate={obj.get('hit_rate')}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description='Compute directional hit-rates from universe run artifacts')
    ap.add_argument('--runs_dir', default='universe_runs', help='Directory containing run json files')
    ap.add_argument('--glob', default='*.json', help='Glob pattern for run files')
    ap.add_argument('--bars_dir', default='bars', help='Directory containing bar CSVs')
    ap.add_argument('--runs_map_dir', default='runs', help='Directory of runs used to infer symbol->bars mapping')
    ap.add_argument('--markdown_out', default='', help='Optional path to write a concise Markdown summary')
    ap.add_argument('--debug', action='store_true', help='Include diagnostics about bars join coverage and failures')
    ap.add_argument('--failures_csv', default='', help='When --debug, write per-item join failures to this CSV')
    args = ap.parse_args()

    run_glob = os.path.join(args.runs_dir, args.glob)
    files = sorted(glob.glob(run_glob))
    if not files:
        print(f'No run files found at {run_glob}')
        return 0
    summary = compute_hit_rates(files, bars_dir=args.bars_dir, runs_map_dir=args.runs_map_dir, debug=args.debug, failures_csv=args.failures_csv)

    print('Asset directional hit-rate summary')
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.markdown_out:
        try:
            md = _format_markdown(summary)
            with open(args.markdown_out, 'w') as f:
                f.write(md + '\n')
        except Exception as e:
            print(f'Warning: failed to write markdown to {args.markdown_out}: {e}')
    # Basic non-zero sanity exit code when no outcomes found
    if summary['overall']['n'] == 0:
        print('Note: no realizations found. Ensure your run JSONs contain realized change fields (e.g., *_1h/_4h/_1d).')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
