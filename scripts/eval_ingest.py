"""
Ingest resolved market rows into monthly CSVs under eval_data/resolved/.

Usage:
  python3 scripts/eval_ingest.py --input path/to/file1.csv [--input path/to/file2.csv ...]

Input CSV required columns (case-sensitive):
  id,asset,title,closed_at,market_prob,internal_prob,outcome,cohort

Behavior:
- Appends rows into eval_data/resolved/YYYYMM.csv based on closed_at UTC month.
- Deduplicates by 'id' within each monthly file.
- Skips invalid rows (missing/invalid probs/outcome).
- Optional auto-commit/push via TB_EVAL_GIT_AUTOCOMMIT/TB_EVAL_GIT_PUSH.
"""
from __future__ import annotations
import argparse
import csv
import os
from datetime import datetime, timezone
from typing import Dict, List

REQUIRED_COLS = [
    "id",
    "asset",
    "title",
    "closed_at",
    "market_prob",
    "internal_prob",
    "outcome",
    "cohort",
]


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", action="append", required=True, help="Input CSV path (repeatable)")
    return ap.parse_args()


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _read_csv_rows(path: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with open(path, newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        missing = [c for c in REQUIRED_COLS if c not in (rdr.fieldnames or [])]
        if missing:
            raise ValueError(f"{path}: missing required columns: {missing}")
        for r in rdr:
            rows.append(r)
    return rows


def _valid_row(r: Dict[str, str]) -> bool:
    try:
        p = float(r.get("internal_prob") or r.get("market_prob") or "")
        if not (0.0 <= p <= 1.0):
            return False
        y = float(r.get("outcome") or "")
        if y not in (0.0, 1.0):
            return False
        # closed_at must be parseable
        _ = datetime.fromisoformat(r["closed_at"].replace("Z", "+00:00"))
        return True
    except Exception:
        return False


def _load_existing_ids(path: str) -> set:
    if not os.path.exists(path):
        return set()
    with open(path, newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        ids = {r.get("id") for r in rdr if r.get("id")}
    return set(ids)


def _append_rows(target_csv: str, rows: List[Dict[str, str]]) -> int:
    new_count = 0
    exists = os.path.exists(target_csv)
    existing_ids = _load_existing_ids(target_csv)
    with open(target_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=REQUIRED_COLS)
        if not exists:
            w.writeheader()
        for r in rows:
            rid = r.get("id")
            if not rid or rid in existing_ids:
                continue
            w.writerow({k: r.get(k, "") for k in REQUIRED_COLS})
            new_count += 1
    return new_count


def main() -> int:
    args = _parse_args()
    base = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(base, ".."))
    out_dir = os.path.join(repo_root, "eval_data", "resolved")
    _ensure_dir(out_dir)

    total_new = 0
    written_paths: List[str] = []

    for p in args.input:
        rows = [r for r in _read_csv_rows(p) if _valid_row(r)]
        # group by YYYYMM derived from closed_at
        groups: Dict[str, List[Dict[str, str]]] = {}
        for r in rows:
            dt = datetime.fromisoformat(r["closed_at"].replace("Z", "+00:00")).astimezone(timezone.utc)
            key = dt.strftime("%Y%m")
            groups.setdefault(key, []).append(r)
        for key, rs in groups.items():
            target_csv = os.path.join(out_dir, f"{key}.csv")
            n = _append_rows(target_csv, rs)
            if n:
                total_new += n
                written_paths.append(target_csv)
                print(f"[eval_ingest] +{n} rows -> {target_csv}")

    # Optional auto-commit/push
    auto = os.getenv("TB_EVAL_GIT_AUTOCOMMIT", "1") == "1"
    do_push = os.getenv("TB_EVAL_GIT_PUSH", "1") == "1"
    if auto and written_paths:
        try:
            import autocommit as ac

            msg = f"eval: ingest {total_new} rows"
            res = ac.auto_commit_and_push(written_paths, extra_message=msg, push_enabled=do_push)
            print(f"[eval_ingest] auto-commit: {res}")
        except Exception as e:
            print(f"[eval_ingest] auto-commit error: {e}")

    if total_new == 0:
        print("[eval_ingest] No new rows ingested.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
