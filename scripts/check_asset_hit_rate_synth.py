#!/usr/bin/env python3
"""
Run asset_hit_rate on the synthetic slice and assert expected metrics.
Exits non-zero on failure for CI regression gating.
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

RUNS_DIR = ROOT / "universe_runs"
BARS_DIR = ROOT / "bars"
MAP_DIR = ROOT / "runs"
SYNTH_FILE = "universe_20250819_synth.json"


def run_cmd(cmd: list[str]) -> tuple[int, str]:
    p = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    out, _ = p.communicate()
    return p.returncode, out


def main() -> int:
    if not (RUNS_DIR / SYNTH_FILE).exists():
        print(f"Synthetic file missing: {RUNS_DIR / SYNTH_FILE}")
        return 2
    cmd = [
        sys.executable,
        "scripts/asset_hit_rate.py",
        "--runs_dir",
        str(RUNS_DIR),
        "--glob",
        SYNTH_FILE,
        "--bars_dir",
        str(BARS_DIR),
        "--runs_map_dir",
        str(MAP_DIR),
        "--debug",
    ]
    code, out = run_cmd(cmd)
    if code != 0:
        print(out)
        print("asset_hit_rate.py exited non-zero")
        return 3
    # Parse JSON from output: last JSON block
    lines = out.strip().splitlines()
    json_start = None
    for i, ln in enumerate(lines):
        if ln.strip().startswith("{"):
            json_start = i
            break
    if json_start is None:
        print(out)
        print("Failed to find JSON in output")
        return 4
    js = "\n".join(lines[json_start:])
    try:
        summary = json.loads(js)
    except Exception as e:
        print(out)
        print(f"Failed to parse JSON: {e}")
        return 5

    overall = summary.get("overall", {})
    n = overall.get("n")
    hr = overall.get("hit_rate")
    diagnostics = summary.get("diagnostics", {})

    # Assertions
    errors = []
    if n != 1:
        errors.append(f"expected overall.n=1, got {n}")
    if hr != 1.0:
        errors.append(f"expected hit_rate=1.0, got {hr}")
    if diagnostics.get("symbols_mapped", 0) < 1:
        errors.append("expected at least 1 symbols_mapped")

    if errors:
        print(json.dumps(summary, indent=2, sort_keys=True))
        print("Regression failures:\n- " + "\n- ".join(errors))
        return 6

    print("Synthetic asset_hit_rate check OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
