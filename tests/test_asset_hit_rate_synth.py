import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run_asset_hit_rate_synth():
    runs_dir = ROOT / "universe_runs"
    bars_dir = ROOT / "bars"
    map_dir = ROOT / "runs"
    synth_file = "universe_20250819_synth.json"

    cmd = [
        sys.executable,
        str(ROOT / "scripts/asset_hit_rate.py"),
        "--runs_dir",
        str(runs_dir),
        "--glob",
        synth_file,
        "--bars_dir",
        str(bars_dir),
        "--runs_map_dir",
        str(map_dir),
        "--debug",
    ]
    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    out, _ = proc.communicate()
    assert proc.returncode == 0, f"asset_hit_rate exited {proc.returncode}:\n{out}"
    # Find JSON block
    lines = out.strip().splitlines()
    json_start = next((i for i, ln in enumerate(lines) if ln.strip().startswith("{")), None)
    assert json_start is not None, f"No JSON found in output:\n{out}"
    summary = json.loads("\n".join(lines[json_start:]))
    return summary


def test_synthetic_hit_rate_ok():
    summary = run_asset_hit_rate_synth()
    overall = summary.get("overall", {})
    assert overall.get("n") == 1
    assert overall.get("hit_rate") == 1.0
    by_hz = summary.get("by_horizon", {})
    assert by_hz.get("1h", {}).get("hit_rate") == 1.0
    diag = summary.get("diagnostics", {})
    assert diag.get("symbols_mapped", 0) >= 1
