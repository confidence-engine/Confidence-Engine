#!/usr/bin/env python3
"""
Safe QA runner: executes a non-destructive validation suite end-to-end.

Checks (conditionally, only if tools/files are available):
1) Unit tests via pytest (if installed)
2) Deterministic consistency gate (scripts/consistency_check.py)
3) Polymarket evaluation runner (scripts/eval_runner.py)
4) Asset directional hit-rate (scripts/asset_hit_rate.py)

All with safe environment toggles: no Telegram/Discord sends; no Git autocommit/push.
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / 'scripts'

SAFE_ENV = {
    'TB_NO_TELEGRAM': '1',
    'TB_NO_DISCORD': '1',
    'TB_UNIVERSE_GIT_AUTOCOMMIT': '0',
    'TB_UNIVERSE_GIT_PUSH': '0',
    'TB_AUTOCOMMIT_DOCS': '0',
}


def _merge_env():
    env = os.environ.copy()
    env.update(SAFE_ENV)
    return env


def _run_cmd(cmd, cwd=None):
    try:
        res = subprocess.run(
            cmd, cwd=str(cwd) if cwd else None, env=_merge_env(),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        return res.returncode, res.stdout
    except FileNotFoundError as e:
        return 127, f'FileNotFoundError: {e}'


def run_pytest():
    if shutil.which('pytest') is None:
        return {
            'skipped': True,
            'reason': 'pytest not installed',
        }
    code, out = _run_cmd(['pytest', '-q', 'tests'], cwd=ROOT)
    return {
        'skipped': False,
        'returncode': code,
        'ok': code == 0,
        'output_tail': '\n'.join(out.splitlines()[-50:]),
    }


def run_consistency_gate():
    script = SCRIPTS / 'consistency_check.py'
    if not script.exists():
        return {'skipped': True, 'reason': 'scripts/consistency_check.py not found'}
    code, out = _run_cmd([
        sys.executable, str(script), '--config', 'config/universe.yaml', '--top', '10'
    ], cwd=ROOT)
    return {
        'skipped': False,
        'returncode': code,
        'ok': code == 0,
        'output_tail': '\n'.join(out.splitlines()[-50:]),
    }


def run_eval_runner():
    script = SCRIPTS / 'eval_runner.py'
    if not script.exists():
        return {'skipped': True, 'reason': 'scripts/eval_runner.py not found'}
    code, out = _run_cmd([sys.executable, str(script)], cwd=ROOT)
    return {
        'skipped': False,
        'returncode': code,
        'ok': code == 0,
        'output_tail': '\n'.join(out.splitlines()[-50:]),
    }


def run_asset_hit_rate():
    script = SCRIPTS / 'asset_hit_rate.py'
    if not script.exists():
        return {'skipped': True, 'reason': 'scripts/asset_hit_rate.py not found'}
    code, out = _run_cmd([sys.executable, str(script)], cwd=ROOT)
    # Try to parse the printed JSON summary
    summary = None
    try:
        # find last JSON object in output
        lines = out.strip().splitlines()
        json_start = None
        for i, ln in enumerate(lines):
            if ln.strip().startswith('{'):
                json_start = i
                break
        if json_start is not None:
            summary = json.loads('\n'.join(lines[json_start:]))
    except Exception:
        summary = None
    return {
        'skipped': False,
        'returncode': code,
        'ok': code == 0,
        'summary': summary,
        'output_tail': '\n'.join(out.splitlines()[-50:]),
    }


def main():
    results = {
        'env': SAFE_ENV,
        'steps': {}
    }
    results['steps']['pytest'] = run_pytest()
    results['steps']['consistency_gate'] = run_consistency_gate()
    results['steps']['eval_runner'] = run_eval_runner()
    results['steps']['asset_hit_rate'] = run_asset_hit_rate()

    print(json.dumps(results, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
