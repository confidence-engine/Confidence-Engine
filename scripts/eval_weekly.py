"""
Weekly evaluation wrapper.

Usage:
  python3 scripts/eval_weekly.py

Behavior:
- Delegates to eval_runner.main() to compute metrics and write outputs to eval_runs/<ts>/.
- Honors TB_EVAL_GIT_AUTOCOMMIT / TB_EVAL_GIT_PUSH / TB_EVAL_GIT_INCLUDE_DATA env flags.
- Safe: no network calls; only reads local CSVs and writes local outputs and git commits/pushes per flags.
"""
from __future__ import annotations
import sys

# Prefer absolute import when run as script
try:
    from scripts import eval_runner
except Exception:  # package-style fallback
    from . import eval_runner  # type: ignore


def main() -> int:
    return eval_runner.main()


if __name__ == "__main__":
    sys.exit(main())
