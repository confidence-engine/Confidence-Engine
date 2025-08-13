"""
Lightweight test runner for eval metrics without pytest dependency.
Runs assertion-based tests from tests/test_eval_metrics.py by invoking functions.
Exits non-zero on failure.
"""
from __future__ import annotations
import sys
import os

# Ensure repo root is on sys.path so 'tests' is importable
HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tests.test_eval_metrics import (  # type: ignore
    test_brier_score_basic,
    test_log_loss_clipping,
    test_calibration_curve_bins,
    test_cohort_win_rates,
)


def main() -> int:
    try:
        test_brier_score_basic()
        test_log_loss_clipping()
        test_calibration_curve_bins()
        test_cohort_win_rates()
    except AssertionError as e:
        print(f"[run_eval_tests] Assertion failed: {e}")
        return 1
    except Exception as e:
        print(f"[run_eval_tests] Error: {e}")
        return 2
    print("[run_eval_tests] All eval metric tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
