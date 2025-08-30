import os

from backtester.walk_forward import run_walk_forward


def test_run_walk_forward(tmp_path):
    out_dir = run_walk_forward(
        bars_dir="bars",
        out_root=str(tmp_path),
        train_days=3,
        test_days=2,
        n_splits=2,
        ema_fast=10,
        ema_slow=20,
        trend_ema=50,
        trend_rule="1h",
        tp_pct=0.02,
        sl_pct=0.01,
        cooldown_sec=600,
        risk_frac=0.01,
    )
    assert os.path.isdir(out_dir)
    # summary_all.csv is optional if no rows, but the directory should exist
