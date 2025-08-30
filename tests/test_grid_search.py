import os

from backtester.grid_search import run_grid


def test_run_grid(tmp_path):
    out_dir = run_grid(
        bars_dir="bars",
        out_root=str(tmp_path),
        ema_fast_list=[8],
        ema_slow_list=[21],
        trend_ema_list=[50],
        sentiment_cutoff_list=[0.5],
        tp_pct_list=[0.03],
        sl_pct_list=[0.02],
        cooldown_list=[1800],
        risk_frac_list=[0.01],
        trend_rule="1h",
    )
    assert os.path.isdir(out_dir)
    assert os.path.isfile(os.path.join(out_dir, "results.csv"))
    assert os.path.isfile(os.path.join(out_dir, "top20.csv"))
