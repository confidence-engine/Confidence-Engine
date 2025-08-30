import os

from backtester.ml_baseline import run_ml_baseline


def test_run_ml_baseline(tmp_path):
    out_dir = run_ml_baseline(bars_dir="bars", out_root=str(tmp_path))
    assert os.path.isdir(out_dir)
    assert os.path.isfile(os.path.join(out_dir, "model.pt"))
    assert os.path.isfile(os.path.join(out_dir, "features.csv"))
    assert os.path.isfile(os.path.join(out_dir, "metrics.json"))
