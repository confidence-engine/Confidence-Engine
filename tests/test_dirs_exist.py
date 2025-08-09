import os
from scripts.preflight import ensure_dirs


def test_runs_and_bars_dirs_exist(tmp_path, monkeypatch):
    # Run in temp cwd
    monkeypatch.chdir(tmp_path)
    ok, created = ensure_dirs()
    assert ok
    assert os.path.isdir(tmp_path / "runs")
    assert os.path.isdir(tmp_path / "bars")


