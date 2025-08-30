import os

import pandas as pd

from backtester.core import DataLoader, Simulator
from backtester.strategies import HybridEMAStrategy, HybridParams


def test_backtester_runs(tmp_path):
    # Use real bars directory
    bars_dir = "bars/"
    loader = DataLoader(bars_dir)
    bars = loader.load()
    assert not bars.empty

    strat = HybridEMAStrategy(HybridParams())
    signals = strat.generate(bars)
    assert {"timestamp", "enter_long", "exit_long"}.issubset(signals.columns)

    sim = Simulator()
    trades, curve, summary = sim.run(bars, signals)

    # Basic assertions
    assert isinstance(trades, list)
    assert not curve.empty
    assert "final_equity" in summary
