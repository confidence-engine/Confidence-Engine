import itertools
import os
from datetime import datetime
from typing import Dict, Iterable, List, Tuple

import pandas as pd

from .core import DataLoader, Simulator, write_reports
from .strategies import HybridEMAStrategy, HybridParams


def param_product(grid: Dict[str, Iterable]):
    keys = list(grid.keys())
    for values in itertools.product(*grid.values()):
        yield dict(zip(keys, values))


def run_grid(
    bars_dir: str,
    out_root: str,
    ema_fast_list: List[int],
    ema_slow_list: List[int],
    trend_ema_list: List[int],
    sentiment_cutoff_list: List[float],
    tp_pct_list: List[float],
    sl_pct_list: List[float],
    cooldown_list: List[int],
    risk_frac_list: List[float],
    trend_rule: str = "1h",
) -> str:
    loader = DataLoader(bars_dir)
    bars = loader.load()

    grid = {
        "ema_fast": ema_fast_list,
        "ema_slow": ema_slow_list,
        "trend_ema": trend_ema_list,
        "sentiment_cutoff": sentiment_cutoff_list,
        "tp_pct": tp_pct_list,
        "sl_pct": sl_pct_list,
        "cooldown_sec": cooldown_list,
        "risk_frac": risk_frac_list,
    }

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(out_root, f"grid_{ts}")
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    for p in param_product(grid):
        strat = HybridEMAStrategy(HybridParams(
            ema_fast=p["ema_fast"],
            ema_slow=p["ema_slow"],
            trend_ema=p["trend_ema"],
            trend_rule=trend_rule,
            use_sentiment=False,
            sentiment_cutoff=p["sentiment_cutoff"],
        ))
        signals = strat.generate(bars)
        sim = Simulator(
            tp_pct=p["tp_pct"],
            sl_pct=p["sl_pct"],
            cooldown_sec=p["cooldown_sec"],
            risk_frac=p["risk_frac"],
        )
        trades, curve, summary = sim.run(bars, signals)
        row = {**p, **summary}
        rows.append(row)
    df = pd.DataFrame(rows)
    df = df.sort_values(["final_equity"], ascending=False)
    df.to_csv(os.path.join(out_dir, "results.csv"), index=False)
    df.head(20).to_csv(os.path.join(out_dir, "top20.csv"), index=False)
    return out_dir
