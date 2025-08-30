import os
from datetime import datetime, timedelta
from typing import Dict, List

import pandas as pd

from .core import DataLoader, Simulator
from .strategies import HybridEMAStrategy, HybridParams


def time_splits(df: pd.DataFrame, train_days: int, test_days: int, n_splits: int) -> List[Dict[str, pd.Timestamp]]:
    start = df["timestamp"].min().normalize()
    end = df["timestamp"].max().normalize()
    spans = []
    cur = start
    for _ in range(n_splits):
        train_start = cur
        train_end = train_start + timedelta(days=train_days)
        test_end = train_end + timedelta(days=test_days)
        if test_end > end:
            break
        spans.append({
            "train_start": train_start,
            "train_end": train_end,
            "test_start": train_end,
            "test_end": test_end,
        })
        cur = cur + timedelta(days=test_days)
    return spans


def run_walk_forward(
    bars_dir: str,
    out_root: str,
    train_days: int = 14,
    test_days: int = 7,
    n_splits: int = 4,
    ema_fast: int = 12,
    ema_slow: int = 26,
    trend_ema: int = 50,
    trend_rule: str = "1h",
    tp_pct: float = 0.05,
    sl_pct: float = 0.02,
    cooldown_sec: int = 3600,
    risk_frac: float = 0.01,
) -> str:
    loader = DataLoader(bars_dir)
    bars = loader.load()

    splits = time_splits(bars, train_days, test_days, n_splits)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(out_root, f"walk_{ts}")
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    for i, sp in enumerate(splits):
        fold_dir = os.path.join(out_dir, f"fold_{i}")
        os.makedirs(fold_dir, exist_ok=True)
        mask_train = (bars["timestamp"] >= sp["train_start"]) & (bars["timestamp"] < sp["train_end"])
        mask_test = (bars["timestamp"] >= sp["test_start"]) & (bars["timestamp"] < sp["test_end"])
        bars_train = bars.loc[mask_train].reset_index(drop=True)
        bars_test = bars.loc[mask_test].reset_index(drop=True)
        if len(bars_test) == 0:
            continue
        # Strategy params could be tuned on train (stub: keep defaults for now)
        strat = HybridEMAStrategy(HybridParams(
            ema_fast=ema_fast,
            ema_slow=ema_slow,
            trend_ema=trend_ema,
            trend_rule=trend_rule,
            use_sentiment=False,
        ))
        signals_test = strat.generate(bars_test)
        sim = Simulator(tp_pct=tp_pct, sl_pct=sl_pct, cooldown_sec=cooldown_sec, risk_frac=risk_frac)
        trades, curve, summary = sim.run(bars_test, signals_test)
        summary_row = {"fold": i, **sp, **summary}
        rows.append(summary_row)
        # Save per-fold equity and trades
        curve.to_csv(os.path.join(fold_dir, "equity.csv"), index=False)
        pd.DataFrame([{k: (v.isoformat() if hasattr(v, 'isoformat') else v) for k, v in summary_row.items()}]).to_csv(
            os.path.join(fold_dir, "summary.csv"), index=False
        )
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(out_dir, "summary_all.csv"), index=False)
    return out_dir
