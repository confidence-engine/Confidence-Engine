import argparse
import os
from datetime import datetime

import pandas as pd

from .core import DataLoader, Simulator, write_reports
from .strategies import HybridEMAStrategy, HybridParams


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bars_dir", default="bars/")
    ap.add_argument("--out_root", default="eval_runs/backtests/")
    ap.add_argument("--ema_fast", type=int, default=12)
    ap.add_argument("--ema_slow", type=int, default=26)
    ap.add_argument("--trend_ema", type=int, default=50)
    ap.add_argument("--trend_rule", default="1h")
    ap.add_argument("--use_sentiment", action="store_true")
    ap.add_argument("--sentiment_cutoff", type=float, default=0.5)
    ap.add_argument("--fee_bps", type=float, default=5.0)
    ap.add_argument("--slippage_bps", type=float, default=2.0)
    ap.add_argument("--tp_pct", type=float, default=0.05)
    ap.add_argument("--sl_pct", type=float, default=0.02)
    ap.add_argument("--cooldown_sec", type=int, default=3600)
    ap.add_argument("--risk_frac", type=float, default=0.01)
    ap.add_argument("--starting_equity", type=float, default=10000.0)
    # Dynamic stop prototypes (backtester-only)
    ap.add_argument("--stop_mode", default="fixed_pct", choices=["fixed_pct", "atr_fixed", "atr_trailing"])
    ap.add_argument("--atr_period", type=int, default=14)
    ap.add_argument("--atr_mult", type=float, default=1.5)
    ap.add_argument("--time_cap_bars", type=int, default=0)
    args = ap.parse_args()

    loader = DataLoader(args.bars_dir)
    bars = loader.load()

    strat = HybridEMAStrategy(HybridParams(
        ema_fast=args.ema_fast,
        ema_slow=args.ema_slow,
        trend_ema=args.trend_ema,
        trend_rule=args.trend_rule,
        use_sentiment=args.use_sentiment,
        sentiment_cutoff=args.sentiment_cutoff,
    ))
    signals = strat.generate(bars)

    sim = Simulator(
        fee_bps=args.fee_bps,
        slippage_bps=args.slippage_bps,
        tp_pct=args.tp_pct,
        sl_pct=args.sl_pct,
        cooldown_sec=args.cooldown_sec,
        risk_frac=args.risk_frac,
        starting_equity=args.starting_equity,
        stop_mode=args.stop_mode,
        atr_period=args.atr_period,
        atr_mult=args.atr_mult,
        time_cap_bars=args.time_cap_bars,
    )

    trades, curve, summary = sim.run(bars, signals)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.out_root, ts)
    params = vars(args)
    write_reports(out_dir, trades, curve, params, summary)
    print(f"Backtest written to: {out_dir}")
    print("Summary:", summary)


if __name__ == "__main__":
    main()
