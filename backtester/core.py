import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class Trade:
    entry_time: pd.Timestamp
    entry_price: float
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    qty: float = 0.0
    reason: str = ""

    @property
    def pnl(self) -> Optional[float]:
        if self.exit_price is None:
            return None
        return (self.exit_price - self.entry_price) * self.qty


class DataLoader:
    def __init__(self, bars_dir: str):
        self.bars_dir = bars_dir

    def load(self) -> pd.DataFrame:
        files = [os.path.join(self.bars_dir, f) for f in os.listdir(self.bars_dir) if f.endswith(".csv")]
        if not files:
            raise FileNotFoundError(f"No CSV files found in {self.bars_dir}")
        frames = []
        for f in sorted(files):
            df = pd.read_csv(f)
            # Expect columns: timestamp,open,high,low,close,volume
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            frames.append(df)
        bars = pd.concat(frames, ignore_index=True)
        bars = bars.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        # Ensure numeric
        for c in ["open", "high", "low", "close", "volume"]:
            bars[c] = pd.to_numeric(bars[c], errors="coerce")
        bars = bars.dropna().reset_index(drop=True)
        return bars


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def resample_ohlc(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    ohlc = df.set_index("timestamp")["close"].resample(rule).ohlc()
    vol = df.set_index("timestamp")["volume"].resample(rule).sum()
    out = ohlc.join(vol)
    out = out.rename(columns={"sum": "volume"}).dropna().reset_index()
    return out


class Metrics:
    @staticmethod
    def equity_curve(trades: List[Trade], bars: pd.DataFrame, starting_equity: float) -> pd.DataFrame:
        equity = starting_equity
        curve = []
        trade_iter = iter(sorted([t for t in trades if t.exit_time is not None], key=lambda x: x.exit_time))
        t = next(trade_iter, None)
        for ts, price in zip(bars["timestamp"], bars["close"]):
            while t is not None and t.exit_time is not None and ts >= t.exit_time:
                equity += t.pnl or 0.0
                t = next(trade_iter, None)
            curve.append({"timestamp": ts, "equity": equity})
        return pd.DataFrame(curve)

    @staticmethod
    def summarize(equity_curve: pd.DataFrame, trades: List[Trade], starting_equity: float) -> Dict[str, float]:
        eq = equity_curve["equity"].values
        if len(eq) == 0:
            return {"cagr": 0.0, "max_dd": 0.0, "sharpe": 0.0, "win_rate": 0.0, "trades": 0}
        ret = np.diff(eq) / starting_equity
        # Dailyized approximations if minute bars: scale factor
        if len(ret) > 1:
            sharpe = (np.mean(ret) / (np.std(ret) + 1e-9)) * np.sqrt(252)
        else:
            sharpe = 0.0
        peak = -np.inf
        max_dd = 0.0
        for v in eq:
            if v > peak:
                peak = v
            dd = (peak - v) / peak if peak > 0 else 0.0
            if dd > max_dd:
                max_dd = dd
        realized = [t for t in trades if t.exit_price is not None]
        wins = [t for t in realized if (t.exit_price - t.entry_price) > 0]
        win_rate = (len(wins) / len(realized)) if realized else 0.0
        # Rough CAGR over the span
        days = max(1, (equity_curve["timestamp"].iloc[-1] - equity_curve["timestamp"].iloc[0]).total_seconds() / 86400)
        cagr = (eq[-1] / starting_equity) ** (365.0 / days) - 1.0 if days > 0 else 0.0
        return {
            "cagr": float(cagr),
            "max_dd": float(max_dd),
            "sharpe": float(sharpe),
            "win_rate": float(win_rate),
            "trades": len(realized),
            "final_equity": float(eq[-1]),
        }


class Simulator:
    def __init__(
        self,
        fee_bps: float = 5.0,
        slippage_bps: float = 2.0,
        tp_pct: float = 0.05,
        sl_pct: float = 0.02,
        cooldown_sec: int = 3600,
        risk_frac: float = 0.01,
        starting_equity: float = 10000.0,
        # Dynamic stop prototypes (backtester-only)
        stop_mode: str = "fixed_pct",  # one of: fixed_pct, atr_fixed, atr_trailing
        atr_period: int = 14,
        atr_mult: float = 1.5,
        time_cap_bars: int = 0,  # 0 disables time-based exit
    ):
        self.fee = fee_bps / 10000.0
        self.slip = slippage_bps / 10000.0
        self.tp = tp_pct
        self.sl = sl_pct
        self.cooldown = timedelta(seconds=cooldown_sec)
        self.risk_frac = risk_frac
        self.starting_equity = starting_equity
        self.stop_mode = stop_mode
        self.atr_period = atr_period
        self.atr_mult = atr_mult
        self.time_cap_bars = max(0, int(time_cap_bars or 0))

    def run(self, bars: pd.DataFrame, signals: pd.DataFrame) -> Tuple[List[Trade], pd.DataFrame, Dict[str, float]]:
        in_pos = False
        entry_price = 0.0
        qty = 0.0
        last_exit_time: Optional[pd.Timestamp] = None
        trades: List[Trade] = []
        equity = self.starting_equity
        current_stop = None  # dynamic stop tracker
        bars_in_trade = 0

        # Precompute ATR (simple true range with Wilder smoothing)
        def _atr(df: pd.DataFrame, period: int) -> pd.Series:
            high = df["high"].astype(float)
            low = df["low"].astype(float)
            close = df["close"].astype(float)
            prev_close = close.shift(1)
            tr = pd.concat([
                (high - low),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ], axis=1).max(axis=1)
            # Wilder's smoothing
            atr = tr.ewm(alpha=1/period, adjust=False).mean()
            return atr

        atr_series = _atr(bars, self.atr_period) if self.stop_mode in ("atr_fixed", "atr_trailing") else None

        for i, row in bars.iterrows():
            ts = row["timestamp"]
            o, h, l, c = row["open"], row["high"], row["low"], row["close"]
            sig = signals.iloc[i]

            if not in_pos:
                if sig.get("enter_long", False):
                    # position sizing by risk fraction
                    price_fill = c * (1 + self.slip)
                    cash = equity * self.risk_frac
                    qty = max(0.0, cash / price_fill)
                    fee = price_fill * qty * self.fee
                    entry_price = price_fill
                    equity -= fee
                    trades.append(Trade(entry_time=ts, entry_price=entry_price, qty=qty, reason="enter_long"))
                    in_pos = True
                    last_exit_time = None
                    bars_in_trade = 0
                    # Initialize dynamic stop
                    if self.stop_mode == "fixed_pct":
                        current_stop = entry_price * (1 - self.sl)
                    elif self.stop_mode in ("atr_fixed", "atr_trailing") and atr_series is not None:
                        atr_val = float(atr_series.iloc[i]) if pd.notna(atr_series.iloc[i]) else float("nan")
                        if np.isfinite(atr_val):
                            current_stop = entry_price - self.atr_mult * atr_val
                        else:
                            current_stop = entry_price * (1 - self.sl)
                    else:
                        current_stop = entry_price * (1 - self.sl)
            else:
                # Update trailing ATR stop if enabled
                if self.stop_mode == "atr_trailing" and atr_series is not None:
                    atr_val = float(atr_series.iloc[i]) if pd.notna(atr_series.iloc[i]) else float("nan")
                    if np.isfinite(atr_val):
                        proposed = c - self.atr_mult * atr_val
                        # Only ratchet upwards for long positions
                        if current_stop is None:
                            current_stop = proposed
                        else:
                            current_stop = max(current_stop, proposed)

                # Check TP/SL intra-bar (use current_stop)
                take = entry_price * (1 + self.tp)
                stop = current_stop if current_stop is not None else entry_price * (1 - self.sl)
                exit_reason = None
                exit_price = None
                if l <= stop:
                    exit_price = stop * (1 - self.slip)
                    exit_reason = "stop"
                elif h >= take:
                    exit_price = take * (1 - self.slip)
                    exit_reason = "take"
                elif sig.get("exit_long", False):
                    exit_price = c * (1 - self.slip)
                    exit_reason = "signal_exit"
                elif self.time_cap_bars > 0 and bars_in_trade >= self.time_cap_bars:
                    exit_price = c * (1 - self.slip)
                    exit_reason = "time_cap"

                if exit_price is not None:
                    fee = exit_price * qty * self.fee
                    trades[-1].exit_time = ts
                    trades[-1].exit_price = exit_price
                    trades[-1].reason = exit_reason
                    equity += (exit_price - entry_price) * qty - fee
                    in_pos = False
                    last_exit_time = ts
                    current_stop = None
                    bars_in_trade = 0
                else:
                    # hold
                    bars_in_trade += 1

            # cooldown: prevent immediate re-entry after an exit
            if not in_pos and last_exit_time is not None:
                if ts - last_exit_time < self.cooldown:
                    # force no entry by clearing enter_long flag for this bar
                    signals.at[i, "enter_long"] = False

        curve = Metrics.equity_curve(trades, bars, self.starting_equity)
        summary = Metrics.summarize(curve, trades, self.starting_equity)
        return trades, curve, summary


def write_reports(out_dir: str, trades: List[Trade], curve: pd.DataFrame, params: Dict, summary: Dict):
    os.makedirs(out_dir, exist_ok=True)
    # trades.csv
    trades_rows = []
    for t in trades:
        trades_rows.append({
            "entry_time": t.entry_time.isoformat() if isinstance(t.entry_time, pd.Timestamp) else str(t.entry_time),
            "entry_price": t.entry_price,
            "exit_time": t.exit_time.isoformat() if isinstance(t.exit_time, pd.Timestamp) else (t.exit_time if t.exit_time is None else str(t.exit_time)),
            "exit_price": t.exit_price,
            "qty": t.qty,
            "reason": t.reason,
            "pnl": t.pnl,
        })
    pd.DataFrame(trades_rows).to_csv(os.path.join(out_dir, "trades.csv"), index=False)
    # equity.csv
    curve.to_csv(os.path.join(out_dir, "equity.csv"), index=False)
    # params.json
    with open(os.path.join(out_dir, "params.json"), "w") as f:
        json.dump(params, f, indent=2)
    # summary.json
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
