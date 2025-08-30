from dataclasses import dataclass
from typing import Dict, Optional

import pandas as pd

from .core import ema, resample_ohlc


@dataclass
class HybridParams:
    ema_fast: int = 12
    ema_slow: int = 26
    trend_ema: int = 50
    trend_rule: str = "1h"  # resample rule for trend timeframe
    use_sentiment: bool = False
    sentiment_cutoff: float = 0.5


class HybridEMAStrategy:
    """Hybrid EMA crossover with higher timeframe trend filter and optional sentiment gate.

    Signals frame columns: enter_long, exit_long (bool)
    """

    def __init__(self, params: HybridParams):
        self.p = params

    def generate(self, bars: pd.DataFrame, sentiment: Optional[pd.Series] = None) -> pd.DataFrame:
        df = bars.copy()
        df["ema_fast"] = ema(df["close"], self.p.ema_fast)
        df["ema_slow"] = ema(df["close"], self.p.ema_slow)
        df["cross_up"] = (df["ema_fast"].shift(1) <= df["ema_slow"].shift(1)) & (df["ema_fast"] > df["ema_slow"])
        df["cross_down"] = (df["ema_fast"].shift(1) >= df["ema_slow"].shift(1)) & (df["ema_fast"] < df["ema_slow"])

        # Trend on higher timeframe
        high_tf = resample_ohlc(df[["timestamp", "close", "volume"]], self.p.trend_rule)
        high_tf["trend_ema"] = ema(high_tf["close"], self.p.trend_ema)
        high_tf["trend_up"] = high_tf["close"] > high_tf["trend_ema"]
        trend = high_tf[["timestamp", "trend_up"]].set_index("timestamp")
        trend = trend.reindex(df.set_index("timestamp").index, method="ffill").reset_index()
        trend = trend.rename(columns={"index": "timestamp"})

        df["trend_up"] = trend["trend_up"].fillna(False)

        # Sentiment gate
        if self.p.use_sentiment and sentiment is not None:
            sent = sentiment.reindex(df["timestamp"]).fillna(method="ffill").fillna(0.0)
            df["sent_ok"] = sent >= self.p.sentiment_cutoff
        else:
            df["sent_ok"] = True

        df["enter_long"] = df["cross_up"] & df["trend_up"] & df["sent_ok"]
        df["exit_long"] = df["cross_down"]
        return df[["timestamp", "enter_long", "exit_long"]]
