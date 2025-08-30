from typing import Dict, Tuple

import numpy as np
import pandas as pd

from .core import ema


def add_returns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ret_1"] = out["close"].pct_change()
    out["ret_fwd_1"] = out["close"].pct_change().shift(-1)
    return out


def build_features(bars: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    df = add_returns(bars)
    df["ema12"] = ema(df["close"], 12)
    df["ema26"] = ema(df["close"], 26)
    df["ema12_slope"] = df["ema12"].pct_change()
    df["ema26_slope"] = df["ema26"].pct_change()
    df["vol"] = df["close"].pct_change().rolling(20).std().fillna(0.0)
    df["vol_chg"] = df["volume"].pct_change().fillna(0.0)
    df["cross_up"] = ((df["ema12"].shift(1) <= df["ema26"].shift(1)) & (df["ema12"] > df["ema26"]))
    df["cross_up"] = df["cross_up"].astype(int)
    # Label: next bar positive return
    y = (df["ret_fwd_1"] > 0).astype(int)
    features = df[[
        "ret_1",
        "ema12_slope",
        "ema26_slope",
        "vol",
        "vol_chg",
        "cross_up",
    ]].fillna(0.0)
    # Replace +/-inf (e.g., from volume pct_change when previous volume=0)
    features = features.replace([np.inf, -np.inf], 0.0)
    # Drop last row with NaN label due to shift
    features = features.iloc[:-1]
    y = y.iloc[:-1]
    return features.reset_index(drop=True), y.reset_index(drop=True)
