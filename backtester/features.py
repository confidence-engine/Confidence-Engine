from typing import Dict, Tuple

import numpy as np
import pandas as pd

from .core import ema


def add_returns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ret_1"] = out["close"].pct_change()
    out["ret_fwd_1"] = out["close"].pct_change().shift(-1)
    return out


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add sophisticated technical indicators for better ML features"""
    out = df.copy()

    # RSI (Relative Strength Index) - Enhanced with smoothing
    delta = out["close"].diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/14).mean()  # Exponential smoothing
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14).mean()
    rs = gain / loss
    out["rsi"] = 100 - (100 / (1 + rs))

    # RSI Divergence signals
    out["rsi_divergence"] = (out["rsi"] - out["rsi"].shift(5)) / out["rsi"].shift(5)

    # MACD (Moving Average Convergence Divergence)
    out["ema12"] = ema(out["close"], 12)
    out["ema26"] = ema(out["close"], 26)
    out["macd"] = out["ema12"] - out["ema26"]
    out["macd_signal"] = ema(out["macd"], 9)
    out["macd_histogram"] = out["macd"] - out["macd_signal"]

    # MACD momentum
    out["macd_momentum"] = out["macd"].diff()

    # Bollinger Bands
    out["sma20"] = out["close"].rolling(window=20).mean()
    out["std20"] = out["close"].rolling(window=20).std()
    out["bb_upper"] = out["sma20"] + (out["std20"] * 2)
    out["bb_lower"] = out["sma20"] - (out["std20"] * 2)
    out["bb_position"] = (out["close"] - out["bb_lower"]) / (out["bb_upper"] - out["bb_lower"])
    out["bb_width"] = (out["bb_upper"] - out["bb_lower"]) / out["sma20"]  # Band width as volatility proxy

    # Enhanced Momentum indicators
    out["momentum_5"] = out["close"] / out["close"].shift(5) - 1
    out["momentum_10"] = out["close"] / out["close"].shift(10) - 1
    out["momentum_20"] = out["close"] / out["close"].shift(20) - 1

    # Rate of Change (ROC)
    out["roc_5"] = out["close"].pct_change(5)
    out["roc_10"] = out["close"].pct_change(10)

    # Volatility measures
    out["close_volatility"] = out["close"].pct_change().rolling(20).std()
    out["high_low_range"] = (out.get("high", out["close"]) - out.get("low", out["close"])) / out["close"]
    out["atr"] = out["high_low_range"].rolling(14).mean()  # Average True Range proxy

    # Volume indicators
    out["volume_sma"] = out["volume"].rolling(window=20).mean()
    out["volume_ratio"] = out["volume"] / out["volume_sma"]
    out["volume_trend"] = out["volume"].pct_change(5)

    # Price patterns
    out["price_acceleration"] = out["close"].pct_change().diff()  # Second derivative
    out["support_resistance"] = (out["close"] - out["close"].rolling(20).min()) / (out["close"].rolling(20).max() - out["close"].rolling(20).min())

    return out


def build_features(bars: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    df = add_returns(bars)
    df = add_technical_indicators(df)

    # Original features
    df["ema12_slope"] = df["ema12"].pct_change()
    df["ema26_slope"] = df["ema26"].pct_change()
    df["vol"] = df["close"].pct_change().rolling(20).std().fillna(0.0)
    df["vol_chg"] = df["volume"].pct_change().fillna(0.0)
    df["cross_up"] = ((df["ema12"].shift(1) <= df["ema26"].shift(1)) & (df["ema12"] > df["ema26"]))
    df["cross_up"] = df["cross_up"].astype(int)

    # Label: next bar positive return
    y = (df["ret_fwd_1"] > 0).astype(int)

    # Enhanced feature set with 25+ indicators
    features = df[[
        "ret_1",
        "ema12_slope",
        "ema26_slope",
        "vol",
        "vol_chg",
        "cross_up",
        "rsi",
        "rsi_divergence",
        "macd",
        "macd_signal",
        "macd_histogram",
        "macd_momentum",
        "bb_position",
        "bb_width",
        "momentum_5",
        "momentum_10",
        "momentum_20",
        "roc_5",
        "roc_10",
        "close_volatility",
        "high_low_range",
        "atr",
        "volume_ratio",
        "volume_trend",
        "price_acceleration",
        "support_resistance",
    ]].fillna(0.0)

    # Replace +/-inf (e.g., from volume pct_change when previous volume=0)
    features = features.replace([np.inf, -np.inf], 0.0)
    # Drop last row with NaN label due to shift
    features = features.iloc[:-1]
    y = y.iloc[:-1]
    return features.reset_index(drop=True), y.reset_index(drop=True)
