import numpy as np
import pandas as pd

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    gain = up.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    loss = down.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = gain / (loss + 1e-12)
    return 100 - (100 / (1 + rs))

def macd_hist(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    sig = macd.ewm(span=signal, adjust=False).mean()
    return macd - sig

def ma_slope(close: pd.Series, short: int = 10, medium: int = 50) -> pd.Series:
    ma_s = close.ewm(span=short, adjust=False).mean()
    ma_m = close.ewm(span=medium, adjust=False).mean()
    return ma_s - ma_m

def volume_z(vol: pd.Series, window: int = 50) -> pd.Series:
    mean = vol.rolling(window).mean()
    std = vol.rolling(window).std(ddof=0).replace(0, np.nan)
    z = (vol - mean) / (std + 1e-12)
    return z.fillna(0.0)

def _unit(x: pd.Series) -> pd.Series:
    z = (x - x.mean()) / (x.std(ddof=0) + 1e-12)
    return np.tanh(z)

def price_score(df: pd.DataFrame):
    if len(df) < 60:
        return 0.0, 0.0
    close = df["close"]
    vol = df["volume"]

    rsi_s = (rsi(close) - 50.0) / 50.0
    macd_s = _unit(macd_hist(close))
    slope_s = _unit(ma_slope(close))

    vol_zs = volume_z(vol)
    vz = float(vol_zs.iloc[-1])

    comp = float(np.mean([float(rsi_s.iloc[-1]), float(macd_s.iloc[-1]), float(slope_s.iloc[-1])]))
    if vz < -0.5:
        comp *= 0.7
    return max(-1.0, min(1.0, comp)), vz
