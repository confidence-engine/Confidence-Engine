from datetime import datetime, timedelta, timezone
from typing import List, Tuple
import pandas as pd
from alpaca_trade_api.rest import REST
from config import settings

def _rest() -> REST:
    return REST(
        key_id=settings.alpaca_key_id,
        secret_key=settings.alpaca_secret_key,
        base_url=settings.alpaca_base_url
    )

def _normalize_crypto_symbol(sym: str) -> str:
    # If user provides BTCUSD, convert to BTC/USD. If already BTC/USD, keep it.
    if "/" in sym:
        return sym
    if len(sym) >= 6:
        base = sym[:-3]
        quote = sym[-3:]
        return f"{base}/{quote}"
    return sym

def recent_bars(symbol: str, minutes: int = 120) -> pd.DataFrame:
    api = _rest()
    sym = _normalize_crypto_symbol(symbol)
    end = datetime.now(timezone.utc)
    start = end - timedelta(minutes=minutes + 5)
    # For crypto, Alpaca expects pair format like BTC/USD
    bars = api.get_crypto_bars(sym, "1Min", start.isoformat(), end.isoformat()).df
    if isinstance(bars.index, pd.MultiIndex):
        bars = bars.xs(sym, level=0)
    # Fix timezone handling - check if index is DatetimeIndex before accessing .tz
    if isinstance(bars.index, pd.DatetimeIndex):
        if bars.index.tz is None:
            bars.index = bars.index.tz_localize("UTC")
        else:
            bars = bars.tz_convert("UTC")
    else:
        # If not a DatetimeIndex, convert to DatetimeIndex with UTC timezone
        if not isinstance(bars.index, pd.DatetimeIndex):
            # Assume the index contains timestamps and convert
            try:
                bars.index = pd.to_datetime(bars.index).tz_localize("UTC")
            except Exception:
                # If conversion fails, create a proper DatetimeIndex
                bars.index = pd.date_range(start=datetime.now(timezone.utc), periods=len(bars), freq='1min', tz='UTC')[:len(bars)]
    bars = bars.loc[bars.index >= (end - timedelta(minutes=minutes))]
    return bars[["open", "high", "low", "close", "volume"]].copy()

def latest_headlines(symbol: str, limit: int = 10) -> List[str]:
    api = _rest()
    sym = _normalize_crypto_symbol(symbol)
    news = api.get_news(sym, limit=limit)
    out: List[str] = []
    for n in news:
        title = getattr(n, "headline", None) or getattr(n, "title", None)
        if title:
            out.append(title.strip())
    return out

def smoke(symbol: str) -> Tuple[int, int]:
    bars = recent_bars(symbol, 60)
    heads = latest_headlines(symbol, 5)
    return len(bars), len(heads)
