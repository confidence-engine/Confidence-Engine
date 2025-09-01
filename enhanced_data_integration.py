#!/usr/bin/env python3
"""
Integration script to replace Alpaca with multi-source free data providers
This allows seamless switching between Alpaca and free alternatives for paper trading
"""

import os
import sys
import pandas as pd
from typing import Optional, List
from datetime import datetime, timedelta, timezone

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from multi_source_data_provider import get_crypto_bars_alternative, get_crypto_news_alternative

# Configuration
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not available

ENABLE_MULTI_SOURCE = os.getenv("TB_ENABLE_MULTI_SOURCE_DATA", "0") == "1"
PREFERRED_SOURCE = os.getenv("TB_PREFERRED_DATA_SOURCE", "yahoo")
ENABLE_FAILOVER = os.getenv("TB_DATA_SOURCE_FAILOVER", "1") == "1"

def enhanced_recent_bars(symbol: str, minutes: int = 120) -> pd.DataFrame:
    """
    Enhanced version of Alpaca's recent_bars with multi-source support
    Automatically falls back between free providers and Alpaca
    """

    if ENABLE_MULTI_SOURCE:
        # Try multi-source providers first
        try:
            # Convert minutes to appropriate limit
            if minutes <= 60:  # 1 hour
                limit = 60  # Roughly 1 bar per minute
                timeframe = "1Min"
            elif minutes <= 480:  # 8 hours
                limit = 60  # Roughly 1 bar per 8 minutes
                timeframe = "15Min"
            else:  # More than 8 hours
                limit = 60  # Roughly 1 bar per hour
                timeframe = "1Hour"

            data = get_crypto_bars_alternative(symbol, timeframe, limit)
            if data is not None and not data.empty:
                # Filter to requested time range
                end_time = datetime.now(timezone.utc)
                start_time = end_time - timedelta(minutes=minutes)

                filtered_data = data[data.index >= start_time]
                if not filtered_data.empty:
                    print(f"✅ Using multi-source data ({PREFERRED_SOURCE}) for {symbol}")
                    return filtered_data

        except Exception as e:
            print(f"⚠️ Multi-source data failed for {symbol}: {e}")
            if not ENABLE_FAILOVER:
                raise

    # Fallback to Alpaca
    try:
        from alpaca import recent_bars as alpaca_recent_bars
        print(f"🏦 Using Alpaca for {symbol} (fallback)")
        return alpaca_recent_bars(symbol, minutes)
    except Exception as e:
        print(f"❌ Both multi-source and Alpaca failed for {symbol}: {e}")
        raise

def enhanced_latest_headlines(symbol: str, limit: int = 10) -> List[str]:
    """
    Enhanced version of Alpaca's latest_headlines with multi-source support
    """

    if ENABLE_MULTI_SOURCE:
        try:
            headlines = get_crypto_news_alternative(symbol, limit)
            if headlines:
                print(f"✅ Using multi-source news for {symbol}")
                return headlines
        except Exception as e:
            print(f"⚠️ Multi-source news failed for {symbol}: {e}")
            if not ENABLE_FAILOVER:
                return []

    # Fallback to Alpaca
    try:
        from alpaca import latest_headlines as alpaca_latest_headlines
        print(f"🏦 Using Alpaca news for {symbol} (fallback)")
        return alpaca_latest_headlines(symbol, limit)
    except Exception as e:
        print(f"❌ Both multi-source and Alpaca news failed for {symbol}: {e}")
        return []

def get_data_source_status() -> dict:
    """Get status of all data sources"""
    from multi_source_data_provider import multi_source_provider

    status = {
        "multi_source_enabled": ENABLE_MULTI_SOURCE,
        "preferred_source": PREFERRED_SOURCE,
        "failover_enabled": ENABLE_FAILOVER,
        "providers": multi_source_provider.get_provider_status()
    }

    # Test Alpaca availability
    try:
        from alpaca import _rest
        api = _rest()
        account = api.get_account()
        status["alpaca_available"] = True
    except Exception:
        status["alpaca_available"] = False

    return status

def switch_to_paper_trading_mode():
    """
    Configure the system for paper trading with free data sources
    This disables live trading but keeps all analysis running
    """

    print("🔄 Switching to Paper Trading Mode with Free Data Sources")
    print("=" * 60)

    # Update environment for paper trading
    updates = {
        "TB_NO_TRADE": "1",           # Disable live trading
        "TB_OFFLINE": "0",            # Enable data fetching
        "TB_ENABLE_MULTI_SOURCE_DATA": "1",  # Enable multi-source
        "TB_PREFERRED_DATA_SOURCE": "yahoo", # Use Yahoo Finance
        "TB_DATA_SOURCE_FAILOVER": "1"       # Enable failover
    }

    print("📝 Configuration updates:")
    for key, value in updates.items():
        print(f"  {key}={value}")

    print("\n✅ Paper trading mode activated!")
    print("💡 Your agent will now use free data sources instead of Alpaca")
    print("💡 All trading signals will be generated but no live orders placed")
    print("💡 You can monitor performance and test strategies risk-free")

    return updates

if __name__ == "__main__":
    print("🔧 Multi-Source Data Integration Tool")
    print("=" * 60)

    # Show current status
    status = get_data_source_status()
    print("📊 Current Configuration:")
    print(f"  Multi-Source Enabled: {'✅' if status['multi_source_enabled'] else '❌'}")
    print(f"  Preferred Source: {status['preferred_source']}")
    print(f"  Failover Enabled: {'✅' if status['failover_enabled'] else '❌'}")
    print(f"  Alpaca Available: {'✅' if status['alpaca_available'] else '❌'}")

    print("\n🔌 Provider Status:")
    for provider, available in status['providers'].items():
        status_icon = "✅" if available else "❌"
        print(f"  {status_icon} {provider}")

    print("\n🧪 Testing Data Sources...")

    # Test with BTC
    try:
        data = enhanced_recent_bars("BTC/USD", minutes=60)
        if data is not None and not data.empty:
            print(f"✅ Successfully fetched {len(data)} bars for BTC/USD")
            print(f"   Latest price: ${data['close'].iloc[-1]:.2f}")
        else:
            print("❌ No data received")
    except Exception as e:
        print(f"❌ Test failed: {e}")

    print("\n📚 Usage:")
    print("  1. Set TB_ENABLE_MULTI_SOURCE_DATA=1 in .env")
    print("  2. Choose preferred source: yahoo, binance, coingecko, alphavantage")
    print("  3. The system will automatically failover if sources fail")
    print("  4. Use enhanced_recent_bars() and enhanced_latest_headlines() in your code")
