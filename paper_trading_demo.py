#!/usr/bin/env python3
"""
Paper Trading Mode Demo
Demonstrates switching from Alpaca to free data sources for paper trading
"""

import os
import sys
import subprocess

def demo_paper_trading_switch():
    """Demonstrate switching to paper trading with free data sources"""

    print("🎯 PAPER TRADING MODE DEMO")
    print("=" * 60)
    print("This demo shows how to switch your trading agent from Alpaca")
    print("to free data sources for paper trading and testing.")
    print()

    print("📊 BEFORE (Current Configuration):")
    print("  - Data Source: Alpaca (paid)")
    print("  - Trading Mode: Live trading enabled")
    print("  - Cost: Alpaca API fees")
    print()

    print("🎯 AFTER (Paper Trading Mode):")
    print("  - Data Source: Yahoo Finance, Binance, CoinGecko (free)")
    print("  - Trading Mode: Paper trading only")
    print("  - Cost: $0 (free data sources)")
    print()

    print("⚙️ Configuration Changes:")
    print("  TB_ENABLE_MULTI_SOURCE_DATA=1    # Enable free data sources")
    print("  TB_NO_TRADE=1                    # Disable live trading")
    print("  TB_OFFLINE=0                     # Enable data fetching")
    print("  TB_PREFERRED_DATA_SOURCE=yahoo   # Use Yahoo Finance")
    print()

    # Show current status
    print("🔍 Current Status:")
    try:
        result = subprocess.run([
            sys.executable, "enhanced_data_integration.py"
        ], capture_output=True, text=True, cwd=os.path.dirname(__file__))

        if result.returncode == 0:
            # Extract relevant lines
            lines = result.stdout.split('\n')
            for line in lines:
                if any(keyword in line for keyword in ['Multi-Source Enabled', 'Provider Status', 'Testing Data Sources']):
                    print(f"  {line}")
        else:
            print("  ❌ Could not get current status")
    except Exception as e:
        print(f"  ❌ Error getting status: {e}")

    print()
    print("🚀 To Activate Paper Trading Mode:")
    print("  1. Run: python3 enhanced_data_integration.py")
    print("  2. Or manually set the .env variables shown above")
    print("  3. Restart your trading agent")
    print()
    print("💡 Benefits of Free Data Sources:")
    print("  ✅ No API costs for testing and development")
    print("  ✅ Multiple providers for redundancy")
    print("  ✅ Same data quality as Alpaca for most use cases")
    print("  ✅ Automatic failover if one source fails")
    print("  ✅ Perfect for paper trading and strategy testing")
    print()

    print("📈 Available Free Data Sources:")
    print("  🏦 Yahoo Finance: Stocks & crypto, 1min to daily")
    print("  🏦 Binance API: Crypto only, high frequency data")
    print("  🏦 CoinGecko: Crypto only, market data & indicators")
    print("  🏦 Alpha Vantage: Stocks & crypto (requires free API key)")
    print()

    print("🔧 Integration:")
    print("  Your existing code works unchanged!")
    print("  Just replace: from alpaca import recent_bars")
    print("  With: from enhanced_data_integration import enhanced_recent_bars")
    print()

if __name__ == "__main__":
    demo_paper_trading_switch()
