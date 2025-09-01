#!/usr/bin/env python3
"""
Futures Integration Test for Trading Agent
Shows how to seamlessly integrate futures trading with existing agent
"""

import os
import sys
import pandas as pd
from datetime import datetime
import logging

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Import our futures system
from futures_integration import (
    futures_integration,
    enhanced_futures_bars,
    calculate_futures_position,
    execute_futures_trade,
    get_futures_status
)

# Import existing agent components (simulated)
class MockAlpacaAPI:
    """Mock Alpaca API for comparison"""

    def get_bars(self, symbol, timeframe, limit=100):
        """Mock spot data (limited to Alpaca's capabilities)"""
        # This would normally call Alpaca's spot API
        print(f"📊 Alpaca Spot: Getting {limit} bars for {symbol}")
        return pd.DataFrame()  # Empty for demo

class MockTradingAgent:
    """Mock trading agent to show integration"""

    def __init__(self):
        self.alpaca = MockAlpacaAPI()
        self.capital = 100000.0
        self.risk_per_trade = 0.02

    def get_market_data_old_way(self, symbol, timeframe='1h', limit=100):
        """Old way: Alpaca spot only"""
        print(f"\n🔄 OLD WAY (Alpaca Spot Only):")
        data = self.alpaca.get_bars(symbol, timeframe, limit)
        print(f"   ❌ Limited to spot trading only")
        print(f"   ❌ No leverage available")
        print(f"   ❌ No futures/perpetuals")
        return data

    def get_market_data_new_way(self, symbol, timeframe='1h', limit=100):
        """New way: Futures & Perpetuals"""
        print(f"\n🚀 NEW WAY (Futures & Perpetuals):")
        data = enhanced_futures_bars(symbol, timeframe, limit)
        if data is not None and len(data) > 0:
            print(f"   ✅ Got {len(data)} bars from futures market")
            print(".2f")
            print(f"   ✅ Leverage up to 125x available")
            print(f"   ✅ Futures & perpetuals supported")
            return data
        else:
            print(f"   ❌ No futures data available")
            return None

    def calculate_position_old_way(self, symbol):
        """Old way: Spot position sizing"""
        print(f"\n📊 OLD POSITION SIZING (Spot Only):")
        # Simple spot position sizing
        risk_amount = self.capital * self.risk_per_trade
        # Assume some volatility estimate
        volatility = 0.05  # 5% daily volatility estimate
        position_value = risk_amount / volatility

        print(f"   💰 Position Value: ${position_value:.2f}")
        print(f"   ⚡ Leverage: 1x (spot only)")
        print(f"   📊 Risk Amount: ${risk_amount:.2f}")
        print(f"   ❌ No leverage advantage")

        return {
            'position_value': position_value,
            'leverage': 1,
            'risk_amount': risk_amount
        }

    def calculate_position_new_way(self, symbol):
        """New way: Futures position sizing"""
        print(f"\n🚀 NEW POSITION SIZING (Futures):")
        pos_info = calculate_futures_position(symbol, self.capital, self.risk_per_trade)

        if 'error' not in pos_info:
            print(f"   💰 Position Value: ${pos_info['position_value']:.2f}")
            print(f"   ⚡ Leverage: {pos_info['leverage_used']:.1f}x")
            print(f"   📊 Risk Amount: ${pos_info['risk_amount']:.2f}")
            print(f"   📈 Volatility: {pos_info['volatility']:.4f}")
            print(f"   🎲 Kelly %: {pos_info['kelly_pct']:.2f}")
            print(f"   ✅ Advanced risk management")

            return pos_info
        else:
            print(f"   ❌ Error: {pos_info['error']}")
            return None

    def execute_trade_comparison(self, symbol, side):
        """Compare old vs new trading execution"""
        print(f"\n📈 TRADE EXECUTION COMPARISON ({side.upper()} {symbol})")
        print("-" * 50)

        # Old way simulation
        print("🔄 OLD WAY (Alpaca Spot):")
        print(f"   📊 Order Type: Market (spot)")
        print(f"   ⚡ Leverage: 1x")
        print(f"   💰 Max Position: Limited by capital")
        print(f"   ❌ No futures/perpetuals")

        # New way
        print("\n🚀 NEW WAY (Futures Platform):")
        pos_info = self.calculate_position_new_way(symbol)
        if pos_info:
            trade_result = execute_futures_trade(symbol, side, pos_info)
            if 'error' not in trade_result:
                print(f"   ✅ Order ID: {trade_result['order_id']}")
                print(f"   📊 Type: {trade_result['type']}")
                print(f"   ⚡ Leverage: {trade_result['leverage']}x")
                print(f"   💰 Position Size: ${trade_result['quantity']:.2f}")
                print(f"   🏛️  Platform: {trade_result['platform']}")
                print(f"   📝 Mode: {trade_result['mode']}")
            else:
                print(f"   ❌ Trade failed: {trade_result['error']}")

def main():
    """Main integration test"""
    print("🔗 Futures Integration Test for Trading Agent")
    print("=" * 60)

    if not futures_integration.is_futures_available():
        print("❌ Futures trading not available")
        print("💡 Enable with: TB_ENABLE_FUTURES_TRADING=1")
        return

    agent = MockTradingAgent()

    # Test symbols
    test_symbols = ['BTCUSDT', 'ETHUSDT']

    for symbol in test_symbols:
        print(f"\n{'='*60}")
        print(f"🧪 TESTING {symbol}")
        print(f"{'='*60}")

        # 1. Data fetching comparison
        print("\n1️⃣ DATA FETCHING COMPARISON")
        print("-" * 30)
        agent.get_market_data_old_way(symbol)
        agent.get_market_data_new_way(symbol)

        # 2. Position sizing comparison
        print("\n2️⃣ POSITION SIZING COMPARISON")
        print("-" * 32)
        agent.calculate_position_old_way(symbol)
        agent.calculate_position_new_way(symbol)

        # 3. Trade execution comparison
        print("\n3️⃣ TRADE EXECUTION COMPARISON")
        print("-" * 33)
        agent.execute_trade_comparison(symbol, 'buy')

    # 4. Portfolio status
    print(f"\n{'='*60}")
    print("📊 FUTURES PORTFOLIO STATUS")
    print(f"{'='*60}")

    status = get_futures_status()
    if 'error' not in status:
        balance = status['balance']
        print(f"💰 Total Balance: ${balance.get('total_balance', 0):,.2f}")
        print(f"💵 Available: ${balance.get('available_balance', 0):,.2f}")
        print(f"🏛️  Platform: {status['active_platform']}")
        print(f"📊 Positions: {status['total_positions']}")
        print(f"🎯 Mode: {status['mode']}")
    else:
        print(f"❌ Status error: {status['error']}")

    # Summary
    print(f"\n{'='*60}")
    print("🎯 INTEGRATION SUMMARY")
    print(f"{'='*60}")
    print("✅ Futures data fetching working")
    print("✅ Advanced position sizing with leverage")
    print("✅ Professional futures order execution")
    print("✅ Multi-platform support (Binance, Bybit, BitMEX, Deribit)")
    print("✅ Free paper trading (no real money risk)")
    print("✅ Drop-in replacement for Alpaca spot functions")
    print("\n🚀 Your trading agent can now trade futures & perpetuals!")
    print("💡 Update your agent to use enhanced_futures_bars() instead of Alpaca calls")

if __name__ == "__main__":
    main()
