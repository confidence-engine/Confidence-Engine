#!/usr/bin/env python3
"""
Futures & Perpetuals Paper Trading Demo
Demonstrates advanced crypto derivatives trading capabilities
"""

import os
import sys
import time
import json
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List
import logging

# Import our futures system
from futures_integration import (
    futures_integration,
    enhanced_futures_bars,
    calculate_futures_position,
    execute_futures_trade,
    get_futures_status
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FuturesPaperTradingDemo:
    """Demo class for futures paper trading"""

    def __init__(self):
        self.capital = 100000.0  # $100k demo capital
        self.risk_per_trade = 0.02  # 2% risk per trade
        self.symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'ADAUSDT', 'DOTUSDT']
        self.trades = []

    def run_demo(self):
        """Run the complete futures trading demo"""
        print("🚀 Futures & Perpetuals Paper Trading Demo")
        print("=" * 60)
        print(f"💰 Starting Capital: ${self.capital:,.2f}")
        print(f"🎯 Risk per Trade: {self.risk_per_trade*100}%")
        print(f"📊 Symbols: {', '.join(self.symbols)}")
        print()

        if not futures_integration.is_futures_available():
            print("❌ Futures trading not available")
            print("💡 Enable with: TB_ENABLE_FUTURES_TRADING=1")
            return

        # 1. Test data availability
        self.test_data_availability()

        # 2. Demonstrate position sizing
        self.demonstrate_position_sizing()

        # 3. Simulate trading strategy
        self.simulate_trading_strategy()

        # 4. Show portfolio status
        self.show_portfolio_status()

        # 5. Multi-asset futures strategy
        self.multi_asset_futures_strategy()

    def test_data_availability(self):
        """Test data availability across symbols"""
        print("📊 Testing Data Availability")
        print("-" * 30)

        for symbol in self.symbols:
            print(f"🔍 Testing {symbol}...")
            data = enhanced_futures_bars(symbol, '1h', 10)

            if data is not None and len(data) > 0:
                latest = data.iloc[-1]
                print(f"  ✅ {len(data)} bars | Latest: ${latest['close']:.2f}")
            else:
                print("  ❌ No data available")
        print()

    def demonstrate_position_sizing(self):
        """Demonstrate advanced position sizing"""
        print("📏 Advanced Position Sizing Demo")
        print("-" * 35)

        for symbol in self.symbols[:3]:  # Test first 3 symbols
            print(f"🎯 Calculating position for {symbol}...")

            pos_info = calculate_futures_position(symbol, self.capital, self.risk_per_trade)

            if 'error' not in pos_info:
                print(f"  💰 Position Value: ${pos_info['position_value']:,.2f}")
                print(f"  ⚡ Leverage: {pos_info['leverage_used']:.1f}x")
                print(f"  📊 Risk Amount: ${pos_info['risk_amount']:,.2f}")
                print(f"  📈 Volatility: {pos_info['volatility']:.4f}")
                print(f"  🎲 Kelly %: {pos_info['kelly_pct']:.4f}")
            else:
                print(f"  ❌ Error: {pos_info['error']}")

            print()

    def simulate_trading_strategy(self):
        """Simulate a simple momentum-based trading strategy"""
        print("📈 Simulating Futures Trading Strategy")
        print("-" * 40)

        for symbol in self.symbols[:2]:  # Test with BTC and ETH
            print(f"🔄 Trading {symbol}...")

            # Get recent data
            data = enhanced_futures_bars(symbol, '1h', 50)
            if data is None or len(data) < 20:
                print("  ❌ Insufficient data")
                continue

            # Simple momentum strategy
            returns = data['close'].pct_change()
            recent_momentum = returns.tail(5).mean()

            if recent_momentum > 0.005:  # Bullish momentum
                side = 'buy'
                signal = 'bullish'
            elif recent_momentum < -0.005:  # Bearish momentum
                side = 'sell'
                signal = 'bearish'
            else:
                print("  ⏸️  No clear signal")
                continue

            # Calculate position
            pos_info = calculate_futures_position(symbol, self.capital, self.risk_per_trade)
            if 'error' in pos_info:
                print(f"  ❌ Position calculation failed: {pos_info['error']}")
                continue

            # Execute trade
            trade_result = execute_futures_trade(symbol, side, pos_info)

            if 'error' not in trade_result:
                print(f"  ✅ {signal.upper()} trade executed")
                print(f"     Order ID: {trade_result['order_id']}")
                print(f"     Quantity: {trade_result['quantity']:.6f}")
                print(f"     Leverage: {trade_result['leverage']}x")
                print(f"     Status: {trade_result['status']}")

                # Record trade
                self.trades.append({
                    'symbol': symbol,
                    'side': side,
                    'signal': signal,
                    'momentum': recent_momentum,
                    'position_info': pos_info,
                    'trade_result': trade_result,
                    'timestamp': datetime.now().isoformat()
                })
            else:
                print(f"  ❌ Trade failed: {trade_result['error']}")

            print()

    def show_portfolio_status(self):
        """Show current portfolio status"""
        print("📊 Portfolio Status")
        print("-" * 20)

        status = get_futures_status()

        if 'error' not in status:
            balance = status['balance']
            print(f"💰 Total Balance: ${balance.get('total_balance', 0):,.2f}")
            print(f"💵 Available: ${balance.get('available_balance', 0):,.2f}")
            print(f"🔗 Used Margin: ${balance.get('used_margin', 0):,.2f}")
            print(f"📈 Unrealized P&L: ${balance.get('unrealized_pnl', 0):,.2f}")
            print(f"🏛️  Platform: {status['active_platform']}")
            print(f"📊 Total Positions: {status['total_positions']}")
        else:
            print(f"❌ Status check failed: {status['error']}")

        print()

    def multi_asset_futures_strategy(self):
        """Demonstrate multi-asset futures strategy"""
        print("🌐 Multi-Asset Futures Strategy")
        print("-" * 35)

        # Get data for all symbols
        portfolio_data = {}
        for symbol in self.symbols:
            data = enhanced_futures_bars(symbol, '1h', 20)
            if data is not None and len(data) > 0:
                portfolio_data[symbol] = data

        if not portfolio_data:
            print("❌ No portfolio data available")
            return

        print(f"📊 Analyzing {len(portfolio_data)} assets...")

        # Calculate correlations and momentum
        momentum_scores = {}
        for symbol, data in portfolio_data.items():
            returns = data['close'].pct_change().dropna()
            momentum = returns.tail(10).mean()
            volatility = returns.tail(10).std()
            momentum_scores[symbol] = {
                'momentum': momentum,
                'volatility': volatility,
                'score': momentum / volatility if volatility > 0 else 0
            }

        # Rank by momentum score
        ranked_assets = sorted(momentum_scores.items(),
                              key=lambda x: x[1]['score'], reverse=True)

        print("\n🏆 Asset Momentum Ranking:")
        for i, (symbol, scores) in enumerate(ranked_assets[:5], 1):
            print(f"  {i}. {symbol}: Score {scores['score']:.4f} "
                  f"(Mom: {scores['momentum']:.4f}, Vol: {scores['volatility']:.4f})")

        # Suggest portfolio allocation
        top_assets = ranked_assets[:3]
        total_score = sum(scores['score'] for _, scores in top_assets)

        print("\n💼 Suggested Portfolio Allocation:")
        for symbol, scores in top_assets:
            weight = scores['score'] / total_score if total_score > 0 else 1/3
            position_value = self.capital * weight * 0.5  # 50% allocation to futures
            print(f"  📈 {symbol}: {weight:.1%} (${position_value:,.2f})")

        print()

def main():
    """Main demo function"""
    # Set demo environment variables if not set
    if 'TB_ENABLE_FUTURES_TRADING' not in os.environ:
        os.environ['TB_ENABLE_FUTURES_TRADING'] = '1'
        os.environ['TB_FUTURES_PLATFORM'] = 'binance'
        os.environ['TB_MAX_LEVERAGE'] = '10'
        os.environ['TB_PAPER_TRADING'] = '1'

    # Run the demo
    demo = FuturesPaperTradingDemo()
    demo.run_demo()

    # Summary
    print("🎯 Demo Summary")
    print("=" * 15)
    print(f"✅ Tested {len(demo.symbols)} crypto assets")
    print(f"✅ Executed {len(demo.trades)} simulated trades")
    print("✅ Demonstrated position sizing & risk management")
    print("✅ Showed multi-asset portfolio strategy")
    print("\n🚀 Ready for live futures paper trading!")
    print("💡 Configure your .env file with futures settings")

if __name__ == "__main__":
    main()
