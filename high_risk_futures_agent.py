#!/usr/bin/env python3
"""
High-Risk Futures Agent
Separate from main agent - focused on leveraged futures/perpetuals trading
"""

import os
import sys
import time
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging
import asyncio

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Import futures platform
from futures_integration import (
    futures_integration,
    enhanced_futures_bars,
    calculate_futures_position,
    execute_futures_trade,
    get_futures_status
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HighRiskFuturesAgent:
    """High-risk futures trading agent - separate from main agent"""

    def __init__(self):
        self.name = "High-Risk Futures Agent"
        self.capital = float(os.getenv("FUTURES_AGENT_CAPITAL", "10000"))  # $10k starting capital
        self.max_leverage = int(os.getenv("FUTURES_MAX_LEVERAGE", "25"))  # High risk = high leverage
        self.risk_per_trade = float(os.getenv("FUTURES_RISK_PER_TRADE", "0.05"))  # 5% risk per trade
        self.max_daily_loss = float(os.getenv("FUTURES_MAX_DAILY_LOSS", "0.20"))  # 20% max daily loss
        self.symbols = os.getenv("FUTURES_SYMBOLS", "BTCUSDT,ETHUSDT,SOLUSDT").split(",")

        # State tracking
        self.daily_pnl = 0.0
        self.trades_today = 0
        self.positions = {}
        self.trade_log = []

        # Strategy parameters
        self.momentum_window = 12  # hours
        self.volatility_window = 24  # hours
        self.min_momentum_threshold = 0.02  # 2% momentum
        self.max_volatility_threshold = 0.08  # 8% max volatility

        logger.info(f"🚀 {self.name} initialized")
        logger.info(f"💰 Capital: ${self.capital}")
        logger.info(f"⚡ Max Leverage: {self.max_leverage}x")
        logger.info(f"🎯 Risk per Trade: {self.risk_per_trade*100}%")
        logger.info(f"📊 Symbols: {', '.join(self.symbols)}")

    def is_market_open(self) -> bool:
        """Check if futures markets are open (crypto markets are 24/7)"""
        return True  # Crypto futures are always open

    def calculate_momentum_signal(self, symbol: str) -> Dict:
        """Calculate momentum-based trading signal"""
        try:
            # Get recent data
            data = enhanced_futures_bars(symbol, '1h', self.momentum_window + 10)
            if data is None or len(data) < self.momentum_window:
                return {'signal': 'neutral', 'strength': 0, 'reason': 'insufficient_data'}

            # Calculate momentum
            prices = data['close']
            momentum = (prices.iloc[-1] - prices.iloc[-self.momentum_window]) / prices.iloc[-self.momentum_window]

            # Calculate volatility
            returns = prices.pct_change().dropna()
            volatility = returns.std() * np.sqrt(24)  # Annualized daily volatility

            # Signal logic
            if momentum > self.min_momentum_threshold and volatility < self.max_volatility_threshold:
                return {
                    'signal': 'buy',
                    'strength': abs(momentum),
                    'momentum': momentum,
                    'volatility': volatility,
                    'reason': f'momentum_{momentum:.4f}_vol_{volatility:.4f}'
                }
            elif momentum < -self.min_momentum_threshold and volatility < self.max_volatility_threshold:
                return {
                    'signal': 'sell',
                    'strength': abs(momentum),
                    'momentum': momentum,
                    'volatility': volatility,
                    'reason': f'momentum_{momentum:.4f}_vol_{volatility:.4f}'
                }
            else:
                return {
                    'signal': 'neutral',
                    'strength': 0,
                    'momentum': momentum,
                    'volatility': volatility,
                    'reason': 'weak_signal_or_high_volatility'
                }

        except Exception as e:
            logger.warning(f"Error calculating momentum for {symbol}: {e}")
            return {'signal': 'neutral', 'strength': 0, 'reason': 'calculation_error'}

    def should_trade(self, symbol: str, signal: Dict) -> bool:
        """Determine if we should execute a trade"""
        # Check daily loss limit
        if self.daily_pnl < -self.max_daily_loss * self.capital:
            logger.warning(f"🚫 Daily loss limit reached: ${self.daily_pnl:.2f}")
            return False

        # Check if we already have a position in this symbol
        if symbol in self.positions:
            logger.info(f"📊 Already have position in {symbol}")
            return False

        # Check signal strength
        if signal['strength'] < self.min_momentum_threshold:
            return False

        # Check market conditions
        if not self.is_market_open():
            return False

        return True

    def execute_trade(self, symbol: str, signal: Dict) -> bool:
        """Execute a futures trade"""
        try:
            side = signal['signal']

            # Calculate position size
            pos_info = calculate_futures_position(
                symbol,
                self.capital,
                self.risk_per_trade
            )

            if 'error' in pos_info:
                logger.warning(f"❌ Position calculation failed: {pos_info['error']}")
                return False

            # Apply leverage limits
            leverage = min(pos_info['leverage_used'], self.max_leverage)
            pos_info['leverage_used'] = leverage

            # Execute trade
            trade_result = execute_futures_trade(symbol, side, pos_info)

            if 'error' not in trade_result:
                # Record position
                entry_price = trade_result.get('price', 0) or 0
                quantity = trade_result.get('quantity', 0) or 0

                self.positions[symbol] = {
                    'side': side,
                    'entry_price': float(entry_price),
                    'quantity': float(quantity),
                    'leverage': leverage,
                    'timestamp': datetime.now().isoformat(),
                    'signal': signal
                }

                # Log trade
                trade_record = {
                    'timestamp': datetime.now().isoformat(),
                    'symbol': symbol,
                    'side': side,
                    'quantity': float(quantity),
                    'price': float(entry_price),
                    'leverage': leverage,
                    'order_id': trade_result['order_id'],
                    'signal_strength': signal['strength'],
                    'reason': signal['reason']
                }
                self.trade_log.append(trade_record)

                self.trades_today += 1

                logger.info(f"✅ {side.upper()} {symbol} x{leverage} @ ${entry_price:.2f}")
                return True
            else:
                logger.warning(f"❌ Trade execution failed: {trade_result['error']}")
                return False

        except Exception as e:
            logger.error(f"Error executing trade for {symbol}: {e}")
            return False

    def check_positions(self):
        """Check and manage open positions"""
        for symbol, position in list(self.positions.items()):
            try:
                # Get current price
                data = enhanced_futures_bars(symbol, '1h', 1)
                if data is None or len(data) == 0:
                    continue

                current_price = data['close'].iloc[-1]
                if current_price is None:
                    continue

                entry_price = position.get('entry_price', 0)
                if entry_price is None or entry_price == 0:
                    continue

                # Calculate P&L
                if position['side'] == 'buy':
                    pnl_pct = (current_price - entry_price) / entry_price
                else:
                    pnl_pct = (entry_price - current_price) / entry_price

                quantity = position.get('quantity', 0) or 0
                leverage = position.get('leverage', 1) or 1
                pnl_amount = pnl_pct * quantity * leverage

                # Simple exit conditions
                if pnl_pct > 0.05:  # 5% profit target
                    logger.info(f"🎯 Profit target hit for {symbol}: +{pnl_pct:.2%}")
                    self.close_position(symbol, 'profit_target')
                elif pnl_pct < -0.03:  # 3% stop loss
                    logger.warning(f"🛑 Stop loss hit for {symbol}: {pnl_pct:.2%}")
                    self.close_position(symbol, 'stop_loss')
                elif pnl_pct > 0.10:  # 10% trailing stop
                    logger.info(f"📈 Trailing stop for {symbol}: +{pnl_pct:.2%}")
                    self.close_position(symbol, 'trailing_stop')

            except Exception as e:
                logger.warning(f"Error checking position for {symbol}: {e}")

    def close_position(self, symbol: str, reason: str):
        """Close a position"""
        if symbol not in self.positions:
            return

        position = self.positions[symbol]

        try:
            # Get current price for exit
            data = enhanced_futures_bars(symbol, '1h', 1)
            if data is not None and len(data) > 0:
                exit_price = data['close'].iloc[-1]
                if exit_price is None:
                    exit_price = position.get('entry_price', 0)
            else:
                exit_price = position.get('entry_price', 0)

            entry_price = position.get('entry_price', 0) or 0
            quantity = position.get('quantity', 0) or 0
            leverage = position.get('leverage', 1) or 1

            # Calculate final P&L
            if position['side'] == 'buy':
                pnl_pct = (exit_price - entry_price) / entry_price if entry_price > 0 else 0
            else:
                pnl_pct = (entry_price - exit_price) / exit_price if exit_price > 0 else 0

            pnl_amount = pnl_pct * quantity * leverage
            self.daily_pnl += pnl_amount

            logger.info(f"🔄 Closed {symbol} position: {reason}")
            logger.info(f"   Entry: ${entry_price:.2f} | Exit: ${exit_price:.2f}")
            logger.info(f"   P&L: ${pnl_amount:.2f} ({pnl_pct:.2%})")

            # Record exit
            exit_record = {
                'timestamp': datetime.now().isoformat(),
                'symbol': symbol,
                'action': 'close',
                'reason': reason,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pnl_amount': pnl_amount,
                'pnl_pct': pnl_pct
            }
            self.trade_log.append(exit_record)

            # Remove position
            del self.positions[symbol]

        except Exception as e:
            logger.error(f"Error closing position for {symbol}: {e}")

    def get_status(self) -> Dict:
        """Get agent status"""
        return {
            'name': self.name,
            'capital': self.capital,
            'daily_pnl': self.daily_pnl,
            'trades_today': self.trades_today,
            'open_positions': len(self.positions),
            'positions': list(self.positions.keys()),
            'total_trades': len(self.trade_log),
            'win_rate': self.calculate_win_rate(),
            'timestamp': datetime.now().isoformat()
        }

    def calculate_win_rate(self) -> float:
        """Calculate win rate from closed trades"""
        closed_trades = [t for t in self.trade_log if t.get('action') == 'close']
        if not closed_trades:
            return 0.0

        winning_trades = sum(1 for t in closed_trades if t['pnl_amount'] > 0)
        return winning_trades / len(closed_trades)

    def run_trading_cycle(self):
        """Run one complete trading cycle"""
        logger.info("🔄 Starting trading cycle...")

        # Check existing positions
        self.check_positions()

        # Look for new trades
        for symbol in self.symbols:
            if symbol in self.positions:
                continue  # Skip if we already have position

            signal = self.calculate_momentum_signal(symbol)

            if signal['signal'] != 'neutral' and self.should_trade(symbol, signal):
                logger.info(f"🎯 Signal detected for {symbol}: {signal['signal']} ({signal['strength']:.4f})")
                self.execute_trade(symbol, signal)

                # Limit to one trade per cycle
                break

        # Log status
        status = self.get_status()
        logger.info(f"📊 Status: ${status['daily_pnl']:.2f} P&L | {status['open_positions']} positions | {status['trades_today']} trades today")

    async def run_continuous(self, interval_seconds: int = 300):
        """Run continuous trading loop"""
        logger.info(f"🚀 Starting continuous futures trading (interval: {interval_seconds}s)")

        while True:
            try:
                self.run_trading_cycle()
                await asyncio.sleep(interval_seconds)

            except KeyboardInterrupt:
                logger.info("🛑 Stopping continuous trading...")
                break
            except Exception as e:
                logger.error(f"Error in trading cycle: {e}")
                await asyncio.sleep(60)  # Wait before retry

def main():
    """Main function"""
    print("🚀 High-Risk Futures Agent")
    print("=" * 50)

    if not futures_integration.is_futures_available():
        print("❌ Futures trading not available")
        print("💡 Enable with: TB_ENABLE_FUTURES_TRADING=1")
        return

    agent = HighRiskFuturesAgent()

    # Run a few test cycles
    print("\n🧪 Running test cycles...")
    for i in range(3):
        print(f"\n--- Cycle {i+1} ---")
        agent.run_trading_cycle()
        time.sleep(2)  # Brief pause between cycles

    # Show final status
    print("\n📊 Final Status:")
    status = agent.get_status()
    print(f"💰 Daily P&L: ${status['daily_pnl']:.2f}")
    print(f"📊 Trades Today: {status['trades_today']}")
    print(f"📈 Open Positions: {status['open_positions']}")
    print(f"🎯 Win Rate: {status['win_rate']:.1%}")
    print(f"📝 Total Trades: {status['total_trades']}")

    print("\n✅ High-risk futures agent ready!")
    print("💡 Run with: python3 high_risk_futures_agent.py --continuous")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='High-Risk Futures Agent')
    parser.add_argument('--continuous', action='store_true', help='Run continuous trading loop')
    parser.add_argument('--interval', type=int, default=300, help='Trading cycle interval in seconds')

    args = parser.parse_args()

    if args.continuous:
        asyncio.run(HighRiskFuturesAgent().run_continuous(args.interval))
    else:
        main()
