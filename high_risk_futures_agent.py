#!/usr/bin/env python3
"""
High-Risk Futures Agent
Se        # Strategy parameters
        self.momentum_window = 12  # hours
        self.volatility_window = 24  # hours
        self.min_momentum_threshold = 0.02  # 2% momentum
        self.max_volatility_threshold = 0.08    # 8% max volatility

        # New: Market regime and correlation tracking
        self.market_regime = 'unknown'
        self.correlation_matrix = {}
        self.trailing_stops = {}  # Track trailing stop levels

        logger.info(f"🚀 {self.name} initialized")
        logger.info(f"💰 Capital: ${self.capital}")
        logger.info(f"⚡ Max Leverage: {self.max_leverage}x")
        logger.info(f"🎯 Risk per Trade: {self.risk_per_trade*100}%")
        logger.info(f"📊 Symbols: {', '.join(self.symbols)}") main agent - focused on leveraged futures/perpetuals trading
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

    def detect_market_regime(self, symbol: str) -> str:
        """Detect if market is trending or ranging"""
        try:
            # Get longer-term data for regime detection
            data = enhanced_futures_bars(symbol, '1h', 48)  # 48 hours of data
            if data is None or len(data) < 24:
                return 'unknown'

            prices = data['close']

            # Calculate trend strength (slope of linear regression)
            x = np.arange(len(prices))
            slope, _ = np.polyfit(x, prices.values, 1)
            trend_strength = abs(slope) / prices.mean()

            # Calculate volatility (standard deviation of returns)
            returns = prices.pct_change().dropna()
            volatility = returns.std()

            # Calculate ADX-like indicator for trend strength
            high_prices = data['high']
            low_prices = data['low']

            # Simple trend detection
            if trend_strength > 0.001 and volatility < 0.03:  # Strong trend, low volatility
                return 'trending'
            elif trend_strength < 0.0005 and volatility > 0.05:  # Weak trend, high volatility
                return 'ranging'
            else:
                return 'sideways'

        except Exception as e:
            logger.warning(f"Error detecting market regime for {symbol}: {e}")
            return 'unknown'

    def calculate_symbol_correlations(self) -> Dict[str, Dict[str, float]]:
        """Calculate correlation matrix between all symbols"""
        try:
            correlations = {}
            symbol_data = {}

            # Get data for all symbols
            for symbol in self.symbols:
                data = enhanced_futures_bars(symbol, '1h', 24)  # 24 hours
                if data is not None and len(data) > 12:
                    symbol_data[symbol] = data['close'].pct_change().dropna()

            # Calculate correlations
            for symbol1 in self.symbols:
                if symbol1 not in symbol_data:
                    continue
                correlations[symbol1] = {}

                for symbol2 in self.symbols:
                    if symbol2 not in symbol_data:
                        correlations[symbol1][symbol2] = 0.0
                        continue

                    try:
                        corr = symbol_data[symbol1].corr(symbol_data[symbol2])
                        correlations[symbol1][symbol2] = corr if not np.isnan(corr) else 0.0
                    except:
                        correlations[symbol1][symbol2] = 0.0

            return correlations

        except Exception as e:
            logger.warning(f"Error calculating correlations: {e}")
            return {}

    def check_correlation_filter(self, symbol: str) -> bool:
        """Check if symbol is too correlated with existing positions"""
        if not self.positions:
            return True  # No positions, so no correlation issue

        # Update correlation matrix if needed
        if not self.correlation_matrix:
            self.correlation_matrix = self.calculate_symbol_correlations()

        # Check correlation with existing positions
        for existing_symbol in self.positions.keys():
            if existing_symbol in self.correlation_matrix and symbol in self.correlation_matrix[existing_symbol]:
                correlation = abs(self.correlation_matrix[existing_symbol][symbol])
                if correlation > 0.7:  # High correlation threshold
                    logger.info(f"⚠️ Skipping {symbol} due to high correlation ({correlation:.2f}) with {existing_symbol}")
                    return False

        return True

    def calculate_dynamic_leverage(self, symbol: str, volatility: float) -> int:
        """Calculate dynamic leverage based on volatility"""
        try:
            # Base leverage
            base_leverage = self.max_leverage

            # Adjust based on volatility
            if volatility > 0.06:  # High volatility
                adjusted_leverage = max(1, int(base_leverage * 0.6))  # Reduce to 60%
            elif volatility > 0.04:  # Medium volatility
                adjusted_leverage = max(1, int(base_leverage * 0.8))  # Reduce to 80%
            else:  # Low volatility
                adjusted_leverage = base_leverage  # Use full leverage

            # Adjust based on market regime
            regime = self.detect_market_regime(symbol)
            if regime == 'ranging':
                adjusted_leverage = max(1, int(adjusted_leverage * 0.7))  # Reduce in ranging markets
            elif regime == 'trending':
                adjusted_leverage = min(self.max_leverage, int(adjusted_leverage * 1.2))  # Increase in trending markets

            return min(adjusted_leverage, self.max_leverage)

        except Exception as e:
            logger.warning(f"Error calculating dynamic leverage for {symbol}: {e}")
            return self.max_leverage
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

        # NEW: Check correlation filter
        if not self.check_correlation_filter(symbol):
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

            # Apply leverage limits with dynamic adjustment
            volatility = signal.get('volatility', 0.05)
            dynamic_leverage = self.calculate_dynamic_leverage(symbol, volatility)
            leverage = min(pos_info['leverage_used'], dynamic_leverage)
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
                    'signal': signal,
                    'highest_price': float(entry_price),  # For trailing stops
                    'lowest_price': float(entry_price),   # For trailing stops
                    'trailing_stop_pct': 0.05,  # 5% trailing stop
                    'profit_target_pct': 0.08   # 8% profit target
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
        """Check and manage open positions with advanced exit timing"""
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

                side = position.get('side', 'buy')
                leverage = position.get('leverage', 1) or 1

                # Calculate current P&L
                if side == 'buy':
                    pnl_pct = (current_price - entry_price) / entry_price
                else:
                    pnl_pct = (entry_price - current_price) / entry_price

                # Update trailing stop levels
                self.update_trailing_stops(symbol, current_price, position)

                # Check exit conditions
                exit_reason = self.should_exit_position(symbol, current_price, pnl_pct, position)

                if exit_reason:
                    self.close_position(symbol, exit_reason)

            except Exception as e:
                logger.warning(f"Error checking position for {symbol}: {e}")

    def update_trailing_stops(self, symbol: str, current_price: float, position: Dict):
        """Update trailing stop levels for a position"""
        try:
            side = position.get('side', 'buy')

            if side == 'buy':
                # For long positions, track highest price
                if current_price > position.get('highest_price', position['entry_price']):
                    position['highest_price'] = current_price
                    # Update trailing stop level
                    trailing_stop_price = current_price * (1 - position.get('trailing_stop_pct', 0.05))
                    position['trailing_stop_price'] = trailing_stop_price
            else:
                # For short positions, track lowest price
                if current_price < position.get('lowest_price', position['entry_price']):
                    position['lowest_price'] = current_price
                    # Update trailing stop level
                    trailing_stop_price = current_price * (1 + position.get('trailing_stop_pct', 0.05))
                    position['trailing_stop_price'] = trailing_stop_price

        except Exception as e:
            logger.warning(f"Error updating trailing stops for {symbol}: {e}")

    def should_exit_position(self, symbol: str, current_price: float, pnl_pct: float, position: Dict) -> str:
        """Determine if position should be exited based on various conditions"""
        try:
            side = position.get('side', 'buy')
            entry_price = position.get('entry_price', 0)
            profit_target_pct = position.get('profit_target_pct', 0.08)
            trailing_stop_pct = position.get('trailing_stop_pct', 0.05)

            # 1. Profit target hit
            if pnl_pct >= profit_target_pct:
                return 'profit_target'

            # 2. Stop loss hit (fixed percentage from entry)
            if pnl_pct <= -0.03:  # 3% stop loss
                return 'stop_loss'

            # 3. Trailing stop hit
            if 'trailing_stop_price' in position:
                trailing_stop_price = position['trailing_stop_price']
                if side == 'buy' and current_price <= trailing_stop_price:
                    return 'trailing_stop'
                elif side == 'sell' and current_price >= trailing_stop_price:
                    return 'trailing_stop'

            # 4. Maximum loss limit (10% from entry)
            if pnl_pct <= -0.10:
                return 'max_loss_limit'

            # 5. Time-based exit (if position is too old)
            position_age_hours = (datetime.now() - datetime.fromisoformat(position['timestamp'])).total_seconds() / 3600
            if position_age_hours > 24:  # Close after 24 hours
                return 'time_limit'

            # 6. Volatility-based exit (if volatility spikes)
            try:
                data = enhanced_futures_bars(symbol, '1h', 6)  # Last 6 hours
                if data is not None and len(data) >= 6:
                    recent_volatility = data['close'].pct_change().std()
                    if recent_volatility > 0.08:  # High volatility
                        return 'high_volatility'
            except:
                pass

            return None  # No exit condition met

        except Exception as e:
            logger.warning(f"Error checking exit conditions for {symbol}: {e}")
            return None

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
        """Get agent status with enhanced information"""
        return {
            'name': self.name,
            'capital': self.capital,
            'daily_pnl': self.daily_pnl,
            'trades_today': self.trades_today,
            'open_positions': len(self.positions),
            'positions': list(self.positions.keys()),
            'total_trades': len(self.trade_log),
            'win_rate': self.calculate_win_rate(),
            'market_regime': self.market_regime,
            'correlation_pairs': len(self.correlation_matrix) if self.correlation_matrix else 0,
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
        """Run one complete trading cycle with enhanced features"""
        logger.info("🔄 Starting trading cycle...")

        # Update market regime and correlations
        self.update_market_context()

        # Check existing positions
        self.check_positions()

        # Look for new trades
        for symbol in self.symbols:
            if symbol in self.positions:
                continue  # Skip if we already have position

            signal = self.calculate_momentum_signal(symbol)

            if signal['signal'] != 'neutral' and self.should_trade(symbol, signal):
                # Log additional context
                regime = self.detect_market_regime(symbol)
                logger.info(f"🎯 Signal detected for {symbol}: {signal['signal']} ({signal['strength']:.4f})")
                logger.info(f"📊 Market regime: {regime} | Volatility: {signal.get('volatility', 0):.4f}")
                self.execute_trade(symbol, signal)

                # Limit to one trade per cycle
                break

        # Log status
        status = self.get_status()
        logger.info(f"📊 Status: ${status['daily_pnl']:.2f} P&L | {status['open_positions']} positions | {status['trades_today']} trades today")

    def update_market_context(self):
        """Update market regime and correlation data"""
        try:
            # Update correlation matrix periodically
            if not self.correlation_matrix or np.random.random() < 0.1:  # 10% chance each cycle
                self.correlation_matrix = self.calculate_symbol_correlations()
                logger.info("📈 Updated correlation matrix")

            # Update market regime for primary symbol
            if self.symbols:
                primary_symbol = self.symbols[0]
                self.market_regime = self.detect_market_regime(primary_symbol)
                logger.info(f"🌍 Market regime: {self.market_regime}")

        except Exception as e:
            logger.warning(f"Error updating market context: {e}")

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
    print(f"🌍 Market Regime: {status['market_regime']}")
    print(f"📊 Correlation Pairs: {status['correlation_pairs']}")

    print("\n✅ Enhanced High-Risk Futures Agent Ready!")
    print("🚀 New Features:")
    print("  • Market regime detection (trending/ranging)")
    print("  • Correlation filtering to avoid correlated positions")
    print("  • Dynamic leverage adjustment based on volatility")
    print("  • Advanced exit timing with trailing stops")
    print("  • Profit targets and time-based exits")
    print("\n💡 Run with: python3 high_risk_futures_agent.py --continuous")

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
