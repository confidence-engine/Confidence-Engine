import os
import sys
import time
import logging
import json
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import torch

# Ensure project root on sys.path
_THIS_DIR = Path(__file__).parent
_PROJ_ROOT = _THIS_DIR.parent
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

try:
    from alpaca_trade_api.rest import REST
    ALPACA_AVAILABLE = True
except Exception:
    ALPACA_AVAILABLE = False
    REST = None

# Import our enhanced components
from advanced_risk_manager import AdvancedRiskManager
from market_regime_detector import MarketRegimeDetector
from ensemble_ml_models import TradingEnsemble, SpecializedModels
from advanced_entry_exit_logic import AdvancedEntryLogic
from adaptive_strategy import AdaptiveStrategy

from config import settings
from telegram_bot import send_message as send_telegram
from scripts.discord_sender import send_discord_digest_to

# ==========================
# Enhanced Multi-Asset Trading Agent
# ==========================

@dataclass
class AssetConfig:
    """Configuration for each tradable asset"""
    symbol: str
    base_currency: str
    quote_currency: str
    min_position_size: float
    max_position_size: float
    commission_rate: float
    enabled: bool = True

class EnhancedTradingAgent:
    """
    Best-performing multi-asset trading agent with all advanced features
    """

    def __init__(self):
        # Core components
        self.risk_manager = AdvancedRiskManager()
        self.regime_detector = MarketRegimeDetector()
        self.ensemble_model = TradingEnsemble(input_dim=37)  # Match your feature count
        self.entry_logic = AdvancedEntryLogic(self.ensemble_model, self.risk_manager)
        self.adaptive_strategy = AdaptiveStrategy()

        # Asset configurations
        self.assets = self._initialize_assets()

        # State management
        self.positions = {}  # Current positions by asset
        self.portfolio_value = 100000.0  # Starting portfolio value
        self.cash = 100000.0

        # Performance tracking
        self.trade_history = []
        self.daily_pnl = []

        # API connection
        self.api = self._initialize_api()

        # Logging
        self.logger = logging.getLogger("enhanced_trader")
        self._setup_logging()

    def _initialize_assets(self) -> Dict[str, AssetConfig]:
        """Initialize supported assets - CONFIRMED ALPACA SUPPORT"""
        return {
            'BTC/USD': AssetConfig(
                symbol='BTC/USD',
                base_currency='BTC',
                quote_currency='USD',
                min_position_size=0.001,
                max_position_size=1.0,
                commission_rate=0.001,  # 0.1%
                enabled=True
            ),
            'ETH/USD': AssetConfig(
                symbol='ETH/USD',
                base_currency='ETH',
                quote_currency='USD',
                min_position_size=0.01,
                max_position_size=10.0,
                commission_rate=0.001,
                enabled=True
            ),
            'SOL/USD': AssetConfig(
                symbol='SOL/USD',
                base_currency='SOL',
                quote_currency='USD',
                min_position_size=0.1,
                max_position_size=100.0,
                commission_rate=0.001,
                enabled=True
            ),
            # ADA/USD is NOT supported by Alpaca paper trading
            # SOL/USD and LINK/USD are supported ✅
            # 'ADA/USD': AssetConfig(...),  # ❌ UNAVAILABLE
            
            'LINK/USD': AssetConfig(
                symbol='LINK/USD',
                base_currency='LINK',
                quote_currency='USD',
                min_position_size=0.1,
                max_position_size=100.0,
                commission_rate=0.001,
                enabled=True
            )
        }

    def _initialize_api(self) -> Optional[Any]:
        """Initialize Alpaca API connection"""
        try:
            if os.getenv("TB_TRADER_OFFLINE", "1") == "1":
                return None

            if not ALPACA_AVAILABLE:
                return None

            return REST(
                key_id=settings.alpaca_key_id,
                secret_key=settings.alpaca_secret_key,
                base_url=settings.alpaca_base_url,
            )
        except Exception as e:
            self.logger.error(f"Failed to initialize API: {e}")
            return None

    def _setup_logging(self):
        """Setup comprehensive logging"""
        self.logger.setLevel(logging.INFO)

        # File handler
        fh = logging.FileHandler("enhanced_trading_agent.log")
        fh.setLevel(logging.INFO)

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

    def run_trading_cycle(self):
        """Main trading cycle - called periodically"""
        try:
            self.logger.info("Starting enhanced trading cycle")

            # 1. Update market data
            market_data = self._fetch_market_data()

            # 2. Detect market regimes
            regimes = self.regime_detector.classify_regime(market_data)

            # 3. Update correlation matrix
            self.risk_manager.update_correlation_matrix(market_data)

            # 4. Get sentiment data
            sentiment_data = self._fetch_sentiment_data()

            # 5. Evaluate trading opportunities
            opportunities = self._evaluate_opportunities(market_data, regimes, sentiment_data)

            # 6. Execute trades
            executed_trades = self._execute_trades(opportunities)

            # 7. Update portfolio
            self._update_portfolio(executed_trades)

            # 8. Check for exits
            exit_signals = self._check_exit_signals(market_data, regimes)
            exit_trades = self._execute_exits(exit_signals)

            # 9. Update performance tracking
            self._update_performance(executed_trades + exit_trades)

            # 10. Adaptive learning
            if self.adaptive_strategy.should_adapt():
                self._adapt_strategy()

            # 11. Logging and notifications
            self._log_cycle_results(executed_trades, exit_trades, regimes)

            self.logger.info("Trading cycle completed successfully")

        except Exception as e:
            self.logger.error(f"Trading cycle failed: {e}")
            # Continue running despite errors

    def _fetch_market_data(self) -> Dict[str, pd.DataFrame]:
        """Fetch market data for all enabled assets"""
        market_data = {}

        for symbol, config in self.assets.items():
            if not config.enabled:
                continue

            try:
                # Fetch 4 hours of 15-minute bars for analysis
                bars = self._fetch_asset_bars(symbol, timeframe="15Min", lookback=16)  # 4 hours
                if bars is not None and len(bars) > 0:
                    market_data[symbol] = bars

            except Exception as e:
                self.logger.warning(f"Failed to fetch data for {symbol}: {e}")

        return market_data

    def _fetch_asset_bars(self, symbol: str, timeframe: str, lookback: int) -> Optional[pd.DataFrame]:
        """Fetch bars for a specific asset"""
        if self.api is None:
            # Return synthetic data for offline testing
            return self._generate_synthetic_bars(symbol, timeframe, lookback)

        try:
            # Use your existing bar fetching logic
            from scripts.hybrid_crypto_trader import fetch_bars
            return fetch_bars(symbol, timeframe, lookback)
        except Exception as e:
            self.logger.error(f"Failed to fetch bars for {symbol}: {e}")
            return None

    def _generate_synthetic_bars(self, symbol: str, timeframe: str, lookback: int) -> pd.DataFrame:
        """Generate synthetic bars for testing"""
        # Use your existing synthetic bar generation
        from scripts.hybrid_crypto_trader import synthetic_bars
        return synthetic_bars(timeframe, lookback)

    def _fetch_sentiment_data(self) -> Dict[str, float]:
        """Fetch sentiment data for assets"""
        sentiment_data = {}

        for symbol in self.assets.keys():
            try:
                # Use your existing sentiment fetching logic
                from scripts.hybrid_crypto_trader import sentiment_via_perplexity

                # Get headlines (simplified)
                headlines = [f"Sample headline for {symbol}"]
                sentiment, _ = sentiment_via_perplexity(headlines)
                sentiment_data[symbol] = sentiment

            except Exception as e:
                self.logger.warning(f"Failed to fetch sentiment for {symbol}: {e}")
                sentiment_data[symbol] = 0.5  # Neutral sentiment

        return sentiment_data

    def _evaluate_opportunities(self,
                              market_data: Dict[str, pd.DataFrame],
                              regimes: Dict[str, Any],
                              sentiment_data: Dict[str, float]) -> List[Dict[str, Any]]:
        """Evaluate trading opportunities for all assets"""

        opportunities = []

        for symbol, data in market_data.items():
            try:
                regime = regimes.get(symbol)
                sentiment = sentiment_data.get(symbol, 0.5)

                if regime is None:
                    continue

                # Get current regime string
                regime_str = self.regime_detector.get_regime_string(regime)

                # Check entry conditions
                entry_signal = self.entry_logic.should_enter(
                    {symbol: data},
                    sentiment,
                    regime_str
                )

                if entry_signal:
                    # Calculate position size using Kelly criterion
                    current_price = data['close'].iloc[-1]
                    win_probability = self._estimate_win_probability(regime, sentiment)
                    win_loss_ratio = self._estimate_win_loss_ratio(regime)

                    position_size = self.risk_manager.calculate_kelly_size(
                        win_probability,
                        win_loss_ratio,
                        self.portfolio_value,
                        regime_str
                    )

                    # Check portfolio limits
                    limits_ok = self.risk_manager.check_portfolio_limits(
                        self.positions,
                        {s: md['close'].iloc[-1] for s, md in market_data.items()},
                        (symbol, position_size)
                    )

                    if all(limits_ok.values()):
                        opportunity = {
                            'symbol': symbol,
                            'direction': entry_signal.direction,
                            'strength': entry_signal.strength,
                            'confidence': entry_signal.confidence,
                            'position_size': position_size,
                            'entry_price': current_price,
                            'regime': regime_str,
                            'sentiment': sentiment,
                            'components': entry_signal.components
                        }
                        opportunities.append(opportunity)

            except Exception as e:
                self.logger.warning(f"Failed to evaluate {symbol}: {e}")

        return opportunities

    def _estimate_win_probability(self, regime, sentiment: float) -> float:
        """Estimate win probability based on regime and sentiment"""
        # Simple estimation - in production use historical data
        base_prob = 0.55  # Base win probability

        # Adjust for regime
        if regime.volatility.value == 'low':
            base_prob += 0.05
        elif regime.volatility.value == 'high':
            base_prob -= 0.05

        if regime.trend.value in ['up', 'strong_up']:
            base_prob += 0.03
        elif regime.trend.value in ['down', 'strong_down']:
            base_prob -= 0.03

        # Adjust for sentiment
        sentiment_adjustment = sentiment * 0.1  # ±10% based on sentiment
        base_prob += sentiment_adjustment

        return np.clip(base_prob, 0.4, 0.7)

    def _estimate_win_loss_ratio(self, regime) -> float:
        """Estimate win/loss ratio based on regime"""
        # Simple estimation - in production use historical data
        base_ratio = 2.0  # Base win/loss ratio

        # Adjust for volatility
        if regime.volatility.value == 'high':
            base_ratio *= 0.8  # Lower ratio in high vol
        elif regime.volatility.value == 'low':
            base_ratio *= 1.2  # Higher ratio in low vol

        return base_ratio

    def _execute_trades(self, opportunities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Execute approved trades"""
        executed_trades = []

        for opportunity in opportunities:
            try:
                if self.api is None:
                    # Offline mode - simulate execution
                    executed_trade = self._simulate_trade(opportunity)
                else:
                    # Live execution
                    executed_trade = self._execute_live_trade(opportunity)

                if executed_trade:
                    executed_trades.append(executed_trade)
                    self.logger.info(f"Executed trade: {executed_trade}")

            except Exception as e:
                self.logger.error(f"Failed to execute trade for {opportunity['symbol']}: {e}")

        return executed_trades

    def _simulate_trade(self, opportunity: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate trade execution for offline testing"""
        return {
            'symbol': opportunity['symbol'],
            'direction': opportunity['direction'],
            'quantity': opportunity['position_size'],
            'entry_price': opportunity['entry_price'],
            'timestamp': datetime.now(),
            'status': 'simulated',
            'order_id': f"sim_{int(time.time())}"
        }

    def _execute_live_trade(self, opportunity: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Execute live trade"""
        # Use your existing bracket order logic
        from scripts.hybrid_crypto_trader import place_bracket

        try:
            # Calculate stop loss and take profit
            entry_price = opportunity['entry_price']
            stop_loss = self.risk_manager.get_dynamic_stop_loss(
                entry_price,
                0.02,  # Placeholder volatility
                opportunity['regime']
            )
            take_profit = self.risk_manager.get_dynamic_take_profit(
                entry_price,
                0.02,  # Placeholder volatility
                opportunity['regime']
            )

            success, order_id, error = place_bracket(
                self.api,
                opportunity['symbol'],
                opportunity['position_size'],
                entry_price,
                entry_price + take_profit,
                entry_price - stop_loss
            )

            if success:
                return {
                    'symbol': opportunity['symbol'],
                    'direction': opportunity['direction'],
                    'quantity': opportunity['position_size'],
                    'entry_price': entry_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'timestamp': datetime.now(),
                    'status': 'executed',
                    'order_id': order_id
                }
            else:
                self.logger.warning(f"Trade execution failed: {error}")
                return None

        except Exception as e:
            self.logger.error(f"Live trade execution failed: {e}")
            return None

    def _check_exit_signals(self,
                          market_data: Dict[str, pd.DataFrame],
                          regimes: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check for exit signals on existing positions"""
        exit_signals = []

        for symbol, position in self.positions.items():
            try:
                if symbol not in market_data:
                    continue

                data = market_data[symbol]
                current_price = data['close'].iloc[-1]
                entry_price = position['entry_price']
                stop_loss = position.get('stop_loss', entry_price * 0.98)
                take_profit = position.get('take_profit', entry_price * 1.05)

                holding_time = (datetime.now() - position['timestamp']).total_seconds()

                # Check exit conditions
                exit_signal = self.entry_logic.should_exit(
                    current_price,
                    entry_price,
                    stop_loss,
                    take_profit,
                    holding_time,
                    regimes.get(symbol, 'trending')
                )

                if exit_signal:
                    exit_signals.append({
                        'symbol': symbol,
                        'reason': exit_signal.reason,
                        'exit_price': exit_signal.target_price,
                        'position': position
                    })

            except Exception as e:
                self.logger.warning(f"Exit check failed for {symbol}: {e}")

        return exit_signals

    def _execute_exits(self, exit_signals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Execute exit orders"""
        exit_trades = []

        for signal in exit_signals:
            try:
                if self.api is None:
                    # Simulate exit
                    exit_trade = self._simulate_exit(signal)
                else:
                    # Live exit
                    exit_trade = self._execute_live_exit(signal)

                if exit_trade:
                    exit_trades.append(exit_trade)
                    # Remove from positions
                    del self.positions[signal['symbol']]

            except Exception as e:
                self.logger.error(f"Exit execution failed for {signal['symbol']}: {e}")

        return exit_trades

    def _simulate_exit(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate position exit"""
        position = signal['position']
        exit_price = signal['exit_price']

        pnl = (exit_price - position['entry_price']) * position['quantity']

        return {
            'symbol': signal['symbol'],
            'direction': 'exit',
            'quantity': position['quantity'],
            'exit_price': exit_price,
            'entry_price': position['entry_price'],
            'pnl': pnl,
            'timestamp': datetime.now(),
            'status': 'simulated_exit',
            'reason': signal['reason']
        }

    def _execute_live_exit(self, signal: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Execute live position exit"""
        from scripts.hybrid_crypto_trader import close_position_if_any

        try:
            order_id = close_position_if_any(self.api, signal['symbol'])

            if order_id:
                position = signal['position']
                exit_price = signal['exit_price']
                pnl = (exit_price - position['entry_price']) * position['quantity']

                return {
                    'symbol': signal['symbol'],
                    'direction': 'exit',
                    'quantity': position['quantity'],
                    'exit_price': exit_price,
                    'entry_price': position['entry_price'],
                    'pnl': pnl,
                    'timestamp': datetime.now(),
                    'status': 'executed_exit',
                    'order_id': order_id,
                    'reason': signal['reason']
                }
            else:
                return None

        except Exception as e:
            self.logger.error(f"Live exit failed: {e}")
            return None

    def _update_portfolio(self, trades: List[Dict[str, Any]]):
        """Update portfolio state after trades"""
        for trade in trades:
            symbol = trade['symbol']

            if trade['direction'] in ['buy', 'long']:
                # Add position
                self.positions[symbol] = {
                    'quantity': trade['quantity'],
                    'entry_price': trade['entry_price'],
                    'timestamp': trade['timestamp'],
                    'stop_loss': trade.get('stop_loss'),
                    'take_profit': trade.get('take_profit')
                }
                # Update cash
                self.cash -= trade['quantity'] * trade['entry_price']

            elif trade['direction'] == 'exit':
                # Remove position and update cash
                if symbol in self.positions:
                    position = self.positions[symbol]
                    exit_value = trade['quantity'] * trade['exit_price']
                    self.cash += exit_value

        # Update portfolio value
        self.portfolio_value = self.cash
        for symbol, position in self.positions.items():
            # This is simplified - in production get current market prices
            self.portfolio_value += position['quantity'] * position['entry_price']

    def _update_performance(self, trades: List[Dict[str, Any]]):
        """Update performance tracking"""
        for trade in trades:
            if 'pnl' in trade:
                self.trade_history.append(trade)

                # Update adaptive strategy
                self.adaptive_strategy.update_performance({
                    'pnl': trade['pnl'],
                    'symbol': trade['symbol'],
                    'direction': trade['direction'],
                    'quantity': trade['quantity'],
                    'entry_price': trade.get('entry_price'),
                    'exit_price': trade.get('exit_price'),
                    'timestamp': trade['timestamp']
                })

    def _adapt_strategy(self):
        """Adapt strategy parameters based on performance"""
        try:
            # Get current performance analysis
            performance_data = self.adaptive_strategy.performance_tracker.analyze_recent_trades()

            # Get current regime (simplified)
            current_regime = 'trending'  # In production, detect from market data

            # Adapt parameters
            adaptation_result = self.adaptive_strategy.adapt_parameters(
                performance_data,
                current_regime
            )

            if adaptation_result['status'] == 'adapted':
                self.logger.info(f"Strategy adapted: {adaptation_result}")
                # Update risk manager and other components with new parameters
                self._apply_adapted_parameters(adaptation_result['params'])

        except Exception as e:
            self.logger.error(f"Strategy adaptation failed: {e}")

    def _apply_adapted_parameters(self, params: Dict[str, Any]):
        """Apply adapted parameters to trading components"""
        try:
            # Update risk manager
            if 'risk_per_trade' in params:
                self.risk_manager.risk_limits.portfolio_var_limit = params['risk_per_trade'] * 2

            # Update entry logic thresholds
            if 'sentiment_threshold' in params:
                # This would require updating the entry logic class
                pass

            if 'ml_confidence_threshold' in params:
                # This would require updating the entry logic class
                pass

        except Exception as e:
            self.logger.warning(f"Failed to apply adapted parameters: {e}")

    def _log_cycle_results(self,
                          executed_trades: List[Dict[str, Any]],
                          exit_trades: List[Dict[str, Any]],
                          regimes: Dict[str, Any]):
        """Log cycle results and send notifications"""
        try:
            # Log summary
            self.logger.info(f"Cycle completed: {len(executed_trades)} entries, {len(exit_trades)} exits")

            # Calculate daily P&L
            daily_pnl = sum(trade.get('pnl', 0) for trade in exit_trades)
            self.daily_pnl.append({
                'date': datetime.now().date(),
                'pnl': daily_pnl,
                'trades': len(exit_trades)
            })

            # Send notifications if configured
            if executed_trades or exit_trades:
                self._send_notifications(executed_trades, exit_trades, regimes)

        except Exception as e:
            self.logger.error(f"Failed to log cycle results: {e}")

    def _send_notifications(self,
                          executed_trades: List[Dict[str, Any]],
                          exit_trades: List[Dict[str, Any]],
                          regimes: Dict[str, Any]):
        """Send trading notifications"""
        try:
            message = f"Enhanced Trading Agent Update:\n"
            message += f"Portfolio Value: ${self.portfolio_value:,.2f}\n"
            message += f"Open Positions: {len(self.positions)}\n"
            message += f"New Trades: {len(executed_trades)}\n"
            message += f"Closed Trades: {len(exit_trades)}\n"

            if executed_trades:
                message += "\nNew Positions:\n"
                for trade in executed_trades[:3]:  # Show first 3
                    message += f"• {trade['symbol']} {trade['direction']} {trade['quantity']:.4f} @ ${trade['entry_price']:.2f}\n"

            if exit_trades:
                message += "\nClosed Positions:\n"
                for trade in exit_trades[:3]:  # Show first 3
                    pnl_str = f"${trade['pnl']:.2f}" if 'pnl' in trade else "N/A"
                    message += f"• {trade['symbol']} exit @ ${trade['exit_price']:.2f} (PnL: {pnl_str})\n"

            # Send via Telegram if configured
            if not os.getenv("TB_NO_TELEGRAM", "1") == "1":
                send_telegram(message)

            # Send via Discord if configured
            if os.getenv("TB_ENABLE_DISCORD", "0") == "1":
                # Use your existing Discord sender
                pass

        except Exception as e:
            self.logger.error(f"Notification failed: {e}")

# ==========================
# Main execution
# ==========================

def main():
    """Main function to run the enhanced trading agent"""
    agent = EnhancedTradingAgent()

    # Run initial setup
    agent.logger.info("Enhanced Trading Agent started")
    agent.logger.info(f"Enabled assets: {[s for s, c in agent.assets.items() if c.enabled]}")

    # Main trading loop
    while True:
        try:
            agent.run_trading_cycle()

            # Wait before next cycle (e.g., every 15 minutes)
            time.sleep(15 * 60)

        except KeyboardInterrupt:
            agent.logger.info("Trading agent stopped by user")
            break
        except Exception as e:
            agent.logger.error(f"Main loop error: {e}")
            time.sleep(60)  # Brief pause before retry

if __name__ == "__main__":
    main()
