#!/usr/bin/env python3
"""
High-Risk Futures Agent
Separate high-risk futures trading agent with enhanced features
"""

import os
import sys
import time
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional
import logging
import asyncio
import traceback

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Import futures platform functions
from futures_integration import (
    enhanced_futures_bars,
    calculate_futures_position,
    execute_futures_trade,
    get_futures_status,
    get_account_balance,
    is_futures_available,
    switch_platform,
    get_platform_config,
    calculate_smart_leverage
)

# Import notification modules
try:
    from scripts.discord_sender import send_discord_digest_to
    from telegram_bot import send_message as send_telegram
    NOTIFICATIONS_AVAILABLE = True
except ImportError:
    NOTIFICATIONS_AVAILABLE = False
    send_discord_digest_to = None
    send_telegram = None

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
        self.symbols = os.getenv("FUTURES_SYMBOLS", "BTCUSDT,ETHUSDT,SOLUSDT,ADAUSDT,DOTUSDT,LINKUSDT,AVAXUSDT,MATICUSDT,UNIUSDT,AAVEUSDT").split(",")
        self.max_positions = int(os.getenv("FUTURES_MAX_POSITIONS", "5"))  # Allow up to 5 concurrent positions
        self.max_trades_per_cycle = int(os.getenv("FUTURES_MAX_TRADES_PER_CYCLE", "3"))  # Allow up to 3 trades per cycle

        # Multi-platform support
        self.available_platforms = ['binance', 'bybit']
        self.current_platform = os.getenv("TB_FUTURES_PLATFORM", "binance")
        self.platform_switch_cooldown = 300  # 5 minutes between platform switches
        self.last_platform_switch = 0

        # Per-platform capital tracking
        self.platform_capital = {
            'binance': float(os.getenv("BINANCE_PAPER_CAPITAL", "15000")),
            'bybit': float(os.getenv("BYBIT_PAPER_CAPITAL", "100000"))
        }

        # State tracking
        self.daily_pnl = 0.0
        self.trades_today = 0
        self.positions = {}
        self.trade_log = []

        # Strategy parameters - More aggressive for frequent trading
        self.momentum_window = 6  # hours (reduced for faster signals)
        self.volatility_window = 12  # hours (reduced for more responsive)
        self.min_momentum_threshold = 0.008  # 0.8% momentum (reduced for more signals)
        self.max_volatility_threshold = 0.12  # 12% max volatility (increased for more opportunities)

        # New: Market regime and correlation tracking
        self.market_regime = 'unknown'
        self.correlation_matrix = {}  # Initialize as empty dict
        self.trailing_stops = {}  # Track trailing stop levels

        # Heartbeat and notification tracking
        self.run_count = 0
        self.last_heartbeat = time.time()
        self.heartbeat_every_n = int(os.getenv("TB_HEARTBEAT_EVERY_N", "12"))
        self.enable_heartbeat = os.getenv("TB_TRADER_NOTIFY_HEARTBEAT", "0") == "1"
        self.enable_notifications = os.getenv("TB_TRADER_NOTIFY", "0") == "1"
        self.enable_discord = os.getenv("TB_ENABLE_DISCORD", "0") == "1"
        self.no_telegram = os.getenv("TB_NO_TELEGRAM", "1") == "1"
        self.discord_webhook = os.getenv("DISCORD_TRADER_WEBHOOK_URL", "") or os.getenv("DISCORD_WEBHOOK_URL", "")

        logger.info(f"🚀 {self.name} initialized")
        logger.info(f"💰 Capital: Binance=${self.platform_capital['binance']}, Bybit=${self.platform_capital['bybit']}")
        logger.info(f"⚡ Max Leverage: {self.max_leverage}x")
        logger.info(f"🎯 Risk per Trade: {self.risk_per_trade*100}%")
        logger.info(f"📊 Symbols: {', '.join(self.symbols)} ({len(self.symbols)} total)")
        logger.info(f"📈 Max Positions: {self.max_positions}")
        logger.info(f"🔄 Max Trades per Cycle: {self.max_trades_per_cycle}")
        logger.info(f"🏛️ Platforms: {', '.join(self.available_platforms)}")
        logger.info(f"🎯 Current Platform: {self.current_platform}")
        logger.info(f"📊 Momentum Window: {self.momentum_window}h")
        logger.info(f"📊 Min Momentum Threshold: {self.min_momentum_threshold*100:.1f}%")
        logger.info(f"📊 Max Volatility Threshold: {self.max_volatility_threshold*100:.1f}%")
        logger.info(f"📢 Notifications: {'Enabled' if self.enable_notifications else 'Disabled'}")
        logger.info(f"💓 Heartbeat: {'Enabled' if self.enable_heartbeat else 'Disabled'} (every {self.heartbeat_every_n} runs)")

    def check_internet_connectivity(self) -> bool:
        """Check if internet connectivity is available"""
        try:
            import requests
            # Try to reach a reliable endpoint
            response = requests.get('https://httpbin.org/status/200', timeout=10)
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"🌐 Internet connectivity check failed: {e}")
            return False

    def switch_platform(self, platform_name: str) -> bool:
        """Switch to a different trading platform"""
        if platform_name not in self.available_platforms:
            logger.warning(f"⚠️ Platform {platform_name} not available")
            return False

        # Check cooldown
        current_time = time.time()
        if current_time - self.last_platform_switch < self.platform_switch_cooldown:
            logger.info(f"⏳ Platform switch cooldown active, waiting...")
            return False

        # Switch platform
        if switch_platform(platform_name):
            self.current_platform = platform_name
            self.last_platform_switch = current_time
            logger.info(f"🔄 Switched to {platform_name} platform")
            return True

        return False

    def get_current_platform_capital(self) -> float:
        """Get REAL capital for current platform from API"""
        try:
            # Try to get real balance from platform
            balance_info = get_account_balance()
            if balance_info and 'available_balance' in balance_info:
                real_balance = balance_info['available_balance']
                logger.info(f"💰 Real available balance: ${real_balance:.2f}")
                return real_balance
            else:
                # Fallback to configured capital
                fallback_capital = self.platform_capital.get(self.current_platform, self.capital)
                logger.warning(f"⚠️ Could not fetch real balance, using fallback: ${fallback_capital:.2f}")
                return fallback_capital
        except Exception as e:
            # Fallback to configured capital
            fallback_capital = self.platform_capital.get(self.current_platform, self.capital)
            logger.warning(f"⚠️ Error fetching real balance: {e}, using fallback: ${fallback_capital:.2f}")
            return fallback_capital

    def should_switch_platform(self) -> Optional[str]:
        """Determine if we should switch platforms based on availability and limits"""
        try:
            # Check if current platform has issues
            status = get_futures_status()
            if 'error' in status:
                logger.warning(f"⚠️ Current platform {self.current_platform} has issues, checking alternatives...")

                # Try alternative platforms
                for platform in self.available_platforms:
                    if platform != self.current_platform:
                        # Check if alternative platform is available
                        if self.switch_platform(platform):
                            return platform

            # Check if we need to switch based on trade size limits
            platform_config = get_platform_config(self.current_platform)
            max_trade_size = platform_config.get('max_trade_size', float('inf'))

            # If current capital exceeds platform limits, switch to platform with higher limits
            current_capital = self.get_current_platform_capital()
            if current_capital > max_trade_size * 10:  # If capital is much larger than trade size
                for platform in self.available_platforms:
                    if platform != self.current_platform:
                        alt_config = get_platform_config(platform)
                        alt_max_size = alt_config.get('max_trade_size', 0)
                        alt_capital = self.platform_capital.get(platform, 0)

                        if alt_max_size > max_trade_size and alt_capital >= current_capital:
                            logger.info(f"💰 Switching to {platform} for larger trade sizes")
                            if self.switch_platform(platform):
                                return platform

            return None

        except Exception as e:
            logger.warning(f"Error checking platform switch: {e}")
            return None

    def is_market_open(self) -> bool:
        """Check if futures markets are open (crypto markets are 24/7)"""
        return True  # Crypto futures are always open

    def notify(self, event: str, payload: Dict) -> None:
        """Send notifications for trades and heartbeat events"""
        logger.info(f"📢 NOTIFY called for event: {event}")
        logger.info(f"📢 enable_notifications: {self.enable_notifications}, NOTIFICATIONS_AVAILABLE: {NOTIFICATIONS_AVAILABLE}")
        
        if not self.enable_notifications or not NOTIFICATIONS_AVAILABLE:
            logger.warning(f"📢 NOTIFICATION BLOCKED - enable_notifications: {self.enable_notifications}, NOTIFICATIONS_AVAILABLE: {NOTIFICATIONS_AVAILABLE}")
            return

        logger.info(f"📢 enable_discord: {self.enable_discord}, discord_webhook length: {len(self.discord_webhook) if self.discord_webhook else 0}")
        logger.info(f"📢 no_telegram: {self.no_telegram}")

        symbol = payload.get("symbol", "")
        status = payload.get("status", "")
        price = payload.get("price", 0)
        qty = payload.get("qty", 0)
        leverage = payload.get("leverage", 1)

        # Build message based on event type
        if event.lower() in ("trade", "buy", "sell"):
            action = payload.get("action", event.upper())
            pnl = payload.get("pnl", 0)
            reason = payload.get("reason", "")

            desc_lines = [
                f"Symbol: {symbol}",
                f"Action: {action}",
                f"Quantity: {qty}",
                f"Price: ${price:.2f}",
                f"Leverage: {leverage}x",
                f"Reason: {reason}",
                f"Status: {status}"
            ]

            if pnl != 0:
                desc_lines.append(f"P&L: ${pnl:.2f}")

            tg_msg = (
                f"High-Risk Futures Agent • {symbol}. {action}. "
                f"qty={qty} price=${price:.2f} x{leverage} {reason}"
            )
            color = 0x2ecc71 if action == "BUY" else 0xe74c3c

        elif event.lower() == "heartbeat":
            desc_lines = [
                f"Agent: {self.name}",
                f"Status: {status}",
                f"Run Count: {self.run_count}",
                f"Active Positions: {len(self.positions)}",
                f"Daily P&L: ${self.daily_pnl:.2f}",
                f"Market Regime: {self.market_regime}"
            ]

            tg_msg = (
                f"High-Risk Futures Agent • Heartbeat. "
                f"Positions: {len(self.positions)} P&L: ${self.daily_pnl:.2f} Regime: {self.market_regime}"
            )
            color = 0x95a5a6

        else:  # Generic event
            desc_lines = [
                f"Event: {event}",
                f"Symbol: {symbol}",
                f"Status: {status}"
            ]
            tg_msg = f"High-Risk Futures Agent • {event}. {symbol} {status}"
            color = 0x95a5a6

        embed = {
            "title": f"Futures Agent: {event} {symbol}",
            "description": "\n".join(desc_lines),
            "color": color,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

        # Send Discord notification
        if self.enable_discord and self.discord_webhook:
            logger.info(f"📢 Sending Discord notification for {event}")
            try:
                send_discord_digest_to(self.discord_webhook, [embed])
                logger.info(f"📢 Discord notification sent for {event}")
            except Exception as e:
                logger.warning(f"Failed to send Discord notification: {e}")

        # Send Telegram notification
        if not self.no_telegram and send_telegram:
            logger.info(f"📢 Sending Telegram notification for {event}")
            try:
                send_telegram(tg_msg)
                logger.info(f"📱 Telegram notification sent for {event}")
            except Exception as e:
                logger.warning(f"Failed to send Telegram notification: {e}")

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
                if correlation > 0.85:  # Higher correlation threshold (reduced from 0.7)
                    logger.info(f"⚠️ Skipping {symbol} due to high correlation ({correlation:.2f}) with {existing_symbol}")
                    return False

        return True

    def calculate_dynamic_leverage(self, symbol: str, volatility: float) -> int:
        """Calculate dynamic leverage based on volatility and market regime"""
        try:
            # Get platform-specific max leverage
            platform_config = get_platform_config(self.current_platform)
            max_platform_leverage = platform_config.get('max_leverage', self.max_leverage)

            # Use smart leverage calculation from futures integration
            market_regime = self.detect_market_regime(symbol)
            smart_leverage = calculate_smart_leverage(
                symbol, max_platform_leverage, volatility, market_regime
            )

            logger.info(f"🎯 Smart leverage for {symbol}: {smart_leverage}x (vol: {volatility:.4f}, regime: {market_regime})")
            return smart_leverage

        except Exception as e:
            logger.warning(f"Error calculating dynamic leverage for {symbol}: {e}")
            return min(self.max_leverage, max_platform_leverage)

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

        # Check maximum positions limit
        if len(self.positions) >= self.max_positions:
            logger.info(f"📊 Maximum positions reached ({self.max_positions}), skipping {symbol}")
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
        """Execute a futures trade with platform switching support"""
        try:
            # Check if we should switch platforms
            switch_platform = self.should_switch_platform()
            if switch_platform:
                logger.info(f"🔄 Switching to {switch_platform} for better trading conditions")

            side = signal['signal']

            # Use platform-specific capital for position sizing
            platform_capital = self.get_current_platform_capital()

            # Calculate position size with platform-specific limits
            pos_info = calculate_futures_position(
                symbol,
                platform_capital,
                self.risk_per_trade
            )

            if 'error' in pos_info:
                logger.warning(f"❌ Position calculation failed: {pos_info['error']}")
                return False

            # Apply smart leverage calculation
            volatility = signal.get('volatility', 0.05)
            market_regime = self.detect_market_regime(symbol)
            smart_leverage = calculate_smart_leverage(
                symbol, pos_info['leverage_used'], volatility, market_regime
            )
            pos_info['leverage_used'] = smart_leverage

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
                    'leverage': smart_leverage,
                    'platform': self.current_platform,
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'signal': signal,
                    'highest_price': float(entry_price),  # For trailing stops
                    'lowest_price': float(entry_price),   # For trailing stops
                    'trailing_stop_pct': 0.05,  # 5% trailing stop
                    'profit_target_pct': 0.08   # 8% profit target
                }

                # Log trade
                trade_record = {
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'symbol': symbol,
                    'side': side,
                    'quantity': float(quantity),
                    'price': float(entry_price),
                    'leverage': smart_leverage,
                    'platform': self.current_platform,
                    'order_id': trade_result['order_id'],
                    'signal_strength': signal['strength'],
                    'reason': signal['reason']
                }
                self.trade_log.append(trade_record)

                self.trades_today += 1

                logger.info(f"✅ {side.upper()} {symbol} x{smart_leverage} @ ${entry_price:.2f} on {self.current_platform}")

                # Send trade notification
                self.notify("trade", {
                    "symbol": symbol,
                    "action": side.upper(),
                    "price": entry_price,
                    "qty": quantity,
                    "leverage": smart_leverage,
                    "platform": self.current_platform,
                    "status": "executed",
                    "reason": signal['reason']
                })

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
            position_timestamp = datetime.fromisoformat(position['timestamp'])
            if position_timestamp.tzinfo is not None:
                # Convert to naive datetime for comparison
                position_timestamp = position_timestamp.replace(tzinfo=None)
            current_time = datetime.now().replace(tzinfo=None)
            position_age_hours = (current_time - position_timestamp).total_seconds() / 3600
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

            # Send close notification
            self.notify("close", {
                "symbol": symbol,
                "action": "CLOSE",
                "price": exit_price,
                "qty": quantity,
                "leverage": leverage,
                "status": reason,
                "pnl": pnl_amount,
                "reason": reason
            })

            # Record exit
            exit_record = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
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
            'platform_capital': self.platform_capital,
            'current_platform': self.current_platform,
            'available_platforms': self.available_platforms,
            'daily_pnl': self.daily_pnl,
            'trades_today': self.trades_today,
            'open_positions': len(self.positions),
            'positions': list(self.positions.keys()),
            'total_trades': len(self.trade_log),
            'win_rate': self.calculate_win_rate(),
            'market_regime': self.market_regime,
            'correlation_pairs': len(self.correlation_matrix) if self.correlation_matrix else 0,
            'timestamp': datetime.now(timezone.utc).isoformat()
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

        # Increment run count and check heartbeat
        self.run_count += 1
        logger.info(f"📊 Run count: {self.run_count}, Heartbeat every: {self.heartbeat_every_n}")
        logger.info(f"💓 Heartbeat enabled: {self.enable_heartbeat}, Notifications enabled: {self.enable_notifications}")
        
        if self.enable_heartbeat and self.run_count % self.heartbeat_every_n == 0:
            logger.info(f"💓 HEARTBEAT CONDITION MET - Sending heartbeat for run {self.run_count}")
            self.notify("heartbeat", {"status": "active"})
        else:
            logger.info(f"💓 Heartbeat condition not met: {self.enable_heartbeat} and {self.run_count % self.heartbeat_every_n == 0}")

        # Sync existing positions on first run
        if self.run_count == 1:
            logger.info("🔄 Syncing existing positions from platform...")
            self.sync_existing_positions()

        # Update market regime and correlations
        self.update_market_context()

        # Check existing positions
        self.check_positions()

        # Look for new trades
        trades_this_cycle = 0
        for symbol in self.symbols:
            if symbol in self.positions:
                continue  # Skip if we already have position

            if trades_this_cycle >= self.max_trades_per_cycle:
                logger.info(f"📊 Maximum trades per cycle reached ({self.max_trades_per_cycle}), stopping for this cycle")
                break

            signal = self.calculate_momentum_signal(symbol)

            if signal['signal'] != 'neutral' and self.should_trade(symbol, signal):
                # Log additional context
                regime = self.detect_market_regime(symbol)
                logger.info(f"🎯 Signal detected for {symbol}: {signal['signal']} ({signal['strength']:.4f})")
                logger.info(f"📊 Market regime: {regime} | Volatility: {signal.get('volatility', 0):.4f}")
                if self.execute_trade(symbol, signal):
                    trades_this_cycle += 1
                    logger.info(f"✅ Trade {trades_this_cycle}/{self.max_trades_per_cycle} executed this cycle")

        # Log status
        status = self.get_status()
        logger.info(f"📊 Status: ${status['daily_pnl']:.2f} P&L | {status['open_positions']} positions | {status['trades_today']} trades today")
        logger.info(f"🏛️ Platform: {status['current_platform']} | Capital: ${self.get_current_platform_capital():.0f}")
        logger.info(f"🌍 Market Regime: {status['market_regime']} | Win Rate: {status['win_rate']:.1%}")

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

    def run_continuous_sync(self, interval_seconds: int = 120):
        """Run continuous trading loop synchronously (for nohup compatibility)"""
        logger.info(f"🚀 Starting continuous futures trading (sync mode, interval: {interval_seconds}s)")

        cycle_count = 0
        while True:
            try:
                cycle_count += 1
                
                # Check internet connectivity before starting cycle
                if not self.check_internet_connectivity():
                    logger.warning("🌐 No internet connectivity, waiting 60s before retry...")
                    time.sleep(60)
                    continue
                
                logger.info(f"🔄 Starting cycle {cycle_count} of continuous loop")
                self.run_trading_cycle()
                
                # Auto-commit database changes after each cycle
                try:
                    import os
                    if os.getenv("TB_AUTOCOMMIT_ARTIFACTS", "1") == "1":
                        push_enabled = os.getenv("TB_AUTOCOMMIT_PUSH", "1") == "1"
                        import subprocess
                        code = subprocess.call([
                            "python3", "-c",
                            (
                                "import autocommit as ac; "
                                "print(ac.auto_commit_and_push(['enhanced_trading.db','futures_agent.log','high_risk_futures_loop.log'], "
                                "extra_message='futures trading database and logs', push_enabled="
                                + ("True" if push_enabled else "False") +
                                "))"
                            )
                        ])
                        logger.info(f"[autocommit] futures database committed with status: {code}")
                except Exception as e:
                    logger.warning(f"[autocommit] failed: {e}")
                
                logger.info(f"✅ Completed cycle {cycle_count}, sleeping for {interval_seconds}s")
                time.sleep(interval_seconds)

            except KeyboardInterrupt:
                logger.info("🛑 Stopping continuous trading...")
                break
            except Exception as e:
                logger.error(f"Error in trading cycle {cycle_count}: {e}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                time.sleep(60)  # Wait before retry

    def sync_existing_positions(self):
        """Sync existing positions from platform into agent management"""
        try:
            status = get_futures_status()
            
            if 'error' in status:
                logger.warning(f"❌ Could not sync positions: {status['error']}")
                return False
            
            platform_positions = status.get('positions', [])
            synced_count = 0
            
            for pos in platform_positions:
                symbol = pos.get('symbol', '').replace('USDT', '') + 'USDT'  # Normalize symbol
                if symbol in self.symbols and symbol not in self.positions:
                    # Import position into agent management
                    self.positions[symbol] = {
                        'side': pos.get('side', 'buy'),
                        'entry_price': float(pos.get('entry_price', 0)),
                        'quantity': float(pos.get('quantity', 0)),
                        'leverage': int(pos.get('leverage', 1)),
                        'platform': pos.get('platform', self.current_platform),
                        'timestamp': datetime.now(timezone.utc).isoformat(),  # Use current time as sync time
                        'signal': {'signal': 'synced', 'strength': 0, 'reason': 'existing_position_sync'},
                        'highest_price': float(pos.get('entry_price', 0)),  # For trailing stops
                        'lowest_price': float(pos.get('entry_price', 0)),   # For trailing stops
                        'trailing_stop_pct': 0.05,  # 5% trailing stop
                        'profit_target_pct': 0.08   # 8% profit target
                    }
                    synced_count += 1
                    logger.info(f"✅ Synced existing position: {symbol} {pos.get('side')} x{pos.get('quantity')}")
            
            if synced_count > 0:
                logger.info(f"🔄 Successfully synced {synced_count} existing positions into agent management")
                return True
            else:
                logger.info("ℹ️  No new positions to sync")
                return True
                
        except Exception as e:
            logger.error(f"Error syncing existing positions: {e}")
            return False

def main():
    """Main function"""
    print("🚀 High-Risk Futures Agent")
    print("=" * 50)

    if not is_futures_available():
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
    print(f"💰 Total Capital: ${status['capital']:.0f}")
    print(f"🏛️ Current Platform: {status['current_platform']}")
    print(f"💰 Platform Capital: Binance=${agent.platform_capital['binance']:.0f}, Bybit=${agent.platform_capital['bybit']:.0f}")
    print(f"📊 Daily P&L: ${status['daily_pnl']:.2f}")
    print(f"📊 Trades Today: {status['trades_today']}")
    print(f"📈 Open Positions: {status['open_positions']}")
    print(f"🎯 Win Rate: {status['win_rate']:.1%}")
    print(f"📝 Total Trades: {status['total_trades']}")
    print(f"🌍 Market Regime: {status['market_regime']}")
    print(f"📊 Correlation Pairs: {status['correlation_pairs']}")

    print("\n✅ Enhanced Multi-Platform High-Risk Futures Agent Ready!")
    print("🚀 New Features:")
    print("  • Multi-platform support (Binance & Bybit)")
    print("  • Per-platform trading limits ($100 Binance, $500 Bybit)")
    print("  • Smart leverage calculation based on risk/reward")
    print("  • Platform switching for optimal trading conditions")
    print("  • Market regime detection (trending/ranging)")
    print("  • Correlation filtering to avoid correlated positions")
    print("  • Advanced exit timing with trailing stops")
    print("  • Profit targets and time-based exits")
    print("\n💡 Run with: python3 high_risk_futures_agent.py --continuous")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='High-Risk Futures Agent')
    parser.add_argument('--continuous', action='store_true', help='Run continuous trading loop')
    parser.add_argument('--interval', type=int, default=120, help='Trading cycle interval in seconds (default: 120)')

    args = parser.parse_args()

    if args.continuous:
        logger.info("Using synchronous continuous loop for better nohup compatibility")
        HighRiskFuturesAgent().run_continuous_sync(args.interval)
    else:
        main()
