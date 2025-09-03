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

# Import world-class technical analysis engine
try:
    from world_class_technical_analysis import (
        TechnicalAnalysisEngine,
        RiskTargets, 
        MarketRegime,
        calculate_world_class_crypto_targets
    )
    WORLD_CLASS_TA_AVAILABLE = True
    print("✅ World-class technical analysis engine loaded for futures agent")
except ImportError as e:
    print(f"❌ Failed to import world-class TA engine for futures: {e}")
    WORLD_CLASS_TA_AVAILABLE = False

# Initialize the technical analysis engine
if WORLD_CLASS_TA_AVAILABLE:
    TA_ENGINE = TechnicalAnalysisEngine(
        atr_period=14,
        rsi_period=14,
        bb_period=20,
        bb_std_dev=2.0,
        min_bars_required=30  # Reduced for futures (faster timeframes)
    )
else:
    TA_ENGINE = None

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

# =============================================================================
# INTELLIGENT TP/SL CONFIGURATION FOR FUTURES
# =============================================================================

# Trade Quality Levels for Futures (more aggressive due to leverage)
FUTURES_TRADE_QUALITY_LEVELS = {
    'excellent': {
        'tp_range': (0.02, 0.03),  # 2-3% TP for excellent signals (with 25x = 50-75% ROI)
        'sl_base': 0.02,           # 2% SL (with 25x = 50% ROI loss)
        'description': 'High conviction setup, strong confluence'
    },
    'good': {
        'tp_range': (0.015, 0.025),  # 1.5-2.5% TP for good signals (with 25x = 37-62% ROI)
        'sl_base': 0.02,             # 2% SL (with 25x = 50% ROI loss)
        'description': 'Good setup, decent confidence'
    },
    'fair': {
        'tp_range': (0.01, 0.02),  # 1-2% TP for fair signals (with 25x = 25-50% ROI)
        'sl_base': 0.015,          # 1.5% SL (with 25x = 37% ROI loss)
        'description': 'Marginal setup, lower confidence'
    }
}

# Asset Difficulty for Futures (adjusted for leverage and volatility)
FUTURES_ASSET_DIFFICULTY = {
    'BTC': 1.6,   # Hardest to move significantly
    'ETH': 1.4,   # Major crypto, decent liquidity
    'SOL': 1.2,   # Top 10, good liquidity
    'AVAX': 1.1,  # Top 20, moderate liquidity
    'LINK': 1.1,  # Established DeFi
    'UNI': 1.0,   # DEX token, baseline
    'AAVE': 0.9,  # DeFi, smaller
    'COMP': 0.8,  # Smaller DeFi
    'YFI': 0.7,   # Smallest, highest volatility
    'XTZ': 0.8,   # Alt L1
    'LTC': 1.0,   # Legacy crypto
    'BCH': 1.0,   # Bitcoin fork
}

# =============================================================================

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

    # =============================================================================
    # INTELLIGENT TP/SL METHODS FOR FUTURES
    # =============================================================================
    
    def get_asset_symbol_clean(self, symbol: str) -> str:
        """Clean symbol for asset lookup"""
        return symbol.replace('USDT', '').replace('/USD', '').replace('USD', '').upper()
    
    def analyze_futures_trade_quality(self, symbol: str, entry_signal_strength: float = 0.5, 
                                    volatility: float = 0.0, volume_profile: float = 1.0) -> str:
        """
        Analyze futures trade quality based on multiple factors
        More sophisticated than the crypto version
        """
        symbol_clean = self.get_asset_symbol_clean(symbol)
        
        # Base quality assessment from signal strength
        base_quality = 'fair'  # Default
        
        if entry_signal_strength > 0.8:
            base_quality = 'excellent'
        elif entry_signal_strength > 0.6:
            base_quality = 'good'
        else:
            base_quality = 'fair'
        
        # Adjust for volatility (moderate volatility is better for futures)
        if 0.02 <= volatility <= 0.08:  # Sweet spot for futures
            # Good volatility, keep quality
            pass
        elif volatility > 0.12:  # Too volatile
            if base_quality == 'excellent':
                base_quality = 'good'
            elif base_quality == 'good':
                base_quality = 'fair'
        elif volatility < 0.01:  # Too quiet
            if base_quality == 'excellent':
                base_quality = 'good'
        
        # Adjust for volume profile
        if volume_profile < 0.5:  # Low volume
            if base_quality == 'excellent':
                base_quality = 'good'
            elif base_quality == 'good':
                base_quality = 'fair'
        
        # Asset-specific adjustments
        if symbol_clean in ['BTC', 'ETH'] and base_quality == 'fair':
            base_quality = 'good'  # Blue chips get benefit of doubt
        
        return base_quality
    
    def calculate_world_class_futures_targets(self, symbol: str, entry_price: float, side: str) -> Dict[str, float]:
        """
        Calculate world-class TP/SL targets using comprehensive technical analysis
        Replaces all hardcoded percentage-based calculations
        """
        try:
            if not WORLD_CLASS_TA_AVAILABLE or TA_ENGINE is None:
                logger.warning(f"⚠️  World-class TA engine not available for {symbol}, using fallback")
                return self.calculate_legacy_technical_targets(symbol, entry_price, side)
            
            # Get bars data for technical analysis
            bars = self.get_comprehensive_bars_data(symbol)
            if bars is None or len(bars) < 30:
                logger.warning(f"Insufficient bars data for {symbol}, using legacy fallback")
                return self.calculate_legacy_technical_targets(symbol, entry_price, side)
            
            # Calculate trade confidence for futures (typically higher risk tolerance)
            confidence = self.calculate_futures_trade_confidence(symbol, bars, side)
            
            # Use world-class technical analysis engine
            targets = TA_ENGINE.calculate_world_class_targets(
                df=bars,
                side=side,
                confidence=confidence,
                symbol=symbol
            )
            
            logger.info(f"🎯 World-class futures targets for {symbol}:")
            logger.info(f"   📊 Entry: ${targets.entry_price:.4f}")
            logger.info(f"   🛑 Stop Loss: ${targets.stop_loss:.4f} ({targets.sl_method})")
            logger.info(f"   🎯 Take Profit: ${targets.take_profit_1:.4f} ({targets.tp_method})")
            logger.info(f"   📈 R/R Ratio: {targets.risk_reward_ratio:.2f}")
            logger.info(f"   📊 Size Multiplier: {targets.position_size_multiplier:.2f}")
            logger.info(f"   🎲 Confidence: {confidence:.2f}")
            
            # Return in legacy format for compatibility
            return {
                'tp_pct': abs(targets.take_profit_1 - entry_price) / entry_price,
                'sl_pct': abs(targets.stop_loss - entry_price) / entry_price,
                'tp_price': targets.take_profit_1,
                'sl_price': targets.stop_loss,
                'tp_price_2': targets.take_profit_2,
                'signal_strength': confidence,
                'volatility': self.estimate_volatility_from_bars(bars),
                'position_size_multiplier': targets.position_size_multiplier,
                'risk_reward_ratio': targets.risk_reward_ratio,
                'method': 'world_class_technical_analysis',
                'sl_method': targets.sl_method,
                'tp_method': targets.tp_method,
                'confidence': confidence
            }
            
        except Exception as e:
            logger.error(f"❌ World-class TA failed for {symbol}: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return self.get_emergency_fallback_targets(side, symbol)
    
    def get_comprehensive_bars_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Get comprehensive bars data for technical analysis"""
        try:
            # Try enhanced futures bars first
            bars = enhanced_futures_bars(symbol, '15m', limit=100)
            if bars is not None and len(bars) >= 30:
                return bars
            
            # Fallback: direct Binance API
            logger.info(f"Trying direct Binance API for {symbol}...")
            bars = self.get_bars_from_binance(symbol, '15m', 100)
            if bars is not None and len(bars) >= 30:
                df = pd.DataFrame(bars)
                return df
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting bars data for {symbol}: {e}")
            return None
    
    def calculate_futures_trade_confidence(self, symbol: str, bars: pd.DataFrame, side: str) -> float:
        """
        Calculate trade confidence specific to futures trading
        Futures typically allow higher risk tolerance due to leverage
        """
        try:
            base_confidence = 0.7  # Higher base for futures
            
            # Volatility bonus (futures traders like volatility)
            vol_pct = bars['close'].pct_change().std() * (24 ** 0.5)  # Daily volatility
            if 0.03 <= vol_pct <= 0.08:  # Sweet spot for futures
                vol_bonus = 0.1
            elif vol_pct > 0.08:  # High volatility
                vol_bonus = 0.05  # Still bonus, but less
            else:
                vol_bonus = -0.05  # Too quiet
            
            # Trend alignment bonus
            ema_short = bars['close'].ewm(span=12).mean().iloc[-1]
            ema_long = bars['close'].ewm(span=26).mean().iloc[-1]
            current_price = bars['close'].iloc[-1]
            
            if side.lower() in ['long', 'buy']:
                trend_bonus = 0.1 if current_price > ema_short > ema_long else -0.1
            else:
                trend_bonus = 0.1 if current_price < ema_short < ema_long else -0.1
            
            # Volume confirmation
            recent_volume = bars['volume'].iloc[-5:].mean() if 'volume' in bars.columns else 0
            avg_volume = bars['volume'].mean() if 'volume' in bars.columns else 0
            
            if recent_volume > avg_volume * 1.2:
                volume_bonus = 0.05
            else:
                volume_bonus = -0.02
            
            final_confidence = base_confidence + vol_bonus + trend_bonus + volume_bonus
            return max(0.4, min(0.95, final_confidence))
            
        except Exception as e:
            logger.error(f"Error calculating futures confidence: {e}")
            return 0.6  # Default futures confidence
    
    def estimate_volatility_from_bars(self, bars: pd.DataFrame) -> float:
        """Estimate current volatility from bars data"""
        try:
            returns = bars['close'].pct_change().dropna()
            daily_vol = returns.std() * (24 ** 0.5)  # Annualized daily volatility
            return max(0.01, min(0.15, daily_vol))
        except:
            return 0.03  # Default volatility estimate
    
    def calculate_legacy_technical_targets(self, symbol: str, entry_price: float, side: str) -> Dict[str, float]:
        """
        Legacy technical analysis fallback when world-class engine is unavailable
        Still uses technical analysis, just less sophisticated
        """
        try:
            # Get bars data
            bars = enhanced_futures_bars(symbol, '15m', limit=50)
            if bars is None:
                bars = self.get_bars_from_binance(symbol, '15m', 50)
            
            if bars is None or len(bars) < 20:
                return self.get_emergency_fallback_targets(side, symbol)
            
            # Calculate basic technical indicators
            closes = bars['close'].values
            highs = bars['high'].values  
            lows = bars['low'].values
            
            # ATR-based targets
            atr = self.calculate_atr(highs, lows, closes, period=14)
            atr_multiplier_tp = 2.5  # Slightly more aggressive for futures
            atr_multiplier_sl = 1.2
            
            # RSI and Bollinger Bands
            rsi = self.calculate_rsi(closes, period=14)
            bb_upper, bb_lower, bb_middle = self.calculate_bollinger_bands(closes, period=20, std=2)
            
            current_price = closes[-1]
            
            if side.lower() in ['long', 'buy']:
                atr_tp = entry_price + (atr * atr_multiplier_tp)
                tp_price = min(atr_tp, bb_upper) if bb_upper > entry_price else atr_tp
                
                atr_sl = entry_price - (atr * atr_multiplier_sl)
                sl_price = max(atr_sl, bb_lower) if bb_lower < entry_price else atr_sl
            else:
                atr_tp = entry_price - (atr * atr_multiplier_tp)
                tp_price = max(atr_tp, bb_lower) if bb_lower < entry_price else atr_tp
                
                atr_sl = entry_price + (atr * atr_multiplier_sl)
                sl_price = min(atr_sl, bb_upper) if bb_upper > entry_price else atr_sl
            
            tp_pct = abs(tp_price - entry_price) / entry_price
            sl_pct = abs(sl_price - entry_price) / entry_price
            
            # Calculate signal strength
            if side.lower() in ['long', 'buy']:
                signal_strength = max(0.4, min(0.8, (50 - rsi) / 50 + 0.6))
            else:
                signal_strength = max(0.4, min(0.8, (rsi - 50) / 50 + 0.6))
            
            recent_volatility = bars['close'].pct_change().std() * (24 ** 0.5)
            
            logger.info(f"📊 Legacy TA for {symbol}: ATR=${atr:.4f}, RSI={rsi:.1f}, Vol={recent_volatility:.1%}")
            
            return {
                'tp_pct': tp_pct,
                'sl_pct': sl_pct,
                'tp_price': tp_price,
                'sl_price': sl_price,
                'signal_strength': signal_strength,
                'volatility': recent_volatility,
                'atr': atr,
                'rsi': rsi,
                'method': 'legacy_technical_analysis'
            }
            
        except Exception as e:
            logger.error(f"Legacy TA failed for {symbol}: {e}")
            return self.get_emergency_fallback_targets(side, symbol)
            
        except Exception as e:
            logger.warning(f"Error calculating world-class targets for {symbol}: {e}")
            return self.get_emergency_fallback_targets(side, symbol)
    
    def get_emergency_fallback_targets(self, side: str, symbol: str) -> Dict[str, float]:
        """
        Emergency fallback when world-class TA completely fails
        Uses minimal technical analysis instead of hardcoded percentages
        """
        logger.warning(f"⚠️  Using emergency fallback for {symbol}")
        
        try:
            # Try to get at least basic ATR from recent bars
            bars = self.get_bars_from_binance(symbol, '15m', 50)
            if bars and len(bars) >= 14:
                df = pd.DataFrame(bars)
                atr = self.calculate_atr(df['high'], df['low'], df['close'], 14)
                current_price = df['close'].iloc[-1]
                atr_pct = atr / current_price if current_price > 0 else 0.02
                
                # Conservative ATR-based targets
                tp_pct = atr_pct * 2.0  # 2x ATR for TP
                sl_pct = atr_pct * 1.0  # 1x ATR for SL
                
                # Cap at reasonable values
                tp_pct = max(0.01, min(0.05, tp_pct))  # 1% to 5%
                sl_pct = max(0.005, min(0.03, sl_pct))  # 0.5% to 3%
                
                return {
                    'tp_pct': tp_pct,
                    'sl_pct': sl_pct,
                    'signal_strength': 0.3,  # Low confidence
                    'volatility': atr_pct,
                    'method': 'emergency_atr_based'
                }
            else:
                raise Exception("No bars available for ATR calculation")
                
        except Exception as e:
            logger.error(f"❌ Emergency fallback failed: {e}")
            # Last resort: ultra-conservative hardcoded values
            return {
                'tp_pct': 0.015,  # 1.5% TP (very conservative)
                'sl_pct': 0.01,   # 1% SL (very tight)
                'signal_strength': 0.2,  # Very low confidence
                'volatility': 0.02,
                'method': 'last_resort_hardcoded'
            }
    
    def calculate_atr(self, highs, lows, closes, period=14):
        """Calculate Average True Range"""
        import numpy as np
        
        high_low = highs - lows
        high_close = np.abs(highs - np.roll(closes, 1))
        low_close = np.abs(lows - np.roll(closes, 1))
        
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = np.mean(true_range[-period:])  # Simple average of last N periods
        
        return atr
    
    def calculate_rsi(self, closes, period=14):
        """Calculate RSI"""
        import numpy as np
        
        deltas = np.diff(closes)
        seed = deltas[:period+1]
        up = seed[seed >= 0].sum() / period
        down = -seed[seed < 0].sum() / period
        
        if down == 0:
            return 100
        
        rs = up / down
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def calculate_bollinger_bands(self, closes, period=20, std=2):
        """Calculate Bollinger Bands"""
        import numpy as np
        
        sma = np.mean(closes[-period:])
        std_dev = np.std(closes[-period:])
        
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        
        return upper, lower, sma
    
    def get_binance_bars_direct(self, symbol: str, interval: str, limit: int):
        """Get bars directly from Binance API as fallback"""
        try:
            import requests
            import pandas as pd
            
            base_url = 'https://testnet.binancefuture.com'
            endpoint = '/fapi/v1/klines'
            
            params = {
                'symbol': symbol,
                'interval': interval,
                'limit': limit
            }
            
            response = requests.get(base_url + endpoint, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                # Convert to DataFrame
                df = pd.DataFrame(data, columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_volume', 'count', 'taker_buy_volume',
                    'taker_buy_quote_volume', 'ignore'
                ])
                
                # Convert to proper types
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = pd.to_numeric(df[col])
                
                logger.info(f"✅ Got {len(df)} bars for {symbol} from direct API")
                return df
            else:
                logger.warning(f"Direct API failed for {symbol}: {response.status_code}")
                return None
                
        except Exception as e:
            logger.warning(f"Error in direct API call for {symbol}: {e}")
            return None

    # =============================================================================

    def check_internet_connectivity(self) -> bool:
        """Check if internet connectivity is available with multiple fallback methods"""
        try:
            import requests
            import socket
            
            # Method 1: Try multiple reliable endpoints
            endpoints = [
                'https://httpbin.org/status/200',
                'https://api.binance.com/api/v3/ping',
                'https://www.google.com',
                'https://httpbin.org/get'
            ]
            
            for endpoint in endpoints:
                try:
                    logger.info(f"🌐 Testing connectivity to: {endpoint}")
                    response = requests.get(endpoint, timeout=5)
                    if response.status_code in [200, 301, 302]:
                        logger.info(f"✅ Internet connectivity confirmed via {endpoint}")
                        return True
                except Exception as e:
                    logger.warning(f"❌ Failed to reach {endpoint}: {e}")
                    continue
            
            # Method 2: Try socket connection to common ports
            try:
                logger.info("🌐 Testing socket connectivity...")
                socket.create_connection(("8.8.8.8", 53), timeout=3)  # Google DNS
                logger.info("✅ Internet connectivity confirmed via socket")
                return True
            except Exception as e:
                logger.warning(f"❌ Socket connectivity test failed: {e}")
            
            # Method 3: Try to resolve a domain
            try:
                logger.info("🌐 Testing DNS resolution...")
                socket.gethostbyname("google.com")
                logger.info("✅ DNS resolution successful")
                return True
            except Exception as e:
                logger.warning(f"❌ DNS resolution failed: {e}")
            
            logger.warning("🌐 All connectivity tests failed")
            return False
            
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
        logger.info(f"🔍 Checking {len(self.positions)} tracked positions...")

        # First, get actual platform positions to ensure we're in sync
        try:
            platform_status = get_futures_status()
            if 'error' not in platform_status:
                platform_positions = platform_status.get('positions', [])
                logger.info(f"📊 Platform has {len(platform_positions)} actual positions")

                # Sync any missing positions
                for pos in platform_positions:
                    symbol = pos.get('symbol', '').replace('USDT', '') + 'USDT'
                    if symbol not in self.positions and symbol in self.symbols:
                        logger.info(f"🔄 Found unsynced position: {symbol}, syncing...")
                        self.positions[symbol] = {
                            'side': pos.get('side', 'buy'),
                            'entry_price': float(pos.get('entry_price', 0)),
                            'quantity': float(pos.get('quantity', 0)),
                            'leverage': int(pos.get('leverage', 1)),
                            'platform': pos.get('platform', self.current_platform),
                            'timestamp': datetime.now(timezone.utc).isoformat(),
                            'signal': {'signal': 'synced', 'strength': 0, 'reason': 'platform_sync'},
                            'highest_price': float(pos.get('entry_price', 0)),
                            'lowest_price': float(pos.get('entry_price', 0)),
                            'trailing_stop_pct': 0.05,
                            'profit_target_pct': 0.08
                        }
                        logger.info(f"✅ Synced position: {symbol} {pos.get('side')} x{pos.get('quantity')}")
        except Exception as e:
            logger.warning(f"Error syncing platform positions: {e}")

        # Now check all tracked positions
        for symbol, position in list(self.positions.items()):
            try:
                logger.info(f"📊 Checking position: {symbol} ({position.get('side', 'buy')})")

                # Get current price
                data = enhanced_futures_bars(symbol, '1h', 1)
                if data is None or len(data) == 0:
                    logger.warning(f"❌ No price data for {symbol}, skipping...")
                    continue

                current_price = data['close'].iloc[-1]
                if current_price is None:
                    logger.warning(f"❌ Invalid price for {symbol}, skipping...")
                    continue

                entry_price = position.get('entry_price', 0)
                if entry_price is None or entry_price == 0:
                    logger.warning(f"❌ Invalid entry price for {symbol}, skipping...")
                    continue

                side = position.get('side', 'buy')
                leverage = position.get('leverage', 1) or 1

                # Calculate current P&L (unleveraged price movement)
                if side == 'buy':
                    pnl_pct = (current_price - entry_price) / entry_price
                else:
                    pnl_pct = (entry_price - current_price) / entry_price

                # Calculate leveraged ROI (what's shown on live platform)
                leveraged_roi = pnl_pct * leverage
                
                logger.info(f"📈 {symbol}: Entry=${entry_price:.2f}, Current=${current_price:.2f}, P&L={pnl_pct:.2%}")
                logger.info(f"⚡ {symbol}: Leveraged ROI={leveraged_roi:+.1%} (Price: {pnl_pct:+.2%} × {leverage}x leverage)")

                # Update trailing stop levels
                self.update_trailing_stops(symbol, current_price, position)

                # Check exit conditions using leveraged ROI
                exit_reason = self.should_exit_position(symbol, current_price, pnl_pct, leveraged_roi, position)

                if exit_reason:
                    logger.info(f"🚨 Exit condition met for {symbol}: {exit_reason}")
                    self.close_position(symbol, exit_reason)
                else:
                    logger.info(f"✅ {symbol} position OK, no exit condition met")

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

    def should_exit_position(self, symbol: str, current_price: float, pnl_pct: float, leveraged_roi: float, position: Dict) -> str:
        """Determine if position should be exited based on intelligent TP/SL conditions"""
        try:
            side = position.get('side', 'buy')
            entry_price = position.get('entry_price', 0)
            
            # Use intelligent TP/SL if enabled, otherwise fall back to fixed levels
            use_intelligent_tpsl = os.getenv("TB_INTELLIGENT_FUTURES_TPSL", "1") == "1"
            
            if use_intelligent_tpsl:
                # Calculate technical analysis-based targets
                logger.info(f"🔬 Calculating technical targets for {symbol}...")
                targets = self.calculate_world_class_futures_targets(symbol, entry_price, side)
                
                # Use technical analysis targets
                profit_target_pct = targets['tp_pct']
                stop_loss_pct = targets['sl_pct']
                
                method = targets.get('method', 'technical')
                if method == 'technical_analysis':
                    logger.info(f"🧠 {symbol} TECHNICAL targets: TP={profit_target_pct:+.1%} SL={-stop_loss_pct:+.1%} PRICE")
                    logger.info(f"   📊 Based on: ATR=${targets.get('atr', 0):.4f} | RSI={targets.get('rsi', 50):.1f} | Volatility={targets.get('volatility', 0):.1%}")
                    logger.info(f"   🎯 TP Price: ${targets.get('tp_price', 0):.2f} | SL Price: ${targets.get('sl_price', 0):.2f}")
                else:
                    logger.info(f"🔧 {symbol} DEFAULT targets: TP={profit_target_pct:+.1%} SL={-stop_loss_pct:+.1%} PRICE (technical analysis failed)")
                
                # Show leveraged ROI equivalent for reference  
                leverage = position.get('leverage', 25) or 25
                roi_tp_example = profit_target_pct * leverage
                roi_sl_example = -stop_loss_pct * leverage
                logger.info(f"   💡 Leveraged ROI equivalent: TP≈{roi_tp_example:+.0%} SL≈{roi_sl_example:+.0%} (for reference)")
            else:
                # Fall back to fixed levels (PRICE-based, not leveraged!)
                profit_target_pct = 0.015  # 1.5% price movement (37% ROI with 25x)
                stop_loss_pct = 0.02       # 2% price movement (50% ROI loss with 25x)
                
                leverage = position.get('leverage', 25) or 25
                roi_tp_example = profit_target_pct * leverage
                roi_sl_example = -stop_loss_pct * leverage
                
                logger.info(f"🔍 {symbol} fixed targets: TP={profit_target_pct:+.1%} SL={-stop_loss_pct:+.1%} PRICE")
                logger.info(f"   💡 Leveraged ROI equivalent: TP≈{roi_tp_example:+.0%} SL≈{roi_sl_example:+.0%} (for reference)")

            # 1. Profit target hit (compare unleveraged price movement to price targets)
            if pnl_pct >= profit_target_pct:
                logger.info(f"🎯 {symbol} hit profit target: {pnl_pct:+.2%} >= {profit_target_pct:+.1%} PRICE")
                logger.info(f"   🚀 Leveraged ROI: {leveraged_roi:+.1%}")
                return 'profit_target'

            # 2. Stop loss hit (compare unleveraged price movement to price targets)
            if pnl_pct <= -stop_loss_pct:
                logger.info(f"🛑 {symbol} hit stop loss: {pnl_pct:+.2%} <= {-stop_loss_pct:+.1%} PRICE")
                logger.info(f"   💸 Leveraged ROI: {leveraged_roi:+.1%}")
                return 'stop_loss'

            # 3. Trailing stop hit
            if 'trailing_stop_price' in position:
                trailing_stop_price = position['trailing_stop_price']
                logger.info(f"🎣 {symbol} trailing stop: Current=${current_price:.2f}, Stop=${trailing_stop_price:.2f}")
                if side == 'buy' and current_price <= trailing_stop_price:
                    logger.info(f"🎣 {symbol} trailing stop hit (long): ${current_price:.2f} <= ${trailing_stop_price:.2f}")
                    return 'trailing_stop'
                elif side == 'sell' and current_price >= trailing_stop_price:
                    logger.info(f"🎣 {symbol} trailing stop hit (short): ${current_price:.2f} >= ${trailing_stop_price:.2f}")
                    return 'trailing_stop'

            # 4. Maximum loss limit (4% price loss = 100% leveraged ROI loss with 25x)
            max_loss_pct = -0.04  # 4% price loss
            if pnl_pct <= max_loss_pct:
                logger.info(f"💀 {symbol} hit max loss limit: {pnl_pct:+.2%} <= {max_loss_pct:+.1%} PRICE")
                logger.info(f"   💸 Leveraged ROI: {leveraged_roi:+.1%}")
                return 'max_loss_limit'

            # 5. Time-based exit (if position is too old)
            position_timestamp = datetime.fromisoformat(position['timestamp'])
            if position_timestamp.tzinfo is not None:
                # Convert to naive datetime for comparison
                position_timestamp = position_timestamp.replace(tzinfo=None)
            current_time = datetime.now().replace(tzinfo=None)
            position_age_hours = (current_time - position_timestamp).total_seconds() / 3600
            if position_age_hours > 24:  # Close after 24 hours
                logger.info(f"⏰ {symbol} hit time limit: {position_age_hours:.1f}h > 24h")
                return 'time_limit'

            # 6. Volatility-based exit (if volatility spikes)
            try:
                data = enhanced_futures_bars(symbol, '1h', 6)  # Last 6 hours
                if data is not None and len(data) >= 6:
                    recent_volatility = data['close'].pct_change().std()
                    volatility_threshold = 0.08
                    if recent_volatility > volatility_threshold:
                        logger.info(f"🌊 {symbol} high volatility exit: {recent_volatility:.4f} > {volatility_threshold}")
                        return 'high_volatility'
            except Exception as e:
                logger.warning(f"Error checking volatility for {symbol}: {e}")

            logger.info(f"✅ {symbol} no exit condition met, holding position")
            return None  # No exit condition met

        except Exception as e:
            logger.warning(f"Error checking exit conditions for {symbol}: {e}")
            return None

    def close_position(self, symbol: str, reason: str):
        """Close a position on the platform and update tracking"""
        if symbol not in self.positions:
            logger.warning(f"⚠️ Position {symbol} not found in tracking, cannot close")
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
            side = position.get('side', 'buy')

            # Calculate final P&L
            if side == 'buy':
                pnl_pct = (exit_price - entry_price) / entry_price if entry_price > 0 else 0
            else:
                pnl_pct = (entry_price - exit_price) / exit_price if exit_price > 0 else 0

            pnl_amount = pnl_pct * quantity * leverage
            self.daily_pnl += pnl_amount

            logger.info(f"🔄 Closing {symbol} position: {reason}")
            logger.info(f"   Entry: ${entry_price:.2f} | Exit: ${exit_price:.2f}")
            logger.info(f"   P&L: ${pnl_amount:.2f} ({pnl_pct:.2%})")

            # Actually close the position on the platform
            try:
                # Determine the opposite side for closing
                close_side = 'sell' if side == 'buy' else 'buy'

                # Place a market order to close the position
                close_result = execute_futures_trade(symbol, close_side, {
                    'position_value': abs(quantity),  # Use absolute quantity
                    'leverage_used': leverage,
                    'platform': self.current_platform
                })

                if 'error' in close_result:
                    logger.error(f"❌ Failed to close {symbol} position on platform: {close_result['error']}")
                    # Still remove from tracking even if platform close failed
                    logger.warning(f"⚠️ Removing {symbol} from tracking despite platform close failure")
                else:
                    logger.info(f"✅ Successfully closed {symbol} position on platform")

            except Exception as e:
                logger.error(f"❌ Exception closing {symbol} position on platform: {e}")
                logger.warning(f"⚠️ Removing {symbol} from tracking despite platform close failure")

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

            # Record exit in trade log
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

            # Remove position from tracking
            del self.positions[symbol]
            logger.info(f"🗑️ Removed {symbol} from position tracking")

        except Exception as e:
            logger.error(f"Error closing position for {symbol}: {e}")
            # Still try to remove from tracking even if there was an error
            if symbol in self.positions:
                del self.positions[symbol]
                logger.warning(f"🗑️ Force-removed {symbol} from tracking due to error")

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

        # Sync existing positions on EVERY RUN for critical risk management
        # This ensures positions are always managed immediately
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
        consecutive_failures = 0
        max_consecutive_failures = 5
        
        while True:
            try:
                cycle_count += 1
                
                # Check internet connectivity before starting cycle
                has_internet = self.check_internet_connectivity()
                
                if not has_internet:
                    consecutive_failures += 1
                    logger.warning(f"🌐 No internet connectivity (failure {consecutive_failures}/{max_consecutive_failures}), waiting 60s before retry...")
                    
                    if consecutive_failures >= max_consecutive_failures:
                        logger.warning("🌐 Too many consecutive connectivity failures, entering offline mode")
                        logger.warning("🌐 Will continue with limited functionality (no new trades, position monitoring only)")
                        # Continue with limited functionality
                        break
                    
                    time.sleep(60)
                    continue
                
                # Reset failure counter on successful connectivity
                consecutive_failures = 0
                
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
            
            # Log what we found on the platform
            logger.info(f"🔍 Found {len(platform_positions)} positions on platform to sync")
            
            for pos in platform_positions:
                symbol = pos.get('symbol', '').replace('USDT', '') + 'USDT'  # Normalize symbol
                
                # Debug logging
                in_symbols = symbol in self.symbols
                in_positions = symbol not in self.positions
                logger.info(f"🔍 Checking {symbol}: in_symbols={in_symbols}, not_in_positions={in_positions}")
                
                if symbol in self.symbols and symbol not in self.positions:
                    # Import position into agent management (IGNORE max_positions during sync)
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
                elif symbol not in self.symbols:
                    logger.info(f"⏭️  Skipping {symbol}: not in configured symbols")
                elif symbol in self.positions:
                    logger.info(f"⏭️  Skipping {symbol}: already in agent positions")
            
            logger.info(f"📊 Total positions after sync: {len(self.positions)} (was {len(self.positions) - synced_count})")
            
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
