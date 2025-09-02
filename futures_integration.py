#!/usr/bin/env python3
"""
Futures & Perpetuals Integration for Trading Agent
Seamlessly integrates futures/perpetuals platforms with existing agent
"""

import os
import sys
import json
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, use system env vars

# Import our futures platform
from futures_trading_platform import (
    futures_platform,
    get_futures_data,
    get_perpetuals_data,
    place_futures_order,
    switch_futures_platform,
    get_futures_platforms,
    get_futures_platform_info
)

logger = logging.getLogger(__name__)

class FuturesIntegration:
    """Integration layer for futures and perpetuals trading"""

    def __init__(self):
        self.futures_enabled = os.getenv("TB_ENABLE_FUTURES_TRADING", "0") == "1"
        self.preferred_platform = os.getenv("TB_FUTURES_PLATFORM", "Bybit Futures")  # Default to Bybit
        self.leverage_limit = int(os.getenv("TB_MAX_LEVERAGE", "10"))
        self.paper_trading = os.getenv("TB_PAPER_TRADING", "1") == "1"

        # Per-platform configuration
        self.platform_configs = {
            'binance': {
                'max_trade_size': float(os.getenv("BINANCE_MAX_TRADE_SIZE", "100")),
                'max_leverage': int(os.getenv("BINANCE_MAX_LEVERAGE", "25")),
                'paper_capital': float(os.getenv("BINANCE_PAPER_CAPITAL", "15000"))
            },
            'bybit': {
                'max_trade_size': float(os.getenv("BYBIT_MAX_TRADE_SIZE", "500")),
                'max_leverage': int(os.getenv("BYBIT_MAX_LEVERAGE", "100")),
                'paper_capital': float(os.getenv("BYBIT_PAPER_CAPITAL", "240000"))
            },
            'Binance Futures': {
                'max_trade_size': float(os.getenv("BINANCE_MAX_TRADE_SIZE", "100")),
                'max_leverage': int(os.getenv("BINANCE_MAX_LEVERAGE", "25")),
                'paper_capital': float(os.getenv("BINANCE_PAPER_CAPITAL", "15000"))
            },
            'Bybit Futures': {
                'max_trade_size': float(os.getenv("BYBIT_MAX_TRADE_SIZE", "500")),
                'max_leverage': int(os.getenv("BYBIT_MAX_LEVERAGE", "100")),
                'paper_capital': float(os.getenv("BYBIT_PAPER_CAPITAL", "240000"))
            }
        }

        # Smart leverage configuration
        self.leverage_risk_multiplier = float(os.getenv("LEVERAGE_RISK_MULTIPLIER", "1.5"))
        self.leverage_volatility_threshold = float(os.getenv("LEVERAGE_VOLATILITY_THRESHOLD", "0.05"))
        self.leverage_market_regime_multiplier = float(os.getenv("LEVERAGE_MARKET_REGIME_MULTIPLIER", "1.2"))

        if self.futures_enabled:
            self._initialize_futures()

    def switch_platform(self, platform_name: str) -> bool:
        """Switch to a different futures platform"""
        # Handle both naming conventions
        platform_mapping = {
            'binance': 'Binance Futures',
            'bybit': 'Bybit Futures',
            'Binance Futures': 'Binance Futures',
            'Bybit Futures': 'Bybit Futures'
        }

        actual_platform_name = platform_mapping.get(platform_name, platform_name)

        if actual_platform_name in get_futures_platforms():
            switch_futures_platform(actual_platform_name)
            self.preferred_platform = actual_platform_name
            logger.info(f"🔄 Switched to {actual_platform_name} platform")
            return True
        logger.warning(f"⚠️ Platform {platform_name} not available")
        return False

    def get_platform_config(self, platform_name: str = None) -> Dict:
        """Get configuration for a specific platform"""
        name = platform_name or self.preferred_platform
        return self.platform_configs.get(name, {})

    def calculate_smart_leverage(self, symbol: str, base_leverage: int, volatility: float,
                                market_regime: str = 'unknown') -> int:
        """Calculate smart leverage based on risk/reward analysis"""
        try:
            # Start with platform-specific max leverage
            platform_config = self.get_platform_config()
            max_platform_leverage = platform_config.get('max_leverage', self.leverage_limit)

            # Base leverage (respect platform limits)
            smart_leverage = min(base_leverage, max_platform_leverage)

            # Adjust based on volatility
            if volatility > self.leverage_volatility_threshold:
                # Reduce leverage in high volatility
                volatility_reduction = min(0.5, (volatility - self.leverage_volatility_threshold) * 2)
                smart_leverage = int(smart_leverage * (1 - volatility_reduction))
            else:
                # Can use slightly higher leverage in low volatility
                smart_leverage = int(smart_leverage * 1.1)

            # Adjust based on market regime
            if market_regime == 'trending':
                # Higher leverage in trending markets
                smart_leverage = int(smart_leverage * self.leverage_market_regime_multiplier)
            elif market_regime == 'ranging':
                # Lower leverage in ranging markets
                smart_leverage = int(smart_leverage * 0.8)

            # Apply risk multiplier
            smart_leverage = int(smart_leverage * self.leverage_risk_multiplier)

            # Ensure minimum leverage of 1 and respect platform limits
            return max(1, min(smart_leverage, max_platform_leverage))

        except Exception as e:
            logger.warning(f"Error calculating smart leverage: {e}")
            return min(base_leverage, max_platform_leverage)

    def _initialize_futures(self):
        """Initialize futures trading capabilities"""
        try:
            # Switch to preferred platform
            if self.preferred_platform in get_futures_platforms():
                switch_futures_platform(self.preferred_platform)
                logger.info(f"🎯 Futures platform: {self.preferred_platform}")
            else:
                logger.warning(f"⚠️  Preferred platform {self.preferred_platform} not available")
                available = get_futures_platforms()
                if available:
                    switch_futures_platform(available[0])
                    logger.info(f"🔄 Using {available[0]} instead")

        except Exception as e:
            logger.error(f"Failed to initialize futures: {e}")
            self.futures_enabled = False

    def is_futures_available(self) -> bool:
        """Check if futures trading is available"""
        return self.futures_enabled and bool(get_futures_platforms())

    def get_futures_bars(self, symbol: str, timeframe: str = '1h', limit: int = 500) -> Optional[pd.DataFrame]:
        """Get futures bars for a symbol"""
        if not self.is_futures_available():
            return None

        try:
            # Try futures first, then perpetuals
            data = get_futures_data(symbol, timeframe, limit)
            if data is None or len(data) == 0:
                # Try perpetuals format (e.g., BTC-PERP)
                perp_symbol = f"{symbol.replace('USDT', '')}-PERPETUAL"
                data = get_perpetuals_data(perp_symbol, timeframe, limit)

            return data

        except Exception as e:
            logger.warning(f"Failed to get futures data for {symbol}: {e}")
            return None

    def calculate_futures_position_size(self, symbol: str, capital: float, risk_pct: float = 0.02,
                                      volatility_multiplier: float = 1.0, market_regime: str = 'unknown') -> Dict:
        """Calculate position size for futures trading with platform-specific limits"""
        if not self.is_futures_available():
            return {'error': 'Futures not available'}

        try:
            # Get platform-specific configuration
            platform_config = self.get_platform_config()
            platform_capital = platform_config.get('paper_capital', capital)
            max_trade_size = platform_config.get('max_trade_size', float('inf'))

            # Get recent data for volatility calculation
            data = self.get_futures_bars(symbol, '1h', 100)
            if data is None or len(data) < 20:
                return {'error': 'Insufficient data for position sizing'}

            # Calculate volatility (standard deviation of returns)
            returns = data['close'].pct_change().dropna()
            volatility = returns.std()

            # Kelly criterion with volatility adjustment
            kelly_pct = (returns.mean() / (volatility ** 2)) * volatility_multiplier

            # Risk-adjusted position size
            risk_amount = platform_capital * risk_pct

            # Calculate base position value
            position_value = risk_amount / volatility if volatility > 0 else risk_amount

            # Apply platform-specific leverage limits
            max_platform_leverage = platform_config.get('max_leverage', self.leverage_limit)
            leveraged_position = position_value * max_platform_leverage

            # Ensure we don't exceed platform capital
            final_position = min(leveraged_position, platform_capital * max_platform_leverage)

            # Apply trade size limits
            if max_trade_size < float('inf'):
                # Get current price to calculate trade value
                current_price = data['close'].iloc[-1]
                max_position_by_size = max_trade_size * max_platform_leverage / current_price
                final_position = min(final_position, max_position_by_size)

            # Calculate smart leverage
            smart_leverage = self.calculate_smart_leverage(symbol, max_platform_leverage, volatility, market_regime)

            return {
                'symbol': symbol,
                'position_value': final_position,
                'leverage_used': smart_leverage,
                'risk_amount': risk_amount,
                'volatility': volatility,
                'kelly_pct': kelly_pct,
                'platform': self.preferred_platform,
                'platform_capital': platform_capital,
                'max_trade_size': max_trade_size,
                'mode': 'paper_trading' if self.paper_trading else 'live'
            }

        except Exception as e:
            logger.error(f"Position sizing failed for {symbol}: {e}")
            return {'error': str(e)}

    def place_futures_trade(self, symbol: str, side: str, position_info: Dict,
                           order_type: str = 'market') -> Dict:
        """Place a futures trade"""
        if not self.is_futures_available():
            return {'error': 'Futures not available'}

        try:
            # Extract position details
            quantity = position_info.get('position_value', 0)
            leverage = int(position_info.get('leverage_used', 1))

            if quantity <= 0:
                return {'error': 'Invalid position size'}

            # Place the order
            order_result = place_futures_order(
                symbol=symbol,
                side=side,
                quantity=quantity,
                order_type=order_type,
                leverage=leverage
            )

            # Add position info
            order_result.update({
                'position_info': position_info,
                'timestamp': datetime.now().isoformat(),
                'platform': self.preferred_platform
            })

            logger.info(f"📈 Futures {side} order placed: {symbol} x{leverage}")
            return order_result

        except Exception as e:
            logger.error(f"Futures trade failed for {symbol}: {e}")
            return {'error': str(e)}

    def get_futures_portfolio_status(self) -> Dict:
        """Get current futures portfolio status"""
        if not self.is_futures_available():
            return {'error': 'Futures not available'}

        try:
            # Get positions from all platforms
            positions = futures_platform.get_positions()

            # Get account balance
            balance = futures_platform.get_account_balance()

            return {
                'positions': positions,
                'balance': balance,
                'total_positions': len(positions),
                'platforms': get_futures_platforms(),
                'active_platform': self.preferred_platform,
                'mode': 'paper_trading' if self.paper_trading else 'live',
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Failed to get portfolio status: {e}")
            return {'error': str(e)}

def get_account_balance() -> Dict:
    """Get account balance from current platform"""
    return futures_platform.get_account_balance()

# Global instance
futures_integration = FuturesIntegration()

def enhanced_futures_bars(symbol: str, timeframe: str = '1h', limit: int = 500) -> Optional[pd.DataFrame]:
    """Enhanced function to get futures bars (drop-in replacement)"""
    return futures_integration.get_futures_bars(symbol, timeframe, limit)

def calculate_futures_position(symbol: str, capital: float, risk_pct: float = 0.02) -> Dict:
    """Calculate futures position size"""
    return futures_integration.calculate_futures_position_size(symbol, capital, risk_pct)

def execute_futures_trade(symbol: str, side: str, position_info: Dict) -> Dict:
    """Execute futures trade"""
    return futures_integration.place_futures_trade(symbol, side, position_info)

def get_futures_status() -> Dict:
    """Get futures trading status"""
    return futures_integration.get_futures_portfolio_status()

if __name__ == "__main__":
    # Test futures integration
    print("🧪 Testing Futures Integration")
    print("=" * 50)

    if futures_integration.is_futures_available():
        print("✅ Futures trading enabled")

        # Test data fetching
        print("\n📊 Testing BTC Futures Data...")
        data = enhanced_futures_bars("BTCUSDT", "1h", 10)
        if data is not None:
            print(f"✅ Got {len(data)} bars")
            print(".2f")
        else:
            print("❌ No data received")

        # Test position sizing
        print("\n📏 Testing Position Sizing...")
        pos_info = calculate_futures_position("BTCUSDT", 10000.0, 0.02)
        if 'error' not in pos_info:
            print(f"✅ Position calculated: ${pos_info['position_value']:.2f} at {pos_info['leverage_used']:.1f}x leverage")
        else:
            print(f"❌ Position sizing failed: {pos_info['error']}")

        # Test portfolio status
        print("\n📊 Portfolio Status:")
        status = get_futures_status()
        if 'error' not in status:
            print(f"💰 Balance: ${status['balance'].get('total_balance', 0):.2f}")
            print(f"📊 Positions: {status['total_positions']}")
            print(f"🏛️  Platform: {status['active_platform']}")
        else:
            print(f"❌ Status check failed: {status['error']}")

    else:
        print("❌ Futures trading not available")
        print("💡 Enable with: TB_ENABLE_FUTURES_TRADING=1")
