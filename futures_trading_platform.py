#!/usr/bin/env python3
"""
Advanced Futures & Perpetuals Trading Platform
Free alternatives to Alpaca for comprehensive crypto derivatives trading
Supports futures, perpetuals, options, and multi-asset strategies
"""

import os
import time
import json
import pandas as pd
import requests
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple, Union
import logging
import hmac
import hashlib
import base64
from urllib.parse import urlencode

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, use system env vars

logger = logging.getLogger(__name__)

class FuturesTradingPlatform:
    """Unified interface for multiple futures and perpetuals platforms"""

    def __init__(self):
        self.platforms = {}
        self.active_platform = None
        self._initialize_platforms()

    def _initialize_platforms(self):
        """Initialize all available futures platforms"""
        platforms = [
            BinanceFuturesPlatform(),
            BybitFuturesPlatform(),
            BitMEXFuturesPlatform(),
            DeribitFuturesPlatform(),
        ]

        for platform in platforms:
            if platform.is_available():
                self.platforms[platform.name] = platform
                logger.info(f"✅ {platform.name} initialized")

        if self.platforms:
            # Set default to Binance (most comprehensive)
            self.active_platform = list(self.platforms.keys())[0]
            logger.info(f"🎯 Active platform: {self.active_platform}")

    def switch_platform(self, platform_name: str) -> bool:
        """Switch active trading platform"""
        if platform_name in self.platforms:
            self.active_platform = platform_name
            logger.info(f"🔄 Switched to {platform_name}")
            return True
        return False

    def get_available_platforms(self) -> List[str]:
        """Get list of available platforms"""
        return list(self.platforms.keys())

    def get_platform_info(self, platform_name: str = None) -> Dict:
        """Get information about a platform"""
        name = platform_name or self.active_platform
        if name in self.platforms:
            platform = self.platforms[name]
            return {
                'name': platform.name,
                'type': platform.platform_type,
                'features': platform.features,
                'instruments': platform.get_available_instruments(),
                'status': 'active' if platform.is_available() else 'inactive'
            }
        return {}

    def get_futures_data(self, symbol: str, timeframe: str = '1h', limit: int = 500) -> Optional[pd.DataFrame]:
        """Get futures/perpetuals data from active platform"""
        if not self.active_platform:
            return None

        platform = self.platforms[self.active_platform]
        return platform.get_futures_data(symbol, timeframe, limit)

    def get_perpetuals_data(self, symbol: str, timeframe: str = '1h', limit: int = 500) -> Optional[pd.DataFrame]:
        """Get perpetuals data"""
        if not self.active_platform:
            return None

        platform = self.platforms[self.active_platform]
        return platform.get_perpetuals_data(symbol, timeframe, limit)

    def place_futures_order(self, symbol: str, side: str, quantity: float, price: Optional[float] = None,
                           order_type: str = 'market', leverage: int = 1) -> Dict:
        """Place futures order (paper trading only)"""
        if not self.active_platform:
            return {'error': 'No active platform'}

        platform = self.platforms[self.active_platform]
        return platform.place_futures_order(symbol, side, quantity, price, order_type, leverage)

    def get_positions(self) -> List[Dict]:
        """Get current positions across all platforms"""
        positions = []
        for name, platform in self.platforms.items():
            try:
                platform_positions = platform.get_positions()
                for pos in platform_positions:
                    pos['platform'] = name
                    positions.append(pos)
            except Exception as e:
                logger.warning(f"Error getting positions from {name}: {e}")
        return positions

    def get_account_balance(self) -> Dict:
        """Get account balance from active platform"""
        if not self.active_platform:
            return {'error': 'No active platform'}

        platform = self.platforms[self.active_platform]
        return platform.get_account_balance()

class BinanceFuturesPlatform:
    """Binance Futures Testnet Platform"""

    def __init__(self):
        self.name = "Binance Futures"
        self.platform_type = "futures_perpetuals"
        self.base_url = "https://testnet.binancefuture.com"
        self.api_key = os.getenv("BINANCE_TESTNET_API_KEY", "")
        self.secret_key = os.getenv("BINANCE_TESTNET_SECRET_KEY", "")
        # Per-platform limits from environment
        self.max_trade_size = float(os.getenv("BINANCE_MAX_TRADE_SIZE", "100"))  # $100 default
        self.max_leverage = int(os.getenv("BINANCE_MAX_LEVERAGE", "25"))  # 25x default
        self.paper_capital = float(os.getenv("BINANCE_PAPER_CAPITAL", "15000"))  # 15k USDT
        self.features = [
            'futures', 'perpetuals', 'linear', 'inverse',
            'leverage_up_to_125x', 'multiple_timeframes',
            'high_frequency_data', 'paper_trading',
            'per_platform_limits', 'smart_leverage'
        ]

    def is_available(self) -> bool:
        """Check if Binance testnet is available"""
        try:
            # Test basic connectivity
            response = requests.get(f"{self.base_url}/fapi/v1/ping", timeout=5)
            return response.status_code == 200
        except:
            return False

    def _generate_signature(self, query_string: str) -> str:
        """Generate HMAC SHA256 signature for Binance"""
        if not self.secret_key:
            return ""
        return hmac.new(
            self.secret_key.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()

    def _make_request(self, method: str, endpoint: str, params: Dict = None) -> Optional[Dict]:
        """Make authenticated request to Binance"""
        try:
            url = f"{self.base_url}{endpoint}"

            if params is None:
                params = {}

            # Add timestamp for authenticated requests
            if self.api_key:
                params['timestamp'] = int(time.time() * 1000)
                query_string = urlencode(params)
                params['signature'] = self._generate_signature(query_string)

                headers = {
                    'X-MBX-APIKEY': self.api_key
                }
            else:
                headers = {}
                query_string = urlencode(params) if params else ""

            if query_string:
                url += f"?{query_string}"

            response = requests.request(method, url, headers=headers, timeout=10)
            response.raise_for_status()
            return response.json()

        except Exception as e:
            logger.warning(f"Binance request failed: {e}")
            return None

    def get_available_instruments(self) -> List[str]:
        """Get available futures instruments"""
        try:
            data = self._make_request('GET', '/fapi/v1/exchangeInfo')
            if data and 'symbols' in data:
                return [symbol['symbol'] for symbol in data['symbols']
                       if symbol['contractType'] in ['PERPETUAL', 'CURRENT_QUARTER']]
            return []
        except:
            return ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'SOLUSDT']  # Fallback

    def get_futures_data(self, symbol: str, timeframe: str = '1h', limit: int = 500) -> Optional[pd.DataFrame]:
        """Get futures kline data"""
        try:
            # Map timeframes
            interval_map = {
                '1m': '1m', '5m': '5m', '15m': '15m', '30m': '30m',
                '1h': '1h', '2h': '2h', '4h': '4h', '6h': '6h',
                '12h': '12h', '1d': '1d', '3d': '3d'
            }
            interval = interval_map.get(timeframe, '1h')

            params = {
                'symbol': symbol,
                'interval': interval,
                'limit': min(limit, 1000)
            }

            data = self._make_request('GET', '/fapi/v1/klines', params)

            if not data:
                return None

            # Convert to DataFrame
            df_data = []
            for kline in data:
                df_data.append({
                    'timestamp': datetime.fromtimestamp(kline[0] / 1000, tz=timezone.utc),
                    'open': float(kline[1]),
                    'high': float(kline[2]),
                    'low': float(kline[3]),
                    'close': float(kline[4]),
                    'volume': float(kline[5]),
                    'close_time': datetime.fromtimestamp(kline[6] / 1000, tz=timezone.utc),
                    'quote_volume': float(kline[7]),
                    'trades': int(kline[8]),
                    'taker_buy_volume': float(kline[9]),
                    'taker_buy_quote_volume': float(kline[10])
                })

            df = pd.DataFrame(df_data)
            df.set_index('timestamp', inplace=True)
            return df[['open', 'high', 'low', 'close', 'volume']]

        except Exception as e:
            logger.warning(f"Failed to get Binance futures data for {symbol}: {e}")
            return None

    def get_perpetuals_data(self, symbol: str, timeframe: str = '1h', limit: int = 500) -> Optional[pd.DataFrame]:
        """Get perpetuals data (same as futures for Binance)"""
        return self.get_futures_data(symbol, timeframe, limit)

    def place_futures_order(self, symbol: str, side: str, quantity: float, price: Optional[float] = None,
                           order_type: str = 'market', leverage: int = 1) -> Dict:
        """Place futures order (paper trading simulation) with per-platform limits"""
        try:
            # Enforce per-platform leverage cap
            leverage = min(leverage, self.max_leverage)

            # Get current price for trade size validation
            current_price = None
            if price:
                current_price = price
            else:
                # Try to get current price
                data = self.get_futures_data(symbol, '1h', 1)
                if data is not None and len(data) > 0:
                    current_price = data['close'].iloc[-1]

            # Calculate trade value and enforce size limits
            if current_price:
                trade_value = quantity * current_price / leverage
                if trade_value > self.max_trade_size:
                    # Scale down quantity to respect max trade size
                    max_quantity = (self.max_trade_size * leverage) / current_price
                    quantity = min(quantity, max_quantity)
                    logger.info(f"📏 Scaled down {symbol} trade to ${self.max_trade_size} limit: {quantity:.4f}")

            # For demo purposes, simulate order placement
            order_id = f"demo_binance_{int(time.time())}_{symbol}"

            return {
                'order_id': order_id,
                'symbol': symbol,
                'side': side,
                'quantity': quantity,
                'price': price or current_price,
                'type': order_type,
                'leverage': leverage,
                'status': 'filled',  # Simulate immediate fill
                'platform': 'binance_futures_testnet',
                'mode': 'paper_trading',
                'max_trade_size': self.max_trade_size,
                'max_leverage': self.max_leverage,
                'paper_capital': self.paper_capital
            }

        except Exception as e:
            logger.warning(f"Error in Binance order placement: {e}")
            return {
                'error': str(e),
                'platform': 'binance_futures_testnet'
            }

    def get_positions(self) -> List[Dict]:
        """Get current positions (simulated for demo)"""
        return []  # No real positions in demo

    def get_account_balance(self) -> Dict:
        """Get account balance (simulated)"""
        return {
            'total_balance': self.paper_capital,  # Use platform-specific paper capital
            'available_balance': self.paper_capital * 0.95,  # 95% available
            'used_margin': self.paper_capital * 0.05,  # 5% used
            'unrealized_pnl': 0.0,
            'platform': 'binance_futures_testnet',
            'mode': 'paper_trading',
            'max_trade_size': self.max_trade_size,
            'max_leverage': self.max_leverage
        }

class BybitFuturesPlatform:
    """Bybit Futures Demo Platform"""

    def __init__(self):
        self.name = "Bybit Futures"
        self.platform_type = "futures_perpetuals"
        self.base_url = "https://api-testnet.bybit.com"
        self.api_key = os.getenv("BYBIT_TESTNET_API_KEY", "")
        self.secret_key = os.getenv("BYBIT_TESTNET_SECRET_KEY", "")
        # Per-platform limits from environment
        self.max_trade_size = float(os.getenv("BYBIT_MAX_TRADE_SIZE", "500"))  # $500 default
        self.max_leverage = int(os.getenv("BYBIT_MAX_LEVERAGE", "100"))  # 100x default
        self.paper_capital = float(os.getenv("BYBIT_PAPER_CAPITAL", "100000"))  # 100k USDT
        self.features = [
            'perpetuals', 'linear', 'inverse',
            'leverage_up_to_100x', 'multiple_timeframes',
            'high_frequency_data', 'paper_trading',
            'per_platform_limits', 'smart_leverage'
        ]

    def is_available(self) -> bool:
        try:
            response = requests.get(f"{self.base_url}/v2/public/time", timeout=5)
            return response.status_code == 200
        except:
            return False

    def get_available_instruments(self) -> List[str]:
        """Get available perpetual instruments"""
        return ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'ADAUSDT', 'DOTUSDT']

    def get_futures_data(self, symbol: str, timeframe: str = '1h', limit: int = 500) -> Optional[pd.DataFrame]:
        """Get futures data from Bybit"""
        try:
            # Map timeframes
            interval_map = {
                '1m': '1', '5m': '5', '15m': '15', '30m': '30',
                '1h': '60', '2h': '120', '4h': '240', '6h': '360',
                '12h': '720', '1d': 'D'
            }
            interval = interval_map.get(timeframe, '60')

            params = {
                'symbol': symbol,
                'interval': interval,
                'limit': min(limit, 200)
            }

            response = requests.get(f"{self.base_url}/v2/public/kline/list", params=params, timeout=10)
            response.raise_for_status()

            data = response.json()

            if data.get('ret_code') != 0 or 'result' not in data:
                return None

            df_data = []
            for kline in data['result']:
                df_data.append({
                    'timestamp': datetime.fromtimestamp(kline['open_time'], tz=timezone.utc),
                    'open': float(kline['open']),
                    'high': float(kline['high']),
                    'low': float(kline['low']),
                    'close': float(kline['close']),
                    'volume': float(kline['volume'])
                })

            df = pd.DataFrame(df_data)
            df.set_index('timestamp', inplace=True)
            return df[['open', 'high', 'low', 'close', 'volume']]

        except Exception as e:
            logger.warning(f"Failed to get Bybit futures data for {symbol}: {e}")
            return None

    def get_perpetuals_data(self, symbol: str, timeframe: str = '1h', limit: int = 500) -> Optional[pd.DataFrame]:
        """Get perpetuals data (same as futures for Bybit)"""
        return self.get_futures_data(symbol, timeframe, limit)

    def place_futures_order(self, symbol: str, side: str, quantity: float, price: Optional[float] = None,
                           order_type: str = 'market', leverage: int = 1) -> Dict:
        """Place futures order (paper trading simulation) with per-platform limits"""
        try:
            # Enforce per-platform leverage cap
            leverage = min(leverage, self.max_leverage)

            # Get current price for trade size validation
            current_price = None
            if price:
                current_price = price
            else:
                # Try to get current price
                data = self.get_futures_data(symbol, '1h', 1)
                if data is not None and len(data) > 0:
                    current_price = data['close'].iloc[-1]

            # Calculate trade value and enforce size limits
            if current_price:
                trade_value = quantity * current_price / leverage
                if trade_value > self.max_trade_size:
                    # Scale down quantity to respect max trade size
                    max_quantity = (self.max_trade_size * leverage) / current_price
                    quantity = min(quantity, max_quantity)
                    logger.info(f"📏 Scaled down {symbol} trade to ${self.max_trade_size} limit: {quantity:.4f}")

            # For demo purposes, simulate order placement
            order_id = f"bybit_demo_{int(time.time())}_{symbol}"

            return {
                'order_id': order_id,
                'symbol': symbol,
                'side': side,
                'quantity': quantity,
                'price': price or current_price,
                'type': order_type,
                'leverage': leverage,
                'status': 'filled',
                'platform': 'bybit_futures_demo',
                'mode': 'paper_trading',
                'max_trade_size': self.max_trade_size,
                'max_leverage': self.max_leverage,
                'paper_capital': self.paper_capital
            }

        except Exception as e:
            logger.warning(f"Error in Bybit order placement: {e}")
            return {
                'error': str(e),
                'platform': 'bybit_futures_demo'
            }

    def get_positions(self) -> List[Dict]:
        """Get current positions (simulated)"""
        return []

    def get_account_balance(self) -> Dict:
        """Get account balance (simulated)"""
        return {
            'total_balance': self.paper_capital,  # Use platform-specific paper capital
            'available_balance': self.paper_capital * 0.95,  # 95% available
            'used_margin': self.paper_capital * 0.05,  # 5% used
            'unrealized_pnl': 0.0,
            'platform': 'bybit_futures_demo',
            'mode': 'paper_trading',
            'max_trade_size': self.max_trade_size,
            'max_leverage': self.max_leverage
        }

class BitMEXFuturesPlatform:
    """BitMEX Futures Testnet Platform"""

    def __init__(self):
        self.name = "BitMEX Futures"
        self.platform_type = "futures_perpetuals"
        self.base_url = "https://testnet.bitmex.com"
        self.api_key = os.getenv("BITMEX_TESTNET_API_KEY", "")
        self.secret_key = os.getenv("BITMEX_TESTNET_SECRET_KEY", "")
        self.features = [
            'futures', 'perpetuals', 'inverse', 'quanto',
            'leverage_up_to_100x', 'professional_tools',
            'high_frequency_data', 'paper_trading'
        ]

    def is_available(self) -> bool:
        try:
            response = requests.get(f"{self.base_url}/api/v1/public/ping", timeout=5)
            return response.status_code == 200
        except:
            return False

    def get_available_instruments(self) -> List[str]:
        """Get available futures instruments"""
        return ['XBTUSD', 'ETHUSD', 'SOLUSD', 'ADAUSD', 'DOTUSD']

    def get_futures_data(self, symbol: str, timeframe: str = '1h', limit: int = 500) -> Optional[pd.DataFrame]:
        """Get futures data from BitMEX"""
        try:
            # Map timeframes
            interval_map = {
                '1m': '1m', '5m': '5m', '1h': '1h', '1d': '1d'
            }
            interval = interval_map.get(timeframe, '1h')

            params = {
                'symbol': symbol,
                'binSize': interval,
                'count': min(limit, 1000),
                'reverse': True
            }

            response = requests.get(f"{self.base_url}/api/v1/trade/bucketed", params=params, timeout=10)
            response.raise_for_status()

            data = response.json()

            if not data:
                return None

            df_data = []
            for bucket in data:
                df_data.append({
                    'timestamp': datetime.fromisoformat(bucket['timestamp'].replace('Z', '+00:00')),
                    'open': float(bucket['open']),
                    'high': float(bucket['high']),
                    'low': float(bucket['low']),
                    'close': float(bucket['close']),
                    'volume': float(bucket['volume'])
                })

            df = pd.DataFrame(df_data)
            df.set_index('timestamp', inplace=True)
            df = df.sort_index()  # BitMEX returns in reverse order
            return df[['open', 'high', 'low', 'close', 'volume']]

        except Exception as e:
            logger.warning(f"Failed to get BitMEX futures data for {symbol}: {e}")
            return None

    def get_perpetuals_data(self, symbol: str, timeframe: str = '1h', limit: int = 500) -> Optional[pd.DataFrame]:
        """Get perpetuals data (same as futures for BitMEX)"""
        return self.get_futures_data(symbol, timeframe, limit)

    def place_futures_order(self, symbol: str, side: str, quantity: float, price: Optional[float] = None,
                           order_type: str = 'market', leverage: int = 1) -> Dict:
        """Place futures order (paper trading simulation)"""
        order_id = f"bitmex_demo_{int(time.time())}_{symbol}"

        return {
            'order_id': order_id,
            'symbol': symbol,
            'side': side,
            'quantity': quantity,
            'price': price,
            'type': order_type,
            'leverage': leverage,
            'status': 'filled',
            'platform': 'bitmex_futures_testnet',
            'mode': 'paper_trading'
        }

    def get_positions(self) -> List[Dict]:
        """Get current positions (simulated)"""
        return []

    def get_account_balance(self) -> Dict:
        """Get account balance (simulated)"""
        return {
            'total_balance': 1000000.0,  # BitMEX uses XBT (BTC) for margin
            'available_balance': 950000.0,
            'used_margin': 50000.0,
            'unrealized_pnl': 0.0,
            'platform': 'bitmex_futures_testnet',
            'mode': 'paper_trading'
        }

class DeribitFuturesPlatform:
    """Deribit Futures & Options Platform"""

    def __init__(self):
        self.name = "Deribit Futures"
        self.platform_type = "futures_options"
        self.base_url = "https://test.deribit.com"
        self.client_id = os.getenv("DERIBIT_TEST_CLIENT_ID", "")
        self.client_secret = os.getenv("DERIBIT_TEST_CLIENT_SECRET", "")
        self.features = [
            'futures', 'options', 'perpetuals',
            'leverage_up_to_100x', 'multiple_timeframes',
            'professional_tools', 'paper_trading'
        ]

    def is_available(self) -> bool:
        try:
            response = requests.get(f"{self.base_url}/api/v2/public/test", timeout=5)
            return response.status_code == 200
        except:
            return False

    def get_available_instruments(self) -> List[str]:
        """Get available futures and options instruments"""
        return ['BTC-PERPETUAL', 'ETH-PERPETUAL', 'SOL-PERPETUAL']

    def get_futures_data(self, symbol: str, timeframe: str = '1h', limit: int = 500) -> Optional[pd.DataFrame]:
        """Get futures data from Deribit"""
        try:
            # Map timeframes
            interval_map = {
                '1m': '1', '5m': '5', '15m': '15', '30m': '30',
                '1h': '60', '2h': '120', '4h': '240', '6h': '360',
                '12h': '720', '1d': '1D'
            }
            resolution = interval_map.get(timeframe, '60')

            params = {
                'instrument_name': symbol,
                'resolution': resolution,
                'amount': min(limit, 5000)
            }

            response = requests.get(f"{self.base_url}/api/v2/public/get_tradingview_chart_data",
                                  params=params, timeout=10)
            response.raise_for_status()

            data = response.json()

            if data.get('result') and 'ticks' in data['result']:
                ticks = data['result']['ticks']
                opens = data['result']['open']
                highs = data['result']['high']
                lows = data['result']['low']
                closes = data['result']['close']
                volumes = data['result']['volume']

                df_data = []
                for i, tick in enumerate(ticks):
                    df_data.append({
                        'timestamp': datetime.fromtimestamp(tick / 1000, tz=timezone.utc),
                        'open': opens[i],
                        'high': highs[i],
                        'low': lows[i],
                        'close': closes[i],
                        'volume': volumes[i] if i < len(volumes) else 0
                    })

                df = pd.DataFrame(df_data)
                df.set_index('timestamp', inplace=True)
                return df[['open', 'high', 'low', 'close', 'volume']]

        except Exception as e:
            logger.warning(f"Failed to get Deribit futures data for {symbol}: {e}")
            return None

    def get_perpetuals_data(self, symbol: str, timeframe: str = '1h', limit: int = 500) -> Optional[pd.DataFrame]:
        """Get perpetuals data (same as futures for Deribit)"""
        return self.get_futures_data(symbol, timeframe, limit)

    def place_futures_order(self, symbol: str, side: str, quantity: float, price: Optional[float] = None,
                           order_type: str = 'market', leverage: int = 1) -> Dict:
        """Place futures order (paper trading simulation)"""
        order_id = f"deribit_demo_{int(time.time())}_{symbol}"

        return {
            'order_id': order_id,
            'symbol': symbol,
            'side': side,
            'quantity': quantity,
            'price': price,
            'type': order_type,
            'leverage': leverage,
            'status': 'filled',
            'platform': 'deribit_futures_test',
            'mode': 'paper_trading'
        }

    def get_positions(self) -> List[Dict]:
        """Get current positions (simulated)"""
        return []

    def get_account_balance(self) -> Dict:
        """Get account balance (simulated)"""
        return {
            'total_balance': 100000.0,
            'available_balance': 95000.0,
            'used_margin': 5000.0,
            'unrealized_pnl': 0.0,
            'platform': 'deribit_futures_test',
            'mode': 'paper_trading'
        }

# Global instance
futures_platform = FuturesTradingPlatform()

def get_futures_data(symbol: str, timeframe: str = '1h', limit: int = 500) -> Optional[pd.DataFrame]:
    """Get futures data from active platform"""
    return futures_platform.get_futures_data(symbol, timeframe, limit)

def get_perpetuals_data(symbol: str, timeframe: str = '1h', limit: int = 500) -> Optional[pd.DataFrame]:
    """Get perpetuals data from active platform"""
    return futures_platform.get_perpetuals_data(symbol, timeframe, limit)

def place_futures_order(symbol: str, side: str, quantity: float, price: Optional[float] = None,
                       order_type: str = 'market', leverage: int = 1) -> Dict:
    """Place futures order"""
    return futures_platform.place_futures_order(symbol, side, quantity, price, order_type, leverage)

def switch_futures_platform(platform_name: str) -> bool:
    """Switch active futures platform"""
    return futures_platform.switch_platform(platform_name)

def get_futures_platforms() -> List[str]:
    """Get available futures platforms"""
    return futures_platform.get_available_platforms()

def get_futures_platform_info(platform_name: str = None) -> Dict:
    """Get platform information"""
    return futures_platform.get_platform_info(platform_name)

if __name__ == "__main__":
    # Test the futures platform
    print("🧪 Testing Futures & Perpetuals Platforms")
    print("=" * 60)

    platforms = get_futures_platforms()
    print(f"📊 Available Platforms: {platforms}")

    if platforms:
        # Test data fetching
        print("\n🧪 Testing BTC Futures Data...")
        data = get_futures_data("BTCUSDT", "1h", 10)
        if data is not None:
            print(f"✅ Got {len(data)} bars")
            print(f"📅 Latest: {data.index[-1].strftime('%Y-%m-%d %H:%M:%S UTC')}")
            print(".2f")
        else:
            print("❌ No data received")

        # Test platform info
        print("\n📋 Platform Information:")
        for platform in platforms:
            info = get_futures_platform_info(platform)
            if info:
                print(f"🏛️  {info['name']}: {info['type']} - {len(info['features'])} features")
