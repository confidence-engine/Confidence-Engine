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
import random

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

        # Circuit breaker and retry configuration
        self.circuit_breaker_failures = {}
        self.circuit_breaker_timeout = 300  # 5 minutes
        self.max_retries = 3
        self.base_retry_delay = 1.0  # 1 second
        self.max_retry_delay = 30.0  # 30 seconds

    def _retry_with_backoff(self, func, *args, **kwargs):
        """Execute function with exponential backoff retry logic"""
        platform_name = self.active_platform or "unknown"

        # Check circuit breaker
        if self._is_circuit_breaker_open(platform_name):
            logger.warning(f"🚫 Circuit breaker open for {platform_name}, skipping request")
            return None

        last_exception = None
        for attempt in range(self.max_retries + 1):
            try:
                result = func(*args, **kwargs)
                # Success - reset circuit breaker
                if platform_name in self.circuit_breaker_failures:
                    del self.circuit_breaker_failures[platform_name]
                return result

            except Exception as e:
                last_exception = e
                logger.warning(f"Attempt {attempt + 1}/{self.max_retries + 1} failed for {platform_name}: {e}")

                if attempt < self.max_retries:
                    # Calculate delay with exponential backoff and jitter
                    delay = min(self.base_retry_delay * (2 ** attempt), self.max_retry_delay)
                    jitter = random.uniform(0.1, 1.0) * delay * 0.1  # 10% jitter
                    total_delay = delay + jitter

                    logger.info(f"⏳ Retrying in {total_delay:.2f} seconds...")
                    time.sleep(total_delay)

        # All retries failed - trigger circuit breaker
        self._trigger_circuit_breaker(platform_name)
        logger.error(f"❌ All {self.max_retries + 1} attempts failed for {platform_name}: {last_exception}")
        return None

    def _is_circuit_breaker_open(self, platform_name: str) -> bool:
        """Check if circuit breaker is open for a platform"""
        if platform_name not in self.circuit_breaker_failures:
            return False

        failures, last_failure_time = self.circuit_breaker_failures[platform_name]
        if failures >= 5:  # Open circuit after 5 failures
            time_since_failure = time.time() - last_failure_time
            if time_since_failure < self.circuit_breaker_timeout:
                return True
            else:
                # Timeout expired, reset circuit breaker
                del self.circuit_breaker_failures[platform_name]
                return False

        return False

    def _trigger_circuit_breaker(self, platform_name: str):
        """Trigger circuit breaker for a platform"""
        current_time = time.time()
        if platform_name not in self.circuit_breaker_failures:
            self.circuit_breaker_failures[platform_name] = [1, current_time]
        else:
            failures, _ = self.circuit_breaker_failures[platform_name]
            self.circuit_breaker_failures[platform_name] = [failures + 1, current_time]

        logger.warning(f"🔌 Circuit breaker triggered for {platform_name} after multiple failures")

    def _initialize_platforms(self):
        """Initialize only Bybit (primary) and Binance (fallback) platforms - both on testnet"""
        platforms = [
            BybitFuturesPlatform(),  # Primary: Testnet Bybit account
            BinanceFuturesPlatform(),  # Fallback: Binance testnet
        ]

        for platform in platforms:
            if platform.is_available():
                self.platforms[platform.name] = platform
                logger.info(f"✅ {platform.name} initialized")

        if self.platforms:
            # Set Bybit as default active platform (testnet trading)
            if "Bybit Futures" in self.platforms:
                self.active_platform = "Bybit Futures"
                logger.info(f"🎯 Active platform: {self.active_platform} (TESTNET TRADING)")
            else:
                # Fallback to first available platform
                self.active_platform = list(self.platforms.keys())[0]
                logger.info(f"🎯 Active platform: {self.active_platform} (fallback)")

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
        """Place futures order with dynamic margin/leverage calculation and retry logic"""
        if not self.active_platform:
            return {'error': 'No active platform'}

        # Calculate dynamic margin and leverage based on risk-reward
        dynamic_params = self._calculate_dynamic_margin_leverage(symbol, side, quantity, price, leverage)
        quantity = dynamic_params['quantity']
        leverage = dynamic_params['leverage']
        margin_used = dynamic_params['margin_used']

        logger.info(f"📊 Dynamic calculation for {symbol}: Margin=${margin_used:.2f}, Leverage={leverage}x, Qty={quantity:.4f}")

        def _place_order_attempt():
            platform = self.platforms[self.active_platform]
            return platform.place_futures_order(symbol, side, quantity, price, order_type, leverage)

        result = self._retry_with_backoff(_place_order_attempt)

        if result is None:
            return {
                'error': f'All retry attempts failed for {self.active_platform}',
                'platform': self.active_platform,
                'circuit_breaker': 'open' if self._is_circuit_breaker_open(self.active_platform) else 'closed'
            }

        return result

    def _calculate_dynamic_margin_leverage(self, symbol: str, side: str, quantity: float,
                                         price: Optional[float] = None, requested_leverage: int = 1) -> Dict:
        """Calculate dynamic margin and leverage based on risk-reward ratio"""
        try:
            # Get platform-specific caps
            platform = self.platforms[self.active_platform]
            MAX_MARGIN_PER_TRADE = getattr(platform, 'max_margin_per_trade', 1000.0)  # $1000 default
            MAX_LEVERAGE_PER_TRADE = getattr(platform, 'max_leverage_per_trade', 100)  # 100x default

            # Get current price if not provided
            if price is None:
                data = self.get_futures_data(symbol, '1h', 1)
                if data is not None and len(data) > 0:
                    price = data['close'].iloc[-1]
                else:
                    price = 50000  # Fallback BTC price

            # Calculate base position value
            position_value = quantity * price

            # Apply leverage cap
            leverage = min(requested_leverage, MAX_LEVERAGE_PER_TRADE)

            # Calculate required margin
            margin_required = position_value / leverage

            # Apply margin cap - if required margin exceeds cap, reduce position size
            if margin_required > MAX_MARGIN_PER_TRADE:
                # Scale down the position to fit within margin cap
                max_position_value = MAX_MARGIN_PER_TRADE * leverage
                quantity = max_position_value / price
                margin_required = MAX_MARGIN_PER_TRADE
                logger.info(f"📏 Scaled down {symbol} position to fit ${MAX_MARGIN_PER_TRADE} margin cap")

            # Ensure minimum position size (0.001 BTC or equivalent)
            min_quantity = 0.001 if 'BTC' in symbol else 0.1
            if quantity < min_quantity:
                quantity = min_quantity
                margin_required = (quantity * price) / leverage
                logger.info(f"📏 Adjusted {symbol} to minimum quantity: {quantity}")

            # Apply precision requirements based on symbol
            quantity = self._apply_quantity_precision(symbol, quantity)
            logger.info(f"📐 Applied precision to {symbol}: {quantity}")

            return {
                'quantity': quantity,
                'leverage': leverage,
                'margin_used': margin_required,
                'position_value': quantity * price,
                'price': price
            }

        except Exception as e:
            logger.warning(f"Error in dynamic calculation: {e}")
            # Return safe defaults
            return {
                'quantity': min(quantity, 0.001),
                'leverage': min(requested_leverage, 10),
                'margin_used': 100.0,
                'position_value': quantity * (price or 50000),
                'price': price or 50000
            }

    def _apply_quantity_precision(self, symbol: str, quantity: float) -> float:
        """Apply appropriate quantity precision based on symbol and platform requirements"""
        try:
            # Get precision requirements from active platform
            platform = self.platforms.get(self.active_platform)
            if not platform:
                return quantity

            # Different platforms have different precision requirements
            if "Binance" in self.active_platform:
                return self._apply_binance_precision(symbol, quantity)
            elif "Bybit" in self.active_platform:
                return self._apply_bybit_precision(symbol, quantity)
            else:
                # Default precision handling
                return self._apply_default_precision(symbol, quantity)

        except Exception as e:
            logger.warning(f"Error applying quantity precision: {e}")
            return quantity

    def _apply_binance_precision(self, symbol: str, quantity: float) -> float:
        """Apply Binance-specific quantity precision requirements"""
        # Binance has different precision requirements for different symbols
        symbol_upper = symbol.upper()
        
        if 'BTC' in symbol_upper and 'USDT' in symbol_upper:
            # BTC pairs: 0.001 precision (3 decimal places)
            precision = 3
        elif 'ETH' in symbol_upper and 'USDT' in symbol_upper:
            # ETH pairs: 0.01 precision (2 decimal places)
            precision = 2
        elif symbol_upper in ['SOLUSDT', 'AVAXUSDT', 'ADAUSDT', 'DOTUSDT', 'LINKUSDT', 'UNIUSDT', 'AAVEUSDT', 'ATOMUSDT', 'ALGOUSDT']:
            # Altcoin pairs: higher precision needed (0.1 to 1.0 range typically)
            precision = 1  # 0.1 precision for most altcoins
        elif 'USDT' in symbol_upper:
            # Other USDT pairs: 0.1 precision (1 decimal place)
            precision = 1
        else:
            # Other pairs: 0.1 precision (1 decimal place)
            precision = 1

        # Round to appropriate precision
        rounded_quantity = round(quantity, precision)

        # Ensure minimum quantity after rounding
        min_quantity = 10 ** (-precision)  # 0.001 for BTC, 0.01 for ETH, 0.1 for others
        if rounded_quantity < min_quantity:
            rounded_quantity = min_quantity

        # For very small quantities that might still cause precision errors, round up to minimum
        if rounded_quantity < min_quantity * 10:  # If less than 10x minimum
            rounded_quantity = min_quantity * 10  # Use a safer minimum

        return rounded_quantity

    def _apply_bybit_precision(self, symbol: str, quantity: float) -> float:
        """Apply Bybit-specific quantity precision requirements"""
        # Bybit has different precision requirements
        if 'BTC' in symbol.upper():
            # BTC pairs: 0.001 precision (3 decimal places)
            precision = 3
        elif 'ETH' in symbol.upper():
            # ETH pairs: 0.01 precision (2 decimal places)
            precision = 2
        else:
            # Other pairs: 0.001 precision (3 decimal places)
            precision = 3

        # Round to appropriate precision
        rounded_quantity = round(quantity, precision)

        # Ensure minimum quantity after rounding
        min_quantity = 10 ** (-precision)
        if rounded_quantity < min_quantity:
            rounded_quantity = min_quantity

        return rounded_quantity

    def _apply_default_precision(self, symbol: str, quantity: float) -> float:
        """Apply default quantity precision requirements"""
        # Default: 0.001 precision for BTC, 0.1 for others
        if 'BTC' in symbol.upper():
            precision = 3
            min_quantity = 0.001
        else:
            precision = 1
            min_quantity = 0.1

        # Round to appropriate precision
        rounded_quantity = round(quantity, precision)

        # Ensure minimum quantity after rounding
        if rounded_quantity < min_quantity:
            rounded_quantity = min_quantity

        return rounded_quantity

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
        # Dynamic margin and leverage caps as specified
        self.max_margin_per_trade = float(os.getenv("BINANCE_MAX_MARGIN_PER_TRADE", "1000.0"))  # $1000 hard cap per trade
        self.max_leverage_per_trade = int(os.getenv("BINANCE_MAX_LEVERAGE", "100"))  # 100x hard cap per trade
        self.max_trade_size = float(os.getenv("BINANCE_MAX_TRADE_SIZE", "10000"))  # Platform limit
        self.max_leverage = int(os.getenv("BINANCE_MAX_LEVERAGE", "100"))  # Platform limit
        self.paper_capital = float(os.getenv("BINANCE_PAPER_CAPITAL", "15000"))  # Testnet capital
        self.features = [
            'futures', 'perpetuals', 'linear', 'inverse',
            'leverage_up_to_125x', 'multiple_timeframes',
            'high_frequency_data', 'paper_trading',
            'dynamic_margin_leverage', 'smart_risk_management'
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
        """Place REAL futures order on Binance testnet with dynamic margin/leverage"""
        try:
            # Get current price if not provided
            current_price = price
            if current_price is None:
                data = self.get_futures_data(symbol, '1h', 1)
                if data is not None and len(data) > 0:
                    current_price = data['close'].iloc[-1]
                else:
                    return {'error': f'Cannot get current price for {symbol}', 'platform': 'binance_futures_testnet'}

            # Validate API credentials
            if not self.api_key or not self.secret_key:
                return {'error': 'Missing Binance API credentials', 'platform': 'binance_futures_testnet'}

            # Set leverage first
            leverage_params = {
                'symbol': symbol,
                'leverage': leverage,
                'timestamp': int(time.time() * 1000),
                'recvWindow': 5000
            }
            
            leverage_query = urlencode(leverage_params)
            leverage_signature = self._generate_signature(leverage_query)
            leverage_params['signature'] = leverage_signature
            
            headers = {'X-MBX-APIKEY': self.api_key}
            
            # Set leverage
            leverage_response = requests.post(
                f'{self.base_url}/fapi/v1/leverage',
                headers=headers,
                data=leverage_params,
                timeout=10
            )
            
            if leverage_response.status_code != 200:
                logger.warning(f"Leverage setting failed: {leverage_response.text}")

            # Prepare order parameters with proper precision
            # Different symbols have different quantity precision requirements
            if 'BTC' in symbol or 'ETH' in symbol:
                quantity_precision = 3  # 0.001 for BTC/ETH
            elif 'USDT' in symbol:
                quantity_precision = 1  # 0.1 for most USDT pairs
            else:
                quantity_precision = 0  # Whole numbers for others
            
            # Round quantity to appropriate precision
            rounded_quantity = round(quantity, quantity_precision)
            
            order_params = {
                'symbol': symbol,
                'side': side.upper(),
                'type': order_type.upper(),
                'quantity': f'{rounded_quantity:.{quantity_precision}f}',
                'timestamp': int(time.time() * 1000),
                'recvWindow': 5000
            }

            # Add price for limit orders
            if order_type.lower() == 'limit' and price:
                order_params['price'] = f'{price:.2f}'
                order_params['timeInForce'] = 'GTC'

            # Create signature
            order_query = urlencode(order_params)
            order_signature = self._generate_signature(order_query)
            order_params['signature'] = order_signature

            # Place the order
            order_response = requests.post(
                f'{self.base_url}/fapi/v1/order',
                headers=headers,
                data=order_params,
                timeout=10
            )

            if order_response.status_code == 200:
                order_data = order_response.json()
                
                # Calculate margin used
                margin_used = (quantity * current_price) / leverage
                
                return {
                    'order_id': str(order_data.get('orderId', '')),
                    'client_order_id': order_data.get('clientOrderId', ''),
                    'symbol': symbol,
                    'side': side,
                    'quantity': quantity,
                    'price': current_price,
                    'type': order_type,
                    'leverage': leverage,
                    'status': order_data.get('status', 'NEW'),
                    'platform': 'binance_futures_testnet',
                    'mode': 'testnet_trading',
                    'max_trade_size': self.max_trade_size,
                    'max_leverage': self.max_leverage,
                    'margin_used': margin_used,
                    'order_time': order_data.get('updateTime', int(time.time() * 1000))
                }
            else:
                error_data = order_response.json() if order_response.content else {}
                error_msg = error_data.get('msg', f'HTTP {order_response.status_code}')
                return {
                    'error': f'Binance order failed: {error_msg}',
                    'platform': 'binance_futures_testnet',
                    'response_code': order_response.status_code,
                    'response_text': order_response.text[:200]
                }

        except Exception as e:
            logger.error(f"Error placing Binance testnet order: {e}")
            return {
                'error': f'Exception: {str(e)}',
                'platform': 'binance_futures_testnet'
            }

        except Exception as e:
            logger.warning(f"Error in Binance order placement: {e}")
            return {
                'error': str(e),
                'platform': 'binance_futures_testnet'
            }

    def get_positions(self) -> List[Dict]:
        """Get current positions from Binance testnet API"""
        try:
            if not self.api_key or not self.secret_key:
                return []  # No API credentials, return empty

            # Real API call to get position information
            import requests
            import time
            from urllib.parse import urlencode
            
            params = {
                'timestamp': int(time.time() * 1000),
                'recvWindow': 5000
            }
            
            query_string = urlencode(params)
            signature = self._generate_signature(query_string)
            params['signature'] = signature
            
            headers = {'X-MBX-APIKEY': self.api_key}
            
            response = requests.get(
                f'{self.base_url}/fapi/v2/positionRisk',
                headers=headers,
                params=params,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                positions = []
                
                for pos in data:
                    position_amt = float(pos.get('positionAmt', 0))
                    
                    # Only include non-zero positions
                    if position_amt != 0:
                        positions.append({
                            'symbol': pos.get('symbol', ''),
                            'side': 'long' if position_amt > 0 else 'short',
                            'quantity': abs(position_amt),
                            'entry_price': float(pos.get('entryPrice', 0)),
                            'mark_price': float(pos.get('markPrice', 0)),
                            'unrealized_pnl': float(pos.get('unrealizedProfit', 0)),
                            'leverage': int(pos.get('leverage', 1)),
                            'liquidation_price': float(pos.get('liquidationPrice', 0)),
                            'platform': 'binance_futures_testnet',
                            'mode': 'real_testnet_api'
                        })
                
                return positions
            else:
                logger.warning(f"Failed to get positions: HTTP {response.status_code} - {response.text}")
                return []
                
        except Exception as e:
            logger.warning(f"Error fetching positions: {e}")
            return []

    def get_account_balance(self) -> Dict:
        """Get REAL account balance from Binance testnet"""
        try:
            if not self.api_key or not self.secret_key:
                # Fallback to paper capital if no API credentials
                return {
                    'total_balance': self.paper_capital,
                    'available_balance': self.paper_capital * 0.95,
                    'used_margin': self.paper_capital * 0.05,
                    'unrealized_pnl': 0.0,
                    'platform': 'binance_futures_testnet',
                    'mode': 'paper_trading',
                    'max_trade_size': self.max_trade_size,
                    'max_leverage': self.max_leverage
                }

            # Real API call to get account information
            import requests
            import time
            from urllib.parse import urlencode
            
            params = {
                'timestamp': int(time.time() * 1000),
                'recvWindow': 5000
            }
            
            query_string = urlencode(params)
            signature = self._generate_signature(query_string)
            params['signature'] = signature
            
            headers = {'X-MBX-APIKEY': self.api_key}
            
            response = requests.get(
                f'{self.base_url}/fapi/v2/account',
                headers=headers,
                params=params,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                total_balance = float(data.get('totalWalletBalance', 0))
                available_balance = float(data.get('availableBalance', 0))
                used_margin = float(data.get('totalInitialMargin', 0))
                unrealized_pnl = float(data.get('totalUnrealizedProfit', 0))
                
                return {
                    'total_balance': total_balance,
                    'available_balance': available_balance,
                    'used_margin': used_margin,
                    'unrealized_pnl': unrealized_pnl,
                    'platform': 'binance_futures_testnet',
                    'mode': 'real_testnet_api',
                    'max_trade_size': self.max_trade_size,
                    'max_leverage': self.max_leverage
                }
            else:
                logger.warning(f"Failed to get account balance: {response.text}")
                # Fallback to paper capital
                return {
                    'total_balance': self.paper_capital,
                    'available_balance': self.paper_capital * 0.95,
                    'used_margin': self.paper_capital * 0.05,
                    'unrealized_pnl': 0.0,
                    'platform': 'binance_futures_testnet',
                    'mode': 'fallback_paper',
                    'max_trade_size': self.max_trade_size,
                    'max_leverage': self.max_leverage
                }
                
        except Exception as e:
            logger.warning(f"Error fetching account balance: {e}")
            # Fallback to paper capital
            return {
                'total_balance': self.paper_capital,
                'available_balance': self.paper_capital * 0.95,
                'used_margin': self.paper_capital * 0.05,
                'unrealized_pnl': 0.0,
                'platform': 'binance_futures_testnet',
                'mode': 'error_fallback',
                'max_trade_size': self.max_trade_size,
                'max_leverage': self.max_leverage
            }

class BybitFuturesPlatform:
    """Bybit Futures Testnet Platform"""

    def __init__(self):
        self.name = "Bybit Futures"
        self.platform_type = "futures_perpetuals"
        self.base_url = "https://api-testnet.bybit.com"  # TESTNET API endpoint
        self.api_key = os.getenv("BYBIT_TESTNET_API_KEY", "")  # Use testnet API key
        self.secret_key = os.getenv("BYBIT_TESTNET_SECRET_KEY", "")  # Use testnet secret key
        # Dynamic margin and leverage caps as specified
        self.max_margin_per_trade = float(os.getenv("BYBIT_MAX_MARGIN_PER_TRADE", "1000.0"))  # $1000 hard cap per trade
        self.max_leverage_per_trade = int(os.getenv("BYBIT_MAX_LEVERAGE", "100"))  # 100x hard cap per trade
        self.max_trade_size = float(os.getenv("BYBIT_MAX_TRADE_SIZE", "10000"))  # Platform limit
        self.max_leverage = int(os.getenv("BYBIT_MAX_LEVERAGE", "100"))  # Platform limit
        self.paper_capital = float(os.getenv("BYBIT_PAPER_CAPITAL", "100000.00"))  # Testnet starting balance
        self.features = [
            'futures', 'perpetuals', 'linear', 'inverse',
            'leverage_up_to_100x', 'multiple_timeframes',
            'high_frequency_data', 'testnet_trading',
            'dynamic_margin_leverage', 'smart_risk_management'
        ]

    def is_available(self) -> bool:
        """Check if Bybit is available (with API key fallback)"""
        # First check if API keys are configured
        if self.api_key and self.secret_key:
            return True  # If we have API keys, consider it available

        # Fallback to connectivity check
        try:
            response = requests.get(f"{self.base_url}/v2/public/time", timeout=5)
            return response.status_code == 200
        except:
            return False  # Only return False if both API keys missing AND connectivity fails

    def _make_request(self, method: str, endpoint: str, params: Dict = None) -> Optional[Dict]:
        """Make authenticated request to Bybit"""
        try:
            url = f"{self.base_url}{endpoint}"

            if params is None:
                params = {}

            # Add timestamp for authentication
            timestamp = str(int(time.time() * 1000))
            params['timestamp'] = timestamp
            params['api_key'] = self.api_key

            # Create signature
            param_str = '&'.join([f"{k}={v}" for k, v in sorted(params.items())])
            signature = self._generate_signature(param_str)
            params['sign'] = signature

            headers = {
                'Content-Type': 'application/json'
            }

            if method.upper() == 'POST':
                response = requests.post(url, json=params, headers=headers, timeout=10)
            else:
                response = requests.get(url, params=params, headers=headers, timeout=10)

            response.raise_for_status()
            return response.json()

        except Exception as e:
            logger.warning(f"Bybit request failed: {e}")
            return None

    def _generate_signature(self, param_str: str) -> str:
        """Generate HMAC SHA256 signature for Bybit"""
        if not self.secret_key:
            return ""
        return hmac.new(
            self.secret_key.encode('utf-8'),
            param_str.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()

    def get_futures_data(self, symbol: str, timeframe: str = '1h', limit: int = 500) -> Optional[pd.DataFrame]:
        """Get futures data from Bybit using V5 API"""
        try:
            # Map timeframes
            interval_map = {
                '1m': '1', '5m': '5', '15m': '15', '30m': '30',
                '1h': '60', '2h': '120', '4h': '240', '6h': '360',
                '12h': '720', '1d': 'D'
            }
            interval = interval_map.get(timeframe, '60')

            params = {
                'category': 'linear',
                'symbol': symbol,
                'interval': interval,
                'limit': min(limit, 200)
            }

            # Use V5 API endpoint
            response = requests.get(f"{self.base_url}/v5/market/kline", params=params, timeout=10)
            response.raise_for_status()

            data = response.json()

            if data.get('retCode') == 0 and 'result' in data:
                df_data = []
                for kline in data['result']['list']:
                    df_data.append({
                        'timestamp': datetime.fromtimestamp(int(kline[0]) / 1000, tz=timezone.utc),
                        'open': float(kline[1]),
                        'high': float(kline[2]),
                        'low': float(kline[3]),
                        'close': float(kline[4]),
                        'volume': float(kline[5])
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
        """Place futures order (TESTNET TRADING - SAFE ORDERS) with dynamic margin/leverage"""
        try:
            # Dynamic calculation is handled by parent class
            # Just validate and place the order with calculated parameters

            # Get current price for validation if not provided
            current_price = price
            if current_price is None:
                data = self.get_futures_data(symbol, '1h', 1)
                if data is not None and len(data) > 0:
                    current_price = data['close'].iloc[-1]

            # Validate position size against platform limits
            if current_price:
                position_value = quantity * current_price
                max_allowed_value = self.max_trade_size

                if position_value > max_allowed_value:
                    # Scale down quantity to respect platform limits
                    quantity = max_allowed_value / current_price
                    logger.info(f"📏 Bybit: Scaled down {symbol} to platform limit ${self.max_trade_size}")

            # TESTNET ORDER PLACEMENT - Use Bybit Testnet API
            if self.api_key and self.secret_key:
                return self._place_testnet_order(symbol, side, quantity, price, order_type, leverage)
            else:
                logger.error("❌ No Bybit testnet API keys configured")
                return {
                    'error': 'No testnet API keys configured',
                    'platform': 'bybit_futures_testnet'
                }

        except Exception as e:
            logger.warning(f"Error in Bybit testnet order placement: {e}")
            return {
                'error': str(e),
                'platform': 'bybit_futures_testnet'
            }

    def _place_testnet_order(self, symbol: str, side: str, quantity: float, price: Optional[float] = None,
                          order_type: str = 'market', leverage: int = 1) -> Dict:
        """Place a testnet order on Bybit using correct API endpoints"""
        try:
            # Use Bybit V5 API for futures testnet
            base_url = "https://api-testnet.bybit.com"
            endpoint = "/v5/order/create"

            # Prepare order parameters
            params = {
                'category': 'linear',  # For USDT perpetuals
                'symbol': symbol,
                'side': side.upper(),
                'orderType': order_type.upper(),
                'qty': str(quantity),
                'timeInForce': 'GTC',
                'reduceOnly': False,
                'closeOnTrigger': False
            }

            if price:
                params['price'] = str(price)

            # Add timestamp for authentication
            timestamp = str(int(time.time() * 1000))
            params['timestamp'] = timestamp
            params['api_key'] = self.api_key

            # Create signature
            param_str = '&'.join([f"{k}={v}" for k, v in sorted(params.items())])
            signature = hmac.new(
                self.secret_key.encode('utf-8'),
                param_str.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()
            params['sign'] = signature

            headers = {
                'Content-Type': 'application/json'
            }

            # Make the API request
            response = requests.post(f"{base_url}{endpoint}", json=params, headers=headers, timeout=10)

            if response.status_code == 200:
                data = response.json()
                if data.get('retCode') == 0:
                    order_data = data.get('result', {})
                    return {
                        'order_id': order_data.get('orderId', ''),
                        'symbol': symbol,
                        'side': side,
                        'quantity': quantity,
                        'price': price or order_data.get('price', 0),
                        'type': order_type,
                        'leverage': leverage,
                        'status': 'placed',
                        'platform': 'bybit_futures_testnet',
                        'mode': 'testnet_trading',
                        'max_trade_size': self.max_trade_size,
                        'max_leverage': self.max_leverage,
                        'paper_capital': self.paper_capital
                    }
                else:
                    error_msg = data.get('retMsg', 'Unknown error')
                    return {
                        'error': f"Bybit testnet API error: {error_msg}",
                        'platform': 'bybit_futures_testnet'
                    }
            else:
                return {
                    'error': f"HTTP {response.status_code}: {response.text[:200]}",
                    'platform': 'bybit_futures_testnet'
                }

        except Exception as e:
            logger.error(f"Failed to place testnet Bybit order: {e}")
            return {
                'error': str(e),
                'platform': 'bybit_futures_testnet'
            }

    def get_positions(self) -> List[Dict]:
        """Get current positions from Bybit using V5 API"""
        try:
            # Try to get real positions from Bybit TESTNET API
            if self.api_key and self.secret_key:
                base_url = "https://api-testnet.bybit.com"  # Use testnet endpoint
                endpoint = "/v5/position/list"

                # Prepare parameters
                params = {
                    'category': 'linear',
                    'timestamp': str(int(time.time() * 1000)),
                    'api_key': self.api_key
                }

                # Create signature
                param_str = '&'.join([f"{k}={v}" for k, v in sorted(params.items())])
                signature = hmac.new(
                    self.secret_key.encode('utf-8'),
                    param_str.encode('utf-8'),
                    hashlib.sha256
                ).hexdigest()
                params['sign'] = signature

                headers = {
                    'Content-Type': 'application/json'
                }

                # Make the API request
                response = requests.get(f"{base_url}{endpoint}", params=params, headers=headers, timeout=10)

                if response.status_code == 200:
                    data = response.json()
                    if data.get('retCode') == 0:
                        positions = []
                        result = data.get('result', {}).get('list', [])
                        for pos in result:
                            size = float(pos.get('size', 0))
                            if size != 0:  # Only include open positions
                                positions.append({
                                    'symbol': pos.get('symbol', ''),
                                    'side': 'long' if size > 0 else 'short',
                                    'quantity': abs(size),
                                    'entry_price': float(pos.get('avgPrice', 0)),
                                    'mark_price': float(pos.get('markPrice', 0)),
                                    'unrealized_pnl': float(pos.get('unrealisedPnl', 0)),
                                    'leverage': int(pos.get('leverage', 1)),
                                    'platform': 'bybit_futures_testnet'
                                })
                        return positions
                    else:
                        error_msg = data.get('retMsg', 'Unknown error')
                        logger.warning(f"Bybit testnet API error getting positions: {error_msg}")

            # Return empty list if no API or no positions
            return []

        except Exception as e:
            logger.warning(f"Failed to get Bybit testnet positions: {e}")
            return []

    def get_account_balance(self) -> Dict:
        """Get account balance from Bybit testnet using V5 API"""
        try:
            # Try to get testnet balance from Bybit API
            if self.api_key and self.secret_key:
                base_url = "https://api-testnet.bybit.com"
                endpoint = "/v5/account/wallet-balance"

                # Prepare parameters
                params = {
                    'accountType': 'UNIFIED',
                    'timestamp': str(int(time.time() * 1000)),
                    'api_key': self.api_key
                }

                # Create signature
                param_str = '&'.join([f"{k}={v}" for k, v in sorted(params.items())])
                signature = hmac.new(
                    self.secret_key.encode('utf-8'),
                    param_str.encode('utf-8'),
                    hashlib.sha256
                ).hexdigest()
                params['sign'] = signature

                headers = {
                    'Content-Type': 'application/json'
                }

                # Make the API request
                response = requests.get(f"{base_url}{endpoint}", params=params, headers=headers, timeout=10)

                if response.status_code == 200:
                    data = response.json()
                    if data.get('retCode') == 0:
                        balance_data = data.get('result', {}).get('list', [{}])[0]
                        total_balance = float(balance_data.get('totalEquity', '0'))
                        available_balance = float(balance_data.get('totalAvailableBalance', '0'))
                        used_margin = float(balance_data.get('totalMarginUsed', '0'))
                        unrealized_pnl = float(balance_data.get('totalPerpetualUnrealisedPnl', '0'))

                        return {
                            'total_balance': total_balance,
                            'available_balance': available_balance,
                            'used_margin': used_margin,
                            'unrealized_pnl': unrealized_pnl,
                            'platform': 'bybit_futures_testnet',
                            'mode': 'testnet_trading',
                            'max_trade_size': self.max_trade_size,
                            'max_leverage': self.max_leverage,
                            'paper_capital': self.paper_capital
                        }
                    else:
                        error_msg = data.get('retMsg', 'Unknown error')
                        logger.warning(f"Bybit testnet API error getting balance: {error_msg}")
                        # Fall back to paper trading data
                        return self._get_paper_balance()
                else:
                    logger.warning(f"HTTP {response.status_code} getting Bybit testnet balance: {response.text[:200]}")
                    # Fall back to paper trading data
                    return self._get_paper_balance()
            else:
                # No API keys, return paper trading data
                return self._get_paper_balance()

        except Exception as e:
            logger.warning(f"Failed to get Bybit testnet account balance: {e}")
            # Fall back to paper trading data
            return self._get_paper_balance()

    def get_available_instruments(self) -> List[str]:
        """Get available futures instruments"""
        return ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'ADAUSDT', 'DOTUSDT', 'LINKUSDT', 'UNIUSDT']

    def _get_paper_balance(self) -> Dict:
        """Get paper trading balance as fallback"""
        return {
            'total_balance': self.paper_capital,
            'available_balance': self.paper_capital * 0.95,  # 95% available
            'used_margin': self.paper_capital * 0.05,  # 5% used
            'unrealized_pnl': 0.0,
            'platform': 'bybit_futures_testnet',
            'mode': 'testnet_trading',
            'max_trade_size': self.max_trade_size,
            'max_leverage': self.max_leverage,
            'paper_capital': self.paper_capital,
            'note': 'Using paper balance - testnet API unavailable'
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
    # Test the futures platform (Bybit primary, Binance fallback - both testnet)
    print("🧪 Testing Futures & Perpetuals Platforms")
    print("🎯 Primary: Bybit Futures (TESTNET TRADING)")
    print("🔄 Fallback: Binance Futures (Testnet)")
    print("=" * 60)

    platforms = get_futures_platforms()
    print(f"📊 Available Platforms: {platforms}")

    if platforms:
        # Test active platform (should be Bybit)
        active_info = get_futures_platform_info()
        if active_info:
            print(f"🎯 Active Platform: {active_info['name']}")
            print(f"📊 Type: {active_info['type']}")
            print(f"⚙️  Mode: {'TESTNET TRADING' if 'bybit' in active_info['name'].lower() else 'PAPER TRADING'}")

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
                mode = "TESTNET" if "bybit" in platform.lower() else "PAPER"
                print(f"🏛️  {info['name']}: {info['type']} - {mode} - {len(info['features'])} features")
