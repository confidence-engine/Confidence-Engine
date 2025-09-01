#!/usr/bin/env python3
"""
Multi-Source Data Provider System
Provides free alternatives to Alpaca for paper trading and testing.
Supports multiple data sources with automatic failover.
"""

import os
import time
import pandas as pd
import requests
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)

class DataSourceProvider:
    """Base class for data source providers"""

    def __init__(self, name: str, priority: int = 1):
        self.name = name
        self.priority = priority
        self.last_request = 0
        self.rate_limit_delay = 1.0  # seconds between requests

    def _rate_limit(self):
        """Enforce rate limiting"""
        elapsed = time.time() - self.last_request
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self.last_request = time.time()

    def get_crypto_bars(self, symbol: str, timeframe: str, limit: int = 200) -> Optional[pd.DataFrame]:
        """Get crypto bars data"""
        raise NotImplementedError

    def get_crypto_news(self, symbol: str, limit: int = 10) -> List[str]:
        """Get crypto news headlines"""
        raise NotImplementedError

    def is_available(self) -> bool:
        """Check if this data source is available"""
        return True

class YahooFinanceProvider(DataSourceProvider):
    """Yahoo Finance data provider using yfinance"""

    def __init__(self):
        super().__init__("Yahoo Finance", priority=1)
        self.rate_limit_delay = 0.5

    def is_available(self) -> bool:
        try:
            import yfinance as yf
            return True
        except ImportError:
            return False

    def get_crypto_bars(self, symbol: str, timeframe: str, limit: int = 200) -> Optional[pd.DataFrame]:
        if not self.is_available():
            return None

        try:
            import yfinance as yf
            self._rate_limit()

            # Convert symbol format (BTC/USD -> BTC-USD)
            yf_symbol = symbol.replace('/', '-')

            # Map timeframes
            interval_map = {
                '1Min': '1m',
                '5Min': '5m',
                '15Min': '15m',
                '1Hour': '1h',
                '1Day': '1d'
            }
            interval = interval_map.get(timeframe, '15m')

            # Calculate period based on limit and interval
            if 'm' in interval:
                minutes = int(interval.replace('m', ''))
                period = f"{max(1, limit * minutes // 1440)}d"  # Convert to days
            elif 'h' in interval:
                hours = int(interval.replace('h', ''))
                period = f"{max(1, limit * hours // 24)}d"
            else:
                period = "30d"

            ticker = yf.Ticker(yf_symbol)
            df = ticker.history(period=period, interval=interval)

            if df.empty:
                return None

            # Convert to our format
            df = df.rename(columns={
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })

            # Ensure UTC timezone
            if df.index.tz is None:
                df.index = df.index.tz_localize('UTC')
            else:
                df.index = df.index.tz_convert('UTC')

            return df[['open', 'high', 'low', 'close', 'volume']].tail(limit)

        except Exception as e:
            logger.warning(f"Yahoo Finance error for {symbol}: {e}")
            return None

    def get_crypto_news(self, symbol: str, limit: int = 10) -> List[str]:
        # Yahoo Finance doesn't provide news headlines via yfinance
        return []

class BinanceProvider(DataSourceProvider):
    """Binance API data provider"""

    def __init__(self):
        super().__init__("Binance", priority=2)
        self.base_url = "https://api.binance.com/api/v3"
        self.rate_limit_delay = 0.1

    def get_crypto_bars(self, symbol: str, timeframe: str, limit: int = 200) -> Optional[pd.DataFrame]:
        try:
            self._rate_limit()

            # Convert symbol format (BTC/USD -> BTCUSDT)
            binance_symbol = symbol.replace('/', '') + 'T' if '/' in symbol else symbol

            # Map timeframes to Binance intervals
            interval_map = {
                '1Min': '1m',
                '5Min': '5m',
                '15Min': '15m',
                '1Hour': '1h',
                '1Day': '1d'
            }
            interval = interval_map.get(timeframe, '15m')

            # Calculate start time
            end_time = int(time.time() * 1000)
            if 'm' in interval:
                minutes = int(interval.replace('m', ''))
                start_time = end_time - (limit * minutes * 60 * 1000)
            elif 'h' in interval:
                hours = int(interval.replace('h', ''))
                start_time = end_time - (limit * hours * 3600 * 1000)
            else:
                start_time = end_time - (limit * 24 * 3600 * 1000)

            url = f"{self.base_url}/klines"
            params = {
                'symbol': binance_symbol,
                'interval': interval,
                'startTime': start_time,
                'endTime': end_time,
                'limit': limit
            }

            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()
            if not data:
                return None

            # Convert to DataFrame
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])

            # Convert types
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            df.set_index('timestamp', inplace=True)
            return df[['open', 'high', 'low', 'close', 'volume']]

        except Exception as e:
            logger.warning(f"Binance error for {symbol}: {e}")
            return None

    def get_crypto_news(self, symbol: str, limit: int = 10) -> List[str]:
        # Binance doesn't provide news headlines
        return []

class CoinGeckoProvider(DataSourceProvider):
    """CoinGecko API data provider"""

    def __init__(self):
        super().__init__("CoinGecko", priority=3)
        self.base_url = "https://api.coingecko.com/api/v3"
        self.rate_limit_delay = 1.2  # CoinGecko has stricter rate limits

    def _get_coin_id(self, symbol: str) -> Optional[str]:
        """Get CoinGecko coin ID from symbol"""
        try:
            # Map common symbols to CoinGecko IDs
            symbol_map = {
                'BTC/USD': 'bitcoin',
                'ETH/USD': 'ethereum',
                'SOL/USD': 'solana',
                'LINK/USD': 'chainlink',
                'LTC/USD': 'litecoin',
                'BCH/USD': 'bitcoin-cash',
                'UNI/USD': 'uniswap',
                'AAVE/USD': 'aave',
                'AVAX/USD': 'avalanche-2',
                'DOT/USD': 'polkadot',
                'MATIC/USD': 'matic-network',
                'MKR/USD': 'maker',
                'COMP/USD': 'compound-governance-token',
                'YFI/USD': 'yearn-finance',
                'CRV/USD': 'curve-dao-token',
                'SNX/USD': 'synthetix',
                'SUSHI/USD': 'sushi',
                'XTZ/USD': 'tezos',
                'GRT/USD': 'the-graph'
            }

            base_symbol = symbol.split('/')[0] if '/' in symbol else symbol
            return symbol_map.get(f"{base_symbol}/USD")
        except:
            return None

    def get_crypto_bars(self, symbol: str, timeframe: str, limit: int = 200) -> Optional[pd.DataFrame]:
        try:
            self._rate_limit()

            coin_id = self._get_coin_id(symbol)
            if not coin_id:
                return None

            # Map timeframes
            days_map = {
                '1Min': 1,
                '5Min': 1,
                '15Min': 1,
                '1Hour': 7,
                '1Day': 30
            }
            days = days_map.get(timeframe, 7)

            url = f"{self.base_url}/coins/{coin_id}/market_chart"
            params = {
                'vs_currency': 'usd',
                'days': days,
                'interval': 'hourly' if timeframe in ['1Hour', '1Min', '5Min', '15Min'] else 'daily'
            }

            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()

            if 'prices' not in data:
                return None

            # Convert to DataFrame
            prices = data['prices']
            volumes = data.get('total_volumes', [])

            df_data = []
            for i, (timestamp, price) in enumerate(prices):
                volume = volumes[i][1] if i < len(volumes) else 0
                df_data.append({
                    'timestamp': datetime.fromtimestamp(timestamp / 1000, tz=timezone.utc),
                    'open': price,
                    'high': price,
                    'low': price,
                    'close': price,
                    'volume': volume
                })

            if not df_data:
                return None

            df = pd.DataFrame(df_data)
            df.set_index('timestamp', inplace=True)

            # Resample to desired timeframe if needed
            if timeframe == '1Min':
                df = df.resample('1min').ffill()
            elif timeframe == '5Min':
                df = df.resample('5min').ffill()
            elif timeframe == '15Min':
                df = df.resample('15min').ffill()

            return df[['open', 'high', 'low', 'close', 'volume']].tail(limit)

        except Exception as e:
            logger.warning(f"CoinGecko error for {symbol}: {e}")
            return None

    def get_crypto_news(self, symbol: str, limit: int = 10) -> List[str]:
        # CoinGecko doesn't provide news headlines
        return []

class AlphaVantageProvider(DataSourceProvider):
    """Alpha Vantage data provider"""

    def __init__(self):
        super().__init__("Alpha Vantage", priority=4)
        self.api_key = os.getenv("ALPHA_VANTAGE_API_KEY", "")
        self.base_url = "https://www.alphavantage.co/query"
        self.rate_limit_delay = 12  # Alpha Vantage has 5 calls/minute free limit

    def is_available(self) -> bool:
        return bool(self.api_key)

    def get_crypto_bars(self, symbol: str, timeframe: str, limit: int = 200) -> Optional[pd.DataFrame]:
        if not self.is_available():
            return None

        try:
            self._rate_limit()

            # Convert symbol format
            base_symbol = symbol.split('/')[0] if '/' in symbol else symbol

            # Map timeframes
            function_map = {
                '1Min': 'CRYPTO_INTRADAY',
                '5Min': 'CRYPTO_INTRADAY',
                '15Min': 'CRYPTO_INTRADAY',
                '1Hour': 'CRYPTO_INTRADAY',
                '1Day': 'DIGITAL_CURRENCY_DAILY'
            }

            interval_map = {
                '1Min': '1min',
                '5Min': '5min',
                '15Min': '15min',
                '1Hour': '60min'
            }

            function = function_map.get(timeframe, 'DIGITAL_CURRENCY_DAILY')
            interval = interval_map.get(timeframe, '15min')

            params = {
                'function': function,
                'symbol': base_symbol,
                'market': 'USD',
                'apikey': self.api_key,
                'outputsize': 'compact'
            }

            if function == 'CRYPTO_INTRADAY':
                params['interval'] = interval

            response = requests.get(self.base_url, params=params, timeout=15)
            response.raise_for_status()

            data = response.json()

            if 'Error Message' in data:
                logger.warning(f"Alpha Vantage error: {data['Error Message']}")
                return None

            # Parse the response
            if function == 'CRYPTO_INTRADAY':
                time_series_key = f"Time Series Crypto ({interval})"
            else:
                time_series_key = "Time Series (Digital Currency Daily)"

            if time_series_key not in data:
                return None

            time_series = data[time_series_key]

            df_data = []
            for timestamp_str, values in time_series.items():
                df_data.append({
                    'timestamp': datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S' if ' ' in timestamp_str else '%Y-%m-%d'),
                    'open': float(values['1. open']),
                    'high': float(values['2. high']),
                    'low': float(values['3. low']),
                    'close': float(values['4. close']),
                    'volume': float(values['5. volume'])
                })

            if not df_data:
                return None

            df = pd.DataFrame(df_data)
            df.set_index('timestamp', inplace=True)

            # Ensure UTC timezone
            if df.index.tz is None:
                df.index = df.index.tz_localize('UTC')

            return df[['open', 'high', 'low', 'close', 'volume']].tail(limit)

        except Exception as e:
            logger.warning(f"Alpha Vantage error for {symbol}: {e}")
            return None

    def get_crypto_news(self, symbol: str, limit: int = 10) -> List[str]:
        # Alpha Vantage doesn't provide news headlines
        return []

class MultiSourceDataProvider:
    """Multi-source data provider with automatic failover"""

    def __init__(self):
        self.providers = []
        self._initialize_providers()

    def _initialize_providers(self):
        """Initialize all available data providers"""
        providers = [
            YahooFinanceProvider(),
            BinanceProvider(),
            CoinGeckoProvider(),
            AlphaVantageProvider()
        ]

        # Sort by priority and availability
        available_providers = [p for p in providers if p.is_available()]
        available_providers.sort(key=lambda x: x.priority)

        self.providers = available_providers
        logger.info(f"Initialized {len(self.providers)} data providers: {[p.name for p in self.providers]}")

    def get_crypto_bars(self, symbol: str, timeframe: str = '15Min', limit: int = 200) -> Optional[pd.DataFrame]:
        """Get crypto bars from the best available provider"""
        for provider in self.providers:
            try:
                logger.debug(f"Trying {provider.name} for {symbol} {timeframe}")
                data = provider.get_crypto_bars(symbol, timeframe, limit)
                if data is not None and not data.empty:
                    logger.debug(f"✅ {provider.name} succeeded for {symbol}")
                    return data
            except Exception as e:
                logger.debug(f"❌ {provider.name} failed for {symbol}: {e}")
                continue

        logger.warning(f"No data provider succeeded for {symbol}")
        return None

    def get_crypto_news(self, symbol: str, limit: int = 10) -> List[str]:
        """Get crypto news from available providers"""
        all_headlines = []

        for provider in self.providers:
            try:
                headlines = provider.get_crypto_news(symbol, limit)
                if headlines:
                    all_headlines.extend(headlines)
            except Exception as e:
                logger.debug(f"News fetch failed for {provider.name}: {e}")
                continue

        return all_headlines[:limit]

    def get_provider_status(self) -> Dict[str, bool]:
        """Get status of all providers"""
        return {provider.name: provider.is_available() for provider in self.providers}

# Global instance
multi_source_provider = MultiSourceDataProvider()

def get_crypto_bars_alternative(symbol: str, timeframe: str = '15Min', limit: int = 200) -> Optional[pd.DataFrame]:
    """Alternative to Alpaca's get_crypto_bars using multiple free sources"""
    return multi_source_provider.get_crypto_bars(symbol, timeframe, limit)

def get_crypto_news_alternative(symbol: str, limit: int = 10) -> List[str]:
    """Alternative to Alpaca's news using multiple free sources"""
    return multi_source_provider.get_crypto_news(symbol, limit)

if __name__ == "__main__":
    # Test the providers
    provider = MultiSourceDataProvider()
    print("Provider Status:", provider.get_provider_status())

    # Test BTC data
    print("\nTesting BTC/USD data...")
    data = provider.get_crypto_bars("BTC/USD", "15Min", 10)
    if data is not None:
        print(f"✅ Got {len(data)} bars")
        print(data.head())
    else:
        print("❌ No data received")
