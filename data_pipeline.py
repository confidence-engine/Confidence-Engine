#!/usr/bin/env python3
"""
Unified Data Pipeline System
Standardizes data from all providers to consistent OHLCV format
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Any, List, Union
from datetime import datetime, timezone
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class DataProvider(Enum):
    ALPACA = "alpaca"
    BINANCE = "binance"
    YAHOO = "yahoo"
    COINGECKO = "coingecko"
    ALPHAVANTAGE = "alphavantage"
    MOCK = "mock"

@dataclass
class StandardizedBar:
    """Standardized OHLCV bar format"""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    symbol: str
    provider: str

class DataStandardizer:
    """
    Converts data from any provider to standard OHLCV format
    Eliminates data pipeline inconsistencies
    """
    
    def __init__(self):
        # Column mappings for each provider
        self.column_maps = {
            DataProvider.ALPACA: {
                'timestamp': 'timestamp',
                'open': 'open',
                'high': 'high', 
                'low': 'low',
                'close': 'close',
                'volume': 'volume'
            },
            DataProvider.BINANCE: {
                'timestamp': 'timestamp',
                'open': 'open',
                'high': 'high',
                'low': 'low', 
                'close': 'close',
                'volume': 'volume'
            },
            DataProvider.YAHOO: {
                'timestamp': 'Datetime',
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            },
            DataProvider.COINGECKO: {
                'timestamp': 'timestamp',
                'open': 'o',
                'high': 'h',
                'low': 'l',
                'close': 'c',
                'volume': 'v'
            },
            DataProvider.ALPHAVANTAGE: {
                'timestamp': 'timestamp',
                'open': '1. open',
                'high': '2. high',
                'low': '3. low',
                'close': '4. close',
                'volume': '5. volume'
            },
            DataProvider.MOCK: {
                'timestamp': 'timestamp',
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'volume': 'volume'
            }
        }
        
        # Expected data types after standardization
        self.standard_dtypes = {
            'timestamp': 'datetime64[ns]',
            'open': 'float64',
            'high': 'float64',
            'low': 'float64',
            'close': 'float64',
            'volume': 'float64'
        }
    
    def standardize_data(self, raw_data: Union[pd.DataFrame, List[Dict], Dict], 
                        provider: DataProvider, symbol: str) -> pd.DataFrame:
        """
        Convert any provider data to standard OHLCV format
        
        Args:
            raw_data: Raw data from provider (DataFrame, list of dicts, or dict)
            provider: Data provider enum
            symbol: Trading symbol
            
        Returns:
            Standardized DataFrame with OHLCV data
        """
        try:
            # Convert input to DataFrame if needed
            if isinstance(raw_data, list):
                df = pd.DataFrame(raw_data)
            elif isinstance(raw_data, dict):
                df = pd.DataFrame([raw_data])
            elif isinstance(raw_data, pd.DataFrame):
                df = raw_data.copy()
            else:
                raise ValueError(f"Unsupported data type: {type(raw_data)}")
            
            if df.empty:
                logger.warning(f"Empty data received from {provider.value} for {symbol}")
                return self._create_empty_dataframe()
            
            # Get column mapping for provider
            column_map = self.column_maps.get(provider)
            if not column_map:
                logger.error(f"No column mapping for provider: {provider.value}")
                return self._create_empty_dataframe()
            
            # Standardize columns
            standardized_df = self._apply_column_mapping(df, column_map, provider, symbol)
            
            # Ensure proper data types
            standardized_df = self._ensure_data_types(standardized_df)
            
            # Validate data quality
            standardized_df = self._validate_and_clean_data(standardized_df, symbol)
            
            # Add metadata
            standardized_df['symbol'] = symbol
            standardized_df['provider'] = provider.value
            
            # Set timestamp as index
            if 'timestamp' in standardized_df.columns:
                standardized_df.set_index('timestamp', inplace=True)
                standardized_df.sort_index(inplace=True)
            
            logger.debug(f"Standardized {len(standardized_df)} bars from {provider.value} for {symbol}")
            return standardized_df
            
        except Exception as e:
            logger.error(f"Failed to standardize data from {provider.value} for {symbol}: {e}")
            return self._create_empty_dataframe()
    
    def _apply_column_mapping(self, df: pd.DataFrame, column_map: Dict[str, str], 
                            provider: DataProvider, symbol: str) -> pd.DataFrame:
        """Apply column mapping to convert provider columns to standard names"""
        try:
            standardized = pd.DataFrame()
            
            for standard_col, provider_col in column_map.items():
                if provider_col in df.columns:
                    standardized[standard_col] = df[provider_col]
                elif provider_col.lower() in [col.lower() for col in df.columns]:
                    # Case-insensitive match
                    actual_col = next(col for col in df.columns if col.lower() == provider_col.lower())
                    standardized[standard_col] = df[actual_col]
                else:
                    logger.warning(f"Column '{provider_col}' not found in {provider.value} data for {symbol}")
                    # Set to NaN for missing columns
                    standardized[standard_col] = np.nan
            
            return standardized
            
        except Exception as e:
            logger.error(f"Failed to apply column mapping for {provider.value}: {e}")
            return self._create_empty_dataframe()
    
    def _ensure_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure proper data types for all columns"""
        try:
            for col, dtype in self.standard_dtypes.items():
                if col in df.columns:
                    if col == 'timestamp':
                        # Handle timestamp conversion
                        df[col] = pd.to_datetime(df[col], utc=True)
                    else:
                        # Convert to numeric, coercing errors to NaN
                        df[col] = pd.to_numeric(df[col], errors='coerce')
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to ensure data types: {e}")
            return df
    
    def _validate_and_clean_data(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Validate and clean the standardized data"""
        try:
            original_length = len(df)
            
            # Remove rows with invalid OHLC data
            required_cols = ['open', 'high', 'low', 'close']
            df = df.dropna(subset=required_cols)
            
            # Validate OHLC relationships
            valid_mask = (
                (df['high'] >= df['open']) &
                (df['high'] >= df['close']) &
                (df['high'] >= df['low']) &
                (df['low'] <= df['open']) &
                (df['low'] <= df['close']) &
                (df['open'] > 0) &
                (df['high'] > 0) &
                (df['low'] > 0) &
                (df['close'] > 0)
            )
            
            df = df[valid_mask]
            
            # Fill missing volume with 0
            df['volume'] = df['volume'].fillna(0)
            
            cleaned_length = len(df)
            if cleaned_length < original_length:
                logger.info(f"Cleaned {symbol} data: {original_length} -> {cleaned_length} bars")
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to validate data for {symbol}: {e}")
            return df
    
    def _create_empty_dataframe(self) -> pd.DataFrame:
        """Create empty DataFrame with standard structure"""
        return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume', 'symbol', 'provider'])
    
    def merge_multiple_sources(self, data_sources: List[Dict[str, Any]], symbol: str) -> pd.DataFrame:
        """
        Merge data from multiple sources with conflict resolution
        
        Args:
            data_sources: List of dicts with 'data', 'provider', and optional 'priority'
            symbol: Trading symbol
            
        Returns:
            Merged and standardized DataFrame
        """
        try:
            standardized_sources = []
            
            for source in data_sources:
                provider = DataProvider(source['provider'])
                raw_data = source['data']
                priority = source.get('priority', 1)
                
                std_data = self.standardize_data(raw_data, provider, symbol)
                if not std_data.empty:
                    std_data['priority'] = priority
                    standardized_sources.append(std_data)
            
            if not standardized_sources:
                logger.warning(f"No valid data sources for {symbol}")
                return self._create_empty_dataframe()
            
            # Merge sources based on priority (higher priority overwrites lower)
            merged_df = standardized_sources[0].copy()
            
            for source_df in standardized_sources[1:]:
                # Merge on timestamp index
                merged_df = self._merge_with_priority(merged_df, source_df)
            
            # Clean up priority column
            if 'priority' in merged_df.columns:
                merged_df.drop('priority', axis=1, inplace=True)
            
            logger.info(f"Merged {len(standardized_sources)} sources for {symbol}: {len(merged_df)} bars")
            return merged_df
            
        except Exception as e:
            logger.error(f"Failed to merge multiple sources for {symbol}: {e}")
            return self._create_empty_dataframe()
    
    def _merge_with_priority(self, df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
        """Merge two DataFrames with priority-based conflict resolution"""
        try:
            # Combine both DataFrames
            combined = pd.concat([df1, df2])
            
            # Sort by priority (higher first) and timestamp
            combined = combined.sort_values(['priority', combined.index], ascending=[False, True])
            
            # Remove duplicates keeping first (highest priority)
            combined = combined[~combined.index.duplicated(keep='first')]
            
            return combined.sort_index()
            
        except Exception as e:
            logger.error(f"Failed to merge with priority: {e}")
            return df1

class UnifiedDataPipeline:
    """
    Complete data pipeline with multiple provider support and failover
    """
    
    def __init__(self):
        self.standardizer = DataStandardizer()
        self.provider_priority = {
            DataProvider.ALPACA: 1,
            DataProvider.BINANCE: 2,
            DataProvider.YAHOO: 3,
            DataProvider.COINGECKO: 4,
            DataProvider.ALPHAVANTAGE: 5
        }
    
    def fetch_standardized_data(self, symbol: str, timeframe: str = '15Min', 
                              limit: int = 200, providers: Optional[List[DataProvider]] = None) -> pd.DataFrame:
        """
        Fetch and standardize data from multiple providers with automatic failover
        
        Args:
            symbol: Trading symbol
            timeframe: Data timeframe (15Min, 1Hour, 1Day)
            limit: Number of bars to fetch
            providers: List of providers to try (in order)
            
        Returns:
            Standardized DataFrame with OHLCV data
        """
        if providers is None:
            providers = [DataProvider.ALPACA, DataProvider.YAHOO, DataProvider.BINANCE]
        
        last_error = None
        
        for provider in providers:
            try:
                logger.debug(f"Attempting to fetch {symbol} data from {provider.value}")
                
                # Fetch raw data from provider
                raw_data = self._fetch_from_provider(provider, symbol, timeframe, limit)
                
                if raw_data is not None:
                    # Standardize the data
                    standardized = self.standardizer.standardize_data(raw_data, provider, symbol)
                    
                    if not standardized.empty:
                        logger.info(f"Successfully fetched {len(standardized)} bars for {symbol} from {provider.value}")
                        return standardized
                
            except Exception as e:
                last_error = e
                logger.warning(f"Failed to fetch {symbol} from {provider.value}: {e}")
                continue
        
        # All providers failed
        logger.error(f"All providers failed for {symbol}. Last error: {last_error}")
        return self.standardizer._create_empty_dataframe()
    
    def _fetch_from_provider(self, provider: DataProvider, symbol: str, 
                           timeframe: str, limit: int) -> Optional[Any]:
        """
        Fetch raw data from specific provider
        This is a placeholder - implement actual provider calls
        """
        try:
            if provider == DataProvider.ALPACA:
                return self._fetch_alpaca_data(symbol, timeframe, limit)
            elif provider == DataProvider.YAHOO:
                return self._fetch_yahoo_data(symbol, timeframe, limit)
            elif provider == DataProvider.BINANCE:
                return self._fetch_binance_data(symbol, timeframe, limit)
            elif provider == DataProvider.MOCK:
                return self._generate_mock_data(symbol, limit)
            else:
                logger.warning(f"Provider {provider.value} not implemented yet")
                return None
                
        except Exception as e:
            logger.error(f"Provider {provider.value} fetch failed: {e}")
            return None
    
    def _fetch_alpaca_data(self, symbol: str, timeframe: str, limit: int):
        """Fetch data from Alpaca (placeholder - implement actual API call)"""
        # This would call your existing Alpaca functions
        try:
            from alpaca import recent_bars
            return recent_bars(symbol, timeframe, limit)
        except ImportError:
            logger.warning("Alpaca module not available")
            return None
    
    def _fetch_yahoo_data(self, symbol: str, timeframe: str, limit: int):
        """Fetch data from Yahoo Finance (placeholder)"""
        # This would implement Yahoo Finance API call
        logger.debug(f"Yahoo fetch for {symbol} not implemented yet")
        return None
    
    def _fetch_binance_data(self, symbol: str, timeframe: str, limit: int):
        """Fetch data from Binance (placeholder)"""
        # This would implement Binance API call
        logger.debug(f"Binance fetch for {symbol} not implemented yet")
        return None
    
    def _generate_mock_data(self, symbol: str, limit: int) -> pd.DataFrame:
        """Generate mock OHLCV data for testing"""
        import random
        
        timestamps = pd.date_range(end=datetime.now(timezone.utc), periods=limit, freq='15T')
        
        mock_data = []
        base_price = 50000 if 'BTC' in symbol else 3000 if 'ETH' in symbol else 100
        
        for i, ts in enumerate(timestamps):
            # Simple random walk
            price_change = random.uniform(-0.02, 0.02)  # ±2% change
            close = base_price * (1 + price_change * i / limit)
            
            open_price = close * random.uniform(0.995, 1.005)
            high = max(open_price, close) * random.uniform(1.0, 1.01)
            low = min(open_price, close) * random.uniform(0.99, 1.0)
            volume = random.uniform(1000, 10000)
            
            mock_data.append({
                'timestamp': ts,
                'open': round(open_price, 2),
                'high': round(high, 2),
                'low': round(low, 2),
                'close': round(close, 2),
                'volume': round(volume, 2)
            })
        
        return pd.DataFrame(mock_data)

# Global pipeline instance
data_pipeline = UnifiedDataPipeline()

# Convenience functions
def get_standardized_data(symbol: str, timeframe: str = '15Min', limit: int = 200) -> pd.DataFrame:
    """Get standardized OHLCV data with automatic provider failover"""
    return data_pipeline.fetch_standardized_data(symbol, timeframe, limit)

def standardize_any_data(raw_data: Any, provider: str, symbol: str) -> pd.DataFrame:
    """Standardize data from any provider"""
    provider_enum = DataProvider(provider.lower())
    return data_pipeline.standardizer.standardize_data(raw_data, provider_enum, symbol)

if __name__ == "__main__":
    # Test the data pipeline
    print("🔧 Testing Unified Data Pipeline")
    print("=" * 40)
    
    # Test with mock data
    test_symbols = ['BTC/USD', 'ETH/USD']
    
    for symbol in test_symbols:
        print(f"\nTesting {symbol}:")
        
        # Test mock data generation
        mock_data = data_pipeline._generate_mock_data(symbol, 10)
        standardized = data_pipeline.standardizer.standardize_data(
            mock_data, DataProvider.MOCK, symbol
        )
        
        print(f"  Generated {len(standardized)} mock bars")
        if not standardized.empty:
            latest = standardized.iloc[-1]
            print(f"  Latest: O={latest['open']} H={latest['high']} L={latest['low']} C={latest['close']} V={latest['volume']}")
            print(f"  Provider: {latest['provider']}, Symbol: {latest['symbol']}")
