#!/usr/bin/env python3
"""
Unified Precision Management System
Single source of truth for all symbol precision across all exchanges
"""

from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

# DEFINITIVE SYMBOL PRECISION MAPPING
# This is the SINGLE SOURCE OF TRUTH for all precision across the system
SYMBOL_PRECISION = {
    # Crypto pairs (Alpaca format)
    'BTC/USD': 2,    # $50,000.12
    'ETH/USD': 2,    # $3,000.45
    'SOL/USD': 3,    # $150.123
    'AVAX/USD': 3,   # $25.456
    'LINK/USD': 3,   # $10.789
    'UNI/USD': 3,    # $8.234
    'AAVE/USD': 2,   # $95.67
    'COMP/USD': 2,   # $45.89
    'YFI/USD': 0,    # $8,000 (whole dollars)
    'XTZ/USD': 4,    # $1.2345
    'LTC/USD': 2,    # $85.67
    'BCH/USD': 2,    # $250.89
    'ADA/USD': 4,    # $0.3456
    'DOT/USD': 3,    # $5.123
    'ALGO/USD': 4,   # $0.1234
    
    # Binance format (if using Binance)
    'BTCUSDT': 2,
    'ETHUSDT': 2,
    'SOLUSDT': 3,
    'AVAXUSDT': 3,
    'LINKUSDT': 3,
    'UNIUSDT': 3,
    'AAVEUSDT': 2,
    'COMPUSDT': 2,
    'YFIUSDT': 0,
    'XTZUSDT': 4,
    'LTCUSDT': 2,
    'BCHUSDT': 2,
    'ADAUSDT': 4,
    'DOTUSDT': 3,
    'ALGOUSDT': 4,
    
    # Stock symbols (if trading stocks)
    'AAPL': 2,
    'TSLA': 2,
    'MSFT': 2,
    'GOOGL': 2,
    'AMZN': 2,
    'NVDA': 2,
    'META': 2,
    
    # Futures symbols
    'BTCUSD': 2,
    'ETHUSD': 2,
    'MBT': 2,    # Micro Bitcoin
    'MET': 2,    # Micro Ether
}

# QUANTITY PRECISION (for order sizes)
QUANTITY_PRECISION = {
    # Crypto - based on minimum order sizes
    'BTC/USD': 6,    # 0.000001 BTC minimum
    'ETH/USD': 5,    # 0.00001 ETH minimum
    'SOL/USD': 3,    # 0.001 SOL minimum
    'AVAX/USD': 2,   # 0.01 AVAX minimum
    'LINK/USD': 2,   # 0.01 LINK minimum
    'UNI/USD': 2,    # 0.01 UNI minimum
    'AAVE/USD': 3,   # 0.001 AAVE minimum
    'COMP/USD': 3,   # 0.001 COMP minimum
    'YFI/USD': 6,    # 0.000001 YFI minimum
    'XTZ/USD': 1,    # 0.1 XTZ minimum
    'LTC/USD': 4,    # 0.0001 LTC minimum
    'BCH/USD': 4,    # 0.0001 BCH minimum
    'ADA/USD': 0,    # 1 ADA minimum
    'DOT/USD': 2,    # 0.01 DOT minimum
    'ALGO/USD': 0,   # 1 ALGO minimum
    
    # Default for unknown symbols
    'DEFAULT': 2,
}

class PrecisionManager:
    """
    Unified precision management for all trading operations
    Prevents precision errors that cause position closing failures
    """
    
    def __init__(self):
        self.price_precision = SYMBOL_PRECISION.copy()
        self.quantity_precision = QUANTITY_PRECISION.copy()
        
    def get_price_precision(self, symbol: str) -> int:
        """Get price precision for a symbol"""
        # Normalize symbol format
        normalized_symbol = self._normalize_symbol(symbol)
        precision = self.price_precision.get(normalized_symbol, 2)  # Default to 2 decimal places
        
        logger.debug(f"Price precision for {symbol} -> {normalized_symbol}: {precision}")
        return precision
    
    def get_quantity_precision(self, symbol: str) -> int:
        """Get quantity precision for a symbol"""
        normalized_symbol = self._normalize_symbol(symbol)
        precision = self.quantity_precision.get(normalized_symbol, 2)  # Default to 2 decimal places
        
        logger.debug(f"Quantity precision for {symbol} -> {normalized_symbol}: {precision}")
        return precision
    
    def round_price(self, symbol: str, price: float) -> float:
        """Round price to correct precision for symbol"""
        precision = self.get_price_precision(symbol)
        rounded = round(price, precision)
        
        logger.debug(f"Rounded price for {symbol}: {price} -> {rounded} (precision: {precision})")
        return rounded
    
    def round_quantity(self, symbol: str, quantity: float) -> float:
        """Round quantity to correct precision for symbol"""
        precision = self.get_quantity_precision(symbol)
        rounded = round(quantity, precision)
        
        logger.debug(f"Rounded quantity for {symbol}: {quantity} -> {rounded} (precision: {precision})")
        return rounded
    
    def format_price(self, symbol: str, price: float) -> str:
        """Format price as string with correct precision"""
        precision = self.get_price_precision(symbol)
        formatted = f"{price:.{precision}f}"
        
        return formatted
    
    def format_quantity(self, symbol: str, quantity: float) -> str:
        """Format quantity as string with correct precision"""
        precision = self.get_quantity_precision(symbol)
        formatted = f"{quantity:.{precision}f}"
        
        return formatted
    
    def validate_order_params(self, symbol: str, price: float, quantity: float) -> bool:
        """Validate order parameters against symbol precision requirements"""
        try:
            # Check if price precision is correct
            rounded_price = self.round_price(symbol, price)
            if abs(price - rounded_price) > 1e-10:  # Account for floating point precision
                logger.warning(f"Price {price} for {symbol} needs rounding to {rounded_price}")
                return False
            
            # Check if quantity precision is correct
            rounded_qty = self.round_quantity(symbol, quantity)
            if abs(quantity - rounded_qty) > 1e-10:  # Account for floating point precision
                logger.warning(f"Quantity {quantity} for {symbol} needs rounding to {rounded_qty}")
                return False
            
            # Check minimum order values (basic validation)
            if quantity <= 0:
                logger.error(f"Invalid quantity {quantity} for {symbol}")
                return False
            
            if price <= 0:
                logger.error(f"Invalid price {price} for {symbol}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Order validation failed for {symbol}: {e}")
            return False
    
    def _normalize_symbol(self, symbol: str) -> str:
        """Normalize symbol format for consistent lookup"""
        # Remove spaces and convert to uppercase
        normalized = symbol.strip().upper()
        
        # Handle different symbol formats
        if '/' in normalized:
            # Already in format like BTC/USD
            return normalized
        elif normalized.endswith('USDT'):
            # Convert BTCUSDT to BTC/USD equivalent for lookup
            base = normalized[:-4]  # Remove USDT
            return f"{base}/USD"
        elif normalized.endswith('USD') and not normalized.endswith('/USD'):
            # Convert BTCUSD to BTC/USD
            base = normalized[:-3]  # Remove USD
            return f"{base}/USD"
        else:
            # Assume it's a stock symbol or other format
            return normalized
    
    def add_symbol_precision(self, symbol: str, price_precision: int, quantity_precision: int):
        """Add precision mapping for a new symbol"""
        normalized_symbol = self._normalize_symbol(symbol)
        self.price_precision[normalized_symbol] = price_precision
        self.quantity_precision[normalized_symbol] = quantity_precision
        
        logger.info(f"Added precision for {symbol} -> {normalized_symbol}: price={price_precision}, quantity={quantity_precision}")
    
    def validate_precision(self, symbol: str, price: float, quantity: float) -> Dict[str, float]:
        """Validate and fix precision for both price and quantity"""
        return {
            'price': self.round_price(symbol, price),
            'quantity': self.round_quantity(symbol, quantity)
        }

# Global precision manager instance
precision_manager = PrecisionManager()

# Convenience functions for backward compatibility
def get_precision(symbol: str) -> int:
    """Get price precision for symbol (backward compatibility)"""
    return precision_manager.get_price_precision(symbol)

def round_price(symbol: str, price: float) -> float:
    """Round price to correct precision"""
    return precision_manager.round_price(symbol, price)

def round_quantity(symbol: str, quantity: float) -> float:
    """Round quantity to correct precision"""
    return precision_manager.round_quantity(symbol, quantity)

def format_price(symbol: str, price: float) -> str:
    """Format price as string with correct precision"""
    return precision_manager.format_price(symbol, price)

def format_quantity(symbol: str, quantity: float) -> str:
    """Format quantity as string with correct precision"""
    return precision_manager.format_quantity(symbol, quantity)

if __name__ == "__main__":
    # Test the precision manager
    print("🔧 Testing Precision Manager")
    print("=" * 40)
    
    test_symbols = ['BTC/USD', 'ETH/USD', 'SOL/USD', 'BTCUSDT', 'AAPL']
    test_price = 1234.56789
    test_quantity = 0.123456789
    
    for symbol in test_symbols:
        price_prec = precision_manager.get_price_precision(symbol)
        qty_prec = precision_manager.get_quantity_precision(symbol)
        rounded_price = precision_manager.round_price(symbol, test_price)
        rounded_qty = precision_manager.round_quantity(symbol, test_quantity)
        
        print(f"\n{symbol}:")
        print(f"  Price precision: {price_prec} -> {test_price} -> {rounded_price}")
        print(f"  Quantity precision: {qty_prec} -> {test_quantity} -> {rounded_qty}")
        print(f"  Formatted: ${precision_manager.format_price(symbol, test_price)} / {precision_manager.format_quantity(symbol, test_quantity)}")
