"""
Precision manager for crypto trading - handles rounding of quantities and prices
"""
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class PrecisionManager:
    """Manages precision for different crypto trading pairs"""
    
    def __init__(self):
        # Alpaca crypto precision rules
        self.price_precision = {
            'BTC/USD': 2,   # $50000.00
            'ETH/USD': 2,   # $3000.00
            'DOGE/USD': 4,  # $0.1000
            'SOL/USD': 2,   # $100.00
            'AVAX/USD': 2,  # $40.00
            'LTC/USD': 2,   # $100.00
            'BCH/USD': 2,   # $400.00
            'LINK/USD': 2,  # $15.00
            'UNI/USD': 2,   # $10.00
            'AAVE/USD': 2,  # $300.00
        }
        
        self.quantity_precision = {
            'BTC/USD': 6,   # 0.000001 BTC
            'ETH/USD': 6,   # 0.000001 ETH
            'DOGE/USD': 0,  # 1 DOGE (whole numbers)
            'SOL/USD': 4,   # 0.0001 SOL
            'AVAX/USD': 4,  # 0.0001 AVAX
            'LTC/USD': 6,   # 0.000001 LTC
            'BCH/USD': 6,   # 0.000001 BCH
            'LINK/USD': 4,  # 0.0001 LINK
            'UNI/USD': 4,   # 0.0001 UNI
            'AAVE/USD': 6,  # 0.000001 AAVE
        }
    
    def round_price(self, symbol: str, price: float) -> float:
        """Round price to the correct precision for the symbol"""
        precision = self.price_precision.get(symbol, 2)  # Default to 2 decimals
        return round(price, precision)
    
    def round_quantity(self, symbol: str, quantity: float) -> float:
        """Round quantity to the correct precision for the symbol"""
        precision = self.quantity_precision.get(symbol, 6)  # Default to 6 decimals
        return round(quantity, precision)
    
    def get_min_quantity(self, symbol: str) -> float:
        """Get minimum tradeable quantity for symbol"""
        precision = self.quantity_precision.get(symbol, 6)
        return 10 ** (-precision)
    
    def validate_order_size(self, symbol: str, quantity: float, price: float) -> bool:
        """Validate if order size meets minimum requirements"""
        min_notional = 1.0  # $1 minimum order value
        notional_value = quantity * price
        min_qty = self.get_min_quantity(symbol)
        
        if quantity < min_qty:
            logger.warning(f"Quantity {quantity} below minimum {min_qty} for {symbol}")
            return False
        
        if notional_value < min_notional:
            logger.warning(f"Order value ${notional_value:.2f} below minimum ${min_notional} for {symbol}")
            return False
        
        return True

# Global instance
precision_manager = PrecisionManager()