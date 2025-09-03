# Minimal futures trading platform stub
# This provides the basic interface needed by futures_integration.py

import os
import requests
import time
import hmac
import hashlib
from urllib.parse import urlencode
from typing import Dict, List, Optional

class MockFuturesPlatform:
    """Minimal futures platform interface for Binance testnet"""
    
    def __init__(self):
        self.base_url = "https://testnet.binancefuture.com"
        self.api_key = os.getenv("BINANCE_TESTNET_API_KEY")
        self.secret_key = os.getenv("BINANCE_TESTNET_SECRET_KEY")
    
    def get_positions(self) -> List[Dict]:
        """Get current positions"""
        try:
            timestamp = int(time.time() * 1000)
            query_string = f'timestamp={timestamp}'
            signature = hmac.new(
                self.secret_key.encode(), 
                query_string.encode(), 
                hashlib.sha256
            ).hexdigest()
            
            url = f'{self.base_url}/fapi/v2/positionRisk?{query_string}&signature={signature}'
            headers = {'X-MBX-APIKEY': self.api_key}
            response = requests.get(url, headers=headers)
            
            if response.status_code == 200:
                positions = response.json()
                return [p for p in positions if float(p.get('positionAmt', 0)) != 0]
            return []
        except Exception as e:
            print(f"Error getting positions: {e}")
            return []
    
    def get_account_balance(self) -> Dict:
        """Get account balance"""
        try:
            timestamp = int(time.time() * 1000)
            query_string = f'timestamp={timestamp}'
            signature = hmac.new(
                self.secret_key.encode(), 
                query_string.encode(), 
                hashlib.sha256
            ).hexdigest()
            
            url = f'{self.base_url}/fapi/v2/account?{query_string}&signature={signature}'
            headers = {'X-MBX-APIKEY': self.api_key}
            response = requests.get(url, headers=headers)
            
            if response.status_code == 200:
                account_data = response.json()
                available_balance = float(account_data.get('availableBalance', 0))
                return {'available_balance': available_balance}
            return {'available_balance': 0.0}
        except Exception as e:
            print(f"Error getting balance: {e}")
            return {'available_balance': 0.0}

# Global platform instance
futures_platform = MockFuturesPlatform()

# Required function stubs
def get_futures_data(symbol: str, timeframe: str, limit: int = 100) -> List[Dict]:
    """Get futures data - stub implementation"""
    return []

def get_perpetuals_data(symbol: str, timeframe: str, limit: int = 100) -> List[Dict]:
    """Get perpetuals data - stub implementation"""
    return []

def place_futures_order(symbol: str, side: str, amount: float, **kwargs) -> Dict:
    """Place futures order - stub implementation"""
    return {"error": "Order placement not implemented in stub"}

def switch_futures_platform(platform_name: str) -> bool:
    """Switch platform - stub implementation"""
    return True

def get_futures_platforms() -> List[str]:
    """Get available platforms"""
    return ["Binance Futures"]

def get_futures_platform_info(platform_name: str) -> Dict:
    """Get platform info"""
    return {"name": platform_name, "available": True}
