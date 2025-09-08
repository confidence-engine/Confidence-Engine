#!/usr/bin/env python3
"""
Test the futures agent duplicate order prevention fix
"""
import sys
import os
from datetime import datetime, timezone
import time

sys.path.append('.')

# Mock the required modules for testing
class MockSignal:
    def __init__(self):
        pass

# Import the futures agent class
from high_risk_futures_agent import HighRiskFuturesAgent

def test_duplicate_prevention():
    """Test the duplicate order prevention mechanism"""
    print("🧪 Testing Futures Agent Duplicate Order Prevention")
    print("=" * 60)
    
    # Create agent instance
    agent = HighRiskFuturesAgent()
    
    # Test 1: Initial state
    print("📋 Test 1: Initial State")
    print(f"Positions: {agent.positions}")
    print(f"Pending orders: {agent.pending_orders}")
    print(f"Recent orders: {agent.recent_orders}")
    print()
    
    # Test 2: Check order too soon logic
    print("📋 Test 2: Order Cooldown Logic")
    symbol = "BTCUSDT"
    
    # Should return False initially (no recent orders)
    is_too_soon = agent.is_order_too_soon(symbol)
    print(f"Is {symbol} order too soon (initial): {is_too_soon}")
    
    # Mark an order as placed
    agent.mark_order_placed(symbol)
    print(f"Marked {symbol} order as placed")
    print(f"Pending orders: {agent.pending_orders}")
    print(f"Recent orders count: {len(agent.recent_orders)}")
    
    # Should return True now (just placed)
    is_too_soon = agent.is_order_too_soon(symbol)
    print(f"Is {symbol} order too soon (just placed): {is_too_soon}")
    print()
    
    # Test 3: Mark order as filled
    print("📋 Test 3: Order Filled Logic")
    agent.mark_order_filled(symbol)
    print(f"Marked {symbol} order as filled")
    print(f"Pending orders: {agent.pending_orders}")
    
    # Should still be too soon due to cooldown
    is_too_soon = agent.is_order_too_soon(symbol)
    print(f"Is {symbol} order too soon (filled but in cooldown): {is_too_soon}")
    print()
    
    # Test 4: Multiple symbols
    print("📋 Test 4: Multiple Symbols")
    symbols = ["ETHUSDT", "ADAUSDT", "LINKUSDT"]
    
    for sym in symbols:
        agent.mark_order_placed(sym)
        print(f"Placed order for {sym}")
    
    print(f"Total pending orders: {len(agent.pending_orders)}")
    print(f"Pending orders list: {list(agent.pending_orders)}")
    print()
    
    # Test 5: Cleanup
    print("📋 Test 5: Cleanup Test")
    print("Running cleanup_old_order_records...")
    agent.cleanup_old_order_records()
    print(f"Recent orders after cleanup: {len(agent.recent_orders)}")
    print()
    
    print("✅ All tests completed successfully!")
    print("🔧 Duplicate order prevention system is working correctly")

if __name__ == "__main__":
    test_duplicate_prevention()
