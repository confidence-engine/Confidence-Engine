#!/usr/bin/env python3
"""
Test script for the enhanced multi-asset hybrid crypto trader
"""

import os
import sys
from pathlib import Path

# Add project root to path
_THIS_DIR = Path(__file__).parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

def test_enhanced_trader():
    """Test the enhanced multi-asset trader in safe mode"""
    
    print("🚀 Testing Enhanced Multi-Asset Hybrid Trader")
    print("=" * 50)
    
    # Set environment variables for safe testing
    test_env = {
        "TB_TRADER_OFFLINE": "1",           # Offline mode
        "TB_NO_TRADE": "1",                 # No real trades
        "TB_NO_TELEGRAM": "1",              # No Telegram notifications
        "TB_ENABLE_DISCORD": "0",           # No Discord notifications
        "TB_MULTI_ASSET": "1",              # Enable multi-asset mode
        "TB_ASSET_LIST": "BTC/USD,ETH/USD,SOL/USD,LINK/USD",  # Alpaca-supported assets
        
        # Enhanced features
        "TB_USE_ENHANCED_RISK": "1",
        "TB_USE_KELLY_SIZING": "1", 
        "TB_USE_REGIME_DETECTION": "1",
        "TB_USE_ENSEMBLE_ML": "0",          # Disable ML for basic test
        "TB_USE_ADAPTIVE_STRATEGY": "1",
        
        # Risk settings
        "TB_MAX_POSITIONS": "3",
        "TB_MAX_CORRELATION": "0.7",
        "TB_PORTFOLIO_VAR_LIMIT": "0.02",
        "TB_MAX_RISK_FRAC": "0.01",
        
        # Trading parameters
        "TB_TP_PCT": "0.05",                # 5% take profit
        "TB_SL_PCT": "0.02",                # 2% stop loss
        "TB_SENTIMENT_CUTOFF": "0.55",      # Slightly bullish sentiment required
    }
    
    # Apply test environment
    for key, value in test_env.items():
        os.environ[key] = value
    
    try:
        # Import and run the enhanced trader
        from scripts.hybrid_crypto_trader import main
        
        print("🔧 Configuration:")
        print(f"  Multi-Asset Mode: ✅")
        print(f"  Assets: {test_env['TB_ASSET_LIST']}")
        print(f"  Enhanced Risk: ✅")
        print(f"  Kelly Sizing: ✅")
        print(f"  Regime Detection: ✅")
        print(f"  Adaptive Strategy: ✅")
        print(f"  Max Positions: {test_env['TB_MAX_POSITIONS']}")
        print()
        
        print("🏃 Running enhanced trader...")
        result = main()
        
        if result == 0:
            print("✅ Enhanced trader test completed successfully!")
        else:
            print(f"⚠️  Enhanced trader returned code: {result}")
            
        return result
        
    except Exception as e:
        print(f"❌ Error testing enhanced trader: {e}")
        import traceback
        traceback.print_exc()
        return 1

def test_single_asset_mode():
    """Test single asset mode for comparison"""
    
    print("\n🔄 Testing Single Asset Mode (for comparison)")
    print("=" * 50)
    
    # Set environment for single asset
    single_env = {
        "TB_TRADER_OFFLINE": "1",
        "TB_NO_TRADE": "1", 
        "TB_NO_TELEGRAM": "1",
        "TB_ENABLE_DISCORD": "0",
        "TB_MULTI_ASSET": "0",              # Disable multi-asset
        "SYMBOL": "BTC/USD",                # Single symbol
        "TB_USE_ENHANCED_RISK": "0",        # Basic mode
    }
    
    for key, value in single_env.items():
        os.environ[key] = value
    
    try:
        from scripts.hybrid_crypto_trader import main
        
        print("🔧 Configuration:")
        print(f"  Single Asset Mode: ✅")
        print(f"  Symbol: BTC/USD")
        print(f"  Enhanced Features: ❌ (basic mode)")
        print()
        
        print("🏃 Running basic trader...")
        result = main()
        
        if result == 0:
            print("✅ Basic trader test completed successfully!")
        else:
            print(f"⚠️  Basic trader returned code: {result}")
            
        return result
        
    except Exception as e:
        print(f"❌ Error testing basic trader: {e}")
        return 1

if __name__ == "__main__":
    print("🧪 Enhanced Hybrid Crypto Trader Test Suite")
    print("=" * 60)
    
    # Test enhanced multi-asset mode
    enhanced_result = test_enhanced_trader()
    
    # Test single asset mode for comparison  
    basic_result = test_single_asset_mode()
    
    print("\n📊 Test Results Summary")
    print("=" * 30)
    print(f"Enhanced Multi-Asset: {'✅ PASS' if enhanced_result == 0 else '❌ FAIL'}")
    print(f"Basic Single Asset:   {'✅ PASS' if basic_result == 0 else '❌ FAIL'}")
    
    if enhanced_result == 0:
        print("\n🎉 Enhanced multi-asset trader is ready!")
        print("💡 To enable live trading:")
        print("   export TB_TRADER_OFFLINE=0")
        print("   export TB_NO_TRADE=0")
        print("   export TB_MULTI_ASSET=1")
        print("   python3 scripts/hybrid_crypto_trader.py")
    else:
        print("\n⚠️  Enhanced trader needs debugging")
    
    sys.exit(max(enhanced_result, basic_result))
