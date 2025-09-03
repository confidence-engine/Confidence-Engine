#!/usr/bin/env python3
"""
Comprehensive testing of all implemented fixes
"""

import os
import sys
from pathlib import Path

def test_futures_agent():
    """Test enhanced futures agent functionality"""
    print('🔧 COMPREHENSIVE FIXES TESTING')
    print('=' * 50)
    
    try:
        # Test 1: Enhanced Futures Agent
        from high_risk_futures_agent import HighRiskFuturesAgent
        agent = HighRiskFuturesAgent()
        
        print('✅ TEST 1: Enhanced Futures Agent')
        print(f'   Kelly sizer: {agent.kelly_sizer is not None}')
        print(f'   Risk manager: {agent.risk_manager is not None}')
        print(f'   Ultra-conservative risk: {agent.risk_per_trade * 100:.1f}% (target: 0.3%)')
        print(f'   Reduced leverage: {agent.max_leverage}x (target: 5x)')
        print(f'   Single position: {agent.max_positions} (target: 1)')
        
        # Test 2: Kelly Risk Calculation
        test_signal = {'volatility': 0.05, 'strength': 0.7}
        enhanced_risk = agent.calculate_enhanced_risk_per_trade('BTCUSDT', test_signal)
        print(f'\n✅ TEST 2: Kelly Risk Calculation')
        print(f'   Base risk: {agent.risk_per_trade:.3f}')
        print(f'   Enhanced risk: {enhanced_risk:.3f}')
        print(f'   Risk reduction active: {enhanced_risk <= agent.risk_per_trade + 0.001}')
        
        print(f'\n✅ TEST 3: Performance Tracking')
        print(f'   Win tracking: {agent.win_count} wins')
        print(f'   Loss tracking: {agent.loss_count} losses')
        print(f'   Consecutive loss tracking: {agent.consecutive_losses}')
        print(f'   Total wins tracking: ${agent.total_wins:.2f}')
        print(f'   Total losses tracking: ${agent.total_losses:.2f}')
        
        # Test results
        kelly_ok = agent.kelly_sizer is not None
        risk_ok = agent.risk_manager is not None
        settings_ok = agent.risk_per_trade == 0.003 and agent.max_leverage == 5 and agent.max_positions == 1
        
        print(f'\n🎯 Futures Agent: {"✅ ALL TESTS PASSED" if kelly_ok and risk_ok and settings_ok else "❌ SOME TESTS FAILED"}')
        
        return kelly_ok and risk_ok and settings_ok
        
    except Exception as e:
        print(f'❌ Futures Agent Test Failed: {e}')
        return False

def test_hybrid_agent():
    """Test hybrid agent compatibility"""
    print(f'\n✅ TEST 4: Hybrid Agent Compatibility')
    
    try:
        # Check if hybrid agent can load with new settings
        from scripts.hybrid_crypto_trader import main
        print('   Hybrid agent imports: ✅')
        
        # Check env settings
        risk_per_trade = float(os.getenv('RISK_PER_TRADE', 0.05))
        max_leverage = int(os.getenv('MAX_LEVERAGE', 10))
        max_positions = int(os.getenv('MAX_POSITIONS', 3))
        
        print(f'   Hybrid risk: {risk_per_trade * 100:.1f}%')
        print(f'   Hybrid leverage: {max_leverage}x')
        print(f'   Hybrid positions: {max_positions}')
        
        return True
        
    except Exception as e:
        print(f'   Hybrid agent test error: {e}')
        return False

def test_env_settings():
    """Test environment configuration"""
    print(f'\n✅ TEST 5: Environment Settings')
    
    # Check futures settings
    futures_risk = float(os.getenv('FUTURES_RISK_PER_TRADE', 0.05))
    futures_leverage = int(os.getenv('FUTURES_MAX_LEVERAGE', 25))
    futures_positions = int(os.getenv('FUTURES_MAX_POSITIONS', 5))
    
    print(f'   Futures risk: {futures_risk * 100:.1f}% (target: 0.3%)')
    print(f'   Futures leverage: {futures_leverage}x (target: 5x)')
    print(f'   Futures positions: {futures_positions} (target: 1)')
    
    # Validate ultra-conservative settings
    conservative_ok = (
        futures_risk == 0.003 and 
        futures_leverage == 5 and 
        futures_positions == 1
    )
    
    print(f'   Ultra-conservative: {"✅" if conservative_ok else "❌"}')
    
    return conservative_ok

def test_performance_files():
    """Test performance optimization files"""
    print(f'\n✅ TEST 6: Performance Files')
    
    files_to_check = [
        'futures_performance_optimizer.py',
        'futures_risk_reduction.py',
        'dotusdt_loss_investigation.py'
    ]
    
    all_exist = True
    for file in files_to_check:
        exists = Path(file).exists()
        print(f'   {file}: {"✅" if exists else "❌"}')
        all_exist = all_exist and exists
    
    return all_exist

def main():
    """Run all tests"""
    print('🚀 TESTING ALL IMPLEMENTED FIXES')
    print('=' * 60)
    
    # Run all tests
    test_results = []
    test_results.append(test_futures_agent())
    test_results.append(test_hybrid_agent())
    test_results.append(test_env_settings())
    test_results.append(test_performance_files())
    
    # Summary
    passed = sum(test_results)
    total = len(test_results)
    
    print(f'\n🎯 FINAL RESULTS')
    print('=' * 30)
    print(f'Tests passed: {passed}/{total}')
    print(f'Success rate: {passed/total*100:.1f}%')
    
    if passed == total:
        print('✅ ALL FIXES WORKING - READY TO RESTART LIVE AGENTS')
        print('\n📋 RESTART INSTRUCTIONS:')
        print('1. Kill existing agents: pkill -f "hybrid_crypto_trader.py" && pkill -f "high_risk_futures_agent.py"')
        print('2. Start futures agent: nohup python3 high_risk_futures_agent.py --live > futures_live.log 2>&1 &')
        print('3. Start hybrid agent: nohup python3 scripts/hybrid_crypto_trader.py > hybrid_live.log 2>&1 &')
        print('4. Monitor: tail -f futures_live.log hybrid_live.log')
    else:
        print('❌ SOME TESTS FAILED - INVESTIGATE BEFORE RESTART')
    
    return passed == total

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
