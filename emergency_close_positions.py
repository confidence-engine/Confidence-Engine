#!/usr/bin/env python3
"""
Emergency Position Closer - Force close all futures positions
"""

import os
import sys
import json
import time
from datetime import datetime

# Add project root to path
sys.path.append('.')

# Load environment
from dotenv import load_dotenv
load_dotenv()

def force_close_all_positions():
    """Force close all positions using the futures agent's position management"""
    try:
        from high_risk_futures_agent import HighRiskFuturesAgent
        
        print('🚨 EMERGENCY POSITION CLOSURE')
        print('=' * 50)
        print('Closing all legacy positions opened with old settings...')
        
        # Initialize agent
        agent = HighRiskFuturesAgent()
        
        # Get current positions from platform
        print('\n🔍 Fetching current positions...')
        agent.sync_existing_positions()
        
        if not agent.positions:
            print('✅ No positions found to close')
            return
        
        print(f'📊 Found {len(agent.positions)} positions to close:')
        for symbol, pos in agent.positions.items():
            pnl = pos.get('unrealized_pnl', 0)
            side = pos.get('side', 'unknown')
            size = pos.get('size', 0)
            print(f'   {symbol}: {side} {size} (PnL: ${pnl:.2f})')
        
        # Close all positions
        print('\n🔄 Closing all positions...')
        closed_count = 0
        
        for symbol in list(agent.positions.keys()):
            try:
                print(f'Closing {symbol}...')
                success = agent.close_position(symbol, reason="emergency_ultra_conservative_cleanup")
                if success:
                    closed_count += 1
                    print(f'✅ {symbol} closed successfully')
                else:
                    print(f'❌ Failed to close {symbol}')
                    
                # Small delay between closes
                time.sleep(1)
                
            except Exception as e:
                print(f'❌ Error closing {symbol}: {e}')
        
        print(f'\n🎯 SUMMARY: {closed_count}/{len(list(agent.positions.keys()))} positions closed')
        
        if closed_count > 0:
            print('✅ Emergency closure complete - ultra-conservative mode fully active')
        else:
            print('⚠️ Manual intervention may be required')
            
    except Exception as e:
        print(f'❌ Emergency closure failed: {e}')
        print('Please manually close positions via exchange interface')

if __name__ == '__main__':
    force_close_all_positions()
