#!/usr/bin/env python3
"""
Automatic Position Monitor for Hybrid Trader
Runs as background service to check positions every 60 seconds
"""

import time
import sys
import os
from datetime import datetime

# Add project root to path
sys.path.append('.')

# Load environment
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Import our dynamic position manager
from manual_position_manager import DynamicPositionManager

def run_position_monitor():
    """Run position monitoring loop"""
    print("🔄 Starting Automatic Position Monitor")
    print("   Checking positions every 60 seconds")
    print("   Press Ctrl+C to stop")
    print()
    
    manager = DynamicPositionManager()
    cycle = 0
    
    try:
        while True:
            cycle += 1
            print(f"🔄 Cycle {cycle} - {datetime.now().strftime('%H:%M:%S')}")
            
            # Check positions
            positions = manager.get_positions()
            
            if not positions:
                print("   No positions to monitor")
            else:
                exit_positions = [p for p in positions if p['should_exit']]
                
                if exit_positions:
                    print(f"   🚨 Found {len(exit_positions)} positions needing exit:")
                    
                    for pos in exit_positions:
                        print(f"      {pos['symbol']}: {pos['exit_reason']}")
                        
                        # Auto-close if enabled
                        if os.getenv('TB_AUTO_POSITION_EXIT', '1') == '1':
                            print(f"      🤖 Auto-closing {pos['symbol']}")
                            if manager.close_position(pos['symbol'], pos['exit_reason']):
                                print(f"      ✅ Successfully closed {pos['symbol']}")
                            else:
                                print(f"      ❌ Failed to close {pos['symbol']}")
                            time.sleep(2)  # Rate limiting
                        else:
                            print(f"      ⏸️  Auto-exit disabled (set TB_AUTO_POSITION_EXIT=1 to enable)")
                else:
                    total_value = sum(p['market_value'] for p in positions)
                    avg_pnl = sum(p['pnl_pct'] for p in positions) / len(positions) * 100
                    print(f"   ✅ {len(positions)} positions OK | Value: ${total_value:,.2f} | Avg P&L: {avg_pnl:+.2f}%")
            
            # Sleep until next cycle
            time.sleep(60)
            
    except KeyboardInterrupt:
        print("\n🛑 Position monitor stopped by user")
    except Exception as e:
        print(f"\n❌ Position monitor error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_position_monitor()
