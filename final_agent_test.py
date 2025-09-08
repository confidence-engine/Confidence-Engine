#!/usr/bin/env python3
"""
Final Comprehensive Test - Both Agents Fixed
"""

import subprocess
import re
import time
from datetime import datetime, timedelta

def check_both_agents():
    """Check if both agents are running properly."""
    
    print("🔍 FINAL AGENT VERIFICATION")
    print("=" * 50)
    
    # 1. Check processes
    try:
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        processes = result.stdout
        
        hybrid_running = 'scripts/hybrid_crypto_trader.py' in processes
        futures_running = 'high_risk_futures_agent.py --continuous' in processes
        
        print(f"🤖 Hybrid Agent: {'✅ RUNNING' if hybrid_running else '❌ STOPPED'}")
        print(f"🚀 Futures Agent: {'✅ RUNNING' if futures_running else '❌ STOPPED'}")
        
    except Exception as e:
        print(f"❌ Process check failed: {e}")
        return False
    
    # 2. Check recent hybrid activity (last 2 minutes)
    print(f"\n📈 HYBRID AGENT RECENT ACTIVITY (last 2 min)")
    print("-" * 40)
    
    cutoff_time = datetime.now() - timedelta(minutes=2)
    
    try:
        with open('trading_agent.log', 'r') as f:
            lines = f.readlines()
        
        recent_orders = []
        lock_messages = []
        errors = []
        
        for line in lines:
            # Extract timestamp
            timestamp_match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
            if not timestamp_match:
                continue
                
            try:
                log_time = datetime.strptime(timestamp_match.group(1), '%Y-%m-%d %H:%M:%S')
                if log_time >= cutoff_time:
                    if 'order_submitted' in line:
                        symbol_match = re.search(r"'symbol': '([^']+)'", line)
                        qty_match = re.search(r"'qty': ([0-9.]+)", line)
                        if symbol_match and qty_match:
                            recent_orders.append({
                                'time': log_time.strftime('%H:%M:%S'),
                                'symbol': symbol_match.group(1),
                                'qty': float(qty_match.group(1))
                            })
                    elif '🔒 Final submission' in line and 'lock held' in line:
                        lock_messages.append(line.strip())
                    elif 'Lock error' in line or 'No such file' in line:
                        errors.append(line.strip())
            except:
                continue
        
        print(f"📦 Orders placed: {len(recent_orders)}")
        for order in recent_orders:
            print(f"  • {order['time']} {order['symbol']}: {order['qty']:.6f}")
        
        print(f"🔒 Lock operations: {len(lock_messages)}")
        if lock_messages:
            for msg in lock_messages[-3:]:  # Show last 3
                match = re.search(r'Final submission for ([^\\s]+)', msg)
                if match:
                    print(f"  • Lock acquired for {match.group(1)}")
        
        if errors:
            print(f"❌ Lock errors: {len(errors)}")
            for err in errors[-2:]:
                print(f"  • {err}")
        else:
            print("✅ No lock errors")
        
    except Exception as e:
        print(f"❌ Hybrid log analysis failed: {e}")
    
    # 3. Check recent futures activity
    print(f"\n🚀 FUTURES AGENT RECENT ACTIVITY (last 2 min)")
    print("-" * 40)
    
    try:
        with open('futures_fixed_duplicate_bug.log', 'r') as f:
            lines = f.readlines()
        
        recent_cooldowns = []
        recent_trades = []
        
        for line in lines:
            timestamp_match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
            if not timestamp_match:
                continue
                
            try:
                log_time = datetime.strptime(timestamp_match.group(1), '%Y-%m-%d %H:%M:%S')
                if log_time >= cutoff_time:
                    if 'Skipping' in line and 'cooldown active' in line:
                        symbol_match = re.search(r'Skipping (\w+)', line)
                        if symbol_match:
                            recent_cooldowns.append(symbol_match.group(1))
                    elif 'Trade' in line and 'executed this cycle' in line:
                        recent_trades.append(line.strip())
            except:
                continue
        
        # Count unique symbols blocked
        unique_blocked = len(set(recent_cooldowns))
        print(f"🛡️ Symbols blocked by cooldown: {unique_blocked}")
        if recent_cooldowns:
            symbol_counts = {}
            for symbol in recent_cooldowns:
                symbol_counts[symbol] = symbol_counts.get(symbol, 0) + 1
            
            for symbol, count in list(symbol_counts.items())[:5]:  # Show top 5
                print(f"  • {symbol}: {count} blocks")
        
        print(f"📊 Trade cycles: {len(recent_trades)}")
        if recent_trades:
            for trade in recent_trades[-3:]:  # Show last 3
                match = re.search(r'Trade (\d+)/(\d+)', trade)
                if match:
                    print(f"  • {match.group(1)}/{match.group(2)} trades in cycle")
        
    except Exception as e:
        print(f"❌ Futures log analysis failed: {e}")
    
    # 4. Duplicate Analysis
    print(f"\n🔍 DUPLICATE ANALYSIS")
    print("-" * 40)
    
    # Check for potential hybrid duplicates
    duplicate_count = 0
    if len(recent_orders) > 1:
        for i in range(len(recent_orders)):
            for j in range(i+1, len(recent_orders)):
                order1, order2 = recent_orders[i], recent_orders[j]
                if (order1['symbol'] == order2['symbol'] and 
                    abs(order1['qty'] - order2['qty']) < 0.01):
                    duplicate_count += 1
                    print(f"⚠️ Potential duplicate: {order1['symbol']} {order1['qty']:.6f} vs {order2['qty']:.6f}")
    
    # 5. Final Assessment
    print(f"\n🎯 FINAL ASSESSMENT")
    print("=" * 50)
    
    hybrid_ok = hybrid_running and len(errors) == 0
    futures_ok = futures_running and unique_blocked > 0
    no_duplicates = duplicate_count == 0
    
    print(f"Hybrid Agent Status: {'✅ HEALTHY' if hybrid_ok else '❌ ISSUES'}")
    print(f"Futures Agent Status: {'✅ HEALTHY' if futures_ok else '❌ ISSUES'}")
    print(f"Duplicate Prevention: {'✅ WORKING' if no_duplicates else '❌ DUPLICATES FOUND'}")
    
    if hybrid_ok and futures_ok and no_duplicates:
        print(f"\n🎉 SUCCESS: Both agents are working correctly!")
        print("✅ Hybrid agent: File locking working, no errors")
        print("✅ Futures agent: Cooldown system active")  
        print("✅ No duplicate orders detected")
        return True
    else:
        print(f"\n⚠️ Issues detected:")
        if not hybrid_ok:
            print("- Hybrid agent has issues")
        if not futures_ok:
            print("- Futures agent cooldown not working")
        if not no_duplicates:
            print("- Duplicate orders still occurring")
        return False

if __name__ == "__main__":
    success = check_both_agents()
    exit(0 if success else 1)
