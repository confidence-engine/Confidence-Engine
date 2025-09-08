#!/usr/bin/env python3
"""
Comprehensive Agent Testing Script
Tests both hybrid and futures agents for duplicate order prevention and proper operation.
"""

import time
import subprocess
import json
from datetime import datetime, timedelta
import re
import os

def get_recent_log_entries(logfile, minutes=5):
    """Get log entries from the last N minutes."""
    try:
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        with open(logfile, 'r') as f:
            lines = f.readlines()
        
        recent_lines = []
        for line in lines:
            # Try to extract timestamp from various log formats
            timestamp_match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
            if timestamp_match:
                try:
                    log_time = datetime.strptime(timestamp_match.group(1), '%Y-%m-%d %H:%M:%S')
                    if log_time >= cutoff_time:
                        recent_lines.append(line.strip())
                except:
                    continue
        
        return recent_lines
    except Exception as e:
        return [f"Error reading {logfile}: {e}"]

def analyze_hybrid_duplicates(log_lines):
    """Analyze hybrid agent logs for duplicate orders."""
    orders = []
    duplicates = []
    lock_messages = []
    
    for line in log_lines:
        # Extract order information
        if "order_submitted" in line:
            try:
                # Extract symbol, qty, entry price, timestamp
                symbol_match = re.search(r"'symbol': '([^']+)'", line)
                qty_match = re.search(r"'qty': ([0-9.]+)", line)
                entry_match = re.search(r"'entry': ([0-9.]+)", line)
                timestamp_match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
                
                if symbol_match and qty_match and entry_match and timestamp_match:
                    order = {
                        'symbol': symbol_match.group(1),
                        'qty': float(qty_match.group(1)),
                        'entry': float(entry_match.group(1)),
                        'timestamp': timestamp_match.group(1),
                        'line': line
                    }
                    orders.append(order)
            except:
                continue
        
        # Check for file lock messages
        if "File lock" in line or "Acquired lock" in line or "Released lock" in line:
            lock_messages.append(line)
    
    # Find potential duplicates (same symbol, similar qty/entry within 60 seconds)
    for i in range(len(orders)):
        for j in range(i+1, len(orders)):
            order1, order2 = orders[i], orders[j]
            if (order1['symbol'] == order2['symbol'] and 
                abs(order1['qty'] - order2['qty']) < 0.01 and
                abs(order1['entry'] - order2['entry']) < 0.01):
                
                # Check if within 60 seconds
                time1 = datetime.strptime(order1['timestamp'], '%Y-%m-%d %H:%M:%S')
                time2 = datetime.strptime(order2['timestamp'], '%Y-%m-%d %H:%M:%S')
                if abs((time2 - time1).total_seconds()) <= 60:
                    duplicates.append((order1, order2))
    
    return orders, duplicates, lock_messages

def analyze_futures_cooldowns(log_lines):
    """Analyze futures agent logs for cooldown system operation."""
    cooldown_skips = []
    trades_executed = []
    
    for line in log_lines:
        if "Skipping" in line and "cooldown active" in line:
            cooldown_skips.append(line)
        elif "Trade" in line and "executed this cycle" in line:
            trades_executed.append(line)
    
    return cooldown_skips, trades_executed

def run_comprehensive_test():
    """Run comprehensive testing of both agents."""
    print("🔬 COMPREHENSIVE AGENT TESTING")
    print("=" * 60)
    
    # Test duration
    test_duration = 5 * 60  # 5 minutes
    print(f"🕐 Testing for {test_duration // 60} minutes...")
    
    start_time = datetime.now()
    
    # Monitor for the test duration
    time.sleep(test_duration)
    
    end_time = datetime.now()
    print(f"✅ Test completed: {start_time.strftime('%H:%M:%S')} - {end_time.strftime('%H:%M:%S')}")
    
    # Analyze results
    print("\n📊 ANALYSIS RESULTS")
    print("=" * 60)
    
    # 1. Hybrid Agent Analysis
    print("\n🤖 HYBRID AGENT (Alpaca)")
    print("-" * 40)
    
    hybrid_logs = get_recent_log_entries('trading_agent.log', minutes=6)
    orders, duplicates, lock_messages = analyze_hybrid_duplicates(hybrid_logs)
    
    print(f"📈 Total orders detected: {len(orders)}")
    print(f"🚨 Duplicate orders found: {len(duplicates)}")
    print(f"🔒 File lock messages: {len(lock_messages)}")
    
    if duplicates:
        print("\n❌ DUPLICATE ORDERS DETECTED:")
        for dup in duplicates:
            order1, order2 = dup
            time_diff = abs((datetime.strptime(order2['timestamp'], '%Y-%m-%d %H:%M:%S') - 
                           datetime.strptime(order1['timestamp'], '%Y-%m-%d %H:%M:%S')).total_seconds())
            print(f"  • {order1['symbol']}: {order1['qty']:.6f} @ ${order1['entry']:.4f} ({order1['timestamp']})")
            print(f"  • {order2['symbol']}: {order2['qty']:.6f} @ ${order2['entry']:.4f} ({order2['timestamp']})")
            print(f"    Time difference: {time_diff:.1f} seconds")
            print()
    else:
        print("✅ No duplicate orders detected!")
    
    if lock_messages:
        print(f"\n🔒 File locking system active ({len(lock_messages)} messages)")
        for msg in lock_messages[-3:]:  # Show last 3
            print(f"  • {msg}")
    
    # 2. Futures Agent Analysis  
    print("\n🚀 FUTURES AGENT (Binance)")
    print("-" * 40)
    
    futures_logs = get_recent_log_entries('futures_fixed_duplicate_bug.log', minutes=6)
    cooldown_skips, trades_executed = analyze_futures_cooldowns(futures_logs)
    
    print(f"⏸️  Cooldown skips: {len(cooldown_skips)}")
    print(f"✅ Trades executed: {len(trades_executed)}")
    
    if cooldown_skips:
        print("\n🛡️ COOLDOWN SYSTEM ACTIVE:")
        # Group by symbol
        symbols_blocked = {}
        for skip in cooldown_skips:
            symbol_match = re.search(r'Skipping (\w+)', skip)
            if symbol_match:
                symbol = symbol_match.group(1)
                symbols_blocked[symbol] = symbols_blocked.get(symbol, 0) + 1
        
        for symbol, count in symbols_blocked.items():
            print(f"  • {symbol}: {count} skips")
    
    if trades_executed:
        print(f"\n📊 Trade execution pattern:")
        for trade in trades_executed[-5:]:  # Show last 5
            match = re.search(r'Trade (\d+)/(\d+) executed', trade)
            if match:
                current, total = match.groups()
                print(f"  • {current}/{total} trades per cycle")
    
    # 3. Overall Assessment
    print("\n🎯 OVERALL ASSESSMENT")
    print("-" * 40)
    
    hybrid_status = "✅ WORKING" if len(duplicates) == 0 else "❌ DUPLICATES DETECTED"
    futures_status = "✅ WORKING" if len(cooldown_skips) > 0 else "⚠️ NO COOLDOWN ACTIVITY"
    
    print(f"Hybrid Agent: {hybrid_status}")
    print(f"Futures Agent: {futures_status}")
    
    if len(duplicates) == 0 and len(cooldown_skips) > 0:
        print("\n🎉 ALL SYSTEMS OPERATIONAL!")
        print("Both agents are working correctly with duplicate prevention active.")
    else:
        print("\n⚠️ ISSUES DETECTED!")
        if duplicates:
            print("- Hybrid agent still creating duplicates")
        if not cooldown_skips:
            print("- Futures agent cooldown system not active")

if __name__ == "__main__":
    run_comprehensive_test()
