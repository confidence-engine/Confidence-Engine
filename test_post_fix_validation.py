#!/usr/bin/env python3
"""
Comprehensive Post-Fix Validation Test
Tests both trading agents for proper operation after duplicate order fixes
"""
import os
import time
import json
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
import re

def run_cmd(cmd):
    """Run shell command and return output"""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result.stdout.strip(), result.stderr.strip(), result.returncode

def test_hybrid_agent():
    """Test hybrid agent for duplicate order prevention"""
    print("🔍 TESTING HYBRID AGENT...")
    
    # 1. Check if process is running
    stdout, stderr, code = run_cmd("pgrep -f hybrid_crypto_trader")
    if code != 0:
        print("❌ Hybrid agent not running")
        return False
    print(f"✅ Hybrid agent processes found: {len(stdout.strip().split())}")
    
    # 2. Analyze recent trading logs for duplicates
    print("\n🔍 Analyzing trading logs for duplicate patterns...")
    
    # Check last 1000 lines for order patterns
    stdout, _, _ = run_cmd("tail -1000 trading_agent.log")
    lines = stdout.split('\n')
    
    # Extract order submissions
    order_submissions = []
    order_fills = []
    
    for line in lines:
        if "'event': 'order_submitted'" in line:
            # Extract symbol, side, qty, timestamp
            symbol_match = re.search(r"'symbol': '([^']+)'", line)
            side_match = re.search(r"'side': '([^']+)'", line)
            qty_match = re.search(r"'qty': ([0-9.]+)", line)
            timestamp_match = re.search(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", line)
            
            if all([symbol_match, side_match, qty_match, timestamp_match]):
                order_submissions.append({
                    'symbol': symbol_match.group(1),
                    'side': side_match.group(1),
                    'qty': float(qty_match.group(1)),
                    'timestamp': timestamp_match.group(1)
                })
        
        elif "'event': 'order_filled'" in line:
            symbol_match = re.search(r"'symbol': '([^']+)'", line)
            side_match = re.search(r"'side': '([^']+)'", line)
            qty_match = re.search(r"'qty': ([0-9.]+)", line)
            timestamp_match = re.search(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", line)
            
            if all([symbol_match, side_match, qty_match, timestamp_match]):
                order_fills.append({
                    'symbol': symbol_match.group(1),
                    'side': side_match.group(1),
                    'qty': float(qty_match.group(1)),
                    'timestamp': timestamp_match.group(1)
                })
    
    print(f"✅ Found {len(order_submissions)} order submissions")
    print(f"✅ Found {len(order_fills)} order fills")
    
    # 3. Check for duplicate orders (same symbol+side within 10 seconds)
    duplicates = []
    for i, order1 in enumerate(order_submissions):
        for j, order2 in enumerate(order_submissions[i+1:], i+1):
            if (order1['symbol'] == order2['symbol'] and 
                order1['side'] == order2['side']):
                
                # Check if within 10 seconds
                t1 = datetime.strptime(order1['timestamp'], '%Y-%m-%d %H:%M:%S')
                t2 = datetime.strptime(order2['timestamp'], '%Y-%m-%d %H:%M:%S')
                
                if abs((t2 - t1).total_seconds()) <= 10:
                    duplicates.append((order1, order2))
    
    if duplicates:
        print(f"❌ FOUND {len(duplicates)} POTENTIAL DUPLICATES:")
        for dup in duplicates[:3]:  # Show first 3
            print(f"  {dup[0]['symbol']} {dup[0]['side']} @ {dup[0]['timestamp']} vs {dup[1]['timestamp']}")
        return False
    else:
        print("✅ No duplicate orders detected in hybrid agent")
    
    # 4. Check order/fill ratio (should be close to 1:1)
    if len(order_submissions) > 0:
        ratio = len(order_fills) / len(order_submissions)
        print(f"✅ Order fill ratio: {ratio:.2f} (should be ~1.0)")
        if ratio < 0.8:
            print("⚠️ Low fill ratio - possible order issues")
    
    return True

def test_futures_agent():
    """Test futures agent for cooldown system"""
    print("\n🔍 TESTING FUTURES AGENT...")
    
    # 1. Check if process is running
    stdout, stderr, code = run_cmd("pgrep -f 'Python.*high_risk_futures'")
    if code != 0:
        print("❌ Futures agent not running")
        return False
    print(f"✅ Futures agent process found: {stdout}")
    
    # 2. Analyze recent logs for cooldown behavior
    print("\n🔍 Analyzing futures logs for cooldown system...")
    
    stdout, _, _ = run_cmd("tail -500 futures_fixed_duplicate_bug.log")
    lines = stdout.split('\n')
    
    # Count cooldown skips and trades
    cooldown_skips = 0
    trades_executed = 0
    symbols_blocked = set()
    
    for line in lines:
        if "Skipping" in line and "cooldown active" in line:
            cooldown_skips += 1
            # Extract symbol
            symbol_match = re.search(r"Skipping (\w+) -", line)
            if symbol_match:
                symbols_blocked.add(symbol_match.group(1))
        
        elif "Trade" in line and "executed this cycle" in line:
            trades_executed += 1
    
    print(f"✅ Cooldown system active: {cooldown_skips} symbols blocked")
    print(f"✅ Unique symbols blocked by cooldown: {len(symbols_blocked)}")
    print(f"✅ Trades executed in recent cycles: {trades_executed}")
    
    # 3. Check for rapid duplicate orders
    trade_attempts = []
    for line in lines:
        if "order placed:" in line and "USDT" in line:
            timestamp_match = re.search(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", line)
            symbol_match = re.search(r"(\w+USDT)", line)
            if timestamp_match and symbol_match:
                trade_attempts.append({
                    'symbol': symbol_match.group(1),
                    'timestamp': timestamp_match.group(1)
                })
    
    # Check for rapid-fire trades (< 300s apart)
    rapid_trades = []
    for i, trade1 in enumerate(trade_attempts):
        for j, trade2 in enumerate(trade_attempts[i+1:], i+1):
            if trade1['symbol'] == trade2['symbol']:
                t1 = datetime.strptime(trade1['timestamp'], '%Y-%m-%d %H:%M:%S')
                t2 = datetime.strptime(trade2['timestamp'], '%Y-%m-%d %H:%M:%S')
                
                time_diff = abs((t2 - t1).total_seconds())
                if time_diff < 300:  # Less than 5 minutes
                    rapid_trades.append((trade1, trade2, time_diff))
    
    if rapid_trades:
        print(f"⚠️ FOUND {len(rapid_trades)} RAPID TRADES (< 300s):")
        for rapid in rapid_trades[:3]:
            print(f"  {rapid[0]['symbol']} trades {rapid[2]:.0f}s apart")
        return False
    else:
        print("✅ No rapid duplicate trades detected (300s cooldown working)")
    
    return True

def test_trading_volumes():
    """Test that trading volumes are reasonable"""
    print("\n🔍 TESTING TRADING VOLUMES...")
    
    # Check recent hybrid trades
    stdout, _, _ = run_cmd("tail -200 trading_agent.log | grep 'Executed.*@'")
    hybrid_trades = stdout.strip().split('\n') if stdout.strip() else []
    
    # Extract trade amounts
    hybrid_amounts = []
    for trade in hybrid_trades:
        amount_match = re.search(r'Executed [A-Z]+ for [^:]+: ([0-9.]+) @', trade)
        if amount_match:
            hybrid_amounts.append(float(amount_match.group(1)))
    
    if hybrid_amounts:
        avg_hybrid = sum(hybrid_amounts) / len(hybrid_amounts)
        max_hybrid = max(hybrid_amounts)
        print(f"✅ Hybrid trades: {len(hybrid_amounts)} recent trades")
        print(f"✅ Average hybrid trade: ${avg_hybrid:.2f}")
        print(f"✅ Max hybrid trade: ${max_hybrid:.2f}")
        
        # Check for excessive trades (> $10,000 each)
        large_trades = [amt for amt in hybrid_amounts if amt > 10000]
        if large_trades:
            print(f"⚠️ {len(large_trades)} large trades detected (> $10k)")
        
        # Hard cap check (should be < $1000 per trade typically)
        if max_hybrid > 50000:  # Very conservative
            print(f"❌ EXCESSIVE TRADE SIZE: ${max_hybrid:.2f}")
            return False
    else:
        print("ℹ️ No recent hybrid trades found")
    
    return True

def main():
    print("=" * 60)
    print("🧪 COMPREHENSIVE POST-FIX VALIDATION TEST")
    print("=" * 60)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    results = []
    
    # Test 1: Hybrid Agent
    results.append(test_hybrid_agent())
    
    # Test 2: Futures Agent  
    results.append(test_futures_agent())
    
    # Test 3: Trading Volumes
    results.append(test_trading_volumes())
    
    print("\n" + "=" * 60)
    print("🏁 TEST SUMMARY")
    print("=" * 60)
    
    if all(results):
        print("✅ ALL TESTS PASSED - Agents working correctly post-fix")
        return 0
    else:
        failed = sum(1 for r in results if not r)
        print(f"❌ {failed} TESTS FAILED - Issues detected")
        return 1

if __name__ == "__main__":
    exit(main())
