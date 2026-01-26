#!/usr/bin/env python3
"""
Monitor both trading agents and their ultra-conservative performance
"""

import os
import time
import subprocess
from datetime import datetime

def check_agent_status():
    """Check if both agents are running"""
    try:
        # Check processes
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        futures_running = 'high_risk_futures_agent.py --continuous' in result.stdout
        hybrid_running = 'hybrid_crypto_trader.py' in result.stdout
        
        print(f"🚀 AGENT STATUS CHECK - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        print(f"Futures Agent: {'✅ RUNNING' if futures_running else '❌ STOPPED'}")
        print(f"Hybrid Agent:  {'✅ RUNNING' if hybrid_running else '❌ STOPPED'}")
        
        return futures_running, hybrid_running
        
    except Exception as e:
        print(f"❌ Error checking agent status: {e}")
        return False, False

def get_recent_logs(log_file, lines=5):
    """Get recent log entries"""
    try:
        if os.path.exists(log_file):
            result = subprocess.run(['tail', '-n', str(lines), log_file], 
                                  capture_output=True, text=True)
            return result.stdout.strip()
        else:
            return "Log file not found"
    except Exception as e:
        return f"Error reading logs: {e}"

def main():
    """Monitor agents"""
    print("🎯 ULTRA-CONSERVATIVE TRADING MONITOR")
    print("=" * 60)
    print("Monitoring Key Changes:")
    print("✅ Futures Risk: 5% → 0.3% (94% reduction)")
    print("✅ Futures Leverage: 25x → 5x (80% reduction)")
    print("✅ Futures Positions: 5 → 1 (80% reduction)")
    print("✅ Kelly Criterion: Active")
    print("✅ Advanced Risk Manager: Active")
    print("✅ Performance Tracking: Active")
    print()
    
    # Check agent status
    futures_running, hybrid_running = check_agent_status()
    
    if not (futures_running and hybrid_running):
        print("⚠️  WARNING: One or both agents are not running!")
        print("\n📋 To restart:")
        print("pkill -f 'hybrid_crypto_trader.py' && pkill -f 'high_risk_futures_agent.py'")
        print("nohup python3 high_risk_futures_agent.py --continuous > futures_live.log 2>&1 &")
        print("nohup python3 scripts/hybrid_crypto_trader.py > hybrid_live.log 2>&1 &")
        return
    
    print("\n📊 RECENT FUTURES AGENT ACTIVITY")
    print("-" * 40)
    futures_logs = get_recent_logs('futures_live.log', 10)
    for line in futures_logs.split('\n'):
        if any(keyword in line for keyword in ['Risk per Trade', 'Max Leverage', 'Max Positions', 'Kelly', 'Trade=True', 'ENTRY', 'EXIT']):
            print(f"🎯 {line}")
    
    print("\n📊 RECENT HYBRID AGENT ACTIVITY")
    print("-" * 40)
    hybrid_logs = get_recent_logs('hybrid_live.log', 10)
    for line in hybrid_logs.split('\n'):
        if any(keyword in line for keyword in ['Trade=True', 'ENTRY', 'EXIT', 'Signal Analysis']):
            print(f"🎯 {line}")
    
    print("\n💼 MONITORING COMMANDS")
    print("-" * 40)
    print("📱 Live futures logs: tail -f futures_live.log")
    print("📱 Live hybrid logs:  tail -f hybrid_live.log")
    print("📊 Agent status:      python3 monitor_agents.py")
    print("🔍 Full logs:         tail -100 futures_live.log hybrid_live.log")

if __name__ == '__main__':
    main()
