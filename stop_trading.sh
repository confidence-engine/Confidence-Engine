#!/usr/bin/env bash
# 🛑 SIMPLE TRADING SYSTEM STOP SCRIPT

echo "🛑 STOPPING TRADING SYSTEM"
echo "=========================="
echo "Date: $(date)"
echo

# Function to safely kill processes
stop_processes() {
    local pattern="$1"
    local name="$2"
    
    if pgrep -f "$pattern" >/dev/null 2>&1; then
        echo "   Stopping $name..."
        pkill -f "$pattern" 2>/dev/null || true
        sleep 2
        
        # Force kill if still running
        if pgrep -f "$pattern" >/dev/null 2>&1; then
            echo "   Force killing $name..."
            pkill -9 -f "$pattern" 2>/dev/null || true
            sleep 1
        fi
        echo "   ✅ $name stopped"
    else
        echo "   ℹ️  $name not running"
    fi
}

stop_processes "python3 scripts/hybrid_crypto_trader.py" "Hybrid Agent"
stop_processes "python3.*high_risk_futures_agent.py" "Futures Agent" 
stop_processes "user_watchdog" "User Watchdog"

echo
echo "✅ ALL TRADING PROCESSES STOPPED"
echo "================================"
echo "💡 To restart: ./start_simple.sh"
