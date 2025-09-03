#!/usr/bin/env bash
# 🚀 SIMPLE TRADING SYSTEM STARTUP (NO ADMIN REQUIRED)
# Minimal deployment that just starts the trading agents

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

echo "🚀 SIMPLE TRADING SYSTEM STARTUP"
echo "================================"
echo "Date: $(date)"
echo "Mode: Real testnet trading (Alpaca + Binance)"
echo

# ============================================================================
# CONFIGURATION
# ============================================================================

echo "⚙️  Setting up configuration..."

# Load environment variables safely
if [ -f .env ]; then
    echo "   Loading .env configuration..."
    # Load only valid environment variable lines
    export $(grep -v '^#' .env | grep -v '^$' | grep '=' | xargs) 2>/dev/null || true
else
    echo "   No .env file found, using defaults..."
fi

# Override critical settings for real testnet trading
export TB_PAPER_TRADING=0
export TB_TRADER_OFFLINE=0
export TB_NO_TRADE=0

echo "✅ Configuration applied:"
echo "   Testnet Trading: ${TB_PAPER_TRADING} (Real testnet platforms: Alpaca + Binance)"
echo "   Offline Mode: ${TB_TRADER_OFFLINE}"
echo "   No Trade Mode: ${TB_NO_TRADE}"
echo

# ============================================================================
# CLEANUP EXISTING PROCESSES
# ============================================================================

echo "🛑 Stopping any existing trading processes..."

# Function to safely kill processes
cleanup_processes() {
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

cleanup_processes "python3 scripts/hybrid_crypto_trader.py" "Hybrid Agent"
cleanup_processes "python3.*high_risk_futures_agent.py" "Futures Agent"
cleanup_processes "user_watchdog" "User Watchdog"

echo "✅ Cleanup complete"
echo

# ============================================================================
# START TRADING AGENTS
# ============================================================================

echo "🤖 Starting trading agents..."

# Function to start agent in background
start_agent_simple() {
    local script="$1"
    local name="$2"
    local pattern="$3"
    
    echo "   Starting $name..."
    if [ -f "$script" ]; then
        local logfile=$(echo "$name" | tr '[:upper:]' '[:lower:]' | tr ' ' '_')_simple.log
        nohup bash "$script" > "$logfile" 2>&1 &
        echo "   ✅ $name startup script launched"
        sleep 3
        
        if pgrep -f "$pattern" >/dev/null 2>&1; then
            local pid=$(pgrep -f "$pattern" | head -1)
            echo "   ✅ $name confirmed running (PID: $pid)"
            return 0
        else
            echo "   ⚠️  $name startup may have issues - check $logfile"
            return 1
        fi
    else
        echo "   ❌ Script not found: $script"
        return 1
    fi
}

# Start Hybrid Agent
start_agent_simple "scripts/start_hybrid_loop.sh" "Hybrid Agent" "python3 scripts/hybrid_crypto_trader.py"

echo

# Start Futures Agent
start_agent_simple "scripts/start_futures_loop.sh" "Futures Agent" "python3.*high_risk_futures_agent.py"

echo "✅ Agent startup complete"
echo

# ============================================================================
# FINAL STATUS
# ============================================================================

echo "📊 SYSTEM STATUS:"
echo "================"

# Check agents
if pgrep -f "python3 scripts/hybrid_crypto_trader.py" >/dev/null 2>&1; then
    HYBRID_PID=$(pgrep -f "python3 scripts/hybrid_crypto_trader.py" | head -1)
    echo "✅ Hybrid Agent: RUNNING (PID: $HYBRID_PID)"
else
    echo "❌ Hybrid Agent: NOT RUNNING"
fi

if pgrep -f "python3.*high_risk_futures_agent.py" >/dev/null 2>&1; then
    FUTURES_PID=$(pgrep -f "python3.*high_risk_futures_agent.py" | head -1)
    echo "✅ Futures Agent: RUNNING (PID: $FUTURES_PID)"
else
    echo "❌ Futures Agent: NOT RUNNING"
fi

echo
echo "📋 MONITORING COMMANDS:"
echo "======================"
echo "# Check status:"
echo "ps aux | grep -E '(hybrid|futures)' | grep -v grep"
echo
echo "# View logs:"
echo "tail -f hybrid_agent_simple.log"
echo "tail -f futures_agent_simple.log"
echo
echo "# Stop all:"
echo "pkill -f 'python3.*hybrid_crypto_trader.py'; pkill -f 'python3.*high_risk_futures_agent.py'"

echo
echo "🚀 SIMPLE TRADING SYSTEM STARTUP COMPLETE!"
echo "=========================================="
echo "✅ Agents started with real testnet trading"
echo "✅ No admin privileges required"
echo "✅ Logs: hybrid_agent_simple.log, futures_agent_simple.log"
echo "🛡️  Simple deployment - no complex monitoring"
echo
echo "💡 To restart: ./start_simple.sh"
echo "💡 To stop: pkill -f 'python3.*hybrid_crypto_trader.py'; pkill -f 'python3.*high_risk_futures_agent.py'"
