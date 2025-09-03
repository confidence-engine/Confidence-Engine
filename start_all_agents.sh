#!/usr/bin/env bash
# 🚀 COMPREHENSIVE TRADING SYSTEM STARTUP GUIDE
# Start both agents with Phase 1C configuration and full monitoring

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

echo "🚀 STARTING COMPREHENSIVE TRADING SYSTEM"
echo "=" * 60
echo "Date: $(date)"
echo "Phase: 1C (Ultra-Aggressive Thresholds)"
echo "Target: Active trading with enhanced signal intelligence"
echo

# Set Phase 1C environment configuration
export TB_MIN_SIGNAL_QUALITY=1.0
export TB_MIN_CONVICTION_SCORE=2.0
export TB_USE_ENHANCED_SIGNALS=1
export TB_USE_REGIME_FILTERING=1
export TB_TRADER_OFFLINE=1
export TB_NO_TRADE=1
export TB_ENABLE_DISCORD=1
export TB_TRADER_NOTIFY=1
export TB_TRADER_NOTIFY_HEARTBEAT=1

echo "📊 Phase 1C Configuration:"
echo "   TB_MIN_SIGNAL_QUALITY=1.0 (ultra-permissive)"
echo "   TB_MIN_CONVICTION_SCORE=2.0 (ultra-permissive)"
echo "   Enhanced signals: ENABLED"
echo "   Regime filtering: ENABLED"
echo "   Notifications: ENABLED"
echo

# Function to check if process is running
is_hybrid_running() {
    pgrep -f "python3 scripts/hybrid_crypto_trader.py" >/dev/null 2>&1
}

is_futures_running() {
    pgrep -f "python3.*high_risk_futures_agent.py" >/dev/null 2>&1
}

# Stop any existing processes
echo "🛑 Stopping any existing trading processes..."
pkill -f "python3 scripts/hybrid_crypto_trader.py" 2>/dev/null || true
pkill -f "python3.*high_risk_futures_agent.py" 2>/dev/null || true
sleep 3

# Start hybrid agent
echo
echo "🧠 STARTING HYBRID CRYPTO AGENT..."
if bash scripts/start_hybrid_loop.sh; then
    echo "✅ Hybrid agent startup script launched"
    sleep 5
    if is_hybrid_running; then
        echo "✅ Hybrid agent confirmed running"
    else
        echo "⚠️  Hybrid agent startup may have issues - check logs"
    fi
else
    echo "❌ Failed to start hybrid agent"
fi

# Start futures agent
echo
echo "⚡ STARTING FUTURES AGENT..."
if bash scripts/start_futures_loop.sh; then
    echo "✅ Futures agent startup script launched"
    sleep 5
    if is_futures_running; then
        echo "✅ Futures agent confirmed running"
    else
        echo "⚠️  Futures agent startup may have issues - check logs"
    fi
else
    echo "❌ Failed to start futures agent"
fi

# Wait a moment for processes to stabilize
echo
echo "⏳ Waiting for processes to stabilize..."
sleep 10

# Check final status
echo
echo "📊 FINAL STATUS CHECK:"
echo "===================="

# Hybrid agent status
if is_hybrid_running; then
    HYBRID_PID=$(pgrep -f "python3 scripts/hybrid_crypto_trader.py" | head -1)
    echo "✅ Hybrid Agent: RUNNING (PID: $HYBRID_PID)"
else
    echo "❌ Hybrid Agent: NOT RUNNING"
fi

# Futures agent status
if is_futures_running; then
    FUTURES_PID=$(pgrep -f "python3.*high_risk_futures_agent.py" | head -1)
    echo "✅ Futures Agent: RUNNING (PID: $FUTURES_PID)"
else
    echo "❌ Futures Agent: NOT RUNNING"
fi

echo
echo "📋 MONITORING COMMANDS:"
echo "======================"
echo "# Check process status:"
echo "ps aux | grep -E '(hybrid_crypto_trader|high_risk_futures_agent)' | grep -v grep"
echo
echo "# View hybrid agent logs:"
echo "tail -f trader_loop.log"
echo "tail -f trader_loop.err"
echo
echo "# View futures agent logs:"
echo "tail -f high_risk_futures_loop.log"
echo "tail -f high_risk_futures_loop.err"
echo
echo "# Run health check:"
echo "bash scripts/health_check.sh"
echo
echo "# Stop all agents:"
echo "pkill -f 'python3.*hybrid_crypto_trader.py'"
echo "pkill -f 'python3.*high_risk_futures_agent.py'"

echo
echo "🚀 TRADING SYSTEM STARTUP COMPLETE"
echo "💡 Both agents should now be running with Phase 1C ultra-aggressive thresholds"
echo "📈 Expected result: Active trading with 3-5 trades per week"
