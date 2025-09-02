#!/bin/bash
# Comprehensive monitoring status for all trading agents

echo "=== TRADING AGENTS MONITORING STATUS ==="
echo "Timestamp: $(date -u +%F_%T)"
echo ""

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

# Function to check process status
check_process() {
    local pattern="$1"
    local name="$2"
    if pgrep -f "$pattern" >/dev/null 2>&1; then
        echo "✅ $name: RUNNING"
        return 0
    else
        echo "❌ $name: STOPPED"
        return 1
    fi
}

# Function to check log freshness
check_log_freshness() {
    local log_file="$1"
    local name="$2"
    local max_age_min=${TB_HEALTH_MAX_LOG_AGE_MIN:-30}

    if [ ! -f "$log_file" ]; then
        echo "❌ $name log: MISSING ($log_file)"
        return 1
    fi

    local now_ts=$(date +%s)
    local mtime=$(stat -f %m "$log_file" 2>/dev/null || stat -c %Y "$log_file" 2>/dev/null || echo 0)
    local age_min=$(( (now_ts - mtime) / 60 ))

    if [ "$age_min" -gt "$max_age_min" ]; then
        echo "⚠️  $name log: STALE (${age_min}m > ${max_age_min}m)"
        return 1
    else
        echo "✅ $name log: FRESH (${age_min}m old)"
        return 0
    fi
}

# Function to check launchd service
check_launchd() {
    local label="$1"
    local name="$2"
    if launchctl list | grep -q "$label"; then
        echo "✅ $name launchd: LOADED"
        return 0
    else
        echo "❌ $name launchd: NOT LOADED"
        return 1
    fi
}

echo "=== PROCESSES ==="
check_process "python3 scripts/hybrid_crypto_trader.py" "Hybrid Crypto Trader"
check_process "python3.*high_risk_futures_agent.py" "High-Risk Futures Agent"
check_process "python3 scripts/crypto_signals_trader.py" "Crypto Signals Trader"
echo ""

echo "=== LOG FRESHNESS ==="
check_log_freshness "trader_loop.log" "Hybrid Trader"
check_log_freshness "trader_loop.err" "Hybrid Trader Error"
check_log_freshness "high_risk_futures_loop.log" "Futures Agent"
check_log_freshness "high_risk_futures_loop.err" "Futures Agent Error"
check_log_freshness "futures_watchdog.log" "Futures Watchdog"
check_log_freshness "futures_watchdog.err" "Futures Watchdog Error"
echo ""

echo "=== LAUNCHD SERVICES ==="
check_launchd "com.tracer.crypto-trader" "Crypto Signals Trader"
check_launchd "com.tracer.futures-watchdog" "Futures Watchdog"
check_launchd "com.tracer.weekly-propose-canary" "Weekly Canary"
echo ""

echo "=== RECENT ACTIVITY ==="

# Check recent runs
echo "Recent Hybrid Trader Runs:"
ls -la runs/ 2>/dev/null | head -6 || echo "No runs directory"

echo ""
echo "Recent Futures Agent Activity:"
if [ -f "high_risk_futures_loop.log" ]; then
    tail -5 high_risk_futures_loop.log | grep -E "(INFO|ERROR|Starting|Stopping)" || echo "No recent activity"
else
    echo "No futures log file"
fi

echo ""
echo "=== HEALTH CHECK STATUS ==="
# Run a quick health check
if [ -x "scripts/health_check.sh" ]; then
    echo "Running health check..."
    bash scripts/health_check.sh 2>/dev/null | grep -E "(OK|FAILED)" || echo "Health check completed"
else
    echo "Health check script not found"
fi

echo ""
echo "=== RECOMMENDATIONS ==="
echo "To start missing services:"
echo "  Futures Watchdog: ./setup_futures_watchdog.sh"
echo "  Hybrid Trader: bash scripts/start_hybrid_loop.sh"
echo "  Futures Agent: bash scripts/start_futures_loop.sh"
echo ""
echo "To monitor logs:"
echo "  tail -f trader_loop.log"
echo "  tail -f high_risk_futures_loop.log"
echo "  tail -f futures_watchdog.log"
