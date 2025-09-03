#!/usr/bin/env bash
# 📊 TRADING SYSTEM STATUS CHECKER

echo "📊 TRADING SYSTEM STATUS"
echo "======================="
echo "Date: $(date)"
echo

# Check agents
echo "AGENT STATUS:"
echo "============"

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

if pgrep -f "user_watchdog" >/dev/null 2>&1; then
    WATCHDOG_PID=$(pgrep -f "user_watchdog" | head -1)
    echo "✅ User Watchdog: RUNNING (PID: $WATCHDOG_PID)"
else
    echo "❌ User Watchdog: NOT RUNNING"
fi

echo
echo "LOG FILES:"
echo "=========="

if [ -f "hybrid_agent_simple.log" ]; then
    echo "✅ Hybrid Agent Log: hybrid_agent_simple.log ($(wc -l < hybrid_agent_simple.log) lines)"
else
    echo "❌ Hybrid Agent Log: Not found"
fi

if [ -f "futures_agent_simple.log" ]; then
    echo "✅ Futures Agent Log: futures_agent_simple.log ($(wc -l < futures_agent_simple.log) lines)"
else
    echo "❌ Futures Agent Log: Not found"
fi

if [ -f "trader_loop.log" ]; then
    echo "✅ Hybrid Loop Log: trader_loop.log ($(wc -l < trader_loop.log) lines)"
else
    echo "❌ Hybrid Loop Log: Not found"
fi

if [ -f "high_risk_futures_loop.log" ]; then
    echo "✅ Futures Loop Log: high_risk_futures_loop.log ($(wc -l < high_risk_futures_loop.log) lines)"
else
    echo "❌ Futures Loop Log: Not found"
fi

echo
echo "RECENT ACTIVITY:"
echo "==============="

if [ -f "trader_loop.log" ]; then
    echo "Hybrid Agent (last 3 entries):"
    tail -3 trader_loop.log 2>/dev/null || echo "   No recent activity"
    echo
fi

if [ -f "high_risk_futures_loop.log" ]; then
    echo "Futures Agent (last 3 entries):"
    tail -3 high_risk_futures_loop.log 2>/dev/null || echo "   No recent activity"
    echo
fi

echo "QUICK COMMANDS:"
echo "==============="
echo "View live logs: tail -f hybrid_agent_simple.log futures_agent_simple.log"
echo "Stop system:    ./stop_trading.sh"
echo "Restart system: ./start_simple.sh"
