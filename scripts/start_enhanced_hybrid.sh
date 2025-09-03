#!/bin/bash
# Enhanced Hybrid Trader with Intelligent Position Manager
# Runs both the main trader and intelligent position monitor in parallel

set -a; [ -f .env ] && source .env; set +a
export PYTHONPATH="${PYTHONPATH:-$PWD}"

echo "[enhanced_hybrid] Starting hybrid trader with intelligent position manager at $(date)"

# Enable intelligent position management
export TB_AUTO_POSITION_EXIT=${TB_AUTO_POSITION_EXIT:-1}
export TB_INTELLIGENT_TPSL=${TB_INTELLIGENT_TPSL:-1}

# Start intelligent position manager in background  
echo "[enhanced_hybrid] Starting intelligent position manager..."
nohup python3 manual_position_manager.py > position_manager.log 2>&1 &
MONITOR_PID=$!
echo "[enhanced_hybrid] Intelligent position manager started with PID $MONITOR_PID"

# Start the main hybrid trader loop  
echo "[enhanced_hybrid] Starting main trading loop..."
bash scripts/start_hybrid_loop.sh

# If we get here, main loop ended - kill the monitor
echo "[enhanced_hybrid] Main loop ended, stopping position manager (PID $MONITOR_PID)"
kill $MONITOR_PID 2>/dev/null || true

echo "[enhanced_hybrid] Enhanced hybrid trader stopped at $(date)"
