#!/usr/bin/env bash
set -euo pipefail
# Start the high-risk futures agent autonomous loop with retry mechanism

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

nohup bash -lc '
set -a; [ -f .env ] && source .env; set +a
export PYTHONPATH="${PYTHONPATH:-$PWD}"

# Futures trading loop with retry mechanism and internet recovery
while true; do
  echo "[start_futures_loop] Starting futures agent at $(date)" >> high_risk_futures_loop.log
  
  # Run the futures agent with error handling - 30 second intervals for real-time risk management
  python3 high_risk_futures_agent.py --continuous --interval 30 || {
    echo "[start_futures_loop] Futures agent crashed at $(date), restarting in 60s..." >> high_risk_futures_loop.log
    sleep 60
  }
  
  # Wait before retry (in case of immediate failure)
  sleep 30
  
  echo "[start_futures_loop] Restarting futures agent loop..." >> high_risk_futures_loop.log
done
' > high_risk_futures_loop.log 2> high_risk_futures_loop.err & disown

echo "[start_futures_loop] launched at $(date -u +%F_%T)" >> high_risk_futures_loop.log
