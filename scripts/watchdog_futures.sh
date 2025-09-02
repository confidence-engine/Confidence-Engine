#!/usr/bin/env bash
set -euo pipefail

# Watchdog for high-risk futures agent
# Checks if process is running, restarts if not, sends alerts

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Configuration
PROCESS_PATTERN="python3.*high_risk_futures_agent.py"
LOG_FILE="high_risk_futures_loop.log"
ERR_FILE="high_risk_futures_loop.err"
RESTART_SCRIPT="scripts/start_futures_loop.sh"
CHECK_INTERVAL=60  # seconds between checks

# Function to check if process is running
is_running() {
    if pgrep -f "$PROCESS_PATTERN" >/dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

# Function to send Discord alert
send_discord_alert() {
    local message="$1"
    local webhook_url="${DISCORD_WEBHOOK_URL:-}"

    if [ -z "$webhook_url" ]; then
        echo "[watchdog] No DISCORD_WEBHOOK_URL set, skipping alert"
        return
    fi

    PYTHONPATH="$ROOT_DIR" python3 - << PY
import os, sys
from scripts.discord_sender import send_discord_digest_to
url = os.environ.get("DISCORD_WEBHOOK_URL")
text = "$message"
embeds = [{
  "title": "Futures Agent Watchdog Alert",
  "description": text,
  "color": 0xFF6B35,
}]
send_discord_digest_to(url, embeds)
PY
}

# Function to check log freshness
check_log_freshness() {
    local max_age_min=${TB_WATCHDOG_MAX_LOG_AGE_MIN:-30}
    local now_ts=$(date +%s)

    for log_file in "$LOG_FILE" "$ERR_FILE"; do
        if [ -f "$log_file" ]; then
            local mtime=$(stat -f %m "$log_file" 2>/dev/null || stat -c %Y "$log_file" 2>/dev/null || echo 0)
            local age_min=$(( (now_ts - mtime) / 60 ))
            if [ "$age_min" -gt "$max_age_min" ]; then
                echo "[watchdog] Log stale: $log_file (${age_min}m > ${max_age_min}m)"
                return 1
            fi
        else
            echo "[watchdog] Missing log: $log_file"
            return 1
        fi
    done
    return 0
}

# Function to restart the agent
restart_agent() {
    echo "[watchdog] $(date -u +%F_%T) - Restarting futures agent..."

    # Kill any existing processes first
    pkill -f "$PROCESS_PATTERN" || true
    sleep 2

    # Start the restart script
    if [ -x "$RESTART_SCRIPT" ]; then
        bash "$RESTART_SCRIPT" &
        local new_pid=$!
        echo "[watchdog] Started restart script with PID: $new_pid"

        # Wait a moment and verify it's running
        sleep 5
        if is_running; then
            echo "[watchdog] Futures agent successfully restarted"
            send_discord_alert "✅ Futures agent restarted successfully"
        else
            echo "[watchdog] Failed to restart futures agent"
            send_discord_alert "❌ Failed to restart futures agent"
        fi
    else
        echo "[watchdog] Restart script not found or not executable: $RESTART_SCRIPT"
        send_discord_alert "❌ Restart script missing: $RESTART_SCRIPT"
    fi
}

# Main monitoring loop
echo "[watchdog] Starting futures agent watchdog (check every ${CHECK_INTERVAL}s)"

while true; do
    if ! is_running; then
        echo "[watchdog] $(date -u +%F_%T) - Futures agent not running, attempting restart"
        send_discord_alert "🚨 Futures agent stopped, attempting restart..."
        restart_agent
    else
        # Process is running, check log freshness
        if ! check_log_freshness; then
            echo "[watchdog] $(date -u +%F_%T) - Logs stale, agent may be hung"
            send_discord_alert "⚠️ Futures agent logs stale, may be hung"
            # Don't restart immediately, give it more time
        fi
    fi

    sleep "$CHECK_INTERVAL"
done
