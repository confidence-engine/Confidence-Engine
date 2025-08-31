#!/usr/bin/env bash
set -euo pipefail
# Watchdog: ensure hybrid loop is running; restart if not. Optional Discord alert on restart.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

is_running() {
  pgrep -f "python3 scripts/hybrid_crypto_trader.py" >/dev/null 2>&1
}

send_discord_alert() {
  # Only attempts if TB_ENABLE_DISCORD=1 and DISCORD_WEBHOOK_URL set
  if [ "${TB_ENABLE_DISCORD:-0}" = "1" ] && [ -n "${DISCORD_WEBHOOK_URL:-}" ]; then
    PYTHONPATH="$ROOT_DIR" python3 - "$DISCORD_WEBHOOK_URL" <<'PY'
import os, sys
from scripts.discord_sender import send_discord_digest_to
url = sys.argv[1]
embeds = [{
  "title": "Hybrid watchdog restart",
  "description": "Watchdog detected trader not running and restarted the loop.",
  "color": 0xFFA500,
  "fields": [{"name": "Host", "value": os.uname().nodename}]
}]
send_discord_digest_to(url, embeds)
PY
  fi
}

if is_running; then
  exit 0
else
  bash scripts/start_hybrid_loop.sh || true
  echo "[watchdog] restarted at $(date -u +%F_%T)" >> trader_loop.log
  send_discord_alert || true
fi
