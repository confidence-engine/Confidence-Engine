#!/usr/bin/env bash
set -euo pipefail
# Daily health check for the hybrid trading agent.
# - Verifies process running, log freshness, and recent runs/ artifacts.
# - On failure, sends alert via Discord/Telegram, gated by env.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

NOW_TS=$(date +%s)
FAIL_MSGS=()

# 1) Process running
if pgrep -f "python3 scripts/hybrid_crypto_trader.py" >/dev/null 2>&1; then
  :
else
  FAIL_MSGS+=("Trader process not running")
fi

# 2) Log freshness
LOG_FILE="trader_loop.log"
ERR_FILE="trader_loop.err"
max_log_age_min=${TB_HEALTH_MAX_LOG_AGE_MIN:-30}
for f in "$LOG_FILE" "$ERR_FILE"; do
  if [ -f "$f" ]; then
    mtime=$(stat -f %m "$f" 2>/dev/null || stat -c %Y "$f" 2>/dev/null || echo 0)
    age_min=$(( (NOW_TS - mtime) / 60 ))
    if [ "$age_min" -gt "$max_log_age_min" ]; then
      FAIL_MSGS+=("Log stale: $f (${age_min}m > ${max_log_age_min}m)")
    fi
  else
    FAIL_MSGS+=("Missing log: $f")
  fi
done

# 3) Recent run artifacts
max_runs_age_hr=${TB_HEALTH_MAX_RUNS_AGE_HR:-6}
latest_run=$(ls -1td runs/* 2>/dev/null | head -n1 || true)
if [ -n "$latest_run" ] && [ -d "$latest_run" ]; then
  mtime=$(stat -f %m "$latest_run" 2>/dev/null || stat -c %Y "$latest_run" 2>/dev/null || echo 0)
  age_hr=$(( (NOW_TS - mtime) / 3600 ))
  if [ "$age_hr" -gt "$max_runs_age_hr" ]; then
    FAIL_MSGS+=("No recent runs updates: ${age_hr}h > ${max_runs_age_hr}h (dir: ${latest_run})")
  fi
else
  FAIL_MSGS+=("No runs/ directories found")
fi

# 3b) Weekly promoted params freshness (ensures cron ran recently)
promoted_json="config/promoted_params.json"
max_promoted_age_days=${TB_HEALTH_MAX_PROMOTED_AGE_DAYS:-8}
if [ -f "$promoted_json" ]; then
  pm_mtime=$(stat -f %m "$promoted_json" 2>/dev/null || stat -c %Y "$promoted_json" 2>/dev/null || echo 0)
  pm_age_days=$(( (NOW_TS - pm_mtime) / 86400 ))
  if [ "$pm_age_days" -gt "$max_promoted_age_days" ]; then
    PROMOTED_STALE=1
  fi
else
  PROMOTED_STALE=1
fi

# If stale/missing, try self-healing once using a simple lock to avoid spam
if [ "${PROMOTED_STALE:-0}" = "1" ]; then
  LOCK_FILE="/tmp/weekly_propose_canary.lock"
  lock_age_max_min=${TB_HEALTH_SELFHEAL_LOCK_MAX_MIN:-360} # 6 hours
  NEED_RUN=1
  if [ -f "$LOCK_FILE" ]; then
    lock_mtime=$(stat -f %m "$LOCK_FILE" 2>/dev/null || stat -c %Y "$LOCK_FILE" 2>/dev/null || echo 0)
    lock_age_min=$(( (NOW_TS - lock_mtime) / 60 ))
    if [ "$lock_age_min" -lt "$lock_age_max_min" ]; then
      NEED_RUN=0
    fi
  fi
  if [ "$NEED_RUN" = "1" ]; then
    date -u +%F_%T > "$LOCK_FILE" || true
    echo "[health_check] Self-heal: running weekly_propose_canary.sh due to stale/missing promoted_params" >> trader_loop.log
    bash scripts/weekly_propose_canary.sh || true
  fi
  # Re-check freshness
  if [ -f "$promoted_json" ]; then
    pm_mtime=$(stat -f %m "$promoted_json" 2>/dev/null || stat -c %Y "$promoted_json" 2>/dev/null || echo 0)
    pm_age_days=$(( (NOW_TS - pm_mtime) / 86400 ))
    if [ "$pm_age_days" -gt "$max_promoted_age_days" ]; then
      FAIL_MSGS+=("promoted_params.json is stale after self-heal: ${pm_age_days}d > ${max_promoted_age_days}d")
    fi
  else
    FAIL_MSGS+=("Missing config/promoted_params.json after self-heal attempt")
  fi
fi

# 4) Exit or alert
if [ ${#FAIL_MSGS[@]} -eq 0 ]; then
  echo "[health_check] OK at $(date -u +%F_%T)"
  exit 0
fi

msg="HYBRID HEALTH CHECK FAILED\n- $(printf "%s\n- " "${FAIL_MSGS[@]}")"

# Discord alert (gated by TB_ENABLE_DISCORD=1 and webhook)
if [ "${TB_ENABLE_DISCORD:-0}" = "1" ] && [ -n "${DISCORD_WEBHOOK_URL:-}" ]; then
  PYTHONPATH="$ROOT_DIR" python3 - "$DISCORD_WEBHOOK_URL" "$msg" <<'PY'
import os, sys
from scripts.discord_sender import send_discord_digest_to
url = sys.argv[1]
text = sys.argv[2]
embeds = [{
  "title": "Hybrid health check FAILED",
  "description": text,
  "color": 0xFF0000,
}]
send_discord_digest_to(url, embeds)
PY
fi

# Telegram alert (gated by TB_NO_TELEGRAM)
if [ "${TB_NO_TELEGRAM:-0}" != "1" ]; then
  python3 - <<'PY'
import os
from scripts.tg_sender import send_telegram_text
text = os.environ.get("HEALTH_FAIL_MSG", "Hybrid health check failed.")
send_telegram_text(text)
PY
fi

# Also write a local marker
echo "[health_check] FAILED at $(date -u +%F_%T):" >> trader_loop.err
for m in "${FAIL_MSGS[@]}"; do echo "  - $m" >> trader_loop.err; done

exit 1
