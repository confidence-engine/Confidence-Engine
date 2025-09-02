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
  FAIL_MSGS+=("Hybrid trader process not running")
fi

# 1b) Futures agent process running
if pgrep -f "python3.*high_risk_futures_agent.py" >/dev/null 2>&1; then
  :
else
  FAIL_MSGS+=("Futures agent process not running")
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
      FAIL_MSGS+=("Hybrid log stale: $f (${age_min}m > ${max_log_age_min}m)")
    fi
  else
    FAIL_MSGS+=("Missing hybrid log: $f")
  fi
done

# 2b) Futures agent log freshness
FUTURES_LOG_FILE="high_risk_futures_loop.log"
FUTURES_ERR_FILE="high_risk_futures_loop.err"
for f in "$FUTURES_LOG_FILE" "$FUTURES_ERR_FILE"; do
  if [ -f "$f" ]; then
    mtime=$(stat -f %m "$f" 2>/dev/null || stat -c %Y "$f" 2>/dev/null || echo 0)
    age_min=$(( (NOW_TS - mtime) / 60 ))
    if [ "$age_min" -gt "$max_log_age_min" ]; then
      FAIL_MSGS+=("Futures log stale: $f (${age_min}m > ${max_log_age_min}m)")
    fi
  else
    FAIL_MSGS+=("Missing futures log: $f")
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

# 4) ML model health: latest symlink, can-infer probe, and ml_prob monitoring
if [ "${TB_USE_ML_GATE:-1}" = "1" ]; then
  ML_LATEST="eval_runs/ml/latest"
  if [ ! -d "$ML_LATEST" ]; then
    FAIL_MSGS+=("Missing ML latest dir: $ML_LATEST")
  else
    # Verify expected files exist
    if [ ! -f "$ML_LATEST/model.pt" ]; then
      FAIL_MSGS+=("ML latest missing model.pt")
    fi
    if [ ! -f "$ML_LATEST/features.csv" ] && [ -z "${TB_ML_FEATURES_PATH:-}" ]; then
      FAIL_MSGS+=("ML latest missing features.csv and TB_ML_FEATURES_PATH not set")
    fi
  fi

  # Lightweight sanity: ensure non-empty artifacts exist (avoid heavy inference here)
  for f in "$ML_LATEST/model.pt" "$ML_LATEST/features.csv"; do
    if [ -f "$f" ]; then
      if [ ! -s "$f" ]; then
        FAIL_MSGS+=("ML artifact empty: $f")
      fi
    fi
  done

  # Alert if latest points to an old directory (stale model)
  MAX_AGE_HR=${TB_ML_LATEST_MAX_AGE_HR:-24}
  # Resolve the real path of the model dir
  REAL_DIR=$(python3 - <<'PY'
import os
import sys
p = os.path.realpath('eval_runs/ml/latest')
print(p)
PY
)
  if [ -d "$REAL_DIR" ]; then
    mtime=$(stat -f %m "$REAL_DIR" 2>/dev/null || stat -c %Y "$REAL_DIR" 2>/dev/null || echo 0)
    age_hr=$(( (NOW_TS - mtime) / 3600 ))
    if [ "$age_hr" -gt "$MAX_AGE_HR" ]; then
      FAIL_MSGS+=("ML latest target is stale: ${age_hr}h > ${MAX_AGE_HR}h (dir: ${REAL_DIR})")
    fi
  else
    FAIL_MSGS+=("ML latest resolves to non-dir: ${REAL_DIR}")
  fi

  # Monitor recent ml_prob values for missing/constant signals
  # Skip this check in offline mode to avoid deterministic synthetic data causing false alarms
  if [ "${TB_TRADER_OFFLINE:-0}" != "1" ]; then
    LOOKBACK=${TB_HEALTH_ML_PROB_LOOKBACK:-20}
    PYTHONPATH="$ROOT_DIR" python3 - "$LOOKBACK" <<'PY'
import glob, json, os, sys
N = int(sys.argv[1])
runs = sorted([p for p in glob.glob('runs/*') if os.path.isdir(p)], reverse=True)[:N]
vals = []
for r in runs:
    p = os.path.join(r, 'inputs.json')
    try:
        with open(p, 'r') as f:
            obj = json.load(f)
        v = obj.get('ml_prob', None)
        if isinstance(v, (int, float)):
            vals.append(float(v))
    except Exception:
        pass
if len(vals) == 0:
    print('[ML health] No recent ml_prob values found')
    sys.exit(11)
# if all equal within tiny tolerance, flag as possibly stuck
if max(vals) - min(vals) < 1e-6:
    print('[ML health] ml_prob constant across recent runs (possible stuck model)')
    sys.exit(11)
print('[ML health] recent ml_prob variability OK')
sys.exit(0)
PY
  fi
  rc=$?
  if [ "$rc" = "10" ]; then
    FAIL_MSGS+=("ml_prob missing across recent runs")
  elif [ "$rc" = "11" ]; then
    FAIL_MSGS+=("ml_prob constant across recent runs (possible stuck model)")
  fi
fi

# 5) Exit or alert
if [ ${#FAIL_MSGS[@]} -eq 0 ]; then
  echo "[health_check] OK at $(date -u +%F_%T)"
  exit 0
fi

msg="TRADING AGENTS HEALTH CHECK FAILED\n- $(printf "%s\n- " "${FAIL_MSGS[@]}")"

# Discord alert (gated by TB_ENABLE_DISCORD=1 and webhook)
if [ "${TB_ENABLE_DISCORD:-0}" = "1" ] && [ -n "${DISCORD_WEBHOOK_URL:-}" ]; then
  PYTHONPATH="$ROOT_DIR" python3 - "$DISCORD_WEBHOOK_URL" "$msg" <<'PY'
import os, sys
from scripts.discord_sender import send_discord_digest_to
url = sys.argv[1]
text = sys.argv[2]
embeds = [{
  "title": "Trading Agents Health Check FAILED",
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
text = os.environ.get("HEALTH_FAIL_MSG", "Trading agents health check failed.")
send_telegram_text(text)
PY
fi

# Also write a local marker
echo "[health_check] FAILED at $(date -u +%F_%T):" >> trader_loop.err
for m in "${FAIL_MSGS[@]}"; do echo "  - $m" >> trader_loop.err; done

exit 1
