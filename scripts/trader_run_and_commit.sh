#!/usr/bin/env bash
set -euo pipefail

# Resolve project root (this script resides in scripts/)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Logs
LOG_OUT="$PROJECT_ROOT/trader_loop.log"
LOG_ERR="$PROJECT_ROOT/trader_loop.err"

# Ensure log files exist without truncating
touch "$LOG_OUT" "$LOG_ERR"

# Ensure a reasonable PATH under launchd
export PATH=${PATH:-/usr/local/bin:/usr/bin:/bin:/opt/homebrew/bin}

# Load project .env into environment (if present), robust under -u
# Some .env lines may reference unset vars; temporarily disable nounset
LOADED_ENV=0
if [ -f .env ]; then
  set +u
  set -a
  . ./.env || true
  set +a
  set -u
  LOADED_ENV=1
fi

# Environment (adjust as needed)
export TB_TRADER_OFFLINE=${TB_TRADER_OFFLINE:-0}
export TB_NO_TRADE=${TB_NO_TRADE:-0}
# Safe-by-default: notifications disabled unless explicitly enabled via .env
export TB_TRADER_NOTIFY=${TB_TRADER_NOTIFY:-0}
export TB_ENABLE_DISCORD=${TB_ENABLE_DISCORD:-0}

# Resolve python3 binary explicitly (launchd may have minimal PATH)
PY_BIN="$(command -v python3 || true)"
if [ -z "${PY_BIN}" ]; then
  PY_BIN="/usr/bin/python3"
fi

# Run trader single pass (no --loop); cooldown handles re-entries
CMD="${PY_BIN} scripts/crypto_signals_trader.py \
  --tf 4h --symbols ${TB_TRADER_SYMBOLS:-BTC/USD} \
  --entry-tolerance-bps 10 --entry-mid-zone --min-rr 2.0 \
  --cooldown-sec 3600 --order-ttl-min 30 --debug"

# Enforce longs-only if requested via env
if [ "${TB_TRADER_LONGS_ONLY:-0}" = "1" ]; then
  CMD+=" --longs-only"
fi

# Emit a brief runtime env snapshot for debugging
{
  echo "[runner] ts=$(date -u +"%Y-%m-%dT%H:%M:%SZ") cwd=$(pwd)"
  if [ "$LOADED_ENV" = "1" ]; then
    echo "[runner] .env: loaded"
  else
    echo "[runner] .env: not found"
  fi
  echo "[runner] env: TF=4h symbols=${TB_TRADER_SYMBOLS:-BTC/USD} longs_only=${TB_TRADER_LONGS_ONLY:-0} allow_shorts=${TB_TRADER_ALLOW_SHORTS:-0}"
  echo "[runner] gates: offline=${TB_TRADER_OFFLINE:-} no_trade=${TB_NO_TRADE:-} notify=${TB_TRADER_NOTIFY:-} discord=${TB_ENABLE_DISCORD:-}"
} >> "$LOG_OUT" 2>> "$LOG_ERR"

# Execute and tee to logs (append)
# Temporarily disable -e so a nonzero STATUS doesn't abort the wrapper
set +e
eval "$CMD" >> "$LOG_OUT" 2>> "$LOG_ERR"
STATUS=$?
set -e
{
  echo "[runner] done status=$STATUS"
} >> "$LOG_OUT" 2>> "$LOG_ERR"

# Auto-commit/push non-code artifacts by default
# Controls:
#  - TB_AUTOCOMMIT_ARTIFACTS: enable/disable (default 1)
#  - TB_AUTOCOMMIT_PUSH: push to origin (default 1)
if [ "${TB_AUTOCOMMIT_ARTIFACTS:-1}" != "0" ]; then
  if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    TS_UTC="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
    PY_PUSH="${TB_AUTOCOMMIT_PUSH:-1}"
    python3 - << PY
import os
import autocommit as ac
push_enabled = os.getenv('TB_AUTOCOMMIT_PUSH','1') == '1'
paths = ['runs','eval_runs','universe_runs']
print(ac.auto_commit_and_push(paths, extra_message='trader artifacts', push_enabled=push_enabled))
PY
  fi
fi

# Give launchd a chance to observe a minimum runtime before exiting cleanly
# (mitigates EX_CONFIG flapping on very short-lived jobs)
sleep 12
# Always exit 0 for launchd; non-zero statuses are recorded above
exit 0
