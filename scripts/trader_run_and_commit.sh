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

# Environment (adjust as needed)
export TB_TRADER_OFFLINE=${TB_TRADER_OFFLINE:-0}
export TB_NO_TRADE=${TB_NO_TRADE:-0}
export TB_TRADER_NOTIFY=${TB_TRADER_NOTIFY:-1}
export TB_ENABLE_DISCORD=${TB_ENABLE_DISCORD:-1}

# Run trader single pass (no --loop); cooldown handles re-entries
CMD="python3 scripts/crypto_signals_trader.py \
  --tf 4h --symbols BTC/USD,ETH/USD \
  --entry-tolerance-bps 10 --entry-mid-zone --min-rr 2.0 \
  --cooldown-sec 3600 --order-ttl-min 30 --debug"

# Execute and tee to logs (append)
# Use bash -lc to ensure env/pyenv/conda shells work when called via launchd
/usr/bin/env bash -lc "$CMD" >> "$LOG_OUT" 2>> "$LOG_ERR" || true

# Auto-commit/push logs (only if there are changes)
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git add trader_loop.log || true
  if ! git diff --cached --quiet; then
    TS_UTC="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
    git commit -m "chore(trader): update trader_loop.log ${TS_UTC}" || true
    # Push only if upstream is configured
    if git rev-parse --abbrev-ref --symbolic-full-name @{u} >/dev/null 2>&1; then
      git push || true
    fi
  fi
fi
