#!/usr/bin/env bash
# Periodic autocommit script to catch any missed artifacts
# Should be run every 5-10 minutes via cron

echo "[$(date)] Starting periodic autocommit..."
cd "$(dirname "$0")/.."

# Check if we're in a git repo
if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "Error: Not in a git repository"
    exit 1
fi

# Commit any missed artifacts
echo "Checking for uncommitted artifacts..."
python3 -c "
import autocommit as ac
import sys
try:
    result = ac.auto_commit_and_push(['runs', 'eval_runs', 'universe_runs', 'bars', 'trader_loop.log', 'trading_agent.log'], extra_message='periodic cleanup', push_enabled=True)
    print(f'Periodic autocommit result: {result}')
except Exception as e:
    print(f'Periodic autocommit error: {e}')
    sys.exit(1)
"

echo "[$(date)] Periodic autocommit completed"
