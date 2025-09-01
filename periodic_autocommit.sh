#!/bin/bash
# Periodic Auto-Commit Script
# Permanent directive: Auto-commit all non-script files, exclude scripts/ and .env always
# This script can be run via cron: */30 * * * * /path/to/periodic_autocommit.sh

echo "$(date): Starting periodic auto-commit..."

# Get the project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Change to project directory
cd "$PROJECT_ROOT"

# Check if we're in a git repository
if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "$(date): Not a git repository, skipping auto-commit"
    exit 1
fi

# Run the enhanced auto-commit system
python3 autocommit_enhanced.py --run

echo "$(date): Periodic auto-commit completed."
