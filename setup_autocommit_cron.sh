#!/bin/bash
# Cron Setup Script for Auto-Commit
# Sets up periodic auto-commit every 30 minutes

echo "Setting up periodic auto-commit..."

# Get the project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="$PROJECT_ROOT/periodic_autocommit.sh"

# Check if script exists and is executable
if [ ! -x "$SCRIPT_PATH" ]; then
    echo "Error: Auto-commit script not found or not executable: $SCRIPT_PATH"
    exit 1
fi

# Create cron job (runs every 30 minutes)
CRON_JOB="*/30 * * * * $SCRIPT_PATH >> $PROJECT_ROOT/autocommit_cron.log 2>&1"

# Check if cron job already exists
if crontab -l 2>/dev/null | grep -q "$SCRIPT_PATH"; then
    echo "Cron job already exists. Updating..."
    # Remove existing cron job
    crontab -l 2>/dev/null | grep -v "$SCRIPT_PATH" | crontab -
fi

# Add new cron job
(crontab -l 2>/dev/null; echo "$CRON_JOB") | crontab -

echo "✅ Cron job added successfully!"
echo "Auto-commit will run every 30 minutes."
echo "Cron job: $CRON_JOB"
echo ""
echo "To view cron jobs: crontab -l"
echo "To remove cron job: crontab -r"
echo "To check cron logs: tail -f $PROJECT_ROOT/autocommit_cron.log"
