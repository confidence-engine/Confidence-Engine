#!/bin/bash
# Setup script for futures agent watchdog launchd service

echo "Setting up futures agent watchdog..."

# Get the project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAUNCHD_PLIST="$PROJECT_ROOT/launchd/com.tracer.futures-watchdog.plist"
LAUNCHD_LABEL="com.tracer.futures-watchdog"

# Check if plist exists
if [ ! -f "$LAUNCHD_PLIST" ]; then
    echo "Error: Launchd plist not found: $LAUNCHD_PLIST"
    exit 1
fi

# Check if watchdog script exists and is executable
WATCHDOG_SCRIPT="$PROJECT_ROOT/scripts/watchdog_futures.sh"
if [ ! -x "$WATCHDOG_SCRIPT" ]; then
    echo "Error: Watchdog script not found or not executable: $WATCHDOG_SCRIPT"
    exit 1
fi

# Unload existing service if running
if launchctl list | grep -q "$LAUNCHD_LABEL"; then
    echo "Unloading existing futures watchdog service..."
    launchctl unload "$LAUNCHD_PLIST" 2>/dev/null || true
fi

# Load the new service
echo "Loading futures watchdog service..."
launchctl load "$LAUNCHD_PLIST"

# Verify the service is loaded
sleep 2
if launchctl list | grep -q "$LAUNCHD_LABEL"; then
    echo "✅ Futures watchdog service loaded successfully!"
    echo "Service label: $LAUNCHD_LABEL"
    echo ""
    echo "To check status: launchctl list | grep futures-watchdog"
    echo "To view logs: tail -f $PROJECT_ROOT/futures_watchdog.log"
    echo "To stop: launchctl unload $LAUNCHD_PLIST"
    echo "To restart: launchctl unload $LAUNCHD_PLIST && launchctl load $LAUNCHD_PLIST"
else
    echo "❌ Failed to load futures watchdog service"
    echo "Check logs: tail -f $PROJECT_ROOT/futures_watchdog.err"
    exit 1
fi

echo ""
echo "Futures watchdog will:"
echo "- Check every 60 seconds if futures agent is running"
echo "- Restart agent if stopped"
echo "- Send Discord alerts on failures"
echo "- Monitor log freshness"
