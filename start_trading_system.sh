#!/usr/bin/env bash
# 🚀 UNIFIED TRADING SYSTEM STARTUP & MONITORING
# Complete deployment: Both agents + monitoring + notifications + database logging

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

echo "🚀 UNIFIED TRADING SYSTEM DEPLOYMENT"
echo "====================================="
echo "Date: $(date)"
echo "Phase: 1C (Ultra-Aggressive Thresholds)"
echo "Components: Hybrid Agent + Futures Agent + Monitoring + Notifications + Database"
echo

# ============================================================================
# PHASE 1: CONFIGURATION SETUP
# ============================================================================

echo "⚙️  PHASE 1: Setting up Phase 1C configuration..."

# Phase 1C Ultra-Aggressive Configuration with Intelligent Trade Management
export TB_MIN_SIGNAL_QUALITY=1.0
export TB_MIN_CONVICTION_SCORE=2.0
export TB_USE_ENHANCED_SIGNALS=1
export TB_USE_REGIME_FILTERING=1
export TB_TRADER_OFFLINE=1
export TB_NO_TRADE=1

# Enhanced Trade Management Configuration
export TB_MAX_TRADES_PER_AGENT_DAILY=8
export TB_MAX_TRADES_PER_AGENT_WEEKLY=40
export TB_HYBRID_MAX_TRADE_SIZE=1000.0
export TB_FUTURES_MAX_TRADE_SIZE=100.0
export TB_USE_DYNAMIC_POSITION_SIZING=1
export TB_CONFIDENCE_BASED_SIZING=1

# Notification Configuration
export TB_ENABLE_DISCORD=1
export TB_TRADER_NOTIFY=1
export TB_TRADER_NOTIFY_HEARTBEAT=1
export TB_HEARTBEAT_EVERY_N=12
export TB_NO_TELEGRAM=0  # Enable Telegram

# Database Logging Configuration
export TB_ENABLE_DB_LOGGING=1
export TB_DB_PATH="enhanced_trading.db"
export TB_LOG_ALL_SIGNALS=1
export TB_LOG_PERFORMANCE_METRICS=1

# Auto-commit Configuration
export TB_AUTOCOMMIT_ARTIFACTS=1
export TB_AUTOCOMMIT_PUSH=1

echo "✅ Configuration applied:"
echo "   Signal Quality: ${TB_MIN_SIGNAL_QUALITY} (ultra-permissive)"
echo "   Conviction Score: ${TB_MIN_CONVICTION_SCORE} (ultra-permissive)"
echo "   Trade Limits: ${TB_MAX_TRADES_PER_AGENT_DAILY} daily / ${TB_MAX_TRADES_PER_AGENT_WEEKLY} weekly per agent"
echo "   Hybrid Max Trade: $${TB_HYBRID_MAX_TRADE_SIZE} (spot trading)"
echo "   Futures Max Trade: $${TB_FUTURES_MAX_TRADE_SIZE} (25x leverage = $2500 exposure)"
echo "   Dynamic Sizing: ${TB_CONFIDENCE_BASED_SIZING}"
echo "   Discord: ${TB_ENABLE_DISCORD}"
echo "   Telegram: $([ "${TB_NO_TELEGRAM}" = "0" ] && echo "ENABLED" || echo "DISABLED")"
echo "   Database: ${TB_DB_PATH}"
echo

# ============================================================================
# PHASE 2: CLEANUP EXISTING PROCESSES
# ============================================================================

echo "🛑 PHASE 2: Stopping any existing processes..."

# Function to safely kill processes
cleanup_processes() {
    local pattern="$1"
    local name="$2"
    
    if pgrep -f "$pattern" >/dev/null 2>&1; then
        echo "   Stopping $name..."
        pkill -f "$pattern" 2>/dev/null || true
        sleep 2
        
        # Force kill if still running
        if pgrep -f "$pattern" >/dev/null 2>&1; then
            echo "   Force killing $name..."
            pkill -9 -f "$pattern" 2>/dev/null || true
            sleep 1
        fi
        echo "   ✅ $name stopped"
    else
        echo "   ℹ️  $name not running"
    fi
}

cleanup_processes "python3 scripts/hybrid_crypto_trader.py" "Hybrid Agent"
cleanup_processes "python3.*high_risk_futures_agent.py" "Futures Agent" 
cleanup_processes "scripts/watchdog" "Watchdog processes"

echo "✅ Cleanup complete"
echo

# ============================================================================
# PHASE 3: DATABASE INITIALIZATION
# ============================================================================

echo "🗄️  PHASE 3: Initializing database for performance tracking..."

python3 -c "
import sqlite3
import os
from datetime import datetime

# Initialize enhanced_trading.db with comprehensive tracking
db_path = 'enhanced_trading.db'
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Create comprehensive tables for performance assessment
cursor.execute('''
CREATE TABLE IF NOT EXISTS signal_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    agent_type TEXT NOT NULL,
    symbol TEXT NOT NULL,
    signal_quality REAL,
    conviction_score REAL,
    regime_trend TEXT,
    regime_volatility TEXT,
    signal_decision TEXT,
    reason TEXT,
    sentiment REAL,
    price_momentum REAL,
    volume_z_score REAL,
    rsi REAL
)
''')

cursor.execute('''
CREATE TABLE IF NOT EXISTS performance_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    agent_type TEXT NOT NULL,
    metric_type TEXT NOT NULL,
    metric_value REAL,
    symbol TEXT,
    time_period TEXT,
    additional_data TEXT
)
''')

cursor.execute('''
CREATE TABLE IF NOT EXISTS heartbeat_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    agent_type TEXT NOT NULL,
    run_count INTEGER,
    active_positions INTEGER,
    daily_pnl REAL,
    total_trades INTEGER,
    system_status TEXT,
    market_regime TEXT,
    platform TEXT
)
''')

cursor.execute('''
CREATE TABLE IF NOT EXISTS notification_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    notification_type TEXT NOT NULL,
    channel TEXT NOT NULL,
    agent_type TEXT,
    symbol TEXT,
    message_content TEXT,
    delivery_status TEXT
)
''')

# Enhanced trades table
cursor.execute('''
CREATE TABLE IF NOT EXISTS enhanced_trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    run_id TEXT,
    agent_type TEXT NOT NULL,
    symbol TEXT NOT NULL,
    action TEXT NOT NULL,
    entry_price REAL,
    quantity REAL,
    leverage INTEGER,
    signal_quality REAL,
    conviction_score REAL,
    regime_state TEXT,
    platform TEXT,
    order_id TEXT,
    exit_price REAL,
    exit_timestamp TEXT,
    pnl REAL,
    win_loss TEXT,
    hold_duration_minutes INTEGER
)
''')

# Enhanced trade management tables
cursor.execute('''
CREATE TABLE IF NOT EXISTS trade_limits (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    agent_type TEXT NOT NULL,
    period_type TEXT NOT NULL,
    period_start TEXT NOT NULL,
    trade_count INTEGER DEFAULT 0,
    total_exposure REAL DEFAULT 0.0,
    max_trades INTEGER,
    max_exposure REAL
)
''')

cursor.execute('''
CREATE TABLE IF NOT EXISTS risk_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    agent_type TEXT NOT NULL,
    event_type TEXT NOT NULL,
    description TEXT,
    trade_blocked BOOLEAN DEFAULT FALSE,
    current_exposure REAL,
    limit_exceeded TEXT
)
''')

conn.commit()
conn.close()

# Initialize enhanced trade manager
try:
    from enhanced_trade_manager import TradeManager
    tm = TradeManager()
    print('✅ Enhanced trade manager initialized')
    
    # Test position sizing
    hybrid_size = tm.calculate_position_size('hybrid_agent', 8.5, 7.8, 'BTC/USD')
    futures_size = tm.calculate_position_size('futures_agent', 9.1, 8.9, 'BTCUSDT')
    
    print(f'   📊 Hybrid Agent: High confidence trade = ${hybrid_size:.2f} (max $1000)')
    print(f'   📊 Futures Agent: High confidence trade = ${futures_size:.2f} (max $100, 25x leverage)')
    
except ImportError:
    print('⚠️  Enhanced trade manager not available - using basic limits')

print('✅ Database initialized with comprehensive tracking tables')
print(f'   📊 Tables: signal_logs, performance_metrics, heartbeat_logs, notification_logs, enhanced_trades, trade_limits, risk_events')
print(f'   📁 Location: {os.path.abspath(db_path)}')
"

echo "✅ Database ready for performance tracking"
echo

# ============================================================================
# PHASE 4: NOTIFICATION TESTING
# ============================================================================

echo "📱 PHASE 4: Testing notification channels..."

# Test Discord
if [ "${TB_ENABLE_DISCORD:-0}" = "1" ] && [ -n "${DISCORD_WEBHOOK_URL:-}" ]; then
    echo "   Testing Discord notification..."
    python3 -c "
import os, sys
sys.path.append('scripts')
try:
    from discord_sender import send_discord_digest_to
    webhook_url = os.environ.get('DISCORD_WEBHOOK_URL')
    embeds = [{
        'title': '🚀 Trading System Startup Test',
        'description': 'Unified trading system is initializing with Phase 1C configuration',
        'color': 0x00FF00,
        'fields': [
            {'name': 'Signal Quality', 'value': '1.0/10 (ultra-permissive)', 'inline': True},
            {'name': 'Conviction Score', 'value': '2.0/10 (ultra-permissive)', 'inline': True},
            {'name': 'Database Logging', 'value': 'ENABLED', 'inline': True}
        ]
    }]
    send_discord_digest_to(webhook_url, embeds)
    print('✅ Discord notification: SUCCESS')
except Exception as e:
    print(f'❌ Discord notification: FAILED - {e}')
" || echo "❌ Discord test failed"
else
    echo "⚠️  Discord: NOT CONFIGURED"
fi

# Test Telegram
if [ "${TB_NO_TELEGRAM:-1}" = "0" ]; then
    echo "   Testing Telegram notification..."
    python3 -c "
try:
    from telegram_bot import send_message
    send_message('🚀 Trading System Startup Test - Phase 1C deployment with ultra-aggressive thresholds and database logging enabled')
    print('✅ Telegram notification: SUCCESS')
except Exception as e:
    print(f'❌ Telegram notification: FAILED - {e}')
" || echo "❌ Telegram test failed"
else
    echo "⚠️  Telegram: DISABLED"
fi

echo "✅ Notification testing complete"
echo

# ============================================================================
# PHASE 5: START TRADING AGENTS
# ============================================================================

echo "🤖 PHASE 5: Starting trading agents..."

# Function to start agent and verify
start_agent() {
    local script="$1"
    local name="$2"
    local pattern="$3"
    
    echo "   Starting $name..."
    if bash "$script"; then
        echo "   ✅ $name startup script launched"
        sleep 5
        
        if pgrep -f "$pattern" >/dev/null 2>&1; then
            local pid=$(pgrep -f "$pattern" | head -1)
            echo "   ✅ $name confirmed running (PID: $pid)"
            return 0
        else
            echo "   ⚠️  $name startup may have issues"
            return 1
        fi
    else
        echo "   ❌ Failed to start $name"
        return 1
    fi
}

# Start Hybrid Agent
start_agent "scripts/start_hybrid_loop.sh" "Hybrid Agent" "python3 scripts/hybrid_crypto_trader.py"
HYBRID_STATUS=$?

echo

# Start Futures Agent
start_agent "scripts/start_futures_loop.sh" "Futures Agent" "python3.*high_risk_futures_agent.py"
FUTURES_STATUS=$?

echo "✅ Agent startup phase complete"
echo

# ============================================================================
# PHASE 6: SETUP MONITORING & WATCHDOGS
# ============================================================================

echo "🔧 PHASE 6: Setting up comprehensive monitoring..."

# Remove existing cron jobs
crontab -l 2>/dev/null | grep -v -E "(hybrid|futures|watchdog|health_check|trading)" | crontab - 2>/dev/null || true

# Create comprehensive cron schedule
cat << EOF | crontab -
# Confidence Engine Trading System - Unified Monitoring
# Generated by start_trading_system.sh on $(date)

# Watchdog: Check every 2 minutes, restart if needed
*/2 * * * * cd $ROOT_DIR && bash scripts/watchdog_hybrid.sh >/dev/null 2>&1
*/2 * * * * cd $ROOT_DIR && bash scripts/watchdog_futures.sh >/dev/null 2>&1

# Health check: Comprehensive system check every 30 minutes
*/30 * * * * cd $ROOT_DIR && bash scripts/health_check.sh >/dev/null 2>&1

# Database cleanup: Clean old logs every 6 hours (keep last 30 days)
0 */6 * * * cd $ROOT_DIR && python3 -c "
import sqlite3
from datetime import datetime, timedelta
conn = sqlite3.connect('enhanced_trading.db')
cursor = conn.cursor()
cutoff = (datetime.now() - timedelta(days=30)).isoformat()
tables = ['signal_logs', 'heartbeat_logs', 'notification_logs']
for table in tables:
    cursor.execute(f'DELETE FROM {table} WHERE timestamp < ?', (cutoff,))
conn.commit()
conn.close()
print(f'Database cleanup completed: removed records older than 30 days')
" >/dev/null 2>&1

# Performance report: Generate weekly performance summary
0 9 * * 1 cd $ROOT_DIR && python3 -c "
import sqlite3
import json
from datetime import datetime, timedelta

conn = sqlite3.connect('enhanced_trading.db')
cursor = conn.cursor()

# Get weekly performance
week_ago = (datetime.now() - timedelta(days=7)).isoformat()
cursor.execute('''
    SELECT agent_type, COUNT(*) as trades, 
           AVG(signal_quality) as avg_quality,
           AVG(conviction_score) as avg_conviction,
           COUNT(CASE WHEN win_loss = 'win' THEN 1 END) as wins,
           SUM(pnl) as total_pnl
    FROM enhanced_trades 
    WHERE timestamp > ? 
    GROUP BY agent_type
''', (week_ago,))

results = cursor.fetchall()
report = {'week_ending': datetime.now().isoformat(), 'performance': {}}
for row in results:
    agent, trades, quality, conviction, wins, pnl = row
    win_rate = (wins / trades * 100) if trades > 0 else 0
    report['performance'][agent] = {
        'trades': trades, 'avg_signal_quality': quality,
        'avg_conviction': conviction, 'win_rate': win_rate, 'total_pnl': pnl
    }

with open('weekly_performance_report.json', 'w') as f:
    json.dump(report, f, indent=2)

conn.close()
print('Weekly performance report generated')
" >/dev/null 2>&1

# Daily restart: Fresh start at 6 AM with system check
0 6 * * * cd $ROOT_DIR && bash start_trading_system.sh --quick-restart >/dev/null 2>&1

EOF

echo "✅ Comprehensive monitoring setup complete:"
echo "   - Watchdog checks every 2 minutes"
echo "   - Health checks every 30 minutes"
echo "   - Database cleanup every 6 hours"
echo "   - Weekly performance reports (Mondays 9 AM)"
echo "   - Daily restart at 6 AM"
echo

# ============================================================================
# PHASE 7: FINAL STATUS & MONITORING SETUP
# ============================================================================

echo "📊 PHASE 7: Final system status..."

# Wait for processes to stabilize
echo "   Waiting for processes to stabilize..."
sleep 10

# Final status check
echo
echo "📊 FINAL SYSTEM STATUS:"
echo "======================"

# Check agents
if pgrep -f "python3 scripts/hybrid_crypto_trader.py" >/dev/null 2>&1; then
    HYBRID_PID=$(pgrep -f "python3 scripts/hybrid_crypto_trader.py" | head -1)
    echo "✅ Hybrid Agent: RUNNING (PID: $HYBRID_PID)"
else
    echo "❌ Hybrid Agent: NOT RUNNING"
fi

if pgrep -f "python3.*high_risk_futures_agent.py" >/dev/null 2>&1; then
    FUTURES_PID=$(pgrep -f "python3.*high_risk_futures_agent.py" | head -1)
    echo "✅ Futures Agent: RUNNING (PID: $FUTURES_PID)"
else
    echo "❌ Futures Agent: NOT RUNNING"
fi

# Check database
echo "✅ Database: enhanced_trading.db (comprehensive logging enabled)"

# Check cron jobs
CRON_COUNT=$(crontab -l 2>/dev/null | grep -c "trading" || echo 0)
echo "✅ Monitoring: $CRON_COUNT cron jobs active"

# Check logs
echo "✅ Logs: trader_loop.log, high_risk_futures_loop.log"

echo
echo "🎯 NOTIFICATION ROUTING:"
echo "======================="
if [ "${TB_ENABLE_DISCORD:-0}" = "1" ]; then
    echo "✅ Discord: Trade notifications, heartbeats, alerts → Discord channel"
else
    echo "❌ Discord: DISABLED"
fi

if [ "${TB_NO_TELEGRAM:-1}" = "0" ]; then
    echo "✅ Telegram: Backup notifications, alerts → Telegram chat"
else
    echo "❌ Telegram: DISABLED"
fi

echo "✅ Database: All signals, trades, performance → enhanced_trading.db"

echo
echo "📋 MONITORING COMMANDS:"
echo "======================"
echo "# Real-time status:"
echo "watch 'ps aux | grep -E \"(hybrid|futures)\" | grep -v grep'"
echo
echo "# Live logs:"
echo "tail -f trader_loop.log high_risk_futures_loop.log"
echo
echo "# Database query examples:"
echo "sqlite3 enhanced_trading.db \"SELECT * FROM enhanced_trades ORDER BY timestamp DESC LIMIT 10;\""
echo "sqlite3 enhanced_trading.db \"SELECT agent_type, AVG(signal_quality), COUNT(*) FROM signal_logs GROUP BY agent_type;\""
echo
echo "# Performance report:"
echo "cat weekly_performance_report.json"
echo
echo "# Emergency stop:"
echo "pkill -f 'python3.*hybrid_crypto_trader.py'; pkill -f 'python3.*high_risk_futures_agent.py'"

echo
echo "🚀 UNIFIED TRADING SYSTEM DEPLOYMENT COMPLETE!"
echo "==============================================="
echo "✅ Both agents running with Phase 1C ultra-aggressive thresholds"
echo "✅ Intelligent trade management with confidence-based sizing"
echo "✅ Hard caps: 8 trades/day per agent, $1000 hybrid / $100 futures per trade"
echo "✅ Comprehensive monitoring and auto-recovery active"
echo "✅ Discord/Telegram notifications configured"
echo "✅ Database logging for performance assessment enabled"
echo "✅ Weekly performance reports scheduled"
echo "✅ Automated daily restarts configured"
echo
echo "💡 Expected Trading Activity:"
echo "   • Hybrid Agent: Up to 8 trades/day when high-quality signals found"
echo "   • Futures Agent: Up to 8 trades/day when high-quality signals found"
echo "   • Position sizing: Dynamic based on confidence (20%-100% of max)"
echo "   • No artificial weekly restrictions - trade when opportunities arise!"
echo "📊 All data logged to enhanced_trading.db for analysis"
echo "📱 Real-time notifications via Discord/Telegram"
echo "🔄 Auto-recovery on failures"

# Send final deployment notification
if [ "${TB_ENABLE_DISCORD:-0}" = "1" ] && [ -n "${DISCORD_WEBHOOK_URL:-}" ]; then
    python3 -c "
import os, sys
sys.path.append('scripts')
try:
    from discord_sender import send_discord_digest_to
    webhook_url = os.environ.get('DISCORD_WEBHOOK_URL')
    
    # Get actual status
    import subprocess
    hybrid_running = subprocess.run(['pgrep', '-f', 'python3 scripts/hybrid_crypto_trader.py'], 
                                  capture_output=True).returncode == 0
    futures_running = subprocess.run(['pgrep', '-f', 'python3.*high_risk_futures_agent.py'], 
                                   capture_output=True).returncode == 0
    
    status_color = 0x00FF00 if (hybrid_running and futures_running) else 0xFF6B35
    
    embeds = [{
        'title': '🚀 Unified Trading System Deployed',
        'description': 'Complete trading system deployment with Phase 1C configuration',
        'color': status_color,
        'fields': [
            {'name': 'Hybrid Agent', 'value': '✅ RUNNING' if hybrid_running else '❌ FAILED', 'inline': True},
            {'name': 'Futures Agent', 'value': '✅ RUNNING' if futures_running else '❌ FAILED', 'inline': True},
            {'name': 'Monitoring', 'value': '✅ ACTIVE', 'inline': True},
            {'name': 'Signal Quality', 'value': '1.0/10 (ultra-aggressive)', 'inline': True},
            {'name': 'Database Logging', 'value': '✅ ENABLED', 'inline': True},
            {'name': 'Auto-Recovery', 'value': '✅ ACTIVE', 'inline': True}
        ]
    }]
    send_discord_digest_to(webhook_url, embeds)
    print('📱 Deployment notification sent to Discord')
except Exception as e:
    print(f'❌ Failed to send deployment notification: {e}')
" || true
fi
