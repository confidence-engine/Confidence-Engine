#!/usr/bin/env python3
"""Database initialization script for the trading system."""

import sqlite3
import os
from datetime import datetime

def init_database():
    """Initialize enhanced_trading.db with comprehensive tracking."""
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
        severity TEXT,
        action_taken TEXT,
        resolved_at TEXT
    )
    ''')

    conn.commit()
    conn.close()

    # Initialize enhanced trade manager if available
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
        print('⚠️  Enhanced trade manager not available, using defaults')
    except Exception as e:
        print(f'⚠️  Trade manager test failed: {e}')

    print("✅ Database initialized successfully")
    return True

if __name__ == "__main__":
    init_database()
