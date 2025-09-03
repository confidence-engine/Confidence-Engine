#!/usr/bin/env python3
"""Database cleanup script - removes old logs to maintain performance."""

import sqlite3
from datetime import datetime, timedelta

def cleanup_database():
    """Clean old records from database tables."""
    conn = sqlite3.connect('enhanced_trading.db')
    cursor = conn.cursor()
    
    cutoff = (datetime.now() - timedelta(days=30)).isoformat()
    tables = ['signal_logs', 'heartbeat_logs', 'notification_logs']
    
    total_deleted = 0
    for table in tables:
        cursor.execute(f'SELECT COUNT(*) FROM {table} WHERE timestamp < ?', (cutoff,))
        count_before = cursor.fetchone()[0]
        
        cursor.execute(f'DELETE FROM {table} WHERE timestamp < ?', (cutoff,))
        deleted = cursor.rowcount
        total_deleted += deleted
        
        if deleted > 0:
            print(f'Cleaned {deleted} old records from {table}')
    
    conn.commit()
    conn.close()
    
    print(f'Database cleanup completed: removed {total_deleted} records older than 30 days')
    return total_deleted

if __name__ == "__main__":
    cleanup_database()
