#!/usr/bin/env python3
"""Weekly performance report generator."""

import sqlite3
import json
from datetime import datetime, timedelta

def generate_performance_report():
    """Generate weekly performance summary."""
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
            'trades': trades, 
            'avg_signal_quality': quality,
            'avg_conviction': conviction, 
            'win_rate': win_rate, 
            'total_pnl': pnl
        }

    with open('weekly_performance_report.json', 'w') as f:
        json.dump(report, f, indent=2)

    conn.close()
    print('Weekly performance report generated')
    return report

if __name__ == "__main__":
    generate_performance_report()
