#!/usr/bin/env python3
"""
Enhanced Database Logger for Trading System
Comprehensive logging of all signals, trades, heartbeats, and notifications
Used by both agents to ensure complete performance tracking
"""

import sqlite3
import json
import os
from datetime import datetime
from typing import Dict, Any, Optional

class TradingSystemLogger:
    def __init__(self, db_path: str = "enhanced_trading.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database with all required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Enhanced signal logs
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
            rsi REAL,
            raw_data TEXT
        )
        ''')
        
        # Performance metrics
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
        
        # Heartbeat logs with enhanced fields
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
            platform TEXT,
            last_signal_quality REAL,
            last_conviction_score REAL,
            cpu_usage REAL,
            memory_usage REAL
        )
        ''')
        
        # Notification logs
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS notification_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            notification_type TEXT NOT NULL,
            channel TEXT NOT NULL,
            agent_type TEXT,
            symbol TEXT,
            message_content TEXT,
            delivery_status TEXT,
            retry_count INTEGER DEFAULT 0
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
            hold_duration_minutes INTEGER,
            entry_reason TEXT,
            exit_reason TEXT
        )
        ''')
        
        conn.commit()
        conn.close()
    
    def log_signal(self, agent_type: str, symbol: str, signal_data: Dict[str, Any]):
        """Log trading signal for performance analysis"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        INSERT INTO signal_logs (
            timestamp, agent_type, symbol, signal_quality, conviction_score,
            regime_trend, regime_volatility, signal_decision, reason,
            sentiment, price_momentum, volume_z_score, rsi, raw_data
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            agent_type,
            symbol,
            signal_data.get('signal_quality'),
            signal_data.get('conviction_score'),
            signal_data.get('regime_trend'),
            signal_data.get('regime_volatility'),
            signal_data.get('signal_decision'),
            signal_data.get('reason'),
            signal_data.get('sentiment'),
            signal_data.get('price_momentum'),
            signal_data.get('volume_z_score'),
            signal_data.get('rsi'),
            json.dumps(signal_data, default=str)
        ))
        
        conn.commit()
        conn.close()
    
    def log_trade(self, agent_type: str, trade_data: Dict[str, Any]):
        """Log trade execution for P&L tracking"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        INSERT INTO enhanced_trades (
            timestamp, run_id, agent_type, symbol, action, entry_price,
            quantity, leverage, signal_quality, conviction_score, regime_state,
            platform, order_id, exit_price, exit_timestamp, pnl, win_loss,
            hold_duration_minutes, entry_reason, exit_reason
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            trade_data.get('run_id'),
            agent_type,
            trade_data.get('symbol'),
            trade_data.get('action'),
            trade_data.get('entry_price'),
            trade_data.get('quantity'),
            trade_data.get('leverage'),
            trade_data.get('signal_quality'),
            trade_data.get('conviction_score'),
            trade_data.get('regime_state'),
            trade_data.get('platform'),
            trade_data.get('order_id'),
            trade_data.get('exit_price'),
            trade_data.get('exit_timestamp'),
            trade_data.get('pnl'),
            trade_data.get('win_loss'),
            trade_data.get('hold_duration_minutes'),
            trade_data.get('entry_reason'),
            trade_data.get('exit_reason')
        ))
        
        conn.commit()
        conn.close()
    
    def log_heartbeat(self, agent_type: str, heartbeat_data: Dict[str, Any]):
        """Log heartbeat for system monitoring"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        INSERT INTO heartbeat_logs (
            timestamp, agent_type, run_count, active_positions, daily_pnl,
            total_trades, system_status, market_regime, platform,
            last_signal_quality, last_conviction_score, cpu_usage, memory_usage
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            agent_type,
            heartbeat_data.get('run_count'),
            heartbeat_data.get('active_positions'),
            heartbeat_data.get('daily_pnl'),
            heartbeat_data.get('total_trades'),
            heartbeat_data.get('system_status'),
            heartbeat_data.get('market_regime'),
            heartbeat_data.get('platform'),
            heartbeat_data.get('last_signal_quality'),
            heartbeat_data.get('last_conviction_score'),
            heartbeat_data.get('cpu_usage'),
            heartbeat_data.get('memory_usage')
        ))
        
        conn.commit()
        conn.close()
    
    def log_notification(self, notification_type: str, channel: str, 
                        agent_type: str, symbol: str, message: str, 
                        delivery_status: str = "sent"):
        """Log notification delivery for audit trail"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        INSERT INTO notification_logs (
            timestamp, notification_type, channel, agent_type, symbol,
            message_content, delivery_status
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            notification_type,
            channel,
            agent_type,
            symbol,
            message,
            delivery_status
        ))
        
        conn.commit()
        conn.close()
    
    def get_performance_summary(self, days: int = 7) -> Dict[str, Any]:
        """Get performance summary for specified period"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        from datetime import timedelta
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        
        # Get trade performance
        cursor.execute('''
        SELECT agent_type, COUNT(*) as trades, 
               AVG(signal_quality) as avg_quality,
               AVG(conviction_score) as avg_conviction,
               COUNT(CASE WHEN win_loss = 'win' THEN 1 END) as wins,
               SUM(pnl) as total_pnl,
               AVG(hold_duration_minutes) as avg_hold_time
        FROM enhanced_trades 
        WHERE timestamp > ? 
        GROUP BY agent_type
        ''', (cutoff,))
        
        trade_results = cursor.fetchall()
        
        # Get signal distribution
        cursor.execute('''
        SELECT agent_type, signal_decision, COUNT(*) as count
        FROM signal_logs 
        WHERE timestamp > ? 
        GROUP BY agent_type, signal_decision
        ''', (cutoff,))
        
        signal_results = cursor.fetchall()
        
        # Get heartbeat status
        cursor.execute('''
        SELECT agent_type, AVG(cpu_usage) as avg_cpu, 
               AVG(memory_usage) as avg_memory,
               COUNT(*) as heartbeat_count
        FROM heartbeat_logs 
        WHERE timestamp > ? 
        GROUP BY agent_type
        ''', (cutoff,))
        
        heartbeat_results = cursor.fetchall()
        
        conn.close()
        
        summary = {
            'period_days': days,
            'trade_performance': {},
            'signal_distribution': {},
            'system_health': {}
        }
        
        # Process trade performance
        for row in trade_results:
            agent, trades, quality, conviction, wins, pnl, hold_time = row
            win_rate = (wins / trades * 100) if trades > 0 else 0
            summary['trade_performance'][agent] = {
                'total_trades': trades,
                'win_rate': round(win_rate, 2),
                'avg_signal_quality': round(quality or 0, 2),
                'avg_conviction': round(conviction or 0, 2),
                'total_pnl': round(pnl or 0, 4),
                'avg_hold_minutes': round(hold_time or 0, 1)
            }
        
        # Process signal distribution
        for row in signal_results:
            agent, decision, count = row
            if agent not in summary['signal_distribution']:
                summary['signal_distribution'][agent] = {}
            summary['signal_distribution'][agent][decision] = count
        
        # Process system health
        for row in heartbeat_results:
            agent, cpu, memory, count = row
            summary['system_health'][agent] = {
                'avg_cpu_usage': round(cpu or 0, 2),
                'avg_memory_usage': round(memory or 0, 2),
                'heartbeat_count': count
            }
        
        return summary
    
    def cleanup_old_logs(self, days_to_keep: int = 30):
        """Clean up old log entries to maintain database size"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        from datetime import timedelta
        cutoff = (datetime.now() - timedelta(days=days_to_keep)).isoformat()
        
        tables = ['signal_logs', 'heartbeat_logs', 'notification_logs']
        total_deleted = 0
        
        for table in tables:
            cursor.execute(f'DELETE FROM {table} WHERE timestamp < ?', (cutoff,))
            deleted = cursor.rowcount
            total_deleted += deleted
            print(f"Cleaned {deleted} old records from {table}")
        
        conn.commit()
        conn.close()
        
        return total_deleted

# Convenience functions for easy import
def log_signal(agent_type: str, symbol: str, signal_data: Dict[str, Any]):
    """Quick function to log trading signals"""
    logger = TradingSystemLogger()
    logger.log_signal(agent_type, symbol, signal_data)

def log_trade(agent_type: str, trade_data: Dict[str, Any]):
    """Quick function to log trades"""
    logger = TradingSystemLogger()
    logger.log_trade(agent_type, trade_data)

def log_heartbeat(agent_type: str, heartbeat_data: Dict[str, Any]):
    """Quick function to log heartbeats"""
    logger = TradingSystemLogger()
    logger.log_heartbeat(agent_type, heartbeat_data)

def log_notification(notification_type: str, channel: str, agent_type: str, 
                    symbol: str, message: str, delivery_status: str = "sent"):
    """Quick function to log notifications"""
    logger = TradingSystemLogger()
    logger.log_notification(notification_type, channel, agent_type, symbol, message, delivery_status)

def get_performance_report(days: int = 7) -> str:
    """Get formatted performance report"""
    logger = TradingSystemLogger()
    summary = logger.get_performance_summary(days)
    
    report = f"\n📊 TRADING SYSTEM PERFORMANCE REPORT ({days} days)\n"
    report += "=" * 50 + "\n\n"
    
    # Trade Performance
    report += "🎯 TRADE PERFORMANCE:\n"
    for agent, data in summary['trade_performance'].items():
        report += f"  {agent}:\n"
        report += f"    Trades: {data['total_trades']}\n"
        report += f"    Win Rate: {data['win_rate']}%\n"
        report += f"    Avg Signal Quality: {data['avg_signal_quality']}/10\n"
        report += f"    Avg Conviction: {data['avg_conviction']}/10\n"
        report += f"    Total P&L: {data['total_pnl']}\n"
        report += f"    Avg Hold Time: {data['avg_hold_minutes']} minutes\n\n"
    
    # Signal Distribution
    report += "📈 SIGNAL DISTRIBUTION:\n"
    for agent, decisions in summary['signal_distribution'].items():
        report += f"  {agent}: {decisions}\n"
    
    # System Health
    report += "\n💻 SYSTEM HEALTH:\n"
    for agent, health in summary['system_health'].items():
        report += f"  {agent}:\n"
        report += f"    CPU Usage: {health['avg_cpu_usage']}%\n"
        report += f"    Memory Usage: {health['avg_memory_usage']}%\n"
        report += f"    Heartbeats: {health['heartbeat_count']}\n\n"
    
    return report

if __name__ == "__main__":
    # Test the database logger
    logger = TradingSystemLogger()
    print("✅ Database logger initialized")
    print(f"📊 Database: {os.path.abspath(logger.db_path)}")
    
    # Generate test performance report
    report = get_performance_report(7)
    print(report)
