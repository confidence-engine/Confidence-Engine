#!/usr/bin/env python3
"""
Enhanced Trade Management System with Dynamic Limits
Implements intelligent trade limits based on confidence and risk management
"""

import sqlite3
import json
import os
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple

class TradeManager:
    """Manages trade limits and risk controls for both agents"""
    
    def __init__(self, db_path: str = "enhanced_trading.db"):
        self.db_path = db_path
        
        # Hard caps per agent (prevent overtrading)
        self.max_trades_per_agent_daily = 8
        self.max_trades_per_agent_weekly = 40  # 8 trades * 5 days (allowing for weekends)
        
        # Position sizing limits
        self.hybrid_max_trade_size = 1000.0  # $1000 per trade
        self.futures_max_trade_size = 100.0  # $100 per trade (25x leverage = $2500 exposure)
        
        # Confidence-based multipliers
        self.confidence_multipliers = {
            9.0: 1.0,    # Max confidence = full size
            8.0: 0.9,    # High confidence = 90% size
            7.0: 0.8,    # Good confidence = 80% size
            6.0: 0.7,    # Decent confidence = 70% size
            5.0: 0.6,    # Moderate confidence = 60% size
            4.0: 0.5,    # Low confidence = 50% size
            3.0: 0.4,    # Very low confidence = 40% size
            2.0: 0.3,    # Minimum confidence = 30% size
            1.0: 0.2,    # Ultra-low confidence = 20% size
        }
        
        self.init_database()
    
    def init_database(self):
        """Initialize trade management tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Trade limits tracking
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
        
        # Risk management events
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
    
    def get_confidence_multiplier(self, signal_quality: float, conviction_score: float) -> float:
        """Calculate confidence multiplier based on signal quality and conviction"""
        # Use the higher of signal quality or conviction score
        confidence = max(signal_quality, conviction_score)
        
        # Find the appropriate multiplier
        for threshold in sorted(self.confidence_multipliers.keys(), reverse=True):
            if confidence >= threshold:
                return self.confidence_multipliers[threshold]
        
        # If below minimum threshold, use minimum multiplier
        return self.confidence_multipliers[1.0]
    
    def calculate_position_size(self, agent_type: str, signal_quality: float, 
                              conviction_score: float, symbol: str) -> float:
        """Calculate position size based on agent type and confidence"""
        
        # Get base trade size
        if agent_type == "hybrid_agent":
            base_size = self.hybrid_max_trade_size
        elif agent_type == "futures_agent":
            base_size = self.futures_max_trade_size
        else:
            base_size = 100.0  # Default fallback
        
        # Apply confidence multiplier
        confidence_multiplier = self.get_confidence_multiplier(signal_quality, conviction_score)
        position_size = base_size * confidence_multiplier
        
        # Log sizing decision
        self.log_risk_event(
            agent_type=agent_type,
            event_type="position_sizing",
            description=f"Symbol: {symbol}, Base: ${base_size}, Confidence: {max(signal_quality, conviction_score):.1f}, Multiplier: {confidence_multiplier:.1f}, Final: ${position_size:.2f}",
            current_exposure=position_size
        )
        
        return position_size
    
    def check_trade_limits(self, agent_type: str, trade_size: float) -> Tuple[bool, str]:
        """Check if trade is within limits"""
        
        now = datetime.now()
        today = now.date()
        week_start = today - timedelta(days=today.weekday())
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Check daily trade count
        cursor.execute('''
        SELECT COUNT(*) FROM enhanced_trades 
        WHERE agent_type = ? AND DATE(timestamp) = ?
        ''', (agent_type, today.isoformat()))
        
        daily_trades = cursor.fetchone()[0]
        
        # Check weekly trade count
        cursor.execute('''
        SELECT COUNT(*) FROM enhanced_trades 
        WHERE agent_type = ? AND DATE(timestamp) >= ?
        ''', (agent_type, week_start.isoformat()))
        
        weekly_trades = cursor.fetchone()[0]
        
        # Check daily exposure
        cursor.execute('''
        SELECT COALESCE(SUM(ABS(quantity * entry_price)), 0) FROM enhanced_trades 
        WHERE agent_type = ? AND DATE(timestamp) = ? AND exit_timestamp IS NULL
        ''', (agent_type, today.isoformat()))
        
        daily_exposure = cursor.fetchone()[0] or 0.0
        
        conn.close()
        
        # Apply limits
        if daily_trades >= self.max_trades_per_agent_daily:
            reason = f"Daily trade limit exceeded: {daily_trades}/{self.max_trades_per_agent_daily}"
            self.log_risk_event(agent_type, "limit_exceeded", reason, trade_blocked=True)
            return False, reason
        
        if weekly_trades >= self.max_trades_per_agent_weekly:
            reason = f"Weekly trade limit exceeded: {weekly_trades}/{self.max_trades_per_agent_weekly}"
            self.log_risk_event(agent_type, "limit_exceeded", reason, trade_blocked=True)
            return False, reason
        
        # Calculate max daily exposure (10x daily trade size limit)
        max_daily_exposure = (self.hybrid_max_trade_size if agent_type == "hybrid_agent" 
                             else self.futures_max_trade_size) * 10
        
        if daily_exposure + trade_size > max_daily_exposure:
            reason = f"Daily exposure limit would be exceeded: ${daily_exposure + trade_size:.2f} > ${max_daily_exposure:.2f}"
            self.log_risk_event(agent_type, "limit_exceeded", reason, trade_blocked=True, 
                              current_exposure=daily_exposure)
            return False, reason
        
        # Trade approved
        self.log_risk_event(
            agent_type=agent_type,
            event_type="trade_approved",
            description=f"Trade size: ${trade_size:.2f}, Daily trades: {daily_trades}/{self.max_trades_per_agent_daily}, Weekly: {weekly_trades}/{self.max_trades_per_agent_weekly}",
            current_exposure=daily_exposure + trade_size
        )
        
        return True, "Trade approved"
    
    def log_risk_event(self, agent_type: str, event_type: str, description: str, 
                      trade_blocked: bool = False, current_exposure: float = 0.0, 
                      limit_exceeded: str = ""):
        """Log risk management event"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        INSERT INTO risk_events (
            timestamp, agent_type, event_type, description, trade_blocked, 
            current_exposure, limit_exceeded
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            agent_type,
            event_type,
            description,
            trade_blocked,
            current_exposure,
            limit_exceeded
        ))
        
        conn.commit()
        conn.close()
    
    def get_trade_limits_status(self, agent_type: str) -> Dict[str, Any]:
        """Get current trade limits status for an agent"""
        
        now = datetime.now()
        today = now.date()
        week_start = today - timedelta(days=today.weekday())
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Daily stats
        cursor.execute('''
        SELECT COUNT(*), COALESCE(SUM(ABS(quantity * entry_price)), 0) 
        FROM enhanced_trades 
        WHERE agent_type = ? AND DATE(timestamp) = ?
        ''', (agent_type, today.isoformat()))
        
        daily_trades, daily_exposure = cursor.fetchone()
        daily_exposure = daily_exposure or 0.0
        
        # Weekly stats
        cursor.execute('''
        SELECT COUNT(*) FROM enhanced_trades 
        WHERE agent_type = ? AND DATE(timestamp) >= ?
        ''', (agent_type, week_start.isoformat()))
        
        weekly_trades = cursor.fetchone()[0]
        
        # Active positions
        cursor.execute('''
        SELECT COUNT(*), COALESCE(SUM(ABS(quantity * entry_price)), 0) 
        FROM enhanced_trades 
        WHERE agent_type = ? AND exit_timestamp IS NULL
        ''', (agent_type,))
        
        active_positions, active_exposure = cursor.fetchone()
        active_exposure = active_exposure or 0.0
        
        conn.close()
        
        max_daily_exposure = (self.hybrid_max_trade_size if agent_type == "hybrid_agent" 
                             else self.futures_max_trade_size) * 10
        
        return {
            'agent_type': agent_type,
            'daily_trades': daily_trades,
            'daily_limit': self.max_trades_per_agent_daily,
            'daily_remaining': max(0, self.max_trades_per_agent_daily - daily_trades),
            'weekly_trades': weekly_trades,
            'weekly_limit': self.max_trades_per_agent_weekly,
            'weekly_remaining': max(0, self.max_trades_per_agent_weekly - weekly_trades),
            'daily_exposure': daily_exposure,
            'max_daily_exposure': max_daily_exposure,
            'exposure_remaining': max(0, max_daily_exposure - daily_exposure),
            'active_positions': active_positions,
            'active_exposure': active_exposure,
            'max_trade_size': self.hybrid_max_trade_size if agent_type == "hybrid_agent" else self.futures_max_trade_size
        }

# Convenience functions
_trade_manager = None

def get_trade_manager() -> TradeManager:
    """Get singleton trade manager instance"""
    global _trade_manager
    if _trade_manager is None:
        _trade_manager = TradeManager()
    return _trade_manager

def calculate_position_size(agent_type: str, signal_quality: float, conviction_score: float, symbol: str) -> float:
    """Calculate appropriate position size"""
    return get_trade_manager().calculate_position_size(agent_type, signal_quality, conviction_score, symbol)

def check_trade_limits(agent_type: str, trade_size: float) -> Tuple[bool, str]:
    """Check if trade is within limits"""
    return get_trade_manager().check_trade_limits(agent_type, trade_size)

def get_trade_status(agent_type: str) -> Dict[str, Any]:
    """Get current trade limits status"""
    return get_trade_manager().get_trade_limits_status(agent_type)

if __name__ == "__main__":
    # Test the trade management system
    tm = TradeManager()
    
    print("🎯 ENHANCED TRADE MANAGEMENT SYSTEM")
    print("=" * 40)
    
    # Test position sizing
    test_cases = [
        ("hybrid_agent", 9.5, 8.7, "BTC/USD"),
        ("hybrid_agent", 6.2, 7.1, "ETH/USD"),
        ("futures_agent", 8.9, 9.2, "BTCUSDT"),
        ("futures_agent", 3.5, 4.1, "ETHUSDT"),
    ]
    
    print("\n📊 POSITION SIZING TESTS:")
    for agent, signal_q, conviction, symbol in test_cases:
        size = tm.calculate_position_size(agent, signal_q, conviction, symbol)
        confidence = max(signal_q, conviction)
        multiplier = tm.get_confidence_multiplier(signal_q, conviction)
        base = tm.hybrid_max_trade_size if agent == "hybrid_agent" else tm.futures_max_trade_size
        
        print(f"  {agent} - {symbol}:")
        print(f"    Confidence: {confidence:.1f}/10 → Multiplier: {multiplier:.1f}")
        print(f"    Base Size: ${base:.0f} → Final Size: ${size:.2f}")
        print()
    
    # Test trade limits
    print("🔒 TRADE LIMITS STATUS:")
    for agent in ["hybrid_agent", "futures_agent"]:
        status = tm.get_trade_limits_status(agent)
        print(f"  {agent}:")
        print(f"    Daily: {status['daily_trades']}/{status['daily_limit']} ({status['daily_remaining']} remaining)")
        print(f"    Weekly: {status['weekly_trades']}/{status['weekly_limit']} ({status['weekly_remaining']} remaining)")
        print(f"    Max Trade Size: ${status['max_trade_size']:.0f}")
        print(f"    Daily Exposure: ${status['daily_exposure']:.2f}/${status['max_daily_exposure']:.0f}")
        print()
    
    print("✅ Trade management system ready!")
