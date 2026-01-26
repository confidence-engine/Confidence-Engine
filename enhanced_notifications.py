#!/usr/bin/env python3
"""
Enhanced Notification System with Database Logging
Wraps Discord/Telegram notifications with comprehensive logging for performance tracking
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List

# Import existing notification functions
try:
    from telegram_bot import send_message as telegram_send_message
    telegram_available = True
except ImportError:
    telegram_available = False

try:
    from scripts.discord_sender import send_discord_digest_to
    discord_available = True
except ImportError:
    discord_available = False

# Import our database logger
try:
    from enhanced_db_logger import log_notification, log_heartbeat, TradingSystemLogger
    db_logging_available = True
except ImportError:
    db_logging_available = False

logger = logging.getLogger(__name__)

class EnhancedNotificationSystem:
    """Unified notification system with database logging"""
    
    def __init__(self):
        self.discord_webhook = os.getenv("DISCORD_WEBHOOK_URL", "")
        self.discord_trader_webhook = os.getenv("DISCORD_TRADER_WEBHOOK_URL", "")
        self.enable_discord = os.getenv("TB_ENABLE_DISCORD", "0") == "1"
        self.enable_telegram = os.getenv("TB_NO_TELEGRAM", "0") != "1"
        self.db_logger = TradingSystemLogger() if db_logging_available else None
        
        # Use trader-specific webhook if available, fallback to general
        self.active_discord_webhook = self.discord_trader_webhook or self.discord_webhook
        
        logger.info(f"📱 Notification system initialized:")
        logger.info(f"   Discord: {'✅' if self.enable_discord and self.active_discord_webhook else '❌'}")
        logger.info(f"   Telegram: {'✅' if self.enable_telegram and telegram_available else '❌'}")
        logger.info(f"   Database: {'✅' if db_logging_available else '❌'}")
    
    def send_trade_notification(self, agent_type: str, symbol: str, trade_data: Dict[str, Any]) -> Dict[str, bool]:
        """Send trade notification to all configured channels with logging"""
        
        # Format trade message
        action = trade_data.get('action', 'UNKNOWN')
        price = trade_data.get('entry_price', trade_data.get('price', 0))
        quantity = trade_data.get('quantity', 0)
        leverage = trade_data.get('leverage', 1)
        signal_quality = trade_data.get('signal_quality', 0)
        conviction = trade_data.get('conviction_score', 0)
        regime = trade_data.get('regime_state', 'unknown')
        
        # Create messages
        short_message = f"🎯 {action.upper()} {symbol} @ ${price:.4f} | Size: {quantity:.3f} | Leverage: {leverage}x | Quality: {signal_quality:.1f}/10 | Conviction: {conviction:.1f}/10"
        
        discord_embed = {
            'title': f'🎯 Trade Signal: {action.upper()} {symbol}',
            'description': f'Agent: {agent_type}',
            'color': 0x00FF00 if action.upper() == 'BUY' else 0xFF6B35 if action.upper() == 'SELL' else 0xFFD700,
            'fields': [
                {'name': 'Price', 'value': f'${price:.4f}', 'inline': True},
                {'name': 'Quantity', 'value': f'{quantity:.3f}', 'inline': True},
                {'name': 'Leverage', 'value': f'{leverage}x', 'inline': True},
                {'name': 'Signal Quality', 'value': f'{signal_quality:.1f}/10', 'inline': True},
                {'name': 'Conviction', 'value': f'{conviction:.1f}/10', 'inline': True},
                {'name': 'Market Regime', 'value': regime, 'inline': True},
            ],
            'timestamp': datetime.now().isoformat(),
            'footer': {'text': f'Confidence Engine • {agent_type}'}
        }
        
        # Add additional context if available
        if 'reason' in trade_data:
            discord_embed['fields'].append({'name': 'Reason', 'value': trade_data['reason'][:1000], 'inline': False})
        
        results = {}
        
        # Send Discord notification
        if self.enable_discord and self.active_discord_webhook and discord_available:
            try:
                success = send_discord_digest_to(self.active_discord_webhook, [discord_embed])
                results['discord'] = success
                
                # Log notification
                if self.db_logger:
                    self.db_logger.log_notification(
                        notification_type="trade",
                        channel="discord",
                        agent_type=agent_type,
                        symbol=symbol,
                        message=json.dumps(discord_embed, default=str),
                        delivery_status="success" if success else "failed"
                    )
                    
                logger.info(f"📱 Discord trade notification: {'✅ SUCCESS' if success else '❌ FAILED'}")
                
            except Exception as e:
                results['discord'] = False
                logger.error(f"❌ Discord trade notification failed: {e}")
                
                if self.db_logger:
                    self.db_logger.log_notification(
                        notification_type="trade",
                        channel="discord",
                        agent_type=agent_type,
                        symbol=symbol,
                        message=str(e),
                        delivery_status="error"
                    )
        else:
            results['discord'] = False
            logger.info("📱 Discord trade notification: DISABLED")
        
        # Send Telegram notification
        if self.enable_telegram and telegram_available:
            try:
                success = telegram_send_message(short_message)
                results['telegram'] = success
                
                # Log notification
                if self.db_logger:
                    self.db_logger.log_notification(
                        notification_type="trade",
                        channel="telegram",
                        agent_type=agent_type,
                        symbol=symbol,
                        message=short_message,
                        delivery_status="success" if success else "failed"
                    )
                    
                logger.info(f"📱 Telegram trade notification: {'✅ SUCCESS' if success else '❌ FAILED'}")
                
            except Exception as e:
                results['telegram'] = False
                logger.error(f"❌ Telegram trade notification failed: {e}")
                
                if self.db_logger:
                    self.db_logger.log_notification(
                        notification_type="trade",
                        channel="telegram",
                        agent_type=agent_type,
                        symbol=symbol,
                        message=str(e),
                        delivery_status="error"
                    )
        else:
            results['telegram'] = False
            logger.info("📱 Telegram trade notification: DISABLED")
        
        return results
    
    def send_heartbeat_notification(self, agent_type: str, heartbeat_data: Dict[str, Any]) -> Dict[str, bool]:
        """Send heartbeat notification with system status"""
        
        # Extract heartbeat info
        run_count = heartbeat_data.get('run_count', 0)
        active_positions = heartbeat_data.get('active_positions', 0)
        daily_pnl = heartbeat_data.get('daily_pnl', 0.0)
        total_trades = heartbeat_data.get('total_trades', 0)
        system_status = heartbeat_data.get('system_status', 'unknown')
        regime = heartbeat_data.get('market_regime', 'unknown')
        last_signal_quality = heartbeat_data.get('last_signal_quality', 0)
        
        # Log heartbeat to database first
        if self.db_logger:
            self.db_logger.log_heartbeat(agent_type, heartbeat_data)
        
        # Create messages
        short_message = f"💗 {agent_type} Heartbeat #{run_count} | Positions: {active_positions} | Daily P&L: ${daily_pnl:.2f} | Trades: {total_trades} | Status: {system_status}"
        
        discord_embed = {
            'title': f'💗 System Heartbeat: {agent_type}',
            'description': 'System health and performance update',
            'color': 0x00AA00,
            'fields': [
                {'name': 'Run Count', 'value': str(run_count), 'inline': True},
                {'name': 'Active Positions', 'value': str(active_positions), 'inline': True},
                {'name': 'Daily P&L', 'value': f'${daily_pnl:.2f}', 'inline': True},
                {'name': 'Total Trades', 'value': str(total_trades), 'inline': True},
                {'name': 'System Status', 'value': system_status, 'inline': True},
                {'name': 'Market Regime', 'value': regime, 'inline': True},
                {'name': 'Last Signal Quality', 'value': f'{last_signal_quality:.1f}/10', 'inline': True},
            ],
            'timestamp': datetime.now().isoformat(),
            'footer': {'text': f'Confidence Engine • Heartbeat Monitor'}
        }
        
        # Add CPU/Memory if available
        if 'cpu_usage' in heartbeat_data:
            discord_embed['fields'].append({
                'name': 'CPU Usage', 
                'value': f"{heartbeat_data['cpu_usage']:.1f}%", 
                'inline': True
            })
        
        if 'memory_usage' in heartbeat_data:
            discord_embed['fields'].append({
                'name': 'Memory Usage', 
                'value': f"{heartbeat_data['memory_usage']:.1f}%", 
                'inline': True
            })
        
        results = {}
        
        # Send Discord heartbeat (less frequently to avoid spam)
        if self.enable_discord and self.active_discord_webhook and discord_available:
            # Only send every 4th heartbeat to Discord to reduce noise
            if run_count % 4 == 0:
                try:
                    success = send_discord_digest_to(self.active_discord_webhook, [discord_embed])
                    results['discord'] = success
                    
                    # Log notification
                    if self.db_logger:
                        self.db_logger.log_notification(
                            notification_type="heartbeat",
                            channel="discord",
                            agent_type=agent_type,
                            symbol="",
                            message=json.dumps(discord_embed, default=str),
                            delivery_status="success" if success else "failed"
                        )
                        
                    logger.info(f"💗 Discord heartbeat #{run_count}: {'✅ SUCCESS' if success else '❌ FAILED'}")
                    
                except Exception as e:
                    results['discord'] = False
                    logger.error(f"❌ Discord heartbeat notification failed: {e}")
                    
                    if self.db_logger:
                        self.db_logger.log_notification(
                            notification_type="heartbeat",
                            channel="discord",
                            agent_type=agent_type,
                            symbol="",
                            message=str(e),
                            delivery_status="error"
                        )
            else:
                results['discord'] = True  # Skipped but not failed
                logger.debug(f"💗 Discord heartbeat #{run_count}: SKIPPED (every 4th only)")
        else:
            results['discord'] = False
        
        # Send Telegram heartbeat (even less frequently)
        if self.enable_telegram and telegram_available:
            # Only send every 12th heartbeat to Telegram
            if run_count % 12 == 0:
                try:
                    success = telegram_send_message(short_message)
                    results['telegram'] = success
                    
                    # Log notification
                    if self.db_logger:
                        self.db_logger.log_notification(
                            notification_type="heartbeat",
                            channel="telegram",
                            agent_type=agent_type,
                            symbol="",
                            message=short_message,
                            delivery_status="success" if success else "failed"
                        )
                        
                    logger.info(f"💗 Telegram heartbeat #{run_count}: {'✅ SUCCESS' if success else '❌ FAILED'}")
                    
                except Exception as e:
                    results['telegram'] = False
                    logger.error(f"❌ Telegram heartbeat notification failed: {e}")
                    
                    if self.db_logger:
                        self.db_logger.log_notification(
                            notification_type="heartbeat",
                            channel="telegram",
                            agent_type=agent_type,
                            symbol="",
                            message=str(e),
                            delivery_status="error"
                        )
            else:
                results['telegram'] = True  # Skipped but not failed
                logger.debug(f"💗 Telegram heartbeat #{run_count}: SKIPPED (every 12th only)")
        else:
            results['telegram'] = False
        
        return results
    
    def send_alert_notification(self, agent_type: str, alert_type: str, message: str, 
                              symbol: str = "", urgent: bool = False) -> Dict[str, bool]:
        """Send alert notification (always sent regardless of frequency)"""
        
        # Create messages
        alert_emoji = "🚨" if urgent else "⚠️"
        short_message = f"{alert_emoji} {agent_type} Alert: {alert_type}\n{message}"
        
        color = 0xFF0000 if urgent else 0xFFA500
        discord_embed = {
            'title': f'{alert_emoji} System Alert: {alert_type}',
            'description': message,
            'color': color,
            'fields': [
                {'name': 'Agent', 'value': agent_type, 'inline': True},
                {'name': 'Severity', 'value': 'URGENT' if urgent else 'WARNING', 'inline': True},
            ],
            'timestamp': datetime.now().isoformat(),
            'footer': {'text': f'Confidence Engine • Alert System'}
        }
        
        if symbol:
            discord_embed['fields'].append({'name': 'Symbol', 'value': symbol, 'inline': True})
        
        results = {}
        
        # Send Discord alert
        if self.enable_discord and self.active_discord_webhook and discord_available:
            try:
                success = send_discord_digest_to(self.active_discord_webhook, [discord_embed])
                results['discord'] = success
                
                # Log notification
                if self.db_logger:
                    self.db_logger.log_notification(
                        notification_type="alert",
                        channel="discord",
                        agent_type=agent_type,
                        symbol=symbol,
                        message=json.dumps(discord_embed, default=str),
                        delivery_status="success" if success else "failed"
                    )
                    
                logger.info(f"🚨 Discord alert notification: {'✅ SUCCESS' if success else '❌ FAILED'}")
                
            except Exception as e:
                results['discord'] = False
                logger.error(f"❌ Discord alert notification failed: {e}")
                
                if self.db_logger:
                    self.db_logger.log_notification(
                        notification_type="alert",
                        channel="discord",
                        agent_type=agent_type,
                        symbol=symbol,
                        message=str(e),
                        delivery_status="error"
                    )
        else:
            results['discord'] = False
        
        # Send Telegram alert
        if self.enable_telegram and telegram_available:
            try:
                success = telegram_send_message(short_message)
                results['telegram'] = success
                
                # Log notification
                if self.db_logger:
                    self.db_logger.log_notification(
                        notification_type="alert",
                        channel="telegram",
                        agent_type=agent_type,
                        symbol=symbol,
                        message=short_message,
                        delivery_status="success" if success else "failed"
                    )
                    
                logger.info(f"🚨 Telegram alert notification: {'✅ SUCCESS' if success else '❌ FAILED'}")
                
            except Exception as e:
                results['telegram'] = False
                logger.error(f"❌ Telegram alert notification failed: {e}")
                
                if self.db_logger:
                    self.db_logger.log_notification(
                        notification_type="alert",
                        channel="telegram",
                        agent_type=agent_type,
                        symbol=symbol,
                        message=str(e),
                        delivery_status="error"
                    )
        else:
            results['telegram'] = False
        
        return results

# Global notification system instance
_notification_system = None

def get_notification_system() -> EnhancedNotificationSystem:
    """Get singleton notification system instance"""
    global _notification_system
    if _notification_system is None:
        _notification_system = EnhancedNotificationSystem()
    return _notification_system

# Convenience functions for easy integration
def send_trade_notification(agent_type: str, symbol: str, trade_data: Dict[str, Any]) -> Dict[str, bool]:
    """Send trade notification to all configured channels"""
    return get_notification_system().send_trade_notification(agent_type, symbol, trade_data)

def send_heartbeat_notification(agent_type: str, heartbeat_data: Dict[str, Any]) -> Dict[str, bool]:
    """Send heartbeat notification"""
    return get_notification_system().send_heartbeat_notification(agent_type, heartbeat_data)

def send_alert_notification(agent_type: str, alert_type: str, message: str, 
                           symbol: str = "", urgent: bool = False) -> Dict[str, bool]:
    """Send alert notification"""
    return get_notification_system().send_alert_notification(agent_type, alert_type, message, symbol, urgent)

if __name__ == "__main__":
    # Test the notification system
    notification_system = EnhancedNotificationSystem()
    
    # Test trade notification
    test_trade_data = {
        'action': 'BUY',
        'entry_price': 43250.50,
        'quantity': 0.025,
        'leverage': 3,
        'signal_quality': 7.2,
        'conviction_score': 8.1,
        'regime_state': 'strong_bull_normal_vol',
        'reason': 'Strong divergence signal detected with high conviction score'
    }
    
    print("🧪 Testing trade notification...")
    trade_results = notification_system.send_trade_notification("test_agent", "BTC/USD", test_trade_data)
    print(f"Trade notification results: {trade_results}")
    
    # Test heartbeat notification
    test_heartbeat_data = {
        'run_count': 24,
        'active_positions': 2,
        'daily_pnl': 127.45,
        'total_trades': 8,
        'system_status': 'healthy',
        'market_regime': 'strong_bull_normal_vol',
        'last_signal_quality': 7.2,
        'cpu_usage': 23.5,
        'memory_usage': 34.2
    }
    
    print("\n💗 Testing heartbeat notification...")
    heartbeat_results = notification_system.send_heartbeat_notification("test_agent", test_heartbeat_data)
    print(f"Heartbeat notification results: {heartbeat_results}")
    
    # Test alert notification
    print("\n🚨 Testing alert notification...")
    alert_results = notification_system.send_alert_notification(
        "test_agent", 
        "Connection Error", 
        "Failed to connect to Alpaca API after 3 retries",
        symbol="BTC/USD",
        urgent=True
    )
    print(f"Alert notification results: {alert_results}")
    
    print("\n✅ Enhanced notification system test complete!")
