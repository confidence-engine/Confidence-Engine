#!/usr/bin/env python3
"""
Enhanced Discord Trade Notifications
Includes signal quality, conviction scoring, and regime detection for both hybrid and futures agents
"""
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
from scripts.discord_sender import send_discord_digest

def create_enhanced_trade_notification(
    symbol: str,
    action: str,
    price: float,
    quantity: float,
    signal_quality: Optional[float] = None,
    conviction_score: Optional[float] = None,
    regime_state: Optional[Any] = None,
    reason: str = "",
    agent_type: str = "hybrid",
    take_profit: Optional[float] = None,
    stop_loss: Optional[float] = None,
    sentiment: Optional[float] = None,
    volatility: Optional[float] = None,
    **kwargs
) -> Dict:
    """
    Create enhanced Discord embed for trade notifications
    
    Args:
        symbol: Trading symbol (e.g., BTC/USD)
        action: BUY or SELL
        price: Entry/exit price
        quantity: Position size
        signal_quality: Quality score 0-10
        conviction_score: Conviction score 0-10
        regime_state: Market regime object
        reason: Trade reason/strategy
        agent_type: 'hybrid' or 'futures'
        take_profit: TP price
        stop_loss: SL price
        sentiment: Sentiment score
        volatility: Volatility metric
    """
    
    # Color coding based on action and quality
    color = 0x00ff00 if action == "BUY" else 0xff0000  # Green for BUY, Red for SELL
    
    # Enhanced color based on signal quality
    if signal_quality:
        if signal_quality >= 8.0:
            color = 0x00ff88 if action == "BUY" else 0xff4444  # Brighter for excellent signals
        elif signal_quality >= 6.0:
            color = 0x00cc66 if action == "BUY" else 0xcc4444  # Medium for good signals
        elif signal_quality < 4.0:
            color = 0x888888  # Gray for poor signals
    
    # Title with emoji and agent type
    agent_emoji = "🚀" if agent_type == "hybrid" else "⚡"
    action_emoji = "📈" if action == "BUY" else "📉"
    title = f"{agent_emoji} {agent_type.upper()} {action_emoji} {action} {symbol}"
    
    # Build description with key metrics
    description = f"**Price:** ${price:,.2f}\n**Quantity:** {quantity:.4f}\n"
    
    if signal_quality is not None:
        quality_emoji = "🌟" if signal_quality >= 8 else "✨" if signal_quality >= 6 else "⭐" if signal_quality >= 4 else "⚠️"
        description += f"**{quality_emoji} Signal Quality:** {signal_quality:.1f}/10\n"
    
    if conviction_score is not None:
        conv_emoji = "🎯" if conviction_score >= 8 else "🔥" if conviction_score >= 6 else "💡" if conviction_score >= 4 else "🤔"
        description += f"**{conv_emoji} Conviction:** {conviction_score:.1f}/10\n"
    
    if reason:
        description += f"**💭 Reason:** {reason}\n"
    
    # Market regime information
    if regime_state and hasattr(regime_state, 'trend_regime'):
        regime_emoji = {"bull": "🐂", "strong_bull": "🚀", "bear": "🐻", "strong_bear": "💥", "sideways": "➡️"}.get(regime_state.trend_regime, "📊")
        vol_emoji = {"low": "😴", "normal": "😊", "high": "😰", "extreme": "🔥"}.get(regime_state.volatility_regime, "📊")
        description += f"**{regime_emoji} Regime:** {regime_state.trend_regime.title()}/{regime_state.volatility_regime.title()} {vol_emoji}\n"
    
    # Create fields for additional details
    fields = []
    
    # TP/SL Information
    if take_profit or stop_loss:
        tp_sl_value = ""
        if take_profit:
            tp_pct = ((take_profit - price) / price * 100) if action == "BUY" else ((price - take_profit) / price * 100)
            tp_sl_value += f"🎯 TP: ${take_profit:.2f} ({tp_pct:+.1f}%)\n"
        if stop_loss:
            sl_pct = ((price - stop_loss) / price * 100) if action == "BUY" else ((stop_loss - price) / price * 100)
            tp_sl_value += f"🛡️ SL: ${stop_loss:.2f} ({sl_pct:+.1f}%)"
        
        fields.append({
            "name": "📊 Risk Management",
            "value": tp_sl_value,
            "inline": True
        })
    
    # Market Metrics
    if sentiment is not None or volatility is not None:
        metrics_value = ""
        if sentiment is not None:
            sent_emoji = "😍" if sentiment > 0.8 else "😊" if sentiment > 0.6 else "😐" if sentiment > 0.4 else "😞" if sentiment > 0.2 else "😭"
            metrics_value += f"{sent_emoji} Sentiment: {sentiment:.1%}\n"
        if volatility is not None:
            vol_emoji = "🌋" if volatility > 0.08 else "🌊" if volatility > 0.04 else "🏞️"
            metrics_value += f"{vol_emoji} Volatility: {volatility:.1%}"
        
        fields.append({
            "name": "📈 Market Metrics", 
            "value": metrics_value,
            "inline": True
        })
    
    # Signal Analysis (if available)
    if signal_quality is not None and conviction_score is not None:
        signal_analysis = ""
        if signal_quality >= 7 and conviction_score >= 7:
            signal_analysis = "🔥 **EXCELLENT** signal quality and conviction"
        elif signal_quality >= 5 and conviction_score >= 5:
            signal_analysis = "✨ **GOOD** signal quality and conviction"
        elif signal_quality >= 3 or conviction_score >= 3:
            signal_analysis = "⚠️ **MODERATE** signal - proceed with caution"
        else:
            signal_analysis = "🚨 **WEAK** signal - high risk trade"
        
        fields.append({
            "name": "🧠 Signal Analysis",
            "value": signal_analysis,
            "inline": False
        })
    
    embed = {
        "title": title,
        "description": description,
        "color": color,
        "fields": fields,
        "timestamp": datetime.utcnow().isoformat(),
        "footer": {
            "text": f"Confidence Engine • {agent_type.title()} Agent"
        }
    }
    
    return embed


def create_enhanced_heartbeat_notification(
    uptime_hours: float,
    total_trades: int,
    active_positions: int,
    current_pnl: float,
    system_health: str = "healthy",
    recent_signals: Optional[List[Dict]] = None
) -> Dict:
    """Create enhanced heartbeat notification with system status"""
    
    # Color based on system health and PnL
    if system_health == "critical":
        color = 0xff0000
    elif system_health == "warning":
        color = 0xffaa00
    elif current_pnl > 0:
        color = 0x00ff00
    else:
        color = 0x666666
    
    # Status emoji
    health_emoji = {"healthy": "💚", "warning": "⚠️", "critical": "🚨"}.get(system_health, "📊")
    
    description = f"**{health_emoji} Status:** {system_health.title()}\n"
    description += f"**⏱️ Uptime:** {uptime_hours:.1f}h\n"
    description += f"**📊 Trades:** {total_trades}\n"
    description += f"**🎯 Positions:** {active_positions}\n"
    
    pnl_emoji = "💰" if current_pnl > 0 else "📉" if current_pnl < 0 else "➖"
    description += f"**{pnl_emoji} P&L:** ${current_pnl:,.2f}\n"
    
    # Recent signals summary
    if recent_signals:
        avg_quality = sum(s.get('signal_quality', 0) for s in recent_signals) / len(recent_signals)
        quality_emoji = "🌟" if avg_quality >= 6 else "⭐" if avg_quality >= 4 else "⚠️"
        description += f"**{quality_emoji} Avg Signal Quality:** {avg_quality:.1f}/10"
    
    embed = {
        "title": "🤖 Confidence Engine Heartbeat",
        "description": description,
        "color": color,
        "timestamp": datetime.utcnow().isoformat(),
        "footer": {
            "text": "Confidence Engine • Enhanced Trading System"
        }
    }
    
    return embed


def send_enhanced_trade_notification(**kwargs):
    """Send enhanced trade notification to Discord"""
    enable_discord = os.getenv("TB_ENABLE_DISCORD", "0") == "1"
    if not enable_discord:
        print("[Discord] Enhanced trade notifications disabled")
        return False
    
    try:
        embed = create_enhanced_trade_notification(**kwargs)
        return send_discord_digest([embed])
    except Exception as e:
        print(f"[Discord] Error sending enhanced trade notification: {e}")
        return False


def send_enhanced_heartbeat(**kwargs):
    """Send enhanced heartbeat notification to Discord"""
    enable_discord = os.getenv("TB_ENABLE_DISCORD", "0") == "1"
    if not enable_discord:
        return False
        
    try:
        embed = create_enhanced_heartbeat_notification(**kwargs)
        return send_discord_digest([embed])
    except Exception as e:
        print(f"[Discord] Error sending enhanced heartbeat: {e}")
        return False


if __name__ == "__main__":
    # Test enhanced notifications
    print("🧪 Testing Enhanced Discord Notifications...")
    
    # Test trade notification
    test_trade = {
        "symbol": "BTC/USD",
        "action": "BUY", 
        "price": 42500.50,
        "quantity": 0.0235,
        "signal_quality": 7.8,
        "conviction_score": 8.2,
        "reason": "High-quality momentum signal in bull market",
        "agent_type": "hybrid",
        "take_profit": 44625.00,
        "stop_loss": 41225.00,
        "sentiment": 0.82,
        "volatility": 0.045
    }
    
    embed = create_enhanced_trade_notification(**test_trade)
    print("✅ Enhanced trade notification created successfully")
    print(f"   Title: {embed['title']}")
    print(f"   Fields: {len(embed['fields'])}")
    
    # Test heartbeat
    heartbeat = create_enhanced_heartbeat_notification(
        uptime_hours=24.5,
        total_trades=15,
        active_positions=3,
        current_pnl=1250.75,
        system_health="healthy",
        recent_signals=[{"signal_quality": 6.2}, {"signal_quality": 7.1}, {"signal_quality": 5.8}]
    )
    print("✅ Enhanced heartbeat notification created successfully")
