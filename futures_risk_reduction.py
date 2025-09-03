#!/usr/bin/env python3
"""
Futures Risk Reduction Configuration
Implements immediate position size reductions for improved win rate
"""

import os
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

def apply_risk_reduction_settings():
    """Apply immediate risk reduction settings to improve win rate"""
    
    # 🎯 IMMEDIATE FIXES - Reduce position sizes by 50%
    original_risk = float(os.getenv("FUTURES_RISK_PER_TRADE", "0.05"))
    reduced_risk = original_risk * 0.5  # 50% reduction
    
    # Update environment variable
    os.environ["FUTURES_RISK_PER_TRADE"] = str(reduced_risk)
    
    logger.info(f"🔧 RISK REDUCTION APPLIED:")
    logger.info(f"   Original risk per trade: {original_risk:.1%}")
    logger.info(f"   Reduced risk per trade: {reduced_risk:.1%}")
    logger.info(f"   Reduction: 50% to improve win rate")
    
    # 🚨 Additional conservative settings
    conservative_settings = {
        "FUTURES_MAX_LEVERAGE": "15",  # Reduce from 25x to 15x
        "FUTURES_MAX_POSITIONS": "3",   # Reduce from 5 to 3
        "FUTURES_MAX_TRADES_PER_CYCLE": "2",  # Reduce from 3 to 2
        "TB_MIN_SIGNAL_QUALITY": "3.0",  # Increase from 1.0 to 3.0
        "TB_MIN_CONVICTION_SCORE": "4.0",  # Increase from 2.0 to 4.0
    }
    
    for setting, value in conservative_settings.items():
        original_value = os.getenv(setting, "not_set")
        os.environ[setting] = value
        logger.info(f"   {setting}: {original_value} → {value}")
    
    logger.info("✅ Conservative settings applied for win rate improvement")
    
    return {
        "risk_reduction": reduced_risk,
        "original_risk": original_risk,
        "settings_applied": conservative_settings,
        "timestamp": datetime.now().isoformat()
    }

def get_kelly_recommendations(win_count: int, loss_count: int, 
                             total_wins: float, total_losses: float) -> dict:
    """Get Kelly Criterion recommendations based on current performance"""
    
    if win_count + loss_count < 5:
        return {
            "recommendation": "insufficient_data",
            "suggested_risk": 0.01,  # 1% ultra-conservative
            "reason": "Not enough trades for Kelly calculation"
        }
    
    total_trades = win_count + loss_count
    win_rate = win_count / total_trades
    
    if loss_count > 0 and total_losses != 0:
        avg_win = total_wins / max(win_count, 1)
        avg_loss = abs(total_losses) / loss_count
        win_loss_ratio = avg_win / avg_loss
    else:
        win_loss_ratio = 1.0
    
    # Kelly formula: f = (bp - q) / b
    p = win_rate
    q = 1 - win_rate
    b = win_loss_ratio
    
    if b > 0:
        kelly_fraction = (b * p - q) / b
    else:
        kelly_fraction = 0
    
    # Apply conservative multiplier (25% of Kelly)
    safe_kelly = max(0, min(0.05, kelly_fraction * 0.25))  # Cap at 5%
    
    # Performance-based adjustments
    if win_rate < 0.20:  # Win rate < 20%
        recommendation = "ultra_conservative"
        suggested_risk = 0.005  # 0.5%
        reason = f"Very low win rate ({win_rate:.1%}), use minimal risk"
    elif win_rate < 0.35:  # Win rate < 35%
        recommendation = "conservative"
        suggested_risk = min(0.015, safe_kelly)  # Max 1.5%
        reason = f"Low win rate ({win_rate:.1%}), use conservative sizing"
    elif win_rate < 0.50:  # Win rate < 50%
        recommendation = "moderate"
        suggested_risk = min(0.025, safe_kelly)  # Max 2.5%
        reason = f"Below average win rate ({win_rate:.1%}), moderate sizing"
    else:
        recommendation = "normal"
        suggested_risk = min(0.05, safe_kelly)  # Max 5%
        reason = f"Good win rate ({win_rate:.1%}), normal Kelly sizing"
    
    return {
        "recommendation": recommendation,
        "suggested_risk": suggested_risk,
        "kelly_fraction": kelly_fraction,
        "safe_kelly": safe_kelly,
        "win_rate": win_rate,
        "win_loss_ratio": win_loss_ratio,
        "reason": reason,
        "total_trades": total_trades
    }

def monitor_performance_and_adjust():
    """Monitor current performance and suggest adjustments"""
    
    try:
        # Try to read current performance from futures agent
        # This would be called by the agent to get dynamic recommendations
        
        logger.info("📊 PERFORMANCE MONITORING & RISK ADJUSTMENT")
        logger.info("=" * 50)
        
        # Check if we have performance data file
        performance_file = "futures_performance.json"
        if os.path.exists(performance_file):
            import json
            with open(performance_file, 'r') as f:
                perf_data = json.load(f)
            
            win_count = perf_data.get('win_count', 0)
            loss_count = perf_data.get('loss_count', 0)
            total_wins = perf_data.get('total_wins', 0.0)
            total_losses = perf_data.get('total_losses', 0.0)
            
            recommendations = get_kelly_recommendations(
                win_count, loss_count, total_wins, total_losses
            )
            
            logger.info(f"Current Performance: {win_count}W/{loss_count}L")
            logger.info(f"Recommendation: {recommendations['recommendation']}")
            logger.info(f"Suggested Risk: {recommendations['suggested_risk']:.1%}")
            logger.info(f"Reason: {recommendations['reason']}")
            
            return recommendations
        else:
            logger.info("No performance data available yet")
            return get_kelly_recommendations(0, 0, 0.0, 0.0)
            
    except Exception as e:
        logger.error(f"Error monitoring performance: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    # Apply immediate risk reduction
    print("🚨 Applying Immediate Risk Reduction Settings")
    print("=" * 50)
    
    settings = apply_risk_reduction_settings()
    
    print(f"✅ Risk reduced from {settings['original_risk']:.1%} to {settings['risk_reduction']:.1%}")
    print("✅ Conservative settings applied")
    print("\n📊 Monitoring recommendations:")
    
    recommendations = monitor_performance_and_adjust()
    print(f"   Suggested risk level: {recommendations.get('suggested_risk', 0.01):.1%}")
    print(f"   Reasoning: {recommendations.get('reason', 'Ultra-conservative until performance improves')}")
