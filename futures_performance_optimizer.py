#!/usr/bin/env python3
"""
Enhanced Futures Performance Analysis and Auto-Optimization
Analyzes current performance and automatically adjusts risk parameters
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

def analyze_current_positions():
    """Analyze current position performance and identify issues"""
    try:
        # This would normally connect to the API, for now we'll use the known data
        current_positions = {
            'SOLUSDT': {'pnl': -9.42, 'leveraged_roi': -4.68, 'entry_price': 209.63},
            'DOTUSDT': {'pnl': -97.93, 'leveraged_roi': -5.10, 'entry_price': 3.90},
            'UNIUSDT': {'pnl': -7.91, 'leveraged_roi': -7.87, 'entry_price': 9.73},
            'DOGEUSDT': {'pnl': 17.74, 'leveraged_roi': 8.87, 'entry_price': 0.22},
            'ALGOUSDT': {'pnl': -27.82, 'leveraged_roi': -13.89, 'entry_price': 0.24},
            'AAVEUSDT': {'pnl': -4.47, 'leveraged_roi': -4.43, 'entry_price': 327.45}
        }
        
        # Calculate aggregate stats
        total_pnl = sum(pos['pnl'] for pos in current_positions.values())
        losing_positions = [sym for sym, pos in current_positions.items() if pos['pnl'] < 0]
        winning_positions = [sym for sym, pos in current_positions.items() if pos['pnl'] > 0]
        
        # Identify major losers (loss > $50)
        major_losers = [sym for sym, pos in current_positions.items() if pos['pnl'] < -50]
        
        analysis = {
            'timestamp': datetime.now().isoformat(),
            'total_positions': len(current_positions),
            'total_pnl': total_pnl,
            'winning_positions': len(winning_positions),
            'losing_positions': len(losing_positions),
            'win_rate': len(winning_positions) / len(current_positions) if current_positions else 0,
            'major_losers': major_losers,
            'worst_position': min(current_positions.items(), key=lambda x: x[1]['pnl']),
            'best_position': max(current_positions.items(), key=lambda x: x[1]['pnl']),
            'avg_loss_per_position': sum(pos['pnl'] for pos in current_positions.values() if pos['pnl'] < 0) / max(len(losing_positions), 1),
            'avg_win_per_position': sum(pos['pnl'] for pos in current_positions.values() if pos['pnl'] > 0) / max(len(winning_positions), 1),
        }
        
        print("📊 CURRENT PERFORMANCE ANALYSIS")
        print("=" * 50)
        print(f"Total Positions: {analysis['total_positions']}")
        print(f"Total P&L: ${analysis['total_pnl']:.2f}")
        print(f"Win Rate: {analysis['win_rate']:.1%}")
        print(f"Winners: {analysis['winning_positions']} | Losers: {analysis['losing_positions']}")
        print(f"Major Losers (>$50): {len(analysis['major_losers'])} - {analysis['major_losers']}")
        print(f"Worst Position: {analysis['worst_position'][0]} (${analysis['worst_position'][1]['pnl']:.2f})")
        print(f"Best Position: {analysis['best_position'][0]} (+${analysis['best_position'][1]['pnl']:.2f})")
        print(f"Avg Loss: ${analysis['avg_loss_per_position']:.2f}")
        print(f"Avg Win: +${analysis['avg_win_per_position']:.2f}")
        
        return analysis
        
    except Exception as e:
        logger.error(f"Error analyzing positions: {e}")
        return None

def calculate_optimal_risk_settings(analysis: Dict) -> Dict:
    """Calculate optimal risk settings based on current performance"""
    
    win_rate = analysis['win_rate']
    total_pnl = analysis['total_pnl']
    avg_loss = abs(analysis['avg_loss_per_position'])
    avg_win = analysis['avg_win_per_position']
    
    # Base risk calculation using Kelly-like formula
    if avg_loss > 0:
        win_loss_ratio = avg_win / avg_loss
    else:
        win_loss_ratio = 1.0
    
    # Kelly fraction: f = (bp - q) / b
    p = win_rate
    q = 1 - win_rate
    b = win_loss_ratio
    
    if b > 0:
        kelly_fraction = (b * p - q) / b
    else:
        kelly_fraction = 0
    
    # Apply conservative multiplier (25% of Kelly)
    safe_kelly = max(0, min(0.05, kelly_fraction * 0.25))
    
    # Performance-based risk adjustment
    if win_rate < 0.20:  # Very poor performance
        recommended_risk = 0.005  # 0.5%
        recommended_leverage = 5
        recommended_positions = 1
        recommendation = "ULTRA_CONSERVATIVE"
        reason = f"Very low win rate ({win_rate:.1%}), use minimal risk until improvement"
    elif win_rate < 0.35:  # Poor performance
        recommended_risk = 0.01  # 1%
        recommended_leverage = 8
        recommended_positions = 2
        recommendation = "CONSERVATIVE"
        reason = f"Low win rate ({win_rate:.1%}), use conservative settings"
    elif win_rate < 0.50:  # Below average
        recommended_risk = 0.02  # 2%
        recommended_leverage = 12
        recommended_positions = 3
        recommendation = "MODERATE"
        reason = f"Below average win rate ({win_rate:.1%}), moderate risk"
    else:  # Good performance
        recommended_risk = min(0.05, safe_kelly)  # Use Kelly up to 5%
        recommended_leverage = 15
        recommended_positions = 5
        recommendation = "NORMAL"
        reason = f"Good win rate ({win_rate:.1%}), use normal Kelly sizing"
    
    # Additional adjustments based on total P&L
    if total_pnl < -100:  # Significant losses
        recommended_risk *= 0.5  # Halve the risk
        recommended_leverage = min(recommended_leverage, 10)
        reason += " + halved due to significant losses"
    
    return {
        'recommended_risk': recommended_risk,
        'recommended_leverage': recommended_leverage,
        'recommended_positions': recommended_positions,
        'recommendation': recommendation,
        'reason': reason,
        'kelly_fraction': kelly_fraction,
        'safe_kelly': safe_kelly,
        'win_loss_ratio': win_loss_ratio
    }

def apply_optimized_settings(optimization: Dict):
    """Apply optimized settings to environment and save backup"""
    
    try:
        # Backup current settings
        current_settings = {
            'FUTURES_RISK_PER_TRADE': os.getenv('FUTURES_RISK_PER_TRADE', '0.025'),
            'FUTURES_MAX_LEVERAGE': os.getenv('FUTURES_MAX_LEVERAGE', '15'),
            'FUTURES_MAX_POSITIONS': os.getenv('FUTURES_MAX_POSITIONS', '3'),
            'backup_timestamp': datetime.now().isoformat()
        }
        
        # Save backup
        with open('futures_settings_backup.json', 'w') as f:
            json.dump(current_settings, f, indent=2)
        
        # Apply new settings
        os.environ['FUTURES_RISK_PER_TRADE'] = str(optimization['recommended_risk'])
        os.environ['FUTURES_MAX_LEVERAGE'] = str(optimization['recommended_leverage'])
        os.environ['FUTURES_MAX_POSITIONS'] = str(optimization['recommended_positions'])
        
        print(f"\n🔧 OPTIMIZATION APPLIED: {optimization['recommendation']}")
        print("=" * 50)
        print(f"Risk per trade: {current_settings['FUTURES_RISK_PER_TRADE']} → {optimization['recommended_risk']:.3f}")
        print(f"Max leverage: {current_settings['FUTURES_MAX_LEVERAGE']} → {optimization['recommended_leverage']}")
        print(f"Max positions: {current_settings['FUTURES_MAX_POSITIONS']} → {optimization['recommended_positions']}")
        print(f"Reasoning: {optimization['reason']}")
        
        # Update .env file for persistence
        update_env_file(optimization)
        
        return True
        
    except Exception as e:
        logger.error(f"Error applying settings: {e}")
        return False

def update_env_file(optimization: Dict):
    """Update .env file with new optimized settings"""
    try:
        # Read current .env file
        with open('.env', 'r') as f:
            lines = f.readlines()
        
        # Update relevant lines
        updated_lines = []
        for line in lines:
            if line.startswith('FUTURES_RISK_PER_TRADE='):
                updated_lines.append(f"FUTURES_RISK_PER_TRADE={optimization['recommended_risk']:.3f}                # 🤖 AUTO-OPTIMIZED: {optimization['recommendation']}\n")
            elif line.startswith('FUTURES_MAX_LEVERAGE='):
                updated_lines.append(f"FUTURES_MAX_LEVERAGE={optimization['recommended_leverage']}                     # 🤖 AUTO-OPTIMIZED: Based on current performance\n")
            elif line.startswith('FUTURES_MAX_POSITIONS='):
                updated_lines.append(f"FUTURES_MAX_POSITIONS={optimization['recommended_positions']}                     # 🤖 AUTO-OPTIMIZED: Quality over quantity\n")
            else:
                updated_lines.append(line)
        
        # Write updated .env file
        with open('.env', 'w') as f:
            f.writelines(updated_lines)
        
        print("✅ .env file updated with optimized settings")
        
    except Exception as e:
        logger.error(f"Error updating .env file: {e}")

def generate_improvement_recommendations(analysis: Dict, optimization: Dict) -> List[str]:
    """Generate specific recommendations for improvement"""
    
    recommendations = []
    
    # Win rate recommendations
    if analysis['win_rate'] < 0.30:
        recommendations.append("🎯 Focus on signal quality: Increase TB_MIN_SIGNAL_QUALITY to 4.0+")
        recommendations.append("🔍 Add more confirmation: Use multiple timeframe analysis")
        recommendations.append("⏰ Consider longer holding periods: Quick exits may be hurting performance")
    
    # Position sizing recommendations
    if abs(analysis['avg_loss_per_position']) > analysis['avg_win_per_position'] * 2:
        recommendations.append("🛡️ Improve stop losses: Large losses are outweighing wins")
        recommendations.append("💰 Consider profit taking: Lock in gains earlier")
    
    # Specific symbol recommendations
    if analysis['major_losers']:
        recommendations.append(f"🚫 Consider blacklisting poor performers: {', '.join(analysis['major_losers'])}")
        recommendations.append("📊 Analyze what went wrong with major losers")
    
    # Risk management recommendations
    if analysis['total_pnl'] < -50:
        recommendations.append("🔥 Implement cooling-off period: Stop trading for 24-48 hours")
        recommendations.append("📚 Review entry criteria: Current signals may be too weak")
    
    return recommendations

def main():
    """Main optimization function"""
    print("🤖 AUTOMATED FUTURES PERFORMANCE OPTIMIZATION")
    print("=" * 60)
    
    # Step 1: Analyze current performance
    analysis = analyze_current_positions()
    if not analysis:
        print("❌ Could not analyze current positions")
        return
    
    # Step 2: Calculate optimal settings
    optimization = calculate_optimal_risk_settings(analysis)
    
    # Step 3: Generate recommendations
    recommendations = generate_improvement_recommendations(analysis, optimization)
    
    # Step 4: Apply optimization
    success = apply_optimized_settings(optimization)
    
    if success:
        print(f"\n💡 IMPROVEMENT RECOMMENDATIONS")
        print("=" * 40)
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec}")
        
        print(f"\n✅ OPTIMIZATION COMPLETE")
        print(f"Kelly fraction: {optimization['kelly_fraction']:.3f}")
        print(f"Win/Loss ratio: {optimization['win_loss_ratio']:.2f}")
        print(f"Next review: In 24 hours or after 10 more trades")
    else:
        print("❌ Optimization failed")

if __name__ == "__main__":
    main()
