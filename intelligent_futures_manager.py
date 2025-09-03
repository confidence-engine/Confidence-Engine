#!/usr/bin/env python3
"""
Intelligent Futures Position Manager
Trade-quality based TP/SL system for futures trading with leverage considerations
"""

import os
import sys
import time
import logging
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass
import numpy as np

# Ensure project root on sys.path
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJ_ROOT = os.path.dirname(_THIS_DIR)
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

try:
    from alpaca_trade_api.rest import REST
    from alpaca import _rest
except ImportError:
    print("❌ Alpaca API not available")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Trade Quality Levels for Futures (more aggressive due to leverage)
FUTURES_TRADE_QUALITY_LEVELS = {
    'excellent': {
        'tp_range': (0.15, 0.25),  # 15-25% TP for excellent signals
        'sl_base': 0.06,           # 6% SL (tighter due to leverage)
        'description': 'High conviction setup, strong confluence'
    },
    'good': {
        'tp_range': (0.10, 0.15),  # 10-15% TP for good signals  
        'sl_base': 0.05,           # 5% SL
        'description': 'Good setup, decent confidence'
    },
    'fair': {
        'tp_range': (0.06, 0.10),  # 6-10% TP for fair signals
        'sl_base': 0.04,           # 4% SL
        'description': 'Marginal setup, lower confidence'
    }
}

# Asset Difficulty for Futures (adjusted for leverage and volatility)
FUTURES_ASSET_DIFFICULTY = {
    'BTC': 1.6,   # Hardest to move significantly
    'ETH': 1.4,   # Major crypto, decent liquidity
    'SOL': 1.2,   # Top 10, good liquidity
    'AVAX': 1.1,  # Top 20, moderate liquidity
    'LINK': 1.1,  # Established DeFi
    'UNI': 1.0,   # DEX token, baseline
    'AAVE': 0.9,  # DeFi, smaller
    'COMP': 0.8,  # Smaller DeFi
    'YFI': 0.7,   # Smallest, highest volatility
    'XTZ': 0.8,   # Alt L1
    'LTC': 1.0,   # Legacy crypto
    'BCH': 1.0,   # Bitcoin fork
}

class IntelligentFuturesManager:
    """Intelligent futures position management with trade-quality based TP/SL"""
    
    def __init__(self):
        self.api = _rest()
        self.leverage_multiplier = 1.2  # Futures are already leveraged, so adjust targets
        
        print(f"🚀 Intelligent Futures Position Manager initialized:")
        print(f"   🎯 Futures Trade-Quality Based TP/SL System")
        print(f"   📈 Excellent: 15-25% TP | Good: 10-15% TP | Fair: 6-10% TP")
        print(f"   🛡️  Leverage-adjusted SL: 4-6%")
        print(f"   🔥 Asset difficulty multipliers applied")
        print()
    
    def get_asset_symbol_clean(self, symbol: str) -> str:
        """Clean symbol for asset lookup"""
        return symbol.replace('/USD', '').replace('USD', '').upper()
    
    def analyze_futures_trade_quality(self, symbol: str, pnl_pct: float, 
                                    entry_conditions: Optional[Dict] = None) -> str:
        """
        Analyze futures trade quality based on multiple factors
        More aggressive than crypto due to leverage
        """
        symbol_clean = self.get_asset_symbol_clean(symbol)
        current_pnl_abs = abs(pnl_pct)
        
        # Futures move faster, so lower thresholds for quality assessment
        if current_pnl_abs > 0.06:  # Already 6%+ move (excellent momentum)
            return 'excellent'
        elif current_pnl_abs > 0.03:  # 3-6% move (good momentum)
            return 'good'  
        elif current_pnl_abs > 0.015:  # 1.5-3% move (fair momentum)
            return 'fair'
        else:
            # New positions - analyze by asset and entry conditions
            if entry_conditions:
                # In production, use actual signal strength
                confidence = entry_conditions.get('confidence', 0.5)
                if confidence > 0.7:
                    return 'excellent'
                elif confidence > 0.5:
                    return 'good'
                else:
                    return 'fair'
            
            # Default by asset class
            if symbol_clean in ['BTC', 'ETH']:
                return 'good'  # Blue chips get benefit of doubt
            elif symbol_clean in ['SOL', 'AVAX', 'LINK']:
                return 'fair'  # Major alts default to fair
            else:
                return 'fair'  # Smaller alts default to fair
    
    def calculate_intelligent_futures_targets(self, symbol: str, entry_price: float, 
                                            current_pnl_pct: float, side: str = 'long') -> Dict[str, float]:
        """Calculate TP/SL based on trade quality and futures asset difficulty"""
        symbol_clean = self.get_asset_symbol_clean(symbol)
        
        # Analyze trade quality for futures
        trade_quality = self.analyze_futures_trade_quality(symbol, current_pnl_pct)
        quality_config = FUTURES_TRADE_QUALITY_LEVELS[trade_quality]
        
        # Get asset difficulty multiplier
        difficulty = FUTURES_ASSET_DIFFICULTY.get(symbol_clean, 1.0)
        
        # Calculate base TP (use middle of range)
        tp_min, tp_max = quality_config['tp_range']
        base_tp = (tp_min + tp_max) / 2
        
        # Adjust TP for asset difficulty and leverage
        adjusted_tp = base_tp * difficulty * self.leverage_multiplier
        
        # SL adjusted for leverage (tighter stops)
        sl_pct = quality_config['sl_base']
        
        # Calculate target prices based on side
        if side.lower() in ['long', 'buy']:
            tp_price = entry_price * (1 + adjusted_tp)
            sl_price = entry_price * (1 - sl_pct)
        else:  # short/sell
            tp_price = entry_price * (1 - adjusted_tp)
            sl_price = entry_price * (1 + sl_pct)
        
        return {
            'tp_pct': adjusted_tp,
            'sl_pct': sl_pct,
            'trade_quality': trade_quality,
            'difficulty': difficulty,
            'quality_description': quality_config['description'],
            'tp_price': tp_price,
            'sl_price': sl_price,
            'side': side
        }
    
    def get_futures_positions(self) -> List[Dict]:
        """Get all futures positions with intelligent TP/SL analysis"""
        positions = []
        
        try:
            for pos in self.api.list_positions():
                symbol = pos.symbol
                qty = float(pos.qty)
                side = 'long' if qty > 0 else 'short'
                entry_price = float(pos.avg_entry_price)
                current_price = float(pos.current_price)
                pnl_pct = float(pos.unrealized_plpc)
                pnl_usd = float(pos.unrealized_pl)
                
                # Get intelligent targets based on trade quality and asset difficulty
                targets = self.calculate_intelligent_futures_targets(symbol, entry_price, pnl_pct, side)
                
                tp_pct = targets['tp_pct']
                sl_pct = targets['sl_pct']
                tp_price = targets['tp_price']
                sl_price = targets['sl_price']
                
                # Determine exit conditions for futures
                should_exit = False
                exit_reason = ""
                
                if side == 'long':
                    if current_price >= tp_price:
                        should_exit = True
                        exit_reason = f"FUTURES TP HIT: ${current_price:.2f} >= ${tp_price:.2f} ({targets['trade_quality']})"
                    elif current_price <= sl_price:
                        should_exit = True
                        exit_reason = f"FUTURES SL HIT: ${current_price:.2f} <= ${sl_price:.2f} ({targets['trade_quality']})"
                else:  # short
                    if current_price <= tp_price:
                        should_exit = True
                        exit_reason = f"FUTURES TP HIT: ${current_price:.2f} <= ${tp_price:.2f} ({targets['trade_quality']})"
                    elif current_price >= sl_price:
                        should_exit = True
                        exit_reason = f"FUTURES SL HIT: ${current_price:.2f} >= ${sl_price:.2f} ({targets['trade_quality']})"
                
                # Additional futures-specific checks
                if abs(pnl_pct) > 0.30:  # 30% moves in futures are extreme
                    should_exit = True
                    exit_reason = f"EXTREME MOVE: {pnl_pct*100:.1f}% - TAKE PROFITS"
                
                positions.append({
                    'symbol': symbol,
                    'side': side,
                    'qty': qty,
                    'entry_price': entry_price,
                    'current_price': current_price,
                    'pnl_pct': pnl_pct,
                    'pnl_usd': pnl_usd,
                    'tp_price': tp_price,
                    'sl_price': sl_price,
                    'tp_pct': tp_pct,
                    'sl_pct': sl_pct,
                    'trade_quality': targets['trade_quality'],
                    'difficulty': targets['difficulty'],
                    'quality_description': targets['quality_description'],
                    'should_exit': should_exit,
                    'exit_reason': exit_reason,
                    'market_value': abs(qty) * current_price
                })
                
        except Exception as e:
            print(f"❌ Error getting futures positions: {e}")
            
        return positions
    
    def print_futures_positions(self):
        """Print current futures positions with intelligent TP/SL analysis"""
        positions = self.get_futures_positions()
        
        if not positions:
            print("🚀 No open futures positions")
            return
            
        print("🚀 CURRENT FUTURES POSITIONS (Intelligent TP/SL):")
        print("=" * 110)
        
        total_value = 0
        exit_count = 0
        
        for pos in positions:
            status = "🚨 EXIT" if pos['should_exit'] else "✅ HOLD"
            quality_emoji = {"excellent": "🔥", "good": "✅", "fair": "⚡"}[pos['trade_quality']]
            side_emoji = "📈" if pos['side'] == 'long' else "📉"
            
            print(f"{status} {pos['symbol']} {side_emoji}{pos['side'].upper()} {quality_emoji}{pos['trade_quality'].upper()}:")
            print(f"   Quality: {pos['quality_description']}")
            print(f"   Difficulty: {pos['difficulty']:.1f}x ({pos['symbol'].replace('/USD', '').replace('USD', '')})")
            print(f"   Entry: ${pos['entry_price']:.2f} → Current: ${pos['current_price']:.2f}")
            print(f"   Qty: {pos['qty']:.4f} | Value: ${pos['market_value']:.2f}")
            print(f"   P&L: {pos['pnl_pct']*100:+.2f}% (${pos['pnl_usd']:+.2f})")
            print(f"   TP: ${pos['tp_price']:.2f} ({pos['tp_pct']*100:.1f}%) | SL: ${pos['sl_price']:.2f} ({pos['sl_pct']*100:.1f}%)")
            
            if pos['should_exit']:
                print(f"   🚨 {pos['exit_reason']}")
                exit_count += 1
                
            total_value += pos['market_value']
            print()
            
        print(f"🚀 Total Futures Portfolio Value: ${total_value:,.2f}")
        print(f"🎯 Trade Quality Distribution: {exit_count} positions ready to exit")
        print()
        
        # Quality summary
        quality_counts = {}
        side_counts = {'long': 0, 'short': 0}
        for pos in positions:
            quality = pos['trade_quality']
            quality_counts[quality] = quality_counts.get(quality, 0) + 1
            side_counts[pos['side']] += 1
            
        if quality_counts:
            print("📊 FUTURES TRADE QUALITY BREAKDOWN:")
            for quality, count in quality_counts.items():
                emoji = {"excellent": "🔥", "good": "✅", "fair": "⚡"}[quality]
                print(f"   {emoji} {quality.capitalize()}: {count} positions")
        
        print(f"📊 SIDE DISTRIBUTION: 📈 Long: {side_counts['long']} | 📉 Short: {side_counts['short']}")
        print()
        print("Legend: 🔥 Excellent Signal | ✅ Good Signal | ⚡ Fair Signal")
        print("        📈 Long Position | 📉 Short Position")

    def close_futures_position(self, symbol: str, reason: str = "Manual close") -> bool:
        """Close a futures position"""
        try:
            # Close the position by market order
            order = self.api.close_position(symbol)
            print(f"✅ Closed futures position {symbol}: {reason}")
            return True
        except Exception as e:
            print(f"❌ Failed to close futures position {symbol}: {e}")
            return False

def main():
    """Main function for futures position management"""
    manager = IntelligentFuturesManager()
    
    if len(sys.argv) < 2:
        print("🚀 Intelligent Futures Position Manager")
        print("Usage:")
        print("  python3 intelligent_futures_manager.py status      # Show positions")
        print("  python3 intelligent_futures_manager.py monitor     # Auto-monitor")
        print("  python3 intelligent_futures_manager.py close SYMBOL # Close specific position")
        print()
        manager.print_futures_positions()
        return
        
    command = sys.argv[1].lower()
    
    if command == "status":
        manager.print_futures_positions()
    elif command == "monitor":
        print("🚀 Starting intelligent futures position monitoring...")
        while True:
            try:
                print(f"\n{'='*110}")
                print(f"🔍 Scanning futures positions at {datetime.now().strftime('%H:%M:%S')}")
                
                positions = manager.get_futures_positions()
                if not positions:
                    print("🚀 No futures positions to monitor")
                else:
                    manager.print_futures_positions()
                
                print(f"💤 Sleeping for 30 seconds... (futures need faster monitoring)")
                time.sleep(30)  # Faster monitoring for futures
                
            except KeyboardInterrupt:
                print(f"\n🛑 Futures position monitoring stopped by user")
                break
            except Exception as e:
                print(f"❌ Error in futures monitoring loop: {e}")
                print(f"💤 Sleeping for 30 seconds before retry...")
                time.sleep(30)
    elif command == "close":
        if len(sys.argv) < 3:
            print("❌ Specify symbol to close")
            return
        symbol = sys.argv[2].upper()
        manager.close_futures_position(symbol, "Manual close")
    else:
        print(f"❌ Unknown command: {command}")

if __name__ == "__main__":
    main()
