#!/usr/bin/env python3
"""
Manual Position Manager for Hybrid Trader
Implements manual TP/SL management since bracket orders are broken
"""

import sys
import os
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional

# Add project root to path
sys.path.append('.')

# Load environment
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import config
from alpaca import _rest, recent_bars

# Trade Quality Based TP/SL System
TRADE_QUALITY_LEVELS = {
    'excellent': {  # Strong signals, high confidence
        'tp_range': (0.12, 0.20),  # 12-20% TP
        'sl_base': 0.06,           # 6% SL
        'description': 'High conviction, strong signals'
    },
    'good': {      # Solid signals, medium confidence  
        'tp_range': (0.08, 0.12),  # 8-12% TP
        'sl_base': 0.05,           # 5% SL
        'description': 'Good setup, decent confidence'
    },
    'fair': {      # Weak signals, low confidence
        'tp_range': (0.05, 0.08),  # 5-8% TP
        'sl_base': 0.04,           # 4% SL
        'description': 'Marginal setup, low confidence'
    }
}

# Asset difficulty multipliers (how hard it is to move)
ASSET_DIFFICULTY = {
    'BTC': 1.5,    # Hardest to move (King of crypto)
    'ETH': 1.3,    # Very hard to move
    'SOL': 1.1,    # Medium-hard to move
    'AVAX': 1.0,   # Baseline difficulty
    'LINK': 1.0,
    'LTC': 1.0,
    'BCH': 1.0,
    'UNI': 0.9,    # Easier to move
    'AAVE': 0.9,
    'DOT': 0.9,
    'XTZ': 0.8,    # Much easier to move
    'YFI': 0.7,    # Easiest to move (low liquidity)
}

class IntelligentPositionManager:
    """Intelligent position management with trade-quality based TP/SL"""
    
    def __init__(self):
        self.api = _rest()
        self.trailing_pct = 0.05  # 5% trailing stop (universal)
        
        print(f"📊 Intelligent Position Manager initialized:")
        print(f"   🎯 Trade-Quality Based TP/SL System")
        print(f"   📈 Excellent: 12-20% TP | Good: 8-12% TP | Fair: 5-8% TP")
        print(f"   🛡️  Asset difficulty multipliers applied")
        print(f"   🔄 Trailing Stop: {self.trailing_pct*100:.1f}%")
        print()
    
    def get_asset_symbol_clean(self, symbol: str) -> str:
        """Clean symbol for asset lookup"""
        return symbol.replace('/USD', '').replace('USD', '').upper()
    
    def analyze_trade_quality(self, symbol: str, pnl_pct: float, holding_time_hours: float = None) -> str:
        """
        Analyze trade quality based on multiple factors
        This is a simplified version - in reality, this would use:
        - Signal strength from entry
        - Market conditions
        - Technical analysis confluence  
        - Sentiment analysis
        - Volume profile
        """
        symbol_clean = self.get_asset_symbol_clean(symbol)
        
        # Simplified heuristics for demo (in production, use actual signal data)
        current_pnl_abs = abs(pnl_pct)
        
        # Quick gains often indicate strong momentum
        if current_pnl_abs > 0.08:  # Already 8%+ move
            return 'excellent'
        elif current_pnl_abs > 0.04:  # 4-8% move
            return 'good'  
        elif current_pnl_abs > 0.02:  # 2-4% move
            return 'fair'
        else:
            # New positions or small moves - analyze by asset
            if symbol_clean in ['BTC', 'ETH']:
                return 'good'  # Blue chips get benefit of doubt
            elif symbol_clean in ['SOL', 'AVAX', 'LINK', 'UNI', 'AAVE']:
                return 'fair'  # Major alts default to fair
            else:
                return 'fair'  # Small caps default to fair
    
    def calculate_intelligent_targets(self, symbol: str, entry_price: float, current_pnl_pct: float) -> Dict[str, float]:
        """Calculate TP/SL based on trade quality and asset difficulty"""
        symbol_clean = self.get_asset_symbol_clean(symbol)
        
        # Analyze trade quality
        trade_quality = self.analyze_trade_quality(symbol, current_pnl_pct)
        quality_config = TRADE_QUALITY_LEVELS[trade_quality]
        
        # Get asset difficulty multiplier
        difficulty = ASSET_DIFFICULTY.get(symbol_clean, 1.0)
        
        # Calculate base TP (use middle of range)
        tp_min, tp_max = quality_config['tp_range']
        base_tp = (tp_min + tp_max) / 2
        
        # Adjust TP for asset difficulty
        # Harder assets (BTC) get higher targets, easier assets get lower
        adjusted_tp = base_tp * difficulty
        
        # SL stays consistent regardless of difficulty
        sl_pct = quality_config['sl_base']
        
        return {
            'tp_pct': adjusted_tp,
            'sl_pct': sl_pct,
            'trade_quality': trade_quality,
            'difficulty': difficulty,
            'quality_description': quality_config['description'],
            'tp_price': entry_price * (1 + adjusted_tp),
            'sl_price': entry_price * (1 - sl_pct)
        }
    
    def get_positions(self) -> List[Dict]:
        """Get current positions with intelligent TP/SL analysis"""
        positions = []
        
        try:
            for pos in self.api.list_positions():
                entry_price = float(pos.avg_entry_price)
                current_price = float(pos.current_price)
                qty = float(pos.qty)
                pnl_pct = float(pos.unrealized_plpc)
                pnl_usd = float(pos.unrealized_pl)
                
                # Get intelligent targets based on trade quality and asset difficulty
                targets = self.calculate_intelligent_targets(pos.symbol, entry_price, pnl_pct)
                
                tp_pct = targets['tp_pct']
                sl_pct = targets['sl_pct']
                tp_price = targets['tp_price']
                sl_price = targets['sl_price']
                
                # Determine exit conditions
                should_exit = False
                exit_reason = ""
                
                if pnl_pct >= tp_pct:
                    should_exit = True
                    exit_reason = f"TAKE PROFIT ({pnl_pct*100:.2f}% >= {tp_pct*100:.1f}% {targets['trade_quality']})"
                elif pnl_pct <= -sl_pct:
                    should_exit = True
                    exit_reason = f"STOP LOSS ({pnl_pct*100:.2f}% <= {-sl_pct*100:.1f}% {targets['trade_quality']})"
                    
                positions.append({
                    'symbol': pos.symbol,
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
                    'market_value': qty * current_price
                })
                
        except Exception as e:
            print(f"❌ Error getting positions: {e}")
            
        return positions
    
    def print_positions(self):
        """Print current positions with intelligent TP/SL analysis"""
        positions = self.get_positions()
        
        if not positions:
            print("📊 No open positions")
            return
            
        print("📊 CURRENT POSITIONS (Intelligent TP/SL):")
        print("=" * 100)
        
        total_value = 0
        exit_count = 0
        
        for pos in positions:
            status = "🚨 EXIT" if pos['should_exit'] else "✅ HOLD"
            quality_emoji = {"excellent": "�", "good": "✅", "fair": "⚡"}[pos['trade_quality']]
            
            print(f"{status} {pos['symbol']} {quality_emoji}{pos['trade_quality'].upper()}:")
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
            
        print(f"📈 Total Portfolio Value: ${total_value:,.2f}")
        print(f"🎯 Trade Quality Distribution: {exit_count} positions ready to exit")
        print()
        
        # Quality summary
        quality_counts = {}
        for pos in positions:
            quality = pos['trade_quality']
            quality_counts[quality] = quality_counts.get(quality, 0) + 1
            
        if quality_counts:
            print("📊 TRADE QUALITY BREAKDOWN:")
            for quality, count in quality_counts.items():
                emoji = {"excellent": "�", "good": "✅", "fair": "⚡"}[quality]
                print(f"   {emoji} {quality.capitalize()}: {count} positions")
        print()
        print("Legend: � Excellent Signal | ✅ Good Signal | ⚡ Fair Signal")
        
    def close_position(self, symbol: str, reason: str = "Manual close") -> bool:
        """Close a specific position"""
        try:
            # Find the position
            pos = None
            for p in self.api.list_positions():
                if p.symbol == symbol:
                    pos = p
                    break
                    
            if not pos:
                print(f"❌ No position found for {symbol}")
                return False
                
            qty = abs(float(pos.qty))
            side = "sell" if float(pos.qty) > 0 else "buy"
            
            print(f"🔄 Closing {symbol}: {side} {qty} shares")
            print(f"   Reason: {reason}")
            
            # Place market order to close
            order = self.api.submit_order(
                symbol=symbol,
                qty=qty,
                side=side,
                type='market',
                time_in_force='gtc'
            )
            
            order_id = getattr(order, 'id', 'unknown')
            print(f"✅ Close order submitted: {order_id}")
            
            # Log the close
            pnl_pct = float(pos.unrealized_plpc) * 100
            pnl_usd = float(pos.unrealized_pl)
            
            print(f"📊 Final P&L: {pnl_pct:+.2f}% (${pnl_usd:+.2f})")
            
            return True
            
        except Exception as e:
            print(f"❌ Error closing {symbol}: {e}")
            return False
    
    def close_profitable_positions(self) -> int:
        """Close all positions that hit their dynamic take profit"""
        positions = self.get_positions()
        closed_count = 0
        
        for pos in positions:
            if pos['pnl_pct'] >= pos['tp_pct']:  # Use dynamic TP level
                print(f"🎯 Closing profitable position: {pos['symbol']} (+{pos['pnl_pct']*100:.2f}% vs {pos['tp_pct']*100:.1f}% target)")
                if self.close_position(pos['symbol'], pos['exit_reason']):
                    closed_count += 1
                    time.sleep(1)  # Rate limiting
                    
        return closed_count
    
    def close_losing_positions(self) -> int:
        """Close all positions that hit their dynamic stop loss"""
        positions = self.get_positions()
        closed_count = 0
        
        for pos in positions:
            if pos['pnl_pct'] <= -pos['sl_pct']:  # Use dynamic SL level
                print(f"🛑 Closing losing position: {pos['symbol']} ({pos['pnl_pct']*100:.2f}% vs {-pos['sl_pct']*100:.1f}% stop)")
                if self.close_position(pos['symbol'], pos['exit_reason']):
                    closed_count += 1
                    time.sleep(1)  # Rate limiting
                    
        return closed_count

def main():
    """Main function"""
    manager = IntelligentPositionManager()
    
    if len(sys.argv) < 2:
        print("📊 Intelligent Position Manager")
        print("Usage:")
        print("  python3 manual_position_manager.py status      # Show positions")
        print("  python3 manual_position_manager.py close_tp    # Close profitable positions")
        print("  python3 manual_position_manager.py close_sl    # Close losing positions") 
        print("  python3 manual_position_manager.py close ALL   # Close all positions")
        print("  python3 manual_position_manager.py close SYMBOL # Close specific position")
        print()
        manager.print_positions()
        return
        
    command = sys.argv[1].lower()
    
    if command == "status":
        manager.print_positions()
        
    elif command == "close_tp":
        print("🎯 Closing profitable positions...")
        closed = manager.close_profitable_positions()
        print(f"✅ Closed {closed} profitable positions")
        
    elif command == "close_sl":
        print("🛑 Closing losing positions...")
        closed = manager.close_losing_positions()
        print(f"✅ Closed {closed} losing positions")
        
    elif command == "close":
        if len(sys.argv) < 3:
            print("❌ Specify symbol or 'ALL'")
            return
            
        target = sys.argv[2].upper()
        
        if target == "ALL":
            print("🚨 Closing ALL positions...")
            positions = manager.get_positions()
            closed = 0
            for pos in positions:
                if manager.close_position(pos['symbol'], "Manual close all"):
                    closed += 1
                    time.sleep(1)
            print(f"✅ Closed {closed} positions")
        else:
            manager.close_position(target, "Manual close")
    else:
        print(f"❌ Unknown command: {command}")

if __name__ == "__main__":
    main()
