#!/usr/bin/env python3
"""
Dual Agent Monitoring Dashboard
Track performance of both main agent and futures agent
"""

import os
import sys
import json
import time
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List
import logging

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DualAgentMonitor:
    """Monitor both main agent and futures agent performance"""

    def __init__(self):
        self.main_agent_log = "main_agent.log"
        self.futures_agent_log = "futures_agent.log"
        self.futures_trade_log = "futures_trades.json"

    def get_main_agent_status(self) -> Dict:
        """Get main agent status from logs"""
        status = {
            'name': 'Main Agent (Low-Risk)',
            'status': 'unknown',
            'last_activity': None,
            'trades_today': 0,
            'pnl_today': 0.0,
            'active_positions': 0
        }

        try:
            if os.path.exists(self.main_agent_log):
                with open(self.main_agent_log, 'r') as f:
                    lines = f.readlines()[-50:]  # Last 50 lines

                # Check if running (look for recent heartbeat)
                recent_lines = [line for line in lines if 'INFO' in line and
                              (datetime.now() - datetime.strptime(
                                  line.split()[0] + ' ' + line.split()[1],
                                  '%Y-%m-%d %H:%M:%S')).seconds < 3600]

                if recent_lines:
                    status['status'] = 'running'
                    status['last_activity'] = recent_lines[-1].split()[0] + ' ' + recent_lines[-1].split()[1]
                else:
                    status['status'] = 'stopped'

                # Extract trade information (simplified parsing)
                for line in lines:
                    if 'TRADE' in line.upper() or 'BUY' in line.upper() or 'SELL' in line.upper():
                        status['trades_today'] += 1

        except Exception as e:
            logger.warning(f"Error reading main agent log: {e}")

        return status

    def get_futures_agent_status(self) -> Dict:
        """Get futures agent status from logs and trade data"""
        status = {
            'name': 'Futures Agent (High-Risk)',
            'status': 'unknown',
            'last_activity': None,
            'trades_today': 0,
            'pnl_today': 0.0,
            'open_positions': 0,
            'win_rate': 0.0,
            'total_trades': 0
        }

        try:
            # Check log file
            if os.path.exists(self.futures_agent_log):
                with open(self.futures_agent_log, 'r') as f:
                    lines = f.readlines()[-50:]

                # Check if running
                recent_lines = [line for line in lines if 'INFO' in line and
                              (datetime.now() - datetime.strptime(
                                  line.split()[0] + ' ' + line.split()[1],
                                  '%Y-%m-%d %H:%M:%S')).seconds < 3600]

                if recent_lines:
                    status['status'] = 'running'
                    status['last_activity'] = recent_lines[-1].split()[0] + ' ' + recent_lines[-1].split()[1]
                else:
                    status['status'] = 'stopped'

            # Check trade log
            if os.path.exists(self.futures_trade_log):
                try:
                    with open(self.futures_trade_log, 'r') as f:
                        trades = json.load(f)

                    # Filter today's trades
                    today = datetime.now().date()
                    today_trades = [t for t in trades if
                                  datetime.fromisoformat(t['timestamp']).date() == today]

                    status['trades_today'] = len(today_trades)

                    # Calculate today's P&L
                    closed_trades = [t for t in today_trades if t.get('action') == 'close']
                    status['pnl_today'] = sum(t.get('pnl_amount', 0) for t in closed_trades)

                    # Count open positions
                    status['open_positions'] = len([t for t in today_trades if t.get('action') != 'close'])

                    # Calculate win rate
                    if closed_trades:
                        winning_trades = sum(1 for t in closed_trades if t.get('pnl_amount', 0) > 0)
                        status['win_rate'] = winning_trades / len(closed_trades)

                    status['total_trades'] = len(trades)

                except json.JSONDecodeError:
                    pass  # File might be empty or corrupted

        except Exception as e:
            logger.warning(f"Error reading futures agent data: {e}")

        return status

    def get_system_status(self) -> Dict:
        """Get overall system status"""
        return {
            'timestamp': datetime.now().isoformat(),
            'main_agent': self.get_main_agent_status(),
            'futures_agent': self.get_futures_agent_status()
        }

    def print_dashboard(self):
        """Print monitoring dashboard"""
        print("🚀 Dual Agent Monitoring Dashboard")
        print("=" * 60)

        status = self.get_system_status()

        # Main Agent Section
        main = status['main_agent']
        print(f"🤖 {main['name']}")
        print(f"   Status: {'✅ RUNNING' if main['status'] == 'running' else '❌ STOPPED'}")
        print(f"   Last Activity: {main['last_activity'] or 'N/A'}")
        print(f"   Trades Today: {main['trades_today']}")
        print(".2f")
        print(f"   Active Positions: {main['active_positions']}")
        print()

        # Futures Agent Section
        futures = status['futures_agent']
        print(f"⚡ {futures['name']}")
        print(f"   Status: {'✅ RUNNING' if futures['status'] == 'running' else '❌ STOPPED'}")
        print(f"   Last Activity: {futures['last_activity'] or 'N/A'}")
        print(f"   Trades Today: {futures['trades_today']}")
        print(".2f")
        print(f"   Open Positions: {futures['open_positions']}")
        print(".1%")
        print(f"   Total Trades: {futures['total_trades']}")
        print()

        # Summary
        total_pnl = main['pnl_today'] + futures['pnl_today']
        total_trades = main['trades_today'] + futures['trades_today']

        print("📊 Daily Summary")
        print(f"   Combined P&L: ${total_pnl:.2f}")
        print(f"   Total Trades: {total_trades}")
        print(f"   System Status: {'🟢 HEALTHY' if main['status'] == 'running' and futures['status'] == 'running' else '🟡 ISSUES'}")
        print()

        # Performance Insights
        print("💡 Performance Insights")
        if futures['trades_today'] > 0:
            print(f"   Futures Agent Active: {futures['trades_today']} trades executed")
        if main['trades_today'] > 0:
            print(f"   Main Agent Active: {main['trades_today']} trades executed")
        if total_pnl > 0:
            print(".2f")
        elif total_pnl < 0:
            print(".2f")
        else:
            print("   📊 No P&L movement today")
        print()

    def export_report(self, filename: str = None):
        """Export detailed performance report"""
        if not filename:
            filename = f"dual_agent_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        status = self.get_system_status()

        with open(filename, 'w') as f:
            json.dump(status, f, indent=2, default=str)

        print(f"📄 Report exported to {filename}")
        return filename

def main():
    """Main monitoring function"""
    import argparse

    parser = argparse.ArgumentParser(description='Dual Agent Monitoring Dashboard')
    parser.add_argument('--export', action='store_true', help='Export detailed report')
    parser.add_argument('--watch', action='store_true', help='Continuous monitoring mode')
    parser.add_argument('--interval', type=int, default=60, help='Watch interval in seconds')

    args = parser.parse_args()

    monitor = DualAgentMonitor()

    if args.watch:
        print("👀 Continuous Monitoring Mode (Ctrl+C to stop)")
        print("=" * 50)
        try:
            while True:
                os.system('clear')  # Clear screen
                monitor.print_dashboard()
                print(f"🔄 Refreshing in {args.interval} seconds...")
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\n🛑 Monitoring stopped")
    else:
        monitor.print_dashboard()

        if args.export:
            monitor.export_report()

if __name__ == "__main__":
    main()
