#!/usr/bin/env python3
"""
Paper Trading Threshold Optimizer
Automatically adjust thresholds for optimal validation data collection
"""

import os
import json
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PaperTradingOptimizer:
    """Optimize thresholds specifically for paper trading validation phase"""
    
    def __init__(self):
        self.current_env = self._load_current_env()
        self.target_signals_per_week = 7  # 1 per day for good validation data
        
    def _load_current_env(self) -> Dict[str, str]:
        """Load current environment configuration"""
        from dotenv import load_dotenv
        load_dotenv()
        
        return {
            'TB_SENTIMENT_CUTOFF': os.getenv('TB_SENTIMENT_CUTOFF', '0.5'),
            'TB_DIVERGENCE_THRESHOLD': os.getenv('TB_DIVERGENCE_THRESHOLD', '0.5'),
            'TB_MIN_CONFIDENCE': os.getenv('TB_MIN_CONFIDENCE', '0.65'),
            'TB_NO_TRADE': os.getenv('TB_NO_TRADE', '0'),
            'TB_TRADER_OFFLINE': os.getenv('TB_TRADER_OFFLINE', '0')
        }
    
    def analyze_current_signal_rate(self, days_back: int = 14) -> Dict[str, Any]:
        """Analyze current signal generation rate"""
        runs_dir = Path("runs")
        if not runs_dir.exists():
            return {'signals_per_week': 0, 'analysis': 'No runs directory found'}
        
        cutoff_date = datetime.now() - timedelta(days=days_back)
        signal_count = 0
        total_runs = 0
        
        for run_file in runs_dir.glob("*.json"):
            try:
                with open(run_file, 'r') as f:
                    run_data = json.load(f)
                    
                run_timestamp = datetime.fromisoformat(run_data.get('timestamp', '1970-01-01'))
                if run_timestamp >= cutoff_date:
                    total_runs += 1
                    if run_data.get('trade_recommended', False):
                        signal_count += 1
                    elif 'payloads' in run_data:
                        # Check if any payload would have triggered with lower thresholds
                        for payload in run_data['payloads']:
                            if payload.get('action') in ['BUY', 'SELL']:
                                signal_count += 1
                                break
            except Exception as e:
                logger.warning(f"Error reading {run_file}: {e}")
        
        signals_per_week = (signal_count / max(1, days_back)) * 7
        
        return {
            'signals_per_week': signals_per_week,
            'total_runs': total_runs,
            'signal_count': signal_count,
            'days_analyzed': days_back,
            'analysis': self._get_signal_rate_analysis(signals_per_week)
        }
    
    def _get_signal_rate_analysis(self, signals_per_week: float) -> str:
        """Analyze signal rate and provide recommendation"""
        if signals_per_week < 2:
            return "TOO_LOW - Need to lower thresholds for more validation data"
        elif signals_per_week > 15:
            return "TOO_HIGH - May be generating low-quality signals"
        elif 5 <= signals_per_week <= 10:
            return "OPTIMAL - Good signal rate for validation"
        else:
            return "ACCEPTABLE - Minor adjustments recommended"
    
    def suggest_threshold_adjustments(self) -> Dict[str, Any]:
        """Suggest threshold adjustments for better validation data collection"""
        signal_analysis = self.analyze_current_signal_rate()
        current_rate = signal_analysis['signals_per_week']
        
        adjustments = {
            'current_thresholds': self.current_env,
            'current_signal_rate': current_rate,
            'target_signal_rate': self.target_signals_per_week,
            'recommended_thresholds': {},
            'justification': ''
        }
        
        # Calculate adjustment factor
        if current_rate > 0:
            adjustment_factor = self.target_signals_per_week / current_rate
        else:
            adjustment_factor = 2.0  # Default to more aggressive if no signals
        
        current_sentiment = float(self.current_env['TB_SENTIMENT_CUTOFF'])
        current_confidence = float(self.current_env.get('TB_MIN_CONFIDENCE', '0.65'))
        current_divergence = float(self.current_env.get('TB_DIVERGENCE_THRESHOLD', '0.5'))
        
        if current_rate < self.target_signals_per_week * 0.7:  # Need more signals
            adjustments['recommended_thresholds'] = {
                'TB_SENTIMENT_CUTOFF': str(max(0.4, current_sentiment - 0.05)),
                'TB_MIN_CONFIDENCE': str(max(0.55, current_confidence - 0.05)),
                'TB_DIVERGENCE_THRESHOLD': str(max(0.3, current_divergence - 0.1)),
                'TB_NO_TRADE': '1',  # Keep paper trading mode
                'TB_TRADER_OFFLINE': '0'  # Allow real data
            }
            adjustments['justification'] = f"Lower thresholds to increase signal rate from {current_rate:.1f} to {self.target_signals_per_week}"
            
        elif current_rate > self.target_signals_per_week * 1.3:  # Too many signals
            adjustments['recommended_thresholds'] = {
                'TB_SENTIMENT_CUTOFF': str(min(0.65, current_sentiment + 0.03)),
                'TB_MIN_CONFIDENCE': str(min(0.75, current_confidence + 0.03)),
                'TB_DIVERGENCE_THRESHOLD': str(min(0.7, current_divergence + 0.05)),
                'TB_NO_TRADE': '1',  # Keep paper trading mode
                'TB_TRADER_OFFLINE': '0'  # Allow real data
            }
            adjustments['justification'] = f"Raise thresholds to reduce signal rate from {current_rate:.1f} to {self.target_signals_per_week}"
            
        else:  # Rate is acceptable
            adjustments['recommended_thresholds'] = self.current_env.copy()
            adjustments['justification'] = f"Current signal rate of {current_rate:.1f}/week is acceptable for validation"
        
        return adjustments
    
    def create_optimized_env_file(self, backup_existing: bool = True) -> str:
        """Create optimized .env file for paper trading validation"""
        adjustments = self.suggest_threshold_adjustments()
        
        # Backup existing .env if requested
        if backup_existing and Path('.env').exists():
            backup_name = f".env.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            Path('.env').rename(backup_name)
            logger.info(f"Backed up existing .env to {backup_name}")
        
        # Create optimized .env content
        env_content = f"""# Optimized for Paper Trading Validation Phase
# Generated on {datetime.now().isoformat()}
# Target: {self.target_signals_per_week} signals per week

# PAPER TRADING SAFETY SETTINGS
TB_NO_TRADE=1                    # NEVER REMOVE - Paper trading only
TB_TRADER_OFFLINE=0              # Allow real market data
TB_ENABLE_DISCORD=1              # Enable notifications
TB_NO_TELEGRAM=0                 # Enable Telegram alerts

# OPTIMIZED THRESHOLDS FOR VALIDATION DATA COLLECTION
TB_SENTIMENT_CUTOFF={adjustments['recommended_thresholds'].get('TB_SENTIMENT_CUTOFF', '0.50')}
TB_MIN_CONFIDENCE={adjustments['recommended_thresholds'].get('TB_MIN_CONFIDENCE', '0.60')}
TB_DIVERGENCE_THRESHOLD={adjustments['recommended_thresholds'].get('TB_DIVERGENCE_THRESHOLD', '0.45')}

# ENHANCED FEATURES FOR VALIDATION
TB_USE_ENHANCED_RISK=1
TB_USE_KELLY_SIZING=1
TB_USE_REGIME_DETECTION=1
TB_USE_ENSEMBLE_ML=1
TB_USE_ADAPTIVE_STRATEGY=1

# RISK MANAGEMENT (Conservative for validation)
TB_MAX_RISK_FRAC=0.005           # 0.5% max risk per trade
TB_PORTFOLIO_VAR_LIMIT=0.015     # 1.5% portfolio VaR limit
TB_MAX_CORRELATION=0.6           # Lower correlation limit

# VALIDATION TRACKING
TB_VALIDATION_MODE=1             # Enable validation tracking
TB_LOG_ALL_SIGNALS=1             # Log even non-trading signals

# MULTI-ASSET SETTINGS
TB_MULTI_ASSET=1                 # Enable multi-asset trading
TB_MAX_POSITIONS=3               # Conservative position limit

# Justification: {adjustments['justification']}
"""
        
        with open('.env.validation', 'w') as f:
            f.write(env_content)
        
        logger.info("Created optimized .env.validation file")
        logger.info(f"Recommended thresholds: {adjustments['recommended_thresholds']}")
        logger.info(f"Justification: {adjustments['justification']}")
        
        return env_content
    
    def create_threshold_experiment_script(self) -> str:
        """Create script to experiment with different threshold levels"""
        script_content = """#!/bin/bash
# Threshold Experiment Script for Paper Trading Validation
# Tests different threshold combinations to find optimal validation settings

echo "🧪 Starting Threshold Experiments for Paper Trading Validation"
echo "=============================================================="

# Backup current .env
cp .env .env.backup_experiment

# Define test configurations
declare -a CONFIGS=(
    "0.45,0.55,0.40:relaxed"
    "0.50,0.60,0.45:moderate" 
    "0.55,0.65,0.50:current"
    "0.60,0.70,0.55:conservative"
)

for config in "${CONFIGS[@]}"; do
    IFS=',' read -r sentiment confidence divergence <<< "${config%:*}"
    name="${config##*:}"
    
    echo ""
    echo "🔍 Testing configuration: $name"
    echo "   Sentiment: $sentiment, Confidence: $confidence, Divergence: $divergence"
    
    # Update .env with test configuration
    cat > .env << EOF
TB_SENTIMENT_CUTOFF=$sentiment
TB_MIN_CONFIDENCE=$confidence
TB_DIVERGENCE_THRESHOLD=$divergence
TB_NO_TRADE=1
TB_TRADER_OFFLINE=0
TB_VALIDATION_MODE=1
EOF
    
    # Run trader for 5 cycles
    echo "Running 5 test cycles..."
    for i in {1..5}; do
        python3 scripts/hybrid_crypto_trader.py
        sleep 10
    done
    
    # Analyze results
    python3 validation_analyzer.py > "experiment_${name}_results.txt"
    echo "Results saved to experiment_${name}_results.txt"
done

# Restore original .env
mv .env.backup_experiment .env

echo ""
echo "✅ Threshold experiments complete!"
echo "📊 Review experiment_*_results.txt files to choose optimal configuration"
"""
        
        with open('run_threshold_experiments.sh', 'w') as f:
            f.write(script_content)
        
        # Make executable
        os.chmod('run_threshold_experiments.sh', 0o755)
        
        logger.info("Created threshold experiment script: run_threshold_experiments.sh")
        return script_content

def main():
    """Run paper trading optimization"""
    optimizer = PaperTradingOptimizer()
    
    print("🎯 Paper Trading Threshold Optimization")
    print("=" * 50)
    
    # Analyze current situation
    signal_analysis = optimizer.analyze_current_signal_rate()
    print(f"\n📊 Current Signal Analysis:")
    print(f"   Signals per week: {signal_analysis['signals_per_week']:.1f}")
    print(f"   Total runs analyzed: {signal_analysis['total_runs']}")
    print(f"   Analysis: {signal_analysis['analysis']}")
    
    # Get recommendations
    adjustments = optimizer.suggest_threshold_adjustments()
    print(f"\n🎯 Recommendations:")
    print(f"   Current rate: {adjustments['current_signal_rate']:.1f}/week")
    print(f"   Target rate: {adjustments['target_signal_rate']}/week")
    print(f"   Action: {adjustments['justification']}")
    
    print(f"\n🔧 Recommended Threshold Changes:")
    for key, value in adjustments['recommended_thresholds'].items():
        current = adjustments['current_thresholds'].get(key, 'not set')
        if current != value:
            print(f"   {key}: {current} → {value}")
    
    # Create optimized configuration
    optimizer.create_optimized_env_file()
    optimizer.create_threshold_experiment_script()
    
    print(f"\n✅ Created optimized configuration files:")
    print(f"   .env.validation - Optimized settings for validation phase")
    print(f"   run_threshold_experiments.sh - Script to test different thresholds")
    
    print(f"\n🚀 Next Steps:")
    print(f"   1. Review .env.validation settings")
    print(f"   2. Copy to .env: cp .env.validation .env")
    print(f"   3. Run validation: python3 validation_analyzer.py")
    print(f"   4. Optional: Run experiments: ./run_threshold_experiments.sh")

if __name__ == "__main__":
    main()
