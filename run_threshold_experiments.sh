#!/bin/bash
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
