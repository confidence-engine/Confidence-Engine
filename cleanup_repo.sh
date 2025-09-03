#!/bin/bash
# Repository Cleanup Script - Remove unessential files
# September 3, 2025

echo "🧹 Starting Repository Cleanup"
echo "=============================="

# 1. Remove large log files (they regenerate)
echo "1. Removing large log files..."
rm -f trading_agent.log
rm -f advanced_risk_manager.log
rm -f intelligent_futures.log
rm -f autocommit.log
echo "   ✅ Removed large log files"

# 2. Remove demo files that are not actively used
echo "2. Removing unused demo files..."
rm -f futures_paper_trading_demo.py
rm -f paper_trading_demo.py
echo "   ✅ Removed demo files"

# 3. Remove polymarket preview files (temporary)
echo "3. Removing polymarket preview files..."
rm -f polymarket_discord_preview.txt
rm -f polymarket_tg_preview.txt
rm -f polymarket_discord_preview_empty.txt
rm -f polymarket_tg_preview_empty.txt
echo "   ✅ Removed polymarket preview files"

# 4. Remove test files in root (should be in tests/ directory)
echo "4. Removing misplaced test files..."
rm -f test_pplx_comprehensive.py
rm -f test_new_key.py
rm -f test_pplx_keys.py
rm -f test_alpaca.py
rm -f test_enhanced_hybrid.py
rm -f test_hard_cap.py
rm -f test_multi_source_data.py
echo "   ✅ Removed misplaced test files"

# 5. Remove integration test artifacts
echo "5. Removing integration test artifacts..."
rm -f integration_test_results_*_*.json
rm -f test_integration_report.json
echo "   ✅ Removed integration test artifacts"

# 6. Remove experimental/deprecated files
echo "6. Removing experimental files..."
rm -f run_threshold_experiments.sh
rm -f setup_enhanced_agent.sh
echo "   ✅ Removed experimental files"

# 7. Clean up empty log files
echo "7. Removing empty log files..."
find . -name "*.log" -type f -size 0 -delete
echo "   ✅ Removed empty log files"

# 8. Remove old environment files
echo "8. Removing old environment files..."
rm -f .env.example
echo "   ✅ Removed environment examples"

# 9. Clean up old validation reports (keep last 50)
echo "9. Cleaning old validation reports..."
VALIDATION_COUNT=$(ls validation_reports/*.json 2>/dev/null | wc -l)
if [ "$VALIDATION_COUNT" -gt 50 ]; then
    ls -t validation_reports/*.json | tail -n +51 | xargs rm -f
    echo "   ✅ Kept latest 50 validation reports, removed $(($VALIDATION_COUNT - 50))"
else
    echo "   ✅ Validation reports count OK ($VALIDATION_COUNT)"
fi

# 10. Remove pytest cache (will regenerate)
echo "10. Removing pytest cache..."
rm -rf .pytest_cache
echo "   ✅ Removed pytest cache"

echo ""
echo "🎯 Cleanup Summary:"
echo "   - Removed large log files (will regenerate)"
echo "   - Removed unused demo files"
echo "   - Removed temporary preview files"
echo "   - Removed misplaced test files"
echo "   - Removed integration test artifacts"
echo "   - Removed experimental scripts"
echo "   - Cleaned old validation reports (kept latest 50)"
echo "   - Removed pytest cache"
echo "   - Kept essential documentation (CLEAN versions)"
echo "   - Kept active test suite in tests/ directory"
echo "   - Kept essential configuration files"

echo ""
echo "✅ Repository cleanup complete!"
echo "📊 Use 'git status' to see what was removed"
