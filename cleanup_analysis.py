#!/usr/bin/env python3
"""
Repository Cleanup Analysis
Identifies files and directories that can be safely removed
"""
import os
from pathlib import Path

def analyze_cleanup_candidates():
    """Analyze repository for cleanup candidates"""
    
    root = Path("/Users/mouryadamarasing/Documents/Project-Tracer-Bullet")
    
    # Core agents and essential files (KEEP)
    core_files = {
        "scripts/hybrid_crypto_trader.py",
        "high_risk_futures_agent.py", 
        "config.py",
        ".env",
        "requirements.txt",
        "README.md",
        "futures_integration.py",
        "world_class_technical_analysis.py",
        "divergence.py",
        "sentiment_utils.py",
        "telegram_bot.py",
        "enhanced_discord_notifications.py",
        "alpaca.py",
        "autocommit.py",
        "market_regime_detector.py"
    }
    
    # Start scripts (KEEP)
    startup_scripts = {
        "scripts/start_hybrid_loop.sh",
        "scripts/start_futures_loop.sh", 
        "start_all_agents.sh",
        "start_simple.sh",
        "stop_trading.sh",
        "check_status.sh"
    }
    
    # Cleanup candidates
    cleanup_candidates = {
        
        # === DOCUMENTATION CLEANUP ===
        "documentation_redundant": [
            "AGENT_STARTUP_GUIDE.md",          # Redundant with README
            "CLEANUP_PLAN.md",                 # Temporary
            "COMPLETE_IMPLEMENTATION_SUMMARY.md", # Outdated
            "COMPLETION_REPORT.md",            # Outdated  
            "COMPREHENSIVE_SYSTEM_ANALYSIS.md", # Outdated
            "DEPLOYMENT_SUMMARY.md",           # Temporary
            "Dev_logs.md",                     # Development notes
            "ENHANCED_SIGNALS_GUIDE.md",       # Redundant
            "ENHANCED_TRADING_SYSTEM_STATUS.md", # Temporary
            "HONEST_SYSTEM_ASSESSMENT.md",     # Assessment doc
            "IMMEDIATE_ACTION_PLAN.md",        # Temporary
            "LATEST_UPDATES.md",               # Changelog (keep recent only)
            "LIVE_LOGS_PERFORMANCE_GUIDE.md",  # Temporary
            "PHASE1_DEPLOYMENT_COMPLETE.md",   # Deployment doc
            "PHASE_1_DEPLOYMENT.md",           # Deployment doc
            "POST_ML_ACTION_PLAN.md",          # Temporary
            "SIMPLE_STARTUP_README.md",        # Redundant with README
            "TESTING_GUIDE.md",                # Development doc
            "WORLD_CLASS_IMPLEMENTATION_COMPLETE.md", # Completion doc
            "about.md",                        # Redundant
            "architecture.md",                 # Outdated
            "env_configuration_complete_status.md", # Temporary
            "glossary.py",                     # Not essential
            "knowledge_wiki.md",               # Development notes
            "roadmap.md",                      # Outdated
            "whitepaper.md",                   # Research doc
            "summary.txt",                     # Temporary
            "EMERGENCY_INTEGRATION_GUIDE.md"   # Temporary guide
        ],
        
        # === INTEGRATION & DEMO FILES ===
        "integration_demos": [
            "CLEAN_INTEGRATION.py",            # Demo file
            "DEMO_INTEGRATION.py",             # Demo file
            "INTEGRATION_EXAMPLE.py",          # Demo file
            "INTEGRATION_GUIDE.py",            # Demo file
            "zero_cost_enhancements.py",       # Enhancement demo
            "system_integration.py",           # Integration example
        ],
        
        # === BACKUP & CONFIG FILES ===
        "backup_configs": [
            ".env.backup",                     # Backup config
            ".env.cleaned",                    # Backup config
            ".env.example",                    # Example config
            ".env.phase1",                     # Phase config
            ".env.validation",                 # Validation config
            "futures_settings_backup.json",    # Backup settings
        ],
        
        # === TESTING & DEVELOPMENT ===
        "testing_development": [
            "tests/",                          # All test files (move to dev branch)
            "debug_sources.py",                # Debug script
            "get_chat_id.py",                  # Setup utility
            "monitor_agents.py",               # Can be replaced with simple scripts
            "validation_analyzer.py",          # Analysis tool
            "generate_performance_report.py",  # Reporting tool
            "view_runs.py",                    # Analysis tool
        ],
        
        # === EXPERIMENTAL & RESEARCH ===
        "experimental": [
            "adaptive_strategy.py",            # Experimental
            "advanced_entry_exit_logic.py",    # Experimental
            "advanced_risk_manager.py",        # Experimental
            "ensemble_ml_models.py",           # ML research
            "local_sentiment_analyzer.py",     # Experimental
            "futures_performance_optimizer.py", # Optimization research
            "futures_risk_reduction.py",       # Research
        ],
        
        # === OUTDATED INTEGRATIONS ===
        "outdated_integrations": [
            "enhanced_data_integration.py",    # Superseded
            "enhanced_trade_manager.py",       # Superseded
            "enhanced_db_logger.py",           # Superseded
            "enhanced_notifications.py",       # Superseded
            "multi_source_data_provider.py",   # Partially superseded
            "config_manager.py",               # Superseded by config.py
            "data_pipeline.py",                # Experimental
            "error_recovery.py",               # Experimental
            "precision_manager.py",            # Experimental
            "system_health.py",                # Experimental
        ],
        
        # === POLYMARKET & ALTERNATIVES ===
        "polymarket_unused": [
            "narrative_analysis.py",           # Polymarket related
            "narrative_analysis_extras.py",    # Polymarket related
            "narrative_dev.py",                # Polymarket related
            "provenance.py",                   # Polymarket related
            "relevance.py",                    # Polymarket related
            "dedupe_utils.py",                 # Polymarket related
            "digest_utils.py",                 # Polymarket related
            "pplx_fetcher.py",                 # Polymarket related
            "pplx_key_manager.py",             # Polymarket related
            "retention.py",                    # Polymarket related
            "source_weights.py",               # Polymarket related
        ],
        
        # === OLD DATA & LOGS ===
        "old_data_logs": [
            "dotusdt_loss_investigation.json", # Investigation file
            "dotusdt_loss_investigation.py",   # Investigation script
            "futures_loss_analysis.json",      # Analysis file
            "*.log",                           # Old log files (keep recent only)
            "*.err",                           # Old error logs
            "enhanced_trading.db",             # Can be recreated
            "validation_analysis.db",          # Analysis database
            "tracer.db",                       # Old database
        ],
        
        # === UTILITIES & SCRIPTS ===
        "utilities_scripts": [
            "bars_stock.py",                   # Stock utility (crypto focus)
            "check_positions.py",              # Utility script
            "close_all_positions.py",          # Emergency script (keep)
            "database_cleanup.py",             # Utility
            "emergency_close_positions.py",    # Emergency script
            "export.py",                       # Export utility
            "export_to_telegram.py",           # Export utility
            "finbert.py",                      # Sentiment utility
            "init_database.py",                # Setup utility
            "real_close_positions.py",         # Emergency script (keep)
            "run.py",                          # Old runner
            "tracer_bullet.py",                # Old main script
            "trading_hours.py",                # Utility
        ],
        
        # === DIRECTORIES ===
        "directories": [
            "backtester/",                     # Backtesting (development)
            "eval_data/",                      # Evaluation data
            "eval_runs/",                      # Evaluation runs  
            "validation_reports/",             # Validation reports
            "underrated_runs/",                # Underrated scanner
            "universe_runs/",                  # Universe data
            "launchd/",                        # macOS services
            "legacy/",                         # Legacy files
            "logs/",                           # Old logs (keep recent only)
            "providers/",                      # Provider implementations
            ".venv/",                          # Virtual environment
        ]
    }
    
    return cleanup_candidates

def recommend_cleanup():
    """Generate cleanup recommendations"""
    candidates = analyze_cleanup_candidates()
    
    print("🧹 REPOSITORY CLEANUP RECOMMENDATIONS")
    print("=" * 50)
    
    total_files = 0
    
    for category, files in candidates.items():
        print(f"\n### 📁 {category.upper().replace('_', ' ')}")
        print(f"Files to remove: {len(files)}")
        total_files += len(files)
        
        for file in files[:5]:  # Show first 5 examples
            print(f"  - {file}")
        
        if len(files) > 5:
            print(f"  ... and {len(files) - 5} more")
    
    print(f"\n🎯 SUMMARY:")
    print(f"Total files/directories for cleanup: {total_files}")
    print(f"Estimated space savings: ~50-70% of repository size")
    
    print(f"\n✅ CORE FILES TO KEEP:")
    core_files = [
        "scripts/hybrid_crypto_trader.py",
        "high_risk_futures_agent.py",
        "futures_integration.py", 
        "world_class_technical_analysis.py",
        "config.py",
        ".env",
        "telegram_bot.py",
        "enhanced_discord_notifications.py",
        "autocommit.py",
        "requirements.txt",
        "README.md",
        "start_all_agents.sh",
        "stop_trading.sh"
    ]
    
    for file in core_files:
        print(f"  ✅ {file}")

if __name__ == "__main__":
    recommend_cleanup()
