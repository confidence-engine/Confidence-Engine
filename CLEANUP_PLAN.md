# 🧹 CONFIDENCE ENGINE - REPOSITORY CLEANUP PLAN

## FILES TO REMOVE (OBSOLETE/REDUNDANT)

### 🔬 Development & Debug Files
- [ ] `debug_env.py` - Debugging script, not needed in production
- [ ] `inspect_env_pplx.py` - Environment inspection, not needed in production
- [ ] `test_new_key.py` - Key testing script, not needed in production
- [ ] `test_pplx_keys.py` - PPLX key testing, not needed in production
- [ ] `test_pplx_comprehensive.py` - Comprehensive PPLX testing, not needed in production
- [ ] `test_alpaca.py` - Alpaca testing, not needed in production
- [ ] `test_enhanced_hybrid.py` - Testing script, not needed in production
- [ ] `test_hard_cap.py` - Testing script, not needed in production
- [ ] `test_intelligent_tpsl.py` - Testing script, not needed in production
- [ ] `test_multi_source_data.py` - Testing script, not needed in production
- [ ] `test_all_fixes.py` - Testing script, not needed in production

### 📊 Paper Trading & Demo Files
- [ ] `paper_trading_demo.py` - Demo file, not needed for production agents
- [ ] `paper_trading_optimizer.py` - Optimizer, not actively used
- [ ] `tests/paper_trading_demo.py` - Test demo file, redundant

### 📖 Documentation Cleanup
- [ ] `about.md` - Redundant, info covered in README
- [ ] `Dev_logs.md` - Keep but archive old content
- [ ] `Dev_logs_CLEAN.md` - Redundant
- [ ] `Roadmap_CLEAN.md` - Redundant  
- [ ] `README_CLEAN.md` - Redundant
- [ ] `polymarket_digest.md` - Polymarket not used by current agents
- [ ] `polymarket_xolution.md` - Polymarket not used by current agents
- [ ] `polymarket.md` - Polymarket not used by current agents
- [ ] `slaydragon.md` - Outdated docs
- [ ] `toDo.md` - Outdated task list
- [ ] `tracer_bullet_about.md` - Redundant
- [ ] `tests/tracer_bullet_about.md` - Redundant
- [ ] `tests/learning_immersion.md` - Not needed
- [ ] `tests/whitepaper.md` - Redundant with main whitepaper.md

### 🚫 Unused/Obsolete Features
- [ ] `polymarket_fetcher.py` - Polymarket not used by current agents
- [ ] `coindesk_rss.py` - RSS not actively used
- [ ] `perplexity_fetcher.py` - Replaced by pplx_fetcher.py
- [ ] `futures_paper_trading_demo.py` - Demo file, not needed
- [ ] `futures_trading_platform.py` - Replaced by futures_integration.py
- [ ] `manual_position_manager.py` - Not used by automated agents
- [ ] `auto_position_monitor.py` - Functionality integrated into main agents
- [ ] `dual_agent_monitor.py` - Not actively used
- [ ] `dual_agent.sh` - Not actively used
- [ ] `industrial_upgrade_plan.py` - Not needed for current agents
- [ ] `intelligent_futures_manager.py` - Functionality integrated
- [ ] `autocommit_enhanced.py` - Redundant with autocommit.py

### 🗂️ Old Configuration/Setup Files
- [ ] `setup_autocommit_cron.sh` - Old setup script
- [ ] `setup_enhanced_agent.sh` - Old setup script  
- [ ] `setup_futures_watchdog.sh` - Old setup script
- [ ] `futures_agent_config.env` - Old config
- [ ] `monitoring_status.sh` - Not actively used
- [ ] `periodic_autocommit.sh` - Old script
- [ ] `cleanup_repo.sh` - Old cleanup script

### 📊 Analysis Tools (Keep Most Recent Only)
- [ ] `explain.py` - Analysis tool, not essential for agents
- [ ] `alpha_summary.py` - Analysis tool, not essential for agents
- [ ] `validation_analyzer.py` - Keep, but clean up old validation reports

## DIRECTORIES TO CLEAN UP

### 🗄️ Archive Old Run Data (Keep Recent Only)
- [ ] `runs/` - Keep last 50 runs, archive older ones
- [ ] `universe_runs/` - Keep last 20 runs, archive older ones
- [ ] `underrated_runs/` - Keep last 10 runs, archive older ones  
- [ ] `validation_reports/` - Keep last 20 reports, archive older ones

### 🧪 Test Files (Keep Essential Only)
- Keep core test files that validate agent functionality
- Remove obsolete test files for unused features

## FILES TO KEEP (ESSENTIAL FOR AGENTS)

### 🤖 Core Agent Files
- ✅ `scripts/hybrid_crypto_trader.py` - Main hybrid agent
- ✅ `high_risk_futures_agent.py` - Main futures agent
- ✅ `futures_integration.py` - Futures trading integration
- ✅ `config.py` - Configuration management
- ✅ `autocommit.py` - Git automation
- ✅ `telegram_bot.py` - Telegram notifications
- ✅ All essential utility files (sentiment_utils.py, etc.)

### 📊 Core Analysis Files  
- ✅ `tracer_bullet.py` - Core trading logic
- ✅ `divergence.py` - Divergence analysis
- ✅ `confirmation.py` - Signal confirmation
- ✅ `narrative_dev.py` - Narrative analysis
- ✅ `market_regime_detector.py` - Market regime detection

### ⚙️ Essential Configuration
- ✅ `.env` - Cleaned production configuration
- ✅ `requirements.txt` - Dependencies
- ✅ Essential documentation (README.md, CONTRIBUTING.md, etc.)

## ESTIMATED SPACE SAVINGS
- **Run Data**: ~500MB → ~50MB (90% reduction)
- **Test Files**: ~100 files → ~30 files (70% reduction)  
- **Documentation**: ~50 files → ~15 files (70% reduction)
- **Obsolete Code**: ~80 files → ~0 files (100% reduction)

**Total Cleanup**: ~150MB space savings, ~200 fewer files
