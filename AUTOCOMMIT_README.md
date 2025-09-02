# Enhanced Auto-Commit System

## Permanent Directive
**Auto-commit all non-script files, exclude scripts/ and .env always**

## Overview
This system automatically commits and pushes all non-script files while strictly excluding:
- Entire `scripts/` directory
- All `.env` files and variants
- Python, shell, and other script files
- Sensitive system files

## 🚀 Current Status (September 2, 2025)
- **✅ Enhanced Auto-Commit**: Active and running with both agents
- **✅ Artifact Management**: Automatically commits runs/, bars/, logs/, eval_runs/
- **✅ Security**: Never commits .py files, .env files, or sensitive data
- **✅ Git Integration**: Seamless integration with repository management
- **✅ Error Handling**: Robust error handling and recovery mechanisms

## Files Auto-Committed
✅ Documentation (`.md`, `.txt`, `.rst`)
✅ Data files (`.json`, `.csv`, `.txt`)
✅ Logs (`.log`)
✅ Configuration files
✅ Artifacts and outputs
✅ All other non-executable files

## Files Excluded
❌ `scripts/` directory (entire)
❌ `.env`, `.env.*`, `.env.local`
❌ `.py`, `.sh`, `.js`, `.ts`, etc.
❌ `.git/`, `.venv/`, `__pycache__/`
❌ System files (`.DS_Store`, etc.)

## Usage

### Command Line
```bash
# List files that would be committed
python3 autocommit_enhanced.py --list

# Run complete auto-commit cycle
python3 autocommit_enhanced.py --run

# Auto-commit non-script files only
python3 autocommit_enhanced.py --main-only

# Auto-commit data files only
python3 autocommit_enhanced.py --data-only

# Add custom commit message
python3 autocommit_enhanced.py --run --message "Custom message"

# Commit without pushing
python3 autocommit_enhanced.py --run --no-push
```

### Automation

#### Git Hook (Automatic)
Runs after every git commit:
```bash
# Already configured in .git/hooks/post-commit
# Triggers automatically on every commit
```

#### Periodic Cron Job
```bash
# Setup cron job (runs every 30 minutes)
./setup_autocommit_cron.sh

# Manual periodic run
./periodic_autocommit.sh
```

#### View Cron Jobs
```bash
crontab -l
```

## System Architecture

### Main Components
- `autocommit_enhanced.py` - Core auto-commit logic
- `.git/hooks/post-commit` - Git post-commit hook
- `periodic_autocommit.sh` - Periodic execution script
- `setup_autocommit_cron.sh` - Cron setup utility

### Dual Branch System
- **Main Branch**: Documentation, configs, general files
- **Data Branch**: Data artifacts, logs, run outputs

### Logging
- All operations logged to `autocommit.log`
- Cron jobs logged to `autocommit_cron.log`

## Examples

### Typical Auto-Commit Output
```
2025-09-01 18:21:20,389 [INFO] Staging 1 files: ['test_integration_report.json']
2025-09-01 18:21:20,463 [INFO] Committing with message: auto: save non-script files @ 2025-09-01T12:51:20.463102+00:00
2025-09-01 18:21:20,928 [INFO] Committed (push disabled).
```

### Files Detected for Commit
```
Files to be auto-committed:
  Dev_logs.md
  test_integration_report.json
  autocommit.log
  bars/data.csv
  runs/trade_log.json
```

## Status
✅ **Active and Operational**
- Git hook installed and working
- Periodic script ready for cron
- Comprehensive file filtering active
- Tested and validated
