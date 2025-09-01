#!/usr/bin/env python3
"""
Enhanced Auto-Commit System
Permanent directive: Auto-commit all non-script files, exclude scripts/ and .env always
"""

import subprocess
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('autocommit.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def _run(cmd):
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, err = p.communicate()
    return p.returncode, out.strip(), err.strip()

def in_git_repo() -> bool:
    code, out, _ = _run(["git", "rev-parse", "--is-inside-work-tree"])
    return code == 0 and out == "true"

def ensure_user():
    code, out, _ = _run(["git", "config", "user.email"])
    if code != 0 or not out:
        _run(["git", "config", "user.email", "auto@tracer.local"])
    code, out, _ = _run(["git", "config", "user.name"])
    if code != 0 or not out:
        _run(["git", "config", "user.name", "Tracer Auto Commit"])

def has_changes() -> bool:
    code, out, _ = _run(["git", "status", "--porcelain"])
    return out != ""

def _is_allowed_path(p: str) -> bool:
    """Allow non-script files globally; block scripts directory and sensitive files.
    Permanent directive: Auto-commit all non-script files, exclude scripts/ and .env always
    """
    if not p or not p.strip():
        return False

    # Normalize path
    p_norm = p.replace("\\", "/").strip()

    # Block entire scripts/ directory
    if p_norm.startswith("scripts/") or "/scripts/" in p_norm:
        return False

    # Block sensitive files and directories
    base = os.path.basename(p_norm)
    if (base == ".env" or
        base.startswith(".env.") or
        base == ".env.local" or
        base.startswith(".env.local.") or
        base == ".git" or
        base.startswith(".git/") or
        base == ".venv" or
        base.startswith(".venv/") or
        base == "__pycache__" or
        base.startswith("__pycache__/") or
        base == ".pytest_cache" or
        base.startswith(".pytest_cache/") or
        base == ".DS_Store"):
        return False

    # Block by extension (scripts and executables)
    blocked_ext = (".py", ".sh", ".bash", ".zsh", ".fish", ".ps1", ".bat", ".cmd", ".exe",
                   ".ipynb", ".js", ".ts", ".jsx", ".tsx", ".go", ".rs", ".cpp", ".c",
                   ".java", ".scala", ".php", ".rb", ".pl", ".lua", ".r", ".m", ".swift")
    if p_norm.lower().endswith(blocked_ext):
        return False

    # Allow all other files (docs, data, configs, logs, etc.)
    return True

def get_modified_files() -> list:
    """Get all modified and untracked files that should be auto-committed"""
    if not in_git_repo():
        return []

    # Get modified files
    code, modified_out, _ = _run(["git", "ls-files", "-m", "--exclude-standard"])
    modified_files = modified_out.splitlines() if code == 0 and modified_out else []

    # Get untracked files
    code, untracked_out, _ = _run(["git", "ls-files", "-o", "--exclude-standard"])
    untracked_files = untracked_out.splitlines() if code == 0 and untracked_out else []

    # Combine and filter
    all_files = modified_files + untracked_files
    allowed_files = [f.strip() for f in all_files if f.strip() and _is_allowed_path(f.strip())]

    return sorted(set(allowed_files))

def stage_allowed_files():
    """Stage all allowed files for commit"""
    files_to_stage = get_modified_files()
    if not files_to_stage:
        logger.info("No allowed files to stage")
        return False

    logger.info(f"Staging {len(files_to_stage)} files: {files_to_stage[:5]}{'...' if len(files_to_stage) > 5 else ''}")
    code, out, err = _run(["git", "add"] + files_to_stage)
    if code != 0:
        logger.error(f"Failed to stage files: {err}")
        return False

    return True

def current_branch() -> str:
    code, out, _ = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    return out if code == 0 else "main"

def push(branch: str) -> bool:
    logger.info(f"Pushing to origin/{branch}")
    code, out, err = _run(["git", "push", "origin", branch])
    if code != 0:
        logger.error(f"Push failed: {err}")
        return False
    return True

def auto_commit_and_push(extra_message: str = "", push_enabled: bool = True) -> str:
    """Auto-commit all allowed files and push"""
    if not in_git_repo():
        msg = "Not a git repository; skipped auto-commit."
        logger.warning(msg)
        return msg

    ensure_user()

    # Stage allowed files
    if not stage_allowed_files():
        return "No files to commit."

    if not has_changes():
        return "No changes to commit."

    # Create commit message
    ts = datetime.now(timezone.utc).isoformat()
    msg = f"auto: save non-script files @ {ts}"
    if extra_message:
        msg += f" | {extra_message}"

    logger.info(f"Committing with message: {msg}")

    # Commit
    code, out, err = _run(["git", "commit", "-m", msg])
    if code != 0:
        combined = f"{out}\n{err}".strip()
        if "nothing to commit" in combined.lower() or "no changes added to commit" in combined.lower():
            return "No changes to commit."
        logger.error(f"Commit failed: {combined}")
        return f"Commit failed: {combined}"

    # Push if enabled
    br = current_branch()
    if push_enabled:
        okp = push(br)
        result = "Committed and pushed." if okp else "Committed, push failed."
    else:
        result = "Committed (push disabled)."

    logger.info(result)
    return result

def auto_commit_data_files(target_branch="data", extra_message="") -> str:
    """Auto-commit data/artifacts to separate branch"""
    if not in_git_repo():
        return "Not a git repository; skipped."

    ensure_user()

    # Get data-related files (bars/, runs/, eval_runs/, logs, etc.)
    data_paths = ["bars/", "runs/", "eval_runs/", "state/", "*.log", "*.json", "*.csv", "*.txt"]
    files_to_commit = []

    for path_pattern in data_paths:
        if "*" in path_pattern:
            # Handle glob patterns
            code, out, _ = _run(["git", "ls-files", "-mo", "--exclude-standard", "--", path_pattern])
            if code == 0 and out:
                files_to_commit.extend(out.splitlines())
        else:
            # Handle directory patterns
            code, out, _ = _run(["git", "ls-files", "-mo", "--exclude-standard", path_pattern])
            if code == 0 and out:
                files_to_commit.extend(out.splitlines())

    # Filter to allowed paths only
    allowed_files = [f.strip() for f in files_to_commit if f.strip() and _is_allowed_path(f.strip())]

    if not allowed_files:
        return "No data files to commit."

    logger.info(f"Committing {len(allowed_files)} data files to {target_branch}")

    # Stash current changes
    _run(["git", "stash", "--keep-index", "-u"])

    # Add and commit data files
    _run(["git", "add"] + allowed_files)

    if not has_changes():
        _run(["git", "stash", "pop"])
        return "No data file changes to commit."

    ts = datetime.now(timezone.utc).isoformat()
    msg = f"auto: save data artifacts @ {ts}"
    if extra_message:
        msg += f" | {extra_message}"

    code, out, err = _run(["git", "commit", "-m", msg])
    if code != 0:
        _run(["git", "stash", "pop"])
        combined = f"{out}\n{err}".strip().lower()
        if "nothing to commit" in combined or "no changes added to commit" in combined:
            return "No data file changes to commit."
        return f"Data commit failed: {err or out or 'unknown error'}"

    # Restore stashed changes
    _run(["git", "stash", "pop"])

    # Push data branch
    _run(["git", "push", "origin", target_branch])

    return f"Committed {len(allowed_files)} data files to {target_branch} and pushed."

def run_autocommit_cycle():
    """Run complete auto-commit cycle for both code and data"""
    logger.info("Starting auto-commit cycle")

    # Auto-commit non-script files to main branch
    result1 = auto_commit_and_push("Auto-commit non-script files")

    # Auto-commit data files to data branch
    result2 = auto_commit_data_files("data", "Data artifacts auto-commit")

    return f"Main: {result1} | Data: {result2}"

# CLI interface
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Enhanced Auto-Commit System')
    parser.add_argument('--run', action='store_true', help='Run complete auto-commit cycle')
    parser.add_argument('--main-only', action='store_true', help='Auto-commit non-script files only')
    parser.add_argument('--data-only', action='store_true', help='Auto-commit data files only')
    parser.add_argument('--list', action='store_true', help='List files that would be committed')
    parser.add_argument('--message', type=str, help='Extra commit message')
    parser.add_argument('--no-push', action='store_true', help='Commit without pushing')

    args = parser.parse_args()

    if args.list:
        files = get_modified_files()
        if files:
            print("Files to be auto-committed:")
            for f in files:
                print(f"  {f}")
        else:
            print("No files to commit.")
    elif args.main_only:
        result = auto_commit_and_push(args.message, not args.no_push)
        print(result)
    elif args.data_only:
        result = auto_commit_data_files("data", args.message)
        print(result)
    elif args.run:
        result = run_autocommit_cycle()
        print(result)
    else:
        print("Enhanced Auto-Commit System")
        print("Permanent directive: Auto-commit all non-script files, exclude scripts/ and .env always")
        print("")
        print("Usage:")
        print("  --run        : Run complete auto-commit cycle")
        print("  --main-only  : Auto-commit non-script files only")
        print("  --data-only  : Auto-commit data files only")
        print("  --list       : List files that would be committed")
        print("  --message MSG: Add extra commit message")
        print("  --no-push    : Commit without pushing")
