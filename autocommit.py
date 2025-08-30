import subprocess
from datetime import datetime, timezone

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
    """Allow non-code files globally; block code extensions and sensitive secrets.
    Block extensions: .py, .sh, .ipynb, .js, .ts, .go, .rs
    Block filenames: .env, .env.*, .env.local
    """
    import os
    p = p.strip()
    if not p:
        return False
    # Normalize
    p_norm = p.replace("\\", "/")
    # Block by extension
    blocked_ext = (".py", ".sh", ".ipynb", ".js", ".ts", ".go", ".rs")
    if p_norm.lower().endswith(blocked_ext):
        return False
    # Block by sensitive filenames
    base = os.path.basename(p_norm)
    if base == ".env" or base.startswith(".env.") or base == ".env.local":
        return False
    # Otherwise allow
    return True

def stage_paths(paths):
    """Stage only allowed files. If a directory is provided, expand to modified/untracked files under it.
    Supports repo-wide usage by passing paths like ['.'].
    """
    import os
    files_to_add = []
    for p in paths:
        if not p:
            continue
        p = p.strip()
        if not p:
            continue
        # Determine if path is file or directory
        if os.path.isdir(p):
            # Use git to list modified/untracked files respecting .gitignore
            code, out, _ = _run(["git", "ls-files", "-mo", "--exclude-standard", p])
            if code == 0 and out:
                for line in out.splitlines():
                    fp = line.strip()
                    if _is_allowed_path(fp):
                        files_to_add.append(fp)
        else:
            # Single file or glob-like path; rely on git add to expand, but filter explicit file first if exists
            if os.path.exists(p):
                if _is_allowed_path(p):
                    files_to_add.append(p)
            else:
                # Try to expand via git ls-files for patterns
                code, out, _ = _run(["bash", "-lc", f"git ls-files -mo --exclude-standard -- {p}"])
                if code == 0 and out:
                    for line in out.splitlines():
                        fp = line.strip()
                        if _is_allowed_path(fp):
                            files_to_add.append(fp)
    # Deduplicate
    files_to_add = sorted(set(files_to_add))
    if not files_to_add:
        return
    _run(["git", "add"] + files_to_add)

def current_branch() -> str:
    code, out, _ = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    return out if code == 0 else "main"

def push(branch: str) -> bool:
    code, _, _ = _run(["git", "push", "origin", branch])
    return code == 0

def auto_commit_and_push(paths, extra_message: str = "", push_enabled: bool = True) -> str:
    if not in_git_repo():
        return "Not a git repository; skipped auto-commit."
    ensure_user()
    stage_paths(paths)
    if not has_changes():
        return "No changes to commit."
    ts = datetime.now(timezone.utc).isoformat()
    msg = f"auto: save tracer run @ {ts}"
    if extra_message:
        msg += f" | {extra_message}"
    code, out, err = _run(["git", "commit", "-m", msg])
    if code != 0:
        combined = f"{out}\n{err}".strip()
        if "nothing to commit" in combined.lower() or "no changes added to commit" in combined.lower():
            return "No changes to commit."
        return "Commit failed."
    br = current_branch()
    if push_enabled:
        okp = push(br)
        return "Committed and pushed." if okp else "Committed, push failed."
    return "Committed (push disabled)."

def auto_commit_to_branch(paths, target_branch="data", extra_message="") -> str:
    if not in_git_repo():
        return "Not a git repository; skipped."
    ensure_user()

    # Stash current changes not in our paths (avoid dirty state issues)
    _run(["git", "stash", "--keep-index", "-u"])

    # Filter to allowlisted-safe paths only
    safe_paths = [p for p in paths if _is_allowed_path(p)]
    if not safe_paths:
        _run(["git", "stash", "pop"])  # restore and exit
        return "No allowed paths to commit for data branch."

    # Create a temporary commit on a detached tree for these paths
    _run(["git", "add"] + safe_paths)
    if not has_changes():
        _run(["git", "stash", "pop"])
        return "No changes to commit for data branch."

    ts = datetime.now(timezone.utc).isoformat()
    msg = f"auto: save tracer run @ {ts}"
    if extra_message:
        msg += f" | {extra_message}"

    # Commit on current branch (temporary)
    code, out, err = _run(["git", "commit", "-m", msg])
    if code != 0:
        _run(["git", "stash", "pop"])
        combined = f"{out}\n{err}".strip().lower()
        if "nothing to commit" in combined or "no changes added to commit" in combined:
            return "No changes to commit for data branch."
        return f"Commit failed: {err or out or 'unknown error'}"

    # Get the commit hash
    _, commit_hash, _ = _run(["git", "rev-parse", "HEAD"])

    # Restore stashed changes to keep dev workflow intact
    _run(["git", "stash", "pop"])

    # Push just this commit’s paths to the target branch by merging the tree
    # Strategy: fetch target, create a worktree ref, merge-tree, and push.
    # Simpler approach: use subtree split-style push for the paths is complex.
    # Practical approach: create a worktree to target_branch, copy files, commit, push.

    # Ensure remote branch exists
    _run(["git", "fetch", "origin", target_branch])
    code, _, _ = _run(["git", "rev-parse", "--verify", f"origin/{target_branch}"])
    if code != 0:
        # Create the branch on origin if missing
        _run(["git", "branch", target_branch])
        _run(["git", "push", "origin", target_branch])

    # Create a temporary worktree for target branch
    worktree_dir = ".git-data-work"
    _run(["rm", "-rf", worktree_dir])
    code, _, err = _run(["git", "worktree", "add", worktree_dir, target_branch])
    if code != 0:
        return f"Worktree error: {err}"

    # Copy files into worktree
    import os, shutil
    try:
        for p in safe_paths:
            if not os.path.exists(p):
                continue
            dst = os.path.join(worktree_dir, p)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if os.path.isdir(p):
                if os.path.exists(dst):
                    shutil.rmtree(dst)
                shutil.copytree(p, dst)
            else:
                os.makedirs(os.path.dirname(dst) or worktree_dir, exist_ok=True)
                shutil.copy2(p, dst)

        # Commit in worktree: add only the copied safe paths
        add_list = []
        for p in safe_paths:
            add_list.append(p)
        _run(["git", "-C", worktree_dir, "add"] + add_list)
        code, _, _ = _run(["git", "-C", worktree_dir, "status", "--porcelain"])
        if code == 0:
            # Make commit only if changes
            stat_code, stat_out, _ = _run(["git", "-C", worktree_dir, "status", "--porcelain"])
            if stat_out:
                _run(["git", "-C", worktree_dir, "commit", "-m", msg])

        # Push target branch
        _run(["git", "-C", worktree_dir, "push", "origin", target_branch])
    finally:
        _run(["git", "worktree", "remove", "--force", worktree_dir])

    return f"Committed artifacts to {target_branch} and pushed."
