from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Tuple


logger = logging.getLogger(__name__)


def prune_directory(target_dir: str | os.PathLike, keep: int = 500) -> Tuple[List[Path], List[Path]]:
    """
    Keep the most-recently-modified `keep` files in `target_dir`, delete older files.

    - Only affects regular files (skips subdirectories)
    - Sorting is by mtime descending (most recent first)
    - Returns (kept_paths, deleted_paths)
    """
    d = Path(target_dir)
    if not d.exists() or not d.is_dir():
        return [], []

    files = [p for p in d.iterdir() if p.is_file()]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    kept = files[: max(keep, 0)]
    to_delete = files[max(keep, 0) :]

    deleted: List[Path] = []
    for p in to_delete:
        try:
            p.unlink(missing_ok=True)  # py3.8+: ignore if already gone
            deleted.append(p)
        except Exception as e:
            logger.warning(f"[retention] failed to delete {p}: {e}")

    if deleted:
        logger.info(
            "[retention] pruned %d files from %s (kept=%d)", len(deleted), str(d), len(kept)
        )
    return kept, deleted


def prune_artifacts(keep: int = 500) -> None:
    """
    Prune artifact directories used by the pipeline.
    Currently: `runs/` and `bars/`.
    Keep last `keep` files per directory by modification time.
    """
    prune_directory("runs", keep=keep)
    prune_directory("bars", keep=keep)


