import os
import time
from pathlib import Path
from retention import prune_directory


def test_prune_directory_keeps_last_n(monkeypatch, tmp_path):
    d = tmp_path / "runs"
    d.mkdir()
    # Create 10 files with increasing mtime
    paths = []
    for i in range(10):
        p = d / f"file_{i}.json"
        p.write_text("{}", encoding="utf-8")
        # Make sure mtime increases
        ts = time.time() + i
        os.utime(p, (ts, ts))
        paths.append(p)

    kept, deleted = prune_directory(d, keep=3)

    # Expect the last 3 by mtime to be kept: file_9, file_8, file_7
    kept_names = {p.name for p in kept}
    assert kept_names == {"file_9.json", "file_8.json", "file_7.json"}
    # Deleted should be 7 files
    assert len(deleted) == 7
    # Files actually removed from disk
    for p in deleted:
        assert not p.exists()


