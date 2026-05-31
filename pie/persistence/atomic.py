"""Atomic file operations for crash-safe persistence (V1.3)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List


def atomic_write(path: Path, data: str) -> None:
    """Write data to *path* atomically via temp file + fsync + rename.

    The sequence ``write tmp → fsync → rename`` ensures that *path*
    always contains either the old content or the new content, never a
    partial write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(data, encoding="utf-8")
    with open(str(tmp), "r+b") as f:
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def atomic_append(path: Path, line: str) -> None:
    """Append *line* to *path* with an explicit fsync.

    For append-only JSONL stores the file is opened in append mode so
    the OS guarantees the write is placed at the end.  The explicit
    ``fsync`` ensures the data reaches stable storage before the call
    returns.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(line)
        f.flush()
        os.fsync(f.fileno())


def recover_tmp_files(directory: Path) -> List[str]:
    """Remove orphan ``.tmp`` files left by interrupted atomic writes.

    Returns the list of removed file names for audit/reporting.
    """
    recovered: List[str] = []
    for tmp in sorted(directory.glob("*.tmp")):
        tmp.unlink()
        recovered.append(tmp.name)
    return recovered


def validate_jsonl_integrity(path: Path) -> List[str]:
    """Check that every line in a JSONL file is valid JSON.

    Returns a list of error descriptions (empty means the file is OK).
    """
    import json

    errors: List[str] = []
    if not path.exists():
        return errors
    for idx, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"line {idx}: {exc}")
    return errors
