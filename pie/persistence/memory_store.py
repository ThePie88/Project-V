"""Append-only memory store for the Pie Kernel."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Iterable, Tuple

from ..contracts.memory import MemoryRecord
from .atomic import atomic_append


def _logical_time_tuple(logical_time: Dict[str, Any]) -> Tuple[int, int, int]:
    return (
        int(logical_time.get("session", 0)),
        int(logical_time.get("turn", 0)),
        int(logical_time.get("idx", 0)),
    )


def compute_memory_id(
    logical_time: Dict[str, Any],
    record_type: str,
    content: Dict[str, Any],
    source_refs: Iterable[int],
) -> str:
    payload = {
        "logical_time": logical_time,
        "type": record_type,
        "content": content,
        "source_refs": list(source_refs),
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return f"mem_{digest}"


class MemoryStore:
    """Append-only JSONL memory store."""

    def __init__(self, path: str = "artifacts/memory.jsonl") -> None:
        self.path = Path(path)

    def reset(self) -> None:
        if self.path.exists():
            self.path.unlink()

    def append(self, record: MemoryRecord) -> None:
        atomic_append(self.path, json.dumps(record.model_dump()) + "\n")

    def read_all(self) -> List[MemoryRecord]:
        if not self.path.exists():
            return []
        records: List[MemoryRecord] = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f.read().splitlines():
                if not line.strip():
                    continue
                obj = json.loads(line)
                records.append(MemoryRecord.model_validate(obj))
        return records

    def query(
        self,
        *,
        type: Optional[str] = None,
        tags: Optional[List[str]] = None,
        since_logical_time: Optional[Dict[str, Any]] = None,
    ) -> List[MemoryRecord]:
        records = self.read_all()
        if type:
            records = [rec for rec in records if rec.type == type]
        if tags:
            tag_set = set(tags)
            records = [
                rec for rec in records if tag_set.issubset(set(rec.tags or []))
            ]
        if since_logical_time:
            since_tuple = _logical_time_tuple(since_logical_time)
            records = [
                rec
                for rec in records
                if _logical_time_tuple(rec.logical_time.model_dump()) >= since_tuple
            ]
        return records

    def latest(self, *, type: Optional[str] = None) -> Optional[MemoryRecord]:
        records = self.query(type=type) if type else self.read_all()
        return records[-1] if records else None
