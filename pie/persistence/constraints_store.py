"""Append-only constraints store for the Pie Kernel."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Iterable, Tuple

from ..contracts.constraint import ConstraintRecord
from .atomic import atomic_append


def _logical_time_tuple(logical_time: Dict[str, Any]) -> Tuple[int, int, int]:
    return (
        int(logical_time.get("session", 0)),
        int(logical_time.get("turn", 0)),
        int(logical_time.get("idx", 0)),
    )


def compute_constraint_id(
    rule_id: str,
    logical_time: Dict[str, Any],
    trigger_events: Iterable[int],
    effects: List[Dict[str, Any]],
) -> str:
    payload = {
        "rule_id": rule_id,
        "logical_time": logical_time,
        "trigger_events": sorted(list(trigger_events)),
        "effects": effects,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return f"ctr_{digest}"


class ConstraintsStore:
    """Append-only JSONL constraints store."""

    def __init__(self, path: str = "artifacts/constraints.jsonl") -> None:
        self.path = Path(path)

    def reset(self) -> None:
        if self.path.exists():
            self.path.unlink()

    def append(self, record: ConstraintRecord) -> None:
        atomic_append(self.path, json.dumps(record.model_dump()) + "\n")

    def read_all(self) -> List[ConstraintRecord]:
        if not self.path.exists():
            return []
        records: List[ConstraintRecord] = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f.read().splitlines():
                if not line.strip():
                    continue
                obj = json.loads(line)
                records.append(ConstraintRecord.model_validate(obj))
        return records

    def latest(self, *, constraint_id: Optional[str] = None) -> Optional[ConstraintRecord]:
        records = self.read_all()
        if constraint_id:
            records = [rec for rec in records if rec.constraint_id == constraint_id]
        if not records:
            return None
        records.sort(key=lambda rec: _logical_time_tuple(rec.logical_time.model_dump()))
        return records[-1]

    def query_active(self) -> List[ConstraintRecord]:
        records = self.read_all()
        latest_by_id: Dict[str, ConstraintRecord] = {}
        for rec in records:
            current = latest_by_id.get(rec.constraint_id)
            if current is None or _logical_time_tuple(rec.logical_time.model_dump()) > _logical_time_tuple(
                current.logical_time.model_dump()
            ):
                latest_by_id[rec.constraint_id] = rec
        active = [rec for rec in latest_by_id.values() if rec.status == "active"]
        active.sort(key=lambda rec: rec.constraint_id)
        return active

    def append_decision(
        self,
        constraint_id: str,
        status: str,
        *,
        explanation: str,
        logical_time: Optional[Dict[str, int]] = None,
    ) -> ConstraintRecord:
        """Append a decision record that updates a constraint's status."""
        latest = self.latest(constraint_id=constraint_id)
        if latest is None:
            raise ValueError(f"Constraint not found: {constraint_id}")
        if status not in {"active", "rejected"}:
            raise ValueError(f"Unsupported decision status: {status}")
        if logical_time is None:
            logical_time = {
                "session": int(latest.logical_time.session),
                "turn": int(latest.logical_time.turn),
                "idx": int(latest.logical_time.idx) + 1,
            }
        record = ConstraintRecord(
            constraint_id=latest.constraint_id,
            logical_time=logical_time,
            family=latest.family,
            rule_id=latest.rule_id,
            severity=latest.severity,
            status=status,
            commit_policy=latest.commit_policy,
            effects=latest.effects,
            trigger_events=latest.trigger_events,
            explanation=explanation,
            strength=latest.strength,
            decay=latest.decay,
            cooldown=latest.cooldown,
        )
        self.append(record)
        return record
