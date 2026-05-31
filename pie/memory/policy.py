"""Deterministic memory policy for the Pie Kernel."""

from __future__ import annotations

from typing import Any, Dict, List

from ..contracts.memory import MemoryRecord
from ..persistence.memory_store import compute_memory_id


def propose_memory_writes(context: Dict[str, Any]) -> List[MemoryRecord]:
    """Propose memory records based on a deterministic policy."""
    user_input = (context.get("user_input") or "").strip()
    logical_time = context.get("logical_time") or {"session": 0, "turn": 0}
    source_refs = context.get("source_refs") or []
    if not user_input:
        return []
    records: List[MemoryRecord] = []
    lower = user_input.lower()

    def _make_record(record_type: str, content: Dict[str, Any], tags: List[str]) -> MemoryRecord:
        idx = len(records) + 1
        record_time = {
            "session": int(logical_time.get("session", 0)),
            "turn": int(logical_time.get("turn", 0)),
            "idx": idx,
        }
        memory_id = compute_memory_id(record_time, record_type, content, source_refs)
        return MemoryRecord(
            memory_id=memory_id,
            logical_time=record_time,
            type=record_type,
            content=content,
            source_refs=source_refs,
            tags=tags,
            context={},
        )

    # Preference: brevity
    if "preferisco" in lower and ("breve" in lower or "brevi" in lower):
        records.append(
            _make_record(
                "Preference",
                {"preference": "brevity", "value": "low"},
                ["preference", "verbosity"],
            )
        )

    # Narrative triggers (simple v0): identity statements
    narrative_triggers = ["mi chiamo", "sono ", "ricorda che"]
    if any(trigger in lower for trigger in narrative_triggers):
        records.append(
            _make_record(
                "NarrativeMemory",
                {"text": user_input},
                ["narrative"],
            )
        )

    # Beliefs (simple v0): explicit claims with confidence
    belief_triggers = ["credo che", "penso che"]
    if any(trigger in lower for trigger in belief_triggers):
        records.append(
            _make_record(
                "Belief",
                {"claim": user_input, "confidence": 0.6},
                ["belief"],
            )
        )

    # Trust updates (simple v0)
    if "non mi fido" in lower:
        records.append(
            _make_record(
                "TrustUpdate",
                {"entity": "user", "delta": -0.1},
                ["trust"],
            )
        )
    elif "mi fido" in lower:
        records.append(
            _make_record(
                "TrustUpdate",
                {"entity": "user", "delta": 0.1},
                ["trust"],
            )
        )

    return records
