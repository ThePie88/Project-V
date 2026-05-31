"""Tests for memory contracts, store, policy, and view."""

from __future__ import annotations

import json

import pytest

from pie.contracts.memory import MemoryRecord
from pie.persistence.memory_store import MemoryStore, compute_memory_id
from pie.memory.policy import propose_memory_writes
from pie.memory.view import build_memory_view


def _make_record(record_type: str, content: dict, source_refs: list[int], logical_time: dict) -> MemoryRecord:
    memory_id = compute_memory_id(logical_time, record_type, content, source_refs)
    return MemoryRecord(
        memory_id=memory_id,
        logical_time=logical_time,
        type=record_type,
        content=content,
        source_refs=source_refs,
        tags=[],
        context={},
    )


def test_memory_record_requires_source_refs() -> None:
    with pytest.raises(ValueError):
        MemoryRecord(
            memory_id="mem_test",
            logical_time={"session": 1, "turn": 1, "idx": 1},
            type="Preference",
            content={"preference": "brevity", "value": "low"},
            source_refs=[],
            tags=[],
            context={},
        )


def test_memory_id_deterministic() -> None:
    logical_time = {"session": 1, "turn": 2, "idx": 1}
    content = {"preference": "brevity", "value": "low"}
    source_refs = [1, 2]
    mem_a = compute_memory_id(logical_time, "Preference", content, source_refs)
    mem_b = compute_memory_id(logical_time, "Preference", content, source_refs)
    assert mem_a == mem_b


def test_memory_store_append_only(tmp_path) -> None:
    path = tmp_path / "memory.jsonl"
    store = MemoryStore(str(path))
    record1 = _make_record(
        "Preference",
        {"preference": "brevity", "value": "low"},
        [1],
        {"session": 1, "turn": 1, "idx": 1},
    )
    record2 = _make_record(
        "Preference",
        {"preference": "tone", "value": "neutral"},
        [2],
        {"session": 1, "turn": 2, "idx": 1},
    )
    store.append(record1)
    lines_before = path.read_text(encoding="utf-8").splitlines()
    store.append(record2)
    lines_after = path.read_text(encoding="utf-8").splitlines()
    assert lines_before[0] == lines_after[0]


def test_memory_query_order_stable(tmp_path) -> None:
    path = tmp_path / "memory.jsonl"
    store = MemoryStore(str(path))
    record1 = _make_record(
        "Belief",
        {"claim": "A", "confidence": 0.5},
        [1],
        {"session": 1, "turn": 1, "idx": 1},
    )
    record2 = _make_record(
        "Belief",
        {"claim": "B", "confidence": 0.4},
        [2],
        {"session": 1, "turn": 2, "idx": 1},
    )
    store.append(record1)
    store.append(record2)
    results = store.query(type="Belief")
    assert [rec.content["claim"] for rec in results] == ["A", "B"]


def test_policy_preference_brevity() -> None:
    context = {
        "logical_time": {"session": 1, "turn": 1},
        "user_input": "Preferisco risposte brevi.",
        "source_refs": [1],
    }
    records = propose_memory_writes(context)
    assert len(records) == 1
    assert records[0].type == "Preference"
    assert records[0].content["preference"] == "brevity"


def test_memory_view_deterministic() -> None:
    record = _make_record(
        "Preference",
        {"preference": "brevity", "value": "low"},
        [1],
        {"session": 1, "turn": 1, "idx": 1},
    )
    view_a = build_memory_view([record]).to_dict()
    view_b = build_memory_view([record]).to_dict()
    assert json.dumps(view_a, sort_keys=True) == json.dumps(view_b, sort_keys=True)
