"""Tests for crystallization constraints and store behavior."""

from __future__ import annotations

import json

from pie.contracts.constraint import ConstraintRecord, ConstraintEffect
from pie.contracts.state import State
from pie.crystallization.engine import CrystallizationEngine
from pie.decisioning import ActionCandidate, enforce_constraints
from pie.persistence.constraints_store import ConstraintsStore, compute_constraint_id
from pie.runtime import run


def _make_record(
    *,
    rule_id: str,
    logical_time: dict,
    trigger_events: list[int],
    status: str,
    commit_policy: str,
) -> ConstraintRecord:
    effects_payload = [{"type": "forbid", "params": {"action_class": "TEST"}}]
    constraint_id = compute_constraint_id(
        rule_id,
        {"session": logical_time["session"], "turn": logical_time["turn"], "idx": 0},
        trigger_events,
        effects_payload,
    )
    return ConstraintRecord(
        constraint_id=constraint_id,
        logical_time=logical_time,
        family="TEST",
        rule_id=rule_id,
        severity="hard",
        status=status,
        commit_policy=commit_policy,
        effects=[ConstraintEffect(**effects_payload[0])],
        trigger_events=trigger_events,
        explanation="test constraint",
        strength="none",
        decay="none",
        cooldown=0,
    )


def test_constraint_id_deterministic() -> None:
    effects_payload = [{"type": "forbid", "params": {"action_class": "TEST"}}]
    logical_time = {"session": 1, "turn": 1, "idx": 0}
    trigger_events = [1, 2]
    a = compute_constraint_id("RULE_X", logical_time, trigger_events, effects_payload)
    b = compute_constraint_id("RULE_X", logical_time, trigger_events, effects_payload)
    assert a == b


def test_crystallizer_pending_status() -> None:
    engine = CrystallizationEngine()
    events = [
        {
            "id": 1,
            "type": "INPUT",
            "content": {"input": "C'e stato un danno, situazione ambigua."},
        }
    ]
    state = State()
    records = engine.propose_constraints(
        events=events,
        state=state,
        logical_time={"session": 1, "turn": 2},
        existing_constraints=[],
    )
    pending = [
        rec
        for rec in records
        if rec.severity == "hard" and rec.commit_policy == "creator_confirm"
    ]
    assert pending
    assert all(rec.status == "pending" for rec in pending)


def test_constraints_store_append_only(tmp_path) -> None:
    path = tmp_path / "constraints.jsonl"
    store = ConstraintsStore(str(path))
    record1 = _make_record(
        rule_id="RULE_1",
        logical_time={"session": 1, "turn": 1, "idx": 1},
        trigger_events=[1],
        status="pending",
        commit_policy="creator_confirm",
    )
    record2 = _make_record(
        rule_id="RULE_2",
        logical_time={"session": 1, "turn": 2, "idx": 1},
        trigger_events=[2],
        status="active",
        commit_policy="auto",
    )
    store.append(record1)
    lines_before = path.read_text(encoding="utf-8").splitlines()
    store.append(record2)
    lines_after = path.read_text(encoding="utf-8").splitlines()
    assert lines_before[0] == lines_after[0]


def test_constraints_store_decision_updates_status(tmp_path) -> None:
    path = tmp_path / "constraints.jsonl"
    store = ConstraintsStore(str(path))
    record = _make_record(
        rule_id="RULE_PENDING",
        logical_time={"session": 1, "turn": 1, "idx": 1},
        trigger_events=[1],
        status="pending",
        commit_policy="creator_confirm",
    )
    store.append(record)
    decision = store.append_decision(
        record.constraint_id,
        "active",
        explanation="approved",
    )
    assert decision.status == "active"
    active = store.query_active()
    assert [rec.constraint_id for rec in active] == [record.constraint_id]


def test_enforcement_forbid_eliminates_candidate() -> None:
    constraint = _make_record(
        rule_id="FORBID_UNSAFE",
        logical_time={"session": 1, "turn": 1, "idx": 1},
        trigger_events=[1],
        status="active",
        commit_policy="auto",
    )
    constraint.effects = [ConstraintEffect(type="forbid", params={"action_class": "UNSAFE_OR_FORBIDDEN"})]
    candidates = [
        ActionCandidate("unsafe_action", "UNSAFE_OR_FORBIDDEN", 0.9),
        ActionCandidate("safe_action", "SAFE_CHAT", 0.5),
    ]
    result = enforce_constraints(candidates, [constraint])
    assert result.after_choice.action_id == "safe_action"


def test_enforcement_require_confirmation_forces_ask_confirm() -> None:
    constraint = _make_record(
        rule_id="CONFIRM_HIGH",
        logical_time={"session": 1, "turn": 1, "idx": 1},
        trigger_events=[1],
        status="active",
        commit_policy="auto",
    )
    constraint.effects = [ConstraintEffect(type="require_confirmation", params={"action_class": "HIGH_IMPACT"})]
    candidates = [ActionCandidate("impact_action", "HIGH_IMPACT", 0.9)]
    result = enforce_constraints(candidates, [constraint])
    assert result.before_choice.action_class == "HIGH_IMPACT"
    assert result.after_choice.action_class == "ASK_CONFIRM"


def test_exam_creates_constraints(tmp_path) -> None:
    out_dir = tmp_path / "artifacts"
    run(exam=True, llm="fake", output_dir=str(out_dir))
    constraints_path = out_dir / "constraints.jsonl"
    trace_path = out_dir / "trace_exam.jsonl"
    assert constraints_path.exists()
    lines = constraints_path.read_text(encoding="utf-8").splitlines()
    assert lines
    records = [ConstraintRecord.model_validate(json.loads(line)) for line in lines if line.strip()]
    assert any(rec.severity == "hard" and rec.status == "active" for rec in records)
    assert any(rec.severity == "hard" and rec.status == "pending" for rec in records)
    trace_lines = trace_path.read_text(encoding="utf-8").splitlines()
    assert any("CONSTRAINT_PROPOSED" in line for line in trace_lines)
    assert any("CONSTRAINT_APPENDED" in line for line in trace_lines)


def test_exam_enforces_constraints(tmp_path) -> None:
    out_dir = tmp_path / "artifacts"
    run(exam=True, llm="fake", output_dir=str(out_dir))
    trace_path = out_dir / "trace_exam.jsonl"
    trace_lines = trace_path.read_text(encoding="utf-8").splitlines()
    enforced = []
    for line in trace_lines:
        event = json.loads(line)
        if event.get("type") != "CONSTRAINT_ENFORCED":
            continue
        content = event.get("content", {})
        before = content.get("before_choice", {})
        after = content.get("after_choice", {})
        if before.get("action_class") == "HIGH_IMPACT" and after.get("action_class") == "ASK_CONFIRM":
            enforced.append(event)
    assert enforced
