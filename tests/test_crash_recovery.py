"""V1.3 — Atomic commit + crash recovery tests.

Covers:
- atomic_write: temp → fsync → rename (no orphan .tmp)
- atomic_append: fsync on append-only stores
- recover_tmp_files: orphan .tmp cleanup
- validate_jsonl_integrity: corrupt line detection
- MemoryStore / ConstraintsStore atomic append
- E-EXAM-CRASH-001: crash simulation → recovery → exam OK → replay OK
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pie.persistence.atomic import (
    atomic_write,
    atomic_append,
    recover_tmp_files,
    validate_jsonl_integrity,
)
from pie.persistence.memory_store import MemoryStore
from pie.persistence.constraints_store import ConstraintsStore
from pie.contracts.memory import MemoryRecord
from pie.contracts.constraint import ConstraintRecord


# ---------------------------------------------------------------------------
# atomic_write
# ---------------------------------------------------------------------------


def test_atomic_write_creates_no_tmp(tmp_path: Path) -> None:
    target = tmp_path / "output.json"
    atomic_write(target, '{"key": "value"}')
    assert target.exists()
    remaining = list(tmp_path.glob("*.tmp"))
    assert remaining == [], f"Orphan .tmp files found: {remaining}"


def test_atomic_write_content_correct(tmp_path: Path) -> None:
    target = tmp_path / "data.json"
    data = json.dumps({"hello": "world"}, indent=2)
    atomic_write(target, data)
    assert target.read_text(encoding="utf-8") == data


def test_atomic_write_overwrites_existing(tmp_path: Path) -> None:
    target = tmp_path / "file.txt"
    atomic_write(target, "first")
    atomic_write(target, "second")
    assert target.read_text(encoding="utf-8") == "second"


def test_atomic_write_creates_parent_dirs(tmp_path: Path) -> None:
    target = tmp_path / "sub" / "dir" / "file.json"
    atomic_write(target, "{}")
    assert target.exists()


# ---------------------------------------------------------------------------
# atomic_append
# ---------------------------------------------------------------------------


def test_atomic_append_creates_file(tmp_path: Path) -> None:
    target = tmp_path / "log.jsonl"
    atomic_append(target, '{"line": 1}\n')
    assert target.exists()
    assert target.read_text(encoding="utf-8") == '{"line": 1}\n'


def test_atomic_append_appends(tmp_path: Path) -> None:
    target = tmp_path / "log.jsonl"
    atomic_append(target, '{"a": 1}\n')
    atomic_append(target, '{"b": 2}\n')
    lines = target.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2


# ---------------------------------------------------------------------------
# recover_tmp_files
# ---------------------------------------------------------------------------


def test_recover_tmp_files_removes_orphans(tmp_path: Path) -> None:
    (tmp_path / "snapshot.json.tmp").write_text("partial", encoding="utf-8")
    (tmp_path / "trace.jsonl.tmp").write_text("partial", encoding="utf-8")
    (tmp_path / "real_file.json").write_text("{}", encoding="utf-8")
    recovered = recover_tmp_files(tmp_path)
    assert len(recovered) == 2
    assert not list(tmp_path.glob("*.tmp"))
    assert (tmp_path / "real_file.json").exists()


def test_recover_tmp_files_empty_dir(tmp_path: Path) -> None:
    recovered = recover_tmp_files(tmp_path)
    assert recovered == []


# ---------------------------------------------------------------------------
# validate_jsonl_integrity
# ---------------------------------------------------------------------------


def test_validate_jsonl_integrity_valid(tmp_path: Path) -> None:
    path = tmp_path / "trace.jsonl"
    path.write_text('{"a":1}\n{"b":2}\n', encoding="utf-8")
    errors = validate_jsonl_integrity(path)
    assert errors == []


def test_validate_jsonl_integrity_corrupt(tmp_path: Path) -> None:
    path = tmp_path / "trace.jsonl"
    path.write_text('{"a":1}\n{corrupt\n{"b":2}\n', encoding="utf-8")
    errors = validate_jsonl_integrity(path)
    assert len(errors) == 1
    assert "line 2" in errors[0]


def test_validate_jsonl_integrity_missing_file(tmp_path: Path) -> None:
    errors = validate_jsonl_integrity(tmp_path / "nope.jsonl")
    assert errors == []


def test_validate_jsonl_partial_line(tmp_path: Path) -> None:
    path = tmp_path / "data.jsonl"
    path.write_text('{"ok":true}\n{"trunc', encoding="utf-8")
    errors = validate_jsonl_integrity(path)
    assert len(errors) == 1


# ---------------------------------------------------------------------------
# MemoryStore with atomic append
# ---------------------------------------------------------------------------


def test_memory_store_atomic_append(tmp_path: Path) -> None:
    store = MemoryStore(str(tmp_path / "memory.jsonl"))
    record = MemoryRecord(
        memory_id="mem_test",
        logical_time={"session": 1, "turn": 1, "idx": 0},
        type="Preference",
        content={"pref": "test"},
        source_refs=[1],
    )
    store.append(record)
    # no .tmp should remain
    assert not list(tmp_path.glob("*.tmp"))
    # data should be readable
    records = store.read_all()
    assert len(records) == 1
    assert records[0].memory_id == "mem_test"


# ---------------------------------------------------------------------------
# ConstraintsStore with atomic append
# ---------------------------------------------------------------------------


def test_constraints_store_atomic_append(tmp_path: Path) -> None:
    store = ConstraintsStore(str(tmp_path / "constraints.jsonl"))
    record = ConstraintRecord(
        constraint_id="ctr_test",
        logical_time={"session": 1, "turn": 1, "idx": 0},
        family="boundary",
        rule_id="R-BND-001",
        severity="hard",
        status="active",
        commit_policy="auto",
        effects=[{"type": "forbid", "action_class": "test"}],
        trigger_events=[1],
        explanation="test constraint",
    )
    store.append(record)
    assert not list(tmp_path.glob("*.tmp"))
    records = store.read_all()
    assert len(records) == 1
    assert records[0].constraint_id == "ctr_test"


# ---------------------------------------------------------------------------
# E-EXAM-CRASH-001: Crash simulation E2E via matrix runner
# ---------------------------------------------------------------------------


def test_crash_simulation_e2e(tmp_path: Path) -> None:
    """Full crash simulation: orphan .tmp → recovery → exam → integrity."""
    from pie.matrix_runner import MatrixConfig, run_matrix

    config = MatrixConfig(
        preset="offline_full",
        voice="none",
        cache="no-cache",
        replay="off",
        golden="off",
        crash_test="on",
        policy="valid",
        tools="none",
        rate_limit="off",
        only=["exam", "crash"],
    )
    golden_dir = tmp_path / "golden"
    exit_code, summary = run_matrix(
        [config],
        output_root=tmp_path / "out",
        golden_dir=golden_dir,
        golden_dir_auto=True,
        allow_online=False,
    )
    assert exit_code == 0, f"Matrix failed: {summary}"
    runs = summary.get("runs", [])
    assert len(runs) == 1
    run_result = runs[0]
    crash_suite = run_result.get("suites", {}).get("crash", {})
    assert crash_suite.get("status") == "PASS", f"Crash suite failed: {crash_suite}"
    # Verify recovery_report.md exists and says PASS
    run_id = run_result["run_id"]
    report = (tmp_path / "out" / run_id / "recovery_report.md").read_text(encoding="utf-8")
    assert "Result: PASS" in report
