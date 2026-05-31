"""Concurrency tests for SessionEngine (V6b-E2).

Verifies that per-session locking serializes turns correctly
and that concurrent sessions don't interfere with each other.

Invariants checked:
- turn_count monotonically increasing per session
- event_id monotonically increasing within journal
- journal is valid JSONL (no interleaved/corrupt writes)
- no duplicate event IDs
- concurrent sessions don't cross-contaminate
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from pie.api.engine import SessionEngine
from pie.state_engine.plugins.neural_snn import NeuralSNNPlugin
from pie.state_engine.registry import StateEngineRegistry


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _neural_engine():
    StateEngineRegistry.reset()
    plugin = NeuralSNNPlugin()
    StateEngineRegistry.register(plugin)
    StateEngineRegistry.set_active("neural_snn")
    yield
    StateEngineRegistry.reset()


@pytest.fixture()
def engine(tmp_path: Path) -> SessionEngine:
    return SessionEngine(
        sessions_root=tmp_path / "sessions",
        seeds_root=Path("."),
        llm="fake",
        no_cache=True,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_journal(sessions_root: Path, session_id: str) -> List[Dict[str, Any]]:
    """Read and parse all journal events for a session."""
    journal = sessions_root / session_id / "journal.jsonl"
    if not journal.exists():
        return []
    text = journal.read_text(encoding="utf-8").strip()
    if not text:
        return []
    events = []
    for i, line in enumerate(text.splitlines()):
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise AssertionError(
                f"Journal line {i} is corrupt JSONL: {e}\nLine: {line!r}"
            )
    return events


def _get_event_id(e: Dict[str, Any]) -> int:
    """Extract event ID from journal event (handles both formats)."""
    return e.get("id", e.get("event_id", -1))


def _get_turn(e: Dict[str, Any]) -> Optional[int]:
    """Extract turn number from journal event (handles both formats)."""
    # Standard Event: content.logical_time.turn
    lt = e.get("content", {}).get("logical_time", {})
    t = lt.get("turn")
    if t is not None:
        return t
    # CV_GATING: logical_time.turn at top level
    lt2 = e.get("logical_time", {})
    return lt2.get("turn")


def _verify_journal_invariants(events: List[Dict[str, Any]], label: str = "") -> None:
    """Verify journal integrity invariants.

    Checks:
    - Event IDs are strictly monotonically increasing
    - No duplicate event IDs
    - Journal lines are all valid JSON (checked in _read_journal)
    - Turn numbers are non-decreasing
    """
    prefix = f"[{label}] " if label else ""

    # Event IDs must be strictly monotonically increasing
    ids = [_get_event_id(e) for e in events]
    for i in range(1, len(ids)):
        assert ids[i] > ids[i - 1], (
            f"{prefix}Event IDs not strictly monotonic: "
            f"id[{i-1}]={ids[i-1]}, id[{i}]={ids[i]}"
        )

    # No duplicate IDs
    assert len(ids) == len(set(ids)), (
        f"{prefix}Duplicate event IDs found"
    )

    # Turn numbers within logical_time must be non-decreasing
    turns = []
    for e in events:
        t = _get_turn(e)
        if t is not None:
            turns.append(t)
    for i in range(1, len(turns)):
        assert turns[i] >= turns[i - 1], (
            f"{prefix}Turn numbers not monotonic: turn[{i-1}]={turns[i-1]}, turn[{i}]={turns[i]}"
        )


# ---------------------------------------------------------------------------
# C1: Same session, concurrent turns — lock must serialize
# ---------------------------------------------------------------------------

class TestConcurrency:

    def test_c1_same_session_serialized(self, engine: SessionEngine, tmp_path: Path):
        """10 threads process_turn on the SAME session concurrently.
        Per-session lock must serialize them. Verify turn_count is
        sequential and journal invariants hold."""
        result = engine.create_session(seed_id="SEED_V0")
        sid = result["session_id"]
        n_threads = 10

        results: List[Dict[str, Any]] = [None] * n_threads  # type: ignore
        errors: List[Exception] = []
        barrier = threading.Barrier(n_threads)

        def worker(idx: int) -> None:
            try:
                barrier.wait(timeout=10)
                r = engine.process_turn(sid, f"Messaggio concorrente {idx}")
                results[idx] = r
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)

        # No errors
        assert not errors, f"Threads raised errors: {errors}"

        # All results received
        assert all(r is not None for r in results), "Some threads got no result"

        # Turn counts must be sequential 1..n_threads
        turn_counts = sorted(r["turn_count"] for r in results)
        assert turn_counts == list(range(1, n_threads + 1)), (
            f"Expected sequential turns 1..{n_threads}, got: {turn_counts}"
        )

        # Journal invariants
        events = _read_journal(tmp_path / "sessions", sid)
        _verify_journal_invariants(events, label="C1")

        # Every turn should have an INPUT event
        input_turns = set()
        for e in events:
            if e.get("type") == "INPUT":
                lt = e.get("content", {}).get("logical_time", {})
                input_turns.add(lt.get("turn"))
        for t in range(1, n_threads + 1):
            assert t in input_turns, f"Turn {t} missing INPUT event"

    # ------------------------------------------------------------------
    # C2: Different sessions, concurrent turns — no cross-contamination
    # ------------------------------------------------------------------

    def test_c2_different_sessions_parallel(self, engine: SessionEngine, tmp_path: Path):
        """5 sessions x 4 turns each = 20 concurrent calls.
        Verify each session has correct turn count and no cross-contamination."""
        n_sessions = 5
        turns_per_session = 4

        # Create all sessions first
        session_ids = []
        for _ in range(n_sessions):
            r = engine.create_session(seed_id="SEED_V0")
            session_ids.append(r["session_id"])

        results_by_session: Dict[str, List[Dict[str, Any]]] = {
            sid: [] for sid in session_ids
        }
        errors: List[Exception] = []
        lock = threading.Lock()
        barrier = threading.Barrier(n_sessions * turns_per_session)

        def worker(sid: str, turn_idx: int) -> None:
            try:
                barrier.wait(timeout=10)
                r = engine.process_turn(sid, f"Sessione {sid[:8]} turno {turn_idx}")
                with lock:
                    results_by_session[sid].append(r)
            except Exception as e:
                errors.append(e)

        threads = []
        for sid in session_ids:
            for t in range(turns_per_session):
                threads.append(threading.Thread(target=worker, args=(sid, t)))

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=120)

        # No errors
        assert not errors, f"Threads raised errors: {errors}"

        # Each session has exactly turns_per_session results
        for sid in session_ids:
            assert len(results_by_session[sid]) == turns_per_session, (
                f"Session {sid[:8]} expected {turns_per_session} results, "
                f"got {len(results_by_session[sid])}"
            )

        # Each session has sequential turn counts 1..4
        for sid in session_ids:
            turns = sorted(r["turn_count"] for r in results_by_session[sid])
            assert turns == list(range(1, turns_per_session + 1)), (
                f"Session {sid[:8]}: expected turns 1..{turns_per_session}, got {turns}"
            )

        # Journal invariants per session
        for sid in session_ids:
            events = _read_journal(tmp_path / "sessions", sid)
            _verify_journal_invariants(events, label=f"C2-{sid[:8]}")

        # Cross-contamination check: each session's journal should only
        # reference its own session context
        for sid in session_ids:
            events = _read_journal(tmp_path / "sessions", sid)
            for e in events:
                content = e.get("content", {})
                # INPUT events should contain messages referencing this session
                if e.get("type") == "INPUT":
                    text = content.get("text", "")
                    if text and "Sessione" in text:
                        assert sid[:8] in text, (
                            f"Session {sid[:8]} journal has input from wrong session: {text}"
                        )

    # ------------------------------------------------------------------
    # C3: Rapid create + process race
    # ------------------------------------------------------------------

    def test_c3_create_then_rapid_process(self, engine: SessionEngine, tmp_path: Path):
        """Create a session and immediately fire 5 concurrent process_turn
        calls. All must succeed with sequential turn counts."""
        result = engine.create_session(seed_id="SEED_V0")
        sid = result["session_id"]
        n_threads = 5

        results: List[Dict[str, Any]] = [None] * n_threads  # type: ignore
        errors: List[Exception] = []

        def worker(idx: int) -> None:
            try:
                r = engine.process_turn(sid, f"Rapido {idx}")
                results[idx] = r
            except Exception as e:
                errors.append(e)

        # Fire all at once (no barrier — test the "immediate" race)
        threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)

        assert not errors, f"Threads raised errors: {errors}"
        assert all(r is not None for r in results)

        turn_counts = sorted(r["turn_count"] for r in results)
        assert turn_counts == list(range(1, n_threads + 1)), (
            f"Expected sequential turns, got: {turn_counts}"
        )

        events = _read_journal(tmp_path / "sessions", sid)
        _verify_journal_invariants(events, label="C3")

    # ------------------------------------------------------------------
    # C4: Read-only calls during writes
    # ------------------------------------------------------------------

    def test_c4_reads_during_writes(self, engine: SessionEngine, tmp_path: Path):
        """Process turns while simultaneously reading state and journal.
        Read calls must never crash or return corrupt data."""
        result = engine.create_session(seed_id="SEED_V0")
        sid = result["session_id"]

        # First do a warmup turn so state exists
        engine.process_turn(sid, "Warmup")

        n_writers = 3
        n_readers = 5
        errors: List[Exception] = []
        barrier = threading.Barrier(n_writers + n_readers)

        def writer(idx: int) -> None:
            try:
                barrier.wait(timeout=10)
                engine.process_turn(sid, f"Write turn {idx}")
            except Exception as e:
                errors.append(e)

        def reader(idx: int) -> None:
            try:
                barrier.wait(timeout=10)
                # Read state multiple times during writes
                for _ in range(3):
                    r = engine.get_state(sid)
                    # State must be a valid dict with expected keys
                    state = r["state"]
                    assert "drives" in state, f"Reader {idx}: missing drives"
                    assert "affect" in state, f"Reader {idx}: missing affect"
                    assert "turn_count" in state, f"Reader {idx}: missing turn_count"

                    # Journal must be valid
                    j = engine.get_journal(sid)
                    assert "events" in j
                    assert isinstance(j["events"], list)
            except Exception as e:
                errors.append(e)

        threads = (
            [threading.Thread(target=writer, args=(i,)) for i in range(n_writers)]
            + [threading.Thread(target=reader, args=(i,)) for i in range(n_readers)]
        )
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)

        assert not errors, f"Threads raised errors: {errors}"

        # Final journal check
        events = _read_journal(tmp_path / "sessions", sid)
        _verify_journal_invariants(events, label="C4")

    # ------------------------------------------------------------------
    # C5: Verify no event ID gaps across serialized turns
    # ------------------------------------------------------------------

    def test_c5_event_id_continuity(self, engine: SessionEngine, tmp_path: Path):
        """Process 8 sequential turns and verify event IDs are
        monotonically increasing, starting from 1."""
        result = engine.create_session(seed_id="SEED_V0")
        sid = result["session_id"]

        for i in range(8):
            engine.process_turn(sid, f"Turno sequenziale {i}")

        events = _read_journal(tmp_path / "sessions", sid)
        ids = [_get_event_id(e) for e in events]

        # IDs should start at 1 and be strictly increasing with no gaps
        assert ids[0] == 1, f"First event ID should be 1, got {ids[0]}"
        assert ids == list(range(1, len(ids) + 1)), (
            f"Event IDs not contiguous 1..{len(ids)}: "
            f"first 10 = {ids[:10]}, last 5 = {ids[-5:]}"
        )
