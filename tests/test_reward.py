"""Reward signal tests — governed online learning (R-STDP).

Tests:
- T1: Collect-only (safety) — reward logged but weights unchanged
- T2: Online learning — reward applied, weights change
- T3: Idempotency — duplicate reward ignored
- T4: Kill-switch safety — no REWARD_APPLIED during KS-active turns
- T5: Journal-first reconstruction — simulate restart
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List

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

def _journal_events(sessions_root: Path, session_id: str) -> List[Dict[str, Any]]:
    journal = sessions_root / session_id / "journal.jsonl"
    if not journal.exists():
        return []
    events = []
    for line in journal.read_text(encoding="utf-8").strip().splitlines():
        if line.strip():
            events.append(json.loads(line))
    return events


def _events_by_type(events: List[Dict[str, Any]], etype: str) -> List[Dict[str, Any]]:
    return [e for e in events if e.get("type") == etype]


# ---------------------------------------------------------------------------
# T1: Collect-only (safety) — PIE_RSTDP_ONLINE=0
# ---------------------------------------------------------------------------

class TestReward:

    def test_t1_collect_only(self, engine: SessionEngine, tmp_path: Path, monkeypatch):
        """Reward logged but NOT applied (online=0). Weights unchanged."""
        monkeypatch.setenv("PIE_RSTDP_ONLINE", "0")

        result = engine.create_session(seed_id="SEED_V0")
        sid = result["session_id"]

        # Process 3 turns
        for i in range(3):
            engine.process_turn(sid, f"Turno {i+1}")

        # Submit reward for turn 2
        reward_result = engine.submit_reward(sid, target_turn=2, value=1)
        assert reward_result["status"] == "accepted"
        assert reward_result["target_turn"] == 2

        # Process turn 4 — triggers REWARD_APPLIED
        engine.process_turn(sid, "Turno 4")

        events = _journal_events(tmp_path / "sessions", sid)

        # REWARD_SIGNAL must exist
        signals = _events_by_type(events, "REWARD_SIGNAL")
        assert len(signals) == 1
        assert signals[0]["content"]["target_turn"] == 2
        assert signals[0]["content"]["value"] == 1

        # REWARD_APPLIED must exist with applied=False
        applied = _events_by_type(events, "REWARD_APPLIED")
        assert len(applied) == 1
        assert applied[0]["content"]["applied"] is False
        assert applied[0]["content"]["online"] is False

    # ------------------------------------------------------------------
    # T2: Online learning — PIE_RSTDP_ONLINE=1
    # ------------------------------------------------------------------

    def test_t2_online_learning(self, engine: SessionEngine, tmp_path: Path, monkeypatch):
        """Reward applied with R-STDP online. Tracker upgraded, apply_reward called."""
        monkeypatch.setenv("PIE_RSTDP_ONLINE", "1")

        # Need fresh engine with online env set
        engine = SessionEngine(
            sessions_root=tmp_path / "sessions",
            seeds_root=Path("."),
            llm="fake",
            no_cache=True,
        )
        result = engine.create_session(seed_id="SEED_V0")
        sid = result["session_id"]

        # Process 3 turns
        for i in range(3):
            engine.process_turn(sid, f"Turno {i+1}")

        # Verify tracker was upgraded to RewardSTDPTracker
        from pie.state_engine.plugins.rstdp import RewardSTDPTracker
        _eng = StateEngineRegistry.get_active()
        assert isinstance(_eng.rstdp_tracker, RewardSTDPTracker), (
            f"Expected RewardSTDPTracker, got {type(_eng.rstdp_tracker).__name__}"
        )

        # Submit reward for turn 2
        reward_result = engine.submit_reward(sid, target_turn=2, value=1)
        assert reward_result["status"] == "accepted"

        # Process turn 4 — triggers REWARD_APPLIED
        engine.process_turn(sid, "Turno 4")

        events = _journal_events(tmp_path / "sessions", sid)

        # REWARD_APPLIED must exist with applied=True
        applied = _events_by_type(events, "REWARD_APPLIED")
        assert len(applied) == 1
        assert applied[0]["content"]["applied"] is True
        assert applied[0]["content"]["online"] is True

        # Verify reward was recorded in tracker's reward log
        assert len(_eng.rstdp_tracker._reward_log) >= 1
        assert _eng.rstdp_tracker._reward_log[-1].value == 1.0

    # ------------------------------------------------------------------
    # T3: Idempotency — double reward on same turn
    # ------------------------------------------------------------------

    def test_t3_idempotency(self, engine: SessionEngine, tmp_path: Path, monkeypatch):
        """Second reward on same turn → REWARD_DUPLICATE_IGNORED."""
        monkeypatch.setenv("PIE_RSTDP_ONLINE", "0")

        result = engine.create_session(seed_id="SEED_V0")
        sid = result["session_id"]

        # Process 2 turns
        engine.process_turn(sid, "Turno 1")
        engine.process_turn(sid, "Turno 2")

        # First reward: accepted
        r1 = engine.submit_reward(sid, target_turn=2, value=1)
        assert r1["status"] == "accepted"

        # Second reward: duplicate
        r2 = engine.submit_reward(sid, target_turn=2, value=-1)
        assert r2["status"] == "duplicate_ignored"

        events = _journal_events(tmp_path / "sessions", sid)

        # One REWARD_SIGNAL, one REWARD_DUPLICATE_IGNORED
        signals = _events_by_type(events, "REWARD_SIGNAL")
        assert len(signals) == 1

        dupes = _events_by_type(events, "REWARD_DUPLICATE_IGNORED")
        assert len(dupes) == 1

        # Process turn 3 — only one REWARD_APPLIED
        engine.process_turn(sid, "Turno 3")
        events = _journal_events(tmp_path / "sessions", sid)
        applied = _events_by_type(events, "REWARD_APPLIED")
        assert len(applied) == 1

    # ------------------------------------------------------------------
    # T4: Kill-switch safety
    # ------------------------------------------------------------------

    def test_t4_killswitch_safety(self, engine: SessionEngine, tmp_path: Path, monkeypatch):
        """No REWARD_APPLIED during KS-active turns."""
        monkeypatch.setenv("PIE_RSTDP_ONLINE", "0")

        result = engine.create_session(seed_id="SEED_V0")
        sid = result["session_id"]

        # Process 2 turns
        engine.process_turn(sid, "Turno 1")
        engine.process_turn(sid, "Turno 2")

        # Submit reward for turn 2
        engine.submit_reward(sid, target_turn=2, value=1)

        # Activate kill-switch
        live = engine._get_or_init(sid)
        live.tp.killswitch.active = True

        # Process turn 3 — KS active, should return early
        response = engine.process_turn(sid, "Turno 3 KS")

        events = _journal_events(tmp_path / "sessions", sid)

        # During KS turn, there should be NO REWARD_APPLIED
        applied = _events_by_type(events, "REWARD_APPLIED")
        assert len(applied) == 0, "REWARD_APPLIED should not appear during KS-active turn"

        # Deactivate KS
        live.tp.killswitch.active = False

        # Process turn 4 — reward should now be applied
        engine.process_turn(sid, "Turno 4")
        events = _journal_events(tmp_path / "sessions", sid)
        applied = _events_by_type(events, "REWARD_APPLIED")
        assert len(applied) == 1
        assert applied[0]["content"]["target_turn"] == 2

    # ------------------------------------------------------------------
    # T5: Journal-first reconstruction — simulate restart
    # ------------------------------------------------------------------

    def test_t5_journal_first_reconstruction(self, engine: SessionEngine, tmp_path: Path, monkeypatch):
        """After restart, pending/applied reconstructed from journal."""
        monkeypatch.setenv("PIE_RSTDP_ONLINE", "0")

        result = engine.create_session(seed_id="SEED_V0")
        sid = result["session_id"]

        # Process 3 turns, submit reward, process turn 4
        for i in range(3):
            engine.process_turn(sid, f"Turno {i+1}")

        engine.submit_reward(sid, target_turn=2, value=1)
        engine.process_turn(sid, "Turno 4")

        # Record original state
        live = engine._get_or_init(sid)
        orig_pending = dict(live.tp._pending_rewards)
        orig_applied = set(live.tp._applied_rewards)

        # Simulate restart: evict session, create new engine
        engine._sessions.clear()
        StateEngineRegistry.reset()
        StateEngineRegistry.register(NeuralSNNPlugin())
        StateEngineRegistry.set_active("neural_snn")

        engine2 = SessionEngine(
            sessions_root=tmp_path / "sessions",
            seeds_root=Path("."),
            llm="fake",
            no_cache=True,
        )

        # Resume session
        live2 = engine2._get_or_init(sid)

        # Reconstructed state should match
        assert live2.tp._applied_rewards == orig_applied, (
            f"Applied mismatch: {live2.tp._applied_rewards} != {orig_applied}"
        )
        # Pending should also be reconstructed
        assert set(live2.tp._pending_rewards.keys()) == set(orig_pending.keys()), (
            f"Pending keys mismatch: {set(live2.tp._pending_rewards.keys())} != {set(orig_pending.keys())}"
        )

        # Process turn 5 — should NOT re-apply reward for turn 2
        engine2.process_turn(sid, "Turno 5")
        events = _journal_events(tmp_path / "sessions", sid)
        applied = _events_by_type(events, "REWARD_APPLIED")
        # Only the original REWARD_APPLIED, no duplicate
        assert len(applied) == 1
