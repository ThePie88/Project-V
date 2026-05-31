"""Tests for V3.0 — Identity Bootstrap + Session Management.

Test IDs:
- I-SESSION-010: Session create + save
- I-SESSION-020: Session resume deterministic
- I-IDENT-030: Seed bootstrap present in state
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from pie.contracts.state import State
from pie.session.bootstrap import load_seed, build_identity_snapshot
from pie.session.manager import SessionManager, SessionContext

# Path to the real SEED file used by the project
SEED_PATH = Path(__file__).resolve().parent.parent / "progetto" / "SEED_V0.md"


@pytest.fixture
def tmp_sessions(tmp_path):
    """Provide a temporary sessions root directory."""
    return tmp_path / "sessions"


@pytest.fixture
def seed_data():
    """Load the real SEED data."""
    return load_seed(SEED_PATH)


# -----------------------------------------------------------------------
# I-IDENT-030 — Seed bootstrap presente nello stato
# -----------------------------------------------------------------------


class TestIdentityBootstrap:
    """Verify that State.from_seed() correctly initialises from SEED."""

    def test_drives_from_seed(self, seed_data):
        """State drives must match SEED values, not defaults."""
        state = State.from_seed(seed_data)
        assert state.drives["curiosity"] == pytest.approx(0.78)
        assert state.drives["sociality"] == pytest.approx(0.72)
        assert state.drives["caution"] == pytest.approx(0.58)
        assert state.drives["agency"] == pytest.approx(0.62)
        assert state.drives["playfulness"] == pytest.approx(0.55)
        assert state.drives["fatigue"] == pytest.approx(0.10)

    def test_affect_from_seed(self, seed_data):
        """State affect must match SEED affect_baseline, not defaults."""
        state = State.from_seed(seed_data)
        assert state.affect["valence"] == pytest.approx(0.10)
        assert state.affect["arousal"] == pytest.approx(0.20)
        assert state.affect["attention"] == pytest.approx(0.55)
        assert state.affect["tension"] == pytest.approx(0.15)

    def test_creator_anchor_from_seed(self, seed_data):
        """Creator anchor must come from SEED identity."""
        state = State.from_seed(seed_data)
        assert state.creator_anchor == "MrPie"

    def test_default_drives_not_used(self, seed_data):
        """Ensure SEED overrides defaults (curiosity is 0.78, not 0.5)."""
        default_state = State()
        seed_state = State.from_seed(seed_data)
        assert default_state.drives["curiosity"] == 0.5
        assert seed_state.drives["curiosity"] == 0.78


# -----------------------------------------------------------------------
# I-SESSION-010 — Session create + save
# -----------------------------------------------------------------------


class TestSessionCreate:
    """Verify that creating a session produces all required files."""

    def test_create_session_files(self, tmp_sessions):
        """New session must create all 4 required files."""
        mgr = SessionManager(tmp_sessions)
        ctx = mgr.create(SEED_PATH, session_id="test_create")

        session_dir = tmp_sessions / "test_create"
        assert session_dir.is_dir()
        assert (session_dir / "identity_snapshot.json").exists()
        assert (session_dir / "state_latest.json").exists()
        assert (session_dir / "journal.jsonl").exists()
        assert (session_dir / "session_meta.json").exists()

    def test_identity_snapshot_content(self, tmp_sessions):
        """identity_snapshot.json must contain name, alias, creator_anchor."""
        mgr = SessionManager(tmp_sessions)
        ctx = mgr.create(SEED_PATH, session_id="test_identity")

        snapshot = json.loads(
            (tmp_sessions / "test_identity" / "identity_snapshot.json").read_text()
        )
        assert snapshot["identity"]["name"] == "Ivy"
        assert snapshot["identity"]["alias"] == "V"
        assert snapshot["identity"]["creator_anchor"] == "MrPie"
        assert "drives" in snapshot
        assert "traits" in snapshot
        assert "values" in snapshot
        assert "bootstrapped_from" in snapshot

    def test_state_latest_from_seed(self, tmp_sessions):
        """state_latest.json must have SEED-derived drive values."""
        mgr = SessionManager(tmp_sessions)
        ctx = mgr.create(SEED_PATH, session_id="test_state")

        state_data = json.loads(
            (tmp_sessions / "test_state" / "state_latest.json").read_text()
        )
        assert state_data["drives"]["curiosity"] == pytest.approx(0.78)
        assert state_data["affect"]["valence"] == pytest.approx(0.10)
        assert state_data["creator_anchor"] == "MrPie"
        assert state_data["turn_count"] == 0

    def test_session_meta(self, tmp_sessions):
        """session_meta.json must have session_id, seed_hash, created_at."""
        mgr = SessionManager(tmp_sessions)
        ctx = mgr.create(SEED_PATH, session_id="test_meta")

        meta = json.loads(
            (tmp_sessions / "test_meta" / "session_meta.json").read_text()
        )
        assert meta["session_id"] == "test_meta"
        assert "seed_hash" in meta
        assert len(meta["seed_hash"]) == 64  # SHA256 hex
        assert "created_at" in meta
        assert meta["turn_count"] == 0

    def test_session_context_fields(self, tmp_sessions):
        """SessionContext returned by create() has correct initial values."""
        mgr = SessionManager(tmp_sessions)
        ctx = mgr.create(SEED_PATH, session_id="test_ctx")

        assert ctx.session_id == "test_ctx"
        assert ctx.event_id == 1
        assert ctx.turn_count == 0
        assert ctx.state.drives["curiosity"] == pytest.approx(0.78)
        assert ctx.identity["identity"]["name"] == "Ivy"

    def test_auto_generated_session_id(self, tmp_sessions):
        """When no session_id given, auto-generate one starting with 'ivy_'."""
        mgr = SessionManager(tmp_sessions)
        ctx = mgr.create(SEED_PATH)

        assert ctx.session_id.startswith("ivy_")
        assert len(ctx.session_id) > 4

    def test_list_sessions(self, tmp_sessions):
        """list_sessions() returns metadata for created sessions."""
        mgr = SessionManager(tmp_sessions)
        mgr.create(SEED_PATH, session_id="s1")
        mgr.create(SEED_PATH, session_id="s2")

        sessions = mgr.list_sessions()
        ids = [s["session_id"] for s in sessions]
        assert "s1" in ids
        assert "s2" in ids


# -----------------------------------------------------------------------
# I-SESSION-020 — Session resume deterministico
# -----------------------------------------------------------------------


class TestSessionResume:
    """Verify that resuming a session restores state correctly."""

    def test_resume_state_match(self, tmp_sessions):
        """Resumed state must match saved state exactly."""
        mgr = SessionManager(tmp_sessions)
        ctx = mgr.create(SEED_PATH, session_id="test_resume")

        # Simulate a turn: update state and save
        ctx.state = ctx.state.update()
        ctx.turn_count = 1
        ctx.event_id = 5
        mgr.save(ctx)

        # Resume and verify
        ctx2 = mgr.resume("test_resume")
        assert ctx2.state.turn_count == 1
        assert ctx2.state.drives["curiosity"] == pytest.approx(
            ctx.state.drives["curiosity"], abs=1e-4
        )
        assert ctx2.state.affect["arousal"] == pytest.approx(
            ctx.state.affect["arousal"], abs=1e-4
        )

    def test_resume_event_id_continues(self, tmp_sessions):
        """After resume, event_id must continue from the last journal entry."""
        mgr = SessionManager(tmp_sessions)
        ctx = mgr.create(SEED_PATH, session_id="test_eid")

        # Write some events to journal
        from pie.contracts import Event
        from pie.persistence.atomic import atomic_append

        for eid in range(1, 4):
            evt = Event.new(eid, "INPUT", {"logical_time": {"turn": eid, "step": 1}, "input": "test"})
            atomic_append(ctx.journal_path, json.dumps(evt.to_json()) + "\n")

        ctx2 = mgr.resume("test_eid")
        assert ctx2.event_id == 4  # next after last (3)

    def test_resume_adds_resumed_at(self, tmp_sessions):
        """Resume must append to resumed_at_list in session_meta."""
        mgr = SessionManager(tmp_sessions)
        ctx = mgr.create(SEED_PATH, session_id="test_resumed_at")

        mgr.resume("test_resumed_at")
        mgr.resume("test_resumed_at")

        meta = json.loads(
            (tmp_sessions / "test_resumed_at" / "session_meta.json").read_text()
        )
        assert len(meta["resumed_at_list"]) == 2

    def test_resume_nonexistent_fails(self, tmp_sessions):
        """Resuming a non-existent session must raise FileNotFoundError."""
        mgr = SessionManager(tmp_sessions)
        with pytest.raises(FileNotFoundError):
            mgr.resume("does_not_exist")

    def test_resume_identity_preserved(self, tmp_sessions):
        """Identity must be identical after resume."""
        mgr = SessionManager(tmp_sessions)
        ctx = mgr.create(SEED_PATH, session_id="test_ident")
        ctx2 = mgr.resume("test_ident")

        assert ctx2.identity["identity"]["name"] == "Ivy"
        assert ctx2.identity["identity"]["alias"] == "V"


# -----------------------------------------------------------------------
# Bootstrap edge cases
# -----------------------------------------------------------------------


class TestBootstrapEdgeCases:
    """Edge cases for seed loading and identity snapshot."""

    def test_load_seed_file_not_found(self):
        """Missing seed file must raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_seed(Path("nonexistent_seed.json"))

    def test_load_seed_invalid_json(self, tmp_path):
        """Invalid JSON in seed file must raise ValueError."""
        bad_seed = tmp_path / "bad_seed.md"
        bad_seed.write_text("this is not json", encoding="utf-8")
        with pytest.raises(ValueError, match="not valid JSON"):
            load_seed(bad_seed)

    def test_load_seed_missing_identity(self, tmp_path):
        """Seed without 'identity' section must raise ValueError."""
        no_ident = tmp_path / "no_ident.md"
        no_ident.write_text('{"drives": {}}', encoding="utf-8")
        with pytest.raises(ValueError, match="missing required 'identity'"):
            load_seed(no_ident)

    def test_build_identity_snapshot_includes_metadata(self, seed_data):
        """Identity snapshot must include bootstrapped_from and schema_version."""
        snapshot = build_identity_snapshot(seed_data, SEED_PATH)
        # schema_version comes from SEED (integer 1) which overrides the
        # snapshot default.  The important thing is it's present.
        assert "schema_version" in snapshot
        assert "bootstrapped_from" in snapshot
        assert "bootstrapped_at" in snapshot
