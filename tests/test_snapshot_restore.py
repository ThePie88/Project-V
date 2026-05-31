"""V6b-E1: Snapshot/Restore Idempotent — test suite.

Tests:
    E-SNP-010: Snapshot contains all required components
    E-SNP-020: Restore idempotent (save-restore-save = same hash)
    E-SNP-030: Migration chain preserves data
    E-SNP-040: Replay after restore matches original decisions
    E-SNP-050: Snapshot events emitted in journal
    + edge cases
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import pytest

from pie.contracts.state import State
from pie.state_engine.plugins.neural_snn import (
    NeuralSNNPlugin,
    NEURON_ORDER,
    SYNAPSE_WEIGHTS,
)
from pie.state_engine.plugins.reservoir import Reservoir
from pie.state_engine.plugins.stdp import STDPTracker
from pie.state_engine.plugins.rstdp import RewardSTDPTracker
from pie.state_engine.control_vector import ControlVector
from pie.state_engine.snapshot import SnapshotSerializer
from pie.state_engine.migration import MigrationRegistry
from pie.state_engine.registry import StateEngineRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_state(**overrides) -> State:
    """Create a minimal State for testing."""
    drives = {
        "curiosity": 0.6, "sociality": 0.5, "caution": 0.4,
        "agency": 0.5, "playfulness": 0.5, "fatigue": 0.3,
    }
    affect = {
        "valence": 0.5, "arousal": 0.5, "attention": 0.5, "tension": 0.4,
    }
    drives.update(overrides.get("drives", {}))
    affect.update(overrides.get("affect", {}))
    return State(
        drives=drives,
        affect=affect,
        turn_count=overrides.get("turn_count", 0),
        creator_anchor=overrides.get("creator_anchor", "MrPie"),
    )


def _run_turns(plugin: NeuralSNNPlugin, state: State, n: int) -> State:
    """Run n turns through the plugin, returning final state."""
    for _ in range(n):
        state = plugin.update(state)
    return state


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_registry():
    """Reset StateEngineRegistry between tests."""
    StateEngineRegistry.reset()
    yield
    StateEngineRegistry.reset()


@pytest.fixture(autouse=True)
def _reset_migrations():
    """Reset MigrationRegistry between tests."""
    MigrationRegistry.reset()
    yield
    MigrationRegistry.reset()


# ---------------------------------------------------------------------------
# E-SNP-010: Snapshot contains all required components
# ---------------------------------------------------------------------------

class TestSnapshotComponents:
    """E-SNP-010: verify snapshot has neurons, reservoir, STDP, CV."""

    def test_snapshot_has_all_keys(self):
        plugin = NeuralSNNPlugin(reservoir_enabled=True, stdp_enabled=True)
        state = _make_state()
        state = _run_turns(plugin, state, 5)

        data = plugin.serialize()

        assert data["engine_id"] == "neural_snn"
        assert data["version"] == "0.3"
        assert data["turn_count"] == 5
        assert data["initialised"] is True

        # 10 neurons
        assert len(data["neurons"]) == 10
        for name in NEURON_ORDER:
            nd = data["neurons"][name]
            assert "v" in nd
            assert "u" in nd
            assert "last_spike" in nd
            assert "_normalized" in nd

        # Reservoir
        assert data["reservoir"] is not None
        assert data["reservoir"]["size"] == 128
        assert len(data["reservoir"]["leak_state"]) == 128
        assert len(data["reservoir"]["neurons"]) == 128

        # STDP
        assert data["stdp"] is not None
        assert "weights" in data["stdp"]
        assert "initial_weights" in data["stdp"]
        assert "last_spike_turn" in data["stdp"]

        # ControlVector
        assert data["control_vector"] is not None
        for ch in ["memory_gate", "tool_gate", "cf_k", "verbosity_bias", "consolidation_urgency"]:
            assert ch in data["control_vector"]

    def test_snapshot_serializable_json(self):
        plugin = NeuralSNNPlugin()
        state = _make_state()
        state = _run_turns(plugin, state, 3)

        data = plugin.serialize()
        # Must be JSON-serializable
        text = json.dumps(data, sort_keys=True)
        roundtrip = json.loads(text)
        assert roundtrip["turn_count"] == 3


# ---------------------------------------------------------------------------
# E-SNP-020: Restore idempotent (save-restore-save = same hash)
# ---------------------------------------------------------------------------

class TestIdempotent:
    """E-SNP-020: hash(save1) == hash(save2) after restore."""

    def test_save_restore_save_same_hash(self, tmp_path: Path):
        # Create plugin and register
        plugin = NeuralSNNPlugin(reservoir_enabled=True, stdp_enabled=True)
        StateEngineRegistry.register(plugin)
        StateEngineRegistry.set_active("neural_snn")

        state = _make_state()
        state = _run_turns(plugin, state, 10)

        # Save 1
        hash1 = SnapshotSerializer.save(tmp_path, StateEngineRegistry)

        # Create fresh plugin and register
        StateEngineRegistry.reset()
        plugin2 = NeuralSNNPlugin(reservoir_enabled=True, stdp_enabled=True)
        # Must init neurons before deserialize puts data in
        state2 = _make_state()
        _ = plugin2.update(state2)  # triggers _init_neurons
        StateEngineRegistry.register(plugin2)
        StateEngineRegistry.set_active("neural_snn")

        # Load
        hash_loaded = SnapshotSerializer.load(tmp_path, StateEngineRegistry)
        assert hash_loaded == hash1

        # Save 2
        hash2 = SnapshotSerializer.save(tmp_path, StateEngineRegistry)
        assert hash1 == hash2

    def test_reservoir_serialize_roundtrip(self):
        """Reservoir serialize->deserialize is lossless for dynamic state."""
        r1 = Reservoir(seed=0xDEADBEEF)
        # Step a few times
        for _ in range(5):
            r1.step([0.5] * 10)

        data = r1.serialize()
        r2 = Reservoir(seed=0xDEADBEEF)
        r2.deserialize(data)

        assert r1._step_count == r2._step_count
        # Compare at serialization precision (10 dp)
        for i in range(128):
            assert round(r1._leak_state[i], 10) == round(r2._leak_state[i], 10)
            assert round(r1._neurons[i].v, 10) == round(r2._neurons[i].v, 10)
            assert round(r1._neurons[i].u, 10) == round(r2._neurons[i].u, 10)

    def test_stdp_serialize_roundtrip(self):
        """STDP serialize->deserialize is lossless."""
        t1 = STDPTracker(initial_weights=dict(SYNAPSE_WEIGHTS))
        t1.record_spikes(1, ["curiosity", "arousal"])
        t1.record_spikes(2, ["fatigue", "caution"])

        data = t1.serialize()
        t2 = STDPTracker(initial_weights=dict(SYNAPSE_WEIGHTS))
        t2.deserialize(data)

        assert t1._weights == t2._weights
        assert t1._last_spike_turn == t2._last_spike_turn

    def test_rstdp_serialize_roundtrip(self):
        """R-STDP serialize->deserialize is lossless."""
        r1 = RewardSTDPTracker(initial_weights=dict(SYNAPSE_WEIGHTS))
        r1.record_spikes(1, ["curiosity", "arousal"])
        r1.apply_reward(1, "tool_success")
        r1.record_spikes(2, ["fatigue"])

        data = r1.serialize()
        r2 = RewardSTDPTracker(initial_weights=dict(SYNAPSE_WEIGHTS))
        r2.deserialize(data)

        assert r1._eligibility == r2._eligibility
        assert r1._gamma == r2._gamma
        assert r1._eta == r2._eta


# ---------------------------------------------------------------------------
# E-SNP-030: Migration chain preserves data
# ---------------------------------------------------------------------------

class TestMigration:
    """E-SNP-030: versioned migrations transform snapshots correctly."""

    def test_synthetic_v0_to_v1_migration(self):
        # Register a synthetic migration
        def migrate_v0_v1(data):
            data["engine_state"]["migrated"] = True
            return data

        MigrationRegistry.register(0, 1, migrate_v0_v1)

        v0_snapshot = {
            "snapshot_schema_version": 0,
            "active_engine_id": "neural_snn",
            "engine_version": "0.3",
            "engine_state": {"turn_count": 5},
        }

        result = MigrationRegistry.migrate(v0_snapshot, 0, 1)
        assert result["snapshot_schema_version"] == 1
        assert result["engine_state"]["migrated"] is True
        assert result["engine_state"]["turn_count"] == 5

    def test_missing_migration_raises(self):
        with pytest.raises(ValueError, match="No migration registered"):
            MigrationRegistry.migrate({}, 0, 1)

    def test_chain_migration(self):
        MigrationRegistry.register(0, 1, lambda d: {**d, "v1": True})
        MigrationRegistry.register(1, 2, lambda d: {**d, "v2": True})

        data = {"snapshot_schema_version": 0}
        result = MigrationRegistry.migrate(data, 0, 2)
        assert result["snapshot_schema_version"] == 2
        assert result["v1"] is True
        assert result["v2"] is True


# ---------------------------------------------------------------------------
# E-SNP-040: Replay after restore matches original decisions
# ---------------------------------------------------------------------------

class TestReplay:
    """E-SNP-040: state after restore + N turns == original + N turns."""

    def test_replay_matches(self, tmp_path: Path):
        # Run 5 turns, save, then continue 5 more
        plugin1 = NeuralSNNPlugin(reservoir_enabled=True, stdp_enabled=True)
        StateEngineRegistry.register(plugin1)
        StateEngineRegistry.set_active("neural_snn")

        state = _make_state()
        state = _run_turns(plugin1, state, 5)

        # Save snapshot
        SnapshotSerializer.save(tmp_path, StateEngineRegistry)

        # Continue original for 5 more turns
        state_original = _run_turns(plugin1, state, 5)
        original_data = plugin1.serialize()

        # Restore into fresh plugin
        StateEngineRegistry.reset()
        plugin2 = NeuralSNNPlugin(reservoir_enabled=True, stdp_enabled=True)
        # Init neurons with same state to provide structure
        state2 = _make_state()
        _ = plugin2.update(state2)
        StateEngineRegistry.register(plugin2)
        StateEngineRegistry.set_active("neural_snn")
        SnapshotSerializer.load(tmp_path, StateEngineRegistry)

        # Run same 5 turns from restored state
        state_restored = _run_turns(plugin2, state, 5)
        restored_data = plugin2.serialize()

        # Neuron states must be close within serialization tolerance.
        # round(x, 10) truncation propagates through the ODE for 5 turns,
        # so we allow 1e-6 tolerance (well within practical relevance).
        for name in NEURON_ORDER:
            v_orig = original_data["neurons"][name]["v"]
            v_rest = restored_data["neurons"][name]["v"]
            assert abs(v_orig - v_rest) < 1e-6, \
                f"Neuron {name} v mismatch: {v_orig} vs {v_rest}"
            u_orig = original_data["neurons"][name]["u"]
            u_rest = restored_data["neurons"][name]["u"]
            assert abs(u_orig - u_rest) < 1e-6, \
                f"Neuron {name} u mismatch: {u_orig} vs {u_rest}"

        # Turn count must match
        assert original_data["turn_count"] == restored_data["turn_count"]


# ---------------------------------------------------------------------------
# E-SNP-050: Snapshot events in journal
# ---------------------------------------------------------------------------

class TestJournalEvents:
    """E-SNP-050: save/resume emit SNAPSHOT_SAVED/RESTORED to journal."""

    def test_save_emits_event(self, tmp_path: Path):
        from pie.session.manager import SessionManager, SessionContext
        from pie.persistence.memory_store import MemoryStore
        from pie.persistence.constraints_store import ConstraintsStore

        session_dir = tmp_path / "test_session"
        session_dir.mkdir()
        journal_path = session_dir / "journal.jsonl"
        journal_path.touch()

        # Write required files
        state = _make_state()
        (session_dir / "state_latest.json").write_text(
            json.dumps(state.snapshot(), indent=2), encoding="utf-8"
        )
        (session_dir / "identity_snapshot.json").write_text(
            json.dumps({"name": "Ivy"}), encoding="utf-8"
        )
        (session_dir / "session_meta.json").write_text(
            json.dumps({"schema_version": "0.1", "session_id": "test", "turn_count": 0}),
            encoding="utf-8",
        )

        # Register a plugin so snapshot has something to save
        plugin = NeuralSNNPlugin()
        s = _make_state()
        _ = plugin.update(s)
        StateEngineRegistry.register(plugin)
        StateEngineRegistry.set_active("neural_snn")

        ctx = SessionContext(
            session_id="test",
            session_dir=session_dir,
            state=state,
            identity={"name": "Ivy"},
            event_id=1,
            turn_count=0,
            journal_path=journal_path,
            memory_store=MemoryStore(str(session_dir / "memory.jsonl")),
            constraints_store=ConstraintsStore(str(session_dir / "constraints.jsonl")),
        )

        mgr = SessionManager(sessions_root=tmp_path)
        mgr.save(ctx)

        # Check journal
        events = [
            json.loads(line)
            for line in journal_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        snapshot_events = [e for e in events if e["type"] == "SNAPSHOT_SAVED"]
        assert len(snapshot_events) == 1
        assert "hash" in snapshot_events[0]["content"]

    def test_resume_emits_event(self, tmp_path: Path):
        from pie.session.manager import SessionManager

        session_id = "snap_test"
        session_dir = tmp_path / session_id
        session_dir.mkdir()

        # Register plugin and save snapshot
        plugin = NeuralSNNPlugin()
        s = _make_state()
        _ = plugin.update(s)
        StateEngineRegistry.register(plugin)
        StateEngineRegistry.set_active("neural_snn")
        SnapshotSerializer.save(session_dir, StateEngineRegistry)

        # Write required session files
        state = _make_state()
        (session_dir / "state_latest.json").write_text(
            json.dumps(state.snapshot(), indent=2), encoding="utf-8"
        )
        (session_dir / "identity_snapshot.json").write_text(
            json.dumps({"name": "Ivy"}), encoding="utf-8"
        )
        (session_dir / "session_meta.json").write_text(
            json.dumps({
                "schema_version": "0.1",
                "session_id": session_id,
                "turn_count": 1,
                "resumed_at_list": [],
            }),
            encoding="utf-8",
        )
        (session_dir / "journal.jsonl").touch()

        # Resume
        StateEngineRegistry.reset()
        # Need a fresh plugin registered for load to work
        plugin2 = NeuralSNNPlugin()
        s2 = _make_state()
        _ = plugin2.update(s2)
        StateEngineRegistry.register(plugin2)
        StateEngineRegistry.set_active("neural_snn")

        mgr = SessionManager(sessions_root=tmp_path)
        ctx = mgr.resume(session_id)

        # Check journal for SNAPSHOT_RESTORED
        events = [
            json.loads(line)
            for line in ctx.journal_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        restored_events = [e for e in events if e["type"] == "SNAPSHOT_RESTORED"]
        assert len(restored_events) == 1
        assert "hash" in restored_events[0]["content"]


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Additional edge-case tests for robustness."""

    def test_no_snapshot_resume_works(self, tmp_path: Path):
        """Session without engine_snapshot.json still works (backward compat)."""
        result = SnapshotSerializer.load(tmp_path, StateEngineRegistry)
        assert result is None

    def test_default_ode_plugin_no_crash(self, tmp_path: Path):
        """DefaultODE (no serialize/deserialize) does not crash."""
        # Default plugin auto-registers
        hash1 = SnapshotSerializer.save(tmp_path, StateEngineRegistry)
        assert hash1 is not None

        # Load back
        StateEngineRegistry.reset()
        hash2 = SnapshotSerializer.load(tmp_path, StateEngineRegistry)
        assert hash2 == hash1

    def test_reservoir_seed_mismatch_raises(self):
        """Deserializing with wrong seed raises ValueError."""
        r1 = Reservoir(seed=0xDEADBEEF)
        r1.step([0.5] * 10)
        data = r1.serialize()

        r2 = Reservoir(seed=0x12345678)
        with pytest.raises(ValueError, match="seed mismatch"):
            r2.deserialize(data)

    def test_reservoir_size_mismatch_raises(self):
        """Deserializing with wrong size raises ValueError."""
        r1 = Reservoir(size=128, seed=0xDEADBEEF)
        r1.step([0.5] * 10)
        data = r1.serialize()

        r2 = Reservoir(size=64, seed=0xDEADBEEF)
        with pytest.raises(ValueError, match="size mismatch"):
            r2.deserialize(data)

    def test_compute_hash_no_file(self, tmp_path: Path):
        assert SnapshotSerializer.compute_hash(tmp_path) is None

    def test_compute_hash_matches_save(self, tmp_path: Path):
        """compute_hash returns same hash as save without loading."""
        # Use default_ode (simplest) to avoid registry state issues
        save_hash = SnapshotSerializer.save(tmp_path, StateEngineRegistry)
        computed = SnapshotSerializer.compute_hash(tmp_path)
        assert save_hash == computed

    def test_snapshot_file_valid_json(self, tmp_path: Path):
        """engine_snapshot.json is valid JSON with sorted keys."""
        plugin = NeuralSNNPlugin()
        s = _make_state()
        _ = plugin.update(s)
        StateEngineRegistry.register(plugin)
        StateEngineRegistry.set_active("neural_snn")

        SnapshotSerializer.save(tmp_path, StateEngineRegistry)
        content = (tmp_path / "engine_snapshot.json").read_text(encoding="utf-8")
        data = json.loads(content)
        assert data["snapshot_schema_version"] == 1
        assert data["active_engine_id"] == "neural_snn"

        # Verify keys are sorted (deterministic)
        keys_in_file = list(data.keys())
        assert keys_in_file == sorted(keys_in_file)
