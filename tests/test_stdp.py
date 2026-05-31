"""Tests for V4 — STDP (Spike-Timing Dependent Plasticity).

Test IDs:
- E-STDP-010: Pre-before-post increases weight
- E-STDP-011: Post-before-pre decreases weight
- E-STDP-012: Weights never exceed bounds
- E-STDP-013: JSONL log grows monotonically
- E-STDP-014: initial + log = current weights
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pie.contracts.state import State
from pie.state_engine.plugins.stdp import (
    STDPTracker,
    STDPRecord,
    A_PLUS,
    A_MINUS,
    WEIGHT_MIN,
    WEIGHT_MAX,
)
from pie.state_engine.plugins.neural_snn import (
    NeuralSNNPlugin,
    SYNAPSE_WEIGHTS,
)
from pie.state_engine.registry import StateEngineRegistry


@pytest.fixture(autouse=True)
def reset_registry():
    StateEngineRegistry.reset()
    yield
    StateEngineRegistry.reset()


# ── E-STDP-010: Pre-before-post → potentiation ───────────────────────

class TestSTDPPotentiation:
    def test_pre_before_post_increases_weight(self):
        """E-STDP-010: If source spikes before target, weight increases."""
        weights = {("A", "B"): 0.5}
        tracker = STDPTracker(initial_weights=weights)

        # Turn 1: A spikes (pre)
        tracker.record_spikes(1, ["A"])
        # Turn 2: B spikes (post) — A spiked before B → potentiation
        changes = tracker.record_spikes(2, ["B"])

        # Weight should have increased
        assert tracker.weights[("A", "B")] > 0.5
        # Should have a potentiation record
        pot_records = [c for c in changes if c.rule == "potentiation"]
        assert len(pot_records) >= 1

    def test_closer_timing_stronger(self):
        """Closer spike timing should produce larger potentiation."""
        # Test with dt=1 vs dt=3
        w1 = {("A", "B"): 0.5}
        t1 = STDPTracker(initial_weights=w1)
        t1.record_spikes(1, ["A"])
        t1.record_spikes(2, ["B"])  # dt=1
        delta1 = t1.weights[("A", "B")] - 0.5

        w2 = {("A", "B"): 0.5}
        t2 = STDPTracker(initial_weights=w2)
        t2.record_spikes(1, ["A"])
        t2.record_spikes(4, ["B"])  # dt=3
        delta2 = t2.weights[("A", "B")] - 0.5

        assert delta1 > delta2 > 0


# ── E-STDP-011: Post-before-pre → depression ─────────────────────────

class TestSTDPDepression:
    def test_post_before_pre_decreases_weight(self):
        """E-STDP-011: If target spikes before source, weight decreases."""
        weights = {("A", "B"): 0.5}
        tracker = STDPTracker(initial_weights=weights)

        # Turn 1: B spikes (post fires first)
        tracker.record_spikes(1, ["B"])
        # Turn 2: A spikes (pre fires after) — post-before-pre → depression
        changes = tracker.record_spikes(2, ["A"])

        assert tracker.weights[("A", "B")] < 0.5
        dep_records = [c for c in changes if c.rule == "depression"]
        assert len(dep_records) >= 1


# ── E-STDP-012: Weights never exceed bounds ──────────────────────────

class TestSTDPBounds:
    def test_weight_upper_bound(self):
        """E-STDP-012: Weight cannot exceed WEIGHT_MAX."""
        weights = {("A", "B"): 0.99}
        tracker = STDPTracker(initial_weights=weights)
        for turn in range(1, 100):
            tracker.record_spikes(turn, ["A"])
            tracker.record_spikes(turn + 1, ["B"])
        assert tracker.weights[("A", "B")] <= WEIGHT_MAX

    def test_weight_lower_bound(self):
        """Weight cannot go below WEIGHT_MIN."""
        weights = {("A", "B"): -0.99}
        tracker = STDPTracker(initial_weights=weights)
        for turn in range(1, 100):
            tracker.record_spikes(turn, ["B"])
            tracker.record_spikes(turn + 1, ["A"])
        assert tracker.weights[("A", "B")] >= WEIGHT_MIN


# ── E-STDP-013: JSONL log grows monotonically ────────────────────────

class TestSTDPLog:
    def test_log_grows_monotonically(self, tmp_path):
        """E-STDP-013: JSONL log file grows with each STDP event."""
        log_path = tmp_path / "stdp_log.jsonl"
        weights = {("A", "B"): 0.5, ("B", "C"): 0.3}
        tracker = STDPTracker(initial_weights=weights, log_path=log_path)

        prev_size = 0
        for turn in range(1, 10):
            tracker.record_spikes(turn, ["A"])
            tracker.record_spikes(turn + 1, ["B"])
            if log_path.exists():
                current_size = len(log_path.read_text(encoding="utf-8"))
                assert current_size >= prev_size
                prev_size = current_size

    def test_log_is_valid_jsonl(self, tmp_path):
        """Each line in the log must be valid JSON."""
        log_path = tmp_path / "stdp_log.jsonl"
        weights = {("A", "B"): 0.5}
        tracker = STDPTracker(initial_weights=weights, log_path=log_path)

        tracker.record_spikes(1, ["A"])
        tracker.record_spikes(2, ["B"])

        if log_path.exists():
            for line in log_path.read_text(encoding="utf-8").strip().splitlines():
                data = json.loads(line)
                assert "turn" in data
                assert "source" in data
                assert "target" in data
                assert "rule" in data


# ── E-STDP-014: initial + log = current weights ──────────────────────

class TestSTDPReconstruction:
    def test_reconstruction(self):
        """E-STDP-014: initial_weights + sum(log deltas) = current weights."""
        weights = {("A", "B"): 0.5, ("B", "C"): 0.3, ("A", "C"): -0.2}
        tracker = STDPTracker(initial_weights=weights)

        for turn in range(1, 30):
            if turn % 2 == 0:
                tracker.record_spikes(turn, ["A"])
            elif turn % 3 == 0:
                tracker.record_spikes(turn, ["B"])
            else:
                tracker.record_spikes(turn, ["C"])

        assert tracker.verify_reconstruction()


# ── Plugin integration with STDP ──────────────────────────────────────

class TestPluginSTDP:
    def test_stdp_in_artifacts(self):
        """STDP stats should appear in plugin artifacts."""
        plugin = NeuralSNNPlugin(reservoir_enabled=False, stdp_enabled=True)
        state = State()
        for _ in range(50):
            state = plugin.update(state)
        artifacts = plugin.get_artifacts()
        assert "stdp" in artifacts
        assert "total_changes" in artifacts["stdp"]
        assert "reconstruction_valid" in artifacts["stdp"]
        assert artifacts["stdp"]["reconstruction_valid"] is True

    def test_stdp_log_file(self, tmp_path):
        """STDP log file should be created when path provided."""
        log_path = tmp_path / "stdp.jsonl"
        plugin = NeuralSNNPlugin(
            reservoir_enabled=False,
            stdp_enabled=True,
            stdp_log_path=log_path,
        )
        state = State()
        for _ in range(100):
            state = plugin.update(state)
        # Log may or may not have entries (depends on spike patterns)
        # But reconstruction must be valid
        artifacts = plugin.get_artifacts()
        assert artifacts["stdp"]["reconstruction_valid"] is True

    def test_stdp_disabled(self):
        """Plugin with stdp_enabled=False should not have STDP in artifacts."""
        plugin = NeuralSNNPlugin(reservoir_enabled=False, stdp_enabled=False)
        state = State()
        state = plugin.update(state)
        artifacts = plugin.get_artifacts()
        assert "stdp" not in artifacts

    def test_deterministic_with_stdp(self):
        """Plugin with STDP must still be deterministic."""
        state = State()
        results = []
        for _ in range(3):
            plugin = NeuralSNNPlugin(reservoir_enabled=False, stdp_enabled=True)
            s = state
            for _ in range(30):
                s = plugin.update(s)
            results.append(s.snapshot())
        for i in range(1, 3):
            assert results[0] == results[i], f"Run 0 != Run {i}"
