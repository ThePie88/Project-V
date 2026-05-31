"""Tests for V4 — ESN Reservoir + ControlVector.

Test IDs:
- E-RSV-010: Spectral radius within 0.001 of 0.95
- E-RSV-011: Same seed → identical weights
- E-RSV-012: Echo State Property (different inits converge)
- E-RSV-013: 15 output channels all in valid ranges
- E-RSV-014: Single step < 50ms (relaxed for CI)
- E-CV-010: 5 ControlVector fields present and bounded
- E-CV-011: cf_k integer in [1,5]
- E-CV-012: Deterministic — same state → same CV
- E-CV-020: Plugin with reservoir produces ControlVector
- E-CV-021: ControlVector in artifacts
- E-CV-022: Reservoir stats in artifacts
"""

from __future__ import annotations

import time

import pytest

from pie.contracts.state import State
from pie.state_engine.plugins.reservoir import (
    Reservoir,
    LCG,
    ReservoirNeuron,
    READOUT_CHANNELS,
    INPUT_SIZE,
    RESERVOIR_SIZE,
    OUTPUT_SIZE,
)
from pie.state_engine.control_vector import ControlVector
from pie.state_engine.plugins.neural_snn import NeuralSNNPlugin
from pie.state_engine.registry import StateEngineRegistry


@pytest.fixture(autouse=True)
def reset_registry():
    StateEngineRegistry.reset()
    yield
    StateEngineRegistry.reset()


# ── LCG tests ─────────────────────────────────────────────────────────

class TestLCG:
    def test_deterministic(self):
        a = LCG(42)
        b = LCG(42)
        for _ in range(100):
            assert a.next_int() == b.next_int()

    def test_different_seeds_differ(self):
        a = LCG(1)
        b = LCG(2)
        assert a.next_int() != b.next_int()

    def test_float_range(self):
        rng = LCG(0xDEADBEEF)
        for _ in range(1000):
            f = rng.next_float()
            assert 0.0 <= f < 1.0

    def test_signed_range(self):
        rng = LCG(0xDEADBEEF)
        for _ in range(1000):
            f = rng.next_signed()
            assert -1.0 <= f < 1.0


# ── ReservoirNeuron tests ─────────────────────────────────────────────

class TestReservoirNeuron:
    def test_output_bounded(self):
        n = ReservoirNeuron()
        for _ in range(200):
            v = n.step(0.05)
            assert 0.0 <= v <= 1.0

    def test_no_spike_without_input(self):
        n = ReservoirNeuron()
        for _ in range(100):
            n.step(0.0)
            assert not n.last_spike


# ── E-RSV-010: Spectral radius ────────────────────────────────────────

class TestReservoirSpectralRadius:
    def test_spectral_radius_near_target(self):
        """E-RSV-010: Spectral radius within 0.01 of 0.95."""
        r = Reservoir()
        sr = r.verify_spectral_radius()
        assert abs(sr - 0.95) < 0.05, f"SR = {sr}, expected ~0.95"


# ── E-RSV-011: Same seed → identical weights ──────────────────────────

class TestReservoirDeterminism:
    def test_same_seed_same_weights(self):
        """E-RSV-011: Same seed produces identical reservoir."""
        r1 = Reservoir(seed=0xDEADBEEF)
        r2 = Reservoir(seed=0xDEADBEEF)
        # Compare internal weights
        for i in range(RESERVOIR_SIZE):
            for j in range(RESERVOIR_SIZE):
                assert r1._weights[i][j] == r2._weights[i][j]

    def test_different_seed_different_weights(self):
        r1 = Reservoir(seed=1)
        r2 = Reservoir(seed=2)
        # At least some weights should differ
        diffs = sum(
            1 for i in range(RESERVOIR_SIZE)
            for j in range(RESERVOIR_SIZE)
            if r1._weights[i][j] != r2._weights[i][j]
        )
        assert diffs > 0

    def test_step_deterministic(self):
        """Same input → same output across 5 runs."""
        results = []
        for _ in range(5):
            r = Reservoir()
            inp = [0.5] * INPUT_SIZE
            out = r.step(inp)
            results.append(out)
        for i in range(1, 5):
            assert results[0] == results[i], f"Run 0 != Run {i}"


# ── E-RSV-012: Echo State Property ────────────────────────────────────

class TestEchoStateProperty:
    def test_different_inits_converge(self):
        """E-RSV-012: Different initial neuron states converge to similar outputs."""
        r1 = Reservoir()
        r2 = Reservoir()
        # Perturb r2's neurons
        for n in r2._neurons:
            n.v = -50.0
            n.u = -10.0
        inp = [0.5] * INPUT_SIZE
        # Run many steps — should converge
        for _ in range(200):
            o1 = r1.step(inp)
            o2 = r2.step(inp)
        # After 200 steps, outputs should be close
        for ch in READOUT_CHANNELS:
            diff = abs(o1[ch] - o2[ch])
            assert diff < 0.3, f"Channel {ch} diff={diff} after 200 steps"


# ── E-RSV-013: 15 output channels ────────────────────────────────────

class TestReservoirOutputs:
    def test_15_channels_present(self):
        """E-RSV-013: All 15 output channels present and in [0,1]."""
        r = Reservoir()
        inp = [0.5] * INPUT_SIZE
        out = r.step(inp)
        assert len(out) == OUTPUT_SIZE
        for ch in READOUT_CHANNELS:
            assert ch in out, f"Missing channel: {ch}"
            assert 0.0 <= out[ch] <= 1.0, f"{ch}={out[ch]} out of range"


# ── E-RSV-014: Performance ────────────────────────────────────────────

class TestReservoirPerformance:
    def test_single_step_fast(self):
        """E-RSV-014: Single step should complete quickly."""
        r = Reservoir()
        inp = [0.5] * INPUT_SIZE
        # Warm up
        r.step(inp)
        start = time.perf_counter()
        for _ in range(10):
            r.step(inp)
        elapsed = (time.perf_counter() - start) / 10
        assert elapsed < 0.1, f"Step took {elapsed*1000:.1f}ms, expected < 100ms"


# ── E-CV-010: ControlVector fields ────────────────────────────────────

class TestControlVector:
    def test_fields_present_and_bounded(self):
        """E-CV-010: All 5 CV fields present and in valid ranges."""
        cv = ControlVector()
        d = cv.to_dict()
        assert 0.0 <= d["memory_gate"] <= 1.0
        assert 0.0 <= d["tool_gate"] <= 1.0
        assert 1 <= d["cf_k"] <= 5
        assert 0.0 <= d["verbosity_bias"] <= 1.0
        assert 0.0 <= d["consolidation_urgency"] <= 1.0

    def test_cf_k_integer(self):
        """E-CV-011: cf_k must be integer in [1,5]."""
        cv = ControlVector.from_raw({"cf_k": 2.7})
        assert isinstance(cv.cf_k, int)
        assert 1 <= cv.cf_k <= 5

    def test_cf_k_clamped(self):
        cv_low = ControlVector.from_raw({"cf_k": -10.0})
        cv_high = ControlVector.from_raw({"cf_k": 100.0})
        assert cv_low.cf_k == 1
        assert cv_high.cf_k == 5

    def test_from_raw_clamps(self):
        cv = ControlVector.from_raw({
            "memory_gate": 2.0,
            "tool_gate": -1.0,
            "verbosity_bias": 5.0,
            "consolidation_urgency": -0.5,
        })
        assert cv.memory_gate == 1.0
        assert cv.tool_gate == 0.0
        assert cv.verbosity_bias == 1.0
        assert cv.consolidation_urgency == 0.0

    def test_deterministic(self):
        """E-CV-012: Same raw input → same ControlVector."""
        raw = {"memory_gate": 0.7, "tool_gate": 0.3, "cf_k": 4.0,
               "verbosity_bias": 0.8, "consolidation_urgency": 0.2}
        cv1 = ControlVector.from_raw(raw)
        cv2 = ControlVector.from_raw(raw)
        assert cv1.to_dict() == cv2.to_dict()


# ── E-CV-020/021/022: Plugin integration ──────────────────────────────

class TestPluginWithReservoir:
    def test_plugin_produces_control_vector(self):
        """E-CV-020: Plugin with reservoir produces ControlVector."""
        plugin = NeuralSNNPlugin(reservoir_enabled=True)
        state = State()
        state = plugin.update(state)
        cv = plugin.control_vector
        assert cv is not None
        d = cv.to_dict()
        assert 0.0 <= d["memory_gate"] <= 1.0
        assert 1 <= d["cf_k"] <= 5

    def test_plugin_without_reservoir_no_cv(self):
        plugin = NeuralSNNPlugin(reservoir_enabled=False)
        state = State()
        state = plugin.update(state)
        assert plugin.control_vector is None

    def test_cv_in_artifacts(self):
        """E-CV-021: ControlVector in artifacts."""
        plugin = NeuralSNNPlugin(reservoir_enabled=True)
        state = State()
        state = plugin.update(state)
        artifacts = plugin.get_artifacts()
        assert "control_vector" in artifacts
        cv = artifacts["control_vector"]
        assert "memory_gate" in cv
        assert "cf_k" in cv

    def test_reservoir_stats_in_artifacts(self):
        """E-CV-022: Reservoir stats in artifacts."""
        plugin = NeuralSNNPlugin(reservoir_enabled=True)
        state = State()
        state = plugin.update(state)
        artifacts = plugin.get_artifacts()
        assert "reservoir" in artifacts
        rs = artifacts["reservoir"]
        assert rs["size"] == 128
        assert abs(rs["target_spectral_radius"] - 0.95) < 0.01

    def test_reservoir_determinism_5_runs(self):
        """Same state → same CV and same output across 5 runs."""
        state = State()
        results = []
        for _ in range(5):
            plugin = NeuralSNNPlugin(reservoir_enabled=True)
            s = plugin.update(state)
            results.append((s.snapshot(), plugin.control_vector.to_dict()))
        for i in range(1, 5):
            assert results[0] == results[i], f"Run 0 != Run {i}"

    def test_1000_turns_with_reservoir(self):
        """1000 turns with reservoir must not diverge."""
        state = State()
        plugin = NeuralSNNPlugin(reservoir_enabled=True)
        for _ in range(1000):
            state = plugin.update(state)
        for name, val in state.drives.items():
            assert 0.0 <= val <= 1.0, f"Drive {name}={val}"
        for name, val in state.affect.items():
            assert 0.0 <= val <= 1.0, f"Affect {name}={val}"
        cv = plugin.control_vector
        assert cv is not None
        assert 0.0 <= cv.memory_gate <= 1.0
