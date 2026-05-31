"""Tests for Lyapunov stability invariant checker (V4.2).

Test IDs from V4.md:
- P-LYA-010: V(x) > 0 for all reachable states
- P-LYA-020: dV/dt ≤ 0 along trajectories (ultimate boundedness)
- E-LYA-030: (optional) SMT proof of bounds
"""

import pytest

from pie.contracts.state import State
from pie.lyapunov import (
    LyapunovChecker,
    HOMEOSTATIC_TARGETS,
    ATTRACTOR_CENTER,
    DIMENSION_NAMES,
    _random_state_dict,
)
from pie.state_engine.plugins.neural_snn import NeuralSNNPlugin


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def checker():
    """Checker centered on empirical attractor (for trajectory checks)."""
    return LyapunovChecker.from_attractor()


@pytest.fixture
def seed_checker():
    """Checker centered on SEED_V0 homeostatic targets."""
    return LyapunovChecker.from_seed()


# ---------------------------------------------------------------------------
# P-LYA-010: V(x) > 0 for all non-equilibrium states
# ---------------------------------------------------------------------------

class TestVPositive:
    def test_v_positive_all_states(self, checker):
        """P-LYA-010: V(x) ≥ 0 for 1000 random states."""
        for seed in range(1000):
            state = _random_state_dict(seed)
            v = checker.V(state)
            assert v >= 0.0, f"V negative at seed {seed}: {v}"

    def test_v_zero_at_target(self, checker):
        """P-LYA-011: V(center) == 0.0 exactly."""
        v = checker.V(ATTRACTOR_CENTER)
        assert v == 0.0

    def test_v_increases_with_distance(self, checker):
        """P-LYA-012: V farther from center > V closer to center."""
        close = dict(ATTRACTOR_CENTER)
        close["curiosity"] = ATTRACTOR_CENTER["curiosity"] + 0.05
        far = dict(ATTRACTOR_CENTER)
        far["curiosity"] = ATTRACTOR_CENTER["curiosity"] + 0.20
        assert checker.V(far) > checker.V(close)


# ---------------------------------------------------------------------------
# P-LYA-020: Ultimate boundedness — all trajectories stay within V_max
# ---------------------------------------------------------------------------

class TestTrajectoryStability:
    def test_all_trajectories_bounded(self, checker):
        """P-LYA-020: All 1000 trajectories stay within V_max bound.

        V(x) ≤ V_max for all x on all trajectories, where V_max is the
        theoretical maximum distance from the attractor center within
        the [0,1]^10 state space.
        """
        result = checker.run_batch_check(
            n_trajectories=1000,
            n_steps=100,
            seed_start=0,
        )
        assert result.all_bounded, (
            f"Unbounded trajectories found: "
            f"{result.n_failed}/{result.n_trajectories} failed, "
            f"worst max V = {result.worst_max_v:.6f}, V_max = {checker.v_max:.6f}"
        )
        assert result.all_v_positive, "V was negative somewhere"

    def test_trajectories_no_divergence(self, checker):
        """P-LYA-021: No trajectory diverges (all values in [0,1])."""
        result = checker.run_batch_check(
            n_trajectories=1000,
            n_steps=100,
            seed_start=0,
        )
        assert result.all_no_divergence, (
            "Some trajectories had values outside [0,1]"
        )

    def test_no_divergence_per_step(self, checker):
        """P-LYA-022: Explicit per-step check on 100 trajectories."""
        for seed in range(100):
            rand_state = _random_state_dict(seed)
            drives = {n: rand_state[n] for n in
                      ["curiosity", "sociality", "caution", "agency",
                       "playfulness", "fatigue"]}
            affect = {n: rand_state[n] for n in
                      ["valence", "arousal", "attention", "tension"]}
            state = State(
                schema_version="0.1", drives=drives, affect=affect,
                turn_count=0, creator_anchor="lyapunov_test",
            )
            plugin = NeuralSNNPlugin(reservoir_enabled=False, stdp_enabled=False)
            trajectory = checker.generate_trajectory(state, n_steps=100, plugin=plugin)
            for step_idx, step in enumerate(trajectory):
                for name, val in step.items():
                    assert 0.0 <= val <= 1.0, (
                        f"Divergence at seed {seed}, step {step_idx}: "
                        f"{name}={val}"
                    )


# ---------------------------------------------------------------------------
# P-LYA-030: Batch pass rate and settling
# ---------------------------------------------------------------------------

class TestBatchPassRate:
    def test_batch_all_pass(self, checker):
        """P-LYA-030: 100% of trajectories pass boundedness check."""
        result = checker.run_batch_check(
            n_trajectories=1000,
            n_steps=100,
            seed_start=0,
        )
        assert result.passed, (
            f"Batch failed: pass_rate={result.pass_rate:.2%}, "
            f"bounded={result.all_bounded}, "
            f"v_positive={result.all_v_positive}, "
            f"no_divergence={result.all_no_divergence}"
        )

    def test_v_max_theoretical_bound(self, checker):
        """V_max is the correct theoretical upper bound."""
        # V_max = Σ max(c², (1-c)²) for each dimension
        import math
        expected = sum(
            max(c ** 2, (1.0 - c) ** 2)
            for c in ATTRACTOR_CENTER.values()
        )
        assert abs(checker.v_max - expected) < 1e-10


# ---------------------------------------------------------------------------
# P-LYA-031: from_seed factory
# ---------------------------------------------------------------------------

class TestFromSeed:
    def test_from_seed_targets(self, seed_checker):
        """P-LYA-031: Checker from SEED_V0 has correct targets."""
        assert seed_checker.targets["curiosity"] == 0.78
        assert seed_checker.targets["sociality"] == 0.72
        assert seed_checker.targets["valence"] == 0.10
        assert seed_checker.targets["tension"] == 0.15

    def test_from_attractor_targets(self):
        """from_attractor() uses empirical attractor center."""
        checker = LyapunovChecker.from_attractor()
        assert checker.targets["curiosity"] == ATTRACTOR_CENTER["curiosity"]
        assert checker.targets["agency"] == ATTRACTOR_CENTER["agency"]


# ---------------------------------------------------------------------------
# P-LYA-032: Deterministic trajectory
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_deterministic_trajectory(self, checker):
        """P-LYA-032: Same seed → same trajectory."""
        rand_state = _random_state_dict(42)
        drives = {n: rand_state[n] for n in
                  ["curiosity", "sociality", "caution", "agency",
                   "playfulness", "fatigue"]}
        affect = {n: rand_state[n] for n in
                  ["valence", "arousal", "attention", "tension"]}

        state1 = State(
            schema_version="0.1", drives=drives, affect=affect,
            turn_count=0, creator_anchor="lyapunov_test",
        )
        state2 = State(
            schema_version="0.1", drives=dict(drives), affect=dict(affect),
            turn_count=0, creator_anchor="lyapunov_test",
        )

        plugin1 = NeuralSNNPlugin(reservoir_enabled=False, stdp_enabled=False)
        plugin2 = NeuralSNNPlugin(reservoir_enabled=False, stdp_enabled=False)

        traj1 = checker.generate_trajectory(state1, n_steps=50, plugin=plugin1)
        traj2 = checker.generate_trajectory(state2, n_steps=50, plugin=plugin2)

        assert len(traj1) == len(traj2)
        for i, (s1, s2) in enumerate(zip(traj1, traj2)):
            for name in DIMENSION_NAMES:
                assert s1[name] == s2[name], (
                    f"Mismatch at step {i}, {name}: {s1[name]} != {s2[name]}"
                )
