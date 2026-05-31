"""Boundedness checker for Pie kernel neural dynamics (V4.2 / V6a-R3a).

Implements a quadratic function V(x) for the 10D neural state space
(6 drives + 4 affect):

    V(x) = Σ_i w_i · (x_i − c_i)²

where c_i is the empirical attractor center of the neural dynamics.

**Important**: This is NOT a Lyapunov stability proof.  The Izhikevich
spiking neurons produce limit-cycle dynamics where V(x) is NOT
monotonically decreasing.  What we verify is:

1. **Positive semi-definiteness**: V(x) ≥ 0 for all x, V(c) = 0 at center
2. **Ultimate boundedness**: V ≤ V_max (theoretical max from [0,1] state
   space), all trajectories stay within [0,1] by construction
3. **Empirical settling**: trajectories from random initial states settle
   into a bounded attractor region (mean V decreases on average)
4. **No divergence**: no dimension leaves [0,1] at any step

See ``progetto/STABILITY.md`` for the full honest stability analysis:
what we prove (boundedness by construction), what we verify empirically
(trajectory checks), and what we do NOT claim (formal Lyapunov stability,
asymptotic stability).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Homeostatic targets (from SEED_V0)
# ---------------------------------------------------------------------------

HOMEOSTATIC_TARGETS: Dict[str, float] = {
    # Drives
    "curiosity": 0.78,
    "sociality": 0.72,
    "caution": 0.58,
    "agency": 0.62,
    "playfulness": 0.55,
    "fatigue": 0.10,
    # Affect baseline
    "valence": 0.10,
    "arousal": 0.20,
    "attention": 0.55,
    "tension": 0.15,
}

# Empirical attractor center: computed by running the neural SNN plugin from
# multiple initial conditions for 2000 steps and averaging the last 500.
# This is where the neural dynamics naturally settle, regardless of initial
# conditions.  Used as the reference point for the Lyapunov function.
ATTRACTOR_CENTER: Dict[str, float] = {
    "curiosity": 0.18,
    "sociality": 0.11,
    "caution": 0.14,
    "agency": 0.1175,
    "playfulness": 0.126,
    "fatigue": 0.115,
    "valence": 0.103,
    "arousal": 0.112,
    "attention": 0.107,
    "tension": 0.14,
}

DIMENSION_NAMES = list(HOMEOSTATIC_TARGETS.keys())


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class TrajectoryResult:
    """Result of checking a single trajectory."""

    v_positive: bool            # V(x) ≥ 0 at every point
    v_bounded: bool             # V ≤ V_max (theoretical bound)
    no_divergence: bool         # all values in [0, 1]
    settles: bool               # mean V last quarter ≤ mean V first quarter + ε
    max_v: float
    mean_v_first_quarter: float
    mean_v_last_quarter: float

    @property
    def passed(self) -> bool:
        return self.v_positive and self.v_bounded and self.no_divergence


@dataclass
class BatchResult:
    """Result of checking N trajectories."""

    n_trajectories: int
    n_passed: int
    n_failed: int
    pass_rate: float
    worst_max_v: float
    all_bounded: bool
    all_v_positive: bool
    all_no_divergence: bool
    n_settled: int
    settle_rate: float

    @property
    def passed(self) -> bool:
        return self.all_bounded and self.all_v_positive and self.all_no_divergence


# ---------------------------------------------------------------------------
# Deterministic LCG for seeded random initial states
# ---------------------------------------------------------------------------

def _lcg_float(seed: int, index: int) -> float:
    """Linear congruential generator → float in [0, 1].

    Same LCG used in reservoir.py for consistency.
    """
    state = (seed * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
    for _ in range(index):
        state = (state * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
    return (state & 0xFFFFFFFF) / 0xFFFFFFFF


def _random_state_dict(seed: int) -> Dict[str, float]:
    """Generate a random state dict with all dimensions in [0, 1]."""
    result = {}
    for i, name in enumerate(DIMENSION_NAMES):
        result[name] = round(_lcg_float(seed, i), 4)
    return result


# ---------------------------------------------------------------------------
# LyapunovChecker
# ---------------------------------------------------------------------------

class LyapunovChecker:
    """Lyapunov stability invariant checker for neural state dynamics.

    V(x) = Σ_i w_i · (x_i − c_i)²

    where c_i is the attractor center and w_i are per-dimension weights.

    For spiking neural systems, we verify ultimate boundedness rather
    than strict monotonic decrease of V, since periodic spiking causes
    transient increases in V that are part of normal dynamics.
    """

    def __init__(
        self,
        targets: Dict[str, float],
        weights: Optional[Dict[str, float]] = None,
    ) -> None:
        self.targets = dict(targets)
        self.weights = weights or {name: 1.0 for name in targets}
        # Theoretical maximum V: max distance from center in [0,1]^10
        self.v_max = sum(
            self.weights.get(name, 1.0) * max(c ** 2, (1.0 - c) ** 2)
            for name, c in self.targets.items()
        )

    # ── Core functions ────────────────────────────────────────────────

    def V(self, state_dict: Dict[str, float]) -> float:
        """Compute Lyapunov function value V(x)."""
        total = 0.0
        for name, target in self.targets.items():
            x = state_dict.get(name, target)
            w = self.weights.get(name, 1.0)
            total += w * (x - target) ** 2
        return total

    def delta_V(
        self,
        state_before: Dict[str, float],
        state_after: Dict[str, float],
    ) -> float:
        """Compute ΔV = V(after) - V(before)."""
        return self.V(state_after) - self.V(state_before)

    # ── Trajectory analysis ───────────────────────────────────────────

    def check_trajectory(
        self,
        trajectory: List[Dict[str, float]],
        settle_tolerance: float = 0.1,
    ) -> TrajectoryResult:
        """Check Lyapunov stability conditions along a trajectory.

        Checks:
        1. V(x) ≥ 0 at every point (positive semi-definiteness)
        2. V ≤ V_max (theoretical bound from [0,1] state space)
        3. No dimension leaves [0, 1] (no divergence)
        4. System settles: mean V in last quarter ≤ mean V in first
           quarter + tolerance (attractor convergence)
        """
        if len(trajectory) < 2:
            return TrajectoryResult(
                v_positive=True, v_bounded=True, no_divergence=True,
                settles=True, max_v=0.0,
                mean_v_first_quarter=0.0, mean_v_last_quarter=0.0,
            )

        # Compute V at each step
        v_values = [self.V(s) for s in trajectory]
        max_v = max(v_values)

        # 1. V ≥ 0 (trivially true for sum of squares, but verify)
        v_positive = all(v >= 0.0 for v in v_values)

        # 2. V ≤ V_max (theoretical bound)
        v_bounded = max_v <= self.v_max + 1e-6

        # 3. No divergence: all values in [0, 1]
        no_divergence = True
        for step in trajectory:
            for val in step.values():
                if val < -1e-6 or val > 1.0 + 1e-6:
                    no_divergence = False
                    break
            if not no_divergence:
                break

        # 4. Settling: compare first quarter vs last quarter
        n = len(v_values)
        q = max(1, n // 4)
        first_q = v_values[:q]
        last_q = v_values[-q:]
        mean_first = sum(first_q) / len(first_q)
        mean_last = sum(last_q) / len(last_q)
        settles = mean_last <= mean_first + settle_tolerance

        return TrajectoryResult(
            v_positive=v_positive,
            v_bounded=v_bounded,
            no_divergence=no_divergence,
            settles=settles,
            max_v=max_v,
            mean_v_first_quarter=mean_first,
            mean_v_last_quarter=mean_last,
        )

    # ── Trajectory generation ─────────────────────────────────────────

    def generate_trajectory(
        self,
        initial_state: "State",
        n_steps: int = 100,
        plugin: Optional[Any] = None,
    ) -> List[Dict[str, float]]:
        """Generate a trajectory by running the neural SNN plugin.

        Returns a list of state dicts (drives + affect merged).
        """
        from pie.state_engine.plugins.neural_snn import NeuralSNNPlugin

        if plugin is None:
            plugin = NeuralSNNPlugin(reservoir_enabled=False, stdp_enabled=False)

        state = initial_state
        trajectory = [self._state_to_dict(state)]

        for _ in range(n_steps):
            state = plugin.update(state)
            trajectory.append(self._state_to_dict(state))

        return trajectory

    @staticmethod
    def _state_to_dict(state: "State") -> Dict[str, float]:
        """Extract drives + affect into a flat dict."""
        d = {}
        d.update(state.drives)
        d.update(state.affect)
        return d

    # ── Batch check ───────────────────────────────────────────────────

    def run_batch_check(
        self,
        n_trajectories: int = 1000,
        n_steps: int = 100,
        seed_start: int = 0,
    ) -> BatchResult:
        """Run Lyapunov check over N random trajectories.

        Each trajectory starts from a random initial state in [0,1]^10.
        Uses deterministic LCG seeding for reproducibility.
        """
        from pie.contracts.state import State
        from pie.state_engine.plugins.neural_snn import NeuralSNNPlugin

        n_passed = 0
        n_failed = 0
        worst_max_v = 0.0
        all_bounded = True
        all_v_positive = True
        all_no_divergence = True
        n_settled = 0

        for i in range(n_trajectories):
            seed = seed_start + i
            rand_state = _random_state_dict(seed)

            drives = {
                name: rand_state[name]
                for name in ["curiosity", "sociality", "caution",
                             "agency", "playfulness", "fatigue"]
            }
            affect = {
                name: rand_state[name]
                for name in ["valence", "arousal", "attention", "tension"]
            }
            state = State(
                schema_version="0.1",
                drives=drives,
                affect=affect,
                turn_count=0,
                creator_anchor="lyapunov_test",
            )

            plugin = NeuralSNNPlugin(
                reservoir_enabled=False, stdp_enabled=False,
            )

            trajectory = self.generate_trajectory(state, n_steps, plugin)
            result = self.check_trajectory(trajectory)

            if result.passed:
                n_passed += 1
            else:
                n_failed += 1

            if result.settles:
                n_settled += 1
            if not result.v_bounded:
                all_bounded = False
            if not result.v_positive:
                all_v_positive = False
            if not result.no_divergence:
                all_no_divergence = False
            if result.max_v > worst_max_v:
                worst_max_v = result.max_v

        pass_rate = n_passed / n_trajectories if n_trajectories > 0 else 0.0
        settle_rate = n_settled / n_trajectories if n_trajectories > 0 else 0.0

        return BatchResult(
            n_trajectories=n_trajectories,
            n_passed=n_passed,
            n_failed=n_failed,
            pass_rate=pass_rate,
            worst_max_v=worst_max_v,
            all_bounded=all_bounded,
            all_v_positive=all_v_positive,
            all_no_divergence=all_no_divergence,
            n_settled=n_settled,
            settle_rate=settle_rate,
        )

    # ── Factory ───────────────────────────────────────────────────────

    @classmethod
    def from_seed(cls, seed_path: str = "progetto/SEED_V0.md") -> "LyapunovChecker":
        """Create checker with SEED_V0 targets (homeostatic setpoints)."""
        path = Path(seed_path)
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            targets = {}
            for name, val in data.get("drives", {}).items():
                targets[name] = float(val)
            for name, val in data.get("affect_baseline", {}).items():
                targets[name] = float(val)
            return cls(targets=targets)
        return cls(targets=dict(HOMEOSTATIC_TARGETS))

    @classmethod
    def from_attractor(cls) -> "LyapunovChecker":
        """Create checker with empirical attractor center as target."""
        return cls(targets=dict(ATTRACTOR_CENTER))


# V6a-R3a: Honest naming alias.
# The class checks boundedness, not Lyapunov stability in the formal sense.
# Kept LyapunovChecker for backward compatibility; prefer BoundednessChecker
# in new code.
BoundednessChecker = LyapunovChecker
