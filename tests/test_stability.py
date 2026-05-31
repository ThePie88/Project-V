"""V6a-R3a — Honest Stability Claims Tests.

Test IDs:
- D-STB-010: STABILITY.md exists and covers all three sections
- P-STB-020: 10000 random states stay in [0,1]^10
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pie.lyapunov import (
    BoundednessChecker,
    LyapunovChecker,
    ATTRACTOR_CENTER,
    _random_state_dict,
)
from pie.contracts.state import State
from pie.state_engine.plugins.neural_snn import NeuralSNNPlugin


# ── D-STB-010: STABILITY.md completeness ──────────────────────────────

class TestStabilityDoc:
    """D-STB-010: STABILITY.md exists and covers all three sections."""

    @pytest.fixture(scope="class")
    def stability_text(self) -> str:
        path = Path("progetto/STABILITY.md")
        assert path.exists(), "progetto/STABILITY.md does not exist"
        return path.read_text(encoding="utf-8")

    def test_stability_md_exists(self, stability_text):
        assert len(stability_text) > 100

    def test_section1_boundedness(self, stability_text):
        """Section 1: Boundedness by construction (theorem)."""
        assert "Boundedness by Construction" in stability_text
        assert "Theorem" in stability_text or "theorem" in stability_text
        # Must mention all three clamping mechanisms
        assert "Pydantic" in stability_text or "validator" in stability_text
        assert "Izhikevich" in stability_text
        assert "Reservoir" in stability_text or "reservoir" in stability_text

    def test_section2_empirical(self, stability_text):
        """Section 2: Empirical trajectory verification."""
        assert "Empirical" in stability_text
        assert "10,000" in stability_text or "10000" in stability_text
        assert "NOT a formal proof" in stability_text or "not a formal proof" in stability_text.lower()

    def test_section3_non_claims(self, stability_text):
        """Section 3: What we do NOT claim."""
        assert "NOT Claim" in stability_text or "NOT claim" in stability_text or "Do NOT Claim" in stability_text
        assert "asymptotic" in stability_text.lower()
        assert "Bounded" in stability_text and "stable" in stability_text.lower()


# ── P-STB-020: 10000 random states stay bounded ──────────────────────

class TestBoundednessVerification:
    """P-STB-020: 10000 random states stay in [0,1]^10."""

    def test_boundedness_checker_alias(self):
        """BoundednessChecker is an alias for LyapunovChecker."""
        assert BoundednessChecker is LyapunovChecker

    def test_10000_states_bounded(self):
        """10000 random initial states → 100 steps each → all in [0,1]^10.

        This is a large-scale empirical verification, not a proof.
        """
        checker = BoundednessChecker.from_attractor()
        result = checker.run_batch_check(
            n_trajectories=10000,
            n_steps=100,
            seed_start=0,
        )
        assert result.all_bounded, (
            f"Unbounded: {result.n_failed}/{result.n_trajectories} failed, "
            f"worst V = {result.worst_max_v:.6f}"
        )
        assert result.all_no_divergence, "State left [0,1]^10"
        assert result.all_v_positive, "V was negative"
        # 100% pass rate
        assert result.pass_rate == 1.0
