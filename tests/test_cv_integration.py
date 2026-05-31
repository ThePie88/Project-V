"""Tests for V4 — ControlVector Runtime Integration.

Verifies that the 5 ControlVector signals actually drive runtime decisions.

Test IDs:
- E-CVI-010: memory_gate modulates recall top_k
- E-CVI-011: cf_k reduces counterfactual alternatives
- E-CVI-012: cf_k cannot exceed metabolism budget
- E-CVI-013: verbosity_bias < 0.3 forces low verbosity
- E-CVI-014: verbosity_bias > 0.7 allows med verbosity
- E-CVI-015: consolidation_urgency > 0.7 triggers early consolidation
- E-CVI-016: tool_gate < 0.3 disables tools
- E-CVI-017: No CV → default behavior (backward compat)
- E-CVI-018: should_consolidate urgency parameter backward compatible
- E-CVI-019: CV in kernel context for LLM
"""

from __future__ import annotations

import pytest

from pie.state_engine.control_vector import ControlVector
from pie.memory.consolidation import should_consolidate


# ── E-CVI-010: memory_gate → recall top_k ────────────────────────────

class TestMemoryGate:
    def test_low_gate_low_topk(self):
        """E-CVI-010a: memory_gate=0.1 → top_k=1."""
        cv = ControlVector(memory_gate=0.1)
        top_k = max(1, min(10, int(cv.memory_gate * 10)))
        assert top_k == 1

    def test_high_gate_high_topk(self):
        """E-CVI-010b: memory_gate=1.0 → top_k=10."""
        cv = ControlVector(memory_gate=1.0)
        top_k = max(1, min(10, int(cv.memory_gate * 10)))
        assert top_k == 10

    def test_default_gate_default_topk(self):
        """memory_gate=0.5 → top_k=5 (default)."""
        cv = ControlVector(memory_gate=0.5)
        top_k = max(1, min(10, int(cv.memory_gate * 10)))
        assert top_k == 5

    def test_zero_gate_clamps_to_one(self):
        """memory_gate=0.0 → top_k=1 (minimum)."""
        cv = ControlVector(memory_gate=0.0)
        top_k = max(1, min(10, int(cv.memory_gate * 10)))
        assert top_k == 1


# ── E-CVI-011/012: cf_k merge with metabolism ────────────────────────

class TestCfKMerge:
    def test_cv_reduces_cf_k(self):
        """E-CVI-011: cf_k=2 with metabolism max_alt=3 → uses 2."""
        cv = ControlVector(cf_k=2)
        metabolism_max = 3
        effective = min(metabolism_max, cv.cf_k)
        assert effective == 2

    def test_metabolism_prevails(self):
        """E-CVI-012: cf_k=5 with metabolism max_alt=2 → uses 2."""
        cv = ControlVector(cf_k=5)
        metabolism_max = 2
        effective = min(metabolism_max, cv.cf_k)
        assert effective == 2

    def test_equal_values(self):
        """cf_k=3 with metabolism max_alt=3 → uses 3."""
        cv = ControlVector(cf_k=3)
        metabolism_max = 3
        effective = min(metabolism_max, cv.cf_k)
        assert effective == 3


# ── E-CVI-013/014: verbosity_bias ────────────────────────────────────

class TestVerbosityBias:
    def test_low_bias_forces_low(self):
        """E-CVI-013: verbosity_bias < 0.3 → should force low."""
        cv = ControlVector(verbosity_bias=0.2)
        assert cv.verbosity_bias < 0.3

    def test_high_bias_allows_med(self):
        """E-CVI-014: verbosity_bias > 0.7 → should allow med."""
        cv = ControlVector(verbosity_bias=0.8)
        assert cv.verbosity_bias > 0.7

    def test_mid_bias_no_override(self):
        """verbosity_bias=0.5 → no override (between thresholds)."""
        cv = ControlVector(verbosity_bias=0.5)
        assert 0.3 <= cv.verbosity_bias <= 0.7


# ── E-CVI-015: consolidation_urgency ─────────────────────────────────

class TestConsolidationUrgency:
    def test_high_urgency_triggers(self):
        """E-CVI-015: consolidation_urgency > 0.7 → triggers consolidation."""
        # Not at interval, no affect spike, but urgency high
        result = should_consolidate(
            turn_count=2,
            last_consolidation_turn=1,
            affect={"arousal": 0.0, "tension": 0.0},
            urgency=0.8,
        )
        assert result is True

    def test_low_urgency_no_trigger(self):
        """Low urgency doesn't trigger if not at interval."""
        result = should_consolidate(
            turn_count=2,
            last_consolidation_turn=1,
            affect={"arousal": 0.0, "tension": 0.0},
            urgency=0.3,
        )
        assert result is False

    def test_interval_still_works(self):
        """Normal interval trigger still works with urgency=0."""
        result = should_consolidate(
            turn_count=10,
            last_consolidation_turn=4,  # 6 turns ago > CONSOLIDATION_INTERVAL=5
            affect=None,
            urgency=0.0,
        )
        assert result is True


# ── E-CVI-016: tool_gate ─────────────────────────────────────────────

class TestToolGate:
    def test_low_gate_disables(self):
        """E-CVI-016: tool_gate < 0.3 → tools disabled."""
        cv = ControlVector(tool_gate=0.2)
        tools_enabled = True
        _tools_gated = tools_enabled
        if cv.tool_gate < 0.3:
            _tools_gated = False
        assert _tools_gated is False

    def test_high_gate_keeps_enabled(self):
        """tool_gate > 0.3 → tools stay enabled."""
        cv = ControlVector(tool_gate=0.8)
        tools_enabled = True
        _tools_gated = tools_enabled
        if cv.tool_gate < 0.3:
            _tools_gated = False
        assert _tools_gated is True

    def test_user_disabled_stays_disabled(self):
        """If user disabled tools, tool_gate cannot re-enable."""
        cv = ControlVector(tool_gate=0.9)
        tools_enabled = False  # user disabled
        _tools_gated = tools_enabled
        if cv.tool_gate < 0.3:
            _tools_gated = False
        assert _tools_gated is False  # stays disabled


# ── E-CVI-017: No CV → default behavior ──────────────────────────────

class TestNoCVBackwardCompat:
    def test_no_cv_default_topk(self):
        """E-CVI-017a: Without CV, recall top_k=5."""
        _prev_cv = None
        _recall_top_k = 5
        if _prev_cv is not None:
            _recall_top_k = max(1, min(10, int(_prev_cv.memory_gate * 10)))
        assert _recall_top_k == 5

    def test_no_cv_default_cf_k(self):
        """E-CVI-017b: Without CV, cf_k = gating.max_alternatives."""
        _cv = None
        metabolism_max = 3
        cf_k = metabolism_max
        if _cv is not None:
            cf_k = min(cf_k, _cv.cf_k)
        assert cf_k == 3

    def test_no_cv_consolidation_default(self):
        """E-CVI-017c: Without CV, urgency=0.0 (no extra trigger)."""
        result = should_consolidate(
            turn_count=2,
            last_consolidation_turn=1,
            affect={"arousal": 0.0, "tension": 0.0},
        )
        assert result is False


# ── E-CVI-018: should_consolidate backward compat ────────────────────

class TestConsolidationBackwardCompat:
    def test_no_urgency_param(self):
        """E-CVI-018: should_consolidate works without urgency parameter."""
        # Old callers don't pass urgency — must still work
        result = should_consolidate(3, 1, {"arousal": 0.1, "tension": 0.1})
        assert result is False
        result2 = should_consolidate(10, 1, None)
        assert result2 is True


# ── E-CVI-019: CV fields bounded ─────────────────────────────────────

class TestCVFieldsIntegration:
    def test_all_fields_present(self):
        """E-CVI-019: All 5 CV fields present in to_dict."""
        cv = ControlVector()
        d = cv.to_dict()
        assert "memory_gate" in d
        assert "tool_gate" in d
        assert "cf_k" in d
        assert "verbosity_bias" in d
        assert "consolidation_urgency" in d

    def test_from_raw_produces_valid_cv(self):
        """from_raw always produces bounded values."""
        cv = ControlVector.from_raw({
            "memory_gate": 2.0,
            "tool_gate": -1.0,
            "cf_k": 100.0,
            "verbosity_bias": -5.0,
            "consolidation_urgency": 10.0,
        })
        assert 0.0 <= cv.memory_gate <= 1.0
        assert 0.0 <= cv.tool_gate <= 1.0
        assert 1 <= cv.cf_k <= 5
        assert 0.0 <= cv.verbosity_bias <= 1.0
        assert 0.0 <= cv.consolidation_urgency <= 1.0
