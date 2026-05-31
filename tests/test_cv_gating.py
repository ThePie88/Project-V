"""V6a-R5 — Observable Causal Authority Tests.

Test IDs:
- E-CVG-010: CALM state produces expected gating
- E-CVG-020: ANXIOUS state restricts tools and recall
- E-CVG-030: CURIOUS state expands recall and verbosity
- I-CVG-040: CV_GATING event schema-validated with causal chain
- E-CVG-050: same input, 3 states, 3 measurably different outputs
- I-CVG-060: gating invariant holds (5 events for 5 decisions)

Validation targets are deterministic artifacts (GatingSnapshot,
CVGatingEvent), NOT LLM text output.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pie.contracts.cv_gating import (
    CVGatingEvent,
    GatingSnapshot,
    gate_tools,
    gate_memory,
    gate_verbosity,
    gate_consolidation,
    gate_counterfactual,
)


# ── State presets ────────────────────────────────────────────────────
# These represent ControlVector channel values for 3 emotional states.
# The values are what the kernel's neural pipeline produces for each state.

CALM_CV = {
    "tool_gate": 0.6,
    "memory_gate": 0.35,
    "verbosity_bias": 0.5,
    "consolidation_urgency": 0.2,
    "cf_k": 3,
}

ANXIOUS_CV = {
    "tool_gate": 0.15,
    "memory_gate": 0.08,
    "verbosity_bias": 0.1,
    "consolidation_urgency": 0.85,
    "cf_k": 5,
}

CURIOUS_CV = {
    "tool_gate": 0.8,
    "memory_gate": 0.55,
    "verbosity_bias": 0.85,
    "consolidation_urgency": 0.3,
    "cf_k": 2,
}


# ── E-CVG-010: CALM state ───────────────────────────────────────────

class TestCalmGating:
    """E-CVG-010: CALM state produces expected gating decisions."""

    def test_calm_tools_allowed(self):
        snap = GatingSnapshot.from_cv(**CALM_CV)
        assert snap.tool_allowed is True

    def test_calm_recall_k(self):
        snap = GatingSnapshot.from_cv(**CALM_CV)
        assert snap.recall_k == 3  # int(0.35 * 10) = 3

    def test_calm_verbosity_normal(self):
        snap = GatingSnapshot.from_cv(**CALM_CV)
        assert snap.verbosity == "normal"

    def test_calm_no_forced_consolidation(self):
        snap = GatingSnapshot.from_cv(**CALM_CV)
        assert snap.consolidate_now is False

    def test_calm_cf_k(self):
        snap = GatingSnapshot.from_cv(**CALM_CV)
        assert snap.cf_k == 3


# ── E-CVG-020: ANXIOUS state ────────────────────────────────────────

class TestAnxiousGating:
    """E-CVG-020: ANXIOUS state restricts tools and recall."""

    def test_anxious_tools_blocked(self):
        snap = GatingSnapshot.from_cv(**ANXIOUS_CV)
        assert snap.tool_allowed is False

    def test_anxious_recall_k_low(self):
        snap = GatingSnapshot.from_cv(**ANXIOUS_CV)
        assert snap.recall_k == 1  # max(1, int(0.08 * 10)) = max(1, 0) = 1

    def test_anxious_verbosity_low(self):
        snap = GatingSnapshot.from_cv(**ANXIOUS_CV)
        assert snap.verbosity == "low"

    def test_anxious_forced_consolidation(self):
        snap = GatingSnapshot.from_cv(**ANXIOUS_CV)
        assert snap.consolidate_now is True

    def test_anxious_cf_k_high(self):
        """High tension/caution → more alternatives (careful deliberation)."""
        snap = GatingSnapshot.from_cv(**ANXIOUS_CV)
        assert snap.cf_k == 5


# ── E-CVG-030: CURIOUS state ────────────────────────────────────────

class TestCuriousGating:
    """E-CVG-030: CURIOUS state expands recall and verbosity."""

    def test_curious_tools_allowed(self):
        snap = GatingSnapshot.from_cv(**CURIOUS_CV)
        assert snap.tool_allowed is True

    def test_curious_recall_k_high(self):
        snap = GatingSnapshot.from_cv(**CURIOUS_CV)
        assert snap.recall_k == 5  # int(0.55 * 10) = 5

    def test_curious_verbosity_high(self):
        snap = GatingSnapshot.from_cv(**CURIOUS_CV)
        assert snap.verbosity == "high"

    def test_curious_no_forced_consolidation(self):
        snap = GatingSnapshot.from_cv(**CURIOUS_CV)
        assert snap.consolidate_now is False

    def test_curious_cf_k_low(self):
        """Low tension → fewer alternatives (confident action)."""
        snap = GatingSnapshot.from_cv(**CURIOUS_CV)
        assert snap.cf_k == 2


# ── I-CVG-040: Schema validation ────────────────────────────────────

class TestCVGatingSchema:
    """I-CVG-040: CV_GATING event has all required fields and causal chain."""

    def test_gate_tools_schema(self):
        event = gate_tools(0.15)
        d = event.to_trace_dict()
        assert d["type"] == "CV_GATING"
        assert d["canale"] == "tool_gate"
        assert d["decisione"] == "BLOCK_TOOLS"
        assert "tools_disabled" in d["effetto"]
        assert "reason" in d

    def test_gate_memory_schema(self):
        event = gate_memory(0.55)
        d = event.to_trace_dict()
        assert d["canale"] == "memory_gate"
        assert "recall_k=5" in d["effetto"]

    def test_gate_verbosity_schema(self):
        event = gate_verbosity(0.85)
        d = event.to_trace_dict()
        assert d["canale"] == "verbosity_bias"
        assert d["decisione"] == "VERBOSITY_HIGH"

    def test_gate_consolidation_schema(self):
        event = gate_consolidation(0.85)
        d = event.to_trace_dict()
        assert d["canale"] == "consolidation_urgency"
        assert d["decisione"] == "CONSOLIDATE_NOW"

    def test_gate_counterfactual_schema(self):
        event = gate_counterfactual(cf_k=3, budget_k=5)
        d = event.to_trace_dict()
        assert d["canale"] == "cf_k"
        assert "deliberate_k=3" in d["effetto"]

    def test_event_json_serializable(self):
        """All events can be serialized to JSON."""
        events = [
            gate_tools(0.6),
            gate_memory(0.35),
            gate_verbosity(0.5),
            gate_consolidation(0.2),
            gate_counterfactual(3, 5),
        ]
        for event in events:
            d = event.to_trace_dict()
            serialized = json.dumps(d)
            parsed = json.loads(serialized)
            assert parsed["type"] == "CV_GATING"

    def test_schema_file_exists(self):
        """Schema file exists and is valid JSON."""
        path = Path("schemas/cv_gating_event.json")
        assert path.exists()
        schema = json.loads(path.read_text(encoding="utf-8"))
        assert schema["title"] == "CVGatingEvent"
        assert "canale" in schema["properties"]


# ── E-CVG-050: 3 states → 3 different gating decisions ──────────────

class TestThreeStates:
    """E-CVG-050: Same input, 3 states, 3 measurably different gating."""

    def test_all_three_differ(self):
        """CALM, ANXIOUS, CURIOUS produce 3 distinct GatingSnapshots."""
        calm = GatingSnapshot.from_cv(**CALM_CV)
        anxious = GatingSnapshot.from_cv(**ANXIOUS_CV)
        curious = GatingSnapshot.from_cv(**CURIOUS_CV)

        assert calm != anxious, f"CALM == ANXIOUS: {calm}"
        assert calm != curious, f"CALM == CURIOUS: {calm}"
        assert anxious != curious, f"ANXIOUS == CURIOUS: {anxious}"

    def test_tool_gate_diverges(self):
        """Tool access differs: CALM=True, ANXIOUS=False, CURIOUS=True."""
        calm = GatingSnapshot.from_cv(**CALM_CV)
        anxious = GatingSnapshot.from_cv(**ANXIOUS_CV)
        curious = GatingSnapshot.from_cv(**CURIOUS_CV)
        assert calm.tool_allowed is True
        assert anxious.tool_allowed is False
        assert curious.tool_allowed is True

    def test_recall_k_diverges(self):
        """Recall K differs across all 3 states."""
        calm = GatingSnapshot.from_cv(**CALM_CV)
        anxious = GatingSnapshot.from_cv(**ANXIOUS_CV)
        curious = GatingSnapshot.from_cv(**CURIOUS_CV)
        ks = {calm.recall_k, anxious.recall_k, curious.recall_k}
        assert len(ks) == 3, f"recall_k not fully divergent: {ks}"

    def test_verbosity_diverges(self):
        """Verbosity differs across all 3 states."""
        calm = GatingSnapshot.from_cv(**CALM_CV)
        anxious = GatingSnapshot.from_cv(**ANXIOUS_CV)
        curious = GatingSnapshot.from_cv(**CURIOUS_CV)
        vs = {calm.verbosity, anxious.verbosity, curious.verbosity}
        assert len(vs) == 3, f"verbosity not fully divergent: {vs}"

    def test_deterministic(self):
        """Same CV values → same gating snapshot."""
        s1 = GatingSnapshot.from_cv(**CALM_CV)
        s2 = GatingSnapshot.from_cv(**CALM_CV)
        assert s1 == s2


# ── I-CVG-060: Gating invariant ─────────────────────────────────────

class TestGatingInvariant:
    """I-CVG-060: 5 gating points → 5 CV_GATING events."""

    def test_five_events_for_five_decisions(self):
        """Emitting all 5 factory functions produces 5 events."""
        cv = CALM_CV
        events = [
            gate_tools(cv["tool_gate"]),
            gate_memory(cv["memory_gate"]),
            gate_verbosity(cv["verbosity_bias"]),
            gate_consolidation(cv["consolidation_urgency"]),
            gate_counterfactual(cv["cf_k"], budget_k=5),
        ]
        assert len(events) == 5
        channels = {e.canale for e in events}
        assert channels == {"tool_gate", "memory_gate", "verbosity_bias",
                           "consolidation_urgency", "cf_k"}

    def test_all_events_have_causal_chain(self):
        """Each event has non-empty decisione, effetto, reason."""
        for cv_dict in [CALM_CV, ANXIOUS_CV, CURIOUS_CV]:
            events = [
                gate_tools(cv_dict["tool_gate"]),
                gate_memory(cv_dict["memory_gate"]),
                gate_verbosity(cv_dict["verbosity_bias"]),
                gate_consolidation(cv_dict["consolidation_urgency"]),
                gate_counterfactual(cv_dict["cf_k"], budget_k=5),
            ]
            for event in events:
                assert event.decisione, f"Empty decisione for {event.canale}"
                assert event.effetto, f"Empty effetto for {event.canale}"
                assert event.reason, f"Empty reason for {event.canale}"
