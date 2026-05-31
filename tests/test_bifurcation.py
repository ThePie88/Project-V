"""Tests for Bifurcation detector (V4.3).

Test IDs from V4.md:
- E-BIF-010: bifurcation threshold reproducible
- E-BIF-020: pre-threshold no constraint
- E-BIF-030: post-threshold constraint proposed
- E-BIF-040: bifurcation diagram artifact generated
"""

import json
import tempfile
from pathlib import Path

import pytest

from pie.bifurcation import (
    BifurcationDetector,
    BifurcationDiagram,
    BifurcationEvent,
    N_NEURONS,
    BIFURC_EVENT_MAP,
    BIFURC_RULE_MAP,
)
from pie.contracts.state import State
from pie.crystallization.engine import CrystallizationEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ALL_NEURONS = [
    "curiosity", "sociality", "caution", "agency", "playfulness", "fatigue",
    "valence", "arousal", "attention", "tension",
]


def _feed_low_spikes(det: BifurcationDetector, n_turns: int = 10) -> None:
    """Feed low spike density (1 neuron per turn) for n turns."""
    for t in range(n_turns):
        det.record_spikes(t, ["curiosity"])
        det.check_bifurcation(t)


def _feed_high_spikes(det: BifurcationDetector, n_turns: int, start: int = 0,
                      neurons: list = None) -> None:
    """Feed high spike density (many neurons per turn)."""
    if neurons is None:
        neurons = ALL_NEURONS[:8]  # 8/10 = 0.8 density per turn
    for t in range(start, start + n_turns):
        det.record_spikes(t, list(neurons))
        det.check_bifurcation(t)


# ---------------------------------------------------------------------------
# E-BIF-010: Bifurcation threshold reproducible
# ---------------------------------------------------------------------------

class TestReproducible:
    def test_same_sequence_same_result(self):
        """E-BIF-010: Identical spike sequences → identical bifurcation."""
        results = []
        for _ in range(3):
            det = BifurcationDetector(window=5, mu_enter=0.6, mu_exit=0.4, sustain=2)
            events = []
            for t in range(20):
                if t < 5:
                    det.record_spikes(t, ["curiosity"])
                else:
                    det.record_spikes(t, ALL_NEURONS[:7])
                evt = det.check_bifurcation(t)
                if evt is not None:
                    events.append(evt)
            results.append(events)

        # All 3 runs must produce identical events
        assert len(results[0]) > 0, "Should have detected at least one bifurcation"
        for i in range(1, 3):
            assert len(results[i]) == len(results[0])
            for a, b in zip(results[0], results[i]):
                assert a.turn == b.turn
                assert a.mu == b.mu
                assert a.bifurcation_type == b.bifurcation_type
                assert a.constraint_rule_id == b.constraint_rule_id


# ---------------------------------------------------------------------------
# E-BIF-020: Pre-threshold no constraint
# ---------------------------------------------------------------------------

class TestPreThreshold:
    def test_low_density_no_bifurcation(self):
        """E-BIF-020: μ < μ_crit → no bifurcation event."""
        det = BifurcationDetector(window=5, mu_enter=0.6, mu_exit=0.4, sustain=2)
        # Feed 1 spike per turn → μ = 1/10 = 0.1
        for t in range(20):
            det.record_spikes(t, ["curiosity"])
            evt = det.check_bifurcation(t)
            assert evt is None, f"Unexpected bifurcation at turn {t}"

        assert not det.is_bifurcated
        assert len(det.events) == 0

    def test_just_below_threshold(self):
        """μ just below enter threshold → no bifurcation."""
        det = BifurcationDetector(window=5, mu_enter=0.6, mu_exit=0.4, sustain=2)
        # 5 neurons per turn → μ = 5/10 = 0.5 < 0.6
        for t in range(20):
            det.record_spikes(t, ALL_NEURONS[:5])
            evt = det.check_bifurcation(t)
            assert evt is None
        assert not det.is_bifurcated


# ---------------------------------------------------------------------------
# E-BIF-030: Post-threshold constraint proposed
# ---------------------------------------------------------------------------

class TestPostThreshold:
    def test_high_density_triggers_bifurcation(self):
        """E-BIF-030: μ > μ_crit for sustain turns → bifurcation event."""
        det = BifurcationDetector(window=5, mu_enter=0.6, mu_exit=0.4, sustain=2)
        # Feed high spikes: 8/10 neurons → μ = 0.8
        events_found = []
        for t in range(10):
            det.record_spikes(t, ALL_NEURONS[:8])
            evt = det.check_bifurcation(t)
            if evt is not None:
                events_found.append(evt)

        assert len(events_found) == 1
        evt = events_found[0]
        assert evt.mu >= 0.6
        assert evt.mu_crit == 0.6
        assert evt.constraint_rule_id in BIFURC_RULE_MAP.values()
        assert evt.event_label in BIFURC_EVENT_MAP.values()
        assert len(evt.dominant_neurons) > 0
        assert det.is_bifurcated

    def test_sustain_required(self):
        """Must exceed threshold for 'sustain' consecutive turns."""
        det = BifurcationDetector(window=5, mu_enter=0.6, mu_exit=0.4, sustain=3)
        # High for 2 turns, then drop — should NOT trigger (sustain=3)
        det.record_spikes(0, ALL_NEURONS[:8])
        assert det.check_bifurcation(0) is None
        det.record_spikes(1, ALL_NEURONS[:8])
        assert det.check_bifurcation(1) is None
        # Drop
        det.record_spikes(2, ["curiosity"])
        assert det.check_bifurcation(2) is None
        assert not det.is_bifurcated

    def test_event_has_correct_fields(self):
        """Bifurcation event has all required fields."""
        det = BifurcationDetector(window=3, mu_enter=0.5, mu_exit=0.3, sustain=2)
        for t in range(5):
            det.record_spikes(t, ALL_NEURONS[:7])
            evt = det.check_bifurcation(t)
            if evt is not None:
                d = evt.to_dict()
                assert "turn" in d
                assert "mu" in d
                assert "mu_crit" in d
                assert "dominant_neurons" in d
                assert "bifurcation_type" in d
                assert "constraint_rule_id" in d
                assert "event_label" in d
                return
        pytest.fail("No bifurcation detected")


# ---------------------------------------------------------------------------
# E-BIF-040: Bifurcation diagram artifact
# ---------------------------------------------------------------------------

class TestDiagramArtifact:
    def test_diagram_generated(self):
        """E-BIF-040: BifurcationDiagram has points after run."""
        det = BifurcationDetector(window=5, mu_enter=0.6, mu_exit=0.4, sustain=2)
        for t in range(10):
            det.record_spikes(t, ALL_NEURONS[:8])
            det.check_bifurcation(t)

        diagram = det.diagram
        assert len(diagram.points) == 10
        # Each point has required fields
        for p in diagram.points:
            assert "turn" in p
            assert "mu" in p
            assert "bifurcated" in p

    def test_diagram_save_json(self):
        """Diagram saves as valid JSON with schema_version."""
        det = BifurcationDetector(window=5, mu_enter=0.6, mu_exit=0.4, sustain=2)
        for t in range(5):
            det.record_spikes(t, ALL_NEURONS[:8])
            det.check_bifurcation(t)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bifurcation_diagram.json"
            det.diagram.save(path)
            assert path.exists()
            data = json.loads(path.read_text(encoding="utf-8"))
            assert data["schema_version"] == "0.1"
            assert data["type"] == "bifurcation_diagram"
            assert isinstance(data["points"], list)
            assert len(data["points"]) == 5

    def test_diagram_marks_bifurcation_point(self):
        """Diagram correctly marks the bifurcation turn."""
        det = BifurcationDetector(window=3, mu_enter=0.5, mu_exit=0.3, sustain=2)
        for t in range(8):
            det.record_spikes(t, ALL_NEURONS[:7])
            det.check_bifurcation(t)

        bifurcated_points = [p for p in det.diagram.points if p["bifurcated"]]
        assert len(bifurcated_points) > 0
        # At least one has an event_label
        labeled = [p for p in bifurcated_points if p["event_label"] is not None]
        assert len(labeled) >= 1


# ---------------------------------------------------------------------------
# Support tests: μ computation
# ---------------------------------------------------------------------------

class TestMuComputation:
    def test_mu_empty(self):
        det = BifurcationDetector(window=5)
        assert det.compute_mu() == 0.0

    def test_mu_full_spikes(self):
        """All neurons spike every turn → μ = 1.0."""
        det = BifurcationDetector(window=5)
        for t in range(5):
            det.record_spikes(t, ALL_NEURONS)
        assert abs(det.compute_mu() - 1.0) < 1e-6

    def test_mu_half_spikes(self):
        """5/10 neurons spike every turn → μ = 0.5."""
        det = BifurcationDetector(window=5)
        for t in range(5):
            det.record_spikes(t, ALL_NEURONS[:5])
        assert abs(det.compute_mu() - 0.5) < 1e-6

    def test_mu_windowed(self):
        """Only last 'window' turns count."""
        det = BifurcationDetector(window=3)
        # First 5 turns: all neurons spike
        for t in range(5):
            det.record_spikes(t, ALL_NEURONS)
        # Next 3 turns: only 1 neuron
        for t in range(5, 8):
            det.record_spikes(t, ["curiosity"])
        # Window covers turns 5,6,7 → μ = 3 / (10*3) = 0.1
        assert abs(det.compute_mu() - 0.1) < 1e-6


# ---------------------------------------------------------------------------
# Support tests: hysteresis
# ---------------------------------------------------------------------------

class TestHysteresis:
    def test_enter_at_mu_enter(self):
        """Enters bifurcated state at mu_enter."""
        det = BifurcationDetector(window=5, mu_enter=0.6, mu_exit=0.4, sustain=2)
        # Build μ ≥ 0.6: 7/10 neurons = 0.7
        for t in range(5):
            det.record_spikes(t, ALL_NEURONS[:7])
            det.check_bifurcation(t)
        assert det.is_bifurcated

    def test_stays_bifurcated_above_exit(self):
        """Once bifurcated, stays bifurcated if μ > mu_exit."""
        det = BifurcationDetector(window=5, mu_enter=0.6, mu_exit=0.4, sustain=2)
        # Enter bifurcation
        for t in range(5):
            det.record_spikes(t, ALL_NEURONS[:7])
            det.check_bifurcation(t)
        assert det.is_bifurcated

        # Drop to μ = 0.5 (above exit=0.4) → still bifurcated
        for t in range(5, 15):
            det.record_spikes(t, ALL_NEURONS[:5])
            det.check_bifurcation(t)
        assert det.is_bifurcated

    def test_exits_below_mu_exit(self):
        """Exits bifurcated state when μ < mu_exit."""
        det = BifurcationDetector(window=5, mu_enter=0.6, mu_exit=0.4, sustain=2)
        # Enter bifurcation
        for t in range(5):
            det.record_spikes(t, ALL_NEURONS[:7])
            det.check_bifurcation(t)
        assert det.is_bifurcated

        # Drop to μ = 0.1 → exits
        for t in range(5, 15):
            det.record_spikes(t, ["curiosity"])
            det.check_bifurcation(t)
        assert not det.is_bifurcated


# ---------------------------------------------------------------------------
# Support tests: dominant neuron classification
# ---------------------------------------------------------------------------

class TestDominantNeurons:
    def test_anxiety_classification(self):
        """tension + arousal dominant → anxiety."""
        det = BifurcationDetector()
        spikes = [["tension", "arousal", "curiosity"]] * 5
        dominant = det.dominant_neurons(spikes)
        btype = det.classify_bifurcation(dominant)
        assert btype == "anxiety"

    def test_impulsivity_classification(self):
        """curiosity + playfulness dominant → impulsivity."""
        det = BifurcationDetector()
        spikes = [["curiosity", "playfulness", "sociality"]] * 5
        dominant = det.dominant_neurons(spikes)
        btype = det.classify_bifurcation(dominant)
        assert btype == "impulsivity"

    def test_exhaustion_classification(self):
        """fatigue dominant → exhaustion."""
        det = BifurcationDetector()
        spikes = [["fatigue", "fatigue", "caution"]] * 5
        dominant = det.dominant_neurons(spikes)
        btype = det.classify_bifurcation(dominant)
        assert btype == "exhaustion"


# ---------------------------------------------------------------------------
# Integration: BifurcationDetector + CrystallizationEngine
# ---------------------------------------------------------------------------

class TestIntegrationCrystallizer:
    def test_bifurcation_event_triggers_crystallizer(self):
        """Bifurcation event label is consumable by CrystallizationEngine."""
        # 1. Detect bifurcation
        det = BifurcationDetector(window=3, mu_enter=0.5, mu_exit=0.3, sustain=2)
        bif_event = None
        for t in range(10):
            det.record_spikes(t, ["tension", "arousal", "caution", "curiosity",
                                   "sociality", "fatigue", "valence"])
            evt = det.check_bifurcation(t)
            if evt is not None:
                bif_event = evt
                break

        assert bif_event is not None, "Should have detected bifurcation"

        # 2. Feed event to CrystallizationEngine
        engine = CrystallizationEngine()
        state = State(
            schema_version="0.1",
            drives={"curiosity": 0.7, "sociality": 0.5, "caution": 0.5,
                    "agency": 0.5, "playfulness": 0.5, "fatigue": 0.3},
            affect={"valence": 0.3, "arousal": 0.8, "attention": 0.5, "tension": 0.8},
            turn_count=bif_event.turn,
            creator_anchor="bifurcation_test",
        )
        events = [{
            "id": 1,
            "type": "BIFURCATION",
            "content": {"label": bif_event.event_label},
        }]
        records = engine.propose_constraints(
            events=events,
            state=state,
            logical_time={"session": 1, "turn": bif_event.turn},
            existing_constraints=[],
        )

        # Should find at least one BIFURC_* constraint
        bifurc_records = [r for r in records if r.rule_id.startswith("BIFURC_")]
        assert len(bifurc_records) >= 1, (
            f"CrystallizationEngine should produce BIFURC_* constraint, got: "
            f"{[r.rule_id for r in records]}"
        )
        rec = bifurc_records[0]
        assert rec.family == "BIFURCATION_NEURAL"
        assert len(rec.trigger_events) > 0
        assert rec.explanation != ""
