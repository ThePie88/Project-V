"""Bifurcation detector for neural spike dynamics (V4.3).

Maps spike density μ(t) as a control parameter.  When μ crosses a
critical threshold (with hysteresis), the system undergoes a
*bifurcation* — the attractor topology changes and a behavioural
constraint is proposed to the CrystallizationEngine.

The neuron **proposes**, the kernel **decides**.

Control parameter:
    μ(t) = spike_count(window) / (n_neurons × window_size)

Bifurcation types (based on dominant spiking neurons):
    ANXIETY       — tension + arousal dominant
    IMPULSIVITY   — curiosity + playfulness dominant
    EXHAUSTION    — fatigue dominant
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_NEURONS = 10  # 6 drives + 4 affect

# Neuron groups for bifurcation type classification
ANXIETY_NEURONS = {"tension", "arousal"}
IMPULSIVITY_NEURONS = {"curiosity", "playfulness"}
EXHAUSTION_NEURONS = {"fatigue"}

# Event labels emitted on bifurcation (consumed by CrystallizationEngine)
BIFURC_EVENT_MAP = {
    "anxiety": "BIFURCATION_ANXIETY",
    "impulsivity": "BIFURCATION_IMPULSIVITY",
    "exhaustion": "BIFURCATION_EXHAUSTION",
}

BIFURC_RULE_MAP = {
    "anxiety": "BIFURC_ANXIETY",
    "impulsivity": "BIFURC_IMPULSIVITY",
    "exhaustion": "BIFURC_EXHAUSTION",
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class BifurcationEvent:
    """A single detected bifurcation."""

    turn: int
    mu: float                       # control parameter value at detection
    mu_crit: float                  # threshold that was crossed
    dominant_neurons: List[str]     # which neurons drove the spike density
    bifurcation_type: str           # "anxiety" | "impulsivity" | "exhaustion"
    constraint_rule_id: str         # e.g. "BIFURC_ANXIETY"
    event_label: str                # e.g. "BIFURCATION_ANXIETY"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "turn": self.turn,
            "mu": round(self.mu, 4),
            "mu_crit": self.mu_crit,
            "dominant_neurons": self.dominant_neurons,
            "bifurcation_type": self.bifurcation_type,
            "constraint_rule_id": self.constraint_rule_id,
            "event_label": self.event_label,
        }


@dataclass
class BifurcationDiagram:
    """Accumulates data points for a bifurcation diagram artifact."""

    points: List[Dict[str, Any]] = field(default_factory=list)

    def add_point(
        self,
        turn: int,
        mu: float,
        bifurcated: bool,
        event_label: Optional[str] = None,
    ) -> None:
        self.points.append({
            "turn": turn,
            "mu": round(mu, 4),
            "bifurcated": bifurcated,
            "event_label": event_label,
        })

    def to_json(self) -> str:
        return json.dumps(
            {"schema_version": "0.1", "type": "bifurcation_diagram", "points": self.points},
            indent=2,
        )

    def save(self, path: Path) -> None:
        path.write_text(self.to_json(), encoding="utf-8")


# ---------------------------------------------------------------------------
# BifurcationDetector
# ---------------------------------------------------------------------------

class BifurcationDetector:
    """Detects bifurcations in neural spike dynamics.

    Tracks spike density μ(t) as control parameter.
    When μ crosses critical threshold with hysteresis → bifurcation.
    Produces event labels for CrystallizationEngine.
    """

    def __init__(
        self,
        window: int = 5,
        mu_enter: float = 0.6,
        mu_exit: float = 0.4,
        sustain: int = 2,
    ) -> None:
        self._window = window
        self._mu_enter = mu_enter
        self._mu_exit = mu_exit
        self._sustain = sustain

        self._spike_history: List[List[str]] = []
        self._bifurcated: bool = False
        self._sustain_count: int = 0
        self._diagram = BifurcationDiagram()
        self._events: List[BifurcationEvent] = []

    # ── Recording ──────────────────────────────────────────────────────

    def record_spikes(self, turn: int, spiked_neurons: List[str]) -> None:
        """Record which neurons spiked this turn."""
        self._spike_history.append(list(spiked_neurons))

    # ── Control parameter ──────────────────────────────────────────────

    def compute_mu(self) -> float:
        """Compute current control parameter μ from spike window.

        μ = total_spikes_in_window / (N_NEURONS × window_size)
        """
        window = self._spike_history[-self._window:]
        if not window:
            return 0.0
        total_spikes = sum(len(spikes) for spikes in window)
        return total_spikes / (N_NEURONS * len(window))

    # ── Dominant neuron classification ─────────────────────────────────

    @staticmethod
    def dominant_neurons(window_spikes: List[List[str]]) -> List[str]:
        """Find which neurons are spiking most in the window."""
        counts: Counter = Counter()
        for spikes in window_spikes:
            counts.update(spikes)
        if not counts:
            return []
        # Return neurons sorted by count (descending), then name (deterministic)
        return [
            name for name, _ in sorted(
                counts.items(), key=lambda x: (-x[1], x[0])
            )
        ]

    @staticmethod
    def classify_bifurcation(dominant: List[str]) -> str:
        """Map dominant spiking neurons to bifurcation type.

        Priority: anxiety > exhaustion > impulsivity (default).
        """
        if not dominant:
            return "anxiety"  # default fallback

        top = set(dominant[:3])  # look at top-3 spikers

        # Count overlap with each group
        anxiety_score = len(top & ANXIETY_NEURONS)
        exhaustion_score = len(top & EXHAUSTION_NEURONS)
        impulsivity_score = len(top & IMPULSIVITY_NEURONS)

        if anxiety_score >= exhaustion_score and anxiety_score >= impulsivity_score:
            return "anxiety"
        if exhaustion_score >= impulsivity_score:
            return "exhaustion"
        return "impulsivity"

    # ── Bifurcation check ──────────────────────────────────────────────

    def check_bifurcation(self, turn: int) -> Optional[BifurcationEvent]:
        """Check if bifurcation threshold crossed.

        Returns BifurcationEvent if a NEW bifurcation is detected, else None.
        Records a diagram point every call.
        """
        mu = self.compute_mu()

        # Hysteresis state machine
        if not self._bifurcated:
            if mu >= self._mu_enter:
                self._sustain_count += 1
            else:
                self._sustain_count = 0

            if self._sustain_count >= self._sustain:
                # BIFURCATION DETECTED
                self._bifurcated = True
                window = self._spike_history[-self._window:]
                dominant = self.dominant_neurons(window)
                btype = self.classify_bifurcation(dominant)
                rule_id = BIFURC_RULE_MAP[btype]
                event_label = BIFURC_EVENT_MAP[btype]

                evt = BifurcationEvent(
                    turn=turn,
                    mu=mu,
                    mu_crit=self._mu_enter,
                    dominant_neurons=dominant,
                    bifurcation_type=btype,
                    constraint_rule_id=rule_id,
                    event_label=event_label,
                )
                self._events.append(evt)
                self._diagram.add_point(turn, mu, True, event_label)
                return evt
        else:
            # Already bifurcated — check for exit
            if mu < self._mu_exit:
                self._bifurcated = False
                self._sustain_count = 0

        # No bifurcation event this turn
        self._diagram.add_point(turn, mu, self._bifurcated, None)
        return None

    # ── Properties ─────────────────────────────────────────────────────

    @property
    def diagram(self) -> BifurcationDiagram:
        return self._diagram

    @property
    def events(self) -> List[BifurcationEvent]:
        return list(self._events)

    @property
    def is_bifurcated(self) -> bool:
        return self._bifurcated
