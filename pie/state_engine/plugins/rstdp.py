"""Reward-Modulated STDP (R-STDP) for Tier-1 synapses (V6a-R4).

Extends STDPTracker with a reward signal that gates weight changes.
Pure STDP is Hebbian (correlation-based); R-STDP adds a third factor
(reward) that modulates whether correlated spiking leads to actual
weight change.

Rule:
    eligibility[t] = γ * eligibility[t-1] + stdp_trace[t]
    Δw = η * reward * eligibility

This closes the learning loop: the system can improve its behavior
based on environmental feedback (tool success, recall hits, etc.).

Reward sources (deterministic only):
    tool_success     → +1
    tool_denied      → -1
    recall_hit       → +1
    goal_reached     → +2
    constraint_violated → -2

Reference: Izhikevich (2007) "Solving the Distal Reward Problem"
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pie.determinism import clamp_round
from pie.state_engine.plugins.stdp import (
    STDPTracker,
    STDPRecord,
    A_PLUS,
    A_MINUS,
    WEIGHT_MIN,
    WEIGHT_MAX,
)


# R-STDP parameters
GAMMA = 0.9         # eligibility trace decay
ETA = 0.005         # learning rate for reward-modulated updates
ELIGIBILITY_MIN = -1.0
ELIGIBILITY_MAX = 1.0

# Deterministic reward values
REWARD_MAP: Dict[str, float] = {
    "tool_success": 1.0,
    "tool_denied": -1.0,
    "recall_hit": 1.0,
    "goal_reached": 2.0,
    "constraint_violated": -2.0,
}


@dataclass
class RewardRecord:
    """Single reward event."""
    turn: int
    source: str
    value: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "turn": self.turn,
            "source": self.source,
            "value": self.value,
        }


class RewardSTDPTracker:
    """Reward-modulated STDP tracker.

    Extends STDPTracker with eligibility traces and reward gating.
    The eligibility trace is a decaying memory of recent STDP events;
    when a reward arrives, it modulates all eligible synapses.

    This is a composition (has-a) over STDPTracker, not inheritance,
    to keep the base class clean and allow independent testing.
    """

    def __init__(
        self,
        initial_weights: Dict[Tuple[str, str], float],
        log_path: Optional[Path] = None,
        reward_log_path: Optional[Path] = None,
        gamma: float = GAMMA,
        eta: float = ETA,
    ) -> None:
        self._stdp = STDPTracker(initial_weights=initial_weights, log_path=log_path)
        self._gamma = gamma
        self._eta = eta
        # Eligibility traces per synapse
        self._eligibility: Dict[Tuple[str, str], float] = {
            k: 0.0 for k in initial_weights
        }
        # Reward log
        self._reward_log: List[RewardRecord] = []
        self._reward_log_path = reward_log_path

    @property
    def weights(self) -> Dict[Tuple[str, str], float]:
        return self._stdp.weights

    @property
    def eligibility(self) -> Dict[Tuple[str, str], float]:
        return dict(self._eligibility)

    @property
    def reward_log(self) -> List[RewardRecord]:
        return list(self._reward_log)

    @property
    def stdp_log(self) -> List[STDPRecord]:
        return self._stdp.log

    def record_spikes(
        self,
        turn: int,
        spiked_neurons: List[str],
    ) -> List[STDPRecord]:
        """Process spikes: compute STDP traces and update eligibility.

        Does NOT directly change weights (that's done by apply_reward).
        The STDP trace is accumulated into the eligibility trace.
        """
        # Get STDP changes (these modify the underlying STDPTracker weights
        # via standard Hebbian STDP — we undo them and use eligibility instead)
        # Save pre-STDP weights
        pre_weights = dict(self._stdp._weights)

        # Run standard STDP to get trace signals
        changes = self._stdp.record_spikes(turn, spiked_neurons)

        # Revert weight changes from standard STDP — we use eligibility instead
        for rec in changes:
            key = (rec.source, rec.target)
            self._stdp._weights[key] = pre_weights[key]

        # Decay eligibility traces
        for key in self._eligibility:
            self._eligibility[key] = clamp_round(
                self._gamma * self._eligibility[key],
                ELIGIBILITY_MIN, ELIGIBILITY_MAX, 6,
            )

        # Accumulate STDP trace into eligibility
        for rec in changes:
            key = (rec.source, rec.target)
            self._eligibility[key] = clamp_round(
                self._eligibility[key] + rec.delta,
                ELIGIBILITY_MIN, ELIGIBILITY_MAX, 6,
            )

        return changes

    def apply_reward(self, turn: int, source: str, value: Optional[float] = None) -> None:
        """Apply a reward signal, modulating all eligible synapses.

        Args:
            turn: Current turn number.
            source: Reward source name (must be in REWARD_MAP or value provided).
            value: Override reward value (default: lookup in REWARD_MAP).
        """
        if value is None:
            value = REWARD_MAP.get(source, 0.0)

        # Record reward
        record = RewardRecord(turn=turn, source=source, value=value)
        self._reward_log.append(record)

        # Append to reward log file
        if self._reward_log_path:
            with self._reward_log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record.to_dict()) + "\n")

        # Modulate weights by reward × eligibility
        for key in sorted(self._eligibility.keys()):
            e = self._eligibility[key]
            if abs(e) < 1e-8:
                continue
            delta = self._eta * value * e
            old_w = self._stdp._weights[key]
            new_w = clamp_round(old_w + delta, WEIGHT_MIN, WEIGHT_MAX, 6)
            if new_w != old_w:
                self._stdp._weights[key] = new_w

    # -------------------------------------------------------------------
    # V6b-E1: Snapshot serialization
    # -------------------------------------------------------------------

    def serialize(self) -> Dict[str, Any]:
        """Return all mutable state as a JSON-serializable dict."""
        return {
            "stdp": self._stdp.serialize(),
            "eligibility": {
                f"{s}->{t}": e for (s, t), e in sorted(self._eligibility.items())
            },
            "gamma": self._gamma,
            "eta": self._eta,
        }

    def deserialize(self, data: Dict[str, Any]) -> None:
        """Restore mutable state from a previously serialized dict."""
        self._stdp.deserialize(data["stdp"])
        self._eligibility = {
            tuple(k.split("->")): float(v)
            for k, v in data["eligibility"].items()
        }
        self._gamma = data["gamma"]
        self._eta = data["eta"]
        # Reset reward log — not serialized (append-only JSONL is source of truth)
        self._reward_log = []

    def get_stats(self) -> Dict[str, Any]:
        """Return R-STDP statistics."""
        n_rewards = len(self._reward_log)
        total_reward = sum(r.value for r in self._reward_log)
        max_elig = max(abs(e) for e in self._eligibility.values()) if self._eligibility else 0.0
        return {
            "n_rewards": n_rewards,
            "total_reward": round(total_reward, 4),
            "max_eligibility": round(max_elig, 6),
            "gamma": self._gamma,
            "eta": self._eta,
            "stdp_stats": self._stdp.get_stats(),
        }
