"""Spike-Timing Dependent Plasticity (STDP) for Tier-1 synapses.

Modifies Tier-1 synapse weights based on relative spike timing:
- Pre-before-post → potentiation (weight increases)
- Post-before-pre → depression (weight decreases)

Applied ONLY to the 11 Tier-1 synapses (SYNAPSE_WEIGHTS).
The reservoir (Tier-2) weights are FIXED per ESN theory.

All changes are logged to an append-only JSONL file for full
auditability and reconstructibility.

Reference: Bi & Poo (1998), Song et al. (2000)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pie.determinism import clamp_round


# STDP parameters
A_PLUS = 0.01       # potentiation amplitude
A_MINUS = 0.012     # depression amplitude (slightly stronger for stability)
WEIGHT_MIN = -1.0
WEIGHT_MAX = 1.0


@dataclass
class STDPRecord:
    """Single weight change event."""
    turn: int
    source: str
    target: str
    old_weight: float
    new_weight: float
    delta: float
    rule: str  # "potentiation" or "depression"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "turn": self.turn,
            "source": self.source,
            "target": self.target,
            "old_weight": self.old_weight,
            "new_weight": self.new_weight,
            "delta": self.delta,
            "rule": self.rule,
        }


class STDPTracker:
    """Tracks spike timing and applies STDP to Tier-1 synapse weights.

    The tracker maintains:
    - Current synapse weights (mutable copy of SYNAPSE_WEIGHTS)
    - Last spike turn for each neuron
    - Append-only log of all weight changes
    """

    def __init__(
        self,
        initial_weights: Dict[Tuple[str, str], float],
        log_path: Optional[Path] = None,
    ) -> None:
        # Mutable copy of synapse weights
        self._weights: Dict[Tuple[str, str], float] = dict(initial_weights)
        self._initial_weights: Dict[Tuple[str, str], float] = dict(initial_weights)
        # Last spike turn per neuron (None = never spiked)
        self._last_spike_turn: Dict[str, Optional[int]] = {}
        # In-memory log
        self._log: List[STDPRecord] = []
        # Optional file log
        self._log_path = log_path

    @property
    def weights(self) -> Dict[Tuple[str, str], float]:
        """Current synapse weights (read-only copy)."""
        return dict(self._weights)

    @property
    def log(self) -> List[STDPRecord]:
        return list(self._log)

    def record_spikes(
        self,
        turn: int,
        spiked_neurons: List[str],
    ) -> List[STDPRecord]:
        """Process spikes for this turn and apply STDP rules.

        For each synapse (src, tgt):
        - If src spiked this turn and tgt spiked recently → potentiation
        - If tgt spiked this turn and src spiked recently → depression

        Returns list of weight changes made this turn.
        """
        changes: List[STDPRecord] = []
        spiked_set = set(spiked_neurons)

        # Apply STDP for each synapse
        for (src, tgt) in sorted(self._weights.keys()):
            src_spiked_now = src in spiked_set
            tgt_spiked_now = tgt in spiked_set

            src_last = self._last_spike_turn.get(src)
            tgt_last = self._last_spike_turn.get(tgt)

            # Pre-before-post: src spiked recently, tgt spikes now
            # (pre fires first → post fires later → potentiation)
            if tgt_spiked_now and src_last is not None and src_last < turn:
                dt = turn - src_last
                if dt <= 5:  # only recent timing matters
                    delta = A_PLUS * (1.0 / dt)  # closer → stronger
                    old_w = self._weights[(src, tgt)]
                    new_w = clamp_round(old_w + delta, WEIGHT_MIN, WEIGHT_MAX, 6)
                    if new_w != old_w:
                        record = STDPRecord(
                            turn=turn, source=src, target=tgt,
                            old_weight=old_w, new_weight=new_w,
                            delta=round(new_w - old_w, 6),
                            rule="potentiation",
                        )
                        self._weights[(src, tgt)] = new_w
                        changes.append(record)

            # Post-before-pre: tgt spiked recently, src spikes now
            # (post fires first → pre fires later → depression)
            if src_spiked_now and tgt_last is not None and tgt_last < turn:
                dt = turn - tgt_last
                if dt <= 5:
                    delta = -A_MINUS * (1.0 / dt)
                    old_w = self._weights[(src, tgt)]
                    new_w = clamp_round(old_w + delta, WEIGHT_MIN, WEIGHT_MAX, 6)
                    if new_w != old_w:
                        record = STDPRecord(
                            turn=turn, source=src, target=tgt,
                            old_weight=old_w, new_weight=new_w,
                            delta=round(new_w - old_w, 6),
                            rule="depression",
                        )
                        self._weights[(src, tgt)] = new_w
                        changes.append(record)

        # Update last spike turns
        for neuron in spiked_neurons:
            self._last_spike_turn[neuron] = turn

        # Append to in-memory log
        self._log.extend(changes)

        # Append to file log if path configured
        if self._log_path and changes:
            with self._log_path.open("a", encoding="utf-8") as f:
                for rec in changes:
                    f.write(json.dumps(rec.to_dict()) + "\n")

        return changes

    # -------------------------------------------------------------------
    # V6b-E1: Snapshot serialization
    # -------------------------------------------------------------------

    def serialize(self) -> Dict[str, Any]:
        """Return all mutable state as a JSON-serializable dict."""
        return {
            "weights": {
                f"{s}->{t}": w for (s, t), w in sorted(self._weights.items())
            },
            "initial_weights": {
                f"{s}->{t}": w for (s, t), w in sorted(self._initial_weights.items())
            },
            "last_spike_turn": {
                k: v for k, v in sorted(self._last_spike_turn.items())
            },
        }

    def deserialize(self, data: Dict[str, Any]) -> None:
        """Restore mutable state from a previously serialized dict."""
        self._weights = {
            tuple(k.split("->")): float(v)
            for k, v in data["weights"].items()
        }
        self._initial_weights = {
            tuple(k.split("->")): float(v)
            for k, v in data["initial_weights"].items()
        }
        self._last_spike_turn = {
            k: v for k, v in data["last_spike_turn"].items()
        }
        # Reset in-memory log — not serialized (reconstructible from JSONL)
        self._log = []

    def verify_reconstruction(self) -> bool:
        """Verify that initial_weights + sum(log deltas) = current weights."""
        reconstructed = dict(self._initial_weights)
        for rec in self._log:
            key = (rec.source, rec.target)
            if key in reconstructed:
                reconstructed[key] = clamp_round(
                    reconstructed[key] + rec.delta,
                    WEIGHT_MIN, WEIGHT_MAX, 6,
                )
        for key in self._weights:
            if abs(self._weights[key] - reconstructed.get(key, 0.0)) > 1e-5:
                return False
        return True

    def get_stats(self) -> Dict[str, Any]:
        """Return STDP statistics for artifacts."""
        potentiations = sum(1 for r in self._log if r.rule == "potentiation")
        depressions = sum(1 for r in self._log if r.rule == "depression")
        return {
            "total_changes": len(self._log),
            "potentiations": potentiations,
            "depressions": depressions,
            "weight_bounds": [WEIGHT_MIN, WEIGHT_MAX],
            "reconstruction_valid": self.verify_reconstruction(),
            "current_weights": {
                f"{s}->{t}": w for (s, t), w in sorted(self._weights.items())
            },
        }
