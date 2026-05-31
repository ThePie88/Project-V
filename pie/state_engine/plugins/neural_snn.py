"""Neural SNN StateEngine plugin (V4-Neurone).

Implements a Spiking Neural Network backend using Izhikevich (2003)
neurons.  Each drive/affect dimension is modelled as a neuron with
biologically-motivated firing patterns (regular spiking, intrinsically
bursting, fast spiking, low-threshold, chattering).

The network has 10 neurons (6 drives + 4 affect) with sparse
connectivity.  When a neuron's membrane potential crosses threshold,
it fires (spike) and resets.  Spikes propagate along weighted
connections to influence connected neurons.

Key invariant: fully deterministic — same input state always produces
the same output state.  No randomness, no floating-point order
dependence (operations are serialised in a fixed neuron order).

Upgrade path from V3.6 LIF:
- LIFNeuron preserved as LIFNeuronLegacy for backward compatibility
- IzhikevichNeuron is the default neuron model
- Same 3-phase update loop, same SYNAPSE_WEIGHTS, same BASE_CURRENTS
- Izhikevich step() handles internal scale [-80,+30] mV → [0,1] output
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pie.contracts.state import State
from pie.determinism import clamp_round


# ---------------------------------------------------------------------------
# LIF Neuron model (V3.6 legacy — preserved for backward compatibility)
# ---------------------------------------------------------------------------

@dataclass
class LIFNeuronLegacy:
    """Leaky Integrate-and-Fire neuron (V3.6 legacy).

    Kept for backward compatibility and test regression.
    """

    neuron_id: str
    membrane_potential: float = 0.0
    tau: float = 0.9
    threshold: float = 0.85
    reset_potential: float = 0.1
    resting_potential: float = 0.0
    last_spike: bool = False

    def step(self, input_current: float) -> Tuple[float, bool]:
        """Advance one timestep. Returns (new_potential, did_spike)."""
        leaked = self.tau * self.membrane_potential + (1 - self.tau) * self.resting_potential
        new_v = leaked + input_current
        spiked = new_v >= self.threshold
        if spiked:
            new_v = self.reset_potential
        new_v = clamp_round(new_v, 0.0, 1.0, 4)
        self.membrane_potential = new_v
        self.last_spike = spiked
        return new_v, spiked


# Backward-compatible alias
LIFNeuron = LIFNeuronLegacy


# ---------------------------------------------------------------------------
# Izhikevich Neuron model (V4)
# ---------------------------------------------------------------------------

@dataclass
class IzhikevichNeuron:
    """Izhikevich (2003) neuron with deterministic dynamics.

    Two coupled equations:
        v' = 0.04*v^2 + 5*v + 140 - u + I
        u' = a*(b*v - u)

    On spike (v >= 30): v = c, u = u + d

    Parameters (a, b, c, d) select firing pattern:
        Regular spiking (RS):     a=0.02, b=0.2,  c=-65, d=8
        Intrinsically bursting:   a=0.02, b=0.2,  c=-55, d=4
        Chattering:               a=0.02, b=0.2,  c=-50, d=2
        Fast spiking (FS):        a=0.1,  b=0.2,  c=-65, d=2
        Low-threshold (LTS):      a=0.02, b=0.25, c=-65, d=2

    Internal dynamics use Izhikevich scale (-80 to +30 mV).
    Output is normalized to [0, 1] for State contract compliance.
    """

    neuron_id: str
    # Internal state (Izhikevich scale, mV)
    v: float = -65.0       # membrane potential
    u: float = -13.0       # recovery variable
    # Parameters
    a: float = 0.02        # recovery time constant
    b: float = 0.2         # sensitivity of u to v
    c: float = -65.0       # post-spike reset of v
    d: float = 8.0         # post-spike jump in u
    # Tracking
    last_spike: bool = False
    # Cache normalized output
    _normalized: float = 0.0

    # Normalization constants (fixed, class-level)
    V_MIN: float = -80.0
    V_MAX: float = 30.0

    @property
    def membrane_potential(self) -> float:
        """Normalized membrane potential [0, 1] for State compatibility."""
        return self._normalized

    def step(self, input_current: float) -> Tuple[float, bool]:
        """Advance one timestep with sub-step Euler integration.

        Args:
            input_current: Kernel-scale current [0, ~0.1] from synapses.

        Returns:
            (normalized_potential, did_spike) where potential is in [0, 1].
        """
        # Scale input from kernel range to Izhikevich mV range
        I_scaled = input_current * 200.0

        # Two sub-steps of dt=0.25 each for numerical stability
        for _ in range(2):
            dv = (0.04 * self.v * self.v + 5.0 * self.v + 140.0 - self.u + I_scaled) * 0.25
            du = self.a * (self.b * self.v - self.u) * 0.25
            self.v += dv
            self.u += du

        # Spike detection and reset
        spiked = self.v >= 30.0
        if spiked:
            self.v = self.c
            self.u = self.u + self.d

        # Safety clamp on internal state (prevent numerical blowup)
        self.v = max(self.V_MIN, min(self.v, self.V_MAX))
        self.u = max(-200.0, min(self.u, 200.0))

        self.last_spike = spiked

        # Normalize to [0, 1] for State contract
        self._normalized = clamp_round(
            (self.v - self.V_MIN) / (self.V_MAX - self.V_MIN),
            0.0, 1.0, 4,
        )
        return self._normalized, spiked


# ---------------------------------------------------------------------------
# Neuron type configurations (firing patterns)
# ---------------------------------------------------------------------------

NEURON_TYPES: Dict[str, Dict[str, float]] = {
    # Drives
    "curiosity":    {"a": 0.02, "b": 0.2,  "c": -55.0, "d": 4.0},   # intrinsically bursting
    "sociality":    {"a": 0.02, "b": 0.2,  "c": -65.0, "d": 8.0},   # regular spiking
    "caution":      {"a": 0.02, "b": 0.25, "c": -65.0, "d": 2.0},   # low-threshold spiking
    "agency":       {"a": 0.02, "b": 0.2,  "c": -50.0, "d": 2.0},   # chattering
    "playfulness":  {"a": 0.02, "b": 0.2,  "c": -55.0, "d": 4.0},   # intrinsically bursting
    "fatigue":      {"a": 0.02, "b": 0.2,  "c": -65.0, "d": 8.0},   # regular spiking
    # Affect
    "valence":      {"a": 0.02, "b": 0.2,  "c": -65.0, "d": 8.0},   # regular spiking
    "arousal":      {"a": 0.1,  "b": 0.2,  "c": -65.0, "d": 2.0},   # fast spiking
    "attention":    {"a": 0.1,  "b": 0.2,  "c": -65.0, "d": 2.0},   # fast spiking
    "tension":      {"a": 0.02, "b": 0.25, "c": -65.0, "d": 2.0},   # low-threshold spiking
}


# ---------------------------------------------------------------------------
# Synapse weights (sparse connectivity) — unchanged from V3.6
# ---------------------------------------------------------------------------

SYNAPSE_WEIGHTS: Dict[Tuple[str, str], float] = {
    # Drive → Drive connections
    ("curiosity", "fatigue"): 0.02,
    ("fatigue", "caution"): 0.03,
    ("sociality", "playfulness"): 0.02,
    ("agency", "curiosity"): 0.01,
    # Drive → Affect connections
    ("curiosity", "arousal"): 0.04,
    ("sociality", "valence"): 0.04,
    ("fatigue", "attention"): -0.03,    # inhibitory
    ("caution", "tension"): 0.03,
    # Affect → Drive feedback
    ("tension", "caution"): 0.02,
    ("valence", "sociality"): 0.01,
    ("arousal", "curiosity"): 0.02,
}

# Base input currents per neuron — unchanged from V3.6
BASE_CURRENTS: Dict[str, float] = {
    "curiosity": 0.05,
    "sociality": 0.03,
    "caution": 0.0,
    "agency": 0.01,
    "playfulness": 0.01,
    "fatigue": 0.02,
    "valence": 0.0,
    "arousal": 0.0,
    "attention": 0.01,
    "tension": 0.0,
}

# Neuron ordering (deterministic iteration order)
NEURON_ORDER = [
    "curiosity", "sociality", "caution", "agency", "playfulness", "fatigue",
    "valence", "arousal", "attention", "tension",
]

DRIVE_NAMES = {"curiosity", "sociality", "caution", "agency", "playfulness", "fatigue"}
AFFECT_NAMES = {"valence", "arousal", "attention", "tension"}


# ---------------------------------------------------------------------------
# Spike record (for artifacts)
# ---------------------------------------------------------------------------

@dataclass
class SpikeEvent:
    """Record of a neuron spike."""

    neuron_id: str
    turn: int
    membrane_potential_before: float
    membrane_potential_after: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "neuron_id": self.neuron_id,
            "turn": self.turn,
            "v_before": self.membrane_potential_before,
            "v_after": self.membrane_potential_after,
        }


# ---------------------------------------------------------------------------
# Neural SNN Plugin (V4 — Izhikevich)
# ---------------------------------------------------------------------------

class NeuralSNNPlugin:
    """Two-tier Spiking Neural Network backend (V4-Neurone).

    Tier 1: 10 Izhikevich neurons (drives + affect) with sparse synapses.
    Tier 2: 128-neuron ESN reservoir producing ControlVector.

    Implements ``StateEnginePlugin`` protocol.  All operations are
    deterministic and serialised in ``NEURON_ORDER``.
    """

    def __init__(self, reservoir_enabled: bool = True, stdp_enabled: bool = True,
                 stdp_log_path: Optional[Path] = None) -> None:
        self._neurons: Dict[str, IzhikevichNeuron] = {}
        self._spike_log: List[SpikeEvent] = []
        self._turn_count = 0
        self._initialised = False
        self._reservoir_enabled = reservoir_enabled
        self._reservoir: Optional[Any] = None
        self._control_vector: Optional[Any] = None
        self._stdp_enabled = stdp_enabled
        self._stdp_log_path = stdp_log_path
        self._stdp_tracker: Optional[Any] = None

    @property
    def engine_id(self) -> str:
        return "neural_snn"

    @property
    def version(self) -> str:
        return "0.3"

    @property
    def spike_log(self) -> List[SpikeEvent]:
        return list(self._spike_log)

    @property
    def rstdp_tracker(self) -> Optional[Any]:
        """Expose STDP/R-STDP tracker for external reward application."""
        return self._stdp_tracker

    def enable_reward_tracking(self) -> None:
        """Upgrade STDPTracker to RewardSTDPTracker (one-way, idempotent).

        Called by TurnProcessor when PIE_RSTDP_ONLINE=1. After upgrade,
        Hebbian weight changes are reverted and accumulated as eligibility
        traces; actual weight modulation only happens via apply_reward().
        """
        if self._stdp_tracker is None:
            return
        from pie.state_engine.plugins.rstdp import RewardSTDPTracker
        if isinstance(self._stdp_tracker, RewardSTDPTracker):
            return
        rstdp = RewardSTDPTracker(
            initial_weights=dict(self._stdp_tracker.weights),
            log_path=self._stdp_log_path,
        )
        rstdp._stdp._last_spike_turn = dict(self._stdp_tracker._last_spike_turn)
        self._stdp_tracker = rstdp

    @property
    def control_vector(self) -> Optional[Any]:
        """Current ControlVector from reservoir (None if reservoir disabled)."""
        return self._control_vector

    def _init_neurons(self, state: State) -> None:
        """Initialise Izhikevich neurons from current state values."""
        for name in NEURON_ORDER:
            if name in DRIVE_NAMES:
                v0_norm = state.drives.get(name, 0.5)
            else:
                v0_norm = state.affect.get(name, 0.5)
            # Convert normalized [0,1] → Izhikevich scale [-80, +30]
            v0_izh = v0_norm * (IzhikevichNeuron.V_MAX - IzhikevichNeuron.V_MIN) + IzhikevichNeuron.V_MIN
            params = NEURON_TYPES[name]
            self._neurons[name] = IzhikevichNeuron(
                neuron_id=name,
                v=v0_izh,
                u=params["b"] * v0_izh,  # u starts at equilibrium
                a=params["a"],
                b=params["b"],
                c=params["c"],
                d=params["d"],
            )
            # Set initial normalized value
            self._neurons[name]._normalized = clamp_round(v0_norm, 0.0, 1.0, 4)
        self._initialised = True

        # Init reservoir if enabled
        if self._reservoir_enabled:
            from pie.state_engine.plugins.reservoir import Reservoir
            from pie.state_engine.control_vector import ControlVector
            self._reservoir = Reservoir()

        # Init STDP tracker if enabled
        if self._stdp_enabled:
            from pie.state_engine.plugins.stdp import STDPTracker
            self._stdp_tracker = STDPTracker(
                initial_weights=dict(SYNAPSE_WEIGHTS),
                log_path=self._stdp_log_path,
            )

    def update(self, state: State) -> State:
        """Apply two-tier neural dynamics to state and return a new State.

        Fully deterministic: same input → same output.
        Five-phase update:
        1. Compute input currents (base + synaptic + spike bonus)
        2. Step Tier-1 neurons in deterministic order
        3. Feed Tier-1 output into Tier-2 reservoir
        4. Apply reservoir delta modulation to Tier-1
        5. Write final potentials back to State + extract ControlVector
        """
        if not self._initialised:
            self._init_neurons(state)

        new_state = state.model_copy(deep=True)
        self._turn_count += 1

        # Get current synapse weights (STDP-modulated or original)
        active_weights = (
            self._stdp_tracker.weights if self._stdp_tracker is not None
            else SYNAPSE_WEIGHTS
        )

        # Phase 1: Compute input currents for each neuron
        currents: Dict[str, float] = {}
        for name in NEURON_ORDER:
            current = BASE_CURRENTS.get(name, 0.0)
            for (src, tgt), weight in sorted(active_weights.items()):
                if tgt == name and src in self._neurons:
                    src_neuron = self._neurons[src]
                    current += weight * src_neuron.membrane_potential
                    if src_neuron.last_spike:
                        current += weight * 0.5
            currents[name] = current

        # Phase 2: Update Tier-1 neurons in deterministic order
        spiked_this_turn: List[str] = []
        for name in NEURON_ORDER:
            neuron = self._neurons[name]
            v_before = neuron.membrane_potential
            v_after, spiked = neuron.step(currents[name])

            if spiked:
                spiked_this_turn.append(name)
                self._spike_log.append(SpikeEvent(
                    neuron_id=name,
                    turn=self._turn_count,
                    membrane_potential_before=v_before,
                    membrane_potential_after=v_after,
                ))

        # Phase 2b: STDP weight update based on spike timing
        if self._stdp_tracker is not None and spiked_this_turn:
            self._stdp_tracker.record_spikes(self._turn_count, spiked_this_turn)

        # Phase 3: Reservoir (Tier-2) — if enabled
        if self._reservoir is not None:
            from pie.state_engine.control_vector import ControlVector

            # Collect Tier-1 output as input vector for reservoir
            tier1_output = [
                self._neurons[name].membrane_potential
                for name in NEURON_ORDER
            ]
            reservoir_out = self._reservoir.step(tier1_output)

            # Phase 4: Apply reservoir deltas to Tier-1 neurons
            delta_scale = 0.1  # cap reservoir influence
            for name in NEURON_ORDER:
                delta_key = f"d_{name}"
                if delta_key in reservoir_out:
                    delta = (reservoir_out[delta_key] - 0.5) * 2.0 * delta_scale
                    neuron = self._neurons[name]
                    # Apply delta to normalized output
                    new_norm = clamp_round(
                        neuron.membrane_potential + delta,
                        0.0, 1.0, 4,
                    )
                    neuron._normalized = new_norm

            # Extract ControlVector from the 5 control channels
            self._control_vector = ControlVector.from_raw({
                "memory_gate": reservoir_out.get("memory_gate", 0.5),
                "tool_gate": reservoir_out.get("tool_gate", 0.5),
                "cf_k": reservoir_out.get("cf_k", 0.5) * 5.0,  # scale [0,1]→[0,5]
                "verbosity_bias": reservoir_out.get("verbosity_bias", 0.5),
                "consolidation_urgency": reservoir_out.get("consolidation_urgency", 0.0),
            })

        # Phase 5: Write neuron potentials back to state
        for name in NEURON_ORDER:
            v = self._neurons[name].membrane_potential
            if name in DRIVE_NAMES:
                new_state.drives[name] = v
            elif name in AFFECT_NAMES:
                new_state.affect[name] = v

        new_state.turn_count += 1
        return new_state

    # -------------------------------------------------------------------
    # V6b-E1: Snapshot serialization
    # -------------------------------------------------------------------

    def serialize(self) -> Dict[str, Any]:
        """Return all mutable internal state as a JSON-serializable dict."""
        neurons_data = {}
        for name in NEURON_ORDER:
            n = self._neurons[name]
            neurons_data[name] = {
                "v": round(n.v, 10),
                "u": round(n.u, 10),
                "last_spike": n.last_spike,
                "_normalized": round(n._normalized, 4),
            }

        reservoir_data = None
        if self._reservoir is not None:
            reservoir_data = self._reservoir.serialize()

        stdp_data = None
        if self._stdp_tracker is not None:
            stdp_data = self._stdp_tracker.serialize()

        cv_data = None
        if self._control_vector is not None:
            cv_data = self._control_vector.to_dict()

        return {
            "engine_id": self.engine_id,
            "version": self.version,
            "turn_count": self._turn_count,
            "initialised": self._initialised,
            "neurons": neurons_data,
            "reservoir": reservoir_data,
            "stdp": stdp_data,
            "control_vector": cv_data,
        }

    def deserialize(self, data: Dict[str, Any]) -> None:
        """Restore mutable internal state from a previously serialized dict."""
        self._turn_count = data["turn_count"]
        self._initialised = data.get("initialised", True)

        # Restore neurons
        for name in NEURON_ORDER:
            nd = data["neurons"][name]
            params = NEURON_TYPES[name]
            self._neurons[name] = IzhikevichNeuron(
                neuron_id=name,
                v=float(nd["v"]),
                u=float(nd["u"]),
                last_spike=bool(nd["last_spike"]),
                a=params["a"],
                b=params["b"],
                c=params["c"],
                d=params["d"],
                _normalized=float(nd["_normalized"]),
            )

        # Restore reservoir
        if data.get("reservoir") is not None and self._reservoir is not None:
            self._reservoir.deserialize(data["reservoir"])
        elif data.get("reservoir") is not None and self._reservoir is None:
            # Reservoir was enabled at save time but disabled now — init it
            from pie.state_engine.plugins.reservoir import Reservoir
            self._reservoir = Reservoir()
            self._reservoir.deserialize(data["reservoir"])

        # Restore STDP
        if data.get("stdp") is not None and self._stdp_tracker is not None:
            self._stdp_tracker.deserialize(data["stdp"])
        elif data.get("stdp") is not None and self._stdp_tracker is None:
            import os
            if os.environ.get("PIE_RSTDP_ONLINE", "0") == "1":
                from pie.state_engine.plugins.rstdp import RewardSTDPTracker
                self._stdp_tracker = RewardSTDPTracker(
                    initial_weights=dict(SYNAPSE_WEIGHTS),
                    log_path=self._stdp_log_path,
                )
            else:
                from pie.state_engine.plugins.stdp import STDPTracker
                self._stdp_tracker = STDPTracker(
                    initial_weights=dict(SYNAPSE_WEIGHTS),
                    log_path=self._stdp_log_path,
                )
            self._stdp_tracker.deserialize(data["stdp"])

        # Restore ControlVector
        if data.get("control_vector") is not None:
            from pie.state_engine.control_vector import ControlVector
            self._control_vector = ControlVector.from_raw(data["control_vector"])

        # Reset ephemeral spike log — starts fresh for new session segment
        self._spike_log = []

    def get_artifacts(self) -> Dict[str, Any]:
        """Return neuron state and spike log for artifact generation."""
        artifacts: Dict[str, Any] = {
            "schema_version": "0.3",
            "engine_id": self.engine_id,
            "engine_version": self.version,
            "neuron_model": "izhikevich",
            "neuron_states": {
                name: {
                    "membrane_potential": n.membrane_potential,
                    "v_internal": round(n.v, 4),
                    "u_recovery": round(n.u, 4),
                    "last_spike": n.last_spike,
                    "firing_pattern": _get_pattern_name(name),
                }
                for name, n in sorted(self._neurons.items())
            },
            "spike_log": [s.to_dict() for s in self._spike_log],
            "total_spikes": len(self._spike_log),
            "turn_count": self._turn_count,
        }
        if self._control_vector is not None:
            artifacts["control_vector"] = self._control_vector.to_dict()
        if self._reservoir is not None:
            artifacts["reservoir"] = self._reservoir.get_stats()
        if self._stdp_tracker is not None:
            artifacts["stdp"] = self._stdp_tracker.get_stats()
        return artifacts

    def save_artifacts(self, path: Path) -> None:
        """Save neuron artifacts to a JSON file."""
        path.write_text(
            json.dumps(self.get_artifacts(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


def _get_pattern_name(neuron_id: str) -> str:
    """Return human-readable firing pattern for a neuron."""
    params = NEURON_TYPES.get(neuron_id, {})
    a, c = params.get("a", 0.02), params.get("c", -65.0)
    b = params.get("b", 0.2)
    if a >= 0.1:
        return "fast_spiking"
    if b >= 0.25:
        return "low_threshold"
    if c >= -50.0:
        return "chattering"
    if c >= -55.0:
        return "intrinsically_bursting"
    return "regular_spiking"
