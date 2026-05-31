"""Echo State Network (ESN) Reservoir — 128 Izhikevich neurons.

Implements a reservoir computing layer that transforms 10 Tier-1 neuron
outputs into 15 control channels (10 drive/affect deltas + 5 ControlVector
signals).

Key properties:
- **Deterministic**: LCG-seeded weights, no Python random, no float order
  dependence.  Same input → same output always.
- **Echo State Property**: Spectral radius = 0.95 ensures fading memory
  without divergence.
- **Fixed weights**: Reservoir internal weights are NEVER modified (ESN
  theory).  Only the readout layer maps reservoir state → output.
- **Sparse**: ~10% connectivity → ~1640 synapses out of 128×128 possible.

Reference: Jaeger (2001) "The echo state approach to analysing and
training recurrent neural networks"
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pie.determinism import clamp_round


# ---------------------------------------------------------------------------
# Deterministic LCG (Linear Congruential Generator)
# ---------------------------------------------------------------------------

class LCG:
    """Deterministic pseudo-random number generator.

    Uses the Numerical Recipes LCG parameters.
    Cross-platform: identical output on any architecture.
    """

    def __init__(self, seed: int = 0xDEADBEEF) -> None:
        self._state = seed & 0xFFFFFFFF

    def next_int(self) -> int:
        """Return next 32-bit unsigned integer."""
        self._state = (self._state * 1664525 + 1013904223) & 0xFFFFFFFF
        return self._state

    def next_float(self) -> float:
        """Return next float in [0, 1)."""
        return self.next_int() / 4294967296.0

    def next_signed(self) -> float:
        """Return next float in [-1, 1)."""
        return self.next_float() * 2.0 - 1.0


# ---------------------------------------------------------------------------
# Reservoir Neuron (simplified Izhikevich — Regular Spiking uniform)
# ---------------------------------------------------------------------------

@dataclass
class ReservoirNeuron:
    """Simplified Izhikevich neuron for reservoir internals.

    All reservoir neurons use Regular Spiking (RS) parameters for
    uniformity — the diversity comes from connectivity, not neuron types.
    """

    v: float = -65.0        # membrane potential (mV)
    u: float = -13.0        # recovery variable
    last_spike: bool = False

    # RS parameters (fixed)
    A: float = 0.02
    B: float = 0.2
    C: float = -65.0
    D: float = 8.0

    def step(self, current: float) -> float:
        """Advance one timestep. Returns normalized output [0, 1]."""
        I_scaled = current * 200.0

        for _ in range(2):
            dv = (0.04 * self.v * self.v + 5.0 * self.v + 140.0 - self.u + I_scaled) * 0.25
            du = self.A * (self.B * self.v - self.u) * 0.25
            self.v += dv
            self.u += du

        spiked = self.v >= 30.0
        if spiked:
            self.v = self.C
            self.u = self.u + self.D

        self.v = max(-80.0, min(self.v, 30.0))
        self.u = max(-200.0, min(self.u, 200.0))
        self.last_spike = spiked

        return clamp_round(
            (self.v - (-80.0)) / (30.0 - (-80.0)),
            0.0, 1.0, 4,
        )


# ---------------------------------------------------------------------------
# Readout channel names
# ---------------------------------------------------------------------------

READOUT_CHANNELS: List[str] = [
    # 10 drive/affect delta channels
    "d_curiosity", "d_sociality", "d_caution", "d_agency",
    "d_playfulness", "d_fatigue",
    "d_valence", "d_arousal", "d_attention", "d_tension",
    # 5 control channels
    "memory_gate", "tool_gate", "cf_k", "verbosity_bias",
    "consolidation_urgency",
]

INPUT_SIZE = 10     # from Tier-1
RESERVOIR_SIZE = 128
OUTPUT_SIZE = 15    # len(READOUT_CHANNELS)

# Tier-1 neuron index mapping (must match NEURON_ORDER in neural_snn.py)
_T1_INDEX = {
    "curiosity": 0, "sociality": 1, "caution": 2, "agency": 3,
    "playfulness": 4, "fatigue": 5, "valence": 6, "arousal": 7,
    "attention": 8, "tension": 9,
}

# Hand-designed control channel readout: each control channel is a
# weighted sum of reservoir neurons, where weights depend on which
# Tier-1 neuron projects into that reservoir neuron.
# Format: {channel_name: {tier1_neuron: weight_contribution}}
_CONTROL_READOUT: Dict[str, Dict[str, float]] = {
    # memory_gate: high curiosity+attention → recall more; high fatigue+tension → recall less
    "memory_gate": {
        "curiosity": +0.4, "attention": +0.4,
        "fatigue": -0.3, "tension": -0.3,
        "arousal": +0.1,
    },
    # tool_gate: high agency → tools available; high caution+tension → tools restricted
    "tool_gate": {
        "agency": +0.5, "curiosity": +0.2,
        "caution": -0.4, "tension": -0.3,
        "fatigue": -0.2,
    },
    # cf_k: high tension/arousal → consider MORE alternatives (careful deliberation)
    "cf_k": {
        "tension": +0.4, "arousal": +0.3, "caution": +0.3,
        "agency": -0.2, "playfulness": -0.1,
    },
    # verbosity_bias: high sociality+playfulness → talk more; high tension → terse
    "verbosity_bias": {
        "sociality": +0.4, "playfulness": +0.3, "curiosity": +0.2,
        "tension": -0.3, "fatigue": -0.3,
    },
    # consolidation_urgency: high arousal+attention → consolidate now; low fatigue
    "consolidation_urgency": {
        "arousal": +0.4, "attention": +0.3, "tension": +0.2,
        "fatigue": -0.3, "valence": -0.1,
    },
}


# ---------------------------------------------------------------------------
# Reservoir
# ---------------------------------------------------------------------------

class Reservoir:
    """128-neuron Echo State Network with Izhikevich internals.

    Architecture:
    - Input: 10 values from Tier-1 injected into first 10 neurons
    - Internal: 128 neurons with sparse (10%) fixed connectivity
    - Readout: Linear projection from 128 neuron states → 15 outputs

    All weights are deterministic (LCG-seeded) and FIXED after init.
    """

    def __init__(
        self,
        size: int = RESERVOIR_SIZE,
        sparsity: float = 0.1,
        spectral_radius: float = 0.95,
        seed: int = 0xDEADBEEF,
        leak_rate: float = 0.3,
        trained_readout_path: Optional[Path] = None,
    ) -> None:
        self._size = size
        self._sparsity = sparsity
        self._target_sr = spectral_radius
        self._seed = seed
        self._leak_rate = leak_rate

        # Init neurons
        self._neurons: List[ReservoirNeuron] = [
            ReservoirNeuron() for _ in range(size)
        ]

        # Leaky integration state — retains memory across steps,
        # preventing different initial conditions from collapsing
        self._leak_state: List[float] = [0.0] * size

        # Generate internal weights (sparse, LCG-seeded)
        rng = LCG(seed)
        self._weights: List[List[float]] = [
            [0.0] * size for _ in range(size)
        ]
        for i in range(size):
            for j in range(size):
                if rng.next_float() < sparsity:
                    self._weights[i][j] = rng.next_signed() * 0.5
                else:
                    # Consume the RNG state for determinism even when sparse
                    rng.next_signed()

        # Scale weights to target spectral radius
        current_sr = self._power_iteration_spectral_radius(500)
        if current_sr > 1e-8:
            scale = spectral_radius / current_sr
            for i in range(size):
                for j in range(size):
                    self._weights[i][j] *= scale

        # Generate input weights: BROAD sparse projection across ALL neurons
        # (not just first 10).  20% connectivity, scale 0.3.
        # Identity mapping on first 10 preserved for Tier-1 signal fidelity.
        rng2 = LCG(seed ^ 0x12345678)
        self._input_weights: List[List[float]] = [
            [0.0] * INPUT_SIZE for _ in range(size)
        ]
        for i in range(min(INPUT_SIZE, size)):
            self._input_weights[i][i] = 1.0  # identity mapping
        for i in range(size):
            for j in range(INPUT_SIZE):
                if rng2.next_float() < 0.20:  # 20% connectivity (was 5%)
                    self._input_weights[i][j] += rng2.next_signed() * 0.3  # scale 0.3 (was 0.1)
                else:
                    rng2.next_signed()  # consume for determinism

        # Generate readout weights for delta channels (128 → 10, random)
        self._readout_weights: List[List[float]] = [
            [0.0] * size for _ in range(OUTPUT_SIZE)
        ]
        for ch_idx, ch_name in enumerate(READOUT_CHANNELS):
            if ch_name in _CONTROL_READOUT:
                # Hand-designed readout for control channels — see below
                continue
            # Random readout for delta channels
            ch_seed = seed
            for c in ch_name:
                ch_seed = (ch_seed * 31 + ord(c)) & 0xFFFFFFFF
            rng3 = LCG(ch_seed)
            for j in range(size):
                self._readout_weights[ch_idx][j] = rng3.next_signed() * 0.1

        # Hand-designed readout for 5 control channels:
        # Each control channel is a SEMANTIC function of which Tier-1
        # neurons project into which reservoir neurons.
        # This ensures ControlVector actually responds to emotional state.
        self._init_control_readout()

        self._step_count = 0

        # Trained readout (V6a-R2): if provided, overrides hand-designed
        # control channel weights with ridge-regression-trained weights.
        self._trained_readout: Optional[Dict[str, Any]] = None
        if trained_readout_path is not None:
            self.load_readout(trained_readout_path)

    def _init_control_readout(self) -> None:
        """Build readout weights for 5 control channels from semantic map.

        For each control channel, reservoir neuron j gets a readout weight
        proportional to how strongly Tier-1 neuron k projects into it
        (via input_weights[j][k]), weighted by the semantic role of k
        for that channel.  This means: if curiosity projects strongly
        into reservoir neuron 42, and memory_gate has curiosity=+0.4,
        then neuron 42 gets readout weight +0.4 * input_weight[42][curiosity].

        Result: ControlVector channels respond to which Tier-1 neurons
        are active, mediated by the reservoir's temporal dynamics.
        """
        for ch_name, semantic_map in _CONTROL_READOUT.items():
            ch_idx = READOUT_CHANNELS.index(ch_name)
            for j in range(self._size):
                w = 0.0
                for t1_name, coeff in semantic_map.items():
                    t1_idx = _T1_INDEX[t1_name]
                    w += coeff * self._input_weights[j][t1_idx]
                self._readout_weights[ch_idx][j] = w

    def _power_iteration_spectral_radius(self, iterations: int = 50) -> float:
        """Estimate spectral radius via power iteration."""
        n = self._size
        # Init vector
        vec = [1.0 / math.sqrt(n)] * n
        eigenvalue = 0.0

        for _ in range(iterations):
            # Matrix-vector multiply
            new_vec = [0.0] * n
            for i in range(n):
                s = 0.0
                for j in range(n):
                    s += self._weights[i][j] * vec[j]
                new_vec[i] = s

            # Compute norm
            norm = math.sqrt(sum(x * x for x in new_vec))
            if norm < 1e-12:
                return 0.0

            eigenvalue = norm
            vec = [x / norm for x in new_vec]

        return eigenvalue

    def verify_spectral_radius(self) -> float:
        """Public method to verify current spectral radius."""
        return self._power_iteration_spectral_radius(500)

    def step(self, input_vector: List[float]) -> Dict[str, float]:
        """Advance reservoir one timestep with 10 input values.

        Args:
            input_vector: 10 normalized values [0,1] from Tier-1 neurons.

        Returns:
            Dict with 15 named output channels.
        """
        assert len(input_vector) == INPUT_SIZE

        # Phase 1: Compute input currents for each reservoir neuron
        currents = [0.0] * self._size
        # External input (Win · x)
        for i in range(self._size):
            for j in range(INPUT_SIZE):
                currents[i] += self._input_weights[i][j] * input_vector[j]
        # Recurrent connections (W · prev_state)
        # Use leak state for recurrence (retains memory across steps)
        for i in range(self._size):
            for j in range(self._size):
                currents[i] += self._weights[i][j] * self._leak_state[j]

        # Phase 2: Step all neurons in order (deterministic)
        # Current scale 0.04 (was 0.01) — enough to drive spiking dynamics
        raw_outputs = [0.0] * self._size
        for i in range(self._size):
            raw_outputs[i] = self._neurons[i].step(currents[i] * 0.04)

        # Leaky integration: state[i] = (1-α)·prev + α·new
        # This prevents washout and maintains input-dependent separation
        alpha = self._leak_rate
        for i in range(self._size):
            self._leak_state[i] = clamp_round(
                (1.0 - alpha) * self._leak_state[i] + alpha * raw_outputs[i],
                0.0, 1.0, 4,
            )

        # Phase 3: Readout — reservoir features + direct Tier-1 contribution
        # If trained readout is available, use it for control channels.
        # Otherwise fall back to hand-designed _CONTROL_READOUT.

        # Trained readout: pre-compute control channel values if available
        trained_cv: Optional[Dict[str, float]] = None
        if self._trained_readout is not None:
            trained_cv = self._apply_trained_readout(self._leak_state)

        result: Dict[str, float] = {}
        for ch_idx, ch_name in enumerate(READOUT_CHANNELS):
            if trained_cv is not None and ch_name in trained_cv:
                # V6a-R2: Use trained readout for control channels
                val = trained_cv[ch_name]
            else:
                # Hand-designed readout (fallback or delta channels)
                val = 0.0
                for j in range(self._size):
                    val += self._readout_weights[ch_idx][j] * self._leak_state[j]

                # Direct Tier-1 contribution for control channels
                if ch_name in _CONTROL_READOUT:
                    direct = 0.0
                    for t1_name, coeff in _CONTROL_READOUT[ch_name].items():
                        t1_idx = _T1_INDEX[t1_name]
                        direct += coeff * input_vector[t1_idx]
                    # Mix: 40% reservoir + 60% direct Tier-1
                    val = 0.4 * val + 0.6 * direct

            # Sigmoid squash to [0, 1]
            val = 1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, val * 10.0))))
            result[ch_name] = clamp_round(val, 0.0, 1.0, 4)

        self._step_count += 1
        return result

    def get_neuron_states(self) -> List[float]:
        """Return leak-integrated reservoir states (the actual features)."""
        return list(self._leak_state)

    # -------------------------------------------------------------------
    # V6b-E1: Snapshot serialization
    # -------------------------------------------------------------------

    def serialize(self) -> Dict[str, Any]:
        """Return all mutable dynamic state as a JSON-serializable dict.

        Deterministic weights (internal, input, readout) are NOT serialized —
        they are regenerated from seed on __init__.  Only ephemeral dynamic
        state (leak_state, neuron v/u/last_spike, step_count) is persisted.
        """
        return {
            "size": self._size,
            "seed": self._seed,
            "leak_rate": self._leak_rate,
            "step_count": self._step_count,
            "leak_state": [round(x, 10) for x in self._leak_state],
            "neurons": [
                {
                    "v": round(n.v, 10),
                    "u": round(n.u, 10),
                    "last_spike": n.last_spike,
                }
                for n in self._neurons
            ],
        }

    def deserialize(self, data: Dict[str, Any]) -> None:
        """Restore mutable dynamic state from a previously serialized dict.

        Validates that size and seed match (config mismatch = invalid restore).
        """
        if data["size"] != self._size:
            raise ValueError(
                f"Reservoir size mismatch: snapshot has {data['size']}, "
                f"current has {self._size}"
            )
        if data["seed"] != self._seed:
            raise ValueError(
                f"Reservoir seed mismatch: snapshot has {data['seed']}, "
                f"current has {self._seed}"
            )
        self._step_count = data["step_count"]
        self._leak_rate = data["leak_rate"]
        self._leak_state = [float(x) for x in data["leak_state"]]
        for i, nd in enumerate(data["neurons"]):
            self._neurons[i].v = float(nd["v"])
            self._neurons[i].u = float(nd["u"])
            self._neurons[i].last_spike = bool(nd["last_spike"])

    def get_stats(self) -> Dict[str, Any]:
        """Return reservoir statistics for artifacts."""
        states = self.get_neuron_states()
        spike_count = sum(1 for n in self._neurons if n.last_spike)
        return {
            "size": self._size,
            "sparsity": self._sparsity,
            "target_spectral_radius": self._target_sr,
            "measured_spectral_radius": round(self.verify_spectral_radius(), 4),
            "step_count": self._step_count,
            "active_spikes": spike_count,
            "mean_potential": round(sum(states) / len(states), 4) if states else 0.0,
            "readout_mode": "trained" if self._trained_readout is not None else "hand_designed",
        }

    # -------------------------------------------------------------------
    # V6a-R2: Trained readout support
    # -------------------------------------------------------------------

    def load_readout(self, path: Path) -> None:
        """Load trained readout weights from JSON.

        Once loaded, the 5 control channels use trained weights
        instead of hand-designed ``_CONTROL_READOUT``.
        The 10 delta channels remain random-initialized.
        """
        data = json.loads(path.read_text(encoding="utf-8"))
        self._trained_readout = data

        # Overwrite readout weights for control channels
        weights = data["weights"]        # (5 × feature_dim)
        channels = data["channels"]      # 5 channel names
        mean_X = data["mean_X"]          # feature_dim
        std_X = data["std_X"]            # feature_dim

        for ch_idx_trained, ch_name in enumerate(channels):
            if ch_name not in [c for c in READOUT_CHANNELS]:
                continue
            ch_idx = READOUT_CHANNELS.index(ch_name)
            w = weights[ch_idx_trained]
            # Store trained weights — applied in step() with normalization
            # We DON'T overwrite _readout_weights directly because
            # trained weights expect normalized input (subtract mean, divide std)
            # The step() method handles this when _trained_readout is set

    @property
    def has_trained_readout(self) -> bool:
        """Whether trained readout weights are loaded."""
        return self._trained_readout is not None

    def _apply_trained_readout(self, leak_state: List[float]) -> Dict[str, float]:
        """Apply trained readout weights to normalized reservoir state.

        Returns dict of control channel values (raw, pre-sigmoid).
        """
        assert self._trained_readout is not None
        data = self._trained_readout
        weights = data["weights"]
        channels = data["channels"]
        mean_X = data["mean_X"]
        std_X = data["std_X"]
        d = len(mean_X)

        # Normalize state
        normalized = [
            (leak_state[j] - mean_X[j]) / std_X[j]
            for j in range(min(d, len(leak_state)))
        ]

        # Compute output for each channel
        result: Dict[str, float] = {}
        for ch_idx, ch_name in enumerate(channels):
            val = sum(weights[ch_idx][j] * normalized[j] for j in range(len(normalized)))
            result[ch_name] = val

        return result
