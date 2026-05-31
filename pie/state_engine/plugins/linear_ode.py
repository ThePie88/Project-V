"""Linear ODE StateEngine plugin — fair baseline for ablation (V6a-R1).

Implements a 10-dimensional linear dynamical system:
    x(t+1) = (1 - decay) * x(t) + gain * u(t)

No nonlinearity, no spikes.  Same dimensionality as Tier-1 Izhikevich
to ensure fair comparison.  The readout is trained separately (same
ridge regression procedure as the Izhikevich conditions).

Grid search: 3 decay rates × 2 input gains = 6 configurations,
all evaluated, best reported.  This prevents strawman accusations.

Conforms to ``StateEnginePlugin`` protocol.
"""

from __future__ import annotations

from typing import Any, Dict, List

from pie.contracts.state import State
from pie.determinism import clamp_round


# Neuron order must match NEURON_ORDER in neural_snn.py
NEURON_ORDER = [
    "curiosity", "sociality", "caution", "agency", "playfulness", "fatigue",
    "valence", "arousal", "attention", "tension",
]
DRIVE_NAMES = {"curiosity", "sociality", "caution", "agency", "playfulness", "fatigue"}
AFFECT_NAMES = {"valence", "arousal", "attention", "tension"}

# Grid search configurations
DECAY_RATES = [0.05, 0.1, 0.2]
INPUT_GAINS = [0.3, 0.6]

DEFAULT_CONFIG_IDX = 0  # index into grid


class LinearODEPlugin:
    """10-dim linear ODE baseline for ablation study.

    x(t+1) = (1 - decay) * x(t) + gain * input(t)
    Output clamped to [0, 1].
    """

    def __init__(
        self,
        decay: float = 0.1,
        gain: float = 0.3,
    ) -> None:
        self._decay = decay
        self._gain = gain
        self._state: List[float] = [0.5] * 10  # internal state
        self._step_count = 0

    @property
    def engine_id(self) -> str:
        return "linear_ode"

    @property
    def version(self) -> str:
        return "0.1"

    def update(self, state: State) -> State:
        """Apply linear dynamics: x' = (1-decay)*x + gain*input."""
        new_state = state.model_copy(deep=True)

        # Read current drives/affect as input
        inputs = [
            state.drives.get(n, 0.5) if n in DRIVE_NAMES
            else state.affect.get(n, 0.5)
            for n in NEURON_ORDER
        ]

        # Linear update
        for i in range(10):
            self._state[i] = clamp_round(
                (1.0 - self._decay) * self._state[i] + self._gain * inputs[i],
                0.0, 1.0, 4,
            )

        # Write back
        for i, name in enumerate(NEURON_ORDER):
            if name in DRIVE_NAMES:
                new_state.drives[name] = self._state[i]
            else:
                new_state.affect[name] = self._state[i]

        new_state.turn_count += 1
        self._step_count += 1
        return new_state

    def get_state_vector(self) -> List[float]:
        """Return current 10-dim state (for readout training)."""
        return list(self._state)

    def step_with_input(self, input_vector: List[float]) -> List[float]:
        """Raw step: take 10-dim input, return 10-dim state.

        For benchmark use (bypass State contract).
        """
        assert len(input_vector) == 10
        for i in range(10):
            self._state[i] = clamp_round(
                (1.0 - self._decay) * self._state[i] + self._gain * input_vector[i],
                0.0, 1.0, 4,
            )
        self._step_count += 1
        return list(self._state)


def grid_configs() -> List[Dict[str, float]]:
    """Return all 6 grid search configurations."""
    configs = []
    for decay in DECAY_RATES:
        for gain in INPUT_GAINS:
            configs.append({"decay": decay, "gain": gain})
    return configs
