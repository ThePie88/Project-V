"""Default ODE plugin — replicates the original hardcoded dynamics.

This plugin extracts the logic that was previously in
``State.update()`` so that it can be swapped for alternative
backends while preserving backward compatibility.
"""

from __future__ import annotations

from pie.contracts.state import State


class DefaultODEPlugin:
    """Simple ODE dynamics identical to the original M0 implementation."""

    @property
    def engine_id(self) -> str:
        return "default_ode"

    @property
    def version(self) -> str:
        return "0.1"

    def update(self, state: State) -> State:
        """Apply rudimentary ODE dynamics and return a new State."""
        new_state = state.model_copy(deep=True)

        def _clamp_round(value: float) -> float:
            v = min(max(value, 0.0), 1.0)
            return round(v, state.DECIMAL_PLACES)

        # Drive dynamics
        new_state.drives["curiosity"] = _clamp_round(state.drives["curiosity"] + 0.05)
        new_state.drives["sociality"] = _clamp_round(state.drives["sociality"] + 0.03)
        new_state.drives["fatigue"] = _clamp_round(state.drives["fatigue"] + 0.02)
        new_state.drives["caution"] = _clamp_round(
            state.drives["caution"] + 0.01 * state.drives["fatigue"]
        )
        # Affect dynamics
        new_state.affect["arousal"] = _clamp_round(
            state.affect["arousal"] + 0.04 * (state.drives["curiosity"] - 0.5)
        )
        new_state.affect["valence"] = _clamp_round(
            state.affect["valence"] + 0.04 * (state.drives["sociality"] - 0.5)
        )
        new_state.turn_count += 1
        return new_state
