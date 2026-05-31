"""StateEngine plugin protocol (V2.5).

Any object that satisfies ``StateEnginePlugin`` can serve as the
backend for state dynamics.  Plugins must be pure functions — same
state in, same state out — to preserve replay determinism.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pie.contracts.state import State


@runtime_checkable
class StateEnginePlugin(Protocol):
    """Protocol that every state-engine backend must implement."""

    @property
    def engine_id(self) -> str:
        """Unique identifier for this engine (e.g. ``default_ode``)."""
        ...

    @property
    def version(self) -> str:
        """Semantic version of this engine (e.g. ``0.1``)."""
        ...

    def update(self, state: State) -> State:
        """Apply dynamics to *state* and return a **new** State.

        Implementations must be deterministic: same input state must
        always produce the same output state.
        """
        ...
