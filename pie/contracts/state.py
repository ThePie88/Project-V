"""State contract for the Pie Kernel.

This module defines the structure of the internal state of the kernel.  A
state snapshot captures the drives, affect, traits, values and any
additional dynamic information at a point in time.  For the M0
implementation we keep the state intentionally simple while laying
groundwork for extensions in future versions.

The ``schema_version`` field must be bumped any time backwards
incompatible changes are made to this model.  Older traces can then be
rejected or migrated appropriately when loaded.
"""

from __future__ import annotations

from typing import Any, Dict, Literal

from pydantic import BaseModel, Field, field_validator


class State(BaseModel):
    """A snapshot of the kernel's internal state.

    Attributes
    ----------
    schema_version:
        The version of the state schema.  This allows older snapshots
        to be detected when replaying traces.
    drives:
        A mapping from drive names to values in the range [0, 1].  In
        the M0 implementation only a handful of drives are tracked.
    affect:
        A mapping from affect names to values.  Values are bounded
        similarly to drives; see specification for ranges.
    turn_count:
        The number of turns that have elapsed.  Used to derive
        deterministic event IDs and for simple fatigue dynamics.
    """

    schema_version: Literal["0.1"] = Field(
        "0.1",
        description="Schema version for State; bump on backwards incompatible change",
    )
    drives: Dict[str, float] = Field(
        default_factory=lambda: {
            "curiosity": 0.5,
            "sociality": 0.5,
            "caution": 0.5,
            "agency": 0.5,
            "playfulness": 0.5,
            "fatigue": 0.0,
        },
        description="Drive values between 0 and 1",
    )
    affect: Dict[str, float] = Field(
        default_factory=lambda: {
            "arousal": 0.5,
            "valence": 0.5,
            "attention": 0.5,
            "tension": 0.5,
        },
        description="Affect values between 0 and 1",
    )
    turn_count: int = Field(0, description="Number of turns elapsed")

    # Identity information used by the governance layer.  The
    # ``creator_anchor`` holds the identifier of the creator of this
    # instance of the kernel.  It is surfaced in snapshots so that
    # downstream components can enforce authority constraints.  If
    # ``creator_anchor`` changes, a new seed and authority should be
    # provisioned.
    creator_anchor: str = Field(
        "MrPie",
        description="Creator anchor identifying the authority under which the kernel runs",
    )

    # Number of decimal places to round state values to.  A policy of
    # rounding after every update prevents floating point jitter from
    # propagating and affecting comparisons and thresholds.  See
    # runtime for details on when this is applied.
    DECIMAL_PLACES: int = 4

    @field_validator("drives", mode="before")
    @classmethod
    def clamp_drives(cls, v: Any) -> Dict[str, float]:
        if not isinstance(v, dict):
            return {}
        return {k: min(max(float(val), 0.0), 1.0) for k, val in v.items()}

    @field_validator("affect", mode="before")
    @classmethod
    def clamp_affect(cls, v: Any) -> Dict[str, float]:
        if not isinstance(v, dict):
            return {}
        return {k: min(max(float(val), 0.0), 1.0) for k, val in v.items()}

    @classmethod
    def from_seed(cls, seed_data: dict) -> "State":
        """Create a State initialised from a SEED identity file.

        The *seed_data* dictionary is expected to follow the format of
        ``progetto/SEED_V0.md`` (JSON with ``drives``, ``affect_baseline``
        and ``identity`` sections).  Any drives or affect values present in
        the seed override the defaults; missing keys keep their defaults.
        """
        _drives_factory = cls.model_fields["drives"].default_factory
        drives = dict(_drives_factory() if _drives_factory else {})
        drives.update(seed_data.get("drives", {}))
        _affect_factory = cls.model_fields["affect"].default_factory
        affect = dict(_affect_factory() if _affect_factory else {})
        affect.update(seed_data.get("affect_baseline", {}))
        creator = (
            seed_data.get("identity", {}).get("creator_anchor", "MrPie")
        )
        return cls(drives=drives, affect=affect, creator_anchor=creator)

    def update(self) -> "State":
        """Return a new state by delegating to the active StateEngine plugin.

        The plugin framework (V2.5) routes this call to whichever
        backend is currently registered.  The default ODE plugin
        replicates the original M0 dynamics exactly.
        """
        from pie.state_engine.registry import StateEngineRegistry

        engine = StateEngineRegistry.get_active()
        return engine.update(self)

    def snapshot(self) -> Dict[str, float]:
        """Return a serialisable view of the state for snapshotting.

        The returned dictionary flattens the drives and affect maps as
        well as the turn count.  This function is used when
        generating ``snapshot_exam.json`` during exam runs.
        """
        # Ensure drives and affect are rounded consistently before serialisation
        def _clamp_round(value: float) -> float:
            v = min(max(value, 0.0), 1.0)
            return round(v, self.DECIMAL_PLACES)
        return {
            "schema_version": self.schema_version,
            "turn_count": self.turn_count,
            "creator_anchor": self.creator_anchor,
            "drives": {k: _clamp_round(v) for k, v in self.drives.items()},
            "affect": {k: _clamp_round(v) for k, v in self.affect.items()},
        }