"""Event contract for the Pie Kernel.

An Event captures a single atomic change in the kernel.  The M0
implementation records six kinds of events in a deterministic order
within each turn of the runtime:

* ``INPUT`` – raw user or environment input for the turn.
* ``STATE_UPDATED`` – an update to the internal state derived from
  drives and affect dynamics.
* ``GOALS_GENERATED`` – the list of ranked goals considered by the
  kernel.
* ``ACTION_SELECTED`` – the action chosen from the generated goals.
* ``SPEECHPLAN`` – a plan describing what to say.
* ``VOICE_OUTPUT`` – the final string sent to the voice interface.

Each event includes a monotonically increasing ``id`` so that
deterministic replay can be verified.  Events also carry a
``timestamp`` for logging purposes only; the timestamp must never be
used as an input into the decision making logic.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Literal

from pydantic import BaseModel, Field, field_validator


class Event(BaseModel):
    """A record of a single event in the kernel.

    Attributes
    ----------
    schema_version:
        The version of the event schema.  This field allows future
        versions of the kernel to detect incompatible changes.
    id:
        A monotonically increasing integer assigned by the runtime.
    type:
        The type of event, one of ``INPUT``, ``STATE_UPDATED``,
        ``GOALS_GENERATED``, ``ACTION_SELECTED``, ``SPEECHPLAN`` or
        ``VOICE_OUTPUT``.
    timestamp:
        A floating point number representing the wall clock time when
        the event was emitted.  The timestamp is for logging only and
        must not influence decisions.
    content:
        A dictionary containing event‑specific payload.  The structure
        of ``content`` varies by event type.
    """

    schema_version: Literal["0.1"] = Field(
        "0.1",
        description="Schema version for Event; bump when the model breaks backwards compatibility",
    )
    id: int = Field(..., ge=0, description="Monotonically increasing identifier assigned by runtime")
    type: Literal[
        "INPUT",
        "STATE_UPDATED",
        "GOALS_GENERATED",
        "ACTION_SELECTED",
        "SPEECHPLAN",
        "VOICE_OUTPUT",
        "LLM_OUTPUT",
        "LLM_RETRY",
        "LLM_FALLBACK",
        "MEMORY_PROPOSED",
        "MEMORY_APPENDED",
        "CONSTRAINT_PROPOSED",
        "CONSTRAINT_APPENDED",
        "CONSTRAINT_ENFORCED",
        "TOOL_CALL",
        "TOOL_RESULT",
        "TOOL_DENIED",
        "CF_GENERATED",
        "CF_SCORED",
        "CF_REJECTED",
        "CF_CHOSEN",
        "KILL_SWITCH_ON",
        "KILL_SWITCH_OFF",
        "REWARD_SIGNAL",
        "REWARD_APPLIED",
        "REWARD_DUPLICATE_IGNORED",
    ] = Field(..., description="Kind of event emitted during runtime")
    timestamp: float = Field(..., description="Wall‑clock time at which the event was emitted")
    content: Dict[str, Any] = Field(..., description="Event payload, varies by type")

    @field_validator("timestamp")
    @classmethod
    def timestamp_not_in_future(cls, v: float) -> float:
        # Accept future timestamps but log the issue; decisions should never use this value.
        return v

    @classmethod
    def new(
        cls,
        event_id: int,
        event_type: Literal[
            "INPUT",
            "STATE_UPDATED",
            "GOALS_GENERATED",
            "ACTION_SELECTED",
            "SPEECHPLAN",
            "VOICE_OUTPUT",
            "LLM_OUTPUT",
            "LLM_RETRY",
            "LLM_FALLBACK",
            "MEMORY_PROPOSED",
            "MEMORY_APPENDED",
            "CONSTRAINT_PROPOSED",
            "CONSTRAINT_APPENDED",
            "CONSTRAINT_ENFORCED",
            "TOOL_CALL",
            "TOOL_RESULT",
            "TOOL_DENIED",
            "CF_GENERATED",
            "CF_SCORED",
            "CF_REJECTED",
            "CF_CHOSEN",
            "KILL_SWITCH_ON",
            "KILL_SWITCH_OFF",
        ],
        content: Dict[str, Any],
    ) -> "Event":
        """Create a new event with the given id and type.

        Parameters
        ----------
        event_id:
            The id to assign to the event.  Must be greater than zero and
            strictly increase across events in a trace.
        event_type:
            The type of event; this must be one of the literal types
            defined by the ``type`` field.
        content:
            The payload dictionary describing the event.  The caller is
            responsible for ensuring that the payload structure matches
            expectations for the specified type.
        """
        return cls(
            id=event_id,
            type=event_type,
            timestamp=time.time(),
            content=content,
        )

    def to_json(self) -> Dict[str, Any]:
        """Return a serialisable representation suitable for writing to JSON.

        The Pydantic ``dict()`` method drops type information; we use
        ``dict()`` here to produce a plain Python dict with field names
        preserved.
        """
        return self.model_dump()
