"""Constraint contract for the Pie Kernel."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Union

from pydantic import BaseModel, Field, field_validator


class ConstraintLogicalTime(BaseModel):
    session: int = Field(..., ge=0, description="Session identifier (deterministic)")
    turn: int = Field(..., ge=0, description="Turn index within the session")
    idx: int = Field(..., ge=0, description="Constraint index within the turn")


EffectType = Literal["forbid", "require_confirmation", "increase_caution", "adjust_trust"]


class ConstraintEffect(BaseModel):
    type: EffectType = Field(..., description="Constraint effect type")
    params: Dict[str, Any] = Field(default_factory=dict, description="Effect parameters")


Severity = Literal["soft", "hard"]
Status = Literal["pending", "active", "expired", "rejected"]
CommitPolicy = Literal["auto", "creator_confirm"]


class ConstraintRecord(BaseModel):
    schema_version: Literal["0.1"] = Field(
        "0.1",
        description="Schema version for ConstraintRecord; bump on backwards incompatible change",
    )
    constraint_id: str = Field(..., description="Deterministic constraint identifier")
    logical_time: ConstraintLogicalTime = Field(..., description="Deterministic logical time")
    family: str = Field(..., description="Constraint family")
    rule_id: str = Field(..., description="Rule identifier")
    severity: Severity = Field(..., description="Constraint severity")
    status: Status = Field(..., description="Constraint status")
    commit_policy: CommitPolicy = Field(..., description="Commit policy for pending review")
    effects: List[ConstraintEffect] = Field(..., description="Effects applied by the constraint")
    trigger_events: List[int] = Field(..., description="Event ids that triggered this constraint")
    explanation: str = Field(..., description="Brief audit explanation")
    strength: Union[float, Literal["none"]] = Field("none", description="Constraint strength")
    decay: Union[float, Literal["none"]] = Field("none", description="Constraint decay")
    cooldown: Union[int, Literal["none"]] = Field("none", description="Cooldown in turns")

    @field_validator("trigger_events")
    @classmethod
    def trigger_events_required(cls, v: List[int]) -> List[int]:
        if not v:
            raise ValueError("trigger_events must not be empty")
        return v
