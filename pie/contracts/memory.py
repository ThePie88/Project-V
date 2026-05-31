"""Memory record contract for the Pie Kernel.

Memory records are append-only and separate from the trace. Each record
must include source references to the trace events that justify it.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal

from pydantic import BaseModel, Field, field_validator


class LogicalTime(BaseModel):
    """Deterministic logical time for memory records."""

    session: int = Field(..., ge=0, description="Session identifier (deterministic)")
    turn: int = Field(..., ge=0, description="Turn index within the session")
    idx: int = Field(..., ge=0, description="Record index within the turn")


MemoryType = Literal["NarrativeMemory", "Belief", "Preference", "TrustUpdate"]


class MemoryRecord(BaseModel):
    """Append-only memory record."""

    schema_version: Literal["0.1"] = Field(
        "0.1",
        description="Schema version for MemoryRecord; bump on backwards incompatible change",
    )
    memory_id: str = Field(..., description="Deterministic memory identifier")
    logical_time: LogicalTime = Field(..., description="Deterministic logical time")
    type: MemoryType = Field(..., description="Memory record type")
    content: Dict[str, Any] = Field(..., description="Structured memory content")
    source_refs: List[int] = Field(..., description="Trace event ids that justify this record")
    tags: List[str] = Field(default_factory=list, description="Optional tags for indexing")
    context: Dict[str, Any] = Field(default_factory=dict, description="Optional context metadata")

    @field_validator("source_refs")
    @classmethod
    def source_refs_required(cls, v: List[int]) -> List[int]:
        if not v:
            raise ValueError("source_refs must not be empty")
        return v
