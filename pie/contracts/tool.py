"""Tool contracts for the Pie Kernel (V2.4).

Defines the data models for tool capabilities, calls and results.
All tool interactions are audited via TOOL_CALL / TOOL_RESULT /
TOOL_DENIED events in the trace.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class ToolCapability(BaseModel):
    """Declares what a tool is allowed to do."""

    tool_id: str = Field(..., description="Unique tool identifier")
    domain: Literal["filesystem", "network"] = Field(
        ..., description="Tool domain"
    )
    allowed_operations: List[str] = Field(
        default_factory=list,
        description="Operations this tool may perform (e.g. read, write, http_get)",
    )
    requires_confirmation: List[str] = Field(
        default_factory=list,
        description="Operations that require explicit user confirmation",
    )


class ToolCall(BaseModel):
    """A request to execute a tool operation."""

    call_id: str = Field(..., description="Unique call identifier")
    tool_id: str = Field(..., description="Tool to invoke")
    operation: str = Field(..., description="Operation to perform")
    params: Dict[str, Any] = Field(
        default_factory=dict, description="Operation parameters"
    )


class ToolResult(BaseModel):
    """The outcome of a tool call."""

    call_id: str = Field(..., description="Matches the originating ToolCall")
    status: Literal["OK", "DENIED", "CONFIRMED", "FAILED"] = Field(
        ..., description="Outcome status"
    )
    output: Optional[Dict[str, Any]] = Field(
        None, description="Result payload (if any)"
    )
    deny_reason: Optional[str] = Field(
        None, description="Why the call was denied (if DENIED)"
    )
