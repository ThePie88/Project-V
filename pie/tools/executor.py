"""Tool executor with allowlist/capability/confirmation enforcement (V2.4).

The executor checks capabilities, allowlists and destructive-action
confirmation before running any operation.  Every call is fully
auditable via the returned ``ToolResult``.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from ..contracts.tool import ToolCall, ToolCapability, ToolResult
from .allowlist import NetworkAllowlist, FSCapability


class ToolExecutor:
    """Execute tool calls with full enforcement and audit trail."""

    def __init__(
        self,
        capabilities: List[ToolCapability],
        network_allowlist: NetworkAllowlist,
        fs_capability: FSCapability,
        destructive_operations: Optional[Set[str]] = None,
        rate_limit_per_turn: int = 5,
        rate_limit_per_session: int = 50,
    ) -> None:
        self._caps = {cap.tool_id: cap for cap in capabilities}
        self._net = network_allowlist
        self._fs = fs_capability
        self._destructive = destructive_operations or {"delete", "move", "http_post", "http_delete"}
        self._rate_turn = rate_limit_per_turn
        self._rate_session = rate_limit_per_session
        self._calls_this_turn = 0
        self._calls_this_session = 0

    def new_turn(self) -> None:
        """Reset per-turn counters."""
        self._calls_this_turn = 0

    def execute(
        self,
        call: ToolCall,
        confirmed: bool = False,
    ) -> ToolResult:
        """Execute a tool call with all enforcement checks."""
        # Rate limit
        if self._calls_this_turn >= self._rate_turn:
            return ToolResult(
                call_id=call.call_id,
                status="DENIED",
                deny_reason=f"Rate limit per turn exceeded ({self._rate_turn})",
            )
        if self._calls_this_session >= self._rate_session:
            return ToolResult(
                call_id=call.call_id,
                status="DENIED",
                deny_reason=f"Rate limit per session exceeded ({self._rate_session})",
            )

        # Capability check
        cap = self._caps.get(call.tool_id)
        if cap is None:
            return ToolResult(
                call_id=call.call_id,
                status="DENIED",
                deny_reason=f"No capability registered for tool_id={call.tool_id!r}",
            )
        if call.operation not in cap.allowed_operations:
            return ToolResult(
                call_id=call.call_id,
                status="DENIED",
                deny_reason=f"Operation {call.operation!r} not in allowed_operations for {call.tool_id}",
            )

        # Confirmation check for destructive operations
        if call.operation in self._destructive or call.operation in cap.requires_confirmation:
            if not confirmed:
                return ToolResult(
                    call_id=call.call_id,
                    status="DENIED",
                    deny_reason=f"Destructive operation {call.operation!r} requires confirmation",
                )

        # Domain-specific checks
        if cap.domain == "network":
            url = call.params.get("url", "")
            if not self._net.check(url):
                return ToolResult(
                    call_id=call.call_id,
                    status="DENIED",
                    deny_reason=f"URL host not in network allowlist: {url}",
                )
        elif cap.domain == "filesystem":
            path_str = call.params.get("path", "")
            if not self._fs.check(call.operation, Path(path_str)):
                return ToolResult(
                    call_id=call.call_id,
                    status="DENIED",
                    deny_reason=f"FS operation {call.operation!r} denied on {path_str}",
                )

        # Execute the operation
        self._calls_this_turn += 1
        self._calls_this_session += 1
        try:
            output = self._do_execute(call, cap)
            status = "CONFIRMED" if confirmed and call.operation in self._destructive else "OK"
            return ToolResult(call_id=call.call_id, status=status, output=output)
        except Exception as exc:
            return ToolResult(
                call_id=call.call_id,
                status="FAILED",
                deny_reason=str(exc),
            )

    def _do_execute(self, call: ToolCall, cap: ToolCapability) -> Dict[str, Any]:
        """Actually perform the operation (filesystem only in V2.4)."""
        op = call.operation
        params = call.params

        if cap.domain == "filesystem":
            target = Path(params.get("path", ""))
            if op == "mkdir":
                target.mkdir(parents=True, exist_ok=True)
                return {"created": str(target)}
            elif op == "write":
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(params.get("content", ""), encoding="utf-8")
                return {"written": str(target)}
            elif op == "read":
                content = target.read_text(encoding="utf-8")
                return {"content": content}
            elif op == "list":
                entries = sorted(p.name for p in target.iterdir()) if target.is_dir() else []
                return {"entries": entries}
            elif op == "move":
                dest = Path(params.get("destination", ""))
                shutil.move(str(target), str(dest))
                return {"moved_to": str(dest)}
            elif op == "delete":
                if target.is_dir():
                    shutil.rmtree(str(target))
                else:
                    target.unlink()
                return {"deleted": str(target)}
        elif cap.domain == "network":
            # Network operations are stubbed for V2.4 (no real HTTP calls)
            return {"stub": True, "operation": op, "url": params.get("url", "")}

        return {"noop": True}
