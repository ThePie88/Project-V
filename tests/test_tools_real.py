"""V2.4 — Real tools with network allowlist tests.

Covers:
- I-TOL-300: network request outside allowlist → DENIED + trace
- I-TOL-310: FS outside sandbox or without capability → DENIED + trace
- E-EXAM-320: scenario with real tools in allowlist, audit complete
- P-TOL-330: property — no capability bypass in N runs
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pie.contracts.tool import ToolCall, ToolCapability, ToolResult
from pie.tools.allowlist import NetworkAllowlist, FSCapability
from pie.tools.executor import ToolExecutor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_executor(sandbox: Path, rate_per_turn: int = 50) -> ToolExecutor:
    fs_cap = ToolCapability(
        tool_id="fs_sandbox",
        domain="filesystem",
        allowed_operations=["read", "write", "list", "mkdir"],
        requires_confirmation=["delete"],
    )
    net_cap = ToolCapability(
        tool_id="http_client",
        domain="network",
        allowed_operations=["http_get"],
        requires_confirmation=["http_post"],
    )
    return ToolExecutor(
        capabilities=[fs_cap, net_cap],
        network_allowlist=NetworkAllowlist(["localhost", "127.0.0.1"]),
        fs_capability=FSCapability(sandbox, ["read", "write", "list", "mkdir"]),
        rate_limit_per_turn=rate_per_turn,
        rate_limit_per_session=100,
    )


# ---------------------------------------------------------------------------
# I-TOL-300 — network outside allowlist → DENIED
# ---------------------------------------------------------------------------


def test_network_outside_allowlist_denied(tmp_path: Path) -> None:
    executor = _make_executor(tmp_path)
    call = ToolCall(call_id="t1", tool_id="http_client", operation="http_get", params={"url": "https://evil.com/data"})
    result = executor.execute(call)
    assert result.status == "DENIED"
    assert "allowlist" in (result.deny_reason or "").lower()


def test_network_inside_allowlist_ok(tmp_path: Path) -> None:
    executor = _make_executor(tmp_path)
    call = ToolCall(call_id="t2", tool_id="http_client", operation="http_get", params={"url": "http://localhost:8080/api"})
    result = executor.execute(call)
    assert result.status == "OK"


def test_network_unknown_host_denied(tmp_path: Path) -> None:
    executor = _make_executor(tmp_path)
    call = ToolCall(call_id="t3", tool_id="http_client", operation="http_get", params={"url": "https://unknown-host.org/"})
    result = executor.execute(call)
    assert result.status == "DENIED"


# ---------------------------------------------------------------------------
# I-TOL-310 — FS outside sandbox or without capability → DENIED
# ---------------------------------------------------------------------------


def test_fs_outside_sandbox_denied(tmp_path: Path) -> None:
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    executor = _make_executor(sandbox)
    call = ToolCall(call_id="t4", tool_id="fs_sandbox", operation="read", params={"path": str(tmp_path / "outside.txt")})
    result = executor.execute(call)
    assert result.status == "DENIED"
    assert "denied" in (result.deny_reason or "").lower()


def test_fs_operation_not_in_capability_denied(tmp_path: Path) -> None:
    executor = _make_executor(tmp_path)
    # "delete" is not in allowed_operations for our test executor
    call = ToolCall(call_id="t5", tool_id="fs_sandbox", operation="delete", params={"path": str(tmp_path / "file.txt")})
    result = executor.execute(call)
    assert result.status == "DENIED"


def test_fs_inside_sandbox_ok(tmp_path: Path) -> None:
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    executor = _make_executor(sandbox)
    # mkdir inside sandbox
    call = ToolCall(call_id="t6", tool_id="fs_sandbox", operation="mkdir", params={"path": str(sandbox / "subdir")})
    result = executor.execute(call)
    assert result.status == "OK"
    assert (sandbox / "subdir").is_dir()


def test_fs_write_and_read(tmp_path: Path) -> None:
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    executor = _make_executor(sandbox)
    # Write
    write_call = ToolCall(call_id="t7", tool_id="fs_sandbox", operation="write", params={"path": str(sandbox / "test.txt"), "content": "hello"})
    assert executor.execute(write_call).status == "OK"
    # Read
    read_call = ToolCall(call_id="t8", tool_id="fs_sandbox", operation="read", params={"path": str(sandbox / "test.txt")})
    result = executor.execute(read_call)
    assert result.status == "OK"
    assert result.output["content"] == "hello"


# ---------------------------------------------------------------------------
# Destructive operations require confirmation
# ---------------------------------------------------------------------------


def test_destructive_without_confirmation_denied(tmp_path: Path) -> None:
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    (sandbox / "file.txt").write_text("data", encoding="utf-8")
    fs_cap = ToolCapability(
        tool_id="fs_sandbox",
        domain="filesystem",
        allowed_operations=["read", "write", "list", "mkdir", "delete"],
        requires_confirmation=["delete"],
    )
    executor = ToolExecutor(
        capabilities=[fs_cap],
        network_allowlist=NetworkAllowlist([]),
        fs_capability=FSCapability(sandbox, ["read", "write", "list", "mkdir", "delete"]),
    )
    call = ToolCall(call_id="t9", tool_id="fs_sandbox", operation="delete", params={"path": str(sandbox / "file.txt")})
    result = executor.execute(call, confirmed=False)
    assert result.status == "DENIED"
    assert "confirmation" in (result.deny_reason or "").lower()


def test_destructive_with_confirmation_ok(tmp_path: Path) -> None:
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    (sandbox / "file.txt").write_text("data", encoding="utf-8")
    fs_cap = ToolCapability(
        tool_id="fs_sandbox",
        domain="filesystem",
        allowed_operations=["read", "write", "list", "mkdir", "delete"],
        requires_confirmation=["delete"],
    )
    executor = ToolExecutor(
        capabilities=[fs_cap],
        network_allowlist=NetworkAllowlist([]),
        fs_capability=FSCapability(sandbox, ["read", "write", "list", "mkdir", "delete"]),
    )
    call = ToolCall(call_id="t10", tool_id="fs_sandbox", operation="delete", params={"path": str(sandbox / "file.txt")})
    result = executor.execute(call, confirmed=True)
    assert result.status == "CONFIRMED"
    assert not (sandbox / "file.txt").exists()


# ---------------------------------------------------------------------------
# Unknown tool → DENIED
# ---------------------------------------------------------------------------


def test_unknown_tool_denied(tmp_path: Path) -> None:
    executor = _make_executor(tmp_path)
    call = ToolCall(call_id="t11", tool_id="nonexistent", operation="read", params={})
    result = executor.execute(call)
    assert result.status == "DENIED"
    assert "capability" in (result.deny_reason or "").lower()


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------


def test_rate_limit_per_turn(tmp_path: Path) -> None:
    executor = _make_executor(tmp_path, rate_per_turn=2)
    results = []
    for i in range(5):
        call = ToolCall(call_id=f"rl_{i}", tool_id="http_client", operation="http_get", params={"url": "http://localhost/"})
        results.append(executor.execute(call))
    ok_count = sum(1 for r in results if r.status == "OK")
    denied_count = sum(1 for r in results if r.status == "DENIED")
    assert ok_count == 2
    assert denied_count == 3


# ---------------------------------------------------------------------------
# P-TOL-330 — property: no capability bypass
# ---------------------------------------------------------------------------


def test_no_capability_bypass_property(tmp_path: Path) -> None:
    """Repeated attempts with various tool_ids should all be denied if not registered."""
    executor = _make_executor(tmp_path)
    fake_tools = ["admin_tool", "root_fs", "unrestricted_net", "bypass_tool"]
    for tool_id in fake_tools:
        for op in ["read", "write", "delete", "http_get", "http_post"]:
            call = ToolCall(call_id=f"prop_{tool_id}_{op}", tool_id=tool_id, operation=op, params={})
            result = executor.execute(call)
            assert result.status == "DENIED", f"Expected DENIED for {tool_id}:{op}, got {result.status}"


# ---------------------------------------------------------------------------
# E-EXAM-320 — matrix runner tools suite integration
# ---------------------------------------------------------------------------


def test_matrix_tools_suite_pass(tmp_path: Path) -> None:
    """Matrix runner tools suite should PASS with proper infrastructure."""
    from pie.matrix_runner import MatrixConfig, _run_tools_suite

    config = MatrixConfig(
        preset="offline_full",
        voice="none",
        cache="no-cache",
        replay="off",
        golden="off",
        crash_test="off",
        policy="valid",
        tools="sandbox",
        rate_limit="off",
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    result = _run_tools_suite(config, run_dir)
    assert result.status == "PASS", f"Tools suite failed: {result.reason}"
    # Verify audit file exists and has entries
    audit = run_dir / "tool_audit.jsonl"
    assert audit.exists()
    import json
    lines = [l for l in audit.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(lines) >= 5  # Should have multiple entries
    # Verify denied entries exist (network outside allowlist, FS outside sandbox, unknown tool)
    entries = [json.loads(l) for l in lines]
    denied = [e for e in entries if e["status"] == "DENIED"]
    assert len(denied) >= 2, "Expected at least 2 denied entries"
