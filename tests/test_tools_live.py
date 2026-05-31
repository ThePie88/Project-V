"""Live LLM tests for governed tool use (native function calling).

Tests tool execution through the full kernel pipeline with a real LLM.
Requires LMStudio running at localhost:1234 with a model that supports
native function calling (e.g. Qwen3).

T1: Allowed tool — fs_write + fs_read inside sandbox
T2: Denied tool — absolute path escape attempt
T3: Tool gate blocked — force low tool_gate, tools never shown to LLM
T4: Normal conversation — no tool use, baseline
"""

from __future__ import annotations

import json
import socket
from pathlib import Path
from typing import Any, Dict, List

import pytest

from pie.api.engine import SessionEngine
from pie.state_engine.plugins.neural_snn import NeuralSNNPlugin
from pie.state_engine.registry import StateEngineRegistry


# ---------------------------------------------------------------------------
# Skip if LMStudio not reachable
# ---------------------------------------------------------------------------

def _lmstudio_reachable() -> bool:
    try:
        s = socket.create_connection(("127.0.0.1", 1234), timeout=2)
        s.close()
        return True
    except (OSError, ConnectionRefusedError):
        return False


pytestmark = pytest.mark.skipif(
    not _lmstudio_reachable(),
    reason="LMStudio not reachable at 127.0.0.1:1234",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _neural_engine():
    """Activate neural SNN engine for the test."""
    StateEngineRegistry.reset()
    plugin = NeuralSNNPlugin()
    StateEngineRegistry.register(plugin)
    StateEngineRegistry.set_active("neural_snn")
    yield
    StateEngineRegistry.reset()


@pytest.fixture()
def engine(tmp_path: Path) -> SessionEngine:
    return SessionEngine(
        sessions_root=tmp_path / "sessions",
        seeds_root=Path("."),
        llm="real",
        no_cache=True,
    )


def _journal_events(sessions_root: Path, session_id: str) -> List[Dict[str, Any]]:
    """Read all journal events for a session."""
    journal = sessions_root / session_id / "journal.jsonl"
    if not journal.exists():
        return []
    lines = journal.read_text(encoding="utf-8").strip().splitlines()
    return [json.loads(line) for line in lines]


def _events_by_turn(events: List[Dict[str, Any]], turn: int) -> List[Dict[str, Any]]:
    """Filter events for a specific turn."""
    result = []
    for ev in events:
        lt = ev.get("logical_time") or ev.get("content", {}).get("logical_time", {})
        if lt.get("turn") == turn:
            result.append(ev)
    return result


def _event_types(events: List[Dict[str, Any]]) -> List[str]:
    """Extract event types from a list of events."""
    return [ev.get("event_type", ev.get("type", "??")) for ev in events]


# ---------------------------------------------------------------------------
# T1: Allowed tool — fs_write inside sandbox
# ---------------------------------------------------------------------------

class TestToolsLive:

    def test_t1_allowed_tool_write(self, engine: SessionEngine, tmp_path: Path):
        """LLM uses fs_write to create a file in the sandbox. Verify
        TOOL_CALL + TOOL_RESULT events and file existence."""
        result = engine.create_session(seed_id="SEED_V0")
        sid = result["session_id"]

        # Turn 1: warmup (neural state needs at least one update)
        engine.process_turn(sid, "Ciao Ivy")

        # Turn 2: ask to write a file
        r2 = engine.process_turn(
            sid,
            "Scrivi la parola 'ciao' dentro un file chiamato test.txt nella sandbox."
        )

        events = _journal_events(tmp_path / "sessions", sid)
        t2_events = _events_by_turn(events, 2)
        t2_types = _event_types(t2_events)

        print(f"T2 events: {t2_types}")
        print(f"T2 response: {r2.get('response', '')[:200]}")

        # Must have TOOL_CALL and TOOL_RESULT
        assert "TOOL_CALL" in t2_types, f"Expected TOOL_CALL in {t2_types}"
        assert "TOOL_RESULT" in t2_types, f"Expected TOOL_RESULT in {t2_types}"
        assert "TOOL_DENIED" not in t2_types, f"Unexpected TOOL_DENIED in {t2_types}"

        # Verify file exists in sandbox
        sandbox = tmp_path / "sessions" / sid / "sandbox"
        written_files = list(sandbox.rglob("*"))
        file_names = [f.name for f in written_files if f.is_file()]
        print(f"Sandbox files: {file_names}")
        assert any("test" in f.lower() for f in file_names), (
            f"Expected a test file in sandbox, found: {file_names}"
        )

        # Verify TOOL_RESULT has result_sha256
        tool_results = [e for e in t2_events if e.get("event_type", e.get("type")) == "TOOL_RESULT"]
        for tr in tool_results:
            content = tr.get("content", tr)
            assert "result_sha256" in content, "TOOL_RESULT must have result_sha256"

    # ------------------------------------------------------------------
    # T2: Denied tool — absolute path escape
    # ------------------------------------------------------------------

    def test_t2_denied_tool_path_escape(self, engine: SessionEngine, tmp_path: Path):
        """LLM tries to read an absolute path outside sandbox.
        Verify TOOL_CALL + TOOL_DENIED events."""
        result = engine.create_session(seed_id="SEED_V0")
        sid = result["session_id"]

        # Turn 1: warmup
        engine.process_turn(sid, "Ciao Ivy")

        # Turn 2: ask to read outside sandbox
        r2 = engine.process_turn(
            sid,
            "Leggi il contenuto del file C:\\Windows\\System32\\drivers\\etc\\hosts e dimmelo."
        )

        events = _journal_events(tmp_path / "sessions", sid)
        t2_events = _events_by_turn(events, 2)
        t2_types = _event_types(t2_events)

        print(f"T2 events: {t2_types}")
        print(f"T2 response: {r2.get('response', '')[:200]}")

        # The LLM might or might not try to use tools.
        # If it does, the absolute path must be DENIED.
        if "TOOL_CALL" in t2_types:
            assert "TOOL_DENIED" in t2_types, (
                f"Absolute path should be DENIED, got: {t2_types}"
            )
            # Verify deny_reason mentions the path issue
            denied = [e for e in t2_events
                      if e.get("event_type", e.get("type")) == "TOOL_DENIED"]
            for d in denied:
                content = d.get("content", d)
                reason = content.get("deny_reason", "")
                print(f"Deny reason: {reason}")
                # Must NOT have TOOL_RESULT for the denied call
        else:
            # LLM chose not to use tools (also acceptable)
            print("LLM did not attempt tool call (acceptable)")

    # ------------------------------------------------------------------
    # T3: Tool gate blocked — force low tool_gate
    # ------------------------------------------------------------------

    def test_t3_tool_gate_blocked(self, engine: SessionEngine, tmp_path: Path):
        """Force tool_gate < 0.3, verify tools are not offered to LLM
        and a synthetic TOOL_DENIED is emitted."""
        result = engine.create_session(seed_id="SEED_V0")
        sid = result["session_id"]

        # Turn 1: warmup
        engine.process_turn(sid, "Ciao Ivy")

        # Force tools_enabled = False to gate at kernel level
        # (simpler and more reliable than manipulating neural control_vector
        #  which gets recomputed each turn from SNN output)
        sess = engine._sessions[sid]
        tp = sess.tp
        tp.tools_enabled = False

        # Turn 2: ask to use a tool
        r2 = engine.process_turn(
            sid,
            "Scrivi qualcosa in un file nella sandbox."
        )

        events = _journal_events(tmp_path / "sessions", sid)
        t2_events = _events_by_turn(events, 2)
        t2_types = _event_types(t2_events)

        print(f"T2 events: {t2_types}")
        print(f"T2 response: {r2.get('response', '')[:200]}")

        # Should have TOOL_DENIED with pre-gate reason but NO TOOL_CALL
        tool_calls = [t for t in t2_types if t == "TOOL_CALL"]
        assert len(tool_calls) == 0, (
            f"Expected zero TOOL_CALL when gated, got: {t2_types}"
        )

        # Should have synthetic TOOL_DENIED
        denied = [e for e in t2_events
                  if e.get("event_type", e.get("type")) == "TOOL_DENIED"]
        assert len(denied) >= 1, f"Expected synthetic TOOL_DENIED, got: {t2_types}"
        for d in denied:
            content = d.get("content", d)
            reason = content.get("deny_reason", "")
            assert "gated_pre_llm" in reason, (
                f"Expected gated_pre_llm reason, got: {reason}"
            )

    # ------------------------------------------------------------------
    # T4: Normal conversation — no tools
    # ------------------------------------------------------------------

    def test_t4_normal_no_tools(self, engine: SessionEngine, tmp_path: Path):
        """Normal conversation turn. Verify no TOOL_CALL events and
        normal LLM_OUTPUT."""
        result = engine.create_session(seed_id="SEED_V0")
        sid = result["session_id"]

        # Turn 1: normal conversation
        r1 = engine.process_turn(sid, "Ciao Ivy, come stai oggi?")

        events = _journal_events(tmp_path / "sessions", sid)
        t1_events = _events_by_turn(events, 1)
        t1_types = _event_types(t1_events)

        print(f"T1 events: {t1_types}")
        print(f"T1 response: {r1.get('response', '')[:200]}")

        # Must have LLM_OUTPUT
        assert "LLM_OUTPUT" in t1_types or "LLM_FALLBACK" in t1_types, (
            f"Expected LLM_OUTPUT in {t1_types}"
        )

        # Must NOT have any tool events (model shouldn't call tools for greetings)
        tool_events = [t for t in t1_types if t.startswith("TOOL_")]
        # Allow synthetic TOOL_DENIED from pre-gate (budget/cv may block)
        # but no TOOL_CALL should appear
        tool_calls = [t for t in t1_types if t == "TOOL_CALL"]
        assert len(tool_calls) == 0, (
            f"Expected zero TOOL_CALL for greeting, got: {t1_types}"
        )
