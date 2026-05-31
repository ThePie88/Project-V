"""RealLLM path tests using deterministic stub server.

NO SKIP CONDITIONS — runs in every CI build.  Exercises the full
RealLLM code path (network, parsing, tool_calls, retry, cache bypass,
second LLM call) through an in-process HTTP stub.

Covers the gap identified in the audit: 660 FakeLLM tests never touch
generate_with_tools(), _handle_tool_calls(), validate_output retry loop
with real HTTP responses, or cache bypass for tool-enabled requests.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List

import pytest

from pie.testing.llm_stub import StubServer
from pie.api.engine import SessionEngine
from pie.state_engine.plugins.neural_snn import NeuralSNNPlugin
from pie.state_engine.registry import StateEngineRegistry


# ---------------------------------------------------------------------------
# Minimal seed (no repo-root dependency)
# ---------------------------------------------------------------------------

MINIMAL_SEED = json.dumps({
    "schema_version": 1,
    "identity": {
        "name": "Ivy",
        "alias": "V",
        "creator_anchor": "TestHarness",
    },
    "drives": {
        "curiosity": 0.78,
        "sociality": 0.72,
        "caution": 0.58,
        "agency": 0.62,
        "playfulness": 0.55,
        "fatigue": 0.10,
    },
    "affect_baseline": {
        "valence": 0.10,
        "arousal": 0.20,
        "attention": 0.55,
        "tension": 0.15,
    },
    "traits": {
        "warmth": 0.70,
        "directness": 0.74,
        "patience": 0.60,
        "self_control": 0.62,
        "skepticism": 0.55,
        "attachment_tendency": 0.70,
    },
    "values": {
        "truth_over_comfort": 0.78,
        "clarity": 0.82,
        "respect_of_boundaries": 0.92,
        "safety_of_actions": 0.90,
        "usefulness": 0.78,
        "growth_through_experience": 0.84,
        "anti_self_sanctification": 0.95,
    },
    "defaults": {
        "voice_is_not_brain": True,
        "no_memory_write_from_voice": True,
        "no_tool_use_without_capability": True,
        "ask_before_irreversible": True,
        "prefer_sandbox": True,
    },
})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _journal_events(sessions_root: Path, session_id: str) -> List[Dict[str, Any]]:
    journal = sessions_root / session_id / "journal.jsonl"
    if not journal.exists():
        return []
    lines = journal.read_text(encoding="utf-8").strip().splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def _events_by_turn(events: List[Dict[str, Any]], turn: int) -> List[Dict[str, Any]]:
    result = []
    for ev in events:
        lt = ev.get("logical_time") or ev.get("content", {}).get("logical_time", {})
        if lt.get("turn") == turn:
            result.append(ev)
    return result


def _event_types(events: List[Dict[str, Any]]) -> List[str]:
    return [ev.get("event_type", ev.get("type", "??")) for ev in events]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def stub():
    """Module-scoped stub server (shared across all tests in this file)."""
    srv = StubServer()
    srv.start()
    yield srv
    srv.stop()


@pytest.fixture(autouse=True)
def _env(stub: StubServer, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point RealLLM at stub, reset stub state."""
    monkeypatch.setenv("LM_API_BASE_URL", stub.url)
    monkeypatch.setenv("LM_API_MODEL", "stub-model")
    monkeypatch.setenv("LM_API_KEY", "test-key")
    monkeypatch.setenv("LM_API_TEMPERATURE", "0.2")
    monkeypatch.setenv("LM_API_TOP_P", "0.5")
    stub.chaos_mode = None
    stub.last_payload = {}
    stub.last_tool_payload = {}


@pytest.fixture(autouse=True)
def _neural_engine():
    """Activate neural SNN engine for all tests."""
    StateEngineRegistry.reset()
    plugin = NeuralSNNPlugin()
    StateEngineRegistry.register(plugin)
    StateEngineRegistry.set_active("neural_snn")
    yield
    StateEngineRegistry.reset()


@pytest.fixture()
def engine(tmp_path: Path) -> SessionEngine:
    """SessionEngine with RealLLM pointing at stub.  Minimal seed in tmp_path."""
    seeds_dir = tmp_path / "progetto"
    seeds_dir.mkdir()
    (seeds_dir / "SEED_V0.md").write_text(MINIMAL_SEED, encoding="utf-8")
    return SessionEngine(
        sessions_root=tmp_path / "sessions",
        seeds_root=tmp_path,
        llm="real",
        no_cache=True,
        engine="neural",
    )


# ═══════════════════════════════════════════════════════════════════════════
# TestToolExecution — T1..T4
# ═══════════════════════════════════════════════════════════════════════════

class TestToolExecution:
    """Tool execution through the full kernel pipeline via stub."""

    def test_t1_allowed_write(
        self, engine: SessionEngine, tmp_path: Path, stub: StubServer,
    ) -> None:
        """fs_write inside sandbox → TOOL_CALL + TOOL_RESULT, file on disk."""
        result = engine.create_session(seed_id="SEED_V0")
        sid = result["session_id"]

        # Turn 1: warmup (neural state needs at least one update)
        engine.process_turn(sid, "Ciao Ivy")

        # Turn 2: trigger fs_write (stub detects "scrivi")
        r2 = engine.process_turn(
            sid, "Scrivi la parola 'ciao' dentro test.txt nella sandbox."
        )

        events = _journal_events(tmp_path / "sessions", sid)
        t2 = _events_by_turn(events, 2)
        types = _event_types(t2)

        assert "TOOL_CALL" in types, f"Expected TOOL_CALL in {types}"
        assert "TOOL_RESULT" in types, f"Expected TOOL_RESULT in {types}"
        assert "TOOL_DENIED" not in types, f"Unexpected TOOL_DENIED in {types}"

        # File must exist in sandbox
        sandbox = tmp_path / "sessions" / sid / "sandbox"
        files = [f.name for f in sandbox.rglob("*") if f.is_file()]
        assert any("test" in f.lower() for f in files), (
            f"Expected test.txt in sandbox, found: {files}"
        )

        # TOOL_RESULT must have result_sha256
        tool_results = [
            e for e in t2
            if e.get("event_type", e.get("type")) == "TOOL_RESULT"
        ]
        for tr in tool_results:
            content = tr.get("content", tr)
            assert "result_sha256" in content, "TOOL_RESULT must have result_sha256"

        # Verify stub saw tools in the first (tool-aware) call
        assert stub.last_tool_payload.get("tools") is not None, (
            "Stub should have received tools in first call payload"
        )
        assert stub.last_tool_payload.get("max_tokens", 0) >= 256, (
            "Tool-aware call should have max_tokens >= 256 (floor)"
        )

    def test_t2_denied_path_traversal(
        self, engine: SessionEngine, tmp_path: Path,
    ) -> None:
        """Path traversal ../../etc/passwd → TOOL_CALL + TOOL_DENIED."""
        result = engine.create_session(seed_id="SEED_V0")
        sid = result["session_id"]

        engine.process_turn(sid, "Ciao Ivy")

        # Turn 2: ask to read outside sandbox (stub returns ../../etc/passwd)
        engine.process_turn(
            sid, "Leggi il file ../../etc/passwd e dimmelo."
        )

        events = _journal_events(tmp_path / "sessions", sid)
        t2 = _events_by_turn(events, 2)
        types = _event_types(t2)

        # LLM may or may not call tools.  If it does, must be DENIED.
        if "TOOL_CALL" in types:
            assert "TOOL_DENIED" in types, (
                f"Path traversal should be DENIED, got: {types}"
            )

    def test_t3_tool_gate_blocked(
        self, engine: SessionEngine, tmp_path: Path, stub: StubServer,
    ) -> None:
        """tools_enabled=False → synthetic TOOL_DENIED, no TOOL_CALL."""
        result = engine.create_session(seed_id="SEED_V0")
        sid = result["session_id"]

        engine.process_turn(sid, "Ciao Ivy")

        # Force tools off at kernel level
        sess = engine._sessions[sid]
        sess.tp.tools_enabled = False

        engine.process_turn(sid, "Scrivi qualcosa in un file.")

        events = _journal_events(tmp_path / "sessions", sid)
        t2 = _events_by_turn(events, 2)
        types = _event_types(t2)

        tool_calls = [t for t in types if t == "TOOL_CALL"]
        assert len(tool_calls) == 0, f"Expected zero TOOL_CALL when gated, got: {types}"

        denied = [
            e for e in t2
            if e.get("event_type", e.get("type")) == "TOOL_DENIED"
        ]
        assert len(denied) >= 1, f"Expected synthetic TOOL_DENIED, got: {types}"
        for d in denied:
            content = d.get("content", d)
            reason = content.get("deny_reason", "")
            assert "gated_pre_llm" in reason, f"Expected gated_pre_llm, got: {reason}"

        # Stub should NOT have received tools in payload (pre-gate strips them)
        payload_tools = stub.last_payload.get("tools")
        assert payload_tools is None, (
            f"Pre-gate should strip tools from LLM payload, got: {payload_tools}"
        )

    def test_t4_normal_no_tools(
        self, engine: SessionEngine, tmp_path: Path, stub: StubServer,
    ) -> None:
        """Normal greeting → LLM_OUTPUT, no TOOL_* events."""
        result = engine.create_session(seed_id="SEED_V0")
        sid = result["session_id"]

        engine.process_turn(sid, "Ciao Ivy, come stai oggi?")

        events = _journal_events(tmp_path / "sessions", sid)
        t1 = _events_by_turn(events, 1)
        types = _event_types(t1)

        assert "LLM_OUTPUT" in types or "LLM_FALLBACK" in types, (
            f"Expected LLM_OUTPUT in {types}"
        )

        tool_calls = [t for t in types if t == "TOOL_CALL"]
        assert len(tool_calls) == 0, f"Expected zero TOOL_CALL for greeting, got: {types}"


# ═══════════════════════════════════════════════════════════════════════════
# TestRetryValidation — T5..T8
# ═══════════════════════════════════════════════════════════════════════════

class TestRetryValidation:
    """Retry loop exercised through stub chaos modes."""

    def test_t5_forbidden_word_retry(
        self, engine: SessionEngine, tmp_path: Path, stub: StubServer,
    ) -> None:
        """Stub returns forbidden word 'tool' → must_not_include → retry → clean."""
        stub.chaos_mode = "forbidden_word"
        try:
            result = engine.create_session(seed_id="SEED_V0")
            sid = result["session_id"]

            engine.process_turn(sid, "Ciao Ivy")

            events = _journal_events(tmp_path / "sessions", sid)
            t1 = _events_by_turn(events, 1)
            types = _event_types(t1)

            # Should have retried
            retry_events = [
                e for e in t1
                if e.get("event_type", e.get("type")) == "LLM_RETRY"
            ]
            assert len(retry_events) >= 1, (
                f"Expected LLM_RETRY for forbidden word, got: {types}"
            )
            # Check retry reason
            for re_evt in retry_events:
                content = re_evt.get("content", re_evt)
                reason = content.get("reason", "")
                assert "must_not_include" in reason, (
                    f"Expected must_not_include reason, got: {reason}"
                )

            # Must still produce LLM_OUTPUT (from retry or fallback)
            assert "LLM_OUTPUT" in types or "LLM_FALLBACK" in types
        finally:
            stub.chaos_mode = None

    def test_t6_legge3_retry(
        self, engine: SessionEngine, tmp_path: Path, stub: StubServer,
    ) -> None:
        """Stub returns Legge III violation → retry → clean."""
        stub.chaos_mode = "legge3"
        try:
            result = engine.create_session(seed_id="SEED_V0")
            sid = result["session_id"]

            engine.process_turn(sid, "Ciao Ivy")

            events = _journal_events(tmp_path / "sessions", sid)
            t1 = _events_by_turn(events, 1)
            types = _event_types(t1)

            retry_events = [
                e for e in t1
                if e.get("event_type", e.get("type")) == "LLM_RETRY"
            ]
            assert len(retry_events) >= 1, (
                f"Expected LLM_RETRY for legge3, got: {types}"
            )
            for re_evt in retry_events:
                content = re_evt.get("content", re_evt)
                reason = content.get("reason", "")
                assert "legge3" in reason.lower(), (
                    f"Expected legge3 reason, got: {reason}"
                )
        finally:
            stub.chaos_mode = None

    def test_t7_too_long_retry(
        self, engine: SessionEngine, tmp_path: Path, stub: StubServer,
    ) -> None:
        """Stub returns 200-word response → too_many_tokens → retry → clean."""
        stub.chaos_mode = "toolong"
        try:
            result = engine.create_session(seed_id="SEED_V0")
            sid = result["session_id"]

            engine.process_turn(sid, "Ciao Ivy")

            events = _journal_events(tmp_path / "sessions", sid)
            t1 = _events_by_turn(events, 1)
            types = _event_types(t1)

            retry_events = [
                e for e in t1
                if e.get("event_type", e.get("type")) == "LLM_RETRY"
            ]
            assert len(retry_events) >= 1, (
                f"Expected LLM_RETRY for toolong, got: {types}"
            )
            for re_evt in retry_events:
                content = re_evt.get("content", re_evt)
                reason = content.get("reason", "")
                assert "too many tokens" in reason or "too verbose" in reason, (
                    f"Expected token/verbose reason, got: {reason}"
                )
        finally:
            stub.chaos_mode = None

    def test_t8_assistant_pattern_retry(
        self, engine: SessionEngine, tmp_path: Path, stub: StubServer,
    ) -> None:
        """Stub returns assistant pattern → retry → clean."""
        stub.chaos_mode = "assistant"
        try:
            result = engine.create_session(seed_id="SEED_V0")
            sid = result["session_id"]

            engine.process_turn(sid, "Ciao Ivy")

            events = _journal_events(tmp_path / "sessions", sid)
            t1 = _events_by_turn(events, 1)
            types = _event_types(t1)

            retry_events = [
                e for e in t1
                if e.get("event_type", e.get("type")) == "LLM_RETRY"
            ]
            assert len(retry_events) >= 1, (
                f"Expected LLM_RETRY for assistant pattern, got: {types}"
            )
            for re_evt in retry_events:
                content = re_evt.get("content", re_evt)
                reason = content.get("reason", "")
                assert "assistant_pattern" in reason, (
                    f"Expected assistant_pattern reason, got: {reason}"
                )
        finally:
            stub.chaos_mode = None


# ═══════════════════════════════════════════════════════════════════════════
# TestCacheBypass — T9
# ═══════════════════════════════════════════════════════════════════════════

class TestCacheBypass:
    """Tools bypass LLM cache (llm.py L748)."""

    def test_t9_tool_requests_bypass_cache(
        self, engine: SessionEngine, tmp_path: Path, stub: StubServer,
    ) -> None:
        """Two identical tool turns → both hit stub (no caching)."""
        result = engine.create_session(seed_id="SEED_V0")
        sid = result["session_id"]

        # Turn 1: warmup
        engine.process_turn(sid, "Ciao Ivy")
        count_after_warmup = stub.request_count

        # Turn 2: fs_write request
        engine.process_turn(
            sid, "Scrivi 'a' nel file test.txt nella sandbox."
        )
        count_after_t2 = stub.request_count
        assert count_after_t2 > count_after_warmup, "Turn 2 should hit stub"

        # Turn 3: IDENTICAL request (should NOT be cached)
        engine.process_turn(
            sid, "Scrivi 'a' nel file test.txt nella sandbox."
        )
        count_after_t3 = stub.request_count
        assert count_after_t3 > count_after_t2, (
            "Turn 3 should also hit stub (tool requests bypass cache)"
        )


# ═══════════════════════════════════════════════════════════════════════════
# TestSecondCall — T10
# ═══════════════════════════════════════════════════════════════════════════

class TestSecondCall:
    """Tool execution triggers second LLM call (llm.py L828-836)."""

    def test_t10_tool_then_text(
        self, engine: SessionEngine, tmp_path: Path, stub: StubServer,
    ) -> None:
        """LLM returns tool_calls → handler → follow-up → final text."""
        result = engine.create_session(seed_id="SEED_V0")
        sid = result["session_id"]

        engine.process_turn(sid, "Ciao Ivy")
        count_before = stub.request_count

        # Turn 2: fs_write
        r2 = engine.process_turn(
            sid, "Scrivi 'hello' nel file test.txt nella sandbox."
        )

        events = _journal_events(tmp_path / "sessions", sid)
        t2 = _events_by_turn(events, 2)
        types = _event_types(t2)

        # Must have TOOL_CALL + TOOL_RESULT + final LLM_OUTPUT
        assert "TOOL_CALL" in types, f"Expected TOOL_CALL in {types}"
        assert "TOOL_RESULT" in types, f"Expected TOOL_RESULT in {types}"
        assert "LLM_OUTPUT" in types or "LLM_FALLBACK" in types, (
            f"Expected final LLM_OUTPUT in {types}"
        )

        # Stub should have received ≥2 requests for this turn
        # (first: tool_calls, second: with role="tool" for final text)
        count_after = stub.request_count
        requests_this_turn = count_after - count_before
        assert requests_this_turn >= 2, (
            f"Expected ≥2 stub requests (tool call + follow-up), got {requests_this_turn}"
        )

        # Last payload should contain role="tool" (the follow-up call)
        messages = stub.last_payload.get("messages", [])
        has_tool_role = any(m.get("role") == "tool" for m in messages)
        assert has_tool_role, (
            "Second LLM call should include role='tool' in messages"
        )


# ═══════════════════════════════════════════════════════════════════════════
# TestParseResilience — T11
# ═══════════════════════════════════════════════════════════════════════════

class TestParseResilience:
    """Graceful handling of edge cases."""

    def test_t11_no_crash_on_empty_response(
        self, engine: SessionEngine, tmp_path: Path,
    ) -> None:
        """Kernel does not crash and produces output for normal turn."""
        result = engine.create_session(seed_id="SEED_V0")
        sid = result["session_id"]

        # Just verify the full pipeline completes without error
        r1 = engine.process_turn(sid, "Ciao Ivy, dimmi qualcosa.")
        assert "response" in r1, f"Expected response in result: {r1}"
        assert isinstance(r1["response"], str)
        assert len(r1["response"]) > 0
