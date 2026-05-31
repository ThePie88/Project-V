"""Tests for V3.1 — CLI Interattiva.

Test IDs:
- I-CLI-010: Smoke 3 turns (FakeLLM)
- I-CLI-020: Commands :save :quit
- I-CLI-030: Resume session + recall
- I-CLI-040: Validate command exit codes
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from pie.session.manager import SessionManager, SessionContext

SEED_PATH = Path(__file__).resolve().parent.parent / "progetto" / "SEED_V0.md"


@pytest.fixture
def tmp_sessions(tmp_path):
    """Provide a temporary sessions root directory."""
    return tmp_path / "sessions"


# -----------------------------------------------------------------------
# I-CLI-010 — Smoke 3 turns (FakeLLM)
# -----------------------------------------------------------------------


class TestCLISmoke:
    """Verify that run_session processes multiple turns with FakeLLM."""

    def test_three_turns_produce_output(self, tmp_sessions):
        """3 turns with FakeLLM must produce 3 outputs + journal entries."""
        from pie.runtime import run_session

        mgr = SessionManager(tmp_sessions)
        ctx = mgr.create(SEED_PATH, session_id="smoke3")

        # Provide 3 inputs via non-interactive mode (reads from _process_turn)
        inputs = ["Hello Ivy", "How are you?", "Tell me a story"]
        original_input = __builtins__.__dict__.get("input") if hasattr(__builtins__, "__dict__") else None

        call_count = [0]

        def fake_input(prompt=""):
            if call_count[0] < len(inputs):
                val = inputs[call_count[0]]
                call_count[0] += 1
                return val
            raise EOFError

        import builtins
        old_input = builtins.input
        builtins.input = fake_input
        try:
            run_session(
                session_ctx=ctx,
                turns=3,
                llm="fake",
                no_cache=True,
                interactive=False,
            )
        finally:
            builtins.input = old_input

        # Verify journal has entries for all 3 turns
        journal_path = tmp_sessions / "smoke3" / "journal.jsonl"
        lines = [
            l for l in journal_path.read_text(encoding="utf-8").splitlines()
            if l.strip()
        ]
        # Each turn produces multiple events (INPUT, STATE_UPDATED, GOALS, etc.)
        assert len(lines) >= 9  # at least 3 events per turn

        # Verify state advanced 3 turns
        state_data = json.loads(
            (tmp_sessions / "smoke3" / "state_latest.json").read_text(encoding="utf-8")
        )
        assert state_data["turn_count"] == 3

    def test_turn_events_in_journal(self, tmp_sessions):
        """Each turn must produce INPUT and LLM_OUTPUT/LLM_FALLBACK events."""
        from pie.runtime import run_session

        mgr = SessionManager(tmp_sessions)
        ctx = mgr.create(SEED_PATH, session_id="smoke_events")

        import builtins
        old_input = builtins.input
        builtins.input = lambda prompt="": "test input"
        try:
            run_session(
                session_ctx=ctx,
                turns=1,
                llm="fake",
                no_cache=True,
                interactive=False,
            )
        except EOFError:
            pass
        finally:
            builtins.input = old_input

        journal_path = tmp_sessions / "smoke_events" / "journal.jsonl"
        events = []
        for line in journal_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                events.append(json.loads(line))

        event_types = [e["type"] for e in events]
        assert "INPUT" in event_types
        assert "STATE_UPDATED" in event_types
        assert "LLM_OUTPUT" in event_types or "LLM_FALLBACK" in event_types


# -----------------------------------------------------------------------
# I-CLI-020 — Commands :save :quit
# -----------------------------------------------------------------------


class TestCLICommands:
    """Verify interactive commands work correctly."""

    def test_save_persists_state(self, tmp_sessions):
        """`:save` command must persist state changes to disk."""
        from pie.runtime import run_session

        mgr = SessionManager(tmp_sessions)
        ctx = mgr.create(SEED_PATH, session_id="cmd_save")

        command_seq = ["Hello", ":save", ":quit"]
        idx = [0]

        import builtins
        old_input = builtins.input
        def fake_input(prompt=""):
            if idx[0] < len(command_seq):
                val = command_seq[idx[0]]
                idx[0] += 1
                return val
            raise EOFError

        builtins.input = fake_input
        try:
            run_session(
                session_ctx=ctx,
                turns=1,
                llm="fake",
                no_cache=True,
                interactive=True,
            )
        finally:
            builtins.input = old_input

        # State must have been saved with turn_count=1
        state_data = json.loads(
            (tmp_sessions / "cmd_save" / "state_latest.json").read_text(encoding="utf-8")
        )
        assert state_data["turn_count"] == 1

    def test_quit_saves_and_exits(self, tmp_sessions, capsys):
        """`:quit` must save session and exit cleanly."""
        from pie.runtime import run_session

        mgr = SessionManager(tmp_sessions)
        ctx = mgr.create(SEED_PATH, session_id="cmd_quit")

        command_seq = [":quit"]
        idx = [0]

        import builtins
        old_input = builtins.input
        def fake_input(prompt=""):
            if idx[0] < len(command_seq):
                val = command_seq[idx[0]]
                idx[0] += 1
                return val
            raise EOFError

        builtins.input = fake_input
        try:
            run_session(
                session_ctx=ctx,
                turns=1,
                llm="fake",
                no_cache=True,
                interactive=True,
            )
        finally:
            builtins.input = old_input

        captured = capsys.readouterr()
        assert "Session saved. Goodbye." in captured.out

    def test_state_command_shows_drives(self, tmp_sessions, capsys):
        """`:state` must display current state as JSON."""
        from pie.runtime import run_session

        mgr = SessionManager(tmp_sessions)
        ctx = mgr.create(SEED_PATH, session_id="cmd_state")

        command_seq = [":state", ":quit"]
        idx = [0]

        import builtins
        old_input = builtins.input
        def fake_input(prompt=""):
            if idx[0] < len(command_seq):
                val = command_seq[idx[0]]
                idx[0] += 1
                return val
            raise EOFError

        builtins.input = fake_input
        try:
            run_session(
                session_ctx=ctx,
                turns=1,
                llm="fake",
                no_cache=True,
                interactive=True,
            )
        finally:
            builtins.input = old_input

        captured = capsys.readouterr()
        assert "curiosity" in captured.out
        assert "0.78" in captured.out

    def test_whoami_shows_identity(self, tmp_sessions, capsys):
        """`:whoami` must display name, alias, creator."""
        from pie.runtime import run_session

        mgr = SessionManager(tmp_sessions)
        ctx = mgr.create(SEED_PATH, session_id="cmd_whoami")

        command_seq = [":whoami", ":quit"]
        idx = [0]

        import builtins
        old_input = builtins.input
        def fake_input(prompt=""):
            if idx[0] < len(command_seq):
                val = command_seq[idx[0]]
                idx[0] += 1
                return val
            raise EOFError

        builtins.input = fake_input
        try:
            run_session(
                session_ctx=ctx,
                turns=1,
                llm="fake",
                no_cache=True,
                interactive=True,
            )
        finally:
            builtins.input = old_input

        captured = capsys.readouterr()
        assert "Ivy" in captured.out
        assert "MrPie" in captured.out

    def test_tools_on_off(self, tmp_sessions, capsys):
        """`:tools on/off` must toggle and print feedback."""
        from pie.runtime import run_session

        mgr = SessionManager(tmp_sessions)
        ctx = mgr.create(SEED_PATH, session_id="cmd_tools")

        command_seq = [":tools off", ":tools on", ":quit"]
        idx = [0]

        import builtins
        old_input = builtins.input
        def fake_input(prompt=""):
            if idx[0] < len(command_seq):
                val = command_seq[idx[0]]
                idx[0] += 1
                return val
            raise EOFError

        builtins.input = fake_input
        try:
            run_session(
                session_ctx=ctx,
                turns=1,
                llm="fake",
                no_cache=True,
                interactive=True,
            )
        finally:
            builtins.input = old_input

        captured = capsys.readouterr()
        assert "Tools disabled." in captured.out
        assert "Tools enabled." in captured.out

    def test_unknown_command_feedback(self, tmp_sessions, capsys):
        """Unknown `:xyz` command must print help, not crash."""
        from pie.runtime import run_session

        mgr = SessionManager(tmp_sessions)
        ctx = mgr.create(SEED_PATH, session_id="cmd_unknown")

        command_seq = [":foobar", ":quit"]
        idx = [0]

        import builtins
        old_input = builtins.input
        def fake_input(prompt=""):
            if idx[0] < len(command_seq):
                val = command_seq[idx[0]]
                idx[0] += 1
                return val
            raise EOFError

        builtins.input = fake_input
        try:
            run_session(
                session_ctx=ctx,
                turns=1,
                llm="fake",
                no_cache=True,
                interactive=True,
            )
        finally:
            builtins.input = old_input

        captured = capsys.readouterr()
        assert "Unknown command: :foobar" in captured.out
        assert ":quit" in captured.out  # help text shown


# -----------------------------------------------------------------------
# I-CLI-030 — Resume session + recall
# -----------------------------------------------------------------------


class TestCLIResume:
    """Verify that resuming a session restores state and memory."""

    def test_resume_continues_turn_count(self, tmp_sessions):
        """Session 2 must continue from session 1 turn count."""
        from pie.runtime import run_session

        mgr = SessionManager(tmp_sessions)
        ctx = mgr.create(SEED_PATH, session_id="resume_tc")

        # Session 1: 2 turns
        inputs = ["Hello", "World"]
        idx = [0]

        import builtins
        old_input = builtins.input
        def fake_input(prompt=""):
            if idx[0] < len(inputs):
                val = inputs[idx[0]]
                idx[0] += 1
                return val
            raise EOFError

        builtins.input = fake_input
        try:
            run_session(
                session_ctx=ctx,
                turns=2,
                llm="fake",
                no_cache=True,
                interactive=False,
            )
        finally:
            builtins.input = old_input

        # Session 2: resume and do 1 more turn
        ctx2 = mgr.resume("resume_tc")
        assert ctx2.turn_count == 2

        idx[0] = 0
        inputs_2 = ["Resumed"]
        def fake_input2(prompt=""):
            if idx[0] < len(inputs_2):
                val = inputs_2[idx[0]]
                idx[0] += 1
                return val
            raise EOFError

        builtins.input = fake_input2
        try:
            run_session(
                session_ctx=ctx2,
                turns=1,
                llm="fake",
                no_cache=True,
                interactive=False,
            )
        finally:
            builtins.input = old_input

        state_data = json.loads(
            (tmp_sessions / "resume_tc" / "state_latest.json").read_text(encoding="utf-8")
        )
        assert state_data["turn_count"] == 3

    def test_resume_identity_and_state_persist(self, tmp_sessions):
        """Identity and state must be restored after resume."""
        from pie.runtime import run_session

        mgr = SessionManager(tmp_sessions)
        ctx = mgr.create(SEED_PATH, session_id="resume_mem")

        # Session 1: do a turn
        import builtins
        old_input = builtins.input
        builtins.input = lambda prompt="": "Hello Ivy"
        try:
            run_session(
                session_ctx=ctx,
                turns=1,
                llm="fake",
                no_cache=True,
                interactive=False,
            )
        except EOFError:
            pass
        finally:
            builtins.input = old_input

        # Resume and verify identity + state restored
        ctx2 = mgr.resume("resume_mem")
        assert ctx2.identity["identity"]["name"] == "Ivy"
        assert ctx2.state.turn_count == 1
        # Journal must have events from session 1
        journal = (tmp_sessions / "resume_mem" / "journal.jsonl").read_text(encoding="utf-8")
        assert "INPUT" in journal


# -----------------------------------------------------------------------
# I-CLI-040 — Validate command exit codes
# -----------------------------------------------------------------------


class TestValidateCommand:
    """Verify that `pie.cli validate` uses correct exit codes."""

    def test_validate_valid_state_exits_zero(self, tmp_path):
        """Validating a valid state JSON must exit with code 0."""
        from pie.contracts.state import State

        state = State()
        state_file = tmp_path / "good_state.json"
        state_file.write_text(
            json.dumps(state.snapshot(), indent=2),
            encoding="utf-8",
        )

        result = subprocess.run(
            [sys.executable, "-m", "pie.cli", "validate", str(state_file)],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).resolve().parent.parent),
        )
        assert result.returncode == 0
        assert "succeeded" in result.stdout.lower()

    def test_validate_invalid_exits_nonzero(self, tmp_path):
        """Validating a corrupted file must exit with code 1."""
        bad_file = tmp_path / "bad.json"
        bad_file.write_text('{"not": "a valid contract"}', encoding="utf-8")

        result = subprocess.run(
            [sys.executable, "-m", "pie.cli", "validate", str(bad_file)],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).resolve().parent.parent),
        )
        assert result.returncode == 1
        assert "failed" in result.stdout.lower()

    def test_validate_missing_file_exits_nonzero(self, tmp_path):
        """Validating a non-existent file must exit with code 1."""
        result = subprocess.run(
            [sys.executable, "-m", "pie.cli", "validate", str(tmp_path / "nope.json")],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).resolve().parent.parent),
        )
        assert result.returncode == 1
        assert "failed" in result.stdout.lower()
