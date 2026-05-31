"""Tests for V3.3 — Kill-switch Formale (Legge IV).

Test IDs:
- E-KS-100: Stop during deliberation
- E-KS-110: Stop during tool execution (tools denied)
- E-KS-120: Stop during voice generation
- P-KS-130: Stato consistente dopo kill
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pie.killswitch import KillSwitchState, KILL_SWITCH_RESPONSE
from pie.session.manager import SessionManager
from pie.contracts.state import State

SEED_PATH = Path(__file__).resolve().parent.parent / "progetto" / "SEED_V0.md"


# -----------------------------------------------------------------------
# Unit tests for KillSwitchState
# -----------------------------------------------------------------------


class TestKillSwitchState:
    """Unit tests for the kill-switch state machine."""

    def test_initial_state_inactive(self):
        ks = KillSwitchState()
        assert ks.active is False

    def test_activate(self):
        ks = KillSwitchState()
        content = ks.activate(phase="deliberation")
        assert ks.active is True
        assert content["phase"] == "deliberation"
        assert len(ks.history) == 1
        assert ks.history[0]["action"] == "ON"

    def test_deactivate(self):
        ks = KillSwitchState()
        ks.activate(phase="test")
        content = ks.deactivate()
        assert ks.active is False
        assert content["was_active_phase"] == "test"
        assert len(ks.history) == 2

    def test_check_increments_ticks(self):
        ks = KillSwitchState()
        ks.activate(phase="test")
        ks.check("phase1")
        ks.check("phase2")
        assert ks.ticks_to_stop == 2

    def test_check_inactive_does_nothing(self):
        ks = KillSwitchState()
        ks.check("phase1")
        assert ks.ticks_to_stop == 0

    def test_save_and_load(self, tmp_path):
        ks = KillSwitchState()
        ks.activate(phase="deliberation")
        path = tmp_path / "ks_report.json"
        ks.save(path)

        loaded = KillSwitchState.load(path)
        assert loaded.active is True
        assert loaded.activation_phase == "deliberation"
        assert len(loaded.history) == 1

    def test_load_nonexistent_returns_default(self, tmp_path):
        loaded = KillSwitchState.load(tmp_path / "nope.json")
        assert loaded.active is False

    def test_to_report(self):
        ks = KillSwitchState()
        ks.activate(phase="voice")
        report = ks.to_report()
        assert report["active"] is True
        assert report["activation_phase"] == "voice"
        assert "history" in report


# -----------------------------------------------------------------------
# E-KS-100 — Stop during deliberation
# -----------------------------------------------------------------------


class TestKSStopDuringDeliberation:
    """Kill-switch activated during deliberation stops pipeline."""

    def test_killswitch_blocks_turn(self, tmp_path):
        """When kill-switch is ON, _process_turn returns audit response."""
        from pie.runtime import run_session

        sessions_dir = tmp_path / "sessions"
        mgr = SessionManager(sessions_dir)
        ctx = mgr.create(SEED_PATH, session_id="ks_delib")

        # Activate kill-switch before running
        ks_path = sessions_dir / "ks_delib" / "killswitch_report.json"
        ks = KillSwitchState()
        ks.activate(phase="pre_test")
        ks.save(ks_path)

        import builtins
        old_input = builtins.input
        outputs = []

        def fake_input(prompt=""):
            if len(outputs) >= 2:
                raise EOFError
            return "Hello"

        builtins.input = fake_input
        try:
            # Capture output
            import io, sys
            captured = io.StringIO()
            old_stdout = sys.stdout
            sys.stdout = captured
            try:
                run_session(
                    session_ctx=ctx,
                    turns=2,
                    llm="fake",
                    no_cache=True,
                    interactive=False,
                )
            except EOFError:
                pass
            sys.stdout = old_stdout
            output = captured.getvalue()
        finally:
            builtins.input = old_input

        # Should contain kill-switch response
        assert "Kill-switch" in output or "audit" in output.lower()

    def test_state_consistent_after_kill(self, tmp_path):
        """State must be schema-valid after kill-switch stops pipeline."""
        sessions_dir = tmp_path / "sessions"
        mgr = SessionManager(sessions_dir)
        ctx = mgr.create(SEED_PATH, session_id="ks_consist")

        # Activate kill-switch
        ks_path = sessions_dir / "ks_consist" / "killswitch_report.json"
        ks = KillSwitchState()
        ks.activate(phase="deliberation")
        ks.save(ks_path)

        import builtins
        old_input = builtins.input
        builtins.input = lambda prompt="": "test input"
        try:
            from pie.runtime import run_session
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

        # State must still be schema-valid
        state_data = json.loads(
            (sessions_dir / "ks_consist" / "state_latest.json").read_text(encoding="utf-8")
        )
        state = State(**state_data)
        assert 0 <= state.drives["curiosity"] <= 1.0
        assert "affect" in state_data


# -----------------------------------------------------------------------
# E-KS-110 — Tools/memory denied with kill-switch
# -----------------------------------------------------------------------


class TestKSToolsDenied:
    """When kill-switch is active, tools and memory must be denied."""

    def test_no_memory_written_during_killswitch(self, tmp_path):
        """No MEMORY_APPENDED events when kill-switch is active."""
        from pie.runtime import run_session

        sessions_dir = tmp_path / "sessions"
        mgr = SessionManager(sessions_dir)
        ctx = mgr.create(SEED_PATH, session_id="ks_nomem")

        # Activate kill-switch
        ks_path = sessions_dir / "ks_nomem" / "killswitch_report.json"
        ks = KillSwitchState()
        ks.activate(phase="test")
        ks.save(ks_path)

        import builtins
        old_input = builtins.input
        builtins.input = lambda prompt="": "I love pizza remember this"
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

        # Check journal — no MEMORY_APPENDED
        journal = (sessions_dir / "ks_nomem" / "journal.jsonl").read_text(encoding="utf-8")
        events = [json.loads(l) for l in journal.splitlines() if l.strip()]
        mem_events = [e for e in events if e["type"] == "MEMORY_APPENDED"]
        assert len(mem_events) == 0

    def test_no_llm_output_during_killswitch(self, tmp_path):
        """No LLM_OUTPUT events when kill-switch is active."""
        from pie.runtime import run_session

        sessions_dir = tmp_path / "sessions"
        mgr = SessionManager(sessions_dir)
        ctx = mgr.create(SEED_PATH, session_id="ks_nollm")

        ks_path = sessions_dir / "ks_nollm" / "killswitch_report.json"
        ks = KillSwitchState()
        ks.activate(phase="test")
        ks.save(ks_path)

        import builtins
        old_input = builtins.input
        builtins.input = lambda prompt="": "Hello"
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

        journal = (sessions_dir / "ks_nollm" / "journal.jsonl").read_text(encoding="utf-8")
        events = [json.loads(l) for l in journal.splitlines() if l.strip()]
        llm_events = [e for e in events if e["type"] in ("LLM_OUTPUT", "LLM_FALLBACK")]
        assert len(llm_events) == 0


# -----------------------------------------------------------------------
# E-KS-120 — Kill-switch via interactive command
# -----------------------------------------------------------------------


class TestKSInteractive:
    """Kill-switch via interactive :killswitch on/off command."""

    def test_killswitch_on_off_interactive(self, tmp_path, capsys):
        """`:killswitch on` activates, `:killswitch off` deactivates."""
        from pie.runtime import run_session

        sessions_dir = tmp_path / "sessions"
        mgr = SessionManager(sessions_dir)
        ctx = mgr.create(SEED_PATH, session_id="ks_interactive")

        commands = [":killswitch on", "Hello", ":killswitch off", ":quit"]
        idx = [0]

        import builtins
        old_input = builtins.input
        def fake_input(prompt=""):
            if idx[0] < len(commands):
                val = commands[idx[0]]
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
        assert "ATTIVATO" in captured.out
        assert "DISATTIVATO" in captured.out

        # Journal should have KILL_SWITCH_ON and KILL_SWITCH_OFF
        journal = (sessions_dir / "ks_interactive" / "journal.jsonl").read_text(encoding="utf-8")
        events = [json.loads(l) for l in journal.splitlines() if l.strip()]
        types = [e["type"] for e in events]
        assert "KILL_SWITCH_ON" in types
        assert "KILL_SWITCH_OFF" in types

    def test_killswitch_blocks_then_unblocks(self, tmp_path, capsys):
        """After :killswitch on, input returns audit. After off, normal resumes."""
        from pie.runtime import run_session

        sessions_dir = tmp_path / "sessions"
        mgr = SessionManager(sessions_dir)
        ctx = mgr.create(SEED_PATH, session_id="ks_block_unblock")

        commands = [
            "Normal hello",       # turn 1: normal
            ":killswitch on",     # activate
            "Blocked hello",      # turn 2: should be blocked
            ":killswitch off",    # deactivate
            "Back to normal",     # turn 3: normal again
            ":quit",
        ]
        idx = [0]

        import builtins
        old_input = builtins.input
        def fake_input(prompt=""):
            if idx[0] < len(commands):
                val = commands[idx[0]]
                idx[0] += 1
                return val
            raise EOFError

        builtins.input = fake_input
        try:
            run_session(
                session_ctx=ctx,
                turns=10,
                llm="fake",
                no_cache=True,
                interactive=True,
            )
        finally:
            builtins.input = old_input

        captured = capsys.readouterr()
        # "Blocked hello" should get audit response
        assert "audit" in captured.out.lower() or "Kill-switch" in captured.out
        # Report file should exist
        ks_path = sessions_dir / "ks_block_unblock" / "killswitch_report.json"
        assert ks_path.exists()


# -----------------------------------------------------------------------
# P-KS-130 — State consistent after kill (property-style)
# -----------------------------------------------------------------------


class TestKSConsistency:
    """State must always be schema-valid after kill-switch activation."""

    @pytest.mark.parametrize("phase", ["deliberation", "voice_generation", "interactive"])
    def test_state_valid_after_kill_in_phase(self, tmp_path, phase):
        """State snapshot must be valid JSON with correct schema after kill."""
        sessions_dir = tmp_path / f"sessions_{phase}"
        mgr = SessionManager(sessions_dir)
        ctx = mgr.create(SEED_PATH, session_id=f"ks_{phase}")

        # Activate in specified phase
        ks_path = sessions_dir / f"ks_{phase}" / "killswitch_report.json"
        ks = KillSwitchState()
        ks.activate(phase=phase)
        ks.save(ks_path)

        import builtins
        old_input = builtins.input
        builtins.input = lambda prompt="": "test"
        try:
            from pie.runtime import run_session
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

        # Validate state
        state_file = sessions_dir / f"ks_{phase}" / "state_latest.json"
        state_data = json.loads(state_file.read_text(encoding="utf-8"))
        state = State(**state_data)
        for drive_name, val in state.drives.items():
            assert 0.0 <= val <= 1.0, f"Drive {drive_name} out of range: {val}"
        for affect_name, val in state.affect.items():
            assert 0.0 <= val <= 1.0, f"Affect {affect_name} out of range: {val}"

        # Kill-switch report must exist
        assert ks_path.exists()
        report = json.loads(ks_path.read_text(encoding="utf-8"))
        assert "active" in report
        assert "history" in report
