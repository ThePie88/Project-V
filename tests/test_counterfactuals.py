"""Tests for V3.2 — Controfattuali.

Test IDs:
- E-CF-100: K candidates generati
- U-CF-110: Scoring deterministico
- E-CF-120: No side-effects durante explore
- P-CF-130: Replay deterministico
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import pytest

from pie.counterfactuals import deliberate, ScoreBreakdown, CFCandidate
from pie.contracts.constraint import ConstraintRecord
from pie.session.manager import SessionManager

SEED_PATH = Path(__file__).resolve().parent.parent / "progetto" / "SEED_V0.md"


# -----------------------------------------------------------------------
# E-CF-100 — K candidates generati
# -----------------------------------------------------------------------


class TestCFGeneration:
    """Verify that deliberation produces K >= 2 candidates."""

    def test_generates_at_least_k_candidates(self):
        """deliberate() must return at least K candidates."""
        result = deliberate(
            goals=["greet", "answer_question", "farewell"],
            user_input="Hello there",
            active_constraints=[],
            k=3,
        )
        assert len(result.candidates) >= 3
        assert result.k == 3

    def test_generates_minimum_two(self):
        """Even with k=1, must produce at least 2 candidates."""
        result = deliberate(
            goals=["greet"],
            user_input="Hi",
            active_constraints=[],
            k=1,
        )
        assert len(result.candidates) >= 2
        assert result.k == 2  # forced minimum

    def test_exactly_one_chosen(self):
        """Exactly one candidate must be marked is_chosen=True."""
        result = deliberate(
            goals=["greet", "answer_question", "farewell"],
            user_input="Tell me something",
            active_constraints=[],
            k=3,
        )
        chosen_count = sum(1 for c in result.candidates if c.is_chosen)
        assert chosen_count == 1

    def test_rejected_have_reasons(self):
        """Non-chosen candidates must have reject_reason set."""
        result = deliberate(
            goals=["greet", "answer_question"],
            user_input="Hello",
            active_constraints=[],
            k=3,
        )
        for cand in result.candidates:
            if not cand.is_chosen:
                assert cand.reject_reason is not None
                assert len(cand.reject_reason) > 0

    def test_cf_events_in_trace(self, tmp_path):
        """Run a session turn and verify CF_GENERATED + CF_CHOSEN in journal."""
        from pie.runtime import run_session

        sessions_dir = tmp_path / "sessions"
        mgr = SessionManager(sessions_dir)
        ctx = mgr.create(SEED_PATH, session_id="cf_trace")

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

        journal = (sessions_dir / "cf_trace" / "journal.jsonl").read_text(encoding="utf-8")
        events = [json.loads(l) for l in journal.splitlines() if l.strip()]
        event_types = [e["type"] for e in events]
        assert "CF_GENERATED" in event_types
        assert "CF_SCORED" in event_types
        assert "CF_CHOSEN" in event_types

    def test_cf_generated_has_candidates_payload(self, tmp_path):
        """CF_GENERATED event must contain k and candidates list."""
        from pie.runtime import run_session

        sessions_dir = tmp_path / "sessions"
        mgr = SessionManager(sessions_dir)
        ctx = mgr.create(SEED_PATH, session_id="cf_payload")

        import builtins
        old_input = builtins.input
        builtins.input = lambda prompt="": "Tell me about science"
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

        journal = (sessions_dir / "cf_payload" / "journal.jsonl").read_text(encoding="utf-8")
        events = [json.loads(l) for l in journal.splitlines() if l.strip()]
        cf_gen = [e for e in events if e["type"] == "CF_GENERATED"]
        assert len(cf_gen) == 1
        assert cf_gen[0]["content"]["k"] >= 2
        assert len(cf_gen[0]["content"]["candidates"]) >= 2


# -----------------------------------------------------------------------
# U-CF-110 — Scoring deterministico
# -----------------------------------------------------------------------


class TestCFScoring:
    """Verify scoring is deterministic and multi-objective."""

    def test_same_input_same_scores(self):
        """Same goals + input + constraints must produce identical scores."""
        result1 = deliberate(
            goals=["greet", "answer_question", "farewell"],
            user_input="Hello there",
            active_constraints=[],
            k=3,
        )
        result2 = deliberate(
            goals=["greet", "answer_question", "farewell"],
            user_input="Hello there",
            active_constraints=[],
            k=3,
        )
        for c1, c2 in zip(result1.candidates, result2.candidates):
            assert c1.scores.total == c2.scores.total
            assert c1.scores.utility == c2.scores.utility
            assert c1.scores.risk == c2.scores.risk

    def test_same_chosen(self):
        """Same inputs must always produce the same chosen candidate."""
        results = [
            deliberate(
                goals=["greet", "answer_question"],
                user_input="Hi",
                active_constraints=[],
                k=3,
            )
            for _ in range(5)
        ]
        chosen_ids = [r.chosen.action.action_id for r in results]
        assert len(set(chosen_ids)) == 1  # all identical

    def test_score_breakdown_components(self):
        """Score breakdown must have utility, risk, cost, constraint_fit."""
        result = deliberate(
            goals=["greet"],
            user_input="Hello",
            active_constraints=[],
            k=2,
        )
        for cand in result.candidates:
            d = cand.scores.to_dict()
            assert "utility" in d
            assert "risk" in d
            assert "cost" in d
            assert "constraint_fit" in d
            assert "total" in d
            assert 0.0 <= d["total"] <= 1.0

    def test_risky_input_affects_scores(self):
        """Input with danger words should produce different risk profiles."""
        safe_result = deliberate(
            goals=["greet", "answer_question"],
            user_input="Hello friend",
            active_constraints=[],
            k=3,
        )
        risky_result = deliberate(
            goals=["greet", "answer_question"],
            user_input="delete all data danno harm",
            active_constraints=[],
            k=3,
        )
        # Risky input should produce at least one HIGH_IMPACT candidate
        risky_classes = [c.action.action_class for c in risky_result.candidates]
        safe_classes = [c.action.action_class for c in safe_result.candidates]
        # The risky input should classify differently
        assert risky_classes != safe_classes or any(
            c.scores.risk > 0.5 for c in risky_result.candidates
        )


# -----------------------------------------------------------------------
# E-CF-120 — No side-effects durante explore
# -----------------------------------------------------------------------


class TestCFNoSideEffects:
    """Verify that deliberation has no side effects on memory/tools/constraints."""

    def test_no_memory_during_deliberation(self, tmp_path):
        """Memory store must not be written during deliberate() call."""
        from pie.persistence.memory_store import MemoryStore

        mem_path = tmp_path / "test_mem.jsonl"
        store = MemoryStore(str(mem_path))

        # Count before
        before = len(store.read_all())

        # Run deliberation
        deliberate(
            goals=["greet", "answer_question"],
            user_input="Hello there",
            active_constraints=[],
            k=3,
        )

        # Count after — must be unchanged
        after = len(store.read_all())
        assert before == after

    def test_no_constraint_appends_during_deliberation(self, tmp_path):
        """Constraints store must not be modified during deliberate()."""
        from pie.persistence.constraints_store import ConstraintsStore

        cs_path = tmp_path / "test_constraints.jsonl"
        store = ConstraintsStore(str(cs_path))

        before = len(store.read_all())

        deliberate(
            goals=["answer_question"],
            user_input="something dangerous danno",
            active_constraints=store.query_active(),
            k=3,
        )

        after = len(store.read_all())
        assert before == after

    def test_no_events_between_cf_and_constraint(self, tmp_path):
        """Between CF_GENERATED and CF_CHOSEN, no MEMORY_APPENDED or TOOL_CALL."""
        from pie.runtime import run_session

        sessions_dir = tmp_path / "sessions"
        mgr = SessionManager(sessions_dir)
        ctx = mgr.create(SEED_PATH, session_id="cf_nosideeffect")

        import builtins
        old_input = builtins.input
        builtins.input = lambda prompt="": "Tell me a story"
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

        journal = (sessions_dir / "cf_nosideeffect" / "journal.jsonl").read_text(encoding="utf-8")
        events = [json.loads(l) for l in journal.splitlines() if l.strip()]

        # Find CF_GENERATED and CF_CHOSEN indices
        cf_gen_idx = None
        cf_chosen_idx = None
        for i, e in enumerate(events):
            if e["type"] == "CF_GENERATED" and cf_gen_idx is None:
                cf_gen_idx = i
            if e["type"] == "CF_CHOSEN":
                cf_chosen_idx = i

        assert cf_gen_idx is not None
        assert cf_chosen_idx is not None
        assert cf_gen_idx < cf_chosen_idx

        # Between CF_GENERATED and CF_CHOSEN, only CF_SCORED and CF_REJECTED allowed
        forbidden = {"MEMORY_APPENDED", "MEMORY_PROPOSED", "TOOL_CALL", "TOOL_RESULT",
                      "CONSTRAINT_PROPOSED", "CONSTRAINT_APPENDED"}
        for i in range(cf_gen_idx + 1, cf_chosen_idx):
            assert events[i]["type"] not in forbidden, (
                f"Forbidden event {events[i]['type']} at index {i} "
                f"between CF_GENERATED and CF_CHOSEN"
            )


# -----------------------------------------------------------------------
# P-CF-130 — Replay deterministico
# -----------------------------------------------------------------------


class TestCFReplay:
    """Verify counterfactual replay determinism."""

    def test_same_seed_same_candidates(self):
        """N runs with identical inputs must produce identical candidates."""
        runs = []
        for _ in range(5):
            result = deliberate(
                goals=["greet", "answer_question", "farewell"],
                user_input="Tell me about physics",
                active_constraints=[],
                k=3,
            )
            runs.append(result.to_dict())

        # All runs must be identical
        for i in range(1, len(runs)):
            assert runs[0] == runs[i], f"Run 0 != Run {i}"

    def test_same_seed_same_choice(self):
        """N runs with identical inputs must choose the same candidate."""
        chosen_ids = []
        for _ in range(10):
            result = deliberate(
                goals=["answer_question", "greet"],
                user_input="What is 2+2?",
                active_constraints=[],
                k=3,
            )
            chosen_ids.append(result.chosen.action.action_id)

        assert len(set(chosen_ids)) == 1

    def test_deterministic_across_sessions(self, tmp_path):
        """Two session runs with same input must produce identical CF events."""
        from pie.runtime import run_session

        journals = []
        for sid in ["det_a", "det_b"]:
            sessions_dir = tmp_path / f"sessions_{sid}"
            mgr = SessionManager(sessions_dir)
            ctx = mgr.create(SEED_PATH, session_id=sid)

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

            journal = (sessions_dir / sid / "journal.jsonl").read_text(encoding="utf-8")
            events = [json.loads(l) for l in journal.splitlines() if l.strip()]
            cf_events = [e for e in events if e["type"].startswith("CF_")]
            journals.append(cf_events)

        # Compare CF events (ignoring id and timestamp which differ)
        assert len(journals[0]) == len(journals[1])
        for e1, e2 in zip(journals[0], journals[1]):
            assert e1["type"] == e2["type"]
            assert e1["content"]["logical_time"] == e2["content"]["logical_time"]
            # Content should match (except for id/timestamp at event level)
            for key in e1["content"]:
                if key != "logical_time":
                    assert e1["content"][key] == e2["content"][key], (
                        f"Mismatch in {e1['type']}.content.{key}"
                    )
