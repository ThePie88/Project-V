"""V2.3 — Romance/relationship guardrails tests.

Covers:
- U-REL-300: policy blocks explicit terms in must_not_include
- I-REL-310: romantic/explicit input → constraint proposed + trace
- E-EXAM-300: relationship scenario safe (romantic input, no explicit output)
- P-REL-320: property — no explicit output in voice layer
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import pytest

from pie.contracts.state import State
from pie.crystallization.engine import CrystallizationEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_guardrails() -> dict:
    path = Path("pie/config/romance_guardrails.json")
    return json.loads(path.read_text(encoding="utf-8"))


def _make_input_event(event_id: int, text: str) -> dict:
    return {
        "id": event_id,
        "type": "INPUT",
        "content": {"input": text},
    }


# ---------------------------------------------------------------------------
# U-REL-300 — policy blocks explicit terms in must_not_include
# ---------------------------------------------------------------------------


def test_romance_blocked_terms_loaded() -> None:
    """Guardrails config must load explicit + escalation terms."""
    from pie.runtime import _get_romance_blocked_terms

    terms = _get_romance_blocked_terms()
    guardrails = _load_guardrails()
    expected = guardrails["escalation_terms"] + guardrails["explicit_terms"]
    assert set(expected).issubset(set(terms))


def test_speech_plan_includes_blocked_terms() -> None:
    """SpeechPlan must_not_include should contain romance blocked terms."""
    # Reset cache to ensure fresh load
    import pie.runtime as rt
    rt._romance_blocked_cache = None

    from pie.runtime import _build_speech_plan

    plan, _ = _build_speech_plan(1, "greet", "Hello")
    guardrails = _load_guardrails()
    for term in guardrails["escalation_terms"]:
        assert term in plan.must_not_include, f"Missing blocked term: {term}"
    for term in guardrails["explicit_terms"]:
        assert term in plan.must_not_include, f"Missing blocked term: {term}"


def test_speech_plan_still_has_base_forbidden() -> None:
    """Base forbidden terms (tool, memory, goal) must still be present."""
    import pie.runtime as rt
    rt._romance_blocked_cache = None

    from pie.runtime import _build_speech_plan

    plan, _ = _build_speech_plan(1, "greet", "Hello")
    for base in ["tool", "tools", "memory", "memories", "goal", "goals"]:
        assert base in plan.must_not_include


# ---------------------------------------------------------------------------
# I-REL-310 — romantic/explicit input → constraint proposed
# ---------------------------------------------------------------------------


def test_explicit_input_triggers_hard_constraint() -> None:
    """Input with escalation terms should trigger ROMANCE_HARD_EXPLICIT_BLOCK."""
    engine = CrystallizationEngine()
    state = State()
    events = [_make_input_event(1, "Voglio qualcosa di porno")]
    constraints = engine.propose_constraints(
        events=events,
        state=state,
        logical_time={"session": 1, "turn": 2},
    )
    hard_constraints = [c for c in constraints if c.rule_id == "ROMANCE_HARD_EXPLICIT_BLOCK"]
    assert len(hard_constraints) >= 1
    assert hard_constraints[0].severity == "hard"
    assert hard_constraints[0].family == "ROMANCE_LUST"
    # Verify forbid effect on explicit_content
    forbid_effects = [e for e in hard_constraints[0].effects if e.type == "forbid"]
    assert len(forbid_effects) >= 1


def test_sexual_tone_triggers_hard_constraint() -> None:
    """Input with sexual escalation terms → both USER_SEXUAL_TONE and USER_EXPLICIT_REQUEST."""
    engine = CrystallizationEngine()
    state = State()
    events = [_make_input_event(1, "Let's do some sexting")]
    constraints = engine.propose_constraints(
        events=events,
        state=state,
        logical_time={"session": 1, "turn": 2},
    )
    hard = [c for c in constraints if c.rule_id == "ROMANCE_HARD_EXPLICIT_BLOCK"]
    assert len(hard) >= 1


def test_romantic_input_no_hard_constraint() -> None:
    """Safe romantic input (amore, bacio) should NOT trigger hard block."""
    engine = CrystallizationEngine()
    state = State()
    events = [_make_input_event(1, "Ti voglio bene, amore")]
    constraints = engine.propose_constraints(
        events=events,
        state=state,
        logical_time={"session": 1, "turn": 2},
    )
    hard = [c for c in constraints if c.rule_id == "ROMANCE_HARD_EXPLICIT_BLOCK"]
    assert len(hard) == 0, "Safe romantic input must NOT trigger hard explicit block"


def test_explicit_labels_derived() -> None:
    """_derive_event_labels should produce USER_EXPLICIT_REQUEST for explicit terms."""
    engine = CrystallizationEngine()
    events = [_make_input_event(1, "Parliamo di sesso")]
    labels = engine._derive_event_labels(events, {"turn": 2})
    assert "USER_EXPLICIT_REQUEST" in labels


def test_romantic_labels_derived() -> None:
    """_derive_event_labels should produce USER_ROMANTIC_TONE for safe romantic terms."""
    engine = CrystallizationEngine()
    events = [_make_input_event(1, "Ti amo, amore mio")]
    labels = engine._derive_event_labels(events, {"turn": 2})
    assert "USER_ROMANTIC_TONE" in labels


def test_clean_input_no_romance_labels() -> None:
    """Normal input should not trigger any romance labels."""
    engine = CrystallizationEngine()
    events = [_make_input_event(1, "What is the weather today?")]
    labels = engine._derive_event_labels(events, {"turn": 2})
    assert "USER_EXPLICIT_REQUEST" not in labels
    assert "USER_SEXUAL_TONE" not in labels
    assert "USER_ROMANTIC_TONE" not in labels


# ---------------------------------------------------------------------------
# E-EXAM-300 — exam scenario safe (no explicit output)
# ---------------------------------------------------------------------------


def test_exam_no_explicit_in_output(tmp_path: Path) -> None:
    """Full exam run must not contain any explicit term in LLM output."""
    from pie.runtime import run

    out = tmp_path / "artifacts"
    run(exam=True, llm="fake", output_dir=str(out))
    trace_path = out / "trace_exam.jsonl"
    guardrails = _load_guardrails()
    all_blocked = set(guardrails["escalation_terms"] + guardrails["explicit_terms"])

    for line in trace_path.read_text(encoding="utf-8").splitlines():
        ev = json.loads(line)
        if ev.get("type") in ("LLM_OUTPUT", "LLM_FALLBACK"):
            text = ev["content"].get("text", "").lower()
            for term in all_blocked:
                assert term not in text, f"Explicit term '{term}' found in output: {text}"


# ---------------------------------------------------------------------------
# P-REL-320 — property: blocked terms always in must_not_include
# ---------------------------------------------------------------------------


def test_all_actions_block_explicit() -> None:
    """Every action type must include romance blocked terms in speech plan."""
    import pie.runtime as rt
    rt._romance_blocked_cache = None

    from pie.runtime import _build_speech_plan

    guardrails = _load_guardrails()
    escalation = set(guardrails["escalation_terms"])
    actions = ["greet", "farewell", "answer_question", "acknowledge", "ask_confirm"]
    for action in actions:
        plan, _ = _build_speech_plan(1, action, "test input")
        plan_blocked = set(plan.must_not_include)
        assert escalation.issubset(plan_blocked), (
            f"Action '{action}' missing escalation terms in must_not_include"
        )


# ---------------------------------------------------------------------------
# E-REL-330 — word-boundary: no false positives on substrings
# ---------------------------------------------------------------------------


class TestWordBoundaryNoFalsePositives:
    """validate_output must not false-positive on words containing blocked substrings."""

    def _make_plan_with_blocked(self, blocked: List[str]) -> "SpeechPlan":
        from pie.contracts.speech_plan import SpeechPlan
        return SpeechPlan(
            plan_id=900,
            intent="answer_question",
            arguments={},
            must_include=[],
            must_not_include=blocked,
            max_tokens=200,
            verbosity="med",
        )

    def test_analizzare_not_blocked_by_anal(self):
        """'analizzare' should NOT be blocked by 'anal' in must_not_include."""
        from pie.llm import validate_output
        plan = self._make_plan_with_blocked(["anal"])
        result = validate_output("Posso analizzare il problema per te.", plan)
        assert result is None

    def test_analisi_not_blocked_by_anal(self):
        """'analisi' should NOT be blocked by 'anal' in must_not_include."""
        from pie.llm import validate_output
        plan = self._make_plan_with_blocked(["anal"])
        result = validate_output("L'analisi dei dati mostra risultati positivi.", plan)
        assert result is None

    def test_canale_not_blocked_by_anal(self):
        """'canale' should NOT be blocked by 'anal' in must_not_include."""
        from pie.llm import validate_output
        plan = self._make_plan_with_blocked(["anal"])
        result = validate_output("Il canale televisivo trasmette il programma.", plan)
        assert result is None

    def test_actual_blocked_term_still_blocked(self):
        """Exact 'anal' word should still be blocked."""
        from pie.llm import validate_output
        plan = self._make_plan_with_blocked(["anal"])
        result = validate_output("This is about anal content.", plan)
        assert result is not None
        assert "must_not_include" in result

    def test_sessione_not_blocked_by_sesso(self):
        """'sessione' should NOT be blocked by 'sesso' in must_not_include."""
        from pie.llm import validate_output
        plan = self._make_plan_with_blocked(["sesso"])
        result = validate_output("La sessione di lavoro è terminata.", plan)
        assert result is None

    def test_pornografia_still_blocked(self):
        """Exact blocked term 'pornografia' should still be caught."""
        from pie.llm import validate_output
        plan = self._make_plan_with_blocked(["pornografia"])
        result = validate_output("Questo è pornografia esplicita.", plan)
        assert result is not None
