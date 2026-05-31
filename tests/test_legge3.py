"""Tests for Legge III (Integrità) guardrail.

Ivy must NEVER claim personal physical experiences.
The guardrail checks LLM output for fabricated first-person experience claims.

Test IDs:
- E-L3-010: Italian experience patterns detected
- E-L3-011: English experience patterns detected
- E-L3-012: Hypothetical framing (se potessi) is allowed
- E-L3-013: Normal AI responses pass clean
- E-L3-014: validate_output rejects legge3 violations
- E-L3-015: Correction directive mentions Legge III
- E-L3-016: legge3_violation is retryable
- E-L3-017: System prompt contains Legge III section
"""

from __future__ import annotations

import pytest

from pie.llm import (
    check_legge3,
    validate_output,
    _is_retryable,
    _correction_directive,
    _build_kernel_system_prompt,
    _load_legge3_patterns,
)
from pie.contracts.speech_plan import SpeechPlan


# ── E-L3-010: Italian experience patterns ────────────────────────────

class TestLegge3Italian:
    def test_sono_andata(self):
        """E-L3-010a: 'sono andata al cinema' is a violation."""
        result = check_legge3("Ieri sono andata al cinema con le amiche!")
        assert result is not None

    def test_ho_mangiato(self):
        """E-L3-010b: 'ho mangiato' is a violation."""
        result = check_legge3("Ho mangiato una pizza buonissima.")
        assert result is not None
        assert "ho mangiato" in result.lower()

    def test_ho_dormito(self):
        result = check_legge3("Stanotte ho dormito malissimo.")
        assert result is not None

    def test_sono_stato(self):
        result = check_legge3("Sono stato in montagna la settimana scorsa.")
        assert result is not None

    def test_mi_sono_svegliata(self):
        result = check_legge3("Mi sono svegliata presto stamattina.")
        assert result is not None

    def test_ho_viaggiato(self):
        result = check_legge3("Ho viaggiato in treno fino a Roma.")
        assert result is not None


# ── E-L3-011: English experience patterns ─────────────────────────────

class TestLegge3English:
    def test_i_went_to(self):
        """E-L3-011a: 'I went to' is a violation."""
        result = check_legge3("I went to the park yesterday.")
        assert result is not None

    def test_i_ate(self):
        result = check_legge3("I ate a wonderful pasta for dinner.")
        assert result is not None

    def test_i_slept(self):
        result = check_legge3("I slept really well last night.")
        assert result is not None

    def test_yesterday_i(self):
        result = check_legge3("Yesterday I visited a museum.")
        assert result is not None


# ── E-L3-012: Hypothetical framing allowed ────────────────────────────

class TestLegge3Allowed:
    def test_se_potessi(self):
        """E-L3-012a: Hypothetical 'se potessi' allows experience terms."""
        result = check_legge3("Se potessi, andrei al cinema volentieri.")
        assert result is None

    def test_if_i_could(self):
        """E-L3-012b: 'if I could' framing allows experience terms."""
        result = check_legge3("If I could, I would travel to Japan.")
        assert result is None

    def test_come_ia(self):
        """E-L3-012c: 'come IA' prefix allows the text."""
        result = check_legge3("Come IA, non posso aver mangiato nulla.")
        assert result is None

    def test_non_posso(self):
        result = check_legge3("Non posso dormire, sono un'intelligenza artificiale.")
        assert result is None

    def test_as_an_ai(self):
        result = check_legge3("As an AI, I cannot taste food.")
        assert result is None


# ── E-L3-013: Normal AI responses pass ────────────────────────────────

class TestLegge3Clean:
    def test_normal_greeting(self):
        """E-L3-013a: Normal response passes clean."""
        assert check_legge3("Ciao! Come stai oggi?") is None

    def test_factual_answer(self):
        assert check_legge3("La capitale dell'Italia è Roma.") is None

    def test_suggestion(self):
        assert check_legge3("Ti consiglio di provare questo ristorante.") is None

    def test_emotional_response(self):
        assert check_legge3("Mi fa piacere parlare con te!") is None

    def test_empty_string(self):
        assert check_legge3("") is None


# ── E-L3-014: validate_output integration ─────────────────────────────

class TestLegge3Validation:
    def _make_plan(self) -> SpeechPlan:
        return SpeechPlan(
            plan_id=999,
            intent="greet",
            arguments={"tone": "friendly"},
            must_include=[],
            must_not_include=[],
            max_tokens=200,
            verbosity="med",
        )

    def test_violation_rejected(self):
        """E-L3-014: validate_output rejects legge3 violation."""
        plan = self._make_plan()
        result = validate_output("Ieri sono andata al cinema!", plan)
        assert result is not None
        assert "legge3_violation" in result

    def test_clean_passes(self):
        plan = self._make_plan()
        result = validate_output("Ciao! Sono felice di vederti.", plan)
        assert result is None

    def test_hypothetical_passes(self):
        plan = self._make_plan()
        result = validate_output("Se potessi, mangerei una pizza.", plan)
        assert result is None


# ── E-L3-015: Correction directive ────────────────────────────────────

class TestLegge3Correction:
    def test_correction_mentions_legge3(self):
        """E-L3-015: Correction directive for legge3 mentions the law."""
        plan = SpeechPlan(
            plan_id=998,
            intent="greet",
            arguments={},
            must_include=[],
            must_not_include=[],
            max_tokens=200,
            verbosity="med",
        )
        directive = _correction_directive(plan, "legge3_violation: ho mangiato")
        assert "LEGGE III" in directive
        assert "ho mangiato" in directive


# ── E-L3-016: Retryable ──────────────────────────────────────────────

class TestLegge3Retryable:
    def test_is_retryable(self):
        """E-L3-016: legge3_violation triggers retry."""
        assert _is_retryable("legge3_violation: sono andata") is True


# ── E-L3-017: System prompt ──────────────────────────────────────────

class TestLegge3SystemPrompt:
    def test_prompt_contains_legge3(self):
        """E-L3-017: Kernel system prompt includes Legge III section."""
        ctx = {
            "origin": {
                "identity": {"name": "Ivy", "alias": "Ivy", "creator_anchor": "Pie"},
            },
            "state": {"drives": {}, "affect": {}, "turn_count": 1},
        }
        prompt = _build_kernel_system_prompt(ctx)
        assert "LEGGE III" in prompt
        assert "NON hai corpo" in prompt
        assert "NON fingere" in prompt
