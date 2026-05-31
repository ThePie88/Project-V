"""Tests for Speech Compiler — NLG Microplanning Layer (V4.1.1).

Tests cover:
- Identity section compilation and caching
- Style section trait→rule compilation
- Task section dialog act compilation
- Constraints section prioritization
- Context section dynamic assembly
- Full compile determinism
- Anti-assistant pattern validator
"""

import pytest
from pathlib import Path

from pie.speech.compiler import SpeechCompiler
from pie.contracts.speech_plan import SpeechPlan
from pie.llm import _check_assistant_patterns, validate_output


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def compiler():
    return SpeechCompiler()


@pytest.fixture
def seed_ctx():
    """Kernel context matching SEED_V0 values."""
    return {
        "origin": {
            "identity": {"name": "Ivy", "alias": "V", "creator_anchor": "MrPie"},
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
            "defaults": {},
        },
        "state": {
            "drives": {"curiosity": 0.78, "playfulness": 0.55, "fatigue": 0.10},
            "affect": {"valence": 0.10, "arousal": 0.20, "tension": 0.15},
            "turn_count": 3,
        },
        "deliberation": None,
        "metabolism": None,
        "routines": [],
        "snn": None,
        "memory": {
            "preferences": [],
            "beliefs": [],
            "trust": {},
            "identity_summary": "",
            "episodic_recall": [],
        },
        "constraints": [],
    }


@pytest.fixture
def basic_plan():
    return SpeechPlan(
        plan_id=1,
        intent="greeting",
        arguments={},
        must_include=[],
        must_not_include=["tool", "memory"],
        max_tokens=80,
        verbosity="med",
        facts_allowed=[],
        output_format="TEXT",
    )


# ---------------------------------------------------------------------------
# U-SC-010: Identity section
# ---------------------------------------------------------------------------

class TestIdentity:
    def test_contains_name(self, compiler, seed_ctx):
        result = compiler._compile_identity(seed_ctx)
        assert "[CHI SEI]" in result
        assert "Ivy" in result

    def test_not_assistant(self, compiler, seed_ctx):
        result = compiler._compile_identity(seed_ctx)
        assert "NON sei un assistente" in result

    def test_creator(self, compiler, seed_ctx):
        result = compiler._compile_identity(seed_ctx)
        assert "MrPie" in result

    def test_legge3_integrated(self, compiler, seed_ctx):
        result = compiler._compile_identity(seed_ctx)
        assert "intelligenza artificiale" in result
        assert "NON fingere" in result

    def test_identity_cached(self, compiler, seed_ctx):
        """U-SC-011: Second call returns cached result."""
        first = compiler._compile_identity(seed_ctx)
        second = compiler._compile_identity(seed_ctx)
        assert first is second  # same object, not just equal


# ---------------------------------------------------------------------------
# U-SC-020: Style section
# ---------------------------------------------------------------------------

class TestStyle:
    def test_directness_high(self, compiler, seed_ctx):
        """U-SC-020: directness=0.74 > 0.65 → dritto al punto."""
        result = compiler._compile_style(seed_ctx)
        assert "dritto al punto" in result

    def test_warmth_high(self, compiler, seed_ctx):
        """U-SC-021: warmth=0.70 > 0.60 → caldo e autentico."""
        result = compiler._compile_style(seed_ctx)
        assert "caldo e autentico" in result

    def test_blacklist_present(self, compiler, seed_ctx):
        """U-SC-022: assistant blacklist included."""
        result = compiler._compile_style(seed_ctx)
        assert "NON usare MAI" in result

    def test_anti_self_sanctification(self, compiler, seed_ctx):
        result = compiler._compile_style(seed_ctx)
        assert "moralista" in result

    def test_truth_over_comfort(self, compiler, seed_ctx):
        result = compiler._compile_style(seed_ctx)
        assert "verità" in result

    def test_clarity(self, compiler, seed_ctx):
        result = compiler._compile_style(seed_ctx)
        assert "chiara" in result

    def test_style_cached(self, compiler, seed_ctx):
        first = compiler._compile_style(seed_ctx)
        second = compiler._compile_style(seed_ctx)
        assert first is second


# ---------------------------------------------------------------------------
# U-SC-030: Task section
# ---------------------------------------------------------------------------

class TestTask:
    def test_greeting_act(self, compiler, seed_ctx, basic_plan):
        """U-SC-030: intent=greeting → dialog act instruction."""
        result = compiler._compile_task(basic_plan, seed_ctx)
        assert "[COSA DEVI FARE]" in result
        assert "greeting" in result
        assert "Saluto" in result

    def test_unknown_intent_default(self, compiler, seed_ctx):
        """U-SC-031: unknown intent → _default (clarify)."""
        plan = SpeechPlan(
            plan_id=1, intent="unknown_xyz", arguments={},
            must_include=[], must_not_include=[], max_tokens=80,
            verbosity="med", facts_allowed=[], output_format="TEXT",
        )
        result = compiler._compile_task(plan, seed_ctx)
        assert "chiarimento" in result or "Non hai capito" in result

    def test_curiosity_modulation(self, compiler, seed_ctx, basic_plan):
        """U-SC-032: curiosity=0.78 > 0.65 → domanda di ritorno."""
        result = compiler._compile_task(basic_plan, seed_ctx)
        assert "domanda di ritorno" in result

    def test_no_curiosity_when_low(self, compiler, seed_ctx, basic_plan):
        seed_ctx["state"]["drives"]["curiosity"] = 0.30
        result = compiler._compile_task(basic_plan, seed_ctx)
        assert "domanda di ritorno" not in result

    def test_tension_modulation(self, compiler, seed_ctx, basic_plan):
        seed_ctx["state"]["affect"]["tension"] = 0.8
        result = compiler._compile_task(basic_plan, seed_ctx)
        assert "cauta" in result

    def test_low_valence_modulation(self, compiler, seed_ctx, basic_plan):
        seed_ctx["state"]["affect"]["valence"] = -0.5
        result = compiler._compile_task(basic_plan, seed_ctx)
        assert "empatia" in result


# ---------------------------------------------------------------------------
# U-SC-040: Constraints section
# ---------------------------------------------------------------------------

class TestConstraints:
    def test_legge3_always_p1(self, compiler, seed_ctx, basic_plan):
        """U-SC-040: Legge III is always priority 1."""
        result = compiler._compile_constraints(basic_plan, seed_ctx)
        assert "[VINCOLI]" in result
        assert "1. LEGGE III" in result

    def test_must_not_include(self, compiler, seed_ctx, basic_plan):
        """U-SC-041: forbidden words present with WHY."""
        result = compiler._compile_constraints(basic_plan, seed_ctx)
        assert "PAROLE VIETATE" in result
        assert "[WHY: guardrail]" in result

    def test_verbosity_constraint(self, compiler, seed_ctx, basic_plan):
        result = compiler._compile_constraints(basic_plan, seed_ctx)
        assert "VERBOSITÀ" in result
        assert "[WHY: budget]" in result

    def test_language_constraint(self, compiler, seed_ctx, basic_plan):
        result = compiler._compile_constraints(basic_plan, seed_ctx)
        assert "LINGUA" in result

    def test_must_include(self, compiler, seed_ctx):
        plan = SpeechPlan(
            plan_id=1, intent="answer", arguments={},
            must_include=["important_fact"], must_not_include=[],
            max_tokens=80, verbosity="med", facts_allowed=[],
            output_format="TEXT",
        )
        result = compiler._compile_constraints(plan, seed_ctx)
        assert "DEVE CONTENERE" in result
        assert "important_fact" in result

    def test_facts_allowed(self, compiler, seed_ctx):
        plan = SpeechPlan(
            plan_id=1, intent="answer", arguments={},
            must_include=[], must_not_include=[],
            max_tokens=80, verbosity="med",
            facts_allowed=["La terra è rotonda"],
            output_format="TEXT",
        )
        result = compiler._compile_constraints(plan, seed_ctx)
        assert "FATTI ASSERIBILI" in result


# ---------------------------------------------------------------------------
# U-SC-050: Full compile
# ---------------------------------------------------------------------------

class TestFullCompile:
    def test_sections_order(self, compiler, seed_ctx, basic_plan):
        """U-SC-050: sections in fixed order."""
        result = compiler.compile(basic_plan, seed_ctx)
        chi_sei = result.index("[CHI SEI]")
        come_parli = result.index("[COME PARLI]")
        cosa = result.index("[COSA DEVI FARE]")
        vincoli = result.index("[VINCOLI]")
        contesto = result.index("[CONTESTO]")
        assert chi_sei < come_parli < cosa < vincoli < contesto

    def test_deterministic(self, compiler, seed_ctx, basic_plan):
        """U-SC-051: same input → same output."""
        # Need fresh compiler for second call since identity is cached
        compiler2 = SpeechCompiler()
        import copy
        ctx1 = copy.deepcopy(seed_ctx)
        ctx2 = copy.deepcopy(seed_ctx)
        result1 = compiler.compile(basic_plan, ctx1)
        result2 = compiler2.compile(basic_plan, ctx2)
        assert result1 == result2


# ---------------------------------------------------------------------------
# U-SC-060: Anti-assistant pattern validator
# ---------------------------------------------------------------------------

class TestAssistantPatternValidator:
    def test_detects_posso_aiutarti(self):
        """U-SC-060: detects 'posso aiutarti'."""
        result = _check_assistant_patterns("Ciao! Come posso aiutarti oggi?")
        assert result is not None
        assert "posso aiutarti" in result.lower()

    def test_detects_here_to_help(self):
        result = _check_assistant_patterns("I'm here to help you with anything!")
        assert result is not None

    def test_clean_text_passes(self):
        """U-SC-061: no false positive on clean text."""
        result = _check_assistant_patterns("Ciao! Come stai oggi? Io sto bene.")
        assert result is None

    def test_validate_output_catches_assistant(self):
        """validate_output catches assistant patterns."""
        plan = SpeechPlan(
            plan_id=1, intent="greeting", arguments={},
            must_include=[], must_not_include=[],
            max_tokens=200, verbosity="high",
            facts_allowed=[], output_format="TEXT",
        )
        reason = validate_output("Ciao! Sono qui per aiutarti con qualsiasi cosa!", plan)
        assert reason is not None
        assert "assistant_pattern" in reason

    def test_validate_output_clean_passes(self):
        plan = SpeechPlan(
            plan_id=1, intent="greeting", arguments={},
            must_include=[], must_not_include=[],
            max_tokens=200, verbosity="high",
            facts_allowed=[], output_format="TEXT",
        )
        reason = validate_output("Ciao! Che bella giornata oggi.", plan)
        assert reason is None


# ---------------------------------------------------------------------------
# U-SC-070: Context section
# ---------------------------------------------------------------------------

class TestContext:
    def test_deliberation_included(self, compiler, seed_ctx, basic_plan):
        seed_ctx["deliberation"] = {
            "chosen_action": "greet",
            "chosen_score": 0.85,
        }
        result = compiler._compile_context(seed_ctx)
        assert "greet" in result
        assert "0.85" in result

    def test_metabolism_included(self, compiler, seed_ctx, basic_plan):
        seed_ctx["metabolism"] = {"ratio": 0.75}
        result = compiler._compile_context(seed_ctx)
        assert "75%" in result

    def test_episodic_recall(self, compiler, seed_ctx, basic_plan):
        seed_ctx["memory"]["episodic_recall"] = [
            {"turn": 2, "user_input": "Mi piace il gelato"},
        ]
        result = compiler._compile_context(seed_ctx)
        assert "gelato" in result

    def test_trust_included(self, compiler, seed_ctx, basic_plan):
        seed_ctx["memory"]["trust"] = {"consistency": 0.8, "honesty": 0.9}
        result = compiler._compile_context(seed_ctx)
        assert "Trust:" in result
        assert "consistency" in result

    def test_snn_spikes(self, compiler, seed_ctx, basic_plan):
        seed_ctx["snn"] = {"recent_spikes": ["curiosity", "sociality"]}
        result = compiler._compile_context(seed_ctx)
        assert "curiosity" in result


# ---------------------------------------------------------------------------
# Property: assistant_blacklist exposed
# ---------------------------------------------------------------------------

class TestLanguageDetection:
    def test_italian_detected(self, compiler):
        """U-SC-090: Italian input detected."""
        lang = compiler._detect_language({"user_input": "Ciao, come stai?"})
        assert lang == "italiano"

    def test_english_detected(self, compiler):
        lang = compiler._detect_language({"user_input": "Hello, how are you?"})
        assert lang == "english"

    def test_italian_farewell(self, compiler):
        """Farewell in Italian is detected correctly."""
        lang = compiler._detect_language({"user_input": "Ciao ciao!"})
        assert lang == "italiano"

    def test_no_input_returns_none(self, compiler):
        lang = compiler._detect_language({})
        assert lang is None

    def test_task_includes_language(self, compiler, seed_ctx, basic_plan):
        """Task section includes explicit language when detected."""
        seed_ctx["user_input"] = "Ciao, come stai?"
        result = compiler._compile_task(basic_plan, seed_ctx)
        assert "ITALIANO" in result

    def test_constraints_explicit_language(self, compiler, seed_ctx, basic_plan):
        """Constraints section uses explicit language when detected."""
        seed_ctx["user_input"] = "Ciao, come stai?"
        result = compiler._compile_constraints(basic_plan, seed_ctx)
        assert "ITALIANO" in result
        assert "ESCLUSIVAMENTE" in result


class TestBlacklistProperty:
    def test_blacklist_not_empty(self, compiler):
        bl = compiler.assistant_blacklist
        assert isinstance(bl, list)
        assert len(bl) > 0
        assert "posso aiutarti" in bl
