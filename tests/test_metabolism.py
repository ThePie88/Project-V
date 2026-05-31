"""Tests for V3.4 — Metabolismo + Routine/Skills.

Test IDs:
- U-META-100: Cost calculation
- E-META-110: Budget gating (degrade controllato)
- I-SKILL-120: Routine library smoke
- E-SKILL-130: Routine riuso migliora outcome
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pie.metabolism import (
    TurnCost,
    calculate_turn_cost,
    BudgetTracker,
    GatingDecision,
    COST_INPUT,
    COST_LLM_CALL,
    COST_LLM_RETRY,
    COST_TOOL_CALL,
    COST_MEMORY_WRITE,
    DEFAULT_BUDGET,
)
from pie.routines import Routine, RoutineLibrary, SkillInvocation


# -----------------------------------------------------------------------
# U-META-100 — Cost calculation
# -----------------------------------------------------------------------


class TestCostCalculation:
    """Verify cost calculation is correct and deterministic."""

    def test_base_turn_cost(self):
        """A turn with no extras must have the base cost."""
        cost = calculate_turn_cost()
        assert cost.total == (
            COST_INPUT + 1 + 3 + 2 + 2 + COST_LLM_CALL
        )  # 1+1+3+2+2+10 = 19

    def test_retries_add_cost(self):
        """Each retry adds COST_LLM_RETRY."""
        cost = calculate_turn_cost(retries=2)
        expected_retry = 2 * COST_LLM_RETRY
        assert cost.retry_cost == expected_retry
        assert cost.total == 19 + expected_retry

    def test_tool_calls_add_cost(self):
        """Each tool call adds COST_TOOL_CALL."""
        cost = calculate_turn_cost(tool_calls=3)
        assert cost.tool_cost == 3 * COST_TOOL_CALL
        assert cost.total == 19 + 3 * COST_TOOL_CALL

    def test_memory_writes_add_cost(self):
        """Each memory write adds COST_MEMORY_WRITE."""
        cost = calculate_turn_cost(memory_writes=5)
        assert cost.memory_cost == 5 * COST_MEMORY_WRITE

    def test_deterministic(self):
        """Same inputs must always produce same cost."""
        c1 = calculate_turn_cost(retries=1, tool_calls=2, memory_writes=3)
        c2 = calculate_turn_cost(retries=1, tool_calls=2, memory_writes=3)
        assert c1.total == c2.total
        assert c1.to_dict() == c2.to_dict()

    def test_to_dict_has_all_fields(self):
        """Cost dict must include all component fields + total."""
        cost = calculate_turn_cost()
        d = cost.to_dict()
        for key in ["input", "state", "deliberation", "constraint",
                     "speech_plan", "llm", "retry", "tool", "memory", "total"]:
            assert key in d


# -----------------------------------------------------------------------
# E-META-110 — Budget gating (degrade controllato)
# -----------------------------------------------------------------------


class TestBudgetGating:
    """Verify budget tracking and graceful degradation."""

    def test_initial_budget(self):
        """Fresh budget must have full initial amount."""
        bt = BudgetTracker()
        assert bt.current_budget == DEFAULT_BUDGET
        assert bt.consumed == 0
        assert bt.remaining_ratio == 1.0

    def test_consume_decrements_budget(self):
        """Consuming cost must decrement the budget."""
        bt = BudgetTracker(initial_budget=100, current_budget=100)
        cost = calculate_turn_cost()  # 19
        bt.consume(cost, turn=1)
        assert bt.current_budget == 81
        assert bt.consumed == 19
        assert len(bt.consumption_history) == 1

    def test_budget_cannot_go_negative(self):
        """Budget must never go below zero."""
        bt = BudgetTracker(initial_budget=10, current_budget=10)
        cost = calculate_turn_cost()  # 19 > 10
        bt.consume(cost, turn=1)
        assert bt.current_budget == 0
        assert bt.is_depleted is True

    def test_gating_healthy_budget(self):
        """High budget → high verbosity, 3 alternatives, tools allowed."""
        bt = BudgetTracker(initial_budget=100, current_budget=100)
        g = bt.get_gating()
        assert g.verbosity == "high"
        assert g.max_alternatives == 3
        assert g.tools_allowed is True
        assert g.reason == "budget_healthy"

    def test_gating_moderate_budget(self):
        """30% budget → med verbosity, 2 alternatives."""
        bt = BudgetTracker(initial_budget=100, current_budget=30)
        g = bt.get_gating()
        assert g.verbosity == "med"
        assert g.max_alternatives == 2

    def test_gating_low_budget(self):
        """10% budget → low verbosity, no tools."""
        bt = BudgetTracker(initial_budget=100, current_budget=10)
        g = bt.get_gating()
        assert g.verbosity == "low"
        assert g.tools_allowed is False

    def test_gating_depleted(self):
        """0 budget → low verbosity, no tools, depleted reason."""
        bt = BudgetTracker(initial_budget=100, current_budget=0)
        g = bt.get_gating()
        assert g.verbosity == "low"
        assert g.tools_allowed is False
        assert g.reason == "budget_depleted"

    def test_no_crash_when_depleted(self):
        """Processing should continue (degrade) when budget is 0."""
        bt = BudgetTracker(initial_budget=50, current_budget=50)
        # Consume multiple turns until depleted
        for turn in range(1, 10):
            cost = calculate_turn_cost()
            bt.consume(cost, turn=turn)
            g = bt.get_gating()
            # Must always return a valid gating — never crash
            assert g.verbosity in ("high", "med", "low")
            assert isinstance(g.max_alternatives, int)
            assert g.max_alternatives >= 2

    def test_save_and_load(self, tmp_path):
        """Budget must survive save/load cycle."""
        bt = BudgetTracker(initial_budget=200, current_budget=150)
        cost = calculate_turn_cost()
        bt.consume(cost, turn=1)
        path = tmp_path / "metabolism.json"
        bt.save(path)

        loaded = BudgetTracker.load(path)
        assert loaded.initial_budget == 200
        assert loaded.current_budget == bt.current_budget
        assert loaded.consumed == bt.consumed


# -----------------------------------------------------------------------
# I-SKILL-120 — Routine library smoke
# -----------------------------------------------------------------------


class TestRoutineLibrary:
    """Verify routine registration, matching, and execution."""

    def test_default_routines_registered(self):
        """Library must have default routines on init."""
        lib = RoutineLibrary()
        assert "summarize" in lib._routines
        assert "plan" in lib._routines
        assert "explain" in lib._routines

    def test_trigger_matching(self):
        """Input with trigger keyword must match the routine."""
        lib = RoutineLibrary()
        match = lib.find_matching("Puoi fare un summarize di questo?")
        assert match is not None
        assert match.routine_id == "summarize"

    def test_no_match_returns_none(self):
        """Input without trigger keywords must return None."""
        lib = RoutineLibrary()
        match = lib.find_matching("Hello how are you?")
        assert match is None

    def test_register_custom_routine(self):
        """Custom routines can be registered."""
        lib = RoutineLibrary()
        r = Routine(
            routine_id="clean",
            trigger_keywords=["pulisci", "clean", "clear"],
            description="Clean up session data",
            action_sequence=["identify_target", "clean", "confirm"],
        )
        lib.register(r)
        match = lib.find_matching("Pulisci la sessione")
        assert match is not None
        assert match.routine_id == "clean"

    def test_record_invocation_updates_stats(self):
        """Recording a use must update success_count and avg_cost."""
        lib = RoutineLibrary()
        lib.record_invocation("summarize", turn=1, success=True, cost=15)
        r = lib._routines["summarize"]
        assert r.success_count == 1
        assert r.avg_cost == 15.0

    def test_routine_score_increases_with_success(self):
        """Routine score must reflect success rate."""
        lib = RoutineLibrary()
        lib.record_invocation("summarize", turn=1, success=True, cost=10)
        lib.record_invocation("summarize", turn=2, success=True, cost=10)
        r = lib._routines["summarize"]
        assert r.score > 0.5  # success rate = 1.0 * 0.95 = 0.95

    def test_routine_score_decreases_with_failure(self):
        """Routine score must drop when failures occur."""
        lib = RoutineLibrary()
        lib.record_invocation("summarize", turn=1, success=False, cost=10)
        lib.record_invocation("summarize", turn=2, success=False, cost=10)
        r = lib._routines["summarize"]
        assert r.score < 0.5

    def test_best_match_by_score(self):
        """When multiple routines match, the highest-scored wins."""
        lib = RoutineLibrary()
        # Make "explain" have high score
        lib.record_invocation("explain", turn=1, success=True, cost=5)
        lib.record_invocation("explain", turn=2, success=True, cost=5)
        # Register a routine with overlapping keywords
        lib.register(Routine(
            routine_id="explain_v2",
            trigger_keywords=["spiega", "explain"],
            description="V2 explain",
            action_sequence=["explain"],
        ))
        # "explain" should win because it has 2 successes
        match = lib.find_matching("Spiega questo concetto")
        assert match.routine_id == "explain"

    def test_save_and_load(self, tmp_path):
        """Library must survive save/load cycle."""
        lib = RoutineLibrary()
        lib.record_invocation("summarize", turn=1, success=True, cost=15)
        path = tmp_path / "skills_run.json"
        lib.save(path)

        loaded = RoutineLibrary.load(path)
        assert loaded._routines["summarize"].success_count == 1
        assert len(loaded._invocations) == 1


# -----------------------------------------------------------------------
# E-SKILL-130 — Routine riuso migliora outcome
# -----------------------------------------------------------------------


class TestRoutineReuse:
    """Verify that reusing a routine improves outcome (lower cost)."""

    def test_second_use_lower_avg_cost(self):
        """Second invocation of a routine should show improvement tracking."""
        lib = RoutineLibrary()

        # First use: higher cost
        inv1 = lib.record_invocation("summarize", turn=1, success=True, cost=20)

        # Second use: lower cost (simulating improvement from reuse)
        inv2 = lib.record_invocation(
            "summarize", turn=2, success=True, cost=12,
            improvement_delta=0.4,  # 40% improvement
        )

        assert inv2.improvement_delta > 0
        r = lib._routines["summarize"]
        assert r.avg_cost < 20  # average decreased
        assert r.success_count == 2

    def test_reuse_e2e_scenario(self, tmp_path):
        """Full E2E: trigger → execute → record → retrigger → better outcome."""
        lib = RoutineLibrary()

        # Turn 1: user asks to summarize
        match = lib.find_matching("Riassumi il discorso")
        assert match is not None
        assert match.routine_id == "summarize"

        # Execute (simulated) — first time costs more
        cost1 = 25
        lib.record_invocation("summarize", turn=1, success=True, cost=cost1)

        # Turn 5: user asks to summarize again
        match2 = lib.find_matching("Puoi fare una sintesi?")
        assert match2 is not None
        assert match2.routine_id == "summarize"

        # Execute again — improvement from reuse
        cost2 = 15
        delta = round((cost1 - cost2) / cost1, 4)
        lib.record_invocation(
            "summarize", turn=5, success=True, cost=cost2,
            improvement_delta=delta,
        )

        # Verify improvement
        r = lib._routines["summarize"]
        assert r.success_count == 2
        assert r.avg_cost == (cost1 + cost2) / 2
        assert r.score > 0.9  # high success rate

        # Save and verify artifact
        path = tmp_path / "skills_run.json"
        lib.save(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        assert len(data["skill_invocations"]) == 2
        assert data["skill_invocations"][1]["improvement_delta"] > 0
