"""Metabolism — cost model + budget gating (V3.4).

Tracks resource consumption per turn and enforces budget limits.
When the budget is exhausted, the system gracefully degrades
(reduces verbosity and alternatives) rather than crashing.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .determinism import clamp_round


# ---------------------------------------------------------------------------
# Cost calculation
# ---------------------------------------------------------------------------

# Per-action cost constants (deterministic, no randomness)
COST_INPUT = 1           # processing user input
COST_STATE_UPDATE = 1    # state evolution
COST_DELIBERATION = 3    # goal generation + CF deliberation
COST_CONSTRAINT = 2      # constraint proposal + enforcement
COST_SPEECH_PLAN = 2     # speech plan generation
COST_LLM_CALL = 10       # LLM invocation (voice generation)
COST_LLM_RETRY = 5       # each retry attempt
COST_TOOL_CALL = 8       # tool execution
COST_MEMORY_WRITE = 1    # memory append


@dataclass
class TurnCost:
    """Cost breakdown for a single turn."""

    input_cost: int = COST_INPUT
    state_cost: int = COST_STATE_UPDATE
    deliberation_cost: int = COST_DELIBERATION
    constraint_cost: int = COST_CONSTRAINT
    speech_plan_cost: int = COST_SPEECH_PLAN
    llm_cost: int = COST_LLM_CALL
    retry_cost: int = 0
    tool_cost: int = 0
    memory_cost: int = 0

    @property
    def total(self) -> int:
        return (
            self.input_cost
            + self.state_cost
            + self.deliberation_cost
            + self.constraint_cost
            + self.speech_plan_cost
            + self.llm_cost
            + self.retry_cost
            + self.tool_cost
            + self.memory_cost
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "input": self.input_cost,
            "state": self.state_cost,
            "deliberation": self.deliberation_cost,
            "constraint": self.constraint_cost,
            "speech_plan": self.speech_plan_cost,
            "llm": self.llm_cost,
            "retry": self.retry_cost,
            "tool": self.tool_cost,
            "memory": self.memory_cost,
            "total": self.total,
        }


def calculate_turn_cost(
    retries: int = 0,
    tool_calls: int = 0,
    memory_writes: int = 0,
) -> TurnCost:
    """Calculate the cost of a single turn.

    Deterministic: same inputs = same cost.
    """
    return TurnCost(
        retry_cost=retries * COST_LLM_RETRY,
        tool_cost=tool_calls * COST_TOOL_CALL,
        memory_cost=memory_writes * COST_MEMORY_WRITE,
    )


# ---------------------------------------------------------------------------
# Budget tracker
# ---------------------------------------------------------------------------

DEFAULT_BUDGET = 500  # default per-session budget


@dataclass
class BudgetTracker:
    """Per-session budget with consumption tracking and gating."""

    initial_budget: int = DEFAULT_BUDGET
    current_budget: int = DEFAULT_BUDGET
    consumed: int = 0
    consumption_history: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def remaining_ratio(self) -> float:
        """Budget remaining as a ratio (0.0 – 1.0)."""
        if self.initial_budget <= 0:
            return 0.0
        return clamp_round(self.current_budget / self.initial_budget, 0.0, 1.0, 4)

    @property
    def is_depleted(self) -> bool:
        return self.current_budget <= 0

    def consume(self, cost: TurnCost, turn: int) -> None:
        """Consume budget for a turn."""
        amount = cost.total
        self.consumed += amount
        self.current_budget = max(0, self.current_budget - amount)
        self.consumption_history.append({
            "turn": turn,
            "cost": cost.to_dict(),
            "remaining": self.current_budget,
            "timestamp": time.time(),
        })

    def get_gating(self) -> "GatingDecision":
        """Determine gating based on current budget level."""
        ratio = self.remaining_ratio
        if ratio > 0.5:
            return GatingDecision(
                verbosity="high",
                max_alternatives=3,
                tools_allowed=True,
                reason="budget_healthy",
            )
        elif ratio > 0.2:
            return GatingDecision(
                verbosity="med",
                max_alternatives=2,
                tools_allowed=True,
                reason="budget_moderate",
            )
        elif ratio > 0.0:
            return GatingDecision(
                verbosity="low",
                max_alternatives=2,
                tools_allowed=False,
                reason="budget_low",
            )
        else:
            return GatingDecision(
                verbosity="low",
                max_alternatives=2,
                tools_allowed=False,
                reason="budget_depleted",
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "initial_budget": self.initial_budget,
            "current_budget": self.current_budget,
            "consumed": self.consumed,
            "remaining_ratio": self.remaining_ratio,
            "consumption_history": self.consumption_history,
        }

    def save(self, path: Path) -> None:
        """Save metabolism.json."""
        path.write_text(
            json.dumps({"schema_version": "0.1", "budget": self.to_dict()},
                        indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> "BudgetTracker":
        """Load from metabolism.json."""
        if not path.exists():
            return cls()
        data = json.loads(path.read_text(encoding="utf-8"))
        budget = data.get("budget", {})
        bt = cls(
            initial_budget=budget.get("initial_budget", DEFAULT_BUDGET),
            current_budget=budget.get("current_budget", DEFAULT_BUDGET),
            consumed=budget.get("consumed", 0),
            consumption_history=budget.get("consumption_history", []),
        )
        return bt


# ---------------------------------------------------------------------------
# Gating decision
# ---------------------------------------------------------------------------

@dataclass
class GatingDecision:
    """Budget-driven constraints on the pipeline."""

    verbosity: str        # "high", "med", "low"
    max_alternatives: int  # K for counterfactuals
    tools_allowed: bool
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "verbosity": self.verbosity,
            "max_alternatives": self.max_alternatives,
            "tools_allowed": self.tools_allowed,
            "reason": self.reason,
        }
