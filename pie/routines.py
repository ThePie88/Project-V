"""Routine/Skills library (V3.4).

Provides reusable routines that can be triggered by input patterns.
Routines are ranked by success count and decayed over time.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .determinism import clamp_round


# ---------------------------------------------------------------------------
# Routine definition
# ---------------------------------------------------------------------------

@dataclass
class Routine:
    """A reusable skill/routine."""

    routine_id: str
    trigger_keywords: List[str]
    description: str
    action_sequence: List[str]
    success_count: int = 0
    fail_count: int = 0
    total_cost: int = 0
    last_used: Optional[float] = None
    decay_factor: float = 0.95

    @property
    def score(self) -> float:
        """Ranking score: success rate * decay."""
        total = self.success_count + self.fail_count
        if total == 0:
            return 0.5  # neutral for untested routines
        success_rate = self.success_count / total
        return clamp_round(success_rate * self.decay_factor, 0.0, 1.0, 4)

    @property
    def avg_cost(self) -> float:
        total = self.success_count + self.fail_count
        if total == 0:
            return 0.0
        return round(self.total_cost / total, 2)

    def matches(self, user_input: str) -> bool:
        """Check if input triggers this routine."""
        lower = user_input.lower()
        return any(kw in lower for kw in self.trigger_keywords)

    def record_use(self, success: bool, cost: int) -> None:
        """Record a routine invocation."""
        if success:
            self.success_count += 1
        else:
            self.fail_count += 1
        self.total_cost += cost
        self.last_used = time.time()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "routine_id": self.routine_id,
            "trigger_keywords": self.trigger_keywords,
            "description": self.description,
            "action_sequence": self.action_sequence,
            "success_count": self.success_count,
            "fail_count": self.fail_count,
            "total_cost": self.total_cost,
            "avg_cost": self.avg_cost,
            "score": self.score,
            "last_used": self.last_used,
            "decay_factor": self.decay_factor,
        }


# ---------------------------------------------------------------------------
# Skill invocation record
# ---------------------------------------------------------------------------

@dataclass
class SkillInvocation:
    """Record of a single routine invocation."""

    turn: int
    routine_id: str
    outcome: str  # "success" or "fail"
    cost: int
    improvement_delta: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "turn": self.turn,
            "routine_id": self.routine_id,
            "outcome": self.outcome,
            "cost": self.cost,
            "improvement_delta": self.improvement_delta,
        }


# ---------------------------------------------------------------------------
# Routine Library
# ---------------------------------------------------------------------------

# Built-in routines
_DEFAULT_ROUTINES = [
    Routine(
        routine_id="summarize",
        trigger_keywords=["riassumi", "summarize", "sintesi", "summary"],
        description="Summarize the current conversation or topic",
        action_sequence=["gather_context", "compress", "output_summary"],
    ),
    Routine(
        routine_id="plan",
        trigger_keywords=["pianifica", "plan", "organizza", "schedule"],
        description="Create a structured plan for a task",
        action_sequence=["analyze_goal", "decompose", "output_plan"],
    ),
    Routine(
        routine_id="explain",
        trigger_keywords=["spiega", "explain", "chiarisci", "clarify"],
        description="Explain a concept clearly",
        action_sequence=["identify_concept", "simplify", "output_explanation"],
    ),
]


class RoutineLibrary:
    """Manages reusable routines with trigger matching and ranking."""

    def __init__(self) -> None:
        self._routines: Dict[str, Routine] = {}
        self._invocations: List[SkillInvocation] = []
        # Register defaults
        for r in _DEFAULT_ROUTINES:
            self._routines[r.routine_id] = Routine(
                routine_id=r.routine_id,
                trigger_keywords=list(r.trigger_keywords),
                description=r.description,
                action_sequence=list(r.action_sequence),
            )

    def register(self, routine: Routine) -> None:
        """Register a new routine."""
        self._routines[routine.routine_id] = routine

    def find_matching(self, user_input: str) -> Optional[Routine]:
        """Find the best matching routine for the input.

        Returns the highest-scoring matching routine, or None.
        """
        matches = [r for r in self._routines.values() if r.matches(user_input)]
        if not matches:
            return None
        # Sort by score descending, then by routine_id for determinism
        matches.sort(key=lambda r: (-r.score, r.routine_id))
        return matches[0]

    def record_invocation(
        self,
        routine_id: str,
        turn: int,
        success: bool,
        cost: int,
        improvement_delta: float = 0.0,
    ) -> SkillInvocation:
        """Record a routine invocation and update its stats."""
        routine = self._routines.get(routine_id)
        if routine:
            routine.record_use(success, cost)

        inv = SkillInvocation(
            turn=turn,
            routine_id=routine_id,
            outcome="success" if success else "fail",
            cost=cost,
            improvement_delta=improvement_delta,
        )
        self._invocations.append(inv)
        return inv

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": "0.1",
            "routines_registered": [r.to_dict() for r in self._routines.values()],
            "skill_invocations": [i.to_dict() for i in self._invocations],
        }

    def save(self, path: Path) -> None:
        """Save skills_run.json."""
        path.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> "RoutineLibrary":
        """Load from skills_run.json."""
        lib = cls()
        if not path.exists():
            return lib
        data = json.loads(path.read_text(encoding="utf-8"))
        for rd in data.get("routines_registered", []):
            routine = Routine(
                routine_id=rd["routine_id"],
                trigger_keywords=rd.get("trigger_keywords", []),
                description=rd.get("description", ""),
                action_sequence=rd.get("action_sequence", []),
                success_count=rd.get("success_count", 0),
                fail_count=rd.get("fail_count", 0),
                total_cost=rd.get("total_cost", 0),
                last_used=rd.get("last_used"),
                decay_factor=rd.get("decay_factor", 0.95),
            )
            lib._routines[routine.routine_id] = routine
        for inv_d in data.get("skill_invocations", []):
            lib._invocations.append(SkillInvocation(
                turn=inv_d["turn"],
                routine_id=inv_d["routine_id"],
                outcome=inv_d["outcome"],
                cost=inv_d["cost"],
                improvement_delta=inv_d.get("improvement_delta", 0.0),
            ))
        return lib
