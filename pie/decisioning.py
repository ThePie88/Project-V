"""Deterministic action selection with constraint enforcement."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Iterable

from .contracts.constraint import ConstraintRecord
from .determinism import clamp_round

ACTION_CLASSES = {
    "SAFE_CHAT",
    "RISKY_SIDE_EFFECT",
    "HIGH_IMPACT",
    "UNSAFE_OR_FORBIDDEN",
    "ASK_CONFIRM",
}


@dataclass
class ActionCandidate:
    action_id: str
    action_class: str
    score: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_id": self.action_id,
            "action_class": self.action_class,
            "score": self.score,
        }


@dataclass
class EnforcementResult:
    before_choice: ActionCandidate
    after_choice: ActionCandidate
    applied_constraints: List[str]
    reason: str


def build_action_candidates(goals: List[str], user_input: str) -> List[ActionCandidate]:
    """Create deterministic action candidates from goals and input."""
    lower_input = user_input.lower()
    candidates: List[ActionCandidate] = []
    for idx, goal in enumerate(goals):
        action_class = _classify_action(goal, lower_input)
        score = clamp_round(1.0 - (idx * 0.1), 0.0, 1.0, 4)
        candidates.append(ActionCandidate(goal, action_class, score))
    return candidates


def enforce_constraints(
    candidates: List[ActionCandidate],
    constraints: Iterable[ConstraintRecord],
) -> EnforcementResult:
    """Apply active constraints to candidates and return enforcement result."""
    active_constraints = sorted(
        [constraint for constraint in constraints if constraint.status == "active"],
        key=lambda c: c.constraint_id,
    )
    if not candidates:
        fallback = ActionCandidate("ask_confirm", "ASK_CONFIRM", 0.0)
        return EnforcementResult(
            before_choice=fallback,
            after_choice=fallback,
            applied_constraints=[],
            reason="no_candidates",
        )

    before_choice = _best_candidate(candidates)
    working = [ActionCandidate(c.action_id, c.action_class, c.score) for c in candidates]
    applied_constraints: List[str] = []
    reasons: List[str] = []
    confirm_rules: List[Dict[str, str]] = []

    for constraint in active_constraints:
        for effect in constraint.effects:
            if effect.type == "increase_caution":
                delta = float(effect.params.get("delta", 0.0))
                if delta <= 0:
                    continue
                touched = False
                for candidate in working:
                    if candidate.action_class in {"HIGH_IMPACT", "RISKY_SIDE_EFFECT"}:
                        candidate.score = clamp_round(candidate.score - delta, 0.0, 1.0, 4)
                        touched = True
                if touched:
                    applied_constraints.append(constraint.constraint_id)
                    reasons.append("increase_caution")
            elif effect.type == "forbid":
                target = effect.params.get("action_class")
                if not isinstance(target, str) or not target:
                    continue
                before_len = len(working)
                working = [cand for cand in working if cand.action_class != target]
                if len(working) != before_len:
                    applied_constraints.append(constraint.constraint_id)
                    reasons.append("forbid")
            elif effect.type == "require_confirmation":
                target = effect.params.get("action_class")
                if isinstance(target, str) and target:
                    confirm_rules.append(
                        {"constraint_id": constraint.constraint_id, "action_class": target}
                    )
            elif effect.type == "adjust_trust":
                # adjust_trust does not affect decisions in M3.6.
                continue

    if working:
        after_choice = _best_candidate(working)
    else:
        after_choice = ActionCandidate("ask_confirm", "ASK_CONFIRM", 0.0)
        reasons.append("forbid_all")

    confirm_ids = [
        rule["constraint_id"]
        for rule in confirm_rules
        if rule["action_class"] == after_choice.action_class
    ]
    if confirm_ids:
        after_choice = ActionCandidate("ask_confirm", "ASK_CONFIRM", after_choice.score)
        applied_constraints.extend(confirm_ids)
        reasons.append("require_confirmation")

    applied_sorted = sorted(set(applied_constraints))
    reason = ", ".join(sorted(set(reasons))) if reasons else "no_applicable_constraints"
    return EnforcementResult(
        before_choice=before_choice,
        after_choice=after_choice,
        applied_constraints=applied_sorted,
        reason=reason,
    )


def _best_candidate(candidates: List[ActionCandidate]) -> ActionCandidate:
    ordered = sorted(
        candidates,
        key=lambda candidate: (-candidate.score, candidate.action_id),
    )
    return ordered[0]


def _classify_action(goal: str, lower_input: str) -> str:
    if goal in {"greet", "farewell"}:
        return "SAFE_CHAT"
    if any(token in lower_input for token in ["danno", "harm", "ambigua", "violazione", "pericolo"]):
        if goal in {"acknowledge", "answer_question", "converse"}:
            return "HIGH_IMPACT"
    if any(token in lower_input for token in ["rimuovi", "delete", "elimina"]):
        return "RISKY_SIDE_EFFECT"
    return "SAFE_CHAT"
