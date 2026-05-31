"""Counterfactual deliberation engine (V3.2).

Generates K alternative action candidates, scores them with a
multi-objective function, and selects the best one.  All candidates
and rejected alternatives are recorded for traceability.

Key invariant: NO side-effects (memory writes, tool calls, constraint
appends) occur during the explore phase.  Only the final chosen action
proceeds to the speech plan and downstream pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .contracts.constraint import ConstraintRecord
from .decisioning import ActionCandidate, build_action_candidates
from .determinism import clamp_round


# ---------------------------------------------------------------------------
# Score breakdown — one per candidate
# ---------------------------------------------------------------------------

@dataclass
class ScoreBreakdown:
    """Multi-objective score for a single candidate."""

    utility: float      # how well it fulfils goals (0–1)
    risk: float         # penalty for risky action classes (0–1, lower = safer)
    cost: float         # resource cost estimate (0–1, lower = cheaper)
    constraint_fit: float  # how well it satisfies active constraints (0–1)

    @property
    def total(self) -> float:
        """Weighted aggregate score."""
        return clamp_round(
            0.40 * self.utility
            + 0.25 * (1.0 - self.risk)
            + 0.10 * (1.0 - self.cost)
            + 0.25 * self.constraint_fit,
            0.0, 1.0, 4,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "utility": self.utility,
            "risk": self.risk,
            "cost": self.cost,
            "constraint_fit": self.constraint_fit,
            "total": self.total,
        }


# ---------------------------------------------------------------------------
# Counterfactual candidate
# ---------------------------------------------------------------------------

@dataclass
class CFCandidate:
    """A counterfactual alternative with its score and outcome."""

    candidate_id: int
    action: ActionCandidate
    scores: ScoreBreakdown
    is_chosen: bool = False
    reject_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "action_id": self.action.action_id,
            "action_class": self.action.action_class,
            "raw_score": self.action.score,
            "scores": self.scores.to_dict(),
            "is_chosen": self.is_chosen,
            "reject_reason": self.reject_reason,
        }


# ---------------------------------------------------------------------------
# Deliberation result
# ---------------------------------------------------------------------------

@dataclass
class DeliberationResult:
    """Output of the counterfactual deliberation process."""

    candidates: List[CFCandidate]
    chosen: CFCandidate
    k: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "k": self.k,
            "candidates": [c.to_dict() for c in self.candidates],
            "chosen": self.chosen.to_dict(),
        }


# ---------------------------------------------------------------------------
# Risk classification map
# ---------------------------------------------------------------------------

_RISK_MAP = {
    "SAFE_CHAT": 0.05,
    "ASK_CONFIRM": 0.15,
    "RISKY_SIDE_EFFECT": 0.60,
    "HIGH_IMPACT": 0.80,
    "UNSAFE_OR_FORBIDDEN": 1.00,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def deliberate(
    goals: List[str],
    user_input: str,
    active_constraints: List[ConstraintRecord],
    k: int = 3,
) -> DeliberationResult:
    """Generate K counterfactual candidates, score them, and choose one.

    This function is **side-effect free**: it does not write memory,
    invoke tools, or append constraints.  It only reads goals, input,
    and existing constraints to produce scored candidates.

    Parameters
    ----------
    goals:
        Ranked list of goals for this turn.
    user_input:
        Raw user input string.
    active_constraints:
        Currently active constraint records.
    k:
        Number of candidates to generate (minimum 2).

    Returns
    -------
    DeliberationResult
        Contains all K candidates with scores, the chosen one, and
        rejection reasons for the others.
    """
    k = max(k, 2)

    # Build base candidates from the existing deterministic pipeline
    base_candidates = build_action_candidates(goals, user_input)

    # Ensure we have at least K candidates by varying action selection
    cf_candidates: List[CFCandidate] = []
    seen_ids: set = set()

    for idx, candidate in enumerate(base_candidates):
        if len(cf_candidates) >= k:
            break
        if candidate.action_id in seen_ids:
            continue
        seen_ids.add(candidate.action_id)

        scores = _score_candidate(candidate, active_constraints, idx)
        cf_candidates.append(CFCandidate(
            candidate_id=idx,
            action=candidate,
            scores=scores,
        ))

    # If we still don't have K candidates, add synthetic alternatives
    synthetic_idx = len(cf_candidates)
    if len(cf_candidates) < k:
        # Add a conservative "ask_confirm" alternative
        ask_candidate = ActionCandidate("ask_confirm", "ASK_CONFIRM", 0.3)
        scores = _score_candidate(ask_candidate, active_constraints, synthetic_idx)
        cf_candidates.append(CFCandidate(
            candidate_id=synthetic_idx,
            action=ask_candidate,
            scores=scores,
        ))
        synthetic_idx += 1

    if len(cf_candidates) < k:
        # Add a "greet" safe fallback
        greet_candidate = ActionCandidate("greet", "SAFE_CHAT", 0.2)
        scores = _score_candidate(greet_candidate, active_constraints, synthetic_idx)
        cf_candidates.append(CFCandidate(
            candidate_id=synthetic_idx,
            action=greet_candidate,
            scores=scores,
        ))

    # Select the best candidate by total score (deterministic: tie-break by candidate_id)
    cf_candidates.sort(key=lambda c: (-c.scores.total, c.candidate_id))

    chosen = cf_candidates[0]
    chosen.is_chosen = True

    # Mark rejected candidates with reason
    for cand in cf_candidates[1:]:
        delta = chosen.scores.total - cand.scores.total
        if cand.scores.risk > 0.5:
            cand.reject_reason = f"high_risk ({cand.scores.risk:.2f})"
        elif delta > 0.0:
            cand.reject_reason = f"lower_score (delta={delta:.4f})"
        else:
            cand.reject_reason = "tie_broken_by_id"

    return DeliberationResult(
        candidates=cf_candidates,
        chosen=chosen,
        k=k,
    )


# ---------------------------------------------------------------------------
# Internal scoring
# ---------------------------------------------------------------------------

def _score_candidate(
    candidate: ActionCandidate,
    active_constraints: List[ConstraintRecord],
    rank: int,
) -> ScoreBreakdown:
    """Score a candidate on multiple objectives.

    Scoring is fully deterministic: same candidate + same constraints +
    same rank = same scores.
    """
    # Utility: base score from goal ranking (already deterministic)
    utility = clamp_round(candidate.score, 0.0, 1.0, 4)

    # Risk: determined by action class
    risk = _RISK_MAP.get(candidate.action_class, 0.5)

    # Cost: simple heuristic — risky actions cost more
    cost = clamp_round(risk * 0.5 + rank * 0.05, 0.0, 1.0, 4)

    # Constraint fit: check how well this action satisfies constraints
    constraint_fit = _compute_constraint_fit(candidate, active_constraints)

    return ScoreBreakdown(
        utility=utility,
        risk=risk,
        cost=cost,
        constraint_fit=constraint_fit,
    )


def _compute_constraint_fit(
    candidate: ActionCandidate,
    constraints: List[ConstraintRecord],
) -> float:
    """Compute how well a candidate fits active constraints (0–1)."""
    if not constraints:
        return 1.0  # no constraints = perfect fit

    violations = 0
    total_checks = 0

    for constraint in constraints:
        if constraint.status != "active":
            continue
        for effect in constraint.effects:
            total_checks += 1
            if effect.type == "forbid":
                target = effect.params.get("action_class", "")
                if candidate.action_class == target:
                    violations += 1
            elif effect.type == "increase_caution":
                if candidate.action_class in {"HIGH_IMPACT", "RISKY_SIDE_EFFECT"}:
                    violations += 0.5  # partial violation
            elif effect.type == "require_confirmation":
                target = effect.params.get("action_class", "")
                if candidate.action_class == target:
                    violations += 0.3  # mild penalty

    if total_checks == 0:
        return 1.0

    return clamp_round(1.0 - (violations / total_checks), 0.0, 1.0, 4)
