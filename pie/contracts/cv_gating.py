"""CV_GATING event contract — observable causal authority (V6a-R5).

Emitted at each of the 5 gating points in runtime.py where a
ControlVector channel affects a pipeline decision.

Invariant: count(CV_GATING events) == count(gating decisions).
No gating without trace, no trace without gating.
"""

from __future__ import annotations

from typing import Any, Dict

from pydantic import BaseModel, Field


class CVGatingEvent(BaseModel):
    """Single CV gating event — one channel affecting one decision."""

    canale: str = Field(
        ...,
        description="CV channel name: tool_gate, memory_gate, verbosity_bias, "
                    "consolidation_urgency, cf_k",
    )
    valore: float = Field(
        ...,
        description="CV channel value at decision time",
    )
    decisione: str = Field(
        ...,
        description="Decision outcome: BLOCK_TOOLS, ALLOW_TOOLS, etc.",
    )
    effetto: str = Field(
        ...,
        description="Concrete effect: tools_disabled, recall_k=3, etc.",
    )
    reason: str = Field(
        ...,
        description="Decision rule: tool_gate < 0.3, memory_gate * 10, etc.",
    )

    def to_trace_dict(self) -> Dict[str, Any]:
        """Convert to trace-compatible dict."""
        return {
            "type": "CV_GATING",
            "canale": self.canale,
            "valore": round(self.valore, 4),
            "decisione": self.decisione,
            "effetto": self.effetto,
            "reason": self.reason,
        }


# ---------------------------------------------------------------------------
# Factory functions for each gating point
# ---------------------------------------------------------------------------

def gate_tools(tool_gate: float) -> CVGatingEvent:
    """Create CV_GATING event for tool access decision."""
    blocked = tool_gate < 0.3
    return CVGatingEvent(
        canale="tool_gate",
        valore=tool_gate,
        decisione="BLOCK_TOOLS" if blocked else "ALLOW_TOOLS",
        effetto="tools_disabled" if blocked else "tools_enabled",
        reason=f"tool_gate={'<' if blocked else '>='} 0.3",
    )


def gate_memory(memory_gate: float) -> CVGatingEvent:
    """Create CV_GATING event for recall top_k decision."""
    recall_k = max(1, min(10, int(memory_gate * 10)))
    return CVGatingEvent(
        canale="memory_gate",
        valore=memory_gate,
        decisione=f"RECALL_K_{recall_k}",
        effetto=f"recall_k={recall_k}",
        reason=f"max(1, min(10, int({memory_gate:.4f} * 10)))",
    )


def gate_verbosity(verbosity_bias: float) -> CVGatingEvent:
    """Create CV_GATING event for verbosity decision."""
    if verbosity_bias < 0.3:
        dec = "VERBOSITY_LOW"
        eff = "verbosity=low, max_tokens<=40"
    elif verbosity_bias > 0.7:
        dec = "VERBOSITY_HIGH"
        eff = "verbosity=med, max_tokens<=80"
    else:
        dec = "VERBOSITY_NORMAL"
        eff = "verbosity=unchanged"
    return CVGatingEvent(
        canale="verbosity_bias",
        valore=verbosity_bias,
        decisione=dec,
        effetto=eff,
        reason=f"verbosity_bias={verbosity_bias:.4f}",
    )


def gate_consolidation(consolidation_urgency: float) -> CVGatingEvent:
    """Create CV_GATING event for consolidation decision."""
    urgent = consolidation_urgency > 0.7
    return CVGatingEvent(
        canale="consolidation_urgency",
        valore=consolidation_urgency,
        decisione="CONSOLIDATE_NOW" if urgent else "CONSOLIDATE_NORMAL",
        effetto="forced_consolidation" if urgent else "normal_schedule",
        reason=f"consolidation_urgency={'>' if urgent else '<='} 0.7",
    )


def gate_counterfactual(cf_k: int, budget_k: int) -> CVGatingEvent:
    """Create CV_GATING event for counterfactual deliberation."""
    actual_k = min(cf_k, budget_k)
    return CVGatingEvent(
        canale="cf_k",
        valore=float(cf_k),
        decisione=f"CF_K_{actual_k}",
        effetto=f"deliberate_k={actual_k}",
        reason=f"min(cv.cf_k={cf_k}, budget={budget_k})",
    )


# ---------------------------------------------------------------------------
# Gating decision snapshot (for E2E testing)
# ---------------------------------------------------------------------------

class GatingSnapshot:
    """Snapshot of all 5 gating decisions from a single pipeline run.

    Used for deterministic E2E testing: same state → same gating decisions.
    """

    def __init__(
        self,
        tool_allowed: bool,
        recall_k: int,
        verbosity: str,
        consolidate_now: bool,
        cf_k: int,
    ):
        self.tool_allowed = tool_allowed
        self.recall_k = recall_k
        self.verbosity = verbosity
        self.consolidate_now = consolidate_now
        self.cf_k = cf_k

    @classmethod
    def from_cv(cls, tool_gate: float, memory_gate: float,
                verbosity_bias: float, consolidation_urgency: float,
                cf_k: int, budget_k: int = 5) -> "GatingSnapshot":
        """Compute all 5 gating decisions from CV channel values."""
        return cls(
            tool_allowed=tool_gate >= 0.3,
            recall_k=max(1, min(10, int(memory_gate * 10))),
            verbosity=(
                "low" if verbosity_bias < 0.3
                else "high" if verbosity_bias > 0.7
                else "normal"
            ),
            consolidate_now=consolidation_urgency > 0.7,
            cf_k=min(cf_k, budget_k),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool_allowed": self.tool_allowed,
            "recall_k": self.recall_k,
            "verbosity": self.verbosity,
            "consolidate_now": self.consolidate_now,
            "cf_k": self.cf_k,
        }

    def __eq__(self, other):
        if not isinstance(other, GatingSnapshot):
            return False
        return self.to_dict() == other.to_dict()

    def __repr__(self):
        return f"GatingSnapshot({self.to_dict()})"
