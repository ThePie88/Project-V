"""ControlVector — neural network output that drives kernel pipeline.

The ControlVector is produced by the Tier-2 reservoir and consumed by
the runtime to modulate memory recall, tool usage, deliberation depth,
verbosity, and consolidation urgency.

All fields are bounded and deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class ControlVector:
    """Neural control output for kernel pipeline modulation.

    Produced by the ESN reservoir readout layer.
    Consumed by runtime._process_turn() to gate subsystems.
    """

    memory_gate: float = 0.5         # [0,1] how much to trust episodic recall
    tool_gate: float = 0.5           # [0,1] availability to use tools
    cf_k: int = 3                    # [1,5] counterfactual alternatives to evaluate
    verbosity_bias: float = 0.5      # [0,1] hint for speech plan length
    consolidation_urgency: float = 0.0  # [0,1] urgency of memory consolidation

    def to_dict(self) -> Dict[str, Any]:
        return {
            "memory_gate": self.memory_gate,
            "tool_gate": self.tool_gate,
            "cf_k": self.cf_k,
            "verbosity_bias": self.verbosity_bias,
            "consolidation_urgency": self.consolidation_urgency,
        }

    @classmethod
    def from_raw(cls, raw: Dict[str, float]) -> ControlVector:
        """Create ControlVector from raw reservoir output, clamping to valid ranges."""
        return cls(
            memory_gate=max(0.0, min(1.0, raw.get("memory_gate", 0.5))),
            tool_gate=max(0.0, min(1.0, raw.get("tool_gate", 0.5))),
            cf_k=max(1, min(5, int(round(raw.get("cf_k", 3.0))))),
            verbosity_bias=max(0.0, min(1.0, raw.get("verbosity_bias", 0.5))),
            consolidation_urgency=max(0.0, min(1.0, raw.get("consolidation_urgency", 0.0))),
        )
