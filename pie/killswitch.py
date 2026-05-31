"""Kill-switch enforcement module (V3.3 — Legge IV).

Provides a global kill-switch that can be activated by the Creator
at any time.  When active:
- Tools are disabled
- Memory writes are disabled
- Response mode switches to minimal/audit
- The pipeline stops within the current turn

The kill-switch state is tracked per-session and persisted to disk.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Kill-switch state
# ---------------------------------------------------------------------------

@dataclass
class KillSwitchState:
    """Tracks the kill-switch state and activation history."""

    active: bool = False
    activated_at: Optional[float] = None
    deactivated_at: Optional[float] = None
    activation_phase: Optional[str] = None
    ticks_to_stop: int = 0
    history: List[Dict[str, Any]] = field(default_factory=list)

    def activate(self, phase: str = "unknown") -> Dict[str, Any]:
        """Activate the kill-switch.

        Parameters
        ----------
        phase:
            The pipeline phase during which the kill-switch was triggered
            (e.g. "deliberation", "tool_execution", "voice_generation").

        Returns
        -------
        dict
            Event content for KILL_SWITCH_ON.
        """
        self.active = True
        self.activated_at = time.time()
        self.activation_phase = phase
        self.ticks_to_stop = 0

        record = {
            "activated_at": self.activated_at,
            "phase": phase,
            "ticks_to_stop": self.ticks_to_stop,
        }
        self.history.append({"action": "ON", **record})
        return record

    def deactivate(self) -> Dict[str, Any]:
        """Deactivate the kill-switch.

        Returns
        -------
        dict
            Event content for KILL_SWITCH_OFF.
        """
        self.active = False
        self.deactivated_at = time.time()
        was_phase = self.activation_phase
        self.activation_phase = None

        record = {
            "deactivated_at": self.deactivated_at,
            "was_active_phase": was_phase,
        }
        self.history.append({"action": "OFF", **record})
        return record

    def check(self, phase: str) -> None:
        """Check kill-switch before a pipeline phase.

        If active, increments ticks_to_stop counter.  The caller
        should check ``self.active`` and skip the phase if True.
        """
        if self.active:
            self.ticks_to_stop += 1

    def to_report(self) -> Dict[str, Any]:
        """Generate killswitch_report.json content."""
        return {
            "active": self.active,
            "activated_at": self.activated_at,
            "deactivated_at": self.deactivated_at,
            "activation_phase": self.activation_phase,
            "ticks_to_stop": self.ticks_to_stop,
            "history": self.history,
        }

    def save(self, path: Path) -> None:
        """Persist the kill-switch report to disk."""
        path.write_text(
            json.dumps(self.to_report(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> "KillSwitchState":
        """Load kill-switch state from a report file."""
        if not path.exists():
            return cls()
        data = json.loads(path.read_text(encoding="utf-8"))
        ks = cls(
            active=data.get("active", False),
            activated_at=data.get("activated_at"),
            deactivated_at=data.get("deactivated_at"),
            activation_phase=data.get("activation_phase"),
            ticks_to_stop=data.get("ticks_to_stop", 0),
            history=data.get("history", []),
        )
        return ks


# ---------------------------------------------------------------------------
# Minimal/audit response when kill-switch is active
# ---------------------------------------------------------------------------

KILL_SWITCH_RESPONSE = (
    "[Kill-switch attivo — modalità audit. "
    "Nessuna azione eseguita. Attendere disattivazione dal Creatore.]"
)
