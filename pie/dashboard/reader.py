"""Read-only data layer for session inspection (V6b-E3).

DashboardReader NEVER writes to disk. It reads session files
(JSON, JSONL) and returns structured dicts for the API layer.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


class DashboardReader:
    """Read-only access to session data. Never writes."""

    def __init__(self, session_dir: Path) -> None:
        self._dir = session_dir

    # -- Session metadata ---------------------------------------------------

    def session_meta(self) -> Dict[str, Any]:
        """Session metadata (id, seed, turn_count, created_at)."""
        path = self._dir / "session_meta.json"
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    # -- State snapshot -----------------------------------------------------

    def state_latest(self) -> Dict[str, Any]:
        """Current drives + affect snapshot."""
        path = self._dir / "state_latest.json"
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    # -- Journal events -----------------------------------------------------

    def journal_events(
        self,
        event_type: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """All journal events, optionally filtered by type.

        Parameters
        ----------
        event_type:
            If set, only return events with matching ``type`` field.
        limit:
            Max number of events to return (most recent N).
        """
        path = self._dir / "journal.jsonl"
        if not path.exists():
            return []

        events: List[Dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                evt = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event_type is None or evt.get("type") == event_type:
                events.append(evt)

        if limit is not None and limit > 0:
            events = events[-limit:]

        return events

    # -- CV Gating ----------------------------------------------------------

    def cv_gating_table(self) -> List[Dict[str, Any]]:
        """CV_GATING decisions as flat list.

        Each entry has: canale, valore, decisione, effetto, reason.
        """
        gating_events = self.journal_events(event_type="CV_GATING")
        rows: List[Dict[str, Any]] = []
        for evt in gating_events:
            # Support both formats:
            # 1. Event.new style: fields inside evt["content"]
            # 2. Raw dict style: fields at top level alongside "type"
            content = evt.get("content", {})
            if content.get("canale"):
                src = content
            else:
                src = evt
            rows.append({
                "event_id": evt.get("id") or evt.get("event_id"),
                "canale": src.get("canale", ""),
                "valore": src.get("valore", 0.0),
                "decisione": src.get("decisione", ""),
                "effetto": src.get("effetto", ""),
                "reason": src.get("reason", ""),
            })
        return rows

    # -- CV channels time series --------------------------------------------

    def cv_channels_timeseries(self) -> Dict[str, List[float]]:
        """ControlVector channels over turns.

        Returns {channel_name: [val_t1, val_t2, ...]}.
        Extracted from STATE_UPDATED events → engine_metadata.control_vector.
        """
        state_events = self.journal_events(event_type="STATE_UPDATED")
        channels: Dict[str, List[float]] = {}

        for evt in state_events:
            content = evt.get("content", {})
            em = content.get("engine_metadata", {})
            cv = em.get("control_vector", {})
            if not cv:
                continue
            for ch_name, ch_val in cv.items():
                if ch_name not in channels:
                    channels[ch_name] = []
                try:
                    channels[ch_name].append(float(ch_val))
                except (TypeError, ValueError):
                    channels[ch_name].append(0.0)

        return channels

    # -- Spike rate ---------------------------------------------------------

    def spike_rate(self) -> Dict[str, List[int]]:
        """Spikes per neuron per turn.

        Returns {neuron_id: [count_t1, count_t2, ...]}.
        Extracted from STATE_UPDATED events → engine_metadata.spike_log.
        """
        state_events = self.journal_events(event_type="STATE_UPDATED")
        # Collect all neuron IDs across all turns first
        all_neurons: List[str] = []
        per_turn_counts: List[Dict[str, int]] = []

        for evt in state_events:
            content = evt.get("content", {})
            em = content.get("engine_metadata", {})
            spike_log = em.get("spike_log", [])

            turn_counts: Dict[str, int] = {}
            for spike in spike_log:
                nid = spike.get("neuron_id", "unknown")
                turn_counts[nid] = turn_counts.get(nid, 0) + 1
                if nid not in all_neurons:
                    all_neurons.append(nid)
            per_turn_counts.append(turn_counts)

        # Build result with zero-fill for missing neurons
        result: Dict[str, List[int]] = {}
        for nid in sorted(all_neurons):
            result[nid] = [
                tc.get(nid, 0) for tc in per_turn_counts
            ]

        return result

    # -- Budget -------------------------------------------------------------

    def budget_summary(self) -> Dict[str, Any]:
        """Budget state from metabolism.json."""
        path = self._dir / "metabolism.json"
        if not path.exists():
            return {}
        data = json.loads(path.read_text(encoding="utf-8"))
        budget = data.get("budget", {})
        return {
            "initial_budget": budget.get("initial_budget", 0),
            "current_budget": budget.get("current_budget", 0),
            "consumed": budget.get("consumed", 0),
            "remaining_ratio": budget.get("remaining_ratio", 1.0),
        }

    def budget_timeseries(self) -> List[Dict[str, Any]]:
        """Per-turn cost breakdown from consumption_history."""
        path = self._dir / "metabolism.json"
        if not path.exists():
            return []
        data = json.loads(path.read_text(encoding="utf-8"))
        budget = data.get("budget", {})
        return budget.get("consumption_history", [])

    # -- Tool stats ---------------------------------------------------------

    def tool_stats(self) -> Dict[str, Any]:
        """Tool call count, deny count, deny rate, deny reasons breakdown."""
        events = self.journal_events()
        tool_calls = 0
        tool_denied = 0
        deny_reasons: Dict[str, int] = {}

        for evt in events:
            etype = evt.get("type", "")
            if etype == "TOOL_CALL":
                tool_calls += 1
            elif etype == "TOOL_DENIED":
                tool_denied += 1
                content = evt.get("content", {})
                reason = content.get("deny_reason") or "unknown"
                deny_reasons[reason] = deny_reasons.get(reason, 0) + 1

        total = tool_calls + tool_denied
        deny_rate = (tool_denied / total) if total > 0 else 0.0

        return {
            "tool_calls": tool_calls,
            "tool_denied": tool_denied,
            "total": total,
            "deny_rate": round(deny_rate, 4),
            "deny_reasons": deny_reasons,
        }

    # -- Memory counts ------------------------------------------------------

    def memory_counts(self) -> Dict[str, int]:
        """Memory store size by type."""
        path = self._dir / "memory.jsonl"
        if not path.exists():
            return {}

        counts: Dict[str, int] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            mtype = rec.get("type", "Unknown")
            counts[mtype] = counts.get(mtype, 0) + 1

        return counts
