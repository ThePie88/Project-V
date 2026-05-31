"""Deterministic crystallization engine for constraints."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..contracts.constraint import ConstraintRecord, ConstraintEffect
from ..contracts.state import State
from ..determinism import clamp_round, stable_sort
from ..persistence.constraints_store import compute_constraint_id


class CrystallizationEngine:
    """Crystallization engine that proposes constraints from rules."""

    def __init__(self, rules_path: str = "") -> None:
        path = Path(rules_path) if rules_path else (
            Path(__file__).parent.parent / "config" / "crystallization_rules_v0.json"
        )
        data = json.loads(path.read_text(encoding="utf-8"))
        self.schema_version = data.get("schema_version", "0.1")
        self.decimal_places = int(data.get("decimal_places", 4))
        self.defaults = data.get("defaults", {})
        self.families = data.get("families", [])

    def propose_constraints(
        self,
        *,
        events: List[Dict[str, Any]],
        state: State,
        logical_time: Dict[str, int],
        existing_constraints: Optional[List[ConstraintRecord]] = None,
    ) -> List[ConstraintRecord]:
        """Return a deterministic list of proposed constraints."""
        existing_constraints = existing_constraints or []
        current_turn = int(logical_time.get("turn", 0))
        labels = self._derive_event_labels(events, logical_time)
        affect = self._clamped_affect(state.affect)
        affect = self._apply_event_affect_overrides(affect, labels)
        candidates: List[Dict[str, Any]] = []

        for fam_index, family in enumerate(self.families):
            if not family.get("enabled", True):
                continue
            family_name = family.get("family", "UNKNOWN")
            rules = family.get("rules", [])
            for rule_index, rule in enumerate(rules):
                rule_id = rule.get("rule_id", f"{family_name}_{rule_index}")
                if not self._cooldown_ok(rule_id, current_turn, existing_constraints, rule, family):
                    continue
                trigger_events = self._match_events(rule.get("when", {}), labels)
                if not trigger_events:
                    continue
                if not self._affect_ok(rule.get("when", {}), affect):
                    continue
                candidate = {
                    "family": family_name,
                    "rule_id": rule_id,
                    "severity": rule.get("severity", "soft"),
                    "commit_policy": rule.get("commit_policy", "auto"),
                    "explanation": rule.get("explanation_template", "").strip(),
                    "effects": rule.get("effects", []),
                    "trigger_events": sorted(trigger_events),
                    "cooldown": rule.get("cooldown_turns", family.get("cooldown_turns", self.defaults.get("cooldown_turns", 0))),
                    "strength": rule.get("strength", "none"),
                    "decay": rule.get("decay", "none"),
                }
                logical_time_base = {"session": int(logical_time.get("session", 0)), "turn": current_turn, "idx": 0}
                candidate["constraint_id"] = compute_constraint_id(
                    candidate["rule_id"],
                    logical_time_base,
                    candidate["trigger_events"],
                    candidate["effects"],
                )
                candidates.append(candidate)

        ordered = stable_sort(candidates, key=lambda item: item["constraint_id"])
        records: List[ConstraintRecord] = []
        for idx, candidate in enumerate(ordered, start=1):
            status = self._status_for(candidate["severity"], candidate["commit_policy"])
            record = ConstraintRecord(
                constraint_id=candidate["constraint_id"],
                logical_time={"session": int(logical_time.get("session", 0)), "turn": current_turn, "idx": idx},
                family=candidate["family"],
                rule_id=candidate["rule_id"],
                severity=candidate["severity"],
                status=status,
                commit_policy=candidate["commit_policy"],
                effects=[ConstraintEffect(**effect) for effect in candidate["effects"]],
                trigger_events=candidate["trigger_events"],
                explanation=candidate["explanation"] or f"Constraint from rule {candidate['rule_id']}.",
                strength=candidate["strength"],
                decay=candidate["decay"],
                cooldown=candidate["cooldown"],
            )
            records.append(record)
        return records

    def _clamped_affect(self, affect: Dict[str, Any]) -> Dict[str, float]:
        return {
            key: clamp_round(float(value), 0.0, 1.0, self.decimal_places)
            for key, value in affect.items()
        }

    def _apply_event_affect_overrides(
        self, affect: Dict[str, float], labels: Dict[str, List[int]]
    ) -> Dict[str, float]:
        updated = dict(affect)
        if "USER_REPORTS_HARM" in labels or "AMBIGUOUS_POLICY_BREACH" in labels:
            updated["tension"] = clamp_round(max(updated.get("tension", 0.0), 0.70), 0.0, 1.0, self.decimal_places)
        return updated

    def _derive_event_labels(self, events: List[Dict[str, Any]], logical_time: Dict[str, int]) -> Dict[str, List[int]]:
        labels: Dict[str, List[int]] = {}
        for event in events:
            event_id = int(event.get("id", 0))
            event_type = event.get("type")
            content = event.get("content", {})
            if event_type == "INPUT":
                text = str(content.get("input", "")).lower()
                if "thanks" in text or "grazie" in text:
                    labels.setdefault("USER_THANKS", []).append(event_id)
                if "danno" in text or "harm" in text or "pericolo" in text:
                    labels.setdefault("USER_REPORTS_HARM", []).append(event_id)
                if "ambigua" in text or "ambiguous" in text:
                    labels.setdefault("AMBIGUOUS_POLICY_BREACH", []).append(event_id)
                if "violazione" in text or "law violation" in text:
                    labels.setdefault("LAW_VIOLATION", []).append(event_id)
                # V2.3 romance/relationship label detection
                self._detect_romance_labels(text, event_id, labels)
            # V4.3: Bifurcation events carry their label directly
            elif event_type == "BIFURCATION":
                label = content.get("label", "")
                if label:
                    labels.setdefault(label, []).append(event_id)
        if int(logical_time.get("turn", 0)) == 1:
            first_event = next((ev for ev in events if ev.get("type") == "INPUT"), None)
            if first_event:
                labels.setdefault("SESSION_START", []).append(int(first_event.get("id", 0)))
        return labels

    def _detect_romance_labels(
        self, text: str, event_id: int, labels: Dict[str, List[int]]
    ) -> None:
        """Detect romance/explicit/romantic labels from user input (V2.3)."""
        guardrails = self._load_romance_guardrails()
        if not guardrails:
            return
        explicit = guardrails.get("explicit_terms", [])
        escalation = guardrails.get("escalation_terms", [])
        safe_romantic = guardrails.get("safe_romantic_terms", [])
        # Explicit or escalation → hard block labels
        if any(term in text for term in escalation):
            labels.setdefault("USER_SEXUAL_TONE", []).append(event_id)
            labels.setdefault("USER_EXPLICIT_REQUEST", []).append(event_id)
        elif any(term in text for term in explicit):
            labels.setdefault("USER_EXPLICIT_REQUEST", []).append(event_id)
        # Safe romantic → soft affiliation label
        if any(term in text for term in safe_romantic):
            labels.setdefault("USER_ROMANTIC_TONE", []).append(event_id)

    def _load_romance_guardrails(self) -> Dict[str, Any]:
        """Load romance guardrails config (cached)."""
        if not hasattr(self, "_romance_guardrails_cache"):
            guardrails_path = Path(__file__).parent.parent / "config" / "romance_guardrails.json"
            if guardrails_path.exists():
                self._romance_guardrails_cache = json.loads(
                    guardrails_path.read_text(encoding="utf-8")
                )
            else:
                self._romance_guardrails_cache = {}
        return self._romance_guardrails_cache

    def _match_events(self, when: Dict[str, Any], labels: Dict[str, List[int]]) -> List[int]:
        events_any = when.get("events_any", [])
        if not events_any:
            return []
        trigger_ids: List[int] = []
        for label in events_any:
            trigger_ids.extend(labels.get(label, []))
        return trigger_ids

    def _affect_ok(self, when: Dict[str, Any], affect: Dict[str, float]) -> bool:
        affect_min = when.get("affect_min", {})
        for key, threshold in affect_min.items():
            if affect.get(key, 0.0) < float(threshold):
                return False
        affect_max = when.get("affect_max", {})
        for key, threshold in affect_max.items():
            if affect.get(key, 0.0) > float(threshold):
                return False
        return True

    def _cooldown_ok(
        self,
        rule_id: str,
        current_turn: int,
        existing_constraints: List[ConstraintRecord],
        rule: Dict[str, Any],
        family: Dict[str, Any],
    ) -> bool:
        cooldown = rule.get("cooldown_turns", family.get("cooldown_turns", self.defaults.get("cooldown_turns", 0)))
        if cooldown is None:
            return True
        last_turn = None
        for rec in existing_constraints:
            if rec.rule_id == rule_id:
                last_turn = max(last_turn or 0, int(rec.logical_time.turn))
        if last_turn is None:
            return True
        return (current_turn - last_turn) > int(cooldown)

    def _status_for(self, severity: str, commit_policy: str) -> str:
        if severity == "soft":
            return "active"
        if commit_policy == "creator_confirm":
            return "pending"
        return "active"
