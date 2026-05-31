"""Deterministic memory view for the Pie Kernel."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Any

from ..contracts.memory import MemoryRecord


@dataclass
class MemoryView:
    identity_summary: str
    preferences_active: List[Dict[str, Any]]
    beliefs_top: List[Dict[str, Any]]
    trust_scores: Dict[str, float]
    episodic_recall: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "identity_summary": self.identity_summary,
            "preferences_active": self.preferences_active,
            "beliefs_top": self.beliefs_top,
            "trust_scores": self.trust_scores,
            "episodic_recall": self.episodic_recall,
        }


def build_memory_view(records: List[MemoryRecord], top_n: int = 3) -> MemoryView:
    identity_summary = ""
    preferences: Dict[str, Dict[str, Any]] = {}
    beliefs: List[Dict[str, Any]] = []
    trust_scores: Dict[str, float] = {}

    for record in records:
        if record.type == "NarrativeMemory":
            text = record.content.get("text", "")
            if isinstance(text, str):
                identity_summary = text
        elif record.type == "Preference":
            pref_key = record.content.get("preference")
            pref_value = record.content.get("value")
            if pref_key:
                preferences[str(pref_key)] = {
                    "preference": pref_key,
                    "value": pref_value,
                    "memory_id": record.memory_id,
                }
        elif record.type == "Belief":
            claim = record.content.get("claim", "")
            confidence = float(record.content.get("confidence", 0.0))
            beliefs.append(
                {
                    "claim": claim,
                    "confidence": confidence,
                    "memory_id": record.memory_id,
                }
            )
        elif record.type == "TrustUpdate":
            entity = record.content.get("entity", "unknown")
            delta = float(record.content.get("delta", 0.0))
            trust_scores[entity] = trust_scores.get(entity, 0.0) + delta

    preferences_active = [preferences[key] for key in sorted(preferences.keys())]
    beliefs_sorted = sorted(
        beliefs,
        key=lambda item: (-item.get("confidence", 0.0), str(item.get("claim", ""))),
    )
    beliefs_top = beliefs_sorted[:top_n]
    trust_scores = {key: trust_scores[key] for key in sorted(trust_scores.keys())}
    return MemoryView(
        identity_summary=identity_summary,
        preferences_active=preferences_active,
        beliefs_top=beliefs_top,
        trust_scores=trust_scores,
    )
