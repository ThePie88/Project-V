"""Journal consolidation — episodic → narrative/semantic/trust.

Every N turns or on affect spike, the consolidation loop:
1. Segments recent journal into episodes
2. Extracts facts (actions, preferences, entities, emotions)
3. Produces a narrative digest
4. Generates MemoryRecord proposals with confidence + decay

No LLM — deterministic extraction via patterns and frequency.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional

from ..contracts.memory import MemoryRecord
from ..persistence.memory_store import compute_memory_id
from .recall import Episode, load_episodes_from_journal, _tokenize

# ── triggers ──────────────────────────────────────────────────────────
CONSOLIDATION_INTERVAL = 5  # every N turns
AFFECT_SPIKE_THRESHOLD = 0.6  # arousal or tension above this → immediate


def should_consolidate(
    turn_count: int,
    last_consolidation_turn: int,
    affect: Optional[Dict[str, float]] = None,
    urgency: float = 0.0,
) -> bool:
    """Check if consolidation should run now."""
    if turn_count - last_consolidation_turn >= CONSOLIDATION_INTERVAL:
        return True
    # V4: ControlVector consolidation_urgency can force early consolidation
    if urgency > 0.7:
        return True
    if affect:
        if affect.get("arousal", 0.0) > AFFECT_SPIKE_THRESHOLD:
            return True
        if affect.get("tension", 0.0) > AFFECT_SPIKE_THRESHOLD:
            return True
    return False


# ── fact extraction patterns (IT + EN) ────────────────────────────────

_FACT_PATTERNS: List[tuple] = [
    # Italian actions: "ho mangiato X", "ho comprato X"
    (r"\bho\s+(\w+)\s+(.+)", "action"),
    # Italian locations: "sono andato a X"
    (r"\bsono\s+(?:andato|andata|stato|stata)\s+(?:a|al|alla|in)\s+(.+)", "location"),
    # Italian preferences: "mi piace X", "adoro X"
    (r"\bmi\s+piace\s+(.+)", "preference"),
    (r"\b(?:adoro|amo)\s+(.+)", "preference"),
    (r"\b(?:odio|detesto)\s+(.+)", "dislike"),
    # Italian occupation: "lavoro come X", "studio X"
    (r"\blavoro\s+(?:come|a|per)\s+(.+)", "occupation"),
    (r"\bstudio\s+(.+)", "study"),
    # Named entities: "si chiama X"
    (r"\bsi\s+chiama\s+(\w+)", "named_entity"),
    # Italian identity: "mi chiamo X", "ho N anni"
    (r"\bmi\s+chiamo\s+(\w+)", "user_name"),
    (r"\bho\s+(\d+)\s+anni", "user_age"),
    # English fallbacks
    (r"\bi\s+(?:ate|had|cooked)\s+(.+)", "action"),
    (r"\bi\s+(?:went|traveled|visited)\s+(?:to\s+)?(.+)", "location"),
    (r"\bi\s+(?:like|love|enjoy)\s+(.+)", "preference"),
    (r"\bi\s+(?:hate|dislike)\s+(.+)", "dislike"),
    (r"\bi\s+(?:work|worked)\s+(?:as|at|for)\s+(.+)", "occupation"),
    (r"\bmy\s+name\s+is\s+(\w+)", "user_name"),
]

_EMOTIONAL_MARKERS = [
    "felice", "triste", "arrabbiato", "preoccupato", "contento",
    "stanco", "annoiato", "entusiasta", "deluso", "sorpreso",
    "happy", "sad", "angry", "worried", "tired", "excited",
]

_TRUST_POSITIVE = [
    "grazie", "bravo", "brava", "perfetto", "esatto", "giusto",
    "hai ragione", "thanks", "great", "correct", "exactly",
]
_TRUST_NEGATIVE = [
    "sbagliato", "non è vero", "errore", "male",
    "wrong", "incorrect", "no that's wrong",
]


@dataclass
class ExtractedFact:
    """A fact extracted from an episode."""
    text: str
    category: str
    confidence: float
    source_turn: int


def extract_facts(episodes: List[Episode]) -> List[ExtractedFact]:
    """Extract structured facts from episodes using pattern matching."""
    facts: List[ExtractedFact] = []
    for ep in episodes:
        lower = ep.user_input.lower().strip()
        # Pattern-based extraction
        for pattern, category in _FACT_PATTERNS:
            match = re.search(pattern, lower)
            if match:
                captured = match.group(match.lastindex) if match.lastindex else match.group(0)
                captured = captured.strip().rstrip(".,!?;:")
                if len(captured) > 1:
                    facts.append(ExtractedFact(
                        text=f"{category}: {captured}",
                        category=category,
                        confidence=0.7,
                        source_turn=ep.turn,
                    ))
        # Emotional state
        for marker in _EMOTIONAL_MARKERS:
            if marker in lower:
                facts.append(ExtractedFact(
                    text=f"emotion: {marker}",
                    category="emotion",
                    confidence=0.6,
                    source_turn=ep.turn,
                ))
                break
        # Trust signals
        for sig in _TRUST_POSITIVE:
            if sig in lower:
                facts.append(ExtractedFact(
                    text=f"trust_positive: {sig}",
                    category="trust_positive",
                    confidence=0.5,
                    source_turn=ep.turn,
                ))
                break
        for sig in _TRUST_NEGATIVE:
            if sig in lower:
                facts.append(ExtractedFact(
                    text=f"trust_negative: {sig}",
                    category="trust_negative",
                    confidence=0.5,
                    source_turn=ep.turn,
                ))
                break
    return facts


def _build_narrative_digest(episodes: List[Episode], facts: List[ExtractedFact]) -> str:
    """Compressed summary — not a transcript."""
    if not episodes:
        return ""
    turn_range = f"{episodes[0].turn}-{episodes[-1].turn}"
    seen: set = set()
    key_facts = []
    for f in facts:
        if f.text not in seen and f.category not in ("trust_positive", "trust_negative"):
            key_facts.append(f.text)
            seen.add(f.text)
    term_freq: Dict[str, int] = {}
    for ep in episodes:
        for token in set(_tokenize(ep.user_input)):
            term_freq[token] = term_freq.get(token, 0) + 1
    top_terms = sorted(term_freq.items(), key=lambda x: (-x[1], x[0]))[:5]
    topics = ", ".join(t[0] for t in top_terms)
    parts = [f"Turni {turn_range}"]
    if key_facts:
        parts.append("; ".join(key_facts[:5]))
    if topics:
        parts.append(f"topic: {topics}")
    return ". ".join(parts) + "."


def consolidate(
    journal_path: Path,
    since_turn: int,
    logical_time: Dict[str, Any],
    source_refs: Optional[List[int]] = None,
) -> List[MemoryRecord]:
    """Run consolidation on journal entries since last consolidation.

    Produces MemoryRecord proposals:
    - 1 NarrativeMemory (episode digest)
    - N Beliefs (extracted facts with confidence)
    - TrustUpdates (net trust delta from signals)
    """
    all_eps = load_episodes_from_journal(journal_path)
    episodes = [ep for ep in all_eps if ep.turn > since_turn]
    if not episodes:
        return []

    facts = extract_facts(episodes)
    records: List[MemoryRecord] = []
    ep_refs: List[int] = []
    for ep in episodes:
        ep_refs.extend(ep.event_ids)
    if not ep_refs:
        ep_refs = source_refs[:1] if source_refs else [0]
    # Cap source_refs length for schema compliance
    refs_capped = ep_refs[:10]
    idx = 0

    # 1. Narrative digest
    digest = _build_narrative_digest(episodes, facts)
    if digest:
        idx += 1
        rt = {
            "session": int(logical_time.get("session", 0)),
            "turn": int(logical_time.get("turn", 0)),
            "idx": idx,
        }
        mid = compute_memory_id(rt, "NarrativeMemory", {"text": digest}, refs_capped)
        records.append(MemoryRecord(
            memory_id=mid,
            logical_time=rt,
            type="NarrativeMemory",
            content={"text": digest},
            source_refs=refs_capped,
            tags=["narrative", "consolidation"],
            context={"range": f"{episodes[0].turn}-{episodes[-1].turn}"},
        ))

    # 2. Beliefs from extracted facts (deduplicated)
    seen_facts: set = set()
    for fact in facts:
        if fact.category in ("trust_positive", "trust_negative", "emotion"):
            continue
        if fact.text in seen_facts:
            continue
        seen_facts.add(fact.text)
        idx += 1
        rt = {
            "session": int(logical_time.get("session", 0)),
            "turn": int(logical_time.get("turn", 0)),
            "idx": idx,
        }
        content = {"claim": fact.text, "confidence": fact.confidence}
        mid = compute_memory_id(rt, "Belief", content, refs_capped)
        records.append(MemoryRecord(
            memory_id=mid,
            logical_time=rt,
            type="Belief",
            content=content,
            source_refs=refs_capped,
            tags=["belief", "consolidation", fact.category],
            context={"source_turn": fact.source_turn},
        ))

    # 3. Trust delta
    trust_pos = sum(1 for f in facts if f.category == "trust_positive")
    trust_neg = sum(1 for f in facts if f.category == "trust_negative")
    net_trust = round((trust_pos - trust_neg) * 0.05, 4)
    if abs(net_trust) > 0.001:
        idx += 1
        rt = {
            "session": int(logical_time.get("session", 0)),
            "turn": int(logical_time.get("turn", 0)),
            "idx": idx,
        }
        content = {"entity": "user", "delta": net_trust}
        mid = compute_memory_id(rt, "TrustUpdate", content, refs_capped)
        records.append(MemoryRecord(
            memory_id=mid,
            logical_time=rt,
            type="TrustUpdate",
            content=content,
            source_refs=refs_capped,
            tags=["trust", "consolidation"],
            context={"positive": trust_pos, "negative": trust_neg},
        ))

    return records
