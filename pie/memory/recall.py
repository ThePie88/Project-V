"""Episodic recall — BM25 retrieval from journal with state bias.

Implements the recall loop: every turn, before the kernel speaks,
retrieve the most relevant past episodes from the journal.  This
transforms the dead journal log into living episodic memory.

No LLM, no embeddings — pure deterministic retrieval.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional


@dataclass
class Episode:
    """A recalled episode from the journal."""

    turn: int
    user_input: str
    llm_output: str
    event_ids: List[int] = field(default_factory=list)
    score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "turn": self.turn,
            "user_input": self.user_input,
            "llm_output": self.llm_output,
            "score": round(self.score, 4),
        }


def _tokenize(text: str) -> List[str]:
    """Lowercase, split on non-alpha, drop tokens < 3 chars (kills stopwords)."""
    return [t for t in re.findall(r"[a-zà-ú0-9]+", text.lower()) if len(t) >= 3]


def load_episodes_from_journal(journal_path: Path) -> List[Episode]:
    """Parse journal.jsonl and extract turn-level episodes."""
    if not journal_path.exists():
        return []
    lines = journal_path.read_text(encoding="utf-8").splitlines()
    by_turn: Dict[int, Dict[str, Any]] = {}
    for line in lines:
        if not line.strip():
            continue
        ev = json.loads(line)
        turn = ev.get("content", {}).get("logical_time", {}).get("turn")
        if turn is None:
            continue
        if turn not in by_turn:
            by_turn[turn] = {"inputs": [], "outputs": [], "ids": []}
        ev_type = ev.get("type", "")
        if ev_type == "INPUT":
            by_turn[turn]["inputs"].append(ev["content"].get("input", ""))
            by_turn[turn]["ids"].append(ev.get("id", 0))
        elif ev_type in ("LLM_OUTPUT", "LLM_FALLBACK"):
            by_turn[turn]["outputs"].append(ev["content"].get("text", ""))
            by_turn[turn]["ids"].append(ev.get("id", 0))
    episodes = []
    for turn in sorted(by_turn):
        d = by_turn[turn]
        user = " ".join(d["inputs"]).strip()
        if user:
            episodes.append(Episode(
                turn=turn,
                user_input=user,
                llm_output=" ".join(d["outputs"]).strip(),
                event_ids=d["ids"],
            ))
    return episodes


def _compute_idf(episodes: List[Episode]) -> Dict[str, float]:
    """Inverse document frequency across all episodes."""
    n = len(episodes)
    if n == 0:
        return {}
    doc_freq: Dict[str, int] = {}
    for ep in episodes:
        tokens = set(_tokenize(ep.user_input + " " + ep.llm_output))
        for t in tokens:
            doc_freq[t] = doc_freq.get(t, 0) + 1
    return {
        term: math.log((n - df + 0.5) / (df + 0.5) + 1.0)
        for term, df in doc_freq.items()
    }


def _bm25_score(
    query_tokens: List[str],
    doc_tokens: List[str],
    idf: Dict[str, float],
    avg_dl: float,
    k1: float = 1.2,
    b: float = 0.75,
) -> float:
    """BM25 score for a query against a document."""
    dl = len(doc_tokens)
    if dl == 0 or avg_dl == 0:
        return 0.0
    tf: Dict[str, int] = {}
    for t in doc_tokens:
        tf[t] = tf.get(t, 0) + 1
    score = 0.0
    for qt in query_tokens:
        if qt not in tf:
            continue
        f = tf[qt]
        score += idf.get(qt, 0.0) * (f * (k1 + 1)) / (f + k1 * (1 - b + b * dl / avg_dl))
    return score


def _state_bias(episode: Episode, state: Dict[str, Any]) -> float:
    """State-driven retrieval bias.

    The kernel's current drives/affect influence what it 'remembers':
    - High curiosity  → boost question/learning episodes
    - High sociality  → boost personal/relational episodes
    - High tension    → boost conflict/danger episodes
    """
    drives = state.get("drives", {})
    affect = state.get("affect", {})
    lower = (episode.user_input + " " + episode.llm_output).lower()
    bias = 1.0
    curiosity = drives.get("curiosity", 0.5)
    if curiosity > 0.6 and any(w in lower for w in ["?", "come", "cosa", "perché", "spiega", "how", "what", "why"]):
        bias *= 1.0 + (curiosity - 0.6) * 0.5
    sociality = drives.get("sociality", 0.5)
    if sociality > 0.6 and any(w in lower for w in ["chiamo", "sono", "famiglia", "amico", "name", "friend"]):
        bias *= 1.0 + (sociality - 0.6) * 0.5
    tension = affect.get("tension", 0.15)
    if tension > 0.3 and any(w in lower for w in ["danno", "pericolo", "harm", "violazione", "problema"]):
        bias *= 1.0 + (tension - 0.3) * 0.8
    return round(bias, 4)


def recall(
    query: str,
    journal_path: Path,
    state: Optional[Dict[str, Any]] = None,
    top_k: int = 5,
    exclude_turn: Optional[int] = None,
) -> List[Episode]:
    """Retrieve the most relevant past episodes from the journal.

    BM25 scoring + state-driven bias + recency bonus.
    No LLM, no embeddings — fully deterministic.
    """
    episodes = load_episodes_from_journal(journal_path)
    if exclude_turn is not None:
        episodes = [ep for ep in episodes if ep.turn != exclude_turn]
    if not episodes:
        return []
    idf = _compute_idf(episodes)
    doc_lengths = [len(_tokenize(ep.user_input + " " + ep.llm_output)) for ep in episodes]
    avg_dl = sum(doc_lengths) / len(doc_lengths) if doc_lengths else 1.0
    query_tokens = _tokenize(query)
    if not query_tokens:
        return []
    state_snap = state or {}
    max_turn = max(ep.turn for ep in episodes)
    for ep in episodes:
        doc_tokens = _tokenize(ep.user_input + " " + ep.llm_output)
        bm25 = _bm25_score(query_tokens, doc_tokens, idf, avg_dl)
        bias = _state_bias(ep, state_snap) if state_snap else 1.0
        recency = 1.0 + 0.1 * (ep.turn / max_turn) if max_turn > 0 else 1.0
        ep.score = round(bm25 * bias * recency, 4)
    episodes.sort(key=lambda e: (-e.score, -e.turn))
    return [ep for ep in episodes if ep.score > 0.0][:top_k]
