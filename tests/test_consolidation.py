"""Tests for journal consolidation — fact extraction + narrative digest + trust."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pie.memory.consolidation import (
    should_consolidate,
    CONSOLIDATION_INTERVAL,
    AFFECT_SPIKE_THRESHOLD,
    extract_facts,
    _build_narrative_digest,
    consolidate,
    ExtractedFact,
)
from pie.memory.recall import Episode, load_episodes_from_journal


# ── Helpers ─────────────────────────────────────────────────────────

def _write_journal(path: Path, turns: list) -> None:
    """Write a minimal journal.jsonl from turn data.

    Each item in *turns* is (turn_num, user_input, llm_output).
    """
    eid = 1
    with path.open("w", encoding="utf-8") as f:
        for turn, user, llm_out in turns:
            ev_input = {
                "id": eid, "type": "INPUT",
                "timestamp": "2025-01-01T00:00:00Z",
                "content": {"logical_time": {"turn": turn, "step": 1}, "input": user},
            }
            f.write(json.dumps(ev_input) + "\n")
            eid += 1
            ev_output = {
                "id": eid, "type": "LLM_OUTPUT",
                "timestamp": "2025-01-01T00:00:00Z",
                "content": {"logical_time": {"turn": turn, "step": 6, "attempt": 1}, "text": llm_out},
            }
            f.write(json.dumps(ev_output) + "\n")
            eid += 1


# ── Trigger tests ──────────────────────────────────────────────────

class TestShouldConsolidate:
    def test_timer_trigger(self):
        assert should_consolidate(
            turn_count=CONSOLIDATION_INTERVAL + 1,
            last_consolidation_turn=0,
        ) is True

    def test_timer_not_yet(self):
        assert should_consolidate(
            turn_count=3,
            last_consolidation_turn=0,
        ) is False

    def test_affect_spike_arousal(self):
        assert should_consolidate(
            turn_count=1,
            last_consolidation_turn=0,
            affect={"arousal": AFFECT_SPIKE_THRESHOLD + 0.1},
        ) is True

    def test_affect_spike_tension(self):
        assert should_consolidate(
            turn_count=1,
            last_consolidation_turn=0,
            affect={"tension": AFFECT_SPIKE_THRESHOLD + 0.1},
        ) is True

    def test_no_trigger(self):
        assert should_consolidate(
            turn_count=1,
            last_consolidation_turn=0,
            affect={"arousal": 0.2, "tension": 0.1},
        ) is False


# ── Fact extraction tests ──────────────────────────────────────────

class TestExtractFacts:
    def test_italian_action(self):
        eps = [Episode(turn=1, user_input="ho mangiato carne", llm_output="ok")]
        facts = extract_facts(eps)
        assert any(f.category == "action" for f in facts)

    def test_italian_preference(self):
        eps = [Episode(turn=1, user_input="mi piace la pizza", llm_output="anch'io")]
        facts = extract_facts(eps)
        assert any(f.category == "preference" for f in facts)

    def test_italian_dislike(self):
        eps = [Episode(turn=1, user_input="odio il traffico", llm_output="capisco")]
        facts = extract_facts(eps)
        assert any(f.category == "dislike" for f in facts)

    def test_user_name(self):
        eps = [Episode(turn=1, user_input="mi chiamo Marco", llm_output="piacere")]
        facts = extract_facts(eps)
        assert any(f.category == "user_name" for f in facts)
        name_fact = [f for f in facts if f.category == "user_name"][0]
        assert "marco" in name_fact.text.lower()

    def test_user_age(self):
        eps = [Episode(turn=1, user_input="ho 25 anni", llm_output="ok")]
        facts = extract_facts(eps)
        assert any(f.category == "user_age" for f in facts)

    def test_emotion_detected(self):
        eps = [Episode(turn=1, user_input="sono felice oggi", llm_output="bene")]
        facts = extract_facts(eps)
        assert any(f.category == "emotion" for f in facts)

    def test_trust_positive(self):
        eps = [Episode(turn=1, user_input="grazie mille", llm_output="prego")]
        facts = extract_facts(eps)
        assert any(f.category == "trust_positive" for f in facts)

    def test_trust_negative(self):
        eps = [Episode(turn=1, user_input="è sbagliato quello che dici", llm_output="scusa")]
        facts = extract_facts(eps)
        assert any(f.category == "trust_negative" for f in facts)

    def test_english_preference(self):
        eps = [Episode(turn=1, user_input="i love pasta", llm_output="nice")]
        facts = extract_facts(eps)
        assert any(f.category == "preference" for f in facts)

    def test_no_facts_from_generic(self):
        eps = [Episode(turn=1, user_input="ciao", llm_output="ciao")]
        facts = extract_facts(eps)
        # "ciao" matches no patterns, no emotions, no trust
        assert len(facts) == 0

    def test_multiple_episodes(self):
        eps = [
            Episode(turn=1, user_input="mi chiamo Luca", llm_output="piacere"),
            Episode(turn=2, user_input="ho mangiato pesce", llm_output="buono"),
            Episode(turn=3, user_input="mi piace il mare", llm_output="bello"),
        ]
        facts = extract_facts(eps)
        categories = {f.category for f in facts}
        assert "user_name" in categories
        assert "action" in categories
        assert "preference" in categories


# ── Narrative digest tests ──────────────────────────────────────────

class TestNarrativeDigest:
    def test_produces_digest(self):
        eps = [
            Episode(turn=1, user_input="ho mangiato carne", llm_output="ok"),
            Episode(turn=2, user_input="mi piace la pizza", llm_output="anch'io"),
        ]
        facts = extract_facts(eps)
        digest = _build_narrative_digest(eps, facts)
        assert "Turni 1-2" in digest
        assert len(digest) > 10

    def test_empty_episodes(self):
        assert _build_narrative_digest([], []) == ""

    def test_includes_topics(self):
        eps = [
            Episode(turn=1, user_input="ho mangiato carne a pranzo", llm_output="buono"),
            Episode(turn=2, user_input="ho mangiato pesce a cena", llm_output="ottimo"),
        ]
        facts = extract_facts(eps)
        digest = _build_narrative_digest(eps, facts)
        assert "topic:" in digest


# ── Full consolidation tests ────────────────────────────────────────

class TestConsolidate:
    def test_produces_records(self, tmp_path):
        journal = tmp_path / "journal.jsonl"
        _write_journal(journal, [
            (1, "mi chiamo Marco", "piacere Marco"),
            (2, "ho mangiato carne", "buon appetito"),
            (3, "mi piace la pizza", "ottima scelta"),
        ])
        records = consolidate(
            journal_path=journal,
            since_turn=0,
            logical_time={"session": 1, "turn": 3},
        )
        assert len(records) > 0
        types = [r.type for r in records]
        assert "NarrativeMemory" in types
        assert "Belief" in types

    def test_filters_by_since_turn(self, tmp_path):
        journal = tmp_path / "journal.jsonl"
        _write_journal(journal, [
            (1, "mi chiamo Marco", "piacere"),
            (2, "ho mangiato carne", "buono"),
            (3, "mi piace leggere", "bello"),
        ])
        # Only turns > 2
        records = consolidate(
            journal_path=journal,
            since_turn=2,
            logical_time={"session": 1, "turn": 3},
        )
        # Should only process turn 3
        for r in records:
            if r.type == "Belief":
                assert r.context.get("source_turn", 0) == 3

    def test_trust_delta_produced(self, tmp_path):
        journal = tmp_path / "journal.jsonl"
        _write_journal(journal, [
            (1, "grazie mille sei bravissimo", "prego"),
            (2, "perfetto esatto", "grazie"),
        ])
        records = consolidate(
            journal_path=journal,
            since_turn=0,
            logical_time={"session": 1, "turn": 2},
        )
        trust_records = [r for r in records if r.type == "TrustUpdate"]
        assert len(trust_records) == 1
        assert trust_records[0].content["delta"] > 0

    def test_negative_trust(self, tmp_path):
        journal = tmp_path / "journal.jsonl"
        _write_journal(journal, [
            (1, "è sbagliato non è vero", "scusa"),
            (2, "errore grave hai fatto male", "mi dispiace"),
        ])
        records = consolidate(
            journal_path=journal,
            since_turn=0,
            logical_time={"session": 1, "turn": 2},
        )
        trust_records = [r for r in records if r.type == "TrustUpdate"]
        assert len(trust_records) == 1
        assert trust_records[0].content["delta"] < 0

    def test_empty_journal(self, tmp_path):
        journal = tmp_path / "journal.jsonl"
        journal.write_text("", encoding="utf-8")
        records = consolidate(
            journal_path=journal,
            since_turn=0,
            logical_time={"session": 1, "turn": 0},
        )
        assert records == []

    def test_no_episodes_since_turn(self, tmp_path):
        journal = tmp_path / "journal.jsonl"
        _write_journal(journal, [
            (1, "ciao", "ciao"),
        ])
        records = consolidate(
            journal_path=journal,
            since_turn=5,
            logical_time={"session": 1, "turn": 5},
        )
        assert records == []

    def test_memory_ids_unique(self, tmp_path):
        journal = tmp_path / "journal.jsonl"
        _write_journal(journal, [
            (1, "mi chiamo Anna", "piacere Anna"),
            (2, "ho mangiato pasta", "buono"),
            (3, "mi piace il cinema", "bello"),
            (4, "grazie mille", "prego"),
        ])
        records = consolidate(
            journal_path=journal,
            since_turn=0,
            logical_time={"session": 1, "turn": 4},
        )
        ids = [r.memory_id for r in records]
        assert len(ids) == len(set(ids)), "Memory IDs must be unique"

    def test_deterministic(self, tmp_path):
        journal = tmp_path / "journal.jsonl"
        _write_journal(journal, [
            (1, "mi chiamo Luca", "ciao Luca"),
            (2, "ho mangiato pesce", "buono"),
        ])
        lt = {"session": 1, "turn": 2}
        r1 = consolidate(journal, since_turn=0, logical_time=lt)
        r2 = consolidate(journal, since_turn=0, logical_time=lt)
        assert len(r1) == len(r2)
        for a, b in zip(r1, r2):
            assert a.memory_id == b.memory_id
            assert a.type == b.type
            assert a.content == b.content
