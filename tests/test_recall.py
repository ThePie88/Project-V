"""Tests for episodic recall — BM25 retrieval from journal."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pie.memory.recall import (
    _tokenize,
    _compute_idf,
    _bm25_score,
    _state_bias,
    Episode,
    load_episodes_from_journal,
    recall,
)


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


# ── Unit tests ──────────────────────────────────────────────────────

class TestTokenize:
    def test_basic(self):
        tokens = _tokenize("Ciao, come stai? Ho mangiato carne.")
        assert "ciao" in tokens
        assert "come" in tokens
        assert "mangiato" in tokens
        assert "ho" not in tokens  # < 3 chars

    def test_empty(self):
        assert _tokenize("") == []

    def test_unicode_italian(self):
        tokens = _tokenize("perché è così")
        assert "perché" in tokens
        assert "così" in tokens


class TestBM25:
    def test_exact_match_scores_higher(self):
        episodes = [
            Episode(turn=1, user_input="ho mangiato carne", llm_output="ok"),
            Episode(turn=2, user_input="buongiorno come stai", llm_output="bene"),
        ]
        idf = _compute_idf(episodes)
        q = _tokenize("cosa hai mangiato?")
        s1 = _bm25_score(q, _tokenize("ho mangiato carne ok"), idf, 4.0)
        s2 = _bm25_score(q, _tokenize("buongiorno come stai bene"), idf, 4.0)
        assert s1 > s2

    def test_empty_query(self):
        assert _bm25_score([], ["ciao"], {}, 1.0) == 0.0

    def test_empty_doc(self):
        assert _bm25_score(["ciao"], [], {"ciao": 1.0}, 1.0) == 0.0


class TestStateBias:
    def test_curiosity_boosts_questions(self):
        ep = Episode(turn=1, user_input="come funziona?", llm_output="spiego")
        high_curiosity = {"drives": {"curiosity": 0.9}, "affect": {}}
        low_curiosity = {"drives": {"curiosity": 0.3}, "affect": {}}
        assert _state_bias(ep, high_curiosity) > _state_bias(ep, low_curiosity)

    def test_tension_boosts_conflict(self):
        ep = Episode(turn=1, user_input="c'è un danno grave", llm_output="attenzione")
        high_tension = {"drives": {}, "affect": {"tension": 0.5}}
        low_tension = {"drives": {}, "affect": {"tension": 0.1}}
        assert _state_bias(ep, high_tension) > _state_bias(ep, low_tension)

    def test_neutral_state_no_bias(self):
        ep = Episode(turn=1, user_input="ciao", llm_output="ciao")
        neutral = {"drives": {"curiosity": 0.5, "sociality": 0.5}, "affect": {"tension": 0.1}}
        assert _state_bias(ep, neutral) == 1.0


# ── Integration tests ───────────────────────────────────────────────

class TestLoadEpisodes:
    def test_loads_from_journal(self, tmp_path):
        journal = tmp_path / "journal.jsonl"
        _write_journal(journal, [
            (1, "Ciao Ivy", "Ciao!"),
            (2, "Ho mangiato carne oggi", "Interessante!"),
            (3, "Come stai?", "Bene grazie"),
        ])
        episodes = load_episodes_from_journal(journal)
        assert len(episodes) == 3
        assert episodes[0].turn == 1
        assert episodes[1].user_input == "Ho mangiato carne oggi"

    def test_missing_journal(self, tmp_path):
        assert load_episodes_from_journal(tmp_path / "nope.jsonl") == []


class TestRecall:
    def test_relevant_episode_recalled(self, tmp_path):
        journal = tmp_path / "journal.jsonl"
        _write_journal(journal, [
            (1, "Ho mangiato carne a pranzo", "Buon appetito!"),
            (2, "Il tempo è bello oggi", "Sì, splendido"),
            (3, "Ieri ho visto un film al cinema", "Quale film?"),
        ])
        results = recall("cosa ho mangiato?", journal)
        assert len(results) > 0
        assert results[0].turn == 1  # carne episode should be top

    def test_no_results_for_unrelated_query(self, tmp_path):
        journal = tmp_path / "journal.jsonl"
        _write_journal(journal, [
            (1, "ciao", "ciao"),
        ])
        results = recall("quantum physics entanglement", journal)
        assert len(results) == 0

    def test_exclude_current_turn(self, tmp_path):
        journal = tmp_path / "journal.jsonl"
        _write_journal(journal, [
            (1, "Ho mangiato carne", "ok"),
            (2, "Ho mangiato pesce", "ok"),
        ])
        results = recall("mangiato", journal, exclude_turn=2)
        assert all(ep.turn != 2 for ep in results)

    def test_deterministic(self, tmp_path):
        journal = tmp_path / "journal.jsonl"
        _write_journal(journal, [
            (1, "Ho mangiato carne", "Buon appetito!"),
            (2, "Il tempo è bello", "Sì!"),
            (3, "Ho comprato un libro", "Bello!"),
        ])
        r1 = recall("cosa ho mangiato?", journal)
        r2 = recall("cosa ho mangiato?", journal)
        assert [e.to_dict() for e in r1] == [e.to_dict() for e in r2]

    def test_state_bias_affects_ranking(self, tmp_path):
        journal = tmp_path / "journal.jsonl"
        _write_journal(journal, [
            (1, "come funziona il motore?", "spiego"),
            (2, "mi chiamo Marco", "piacere Marco"),
        ])
        # High curiosity → questions ranked higher
        high_curiosity = {"drives": {"curiosity": 0.9}, "affect": {}}
        r = recall("dimmi qualcosa", journal, state=high_curiosity)
        if len(r) >= 2:
            assert r[0].turn == 1  # question episode boosted

    def test_empty_journal(self, tmp_path):
        journal = tmp_path / "journal.jsonl"
        journal.write_text("", encoding="utf-8")
        assert recall("ciao", journal) == []
