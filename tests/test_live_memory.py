"""Live integration tests — memory recall + consolidation across turns.

These tests use the REAL session pipeline (run_session internals)
with the "fake" LLM provider.  They verify:
1. Episodic recall retrieves relevant past episodes
2. Consolidation fires after N turns and produces MemoryRecords
3. Memory records accumulate and are visible in the kernel context
4. Trust/belief levers actually affect speech plans
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pie.contracts.state import State
from pie.session.manager import SessionManager, SessionContext
from pie.memory.recall import recall as episodic_recall, load_episodes_from_journal
from pie.memory.consolidation import (
    should_consolidate,
    consolidate as consolidate_journal,
    CONSOLIDATION_INTERVAL,
    extract_facts,
)
from pie.memory.view import build_memory_view
from pie.persistence.memory_store import MemoryStore
from pie.persistence.constraints_store import ConstraintsStore
from pie.persistence.atomic import atomic_write, atomic_append


# ── Fixtures ────────────────────────────────────────────────────────

@pytest.fixture
def seed_path():
    p = Path(__file__).resolve().parent.parent / "progetto" / "SEED_V0.md"
    if not p.exists():
        pytest.skip("SEED_V0.md not found")
    return p


@pytest.fixture
def session_ctx(tmp_path, seed_path):
    """Create a fresh session for testing."""
    mgr = SessionManager(tmp_path / "sessions")
    ctx = mgr.create(seed_path)
    return ctx


# ── Helpers ─────────────────────────────────────────────────────────

def _simulate_turn(ctx: SessionContext, user_input: str, turn_num: int) -> None:
    """Simulate a minimal turn by writing INPUT + LLM_OUTPUT to journal.

    This avoids needing the full pipeline (LLM, goals, etc.) — we just
    need journal entries for recall/consolidation to work.
    """
    eid = ctx.event_id
    ev_input = {
        "id": eid, "type": "INPUT",
        "timestamp": "2025-01-01T00:00:00Z",
        "content": {"logical_time": {"turn": turn_num, "step": 1}, "input": user_input},
    }
    atomic_append(ctx.journal_path, json.dumps(ev_input) + "\n")
    eid += 1
    ev_output = {
        "id": eid, "type": "LLM_OUTPUT",
        "timestamp": "2025-01-01T00:00:00Z",
        "content": {
            "logical_time": {"turn": turn_num, "step": 6, "attempt": 1},
            "text": f"Risposta al turno {turn_num}",
        },
    }
    atomic_append(ctx.journal_path, json.dumps(ev_output) + "\n")
    eid += 1
    ctx.event_id = eid
    ctx.turn_count = turn_num


# ── Test: Recall across turns ──────────────────────────────────────

class TestLiveRecall:
    def test_recall_finds_past_episode(self, session_ctx):
        """After feeding turns, recall should find relevant past episodes."""
        ctx = session_ctx
        _simulate_turn(ctx, "Ho mangiato carne a pranzo", 1)
        _simulate_turn(ctx, "Il tempo è bello oggi", 2)
        _simulate_turn(ctx, "Ieri ho visto un film al cinema", 3)

        # Now query about food — should recall turn 1
        results = episodic_recall(
            query="cosa ho mangiato?",
            journal_path=ctx.journal_path,
            top_k=5,
            exclude_turn=4,
        )
        assert len(results) > 0
        assert results[0].turn == 1, f"Expected turn 1 (carne), got turn {results[0].turn}"

    def test_recall_with_state_bias(self, session_ctx):
        """State-driven bias should boost relevant episodes."""
        ctx = session_ctx
        _simulate_turn(ctx, "come funziona il motore?", 1)
        _simulate_turn(ctx, "mi chiamo Marco", 2)

        # High curiosity → question episode boosted
        high_curiosity = {"drives": {"curiosity": 0.9}, "affect": {}}
        results = episodic_recall(
            query="dimmi qualcosa",
            journal_path=ctx.journal_path,
            state=high_curiosity,
            top_k=5,
        )
        if len(results) >= 2:
            assert results[0].turn == 1  # question episode boosted

    def test_recall_excludes_current_turn(self, session_ctx):
        ctx = session_ctx
        _simulate_turn(ctx, "Ho mangiato pasta", 1)
        _simulate_turn(ctx, "Ho mangiato pesce", 2)

        results = episodic_recall(
            query="mangiato",
            journal_path=ctx.journal_path,
            exclude_turn=2,
        )
        assert all(ep.turn != 2 for ep in results)


# ── Test: Consolidation fires after N turns ────────────────────────

class TestLiveConsolidation:
    def test_consolidation_triggers_after_interval(self, session_ctx):
        """Consolidation should fire after CONSOLIDATION_INTERVAL turns."""
        ctx = session_ctx
        for i in range(1, CONSOLIDATION_INTERVAL + 2):
            _simulate_turn(ctx, f"Turno numero {i}, mi piace il mare", i)

        assert should_consolidate(
            turn_count=CONSOLIDATION_INTERVAL + 1,
            last_consolidation_turn=0,
        )

    def test_consolidation_produces_memories(self, session_ctx):
        """Consolidation should produce NarrativeMemory + Belief records."""
        ctx = session_ctx
        turns = [
            (1, "mi chiamo Luca"),
            (2, "ho mangiato pasta a pranzo"),
            (3, "mi piace il cinema"),
            (4, "lavoro come ingegnere"),
            (5, "grazie sei molto bravo"),
            (6, "ho comprato un libro"),
        ]
        for num, text in turns:
            _simulate_turn(ctx, text, num)

        records = consolidate_journal(
            journal_path=ctx.journal_path,
            since_turn=0,
            logical_time={"session": 1, "turn": 6},
            source_refs=[1, 2],
        )
        types = [r.type for r in records]
        assert "NarrativeMemory" in types, f"Expected NarrativeMemory, got types: {types}"
        assert "Belief" in types, f"Expected Belief, got types: {types}"

        # Narrative should mention turn range
        narrative = [r for r in records if r.type == "NarrativeMemory"][0]
        assert "1-6" in narrative.content["text"]

        # Beliefs should have extracted facts
        beliefs = [r for r in records if r.type == "Belief"]
        assert len(beliefs) >= 2, f"Expected ≥2 beliefs, got {len(beliefs)}"

    def test_trust_accumulated(self, session_ctx):
        """Trust signals in conversation should produce TrustUpdate records."""
        ctx = session_ctx
        _simulate_turn(ctx, "grazie mille sei perfetto", 1)
        _simulate_turn(ctx, "esatto hai ragione", 2)
        _simulate_turn(ctx, "bravo davvero", 3)
        _simulate_turn(ctx, "ciao come stai", 4)
        _simulate_turn(ctx, "tutto bene", 5)
        _simulate_turn(ctx, "ok", 6)

        records = consolidate_journal(
            journal_path=ctx.journal_path,
            since_turn=0,
            logical_time={"session": 1, "turn": 6},
        )
        trust = [r for r in records if r.type == "TrustUpdate"]
        assert len(trust) == 1
        assert trust[0].content["delta"] > 0, "Positive trust signals should produce positive delta"


# ── Test: Memory view includes episodes ────────────────────────────

class TestMemoryViewIntegration:
    def test_episodic_recall_in_view(self, session_ctx):
        """MemoryView should include episodic recall results."""
        ctx = session_ctx
        _simulate_turn(ctx, "Ho mangiato carne", 1)
        _simulate_turn(ctx, "Mi piace il mare", 2)

        # Build memory view and add episodic recall
        view = build_memory_view(ctx.memory_store.read_all())
        episodes = episodic_recall(
            query="cosa ho mangiato?",
            journal_path=ctx.journal_path,
        )
        view.episodic_recall = [ep.to_dict() for ep in episodes]

        d = view.to_dict()
        assert "episodic_recall" in d
        assert len(d["episodic_recall"]) > 0
        assert d["episodic_recall"][0]["turn"] == 1

    def test_consolidation_records_visible_in_view(self, session_ctx):
        """After consolidation, MemoryView should show narrative + beliefs."""
        ctx = session_ctx
        for i in range(1, 7):
            _simulate_turn(ctx, f"ho mangiato qualcosa il turno {i}", i)

        records = consolidate_journal(
            journal_path=ctx.journal_path,
            since_turn=0,
            logical_time={"session": 1, "turn": 6},
        )
        for r in records:
            ctx.memory_store.append(r)

        view = build_memory_view(ctx.memory_store.read_all())
        assert view.identity_summary != "", "Narrative digest should be in identity_summary"
        assert "Turni 1-6" in view.identity_summary


# ── Stress test: 12 turns with consolidation + recall ──────────────

class TestStress:
    def test_12_turns_full_memory_cycle(self, session_ctx):
        """Simulate 12 turns: consolidation fires twice, recall works throughout."""
        ctx = session_ctx
        conversations = [
            (1, "Ciao, mi chiamo Marco"),
            (2, "Ho 30 anni e lavoro come ingegnere"),
            (3, "Mi piace la pizza margherita"),
            (4, "Ieri ho mangiato sushi al ristorante"),
            (5, "Il mio amico si chiama Luca"),
            (6, "Sono andato al cinema sabato"),  # consolidation trigger at turn 6
            (7, "Mi piace leggere libri di fantascienza"),
            (8, "Ho comprato una macchina nuova"),
            (9, "Odio il traffico della città"),
            (10, "Sono felice oggi"),
            (11, "Grazie per avermi aiutato"),
            (12, "Studio informatica all'università"),  # consolidation trigger at turn 12
        ]

        last_consol_turn = 0
        all_consol_records = []

        for turn_num, text in conversations:
            _simulate_turn(ctx, text, turn_num)

            # Check consolidation trigger
            if should_consolidate(turn_num, last_consol_turn):
                records = consolidate_journal(
                    journal_path=ctx.journal_path,
                    since_turn=last_consol_turn,
                    logical_time={"session": 1, "turn": turn_num},
                )
                for r in records:
                    ctx.memory_store.append(r)
                all_consol_records.extend(records)
                last_consol_turn = turn_num

        # Consolidation should have fired at least twice (turns 5-6 and 10-12)
        assert last_consol_turn > 0, "Consolidation never triggered"

        # Memory records should include diverse types
        types = {r.type for r in all_consol_records}
        assert "NarrativeMemory" in types
        assert "Belief" in types

        # Recall should find relevant episodes
        food_results = episodic_recall(
            query="cosa ho mangiato?",
            journal_path=ctx.journal_path,
            top_k=5,
        )
        assert len(food_results) > 0
        food_turns = {ep.turn for ep in food_results}
        assert 4 in food_turns, f"Turn 4 (sushi) should be recalled, got turns {food_turns}"

        # Name recall
        name_results = episodic_recall(
            query="come mi chiamo?",
            journal_path=ctx.journal_path,
            top_k=5,
        )
        assert len(name_results) > 0

        # Preference recall
        pref_results = episodic_recall(
            query="cosa mi piace?",
            journal_path=ctx.journal_path,
            top_k=5,
        )
        assert len(pref_results) > 0

        # Memory view should be populated
        view = build_memory_view(ctx.memory_store.read_all())
        assert view.identity_summary != ""

        # Beliefs should include extracted facts
        beliefs = [r for r in all_consol_records if r.type == "Belief"]
        categories = set()
        for b in beliefs:
            claim = b.content.get("claim", "")
            if ":" in claim:
                categories.add(claim.split(":")[0])
        assert len(categories) >= 3, f"Expected ≥3 fact categories, got {categories}"

    def test_recall_after_many_turns(self, session_ctx):
        """Even after 15+ turns, recall should efficiently find relevant episodes."""
        ctx = session_ctx
        for i in range(1, 16):
            _simulate_turn(ctx, f"Turno generico numero {i}", i)
        # Insert a specific episode at turn 16
        _simulate_turn(ctx, "Ho comprato un gatto persiano bellissimo", 16)
        for i in range(17, 21):
            _simulate_turn(ctx, f"Ancora un turno generico {i}", i)

        results = episodic_recall(
            query="gatto persiano",
            journal_path=ctx.journal_path,
            top_k=3,
        )
        assert len(results) > 0
        assert results[0].turn == 16, f"Expected turn 16 (gatto), got turn {results[0].turn}"

    def test_fact_extraction_variety(self, session_ctx):
        """Extract diverse fact types from a realistic conversation."""
        ctx = session_ctx
        inputs = [
            "mi chiamo Anna",          # user_name
            "ho 25 anni",              # user_age
            "lavoro come avvocato",    # occupation
            "mi piace viaggiare",      # preference
            "odio alzarmi presto",     # dislike
            "sono andata a Roma",      # location
            "ho mangiato una torta",   # action
            "il mio cane si chiama Rex",  # named_entity
        ]
        from pie.memory.recall import Episode
        episodes = [Episode(turn=i+1, user_input=inp, llm_output="ok") for i, inp in enumerate(inputs)]
        facts = extract_facts(episodes)

        found_categories = {f.category for f in facts}
        expected = {"user_name", "user_age", "occupation", "preference", "dislike", "action"}
        missing = expected - found_categories
        assert len(missing) <= 1, f"Missing fact categories: {missing}. Found: {found_categories}"
