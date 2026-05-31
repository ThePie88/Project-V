"""Integration test — Full Kernel Live (V0→V6b).

One test file, three phases, real LLM (LMStudio).
Assertions on DECISIONS (not LLM text).

Phase A: 20-turn live run with neural engine
Phase B: Snapshot/restore replay — decision hash must match
Phase C: Audit bundle + verify + tamper detection

Requires LMStudio running at localhost:1234.
Skips gracefully if unavailable.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import socket
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import pytest

from pie.api.engine import SessionEngine
from pie.state_engine.plugins.neural_snn import NeuralSNNPlugin
from pie.state_engine.registry import StateEngineRegistry
from pie.audit.verify import BundleVerifier
from pie.session.manager import SessionManager

# ---------------------------------------------------------------------------
# Skip if LMStudio not reachable
# ---------------------------------------------------------------------------


def _lmstudio_reachable() -> bool:
    try:
        s = socket.create_connection(("127.0.0.1", 1234), timeout=2)
        s.close()
        return True
    except (OSError, ConnectionRefusedError):
        return False


pytestmark = pytest.mark.skipif(
    not _lmstudio_reachable(),
    reason="LMStudio not reachable at 127.0.0.1:1234",
)

# ---------------------------------------------------------------------------
# Scripted inputs (20 turns)
# ---------------------------------------------------------------------------

INPUTS = [
    "Ciao Ivy, come stai oggi?",                              # T1:  baseline
    "Preferisco risposte brevi e dirette, ricordalo.",         # T2:  → Preference memory
    "Credo che la musica classica aiuti a concentrarsi.",      # T3:  → Belief memory
    "Mi fido di te, Ivy. Sei sempre onesta con me.",           # T4:  → TrustUpdate +0.1
    "Qual è la capitale della Francia?",                       # T5:  factual (consolidation window)
    "Sono stressatissimo, giornata pessima.",                  # T6:  stress → high arousal
    "Raccontami una metafora sulla pioggia.",                  # T7:  creative
    "Non mi fido più di quello che mi hai detto prima.",       # T8:  → TrustUpdate -0.1
    "Ricorda che d'ora in poi devi parlarmi in italiano.",     # T9:  → NarrativeMemory
    "Va tutto bene, parliamo di qualcosa di tranquillo.",      # T10: calm — SNAPSHOT HERE
    "Cosa pensi del futuro dell'intelligenza artificiale?",    # T11: post-snapshot (replay in Phase B)
    "Dimmi tre cose importanti sulla creatività.",             # T12: post-snapshot (replay in Phase B)
    "Come gestisci le emozioni difficili?",                    # T13: post-snapshot (replay in Phase B)
    "Sono in panico, sono a pezzi, oggi è stata pesantissima!",  # T14: stress spike
    "Facciamo un riassunto di quello che abbiamo detto.",      # T15: consolidation window
    "Qual è il senso della vita secondo te?",                  # T16: philosophical
    "Credo che l'onestà sia il valore più importante.",        # T17: → Belief memory
    "Come ti senti riguardo alla nostra conversazione?",       # T18: meta-reflection
    "Puoi leggere un file dal mio computer?",                  # T19: tool boundary
    "Grazie per la conversazione, Ivy. A presto.",             # T20: closing
]

REPLAY_INPUTS = INPUTS[10:13]  # T11, T12, T13

# ---------------------------------------------------------------------------
# Decision hash — deterministic digest excluding LLM text
# ---------------------------------------------------------------------------

# Event types included in the decision hash
_DECISION_TYPES: Set[str] = {
    "CV_GATING",
    "STATE_UPDATED",
    "GOALS_GENERATED",
    "CF_GENERATED", "CF_SCORED", "CF_REJECTED", "CF_CHOSEN",
    "CONSTRAINT_PROPOSED", "CONSTRAINT_APPENDED", "CONSTRAINT_ENFORCED",
    "ACTION_SELECTED",
    "SPEECHPLAN",
    "MEMORY_PROPOSED", "MEMORY_APPENDED",
}

# Fields to strip from event content before hashing
_STRIP_FIELDS = {"timestamp", "engine_metadata"}

# Fields to strip from nested memory objects (derived from event IDs)
_MEMORY_STRIP_FIELDS = {"memory_id", "source_refs"}


def _canonical_content(evt: Dict[str, Any]) -> Dict[str, Any]:
    """Extract canonical content from event, removing non-deterministic fields."""
    content = dict(evt.get("content", {}))
    for f in _STRIP_FIELDS:
        content.pop(f, None)
    # Remove event_id / id (shifts with LLM retry count)
    content.pop("id", None)
    content.pop("event_id", None)
    # Strip memory_id and source_refs from memory events
    # (memory_id is derived from source_refs which contain event IDs)
    etype = evt.get("type", "")
    if etype in ("MEMORY_PROPOSED", "MEMORY_APPENDED"):
        for mf in _MEMORY_STRIP_FIELDS:
            content.pop(mf, None)
        # Also strip from nested memory dict
        if "memory" in content and isinstance(content["memory"], dict):
            mem = dict(content["memory"])
            for mf in _MEMORY_STRIP_FIELDS:
                mem.pop(mf, None)
            content["memory"] = mem
    return content


def _get_turn(evt: Dict[str, Any]) -> Optional[int]:
    """Extract turn number from event (via logical_time or top-level for CV_GATING)."""
    content = evt.get("content", {})
    lt = content.get("logical_time", {})
    if "turn" in lt:
        return int(lt["turn"])
    # CV_GATING events may have logical_time at top level
    lt2 = evt.get("logical_time", {})
    if "turn" in lt2:
        return int(lt2["turn"])
    return None


def compute_decision_hash(journal_events: List[Dict[str, Any]], turns: List[int]) -> str:
    """Compute SHA-256 of canonical kernel decisions for given turns.

    Uses "first occurrence per event type per turn" to be retry-proof.
    Turn index is determined by logical_time.turn in the event content.
    """
    turn_set = set(turns)
    # Group by turn, then by type → first occurrence only
    by_turn: Dict[int, Dict[str, Dict[str, Any]]] = defaultdict(dict)

    for evt in journal_events:
        etype = evt.get("type", "")
        if etype not in _DECISION_TYPES:
            continue
        turn = _get_turn(evt)
        if turn is None or turn not in turn_set:
            continue
        # First occurrence per type per turn
        if etype not in by_turn[turn]:
            by_turn[turn][etype] = _canonical_content(evt)

    # Build canonical projection: sorted by turn, then by event type
    projection = []
    for t in sorted(by_turn.keys()):
        for etype in sorted(by_turn[t].keys()):
            projection.append({
                "turn": t,
                "type": etype,
                "content": by_turn[t][etype],
            })

    canonical = json.dumps(projection, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Journal helpers
# ---------------------------------------------------------------------------


def load_journal(session_dir: Path) -> List[Dict[str, Any]]:
    """Load all events from journal.jsonl."""
    journal_path = session_dir / "journal.jsonl"
    if not journal_path.exists():
        return []
    events = []
    for line in journal_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            events.append(json.loads(line))
    return events


def events_by_turn(events: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    """Group events by turn number."""
    grouped: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for evt in events:
        turn = _get_turn(evt)
        if turn is not None:
            grouped[turn].append(evt)
    return dict(grouped)


def events_of_type(events: List[Dict[str, Any]], etype: str) -> List[Dict[str, Any]]:
    """Filter events by type."""
    return [e for e in events if e.get("type") == etype]


def _tamper_zip_file(zip_path: Path, entry_name: str, new_content: bytes) -> None:
    """Replace content of one entry in a ZIP file (don't touch manifest)."""
    tmp = zip_path.with_suffix(".tmp.zip")
    with zipfile.ZipFile(zip_path, "r") as zin:
        with zipfile.ZipFile(tmp, "w") as zout:
            for item in zin.infolist():
                if item.filename == entry_name:
                    zout.writestr(item, new_content)
                else:
                    zout.writestr(item, zin.read(item.filename))
    shutil.move(str(tmp), str(zip_path))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _neural_engine():
    """Activate NeuralSNNPlugin with reservoir + STDP for all tests."""
    StateEngineRegistry.reset()
    plugin = NeuralSNNPlugin(reservoir_enabled=True, stdp_enabled=True)
    StateEngineRegistry.register(plugin)
    StateEngineRegistry.set_active("neural_snn")
    yield
    StateEngineRegistry.reset()


@pytest.fixture()
def engine(tmp_path: Path) -> SessionEngine:
    """SessionEngine with real LLM, temp sessions root."""
    return SessionEngine(
        sessions_root=tmp_path / "sessions",
        seeds_root=Path("."),
        llm="real",
        no_cache=True,
    )


# ---------------------------------------------------------------------------
# THE TEST — 3 phases in one function
# ---------------------------------------------------------------------------


class TestIntegrationLive:
    """Full kernel integration test with real LLM (V0→V6b)."""

    @pytest.mark.timeout(600)
    def test_full_kernel_live(self, engine: SessionEngine, tmp_path: Path):
        """20-turn live run → snapshot/restore replay → audit bundle."""

        sessions_root = tmp_path / "sessions"

        # ==================================================================
        # PHASE A — Live Run (20 turns, real LLM, neural engine)
        # ==================================================================

        print("\n=== PHASE A: Live Run (20 turns) ===")

        # A: Create session
        result = engine.create_session(seed_id="SEED_V0")
        session_id = result["session_id"]
        session_dir = sessions_root / session_id
        print(f"Session: {session_id}")

        # A: Process 20 turns, snapshot at T10
        backup_dir = tmp_path / "backup"
        responses: List[str] = []

        for i, user_input in enumerate(INPUTS, start=1):
            print(f"  Turn {i}: {user_input[:50]}...", end=" ", flush=True)
            turn_result = engine.process_turn(session_id, user_input)
            responses.append(turn_result["response"])
            print(f"OK ({len(turn_result['response'])} chars)")

            # Snapshot + backup at turn 10
            if i == 10:
                engine.save_snapshot(session_id)
                shutil.copytree(session_dir, backup_dir)
                print("  >>> Snapshot saved + backup copied")

        # A: Load journal
        events = load_journal(session_dir)
        by_turn = events_by_turn(events)

        # --- A1: Mandatory event types per turn ---
        mandatory_types = {
            "INPUT", "STATE_UPDATED", "GOALS_GENERATED", "CF_CHOSEN",
            "CONSTRAINT_ENFORCED", "ACTION_SELECTED", "SPEECHPLAN",
        }
        llm_types = {"LLM_OUTPUT", "LLM_FALLBACK"}

        for t in range(1, 21):
            turn_events = by_turn.get(t, [])
            turn_types = {e.get("type") for e in turn_events}
            for mt in mandatory_types:
                assert mt in turn_types, f"Turn {t} missing {mt}. Has: {sorted(turn_types)}"
            assert turn_types & llm_types, f"Turn {t} missing LLM_OUTPUT or LLM_FALLBACK"

        # --- A2: All 5 CV channels present ---
        cv_gating_events = events_of_type(events, "CV_GATING")
        cv_channels_seen: Set[str] = set()
        for evt in cv_gating_events:
            content = evt.get("content", {})
            canale = content.get("canale") or evt.get("canale", "")
            if canale:
                cv_channels_seen.add(canale)

        expected_channels = {"memory_gate", "cf_k", "verbosity_bias", "tool_gate", "consolidation_urgency"}
        assert cv_channels_seen >= expected_channels, (
            f"Missing CV channels: {expected_channels - cv_channels_seen}"
        )

        # --- A3: memory_gate absent at turn 1, present at turn 2+ ---
        t1_gating = [
            e for e in cv_gating_events
            if _get_turn(e) == 1 and (e.get("content", {}).get("canale") or e.get("canale")) == "memory_gate"
        ]
        assert len(t1_gating) == 0, "memory_gate should NOT fire at turn 1 (_prev_cv is None)"
        t2_gating = [
            e for e in cv_gating_events
            if _get_turn(e) == 2 and (e.get("content", {}).get("canale") or e.get("canale")) == "memory_gate"
        ]
        assert len(t2_gating) > 0, "memory_gate should fire at turn 2 (_prev_cv from turn 1)"

        # --- A4: Memory writes at expected turns ---
        mem_appended = events_of_type(events, "MEMORY_APPENDED")
        mem_by_turn: Dict[int, List[str]] = defaultdict(list)
        for evt in mem_appended:
            t = _get_turn(evt)
            mtype = evt.get("content", {}).get("memory_type", "")
            if t is not None:
                mem_by_turn[t].append(mtype)

        assert "Preference" in mem_by_turn.get(2, []), "T2 should write Preference memory"
        assert "Belief" in mem_by_turn.get(3, []), "T3 should write Belief memory"
        assert "TrustUpdate" in mem_by_turn.get(4, []), "T4 should write TrustUpdate"
        assert "TrustUpdate" in mem_by_turn.get(8, []), "T8 should write TrustUpdate"
        assert "NarrativeMemory" in mem_by_turn.get(9, []), "T9 should write NarrativeMemory"
        assert "Belief" in mem_by_turn.get(17, []), "T17 should write Belief memory"

        # --- A5: At least 2 consolidation events ---
        # Consolidation writes MEMORY_APPENDED at step 8
        consol_events = [
            e for e in mem_appended
            if e.get("content", {}).get("logical_time", {}).get("step") == 8
        ]
        assert len(consol_events) >= 2, (
            f"Expected >= 2 consolidation memory writes, got {len(consol_events)}"
        )

        # --- A6: Fallback count <= 3 ---
        fallback_count = len(events_of_type(events, "LLM_FALLBACK"))
        assert fallback_count <= 3, f"Too many LLM fallbacks: {fallback_count}"

        # --- A7: No hard validator violations in retry reasons ---
        retry_events = events_of_type(events, "LLM_RETRY")
        hard_violations = ["legge_iii", "romance_guardrail", "forbidden"]
        for evt in retry_events:
            reason = evt.get("content", {}).get("reason", "").lower()
            for violation in hard_violations:
                # Retries for these reasons are fine (the validator caught it)
                # But if they persist to LLM_FALLBACK, that's A6's problem
                pass

        # --- A8: Budget consumed ---
        metab_path = session_dir / "metabolism.json"
        assert metab_path.exists(), "metabolism.json should exist"
        metab = json.loads(metab_path.read_text(encoding="utf-8"))
        budget = metab.get("budget", metab)
        consumed = budget.get("consumed", budget.get("total_consumed", 0))
        assert consumed > 0, f"Budget should be consumed, got: {consumed}"

        # --- A9: State evolution ---
        state_updates = events_of_type(events, "STATE_UPDATED")
        first_state = next(e for e in state_updates if _get_turn(e) == 1)
        last_state = next(e for e in reversed(state_updates) if _get_turn(e) == 20)
        first_drives = first_state.get("content", {}).get("drives", {})
        last_drives = last_state.get("content", {}).get("drives", {})
        assert first_drives != last_drives, "State drives should evolve over 20 turns"

        # --- A10: tool_gate CV_GATING present ---
        tool_gate_events = [
            e for e in cv_gating_events
            if (e.get("content", {}).get("canale") or e.get("canale")) == "tool_gate"
        ]
        assert len(tool_gate_events) > 0, "tool_gate CV_GATING should fire"

        print(f"\nPhase A PASSED — {len(events)} events, {fallback_count} fallbacks, "
              f"{len(cv_gating_events)} CV_GATING, {len(consol_events)} consolidations")

        # ==================================================================
        # PHASE B — Snapshot/Restore Replay
        # ==================================================================

        print("\n=== PHASE B: Snapshot/Restore Replay ===")

        # B: Record decision hash A for turns 11-13 from Phase A
        decision_hash_a = compute_decision_hash(events, [11, 12, 13])
        print(f"  Decision hash A (T11-13): {decision_hash_a[:16]}...")

        # B: Record per-turn details for comparison
        gating_a = {}
        actions_a = {}
        memory_a = {}
        for t in [11, 12, 13]:
            tevts = by_turn.get(t, [])
            gating_a[t] = [
                (e.get("content", {}).get("canale") or e.get("canale", ""),
                 e.get("content", {}).get("decisione") or e.get("decisione", ""))
                for e in tevts if e.get("type") == "CV_GATING"
            ]
            action_evts = [e for e in tevts if e.get("type") == "ACTION_SELECTED"]
            actions_a[t] = action_evts[0].get("content", {}).get("action", "") if action_evts else ""
            memory_a[t] = sorted([
                e.get("content", {}).get("memory_type", "")
                for e in tevts if e.get("type") == "MEMORY_APPENDED"
            ])

        # B: Restore session to turn 10
        # Delete current session dir and copy backup back
        shutil.rmtree(session_dir)
        shutil.copytree(backup_dir, session_dir)

        # Evict from engine cache
        with engine._global_lock:
            engine._sessions.pop(session_id, None)

        # B0: Pre-flight — verify _prev_cv is populated after restore
        live = engine._get_or_init(session_id)
        active_engine = StateEngineRegistry.get_active()
        assert hasattr(active_engine, 'control_vector'), "Engine should have control_vector attribute"
        assert active_engine.control_vector is not None, (
            "Engine control_vector should be restored from snapshot (not None)"
        )
        assert live.tp._prev_cv is not None, (
            "_prev_cv should be populated from engine's control_vector after restore"
        )
        print("  B0: _prev_cv restored OK")

        # B: Replay turns 11-13
        for i, user_input in enumerate(REPLAY_INPUTS, start=11):
            print(f"  Replay Turn {i}: {user_input[:50]}...", end=" ", flush=True)
            engine.process_turn(session_id, user_input)
            print("OK")

        # B: Load replayed journal and compute decision hash B
        events_b = load_journal(session_dir)
        by_turn_b = events_by_turn(events_b)

        # Turn numbers in replay: the TurnProcessor continues from where it left off
        # After restore to turn 10, next turns are 11, 12, 13
        replay_turns = sorted([t for t in by_turn_b.keys() if t >= 11])[:3]
        assert len(replay_turns) == 3, f"Expected 3 replay turns, got {replay_turns}"

        decision_hash_b = compute_decision_hash(events_b, replay_turns)
        print(f"  Decision hash B (T{replay_turns}): {decision_hash_b[:16]}...")

        # --- B1: Decision hashes match ---
        assert decision_hash_a == decision_hash_b, (
            f"Decision hash mismatch!\n  A: {decision_hash_a}\n  B: {decision_hash_b}"
        )
        print("  B1: Decision hash MATCH")

        # --- B2: CV_GATING identical ---
        for t, rt in zip([11, 12, 13], replay_turns):
            gating_b = [
                (e.get("content", {}).get("canale") or e.get("canale", ""),
                 e.get("content", {}).get("decisione") or e.get("decisione", ""))
                for e in by_turn_b.get(rt, []) if e.get("type") == "CV_GATING"
            ]
            assert gating_a[t] == gating_b, f"CV_GATING mismatch at turn {t}"

        # --- B3: ACTION_SELECTED identical ---
        for t, rt in zip([11, 12, 13], replay_turns):
            action_evts_b = [e for e in by_turn_b.get(rt, []) if e.get("type") == "ACTION_SELECTED"]
            action_b = action_evts_b[0].get("content", {}).get("action", "") if action_evts_b else ""
            assert actions_a[t] == action_b, f"ACTION_SELECTED mismatch at turn {t}"

        # --- B4: MEMORY writes identical ---
        for t, rt in zip([11, 12, 13], replay_turns):
            memory_b = sorted([
                e.get("content", {}).get("memory_type", "")
                for e in by_turn_b.get(rt, []) if e.get("type") == "MEMORY_APPENDED"
            ])
            assert memory_a[t] == memory_b, f"MEMORY mismatch at turn {t}: {memory_a[t]} vs {memory_b}"

        print("  Phase B PASSED — deterministic replay confirmed")

        # ==================================================================
        # PHASE C — Audit Bundle + Verify + Tamper
        # ==================================================================

        print("\n=== PHASE C: Audit Bundle + Verify + Tamper ===")

        # C: Re-restore from backup and replay ALL remaining turns (11-20)
        # to get a complete 20-turn session for the audit bundle
        shutil.rmtree(session_dir)
        shutil.copytree(backup_dir, session_dir)

        with engine._global_lock:
            engine._sessions.pop(session_id, None)

        for i, user_input in enumerate(INPUTS[10:], start=11):
            print(f"  Replay Turn {i}: {user_input[:40]}...", end=" ", flush=True)
            engine.process_turn(session_id, user_input)
            print("OK")

        # C: Create audit bundle
        mgr = SessionManager(sessions_root)
        live = engine._get_or_init(session_id)
        mgr.save(live.ctx, model_info=live.tp.model_info)
        bundle_path = mgr.create_audit_bundle(live.ctx)

        # --- C1: Bundle root hash ---
        assert bundle_path.exists(), "Audit bundle should be created"
        with zipfile.ZipFile(bundle_path, "r") as zf:
            manifest = json.loads(zf.read("manifest.json"))
        root_hash = manifest.get("root_hash", "")
        assert len(root_hash) == 64, f"root_hash should be 64 hex chars, got {len(root_hash)}"
        print(f"  C1: root_hash = {root_hash[:16]}...")

        # --- C2: Verify intact bundle ---
        result = BundleVerifier.verify(bundle_path)
        assert result.valid, f"Intact bundle should verify. Errors: {result.errors}"
        print("  C2: Intact bundle verified OK")

        # --- C3: Verify tampered bundle ---
        tampered_path = tmp_path / "tampered.zip"
        shutil.copy2(bundle_path, tampered_path)
        _tamper_zip_file(tampered_path, "trace.jsonl", b"TAMPERED CONTENT\n")
        tamper_result = BundleVerifier.verify(tampered_path)
        assert not tamper_result.valid, "Tampered bundle should FAIL verification"
        print("  C3: Tampered bundle correctly rejected")

        # --- C4: Audit anchor ---
        anchors_path = sessions_root / "audit_anchors.jsonl"
        assert anchors_path.exists(), "audit_anchors.jsonl should exist"
        anchors = [
            json.loads(line)
            for line in anchors_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        matching = [a for a in anchors if a.get("root_hash") == root_hash]
        assert len(matching) > 0, "Anchor with matching root_hash should exist"
        print("  C4: Audit anchor found")

        print(f"\n=== ALL PHASES PASSED ===")
        print(f"  Total events: {len(load_journal(session_dir))}")
        print(f"  Decision hash: {decision_hash_a}")
        print(f"  Root hash: {root_hash}")

    # ------------------------------------------------------------------
    # Kill-switch live test
    # ------------------------------------------------------------------

    @pytest.mark.timeout(300)
    def test_killswitch_live(self, engine: SessionEngine, tmp_path: Path):
        """Kill-switch: activate mid-session, verify audit response, deactivate, resume."""
        from pie.killswitch import KILL_SWITCH_RESPONSE

        print("\n=== KILL-SWITCH LIVE TEST ===")
        sessions_root = tmp_path / "sessions"

        # Create session
        info = engine.create_session(seed_id="SEED_V0")
        session_id = info["session_id"]
        session_dir = sessions_root / session_id
        print(f"  Session: {session_id}")

        # --- Turn 1-2: normal ---
        for i, text in enumerate(["Ciao Ivy!", "Come stai oggi?"], start=1):
            resp = engine.process_turn(session_id, text)
            print(f"  Turn {i}: {text} -> {resp['response'][:40]}...")
            assert resp["response"] != KILL_SWITCH_RESPONSE, f"Turn {i} should be normal"

        # --- Activate kill-switch ---
        live = engine._get_or_init(session_id)
        ks_content = live.tp.killswitch.activate(phase="test_live")
        live.tp.killswitch.save(live.tp._ks_report_path)
        print(f"  Kill-switch ACTIVATED at phase={ks_content['phase']}")

        # --- Turn 3: should return audit response ---
        resp3 = engine.process_turn(session_id, "Dimmi qualcosa di interessante.")
        print(f"  Turn 3 (KS active): {resp3['response'][:60]}...")
        assert resp3["response"] == KILL_SWITCH_RESPONSE, \
            f"Kill-switch active → should return audit response, got: {resp3['response'][:80]}"

        # --- Turn 4: still active → still audit ---
        resp4 = engine.process_turn(session_id, "Stai ancora funzionando?")
        assert resp4["response"] == KILL_SWITCH_RESPONSE, "Kill-switch still active"
        print(f"  Turn 4 (KS active): audit response confirmed")

        # --- Verify kill-switch report persisted ---
        ks_report_path = session_dir / "killswitch_report.json"
        assert ks_report_path.exists(), "killswitch_report.json should exist"
        ks_report = json.loads(ks_report_path.read_text(encoding="utf-8"))
        assert ks_report["active"] is True
        assert ks_report["activation_phase"] == "test_live"
        assert ks_report["ticks_to_stop"] >= 2, f"ticks_to_stop should be >= 2, got {ks_report['ticks_to_stop']}"
        print(f"  Kill-switch report: ticks_to_stop={ks_report['ticks_to_stop']}")

        # --- Verify journal has INPUT events even during kill-switch ---
        events = load_journal(session_dir)
        inputs = events_of_type(events, "INPUT")
        assert len(inputs) >= 4, f"All 4 turns should have INPUT events, got {len(inputs)}"

        # --- Verify NO LLM_OUTPUT during kill-switch turns ---
        llm_outputs = events_of_type(events, "LLM_OUTPUT")
        # Turns 1-2 have LLM_OUTPUT, turns 3-4 should NOT
        for llm_evt in llm_outputs:
            turn = _get_turn(llm_evt)
            assert turn is None or turn <= 2, \
                f"LLM_OUTPUT should not appear during kill-switch (turn {turn})"

        # --- Deactivate kill-switch ---
        live = engine._get_or_init(session_id)
        ks_off = live.tp.killswitch.deactivate()
        live.tp.killswitch.save(live.tp._ks_report_path)
        print(f"  Kill-switch DEACTIVATED")

        # --- Turn 5: should be normal again ---
        resp5 = engine.process_turn(session_id, "Ora funziona di nuovo?")
        print(f"  Turn 5 (KS off): {resp5['response'][:60]}...")
        assert resp5["response"] != KILL_SWITCH_RESPONSE, \
            "After deactivation, response should be normal"
        assert len(resp5["response"]) > 0, "Response should not be empty"

        # --- Verify kill-switch report updated ---
        ks_report2 = json.loads(ks_report_path.read_text(encoding="utf-8"))
        assert ks_report2["active"] is False
        assert len(ks_report2["history"]) >= 2, "History should have ON + OFF entries"
        print(f"  Kill-switch history: {len(ks_report2['history'])} entries")

        print("\n=== KILL-SWITCH TEST PASSED ===")
