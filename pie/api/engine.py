"""SessionEngine — stateless turn-processing wrapper (V6b-E2).

Wraps SessionManager + TurnProcessor for API-driven single-turn calls.
Per-session locking prevents interleaved requests from corrupting state.
Seed paths are resolved from an allowlist (no arbitrary filesystem access).
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from pie.session.manager import SessionContext, SessionManager
from pie.runtime import TurnProcessor
from pie.state_engine.snapshot import SnapshotSerializer
from pie.state_engine.registry import StateEngineRegistry
from pie.dashboard.reader import DashboardReader

# ---------------------------------------------------------------------------
# Kernel / API version constants
# ---------------------------------------------------------------------------

KERNEL_VERSION = "0.0.0"
API_VERSION = "1"
SCHEMA_VERSION = "1"
TELEMETRY_VERSION = "1"

# ---------------------------------------------------------------------------
# Seed allowlist
# ---------------------------------------------------------------------------

_DEFAULT_SEEDS: Dict[str, Path] = {
    "SEED_V0": Path("progetto/SEED_V0.md"),
}


def _resolve_seed(seed_id: str, seeds_root: Optional[Path] = None) -> Path:
    """Resolve a seed_id to a filesystem path via allowlist.

    Raises ValueError if the seed_id is not in the allowlist.
    """
    if seed_id in _DEFAULT_SEEDS:
        base = seeds_root or Path(".")
        return base / _DEFAULT_SEEDS[seed_id]
    raise ValueError(
        f"Unknown seed_id: {seed_id!r}. "
        f"Available: {sorted(_DEFAULT_SEEDS.keys())}"
    )


# ---------------------------------------------------------------------------
# Live session holder
# ---------------------------------------------------------------------------

class _LiveSession:
    """In-memory state for a session that has been initialized."""

    __slots__ = ("ctx", "tp", "lock")

    def __init__(self, ctx: SessionContext, tp: TurnProcessor) -> None:
        self.ctx = ctx
        self.tp = tp
        self.lock = threading.Lock()


# ---------------------------------------------------------------------------
# Telemetry delta builder
# ---------------------------------------------------------------------------

def _build_telemetry_delta(
    reader: DashboardReader,
    events_before: int,
) -> Dict[str, Any]:
    """Build telemetry delta from journal events added this turn.

    ``events_before`` is the count of journal events before the turn.
    We return only events added after that index.
    """
    all_events = reader.journal_events()
    new_events = all_events[events_before:]

    # Extract CV channels from STATE_UPDATED in this turn
    cv_snapshot: Dict[str, float] = {}
    spikes: List[Dict[str, Any]] = []
    gating: List[Dict[str, Any]] = []
    budget_delta: Dict[str, Any] = {}

    for evt in new_events:
        etype = evt.get("type", "")
        content = evt.get("content", {})

        if etype == "STATE_UPDATED":
            em = content.get("engine_metadata", {})
            cv = em.get("control_vector", {})
            if cv:
                cv_snapshot = {k: float(v) for k, v in cv.items()}
            spike_log = em.get("spike_log", [])
            spikes = spike_log

        elif etype == "CV_GATING":
            src = content if content.get("canale") else evt
            gating.append({
                "canale": src.get("canale", ""),
                "valore": src.get("valore", 0.0),
                "decisione": src.get("decisione", ""),
                "effetto": src.get("effetto", ""),
                "reason": src.get("reason", ""),
            })

    budget_summary = reader.budget_summary()
    if budget_summary:
        budget_delta = budget_summary

    return {
        "cv_channels": cv_snapshot,
        "spikes": spikes,
        "gating_decisions": gating,
        "budget": budget_delta,
        "new_event_count": len(new_events),
    }


# ---------------------------------------------------------------------------
# SessionEngine
# ---------------------------------------------------------------------------

class SessionEngine:
    """Wraps SessionManager + runtime for stateless single-turn API calls.

    Thread-safe: each session has its own lock. Concurrent turns on the
    *same* session are serialized. Different sessions can proceed in parallel.
    """

    def __init__(
        self,
        sessions_root: Path = Path("sessions"),
        seeds_root: Optional[Path] = None,
        llm: str = "fake",
        no_cache: bool = False,
        engine: str = "ode",
    ) -> None:
        self._mgr = SessionManager(sessions_root)
        self._seeds_root = seeds_root
        self._llm = llm
        self._no_cache = no_cache
        self._sessions: Dict[str, _LiveSession] = {}
        self._global_lock = threading.Lock()

        # Register neural engine if requested
        if engine == "neural":
            from pie.state_engine.plugins.neural_snn import NeuralSNNPlugin
            plugin = NeuralSNNPlugin(reservoir_enabled=True)
            StateEngineRegistry.register(plugin)
            StateEngineRegistry.set_active(plugin.engine_id)

    # -- helpers ------------------------------------------------------------

    def _response_meta(self, session_id: str, turn_id: Optional[int] = None) -> Dict[str, Any]:
        """Standard metadata included in every API response."""
        meta: Dict[str, Any] = {
            "kernel_version": KERNEL_VERSION,
            "api_version": API_VERSION,
            "schema_version": SCHEMA_VERSION,
            "session_id": session_id,
        }
        if turn_id is not None:
            meta["turn_id"] = turn_id
        return meta

    def _get_or_init(self, session_id: str) -> _LiveSession:
        """Get live session, or lazy-init from disk."""
        with self._global_lock:
            if session_id in self._sessions:
                return self._sessions[session_id]

        # Resume from disk (outside global lock — IO bound)
        ctx = self._mgr.resume(session_id)
        tp = TurnProcessor(ctx, llm=self._llm, no_cache=self._no_cache)
        live = _LiveSession(ctx, tp)

        with self._global_lock:
            # Double-check: another thread may have raced us
            if session_id not in self._sessions:
                self._sessions[session_id] = live
            return self._sessions[session_id]

    # -- public API ---------------------------------------------------------

    def create_session(
        self,
        seed_id: str = "SEED_V0",
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a new session from an allowlisted seed.

        Returns session metadata dict.
        """
        seed_path = _resolve_seed(seed_id, self._seeds_root)
        if not seed_path.exists():
            raise FileNotFoundError(
                f"Seed file not found: {seed_path}. "
                "Run from the repository root or pass seeds_root to SessionEngine."
            )

        ctx = self._mgr.create(seed_path, session_id=session_id)
        tp = TurnProcessor(ctx, llm=self._llm, no_cache=self._no_cache)
        live = _LiveSession(ctx, tp)

        with self._global_lock:
            self._sessions[ctx.session_id] = live

        return {
            **self._response_meta(ctx.session_id),
            "created_at": time.time(),
            "seed_id": seed_id,
        }

    def process_turn(self, session_id: str, user_input: str) -> Dict[str, Any]:
        """Process one turn. Returns response + state + telemetry_delta.

        Raises FileNotFoundError if session doesn't exist.
        """
        live = self._get_or_init(session_id)

        with live.lock:
            reader = DashboardReader(live.ctx.session_dir)
            events_before = len(reader.journal_events())

            response_text = live.tp.process(user_input)
            turn_id = live.tp.turn_index

            # Re-read for telemetry delta
            reader = DashboardReader(live.ctx.session_dir)
            telemetry = _build_telemetry_delta(reader, events_before)

            state = live.tp.state.snapshot()

        return {
            **self._response_meta(session_id, turn_id=turn_id),
            "telemetry_version": TELEMETRY_VERSION,
            "response": response_text,
            "turn_count": turn_id,
            "state": state,
            "telemetry_delta": telemetry,
        }

    def submit_reward(
        self,
        session_id: str,
        target_turn: int,
        value: int,
        source: str = "thumb",
        reason: Optional[str] = None,
        actor_id: str = "local_ui",
    ) -> Dict[str, Any]:
        """Submit a reward signal for a completed turn. Race-safe."""
        live = self._get_or_init(session_id)
        with live.lock:
            result = live.tp.submit_reward(
                target_turn, value, source, reason, actor_id
            )
        return {
            **self._response_meta(session_id),
            **result,
        }

    def get_state(self, session_id: str) -> Dict[str, Any]:
        """Read-only: current state snapshot."""
        live = self._get_or_init(session_id)
        with live.lock:
            state = live.tp.state.snapshot()
            turn_id = live.tp.turn_index
        return {
            **self._response_meta(session_id, turn_id=turn_id),
            "state": state,
        }

    def get_journal(
        self,
        session_id: str,
        limit: int = 100,
        offset: int = 0,
        event_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Read-only: journal events with pagination."""
        live = self._get_or_init(session_id)
        with live.lock:
            reader = DashboardReader(live.ctx.session_dir)
            events = reader.journal_events(event_type=event_type)
            total = len(events)
            page = events[offset:offset + limit]
        return {
            **self._response_meta(session_id),
            "events": page,
            "total": total,
            "offset": offset,
            "limit": limit,
        }

    def get_telemetry(
        self,
        session_id: str,
        turn_start: int = 0,
        turn_end: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Bulk telemetry for an existing session.

        Returns CV channels, spikes, gating, budget across a turn range.
        """
        live = self._get_or_init(session_id)
        with live.lock:
            reader = DashboardReader(live.ctx.session_dir)
            return {
                **self._response_meta(session_id),
                "telemetry_version": TELEMETRY_VERSION,
                "cv_channels": reader.cv_channels_timeseries(),
                "spike_rate": reader.spike_rate(),
                "gating_decisions": reader.cv_gating_table(),
                "budget_summary": reader.budget_summary(),
                "budget_timeseries": reader.budget_timeseries(),
                "tool_stats": reader.tool_stats(),
                "memory_counts": reader.memory_counts(),
            }

    def save_snapshot(self, session_id: str) -> Dict[str, Any]:
        """Save engine snapshot (E1 integration)."""
        live = self._get_or_init(session_id)
        with live.lock:
            snap_hash = SnapshotSerializer.save(
                live.ctx.session_dir, StateEngineRegistry
            )
        return {
            **self._response_meta(session_id),
            "hash": snap_hash,
        }

    def restore_snapshot(self, session_id: str) -> Dict[str, Any]:
        """Restore engine snapshot (E1 integration)."""
        live = self._get_or_init(session_id)
        with live.lock:
            snap_hash = SnapshotSerializer.load(
                live.ctx.session_dir, StateEngineRegistry
            )
            if snap_hash is None:
                raise FileNotFoundError(
                    f"No snapshot found for session {session_id}"
                )
        return {
            **self._response_meta(session_id),
            "hash": snap_hash,
        }
