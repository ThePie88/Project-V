"""Session lifecycle management (V3.0).

Provides ``SessionManager`` for creating, resuming and saving
persistent Ivy sessions, and the ``SessionContext`` dataclass that
holds all runtime state needed during a session.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..contracts.state import State
from ..persistence.atomic import atomic_write, atomic_append
from ..persistence.memory_store import MemoryStore
from ..persistence.constraints_store import ConstraintsStore
from .bootstrap import load_seed, build_identity_snapshot


# ---------------------------------------------------------------------------
# SessionContext — everything the runtime needs for one session
# ---------------------------------------------------------------------------

@dataclass
class SessionContext:
    """Runtime context for an active session."""

    session_id: str
    session_dir: Path
    state: State
    identity: Dict[str, Any]
    event_id: int
    turn_count: int
    journal_path: Path
    memory_store: MemoryStore
    constraints_store: ConstraintsStore


# ---------------------------------------------------------------------------
# SessionManager — create / resume / save / list
# ---------------------------------------------------------------------------

class SessionManager:
    """Manages persistent Ivy sessions on disk.

    Sessions are stored under ``sessions_root/<session_id>/`` with
    four files: ``identity_snapshot.json``, ``state_latest.json``,
    ``journal.jsonl``, and ``session_meta.json``.
    """

    REQUIRED_FILES = [
        "identity_snapshot.json",
        "state_latest.json",
        "journal.jsonl",
        "session_meta.json",
    ]

    def __init__(self, sessions_root: Path = Path("sessions")) -> None:
        self._root = sessions_root

    # -- public API ---------------------------------------------------------

    def create(
        self,
        seed_path: Path,
        session_id: Optional[str] = None,
    ) -> SessionContext:
        """Bootstrap a new session from a SEED file.

        Parameters
        ----------
        seed_path:
            Path to the SEED identity file (JSON content).
        session_id:
            Optional user-provided session ID.  If ``None`` an ID is
            generated deterministically from the seed content hash and
            the current wall-clock time (time is logging-only metadata).
        """
        seed_data = load_seed(seed_path)

        if session_id is None:
            session_id = self._generate_id(seed_path)

        session_dir = self._root / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        # 1. Identity snapshot
        identity = build_identity_snapshot(seed_data, seed_path)
        atomic_write(
            session_dir / "identity_snapshot.json",
            json.dumps(identity, indent=2, ensure_ascii=False),
        )

        # 2. Initial state from SEED
        state = State.from_seed(seed_data)
        atomic_write(
            session_dir / "state_latest.json",
            json.dumps(state.snapshot(), indent=2, ensure_ascii=False),
        )

        # 3. Session metadata
        seed_hash = hashlib.sha256(
            seed_path.read_bytes()
        ).hexdigest()
        meta = {
            "schema_version": "0.1",
            "session_id": session_id,
            "seed_path": str(seed_path),
            "seed_hash": seed_hash,
            "created_at": time.time(),
            "resumed_at_list": [],
            "turn_count": 0,
        }
        atomic_write(
            session_dir / "session_meta.json",
            json.dumps(meta, indent=2, ensure_ascii=False),
        )

        # 4. Empty journal (touch)
        journal_path = session_dir / "journal.jsonl"
        if not journal_path.exists():
            journal_path.touch()

        # 5. Session-scoped stores
        memory_store = MemoryStore(str(session_dir / "memory.jsonl"))
        constraints_store = ConstraintsStore(
            str(session_dir / "constraints.jsonl")
        )

        return SessionContext(
            session_id=session_id,
            session_dir=session_dir,
            state=state,
            identity=identity,
            event_id=1,
            turn_count=0,
            journal_path=journal_path,
            memory_store=memory_store,
            constraints_store=constraints_store,
        )

    def resume(self, session_id: str) -> SessionContext:
        """Resume an existing session from disk.

        Raises
        ------
        FileNotFoundError
            If the session directory or any required file is missing.
        ValueError
            If metadata is incompatible.
        """
        session_dir = self._root / session_id
        if not session_dir.is_dir():
            raise FileNotFoundError(
                f"Session not found: {session_dir}"
            )

        for fname in self.REQUIRED_FILES:
            if not (session_dir / fname).exists():
                raise FileNotFoundError(
                    f"Session {session_id} missing required file: {fname}"
                )

        # Load metadata
        meta = json.loads(
            (session_dir / "session_meta.json").read_text(encoding="utf-8")
        )

        # Load identity
        identity = json.loads(
            (session_dir / "identity_snapshot.json").read_text(
                encoding="utf-8"
            )
        )

        # Load state
        state_data = json.loads(
            (session_dir / "state_latest.json").read_text(encoding="utf-8")
        )
        state = State(
            drives=state_data.get("drives", {}),
            affect=state_data.get("affect", {}),
            turn_count=state_data.get("turn_count", 0),
            creator_anchor=state_data.get("creator_anchor", "MrPie"),
        )

        # Determine next event_id from journal
        journal_path = session_dir / "journal.jsonl"
        last_event_id = 0
        if journal_path.exists():
            for line in journal_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    evt = json.loads(line)
                    eid = evt.get("id", 0)
                    if eid > last_event_id:
                        last_event_id = eid
                except json.JSONDecodeError:
                    continue

        # Update resumed_at in metadata
        meta.setdefault("resumed_at_list", []).append(time.time())
        atomic_write(
            session_dir / "session_meta.json",
            json.dumps(meta, indent=2, ensure_ascii=False),
        )

        # Session-scoped stores
        memory_store = MemoryStore(str(session_dir / "memory.jsonl"))
        constraints_store = ConstraintsStore(
            str(session_dir / "constraints.jsonl")
        )

        # V6b-E1: Restore engine snapshot (if present)
        from pie.state_engine.snapshot import SnapshotSerializer
        from pie.state_engine.registry import StateEngineRegistry
        snap_hash = SnapshotSerializer.load(session_dir, StateEngineRegistry)
        if snap_hash is not None:
            event = {
                "schema_version": "0.1",
                "id": last_event_id + 1,
                "type": "SNAPSHOT_RESTORED",
                "timestamp": time.time(),
                "content": {"hash": snap_hash},
            }
            atomic_append(journal_path, json.dumps(event) + "\n")
            last_event_id += 1

        return SessionContext(
            session_id=session_id,
            session_dir=session_dir,
            state=state,
            identity=identity,
            event_id=last_event_id + 1,
            turn_count=state.turn_count,
            journal_path=journal_path,
            memory_store=memory_store,
            constraints_store=constraints_store,
        )

    def save(
        self,
        ctx: SessionContext,
        model_info: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Persist the current session state to disk.

        Updates ``state_latest.json`` and ``session_meta.json`` with the
        current turn count.  Also saves engine snapshot (V6b-E1) and
        session-captured environment/model_info (V6b-E4).
        """
        atomic_write(
            ctx.session_dir / "state_latest.json",
            json.dumps(ctx.state.snapshot(), indent=2, ensure_ascii=False),
        )

        # Update turn_count in metadata
        meta_path = ctx.session_dir / "session_meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["turn_count"] = ctx.turn_count
        atomic_write(
            meta_path,
            json.dumps(meta, indent=2, ensure_ascii=False),
        )

        # V6b-E1: Engine snapshot
        from pie.state_engine.snapshot import SnapshotSerializer
        from pie.state_engine.registry import StateEngineRegistry
        snap_hash = SnapshotSerializer.save(ctx.session_dir, StateEngineRegistry)
        event = {
            "schema_version": "0.1",
            "id": ctx.event_id,
            "type": "SNAPSHOT_SAVED",
            "timestamp": time.time(),
            "content": {"hash": snap_hash},
        }
        atomic_append(ctx.journal_path, json.dumps(event) + "\n")
        ctx.event_id += 1

        # V6b-E4: Session-captured environment
        env_data = self._capture_environment()
        atomic_write(
            ctx.session_dir / "environment.json",
            json.dumps(env_data, indent=2, sort_keys=True, ensure_ascii=False),
        )

        # V6b-E4: Session-captured model info
        if model_info is not None:
            atomic_write(
                ctx.session_dir / "model_info.json",
                json.dumps(
                    model_info, indent=2, sort_keys=True, ensure_ascii=False
                ),
            )

    def create_audit_bundle(self, ctx: SessionContext) -> Path:
        """Create audit bundle from session-captured files (V6b-E4).

        Call AFTER ``save()``.  The bundle reads files from disk and does
        NOT generate env/model_info (those are session-captured by save).

        Event ordering: ``AUDIT_BUNDLE_CREATED`` is emitted AFTER bundling.
        The bundle contains the trace up to the last save, NOT including
        this event.
        """
        from pie.audit.bundle import AuditBundler

        output_path = ctx.session_dir / f"audit_{ctx.session_id}.zip"
        manifest = AuditBundler.create(
            ctx.session_dir, output_path, session_id=ctx.session_id
        )

        # Anchor root_hash externally (append-only cross-session log)
        anchor_path = self._root / "audit_anchors.jsonl"
        anchor = {
            "session_id": ctx.session_id,
            "root_hash": manifest["root_hash"],
            "timestamp": time.time(),
        }
        atomic_append(anchor_path, json.dumps(anchor) + "\n")

        # Emit event to journal (AFTER bundle — not included in bundle trace)
        event = {
            "schema_version": "0.1",
            "id": ctx.event_id,
            "type": "AUDIT_BUNDLE_CREATED",
            "timestamp": time.time(),
            "content": {
                "root_hash": manifest["root_hash"],
                "path": str(output_path),
            },
        }
        atomic_append(ctx.journal_path, json.dumps(event) + "\n")
        ctx.event_id += 1
        return output_path

    def list_sessions(self) -> List[Dict[str, Any]]:
        """Return metadata for all sessions under the root directory."""
        sessions: List[Dict[str, Any]] = []
        if not self._root.exists():
            return sessions
        for child in sorted(self._root.iterdir()):
            meta_path = child / "session_meta.json"
            if child.is_dir() and meta_path.exists():
                try:
                    meta = json.loads(
                        meta_path.read_text(encoding="utf-8")
                    )
                    sessions.append(meta)
                except (json.JSONDecodeError, OSError):
                    continue
        return sessions

    # -- internal -----------------------------------------------------------

    @staticmethod
    def _capture_environment() -> Dict[str, Any]:
        """Capture runtime environment for audit (V6b-E4).

        Called at session save time (not bundle time) to ensure
        reproducible root_hash.
        """
        import platform
        import sys

        pip_freeze: List[str] = []
        try:
            import importlib.metadata

            dists = sorted(
                importlib.metadata.distributions(),
                key=lambda d: (d.metadata["Name"] or "").lower(),
            )
            pip_freeze = [
                f"{d.metadata['Name']}=={d.version}"
                for d in dists
                if d.metadata.get("Name")
            ]
        except Exception:
            pass

        return {
            "python_version": sys.version,
            "platform": sys.platform,
            "os_version": platform.platform(),
            "kernel_version": "0.0.0",
            "pip_freeze": pip_freeze,
        }

    def _generate_id(self, seed_path: Path) -> str:
        """Generate a deterministic-prefix session ID."""
        seed_bytes = seed_path.read_bytes()
        h = hashlib.sha256(seed_bytes + str(time.time()).encode()).hexdigest()
        return f"ivy_{h[:8]}"
