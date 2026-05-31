"""Engine snapshot serialization (V6b-E1).

Central orchestrator that collects all neural engine state into one
atomic JSON file (``engine_snapshot.json``) and restores it exactly.

The save-restore-save cycle is idempotent: ``hash(snap1) == hash(snap2)``
thanks to ``sort_keys=True`` and deterministic rounding in each plugin's
``serialize()`` method.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional

from pie.persistence.atomic import atomic_write
from pie.state_engine.migration import MigrationRegistry


class SnapshotSerializer:
    """Serialize and restore engine state to/from a single JSON file."""

    SCHEMA_VERSION = 1
    FILENAME = "engine_snapshot.json"

    @classmethod
    def save(cls, session_dir: Path, registry: type) -> str:
        """Serialize active engine state to engine_snapshot.json.

        Returns SHA-256 hex digest of the written content.
        """
        plugin = registry.get_active()
        engine_state: Dict[str, Any] = {}
        if hasattr(plugin, "serialize"):
            engine_state = plugin.serialize()

        payload = {
            "snapshot_schema_version": cls.SCHEMA_VERSION,
            "active_engine_id": plugin.engine_id,
            "engine_version": plugin.version,
            "engine_state": engine_state,
        }
        content = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False)
        atomic_write(session_dir / cls.FILENAME, content)
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    @classmethod
    def load(cls, session_dir: Path, registry: type) -> Optional[str]:
        """Restore engine state from engine_snapshot.json.

        Returns SHA-256 hex digest of loaded content, or None if no snapshot.
        """
        path = session_dir / cls.FILENAME
        if not path.exists():
            return None

        content = path.read_text(encoding="utf-8")
        data = json.loads(content)

        # Migration chain if needed
        version = data.get("snapshot_schema_version", 1)
        if version < cls.SCHEMA_VERSION:
            data = MigrationRegistry.migrate(data, version, cls.SCHEMA_VERSION)

        # Ensure correct plugin is active
        engine_id = data["active_engine_id"]
        available = registry.list_plugins()
        if engine_id in available:
            registry.set_active(engine_id)

        plugin = registry.get_active()
        if hasattr(plugin, "deserialize") and data.get("engine_state"):
            plugin.deserialize(data["engine_state"])

        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    @classmethod
    def compute_hash(cls, session_dir: Path) -> Optional[str]:
        """Compute hash of existing snapshot without loading.

        Uses read_text + encode to normalize line endings on Windows.
        """
        path = session_dir / cls.FILENAME
        if not path.exists():
            return None
        content = path.read_text(encoding="utf-8")
        return hashlib.sha256(content.encode("utf-8")).hexdigest()
