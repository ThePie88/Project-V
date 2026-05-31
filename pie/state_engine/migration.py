"""Versioned schema migrations for engine snapshots (V6b-E1).

Provides a registry for migration functions that transform snapshot
data from one schema version to the next.  Migrations are applied
as a chain: v1 -> v2 -> ... -> target.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Tuple


class MigrationRegistry:
    """Registry of snapshot schema migration functions."""

    _migrations: Dict[Tuple[int, int], Callable[[Dict[str, Any]], Dict[str, Any]]] = {}

    @classmethod
    def register(cls, from_ver: int, to_ver: int, fn: Callable[[Dict[str, Any]], Dict[str, Any]]) -> None:
        """Register a migration function for from_ver -> to_ver."""
        cls._migrations[(from_ver, to_ver)] = fn

    @classmethod
    def migrate(cls, data: Dict[str, Any], from_ver: int, to_ver: int) -> Dict[str, Any]:
        """Apply migration chain from_ver -> from_ver+1 -> ... -> to_ver."""
        current = from_ver
        while current < to_ver:
            key = (current, current + 1)
            if key not in cls._migrations:
                raise ValueError(f"No migration registered for {current} -> {current + 1}")
            data = cls._migrations[key](data)
            data["snapshot_schema_version"] = current + 1
            current += 1
        return data

    @classmethod
    def reset(cls) -> None:
        """Clear all registered migrations (for testing)."""
        cls._migrations.clear()
