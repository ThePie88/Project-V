"""StateEngine registry — manages plugin discovery and selection.

The registry is a singleton that holds all registered plugins and
routes ``update()`` calls to the active backend.  The default ODE
plugin is registered automatically on first access.
"""

from __future__ import annotations

from typing import Dict, Optional

from .protocol import StateEnginePlugin


class StateEngineRegistry:
    """Global registry for state-engine plugins."""

    _plugins: Dict[str, StateEnginePlugin] = {}
    _active: Optional[str] = None

    @classmethod
    def register(cls, plugin: StateEnginePlugin) -> None:
        """Register a plugin.  Overwrites if same ``engine_id`` exists."""
        cls._plugins[plugin.engine_id] = plugin

    @classmethod
    def get_active(cls) -> StateEnginePlugin:
        """Return the currently active plugin (lazy-init default)."""
        if cls._active is None or cls._active not in cls._plugins:
            cls._ensure_default()
        return cls._plugins[cls._active]  # type: ignore[index]

    @classmethod
    def set_active(cls, engine_id: str) -> None:
        """Select a registered plugin as the active backend."""
        if engine_id not in cls._plugins:
            raise KeyError(f"No plugin registered with engine_id={engine_id!r}")
        cls._active = engine_id

    @classmethod
    def list_plugins(cls) -> Dict[str, str]:
        """Return ``{engine_id: version}`` for all registered plugins."""
        cls._ensure_default()
        return {pid: p.version for pid, p in cls._plugins.items()}

    @classmethod
    def reset(cls) -> None:
        """Clear all plugins and the active selection (for testing)."""
        cls._plugins.clear()
        cls._active = None

    @classmethod
    def _ensure_default(cls) -> None:
        """Lazily register the default ODE plugin if nothing else is present."""
        if "default_ode" not in cls._plugins:
            from .plugins.default_ode import DefaultODEPlugin

            cls.register(DefaultODEPlugin())
        if cls._active is None:
            cls._active = "default_ode"
