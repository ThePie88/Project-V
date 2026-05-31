"""V2.5 — StateEngine plugin framework tests.

Covers:
- U-STA-400: plugin registers engine_id/version, is queryable
- I-STA-410: swap plugin → output identical if same logic
- P-STA-420: seed + default_ode → same deltas as original hardcoded (backward compat)
- E-EXAM-430: scenario exam with engine metadata in trace
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pie.contracts.state import State
from pie.state_engine.protocol import StateEnginePlugin
from pie.state_engine.registry import StateEngineRegistry
from pie.state_engine.plugins.default_ode import DefaultODEPlugin


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_registry():
    """Ensure a clean registry for every test."""
    StateEngineRegistry.reset()
    yield
    StateEngineRegistry.reset()


# ---------------------------------------------------------------------------
# U-STA-400 — plugin registers engine_id/version, is queryable
# ---------------------------------------------------------------------------


def test_default_plugin_identity() -> None:
    plugin = DefaultODEPlugin()
    assert plugin.engine_id == "default_ode"
    assert plugin.version == "0.1"


def test_default_plugin_satisfies_protocol() -> None:
    plugin = DefaultODEPlugin()
    assert isinstance(plugin, StateEnginePlugin)


def test_registry_auto_registers_default() -> None:
    engine = StateEngineRegistry.get_active()
    assert engine.engine_id == "default_ode"


def test_registry_list_plugins() -> None:
    plugins = StateEngineRegistry.list_plugins()
    assert "default_ode" in plugins
    assert plugins["default_ode"] == "0.1"


def test_registry_set_active_unknown_raises() -> None:
    with pytest.raises(KeyError):
        StateEngineRegistry.set_active("nonexistent_engine")


# ---------------------------------------------------------------------------
# I-STA-410 — swap plugin → output identical if same logic
# ---------------------------------------------------------------------------


class CloneODEPlugin:
    """A clone of DefaultODEPlugin with a different engine_id."""

    @property
    def engine_id(self) -> str:
        return "clone_ode"

    @property
    def version(self) -> str:
        return "0.1"

    def update(self, state: State) -> State:
        # Exact same logic as DefaultODEPlugin
        new_state = state.model_copy(deep=True)

        def _clamp_round(value: float) -> float:
            v = min(max(value, 0.0), 1.0)
            return round(v, state.DECIMAL_PLACES)

        new_state.drives["curiosity"] = _clamp_round(state.drives["curiosity"] + 0.05)
        new_state.drives["sociality"] = _clamp_round(state.drives["sociality"] + 0.03)
        new_state.drives["fatigue"] = _clamp_round(state.drives["fatigue"] + 0.02)
        new_state.drives["caution"] = _clamp_round(
            state.drives["caution"] + 0.01 * state.drives["fatigue"]
        )
        new_state.affect["arousal"] = _clamp_round(
            state.affect["arousal"] + 0.04 * (state.drives["curiosity"] - 0.5)
        )
        new_state.affect["valence"] = _clamp_round(
            state.affect["valence"] + 0.04 * (state.drives["sociality"] - 0.5)
        )
        new_state.turn_count += 1
        return new_state


def test_swap_plugin_identical_output() -> None:
    state = State()

    # Run with default
    StateEngineRegistry.get_active()  # ensure default loaded
    result_default = state.update()

    # Register clone and swap
    StateEngineRegistry.register(CloneODEPlugin())
    StateEngineRegistry.set_active("clone_ode")
    result_clone = state.update()

    assert result_default.snapshot() == result_clone.snapshot()


def test_swap_plugin_back_and_forth() -> None:
    state = State()
    # Trigger lazy init of default_ode first
    StateEngineRegistry.get_active()
    StateEngineRegistry.register(CloneODEPlugin())

    StateEngineRegistry.set_active("default_ode")
    r1 = state.update()
    StateEngineRegistry.set_active("clone_ode")
    r2 = state.update()
    StateEngineRegistry.set_active("default_ode")
    r3 = state.update()

    assert r1.snapshot() == r2.snapshot() == r3.snapshot()


# ---------------------------------------------------------------------------
# P-STA-420 — backward compatibility: same deltas as original hardcoded
# ---------------------------------------------------------------------------


def test_backward_compat_single_update() -> None:
    """Default ODE plugin must produce exactly the same values as original."""
    state = State()
    updated = state.update()

    # Original hardcoded values (from M0 State.update):
    assert updated.drives["curiosity"] == round(0.5 + 0.05, 4)  # 0.55
    assert updated.drives["sociality"] == round(0.5 + 0.03, 4)  # 0.53
    assert updated.drives["fatigue"] == round(0.0 + 0.02, 4)  # 0.02
    assert updated.drives["caution"] == round(0.5 + 0.01 * 0.0, 4)  # 0.5
    assert updated.affect["arousal"] == round(0.5 + 0.04 * (0.5 - 0.5), 4)  # 0.5
    assert updated.affect["valence"] == round(0.5 + 0.04 * (0.5 - 0.5), 4)  # 0.5
    assert updated.turn_count == 1


def test_backward_compat_multi_update() -> None:
    """Multiple updates should chain deterministically."""
    state = State()
    for _ in range(5):
        state = state.update()
    assert state.turn_count == 5
    # Curiosity after 5 steps: 0.5 + 5*0.05 = 0.75
    assert state.drives["curiosity"] == 0.75
    # All values in [0,1]
    for v in state.drives.values():
        assert 0.0 <= v <= 1.0
    for v in state.affect.values():
        assert 0.0 <= v <= 1.0


def test_backward_compat_clamp_at_boundary() -> None:
    """Values must be clamped to [0, 1] even after many updates."""
    state = State()
    for _ in range(30):
        state = state.update()
    for v in state.drives.values():
        assert 0.0 <= v <= 1.0
    for v in state.affect.values():
        assert 0.0 <= v <= 1.0


# ---------------------------------------------------------------------------
# E-EXAM-430 — engine metadata in trace STATE_UPDATED events
# ---------------------------------------------------------------------------


def test_engine_metadata_in_exam(tmp_path: Path) -> None:
    """Exam run must include engine_id and engine_version in STATE_UPDATED."""
    from pie.runtime import run

    out = tmp_path / "artifacts"
    run(exam=True, llm="fake", output_dir=str(out))

    trace_path = out / "trace_exam.jsonl"
    assert trace_path.exists()

    state_events = []
    for line in trace_path.read_text(encoding="utf-8").splitlines():
        ev = json.loads(line)
        if ev.get("type") == "STATE_UPDATED":
            state_events.append(ev)

    assert len(state_events) > 0
    for ev in state_events:
        content = ev["content"]
        assert content["engine_id"] == "default_ode"
        assert content["engine_version"] == "0.1"
