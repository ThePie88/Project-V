"""Tests for V6b-E3 — Dashboard Read-Only.

E-DSH-010..040: invariants, smoke, data correctness, post-mortem.
"""

from __future__ import annotations

import json
import re
import textwrap
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def session_dir(tmp_path: Path) -> Path:
    """Create a minimal session directory with sample data."""
    d = tmp_path / "test_session"
    d.mkdir()

    # session_meta.json
    (d / "session_meta.json").write_text(json.dumps({
        "schema_version": "0.1",
        "session_id": "test_session",
        "seed_path": "progetto/SEED_V0.md",
        "seed_hash": "abc123",
        "created_at": 1700000000.0,
        "resumed_at_list": [],
        "turn_count": 3,
    }), encoding="utf-8")

    # state_latest.json
    (d / "state_latest.json").write_text(json.dumps({
        "drives": {"curiosity": 0.7, "attachment": 0.5, "safety": 0.8},
        "affect": {"valence": 0.6, "arousal": 0.4},
        "turn_count": 3,
    }), encoding="utf-8")

    # journal.jsonl with mixed event types
    events = [
        # STATE_UPDATED with engine_metadata
        {
            "id": 1, "type": "STATE_UPDATED", "timestamp": 1700000001.0,
            "content": {
                "engine_metadata": {
                    "control_vector": {
                        "memory_gate": 0.5, "tool_gate": 0.8,
                        "cf_k": 3, "verbosity_bias": 0.6,
                        "consolidation_urgency": 0.2,
                    },
                    "spike_log": [
                        {"neuron_id": "n0", "time": 1, "voltage_peak": 30.0, "isi": None},
                        {"neuron_id": "n1", "time": 2, "voltage_peak": 30.0, "isi": None},
                        {"neuron_id": "n0", "time": 3, "voltage_peak": 30.0, "isi": 2},
                    ],
                },
            },
        },
        # STATE_UPDATED turn 2
        {
            "id": 2, "type": "STATE_UPDATED", "timestamp": 1700000002.0,
            "content": {
                "engine_metadata": {
                    "control_vector": {
                        "memory_gate": 0.6, "tool_gate": 0.3,
                        "cf_k": 2, "verbosity_bias": 0.7,
                        "consolidation_urgency": 0.4,
                    },
                    "spike_log": [
                        {"neuron_id": "n0", "time": 1, "voltage_peak": 30.0, "isi": None},
                    ],
                },
            },
        },
        # CV_GATING events
        {
            "id": 3, "type": "CV_GATING", "timestamp": 1700000003.0,
            "content": {
                "canale": "tool_gate",
                "valore": 0.8,
                "decisione": "ALLOW_TOOLS",
                "effetto": "tools_enabled",
                "reason": "tool_gate >= 0.3",
            },
        },
        {
            "id": 4, "type": "CV_GATING", "timestamp": 1700000004.0,
            "content": {
                "canale": "tool_gate",
                "valore": 0.1,
                "decisione": "BLOCK_TOOLS",
                "effetto": "tools_disabled",
                "reason": "tool_gate < 0.3",
            },
        },
        # TOOL_CALL and TOOL_DENIED
        {
            "id": 5, "type": "TOOL_CALL", "timestamp": 1700000005.0,
            "content": {"tool": "search", "args": {}},
        },
        {
            "id": 6, "type": "TOOL_CALL", "timestamp": 1700000006.0,
            "content": {"tool": "write_file", "args": {}},
        },
        {
            "id": 7, "type": "TOOL_DENIED", "timestamp": 1700000007.0,
            "content": {"tool": "write_file", "deny_reason": "capability_missing"},
        },
        {
            "id": 8, "type": "TOOL_DENIED", "timestamp": 1700000008.0,
            "content": {"tool": "delete", "deny_reason": "rate_limit"},
        },
    ]
    journal_lines = "\n".join(json.dumps(e) for e in events) + "\n"
    (d / "journal.jsonl").write_text(journal_lines, encoding="utf-8")

    # metabolism.json
    (d / "metabolism.json").write_text(json.dumps({
        "schema_version": "0.1",
        "budget": {
            "initial_budget": 500,
            "current_budget": 350,
            "consumed": 150,
            "remaining_ratio": 0.7,
            "consumption_history": [
                {"turn": 1, "cost": {"total": 80}, "remaining": 420, "timestamp": 1700000001.0},
                {"turn": 2, "cost": {"total": 50}, "remaining": 370, "timestamp": 1700000002.0},
                {"turn": 3, "cost": {"total": 20}, "remaining": 350, "timestamp": 1700000003.0},
            ],
        },
    }), encoding="utf-8")

    # memory.jsonl
    memories = [
        {"type": "NarrativeMemory", "memory_id": "m1", "content": {"text": "hello"}},
        {"type": "NarrativeMemory", "memory_id": "m2", "content": {"text": "world"}},
        {"type": "Belief", "memory_id": "m3", "content": {"belief": "sky is blue"}},
        {"type": "Preference", "memory_id": "m4", "content": {"pref": "dark mode"}},
    ]
    mem_lines = "\n".join(json.dumps(m) for m in memories) + "\n"
    (d / "memory.jsonl").write_text(mem_lines, encoding="utf-8")

    return d


# ---------------------------------------------------------------------------
# E-DSH-010 — No write endpoints
# ---------------------------------------------------------------------------


class TestReadOnlyInvariants:
    """E-DSH-010/011: Verify dashboard is strictly read-only."""

    def test_e_dsh_010_no_write_endpoints(self):
        """app.py must not define POST/PUT/DELETE/PATCH routes."""
        app_source = (
            Path(__file__).parent.parent / "pie" / "dashboard" / "app.py"
        ).read_text(encoding="utf-8")

        for method in ["post", "put", "delete", "patch"]:
            pattern = rf"@app\.{method}\b"
            assert not re.search(pattern, app_source), (
                f"Found @app.{method} in app.py — violates read-only invariant"
            )

    def test_e_dsh_011_no_write_imports(self):
        """reader.py and app.py must not import write functions."""
        for fname in ["reader.py", "app.py"]:
            source = (
                Path(__file__).parent.parent / "pie" / "dashboard" / fname
            ).read_text(encoding="utf-8")
            for forbidden in ["atomic_write", "atomic_append", "write_text",
                              "write_bytes", 'open("w"', "open('w'"]:
                assert forbidden not in source, (
                    f"Found '{forbidden}' in {fname} — violates read-only invariant"
                )


# ---------------------------------------------------------------------------
# E-DSH-020 — Smoke: all reader methods return data
# ---------------------------------------------------------------------------


class TestReaderSmoke:
    """E-DSH-020: All DashboardReader methods work on valid session."""

    def test_e_dsh_020_all_methods_return_data(self, session_dir: Path):
        from pie.dashboard.reader import DashboardReader

        reader = DashboardReader(session_dir)

        meta = reader.session_meta()
        assert meta["session_id"] == "test_session"
        assert meta["turn_count"] == 3

        state = reader.state_latest()
        assert "drives" in state
        assert state["drives"]["curiosity"] == 0.7

        events = reader.journal_events()
        assert len(events) == 8

        filtered = reader.journal_events(event_type="CV_GATING")
        assert len(filtered) == 2

        limited = reader.journal_events(limit=3)
        assert len(limited) == 3

        gating = reader.cv_gating_table()
        assert len(gating) >= 1

        cv = reader.cv_channels_timeseries()
        assert len(cv) > 0

        spikes = reader.spike_rate()
        assert len(spikes) > 0

        budget = reader.budget_summary()
        assert budget["consumed"] == 150

        budget_ts = reader.budget_timeseries()
        assert len(budget_ts) == 3

        tools = reader.tool_stats()
        assert tools["total"] > 0

        memory = reader.memory_counts()
        assert sum(memory.values()) == 4


# ---------------------------------------------------------------------------
# E-DSH-021 — CV Gating table structure
# ---------------------------------------------------------------------------


class TestCVGating:
    """E-DSH-021: cv_gating_table returns correct structure."""

    def test_e_dsh_021_gating_table_structure(self, session_dir: Path):
        from pie.dashboard.reader import DashboardReader

        rows = DashboardReader(session_dir).cv_gating_table()
        assert len(rows) == 2

        row0 = rows[0]
        assert row0["canale"] == "tool_gate"
        assert row0["valore"] == 0.8
        assert row0["decisione"] == "ALLOW_TOOLS"
        assert row0["effetto"] == "tools_enabled"
        assert row0["reason"] == "tool_gate >= 0.3"
        assert row0["event_id"] == 3

        row1 = rows[1]
        assert row1["decisione"] == "BLOCK_TOOLS"


# ---------------------------------------------------------------------------
# E-DSH-022 — CV channels time series
# ---------------------------------------------------------------------------


class TestCVChannels:
    """E-DSH-022: cv_channels_timeseries matches journal events."""

    def test_e_dsh_022_cv_timeseries(self, session_dir: Path):
        from pie.dashboard.reader import DashboardReader

        cv = DashboardReader(session_dir).cv_channels_timeseries()

        # Two STATE_UPDATED events → 2 data points per channel
        assert "memory_gate" in cv
        assert cv["memory_gate"] == [0.5, 0.6]
        assert cv["tool_gate"] == [0.8, 0.3]
        assert cv["verbosity_bias"] == [0.6, 0.7]


# ---------------------------------------------------------------------------
# E-DSH-023 — Spike rate
# ---------------------------------------------------------------------------


class TestSpikeRate:
    """E-DSH-023: spike_rate matches journal spike events."""

    def test_e_dsh_023_spike_rate(self, session_dir: Path):
        from pie.dashboard.reader import DashboardReader

        spikes = DashboardReader(session_dir).spike_rate()

        # Turn 1: n0=2, n1=1. Turn 2: n0=1, n1=0
        assert spikes["n0"] == [2, 1]
        assert spikes["n1"] == [1, 0]


# ---------------------------------------------------------------------------
# E-DSH-024 — Budget summary
# ---------------------------------------------------------------------------


class TestBudget:
    """E-DSH-024: budget_summary matches metabolism.json."""

    def test_e_dsh_024_budget_summary(self, session_dir: Path):
        from pie.dashboard.reader import DashboardReader

        budget = DashboardReader(session_dir).budget_summary()
        assert budget["initial_budget"] == 500
        assert budget["current_budget"] == 350
        assert budget["consumed"] == 150
        assert budget["remaining_ratio"] == 0.7

    def test_e_dsh_024_budget_timeseries(self, session_dir: Path):
        from pie.dashboard.reader import DashboardReader

        ts = DashboardReader(session_dir).budget_timeseries()
        assert len(ts) == 3
        assert ts[0]["turn"] == 1
        assert ts[2]["remaining"] == 350


# ---------------------------------------------------------------------------
# E-DSH-025 — Tool stats
# ---------------------------------------------------------------------------


class TestToolStats:
    """E-DSH-025: tool_stats deny rate computed correctly."""

    def test_e_dsh_025_tool_stats(self, session_dir: Path):
        from pie.dashboard.reader import DashboardReader

        stats = DashboardReader(session_dir).tool_stats()
        assert stats["tool_calls"] == 2
        assert stats["tool_denied"] == 2
        assert stats["total"] == 4
        assert stats["deny_rate"] == 0.5
        assert stats["deny_reasons"]["capability_missing"] == 1
        assert stats["deny_reasons"]["rate_limit"] == 1


# ---------------------------------------------------------------------------
# E-DSH-026 — Memory counts
# ---------------------------------------------------------------------------


class TestMemoryCounts:
    """E-DSH-026: memory_counts matches memory.jsonl types."""

    def test_e_dsh_026_memory_counts(self, session_dir: Path):
        from pie.dashboard.reader import DashboardReader

        counts = DashboardReader(session_dir).memory_counts()
        assert counts["NarrativeMemory"] == 2
        assert counts["Belief"] == 1
        assert counts["Preference"] == 1


# ---------------------------------------------------------------------------
# E-DSH-030 — Cross-check consistency
# ---------------------------------------------------------------------------


class TestConsistency:
    """E-DSH-030: displayed data consistent with journal."""

    def test_e_dsh_030_gating_events_match_journal(self, session_dir: Path):
        from pie.dashboard.reader import DashboardReader

        reader = DashboardReader(session_dir)
        gating_rows = reader.cv_gating_table()
        gating_events = reader.journal_events(event_type="CV_GATING")

        assert len(gating_rows) == len(gating_events)
        for row, evt in zip(gating_rows, gating_events):
            # Support both formats (content-wrapped and top-level)
            src = evt.get("content", {})
            if not src.get("canale"):
                src = evt
            assert row["canale"] == src["canale"]
            assert row["valore"] == src["valore"]


# ---------------------------------------------------------------------------
# E-DSH-040 — Post-mortem (no live state needed)
# ---------------------------------------------------------------------------


class TestPostMortem:
    """E-DSH-040: dashboard works on completed session (no live state)."""

    def test_e_dsh_040_works_on_completed_session(self, session_dir: Path):
        """Dashboard reads files only — no SessionContext needed."""
        from pie.dashboard.reader import DashboardReader

        reader = DashboardReader(session_dir)

        # All methods should work without any live state object
        assert reader.session_meta()["session_id"] == "test_session"
        assert reader.state_latest()["drives"]["curiosity"] == 0.7
        assert len(reader.journal_events()) > 0
        assert len(reader.cv_gating_table()) > 0
        assert len(reader.cv_channels_timeseries()) > 0
        assert len(reader.spike_rate()) > 0
        assert reader.budget_summary()["consumed"] == 150
        assert len(reader.budget_timeseries()) > 0
        assert reader.tool_stats()["total"] > 0
        assert sum(reader.memory_counts().values()) > 0


# ---------------------------------------------------------------------------
# Edge cases — empty / missing files
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Graceful handling of empty or missing session files."""

    def test_empty_session_dir(self, tmp_path: Path):
        from pie.dashboard.reader import DashboardReader

        d = tmp_path / "empty"
        d.mkdir()
        reader = DashboardReader(d)

        assert reader.session_meta() == {}
        assert reader.state_latest() == {}
        assert reader.journal_events() == []
        assert reader.cv_gating_table() == []
        assert reader.cv_channels_timeseries() == {}
        assert reader.spike_rate() == {}
        assert reader.budget_summary() == {}
        assert reader.budget_timeseries() == []
        assert reader.tool_stats() == {
            "tool_calls": 0, "tool_denied": 0,
            "total": 0, "deny_rate": 0.0, "deny_reasons": {},
        }
        assert reader.memory_counts() == {}

    def test_no_engine_metadata(self, tmp_path: Path):
        """STATE_UPDATED without engine_metadata → empty CV/spikes."""
        from pie.dashboard.reader import DashboardReader

        d = tmp_path / "no_engine"
        d.mkdir()
        (d / "journal.jsonl").write_text(
            json.dumps({"id": 1, "type": "STATE_UPDATED", "content": {}}) + "\n",
            encoding="utf-8",
        )
        reader = DashboardReader(d)
        assert reader.cv_channels_timeseries() == {}
        assert reader.spike_rate() == {}


# ---------------------------------------------------------------------------
# E-DSH-050 — Runtime format: top-level CV_GATING + neural engine metadata
# ---------------------------------------------------------------------------


class TestRuntimeFormat:
    """Test with event formats as actually emitted by run_session()."""

    @pytest.fixture()
    def neural_session_dir(self, tmp_path: Path) -> Path:
        """Session with runtime-format events (neural_snn engine)."""
        d = tmp_path / "neural_session"
        d.mkdir()

        (d / "session_meta.json").write_text(json.dumps({
            "schema_version": "0.1",
            "session_id": "neural_session",
            "seed_path": "progetto/SEED_V0.md",
            "seed_hash": "abc",
            "created_at": 1700000000.0,
            "resumed_at_list": [],
            "turn_count": 3,
        }), encoding="utf-8")

        (d / "state_latest.json").write_text(json.dumps({
            "drives": {"curiosity": 0.7, "sociality": 0.6},
            "affect": {"arousal": 0.3, "valence": 0.5},
            "turn_count": 3,
        }), encoding="utf-8")

        # Journal with runtime-format events
        events = [
            # STATE_UPDATED with engine_metadata (neural_snn format)
            {
                "schema_version": "0.1", "id": 1,
                "type": "STATE_UPDATED", "timestamp": 1700000001.0,
                "content": {
                    "logical_time": {"turn": 1, "step": 2},
                    "engine_id": "neural_snn", "engine_version": "0.3",
                    "drives": {"curiosity": 0.75},
                    "affect": {"arousal": 0.3},
                    "engine_metadata": {
                        "schema_version": "0.3",
                        "engine_id": "neural_snn",
                        "neuron_states": {
                            "curiosity_excitatory": {
                                "membrane_potential": 0.5,
                                "v_internal": -65.0,
                                "u_recovery": -14.0,
                            },
                        },
                        "spike_log": [
                            {"neuron_id": "curiosity_excitatory", "time": 1,
                             "voltage_peak": 30.0, "isi": None},
                            {"neuron_id": "curiosity_excitatory", "time": 3,
                             "voltage_peak": 30.0, "isi": 2},
                            {"neuron_id": "caution_inhibitory", "time": 2,
                             "voltage_peak": 30.0, "isi": None},
                        ],
                        "total_spikes": 3,
                        "control_vector": {
                            "memory_gate": 0.45,
                            "tool_gate": 0.72,
                            "cf_k": 3,
                            "verbosity_bias": 0.55,
                            "consolidation_urgency": 0.18,
                        },
                        "reservoir": {"spectral_radius": 0.95},
                    },
                },
            },
            # STATE_UPDATED turn 2
            {
                "schema_version": "0.1", "id": 10,
                "type": "STATE_UPDATED", "timestamp": 1700000010.0,
                "content": {
                    "logical_time": {"turn": 2, "step": 2},
                    "engine_id": "neural_snn", "engine_version": "0.3",
                    "drives": {"curiosity": 0.80},
                    "affect": {"arousal": 0.4},
                    "engine_metadata": {
                        "spike_log": [
                            {"neuron_id": "curiosity_excitatory", "time": 1,
                             "voltage_peak": 30.0, "isi": None},
                        ],
                        "total_spikes": 1,
                        "control_vector": {
                            "memory_gate": 0.60,
                            "tool_gate": 0.35,
                            "cf_k": 2,
                            "verbosity_bias": 0.70,
                            "consolidation_urgency": 0.40,
                        },
                    },
                },
            },
            # STATE_UPDATED turn 3
            {
                "schema_version": "0.1", "id": 20,
                "type": "STATE_UPDATED", "timestamp": 1700000020.0,
                "content": {
                    "logical_time": {"turn": 3, "step": 2},
                    "engine_id": "neural_snn", "engine_version": "0.3",
                    "drives": {"curiosity": 0.65},
                    "affect": {"arousal": 0.2},
                    "engine_metadata": {
                        "spike_log": [],
                        "total_spikes": 0,
                        "control_vector": {
                            "memory_gate": 0.30,
                            "tool_gate": 0.90,
                            "cf_k": 4,
                            "verbosity_bias": 0.40,
                            "consolidation_urgency": 0.10,
                        },
                    },
                },
            },
            # CV_GATING events — runtime format (top-level fields, event_id not id)
            {
                "event_id": 5, "type": "CV_GATING",
                "canale": "memory_gate", "valore": 0.45,
                "decisione": "RECALL_K_5", "effetto": "recall_k=5",
                "reason": "memory_gate=0.45",
                "logical_time": {"turn": 2, "step": 0},
            },
            {
                "event_id": 11, "type": "CV_GATING",
                "canale": "tool_gate", "valore": 0.72,
                "decisione": "ALLOW_TOOLS", "effetto": "tools_enabled",
                "reason": "tool_gate >= 0.3",
                "logical_time": {"turn": 1, "step": 4},
            },
            {
                "event_id": 15, "type": "CV_GATING",
                "canale": "tool_gate", "valore": 0.35,
                "decisione": "ALLOW_TOOLS", "effetto": "tools_enabled",
                "reason": "tool_gate >= 0.3",
                "logical_time": {"turn": 2, "step": 4},
            },
            {
                "event_id": 25, "type": "CV_GATING",
                "canale": "verbosity_bias", "valore": 0.40,
                "decisione": "VERBOSITY_LOW", "effetto": "verbosity=low",
                "reason": "verbosity_bias < 0.5",
                "logical_time": {"turn": 3, "step": 5},
            },
            # TOOL_CALL and TOOL_DENIED
            {
                "schema_version": "0.1", "id": 12,
                "type": "TOOL_CALL", "timestamp": 1700000012.0,
                "content": {"tool": "search", "args": {"query": "test"}},
            },
            {
                "schema_version": "0.1", "id": 16,
                "type": "TOOL_DENIED", "timestamp": 1700000016.0,
                "content": {"tool": "write_file", "deny_reason": "capability_missing"},
            },
        ]
        journal_lines = "\n".join(json.dumps(e) for e in events) + "\n"
        (d / "journal.jsonl").write_text(journal_lines, encoding="utf-8")

        # metabolism.json
        (d / "metabolism.json").write_text(json.dumps({
            "schema_version": "0.1",
            "budget": {
                "initial_budget": 500,
                "current_budget": 440,
                "consumed": 60,
                "remaining_ratio": 0.88,
                "consumption_history": [
                    {"turn": 1, "cost": {"total": 20}, "remaining": 480},
                    {"turn": 2, "cost": {"total": 20}, "remaining": 460},
                    {"turn": 3, "cost": {"total": 20}, "remaining": 440},
                ],
            },
        }), encoding="utf-8")

        # memory.jsonl
        memories = [
            {"type": "NarrativeMemory", "memory_id": "m1", "content": {"text": "hello"}},
            {"type": "NarrativeMemory", "memory_id": "m2", "content": {"text": "world"}},
            {"type": "Belief", "memory_id": "m3", "content": {"belief": "sky"}},
            {"type": "Preference", "memory_id": "m4", "content": {"pref": "dark"}},
            {"type": "TrustUpdate", "memory_id": "m5", "content": {"trust": 0.8}},
        ]
        (d / "memory.jsonl").write_text(
            "\n".join(json.dumps(m) for m in memories) + "\n",
            encoding="utf-8",
        )

        return d

    def test_cv_gating_runtime_format(self, neural_session_dir: Path):
        """CV_GATING with top-level fields (runtime format) are parsed."""
        from pie.dashboard.reader import DashboardReader

        rows = DashboardReader(neural_session_dir).cv_gating_table()
        assert len(rows) == 4

        # First row: memory_gate
        assert rows[0]["canale"] == "memory_gate"
        assert rows[0]["valore"] == 0.45
        assert rows[0]["decisione"] == "RECALL_K_5"
        assert rows[0]["event_id"] == 5

        # tool_gate rows
        tool_rows = [r for r in rows if r["canale"] == "tool_gate"]
        assert len(tool_rows) == 2

        # verbosity
        verb_rows = [r for r in rows if r["canale"] == "verbosity_bias"]
        assert len(verb_rows) == 1
        assert verb_rows[0]["decisione"] == "VERBOSITY_LOW"

    def test_cv_channels_with_engine_metadata(self, neural_session_dir: Path):
        """CV channels from engine_metadata across 3 turns."""
        from pie.dashboard.reader import DashboardReader

        cv = DashboardReader(neural_session_dir).cv_channels_timeseries()

        assert "memory_gate" in cv
        assert len(cv["memory_gate"]) == 3
        assert cv["memory_gate"] == [0.45, 0.60, 0.30]
        assert cv["tool_gate"] == [0.72, 0.35, 0.90]
        assert cv["verbosity_bias"] == [0.55, 0.70, 0.40]

    def test_spike_rate_with_engine_metadata(self, neural_session_dir: Path):
        """Spike rate from engine_metadata.spike_log across 3 turns."""
        from pie.dashboard.reader import DashboardReader

        spikes = DashboardReader(neural_session_dir).spike_rate()

        assert "curiosity_excitatory" in spikes
        assert "caution_inhibitory" in spikes
        # Turn 1: curiosity=2, caution=1. Turn 2: curiosity=1, caution=0. Turn 3: both=0
        assert spikes["curiosity_excitatory"] == [2, 1, 0]
        assert spikes["caution_inhibitory"] == [1, 0, 0]

    def test_tool_stats_mixed(self, neural_session_dir: Path):
        """Tool stats with both calls and denies."""
        from pie.dashboard.reader import DashboardReader

        stats = DashboardReader(neural_session_dir).tool_stats()
        assert stats["tool_calls"] == 1
        assert stats["tool_denied"] == 1
        assert stats["total"] == 2
        assert stats["deny_rate"] == 0.5

    def test_memory_counts_all_types(self, neural_session_dir: Path):
        """Memory counts include all 4 types."""
        from pie.dashboard.reader import DashboardReader

        counts = DashboardReader(neural_session_dir).memory_counts()
        assert counts["NarrativeMemory"] == 2
        assert counts["Belief"] == 1
        assert counts["Preference"] == 1
        assert counts["TrustUpdate"] == 1
        assert sum(counts.values()) == 5

    def test_budget_with_3_turns(self, neural_session_dir: Path):
        """Budget timeseries with 3 turns."""
        from pie.dashboard.reader import DashboardReader

        reader = DashboardReader(neural_session_dir)
        summary = reader.budget_summary()
        assert summary["consumed"] == 60
        assert summary["remaining_ratio"] == 0.88

        ts = reader.budget_timeseries()
        assert len(ts) == 3
        assert ts[0]["remaining"] == 480
        assert ts[2]["remaining"] == 440
