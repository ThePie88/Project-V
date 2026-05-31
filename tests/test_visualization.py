"""V4.5 — Neural Visualization & Artifact File Tests.

Test IDs:
- E-NART-010: All neural artifact files present after neural exam run
- E-NART-020: Visualization generates valid PNG files
- E-NART-030: spike_log.jsonl records are valid JSON
- E-NART-040: neuron_state.jsonl has all 10 neurons
- E-NART-050: weight_history.jsonl records are valid STDP records
- E-NART-060: plot_all generates at least spike_raster.png
"""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import pytest

from pie.state_engine.plugins.neural_snn import (
    NeuralSNNPlugin,
    NEURON_ORDER,
    SpikeEvent,
)
from pie.bifurcation import BifurcationDetector, BifurcationDiagram


# ── Fixtures ─────────────────────────────────────────────────────────

@pytest.fixture
def neural_plugin():
    """Create a neural plugin and run several updates to generate spikes."""
    from pie.contracts.state import State
    plugin = NeuralSNNPlugin(reservoir_enabled=True, stdp_enabled=True)
    state = State()
    # Run enough turns to produce spikes
    for _ in range(10):
        state = plugin.update(state)
    return plugin, state


@pytest.fixture
def spike_log_data() -> List[Dict[str, Any]]:
    """Synthetic spike log for plot testing."""
    return [
        {"neuron_id": "curiosity", "turn": 1, "v_before": 0.6, "v_after": 0.1},
        {"neuron_id": "arousal", "turn": 1, "v_before": 0.7, "v_after": 0.1},
        {"neuron_id": "curiosity", "turn": 2, "v_before": 0.5, "v_after": 0.1},
        {"neuron_id": "tension", "turn": 3, "v_before": 0.8, "v_after": 0.1},
        {"neuron_id": "agency", "turn": 4, "v_before": 0.6, "v_after": 0.1},
        {"neuron_id": "fatigue", "turn": 5, "v_before": 0.7, "v_after": 0.1},
        {"neuron_id": "sociality", "turn": 5, "v_before": 0.6, "v_after": 0.1},
        {"neuron_id": "playfulness", "turn": 6, "v_before": 0.5, "v_after": 0.1},
        {"neuron_id": "valence", "turn": 7, "v_before": 0.7, "v_after": 0.1},
        {"neuron_id": "attention", "turn": 8, "v_before": 0.8, "v_after": 0.1},
    ]


@pytest.fixture
def trajectory_data() -> List[Dict[str, Dict[str, float]]]:
    """Synthetic trajectory for plot testing."""
    data = []
    for t in range(20):
        entry = {}
        for name in NEURON_ORDER:
            # Simple sinusoidal with different phases
            idx = NEURON_ORDER.index(name)
            val = 0.5 + 0.3 * math.sin(t * 0.5 + idx * 0.3)
            entry[name] = {"membrane_potential": round(val, 4)}
        data.append(entry)
    return data


@pytest.fixture
def bifurcation_diagram_data() -> Dict[str, Any]:
    """Synthetic bifurcation diagram data."""
    points = []
    for t in range(20):
        mu = 0.3 + 0.03 * t  # ramp up
        bifurcated = mu >= 0.6
        label = "BIFURCATION_ANXIETY" if t == 10 else None
        points.append({
            "turn": t + 1,
            "mu": round(mu, 4),
            "bifurcated": bifurcated,
            "event_label": label,
        })
    return {"schema_version": "0.1", "type": "bifurcation_diagram", "points": points}


# ── E-NART-020: Visualization generates valid PNGs ───────────────────

class TestVisualizationPlots:
    """Test that matplotlib plot functions generate valid PNG files."""

    def test_spike_raster_png(self, spike_log_data, tmp_path):
        """E-NART-020a: spike raster plot generates PNG."""
        from pie.visualization.neural_plots import plot_spike_raster
        out = plot_spike_raster(spike_log_data, tmp_path / "raster.png")
        assert out.exists()
        assert out.stat().st_size > 1000  # non-trivial PNG
        # Check PNG magic bytes
        with open(out, "rb") as f:
            header = f.read(8)
        assert header[:4] == b"\x89PNG"

    def test_trajectory_png(self, trajectory_data, tmp_path):
        """E-NART-020b: trajectory plot generates PNG."""
        from pie.visualization.neural_plots import plot_trajectory
        out = plot_trajectory(trajectory_data, tmp_path / "trajectory.png")
        assert out.exists()
        assert out.stat().st_size > 1000
        with open(out, "rb") as f:
            header = f.read(4)
        assert header == b"\x89PNG"

    def test_bifurcation_diagram_png(self, bifurcation_diagram_data, tmp_path):
        """E-NART-020c: bifurcation diagram plot generates PNG."""
        from pie.visualization.neural_plots import plot_bifurcation_diagram
        out = plot_bifurcation_diagram(
            bifurcation_diagram_data, tmp_path / "bifurc.png"
        )
        assert out.exists()
        assert out.stat().st_size > 1000
        with open(out, "rb") as f:
            header = f.read(4)
        assert header == b"\x89PNG"

    def test_empty_bifurcation_diagram(self, tmp_path):
        """Empty diagram produces a valid PNG (no crash)."""
        from pie.visualization.neural_plots import plot_bifurcation_diagram
        out = plot_bifurcation_diagram(
            {"points": []}, tmp_path / "empty.png"
        )
        assert out.exists()


# ── E-NART-060: plot_all convenience function ────────────────────────

class TestPlotAll:
    """Test the plot_all convenience function."""

    def test_plot_all_generates_files(
        self, spike_log_data, trajectory_data, bifurcation_diagram_data, tmp_path
    ):
        """E-NART-060: plot_all generates at least spike_raster.png."""
        from pie.visualization.neural_plots import plot_all

        artifacts = {"spike_log": spike_log_data}
        paths = plot_all(
            artifacts,
            bifurcation_diagram=bifurcation_diagram_data,
            trajectory=trajectory_data,
            out_dir=tmp_path,
        )
        names = [p.name for p in paths]
        assert "spike_raster.png" in names
        assert "trajectory.png" in names
        assert "bifurcation_diagram.png" in names
        assert len(paths) == 3
        for p in paths:
            assert p.exists()


# ── E-NART-010: Artifact files from neural plugin ───────────────────

class TestNeuralArtifactFiles:
    """Test that neural artifact files can be generated correctly."""

    def test_neural_artifacts_json(self, neural_plugin, tmp_path):
        """E-NART-010a: neural_artifacts.json is valid."""
        plugin, state = neural_plugin
        artifacts = plugin.get_artifacts()
        out = tmp_path / "neural_artifacts.json"
        out.write_text(json.dumps(artifacts, indent=2), encoding="utf-8")
        # Re-read and validate
        loaded = json.loads(out.read_text(encoding="utf-8"))
        assert loaded["engine_id"] == "neural_snn"
        assert loaded["neuron_model"] == "izhikevich"
        assert "neuron_states" in loaded
        assert "spike_log" in loaded

    def test_spike_log_jsonl(self, neural_plugin, tmp_path):
        """E-NART-030: spike_log.jsonl records are valid JSON."""
        plugin, _ = neural_plugin
        artifacts = plugin.get_artifacts()
        out = tmp_path / "spike_log.jsonl"
        with out.open("w", encoding="utf-8") as f:
            for spike in artifacts.get("spike_log", []):
                f.write(json.dumps(spike) + "\n")
        # Re-read and validate each line
        lines = out.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) >= 1  # at least some spikes after 10 turns
        for line in lines:
            if line.strip():
                record = json.loads(line)
                assert "neuron_id" in record
                assert "turn" in record

    def test_neuron_state_jsonl(self, neural_plugin, tmp_path):
        """E-NART-040: neuron_state.jsonl has all 10 neurons."""
        plugin, _ = neural_plugin
        artifacts = plugin.get_artifacts()
        out = tmp_path / "neuron_state.jsonl"
        with out.open("w", encoding="utf-8") as f:
            for name, data in sorted(artifacts.get("neuron_states", {}).items()):
                record = {"neuron_id": name, **data}
                f.write(json.dumps(record) + "\n")
        # Re-read and validate
        lines = [l for l in out.read_text(encoding="utf-8").strip().split("\n") if l.strip()]
        assert len(lines) == 10
        neuron_ids = set()
        for line in lines:
            record = json.loads(line)
            neuron_ids.add(record["neuron_id"])
            assert "membrane_potential" in record
            assert "firing_pattern" in record
        assert neuron_ids == set(NEURON_ORDER)

    def test_weight_history_jsonl(self, neural_plugin, tmp_path):
        """E-NART-050: weight_history.jsonl records are valid STDP records."""
        plugin, _ = neural_plugin
        tracker = plugin._stdp_tracker
        if tracker is None:
            pytest.skip("STDP tracker not enabled")
        out = tmp_path / "weight_history.jsonl"
        with out.open("w", encoding="utf-8") as f:
            for rec in tracker.log:
                f.write(json.dumps(rec.to_dict()) + "\n")
        # Validate structure
        content = out.read_text(encoding="utf-8").strip()
        if content:
            for line in content.split("\n"):
                record = json.loads(line)
                assert "turn" in record
                assert "source" in record
                assert "target" in record
                assert "rule" in record
                assert record["rule"] in ("potentiation", "depression")


# ── Integration: from artifacts to plots ─────────────────────────────

class TestEndToEndVisualization:
    """Integration: generate artifacts from neural plugin, then plot."""

    def test_artifacts_to_plots(self, neural_plugin, tmp_path):
        """Full pipeline: neural plugin → artifacts → plots."""
        from pie.visualization.neural_plots import plot_spike_raster

        plugin, state = neural_plugin
        artifacts = plugin.get_artifacts()
        spike_log = artifacts.get("spike_log", [])

        if not spike_log:
            pytest.skip("No spikes generated in 10 turns")

        out = plot_spike_raster(spike_log, tmp_path / "e2e_raster.png")
        assert out.exists()
        assert out.stat().st_size > 1000
