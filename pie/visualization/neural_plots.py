"""Neural visualization — matplotlib plots for V4.5 artifacts.

Generates three plot types from neural engine artifacts:
1. Spike raster plot — turn vs neuron, dot on spike
2. Drive/affect trajectory — time series of all 10 channels
3. Bifurcation diagram — mu(t) with threshold and transition markers

All functions accept artifact dicts (as returned by NeuralSNNPlugin.get_artifacts()
and BifurcationDiagram.to_json()) and write PNG files to disk.

Usage:
    from pie.visualization.neural_plots import plot_all
    plot_all(artifacts, bifurcation_diagram, out_dir)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

# Lazy import — matplotlib is optional
_MPL_AVAILABLE: Optional[bool] = None


def _check_matplotlib() -> bool:
    """Check if matplotlib is available (cached)."""
    global _MPL_AVAILABLE
    if _MPL_AVAILABLE is None:
        try:
            import matplotlib  # noqa: F401
            _MPL_AVAILABLE = True
        except ImportError:
            _MPL_AVAILABLE = False
    return _MPL_AVAILABLE


# ── Neuron ordering (must match neural_snn.py) ──────────────────────

NEURON_ORDER = [
    "curiosity", "sociality", "caution", "agency", "playfulness", "fatigue",
    "valence", "arousal", "attention", "tension",
]

DRIVE_NAMES = {"curiosity", "sociality", "caution", "agency", "playfulness", "fatigue"}
AFFECT_NAMES = {"valence", "arousal", "attention", "tension"}


# ── 1. Spike Raster Plot ────────────────────────────────────────────

def plot_spike_raster(
    spike_log: List[Dict[str, Any]],
    out_path: Path,
    title: str = "Spike Raster Plot",
) -> Path:
    """Generate a raster plot of neural spikes.

    Args:
        spike_log: List of spike records with 'neuron_id' and 'turn' keys.
        out_path: Path to save PNG file.
        title: Plot title.

    Returns:
        Path to the saved PNG file.
    """
    if not _check_matplotlib():
        raise ImportError("matplotlib is required for visualization")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 5))

    neuron_to_y = {name: i for i, name in enumerate(NEURON_ORDER)}

    turns = []
    ys = []
    colors = []
    for spike in spike_log:
        nid = spike["neuron_id"]
        if nid in neuron_to_y:
            turns.append(spike["turn"])
            ys.append(neuron_to_y[nid])
            colors.append("#2196F3" if nid in DRIVE_NAMES else "#FF5722")

    ax.scatter(turns, ys, c=colors, s=20, marker="|", linewidths=1.5)
    ax.set_yticks(range(len(NEURON_ORDER)))
    ax.set_yticklabels(NEURON_ORDER, fontsize=9)
    ax.set_xlabel("Turn")
    ax.set_ylabel("Neuron")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.3)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="|", color="#2196F3", label="Drive", markersize=10, linestyle="None"),
        Line2D([0], [0], marker="|", color="#FF5722", label="Affect", markersize=10, linestyle="None"),
    ]
    ax.legend(handles=legend_elements, loc="upper right")

    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)
    return out_path


# ── 2. Drive/Affect Trajectory Plot ─────────────────────────────────

def plot_trajectory(
    trajectory: List[Dict[str, Dict[str, float]]],
    out_path: Path,
    title: str = "Drive/Affect Trajectory",
) -> Path:
    """Generate time series plot for all 10 neuron channels.

    Args:
        trajectory: List of per-turn neuron state dicts.
            Each entry: {"curiosity": {"membrane_potential": 0.5}, ...}
        out_path: Path to save PNG file.
        title: Plot title.

    Returns:
        Path to the saved PNG file.
    """
    if not _check_matplotlib():
        raise ImportError("matplotlib is required for visualization")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_drive, ax_affect) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    turns_range = list(range(1, len(trajectory) + 1))

    # Drive channels
    drive_colors = ["#1976D2", "#388E3C", "#F57C00", "#7B1FA2", "#C2185B", "#455A64"]
    drive_neurons = [n for n in NEURON_ORDER if n in DRIVE_NAMES]
    for i, name in enumerate(drive_neurons):
        values = []
        for t in trajectory:
            if name in t:
                val = t[name]
                if isinstance(val, dict):
                    val = val.get("membrane_potential", 0.0)
                values.append(val)
            else:
                values.append(0.0)
        ax_drive.plot(turns_range[:len(values)], values, label=name,
                      color=drive_colors[i % len(drive_colors)], linewidth=1.5)

    ax_drive.set_ylabel("Normalized Value [0,1]")
    ax_drive.set_title(f"{title} — Drives")
    ax_drive.legend(loc="upper right", fontsize=8)
    ax_drive.set_ylim(-0.05, 1.05)
    ax_drive.grid(alpha=0.3)

    # Affect channels
    affect_colors = ["#E91E63", "#FF9800", "#009688", "#673AB7"]
    affect_neurons = [n for n in NEURON_ORDER if n in AFFECT_NAMES]
    for i, name in enumerate(affect_neurons):
        values = []
        for t in trajectory:
            if name in t:
                val = t[name]
                if isinstance(val, dict):
                    val = val.get("membrane_potential", 0.0)
                values.append(val)
            else:
                values.append(0.0)
        ax_affect.plot(turns_range[:len(values)], values, label=name,
                       color=affect_colors[i % len(affect_colors)], linewidth=1.5)

    ax_affect.set_xlabel("Turn")
    ax_affect.set_ylabel("Normalized Value [0,1]")
    ax_affect.set_title(f"{title} — Affect")
    ax_affect.legend(loc="upper right", fontsize=8)
    ax_affect.set_ylim(-0.05, 1.05)
    ax_affect.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)
    return out_path


# ── 3. Bifurcation Diagram Plot ─────────────────────────────────────

def plot_bifurcation_diagram(
    diagram_data: Dict[str, Any],
    out_path: Path,
    title: str = "Bifurcation Diagram",
) -> Path:
    """Generate bifurcation diagram: mu(t) with threshold markers.

    Args:
        diagram_data: Dict with 'points' list, each having
            'turn', 'mu', 'bifurcated', 'event_label'.
        out_path: Path to save PNG file.
        title: Plot title.

    Returns:
        Path to the saved PNG file.
    """
    if not _check_matplotlib():
        raise ImportError("matplotlib is required for visualization")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    points = diagram_data.get("points", [])
    if not points:
        # Empty diagram — create minimal plot
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.set_title(f"{title} (no data)")
        ax.set_xlabel("Turn")
        ax.set_ylabel("mu (spike density)")
        fig.savefig(str(out_path), dpi=150)
        plt.close(fig)
        return out_path

    fig, ax = plt.subplots(figsize=(10, 5))

    turns = [p["turn"] for p in points]
    mus = [p["mu"] for p in points]
    bifurcated = [p["bifurcated"] for p in points]

    # Plot mu trajectory
    ax.plot(turns, mus, color="#1565C0", linewidth=1.5, label="mu(t)")

    # Highlight bifurcated regions
    bif_turns = [t for t, b in zip(turns, bifurcated) if b]
    bif_mus = [m for m, b in zip(mus, bifurcated) if b]
    if bif_turns:
        ax.scatter(bif_turns, bif_mus, color="#D32F2F", s=40, zorder=5,
                   label="Bifurcated")

    # Mark bifurcation events with labels
    event_colors = {
        "BIFURCATION_ANXIETY": "#E91E63",
        "BIFURCATION_IMPULSIVITY": "#FF9800",
        "BIFURCATION_EXHAUSTION": "#795548",
    }
    for p in points:
        label = p.get("event_label")
        if label:
            color = event_colors.get(label, "#D32F2F")
            ax.annotate(
                label.replace("BIFURCATION_", ""),
                xy=(p["turn"], p["mu"]),
                xytext=(0, 15),
                textcoords="offset points",
                fontsize=7,
                color=color,
                fontweight="bold",
                ha="center",
            )

    # Threshold lines
    ax.axhline(y=0.6, color="#F44336", linestyle="--", alpha=0.7, label="mu_enter=0.6")
    ax.axhline(y=0.4, color="#4CAF50", linestyle="--", alpha=0.7, label="mu_exit=0.4")

    ax.set_xlabel("Turn")
    ax.set_ylabel("mu (spike density)")
    ax.set_title(title)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)
    return out_path


# ── Convenience: plot all from artifacts ─────────────────────────────

def plot_all(
    engine_artifacts: Dict[str, Any],
    bifurcation_diagram: Optional[Dict[str, Any]] = None,
    trajectory: Optional[List[Dict[str, Dict[str, float]]]] = None,
    out_dir: Path = Path("artifacts"),
) -> List[Path]:
    """Generate all available neural plots.

    Args:
        engine_artifacts: From NeuralSNNPlugin.get_artifacts().
        bifurcation_diagram: From BifurcationDiagram JSON (optional).
        trajectory: Per-turn neuron states for trajectory plot (optional).
        out_dir: Directory to save plots.

    Returns:
        List of paths to generated PNG files.
    """
    if not _check_matplotlib():
        raise ImportError("matplotlib is required for visualization")

    out_dir.mkdir(parents=True, exist_ok=True)
    generated: List[Path] = []

    # 1. Spike raster
    spike_log = engine_artifacts.get("spike_log", [])
    if spike_log:
        path = plot_spike_raster(spike_log, out_dir / "spike_raster.png")
        generated.append(path)

    # 2. Trajectory (if provided)
    if trajectory:
        path = plot_trajectory(trajectory, out_dir / "trajectory.png")
        generated.append(path)

    # 3. Bifurcation diagram (if provided)
    if bifurcation_diagram:
        path = plot_bifurcation_diagram(
            bifurcation_diagram, out_dir / "bifurcation_diagram.png"
        )
        generated.append(path)

    return generated
