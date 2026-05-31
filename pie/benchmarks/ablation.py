"""Ablation study runner: Izhikevich vs Linear ODE (V6a-R1).

Three conditions × two tasks, with per-condition readout training.
Each condition trains its own readout on its own state representation
using the same ridge regression procedure.

Conditions:
1. Linear ODE (10-dim) — grid search 3×2=6 configs, best reported
2. Izhikevich only (10-dim) — Tier-1 neurons, no reservoir
3. Izhikevich + Reservoir (128-dim) — full system

Tasks:
- Delayed XOR: accuracy (higher = better)
- NARMA-10: NRMSE (lower = better)
- Memory Capacity: MC (higher = better)

Reference: Lukoševičius (2012) "A Practical Guide to Applying ESNs"
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pie.benchmarks.linalg import (
    solve_ridge,
    mean,
    std,
    r_squared,
)
from pie.benchmarks.temporal_tasks import (
    generate_delayed_xor,
    generate_narma10,
    generate_memory_capacity,
)
from pie.state_engine.plugins.linear_ode import (
    LinearODEPlugin,
    grid_configs,
)
from pie.state_engine.plugins.reservoir import (
    LCG,
    Reservoir,
    RESERVOIR_SIZE,
    INPUT_SIZE,
)
from pie.state_engine.plugins.readout_training import deterministic_split


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def nrmse(predicted: List[float], actual: List[float]) -> float:
    """Normalized Root Mean Squared Error: sqrt(MSE) / std(actual)."""
    if not predicted:
        return float('inf')
    mse = sum((p - a) ** 2 for p, a in zip(predicted, actual)) / len(predicted)
    s = std(actual)
    if s < 1e-12:
        return float('inf')
    return math.sqrt(mse) / s


def correlation(x: List[float], y: List[float]) -> float:
    """Pearson correlation coefficient."""
    n = len(x)
    if n < 2:
        return 0.0
    mx = mean(x)
    my = mean(y)
    num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    den_x = math.sqrt(sum((xi - mx) ** 2 for xi in x))
    den_y = math.sqrt(sum((yi - my) ** 2 for yi in y))
    if den_x < 1e-12 or den_y < 1e-12:
        return 0.0
    return num / (den_x * den_y)


def memory_capacity(
    predicted_delays: List[List[float]],
    actual_delays: List[List[float]],
) -> float:
    """Memory capacity: MC = sum_k corr(y_k, u(t-k))^2."""
    mc = 0.0
    for pred_k, act_k in zip(predicted_delays, actual_delays):
        c = correlation(pred_k, act_k)
        mc += c * c
    return mc


def separation_ratio(
    states_a: List[List[float]],
    states_b: List[List[float]],
    inputs_a: List[float],
    inputs_b: List[float],
) -> float:
    """Separation ratio: avg RMS_dist(state_A, state_B) / avg dist(input_A, input_B).

    Uses per-dimension RMS distance (Euclidean / sqrt(dim)) so that
    comparison across different feature dimensions is fair.
    """
    n = min(len(states_a), len(states_b))
    if n == 0:
        return 0.0

    state_dists = []
    input_dists = []
    for i in range(n):
        dim = len(states_a[i])
        sd = math.sqrt(sum((a - b) ** 2 for a, b in zip(states_a[i], states_b[i])))
        # Normalize by sqrt(dim) → per-dimension RMS distance
        if dim > 0:
            sd /= math.sqrt(dim)
        state_dists.append(sd)
        id_ = abs(inputs_a[i] - inputs_b[i])
        input_dists.append(id_)

    avg_sd = mean(state_dists)
    avg_id = mean(input_dists)
    if avg_id < 1e-12:
        return float('inf') if avg_sd > 1e-12 else 1.0
    return avg_sd / avg_id


# ---------------------------------------------------------------------------
# Engine wrappers (uniform interface for ablation)
# ---------------------------------------------------------------------------

class _LinearEngine:
    """Wrapper for Linear ODE in ablation.

    Scalar input projected to 10-dim via deterministic LCG-seeded
    projection vector.  Each dimension gets different sensitivity.
    """

    def __init__(self, decay: float, gain: float, proj_seed: int = 0xABCD):
        self._decay = decay
        self._gain = gain
        # Deterministic input projection (scalar → 10-dim)
        rng = LCG(proj_seed)
        self._projection = [rng.next_float() * 0.8 + 0.1 for _ in range(10)]
        self.plugin = LinearODEPlugin(decay=decay, gain=gain)
        self.name = f"linear_ode(d={decay},g={gain})"
        self.feature_dim = 10

    def reset(self):
        self.plugin = LinearODEPlugin(decay=self._decay, gain=self._gain)

    def step(self, input_val: float) -> List[float]:
        """Feed scalar input, return state vector."""
        inp = [input_val * p for p in self._projection]
        return self.plugin.step_with_input(inp)


class _IzhOnlyEngine:
    """Wrapper for Izhikevich-only (no reservoir).

    Directly uses IzhikevichNeuron objects with proper temporal dynamics.
    Maintains neuron state across steps — the key for memory/nonlinearity.
    """

    def __init__(self, seed: int = 0xDEADBEEF):
        from pie.state_engine.plugins.neural_snn import (
            IzhikevichNeuron, NEURON_TYPES,
            NEURON_ORDER as N_ORDER,
            SYNAPSE_WEIGHTS, BASE_CURRENTS,
        )
        self.seed = seed
        self._neuron_order = N_ORDER
        self._neuron_types = NEURON_TYPES
        self._synapse_weights = SYNAPSE_WEIGHTS
        self._base_currents = BASE_CURRENTS
        self._neurons: Dict = {}
        self._init_neurons()
        # Input projection (scalar → per-neuron current, diverse)
        rng = LCG(seed ^ 0x55555555)
        self._projection = {
            name: rng.next_float() * 0.06 + 0.02
            for name in self._neuron_order
        }
        self.name = "izh_only"
        self.feature_dim = 10

    def _init_neurons(self):
        from pie.state_engine.plugins.neural_snn import IzhikevichNeuron
        self._neurons = {}
        for name in self._neuron_order:
            params = self._neuron_types.get(name, {})
            self._neurons[name] = IzhikevichNeuron(
                neuron_id=name,
                a=params.get("a", 0.02),
                b=params.get("b", 0.2),
                c=params.get("c", -65.0),
                d=params.get("d", 8.0),
            )

    def reset(self):
        self._init_neurons()

    def step(self, input_val: float) -> List[float]:
        """Feed scalar input as current, return 10-dim neuron potentials."""
        # Compute currents: base + synaptic + external input
        currents: Dict[str, float] = {}
        for name in self._neuron_order:
            c = self._base_currents.get(name, 0.0)
            for (src, tgt), w in sorted(self._synapse_weights.items()):
                if tgt == name and src in self._neurons:
                    c += w * self._neurons[src].membrane_potential
                    if self._neurons[src].last_spike:
                        c += w * 0.5
            # External input with per-neuron projection
            c += input_val * self._projection[name]
            currents[name] = c

        # Step all neurons in deterministic order
        for name in self._neuron_order:
            self._neurons[name].step(currents[name])

        # Normalize to [0, 1] like reservoir does: (v + 80) / 110
        return [
            max(0.0, min(1.0, (self._neurons[n].membrane_potential + 80.0) / 110.0))
            for n in self._neuron_order
        ]


class _IzhReservoirEngine:
    """Wrapper for Izhikevich + Reservoir (full ESN system).

    Feeds the projected input directly into the 128-neuron ESN.
    This is the standard reservoir computing setup: the reservoir's
    internal Izhikevich neurons provide the nonlinear temporal processing.
    128-dim leak_state is the feature vector for readout.

    The reservoir already has Izhikevich neurons internally, so adding
    Tier-1 Izh on top would be double-processing (not standard ESN).
    """

    def __init__(self, seed: int = 0xDEADBEEF, proj_seed: int = 0xABCD):
        self.seed = seed
        self._reservoir = Reservoir(seed=seed)
        # Same projection as linear ODE for fair input comparison
        rng = LCG(proj_seed)
        self._projection = [rng.next_float() * 0.8 + 0.1 for _ in range(INPUT_SIZE)]
        self.name = "izh_reservoir"
        self.feature_dim = RESERVOIR_SIZE  # 128

    def reset(self):
        self._reservoir = Reservoir(seed=self.seed)

    def step(self, input_val: float) -> List[float]:
        """Feed scalar → 10-dim projection → 128-neuron ESN → 128-dim state."""
        inp = [input_val * p for p in self._projection]
        self._reservoir.step(inp)
        return self._reservoir.get_neuron_states()


# ---------------------------------------------------------------------------
# Ablation Runner
# ---------------------------------------------------------------------------

@dataclass
class AblationResult:
    """Results for one condition on one task."""
    condition: str
    task: str
    metric_name: str
    metric_value: float
    feature_dim: int
    config: Optional[Dict[str, Any]] = None


@dataclass
class AblationReport:
    """Full ablation report."""
    results: List[AblationResult] = field(default_factory=list)
    seed: int = 42
    schema_version: str = "1.0"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "seed": self.seed,
            "results": [
                {
                    "condition": r.condition,
                    "task": r.task,
                    "metric_name": r.metric_name,
                    "metric_value": round(r.metric_value, 6),
                    "feature_dim": r.feature_dim,
                    "config": r.config,
                }
                for r in self.results
            ],
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def best_for(self, task: str, condition_prefix: str) -> Optional[AblationResult]:
        """Find best result for a task/condition prefix."""
        candidates = [
            r for r in self.results
            if r.task == task and r.condition.startswith(condition_prefix)
        ]
        if not candidates:
            return None
        if task == "narma10":
            return min(candidates, key=lambda r: r.metric_value)  # lower NRMSE better
        return max(candidates, key=lambda r: r.metric_value)  # higher better


def _run_task_on_engine(
    engine,
    inputs: List[float],
    targets: List[float],
    task_name: str,
    lambda_reg: float = 1e-4,
    seed: int = 42,
) -> float:
    """Run a task through an engine, train readout, evaluate."""
    engine.reset()
    d = engine.feature_dim

    # Collect states
    states: List[List[float]] = []
    for inp in inputs:
        s = engine.step(inp)
        states.append(s)

    N = len(states)
    train_idx, test_idx = deterministic_split(N, seed)

    if not test_idx or not train_idx:
        return float('inf') if task_name == "narma10" else 0.0

    # Build matrices for ridge regression
    X_train = [[states[i][j] for i in train_idx] for j in range(d)]
    Y_train = [[targets[i] for i in train_idx]]

    # Train readout
    try:
        W = solve_ridge(X_train, Y_train, lambda_reg)
    except ValueError:
        return float('inf') if task_name == "narma10" else 0.0

    # Predict on test set
    predicted = []
    actual = []
    for i in test_idx:
        pred = sum(W[0][j] * states[i][j] for j in range(d))
        predicted.append(pred)
        actual.append(targets[i])

    if task_name == "delayed_xor":
        # Binary classification: threshold at 0.5
        correct = sum(
            1 for p, a in zip(predicted, actual)
            if (p > 0.5) == (a > 0.5)
        )
        return correct / len(predicted) if predicted else 0.0
    elif task_name == "narma10":
        return nrmse(predicted, actual)
    else:
        return 0.0


def _run_memory_capacity(
    engine,
    seq_length: int = 300,
    max_delay: int = 15,
    seed: int = 42,
    lambda_reg: float = 1e-4,
) -> float:
    """Run memory capacity test on an engine."""
    inputs, delay_targets = generate_memory_capacity(seq_length, max_delay, seed)
    engine.reset()
    d = engine.feature_dim

    # Collect states
    states: List[List[float]] = []
    for inp in inputs:
        s = engine.step(inp)
        states.append(s)

    N = len(states)
    train_idx, test_idx = deterministic_split(N, seed)

    if not test_idx or not train_idx:
        return 0.0

    mc_total = 0.0
    for k in range(max_delay):
        target_k = delay_targets[k]

        X_train = [[states[i][j] for i in train_idx] for j in range(d)]
        Y_train = [[target_k[i] for i in train_idx]]

        try:
            W = solve_ridge(X_train, Y_train, lambda_reg)
        except ValueError:
            continue

        predicted = [sum(W[0][j] * states[i][j] for j in range(d)) for i in test_idx]
        actual = [target_k[i] for i in test_idx]
        c = correlation(predicted, actual)
        mc_total += c * c

    return mc_total


def _run_separation(
    engine,
    seed: int = 42,
    n_steps: int = 100,
) -> float:
    """Run separation ratio test."""
    rng = LCG(seed)
    inputs_a = [rng.next_float() for _ in range(n_steps)]
    inputs_b = [rng.next_float() for _ in range(n_steps)]

    engine.reset()
    states_a = [engine.step(inp) for inp in inputs_a]

    engine.reset()
    states_b = [engine.step(inp) for inp in inputs_b]

    return separation_ratio(states_a, states_b, inputs_a, inputs_b)


def run_ablation(
    seed: int = 42,
    seq_length: int = 300,
    lambda_reg: float = 1e-4,
) -> AblationReport:
    """Run full ablation study: 3 conditions × 2 tasks + MC + separation.

    Returns AblationReport with all results.
    """
    report = AblationReport(seed=seed)

    # Generate tasks
    xor_inputs, xor_targets = generate_delayed_xor(seq_length, delay=3, seed=seed)
    narma_inputs, narma_targets = generate_narma10(seq_length, seed=seed)

    # --- Condition 1: Linear ODE (grid search) ---
    for config in grid_configs():
        engine = _LinearEngine(config["decay"], config["gain"])

        # Delayed XOR
        acc = _run_task_on_engine(engine, xor_inputs, xor_targets, "delayed_xor", lambda_reg, seed)
        report.results.append(AblationResult(
            condition=engine.name, task="delayed_xor",
            metric_name="accuracy", metric_value=acc,
            feature_dim=engine.feature_dim, config=config,
        ))

        # NARMA-10
        nrmse_val = _run_task_on_engine(engine, narma_inputs, narma_targets, "narma10", lambda_reg, seed)
        report.results.append(AblationResult(
            condition=engine.name, task="narma10",
            metric_name="nrmse", metric_value=nrmse_val,
            feature_dim=engine.feature_dim, config=config,
        ))

        # Memory capacity
        mc = _run_memory_capacity(engine, seq_length, max_delay=15, seed=seed, lambda_reg=lambda_reg)
        report.results.append(AblationResult(
            condition=engine.name, task="memory_capacity",
            metric_name="mc", metric_value=mc,
            feature_dim=engine.feature_dim, config=config,
        ))

        # Separation ratio
        sep = _run_separation(engine, seed=seed)
        report.results.append(AblationResult(
            condition=engine.name, task="separation",
            metric_name="separation_ratio", metric_value=sep,
            feature_dim=engine.feature_dim, config=config,
        ))

    # --- Condition 2: Izhikevich only ---
    izh_engine = _IzhOnlyEngine(seed=0xDEADBEEF)

    acc = _run_task_on_engine(izh_engine, xor_inputs, xor_targets, "delayed_xor", lambda_reg, seed)
    report.results.append(AblationResult(
        condition="izh_only", task="delayed_xor",
        metric_name="accuracy", metric_value=acc,
        feature_dim=izh_engine.feature_dim,
    ))

    nrmse_val = _run_task_on_engine(izh_engine, narma_inputs, narma_targets, "narma10", lambda_reg, seed)
    report.results.append(AblationResult(
        condition="izh_only", task="narma10",
        metric_name="nrmse", metric_value=nrmse_val,
        feature_dim=izh_engine.feature_dim,
    ))

    mc = _run_memory_capacity(izh_engine, seq_length, max_delay=15, seed=seed, lambda_reg=lambda_reg)
    report.results.append(AblationResult(
        condition="izh_only", task="memory_capacity",
        metric_name="mc", metric_value=mc,
        feature_dim=izh_engine.feature_dim,
    ))

    sep = _run_separation(izh_engine, seed=seed)
    report.results.append(AblationResult(
        condition="izh_only", task="separation",
        metric_name="separation_ratio", metric_value=sep,
        feature_dim=izh_engine.feature_dim,
    ))

    # --- Condition 3: Izhikevich + Reservoir ---
    res_engine = _IzhReservoirEngine(seed=0xDEADBEEF)

    acc = _run_task_on_engine(res_engine, xor_inputs, xor_targets, "delayed_xor", lambda_reg, seed)
    report.results.append(AblationResult(
        condition="izh_reservoir", task="delayed_xor",
        metric_name="accuracy", metric_value=acc,
        feature_dim=res_engine.feature_dim,
    ))

    nrmse_val = _run_task_on_engine(res_engine, narma_inputs, narma_targets, "narma10", lambda_reg, seed)
    report.results.append(AblationResult(
        condition="izh_reservoir", task="narma10",
        metric_name="nrmse", metric_value=nrmse_val,
        feature_dim=res_engine.feature_dim,
    ))

    mc = _run_memory_capacity(res_engine, seq_length, max_delay=15, seed=seed, lambda_reg=lambda_reg)
    report.results.append(AblationResult(
        condition="izh_reservoir", task="memory_capacity",
        metric_name="mc", metric_value=mc,
        feature_dim=res_engine.feature_dim,
    ))

    sep = _run_separation(res_engine, seed=seed)
    report.results.append(AblationResult(
        condition="izh_reservoir", task="separation",
        metric_name="separation_ratio", metric_value=sep,
        feature_dim=res_engine.feature_dim,
    ))

    return report
