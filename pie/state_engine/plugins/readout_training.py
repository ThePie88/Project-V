"""Readout training for the ESN reservoir via ridge regression (V6a-R2).

This module implements **distillation**: the oracle captures the current
hand-designed ``_CONTROL_READOUT`` semantic mapping as training targets,
and the reservoir learns to reproduce them through its own nonlinear
dynamics.

Key properties:
- **Pure Python**: no numpy, no scipy — deterministic across platforms.
- **Ridge regression via Cholesky**: numerically stable, positive-definite.
- **Rounding at boundaries only**: no intermediate rounding inside solver.
- **Dataset determinism**: ``dataset_id = SHA256(seed || preset || ...)``.
- **Split determinism**: 80/20 via LCG index, no random.shuffle.

Usage::

    trainer = ReadoutTrainer(seed=42)
    trainer.collect(reservoir_states, oracle_targets)  # N times
    result = trainer.train(lambda_reg=1e-4)
    result.save("artifacts/readout_weights.json")

Reference: Lukoševičius & Jaeger (2009) "Reservoir computing approaches
to recurrent neural network training"
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pie.benchmarks.linalg import (
    solve_ridge,
    mat_transpose,
    mean,
    std,
    mae,
    r_squared,
    accuracy,
)
from pie.state_engine.plugins.reservoir import (
    LCG,
    Reservoir,
    _CONTROL_READOUT,
    _T1_INDEX,
)


# ---------------------------------------------------------------------------
# Oracle — distillation target from hand-designed map
# ---------------------------------------------------------------------------

# CV channel names in canonical order (matches ControlVector fields)
CV_CHANNELS = ["memory_gate", "tool_gate", "cf_k", "verbosity_bias", "consolidation_urgency"]

# Oracle version — bump when _CONTROL_READOUT changes
ORACLE_VERSION = "v1.0"


def oracle_cv(
    tier1_output: List[float],
) -> Dict[str, float]:
    """Distillation oracle: computes target CV from Tier-1 neuron outputs.

    This IS the current hand-designed ``_CONTROL_READOUT`` mapping,
    packaged as a pure function for training data collection.

    Not "ground truth" — it's the teacher signal for the reservoir
    readout to learn.  The reservoir then reproduces this mapping
    through its own nonlinear temporal dynamics.

    Args:
        tier1_output: 10 normalized values [0,1] from Tier-1 neurons,
            in NEURON_ORDER (curiosity, sociality, caution, agency,
            playfulness, fatigue, valence, arousal, attention, tension).

    Returns:
        Dict with 5 CV channel target values (raw, pre-sigmoid).
    """
    result: Dict[str, float] = {}
    for ch_name, semantic_map in _CONTROL_READOUT.items():
        val = 0.0
        for t1_name, coeff in semantic_map.items():
            t1_idx = _T1_INDEX[t1_name]
            val += coeff * tier1_output[t1_idx]
        result[ch_name] = val
    return result


# ---------------------------------------------------------------------------
# Dataset ID
# ---------------------------------------------------------------------------

def compute_dataset_id(
    seed: int,
    n_episodes: int,
    oracle_version: str = ORACLE_VERSION,
    engine_version: str = "0.3",
    preset: str = "default",
) -> str:
    """Deterministic dataset identifier via SHA-256.

    Captures all parameters that affect training data generation.
    """
    payload = f"{seed}|{preset}|{n_episodes}|{oracle_version}|{engine_version}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Deterministic train/test split
# ---------------------------------------------------------------------------

def deterministic_split(
    n: int,
    seed: int,
) -> Tuple[List[int], List[int]]:
    """Split indices into 80% train / 20% test using LCG.

    No random.shuffle, no sklearn.  Fully deterministic.

    Returns:
        (train_indices, test_indices)
    """
    rng = LCG(seed ^ 0xABCD1234)
    train_idx: List[int] = []
    test_idx: List[int] = []
    for i in range(n):
        if rng.next_int() % 5 != 0:
            train_idx.append(i)
        else:
            test_idx.append(i)
    return train_idx, test_idx


# ---------------------------------------------------------------------------
# Training result
# ---------------------------------------------------------------------------

@dataclass
class ReadoutWeights:
    """Trained readout weights + normalization statistics."""

    weights: List[List[float]]           # (5 × feature_dim)
    channels: List[str]                  # channel names in order
    feature_dim: int                     # 128 for reservoir, 10 for Izh-only
    mean_X: List[float]                  # per-feature mean (for normalization)
    std_X: List[float]                   # per-feature std
    lambda_reg: float
    seed: int
    dataset_id: str
    oracle_version: str
    n_samples_train: int
    n_samples_test: int
    metrics: Dict[str, Dict[str, float]]  # per-channel {MAE, R2} or {accuracy}
    schema_version: str = "1.0"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "feature_dim": self.feature_dim,
            "channels": self.channels,
            "weights": self.weights,
            "mean_X": self.mean_X,
            "std_X": self.std_X,
            "lambda_reg": self.lambda_reg,
            "seed": self.seed,
            "dataset_id": self.dataset_id,
            "oracle_version": self.oracle_version,
            "n_samples_train": self.n_samples_train,
            "n_samples_test": self.n_samples_test,
            "metrics": self.metrics,
        }

    def save(self, path: Path) -> None:
        """Save to JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> "ReadoutWeights":
        """Load from JSON."""
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            weights=data["weights"],
            channels=data["channels"],
            feature_dim=data["feature_dim"],
            mean_X=data["mean_X"],
            std_X=data["std_X"],
            lambda_reg=data["lambda_reg"],
            seed=data["seed"],
            dataset_id=data["dataset_id"],
            oracle_version=data["oracle_version"],
            n_samples_train=data["n_samples_train"],
            n_samples_test=data["n_samples_test"],
            metrics=data["metrics"],
            schema_version=data.get("schema_version", "1.0"),
        )


# ---------------------------------------------------------------------------
# ReadoutTrainer
# ---------------------------------------------------------------------------

class ReadoutTrainer:
    """Trains a ridge-regression readout from reservoir states to CV targets.

    Workflow:
    1. ``collect(features, targets)`` — accumulate (state, target) pairs
    2. ``train(lambda_reg)`` — fit ridge regression via Cholesky
    3. Result contains weights, normalization stats, held-out metrics

    The features can be 128-dim (reservoir leak_state) or 10-dim (Tier-1
    for ablation).  Each condition trains its own readout.
    """

    def __init__(
        self,
        seed: int = 42,
        oracle_version: str = ORACLE_VERSION,
        engine_version: str = "0.3",
    ) -> None:
        self._seed = seed
        self._oracle_version = oracle_version
        self._engine_version = engine_version
        self._features: List[List[float]] = []   # N × d
        self._targets: List[Dict[str, float]] = []  # N × {channel: value}

    @property
    def n_samples(self) -> int:
        return len(self._features)

    def collect(
        self,
        features: List[float],
        targets: Dict[str, float],
    ) -> None:
        """Add one (feature_vector, target_cv) pair to the dataset.

        Args:
            features: Feature vector (e.g., 128-dim reservoir state).
            targets: Target CV values from oracle (5 channels).
        """
        self._features.append(list(features))
        self._targets.append(dict(targets))

    def collect_from_reservoir(
        self,
        reservoir: Reservoir,
        tier1_output: List[float],
    ) -> Dict[str, float]:
        """Convenience: step reservoir, collect features + oracle targets.

        Returns the oracle targets (for verification).
        """
        # Step reservoir to get features
        reservoir.step(tier1_output)
        features = reservoir.get_neuron_states()  # 128-dim

        # Compute oracle targets from tier1
        targets = oracle_cv(tier1_output)

        self.collect(features, targets)
        return targets

    def train(
        self,
        lambda_reg: float = 1e-4,
    ) -> ReadoutWeights:
        """Train ridge regression on collected data.

        Returns ReadoutWeights with trained weights and held-out metrics.
        """
        if self.n_samples < 5:
            raise ValueError(f"Need at least 5 samples, got {self.n_samples}")

        N = self.n_samples
        d = len(self._features[0])

        # --- Deterministic split ---
        train_idx, test_idx = deterministic_split(N, self._seed)
        if not test_idx:
            # If all in train, move last 20% to test
            split = int(N * 0.8)
            train_idx = list(range(split))
            test_idx = list(range(split, N))

        # --- Build matrices ---
        X_train_raw = [self._features[i] for i in train_idx]  # list of d-vectors
        X_test_raw = [self._features[i] for i in test_idx]

        # --- Compute normalization stats from TRAINING set only ---
        mean_X = [0.0] * d
        for sample in X_train_raw:
            for j in range(d):
                mean_X[j] += sample[j]
        for j in range(d):
            mean_X[j] /= len(X_train_raw)

        std_X = [0.0] * d
        for sample in X_train_raw:
            for j in range(d):
                std_X[j] += (sample[j] - mean_X[j]) ** 2
        for j in range(d):
            std_X[j] = math.sqrt(std_X[j] / len(X_train_raw)) if len(X_train_raw) > 1 else 1.0
            if std_X[j] < 1e-8:
                std_X[j] = 1.0  # avoid division by zero

        # --- Normalize ---
        def normalize(samples: List[List[float]]) -> List[List[float]]:
            return [
                [(s[j] - mean_X[j]) / std_X[j] for j in range(d)]
                for s in samples
            ]

        X_train_norm = normalize(X_train_raw)
        X_test_norm = normalize(X_test_raw)

        # --- Build target matrices per channel ---
        channels = CV_CHANNELS
        k = len(channels)

        # X matrix: (d × N_train) — each column is a sample
        X_mat = [[X_train_norm[s][j] for s in range(len(X_train_norm))]
                 for j in range(d)]

        # Y matrix: (k × N_train) — each column is a target vector
        Y_mat = [[self._targets[train_idx[s]].get(ch, 0.0) for s in range(len(train_idx))]
                 for ch in channels]

        # --- Solve ridge regression ---
        W = solve_ridge(X_mat, Y_mat, lambda_reg)  # (k × d)

        # --- Evaluate on held-out set ---
        metrics: Dict[str, Dict[str, float]] = {}
        for ch_idx, ch_name in enumerate(channels):
            w_row = W[ch_idx]  # d-dim weight vector

            # Predict on test set
            predicted = []
            actual = []
            for s_idx, s in enumerate(X_test_norm):
                pred = sum(w_row[j] * s[j] for j in range(d))
                predicted.append(pred)
                actual.append(self._targets[test_idx[s_idx]].get(ch_name, 0.0))

            if ch_name == "cf_k":
                # Discrete channel: accuracy after rounding
                pred_int = [max(1, min(5, int(round(p * 5.0)))) for p in predicted]
                actual_int = [max(1, min(5, int(round(a * 5.0)))) for a in actual]
                metrics[ch_name] = {
                    "accuracy": round(accuracy(pred_int, actual_int), 4),
                    "mae": round(mae(predicted, actual), 4),
                }
            else:
                # Continuous channel: MAE + R²
                metrics[ch_name] = {
                    "mae": round(mae(predicted, actual), 4),
                    "r_squared": round(r_squared(predicted, actual), 4),
                }

        # --- Build result ---
        dataset_id = compute_dataset_id(
            seed=self._seed,
            n_episodes=N,
            oracle_version=self._oracle_version,
            engine_version=self._engine_version,
        )

        # Round weights at output boundary
        W_rounded = [[round(w, 10) for w in row] for row in W]

        return ReadoutWeights(
            weights=W_rounded,
            channels=channels,
            feature_dim=d,
            mean_X=[round(m, 10) for m in mean_X],
            std_X=[round(s, 10) for s in std_X],
            lambda_reg=lambda_reg,
            seed=self._seed,
            dataset_id=dataset_id,
            oracle_version=self._oracle_version,
            n_samples_train=len(train_idx),
            n_samples_test=len(test_idx),
            metrics=metrics,
        )
