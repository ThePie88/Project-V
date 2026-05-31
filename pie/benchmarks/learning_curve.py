"""Learning curve runner for R-STDP evaluation (V6a-R4).

Runs three conditions over N episodes with deterministic synthetic
environment, measuring how synapse weights and behavior metrics
evolve over time.

Conditions:
1. No plasticity — fixed weights
2. STDP only — Hebbian, no reward signal
3. R-STDP — reward-modulated STDP

The "environment" generates deterministic episodes: sequences of
neuron inputs, spike patterns, and reward signals.  This allows
fully reproducible comparison without requiring the LLM or runtime.

Metrics tracked per episode:
- tool_deny_rate: fraction of tool attempts that were denied
- recall_precision: fraction of recall attempts that hit
- mean_cost: average cost proxy per turn

Pass thresholds (at least 1 of):
- tool_deny_rate ↓ 10% relative vs no-plasticity within 20 episodes
- recall_precision ↑ 5% absolute within 20 episodes
- mean_cost ↓ 8% relative within 20 episodes
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pie.state_engine.plugins.reservoir import LCG
from pie.state_engine.plugins.stdp import STDPTracker, WEIGHT_MIN, WEIGHT_MAX
from pie.state_engine.plugins.rstdp import RewardSTDPTracker, REWARD_MAP
from pie.state_engine.plugins.neural_snn import (
    IzhikevichNeuron, NEURON_TYPES, NEURON_ORDER,
    SYNAPSE_WEIGHTS, BASE_CURRENTS,
)
from pie.determinism import clamp_round


TURNS_PER_EPISODE = 20
N_EPISODES_DEFAULT = 20


@dataclass
class EpisodeMetrics:
    """Metrics for a single episode."""
    episode: int
    tool_deny_rate: float
    recall_precision: float
    mean_cost: float
    total_reward: float
    weight_norm: float  # L1 norm of weight changes from initial


@dataclass
class LearningCurveReport:
    """Full learning curve report across all conditions."""
    seed: int
    n_episodes: int
    turns_per_episode: int
    conditions: Dict[str, List[EpisodeMetrics]] = field(default_factory=dict)
    schema_version: str = "1.0"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "seed": self.seed,
            "n_episodes": self.n_episodes,
            "turns_per_episode": self.turns_per_episode,
            "conditions": {
                name: [
                    {
                        "episode": m.episode,
                        "tool_deny_rate": round(m.tool_deny_rate, 6),
                        "recall_precision": round(m.recall_precision, 6),
                        "mean_cost": round(m.mean_cost, 6),
                        "total_reward": round(m.total_reward, 4),
                        "weight_norm": round(m.weight_norm, 6),
                    }
                    for m in metrics
                ]
                for name, metrics in self.conditions.items()
            },
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def improvement(self, condition: str, metric: str) -> Optional[float]:
        """Compute improvement of condition vs no_plasticity on a metric.

        Returns relative improvement (negative = better for deny_rate/cost,
        positive = better for recall_precision).
        """
        if condition not in self.conditions or "no_plasticity" not in self.conditions:
            return None
        baseline = self.conditions["no_plasticity"]
        target = self.conditions[condition]
        if not baseline or not target:
            return None

        # Use last 5 episodes average vs first 5
        def avg_metric(episodes: List[EpisodeMetrics], start: int, end: int) -> float:
            vals = [getattr(e, metric) for e in episodes[start:end]]
            return sum(vals) / len(vals) if vals else 0.0

        base_late = avg_metric(baseline, -5, len(baseline))
        tgt_late = avg_metric(target, -5, len(target))

        if metric == "recall_precision":
            return tgt_late - base_late  # absolute improvement
        else:
            # Relative improvement (lower = better for deny_rate, cost)
            if abs(base_late) < 1e-12:
                return 0.0
            return (base_late - tgt_late) / abs(base_late)


class _SyntheticEnvironment:
    """Deterministic synthetic environment for learning curve evaluation.

    Generates neuron inputs, determines spikes, and produces reward
    signals based on the current neural state and weights.
    """

    def __init__(self, seed: int):
        self._rng = LCG(seed)
        self._turn = 0

    def generate_turn(
        self,
        neurons: Dict[str, IzhikevichNeuron],
        weights: Dict[Tuple[str, str], float],
    ) -> Tuple[List[str], str, float]:
        """Generate one turn of the synthetic environment.

        Returns:
            (spiked_neurons, reward_source, reward_metric_contribution)
        """
        self._turn += 1

        # Generate external input currents (deterministic)
        currents: Dict[str, float] = {}
        for name in NEURON_ORDER:
            base = BASE_CURRENTS.get(name, 0.0)
            external = self._rng.next_float() * 0.08
            # Synaptic input from other neurons
            synaptic = 0.0
            for (src, tgt), w in sorted(weights.items()):
                if tgt == name and src in neurons:
                    synaptic += w * neurons[src].membrane_potential
                    if neurons[src].last_spike:
                        synaptic += w * 0.5
            currents[name] = base + external + synaptic

        # Step neurons
        spiked = []
        for name in NEURON_ORDER:
            neurons[name].step(currents[name])
            if neurons[name].last_spike:
                spiked.append(name)

        # Determine reward event based on neural state
        # This is a simplified model:
        # - Tool attempts happen when agency spikes
        # - Tool denial when caution is high (tension high)
        # - Recall hits when curiosity + attention both spike
        # - Cost is proportional to number of spikes
        r = self._rng.next_float()
        if "agency" in spiked and "caution" not in spiked:
            return spiked, "tool_success", 0.0
        elif "agency" in spiked and "caution" in spiked:
            return spiked, "tool_denied", 1.0  # deny cost
        elif "curiosity" in spiked and "attention" in spiked:
            return spiked, "recall_hit", 0.0
        elif "tension" in spiked and "caution" in spiked:
            return spiked, "constraint_violated", 2.0  # high cost
        else:
            return spiked, "", 0.5  # neutral cost

    def reset_turn_counter(self):
        self._turn = 0


def _create_neurons() -> Dict[str, IzhikevichNeuron]:
    """Create fresh set of Tier-1 neurons."""
    neurons = {}
    for name in NEURON_ORDER:
        params = NEURON_TYPES.get(name, {})
        neurons[name] = IzhikevichNeuron(
            neuron_id=name,
            a=params.get("a", 0.02),
            b=params.get("b", 0.2),
            c=params.get("c", -65.0),
            d=params.get("d", 8.0),
        )
    return neurons


def _weight_norm(
    current: Dict[Tuple[str, str], float],
    initial: Dict[Tuple[str, str], float],
) -> float:
    """L1 norm of weight changes from initial."""
    return sum(
        abs(current.get(k, 0.0) - initial.get(k, 0.0))
        for k in initial
    )


def _run_condition(
    condition: str,
    seed: int,
    n_episodes: int,
    turns_per_episode: int,
    eta: float = 0.05,
) -> List[EpisodeMetrics]:
    """Run one condition for N episodes."""
    initial_weights = dict(SYNAPSE_WEIGHTS)
    results: List[EpisodeMetrics] = []

    # Create tracker based on condition
    if condition == "no_plasticity":
        tracker = None
    elif condition == "stdp_only":
        tracker = STDPTracker(initial_weights=dict(initial_weights))
    elif condition == "rstdp":
        tracker = RewardSTDPTracker(
            initial_weights=dict(initial_weights),
            eta=eta,
        )
    else:
        raise ValueError(f"Unknown condition: {condition}")

    for ep in range(n_episodes):
        # Fresh neurons each episode, same seed pattern
        neurons = _create_neurons()
        env = _SyntheticEnvironment(seed=seed ^ (ep * 0x1337))

        tool_attempts = 0
        tool_denials = 0
        recall_attempts = 0
        recall_hits = 0
        total_cost = 0.0
        total_reward = 0.0

        current_weights = (
            tracker.weights if tracker else dict(initial_weights)
        )

        for turn in range(turns_per_episode):
            spiked, reward_source, cost = env.generate_turn(
                neurons, current_weights,
            )

            # Track metrics
            if reward_source == "tool_success":
                tool_attempts += 1
            elif reward_source == "tool_denied":
                tool_attempts += 1
                tool_denials += 1
            elif reward_source == "recall_hit":
                recall_attempts += 1
                recall_hits += 1
            elif reward_source == "constraint_violated":
                pass  # tracked via cost

            total_cost += cost

            # Apply plasticity
            if tracker is not None:
                if isinstance(tracker, RewardSTDPTracker):
                    tracker.record_spikes(
                        turn=ep * turns_per_episode + turn,
                        spiked_neurons=spiked,
                    )
                    if reward_source in REWARD_MAP:
                        tracker.apply_reward(
                            turn=ep * turns_per_episode + turn,
                            source=reward_source,
                        )
                        total_reward += REWARD_MAP[reward_source]
                elif isinstance(tracker, STDPTracker):
                    tracker.record_spikes(
                        turn=ep * turns_per_episode + turn,
                        spiked_neurons=spiked,
                    )

                current_weights = tracker.weights

        # Compute episode metrics
        deny_rate = tool_denials / max(tool_attempts, 1)
        precision = recall_hits / max(recall_attempts, 1)
        mean_cost = total_cost / max(turns_per_episode, 1)

        results.append(EpisodeMetrics(
            episode=ep,
            tool_deny_rate=deny_rate,
            recall_precision=precision,
            mean_cost=mean_cost,
            total_reward=total_reward,
            weight_norm=_weight_norm(current_weights, initial_weights),
        ))

    return results


def run_learning_curve(
    seed: int = 42,
    n_episodes: int = N_EPISODES_DEFAULT,
    turns_per_episode: int = TURNS_PER_EPISODE,
    eta: float = 0.05,
) -> LearningCurveReport:
    """Run full learning curve: 3 conditions × N episodes.

    Args:
        eta: R-STDP learning rate for benchmark. Higher than kernel
             default (0.005) to produce observable effects in short runs.

    Returns LearningCurveReport with per-episode metrics for each condition.
    """
    report = LearningCurveReport(
        seed=seed,
        n_episodes=n_episodes,
        turns_per_episode=turns_per_episode,
    )

    for condition in ["no_plasticity", "stdp_only", "rstdp"]:
        report.conditions[condition] = _run_condition(
            condition=condition,
            seed=seed,
            n_episodes=n_episodes,
            turns_per_episode=turns_per_episode,
            eta=eta,
        )

    return report
