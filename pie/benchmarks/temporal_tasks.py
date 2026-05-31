"""Temporal processing benchmark tasks for ablation study (V6a-R1).

Two standard tasks from reservoir computing literature:
1. **Delayed XOR**: tests nonlinear short-term memory
2. **NARMA-10**: tests 10th-order nonlinear autoregressive memory

All sequences are deterministic (LCG-seeded), cross-platform reproducible.
No numpy, no random module.

Reference: Jaeger (2001), Atiya & Parlos (2000)
"""

from __future__ import annotations

from typing import List, Tuple

from pie.state_engine.plugins.reservoir import LCG


# ---------------------------------------------------------------------------
# Delayed XOR
# ---------------------------------------------------------------------------

def generate_delayed_xor(
    seq_length: int = 500,
    delay: int = 3,
    seed: int = 42,
) -> Tuple[List[float], List[float]]:
    """Generate delayed XOR task.

    Input: random binary sequence u(t) ∈ {0, 1}
    Target: y(t) = XOR(u(t), u(t - delay))

    For t < delay, target = 0.

    Args:
        seq_length: Length of sequence.
        delay: XOR delay in timesteps.
        seed: LCG seed for input generation.

    Returns:
        (inputs, targets) — both lists of floats (0.0 or 1.0).
    """
    rng = LCG(seed)
    inputs: List[float] = []
    targets: List[float] = []

    for t in range(seq_length):
        u = 1.0 if rng.next_float() > 0.5 else 0.0
        inputs.append(u)

        if t >= delay:
            xor_val = 1.0 if inputs[t] != inputs[t - delay] else 0.0
        else:
            xor_val = 0.0
        targets.append(xor_val)

    return inputs, targets


# ---------------------------------------------------------------------------
# NARMA-10
# ---------------------------------------------------------------------------

def generate_narma10(
    seq_length: int = 500,
    seed: int = 42,
    burn_in: int = 100,
) -> Tuple[List[float], List[float]]:
    """Generate NARMA-10 task.

    System:
        y(t+1) = 0.3*y(t) + 0.05*y(t)*sum(y(t-i), i=0..9)
                 + 1.5*u(t-9)*u(t) + 0.1

    Input u(t) ~ Uniform[0, 0.5] via LCG.
    float64, clip [-1e6, 1e6] safety, NO per-step rounding.

    Args:
        seq_length: Number of valid output samples (after burn-in).
        seed: LCG seed for input generation.
        burn_in: Number of initial steps to discard (system warmup).

    Returns:
        (inputs, targets) — after burn-in, both length = seq_length.
    """
    total = seq_length + burn_in + 10  # +10 for NARMA history buffer
    rng = LCG(seed)

    # Generate input sequence
    u_all: List[float] = []
    for _ in range(total):
        u_all.append(rng.next_float() * 0.5)  # [0, 0.5)

    # Compute NARMA-10
    y_all: List[float] = [0.0] * total

    for t in range(10, total - 1):
        y_sum = 0.0
        for i in range(10):
            y_sum += y_all[t - i]

        y_next = (
            0.3 * y_all[t]
            + 0.05 * y_all[t] * y_sum
            + 1.5 * u_all[t - 9] * u_all[t]
            + 0.1
        )

        # Safety clip (no rounding)
        y_next = max(-1e6, min(1e6, y_next))
        y_all[t + 1] = y_next

    # Discard burn-in
    start = burn_in + 10
    inputs = u_all[start : start + seq_length]
    targets = y_all[start : start + seq_length]

    return inputs, targets


# ---------------------------------------------------------------------------
# Memory Capacity
# ---------------------------------------------------------------------------

def generate_memory_capacity(
    seq_length: int = 500,
    max_delay: int = 20,
    seed: int = 42,
) -> Tuple[List[float], List[List[float]]]:
    """Generate memory capacity test data.

    Input: random sequence u(t) ~ Uniform[-1, 1] via LCG.
    Targets: [u(t-1), u(t-2), ..., u(t-max_delay)] for each t.

    MC = sum_k corr(y_k, u(t-k))^2

    Args:
        seq_length: Length of sequence.
        max_delay: Maximum delay to test.
        seed: LCG seed.

    Returns:
        (inputs, delayed_targets) where delayed_targets[k] is
        the sequence u(t-k-1).
    """
    rng = LCG(seed)
    inputs: List[float] = []
    for _ in range(seq_length + max_delay):
        inputs.append(rng.next_signed())

    # Trim to seq_length (skip first max_delay for history)
    valid_inputs = inputs[max_delay:]

    # Build delay targets
    delayed_targets: List[List[float]] = []
    for k in range(1, max_delay + 1):
        target_k = inputs[max_delay - k : max_delay - k + seq_length]
        delayed_targets.append(target_k)

    return valid_inputs, delayed_targets
