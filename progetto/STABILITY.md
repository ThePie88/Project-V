# Stability Analysis — Pie Kernel v0.0.0

## 1. Boundedness by Construction (Theorem)

**Claim**: All state variables are bounded in [0, 1]^10 by construction.

**Proof**:

The Pie kernel state space consists of 10 dimensions: 6 drives
(curiosity, sociality, caution, agency, playfulness, fatigue) and
4 affect channels (valence, arousal, attention, tension).

Each dimension is bounded by three independent mechanisms:

### 1a. Pydantic contract validators

The `State` contract (`pie/contracts/state.py`) enforces:

```python
@validator("drives", each_item=True)
def clamp_drives(cls, v):
    return max(0.0, min(1.0, v))

@validator("affect", each_item=True)
def clamp_affect(cls, v):
    return max(0.0, min(1.0, v))
```

**Every** write to `state.drives` or `state.affect` passes through these
validators.  No code path can set a value outside [0, 1].

### 1b. Izhikevich neuron clamping

The Tier-1 neurons (`pie/state_engine/plugins/neural_snn.py`) clamp:

- Membrane potential: `v ∈ [-80, 30]` (hard clamp after each step)
- Recovery variable: `u ∈ [-200, 200]` (hard clamp after each step)
- Normalized output: `(v + 80) / 110` mapped to [0, 1] via `clamp_round`

### 1c. Reservoir leak state clamping

The ESN reservoir (`pie/state_engine/plugins/reservoir.py`):

- Leak state: `leak_state[i] ∈ [0, 1]` via `clamp_round(val, 0.0, 1.0, 4)`
- Updated each step: `(1-α)*prev + α*new` where both `prev ∈ [0,1]`
  and `new ∈ [0,1]` → convex combination stays in [0, 1]

### Conclusion

State ∈ [0, 1]^10 is guaranteed by construction at three independent
layers (contract, neuron, reservoir).  No external input, no sequence
of events, and no number of steps can cause a state variable to leave
this hypercube.

---

## 2. Empirical Trajectory Verification (Test)

We verify empirically that 10,000 random initial states, each evolved
for 100 steps through the NeuralSNN plugin, remain within [0, 1]^10
at every step.

**This is NOT a formal proof.**  It is a large-scale verification that
the boundedness-by-construction claim holds in practice, including under
numerical edge cases (rounding, floating-point accumulation).

Test: `P-STB-020` in `tests/test_stability.py`.

Method:
- 10,000 initial states sampled uniformly from [0, 1]^10 via deterministic LCG
- Each state evolved for 100 steps using `NeuralSNNPlugin(reservoir_enabled=False, stdp_enabled=False)`
- At each step, all 10 dimensions checked: `0.0 <= x <= 1.0`
- Deterministic: same seed produces same trajectories

Result: 100% of trajectories stay bounded.  Zero divergence events
across 10,000 × 100 = 1,000,000 state transitions.

---

## 3. What We Do NOT Claim

### No formal Lyapunov function

The `LyapunovChecker` class (`pie/lyapunov.py`) computes a quadratic
function V(x) = Σ w_i (x_i - c_i)² centered on the empirical attractor.
This function is:

- **Positive semi-definite** (sum of squares)
- **Bounded above** (state in [0,1]^10 → V ≤ V_max)
- **NOT monotonically decreasing** along trajectories

The Izhikevich neurons produce limit-cycle dynamics with periodic
spikes.  Each spike causes a transient increase in V.  We verify
"ultimate boundedness" (trajectories stay within V_max), but this is
NOT a Lyapunov stability proof in the formal dynamical systems sense.

### No certificate of asymptotic stability

We do not claim that the system converges to a fixed point.  The
neural dynamics produce ongoing oscillations (spiking behavior).
The system is bounded but not asymptotically stable.

### Bounded ≠ stable

In dynamical systems theory, "stable" has a precise meaning
(Lyapunov stability: nearby trajectories stay nearby).  We do not
prove this.  We prove only that:

1. All states remain in [0, 1]^10 (boundedness)
2. The quadratic function V is bounded above by V_max (ultimate boundedness)
3. Trajectories empirically settle into a bounded attractor region

These are useful engineering guarantees but are weaker than formal
stability in the dynamical systems sense.

---

## Summary

| Property | Status | Evidence |
|----------|--------|----------|
| Boundedness [0,1]^10 | **Theorem** | Contract validators + neuron clamps + reservoir clamps |
| No divergence | **Empirical** | 10,000 trajectories × 100 steps, 0 violations |
| V(x) ≤ V_max | **Empirical** | Quadratic V bounded by construction of [0,1]^10 |
| Asymptotic stability | **NOT CLAIMED** | Spiking neurons produce limit cycles |
| Formal Lyapunov proof | **NOT CLAIMED** | V is not monotonically decreasing |
