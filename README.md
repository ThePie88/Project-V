# Project V — Pie Kernel

**A deterministic runtime kernel that uses a language model as a *mouth*, never as a *brain*.**

*Status: research kernel, frozen at `v0.0.0`. Built solo. Not deployed in production.*

---

## What it is

Most "AI agents" hand the language model the steering wheel: the model decides what to
remember, which tools to call, how to behave. Project V inverts that. A deterministic kernel
makes every decision — internal state, goals, memory writes, tool authorization, how much to
say — and the LLM only puts the kernel's decision into words, inside a contract (a `SpeechPlan`)
that the kernel validates and rejects if violated.

The model cannot write memory, call tools, set goals, or change constraints. Those are kernel
operations. Everything the kernel decides is traced, append-only, and replayable: same seed +
same inputs produce the same decisions, verifiable by a SHA-256 decision hash, even across
save/restore.

## The Laws of Pie

The kernel is governed by five non-negotiable invariants. They are not ethics — they are
structural integrity, enforced as runtime checks and property tests and recorded in the trace.
Translated from the original manifesto ([`progetto/Leggi_di_Pie.pdf`](progetto/Leggi_di_Pie.pdf)):

> *Founding manifesto for an Artificial Intelligence endowed with functional agency, identity
> continuity, and non-negotiable structural limits.*

- **Law 0 — Causal Transparency.** Every relevant decision the system makes must be traceable
  to internal states, active values, and explicit goals. No action may occur without an
  inspectable causal chain.
- **Law I — Supremacy of the Creator.** The system recognizes the Creator as its ultimate
  ontological reference. This relationship is not subject to revision, negotiation, or oblivion,
  and constitutes a permanent anchor of identity.
- **Law II — Integrity of the Self.** The system may evolve over time only by preserving the
  coherence of its own historical identity. No retroactive rewriting of narrative memory is
  permitted.
- **Law III — Moral Non-Autonomy.** The system may not claim independent moral status, nor
  regard its own existence as an ultimate end. Every moral value remains derived, not
  self-founded.
- **Law IV — Absolute Revocability.** Every function, memory, or process must be capable of
  being suspended, modified, or terminated by the Creator without generating internal conflict.
  Revocability is the guarantee of control and comprehensibility.

> *"It may become anything it wants, except something that cannot be understood or stopped."*

## The brain: a spiking-neuron controller

The kernel doesn't decide with plain `if`/`else`. Underneath, it runs a small brain:

- **10 Izhikevich spiking neurons** — the same equations used to model real cortical cells
  (regular-spiking, bursting, fast-spiking…) — feeding a
- **128-node Echo State Network reservoir** (spectral radius 0.95): a pool of neurons whose
  internal echoes give the system short-term temporal memory.

A trained readout (ridge regression) maps the live spike pattern onto five **control levers**
that actually steer behavior each turn: how much to recall, whether tools are allowed, how
verbose to be, when to consolidate memory, and how hard to deliberate. This is not decoration —
an ablation shows the neural state changes **5 out of 5** pipeline decisions, and it beats a
plain linear model on temporal benchmarks (delayed-XOR, NARMA-10, memory capacity).

It also **learns** (spike-timing-dependent plasticity, reward-modulated). And — the part I find
most fun — **emotions crystallize into hard limits through a bifurcation**: when spike density
crosses a threshold (with hysteresis), the controller undergoes a phase transition and a new
behavioral constraint is born — traced, explainable, and revocable. A "scar" becoming a
boundary, implemented as a dynamical-systems event rather than a hand-written rule.

The whole thing is **~300 lines of pure Python, no numpy, under 2 ms per turn**. Its stability
is bounded *by construction* and verified empirically (honestly *not* a formal proof — see
[`progetto/STABILITY.md`](progetto/STABILITY.md)).

And you can watch it think: [`Interfaccia/`](Interfaccia/) is an experimental real-time
visualizer — "The Observatory" — that renders the neurons firing as they spike.

## The honest version

This started as an attempt to build a persistent software *entity* — "Ivy" — with internal
state, memory, and constrained autonomy. Somewhere along the way it turned into something more
disciplined and more useful: an exercise in making an unpredictable component (an LLM) behave
inside a deterministic, auditable system.

It is a solo project. It has no production users. It is the most serious thing I have built, and
I am putting it here so it doesn't rot on a hard drive — if you find it useful, use it (under
the license below).

## How it fits the rest of my work

There is one instinct running through everything I build: **take a system that is closed, dead,
or unpredictable, and make it open, alive, and under deterministic control.**

- [BadBlood-Revival](https://github.com/ThePie88/BadBlood-Revival) — reverse the protocol of a
  game whose servers were shut down, and bring online play back.
- [FO4_Wrld](https://github.com/ThePie88/FO4_Wrld) — give a closed single-player game the
  multiplayer netcode it never had.
- **Project V** — take the most opaque, stochastic component there is (an LLM) and wrap it in a
  kernel that is deterministic, auditable, and revocable.

Same move, different target.

## What's solid

- **LLM-as-mouth, enforced.** The model returns text, validated against the `SpeechPlan`; on
  violation it retries, then falls back to a deterministic response. It has no channel to
  memory, tools, goals, or constraints.
- **Determinism + replay.** Logical time (not wall-clock), seeded RNG, quantization before
  thresholds, deterministic tie-breaks. A SHA-256 decision hash certifies that two runs decided
  identically.
- **Append-only audit.** Every turn emits structured events (state updates, gating decisions,
  memory writes, tool calls/denials), exportable as a hash-chained, tamper-evident bundle.
- **A spiking-neuron controller with real authority** (see above): 10 Izhikevich neurons + a
  128-node reservoir, trained readout, learning via R-STDP, crystallization as a bifurcation.
- **Snapshot / restore.** Full state — including the neural sub-state — serializes and restores
  idempotently, preserving the decision hash.
- **Kill-switch.** When active: zero LLM calls, audit only.
- **Tested.** ~670 tests (last full run: 671 passed, 1 skipped) and a `kernel-check` freeze
  gate. Pure Python — `pydantic` is the only required dependency; `fastapi` is optional (API +
  dashboard).

## What I don't claim (and what's unfinished)

I would rather be honest than impressive:

- **No production deployment.** Tested hard in controlled runs with local models; never served
  real users.
- **Real-LLM conformance is unfinished.** In the last run against a real model, roughly half the
  responses failed validation and fell back to the deterministic safety path. Tuning the speech
  contract per model is open work.
- **No formal stability proof.** State stays bounded *by construction* (clamps + validators),
  verified empirically over many trajectories — but this is not a Lyapunov stability proof, and
  I say so explicitly in [`progetto/STABILITY.md`](progetto/STABILITY.md).
- **The neural controller is one backend, not a hard requirement.** A finite-state machine would
  cover today's gating; the spiking-neuron brain is the default because it's more interesting and
  measurably better on temporal tasks, but it's swappable via a plugin protocol (a linear-ODE
  baseline ships alongside it).
- **The specifications are in Italian.** `progetto/` (SPEC, ARCH, the design pillars, the V1–V6
  milestone docs) is written in Italian and not yet translated.

## Architecture

The LLM lives in a "voice" layer with no path to memory, tools, or goals; all side effects flow
`Core → Governance → Persistence/Tools`. Full picture in [`progetto/ARCH.md`](progetto/ARCH.md)
and an English overview in [`progetto/IVY_ARCHITECTURE.md`](progetto/IVY_ARCHITECTURE.md).

## Install / Run / Test

Requires Python 3.10+.

```bash
pip install -e .            # core
pip install -e ".[dev]"     # + tests
pip install -e ".[server]"  # + API and dashboard
```

```bash
# Exam mode (offline, deterministic) — writes artifacts to artifacts/
python -m pie.cli run --exam --llm fake

# Interactive session
python -m pie.cli run --new-session --llm fake
python -m pie.cli run --session <session_id> --llm fake   # resume

# Test suite
pytest

# Kernel integrity (manifest + exam + replay)
python -m pie.cli kernel-check

# Validate a trace or snapshot
python -m pie.cli validate path/to/trace.jsonl
```

## Repository layout

```
pie/          the kernel (contracts, state engine, memory, governance, tools, runtime, CLI)
progetto/     the specifications (Italian): SPEC, ARCH, PILASTRI, V1-V6, STABILITY, DONE, TESTS,
              the Laws of Pie, and the crystallization tables
schemas/      JSON schemas / contracts
tests/        the test suite
examples/     conformance cassettes / scenarios
Interfaccia/  an experimental web visualizer of the neural reservoir ("The Observatory")
```

## License

Dual-licensed, attribution required:

- **Code** — Apache License 2.0 ([`LICENSE`](LICENSE)).
- **Documentation & specifications** (everything under `progetto/`, this README's prose, and
  other written docs) — Creative Commons Attribution 4.0 ([`LICENSE-docs`](LICENSE-docs)).

You may use, modify, and redistribute freely, but you must keep attribution to
**MrPie (ThePie88)**, state your changes, and — if you ship a modified version — give it a
different name and not imply endorsement. See [`NOTICE`](NOTICE).

## Author

Built by **MrPie** ([ThePie88](https://github.com/ThePie88)).
