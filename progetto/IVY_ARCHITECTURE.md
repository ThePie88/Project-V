# Ivy: A Deterministic Cognitive Architecture for Governed AI Systems

## What Ivy Is

Ivy is a cognitive architecture that separates **decision-making** from **language generation**. The kernel makes every decision — what to remember, which tools to authorize, how much to say, how deeply to reason. The language model only generates text, constrained by a contract the kernel validates and rejects if violated.

Every decision is traceable. Every session is reproducible. The model cannot override the system's rules — not by prompt injection, not by hallucination, not by emergent behavior. This is enforced at the architecture level, not the prompt level.

---

## Why This Matters

Current AI systems treat the language model as both brain and mouth. When the model decides what to remember, which tools to call, and how to enforce safety — you're trusting a stochastic text generator with governance. The result: unreproducible behavior, unauditable decisions, and safety guardrails that work until they don't.

Ivy inverts this. The kernel is deterministic and auditable. The LLM is a constrained output channel. You can freeze a session, restore it weeks later, and get the same decisions — verified by cryptographic hash. You can trace exactly why any decision was made, down to which neural channel gated which capability and by how much.

This is not a prompting strategy. It's a runtime architecture.

---

## Architecture

```
User Input
    |
    v
+-----------------------------------------------+
|  KERNEL (deterministic, auditable)             |
|                                                |
|  Neural Controller ----> ControlVector         |
|  (10 Izhikevich SNN     5 channels:           |
|   + 128 ESN reservoir)   - memory_gate         |
|                          - tool_gate           |
|                          - verbosity_bias      |
|                          - consolidation       |
|                          - counterfactual_k    |
|                                                |
|  CV Gating --> Deliberation --> SpeechPlan     |
|  (5 gates)    (K alternatives    (contract     |
|                scored/ranked)     for LLM)     |
|                                                |
|  Memory ----> Recall (BM25) --> Consolidation  |
|  (append-only, deterministic, no LLM in loop)  |
|                                                |
|  Tool Governance --> Executor --> Sandbox       |
|  (CV gate + budget    (allowlist    (isolated   |
|   + capability)        + rate limit)  FS)      |
|                                                |
|  Kill-Switch: zero LLM calls when active       |
+----------------------+------------------------+
                       | SpeechPlan + context
                       v
                 +-----------+
                 |    LLM    |  <-- generates text, nothing else
                 |  (voice)  |  <-- validated, rejected if violated
                 +-----------+
                       |
                       v
                 Journal (JSONL, append-only, hash chain)
```

---

## Three Design Principles

### 1. The Model Cannot Violate System Rules

The LLM receives a SpeechPlan contract specifying intent, required terms, forbidden terms, token limits, and verbosity level. The kernel validates every response against 8 constraint types: format, forbidden words, required content, token limits, verbosity bounds, fabricated experience detection (Legge III), assistant-pattern detection, and language enforcement. Violations trigger automatic retry with correction directives. After maximum retries, a deterministic fallback fires.

The LLM cannot write to memory, activate tools, modify goals, or change constraints. These are kernel operations that happen before and after the LLM call. The model's output is text — nothing more.

### 2. Every Decision Is Reproducible

Same seed, same inputs, same decisions. The neural controller (SNN + reservoir) uses a deterministic RNG. Memory writes are proposed by regex policy on user input, never on LLM text. Consolidation extracts facts from the journal using pattern matching, not generation. Counterfactual deliberation scores and ranks K alternatives using deterministic scoring functions.

A decision hash — SHA-256 computed from kernel events, excluding LLM text, timestamps, and volatile IDs — certifies that two sessions made identical decisions. Snapshot/restore produces the same decision hash across freeze-resume cycles.

### 3. Every Decision Is Traceable

Every gating decision emits a structured event:

```json
{
  "type": "CV_GATING",
  "content": {
    "channel": "tool_gate",
    "value": 0.24,
    "threshold": 0.3,
    "decision": "BLOCK",
    "reason": "tool_gate < threshold"
  }
}
```

There are 5 gating points per turn (memory, tools, verbosity, consolidation, counterfactual depth). Tool calls emit TOOL_CALL / TOOL_RESULT / TOOL_DENIED with call IDs, operation types, and deny reasons. Memory writes emit MEMORY_PROPOSED / MEMORY_APPENDED with content hashes. The full journal is an append-only, hash-chained forensic record that can be bundled into a tamper-evident ZIP with external anchor verification.

---

## Technical Stack

### Neural Controller
- 10 Izhikevich spiking neurons with Spike-Timing Dependent Plasticity (STDP)
- 128-unit Echo State Network reservoir (spectral radius 0.95, deterministic LCG)
- Trained readout via ridge regression (Cholesky decomposition, no numpy)
- Reward-modulated STDP (R-STDP) with eligibility traces for online learning
- Causal authority proven: 5/5 pipeline decisions diverge across 3 neural conditions

### Governance
- Kill-switch: when active, zero LLM calls, zero decisions, only audit (INPUT logged)
- Tool gating: 3-gate stack (ControlVector threshold + budget allowance + capability check) before the LLM sees tool schemas
- Tool execution: allowlist + filesystem capability + rate limiting + sandbox isolation
- Romance guardrails and fabricated-experience detection (Legge III) enforced at validation, not prompting
- Constraint crystallization: kernel proposes and applies behavioral constraints from observed patterns

### Memory
- BM25 episodic recall with state-driven bias and recency weighting
- Deterministic consolidation: regex fact extraction (Italian + English), narrative digest
- Append-only stores with atomic writes (temp + fsync + rename)
- Memory influences decisions through the SpeechPlan (beliefs become assertable facts, preferences modulate verbosity)

### Persistence and Audit
- Snapshot/restore: full neural state serialization with SHA-256 idempotency verification
- Migration registry for schema evolution (chain v1 -> v2 -> ... -> target)
- Audit bundles: deterministic ZIP with SHA-256 hash chain, external append-only anchor, zip-slip defense
- Bundle verification: tamper detection on any modified, added, or removed file

### API and SDK
- FastAPI server with 7 versioned endpoints, session locks, seed allowlist
- PieClient SDK (urllib-based, zero external dependencies)
- JSON Schema contracts with canonical fingerprints for breaking-change detection
- Telemetry: per-turn delta and bulk retrieval, journal pagination

### Testing
- 671 tests, 0 failures
- Deterministic LLM stub server (OpenAI-compatible, stdlib-only HTTP) exercises the full RealLLM code path in CI without GPU: tool calling, retry/validation loop, cache bypass, second LLM call, parse resilience
- Live integration tests with real models (LMStudio) verify end-to-end behavior including decision hash determinism across snapshot/restore cycles

---

## What Ivy Is Not

**Not a prompt wrapper.** The kernel is a compiled pipeline with typed contracts, not a system prompt that hopes the model follows instructions.

**Not an agent framework.** Agents delegate decisions to the model and observe outcomes. Ivy makes decisions in the kernel and delegates only text generation to the model.

**Not a RAG system with memory.** Memory writes are kernel operations governed by policy. The model cannot decide what to remember. Recall is BM25 over structured episodes, not vector similarity over chunks.

**Not dependent on a specific model.** The architecture is model-agnostic. The LLM is a replaceable output channel. Tested with Qwen3, Ministral, and a deterministic stub — same kernel behavior across all three.

---

## Verification

The system can be verified at three levels:

1. **Unit level**: 671 tests covering every kernel component — contracts, state engine, neural controller, memory, governance, persistence, audit, API
2. **Integration level**: LLM stub exercises the full RealLLM path (tool calling, retry loop, cache bypass, second LLM call) without external dependencies
3. **Live level**: 20-turn sessions with real models, snapshot/restore replay, decision hash comparison, audit bundle generation and tamper verification

```
$ python -m pytest tests/ -q
671 passed, 1 skipped in 255s
```

---

---

Built by **MrPie** ([ThePie88](https://github.com/ThePie88)). Kernel frozen at v0.0.0.
