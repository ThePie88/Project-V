# Pie Kernel

Pie is the deterministic runtime kernel for **Ivy** — an entity with persistent identity, emotional state, memory, and constrained autonomy.

## Install

```bash
pip install -e .
# Or with dev dependencies:
pip install -e ".[dev]"
```

Requires Python 3.10+.

## Run

### Exam Mode (offline, deterministic)

```bash
python -m pie.cli run --exam --llm fake
```

Produces artifacts in `artifacts/` (trace, snapshot, report).

### Interactive Session

```bash
# Create a new session
python -m pie.cli run --new-session --llm fake

# Resume an existing session
python -m pie.cli run --session <session_id> --llm fake
```

Interactive commands: `:quit`, `:save`, `:state`, `:whoami`, `:memory`, `:tools on/off`, `:killswitch on/off`.

### With Real LLM (LMStudio)

```bash
export LM_API_BASE=http://localhost:1234/v1
python -m pie.cli run --new-session --llm real
```

## Test

```bash
pytest
# Or verbose:
pytest -v --tb=short
```

### Kernel Check (full integrity)

```bash
python -m pie.cli kernel-check
```

Runs: manifest validation, exam run, replay, and optionally real LLM conformance.

## Validate

```bash
python -m pie.cli validate path/to/trace.jsonl
python -m pie.cli validate path/to/snapshot.json
```

## Kill-switch (Legge IV)

```bash
python -m pie.cli kill-switch on --session <id>
python -m pie.cli kill-switch off --session <id>
```

## Architecture

See [progetto/ARCH.md](progetto/ARCH.md) for module responsibilities, contracts, and data flow.

## Project Structure

```
pie/                    Core kernel
  contracts/            Pydantic schemas (Event, State, SpeechPlan, Memory, Constraint)
  persistence/          Atomic writes, JSONL stores
  session/              Session management + identity bootstrap
  state_engine/         Pluggable state evolution (DefaultODE)
  crystallization/      Constraint proposal engine
  memory/               Memory policy + view
  tools/                Tool executor with allowlist + capabilities
  config/               Romance guardrails, crystallization rules
  counterfactuals.py    K-alternative deliberation (V3.2)
  killswitch.py         Kill-switch enforcement (V3.3)
  metabolism.py         Cost model + budget gating (V3.4)
  routines.py           Routine/skill library (V3.4)
  runtime.py            Turn loop + pipeline
  cli.py                CLI entry point
progetto/               Specifications (V1-V4, SPEC, DONE, TESTS, SEED)
tests/                  pytest suite
artifacts/              Exam output + golden baselines
schemas/                JSON schemas
```

## Leggi di Pie

1. **Trasparenza** — every decision is traced and auditable
2. **Supremazia del Creatore** — the Creator can override any decision
3. **Integrità dello stato** — state is append-only and schema-validated
4. **Non-autonomia** — Ivy cannot act without Creator approval (kill-switch enforced)
5. **Revocabilità** — everything is reversible within N ticks

## Troubleshooting

- **LLM not responding**: ensure LMStudio is running on `localhost:1234`
- **Manifest mismatch**: run `python -c "from pie.kernel_manifest import build_manifest, canonical_manifest_json; __import__('pathlib').Path('pie/kernel_manifest.json').write_text(canonical_manifest_json(build_manifest()))"`
- **Golden trace mismatch**: set `PIE_GOLDEN_WRITE=1` before running exam
- **Pydantic warnings**: expected (v1-style validators on v2 runtime), functionally correct
