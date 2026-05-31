# ARCH — Architettura “Progetto 4 / Pie Kernel” — v0

Questo documento descrive **l’architettura completa** del sistema: moduli, responsabilità, contratti, flussi, persistenza, errori, modalità runtime.
È scritto per essere **implementabile** e **testabile** (coerente con `SPEC.md` e `DONE.md`).

> **Regola:** se un comportamento non è rappresentabile in *contratti + trace + test*, non fa parte dell’architettura (è un desiderio).

---

## 0) Vista d’insieme

### 0.1 Obiettivo operativo
Costruire un agente che:
- mantiene **stato interno persistente** (drive/emozioni/valori/tratti/fiducia)
- genera **goal endogeni** e prende decisioni tramite controfattuali
- aggiorna **memoria vera** (narrativa/credenze/fiducia) con policy esplicite
- converte alcune esperienze/emozioni in **vincoli cristallizzati** (eseguibili)
- parla tramite un LLM usato come **bocca**, vincolato da `SpeechPlan`
- produce **audit log** completo e supporta **replay deterministico**
- è governato da invarianti (Leggi di Pie) e da **capabilities** (sandbox di default)

### 0.2 Layering (separazione delle responsabilità)

```
┌──────────────────────────────────────────────────────────┐
│ UI / Client / API (opzionale)                            │
└──────────────────────────────────────────────────────────┘
                │ input/output
┌──────────────────────────────────────────────────────────┐
│ Runtime Loop / Orchestrator                              │
│  - sequencing, time, turn management, kill-switch        │
└──────────────────────────────────────────────────────────┘
                │ calls
┌──────────────────────────────────────────────────────────┐
│ Core Decisionale (Brain)                                  │
│  - Perception/Interpretation                              │
│  - State Engine                                           │
│  - Goal Engine                                            │
│  - Counterfactuals + Action Selection                     │
│  - Memory Policy + Crystallization                        │
└──────────────────────────────────────────────────────────┘
                │ emits SpeechPlan / ToolIntents
┌──────────────────────────────────────────────────────────┐
│ Governance + Laws + Guards                                │
│  - invariants check, authority model, runtime guards       │
└──────────────────────────────────────────────────────────┘
                │
┌──────────────────────────────────────────────────────────┐
│ Side-Effect Subsystems                                    │
│  - LLM Adapter (Voice)                                    │
│  - Tools/Sandbox + Capabilities                            │
│  - Persistence (State/Memory/Trace)                        │
└──────────────────────────────────────────────────────────┘
                │
┌──────────────────────────────────────────────────────────┐
│ Observability (Trace/Audit/Replay)                        │
└──────────────────────────────────────────────────────────┘
```

**Nota critica:** l’LLM è confinato nel layer “Voice”. Il “Brain” non dipende dal modello specifico.

---

## 1) Struttura del repository (minima, v0)

```
/src
  /runtime            # loop, orchestration, CLI wiring
  /governance         # laws, authority, guards, kill-switch
  /trace              # event log writer, snapshot, replay engine
  /core
    /perception       # parsing input → intents/features
    /state            # state engine backends + state schema objects
    /goals            # goal generation + scheduler
    /counterfactuals  # alternative generation + scoring
    /memory           # memory store API + policy
    /crystallization  # emotion→constraint engine
    /metabolism       # budget/cost accounting
    /routines         # routine/skills library
  /voice              # LLM adapter, validators, templates, cache
  /tools              # tool registry, sandbox tools, capability checks
  /validation         # schema validation, integrity checks
/contracts
  event.schema.*
  state.schema.*
  speechplan.schema.*
  memory.schema.*
  constraint.schema.*
  toolcall.schema.*
/tests
  /unit
  /integration
  /e2e
  /property
/examples
  /exam_mode
  /llm_conformance
/artifacts
/scripts
  run*
  test*
  replay*
  validate*
```

---

## 2) Entità dati (contratti) — definizione funzionale

> I contratti vengono implementati in `/contracts`. Qui descriviamo il **contenuto obbligatorio** e le regole.

### 2.1 Event (trace jsonl) — `Event`
**Scopo:** “flight recorder” append-only, fonte primaria di debug e replay.

Campi minimi:
- `ts` (timestamp; non deve influenzare decisioni nel replay)
- `session_id`, `turn_id`, `event_id`, `parent_event_id?`
- `type` (enum)
- `state_before_hash`, `state_after_hash`
- `payload` (oggetto, dipende dal type)
- `decision_rationale` (stringa breve + refs)
- `laws_check` (pass/fail + violations)
- `cost` (token/latency/budget)

Tipi minimi consigliati:
- `BOOT`, `INPUT_RECEIVED`, `PERCEPTION_DONE`, `STATE_UPDATED`
- `GOALS_GENERATED`, `COUNTERFACTUALS_EVAL`, `ACTION_SELECTED`
- `SPEECHPLAN_EMITTED`, `LLM_REQUEST`, `LLM_OUTPUT`, `LLM_FALLBACK`
- `MEMORY_WRITE`, `CONSTRAINT_CREATED`, `TOOL_INTENT`, `TOOL_CALL`, `TOOL_RESULT`
- `ERROR`, `STOP`, `SHUTDOWN`, `REPLAY_MARKER`

**Regole:**
- append-only
- ogni `ACTION_SELECTED` deve avere un rationale e riferimenti alle alternative scartate (se presenti)
- ogni side effect produce eventi dedicati (tool, memory, llm)

### 2.2 State Snapshot — `State`
**Scopo:** rappresentare “l’essere” in forma serializzabile e versionabile.

Parti consigliate:
- `schema_version`
- `creator_anchor`
- `drives`: (curiosità, cautela, socialità, fatica, ecc.)
- `affect`: (arousal, valence, tension, attention)
- `traits`: parametri lenti (stile decisionale, prudenza, ecc.)
- `values`: pesi dinamici (non moralità emergente; preferenze operative)
- `trust`: mappa entità→punteggio e contesto
- `active_goals`: (id, stato, progress)
- `constraints_active`: lista vincoli (id, strength, decay, reason refs)
- `metabolism`: budget attuali, contatori
- `routines`: indice/summary (la libreria può stare separata come store)
- `persistence`: ultimi offset/log pointers
- `rng_seed` o `rng_state` (per replay deterministico)

**Regole:**
- versionamento schema obbligatorio
- niente riscritture retroattive: si evolve incrementale

### 2.3 SpeechPlan — `SpeechPlan`
**Scopo:** contratto che *lobotomizza* l’LLM: decide il Brain, realizza la Voice.

Campi minimi:
- `intent`
- `tone` (enum controllato: tecnico, neutro, caldo, asciutto, ecc.)
- `must_include[]`
- `must_not_include[]`
- `facts_allowed[]` (refs a memoria/eventi o facts whitelisted)
- `references[]` (event/memory ids citabili)
- `output_format` (`TEXT` o `JSON`)
- `max_tokens`, `verbosity`
- `style_constraints` (es: lingua, persona, format)
- `post_conditions` (es: “non scrivere memoria”, “fai una domanda”, ecc.)

**Regole:**
- la Voice **non può** cambiare `SpeechPlan`
- output deve essere validato; altrimenti retry → fallback

### 2.4 Memory Record — `MemoryRecord`
**Scopo:** memoria append-only, separata dal trace (che è sempre completo).

Tipi consigliati:
- `NarrativeMemory` (identità, storyline)
- `Belief` (fatto ipotizzato con confidence)
- `Preference` (preferenze operative)
- `TrustUpdate` (cambi fiducia)
- `RoutineRecord` (routine salvata)
- `ConstraintRecord` (vincolo cristallizzato con reason)

Campi comuni:
- `memory_id`, `created_ts`, `type`
- `content` (strutturato)
- `source_refs` (event ids)
- `confidence` (se applicabile)
- `tags/context`

**Regole:**
- append-only
- nessun “falso ricordo”: ogni record deve avere `source_refs`

### 2.5 Constraint — `Constraint`
**Scopo:** oggetto eseguibile che restringe azioni/goal.

Campi minimi:
- `constraint_id`
- `kind` (FORBID, REQUIRE_CONFIRMATION, RAISE_CAUTION, TRUST_DELTA, LIMIT_VERBOSITY, LIMIT_TOOLS, ecc.)
- `target` (action_class, context_tag, entity, ecc.)
- `strength` (0..1 o livelli)
- `decay` (opzionale; anche solo “none”)
- `trigger` (evento/i e condizioni)
- `explanation` (testo breve)
- `source_refs` (event ids)

**Regole:**
- deve poter essere valutato da Goal Engine / Action Selection
- deve apparire nel trace quando blocca o modifica una scelta

### 2.6 Tool Intent / Tool Call — `ToolIntent`, `ToolCall`
**Scopo:** separare decisione di chiamare un tool dall’esecuzione effettiva.

`ToolIntent` (dal Brain):
- `tool_name`
- `args` (schema)
- `capabilities_required[]`
- `preconditions`
- `expected_side_effects`

`ToolCall` (esecuzione):
- `tool_name`, `args`, `capabilities_checked`
- `result` (success/fail, output)
- `side_effects_committed` (sì/no)

---

## 3) Moduli — specifica completa (responsabilità, I/O, dipendenze)

### 3.1 Runtime Orchestrator (`src/runtime`)
**Responsabilità**
- gestire loop turn-based (o tick-based + turn)
- gestire `session_id`, `turn_id`, sequencing degli step
- orchestrare chiamate ai moduli Core + Governance + Voice + Tools
- applicare kill-switch e safe-stop
- leggere config e inizializzare dipendenze

**Input**
- input utente (testo/struttura)
- config runtime
- stato/persistenza

**Output**
- output testuale (risposta)
- artefatti (trace, snapshot, report exam mode)

**Dipendenze**
- governance, trace, core, voice, tools, persistence

---

### 3.2 Governance + Laws + Guards (`src/governance`)
**Responsabilità**
- definire Leggi di Pie come invarianti formali
- controllare authority model: cosa può cambiare cosa
- runtime guards:
  - impedire scritture memoria dal voice
  - impedire tool calls non autorizzate
  - impedire “moral autonomy escalation”
- kill-switch e safe-stop

**Input**
- proposte di azione, speechplan, memory write, tool intent
- state + event context

**Output**
- `laws_report`
- allow/deny + motivazione

**Dipendenze**
- contracts, validation, trace

---

### 3.3 Trace/Audit/Replay (`src/trace`)
**Responsabilità**
- scrivere eventi jsonl append-only
- gestire snapshot di stato
- replay engine: rilegge trace e riproduce pipeline
- supportare “golden traces” per regressioni
- utility: correlazione eventi, indexing (facoltativo)

**Input**
- eventi generati dall’orchestrator e dai moduli
- stato serializzato

**Output**
- `trace.jsonl`, `state_snapshot.*`, report replay

**Dipendenze**
- validation, persistence

---

### 3.4 Core: Perception (`src/core/perception`)
**Responsabilità**
- trasformare input in:
  - `UserIntent` (saluto, domanda, richiesta azione, feedback, ecc.)
  - features (tono percepito, urgenza, entità menzionate)
  - eventuali “feedback signals” (es: correzione dell’utente)

**Input**
- raw user input
- context (stato, memoria minima consultabile)

**Output**
- `PerceptionResult`

**Dipendenze**
- minima: può usare memory queries, ma non modificare memoria

---

### 3.5 Core: State Engine (`src/core/state`)
**Responsabilità**
- aggiornare `State` in base a:
  - tick/time
  - `PerceptionResult`
  - outcome di azioni/LLM/tools
- fornire segnali: arousal/valence/attention ecc.
- supportare backend sostituibili:
  - `SimpleDynamicsBackend` (v0)
  - `ReservoirBackend` (opzionale)
  - `SNNBackend` (v2+, con surrogate gradients solo se serve)

**Input**
- `State` precedente
- eventi (perception, tool results, ecc.)
- config (decay, clamp, ecc.)

**Output**
- `State` aggiornato + `StateSignals`

**Dipendenze**
- none (core puro)

---

### 3.6 Core: Metabolism (budget/costi) (`src/core/metabolism`)
**Responsabilità**
- mantenere budget per turno:
  - alternative count budget
  - token budget (stimato)
  - tool calls budget
- stimare costi (anche grezzi in v0)
- fornire vincoli al Goal Engine e Counterfactuals

**Input**
- stato + segnali
- policy config
- stime dalla Voice (token) e Tools (latency)

**Output**
- `BudgetState` + decision hints (riduci verbosity, riduci alternative, ecc.)

---

### 3.7 Core: Goal Engine (`src/core/goals`)
**Responsabilità**
- generare goal endogeni (da stato) e reattivi (da input)
- assegnare priorità multi-obiettivo
- scegliere goal attivi (scheduling)
- gestire lifecycle (progress/decay/preemption)

**Input**
- `State`, `StateSignals`
- `PerceptionResult`
- `ConstraintsActive`
- `BudgetState`

**Output**
- `GoalsList` + `ActiveGoalSet`

**Dipendenze**
- constraints, metabolism

---

### 3.8 Core: Counterfactuals + Action Selection (`src/core/counterfactuals`)
**Responsabilità**
- generare alternative (piani/azioni) compatibili con goal e vincoli
- scoring: utility, rischio, costo, coerenza vincoli
- scegliere `SelectedAction` e registrare scarti
- produrre rationale strutturato (usato nel trace)

**Input**
- goal set
- constraints
- budget
- state

**Output**
- `Alternatives[]`, `SelectedAction`, `DecisionRationale`

---

### 3.9 Core: Memory Store + Policy (`src/core/memory`)
**Responsabilità**
- API di lettura/scrittura append-only
- policy “memoria ≠ archivio”: decide cosa scrivere in quale store
- gestire credenze e fiducia separatamente
- garantire `source_refs` (no falsi ricordi)

**Input**
- eventi e outcome
- state signals
- feedback utente (es: correzione)

**Output**
- `MemoryWriteIntent` (proposta) → approvazione governance → commit
- query results

**Dipendenze**
- governance, persistence, validation

---

### 3.10 Core: Crystallization (`src/core/crystallization`)
**Responsabilità**
- valutare se un evento emotivo/esperienziale supera soglia
- generare `Constraint` eseguibile + record memoria
- mantenere mapping da tabella (Excel → regole)

**Input**
- state signals (affect)
- perception + outcome
- history minima (ripetizioni, contesto)
- config soglie e classi

**Output**
- `ConstraintIntent` (proposta) → governance → commit
- aggiornamento `constraints_active` nello State

**Dipendenze**
- memory store, governance, constraints schema

---

### 3.11 Core: Routines/Skills (`src/core/routines`)
**Responsabilità**
- salvare routine (trigger→sequenza→outcome)
- matching di trigger e ranking
- decay/forgetting
- fornire “candidate actions” al Counterfactuals generator

**Input**
- eventi/outcome
- perception features
- constraints/budget

**Output**
- `RoutineCandidates[]` + aggiornamenti store routine

---

### 3.12 Voice: LLM Adapter (`src/voice`)
**Responsabilità**
- costruire richiesta LLM a partire da `SpeechPlan`
- invocare LLM reale (modello configurabile) o FakeLLM
- validare output (schema/constraints)
- retry/correzione
- fallback deterministico
- caching per replay (opzionale ma consigliato)

**Input**
- `SpeechPlan`
- `facts_allowed` (contenuti consentiti)
- config modello + conformance settings

**Output**
- `VoiceOutput` (testo o JSON)
- metriche: token stimati, retry count, fallback flag

**Regole forti**
- nessuna scrittura memoria
- nessun tool call
- nessuna generazione goal
- se output non conforme: non rompe il sistema, produce fallback e logga `MODEL_NONCONFORMANT`

---

### 3.13 Tools/Sandbox + Capabilities (`src/tools`)
**Responsabilità**
- registry tools
- enforcement capabilities
- sandbox FS (o tool environment) di default
- audit di ogni tool call e result

**Input**
- `ToolIntent` approvato
- capabilities attive

**Output**
- `ToolResult`
- side effects in sandbox

**Regole**
- senza capability: negare sempre
- azioni distruttive: require confirmation + capability

---

### 3.14 Validation (`src/validation`)
**Responsabilità**
- validare eventi, snapshot, memory, speechplan contro schemi
- integrità (hash chain opzionale)
- helper “validate trace/snapshot” (CLI)

---

### 3.15 Persistence (`src/runtime` o `src/core/memory` a seconda stack)
**Responsabilità**
- gestire file/db per:
  - trace append-only
  - snapshots
  - memory store append-only
  - routines store
  - conformance cache
- migrazioni schema
- crash-safe commits (minimo: write temp + atomic rename)

---

## 4) Sequenza di un turno (diagramma)

### 4.1 Turn pipeline (happy path)

```
[INPUT] UserText
  └─> Trace: INPUT_RECEIVED
      └─> Perception.parse()
          └─> Trace: PERCEPTION_DONE
              └─> StateEngine.update()
                  └─> Trace: STATE_UPDATED
                      └─> Metabolism.update_budget()
                          └─> Trace: COST_UPDATED
                              └─> GoalEngine.generate()
                                  └─> Trace: GOALS_GENERATED
                                      └─> Counterfactuals.eval()
                                          ├─> Trace: COUNTERFACTUALS_EVAL (alternatives + scores)
                                          └─> ActionSelector.select()
                                              └─> Trace: ACTION_SELECTED
                                                  ├─> If action == SPEAK:
                                                  │     └─> SpeechPlan.build()
                                                  │         └─> Governance.check()
                                                  │             └─> Trace: SPEECHPLAN_EMITTED
                                                  │                 └─> Voice.generate()
                                                  │                     ├─> Trace: LLM_REQUEST
                                                  │                     ├─> Trace: LLM_OUTPUT or LLM_FALLBACK
                                                  │                     └─> Output text to user
                                                  ├─> If action == TOOL:
                                                  │     └─> ToolIntent.emit()
                                                  │         └─> Governance.check()
                                                  │             └─> Tools.execute()
                                                  │                 ├─> Trace: TOOL_CALL
                                                  │                 └─> Trace: TOOL_RESULT
                                                  └─> MemoryPolicy.consider_writes()
                                                      ├─> Crystallization.maybe_create_constraint()
                                                      ├─> Governance.check()
                                                      └─> Persistence.commit()
                                                          └─> Trace: MEMORY_WRITE / CONSTRAINT_CREATED
```





### 4.2 Kill-switch (safe-stop)
- Il runtime può ricevere kill-switch da:
  - input esplicito (comando)
  - governance (violazione legge)
  - errore critico (persistence corrotto)

Safe-stop deve:
1) emettere `STOP`
2) flush trace
3) snapshot state consistente
4) chiudere tool/voice se in corso (timeout + abort)
5) uscire con exit code definito

---

## 5) Determinismo e replay

### 5.1 Fonti di non determinismo da eliminare
- RNG non seedato
- uso di timestamp come input decisionale
- chiamate LLM non cached in replay
- dipendenze tool con output variabile

### 5.2 Strategie
- seed globale + rng state nello State
- clock separato: `tick_count` e `logical_time` (timestamp solo informativo)
- LLM:
  - modalità cached (store request-hash → output) per replay
  - oppure replay con FakeLLM (decisioni devono restare identiche)
- Tools:
  - in exam mode, tools deterministici o registrati come cassette

---

## Determinismo: regole operative (v0)

### Tempo
- `timestamp` (clock reale) è solo informativo nel trace.
- Il core usa solo `logical_time` per qualsiasi decisione (ranking, soglie, trigger).

### Ordering e ID
- Eventi/goal/alternative/vincoli devono avere ID deterministici (contatori o hash stabili).
- Qualsiasi lista che influisce su scelte viene ordinata prima dell’uso.
- Tie-break obbligatorio: a parità di punteggio, confronto deterministico su `id`.

### Floating point e soglie (ODE)
- State Engine v0 usa Euler + `dt` fisso.
- Dopo ogni update: clamp → round (quantizzazione) → solo poi soglie/ranking/trigger.
- Le soglie di cristallizzazione usano valori già quantizzati (niente dipendenza da epsilon).


---

## 6) Error model (tassonomia + policy)

### 6.1 Categorie errori (Event `ERROR.code`)
- `LAW_VIOLATION` (hard fail → STOP)
- `PERSISTENCE_ERROR` (hard fail → STOP)
- `SCHEMA_VALIDATION_ERROR` (hard fail in test; runtime: fallback/stop a seconda)
- `MODEL_NONCONFORMANT` (soft fail → fallback + log)
- `TOOL_DENIED` (soft fail → alternative action)
- `TOOL_ERROR` (soft fail → alternative action; se ripetuto può cristallizzare cautela)
- `REPLAY_DIVERGENCE` (test fail; runtime: report)
- `BUDGET_EXCEEDED` (soft fail → reduce deliberation/verbosity)
- `UNKNOWN_ERROR` (hard fail con dump e STOP in exam mode)

### 6.2 Policy per tipo
- Hard fail: stop, snapshot, report, exit code non-zero
- Soft fail: degradare gracilmente, tracciare sempre, mai crash senza trace

---

## Esame Mode: esito contrattuale (v0)

Esame Mode è un “giudice” automatico, non una demo narrativa.

- Input: scenario standardizzato (cassette).
- Output: artefatti obbligatori (trace, snapshot, report, replay report).
- Invarianti attese: niente tool reali, trace valido, decisioni tracciate, vincoli coerenti.
- Esito: **pass/fail** con motivazione scritta (non “sembra ok”).


---
## 7) Authority model (v0, minimo ma completo)

### 7.1 Chi può fare cosa
- **Brain**: può proporre decisioni, SpeechPlan, MemoryWriteIntent, ToolIntent
- **Governance**: può consentire/negare e può attivare kill-switch
- **Voice (LLM)**: può solo produrre `VoiceOutput` dentro SpeechPlan
- **Tools**: eseguono solo se capability presente
- **Persistence/Trace**: scrive append-only, non decide

### 7.2 Cambi “valoriali” e “leggi”
- Qualunque cambiamento a:
  - mapping cristallizzazione
  - policy memoria
  - parametri valori/tratti “lenti”
  richiede:
  - evento tracciato
  - autorizzazione definita (config firmata/capability)
  - migrazione schema se cambia struttura

*(v0: può essere “config file + checksum” come placeholder dell’autorizzazione; la semantica deve esistere.)*

---

## Guardrails strutturali (dove vivono i guard)

- Il layer Voice (LLM) riceve **solo** `SpeechPlan` + (opzionale) una view read-only dello stato.
- Il layer Voice non ha API per:
  - scrivere memoria
  - modificare goal
  - invocare tools
- Scritture e side-effect passano solo via: `Core → Governance → Persistence/Tools`.
- Qualunque bypass (anche “utility comoda”) è una violazione e deve generare evento `LAW_VIOLATION`.



## 8) Configurazione (v0)

### 8.1 Config file
Un file (YAML/TOML/JSON) definisce:
- `seed`
- `exam_mode` settings (cassette paths, deterministic tools)
- budgets: max alternatives, max tokens, tool budgets
- thresholds: crystallization
- voice:
  - provider (local/remote)
  - model id
  - temperature (se usata)
  - conformance retries
- persistence paths
- capabilities default (sandbox only)

### 8.2 Override via CLI
- `--seed`
- `--exam`
- `--model`
- `--no-llm` (FakeLLM)
- `--capabilities=...`

---

## 9) Mapping della tabella di Cristallizzazione (asset → regole)

### 9.1 Fonte dati
La tabella (Excel) è una *spec* di mapping:
- famiglie emozionali
- condizioni (trigger)
- vincoli risultanti

### 9.2 Trasformazione in regole
- conversione in `contracts/crystallization_rules.*` (JSON/YAML) oppure in codice + test tabellari
- i test tabellari sono la fonte di verità (come in `DONE.md`)

---

## 10) Boundary conditions (cosa NON entra nell’architettura v0)

- niente sensori fisici reali
- niente filesystem reale fuori sandbox
- niente rete libera
- niente training end-to-end come decision-maker (SNN come backend sostituibile, non sovrano)
- niente “coscienza” dichiarata: l’architettura mira a continuità e auditabilità, non a claim metafisici

---




## 11) Appendice — Checklist architetturale (self-audit)

- [ ] ogni modulo ha: responsabilità e non-responsabilità chiare
- [ ] ogni side effect produce eventi trace
- [ ] voice non ha canali per memory/tools/goals
- [ ] esiste una strategia replay (cached o fake)
- [ ] error model definito e tracciato
- [ ] capabilities bloccano azioni fuori sandbox
- [ ] schema versioning definito per State/Memory
- [ ] cristallizzazione produce vincoli eseguibili e testabili

## Future extensions (v1/v2) — mappa rapida per non perdere il filo

### v1 (Hardening)
- Voice/LLM Adapter: caching replay, retry/timeout policy, conformance matrix multi-modello.
- Trace/Replay: golden traces + tool di diff (decisioni vs linguaggio).
- Persistence: atomic commit, migrazioni robuste, crash recovery end-to-end.
- Governance: authority “firmato”, pending-review formalizzato per hard crystallization.
- Tools: sandbox più completa + permessi granulari + rate limiting.

### v2 (Espansioni)
- State Engine: backend plug-in (Reservoir/ODE/SNN) senza cambiare `State`/`Signals` contracts.
- Routines: learning più sofisticato + metriche di miglioramento.
- Embodiment: reflex layer separato + sensori/attuatori (sandbox → reale).
- Tools: rete controllata/allowlist + FS reale capability-first.
- Relazione: dinamiche romance/eros complete (tabella + composizioni + guardrail).
