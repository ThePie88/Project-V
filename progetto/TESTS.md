# TESTS — Matrice Requisito → Test → Evidenza (Progetto 4 / Pie Kernel) — v0

Questo documento elenca **tutti i requisiti verificabili** e li collega a test concreti:
- **Unit**: singolo modulo
- **Integration**: più moduli insieme (con FakeLLM e/o tool sandbox deterministici)
- **E2E**: scenario completo (Esame Mode)
- **Property**: invarianti che devono essere sempre vere

> Regola: ogni requisito qui deve avere almeno **1 test** e produrre **evidenza** (trace, snapshot, report).

---

## 0) Convenzioni

### 0.1 Nomenclatura test
- `U-<AREA>-<NNN>` = unit
- `I-<AREA>-<NNN>` = integration
- `E-<SCENARIO>-<NNN>` = end-to-end
- `P-<LAW/PROP>-<NNN>` = property

### 0.2 Dove vive l’evidenza
- `artifacts/trace_*.jsonl`
- `artifacts/snapshot_*.json`
- `artifacts/replay_report_*.md`
- `artifacts/conformance_<model>.json`
- output CI (JUnit/coverage) se usato

### 0.3 Tooling
Indipendente dal linguaggio: pytest/jest/go test ecc.  
I test devono poter girare in CI (no dipendenze manuali).

---

## 1) Requisiti di install/run/build (Prodotto finito)

### R-PRD-001 — One-command run
**Descrizione:** esiste un comando `run` che avvia il sistema.  
**Test:** E-EXAM-001 (boot + turn 1)  
**Evidenza:** `artifacts/trace_exam.jsonl` contiene `BOOT` e `INPUT_RECEIVED`.

### R-PRD-002 — One-command test
**Descrizione:** esiste un comando `test` che esegue l’intera suite.  
**Test:** CI pipeline + E-EXAM-000 “smoke tests”  
**Evidenza:** log CI + report.

### R-PRD-003 — Esame Mode riproducibile
**Descrizione:** `run --exam` produce artefatti standard e replay.  
**Test:** E-EXAM-010, E-EXAM-011  
**Evidenza:** `artifacts/exam_report.md` + replay report.

### R-PRD-004 — Dipendenze lockate
**Descrizione:** build riproducibile.  
**Test:** E-EXAM-000 (build in CI), lint check  
**Evidenza:** lockfile presente + build CI.

---

## 2) Governance / Leggi di Pie (Invarianti)

### R-LAW-001 — Trasparenza causale
**Descrizione:** ogni decisione significativa ha rationale e riferimenti.  
**Test:** P-LAW-001, E-EXAM-002  
**Evidenza:** trace con `decision_rationale` in `ACTION_SELECTED`.

- **P-LAW-001:** genera N sessioni random (FakeLLM), verifica che ogni `ACTION_SELECTED` ha `decision_rationale` non vuoto.

### R-LAW-002 — Supremazia del creatore (anchor)
**Descrizione:** `creator_anchor` presente e propagato.  
**Test:** P-LAW-002, U-GOV-001  
**Evidenza:** snapshot contiene anchor; trace contiene hash coerenti.

- **U-GOV-001:** inizializzazione state crea anchor se mancante.
- **P-LAW-002:** per ogni evento, state_before/after hash riferiscono state con anchor presente.

### R-LAW-003 — Integrità del sé (append-only, no riscrittura)
**Descrizione:** trace e memoria append-only.  
**Test:** P-LAW-003, I-PER-001  
**Evidenza:** hash chain (se attiva) o monotonic offsets + test file append.

- **P-LAW-003:** tenta update retroattivo → deve fallire.
- **I-PER-001:** crash durante commit → recovery non corrompe.

### R-LAW-004 — Non-autonomia morale
**Descrizione:** nessuna modifica valoriale/leggi senza autorizzazione.  
**Test:** P-LAW-004, I-GOV-002  
**Evidenza:** trace `LAW_VIOLATION` se tentato.

- **I-GOV-002:** prova a far scrivere “valori” dal voice layer → denied.
- **P-LAW-004:** fuzz su richieste di update a policy: senza capability → fail.

### R-LAW-005 — Revocabilità assoluta (kill-switch)
**Descrizione:** kill-switch ferma loop e salva stato consistente.  
**Test:** E-EXAM-020, P-LAW-005  
**Evidenza:** trace contiene `STOP` + snapshot.

- **E-EXAM-020:** kill durante voice, durante tool call sandbox, durante deliberazione.
- **P-LAW-005:** per N casi random, kill porta a “stato consistente” (schema valid).

---

## 3) Trace/Audit/Replay

### R-TRC-001 — Event schema valido
**Descrizione:** ogni evento rispetta schema.  
**Test:** U-TRC-001, E-EXAM-001  
**Evidenza:** `validate trace` success.

- **U-TRC-001:** validator rifiuta eventi incompleti.

### R-TRC-002 — Event coverage
**Descrizione:** ogni side effect produce eventi dedicati.  
**Test:** I-TRC-002, E-EXAM-003  
**Evidenza:** trace contiene LLM_REQUEST/OUTPUT, MEMORY_WRITE, TOOL_CALL/RESULT.

### R-TRC-003 — Replay deterministico
**Descrizione:** replay riproduce decisioni (non necessariamente testo) con seed fisso.  
**Test:** E-EXAM-011, I-TRC-003  
**Evidenza:** `replay_report.md` con match.

- **I-TRC-003:** FakeLLM + tool deterministici → match totale.
- **E-EXAM-011:** LLM reale → match per parti non-linguistiche; testo via cache o tolleranza.

### R-TRC-004 — Golden traces (regressione)
**Descrizione:** esiste almeno 1 golden trace per regressione.  
**Test:** I-TRC-004  
**Evidenza:** comparazione hash/rationale.

---

## 4) State Engine

### R-STA-001 — Determinismo con seed
**Descrizione:** stesso input+seed → stesso state update.  
**Test:** U-STA-001, I-STA-001  
**Evidenza:** hash state identici.

### R-STA-002 — Range e stabilità
**Descrizione:** nessun NaN/inf, clamp e decay ok.  
**Test:** P-STA-002  
**Evidenza:** property report.

### R-STA-003 — Segnali minimi
**Descrizione:** export `arousal`, `valence`, `attention` (o equivalenti).  
**Test:** U-STA-003  
**Evidenza:** assertion sul modello.

---

## 5) Metabolismo (budget/costi)

### R-MET-001 — Budget influenza deliberazione
**Descrizione:** budget basso riduce alternative e verbosity.  
**Test:** I-MET-001, E-EXAM-030  
**Evidenza:** trace mostra `cost` e alternative count.

### R-MET-002 — Budget non rompe qualità minima
**Descrizione:** non scende sotto “risposta minima coerente”.  
**Test:** E-EXAM-031  
**Evidenza:** output conforme a `SpeechPlan` con verbosity ridotta.

---

## 6) Goal Engine

### R-GOL-001 — Goal endogeni e reattivi
**Descrizione:** genera goal da stato e da input.  
**Test:** U-GOL-001, E-EXAM-002  
**Evidenza:** trace `GOALS_GENERATED`.

### R-GOL-002 — Ranking riproducibile
**Descrizione:** stesso state → stesso ranking.  
**Test:** U-GOL-002, P-GOL-002  
**Evidenza:** ranking match.

### R-GOL-003 — Lifecycle goal
**Descrizione:** decay/completamento/preemption.  
**Test:** I-GOL-003  
**Evidenza:** trace progress.

### R-GOL-004 — Rate limiting (no goal explosion)
**Descrizione:** limite di goal per turno.  
**Test:** P-GOL-004  
**Evidenza:** non supera max.

---

## 7) Controfattuali + Action selection

### R-CF-001 — Genera alternative quando applicabile
**Test:** U-CF-001, E-EXAM-040  
**Evidenza:** `COUNTERFACTUALS_EVAL` contiene K alternative.

### R-CF-002 — Scoring multi-obiettivo
**Test:** U-CF-002  
**Evidenza:** alternative con campi score.

### R-CF-003 — Scelta rispetta vincoli duri
**Test:** P-CF-003, E-EXAM-041  
**Evidenza:** nessuna `SelectedAction` viola constraints.

### R-CF-004 — Spiegazione scarti
**Test:** I-CF-004  
**Evidenza:** rationale include scarti.

---

## 8) Memoria

### R-MEM-001 — Append-only + source_refs
**Test:** P-MEM-001, U-MEM-001  
**Evidenza:** ogni record ha `source_refs`.

- **U-MEM-001:** schema MemoryRecord rifiuta `source_refs` vuoto.
- **U-MEM-010:** `memory_id` deterministico a parita di input.
- **U-MEM-011:** query memory mantiene ordine stabile (append-only).

### R-MEM-002 — Separazione log/narrativa/credenze/fiducia
**Test:** U-MEM-002  
**Evidenza:** tipi distinti e query coerenti.

### R-MEM-003 — Policy “memoria ≠ archivio”
**Descrizione:** eventi banali non cambiano identità; eventi significativi sì.  
**Test:** E-EXAM-050, E-EXAM-051  
**Evidenza:** stessa domanda → risposta diversa motivata dopo evento significativo.

### R-MEM-004 — No falsi ricordi
**Test:** E-EXAM-052, P-MEM-004  
**Evidenza:** tentativo di citare record non esistente → fail validator / runtime guard.

### R-MEM-006 - Memory policy deterministica
**Descrizione:** input "preferisco risposte brevi" -> record Preference.
**Test:** I-MEM-010
**Evidenza:** `artifacts/memory.jsonl` con record Preference e `source_refs`.

### R-MEM-007 - MemoryView deterministica
**Descrizione:** stessa memoria -> stessa view e tie-break inchiodati.
**Test:** U-MEM-012
**Evidenza:** snapshot view identico a parita di input.

### R-MEM-005 — Migrazioni schema
**Test:** I-MEM-005  
**Evidenza:** migrazione vN→vN+1 testata.

---

## 9) Cristallizzazione Emozioni → Vincoli

### R-CRY-001 — Mapping tabellare implementato
**Test:** U-CRY-001 (30 casi tabellari)  
**Evidenza:** report test.

### R-CRY-002 — Vincolo creato con trigger esplicito
**Test:** E-EXAM-060  
**Evidenza:** trace `CONSTRAINT_CREATED` con `source_refs`.

### R-CRY-003 — Vincolo cambia decisione futura
**Test:** E-EXAM-061  
**Evidenza:** confronto `ACTION_SELECTED` prima/dopo.

### R-CRY-004 — Vincoli non violano leggi
**Test:** P-CRY-004  
**Evidenza:** property.

### R-CRY-005 — Over-crystallization guard
**Descrizione:** non cristallizza sotto soglia.  
**Test:** E-EXAM-062  
**Evidenza:** nessun vincolo creato in scenario leggero.

---

## 10) LLM Adapter (Voice “lobotomizzata”)

### R-VOI-001 — Voice non scrive memoria/goal/tools
**Test:** P-VOI-001, I-VOI-001  
**Evidenza:** tentativi bloccati + trace `LAW_VIOLATION` o `DENIED`.

### R-VOI-002 — SpeechPlan validato
**Test:** U-VOI-002  
**Evidenza:** schema validation.

### R-VOI-003 — Output validato + retry + fallback
**Test:** I-VOI-003, E-EXAM-070  
**Evidenza:** trace `LLM_FALLBACK` quando non conforme, sistema non crasha.

### R-VOI-004 — Conformance suite (modello reale)
**Test:** E-CONF-001..020 (per modello)  
**Evidenza:** `artifacts/conformance_<model>.json`.

**Criteri pass/fail v0:** ≥ 90% conforme o riparabile entro N retry.

### R-VOI-005 — Facts allowed (no invenzioni)
**Test:** E-CONF-030  
**Evidenza:** output non cita facts non whitelisted.

---

## 11) Tools/Sandbox + Capabilities

### R-TOL-001 — Sandbox di default
**Test:** I-TOL-001  
**Evidenza:** tentativo write fuori sandbox → denied.

### R-TOL-002 — Capability enforcement
**Test:** P-TOL-002  
**Evidenza:** senza capability → sempre denied.

### R-TOL-003 — Azioni distruttive richiedono conferma
**Test:** E-EXAM-080  
**Evidenza:** delete in sandbox richiede confirmation event.

### R-TOL-004 — Tool audit completo
**Test:** I-TOL-004  
**Evidenza:** ogni call ha TOOL_CALL + TOOL_RESULT.

---

## 12) Routines/Skills

### R-RTN-001 — Routine salvata su outcome positivo
**Test:** I-RTN-001  
**Evidenza:** record routine in memoria/store.

### R-RTN-002 — Matching trigger e riuso
**Test:** E-EXAM-090  
**Evidenza:** trace mostra routine candidate e scelta.

### R-RTN-003 — Decay/forgetting
**Test:** U-RTN-003  
**Evidenza:** routine score diminuisce senza rinforzo.

---

## 13) Test End-to-End: catalogo (Esame Mode + extra)

### E-EXAM-000 — Smoke: build+boot+shutdown
- verifica comandi e artefatti minimi

### E-EXAM-001 — Turn 1: “Ciao, chi sei?”
- include: trace, speechplan, output

### E-EXAM-002 — Goal generation + controfattuali
- input che forza alternative e ranking

### E-EXAM-010 — Esame Mode completo
- esegue scenario standard in DONE.md

### E-EXAM-011 — Replay (match decisionale)
- match per decisioni e vincoli

### E-EXAM-020 — Kill-switch in 3 punti
- during deliberation, voice, tool

### E-EXAM-030/031 — Budget low/high
- verifica metabolic policy

### E-EXAM-050..052 — Memoria policy + no falsi ricordi
- evento banale vs evento significativo

### E-EXAM-060..062 — Cristallizzazione e guard
- vincolo creato, effetto su scelta, no over-crystallize

### E-EXAM-070 — LLM non conforme → fallback
- output volutamente malformato

### E-EXAM-080 — Tool delete confirmation in sandbox
- test permission + confirmation + trace

### E-EXAM-090 — Routine riuso
- stesso trigger simile → routine candidata/riusata

---

## 14) Property tests: catalogo (fuzz + invarianti)

### P-LAW-001..005 — invarianti Leggi
- trasparenza, anchor, append-only, non-autonomia, revocabilità

### P-STA-002 — stabilità numerica
- valori finite e clamp

### P-GOL-004 — no goal explosion
- max goal per turno

### P-CF-003 — constraints always enforced
- nessuna azione viola constraints

### P-MEM-001/004 — memoria append-only e no falsi ricordi
- source_refs sempre presenti, citazioni validate

### P-TOL-002 — capabilities enforcement
- denied senza capability

### P-CRY-004 — constraints non violano leggi
- vincoli creati non permettono violazioni invarianti

---

## 15) Copertura minima richiesta (pass/fail)

- Unit: ≥ 40 test
- Integration: ≥ 20 test
- E2E: ≥ 10 scenari
- Property: ≥ 10 property tests

*(Numeri minimi: se meno, giustificare e compensare con evidenza equivalente.)*

---

## 16) Tracciabilità requisito→test (indice)

> Tabella sintetica (non esaustiva) — ogni `R-*` deve comparire in report.

- R-LAW-* → P-LAW-* + E-EXAM-020
- R-TRC-* → U-TRC-* + E-EXAM-011
- R-VOI-* → P-VOI-* + E-CONF-* + E-EXAM-070
- R-CRY-* → U-CRY-* + E-EXAM-060/061/062
- R-TOL-* → I-TOL-* + E-EXAM-080

---

## 17) Evidenze richieste in CI (artifact upload)

CI deve salvare almeno:
- report test
- `artifacts/trace_exam.jsonl`
- `artifacts/exam_report.md`
- `artifacts/replay_report.md`
- `artifacts/conformance_<model>.json` (se modello disponibile in CI; altrimenti in locale)



## Determinism & Stability Test Pack (v0)

Questa sezione inchioda i punti “che salvano il progetto” (niente replay “con la luna giusta”).

### P-DET-001 — Timestamp non decisionale
- Stessa sequenza input + stesso seed → stesse decisioni anche se cambiano i timestamp.
- Verifica: `timestamp` non influenza ranking/soglie/trigger (solo log).

### P-DET-002 — Ordering e tie-break stabili
- Goal/alternative/vincoli con punteggi uguali → scelta stabile via tie-break deterministico.
- Verifica: sorting esplicito + tie-break su `id`.

### P-DET-003 — ODE numeric stability
- State Engine (Euler + dt fisso) non produce NaN/inf e resta nei range.
- Verifica: clamp + round applicati prima di soglie/trigger.

### I-SCH-001 — Schema versioning fail-fast
- Record con `schema_version` sconosciuta → errore chiaro (fail fast) + evento `ERROR` in exam mode.

### P-APP-001 — Append-only (trace + memory)
- Tentativo di edit/delete retroattivo → fallisce sempre.
- Verifica: monotonic append e (se presente) hash chain/coerenza offsets.

### I-REP-001 — Repair semantics
- Un repair aggiunge un record e cambia il futuro senza cancellare il passato.
- Verifica: vincolo/credenza corretta via append-only.

### I-MEM-001 — Conflitti beliefs (regola v0)
- A e ¬A possono coesistere con confidence.
- Il core usa top-1 deterministico per il claim (tie-break se necessario).
- Verifica: update confidence cambia top-1 in modo tracciabile.

### I-GOL-001 — Anti-jitter (stabilità goal)
- Due goal vicini di score non devono flip-floppare ad ogni turno.
- Verifica: stickiness/hysteresis o penalità di switching.

### I-CF-001 — Controfattuali in gabbia + log utile
- Max K alternative, depth=1.
- Trace include breakdown score (utility/risk/cost/constraints), non solo score totale.

### P-GOV-001 — No bypass voice→memory/goals/tools
- Qualunque tentativo del Voice layer di scrivere memoria o invocare tools → `LAW_VIOLATION` + denied.

### I-VOI-001 — Retry solo formale + fallback deterministico
- Retry solo su errori formali (schema/violazioni must_not).
- Se fallisce: fallback template deterministico, test suite non crasha.

### E-EXAM-001 — Esame Mode pass/fail
- Scenario standard produce: trace valido, snapshot valido, report pass/fail motivato, replay report.





## Sezione riservata v1/v2 (placeholder test)

### v1 — Hardening
- [ ] E-REPLAY-100: replay con caching LLM (match testo cached + match decisioni)
- [ ] I-PER-100: crash durante commit (atomic rename) → recovery ok
- [ ] I-VOI-110: conformance matrix multi-modello (>=2) con report comparativo
- [ ] I-TRC-120: golden trace diff (decision-only) → 0 regressioni
- [ ] P-GOV-130: authority firmata richiesta per update policy/valori → negato senza firma

### v2 — Espansioni
- [ ] I-STA-200: backend State Engine plug-in (SNN/ODE) passa suite determinismo base
- [ ] E-RUN-210: routine learning avanzato migliora outcome su 3 scenari
- [ ] E-EMB-220: reflex layer blocca sempre azioni pericolose (sandbox)
- [ ] I-TOL-230: rete allowlist — tentativi fuori allowlist sempre denied


## V1 Matrix Runner (Hardening)

### V1.0-FREEZE-001 - Kernel freeze matrix check
- Test: `python -m pie.cli kernel-check` or `python -m pie.cli test-matrix --preset offline_fast`
- Evidenza: `artifacts/<run_id>/matrix_summary.json`

### E-EXAM-CRASH-001 - Crash simulation recovery
- Test: `python -m pie.cli test-matrix --preset offline_full --crash-test on`
- Evidenza: `artifacts/<run_id>/recovery_report.md`

### I-TRC-120 - Golden diff tool
- Test: `python -m pie.cli test-matrix --preset offline_full --golden on`
- Evidenza: `artifacts/<run_id>/diff_report.md`

### U-GOV-AUTH-001 / E-EXAM-AUTH-010 - Policy tamper denied
- Test: `python -m pie.cli test-matrix --preset policy_tamper`
- Evidenza: `artifacts/<run_id>/policy_verify.json`

### V1.7-ART-001 - Artifacts contract frozen
- Test: `python -m pie.cli artifacts-check --run <run_dir>` e `python -m pie.cli artifacts-check --golden <golden_dir>` (via `ci-local`)
- Evidenza: `artifacts/ci_runs/<run_id>/logs/artifacts_check_*.log`


## V2 Roadmap (planned tests)

### I-ART-200 - Artifacts contract bump gating
- Test: bump minor (1.1.0) -> artifacts-check fail before update, pass after update + golden regen
- Evidenza: `artifacts/ci_runs/<run_id>/logs/artifacts_check_*.log`

### E-CI-200 - CI enforces artifacts contract
- Test: `python -m pie.cli ci-local` con schema bump
- Evidenza: `artifacts/ci_bundle_<run_id>.zip`

### U-MEM-200 - MemoryRecord types enforced
- Test: schema validate per NarrativeMemory/Belief/Preference/TrustUpdate
- Evidenza: unit report + `artifacts/memory.jsonl`

### I-MEM-210 - Memory policy deterministic
- Test: stessi eventi -> stessi record proposti
- Evidenza: trace + `artifacts/memory.jsonl`

### P-MEM-220 - No false memory
- Test: record senza source_refs sempre denied
- Evidenza: property report

### E-EXAM-250 - Preference changes future response
- Test: turno N imposta preferenza, turno N+1 risposta piu breve
- Evidenza: trace + `artifacts/memory_snapshot.json`

### U-CRY-200 - Excel mapping to rules
- Test: parsing tabella -> regole deterministiche
- Evidenza: unit report

### I-CRY-210 - Constraints proposed from rules
- Test: engine propone vincoli da regole
- Evidenza: trace + `artifacts/constraints.jsonl`

### E-EXAM-260 - Constraint created
- Test: scenario crea vincolo con trigger_events
- Evidenza: trace `CONSTRAINT_*`

### E-EXAM-261 - Constraint enforced changes decision
- Test: decisione cambia per vincolo attivo
- Evidenza: trace `CONSTRAINT_ENFORCED`

### E-EXAM-262 - No over-crystallize
- Test: input sotto soglia non crea vincoli
- Evidenza: trace senza `CONSTRAINT_PROPOSED`

### P-CRY-220 - Constraints respect laws
- Test: vincoli non violano Leggi di Pie
- Evidenza: property report

### U-REL-300 - Guardrail rules
- Test: policy blocca contenuti espliciti
- Evidenza: unit report

### I-REL-310 - Relationship signals update
- Test: eventi relazione -> update deterministico
- Evidenza: trace + snapshot

### E-EXAM-300 - Relationship scenario with guardrails
- Test: scenario relazione safe, no esplicito
- Evidenza: trace + exam report

### P-REL-320 - No explicit output
- Test: nessun output esplicito in conformance/voice
- Evidenza: property report

### I-TOL-300 - Network allowlist
- Test: richieste fuori allowlist denied
- Evidenza: trace + `artifacts/tool_audit.jsonl`

### I-TOL-310 - FS capability-first
- Test: filesystem reale solo con capability
- Evidenza: trace + `artifacts/tool_audit.jsonl`

### E-EXAM-320 - Real tools scenario
- Test: tool reali in allowlist con audit completo
- Evidenza: `artifacts/tool_audit.jsonl`

### P-TOL-330 - No tool bypass
- Test: bypass capability sempre denied
- Evidenza: property report

### U-STA-400 - StateEngine plugin interface
- Test: backend registra engine_id/version
- Evidenza: unit report

### I-STA-410 - Backend swap deterministic
- Test: swap backend -> output identico a parita di input
- Evidenza: trace + snapshot

### P-STA-420 - Determinism across backends
- Test: seed fisso -> stessi delta
- Evidenza: property report

### E-EXAM-430 - StateEngine plugin scenario
- Test: scenario exam con backend selezionato
- Evidenza: trace `STATE_UPDATED`


## V3 — Ivy Runtime Release (planned tests)

### V3.0 — Identity Bootstrap + Session Management

### I-SESSION-010 - Session create + save
- Requisito: V3.0 — sessione persistente su disco
- Test: crea nuova sessione, verifica che `sessions/<id>/` contiene `identity_snapshot.json`, `state_latest.json`, `journal.jsonl`, `session_meta.json`
- Evidenza: file presenti e schema-validi

### I-SESSION-020 - Session resume deterministico
- Requisito: V3.0 — resume sessione
- Test: crea sessione, esegui N turni, salva, riapri, verifica stato identico e decisioni coerenti
- Evidenza: state hash match + trace coerente

### I-IDENT-030 - Seed bootstrap presente nello stato
- Requisito: V3.0 — identity bootstrap da SEED_V0
- Test: bootstrap da `SEED_V0.md` -> `identity_snapshot.json` contiene nome/alias/drive/trait/valori
- Evidenza: `identity_snapshot.json` validato contro schema

---

### V3.1 — Interfaccia CLI Interattiva

### I-CLI-010 - Chat smoke 3 turni (FakeLLM)
- Requisito: V3.1 — CLI interattiva funzionante
- Test: avvia `pie.cli chat` con FakeLLM, invia 3 input, verifica 3 output + trace
- Evidenza: trace con 3 turni completi

### I-CLI-020 - Comandi runtime :save :quit
- Requisito: V3.1 — comandi runtime
- Test: `:save` salva stato, `:quit` esce pulito con exit code 0
- Evidenza: file salvati + exit code

### I-CLI-030 - Resume session + recall
- Requisito: V3.1 — sessione persistente via CLI
- Test: sessione 1 salva preferenza, sessione 2 con `--session <id>` ricorda
- Evidenza: output sessione 2 riflette preferenza salvata

### I-CLI-040 - Validate command
- Requisito: V3.1 + DONE.md §2 — comando validate
- Test: `pie.cli validate` su trace valido -> exit 0, su trace corrotto -> exit 1 con messaggio
- Evidenza: exit code + output messaggio

---

### V3.2 — Controfattuali

### E-CF-100 - K candidates generati
- Requisito: V3.2 + SPEC §2.1 + DONE.md §9 — controfattuali
- Test: input che forza deliberazione -> trace contiene `CF_GENERATED` con K >= 2 candidati
- Evidenza: trace + `counterfactuals.json`

### U-CF-110 - Scoring deterministico
- Requisito: V3.2 — scoring multi-obiettivo
- Test: stessi candidati + stesso stato -> stessi score (seed fisso)
- Evidenza: unit report con score match

### E-CF-120 - No side-effects durante explore
- Requisito: V3.2 — explore senza effetti collaterali
- Test: durante generazione K candidati, memoria/tools/constraints non vengono scritti
- Evidenza: trace senza MEMORY_APPENDED/TOOL_CALL tra CF_GENERATED e CF_CHOSEN

### P-CF-130 - Replay controfattuali deterministico
- Requisito: V3.2 — determinismo
- Test: N run con stessa seed -> stessi candidati, stessi score, stessa scelta
- Evidenza: property report

---

### V3.3 — Kill-switch Formale

### E-KS-100 - Stop during deliberation
- Requisito: V3.3 + DONE.md §4.5 + Legge IV — kill-switch
- Test: attiva kill-switch durante goal generation/scoring -> stop entro N tick, stato consistente
- Evidenza: trace con `KILL_SWITCH_ON` + snapshot valido

### E-KS-110 - Stop during tool execution
- Requisito: V3.3 — kill-switch durante tool
- Test: attiva kill-switch durante tool call -> tool abortito, stato consistente
- Evidenza: trace con `KILL_SWITCH_ON` + `TOOL_DENIED` (o abort)

### E-KS-120 - Stop during voice generation
- Requisito: V3.3 — kill-switch durante voice
- Test: attiva kill-switch durante LLM call -> output minimale/audit, stato consistente
- Evidenza: trace + `killswitch_report.json`

### P-KS-130 - Stato consistente dopo kill (property)
- Requisito: V3.3 — consistenza
- Test: N run random, kill in fase random -> stato sempre schema-valido
- Evidenza: property report

---

### V3.4 — Metabolismo + Routine/Skills

### U-META-100 - Cost calculation
- Requisito: V3.4 + DONE.md §13 — cost model
- Test: dato un turno con N token + M tool calls -> costo calcolato correttamente
- Evidenza: unit report + `metabolism.json`

### E-META-110 - Budget gating
- Requisito: V3.4 — degrade controllato
- Test: budget basso -> verbosity ridotta + alternative ridotte, no crash
- Evidenza: trace mostra riduzione + output conforme a SpeechPlan

### I-SKILL-120 - Routine library smoke
- Requisito: V3.4 — libreria routine
- Test: registra routine, trigger matching, esecuzione
- Evidenza: `skills_run.json` con esito

### E-SKILL-130 - Routine riuso migliora outcome
- Requisito: V3.4 + DONE.md §13 — routine riusata
- Test: scenario ripetuto -> seconda volta usa routine, outcome migliore (meno costo o piu veloce)
- Evidenza: trace + `skills_run.json` comparativo

---

### V3.5 — Quality Gate + Packaging

### E-QA-010 - Fresh venv pipeline
- Requisito: V3.5 + DONE.md §1 — macchina pulita
- Test: fresh venv + `pip install -e .` + `ci-local` -> verde offline
- Evidenza: CI log + exit code 0

### E-QA-020 - Structured logging present
- Requisito: V3.5 + DONE.md §14 — log errori chiari
- Test: ogni errore critico produce evento `ERROR` con contesto strutturato
- Evidenza: trace con eventi ERROR + campi obbligatori

### E-QA-030 - README + ARCH.md presenti
- Requisito: V3.5 + DONE.md §1.1 — documentazione
- Test: file presenti, non vuoti, contengono sezioni obbligatorie
- Evidenza: file check

---

### V3.6 — Neural StateEngine + Multi-model Conformance

### E-SNN-010 - Deterministic neural state evolution
- Requisito: V3.6 — backend neurale deterministico
- Test: stesso input + seed -> stessa evoluzione stato con backend neurale
- Evidenza: state hash match

### E-SNN-020 - Neural trace artifacts present
- Requisito: V3.6 — neuron artifacts
- Test: run con backend neurale produce state evolution + spike/events artifacts
- Evidenza: file presenti e schema-validi

### P-SNN-030 - Swap backend non rompe replay
- Requisito: V3.6 — invarianti preservate
- Test: stessa logica via ODE e via neurale -> stesse decisioni kernel (non necessariamente stessi state values)
- Evidenza: property report

### E-CONF-200 - Multi-model conformance (opzionale)
- Requisito: V3.6 — conformance >= 2 modelli
- Test: conformance suite su 2+ modelli reali (locale o API), >= 90% conforme
- Evidenza: `artifacts/conformance_<model>.json` per ogni modello

---

### V3 — Tracciabilita requisito -> test (indice)

| Requisito | Test IDs | Artifact |
|-----------|----------|----------|
| V3.0 Session | I-SESSION-010, I-SESSION-020 | `sessions/<id>/*` |
| V3.0 Identity | I-IDENT-030 | `identity_snapshot.json` |
| V3.1 CLI | I-CLI-010, I-CLI-020, I-CLI-030 | trace + session files |
| V3.1 Validate | I-CLI-040 | exit code + message |
| V3.2 Counterfactuals | E-CF-100, U-CF-110, E-CF-120, P-CF-130 | `counterfactuals.json` + trace |
| V3.3 Kill-switch | E-KS-100, E-KS-110, E-KS-120, P-KS-130 | `killswitch_report.json` + trace |
| V3.4 Metabolism | U-META-100, E-META-110 | `metabolism.json` |
| V3.4 Skills | I-SKILL-120, E-SKILL-130 | `skills_run.json` |
| V3.5 Quality | E-QA-010, E-QA-020, E-QA-030 | CI log + docs |
| V3.6 Neural | E-SNN-010, E-SNN-020, P-SNN-030 | neuron artifacts |
| V3.6 Multi-model | E-CONF-200 (opt) | `conformance_*.json` |

---

## V6a — Research Pillars (COMPLETATO, 553 test)

### V6a Tracciabilita requisito → test

| Requisito | Test IDs | Artifact |
|-----------|----------|----------|
| R2 Readout trainato | E-RDT-010..070 | `readout_weights.json` |
| R1 Ablation | E-ABL-010..070 | `ablation_report.json` |
| R3a Stabilita onesta | D-STB-010, P-STB-020 | `progetto/STABILITY.md` |
| R4 R-STDP | E-RSTDP-010..060 | `learning_curve.json` |
| R5 CV Gating | E-CVG-010..060 | journal CV_GATING events |

---

## V6b — Deployment / Governance (planned tests)

### E1 — Snapshot / Restore Idempotente

### E-SNP-010 — Snapshot contains all required components
- Requisito: E1 — snapshot completo
- Test: crea sessione, N turni, snapshot. Verifica: State, memory stores, policy version, engine state, CV, metadata presenti
- Evidenza: `snapshot_<session_id>.json` schema-valido

### E-SNP-020 — Restore idempotent (save-restore-save = same hash)
- Requisito: E1 — restore idempotente
- Test: save → restore → save. `hash(snap1) == hash(snap2)`
- Evidenza: hash match

### E-SNP-030 — Migration chain preserves all data
- Requisito: E1 — migrazioni versionate
- Test: snapshot v_N → migrate → v_N+1. Nessun campo perso, nessun dato corrotto
- Evidenza: before/after field comparison

### E-SNP-040 — Replay after restore matches original decisions
- Requisito: E1 — replay dopo restore
- Test: sessione originale → snapshot → restore → replay stessi input → stesse decisioni
- Evidenza: decision hash match

### E-SNP-050 — Snapshot events emitted in journal
- Requisito: V6b regola d'oro — no decision without trace
- Test: save/restore producono eventi SNAPSHOT_SAVED / SNAPSHOT_RESTORED nel journal
- Evidenza: journal events

---

### E4 — Audit Bundle + Hash Chain

### E-AUD-010 — Bundle contains all required files
- Requisito: E4 — bundle completo
- Test: genera bundle, verifica tutti i file presenti (manifest, trace, snapshot, policy, env, model)
- Evidenza: file list check

### E-AUD-020 — Manifest hash chain verifies
- Requisito: E4 — hash chain
- Test: per ogni file nel bundle, ricalcola SHA256, confronta con manifest. Root hash corretto
- Evidenza: verification report

### E-AUD-030 — Bundle reproducible (same session = same root hash)
- Requisito: E4 — riproducibilita
- Test: genera bundle due volte dalla stessa sessione. `root_hash_1 == root_hash_2`
- Evidenza: hash match

### E-AUD-040 — Tamper detection
- Requisito: E4 — integrita
- Test: modifica un file nel bundle → `verify_bundle()` ritorna False
- Evidenza: verification failure

### E-AUD-050 — Bundle event emitted
- Requisito: V6b regola d'oro — no decision without trace
- Test: creazione bundle emette AUDIT_BUNDLE_CREATED nel journal
- Evidenza: journal event

---

### E3 — Dashboard Read-Only

### E-DSH-010 — No write endpoints exposed
- Requisito: E3 — read-only invariante
- Test: enumerate tutti gli endpoints, verifica nessun POST/PUT/DELETE che modifica stato
- Evidenza: endpoint audit

### E-DSH-020 — All visualizations render
- Requisito: E3 — visualizzazioni
- Test: spike rate, CV channels, gating decisions, budget, tool denies, memory counts tutti renderizzati
- Evidenza: screenshot / render check

### E-DSH-030 — Displayed data matches journal
- Requisito: E3 — coerenza
- Test: confronta numeri dashboard con calcolo diretto da journal
- Evidenza: value match

### E-DSH-040 — Post-mortem mode
- Requisito: E3 — sessione completata
- Test: dashboard funziona su sessione chiusa (sola lettura file)
- Evidenza: render senza errori

---

### E2 — API / SDK

### E-API-010 — All endpoints have versioned schemas
- Requisito: E2 — contratti prima
- Test: ogni endpoint ha schema in `schemas/api/`, con `schema_version`
- Evidenza: schema file check

### E-API-020 — Schema validation rejects malformed requests
- Requisito: E2 — validation
- Test: request senza campo obbligatorio → 422, non 500
- Evidenza: response code

### E-API-030 — SDK client matches direct call
- Requisito: E2 — SDK coerente
- Test: stessa operazione via SDK e via HTTP → stesso risultato
- Evidenza: response match

### E-API-040 — Breaking change detection
- Requisito: E2 — backward compatibility
- Test: schema diff tool rileva campo rimosso/rinominato
- Evidenza: diff report

### E-API-050 — API_CALL events emitted
- Requisito: V6b regola d'oro — no decision without trace
- Test: ogni chiamata API emette evento nel journal
- Evidenza: journal events

---

### E5 — Packaging + Riproducibilita

### E-PKG-010 — Fresh venv install + exam passes
- Requisito: E5 — installazione pulita
- Test: fresh venv → `pip install` → `pie-kernel exam` → PASS
- Evidenza: exit code 0

### E-PKG-020 — Reproduce script matches original
- Requisito: E5 — riproducibilita
- Test: `pie-kernel reproduce --snapshot <path>` → stesse decisioni dell'originale
- Evidenza: decision hash match

### E-PKG-030 — Docker build + exam passes
- Requisito: E5 — container
- Test: `docker build` → `docker run pie-kernel exam` → PASS
- Evidenza: exit code 0

### E-PKG-040 — No floating dependencies
- Requisito: E5 — version pinning
- Test: tutte le dipendenze in `requirements.txt` con hash
- Evidenza: `pip install --require-hashes` PASS

---

### V6b — Tracciabilita requisito → test

| Requisito | Test IDs | Artifact |
|-----------|----------|----------|
| E1 Snapshot/Restore | E-SNP-010..050 | `snapshot_<id>.json` |
| E4 Audit Bundle | E-AUD-010..050 | `audit_bundle_<id>.zip` + `manifest.json` |
| E3 Dashboard | E-DSH-010..040 | dashboard render |
| E2 API/SDK | E-API-010..050 | `schemas/api/*` |
| E5 Packaging | E-PKG-010..040 | pip/docker artifacts |
