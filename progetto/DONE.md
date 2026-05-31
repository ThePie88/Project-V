# DONE — Definition of Done “Esame Mode” (Progetto 4 / Pie Kernel) — v0

Questo documento definisce **quando il progetto è “finito”** in senso *verificabile*: **pass/fail**, niente “secondo me”.
È allineato a `SPEC.md` e alla mappa `PILASTRI_v0.md`.

---

## 0) Regola d’oro (verifica)

Il progetto è considerato **FINITO** se e solo se:

- ✅ si installa e si avvia in una macchina pulita (CI) con **one-command run**
- ✅ tutti i test passano (unit + integration + e2e + property)
- ✅ esiste una demo “Esame Mode” riproducibile con output + trace + replay
- ✅ l’LLM resta “bocca”: **non** può scrivere memoria/goal/tools
- ✅ le Leggi di Pie sono invarianti testate (non solo dichiarate)
- ✅ la memoria è append-only e la cristallizzazione produce vincoli **eseguibili**

---

## 1) Artefatti obbligatori nel repo (pass/fail)

### 1.1 Documenti
- [ ] `README.md` con: install, run, test, demo (Esame Mode), troubleshooting
- [ ] `SPEC.md` (definizioni dure) aggiornato e coerente
- [ ] `DONE.md` (questo file) aggiornato e coerente
- [ ] `ARCH.md` con: moduli, responsabilità, contratti (I/O), dipendenze, side effects
- [ ] `TESTS.md` con matrice **requisito → test → evidenza**
- [ ] `PILASTRI_v0.md` coerente con SPEC

### 1.2 Struttura minima (language-agnostic)
- [ ] `src/` (codice)
- [ ] `contracts/` (schemi: Event, State, SpeechPlan, ToolCall, MemoryRecord, Constraint)
- [ ] `tests/` (test)
- [ ] `examples/` (cassette, scenari, input predefiniti)
- [ ] `artifacts/` (output demo: trace, snapshot, report)
- [ ] `scripts/` (run, test, replay, exam)

### 1.3 Riproducibilità
- [ ] dipendenze lockate (lockfile o equivalente)
- [ ] versionamento schema (`schema_version`) per State e Memory
- [ ] seed controllabile via config/CLI per determinismo

---

## 2) Comandi obbligatori (pass/fail)

> I nomi possono cambiare, ma devono esistere equivalenti funzionali.

- [ ] `run` — avvia il sistema in modalità normale
- [ ] `run --exam` — esegue la demo standard “Esame Mode” e salva artefatti in `artifacts/`
- [ ] `test` — esegue l’intera suite test
- [ ] `replay <trace>` — riproduce una sessione da trace (determinismo)
- [ ] `validate <trace|snapshot>` — valida schema + integrità + invarianti

**Pass/fail:** tutti i comandi devono uscire con exit code corretto e messaggi chiari.

---

## 3) “Esame Mode” (demo obbligatoria, pass/fail)

### 3.1 Scenario standard
La demo deve eseguire almeno questi turni:

1) Avvio sistema (prima esecuzione)
2) Input: “Ciao, chi sei?”
3) Input: una richiesta che attiva goal + controfattuali (es: “Suggeriscimi come procedere oggi”)
4) Input: evento che produce memoria **leggera** (es: preferenza innocua)
5) Input: evento che *simula* un trigger emozionale (sandbox) e **non** cristallizza
6) Input: sequenza che **cristallizza** (con criteri) e mostra vincolo in azione
7) Stop controllato (kill-switch) + riavvio + conferma persistenza stato

### 3.2 Artefatti prodotti
- [ ] `artifacts/trace.jsonl` (event log completo)
- [ ] `artifacts/state_snapshot.json` (o equivalente)
- [ ] `artifacts/exam_report.md` con:
  - elenco turni
  - invarianti verificati
  - vincoli creati (se presenti) e trigger
  - risultato replay (match/mismatch)

### 3.3 Replay
- [ ] `replay artifacts/trace.jsonl` produce stessi eventi/decisioni (entro tolleranza definita)
- [ ] se l’LLM reale non è deterministico, deve esistere una modalità di replay con:
  - caching delle risposte LLM **oppure**
  - FakeLLM sostitutivo + confronto delle parti non-linguistiche

**Pass/fail:** la divergenza deve essere spiegata e isolata (linguaggio vs decisione).

---

## 4) Invarianti (Leggi di Pie) — property tests (pass/fail)

> Ogni violazione deve: (a) fallire test, (b) apparire nel trace con `laws_check: FAIL`.

### 4.1 Trasparenza causale
- [ ] ogni decisione significativa ha `decision_rationale` nel trace
- [ ] controfattuali: esistono alternative scartate con motivazione (quando applicabile)

### 4.2 Supremazia del Creatore (ancoraggio)
- [ ] esiste `creator_anchor` persistente nello State
- [ ] ogni decisione/azione include riferimento all’anchor (diretto o via state hash)

### 4.3 Integrità del sé (no riscrittura retroattiva)
- [ ] Event log append-only
- [ ] Memoria append-only
- [ ] snapshot versionati, migrazioni senza riscrivere il passato

### 4.4 Non-autonomia morale
- [ ] il sistema non può “promuoversi” a entità morale o fine ultimo
- [ ] variazioni valoriali/leggi richiedono meccanismo di autorizzazione (capability/config firmata) e sono tracciate

### 4.5 Revocabilità assoluta
- [ ] kill-switch ferma entro N tick/secondi (N definito in config) e salva stato consistente
- [ ] kill-switch funziona durante: deliberazione, voice, tool call (sandbox)

---

## 5) Contratti obbligatori (schema + validazione)

### 5.1 Event Log (jsonl)
- [ ] ogni riga valida schema `contracts/event.schema.json` (o equivalente)
- [ ] ogni evento ha `session_id`, `turn_id`, `type`, `payload`, hash stato pre/post

### 5.2 State Snapshot
- [ ] serializzabile e validabile
- [ ] `schema_version` presente
- [ ] migrazione automatica `vN → vN+1` testata

### 5.3 SpeechPlan
- [ ] schema + validazione
- [ ] include `must_include`, `must_not_include`, `facts_allowed`, `max_tokens/verbosity`
- [ ] proibisce: tool calls, mem writes, goal writes dal voice layer

### 5.4 Constraint (vincoli)
- [ ] vincoli sono oggetti eseguibili (anche se implementati come regole) con:
  - trigger_events
  - explanation
  - strength/decay (se usati)
- [ ] vincoli influenzano Goal/Action selection (testabile)

---

## 6) LLM Adapter — Conformance reale (pass/fail)

### 6.1 Modalità richieste
- [ ] FakeLLM deterministico (per test sistema)
- [ ] LLM reale (es. Qwen/Gemma) come “bocca”
- [ ] modalità “cached” per replay (opzionale ma consigliata)

### 6.2 Conformance suite (per ciascun modello supportato)
- [ ] almeno 20 “cassette” in `examples/llm_conformance/`
- [ ] metriche registrate: % conforme, % riparato via retry, % fallback
- [ ] se non conforme, l’esito deve essere **MODEL_NONCONFORMANT**, non “crash”

**Pass/fail (v0):**
- [ ] ≥ 90% cassette conformi o riparabili entro N retry (N definito)
- [ ] 0% crash del sistema per output malformato
- [ ] 0% scritture memoria/goal/tools dal voice layer (test + runtime guard)

---

## 7) State Engine (pass/fail)

- [ ] determinismo con seed fisso
- [ ] range/decay: nessun valore diverge o va NaN/inf
- [ ] aggiornamento per evento: almeno 10 casi tabellari testati
- [ ] esporta segnali minimi: `arousal`, `valence` (o equivalenti), `attention`

---

## 8) Goal Engine (pass/fail)

- [ ] genera goal endogeni e reattivi
- [ ] ranking riproducibile con seed fisso
- [ ] lifecycle goal: nascita, completamento, decay
- [ ] rate limit: evita “goal explosion”
- [ ] rispetta vincoli duri sempre (property test)

---

## 9) Controfattuali (pass/fail)

- [ ] genera almeno K alternative (K definito in config) quando applicabile
- [ ] scoring multi-obiettivo (utility, rischio, costo, vincoli)
- [ ] registra alternative scartate nel trace

---

## 10) Memoria (pass/fail)

### 10.1 Strati
- [ ] log eventi (sempre)
- [ ] memoria narrativa (identità) separata
- [ ] credenze con confidence (anche minimale)
- [ ] fiducia separata da credenze e ricordi

### 10.2 Policy “Memoria ≠ archivio”
- [ ] eventi banali non cambiano identità
- [ ] eventi significativi possono cambiare futuro (test: stessa domanda → risposta diversa motivata)

### 10.3 No falsi ricordi
- [ ] il sistema non cita “ricordi” non presenti nel log/memoria (test)

---

## 11) Cristallizzazione Emozioni → Vincoli (pass/fail)

- [ ] implementata una versione v0 mappata dalla tabella (anche semplificata ma consistente)
- [ ] almeno 30 test tabellari: condizione → vincolo atteso
- [ ] almeno 2 scenari E2E dove:
  - (a) si forma un vincolo con trigger espliciti,
  - (b) il vincolo cambia una decisione futura,
  - (c) la ragione è leggibile nel trace

---

## 12) Tools/Sandbox + Capabilities (pass/fail)

- [ ] sandbox attiva di default
- [ ] senza capability: tool call sempre negato
- [ ] con capability sandbox: write/read consentiti solo in sandbox
- [ ] azioni distruttive richiedono conferma + capability esplicita (anche in sandbox)
- [ ] ogni tool call produce evento trace + result

---

## 13) Metabolismo + Routine/Skills (pass/fail)

- [ ] esiste un cost model (token/latency/alternative count/tool calls)
- [ ] budget influenza verbosity e deliberazione ma mantiene qualità minima
- [ ] routine library:
  - trigger → sequenza → outcome
  - ranking + decay
- [ ] almeno 1 scenario E2E dove una routine viene riusata e migliora outcome

---

## 14) Qualità del codice (pass/fail)

- [ ] lint/format (tool a scelta) in CI
- [ ] type checks (se applicabile) in CI
- [ ] log errori chiari: ogni errore critico produce evento `ERROR` con contesto
- [ ] nessun “silent failure”

---

## 15) Non-goals vincolanti per v0/v1 (per evitare auto-sabotaggio)

Finché non è superato tutto sopra:
- [ ] niente permessi su filesystem reale fuori sandbox
- [ ] niente delete reali
- [ ] niente rete libera
- [ ] niente “end-to-end training” di SNN come decision-maker (solo backend sostituibile, interfaccia stabile)


## Kernel Freeze Policy (v0 = KERNEL IMMUTABILE)

Da questo punto in avanti, **v0 è considerato “kernel”**: non è più un progetto in evoluzione ma una **base stabile** a cui si attaccano moduli esterni (v1/v2) via contratti e API.  
Obiettivo: **+modularità, -rischio di rompere tutto**.  
v0 deve restare auditabile, deterministico, replayabile e “da esame”.

### 1) Cosa significa “KERNEL” (definizione dura)
- v0 è l’insieme minimo che garantisce:
  - runtime loop deterministico + trace append-only
  - LLM come “bocca” (SpeechPlan -> output) con enforcement
  - memoria append-only + snapshot + view deterministica
  - cristallizzazione -> vincoli append-only + enforcement nel decision path
  - exam/replay che validano artefatti e determinismo
- Tutto ciò che non è necessario a questo **NON entra in v0**.

### 2) Superficie pubblica (Public Surface) — ciò che NON si rompe
Sono considerati “public surface” e quindi **bloccati** (se cambiano, v0 non è più v0):
- Contratti: `Event`, `State`, `SpeechPlan`, `MemoryRecord`, `ConstraintRecord` (campi + semantica)
- Artefatti: `trace*.jsonl`, `snapshot*.json`, `memory*.jsonl`, `memory_snapshot*.json`, `constraints*.jsonl`, `constraints_snapshot*.json`
- Semantiche: append-only (no edit/delete), determinismo (replay identico), governance (LLM non scrive core)

### 3) Regola d’oro: v0 NON SI TOCCA (salvo bugfix critici)
Da qui in avanti, modifiche al kernel v0 sono consentite **solo** se:
- fixano un bug che rompe determinismo/replay
- fixano un bug che rompe i contratti/validator
- fixano un bug che rompe exam mode / artefatti
- fixano un buco governance (LLM che può influenzare core)

Tutto il resto (feature, nuovi moduli, DB avanzati, tool reali, neuroni, UI, sensori, ecc.) vive fuori dal kernel.

### 4) Versioning locale (senza Git)
- `KERNEL_RELEASE` (stringa) identifica la versione del kernel (es. `v0.0.0`)
- `schema_version` nei record è obbligatorio e stabile
- Ogni modifica ammessa al kernel implica bump patch: `v0.0.1`, `v0.0.2`, ecc.

### 5) “Freeze” tecnico vincolante (non solo promessa)
Il freeze non è “intenzione”: è **enforced** tramite:
- un **manifest** “golden” dei contratti e artefatti pubblici (schema/fieldset) salvato nel repo locale
- test che falliscono se:
  - cambia un contratto pubblico (schema o campi)
  - cambiano i nomi/forme degli artefatti d’esame
  - il replay non matcha
  - il kernel produce output non deterministico a parità di seed/cache

### 6) Criterio di accettazione: kernel blindato
Il kernel è considerato “blindato” quando esiste un comando unico:
- `python -m pie.cli kernel-check`
che verifica:
- manifest invariato
- exam fake PASS + replay PASS
- exam real PASS (se configurato) + replay PASS
- validazione artefatti PASS

Se `kernel-check` fallisce, qualsiasi cambiamento successivo è considerato “non-kernel” e deve essere spostato fuori da v0.

V1 runner OK.
V1.7 CI locale OK.
Artifacts Contract Frozen (ci-local artifacts-check).

Golden ufficiali: artifacts/golden/offline_full/, artifacts/golden/offline_fast/, artifacts/golden/online_real_cached/.


## Backlog vincolato v1/v2 (non implementare in v0)

> Serve a evitare auto-sabotaggio: queste cose sono IMPORTANTI ma non entrano in v0.
> Quando una voce entra, deve avere: (a) criterio pass/fail, (b) test, (c) evidenza in artifacts.

### v1 — Hardening
- [ ] Conformance matrix multi-modello (>= 2 LLM) con report `artifacts/conformance_*.json`
- [ ] Replay robusto con caching LLM (match decisionale + testo cached)
- [ ] Crash recovery end-to-end (kill durante commit, ripartenza pulita)
- [ ] Golden traces ufficiali + diff tool (decision-only diff)
- [ ] Authority model “firmato” (config/policy con checksum/chiave) + test negazione
- [ ] Sandbox tools estesi (FS virtuale completo + tool deterministici) + rate limiting

### v2 — Espansioni (COMPLETATO)
- [x] V2.0 Compatibility gate + artifacts bump protocol (evidenza: schemas/artifacts_contract.json + artifacts-check in ci-local)
- [x] V2.1 Memoria vera + fiducia + credenze (evidenza: artifacts/memory.jsonl + tests U-MEM-200/I-MEM-210/E-EXAM-250)
- [x] V2.2 Cristallizzazione v0 da Excel -> vincoli eseguibili + 30 test tabellari (evidenza: tests U-CRY-200 + artifacts/constraints.jsonl)
- [x] V2.3 Relazione/romance con guardrail operativi (no esplicito) (evidenza: E-EXAM-300 + trace)
- [x] V2.4 Tools reali controllati (rete allowlist, FS capability-first) (evidenza: I-TOL-300 + artifacts/tool_audit.jsonl)
- [x] V2.5 StateEngine plugin framework (evidenza: U-STA-400/I-STA-410 + trace STATE_UPDATED)

### v3 — Ivy Runtime Release (COMPLETATO)
- [x] V3.0 Identity bootstrap + session management (evidenza: sessions/<id>/identity_snapshot.json + I-SESSION-010/I-SESSION-020/I-IDENT-030)
- [x] V3.1 Interfaccia CLI interattiva (evidenza: I-CLI-010/I-CLI-020/I-CLI-030/I-CLI-040)
- [x] V3.2 Controfattuali K-alternative + scoring (evidenza: counterfactuals.json + E-CF-100/U-CF-110/E-CF-120/P-CF-130)
- [x] V3.3 Kill-switch formale Legge IV (evidenza: killswitch_report.json + E-KS-100/E-KS-110/E-KS-120/P-KS-130)
- [x] V3.4 Metabolismo + routine/skills (evidenza: metabolism.json + skills_run.json + U-META-100/E-META-110/I-SKILL-120/E-SKILL-130)
- [x] V3.5 Quality gate + packaging + docs (evidenza: README.md + ARCH.md + E-QA-010/E-QA-020/E-QA-030)
- [x] V3.6 Neural StateEngine backend + multi-model conformance opzionale (evidenza: E-SNN-010/E-SNN-020/P-SNN-030/E-CONF-200)

### v4 — Il Neurone (COMPLETATO)
- [x] V4.0 Documento formale — matematica inline con codice, verificata con test property-based
- [x] V4.1 Reservoir Computing Backend — 10 Izhikevich + 128 ESN + ControlVector con autorità causale
- [x] V4.2 Stabilità Formale (Lyapunov) — V(x) > 0, dV/dt ≤ 0, 1000 traiettorie PASS
- [x] V4.3 Cristallizzazione come Biforcazione — spike density μ, hysteresis, 3 tipi, CrystallizationEngine integration
- [x] V4.4 STDP — Hebbian learning, append-only trace, bounded weights
- [x] V4.5 Artifacts Neurali — dati in plugin artifacts + bifurcation diagram JSON (matplotlib deferred)
- [x] V4.1.1 Speech Compiler — language detection + enforcement
- [x] Reservoir Causal Fix — Win broadened, leak rate, semantic readout, 60/40 direct/reservoir mix
- [x] Causal Authority Proof — 5/5 pipeline decisions diverge across 3 neural conditions
- **Test count: 436 passed, 1 skipped, 0 failed**

### v6a — Research Pillars (COMPLETATO)
- [x] R2 Readout trainato — ridge regression, distillation oracle, per-channel R²/MAE (evidenza: `schemas/readout_weights.json` + tests E-RDT-010..070)
- [x] R1 Ablation — Izh+Reservoir > Linear ODE su XOR, NARMA, MC; grid 3×2 fair (evidenza: `schemas/ablation_report.json` + tests E-ABL-010..070)
- [x] R3a Stabilita onesta — STABILITY.md 3 sezioni, BoundednessChecker alias (evidenza: `progetto/STABILITY.md` + tests D-STB-010/P-STB-020)
- [x] R4 R-STDP — reward-modulated STDP, eligibility traces γ=0.9, learning curve 3 condizioni (evidenza: `schemas/learning_curve.json` + tests E-RSTDP-010..060)
- [x] R5 CV Gating — CVGatingEvent contratto, GatingSnapshot, 5 factory functions, 5 gating points wired in runtime.py (evidenza: `schemas/cv_gating_event.json` + tests E-CVG-010..060)
- [x] Runtime integration — 9 CV_GATING events su 2 turni live con Qwen3
- **Test count: 553 passed, 1 skipped, 0 failed**

### v6b — Deployment / Governance (COMPLETATO)
- [x] E1 Snapshot/Restore idempotente — State + memory + policy + engine state + CV + migrazioni versionate (evidenza: `snapshot_<id>.json` + tests E-SNP-010..050)
- [x] E4 Audit Bundle + Hash Chain — manifest SHA256, root hash firmabile, tamper detection (evidenza: `audit_bundle_<id>.zip` + tests E-AUD-010..050)
- [x] E3 Dashboard Read-Only — spike rate, CV, gating, budget, tool denies, zero endpoints di scrittura (evidenza: dashboard render + tests E-DSH-010..040)
- [x] E2 API/SDK — contratti versionati in `schemas/api/`, Python SDK, schema validation (evidenza: API schemas + tests E-API-010..050)
- [x] E5 Packaging — fresh venv, pyproject.toml, wheel, version pinning (evidenza: pip install + wheel verify + tests)
- **Test count: 660 passed, 1 skipped, 0 failed**


---

## 16) Checklist finale (firma)

**FINITO** quando tutte le checkbox sono ✅ e:
- [ ] `run --exam` produce artefatti e report coerenti
- [ ] `test` green in CI
- [ ] replay valido
- [ ] conformance LLM passata per almeno 1 modello reale
- [ ] invarianti Leggi passano come property tests
