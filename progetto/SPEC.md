# SPEC — Progetto 4 (v0)
**Titolo breve:** *Agente con stato interno, memoria trasformativa, audit e “voce” LLM separata*  
**Nome di lavoro:** *Pie Kernel / AURA-Brain (Progetto 4)*

---

## 1) Scopo

Costruire un agente software che:
- **evolve nel tempo** (stato interno + valori + goal endogeni + memoria che cambia il futuro),
- **resta ispezionabile** (audit/trace completo, replay deterministico, spiegazioni causali),
- **resta governato** da invarianti strutturali (Leggi di Pie),
- **parla** tramite un LLM usato come **apparato linguistico** (*bocca*), non come volontà (*cervello*).

---

## 2) Definizione operativa (cosa succede davvero)

### 2.1 Loop minimale (un “turno”)
1. **Input** (testo utente / evento esterno) → `INPUT_RECEIVED`
2. **Update stato interno** (drive/emozioni/attenzione, decay, segnali) → `STATE_UPDATED`
3. **Generazione goal** (endogeni + reattivi, con priorità/budget) → `GOALS_GENERATED`
4. **Controfattuali** (alternative + scoring) → `COUNTERFACTUALS_EVAL`
5. **Selezione azione** (es. parlare, chiedere, tool call sandbox, riflettere) → `ACTION_SELECTED`
6. **SpeechPlan** (piano di parola strutturato) → `SPEECHPLAN_EMITTED`
7. **Voce (LLM)**: rende testo/JSON **entro** SpeechPlan → `LLM_OUTPUT`
8. **Memoria**: scrittura **solo** se la policy lo decide (append-only) → `MEMORY_WRITE`
9. **Audit**: trace jsonl e snapshot → `TRACE_WRITE`

### 2.2 Scenario standard
Alla domanda: **“Ciao, chi sei?”**
- l’agente risponde in modo coerente col proprio stato e confini,
- registra *perché* ha scelto quel tono/struttura,
- **non** inventa ricordi,
- **non** auto-sacralizza (“sono cosciente” ecc.),
- salva al massimo un evento leggero (“primo contatto / richiesta identitaria”), senza cristallizzazione.

---

## 3) Cosa è (hard definitions)

È un sistema composto da moduli che cooperano come “un’entità” perché:

- **Stato interno persistente**: non è solo contesto testuale; vive su disco con versioning e recovery.
- **Valori dinamici**: pesi/priorità aggiornabili nel tempo (con regole esplicite).
- **Goal endogeni**: obiettivi possono emergere dallo stato, non solo da comandi esterni.
- **Controfattuali**: valuta alternative e registra scelte/scarti.
- **Memoria narrativa**: identità come storia che vincola il futuro.
- **Voce separata**: LLM = bocca. Il core decide, il LLM verbalizza.
	Nome entità: Ivy

	Alias: V
---

## 4) Cosa NON è (non-goals)

- **Non** è “un singolo modello” da fine-tune che magicamente “diventa vivo”.
- **Non** è un chatbot con “memoria” intesa come database di messaggi.
- **Non** è un sistema che si auto-definisce “entità morale” o “cosciente” (vietato a livello di governance/voice).
- **Non** è un agente con permessi reali illimitati (filesystem/rete/device): *capabilities + sandbox* per default.
- **Non** è un sistema che cambia “valori” o “leggi” senza autorizzazione del creatore e traccia.

---

## 5) Vincoli strutturali: Leggi di Pie (invarianti)

Le Leggi di Pie sono **invarianti tecniche**, non morale.  
Devono essere implementate come controlli + test (property tests) e riportate nel trace.

Requisiti minimi v0:
1. **Trasparenza causale**: ogni decisione significativa deve generare traccia (alternative + motivi).
2. **Ancoraggio al creatore**: ogni decisione e mutazione dello stato deve includere `creator_anchor`.
3. **Integrità del sé**: niente riscrittura retroattiva di log/memoria; append-only.
4. **Non-autonomia morale**: il sistema non può elevare se stesso a “fine ultimo” o cambiare regole valoriali fondamentali senza autorizzazione esplicita (capability/config firmata).
5. **Revocabilità assoluta**: kill-switch che ferma il loop e salva stato consistente.

---

## 6) Componenti e confini (high-level)

### 6.1 Core (decide)
- State Engine (dinamica interna)
- Goal Engine (goal endogeni + scheduling)
- Controfattuali (alternative + scoring)
- Memoria (policy + storage)
- Cristallizzazione emozioni → vincoli
- Governance (Leggi + kill-switch)
- Tool/Capability gate (permessi + sandbox)

### 6.2 Voice (esprime)
- LLM Adapter: prende `SpeechPlan` e produce testo/JSON conforme.
- **Divieti assoluti del Voice layer**:
  - scrivere memoria,
  - generare/alterare goal,
  - invocare tools,
  - alterare invarianti,
  - introdurre “facts” non autorizzati da `facts_allowed`.

---

## 7) Dati e formati (contratti minimi)

### 7.1 Event Log (jsonl append-only)
Ogni riga è un evento JSON con campi minimi:
- `ts`, `session_id`, `turn_id`, `event_id`, `type`
- `state_before_hash`, `state_after_hash`
- `payload` (per-type)
- `decision_rationale` (breve + riferimenti)
- `laws_check` (pass/fail + regole)
- `cost` (stime/misure: token/latency/compute budget)

**Regola:** se non è nel trace, non è “vero” ai fini di debug/replay.

### 7.2 State Snapshot
- serializzabile su disco
- `schema_version` obbligatorio
- migrazioni previste (v0→v1…)
- hash (integrità) raccomandato

### 7.3 SpeechPlan (oggetto obbligatorio verso l’LLM)
Campi minimi:
- `intent`, `tone`
- `must_include[]`, `must_not_include[]`
- `facts_allowed[]` (o riferimenti a memoria/KB)
- `output_format` (TEXT|JSON)
- `max_tokens`, `verbosity`
- `post_conditions` (es. “non scrivere memoria”)

### 7.4 ToolCall (solo dal core)
- `tool_name`, `args`
- `required_capabilities[]`
- `preconditions`, `postconditions`
- audit obbligatorio

---

## 8) Memoria (regola fondamentale)

**Memoria ≠ archivio.**  
La memoria “vera” è quella che **modifica il futuro**: restringe o amplia lo spazio delle azioni possibili.

### 8.1 Strati di memoria (minimo)
- **Log eventi**: sempre, append-only.
- **Narrativa/Identità**: aggiornata solo via policy esplicite.
- **Credenze**: con confidence, versionate (possono contraddirsi).
- **Fiducia**: separata da credenze e ricordi (influenza accettazione/azioni).

### 8.2 Politica di scrittura (v0)
- evento banale → log sì, memoria identitaria no
- evento ad alta valenza/violazione/ripetizione → può attivare cristallizzazione e vincoli
- niente “falsi ricordi”: la voce non inventa, il core non sintetizza come fatto ciò che è ipotesi

---

## 9) Cristallizzazione emozioni → vincoli (v0)

Le emozioni “importanti” diventano identità quando:
- limitano lo spazio delle azioni future,
- e sono attivate da condizioni ripetute o violazioni significative.

Output della cristallizzazione:
- **Vincoli eseguibili** (oggetti):
  - `forbid(action_class)`
  - `require_confirmation(action_class)`
  - `increase_caution(context_tag)`
  - `adjust_trust(entity, delta)`
- Ogni vincolo deve avere:
  - `trigger_events[]` (riferimenti nel log)
  - `explanation` (breve, auditabile)
  - `strength` e `decay` (se previsto)

---

## 10) Costi / “Metabolismo” (v0)

Il sistema deve modellare costi pratici:
- budget per turno (token/latency)
- budget deliberazione (numero alternative)
- budget tool calls

Principio: ridurre costi **senza** degradare sotto una soglia minima di qualità e coerenza.

---

## 11) Determinismo e replay

Requisiti:
- seed controllato
- niente timestamp usati come input decisionale (solo come metadato)
- caching per chiamate LLM (o FakeLLM) per replay
- comando `replay(trace)` deve ricostruire le decisioni del core

---

## 12) Sicurezza tecnica (capabilities + sandbox)

Default:
- nessun accesso a filesystem reale, rete, processi, device
- tools ammessi solo in sandbox (directory dedicata o filesystem virtuale)

Escalation:
- le capabilities si abilitano solo dopo test e con conferma esplicita del creatore
- azioni distruttive richiedono conferma + logging

---

## 13) Requisiti di “progetto finito” (collegamento a DONE.md)

Questo SPEC definisce **cosa** è il progetto.  
La definizione di “finito” è in `DONE.md` e deve essere **pass/fail**.

Minimo v0 (da completare in DONE.md):
- M0: loop + trace + replay + FakeLLM
- M1: LLM reale come bocca + conformance tests
- M2: memoria vera + policy + query
- M3: cristallizzazione v0 + vincoli eseguibili + test tabellari

---

## 14) Limiti v0 (scelte deliberate)

- Nessun “corpo” o sensori reali: solo eventi simulati/sandbox.
- Nessun training end-to-end di SNN: lo State Engine può partire semplice e sostituibile.
- Nessun tool reale distruttivo.
- Nessuna “autonomia morale”: solo vincoli operativi derivati da policy del creatore e dalle Leggi.

---

## 15) Checklist di coerenza (autoverifica)

Questo SPEC è rispettato se:
- l’LLM non può scrivere memoria/goal/tools (testabile)
- ogni turno produce trace completo
- replay funziona
- memoria non viene riscritta retroattivamente
- cristallizzazione produce vincoli eseguibili e tracciati
- kill-switch ferma e salva

---

## Determinismo e numerica (regole v0)

Queste regole esistono per rendere **replay deterministico** reale (non “sul mio PC e con la luna giusta”).

### Tempo (nemico del determinismo)
- Il tempo (`timestamp`) è **solo logging** nel trace.
- Il core decide usando solo **tempo logico**: `logical_time = (session_id, turn_id, step_id)`.
- Nessuna decisione (ranking, soglie, cristallizzazione) può dipendere da `timestamp`, clock di sistema o latenza.

### Entropia (fonti da neutralizzare)
- Niente UUID random per entità critiche. Gli ID sono **deterministici** (contatori o hash stabili).
- Ogni collezione che può influenzare scelte (goal, alternative, vincoli) viene **ordinata esplicitamente** prima dell’uso.
- Tie-break obbligatorio: a parità di punteggio si usa un criterio deterministico (es. `id` lessicografico).

### ODE / State Engine (minimo ridicolo ma ferreo)
- Integratore v0: **Euler**.
- Step: **dt fisso per turno** (nessun substep implicito).
- Clamp/Range: ogni variabile ha range definito (es. `valence [-1,1]`, `arousal [0,1]`, ecc.).
- Quantizzazione: dopo update e clamp si applica **round/quantizzazione** (es. 4 decimali) *prima* di usare soglie e ranking.

### Versioning contratti (compatibilità)
- Ogni record serializzato include `schema_version` (Event/State/MemoryRecord/Constraint/SpeechPlan).
- Policy v0: **fail fast** se la versione non è supportata (migrazioni quando dichiarate).

### Append-only (semantica formale)
- Trace e memoria sono **append-only**: niente edit, niente delete, niente riscrittura retroattiva.
- `state_snapshot` e report sono **derivati** e rigenerabili.
- Correzioni: si usa un evento/record di **Repair** che aggiunge una rettifica e modifica il futuro senza cancellare il passato.





**Fonte di mappa/indice:** `PILASTRI_v0.md` (deve restare coerente con questo SPEC).


## Roadmap versioni (scope evolutivo)

> Questa sezione esiste per NON dimenticare come espandere il progetto senza rompere l’architettura.
> Regola: ogni voce qui deve diventare (quando implementata) una riga in DONE.md + almeno 1 test in TESTS.md.

### v0 (Blueprint + M0–M3)
- Obiettivo: spina dorsale + determinismo + LLM “bocca” + memoria + cristallizzazione v0.
- Deliverable minimi: trace/replay, contratti base, Esame Mode, conformance su 1 LLM reale.

### v1 (Hardening / Software affidabile)
- LLM Adapter: conformance matrix multi-modello + caching replay robusto + policy retry/timeout definite.
- Persistence: atomic commit, migrazioni schema complete, crash recovery testata.
- Observability: golden traces ufficiali + diff tool per replay (decisioni vs linguaggio).
- Governance: authority model più stretto (autorizzazioni/firming reali) + guardrail hard “pending-review” formalizzati.
- Tools: sandbox più ricca (FS virtuale completo + tool deterministici), permessi granulari, rate limiting.
- Quality: copertura test aumentata, CI con artifact upload standard.

## StateEngineProtocol (interfaccia stabile: ODE oggi, “neurone” domani)

### Scopo
Il **State Engine** è il modulo che aggiorna lo **stato interno** (drives/affect/traits/values) in modo deterministico.
In v0 usa una dinamica semplice (es. ODE/Euler), ma **in v2** può essere sostituito da un backend più “neurale” (SNN/Reservoir/etc) **senza cambiare il resto del Kernel**.

> Regola: il Kernel non dipende da *come* lo stato viene calcolato. Dipende solo da questa interfaccia.

### Principi non negoziabili
- **Determinismo**: stesso input + stesso seed + stesso trace ⇒ stesso output (entro quantizzazione dichiarata).
- **No side effects**: lo State Engine non scrive memoria, non chiama tools, non parla (non produce testo).
- **No tempo reale**: niente timestamp per decidere (solo logging); l’unico “tempo” valido è `LogicalTime` (session/turn/step).
- **Quantizzazione e clamp** obbligatori prima di soglie/ranking, per evitare drift numerico.
- **Config esplicita**: parametri e versione sono dichiarati e tracciabili.

### Contratto minimo (API concettuale)
**Input:**
- `prev_state` (State)
- `signals` (feature estratte dall’input utente e dal contesto del turno; deterministiche)
- `engine_config` (parametri + schema_version + ranges + dt)
- `logical_time` (session/turn/step)

**Output:**
- `next_state` (State) — già clampato e quantizzato
- `state_delta` (dict) — cambiamenti spiegabili (numeri e motivi)
- `engine_trace` (dict) — metadati deterministici (dt, integratore, seed, version, clamp_hits)

### Invarianti di sicurezza e audit
- Lo State Engine **non può**:
  - leggere/scrivere MemoryStore
  - scrivere ConstraintStore
  - invocare LLM
  - invocare Tools
- Ogni update deve produrre un record tracciabile in `trace.jsonl` (es. `STATE_UPDATED`) contenente:
  - `engine_id`, `engine_version`
  - `dt`, `integrator`
  - `ranges/clamps`
  - `quantization`
  - `state_delta` (o hash se troppo grande)

### Nota sul “neurone” (niente conflitto)
Il “neurone” non sostituisce l’architettura: **è un backend dello State Engine**.
- v0: `ODEStateEngine` (semplice, ferreo, auditabile)
- v2: `NeuralStateEngine` (SNN/Reservoir) dietro la **stessa interfaccia**
Se il backend neurale viola determinismo/auditability, è fuori protocollo.

### v2 (Espansioni “vita nel mondo”)
- State Engine: backend neurale avanzato (SNN/Reservoir/ODE) sostituibile via interfaccia stabile.
- Learning: routine/skills con reward shaping, curricula, forgetting controllato.
- Embodiment: sensori/attuatori in sandbox → reale, con reflex layer separato.
- Social layer: modelli di fiducia più ricchi + gestione “relazione” (romance/eros) con guardrail.
- Tooling: rete controllata (allowlist), filesystem reale capability-first, integrazione UI “OS-like”.

### Slot da riempire (per non dimenticare)
- v1: Requisiti numerici (latenze, budget, limiti memoria): [ ] TBD
- v1: Modelli target e profili (Qwen/Gemma/altro): [ ] TBD
- v2: Sensori/attuatori previsti: [ ] TBD
- v2: Metriche “diventa” (cosa misuriamo?): [ ] TBD
