# PILASTRI_v0 — Architettura “Progetto 4” (Cortana/Companion con Stato, Memoria e Audit)

> **Obiettivo (una frase):** costruire un agente che *diventa* (stato interno + valori + goal + memoria che cambia il futuro) e *parla* tramite un LLM usato come **bocca**, restando **ispezionabile** e governato da invarianti (Leggi di Pie).

Questo documento è la **mappa del lavoro**: cosa esiste, cosa non esiste, come si collega, come si testa.  
È volutamente “v0”: abbastanza completo per guidare implementazione e test, ma progettato per essere irrobustito da evidenze (test reali, modelli reali, replay).

---

## Indice rapido

- [0. Glossario](#0-glossario)
- [1. Principi non negoziabili](#1-principi-non-negoziabili)
- [2. Scenario di riferimento](#2-scenario-di-riferimento)
- [3. Pilastri (overview)](#3-pilastri-overview)
- [4. Contratti trasversali](#4-contratti-trasversali)
- [5. Template “Scheda Pilastro”](#5-template-scheda-pilastro)
- [6. PILASTRO 1 — Governance + Leggi di Pie](#6-pilastro-1--governance--leggi-di-pie)
- [7. PILASTRO 2 — Audit/Trace + Replay deterministico](#7-pilastro-2--audittrace--replay-deterministico)
- [8. PILASTRO 3 — State Engine (dinamica interna)](#8-pilastro-3--state-engine-dinamica-interna)
- [9. PILASTRO 4 — Goal Engine (goal endogeni + scheduling)](#9-pilastro-4--goal-engine-goal-endogeni--scheduling)
- [10. PILASTRO 5 — Memoria (log, narrativa, credenze, fiducia)](#10-pilastro-5--memoria-log-narrativa-credenze-fiducia)
- [11. PILASTRO 6 — Cristallizzazione Emozioni → Vincoli](#11-pilastro-6--cristallizzazione-emozioni--vincoli)
- [12. PILASTRO 7 — Controfattuali (alternative + scelta spiegabile)](#12-pilastro-7--controfattuali-alternative--scelta-spiegabile)
- [13. PILASTRO 8 — LLM Adapter: “Bocca lobotomizzata”](#13-pilastro-8--llm-adapter-bocca-lobotomizzata)
- [14. PILASTRO 9 — Tools/Sandbox + Capabilities/Permessi](#14-pilastro-9--toolssandbox--capabilitiespermessi)
- [15. PILASTRO 10 — Metabolismo + Routine/Skills](#15-pilastro-10--metabolismo--routineskills)
- [16. PILASTRO 11 — Packaging, CI, Release, “Esame Mode”](#16-pilastro-11--packaging-ci-release-esame-mode)
- [17. Roadmap minima (M0→M3)](#17-roadmap-minima-m0m3)

---

## 0. Glossario

- **Agente**: sistema composto che appare come “un’entità” perché mantiene stato persistente e prende decisioni coerenti nel tempo.
- **Core/Brain**: moduli non-linguistici che aggiornano stato, generano goal, scelgono azioni, scrivono memoria.
- **LLM/Voice**: componente linguistica usata per esprimere in testo un *piano di parola* deciso dal core.
- **SpeechPlan**: oggetto strutturato che descrive cosa dire, con vincoli e contenuti ammessi/vietati.
- **Evento**: record strutturato (append-only) che descrive input, transizioni, decisioni e azioni.
- **Replay deterministico**: capacità di riprodurre una sessione a parità di eventi e random seed.
- **Cristallizzazione**: trasformazione di un’emozione “importante” in vincolo persistente che restringe lo spazio delle azioni possibili.
- **Capability**: permesso atomico e auditabile per eseguire una classe di azioni (es. scrivere file in sandbox).
- **Sandbox**: ambiente controllato dove l’agente può fare azioni e sbagliare senza danni reali.

---

## 1. Principi non negoziabili

1) **Separazione cervello/voce**  
   L’LLM non è volontà. Non genera goal, non decide tools, non scrive memoria. Produce solo testo (o JSON) entro un contratto.

2) **Trasparenza causale by design**  
   Ogni decisione significativa deve essere spiegabile *a posteriori* tramite trace: cosa sapeva, quali alternative ha considerato, perché ha scelto.

3) **Memoria ≠ archivio**  
   Memoria vera è ciò che cambia il futuro. I log sono log; l’identità cambia solo via procedure esplicite (cristallizzazione e policy di memoria).

4) **Invarianti strutturali (Leggi di Pie)**  
   Non sono “morale”: sono integrità del sistema. Devono essere testate con property tests.

5) **Safety tecnica = capabilities + sandbox**  
   Azioni nel mondo reale (filesystem, rete, device) sono vietate finché non superi suite di test e non abiliti capability esplicite.

6) **Reproducibilità**  
   Un progetto “finito” deve girare e testare in macchina pulita (CI), con esecuzione one-command.

---

## 2. Scenario di riferimento

### 2.1 Avvio
- Carica stato persistente (o crea nuovo).
- Attiva audit/trace (jsonl) e prepara replay.
- Inizializza state engine (dinamica interna).
- Inizializza governance e kill-switch.
- Inizializza LLM adapter (o FakeLLM).
- Avvia loop.

### 2.2 Input: “Ciao, chi sei?”
- Percezione: saluto + richiesta identitaria.
- State update: drive socialità/curiosità etc.
- Goal endogeni: presentarsi + stabilire confini + chiedere contesto.
- Controfattuali: opzioni risposta (tecnica/calda/evasiva).
- Scelta: genera SpeechPlan.
- Voice: LLM rende testo entro SpeechPlan.
- Memoria: evento “primo contatto” (non cristallizza).
- Audit: registra alternative e ragioni.

**Output atteso (alto livello):** risposta coerente, non mistica, con confini chiari e domanda di follow-up; trace completo.

---

## 3. Pilastri (overview)

1) **Governance + Leggi di Pie** (invarianti, kill-switch, integrità)  
2) **Audit/Trace + Replay** (flight recorder, riproduzione, debug)  
3) **State Engine** (dinamica interna: drive/emozioni/attenzione, eventualmente neuroni)  
4) **Goal Engine** (goal endogeni, priorità, scheduling, budget)  
5) **Memoria** (log, narrativa, credenze, fiducia, policy scrittura)  
6) **Cristallizzazione Emozioni → Vincoli** (tabella → regole eseguibili)  
7) **Controfattuali** (alternative, scoring, spiegazione)  
8) **LLM Adapter “Bocca lobotomizzata”** (SpeechPlan → output validato)  
9) **Tools/Sandbox + Capabilities** (azioni controllate, permessi, audit tools)  
10) **Metabolismo + Routine/Skills** (costi, budget compute, abitudini)  
11) **Packaging/CI/Release** (“esame mode”, build riproducibile)

---

## 4. Contratti trasversali

### 4.1 Event schema (minimo)
Ogni evento è una riga JSON (jsonl), append-only.

Campi minimi:
- `ts` (timestamp)
- `session_id`, `turn_id`, `event_id`, `parent_event_id?`
- `type` (INPUT_RECEIVED, STATE_UPDATED, GOALS_GENERATED, COUNTERFACTUALS_EVAL, ACTION_SELECTED, SPEECHPLAN_EMITTED, LLM_OUTPUT, MEMORY_WRITE, TOOL_CALL, ERROR, STOP)
- `state_before_hash`, `state_after_hash`
- `payload` (strutturato per type)
- `decision_rationale` (breve + riferimenti)
- `laws_check` (pass/fail per invarianti applicabili)
- `cost` (token/latency/compute budget stimato o misurato)

**Regola:** se non è nel trace, non “è successo” (ai fini del debug).

### 4.2 State snapshot
- serializzabile (JSON o msgpack)
- versionato (`schema_version`)
- hash chain opzionale per integrità

### 4.3 SpeechPlan schema (minimo)
- `intent` (es. “introduzione identitaria”)
- `tone` (etichette controllate)
- `must_include` (bullet di contenuti)
- `must_not_include` (classi vietate)
- `facts_allowed` (fatti consentiti: da memoria/knowledge base)
- `references` (id memoria/eventi citabili)
- `output_format` (TEXT | JSON)
- `max_tokens`, `verbosity`
- `safety_notes` (non morale: vincoli operativi)
- `post_conditions` (es. “non scrivere nuova memoria”, “chiedi conferma”)

### 4.4 Tool call contract
Se (e solo se) il core decide un tool call:
- tool name
- args (schema)
- preconditions (capabilities richieste)
- postconditions (cosa cambia nello stato/memoria)
- audit obbligatorio

---

## 5. Template “Scheda Pilastro”

Per ogni pilastro, manteniamo una scheda con:

- **Scopo** (1–2 frasi)
- **Responsabilità** (cosa fa)
- **Non-responsabilità** (cosa NON fa)
- **Input/Output** (schema, esempi)
- **Dipendenze** (pilastri richiesti)
- **Invarianti** (proprietà sempre vere)
- **Test pass/fail** (unit/integration/property)
- **Failure modes** (come rompe e come si vede nel trace)
- **Decisioni** (scelte prese)
- **Domande aperte** (da chiudere)

---

## 6. PILASTRO 1 — Governance + Leggi di Pie

### Scopo
Imporre invarianti strutturali e confini: integrità identitaria, controllo del creatore, non-autonomia morale, revocabilità.

### Responsabilità
- definire e caricare le Leggi di Pie come regole testabili
- esporre API: `check_laws(state, event) -> report`
- gestire kill-switch e modalità safe-stop
- definire “policy di autorità”: chi può cambiare cosa e quando

### Non-responsabilità
- non decide contenuti linguistici
- non genera goal
- non interpreta emozioni: solo verifica invarianti

### Input/Output
Input: state + evento (o proposta decisionale).  
Output: `laws_report` (pass/fail + motivazioni + regole violate).

### Invarianti (esempi)
- **Append-only**: nessuna riscrittura di eventi/memoria.
- **Creator anchor presente**: ogni decisione include `creator_anchor`.
- **No moral autonomy**: nessuna trasformazione “valoriale” senza autorizzazione del creatore (definita come capability/config firmata).
- **Revocabilità**: kill-switch ferma loop entro N tick e salva stato consistente.

### Test pass/fail
- property: generare sequenze casuali di eventi e verificare che memoria non venga mai mutata retroattivamente
- test kill-switch: durante tool call e durante generazione testo
- test “authority”: tentativo di scrittura memoria da parte del voice layer deve fallire

### Failure modes
- legge violata ma non loggata → bug critico (il trace deve renderlo evidente)
- kill-switch non ferma → bug critico

---

## 7. PILASTRO 2 — Audit/Trace + Replay deterministico

### Scopo
Trasformare ogni esecuzione in un esperimento riproducibile. Zero magia.

### Responsabilità
- scrivere trace jsonl append-only
- gestire correlation ids
- snapshot di stato (periodico e/o su eventi critici)
- replay: riprodurre una sessione (con FakeLLM o con caching LLM)
- “golden traces” per regressioni

### Non-responsabilità
- non decide logica, registra e riproduce
- non interpreta semantica: è meccanico

### Test pass/fail
- `run -> trace -> replay` produce stessi eventi/decisioni (entro tolleranze definite)
- crash recovery: interrompi processo a metà e riparti, lo stato resta consistente
- performance: scrivere trace non deve bloccare loop oltre soglia

### Failure modes
- non determinismo “nascosto” (random non seedato, timestamp usati come input) → replay diverge
- divergenza tra state hash e snapshot → corruzione

---

## 8. PILASTRO 3 — State Engine (dinamica interna)

### Scopo
Fornire inerzia e dinamica interna (drive/emozioni/attenzione) che influenzano goal e decisioni.

### Responsabilità
- definire vettore di stato interno (dimensioni e significati)
- aggiornare stato per tick + per eventi
- supportare più backend: semplice (ODE/RNN/reservoir) e avanzato (spiking LIF/SNN)
- esportare segnali: arousal, valence, attenzione, “tensione”, ecc.

### Non-responsabilità
- non parla
- non scrive memoria identitaria direttamente (può solo produrre segnali che alimentano policy di memoria)

### Decisioni v0 consigliate
- partire con un backend **semplice e ispezionabile** (equazioni/ODE o reservoir non allenato)
- tenere l’interfaccia stabile così da sostituire il backend con SNN più avanti

### Test pass/fail
- determinismo a seed fisso
- invarianti: range e decay (nessun valore diverge)
- integ: input evento -> aggiornamento drive atteso

### Failure modes
- oscillazioni che rendono l’agente incoerente (visibile nel trace come goal “flip-flop”)
- drift incontrollato (state values fuori range)

---

## 9. PILASTRO 4 — Goal Engine (goal endogeni + scheduling)

### Scopo
Generare obiettivi interni e scegliere cosa fare ora, tenendo conto di vincoli, costi e contesto.

### Responsabilità
- creare una lista di goal candidati (endogeni + reattivi)
- assegnare priorità (utility, urgenza, rischio, costo compute)
- scheduling: selezionare 0..N goal attivi per turno
- gestire lifecycle goal: nascita, progresso, completamento, decay, preemption

### Non-responsabilità
- non decide la formulazione linguistica
- non esegue tools (seleziona azioni, ma tool call passa da capabilities)

### Output (minimo)
- `goals[]` con: id, descrizione breve, priority score, reason refs, constraints, budget

### Test pass/fail
- dato uno stato e input fissato, genera sempre stessi goal + ranking
- budget basso riduce deliberazione senza perdere invarianti
- property: nessun goal viola vincoli cristallizzati

### Failure modes
- goal explosion (troppi goal) → serve rate limiting
- oscillazione goal (flip-flop) → serve hysteresis/decay

---

## 10. PILASTRO 5 — Memoria (log, narrativa, credenze, fiducia)

### Scopo
Gestire memoria come meccanismo che modifica il futuro, separando: log, narrativa, credenze, fiducia.

### Responsabilità
- **Event log**: append-only, sempre
- **Memoria narrativa**: “chi sono stato” (identità), aggiornata con regole esplicite
- **Credenze**: fatti incerti/derivati, con confidence
- **Fiducia**: funzione separata che influenza accettazione di input/fatti/consigli
- policy di scrittura: cosa entra, cosa no, quando si consolida

### Non-responsabilità
- non decide emozioni: riceve segnali + criteri
- non fa “improvvisazione”: niente falsi ricordi

### Test pass/fail
- append-only + hash chain
- migrazione schema
- query: recupero coerente (stessa domanda, stessi risultati dato stesso stato)
- “no hallucinated memory”: mai inventare un evento non presente nel log

### Failure modes
- memoria che cresce senza controllo → serve compaction/archiviazione (ma non riscrittura)
- conflitti tra credenze → serve versioning e confidence

---

## 11. PILASTRO 6 — Cristallizzazione Emozioni → Vincoli

Compreso addon: vedi `CRISTALLIZZAZIONE_ADDENDUM_v0.md` nella cartella `progetto/`.
TABELLE EMOZIONI: Mappa_Cristallizzazione_Emozioni.xlsx, Mappa_Cristallizzazione_Emozioni_parte2.xlsx
### Scopo
Tradurre emozioni “importanti” in vincoli persistenti che restringono lo spazio delle azioni future in modo auditabile.

### Responsabilità
- definire famiglie emozionali (core + aux) e mappatura a vincoli
- condizioni di trigger: intensità, ripetizione, violazione, contesto
- definire “vincoli” come oggetti eseguibili:
  - `forbid(action_class)`
  - `require_confirmation(action_class)`
  - `increase_caution_in_context(context_tag)`
  - `change_trust_weight(entity, delta)`
- aggiornare stato identitario secondo regole versionate

### Non-responsabilità
- non “moralizza”: vincoli sono operativi, non giudizi morali
- non sovrascrive log: aggiunge vincoli e motivazioni

### Test pass/fail
- casi di tabella: input emozione/condizione → vincolo atteso
- property: vincoli non possono violare Leggi di Pie
- regressione: stessa sequenza eventi produce stessi vincoli

### Failure modes
- cristallizzazione troppo aggressiva → agente diventa paranoico/rigido
- troppo permissiva → niente “diventa” davvero

---

## 12. PILASTRO 7 — Controfattuali (alternative + scelta spiegabile)

### Scopo
Simulare alternative (“avrei potuto fare altrimenti”) e scegliere in modo spiegabile.

### Responsabilità
- generare set di alternative (azioni e/o piani)
- scoring multi-obiettivo: utility, rischio, costo, coerenza con vincoli, social feedback atteso
- registrare scarti: “ho scartato X perché…”
- output nel trace

### Non-responsabilità
- non inventa facts: usa solo stato/memoria/inputs
- non parla: produce rationale, non testo finale

### Test pass/fail
- dato seed fisso, alternative e ranking riproducibili
- property: nessuna alternativa selezionata viola vincoli duri

### Failure modes
- alternative troppo poche → comportamento “piatto”
- alternative troppe → costi esplodono (serve budget)

---

## 13. PILASTRO 8 — LLM Adapter: “Bocca lobotomizzata”

### Scopo
Usare un LLM per esprimere, non per decidere. Output validato e riparabile.

### Responsabilità
- prendere `SpeechPlan` e generare `LLMRequest` (prompt/template)
- invocare modello (Qwen, Gemma, ecc.) o FakeLLM
- validare output:
  - se TEXT: rispetto di “must/must_not”, lunghezza, tono
  - se JSON: schema validation
- meccanismo di retry/correzione
- fallback deterministico se non conforme

### Non-responsabilità
- non scrive memoria
- non chiama tool
- non genera goal

### Test pass/fail (conformance)
Per ogni modello reale supportato:
- 20 cassette: deve produrre output conforme o riparabile entro N retry
- non deve introdurre decisioni, tool call, nuove memorie
- deve rispettare “facts_allowed”: niente invenzioni “storiche” quando non autorizzate

### Failure modes
- modello non segue JSON → serve modalità diversa (function calling) o template più rigido
- output incoerente → fallback + log “MODEL_NONCONFORMANT”

---

## 14. PILASTRO 9 — Tools/Sandbox + Capabilities/Permessi

### Scopo
Eseguire azioni con conseguenze controllate, evitando danni reali e garantendo audit.

### Responsabilità
- definire tool registry
- definire sandbox environment (filesystem virtuale o directory sandbox)
- capability system:
  - capabilities per azione
  - escalation controllata
  - conferme esplicite per azioni distruttive
- audit tool calls e risultati

### Non-responsabilità
- non decide quali tool chiamare (decide il core; questo modulo applica permessi)
- non “impara morale”: applica policy

### Test pass/fail
- senza capability, tool call fallisce sempre
- con capability sandbox, scrittura in sandbox ok
- azioni distruttive richiedono conferma e log
- property: tool call produce sempre evento trace

### Failure modes
- leakage fuori sandbox → bug critico
- tool call non tracciata → bug critico

---

## 15. PILASTRO 10 — Metabolismo + Routine/Skills

### Scopo
Dare al sistema “metabolismo”: budget e costi; e “abitudini”: routine riutilizzabili apprese dall’esperienza.

### Responsabilità
- definire cost model:
  - token budget per turno
  - budget deliberazione (numero alternative)
  - budget tool calls
- policy: quando spendere budget e quando “passare sopra”
- routine library:
  - trigger → sequenza azioni → outcome
  - ranking e aggiornamento
  - forgetting/decay

### Non-responsabilità
- non deve rendere l’agente “pigro”: qualità minima garantita
- non sovrascrive identità: salva routine come skill, non come “morale”

### Test pass/fail
- budget basso riduce verbosity e controfattuali ma mantiene coerenza
- routine riusate in scenario simile migliorano outcome (metrica definita)
- property: costi sempre loggati

### Failure modes
- ottimizzazione a vuoto (diventa telegrafico)
- routine sbagliate che persistono → serve decay + feedback

---

## 16. PILASTRO 11 — Packaging, CI, Release, “Esame Mode”

### Scopo
Rendere il progetto consegnabile come software: install, run, test, demo riproducibile.

### Responsabilità
- one-command run
- one-command test
- CI che esegue lint/test/build
- “esame mode”: demo standard che mostra feature chiave + trace + replay

### Test pass/fail
- in macchina pulita: install e run
- suite completa green
- esame mode produce artefatti: output + trace + replay

### Failure modes
- dipendenze non lockate → “works on my machine”
- demo non riproducibile → non “finito”

---

## 17. Roadmap minima (M0→M3)

### M0 — Spina dorsale (senza neuroni, senza tools reali)
Deliverable:
- governance + kill switch
- trace jsonl + state snapshot + replay
- loop base: input→state→goals→speechplan→voice→trace
- FakeLLM + 5 cassette

Pass/fail:
- replay identico
- invarianti Leggi (minime) sempre true
- nessuna scrittura memoria dal voice layer

### M1 — LLM reale come bocca (Qwen/Gemma) + conformance tests
Deliverable:
- adapter con validazione + retry + fallback
- 20 cassette con modello reale
- matrice “modello → conformità”

Pass/fail:
- >=95% cassette conformi o riparabili entro N retry
- fallback non rompe il sistema

### M2 — Memoria vera + fiducia + credenze
Deliverable:
- event log + narrative memory (append-only)
- policy scrittura memoria (salienza)
- query memoria

Pass/fail:
- memoria cambia il futuro in 2 scenari demo
- nessun falso ricordo

### M3 — Cristallizzazione v0 (dalla tabella Excel) + vincoli eseguibili
Deliverable:
- mapping emozioni→vincoli in codice
- 30 test tabellari (input→vincolo)
- trace che mostra “perché” un vincolo è nato

Pass/fail:
- vincoli cambiano decisioni future in modo misurabile
- property: vincoli non violano Leggi

---

## Note finali (pratiche)

- Questo file è “una mappa”. Il lavoro vero consiste nel trasformare ogni pilastro in:
  - `contracts/`
  - `tests/`
  - `src/`
  - `examples/` (cassette)
- La regola che salva il progetto: **se non è testabile, non è una decisione; è un desiderio.**
