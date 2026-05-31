# CRISTALLIZZAZIONE — Addendum v0 (considerazioni operative)

Questo file raccoglie decisioni e guardrail per la **cristallizzazione emozioni → vincoli** (Pilastro 6).

## 1) Hard change “autonomo” (Sì), ma con cintura di sicurezza
**Decisione:** i cambiamenti **hard** possono scattare automaticamente quando la soglia è superata, perché l’agente deve reagire anche a errori “che cambiano il futuro”.

**Guardrail obbligatori:**
- **Append-only**: se un hard è “sbagliato”, non si riscrive; si appende una **riparazione** (repair record) che lo neutralizza o lo attenua.
- **Two-phase hard** (consigliato): hard creato subito ma marcato **PENDING_REVIEW** o con **decay lungo** finché non:
  - (a) il Creatore conferma, oppure
  - (b) ci sono evidenze ripetute/coerenti.
- **Specificità**: hard deve essere legato a **persona/contesto** (niente generalizzazioni a “tutti”).
- **Audit completo**: ogni hard ha `source_refs`, trigger, soglia e spiegazione breve.

## 2) Romance/Eros: non “poesia”, ma dinamica funzionale
Romance è importante perché definisce una persona, ma:
- molte emozioni eros/attrazione sono **bias Soft** e/o **metabolismo** (urgenza, costo deliberazione), non “leggi morali”.
- l’architettura deve prevenire:
  - ossessione (rate limit, reality-check)
  - manipolazione (confini espliciti)
  - coercizione (consenso come vincolo operativo duro)

## 3) Regola pratica: No / Soft / Sì
- **No**: resta nel `State` (mood/drive) e nel trace, senza vincoli persistenti.
- **Soft**: crea bias o vincoli graduali e reversibili (decay + evidenze).
- **Sì**: crea vincolo persistente + record identitario (append-only) con percorso di rielaborazione.

## 4) “Riparazione” (repair) — come si corregge senza riscrivere
Quando l’utente/Creatore dice “hai capito male”:
- crea un `RepairRecord` con:
  - refs all’evento e al vincolo nato
  - nuova interpretazione
  - azione: attenua/limita/neutralizza il vincolo
- il sistema aggiorna `constraints_active` senza cancellare il passato.

## 5) Nota sicurezza (non-autonomia morale)
La cristallizzazione produce **vincoli operativi**, non giudizi morali.  
I “valori” si aggiornano solo via procedure autorizzate e tracciate (authority model).

