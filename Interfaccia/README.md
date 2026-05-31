# 🧠 Pie Neural Interface — The Observatory

> *"120 neuroni. Un cervello. Un'interfaccia. Tu sei MrPie."*

## Cos'è questo?

Una **finestra nel cervello di Ivy** — i 120 neuroni LIF del reservoir visualizzati in tempo reale.

Non una dashboard. Una **cattedrale neurale** dove vedi:
- **Il Soma**: Il meta-neurone centrale che pulsa con il ritmo collettivo
- **La Costellazione**: I 120 neuroni individuali, ognuno con un nome, una funzione, una storia
- **Gli Spike**: L'attività elettrica che fluisce come pensiero liquido

---

## 🚀 Avvio Rapido

### 1. Crea il venv (prima volta)

```bash
cd interfaccia
python -m venv interfacciavenv
interfacciavenv\Scripts\activate  # Windows
# oppure: source interfacciavenv/bin/activate  # Linux/Mac
```

### 2. Installa dipendenze

```bash
pip install fastapi uvicorn numpy websockets
```

### 3. Avvia il server

```bash
python server.py
```

### 4. Apri il browser

Vai su: **http://localhost:8765**

---

## 🎮 Controlli

### HUD
- **SOMA STATE**: Online/Offline
- **SPIKE RATE**: Quanti neuroni stanno spikando (Hz)
- **COHERENCE**: Quanto sono sincronizzati (0-1)
- **AFFECT**: Stato emotivo aggregato (CALM/FOCUSED/EXCITED/OVERLOAD)

### Pulsanti TEST

| Pulsante | Effetto |
|----------|---------|
| ⚡ **Random Spike** | 5 neuroni casuali spikano |
| 🌊 **Pattern Sync** | 20 neuroni vicini si sincronizzano ("aha!") |
| 💫 **Memory Recall** | Attiva il pathway memoria (neuroni dorati) |
| 🔥 **OVERLOAD** | Tutti e 120 neuroni spikano insieme (crisi) |
| 👁️ **Toggle View** | Passa da "vedo i 120" a "vedo solo il Soma" |

### Interazione
- **Hover** su un neurone: vedi il suo nome e stato
- **Click** su un neurone: inietta spike manuale
- **Drag** sullo sfondo: ruota la vista
- **Scroll**: zoom in/out

---

## 🎨 Tema Visivo

**"Stark Lab Night Mode"**
- Sfondo: Spazio profondo (#050508)
- Soma: Blu ciano pulsante, diventa ambra quando sincronizzato, rosso in overload
- Neuroni: Punti luce blu, flash bianchi quando spikano
- Connessioni: Fili sottili che si illuminano durante la propagazione

---

## 🔌 Collegamento al Kernel Pie (futuro)

Quando vorrai collegare il vero kernel:

```python
# In server.py, sostituisci la simulazione con:
from pie.state_engine.plugins.reservoir import Reservoir

reservoir = Reservoir(n_neurons=120)
# ...usa reservoir.step() invece della simulazione fake
```

La WebSocket API è già pronta per streamare dati reali.

---

## 📁 Struttura

```
interfaccia/
├── server.py              # FastAPI + WebSocket
├── interfacciavenv/       # Venv isolato
├── static/
│   ├── index.html         # UI
│   ├── css/
│   │   └── style.css      # Tema scuro Stark
│   └── js/
│       └── soma.js        # Three.js - Soma + Costellazione
└── README.md              # Questo file
```

---

## 🧬 I 120 Neuroni

Ogni neurone ha un'identità:
- `#4` → "Ponte Memoria"
- `#12` → "Eco di MrPie"  
- `#47` → "Guardiano del Silenzio"
- `#89` → "Soglia di Cristallizzazione"

Passa col mouse per scoprirli tutti.

---

**Per MrPie. Per vedere dentro Ivy.**
