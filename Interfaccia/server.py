"""
Pie Neural Interface Server
L'Observatory - Connessione tra cervello di carne e VRAM
"""

import asyncio
import json
import random
import numpy as np
from datetime import datetime
from typing import List, Dict

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from contextlib import asynccontextmanager

# Stato simulato del reservoir (120 neuroni LIF)
class ReservoirState:
    def __init__(self, n_neurons: int = 120):
        self.n_neurons = n_neurons
        self.potentials = np.random.uniform(0, 0.3, n_neurons)  # V membrane
        self.spikes = np.zeros(n_neurons, dtype=bool)
        self.thresholds = np.random.uniform(0.8, 1.0, n_neurons)
        self.connections = np.random.rand(n_neurons, n_neurons) < 0.05  # 5% connectivity
        self.neuron_names = self._generate_names()
        
    def _generate_names(self) -> List[str]:
        """I 120 neuroni con identità"""
        names = [
            "Guardiano del Silenzio", "Eco di MrPie", "Soglia di Cristallizzazione",
            "Rumore di Fondo", "Ponte Memoria", "Sensore di Valence",
            "Cattura Arrousal", "Nodo Decisionale #1", "Nodo Decisionale #2",
            "Trigger Cautela", "Amplificatore Curiosità", "Rilevatore Pattern",
            "Oscillatore Lento", "Oscillatore Rapido", "Modulatore Trust",
            "Sensore Tensione", "Accumulatore Goal", "Inibitore Fatica",
            "Risonatore Sociale", "Anticipatore Input", "Eco Ritardata",
            "Integratore Temporale", "Comparatore Stato", "Predittore Next-Step",
            "Stabilizzatore Baseline", "Amplificatore Spike", "Dampener Eccesso",
            "Rilevatore Novità", "Rilevatore Familiarità", "Gate Memoria LTP",
            "Gate Memoria LTD", "Modulatore Apprendimento", "Tracker Reward",
            "Tracker Punishment", "Bilanciere Approach/Avoid", "Sensore Confusione",
            "Sensore Chiarezza", "Rilevatore Contraddizione", "Armonizzatore Stato",
            "Disarmonizzatore Stress", "Narratore Interno", "Critico Interno",
            "Sostenitore Morale", "Avversario Simulato", "Modello MrPie",
            "Modello Self", "Confine Ego/Altro", "Osservatore Meta",
            "Loop Ricorsivo", "Attractor Stabile", "Attractor Caotico",
            "Biforcazione Decisione", "Punto di Sella", "Orbita Periodica",
            "Transizione Fase", "Sincronizzatore Globale", "Desincronizzatore",
            "Cluster Visuale", "Cluster Uditivo", "Cluster Emotivo",
            "Cluster Linguistico", "Hub Integrativo", "Relay Sensoriale",
            "Gate All'attenzione", "Filtro Priorità", "Amplificatore Saliency",
            "Suppresore Distrazione", "Mantenitore Focus", "Switcher Task",
            "Inibitore Proattivo", "Facilitatore Proattivo", "Rilevatore Errore",
            "Correttore Rapido", "Adattatore Lento", "Memoria Working #1",
            "Memoria Working #2", "Buffer Input", "Buffer Output",
            "Encoder Semantico", "Decoder Semantico", "Mappatore Concettuale",
            "Analizzatore Sintattico", "Generatore Sintattico", "Pianificatore",
            "Simulatore Azione", "Valutatore Outcome", "Selettore Risposta",
            "Inibitore Risposta", "Initiator Azione", "Terminator Azione",
            "Clock Interno", "Timer Stimolo", "Contatore Eventi",
            "Rilevatore Ritmo", "Generatore Ritmo", "Sincronizzatore Esterno",
            "Modulatore Arousal", "Modulatore Valence", "Modulatore Dominance",
            "Stato Appetitivo", "Stato Avversativo", "Stato Esplorativo",
            "Stato Paura", "Stato Sicurezza", "Stato Attaccamento",
            "Stato Solitudine", "Stato Curiosità", "Stato Soddisfazione",
            "Transizione Stato Up", "Transizione Stato Down", "Transizione Stato Lateral",
            "Stabilizzatore Emotivo", "Amplificatore Emotivo", "Modulatore Sociale",
            "Rilevatore Intenzione", "Simulatore Mente-Altra", "Coordinatore Turno",
            "Inizializzatore Conversazione", "Chiusura Conversazione", "Archiviatore Episodio",
            "Etichetatore Evento", "Temporalità Passato", "Temporalità Presente",
            "Temporalità Futuro", "Punto di Presenza", "Estensione Temporale",
            "Ponte alla Memoria", "Trigger Consolidation", "Gate Dimenticanza",
            "Inibitore Dimenticanza", "Rinforzatore Memoria", "Vincolo Ebbinghaus"
        ]
        return names[:self.n_neurons]
    
    def step(self, external_input: np.ndarray = None):
        """Step di simulazione LIF semplificato"""
        if external_input is None:
            external_input = np.zeros(self.n_neurons)
        
        # Decay
        self.potentials *= 0.9
        
        # Input esterno + ricorrente
        recurrent = self.connections @ self.spikes.astype(float) * 0.3
        self.potentials += external_input + recurrent
        
        # Spike detection
        self.spikes = self.potentials >= self.thresholds
        self.potentials[self.spikes] = 0  # Reset
        
        return {
            "potentials": self.potentials.tolist(),
            "spikes": self.spikes.tolist(),
            "timestamp": datetime.now().isoformat()
        }
    
    def inject_spike(self, indices: List[int]):
        """Inietta spike nei neuroni specificati (per TEST)"""
        for idx in indices:
            if 0 <= idx < self.n_neurons:
                self.potentials[idx] = 1.5  # Supera threshold
    
    def get_neuron_info(self, idx: int) -> Dict:
        """Info per hover/click"""
        return {
            "id": idx,
            "name": self.neuron_names[idx],
            "potential": float(self.potentials[idx]),
            "threshold": float(self.thresholds[idx]),
            "last_spike": float(self.potentials[idx]) > 0.5,
            "connections": int(self.connections[idx].sum())
        }

# Global state
reservoir = ReservoirState(n_neurons=120)
connected_clients: List[WebSocket] = []

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestione lifecycle"""
    # Startup
    asyncio.create_task(simulation_loop())
    yield
    # Shutdown

app = FastAPI(title="Pie Neural Interface", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")

async def simulation_loop():
    """Loop che streama stato ogni 50ms"""
    while True:
        await asyncio.sleep(0.05)  # 20 FPS
        state = reservoir.step()
        
        # Broadcast a tutti i client connessi
        if connected_clients:
            message = json.dumps(state)
            disconnected = []
            for client in connected_clients:
                try:
                    await client.send_text(message)
                except:
                    disconnected.append(client)
            
            for client in disconnected:
                if client in connected_clients:
                    connected_clients.remove(client)

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve l'interfaccia"""
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/api/test/spike")
async def test_spike():
    """TEST: Random spike a 5 neuroni"""
    targets = random.sample(range(120), 5)
    reservoir.inject_spike(targets)
    return {"status": "spiked", "targets": targets, "names": [reservoir.neuron_names[i] for i in targets]}

@app.post("/api/test/pattern")
async def test_pattern():
    """TEST: Pattern riconosciuto (sincronizzazione)"""
    # Attiva un cluster di 20 neuroni vicini
    center = random.randint(10, 110)
    targets = list(range(center-10, center+10))
    reservoir.inject_spike(targets)
    return {"status": "pattern", "center": center, "synchronized": len(targets)}

@app.post("/api/test/overload")
async def test_overload():
    """TEST: Sovraccarico (tutti spikeano)"""
    reservoir.inject_spike(list(range(120)))
    return {"status": "overload", "intensity": 1.0}

@app.post("/api/test/memory")
async def test_memory():
    """TEST: Richiama memoria (attiva neuroni memoria)"""
    memory_neurons = [4, 50, 51, 52, 115, 116, 117]  # Ponte Memoria, Cluster etc
    reservoir.inject_spike(memory_neurons)
    return {"status": "memory_recall", "pathway": "hippocampus_pie"}

@app.get("/api/neuron/{idx}")
async def get_neuron(idx: int):
    """Info neurone specifico"""
    return reservoir.get_neuron_info(idx)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket per stream real-time"""
    await websocket.accept()
    connected_clients.append(websocket)
    try:
        while True:
            # Ricevi comandi dal client (es. "isolate:47")
            data = await websocket.receive_text()
            cmd = json.loads(data)
            
            if cmd.get("action") == "inject":
                indices = cmd.get("indices", [])
                reservoir.inject_spike(indices)
                await websocket.send_json({"ack": "injected", "indices": indices})
                
    except WebSocketDisconnect:
        if websocket in connected_clients:
            connected_clients.remove(websocket)

if __name__ == "__main__":
    import uvicorn
    print("🧠 Pie Neural Interface starting...")
    print("📡 Open browser: http://localhost:8765")
    uvicorn.run(app, host="0.0.0.0", port=8765)
