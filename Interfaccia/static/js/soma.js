/**
 * Pie Neural Interface v3.0 - GOLDEN NEURAL SPHERE
 * Multi-layered wireframe orrery with particle constellations
 * Inspired by Doctor Strange / Ancient-Tech interfaces
 */

let scene, camera, renderer, composer, controls;
let goldenSphere, constellation, particleNebula;
let neurons = [];
let neuralConnections = [];
let ws, raycaster, mouse, hoveredNeuron = null;
let cameraShake = 0, time = 0;

// Palette oro/ambra per il tema JARVIS/Ancient
const PALETTE = {
    gold: 0xffb700,
    amber: 0xff8c00,
    bronze: 0xcd7f32,
    cyan: 0x00d9ff,
    deepBlue: 0x001a33,
    coreWhite: 0xfff8e7
};

// Stato neuroni
let neuronStates = { potentials: new Array(120).fill(0), spikes: new Array(120).fill(false) };

function init() {
    const container = document.getElementById('canvas-container');
    
    // Scene
    scene = new THREE.Scene();
    scene.fog = new THREE.FogExp2(0x020205, 0.02);
    
    // Camera
    camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 1000);
    camera.position.set(0, 0, 22);
    
    // Renderer
    renderer = new THREE.WebGLRenderer({ antialias: false, alpha: true, powerPreference: "high-performance" });
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.toneMapping = THREE.ReinhardToneMapping;
    container.appendChild(renderer.domElement);
    
    // Post-processing
    setupPostProcessing();
    
    // Controls
    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.03;
    controls.rotateSpeed = 0.4;
    controls.autoRotate = true;
    controls.autoRotateSpeed = 0.3;
    controls.minDistance = 12;
    controls.maxDistance = 40;
    
    // Raycaster
    raycaster = new THREE.Raycaster();
    mouse = new THREE.Vector2();
    
    // Costruisci la scena
    createGoldenSphere();
    createNeuralConstellation();
    createNebulaParticles();
    createAmbientParticles();
    
    // Eventi
    window.addEventListener('resize', onWindowResize);
    window.addEventListener('mousemove', onMouseMove);
    window.addEventListener('click', onClick);
    
    connectWebSocket();
    animate();
    
    log("Golden Neural Sphere initialized");
    log("Architecture: Multi-layered wireframe orrery");
}

function setupPostProcessing() {
    const renderScene = new THREE.RenderPass(scene, camera);
    
    const bloomPass = new THREE.UnrealBloomPass(
        new THREE.Vector2(window.innerWidth, window.innerHeight),
        2.0,  // strength aumentata
        0.5,  // radius
        0.7   // threshold
    );
    
    composer = new THREE.EffectComposer(renderer);
    composer.addPass(renderScene);
    composer.addPass(bloomPass);
}

// LA SFERA DORATA PRINCIPALE (tipo nell'immagine)
function createGoldenSphere() {
    goldenSphere = new THREE.Group();
    
    // 1. NUCLO - Sfera bianca intensa al centro
    const coreGeo = new THREE.IcosahedronGeometry(0.8, 2);
    const coreMat = new THREE.MeshBasicMaterial({
        color: PALETTE.coreWhite,
        wireframe: true
    });
    const core = new THREE.Mesh(coreGeo, coreMat);
    goldenSphere.add(core);
    
    // 2. STRATI DI WIREFRAME CONCENTRICI (l'effetto oro stratificato)
    const layers = [
        { radius: 1.5, detail: 3, color: PALETTE.gold, opacity: 0.4, speed: 0.005 },
        { radius: 2.2, detail: 2, color: PALETTE.amber, opacity: 0.3, speed: -0.008 },
        { radius: 3.0, detail: 2, color: PALETTE.bronze, opacity: 0.25, speed: 0.012 },
        { radius: 3.8, detail: 1, color: PALETTE.gold, opacity: 0.2, speed: -0.015 },
        { radius: 4.5, detail: 1, color: PALETTE.amber, opacity: 0.15, speed: 0.02 }
    ];
    
    layers.forEach((layer, i) => {
        const geo = new THREE.IcosahedronGeometry(layer.radius, layer.detail);
        const mat = new THREE.MeshBasicMaterial({
            color: layer.color,
            wireframe: true,
            transparent: true,
            opacity: layer.opacity,
            blending: THREE.AdditiveBlending
        });
        const mesh = new THREE.Mesh(geo, mat);
        mesh.userData = { rotationSpeed: layer.speed, originalOpacity: layer.opacity };
        goldenSphere.add(mesh);
    });
    
    // 3. ANELLI ORBITALI COMPLESSI (tipo Saturno ma dorato)
    const ringConfigs = [
        { inner: 2.8, outer: 3.0, color: PALETTE.gold, tiltX: 0.5, tiltY: 0.3, speed: 0.01 },
        { inner: 3.5, outer: 3.6, color: PALETTE.amber, tiltX: 1.2, tiltY: 0.8, speed: -0.015 },
        { inner: 4.2, outer: 4.4, color: PALETTE.bronze, tiltX: 0.3, tiltY: 1.5, speed: 0.008 },
        { inner: 5.0, outer: 5.1, color: PALETTE.gold, tiltX: 2.0, tiltY: 0.5, speed: -0.012 }
    ];
    
    ringConfigs.forEach(config => {
        const geo = new THREE.RingGeometry(config.inner, config.outer, 128);
        const mat = new THREE.MeshBasicMaterial({
            color: config.color,
            transparent: true,
            opacity: 0.3,
            side: THREE.DoubleSide,
            blending: THREE.AdditiveBlending
        });
        const ring = new THREE.Mesh(geo, mat);
        ring.rotation.x = config.tiltX;
        ring.rotation.y = config.tiltY;
        ring.userData = { rotationSpeed: config.speed };
        goldenSphere.add(ring);
    });
    
    // 4. LINEE DI ENERGIA RADIALI (raggi che partono dal centro)
    for (let i = 0; i < 24; i++) {
        const angle = (i / 24) * Math.PI * 2;
        const points = [
            new THREE.Vector3(0, 0, 0),
            new THREE.Vector3(Math.cos(angle) * 5, Math.sin(angle) * 5, (Math.random() - 0.5) * 2)
        ];
        const geo = new THREE.BufferGeometry().setFromPoints(points);
        const mat = new THREE.LineBasicMaterial({
            color: PALETTE.gold,
            transparent: true,
            opacity: 0.1
        });
        const line = new THREE.Line(geo, mat);
        line.userData = { originalOpacity: 0.1, pulsePhase: Math.random() * Math.PI * 2 };
        goldenSphere.add(line);
    }
    
    // 5. LUCE CENTRALE POTENTE
    const light = new THREE.PointLight(PALETTE.gold, 3, 30);
    goldenSphere.add(light);
    
    scene.add(goldenSphere);
}

// COSTELLAZIONE NEURALE (120 neuroni orbitali)
function createNeuralConstellation() {
    constellation = new THREE.Group();
    
    const positions = generateFibonacciSphere(120, 7); // Raggio 7, fuori dalla sfera d'oro
    
    positions.forEach((pos, i) => {
        const neuronGroup = new THREE.Group();
        neuronGroup.position.copy(pos);
        neuronGroup.lookAt(0, 0, 0);
        
        // Nucleo del neurone (glow intenso)
        const coreGeo = new THREE.SphereGeometry(0.08, 16, 16);
        const coreMat = new THREE.MeshBasicMaterial({
            color: PALETTE.cyan,
            transparent: true,
            opacity: 0.8
        });
        const core = new THREE.Mesh(coreGeo, coreMat);
        neuronGroup.add(core);
        
        // Corona dorata (anello attorno)
        const ringGeo = new THREE.RingGeometry(0.12, 0.15, 16);
        const ringMat = new THREE.MeshBasicMaterial({
            color: PALETTE.gold,
            transparent: true,
            opacity: 0.6,
            side: THREE.DoubleSide,
            blending: THREE.AdditiveBlending
        });
        const ring = new THREE.Mesh(ringGeo, ringMat);
        ring.lookAt(0, 0, 0);
        neuronGroup.add(ring);
        
        // Sprite glow
        const spriteMat = new THREE.SpriteMaterial({
            color: PALETTE.cyan,
            transparent: true,
            opacity: 0.5,
            blending: THREE.AdditiveBlending
        });
        const sprite = new THREE.Sprite(spriteMat);
        sprite.scale.set(0.5, 0.5, 1);
        neuronGroup.add(sprite);
        
        // Metadata
        neuronGroup.userData = {
            id: i,
            name: getNeuronName(i),
            core: core,
            ring: ring,
            sprite: sprite,
            basePos: pos.clone()
        };
        
        neurons.push(neuronGroup);
        constellation.add(neuronGroup);
    });
    
    // Connetti neuroni vicini con linee sottili
    createNeuralConnections();
    
    scene.add(constellation);
}

function createNeuralConnections() {
    const material = new THREE.LineBasicMaterial({
        color: PALETTE.gold,
        transparent: true,
        opacity: 0.05,
        blending: THREE.AdditiveBlending
    });
    
    for (let i = 0; i < neurons.length; i++) {
        for (let j = i + 1; j < neurons.length; j++) {
            const dist = neurons[i].position.distanceTo(neurons[j].position);
            if (dist < 2.5) {
                const geo = new THREE.BufferGeometry().setFromPoints([
                    neurons[i].position,
                    neurons[j].position
                ]);
                const line = new THREE.Line(geo, material.clone());
                line.userData = { from: i, to: j, active: false };
                neuralConnections.push(line);
                scene.add(line);
            }
        }
    }
}

// NEBULA DI PARTICELLE (la polvere dorata che fluttua)
function createNebulaParticles() {
    const particleCount = 3000;
    const geometry = new THREE.BufferGeometry();
    const positions = [];
    const colors = [];
    const sizes = [];
    
    for (let i = 0; i < particleCount; i++) {
        // Distribuzione sferica con bias verso il centro
        const radius = 5 + Math.random() * 10;
        const theta = Math.random() * Math.PI * 2;
        const phi = Math.acos(2 * Math.random() - 1);
        
        const x = radius * Math.sin(phi) * Math.cos(theta);
        const y = radius * Math.sin(phi) * Math.sin(theta);
        const z = radius * Math.cos(phi);
        
        positions.push(x, y, z);
        
        // Colori: oro, ambra, bianco sporco
        const colorType = Math.random();
        if (colorType < 0.5) {
            colors.push(1, 0.72, 0); // Gold
        } else if (colorType < 0.8) {
            colors.push(1, 0.55, 0); // Amber
        } else {
            colors.push(1, 0.97, 0.9); // Warm white
        }
        
        sizes.push(Math.random() * 2);
    }
    
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
    geometry.setAttribute('size', new THREE.Float32BufferAttribute(sizes, 1));
    
    const material = new THREE.PointsMaterial({
        size: 0.15,
        vertexColors: true,
        transparent: true,
        opacity: 0.6,
        blending: THREE.AdditiveBlending,
        sizeAttenuation: true
    });
    
    particleNebula = new THREE.Points(geometry, material);
    scene.add(particleNebula);
}

// Particelle ambientali extra (polvere)
function createAmbientParticles() {
    const geometry = new THREE.BufferGeometry();
    const positions = [];
    
    for (let i = 0; i < 1000; i++) {
        const r = 15 + Math.random() * 20;
        const theta = Math.random() * Math.PI * 2;
        const phi = Math.acos(2 * Math.random() - 1);
        
        positions.push(
            r * Math.sin(phi) * Math.cos(theta),
            r * Math.sin(phi) * Math.sin(theta),
            r * Math.cos(phi)
        );
    }
    
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    
    const material = new THREE.PointsMaterial({
        color: 0xffb700,
        size: 0.08,
        transparent: true,
        opacity: 0.3,
        blending: THREE.AdditiveBlending
    });
    
    const ambient = new THREE.Points(geometry, material);
    scene.add(ambient);
}

// Utility: sfera di Fibonacci (distribuzione uniforme)
function generateFibonacciSphere(n, radius) {
    const positions = [];
    const phi = Math.PI * (3 - Math.sqrt(5));
    
    for (let i = 0; i < n; i++) {
        const y = 1 - (i / (n - 1)) * 2;
        const r = Math.sqrt(1 - y * y);
        const theta = phi * i;
        
        positions.push(new THREE.Vector3(
            Math.cos(theta) * r * radius,
            y * radius,
            Math.sin(theta) * r * radius
        ));
    }
    return positions;
}

function getNeuronName(idx) {
    const names = [
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
    ];
    return names[idx] || `Neurone #${idx}`;
}

// WebSocket
function connectWebSocket() {
    const wsUrl = `ws://${window.location.host}/ws`;
    ws = new WebSocket(wsUrl);
    
    ws.onopen = () => {
        log("Neural link established [GOLDEN SPHERE ACTIVE]");
        document.getElementById('soma-state').textContent = "ONLINE";
        document.getElementById('soma-state').style.color = "#ffb700";
    };
    
    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        updateNeurons(data);
    };
    
    ws.onclose = () => {
        log("Neural link severed. Reconnecting...");
        setTimeout(connectWebSocket, 2000);
    };
}

function updateNeurons(data) {
    if (!data.potentials || !data.spikes) return;
    
    neuronStates.potentials = data.potentials;
    neuronStates.spikes = data.spikes;
    
    const spikeCount = data.spikes.filter(s => s).length;
    const avgPotential = data.potentials.reduce((a, b) => a + b, 0) / 120;
    const coherence = 1 - Math.min(
        data.potentials.reduce((sum, p) => sum + Math.pow(p - avgPotential, 2), 0) / 120 * 5,
        1
    );
    
    // Update HUD
    document.getElementById('spike-rate').textContent = `${(spikeCount * 2.5).toFixed(0)} Hz`;
    document.getElementById('coherence').textContent = coherence.toFixed(2);
    
    let affect = "CALM", color = "#00d9ff";
    if (spikeCount > 80) { affect = "OVERLOAD"; color = "#ff1744"; cameraShake = 0.5; }
    else if (spikeCount > 40) { affect = "EXCITED"; color = "#ff9100"; cameraShake = 0.2; }
    else if (coherence > 0.7) { affect = "FOCUSED"; color = "#76ff03"; cameraShake = 0.05; }
    else { cameraShake *= 0.95; }
    
    document.getElementById('affect').textContent = affect;
    document.getElementById('affect').style.color = color;
    
    // Update visuale
    updateVisuals(data.spikes, data.potentials, coherence, spikeCount);
}

function updateVisuals(spikes, potentials, coherence, spikeCount) {
    // Aggiorna sfera d'oro - intensità in base all'attività
    const activityLevel = spikeCount / 120;
    
    goldenSphere.children.forEach(child => {
        if (child.userData.rotationSpeed) {
            // Aumenta velocità rotazione con l'attività
            child.rotation.y += child.userData.rotationSpeed * (1 + activityLevel * 5);
            
            // Cambia opacità
            if (child.material.opacity !== undefined) {
                const targetOpacity = child.userData.originalOpacity * (1 + activityLevel);
                child.material.opacity = Math.min(targetOpacity, 0.8);
            }
        }
        
        // Pulse delle linee radiali
        if (child.userData.pulsePhase !== undefined) {
            child.userData.pulsePhase += 0.05 + activityLevel * 0.2;
            child.material.opacity = child.userData.originalOpacity * 
                (0.5 + 0.5 * Math.sin(child.userData.pulsePhase));
        }
    });
    
    // Aggiorna neuroni
    neurons.forEach((neuron, i) => {
        const { core, ring, sprite } = neuron.userData;
        
        if (spikes[i]) {
            // SPIKE!
            core.material.color.setHex(PALETTE.coreWhite);
            core.scale.set(3, 3, 3);
            ring.material.color.setHex(PALETTE.coreWhite);
            ring.material.opacity = 1;
            sprite.material.color.setHex(PALETTE.coreWhite);
            sprite.scale.set(1.5, 1.5, 1);
            sprite.material.opacity = 0.9;
        } else {
            const p = potentials[i];
            core.scale.lerp(new THREE.Vector3(1, 1, 1), 0.1);
            
            if (p > 0.5) {
                core.material.color.setHex(PALETTE.cyan);
                sprite.material.color.setHex(PALETTE.cyan);
            } else {
                core.material.color.setHex(0x004466);
            }
            
            ring.material.color.setHex(PALETTE.gold);
            ring.material.opacity = 0.6;
            sprite.scale.lerp(new THREE.Vector3(0.5, 0.5, 1), 0.1);
            sprite.material.opacity = 0.5;
        }
    });
    
    // Aggiorna connessioni
    neuralConnections.forEach(conn => {
        if (spikes[conn.userData.from] || spikes[conn.userData.to]) {
            conn.material.opacity = 0.4;
            conn.material.color.setHex(PALETTE.gold);
        } else {
            conn.material.opacity = Math.max(0.05, conn.material.opacity * 0.95);
            if (conn.material.opacity < 0.1) {
                conn.material.color.setHex(PALETTE.bronze);
            }
        }
    });
    
    // Ruota nebula più velocemente con l'attività
    if (particleNebula) {
        particleNebula.rotation.y += 0.0005 + activityLevel * 0.002;
    }
}

// Mouse interaction
function onMouseMove(event) {
    mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
    mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;
    
    raycaster.setFromCamera(mouse, camera);
    const cores = neurons.map(n => n.userData.core);
    const intersects = raycaster.intersectObjects(cores);
    
    if (intersects.length > 0) {
        const core = intersects[0].object;
        const neuron = core.parent;
        if (hoveredNeuron !== neuron) {
            hoveredNeuron = neuron;
            showNeuronInfo(neuron.userData);
            document.body.style.cursor = 'pointer';
        }
    } else {
        if (hoveredNeuron) {
            hoveredNeuron = null;
            hideNeuronInfo();
            document.body.style.cursor = 'default';
        }
    }
}

function onClick(event) {
    if (hoveredNeuron) {
        injectSpike([hoveredNeuron.userData.id]);
        log(`>> Spike: ${hoveredNeuron.userData.name}`);
    }
}

function showNeuronInfo(data) {
    const panel = document.getElementById('neuron-info');
    panel.classList.remove('hidden');
    document.getElementById('neuron-name').textContent = data.name;
    
    const p = neuronStates.potentials[data.id] || 0;
    const spiking = neuronStates.spikes[data.id];
    
    document.getElementById('neuron-potential').textContent = p.toFixed(3);
    document.getElementById('neuron-status').textContent = spiking ? '⚡ SPIKING ⚡' : (p > 0.5 ? 'CHARGING' : 'IDLE');
    document.getElementById('neuron-status').style.color = spiking ? '#ff1744' : (p > 0.5 ? '#ff9100' : '#00d9ff');
}

function hideNeuronInfo() {
    document.getElementById('neuron-info').classList.add('hidden');
}

function onWindowResize() {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
    composer.setSize(window.innerWidth, window.innerHeight);
}

// Animation
function animate() {
    requestAnimationFrame(animate);
    time += 0.01;
    
    controls.update();
    
    // Camera shake
    if (cameraShake > 0.001) {
        camera.position.x = (Math.random() - 0.5) * cameraShake;
        camera.position.y = (Math.random() - 0.5) * cameraShake;
        cameraShake *= 0.95;
    } else {
        camera.position.x *= 0.95;
        camera.position.y *= 0.95;
    }
    
    // Rotazione costellazione
    if (constellation) {
        constellation.rotation.y = Math.sin(time * 0.1) * 0.1;
    }
    
    // Pulsazione sfera centrale
    if (goldenSphere) {
        const breathe = 1 + Math.sin(time * 3) * 0.02;
        goldenSphere.scale.set(breathe, breathe, breathe);
    }
    
    composer.render();
}

// API
async function testSpike() { simulate(5); log(">> Random spike"); }
async function testPattern() { simulate(20); log(">> Pattern sync"); }
async function testMemory() { simulate(7); log(">> Memory recall"); }
async function testOverload() { 
    simulate(120); 
    cameraShake = 1; 
    log(">> ⚠️ OVERLOAD"); 
}

function simulate(count) {
    const indices = [];
    for (let i = 0; i < count; i++) indices.push(Math.floor(Math.random() * 120));
    
    indices.forEach(idx => {
        neuronStates.spikes[idx] = true;
        neuronStates.potentials[idx] = 1.0;
    });
    
    updateVisuals(neuronStates.spikes, neuronStates.potentials, 0.5, count);
    setTimeout(() => indices.forEach(idx => neuronStates.spikes[idx] = false), 300);
}

function toggleView() {
    const visible = constellation.visible;
    constellation.visible = !visible;
    neuralConnections.forEach(c => c.visible = !visible);
    log(visible ? ">> Soma view" : ">> Constellation view");
}

function injectSpike(indices) {
    if (ws?.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ action: "inject", indices }));
    }
}

function triggerKillSwitch() {
    log(">> 🔴 KILL-SWITCH 🔴");
    goldenSphere.children.forEach(c => {
        if (c.material?.emissive) c.material.emissive.setHex(0xff0000);
    });
    setTimeout(() => alert("◆ KILL-SWITCH ACTIVATED ◆\n\nIvy halted."), 100);
}

function log(msg) {
    const div = document.getElementById('status-log');
    const entry = document.createElement('div');
    entry.className = 'log-entry';
    entry.textContent = `[${new Date().toLocaleTimeString()}] ${msg}`;
    div.appendChild(entry);
    div.scrollTop = div.scrollHeight;
}

init();
