// Main entry point for Flappy Bird DQN
import { Game } from './game.js';
import { Environment } from './environment.js';
import { DDQN } from './dqn.js';
import { Agent } from './agent.js';

class FlappyBirdDQN {
    constructor() {
        this.canvas = document.getElementById('flappy-canvas');
        this.statusDiv = document.getElementById('training-status');
        this.controlsDiv = document.getElementById('controls-params');
        this.logDiv = document.getElementById('training-log-container');

        // Initialize components
        this.game = new Game(this.canvas);
        this.env = new Environment(this.game);
        // Default model hidden dimension
        this.hiddenDim = 64;
        this.model = new DDQN(4, 2, this.hiddenDim);  // state_dim=4, action_dim=2
        this.agent = new Agent(this.env, this.model, {
            gamma: 0.99,
            epsilon: 0.3,
            epsilonDecay: 0.9995,
            epsilonMin: 0.01,
            batchSize: 32,
            memoryMaxLen: 10000,
            targetUpdateFreq: 10
        });

        // Training state
        this.isTraining = false;
        this.isTrainingPaused = false;  // Track if training is paused
        this.trainingSpeed = 50;
        this.animationFrame = null;
        // Manual play cadence (ms per logic update). Larger value => slower forward speed.
        this.manualUpdateIntervalMs = 50;
        this._lastRenderTs = performance.now();
        this._accumulatedMs = 0;
        this.skipNextStopTrainingCallback = false;

        // Cache canvas 2D context
        this.ctx = this.canvas.getContext('2d');
        // Track total episodes for status display (current/total)
        this.totalEpisodes = 10000;

        // Setup UI
        this.setupUI();

        // Setup callbacks
        this.setupCallbacks();

        // Setup manual play input
        this.setupManualInput();

        // Start rendering
        this.startRendering();
    }

    clamp(value, min, max) {
        if (!Number.isFinite(value)) return min;
        return Math.min(max, Math.max(min, value));
    }

    setControlButtonLabel(buttonId, newText) {
        const btn = document.getElementById(buttonId);
        if (!btn) return;
        const label = btn.querySelector('.btn-label');
        if (label) {
            label.textContent = newText;
        } else {
            btn.textContent = newText;
        }
    }

    setupUI() {
        // --- Status card ---
        this.statusDiv.classList.add('card');
        this.statusDiv.innerHTML = `
          <h3>Training Status</h3>
          <div class="stats">
            <div class="stat-row">
              <span class="label">Status:</span>
              <span class="value" id="training-state" style="font-weight: bold; color: #666;">⏹️ Stopped</span>
            </div>
            <div class="stat-row"><span class="label">Episode:</span><span class="value" id="episode">0</span></div>
            <div class="stat-row"><span class="label">Score:</span><span class="value" id="score">0</span></div>
            <div class="stat-row">
              <span class="label">Epsilon:</span>
              <span class="value"><span id="epsilon">0.30</span></span>
            </div>
            <div class="meter" title="Exploration ratio"><span id="epsilon-bar" style="width:30%"></span></div>
      
            <div class="stat-row"><span class="label">Memory Size:</span><span class="value" id="memory-size">0</span></div>
            <div class="meter" title="Replay buffer usage"><span id="memory-bar" style="width:0%"></span></div>
      
            <div class="stat-row"><span class="label">Avg Score (100 ep):</span><span class="value" id="avg-score">0.00</span></div>
            <div class="stat-row"><span class="label">Max Score:</span><span class="value" id="max-score">0</span></div>
          </div>
        `;

        // --- Controls / params card ---
        this.controlsDiv.classList.add('card');
        const initSpeed = 50;
        const initEps = 30;
        const initHidden = this.hiddenDim || 64;
        const initEpsFloat = (initEps / 100).toFixed(2);

        this.trainingSpeed = initSpeed;

        this.controlsDiv.innerHTML = `
            <h3>Parameters</h3>
            <div class="parameters">
            <div class="param-row">
                <label for="speed-slider">Speed:</label>
                <input type="range" id="speed-slider" min="0" max="100" value="${initSpeed}" aria-label="Training speed">
                <span id="speed-value">${initSpeed}ms</span>
            </div>
            <div class="param-row">
                <label for="epsilon-slider">Epsilon:</label>
                <input type="range" id="epsilon-slider" min="0" max="100" value="${initEps}" aria-label="Exploration epsilon">
                <span id="epsilon-value">${initEpsFloat}</span>
            </div>
            <div class="param-row">
                <label for="hidden-select">Hidden:</label>
                <select id="hidden-select" aria-label="Hidden units">
                    <option value="32" ${initHidden === 32 ? 'selected' : ''}>32</option>
                    <option value="64" ${initHidden === 64 ? 'selected' : ''}>64</option>
                    <option value="128" ${initHidden === 128 ? 'selected' : ''}>128</option>
                    <option value="256" ${initHidden === 256 ? 'selected' : ''}>256</option>
                </select>
            </div>
            <div class="param-row">
                <label for="episodes-input">Episodes:</label>
                <input type="number" id="episodes-input" value="10000" min="100" max="50000" step="20" aria-label="Episodes">
            </div>
            </div>
        `;

        // --- Log card ---
        this.logDiv.classList.add('card');
        this.logDiv.innerHTML = `
            <h3>Training Log</h3>
            <div id="training-log" class="training-log">
                <div class="log-entry">No logs yet. Start training to view progress.</div>
            </div>
        `;

        // Cache frequently used DOM nodes
        this.dom = {
            startBtn: document.getElementById('start-btn'),
            stopBtn: document.getElementById('stop-btn'),
            resetBtn: document.getElementById('reset-btn'),
            exportBtn: document.getElementById('export-btn'),
            importBtn: document.getElementById('import-btn'),
            modelFileInput: document.getElementById('model-file-input'),
            loadPretrainedBtn: document.getElementById('load-pretrained-btn'),

            speedSlider: document.getElementById('speed-slider'),
            speedValue: document.getElementById('speed-value'),
            epsilonSlider: document.getElementById('epsilon-slider'),
            epsilonValue: document.getElementById('epsilon-value'),
            episodesInput: document.getElementById('episodes-input'),
            hiddenSelect: document.getElementById('hidden-select'),

            trainingState: document.getElementById('training-state'),
            episodeLabel: document.getElementById('episode'),
            scoreLabel: document.getElementById('score'),
            epsilonLabel: document.getElementById('epsilon'),
            memorySizeLabel: document.getElementById('memory-size'),
            avgScoreLabel: document.getElementById('avg-score'),
            maxScoreLabel: document.getElementById('max-score'),
            epsilonBar: document.getElementById('epsilon-bar'),
            memoryBar: document.getElementById('memory-bar'),

            trainingLog: document.getElementById('training-log')
        };

        if (this.dom.episodesInput) {
            const total = this.clamp(parseInt(this.dom.episodesInput.value), 100, 50000);
            this.totalEpisodes = total;
            this.dom.episodesInput.value = String(total);
        }
        if (this.dom.episodeLabel) {
            this.dom.episodeLabel.textContent = `0/${this.totalEpisodes}`;
        }

        // Events
        if (this.dom.startBtn) this.dom.startBtn.addEventListener('click', () => { this.startTraining(); });
        if (this.dom.stopBtn) this.dom.stopBtn.addEventListener('click', () => this.stopTraining());
        if (this.dom.resetBtn) this.dom.resetBtn.addEventListener('click', () => this.reset());
        // Export / Import
        if (this.dom.exportBtn) this.dom.exportBtn.addEventListener('click', () => this.exportModel());
        if (this.dom.importBtn) this.dom.importBtn.addEventListener('click', () => this.importModel());
        if (this.dom.modelFileInput) {
            this.dom.modelFileInput.setAttribute('accept', '.json.gz,application/gzip');
            this.dom.modelFileInput.addEventListener('change', (e) => this.handleFileImport(e));
        }
        if (this.dom.loadPretrainedBtn) this.dom.loadPretrainedBtn.addEventListener('click', () => this.loadPretrainedModel());

        if (this.dom.speedSlider) this.dom.speedSlider.addEventListener('input', (e) => {
            const raw = parseInt(e.target.value);
            this.trainingSpeed = this.clamp(raw, 0, 100);
            if (this.dom.speedSlider) this.dom.speedSlider.value = String(this.trainingSpeed);
            if (this.dom.speedValue) this.dom.speedValue.textContent = `${this.trainingSpeed}ms`;
            // Apply immediately during training
            if (this.agent) {
                this.agent.trainingDelay = this.trainingSpeed;
            }
        });

        if (this.dom.epsilonSlider) this.dom.epsilonSlider.addEventListener('input', (e) => {
            const raw = this.clamp(parseInt(e.target.value), 0, 100);
            if (this.dom.epsilonSlider) this.dom.epsilonSlider.value = String(raw);
            const epsilon = raw / 100;
            this.agent.epsilon = epsilon;
            if (this.dom.epsilonValue) this.dom.epsilonValue.textContent = epsilon.toFixed(2);
            if (this.dom.epsilonBar) this.dom.epsilonBar.style.width = `${Math.round(epsilon * 100)}%`;
            // Reflect immediately in Training Status
            this.updateEpsilonDisplay(epsilon);
        });

        if (this.dom.episodesInput) this.dom.episodesInput.addEventListener('input', (e) => {
            const total = this.clamp(parseInt(e.target.value), 100, 50000);
            this.totalEpisodes = total;
            this.dom.episodesInput.value = String(total);
            if (this.dom.episodeLabel && !this.isTraining) {
                this.dom.episodeLabel.textContent = `0/${this.totalEpisodes}`;
            }
        });

        // Hidden units change via combobox: reset and remain stopped
        const hiddenSelect = this.dom.hiddenSelect;
        if (hiddenSelect) {
            hiddenSelect.addEventListener('change', (e) => {
                const newHidden = parseInt(e.target.value);
                if (!Number.isFinite(newHidden) || newHidden <= 0) return;
                if (this.hiddenDim === newHidden) return;
                this.hiddenDim = newHidden;
                // Reset model and UI to stopped state; user manually starts training
                this.reset();
            });
        }
    }

    setupCallbacks() {
        // Log callback
        this.agent.logCallback = (data) => {
            this.updateStats(data);
            this.addLogEntry(data);
        };
    }

    setupManualInput() {
        const onJump = () => {
            // Block input if training is active or paused
            if (this.isTraining || this.isTrainingPaused) return;
            // In HOME state, jump() will start the game; in PLAYING, it boosts; in GAME_OVER, ignore
            this.game.jump();
        };

        const onKey = (e) => {
            // Block input if training is active or paused
            if (this.isTraining || this.isTrainingPaused) return;
            if (e.code === 'Space' || e.code === 'ArrowUp') {
                e.preventDefault();
                // If game over, go back to HOME; only HOME click/press starts game
                if (this.game.gameState === this.game.GAME_OVER) {
                    this.game.initializeGame();
                    return;
                }
                onJump();
            } else if (e.code === 'KeyR') {
                e.preventDefault();
                // Allow reset even when paused (will clear the pause state)
                if (this.isTrainingPaused) {
                    this.reset();
                } else {
                    this.game.initializeGame();
                }
            }
        };

        const onClick = (e) => {
            // Block input if training is active or paused
            if (this.isTraining || this.isTrainingPaused) return;
            if (this.game.gameState === this.game.GAME_OVER) {
                this.game.initializeGame();
                return;
            }
            onJump();
        };

        // Attach listeners
        window.addEventListener('keydown', onKey);
        this.canvas.addEventListener('mousedown', onClick);
        this.canvas.addEventListener('touchstart', (e) => { e.preventDefault(); onClick(e); }, { passive: false });
    }

    updateStats(data) {
        if (this.dom && this.dom.episodeLabel) this.dom.episodeLabel.textContent = `${data.episode}/${this.totalEpisodes}`;
        if (this.dom && this.dom.scoreLabel) this.dom.scoreLabel.textContent = data.score;
        if (this.dom && this.dom.epsilonLabel) this.dom.epsilonLabel.textContent = data.epsilon.toFixed(2);
        if (this.dom && this.dom.memorySizeLabel) this.dom.memorySizeLabel.textContent = data.memorySize;

        const stats = this.agent.getStatistics();
        if (this.dom && this.dom.avgScoreLabel) this.dom.avgScoreLabel.textContent = stats.avgScore.toFixed(2);
        if (this.dom && this.dom.maxScoreLabel) this.dom.maxScoreLabel.textContent = stats.maxScore;

        // visual meters
        const epsPct = Math.max(0, Math.min(100, Math.round(data.epsilon * 100)));
        if (this.dom && this.dom.epsilonBar) this.dom.epsilonBar.style.width = `${epsPct}%`;
        if (this.dom && this.dom.epsilonSlider) this.dom.epsilonSlider.value = String(epsPct);
        if (this.dom && this.dom.epsilonValue) this.dom.epsilonValue.textContent = data.epsilon.toFixed(2);

        const memPct = Math.max(0, Math.min(100, Math.round((data.memorySize / this.agent.memoryMaxLen) * 100)));
        if (this.dom && this.dom.memoryBar) this.dom.memoryBar.style.width = `${memPct}%`;
    }

    addLogEntry(data) {
        const log = document.getElementById('training-log');
        // Clear placeholder if it's the first entry
        if (log.innerHTML.includes('No logs yet')) {
            log.innerHTML = '';
        }
        const entry = document.createElement('div');
        entry.className = 'log-entry';

        // Check if this is a model loaded notification
        if (data.isModelLoaded) {
            entry.className += ' loaded';
            entry.style.color = '#4caf50';
            entry.textContent = data.message || `📦 Loaded pretrained model (hidden=${this.hiddenDim}, epsilon=${this.agent.epsilon.toFixed(2)})`;
        } else if (data.isPaused) {
            entry.className += ' paused';
            entry.style.color = '#ff9800';
            entry.textContent = `⏸️ Training paused at step ${data.steps} of episode ${data.episode + 1}`;
        } else {
            entry.textContent = `Episode ${data.episode}: Score=${data.score}, Reward=${data.reward.toFixed(2)}, Steps=${data.steps}`;
        }

        log.insertBefore(entry, log.firstChild);

        // Keep only last 10 entries
        while (log.children.length > 10) {
            log.removeChild(log.lastChild);
        }
    }

    // Gzip helpers using browser CompressionStream/DecompressionStream
    async compressJSON(data) {
        if (typeof CompressionStream === 'undefined') {
            throw new Error('CompressionStream is not supported in this browser.');
        }
        const jsonText = JSON.stringify(data);
        const cs = new CompressionStream('gzip');
        const compressedStream = new Blob([jsonText]).stream().pipeThrough(cs);
        const rawBlob = await new Response(compressedStream).blob();
        return new Blob([rawBlob], { type: 'application/gzip' });
    }

    async decompressJSONFromBlob(blob) {
        if (typeof DecompressionStream === 'undefined') {
            throw new Error('DecompressionStream is not supported in this browser.');
        }
        const ds = new DecompressionStream('gzip');
        const decompressedStream = blob.stream().pipeThrough(ds);
        const text = await new Response(decompressedStream).text();
        return JSON.parse(text);
    }

    async decompressJSONFromArrayBuffer(buffer) {
        return this.decompressJSONFromBlob(new Blob([buffer]));
    }

    async startTraining() {
        if (this.isTraining) return;

        this.isTraining = true;
        this.isTrainingPaused = false;  // Clear pause state when starting
        if (this.dom && this.dom.startBtn) this.dom.startBtn.disabled = true;
        if (this.dom && this.dom.stopBtn) this.dom.stopBtn.disabled = false;

        // Clamp and read total episodes
        if (this.dom && this.dom.episodesInput) {
            this.totalEpisodes = this.clamp(parseInt(this.dom.episodesInput.value), 100, 50000);
            this.dom.episodesInput.value = String(this.totalEpisodes);
        }
        const episodes = this.totalEpisodes;

        // Update training state display
        const stateEl = (this.dom && this.dom.trainingState) || document.getElementById('training-state');
        if (stateEl) {
            stateEl.textContent = '▶️ Training';
            stateEl.style.color = '#4caf50';
        }

        // Start training in background (will resume if paused)
        this.agent.train(episodes, true, this.trainingSpeed).then(() => {
            if (this.skipNextStopTrainingCallback) {
                this.skipNextStopTrainingCallback = false;
                return;
            }
            this.stopTraining();
        });
    }

    stopTraining(options = {}) {
        const { silent = false } = options;
        this.isTraining = false;
        this.agent.stop();

        if (this.dom && this.dom.startBtn) this.dom.startBtn.disabled = false;
        if (this.dom && this.dom.stopBtn) this.dom.stopBtn.disabled = true;

        // Update training state display
        const stateEl = (this.dom && this.dom.trainingState) || document.getElementById('training-state');
        if (!silent && this.agent.isPaused && this.agent.currentEpisodeState) {
            // Training was paused mid-episode
            this.isTrainingPaused = true;  // Set pause flag
            this.setControlButtonLabel('start-btn', 'Start Training');
            if (stateEl) {
                stateEl.textContent = `⏸️ Paused (Step ${this.agent.currentEpisodeState.steps})`;
                stateEl.style.color = '#ff9800';
            }
            // Add visual indicator that training is paused
            this.addLogEntry({
                episode: this.agent.episode,
                score: this.env.game.score,
                reward: this.agent.currentEpisodeState.totalReward,
                steps: this.agent.currentEpisodeState.steps,
                isPaused: true
            });
        } else {
            // Training completed normally
            this.isTrainingPaused = false;  // Clear pause flag
            this.setControlButtonLabel('start-btn', 'Start Training');
            if (stateEl) {
                stateEl.textContent = '⏹️ Stopped';
                stateEl.style.color = '#666';
            }
        }
    }

    reset() {
        // Ensure any pending stop callback from an in-flight training won't override this reset UI
        this.skipNextStopTrainingCallback = true;
        this.stopTraining({ silent: true });
        this.isTrainingPaused = false;  // Clear pause flag on reset

        // Use the new reset method in agent
        this.agent.reset();
        this.agent.epsilon = 0.3; // Reset epsilon to initial value

        // Reset model (dispose old first if available)
        if (this.model && typeof this.model.dispose === 'function') {
            try { this.model.dispose(); } catch (e) { /* no-op */ }
        }
        this.model = new DDQN(4, 2, this.hiddenDim);
        this.agent.model = this.model;

        // Reset environment and game to HOME state
        this.game.initializeGame();  // Use initializeGame to set HOME state, not PLAYING
        this.env.previousScore = 0;

        // Reset UI
        this.setControlButtonLabel('start-btn', 'Start Training');
        const stateEl = (this.dom && this.dom.trainingState) || document.getElementById('training-state');
        if (stateEl) {
            stateEl.textContent = '⏹️ Stopped';
            stateEl.style.color = '#666';
        }

        // Clear stats and sync epsilon controls back to default
        if (this.dom && this.dom.epsilonSlider) this.dom.epsilonSlider.value = '30';
        if (this.dom && this.dom.epsilonValue) this.dom.epsilonValue.textContent = '0.30';
        this.updateEpsilonDisplay(0.3);
        // Reset episode label to 0/total (use current input if available)
        if (this.dom && this.dom.episodesInput) {
            this.totalEpisodes = this.clamp(parseInt(this.dom.episodesInput.value), 100, 50000);
            this.dom.episodesInput.value = String(this.totalEpisodes);
        }
        if (this.dom && this.dom.episodeLabel) this.dom.episodeLabel.textContent = `0/${this.totalEpisodes}`;

        this.updateStats({
            episode: 0,
            score: 0,
            epsilon: 0.3,
            memorySize: 0,
            reward: 0,
            steps: 0
        });

        // Clear log and set placeholder
        if (this.dom && this.dom.trainingLog) this.dom.trainingLog.innerHTML = '<div class="log-entry">No logs yet. Start training to view progress.</div>';
    }

    async applyModelData(modelData, options = {}) {
        if (!modelData || !modelData.weights || !Array.isArray(modelData.weights)) {
            throw new Error('Invalid model file format');
        }

        const { sourceLabel } = options; // e.g., 'Pretrained' or `Imported: filename.json`

        // Read config
        const cfg = modelData.config || {};
        const loadedHidden = Number.isFinite(cfg.hiddenDim) ? cfg.hiddenDim : 64;
        const loadedEpsilon = Number.isFinite(cfg.epsilon) ? cfg.epsilon : this.agent.epsilon;

        // Apply Speed reset to 50
        this.trainingSpeed = 50;
        this.agent.trainingDelay = 50;
        const speedSlider = document.getElementById('speed-slider');
        const speedValue = document.getElementById('speed-value');
        if (speedSlider) speedSlider.value = '50';
        if (speedValue) speedValue.textContent = '50ms';

        // Apply Hidden from model
        this.hiddenDim = loadedHidden;
        const hiddenSelectEl = document.getElementById('hidden-select');
        if (hiddenSelectEl) hiddenSelectEl.value = String(loadedHidden);

        // Recreate model with loadedHidden and set weights
        this.model = new DDQN(4, 2, this.hiddenDim);
        const tensors = modelData.weights.map(w => tf.tensor(w));
        try {
            this.model.model.setWeights(tensors);
        } finally {
            // Dispose created tensors to avoid memory leaks
            tensors.forEach(t => t.dispose());
        }
        if (this.model.updateTarget) {
            this.model.updateTarget();
        }
        // Reattach to agent
        this.agent.model = this.model;

        // Apply Epsilon from model
        this.agent.epsilon = loadedEpsilon;
        const epsSlider = document.getElementById('epsilon-slider');
        const epsValue = document.getElementById('epsilon-value');
        if (epsSlider) epsSlider.value = String(Math.round(loadedEpsilon * 100));
        if (epsValue) epsValue.textContent = loadedEpsilon.toFixed(2);
        this.updateEpsilonDisplay(loadedEpsilon);

        // Update stats panel
        this.updateStats({
            episode: this.agent.episode,
            score: 0,
            epsilon: this.agent.epsilon,
            memorySize: this.agent.memory.length,
            reward: 0,
            steps: 0
        });

        // Add log entry for model loading
        const baseMsg = `📦 ${sourceLabel ? sourceLabel + ' ' : ''}model loaded (hidden=${this.hiddenDim}, epsilon=${this.agent.epsilon.toFixed(2)})`;
        this.addLogEntry({
            isModelLoaded: true,
            message: baseMsg
        });
    }

    async loadPretrainedModel() {
        try {
            // Reset to a clean state before loading weights
            this.reset();

            // Fetch pretrained model JSON.GZ and decompress
            const response = await fetch('models/flappy-bird-model.json.gz', { cache: 'no-cache' });
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            const buffer = await response.arrayBuffer();
            const modelData = await this.decompressJSONFromArrayBuffer(buffer);
            await this.applyModelData(modelData, { sourceLabel: 'Pretrained' });
        } catch (error) {
            console.error('Failed to load pretrained model:', error);
            alert('Failed to load pretrained model.');
        }
    }

    async exportModel() {
        try {
            const weightTensors = this.model.model.getWeights();
            const weights = weightTensors.map(w => w.arraySync());
            const modelData = {
                weights: weights,
                config: {
                    episode: this.agent.episode,
                    epsilon: this.agent.epsilon,
                    memorySize: this.agent.memory.length,
                    hiddenDim: this.hiddenDim,
                    statistics: this.agent.getStatistics(),
                    timestamp: new Date().toISOString()
                }
            };

            // Compress JSON payload using browser CompressionStream (gzip)
            const dataBlob = await this.compressJSON(modelData);
            const link = document.createElement('a');
            link.href = URL.createObjectURL(dataBlob);
            link.download = `flappy-bird-model-episode-${this.agent.episode}.json.gz`;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        } catch (error) {
            console.error('Failed to export model:', error);
            alert('Failed to export model.');
        }
    }

    importModel() {
        const fileInput = document.getElementById('model-file-input');
        if (fileInput) {
            fileInput.click();
        }
    }

    async handleFileImport(event) {
        const file = event.target.files && event.target.files[0];
        if (!file) return;
        try {
            const reader = new FileReader();
            reader.onload = async (e) => {
                try {
                    const buffer = e.target && e.target.result ? e.target.result : null;
                    if (!buffer) throw new Error('Empty file buffer');
                    const modelData = await this.decompressJSONFromArrayBuffer(buffer);
                    // Reset first, then apply imported model and log
                    this.reset();
                    await this.applyModelData(modelData, { sourceLabel: `Imported: ${file.name}` });
                } catch (parseErr) {
                    console.error('Failed to parse compressed model file:', parseErr);
                    alert('Invalid compressed model file.');
                }
            };
            reader.readAsArrayBuffer(file);
        } catch (err) {
            console.error('Import error:', err);
            alert('Failed to import model.');
        }
        // Clear input value to allow re-importing same file later
        event.target.value = '';
    }

    updateEpsilonDisplay(epsilon) {
        const epsLabel = document.getElementById('epsilon');
        if (epsLabel) epsLabel.textContent = epsilon.toFixed(2);
        const epsBar = document.getElementById('epsilon-bar');
        if (epsBar) epsBar.style.width = `${Math.round(epsilon * 100)}%`;
    }

    startRendering() {
        const render = () => {
            const now = performance.now();
            const dt = now - this._lastRenderTs;
            this._lastRenderTs = now;

            // Only update game logic if:
            // 1. Not training (manual play)
            // 2. AND not paused (not in a paused training state)
            if (!this.isTraining && !this.isTrainingPaused) {
                this._accumulatedMs += dt;
                while (this._accumulatedMs >= this.manualUpdateIntervalMs) {
                    this.game.update();
                    this._accumulatedMs -= this.manualUpdateIntervalMs;
                }
            }
            // Always render the current frame (frozen if paused)
            this.env.render();

            // Draw pause overlay if training is paused
            if (this.isTrainingPaused) {
                const ctx = this.ctx || this.canvas.getContext('2d');
                // Semi-transparent overlay
                ctx.fillStyle = 'rgba(0, 0, 0, 0.3)';
                ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);

                // Pause icon and text
                ctx.fillStyle = 'white';
                ctx.font = 'bold 24px Arial';
                ctx.textAlign = 'center';
                ctx.fillText('⏸️ PAUSED', this.canvas.width / 2, this.canvas.height / 2 - 20);
                ctx.font = '16px Arial';
                ctx.fillText('Click "Start Training" to continue', this.canvas.width / 2, this.canvas.height / 2 + 10);

                // Display current step info
                if (this.agent.currentEpisodeState) {
                    ctx.font = '14px Arial';
                    ctx.fillText(`Step: ${this.agent.currentEpisodeState.steps}`, this.canvas.width / 2, this.canvas.height / 2 + 35);
                }
            }

            this.animationFrame = requestAnimationFrame(render);
        };
        // Initialize timestamp before first loop
        this._lastRenderTs = performance.now();
        render();
    }

    stopRendering() {
        if (this.animationFrame) {
            cancelAnimationFrame(this.animationFrame);
        }
    }
}

// Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        window.flappyBirdDQN = new FlappyBirdDQN();
    });
} else {
    window.flappyBirdDQN = new FlappyBirdDQN();
}