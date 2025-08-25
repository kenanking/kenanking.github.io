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
        // Load saved UI settings for model hidden dimension (if any)
        const savedInit = JSON.parse(localStorage.getItem('fb-dqn-ui') || '{}');
        this.hiddenDim = Number.isFinite(savedInit.hidden) ? savedInit.hidden : 64;
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

        // Setup UI
        this.setupUI();

        // Setup callbacks
        this.setupCallbacks();

        // Setup manual play input
        this.setupManualInput();

        // Start rendering
        this.startRendering();
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
        const saved = JSON.parse(localStorage.getItem('fb-dqn-ui') || '{}');
        const initSpeed = Number.isFinite(saved.speed) ? saved.speed : 50;
        const initEps = Number.isFinite(saved.eps) ? saved.eps : 30;
        const initEpNum = Number.isFinite(saved.episodes) ? saved.episodes : 10000;
        const initHidden = Number.isFinite(saved.hidden) ? saved.hidden : (this.hiddenDim || 64);
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

        // Events
        const persist = () => {
            localStorage.setItem('fb-dqn-ui', JSON.stringify({
                speed: this.trainingSpeed,
                eps: Math.round(this.agent.epsilon * 100),
                episodes: parseInt(document.getElementById('episodes-input').value) || 10000,
                hidden: this.hiddenDim
            }));
        };

        document.getElementById('start-btn').addEventListener('click', () => { this.startTraining(); persist(); });
        document.getElementById('stop-btn').addEventListener('click', () => this.stopTraining());
        document.getElementById('reset-btn').addEventListener('click', () => this.reset());
        // Export / Import
        const exportBtn = document.getElementById('export-btn');
        const importBtn = document.getElementById('import-btn');
        const fileInput = document.getElementById('model-file-input');
        const loadPretrainedBtn = document.getElementById('load-pretrained-btn');
        if (exportBtn) exportBtn.addEventListener('click', () => this.exportModel());
        if (importBtn) importBtn.addEventListener('click', () => this.importModel());
        if (fileInput) fileInput.addEventListener('change', (e) => this.handleFileImport(e));
        if (loadPretrainedBtn) loadPretrainedBtn.addEventListener('click', () => this.loadPretrainedModel());

        document.getElementById('speed-slider').addEventListener('input', (e) => {
            this.trainingSpeed = parseInt(e.target.value);
            document.getElementById('speed-value').textContent = `${this.trainingSpeed}ms`;
            // Apply immediately during training
            if (this.agent) {
                this.agent.trainingDelay = this.trainingSpeed;
            }
            persist();
        });

        document.getElementById('epsilon-slider').addEventListener('input', (e) => {
            const epsilon = parseInt(e.target.value) / 100;
            this.agent.epsilon = epsilon;
            document.getElementById('epsilon-value').textContent = epsilon.toFixed(2);
            document.getElementById('epsilon-bar').style.width = `${Math.round(epsilon * 100)}%`;
            // Reflect immediately in Training Status
            this.updateEpsilonDisplay(epsilon);
            persist();
        });

        // Hidden units change via combobox: persist, reset, remain stopped
        const hiddenSelect = document.getElementById('hidden-select');
        if (hiddenSelect) {
            hiddenSelect.addEventListener('change', (e) => {
                const newHidden = parseInt(e.target.value);
                if (!Number.isFinite(newHidden) || newHidden <= 0) return;
                if (this.hiddenDim === newHidden) return;
                this.hiddenDim = newHidden;
                persist();
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
        document.getElementById('episode').textContent = data.episode;
        document.getElementById('score').textContent = data.score;
        document.getElementById('epsilon').textContent = data.epsilon.toFixed(2);
        document.getElementById('memory-size').textContent = data.memorySize;

        const stats = this.agent.getStatistics();
        document.getElementById('avg-score').textContent = stats.avgScore.toFixed(2);
        document.getElementById('max-score').textContent = stats.maxScore;

        // visual meters
        const epsPct = Math.max(0, Math.min(100, Math.round(data.epsilon * 100)));
        document.getElementById('epsilon-bar').style.width = `${epsPct}%`;

        const memPct = Math.max(0, Math.min(100, Math.round((data.memorySize / this.agent.memoryMaxLen) * 100)));
        document.getElementById('memory-bar').style.width = `${memPct}%`;
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

    async startTraining() {
        if (this.isTraining) return;

        this.isTraining = true;
        this.isTrainingPaused = false;  // Clear pause state when starting
        document.getElementById('start-btn').disabled = true;
        document.getElementById('stop-btn').disabled = false;

        const episodes = parseInt(document.getElementById('episodes-input').value);

        // Update training state display
        const stateEl = document.getElementById('training-state');
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

        document.getElementById('start-btn').disabled = false;
        document.getElementById('stop-btn').disabled = true;

        // Update training state display
        const stateEl = document.getElementById('training-state');
        const startBtn = document.getElementById('start-btn');

        if (!silent && this.agent.isPaused && this.agent.currentEpisodeState) {
            // Training was paused mid-episode
            this.isTrainingPaused = true;  // Set pause flag
            startBtn.textContent = 'Start Training';
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
            startBtn.textContent = 'Start Training';
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

        // Reset model
        this.model = new DDQN(4, 2, this.hiddenDim);
        this.agent.model = this.model;

        // Reset environment and game to HOME state
        this.game.initializeGame();  // Use initializeGame to set HOME state, not PLAYING
        this.env.previousScore = 0;

        // Reset UI
        document.getElementById('start-btn').textContent = 'Start Training';
        const stateEl = document.getElementById('training-state');
        if (stateEl) {
            stateEl.textContent = '⏹️ Stopped';
            stateEl.style.color = '#666';
        }

        // Clear stats
        this.updateStats({
            episode: 0,
            score: 0,
            epsilon: 0.3,
            memorySize: 0,
            reward: 0,
            steps: 0
        });

        // Clear log and set placeholder
        document.getElementById('training-log').innerHTML = '<div class="log-entry">No logs yet. Start training to view progress.</div>';
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

        // Persist updated UI state
        localStorage.setItem('fb-dqn-ui', JSON.stringify({
            speed: this.trainingSpeed,
            eps: Math.round(this.agent.epsilon * 100),
            episodes: parseInt(document.getElementById('episodes-input').value) || 10000,
            hidden: this.hiddenDim
        }));

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

            // Fetch pretrained model JSON
            const response = await fetch('models/flappy-bird-model.json', { cache: 'no-cache' });
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            const modelData = await response.json();
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

            const dataStr = JSON.stringify(modelData);
            const dataBlob = new Blob([dataStr], { type: 'application/json' });
            const link = document.createElement('a');
            link.href = URL.createObjectURL(dataBlob);
            link.download = `flappy-bird-model-episode-${this.agent.episode}.json`;
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
                    const text = e.target && e.target.result ? e.target.result : '';
                    const modelData = JSON.parse(text);
                    // Reset first, then apply imported model and log
                    this.reset();
                    await this.applyModelData(modelData, { sourceLabel: `Imported: ${file.name}` });
                } catch (parseErr) {
                    console.error('Failed to parse model file:', parseErr);
                    alert('Invalid model file.');
                }
            };
            reader.readAsText(file);
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
                const ctx = this.canvas.getContext('2d');
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