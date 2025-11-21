// Reinforcement Learning Agent
export class Agent {
    constructor(env, model, config = {}) {
        this.env = env;
        this.model = model;

        // Hyperparameters
        this.gamma = config.gamma || 0.99;
        this.epsilon = config.epsilon || 0.3;
        this.epsilonDecay = config.epsilonDecay || 0.9995;
        this.epsilonMin = config.epsilonMin || 0.01;
        this.batchSize = config.batchSize || 32;
        this.memoryMaxLen = config.memoryMaxLen || 10000;
        this.targetUpdateFreq = config.targetUpdateFreq || 10;

        // Memory for experience replay
        this.memory = [];
        this.memoryIndex = 0;

        // Training state
        this.episode = 0;
        this.totalSteps = 0;
        this.isTraining = false;

        // State for pause/resume support
        this.currentEpisodeState = null;
        this.isPaused = false;

        // Statistics
        this.episodeRewards = [];
        this.episodeScores = [];
        this.episodeLengths = [];
    }

    act(state) {
        // Epsilon-greedy action selection
        if (Math.random() < this.epsilon) {
            // Exploration: random action
            if (Math.random() < 0.75) {
                return 0;  // Bias towards not jumping
            } else {
                return 1;  // Jump
            }
        } else {
            // Exploitation: choose best action based on Q-values
            return tf.tidy(() => {
                const qValues = this.model.predict([state]);
                const action = qValues.argMax(1).dataSync()[0];
                return action;
            });
        }
    }

    remember(state, action, reward, nextState, done) {
        const experience = {
            state: state,
            action: action,
            reward: reward,
            nextState: nextState,
            done: done
        };

        if (this.memory.length < this.memoryMaxLen) {
            this.memory.push(experience);
        } else {
            this.memory[this.memoryIndex % this.memoryMaxLen] = experience;
        }
        this.memoryIndex++;
    }

    async replay(isDDQN = false) {
        if (this.memory.length < this.batchSize) {
            return;
        }

        // Sample batch from memory
        const batch = this.sampleBatch(this.batchSize);

        const states = [];
        const targets = [];

        for (let experience of batch) {
            const { state, action, reward, nextState, done } = experience;

            states.push(state);

            // Calculate target Q-value
            const qValues = await tf.tidy(() => {
                return this.model.predict([state]).dataSync();
            });

            if (done) {
                qValues[action] = reward;
            } else {
                const nextQValues = await tf.tidy(() => {
                    if (isDDQN && this.model.predictTarget) {
                        // Use target network for DDQN
                        return this.model.predictTarget([nextState]).max(1).dataSync();
                    } else {
                        // Use main network for regular DQN
                        return this.model.predict([nextState]).max(1).dataSync();
                    }
                });

                qValues[action] = reward + this.gamma * nextQValues[0];
            }

            targets.push(Array.from(qValues));
        }

        // Update model
        await this.model.update(states, targets);
    }

    sampleBatch(size) {
        const batch = [];
        const indices = new Set();

        while (indices.size < size) {
            const index = Math.floor(Math.random() * this.memory.length);
            if (!indices.has(index)) {
                indices.add(index);
                batch.push(this.memory[index]);
            }
        }

        return batch;
    }

    async trainEpisode(isDDQN = false) {
        // Update target network for DDQN
        if (isDDQN && this.episode % this.targetUpdateFreq === 0 && this.model.updateTarget) {
            this.model.updateTarget();
        }

        // Check if resuming from a paused state
        let state, done, totalReward, steps;
        const maxSteps = 30000;

        if (this.currentEpisodeState && this.isPaused) {
            // Resume from saved state
            state = this.currentEpisodeState.state;
            done = this.currentEpisodeState.done;
            totalReward = this.currentEpisodeState.totalReward;
            steps = this.currentEpisodeState.steps;
            this.isPaused = false;
            console.log(`Resuming training from step ${steps}`);
        } else {
            // Start new episode
            state = this.env.reset();
            done = false;
            totalReward = 0;
            steps = 0;
        }

        while (!done && steps < maxSteps && this.isTraining) {
            // Choose action
            const action = this.act(state);
            this.env.lastAction = action;

            // Execute action
            const result = this.env.step(action);
            const nextState = result.state;
            const reward = result.reward;
            done = result.done;

            // Store experience
            this.remember(state, action, reward, nextState, done);

            // Update state
            state = nextState;
            totalReward += reward;
            steps++;
            this.totalSteps++;

            // Small delay for visualization
            if (this.trainingDelay > 0) {
                await this.sleep(this.trainingDelay);
            }

            // Check if training was stopped mid-episode
            if (!this.isTraining && !done) {
                // Save current state for resume
                this.currentEpisodeState = {
                    state: state,
                    done: done,
                    totalReward: totalReward,
                    steps: steps
                };
                this.isPaused = true;
                console.log(`Training paused at step ${steps}`);
                return null; // Return null to indicate incomplete episode
            }
        }

        // Clear saved state when episode completes
        this.currentEpisodeState = null;
        this.isPaused = false;

        // Experience replay
        await this.replay(isDDQN);

        // Update epsilon
        this.epsilon = Math.max(this.epsilonMin, this.epsilon * this.epsilonDecay);

        // Store statistics
        this.episodeRewards.push(totalReward);
        this.episodeScores.push(this.env.game.score);
        this.episodeLengths.push(steps);

        // Log progress
        if (this.logCallback) {
            this.logCallback({
                episode: this.episode,
                score: this.env.game.score,
                reward: totalReward,
                epsilon: this.epsilon,
                steps: steps,
                memorySize: this.memory.length
            });
        }

        this.episode++;

        return {
            episode: this.episode,
            score: this.env.game.score,
            reward: totalReward,
            steps: steps
        };
    }

    async train(episodes, isDDQN = false, delay = 0) {
        this.isTraining = true;
        this.trainingDelay = delay;

        // Calculate remaining episodes if resuming
        let remainingEpisodes = episodes;
        if (this.isPaused && this.currentEpisodeState) {
            // Don't count the interrupted episode
            remainingEpisodes = episodes;
        }

        for (let i = 0; i < remainingEpisodes && this.isTraining; i++) {
            const result = await this.trainEpisode(isDDQN);

            // If training was paused mid-episode, break the loop
            if (result === null) {
                break;
            }
        }

        this.isTraining = false;
    }

    stop() {
        this.isTraining = false;
    }

    reset() {
        // Reset all training state
        this.isTraining = false;
        this.currentEpisodeState = null;
        this.isPaused = false;
        this.episode = 0;
        this.totalSteps = 0;
        this.memory = [];
        this.memoryIndex = 0;
        this.episodeRewards = [];
        this.episodeScores = [];
        this.episodeLengths = [];
    }

    sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    getStatistics() {
        const recentScores = this.episodeScores.slice(-100);
        const recentRewards = this.episodeRewards.slice(-100);

        return {
            avgScore: recentScores.length > 0 ?
                recentScores.reduce((a, b) => a + b, 0) / recentScores.length : 0,
            maxScore: Math.max(...this.episodeScores, 0),
            avgReward: recentRewards.length > 0 ?
                recentRewards.reduce((a, b) => a + b, 0) / recentRewards.length : 0,
            totalEpisodes: this.episode,
            totalSteps: this.totalSteps
        };
    }
}