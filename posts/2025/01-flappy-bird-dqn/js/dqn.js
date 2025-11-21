// Deep Q-Network Implementation
export class DQN {
    constructor(stateDim, actionDim, hiddenDim = 64) {
        this.stateDim = stateDim;
        this.actionDim = actionDim;
        this.hiddenDim = hiddenDim;

        // Build model
        this.model = this.buildModel();
    }

    buildModel() {
        const model = tf.sequential({
            layers: [
                tf.layers.dense({
                    units: this.hiddenDim,
                    inputShape: [this.stateDim],
                    activation: 'relu'
                }),
                tf.layers.dense({
                    units: this.hiddenDim * 2,
                    activation: 'relu'
                }),
                tf.layers.dense({
                    units: this.actionDim,
                    activation: 'linear'
                })
            ]
        });

        model.compile({
            optimizer: tf.train.adam(0.001),
            loss: 'meanSquaredError'
        });

        return model;
    }

    dispose() {
        if (this.model && typeof this.model.dispose === 'function') {
            this.model.dispose();
        }
    }

    predict(state) {
        return tf.tidy(() => {
            const stateTensor = tf.tensor2d(state, [state.length, this.stateDim]);
            return this.model.predict(stateTensor);
        });
    }

    async update(states, targets) {
        const stateTensor = tf.tensor2d(states, [states.length, this.stateDim]);
        const targetTensor = tf.tensor2d(targets, [targets.length, this.actionDim]);

        await this.model.fit(stateTensor, targetTensor, {
            epochs: 1,
            verbose: 0
        });

        stateTensor.dispose();
        targetTensor.dispose();
    }

    async save(name) {
        await this.model.save('localstorage://' + name);
    }

    async load(name) {
        this.model = await tf.loadLayersModel('localstorage://' + name);
        this.model.compile({
            optimizer: tf.train.adam(0.001),
            loss: 'meanSquaredError'
        });
    }

    getWeights() {
        return this.model.getWeights();
    }

    setWeights(weights) {
        this.model.setWeights(weights);
    }
}

// Double Deep Q-Network
export class DDQN extends DQN {
    constructor(stateDim, actionDim, hiddenDim = 64) {
        super(stateDim, actionDim, hiddenDim);

        // Create target network
        this.targetModel = this.buildModel();
        this.updateTarget();
    }

    predictTarget(state) {
        return tf.tidy(() => {
            const stateTensor = tf.tensor2d(state, [state.length, this.stateDim]);
            return this.targetModel.predict(stateTensor);
        });
    }

    updateTarget() {
        // Copy weights from main model to target model
        const weights = this.model.getWeights();
        const weightsCopy = weights.map(w => w.clone());
        this.targetModel.setWeights(weightsCopy);
    }

    dispose() {
        if (this.targetModel && typeof this.targetModel.dispose === 'function') {
            this.targetModel.dispose();
        }
        super.dispose();
    }
}