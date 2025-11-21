// RL Environment Interface for Flappy Bird
export class Environment {
    constructor(game) {
        this.game = game;
        this.actionSpace = {
            STAY: 0,
            JUMP: 1
        };
        this.previousScore = 0;
        // Full pipe sprite geometry: upper 17px, gap 12px (center offset = 17 + 6)
        this.tubeUpperHeight = 17;
    }

    reset() {
        this.game.reset();
        this.previousScore = 0;
        return this.getState();
    }

    getState() {
        const closestTube = this.game.getClosestTube();

        // State vector: [bird_y_speed, tube_x, diff_y, bird_y]
        const state = [
            this.game.birdYSpeed / 2.0,  // Normalize speed
            (closestTube.x - this.game.birdX) / this.game.width,  // Normalize x distance
            // With full-sprite tube, tube.y is sprite top; gap center is at (tube.y + 17 + gap/2)
            ((closestTube.y + this.tubeUpperHeight + this.game.tubeGap / 2) - (this.game.birdY + this.game.birdHeight / 2)) / this.game.height,  // Normalize y difference
            this.game.birdY / this.game.height  // Normalize bird y position
        ];

        return state;
    }

    step(action) {
        // Execute action
        if (action === this.actionSpace.JUMP) {
            this.game.jump();
        }
        // Track last action for reward shaping
        this.lastAction = action;

        // Update game
        this.game.update();

        // Calculate reward
        let reward = this.calculateReward();

        // Get new state
        const nextState = this.getState();

        // Check if episode is done
        const done = this.game.gameState === this.game.GAME_OVER;

        return {
            state: nextState,
            reward: reward,
            done: done
        };
    }

    calculateReward() {
        let reward = 0.1;  // Small reward for staying alive

        // Check if bird passed a tube
        if (this.game.score > this.previousScore) {
            reward = 5.0;  // Large reward for passing tube
            this.previousScore = this.game.score;
        }

        // Check if game over
        if (this.game.gameState === this.game.GAME_OVER) {
            reward = -10.0;  // Negative reward for dying

            // Additional penalty based on how the bird died
            const closestTube = this.game.getClosestTube();
            const diffY = (closestTube.y + this.tubeUpperHeight + this.game.tubeGap / 2) - (this.game.birdY + this.game.birdHeight / 2);

            // If bird was too high and jumped, extra penalty
            if (diffY > 0 && this.lastAction === this.actionSpace.JUMP) {
                reward -= 5.0;
            }
            // If bird was too low and didn't jump, extra penalty
            else if (diffY < 0 && this.lastAction === this.actionSpace.STAY) {
                reward -= 5.0;
            }
        } else {
            // Small penalty for being far from center of gap
            const closestTube = this.game.getClosestTube();
            const diffY = Math.abs((closestTube.y + this.tubeUpperHeight + this.game.tubeGap / 2) - (this.game.birdY + this.game.birdHeight / 2));
            reward -= diffY * 0.01;
        }

        return reward;
    }

    render() {
        this.game.render();
    }
}