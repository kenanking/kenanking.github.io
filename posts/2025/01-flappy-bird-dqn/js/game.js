// Flappy Bird Game Core
export class Game {
    constructor(canvas) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.scale = 10;
        this.width = 32;
        this.height = 32;
        this.canvas.width = this.width * this.scale;
        this.canvas.height = this.height * this.scale;

        // Game states
        this.HOME = 0;
        this.PLAYING = 1;
        this.GAME_OVER = 2;

        // Bird properties
        this.birdX = 5;
        this.birdY = 14;
        this.birdYSpeed = 0;
        this.birdWidth = 5;
        this.birdHeight = 3;
        this.birdFrame = 0;
        this.gravity = 0.25;  // Gravity force applied each frame

        // Tube properties
        this.tubes = [];
        this.tubeWidth = 6;
        this.tubeGap = 12;
        this.activeTube = 0;

        // Game state
        this.gameState = this.HOME;
        this.score = 0;
        this.groundX = 0;

        // Sprite locations
        this.bgLoc = { x: 0, y: 0, width: 32, height: 32 };
        this.groundLoc = { x: 0, y: 31, width: 35, height: 1 };
        this.birdLocs = [
            { x: 32, y: 0, width: 5, height: 3 },
            { x: 32, y: 3, width: 5, height: 3 },
            { x: 32, y: 6, width: 5, height: 3 }
        ];
        this.tubeLoc = { x: 0, y: 32, width: 6, height: 44 };

        // Load sprite sheet
        this.initSpriteSheet();

        // Offscreen canvases for pixel-perfect collision (base resolution 32x32)
        this.renderCanvas = document.createElement('canvas');
        this.renderCanvas.willReadFrequently = true;
        this.renderCanvas.width = this.width;
        this.renderCanvas.height = this.height;
        this.renderCtx = this.renderCanvas.getContext('2d', { willReadFrequently: true });
        // Draw behind existing content so tubes appear in front of bird for collision compare
        this.renderCtx.globalCompositeOperation = 'destination-over';

        this.collisionCanvas = document.createElement('canvas');
        this.collisionCanvas.willReadFrequently = true;
        this.collisionCanvas.width = this.width;
        this.collisionCanvas.height = this.height;
        this.collisionCtx = this.collisionCanvas.getContext('2d', { willReadFrequently: true });

        // Initialize game with HOME state
        this.initializeGame();
    }

    initSpriteSheet() {
        // Original sprite sheet base64 data
        const flappyBirdSource = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACUAAABQCAYAAACecbxxAAACY0lEQVRoge2XPW4CMRCF5yooLeegpIw4SZp0dBF34DBIKSMOkKQJUgpyBuQUyYIZv/mx1wsGraXR4vXu+Jv3Zheg59e3cM3YbqbH+Dl8BiIiujYUEVEHtZvPwm4+C01AERF9LB+On5uAas6+uKdG+27Oviafvu1mGlq0L40WoOLRFFR8HKFGqLuFau7pa/I91SRUk/aJjb4/vB+D3xivea/xrkMotOElA9r38r0K14y4wVuC6sYI5YY6RkNQpzFC3TTU02Ya1vtF+AqrcMnP6/0iWUvAuoXH5QQ/EXZY16N16R4din8FoHmcULqO50Pnzs5bUNKmfBMNNlaFQ8HzfaG0Y7Q5ylkOhRTwHNnmIlSxfaVHL1SsWBEUUo5bADdKbZKvrWEfggIWovNZ9qH3x6VChvKoxNclJapCeY5oc+2zBO2ybwgobb0alKTCVaHQRpaCWjHZUJYtsGKhmVX1HpeTwKH+wUwr+LoEZdhHyXUaVIVIlIJWpfDqKAXJyYfW9CTIIm1eqUg7Ceopo4/cRRYUdkpUE4qYMuDcnUD16amL2uesEqqTqfZ5IiOycvWFou6GHraJeXrkqlZdTdX/Et4tlMs+5QvZhCrpKV6UWOCA/2ZUxYqg0BPHq+XrTvVs1a1fnkhyVGFGn/VTqkkodPMQUFXti5PB5AKUES6oM+UcCuW8JvBwvBI81dWByYBCm+ZA5heU+w85Xtd6SpubfdgHKq5SmqOeM/uwApQ6d0KdjwGUKoHyK8X955ta6/yI7onniWJsmEoZVoj3I8USpQaAUmEGhbKsQHMElWWfEtZ6n6BfqpBLl8a8BXQAAAAASUVORK5CYII=";

        this.spriteSheetImage = new Image();
        this.spriteSheetImage.src = flappyBirdSource;

        // Create offscreen canvas for sprite sheet
        this.spriteSheetCanvas = document.createElement("canvas");
        this.spriteSheetCanvas.width = 37;
        this.spriteSheetCanvas.height = 80;
        this.spriteSheetContext = this.spriteSheetCanvas.getContext("2d");

        // When image loads, draw it to the sprite sheet canvas
        this.spriteSheetImage.onload = () => {
            this.spriteSheetContext.drawImage(this.spriteSheetImage, 0, 0);
            this.spriteLoaded = true;
        };
    }

    drawSpriteSheetImage(locRect, x, y, scale = this.scale) {
        if (!this.spriteLoaded) return;

        this.ctx.imageSmoothingEnabled = false;
        this.ctx.drawImage(
            this.spriteSheetImage,
            locRect.x, locRect.y, locRect.width, locRect.height,
            x * scale, y * scale, locRect.width * scale, locRect.height * scale
        );
    }

    // Helper to draw to a specific context (used by offscreen canvases)
    drawSpriteSheetImageTo(targetCtx, locRect, x, y, scale = 1) {
        if (!this.spriteLoaded) return;
        targetCtx.imageSmoothingEnabled = false;
        targetCtx.drawImage(
            this.spriteSheetImage,
            locRect.x, locRect.y, locRect.width, locRect.height,
            x * scale, y * scale, locRect.width * scale, locRect.height * scale
        );
    }

    reset() {
        this.birdY = 14;
        this.birdYSpeed = 0;
        this.score = 0;
        this.birdFrame = 0;
        this.gameState = this.PLAYING;
        this.resetTubes();
    }

    initializeGame() {
        this.birdY = 14;
        this.birdYSpeed = 0;
        this.score = 0;
        this.birdFrame = 0;
        this.gameState = this.HOME;
        this.resetTubes();
    }

    resetTubes() {
        this.tubes = [];
        for (let i = 0; i < 2; i++) {
            this.tubes[i] = {
                x: Math.round(48 + i * 19),
                y: 0
            };
            this.setTubeY(this.tubes[i]);
        }
    }

    setTubeY(tube) {
        // Using full-pipe sprite (44px tall) where:
        // upper pipe height = 17, gap = 12, lower pipe height = 15
        // Gap center within the sprite is at 17 + 6 = 23 from the sprite top
        // We pick a gap center in play area and back-compute sprite top (tube.y)
        const gapHalf = this.tubeGap / 2; // 6
        const minGapCenter = 4 + gapHalf; // keep away from extreme edges
        const maxGapCenter = this.height - 1 - 4 - gapHalf;
        const gapCenter = Math.floor(Math.random() * (maxGapCenter - minGapCenter + 1)) + minGapCenter;
        // Position the full sprite so its internal gap center aligns with chosen gapCenter
        const spriteGapCenterOffset = 17 + 6; // 23
        tube.y = Math.round(gapCenter - spriteGapCenterOffset);
    }

    jump() {
        if (this.gameState === this.HOME) {
            this.gameState = this.PLAYING;
            this.birdYSpeed = -1.4;
        } else if (this.gameState === this.PLAYING) {
            this.birdYSpeed = -1.4;
        }
    }

    update() {
        if (this.gameState === this.HOME) {
            // In HOME state, just animate the bird
            this.birdFrame++;
            this.birdFrame %= 3;
            return;
        }

        if (this.gameState !== this.PLAYING) return;

        // Update bird physics
        this.birdY = Math.round(this.birdY + this.birdYSpeed);
        this.birdYSpeed += this.gravity;  // Apply gravity

        // Bird boundaries
        if (this.birdY < 0) {
            this.birdY = 0;
            this.birdYSpeed = 0;
        }
        if (this.birdY + this.birdHeight > this.height - 1) {
            this.birdY = this.height - this.birdHeight - 1;
            this.gameState = this.GAME_OVER;
        }

        // Update tubes
        this.activeTube = this.tubes[0].x < this.tubes[1].x ? 0 : 1;

        for (let tube of this.tubes) {
            if (--tube.x <= -this.tubeWidth) {
                tube.x = 32;
                this.setTubeY(tube);
            }

            // Score when bird passes tube
            if (tube.x === this.birdX - this.tubeWidth) {
                this.score++;
            }
        }

        // Prepare offscreen buffers and check collision using pixel compare
        this.updateCollisionBuffers();
        this.checkCollision();

        // Update ground
        if (--this.groundX < this.bgLoc.width - this.groundLoc.width) {
            this.groundX = 0;
        }

        // Update bird frame for animation
        this.birdFrame++;
        this.birdFrame %= 3;
    }

    checkCollision() {
        if (!this.spriteLoaded) return;

        // Compare alpha channel between renderCtx (tubes in front due to destination-over) and bird-only collisionCtx
        const bw = this.birdWidth;
        const bh = this.birdHeight;
        const imageA = this.collisionCtx.getImageData(this.birdX, this.birdY, bw, bh).data; // bird only
        const imageB = this.renderCtx.getImageData(this.birdX, this.birdY, bw, bh).data;     // tubes in front
        for (let i = 0; i < imageA.length; i += 4) {
            if (imageA[i + 3] !== imageB[i + 3]) {
                this.gameState = this.GAME_OVER;
                break;
            }
        }
    }

    updateCollisionBuffers() {
        if (!this.spriteLoaded) return;
        // Clear
        this.renderCtx.clearRect(0, 0, this.width, this.height);
        this.collisionCtx.clearRect(0, 0, this.width, this.height);

        // Draw world to renderCtx first: tubes then bird using destination-over,
        // so bird ends up behind tubes, enabling alpha-diff collision
        this.renderCtx.globalCompositeOperation = 'source-over';
        // Ground
        this.drawSpriteSheetImageTo(this.renderCtx, this.groundLoc, this.groundX, 31, 1);
        // Tubes
        for (let tube of this.tubes) {
            this.drawSpriteSheetImageTo(this.renderCtx, this.tubeLoc, tube.x, tube.y, 1);
        }

        // Draw bird last using destination-over so it is placed behind existing tubes
        this.renderCtx.globalCompositeOperation = 'destination-over';
        this.drawSpriteSheetImageTo(this.renderCtx, this.birdLocs[this.birdFrame], this.birdX, this.birdY, 1);
        // Reset composite for safety
        this.renderCtx.globalCompositeOperation = 'source-over';

        // Draw bird into collision canvas (no special composite needed)
        // Use direct drawImage from sprite sheet to avoid changing drawSpriteSheetImage state
        this.collisionCtx.imageSmoothingEnabled = false;
        this.collisionCtx.drawImage(
            this.spriteSheetImage,
            this.birdLocs[this.birdFrame].x, this.birdLocs[this.birdFrame].y,
            this.birdLocs[this.birdFrame].width, this.birdLocs[this.birdFrame].height,
            this.birdX, this.birdY,
            this.birdLocs[this.birdFrame].width, this.birdLocs[this.birdFrame].height
        );
    }

    render() {
        if (!this.spriteLoaded) return;

        // Draw background
        this.drawSpriteSheetImage(this.bgLoc, 0, 0);

        // Draw tubes
        for (let tube of this.tubes) {
            this.drawSpriteSheetImage(this.tubeLoc, tube.x, tube.y);
        }

        // Draw ground
        this.drawSpriteSheetImage(this.groundLoc, this.groundX, 31);

        // Draw bird
        this.drawSpriteSheetImage(this.birdLocs[this.birdFrame], this.birdX, this.birdY);

        // Draw score only if playing or game over
        if (this.gameState !== this.HOME) {
            this.ctx.fillStyle = 'white';
            this.ctx.font = 'bold 20px Arial';
            this.ctx.textAlign = 'left';
            this.ctx.fillText('Score: ' + this.score, 10, 30);
        }

        // Draw HOME state instructions
        if (this.gameState === this.HOME) {
            // Draw semi-transparent overlay
            this.ctx.fillStyle = 'rgba(0, 0, 0, 0.3)';
            this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);

            // Draw title
            this.ctx.fillStyle = 'white';
            this.ctx.font = 'bold 36px Arial';
            this.ctx.textAlign = 'center';
            this.ctx.shadowColor = 'rgba(0, 0, 0, 0.5)';
            this.ctx.shadowBlur = 4;
            this.ctx.shadowOffsetX = 2;
            this.ctx.shadowOffsetY = 2;
            this.ctx.fillText('FLAPPY BIRD', this.canvas.width / 2, this.canvas.height / 2 - 40);

            // Draw instructions
            this.ctx.font = '20px Arial';
            this.ctx.fillText('Press SPACE or Click to Start', this.canvas.width / 2, this.canvas.height / 2 + 10);

            // Reset shadow
            this.ctx.shadowColor = 'transparent';
            this.ctx.shadowBlur = 0;
            this.ctx.shadowOffsetX = 0;
            this.ctx.shadowOffsetY = 0;
        }

        // Draw game over text if needed
        if (this.gameState === this.GAME_OVER) {
            this.ctx.fillStyle = 'rgba(0, 0, 0, 0.5)';
            this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);

            this.ctx.fillStyle = 'white';
            this.ctx.font = 'bold 30px Arial';
            this.ctx.textAlign = 'center';
            this.ctx.fillText('GAME OVER', this.canvas.width / 2, this.canvas.height / 2);

            this.ctx.font = '20px Arial';
            this.ctx.fillText('Score: ' + this.score, this.canvas.width / 2, this.canvas.height / 2 + 30);
        }
    }

    getClosestTube() {
        // Get the tube that's closest ahead of the bird
        let closest = this.tubes[0];
        for (let tube of this.tubes) {
            if (tube.x + this.tubeWidth > this.birdX) {
                if (tube.x < closest.x || closest.x + this.tubeWidth <= this.birdX) {
                    closest = tube;
                }
            }
        }
        return closest;
    }
}