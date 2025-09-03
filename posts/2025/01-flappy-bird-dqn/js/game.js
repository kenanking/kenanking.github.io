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
        this.tubeLoc = { x: 37, y: 0, width: 6, height: 44 };

        // Load sprite sheet
        this.initSpriteSheet();

        // Precomputed alpha masks for pixel collision
        this.birdMasks = []; // one per bird frame
        this.tubeMask = null; // full pipe sprite mask

        // Initialize game with HOME state
        this.initializeGame();
    }

    initSpriteSheet() {
        // Original sprite sheet base64 data
        const flappyBirdSource = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACsAAAAsBAMAAAAOdlxLAAAACXBIWXMAAAsTAAALEwEAmpwYAAAAJ1BMVEUAAAD///9WgiF2wix7xc2J5oue6ljPwizcgiPi/4vjTkbq/dvr/d1TcH2qAAAAAnRSTlMAAHaTzTgAAABpSURBVCjPY3BBA+U1jJZGAgzowuxVC7EJMzRgV43dkHJqGILDJQXYDREUFBw0wkKWRgrECwOBAJWEd29x8d6928UFiqHCu9EAVDgUDQweYQYGJlBQYQqDw9tsciYySs40AAsLYoKRLgwAuZnLUt0EGKwAAAAASUVORK5CYII=";

        this.spriteSheetImage = new Image();
        this.spriteSheetImage.src = flappyBirdSource;

        // Create offscreen canvas for sprite sheet
        this.spriteSheetCanvas = document.createElement("canvas");
        this.spriteSheetCanvas.width = 37;
        this.spriteSheetCanvas.height = 80;

        // When image loads, draw it to the sprite sheet canvas
        this.spriteSheetImage.onload = () => {
            this.spriteLoaded = true;

            // Build alpha masks for collision
            this.birdMasks = this.birdLocs.map(loc => this.buildAlphaMaskFromSprite(loc));
            this.tubeMask = this.buildAlphaMaskFromSprite(this.tubeLoc);
        };
    }

    // Build a boolean alpha mask (true where alpha > 0) from a sprite rect
    buildAlphaMaskFromSprite(locRect) {
        const temp = document.createElement('canvas');
        temp.width = locRect.width;
        temp.height = locRect.height;
        const tc = temp.getContext('2d');
        tc.imageSmoothingEnabled = false;
        tc.drawImage(
            this.spriteSheetImage,
            locRect.x, locRect.y, locRect.width, locRect.height,
            0, 0, locRect.width, locRect.height
        );
        const img = tc.getImageData(0, 0, locRect.width, locRect.height).data;
        const mask = new Array(locRect.width * locRect.height);
        for (let i = 0; i < img.length; i += 4) {
            mask[i >> 2] = img[i + 3] > 0;
        }
        return { mask, w: locRect.width, h: locRect.height };
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
        this.birdY = this.birdY + this.birdYSpeed;
        this.birdYSpeed += this.gravity;  // Apply gravity

        // Bird boundaries
        if (this.birdY < 0) {
            this.birdY = 0;
            this.birdYSpeed = 0;
        }
        if (Math.floor(this.birdY) + this.birdHeight > this.height - 1) {
            this.birdY = this.height - this.birdHeight - 1;
            this.gameState = this.GAME_OVER;
        }

        // Update tubes
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

        // Check collision using precomputed masks
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
        if (!this.spriteLoaded || !this.tubeMask || this.birdMasks.length === 0) return;

        const bw = this.birdWidth;
        const bh = this.birdHeight;
        const birdX = this.birdX;
        const birdY = Math.floor(this.birdY);
        const birdMask = this.birdMasks[this.birdFrame];

        for (let i = 0; i < this.tubes.length; i++) {
            const tube = this.tubes[i];
            const tw = this.tubeMask.w;
            const th = this.tubeMask.h;
            const tx = tube.x;
            const ty = tube.y;

            // AABB broad-phase
            const x0 = Math.max(birdX, tx);
            const y0 = Math.max(birdY, ty);
            const x1 = Math.min(birdX + bw, tx + tw);
            const y1 = Math.min(birdY + bh, ty + th);
            if (x0 >= x1 || y0 >= y1) continue;

            // Narrow-phase: mask overlap
            for (let y = y0; y < y1; y++) {
                const byRow = (y - birdY) * birdMask.w;
                const tyRow = (y - ty) * this.tubeMask.w;
                for (let x = x0; x < x1; x++) {
                    const bi = (x - birdX) + byRow;
                    const ti = (x - tx) + tyRow;
                    if (birdMask.mask[bi] && this.tubeMask.mask[ti]) {
                        this.gameState = this.GAME_OVER;
                        return;
                    }
                }
            }
        }
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
        this.drawSpriteSheetImage(this.birdLocs[this.birdFrame], this.birdX, Math.floor(this.birdY));

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