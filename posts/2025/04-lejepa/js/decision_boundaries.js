/**
 * LeJEPA Demo: Isotropic vs Anisotropic Embedding Space
 * 
 * This script simulates Logistic Regression on 2D Gaussian data to demonstrate
 * how covariance structure affects the variance of the learned decision boundary.
 */

// --- State & Cache ---
const state = {
    nSamples: 100,
    nTrials: 50,
    condNumber: 20,
    muBlue: [-1.0, 1.0],
    muRed: [1.0, -1.0],
    isRunning: false
};

// Cache for re-drawing without re-running simulation
const cache = {
    iso: { samplesBlue: [], samplesRed: [], boundaries: [] },
    aniso: { samplesBlue: [], samplesRed: [], boundaries: [] }
};

// --- Math Rendering Helpers ---

const katexRenderOptions = {
    throwOnError: false,
    strict: 'ignore'
};

function updateVarianceStat(statsId, variance) {
    const el = document.getElementById(statsId);
    if (!el || !window.katex || typeof window.katex.render !== 'function') return;

    const latex = (typeof variance === 'number' && isFinite(variance))
        ? `\\operatorname{Var}(\\beta)=${variance.toFixed(4)}`
        : '\\operatorname{Var}(\\beta)=\\text{--}';

    window.katex.render(latex, el, katexRenderOptions);
}

// --- Math Utilities ---

function matMul(A, B) {
    // Simple 2x2 matrix multiplication
    return [
        [A[0][0] * B[0][0] + A[0][1] * B[1][0], A[0][0] * B[0][1] + A[0][1] * B[1][1]],
        [A[1][0] * B[0][0] + A[1][1] * B[1][0], A[1][0] * B[0][1] + A[1][1] * B[1][1]]
    ];
}

function matVecMul(A, v) {
    // 2x2 matrix * 2D vector
    return [
        A[0][0] * v[0] + A[0][1] * v[1],
        A[1][0] * v[0] + A[1][1] * v[1]
    ];
}

function cholesky2x2(Sigma) {
    // Cholesky decomposition for 2x2 matrix
    // Sigma = L * L.T
    const L = [[0, 0], [0, 0]];
    L[0][0] = Math.sqrt(Sigma[0][0]);
    L[1][0] = Sigma[1][0] / L[0][0];
    L[1][1] = Math.sqrt(Sigma[1][1] - L[1][0] * L[1][0]);
    return L;
}

function generateGaussian(mean, cov, n) {
    // Generate n samples from N(mean, cov)
    const L = cholesky2x2(cov);
    const samples = [];
    for (let i = 0; i < n; i++) {
        // Standard normal
        const u1 = Math.random();
        const u2 = Math.random();
        const z0 = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
        const z1 = Math.sqrt(-2.0 * Math.log(u1)) * Math.sin(2.0 * Math.PI * u2);

        // Transform: x = L * z + mu
        const x = matVecMul(L, [z0, z1]);
        samples.push([x[0] + mean[0], x[1] + mean[1]]);
    }
    return samples;
}

function getCovariances(condNumber) {
    // Isotropic: Identity
    const covIso = [[1, 0], [0, 1]];

    // Anisotropic
    // Trace = 2
    // lam_max / lam_min = condNumber
    const lamMin = 2 / (condNumber + 1);
    const lamMax = condNumber * lamMin;

    // Rotation -45 degrees
    const theta = -Math.PI / 4;
    const c = Math.cos(theta);
    const s = Math.sin(theta);
    const R = [[c, -s], [s, c]];
    const Rt = [[c, s], [-s, c]]; // Transpose of rotation matrix

    const L = [[lamMax, 0], [0, lamMin]];

    // Cov = R * L * R.T
    const temp = matMul(R, L);
    const covAniso = matMul(temp, Rt);

    return { covIso, covAniso };
}

// --- Logistic Regression (Simplified) ---

class LogisticRegression {
    constructor(learningRate = 0.1, nIter = 100) {
        this.lr = learningRate;
        this.nIter = nIter;
        this.weights = [0, 0]; // w1, w2
        this.bias = 0;
    }

    sigmoid(z) {
        return 1 / (1 + Math.exp(-z));
    }

    fit(X, y) {
        // Reset weights
        this.weights = [0, 0];
        this.bias = 0;

        const m = X.length;

        // Simple Gradient Descent
        for (let i = 0; i < this.nIter; i++) {
            let dw1 = 0;
            let dw2 = 0;
            let db = 0;

            for (let j = 0; j < m; j++) {
                const z = this.weights[0] * X[j][0] + this.weights[1] * X[j][1] + this.bias;
                const yPred = this.sigmoid(z);
                const error = yPred - y[j];

                dw1 += error * X[j][0];
                dw2 += error * X[j][1];
                db += error;
            }

            this.weights[0] -= (this.lr * dw1) / m;
            this.weights[1] -= (this.lr * dw2) / m;
            this.bias -= (this.lr * db) / m;
        }
    }
}

// --- Visualization ---

function drawPlot(canvasId, data, statsId) {
    if (!data.boundaries.length) {
        updateVarianceStat(statsId, null);
        return;
    }

    const canvas = document.getElementById(canvasId);
    const ctx = canvas.getContext('2d');
    // width/height will be defined later based on logicalSize

    // Get theme colors from computed styles
    const container = document.getElementById('lejepa-container');
    const style = getComputedStyle(container);

    // Read CSS variables for grid/axis
    const gridColor = style.getPropertyValue('--lejepa-grid').trim() || 'rgba(0,0,0,0.1)';
    const axisColor = style.getPropertyValue('--lejepa-axis').trim() || 'rgba(0,0,0,0.4)';

    // High DPI Support
    const dpr = window.devicePixelRatio || 1;
    const logicalSize = 400; // Fixed logical size matching HTML width/height

    // Resize canvas to physical pixels
    canvas.width = logicalSize * dpr;
    canvas.height = logicalSize * dpr;

    // Scale context to match logical coordinates
    ctx.scale(dpr, dpr);

    // Use logical size for drawing calculations
    const width = logicalSize;
    const height = logicalSize;

    // Clear (using logical coordinates because of scale, but clearRect works in transformed space? 
    // Actually clearRect clears pixels. It's safer to clear the whole physical buffer or use logical with scale)
    // Let's use resetTransform to clear safely, then rescale.
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.scale(dpr, dpr);

    // Coordinate system: [-4, 4] -> [0, width]
    const scale = width / 8;
    const originX = width / 2;
    const originY = height / 2;

    function toCanvas(x, y) {
        return [originX + x * scale, originY - y * scale];
    }

    // Draw Grid
    ctx.save();
    ctx.strokeStyle = gridColor;
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let i = -4; i <= 4; i++) {
        const [x1, y1] = toCanvas(i, -4);
        const [x2, y2] = toCanvas(i, 4);
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        const [x3, y3] = toCanvas(-4, i);
        const [x4, y4] = toCanvas(4, i);
        ctx.moveTo(x3, y3);
        ctx.lineTo(x4, y4);
    }
    ctx.stroke();
    ctx.restore();

    // Draw Axes
    ctx.save();
    ctx.strokeStyle = axisColor;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    const [x0, y0] = toCanvas(0, 0);
    ctx.moveTo(0, y0);
    ctx.lineTo(width, y0);
    ctx.moveTo(x0, 0);
    ctx.lineTo(x0, height);
    ctx.stroke();
    ctx.restore();

    // Draw Samples (Blue)
    ctx.fillStyle = 'rgba(52, 152, 219, 0.4)';
    ctx.strokeStyle = '#3498db';
    data.samplesBlue.forEach(pt => {
        const [cx, cy] = toCanvas(pt[0], pt[1]);
        ctx.beginPath();
        ctx.arc(cx, cy, 3, 0, 2 * Math.PI);
        ctx.fill();
        ctx.stroke();
    });

    // Draw Samples (Red)
    ctx.fillStyle = 'rgba(231, 76, 60, 0.4)';
    ctx.strokeStyle = '#e74c3c';
    data.samplesRed.forEach(pt => {
        const [cx, cy] = toCanvas(pt[0], pt[1]);
        ctx.beginPath();
        ctx.arc(cx, cy, 3, 0, 2 * Math.PI);
        ctx.fill();
        ctx.stroke();
    });

    // Draw Learned Boundaries
    ctx.strokeStyle = 'rgba(155, 89, 182, 0.3)';
    ctx.lineWidth = 1.5;
    data.boundaries.forEach(b => {
        const w1 = b.weights[0];
        const w2 = b.weights[1];
        const bias = b.bias;

        if (Math.abs(w2) < 1e-5) return;

        ctx.beginPath();
        const xStart = -4;
        const yStart = -(w1 * xStart + bias) / w2;
        const xEnd = 4;
        const yEnd = -(w1 * xEnd + bias) / w2;

        const [cx1, cy1] = toCanvas(xStart, yStart);
        const [cx2, cy2] = toCanvas(xEnd, yEnd);

        ctx.moveTo(cx1, cy1);
        ctx.lineTo(cx2, cy2);
        ctx.stroke();
    });

    // Draw True Boundary (y = x)
    ctx.strokeStyle = '#2ecc71';
    ctx.lineWidth = 3;
    ctx.beginPath();
    const [tx1, ty1] = toCanvas(-4, -4);
    const [tx2, ty2] = toCanvas(4, 4);
    ctx.moveTo(tx1, ty1);
    ctx.lineTo(tx2, ty2);
    ctx.stroke();

    // Calculate Variance
    if (data.boundaries.length > 1) {
        let sumW1 = 0, sumW2 = 0;
        data.boundaries.forEach(b => { sumW1 += b.weights[0]; sumW2 += b.weights[1]; });
        const meanW1 = sumW1 / data.boundaries.length;
        const meanW2 = sumW2 / data.boundaries.length;

        let varSum = 0;
        data.boundaries.forEach(b => {
            varSum += (b.weights[0] - meanW1) ** 2 + (b.weights[1] - meanW2) ** 2;
        });
        const variance = varSum / (data.boundaries.length - 1);

        updateVarianceStat(statsId, variance);
    } else {
        updateVarianceStat(statsId, null);
    }
}

// --- Main Experiment Loop ---

async function runExperiment() {
    if (state.isRunning) return;
    state.isRunning = true;

    // Update params from UI
    state.nSamples = parseInt(document.getElementById('samples').value);
    state.nTrials = parseInt(document.getElementById('trials').value);
    state.condNumber = parseInt(document.getElementById('cond').value);

    const { covIso, covAniso } = getCovariances(state.condNumber);

    // --- Run Isotropic ---
    cache.iso.boundaries = [];
    for (let i = 0; i < state.nTrials; i++) {
        const X_blue = generateGaussian(state.muBlue, covIso, state.nSamples / 2);
        const X_red = generateGaussian(state.muRed, covIso, state.nSamples / 2);
        const X = [...X_blue, ...X_red];
        const y = [...Array(Math.floor(state.nSamples / 2)).fill(0), ...Array(Math.floor(state.nSamples / 2)).fill(1)];

        const clf = new LogisticRegression(0.5, 200);
        clf.fit(X, y);
        cache.iso.boundaries.push({ weights: clf.weights, bias: clf.bias });

        if (i === state.nTrials - 1) {
            cache.iso.samplesBlue = X_blue;
            cache.iso.samplesRed = X_red;
        }

        if (i % 10 === 0) await new Promise(r => setTimeout(r, 0));
    }
    drawPlot('canvas-iso', cache.iso, 'stats-iso');

    // --- Run Anisotropic ---
    cache.aniso.boundaries = [];
    for (let i = 0; i < state.nTrials; i++) {
        const X_blue = generateGaussian(state.muBlue, covAniso, state.nSamples / 2);
        const X_red = generateGaussian(state.muRed, covAniso, state.nSamples / 2);
        const X = [...X_blue, ...X_red];
        const y = [...Array(Math.floor(state.nSamples / 2)).fill(0), ...Array(Math.floor(state.nSamples / 2)).fill(1)];

        const clf = new LogisticRegression(0.5, 200);
        clf.fit(X, y);
        cache.aniso.boundaries.push({ weights: clf.weights, bias: clf.bias });

        if (i === state.nTrials - 1) {
            cache.aniso.samplesBlue = X_blue;
            cache.aniso.samplesRed = X_red;
        }

        if (i % 10 === 0) await new Promise(r => setTimeout(r, 0));
    }
    drawPlot('canvas-aniso', cache.aniso, 'stats-aniso');

    state.isRunning = false;
}

function reDrawAll() {
    if (state.isRunning) return; // Don't interrupt running sim
    drawPlot('canvas-iso', cache.iso, 'stats-iso');
    drawPlot('canvas-aniso', cache.aniso, 'stats-aniso');
}

// --- Initialization ---

let debounceTimer;
function debouncedRun() {
    if (debounceTimer) clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => {
        runExperiment();
    }, 300);
}

function init() {
    // Ensure math placeholders render immediately
    ['stats-iso', 'stats-aniso'].forEach(id => updateVarianceStat(id, null));

    // Bind controls
    ['samples', 'trials', 'cond'].forEach(id => {
        const el = document.getElementById(id);
        const valEl = document.getElementById('val-' + id);
        if (el && valEl) {
            el.addEventListener('input', (e) => {
                valEl.textContent = e.target.value;
                debouncedRun();
            });
        } else {
            console.warn(`LeJEPA Demo: Element with id '${id}' or 'val-${id}' not found.`);
        }
    });

    // Initial run with small delay to ensure styles are applied
    if (document.getElementById('canvas-iso')) {
        setTimeout(() => {
            runExperiment();
        }, 100);
    }

    // Watch for theme changes
    // We observe the <html> element for class changes (common in dark mode implementations)
    const observer = new MutationObserver((mutations) => {
        mutations.forEach((mutation) => {
            if (mutation.type === 'attributes' && (mutation.attributeName === 'class' || mutation.attributeName === 'data-theme')) {
                // Small delay to allow CSS to propagate
                setTimeout(reDrawAll, 50);
            }
        });
    });

    observer.observe(document.documentElement, {
        attributes: true,
        attributeFilter: ['class', 'data-theme']
    });
}

// Wait for DOM
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}
