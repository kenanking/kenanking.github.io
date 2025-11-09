// Interactive visualization showing SGD optimization with different parameters

import * as d3 from "https://cdn.jsdelivr.net/npm/d3@7/+esm";
import * as d3Contour from "https://cdn.jsdelivr.net/npm/d3-contour@4/+esm";
import { interpolateYlGnBu } from "https://cdn.jsdelivr.net/npm/d3-scale-chromatic@3/+esm";

// Visualization dimensions
const width = 800;
const height = 500;
const nx = parseInt(width / 5);
const ny = parseInt(height / 5);
const h = 1e-7; // step for gradient approximation
const drawing_time = 8; // animation time per step (ms) - faster animation

// Domain for the loss function
const domain_x = [-5, 22];  // b range (bias)
const domain_y = [-6, 10];  // w range (weight)

// Default starting point
const default_start_w = 6;
const default_start_b = -3;

// Track current starting point
let current_start_w = null;
let current_start_b = null;

// Flag to prevent multiple optimization calls during reset
let isResetting = false;

// Legend state tracking
let legendInitialized = false;

// Simple pseudo-random number generator with seed (Mulberry32)
function seededRandom(seed) {
    return function () {
        let t = seed += 0x6D2B79F5;
        t = Math.imul(t ^ t >>> 15, t | 1);
        t ^= t + Math.imul(t ^ t >>> 7, t | 61);
        return ((t ^ t >>> 14) >>> 0) / 4294967296;
    }
}

// Generate training data: y = 0.75*x^2 + x + 2 + noise
// True function: y = (3/4)*x^2 + x + 2
function generateData(numPoints = 50, seed = 2025) {
    const data = [];
    const xMin = -5;
    const xMax = 5;
    const random = seededRandom(seed);

    for (let i = 0; i < numPoints; i++) {
        const x = xMin + (xMax - xMin) * random();
        const yTrue = 0.75 * x * x + x + 2;
        const noise = (random() - 0.5) * 3; // noise in range [-1.5, 1.5]
        const y = yTrue + noise;
        data.push({ x, y });
    }

    return data;
}

// Generate data once at load time
const trainingData = generateData(50);

// Loss functions for linear regression: y = w*x + b
const lossFunctions = {
    'MSE': (w, b) => {
        // Mean Squared Error
        let sum = 0;
        for (const point of trainingData) {
            const prediction = w * point.x + b;
            const error = prediction - point.y;
            sum += error * error;
        }
        return sum / trainingData.length;
    },
    'L1': (w, b) => {
        // Mean Absolute Error
        let sum = 0;
        for (const point of trainingData) {
            const prediction = w * point.x + b;
            const error = Math.abs(prediction - point.y);
            sum += error;
        }
        return sum / trainingData.length;
    },
    'Huber': (w, b) => {
        // Huber loss (delta = 1.0)
        const delta = 1.0;
        let sum = 0;
        for (const point of trainingData) {
            const prediction = w * point.x + b;
            const error = Math.abs(prediction - point.y);
            if (error <= delta) {
                sum += 0.5 * error * error;
            } else {
                sum += delta * (error - 0.5 * delta);
            }
        }
        return sum / trainingData.length;
    },
    'SmoothL1': (w, b) => {
        // Smooth L1 loss (beta = 0.25)
        const beta = 0.25;
        let sum = 0;
        for (const point of trainingData) {
            const prediction = w * point.x + b;
            const error = Math.abs(prediction - point.y);
            if (error < beta) {
                sum += 0.5 * error * error / beta;
            } else {
                sum += error - 0.5 * beta;
            }
        }
        return sum / trainingData.length;
    }
};

// Current loss function
let current_loss_fn = 'MSE';
let f = (w, b) => lossFunctions[current_loss_fn](w, b);

// Gradient calculation using finite differences
function grad_f(w, b) {
    const grad_w = (f(w + h, b) - f(w, b)) / h;
    const grad_b = (f(w, b + h) - f(w, b)) / h;
    return [grad_w, grad_b];
}

// SGD optimizer with momentum, Nesterov, dampening, and weight decay
function get_sgd_path(w0, b0, learning_rate, momentum, weight_decay, dampening, nesterov, num_steps) {
    let w = w0;
    let b = b0;
    let v_w = 0;
    let v_b = 0;

    const history = [{ w: w, b: b }];

    for (let i = 0; i < num_steps; i++) {
        let [grad_w, grad_b] = grad_f(w, b);

        // Weight decay
        if (weight_decay > 0) {
            grad_w += weight_decay * w;
            grad_b += weight_decay * b;
        }

        // Momentum with dampening (PyTorch semantics)
        if (momentum > 0) {
            const oneMinusDamp = 1 - dampening;
            v_w = momentum * v_w + oneMinusDamp * grad_w;
            v_b = momentum * v_b + oneMinusDamp * grad_b;

            if (nesterov) {
                grad_w = grad_w + momentum * v_w;
                grad_b = grad_b + momentum * v_b;
            } else {
                grad_w = v_w;
                grad_b = v_b;
            }
        }

        // Update parameters
        w = w - learning_rate * grad_w;
        b = b - learning_rate * grad_b;

        history.push({ w: w, b: b });

        // Early stopping if converged
        if (Math.abs(grad_w) < 1e-6 && Math.abs(grad_b) < 1e-6) {
            break;
        }
    }

    return history;
}

// Scales for coordinate transformation
const scale_x = d3.scaleLinear()
    .domain([0, width])
    .range(domain_x);

const scale_y = d3.scaleLinear()
    .domain([0, height])
    .range([domain_y[1], domain_y[0]]); // Reversed for SVG coordinates

// Inverse scales
const scale_x_inv = d3.scaleLinear()
    .domain(domain_x)
    .range([0, width]);

const scale_y_inv = d3.scaleLinear()
    .domain([domain_y[1], domain_y[0]]) // Reversed for SVG coordinates
    .range([0, height]);

// Generate loss values for contour plot
function get_loss_values(nx, ny) {
    const grid = new Array(nx * ny);
    for (let i = 0; i < nx; i++) {
        for (let j = 0; j < ny; j++) {
            const b = scale_x(parseFloat(i) / nx * width);
            const w = scale_y(parseFloat(j) / ny * height);
            grid[i + j * nx] = f(w, b);
        }
    }
    return grid;
}

// Create SVG container with responsive scaling
const svg = d3.create("svg")
    .attr("viewBox", `0 0 ${width} ${height}`)
    .attr("preserveAspectRatio", "xMidYMid meet")
    .style("width", "100%")
    .style("height", "auto")
    .style("max-width", `${width}px`)
    .style("border", "1px solid #ccc")
    .style("cursor", "crosshair");

const function_g = svg.append("g");
const gradient_path_g = svg.append("g");

// Color scale for contours (domain set dynamically in drawContours)
const color_scale = d3.scaleSequential()
    .domain([0, 1])
    .interpolator(interpolateYlGnBu);

// Draw contour plot
function drawContours() {
    function_g.selectAll("path").remove();

    const loss_values = get_loss_values(nx, ny);

    // Recalculate thresholds based on actual data range
    const min_val = d3.min(loss_values);
    const max_val = d3.max(loss_values);

    // Color scale spans the actual data range
    color_scale.domain([min_val, max_val]);

    const contours = d3Contour.contours()
        .size([nx, ny]);

    const contourData = contours(loss_values);

    const geoPath = d3.geoPath(d3.geoIdentity().scale(width / nx));

    function_g.selectAll("path")
        .data(contourData)
        .enter()
        .append("path")
        .attr("d", geoPath)
        .attr("fill", d => color_scale(d.value))
        .attr("stroke", "none");
}

// Ensure legend has consistent structure for displaying coordinates
function ensureLegendStructure() {
    if (legendInitialized) {
        return;
    }

    const legendItems = document.querySelectorAll('#sgd-canvas .legend .legend-item');
    legendItems.forEach((item, index) => {
        const marker = item.querySelector('.legend-marker');
        const label = item.querySelector('span:not(.legend-marker)');

        if (!marker || !label) {
            return;
        }

        const markerClassName = marker.className;
        const labelText = label.textContent.trim();

        item.innerHTML = '';

        const newMarker = document.createElement('span');
        newMarker.className = markerClassName;

        const textWrapper = document.createElement('div');
        textWrapper.className = 'legend-text';

        const labelSpan = document.createElement('span');
        labelSpan.className = 'legend-label';
        labelSpan.textContent = labelText;
        textWrapper.appendChild(labelSpan);

        if (index < 2) {
            const coordSpan = document.createElement('span');
            coordSpan.className = 'legend-coord';
            coordSpan.textContent = '--';
            textWrapper.appendChild(coordSpan);
        }

        item.appendChild(newMarker);
        item.appendChild(textWrapper);
    });

    legendInitialized = true;
}

function formatLegendCoord(point) {
    if (!point) {
        return '--';
    }

    const { w, b } = point;
    const wInRange = w >= domain_y[0] && w <= domain_y[1];
    const bInRange = b >= domain_x[0] && b <= domain_x[1];

    if (!wInRange || !bInRange) {
        return '超出边界';
    }

    return `(w=${w.toFixed(2)}, b=${b.toFixed(2)})`;
}

function updateLegendCoordinates(startPoint, endPoint) {
    ensureLegendStructure();

    const legendItems = document.querySelectorAll('#sgd-canvas .legend .legend-item');
    const points = [startPoint, endPoint];

    points.forEach((point, idx) => {
        const item = legendItems[idx];
        if (!item) {
            return;
        }
        const coordSpan = item.querySelector('.legend-coord');
        if (!coordSpan) {
            return;
        }
        coordSpan.textContent = formatLegendCoord(point);
    });
}

function resetLegendCoordinates() {
    updateLegendCoordinates(null, null);
}

// Line generator for path
const line = d3.line()
    .x(d => scale_x_inv(d.b))
    .y(d => scale_y_inv(d.w));

// Draw optimization path
function draw_path(path_data) {
    gradient_path_g.selectAll("*").remove();

    if (path_data.length === 0) {
        resetLegendCoordinates();
        return;
    }

    // Sample points for display (every Nth point to speed up animation)
    const sample_rate = Math.max(1, Math.floor(path_data.length / 50)); // Max 50 points
    const sampled_points = path_data.filter((d, i) => i % sample_rate === 0 || i === path_data.length - 1);

    // Draw animated path (use full path for smooth line)
    const path = gradient_path_g.append("path")
        .datum(path_data)
        .attr("fill", "none")
        .attr("stroke", "#e74c3c")
        .attr("stroke-width", 2.5)
        .attr("stroke-opacity", 0.8)
        .attr("d", line);

    const totalLength = path.node().getTotalLength();

    // Faster animation (fixed duration regardless of path length)
    path
        .attr("stroke-dasharray", totalLength + " " + totalLength)
        .attr("stroke-dashoffset", totalLength)
        .transition()
        .duration(Math.min(2000, sampled_points.length * drawing_time)) // Max 2 seconds
        .ease(d3.easeLinear)
        .attr("stroke-dashoffset", 0);

    // Draw sampled points only
    gradient_path_g.selectAll("circle.path-point")
        .data(sampled_points)
        .enter()
        .append("circle")
        .attr("class", "path-point")
        .attr("cx", d => scale_x_inv(d.b))
        .attr("cy", d => scale_y_inv(d.w))
        .attr("r", 3)
        .attr("fill", "#c0392b")
        .attr("stroke", "white")
        .attr("stroke-width", 1)
        .attr("opacity", 0)
        .transition()
        .delay((d, i) => i * drawing_time * 2)
        .duration(200)
        .attr("opacity", 1);

    // Highlight start and end points
    gradient_path_g.append("circle")
        .attr("cx", scale_x_inv(path_data[0].b))
        .attr("cy", scale_y_inv(path_data[0].w))
        .attr("r", 6)
        .attr("fill", "#2ecc71")
        .attr("stroke", "white")
        .attr("stroke-width", 2);

    gradient_path_g.append("circle")
        .attr("cx", scale_x_inv(path_data[path_data.length - 1].b))
        .attr("cy", scale_y_inv(path_data[path_data.length - 1].w))
        .attr("r", 6)
        .attr("fill", "#3498db")
        .attr("stroke", "white")
        .attr("stroke-width", 2);

    updateLegendCoordinates(path_data[0], path_data[path_data.length - 1]);
}

// Run optimization
function optimize(w0, b0) {
    // Save current starting point
    current_start_w = w0;
    current_start_b = b0;

    const learning_rate = parseFloat(document.getElementById("learning-rate").value);
    const momentum = parseFloat(document.getElementById("momentum").value);
    const dampening = parseFloat(document.getElementById("dampening").value);
    const nesterov = document.getElementById("nesterov").checked;
    const weight_decay = parseFloat(document.getElementById("weight-decay").value);
    const num_steps = 200; // Reduced from 500 for faster visualization

    const path = get_sgd_path(w0, b0, learning_rate, momentum, weight_decay, dampening, nesterov, num_steps);
    draw_path(path);

    // Update display values
    document.getElementById("lr-value").textContent = learning_rate.toFixed(4);
    document.getElementById("momentum-value").textContent = momentum.toFixed(3);
    document.getElementById("dampening-value").textContent = dampening.toFixed(3);
    document.getElementById("wd-value").textContent = weight_decay.toFixed(4);
}

// Re-run optimization with current starting point (if exists)
function reOptimize() {
    if (!isResetting && current_start_w !== null && current_start_b !== null) {
        optimize(current_start_w, current_start_b);
    }
}

// Click handler for SVG
svg.on("click", function (event) {
    const [x, y] = d3.pointer(event);
    const b = scale_x(x);
    const w = scale_y(y);
    optimize(w, b);
});

// Initialize function
function init() {
    // Initialize visualization
    drawContours();

    // Mount SVG to DOM
    const canvasElement = document.getElementById("sgd-canvas");
    if (!canvasElement) {
        console.error("sgd-canvas element not found!");
        return;
    }
    canvasElement.appendChild(svg.node());

    ensureLegendStructure();
    resetLegendCoordinates();

    // Control panel event listeners
    document.getElementById("loss-function").addEventListener("change", function (e) {
        current_loss_fn = e.target.value;
        f = (w, b) => lossFunctions[current_loss_fn](w, b);
        drawContours();
        // Re-run optimization with new loss function if starting point exists
        reOptimize();
    });

    document.getElementById("learning-rate").addEventListener("input", function (e) {
        document.getElementById("lr-value").textContent = parseFloat(e.target.value).toFixed(4);
        reOptimize();
    });

    document.getElementById("momentum").addEventListener("input", function (e) {
        document.getElementById("momentum-value").textContent = parseFloat(e.target.value).toFixed(3);
        reOptimize();
    });

    document.getElementById("dampening").addEventListener("input", function (e) {
        document.getElementById("dampening-value").textContent = parseFloat(e.target.value).toFixed(3);
        reOptimize();
    });

    document.getElementById("nesterov").addEventListener("change", function (e) {
        reOptimize();
    });

    document.getElementById("weight-decay").addEventListener("input", function (e) {
        document.getElementById("wd-value").textContent = parseFloat(e.target.value).toFixed(4);
        reOptimize();
    });

    document.getElementById("reset-btn").addEventListener("click", function () {
        // Set flag to prevent multiple optimization calls
        isResetting = true;

        // Reset all SGD parameters to default values
        document.getElementById("learning-rate").value = 0.1;
        document.getElementById("momentum").value = 0;
        document.getElementById("dampening").value = 0;
        document.getElementById("nesterov").checked = false;
        document.getElementById("weight-decay").value = 0;

        // Update display values
        document.getElementById("lr-value").textContent = "0.1000";
        document.getElementById("momentum-value").textContent = "0.000";
        document.getElementById("dampening-value").textContent = "0.000";
        document.getElementById("wd-value").textContent = "0.0000";

        // Reset flag
        isResetting = false;

        // Run optimization from default starting point
        optimize(default_start_w, default_start_b);
    });

    document.getElementById("clear-btn").addEventListener("click", function () {
        gradient_path_g.selectAll("*").remove();
        // Clear current starting point
        current_start_w = null;
        current_start_b = null;
        resetLegendCoordinates();
    });

    // Run initial optimization from default starting point
    optimize(default_start_w, default_start_b);
}

// Wait for DOM to be ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}
