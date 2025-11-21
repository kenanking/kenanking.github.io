import * as d3 from 'https://cdn.jsdelivr.net/npm/d3@7/+esm';

const root = document.getElementById('slicing-example-container');

const katexRenderOptions = {
    throwOnError: false,
    strict: 'ignore'
};

function renderAxisLabel(elementId, latexString) {
    const el = document.getElementById(elementId);
    if (!el || !window.katex || typeof window.katex.render !== 'function') {
        if (el) el.textContent = latexString;
        return;
    }
    window.katex.render(latexString, el, katexRenderOptions);
}

const inputs = {
    n: document.getElementById('slicing-example-n'),
    rho: document.getElementById('slicing-example-rho'),
    m: document.getElementById('slicing-example-m')
};

const labels = {
    n: document.getElementById('slicing-example-n-val'),
    rho: document.getElementById('slicing-example-rho-val'),
    m: document.getElementById('slicing-example-m-val')
};

const canvases = {
    marginals: d3.select('#slicing-example-marginals'),
    density: document.getElementById('slicing-example-density'),
    projections: d3.select('#slicing-example-projections'),
    stats: d3.select('#slicing-example-stats')
};

const state = {
    nSamples: Number(inputs.n?.value) || 20000,
    rho: Number(inputs.rho?.value) || 0.98,
    mDirections: Number(inputs.m?.value) || 10,
    visibleStats: {
        vc: true,
        jb: true,
        wat: true,
        cvm: true,
        ad: true,
        ep: true
    }
};

const statsConfigs = [
    { key: 'vc', label: 'vcreg', color: '#1abc9c', dash: '4,3', marker: 'circle' },
    { key: 'jb', label: 'ext_jarque_beta', color: '#e67e22', dash: '0', marker: 'diamond' },
    { key: 'wat', label: 'watson', color: '#8e44ad', dash: '0', marker: 'triangle-down' },
    { key: 'cvm', label: 'cramer_von_mises', color: '#2c3e50', dash: '0', marker: 'square' },
    { key: 'ad', label: 'anderson_darling', color: '#c0392b', dash: '0', marker: 'triangle-up' },
    { key: 'ep', label: 'epps_pulley', color: '#f39c12', dash: '1,3', marker: 'star', width: 3.5 }
];

const statsSymbolMap = {
    'circle': d3.symbolCircle,
    'diamond': d3.symbolDiamond,
    'triangle-down': d3.symbolTriangle, // Rotate 180
    'square': d3.symbolSquare,
    'triangle-up': d3.symbolTriangle,
    'star': d3.symbolStar
};

const colorScale = (idx, total) => {
    const denom = Math.max(total - 1, 1);
    return d3.interpolateRdBu(1 - idx / denom);
};

const CHART_CONFIG = {
    margin: {
        small: { top: 32, right: 10, bottom: 46, left: 32 },
        medium: { top: 32, right: 14, bottom: 56, left: 40 },
        large: { top: 32, right: 16, bottom: 52, left: 44 }
    },
    fontSize: 14,
    dimensions: {
        marginals: { width: 400, height: 400 },
        density: { width: 400, height: 400 },
        projections: { width: 800, height: 480 },
        stats: { width: 800, height: 320 }
    }
};

const getThemeColors = () => ({
    text: getComputedStyle(root).getPropertyValue('--lejepa-text').trim() || '#111',
    axis: getComputedStyle(root).getPropertyValue('--lejepa-axis').trim() || '#000',
    bg: getComputedStyle(root).getPropertyValue('--lejepa-input-bg').trim() || '#ffffff'
});

const styleAxis = (axisGroup, colors, fontSize = CHART_CONFIG.fontSize) => {
    axisGroup.selectAll('text').attr('fill', colors.text).attr('font-size', fontSize);
    axisGroup.selectAll('line').attr('stroke', colors.axis);
    axisGroup.select('.domain').attr('stroke', colors.axis).attr('stroke-width', 1.5);
};

let cachedResult = null;
let isRunning = false;
let resizeTimer;

function randnPair() {
    const u1 = Math.random();
    const u2 = Math.random();
    const r = Math.sqrt(-2.0 * Math.log(Math.max(u1, 1e-12)));
    const theta = 2.0 * Math.PI * u2;
    return [r * Math.cos(theta), r * Math.sin(theta)];
}

function cholesky2x2(Sigma) {
    const L = [[0, 0], [0, 0]];
    L[0][0] = Math.sqrt(Math.max(Sigma[0][0], 1e-12));
    L[1][0] = Sigma[1][0] / L[0][0];
    L[1][1] = Math.sqrt(Math.max(Sigma[1][1] - L[1][0] * L[1][0], 1e-12));
    return L;
}

function matVecMul(A, v) {
    return [
        A[0][0] * v[0] + A[0][1] * v[1],
        A[1][0] * v[0] + A[1][1] * v[1]
    ];
}

function sampleGaussian(mean, cov, n) {
    const L = cholesky2x2(cov);
    const samples = [];
    for (let i = 0; i < n; i++) {
        const [z0, z1] = randnPair();
        const [x, y] = matVecMul(L, [z0, z1]);
        samples.push([x + mean[0], y + mean[1]]);
    }
    return samples;
}

function shuffle(array) {
    for (let i = array.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [array[i], array[j]] = [array[j], array[i]];
    }
    return array;
}

function generateXDistribution(nSamples, rho) {
    const n1 = Math.floor(nSamples / 2);
    const n2 = nSamples - n1;
    const cov1 = [[1, rho], [rho, 1]];
    const cov2 = [[1, -rho], [-rho, 1]];
    const X1 = sampleGaussian([0, 0], cov1, n1);
    const X2 = sampleGaussian([0, 0], cov2, n2);
    return shuffle([...X1, ...X2]);
}

function projectSamples(samples, m) {
    const angles = Array.from({ length: m }, (_, i) => (i * Math.PI) / (m - 1));
    const vectors = angles.map(theta => [Math.cos(theta), Math.sin(theta)]);
    const projections = vectors.map(vec =>
        samples.map(pt => pt[0] * vec[0] + pt[1] * vec[1])
    );
    return { vectors, projections };
}

function invert2x2(matrix) {
    const [a, b] = matrix[0];
    const [c, d] = matrix[1];
    const det = a * d - b * c;
    const invDet = 1 / det;
    return {
        det,
        inv: [
            [d * invDet, -b * invDet],
            [-c * invDet, a * invDet]
        ]
    };
}

function gaussianPdf(point, meta) {
    const x = point[0];
    const y = point[1];
    const [ixx, ixy] = meta.inv[0];
    const [iyx, iyy] = meta.inv[1];
    const quad = ixx * x * x + (ixy + iyx) * x * y + iyy * y * y;
    const norm = 1 / (2 * Math.PI * Math.sqrt(Math.abs(meta.det)));
    return norm * Math.exp(-0.5 * quad);
}

function computeDensityField(rho, size = 160, range = 3.5) {
    const cov1 = [[1, rho], [rho, 1]];
    const cov2 = [[1, -rho], [-rho, 1]];
    const meta1 = invert2x2(cov1);
    const meta2 = invert2x2(cov2);
    const values = new Float32Array(size * size);
    const coords = Array.from({ length: size }, (_, idx) => -range + (2 * range * idx) / (size - 1));
    let min = Number.POSITIVE_INFINITY;
    let max = Number.NEGATIVE_INFINITY;

    for (let yi = 0; yi < size; yi++) {
        const y = coords[yi];
        for (let xi = 0; xi < size; xi++) {
            const x = coords[xi];
            const pdf = 0.5 * gaussianPdf([x, y], meta1) + 0.5 * gaussianPdf([x, y], meta2);
            const idx = yi * size + xi;
            values[idx] = pdf;
            if (pdf > max) max = pdf;
            if (pdf < min) min = pdf;
        }
    }

    return { values, size, range, coords, min, max };
}

function mean(arr) {
    const total = arr.reduce((acc, val) => acc + val, 0);
    return total / (arr.length || 1);
}

function summarize(values) {
    const n = values.length || 1;
    let sum = 0;
    let sumSq = 0;
    for (let i = 0; i < n; i++) {
        const v = values[i];
        sum += v;
        sumSq += v * v;
    }
    const meanVal = sum / n;
    const variance = Math.max(sumSq / n - meanVal * meanVal, 0);
    const std = Math.sqrt(variance) || 1e-6;
    return { mean: meanVal, variance, std };
}

function vcregLoss(stats) {
    return stats.mean * stats.mean + (stats.std - 1) ** 2;
}

function jarqueBeraLoss(values, stats = summarize(values)) {
    const n = values.length || 1;
    let skewSum = 0;
    let kurtSum = 0;
    for (let i = 0; i < n; i++) {
        const v = (values[i] - stats.mean) / stats.std;
        skewSum += v ** 3;
        kurtSum += v ** 4;
    }
    const skew = skewSum / n;
    const kurt = kurtSum / n;
    return (n / 6) * (skew * skew + ((kurt - 3) ** 2) / 4);
}

function erf(x) {
    const sign = Math.sign(x);
    const absX = Math.abs(x);
    const t = 1 / (1 + 0.5 * absX);
    const tau = t * Math.exp(
        -absX * absX
        - 1.26551223
        + 1.00002368 * t
        + 0.37409196 * t ** 2
        + 0.09678418 * t ** 3
        - 0.18628806 * t ** 4
        + 0.27886807 * t ** 5
        - 1.13520398 * t ** 6
        + 1.48851587 * t ** 7
        - 0.82215223 * t ** 8
        + 0.17087277 * t ** 9
    );
    return sign * (1 - tau);
}

function normCDF(x) {
    return 0.5 * (1 + erf(x / Math.SQRT2));
}

function sortedNormalCdf(values) {
    return [...values].sort((a, b) => a - b).map(normCDF);
}

function cramerVonMisesLossFromCdf(cdfVals) {
    const n = cdfVals.length;
    let stat = 1 / (12 * n);
    for (let i = 0; i < n; i++) {
        const diff = cdfVals[i] - (2 * i + 1) / (2 * n);
        stat += diff * diff;
    }
    return stat;
}

function andersonDarlingLossFromCdf(cdfVals) {
    const n = cdfVals.length;
    let sum = 0;
    for (let i = 0; i < n; i++) {
        const Fi = Math.min(Math.max(cdfVals[i], 1e-6), 1 - 1e-6);
        const Fj = Math.min(Math.max(cdfVals[n - 1 - i], 1e-6), 1 - 1e-6);
        sum += (2 * i + 1) * (Math.log(Fi) + Math.log(1 - Fj));
    }
    return -n - (sum / n);
}

function watsonLossFromCdf(cdfVals, cvmValue) {
    const n = cdfVals.length;
    const cvm = cvmValue ?? cramerVonMisesLossFromCdf(cdfVals);
    const meanF = mean(cdfVals);
    return cvm - n * (meanF - 0.5) ** 2;
}

function trapezoid(xVals, yVals) {
    let total = 0;
    for (let i = 0; i < xVals.length - 1; i++) {
        const width = xVals[i + 1] - xVals[i];
        total += width * (yVals[i] + yVals[i + 1]) * 0.5;
    }
    return total;
}

function eppsPulleyLoss(z, nPoints = 17) {
    const tVals = Array.from({ length: nPoints }, (_, i) => -5 + (10 * i) / (nPoints - 1));
    const phiTarget = tVals.map(t => Math.exp(-0.5 * t * t));
    const ecfReal = new Array(nPoints).fill(0);
    const ecfImag = new Array(nPoints).fill(0);
    const n = z.length;

    for (let i = 0; i < n; i++) {
        const val = z[i];
        for (let j = 0; j < nPoints; j++) {
            const angle = val * tVals[j];
            ecfReal[j] += Math.cos(angle);
            ecfImag[j] += Math.sin(angle);
        }
    }

    for (let j = 0; j < nPoints; j++) {
        ecfReal[j] /= n;
        ecfImag[j] /= n;
    }

    const err = tVals.map((_, idx) => {
        const diffRe = ecfReal[idx] - phiTarget[idx];
        const diffIm = ecfImag[idx];
        const magSq = diffRe * diffRe + diffIm * diffIm;
        return magSq * phiTarget[idx];
    });

    return trapezoid(tVals, err) * n;
}

function computeStats(projections) {
    const stats = {
        vc: [],
        jb: [],
        wat: [],
        cvm: [],
        ad: [],
        ep: []
    };

    projections.forEach(values => {
        const summary = summarize(values);
        const cdfVals = sortedNormalCdf(values);
        const cvm = cramerVonMisesLossFromCdf(cdfVals);

        stats.vc.push(vcregLoss(summary));
        stats.jb.push(jarqueBeraLoss(values, summary));
        stats.cvm.push(cvm);
        stats.ad.push(andersonDarlingLossFromCdf(cdfVals));
        stats.wat.push(watsonLossFromCdf(cdfVals, cvm));
        stats.ep.push(eppsPulleyLoss(values));
    });

    const normalized = {};
    Object.entries(stats).forEach(([key, arr]) => {
        const min = Math.min(...arr);
        const max = Math.max(...arr);
        const range = max - min;
        normalized[key] = range < 1e-9 ? arr.map(() => 0) : arr.map(val => (val - min) / range);
    });

    return { raw: stats, normalized };
}

function drawMarginals(samples) {
    const svg = canvases.marginals;
    if (!svg.node()) return;
    const { width, height } = CHART_CONFIG.dimensions.marginals;
    const margin = CHART_CONFIG.margin.small;
    svg.attr('viewBox', `0 0 ${width} ${height}`).attr('preserveAspectRatio', 'xMidYMid meet');
    svg.selectAll('*').remove();

    const xs = samples.map(d => d[0]);
    const ys = samples.map(d => d[1]);
    const domain = [-4, 4];
    const bins = 45;

    const xScale = d3.scaleLinear().domain(domain).range([margin.left, width - margin.right]);
    const yScale = d3.scaleLinear().range([height - margin.bottom, margin.top]);

    const binX = d3.bin().domain(domain).thresholds(bins)(xs).map(b => ({
        ...b,
        density: b.length / (xs.length * (b.x1 - b.x0))
    }));
    const binY = d3.bin().domain(domain).thresholds(bins)(ys).map(b => ({
        ...b,
        density: b.length / (ys.length * (b.x1 - b.x0))
    }));

    const maxDensity = Math.max(
        d3.max(binX, d => d.density) || 1,
        d3.max(binY, d => d.density) || 1
    );
    yScale.domain([0, maxDensity]);

    svg.append('g')
        .selectAll('rect')
        .data(binX)
        .enter()
        .append('rect')
        .attr('x', d => xScale(d.x0) + 1)
        .attr('width', d => Math.max(0, xScale(d.x1) - xScale(d.x0) - 1))
        .attr('y', d => yScale(d.density))
        .attr('height', d => yScale(0) - yScale(d.density))
        .attr('fill', 'rgba(52,152,219,0.3)');

    svg.append('g')
        .selectAll('rect')
        .data(binY)
        .enter()
        .append('rect')
        .attr('x', d => xScale(d.x0) + 1)
        .attr('width', d => Math.max(0, xScale(d.x1) - xScale(d.x0) - 1))
        .attr('y', d => yScale(d.density))
        .attr('height', d => yScale(0) - yScale(d.density))
        .attr('fill', 'rgba(231,76,60,0.3)');

    const area = d3.area()
        .x(d => xScale((d.x0 + d.x1) / 2))
        .y0(() => yScale(0))
        .y1(d => yScale(d.density))
        .curve(d3.curveCatmullRom.alpha(0.5));

    svg.append('path')
        .datum(binX)
        .attr('fill', 'none')
        .attr('stroke', '#1f78b4')
        .attr('stroke-width', 2)
        .attr('d', area.lineY1());

    svg.append('path')
        .datum(binY)
        .attr('fill', 'none')
        .attr('stroke', '#c0392b')
        .attr('stroke-width', 2)
        .attr('d', area.lineY1());

    const colors = getThemeColors();
    const xAxis = d3.axisBottom(xScale).ticks(5);
    const yAxis = d3.axisLeft(yScale).ticks(4);

    const xAxisGroup = svg.append('g')
        .attr('transform', `translate(0,${height - margin.bottom})`)
        .call(xAxis);

    styleAxis(xAxisGroup, colors);

    const yAxisGroup = svg.append('g')
        .attr('transform', `translate(${margin.left},0)`)
        .call(yAxis);

    styleAxis(yAxisGroup, colors);

    renderAxisLabel('slicing-example-marginals-xlabel', 'x_1(\\text{ blue })-x_2(\\text{ red })');
    renderAxisLabel('slicing-example-marginals-ylabel', '\\text{Count}');
}

function drawDensity(density, vectors) {
    const canvas = canvases.density;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;

    const { width: logicalWidth, height: logicalHeight } = CHART_CONFIG.dimensions.density;
    const margin = CHART_CONFIG.margin.small;

    canvas.width = logicalWidth * dpr;
    canvas.height = logicalHeight * dpr;
    ctx.scale(dpr, dpr);

    const colors = getThemeColors();
    ctx.fillStyle = colors.bg;
    ctx.fillRect(0, 0, logicalWidth, logicalHeight);

    const plotWidth = logicalWidth - margin.left - margin.right;
    const plotHeight = logicalHeight - margin.top - margin.bottom;

    const color = d3.scaleSequential(d3.interpolateMagma).domain([density.min, density.max]);
    const cutoff = Math.max(density.min + 0.0005, density.min);

    const fillThresholds = d3.scaleLinear().domain([cutoff, density.max]).ticks(48);
    const fillContourGen = d3.contours()
        .size([density.size, density.size])
        .thresholds(fillThresholds);
    const contourPathFill = d3.geoPath(undefined, ctx);

    ctx.save();
    ctx.translate(margin.left, margin.top);
    ctx.scale(plotWidth / density.size, plotHeight / density.size);
    fillContourGen(density.values).forEach(feature => {
        ctx.beginPath();
        contourPathFill(feature);
        ctx.fillStyle = color(feature.value);
        ctx.fill();
    });
    ctx.restore();

    const scaleX = plotWidth / (2 * density.range);
    const scaleY = plotHeight / (2 * density.range);

    function toCanvas(x, y) {
        return [
            margin.left + plotWidth / 2 + x * scaleX,
            margin.top + plotHeight / 2 - y * scaleY
        ];
    }

    const contourThresholds = d3.range(cutoff * 1.1, density.max, (density.max - cutoff) / 12);
    const contourGenerator = d3.contours()
        .size([density.size, density.size])
        .thresholds(contourThresholds);
    const contourPath = d3.geoPath(undefined, ctx);
    ctx.save();
    ctx.translate(margin.left, margin.top);
    ctx.scale(plotWidth / density.size, plotHeight / density.size);
    ctx.strokeStyle = 'rgba(255,255,255,0.5)';
    ctx.lineWidth = 1;
    contourGenerator(density.values).forEach(feature => {
        ctx.beginPath();
        contourPath(feature);
        ctx.stroke();
    });
    ctx.restore();

    const centerX = margin.left + plotWidth / 2;
    const centerY = margin.top + plotHeight / 2;

    vectors.forEach((vec, idx) => {
        const [endX, endY] = toCanvas(vec[0] * density.range * 0.8, vec[1] * density.range * 0.8);
        const color = colorScale(idx, vectors.length);

        ctx.save();
        ctx.strokeStyle = color;
        ctx.fillStyle = color;
        ctx.lineWidth = 2.5;

        ctx.beginPath();
        ctx.moveTo(centerX, centerY);
        ctx.lineTo(endX, endY);
        ctx.stroke();

        const angle = Math.atan2(centerY - endY, endX - centerX);
        const headLen = 10;

        ctx.beginPath();
        ctx.moveTo(endX, endY);
        ctx.lineTo(
            endX - headLen * Math.cos(angle - Math.PI / 6),
            endY + headLen * Math.sin(angle - Math.PI / 6)
        );
        ctx.lineTo(
            endX - headLen * Math.cos(angle + Math.PI / 6),
            endY + headLen * Math.sin(angle + Math.PI / 6)
        );
        ctx.closePath();
        ctx.fill();

        ctx.restore();
    });

    ctx.save();
    ctx.strokeStyle = colors.axis;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(margin.left, logicalHeight - margin.bottom);
    ctx.lineTo(logicalWidth - margin.right, logicalHeight - margin.bottom);
    ctx.moveTo(margin.left, margin.top);
    ctx.lineTo(margin.left, logicalHeight - margin.bottom);
    ctx.stroke();
    ctx.restore();

    ctx.save();
    ctx.fillStyle = colors.text;
    ctx.strokeStyle = colors.axis;
    ctx.lineWidth = 1;
    const fontSize = `${CHART_CONFIG.fontSize}px`;
    ctx.font = `${fontSize} -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';

    const xTicks = [-3, -2, -1, 0, 1, 2, 3];
    xTicks.forEach(tick => {
        const [x, y] = toCanvas(tick, 0);
        ctx.beginPath();
        ctx.moveTo(x, logicalHeight - margin.bottom);
        ctx.lineTo(x, logicalHeight - margin.bottom + 5);
        ctx.stroke();
        ctx.fillText(tick.toString(), x, logicalHeight - margin.bottom + 8);
    });

    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    const yTicks = [-3, -2, -1, 0, 1, 2, 3];
    yTicks.forEach(tick => {
        const [x, y] = toCanvas(0, tick);
        ctx.beginPath();
        ctx.moveTo(margin.left, y);
        ctx.lineTo(margin.left - 5, y);
        ctx.stroke();
        ctx.fillText(tick.toString(), margin.left - 8, y);
    });

    ctx.restore();

    renderAxisLabel('slicing-example-density-xlabel', 'x_1');
    renderAxisLabel('slicing-example-density-ylabel', 'x_2');
}

function computeHistogram(values, bins = 40, range = [-4.5, 4.5]) {
    const hist = d3.bin().domain(range).thresholds(bins)(values);
    return hist.map(b => ({
        ...b,
        density: b.length / (values.length * (b.x1 - b.x0 || 1))
    }));
}

function drawProjections(projections, vectors) {
    const svg = canvases.projections;
    if (!svg.node()) return;
    const { width, height } = CHART_CONFIG.dimensions.projections;
    const margin = CHART_CONFIG.margin.medium;

    const availableHeight = height - margin.top - margin.bottom;
    const yStep = availableHeight / projections.length;

    svg.attr('viewBox', `0 0 ${width} ${height}`).attr('preserveAspectRatio', 'xMidYMid meet');
    svg.selectAll('*').remove();

    const range = [-4.5, 4.5];
    const xScale = d3.scaleLinear().domain(range).range([margin.left, width - margin.right]);
    const allHist = projections.map(values => computeHistogram(values, 45, range));
    const maxDensity = d3.max(allHist.flat(), d => d.density) || 1;
    const scaleFactor = (yStep * 0.75) / maxDensity;

    const colors = getThemeColors();

    svg.append('line')
        .attr('x1', margin.left)
        .attr('x2', width - margin.right)
        .attr('y1', height - margin.bottom)
        .attr('y2', height - margin.bottom)
        .attr('stroke', colors.axis)
        .attr('stroke-width', 1.5);

    projections.forEach((_, idx) => {
        const i = idx;
        const baseline = (height - margin.bottom) - (i * yStep) - (yStep * 0.1);

        const data = allHist[i];
        const fillColor = colorScale(i, projections.length);

        const area = d3.area()
            .x(d => xScale((d.x0 + d.x1) / 2))
            .y0(() => baseline)
            .y1(d => baseline - d.density * scaleFactor)
            .curve(d3.curveBasis);

        svg.append('path')
            .datum(data)
            .attr('fill', fillColor)
            .attr('opacity', 0.9)
            .attr('stroke', colors.text)
            .attr('stroke-width', 0.8)
            .attr('d', area);

        svg.append('line')
            .attr('x1', xScale(range[0]))
            .attr('x2', xScale(range[1]))
            .attr('y1', baseline)
            .attr('y2', baseline)
            .attr('stroke', colors.axis)
            .attr('stroke-width', 1);

        svg.append('text')
            .attr('x', margin.left - 10)
            .attr('y', baseline - 2)
            .attr('text-anchor', 'end')
            .attr('fill', fillColor)
            .attr('font-weight', 'bold')
            .attr('font-size', CHART_CONFIG.fontSize)
            .text(`i:${i}`);
    });

    svg.append('line')
        .attr('x1', margin.left)
        .attr('x2', margin.left)
        .attr('y1', margin.top)
        .attr('y2', height - margin.bottom)
        .attr('stroke', colors.axis)
        .attr('stroke-width', 1.5);

    const axis = d3.axisBottom(xScale).ticks(5);
    const xAxisGroup = svg.append('g')
        .attr('transform', `translate(0,${height - margin.bottom})`)
        .call(axis);

    styleAxis(xAxisGroup, colors);

    renderAxisLabel('slicing-example-projections-xlabel', '\\left\\langle x, a_i\\right\\rangle');
    renderAxisLabel('slicing-example-projections-ylabel', 'p(\\left\\langle x, a_i\\right\\rangle)');
}

function drawStats(normalizedStats) {
    const svg = canvases.stats;
    if (!svg.node()) return;
    const { width, height } = CHART_CONFIG.dimensions.stats;
    const margin = CHART_CONFIG.margin.large;
    svg.attr('viewBox', `0 0 ${width} ${height}`).attr('preserveAspectRatio', 'xMidYMid meet');
    svg.selectAll('*').remove();

    const indices = normalizedStats.vc.map((_, i) => i);
    const xScale = d3.scalePoint()
        .domain(indices)
        .range([margin.left, width - margin.right])
        .padding(0.5);
    const yScale = d3.scaleLinear().domain([-0.1, 1]).range([height - margin.bottom, margin.top]);

    const colors = getThemeColors();
    const visibleConfigs = statsConfigs.filter(config => state.visibleStats[config.key]);

    visibleConfigs.forEach(config => {
        for (let i = 0; i < indices.length - 1; i++) {
            const c = colorScale(i + 0.5, indices.length);
            svg.append('line')
                .attr('x1', xScale(indices[i]))
                .attr('y1', yScale(normalizedStats[config.key][i]))
                .attr('x2', xScale(indices[i + 1]))
                .attr('y2', yScale(normalizedStats[config.key][i + 1]))
                .attr('stroke', c)
                .attr('stroke-width', config.width || 2.5)
                .attr('stroke-dasharray', config.dash)
                .attr('opacity', 0.9);
        }
    });

    visibleConfigs.forEach(config => {
        const symbolType = statsSymbolMap[config.marker] || d3.symbolCircle;
        const symbol = d3.symbol().type(symbolType).size(config.key === 'ep' ? 100 : 64);

        svg.selectAll(`.marker-${config.key}`)
            .data(indices)
            .enter()
            .append('path')
            .attr('d', symbol)
            .attr('transform', (d, i) => {
                let rot = 0;
                if (config.marker === 'triangle-down') rot = 180;
                return `translate(${xScale(d)},${yScale(normalizedStats[config.key][d])}) rotate(${rot})`;
            })
            .attr('fill', (d, i) => colorScale(i, indices.length))
            .attr('stroke', 'black')
            .attr('stroke-width', 0.8);
    });

    const xAxis = d3.axisBottom(xScale).tickFormat(d => `i:${d}`);
    const xAxisGroup = svg.append('g')
        .attr('transform', `translate(0,${height - margin.bottom})`)
        .call(xAxis);

    xAxisGroup.selectAll('text')
        .attr('fill', (d, i) => colorScale(i, indices.length))
        .attr('font-weight', 'bold')
        .attr('font-size', CHART_CONFIG.fontSize);
    xAxisGroup.selectAll('line').attr('stroke', colors.axis);
    xAxisGroup.select('.domain').attr('stroke', colors.axis).attr('stroke-width', 1.5);

    const yAxis = d3.axisLeft(yScale).ticks(4);
    const yAxisGroup = svg.append('g')
        .attr('transform', `translate(${margin.left},0)`)
        .call(yAxis);

    styleAxis(yAxisGroup, colors);

    renderAxisLabel('slicing-example-stats-ylabel', '\\ell_1 \\text{ and } \\ell_2');

    updateStatsLegend();
}

function updateStatsLegend() {
    const legendContainer = document.getElementById('slicing-example-stats-legend');
    if (legendContainer) {
        legendContainer.innerHTML = '';
        statsConfigs.forEach(config => {
            const item = document.createElement('div');
            item.className = `legend-item ${state.visibleStats[config.key] ? 'active' : 'inactive'}`;
            item.style.cursor = 'pointer';
            item.onclick = () => {
                state.visibleStats[config.key] = !state.visibleStats[config.key];
                if (cachedResult) {
                    drawStats(cachedResult.stats.normalized);
                }
            };

            const colorBox = document.createElement('span');
            colorBox.className = 'color-box';
            colorBox.style.backgroundColor = state.visibleStats[config.key] ? config.color : '#ccc';
            colorBox.style.display = 'flex';
            colorBox.style.alignItems = 'center';
            colorBox.style.justifyContent = 'center';

            const svgIcon = d3.create('svg')
                .attr('width', 16)
                .attr('height', 16)
                .attr('viewBox', '-8 -8 16 16')
                .style('display', 'block');

            const symbolType = statsSymbolMap[config.marker] || d3.symbolCircle;
            const symbol = d3.symbol().type(symbolType).size(60);

            svgIcon.append('path')
                .attr('d', symbol)
                .attr('transform', config.marker === 'triangle-down' ? 'rotate(180)' : '')
                .attr('fill', 'rgba(255, 255, 255, 0.9)')
                .attr('stroke', 'rgba(0, 0, 0, 0.6)')
                .attr('stroke-width', 1);

            colorBox.appendChild(svgIcon.node());

            const label = document.createElement('span');
            label.textContent = config.label;
            label.style.color = state.visibleStats[config.key] ? 'var(--lejepa-text)' : 'var(--lejepa-text-muted)';

            item.appendChild(colorBox);
            item.appendChild(label);
            legendContainer.appendChild(item);
        });
    }
}

function renderAll(data) {
    drawMarginals(data.samples);
    drawDensity(data.density, data.vectors);
    drawProjections(data.projections, data.vectors);
    drawStats(data.stats.normalized);
}

async function regenerate() {
    if (isRunning) return;
    isRunning = true;
    await Promise.resolve();

    const samples = generateXDistribution(state.nSamples, state.rho);
    const { vectors, projections } = projectSamples(samples, state.mDirections);
    const density = computeDensityField(state.rho);
    const stats = computeStats(projections);

    cachedResult = { samples, vectors, projections, density, stats };
    renderAll(cachedResult);
    isRunning = false;
}

function setLabel(id, value) {
    if (labels[id]) {
        labels[id].textContent = typeof value === 'number' ? value.toString() : value;
    }
}

function handleInputChange(evt) {
    if (!evt?.target) return;
    const { id, value } = evt.target;
    if (id === 'slicing-example-n') {
        state.nSamples = Number(value);
        setLabel('n', state.nSamples);
    } else if (id === 'slicing-example-rho') {
        state.rho = Number(value);
        setLabel('rho', state.rho.toFixed(2));
    } else if (id === 'slicing-example-m') {
        state.mDirections = Number(value);
        setLabel('m', state.mDirections);
    }
    debouncedRegenerate();
}

function debounce(fn, delay) {
    let timer;
    return function debounced(...args) {
        clearTimeout(timer);
        timer = setTimeout(() => fn.apply(this, args), delay);
    };
}

const debouncedRegenerate = debounce(regenerate, 200);

['n', 'rho', 'm'].forEach(key => {
    if (inputs[key]) {
        inputs[key].addEventListener('input', handleInputChange);
    }
});

window.addEventListener('resize', () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => {
        if (cachedResult) {
            renderAll(cachedResult);
        }
    }, 200);
});

const themeObserver = new MutationObserver(() => {
    if (cachedResult) {
        renderAll(cachedResult);
    }
});

themeObserver.observe(document.documentElement, {
    attributes: true,
    attributeFilter: ['class', 'data-theme']
});

renderAxisLabel('slicing-example-label-n', 'n');
renderAxisLabel('slicing-example-label-rho', '\\rho');
renderAxisLabel('slicing-example-label-m', 'M');
renderAxisLabel('slicing-example-label-m-title', 'M');

setLabel('n', state.nSamples);
setLabel('rho', state.rho.toFixed(2));
setLabel('m', state.mDirections);

regenerate();
