/**
 * app.js — Dashboard controller for AL-HSI-SAM2 visualization.
 *
 * Renders 4 synchronized panels:
 *   1. False-color composite (PCA → RGB)
 *   2. Segmentation map (GT / predicted, toggleable)
 *   3. Uncertainty heatmap (BALD/entropy, inferno colormap)
 *   4. AL query locations + annotation-efficiency chart
 *
 * Data sources:
 *   - Exported JSON from visualization/export_results.py
 *   - OR: synthetic demo data generated locally for demonstration
 *
 * NOTE: This is a SIMULATED multi-panel display, not a real Liquid Galaxy.
 */

// ============================================================================
// Color Palettes
// ============================================================================

/** Indian Pines class colors (16 vegetation/land-use classes + background) */
const CLASS_COLORS_IP = [
    [30, 30, 30],      // 0: Background
    [140, 67, 46],      // 1: Alfalfa
    [0, 128, 0],        // 2: Corn-notill
    [160, 220, 100],    // 3: Corn-mintill
    [30, 170, 60],      // 4: Corn
    [128, 128, 0],      // 5: Grass-pasture
    [34, 139, 34],      // 6: Grass-trees
    [107, 142, 35],     // 7: Grass-pasture-mowed
    [210, 180, 140],    // 8: Hay-windrowed
    [255, 69, 0],       // 9: Oats
    [184, 134, 11],     // 10: Soybean-notill
    [218, 165, 32],     // 11: Soybean-mintill
    [189, 183, 107],    // 12: Soybean-clean
    [255, 215, 0],      // 13: Wheat
    [85, 107, 47],      // 14: Woods
    [148, 103, 189],    // 15: Buildings-Grass-Trees-Drives
    [112, 128, 144],    // 16: Stone-Steel-Towers
];

const CLASS_NAMES_IP = [
    "Background", "Alfalfa", "Corn-notill", "Corn-mintill", "Corn",
    "Grass-pasture", "Grass-trees", "Grass-mowed", "Hay-windrowed", "Oats",
    "Soybean-notill", "Soybean-mintill", "Soybean-clean", "Wheat", "Woods",
    "Bldg-Grass-Trees", "Stone-Steel",
];

/** Pavia University class colors (9 classes + background) */
const CLASS_COLORS_PV = [
    [30, 30, 30],       // 0: Background
    [192, 0, 0],        // 1: Asphalt
    [0, 128, 0],        // 2: Meadows
    [128, 128, 128],    // 3: Gravel
    [34, 139, 34],      // 4: Trees
    [112, 128, 144],    // 5: Metal Sheets
    [210, 180, 140],    // 6: Bare Soil
    [100, 100, 200],    // 7: Bitumen
    [255, 69, 0],       // 8: Bricks
    [70, 130, 180],     // 9: Shadows
];

const CLASS_NAMES_PV = [
    "Background", "Asphalt", "Meadows", "Gravel", "Trees",
    "Metal Sheets", "Bare Soil", "Bitumen", "Bricks", "Shadows",
];

/** Inferno colormap lookup (0 → 1 mapped to RGB) */
function infernoColor(t) {
    t = Math.max(0, Math.min(1, t));
    // Simplified inferno using key stops
    const stops = [
        [0.0, 13, 8, 135],
        [0.13, 70, 3, 159],
        [0.25, 114, 1, 168],
        [0.38, 156, 23, 158],
        [0.5, 189, 55, 134],
        [0.63, 216, 87, 107],
        [0.75, 237, 121, 83],
        [0.88, 251, 159, 58],
        [1.0, 240, 249, 33],
    ];
    // Find the two stops we're between
    let lo = stops[0], hi = stops[stops.length - 1];
    for (let i = 0; i < stops.length - 1; i++) {
        if (t >= stops[i][0] && t <= stops[i + 1][0]) {
            lo = stops[i];
            hi = stops[i + 1];
            break;
        }
    }
    const f = (t - lo[0]) / (hi[0] - lo[0] + 1e-8);
    return [
        Math.round(lo[1] + f * (hi[1] - lo[1])),
        Math.round(lo[2] + f * (hi[2] - lo[2])),
        Math.round(lo[3] + f * (hi[3] - lo[3])),
    ];
}

// ============================================================================
// State
// ============================================================================

const state = {
    dataset: "indian_pines",
    round: 0,
    maxRound: 0,
    strategy: "bald",
    showGT: true,
    playing: false,
    playInterval: null,

    // Data (null until loaded)
    falseColor: null,       // { width, height, pixels: number[][][] }
    uncertainty: null,      // { width, height, values: number[][] }
    segmentation: null,     // { width, height, ground_truth, predictions }
    queryHistory: null,     // [ { round, coordinates, num_new } ]
    metrics: null,          // { initial, rounds, strategy }

    classColors: CLASS_COLORS_IP,
    classNames: CLASS_NAMES_IP,
};

// ============================================================================
// DOM References
// ============================================================================

const $ = (sel) => document.querySelector(sel);
const canvasRGB = $("#canvas-rgb");
const canvasSeg = $("#canvas-seg");
const canvasUnc = $("#canvas-uncertainty");
const canvasQueries = $("#canvas-queries");
const chartCanvas = $("#chart-efficiency");

const roundSlider = $("#round-slider");
const roundDisplay = $("#round-display");
const datasetSelect = $("#dataset-select");
const strategySelect = $("#strategy-select");
const toggleGT = $("#toggle-gt");
const playBtn = $("#play-btn");
const loadDemoBtn = $("#load-demo-btn");
const exportBtn = $("#export-btn");
const statusText = $("#status-text");

// ============================================================================
// Demo Data Generator
// ============================================================================

/**
 * Generate synthetic demo data for visualization when real data isn't available.
 * This creates a plausible-looking HSI scene with class regions and uncertainty.
 */
function generateDemoData(dataset = "indian_pines") {
    const W = 145, H = 145;
    const numClasses = dataset === "indian_pines" ? 17 : 10;
    const numRounds = 10;

    state.classColors = dataset === "indian_pines" ? CLASS_COLORS_IP : CLASS_COLORS_PV;
    state.classNames = dataset === "indian_pines" ? CLASS_NAMES_IP : CLASS_NAMES_PV;

    // --- Generate false color (procedural landscape) ---
    const pixels = [];
    const gt = [];
    const pred = [];
    const uncValues = [];

    // Create class regions using simple Voronoi-like pattern
    const seeds = [];
    const rng = mulberry32(42);
    for (let c = 0; c < numClasses; c++) {
        seeds.push({
            x: Math.floor(rng() * W),
            y: Math.floor(rng() * H),
            cls: c,
        });
    }

    for (let y = 0; y < H; y++) {
        const row_px = [], row_gt = [], row_pred = [], row_unc = [];
        for (let x = 0; x < W; x++) {
            // Nearest seed → class assignment
            let minDist = Infinity, cls = 0;
            for (const s of seeds) {
                const d = Math.hypot(x - s.x, y - s.y);
                if (d < minDist) { minDist = d; cls = s.cls; }
            }

            // False color: class color + noise
            const baseColor = state.classColors[cls] || [100, 100, 100];
            row_px.push([
                clamp(baseColor[0] + (rng() - 0.5) * 40, 0, 255),
                clamp(baseColor[1] + (rng() - 0.5) * 40, 0, 255),
                clamp(baseColor[2] + (rng() - 0.5) * 40, 0, 255),
            ]);

            row_gt.push(cls);

            // Predictions: mostly correct, with noise near boundaries
            if (minDist < 5 || rng() > 0.85) {
                // Near boundary or random error
                row_pred.push(Math.floor(rng() * numClasses));
            } else {
                row_pred.push(cls);
            }

            // Uncertainty: higher near class boundaries
            const boundaryFactor = Math.exp(-minDist / 8);
            row_unc.push(boundaryFactor * 0.8 + rng() * 0.2);
        }
        pixels.push(row_px);
        gt.push(row_gt);
        pred.push(row_pred);
        uncValues.push(row_unc);
    }

    state.falseColor = { width: W, height: H, pixels };
    state.segmentation = { width: W, height: H, ground_truth: gt, predictions: pred };
    state.uncertainty = { width: W, height: H, values: uncValues, min_raw: 0, max_raw: 1 };

    // --- Generate query history ---
    const queryHistory = [];
    for (let r = 1; r <= numRounds; r++) {
        const coords = [];
        const numQueries = 20 + Math.floor(rng() * 30);
        for (let q = 0; q < numQueries; q++) {
            // Bias queries toward high uncertainty regions
            let bestX = 0, bestY = 0, bestUnc = -1;
            for (let attempt = 0; attempt < 5; attempt++) {
                const cx = Math.floor(rng() * W);
                const cy = Math.floor(rng() * H);
                if (uncValues[cy][cx] > bestUnc) {
                    bestUnc = uncValues[cy][cx];
                    bestX = cx; bestY = cy;
                }
            }
            coords.push([bestY, bestX]);
        }
        queryHistory.push({ round: r, coordinates: coords, num_new: coords.length });
    }
    state.queryHistory = queryHistory;

    // --- Generate metrics ---
    const baseMiou = 0.25;
    const rounds = [];
    let labeled = 500;
    for (let r = 1; r <= numRounds; r++) {
        labeled += 30 + Math.floor(rng() * 20);
        const miou = Math.min(0.92, baseMiou + 0.06 * r + rng() * 0.02);
        rounds.push({
            round: r,
            labeled_count: labeled,
            miou: miou,
            mean_entropy: 0.8 - 0.05 * r,
            mean_bald: 0.5 - 0.03 * r,
        });
    }
    state.metrics = {
        initial: { labeled_count: 500, miou: baseMiou },
        rounds,
        strategy: "bald",
    };

    state.maxRound = numRounds;
    roundSlider.max = numRounds;
}

// ============================================================================
// Rendering Functions
// ============================================================================

function renderFalseColor() {
    if (!state.falseColor) return;
    const { width, height, pixels } = state.falseColor;
    canvasRGB.width = width;
    canvasRGB.height = height;
    const ctx = canvasRGB.getContext("2d");
    const imgData = ctx.createImageData(width, height);

    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const idx = (y * width + x) * 4;
            const px = pixels[y][x];
            imgData.data[idx] = Math.round(px[0]);
            imgData.data[idx + 1] = Math.round(px[1]);
            imgData.data[idx + 2] = Math.round(px[2]);
            imgData.data[idx + 3] = 255;
        }
    }
    ctx.putImageData(imgData, 0, 0);
    $("#stat-resolution").textContent = `${width} × ${height}`;
}

function renderSegmentation() {
    if (!state.segmentation) return;
    const { width, height, ground_truth, predictions } = state.segmentation;
    canvasSeg.width = width;
    canvasSeg.height = height;
    const ctx = canvasSeg.getContext("2d");
    const imgData = ctx.createImageData(width, height);

    const mapData = state.showGT ? ground_truth : predictions;
    const colors = state.classColors;

    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const idx = (y * width + x) * 4;
            const cls = mapData[y][x];
            const color = colors[cls] || [50, 50, 50];
            imgData.data[idx] = color[0];
            imgData.data[idx + 1] = color[1];
            imgData.data[idx + 2] = color[2];
            imgData.data[idx + 3] = 255;
        }
    }
    ctx.putImageData(imgData, 0, 0);

    // Update metrics display
    if (state.metrics) {
        const roundData = getCurrentRoundMetrics();
        if (roundData) {
            $("#stat-miou").textContent = `mIoU: ${roundData.miou.toFixed(4)}`;
            $("#stat-oa").textContent = `Round ${state.round}`;
        }
    }
}

function renderUncertainty() {
    if (!state.uncertainty) return;
    const { width, height, values } = state.uncertainty;
    canvasUnc.width = width;
    canvasUnc.height = height;
    const ctx = canvasUnc.getContext("2d");
    const imgData = ctx.createImageData(width, height);

    let sum = 0, maxVal = 0;
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const idx = (y * width + x) * 4;
            const v = values[y][x];
            const [r, g, b] = infernoColor(v);
            imgData.data[idx] = r;
            imgData.data[idx + 1] = g;
            imgData.data[idx + 2] = b;
            imgData.data[idx + 3] = 255;
            sum += v;
            if (v > maxVal) maxVal = v;
        }
    }
    ctx.putImageData(imgData, 0, 0);

    const mean = sum / (width * height);
    $("#stat-mean-unc").textContent = `Mean: ${mean.toFixed(3)}`;
    $("#stat-max-unc").textContent = `Max: ${maxVal.toFixed(3)}`;
}

function renderQueryMap() {
    if (!state.falseColor || !state.queryHistory) return;
    const { width, height, pixels } = state.falseColor;
    canvasQueries.width = width;
    canvasQueries.height = height;
    const ctx = canvasQueries.getContext("2d");

    // Draw dimmed false-color background
    const imgData = ctx.createImageData(width, height);
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const idx = (y * width + x) * 4;
            const px = pixels[y][x];
            imgData.data[idx] = Math.round(px[0] * 0.3);
            imgData.data[idx + 1] = Math.round(px[1] * 0.3);
            imgData.data[idx + 2] = Math.round(px[2] * 0.3);
            imgData.data[idx + 3] = 255;
        }
    }
    ctx.putImageData(imgData, 0, 0);

    // Draw query markers for rounds up to current
    const roundColors = [
        "#6366f1", "#06b6d4", "#10b981", "#f59e0b", "#f43f5e",
        "#8b5cf6", "#ec4899", "#14b8a6", "#84cc16", "#f97316",
    ];

    for (let r = 0; r < Math.min(state.round, state.queryHistory.length); r++) {
        const roundData = state.queryHistory[r];
        const color = roundColors[r % roundColors.length];
        ctx.fillStyle = color;
        ctx.globalAlpha = 0.8;

        for (const coord of roundData.coordinates) {
            const [row, col] = coord;
            ctx.beginPath();
            ctx.arc(col, row, 2, 0, Math.PI * 2);
            ctx.fill();
        }
    }
    ctx.globalAlpha = 1.0;

    // Update labels
    const totalLabeled = state.round > 0 && state.metrics?.rounds?.length >= state.round
        ? state.metrics.rounds[state.round - 1].labeled_count
        : state.metrics?.initial?.labeled_count || 0;
    $("#stat-labeled").textContent = `Labeled: ${totalLabeled}`;
    $("#stat-round-info").textContent = `Round: ${state.round}/${state.maxRound}`;
}

function renderEfficiencyChart() {
    if (!state.metrics) return;
    const canvas = chartCanvas;
    const ctx = canvas.getContext("2d");

    // Get container dimensions
    const rect = canvas.parentElement.getBoundingClientRect();
    canvas.width = rect.width * window.devicePixelRatio;
    canvas.height = rect.height * window.devicePixelRatio;
    ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
    const W = rect.width;
    const H = rect.height;

    // Clear
    ctx.fillStyle = getComputedStyle(document.documentElement)
        .getPropertyValue('--bg-primary').trim();
    ctx.fillRect(0, 0, W, H);

    // Data points
    const points = [];
    if (state.metrics.initial) {
        points.push({
            x: state.metrics.initial.labeled_count,
            y: state.metrics.initial.miou,
        });
    }
    for (const r of state.metrics.rounds) {
        points.push({ x: r.labeled_count, y: r.miou });
    }

    if (points.length < 2) return;

    // Compute axis ranges
    const xMin = Math.min(...points.map(p => p.x)) * 0.9;
    const xMax = Math.max(...points.map(p => p.x)) * 1.1;
    const yMin = 0;
    const yMax = 1;

    // Chart area (with padding for labels)
    const pad = { top: 16, right: 16, bottom: 28, left: 42 };
    const cw = W - pad.left - pad.right;
    const ch = H - pad.top - pad.bottom;

    const mapX = (v) => pad.left + ((v - xMin) / (xMax - xMin)) * cw;
    const mapY = (v) => pad.top + ch - ((v - yMin) / (yMax - yMin)) * ch;

    // Grid lines
    ctx.strokeStyle = "rgba(255,255,255,0.06)";
    ctx.lineWidth = 1;
    for (let y = 0; y <= 1; y += 0.2) {
        ctx.beginPath();
        ctx.moveTo(pad.left, mapY(y));
        ctx.lineTo(W - pad.right, mapY(y));
        ctx.stroke();
    }

    // Axis labels
    ctx.fillStyle = "#6b7280";
    ctx.font = "9px Inter, sans-serif";
    ctx.textAlign = "right";
    for (let y = 0; y <= 1; y += 0.2) {
        ctx.fillText(y.toFixed(1), pad.left - 4, mapY(y) + 3);
    }
    ctx.textAlign = "center";
    ctx.fillText("Labeled Pixels", W / 2, H - 2);

    // Draw line
    ctx.strokeStyle = "#6366f1";
    ctx.lineWidth = 2;
    ctx.lineJoin = "round";
    ctx.beginPath();

    // Only draw up to current round
    const activePoints = points.slice(0, state.round + 1);

    for (let i = 0; i < activePoints.length; i++) {
        const px = mapX(activePoints[i].x);
        const py = mapY(activePoints[i].y);
        if (i === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
    }
    ctx.stroke();

    // Draw future rounds as dashed
    if (state.round < points.length - 1) {
        ctx.strokeStyle = "rgba(99, 102, 241, 0.3)";
        ctx.setLineDash([4, 4]);
        ctx.beginPath();
        const startPt = points[state.round];
        ctx.moveTo(mapX(startPt.x), mapY(startPt.y));
        for (let i = state.round + 1; i < points.length; i++) {
            ctx.lineTo(mapX(points[i].x), mapY(points[i].y));
        }
        ctx.stroke();
        ctx.setLineDash([]);
    }

    // Draw dots
    for (let i = 0; i < activePoints.length; i++) {
        const px = mapX(activePoints[i].x);
        const py = mapY(activePoints[i].y);
        const isCurrentRound = i === state.round;

        ctx.beginPath();
        ctx.arc(px, py, isCurrentRound ? 5 : 3, 0, Math.PI * 2);
        ctx.fillStyle = isCurrentRound ? "#06b6d4" : "#6366f1";
        ctx.fill();
        ctx.strokeStyle = "#1a1f36";
        ctx.lineWidth = 1.5;
        ctx.stroke();

        // Label current round
        if (isCurrentRound) {
            ctx.fillStyle = "#06b6d4";
            ctx.font = "bold 9px JetBrains Mono, monospace";
            ctx.textAlign = "center";
            ctx.fillText(
                activePoints[i].y.toFixed(3),
                px, py - 10,
            );
        }
    }
}

function getCurrentRoundMetrics() {
    if (!state.metrics) return null;
    if (state.round === 0) return state.metrics.initial;
    if (state.round <= state.metrics.rounds.length) {
        return state.metrics.rounds[state.round - 1];
    }
    return null;
}

// ============================================================================
// Legend
// ============================================================================

function renderLegend() {
    const container = $("#legend-items");
    container.innerHTML = "";
    state.classNames.forEach((name, idx) => {
        if (idx === 0) return; // Skip background
        const color = state.classColors[idx] || [128, 128, 128];
        const item = document.createElement("div");
        item.className = "legend-item";
        item.innerHTML = `
            <span class="legend-swatch" style="background:rgb(${color[0]},${color[1]},${color[2]})"></span>
            <span>${name}</span>
        `;
        container.appendChild(item);
    });
}

// ============================================================================
// Full Render
// ============================================================================

function renderAll() {
    renderFalseColor();
    renderSegmentation();
    renderUncertainty();
    renderQueryMap();
    renderEfficiencyChart();
}

// ============================================================================
// Event Handlers
// ============================================================================

roundSlider.addEventListener("input", (e) => {
    state.round = parseInt(e.target.value);
    roundDisplay.textContent = state.round;
    renderSegmentation();
    renderQueryMap();
    renderEfficiencyChart();
});

datasetSelect.addEventListener("change", (e) => {
    state.dataset = e.target.value;
    state.round = 0;
    roundSlider.value = 0;
    roundDisplay.textContent = "0";
    loadData();
});

strategySelect.addEventListener("change", (e) => {
    state.strategy = e.target.value;
    $("#uncertainty-type").textContent = e.target.value.toUpperCase();
});

toggleGT.addEventListener("click", () => {
    state.showGT = !state.showGT;
    toggleGT.dataset.active = state.showGT.toString();
    toggleGT.textContent = state.showGT ? "Showing GT" : "Showing Pred";
    renderSegmentation();
});

playBtn.addEventListener("click", () => {
    if (state.playing) {
        clearInterval(state.playInterval);
        state.playing = false;
        playBtn.classList.remove("playing");
        playBtn.innerHTML = '<svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor"><polygon points="3,1 13,8 3,15"/></svg>';
    } else {
        state.playing = true;
        playBtn.classList.add("playing");
        playBtn.innerHTML = '<svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor"><rect x="3" y="2" width="4" height="12"/><rect x="9" y="2" width="4" height="12"/></svg>';
        state.playInterval = setInterval(() => {
            state.round = (state.round + 1) % (state.maxRound + 1);
            roundSlider.value = state.round;
            roundDisplay.textContent = state.round;
            renderSegmentation();
            renderQueryMap();
            renderEfficiencyChart();
        }, 1500);
    }
});

loadDemoBtn.addEventListener("click", () => {
    loadData();
});

exportBtn.addEventListener("click", () => {
    // Capture the grid as a composite image
    statusText.textContent = "Export not yet implemented — use screenshots";
});

// Toggle legend on seg panel click
$("#panel-seg .panel__title").addEventListener("click", () => {
    const legend = $("#legend-panel");
    legend.classList.toggle("visible");
});

// ============================================================================
// Data Loading
// ============================================================================

async function loadData() {
    statusText.textContent = "Loading data...";

    // Try loading real data first
    try {
        const dataDir = `data/`;
        const [fcRes, uncRes, segRes, qhRes, metRes] = await Promise.all([
            fetch(dataDir + "false_color.json"),
            fetch(dataDir + "uncertainty_map.json"),
            fetch(dataDir + "segmentation_map.json"),
            fetch(dataDir + "query_history.json"),
            fetch(dataDir + "metrics_summary.json"),
        ]);

        if (fcRes.ok && uncRes.ok && segRes.ok && qhRes.ok && metRes.ok) {
            state.falseColor = await fcRes.json();
            state.uncertainty = await uncRes.json();
            state.segmentation = await segRes.json();
            state.queryHistory = await qhRes.json();
            state.metrics = await metRes.json();
            state.maxRound = state.queryHistory.length;
            roundSlider.max = state.maxRound;

            statusText.textContent = `Loaded real data — ${state.dataset}`;
            exportBtn.disabled = false;
            renderLegend();
            renderAll();
            return;
        }
    } catch (e) {
        // Fall through to demo data
    }

    // Generate synthetic demo data
    generateDemoData(state.dataset);
    statusText.textContent = `Demo data loaded (synthetic) — ${state.dataset}`;
    exportBtn.disabled = false;
    renderLegend();
    renderAll();
}

// ============================================================================
// Utility Functions
// ============================================================================

function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

/** Seeded PRNG (mulberry32) for reproducible demo data */
function mulberry32(seed) {
    return function () {
        seed |= 0; seed = seed + 0x6D2B79F5 | 0;
        let t = Math.imul(seed ^ seed >>> 15, 1 | seed);
        t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
        return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
}

// ============================================================================
// Resize handler
// ============================================================================
let resizeTimer;
window.addEventListener("resize", () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => {
        renderEfficiencyChart();
    }, 150);
});

// ============================================================================
// Initial Load
// ============================================================================
document.addEventListener("DOMContentLoaded", () => {
    // Set initial GT toggle state
    toggleGT.dataset.active = "true";
    toggleGT.textContent = "Showing GT";

    // Mark panels as loading
    document.querySelectorAll(".panel").forEach(p => p.classList.add("panel--loading"));

    statusText.textContent = "Ready — click 'Load Demo Data' to begin";
});
