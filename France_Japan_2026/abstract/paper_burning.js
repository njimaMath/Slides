/**
 * paper_burning.js — Stochastic paper-burning / fire-spread animation
 *
 * A piece of paper on a dark table is ignited at several bottom-edge points.
 * Fire spreads stochastically to neighbouring cells on a 2-D grid.
 * Each cell passes through: unburned paper → burning (flame) → ash (charcoal).
 *
 * Usage:
 *   <div id="burning-figure"></div>
 *   <script src="paper_burning.js"></script>
 *
 * If #burning-figure is absent the figure is appended to document.body.
 */
(function () {
    /* ── grid dimensions ── */
    const COLS      = 180;   // cells across
    const ROWS      = 120;   // cells down
    const CELL      = 4;     // pixels per cell
    const W         = COLS * CELL;
    const H         = ROWS * CELL;

    /* ── fire parameters ── */
    const BURN_STEPS = 16;    // how many sim-steps a cell spends burning
    const SPREAD_P   = 0.18;  // ignition probability per burning neighbour per step
    const UP_BIAS    = 1.6;   // fire spreads upward this many times more readily
    const SIM_FPS   = 28;     // simulation steps per second

    /* ── colour helpers ── */
    // Per-cell paper texture noise (generated once, stored as [r,g,b] per cell)
    const paperNoise = new Uint8Array(COLS * ROWS * 3);
    for (let i = 0; i < paperNoise.length; i += 3) {
        const n = (Math.random() - 0.5) * 20 | 0;
        paperNoise[i]     = Math.max(0, Math.min(255, 238 + n));
        paperNoise[i + 1] = Math.max(0, Math.min(255, 224 + n));
        paperNoise[i + 2] = Math.max(0, Math.min(255, 188 + n));
    }

    // Ash also gets per-cell variation
    const ashNoise = new Uint8Array(COLS * ROWS * 3);
    for (let i = 0; i < ashNoise.length; i += 3) {
        const n = (Math.random() - 0.5) * 12 | 0;
        ashNoise[i]     = Math.max(0, Math.min(80, 30 + n));
        ashNoise[i + 1] = Math.max(0, Math.min(40, 22 + n));
        ashNoise[i + 2] = Math.max(0, Math.min(25, 16 + n));
    }

    /**
     * Fire colour based on normalised lifetime t ∈ [0, 1].
     * t = 1 → just ignited (bright yellow-white)
     * t = 0 → about to die (deep red / near-black)
     */
    function fireColor(t) {
        if (t > 0.80) {
            // white-hot → bright yellow
            const s = (t - 0.80) / 0.20;
            return [255, Math.round(240 - 10 * (1 - s)), Math.round(180 * s)];
        } else if (t > 0.55) {
            // bright yellow → orange
            const s = (t - 0.55) / 0.25;
            return [255, Math.round(180 * s + 30), 0];
        } else if (t > 0.30) {
            // orange → red
            const s = (t - 0.30) / 0.25;
            return [255, Math.round(30 * s), 0];
        } else if (t > 0.10) {
            // red → dark red
            const s = (t - 0.10) / 0.20;
            return [Math.round(180 * s + 60), 0, 0];
        } else {
            // darkening ember
            const s = t / 0.10;
            return [Math.round(60 * s + 10), 0, 0];
        }
    }

    /* ── DOM setup ── */
    const mount = document.getElementById('burning-figure') || document.body;

    const wrapper = document.createElement('div');
    wrapper.style.cssText =
        'display:inline-flex;flex-direction:column;align-items:center;gap:10px;' +
        'font-family:"Helvetica Neue",sans-serif;background:#111;padding:20px;' +
        'border-radius:8px;';

    const canvas = document.createElement('canvas');
    canvas.width  = W;
    canvas.height = H;
    canvas.style.cssText =
        `display:block;width:${W}px;height:${H}px;` +
        'box-shadow:0 0 30px rgba(255,140,0,0.25);border-radius:2px;';

    const btnRow = document.createElement('div');
    btnRow.style.cssText = 'display:flex;gap:12px;align-items:center;';

    const btn = document.createElement('button');
    btn.textContent = '🔥 Restart';
    btn.style.cssText =
        'background:#c0392b;color:#fff;border:none;border-radius:6px;' +
        'padding:6px 16px;font-size:13px;cursor:pointer;font-weight:500;';

    const statusEl = document.createElement('span');
    statusEl.style.cssText = 'font-size:12px;color:#aaa;letter-spacing:0.05em;';
    statusEl.textContent = 'burning…';

    btnRow.append(btn, statusEl);
    wrapper.append(canvas, btnRow);
    mount.appendChild(wrapper);

    const ctx     = canvas.getContext('2d');
    const imgData = ctx.createImageData(W, H);

    /* ── simulation state ── */
    //  state[i] : 0 = paper | 1..BURN_STEPS = burning | BURN_STEPS+1 = ash
    const ASH_ST = BURN_STEPS + 1;
    const state  = new Int16Array(COLS * ROWS);
    let   front  = [];   // array of {ci, timer} for currently-burning cells
    let   animId = null;
    let   lastT  = 0;
    const interval = 1000 / SIM_FPS;
    const dirs4 = [[0, -1, UP_BIAS], [0, 1, 1], [-1, 0, 1], [1, 0, 1]]; // [dx, dy, weight]

    /* ── pixel helpers ── */
    function setCell(x, y, r, g, b) {
        for (let dy = 0; dy < CELL; dy++) {
            for (let dx = 0; dx < CELL; dx++) {
                const p = ((y * CELL + dy) * W + (x * CELL + dx)) * 4;
                imgData.data[p]     = r;
                imgData.data[p + 1] = g;
                imgData.data[p + 2] = b;
                imgData.data[p + 3] = 255;
            }
        }
    }

    /* ── reset & paint blank paper ── */
    function reset() {
        if (animId) { cancelAnimationFrame(animId); animId = null; }
        state.fill(0);
        front = [];

        // Draw background (dark table)
        imgData.data.fill(0);
        for (let i = 3; i < imgData.data.length; i += 4) imgData.data[i] = 255;

        // Draw paper with texture
        for (let y = 0; y < ROWS; y++) {
            for (let x = 0; x < COLS; x++) {
                const ni = (y * COLS + x) * 3;
                setCell(x, y, paperNoise[ni], paperNoise[ni+1], paperNoise[ni+2]);
            }
        }

        // Optional: faint horizontal lines (like ruled paper)
        for (let y = 8; y < ROWS; y += 8) {
            for (let x = 0; x < COLS; x++) {
                const ni = (y * COLS + x) * 3;
                const r = Math.max(0, paperNoise[ni]   - 14);
                const g = Math.max(0, paperNoise[ni+1] - 10);
                const b = Math.max(0, paperNoise[ni+2] - 6);
                // Only the top pixel row of each ruled line
                const p = (y * CELL * W + x * CELL) * 4;
                for (let dx = 0; dx < CELL; dx++) {
                    imgData.data[p + dx*4]     = r;
                    imgData.data[p + dx*4 + 1] = g;
                    imgData.data[p + dx*4 + 2] = b;
                    imgData.data[p + dx*4 + 3] = 255;
                }
            }
        }

        ctx.putImageData(imgData, 0, 0);
        statusEl.textContent = 'burning…';

        // Ignite a cluster of cells along the bottom edge
        const startX = COLS >> 1;
        const startY = ROWS - 1;
        for (let dx = -2; dx <= 2; dx++) ignite(startX + dx, startY);

        animId = requestAnimationFrame(frame);
    }

    function ignite(x, y) {
        if (x < 0 || x >= COLS || y < 0 || y >= ROWS) return;
        const ci = y * COLS + x;
        if (state[ci] !== 0) return;
        state[ci] = BURN_STEPS;
        const [r, g, b] = fireColor(1.0);
        setCell(x, y, r, g, b);
        front.push(ci);
    }

    /* ── one simulation step ── */
    function simStep() {
        const nextFront = [];
        const newly     = [];

        for (const ci of front) {
            const s = state[ci];
            if (s <= 0 || s === ASH_ST) continue;

            const cx = ci % COLS, cy = (ci / COLS) | 0;

            if (s === 1) {
                // transition to ash
                state[ci] = ASH_ST;
                const ni = ci * 3;
                setCell(cx, cy, ashNoise[ni], ashNoise[ni+1], ashNoise[ni+2]);
            } else {
                state[ci] = s - 1;
                const t = (s - 1) / BURN_STEPS;
                const [r, g, b] = fireColor(t);
                // add flicker via slight colour jitter
                const jit = ((Math.random() - 0.5) * 20) | 0;
                setCell(cx, cy,
                    Math.max(0, Math.min(255, r + jit)),
                    Math.max(0, Math.min(255, g + (jit >> 1))),
                    Math.max(0, b));
                nextFront.push(ci);

                // stochastic spread to neighbours
                for (const [dx, dy, weight] of dirs4) {
                    const nx = cx + dx, ny = cy + dy;
                    if (nx < 0 || nx >= COLS || ny < 0 || ny >= ROWS) continue;
                    const ni = ny * COLS + nx;
                    if (state[ni] === 0 && Math.random() < SPREAD_P * weight) {
                        state[ni] = BURN_STEPS;
                        newly.push(ni);
                        const [fr, fg, fb] = fireColor(1.0);
                        setCell(nx, ny, fr, fg, fb);
                    }
                }
            }
        }

        front = nextFront.concat(newly);

        // Update status
        const ashCount = state.reduce((acc, v) => acc + (v === ASH_ST ? 1 : 0), 0);
        const pct = ((ashCount / (COLS * ROWS)) * 100).toFixed(1);
        statusEl.textContent = front.length > 0
            ? `burning… ${pct}% ash`
            : `✓ fully burned (${pct}% ash)`;
    }

    /* ── animation loop ── */
    function frame(ts) {
        animId = requestAnimationFrame(frame);
        if (ts - lastT < interval) return;
        lastT = ts;
        simStep();
        ctx.putImageData(imgData, 0, 0);
        if (front.length === 0) {
            cancelAnimationFrame(animId); animId = null;
        }
    }

    btn.addEventListener('click', reset);

    /* auto-start on scroll into view */
    if (typeof IntersectionObserver !== 'undefined') {
        new IntersectionObserver(entries => {
            entries.forEach(e => { if (e.isIntersecting && !animId) reset(); });
        }, { threshold: 0.5 }).observe(wrapper);
    } else {
        reset();
    }
})();
