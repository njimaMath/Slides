/**
 * cell_colony.js — Eden model cell colony growth animation
 *
 * A cell colony expands by randomly filling adjacent empty sites on a
 * circular petri dish, producing a rough fractal-like boundary.
 *
 * Usage:
 *   <div id="colony-figure"></div>
 *   <script src="cell_colony.js"></script>
 *
 * If #colony-figure is absent the figure is appended to document.body.
 */
(function () {
    /* ── tunables ── */
    const GRID       = 600;     // lattice side length (cells)
    const CELL_PX    = 1;       // 1 px per cell → canvas is GRID × GRID
    const BATCH      = 180;     // cells added per animation frame
    const MAX_CELLS  = 120000;  // stop after this many cells
    const FPS        = 60;

    /* ── colour palette (agar plate + colony) ── */
    const AGAR_CENTRE  = [58, 38, 28];    // warm dark-brown centre
    const AGAR_EDGE    = [72, 50, 38];    // lighter rim
    const COLONY_BASE  = [235, 225, 200]; // cream / off-white base
    const COLONY_VAR   = 18;              // per-channel noise amplitude

    /* ── DOM setup ── */
    const mount = document.getElementById('colony-figure') || document.body;

    const wrapper = document.createElement('div');
    wrapper.style.cssText =
        'display:inline-flex;flex-direction:column;align-items:center;gap:8px;' +
        'font-family:"Helvetica Neue",sans-serif;';

    const canvas = document.createElement('canvas');
    const cSize  = GRID * CELL_PX;
    canvas.width  = cSize;
    canvas.height = cSize;
    canvas.style.cssText =
        'width:min(55vh,55vw);height:min(55vh,55vw);' +
        'border-radius:50%;box-shadow:0 0 40px rgba(0,0,0,0.25);display:block;';

    const btnRow = document.createElement('div');
    btnRow.style.cssText = 'display:flex;gap:12px;align-items:center;';

    const btn = document.createElement('button');
    btn.textContent = '▶ Restart';
    btn.style.cssText =
        'background:#e74c3c;color:#fff;border:none;border-radius:6px;' +
        'padding:6px 16px;font-size:14px;cursor:pointer;font-weight:500;letter-spacing:0.05em;';

    const counter = document.createElement('span');
    counter.style.cssText = 'font-size:13px;color:#999;letter-spacing:0.05em;';
    counter.textContent   = 'cells: 0';

    btnRow.append(btn, counter);
    wrapper.append(canvas, btnRow);
    mount.appendChild(wrapper);

    /* ── canvas state ── */
    const ctx      = canvas.getContext('2d');
    const occupied   = new Uint8Array(GRID * GRID);
    const inFrontier = new Uint8Array(GRID * GRID);
    let   frontier   = [];
    let   cellCount  = 0;
    let   animId     = null;
    let   imgData    = ctx.createImageData(cSize, cSize);

    const cx = GRID >> 1, cy = GRID >> 1;
    const R  = (GRID >> 1) - 2;   // petri-dish visible radius

    const idx  = (x, y) => y * GRID + x;
    const dirs = [[1,0],[-1,0],[0,1],[0,-1]];

    /* ── background ── */
    function paintAgar() {
        const d = imgData.data;
        for (let y = 0; y < GRID; y++) {
            for (let x = 0; x < GRID; x++) {
                const dx = x - cx, dy = y - cy;
                const r  = Math.sqrt(dx*dx + dy*dy);
                const t  = Math.min(r / R, 1);
                const p  = (y * GRID + x) * 4;
                if (r > R + 1) {
                    d[p] = d[p+1] = d[p+2] = 30; d[p+3] = 255;
                } else {
                    const n = (Math.random() - 0.5) * 8;
                    d[p]   = AGAR_CENTRE[0] + (AGAR_EDGE[0] - AGAR_CENTRE[0]) * t + n | 0;
                    d[p+1] = AGAR_CENTRE[1] + (AGAR_EDGE[1] - AGAR_CENTRE[1]) * t + n | 0;
                    d[p+2] = AGAR_CENTRE[2] + (AGAR_EDGE[2] - AGAR_CENTRE[2]) * t + n | 0;
                    d[p+3] = 255;
                }
            }
        }
    }

    /* ── paint a single colony cell ── */
    function paintCell(x, y) {
        const p = (y * GRID + x) * 4;
        const d = imgData.data;
        const n1 = (Math.random() - 0.5) * COLONY_VAR * 2;
        const n2 = (Math.random() - 0.5) * COLONY_VAR * 2;
        const n3 = (Math.random() - 0.5) * COLONY_VAR * 2;
        const dx = x - cx, dy = y - cy;
        const dist = Math.sqrt(dx*dx + dy*dy);
        const edgeDim = Math.max(0, 1 - 0.15 * (dist / R));
        d[p]   = Math.min(255, Math.max(0, (COLONY_BASE[0] + n1) * edgeDim)) | 0;
        d[p+1] = Math.min(255, Math.max(0, (COLONY_BASE[1] + n2) * edgeDim)) | 0;
        d[p+2] = Math.min(255, Math.max(0, (COLONY_BASE[2] + n3) * edgeDim)) | 0;
        d[p+3] = 255;
    }

    /* ── add a cell and update the frontier ── */
    function addCell(x, y) {
        occupied[idx(x,y)] = 1;
        paintCell(x, y);
        cellCount++;
        for (const [dx, dy] of dirs) {
            const nx = x + dx, ny = y + dy;
            if (nx < 0 || nx >= GRID || ny < 0 || ny >= GRID) continue;
            const ni = idx(nx, ny);
            if (!occupied[ni] && !inFrontier[ni]) {
                const ddx = nx - cx, ddy = ny - cy;
                if (ddx*ddx + ddy*ddy > R*R) continue;
                frontier.push(ni);
                inFrontier[ni] = 1;
            }
        }
    }

    /* ── reset to initial state ── */
    function reset() {
        if (animId) { cancelAnimationFrame(animId); animId = null; }
        occupied.fill(0);
        inFrontier.fill(0);
        frontier  = [];
        cellCount = 0;
        paintAgar();
        for (let dy = -1; dy <= 1; dy++)
            for (let dx = -1; dx <= 1; dx++)
                addCell(cx + dx, cy + dy);
        ctx.putImageData(imgData, 0, 0);
        counter.textContent = 'cells: ' + cellCount;
    }

    /* ── animation loop ── */
    let lastTime = 0;
    const interval = 1000 / FPS;

    function frame(ts) {
        animId = requestAnimationFrame(frame);
        if (ts - lastTime < interval) return;
        lastTime = ts;

        if (frontier.length === 0 || cellCount >= MAX_CELLS) {
            cancelAnimationFrame(animId); animId = null; return;
        }

        const steps = Math.min(BATCH, frontier.length);
        for (let i = 0; i < steps; i++) {
            const ri = (Math.random() * frontier.length) | 0;
            const ni = frontier[ri];
            frontier[ri] = frontier[frontier.length - 1];
            frontier.pop();
            inFrontier[ni] = 0;
            const fx = ni % GRID, fy = (ni / GRID) | 0;
            if (occupied[ni]) continue;
            addCell(fx, fy);
        }
        ctx.putImageData(imgData, 0, 0);
        counter.textContent = 'cells: ' + cellCount.toLocaleString();
    }

    function start() { reset(); animId = requestAnimationFrame(frame); }

    btn.addEventListener('click', start);

    /* auto-start when figure scrolls into view */
    if (typeof IntersectionObserver !== 'undefined') {
        new IntersectionObserver(entries => {
            entries.forEach(e => { if (e.isIntersecting && !animId) start(); });
        }, { threshold: 0.5 }).observe(wrapper);
    } else {
        start();
    }
})();
