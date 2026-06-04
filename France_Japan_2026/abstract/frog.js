/**
 * frog.js — Frog Model simulation animation
 *
 * Rules:
 *   - One infected frog starts at the origin; every other vertex holds
 *     one sleeping (healthy) frog.
 *   - Only infected frogs perform independent simple random walks.
 *   - When an infected frog lands on a sleeping frog's vertex, the
 *     sleeping frog wakes up and becomes infected.
 *
 * Usage:
 *   <div id="frog-figure"></div>
 *   <script src="frog.js"></script>
 *
 * If #frog-figure is absent the figure is appended to document.body.
 */
(function () {
    /* ── grid / display constants ── */
    const CELL  = 22;    // px per lattice cell
    const R     = 7;     // grid radius:  −R … +R  →  (2R+1)² vertices
    const W     = 340;   // canvas logical width  (px)
    const H     = 340;   // canvas logical height (px)
    const ox    = W / 2; // canvas centre = lattice origin x
    const oy    = H / 2; // canvas centre = lattice origin y
    const dirs  = [[1,0],[-1,0],[0,1],[0,-1]];

    /* ── frog drawing constants ── */
    const SLEEP_COL   = '#3498db';          // healthy / sleeping
    const INFECT_COL  = '#e74c3c';          // infected / active
    const SLEEP_GLOW  = 'rgba(52,152,219,0.22)';
    const INFECT_GLOW = 'rgba(231,76,60,0.25)';
    const FROG_R      = CELL * 0.42;        // drawn radius

    /* ── DOM setup ── */
    const mount = document.getElementById('frog-figure') || document.body;

    const wrapper = document.createElement('div');
    wrapper.style.cssText =
        'display:inline-flex;flex-direction:column;align-items:center;gap:10px;' +
        'font-family:"Helvetica Neue",sans-serif;';

    const row = document.createElement('div');
    row.style.cssText = 'display:flex;align-items:center;gap:16px;';

    // -- Canvas --
    const canvas = document.createElement('canvas');
    canvas.width  = W;
    canvas.height = H;
    canvas.style.cssText =
        `width:${W}px;height:${H}px;border:1px solid #ddd;` +
        'border-radius:8px;background:#f5f5f0;display:block;';

    // -- Controls panel --
    const panel = document.createElement('div');
    panel.style.cssText =
        'display:flex;flex-direction:column;gap:8px;align-items:flex-start;min-width:120px;';

    function makeButton(label, bg) {
        const b = document.createElement('button');
        b.textContent = label;
        b.style.cssText =
            `padding:6px 14px;font-size:13px;border:none;border-radius:6px;cursor:pointer;` +
            `background:${bg};color:#fff;font-weight:600;width:100%;`;
        return b;
    }

    const btnStart = makeButton('▶ Start',  '#27ae60');
    const btnReset = makeButton('⟳ Reset',  '#e74c3c');

    const speedLabel = document.createElement('label');
    speedLabel.textContent = 'Speed';
    speedLabel.style.cssText = 'font-size:12px;color:#666;margin-top:4px;';

    const speedSel = document.createElement('select');
    speedSel.style.cssText =
        'padding:4px 8px;font-size:12px;border:1px solid #ccc;border-radius:6px;' +
        'background:#fff;color:#333;cursor:pointer;width:100%;';
    [['Very slow', 1000], ['Normal', 350], ['Fast', 100]].forEach(([label, val]) => {
        const opt = document.createElement('option');
        opt.value       = val;
        opt.textContent = label;
        if (val === 350) opt.selected = true;
        speedSel.appendChild(opt);
    });

    const stepsEl  = document.createElement('span');
    stepsEl.style.cssText = 'font-size:12px;color:#888;margin-top:4px;';
    stepsEl.textContent   = 'Step: 0';

    const infEl = document.createElement('span');
    infEl.style.cssText = 'font-size:12px;color:#e74c3c;font-weight:600;';
    infEl.textContent   = 'Infected: 1';

    const hlthEl = document.createElement('span');
    hlthEl.style.cssText = 'font-size:12px;color:#3498db;font-weight:600;';
    hlthEl.textContent   = 'Healthy: 0';

    panel.append(btnStart, btnReset, speedLabel, speedSel, stepsEl, infEl, hlthEl);
    row.append(canvas, panel);
    wrapper.append(row);
    mount.appendChild(wrapper);

    /* ── canvas context (HiDPI) ── */
    const dpr = Math.max(1, window.devicePixelRatio || 1);
    canvas.width  = W * dpr;
    canvas.height = H * dpr;
    canvas.style.width  = W + 'px';
    canvas.style.height = H + 'px';
    const ctx = canvas.getContext('2d');
    ctx.scale(dpr, dpr);

    /* ── coordinate helpers ── */
    const lx = x => ox + x * CELL;
    const ly = y => oy - y * CELL;
    const pk = (x, y) => x + ',' + y;

    /* ── drawing ── */
    function drawGrid() {
        ctx.clearRect(0, 0, W, H);
        ctx.strokeStyle = '#e0e0d8';
        ctx.lineWidth   = 0.5;
        const lo = -R * CELL, hi = R * CELL;
        for (let i = -R; i <= R; i++) {
            const px = ox + i * CELL;
            ctx.beginPath(); ctx.moveTo(px, oy + lo); ctx.lineTo(px, oy + hi); ctx.stroke();
            const py = oy + i * CELL;
            ctx.beginPath(); ctx.moveTo(ox + lo, py); ctx.lineTo(ox + hi, py); ctx.stroke();
        }
        ctx.fillStyle = 'rgba(231,76,60,0.18)';
        ctx.beginPath(); ctx.arc(ox, oy, 5, 0, 2 * Math.PI); ctx.fill();
    }

    function drawFrog(px, py, infected) {
        const col  = infected ? INFECT_COL  : SLEEP_COL;
        const glow = infected ? INFECT_GLOW : SLEEP_GLOW;
        ctx.fillStyle = glow;
        ctx.beginPath(); ctx.arc(px, py, FROG_R + 2, 0, 2 * Math.PI); ctx.fill();
        ctx.fillStyle = col;
        ctx.beginPath(); ctx.arc(px, py, FROG_R, 0, 2 * Math.PI); ctx.fill();
        // tiny eyes
        ctx.fillStyle = '#fff';
        const er = FROG_R * 0.22;
        ctx.beginPath(); ctx.arc(px - FROG_R * 0.32, py - FROG_R * 0.30, er, 0, 2*Math.PI); ctx.fill();
        ctx.beginPath(); ctx.arc(px + FROG_R * 0.32, py - FROG_R * 0.30, er, 0, 2*Math.PI); ctx.fill();
        ctx.fillStyle = '#222';
        const pr = er * 0.55;
        ctx.beginPath(); ctx.arc(px - FROG_R * 0.32, py - FROG_R * 0.28, pr, 0, 2*Math.PI); ctx.fill();
        ctx.beginPath(); ctx.arc(px + FROG_R * 0.32, py - FROG_R * 0.28, pr, 0, 2*Math.PI); ctx.fill();
    }

    function render() {
        drawGrid();
        stepsEl.textContent  = 'Step: '     + stepCount;
        infEl.textContent    = 'Infected: ' + infected.length;
        hlthEl.textContent   = 'Healthy: '  + healthy.size;

        healthy.forEach(k => {
            const [a, b] = k.split(',');
            drawFrog(lx(+a), ly(+b), false);
        });
        for (const f of infected) {
            if (Math.abs(f.x) <= R + 1 && Math.abs(f.y) <= R + 1)
                drawFrog(lx(f.x), ly(f.y), true);
        }
    }

    /* ── simulation state ── */
    let healthy;
    let infected;
    let running   = false;
    let animId    = null;
    let stepCount = 0;

    function init() {
        healthy   = new Set();
        infected  = [];
        stepCount = 0;
        running   = false;
        if (animId) { cancelAnimationFrame(animId); animId = null; }

        for (let x = -R; x <= R; x++)
            for (let y = -R; y <= R; y++)
                if (x !== 0 || y !== 0) healthy.add(pk(x, y));

        infected.push({ x: 0, y: 0 });
    }

    /* ── one simulation step ── */
    function simStep() {
        const born = [];
        for (const f of infected) {
            const d = dirs[Math.random() * 4 | 0];
            f.x += d[0];
            f.y += d[1];
            const k = pk(f.x, f.y);
            if (healthy.has(k)) {
                healthy.delete(k);
                born.push({ x: f.x, y: f.y });
            }
        }
        for (const b of born) infected.push(b);
        stepCount++;
    }

    /* ── animation loop ── */
    let last = 0;
    function getDelay() { return Number(speedSel.value) || 350; }

    function tick(ts) {
        if (!running) return;
        if (ts - last >= getDelay()) {
            last = ts;
            simStep();
            render();
        }
        animId = requestAnimationFrame(tick);
    }

    /* ── controls ── */
    btnStart.addEventListener('click', () => {
        if (running) {
            running = false;
            cancelAnimationFrame(animId);
            btnStart.textContent      = '▶ Start';
            btnStart.style.background = '#27ae60';
        } else {
            running = true;
            last    = 0;
            btnStart.textContent      = '⏸ Pause';
            btnStart.style.background = '#f39c12';
            animId = requestAnimationFrame(tick);
        }
    });

    btnReset.addEventListener('click', () => {
        init();
        btnStart.textContent      = '▶ Start';
        btnStart.style.background = '#27ae60';
        render();
    });

    /* ── initial draw ── */
    init();
    render();
})();
