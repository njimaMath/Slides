/**
 * ks_infection.js — KS Infection Model simulation animation
 *
 * Rules (contrast with the Frog Model):
 *   - ALL frogs (healthy and infected alike) perform independent
 *     simple random walks inside a reflecting box.
 *   - When an infected frog occupies the same vertex as a healthy
 *     frog, the healthy frog instantly becomes infected.
 *
 * Usage:
 *   <div id="ks-figure"></div>
 *   <script src="ks_infection.js"></script>
 *
 * If #ks-figure is absent the figure is appended to document.body.
 */
(function () {
    /* ── grid / display constants ── */
    const CELL  = 22;    // px per lattice cell
    const BOX_R = 7;     // reflecting box radius: −BOX_R … +BOX_R
    const W     = 340;
    const H     = 340;
    const ox    = W / 2;
    const oy    = H / 2;
    const dirs  = [[1,0],[-1,0],[0,1],[0,-1]];

    /* ── colours ── */
    const SLEEP_COL   = '#3498db';
    const INFECT_COL  = '#e74c3c';
    const SLEEP_GLOW  = 'rgba(52,152,219,0.22)';
    const INFECT_GLOW = 'rgba(231,76,60,0.25)';
    const FROG_R      = CELL * 0.42;

    /* ── DOM setup ── */
    const mount = document.getElementById('ks-figure') || document.body;

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

    /* ── drawing ── */
    function drawGrid() {
        ctx.clearRect(0, 0, W, H);
        ctx.strokeStyle = '#e0e0d8';
        ctx.lineWidth   = 0.5;
        const lo = -BOX_R * CELL, hi = BOX_R * CELL;
        for (let i = -BOX_R; i <= BOX_R; i++) {
            const px = ox + i * CELL;
            ctx.beginPath(); ctx.moveTo(px, oy + lo); ctx.lineTo(px, oy + hi); ctx.stroke();
            const py = oy + i * CELL;
            ctx.beginPath(); ctx.moveTo(ox + lo, py); ctx.lineTo(ox + hi, py); ctx.stroke();
        }
        // reflecting boundary
        ctx.strokeStyle = '#888';
        ctx.lineWidth   = 1.5;
        ctx.strokeRect(ox + lo, oy + lo, hi - lo, hi - lo);
        // origin marker
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
        let infectedCount = 0;
        for (const f of frogs) {
            if (Math.abs(f.x) > BOX_R + 1 || Math.abs(f.y) > BOX_R + 1) continue;
            drawFrog(lx(f.x), ly(f.y), f.infected);
            if (f.infected) infectedCount++;
        }
        const healthyCount = frogs.length - infectedCount;
        stepsEl.textContent  = 'Step: '     + stepCount;
        infEl.textContent    = 'Infected: ' + infectedCount;
        hlthEl.textContent   = 'Healthy: '  + healthyCount;
    }

    /* ── simulation state ── */
    let frogs;
    let running   = false;
    let animId    = null;
    let stepCount = 0;

    function init() {
        frogs     = [];
        stepCount = 0;
        running   = false;
        if (animId) { cancelAnimationFrame(animId); animId = null; }

        for (let x = -BOX_R; x <= BOX_R; x++)
            for (let y = -BOX_R; y <= BOX_R; y++)
                frogs.push({ x, y, infected: (x === 0 && y === 0) });
    }

    /* ── reflection helper ── */
    function reflect(v, bound) {
        if (v >  bound) return 2 * bound - v;
        if (v < -bound) return -2 * bound - v;
        return v;
    }

    function infectAllAt(x, y) {
        for (const f of frogs)
            if (!f.infected && f.x === x && f.y === y) f.infected = true;
    }

    function shuffle(arr) {
        for (let i = arr.length - 1; i > 0; i--) {
            const j = Math.random() * (i + 1) | 0;
            [arr[i], arr[j]] = [arr[j], arr[i]];
        }
        return arr;
    }

    /* ── one simulation step (asynchronous update) ── */
    function simStep() {
        const order = shuffle(frogs.slice());
        for (const f of order) {
            const d = dirs[Math.random() * 4 | 0];
            f.x = reflect(f.x + d[0], BOX_R);
            f.y = reflect(f.y + d[1], BOX_R);

            if (f.infected) {
                infectAllAt(f.x, f.y);
            } else {
                for (const g of frogs) {
                    if (g !== f && g.infected && g.x === f.x && g.y === f.y) {
                        f.infected = true;
                        infectAllAt(f.x, f.y);
                        break;
                    }
                }
            }
        }
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
