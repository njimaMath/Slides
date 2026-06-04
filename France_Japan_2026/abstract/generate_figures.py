#!/usr/bin/env python3
"""
generate_figures.py
Creates 4 PNG images (600×600 px each) for the abstract folder:
  cell_colony.png  —  Eden model (cell colony growth)
  paper_burning.png — stochastic fire spread on paper
  frog.png          — Frog Model
  ks_infection.png  — KS Infection Model
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

RNG  = np.random.default_rng(2026)
DIR  = os.path.dirname(os.path.abspath(__file__))
PX   = 600        # output pixel size (width = height)
DPI  = 100        # → figure size = PX/DPI = 6.0 inches
INCH = PX / DPI


def save(fig, name):
    """Save figure at exactly PX×PX pixels (no bbox_inches='tight')."""
    path = os.path.join(DIR, name)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print(f"  ✓  {name}  ({PX}×{PX} px)")


# ═══════════════════════════════════════════════════════════════════
# 1. Cell Colony — Eden Model
# ═══════════════════════════════════════════════════════════════════
def fig_cell_colony():
    G  = 200            # grid side (cells)
    cx = cy = G // 2
    R  = G // 2 - 4     # petri-dish radius (cells)
    DIRS = [(1,0),(-1,0),(0,1),(0,-1)]

    occ  = np.zeros((G, G), dtype=bool)
    in_f = np.zeros((G, G), dtype=bool)
    front = []

    def add(x, y):
        occ[y, x] = True
        for dx, dy in DIRS:
            nx, ny = x+dx, y+dy
            if 0 <= nx < G and 0 <= ny < G and not occ[ny,nx] and not in_f[ny,nx]:
                if (nx-cx)**2 + (ny-cy)**2 <= R*R:
                    front.append((nx, ny))
                    in_f[ny, nx] = True

    for dy in range(-2, 3):
        for dx in range(-2, 3):
            add(cx+dx, cy+dy)

    target = int(0.58 * np.pi * R * R)
    while front and int(occ.sum()) < target:
        ri = int(RNG.integers(0, len(front)))
        fx, fy = front[ri]
        front[ri] = front[-1]; front.pop()
        in_f[fy, fx] = False
        if not occ[fy, fx]:
            add(fx, fy)

    # ── vectorised colour image ───────────────────────────────────
    ys, xs = np.mgrid[0:G, 0:G]
    dist = np.hypot(xs - cx, ys - cy)
    t    = np.clip(dist / R, 0, 1)
    noise = RNG.uniform(-0.04, 0.04, (G, G))

    img  = np.zeros((G, G, 3))
    out  = dist > R + 1
    agar = (~out) & (~occ)

    # outside dish — very dark
    for c in range(3):
        img[:,:,c] = np.where(out, 0.10, img[:,:,c])

    # agar (warm brown radial gradient)
    ac = [0.228, 0.149, 0.110]
    ae = [0.282, 0.196, 0.149]
    for c in range(3):
        img[:,:,c] = np.where(agar,
            np.clip(ac[c] + (ae[c]-ac[c])*t + noise*0.7, 0, 1), img[:,:,c])

    # colony cells (cream, slightly darker near edge)
    edge_d = np.clip(1 - 0.12*(dist/R), 0, 1)
    nc = RNG.uniform(-0.04, 0.04, (G, G))
    bc = [0.921, 0.882, 0.784]
    for c in range(3):
        img[:,:,c] = np.where(occ,
            np.clip((bc[c] + nc*1.5) * edge_d, 0, 1), img[:,:,c])

    fig, ax = plt.subplots(figsize=(INCH, INCH), dpi=DPI, facecolor='#111111')
    ax.set_position([0, 0, 1, 1])
    ax.set_facecolor('#111111')
    ax.imshow(img, origin='upper', interpolation='nearest')
    ax.add_patch(plt.Circle((cx, cy), R,
                             color='white', fill=False, lw=1.0, alpha=0.30))
    ax.set_xlim(0, G); ax.set_ylim(G, 0)
    ax.set_aspect('equal'); ax.axis('off')
    return fig


# ═══════════════════════════════════════════════════════════════════
# 2. Paper Burning  (realistic version)
# ═══════════════════════════════════════════════════════════════════
def fig_paper_burning():
    from scipy.ndimage import gaussian_filter

    # ── simulation grid ──────────────────────────────────────────
    COLS, ROWS = 300, 300

    # Each cell counts down:  TOTAL → 1 → ASH_ST
    #   TOTAL .. EMBER_T+1  = actively burning (fire)
    #   EMBER_T .. 1        = glowing ember (dying)
    #   ASH_ST              = cooled ash
    BURN_T  = 22   # steps cell is on fire
    EMBER_T = 12   # steps cell glows as ember
    TOTAL   = BURN_T + EMBER_T
    ASH_ST  = TOTAL + 1

    # Spread probabilities (fire phase only)
    SP_UP   = 0.34
    SP_SIDE = 0.17
    SP_DOWN = 0.07
    SP_DIAG_U = 0.18  # upward diagonals

    state = np.zeros((ROWS, COLS), dtype=np.int16)
    in_f  = np.zeros((ROWS, COLS), dtype=bool)
    front = []

    def ignite(x, y):
        if 0 <= x < COLS and 0 <= y < ROWS and state[y, x] == 0 and not in_f[y, x]:
            state[y, x] = TOTAL
            in_f[y, x]  = True
            front.append((x, y))

    cx = COLS // 2
    for dx in range(-7, 8):
        ignite(cx + dx, ROWS - 1)

    def sim_step():
        nxt, born = [], []
        for (bx, by) in front:
            s = state[by, bx]
            if s == 1:
                state[by, bx] = ASH_ST
                in_f[by, bx]  = False
            else:
                state[by, bx] = s - 1
                nxt.append((bx, by))
                # spread only during fire phase
                if s > EMBER_T:
                    for dxs, dys, p in [
                        ( 0, -1, SP_UP),
                        (-1,  0, SP_SIDE), ( 1,  0, SP_SIDE),
                        ( 0,  1, SP_DOWN),
                        (-1, -1, SP_DIAG_U), ( 1, -1, SP_DIAG_U),
                        (-1,  1, SP_DOWN * 0.5), ( 1,  1, SP_DOWN * 0.5),
                    ]:
                        nx, ny = bx + dxs, by + dys
                        if 0 <= nx < COLS and 0 <= ny < ROWS \
                                and state[ny, nx] == 0 and not in_f[ny, nx]:
                            if RNG.random() < p:
                                state[ny, nx] = TOTAL
                                in_f[ny, nx]  = True
                                born.append((nx, ny))
        front.clear(); front.extend(nxt); front.extend(born)

    ash_target = int(0.62 * COLS * ROWS)
    for _ in range(6000):
        sim_step()
        if not front or int(np.sum(state == ASH_ST)) >= ash_target:
            break

    # ── masks & normalised timers ─────────────────────────────────
    is_ash   = (state == ASH_ST)
    is_fire  = (state > EMBER_T) & (~is_ash)
    is_ember = (state > 0) & (state <= EMBER_T)
    is_paper = (state == 0)

    fire_t  = np.where(is_fire,  np.clip((state - EMBER_T) / BURN_T,  0, 1), 0.0)
    ember_t = np.where(is_ember, np.clip(state / EMBER_T,              0, 1), 0.0)

    # ── 1. Paper texture ─────────────────────────────────────────
    # multi-scale grain: coarse fibre + medium grain + fine speckle
    g_coarse = gaussian_filter(RNG.standard_normal((ROWS, COLS)), sigma=5.0) * 0.045
    g_mid    = gaussian_filter(RNG.standard_normal((ROWS, COLS)), sigma=1.5) * 0.020
    g_fine   = RNG.standard_normal((ROWS, COLS)) * 0.008
    grain    = g_coarse + g_mid + g_fine

    # faint horizontal fibre streaks (paper texture)
    streak = gaussian_filter(RNG.standard_normal((ROWS, COLS)), sigma=[0.4, 12]) * 0.018

    # ruled lines (light blue-grey)
    ruled = np.zeros((ROWS, COLS, 3))
    for ry in range(10, ROWS, 14):
        ruled[ry,   :] = [-0.012, -0.005,  0.028]
    # left margin line (red)
    mx = int(COLS * 0.13)
    ruled[:, mx,   0] += 0.12
    ruled[:, mx,   1] -= 0.03
    ruled[:, mx,   2] -= 0.03
    ruled[:, mx+1, 0] += 0.04

    pb = np.array([0.938, 0.922, 0.888])   # warm white paper base

    img = np.zeros((ROWS, COLS, 3))
    tex = grain + streak
    for c in range(3):
        img[:,:,c] = np.where(is_paper,
            np.clip(pb[c] + tex + ruled[:,:,c], 0, 1), 0.0)

    # ── 2. Heat-scorch gradient (paper browning near fire) ────────
    heat_map = gaussian_filter((state > 0).astype(np.float32), sigma=9.0)
    heat_map = np.clip(heat_map * 3.5, 0, 1)

    # scorch palette: warm white → golden → amber → dark brown
    sc = np.array([0.55, 0.28, 0.04])   # fully scorched colour
    for c in range(3):
        scorch_val = pb[c] * (1 - heat_map * 0.85) + sc[c] * heat_map * 0.85
        img[:,:,c] = np.where(is_paper,
            np.clip(scorch_val + tex + ruled[:,:,c], 0, 1), img[:,:,c])

    # ── 3. Ash (dark charcoal with crinkle texture) ───────────────
    a_crinkle = gaussian_filter(RNG.standard_normal((ROWS, COLS)), sigma=2.5) * 0.040
    a_fine    = RNG.standard_normal((ROWS, COLS)) * 0.012
    ab = np.array([0.088, 0.058, 0.038])
    for c in range(3):
        img[:,:,c] = np.where(is_ash,
            np.clip(ab[c] + a_crinkle + a_fine, 0, 1), img[:,:,c])

    # ── 4. Ember glow (deep orange-red, fading) ──────────────────
    e_hot  = np.array([0.95, 0.28, 0.02])
    e_cold = np.array([0.30, 0.04, 0.00])
    for c in range(3):
        ec = e_hot[c] * ember_t + e_cold[c] * (1 - ember_t)
        img[:,:,c] = np.where(is_ember, np.clip(ec, 0, 1), img[:,:,c])

    # ── 5. Active fire (white-yellow core → orange → deep red) ───
    fn = RNG.standard_normal((ROWS, COLS)) * 0.035
    r_ = np.ones_like(fire_t)
    g_ = np.where(fire_t > 0.80, 0.88 + 0.12*(fire_t-0.80)/0.20,
         np.where(fire_t > 0.58, 0.32 + 0.56*(fire_t-0.58)/0.22,
         np.where(fire_t > 0.35, 0.06 + 0.26*(fire_t-0.35)/0.23,
                                  0.00 + 0.06*(fire_t / 0.35))))
    b_ = np.where(fire_t > 0.82, 0.72*(fire_t-0.82)/0.18, 0.0)

    img[:,:,0] = np.where(is_fire, np.clip(r_ + fn*0.25, 0, 1), img[:,:,0])
    img[:,:,1] = np.where(is_fire, np.clip(g_ + fn*0.50, 0, 1), img[:,:,1])
    img[:,:,2] = np.where(is_fire, np.clip(b_,           0, 1), img[:,:,2])

    # ── 6. Bloom / light-scatter pass ────────────────────────────
    # Wide soft glow from fire illuminates surrounding paper/ash
    fire_lum = np.zeros((ROWS, COLS, 3))
    fire_lum[:,:,0] = np.where(is_fire, r_ * fire_t, 0)
    fire_lum[:,:,1] = np.where(is_fire, g_ * fire_t, 0)
    fire_lum[:,:,2] = np.where(is_fire, b_ * fire_t, 0)
    for c in range(3):
        wide_bloom  = gaussian_filter(fire_lum[:,:,c], sigma=12.0) * 0.55
        tight_bloom = gaussian_filter(fire_lum[:,:,c], sigma=3.5)  * 0.70
        img[:,:,c]  = np.clip(img[:,:,c] + wide_bloom + tight_bloom, 0, 1)

    # Ember bloom (warm red glow over ash)
    e_lum = np.where(is_ember, ember_t, 0.0).astype(np.float32)
    eb    = gaussian_filter(e_lum, sigma=5.0) * 0.55
    img[:,:,0] = np.clip(img[:,:,0] + eb * 0.92, 0, 1)
    img[:,:,1] = np.clip(img[:,:,1] + eb * 0.22, 0, 1)

    # Bright hot-spot core (tight white-yellow highlight)
    core_mask = np.where(is_fire & (fire_t > 0.78), fire_t, 0.0).astype(np.float32)
    core_b    = gaussian_filter(core_mask, sigma=1.8) * 0.90
    for c, w in enumerate([1.0, 0.88, 0.45]):
        img[:,:,c] = np.clip(img[:,:,c] + core_b * w, 0, 1)

    # ── 7. Final tone-map: dark background, slight vignette ──────
    ys, xs = np.mgrid[0:ROWS, 0:COLS]
    cx2, cy2 = COLS / 2, ROWS / 2
    vignette = 1.0 - 0.18 * np.clip(
        np.hypot((xs - cx2) / cx2, (ys - cy2) / cy2), 0, 1)
    for c in range(3):
        img[:,:,c] *= vignette

    fig, ax = plt.subplots(figsize=(INCH, INCH), dpi=DPI, facecolor='#080808')
    ax.set_position([0, 0, 1, 1])
    ax.set_facecolor('#080808')
    ax.imshow(img, origin='upper', interpolation='bilinear')
    ax.axis('off')
    return fig


# ═══════════════════════════════════════════════════════════════════
# 3. Frog Model
# ═══════════════════════════════════════════════════════════════════
def fig_frog():
    R     = 7
    STEPS = 55
    DIRS  = [(1,0),(-1,0),(0,1),(0,-1)]

    healthy  = {(x,y) for x in range(-R,R+1) for y in range(-R,R+1)
                if (x, y) != (0, 0)}
    infected = [{'x': 0, 'y': 0}]

    for _ in range(STEPS):
        born = []
        for f in infected:
            d = DIRS[int(RNG.integers(0, 4))]
            f['x'] += d[0]; f['y'] += d[1]
            k = (f['x'], f['y'])
            if k in healthy:
                healthy.discard(k)
                born.append({'x': f['x'], 'y': f['y']})
        infected.extend(born)

    fig, ax = plt.subplots(figsize=(INCH, INCH), dpi=DPI, facecolor='#f5f5f0')
    ax.set_position([0.04, 0.04, 0.92, 0.92])
    ax.set_facecolor('#f5f5f0')

    for i in range(-R, R+1):
        ax.axhline(i, color='#dedede', lw=0.6, zorder=0)
        ax.axvline(i, color='#dedede', lw=0.6, zorder=0)

    if healthy:
        hx, hy = zip(*healthy)
        ax.scatter(hx, hy, s=120, c='#3498db', alpha=0.82, zorder=3,
                   linewidths=0.5, edgecolors='#1f78b4')

    ix = [f['x'] for f in infected]
    iy = [f['y'] for f in infected]
    ax.scatter(ix, iy, s=140, c='#e74c3c', alpha=0.92, zorder=4,
               linewidths=0.5, edgecolors='#c0392b')

    # origin star
    ax.scatter([0], [0], s=280, c='#e74c3c', marker='*', zorder=5, linewidths=0)

    lim = R + 0.8
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_aspect('equal'); ax.axis('off')
    return fig


# ═══════════════════════════════════════════════════════════════════
# 4. KS Infection
# ═══════════════════════════════════════════════════════════════════
def fig_ks_infection():
    BOX_R = 7
    STEPS = 70
    DIRS  = [(1,0),(-1,0),(0,1),(0,-1)]

    frogs = [{'x': x, 'y': y, 'inf': (x == 0 and y == 0)}
             for x in range(-BOX_R, BOX_R+1)
             for y in range(-BOX_R, BOX_R+1)]

    def ref(v, b):
        if v >  b: return 2*b - v
        if v < -b: return -2*b - v
        return v

    def infect_all(x, y):
        for f in frogs:
            if not f['inf'] and f['x'] == x and f['y'] == y:
                f['inf'] = True

    for _ in range(STEPS):
        for i in RNG.permutation(len(frogs)):
            f = frogs[i]
            d = DIRS[int(RNG.integers(0, 4))]
            f['x'] = ref(f['x'] + d[0], BOX_R)
            f['y'] = ref(f['y'] + d[1], BOX_R)
            if f['inf']:
                infect_all(f['x'], f['y'])
            else:
                for g in frogs:
                    if g is not f and g['inf'] and g['x'] == f['x'] and g['y'] == f['y']:
                        f['inf'] = True; break

    fig, ax = plt.subplots(figsize=(INCH, INCH), dpi=DPI, facecolor='#f5f5f0')
    ax.set_position([0.04, 0.04, 0.92, 0.92])
    ax.set_facecolor('#f5f5f0')

    for i in range(-BOX_R, BOX_R+1):
        ax.axhline(i, color='#dedede', lw=0.6, zorder=0)
        ax.axvline(i, color='#dedede', lw=0.6, zorder=0)

    # reflecting boundary box
    ax.add_patch(mpatches.FancyBboxPatch(
        (-BOX_R-0.5, -BOX_R-0.5), 2*BOX_R+1, 2*BOX_R+1,
        boxstyle='square,pad=0', fill=False,
        edgecolor='#777777', lw=2.0, zorder=2))

    healthy  = [(f['x'], f['y']) for f in frogs if not f['inf']]
    infected = [(f['x'], f['y']) for f in frogs if f['inf']]

    if healthy:
        hx, hy = zip(*healthy)
        ax.scatter(hx, hy, s=120, c='#3498db', alpha=0.82, zorder=3,
                   linewidths=0.5, edgecolors='#1f78b4')
    if infected:
        ix, iy = zip(*infected)
        ax.scatter(ix, iy, s=140, c='#e74c3c', alpha=0.92, zorder=4,
                   linewidths=0.5, edgecolors='#c0392b')

    ax.scatter([0], [0], s=280, c='#e74c3c', marker='*', zorder=5, linewidths=0)

    lim = BOX_R + 1.0
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_aspect('equal'); ax.axis('off')
    return fig


# ═══════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print(f'Generating 4 × {PX}×{PX} px PNG figures in:\n  {DIR}\n')
    save(fig_cell_colony(),    'cell_colony.png')
    save(fig_paper_burning(),  'paper_burning.png')
    save(fig_frog(),           'frog.png')
    save(fig_ks_infection(),   'ks_infection.png')
    print('\nDone.')
