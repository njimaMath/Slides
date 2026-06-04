"""
Frog Model on Z — KPZ Fluctuation Exponent  (C-accelerated)
════════════════════════════════════════════════════════════
Confirms numerically:   Std(T_n)  ~  n^{1/3}

Uses a compiled C shared library for the inner simulation loop.
One trial records T_x for ALL x = 1 … max_n simultaneously.

Key insight: visited sites on Z always form a contiguous interval
[L, R], so frontier detection is O(1) per frog per step.

Output: log-log plot  →  frog_fluctuation_exponent.png
"""

import ctypes, subprocess, sys, time, os
import numpy as np
import matplotlib.pyplot as plt

# ══════════════════════════════════════════════════════════════
#  Parameters  (adjust as needed)
# ══════════════════════════════════════════════════════════════
MAX_N    = 10_000       # record T_1 … T_{MAX_N} per trial
N_TRIALS = 10           # independent replications
N_POINTS = 50           # evaluation points on the n-axis (geom-spaced)

# ══════════════════════════════════════════════════════════════
#  Compile C code
# ══════════════════════════════════════════════════════════════
HERE = os.path.dirname(os.path.abspath(__file__))
SRC  = os.path.join(HERE, 'frog_sim.c')
LIB  = os.path.join(HERE, 'frog_sim.so')

print("Compiling C simulation …")
ret = subprocess.run(
    ['gcc', '-O3', '-march=native', '-shared', '-fPIC', '-o', LIB, SRC],
    capture_output=True, text=True
)
if ret.returncode != 0:
    print("Compilation failed:\n", ret.stderr)
    sys.exit(1)
print(f"  → {LIB}")

lib = ctypes.CDLL(LIB)
lib.simulate_frog.argtypes = [
    ctypes.c_int,                          # max_n
    ctypes.POINTER(ctypes.c_int),          # T[0..max_n]
    ctypes.c_uint64,                       # seed
]
lib.simulate_frog.restype = None

# ══════════════════════════════════════════════════════════════
#  Run simulations
# ══════════════════════════════════════════════════════════════

def run_one(max_n: int, seed: int) -> np.ndarray:
    """Run one trial, return T[1..max_n] as int32 array."""
    T = np.zeros(max_n + 1, dtype=np.int32)
    ptr = T.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    lib.simulate_frog(max_n, ptr, seed)
    return T[1:]                           # T[1], T[2], …, T[max_n]


print(f"\nFrog Model on Z — max_n = {MAX_N:,}, N = {N_TRIALS} trials")
print("=" * 60)

# Time one trial to give the user an ETA
t0 = time.perf_counter()
first = run_one(MAX_N, 0)
dt1 = time.perf_counter() - t0
est = dt1 * N_TRIALS
print(f"  One trial: {dt1:.2f} s   →   est. total ≈ {est:.0f} s  ({est/60:.1f} min)\n")

# Collect all trials
all_T = np.empty((N_TRIALS, MAX_N), dtype=np.int32)
all_T[0] = first

t_start = time.perf_counter()
for j in range(1, N_TRIALS):
    all_T[j] = run_one(MAX_N, seed=j * 999_999_937 + 1)
    if (j + 1) % 5 == 0 or j == N_TRIALS - 1:
        elapsed = time.perf_counter() - t_start
        eta = elapsed / (j + 1) * (N_TRIALS - j - 1)
        print(f"\r  Trial {j+1:4d}/{N_TRIALS}   "
              f"elapsed {elapsed:.0f}s   ETA {eta:.0f}s", end="", flush=True)
print("\n")

# ══════════════════════════════════════════════════════════════
#  Compute statistics
# ══════════════════════════════════════════════════════════════

n_vals = np.unique(np.geomspace(5, MAX_N, N_POINTS).astype(int))
ns     = n_vals.astype(float)

means = np.array([all_T[:, n - 1].mean()         for n in n_vals])
stds  = np.array([all_T[:, n - 1].std(ddof=1)    for n in n_vals])

# Log-log regression: log Std = α log n + const
log_n   = np.log(ns)
log_std = np.log(stds)

half = len(ns) // 2
alpha_all, c_all = np.polyfit(log_n,        log_std,        1)
alpha_hi,  c_hi  = np.polyfit(log_n[half:], log_std[half:], 1)

print(f"Fitted exponent  α  (Std ~ n^α):")
print(f"  all n:          α = {alpha_all:.4f}")
print(f"  large n only:   α = {alpha_hi:.4f}")
print(f"  expected (KPZ): 1/3 ≈ 0.3333")

# ══════════════════════════════════════════════════════════════
#  Log-log plot
# ══════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(9, 6.5))

# Data
ax.loglog(ns, stds, 'o', color='#e74c3c', ms=4.5, zorder=5,
          label=r'$\mathrm{Std}(T_n)$  (simulation)')

# Reference lines anchored at the midpoint
mid    = len(ns) // 2
n_fine = np.logspace(np.log10(ns[0]), np.log10(ns[-1]), 300)

C13 = stds[mid] / ns[mid] ** (1 / 3)
C12 = stds[mid] / ns[mid] ** (1 / 2)
C_fit = np.exp(c_all)

ax.loglog(n_fine, C13 * n_fine ** (1/3), '-',  color='#2ecc71', lw=2.2, alpha=0.85,
          label=r'$C \, n^{1/3}$  (KPZ)')
ax.loglog(n_fine, C12 * n_fine ** (1/2), ':',  color='#95a5a6', lw=2,   alpha=0.7,
          label=r'$C \, n^{1/2}$  (CLT)')
ax.loglog(n_fine, C_fit * n_fine ** alpha_all, '--', color='#3498db', lw=1.5, alpha=0.7,
          label=rf'Fit: $n^{{{alpha_all:.3f}}}$')

ax.set_xlabel(r'$n$', fontsize=15)
ax.set_ylabel(r'$\mathrm{Std}(T_n)$', fontsize=15)
ax.set_title(
    rf'Frog Model on $\mathbb{{Z}}$:  fluctuation of $T_n$'
    '\n'
    rf'$N = {N_TRIALS}$ trials,  $n$ up to ${MAX_N:,}$',
    fontsize=14)

ax.legend(fontsize=11, loc='upper left')
ax.grid(True, which='both', ls=':', alpha=0.35)
ax.tick_params(labelsize=12)

# Annotation box
ax.text(0.97, 0.08,
        rf'$\alpha_{{\mathrm{{all}}}} = {alpha_all:.3f}$' + '\n'
        rf'$\alpha_{{\mathrm{{large\;n}}}} = {alpha_hi:.3f}$' + '\n'
        rf'KPZ prediction $= 1/3 \approx 0.333$',
        transform=ax.transAxes, fontsize=11,
        ha='right', va='bottom',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='wheat', alpha=0.6))

fig.tight_layout()
out = os.path.join(HERE, 'frog_fluctuation_exponent.png')
fig.savefig(out, dpi=200, bbox_inches='tight')
print(f"\nFigure saved → {out}")
plt.show()
