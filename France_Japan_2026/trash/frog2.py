import numpy as np
import matplotlib.pyplot as plt
import os
from concurrent.futures import ProcessPoolExecutor

# Steps for simple random walk on Z^2
DX = np.array([1, -1, 0, 0], dtype=np.int32)
DY = np.array([0, 0, 1, -1], dtype=np.int32)
_MASK32 = (1 << 32) - 1


def _pack_xy(xi, yi):
    """Pack signed 32-bit coordinates into one Python int key."""
    return ((xi & _MASK32) << 32) | (yi & _MASK32)


def simulate_one_run(n_values, rng, max_steps=50_000):
    """
    One simulation of the frog model on Z^2.
    Returns T_n for each n in n_values (same order), where T_n is activation
    time of site (n,0).
    """
    n_values = np.asarray(n_values, dtype=np.int32)
    target_idx = {int(n): j for j, n in enumerate(n_values)}
    unresolved = set(int(n) for n in n_values)
    T = np.full(len(n_values), -1, dtype=np.int32)

    # n=0 edge case
    if 0 in target_idx:
        T[target_idx[0]] = 0
        unresolved.discard(0)

    # Dynamic arrays for active frog positions
    cap = 1024
    x = np.empty(cap, dtype=np.int32)
    y = np.empty(cap, dtype=np.int32)
    x[0], y[0] = 0, 0
    m = 1  # number of active frogs

    # Sites already awakened (packed int key instead of tuple for speed)
    activated = {_pack_xy(0, 0)}

    for t in range(1, max_steps + 1):
        # Move all active frogs
        steps = rng.integers(0, 4, size=m)
        x[:m] += DX[steps]
        y[:m] += DY[steps]

        # Preallocate worst-case buffers (all frogs discover new sites)
        new_x = np.empty(m, dtype=np.int32)
        new_y = np.empty(m, dtype=np.int32)
        k = 0

        # First-visit check: awaken sleeping frog if site not activated before
        x_view = x
        y_view = y
        for i in range(m):
            xi = int(x_view[i])
            yi = int(y_view[i])
            key = _pack_xy(xi, yi)

            if key not in activated:
                activated.add(key)
                new_x[k] = xi
                new_y[k] = yi
                k += 1

                if yi == 0 and xi in unresolved:
                    T[target_idx[xi]] = t
                    unresolved.remove(xi)

        # Add newly awakened frogs to active list
        if k > 0:
            need = m + k
            if need > cap:
                while cap < need:
                    cap *= 2
                x2 = np.empty(cap, dtype=np.int32)
                y2 = np.empty(cap, dtype=np.int32)
                x2[:m] = x[:m]
                y2[:m] = y[:m]
                x, y = x2, y2

            x[m:m + k] = new_x[:k]
            y[m:m + k] = new_y[:k]
            m += k

        if not unresolved:
            return T

    raise RuntimeError(
        f"max_steps={max_steps} reached before all targets activated. "
        f"Try larger max_steps or smaller n_values."
    )


def _simulate_one_run_from_seed(seed, n_values, max_steps):
    rng = np.random.default_rng(seed)
    return simulate_one_run(n_values, rng, max_steps=max_steps)


def sample_T_matrix(
    n_values,
    runs,
    seed=0,
    max_steps=50_000,
    verbose=True,
    workers=1,
):
    out = np.empty((runs, len(n_values)), dtype=np.int32)

    if workers is None or workers <= 1:
        rng = np.random.default_rng(seed)
        for r in range(runs):
            out[r] = simulate_one_run(n_values, rng, max_steps=max_steps)
            if verbose and (r + 1) % 10 == 0:
                print(f"  finished {r+1}/{runs} runs")
        return out

    # Independent per-run RNG seeds for parallel execution
    ss = np.random.SeedSequence(seed)
    child_seeds = [
        int(s.generate_state(1, dtype=np.uint64)[0]) for s in ss.spawn(runs)
    ]

    with ProcessPoolExecutor(max_workers=workers) as ex:
        iterator = ex.map(
            _simulate_one_run_from_seed,
            child_seeds,
            [n_values] * runs,
            [max_steps] * runs,
            chunksize=1,
        )
        for r, row in enumerate(iterator, start=1):
            out[r - 1] = row
            if verbose and r % 10 == 0:
                print(f"  finished {r}/{runs} runs")

    return out


def log_ratio_matrix(T_samples, ET_hat, n_values):
    """
    R_{i,j} = log |T_{i,j} - ET_hat_j| / log n_j
    NaN when |.| = 0.
    """
    R = np.full(T_samples.shape, np.nan, dtype=float)
    logn = np.log(np.asarray(n_values, dtype=float))
    for j in range(len(n_values)):
        dev = np.abs(T_samples[:, j] - ET_hat[j])
        nz = dev > 0
        R[nz, j] = np.log(dev[nz]) / logn[j]
    return R


def main():
    # Use n =   2^k, k = 1, 2, 3,.., 10 (and n >= 2 for denominator log(n))
    n_values = np.array([(2**k) for k in range(1, 10)], dtype=np.int32)

    runs = 10        # one shared batch for mean + fluctuation diagnostics
    max_steps = 40_000
    workers = min(4, os.cpu_count() or 1)

    print("Running one shared batch for E[T_n] and fluctuation diagnostics")
    T_samples = sample_T_matrix(
        n_values,
        runs=runs,
        seed=123,
        max_steps=max_steps,
        verbose=True,
        workers=workers,
    )
    ET_hat = T_samples.mean(axis=0)
    R = log_ratio_matrix(T_samples, ET_hat, n_values)

    R_med = np.nanmedian(R, axis=0)
    R_mean = np.nanmean(R, axis=0)

    # Extra diagnostic: MAD scaling
    mad = np.median(np.abs(T_samples - ET_hat), axis=0)
    mask = mad > 0
    alpha = np.polyfit(np.log(n_values[mask]), np.log(mad[mask]), 1)[0]

    print("\nResults")
    print("n\tEhat[T_n]\tmedian(log-ratio)\tmean(log-ratio)\tMAD")
    for n, e, rm, rmu, m in zip(n_values, ET_hat, R_med, R_mean, mad):
        print(f"{n}\t{e:.2f}\t\t{rm:.3f}\t\t\t{rmu:.3f}\t\t{m:.2f}")

    print(f"\nEstimated exponent from MAD ~ n^alpha: alpha = {alpha:.3f}")
    print("Reference value: 1/3 = 0.333...")

    # Plots
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))

    ax[0].plot(n_values, ET_hat, "o-")
    ax[0].set_xlabel("n")
    ax[0].set_ylabel("sample mean of T_n")
    ax[0].set_title("Estimated E[T_n]")

    q25 = np.nanpercentile(R, 25, axis=0)
    q75 = np.nanpercentile(R, 75, axis=0)
    ax[1].plot(n_values, R_med, "o-", label="median ratio")
    ax[1].fill_between(n_values, q25, q75, alpha=0.2, label="IQR")
    ax[1].axhline(1 / 3, linestyle="--", color="red", label="1/3")
    ax[1].set_xlabel("n")
    ax[1].set_ylabel("log|T_n - Ehat[T_n]| / log n")
    ax[1].set_title("Fluctuation exponent diagnostic")
    ax[1].legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()