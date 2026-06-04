// KS infection simulator on Z^2 (finite initial window).
//
// Model:
// 1. At time 0, one infected frog at (0,0), one healthy frog at every
//    other site in the rectangle
//    [-floor(0.1*n), floor(1.1*n)] x [-floor(n^(3/4)), floor(n^(3/4))].
// 2. Both infected and healthy frogs move as independent simple random walks
//    with reflection in the rectangle
//    [-floor(0.2*n), floor(1.2*n)] x [-floor(n^(3/4)), floor(n^(3/4))].
//    To break bipartite parity locking in synchronous discrete time, each frog
//    uses a lazy clock: with probability 1/5 it stays put for that step.
// 3. After movement, if a healthy frog shares a site with at least one infected
//    frog, it becomes infected.
//
// Quantity of interest:
// T_n = first time the frog that started at (n,0) becomes infected.
// We run multiple trials, estimate E[T_n] by sample mean, and print:
//   log |T_n - Ehat[T_n]| / log n
//
// Build (GCC/Clang):
//   gcc -O3 -march=native -flto -std=c11 -fopenmp KS.c -lm -o KS
//
// Example:
//   ./KS --n 64 --trials 5 --max-steps 60000 --seed 42
// 'c:\Users\shuta\Documents\Github\Research\Slide\France_Japan_2026\KS.exe' --n 5000
#include <math.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _OPENMP
#include <omp.h>
#endif

static inline uint64_t hash64(uint64_t x) {
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebULL;
    x ^= x >> 31;
    return x;
}

static inline uint64_t rng_next(uint64_t *state) {
    uint64_t x = *state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    return x * 0x2545F4914F6CDD1DULL;
}

static inline uint64_t seed_mix(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    return hash64(x);
}

static inline uint64_t pack_xy(int32_t x, int32_t y) {
    return ((uint64_t)(uint32_t)x << 32) | (uint64_t)(uint32_t)y;
}

static const int32_t DX[4] = {1, -1, 0, 0};
static const int32_t DY[4] = {0, 0, 1, -1};

typedef struct {
    uint64_t *keys;
    uint32_t *stamp;
    uint32_t epoch;
    size_t cap;
} PosSet;

static int posset_init(PosSet *ps, size_t min_cap) {
    size_t cap = 64;
    while (cap < min_cap) {
        if (cap > (SIZE_MAX >> 1)) return -1;
        cap <<= 1;
    }

    ps->keys = (uint64_t *)malloc(cap * sizeof(uint64_t));
    ps->stamp = (uint32_t *)calloc(cap, sizeof(uint32_t));
    if (!ps->keys || !ps->stamp) {
        free(ps->keys);
        free(ps->stamp);
        return -1;
    }

    ps->epoch = 1;
    ps->cap = cap;
    return 0;
}

static void posset_free(PosSet *ps) {
    free(ps->keys);
    free(ps->stamp);
}

static inline void posset_clear(PosSet *ps) {
    ps->epoch++;
    if (ps->epoch == 0) {
        memset(ps->stamp, 0, ps->cap * sizeof(uint32_t));
        ps->epoch = 1;
    }
}

static inline void posset_insert(PosSet *ps, int32_t x, int32_t y) {
    uint64_t key = pack_xy(x, y);
    size_t mask = ps->cap - 1;
    size_t p = (size_t)(hash64(key) & mask);

    while (ps->stamp[p] == ps->epoch) {
        if (ps->keys[p] == key) return;
        p = (p + 1) & mask;
    }

    ps->keys[p] = key;
    ps->stamp[p] = ps->epoch;
}

static inline int posset_contains(const PosSet *ps, int32_t x, int32_t y) {
    uint64_t key = pack_xy(x, y);
    size_t mask = ps->cap - 1;
    size_t p = (size_t)(hash64(key) & mask);

    while (ps->stamp[p] == ps->epoch) {
        if (ps->keys[p] == key) return 1;
        p = (p + 1) & mask;
    }
    return 0;
}

typedef struct {
    int32_t init_x_lo;
    int32_t init_x_hi;
    int32_t init_y_lo;
    int32_t init_y_hi;
    int32_t box_x_lo;
    int32_t box_x_hi;
    int32_t box_y_lo;
    int32_t box_y_hi;
} SimBounds;

static int bounds_from_n(int32_t n, SimBounds *bounds) {
    if (n <= 0 || bounds == NULL) return -1;

    int64_t n64 = (int64_t)n;
    int64_t floor_01n = n64 / 10LL;
    int64_t floor_02n = n64 / 5LL;
    int64_t floor_n34 = (int64_t)floorl(powl((long double)n, 0.75L));
    int64_t floor_11n = (11LL * n64) / 10LL;
    int64_t floor_12n = (6LL * n64) / 5LL;

    if (floor_12n > (int64_t)INT_MAX || floor_n34 > (int64_t)INT_MAX) return -1;

    bounds->init_x_lo = (int32_t)(-floor_01n);
    bounds->init_x_hi = (int32_t)floor_11n;
    bounds->init_y_lo = (int32_t)(-floor_n34);
    bounds->init_y_hi = (int32_t)floor_n34;
    bounds->box_x_lo = (int32_t)(-floor_02n);
    bounds->box_x_hi = (int32_t)floor_12n;
    bounds->box_y_lo = (int32_t)(-floor_n34);
    bounds->box_y_hi = (int32_t)floor_n34;
    return 0;
}

// Returns:
// 0 success
// 1 max_steps reached before the frog initially at (n,0) becomes infected
// -1 allocation failure or invalid parameters
static int simulate_one_tn(int32_t n, uint64_t seed, int max_steps, int32_t *t_out) {
    if (n == 0) {
        *t_out = 0;
        return 0;
    }
    if (max_steps <= 0) return -1;

    SimBounds bounds;
    if (bounds_from_n(n, &bounds) != 0) return -1;

    int64_t width64 = (int64_t)bounds.init_x_hi - (int64_t)bounds.init_x_lo + 1LL;
    int64_t height64 = (int64_t)bounds.init_y_hi - (int64_t)bounds.init_y_lo + 1LL;
    if (width64 <= 0 || height64 <= 0) return -1;
    if ((uint64_t)width64 > (uint64_t)(SIZE_MAX / (size_t)height64)) return -1;
    size_t frog_count = (size_t)(width64 * height64);

    int32_t *fx = (int32_t *)malloc(frog_count * sizeof(int32_t));
    int32_t *fy = (int32_t *)malloc(frog_count * sizeof(int32_t));
    uint8_t *infected = (uint8_t *)calloc(frog_count, 1);
    PosSet infected_pos;

    size_t min_cap = frog_count;
    if (min_cap > SIZE_MAX / 4) min_cap = SIZE_MAX;
    else min_cap *= 4;

    int ps_ok = posset_init(&infected_pos, min_cap);
    if (!fx || !fy || !infected || ps_ok != 0) {
        free(fx);
        free(fy);
        free(infected);
        if (ps_ok == 0) posset_free(&infected_pos);
        return -1;
    }

    size_t idx = 0;
    size_t target_idx = SIZE_MAX;
    for (int32_t y = bounds.init_y_lo; y <= bounds.init_y_hi; y++) {
        for (int32_t x = bounds.init_x_lo; x <= bounds.init_x_hi; x++) {
            fx[idx] = x;
            fy[idx] = y;
            infected[idx] = (x == 0 && y == 0) ? 1u : 0u;
            if (x == n && y == 0) target_idx = idx;
            idx++;
        }
    }
    if (target_idx == SIZE_MAX) {
        posset_free(&infected_pos);
        free(fx);
        free(fy);
        free(infected);
        return -1;
    }

    uint64_t rng_state = seed | 1ULL;

    for (int t = 1; t <= max_steps; t++) {
        for (size_t i = 0; i < frog_count; i++) {
            uint64_t r = rng_next(&rng_state);
            if (r % 5ULL == 0ULL) continue; // lazy step: stay put with probability 1/5
            int d = (int)((r >> 3) & 3ULL);

            int64_t nx = (int64_t)fx[i] + (int64_t)DX[d];
            int64_t ny = (int64_t)fy[i] + (int64_t)DY[d];

            if (nx > (int64_t)bounds.box_x_hi) nx = 2LL * (int64_t)bounds.box_x_hi - nx;
            else if (nx < (int64_t)bounds.box_x_lo) nx = 2LL * (int64_t)bounds.box_x_lo - nx;
            if (ny > (int64_t)bounds.box_y_hi) ny = 2LL * (int64_t)bounds.box_y_hi - ny;
            else if (ny < (int64_t)bounds.box_y_lo) ny = 2LL * (int64_t)bounds.box_y_lo - ny;

            fx[i] = (int32_t)nx;
            fy[i] = (int32_t)ny;
        }

        posset_clear(&infected_pos);
        for (size_t i = 0; i < frog_count; i++) {
            if (infected[i]) posset_insert(&infected_pos, fx[i], fy[i]);
        }

        // Synchronous infection update: meeting infected frogs at this time step
        // causes infection, and newly infected frogs do not transmit until next step.
        int target_newly_infected = 0;
        for (size_t i = 0; i < frog_count; i++) {
            if (!infected[i] && posset_contains(&infected_pos, fx[i], fy[i])) {
                infected[i] = 1u;
                if (i == target_idx) target_newly_infected = 1;
            }
        }
        if (target_newly_infected) {
            *t_out = t;
            posset_free(&infected_pos);
            free(fx);
            free(fy);
            free(infected);
            return 0;
        }
    }

    posset_free(&infected_pos);
    free(fx);
    free(fy);
    free(infected);
    return 1;
}

static int cmp_double(const void *a, const void *b) {
    double x = *(const double *)a;
    double y = *(const double *)b;
    return (x > y) - (x < y);
}

static double median_sorted(const double *a, int n) {
    if (n <= 0) return NAN;
    if (n & 1) return a[n / 2];
    return 0.5 * (a[n / 2 - 1] + a[n / 2]);
}

static void print_usage(const char *prog) {
    printf("Usage: %s [--n N] [--trials K] [--max-steps M] [--seed S] [--threads P]\n", prog);
    printf("Initial particles: [-floor(0.1*n), floor(1.1*n)] x [-floor(n^(3/4)), floor(n^(3/4))].\n");
    printf("Reflecting box: [-floor(0.2*n), floor(1.2*n)] x [-floor(n^(3/4)), floor(n^(3/4))].\n");
    printf("Defaults: n=500, trials=5, max-steps=1000000, seed=42, threads=1\n");
}

int main(int argc, char **argv) {
    int32_t n = 500;
    int trials = 5;
    int max_steps = 1000000;
    uint64_t seed = 42;
    int threads = 1;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--n") == 0 && i + 1 < argc) {
            n = (int32_t)strtol(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--trials") == 0 && i + 1 < argc) {
            trials = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--max-steps") == 0 && i + 1 < argc) {
            max_steps = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            seed = (uint64_t)strtoull(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
            threads = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        } else {
            print_usage(argv[0]);
            return 2;
        }
    }

    if (n <= 0) {
        fprintf(stderr, "Require n>0.\n");
        return 2;
    }

    SimBounds bounds;
    if (bounds_from_n(n, &bounds) != 0) {
        fprintf(stderr, "n is too large for geometry computation.\n");
        return 2;
    }

    if (trials <= 1 || max_steps <= 0 || threads <= 0) {
        fprintf(stderr, "Require trials>1, max-steps>0, threads>0.\n");
        return 2;
    }

#ifdef _OPENMP
    omp_set_num_threads(threads);
#else
    (void)threads;
#endif

    int32_t *samples = (int32_t *)malloc((size_t)trials * sizeof(int32_t));
    if (!samples) {
        fprintf(stderr, "Allocation failure for sample array.\n");
        return 1;
    }

    clock_t t0 = clock();
    int fail_code = 0;

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for (int r = 0; r < trials; r++) {
        uint64_t s = seed_mix(seed + 0x9e3779b97f4a7c15ULL * (uint64_t)(r + 1));
        int rc = simulate_one_tn(n, s, max_steps, &samples[r]);
        if (rc != 0) {
            int code = (rc == 1) ? 1 : 2;
#ifdef _OPENMP
#pragma omp critical
#endif
            {
                if (code > fail_code) fail_code = code;
            }
        }
    }

    if (fail_code != 0) {
        if (fail_code == 1) {
            fprintf(stderr, "At least one trial did not finish. Increase --max-steps.\n");
        } else {
            fprintf(stderr, "Allocation failure or invalid parameter range during simulation.\n");
        }
        free(samples);
        return 1;
    }

    double ehat = 0.0;
    for (int r = 0; r < trials; r++) ehat += (double)samples[r];
    ehat /= (double)trials;

    double *ratios = (double *)malloc((size_t)trials * sizeof(double));
    if (!ratios) {
        fprintf(stderr, "Allocation failure for ratio array.\n");
        free(samples);
        return 1;
    }

    int cnt = 0;
    double ratio_sum = 0.0;
    double logn = log((double)n);
    for (int r = 0; r < trials; r++) {
        double dev = fabs((double)samples[r] - ehat);
        if (dev <= 0.0) continue;
        double val = log(dev) / logn;
        ratios[cnt++] = val;
        ratio_sum += val;
    }

    if (cnt > 0) qsort(ratios, (size_t)cnt, sizeof(double), cmp_double);
    double ratio_mean = (cnt > 0) ? (ratio_sum / (double)cnt) : NAN;
    double ratio_median = (cnt > 0) ? median_sorted(ratios, cnt) : NAN;

    clock_t t1 = clock();
    double elapsed = (double)(t1 - t0) / (double)CLOCKS_PER_SEC;

    printf("KS infection model (finite rectangular initialization on Z^2)\n");
    printf("n=%d, trials=%d, max_steps=%d, seed=%llu\n",
           n, trials, max_steps, (unsigned long long)seed);
    printf("Initial particles on [%d,%d] x [%d,%d]\n",
           bounds.init_x_lo, bounds.init_x_hi, bounds.init_y_lo, bounds.init_y_hi);
    printf("Reflecting box on [%d,%d] x [%d,%d]\n",
           bounds.box_x_lo, bounds.box_x_hi, bounds.box_y_lo, bounds.box_y_hi);
    printf("T_n = first time the frog initially at (n,0) becomes infected\n");
    printf("Sample mean Ehat[T_n] = %.6f\n", ehat);
    printf("Per-trial values of log|T_n - Ehat[T_n]| / log n:\n");

    for (int r = 0; r < trials; r++) {
        double dev = fabs((double)samples[r] - ehat);
        if (dev <= 0.0) {
            printf("trial %2d: T_n=%d  ratio=NaN (zero deviation)\n", r + 1, samples[r]);
        } else {
            double val = log(dev) / logn;
            printf("trial %2d: T_n=%d  ratio=%.6f\n", r + 1, samples[r], val);
        }
    }

    printf("Summary across nonzero deviations (%d/%d trials):\n", cnt, trials);
    printf("mean ratio   = %.6f\n", ratio_mean);
    printf("median ratio = %.6f\n", ratio_median);
    printf("target value = 0.333333 (1/3)\n");
    printf("elapsed      = %.3f sec\n", elapsed);

    free(ratios);
    free(samples);
    return 0;
}
