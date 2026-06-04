// Distribution simulator for T_n in the frog model on Z^2.
//
// Model:
// 1. At time 0, one infected frog at (0,0), one healthy frog at every other site
//    in the rectangle
//    [-floor(0.1*n), floor(1.1*n)] x [-floor(n^(3/4)), floor(n^(3/4))].
// 2. Only infected frogs move as independent simple random walks with reflection
//    in the rectangle
//    [-floor(0.2*n), floor(1.2*n)] x [-floor(n^(3/4)), floor(n^(3/4))].
// 3. When an infected frog first reaches a site in the initial rectangle, the
//    healthy frog there becomes infected.
//
// Quantity of interest:
// T_n = first infection time of site (n,0).
// We run multiple trials for one fixed n and print empirical distribution
// summaries. Optional CSV output is meant for overlaying the standardized
// samples with a Tracy-Widom density or CDF in Python, R, Mathematica, etc.
//
// Build (GCC/Clang):
//   gcc -O3 -march=native -flto -std=c11 -fopenmp frog_distr.c -lm -o frog_distr
//
// Example:
//   ./frog_distr --n 1000 --trials 1000 --csv frog_tn_n1000.csv

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

static inline int point_in_init_window(const SimBounds *bounds, int32_t x, int32_t y) {
    return bounds->init_x_lo <= x && x <= bounds->init_x_hi &&
           bounds->init_y_lo <= y && y <= bounds->init_y_hi;
}

static const int32_t DX[4] = {1, -1, 0, 0};
static const int32_t DY[4] = {0, 0, 1, -1};

#define CB_BITS   7
#define CB_SIDE   (1 << CB_BITS)
#define CB_MASK   (CB_SIDE - 1)
#define CB_BYTES  (CB_SIDE * CB_SIDE / 8)
#define CB_OFFSET (1 << 20)

typedef struct Arena {
    uint8_t *data;
    size_t used;
    size_t cap;
    struct Arena *prev;
} Arena;

static Arena *arena_new(size_t cap) {
    Arena *a = (Arena *)malloc(sizeof(Arena));
    if (!a) return NULL;
    a->data = (uint8_t *)calloc(cap, 1);
    if (!a->data) {
        free(a);
        return NULL;
    }
    a->used = 0;
    a->cap = cap;
    a->prev = NULL;
    return a;
}

static void arena_free_all(Arena *a) {
    while (a) {
        Arena *p = a->prev;
        free(a->data);
        free(a);
        a = p;
    }
}

static uint8_t *arena_alloc(Arena **head, size_t nbytes) {
    Arena *a = *head;
    if (a->used + nbytes > a->cap) {
        size_t nc = a->cap << 1;
        if (nc < nbytes) nc = nbytes;
        Arena *b = arena_new(nc);
        if (!b) return NULL;
        b->prev = a;
        *head = b;
        a = b;
    }
    uint8_t *p = a->data + a->used;
    a->used += nbytes;
    return p;
}

typedef struct {
    uint64_t *keys;
    uint8_t **bitmaps;
    uint8_t *occupied;
    size_t cap;
    size_t size;
    Arena *arena;
} ChunkMap;

static int chunkmap_init(ChunkMap *cm, size_t init_cap) {
    size_t cap = 64;
    while (cap < init_cap) cap <<= 1;

    cm->keys = (uint64_t *)malloc(cap * sizeof(uint64_t));
    cm->bitmaps = (uint8_t **)malloc(cap * sizeof(uint8_t *));
    cm->occupied = (uint8_t *)calloc(cap, 1);
    if (!cm->keys || !cm->bitmaps || !cm->occupied) {
        free(cm->keys);
        free(cm->bitmaps);
        free(cm->occupied);
        return -1;
    }

    cm->cap = cap;
    cm->size = 0;
    cm->arena = arena_new(512 * (size_t)CB_BYTES);
    if (!cm->arena) {
        free(cm->keys);
        free(cm->bitmaps);
        free(cm->occupied);
        return -1;
    }
    return 0;
}

static void chunkmap_free(ChunkMap *cm) {
    arena_free_all(cm->arena);
    free(cm->keys);
    free(cm->bitmaps);
    free(cm->occupied);
}

static int chunkmap_rehash(ChunkMap *cm) {
    size_t nc = cm->cap << 1;
    size_t mask = nc - 1;

    uint64_t *nk = (uint64_t *)malloc(nc * sizeof(uint64_t));
    uint8_t **nb = (uint8_t **)malloc(nc * sizeof(uint8_t *));
    uint8_t *no = (uint8_t *)calloc(nc, 1);
    if (!nk || !nb || !no) {
        free(nk);
        free(nb);
        free(no);
        return -1;
    }

    for (size_t i = 0; i < cm->cap; i++) {
        if (!cm->occupied[i]) continue;
        size_t p = (size_t)(hash64(cm->keys[i]) & mask);
        while (no[p]) p = (p + 1) & mask;
        nk[p] = cm->keys[i];
        nb[p] = cm->bitmaps[i];
        no[p] = 1;
    }

    free(cm->keys);
    free(cm->bitmaps);
    free(cm->occupied);
    cm->keys = nk;
    cm->bitmaps = nb;
    cm->occupied = no;
    cm->cap = nc;
    return 0;
}

// Returns 1 if point was newly activated, 0 if already active, -1 on allocation failure.
static inline int chunkmap_test_set(ChunkMap *cm, int32_t x, int32_t y) {
    uint32_t ux = (uint32_t)(x + CB_OFFSET);
    uint32_t uy = (uint32_t)(y + CB_OFFSET);

    uint64_t ckey = ((uint64_t)(ux >> CB_BITS) << 32) | (uint64_t)(uy >> CB_BITS);
    uint32_t bidx = ((uy & CB_MASK) << CB_BITS) | (ux & CB_MASK);
    uint32_t boff = bidx >> 3;
    uint8_t bmsk = (uint8_t)(1u << (bidx & 7));

    size_t mask = cm->cap - 1;
    size_t pos = (size_t)(hash64(ckey) & mask);

    while (cm->occupied[pos]) {
        if (cm->keys[pos] == ckey) {
            uint8_t *bmp = cm->bitmaps[pos];
            if (bmp[boff] & bmsk) return 0;
            bmp[boff] |= bmsk;
            return 1;
        }
        pos = (pos + 1) & mask;
    }

    if ((cm->size + 1) * 10 >= cm->cap * 7) {
        if (chunkmap_rehash(cm) != 0) return -1;
        mask = cm->cap - 1;
        pos = (size_t)(hash64(ckey) & mask);
        while (cm->occupied[pos]) pos = (pos + 1) & mask;
    }

    uint8_t *bmp = arena_alloc(&cm->arena, CB_BYTES);
    if (!bmp) return -1;

    cm->keys[pos] = ckey;
    cm->bitmaps[pos] = bmp;
    cm->occupied[pos] = 1;
    cm->size++;

    bmp[boff] |= bmsk;
    return 1;
}

// Simulate one run. Returns:
// 0 success, 1 if max_steps reached without infecting (n,0), -1 allocation failure.
static int simulate_one_tn(int32_t n, uint64_t seed, int max_steps, int32_t *t_out) {
    if (t_out == NULL) return -1;
    if (n == 0) {
        *t_out = 0;
        return 0;
    }
    if (max_steps <= 0) return -1;

    SimBounds bounds;
    if (bounds_from_n(n, &bounds) != 0) return -1;

    if (bounds.init_x_lo <= -CB_OFFSET || bounds.init_x_hi >= CB_OFFSET ||
        bounds.init_y_lo <= -CB_OFFSET || bounds.init_y_hi >= CB_OFFSET) {
        return -1;
    }

    size_t cap = 8192;
    size_t m = 1;
    int32_t *fx = (int32_t *)malloc(cap * sizeof(int32_t));
    int32_t *fy = (int32_t *)malloc(cap * sizeof(int32_t));
    size_t new_cap = 4096;
    int32_t *new_x = (int32_t *)malloc(new_cap * sizeof(int32_t));
    int32_t *new_y = (int32_t *)malloc(new_cap * sizeof(int32_t));
    ChunkMap activated;
    int cm_ok = chunkmap_init(&activated, 256);

    if (!fx || !fy || !new_x || !new_y || cm_ok != 0) {
        free(fx);
        free(fy);
        free(new_x);
        free(new_y);
        if (cm_ok == 0) chunkmap_free(&activated);
        return -1;
    }

    fx[0] = 0;
    fy[0] = 0;
    chunkmap_test_set(&activated, 0, 0);

    uint64_t rng_state = seed | 1ULL;

    for (int t = 1; t <= max_steps; t++) {
        for (size_t i = 0; i < m; i++) {
            uint64_t r = rng_next(&rng_state);
            int d = (int)(r & 3ULL);
            int64_t nx = (int64_t)fx[i] + (int64_t)DX[d];
            int64_t ny = (int64_t)fy[i] + (int64_t)DY[d];

            if (nx > (int64_t)bounds.box_x_hi) nx = 2LL * (int64_t)bounds.box_x_hi - nx;
            else if (nx < (int64_t)bounds.box_x_lo) nx = 2LL * (int64_t)bounds.box_x_lo - nx;
            if (ny > (int64_t)bounds.box_y_hi) ny = 2LL * (int64_t)bounds.box_y_hi - ny;
            else if (ny < (int64_t)bounds.box_y_lo) ny = 2LL * (int64_t)bounds.box_y_lo - ny;

            fx[i] = (int32_t)nx;
            fy[i] = (int32_t)ny;
        }

        size_t k = 0;
        for (size_t i = 0; i < m; i++) {
            int32_t xi = fx[i];
            int32_t yi = fy[i];

            if (!point_in_init_window(&bounds, xi, yi)) continue;

            int ins = chunkmap_test_set(&activated, xi, yi);
            if (ins < 0) goto fail;
            if (ins == 0) continue;

            if (xi == n && yi == 0) {
                *t_out = t;
                goto done;
            }

            if (k == new_cap) {
                size_t nc = new_cap << 1;
                int32_t *tx = (int32_t *)realloc(new_x, nc * sizeof(int32_t));
                if (!tx) goto fail;
                new_x = tx;
                int32_t *ty = (int32_t *)realloc(new_y, nc * sizeof(int32_t));
                if (!ty) goto fail;
                new_y = ty;
                new_cap = nc;
            }
            new_x[k] = xi;
            new_y[k] = yi;
            k++;
        }

        if (k > 0) {
            size_t need = m + k;
            if (need > cap) {
                while (cap < need) cap <<= 1;
                int32_t *tx = (int32_t *)realloc(fx, cap * sizeof(int32_t));
                if (!tx) goto fail;
                fx = tx;
                int32_t *ty = (int32_t *)realloc(fy, cap * sizeof(int32_t));
                if (!ty) goto fail;
                fy = ty;
            }
            memcpy(fx + m, new_x, k * sizeof(int32_t));
            memcpy(fy + m, new_y, k * sizeof(int32_t));
            m = need;
        }
    }

    chunkmap_free(&activated);
    free(fx);
    free(fy);
    free(new_x);
    free(new_y);
    return 1;

done:
    chunkmap_free(&activated);
    free(fx);
    free(fy);
    free(new_x);
    free(new_y);
    return 0;

fail:
    chunkmap_free(&activated);
    free(fx);
    free(fy);
    free(new_x);
    free(new_y);
    return -1;
}

static int cmp_int32(const void *a, const void *b) {
    int32_t x = *(const int32_t *)a;
    int32_t y = *(const int32_t *)b;
    return (x > y) - (x < y);
}

static double quantile_sorted_int32(const int32_t *a, int n, double p) {
    if (n <= 0) return NAN;
    if (p <= 0.0) return (double)a[0];
    if (p >= 1.0) return (double)a[n - 1];

    double pos = p * (double)(n - 1);
    int lo = (int)floor(pos);
    int hi = (int)ceil(pos);
    double w = pos - (double)lo;
    return (1.0 - w) * (double)a[lo] + w * (double)a[hi];
}

static int write_samples_csv(const char *path, const int32_t *samples, int trials,
                             int32_t n, double mean, double sd) {
    FILE *fp = fopen(path, "w");
    if (!fp) return -1;

    double n13 = cbrt((double)n);
    fprintf(fp, "trial,T_n,centered_by_n13,zscore\n");
    for (int r = 0; r < trials; r++) {
        double centered = ((double)samples[r] - mean) / n13;
        double z = (sd > 0.0) ? (((double)samples[r] - mean) / sd) : NAN;
        fprintf(fp, "%d,%d,%.17g,%.17g\n", r + 1, samples[r], centered, z);
    }

    return fclose(fp);
}

static int write_hist_csv(const char *path, const int32_t *samples, int trials,
                          double mean, double sd, int bins) {
    if (bins <= 0 || trials <= 0) return -1;

    int32_t min_t = samples[0];
    int32_t max_t = samples[0];
    for (int r = 1; r < trials; r++) {
        if (samples[r] < min_t) min_t = samples[r];
        if (samples[r] > max_t) max_t = samples[r];
    }

    double width = ((double)max_t - (double)min_t + 1.0) / (double)bins;
    if (width <= 0.0) width = 1.0;

    int *counts = (int *)calloc((size_t)bins, sizeof(int));
    if (!counts) return -1;

    for (int r = 0; r < trials; r++) {
        int b = (int)(((double)samples[r] - (double)min_t) / width);
        if (b < 0) b = 0;
        if (b >= bins) b = bins - 1;
        counts[b]++;
    }

    FILE *fp = fopen(path, "w");
    if (!fp) {
        free(counts);
        return -1;
    }

    fprintf(fp, "bin_left,bin_mid,bin_right,count,density,z_mid\n");
    for (int b = 0; b < bins; b++) {
        double left = (double)min_t + width * (double)b;
        double right = (double)min_t + width * (double)(b + 1);
        double mid = 0.5 * (left + right);
        double density = (double)counts[b] / ((double)trials * width);
        double z_mid = (sd > 0.0) ? ((mid - mean) / sd) : NAN;
        fprintf(fp, "%.17g,%.17g,%.17g,%d,%.17g,%.17g\n",
                left, mid, right, counts[b], density, z_mid);
    }

    int close_rc = fclose(fp);
    free(counts);
    return close_rc;
}

static void print_usage(const char *prog) {
    printf("Usage: %s [--n N] [--trials K] [--max-steps M] [--seed S] [--threads P]\n", prog);
    printf("          [--csv FILE] [--hist FILE] [--bins B] [--print-samples]\n");
    printf("Initial frogs: [-floor(0.1*n), floor(1.1*n)] x [-floor(n^(3/4)), floor(n^(3/4))].\n");
    printf("Reflecting box: [-floor(0.2*n), floor(1.2*n)] x [-floor(n^(3/4)), floor(n^(3/4))].\n");
    printf("Defaults: n=1000, trials=1000, max-steps=200000, seed=42, bins=50\n");
}

int main(int argc, char **argv) {
    int32_t n = 1000;
    int trials = 1000;
    int max_steps = 200000;
    uint64_t seed = 42;
    int threads = 1;
    int bins = 50;
    int print_samples = 0;
    const char *csv_path = NULL;
    const char *hist_path = NULL;

#ifdef _OPENMP
    threads = omp_get_max_threads();
#endif

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
        } else if (strcmp(argv[i], "--csv") == 0 && i + 1 < argc) {
            csv_path = argv[++i];
        } else if (strcmp(argv[i], "--hist") == 0 && i + 1 < argc) {
            hist_path = argv[++i];
        } else if (strcmp(argv[i], "--bins") == 0 && i + 1 < argc) {
            bins = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--print-samples") == 0) {
            print_samples = 1;
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
    if (bounds.init_x_lo <= -CB_OFFSET || bounds.init_x_hi >= CB_OFFSET ||
        bounds.init_y_lo <= -CB_OFFSET || bounds.init_y_hi >= CB_OFFSET) {
        fprintf(stderr, "Require initial window to stay within (%d,%d) due to internal coordinate packing.\n",
                -CB_OFFSET, CB_OFFSET);
        return 2;
    }
    if (trials <= 1 || max_steps <= 0 || threads <= 0 || bins <= 0) {
        fprintf(stderr, "Require trials>1, max-steps>0, threads>0, bins>0.\n");
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

    double mean = 0.0;
    for (int r = 0; r < trials; r++) mean += (double)samples[r];
    mean /= (double)trials;

    double var = 0.0;
    double third_moment = 0.0;
    double fourth_moment = 0.0;
    for (int r = 0; r < trials; r++) {
        double d = (double)samples[r] - mean;
        var += d * d;
        third_moment += d * d * d;
        fourth_moment += d * d * d * d;
    }
    var = (trials > 1) ? (var / (double)(trials - 1)) : 0.0;
    double sd = sqrt(var);
    double pop_var = var * (double)(trials - 1) / (double)trials;
    double pop_sd = sqrt(pop_var);
    double skewness = (pop_sd > 0.0) ? (third_moment / (double)trials / (pop_sd * pop_sd * pop_sd)) : NAN;
    double excess_kurtosis = (pop_var > 0.0) ? (fourth_moment / (double)trials / (pop_var * pop_var) - 3.0) : NAN;

    int32_t *sorted = (int32_t *)malloc((size_t)trials * sizeof(int32_t));
    if (!sorted) {
        fprintf(stderr, "Allocation failure for sorted sample array.\n");
        free(samples);
        return 1;
    }
    memcpy(sorted, samples, (size_t)trials * sizeof(int32_t));
    qsort(sorted, (size_t)trials, sizeof(int32_t), cmp_int32);

    double n13 = cbrt((double)n);
    double logn = log((double)n);
    double sd_exponent = (sd > 0.0) ? (log(sd) / logn) : NAN;

    clock_t t1 = clock();
    double elapsed = (double)(t1 - t0) / (double)CLOCKS_PER_SEC;

    printf("Frog model T_n distribution (finite rectangular initialization on Z^2)\n");
    printf("n=%d, trials=%d, max_steps=%d, seed=%llu\n",
           n, trials, max_steps, (unsigned long long)seed);
    printf("Initial frogs on [%d,%d] x [%d,%d]\n",
           bounds.init_x_lo, bounds.init_x_hi, bounds.init_y_lo, bounds.init_y_hi);
    printf("Reflecting box on [%d,%d] x [%d,%d]\n",
           bounds.box_x_lo, bounds.box_x_hi, bounds.box_y_lo, bounds.box_y_hi);
    printf("Mean(T_n)       = %.6f\n", mean);
    printf("SD(T_n)         = %.6f\n", sd);
    printf("SD(T_n)/n^(1/3) = %.6f\n", sd / n13);
    printf("log(SD)/log(n)  = %.6f\n", sd_exponent);
    printf("skewness        = %.6f\n", skewness);
    printf("excess kurtosis = %.6f\n", excess_kurtosis);
    printf("min             = %d\n", sorted[0]);
    printf("q05             = %.6f\n", quantile_sorted_int32(sorted, trials, 0.05));
    printf("q10             = %.6f\n", quantile_sorted_int32(sorted, trials, 0.10));
    printf("q25             = %.6f\n", quantile_sorted_int32(sorted, trials, 0.25));
    printf("median          = %.6f\n", quantile_sorted_int32(sorted, trials, 0.50));
    printf("q75             = %.6f\n", quantile_sorted_int32(sorted, trials, 0.75));
    printf("q90             = %.6f\n", quantile_sorted_int32(sorted, trials, 0.90));
    printf("q95             = %.6f\n", quantile_sorted_int32(sorted, trials, 0.95));
    printf("max             = %d\n", sorted[trials - 1]);
    printf("For Tracy-Widom comparison, overlay the empirical zscore column against a standardized TW law.\n");

    if (print_samples) {
        printf("Samples:\n");
        printf("trial,T_n,centered_by_n13,zscore\n");
        for (int r = 0; r < trials; r++) {
            double centered = ((double)samples[r] - mean) / n13;
            double z = (sd > 0.0) ? (((double)samples[r] - mean) / sd) : NAN;
            printf("%d,%d,%.17g,%.17g\n", r + 1, samples[r], centered, z);
        }
    }

    if (csv_path) {
        if (write_samples_csv(csv_path, samples, trials, n, mean, sd) != 0) {
            fprintf(stderr, "Could not write sample CSV: %s\n", csv_path);
            free(sorted);
            free(samples);
            return 1;
        }
        printf("sample_csv      = %s\n", csv_path);
    }

    if (hist_path) {
        if (write_hist_csv(hist_path, samples, trials, mean, sd, bins) != 0) {
            fprintf(stderr, "Could not write histogram CSV: %s\n", hist_path);
            free(sorted);
            free(samples);
            return 1;
        }
        printf("histogram_csv   = %s\n", hist_path);
        printf("histogram_bins  = %d\n", bins);
    }

    printf("elapsed      = %.3f sec\n", elapsed);

    free(sorted);
    free(samples);
    return 0;
}
