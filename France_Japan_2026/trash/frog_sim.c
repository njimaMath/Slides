// Fast C implementation of frog model simulation on Z^2
//
// Build:
//   gcc -O3 -march=native -flto -funroll-loops -fopenmp -std=c11 frog_sim.c -lm -o frog_sim
//
// Run:
//   ./frog_sim
//   ./frog_sim --runs 200 --max-steps 500000 --seed 42 --threads 8
//
// Optimisations over baseline:
//   1. Chunked 2-D bitmap replaces per-point hash set
//      → orders-of-magnitude fewer hash probes, great cache locality
//   2. Batch RNG: 32 two-bit directions per 64-bit random word (32× fewer calls)
//   3. Lookup table for direction deltas (no branch / switch)
//   4. O(1) power-of-two target detection via __builtin_ctz (no linear scan)
//   5. Arena allocator for bitmap chunks (no per-chunk malloc)
//   6. Auto-detect OpenMP thread count

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/* ────────────────────────────────────────────────────────
 *  RNG & hash
 * ──────────────────────────────────────────────────────── */

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

/* Direction lookup: 0→+x  1→−x  2→+y  3→−y */
static const int32_t DX[4] = { 1, -1,  0,  0};
static const int32_t DY[4] = { 0,  0,  1, -1};

/* ────────────────────────────────────────────────────────
 *  Chunked 2-D bitmap  (sparse visited-set)
 *
 *  The plane is tiled into 128×128 chunks.  A small open-
 *  addressed hash map maps chunk coordinates → 2 KB bitmap.
 *  Advantages over a per-point hash set:
 *    • far fewer hash-table entries  →  short probe chains
 *    • bitmap checks are simple bit-ops with good locality
 *    • arena allocator avoids per-chunk malloc overhead
 * ──────────────────────────────────────────────────────── */

#define CB_BITS   7                          /* log2(chunk side)    */
#define CB_SIDE   (1 << CB_BITS)             /* 128                 */
#define CB_MASK   (CB_SIDE - 1)              /* 0x7F                */
#define CB_BYTES  (CB_SIDE * CB_SIDE / 8)    /* 2048 B per chunk    */
#define CB_OFFSET (1 << 20)                  /* shift to non-neg    */

/* ---- arena allocator for chunk bitmaps ---- */

typedef struct Arena {
	uint8_t       *data;
	size_t         used, cap;
	struct Arena  *prev;
} Arena;

static Arena *arena_new(size_t cap) {
	Arena *a = (Arena *)malloc(sizeof(Arena));
	if (!a) return NULL;
	a->data = (uint8_t *)calloc(cap, 1);
	if (!a->data) { free(a); return NULL; }
	a->used = 0;
	a->cap  = cap;
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
		size_t nc = a->cap * 2;
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

/* ---- chunk hash map ---- */

typedef struct {
	uint64_t *keys;       /* packed (chunk_x, chunk_y) */
	uint8_t **bitmaps;    /* pointer into arena        */
	uint8_t  *occupied;
	size_t    cap, size;
	Arena    *arena;
} ChunkMap;

static int chunkmap_init(ChunkMap *cm, size_t init_cap) {
	size_t cap = 64;
	while (cap < init_cap) cap <<= 1;
	cm->keys    = (uint64_t *)malloc(cap * sizeof(uint64_t));
	cm->bitmaps = (uint8_t **)malloc(cap * sizeof(uint8_t *));
	cm->occupied= (uint8_t  *)calloc(cap, 1);
	if (!cm->keys || !cm->bitmaps || !cm->occupied) {
		free(cm->keys); free(cm->bitmaps); free(cm->occupied);
		return -1;
	}
	cm->cap  = cap;
	cm->size = 0;
	cm->arena = arena_new(512 * (size_t)CB_BYTES);  /* ~1 MB initial */
	if (!cm->arena) {
		free(cm->keys); free(cm->bitmaps); free(cm->occupied);
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
	size_t nc = cm->cap << 1, mask = nc - 1;
	uint64_t *nk = (uint64_t *)malloc(nc * sizeof(uint64_t));
	uint8_t **nb = (uint8_t **)malloc(nc * sizeof(uint8_t *));
	uint8_t  *no = (uint8_t  *)calloc(nc, 1);
	if (!nk || !nb || !no) { free(nk); free(nb); free(no); return -1; }
	for (size_t i = 0; i < cm->cap; i++) {
		if (!cm->occupied[i]) continue;
		size_t p = (size_t)(hash64(cm->keys[i]) & mask);
		while (no[p]) p = (p + 1) & mask;
		nk[p] = cm->keys[i];
		nb[p] = cm->bitmaps[i];
		no[p] = 1;
	}
	free(cm->keys); free(cm->bitmaps); free(cm->occupied);
	cm->keys = nk; cm->bitmaps = nb; cm->occupied = no; cm->cap = nc;
	return 0;
}

/*  Test-and-set a point in the chunked bitmap.
 *  Returns:  1 = newly set,  0 = already present,  -1 = alloc error */
static inline int chunkmap_test_set(ChunkMap *cm, int32_t x, int32_t y) {
	uint32_t ux = (uint32_t)(x + CB_OFFSET);
	uint32_t uy = (uint32_t)(y + CB_OFFSET);

	uint64_t ckey = ((uint64_t)(ux >> CB_BITS) << 32) | (uy >> CB_BITS);
	uint32_t bidx = ((uy & CB_MASK) << CB_BITS) | (ux & CB_MASK);
	uint32_t boff = bidx >> 3;
	uint8_t  bmsk = 1u << (bidx & 7);

	size_t mask = cm->cap - 1;
	size_t pos  = (size_t)(hash64(ckey) & mask);

	while (cm->occupied[pos]) {
		if (cm->keys[pos] == ckey) {
			uint8_t *bmp = cm->bitmaps[pos];
			if (bmp[boff] & bmsk) return 0;   /* already set */
			bmp[boff] |= bmsk;
			return 1;
		}
		pos = (pos + 1) & mask;
	}

	/* Chunk doesn't exist → allocate */
	if ((cm->size + 1) * 10 >= cm->cap * 7) {
		if (chunkmap_rehash(cm) != 0) return -1;
		mask = cm->cap - 1;
		pos = (size_t)(hash64(ckey) & mask);
		while (cm->occupied[pos]) pos = (pos + 1) & mask;
	}
	uint8_t *bmp = arena_alloc(&cm->arena, CB_BYTES);
	if (!bmp) return -1;

	cm->keys[pos]    = ckey;
	cm->bitmaps[pos] = bmp;
	cm->occupied[pos]= 1;
	cm->size++;

	bmp[boff] |= bmsk;
	return 1;
}

/* ────────────────────────────────────────────────────────
 *  Single simulation run
 * ──────────────────────────────────────────────────────── */

// returns 0 on success, 1 if max_steps reached, -1 on alloc failure
static int simulate_one_run(
	const int32_t *n_values,
	int n_count,
	uint64_t seed,
	int max_steps,
	int32_t *T_out)
{
	for (int j = 0; j < n_count; j++) T_out[j] = -1;
	int unresolved = n_count;

	/* O(1) target lookup for power-of-two targets via __builtin_ctz */
	int8_t target_idx[32];
	memset(target_idx, -1, sizeof(target_idx));
	int32_t n_min = n_values[0], n_max = n_values[n_count - 1];
	for (int j = 0; j < n_count; j++) {
		int32_t v = n_values[j];
		if (v > 0 && (v & (v - 1)) == 0) {
			int b = __builtin_ctz((unsigned)v);
			if (b < 32) target_idx[b] = (int8_t)j;
		}
	}

	/* Active frog positions */
	size_t cap = 8192, m = 1;
	int32_t *fx = (int32_t *)malloc(cap * sizeof(int32_t));
	int32_t *fy = (int32_t *)malloc(cap * sizeof(int32_t));
	/* Temp buffer for newly awakened frogs */
	size_t new_cap = 4096;
	int32_t *new_x = (int32_t *)malloc(new_cap * sizeof(int32_t));
	int32_t *new_y = (int32_t *)malloc(new_cap * sizeof(int32_t));

	ChunkMap activated;
	int cm_ok = chunkmap_init(&activated, 256);
	if (!fx || !fy || !new_x || !new_y || cm_ok != 0) {
		free(fx); free(fy); free(new_x); free(new_y);
		if (cm_ok == 0) chunkmap_free(&activated);
		return -1;
	}

	fx[0] = 0;  fy[0] = 0;
	chunkmap_test_set(&activated, 0, 0);

	uint64_t rng_state = seed | 1ULL;

	for (int t = 1; t <= max_steps; t++) {

		/* ── Move all active frogs ──
		 * Batch RNG: extract 32 two-bit directions per 64-bit word. */
		{
			uint64_t rbuf = 0;
			int rbits = 0;
			for (size_t i = 0; i < m; i++) {
				if (rbits == 0) { rbuf = rng_next(&rng_state); rbits = 32; }
				int d = (int)(rbuf & 3);
				rbuf >>= 2;
				rbits--;
				fx[i] += DX[d];
				fy[i] += DY[d];
			}
		}

		/* ── First-visit checks & awakenings ── */
		size_t k = 0;
		for (size_t i = 0; i < m; i++) {
			int32_t xi = fx[i], yi = fy[i];

			int ins = chunkmap_test_set(&activated, xi, yi);
			if (ins < 0) goto fail;
			if (ins == 0) continue;          /* already visited */

			/* New activation → buffer the new frog */
			if (k >= new_cap) {
				size_t nc = new_cap << 1;
				int32_t *tx = (int32_t *)realloc(new_x, nc * sizeof(int32_t));
				if (!tx) goto fail;  new_x = tx;
				int32_t *ty = (int32_t *)realloc(new_y, nc * sizeof(int32_t));
				if (!ty) goto fail;  new_y = ty;
				new_cap = nc;
			}
			new_x[k] = xi;
			new_y[k] = yi;
			k++;

			/* Is this one of our target sites? */
			if (yi == 0 && xi >= n_min && xi <= n_max
			    && (xi & (xi - 1)) == 0) {
				int b = __builtin_ctz((unsigned)xi);
				int8_t j = (b < 32) ? target_idx[b] : -1;
				if (j >= 0 && T_out[j] == -1) {
					T_out[j] = t;
					if (--unresolved == 0) goto done;
				}
			}
		}

		/* ── Append awakened frogs ── */
		if (k > 0) {
			size_t need = m + k;
			if (need > cap) {
				while (cap < need) cap <<= 1;
				int32_t *tx = (int32_t *)realloc(fx, cap * sizeof(int32_t));
				if (!tx) goto fail;  fx = tx;
				int32_t *ty = (int32_t *)realloc(fy, cap * sizeof(int32_t));
				if (!ty) goto fail;  fy = ty;
			}
			memcpy(fx + m, new_x, k * sizeof(int32_t));
			memcpy(fy + m, new_y, k * sizeof(int32_t));
			m += k;
		}
	}

	/* max_steps reached without resolving all targets */
	chunkmap_free(&activated);
	free(fx); free(fy); free(new_x); free(new_y);
	return 1;

done:
	chunkmap_free(&activated);
	free(fx); free(fy); free(new_x); free(new_y);
	return 0;

fail:
	chunkmap_free(&activated);
	free(fx); free(fy); free(new_x); free(new_y);
	return -1;
}

/* ────────────────────────────────────────────────────────
 *  Statistics helpers
 * ──────────────────────────────────────────────────────── */

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

static double quantile_sorted_linear(const double *a, int n, double q) {
	if (n <= 0) return NAN;
	if (q <= 0.0) return a[0];
	if (q >= 1.0) return a[n - 1];
	double pos = q * (n - 1);
	int i = (int)floor(pos);
	int j = i + 1;
	if (j >= n) return a[n - 1];
	double w = pos - i;
	return a[i] * (1.0 - w) + a[j] * w;
}

/* ────────────────────────────────────────────────────────
 *  main
 * ──────────────────────────────────────────────────────── */

static void print_usage(const char *prog) {
	printf("Usage: %s [--runs N] [--max-steps N] [--seed N] "
	       "[--threads N] [--csv PATH]\n", prog);
}

int main(int argc, char **argv) {
	/* n = 2^k, k = 5 … 12 */
	int32_t n_values[] = {32, 64, 128, 256, 512, 1024, 2048, 4096};
	const int n_count = (int)(sizeof(n_values) / sizeof(n_values[0]));

	int      runs      = 200;
	int      max_steps = 500000;
	uint64_t seed      = 42;
	int      threads   = 1;
	const char *csv_path = "frog_results.csv";

#ifdef _OPENMP
	threads = omp_get_max_threads();
#endif

	for (int i = 1; i < argc; ++i) {
		if (strcmp(argv[i], "--runs") == 0 && i + 1 < argc) {
			runs = atoi(argv[++i]);
		} else if (strcmp(argv[i], "--max-steps") == 0 && i + 1 < argc) {
			max_steps = atoi(argv[++i]);
		} else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
			seed = (uint64_t)strtoull(argv[++i], NULL, 10);
		} else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
			threads = atoi(argv[++i]);
		} else if (strcmp(argv[i], "--csv") == 0 && i + 1 < argc) {
			csv_path = argv[++i];
		} else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
			print_usage(argv[0]);
			return 0;
		} else {
			print_usage(argv[0]);
			return 2;
		}
	}

	if (runs <= 0 || max_steps <= 0 || threads <= 0) {
		fprintf(stderr, "runs, max-steps, threads must be positive\n");
		return 2;
	}

#ifdef _OPENMP
	omp_set_num_threads(threads);
#else
	(void)threads;
#endif

	printf("Frog model on Z^2  |  runs=%d  max_steps=%d  threads=%d  seed=%llu\n",
	       runs, max_steps, threads, (unsigned long long)seed);
	printf("Targets: n =");
	for (int j = 0; j < n_count; j++) printf(" %d", n_values[j]);
	printf("\n\n");

	struct timespec wall0;
	clock_gettime(CLOCK_MONOTONIC, &wall0);

	int32_t *T_samples = (int32_t *)malloc((size_t)runs * n_count * sizeof(int32_t));
	if (!T_samples) {
		fprintf(stderr, "allocation failed for T_samples\n");
		return 1;
	}

	int failed = 0;

	/* Parallel over independent runs */
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
	for (int r = 0; r < runs; ++r) {
		uint64_t s = seed_mix(seed + (uint64_t)r * 0x9e3779b97f4a7c15ULL);
		int rc = simulate_one_run(n_values, n_count, s, max_steps,
		                          T_samples + (size_t)r * n_count);
		if (rc != 0) {
#ifdef _OPENMP
#pragma omp atomic write
#endif
			failed = 1;
		}
	}

	struct timespec wall1;
	clock_gettime(CLOCK_MONOTONIC, &wall1);
	double elapsed = (wall1.tv_sec  - wall0.tv_sec)
	               + (wall1.tv_nsec - wall0.tv_nsec) * 1e-9;

	if (failed) {
		fprintf(stderr,
		    "Some runs failed (likely max_steps too small). "
		    "Increase --max-steps.\n");
		free(T_samples);
		return 1;
	}

	/* ── Compute statistics ── */

	double *ET_hat = (double *)calloc(n_count, sizeof(double));
	double *R_med  = (double *)malloc(n_count * sizeof(double));
	double *R_mean = (double *)malloc(n_count * sizeof(double));
	double *q25    = (double *)malloc(n_count * sizeof(double));
	double *q75    = (double *)malloc(n_count * sizeof(double));
	double *mad    = (double *)malloc(n_count * sizeof(double));
	double *tmp    = (double *)malloc((size_t)runs * sizeof(double));

	if (!ET_hat || !R_med || !R_mean || !q25 || !q75 || !mad || !tmp) {
		fprintf(stderr, "allocation failed for stats arrays\n");
		free(T_samples);
		free(ET_hat); free(R_med); free(R_mean);
		free(q25); free(q75); free(mad); free(tmp);
		return 1;
	}

	for (int j = 0; j < n_count; ++j) {
		double sum = 0.0;
		for (int r = 0; r < runs; ++r)
			sum += (double)T_samples[(size_t)r * n_count + j];
		ET_hat[j] = sum / runs;
	}

	for (int j = 0; j < n_count; ++j) {
		double logn = log((double)n_values[j]);
		int cnt = 0;
		double sum = 0.0;
		for (int r = 0; r < runs; ++r) {
			double dev = fabs((double)T_samples[(size_t)r * n_count + j] - ET_hat[j]);
			if (dev > 0.0) {
				double val = log(dev) / logn;
				tmp[cnt++] = val;
				sum += val;
			}
		}
		if (cnt == 0) {
			R_mean[j] = R_med[j] = q25[j] = q75[j] = NAN;
		} else {
			qsort(tmp, (size_t)cnt, sizeof(double), cmp_double);
			R_mean[j] = sum / cnt;
			R_med[j]  = median_sorted(tmp, cnt);
			q25[j]    = quantile_sorted_linear(tmp, cnt, 0.25);
			q75[j]    = quantile_sorted_linear(tmp, cnt, 0.75);
		}
	}

	for (int j = 0; j < n_count; ++j) {
		for (int r = 0; r < runs; ++r)
			tmp[r] = fabs((double)T_samples[(size_t)r * n_count + j] - ET_hat[j]);
		qsort(tmp, (size_t)runs, sizeof(double), cmp_double);
		mad[j] = median_sorted(tmp, runs);
	}

	/* alpha from MAD ~ n^alpha via least squares on log-log */
	double sx = 0.0, sy = 0.0, sxx = 0.0, sxy = 0.0;
	int k = 0;
	for (int j = 0; j < n_count; ++j) {
		if (mad[j] <= 0.0) continue;
		double x = log((double)n_values[j]);
		double y = log(mad[j]);
		sx += x; sy += y; sxx += x * x; sxy += x * y; k++;
	}
	double alpha = NAN;
	if (k >= 2) {
		double den = k * sxx - sx * sx;
		if (fabs(den) > 1e-15) alpha = (k * sxy - sx * sy) / den;
	}

	printf("Results  (%.1f s elapsed)\n", elapsed);
	printf("n\tEhat[T_n]\tmedian(log-ratio)\tmean(log-ratio)\tMAD\tQ25\tQ75\n");
	for (int j = 0; j < n_count; ++j) {
		printf(
			"%d\t%.2f\t\t%.3f\t\t\t%.3f\t\t%.2f\t%.3f\t%.3f\n",
			n_values[j], ET_hat[j], R_med[j], R_mean[j], mad[j], q25[j], q75[j]
		);
	}
	printf("\nEstimated exponent from MAD ~ n^alpha: alpha = %.3f\n", alpha);
	printf("Reference value: 1/3 = 0.333...\n");

	FILE *fp = fopen(csv_path, "w");
	if (fp) {
		fprintf(fp, "n,ehat_tn,median_log_ratio,mean_log_ratio,mad,q25,q75\n");
		for (int j = 0; j < n_count; ++j) {
			fprintf(fp, "%d,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g\n",
				n_values[j], ET_hat[j], R_med[j], R_mean[j],
				mad[j], q25[j], q75[j]);
		}
		fclose(fp);
		printf("CSV written to %s\n", csv_path);
	} else {
		fprintf(stderr, "failed to open CSV output: %s\n", csv_path);
	}

	free(T_samples);
	free(ET_hat); free(R_med); free(R_mean);
	free(q25); free(q75); free(mad); free(tmp);
	return 0;
}
