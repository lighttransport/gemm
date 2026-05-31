/*
 * Flash Attention Benchmark for AMD Zen2 (Ryzen 9 3950X)
 *
 * Measures performance and correctness of:
 *   - AVX2/FMA optimized attention
 *   - Reference implementation
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>
#include <sched.h>

#include "flash_attn_avx2.h"

/*=============================================================================
 * Timing utilities
 *============================================================================*/

static inline double get_time_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static inline uint64_t rdtscp(void) {
    uint32_t lo, hi, aux;
    __asm__ __volatile__ ("rdtscp" : "=a"(lo), "=d"(hi), "=c"(aux));
    return ((uint64_t)hi << 32) | lo;
}

static inline void mfence(void) {
    __asm__ __volatile__ ("mfence" ::: "memory");
}

static double detect_cpu_freq_ghz(void) {
    double t0 = get_time_sec();
    uint64_t start = rdtscp();
    mfence();

    while (get_time_sec() - t0 < 0.1) { }

    mfence();
    uint64_t end = rdtscp();
    double t1 = get_time_sec();

    return (double)(end - start) / (t1 - t0) / 1e9;
}

/*=============================================================================
 * Matrix utilities
 *============================================================================*/

static float *alloc_aligned(size_t n) {
    float *ptr = NULL;
    if (posix_memalign((void **)&ptr, 32, n * sizeof(float)) != 0) {
        return NULL;
    }
    return ptr;
}

/* PCG32 random number generator */
typedef struct {
    uint64_t state;
    uint64_t inc;
} pcg32_t;

static inline uint32_t pcg32_next(pcg32_t *rng) {
    uint64_t oldstate = rng->state;
    rng->state = oldstate * 6364136223846793005ULL + rng->inc;
    uint32_t xorshifted = (uint32_t)(((oldstate >> 18u) ^ oldstate) >> 27u);
    uint32_t rot = (uint32_t)(oldstate >> 59u);
    return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}

static inline float pcg32_float(pcg32_t *rng) {
    return (float)pcg32_next(rng) / (float)UINT32_MAX;
}

static void init_random(float *A, size_t n, uint64_t seed) {
    pcg32_t rng = { seed, seed ^ 0xdeadbeefULL };
    for (size_t i = 0; i < n; i++) {
        A[i] = (pcg32_float(&rng) - 0.5f) * 2.0f;  /* [-1, 1] */
    }
}

/*=============================================================================
 * Correctness verification
 *============================================================================*/

static double compute_max_error(const float *ref, const float *test, size_t n) {
    double max_err = 0.0;
    for (size_t i = 0; i < n; i++) {
        double err = fabs((double)ref[i] - (double)test[i]);
        if (err > max_err) {
            max_err = err;
        }
    }
    return max_err;
}

static double compute_relative_error(const float *ref, const float *test, size_t n) {
    double max_rel_err = 0.0;
    for (size_t i = 0; i < n; i++) {
        double r = (double)ref[i];
        double t = (double)test[i];
        double err = fabs(r - t);
        double denom = fmax(fabs(r), 1e-8);
        double rel_err = err / denom;
        if (rel_err > max_rel_err) {
            max_rel_err = rel_err;
        }
    }
    return max_rel_err;
}

/*=============================================================================
 * Benchmark runner
 *============================================================================*/

typedef struct {
    double time_sec;
    double gflops;
    uint64_t cycles;
} bench_result_t;

static bench_result_t benchmark_tile_attention(
    void (*attn_func)(const float*, const float*, const float*, float*,
                      float*, float*, float*),
    const float *Q, const float *K, const float *V,
    float *O, float *S, float *m, float *l,
    int warmup_iters, int bench_iters
) {
    bench_result_t result = {0};

    /* Warm up */
    for (int i = 0; i < warmup_iters; i++) {
        attn_func(Q, K, V, O, S, m, l);
    }

    /* Benchmark */
    mfence();
    uint64_t start_cycles = rdtscp();
    double start_time = get_time_sec();

    for (int i = 0; i < bench_iters; i++) {
        attn_func(Q, K, V, O, S, m, l);
    }

    mfence();
    double end_time = get_time_sec();
    uint64_t end_cycles = rdtscp();

    result.time_sec = (end_time - start_time) / bench_iters;
    result.cycles = (end_cycles - start_cycles) / bench_iters;

    /*
     * FLOPs calculation for Flash Attention tile:
     * Pass 1 (S = Q @ K^T): BR * BC * D * 2 = 4 * 64 * 64 * 2 = 32,768
     * Pass 2 (O = P @ V):   BR * D * BC * 2 = 4 * 64 * 64 * 2 = 32,768
     * exp:                  BR * BC * ~8 ops = 4 * 64 * 8 = 2,048
     * Total: ~67,584 FLOPs
     */
    double flops = 2.0 * FA_BR * FA_BC * FA_D +   /* S = Q @ K^T */
                   2.0 * FA_BR * FA_D * FA_BC +   /* O = P @ V */
                   8.0 * FA_BR * FA_BC;           /* exp + softmax */

    result.gflops = flops / result.time_sec / 1e9;

    return result;
}

static bench_result_t benchmark_seq_attention(
    const float *Q, const float *K, const float *V, float *O,
    size_t seq_len, int warmup_iters, int bench_iters
) {
    bench_result_t result = {0};

    /* Warm up */
    for (int i = 0; i < warmup_iters; i++) {
        flash_attention_avx2(Q, K, V, O, seq_len);
    }

    /* Benchmark */
    mfence();
    uint64_t start_cycles = rdtscp();
    double start_time = get_time_sec();

    for (int i = 0; i < bench_iters; i++) {
        flash_attention_avx2(Q, K, V, O, seq_len);
    }

    mfence();
    double end_time = get_time_sec();
    uint64_t end_cycles = rdtscp();

    result.time_sec = (end_time - start_time) / bench_iters;
    result.cycles = (end_cycles - start_cycles) / bench_iters;

    /* FLOPs: 2 * BR * seq_len * D (for QK^T) + 2 * BR * D * seq_len (for PV) */
    double flops = 2.0 * FA_BR * seq_len * FA_D +
                   2.0 * FA_BR * FA_D * seq_len +
                   8.0 * FA_BR * seq_len;

    result.gflops = flops / result.time_sec / 1e9;

    return result;
}

/*=============================================================================
 * Test functions
 *============================================================================*/

static void test_exp_avx2(void) {
    printf("Testing exp_avx2...\n");

    float input[32] __attribute__((aligned(32)));
    float output[32] __attribute__((aligned(32)));
    float ref[32];

    /* Initialize with test values */
    for (int i = 0; i < 32; i++) {
        input[i] = -10.0f + 20.0f * i / 31.0f;  /* [-10, 10] */
    }

    /* Compute */
    exp_avx2(input, output, 32);

    /* Reference */
    for (int i = 0; i < 32; i++) {
        ref[i] = expf(input[i]);
    }

    /* Check error */
    double max_err = 0.0;
    double max_rel_err = 0.0;
    for (int i = 0; i < 32; i++) {
        double err = fabs((double)output[i] - (double)ref[i]);
        double rel_err = err / fmax(fabs((double)ref[i]), 1e-8);
        if (err > max_err) max_err = err;
        if (rel_err > max_rel_err) max_rel_err = rel_err;
    }

    printf("  max_error=%.6e, max_rel_error=%.6e\n", max_err, max_rel_err);
    if (max_rel_err < 1e-5) {
        printf("  PASSED\n\n");
    } else {
        printf("  FAILED\n\n");
    }
}

static void test_tile_attention(void) {
    printf("Testing flash_attention_tile_avx2...\n");

    /* Allocate */
    float *Q = alloc_aligned(FA_BR * FA_D);
    float *K = alloc_aligned(FA_BC * FA_D);
    float *V = alloc_aligned(FA_BC * FA_D);
    float *O_avx2 = alloc_aligned(FA_BR * FA_D);
    float *O_ref = alloc_aligned(FA_BR * FA_D);
    float *S = alloc_aligned(FA_BR * FA_BC);
    float m[FA_BR], l[FA_BR];

    /* Initialize */
    init_random(Q, FA_BR * FA_D, 12345);
    init_random(K, FA_BC * FA_D, 67890);
    init_random(V, FA_BC * FA_D, 11111);
    memset(O_avx2, 0, FA_BR * FA_D * sizeof(float));
    memset(O_ref, 0, FA_BR * FA_D * sizeof(float));

    /* Scale inputs for numerical stability */
    float scale = 1.0f / sqrtf((float)FA_D);
    for (size_t i = 0; i < FA_BR * FA_D; i++) Q[i] *= scale;

    /* Compute */
    flash_attention_tile_avx2(Q, K, V, O_avx2, S, m, l);
    flash_attention_ref(Q, K, V, O_ref);

    /* Check error */
    double max_err = compute_max_error(O_ref, O_avx2, FA_BR * FA_D);
    double rel_err = compute_relative_error(O_ref, O_avx2, FA_BR * FA_D);

    printf("  max_error=%.6e, rel_error=%.6e\n", max_err, rel_err);
    if (rel_err < 1e-4) {
        printf("  PASSED\n\n");
    } else {
        printf("  FAILED\n\n");

        /* Debug: print first few values */
        printf("  First 8 values:\n");
        printf("  ref:  ");
        for (int i = 0; i < 8; i++) printf("%.4f ", O_ref[i]);
        printf("\n  avx2: ");
        for (int i = 0; i < 8; i++) printf("%.4f ", O_avx2[i]);
        printf("\n\n");
    }

    free(Q); free(K); free(V); free(O_avx2); free(O_ref); free(S);
}

/*=============================================================================
 * Main benchmark
 *============================================================================*/

int main(void) {
    printf("================================================================================\n");
    printf("            Flash Attention Benchmark for AMD Zen2 (AVX2 + FMA3)\n");
    printf("================================================================================\n\n");

    /* Pin to CPU 0 */
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(0, &cpuset);
    if (sched_setaffinity(0, sizeof(cpuset), &cpuset) != 0) {
        printf("Warning: Could not pin to CPU 0\n");
    }

    /* Detect CPU frequency */
    printf("Detecting CPU frequency...\n");
    double freq_ghz = detect_cpu_freq_ghz();
    printf("  TSC frequency: %.2f GHz\n", freq_ghz);
    double peak_gflops = 32.0 * freq_ghz;  /* Zen2: 32 FLOPs/cycle */
    printf("  Theoretical peak: %.1f GFLOPS\n\n", peak_gflops);

    /* Run correctness tests */
    printf("================================================================================\n");
    printf("Correctness Tests\n");
    printf("================================================================================\n\n");

    test_exp_avx2();
    test_tile_attention();

    /* Benchmark tile attention */
    printf("================================================================================\n");
    printf("Performance Benchmarks\n");
    printf("================================================================================\n\n");

    printf("Tile parameters: BR=%d, BC=%d, D=%d\n", FA_BR, FA_BC, FA_D);
    printf("FLOPs per tile: %.0f\n\n", 2.0 * FA_BR * FA_BC * FA_D * 2 + 8.0 * FA_BR * FA_BC);

    /* Allocate for benchmarks */
    float *Q = alloc_aligned(FA_BR * FA_D);
    float *K = alloc_aligned(FA_BC * FA_D);
    float *V = alloc_aligned(FA_BC * FA_D);
    float *O = alloc_aligned(FA_BR * FA_D);
    float *S = alloc_aligned(FA_BR * FA_BC);
    float m[FA_BR], l[FA_BR];

    init_random(Q, FA_BR * FA_D, 12345);
    init_random(K, FA_BC * FA_D, 67890);
    init_random(V, FA_BC * FA_D, 11111);

    float scale = 1.0f / sqrtf((float)FA_D);
    for (size_t i = 0; i < FA_BR * FA_D; i++) Q[i] *= scale;

    /* Benchmark single tile */
    printf("Single Tile (BR=%d, BC=%d, D=%d):\n", FA_BR, FA_BC, FA_D);
    printf("--------------------------------------------------------------------------------\n");

    bench_result_t result = benchmark_tile_attention(
        flash_attention_tile_avx2, Q, K, V, O, S, m, l, 100, 1000
    );

    printf("  Time: %.3f us\n", result.time_sec * 1e6);
    printf("  GFLOPS: %.2f (%.1f%% of peak)\n",
           result.gflops, result.gflops / peak_gflops * 100.0);
    printf("  Cycles: %lu\n\n", result.cycles);

    free(Q); free(K); free(V); free(O); free(S);

    /* Benchmark different sequence lengths */
    printf("Variable Sequence Length (BR=%d, D=%d):\n", FA_BR, FA_D);
    printf("--------------------------------------------------------------------------------\n");
    printf("%-12s  %-12s  %-12s  %-12s\n", "Seq Len", "Time (ms)", "GFLOPS", "Efficiency");
    printf("--------------------------------------------------------------------------------\n");

    size_t seq_lens[] = {64, 128, 256, 512, 1024, 2048, 4096};
    int num_lens = sizeof(seq_lens) / sizeof(seq_lens[0]);

    for (int i = 0; i < num_lens; i++) {
        size_t seq_len = seq_lens[i];

        float *Q_seq = alloc_aligned(FA_BR * FA_D);
        float *K_seq = alloc_aligned(seq_len * FA_D);
        float *V_seq = alloc_aligned(seq_len * FA_D);
        float *O_seq = alloc_aligned(FA_BR * FA_D);

        init_random(Q_seq, FA_BR * FA_D, 12345);
        init_random(K_seq, seq_len * FA_D, 67890);
        init_random(V_seq, seq_len * FA_D, 11111);

        for (size_t j = 0; j < FA_BR * FA_D; j++) Q_seq[j] *= scale;

        int iters = (seq_len <= 512) ? 100 : (seq_len <= 2048) ? 20 : 5;

        bench_result_t res = benchmark_seq_attention(
            Q_seq, K_seq, V_seq, O_seq, seq_len, 3, iters
        );

        double efficiency = res.gflops / peak_gflops * 100.0;

        printf("%-12zu  %-12.3f  %-12.2f  %-11.1f%%\n",
               seq_len, res.time_sec * 1000.0, res.gflops, efficiency);

        free(Q_seq); free(K_seq); free(V_seq); free(O_seq);
    }

    printf("\n");
    printf("================================================================================\n");
    printf("Notes:\n");
    printf("  - Flash Attention uses online softmax for memory efficiency\n");
    printf("  - Tile size optimized for Zen2 L1/L2 cache (BR=4, BC=64, D=64)\n");
    printf("  - AVX2 vectorized exp using polynomial approximation\n");
    printf("================================================================================\n");

    return 0;
}
