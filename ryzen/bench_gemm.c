/*
 * GEMM Benchmark for AMD Zen2 (Ryzen 9 3950X)
 *
 * Measures GFLOPS performance for:
 *   - Optimized AVX2/FMA GEMM kernel
 *   - Naive reference implementation
 *
 * Includes correctness verification against naive implementation.
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>
#include <sched.h>

#include "gemm_avx2.h"

/*=============================================================================
 * Timing utilities
 *============================================================================*/

/* High-resolution timer using clock_gettime */
static inline double get_time_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* Read CPU timestamp counter (TSC) */
static inline uint64_t rdtsc(void) {
    uint32_t lo, hi;
    __asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
    return ((uint64_t)hi << 32) | lo;
}

/* Serialize instructions before/after TSC read */
static inline uint64_t rdtscp(void) {
    uint32_t lo, hi, aux;
    __asm__ __volatile__ ("rdtscp" : "=a"(lo), "=d"(hi), "=c"(aux));
    return ((uint64_t)hi << 32) | lo;
}

/* Memory fence to ensure all loads/stores complete */
static inline void mfence(void) {
    __asm__ __volatile__ ("mfence" ::: "memory");
}

/*=============================================================================
 * CPU frequency detection
 *============================================================================*/

static double detect_cpu_freq_ghz(void) {
    /* Warm up */
    volatile uint64_t start, end;
    start = rdtsc();

    /* Measure TSC over a known time period */
    double t0 = get_time_sec();
    start = rdtscp();
    mfence();

    /* Wait ~100ms */
    while (get_time_sec() - t0 < 0.1) {
        /* busy wait */
    }

    mfence();
    end = rdtscp();
    double t1 = get_time_sec();

    double elapsed = t1 - t0;
    uint64_t cycles = end - start;

    return (double)cycles / elapsed / 1e9;
}

/*=============================================================================
 * Matrix utilities
 *============================================================================*/

static float *alloc_matrix(size_t M, size_t N) {
    float *ptr = NULL;
    if (posix_memalign((void **)&ptr, 32, M * N * sizeof(float)) != 0) {
        return NULL;
    }
    return ptr;
}

/* PCG32 random number generator */
typedef struct {
    uint64_t state;
    uint64_t inc;
} pcg32_random_t;

static inline uint32_t pcg32_random_r(pcg32_random_t *rng) {
    uint64_t oldstate = rng->state;
    rng->state = oldstate * 6364136223846793005ULL + rng->inc;
    uint32_t xorshifted = (uint32_t)(((oldstate >> 18u) ^ oldstate) >> 27u);
    uint32_t rot = (uint32_t)(oldstate >> 59u);
    return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}

static inline float pcg32_float(pcg32_random_t *rng) {
    return (float)pcg32_random_r(rng) / (float)UINT32_MAX;
}

static void init_matrix_random(float *A, size_t M, size_t N, uint64_t seed) {
    pcg32_random_t rng = { seed, seed ^ 0xdeadbeefULL };
    for (size_t i = 0; i < M * N; i++) {
        A[i] = pcg32_float(&rng) * 2.0f - 1.0f;  /* [-1, 1] */
    }
}

static void init_matrix_zero(float *A, size_t M, size_t N) {
    memset(A, 0, M * N * sizeof(float));
}

/*=============================================================================
 * Correctness verification
 *============================================================================*/

static double compute_max_error(const float *C_ref, const float *C_test,
                                 size_t M, size_t N, size_t ldc) {
    double max_err = 0.0;
    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j++) {
            double ref = C_ref[i * ldc + j];
            double test = C_test[i * ldc + j];
            double err = fabs(ref - test);
            if (err > max_err) {
                max_err = err;
            }
        }
    }
    return max_err;
}

static double compute_relative_error(const float *C_ref, const float *C_test,
                                       size_t M, size_t N, size_t ldc) {
    double max_rel_err = 0.0;
    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j++) {
            double ref = C_ref[i * ldc + j];
            double test = C_test[i * ldc + j];
            double err = fabs(ref - test);
            double denom = fmax(fabs(ref), 1e-8);
            double rel_err = err / denom;
            if (rel_err > max_rel_err) {
                max_rel_err = rel_err;
            }
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
    double cycles_per_fma;
} bench_result_t;

static bench_result_t benchmark_gemm(
    void (*gemm_func)(size_t, size_t, size_t, float,
                      const float *, size_t,
                      const float *, size_t,
                      float, float *, size_t),
    size_t M, size_t N, size_t K,
    const float *A, const float *B, float *C,
    int warmup_iters, int bench_iters
) {
    bench_result_t result = {0};

    /* Warm up */
    for (int i = 0; i < warmup_iters; i++) {
        gemm_func(M, N, K, 1.0f, A, K, B, N, 0.0f, C, N);
    }

    /* Benchmark */
    mfence();
    uint64_t start_cycles = rdtscp();
    double start_time = get_time_sec();

    for (int i = 0; i < bench_iters; i++) {
        gemm_func(M, N, K, 1.0f, A, K, B, N, 0.0f, C, N);
    }

    mfence();
    double end_time = get_time_sec();
    uint64_t end_cycles = rdtscp();

    result.time_sec = (end_time - start_time) / bench_iters;
    result.cycles = (end_cycles - start_cycles) / bench_iters;

    /* Calculate GFLOPS: 2*M*N*K FLOPs (multiply-add) */
    double flops = 2.0 * M * N * K;
    result.gflops = flops / result.time_sec / 1e9;

    /* Calculate cycles per FMA (each FMA is 2 FLOPs) */
    double fma_ops = (double)M * N * K;  /* One FMA per multiply-add */
    result.cycles_per_fma = (double)result.cycles / fma_ops;

    return result;
}

/*=============================================================================
 * Print utilities
 *============================================================================*/

static void print_header(void) {
    printf("================================================================================\n");
    printf("                    AMD Zen2 GEMM Benchmark (AVX2 + FMA3)\n");
    printf("================================================================================\n\n");
}

static void print_cpu_info(double freq_ghz) {
    printf("CPU Information:\n");
    printf("  Detected TSC frequency: %.2f GHz\n", freq_ghz);

    /* Calculate theoretical peak GFLOPS for Zen2 */
    /* Zen2: 2x 256-bit FMA units per core, 8 FP32 per 256-bit, 2 FLOPs per FMA */
    /* Peak = 2 * 8 * 2 * freq = 32 * freq GFLOPS per core */
    double peak_gflops_per_core = 32.0 * freq_ghz;
    printf("  Theoretical peak (1 core): %.1f GFLOPS\n", peak_gflops_per_core);
    printf("\n");
}

static void print_result(const char *name, size_t M, size_t N, size_t K,
                          bench_result_t result, double peak_gflops) {
    double efficiency = result.gflops / peak_gflops * 100.0;
    printf("%-20s  M=%4zu  N=%4zu  K=%4zu  "
           "%.3f ms  %7.2f GFLOPS  (%5.1f%%)  %.3f cycles/FMA\n",
           name, M, N, K,
           result.time_sec * 1000.0,
           result.gflops,
           efficiency,
           result.cycles_per_fma);
}

/*=============================================================================
 * Main benchmark
 *============================================================================*/

int main(int argc, char **argv) {
    /* Default matrix sizes */
    size_t sizes[][3] = {
        {64, 64, 64},
        {128, 128, 128},
        {256, 256, 256},
        {512, 512, 512},
        {1024, 1024, 1024},
        {2048, 2048, 2048},
        {4096, 4096, 4096},
    };
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    /* Parse command line for custom size */
    size_t custom_M = 0, custom_N = 0, custom_K = 0;
    int use_custom = 0;
    if (argc == 4) {
        custom_M = (size_t)atol(argv[1]);
        custom_N = (size_t)atol(argv[2]);
        custom_K = (size_t)atol(argv[3]);
        use_custom = 1;
    } else if (argc == 2) {
        size_t s = (size_t)atol(argv[1]);
        custom_M = custom_N = custom_K = s;
        use_custom = 1;
    }

    print_header();

    /* Try to pin to a single core for consistent results */
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(0, &cpuset);
    if (sched_setaffinity(0, sizeof(cpuset), &cpuset) != 0) {
        printf("Warning: Could not pin to CPU 0\n");
    }

    /* Detect CPU frequency */
    printf("Detecting CPU frequency...\n");
    double freq_ghz = detect_cpu_freq_ghz();
    print_cpu_info(freq_ghz);
    double peak_gflops = 32.0 * freq_ghz;  /* Zen2: 32 FLOPs/cycle per core */

    /* Benchmark parameters */
    int warmup_iters = 3;
    int bench_iters = 10;

    if (use_custom) {
        /* Single custom size */
        size_t M = custom_M, N = custom_N, K = custom_K;

        printf("Custom size: M=%zu, N=%zu, K=%zu\n\n", M, N, K);

        float *A = alloc_matrix(M, K);
        float *B = alloc_matrix(K, N);
        float *C_avx2 = alloc_matrix(M, N);
        float *C_naive = alloc_matrix(M, N);

        if (!A || !B || !C_avx2 || !C_naive) {
            fprintf(stderr, "Memory allocation failed\n");
            return 1;
        }

        init_matrix_random(A, M, K, 12345);
        init_matrix_random(B, K, N, 67890);
        init_matrix_zero(C_avx2, M, N);
        init_matrix_zero(C_naive, M, N);

        /* Run naive for reference */
        printf("Running naive reference (may be slow for large sizes)...\n");
        sgemm_naive(M, N, K, 1.0f, A, K, B, N, 0.0f, C_naive, N);

        /* Verify correctness */
        sgemm_avx2(M, N, K, 1.0f, A, K, B, N, 0.0f, C_avx2, N);
        double max_err = compute_max_error(C_naive, C_avx2, M, N, N);
        double rel_err = compute_relative_error(C_naive, C_avx2, M, N, N);
        printf("Correctness check: max_error=%.6e, rel_error=%.6e\n", max_err, rel_err);
        if (rel_err < 1e-4) {
            printf("PASSED\n\n");
        } else {
            printf("FAILED (relative error too high)\n\n");
        }

        /* Benchmark AVX2 */
        printf("Benchmark results:\n");
        printf("--------------------------------------------------------------------------------\n");
        bench_result_t result = benchmark_gemm(sgemm_avx2, M, N, K, A, B, C_avx2,
                                                warmup_iters, bench_iters);
        print_result("sgemm_avx2", M, N, K, result, peak_gflops);

        /* Benchmark naive for small sizes */
        if (M * N * K <= 512 * 512 * 512) {
            result = benchmark_gemm(sgemm_naive, M, N, K, A, B, C_naive,
                                    warmup_iters, bench_iters);
            print_result("sgemm_naive", M, N, K, result, peak_gflops);
        }

        free(A);
        free(B);
        free(C_avx2);
        free(C_naive);
    } else {
        /* Run all predefined sizes */
        printf("Benchmark results:\n");
        printf("--------------------------------------------------------------------------------\n");
        printf("%-20s  %-6s  %-6s  %-6s  %-10s  %-14s  %-8s  %s\n",
               "Kernel", "M", "N", "K", "Time", "GFLOPS", "Eff", "Cyc/FMA");
        printf("--------------------------------------------------------------------------------\n");

        /* Find max size for allocation */
        size_t max_dim = 0;
        for (int i = 0; i < num_sizes; i++) {
            if (sizes[i][0] > max_dim) max_dim = sizes[i][0];
            if (sizes[i][1] > max_dim) max_dim = sizes[i][1];
            if (sizes[i][2] > max_dim) max_dim = sizes[i][2];
        }

        float *A = alloc_matrix(max_dim, max_dim);
        float *B = alloc_matrix(max_dim, max_dim);
        float *C_avx2 = alloc_matrix(max_dim, max_dim);
        float *C_naive = alloc_matrix(max_dim, max_dim);

        if (!A || !B || !C_avx2 || !C_naive) {
            fprintf(stderr, "Memory allocation failed\n");
            return 1;
        }

        init_matrix_random(A, max_dim, max_dim, 12345);
        init_matrix_random(B, max_dim, max_dim, 67890);

        for (int i = 0; i < num_sizes; i++) {
            size_t M = sizes[i][0];
            size_t N = sizes[i][1];
            size_t K = sizes[i][2];

            init_matrix_zero(C_avx2, M, N);
            init_matrix_zero(C_naive, M, N);

            /* Verify correctness for smaller sizes */
            if (M * N * K <= 256 * 256 * 256) {
                sgemm_naive(M, N, K, 1.0f, A, K, B, N, 0.0f, C_naive, N);
                sgemm_avx2(M, N, K, 1.0f, A, K, B, N, 0.0f, C_avx2, N);
                double rel_err = compute_relative_error(C_naive, C_avx2, M, N, N);
                if (rel_err >= 1e-4) {
                    printf("WARNING: Correctness check failed for M=%zu N=%zu K=%zu\n",
                           M, N, K);
                }
            }

            /* Adjust iterations based on size */
            int iters = bench_iters;
            if (M >= 2048) iters = 3;
            if (M >= 4096) iters = 1;

            bench_result_t result = benchmark_gemm(sgemm_avx2, M, N, K, A, B, C_avx2,
                                                    warmup_iters, iters);
            print_result("sgemm_avx2", M, N, K, result, peak_gflops);
        }

        free(A);
        free(B);
        free(C_avx2);
        free(C_naive);
    }

    printf("\n");
    printf("================================================================================\n");
    printf("Notes:\n");
    printf("  - Efficiency = actual GFLOPS / theoretical peak (32 * freq)\n");
    printf("  - Zen2 has 2x 256-bit FMA units, each can do 8 FP32 FMAs/cycle\n");
    printf("  - Cycles/FMA: lower is better (ideal = 1/(2*8) = 0.0625 for full utilization)\n");
    printf("================================================================================\n");

    return 0;
}
