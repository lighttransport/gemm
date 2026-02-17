/*
 * Benchmark and correctness test for FlashAttention-1 (Zen2 AVX2).
 *
 * Tests:
 *   1. Correctness: compare against naive O(L^2) attention
 *   2. Performance: GFLOPS, FLOPS/cycle for various (L, d)
 *
 * Usage: ./bench_flash [cpu_freq_ghz]
 */

#include "flash_attention.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <time.h>
#include <sys/mman.h>

/* ------------------------------------------------------------------ */
/*  Utilities                                                          */
/* ------------------------------------------------------------------ */

static double get_time(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static inline unsigned long long rdtsc(void)
{
    unsigned lo, hi;
    __asm__ __volatile__("rdtsc" : "=a"(lo), "=d"(hi));
    return ((unsigned long long)hi << 32) | lo;
}

static float frand(void)
{
    return (float)rand() / (float)RAND_MAX - 0.5f;
}

static float *alloc_matrix(int rows, int cols)
{
    size_t bytes = (size_t)rows * cols * sizeof(float);
    float *p = NULL;
    posix_memalign((void **)&p, 64, bytes);
    madvise(p, bytes, MADV_HUGEPAGE);
    return p;
}

static void fill_random(float *M, int rows, int cols)
{
    for (int i = 0; i < rows * cols; i++)
        M[i] = frand();
}

static double read_cpu_freq_ghz(void)
{
    FILE *f = fopen("/proc/cpuinfo", "r");
    if (!f) return 3.5;
    char line[256];
    while (fgets(line, sizeof(line), f)) {
        double mhz;
        if (sscanf(line, "cpu MHz : %lf", &mhz) == 1) {
            fclose(f);
            return mhz / 1000.0;
        }
    }
    fclose(f);
    return 3.5;
}

/* ------------------------------------------------------------------ */
/*  Naive O(L^2) attention (reference)                                 */
/* ------------------------------------------------------------------ */

static void naive_attention(
    const float *Q, const float *K, const float *V,
    float *O, int L, int d, float scale)
{
    /* S = Q × K^T × scale */
    float *S = (float *)malloc((size_t)L * L * sizeof(float));

    for (int i = 0; i < L; i++)
        for (int j = 0; j < L; j++) {
            float dot = 0.0f;
            for (int k = 0; k < d; k++)
                dot += Q[i * d + k] * K[j * d + k];
            S[i * L + j] = dot * scale;
        }

    /* Row-wise softmax */
    for (int i = 0; i < L; i++) {
        float max_val = -FLT_MAX;
        for (int j = 0; j < L; j++)
            if (S[i * L + j] > max_val)
                max_val = S[i * L + j];

        float sum = 0.0f;
        for (int j = 0; j < L; j++) {
            S[i * L + j] = expf(S[i * L + j] - max_val);
            sum += S[i * L + j];
        }
        float inv = 1.0f / sum;
        for (int j = 0; j < L; j++)
            S[i * L + j] *= inv;
    }

    /* O = S × V */
    for (int i = 0; i < L; i++)
        for (int j = 0; j < d; j++) {
            float sum = 0.0f;
            for (int k = 0; k < L; k++)
                sum += S[i * L + k] * V[k * d + j];
            O[i * d + j] = sum;
        }

    free(S);
}

/* ------------------------------------------------------------------ */
/*  Correctness test                                                   */
/* ------------------------------------------------------------------ */

static int test_correctness(int L, int d)
{
    printf("Correctness test: L=%d, d=%d\n", L, d);

    size_t s_bytes = (size_t)L * L * sizeof(float);

    /* Check if naive attention would need too much memory */
    if (s_bytes > (size_t)2 * 1024 * 1024 * 1024) {
        printf("  Skipped (naive needs %.1f GB for S matrix)\n",
               (double)s_bytes / (1024.0 * 1024 * 1024));
        return 1;
    }

    float *Q     = alloc_matrix(L, d);
    float *K     = alloc_matrix(L, d);
    float *V     = alloc_matrix(L, d);
    float *O_ref = alloc_matrix(L, d);
    float *O_fla = alloc_matrix(L, d);

    srand(42);
    fill_random(Q, L, d);
    fill_random(K, L, d);
    fill_random(V, L, d);

    float scale = 1.0f / sqrtf((float)d);

    naive_attention(Q, K, V, O_ref, L, d, scale);
    flash_attention_fp32(Q, K, V, O_fla, L, d, scale);

    /* Compare: use combined abs/rel metric.
     * Attention outputs span wide magnitudes; near-zero values
     * can inflate pure relative error. */
    float max_abs_err = 0.0f;
    float max_rel_err = 0.0f;
    int worst_abs_i = 0, worst_abs_j = 0;
    int worst_rel_i = 0, worst_rel_j = 0;

    for (int i = 0; i < L; i++)
        for (int j = 0; j < d; j++) {
            float ref = O_ref[i * d + j];
            float fla = O_fla[i * d + j];
            float abs_err = fabsf(ref - fla);
            float denom = fmaxf(fabsf(ref), fmaxf(fabsf(fla), 1e-5f));
            float rel_err = abs_err / denom;
            if (abs_err > max_abs_err) {
                max_abs_err = abs_err;
                worst_abs_i = i; worst_abs_j = j;
            }
            if (rel_err > max_rel_err) {
                max_rel_err = rel_err;
                worst_rel_i = i; worst_rel_j = j;
            }
        }

    int pass = (max_abs_err < 1e-3f);
    printf("  max_abs_err = %.6e at (%d,%d)  ref=%.6f flash=%.6f\n",
           max_abs_err, worst_abs_i, worst_abs_j,
           O_ref[worst_abs_i * d + worst_abs_j],
           O_fla[worst_abs_i * d + worst_abs_j]);
    printf("  max_rel_err = %.6e at (%d,%d)  ref=%.6f flash=%.6f\n",
           max_rel_err, worst_rel_i, worst_rel_j,
           O_ref[worst_rel_i * d + worst_rel_j],
           O_fla[worst_rel_i * d + worst_rel_j]);
    printf("  %s\n\n", pass ? "PASS" : "FAIL");

    free(Q); free(K); free(V); free(O_ref); free(O_fla);
    return pass;
}

/* ------------------------------------------------------------------ */
/*  Performance benchmark                                              */
/* ------------------------------------------------------------------ */

static void bench_flash(int L, int d, double freq_ghz)
{
    double peak_gflops = 32.0 * freq_ghz;  /* 2 FMA × 8-wide × 2 units */
    double total_flops = 4.0 * (double)L * (double)L * (double)d;
    size_t mat_bytes = (size_t)L * d * sizeof(float);

    /* Skip if matrices alone exceed 4 GB */
    if (4 * mat_bytes > (size_t)4 * 1024 * 1024 * 1024) {
        printf("L=%5d d=%3d  skipped (matrices exceed 4 GB)\n", L, d);
        return;
    }

    float *Q = alloc_matrix(L, d);
    float *K = alloc_matrix(L, d);
    float *V = alloc_matrix(L, d);
    float *O = alloc_matrix(L, d);

    srand(123);
    fill_random(Q, L, d);
    fill_random(K, L, d);
    fill_random(V, L, d);

    float scale = 1.0f / sqrtf((float)d);

    /* Warm-up */
    flash_attention_fp32(Q, K, V, O, L, d, scale);

    /* Determine iteration count: aim for >= 0.5 s */
    int iters = 1;
    {
        double t0 = get_time();
        flash_attention_fp32(Q, K, V, O, L, d, scale);
        double dt = get_time() - t0;
        if (dt > 0.0)
            iters = (int)(0.5 / dt) + 1;
        if (iters < 1) iters = 1;
        if (iters > 100) iters = 100;
    }

    double best_time = 1e30;
    unsigned long long best_cycles = (unsigned long long)-1;

    for (int it = 0; it < iters; it++) {
        unsigned long long c0 = rdtsc();
        double t0 = get_time();

        flash_attention_fp32(Q, K, V, O, L, d, scale);

        double t1 = get_time();
        unsigned long long c1 = rdtsc();

        double dt = t1 - t0;
        unsigned long long dc = c1 - c0;
        if (dt < best_time) best_time = dt;
        if (dc < best_cycles) best_cycles = dc;
    }

    double gflops = total_flops / best_time / 1e9;
    double flops_per_cycle = total_flops / (double)best_cycles;
    double actual_ghz = (double)best_cycles / best_time / 1e9;
    double pct = gflops / peak_gflops * 100.0;

    printf("L=%5d d=%3d  %7.1f GFLOPS  %5.1f%%  %5.2f FLOPS/cyc  "
           "%.2f GHz  %.4f s  (%d iters)\n",
           L, d, gflops, pct, flops_per_cycle, actual_ghz,
           best_time, iters);

    free(Q); free(K); free(V); free(O);
}

/* ------------------------------------------------------------------ */
/*  Main                                                               */
/* ------------------------------------------------------------------ */

int main(int argc, char **argv)
{
    double freq_ghz = read_cpu_freq_ghz();
    if (argc > 1)
        freq_ghz = atof(argv[1]);

    double peak_gflops = 32.0 * freq_ghz;

    printf("=== Zen2 AVX2 FlashAttention-1 Benchmark ===\n");
    printf("CPU freq: %.2f GHz, peak: %.1f GFLOPS (FP32)\n\n", freq_ghz, peak_gflops);

    /* --- Correctness tests --- */
    printf("--- Correctness Tests ---\n");
    int all_pass = 1;
    all_pass &= test_correctness(128, 64);
    all_pass &= test_correctness(256, 128);
    all_pass &= test_correctness(512, 128);
    all_pass &= test_correctness(1024, 128);

    if (!all_pass) {
        printf("CORRECTNESS FAILURE — skipping performance benchmarks.\n");
        flash_attention_cleanup();
        return 1;
    }

    /* --- Performance benchmarks --- */
    printf("--- Performance Benchmarks ---\n");
    printf("%-12s %-4s  %10s  %6s  %12s  %6s  %8s\n",
           "L", "d", "GFLOPS", "Peak%", "FLOPS/cycle", "GHz", "Time(s)");
    printf("----------------------------------------------------------------------\n");

    bench_flash(1024, 128, freq_ghz);
    bench_flash(4096, 128, freq_ghz);
    bench_flash(4096, 256, freq_ghz);
    bench_flash(16384, 128, freq_ghz);

    printf("\n");
    flash_attention_cleanup();
    return 0;
}
