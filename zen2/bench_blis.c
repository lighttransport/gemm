/*
 * BLIS SGEMM benchmark for Zen2
 *
 * Computes C = alpha*A*B + beta*C using bli_sgemm (typed API).
 * A is MxK, B is KxN, C is MxN.
 * Column-major storage (rs=1, cs=M for A and C; rs=1, cs=K for B).
 * BLIS_NO_TRANSPOSE for both A and B, alpha=1, beta=0.
 *
 * Peak: 32 FLOPS/cycle * 3.5 GHz = 112 GFLOPS (SP)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "blis.h"

static double get_time_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void run_sgemm_bench(dim_t M, dim_t N, dim_t K, int nreps)
{
    /* Column-major: row stride = 1, col stride = #rows */
    inc_t rs_a = 1, cs_a = M;
    inc_t rs_b = 1, cs_b = K;
    inc_t rs_c = 1, cs_c = M;

    float alpha = 1.0f;
    float beta  = 0.0f;

    float *a = (float *)aligned_alloc(64, (size_t)M * K * sizeof(float));
    float *b = (float *)aligned_alloc(64, (size_t)K * N * sizeof(float));
    float *c = (float *)aligned_alloc(64, (size_t)M * N * sizeof(float));

    if (!a || !b || !c) {
        fprintf(stderr, "Allocation failed for M=%ld N=%ld K=%ld\n",
                (long)M, (long)N, (long)K);
        free(a); free(b); free(c);
        return;
    }

    /* Initialize with small random values */
    for (long i = 0; i < (long)M * K; i++) a[i] = (float)(rand() % 100) / 100.0f;
    for (long i = 0; i < (long)K * N; i++) b[i] = (float)(rand() % 100) / 100.0f;
    memset(c, 0, (size_t)M * N * sizeof(float));

    /* Warmup */
    bli_sgemm(BLIS_NO_TRANSPOSE, BLIS_NO_TRANSPOSE,
              M, N, K,
              &alpha, a, rs_a, cs_a,
                      b, rs_b, cs_b,
              &beta,  c, rs_c, cs_c);

    /* Timed runs */
    double best = 1e30;
    for (int rep = 0; rep < nreps; rep++) {
        double t0 = get_time_sec();
        bli_sgemm(BLIS_NO_TRANSPOSE, BLIS_NO_TRANSPOSE,
                  M, N, K,
                  &alpha, a, rs_a, cs_a,
                          b, rs_b, cs_b,
                  &beta,  c, rs_c, cs_c);
        double t1 = get_time_sec();
        double elapsed = t1 - t0;
        if (elapsed < best) best = elapsed;
    }

    double flops = 2.0 * (double)M * (double)N * (double)K;
    double gflops = flops / best / 1e9;
    double peak = 112.0; /* 32 FLOPS/cycle * 3.5 GHz */
    double pct = gflops / peak * 100.0;

    printf("M=%5ld  N=%5ld  K=%5ld  |  best_time = %.6f s  |  %.2f GFLOPS  |  %.1f%% of %.0f GFLOPS peak\n",
           (long)M, (long)N, (long)K, best, gflops, pct, peak);

    free(a);
    free(b);
    free(c);
}

int main(void)
{
    printf("BLIS SGEMM Benchmark (single-threaded, Zen2)\n");
    printf("Peak = 32 FLOPS/cycle * 3.5 GHz = 112 GFLOPS (SP)\n");
    printf("=====================================================\n\n");

    bli_init();

    int nreps = 10;

    run_sgemm_bench(4096, 4096, 256, nreps);
    run_sgemm_bench(4096, 4096, 128, nreps);
    run_sgemm_bench(16384, 16384, 256, nreps);

    bli_finalize();

    return 0;
}
