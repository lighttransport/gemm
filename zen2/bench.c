#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <x86intrin.h>
#include <sys/mman.h>
#include "gemm.h"
#include "gemm_pack.h"

static double get_time(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static inline uint64_t rdtsc(void)
{
    unsigned int lo, hi;
    __asm__ volatile("rdtsc" : "=a"(lo), "=d"(hi));
    return ((uint64_t)hi << 32) | lo;
}

static float frand(void)
{
    return (float)rand() / (float)RAND_MAX - 0.5f;
}

/*
 * Naive GEMM for correctness reference.
 * C[M×N] = A[M×K] × B[N×K]^T  (B stored as N×K row-major)
 */
static void gemm_naive(const float *A, int lda,
                       const float *B, int ldb,
                       float *C, int ldc,
                       int M, int N, int K)
{
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * lda + k] * B[j * ldb + k];
            }
            C[i * ldc + j] = sum;
        }
    }
}

/*
 * Microkernel-only benchmark: runs the 6×16 kernel on L1-resident data.
 * Measures both GFLOPS (wall-clock) and FLOPS/cycle (rdtsc).
 */
static void bench_microkernel_k(int K, double peak_gflops)
{
    float *A_pack = NULL, *B_pack = NULL;
    float C[MR * NR];

    /* Allocate with 64 extra bytes of padding to handle software-pipelined
     * B preloads that read slightly beyond the buffer on the last iteration */
    posix_memalign((void **)&A_pack, 64, (size_t)MR * K * sizeof(float) + 64);
    posix_memalign((void **)&B_pack, 64, (size_t)NR * K * sizeof(float) + 64);

    for (int i = 0; i < MR * K; i++) A_pack[i] = frand();
    for (int i = 0; i < NR * K; i++) B_pack[i] = frand();
    memset(C, 0, sizeof(C));

    int64_t ldc_bytes = NR * sizeof(float);
    int iters = 2000000 / (K / 64 + 1);
    if (iters < 100000) iters = 100000;

    /* Warm up */
    for (int t = 0; t < 1000; t++) {
        gemm_kernel_6x16(A_pack, B_pack, C, (int64_t)K, ldc_bytes);
    }

    /* Timed run with both wall-clock and rdtsc */
    double t0 = get_time();
    uint64_t c0 = rdtsc();
    for (int t = 0; t < iters; t++) {
        gemm_kernel_6x16(A_pack, B_pack, C, (int64_t)K, ldc_bytes);
    }
    uint64_t c1 = rdtsc();
    double t1 = get_time();

    double elapsed = t1 - t0;
    uint64_t cycles = c1 - c0;
    double total_flops = 2.0 * MR * NR * K * (double)iters;
    double gflops = total_flops / elapsed / 1e9;
    double flops_per_cycle = total_flops / (double)cycles;
    double pct = gflops / peak_gflops * 100.0;
    double actual_ghz = (double)cycles / elapsed / 1e9;

    printf("  K=%3d:  %7.2f GFLOPS  (%5.1f%% peak)  %.1f FLOPS/cyc (of 32)  "
           "[actual %.2f GHz]\n",
           K, gflops, pct, flops_per_cycle, actual_ghz);

    free(A_pack);
    free(B_pack);
}

static void bench_microkernel(double peak_gflops)
{
    int kvals[] = { 64, 128, 256 };
    int nk = sizeof(kvals) / sizeof(kvals[0]);
    for (int i = 0; i < nk; i++) {
        bench_microkernel_k(kvals[i], peak_gflops);
    }
}

/*
 * Correctness test: compare optimized GEMM against naive for small sizes.
 */
static int test_correctness(void)
{
    const int M = 64, N = 64, K = 128;
    float *A = NULL, *B = NULL, *C = NULL, *C_ref = NULL;

    posix_memalign((void **)&A, 64, (size_t)M * K * sizeof(float));
    posix_memalign((void **)&B, 64, (size_t)N * K * sizeof(float));
    posix_memalign((void **)&C, 64, (size_t)M * N * sizeof(float));
    posix_memalign((void **)&C_ref, 64, (size_t)M * N * sizeof(float));

    srand(42);
    for (int i = 0; i < M * K; i++) A[i] = frand();
    for (int i = 0; i < N * K; i++) B[i] = frand();

    gemm_naive(A, K, B, K, C_ref, N, M, N, K);
    gemm_fp32(A, K, B, K, C, N, M, N, K);

    float max_err = 0.0f;
    float max_rel = 0.0f;
    for (int i = 0; i < M * N; i++) {
        float err = fabsf(C[i] - C_ref[i]);
        float rel = (fabsf(C_ref[i]) > 1e-6f) ? err / fabsf(C_ref[i]) : err;
        if (err > max_err) max_err = err;
        if (rel > max_rel) max_rel = rel;
    }

    int pass = max_rel < 1e-4f;
    printf("Correctness (M=%d, N=%d, K=%d): max_abs_err=%.2e, max_rel_err=%.2e  [%s]\n",
           M, N, K, max_err, max_rel, pass ? "PASS" : "FAIL");

    free(A);
    free(B);
    free(C);
    free(C_ref);

    return pass;
}

/*
 * Full GEMM benchmark for a given size.
 */
static void bench_gemm(int M, int N, int K, double peak_gflops)
{
    float *A = NULL, *B = NULL, *C = NULL;

    size_t A_size = (size_t)M * K * sizeof(float);
    size_t B_size = (size_t)N * K * sizeof(float);
    size_t C_size = (size_t)M * N * sizeof(float);

    /* Check memory: skip if total > 8 GB */
    size_t total = A_size + B_size + C_size;
    if (total > (size_t)8 * 1024 * 1024 * 1024) {
        printf("M=%6d, N=%6d, K=%4d:  SKIPPED (%.1f GB needed)\n",
               M, N, K, (double)total / (1024.0 * 1024.0 * 1024.0));
        return;
    }

    posix_memalign((void **)&A, 64, A_size);
    posix_memalign((void **)&B, 64, B_size);
    posix_memalign((void **)&C, 64, C_size);

    if (A) madvise(A, A_size, MADV_HUGEPAGE);
    if (B) madvise(B, B_size, MADV_HUGEPAGE);
    if (C) madvise(C, C_size, MADV_HUGEPAGE);

    if (!A || !B || !C) {
        printf("M=%6d, N=%6d, K=%4d:  SKIPPED (allocation failed)\n", M, N, K);
        free(A); free(B); free(C);
        return;
    }

    srand(123);
    for (size_t i = 0; i < (size_t)M * K; i++) A[i] = frand();
    for (size_t i = 0; i < (size_t)N * K; i++) B[i] = frand();

    double flops = 2.0 * M * N * K;

    /* Decide iteration count: aim for >= 0.5 sec */
    int iters = 1;
    if (flops < 1e9)  iters = 20;
    else if (flops < 1e10) iters = 5;
    else if (flops < 1e11) iters = 2;

    /* Warm up */
    gemm_fp32(A, K, B, K, C, N, M, N, K);

    double t0 = get_time();
    uint64_t c0 = rdtsc();
    for (int t = 0; t < iters; t++) {
        gemm_fp32(A, K, B, K, C, N, M, N, K);
    }
    uint64_t c1 = rdtsc();
    double elapsed = get_time() - t0;

    uint64_t cycles = c1 - c0;
    double total_flops = flops * iters;
    double gflops = total_flops / elapsed / 1e9;
    double flops_per_cycle = total_flops / (double)cycles;
    double pct = gflops / peak_gflops * 100.0;
    double actual_ghz = (double)cycles / elapsed / 1e9;

    printf("M=%6d, N=%6d, K=%4d:  %7.2f GFLOPS  (%5.1f%% peak)  "
           "%.1f FLOPS/cyc  [%.2f GHz, %.3f s, %d iters]\n",
           M, N, K, gflops, pct, flops_per_cycle, actual_ghz, elapsed, iters);

    free(A);
    free(B);
    free(C);
}

/*
 * Try to read CPU base frequency from /proc/cpuinfo (MHz).
 * Returns frequency in GHz, or 0 if not found.
 */
static double read_cpu_freq_ghz(void)
{
    FILE *f = fopen("/proc/cpuinfo", "r");
    if (!f) return 0.0;

    char line[256];
    double mhz = 0.0;
    while (fgets(line, sizeof(line), f)) {
        if (strncmp(line, "cpu MHz", 7) == 0) {
            char *p = strchr(line, ':');
            if (p) mhz = atof(p + 1);
            break;
        }
    }
    fclose(f);
    return mhz / 1000.0;
}

int main(int argc, char **argv)
{
    double freq_ghz = 0.0;

    /* Allow user to override frequency: ./bench <freq_ghz> */
    if (argc > 1) {
        freq_ghz = atof(argv[1]);
    }
    if (freq_ghz <= 0.0) {
        freq_ghz = read_cpu_freq_ghz();
    }
    if (freq_ghz <= 0.0) {
        freq_ghz = 3.5;  /* Default: Ryzen 3950X base */
        printf("Warning: could not detect CPU frequency, using %.1f GHz\n", freq_ghz);
    }

    /* Zen2: 2 FMA units × 8 FP32/FMA × 2 (mul+add) = 32 FP32 FLOPS/cycle */
    double peak_gflops = 32.0 * freq_ghz;

    printf("=== Zen2 FP32 GEMM Benchmark ===\n");
    printf("CPU frequency: %.2f GHz\n", freq_ghz);
    printf("Peak FP32: %.1f GFLOPS/core\n", peak_gflops);
    printf("Microkernel: MR=%d, NR=%d\n\n", MR, NR);

    /* 1. Microkernel-only benchmark */
    printf("--- Microkernel benchmark ---\n");
    bench_microkernel(peak_gflops);
    printf("\n");

    /* 2. Correctness test */
    printf("--- Correctness test ---\n");
    if (!test_correctness()) {
        printf("ABORTING: correctness check failed!\n");
        return 1;
    }
    printf("\n");

    /* 3. Full GEMM benchmarks: attention-relevant sizes */
    printf("--- Full GEMM benchmarks ---\n");

    struct { int M, N, K; const char *desc; } sizes[] = {
        {  1024,  1024, 128, "Warm-up" },
        {  4096,  4096, 128, "Medium d=128" },
        {  1024,  1024, 256, "d=256 warm-up" },
        {  4096,  4096, 256, "Medium d=256" },
        { 16384, 16384, 128, "Large d=128" },
        { 16384, 16384, 256, "Large d=256" },
        { 65536, 65536, 128, "Full L=64K attention" },
    };
    int nsizes = sizeof(sizes) / sizeof(sizes[0]);

    for (int i = 0; i < nsizes; i++) {
        bench_gemm(sizes[i].M, sizes[i].N, sizes[i].K, peak_gflops);
    }

    gemm_cleanup();
    printf("\nDone.\n");
    return 0;
}
