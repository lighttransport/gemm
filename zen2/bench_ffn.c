/*
 * Benchmark and correctness test for FFN + Activation functions (Zen2 AVX2).
 *
 * Tests:
 *   1. Activation correctness: AVX2 vs scalar reference
 *   2. Activation throughput: Gelem/s, GB/s, cycles/elem
 *   3. FFN correctness: SwiGLU/standard vs naive reference
 *   4. FFN performance: Qwen3-style configs
 *
 * Usage: ./bench_ffn [cpu_freq_ghz]
 */

#include "activation.h"
#include "ffn.h"
#include "gemm.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
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

static void fill_random(float *M, int n)
{
    for (int i = 0; i < n; i++)
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
/*  Scalar reference activations                                       */
/* ------------------------------------------------------------------ */

static float ref_silu(float x)
{
    return x / (1.0f + expf(-x));
}

static float ref_gelu(float x)
{
    float t = 0.7978845608f * (x + 0.044715f * x * x * x);
    return 0.5f * x * (1.0f + tanhf(t));
}

static float ref_relu(float x)
{
    return x > 0.0f ? x : 0.0f;
}

/* ------------------------------------------------------------------ */
/*  Activation correctness tests                                       */
/* ------------------------------------------------------------------ */

typedef void (*act_func_t)(const float *, float *, int);
typedef float (*ref_func_t)(float);

static int test_activation(const char *name, act_func_t avx2_fn,
                           ref_func_t ref_fn, int n, float tol)
{
    float *input  = alloc_matrix(1, n);
    float *output = alloc_matrix(1, n);

    /* Fill with varied values including edge cases */
    srand(42);
    for (int i = 0; i < n; i++)
        input[i] = frand() * 10.0f;  /* range [-5, 5] */

    /* Inject edge cases */
    if (n > 4) {
        input[0] = 0.0f;
        input[1] = 88.0f;
        input[2] = -88.0f;
        input[3] = 1e-7f;
        input[4] = -1e-7f;
    }

    avx2_fn(input, output, n);

    float max_abs_err = 0.0f;
    int worst_idx = 0;
    for (int i = 0; i < n; i++) {
        float ref = ref_fn(input[i]);
        float err = fabsf(output[i] - ref);
        if (err > max_abs_err) {
            max_abs_err = err;
            worst_idx = i;
        }
    }

    int pass = (max_abs_err < tol);
    printf("  %-8s n=%5d  max_abs_err=%.6e at idx=%d (x=%.4f avx2=%.6f ref=%.6f)  %s\n",
           name, n, max_abs_err, worst_idx,
           input[worst_idx], output[worst_idx], ref_fn(input[worst_idx]),
           pass ? "PASS" : "FAIL");

    free(input);
    free(output);
    return pass;
}

static int test_swiglu_correctness(int n, float tol)
{
    float *gate   = alloc_matrix(1, n);
    float *up     = alloc_matrix(1, n);
    float *output = alloc_matrix(1, n);

    srand(42);
    for (int i = 0; i < n; i++) {
        gate[i] = frand() * 10.0f;
        up[i]   = frand() * 10.0f;
    }

    swiglu_avx2(gate, up, output, n);

    float max_abs_err = 0.0f;
    int worst_idx = 0;
    for (int i = 0; i < n; i++) {
        float ref = ref_silu(gate[i]) * up[i];
        float err = fabsf(output[i] - ref);
        if (err > max_abs_err) {
            max_abs_err = err;
            worst_idx = i;
        }
    }

    int pass = (max_abs_err < tol);
    printf("  %-8s n=%5d  max_abs_err=%.6e at idx=%d  %s\n",
           "SwiGLU", n, max_abs_err, worst_idx,
           pass ? "PASS" : "FAIL");

    free(gate);
    free(up);
    free(output);
    return pass;
}

static int test_activation_inplace(const char *name, act_func_t avx2_fn,
                                   ref_func_t ref_fn, int n, float tol)
{
    float *buf = alloc_matrix(1, n);
    float *ref = alloc_matrix(1, n);

    srand(99);
    for (int i = 0; i < n; i++) {
        buf[i] = frand() * 10.0f;
        ref[i] = ref_fn(buf[i]);
    }

    /* In-place: output == input */
    avx2_fn(buf, buf, n);

    float max_abs_err = 0.0f;
    for (int i = 0; i < n; i++) {
        float err = fabsf(buf[i] - ref[i]);
        if (err > max_abs_err)
            max_abs_err = err;
    }

    int pass = (max_abs_err < tol);
    printf("  %-8s n=%5d  in-place max_abs_err=%.6e  %s\n",
           name, n, max_abs_err, pass ? "PASS" : "FAIL");

    free(buf);
    free(ref);
    return pass;
}

/* ------------------------------------------------------------------ */
/*  Activation throughput benchmark                                    */
/* ------------------------------------------------------------------ */

static void bench_activation(const char *name, act_func_t fn, int n,
                             double freq_ghz)
{
    float *input  = alloc_matrix(1, n);
    float *output = alloc_matrix(1, n);

    srand(123);
    fill_random(input, n);

    /* Warm up */
    fn(input, output, n);

    /* Determine iteration count: aim for >= 0.3 s */
    int iters = 1;
    {
        double t0 = get_time();
        fn(input, output, n);
        double dt = get_time() - t0;
        if (dt > 0.0)
            iters = (int)(0.3 / dt) + 1;
        if (iters < 1) iters = 1;
        if (iters > 10000) iters = 10000;
    }

    double best_time = 1e30;
    unsigned long long best_cycles = (unsigned long long)-1;

    for (int it = 0; it < iters; it++) {
        unsigned long long c0 = rdtsc();
        double t0 = get_time();

        fn(input, output, n);

        double t1 = get_time();
        unsigned long long c1 = rdtsc();

        double dt = t1 - t0;
        unsigned long long dc = c1 - c0;
        if (dt < best_time) best_time = dt;
        if (dc < best_cycles) best_cycles = dc;
    }

    double gelem_s = (double)n / best_time / 1e9;
    double gb_s = (double)n * 2 * sizeof(float) / best_time / 1e9;  /* read+write */
    double cyc_elem = (double)best_cycles / (double)n;

    printf("  %-8s n=%7d  %6.2f Gelem/s  %6.1f GB/s  %5.2f cyc/elem  (%d iters)\n",
           name, n, gelem_s, gb_s, cyc_elem, iters);

    free(input);
    free(output);
}

static void bench_swiglu_throughput(int n, double freq_ghz)
{
    float *gate   = alloc_matrix(1, n);
    float *up     = alloc_matrix(1, n);
    float *output = alloc_matrix(1, n);

    srand(123);
    fill_random(gate, n);
    fill_random(up, n);

    swiglu_avx2(gate, up, output, n);

    int iters = 1;
    {
        double t0 = get_time();
        swiglu_avx2(gate, up, output, n);
        double dt = get_time() - t0;
        if (dt > 0.0)
            iters = (int)(0.3 / dt) + 1;
        if (iters < 1) iters = 1;
        if (iters > 10000) iters = 10000;
    }

    double best_time = 1e30;
    unsigned long long best_cycles = (unsigned long long)-1;

    for (int it = 0; it < iters; it++) {
        unsigned long long c0 = rdtsc();
        double t0 = get_time();

        swiglu_avx2(gate, up, output, n);

        double t1 = get_time();
        unsigned long long c1 = rdtsc();

        double dt = t1 - t0;
        unsigned long long dc = c1 - c0;
        if (dt < best_time) best_time = dt;
        if (dc < best_cycles) best_cycles = dc;
    }

    double gelem_s = (double)n / best_time / 1e9;
    /* SwiGLU: 2 reads (gate+up) + 1 write = 12 bytes/elem */
    double gb_s = (double)n * 3 * sizeof(float) / best_time / 1e9;
    double cyc_elem = (double)best_cycles / (double)n;

    printf("  %-8s n=%7d  %6.2f Gelem/s  %6.1f GB/s  %5.2f cyc/elem  (%d iters)\n",
           "SwiGLU", n, gelem_s, gb_s, cyc_elem, iters);

    free(gate);
    free(up);
    free(output);
}

/* ------------------------------------------------------------------ */
/*  FFN correctness: naive reference                                   */
/* ------------------------------------------------------------------ */

/* Naive GEMM: C[M,N] = A[M,K] * B[N,K]^T (B row-major as N×K) */
static void naive_gemm(const float *A, const float *B, float *C,
                       int M, int N, int K)
{
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++)
                sum += A[i * K + k] * B[j * K + k];
            C[i * N + j] = sum;
        }
}

static void naive_swiglu_ffn(const float *X, const float *W_gate,
                             const float *W_up, const float *W_down,
                             float *output, int M, int D, int D_ff)
{
    float *gate = (float *)malloc((size_t)M * D_ff * sizeof(float));
    float *up   = (float *)malloc((size_t)M * D_ff * sizeof(float));

    /* gate = X @ W_gate^T */
    naive_gemm(X, W_gate, gate, M, D_ff, D);
    /* up = X @ W_up^T */
    naive_gemm(X, W_up, up, M, D_ff, D);

    /* SwiGLU: gate[i] = SiLU(gate[i]) * up[i] */
    for (int i = 0; i < M * D_ff; i++) {
        float g = gate[i];
        gate[i] = (g / (1.0f + expf(-g))) * up[i];
    }

    /* output = gate @ W_down^T */
    naive_gemm(gate, W_down, output, M, D, D_ff);

    free(gate);
    free(up);
}

static void naive_standard_ffn(const float *X, const float *W1,
                               const float *W2, float *output,
                               int M, int D, int D_ff,
                               ref_func_t act_fn)
{
    float *buf = (float *)malloc((size_t)M * D_ff * sizeof(float));

    naive_gemm(X, W1, buf, M, D_ff, D);

    for (int i = 0; i < M * D_ff; i++)
        buf[i] = act_fn(buf[i]);

    naive_gemm(buf, W2, output, M, D, D_ff);

    free(buf);
}

static int test_ffn_swiglu(int M, int D, int D_ff, float tol)
{
    printf("  SwiGLU FFN: M=%d, D=%d, D_ff=%d\n", M, D, D_ff);

    float *X      = alloc_matrix(M, D);
    float *W_gate = alloc_matrix(D_ff, D);
    float *W_up   = alloc_matrix(D_ff, D);
    float *W_down = alloc_matrix(D, D_ff);
    float *out_ref = alloc_matrix(M, D);
    float *out_avx = alloc_matrix(M, D);

    srand(42);
    fill_random(X,      M * D);
    fill_random(W_gate, D_ff * D);
    fill_random(W_up,   D_ff * D);
    fill_random(W_down, D * D_ff);

    naive_swiglu_ffn(X, W_gate, W_up, W_down, out_ref, M, D, D_ff);
    ffn_swiglu_fp32(X, W_gate, W_up, W_down, out_avx, M, D, D_ff);

    float max_abs_err = 0.0f;
    int worst_i = 0, worst_j = 0;
    for (int i = 0; i < M; i++)
        for (int j = 0; j < D; j++) {
            float err = fabsf(out_ref[i * D + j] - out_avx[i * D + j]);
            if (err > max_abs_err) {
                max_abs_err = err;
                worst_i = i;
                worst_j = j;
            }
        }

    int pass = (max_abs_err < tol);
    printf("    max_abs_err=%.6e at (%d,%d)  ref=%.6f avx2=%.6f  %s\n",
           max_abs_err, worst_i, worst_j,
           out_ref[worst_i * D + worst_j],
           out_avx[worst_i * D + worst_j],
           pass ? "PASS" : "FAIL");

    free(X); free(W_gate); free(W_up); free(W_down);
    free(out_ref); free(out_avx);
    return pass;
}

static int test_ffn_standard(int M, int D, int D_ff, float tol)
{
    printf("  Standard FFN (GELU): M=%d, D=%d, D_ff=%d\n", M, D, D_ff);

    float *X   = alloc_matrix(M, D);
    float *W1  = alloc_matrix(D_ff, D);
    float *W2  = alloc_matrix(D, D_ff);
    float *out_ref = alloc_matrix(M, D);
    float *out_avx = alloc_matrix(M, D);

    srand(42);
    fill_random(X,  M * D);
    fill_random(W1, D_ff * D);
    fill_random(W2, D * D_ff);

    naive_standard_ffn(X, W1, W2, out_ref, M, D, D_ff, ref_gelu);
    ffn_standard_fp32(X, W1, W2, out_avx, M, D, D_ff, FFN_ACT_GELU);

    float max_abs_err = 0.0f;
    int worst_i = 0, worst_j = 0;
    for (int i = 0; i < M; i++)
        for (int j = 0; j < D; j++) {
            float err = fabsf(out_ref[i * D + j] - out_avx[i * D + j]);
            if (err > max_abs_err) {
                max_abs_err = err;
                worst_i = i;
                worst_j = j;
            }
        }

    int pass = (max_abs_err < tol);
    printf("    max_abs_err=%.6e at (%d,%d)  ref=%.6f avx2=%.6f  %s\n",
           max_abs_err, worst_i, worst_j,
           out_ref[worst_i * D + worst_j],
           out_avx[worst_i * D + worst_j],
           pass ? "PASS" : "FAIL");

    free(X); free(W1); free(W2); free(out_ref); free(out_avx);
    return pass;
}

/* ------------------------------------------------------------------ */
/*  FFN performance benchmark                                          */
/* ------------------------------------------------------------------ */

static void bench_ffn_swiglu(int M, int D, int D_ff, double freq_ghz,
                             const char *label)
{
    double peak_gflops = 32.0 * freq_ghz;  /* 2 FMA × 8-wide × 2 units */
    double total_flops = 6.0 * (double)M * (double)D * (double)D_ff;

    float *X      = alloc_matrix(M, D);
    float *W_gate = alloc_matrix(D_ff, D);
    float *W_up   = alloc_matrix(D_ff, D);
    float *W_down = alloc_matrix(D, D_ff);
    float *output = alloc_matrix(M, D);

    srand(123);
    fill_random(X,      M * D);
    fill_random(W_gate, D_ff * D);
    fill_random(W_up,   D_ff * D);
    fill_random(W_down, D * D_ff);

    /* Warm up */
    ffn_swiglu_fp32(X, W_gate, W_up, W_down, output, M, D, D_ff);

    /* Determine iteration count: aim for >= 0.5 s */
    int iters = 1;
    {
        double t0 = get_time();
        ffn_swiglu_fp32(X, W_gate, W_up, W_down, output, M, D, D_ff);
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

        ffn_swiglu_fp32(X, W_gate, W_up, W_down, output, M, D, D_ff);

        double t1 = get_time();
        unsigned long long c1 = rdtsc();

        double dt = t1 - t0;
        unsigned long long dc = c1 - c0;
        if (dt < best_time) best_time = dt;
        if (dc < best_cycles) best_cycles = dc;
    }

    double gflops = total_flops / best_time / 1e9;
    double flops_per_cycle = total_flops / (double)best_cycles;
    double pct = gflops / peak_gflops * 100.0;

    printf("  %-12s M=%4d D=%5d D_ff=%5d  %7.2f GFLOPS  %5.1f%%  %5.2f FLOPS/cyc  %.4f s  (%d iters)\n",
           label, M, D, D_ff, gflops, pct, flops_per_cycle, best_time, iters);

    free(X); free(W_gate); free(W_up); free(W_down); free(output);
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

    printf("=== Zen2 AVX2 FFN + Activation Benchmark ===\n");
    printf("CPU freq: %.2f GHz, peak: %.1f GFLOPS (FP32)\n\n", freq_ghz, peak_gflops);

    /* ============================================================= */
    /*  Activation Correctness Tests                                  */
    /* ============================================================= */

    printf("--- Activation Correctness Tests ---\n");
    int all_pass = 1;

    /* Test with various sizes including non-multiple-of-8.
     * Tolerances reflect fast_exp (~1e-6 rel) + rcp+NR pipeline:
     *   SiLU/GELU: ~2e-4 max abs, SwiGLU: ~1e-3 (product amplifies). */
    int test_sizes[] = {1031, 1024, 65536};
    for (int t = 0; t < 3; t++) {
        int n = test_sizes[t];
        all_pass &= test_activation("SiLU", silu_avx2, ref_silu, n, 2e-4f);
        all_pass &= test_activation("GELU", gelu_avx2, ref_gelu, n, 2e-4f);
        all_pass &= test_activation("ReLU", relu_avx2, ref_relu, n, 1e-7f);
        all_pass &= test_swiglu_correctness(n, 1e-3f);
    }

    /* In-place tests */
    printf("\n--- In-place Activation Tests ---\n");
    all_pass &= test_activation_inplace("SiLU", silu_avx2, ref_silu, 1031, 2e-4f);
    all_pass &= test_activation_inplace("GELU", gelu_avx2, ref_gelu, 1031, 2e-4f);
    all_pass &= test_activation_inplace("ReLU", relu_avx2, ref_relu, 1031, 1e-7f);
    printf("\n");

    if (!all_pass) {
        printf("ACTIVATION CORRECTNESS FAILURE — skipping remaining tests.\n");
        return 1;
    }

    /* ============================================================= */
    /*  Activation Throughput Benchmarks                               */
    /* ============================================================= */

    printf("--- Activation Throughput ---\n");
    int bench_sizes[] = {1024, 65536, 1048576};
    for (int t = 0; t < 3; t++) {
        int n = bench_sizes[t];
        bench_activation("SiLU", silu_avx2, n, freq_ghz);
        bench_activation("GELU", gelu_avx2, n, freq_ghz);
        bench_activation("ReLU", relu_avx2, n, freq_ghz);
        bench_swiglu_throughput(n, freq_ghz);
        if (t < 2) printf("\n");
    }
    printf("\n");

    /* ============================================================= */
    /*  FFN Correctness Tests                                         */
    /* ============================================================= */

    printf("--- FFN Correctness Tests ---\n");
    /* Use small sizes so naive reference is fast */
    all_pass &= test_ffn_swiglu(16, 64, 128, 1e-3f);
    all_pass &= test_ffn_swiglu(32, 128, 256, 1e-2f);
    all_pass &= test_ffn_standard(16, 64, 128, 1e-3f);
    printf("\n");

    if (!all_pass) {
        printf("FFN CORRECTNESS FAILURE — skipping performance benchmarks.\n");
        ffn_cleanup();
        gemm_cleanup();
        return 1;
    }

    /* ============================================================= */
    /*  FFN Performance Benchmarks (SwiGLU)                           */
    /* ============================================================= */

    printf("--- FFN Performance (SwiGLU) ---\n");
    printf("%-14s %-6s %-7s %-7s  %10s  %6s  %12s  %8s\n",
           "Config", "M", "D", "D_ff", "GFLOPS", "Peak%", "FLOPS/cycle", "Time(s)");
    printf("---------------------------------------------------------------------------------\n");

    /* Small test config */
    bench_ffn_swiglu(1,   256,  1024,  freq_ghz, "Small");
    bench_ffn_swiglu(32,  256,  1024,  freq_ghz, "Small");
    bench_ffn_swiglu(128, 256,  1024,  freq_ghz, "Small");

    printf("\n");

    /* Qwen3-0.5B: D=896, D_ff=4864 */
    bench_ffn_swiglu(1,   896,  4864,  freq_ghz, "Qwen3-0.5B");
    bench_ffn_swiglu(32,  896,  4864,  freq_ghz, "Qwen3-0.5B");
    bench_ffn_swiglu(128, 896,  4864,  freq_ghz, "Qwen3-0.5B");

    printf("\n");

    /* Qwen3-1.5B: D=1536, D_ff=8960 */
    bench_ffn_swiglu(1,   1536, 8960,  freq_ghz, "Qwen3-1.5B");
    bench_ffn_swiglu(32,  1536, 8960,  freq_ghz, "Qwen3-1.5B");

    printf("\n");

    ffn_cleanup();
    gemm_cleanup();
    return 0;
}
