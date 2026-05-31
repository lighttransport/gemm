/*
 * Neural Network Operations Benchmark for AMD Zen2 (Ryzen 9 3950X)
 *
 * Benchmarks:
 *   - RMSNorm
 *   - LayerNorm
 *   - Activation functions (GELU, SiLU, QuickGELU, ReLU)
 *   - FFN modules
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>
#include <sched.h>

#include "nn_ops_avx2.h"

/* ============================================
 * Timing utilities
 * ============================================ */

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

/* ============================================
 * Memory utilities
 * ============================================ */

static float *alloc_aligned(size_t n) {
    float *ptr = NULL;
    if (posix_memalign((void **)&ptr, 32, n * sizeof(float)) != 0) {
        return NULL;
    }
    return ptr;
}

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
        A[i] = (pcg32_float(&rng) - 0.5f) * 2.0f;
    }
}

static void init_ones(float *A, size_t n) {
    for (size_t i = 0; i < n; i++) {
        A[i] = 1.0f;
    }
}

/* ============================================
 * Error checking
 * ============================================ */

static double max_rel_error(const float *ref, const float *test, size_t n) {
    double max_err = 0.0;
    for (size_t i = 0; i < n; i++) {
        double r = (double)ref[i];
        double t = (double)test[i];
        double err = fabs(r - t);
        double denom = fmax(fabs(r), 1e-8);
        double rel = err / denom;
        if (rel > max_err) max_err = rel;
    }
    return max_err;
}

/* ============================================
 * Benchmark runner
 * ============================================ */

typedef struct {
    double time_us;
    double bandwidth_gb_s;
    uint64_t cycles;
} bench_result_t;

#define WARMUP_ITERS 10
#define BENCH_ITERS 100

/* ============================================
 * RMSNorm benchmark
 * ============================================ */

static void bench_rmsnorm(size_t dim, double freq_ghz __attribute__((unused))) {
    float *input = alloc_aligned(dim);
    float *gamma = alloc_aligned(dim);
    float *output_avx2 = alloc_aligned(dim);
    float *output_ref = alloc_aligned(dim);

    init_random(input, dim, 12345);
    init_ones(gamma, dim);

    /* Correctness check */
    rmsnorm_f32_avx2(input, gamma, output_avx2, dim, 1e-6f);
    rmsnorm_f32_ref(input, gamma, output_ref, dim, 1e-6f);
    double err = max_rel_error(output_ref, output_avx2, dim);

    /* Warmup */
    for (int i = 0; i < WARMUP_ITERS; i++) {
        rmsnorm_f32_avx2(input, gamma, output_avx2, dim, 1e-6f);
    }

    /* Benchmark */
    mfence();
    uint64_t start = rdtscp();
    double t0 = get_time_sec();

    for (int i = 0; i < BENCH_ITERS; i++) {
        rmsnorm_f32_avx2(input, gamma, output_avx2, dim, 1e-6f);
    }

    mfence();
    double t1 = get_time_sec();
    uint64_t end = rdtscp();

    double time_us = (t1 - t0) / BENCH_ITERS * 1e6;
    uint64_t cycles = (end - start) / BENCH_ITERS;
    /* Memory: read input + gamma, write output = 3 * dim * 4 bytes */
    double bytes = 3.0 * dim * sizeof(float);
    double bandwidth = bytes / ((t1 - t0) / BENCH_ITERS) / 1e9;

    printf("  RMSNorm      dim=%-6zu  %7.2f us  %6.2f GB/s  %8lu cyc  err=%.2e\n",
           dim, time_us, bandwidth, cycles, err);

    free(input); free(gamma); free(output_avx2); free(output_ref);
}

/* ============================================
 * LayerNorm benchmark
 * ============================================ */

static void bench_layernorm(size_t dim, double freq_ghz __attribute__((unused))) {
    float *input = alloc_aligned(dim);
    float *gamma = alloc_aligned(dim);
    float *beta = alloc_aligned(dim);
    float *output_avx2 = alloc_aligned(dim);
    float *output_ref = alloc_aligned(dim);

    init_random(input, dim, 12345);
    init_ones(gamma, dim);
    memset(beta, 0, dim * sizeof(float));

    /* Correctness check */
    layernorm_f32_avx2(input, gamma, beta, output_avx2, dim, 1e-6f);
    layernorm_f32_ref(input, gamma, beta, output_ref, dim, 1e-6f);
    double err = max_rel_error(output_ref, output_avx2, dim);

    /* Warmup */
    for (int i = 0; i < WARMUP_ITERS; i++) {
        layernorm_f32_avx2(input, gamma, beta, output_avx2, dim, 1e-6f);
    }

    /* Benchmark */
    mfence();
    uint64_t start = rdtscp();
    double t0 = get_time_sec();

    for (int i = 0; i < BENCH_ITERS; i++) {
        layernorm_f32_avx2(input, gamma, beta, output_avx2, dim, 1e-6f);
    }

    mfence();
    double t1 = get_time_sec();
    uint64_t end = rdtscp();

    double time_us = (t1 - t0) / BENCH_ITERS * 1e6;
    uint64_t cycles = (end - start) / BENCH_ITERS;
    double bytes = 4.0 * dim * sizeof(float);  /* input, gamma, beta, output */
    double bandwidth = bytes / ((t1 - t0) / BENCH_ITERS) / 1e9;

    printf("  LayerNorm    dim=%-6zu  %7.2f us  %6.2f GB/s  %8lu cyc  err=%.2e\n",
           dim, time_us, bandwidth, cycles, err);

    free(input); free(gamma); free(beta); free(output_avx2); free(output_ref);
}

/* ============================================
 * Activation benchmarks
 * ============================================ */

static void bench_activation(const char *name,
                              void (*act_avx2)(const float*, float*, size_t),
                              void (*act_ref)(const float*, float*, size_t),
                              size_t n) {
    float *input = alloc_aligned(n);
    float *output_avx2 = alloc_aligned(n);
    float *output_ref = alloc_aligned(n);

    init_random(input, n, 12345);

    /* Correctness check */
    act_avx2(input, output_avx2, n);
    if (act_ref) {
        act_ref(input, output_ref, n);
    } else {
        memcpy(output_ref, output_avx2, n * sizeof(float));
    }
    double err = max_rel_error(output_ref, output_avx2, n);

    /* Warmup */
    for (int i = 0; i < WARMUP_ITERS; i++) {
        act_avx2(input, output_avx2, n);
    }

    /* Benchmark */
    mfence();
    uint64_t start = rdtscp();
    double t0 = get_time_sec();

    for (int i = 0; i < BENCH_ITERS; i++) {
        act_avx2(input, output_avx2, n);
    }

    mfence();
    double t1 = get_time_sec();
    uint64_t end = rdtscp();

    double time_us = (t1 - t0) / BENCH_ITERS * 1e6;
    uint64_t cycles = (end - start) / BENCH_ITERS;
    double bytes = 2.0 * n * sizeof(float);
    double bandwidth = bytes / ((t1 - t0) / BENCH_ITERS) / 1e9;

    printf("  %-12s n=%-8zu  %7.2f us  %6.2f GB/s  %8lu cyc  err=%.2e\n",
           name, n, time_us, bandwidth, cycles, err);

    free(input); free(output_avx2); free(output_ref);
}

/* ============================================
 * FFN benchmark
 * ============================================ */

static void bench_ffn(size_t in_dim, size_t hidden_dim, size_t out_dim) {
    size_t batch = 1;

    float *input = alloc_aligned(batch * in_dim);
    float *W1 = alloc_aligned(hidden_dim * in_dim);
    float *b1 = alloc_aligned(hidden_dim);
    float *W2 = alloc_aligned(out_dim * hidden_dim);
    float *b2 = alloc_aligned(out_dim);
    float *output = alloc_aligned(batch * out_dim);
    float *hidden = alloc_aligned(batch * hidden_dim);

    init_random(input, batch * in_dim, 12345);
    init_random(W1, hidden_dim * in_dim, 67890);
    init_random(b1, hidden_dim, 11111);
    init_random(W2, out_dim * hidden_dim, 22222);
    init_random(b2, out_dim, 33333);

    /* Warmup */
    for (int i = 0; i < WARMUP_ITERS; i++) {
        ffn_f32_avx2(input, W1, b1, W2, b2, output, hidden,
                     batch, in_dim, hidden_dim, out_dim, FFN_ACT_GELU);
    }

    /* Benchmark */
    mfence();
    uint64_t start = rdtscp();
    double t0 = get_time_sec();

    for (int i = 0; i < BENCH_ITERS; i++) {
        ffn_f32_avx2(input, W1, b1, W2, b2, output, hidden,
                     batch, in_dim, hidden_dim, out_dim, FFN_ACT_GELU);
    }

    mfence();
    double t1 = get_time_sec();
    uint64_t end = rdtscp();

    double time_us = (t1 - t0) / BENCH_ITERS * 1e6;
    uint64_t cycles = (end - start) / BENCH_ITERS;
    /* FLOPs: 2*in*hidden + 2*hidden*out + activations */
    double flops = 2.0 * in_dim * hidden_dim + 2.0 * hidden_dim * out_dim;
    double gflops = flops / ((t1 - t0) / BENCH_ITERS) / 1e9;

    printf("  FFN(GELU)    %zu->%zu->%zu  %7.2f us  %6.2f GFLOPS  %8lu cyc\n",
           in_dim, hidden_dim, out_dim, time_us, gflops, cycles);

    free(input); free(W1); free(b1); free(W2); free(b2);
    free(output); free(hidden);
}

static void bench_ffn_swiglu(size_t in_dim, size_t hidden_dim, size_t out_dim) {
    size_t batch = 1;

    float *input = alloc_aligned(batch * in_dim);
    float *W_gate = alloc_aligned(hidden_dim * in_dim);
    float *W_up = alloc_aligned(hidden_dim * in_dim);
    float *W_down = alloc_aligned(out_dim * hidden_dim);
    float *output = alloc_aligned(batch * out_dim);
    float *gate = alloc_aligned(batch * hidden_dim);
    float *up = alloc_aligned(batch * hidden_dim);

    init_random(input, batch * in_dim, 12345);
    init_random(W_gate, hidden_dim * in_dim, 67890);
    init_random(W_up, hidden_dim * in_dim, 11111);
    init_random(W_down, out_dim * hidden_dim, 22222);

    /* Warmup */
    for (int i = 0; i < WARMUP_ITERS; i++) {
        ffn_swiglu_f32_avx2(input, W_gate, W_up, W_down, output, gate, up,
                            batch, in_dim, hidden_dim, out_dim);
    }

    /* Benchmark */
    mfence();
    uint64_t start = rdtscp();
    double t0 = get_time_sec();

    for (int i = 0; i < BENCH_ITERS; i++) {
        ffn_swiglu_f32_avx2(input, W_gate, W_up, W_down, output, gate, up,
                            batch, in_dim, hidden_dim, out_dim);
    }

    mfence();
    double t1 = get_time_sec();
    uint64_t end = rdtscp();

    double time_us = (t1 - t0) / BENCH_ITERS * 1e6;
    uint64_t cycles = (end - start) / BENCH_ITERS;
    /* FLOPs: 2*in*hidden (gate) + 2*in*hidden (up) + 2*hidden*out (down) */
    double flops = 4.0 * in_dim * hidden_dim + 2.0 * hidden_dim * out_dim;
    double gflops = flops / ((t1 - t0) / BENCH_ITERS) / 1e9;

    printf("  FFN(SwiGLU)  %zu->%zu->%zu  %7.2f us  %6.2f GFLOPS  %8lu cyc\n",
           in_dim, hidden_dim, out_dim, time_us, gflops, cycles);

    free(input); free(W_gate); free(W_up); free(W_down);
    free(output); free(gate); free(up);
}

/* ============================================
 * Main
 * ============================================ */

int main(void) {
    printf("================================================================================\n");
    printf("         Neural Network Operations Benchmark for AMD Zen2 (AVX2 + FMA3)\n");
    printf("================================================================================\n\n");

    /* Pin to CPU 0 */
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(0, &cpuset);
    sched_setaffinity(0, sizeof(cpuset), &cpuset);

    /* Detect CPU frequency */
    printf("Detecting CPU frequency...\n");
    double freq_ghz = detect_cpu_freq_ghz();
    printf("  TSC frequency: %.2f GHz\n\n", freq_ghz);

    /* ============================================
     * RMSNorm benchmarks
     * ============================================ */
    printf("RMSNorm Benchmarks:\n");
    printf("--------------------------------------------------------------------------------\n");
    bench_rmsnorm(64, freq_ghz);
    bench_rmsnorm(128, freq_ghz);
    bench_rmsnorm(256, freq_ghz);
    bench_rmsnorm(512, freq_ghz);
    bench_rmsnorm(768, freq_ghz);    /* GPT-2 small */
    bench_rmsnorm(1024, freq_ghz);   /* GPT-2 medium */
    bench_rmsnorm(2048, freq_ghz);
    bench_rmsnorm(4096, freq_ghz);   /* LLaMA 7B */
    printf("\n");

    /* ============================================
     * LayerNorm benchmarks
     * ============================================ */
    printf("LayerNorm Benchmarks:\n");
    printf("--------------------------------------------------------------------------------\n");
    bench_layernorm(64, freq_ghz);
    bench_layernorm(128, freq_ghz);
    bench_layernorm(256, freq_ghz);
    bench_layernorm(512, freq_ghz);
    bench_layernorm(768, freq_ghz);
    bench_layernorm(1024, freq_ghz);
    bench_layernorm(2048, freq_ghz);
    bench_layernorm(4096, freq_ghz);
    printf("\n");

    /* ============================================
     * Activation function benchmarks
     * ============================================ */
    printf("Activation Function Benchmarks:\n");
    printf("--------------------------------------------------------------------------------\n");

    size_t act_sizes[] = {1024, 4096, 16384, 65536};
    int num_sizes = sizeof(act_sizes) / sizeof(act_sizes[0]);

    for (int i = 0; i < num_sizes; i++) {
        size_t n = act_sizes[i];
        printf("  Size = %zu:\n", n);
        bench_activation("GELU", gelu_f32_avx2, gelu_f32_ref, n);
        bench_activation("GELU_tanh", gelu_tanh_f32_avx2, NULL, n);
        bench_activation("SiLU", silu_f32_avx2, silu_f32_ref, n);
        bench_activation("QuickGELU", quickgelu_f32_avx2, NULL, n);
        bench_activation("Sigmoid", sigmoid_f32_avx2, NULL, n);
        bench_activation("ReLU", relu_f32_avx2, NULL, n);
        printf("\n");
    }

    /* ============================================
     * FFN benchmarks
     * ============================================ */
    printf("FFN Benchmarks:\n");
    printf("--------------------------------------------------------------------------------\n");

    /* Common configurations */
    bench_ffn(768, 3072, 768);      /* GPT-2 small */
    bench_ffn(1024, 4096, 1024);    /* GPT-2 medium */
    bench_ffn(4096, 11008, 4096);   /* LLaMA 7B style */
    printf("\n");

    bench_ffn_swiglu(768, 2048, 768);
    bench_ffn_swiglu(1024, 2730, 1024);
    bench_ffn_swiglu(4096, 11008, 4096);
    printf("\n");

    printf("================================================================================\n");
    printf("Notes:\n");
    printf("  - All operations are single-precision (FP32)\n");
    printf("  - Bandwidth = (input + output bytes) / time\n");
    printf("  - FFN includes GEMV operations (memory-bound for single batch)\n");
    printf("================================================================================\n");

    return 0;
}
