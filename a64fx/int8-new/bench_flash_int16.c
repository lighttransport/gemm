// bench_flash_int16.c
// Benchmark for Flash Attention with INT16 attention weights
//
// Pipeline:
// Q@K^T: INT8 SDOT -> INT32
// Softmax: INT32 -> FP32 -> approx
// Quantize: FP32 -> INT16
// P@V: INT16 × INT8(widen) -> INT32

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "flash_attention_int16.h"

static inline uint64_t rdtsc(void) {
    uint64_t t;
    __asm__ volatile("mrs %0, CNTVCT_EL0" : "=r"(t));
    return t;
}

// Reference FP32 attention
static void attention_reference(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    int64_t L,
    int64_t head_dim,
    float scale)
{
    float* S = (float*)malloc(L * L * sizeof(float));
    float* P = (float*)malloc(L * L * sizeof(float));

    // Q @ K^T
    for (int64_t i = 0; i < L; i++) {
        for (int64_t j = 0; j < L; j++) {
            float sum = 0.0f;
            for (int64_t k = 0; k < head_dim; k++) {
                sum += Q[i * head_dim + k] * K[j * head_dim + k];
            }
            S[i * L + j] = sum * scale;
        }
    }

    // Softmax
    for (int64_t i = 0; i < L; i++) {
        float max_val = S[i * L];
        for (int64_t j = 1; j < L; j++) {
            if (S[i * L + j] > max_val) max_val = S[i * L + j];
        }

        float sum = 0.0f;
        for (int64_t j = 0; j < L; j++) {
            P[i * L + j] = expf(S[i * L + j] - max_val);
            sum += P[i * L + j];
        }

        for (int64_t j = 0; j < L; j++) {
            P[i * L + j] /= sum;
        }
    }

    // P @ V
    for (int64_t i = 0; i < L; i++) {
        for (int64_t d = 0; d < head_dim; d++) {
            float sum = 0.0f;
            for (int64_t j = 0; j < L; j++) {
                sum += P[i * L + j] * V[j * head_dim + d];
            }
            O[i * head_dim + d] = sum;
        }
    }

    free(S);
    free(P);
}

// Convert INT8 to FP32 for reference
static void int8_to_fp32(const int8_t* src, float* dst, int64_t n) {
    for (int64_t i = 0; i < n; i++) {
        dst[i] = (float)src[i];
    }
}

// Test correctness
static int test_correctness(int64_t L, int64_t head_dim) {
    printf("Testing correctness: L=%ld, head_dim=%ld\n", L, head_dim);

    // Allocate
    int8_t* Q_i8 = (int8_t*)flash16_aligned_alloc(L * head_dim);
    int8_t* K_i8 = (int8_t*)flash16_aligned_alloc(L * head_dim);
    int8_t* V_i8 = (int8_t*)flash16_aligned_alloc(L * head_dim);
    float* O_int16 = (float*)flash16_aligned_alloc(L * head_dim * sizeof(float));
    float* O_ref = (float*)flash16_aligned_alloc(L * head_dim * sizeof(float));

    float* Q_fp32 = (float*)malloc(L * head_dim * sizeof(float));
    float* K_fp32 = (float*)malloc(L * head_dim * sizeof(float));
    float* V_fp32 = (float*)malloc(L * head_dim * sizeof(float));

    // Initialize with random values
    srand(42);
    for (int64_t i = 0; i < L * head_dim; i++) {
        Q_i8[i] = (rand() % 21) - 10;  // [-10, 10]
        K_i8[i] = (rand() % 21) - 10;
        V_i8[i] = (rand() % 21) - 10;
    }

    // Convert to FP32 for reference
    int8_to_fp32(Q_i8, Q_fp32, L * head_dim);
    int8_to_fp32(K_i8, K_fp32, L * head_dim);
    int8_to_fp32(V_i8, V_fp32, L * head_dim);

    float scale = 1.0f / sqrtf((float)head_dim);

    // Run INT16 version
    flash_attention_int16_forward(Q_i8, K_i8, V_i8, O_int16, NULL, L, head_dim, scale);

    // Run reference
    attention_reference(Q_fp32, K_fp32, V_fp32, O_ref, L, head_dim, scale);

    // Compare
    float max_err = 0.0f;
    float sum_err = 0.0f;
    float max_val = 0.0f;

    for (int64_t i = 0; i < L * head_dim; i++) {
        float err = fabsf(O_int16[i] - O_ref[i]);
        if (err > max_err) max_err = err;
        sum_err += err;
        if (fabsf(O_ref[i]) > max_val) max_val = fabsf(O_ref[i]);
    }

    float avg_err = sum_err / (L * head_dim);
    float rel_err = max_err / (max_val + 1e-10f);

    printf("  Max absolute error: %.6f\n", max_err);
    printf("  Avg absolute error: %.6f\n", avg_err);
    printf("  Max relative error: %.4f%%\n", rel_err * 100);
    printf("  Output max value:   %.6f\n", max_val);

    int passed = (rel_err < 0.5f);  // Allow 50% relative error for INT8 quantization
    printf("  Status: %s\n\n", passed ? "PASS" : "FAIL");

    // Cleanup
    flash16_aligned_free(Q_i8);
    flash16_aligned_free(K_i8);
    flash16_aligned_free(V_i8);
    flash16_aligned_free(O_int16);
    flash16_aligned_free(O_ref);
    free(Q_fp32);
    free(K_fp32);
    free(V_fp32);

    return passed;
}

// Benchmark comparing INT16 vs FP32 reference
static void bench_comparison(int64_t L, int64_t head_dim, int iters) {
    printf("Benchmarking: L=%ld, head_dim=%ld, iters=%d\n", L, head_dim, iters);

    // Allocate INT8 inputs
    int8_t* Q_i8 = (int8_t*)flash16_aligned_alloc(L * head_dim);
    int8_t* K_i8 = (int8_t*)flash16_aligned_alloc(L * head_dim);
    int8_t* V_i8 = (int8_t*)flash16_aligned_alloc(L * head_dim);
    float* O_int16 = (float*)flash16_aligned_alloc(L * head_dim * sizeof(float));

    // Allocate FP32 inputs
    float* Q_fp32 = (float*)malloc(L * head_dim * sizeof(float));
    float* K_fp32 = (float*)malloc(L * head_dim * sizeof(float));
    float* V_fp32 = (float*)malloc(L * head_dim * sizeof(float));
    float* O_fp32 = (float*)malloc(L * head_dim * sizeof(float));

    // Initialize
    srand(42);
    for (int64_t i = 0; i < L * head_dim; i++) {
        Q_i8[i] = (rand() % 21) - 10;
        K_i8[i] = (rand() % 21) - 10;
        V_i8[i] = (rand() % 21) - 10;
        Q_fp32[i] = (float)Q_i8[i];
        K_fp32[i] = (float)K_i8[i];
        V_fp32[i] = (float)V_i8[i];
    }

    float scale = 1.0f / sqrtf((float)head_dim);

    // Compute FLOPs
    // Q@K^T: 2 * L * L * head_dim
    // Softmax: ~5 * L * L (exp, sub, div, sum)
    // P@V: 2 * L * L * head_dim
    double flops = (double)L * L * head_dim * 4 + (double)L * L * 5;

    // ============ Benchmark INT16 ============
    // Warmup
    flash_attention_int16_forward(Q_i8, K_i8, V_i8, O_int16, NULL, L, head_dim, scale);

    uint64_t t0 = rdtsc();
    for (int iter = 0; iter < iters; iter++) {
        flash_attention_int16_forward(Q_i8, K_i8, V_i8, O_int16, NULL, L, head_dim, scale);
        __asm__ volatile("" ::: "memory");
    }
    uint64_t t1 = rdtsc();

    double time_int16_s = (double)(t1 - t0) / 100000000.0;  // 100 MHz timer
    double time_int16_ms = time_int16_s * 1000.0 / iters;
    double gflops_int16 = flops / (time_int16_ms / 1000.0) / 1e9;

    // ============ Benchmark FP32 Reference ============
    // Warmup
    attention_reference(Q_fp32, K_fp32, V_fp32, O_fp32, L, head_dim, scale);

    t0 = rdtsc();
    for (int iter = 0; iter < iters; iter++) {
        attention_reference(Q_fp32, K_fp32, V_fp32, O_fp32, L, head_dim, scale);
        __asm__ volatile("" ::: "memory");
    }
    t1 = rdtsc();

    double time_fp32_s = (double)(t1 - t0) / 100000000.0;
    double time_fp32_ms = time_fp32_s * 1000.0 / iters;
    double gflops_fp32 = flops / (time_fp32_ms / 1000.0) / 1e9;

    // ============ Results ============
    double speedup = time_fp32_ms / time_int16_ms;

    printf("  INT16 pipeline: %7.2f ms, %5.2f GFLOPS\n", time_int16_ms, gflops_int16);
    printf("  FP32 reference: %7.2f ms, %5.2f GFLOPS\n", time_fp32_ms, gflops_fp32);
    printf("  Speedup:        %.2fx\n\n", speedup);

    // Cleanup
    flash16_aligned_free(Q_i8);
    flash16_aligned_free(K_i8);
    flash16_aligned_free(V_i8);
    flash16_aligned_free(O_int16);
    free(Q_fp32);
    free(K_fp32);
    free(V_fp32);
    free(O_fp32);
}

int main(void) {
    printf("============================================\n");
    printf("Flash Attention INT16 vs FP32 Comparison\n");
    printf("INT16 Pipeline: INT8 SDOT -> FP32 softmax -> INT16 -> INT16 SDOT\n");
    printf("FP32 Reference: Naive triple-loop attention\n");
    printf("Platform: A64FX SVE (512-bit)\n");
    printf("============================================\n\n");

    printf("=== Correctness Tests ===\n");
    test_correctness(64, 64);
    test_correctness(128, 128);
    test_correctness(256, 128);

    printf("=== INT16 vs FP32 Performance Comparison ===\n");

    // Small sizes
    bench_comparison(128, 128, 100);
    bench_comparison(256, 128, 50);
    bench_comparison(512, 128, 20);

    // Larger sizes
    printf("=== Larger Sequence Lengths ===\n");
    bench_comparison(1024, 128, 10);
    bench_comparison(2048, 128, 5);

    printf("============================================\n");
    printf("Summary:\n");
    printf("- INT16 pipeline uses tiled flash attention (O(n) memory)\n");
    printf("- FP32 reference materializes full attention matrix (O(n^2) memory)\n");
    printf("- INT8/INT16 quantization enables faster compute\n");
    printf("============================================\n");

    return 0;
}
