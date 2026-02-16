/*
 * FP8 GEMM with Pre-Conversion - Testing amortized conversion cost
 *
 * Approach: Convert entire A matrix once, then run pure FP32 GEMM
 * This amortizes conversion cost across multiple iterations
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <arm_sve.h>

#define ALIGN 64
#define CPU_FREQ_MHZ 2000

typedef uint8_t fp8_e4m3_t;
uint32_t fp8_to_fp32_lut[256] __attribute__((aligned(64)));

static inline uint64_t get_ticks(void) {
    uint64_t val;
    __asm__ volatile("isb" ::: "memory");
    __asm__ volatile("mrs %0, cntvct_el0" : "=r"(val));
    __asm__ volatile("isb" ::: "memory");
    return val;
}

void init_lut(void) {
    for (int i = 0; i < 256; i++) {
        uint8_t sign = (i >> 7) & 1;
        uint8_t exp = (i >> 3) & 0xF;
        uint8_t mant = i & 0x7;
        uint32_t fp32;
        if (exp == 0) {
            if (mant == 0) fp32 = (uint32_t)sign << 31;
            else {
                int shift = 0;
                uint8_t m = mant;
                while ((m & 0x4) == 0) { m <<= 1; shift++; }
                fp32 = ((uint32_t)sign << 31) | ((127 - 6 - shift) << 23) | ((uint32_t)(m & 3) << 21);
            }
        } else if (exp == 15 && mant == 7) {
            fp32 = ((uint32_t)sign << 31) | 0x7FC00000;
        } else {
            fp32 = ((uint32_t)sign << 31) | ((exp + 120) << 23) | ((uint32_t)mant << 20);
        }
        fp8_to_fp32_lut[i] = fp32;
    }
}

// Convert entire A matrix from FP8 to FP32 (row-major to packed format)
// Output: A_fp32[panel][k][mr] for direct kernel access
void convert_A_full(const fp8_e4m3_t* A, float* A_fp32,
                    int64_t M, int64_t K, int64_t lda, int64_t MR) {
    int64_t M_panels = M / MR;
    for (int64_t p = 0; p < M_panels; p++) {
        float* dst = A_fp32 + p * MR * K;
        for (int64_t k = 0; k < K; k++) {
            for (int64_t m = 0; m < MR; m++) {
                uint32_t bits = fp8_to_fp32_lut[A[(p*MR+m)*lda + k]];
                dst[k * MR + m] = *((float*)&bits);
            }
        }
    }
}

// Single panel conversion
void convert_A_panel(const fp8_e4m3_t* A, float* A_fp32,
                     int64_t panel, int64_t K, int64_t lda, int64_t MR) {
    for (int64_t k = 0; k < K; k++) {
        for (int64_t m = 0; m < MR; m++) {
            uint32_t bits = fp8_to_fp32_lut[A[(panel*MR+m)*lda + k]];
            A_fp32[k * MR + m] = *((float*)&bits);
        }
    }
}

void convert_fp8_to_fp32_sve(const fp8_e4m3_t* src, float* dst, int64_t count) {
    int64_t vl = svcntw();
    for (int64_t i = 0; i < count; i += vl) {
        svbool_t pg = svwhilelt_b32(i, count);
        svuint32_t indices = svld1ub_u32(pg, src + i);
        svuint32_t fp32_bits = svld1_gather_index(pg, fp8_to_fp32_lut, indices);
        svst1(pg, dst + i, svreinterpret_f32(fp32_bits));
    }
}

extern void micro_kernel_fp32_8x3(const float* A, const float* B, float* C,
                                   int64_t K, int64_t unused, int64_t ldc);

int main() {
    printf("=== FP8 GEMM Pre-Conversion Benchmark ===\n\n");
    init_lut();

    const int64_t MR = 8, NR = 3, VL = 16;
    int64_t M = 384, K = 512;
    int64_t N = NR * VL;  // 48
    int64_t M_panels = M / MR;  // 48
    int64_t ldc_bytes = N * sizeof(float);

    double flops_panel = 2.0 * MR * N * K;  // 393,216
    double flops_gemm = 2.0 * M * N * K;    // 18,874,368
    double peak_gflops = 128.0;  // 2 GHz × 2 FMA pipes × 16 elems × 2 ops

    printf("M=%ld, N=%ld, K=%ld, panels=%ld\n", M, N, K, M_panels);
    printf("FLOPs per panel: %.0f\n", flops_panel);
    printf("FLOPs per GEMM: %.0fM\n\n", flops_gemm / 1e6);

    // Allocate
    fp8_e4m3_t* A_fp8 = aligned_alloc(ALIGN, M * K);
    fp8_e4m3_t* B_fp8 = aligned_alloc(ALIGN, K * N);
    float* B_fp32 = aligned_alloc(ALIGN, K * N * sizeof(float));
    float* A_fp32_panel = aligned_alloc(ALIGN, MR * K * sizeof(float));  // per-panel
    float* A_fp32_full = aligned_alloc(ALIGN, M * K * sizeof(float));     // full pre-converted
    float* C = aligned_alloc(ALIGN, M * N * sizeof(float));

    // Initialize
    srand(42);
    for (int64_t i = 0; i < M * K; i++) {
        A_fp8[i] = ((rand() % 14 + 1) << 3) | (rand() & 7);
    }
    for (int64_t i = 0; i < K * N; i++) {
        B_fp8[i] = ((rand() % 14 + 1) << 3) | (rand() & 7);
    }

    // Pre-convert B
    for (int64_t k = 0; k < K; k++) {
        convert_fp8_to_fp32_sve(B_fp8 + k * N, B_fp32 + k * N, N);
    }

    int iters = 100;
    uint64_t t0, t1, total;

    // ===========================================
    // Test 1: Pure FP32 kernel (baseline)
    // ===========================================
    printf("1. Pure FP32 kernel (1 panel × %d iters):\n", iters);
    convert_A_panel(A_fp8, A_fp32_panel, 0, K, K, MR);
    memset(C, 0, M * N * sizeof(float));

    t0 = get_ticks();
    for (int it = 0; it < iters; it++) {
        micro_kernel_fp32_8x3(A_fp32_panel, B_fp32, C, K, 0, ldc_bytes);
    }
    t1 = get_ticks();

    total = t1 - t0;
    uint64_t ticks_kernel = total / iters;
    uint64_t cycles_kernel = ticks_kernel * 20;  // 100 MHz timer, 2000 MHz CPU
    double gflops_kernel = flops_panel * 2000.0 / (cycles_kernel * 1e3);
    printf("   %lu ticks/call = %lu cycles\n", ticks_kernel, cycles_kernel);
    printf("   %.2f GFLOPS (%.1f%% of peak)\n\n", gflops_kernel, 100.0 * gflops_kernel / peak_gflops);

    // ===========================================
    // Test 2: Conversion cost measurement
    // ===========================================
    printf("2. A conversion cost (full matrix × %d iters):\n", iters);
    t0 = get_ticks();
    for (int it = 0; it < iters; it++) {
        convert_A_full(A_fp8, A_fp32_full, M, K, K, MR);
    }
    t1 = get_ticks();

    total = t1 - t0;
    uint64_t ticks_conv = total / iters;
    uint64_t cycles_conv = ticks_conv * 20;
    double conv_per_elem = (double)cycles_conv / (M * K);
    printf("   %lu ticks = %lu cycles (%.2f cycles/elem)\n\n",
           ticks_conv, cycles_conv, conv_per_elem);

    // ===========================================
    // Test 3: GEMM with per-iteration conversion (current approach)
    // ===========================================
    printf("3. GEMM with per-iter conversion (%d iters):\n", iters);
    memset(C, 0, M * N * sizeof(float));

    t0 = get_ticks();
    for (int it = 0; it < iters; it++) {
        for (int64_t p = 0; p < M_panels; p++) {
            convert_A_panel(A_fp8, A_fp32_panel, p, K, K, MR);
            micro_kernel_fp32_8x3(A_fp32_panel, B_fp32, C + p * MR * N, K, 0, ldc_bytes);
        }
    }
    t1 = get_ticks();

    total = t1 - t0;
    uint64_t ticks_per_gemm = total / iters;
    uint64_t cycles_per_gemm = ticks_per_gemm * 20;
    double gflops_per_iter = flops_gemm * 2000.0 / (cycles_per_gemm * 1e3);
    printf("   %lu ticks/GEMM = %lu cycles\n", ticks_per_gemm, cycles_per_gemm);
    printf("   %.2f GFLOPS (%.1f%% of peak)\n\n", gflops_per_iter, 100.0 * gflops_per_iter / peak_gflops);

    // ===========================================
    // Test 4: GEMM with pre-conversion (amortized)
    // ===========================================
    printf("4. GEMM with pre-conversion (amortized over %d iters):\n", iters);

    // Pre-convert once
    t0 = get_ticks();
    convert_A_full(A_fp8, A_fp32_full, M, K, K, MR);
    t1 = get_ticks();
    uint64_t ticks_preconv = t1 - t0;

    // Run GEMM iterations
    memset(C, 0, M * N * sizeof(float));
    t0 = get_ticks();
    for (int it = 0; it < iters; it++) {
        for (int64_t p = 0; p < M_panels; p++) {
            float* A_panel = A_fp32_full + p * MR * K;
            micro_kernel_fp32_8x3(A_panel, B_fp32, C + p * MR * N, K, 0, ldc_bytes);
        }
    }
    t1 = get_ticks();

    total = t1 - t0;
    uint64_t ticks_gemm_only = total / iters;
    uint64_t ticks_amortized = ticks_gemm_only + ticks_preconv / iters;
    uint64_t cycles_gemm_only = ticks_gemm_only * 20;
    uint64_t cycles_amortized = ticks_amortized * 20;

    double gflops_gemm_only = flops_gemm * 2000.0 / (cycles_gemm_only * 1e3);
    double gflops_amortized = flops_gemm * 2000.0 / (cycles_amortized * 1e3);

    printf("   Pre-conversion: %lu ticks = %lu cycles\n", ticks_preconv, ticks_preconv * 20);
    printf("   Pure GEMM: %lu ticks/iter = %lu cycles\n", ticks_gemm_only, cycles_gemm_only);
    printf("   Amortized: %lu ticks/iter = %lu cycles\n", ticks_amortized, cycles_amortized);
    printf("   Pure GEMM: %.2f GFLOPS (%.1f%% of peak)\n", gflops_gemm_only, 100.0 * gflops_gemm_only / peak_gflops);
    printf("   Amortized: %.2f GFLOPS (%.1f%% of peak)\n\n", gflops_amortized, 100.0 * gflops_amortized / peak_gflops);

    // ===========================================
    // Summary
    // ===========================================
    printf("=== Summary ===\n");
    printf("Kernel efficiency:     %.1f%%\n", 100.0 * gflops_kernel / peak_gflops);
    printf("Per-iter conversion:   %.1f%%\n", 100.0 * gflops_per_iter / peak_gflops);
    printf("Pre-conv (pure GEMM):  %.1f%%\n", 100.0 * gflops_gemm_only / peak_gflops);
    printf("Pre-conv (amortized):  %.1f%%\n", 100.0 * gflops_amortized / peak_gflops);

    printf("\n=== Theoretical Analysis ===\n");
    uint64_t ideal_cycles = (uint64_t)(flops_gemm / 128.0);  // at peak
    printf("Ideal cycles per GEMM: %lu\n", ideal_cycles);
    printf("Kernel cycles (48 panels): %lu\n", 48 * cycles_kernel);
    printf("Gap to 90%%: need %lu cycles, have %lu\n",
           (uint64_t)(flops_gemm / (128.0 * 0.9)), cycles_gemm_only);

    free(A_fp8); free(B_fp8); free(B_fp32);
    free(A_fp32_panel); free(A_fp32_full); free(C);
    return 0;
}
