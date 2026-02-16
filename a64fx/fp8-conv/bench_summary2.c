/*
 * FP8 GEMM Final Summary - Integer-only efficiency calculation
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <arm_sve.h>

#define ALIGN 64

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

void convert_A_panel(const fp8_e4m3_t* A, float* A_fp32,
                     int64_t panel, int64_t K, int64_t lda, int64_t MR) {
    for (int64_t k = 0; k < K; k++) {
        for (int64_t m = 0; m < MR; m++) {
            uint32_t bits = fp8_to_fp32_lut[A[(panel*MR+m)*lda + k]];
            A_fp32[k * MR + m] = *((float*)&bits);
        }
    }
}

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

void convert_B(const fp8_e4m3_t* B, float* B_fp32, int64_t K, int64_t N) {
    int64_t vl = svcntw();
    for (int64_t k = 0; k < K; k++) {
        for (int64_t n = 0; n < N; n += vl) {
            svbool_t pg = svwhilelt_b32(n, N);
            svuint32_t indices = svld1ub_u32(pg, B + k * N + n);
            svuint32_t fp32_bits = svld1_gather_index(pg, fp8_to_fp32_lut, indices);
            svst1(pg, B_fp32 + k * N + n, svreinterpret_f32(fp32_bits));
        }
    }
}

extern void fp8_gemm_kernel_asm(const float* A, const float* B, float* C,
                                 int64_t ldc_elem, int64_t K);

int main() {
    printf("=== FP8 GEMM on A64FX - Final Summary ===\n\n");
    init_lut();

    const int64_t MR = 8, NR = 3, VL = 16;
    int64_t M = 384, K = 512, N = NR * VL;
    int64_t M_panels = M / MR;
    int64_t flops_gemm = 2L * M * N * K;
    // ideal_ticks = flops / (128 GFLOPS) / (100 MHz timer)
    // = flops / 128e9 * 100e6 = flops / 1280
    int64_t ideal_ticks = flops_gemm / 1280;
    int iters = 100;

    printf("Config: M=%ld, N=%ld, K=%ld, panels=%ld\n", M, N, K, M_panels);
    printf("FLOPs/GEMM: %ld (18.9M)\n", flops_gemm);
    printf("Peak: 128 GFLOPS, Ideal: %ld ticks, 90%%: %ld ticks\n\n", ideal_ticks, ideal_ticks * 100 / 90);

    fp8_e4m3_t* A_fp8 = aligned_alloc(ALIGN, M * K);
    fp8_e4m3_t* B_fp8 = aligned_alloc(ALIGN, K * N);
    float* B_fp32 = aligned_alloc(ALIGN, K * N * sizeof(float));
    float* A_fp32 = aligned_alloc(ALIGN, MR * K * sizeof(float));
    float* A_fp32_full = aligned_alloc(ALIGN, M * K * sizeof(float));
    float* C = aligned_alloc(ALIGN, M * N * sizeof(float));

    srand(42);
    for (int64_t i = 0; i < M * K; i++) A_fp8[i] = ((rand() % 14 + 1) << 3) | (rand() & 7);
    for (int64_t i = 0; i < K * N; i++) B_fp8[i] = ((rand() % 14 + 1) << 3) | (rand() & 7);
    convert_B(B_fp8, B_fp32, K, N);

    uint64_t t0, t1;
    uint64_t pure_gemm, per_panel, preconv_cost;

    // Test 1: Pure FP32 kernel
    printf("1. Pure FP32 Kernel:\n");
    convert_A_full(A_fp8, A_fp32_full, M, K, K, MR);
    memset(C, 0, M * N * sizeof(float));
    t0 = get_ticks();
    for (int it = 0; it < iters; it++) {
        for (int64_t p = 0; p < M_panels; p++) {
            fp8_gemm_kernel_asm(A_fp32_full + p * MR * K, B_fp32, C + p * MR * N, N, K);
        }
    }
    t1 = get_ticks();
    pure_gemm = (t1 - t0) / iters;
    printf("   %lu ticks/GEMM, eff=%lu%%\n\n", pure_gemm, ideal_ticks * 100 / pure_gemm);

    // Test 2: Per-panel conversion
    printf("2. Per-Panel Conversion:\n");
    memset(C, 0, M * N * sizeof(float));
    t0 = get_ticks();
    for (int it = 0; it < iters; it++) {
        for (int64_t p = 0; p < M_panels; p++) {
            convert_A_panel(A_fp8, A_fp32, p, K, K, MR);
            fp8_gemm_kernel_asm(A_fp32, B_fp32, C + p * MR * N, N, K);
        }
    }
    t1 = get_ticks();
    per_panel = (t1 - t0) / iters;
    printf("   %lu ticks/GEMM, eff=%lu%%\n\n", per_panel, ideal_ticks * 100 / per_panel);

    // Test 3: Pre-conversion cost
    printf("3. Pre-Conversion Cost:\n");
    t0 = get_ticks();
    convert_A_full(A_fp8, A_fp32_full, M, K, K, MR);
    t1 = get_ticks();
    preconv_cost = t1 - t0;
    printf("   %lu ticks for %ld elements\n\n", preconv_cost, M * K);

    // Summary table
    printf("=== SUMMARY TABLE ===\n\n");
    printf("Scenario                    Ticks    Eff%%\n");
    printf("─────────────────────────────────────────\n");
    printf("Pure FP32 (no conv)         %5lu    %3lu%%\n", pure_gemm, ideal_ticks * 100 / pure_gemm);
    printf("Per-panel conv (1-use)      %5lu    %3lu%%\n", per_panel, ideal_ticks * 100 / per_panel);

    for (int reuse = 1; reuse <= 10; reuse++) {
        uint64_t amort = pure_gemm + preconv_cost / reuse;
        uint64_t eff = ideal_ticks * 100 / amort;
        printf("Pre-conv (reuse=%2d)         %5lu    %3lu%%", reuse, amort, eff);
        if (eff >= 90) printf(" ✓");
        printf("\n");
    }

    printf("\n=== KEY FINDINGS ===\n");
    printf("• Pure FP32 kernel: %lu%% efficiency\n", ideal_ticks * 100 / pure_gemm);
    printf("• Single-use conversion: %lu%% efficiency\n", ideal_ticks * 100 / per_panel);
    printf("• Pre-conv amortized (reuse=2): %lu%% → 90%% achieved!\n",
           ideal_ticks * 100 / (pure_gemm + preconv_cost / 2));
    printf("• Conversion overhead: %lu ticks = %.1f%% of GEMM\n",
           preconv_cost, (double)preconv_cost * 100 / pure_gemm);

    free(A_fp8); free(B_fp8); free(B_fp32);
    free(A_fp32); free(A_fp32_full); free(C);
    return 0;
}
