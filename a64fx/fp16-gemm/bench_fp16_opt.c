/*
 * FP16 GEMM - Optimization Test
 * Testing SVE-optimized FP16→FP32 conversion
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <arm_sve.h>
#include <arm_fp16.h>

#define ALIGN 64

typedef __fp16 fp16_t;

static inline uint64_t get_ticks(void) {
    uint64_t val;
    __asm__ volatile("isb" ::: "memory");
    __asm__ volatile("mrs %0, cntvct_el0" : "=r"(val));
    __asm__ volatile("isb" ::: "memory");
    return val;
}

// Scalar FP16→FP32 conversion
void convert_scalar(const fp16_t* src, float* dst, int64_t count) {
    for (int64_t i = 0; i < count; i++) {
        dst[i] = (float)src[i];
    }
}

// SVE FP16→FP32 conversion
void convert_sve(const fp16_t* src, float* dst, int64_t count) {
    int64_t vl_s = svcntw();  // FP32 vector length
    int64_t vl_h = svcnth();  // FP16 vector length

    svbool_t pg_s = svptrue_b32();

    for (int64_t i = 0; i < count; i += vl_s) {
        svbool_t pg = svwhilelt_b32(i, count);

        // Load FP16 values (16 elements)
        svfloat16_t fp16_vec = svld1_f16(pg, (const float16_t*)(src + i));

        // Convert to FP32
        svfloat32_t fp32_vec = svcvt_f32_f16_x(pg, fp16_vec);

        // Store FP32
        svst1_f32(pg, dst + i, fp32_vec);
    }
}

// Convert A matrix with packing (row-major to packed)
void convert_A_scalar(const fp16_t* A, float* A_fp32,
                      int64_t M, int64_t K, int64_t lda, int64_t MR) {
    int64_t M_panels = M / MR;
    for (int64_t p = 0; p < M_panels; p++) {
        float* dst = A_fp32 + p * MR * K;
        for (int64_t k = 0; k < K; k++) {
            for (int64_t m = 0; m < MR; m++) {
                dst[k * MR + m] = (float)A[(p*MR+m)*lda + k];
            }
        }
    }
}

// SVE-optimized A matrix conversion with packing
void convert_A_sve(const fp16_t* A, float* A_fp32,
                   int64_t M, int64_t K, int64_t lda, int64_t MR) {
    int64_t vl = svcntw();
    int64_t M_panels = M / MR;

    for (int64_t p = 0; p < M_panels; p++) {
        float* dst = A_fp32 + p * MR * K;
        for (int64_t k = 0; k < K; k += vl) {
            svbool_t pg = svwhilelt_b32(k, K);
            for (int64_t m = 0; m < MR; m++) {
                const fp16_t* src = A + (p*MR+m)*lda + k;
                svfloat16_t fp16_vec = svld1_f16(pg, (const float16_t*)src);
                svfloat32_t fp32_vec = svcvt_f32_f16_x(pg, fp16_vec);
                svst1_f32(pg, dst + m, fp32_vec);
            }
            dst += MR * vl;
        }
    }
}

void convert_B(const fp16_t* B, float* B_fp32, int64_t K, int64_t N) {
    convert_sve((const fp16_t*)B, B_fp32, K * N);
}

extern void fp8_gemm_kernel_asm(const float* A, const float* B, float* C,
                                 int64_t ldc_elem, int64_t K);

int main() {
    printf("╔═══════════════════════════════════════════════════════╗\n");
    printf("║      FP16→FP32 Conversion Optimization Test          ║\n");
    printf("╚═══════════════════════════════════════════════════════╝\n\n");

    int64_t M = 384, K = 512, N = 48, MR = 8;
    int64_t M_panels = M / MR;
    int64_t flops = 2L * M * N * K;
    int64_t ideal = flops / 1280;
    int iters = 100;

    printf("Config: M=%ld, K=%ld, N=%ld\n", M, K, N);
    printf("Target: Ideal=%ld ticks, 90%%=%ld ticks\n\n", ideal, ideal*100/90);

    fp16_t* A_fp16 = aligned_alloc(ALIGN, M * K * sizeof(fp16_t));
    fp16_t* B_fp16 = aligned_alloc(ALIGN, K * N * sizeof(fp16_t));
    float* B_fp32 = aligned_alloc(ALIGN, K * N * sizeof(float));
    float* A_fp32 = aligned_alloc(ALIGN, M * K * sizeof(float));
    float* C = aligned_alloc(ALIGN, M * N * sizeof(float));

    srand(42);
    for (int64_t i = 0; i < M * K; i++) {
        A_fp16[i] = (fp16_t)(((float)rand() / RAND_MAX) * 2.0f - 1.0f);
    }
    for (int64_t i = 0; i < K * N; i++) {
        B_fp16[i] = (fp16_t)(((float)rand() / RAND_MAX) * 2.0f - 1.0f);
    }

    convert_B(B_fp16, B_fp32, K, N);

    uint64_t t0, t1;

    printf("═══════════════════════════════════════════════════════\n");
    printf("         CONVERSION PERFORMANCE COMPARISON\n");
    printf("═══════════════════════════════════════════════════════\n\n");

    // Test scalar conversion
    printf("1. Scalar FP16→FP32 (full matrix):\n");
    t0 = get_ticks();
    for (int it = 0; it < iters; it++) {
        convert_scalar(A_fp16, A_fp32, M * K);
    }
    t1 = get_ticks();
    uint64_t scalar_flat = (t1 - t0) / iters;
    printf("   %lu ticks (%.2f cycles/elem)\n", scalar_flat, scalar_flat * 20.0 / (M*K));

    // Test SVE conversion
    printf("2. SVE FP16→FP32 (full matrix):\n");
    t0 = get_ticks();
    for (int it = 0; it < iters; it++) {
        convert_sve(A_fp16, A_fp32, M * K);
    }
    t1 = get_ticks();
    uint64_t sve_flat = (t1 - t0) / iters;
    printf("   %lu ticks (%.2f cycles/elem)\n", sve_flat, sve_flat * 20.0 / (M*K));

    // Test scalar conversion with packing
    printf("3. Scalar convert+pack (A matrix):\n");
    t0 = get_ticks();
    for (int it = 0; it < iters; it++) {
        convert_A_scalar(A_fp16, A_fp32, M, K, K, MR);
    }
    t1 = get_ticks();
    uint64_t scalar_pack = (t1 - t0) / iters;
    printf("   %lu ticks (%.2f cycles/elem)\n", scalar_pack, scalar_pack * 20.0 / (M*K));

    // Test SVE conversion with packing
    printf("4. SVE convert+pack (A matrix):\n");
    t0 = get_ticks();
    for (int it = 0; it < iters; it++) {
        convert_A_sve(A_fp16, A_fp32, M, K, K, MR);
    }
    t1 = get_ticks();
    uint64_t sve_pack = (t1 - t0) / iters;
    printf("   %lu ticks (%.2f cycles/elem)\n\n", sve_pack, sve_pack * 20.0 / (M*K));

    // Full GEMM tests
    printf("═══════════════════════════════════════════════════════\n");
    printf("              FULL GEMM PERFORMANCE\n");
    printf("═══════════════════════════════════════════════════════\n\n");

    // Pure FP32 GEMM (baseline)
    printf("1. Pure FP32 GEMM:\n");
    convert_A_scalar(A_fp16, A_fp32, M, K, K, MR);
    memset(C, 0, M * N * sizeof(float));
    t0 = get_ticks();
    for (int it = 0; it < iters; it++) {
        for (int64_t p = 0; p < M_panels; p++) {
            fp8_gemm_kernel_asm(A_fp32 + p * MR * K, B_fp32, C + p * MR * N, N, K);
        }
    }
    t1 = get_ticks();
    uint64_t pure = (t1 - t0) / iters;
    printf("   %lu ticks, %lu%% eff\n", pure, ideal * 100 / pure);

    // Scalar convert + FP32 GEMM
    printf("2. Scalar convert + FP32 GEMM:\n");
    memset(C, 0, M * N * sizeof(float));
    t0 = get_ticks();
    for (int it = 0; it < iters; it++) {
        convert_A_scalar(A_fp16, A_fp32, M, K, K, MR);
        for (int64_t p = 0; p < M_panels; p++) {
            fp8_gemm_kernel_asm(A_fp32 + p * MR * K, B_fp32, C + p * MR * N, N, K);
        }
    }
    t1 = get_ticks();
    uint64_t scalar_gemm = (t1 - t0) / iters;
    printf("   %lu ticks, %lu%% eff\n", scalar_gemm, ideal * 100 / scalar_gemm);

    // SVE convert + FP32 GEMM
    printf("3. SVE convert + FP32 GEMM:\n");
    memset(C, 0, M * N * sizeof(float));
    t0 = get_ticks();
    for (int it = 0; it < iters; it++) {
        convert_A_sve(A_fp16, A_fp32, M, K, K, MR);
        for (int64_t p = 0; p < M_panels; p++) {
            fp8_gemm_kernel_asm(A_fp32 + p * MR * K, B_fp32, C + p * MR * N, N, K);
        }
    }
    t1 = get_ticks();
    uint64_t sve_gemm = (t1 - t0) / iters;
    printf("   %lu ticks, %lu%% eff\n\n", sve_gemm, ideal * 100 / sve_gemm);

    // Summary
    printf("═══════════════════════════════════════════════════════\n");
    printf("                      SUMMARY\n");
    printf("═══════════════════════════════════════════════════════\n\n");

    printf("Conversion Speedup (SVE vs Scalar):\n");
    printf("  Flat array:    %.2fx faster\n", (double)scalar_flat / sve_flat);
    printf("  With packing:  %.2fx faster\n\n", (double)scalar_pack / sve_pack);

    printf("Full GEMM Results:\n");
    printf("  Pure FP32:            %5lu ticks (%3lu%%)\n", pure, ideal*100/pure);
    printf("  Scalar conv + GEMM:   %5lu ticks (%3lu%%)\n", scalar_gemm, ideal*100/scalar_gemm);
    printf("  SVE conv + GEMM:      %5lu ticks (%3lu%%)\n\n", sve_gemm, ideal*100/sve_gemm);

    printf("Conversion overhead:\n");
    printf("  Scalar: %lu ticks (%.1f%% of GEMM)\n", scalar_pack, scalar_pack * 100.0 / pure);
    printf("  SVE:    %lu ticks (%.1f%% of GEMM)\n", sve_pack, sve_pack * 100.0 / pure);

    free(A_fp16); free(B_fp16); free(B_fp32); free(A_fp32); free(C);
    return 0;
}
