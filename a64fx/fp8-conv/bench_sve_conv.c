/*
 * FP8 GEMM with SVE-optimized conversion
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

// SVE gather-based conversion
void convert_sve(const fp8_e4m3_t* A, float* A_fp32,
                 int64_t M, int64_t K, int64_t lda, int64_t MR) {
    int64_t vl = svcntw();
    int64_t M_panels = M / MR;

    for (int64_t p = 0; p < M_panels; p++) {
        float* dst = A_fp32 + p * MR * K;
        for (int64_t k = 0; k < K; k += vl) {
            svbool_t pg = svwhilelt_b32(k, K);
            for (int64_t m = 0; m < MR; m++) {
                const fp8_e4m3_t* src = A + (p*MR+m)*lda + k;
                svuint32_t indices = svld1ub_u32(pg, src);
                svuint32_t fp32_bits = svld1_gather_index(pg, fp8_to_fp32_lut, indices);
                svst1(pg, dst + m, svreinterpret_f32(fp32_bits));
            }
            dst += MR * vl;
        }
    }
}

// Scalar conversion for comparison
void convert_scalar(const fp8_e4m3_t* A, float* A_fp32,
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
    printf("=== FP8 GEMM with SVE Conversion ===\n\n");
    init_lut();

    int64_t M = 384, K = 512, N = 48, MR = 8;
    int64_t M_panels = M / MR;
    int64_t flops = 2L * M * N * K;
    int64_t ideal = flops / 1280;  // ticks at 128 GFLOPS
    int iters = 100;

    printf("M=%ld, K=%ld, N=%ld, panels=%ld\n", M, K, N, M_panels);
    printf("FLOPs=%ld, Ideal=%ld ticks, 90%%=%ld ticks\n\n", flops, ideal, ideal*100/90);

    fp8_e4m3_t* A_fp8 = aligned_alloc(ALIGN, M * K);
    fp8_e4m3_t* B_fp8 = aligned_alloc(ALIGN, K * N);
    float* B_fp32 = aligned_alloc(ALIGN, K * N * sizeof(float));
    float* A_fp32 = aligned_alloc(ALIGN, M * K * sizeof(float));
    float* C = aligned_alloc(ALIGN, M * N * sizeof(float));

    srand(42);
    for (int64_t i = 0; i < M * K; i++) A_fp8[i] = ((rand() % 14 + 1) << 3) | (rand() & 7);
    for (int64_t i = 0; i < K * N; i++) B_fp8[i] = ((rand() % 14 + 1) << 3) | (rand() & 7);
    convert_B(B_fp8, B_fp32, K, N);

    uint64_t t0, t1;

    // 1. Pure GEMM (baseline)
    printf("1. Pure FP32 GEMM:\n");
    convert_scalar(A_fp8, A_fp32, M, K, K, MR);
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

    // 2. Scalar conv + GEMM
    printf("2. Scalar conv + GEMM:\n");
    memset(C, 0, M * N * sizeof(float));
    t0 = get_ticks();
    for (int it = 0; it < iters; it++) {
        convert_scalar(A_fp8, A_fp32, M, K, K, MR);
        for (int64_t p = 0; p < M_panels; p++) {
            fp8_gemm_kernel_asm(A_fp32 + p * MR * K, B_fp32, C + p * MR * N, N, K);
        }
    }
    t1 = get_ticks();
    uint64_t scalar = (t1 - t0) / iters;
    printf("   %lu ticks, %lu%% eff\n", scalar, ideal * 100 / scalar);

    // 3. SVE conv + GEMM
    printf("3. SVE conv + GEMM:\n");
    memset(C, 0, M * N * sizeof(float));
    t0 = get_ticks();
    for (int it = 0; it < iters; it++) {
        convert_sve(A_fp8, A_fp32, M, K, K, MR);
        for (int64_t p = 0; p < M_panels; p++) {
            fp8_gemm_kernel_asm(A_fp32 + p * MR * K, B_fp32, C + p * MR * N, N, K);
        }
    }
    t1 = get_ticks();
    uint64_t sve = (t1 - t0) / iters;
    printf("   %lu ticks, %lu%% eff\n\n", sve, ideal * 100 / sve);

    // Summary
    printf("=== SUMMARY ===\n");
    printf("Pure GEMM:      %5lu ticks (%3lu%%)\n", pure, ideal*100/pure);
    printf("Scalar + GEMM:  %5lu ticks (%3lu%%)\n", scalar, ideal*100/scalar);
    printf("SVE + GEMM:     %5lu ticks (%3lu%%)\n", sve, ideal*100/sve);
    printf("\nSVE conversion improves overall efficiency from %lu%% to %lu%%\n",
           ideal*100/scalar, ideal*100/sve);

    free(A_fp8); free(B_fp8); free(B_fp32); free(A_fp32); free(C);
    return 0;
}
