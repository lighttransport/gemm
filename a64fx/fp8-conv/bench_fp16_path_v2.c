/*
 * FP8 GEMM via FP16 Intermediate Path - v2 (raw ticks output)
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <arm_sve.h>

#define ALIGN 64

typedef uint8_t fp8_e4m3_t;
typedef uint16_t fp16_t;

uint16_t fp8_to_fp16_lut[256] __attribute__((aligned(64)));
uint32_t fp8_to_fp32_lut[256] __attribute__((aligned(64)));

static inline uint64_t get_ticks(void) {
    uint64_t val;
    __asm__ volatile("isb" ::: "memory");
    __asm__ volatile("mrs %0, cntvct_el0" : "=r"(val));
    __asm__ volatile("isb" ::: "memory");
    return val;
}

static inline uint16_t fp32_to_fp16(uint32_t f32) {
    uint32_t sign = (f32 >> 31) & 1;
    int32_t exp = ((f32 >> 23) & 0xFF) - 127;
    uint32_t mant = f32 & 0x7FFFFF;
    if (exp < -14) return (uint16_t)(sign << 15);
    if (exp > 15) return (uint16_t)((sign << 15) | 0x7C00);
    return (uint16_t)((sign << 15) | ((exp + 15) << 10) | (mant >> 13));
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
        fp8_to_fp16_lut[i] = fp32_to_fp16(fp32);
    }
}

void convert_A_fp8_to_fp16(const fp8_e4m3_t* A, fp16_t* A_fp16,
                           int64_t panel, int64_t K, int64_t lda, int64_t MR) {
    for (int64_t k = 0; k < K; k++) {
        for (int64_t m = 0; m < MR; m++) {
            A_fp16[k * MR + m] = fp8_to_fp16_lut[A[(panel*MR+m)*lda + k]];
        }
    }
}

void convert_A_full_fp16(const fp8_e4m3_t* A, fp16_t* A_fp16,
                          int64_t M, int64_t K, int64_t lda, int64_t MR) {
    int64_t M_panels = M / MR;
    for (int64_t p = 0; p < M_panels; p++) {
        fp16_t* dst = A_fp16 + p * MR * K;
        for (int64_t k = 0; k < K; k++) {
            for (int64_t m = 0; m < MR; m++) {
                dst[k * MR + m] = fp8_to_fp16_lut[A[(p*MR+m)*lda + k]];
            }
        }
    }
}

void convert_A_fp8_to_fp32(const fp8_e4m3_t* A, float* A_fp32,
                           int64_t panel, int64_t K, int64_t lda, int64_t MR) {
    for (int64_t k = 0; k < K; k++) {
        for (int64_t m = 0; m < MR; m++) {
            uint32_t bits = fp8_to_fp32_lut[A[(panel*MR+m)*lda + k]];
            A_fp32[k * MR + m] = *((float*)&bits);
        }
    }
}

void convert_A_full_fp32(const fp8_e4m3_t* A, float* A_fp32,
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

void convert_B_fp8_to_fp32(const fp8_e4m3_t* B, float* B_fp32, int64_t K, int64_t N) {
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

extern void micro_kernel_fp16fp32_8x3(const fp16_t* A, const float* B, float* C,
                                       int64_t K, int64_t unused, int64_t ldc_bytes);

int main() {
    printf("=== FP8 via FP16 Path v2 (raw ticks) ===\n");
    printf("Timer: 100 MHz, CPU: 2000 MHz (ticks × 20 = cycles)\n\n");
    init_lut();

    const int64_t MR = 8, NR = 3, VL = 16;
    int64_t M = 384, K = 512, N = NR * VL;
    int64_t M_panels = M / MR;
    int64_t ldc_bytes = N * sizeof(float);
    int iters = 100;

    printf("M=%ld, N=%ld, K=%ld, panels=%ld, iters=%d\n", M, N, K, M_panels, iters);
    printf("FLOPs/panel: %ld, FLOPs/GEMM: %ld\n", 2L*MR*N*K, 2L*M*N*K);
    printf("Peak=128 GFLOPS. Ideal ticks/panel=%ld\n\n",
           2L * MR * N * K / 128 / 100 * 5);

    fp8_e4m3_t* A_fp8 = aligned_alloc(ALIGN, M * K);
    fp8_e4m3_t* B_fp8 = aligned_alloc(ALIGN, K * N);
    float* B_fp32 = aligned_alloc(ALIGN, K * N * sizeof(float));
    float* A_fp32 = aligned_alloc(ALIGN, MR * K * sizeof(float));
    float* A_fp32_full = aligned_alloc(ALIGN, M * K * sizeof(float));
    fp16_t* A_fp16 = aligned_alloc(ALIGN, MR * K * sizeof(fp16_t));
    fp16_t* A_fp16_full = aligned_alloc(ALIGN, M * K * sizeof(fp16_t));
    float* C = aligned_alloc(ALIGN, M * N * sizeof(float));

    srand(42);
    for (int64_t i = 0; i < M * K; i++) {
        A_fp8[i] = ((rand() % 14 + 1) << 3) | (rand() & 7);
    }
    for (int64_t i = 0; i < K * N; i++) {
        B_fp8[i] = ((rand() % 14 + 1) << 3) | (rand() & 7);
    }

    convert_B_fp8_to_fp32(B_fp8, B_fp32, K, N);

    uint64_t t0, t1;

    // === Conversion costs ===
    printf("=== Conversion Costs (1 panel × %d) ===\n", iters);

    t0 = get_ticks();
    for (int it = 0; it < iters; it++) {
        convert_A_fp8_to_fp32(A_fp8, A_fp32, 0, K, K, MR);
    }
    t1 = get_ticks();
    printf("FP8->FP32: %lu ticks (%lu per call)\n", t1-t0, (t1-t0)/iters);

    t0 = get_ticks();
    for (int it = 0; it < iters; it++) {
        convert_A_fp8_to_fp16(A_fp8, A_fp16, 0, K, K, MR);
    }
    t1 = get_ticks();
    printf("FP8->FP16: %lu ticks (%lu per call)\n\n", t1-t0, (t1-t0)/iters);

    // === Kernel costs ===
    printf("=== Kernel Costs (1 panel × %d) ===\n", iters);

    convert_A_fp8_to_fp32(A_fp8, A_fp32, 0, K, K, MR);
    memset(C, 0, M * N * sizeof(float));
    t0 = get_ticks();
    for (int it = 0; it < iters; it++) {
        fp8_gemm_kernel_asm(A_fp32, B_fp32, C, N, K);
    }
    t1 = get_ticks();
    printf("FP32 kernel: %lu ticks (%lu per call)\n", t1-t0, (t1-t0)/iters);

    convert_A_fp8_to_fp16(A_fp8, A_fp16, 0, K, K, MR);
    memset(C, 0, M * N * sizeof(float));
    t0 = get_ticks();
    for (int it = 0; it < iters; it++) {
        micro_kernel_fp16fp32_8x3(A_fp16, B_fp32, C, K, 0, ldc_bytes);
    }
    t1 = get_ticks();
    printf("FP16->FP32 kernel: %lu ticks (%lu per call)\n\n", t1-t0, (t1-t0)/iters);

    // === Full GEMM tests ===
    printf("=== Full GEMM (%ld panels × %d) ===\n", M_panels, iters);

    // Approach 1: FP8->FP32 per-panel + FP32 kernel
    memset(C, 0, M * N * sizeof(float));
    t0 = get_ticks();
    for (int it = 0; it < iters; it++) {
        for (int64_t p = 0; p < M_panels; p++) {
            convert_A_fp8_to_fp32(A_fp8, A_fp32, p, K, K, MR);
            fp8_gemm_kernel_asm(A_fp32, B_fp32, C + p * MR * N, N, K);
        }
    }
    t1 = get_ticks();
    printf("FP8->FP32 per-panel: %lu ticks (%lu per GEMM)\n", t1-t0, (t1-t0)/iters);

    // Approach 2: FP8->FP16 per-panel + FP16->FP32 kernel
    memset(C, 0, M * N * sizeof(float));
    t0 = get_ticks();
    for (int it = 0; it < iters; it++) {
        for (int64_t p = 0; p < M_panels; p++) {
            convert_A_fp8_to_fp16(A_fp8, A_fp16, p, K, K, MR);
            micro_kernel_fp16fp32_8x3(A_fp16, B_fp32, C + p * MR * N, K, 0, ldc_bytes);
        }
    }
    t1 = get_ticks();
    printf("FP8->FP16 per-panel: %lu ticks (%lu per GEMM)\n", t1-t0, (t1-t0)/iters);

    // Approach 3: Pre-convert to FP32, run FP32 kernel
    t0 = get_ticks();
    convert_A_full_fp32(A_fp8, A_fp32_full, M, K, K, MR);
    t1 = get_ticks();
    uint64_t preconv_fp32 = t1 - t0;

    memset(C, 0, M * N * sizeof(float));
    t0 = get_ticks();
    for (int it = 0; it < iters; it++) {
        for (int64_t p = 0; p < M_panels; p++) {
            fp8_gemm_kernel_asm(A_fp32_full + p * MR * K, B_fp32, C + p * MR * N, N, K);
        }
    }
    t1 = get_ticks();
    printf("Pre-conv FP32: preconv=%lu ticks, GEMM=%lu ticks (%lu per GEMM)\n",
           preconv_fp32, t1-t0, (t1-t0)/iters);

    // Approach 4: Pre-convert to FP16, run FP16->FP32 kernel
    t0 = get_ticks();
    convert_A_full_fp16(A_fp8, A_fp16_full, M, K, K, MR);
    t1 = get_ticks();
    uint64_t preconv_fp16 = t1 - t0;

    memset(C, 0, M * N * sizeof(float));
    t0 = get_ticks();
    for (int it = 0; it < iters; it++) {
        for (int64_t p = 0; p < M_panels; p++) {
            micro_kernel_fp16fp32_8x3(A_fp16_full + p * MR * K, B_fp32, C + p * MR * N, K, 0, ldc_bytes);
        }
    }
    t1 = get_ticks();
    printf("Pre-conv FP16: preconv=%lu ticks, GEMM=%lu ticks (%lu per GEMM)\n\n",
           preconv_fp16, t1-t0, (t1-t0)/iters);

    printf("=== Expected Efficiency (compute manually) ===\n");
    printf("Peak = 128 GFLOPS = 128e9 / 2e9 = 64 FLOP/cycle\n");
    printf("FLOPs/GEMM = %ld = 18.9M\n", 2L*M*N*K);
    printf("Ideal cycles/GEMM = 18.9M / 64 = 295,000\n");
    printf("Ideal ticks/GEMM = 295,000 / 20 = 14750\n");
    printf("90%% target = 14750 / 0.9 = 16389 ticks\n");

    free(A_fp8); free(B_fp8); free(B_fp32);
    free(A_fp32); free(A_fp32_full); free(A_fp16); free(A_fp16_full); free(C);
    return 0;
}
