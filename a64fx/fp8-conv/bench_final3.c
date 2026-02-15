/*
 * Final FP8 GEMM Benchmark - with raw tick output
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

void convert_fp8_to_fp32_sve(const fp8_e4m3_t* src, float* dst, int64_t count) {
    int64_t vl = svcntw();
    for (int64_t i = 0; i < count; i += vl) {
        svbool_t pg = svwhilelt_b32(i, count);
        svuint32_t indices = svld1ub_u32(pg, src + i);
        svuint32_t fp32_bits = svld1_gather_index(pg, fp8_to_fp32_lut, indices);
        svst1(pg, dst + i, svreinterpret_f32(fp32_bits));
    }
}

void pack_A_fp8(const fp8_e4m3_t* A, fp8_e4m3_t* A_packed,
                int64_t panel, int64_t K, int64_t lda, int64_t MR) {
    for (int64_t k = 0; k < K; k++) {
        for (int64_t m = 0; m < MR; m++) {
            A_packed[k * MR + m] = A[(panel*MR+m)*lda + k];
        }
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

extern void micro_kernel_fp32_8x3(const float* A, const float* B, float* C,
                                   int64_t K, int64_t unused, int64_t ldc);

extern void fp8_fused_kernel_v3(const fp8_e4m3_t* A_fp8, const float* B_fp32,
                                 float* C, int64_t ldc, int64_t K, uint32_t* lut);

int main() {
    printf("=== FP8 GEMM Final Benchmark v3 ===\n\n");
    init_lut();

    const int64_t MR = 8, NR = 3, VL = 16;
    int64_t M = 384, K = 512;
    int64_t N = NR * VL;
    int64_t M_panels = M / MR;
    int64_t ldc_bytes = N * sizeof(float);

    // Timer frequency for manual calculation
    // A64FX: 100 MHz timer, 2000 MHz CPU -> multiply ticks by 20 for cycles
    printf("Timer: 100 MHz, CPU: 2000 MHz\n");
    printf("Ticks × 20 = cycles\n\n");

    printf("M=%ld, N=%ld, K=%ld, panels=%ld\n", M, N, K, M_panels);
    printf("FLOPs per panel: %ld\n", 2L * MR * N * K);
    printf("FLOPs per GEMM: %ld\n\n", 2L * M * N * K);

    fp8_e4m3_t* A_fp8 = aligned_alloc(ALIGN, M * K);
    fp8_e4m3_t* A_packed_fp8 = aligned_alloc(ALIGN, MR * K);
    fp8_e4m3_t* B_fp8 = aligned_alloc(ALIGN, K * N);
    float* B_fp32 = aligned_alloc(ALIGN, K * N * sizeof(float));
    float* A_fp32 = aligned_alloc(ALIGN, MR * K * sizeof(float));
    float* C = aligned_alloc(ALIGN, M * N * sizeof(float));

    srand(42);
    for (int64_t i = 0; i < M * K; i++) {
        A_fp8[i] = ((rand() % 14 + 1) << 3) | (rand() & 7);
    }
    for (int64_t i = 0; i < K * N; i++) {
        B_fp8[i] = ((rand() % 14 + 1) << 3) | (rand() & 7);
    }

    int iters = 100;
    uint64_t t0, t1;

    // Pre-convert B
    for (int64_t k = 0; k < K; k++) {
        convert_fp8_to_fp32_sve(B_fp8 + k * N, B_fp32 + k * N, N);
    }

    // Test 1: FP32 kernel only
    printf("1. FP32 kernel (1 panel × %d):\n", iters);
    convert_A_panel(A_fp8, A_fp32, 0, K, K, MR);
    memset(C, 0, M * N * sizeof(float));
    t0 = get_ticks();
    for (int it = 0; it < iters; it++) {
        micro_kernel_fp32_8x3(A_fp32, B_fp32, C, K, 0, ldc_bytes);
    }
    t1 = get_ticks();
    printf("   Ticks: %lu (per call: %lu)\n", t1-t0, (t1-t0)/iters);

    // Test 2: Fused kernel
    printf("2. Fused kernel v3 (1 panel × %d):\n", iters);
    pack_A_fp8(A_fp8, A_packed_fp8, 0, K, K, MR);
    memset(C, 0, M * N * sizeof(float));
    t0 = get_ticks();
    for (int it = 0; it < iters; it++) {
        fp8_fused_kernel_v3(A_packed_fp8, B_fp32, C, ldc_bytes, K, fp8_to_fp32_lut);
    }
    t1 = get_ticks();
    printf("   Ticks: %lu (per call: %lu)\n", t1-t0, (t1-t0)/iters);

    // Test 3: A packing
    printf("3. A FP8 packing (1 panel × %d):\n", iters);
    t0 = get_ticks();
    for (int it = 0; it < iters; it++) {
        pack_A_fp8(A_fp8, A_packed_fp8, 0, K, K, MR);
    }
    t1 = get_ticks();
    printf("   Ticks: %lu (per call: %lu)\n", t1-t0, (t1-t0)/iters);

    // Test 4: A conversion
    printf("4. A FP8->FP32 convert (1 panel × %d):\n", iters);
    t0 = get_ticks();
    for (int it = 0; it < iters; it++) {
        convert_A_panel(A_fp8, A_fp32, 0, K, K, MR);
    }
    t1 = get_ticks();
    printf("   Ticks: %lu (per call: %lu)\n", t1-t0, (t1-t0)/iters);

    // Full GEMM tests
    printf("\n=== Full GEMM (%ld panels × %d iters) ===\n", M_panels, iters);

    // Approach 1
    printf("Approach 1 (convert+FP32): ");
    memset(C, 0, M * N * sizeof(float));
    t0 = get_ticks();
    for (int it = 0; it < iters; it++) {
        for (int64_t p = 0; p < M_panels; p++) {
            convert_A_panel(A_fp8, A_fp32, p, K, K, MR);
            micro_kernel_fp32_8x3(A_fp32, B_fp32, C + p * MR * N, K, 0, ldc_bytes);
        }
    }
    t1 = get_ticks();
    printf("%lu ticks (%lu per GEMM)\n", t1-t0, (t1-t0)/iters);

    // Approach 2
    printf("Approach 2 (pack+fused):   ");
    memset(C, 0, M * N * sizeof(float));
    t0 = get_ticks();
    for (int it = 0; it < iters; it++) {
        for (int64_t p = 0; p < M_panels; p++) {
            pack_A_fp8(A_fp8, A_packed_fp8, p, K, K, MR);
            fp8_fused_kernel_v3(A_packed_fp8, B_fp32, C + p * MR * N, ldc_bytes, K, fp8_to_fp32_lut);
        }
    }
    t1 = get_ticks();
    printf("%lu ticks (%lu per GEMM)\n", t1-t0, (t1-t0)/iters);

    printf("\n=== Manual Calculation ===\n");
    printf("Peak = 128 GFLOPS\n");
    printf("FLOPs per GEMM = %ld\n", 2L * M * N * K);
    printf("At 100%% efficiency: %ld ticks/GEMM = %.1f cycles/GEMM\n",
           (2L * M * N * K) / 128 / 100 * 5, (double)(2L * M * N * K) / 128.0);

    free(A_fp8); free(A_packed_fp8); free(B_fp8); free(B_fp32); free(A_fp32); free(C);
    return 0;
}
