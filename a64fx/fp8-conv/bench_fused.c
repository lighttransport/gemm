/*
 * Benchmark FP8 GEMM with fused conversion
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
    __asm__ volatile("mrs %0, cntvct_el0" : "=r"(val));
    return val;
}

static inline uint64_t get_freq(void) {
    uint64_t val;
    __asm__ volatile("mrs %0, cntfrq_el0" : "=r"(val));
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

// Pack A from [M][K] FP8 to [K][MR] FP8 for fused kernel
void pack_A_fp8(const fp8_e4m3_t* A, fp8_e4m3_t* A_packed,
                int64_t panel, int64_t K, int64_t lda, int64_t MR) {
    for (int64_t k = 0; k < K; k++) {
        for (int64_t m = 0; m < MR; m++) {
            A_packed[k * MR + m] = A[(panel*MR+m)*lda + k];
        }
    }
}

// Reference FP32 kernel (external)
extern void micro_kernel_fp32_8x3(const float* A, const float* B, float* C,
                                   int64_t K, int64_t unused, int64_t ldc);

// Fused FP8 kernel (external)
extern void fp8_fused_kernel_8x3(const fp8_e4m3_t* A_fp8, const float* B_fp32,
                                  float* C, int64_t ldc, int64_t K, uint32_t* lut);

int main() {
    printf("=== FP8 GEMM Fused Conversion Benchmark ===\n\n");
    init_lut();

    uint64_t freq = get_freq();
    double ticks_to_cycles = (double)CPU_FREQ_MHZ * 1e6 / freq;

    const int64_t MR = 8, NR = 3, VL = 16;
    int64_t M = 384, K = 512;
    int64_t N = NR * VL;  // 48
    int64_t M_panels = M / MR;
    double flops = 2.0 * M * N * K;

    printf("M=%ld, N=%ld, K=%ld, FLOPs=%.2fM\n\n", M, N, K, flops/1e6);

    // Allocate buffers
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

    // Pre-convert B
    for (int64_t k = 0; k < K; k++) {
        convert_fp8_to_fp32_sve(B_fp8 + k * N, B_fp32 + k * N, N);
    }

    int iters = 100;
    volatile uint64_t start, end;
    double cycles;
    int64_t ldc_bytes = N * sizeof(float);

    // Test 1: FP32 reference kernel (A pre-converted)
    printf("1. FP32 Reference (A pre-converted):\n");

    // Measure A conversion + kernel
    start = get_ticks();
    for (int it = 0; it < iters; it++) {
        for (int64_t p = 0; p < M_panels; p++) {
            // Convert A panel
            for (int64_t k = 0; k < K; k++) {
                for (int64_t m = 0; m < MR; m++) {
                    uint32_t bits = fp8_to_fp32_lut[A_fp8[(p*MR+m)*K + k]];
                    A_fp32[k * MR + m] = *((float*)&bits);
                }
            }
            // Run kernel
            micro_kernel_fp32_8x3(A_fp32, B_fp32, C + p * MR * N, K, 0, ldc_bytes);
        }
    }
    end = get_ticks();
    cycles = (double)(end - start) / iters * ticks_to_cycles;
    double gflops_ref = flops * CPU_FREQ_MHZ / (cycles * 1e3);
    printf("   Total: %.0f cycles, %.2f GFLOPS (%.1f%% of 128)\n",
           cycles, gflops_ref, 100.0 * gflops_ref / 128.0);

    // Test 2: Fused kernel (A conversion inside kernel)
    printf("\n2. Fused Kernel (A conversion inside):\n");

    start = get_ticks();
    for (int it = 0; it < iters; it++) {
        for (int64_t p = 0; p < M_panels; p++) {
            // Pack A to [K][MR] FP8 format
            pack_A_fp8(A_fp8, A_packed_fp8, p, K, K, MR);
            // Run fused kernel
            fp8_fused_kernel_8x3(A_packed_fp8, B_fp32, C + p * MR * N,
                                  ldc_bytes, K, fp8_to_fp32_lut);
        }
    }
    end = get_ticks();
    cycles = (double)(end - start) / iters * ticks_to_cycles;
    double gflops_fused = flops * CPU_FREQ_MHZ / (cycles * 1e3);
    printf("   Total: %.0f cycles, %.2f GFLOPS (%.1f%% of 128)\n",
           cycles, gflops_fused, 100.0 * gflops_fused / 128.0);

    // Test 3: Kernel only (no conversion, for reference)
    printf("\n3. FP32 Kernel Only (data in cache):\n");

    // Pre-convert one A panel
    for (int64_t k = 0; k < K; k++) {
        for (int64_t m = 0; m < MR; m++) {
            uint32_t bits = fp8_to_fp32_lut[A_fp8[m*K + k]];
            A_fp32[k * MR + m] = *((float*)&bits);
        }
    }

    start = get_ticks();
    for (int it = 0; it < iters; it++) {
        for (int64_t p = 0; p < M_panels; p++) {
            micro_kernel_fp32_8x3(A_fp32, B_fp32, C + p * MR * N, K, 0, ldc_bytes);
        }
    }
    end = get_ticks();
    cycles = (double)(end - start) / iters * ticks_to_cycles;
    double gflops_kernel = flops * CPU_FREQ_MHZ / (cycles * 1e3);
    printf("   Total: %.0f cycles, %.2f GFLOPS (%.1f%% of 128)\n",
           cycles, gflops_kernel, 100.0 * gflops_kernel / 128.0);

    printf("\n=== Summary ===\n");
    printf("FP32 kernel only: %.1f%% efficiency\n", 100.0 * gflops_kernel / 128.0);
    printf("FP32 + A convert: %.1f%% efficiency\n", 100.0 * gflops_ref / 128.0);
    printf("Fused kernel:     %.1f%% efficiency\n", 100.0 * gflops_fused / 128.0);

    free(A_fp8); free(A_packed_fp8); free(B_fp8); free(B_fp32); free(A_fp32); free(C);
    return 0;
}
