/*
 * Final FP8 GEMM Benchmark
 *
 * Compare two approaches:
 * 1. Pre-convert: A FP8→FP32 conversion + FP32 kernel
 * 2. Fused: A FP8 packing + fused kernel (conversion inside)
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

// Pack A from [M][K] FP8 to [K][MR] FP8 (transpose without conversion)
void pack_A_fp8(const fp8_e4m3_t* A, fp8_e4m3_t* A_packed,
                int64_t panel, int64_t K, int64_t lda, int64_t MR) {
    for (int64_t k = 0; k < K; k++) {
        for (int64_t m = 0; m < MR; m++) {
            A_packed[k * MR + m] = A[(panel*MR+m)*lda + k];
        }
    }
}

// Convert A from [M][K] FP8 to [K][MR] FP32 (transpose + convert)
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
    printf("=== FP8 GEMM Final Benchmark ===\n\n");
    init_lut();

    uint64_t freq = get_freq();
    double ticks_to_cycles = (double)CPU_FREQ_MHZ * 1e6 / freq;

    const int64_t MR = 8, NR = 3, VL = 16;
    int64_t M = 384, K = 512;
    int64_t N = NR * VL;  // 48
    int64_t M_panels = M / MR;
    double flops = 2.0 * M * N * K;
    int64_t ldc_bytes = N * sizeof(float);

    printf("M=%ld, N=%ld, K=%ld\n", M, N, K);
    printf("FLOPs per GEMM: %.2f M\n", flops / 1e6);
    printf("Peak: 128 GFLOPS\n\n");

    // Allocate
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

    // ===== B Conversion (one-time) =====
    printf("=== B Conversion (one-time cost) ===\n");
    t0 = get_ticks();
    for (int it = 0; it < iters; it++) {
        for (int64_t k = 0; k < K; k++) {
            convert_fp8_to_fp32_sve(B_fp8 + k * N, B_fp32 + k * N, N);
        }
    }
    t1 = get_ticks();
    double b_ticks = (double)(t1 - t0) / iters;
    double b_cycles = b_ticks * ticks_to_cycles;
    printf("B convert: %.0f cycles (%.2f cycles/elem)\n\n", b_cycles, b_cycles / (K * N));

    // Pre-convert B for subsequent tests
    for (int64_t k = 0; k < K; k++) {
        convert_fp8_to_fp32_sve(B_fp8 + k * N, B_fp32 + k * N, N);
    }

    // ===== Approach 1: Pre-convert A + FP32 Kernel =====
    printf("=== Approach 1: Pre-convert A + FP32 Kernel ===\n");
    memset(C, 0, M * N * sizeof(float));

    t0 = get_ticks();
    for (int it = 0; it < iters; it++) {
        for (int64_t p = 0; p < M_panels; p++) {
            // Convert A panel: FP8 [M][K] -> FP32 [K][MR]
            convert_A_panel(A_fp8, A_fp32, p, K, K, MR);
            // FP32 kernel
            micro_kernel_fp32_8x3(A_fp32, B_fp32, C + p * MR * N, K, 0, ldc_bytes);
        }
    }
    t1 = get_ticks();
    double approach1_ticks = (double)(t1 - t0) / iters;
    double approach1_cycles = approach1_ticks * ticks_to_cycles;
    double approach1_gflops = flops * CPU_FREQ_MHZ / (approach1_cycles * 1e3);
    printf("Total: %.0f cycles, %.2f GFLOPS (%.1f%%)\n\n",
           approach1_cycles, approach1_gflops, 100.0 * approach1_gflops / 128.0);

    // ===== Approach 2: Pack A FP8 + Fused Kernel =====
    printf("=== Approach 2: Pack A FP8 + Fused Kernel (v3) ===\n");
    memset(C, 0, M * N * sizeof(float));

    t0 = get_ticks();
    for (int it = 0; it < iters; it++) {
        for (int64_t p = 0; p < M_panels; p++) {
            // Pack A: FP8 [M][K] -> FP8 [K][MR] (just transpose, no conversion!)
            pack_A_fp8(A_fp8, A_packed_fp8, p, K, K, MR);
            // Fused kernel (conversion inside)
            fp8_fused_kernel_v3(A_packed_fp8, B_fp32, C + p * MR * N, ldc_bytes, K, fp8_to_fp32_lut);
        }
    }
    t1 = get_ticks();
    double approach2_ticks = (double)(t1 - t0) / iters;
    double approach2_cycles = approach2_ticks * ticks_to_cycles;
    double approach2_gflops = flops * CPU_FREQ_MHZ / (approach2_cycles * 1e3);
    printf("Total: %.0f cycles, %.2f GFLOPS (%.1f%%)\n\n",
           approach2_cycles, approach2_gflops, 100.0 * approach2_gflops / 128.0);

    // ===== Component Breakdown =====
    printf("=== Component Breakdown ===\n");

    // A packing only (FP8 transpose)
    t0 = get_ticks();
    for (int it = 0; it < iters; it++) {
        for (int64_t p = 0; p < M_panels; p++) {
            pack_A_fp8(A_fp8, A_packed_fp8, p, K, K, MR);
        }
    }
    t1 = get_ticks();
    double pack_ticks = (double)(t1 - t0) / iters;
    double pack_cycles = pack_ticks * ticks_to_cycles;
    printf("A pack (FP8 transpose): %.0f cycles (%.2f cycles/elem)\n",
           pack_cycles, pack_cycles / (M * K));

    // A conversion only (FP8 -> FP32 + pack)
    t0 = get_ticks();
    for (int it = 0; it < iters; it++) {
        for (int64_t p = 0; p < M_panels; p++) {
            convert_A_panel(A_fp8, A_fp32, p, K, K, MR);
        }
    }
    t1 = get_ticks();
    double conv_ticks = (double)(t1 - t0) / iters;
    double conv_cycles = conv_ticks * ticks_to_cycles;
    printf("A convert (FP8->FP32+pack): %.0f cycles (%.2f cycles/elem)\n",
           conv_cycles, conv_cycles / (M * K));

    // FP32 kernel only
    convert_A_panel(A_fp8, A_fp32, 0, K, K, MR);  // Pre-convert one panel
    t0 = get_ticks();
    for (int it = 0; it < iters; it++) {
        for (int64_t p = 0; p < M_panels; p++) {
            micro_kernel_fp32_8x3(A_fp32, B_fp32, C + p * MR * N, K, 0, ldc_bytes);
        }
    }
    t1 = get_ticks();
    double fp32_ticks = (double)(t1 - t0) / iters;
    double fp32_cycles = fp32_ticks * ticks_to_cycles;
    double fp32_gflops = flops * CPU_FREQ_MHZ / (fp32_cycles * 1e3);
    printf("FP32 kernel only: %.0f cycles, %.2f GFLOPS (%.1f%%)\n",
           fp32_cycles, fp32_gflops, 100.0 * fp32_gflops / 128.0);

    // Fused kernel only
    pack_A_fp8(A_fp8, A_packed_fp8, 0, K, K, MR);  // Pre-pack one panel
    t0 = get_ticks();
    for (int it = 0; it < iters; it++) {
        for (int64_t p = 0; p < M_panels; p++) {
            fp8_fused_kernel_v3(A_packed_fp8, B_fp32, C + p * MR * N, ldc_bytes, K, fp8_to_fp32_lut);
        }
    }
    t1 = get_ticks();
    double fused_ticks = (double)(t1 - t0) / iters;
    double fused_cycles = fused_ticks * ticks_to_cycles;
    double fused_gflops = flops * CPU_FREQ_MHZ / (fused_cycles * 1e3);
    printf("Fused kernel only: %.0f cycles, %.2f GFLOPS (%.1f%%)\n\n",
           fused_cycles, fused_gflops, 100.0 * fused_gflops / 128.0);

    // ===== Summary =====
    printf("=== Summary ===\n");
    printf("Approach 1 (pre-convert): %.1f%% efficiency\n", 100.0 * approach1_gflops / 128.0);
    printf("Approach 2 (fused):       %.1f%% efficiency\n", 100.0 * approach2_gflops / 128.0);
    printf("\nFused kernel advantage:\n");
    printf("  - A packing cost: %.2f cycles/elem (vs %.2f for full conversion)\n",
           pack_cycles / (M * K), conv_cycles / (M * K));
    printf("  - Savings: %.1fx less overhead for A\n", conv_cycles / pack_cycles);

    free(A_fp8); free(A_packed_fp8); free(B_fp8); free(B_fp32); free(A_fp32); free(C);
    return 0;
}
