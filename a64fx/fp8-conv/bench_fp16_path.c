/*
 * FP8 GEMM via FP16 Intermediate Path
 *
 * Test the user's suggested approach:
 * 1. FP8 -> FP16 (LUT to L1 cache)
 * 2. FP16 -> FP32 via FCVT in kernel
 *
 * This avoids the expensive FP8 gather inside the kernel by converting
 * to FP16 first, then using the fast ld1rh + fcvt path.
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <arm_sve.h>

#define ALIGN 64
#define CPU_FREQ_MHZ 2000

typedef uint8_t fp8_e4m3_t;
typedef uint16_t fp16_t;

// LUT for FP8 to FP16 conversion (smaller than FP32, 512 bytes vs 1KB)
uint16_t fp8_to_fp16_lut[256] __attribute__((aligned(64)));

// LUT for FP8 to FP32 conversion (for comparison)
uint32_t fp8_to_fp32_lut[256] __attribute__((aligned(64)));

static inline uint64_t get_ticks(void) {
    uint64_t val;
    __asm__ volatile("isb" ::: "memory");
    __asm__ volatile("mrs %0, cntvct_el0" : "=r"(val));
    __asm__ volatile("isb" ::: "memory");
    return val;
}

// Convert FP32 bits to FP16 bits (software)
static inline uint16_t fp32_to_fp16(uint32_t f32) {
    uint32_t sign = (f32 >> 31) & 1;
    int32_t exp = ((f32 >> 23) & 0xFF) - 127;
    uint32_t mant = f32 & 0x7FFFFF;

    if (exp < -14) {
        // Underflow - return signed zero
        return (uint16_t)(sign << 15);
    } else if (exp > 15) {
        // Overflow - return infinity
        return (uint16_t)((sign << 15) | 0x7C00);
    } else {
        // Normal range
        uint16_t exp16 = (uint16_t)(exp + 15);
        uint16_t mant16 = (uint16_t)(mant >> 13);
        return (uint16_t)((sign << 15) | (exp16 << 10) | mant16);
    }
}

void init_lut(void) {
    // FP8 E4M3 to FP32 LUT
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

// Convert FP8 array to FP16 (scalar, for packing)
void convert_fp8_to_fp16_scalar(const fp8_e4m3_t* src, fp16_t* dst, int64_t count) {
    for (int64_t i = 0; i < count; i++) {
        dst[i] = fp8_to_fp16_lut[src[i]];
    }
}

// Convert FP8 A panel to FP16 packed format [K][MR]
void convert_A_fp8_to_fp16(const fp8_e4m3_t* A, fp16_t* A_fp16,
                           int64_t panel, int64_t K, int64_t lda, int64_t MR) {
    for (int64_t k = 0; k < K; k++) {
        for (int64_t m = 0; m < MR; m++) {
            A_fp16[k * MR + m] = fp8_to_fp16_lut[A[(panel*MR+m)*lda + k]];
        }
    }
}

// Convert FP8 A full matrix to FP16 packed format
void convert_A_full_fp8_to_fp16(const fp8_e4m3_t* A, fp16_t* A_fp16,
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

// Convert FP8 A panel to FP32 packed format (for comparison)
void convert_A_fp8_to_fp32(const fp8_e4m3_t* A, float* A_fp32,
                           int64_t panel, int64_t K, int64_t lda, int64_t MR) {
    for (int64_t k = 0; k < K; k++) {
        for (int64_t m = 0; m < MR; m++) {
            uint32_t bits = fp8_to_fp32_lut[A[(panel*MR+m)*lda + k]];
            A_fp32[k * MR + m] = *((float*)&bits);
        }
    }
}

// SVE conversion of B from FP8 to FP32
void convert_B_fp8_to_fp32(const fp8_e4m3_t* B, float* B_fp32, int64_t K, int64_t N) {
    int64_t vl = svcntw();
    for (int64_t k = 0; k < K; k++) {
        const fp8_e4m3_t* src = B + k * N;
        float* dst = B_fp32 + k * N;
        for (int64_t n = 0; n < N; n += vl) {
            svbool_t pg = svwhilelt_b32(n, N);
            svuint32_t indices = svld1ub_u32(pg, src + n);
            svuint32_t fp32_bits = svld1_gather_index(pg, fp8_to_fp32_lut, indices);
            svst1(pg, dst + n, svreinterpret_f32(fp32_bits));
        }
    }
}

// External kernels
extern void fp8_gemm_kernel_asm(const float* A, const float* B, float* C,
                                 int64_t ldc_elem, int64_t K);

extern void micro_kernel_fp16fp32_8x3(const fp16_t* A, const float* B, float* C,
                                       int64_t K, int64_t unused, int64_t ldc_bytes);

int main() {
    printf("=== FP8 via FP16 Intermediate Path ===\n\n");
    init_lut();

    const int64_t MR = 8, NR = 3, VL = 16;
    int64_t M = 384, K = 512;
    int64_t N = NR * VL;  // 48
    int64_t M_panels = M / MR;
    int64_t ldc_bytes = N * sizeof(float);

    double flops_panel = 2.0 * MR * N * K;
    double flops_gemm = 2.0 * M * N * K;
    double peak_gflops = 128.0;

    printf("M=%ld, N=%ld, K=%ld, panels=%ld\n", M, N, K, M_panels);
    printf("FLOPs/panel=%.0f, FLOPs/GEMM=%.0fM\n\n", flops_panel, flops_gemm/1e6);

    // Allocate
    fp8_e4m3_t* A_fp8 = aligned_alloc(ALIGN, M * K);
    fp8_e4m3_t* B_fp8 = aligned_alloc(ALIGN, K * N);
    float* B_fp32 = aligned_alloc(ALIGN, K * N * sizeof(float));
    float* A_fp32 = aligned_alloc(ALIGN, MR * K * sizeof(float));
    fp16_t* A_fp16 = aligned_alloc(ALIGN, MR * K * sizeof(fp16_t));
    fp16_t* A_fp16_full = aligned_alloc(ALIGN, M * K * sizeof(fp16_t));
    float* C = aligned_alloc(ALIGN, M * N * sizeof(float));

    // Initialize
    srand(42);
    for (int64_t i = 0; i < M * K; i++) {
        A_fp8[i] = ((rand() % 14 + 1) << 3) | (rand() & 7);
    }
    for (int64_t i = 0; i < K * N; i++) {
        B_fp8[i] = ((rand() % 14 + 1) << 3) | (rand() & 7);
    }

    // Pre-convert B (always done once)
    convert_B_fp8_to_fp32(B_fp8, B_fp32, K, N);

    int iters = 100;
    uint64_t t0, t1, total;

    printf("=== Conversion Cost Comparison ===\n\n");

    // Test FP8 -> FP32 conversion cost
    printf("FP8 -> FP32 (1 panel × %d):\n", iters);
    t0 = get_ticks();
    for (int it = 0; it < iters; it++) {
        convert_A_fp8_to_fp32(A_fp8, A_fp32, 0, K, K, MR);
    }
    t1 = get_ticks();
    total = t1 - t0;
    double fp32_conv_cycles = (double)total / iters * 20;
    printf("   %.0f cycles (%.2f cycles/elem)\n", fp32_conv_cycles, fp32_conv_cycles / (MR * K));

    // Test FP8 -> FP16 conversion cost
    printf("FP8 -> FP16 (1 panel × %d):\n", iters);
    t0 = get_ticks();
    for (int it = 0; it < iters; it++) {
        convert_A_fp8_to_fp16(A_fp8, A_fp16, 0, K, K, MR);
    }
    t1 = get_ticks();
    total = t1 - t0;
    double fp16_conv_cycles = (double)total / iters * 20;
    printf("   %.0f cycles (%.2f cycles/elem)\n\n", fp16_conv_cycles, fp16_conv_cycles / (MR * K));

    printf("=== Kernel Comparison ===\n\n");

    // FP32 kernel (baseline)
    printf("FP32 kernel (1 panel × %d):\n", iters);
    convert_A_fp8_to_fp32(A_fp8, A_fp32, 0, K, K, MR);
    memset(C, 0, M * N * sizeof(float));

    t0 = get_ticks();
    for (int it = 0; it < iters; it++) {
        fp8_gemm_kernel_asm(A_fp32, B_fp32, C, N, K);
    }
    t1 = get_ticks();
    total = t1 - t0;
    double fp32_kernel_cycles = (double)total / iters * 20;
    double fp32_gflops = flops_panel * 2000.0 / (fp32_kernel_cycles * 1e3);
    printf("   %.0f cycles, %.2f GFLOPS (%.1f%%)\n",
           fp32_kernel_cycles, fp32_gflops, 100.0 * fp32_gflops / peak_gflops);

    // FP16->FP32 kernel
    printf("FP16->FP32 kernel (1 panel × %d):\n", iters);
    convert_A_fp8_to_fp16(A_fp8, A_fp16, 0, K, K, MR);
    memset(C, 0, M * N * sizeof(float));

    t0 = get_ticks();
    for (int it = 0; it < iters; it++) {
        micro_kernel_fp16fp32_8x3(A_fp16, B_fp32, C, K, 0, ldc_bytes);
    }
    t1 = get_ticks();
    total = t1 - t0;
    double fp16_kernel_cycles = (double)total / iters * 20;
    double fp16_gflops = flops_panel * 2000.0 / (fp16_kernel_cycles * 1e3);
    printf("   %.0f cycles, %.2f GFLOPS (%.1f%%)\n\n",
           fp16_kernel_cycles, fp16_gflops, 100.0 * fp16_gflops / peak_gflops);

    printf("=== Full GEMM Comparison ===\n\n");

    // Approach 1: FP8 -> FP32 per-panel + FP32 kernel
    printf("Approach 1: FP8->FP32 per-panel + FP32 kernel\n");
    memset(C, 0, M * N * sizeof(float));
    t0 = get_ticks();
    for (int it = 0; it < iters; it++) {
        for (int64_t p = 0; p < M_panels; p++) {
            convert_A_fp8_to_fp32(A_fp8, A_fp32, p, K, K, MR);
            fp8_gemm_kernel_asm(A_fp32, B_fp32, C + p * MR * N, N, K);
        }
    }
    t1 = get_ticks();
    total = t1 - t0;
    double a1_cycles = (double)total / iters * 20;
    double a1_gflops = flops_gemm * 2000.0 / (a1_cycles * 1e3);
    printf("   %.0f cycles, %.2f GFLOPS (%.1f%%)\n", a1_cycles, a1_gflops, 100.0 * a1_gflops / peak_gflops);

    // Approach 2: FP8 -> FP16 per-panel + FP16->FP32 kernel
    printf("Approach 2: FP8->FP16 per-panel + FP16->FP32 kernel\n");
    memset(C, 0, M * N * sizeof(float));
    t0 = get_ticks();
    for (int it = 0; it < iters; it++) {
        for (int64_t p = 0; p < M_panels; p++) {
            convert_A_fp8_to_fp16(A_fp8, A_fp16, p, K, K, MR);
            micro_kernel_fp16fp32_8x3(A_fp16, B_fp32, C + p * MR * N, K, 0, ldc_bytes);
        }
    }
    t1 = get_ticks();
    total = t1 - t0;
    double a2_cycles = (double)total / iters * 20;
    double a2_gflops = flops_gemm * 2000.0 / (a2_cycles * 1e3);
    printf("   %.0f cycles, %.2f GFLOPS (%.1f%%)\n\n", a2_cycles, a2_gflops, 100.0 * a2_gflops / peak_gflops);

    // Approach 3: Pre-convert all A to FP16, then run FP16->FP32 kernel
    printf("Approach 3: Pre-convert A to FP16 + FP16->FP32 kernel\n");

    t0 = get_ticks();
    convert_A_full_fp8_to_fp16(A_fp8, A_fp16_full, M, K, K, MR);
    t1 = get_ticks();
    uint64_t preconv_ticks = t1 - t0;

    memset(C, 0, M * N * sizeof(float));
    t0 = get_ticks();
    for (int it = 0; it < iters; it++) {
        for (int64_t p = 0; p < M_panels; p++) {
            fp16_t* A_panel = A_fp16_full + p * MR * K;
            micro_kernel_fp16fp32_8x3(A_panel, B_fp32, C + p * MR * N, K, 0, ldc_bytes);
        }
    }
    t1 = get_ticks();
    total = t1 - t0;
    double a3_pure = (double)total / iters * 20;
    double a3_amort = ((double)total / iters + (double)preconv_ticks / iters) * 20;
    double a3_pure_gflops = flops_gemm * 2000.0 / (a3_pure * 1e3);
    double a3_amort_gflops = flops_gemm * 2000.0 / (a3_amort * 1e3);
    printf("   Pre-conversion: %lu ticks\n", preconv_ticks);
    printf("   Pure GEMM: %.0f cycles, %.2f GFLOPS (%.1f%%)\n", a3_pure, a3_pure_gflops, 100.0 * a3_pure_gflops / peak_gflops);
    printf("   Amortized: %.0f cycles, %.2f GFLOPS (%.1f%%)\n\n", a3_amort, a3_amort_gflops, 100.0 * a3_amort_gflops / peak_gflops);

    printf("=== Summary ===\n");
    printf("Conversion costs:\n");
    printf("  FP8->FP32: %.2f cycles/elem\n", fp32_conv_cycles / (MR * K));
    printf("  FP8->FP16: %.2f cycles/elem\n", fp16_conv_cycles / (MR * K));
    printf("\nKernel efficiency:\n");
    printf("  FP32:         %.1f%%\n", 100.0 * fp32_gflops / peak_gflops);
    printf("  FP16->FP32:   %.1f%%\n", 100.0 * fp16_gflops / peak_gflops);
    printf("\nFull GEMM efficiency:\n");
    printf("  FP8->FP32 per-panel: %.1f%%\n", 100.0 * a1_gflops / peak_gflops);
    printf("  FP8->FP16 per-panel: %.1f%%\n", 100.0 * a2_gflops / peak_gflops);
    printf("  Pre-conv FP16 (pure): %.1f%%\n", 100.0 * a3_pure_gflops / peak_gflops);
    printf("  Pre-conv FP16 (amort): %.1f%%\n", 100.0 * a3_amort_gflops / peak_gflops);

    free(A_fp8); free(B_fp8); free(B_fp32);
    free(A_fp32); free(A_fp16); free(A_fp16_full); free(C);
    return 0;
}
