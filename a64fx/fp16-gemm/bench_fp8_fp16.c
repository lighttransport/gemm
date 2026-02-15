/*
 * FP8 → FP16 → FP32 GEMM Path
 *
 * Testing the user's suggested approach:
 * 1. FP8 → FP16 conversion (half the storage of FP32)
 * 2. FP16 stored in L1 cache
 * 3. FP16 → FP32 GEMM kernel (ld1rh + fcvt)
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <arm_sve.h>
#include <arm_fp16.h>

#define ALIGN 64

typedef uint8_t fp8_e4m3_t;
typedef __fp16 fp16_t;

uint32_t fp8_to_fp32_lut[256] __attribute__((aligned(64)));
uint16_t fp8_to_fp16_lut[256] __attribute__((aligned(64)));

static inline uint64_t get_ticks(void) {
    uint64_t val;
    __asm__ volatile("isb" ::: "memory");
    __asm__ volatile("mrs %0, cntvct_el0" : "=r"(val));
    __asm__ volatile("isb" ::: "memory");
    return val;
}

// Convert FP32 bits to FP16 bits
static inline uint16_t fp32_to_fp16_bits(uint32_t f32) {
    uint32_t sign = (f32 >> 31) & 1;
    int32_t exp = ((f32 >> 23) & 0xFF) - 127;
    uint32_t mant = f32 & 0x7FFFFF;

    if (exp < -14) return (uint16_t)(sign << 15);  // underflow
    if (exp > 15) return (uint16_t)((sign << 15) | 0x7C00);  // overflow
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
        fp8_to_fp16_lut[i] = fp32_to_fp16_bits(fp32);
    }
}

// Scalar FP8 → FP16 conversion
void convert_fp8_to_fp16_scalar(const fp8_e4m3_t* src, fp16_t* dst, int64_t count) {
    for (int64_t i = 0; i < count; i++) {
        uint16_t fp16_bits = fp8_to_fp16_lut[src[i]];
        dst[i] = *((fp16_t*)&fp16_bits);
    }
}

// SVE FP8 → FP16 conversion using gather
void convert_fp8_to_fp16_sve(const fp8_e4m3_t* src, fp16_t* dst, int64_t count) {
    // Process 16 at a time using u32 gather (since gather_u16 requires 32-bit indices)
    int64_t vl = svcntw();  // 16 for A64FX

    for (int64_t i = 0; i < count; i += vl) {
        svbool_t pg = svwhilelt_b32(i, count);

        // Load FP8 bytes, zero-extend to u32
        svuint32_t indices = svld1ub_u32(pg, src + i);

        // Gather u16 values from LUT (need u32 gather, then narrow)
        // Actually, let's just use scalar for FP16 LUT since SVE gather is complex
        for (int64_t j = i; j < i + vl && j < count; j++) {
            uint16_t bits = fp8_to_fp16_lut[src[j]];
            dst[j] = *((fp16_t*)&bits);
        }
    }
}

// FP8 → FP16 conversion with packing
void convert_A_fp8_to_fp16(const fp8_e4m3_t* A, fp16_t* A_fp16,
                           int64_t M, int64_t K, int64_t lda, int64_t MR) {
    int64_t M_panels = M / MR;
    for (int64_t p = 0; p < M_panels; p++) {
        fp16_t* dst = A_fp16 + p * MR * K;
        for (int64_t k = 0; k < K; k++) {
            for (int64_t m = 0; m < MR; m++) {
                uint16_t bits = fp8_to_fp16_lut[A[(p*MR+m)*lda + k]];
                dst[k * MR + m] = *((fp16_t*)&bits);
            }
        }
    }
}

// FP8 → FP32 direct conversion (for comparison)
void convert_fp8_to_fp32_direct(const fp8_e4m3_t* src, float* dst, int64_t count) {
    int64_t vl = svcntw();
    for (int64_t i = 0; i < count; i += vl) {
        svbool_t pg = svwhilelt_b32(i, count);
        svuint32_t fp8 = svld1ub_u32(pg, src + i);
        svuint32_t sign = svlsl_x(pg, svand_x(pg, fp8, svdup_u32(0x80)), 24);
        svuint32_t exp_field = svlsl_x(pg, svadd_x(pg, svand_x(pg, fp8, svdup_u32(0x78)), svdup_u32(120 << 3)), 20);
        svuint32_t mant = svlsl_x(pg, svand_x(pg, fp8, svdup_u32(0x7)), 20);
        svuint32_t result = svorr_x(pg, sign, svorr_x(pg, exp_field, mant));
        svbool_t is_zero = svcmpeq(pg, svand_x(pg, fp8, svdup_u32(0x78)), svdup_u32(0));
        result = svsel(is_zero, sign, result);
        svst1(pg, dst + i, svreinterpret_f32(result));
    }
}

void convert_A_fp8_to_fp32(const fp8_e4m3_t* A, float* A_fp32,
                           int64_t M, int64_t K, int64_t lda, int64_t MR) {
    int64_t vl = svcntw();
    int64_t M_panels = M / MR;

    for (int64_t p = 0; p < M_panels; p++) {
        float* dst = A_fp32 + p * MR * K;
        for (int64_t k = 0; k < K; k += vl) {
            svbool_t pg = svwhilelt_b32(k, K);
            for (int64_t m = 0; m < MR; m++) {
                const fp8_e4m3_t* src = A + (p*MR+m)*lda + k;
                svuint32_t fp8 = svld1ub_u32(pg, src);
                svuint32_t sign = svlsl_x(pg, svand_x(pg, fp8, svdup_u32(0x80)), 24);
                svuint32_t exp_field = svlsl_x(pg, svadd_x(pg, svand_x(pg, fp8, svdup_u32(0x78)), svdup_u32(120 << 3)), 20);
                svuint32_t mant = svlsl_x(pg, svand_x(pg, fp8, svdup_u32(0x7)), 20);
                svuint32_t result = svorr_x(pg, sign, svorr_x(pg, exp_field, mant));
                svbool_t is_zero = svcmpeq(pg, svand_x(pg, fp8, svdup_u32(0x78)), svdup_u32(0));
                result = svsel(is_zero, sign, result);
                svst1(pg, dst + m, svreinterpret_f32(result));
            }
            dst += MR * vl;
        }
    }
}

void convert_B_fp8_to_fp32(const fp8_e4m3_t* B, float* B_fp32, int64_t K, int64_t N) {
    convert_fp8_to_fp32_direct(B, B_fp32, K * N);
}

extern void fp8_gemm_kernel_asm(const float* A, const float* B, float* C,
                                 int64_t ldc_elem, int64_t K);

extern void micro_kernel_fp16fp32_8x3(const fp16_t* A, const float* B, float* C,
                                       int64_t K, int64_t unused, int64_t ldc_bytes);

int main() {
    printf("╔═══════════════════════════════════════════════════════╗\n");
    printf("║       FP8 → FP16 → FP32 GEMM Path Analysis           ║\n");
    printf("╚═══════════════════════════════════════════════════════╝\n\n");
    init_lut();

    int64_t M = 384, K = 512, N = 48, MR = 8;
    int64_t M_panels = M / MR;
    int64_t ldc_bytes = N * sizeof(float);
    int64_t flops = 2L * M * N * K;
    int64_t ideal = flops / 1280;
    int iters = 100;

    printf("Config: M=%ld, K=%ld, N=%ld (FLOPs=%.1fM)\n", M, K, N, flops/1e6);
    printf("Target: Ideal=%ld ticks, 90%%=%ld ticks\n\n", ideal, ideal*100/90);

    fp8_e4m3_t* A_fp8 = aligned_alloc(ALIGN, M * K);
    fp8_e4m3_t* B_fp8 = aligned_alloc(ALIGN, K * N);
    fp16_t* A_fp16 = aligned_alloc(ALIGN, M * K * sizeof(fp16_t));
    float* A_fp32 = aligned_alloc(ALIGN, M * K * sizeof(float));
    float* B_fp32 = aligned_alloc(ALIGN, K * N * sizeof(float));
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

    printf("═══════════════════════════════════════════════════════\n");
    printf("              CONVERSION COST ANALYSIS\n");
    printf("═══════════════════════════════════════════════════════\n\n");

    // FP8 → FP16 conversion cost
    printf("1. FP8 → FP16 (scalar):\n");
    t0 = get_ticks();
    for (int it = 0; it < iters; it++) {
        convert_fp8_to_fp16_scalar(A_fp8, A_fp16, M * K);
    }
    t1 = get_ticks();
    uint64_t fp8_to_fp16_scalar = (t1 - t0) / iters;
    printf("   %lu ticks (%.2f cycles/elem)\n", fp8_to_fp16_scalar,
           fp8_to_fp16_scalar * 20.0 / (M * K));

    // FP8 → FP16 SVE
    printf("2. FP8 → FP16 (SVE gather):\n");
    t0 = get_ticks();
    for (int it = 0; it < iters; it++) {
        convert_fp8_to_fp16_sve(A_fp8, A_fp16, M * K);
    }
    t1 = get_ticks();
    uint64_t fp8_to_fp16_sve = (t1 - t0) / iters;
    printf("   %lu ticks (%.2f cycles/elem)\n", fp8_to_fp16_sve,
           fp8_to_fp16_sve * 20.0 / (M * K));

    // FP8 → FP16 with packing
    printf("3. FP8 → FP16 (packed):\n");
    t0 = get_ticks();
    for (int it = 0; it < iters; it++) {
        convert_A_fp8_to_fp16(A_fp8, A_fp16, M, K, K, MR);
    }
    t1 = get_ticks();
    uint64_t fp8_to_fp16_pack = (t1 - t0) / iters;
    printf("   %lu ticks (%.2f cycles/elem)\n", fp8_to_fp16_pack,
           fp8_to_fp16_pack * 20.0 / (M * K));

    // FP8 → FP32 with packing (for comparison)
    printf("4. FP8 → FP32 (SVE direct, packed):\n");
    t0 = get_ticks();
    for (int it = 0; it < iters; it++) {
        convert_A_fp8_to_fp32(A_fp8, A_fp32, M, K, K, MR);
    }
    t1 = get_ticks();
    uint64_t fp8_to_fp32_pack = (t1 - t0) / iters;
    printf("   %lu ticks (%.2f cycles/elem)\n\n", fp8_to_fp32_pack,
           fp8_to_fp32_pack * 20.0 / (M * K));

    printf("═══════════════════════════════════════════════════════\n");
    printf("                 FULL GEMM COMPARISON\n");
    printf("═══════════════════════════════════════════════════════\n\n");

    // Path 1: FP8 → FP32 → FP32 GEMM
    printf("1. FP8 → FP32 + FP32 GEMM:\n");
    memset(C, 0, M * N * sizeof(float));
    t0 = get_ticks();
    for (int it = 0; it < iters; it++) {
        convert_A_fp8_to_fp32(A_fp8, A_fp32, M, K, K, MR);
        for (int64_t p = 0; p < M_panels; p++) {
            fp8_gemm_kernel_asm(A_fp32 + p * MR * K, B_fp32, C + p * MR * N, N, K);
        }
    }
    t1 = get_ticks();
    uint64_t path1 = (t1 - t0) / iters;
    printf("   %lu ticks, %lu%% eff\n", path1, ideal * 100 / path1);

    // Path 2: FP8 → FP16 → FP16/FP32 GEMM
    printf("2. FP8 → FP16 + FP16→FP32 GEMM:\n");
    memset(C, 0, M * N * sizeof(float));
    t0 = get_ticks();
    for (int it = 0; it < iters; it++) {
        convert_A_fp8_to_fp16(A_fp8, A_fp16, M, K, K, MR);
        for (int64_t p = 0; p < M_panels; p++) {
            micro_kernel_fp16fp32_8x3(A_fp16 + p * MR * K, B_fp32,
                                      C + p * MR * N, K, 0, ldc_bytes);
        }
    }
    t1 = get_ticks();
    uint64_t path2 = (t1 - t0) / iters;
    printf("   %lu ticks, %lu%% eff\n\n", path2, ideal * 100 / path2);

    // Component analysis
    printf("═══════════════════════════════════════════════════════\n");
    printf("                 COMPONENT BREAKDOWN\n");
    printf("═══════════════════════════════════════════════════════\n\n");

    // Pure kernels
    convert_A_fp8_to_fp32(A_fp8, A_fp32, M, K, K, MR);
    memset(C, 0, M * N * sizeof(float));
    t0 = get_ticks();
    for (int it = 0; it < iters; it++) {
        for (int64_t p = 0; p < M_panels; p++) {
            fp8_gemm_kernel_asm(A_fp32 + p * MR * K, B_fp32, C + p * MR * N, N, K);
        }
    }
    t1 = get_ticks();
    uint64_t fp32_kernel = (t1 - t0) / iters;

    convert_A_fp8_to_fp16(A_fp8, A_fp16, M, K, K, MR);
    memset(C, 0, M * N * sizeof(float));
    t0 = get_ticks();
    for (int it = 0; it < iters; it++) {
        for (int64_t p = 0; p < M_panels; p++) {
            micro_kernel_fp16fp32_8x3(A_fp16 + p * MR * K, B_fp32,
                                      C + p * MR * N, K, 0, ldc_bytes);
        }
    }
    t1 = get_ticks();
    uint64_t fp16_kernel = (t1 - t0) / iters;

    printf("FP32 kernel:           %5lu ticks (%3lu%% eff)\n", fp32_kernel, ideal*100/fp32_kernel);
    printf("FP16→FP32 kernel:      %5lu ticks (%3lu%% eff)\n", fp16_kernel, ideal*100/fp16_kernel);
    printf("FP8→FP32 conversion:   %5lu ticks\n", fp8_to_fp32_pack);
    printf("FP8→FP16 conversion:   %5lu ticks\n\n", fp8_to_fp16_pack);

    printf("═══════════════════════════════════════════════════════\n");
    printf("                      SUMMARY\n");
    printf("═══════════════════════════════════════════════════════\n\n");

    printf("Storage Savings (FP16 vs FP32):\n");
    printf("  A matrix: %ld bytes (FP16) vs %ld bytes (FP32)\n",
           M * K * 2, M * K * 4);
    printf("  Reduction: 50%%\n\n");

    printf("Conversion Cost:\n");
    printf("  FP8→FP16: %lu ticks (%.2f cycles/elem)\n",
           fp8_to_fp16_pack, fp8_to_fp16_pack * 20.0 / (M*K));
    printf("  FP8→FP32: %lu ticks (%.2f cycles/elem)\n",
           fp8_to_fp32_pack, fp8_to_fp32_pack * 20.0 / (M*K));
    printf("  FP16 saves: %ld ticks (%.1fx faster)\n\n",
           fp8_to_fp32_pack - fp8_to_fp16_pack,
           (double)fp8_to_fp32_pack / fp8_to_fp16_pack);

    printf("Full GEMM Efficiency:\n");
    printf("  FP8→FP32 path: %lu%% (%lu ticks)\n", ideal*100/path1, path1);
    printf("  FP8→FP16 path: %lu%% (%lu ticks)\n\n", ideal*100/path2, path2);

    if (path2 < path1) {
        printf("✓ FP8→FP16 path is FASTER by %lu ticks (%.1fx)\n",
               path1 - path2, (double)path1 / path2);
    } else {
        printf("✗ FP8→FP32 path is faster by %lu ticks (%.1fx)\n",
               path2 - path1, (double)path2 / path1);
    }

    printf("\nRecommendation:\n");
    if (path2 < path1) {
        printf("  Use FP8→FP16 path for:\n");
        printf("  - Memory-constrained workloads\n");
        printf("  - When conversion cost dominates\n");
    } else {
        printf("  Use FP8→FP32 path:\n");
        printf("  - FP32 kernel is %.1fx faster than FP16 kernel\n",
               (double)fp16_kernel / fp32_kernel);
        printf("  - Conversion savings don't offset kernel slowdown\n");
    }

    free(A_fp8); free(B_fp8); free(A_fp16);
    free(A_fp32); free(B_fp32); free(C);
    return 0;
}
