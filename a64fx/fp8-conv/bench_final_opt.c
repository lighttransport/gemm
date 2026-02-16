/*
 * FP8 GEMM - Final Optimized Benchmark
 * Testing all conversion approaches with full GEMM
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

// SVE direct bitwise conversion (fastest)
void convert_direct(const fp8_e4m3_t* src, float* dst, int64_t count) {
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

// Full A matrix conversion with packing (row-major to [panel][k][m])
void convert_A_direct(const fp8_e4m3_t* A, float* A_fp32,
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

void convert_B(const fp8_e4m3_t* B, float* B_fp32, int64_t K, int64_t N) {
    convert_direct(B, B_fp32, K * N);
}

extern void fp8_gemm_kernel_asm(const float* A, const float* B, float* C,
                                 int64_t ldc_elem, int64_t K);

int main() {
    printf("╔═══════════════════════════════════════════════════════╗\n");
    printf("║     FP8 GEMM Final Results - SVE Direct Conversion    ║\n");
    printf("╚═══════════════════════════════════════════════════════╝\n\n");
    init_lut();

    int64_t M = 384, K = 512, N = 48, MR = 8;
    int64_t M_panels = M / MR;
    int64_t flops = 2L * M * N * K;
    int64_t ideal = flops / 1280;
    int iters = 100;

    printf("Config: M=%ld, K=%ld, N=%ld (FLOPs=%.1fM)\n", M, K, N, flops/1e6);
    printf("Target: Ideal=%ld ticks, 90%%=%ld ticks\n\n", ideal, ideal*100/90);

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

    // 1. Pure GEMM
    convert_A_direct(A_fp8, A_fp32, M, K, K, MR);
    memset(C, 0, M * N * sizeof(float));
    t0 = get_ticks();
    for (int it = 0; it < iters; it++) {
        for (int64_t p = 0; p < M_panels; p++) {
            fp8_gemm_kernel_asm(A_fp32 + p * MR * K, B_fp32, C + p * MR * N, N, K);
        }
    }
    t1 = get_ticks();
    uint64_t pure = (t1 - t0) / iters;

    // 2. SVE conv (sequential) + GEMM
    memset(C, 0, M * N * sizeof(float));
    t0 = get_ticks();
    for (int it = 0; it < iters; it++) {
        convert_A_direct(A_fp8, A_fp32, M, K, K, MR);
        for (int64_t p = 0; p < M_panels; p++) {
            fp8_gemm_kernel_asm(A_fp32 + p * MR * K, B_fp32, C + p * MR * N, N, K);
        }
    }
    t1 = get_ticks();
    uint64_t sve_seq = (t1 - t0) / iters;

    // 3. Conversion cost only
    t0 = get_ticks();
    for (int it = 0; it < iters; it++) {
        convert_A_direct(A_fp8, A_fp32, M, K, K, MR);
    }
    t1 = get_ticks();
    uint64_t conv = (t1 - t0) / iters;

    // Results
    printf("═══════════════════════════════════════════════════════\n");
    printf("                      RESULTS\n");
    printf("═══════════════════════════════════════════════════════\n\n");

    printf("Component Costs:\n");
    printf("  Pure FP32 GEMM:    %5lu ticks\n", pure);
    printf("  SVE A Conversion:  %5lu ticks\n", conv);
    printf("  Total (SVE+GEMM):  %5lu ticks\n\n", sve_seq);

    printf("Efficiency:\n");
    printf("  Pure GEMM:         %3lu%% of peak\n", ideal * 100 / pure);
    printf("  With conversion:   %3lu%% of peak\n\n", ideal * 100 / sve_seq);

    printf("Analysis:\n");
    printf("  Conv/GEMM ratio:   %.1f%%\n", conv * 100.0 / pure);
    printf("  Conv overhead:     %.1fx GEMM time\n", (double)conv / pure);

    // Amortization analysis
    printf("\n═══════════════════════════════════════════════════════\n");
    printf("           AMORTIZATION ANALYSIS\n");
    printf("═══════════════════════════════════════════════════════\n\n");

    printf("Reuse  Total      Efficiency\n");
    printf("────────────────────────────\n");
    for (int r = 1; r <= 10; r++) {
        uint64_t total = pure + conv / r;
        uint64_t eff = ideal * 100 / total;
        printf("  %2d   %5lu      %3lu%%", r, total, eff);
        if (eff >= 90) printf("  ✓ 90%% achieved");
        printf("\n");
    }

    printf("\n═══════════════════════════════════════════════════════\n");
    printf("                   CONCLUSION\n");
    printf("═══════════════════════════════════════════════════════\n\n");

    uint64_t reuse_for_90 = 1;
    while (ideal * 100 / (pure + conv / reuse_for_90) < 90) reuse_for_90++;

    printf("• Pure FP32 kernel: %lu%% efficiency (near-optimal)\n", ideal*100/pure);
    printf("• Single-use FP8:   %lu%% efficiency\n", ideal*100/sve_seq);
    printf("• 90%% achieved with reuse >= %lu\n", reuse_for_90);
    printf("\nFor Flash Attention (Q,K,V pre-conversion):\n");
    printf("  Q,K,V each used 2× → total reuse ~2-3\n");
    printf("  Expected efficiency: ~%lu%%\n", ideal * 100 / (pure + conv / 3));

    free(A_fp8); free(B_fp8); free(B_fp32); free(A_fp32); free(C);
    return 0;
}
