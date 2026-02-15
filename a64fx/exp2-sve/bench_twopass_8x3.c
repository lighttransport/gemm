/*
 * Two-pass FlashAttention Stage 2: exp2_fast + 8x3 FP32 GEMM
 *
 * Pass 1: Vectorized exp2 (P = exp2(S * scale - max))
 * Pass 2: High-efficiency 8x3 FP32 GEMM (O = P @ V)
 *
 * Target: Maximize throughput using 8x3 kernel (~92% peak for GEMM)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

/* exp2 kernel - vectorized across rows, row-major output */
extern void exp2_fast_rows(
    const int32_t* S, float* P, int M, int Nc,
    float scale, float neg_max, int ld_s, int ld_p);

/* exp2 kernel - column-major output for 8 rows (no pack needed) */
extern void exp2_colmajor_8row(
    const int32_t* S, float* P_packed, int Nc,
    float scale, float neg_max);

/* 8x3 FP32 GEMM microkernel
 * void micro_kernel_fp32_8x3_unroll4(
 *     const float* A,  // x0: column-major packed [K][8]
 *     const float* B,  // x1: row-major [K][48] (3 SVE vectors)
 *     float* C,        // x2: row-major [8][48]
 *     int K,           // x3: must be multiple of 4
 *     int unused,      // x4
 *     int ld_c         // x5: C stride in bytes
 * );
 */
extern void micro_kernel_fp32_8x3_unroll4(
    const float* A, const float* B, float* C,
    int K, int unused, int ld_c);

static inline uint64_t get_cycles(void) {
    uint64_t val;
    __asm__ volatile("mrs %0, cntvct_el0" : "=r"(val));
    return val;
}

/* Pack P from row-major [M][Nc] to column-major [Nc][M] for 8x3 kernel */
static void pack_p_colmajor(const float* P, float* P_packed, int M, int Nc) {
    for (int k = 0; k < Nc; k++) {
        for (int m = 0; m < M; m++) {
            P_packed[k * M + m] = P[m * Nc + k];
        }
    }
}

/* Reference implementation */
static void ref_exp2_gemm(const int32_t* S, const float* V, float* O,
                          int M, int Nc, int D,
                          float scale, float max_val) {
    for (int i = 0; i < M * D; i++) O[i] = 0.0f;
    for (int i = 0; i < M; i++) {
        for (int k = 0; k < Nc; k++) {
            float p = exp2f((float)S[i * Nc + k] * scale - max_val);
            for (int j = 0; j < D; j++) {
                O[i * D + j] += p * V[k * D + j];
            }
        }
    }
}

static int verify(const float* ref, const float* test, int n, float* max_err) {
    *max_err = 0.0f;
    int errors = 0;
    for (int i = 0; i < n; i++) {
        float err = fabsf(ref[i] - test[i]);
        float rel = (fabsf(ref[i]) > 1e-6f) ? err / fabsf(ref[i]) : err;
        if (rel > *max_err) *max_err = rel;
        if (rel > 0.05f && err > 1e-3f) errors++;  // Allow 5% for FEXPA approximation
    }
    return errors;
}

int main(int argc, char** argv) {
    int Nc = 512;
    int D = 48;   // 3 SVE vectors = 48 floats for 8x3 kernel
    int iterations = 1000;

    if (argc > 1) Nc = atoi(argv[1]);
    if (argc > 2) D = atoi(argv[2]);
    if (argc > 3) iterations = atoi(argv[3]);

    /* Align dimensions */
    Nc = (Nc / 4) * 4;  // K must be multiple of 4 for 8x3 kernel
    if (Nc < 4) Nc = 4;
    D = (D / 48) * 48;  // D must be multiple of 48 for 8x3 kernel
    if (D < 48) D = 48;

    int M = 8;  // Fixed for 8x3 kernel

    printf("=== Two-Pass FlashAttention: exp2_fast + 8x3 GEMM ===\n");
    printf("M=%d, Nc=%d (K), D=%d\n", M, Nc, D);
    printf("Iterations: %d\n\n", iterations);

    /* Allocate */
    int32_t* S = aligned_alloc(64, M * Nc * sizeof(int32_t));
    float* V = aligned_alloc(64, Nc * D * sizeof(float));
    float* P = aligned_alloc(64, M * Nc * sizeof(float));         // Row-major
    float* P_packed = aligned_alloc(64, Nc * M * sizeof(float));  // Col-major for GEMM
    float* O_ref = aligned_alloc(64, M * D * sizeof(float));
    float* O_test = aligned_alloc(64, M * D * sizeof(float));

    /* Initialize */
    srand(42);
    for (int i = 0; i < M * Nc; i++) S[i] = (rand() % 201) - 100;
    for (int i = 0; i < Nc * D; i++) V[i] = ((float)(rand() % 1000) - 500) / 500.0f;

    float scale = 1.0f / 64.0f;
    float max_val = 1.5f;
    float neg_max = -max_val;

    /* Reference */
    ref_exp2_gemm(S, V, O_ref, M, Nc, D, scale, max_val);

    double timer_freq = 100e6;
    double cpu_freq = 2e9;
    double fp32_peak = 128.0;  // GFLOPS
    long long total_fmla_ops = (long long)M * Nc * D * 2;

    volatile uint64_t start, end;
    double elapsed, cycles, gflops;
    float max_err;
    int errors;

    int n_tiles = D / 48;
    int ld_c = D * sizeof(float);

    /*========================================================================
     * Test 1: Two-pass (exp2_fast_rows + pack + 8x3 GEMM)
     *========================================================================*/
    printf("=== Two-Pass: exp2_fast + pack + 8x3 GEMM ===\n");

    /* Correctness check */
    memset(O_test, 0, M * D * sizeof(float));

    /* Pass 1: exp2 */
    exp2_fast_rows(S, P, M, Nc, scale, neg_max, Nc, Nc);

    /* Pack P for 8x3 kernel */
    pack_p_colmajor(P, P_packed, M, Nc);

    /* Pass 2: 8x3 GEMM for each tile */
    for (int tile = 0; tile < n_tiles; tile++) {
        float* V_tile = V + tile * 48;
        float* O_tile = O_test + tile * 48;
        micro_kernel_fp32_8x3_unroll4(P_packed, V_tile, O_tile, Nc, 0, ld_c);
    }

    errors = verify(O_ref, O_test, M * D, &max_err);
    printf("  Correctness: max_error=%.4f%% %s\n", max_err * 100, errors == 0 ? "PASS" : "FAIL");

    /* Warmup */
    for (int i = 0; i < 10; i++) {
        exp2_fast_rows(S, P, M, Nc, scale, neg_max, Nc, Nc);
        pack_p_colmajor(P, P_packed, M, Nc);
        for (int tile = 0; tile < n_tiles; tile++) {
            float* V_tile = V + tile * 48;
            float* O_tile = O_test + tile * 48;
            micro_kernel_fp32_8x3_unroll4(P_packed, V_tile, O_tile, Nc, 0, ld_c);
        }
    }

    /* Benchmark full two-pass */
    start = get_cycles();
    for (int i = 0; i < iterations; i++) {
        exp2_fast_rows(S, P, M, Nc, scale, neg_max, Nc, Nc);
        pack_p_colmajor(P, P_packed, M, Nc);
        for (int tile = 0; tile < n_tiles; tile++) {
            float* V_tile = V + tile * 48;
            float* O_tile = O_test + tile * 48;
            micro_kernel_fp32_8x3_unroll4(P_packed, V_tile, O_tile, Nc, 0, ld_c);
        }
    }
    end = get_cycles();

    elapsed = (double)(end - start);
    cycles = elapsed * (cpu_freq / timer_freq) / iterations;
    gflops = (double)total_fmla_ops * iterations / (elapsed / timer_freq) / 1e9;
    printf("  Total cycles: %.1f\n", cycles);
    printf("  GFLOPS: %.2f (%.1f%% of FP32 peak)\n", gflops, gflops / fp32_peak * 100);
    printf("  Cycles/K: %.2f\n\n", cycles / Nc);

    /*========================================================================
     * Test 1b: Two-pass with fused exp2_colmajor (no pack)
     *========================================================================*/
    printf("=== Two-Pass: exp2_colmajor (no pack) + 8x3 GEMM ===\n");

    memset(O_test, 0, M * D * sizeof(float));

    /* Fused exp2 + transpose */
    exp2_colmajor_8row(S, P_packed, Nc, scale, neg_max);

    /* 8x3 GEMM */
    for (int tile = 0; tile < n_tiles; tile++) {
        float* V_tile = V + tile * 48;
        float* O_tile = O_test + tile * 48;
        micro_kernel_fp32_8x3_unroll4(P_packed, V_tile, O_tile, Nc, 0, ld_c);
    }

    errors = verify(O_ref, O_test, M * D, &max_err);
    printf("  Correctness: max_error=%.4f%% %s\n", max_err * 100, errors == 0 ? "PASS" : "FAIL");

    /* Warmup */
    for (int i = 0; i < 10; i++) {
        exp2_colmajor_8row(S, P_packed, Nc, scale, neg_max);
        for (int tile = 0; tile < n_tiles; tile++) {
            float* V_tile = V + tile * 48;
            float* O_tile = O_test + tile * 48;
            micro_kernel_fp32_8x3_unroll4(P_packed, V_tile, O_tile, Nc, 0, ld_c);
        }
    }

    /* Benchmark fused version */
    start = get_cycles();
    for (int i = 0; i < iterations; i++) {
        exp2_colmajor_8row(S, P_packed, Nc, scale, neg_max);
        for (int tile = 0; tile < n_tiles; tile++) {
            float* V_tile = V + tile * 48;
            float* O_tile = O_test + tile * 48;
            micro_kernel_fp32_8x3_unroll4(P_packed, V_tile, O_tile, Nc, 0, ld_c);
        }
    }
    end = get_cycles();

    double fused_elapsed = (double)(end - start);
    double fused_cycles = fused_elapsed * (cpu_freq / timer_freq) / iterations;
    double fused_gflops = (double)total_fmla_ops * iterations / (fused_elapsed / timer_freq) / 1e9;
    printf("  Total cycles: %.1f\n", fused_cycles);
    printf("  GFLOPS: %.2f (%.1f%% of FP32 peak)\n", fused_gflops, fused_gflops / fp32_peak * 100);
    printf("  Cycles/K: %.2f\n\n", fused_cycles / Nc);

    /* Measure exp2_colmajor alone */
    start = get_cycles();
    for (int i = 0; i < iterations; i++) {
        exp2_colmajor_8row(S, P_packed, Nc, scale, neg_max);
    }
    end = get_cycles();
    double exp2cm_cycles = (end - start) * (cpu_freq / timer_freq) / iterations;
    printf("  exp2_colmajor_8row: %.1f cycles (%.2f cycles/element)\n\n",
           exp2cm_cycles, exp2cm_cycles / (M * Nc));

    /*========================================================================
     * Test 2: Measure each phase separately
     *========================================================================*/
    printf("=== Phase Breakdown (separate exp2 + pack) ===\n");

    /* exp2 only */
    start = get_cycles();
    for (int i = 0; i < iterations; i++) {
        exp2_fast_rows(S, P, M, Nc, scale, neg_max, Nc, Nc);
    }
    end = get_cycles();
    double exp2_cycles = (end - start) * (cpu_freq / timer_freq) / iterations;
    printf("  exp2_fast_rows: %.1f cycles (%.2f cycles/element)\n",
           exp2_cycles, exp2_cycles / (M * Nc));

    /* Pack only */
    start = get_cycles();
    for (int i = 0; i < iterations; i++) {
        pack_p_colmajor(P, P_packed, M, Nc);
    }
    end = get_cycles();
    double pack_cycles = (end - start) * (cpu_freq / timer_freq) / iterations;
    printf("  pack_p_colmajor: %.1f cycles (%.2f cycles/element)\n",
           pack_cycles, pack_cycles / (M * Nc));

    /* GEMM only */
    start = get_cycles();
    for (int i = 0; i < iterations; i++) {
        for (int tile = 0; tile < n_tiles; tile++) {
            float* V_tile = V + tile * 48;
            float* O_tile = O_test + tile * 48;
            micro_kernel_fp32_8x3_unroll4(P_packed, V_tile, O_tile, Nc, 0, ld_c);
        }
    }
    end = get_cycles();
    double gemm_cycles = (end - start) * (cpu_freq / timer_freq) / iterations;
    double gemm_gflops = (double)total_fmla_ops * iterations / ((end - start) / timer_freq) / 1e9;
    printf("  8x3 GEMM: %.1f cycles, %.2f GFLOPS (%.1f%% of peak)\n",
           gemm_cycles, gemm_gflops, gemm_gflops / fp32_peak * 100);
    printf("  GEMM cycles/K: %.2f\n\n", gemm_cycles / Nc);

    /*========================================================================
     * Test 3: Pure GEMM (no exp2, no pack) - Upper bound
     *========================================================================*/
    printf("=== Pure 8x3 GEMM (upper bound) ===\n");

    /* Pre-pack P with 1.0f */
    for (int i = 0; i < Nc * M; i++) P_packed[i] = 1.0f;

    start = get_cycles();
    for (int i = 0; i < iterations; i++) {
        for (int tile = 0; tile < n_tiles; tile++) {
            float* V_tile = V + tile * 48;
            float* O_tile = O_test + tile * 48;
            micro_kernel_fp32_8x3_unroll4(P_packed, V_tile, O_tile, Nc, 0, ld_c);
        }
    }
    end = get_cycles();

    double pure_gemm_cycles = (end - start) * (cpu_freq / timer_freq) / iterations;
    double pure_gemm_gflops = (double)total_fmla_ops * iterations / ((end - start) / timer_freq) / 1e9;
    printf("  Pure GEMM: %.1f cycles, %.2f GFLOPS (%.1f%% of peak)\n",
           pure_gemm_cycles, pure_gemm_gflops, pure_gemm_gflops / fp32_peak * 100);
    printf("  Cycles/K: %.2f\n\n", pure_gemm_cycles / Nc);

    /*========================================================================
     * Analysis
     *========================================================================*/
    printf("=== Analysis ===\n");
    printf("Phase breakdown:\n");
    printf("  exp2:  %.1f cycles (%.1f%%)\n", exp2_cycles, exp2_cycles / cycles * 100);
    printf("  pack:  %.1f cycles (%.1f%%)\n", pack_cycles, pack_cycles / cycles * 100);
    printf("  GEMM:  %.1f cycles (%.1f%%)\n", gemm_cycles, gemm_cycles / cycles * 100);
    printf("  Total: %.1f cycles\n\n", cycles);

    printf("Overhead vs pure GEMM:\n");
    printf("  exp2 overhead: %.1f%% of GEMM time\n", exp2_cycles / pure_gemm_cycles * 100);
    printf("  pack overhead: %.1f%% of GEMM time\n", pack_cycles / pure_gemm_cycles * 100);
    printf("  Total overhead: %.1f%% of GEMM time\n",
           (exp2_cycles + pack_cycles) / pure_gemm_cycles * 100);

    printf("\nEffective throughput:\n");
    printf("  Two-pass: %.1f%% of pure GEMM\n", pure_gemm_cycles / cycles * 100);
    printf("  Two-pass: %.1f%% of FP32 peak\n", gflops / fp32_peak * 100);

    free(S);
    free(V);
    free(P);
    free(P_packed);
    free(O_ref);
    free(O_test);

    return 0;
}
