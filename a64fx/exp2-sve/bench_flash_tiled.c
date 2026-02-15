/*
 * Benchmark: FlashAttention-style exp2 + GEMM with LD1RW
 *
 * Compare:
 * 1. exp2_flash_tiled_4x4 - Basic LD1RW version
 * 2. exp2_flash_ld1rw_4x4 - Optimized store-to-load forwarding
 * 3. Baseline (separate exp2 + GEMM)
 * 4. Pure GEMM (reference)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <arm_sve.h>

#include "exp2_flash_tiled.h"
#include "exp2_fmla_fused.h"

/* Timing */
static inline uint64_t rdcycle(void) {
    uint64_t val;
    asm volatile("mrs %0, cntvct_el0" : "=r"(val));
    return val;
}

static inline uint64_t rdfreq(void) {
    uint64_t val;
    asm volatile("mrs %0, cntfrq_el0" : "=r"(val));
    return val;
}

/* Reference exp2 + GEMM */
void reference_exp2_gemm(
    const int32_t* S, const float* V, float* O, float* P,
    int M, int Nc, int D, float scale, float max_val,
    int ld_s, int ld_v, int ld_o
) {
    /* exp2 */
    for (int m = 0; m < M; m++) {
        for (int k = 0; k < Nc; k++) {
            float x = (float)S[m * ld_s + k] * scale - max_val;
            P[m * Nc + k] = exp2f(x);
        }
    }
    /* GEMM */
    for (int m = 0; m < M; m++) {
        for (int d = 0; d < D; d++) {
            float sum = 0;
            for (int k = 0; k < Nc; k++) {
                sum += P[m * Nc + k] * V[k * ld_v/4 + d];
            }
            O[m * ld_o/4 + d] = sum;
        }
    }
}

int main(int argc, char** argv) {
    const int M = 4;
    const int Nc = argc > 1 ? atoi(argv[1]) : 64;
    const int D = 64;
    const int ld_s = Nc;
    const int ld_v = D * 4;
    const int ld_o = D * 4;

    printf("=== FlashAttention-style exp2 + GEMM Benchmark ===\n");
    printf("M=%d, Nc=%d, D=%d\n", M, Nc, D);
    printf("Key: LD1RW runs on LD/ST pipe, not FLA pipe\n\n");

    /* Allocate aligned buffers */
    int32_t* S = aligned_alloc(64, M * Nc * sizeof(int32_t));
    float* V = aligned_alloc(64, Nc * D * sizeof(float));
    float* O = aligned_alloc(64, M * D * sizeof(float));
    float* O_ref = aligned_alloc(64, M * D * sizeof(float));
    float* P = aligned_alloc(64, M * Nc * sizeof(float));

    if (!S || !V || !O || !O_ref || !P) {
        fprintf(stderr, "Allocation failed\n");
        return 1;
    }

    /* Initialize */
    for (int i = 0; i < M * Nc; i++) {
        S[i] = (rand() % 2000) - 1000;
    }
    for (int i = 0; i < Nc * D; i++) {
        V[i] = (float)(rand() % 100) / 100.0f - 0.5f;
    }

    float scale = 1.0f / 64.0f;
    float max_val = 10.0f;

    /* Reference */
    reference_exp2_gemm(S, V, O_ref, P, M, Nc, D, scale, max_val, ld_s, ld_v, ld_o);

    const int warmup = 100;
    const int iters = 1000;
    uint64_t start, end;
    double cycles, gflops;
    uint64_t freq = rdfreq();
    double total_flops = 2.0 * M * Nc * D;

    /* ============================================================ */
    /* Test 1: exp2_flash_tiled_4x4 */
    /* ============================================================ */
    memset(O, 0, M * D * sizeof(float));
    for (int i = 0; i < warmup; i++) {
        exp2_flash_tiled_4x4(S, V, O, P, Nc, scale, max_val, ld_s, ld_v, ld_o);
    }

    start = rdcycle();
    for (int i = 0; i < iters; i++) {
        exp2_flash_tiled_4x4(S, V, O, P, Nc, scale, max_val, ld_s, ld_v, ld_o);
    }
    end = rdcycle();

    cycles = (double)(end - start) / iters;
    gflops = total_flops / cycles * (freq / 1e9);

    /* Verify */
    float max_err = 0;
    for (int i = 0; i < M * D; i++) {
        float err = fabsf(O[i] - O_ref[i]);
        if (err > max_err) max_err = err;
    }

    printf("1. exp2_flash_tiled_4x4:\n");
    printf("   Cycles: %.1f, GFLOPS: %.2f, Peak%%: %.1f%%\n",
           cycles, gflops, gflops / 128.0 * 100);
    printf("   Max error: %.6f %s\n\n", max_err, max_err < 0.01 ? "OK" : "FAIL");

    /* ============================================================ */
    /* Test 2: exp2_flash_ld1rw_4x4 (optimized) */
    /* ============================================================ */
    memset(O, 0, M * D * sizeof(float));
    for (int i = 0; i < warmup; i++) {
        exp2_flash_ld1rw_4x4(S, V, O, P, Nc, scale, max_val, ld_s, ld_v, ld_o);
    }

    start = rdcycle();
    for (int i = 0; i < iters; i++) {
        exp2_flash_ld1rw_4x4(S, V, O, P, Nc, scale, max_val, ld_s, ld_v, ld_o);
    }
    end = rdcycle();

    cycles = (double)(end - start) / iters;
    gflops = total_flops / cycles * (freq / 1e9);

    /* Verify */
    max_err = 0;
    for (int i = 0; i < M * D; i++) {
        float err = fabsf(O[i] - O_ref[i]);
        if (err > max_err) max_err = err;
    }

    printf("2. exp2_flash_ld1rw_4x4 (store-to-load fwd):\n");
    printf("   Cycles: %.1f, GFLOPS: %.2f, Peak%%: %.1f%%\n",
           cycles, gflops, gflops / 128.0 * 100);
    printf("   Max error: %.6f %s\n\n", max_err, max_err < 0.01 ? "OK" : "FAIL");

    /* ============================================================ */
    /* Test 3: Baseline (separate exp2_rows + gemm_fp32_4x4) */
    /* ============================================================ */
    int ld_a = Nc * 4;  /* A (P) leading dim in bytes */
    memset(O, 0, M * D * sizeof(float));

    for (int i = 0; i < warmup; i++) {
        exp2_rows(S, P, M, Nc, scale, max_val, ld_s, Nc);
        gemm_fp32_4x4(P, V, O, Nc, ld_a, ld_v, ld_o);
    }

    start = rdcycle();
    for (int i = 0; i < iters; i++) {
        exp2_rows(S, P, M, Nc, scale, max_val, ld_s, Nc);
        gemm_fp32_4x4(P, V, O, Nc, ld_a, ld_v, ld_o);
    }
    end = rdcycle();

    cycles = (double)(end - start) / iters;
    gflops = total_flops / cycles * (freq / 1e9);

    /* Verify */
    max_err = 0;
    for (int i = 0; i < M * D; i++) {
        float err = fabsf(O[i] - O_ref[i]);
        if (err > max_err) max_err = err;
    }

    printf("3. Baseline (exp2_rows + gemm_fp32_4x4):\n");
    printf("   Cycles: %.1f, GFLOPS: %.2f, Peak%%: %.1f%%\n",
           cycles, gflops, gflops / 128.0 * 100);
    printf("   Max error: %.6f %s\n\n", max_err, max_err < 0.01 ? "OK" : "FAIL");

    /* ============================================================ */
    /* Test 4: Pure GEMM (reference peak) */
    /* ============================================================ */
    /* Use P from exp2_rows above */
    memset(O, 0, M * D * sizeof(float));

    for (int i = 0; i < warmup; i++) {
        gemm_fp32_4x4(P, V, O, Nc, ld_a, ld_v, ld_o);
    }

    start = rdcycle();
    for (int i = 0; i < iters; i++) {
        gemm_fp32_4x4(P, V, O, Nc, ld_a, ld_v, ld_o);
    }
    end = rdcycle();

    cycles = (double)(end - start) / iters;
    gflops = total_flops / cycles * (freq / 1e9);

    printf("4. Pure GEMM (gemm_fp32_4x4):\n");
    printf("   Cycles: %.1f, GFLOPS: %.2f, Peak%%: %.1f%%\n\n",
           cycles, gflops, gflops / 128.0 * 100);

    /* Summary */
    printf("=== Analysis ===\n");
    printf("LD1RW advantage: Runs on LD/ST pipe, not FLA\n");
    printf("  - DUP runs on FLA pipe (competes with FMLA)\n");
    printf("  - LD1RW runs on LD/ST pipe (parallel with FMLA)\n");
    printf("Expected: Better overlap of exp2 store + GEMM compute\n");

    free(S);
    free(V);
    free(O);
    free(O_ref);
    free(P);

    return 0;
}
