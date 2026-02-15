#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>

extern void exp2_flash_tiled_4x4(
    const int32_t* S, const float* V, float* O, float* P,
    int Nc, float scale, float max_val,
    int ld_s, int ld_v, int ld_o
);

extern void exp2_flash_ld1rw_4x4(
    const int32_t* S, const float* V, float* O, float* P,
    int Nc, float scale, float max_val,
    int ld_s, int ld_v, int ld_o
);

extern void gemm_baseline_4x4(
    const float* P, const float* V, float* O,
    int Nc, int ld_v, int ld_o
);

extern void exp2_gather_4x4(
    const int32_t* S, const float* V, float* O, float* P,
    int Nc, float scale, float max_val,
    int ld_s, int ld_v, int ld_o
);

extern void exp2_4k_4x4(
    const int32_t* S, const float* V, float* O, float* P,
    int Nc, float scale, float max_val,
    int ld_s, int ld_v, int ld_o
);

extern void exp2_nop_4x4(
    const int32_t* S, const float* V, float* O, float* P,
    int Nc, float scale, float max_val,
    int ld_s, int ld_v, int ld_o
);

extern void exp2_best_4x4(
    const int32_t* S, const float* V, float* O, float* P,
    int Nc, float scale, float max_val,
    int ld_s, int ld_v, int ld_o
);

static inline uint64_t get_cycles(void) {
    uint64_t val;
    asm volatile("mrs %0, cntvct_el0" : "=r" (val));
    return val;
}

static inline uint64_t get_freq(void) {
    uint64_t val;
    asm volatile("mrs %0, cntfrq_el0" : "=r" (val));
    return val;
}

int main(int argc, char** argv) {
    int M = 4;
    int Nc = 512;         // Typical attention sequence length
    int D = 64;           // Typical head dimension (4 SVE vectors)
    int warmup = 10;
    int repeat = 100;

    if (argc > 1) Nc = atoi(argv[1]);
    if (argc > 2) repeat = atoi(argv[2]);

    // Allocate aligned memory
    int32_t* S = aligned_alloc(64, M * Nc * sizeof(int32_t));
    float* V = aligned_alloc(64, Nc * D * sizeof(float));
    float* O = aligned_alloc(64, M * D * sizeof(float));
    float* P = aligned_alloc(64, M * Nc * sizeof(float));
    float* O_ref = aligned_alloc(64, M * D * sizeof(float));

    // Initialize S with random int32 values in reasonable range
    srand(42);
    for (int i = 0; i < M * Nc; i++) {
        S[i] = (rand() % 200) - 100;  // Range: -100 to +99
    }

    // Initialize V with random values
    for (int i = 0; i < Nc * D; i++) {
        V[i] = (float)(rand() % 1000) / 1000.0f - 0.5f;
    }

    float scale = 1.0f / sqrtf(64.0f);  // Typical attention scale
    float max_val = 50.0f * scale;       // Approximate max for stability

    int ld_s = Nc;       // S stride
    int ld_v = D * 4;    // V stride in bytes (D floats = D*4 bytes)
    int ld_o = D * 4;    // O stride in bytes

    printf("=== FlashAttention-style exp2+GEMM Benchmark ===\n");
    printf("M=%d, Nc=%d, D=%d\n", M, Nc, D);
    printf("scale=%.6f, max_val=%.4f\n", scale, max_val);
    printf("\n");

    // Reference: compute exp2 then GEMM
    memset(O_ref, 0, M * D * sizeof(float));
    for (int m = 0; m < M; m++) {
        for (int k = 0; k < Nc; k++) {
            float x = (float)S[m * Nc + k] * scale - max_val;
            float p = exp2f(x);
            P[m * Nc + k] = p;
            for (int d = 0; d < D; d++) {
                O_ref[m * D + d] += p * V[k * D + d];
            }
        }
    }

    // Warmup
    for (int w = 0; w < warmup; w++) {
        memset(O, 0, M * D * sizeof(float));
        exp2_flash_tiled_4x4(S, V, O, P, Nc, scale, max_val, ld_s, ld_v, ld_o);
    }

    // Benchmark exp2_flash_tiled_4x4
    uint64_t t0 = get_cycles();
    for (int r = 0; r < repeat; r++) {
        memset(O, 0, M * D * sizeof(float));
        exp2_flash_tiled_4x4(S, V, O, P, Nc, scale, max_val, ld_s, ld_v, ld_o);
    }
    uint64_t t1 = get_cycles();
    double cycles_tiled = (double)(t1 - t0) / repeat;

    // Check correctness
    float max_err = 0;
    for (int i = 0; i < M * D; i++) {
        float err = fabsf(O[i] - O_ref[i]) / (fabsf(O_ref[i]) + 1e-6f);
        if (err > max_err) max_err = err;
    }

    // Warmup exp2_flash_ld1rw_4x4
    for (int w = 0; w < warmup; w++) {
        memset(O, 0, M * D * sizeof(float));
        exp2_flash_ld1rw_4x4(S, V, O, P, Nc, scale, max_val, ld_s, ld_v, ld_o);
    }

    // Benchmark exp2_flash_ld1rw_4x4
    t0 = get_cycles();
    for (int r = 0; r < repeat; r++) {
        memset(O, 0, M * D * sizeof(float));
        exp2_flash_ld1rw_4x4(S, V, O, P, Nc, scale, max_val, ld_s, ld_v, ld_o);
    }
    t1 = get_cycles();
    double cycles_ld1rw = (double)(t1 - t0) / repeat;

    // Check correctness of ld1rw version
    float max_err_ld1rw = 0;
    for (int i = 0; i < M * D; i++) {
        float err = fabsf(O[i] - O_ref[i]) / (fabsf(O_ref[i]) + 1e-6f);
        if (err > max_err_ld1rw) max_err_ld1rw = err;
    }

    // Calculate metrics
    // FLOPs:
    //   - exp2: ~10 ops per element (approximate)
    //   - GEMM: M * Nc * D * 2 (mul + add)
    int64_t gemm_flops = (int64_t)M * Nc * D * 2;
    int64_t exp2_flops = (int64_t)M * Nc * 10;  // Approximate
    int64_t total_flops = gemm_flops + exp2_flops;

    double freq = (double)get_freq();
    double time_tiled = cycles_tiled / freq;
    double time_ld1rw = cycles_ld1rw / freq;

    double gflops_tiled = (double)total_flops / time_tiled / 1e9;
    double gflops_ld1rw = (double)total_flops / time_ld1rw / 1e9;

    // A64FX peak: 2 FLA pipes * 16 FP32/vector * 2 GHz = 64 GFLOPS/core
    double peak_gflops = 64.0;

    printf("Results:\n");
    printf("  exp2_flash_tiled_4x4:\n");
    printf("    Cycles: %.0f\n", cycles_tiled);
    printf("    Time: %.2f us\n", time_tiled * 1e6);
    printf("    GFLOPS: %.2f (%.1f%% of peak)\n", gflops_tiled, gflops_tiled / peak_gflops * 100);
    printf("    Max error vs ref: %.2e\n", max_err);
    printf("\n");
    printf("  exp2_flash_ld1rw_4x4:\n");
    printf("    Cycles: %.0f\n", cycles_ld1rw);
    printf("    Time: %.2f us\n", time_ld1rw * 1e6);
    printf("    GFLOPS: %.2f (%.1f%% of peak)\n", gflops_ld1rw, gflops_ld1rw / peak_gflops * 100);
    printf("    Max error vs ref: %.2e\n", max_err_ld1rw);
    printf("\n");
    printf("  Speedup (ld1rw vs tiled): %.2fx\n", cycles_tiled / cycles_ld1rw);

    // Benchmark exp2_gather_4x4
    for (int w = 0; w < warmup; w++) {
        memset(O, 0, M * D * sizeof(float));
        exp2_gather_4x4(S, V, O, P, Nc, scale, max_val, ld_s, ld_v, ld_o);
    }

    t0 = get_cycles();
    for (int r = 0; r < repeat; r++) {
        memset(O, 0, M * D * sizeof(float));
        exp2_gather_4x4(S, V, O, P, Nc, scale, max_val, ld_s, ld_v, ld_o);
    }
    t1 = get_cycles();
    double cycles_gather = (double)(t1 - t0) / repeat;

    // Check correctness of gather version
    float max_err_gather = 0;
    for (int i = 0; i < M * D; i++) {
        float err = fabsf(O[i] - O_ref[i]) / (fabsf(O_ref[i]) + 1e-6f);
        if (err > max_err_gather) max_err_gather = err;
    }

    double time_gather = cycles_gather / freq;
    double gflops_gather = (double)total_flops / time_gather / 1e9;

    printf("\n  exp2_gather_4x4 (SVE gather S):\n");
    printf("    Cycles: %.0f\n", cycles_gather);
    printf("    Time: %.2f us\n", time_gather * 1e6);
    printf("    GFLOPS: %.2f (%.1f%% of peak)\n", gflops_gather, gflops_gather / peak_gflops * 100);
    printf("    Max error vs ref: %.2e\n", max_err_gather);

    // Benchmark exp2_4k_4x4 (4K unrolled)
    for (int w = 0; w < warmup; w++) {
        memset(O, 0, M * D * sizeof(float));
        exp2_4k_4x4(S, V, O, P, Nc, scale, max_val, ld_s, ld_v, ld_o);
    }

    t0 = get_cycles();
    for (int r = 0; r < repeat; r++) {
        memset(O, 0, M * D * sizeof(float));
        exp2_4k_4x4(S, V, O, P, Nc, scale, max_val, ld_s, ld_v, ld_o);
    }
    t1 = get_cycles();
    double cycles_4k = (double)(t1 - t0) / repeat;

    float max_err_4k = 0;
    for (int i = 0; i < M * D; i++) {
        float err = fabsf(O[i] - O_ref[i]) / (fabsf(O_ref[i]) + 1e-6f);
        if (err > max_err_4k) max_err_4k = err;
    }

    double time_4k = cycles_4k / freq;
    double gflops_4k = (double)total_flops / time_4k / 1e9;

    printf("\n  exp2_4k_4x4 (4K unrolled):\n");
    printf("    Cycles: %.0f\n", cycles_4k);
    printf("    Time: %.2f us\n", time_4k * 1e6);
    printf("    GFLOPS: %.2f (%.1f%% of peak)\n", gflops_4k, gflops_4k / peak_gflops * 100);
    printf("    Max error vs ref: %.2e\n", max_err_4k);

    // Benchmark exp2_nop_4x4 (no P buffer, DUP broadcast)
    for (int w = 0; w < warmup; w++) {
        memset(O, 0, M * D * sizeof(float));
        exp2_nop_4x4(S, V, O, P, Nc, scale, max_val, ld_s, ld_v, ld_o);
    }

    t0 = get_cycles();
    for (int r = 0; r < repeat; r++) {
        memset(O, 0, M * D * sizeof(float));
        exp2_nop_4x4(S, V, O, P, Nc, scale, max_val, ld_s, ld_v, ld_o);
    }
    t1 = get_cycles();
    double cycles_nop = (double)(t1 - t0) / repeat;

    float max_err_nop = 0;
    for (int i = 0; i < M * D; i++) {
        float err = fabsf(O[i] - O_ref[i]) / (fabsf(O_ref[i]) + 1e-6f);
        if (err > max_err_nop) max_err_nop = err;
    }

    double time_nop = cycles_nop / freq;
    double gflops_nop = (double)total_flops / time_nop / 1e9;

    printf("\n  exp2_nop_4x4 (DUP broadcast, no P):\n");
    printf("    Cycles: %.0f\n", cycles_nop);
    printf("    Time: %.2f us\n", time_nop * 1e6);
    printf("    GFLOPS: %.2f (%.1f%% of peak)\n", gflops_nop, gflops_nop / peak_gflops * 100);
    printf("    Max error vs ref: %.2e\n", max_err_nop);

    // Benchmark exp2_best_4x4 (4K unrolled + DUP)
    for (int w = 0; w < warmup; w++) {
        memset(O, 0, M * D * sizeof(float));
        exp2_best_4x4(S, V, O, P, Nc, scale, max_val, ld_s, ld_v, ld_o);
    }

    t0 = get_cycles();
    for (int r = 0; r < repeat; r++) {
        memset(O, 0, M * D * sizeof(float));
        exp2_best_4x4(S, V, O, P, Nc, scale, max_val, ld_s, ld_v, ld_o);
    }
    t1 = get_cycles();
    double cycles_best = (double)(t1 - t0) / repeat;

    float max_err_best = 0;
    for (int i = 0; i < M * D; i++) {
        float err = fabsf(O[i] - O_ref[i]) / (fabsf(O_ref[i]) + 1e-6f);
        if (err > max_err_best) max_err_best = err;
    }

    double time_best = cycles_best / freq;
    double gflops_best = (double)total_flops / time_best / 1e9;

    printf("\n  exp2_best_4x4 (4K+DUP, optimized):\n");
    printf("    Cycles: %.0f\n", cycles_best);
    printf("    Time: %.2f us\n", time_best * 1e6);
    printf("    GFLOPS: %.2f (%.1f%% of peak)\n", gflops_best, gflops_best / peak_gflops * 100);
    printf("    Max error vs ref: %.2e\n", max_err_best);
    printf("\n  *** Best speedup vs baseline tiled: %.2fx ***\n", cycles_tiled / cycles_best);

    // Benchmark baseline GEMM (P already computed from above)
    for (int w = 0; w < warmup; w++) {
        memset(O, 0, M * D * sizeof(float));
        gemm_baseline_4x4(P, V, O, Nc, ld_v, ld_o);
    }

    t0 = get_cycles();
    for (int r = 0; r < repeat; r++) {
        memset(O, 0, M * D * sizeof(float));
        gemm_baseline_4x4(P, V, O, Nc, ld_v, ld_o);
    }
    t1 = get_cycles();
    double cycles_gemm = (double)(t1 - t0) / repeat;

    double time_gemm = cycles_gemm / freq;
    double gflops_gemm = (double)gemm_flops / time_gemm / 1e9;

    printf("\n  gemm_baseline_4x4 (no exp2):\n");
    printf("    Cycles: %.0f\n", cycles_gemm);
    printf("    Time: %.2f us\n", time_gemm * 1e6);
    printf("    GFLOPS: %.2f (%.1f%% of peak)\n", gflops_gemm, gflops_gemm / peak_gflops * 100);

    // Detailed cycle analysis
    printf("\nCycle Analysis:\n");
    printf("  GEMM FLOPs: %ld (M=%d * Nc=%d * D=%d * 2)\n", gemm_flops, M, Nc, D);
    printf("  Total FLOPs: %ld\n", total_flops);
    printf("  Cycles per K iteration (gemm only): %.1f\n", cycles_gemm / Nc);
    printf("  Cycles per K iteration (tiled): %.1f\n", cycles_tiled / Nc);
    printf("  Cycles per K iteration (gather): %.1f\n", cycles_gather / Nc);
    printf("  Cycles per K iteration (4k unrolled): %.1f\n", cycles_4k / Nc);
    printf("  Cycles per K iteration (nop/DUP): %.1f\n", cycles_nop / Nc);
    printf("  Cycles per K iteration (ld1rw): %.1f\n", cycles_ld1rw / Nc);
    printf("  exp2 overhead per K (tiled): %.1f cycles (%.0f%%)\n",
           (cycles_tiled - cycles_gemm) / Nc,
           (cycles_tiled - cycles_gemm) / cycles_tiled * 100);
    printf("  exp2 overhead per K (4k unrolled): %.1f cycles (%.0f%%)\n",
           (cycles_4k - cycles_gemm) / Nc,
           (cycles_4k - cycles_gemm) / cycles_4k * 100);
    printf("  Cycles per FMLA (gemm): %.2f (16 FMLAs per K)\n", cycles_gemm / Nc / 16);
    printf("  Cycles per FMLA (tiled): %.2f (16 FMLAs per K)\n", cycles_tiled / Nc / 16);
    printf("  Cycles per FMLA (4k unrolled): %.2f (16 FMLAs per K)\n", cycles_4k / Nc / 16);

    free(S);
    free(V);
    free(O);
    free(P);
    free(O_ref);

    return 0;
}
