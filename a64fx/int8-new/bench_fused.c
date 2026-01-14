// bench_fused.c
// Benchmark fused GEMM: O = (A @ B^T) @ C
// A, B, C: [L, d] where d=256, L=4096 or 8192

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include "fused_gemm.h"

static inline uint64_t rdtimer(void) {
    uint64_t val;
    __asm__ volatile("isb" ::: "memory");
    __asm__ volatile("mrs %0, cntvct_el0" : "=r"(val));
    return val;
}

static inline uint64_t rdfreq(void) {
    uint64_t val;
    __asm__ volatile("mrs %0, cntfrq_el0" : "=r"(val));
    return val;
}

// Initialize with small random values to avoid overflow
static void init_matrix(int8_t* M, int L, int d) {
    for (int i = 0; i < L * d; i++) {
        M[i] = (int8_t)((rand() % 5) - 2);  // -2 to 2
    }
}

// Compare outputs with tolerance accounting for quantization scaling
// scale = scale factor applied to fused GEMM intermediate
static int compare_outputs_scaled(const int32_t* O_fused, const int32_t* O_ref,
                                   int L, int d, float scale,
                                   int64_t* max_diff, double* max_rel_err) {
    int errors = 0;
    *max_diff = 0;
    *max_rel_err = 0.0;

    for (int i = 0; i < L; i++) {
        for (int j = 0; j < d; j++) {
            // Scale reference to match fused (which has scale applied)
            double ref_scaled = (double)O_ref[i * d + j] * (double)scale;
            double fused_val = (double)O_fused[i * d + j];
            double diff = fabs(fused_val - ref_scaled);
            double ref_abs = fabs(ref_scaled);
            double rel_err = (ref_abs > 1.0) ? (diff / ref_abs) : diff;

            if ((int64_t)diff > *max_diff) *max_diff = (int64_t)diff;
            if (rel_err > *max_rel_err) *max_rel_err = rel_err;

            // Allow 50% relative error due to quantization
            if (rel_err > 0.5 && diff > 100) {
                if (errors < 5) {
                    printf("Mismatch at [%d,%d]: fused=%d, ref_scaled=%.1f, rel_err=%.2f\n",
                           i, j, O_fused[i * d + j], ref_scaled, rel_err);
                }
                errors++;
            }
        }
    }
    return errors;
}

int main(int argc, char** argv) {
    printf("=== Fused GEMM Benchmark: O = (A @ B^T) @ C ===\n\n");

    volatile uint64_t freq = rdfreq();
    printf("Timer frequency: %lu Hz\n", (unsigned long)freq);

    int d = 256;  // Fixed feature dimension
    int test_sizes[] = {256, 1024, 4096, 8192};  // L values to test
    int num_sizes = sizeof(test_sizes) / sizeof(test_sizes[0]);

    for (int t = 0; t < num_sizes; t++) {
        int L = test_sizes[t];
        printf("\n========================================\n");
        printf("Testing L=%d, d=%d\n", L, d);
        printf("========================================\n");

        // Allocate matrices
        int8_t* A = malloc((size_t)L * d);
        int8_t* B = malloc((size_t)L * d);
        int8_t* C = malloc((size_t)L * d);
        int32_t* O_fused = calloc((size_t)L * d, sizeof(int32_t));
        int32_t* O_ref = calloc((size_t)L * d, sizeof(int32_t));

        if (!A || !B || !C || !O_fused || !O_ref) {
            printf("Memory allocation failed for L=%d\n", L);
            continue;
        }

        // Initialize matrices
        srand(42);
        init_matrix(A, L, d);
        init_matrix(B, L, d);
        init_matrix(C, L, d);

        // Pack matrices
        printf("Packing matrices...\n");
        uint64_t t0 = rdtimer();
        fused_matrix_t* Apack = pack_A_fused(A, L, d);
        fused_matrix_t* Bpack = pack_B_fused(B, L, d);
        fused_matrix_t* Cpack = pack_C_fused(C, L, d);
        uint64_t t1 = rdtimer();

        if (!Apack || !Bpack || !Cpack) {
            printf("Packing failed\n");
            continue;
        }

        printf("Pack time: %.3f ms\n", (double)(t1 - t0) / freq * 1000.0);

        // Compute reference (only for small sizes)
        if (L <= 1024) {
            printf("Computing reference...\n");
            t0 = rdtimer();
            ref_gemm_ABtC(A, B, C, O_ref, L, d);
            t1 = rdtimer();
            printf("Reference time: %.3f ms\n", (double)(t1 - t0) / freq * 1000.0);
        }

        // Warmup fused GEMM
        printf("Warming up fused GEMM...\n");
        // Stage 1 output range: ~256 * 2 * 2 = 1024 (for random -2..2 inputs)
        // But we sum d=256 products, so max ~256 * 4 = 1024, need scale ~1/8 to fit int8
        float scale1 = 1.0f / 16.0f;   // Scale for stage 1
        float scale2 = 1.0f;           // Scale for stage 2
        fused_gemm_ABtC(Apack, Bpack, Cpack, O_fused, d, scale1, scale2);

        // Verify correctness (small sizes only)
        if (L <= 1024) {
            int64_t max_diff;
            double max_rel_err;
            int errors = compare_outputs_scaled(O_fused, O_ref, L, d, scale1,
                                                 &max_diff, &max_rel_err);
            if (errors > 0) {
                printf("WARNING: %d mismatches, max_diff=%ld, max_rel_err=%.2f\n",
                       errors, (long)max_diff, max_rel_err);
            } else {
                printf("Correctness: PASS (max_diff=%ld, max_rel_err=%.2f)\n",
                       (long)max_diff, max_rel_err);
            }
        }

        // Benchmark fused GEMM
        int iterations = (L <= 1024) ? 10 : 3;
        printf("\nBenchmarking fused GEMM (%d iterations)...\n", iterations);

        memset(O_fused, 0, (size_t)L * d * sizeof(int32_t));
        t0 = rdtimer();
        for (int i = 0; i < iterations; i++) {
            fused_gemm_ABtC(Apack, Bpack, Cpack, O_fused, d, scale1, scale2);
        }
        t1 = rdtimer();

        double time_s = (double)(t1 - t0) / freq;
        double time_per_call = time_s / iterations * 1000.0;  // ms

        // Calculate FLOPS
        // Stage 1: [L, d] @ [L, d]^T = 2*L*L*d ops
        // Stage 2: [L, L] @ [L, d] = 2*L*L*d ops
        // Total: 4*L*L*d ops (but we do it in tiles)
        double ops_stage1 = 2.0 * L * L * d;
        double ops_stage2 = 2.0 * L * L * d;
        double total_ops = ops_stage1 + ops_stage2;
        double gops = total_ops * iterations / time_s / 1e9;

        printf("\nFused GEMM Results:\n");
        printf("  Time per call: %.3f ms\n", time_per_call);
        printf("  Total ops: %.2e (Stage1: %.2e + Stage2: %.2e)\n",
               total_ops, ops_stage1, ops_stage2);
        printf("  GOPS: %.2f\n", gops);

        // Memory analysis
        double A_bytes = (double)L * (double)d;
        double B_bytes = (double)L * (double)d;
        double C_bytes = (double)L * (double)d;
        double S_bytes = (double)L * (double)L * (double)sizeof(int32_t);  // If not fused
        double O_bytes = (double)L * (double)d * (double)sizeof(int32_t);

        printf("\nMemory Analysis:\n");
        printf("  A, B, C: %.3f MB each\n", A_bytes / 1000000.0);
        printf("  S (if not fused): %.3f MB\n", S_bytes / 1000000.0);
        printf("  O: %.3f MB\n", O_bytes / 1000000.0);
        printf("  Fused saves: %.3f MB intermediate\n", S_bytes / 1000000.0);

        // Cleanup
        free_fused_matrix(Apack);
        free_fused_matrix(Bpack);
        free_fused_matrix(Cpack);
        free(A);
        free(B);
        free(C);
        free(O_fused);
        free(O_ref);
    }

    printf("\n=== Benchmark Complete ===\n");
    return 0;
}
