// bench_attention.c
// Benchmark fused attention: O = softmax(Q @ K^T / sqrt(d)) @ V
// Q, K, V: [L, d] where d=256, L=256, 1024, 4096

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include "fused_attention.h"

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

// Initialize with random values biased positive (for V to have non-zero average)
static void init_matrix(int8_t* M, int L, int d) {
    for (int i = 0; i < L * d; i++) {
        M[i] = (int8_t)((rand() % 7) - 3);  // -3 to 3
    }
}

// Initialize V with positive bias to get meaningful outputs
static void init_matrix_positive(int8_t* M, int L, int d) {
    for (int i = 0; i < L * d; i++) {
        M[i] = (int8_t)(rand() % 5);  // 0 to 4
    }
}

// Compare outputs with tolerance
// scale_factor: fused output is scaled by this factor vs reference
static int compare_outputs_scaled(const int32_t* O_fused, const int32_t* O_ref,
                                   int L, int d, float scale_factor,
                                   float* max_rel_err) {
    int errors = 0;
    *max_rel_err = 0.0f;

    for (int i = 0; i < L; i++) {
        for (int j = 0; j < d; j++) {
            // Scale fused output back to reference scale
            float v_fused = (float)O_fused[i * d + j] / scale_factor;
            float v_ref = (float)O_ref[i * d + j];
            float diff = fabsf(v_fused - v_ref);
            float ref_abs = fabsf(v_ref);
            float rel_err = (ref_abs > 1.0f) ? (diff / ref_abs) : diff;

            if (rel_err > *max_rel_err) *max_rel_err = rel_err;

            // Allow larger tolerance due to quantization
            if (rel_err > 0.5f && diff > 5.0f) {
                if (errors < 5) {
                    printf("Mismatch at [%d,%d]: fused_scaled=%.1f, ref=%.1f, rel_err=%.2f\n",
                           i, j, v_fused, v_ref, rel_err);
                }
                errors++;
            }
        }
    }
    return errors;
}

int main(int argc, char** argv) {
    printf("=== Fused Attention Benchmark ===\n");
    printf("O = softmax(Q @ K^T / sqrt(d)) @ V\n\n");

    volatile uint64_t freq = rdfreq();
    printf("Timer frequency: %lu Hz\n", (unsigned long)freq);

    int d = 256;  // Fixed feature dimension
    float scale = 1.0f / sqrtf((float)d);  // 1/sqrt(d)
    printf("Scale factor: %.6f (1/sqrt(%d))\n\n", scale, d);

    int test_sizes[] = {256, 512, 1024};
    int num_sizes = sizeof(test_sizes) / sizeof(test_sizes[0]);

    for (int t = 0; t < num_sizes; t++) {
        int L = test_sizes[t];
        printf("\n========================================\n");
        printf("Testing L=%d, d=%d\n", L, d);
        printf("========================================\n");

        // Allocate matrices
        int8_t* Q = malloc((size_t)L * d);
        int8_t* K = malloc((size_t)L * d);
        int8_t* V = malloc((size_t)L * d);
        int32_t* O_fused = calloc((size_t)L * d, sizeof(int32_t));
        int32_t* O_ref = calloc((size_t)L * d, sizeof(int32_t));

        if (!Q || !K || !V || !O_fused || !O_ref) {
            printf("Memory allocation failed for L=%d\n", L);
            continue;
        }

        // Initialize matrices
        srand(42);
        init_matrix(Q, L, d);
        init_matrix(K, L, d);
        init_matrix_positive(V, L, d);  // Use positive values for V

        // Pack matrices
        printf("Packing matrices...\n");
        uint64_t t0 = rdtimer();
        fused_matrix_t* Qpack = pack_A_fused(Q, L, d);
        fused_matrix_t* Kpack = pack_B_fused(K, L, d);
        fused_matrix_t* Vpack = pack_C_fused(V, L, d);
        uint64_t t1 = rdtimer();

        if (!Qpack || !Kpack || !Vpack) {
            printf("Packing failed\n");
            continue;
        }

        printf("Pack time: %.3f ms\n", (double)(t1 - t0) / (double)freq * 1000.0);

        // Compute reference
        printf("Computing reference attention...\n");
        t0 = rdtimer();
        ref_attention(Q, K, V, O_ref, L, d, scale);
        t1 = rdtimer();
        printf("Reference time: %.3f ms\n", (double)(t1 - t0) / (double)freq * 1000.0);

        // Test INT8 variant
        printf("\n--- INT8 Softmax Variant ---\n");
        printf("Warming up...\n");
        memset(O_fused, 0, (size_t)L * d * sizeof(int32_t));
        fused_attention_int8(Qpack, Kpack, Vpack, O_fused, d, scale);

        // Verify correctness
        // INT8 softmax scales P to [0, 127], so output is scaled by ~127
        float int8_scale = 127.0f;
        float max_rel_err;
        int errors = compare_outputs_scaled(O_fused, O_ref, L, d, int8_scale, &max_rel_err);
        if (errors > 0) {
            printf("WARNING: %d mismatches, max_rel_err=%.2f\n", errors, max_rel_err);
        } else {
            printf("Correctness: PASS (max_rel_err=%.2f)\n", max_rel_err);
        }

        // Benchmark INT8
        int iterations = (L <= 512) ? 10 : 3;
        printf("Benchmarking (%d iterations)...\n", iterations);

        memset(O_fused, 0, (size_t)L * d * sizeof(int32_t));
        t0 = rdtimer();
        for (int i = 0; i < iterations; i++) {
            fused_attention_int8(Qpack, Kpack, Vpack, O_fused, d, scale);
        }
        t1 = rdtimer();

        double time_s = (double)(t1 - t0) / (double)freq;
        double time_per_call = time_s / iterations * 1000.0;

        // Calculate FLOPS (approximate)
        // Stage 1: 2*L*L*d ops (Q @ K^T)
        // Softmax: ~5*L*L ops (exp, sum, div)
        // Stage 2: 2*L*L*d ops (P @ V)
        double ops_gemm = 4.0 * L * L * d;  // Both GEMMs
        double ops_softmax = 5.0 * L * L;
        double total_ops = ops_gemm + ops_softmax;
        double gops = total_ops * iterations / time_s / 1e9;

        printf("INT8 Results:\n");
        printf("  Time per call: %.3f ms\n", time_per_call);
        printf("  GOPS: %.2f (GEMM: %.2e, softmax: %.2e)\n", gops, ops_gemm, ops_softmax);

        // Test UINT8 variant
        printf("\n--- UINT8 Softmax Variant (with bias correction) ---\n");
        memset(O_fused, 0, (size_t)L * d * sizeof(int32_t));

        printf("Warming up...\n");
        fused_attention_uint8(Qpack, Kpack, Vpack, O_fused, d, scale);

        // UINT8 softmax scales P to [0, 255], so output is scaled by ~255
        float uint8_scale = 255.0f;
        errors = compare_outputs_scaled(O_fused, O_ref, L, d, uint8_scale, &max_rel_err);
        if (errors > 0) {
            printf("WARNING: %d mismatches, max_rel_err=%.2f\n", errors, max_rel_err);
        } else {
            printf("Correctness: PASS (max_rel_err=%.2f)\n", max_rel_err);
        }

        // Benchmark UINT8
        printf("Benchmarking (%d iterations)...\n", iterations);

        memset(O_fused, 0, (size_t)L * d * sizeof(int32_t));
        t0 = rdtimer();
        for (int i = 0; i < iterations; i++) {
            fused_attention_uint8(Qpack, Kpack, Vpack, O_fused, d, scale);
        }
        t1 = rdtimer();

        time_s = (double)(t1 - t0) / (double)freq;
        time_per_call = time_s / iterations * 1000.0;
        gops = total_ops * iterations / time_s / 1e9;

        printf("UINT8 Results:\n");
        printf("  Time per call: %.3f ms\n", time_per_call);
        printf("  GOPS: %.2f\n", gops);

        // Memory analysis
        double QKV_bytes = 3.0 * L * d;
        double S_bytes = (double)L * L * sizeof(float);
        double O_bytes = (double)L * d * sizeof(int32_t);

        printf("\nMemory Analysis:\n");
        printf("  Q, K, V: %.3f MB total\n", QKV_bytes / 1e6);
        printf("  S (if not fused): %.3f MB\n", S_bytes / 1e6);
        printf("  O: %.3f MB\n", O_bytes / 1e6);
        printf("  Fused saves: %.3f MB intermediate\n", S_bytes / 1e6);

        // Cleanup
        free_fused_matrix(Qpack);
        free_fused_matrix(Kpack);
        free_fused_matrix(Vpack);
        free(Q);
        free(K);
        free(V);
        free(O_fused);
        free(O_ref);
    }

    printf("\n=== Benchmark Complete ===\n");
    return 0;
}
