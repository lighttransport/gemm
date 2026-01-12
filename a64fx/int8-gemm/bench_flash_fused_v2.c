#include "flash_attention_fused_v2.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif

// =============================================================================
// Timing utilities
// =============================================================================

static double get_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1000000.0;
}

// =============================================================================
// Reference implementation (parallel scalar, for correctness checking)
// =============================================================================

static void reference_attention_parallel(
    const int8_t* Q,
    const int8_t* K,
    const int8_t* V,
    float* O_ref,
    int64_t L,
    int64_t head_dim,
    int num_threads)
{
    float scale = 1.0f / sqrtf((float)head_dim);

    // Allocate S matrix (shared across threads)
    float* S = (float*)malloc(L * L * sizeof(float));
    if (!S) {
        fprintf(stderr, "Failed to allocate S matrix\n");
        return;
    }

    // Parallel computation over rows
    #ifdef _OPENMP
    omp_set_num_threads(num_threads);
    #pragma omp parallel
    {
        #pragma omp for schedule(dynamic, 8)
    #endif
        for (int64_t i = 0; i < L; i++) {
            // Compute scores: S[i,:] = Q[i,:] @ K^T * scale
            for (int64_t j = 0; j < L; j++) {
                int32_t dot = 0;
                for (int64_t k = 0; k < head_dim; k++) {
                    dot += (int32_t)Q[i * head_dim + k] * (int32_t)K[j * head_dim + k];
                }
                S[i * L + j] = (float)dot * scale;
            }

            // Softmax (per-row, no synchronization needed)
            float max_val = S[i * L];
            for (int64_t j = 1; j < L; j++) {
                if (S[i * L + j] > max_val) max_val = S[i * L + j];
            }

            float sum = 0.0f;
            for (int64_t j = 0; j < L; j++) {
                S[i * L + j] = expf(S[i * L + j] - max_val);
                sum += S[i * L + j];
            }

            for (int64_t j = 0; j < L; j++) {
                S[i * L + j] /= sum;
            }

            // Output: O[i,:] = S[i,:] @ V
            for (int64_t d = 0; d < head_dim; d++) {
                float acc = 0.0f;
                for (int64_t j = 0; j < L; j++) {
                    acc += S[i * L + j] * (float)V[j * head_dim + d];
                }
                O_ref[i * head_dim + d] = acc;
            }
        }
    #ifdef _OPENMP
    }
    #endif

    free(S);
}

// Single-threaded wrapper (for backward compatibility)
static void reference_attention(
    const int8_t* Q,
    const int8_t* K,
    const int8_t* V,
    float* O_ref,
    int64_t L,
    int64_t head_dim)
{
    reference_attention_parallel(Q, K, V, O_ref, L, head_dim, 1);
}

// =============================================================================
// Correctness checking
// =============================================================================

static void check_correctness(
    const float* O_ref,
    const float* O_test,
    int64_t L,
    int64_t head_dim,
    const char* test_name)
{
    double max_abs_diff = 0.0;
    double max_rel_diff = 0.0;
    int64_t mismatch_count = 0;

    for (int64_t i = 0; i < L * head_dim; i++) {
        double abs_diff = fabs(O_ref[i] - O_test[i]);
        double rel_diff = abs_diff / (fabs(O_ref[i]) + 1e-8);

        if (abs_diff > max_abs_diff) max_abs_diff = abs_diff;
        if (rel_diff > max_rel_diff) max_rel_diff = rel_diff;

        if (abs_diff > 1.0) {
            mismatch_count++;
            if (mismatch_count <= 10) {
                printf("  Mismatch at [%ld]: ref=%.6f, test=%.6f, diff=%.6f\n",
                       i, O_ref[i], O_test[i], abs_diff);
            }
        }
    }

    printf("\n%s Correctness Check:\n", test_name);
    printf("  Max absolute diff: %.6e\n", max_abs_diff);
    printf("  Max relative diff: %.6e\n", max_rel_diff);
    printf("  Mismatch count (|diff| > 1.0): %ld / %ld\n",
           mismatch_count, L * head_dim);

    if (max_abs_diff < 5.0) {
        printf("  ✓ PASS (max diff < 5.0)\n");
    } else {
        printf("  ✗ FAIL (max diff >= 5.0)\n");
    }
}

// =============================================================================
// Performance benchmark
// =============================================================================

static void benchmark(
    const int8_t* Q,
    const int8_t* K,
    const int8_t* V,
    float* O,
    int64_t L,
    int64_t head_dim,
    int num_iters)
{
    printf("\nBenchmarking L=%ld, head_dim=%ld, iters=%d\n", L, head_dim, num_iters);

    // Warmup
    flash_attention_fused_forward(Q, K, V, O, NULL, L, head_dim);

    // Timed runs
    double total_time = 0.0;
    double min_time = 1e9;

    for (int iter = 0; iter < num_iters; iter++) {
        double t_start = get_time_ms();
        flash_attention_fused_forward(Q, K, V, O, NULL, L, head_dim);
        double t_end = get_time_ms();

        double elapsed = t_end - t_start;
        total_time += elapsed;
        if (elapsed < min_time) min_time = elapsed;
    }

    double avg_time = total_time / num_iters;

    // Compute FLOPS
    // Q@K^T: 2*L*L*head_dim ops (INT8)
    // P@V: 2*L*L*head_dim ops (FP32)
    // Softmax: ~20 ops/elem * L*L
    double gemm_ops = 2.0 * L * L * head_dim;  // Q@K^T
    double softmax_ops = 20.0 * L * L;
    double pv_ops = 2.0 * L * L * head_dim;  // P@V
    double total_ops = gemm_ops + softmax_ops + pv_ops;

    double gops = total_ops / (avg_time * 1e6);
    double gemm_gops = gemm_ops / (avg_time * 1e6);
    double pv_gflops = pv_ops / (avg_time * 1e6);

    // Expected peaks:
    // INT8 GEMM: 3072 GOPS (SDOT)
    // FP32 GEMM: 2048 GFLOPS (FMLA)
    double int8_peak = 3072.0;
    double fp32_peak = 2048.0;

    // Approximate time breakdown (assuming 50/50 split)
    double gemm_efficiency = (gemm_gops / int8_peak) * 100.0;
    double pv_efficiency = (pv_gflops / fp32_peak) * 100.0;
    double softmax_time_pct = (softmax_ops / total_ops) * 100.0;

    printf("\nPerformance Results:\n");
    printf("  Average time:       %.3f ms\n", avg_time);
    printf("  Min time:           %.3f ms\n", min_time);
    printf("  Total throughput:   %.1f GOPS\n", gops);
    printf("\nComponent Breakdown:\n");
    printf("  Q@K^T (INT8 GEMM):  %.1f GOPS (%.1f%% of %d GOPS peak)\n",
           gemm_gops, gemm_efficiency, (int)int8_peak);
    printf("  P@V (FP32 GEMM):    %.1f GFLOPS (%.1f%% of %d GFLOPS peak)\n",
           pv_gflops, pv_efficiency, (int)fp32_peak);
    printf("  Softmax overhead:   ~%.1f%% of total ops\n", softmax_time_pct);

    // Memory bandwidth estimate
    size_t qkv_bytes = 3 * L * head_dim;  // Q, K, V (INT8)
    size_t o_bytes = L * head_dim * 4;    // O (FP32)
    double total_bytes = qkv_bytes + o_bytes;
    double bandwidth_gb_s = total_bytes / (avg_time * 1e6);
    printf("  Memory bandwidth:   %.1f GB/s (min data movement)\n", bandwidth_gb_s);

    printf("\n");
}

// =============================================================================
// Main
// =============================================================================

int main(int argc, char** argv) {
    // Parse arguments
    int64_t L = 1024;
    int64_t head_dim = 128;
    int num_iters = 5;
    int num_threads = 1;

    if (argc >= 2) L = atol(argv[1]);
    if (argc >= 3) head_dim = atol(argv[2]);
    if (argc >= 4) num_iters = atoi(argv[3]);
    if (argc >= 5) num_threads = atoi(argv[4]);

    // Default to 12 threads if OpenMP available and not specified
    #ifdef _OPENMP
    if (argc < 5) {
        num_threads = omp_get_max_threads();
        if (num_threads > 12) num_threads = 12;  // A64FX has 12 cores
    }
    #endif

    printf("=================================================================\n");
    printf("Fused FlashAttention V2 Benchmark\n");
    printf("=================================================================\n");
    printf("Configuration:\n");
    printf("  L (sequence length):  %ld\n", L);
    printf("  head_dim:             %ld\n", head_dim);
    printf("  Tile sizes:           TILE_BR=%d, TILE_BC=%d\n", FA_TILE_BR, FA_TILE_BC);
    printf("  L1D cache:            64 KB\n");
    printf("  Expected working set: ~46 KB\n");
    #ifdef _OPENMP
    printf("  Reference threads:    %d\n", num_threads);
    #else
    printf("  Reference threads:    1 (OpenMP not available)\n");
    #endif
    printf("\n");

    // Allocate matrices
    int8_t* Q = (int8_t*)flash_aligned_alloc(L * head_dim);
    int8_t* K = (int8_t*)flash_aligned_alloc(L * head_dim);
    int8_t* V = (int8_t*)flash_aligned_alloc(L * head_dim);
    float* O = (float*)flash_aligned_alloc(L * head_dim * sizeof(float));
    float* O_ref = (float*)flash_aligned_alloc(L * head_dim * sizeof(float));

    if (!Q || !K || !V || !O || !O_ref) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    // Initialize with random data
    srand(42);
    for (int64_t i = 0; i < L * head_dim; i++) {
        Q[i] = (int8_t)(rand() % 32 - 16);
        K[i] = (int8_t)(rand() % 32 - 16);
        V[i] = (int8_t)(rand() % 32 - 16);
    }

    // Correctness check with timing
    if (L <= 512) {
        printf("Running correctness check (L=%ld)...\n", L);

        // Time reference implementation
        double ref_start = get_time_ms();
        reference_attention_parallel(Q, K, V, O_ref, L, head_dim, num_threads);
        double ref_end = get_time_ms();
        double ref_time = ref_end - ref_start;

        // Time fused implementation
        double fused_start = get_time_ms();
        flash_attention_fused_forward(Q, K, V, O, NULL, L, head_dim);
        double fused_end = get_time_ms();
        double fused_time = fused_end - fused_start;

        // Check correctness
        check_correctness(O_ref, O, L, head_dim, "Fused FlashAttention V2");

        // Report timing
        printf("\nTiming Comparison:\n");
        printf("  Reference (FP32, %d threads): %.3f ms\n", num_threads, ref_time);
        printf("  Fused V2 (INT8, 1 thread):    %.3f ms\n", fused_time);
        printf("  Speedup vs reference:         %.2fx\n", ref_time / fused_time);

        // Estimate single-threaded reference time
        double ref_single_thread_est = ref_time * num_threads;
        printf("  Est. speedup vs 1-thread ref: %.2fx\n", ref_single_thread_est / fused_time);
    } else {
        printf("Skipping correctness check (L=%ld too large, use L<=512)\n", L);
        printf("Running reference benchmark for comparison...\n");

        // Just time reference for large inputs
        double ref_start = get_time_ms();
        reference_attention_parallel(Q, K, V, O_ref, L, head_dim, num_threads);
        double ref_end = get_time_ms();
        double ref_time = ref_end - ref_start;

        printf("  Reference (FP32, %d threads): %.3f ms\n", num_threads, ref_time);
    }

    // Performance benchmark
    benchmark(Q, K, V, O, L, head_dim, num_iters);

    // Cleanup
    flash_aligned_free(Q);
    flash_aligned_free(K);
    flash_aligned_free(V);
    flash_aligned_free(O);
    flash_aligned_free(O_ref);

    printf("=================================================================\n");
    printf("Benchmark complete\n");
    printf("=================================================================\n");

    return 0;
}
