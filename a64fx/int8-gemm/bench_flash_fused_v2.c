#include "flash_attention_fused_v2.h"
#include "gqa_pack.h"
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
        printf("  PASS (max diff < 5.0)\n");
    } else {
        printf("  FAIL (max diff >= 5.0)\n");
    }
}

// =============================================================================
// Performance benchmark
// =============================================================================

static void benchmark_original(
    const int8_t* Q,
    const int8_t* K,
    const int8_t* V,
    float* O,
    int64_t L,
    int64_t head_dim,
    int num_iters,
    double* avg_time_out)
{
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

    *avg_time_out = total_time / num_iters;
    printf("  Original (packed V):   avg %.3f ms, min %.3f ms\n", *avg_time_out, min_time);
}

static void benchmark_optimized(
    const int8_t* Q,
    const int8_t* Kp,
    const int8_t* V,
    float* O,
    int64_t L,
    int64_t head_dim,
    int num_iters,
    double* avg_time_out)
{
    // Warmup
    flash_attention_fused_forward_opt(Q, Kp, V, O, NULL, L, head_dim);

    // Timed runs
    double total_time = 0.0;
    double min_time = 1e9;

    for (int iter = 0; iter < num_iters; iter++) {
        double t_start = get_time_ms();
        flash_attention_fused_forward_opt(Q, Kp, V, O, NULL, L, head_dim);
        double t_end = get_time_ms();

        double elapsed = t_end - t_start;
        total_time += elapsed;
        if (elapsed < min_time) min_time = elapsed;
    }

    *avg_time_out = total_time / num_iters;
    printf("  Optimized (L2-block):  avg %.3f ms, min %.3f ms\n", *avg_time_out, min_time);
}

static void compute_metrics(double avg_time, int64_t L, int64_t head_dim) {
    // Compute FLOPS
    // Q@K^T: 2*L*L*head_dim ops (INT8)
    // P@V: 2*L*L*head_dim ops (FP32)
    // Softmax: ~20 ops/elem * L*L
    double gemm_ops = 2.0 * L * L * head_dim;  // Q@K^T
    double softmax_ops = 20.0 * L * L;
    double pv_ops = 2.0 * L * L * head_dim;  // P@V
    double total_ops = gemm_ops + softmax_ops + pv_ops;

    double gops = total_ops / (avg_time * 1e6);

    // Expected peaks:
    // INT8 GEMM: 3072 GOPS (SDOT)
    // FP32 GEMM: 1536 GFLOPS (FMLA)
    double int8_peak = 3072.0;
    double fp32_peak = 1536.0;

    // Rough estimation: assume equal time split
    // Combined peak = harmonic mean weighted by ops
    double combined_peak = (gemm_ops + pv_ops) / (gemm_ops/int8_peak + pv_ops/fp32_peak);
    double efficiency = gops / combined_peak * 100.0;

    printf("  Total throughput: %.1f GOPS\n", gops);
    printf("  Combined efficiency: %.1f%% (weighted peak: %.0f GOPS)\n",
           efficiency, combined_peak);
}

// =============================================================================
// Main
// =============================================================================

int main(int argc, char** argv) {
    // Parse arguments
    int64_t L = 1024;
    int64_t head_dim = 128;
    int num_iters = 5;

    if (argc >= 2) L = atol(argv[1]);
    if (argc >= 3) head_dim = atol(argv[2]);
    if (argc >= 4) num_iters = atoi(argv[3]);

    printf("=================================================================\n");
    printf("Fused FlashAttention V2 Benchmark (with L2-blocked P@V)\n");
    printf("=================================================================\n");
    printf("Configuration:\n");
    printf("  L (sequence length):  %ld\n", L);
    printf("  head_dim:             %ld\n", head_dim);
    printf("  Tile sizes:           TILE_BR=%d, TILE_BC=%d\n", FA_TILE_BR, FA_TILE_BC);
    printf("  P@V kernel:           10x2 microkernel, K-block=256\n");
    printf("  L1D cache:            64 KB\n");
    printf("\n");

    // Allocate matrices
    int8_t* Q = (int8_t*)flash_aligned_alloc(L * head_dim);
    int8_t* K = (int8_t*)flash_aligned_alloc(L * head_dim);
    int8_t* V = (int8_t*)flash_aligned_alloc(L * head_dim);
    float* O = (float*)flash_aligned_alloc(L * head_dim * sizeof(float));
    float* O_opt = (float*)flash_aligned_alloc(L * head_dim * sizeof(float));
    float* O_ref = (float*)flash_aligned_alloc(L * head_dim * sizeof(float));

    // Pre-pack K for optimized version
    size_t Kp_size = flash_kp_size(L, head_dim);
    int8_t* Kp = (int8_t*)flash_aligned_alloc(Kp_size);

    if (!Q || !K || !V || !O || !O_opt || !O_ref || !Kp) {
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

    // Pack K
    pack_k_for_flash_attention(K, Kp, L, head_dim);

    // Correctness check
    if (L <= 512) {
        printf("Running correctness check (L=%ld)...\n", L);

        // Reference
        double ref_start = get_time_ms();
        flash_attention_fused_reference(Q, K, V, O_ref, NULL, L, head_dim);
        double ref_end = get_time_ms();
        printf("  Reference: %.3f ms\n", ref_end - ref_start);

        // Original
        double orig_start = get_time_ms();
        flash_attention_fused_forward(Q, K, V, O, NULL, L, head_dim);
        double orig_end = get_time_ms();
        printf("  Original:  %.3f ms\n", orig_end - orig_start);

        // Optimized
        double opt_start = get_time_ms();
        flash_attention_fused_forward_opt(Q, Kp, V, O_opt, NULL, L, head_dim);
        double opt_end = get_time_ms();
        printf("  Optimized: %.3f ms\n", opt_end - opt_start);

        // Check original vs reference
        check_correctness(O_ref, O, L, head_dim, "Original vs Reference");

        // Check optimized vs reference
        check_correctness(O_ref, O_opt, L, head_dim, "Optimized vs Reference");
    } else {
        printf("Skipping correctness check (L=%ld too large, use L<=512)\n", L);
    }

    // Performance benchmark
    printf("\n=================================================================\n");
    printf("Performance Benchmark (L=%ld, head_dim=%ld, iters=%d)\n", L, head_dim, num_iters);
    printf("=================================================================\n");

    double orig_time, opt_time;

    benchmark_original(Q, K, V, O, L, head_dim, num_iters, &orig_time);
    benchmark_optimized(Q, Kp, V, O_opt, L, head_dim, num_iters, &opt_time);

    printf("\nSpeedup: %.2fx\n", orig_time / opt_time);

    printf("\nOriginal metrics:\n");
    compute_metrics(orig_time, L, head_dim);

    printf("\nOptimized metrics:\n");
    compute_metrics(opt_time, L, head_dim);

    // Breakdown of P@V portion estimate
    printf("\n=================================================================\n");
    printf("P@V Kernel Analysis\n");
    printf("=================================================================\n");
    double pv_ops = 2.0 * L * L * head_dim;
    double pv_gflops_orig = pv_ops / (orig_time * 1e6);
    double pv_gflops_opt = pv_ops / (opt_time * 1e6);
    double fp32_peak = 1536.0;

    // Assuming P@V takes ~50% of total time (rough estimate)
    double pv_time_orig_est = orig_time * 0.5;
    double pv_time_opt_est = opt_time * 0.5;
    double pv_gflops_orig_est = pv_ops / (pv_time_orig_est * 1e6);
    double pv_gflops_opt_est = pv_ops / (pv_time_opt_est * 1e6);

    printf("Assuming P@V takes ~50%% of total time:\n");
    printf("  Original P@V:  ~%.1f GFLOPS (%.1f%% of %.0f GFLOPS peak)\n",
           pv_gflops_orig_est, pv_gflops_orig_est / fp32_peak * 100.0, fp32_peak);
    printf("  Optimized P@V: ~%.1f GFLOPS (%.1f%% of %.0f GFLOPS peak)\n",
           pv_gflops_opt_est, pv_gflops_opt_est / fp32_peak * 100.0, fp32_peak);
    printf("  Target:        ~720 GFLOPS (47%% efficiency from standalone test)\n");

    // Cleanup
    flash_aligned_free(Q);
    flash_aligned_free(K);
    flash_aligned_free(V);
    flash_aligned_free(O);
    flash_aligned_free(O_opt);
    flash_aligned_free(O_ref);
    flash_aligned_free(Kp);

    printf("\n=================================================================\n");
    printf("Benchmark complete\n");
    printf("=================================================================\n");

    return 0;
}
