#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "flash_attn_2pass.h"
#include "pcg32.h"

// ============================================
// Utilities
// ============================================

static void* aligned_alloc_wrapper(size_t align, size_t size) {
    void* ptr = NULL;
    posix_memalign(&ptr, align, size);
    return ptr;
}

static void init_random(float* arr, size_t n, unsigned int seed) {
    pcg32_seed(seed, seed);
    for (size_t i = 0; i < n; i++) {
        arr[i] = pcg32_float_signed();
    }
}

static float max_error(const float* a, const float* b, size_t n) {
    float max_err = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float err = fabsf(a[i] - b[i]);
        if (err > max_err) max_err = err;
    }
    return max_err;
}

static float rel_error(const float* a, const float* b, size_t n) {
    float max_rel = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float ref = fabsf(b[i]) > 1e-6f ? fabsf(b[i]) : 1e-6f;
        float rel = fabsf(a[i] - b[i]) / ref;
        if (rel > max_rel) max_rel = rel;
    }
    return max_rel;
}

static inline uint64_t read_cycle_counter(void) {
    uint64_t val;
    __asm__ volatile("mrs %0, cntvct_el0" : "=r"(val));
    return val;
}

static inline uint64_t get_timer_freq(void) {
    uint64_t freq;
    __asm__ volatile("mrs %0, cntfrq_el0" : "=r"(freq));
    return freq;
}

// Global timer frequency, initialized in main()
static uint64_t g_timer_freq = 0;

static double get_time(void) {
    return (double)read_cycle_counter() / (double)g_timer_freq;
}

static void print_matrix(const char* name, const float* mat, int rows, int cols) {
    printf("%s [%d, %d]:\n", name, rows, cols);
    for (int i = 0; i < rows && i < 4; i++) {
        printf("  [");
        for (int j = 0; j < cols && j < 8; j++) {
            printf("%8.4f ", mat[i * cols + j]);
        }
        if (cols > 8) printf("...");
        printf("]\n");
    }
    if (rows > 4) printf("  ...\n");
}

// ============================================
// Main
// ============================================

int main(int argc, char** argv) {
    // Initialize timer frequency first
    g_timer_freq = get_timer_freq();

    int iterations = 10000;
    int warmup = 1000;
    int verbose = 0;

    if (argc > 1) iterations = atoi(argv[1]);
    if (argc > 2) verbose = atoi(argv[2]);

    printf("==============================================\n");
    printf("2-Pass Flash Attention Benchmark (A64FX)\n");
    printf("==============================================\n\n");

    printf("Tile parameters:\n");
    printf("  BR (query rows): %d\n", BR);
    printf("  BC (KV cols):    %d\n", BC);
    printf("  D  (head dim):   %d\n", D);
    printf("  Iterations:      %d\n\n", iterations);

    // Memory sizes
    size_t Q_size = BR * D * sizeof(float);
    size_t K_size = BC * D * sizeof(float);
    size_t V_size = BC * D * sizeof(float);
    size_t O_size = BR * D * sizeof(float);
    size_t S_size = BR * BC * sizeof(float);

    printf("Memory:\n");
    printf("  Q: %zu bytes\n", Q_size);
    printf("  K: %zu bytes\n", K_size);
    printf("  V: %zu bytes\n", V_size);
    printf("  S: %zu bytes (scratch)\n", S_size);
    printf("  O: %zu bytes\n\n", O_size);

    // Allocate
    float* Q = aligned_alloc_wrapper(ALIGN, Q_size);
    float* K = aligned_alloc_wrapper(ALIGN, K_size);
    float* V = aligned_alloc_wrapper(ALIGN, V_size);
    float* O_ref = aligned_alloc_wrapper(ALIGN, O_size);
    float* O_asm = aligned_alloc_wrapper(ALIGN, O_size);
    float* S_scratch = aligned_alloc_wrapper(ALIGN, S_size);
    float* m = aligned_alloc_wrapper(ALIGN, BR * sizeof(float));
    float* l = aligned_alloc_wrapper(ALIGN, BR * sizeof(float));

    if (!Q || !K || !V || !O_ref || !O_asm || !S_scratch || !m || !l) {
        fprintf(stderr, "Allocation failed\n");
        return 1;
    }

    // Initialize
    init_random(Q, BR * D, 42);
    init_random(K, BC * D, 123);
    init_random(V, BC * D, 456);

    if (verbose) {
        print_matrix("Q", Q, BR, D);
        print_matrix("K", K, BC, D);
        print_matrix("V", V, BC, D);
    }

    // ========================================
    // Correctness test
    // ========================================
    printf("--- Correctness Test ---\n");

    // Reference
    memset(O_ref, 0, O_size);
    flash_attention_ref(Q, K, V, O_ref);

    // ASM
    memset(O_asm, 0, O_size);
    memset(S_scratch, 0, S_size);
    flash_attention_tile(Q, K, V, O_asm, S_scratch, m, l);

    float err_abs = max_error(O_ref, O_asm, BR * D);
    float err_rel = rel_error(O_ref, O_asm, BR * D);
    int pass = err_rel < 0.01f;  // 1% relative error threshold

    printf("ASM vs Reference:\n");
    printf("  Max absolute error: %e\n", err_abs);
    printf("  Max relative error: %e\n", err_rel);
    printf("  Result: %s\n\n", pass ? "[PASS]" : "[FAIL]");

    if (verbose) {
        print_matrix("O_ref", O_ref, BR, D);
        print_matrix("O_asm", O_asm, BR, D);
        printf("m = [%.4f, %.4f, %.4f, %.4f]\n", m[0], m[1], m[2], m[3]);
        printf("l = [%.4f, %.4f, %.4f, %.4f]\n\n", l[0], l[1], l[2], l[3]);

        // Debug: print S values
        printf("S (first 8 cols per row):\n");
        for (int i = 0; i < BR; i++) {
            printf("  Row %d: ", i);
            for (int j = 0; j < 8; j++) {
                printf("%.3f ", S_scratch[i * BC + j]);
            }
            printf("...\n");
        }

        // Debug: compute expected exp sums
        printf("\nExpected exp sums (reference):\n");
        for (int i = 0; i < BR; i++) {
            float sum = 0.0f;
            float row_max = -1e30f;
            for (int j = 0; j < BC; j++) {
                if (S_scratch[i * BC + j] > row_max) row_max = S_scratch[i * BC + j];
            }
            for (int j = 0; j < BC; j++) {
                sum += expf(S_scratch[i * BC + j] - row_max);
            }
            printf("  Row %d: max=%.4f, exp_sum=%.4f\n", i, row_max, sum);
        }
    }

    // ========================================
    // Performance test
    // ========================================
    printf("--- Performance Test ---\n");

    // FLOPs calculation
    // Pass 1: S = Q @ K^T -> BR * BC * D * 2
    // Pass 2 exp: BR * BC * ~8
    // Pass 2 P@V: BR * D * BC * 2
    double flops_pass1 = (double)BR * BC * D * 2;
    double flops_exp = (double)BR * BC * 8;
    double flops_pass2_pv = (double)BR * D * BC * 2;
    double flops_total = flops_pass1 + flops_exp + flops_pass2_pv;

    printf("FLOPs breakdown:\n");
    printf("  Pass 1 (Q@K^T):  %.0f\n", flops_pass1);
    printf("  Exp:             %.0f\n", flops_exp);
    printf("  Pass 2 (P@V):    %.0f\n", flops_pass2_pv);
    printf("  Total:           %.0f\n\n", flops_total);

    // Warmup
    for (int i = 0; i < warmup; i++) {
        flash_attention_tile(Q, K, V, O_asm, S_scratch, m, l);
    }

    // Benchmark combined
    uint64_t c0 = read_cycle_counter();
    for (int i = 0; i < iterations; i++) {
        flash_attention_tile(Q, K, V, O_asm, S_scratch, m, l);
    }
    uint64_t c1 = read_cycle_counter();
    double elapsed = (double)(c1 - c0) / (double)g_timer_freq;
    double gflops = (flops_total * iterations) / elapsed / 1e9;
    double ns_per_call = elapsed / iterations * 1e9;
    double cycles_per_call = ns_per_call * 2.0;  // Assume 2.0 GHz

    printf("Combined (Pass1 + Pass2):\n");
    printf("  Total time:     %.3f ms\n", elapsed * 1000);
    printf("  Per call:       %.1f ns (%.0f cycles @ 2.0GHz)\n", 
           ns_per_call, cycles_per_call);
    printf("  Throughput:     %.2f GFLOPS\n", gflops);
    printf("\n");

    // Benchmark Pass 1 only
    for (int i = 0; i < warmup; i++) {
        pass1_qkt_rowmax(Q, K, S_scratch, m);
    }
    c0 = read_cycle_counter();
    for (int i = 0; i < iterations; i++) {
        pass1_qkt_rowmax(Q, K, S_scratch, m);
    }
    c1 = read_cycle_counter();

    elapsed = (double)(c1 - c0) / (double)g_timer_freq;
    double gflops_p1 = (flops_pass1 * iterations) / elapsed / 1e9;
    double ns_p1 = elapsed / iterations * 1e9;

    printf("Pass 1 (Q@K^T + rowmax):\n");
    printf("  Per call:       %.1f ns (%.0f cycles)\n", ns_p1, ns_p1 * 2.0);
    printf("  Throughput:     %.2f GFLOPS\n", gflops_p1);
    printf("\n");

    // Benchmark Pass 2 only (ASM version - may be broken)
    for (int i = 0; i < warmup; i++) {
        pass2_softmax_pv(S_scratch, V, m, O_asm, l);
    }
    c0 = read_cycle_counter();
    for (int i = 0; i < iterations; i++) {
        pass2_softmax_pv(S_scratch, V, m, O_asm, l);
    }
    c1 = read_cycle_counter();

    elapsed = (double)(c1 - c0) / (double)g_timer_freq;
    double gflops_p2 = ((flops_exp + flops_pass2_pv) * iterations) / elapsed / 1e9;
    double ns_p2 = elapsed / iterations * 1e9;

    printf("Pass 2 (softmax + P@V) [ASM]:\n");
    printf("  Per call:       %.1f ns (%.0f cycles)\n", ns_p2, ns_p2 * 2.0);
    printf("  Throughput:     %.2f GFLOPS\n", gflops_p2);
    printf("\n");

    // ========================================
    // Summary
    // ========================================
    printf("--- Summary ---\n");
    double peak_gflops = 128.0;  // A64FX single core @ 2.0GHz
    printf("A64FX theoretical peak: %.1f GFLOPS (single core @ 2.0GHz)\n", peak_gflops);
    printf("Combined efficiency:    %.1f%%\n", gflops / peak_gflops * 100);
    printf("Pass 1 efficiency:      %.1f%%\n", gflops_p1 / peak_gflops * 100);
    printf("Pass 2 efficiency:      %.1f%%\n", gflops_p2 / peak_gflops * 100);
    printf("\n");

    // Cycle breakdown
    printf("--- Cycle Breakdown ---\n");
    printf("Pass 1: %.0f cycles\n", ns_p1 * 2.0);
    printf("Pass 2: %.0f cycles\n", ns_p2 * 2.0);
    printf("Total:  %.0f cycles\n", (ns_p1 + ns_p2) * 2.0);
    printf("\n");

    // Theoretical analysis
    printf("--- Theoretical Analysis ---\n");
    
    // Pass 1: 64 K rows, each: 4 LD + 16 FMA + 4 FADDV + 4 STR + 4 FMAX
    int p1_fma = BC * (D / VL) * BR;  // 64 * 4 * 4 = 1024 FMA (but fused into 4 per j)
    printf("Pass 1:\n");
    printf("  FMA ops:   %d (per K row: %d)\n", 64 * 16, 16);
    printf("  Min cycles (FMA bound): %d\n", 64 * 16 / 2);
    
    // Pass 2: 4 chunks, each: exp(4*16) + 16*4*4 FMA
    int p2_exp = 4 * (BR * VL);  // 4 * 64 = 256 exp
    int p2_fma = 4 * 16 * BR * (D / VL);  // 4 * 16 * 4 * 4 = 1024 FMA
    printf("Pass 2:\n");
    printf("  Exp ops:   %d\n", p2_exp);
    printf("  FMA ops:   %d\n", p2_fma);
    printf("  Min cycles (FMA bound): %d\n", p2_fma / 2);

    // Cleanup
    free(Q);
    free(K);
    free(V);
    free(O_ref);
    free(O_asm);
    free(S_scratch);
    free(m);
    free(l);

    return pass ? 0 : 1;
}
