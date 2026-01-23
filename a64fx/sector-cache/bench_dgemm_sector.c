/*
 * DGEMM Sector Cache Benchmark
 *
 * Compare Fujitsu reference kernel (with sector hints) vs version without
 * to measure the effect of sector cache hints on GEMM performance.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

// External kernel with sector hints (from Fujitsu reference)
extern void dl_gmwn1_base_(int* ma, int* nb_th, int* kk,
                           double* a, double* pb, int* ldb,
                           double* pc, int* ldc);

// External kernel WITHOUT sector hints
extern void dl_gmwn1_no_sector_(int* ma, int* nb_th, int* kk,
                                 double* a, double* pb, int* ldb,
                                 double* pc, int* ldc);

// Timer
static inline uint64_t read_cycle(void) {
    uint64_t val;
    __asm__ volatile("mrs %0, cntvct_el0" : "=r"(val));
    return val;
}

static inline uint64_t get_freq(void) {
    uint64_t freq;
    __asm__ volatile("mrs %0, cntfrq_el0" : "=r"(freq));
    return freq;
}

// Memory barrier
static inline void memory_fence(void) {
    __asm__ volatile("dmb ish" ::: "memory");
}

// Flush cache
static void flush_cache(void* ptr, size_t size) {
    char* p = (char*)ptr;
    for (size_t i = 0; i < size; i += 256) {
        __asm__ volatile("dc civac, %0" :: "r"(p + i) : "memory");
    }
    __asm__ volatile("dsb ish" ::: "memory");
}

// Aligned allocation
static double* alloc_matrix(int rows, int cols) {
    double* ptr = NULL;
    size_t size = (size_t)rows * cols * sizeof(double);
    if (posix_memalign((void**)&ptr, 256, size) != 0) {
        return NULL;
    }
    return ptr;
}

// Initialize matrix
static void init_matrix(double* m, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        m[i] = (double)(rand() % 100) / 100.0;
    }
}

// Clear matrix
static void clear_matrix(double* m, int rows, int cols) {
    memset(m, 0, (size_t)rows * cols * sizeof(double));
}

int main(int argc, char* argv[]) {
    printf("=== DGEMM Sector Cache Comparison ===\n\n");

    uint64_t freq = get_freq();
    printf("Timer frequency: %lu Hz\n", freq);

    // Test sizes - kernel tile is 32x5
    // Use sizes that stress the cache
    int M = 256;
    int N = 160;  // Must be multiple of 5
    int K = 512;

    if (argc > 1) M = atoi(argv[1]);
    if (argc > 2) N = atoi(argv[2]);
    if (argc > 3) K = atoi(argv[3]);

    // Ensure N is multiple of 5
    N = (N / 5) * 5;
    if (N < 5) N = 5;

    printf("Matrix dimensions: M=%d, N=%d, K=%d\n", M, N, K);

    double flops_per_gemm = 2.0 * M * N * K;
    printf("FLOPS per GEMM: %.2f MFLOPS\n", flops_per_gemm / 1e6);

    // Memory sizes
    size_t size_A = (size_t)M * K * sizeof(double);
    size_t size_B = (size_t)K * N * sizeof(double);
    size_t size_C = (size_t)M * N * sizeof(double);
    printf("Memory: A=%.1f KB, B=%.1f KB, C=%.1f KB\n",
           size_A/1024.0, size_B/1024.0, size_C/1024.0);
    printf("L1 cache: 64 KB (32 KB per sector)\n\n");

    // Allocate matrices
    double* A = alloc_matrix(M, K);
    double* B = alloc_matrix(K, N);
    double* C1 = alloc_matrix(M, N);
    double* C2 = alloc_matrix(M, N);

    if (!A || !B || !C1 || !C2) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    // Initialize
    srand(42);
    init_matrix(A, M, K);
    init_matrix(B, K, N);

    // Kernel parameters (from assembly analysis)
    int ma = M;
    int nb_th = N / 5;  // Number of 5-column blocks
    int kk = K / 2;     // K iterations (unrolled by 2)
    int ldb = K;
    int ldc = M;

    printf("Kernel params: ma=%d, nb_th=%d, kk=%d\n\n", ma, nb_th, kk);

    int num_iters = 50;
    double total_flops = flops_per_gemm * num_iters;
    uint64_t start, end;
    double cycles, time_sec, gflops;

    // Peak GFLOPS: 2 FPUs * 2.0 GHz * 2 FMLA/cycle * 8 DP = 64 GFLOPS
    double peak_gflops = 64.0;

    printf("=== Warmup ===\n");
    for (int i = 0; i < 5; i++) {
        clear_matrix(C1, M, N);
        dl_gmwn1_base_(&ma, &nb_th, &kk, A, B, &ldb, C1, &ldc);
    }

    // Test 1: With sector hints
    printf("\n=== Test 1: WITH Sector Cache Hints ===\n");
    printf("A matrix: Tag 0x41 (Sector 0 + prefetch hint)\n");
    printf("C matrix: Tag 0x2  (Sector 1 - streaming)\n\n");

    flush_cache(A, size_A);
    flush_cache(B, size_B);
    flush_cache(C1, size_C);
    memory_fence();

    start = read_cycle();
    for (int iter = 0; iter < num_iters; iter++) {
        clear_matrix(C1, M, N);
        dl_gmwn1_base_(&ma, &nb_th, &kk, A, B, &ldb, C1, &ldc);
    }
    end = read_cycle();

    cycles = (double)(end - start);
    time_sec = cycles / freq;
    gflops = total_flops / time_sec / 1e9;

    printf("Cycles: %.0f\n", cycles);
    printf("Time: %.4f sec\n", time_sec);
    printf("Performance: %.2f GFLOPS (%.1f%% of peak)\n", gflops, gflops/peak_gflops*100);

    double gflops_with_hint = gflops;

    // Test 2: Without sector hints
    printf("\n=== Test 2: WITHOUT Sector Cache Hints ===\n");
    printf("No sector tags applied to pointers\n\n");

    flush_cache(A, size_A);
    flush_cache(B, size_B);
    flush_cache(C2, size_C);
    memory_fence();

    start = read_cycle();
    for (int iter = 0; iter < num_iters; iter++) {
        clear_matrix(C2, M, N);
        dl_gmwn1_no_sector_(&ma, &nb_th, &kk, A, B, &ldb, C2, &ldc);
    }
    end = read_cycle();

    cycles = (double)(end - start);
    time_sec = cycles / freq;
    gflops = total_flops / time_sec / 1e9;

    printf("Cycles: %.0f\n", cycles);
    printf("Time: %.4f sec\n", time_sec);
    printf("Performance: %.2f GFLOPS (%.1f%% of peak)\n", gflops, gflops/peak_gflops*100);

    double gflops_no_hint = gflops;

    // Comparison
    printf("\n=== Comparison ===\n");
    printf("With sector hints:    %.2f GFLOPS\n", gflops_with_hint);
    printf("Without sector hints: %.2f GFLOPS\n", gflops_no_hint);
    printf("Speedup from hints:   %.2fx\n", gflops_with_hint / gflops_no_hint);

    // Verify results match
    double max_diff = 0.0;
    for (int i = 0; i < M * N; i++) {
        double diff = C1[i] - C2[i];
        if (diff < 0) diff = -diff;
        if (diff > max_diff) max_diff = diff;
    }
    printf("\nVerification: max diff = %e (should be ~0)\n", max_diff);

    printf("\n=== Summary ===\n");
    printf("Sector cache hints in GEMM:\n");
    printf("- A matrix (reused tile): Sector 0 keeps data for reuse\n");
    printf("- C matrix (streaming output): Sector 1 for write-through\n");
    printf("- Prevents C writes from evicting A tile data\n");

    // Cleanup
    free(A);
    free(B);
    free(C1);
    free(C2);

    return 0;
}
