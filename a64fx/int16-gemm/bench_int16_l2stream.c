// bench_int16_l2stream.c
// INT16 SDOT L2 streaming benchmark for A64FX SVE
// Tests sustained throughput with B matrix streaming from L2
//
// INT16 SDOT: sdot z.d, zn.h, zm.h
//   - 8 int64 lanes, 4 MACs per lane = 32 ops per SDOT
//   - Peak: 2 SDOT/cycle x 32 ops x 2 GHz = 128 GOPS

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#define MR 5
#define NR 4
#define VL_BYTES 64
#define VL_INT64 8
#define VL_INT16 32

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

static void* aligned_alloc_wrapper(size_t align, size_t size) {
    void* ptr = NULL;
    posix_memalign(&ptr, align, size);
    return ptr;
}

// ASM kernel prototypes
extern void int16_sdot_kernel_5x4_l2(
    const int16_t* A,
    const int16_t* B,
    int64_t* C,
    int64_t k_iters
);

extern void int16_sdot_kernel_5x4_l2_db2(
    const int16_t* A,
    const int16_t* B,
    int64_t* C,
    int64_t k_iters
);

extern void int16_sdot_bench_l2(
    const int16_t* A,
    const int16_t* B,
    int64_t* C,
    int64_t k_iters,
    int64_t outer_iters
);

extern void int16_sdot_bench_l2_db(
    const int16_t* A,
    const int16_t* B,
    int64_t* C,
    int64_t k_iters,
    int64_t outer_iters
);

// Reference implementation for correctness check
static void reference_gemm(
    const int16_t* A,
    const int16_t* B,
    int64_t* C,
    int64_t K
) {
    // A: [MR, K] = [5, K]
    // B: [K/4, NR*VL_INT16] = [K/4, 128], but organized for SDOT
    // C: [MR, NR*VL_INT64] = [5, 32]

    // Initialize C
    for (int m = 0; m < MR; m++) {
        for (int n = 0; n < NR * VL_INT64; n++) {
            C[m * NR * VL_INT64 + n] = 0;
        }
    }

    // GEMM: each C[m,n] is sum of A[m,k] * B[k,n] products
    // But B is organized for SDOT: B[k_iter, vec, lane*4 + idx]
    // Each k_iter processes 4 consecutive k values

    int64_t k_iters = K / 4;

    for (int64_t ki = 0; ki < k_iters; ki++) {
        for (int m = 0; m < MR; m++) {
            // A values for this row and k iteration: A[m, ki*4 : ki*4+4]
            int16_t a0 = A[m * K + ki * 4 + 0];
            int16_t a1 = A[m * K + ki * 4 + 1];
            int16_t a2 = A[m * K + ki * 4 + 2];
            int16_t a3 = A[m * K + ki * 4 + 3];

            // For each output vector (NR vectors, VL_INT64 lanes each)
            for (int v = 0; v < NR; v++) {
                for (int lane = 0; lane < VL_INT64; lane++) {
                    // B layout: B[ki * NR * VL_INT16 + v * VL_INT16 + lane * 4 + idx]
                    // Each int64 lane uses 4 consecutive int16 from B
                    int64_t b_base = ki * NR * VL_INT16 + v * VL_INT16 + lane * 4;
                    int16_t b0 = B[b_base + 0];
                    int16_t b1 = B[b_base + 1];
                    int16_t b2 = B[b_base + 2];
                    int16_t b3 = B[b_base + 3];

                    // SDOT: accumulate 4 products
                    int64_t prod = (int64_t)a0 * b0 + (int64_t)a1 * b1 +
                                   (int64_t)a2 * b2 + (int64_t)a3 * b3;

                    C[m * NR * VL_INT64 + v * VL_INT64 + lane] += prod;
                }
            }
        }
    }
}

int main(int argc, char** argv) {
    uint64_t timer_freq = get_timer_freq();

    int64_t K = 512;        // K dimension (multiple of 4)
    int64_t outer = 10000;  // Outer iterations for benchmark
    int warmup = 1000;

    if (argc > 1) K = atoll(argv[1]);
    if (argc > 2) outer = atoll(argv[2]);

    // K must be multiple of 4
    K = (K + 3) / 4 * 4;
    int64_t k_iters = K / 4;

    printf("==============================================\n");
    printf("INT16 SDOT L2 Streaming Benchmark (A64FX SVE)\n");
    printf("==============================================\n\n");

    printf("Configuration:\n");
    printf("  MR (rows):         %d\n", MR);
    printf("  NR (SIMD vectors): %d\n", NR);
    printf("  K dimension:       %ld\n", K);
    printf("  K/4 iterations:    %ld\n", k_iters);
    printf("  Outer iterations:  %ld\n", outer);
    printf("  Timer freq:        %lu Hz\n\n", timer_freq);

    // Memory sizes
    // A: [MR, K] int16 = 5 * K * 2 bytes
    // B: [K/4, NR*VL_INT16] int16 = k_iters * 128 * 2 bytes
    // C: [MR, NR*VL_INT64] int64 = 5 * 32 * 8 bytes
    size_t A_size = MR * K * sizeof(int16_t);
    size_t B_size = k_iters * NR * VL_INT16 * sizeof(int16_t);
    size_t C_size = MR * NR * VL_INT64 * sizeof(int64_t);

    printf("Memory sizes:\n");
    printf("  A: %zu bytes (%.2f KB) - L1 resident\n", A_size, A_size / 1024.0);
    printf("  B: %zu bytes (%.2f KB) - L2 streaming\n", B_size, B_size / 1024.0);
    printf("  C: %zu bytes (%.2f KB)\n", C_size, C_size / 1024.0);
    printf("  Total: %.2f KB\n\n", (A_size + B_size + C_size) / 1024.0);

    // Allocate
    int16_t* A = (int16_t*)aligned_alloc_wrapper(64, A_size);
    int16_t* B = (int16_t*)aligned_alloc_wrapper(64, B_size);
    int64_t* C = (int64_t*)aligned_alloc_wrapper(64, C_size);
    int64_t* C_ref = (int64_t*)aligned_alloc_wrapper(64, C_size);

    if (!A || !B || !C || !C_ref) {
        fprintf(stderr, "Allocation failed\n");
        return 1;
    }

    // Initialize with small values to avoid overflow
    for (size_t i = 0; i < MR * K; i++) {
        A[i] = (int16_t)((i % 7) - 3);  // -3 to 3
    }
    for (size_t i = 0; i < k_iters * NR * VL_INT16; i++) {
        B[i] = (int16_t)((i % 5) - 2);  // -2 to 2
    }
    memset(C, 0, C_size);
    memset(C_ref, 0, C_size);

    // ========================================
    // Correctness check
    // ========================================
    printf("--- Correctness Check ---\n");

    // Reference
    reference_gemm(A, B, C_ref, K);

    // ASM kernel
    int16_sdot_kernel_5x4_l2(A, B, C, k_iters);

    // Compare
    int mismatches = 0;
    for (int i = 0; i < MR * NR * VL_INT64; i++) {
        if (C[i] != C_ref[i]) {
            if (mismatches < 10) {
                printf("  Mismatch at [%d]: got %ld, expected %ld\n",
                       i, C[i], C_ref[i]);
            }
            mismatches++;
        }
    }

    if (mismatches == 0) {
        printf("  PASS: All %d values match\n\n", MR * NR * VL_INT64);
    } else {
        printf("  FAIL: %d mismatches out of %d\n\n", mismatches, MR * NR * VL_INT64);
    }

    // ========================================
    // Performance benchmark
    // ========================================
    printf("--- Performance Benchmark ---\n");

    // Ops per outer iteration:
    //   - k_iters inner loops
    //   - Each inner: 20 SDOT
    //   - Each SDOT: 32 ops (8 lanes x 4 MACs)
    double ops_per_outer = (double)k_iters * MR * NR * VL_INT64 * 4;
    double total_ops = ops_per_outer * (double)outer;

    // B data read per outer iteration
    double b_bytes_per_outer = (double)B_size;
    double total_b_bytes = b_bytes_per_outer * (double)outer;

    printf("  Ops per outer iteration: %.0f\n", ops_per_outer);
    printf("  B bytes per outer: %.0f\n\n", b_bytes_per_outer);

    // Warmup
    for (int i = 0; i < warmup; i++) {
        int16_sdot_kernel_5x4_l2(A, B, C, k_iters);
    }
    __asm__ volatile("isb" ::: "memory");

    // Benchmark
    uint64_t t0 = read_cycle_counter();
    int16_sdot_bench_l2(A, B, C, k_iters, outer);
    uint64_t t1 = read_cycle_counter();

    uint64_t delta = t1 - t0;
    double elapsed = (double)delta / (double)timer_freq;
    double gops = total_ops / elapsed / 1e9;
    double bw_gb = total_b_bytes / elapsed / 1e9;

    // Per-iteration metrics
    double ns_per_outer = elapsed / (double)outer * 1e9;
    double cycles_per_outer = ns_per_outer * 2.0;
    double sdot_per_outer = (double)k_iters * MR * NR;
    double sdot_per_cycle = sdot_per_outer / cycles_per_outer;

    printf("Results (Basic L2 Streaming):\n");
    printf("  Total time:       %.3f ms\n", elapsed * 1000);
    printf("  Per outer iter:   %.2f ns (%.1f cycles @ 2.0GHz)\n",
           ns_per_outer, cycles_per_outer);
    printf("  Throughput:       %.2f GOPS (INT16)\n", gops);
    printf("  B bandwidth:      %.2f GB/s\n", bw_gb);
    printf("  SDOT/cycle:       %.2f\n", sdot_per_cycle);
    printf("  Efficiency:       %.1f%% of 128 GOPS peak\n", gops / 128.0 * 100);
    printf("\n");

    // ========================================
    // Double-buffered benchmark
    // ========================================
    printf("--- Double-Buffered Benchmark ---\n");

    // Correctness check for double-buffered
    memset(C, 0, C_size);
    int16_sdot_kernel_5x4_l2_db2(A, B, C, k_iters);

    int mismatches_db = 0;
    for (int i = 0; i < MR * NR * VL_INT64; i++) {
        if (C[i] != C_ref[i]) {
            if (mismatches_db < 5) {
                printf("  DB Mismatch at [%d]: got %ld, expected %ld\n",
                       i, C[i], C_ref[i]);
            }
            mismatches_db++;
        }
    }
    if (mismatches_db == 0) {
        printf("  Correctness: PASS\n");
    } else {
        printf("  Correctness: FAIL (%d mismatches)\n", mismatches_db);
    }

    // Warmup
    for (int i = 0; i < warmup; i++) {
        int16_sdot_kernel_5x4_l2_db2(A, B, C, k_iters);
    }
    __asm__ volatile("isb" ::: "memory");

    // Benchmark double-buffered
    t0 = read_cycle_counter();
    int16_sdot_bench_l2_db(A, B, C, k_iters, outer);
    t1 = read_cycle_counter();

    delta = t1 - t0;
    elapsed = (double)delta / (double)timer_freq;
    gops = total_ops / elapsed / 1e9;
    bw_gb = total_b_bytes / elapsed / 1e9;
    ns_per_outer = elapsed / (double)outer * 1e9;
    cycles_per_outer = ns_per_outer * 2.0;
    sdot_per_cycle = sdot_per_outer / cycles_per_outer;

    printf("Results (Double-Buffered):\n");
    printf("  Total time:       %.3f ms\n", elapsed * 1000);
    printf("  Per outer iter:   %.2f ns (%.1f cycles @ 2.0GHz)\n",
           ns_per_outer, cycles_per_outer);
    printf("  Throughput:       %.2f GOPS (INT16)\n", gops);
    printf("  B bandwidth:      %.2f GB/s\n", bw_gb);
    printf("  SDOT/cycle:       %.2f\n", sdot_per_cycle);
    printf("  Efficiency:       %.1f%% of 128 GOPS peak\n", gops / 128.0 * 100);
    printf("\n");

    // ========================================
    // Sweep K values (compare basic vs double-buffered)
    // ========================================
    printf("--- K Sweep: Basic vs Double-Buffered (outer=%ld) ---\n", outer);
    printf("%8s %12s %12s %12s %12s\n", "K", "Basic GOPS", "DB GOPS", "Basic Eff", "DB Eff");

    int k_values[] = {64, 128, 256, 512, 1024, 2048};
    int num_k = sizeof(k_values) / sizeof(k_values[0]);

    for (int ki = 0; ki < num_k; ki++) {
        int64_t test_K = k_values[ki];
        int64_t test_k_iters = test_K / 4;

        // Reallocate B for new K
        size_t test_B_size = test_k_iters * NR * VL_INT16 * sizeof(int16_t);
        int16_t* test_B = (int16_t*)aligned_alloc_wrapper(64, test_B_size);
        int16_t* test_A = (int16_t*)aligned_alloc_wrapper(64, MR * test_K * sizeof(int16_t));

        for (size_t i = 0; i < MR * test_K; i++) {
            test_A[i] = (int16_t)((i % 7) - 3);
        }
        for (size_t i = 0; i < test_k_iters * NR * VL_INT16; i++) {
            test_B[i] = (int16_t)((i % 5) - 2);
        }

        double test_ops = (double)test_k_iters * MR * NR * VL_INT64 * 4 * outer;

        // Benchmark basic
        for (int i = 0; i < 100; i++) {
            int16_sdot_kernel_5x4_l2(test_A, test_B, C, test_k_iters);
        }
        __asm__ volatile("isb" ::: "memory");

        t0 = read_cycle_counter();
        int16_sdot_bench_l2(test_A, test_B, C, test_k_iters, outer);
        t1 = read_cycle_counter();

        delta = t1 - t0;
        elapsed = (double)delta / (double)timer_freq;
        double basic_gops = test_ops / elapsed / 1e9;
        double basic_eff = basic_gops / 128.0 * 100;

        // Benchmark double-buffered
        for (int i = 0; i < 100; i++) {
            int16_sdot_kernel_5x4_l2_db2(test_A, test_B, C, test_k_iters);
        }
        __asm__ volatile("isb" ::: "memory");

        t0 = read_cycle_counter();
        int16_sdot_bench_l2_db(test_A, test_B, C, test_k_iters, outer);
        t1 = read_cycle_counter();

        delta = t1 - t0;
        elapsed = (double)delta / (double)timer_freq;
        double db_gops = test_ops / elapsed / 1e9;
        double db_eff = db_gops / 128.0 * 100;

        printf("%8ld %12.2f %12.2f %11.1f%% %11.1f%%\n",
               test_K, basic_gops, db_gops, basic_eff, db_eff);

        free(test_A);
        free(test_B);
    }
    printf("\n");

    // ========================================
    // Summary
    // ========================================
    printf("--- Summary ---\n");
    printf("INT16 SDOT L2 Streaming:\n");
    printf("  - 8 int64 lanes x 4 MACs = 32 ops/SDOT\n");
    printf("  - Peak: 128 GOPS (2 SDOT/cycle x 32 ops x 2 GHz)\n");
    printf("  - Uses sector cache hints for A/B/C separation\n");
    printf("  - Software prefetch for B matrix streaming\n");
    printf("\n");

    // Cleanup
    free(A);
    free(B);
    free(C);
    free(C_ref);

    return 0;
}
