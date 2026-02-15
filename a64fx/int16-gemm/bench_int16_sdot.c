// bench_int16_sdot.c
// INT16 SDOT GEMM benchmark for A64FX SVE
// Measures pure SVE SDOT throughput using L1-resident data
//
// INT16 SDOT: SDOT Zda.D, Zn.H, Zm.H
//   - Accumulates 4 int16 products into int64 (per lane)
//   - 8 int64 lanes per 512-bit vector
//   - 32 int16 elements per vector
//   - 32 ops per SDOT (8 lanes x 4 MACs)
//
// Configuration: 5(rows) x 4(simd) x 2(double-buffering)

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <arm_sve.h>

// Blocking parameters
#define MR 5         // Rows of A tile
#define NR 4         // Number of SIMD vectors
#define VL_BYTES 64  // SVE vector length in bytes (A64FX = 512 bits)
#define VL_INT64 8   // Number of int64 lanes per vector
#define VL_INT16 32  // Number of int16 elements per vector

// Timer functions
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

// Aligned allocation
static void* aligned_alloc_wrapper(size_t align, size_t size) {
    void* ptr = NULL;
    posix_memalign(&ptr, align, size);
    return ptr;
}

// ASM kernel prototypes
extern void int16_sdot_kernel_5x4_l1(
    const int16_t* A,
    const int16_t* B,
    int64_t* C,
    int64_t iterations
);

extern void int16_sdot_kernel_5x4_l1_opt(
    const int16_t* A,
    const int16_t* B,
    int64_t* C,
    int64_t iterations
);

// L1 throughput version using C intrinsics
static void int16_sdot_intrinsic_l1(
    const int16_t* __restrict__ A,
    const int16_t* __restrict__ B,
    int64_t* __restrict__ C,
    int64_t iterations
) {
    // 5 rows x 4 vectors = 20 accumulators (int64)
    svint64_t c00 = svdup_s64(0), c01 = svdup_s64(0), c02 = svdup_s64(0), c03 = svdup_s64(0);
    svint64_t c10 = svdup_s64(0), c11 = svdup_s64(0), c12 = svdup_s64(0), c13 = svdup_s64(0);
    svint64_t c20 = svdup_s64(0), c21 = svdup_s64(0), c22 = svdup_s64(0), c23 = svdup_s64(0);
    svint64_t c30 = svdup_s64(0), c31 = svdup_s64(0), c32 = svdup_s64(0), c33 = svdup_s64(0);
    svint64_t c40 = svdup_s64(0), c41 = svdup_s64(0), c42 = svdup_s64(0), c43 = svdup_s64(0);

    svbool_t pg16 = svptrue_b16();
    svbool_t pg64 = svptrue_b64();

    // Double-buffered loads
    svint16_t a0_buf0, a1_buf0, a2_buf0, a3_buf0, a4_buf0;
    svint16_t b0_buf0, b1_buf0, b2_buf0, b3_buf0;
    svint16_t a0_buf1, a1_buf1, a2_buf1, a3_buf1, a4_buf1;
    svint16_t b0_buf1, b1_buf1, b2_buf1, b3_buf1;

    // Initial loads
    // A: 5 rows, each row needs 2 int16 values replicated (for SDOT .d which uses 2 int16 per int64)
    // Use svld1rq to load 8 int16 (16 bytes) and replicate
    a0_buf0 = svld1rq_s16(pg16, A + 0);
    a1_buf0 = svld1rq_s16(pg16, A + 8);
    a2_buf0 = svld1rq_s16(pg16, A + 16);
    a3_buf0 = svld1rq_s16(pg16, A + 24);
    a4_buf0 = svld1rq_s16(pg16, A + 32);

    // B: 4 vectors of 32 int16 each
    b0_buf0 = svld1_s16(pg16, B + 0);
    b1_buf0 = svld1_s16(pg16, B + 32);
    b2_buf0 = svld1_s16(pg16, B + 64);
    b3_buf0 = svld1_s16(pg16, B + 96);

    // Main loop
    for (int64_t i = 0; i < iterations; i += 2) {
        // === Iteration 0: use buffer 0, load buffer 1 ===
        a0_buf1 = svld1rq_s16(pg16, A + 0);
        a1_buf1 = svld1rq_s16(pg16, A + 8);
        c00 = svdot_s64(c00, a0_buf0, b0_buf0);
        c01 = svdot_s64(c01, a0_buf0, b1_buf0);
        c02 = svdot_s64(c02, a0_buf0, b2_buf0);
        c03 = svdot_s64(c03, a0_buf0, b3_buf0);

        a2_buf1 = svld1rq_s16(pg16, A + 16);
        a3_buf1 = svld1rq_s16(pg16, A + 24);
        c10 = svdot_s64(c10, a1_buf0, b0_buf0);
        c11 = svdot_s64(c11, a1_buf0, b1_buf0);
        c12 = svdot_s64(c12, a1_buf0, b2_buf0);
        c13 = svdot_s64(c13, a1_buf0, b3_buf0);

        a4_buf1 = svld1rq_s16(pg16, A + 32);
        b0_buf1 = svld1_s16(pg16, B + 0);
        c20 = svdot_s64(c20, a2_buf0, b0_buf0);
        c21 = svdot_s64(c21, a2_buf0, b1_buf0);
        c22 = svdot_s64(c22, a2_buf0, b2_buf0);
        c23 = svdot_s64(c23, a2_buf0, b3_buf0);

        b1_buf1 = svld1_s16(pg16, B + 32);
        b2_buf1 = svld1_s16(pg16, B + 64);
        c30 = svdot_s64(c30, a3_buf0, b0_buf0);
        c31 = svdot_s64(c31, a3_buf0, b1_buf0);
        c32 = svdot_s64(c32, a3_buf0, b2_buf0);
        c33 = svdot_s64(c33, a3_buf0, b3_buf0);

        b3_buf1 = svld1_s16(pg16, B + 96);
        c40 = svdot_s64(c40, a4_buf0, b0_buf0);
        c41 = svdot_s64(c41, a4_buf0, b1_buf0);
        c42 = svdot_s64(c42, a4_buf0, b2_buf0);
        c43 = svdot_s64(c43, a4_buf0, b3_buf0);

        // === Iteration 1: use buffer 1, load buffer 0 ===
        a0_buf0 = svld1rq_s16(pg16, A + 0);
        a1_buf0 = svld1rq_s16(pg16, A + 8);
        c00 = svdot_s64(c00, a0_buf1, b0_buf1);
        c01 = svdot_s64(c01, a0_buf1, b1_buf1);
        c02 = svdot_s64(c02, a0_buf1, b2_buf1);
        c03 = svdot_s64(c03, a0_buf1, b3_buf1);

        a2_buf0 = svld1rq_s16(pg16, A + 16);
        a3_buf0 = svld1rq_s16(pg16, A + 24);
        c10 = svdot_s64(c10, a1_buf1, b0_buf1);
        c11 = svdot_s64(c11, a1_buf1, b1_buf1);
        c12 = svdot_s64(c12, a1_buf1, b2_buf1);
        c13 = svdot_s64(c13, a1_buf1, b3_buf1);

        a4_buf0 = svld1rq_s16(pg16, A + 32);
        b0_buf0 = svld1_s16(pg16, B + 0);
        c20 = svdot_s64(c20, a2_buf1, b0_buf1);
        c21 = svdot_s64(c21, a2_buf1, b1_buf1);
        c22 = svdot_s64(c22, a2_buf1, b2_buf1);
        c23 = svdot_s64(c23, a2_buf1, b3_buf1);

        b1_buf0 = svld1_s16(pg16, B + 32);
        b2_buf0 = svld1_s16(pg16, B + 64);
        c30 = svdot_s64(c30, a3_buf1, b0_buf1);
        c31 = svdot_s64(c31, a3_buf1, b1_buf1);
        c32 = svdot_s64(c32, a3_buf1, b2_buf1);
        c33 = svdot_s64(c33, a3_buf1, b3_buf1);

        b3_buf0 = svld1_s16(pg16, B + 96);
        c40 = svdot_s64(c40, a4_buf1, b0_buf1);
        c41 = svdot_s64(c41, a4_buf1, b1_buf1);
        c42 = svdot_s64(c42, a4_buf1, b2_buf1);
        c43 = svdot_s64(c43, a4_buf1, b3_buf1);
    }

    // Store results
    const int64_t C_row_stride = NR * VL_INT64;  // 32 int64 per row
    svst1_s64(pg64, C + 0 * C_row_stride + 0 * VL_INT64, c00);
    svst1_s64(pg64, C + 0 * C_row_stride + 1 * VL_INT64, c01);
    svst1_s64(pg64, C + 0 * C_row_stride + 2 * VL_INT64, c02);
    svst1_s64(pg64, C + 0 * C_row_stride + 3 * VL_INT64, c03);

    svst1_s64(pg64, C + 1 * C_row_stride + 0 * VL_INT64, c10);
    svst1_s64(pg64, C + 1 * C_row_stride + 1 * VL_INT64, c11);
    svst1_s64(pg64, C + 1 * C_row_stride + 2 * VL_INT64, c12);
    svst1_s64(pg64, C + 1 * C_row_stride + 3 * VL_INT64, c13);

    svst1_s64(pg64, C + 2 * C_row_stride + 0 * VL_INT64, c20);
    svst1_s64(pg64, C + 2 * C_row_stride + 1 * VL_INT64, c21);
    svst1_s64(pg64, C + 2 * C_row_stride + 2 * VL_INT64, c22);
    svst1_s64(pg64, C + 2 * C_row_stride + 3 * VL_INT64, c23);

    svst1_s64(pg64, C + 3 * C_row_stride + 0 * VL_INT64, c30);
    svst1_s64(pg64, C + 3 * C_row_stride + 1 * VL_INT64, c31);
    svst1_s64(pg64, C + 3 * C_row_stride + 2 * VL_INT64, c32);
    svst1_s64(pg64, C + 3 * C_row_stride + 3 * VL_INT64, c33);

    svst1_s64(pg64, C + 4 * C_row_stride + 0 * VL_INT64, c40);
    svst1_s64(pg64, C + 4 * C_row_stride + 1 * VL_INT64, c41);
    svst1_s64(pg64, C + 4 * C_row_stride + 2 * VL_INT64, c42);
    svst1_s64(pg64, C + 4 * C_row_stride + 3 * VL_INT64, c43);
}

int main(int argc, char** argv) {
    uint64_t timer_freq = get_timer_freq();

    int64_t iterations = 1000000;
    int warmup = 10000;

    if (argc > 1) iterations = atoll(argv[1]);

    printf("==============================================\n");
    printf("INT16 SDOT GEMM Benchmark (A64FX SVE)\n");
    printf("==============================================\n\n");

    printf("Configuration:\n");
    printf("  MR (rows):         %d\n", MR);
    printf("  NR (SIMD vectors): %d\n", NR);
    printf("  VL_INT64:          %d (int64 lanes per vector)\n", VL_INT64);
    printf("  VL_INT16:          %d (int16 elements per vector)\n", VL_INT16);
    printf("  C tile size:       %d x %d = %d int64 values\n", MR, NR * VL_INT64, MR * NR * VL_INT64);
    printf("  Timer freq:        %lu Hz\n", timer_freq);
    printf("  Iterations:        %ld\n\n", iterations);

    // Allocate L1-resident buffers
    // A: [MR, 8] = [5, 8] = 40 int16 = 80 bytes (for ld1rq, 16-byte chunks)
    // B: [NR*VL_INT16] = [128] = 128 int16 = 256 bytes (4 vectors)
    // C: [MR, NR*VL_INT64] = [5, 32] = 160 int64 = 1280 bytes
    size_t A_size = MR * 8 * sizeof(int16_t);  // 80 bytes
    size_t B_size = NR * VL_INT16 * sizeof(int16_t);  // 256 bytes
    size_t C_size = MR * NR * VL_INT64 * sizeof(int64_t);  // 1280 bytes

    int16_t* A = (int16_t*)aligned_alloc_wrapper(64, A_size);
    int16_t* B = (int16_t*)aligned_alloc_wrapper(64, B_size);
    int64_t* C = (int64_t*)aligned_alloc_wrapper(64, C_size);

    if (!A || !B || !C) {
        fprintf(stderr, "Allocation failed\n");
        return 1;
    }

    // Initialize
    for (size_t i = 0; i < MR * 8; i++) {
        A[i] = (int16_t)(i % 7 - 3);
    }
    for (size_t i = 0; i < NR * VL_INT16; i++) {
        B[i] = (int16_t)(i % 5 - 2);
    }
    memset(C, 0, C_size);

    printf("Buffer sizes (all L1 resident):\n");
    printf("  A: %zu bytes\n", A_size);
    printf("  B: %zu bytes\n", B_size);
    printf("  C: %zu bytes\n", C_size);
    printf("  Total: %zu bytes (L1 = 64KB per core)\n\n", A_size + B_size + C_size);

    // ========================================
    // Performance metrics (counting MACs as ops)
    // ========================================
    // Per iteration:
    //   - 5 rows x 4 vectors = 20 SDOT instructions
    //   - Each SDOT: 8 int64 lanes x 4 int16 MACs = 32 ops
    //   - Total: 20 x 32 = 640 int16 ops per iteration
    //
    // A64FX theoretical peak INT16:
    //   - 2 SDOT/cycle x 32 ops/SDOT x 2.0 GHz = 128 GOPS
    //
    // Comparison:
    //   - INT8:  16 lanes x 4 MACs = 64 ops/SDOT -> 256 GOPS peak
    //   - INT16:  8 lanes x 4 MACs = 32 ops/SDOT -> 128 GOPS peak

    double ops_per_iter = (double)MR * NR * VL_INT64 * 4;  // 5 * 4 * 8 * 4 = 640 ops
    printf("Ops per iteration: %.0f INT16 ops (MACs)\n", ops_per_iter);
    printf("SDOT per iteration: %d\n\n", MR * NR);

    // ========================================
    // Benchmark C intrinsic L1 version
    // ========================================
    printf("--- Benchmark: C Intrinsic (L1 same address) ---\n");

    // Warmup
    for (int i = 0; i < warmup; i++) {
        int16_sdot_intrinsic_l1(A, B, C, 2);
    }
    __asm__ volatile("isb" ::: "memory");

    uint64_t t0 = read_cycle_counter();
    int16_sdot_intrinsic_l1(A, B, C, iterations);
    uint64_t t1 = read_cycle_counter();

    uint64_t delta = t1 - t0;
    double elapsed = (double)delta / (double)timer_freq;
    double total_ops = ops_per_iter * (double)iterations;
    double gops = total_ops / elapsed / 1e9;
    double ns_per_iter = elapsed / (double)iterations * 1e9;
    double cycles_per_iter = ns_per_iter * 2.0;

    printf("  Time:           %.3f ms\n", elapsed * 1000);
    printf("  Per iteration:  %.2f ns (%.1f cycles @ 2.0GHz)\n", ns_per_iter, cycles_per_iter);
    printf("  Throughput:     %.2f GOPS (INT16)\n", gops);
    printf("  SDOT/cycle:     %.2f\n", (double)(MR * NR) / cycles_per_iter);
    printf("\n");

#ifdef HAS_ASM_KERNEL
    // ========================================
    // Benchmark ASM kernel (double-buffered)
    // ========================================
    printf("--- Benchmark: ASM Kernel (L1 same address, double-buffered) ---\n");

    memset(C, 0, C_size);

    for (int i = 0; i < warmup; i++) {
        int16_sdot_kernel_5x4_l1(A, B, C, 2);
    }
    __asm__ volatile("isb" ::: "memory");

    t0 = read_cycle_counter();
    int16_sdot_kernel_5x4_l1(A, B, C, iterations);
    t1 = read_cycle_counter();

    delta = t1 - t0;
    elapsed = (double)delta / (double)timer_freq;
    gops = total_ops / elapsed / 1e9;
    ns_per_iter = elapsed / (double)iterations * 1e9;
    cycles_per_iter = ns_per_iter * 2.0;

    printf("  Time:           %.3f ms\n", elapsed * 1000);
    printf("  Per iteration:  %.2f ns (%.1f cycles @ 2.0GHz)\n", ns_per_iter, cycles_per_iter);
    printf("  Throughput:     %.2f GOPS (INT16)\n", gops);
    printf("  SDOT/cycle:     %.2f\n", (double)(MR * NR) / cycles_per_iter);
    printf("\n");

    // ========================================
    // Benchmark optimized ASM kernel
    // ========================================
    printf("--- Benchmark: ASM Kernel Optimized (pure SDOT) ---\n");

    memset(C, 0, C_size);

    for (int i = 0; i < warmup; i++) {
        int16_sdot_kernel_5x4_l1_opt(A, B, C, 2);
    }
    __asm__ volatile("isb" ::: "memory");

    t0 = read_cycle_counter();
    int16_sdot_kernel_5x4_l1_opt(A, B, C, iterations);
    t1 = read_cycle_counter();

    delta = t1 - t0;
    elapsed = (double)delta / (double)timer_freq;
    gops = total_ops / elapsed / 1e9;
    ns_per_iter = elapsed / (double)iterations * 1e9;
    cycles_per_iter = ns_per_iter * 2.0;

    printf("  Time:           %.3f ms\n", elapsed * 1000);
    printf("  Per iteration:  %.2f ns (%.1f cycles @ 2.0GHz)\n", ns_per_iter, cycles_per_iter);
    printf("  Throughput:     %.2f GOPS (INT16)\n", gops);
    printf("  SDOT/cycle:     %.2f\n", (double)(MR * NR) / cycles_per_iter);
    printf("\n");
#endif

    // ========================================
    // Summary
    // ========================================
    printf("--- Summary ---\n");
    printf("A64FX SVE INT16 SDOT:\n");
    printf("  - Vector length: 512 bits = 32 int16 = 8 int64 lanes\n");
    printf("  - SDOT: accumulates 4 int16 products into int64\n");
    printf("  - Blocking: 5x4 = 20 accumulators (z0-z19)\n");
    printf("\n");
    printf("Theoretical peak (2.0 GHz, 2 SDOT/cycle, counting MACs):\n");
    printf("  - Per SDOT: 8 lanes x 4 MACs = 32 ops\n");
    printf("  - Peak: 2 x 32 x 2.0G = 128 GOPS\n");
    printf("\n");
    printf("Comparison with INT8 SDOT:\n");
    printf("  - INT8:  16 lanes x 4 MACs = 64 ops/SDOT, peak 256 GOPS\n");
    printf("  - INT16:  8 lanes x 4 MACs = 32 ops/SDOT, peak 128 GOPS\n");
    printf("  - Ratio: INT8 has 2x more ops per SDOT (2x more lanes)\n");
    printf("\n");

    // Cleanup
    free(A);
    free(B);
    free(C);

    return 0;
}
