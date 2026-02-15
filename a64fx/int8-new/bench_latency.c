// bench_latency.c - Compare latency-optimized kernels
// Tests 5x4 double-buffer vs 6x4 OoO for hiding 11-cycle SVE load latency
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

static inline uint64_t rdtsc(void) {
    uint64_t t;
    asm volatile("mrs %0, CNTVCT_EL0" : "=r"(t));
    return t;
}

// Baseline 6x4 (no interleaving)
extern void kernel_6x4_same_addr(const int8_t* A, const int8_t* B, int32_t* C, int K_iters, int ldc);

// Interleaved 6x4 (A loads interleaved with SDOT)
extern void kernel_6x4_interleave_same(const int8_t* A, const int8_t* B, int32_t* C, int K_iters, int ldc);

// New: 6x4 OoO (exploits register renaming for post-SDOT A loads)
extern void kernel_6x4_ooo_same(const int8_t* A, const int8_t* B, int32_t* C, int K_iters, int ldc);

// New: 5x4 double-buffer (double-buffers A to hide latency)
extern void kernel_5x4_dblbuf_same(const int8_t* A, const int8_t* B, int32_t* C, int K_iters, int ldc);

#define MR6 6
#define MR5 5
#define NR 64

int main(void) {
    printf("=== Latency-Optimized Kernel Comparison ===\n");
    printf("Timer: 100MHz, CPU: 2GHz (ratio=20)\n");
    printf("SVE load latency: 11 cycles (constraint being addressed)\n\n");

    // Allocate buffers (small, L1D-resident)
    int8_t* A6 = aligned_alloc(256, 256);  // For 6x4 kernels
    int8_t* A5 = aligned_alloc(256, 256);  // For 5x4 kernel
    int8_t* B = aligned_alloc(256, 512);
    int32_t* C6 = aligned_alloc(256, MR6 * NR * 4);
    int32_t* C5 = aligned_alloc(256, MR5 * NR * 4);

    // Initialize with small values
    for (int i = 0; i < 24; i++) A6[i] = (i % 7) - 3;
    for (int i = 0; i < 20; i++) A5[i] = (i % 7) - 3;
    for (int i = 0; i < 256; i++) B[i] = ((i * 3) % 7) - 3;

    int k_iters = 10000000;

    printf("=== Same-Address Test (Pure L1D) ===\n\n");

    // ========================================
    // 6x4 kernels: 24 SDOT per iteration, ideal = 12 cycles
    // ========================================
    printf("--- 6x4 Kernels (24 SDOT/iter, ideal 12 cyc/iter) ---\n");

    // Warmup
    kernel_6x4_same_addr(A6, B, C6, 100000, NR * 4);
    kernel_6x4_interleave_same(A6, B, C6, 100000, NR * 4);
    kernel_6x4_ooo_same(A6, B, C6, 100000, NR * 4);

    // 6x4 Baseline
    memset(C6, 0, MR6 * NR * 4);
    uint64_t t0 = rdtsc();
    kernel_6x4_same_addr(A6, B, C6, k_iters, NR * 4);
    uint64_t t1 = rdtsc();
    uint64_t ticks_baseline = t1 - t0;
    double cyc_baseline = (double)ticks_baseline * 20.0 / k_iters;
    double eff_baseline = 12.0 / cyc_baseline * 100.0;
    double giops_baseline = (double)k_iters * 24 * 128 / ((double)ticks_baseline / 100000000.0) / 1e9;
    printf("Baseline:       %7.1f GIOPS, %.2f cyc/iter, %5.1f%% eff\n",
           giops_baseline, cyc_baseline, eff_baseline);

    // 6x4 Interleaved
    memset(C6, 0, MR6 * NR * 4);
    t0 = rdtsc();
    kernel_6x4_interleave_same(A6, B, C6, k_iters, NR * 4);
    t1 = rdtsc();
    uint64_t ticks_interleaved = t1 - t0;
    double cyc_interleaved = (double)ticks_interleaved * 20.0 / k_iters;
    double eff_interleaved = 12.0 / cyc_interleaved * 100.0;
    double giops_interleaved = (double)k_iters * 24 * 128 / ((double)ticks_interleaved / 100000000.0) / 1e9;
    printf("Interleaved:    %7.1f GIOPS, %.2f cyc/iter, %5.1f%% eff\n",
           giops_interleaved, cyc_interleaved, eff_interleaved);

    // 6x4 OoO (new)
    memset(C6, 0, MR6 * NR * 4);
    t0 = rdtsc();
    kernel_6x4_ooo_same(A6, B, C6, k_iters, NR * 4);
    t1 = rdtsc();
    uint64_t ticks_ooo = t1 - t0;
    double cyc_ooo = (double)ticks_ooo * 20.0 / k_iters;
    double eff_ooo = 12.0 / cyc_ooo * 100.0;
    double giops_ooo = (double)k_iters * 24 * 128 / ((double)ticks_ooo / 100000000.0) / 1e9;
    printf("OoO (new):      %7.1f GIOPS, %.2f cyc/iter, %5.1f%% eff\n",
           giops_ooo, cyc_ooo, eff_ooo);

    // ========================================
    // 5x4 double-buffer: 40 SDOT per iteration, ideal = 20 cycles
    // ========================================
    printf("\n--- 5x4 Double-Buffer (40 SDOT/iter, ideal 20 cyc/iter) ---\n");

    // Warmup
    kernel_5x4_dblbuf_same(A5, B, C5, 100000, NR * 4);

    // 5x4 Double-Buffer (new)
    memset(C5, 0, MR5 * NR * 4);
    t0 = rdtsc();
    kernel_5x4_dblbuf_same(A5, B, C5, k_iters, NR * 4);
    t1 = rdtsc();
    uint64_t ticks_dblbuf = t1 - t0;
    double cyc_dblbuf = (double)ticks_dblbuf * 20.0 / k_iters;
    double eff_dblbuf = 20.0 / cyc_dblbuf * 100.0;  // Ideal is 20 cycles for 40 SDOT
    double giops_dblbuf = (double)k_iters * 40 * 128 / ((double)ticks_dblbuf / 100000000.0) / 1e9;
    printf("DblBuf (new):   %7.1f GIOPS, %.2f cyc/iter, %5.1f%% eff\n",
           giops_dblbuf, cyc_dblbuf, eff_dblbuf);

    // ========================================
    // Summary - normalized to peak throughput
    // ========================================
    printf("\n=== Summary (peak = 512 GIOPS @ 2GHz with 2 FPUs) ===\n");
    printf("6x4 Baseline:    %5.1f%% of peak\n", giops_baseline / 512.0 * 100.0);
    printf("6x4 Interleaved: %5.1f%% of peak\n", giops_interleaved / 512.0 * 100.0);
    printf("6x4 OoO:         %5.1f%% of peak\n", giops_ooo / 512.0 * 100.0);
    printf("5x4 DblBuf:      %5.1f%% of peak\n", giops_dblbuf / 512.0 * 100.0);

    printf("\n=== Best 6x4 vs 5x4 ===\n");
    double best_6x4 = giops_baseline;
    const char* best_6x4_name = "Baseline";
    if (giops_interleaved > best_6x4) { best_6x4 = giops_interleaved; best_6x4_name = "Interleaved"; }
    if (giops_ooo > best_6x4) { best_6x4 = giops_ooo; best_6x4_name = "OoO"; }

    printf("Best 6x4 (%s): %.1f GIOPS\n", best_6x4_name, best_6x4);
    printf("5x4 DblBuf:          %.1f GIOPS\n", giops_dblbuf);
    printf("Difference: %.1f%% (%s)\n",
           (giops_dblbuf - best_6x4) / best_6x4 * 100.0,
           giops_dblbuf > best_6x4 ? "5x4 wins" : "6x4 wins");

    free(A6); free(A5); free(B); free(C6); free(C5);
    return 0;
}
