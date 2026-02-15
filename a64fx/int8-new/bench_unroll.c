// bench_unroll.c - Benchmark fully unrolled kernel
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include "gemm_prepacked.h"

// Declare the fully unrolled kernel
extern void kernel_6x4_unroll(const int8_t* Apack, const int8_t* Bpack,
                               int32_t* C, int ldc);

int main() {
    printf("=== Fully Unrolled Kernel Benchmark ===\n");
    printf("Target: 95%% of 512 GOPS = 486 GOPS\n\n");

    uint64_t freq;
    __asm__ volatile("mrs %0, cntfrq_el0" : "=r"(freq));
    printf("Timer frequency: %lu Hz\n", (unsigned long)freq);

    // Single tile: 6x64 output from 6x256 A and 64x256 B
    size_t A_pack_size = 6 * 256;    // 1536 bytes
    size_t B_pack_size = 64 * 256;   // 16384 bytes

    int8_t* Apack = NULL;
    int8_t* Bpack = NULL;
    int32_t* C = NULL;

    posix_memalign((void**)&Apack, 256, A_pack_size);
    posix_memalign((void**)&Bpack, 256, B_pack_size);
    posix_memalign((void**)&C, 256, 6 * 64 * sizeof(int32_t));

    // Initialize with test pattern
    for (size_t i = 0; i < A_pack_size; i++) Apack[i] = 1;
    for (size_t i = 0; i < B_pack_size; i++) Bpack[i] = 1;
    memset(C, 0, 6 * 64 * sizeof(int32_t));

    // Warmup and verify correctness
    kernel_6x4_unroll(Apack, Bpack, C, 64 * sizeof(int32_t));
    printf("Warmup result C[0]=%d (expected 256)\n", C[0]);

    if (C[0] != 256) {
        printf("ERROR: Incorrect result!\n");
        // Check reference kernel
        memset(C, 0, 6 * 64 * sizeof(int32_t));
        kernel_6x4_ultra(Apack, Bpack, C, 64 * sizeof(int32_t));
        printf("Ultra kernel C[0]=%d\n", C[0]);
        return 1;
    }

    // Benchmark with many iterations
    int iterations = 1000000;  // 1M iterations
    printf("\nRunning %d kernel iterations...\n", iterations);

    uint64_t t0, t1;
    __asm__ volatile("mrs %0, cntvct_el0" : "=r"(t0));

    for (int i = 0; i < iterations; i++) {
        kernel_6x4_unroll(Apack, Bpack, C, 64 * sizeof(int32_t));
    }

    __asm__ volatile("mrs %0, cntvct_el0" : "=r"(t1));

    // Compute stats
    uint64_t cycles = t1 - t0;
    double time_s = (double)cycles / (double)freq;
    double cycles_per_call = (double)cycles / iterations;

    // Each kernel call: 6 rows × 64 cols × 256 K × 2 ops = 196,608 ops
    double ops_per_call = 2.0 * 6 * 64 * 256;
    double total_ops = ops_per_call * iterations;
    double gops = total_ops / time_s / 1e9;

    printf("\n=== Fully Unrolled Kernel Results ===\n");
    printf("  Total cycles: %lu\n", (unsigned long)cycles);
    printf("  Time: %.3f seconds\n", time_s);
    printf("  Cycles per kernel call: %.1f (timer cycles)\n", cycles_per_call);
    printf("  CPU cycles per call: %.0f (at 2 GHz)\n", cycles_per_call * 20);
    printf("  GOPS: %.2f\n", gops);
    printf("  Efficiency: %.1f%% of 512 GOPS peak\n", 100.0 * gops / 512.0);

    // Compare to ultra kernel
    printf("\n=== Comparison with Ultra Kernel ===\n");
    memset(C, 0, 6 * 64 * sizeof(int32_t));

    __asm__ volatile("mrs %0, cntvct_el0" : "=r"(t0));

    for (int i = 0; i < iterations; i++) {
        kernel_6x4_ultra(Apack, Bpack, C, 64 * sizeof(int32_t));
    }

    __asm__ volatile("mrs %0, cntvct_el0" : "=r"(t1));

    cycles = t1 - t0;
    time_s = (double)cycles / (double)freq;
    cycles_per_call = (double)cycles / iterations;
    gops = total_ops / time_s / 1e9;

    printf("  Cycles per kernel call: %.1f (timer cycles)\n", cycles_per_call);
    printf("  CPU cycles per call: %.0f (at 2 GHz)\n", cycles_per_call * 20);
    printf("  GOPS: %.2f\n", gops);
    printf("  Efficiency: %.1f%% of 512 GOPS peak\n", 100.0 * gops / 512.0);

    // Theoretical analysis
    printf("\n=== Theoretical Analysis ===\n");
    printf("  Minimum cycles (24 SDOT × 64 groups / 2 pipes): 768 CPU cycles\n");
    printf("  Target for 95%%: 768 / 0.95 = 808 CPU cycles\n");

    free(Apack);
    free(Bpack);
    free(C);

    return 0;
}
