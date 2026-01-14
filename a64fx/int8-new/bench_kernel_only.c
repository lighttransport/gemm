// bench_kernel_only.c - Benchmark just the microkernel with pre-packed data
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include "gemm_pack_opt.h"

int main() {
    printf("=== Microkernel-Only Benchmark ===\n");
    printf("Testing raw kernel performance without packing overhead\n\n");

    // Single tile: 6x64 output from 6x256 A and 64x256 B
    size_t A_pack_size = 6 * 256;    // 1536 bytes
    size_t B_pack_size = 64 * 256;   // 16384 bytes

    // Allocate aligned buffers
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

    // Get frequency
    uint64_t freq;
    __asm__ volatile("mrs %0, cntfrq_el0" : "=r"(freq));
    printf("Timer frequency: %lu Hz\n", (unsigned long)freq);

    // Warmup
    kernel_6x4_ultra(Apack, Bpack, C, 64 * sizeof(int32_t));

    // Verify result (should be 256 for all ones)
    printf("Warmup result C[0]=%d (expected 256)\n", C[0]);

    // Benchmark with many iterations
    int iterations = 1000000;  // 1M iterations
    printf("Running %d kernel iterations...\n", iterations);

    uint64_t t0, t1;
    __asm__ volatile("mrs %0, cntvct_el0" : "=r"(t0));

    for (int i = 0; i < iterations; i++) {
        kernel_6x4_ultra(Apack, Bpack, C, 64 * sizeof(int32_t));
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

    printf("\nResults:\n");
    printf("  Total cycles: %lu\n", (unsigned long)cycles);
    printf("  Time: %.3f seconds\n", time_s);
    printf("  Cycles per kernel call: %.1f\n", cycles_per_call);
    printf("  Ops per kernel call: %.0f\n", ops_per_call);
    printf("  GOPS: %.2f\n", gops);
    printf("  Efficiency: %.1f%% of 512 GOPS peak\n", 100.0 * gops / 512.0);

    // Theoretical analysis
    printf("\nTheoretical analysis (A64FX at 2 GHz):\n");
    printf("  Minimum cycles (24 SDOT / 2 pipes): 12 cycles\n");
    printf("  Your cycles per call: %.1f cycles\n", cycles_per_call);
    printf("  Efficiency vs theoretical: %.1f%%\n", 100.0 * 12.0 / cycles_per_call);

    // Test pipe kernel too
    printf("\n--- Testing Pipe Kernel ---\n");
    memset(C, 0, 6 * 64 * sizeof(int32_t));
    kernel_6x4_pipe(Apack, Bpack, C, 64 * sizeof(int32_t));
    printf("Pipe result C[0]=%d (expected 256)\n", C[0]);

    __asm__ volatile("mrs %0, cntvct_el0" : "=r"(t0));

    for (int i = 0; i < iterations; i++) {
        kernel_6x4_pipe(Apack, Bpack, C, 64 * sizeof(int32_t));
    }

    __asm__ volatile("mrs %0, cntvct_el0" : "=r"(t1));

    cycles = t1 - t0;
    time_s = (double)cycles / (double)freq;
    cycles_per_call = (double)cycles / iterations;
    gops = total_ops / time_s / 1e9;

    printf("\nPipe kernel results:\n");
    printf("  Cycles per kernel call: %.1f\n", cycles_per_call);
    printf("  GOPS: %.2f\n", gops);
    printf("  Efficiency: %.1f%% of 512 GOPS peak\n", 100.0 * gops / 512.0);

    free(Apack);
    free(Bpack);
    free(C);

    return 0;
}
