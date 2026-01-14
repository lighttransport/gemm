// bench_pipe2.c - Benchmark pipelined unrolled kernels
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include "gemm_prepacked.h"

// Declare kernels
extern void kernel_6x4_unroll(const int8_t* Apack, const int8_t* Bpack,
                               int32_t* C, int ldc);
extern void kernel_6x4_pipe2(const int8_t* Apack, const int8_t* Bpack,
                              int32_t* C, int ldc);

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

int main() {
    printf("=== Pipelined Unrolled Kernel Benchmark ===\n");
    printf("Target: 95%% of 512 GOPS = 486 GOPS\n\n");

    volatile uint64_t freq = rdfreq();
    printf("Timer frequency: %lu Hz\n", (unsigned long)freq);

    // Single tile: 6x64 output from 6x256 A and 64x256 B
    size_t A_pack_size = 6 * 256;    // 1536 bytes
    size_t B_pack_size = 64 * 256;   // 16384 bytes

    int8_t* Apack = NULL;
    int8_t* Bpack = NULL;
    int32_t* C = NULL;
    int32_t* C_ref = NULL;

    posix_memalign((void**)&Apack, 256, A_pack_size);
    posix_memalign((void**)&Bpack, 256, B_pack_size);
    posix_memalign((void**)&C, 256, 6 * 64 * sizeof(int32_t));
    posix_memalign((void**)&C_ref, 256, 6 * 64 * sizeof(int32_t));

    // Initialize with test pattern
    for (size_t i = 0; i < A_pack_size; i++) Apack[i] = 1;
    for (size_t i = 0; i < B_pack_size; i++) Bpack[i] = 1;
    memset(C, 0, 6 * 64 * sizeof(int32_t));
    memset(C_ref, 0, 6 * 64 * sizeof(int32_t));

    // Warmup and verify correctness for unroll kernel
    kernel_6x4_unroll(Apack, Bpack, C_ref, 64 * sizeof(int32_t));
    printf("Unroll kernel C[0]=%d (expected 256)\n", C_ref[0]);

    // Warmup and verify correctness for pipe2 kernel
    kernel_6x4_pipe2(Apack, Bpack, C, 64 * sizeof(int32_t));
    printf("Pipe2 kernel C[0]=%d (expected 256)\n", C[0]);

    // Verify all elements match
    int errors = 0;
    for (int i = 0; i < 6 * 64; i++) {
        if (C[i] != C_ref[i]) {
            if (errors < 5) {
                printf("Mismatch at %d: pipe2=%d, unroll=%d\n", i, C[i], C_ref[i]);
            }
            errors++;
        }
    }
    if (errors > 0) {
        printf("ERROR: %d mismatches detected!\n", errors);
        return 1;
    }
    printf("Correctness: PASS\n\n");

    // Benchmark parameters
    int iterations = 1000000;  // 1M iterations
    double ops_per_call = 2.0 * 6 * 64 * 256;  // 196,608 ops per call
    double total_ops = ops_per_call * iterations;

    // Benchmark unroll kernel
    printf("=== Unrolled Kernel (baseline) ===\n");
    printf("Running %d iterations...\n", iterations);

    memset(C, 0, 6 * 64 * sizeof(int32_t));
    uint64_t t0 = rdtimer();
    for (int i = 0; i < iterations; i++) {
        kernel_6x4_unroll(Apack, Bpack, C, 64 * sizeof(int32_t));
    }
    uint64_t t1 = rdtimer();

    uint64_t cycles = t1 - t0;
    double time_s = (double)cycles / (double)freq;
    double cycles_per_call = (double)cycles / iterations;
    double gops = total_ops / time_s / 1e9;

    printf("  Total cycles: %lu\n", (unsigned long)cycles);
    printf("  Time: %.3f seconds\n", time_s);
    printf("  Cycles per call: %.1f (timer) / %.0f (CPU @ 2GHz)\n",
           cycles_per_call, cycles_per_call * 20);
    printf("  GOPS: %.2f\n", gops);
    printf("  Efficiency: %.1f%% of 512 GOPS peak\n\n", 100.0 * gops / 512.0);

    // Benchmark pipe2 kernel
    printf("=== Pipelined Unrolled Kernel (optimized) ===\n");
    printf("Running %d iterations...\n", iterations);

    memset(C, 0, 6 * 64 * sizeof(int32_t));
    t0 = rdtimer();
    for (int i = 0; i < iterations; i++) {
        kernel_6x4_pipe2(Apack, Bpack, C, 64 * sizeof(int32_t));
    }
    t1 = rdtimer();

    cycles = t1 - t0;
    time_s = (double)cycles / (double)freq;
    cycles_per_call = (double)cycles / iterations;
    gops = total_ops / time_s / 1e9;

    printf("  Total cycles: %lu\n", (unsigned long)cycles);
    printf("  Time: %.3f seconds\n", time_s);
    printf("  Cycles per call: %.1f (timer) / %.0f (CPU @ 2GHz)\n",
           cycles_per_call, cycles_per_call * 20);
    printf("  GOPS: %.2f\n", gops);
    printf("  Efficiency: %.1f%% of 512 GOPS peak\n\n", 100.0 * gops / 512.0);

    // Benchmark ultra kernel for comparison
    printf("=== Ultra Kernel (looped) ===\n");
    printf("Running %d iterations...\n", iterations);

    memset(C, 0, 6 * 64 * sizeof(int32_t));
    t0 = rdtimer();
    for (int i = 0; i < iterations; i++) {
        kernel_6x4_ultra(Apack, Bpack, C, 64 * sizeof(int32_t));
    }
    t1 = rdtimer();

    cycles = t1 - t0;
    time_s = (double)cycles / (double)freq;
    cycles_per_call = (double)cycles / iterations;
    gops = total_ops / time_s / 1e9;

    printf("  Total cycles: %lu\n", (unsigned long)cycles);
    printf("  Time: %.3f seconds\n", time_s);
    printf("  Cycles per call: %.1f (timer) / %.0f (CPU @ 2GHz)\n",
           cycles_per_call, cycles_per_call * 20);
    printf("  GOPS: %.2f\n", gops);
    printf("  Efficiency: %.1f%% of 512 GOPS peak\n\n", 100.0 * gops / 512.0);

    // Theoretical analysis
    printf("=== Theoretical Analysis ===\n");
    printf("  Operations per kernel: 6×64×256×2 = 196,608\n");
    printf("  SDOT ops: 24 per K-group × 64 groups = 1536 SDOTs\n");
    printf("  Minimum cycles at 2 SDOT/cycle: 768 CPU cycles\n");
    printf("  Target for 95%%: 768 / 0.95 = 808 CPU cycles\n");
    printf("  Target for 90%%: 768 / 0.90 = 853 CPU cycles\n");

    free(Apack);
    free(Bpack);
    free(C);
    free(C_ref);

    return 0;
}
