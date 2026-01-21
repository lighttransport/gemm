// bench_tile_groups.c
// Investigate how many tile groups needed to hide L2 latency
// and achieve peak SDOT throughput (2 SDOT/cycle)

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <arm_sve.h>

static inline uint64_t rdtsc(void) {
    uint64_t val;
    __asm__ volatile("mrs %0, cntvct_el0" : "=r"(val));
    return val;
}

// Analysis:
// - L2 latency: ~32 cycles
// - SDOT throughput: 2/cycle
// - To hide 32 cycles of L2 latency, need 64 independent SDOTs
// - Per tile (4 rows × 4 K-vecs): 16 SDOTs
// - Need ~4 tiles worth of compute to hide L2 latency
//
// Question: How many M-tile groups (each 4 rows) needed?

// Single tile group (MR=4, 16 accumulators)
// Compute per d_group: 16 SDOT = 8 cycles
void single_tile_group(
    const int8_t* Q,      // [4, D]
    const int8_t* K,      // [D, N] packed
    int32_t* S,           // [4, N]
    int D, int N
) {
    int d_groups = D / 4;
    int n_tiles = N / 16;  // 16 outputs per vector

    for (int nt = 0; nt < n_tiles; nt++) {
        __asm__ volatile(
            "ptrue p0.b\n\t"
            "ptrue p1.s\n\t"

            "mov x0, %[q]\n\t"
            "add x1, x0, %[d]\n\t"
            "add x2, x1, %[d]\n\t"
            "add x3, x2, %[d]\n\t"
            "mov x4, %[k]\n\t"

            // 16 accumulators
            "dup z16.s, #0\n\t"
            "dup z17.s, #0\n\t"
            "dup z18.s, #0\n\t"
            "dup z19.s, #0\n\t"
            "dup z20.s, #0\n\t"
            "dup z21.s, #0\n\t"
            "dup z22.s, #0\n\t"
            "dup z23.s, #0\n\t"
            "dup z24.s, #0\n\t"
            "dup z25.s, #0\n\t"
            "dup z26.s, #0\n\t"
            "dup z27.s, #0\n\t"
            "dup z28.s, #0\n\t"
            "dup z29.s, #0\n\t"
            "dup z30.s, #0\n\t"
            "dup z31.s, #0\n\t"

            "mov x5, %[dg]\n\t"
            "1:\n\t"
            "ld1b {z0.b}, p0/z, [x4]\n\t"
            "ld1b {z1.b}, p0/z, [x4, #1, mul vl]\n\t"
            "ld1b {z2.b}, p0/z, [x4, #2, mul vl]\n\t"
            "ld1b {z3.b}, p0/z, [x4, #3, mul vl]\n\t"
            "ld1rw {z4.s}, p1/z, [x0]\n\t"
            "ld1rw {z5.s}, p1/z, [x1]\n\t"
            "ld1rw {z6.s}, p1/z, [x2]\n\t"
            "ld1rw {z7.s}, p1/z, [x3]\n\t"

            "sdot z16.s, z0.b, z4.b\n\t"
            "sdot z20.s, z0.b, z5.b\n\t"
            "sdot z24.s, z0.b, z6.b\n\t"
            "sdot z28.s, z0.b, z7.b\n\t"

            "sdot z17.s, z1.b, z4.b\n\t"
            "sdot z21.s, z1.b, z5.b\n\t"
            "sdot z25.s, z1.b, z6.b\n\t"
            "sdot z29.s, z1.b, z7.b\n\t"

            "sdot z18.s, z2.b, z4.b\n\t"
            "sdot z22.s, z2.b, z5.b\n\t"
            "sdot z26.s, z2.b, z6.b\n\t"
            "sdot z30.s, z2.b, z7.b\n\t"

            "sdot z19.s, z3.b, z4.b\n\t"
            "sdot z23.s, z3.b, z5.b\n\t"
            "sdot z27.s, z3.b, z6.b\n\t"
            "sdot z31.s, z3.b, z7.b\n\t"

            "add x4, x4, #256\n\t"
            "add x0, x0, #4\n\t"
            "add x1, x1, #4\n\t"
            "add x2, x2, #4\n\t"
            "add x3, x3, #4\n\t"
            "subs x5, x5, #1\n\t"
            "b.ne 1b\n\t"

            // Merge and store
            "add z16.s, z16.s, z17.s\n\t"
            "add z18.s, z18.s, z19.s\n\t"
            "add z16.s, z16.s, z18.s\n\t"
            "add z20.s, z20.s, z21.s\n\t"
            "add z22.s, z22.s, z23.s\n\t"
            "add z20.s, z20.s, z22.s\n\t"
            "add z24.s, z24.s, z25.s\n\t"
            "add z26.s, z26.s, z27.s\n\t"
            "add z24.s, z24.s, z26.s\n\t"
            "add z28.s, z28.s, z29.s\n\t"
            "add z30.s, z30.s, z31.s\n\t"
            "add z28.s, z28.s, z30.s\n\t"

            "st1w {z16.s}, p1, [%[s]]\n\t"
            "st1w {z20.s}, p1, [%[s], #1, mul vl]\n\t"
            "st1w {z24.s}, p1, [%[s], #2, mul vl]\n\t"
            "st1w {z28.s}, p1, [%[s], #3, mul vl]\n\t"
            :
            : [q] "r"(Q), [k] "r"(K + nt * 16 * D), [s] "r"(S + nt * 16),
              [d] "r"((uint64_t)D), [dg] "r"((uint64_t)d_groups)
            : "x0", "x1", "x2", "x3", "x4", "x5",
              "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7",
              "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23",
              "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31",
              "p0", "p1", "memory"
        );
    }
}

// Two interleaved tile groups (MR=4 × 2 = 8 rows, 32 accumulators total)
// But we only have 32 Z registers, so need to be clever
// Use 8 registers for K data, 8 for Q broadcasts, 16 for accumulators
// Process 2 M-tiles but share K loads
void two_tile_groups(
    const int8_t* Q,      // [8, D] - two groups of 4 rows
    const int8_t* K,      // [D, N] packed
    int32_t* S,           // [8, N]
    int D, int N
) {
    // For simplicity, process each group but interleave at d_group level
    // This simulates having 2 tiles in flight

    int d_groups = D / 4;

    __asm__ volatile(
        "ptrue p0.b\n\t"
        "ptrue p1.s\n\t"

        // Q pointers for group 0 (rows 0-3)
        "mov x0, %[q]\n\t"
        "add x1, x0, %[d]\n\t"
        "add x2, x1, %[d]\n\t"
        "add x3, x2, %[d]\n\t"
        // Q pointers for group 1 (rows 4-7)
        "add x6, x3, %[d]\n\t"
        "add x7, x6, %[d]\n\t"
        "add x8, x7, %[d]\n\t"
        "add x9, x8, %[d]\n\t"

        "mov x4, %[k]\n\t"

        // 8 accumulators for group 0 (rows 0-3, K[0-1])
        "dup z16.s, #0\n\t"
        "dup z17.s, #0\n\t"
        "dup z18.s, #0\n\t"
        "dup z19.s, #0\n\t"
        // 8 accumulators for group 1 (rows 4-7, K[0-1])
        "dup z20.s, #0\n\t"
        "dup z21.s, #0\n\t"
        "dup z22.s, #0\n\t"
        "dup z23.s, #0\n\t"
        // 8 more for K[2-3] of both groups (reuse after accumulation)
        "dup z24.s, #0\n\t"
        "dup z25.s, #0\n\t"
        "dup z26.s, #0\n\t"
        "dup z27.s, #0\n\t"
        "dup z28.s, #0\n\t"
        "dup z29.s, #0\n\t"
        "dup z30.s, #0\n\t"
        "dup z31.s, #0\n\t"

        "mov x5, %[dg]\n\t"
        "1:\n\t"
        // Load K (shared between both groups)
        "ld1b {z0.b}, p0/z, [x4]\n\t"
        "ld1b {z1.b}, p0/z, [x4, #1, mul vl]\n\t"
        "ld1b {z2.b}, p0/z, [x4, #2, mul vl]\n\t"
        "ld1b {z3.b}, p0/z, [x4, #3, mul vl]\n\t"

        // Load Q for group 0
        "ld1rw {z4.s}, p1/z, [x0]\n\t"
        "ld1rw {z5.s}, p1/z, [x1]\n\t"
        "ld1rw {z6.s}, p1/z, [x2]\n\t"
        "ld1rw {z7.s}, p1/z, [x3]\n\t"

        // SDOT group 0 with K[0]
        "sdot z16.s, z0.b, z4.b\n\t"
        "sdot z17.s, z0.b, z5.b\n\t"
        "sdot z18.s, z0.b, z6.b\n\t"
        "sdot z19.s, z0.b, z7.b\n\t"

        // Load Q for group 1
        "ld1rw {z8.s}, p1/z, [x6]\n\t"
        "ld1rw {z9.s}, p1/z, [x7]\n\t"
        "ld1rw {z10.s}, p1/z, [x8]\n\t"
        "ld1rw {z11.s}, p1/z, [x9]\n\t"

        // SDOT group 1 with K[0]
        "sdot z20.s, z0.b, z8.b\n\t"
        "sdot z21.s, z0.b, z9.b\n\t"
        "sdot z22.s, z0.b, z10.b\n\t"
        "sdot z23.s, z0.b, z11.b\n\t"

        // SDOT group 0 with K[1]
        "sdot z24.s, z1.b, z4.b\n\t"
        "sdot z25.s, z1.b, z5.b\n\t"
        "sdot z26.s, z1.b, z6.b\n\t"
        "sdot z27.s, z1.b, z7.b\n\t"

        // SDOT group 1 with K[1]
        "sdot z28.s, z1.b, z8.b\n\t"
        "sdot z29.s, z1.b, z9.b\n\t"
        "sdot z30.s, z1.b, z10.b\n\t"
        "sdot z31.s, z1.b, z11.b\n\t"

        // SDOT group 0 with K[2] (accumulate to existing)
        "sdot z16.s, z2.b, z4.b\n\t"
        "sdot z17.s, z2.b, z5.b\n\t"
        "sdot z18.s, z2.b, z6.b\n\t"
        "sdot z19.s, z2.b, z7.b\n\t"

        // SDOT group 1 with K[2]
        "sdot z20.s, z2.b, z8.b\n\t"
        "sdot z21.s, z2.b, z9.b\n\t"
        "sdot z22.s, z2.b, z10.b\n\t"
        "sdot z23.s, z2.b, z11.b\n\t"

        // SDOT group 0 with K[3]
        "sdot z24.s, z3.b, z4.b\n\t"
        "sdot z25.s, z3.b, z5.b\n\t"
        "sdot z26.s, z3.b, z6.b\n\t"
        "sdot z27.s, z3.b, z7.b\n\t"

        // SDOT group 1 with K[3]
        "sdot z28.s, z3.b, z8.b\n\t"
        "sdot z29.s, z3.b, z9.b\n\t"
        "sdot z30.s, z3.b, z10.b\n\t"
        "sdot z31.s, z3.b, z11.b\n\t"

        "add x4, x4, #256\n\t"
        "add x0, x0, #4\n\t"
        "add x1, x1, #4\n\t"
        "add x2, x2, #4\n\t"
        "add x3, x3, #4\n\t"
        "add x6, x6, #4\n\t"
        "add x7, x7, #4\n\t"
        "add x8, x8, #4\n\t"
        "add x9, x9, #4\n\t"
        "subs x5, x5, #1\n\t"
        "b.ne 1b\n\t"

        // Merge accumulators
        "add z16.s, z16.s, z24.s\n\t"
        "add z17.s, z17.s, z25.s\n\t"
        "add z18.s, z18.s, z26.s\n\t"
        "add z19.s, z19.s, z27.s\n\t"
        "add z20.s, z20.s, z28.s\n\t"
        "add z21.s, z21.s, z29.s\n\t"
        "add z22.s, z22.s, z30.s\n\t"
        "add z23.s, z23.s, z31.s\n\t"

        // Store
        "st1w {z16.s}, p1, [%[s]]\n\t"
        "st1w {z17.s}, p1, [%[s], #1, mul vl]\n\t"
        "st1w {z18.s}, p1, [%[s], #2, mul vl]\n\t"
        "st1w {z19.s}, p1, [%[s], #3, mul vl]\n\t"
        "st1w {z20.s}, p1, [%[s], #4, mul vl]\n\t"
        "st1w {z21.s}, p1, [%[s], #5, mul vl]\n\t"
        "st1w {z22.s}, p1, [%[s], #6, mul vl]\n\t"
        "st1w {z23.s}, p1, [%[s], #7, mul vl]\n\t"
        :
        : [q] "r"(Q), [k] "r"(K), [s] "r"(S),
          [d] "r"((uint64_t)D), [dg] "r"((uint64_t)(D/4))
        : "x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9",
          "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7",
          "z8", "z9", "z10", "z11",
          "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23",
          "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31",
          "p0", "p1", "memory"
    );
}

// Measure pure SDOT throughput (no memory, just compute)
void pure_sdot_throughput(int num_sdot) {
    __asm__ volatile(
        "ptrue p0.b\n\t"
        "dup z0.b, #1\n\t"
        "dup z1.b, #2\n\t"
        "dup z2.b, #3\n\t"
        "dup z3.b, #4\n\t"
        "dup z4.b, #5\n\t"
        "dup z5.b, #6\n\t"
        "dup z6.b, #7\n\t"
        "dup z7.b, #8\n\t"

        // 16 independent accumulators
        "dup z16.s, #0\n\t"
        "dup z17.s, #0\n\t"
        "dup z18.s, #0\n\t"
        "dup z19.s, #0\n\t"
        "dup z20.s, #0\n\t"
        "dup z21.s, #0\n\t"
        "dup z22.s, #0\n\t"
        "dup z23.s, #0\n\t"
        "dup z24.s, #0\n\t"
        "dup z25.s, #0\n\t"
        "dup z26.s, #0\n\t"
        "dup z27.s, #0\n\t"
        "dup z28.s, #0\n\t"
        "dup z29.s, #0\n\t"
        "dup z30.s, #0\n\t"
        "dup z31.s, #0\n\t"

        "mov x0, %[n]\n\t"
        "1:\n\t"
        // 16 SDOTs per iteration, all independent
        "sdot z16.s, z0.b, z4.b\n\t"
        "sdot z17.s, z1.b, z5.b\n\t"
        "sdot z18.s, z2.b, z6.b\n\t"
        "sdot z19.s, z3.b, z7.b\n\t"
        "sdot z20.s, z0.b, z5.b\n\t"
        "sdot z21.s, z1.b, z6.b\n\t"
        "sdot z22.s, z2.b, z7.b\n\t"
        "sdot z23.s, z3.b, z4.b\n\t"
        "sdot z24.s, z0.b, z6.b\n\t"
        "sdot z25.s, z1.b, z7.b\n\t"
        "sdot z26.s, z2.b, z4.b\n\t"
        "sdot z27.s, z3.b, z5.b\n\t"
        "sdot z28.s, z0.b, z7.b\n\t"
        "sdot z29.s, z1.b, z4.b\n\t"
        "sdot z30.s, z2.b, z5.b\n\t"
        "sdot z31.s, z3.b, z6.b\n\t"
        "subs x0, x0, #16\n\t"
        "b.gt 1b\n\t"
        :
        : [n] "r"((uint64_t)num_sdot)
        : "x0",
          "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7",
          "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23",
          "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31",
          "p0", "memory"
    );
}

int main() {
    printf("==============================================\n");
    printf("Tile Groups Analysis for L2 Latency Hiding\n");
    printf("==============================================\n\n");

    printf("A64FX Parameters:\n");
    printf("  L2 latency: ~32 cycles\n");
    printf("  SDOT throughput: 2/cycle (2 FPUs)\n");
    printf("  SDOT latency: 9 cycles\n\n");

    printf("To hide L2 latency with compute:\n");
    printf("  Need: 32 cycles × 2 SDOT/cy = 64 independent SDOTs\n");
    printf("  Per 4-row tile: 16 SDOTs/d_group\n");
    printf("  Need: ~4 d_groups OR multiple tiles in flight\n\n");

    int iters = 10000;
    int warmup = 1000;
    uint64_t start, end;
    double ticks, cycles;

    // First, measure pure SDOT throughput
    printf("=== Pure SDOT Throughput (no memory) ===\n");
    int num_sdot = 1024;
    for (int i = 0; i < warmup; i++) pure_sdot_throughput(num_sdot);
    start = rdtsc();
    for (int i = 0; i < iters; i++) pure_sdot_throughput(num_sdot);
    end = rdtsc();
    ticks = (double)(end - start) / iters;
    cycles = ticks * 20.0;
    printf("  %d SDOTs: %.1f cycles, %.2f SDOT/cycle\n\n",
           num_sdot, cycles, num_sdot / cycles);

    // Test with different D sizes (affects working set)
    printf("=== Single Tile Group (MR=4) ===\n");
    printf("%-8s %-10s %-10s %-10s %-10s\n",
           "D", "Cycles", "SDOT/cy", "WS(KB)", "From");
    printf("------------------------------------------------------\n");

    int D_sizes[] = {64, 128, 256, 512, 1024};
    int N = 64;  // Fixed N for comparison

    for (int di = 0; di < 5; di++) {
        int D = D_sizes[di];
        int d_groups = D / 4;
        int total_sdot = d_groups * 16;  // 16 SDOT per d_group

        // Working set: Q[4,D] + K[D,64] + S[4,64]
        size_t ws_bytes = 4 * D + D * 64 + 4 * 64 * 4;
        double ws_kb = ws_bytes / 1024.0;
        const char* from = ws_kb < 64 ? "L1" : (ws_kb < 8192 ? "L2" : "HBM");

        int8_t* Q = (int8_t*)aligned_alloc(256, 4 * D);
        int8_t* K = (int8_t*)aligned_alloc(256, D * N);
        int32_t* S = (int32_t*)aligned_alloc(256, 4 * N * 4);

        memset(Q, 1, 4 * D);
        memset(K, 2, D * N);
        memset(S, 0, 4 * N * 4);

        for (int i = 0; i < warmup; i++) single_tile_group(Q, K, S, D, N);
        start = rdtsc();
        for (int i = 0; i < iters; i++) single_tile_group(Q, K, S, D, N);
        end = rdtsc();
        ticks = (double)(end - start) / iters;
        cycles = ticks * 20.0;

        printf("%-8d %-10.1f %-10.2f %-10.1f %-10s\n",
               D, cycles, total_sdot / cycles, ws_kb, from);

        free(Q); free(K); free(S);
    }

    printf("\n=== Two Tile Groups (MR=8, shared K loads) ===\n");
    printf("%-8s %-10s %-10s %-10s %-10s\n",
           "D", "Cycles", "SDOT/cy", "WS(KB)", "From");
    printf("------------------------------------------------------\n");

    for (int di = 0; di < 5; di++) {
        int D = D_sizes[di];
        int d_groups = D / 4;
        int total_sdot = d_groups * 32;  // 32 SDOT per d_group (2 groups)

        // Working set: Q[8,D] + K[D,64] + S[8,64]
        size_t ws_bytes = 8 * D + D * 64 + 8 * 64 * 4;
        double ws_kb = ws_bytes / 1024.0;
        const char* from = ws_kb < 64 ? "L1" : (ws_kb < 8192 ? "L2" : "HBM");

        int8_t* Q = (int8_t*)aligned_alloc(256, 8 * D);
        int8_t* K = (int8_t*)aligned_alloc(256, D * N);
        int32_t* S = (int32_t*)aligned_alloc(256, 8 * N * 4);

        memset(Q, 1, 8 * D);
        memset(K, 2, D * N);
        memset(S, 0, 8 * N * 4);

        for (int i = 0; i < warmup; i++) two_tile_groups(Q, K, S, D, N);
        start = rdtsc();
        for (int i = 0; i < iters; i++) two_tile_groups(Q, K, S, D, N);
        end = rdtsc();
        ticks = (double)(end - start) / iters;
        cycles = ticks * 20.0;

        printf("%-8d %-10.1f %-10.2f %-10.1f %-10s\n",
               D, cycles, total_sdot / cycles, ws_kb, from);

        free(Q); free(K); free(S);
    }

    printf("\n=== Analysis ===\n");
    printf("L2 latency hiding requires:\n");
    printf("  - 32 cy latency / 8 cy per d_group = 4 d_groups prefetched\n");
    printf("  - OR: 2+ tile groups to double compute per K load\n");
    printf("  - OR: Software pipelining with deep unroll\n");

    return 0;
}
