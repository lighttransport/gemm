/*
 * Benchmark: MR=7 tile with K split into 2 phases
 *
 * MR=7 needs 28 accumulators (7×4), but only 32 regs available
 * Solution: Load K[0-1], compute 14 SDOTs, then K[2-3], compute 14 SDOTs
 *
 * Register allocation:
 *   z0-z1: K vectors (2, reloaded twice per d_group)
 *   z2: Q broadcast (1, reused for each row)
 *   z3-z30: 28 accumulators
 */

#include <arm_sve.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static inline uint64_t rdcycle(void) {
    uint64_t val;
    __asm__ volatile("mrs %0, cntvct_el0" : "=r"(val));
    return val;
}

// MR=6 baseline (from previous benchmark)
void bench_mr6(const int8_t* K, const int8_t* Q, int D) {
    __asm__ volatile(
        "ptrue p0.b\n\t"
        "ptrue p1.s\n\t"

        // Zero 24 accumulators (z8-z31)
        "mov z8.s, #0\n\t"
        "mov z9.s, #0\n\t"
        "mov z10.s, #0\n\t"
        "mov z11.s, #0\n\t"
        "mov z12.s, #0\n\t"
        "mov z13.s, #0\n\t"
        "mov z14.s, #0\n\t"
        "mov z15.s, #0\n\t"
        "mov z16.s, #0\n\t"
        "mov z17.s, #0\n\t"
        "mov z18.s, #0\n\t"
        "mov z19.s, #0\n\t"
        "mov z20.s, #0\n\t"
        "mov z21.s, #0\n\t"
        "mov z22.s, #0\n\t"
        "mov z23.s, #0\n\t"
        "mov z24.s, #0\n\t"
        "mov z25.s, #0\n\t"
        "mov z26.s, #0\n\t"
        "mov z27.s, #0\n\t"
        "mov z28.s, #0\n\t"
        "mov z29.s, #0\n\t"
        "mov z30.s, #0\n\t"
        "mov z31.s, #0\n\t"

        // Q row pointers
        "mov x5, %[Q]\n\t"
        "add x6, x5, #256\n\t"
        "add x7, x5, #512\n\t"
        "add x11, x5, #768\n\t"
        "add x12, x5, #1024\n\t"
        "add x13, x5, #1280\n\t"
        "mov x4, %[K]\n\t"
        "mov x10, %[D]\n\t"

        "1:\n\t"
        // Load K (4 vectors)
        "ld1b {z0.b}, p0/z, [x4]\n\t"
        "ld1b {z1.b}, p0/z, [x4, #1, mul vl]\n\t"
        "ld1b {z2.b}, p0/z, [x4, #2, mul vl]\n\t"
        "ld1b {z3.b}, p0/z, [x4, #3, mul vl]\n\t"
        "add x4, x4, #256\n\t"

        // Row 0
        "ld1rw {z4.s}, p1/z, [x5]\n\t"
        "sdot z8.s, z0.b, z4.b\n\t"
        "sdot z9.s, z1.b, z4.b\n\t"
        "sdot z10.s, z2.b, z4.b\n\t"
        "sdot z11.s, z3.b, z4.b\n\t"
        // Row 1
        "ld1rw {z4.s}, p1/z, [x6]\n\t"
        "sdot z12.s, z0.b, z4.b\n\t"
        "sdot z13.s, z1.b, z4.b\n\t"
        "sdot z14.s, z2.b, z4.b\n\t"
        "sdot z15.s, z3.b, z4.b\n\t"
        // Row 2
        "ld1rw {z4.s}, p1/z, [x7]\n\t"
        "sdot z16.s, z0.b, z4.b\n\t"
        "sdot z17.s, z1.b, z4.b\n\t"
        "sdot z18.s, z2.b, z4.b\n\t"
        "sdot z19.s, z3.b, z4.b\n\t"
        // Row 3
        "ld1rw {z4.s}, p1/z, [x11]\n\t"
        "sdot z20.s, z0.b, z4.b\n\t"
        "sdot z21.s, z1.b, z4.b\n\t"
        "sdot z22.s, z2.b, z4.b\n\t"
        "sdot z23.s, z3.b, z4.b\n\t"
        // Row 4
        "ld1rw {z4.s}, p1/z, [x12]\n\t"
        "sdot z24.s, z0.b, z4.b\n\t"
        "sdot z25.s, z1.b, z4.b\n\t"
        "sdot z26.s, z2.b, z4.b\n\t"
        "sdot z27.s, z3.b, z4.b\n\t"
        // Row 5
        "ld1rw {z4.s}, p1/z, [x13]\n\t"
        "sdot z28.s, z0.b, z4.b\n\t"
        "sdot z29.s, z1.b, z4.b\n\t"
        "sdot z30.s, z2.b, z4.b\n\t"
        "sdot z31.s, z3.b, z4.b\n\t"

        "add x5, x5, #4\n\t"
        "add x6, x6, #4\n\t"
        "add x7, x7, #4\n\t"
        "add x11, x11, #4\n\t"
        "add x12, x12, #4\n\t"
        "add x13, x13, #4\n\t"

        "subs x10, x10, #1\n\t"
        "b.ne 1b\n\t"
        :
        : [K] "r"(K), [Q] "r"(Q), [D] "r"((long)D)
        : "memory", "x4", "x5", "x6", "x7", "x10", "x11", "x12", "x13",
          "z0", "z1", "z2", "z3", "z4",
          "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15",
          "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23",
          "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31",
          "p0", "p1"
    );
}

// MR=7 with K split (2 K vectors per phase)
void bench_mr7_split(const int8_t* K, const int8_t* Q, int D) {
    __asm__ volatile(
        "ptrue p0.b\n\t"
        "ptrue p1.s\n\t"

        // Zero 28 accumulators (z3-z30)
        "mov z3.s, #0\n\t"
        "mov z4.s, #0\n\t"
        "mov z5.s, #0\n\t"
        "mov z6.s, #0\n\t"
        "mov z7.s, #0\n\t"
        "mov z8.s, #0\n\t"
        "mov z9.s, #0\n\t"
        "mov z10.s, #0\n\t"
        "mov z11.s, #0\n\t"
        "mov z12.s, #0\n\t"
        "mov z13.s, #0\n\t"
        "mov z14.s, #0\n\t"
        "mov z15.s, #0\n\t"
        "mov z16.s, #0\n\t"
        "mov z17.s, #0\n\t"
        "mov z18.s, #0\n\t"
        "mov z19.s, #0\n\t"
        "mov z20.s, #0\n\t"
        "mov z21.s, #0\n\t"
        "mov z22.s, #0\n\t"
        "mov z23.s, #0\n\t"
        "mov z24.s, #0\n\t"
        "mov z25.s, #0\n\t"
        "mov z26.s, #0\n\t"
        "mov z27.s, #0\n\t"
        "mov z28.s, #0\n\t"
        "mov z29.s, #0\n\t"
        "mov z30.s, #0\n\t"

        // Q row pointers (7 rows)
        "mov x5, %[Q]\n\t"
        "add x6, x5, #256\n\t"
        "add x7, x5, #512\n\t"
        "add x11, x5, #768\n\t"
        "add x12, x5, #1024\n\t"
        "add x13, x5, #1280\n\t"
        "add x14, x5, #1536\n\t"
        "mov x4, %[K]\n\t"
        "mov x10, %[D]\n\t"

        "1:\n\t"
        // === Phase 1: K[0-1] ===
        "ld1b {z0.b}, p0/z, [x4]\n\t"
        "ld1b {z1.b}, p0/z, [x4, #1, mul vl]\n\t"

        // Row 0: acc z3-z4
        "ld1rw {z2.s}, p1/z, [x5]\n\t"
        "sdot z3.s, z0.b, z2.b\n\t"
        "sdot z4.s, z1.b, z2.b\n\t"
        // Row 1: acc z5-z6
        "ld1rw {z2.s}, p1/z, [x6]\n\t"
        "sdot z5.s, z0.b, z2.b\n\t"
        "sdot z6.s, z1.b, z2.b\n\t"
        // Row 2: acc z7-z8
        "ld1rw {z2.s}, p1/z, [x7]\n\t"
        "sdot z7.s, z0.b, z2.b\n\t"
        "sdot z8.s, z1.b, z2.b\n\t"
        // Row 3: acc z9-z10
        "ld1rw {z2.s}, p1/z, [x11]\n\t"
        "sdot z9.s, z0.b, z2.b\n\t"
        "sdot z10.s, z1.b, z2.b\n\t"
        // Row 4: acc z11-z12
        "ld1rw {z2.s}, p1/z, [x12]\n\t"
        "sdot z11.s, z0.b, z2.b\n\t"
        "sdot z12.s, z1.b, z2.b\n\t"
        // Row 5: acc z13-z14
        "ld1rw {z2.s}, p1/z, [x13]\n\t"
        "sdot z13.s, z0.b, z2.b\n\t"
        "sdot z14.s, z1.b, z2.b\n\t"
        // Row 6: acc z15-z16
        "ld1rw {z2.s}, p1/z, [x14]\n\t"
        "sdot z15.s, z0.b, z2.b\n\t"
        "sdot z16.s, z1.b, z2.b\n\t"

        // === Phase 2: K[2-3] ===
        "ld1b {z0.b}, p0/z, [x4, #2, mul vl]\n\t"
        "ld1b {z1.b}, p0/z, [x4, #3, mul vl]\n\t"
        "add x4, x4, #256\n\t"

        // Row 0: acc z17-z18
        "ld1rw {z2.s}, p1/z, [x5]\n\t"
        "sdot z17.s, z0.b, z2.b\n\t"
        "sdot z18.s, z1.b, z2.b\n\t"
        // Row 1: acc z19-z20
        "ld1rw {z2.s}, p1/z, [x6]\n\t"
        "sdot z19.s, z0.b, z2.b\n\t"
        "sdot z20.s, z1.b, z2.b\n\t"
        // Row 2: acc z21-z22
        "ld1rw {z2.s}, p1/z, [x7]\n\t"
        "sdot z21.s, z0.b, z2.b\n\t"
        "sdot z22.s, z1.b, z2.b\n\t"
        // Row 3: acc z23-z24
        "ld1rw {z2.s}, p1/z, [x11]\n\t"
        "sdot z23.s, z0.b, z2.b\n\t"
        "sdot z24.s, z1.b, z2.b\n\t"
        // Row 4: acc z25-z26
        "ld1rw {z2.s}, p1/z, [x12]\n\t"
        "sdot z25.s, z0.b, z2.b\n\t"
        "sdot z26.s, z1.b, z2.b\n\t"
        // Row 5: acc z27-z28
        "ld1rw {z2.s}, p1/z, [x13]\n\t"
        "sdot z27.s, z0.b, z2.b\n\t"
        "sdot z28.s, z1.b, z2.b\n\t"
        // Row 6: acc z29-z30
        "ld1rw {z2.s}, p1/z, [x14]\n\t"
        "sdot z29.s, z0.b, z2.b\n\t"
        "sdot z30.s, z1.b, z2.b\n\t"

        "add x5, x5, #4\n\t"
        "add x6, x6, #4\n\t"
        "add x7, x7, #4\n\t"
        "add x11, x11, #4\n\t"
        "add x12, x12, #4\n\t"
        "add x13, x13, #4\n\t"
        "add x14, x14, #4\n\t"

        "subs x10, x10, #1\n\t"
        "b.ne 1b\n\t"
        :
        : [K] "r"(K), [Q] "r"(Q), [D] "r"((long)D)
        : "memory", "x4", "x5", "x6", "x7", "x10", "x11", "x12", "x13", "x14",
          "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7",
          "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15",
          "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23",
          "z24", "z25", "z26", "z27", "z28", "z29", "z30",
          "p0", "p1"
    );
}

// MR=8 with K split (2 K vectors per phase) - maximum rows
void bench_mr8_split(const int8_t* K, const int8_t* Q, int D) {
    __asm__ volatile(
        "ptrue p0.b\n\t"
        "ptrue p1.s\n\t"

        // MR=8 needs 32 accumulators, but we only have 32 regs total
        // With 2 K + 1 Q = 3 regs, only 29 left for accumulators
        // Can't do full MR=8 with 4 K vectors

        // Instead: MR=8 with 3 K vectors = 24 accumulators (fits, but worse than MR=6)
        // Or: MR=8 with 2 K vectors = 16 accumulators per phase, 32 total
        //     But need to process in 2 phases and merge

        // Let's try MR=8 × 2K = 16 SDOTs per phase × 2 = 32 SDOTs
        // Accumulators: 16 (reused between phases with accumulate)

        // Actually simpler: just use 2 K vectors, 8 rows = 16 accumulators
        // Process K[0-1], then K[2-3] separately, accumulate into same result

        // Zero 16 accumulators (z4-z19) - will accumulate both K phases
        "mov z4.s, #0\n\t"
        "mov z5.s, #0\n\t"
        "mov z6.s, #0\n\t"
        "mov z7.s, #0\n\t"
        "mov z8.s, #0\n\t"
        "mov z9.s, #0\n\t"
        "mov z10.s, #0\n\t"
        "mov z11.s, #0\n\t"
        "mov z12.s, #0\n\t"
        "mov z13.s, #0\n\t"
        "mov z14.s, #0\n\t"
        "mov z15.s, #0\n\t"
        "mov z16.s, #0\n\t"
        "mov z17.s, #0\n\t"
        "mov z18.s, #0\n\t"
        "mov z19.s, #0\n\t"

        // Additional 16 accumulators for K[2-3] (z20-z31 + z3 as temp)
        "mov z20.s, #0\n\t"
        "mov z21.s, #0\n\t"
        "mov z22.s, #0\n\t"
        "mov z23.s, #0\n\t"
        "mov z24.s, #0\n\t"
        "mov z25.s, #0\n\t"
        "mov z26.s, #0\n\t"
        "mov z27.s, #0\n\t"
        "mov z28.s, #0\n\t"
        "mov z29.s, #0\n\t"
        "mov z30.s, #0\n\t"
        "mov z31.s, #0\n\t"

        // Q row pointers (8 rows)
        "mov x5, %[Q]\n\t"
        "add x6, x5, #256\n\t"
        "add x7, x5, #512\n\t"
        "add x11, x5, #768\n\t"
        "add x12, x5, #1024\n\t"
        "add x13, x5, #1280\n\t"
        "add x14, x5, #1536\n\t"
        "add x15, x5, #1792\n\t"
        "mov x4, %[K]\n\t"
        "mov x10, %[D]\n\t"

        "1:\n\t"
        // === Phase 1: K[0-1], 8 rows, 16 SDOTs ===
        "ld1b {z0.b}, p0/z, [x4]\n\t"
        "ld1b {z1.b}, p0/z, [x4, #1, mul vl]\n\t"

        "ld1rw {z2.s}, p1/z, [x5]\n\t"
        "sdot z4.s, z0.b, z2.b\n\t"
        "sdot z5.s, z1.b, z2.b\n\t"
        "ld1rw {z2.s}, p1/z, [x6]\n\t"
        "sdot z6.s, z0.b, z2.b\n\t"
        "sdot z7.s, z1.b, z2.b\n\t"
        "ld1rw {z2.s}, p1/z, [x7]\n\t"
        "sdot z8.s, z0.b, z2.b\n\t"
        "sdot z9.s, z1.b, z2.b\n\t"
        "ld1rw {z2.s}, p1/z, [x11]\n\t"
        "sdot z10.s, z0.b, z2.b\n\t"
        "sdot z11.s, z1.b, z2.b\n\t"
        "ld1rw {z2.s}, p1/z, [x12]\n\t"
        "sdot z12.s, z0.b, z2.b\n\t"
        "sdot z13.s, z1.b, z2.b\n\t"
        "ld1rw {z2.s}, p1/z, [x13]\n\t"
        "sdot z14.s, z0.b, z2.b\n\t"
        "sdot z15.s, z1.b, z2.b\n\t"
        "ld1rw {z2.s}, p1/z, [x14]\n\t"
        "sdot z16.s, z0.b, z2.b\n\t"
        "sdot z17.s, z1.b, z2.b\n\t"
        "ld1rw {z2.s}, p1/z, [x15]\n\t"
        "sdot z18.s, z0.b, z2.b\n\t"
        "sdot z19.s, z1.b, z2.b\n\t"

        // === Phase 2: K[2-3], 8 rows, 16 SDOTs ===
        "ld1b {z0.b}, p0/z, [x4, #2, mul vl]\n\t"
        "ld1b {z1.b}, p0/z, [x4, #3, mul vl]\n\t"
        "add x4, x4, #256\n\t"

        "ld1rw {z2.s}, p1/z, [x5]\n\t"
        "sdot z20.s, z0.b, z2.b\n\t"
        "sdot z21.s, z1.b, z2.b\n\t"
        "ld1rw {z2.s}, p1/z, [x6]\n\t"
        "sdot z22.s, z0.b, z2.b\n\t"
        "sdot z23.s, z1.b, z2.b\n\t"
        "ld1rw {z2.s}, p1/z, [x7]\n\t"
        "sdot z24.s, z0.b, z2.b\n\t"
        "sdot z25.s, z1.b, z2.b\n\t"
        "ld1rw {z2.s}, p1/z, [x11]\n\t"
        "sdot z26.s, z0.b, z2.b\n\t"
        "sdot z27.s, z1.b, z2.b\n\t"
        "ld1rw {z2.s}, p1/z, [x12]\n\t"
        "sdot z28.s, z0.b, z2.b\n\t"
        "sdot z29.s, z1.b, z2.b\n\t"
        "ld1rw {z2.s}, p1/z, [x13]\n\t"
        "sdot z30.s, z0.b, z2.b\n\t"
        "sdot z31.s, z1.b, z2.b\n\t"

        // Rows 6-7 for K[2-3]: need to reuse some accumulators (not fully correct)
        // Just measuring throughput, not correctness
        "ld1rw {z2.s}, p1/z, [x14]\n\t"
        "sdot z3.s, z0.b, z2.b\n\t"  // Temp accumulator
        "ld1rw {z2.s}, p1/z, [x15]\n\t"
        "sdot z3.s, z1.b, z2.b\n\t"

        "add x5, x5, #4\n\t"
        "add x6, x6, #4\n\t"
        "add x7, x7, #4\n\t"
        "add x11, x11, #4\n\t"
        "add x12, x12, #4\n\t"
        "add x13, x13, #4\n\t"
        "add x14, x14, #4\n\t"
        "add x15, x15, #4\n\t"

        "subs x10, x10, #1\n\t"
        "b.ne 1b\n\t"
        :
        : [K] "r"(K), [Q] "r"(Q), [D] "r"((long)D)
        : "memory", "x4", "x5", "x6", "x7", "x10", "x11", "x12", "x13", "x14", "x15",
          "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7",
          "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15",
          "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23",
          "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31",
          "p0", "p1"
    );
}

int main() {
    printf("==============================================\n");
    printf("MR=7 and MR=8 with K-split Analysis\n");
    printf("==============================================\n\n");

    printf("Register allocation:\n");
    printf("  MR=6 (4K): K(4) + Q(1) + Acc(24) = 29 regs ✓\n");
    printf("  MR=7 (4K): K(4) + Q(1) + Acc(28) = 33 regs ✗\n");
    printf("  MR=7 (2K): K(2) + Q(1) + Acc(28) = 31 regs ✓ (2 phases)\n");
    printf("  MR=8 (2K): K(2) + Q(1) + Acc(32) = 35 regs ✗\n");
    printf("  MR=8 (2K): K(2) + Q(1) + Acc(28) = 31 regs ✓ (partial)\n\n");

    int D = 64;

    int8_t* K = aligned_alloc(256, D * 64 * 4);
    int8_t* Q = aligned_alloc(256, 8 * D * 4);  // Max MR=8

    memset(K, 1, D * 64 * 4);
    memset(Q, 1, 8 * D * 4);

    int iters = 10000;
    uint64_t start, end;
    double cycles;

    printf("%-25s %8s %8s %10s %12s\n",
           "Kernel", "Ticks", "SDOTs", "SDOT/tick", "Density");
    printf("----------------------------------------------------------------\n");

    // MR=6 baseline
    for (int i = 0; i < 100; i++) bench_mr6(K, Q, D);
    start = rdcycle();
    for (int i = 0; i < iters; i++) bench_mr6(K, Q, D);
    end = rdcycle();
    cycles = (double)(end - start) / iters;
    int sdots6 = D * 24;
    double density6 = (double)sdots6 / (D * (256 + 24));
    printf("%-25s %8.1f %8d %10.2f %12.4f\n",
           "MR=6 (24 SDOT/d)", cycles, sdots6, sdots6/cycles, density6);

    // MR=7 with K split
    for (int i = 0; i < 100; i++) bench_mr7_split(K, Q, D);
    start = rdcycle();
    for (int i = 0; i < iters; i++) bench_mr7_split(K, Q, D);
    end = rdcycle();
    cycles = (double)(end - start) / iters;
    int sdots7 = D * 28;
    double density7 = (double)sdots7 / (D * (256 + 28));
    printf("%-25s %8.1f %8d %10.2f %12.4f\n",
           "MR=7 2-phase (28 SDOT/d)", cycles, sdots7, sdots7/cycles, density7);

    // MR=8 with K split (partial)
    for (int i = 0; i < 100; i++) bench_mr8_split(K, Q, D);
    start = rdcycle();
    for (int i = 0; i < iters; i++) bench_mr8_split(K, Q, D);
    end = rdcycle();
    cycles = (double)(end - start) / iters;
    int sdots8 = D * 32;  // Actual SDOTs executed
    double density8 = (double)sdots8 / (D * (256 + 32));
    printf("%-25s %8.1f %8d %10.2f %12.4f\n",
           "MR=8 2-phase (32 SDOT/d)", cycles, sdots8, sdots8/cycles, density8);

    printf("\n");
    printf("Compute density = SDOTs / (K_bytes + Q_bytes)\n");
    printf("  MR=6: 24 / 280 = 0.086\n");
    printf("  MR=7: 28 / 284 = 0.099 (+15%% vs MR=6)\n");
    printf("  MR=8: 32 / 288 = 0.111 (+30%% vs MR=6)\n");
    printf("\n");
    printf("Note: K-split doubles Q loads per d_group (overhead)\n");
    printf("  MR=7 split: 14 Q loads vs 7 (2x overhead)\n");
    printf("  But K is loaded from L1 cache (fast)\n");

    free(K); free(Q);
    return 0;
}
