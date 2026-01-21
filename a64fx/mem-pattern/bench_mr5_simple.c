/*
 * Benchmark: MR=5 single tile efficiency
 * Compare with MR=4 single and MR=4×2
 *
 * Goal: See if 20 SDOTs/iteration (MR=5) is enough to approach peak
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

// MR=4 single tile - baseline (16 SDOTs per d_group)
void bench_mr4_single(const int8_t* K, const int8_t* Q, int D) {
    __asm__ volatile(
        "ptrue p0.b\n\t"
        "ptrue p1.s\n\t"

        // Zero 16 accumulators
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

        "mov x5, %[Q]\n\t"
        "add x6, x5, #256\n\t"
        "add x7, x5, #512\n\t"
        "add x8, x5, #768\n\t"
        "mov x4, %[K]\n\t"
        "mov x10, %[D]\n\t"

        "1:\n\t"
        // Load K
        "ld1b {z0.b}, p0/z, [x4]\n\t"
        "ld1b {z1.b}, p0/z, [x4, #1, mul vl]\n\t"
        "ld1b {z2.b}, p0/z, [x4, #2, mul vl]\n\t"
        "ld1b {z3.b}, p0/z, [x4, #3, mul vl]\n\t"
        "add x4, x4, #256\n\t"

        // Load Q
        "ld1rw {z4.s}, p1/z, [x5]\n\t"
        "ld1rw {z5.s}, p1/z, [x6]\n\t"
        "ld1rw {z6.s}, p1/z, [x7]\n\t"
        "ld1rw {z7.s}, p1/z, [x8]\n\t"
        "add x5, x5, #4\n\t"
        "add x6, x6, #4\n\t"
        "add x7, x7, #4\n\t"
        "add x8, x8, #4\n\t"

        // 16 SDOTs (4 rows × 4 K)
        "sdot z8.s, z0.b, z4.b\n\t"
        "sdot z9.s, z1.b, z4.b\n\t"
        "sdot z10.s, z2.b, z4.b\n\t"
        "sdot z11.s, z3.b, z4.b\n\t"
        "sdot z12.s, z0.b, z5.b\n\t"
        "sdot z13.s, z1.b, z5.b\n\t"
        "sdot z14.s, z2.b, z5.b\n\t"
        "sdot z15.s, z3.b, z5.b\n\t"
        "sdot z16.s, z0.b, z6.b\n\t"
        "sdot z17.s, z1.b, z6.b\n\t"
        "sdot z18.s, z2.b, z6.b\n\t"
        "sdot z19.s, z3.b, z6.b\n\t"
        "sdot z20.s, z0.b, z7.b\n\t"
        "sdot z21.s, z1.b, z7.b\n\t"
        "sdot z22.s, z2.b, z7.b\n\t"
        "sdot z23.s, z3.b, z7.b\n\t"

        "subs x10, x10, #1\n\t"
        "b.ne 1b\n\t"
        :
        : [K] "r"(K), [Q] "r"(Q), [D] "r"((long)D)
        : "memory", "x4", "x5", "x6", "x7", "x8", "x10",
          "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7",
          "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15",
          "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23",
          "p0", "p1"
    );
}

// MR=5 single tile (20 SDOTs per d_group)
void bench_mr5_single(const int8_t* K, const int8_t* Q, int D) {
    __asm__ volatile(
        "ptrue p0.b\n\t"
        "ptrue p1.s\n\t"

        // Zero 20 accumulators (z9-z28)
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

        // Q row pointers (5 rows)
        "mov x5, %[Q]\n\t"
        "add x6, x5, #256\n\t"
        "add x7, x5, #512\n\t"
        "add x8, x5, #768\n\t"
        "add x9, x5, #1024\n\t"
        "mov x4, %[K]\n\t"
        "mov x10, %[D]\n\t"

        "1:\n\t"
        // Load K (4 vectors)
        "ld1b {z0.b}, p0/z, [x4]\n\t"
        "ld1b {z1.b}, p0/z, [x4, #1, mul vl]\n\t"
        "ld1b {z2.b}, p0/z, [x4, #2, mul vl]\n\t"
        "ld1b {z3.b}, p0/z, [x4, #3, mul vl]\n\t"
        "add x4, x4, #256\n\t"

        // Load Q (5 broadcasts)
        "ld1rw {z4.s}, p1/z, [x5]\n\t"
        "ld1rw {z5.s}, p1/z, [x6]\n\t"
        "ld1rw {z6.s}, p1/z, [x7]\n\t"
        "ld1rw {z7.s}, p1/z, [x8]\n\t"
        "ld1rw {z8.s}, p1/z, [x9]\n\t"
        "add x5, x5, #4\n\t"
        "add x6, x6, #4\n\t"
        "add x7, x7, #4\n\t"
        "add x8, x8, #4\n\t"
        "add x9, x9, #4\n\t"

        // 20 SDOTs (5 rows × 4 K)
        // Row 0
        "sdot z9.s, z0.b, z4.b\n\t"
        "sdot z10.s, z1.b, z4.b\n\t"
        "sdot z11.s, z2.b, z4.b\n\t"
        "sdot z12.s, z3.b, z4.b\n\t"
        // Row 1
        "sdot z13.s, z0.b, z5.b\n\t"
        "sdot z14.s, z1.b, z5.b\n\t"
        "sdot z15.s, z2.b, z5.b\n\t"
        "sdot z16.s, z3.b, z5.b\n\t"
        // Row 2
        "sdot z17.s, z0.b, z6.b\n\t"
        "sdot z18.s, z1.b, z6.b\n\t"
        "sdot z19.s, z2.b, z6.b\n\t"
        "sdot z20.s, z3.b, z6.b\n\t"
        // Row 3
        "sdot z21.s, z0.b, z7.b\n\t"
        "sdot z22.s, z1.b, z7.b\n\t"
        "sdot z23.s, z2.b, z7.b\n\t"
        "sdot z24.s, z3.b, z7.b\n\t"
        // Row 4
        "sdot z25.s, z0.b, z8.b\n\t"
        "sdot z26.s, z1.b, z8.b\n\t"
        "sdot z27.s, z2.b, z8.b\n\t"
        "sdot z28.s, z3.b, z8.b\n\t"

        "subs x10, x10, #1\n\t"
        "b.ne 1b\n\t"
        :
        : [K] "r"(K), [Q] "r"(Q), [D] "r"((long)D)
        : "memory", "x4", "x5", "x6", "x7", "x8", "x9", "x10",
          "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8",
          "z9", "z10", "z11", "z12", "z13", "z14", "z15", "z16",
          "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24",
          "z25", "z26", "z27", "z28", "p0", "p1"
    );
}

// MR=6 single tile (24 SDOTs per d_group) - pushes register limit
void bench_mr6_single(const int8_t* K, const int8_t* Q, int D) {
    __asm__ volatile(
        "ptrue p0.b\n\t"
        "ptrue p1.s\n\t"

        // Zero 24 accumulators (z8-z31) - uses ALL remaining registers
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

        // Q row pointers (6 rows) - need x5-x10
        "mov x5, %[Q]\n\t"
        "add x6, x5, #256\n\t"
        "add x7, x5, #512\n\t"
        "add x11, x5, #768\n\t"   // Can't use x8-x10 for all, using x11-x13
        "add x12, x5, #1024\n\t"
        "add x13, x5, #1280\n\t"
        "mov x4, %[K]\n\t"
        "mov x10, %[D]\n\t"

        "1:\n\t"
        // Load K (4 vectors) - reuse z0-z3, then load Q into z4-z7 one at a time
        "ld1b {z0.b}, p0/z, [x4]\n\t"
        "ld1b {z1.b}, p0/z, [x4, #1, mul vl]\n\t"
        "ld1b {z2.b}, p0/z, [x4, #2, mul vl]\n\t"
        "ld1b {z3.b}, p0/z, [x4, #3, mul vl]\n\t"
        "add x4, x4, #256\n\t"

        // Row 0: load Q[0], compute, reuse z4
        "ld1rw {z4.s}, p1/z, [x5]\n\t"
        "sdot z8.s, z0.b, z4.b\n\t"
        "sdot z9.s, z1.b, z4.b\n\t"
        "sdot z10.s, z2.b, z4.b\n\t"
        "sdot z11.s, z3.b, z4.b\n\t"

        // Row 1: load Q[1], compute
        "ld1rw {z4.s}, p1/z, [x6]\n\t"
        "sdot z12.s, z0.b, z4.b\n\t"
        "sdot z13.s, z1.b, z4.b\n\t"
        "sdot z14.s, z2.b, z4.b\n\t"
        "sdot z15.s, z3.b, z4.b\n\t"

        // Row 2: load Q[2], compute
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

// MR=4 × 2 tiles (32 SDOTs per d_group, shared K) - from bench_tile_groups
void bench_mr4x2(const int8_t* K, const int8_t* Q_A, const int8_t* Q_B, int D) {
    __asm__ volatile(
        "ptrue p0.b\n\t"
        "ptrue p1.s\n\t"

        // Zero 32 accumulators - need to spill K after loading
        // Strategy: load K, compute group A, reload K (from cache), compute group B
        // Accumulators: z8-z23 (16) for A, z24-z31 + z5-z7 (12) for B... won't fit

        // Alternative: process row by row, accumulate into fewer registers
        // This is complex - let's use the simpler interleaved approach from tile_groups

        // Actually use the working version from bench_tile_groups.c
        // z0-z3: K, z4-z7: Q_A, z8-z11: Q_B (reuse after K)
        // z12-z27: 16 accum A, z28-z31 + need more: won't fit for full 32

        // Simplified: keep 16 accum for A in regs, accumulate B on the fly with add
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

        // Q_A pointers
        "mov x5, %[Q_A]\n\t"
        "add x6, x5, #256\n\t"
        "add x7, x5, #512\n\t"
        "add x8, x5, #768\n\t"
        // Q_B pointers
        "mov x11, %[Q_B]\n\t"
        "add x12, x11, #256\n\t"
        "add x13, x11, #512\n\t"
        "add x14, x11, #768\n\t"

        "mov x4, %[K]\n\t"
        "mov x10, %[D]\n\t"

        "1:\n\t"
        // Load K
        "ld1b {z0.b}, p0/z, [x4]\n\t"
        "ld1b {z1.b}, p0/z, [x4, #1, mul vl]\n\t"
        "ld1b {z2.b}, p0/z, [x4, #2, mul vl]\n\t"
        "ld1b {z3.b}, p0/z, [x4, #3, mul vl]\n\t"
        "add x4, x4, #256\n\t"

        // Load Q_A
        "ld1rw {z4.s}, p1/z, [x5]\n\t"
        "ld1rw {z5.s}, p1/z, [x6]\n\t"
        "ld1rw {z6.s}, p1/z, [x7]\n\t"
        "ld1rw {z7.s}, p1/z, [x8]\n\t"

        // 16 SDOTs group A - interleave with Q_B loads
        "sdot z16.s, z0.b, z4.b\n\t"
        "ld1rw {z8.s}, p1/z, [x11]\n\t"
        "sdot z17.s, z1.b, z4.b\n\t"
        "ld1rw {z9.s}, p1/z, [x12]\n\t"
        "sdot z18.s, z2.b, z4.b\n\t"
        "ld1rw {z10.s}, p1/z, [x13]\n\t"
        "sdot z19.s, z3.b, z4.b\n\t"
        "ld1rw {z11.s}, p1/z, [x14]\n\t"

        "sdot z20.s, z0.b, z5.b\n\t"
        "sdot z21.s, z1.b, z5.b\n\t"
        "sdot z22.s, z2.b, z5.b\n\t"
        "sdot z23.s, z3.b, z5.b\n\t"
        "sdot z24.s, z0.b, z6.b\n\t"
        "sdot z25.s, z1.b, z6.b\n\t"
        "sdot z26.s, z2.b, z6.b\n\t"
        "sdot z27.s, z3.b, z6.b\n\t"
        "sdot z28.s, z0.b, z7.b\n\t"
        "sdot z29.s, z1.b, z7.b\n\t"
        "sdot z30.s, z2.b, z7.b\n\t"
        "sdot z31.s, z3.b, z7.b\n\t"

        // 16 SDOTs group B - accumulate into z4-z7 (reuse Q_A regs)
        // But we need separate accumulators... this doesn't work properly
        // Let's just count SDOTs for now
        "sdot z4.s, z0.b, z8.b\n\t"
        "sdot z5.s, z1.b, z8.b\n\t"
        "sdot z6.s, z2.b, z8.b\n\t"
        "sdot z7.s, z3.b, z8.b\n\t"
        "sdot z12.s, z0.b, z9.b\n\t"
        "sdot z13.s, z1.b, z9.b\n\t"
        "sdot z14.s, z2.b, z9.b\n\t"
        "sdot z15.s, z3.b, z9.b\n\t"

        // Remaining B SDOTs - reuse some regs (correctness sacrificed for throughput test)
        "sdot z4.s, z0.b, z10.b\n\t"
        "sdot z5.s, z1.b, z10.b\n\t"
        "sdot z6.s, z2.b, z10.b\n\t"
        "sdot z7.s, z3.b, z10.b\n\t"
        "sdot z12.s, z0.b, z11.b\n\t"
        "sdot z13.s, z1.b, z11.b\n\t"
        "sdot z14.s, z2.b, z11.b\n\t"
        "sdot z15.s, z3.b, z11.b\n\t"

        "add x5, x5, #4\n\t"
        "add x6, x6, #4\n\t"
        "add x7, x7, #4\n\t"
        "add x8, x8, #4\n\t"
        "add x11, x11, #4\n\t"
        "add x12, x12, #4\n\t"
        "add x13, x13, #4\n\t"
        "add x14, x14, #4\n\t"

        "subs x10, x10, #1\n\t"
        "b.ne 1b\n\t"
        :
        : [K] "r"(K), [Q_A] "r"(Q_A), [Q_B] "r"(Q_B), [D] "r"((long)D)
        : "memory", "x4", "x5", "x6", "x7", "x8", "x10", "x11", "x12", "x13", "x14",
          "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7",
          "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15",
          "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23",
          "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31",
          "p0", "p1"
    );
}

int main() {
    printf("==============================================\n");
    printf("MR Tile Size Comparison\n");
    printf("==============================================\n\n");

    printf("Register budget (32 SVE regs):\n");
    printf("  MR=4: K(4) + Q(4) + Acc(16) = 24 regs ✓\n");
    printf("  MR=5: K(4) + Q(5) + Acc(20) = 29 regs ✓\n");
    printf("  MR=6: K(4) + Q(1,reuse) + Acc(24) = 29 regs ✓\n");
    printf("  MR=4×2: K(4) + Q(8) + Acc(32) = 44 regs ✗\n\n");

    int D = 64;
    int N = 64;

    int8_t* K = aligned_alloc(256, D * N * 4);
    int8_t* Q = aligned_alloc(256, 6 * D * 4);  // Max MR=6
    int8_t* Q_B = aligned_alloc(256, 4 * D * 4);

    memset(K, 1, D * N * 4);
    memset(Q, 1, 6 * D * 4);
    memset(Q_B, 1, 4 * D * 4);

    int iters = 10000;
    uint64_t start, end;
    double cycles;

    printf("%-20s %10s %10s %10s\n", "Kernel", "Cycles", "SDOTs", "SDOT/cy");
    printf("------------------------------------------------------------\n");

    // MR=4 single
    for (int i = 0; i < 100; i++) bench_mr4_single(K, Q, D);
    start = rdcycle();
    for (int i = 0; i < iters; i++) bench_mr4_single(K, Q, D);
    end = rdcycle();
    cycles = (double)(end - start) / iters;
    printf("%-20s %10.1f %10d %10.2f\n", "MR=4 (16 SDOT/iter)", cycles, D*16, (D*16)/cycles);

    // MR=5 single
    for (int i = 0; i < 100; i++) bench_mr5_single(K, Q, D);
    start = rdcycle();
    for (int i = 0; i < iters; i++) bench_mr5_single(K, Q, D);
    end = rdcycle();
    cycles = (double)(end - start) / iters;
    printf("%-20s %10.1f %10d %10.2f\n", "MR=5 (20 SDOT/iter)", cycles, D*20, (D*20)/cycles);

    // MR=6 single
    for (int i = 0; i < 100; i++) bench_mr6_single(K, Q, D);
    start = rdcycle();
    for (int i = 0; i < iters; i++) bench_mr6_single(K, Q, D);
    end = rdcycle();
    cycles = (double)(end - start) / iters;
    printf("%-20s %10.1f %10d %10.2f\n", "MR=6 (24 SDOT/iter)", cycles, D*24, (D*24)/cycles);

    // MR=4×2 (32 SDOT but register-limited)
    for (int i = 0; i < 100; i++) bench_mr4x2(K, Q, Q_B, D);
    start = rdcycle();
    for (int i = 0; i < iters; i++) bench_mr4x2(K, Q, Q_B, D);
    end = rdcycle();
    cycles = (double)(end - start) / iters;
    printf("%-20s %10.1f %10d %10.2f\n", "MR=4×2 (32 SDOT/iter)", cycles, D*32, (D*32)/cycles);

    printf("\n");
    printf("Theoretical peak: 2.0 SDOT/cycle (2 FPUs)\n");
    printf("Compute density = SDOTs / (K_bytes + Q_bytes)\n");
    printf("  MR=4: 16 / (256 + 16) = 0.059 SDOT/byte\n");
    printf("  MR=5: 20 / (256 + 20) = 0.072 SDOT/byte (+22%%)\n");
    printf("  MR=6: 24 / (256 + 24) = 0.086 SDOT/byte (+45%%)\n");

    free(K); free(Q); free(Q_B);
    return 0;
}
