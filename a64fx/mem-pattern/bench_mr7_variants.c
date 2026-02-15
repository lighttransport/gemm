/*
 * MR=7 Variants Comparison:
 * 1. MR=7 with 3 K vectors (21 SDOTs, fits in registers)
 * 2. MR=7 with 4 K vectors + accumulator spilling to L1
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

// MR=6 baseline (4 K vectors, 24 SDOTs)
void bench_mr6_4k(const int8_t* K, const int8_t* Q, int D) {
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

        "mov x5, %[Q]\n\t"
        "add x6, x5, #256\n\t"
        "add x7, x5, #512\n\t"
        "add x11, x5, #768\n\t"
        "add x12, x5, #1024\n\t"
        "add x13, x5, #1280\n\t"
        "mov x4, %[K]\n\t"
        "mov x10, %[D]\n\t"

        "1:\n\t"
        "ld1b {z0.b}, p0/z, [x4]\n\t"
        "ld1b {z1.b}, p0/z, [x4, #1, mul vl]\n\t"
        "ld1b {z2.b}, p0/z, [x4, #2, mul vl]\n\t"
        "ld1b {z3.b}, p0/z, [x4, #3, mul vl]\n\t"
        "add x4, x4, #256\n\t"

        "ld1rw {z4.s}, p1/z, [x5]\n\t"
        "sdot z8.s, z0.b, z4.b\n\t"
        "sdot z9.s, z1.b, z4.b\n\t"
        "sdot z10.s, z2.b, z4.b\n\t"
        "sdot z11.s, z3.b, z4.b\n\t"

        "ld1rw {z4.s}, p1/z, [x6]\n\t"
        "sdot z12.s, z0.b, z4.b\n\t"
        "sdot z13.s, z1.b, z4.b\n\t"
        "sdot z14.s, z2.b, z4.b\n\t"
        "sdot z15.s, z3.b, z4.b\n\t"

        "ld1rw {z4.s}, p1/z, [x7]\n\t"
        "sdot z16.s, z0.b, z4.b\n\t"
        "sdot z17.s, z1.b, z4.b\n\t"
        "sdot z18.s, z2.b, z4.b\n\t"
        "sdot z19.s, z3.b, z4.b\n\t"

        "ld1rw {z4.s}, p1/z, [x11]\n\t"
        "sdot z20.s, z0.b, z4.b\n\t"
        "sdot z21.s, z1.b, z4.b\n\t"
        "sdot z22.s, z2.b, z4.b\n\t"
        "sdot z23.s, z3.b, z4.b\n\t"

        "ld1rw {z4.s}, p1/z, [x12]\n\t"
        "sdot z24.s, z0.b, z4.b\n\t"
        "sdot z25.s, z1.b, z4.b\n\t"
        "sdot z26.s, z2.b, z4.b\n\t"
        "sdot z27.s, z3.b, z4.b\n\t"

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

// MR=7 with 3 K vectors (21 SDOTs per d_group)
// Register: K(3) + Q(1) + Acc(21) = 25 regs
void bench_mr7_3k(const int8_t* K, const int8_t* Q, int D) {
    __asm__ volatile(
        "ptrue p0.b\n\t"
        "ptrue p1.s\n\t"

        // Zero 21 accumulators (z4-z24)
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
        // Load 3 K vectors (192 bytes)
        "ld1b {z0.b}, p0/z, [x4]\n\t"
        "ld1b {z1.b}, p0/z, [x4, #1, mul vl]\n\t"
        "ld1b {z2.b}, p0/z, [x4, #2, mul vl]\n\t"
        "add x4, x4, #192\n\t"  // 3 × 64B

        // Row 0: acc z4-z6
        "ld1rw {z3.s}, p1/z, [x5]\n\t"
        "sdot z4.s, z0.b, z3.b\n\t"
        "sdot z5.s, z1.b, z3.b\n\t"
        "sdot z6.s, z2.b, z3.b\n\t"

        // Row 1: acc z7-z9
        "ld1rw {z3.s}, p1/z, [x6]\n\t"
        "sdot z7.s, z0.b, z3.b\n\t"
        "sdot z8.s, z1.b, z3.b\n\t"
        "sdot z9.s, z2.b, z3.b\n\t"

        // Row 2: acc z10-z12
        "ld1rw {z3.s}, p1/z, [x7]\n\t"
        "sdot z10.s, z0.b, z3.b\n\t"
        "sdot z11.s, z1.b, z3.b\n\t"
        "sdot z12.s, z2.b, z3.b\n\t"

        // Row 3: acc z13-z15
        "ld1rw {z3.s}, p1/z, [x11]\n\t"
        "sdot z13.s, z0.b, z3.b\n\t"
        "sdot z14.s, z1.b, z3.b\n\t"
        "sdot z15.s, z2.b, z3.b\n\t"

        // Row 4: acc z16-z18
        "ld1rw {z3.s}, p1/z, [x12]\n\t"
        "sdot z16.s, z0.b, z3.b\n\t"
        "sdot z17.s, z1.b, z3.b\n\t"
        "sdot z18.s, z2.b, z3.b\n\t"

        // Row 5: acc z19-z21
        "ld1rw {z3.s}, p1/z, [x13]\n\t"
        "sdot z19.s, z0.b, z3.b\n\t"
        "sdot z20.s, z1.b, z3.b\n\t"
        "sdot z21.s, z2.b, z3.b\n\t"

        // Row 6: acc z22-z24
        "ld1rw {z3.s}, p1/z, [x14]\n\t"
        "sdot z22.s, z0.b, z3.b\n\t"
        "sdot z23.s, z1.b, z3.b\n\t"
        "sdot z24.s, z2.b, z3.b\n\t"

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
          "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24",
          "p0", "p1"
    );
}

// MR=8 with 3 K vectors (24 SDOTs per d_group) - same as MR=6 with 4K!
// Register: K(3) + Q(1) + Acc(24) = 28 regs
void bench_mr8_3k(const int8_t* K, const int8_t* Q, int D) {
    __asm__ volatile(
        "ptrue p0.b\n\t"
        "ptrue p1.s\n\t"

        // Zero 24 accumulators (z4-z27)
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
        // Load 3 K vectors
        "ld1b {z0.b}, p0/z, [x4]\n\t"
        "ld1b {z1.b}, p0/z, [x4, #1, mul vl]\n\t"
        "ld1b {z2.b}, p0/z, [x4, #2, mul vl]\n\t"
        "add x4, x4, #192\n\t"

        // 8 rows × 3 K = 24 SDOTs
        "ld1rw {z3.s}, p1/z, [x5]\n\t"
        "sdot z4.s, z0.b, z3.b\n\t"
        "sdot z5.s, z1.b, z3.b\n\t"
        "sdot z6.s, z2.b, z3.b\n\t"

        "ld1rw {z3.s}, p1/z, [x6]\n\t"
        "sdot z7.s, z0.b, z3.b\n\t"
        "sdot z8.s, z1.b, z3.b\n\t"
        "sdot z9.s, z2.b, z3.b\n\t"

        "ld1rw {z3.s}, p1/z, [x7]\n\t"
        "sdot z10.s, z0.b, z3.b\n\t"
        "sdot z11.s, z1.b, z3.b\n\t"
        "sdot z12.s, z2.b, z3.b\n\t"

        "ld1rw {z3.s}, p1/z, [x11]\n\t"
        "sdot z13.s, z0.b, z3.b\n\t"
        "sdot z14.s, z1.b, z3.b\n\t"
        "sdot z15.s, z2.b, z3.b\n\t"

        "ld1rw {z3.s}, p1/z, [x12]\n\t"
        "sdot z16.s, z0.b, z3.b\n\t"
        "sdot z17.s, z1.b, z3.b\n\t"
        "sdot z18.s, z2.b, z3.b\n\t"

        "ld1rw {z3.s}, p1/z, [x13]\n\t"
        "sdot z19.s, z0.b, z3.b\n\t"
        "sdot z20.s, z1.b, z3.b\n\t"
        "sdot z21.s, z2.b, z3.b\n\t"

        "ld1rw {z3.s}, p1/z, [x14]\n\t"
        "sdot z22.s, z0.b, z3.b\n\t"
        "sdot z23.s, z1.b, z3.b\n\t"
        "sdot z24.s, z2.b, z3.b\n\t"

        "ld1rw {z3.s}, p1/z, [x15]\n\t"
        "sdot z25.s, z0.b, z3.b\n\t"
        "sdot z26.s, z1.b, z3.b\n\t"
        "sdot z27.s, z2.b, z3.b\n\t"

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
          "z24", "z25", "z26", "z27",
          "p0", "p1"
    );
}

// MR=7 with 4 K vectors + L1 accumulator spilling
// Process 4 d_groups, then spill/restore 4 accumulators
void bench_mr7_4k_spill(const int8_t* K, const int8_t* Q, int32_t* spill, int D) {
    __asm__ volatile(
        "ptrue p0.b\n\t"
        "ptrue p1.s\n\t"

        // We need 28 accumulators but only have 32-5=27 available
        // Strategy: keep 24 in regs (z8-z31), spill 4 (row6) to L1 every iteration

        // Zero 24 accumulators for rows 0-5
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

        // Zero spill buffer for row 6 (4 accumulators)
        "mov x15, %[spill]\n\t"
        "mov z5.s, #0\n\t"
        "st1w {z5.s}, p1, [x15]\n\t"
        "st1w {z5.s}, p1, [x15, #1, mul vl]\n\t"
        "st1w {z5.s}, p1, [x15, #2, mul vl]\n\t"
        "st1w {z5.s}, p1, [x15, #3, mul vl]\n\t"

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
        // Load K
        "ld1b {z0.b}, p0/z, [x4]\n\t"
        "ld1b {z1.b}, p0/z, [x4, #1, mul vl]\n\t"
        "ld1b {z2.b}, p0/z, [x4, #2, mul vl]\n\t"
        "ld1b {z3.b}, p0/z, [x4, #3, mul vl]\n\t"
        "add x4, x4, #256\n\t"

        // Rows 0-5 (24 SDOTs, all in registers)
        "ld1rw {z4.s}, p1/z, [x5]\n\t"
        "sdot z8.s, z0.b, z4.b\n\t"
        "sdot z9.s, z1.b, z4.b\n\t"
        "sdot z10.s, z2.b, z4.b\n\t"
        "sdot z11.s, z3.b, z4.b\n\t"

        "ld1rw {z4.s}, p1/z, [x6]\n\t"
        "sdot z12.s, z0.b, z4.b\n\t"
        "sdot z13.s, z1.b, z4.b\n\t"
        "sdot z14.s, z2.b, z4.b\n\t"
        "sdot z15.s, z3.b, z4.b\n\t"

        "ld1rw {z4.s}, p1/z, [x7]\n\t"
        "sdot z16.s, z0.b, z4.b\n\t"
        "sdot z17.s, z1.b, z4.b\n\t"
        "sdot z18.s, z2.b, z4.b\n\t"
        "sdot z19.s, z3.b, z4.b\n\t"

        "ld1rw {z4.s}, p1/z, [x11]\n\t"
        "sdot z20.s, z0.b, z4.b\n\t"
        "sdot z21.s, z1.b, z4.b\n\t"
        "sdot z22.s, z2.b, z4.b\n\t"
        "sdot z23.s, z3.b, z4.b\n\t"

        "ld1rw {z4.s}, p1/z, [x12]\n\t"
        "sdot z24.s, z0.b, z4.b\n\t"
        "sdot z25.s, z1.b, z4.b\n\t"
        "sdot z26.s, z2.b, z4.b\n\t"
        "sdot z27.s, z3.b, z4.b\n\t"

        "ld1rw {z4.s}, p1/z, [x13]\n\t"
        "sdot z28.s, z0.b, z4.b\n\t"
        "sdot z29.s, z1.b, z4.b\n\t"
        "sdot z30.s, z2.b, z4.b\n\t"
        "sdot z31.s, z3.b, z4.b\n\t"

        // Row 6: load from spill, compute, store back
        "ld1w {z5.s}, p1/z, [x15]\n\t"
        "ld1w {z6.s}, p1/z, [x15, #1, mul vl]\n\t"
        "ld1w {z7.s}, p1/z, [x15, #2, mul vl]\n\t"
        "ld1rw {z4.s}, p1/z, [x14]\n\t"
        "sdot z5.s, z0.b, z4.b\n\t"
        "sdot z6.s, z1.b, z4.b\n\t"
        "sdot z7.s, z2.b, z4.b\n\t"
        "st1w {z5.s}, p1, [x15]\n\t"
        "st1w {z6.s}, p1, [x15, #1, mul vl]\n\t"
        "st1w {z7.s}, p1, [x15, #2, mul vl]\n\t"
        // 4th accumulator for row 6
        "ld1w {z5.s}, p1/z, [x15, #3, mul vl]\n\t"
        "sdot z5.s, z3.b, z4.b\n\t"
        "st1w {z5.s}, p1, [x15, #3, mul vl]\n\t"

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
        : [K] "r"(K), [Q] "r"(Q), [D] "r"((long)D), [spill] "r"(spill)
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
    printf("MR=7 Variants: 3K vs 4K+Spill\n");
    printf("==============================================\n\n");

    printf("Configurations:\n");
    printf("  MR=6, 4K: 24 SDOT/d, 280 bytes/d (baseline)\n");
    printf("  MR=7, 3K: 21 SDOT/d, 220 bytes/d\n");
    printf("  MR=8, 3K: 24 SDOT/d, 224 bytes/d\n");
    printf("  MR=7, 4K+spill: 28 SDOT/d, 284 bytes/d + 8 L1 ops\n\n");

    int D = 64;

    int8_t* K = aligned_alloc(256, D * 64 * 4);
    int8_t* Q = aligned_alloc(256, 8 * D * 4);
    int32_t* spill = aligned_alloc(256, 4 * 64);  // 4 vectors

    memset(K, 1, D * 64 * 4);
    memset(Q, 1, 8 * D * 4);

    int iters = 10000;
    uint64_t start, end;
    double ticks;

    printf("%-25s %8s %8s %10s %10s\n",
           "Kernel", "Ticks", "SDOTs", "SDOT/tick", "Density");
    printf("------------------------------------------------------------------\n");

    // MR=6 4K baseline
    for (int i = 0; i < 100; i++) bench_mr6_4k(K, Q, D);
    start = rdcycle();
    for (int i = 0; i < iters; i++) bench_mr6_4k(K, Q, D);
    end = rdcycle();
    ticks = (double)(end - start) / iters;
    printf("%-25s %8.1f %8d %10.2f %10.4f\n",
           "MR=6, 4K (24 SDOT)", ticks, D*24, (D*24)/ticks, 24.0/280);

    // MR=7 3K
    for (int i = 0; i < 100; i++) bench_mr7_3k(K, Q, D);
    start = rdcycle();
    for (int i = 0; i < iters; i++) bench_mr7_3k(K, Q, D);
    end = rdcycle();
    ticks = (double)(end - start) / iters;
    printf("%-25s %8.1f %8d %10.2f %10.4f\n",
           "MR=7, 3K (21 SDOT)", ticks, D*21, (D*21)/ticks, 21.0/220);

    // MR=8 3K
    for (int i = 0; i < 100; i++) bench_mr8_3k(K, Q, D);
    start = rdcycle();
    for (int i = 0; i < iters; i++) bench_mr8_3k(K, Q, D);
    end = rdcycle();
    ticks = (double)(end - start) / iters;
    printf("%-25s %8.1f %8d %10.2f %10.4f\n",
           "MR=8, 3K (24 SDOT)", ticks, D*24, (D*24)/ticks, 24.0/224);

    // MR=7 4K with spill
    for (int i = 0; i < 100; i++) bench_mr7_4k_spill(K, Q, spill, D);
    start = rdcycle();
    for (int i = 0; i < iters; i++) bench_mr7_4k_spill(K, Q, spill, D);
    end = rdcycle();
    ticks = (double)(end - start) / iters;
    printf("%-25s %8.1f %8d %10.2f %10.4f\n",
           "MR=7, 4K+spill (28 SDOT)", ticks, D*28, (D*28)/ticks, 28.0/284);

    printf("\n");
    printf("Density = SDOT / (K_bytes + Q_bytes)\n");
    printf("  Higher density = better compute/memory ratio\n");
    printf("\n");
    printf("Spill overhead per d_group:\n");
    printf("  4 ld1w (load spilled acc) + 4 st1w (store back)\n");
    printf("  = 256B load + 256B store per iteration\n");

    free(K); free(Q); free(spill);
    return 0;
}
