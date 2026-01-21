/*
 * Optimized Fused GEMM MR=6
 *
 * Optimizations:
 * 1. Unroll D-group loop by 2 to reduce loop overhead
 * 2. Delay accumulator merge until end
 * 3. Software pipeline: overlap S store with V load
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

// Baseline MR=6 (from previous benchmark)
void fused_mr6_baseline(
    const int8_t* Q, const int8_t* K, const int8_t* V,
    int32_t* O, int D_groups
) {
    int32_t S[6 * 64] __attribute__((aligned(64)));

    __asm__ volatile(
        "ptrue p0.b\n\t"
        "ptrue p1.s\n\t"

        // Phase 1: Q @ K^T
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
        "mov x10, %[D_groups]\n\t"

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

        // Merge
        "add z8.s, z8.s, z9.s\n\t"
        "add z10.s, z10.s, z11.s\n\t"
        "add z8.s, z8.s, z10.s\n\t"
        "add z12.s, z12.s, z13.s\n\t"
        "add z14.s, z14.s, z15.s\n\t"
        "add z12.s, z12.s, z14.s\n\t"
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

        "mov x14, %[S]\n\t"
        "st1w {z8.s}, p1, [x14]\n\t"
        "st1w {z12.s}, p1, [x14, #1, mul vl]\n\t"
        "st1w {z16.s}, p1, [x14, #2, mul vl]\n\t"
        "st1w {z20.s}, p1, [x14, #3, mul vl]\n\t"
        "st1w {z24.s}, p1, [x14, #4, mul vl]\n\t"
        "st1w {z28.s}, p1, [x14, #5, mul vl]\n\t"

        // Phase 2
        "mov x15, #4\n\t"
        "mov x16, %[V]\n\t"
        "mov x17, %[O]\n\t"

        "2:\n\t"
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

        "mov x5, %[S]\n\t"
        "add x6, x5, #256\n\t"
        "add x7, x5, #512\n\t"
        "add x11, x5, #768\n\t"
        "add x12, x5, #1024\n\t"
        "add x13, x5, #1280\n\t"

        "mov x4, x16\n\t"
        "mov x10, #16\n\t"

        "3:\n\t"
        "ld1b {z0.b}, p0/z, [x4]\n\t"
        "ld1b {z1.b}, p0/z, [x4, #1, mul vl]\n\t"
        "ld1b {z2.b}, p0/z, [x4, #2, mul vl]\n\t"
        "ld1b {z3.b}, p0/z, [x4, #3, mul vl]\n\t"
        "add x4, x4, #1024\n\t"

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
        "b.ne 3b\n\t"

        "add z8.s, z8.s, z9.s\n\t"
        "add z10.s, z10.s, z11.s\n\t"
        "add z8.s, z8.s, z10.s\n\t"
        "add z12.s, z12.s, z13.s\n\t"
        "add z14.s, z14.s, z15.s\n\t"
        "add z12.s, z12.s, z14.s\n\t"
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

        "st1w {z8.s}, p1, [x17]\n\t"
        "add x17, x17, #1024\n\t"
        "st1w {z12.s}, p1, [x17]\n\t"
        "add x17, x17, #1024\n\t"
        "st1w {z16.s}, p1, [x17]\n\t"
        "add x17, x17, #1024\n\t"
        "st1w {z20.s}, p1, [x17]\n\t"
        "add x17, x17, #1024\n\t"
        "st1w {z24.s}, p1, [x17]\n\t"
        "add x17, x17, #1024\n\t"
        "st1w {z28.s}, p1, [x17]\n\t"

        "sub x17, x17, #4096\n\t"
        "sub x17, x17, #1024\n\t"
        "add x17, x17, #64\n\t"
        "add x16, x16, #64\n\t"

        "subs x15, x15, #1\n\t"
        "b.ne 2b\n\t"

        :
        : [Q] "r"(Q), [K] "r"(K), [V] "r"(V), [O] "r"(O),
          [S] "r"(S), [D_groups] "r"((long)D_groups)
        : "memory", "x4", "x5", "x6", "x7", "x10", "x11", "x12", "x13",
          "x14", "x15", "x16", "x17",
          "z0", "z1", "z2", "z3", "z4",
          "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15",
          "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23",
          "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31",
          "p0", "p1"
    );
}

// Optimized: 2x unroll + reduced pointer arithmetic
void fused_mr6_unroll2(
    const int8_t* Q, const int8_t* K, const int8_t* V,
    int32_t* O, int D_groups
) {
    int32_t S[6 * 64] __attribute__((aligned(64)));

    __asm__ volatile(
        "ptrue p0.b\n\t"
        "ptrue p1.s\n\t"

        // Phase 1: Q @ K^T with 2x unroll
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
        "mov x10, %[D_groups]\n\t"
        "lsr x10, x10, #1\n\t"  // D_groups / 2

        "1:\n\t"
        // === Iteration 0 ===
        "ld1b {z0.b}, p0/z, [x4]\n\t"
        "ld1b {z1.b}, p0/z, [x4, #1, mul vl]\n\t"
        "ld1b {z2.b}, p0/z, [x4, #2, mul vl]\n\t"
        "ld1b {z3.b}, p0/z, [x4, #3, mul vl]\n\t"

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

        // === Iteration 1 ===
        "ld1b {z0.b}, p0/z, [x4, #4, mul vl]\n\t"
        "ld1b {z1.b}, p0/z, [x4, #5, mul vl]\n\t"
        "ld1b {z2.b}, p0/z, [x4, #6, mul vl]\n\t"
        "ld1b {z3.b}, p0/z, [x4, #7, mul vl]\n\t"
        "add x4, x4, #512\n\t"  // 2 × 256

        "ld1rw {z4.s}, p1/z, [x5, #4]\n\t"
        "sdot z8.s, z0.b, z4.b\n\t"
        "sdot z9.s, z1.b, z4.b\n\t"
        "sdot z10.s, z2.b, z4.b\n\t"
        "sdot z11.s, z3.b, z4.b\n\t"

        "ld1rw {z4.s}, p1/z, [x6, #4]\n\t"
        "sdot z12.s, z0.b, z4.b\n\t"
        "sdot z13.s, z1.b, z4.b\n\t"
        "sdot z14.s, z2.b, z4.b\n\t"
        "sdot z15.s, z3.b, z4.b\n\t"

        "ld1rw {z4.s}, p1/z, [x7, #4]\n\t"
        "sdot z16.s, z0.b, z4.b\n\t"
        "sdot z17.s, z1.b, z4.b\n\t"
        "sdot z18.s, z2.b, z4.b\n\t"
        "sdot z19.s, z3.b, z4.b\n\t"

        "ld1rw {z4.s}, p1/z, [x11, #4]\n\t"
        "sdot z20.s, z0.b, z4.b\n\t"
        "sdot z21.s, z1.b, z4.b\n\t"
        "sdot z22.s, z2.b, z4.b\n\t"
        "sdot z23.s, z3.b, z4.b\n\t"

        "ld1rw {z4.s}, p1/z, [x12, #4]\n\t"
        "sdot z24.s, z0.b, z4.b\n\t"
        "sdot z25.s, z1.b, z4.b\n\t"
        "sdot z26.s, z2.b, z4.b\n\t"
        "sdot z27.s, z3.b, z4.b\n\t"

        "ld1rw {z4.s}, p1/z, [x13, #4]\n\t"
        "sdot z28.s, z0.b, z4.b\n\t"
        "sdot z29.s, z1.b, z4.b\n\t"
        "sdot z30.s, z2.b, z4.b\n\t"
        "sdot z31.s, z3.b, z4.b\n\t"

        "add x5, x5, #8\n\t"
        "add x6, x6, #8\n\t"
        "add x7, x7, #8\n\t"
        "add x11, x11, #8\n\t"
        "add x12, x12, #8\n\t"
        "add x13, x13, #8\n\t"

        "subs x10, x10, #1\n\t"
        "b.ne 1b\n\t"

        // Merge
        "add z8.s, z8.s, z9.s\n\t"
        "add z10.s, z10.s, z11.s\n\t"
        "add z8.s, z8.s, z10.s\n\t"
        "add z12.s, z12.s, z13.s\n\t"
        "add z14.s, z14.s, z15.s\n\t"
        "add z12.s, z12.s, z14.s\n\t"
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

        "mov x14, %[S]\n\t"
        "st1w {z8.s}, p1, [x14]\n\t"
        "st1w {z12.s}, p1, [x14, #1, mul vl]\n\t"
        "st1w {z16.s}, p1, [x14, #2, mul vl]\n\t"
        "st1w {z20.s}, p1, [x14, #3, mul vl]\n\t"
        "st1w {z24.s}, p1, [x14, #4, mul vl]\n\t"
        "st1w {z28.s}, p1, [x14, #5, mul vl]\n\t"

        // Phase 2 with 2x unroll
        "mov x15, #4\n\t"
        "mov x16, %[V]\n\t"
        "mov x17, %[O]\n\t"

        "2:\n\t"
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

        "mov x5, %[S]\n\t"
        "add x6, x5, #256\n\t"
        "add x7, x5, #512\n\t"
        "add x11, x5, #768\n\t"
        "add x12, x5, #1024\n\t"
        "add x13, x5, #1280\n\t"

        "mov x4, x16\n\t"
        "mov x10, #8\n\t"  // 16 / 2

        "3:\n\t"
        // Iteration 0
        "ld1b {z0.b}, p0/z, [x4]\n\t"
        "ld1b {z1.b}, p0/z, [x4, #1, mul vl]\n\t"
        "ld1b {z2.b}, p0/z, [x4, #2, mul vl]\n\t"
        "ld1b {z3.b}, p0/z, [x4, #3, mul vl]\n\t"

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

        // Iteration 1
        "add x4, x4, #1024\n\t"
        "ld1b {z0.b}, p0/z, [x4]\n\t"
        "ld1b {z1.b}, p0/z, [x4, #1, mul vl]\n\t"
        "ld1b {z2.b}, p0/z, [x4, #2, mul vl]\n\t"
        "ld1b {z3.b}, p0/z, [x4, #3, mul vl]\n\t"
        "add x4, x4, #1024\n\t"

        "ld1rw {z4.s}, p1/z, [x5, #4]\n\t"
        "sdot z8.s, z0.b, z4.b\n\t"
        "sdot z9.s, z1.b, z4.b\n\t"
        "sdot z10.s, z2.b, z4.b\n\t"
        "sdot z11.s, z3.b, z4.b\n\t"

        "ld1rw {z4.s}, p1/z, [x6, #4]\n\t"
        "sdot z12.s, z0.b, z4.b\n\t"
        "sdot z13.s, z1.b, z4.b\n\t"
        "sdot z14.s, z2.b, z4.b\n\t"
        "sdot z15.s, z3.b, z4.b\n\t"

        "ld1rw {z4.s}, p1/z, [x7, #4]\n\t"
        "sdot z16.s, z0.b, z4.b\n\t"
        "sdot z17.s, z1.b, z4.b\n\t"
        "sdot z18.s, z2.b, z4.b\n\t"
        "sdot z19.s, z3.b, z4.b\n\t"

        "ld1rw {z4.s}, p1/z, [x11, #4]\n\t"
        "sdot z20.s, z0.b, z4.b\n\t"
        "sdot z21.s, z1.b, z4.b\n\t"
        "sdot z22.s, z2.b, z4.b\n\t"
        "sdot z23.s, z3.b, z4.b\n\t"

        "ld1rw {z4.s}, p1/z, [x12, #4]\n\t"
        "sdot z24.s, z0.b, z4.b\n\t"
        "sdot z25.s, z1.b, z4.b\n\t"
        "sdot z26.s, z2.b, z4.b\n\t"
        "sdot z27.s, z3.b, z4.b\n\t"

        "ld1rw {z4.s}, p1/z, [x13, #4]\n\t"
        "sdot z28.s, z0.b, z4.b\n\t"
        "sdot z29.s, z1.b, z4.b\n\t"
        "sdot z30.s, z2.b, z4.b\n\t"
        "sdot z31.s, z3.b, z4.b\n\t"

        "add x5, x5, #8\n\t"
        "add x6, x6, #8\n\t"
        "add x7, x7, #8\n\t"
        "add x11, x11, #8\n\t"
        "add x12, x12, #8\n\t"
        "add x13, x13, #8\n\t"

        "subs x10, x10, #1\n\t"
        "b.ne 3b\n\t"

        "add z8.s, z8.s, z9.s\n\t"
        "add z10.s, z10.s, z11.s\n\t"
        "add z8.s, z8.s, z10.s\n\t"
        "add z12.s, z12.s, z13.s\n\t"
        "add z14.s, z14.s, z15.s\n\t"
        "add z12.s, z12.s, z14.s\n\t"
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

        "st1w {z8.s}, p1, [x17]\n\t"
        "add x17, x17, #1024\n\t"
        "st1w {z12.s}, p1, [x17]\n\t"
        "add x17, x17, #1024\n\t"
        "st1w {z16.s}, p1, [x17]\n\t"
        "add x17, x17, #1024\n\t"
        "st1w {z20.s}, p1, [x17]\n\t"
        "add x17, x17, #1024\n\t"
        "st1w {z24.s}, p1, [x17]\n\t"
        "add x17, x17, #1024\n\t"
        "st1w {z28.s}, p1, [x17]\n\t"

        "sub x17, x17, #4096\n\t"
        "sub x17, x17, #1024\n\t"
        "add x17, x17, #64\n\t"
        "add x16, x16, #64\n\t"

        "subs x15, x15, #1\n\t"
        "b.ne 2b\n\t"

        :
        : [Q] "r"(Q), [K] "r"(K), [V] "r"(V), [O] "r"(O),
          [S] "r"(S), [D_groups] "r"((long)D_groups)
        : "memory", "x4", "x5", "x6", "x7", "x10", "x11", "x12", "x13",
          "x14", "x15", "x16", "x17",
          "z0", "z1", "z2", "z3", "z4",
          "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15",
          "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23",
          "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31",
          "p0", "p1"
    );
}

int main() {
    printf("==============================================\n");
    printf("Fused GEMM MR=6 Optimization Comparison\n");
    printf("==============================================\n\n");

    int D_groups = 64;

    int8_t* Q = aligned_alloc(256, 6 * 256);
    int8_t* K = aligned_alloc(256, D_groups * 64 * 4);
    int8_t* V = aligned_alloc(256, 64 * 256 * 4);
    int32_t* O = aligned_alloc(256, 6 * 256 * 4);

    memset(Q, 1, 6 * 256);
    memset(K, 1, D_groups * 64 * 4);
    memset(V, 1, 64 * 256 * 4);

    int iters = 5000;
    uint64_t start, end;
    double ticks;

    int total_sdot = 3072;  // Phase 1 + Phase 2

    printf("%-25s %10s %12s\n", "Kernel", "Ticks", "SDOT/tick");
    printf("------------------------------------------------\n");

    // Warmup
    for (int i = 0; i < 100; i++) fused_mr6_baseline(Q, K, V, O, D_groups);

    // Baseline
    start = rdcycle();
    for (int i = 0; i < iters; i++) {
        fused_mr6_baseline(Q, K, V, O, D_groups);
    }
    end = rdcycle();
    ticks = (double)(end - start) / iters;
    printf("%-25s %10.1f %12.2f\n", "Baseline", ticks, total_sdot/ticks);

    // Warmup
    for (int i = 0; i < 100; i++) fused_mr6_unroll2(Q, K, V, O, D_groups);

    // Unroll 2x
    start = rdcycle();
    for (int i = 0; i < iters; i++) {
        fused_mr6_unroll2(Q, K, V, O, D_groups);
    }
    end = rdcycle();
    ticks = (double)(end - start) / iters;
    printf("%-25s %10.1f %12.2f\n", "Unroll 2x", ticks, total_sdot/ticks);

    printf("\n");
    printf("Theoretical peak: ~2.0 SDOT/cycle\n");
    printf("With ~20x counter factor: ~40 SDOT/tick at peak\n");

    free(Q); free(K); free(V); free(O);
    return 0;
}
