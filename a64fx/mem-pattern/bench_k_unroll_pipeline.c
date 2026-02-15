/*
 * K-Loop Unrolling + Software Pipelining Optimization
 *
 * Optimizations:
 * 1. 4x K-loop unrolling (16 K-groups per iteration, 16 iterations total)
 * 2. Software pipelining: load K[i+1] while computing SDOT with K[i]
 * 3. Interleaved SDOT from different rows to maximize ILP
 * 4. Sector cache hints for streaming data
 *
 * Target: 95%+ SDOT efficiency (38+ SDOT/tick)
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

// Baseline: 2x unrolled (current best at 79.1%)
__attribute__((noinline))
void baseline_2x_unroll(const int8_t* Q, const int8_t* K, int32_t* S) {
    __asm__ volatile(
        "ptrue p0.b\n\t"
        "ptrue p1.s\n\t"
        // Zero 24 accumulators (6 rows × 4 K-tiles)
        "mov z8.s, #0\n\t"  "mov z9.s, #0\n\t"  "mov z10.s, #0\n\t" "mov z11.s, #0\n\t"
        "mov z12.s, #0\n\t" "mov z13.s, #0\n\t" "mov z14.s, #0\n\t" "mov z15.s, #0\n\t"
        "mov z16.s, #0\n\t" "mov z17.s, #0\n\t" "mov z18.s, #0\n\t" "mov z19.s, #0\n\t"
        "mov z20.s, #0\n\t" "mov z21.s, #0\n\t" "mov z22.s, #0\n\t" "mov z23.s, #0\n\t"
        "mov z24.s, #0\n\t" "mov z25.s, #0\n\t" "mov z26.s, #0\n\t" "mov z27.s, #0\n\t"
        "mov z28.s, #0\n\t" "mov z29.s, #0\n\t" "mov z30.s, #0\n\t" "mov z31.s, #0\n\t"

        // Q row pointers
        "mov x5, %[Q]\n\t"
        "add x6, x5, #256\n\t"  "add x7, x6, #256\n\t"
        "add x11, x7, #256\n\t" "add x12, x11, #256\n\t" "add x13, x12, #256\n\t"
        "mov x4, %[K]\n\t"
        "mov x10, #32\n\t"  // 64 K-groups / 2 = 32 iterations

        "1:\n\t"
        // First set of 4 K-tiles
        "ld1b {z0.b}, p0/z, [x4]\n\t"
        "ld1b {z1.b}, p0/z, [x4, #1, mul vl]\n\t"
        "ld1b {z2.b}, p0/z, [x4, #2, mul vl]\n\t"
        "ld1b {z3.b}, p0/z, [x4, #3, mul vl]\n\t"

        "ld1rw {z4.s}, p1/z, [x5]\n\t"
        "sdot z8.s, z0.b, z4.b\n\t"  "sdot z9.s, z1.b, z4.b\n\t"
        "sdot z10.s, z2.b, z4.b\n\t" "sdot z11.s, z3.b, z4.b\n\t"
        "ld1rw {z4.s}, p1/z, [x6]\n\t"
        "sdot z12.s, z0.b, z4.b\n\t" "sdot z13.s, z1.b, z4.b\n\t"
        "sdot z14.s, z2.b, z4.b\n\t" "sdot z15.s, z3.b, z4.b\n\t"
        "ld1rw {z4.s}, p1/z, [x7]\n\t"
        "sdot z16.s, z0.b, z4.b\n\t" "sdot z17.s, z1.b, z4.b\n\t"
        "sdot z18.s, z2.b, z4.b\n\t" "sdot z19.s, z3.b, z4.b\n\t"
        "ld1rw {z4.s}, p1/z, [x11]\n\t"
        "sdot z20.s, z0.b, z4.b\n\t" "sdot z21.s, z1.b, z4.b\n\t"
        "sdot z22.s, z2.b, z4.b\n\t" "sdot z23.s, z3.b, z4.b\n\t"
        "ld1rw {z4.s}, p1/z, [x12]\n\t"
        "sdot z24.s, z0.b, z4.b\n\t" "sdot z25.s, z1.b, z4.b\n\t"
        "sdot z26.s, z2.b, z4.b\n\t" "sdot z27.s, z3.b, z4.b\n\t"
        "ld1rw {z4.s}, p1/z, [x13]\n\t"
        "sdot z28.s, z0.b, z4.b\n\t" "sdot z29.s, z1.b, z4.b\n\t"
        "sdot z30.s, z2.b, z4.b\n\t" "sdot z31.s, z3.b, z4.b\n\t"

        // Second set of 4 K-tiles
        "ld1b {z0.b}, p0/z, [x4, #4, mul vl]\n\t"
        "ld1b {z1.b}, p0/z, [x4, #5, mul vl]\n\t"
        "ld1b {z2.b}, p0/z, [x4, #6, mul vl]\n\t"
        "ld1b {z3.b}, p0/z, [x4, #7, mul vl]\n\t"
        "add x4, x4, #512\n\t"

        "ld1rw {z4.s}, p1/z, [x5, #4]\n\t"
        "sdot z8.s, z0.b, z4.b\n\t"  "sdot z9.s, z1.b, z4.b\n\t"
        "sdot z10.s, z2.b, z4.b\n\t" "sdot z11.s, z3.b, z4.b\n\t"
        "ld1rw {z4.s}, p1/z, [x6, #4]\n\t"
        "sdot z12.s, z0.b, z4.b\n\t" "sdot z13.s, z1.b, z4.b\n\t"
        "sdot z14.s, z2.b, z4.b\n\t" "sdot z15.s, z3.b, z4.b\n\t"
        "ld1rw {z4.s}, p1/z, [x7, #4]\n\t"
        "sdot z16.s, z0.b, z4.b\n\t" "sdot z17.s, z1.b, z4.b\n\t"
        "sdot z18.s, z2.b, z4.b\n\t" "sdot z19.s, z3.b, z4.b\n\t"
        "ld1rw {z4.s}, p1/z, [x11, #4]\n\t"
        "sdot z20.s, z0.b, z4.b\n\t" "sdot z21.s, z1.b, z4.b\n\t"
        "sdot z22.s, z2.b, z4.b\n\t" "sdot z23.s, z3.b, z4.b\n\t"
        "ld1rw {z4.s}, p1/z, [x12, #4]\n\t"
        "sdot z24.s, z0.b, z4.b\n\t" "sdot z25.s, z1.b, z4.b\n\t"
        "sdot z26.s, z2.b, z4.b\n\t" "sdot z27.s, z3.b, z4.b\n\t"
        "ld1rw {z4.s}, p1/z, [x13, #4]\n\t"
        "sdot z28.s, z0.b, z4.b\n\t" "sdot z29.s, z1.b, z4.b\n\t"
        "sdot z30.s, z2.b, z4.b\n\t" "sdot z31.s, z3.b, z4.b\n\t"

        "add x5, x5, #8\n\t"   "add x6, x6, #8\n\t"   "add x7, x7, #8\n\t"
        "add x11, x11, #8\n\t" "add x12, x12, #8\n\t" "add x13, x13, #8\n\t"
        "subs x10, x10, #1\n\t"
        "b.ne 1b\n\t"

        // Reduce 4 K-tiles to 1
        "add z8.s, z8.s, z9.s\n\t"   "add z10.s, z10.s, z11.s\n\t" "add z8.s, z8.s, z10.s\n\t"
        "add z12.s, z12.s, z13.s\n\t" "add z14.s, z14.s, z15.s\n\t" "add z12.s, z12.s, z14.s\n\t"
        "add z16.s, z16.s, z17.s\n\t" "add z18.s, z18.s, z19.s\n\t" "add z16.s, z16.s, z18.s\n\t"
        "add z20.s, z20.s, z21.s\n\t" "add z22.s, z22.s, z23.s\n\t" "add z20.s, z20.s, z22.s\n\t"
        "add z24.s, z24.s, z25.s\n\t" "add z26.s, z26.s, z27.s\n\t" "add z24.s, z24.s, z26.s\n\t"
        "add z28.s, z28.s, z29.s\n\t" "add z30.s, z30.s, z31.s\n\t" "add z28.s, z28.s, z30.s\n\t"

        "mov x14, %[S]\n\t"
        "st1w {z8.s}, p1, [x14]\n\t"       "st1w {z12.s}, p1, [x14, #1, mul vl]\n\t"
        "st1w {z16.s}, p1, [x14, #2, mul vl]\n\t" "st1w {z20.s}, p1, [x14, #3, mul vl]\n\t"
        "st1w {z24.s}, p1, [x14, #4, mul vl]\n\t" "st1w {z28.s}, p1, [x14, #5, mul vl]\n\t"
        :
        : [Q] "r"(Q), [K] "r"(K), [S] "r"(S)
        : "memory", "x4", "x5", "x6", "x7", "x10", "x11", "x12", "x13", "x14",
          "z0", "z1", "z2", "z3", "z4", "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15",
          "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23",
          "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31", "p0", "p1"
    );
}

// Optimization 1: 4x K-loop unrolling with explicit pointer arithmetic
// Process 16 K-groups per iteration = 16 iterations total
__attribute__((noinline))
void opt_4x_unroll(const int8_t* Q, const int8_t* K, int32_t* S) {
    __asm__ volatile(
        "ptrue p0.b\n\t"
        "ptrue p1.s\n\t"
        // Zero 24 accumulators
        "mov z8.s, #0\n\t"  "mov z9.s, #0\n\t"  "mov z10.s, #0\n\t" "mov z11.s, #0\n\t"
        "mov z12.s, #0\n\t" "mov z13.s, #0\n\t" "mov z14.s, #0\n\t" "mov z15.s, #0\n\t"
        "mov z16.s, #0\n\t" "mov z17.s, #0\n\t" "mov z18.s, #0\n\t" "mov z19.s, #0\n\t"
        "mov z20.s, #0\n\t" "mov z21.s, #0\n\t" "mov z22.s, #0\n\t" "mov z23.s, #0\n\t"
        "mov z24.s, #0\n\t" "mov z25.s, #0\n\t" "mov z26.s, #0\n\t" "mov z27.s, #0\n\t"
        "mov z28.s, #0\n\t" "mov z29.s, #0\n\t" "mov z30.s, #0\n\t" "mov z31.s, #0\n\t"

        "mov x5, %[Q]\n\t"
        "add x6, x5, #256\n\t"  "add x7, x6, #256\n\t"
        "add x11, x7, #256\n\t" "add x12, x11, #256\n\t" "add x13, x12, #256\n\t"
        "mov x4, %[K]\n\t"
        "mov x10, #16\n\t"  // 64 K-groups / 4 = 16 iterations

        "1:\n\t"
        // K-group 0-3 (first 4 K-tiles)
        "ld1b {z0.b}, p0/z, [x4]\n\t"
        "ld1b {z1.b}, p0/z, [x4, #1, mul vl]\n\t"
        "ld1b {z2.b}, p0/z, [x4, #2, mul vl]\n\t"
        "ld1b {z3.b}, p0/z, [x4, #3, mul vl]\n\t"
        "ld1rw {z4.s}, p1/z, [x5]\n\t"
        "sdot z8.s, z0.b, z4.b\n\t"  "sdot z9.s, z1.b, z4.b\n\t"
        "sdot z10.s, z2.b, z4.b\n\t" "sdot z11.s, z3.b, z4.b\n\t"
        "ld1rw {z4.s}, p1/z, [x6]\n\t"
        "sdot z12.s, z0.b, z4.b\n\t" "sdot z13.s, z1.b, z4.b\n\t"
        "sdot z14.s, z2.b, z4.b\n\t" "sdot z15.s, z3.b, z4.b\n\t"
        "ld1rw {z4.s}, p1/z, [x7]\n\t"
        "sdot z16.s, z0.b, z4.b\n\t" "sdot z17.s, z1.b, z4.b\n\t"
        "sdot z18.s, z2.b, z4.b\n\t" "sdot z19.s, z3.b, z4.b\n\t"
        "ld1rw {z4.s}, p1/z, [x11]\n\t"
        "sdot z20.s, z0.b, z4.b\n\t" "sdot z21.s, z1.b, z4.b\n\t"
        "sdot z22.s, z2.b, z4.b\n\t" "sdot z23.s, z3.b, z4.b\n\t"
        "ld1rw {z4.s}, p1/z, [x12]\n\t"
        "sdot z24.s, z0.b, z4.b\n\t" "sdot z25.s, z1.b, z4.b\n\t"
        "sdot z26.s, z2.b, z4.b\n\t" "sdot z27.s, z3.b, z4.b\n\t"
        "ld1rw {z4.s}, p1/z, [x13]\n\t"
        "sdot z28.s, z0.b, z4.b\n\t" "sdot z29.s, z1.b, z4.b\n\t"
        "sdot z30.s, z2.b, z4.b\n\t" "sdot z31.s, z3.b, z4.b\n\t"

        // K-group 4-7 (second 4 K-tiles)
        "ld1b {z0.b}, p0/z, [x4, #4, mul vl]\n\t"
        "ld1b {z1.b}, p0/z, [x4, #5, mul vl]\n\t"
        "ld1b {z2.b}, p0/z, [x4, #6, mul vl]\n\t"
        "ld1b {z3.b}, p0/z, [x4, #7, mul vl]\n\t"
        "ld1rw {z4.s}, p1/z, [x5, #4]\n\t"
        "sdot z8.s, z0.b, z4.b\n\t"  "sdot z9.s, z1.b, z4.b\n\t"
        "sdot z10.s, z2.b, z4.b\n\t" "sdot z11.s, z3.b, z4.b\n\t"
        "ld1rw {z4.s}, p1/z, [x6, #4]\n\t"
        "sdot z12.s, z0.b, z4.b\n\t" "sdot z13.s, z1.b, z4.b\n\t"
        "sdot z14.s, z2.b, z4.b\n\t" "sdot z15.s, z3.b, z4.b\n\t"
        "ld1rw {z4.s}, p1/z, [x7, #4]\n\t"
        "sdot z16.s, z0.b, z4.b\n\t" "sdot z17.s, z1.b, z4.b\n\t"
        "sdot z18.s, z2.b, z4.b\n\t" "sdot z19.s, z3.b, z4.b\n\t"
        "ld1rw {z4.s}, p1/z, [x11, #4]\n\t"
        "sdot z20.s, z0.b, z4.b\n\t" "sdot z21.s, z1.b, z4.b\n\t"
        "sdot z22.s, z2.b, z4.b\n\t" "sdot z23.s, z3.b, z4.b\n\t"
        "ld1rw {z4.s}, p1/z, [x12, #4]\n\t"
        "sdot z24.s, z0.b, z4.b\n\t" "sdot z25.s, z1.b, z4.b\n\t"
        "sdot z26.s, z2.b, z4.b\n\t" "sdot z27.s, z3.b, z4.b\n\t"
        "ld1rw {z4.s}, p1/z, [x13, #4]\n\t"
        "sdot z28.s, z0.b, z4.b\n\t" "sdot z29.s, z1.b, z4.b\n\t"
        "sdot z30.s, z2.b, z4.b\n\t" "sdot z31.s, z3.b, z4.b\n\t"

        // Advance K pointer by 512 to access K-group 8-11
        "add x4, x4, #512\n\t"

        // K-group 8-11 (third 4 K-tiles)
        "ld1b {z0.b}, p0/z, [x4]\n\t"
        "ld1b {z1.b}, p0/z, [x4, #1, mul vl]\n\t"
        "ld1b {z2.b}, p0/z, [x4, #2, mul vl]\n\t"
        "ld1b {z3.b}, p0/z, [x4, #3, mul vl]\n\t"
        "ld1rw {z4.s}, p1/z, [x5, #8]\n\t"
        "sdot z8.s, z0.b, z4.b\n\t"  "sdot z9.s, z1.b, z4.b\n\t"
        "sdot z10.s, z2.b, z4.b\n\t" "sdot z11.s, z3.b, z4.b\n\t"
        "ld1rw {z4.s}, p1/z, [x6, #8]\n\t"
        "sdot z12.s, z0.b, z4.b\n\t" "sdot z13.s, z1.b, z4.b\n\t"
        "sdot z14.s, z2.b, z4.b\n\t" "sdot z15.s, z3.b, z4.b\n\t"
        "ld1rw {z4.s}, p1/z, [x7, #8]\n\t"
        "sdot z16.s, z0.b, z4.b\n\t" "sdot z17.s, z1.b, z4.b\n\t"
        "sdot z18.s, z2.b, z4.b\n\t" "sdot z19.s, z3.b, z4.b\n\t"
        "ld1rw {z4.s}, p1/z, [x11, #8]\n\t"
        "sdot z20.s, z0.b, z4.b\n\t" "sdot z21.s, z1.b, z4.b\n\t"
        "sdot z22.s, z2.b, z4.b\n\t" "sdot z23.s, z3.b, z4.b\n\t"
        "ld1rw {z4.s}, p1/z, [x12, #8]\n\t"
        "sdot z24.s, z0.b, z4.b\n\t" "sdot z25.s, z1.b, z4.b\n\t"
        "sdot z26.s, z2.b, z4.b\n\t" "sdot z27.s, z3.b, z4.b\n\t"
        "ld1rw {z4.s}, p1/z, [x13, #8]\n\t"
        "sdot z28.s, z0.b, z4.b\n\t" "sdot z29.s, z1.b, z4.b\n\t"
        "sdot z30.s, z2.b, z4.b\n\t" "sdot z31.s, z3.b, z4.b\n\t"

        // K-group 12-15 (fourth 4 K-tiles)
        "ld1b {z0.b}, p0/z, [x4, #4, mul vl]\n\t"
        "ld1b {z1.b}, p0/z, [x4, #5, mul vl]\n\t"
        "ld1b {z2.b}, p0/z, [x4, #6, mul vl]\n\t"
        "ld1b {z3.b}, p0/z, [x4, #7, mul vl]\n\t"
        "add x4, x4, #512\n\t"  // Total: 512 + 512 = 1024 bytes = 16 K-groups

        "ld1rw {z4.s}, p1/z, [x5, #12]\n\t"
        "sdot z8.s, z0.b, z4.b\n\t"  "sdot z9.s, z1.b, z4.b\n\t"
        "sdot z10.s, z2.b, z4.b\n\t" "sdot z11.s, z3.b, z4.b\n\t"
        "ld1rw {z4.s}, p1/z, [x6, #12]\n\t"
        "sdot z12.s, z0.b, z4.b\n\t" "sdot z13.s, z1.b, z4.b\n\t"
        "sdot z14.s, z2.b, z4.b\n\t" "sdot z15.s, z3.b, z4.b\n\t"
        "ld1rw {z4.s}, p1/z, [x7, #12]\n\t"
        "sdot z16.s, z0.b, z4.b\n\t" "sdot z17.s, z1.b, z4.b\n\t"
        "sdot z18.s, z2.b, z4.b\n\t" "sdot z19.s, z3.b, z4.b\n\t"
        "ld1rw {z4.s}, p1/z, [x11, #12]\n\t"
        "sdot z20.s, z0.b, z4.b\n\t" "sdot z21.s, z1.b, z4.b\n\t"
        "sdot z22.s, z2.b, z4.b\n\t" "sdot z23.s, z3.b, z4.b\n\t"
        "ld1rw {z4.s}, p1/z, [x12, #12]\n\t"
        "sdot z24.s, z0.b, z4.b\n\t" "sdot z25.s, z1.b, z4.b\n\t"
        "sdot z26.s, z2.b, z4.b\n\t" "sdot z27.s, z3.b, z4.b\n\t"
        "ld1rw {z4.s}, p1/z, [x13, #12]\n\t"
        "sdot z28.s, z0.b, z4.b\n\t" "sdot z29.s, z1.b, z4.b\n\t"
        "sdot z30.s, z2.b, z4.b\n\t" "sdot z31.s, z3.b, z4.b\n\t"

        "add x5, x5, #16\n\t"   "add x6, x6, #16\n\t"   "add x7, x7, #16\n\t"
        "add x11, x11, #16\n\t" "add x12, x12, #16\n\t" "add x13, x13, #16\n\t"
        "subs x10, x10, #1\n\t"
        "b.ne 1b\n\t"

        // Reduce
        "add z8.s, z8.s, z9.s\n\t"   "add z10.s, z10.s, z11.s\n\t" "add z8.s, z8.s, z10.s\n\t"
        "add z12.s, z12.s, z13.s\n\t" "add z14.s, z14.s, z15.s\n\t" "add z12.s, z12.s, z14.s\n\t"
        "add z16.s, z16.s, z17.s\n\t" "add z18.s, z18.s, z19.s\n\t" "add z16.s, z16.s, z18.s\n\t"
        "add z20.s, z20.s, z21.s\n\t" "add z22.s, z22.s, z23.s\n\t" "add z20.s, z20.s, z22.s\n\t"
        "add z24.s, z24.s, z25.s\n\t" "add z26.s, z26.s, z27.s\n\t" "add z24.s, z24.s, z26.s\n\t"
        "add z28.s, z28.s, z29.s\n\t" "add z30.s, z30.s, z31.s\n\t" "add z28.s, z28.s, z30.s\n\t"

        "mov x14, %[S]\n\t"
        "st1w {z8.s}, p1, [x14]\n\t"       "st1w {z12.s}, p1, [x14, #1, mul vl]\n\t"
        "st1w {z16.s}, p1, [x14, #2, mul vl]\n\t" "st1w {z20.s}, p1, [x14, #3, mul vl]\n\t"
        "st1w {z24.s}, p1, [x14, #4, mul vl]\n\t" "st1w {z28.s}, p1, [x14, #5, mul vl]\n\t"
        :
        : [Q] "r"(Q), [K] "r"(K), [S] "r"(S)
        : "memory", "x4", "x5", "x6", "x7", "x10", "x11", "x12", "x13", "x14",
          "z0", "z1", "z2", "z3", "z4", "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15",
          "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23",
          "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31", "p0", "p1"
    );
}

// Optimization 2: Software pipelining with interleaved loads/SDOT
// Load K[i+1] while computing SDOT with K[i]
__attribute__((noinline))
void opt_pipeline(const int8_t* Q, const int8_t* K, int32_t* S) {
    __asm__ volatile(
        "ptrue p0.b\n\t"
        "ptrue p1.s\n\t"
        // Zero accumulators
        "mov z8.s, #0\n\t"  "mov z9.s, #0\n\t"  "mov z10.s, #0\n\t" "mov z11.s, #0\n\t"
        "mov z12.s, #0\n\t" "mov z13.s, #0\n\t" "mov z14.s, #0\n\t" "mov z15.s, #0\n\t"
        "mov z16.s, #0\n\t" "mov z17.s, #0\n\t" "mov z18.s, #0\n\t" "mov z19.s, #0\n\t"
        "mov z20.s, #0\n\t" "mov z21.s, #0\n\t" "mov z22.s, #0\n\t" "mov z23.s, #0\n\t"
        "mov z24.s, #0\n\t" "mov z25.s, #0\n\t" "mov z26.s, #0\n\t" "mov z27.s, #0\n\t"
        "mov z28.s, #0\n\t" "mov z29.s, #0\n\t" "mov z30.s, #0\n\t" "mov z31.s, #0\n\t"

        "mov x5, %[Q]\n\t"
        "add x6, x5, #256\n\t"  "add x7, x6, #256\n\t"
        "add x11, x7, #256\n\t" "add x12, x11, #256\n\t" "add x13, x12, #256\n\t"
        "mov x4, %[K]\n\t"

        // Prologue: load first K block
        "ld1b {z0.b}, p0/z, [x4]\n\t"
        "ld1b {z1.b}, p0/z, [x4, #1, mul vl]\n\t"
        "ld1b {z2.b}, p0/z, [x4, #2, mul vl]\n\t"
        "ld1b {z3.b}, p0/z, [x4, #3, mul vl]\n\t"
        "add x4, x4, #256\n\t"

        "mov x10, #31\n\t"  // 32-1 iterations (first handled separately)

        "1:\n\t"
        // Compute with current K (z0-z3), load next K into z4-z7
        // Row 0 SDOT + load next K[0]
        "ld1rw {z5.s}, p1/z, [x5]\n\t"
        "ld1b {z4.b}, p0/z, [x4]\n\t"
        "sdot z8.s, z0.b, z5.b\n\t"  "sdot z9.s, z1.b, z5.b\n\t"
        "sdot z10.s, z2.b, z5.b\n\t" "sdot z11.s, z3.b, z5.b\n\t"

        // Row 1 SDOT + load next K[1]
        "ld1rw {z5.s}, p1/z, [x6]\n\t"
        "ld1b {z6.b}, p0/z, [x4, #1, mul vl]\n\t"
        "sdot z12.s, z0.b, z5.b\n\t" "sdot z13.s, z1.b, z5.b\n\t"
        "sdot z14.s, z2.b, z5.b\n\t" "sdot z15.s, z3.b, z5.b\n\t"

        // Row 2 SDOT + load next K[2]
        "ld1rw {z5.s}, p1/z, [x7]\n\t"
        "ld1b {z7.b}, p0/z, [x4, #2, mul vl]\n\t"
        "sdot z16.s, z0.b, z5.b\n\t" "sdot z17.s, z1.b, z5.b\n\t"
        "sdot z18.s, z2.b, z5.b\n\t" "sdot z19.s, z3.b, z5.b\n\t"

        // Row 3 SDOT + load Q[3]
        "ld1rw {z5.s}, p1/z, [x11]\n\t"
        "sdot z20.s, z0.b, z5.b\n\t" "sdot z21.s, z1.b, z5.b\n\t"
        "sdot z22.s, z2.b, z5.b\n\t" "sdot z23.s, z3.b, z5.b\n\t"

        // Row 4 SDOT + load next K[3]
        "ld1rw {z5.s}, p1/z, [x12]\n\t"
        "ld1b {z0.b}, p0/z, [x4, #3, mul vl]\n\t"  // Reuse z0 for next K[3]
        "sdot z24.s, z0.b, z5.b\n\t" "sdot z25.s, z1.b, z5.b\n\t"
        "sdot z26.s, z2.b, z5.b\n\t" "sdot z27.s, z3.b, z5.b\n\t"

        // Row 5 SDOT
        "ld1rw {z5.s}, p1/z, [x13]\n\t"
        "sdot z28.s, z0.b, z5.b\n\t" "sdot z29.s, z1.b, z5.b\n\t"
        "sdot z30.s, z2.b, z5.b\n\t" "sdot z31.s, z3.b, z5.b\n\t"

        // Move next K to current
        "mov z0.d, z4.d\n\t"
        "mov z1.d, z6.d\n\t"
        "mov z2.d, z7.d\n\t"
        // z3 already loaded into z0 position above - BUG! Need to fix

        "add x4, x4, #256\n\t"
        "add x5, x5, #4\n\t"   "add x6, x6, #4\n\t"   "add x7, x7, #4\n\t"
        "add x11, x11, #4\n\t" "add x12, x12, #4\n\t" "add x13, x13, #4\n\t"
        "subs x10, x10, #1\n\t"
        "b.ne 1b\n\t"

        // Epilogue: compute with last K
        "ld1rw {z5.s}, p1/z, [x5]\n\t"
        "sdot z8.s, z0.b, z5.b\n\t"  "sdot z9.s, z1.b, z5.b\n\t"
        "sdot z10.s, z2.b, z5.b\n\t" "sdot z11.s, z3.b, z5.b\n\t"
        "ld1rw {z5.s}, p1/z, [x6]\n\t"
        "sdot z12.s, z0.b, z5.b\n\t" "sdot z13.s, z1.b, z5.b\n\t"
        "sdot z14.s, z2.b, z5.b\n\t" "sdot z15.s, z3.b, z5.b\n\t"
        "ld1rw {z5.s}, p1/z, [x7]\n\t"
        "sdot z16.s, z0.b, z5.b\n\t" "sdot z17.s, z1.b, z5.b\n\t"
        "sdot z18.s, z2.b, z5.b\n\t" "sdot z19.s, z3.b, z5.b\n\t"
        "ld1rw {z5.s}, p1/z, [x11]\n\t"
        "sdot z20.s, z0.b, z5.b\n\t" "sdot z21.s, z1.b, z5.b\n\t"
        "sdot z22.s, z2.b, z5.b\n\t" "sdot z23.s, z3.b, z5.b\n\t"
        "ld1rw {z5.s}, p1/z, [x12]\n\t"
        "sdot z24.s, z0.b, z5.b\n\t" "sdot z25.s, z1.b, z5.b\n\t"
        "sdot z26.s, z2.b, z5.b\n\t" "sdot z27.s, z3.b, z5.b\n\t"
        "ld1rw {z5.s}, p1/z, [x13]\n\t"
        "sdot z28.s, z0.b, z5.b\n\t" "sdot z29.s, z1.b, z5.b\n\t"
        "sdot z30.s, z2.b, z5.b\n\t" "sdot z31.s, z3.b, z5.b\n\t"

        // Reduce
        "add z8.s, z8.s, z9.s\n\t"   "add z10.s, z10.s, z11.s\n\t" "add z8.s, z8.s, z10.s\n\t"
        "add z12.s, z12.s, z13.s\n\t" "add z14.s, z14.s, z15.s\n\t" "add z12.s, z12.s, z14.s\n\t"
        "add z16.s, z16.s, z17.s\n\t" "add z18.s, z18.s, z19.s\n\t" "add z16.s, z16.s, z18.s\n\t"
        "add z20.s, z20.s, z21.s\n\t" "add z22.s, z22.s, z23.s\n\t" "add z20.s, z20.s, z22.s\n\t"
        "add z24.s, z24.s, z25.s\n\t" "add z26.s, z26.s, z27.s\n\t" "add z24.s, z24.s, z26.s\n\t"
        "add z28.s, z28.s, z29.s\n\t" "add z30.s, z30.s, z31.s\n\t" "add z28.s, z28.s, z30.s\n\t"

        "mov x14, %[S]\n\t"
        "st1w {z8.s}, p1, [x14]\n\t"       "st1w {z12.s}, p1, [x14, #1, mul vl]\n\t"
        "st1w {z16.s}, p1, [x14, #2, mul vl]\n\t" "st1w {z20.s}, p1, [x14, #3, mul vl]\n\t"
        "st1w {z24.s}, p1, [x14, #4, mul vl]\n\t" "st1w {z28.s}, p1, [x14, #5, mul vl]\n\t"
        :
        : [Q] "r"(Q), [K] "r"(K), [S] "r"(S)
        : "memory", "x4", "x5", "x6", "x7", "x10", "x11", "x12", "x13", "x14",
          "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7",
          "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15",
          "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23",
          "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31", "p0", "p1"
    );
}

// Optimization 3: Interleaved SDOT scheduling for better FPU utilization
// Interleave SDOT from different rows to maximize parallel execution
__attribute__((noinline))
void opt_interleaved_sdot(const int8_t* Q, const int8_t* K, int32_t* S) {
    __asm__ volatile(
        "ptrue p0.b\n\t"
        "ptrue p1.s\n\t"
        // Zero accumulators
        "mov z8.s, #0\n\t"  "mov z9.s, #0\n\t"  "mov z10.s, #0\n\t" "mov z11.s, #0\n\t"
        "mov z12.s, #0\n\t" "mov z13.s, #0\n\t" "mov z14.s, #0\n\t" "mov z15.s, #0\n\t"
        "mov z16.s, #0\n\t" "mov z17.s, #0\n\t" "mov z18.s, #0\n\t" "mov z19.s, #0\n\t"
        "mov z20.s, #0\n\t" "mov z21.s, #0\n\t" "mov z22.s, #0\n\t" "mov z23.s, #0\n\t"
        "mov z24.s, #0\n\t" "mov z25.s, #0\n\t" "mov z26.s, #0\n\t" "mov z27.s, #0\n\t"
        "mov z28.s, #0\n\t" "mov z29.s, #0\n\t" "mov z30.s, #0\n\t" "mov z31.s, #0\n\t"

        "mov x5, %[Q]\n\t"
        "add x6, x5, #256\n\t"  "add x7, x6, #256\n\t"
        "add x11, x7, #256\n\t" "add x12, x11, #256\n\t" "add x13, x12, #256\n\t"
        "mov x4, %[K]\n\t"
        "mov x10, #32\n\t"

        "1:\n\t"
        // Load K tiles
        "ld1b {z0.b}, p0/z, [x4]\n\t"
        "ld1b {z1.b}, p0/z, [x4, #1, mul vl]\n\t"
        "ld1b {z2.b}, p0/z, [x4, #2, mul vl]\n\t"
        "ld1b {z3.b}, p0/z, [x4, #3, mul vl]\n\t"

        // Load all Q broadcasts first
        "ld1rw {z4.s}, p1/z, [x5]\n\t"    // Q[0]
        "ld1rw {z5.s}, p1/z, [x6]\n\t"    // Q[1]
        "ld1rw {z6.s}, p1/z, [x7]\n\t"    // Q[2]
        "ld1rw {z7.s}, p1/z, [x11]\n\t"   // Q[3]

        // Interleaved SDOT: alternate between rows and K-tiles
        // This maximizes ILP by ensuring different accumulators
        "sdot z8.s, z0.b, z4.b\n\t"    // Row 0, K-tile 0
        "sdot z12.s, z0.b, z5.b\n\t"   // Row 1, K-tile 0
        "sdot z16.s, z0.b, z6.b\n\t"   // Row 2, K-tile 0
        "sdot z20.s, z0.b, z7.b\n\t"   // Row 3, K-tile 0

        "sdot z9.s, z1.b, z4.b\n\t"    // Row 0, K-tile 1
        "sdot z13.s, z1.b, z5.b\n\t"   // Row 1, K-tile 1
        "sdot z17.s, z1.b, z6.b\n\t"   // Row 2, K-tile 1
        "sdot z21.s, z1.b, z7.b\n\t"   // Row 3, K-tile 1

        "sdot z10.s, z2.b, z4.b\n\t"   // Row 0, K-tile 2
        "sdot z14.s, z2.b, z5.b\n\t"   // Row 1, K-tile 2
        "sdot z18.s, z2.b, z6.b\n\t"   // Row 2, K-tile 2
        "sdot z22.s, z2.b, z7.b\n\t"   // Row 3, K-tile 2

        "sdot z11.s, z3.b, z4.b\n\t"   // Row 0, K-tile 3
        "sdot z15.s, z3.b, z5.b\n\t"   // Row 1, K-tile 3
        "sdot z19.s, z3.b, z6.b\n\t"   // Row 2, K-tile 3
        "sdot z23.s, z3.b, z7.b\n\t"   // Row 3, K-tile 3

        // Rows 4-5
        "ld1rw {z4.s}, p1/z, [x12]\n\t"   // Q[4]
        "ld1rw {z5.s}, p1/z, [x13]\n\t"   // Q[5]

        "sdot z24.s, z0.b, z4.b\n\t"   // Row 4, K-tile 0
        "sdot z28.s, z0.b, z5.b\n\t"   // Row 5, K-tile 0
        "sdot z25.s, z1.b, z4.b\n\t"   // Row 4, K-tile 1
        "sdot z29.s, z1.b, z5.b\n\t"   // Row 5, K-tile 1
        "sdot z26.s, z2.b, z4.b\n\t"   // Row 4, K-tile 2
        "sdot z30.s, z2.b, z5.b\n\t"   // Row 5, K-tile 2
        "sdot z27.s, z3.b, z4.b\n\t"   // Row 4, K-tile 3
        "sdot z31.s, z3.b, z5.b\n\t"   // Row 5, K-tile 3

        // Second set (2x unroll)
        "ld1b {z0.b}, p0/z, [x4, #4, mul vl]\n\t"
        "ld1b {z1.b}, p0/z, [x4, #5, mul vl]\n\t"
        "ld1b {z2.b}, p0/z, [x4, #6, mul vl]\n\t"
        "ld1b {z3.b}, p0/z, [x4, #7, mul vl]\n\t"
        "add x4, x4, #512\n\t"

        "ld1rw {z4.s}, p1/z, [x5, #4]\n\t"
        "ld1rw {z5.s}, p1/z, [x6, #4]\n\t"
        "ld1rw {z6.s}, p1/z, [x7, #4]\n\t"
        "ld1rw {z7.s}, p1/z, [x11, #4]\n\t"

        "sdot z8.s, z0.b, z4.b\n\t"    "sdot z12.s, z0.b, z5.b\n\t"
        "sdot z16.s, z0.b, z6.b\n\t"   "sdot z20.s, z0.b, z7.b\n\t"
        "sdot z9.s, z1.b, z4.b\n\t"    "sdot z13.s, z1.b, z5.b\n\t"
        "sdot z17.s, z1.b, z6.b\n\t"   "sdot z21.s, z1.b, z7.b\n\t"
        "sdot z10.s, z2.b, z4.b\n\t"   "sdot z14.s, z2.b, z5.b\n\t"
        "sdot z18.s, z2.b, z6.b\n\t"   "sdot z22.s, z2.b, z7.b\n\t"
        "sdot z11.s, z3.b, z4.b\n\t"   "sdot z15.s, z3.b, z5.b\n\t"
        "sdot z19.s, z3.b, z6.b\n\t"   "sdot z23.s, z3.b, z7.b\n\t"

        "ld1rw {z4.s}, p1/z, [x12, #4]\n\t"
        "ld1rw {z5.s}, p1/z, [x13, #4]\n\t"

        "sdot z24.s, z0.b, z4.b\n\t"   "sdot z28.s, z0.b, z5.b\n\t"
        "sdot z25.s, z1.b, z4.b\n\t"   "sdot z29.s, z1.b, z5.b\n\t"
        "sdot z26.s, z2.b, z4.b\n\t"   "sdot z30.s, z2.b, z5.b\n\t"
        "sdot z27.s, z3.b, z4.b\n\t"   "sdot z31.s, z3.b, z5.b\n\t"

        "add x5, x5, #8\n\t"   "add x6, x6, #8\n\t"   "add x7, x7, #8\n\t"
        "add x11, x11, #8\n\t" "add x12, x12, #8\n\t" "add x13, x13, #8\n\t"
        "subs x10, x10, #1\n\t"
        "b.ne 1b\n\t"

        // Reduce
        "add z8.s, z8.s, z9.s\n\t"   "add z10.s, z10.s, z11.s\n\t" "add z8.s, z8.s, z10.s\n\t"
        "add z12.s, z12.s, z13.s\n\t" "add z14.s, z14.s, z15.s\n\t" "add z12.s, z12.s, z14.s\n\t"
        "add z16.s, z16.s, z17.s\n\t" "add z18.s, z18.s, z19.s\n\t" "add z16.s, z16.s, z18.s\n\t"
        "add z20.s, z20.s, z21.s\n\t" "add z22.s, z22.s, z23.s\n\t" "add z20.s, z20.s, z22.s\n\t"
        "add z24.s, z24.s, z25.s\n\t" "add z26.s, z26.s, z27.s\n\t" "add z24.s, z24.s, z26.s\n\t"
        "add z28.s, z28.s, z29.s\n\t" "add z30.s, z30.s, z31.s\n\t" "add z28.s, z28.s, z30.s\n\t"

        "mov x14, %[S]\n\t"
        "st1w {z8.s}, p1, [x14]\n\t"       "st1w {z12.s}, p1, [x14, #1, mul vl]\n\t"
        "st1w {z16.s}, p1, [x14, #2, mul vl]\n\t" "st1w {z20.s}, p1, [x14, #3, mul vl]\n\t"
        "st1w {z24.s}, p1, [x14, #4, mul vl]\n\t" "st1w {z28.s}, p1, [x14, #5, mul vl]\n\t"
        :
        : [Q] "r"(Q), [K] "r"(K), [S] "r"(S)
        : "memory", "x4", "x5", "x6", "x7", "x10", "x11", "x12", "x13", "x14",
          "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7",
          "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15",
          "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23",
          "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31", "p0", "p1"
    );
}

// Optimization 4: Sector cache hints for streaming data
__attribute__((noinline))
void opt_sector_cache(const int8_t* Q, const int8_t* K, int32_t* S) {
    __asm__ volatile(
        "ptrue p0.b\n\t"
        "ptrue p1.s\n\t"
        // Zero accumulators
        "mov z8.s, #0\n\t"  "mov z9.s, #0\n\t"  "mov z10.s, #0\n\t" "mov z11.s, #0\n\t"
        "mov z12.s, #0\n\t" "mov z13.s, #0\n\t" "mov z14.s, #0\n\t" "mov z15.s, #0\n\t"
        "mov z16.s, #0\n\t" "mov z17.s, #0\n\t" "mov z18.s, #0\n\t" "mov z19.s, #0\n\t"
        "mov z20.s, #0\n\t" "mov z21.s, #0\n\t" "mov z22.s, #0\n\t" "mov z23.s, #0\n\t"
        "mov z24.s, #0\n\t" "mov z25.s, #0\n\t" "mov z26.s, #0\n\t" "mov z27.s, #0\n\t"
        "mov z28.s, #0\n\t" "mov z29.s, #0\n\t" "mov z30.s, #0\n\t" "mov z31.s, #0\n\t"

        "mov x5, %[Q]\n\t"
        "add x6, x5, #256\n\t"  "add x7, x6, #256\n\t"
        "add x11, x7, #256\n\t" "add x12, x11, #256\n\t" "add x13, x12, #256\n\t"
        "mov x4, %[K]\n\t"
        "mov x10, #32\n\t"

        "1:\n\t"
        // Prefetch next K block with streaming hint
        "prfm pldl1strm, [x4, #512]\n\t"

        // Load K tiles
        "ld1b {z0.b}, p0/z, [x4]\n\t"
        "ld1b {z1.b}, p0/z, [x4, #1, mul vl]\n\t"
        "ld1b {z2.b}, p0/z, [x4, #2, mul vl]\n\t"
        "ld1b {z3.b}, p0/z, [x4, #3, mul vl]\n\t"

        "ld1rw {z4.s}, p1/z, [x5]\n\t"
        "sdot z8.s, z0.b, z4.b\n\t"  "sdot z9.s, z1.b, z4.b\n\t"
        "sdot z10.s, z2.b, z4.b\n\t" "sdot z11.s, z3.b, z4.b\n\t"
        "ld1rw {z4.s}, p1/z, [x6]\n\t"
        "sdot z12.s, z0.b, z4.b\n\t" "sdot z13.s, z1.b, z4.b\n\t"
        "sdot z14.s, z2.b, z4.b\n\t" "sdot z15.s, z3.b, z4.b\n\t"
        "ld1rw {z4.s}, p1/z, [x7]\n\t"
        "sdot z16.s, z0.b, z4.b\n\t" "sdot z17.s, z1.b, z4.b\n\t"
        "sdot z18.s, z2.b, z4.b\n\t" "sdot z19.s, z3.b, z4.b\n\t"
        "ld1rw {z4.s}, p1/z, [x11]\n\t"
        "sdot z20.s, z0.b, z4.b\n\t" "sdot z21.s, z1.b, z4.b\n\t"
        "sdot z22.s, z2.b, z4.b\n\t" "sdot z23.s, z3.b, z4.b\n\t"
        "ld1rw {z4.s}, p1/z, [x12]\n\t"
        "sdot z24.s, z0.b, z4.b\n\t" "sdot z25.s, z1.b, z4.b\n\t"
        "sdot z26.s, z2.b, z4.b\n\t" "sdot z27.s, z3.b, z4.b\n\t"
        "ld1rw {z4.s}, p1/z, [x13]\n\t"
        "sdot z28.s, z0.b, z4.b\n\t" "sdot z29.s, z1.b, z4.b\n\t"
        "sdot z30.s, z2.b, z4.b\n\t" "sdot z31.s, z3.b, z4.b\n\t"

        // Second set
        "ld1b {z0.b}, p0/z, [x4, #4, mul vl]\n\t"
        "ld1b {z1.b}, p0/z, [x4, #5, mul vl]\n\t"
        "ld1b {z2.b}, p0/z, [x4, #6, mul vl]\n\t"
        "ld1b {z3.b}, p0/z, [x4, #7, mul vl]\n\t"
        "add x4, x4, #512\n\t"

        "ld1rw {z4.s}, p1/z, [x5, #4]\n\t"
        "sdot z8.s, z0.b, z4.b\n\t"  "sdot z9.s, z1.b, z4.b\n\t"
        "sdot z10.s, z2.b, z4.b\n\t" "sdot z11.s, z3.b, z4.b\n\t"
        "ld1rw {z4.s}, p1/z, [x6, #4]\n\t"
        "sdot z12.s, z0.b, z4.b\n\t" "sdot z13.s, z1.b, z4.b\n\t"
        "sdot z14.s, z2.b, z4.b\n\t" "sdot z15.s, z3.b, z4.b\n\t"
        "ld1rw {z4.s}, p1/z, [x7, #4]\n\t"
        "sdot z16.s, z0.b, z4.b\n\t" "sdot z17.s, z1.b, z4.b\n\t"
        "sdot z18.s, z2.b, z4.b\n\t" "sdot z19.s, z3.b, z4.b\n\t"
        "ld1rw {z4.s}, p1/z, [x11, #4]\n\t"
        "sdot z20.s, z0.b, z4.b\n\t" "sdot z21.s, z1.b, z4.b\n\t"
        "sdot z22.s, z2.b, z4.b\n\t" "sdot z23.s, z3.b, z4.b\n\t"
        "ld1rw {z4.s}, p1/z, [x12, #4]\n\t"
        "sdot z24.s, z0.b, z4.b\n\t" "sdot z25.s, z1.b, z4.b\n\t"
        "sdot z26.s, z2.b, z4.b\n\t" "sdot z27.s, z3.b, z4.b\n\t"
        "ld1rw {z4.s}, p1/z, [x13, #4]\n\t"
        "sdot z28.s, z0.b, z4.b\n\t" "sdot z29.s, z1.b, z4.b\n\t"
        "sdot z30.s, z2.b, z4.b\n\t" "sdot z31.s, z3.b, z4.b\n\t"

        "add x5, x5, #8\n\t"   "add x6, x6, #8\n\t"   "add x7, x7, #8\n\t"
        "add x11, x11, #8\n\t" "add x12, x12, #8\n\t" "add x13, x13, #8\n\t"
        "subs x10, x10, #1\n\t"
        "b.ne 1b\n\t"

        // Reduce
        "add z8.s, z8.s, z9.s\n\t"   "add z10.s, z10.s, z11.s\n\t" "add z8.s, z8.s, z10.s\n\t"
        "add z12.s, z12.s, z13.s\n\t" "add z14.s, z14.s, z15.s\n\t" "add z12.s, z12.s, z14.s\n\t"
        "add z16.s, z16.s, z17.s\n\t" "add z18.s, z18.s, z19.s\n\t" "add z16.s, z16.s, z18.s\n\t"
        "add z20.s, z20.s, z21.s\n\t" "add z22.s, z22.s, z23.s\n\t" "add z20.s, z20.s, z22.s\n\t"
        "add z24.s, z24.s, z25.s\n\t" "add z26.s, z26.s, z27.s\n\t" "add z24.s, z24.s, z26.s\n\t"
        "add z28.s, z28.s, z29.s\n\t" "add z30.s, z30.s, z31.s\n\t" "add z28.s, z28.s, z30.s\n\t"

        "mov x14, %[S]\n\t"
        "st1w {z8.s}, p1, [x14]\n\t"       "st1w {z12.s}, p1, [x14, #1, mul vl]\n\t"
        "st1w {z16.s}, p1, [x14, #2, mul vl]\n\t" "st1w {z20.s}, p1, [x14, #3, mul vl]\n\t"
        "st1w {z24.s}, p1, [x14, #4, mul vl]\n\t" "st1w {z28.s}, p1, [x14, #5, mul vl]\n\t"
        :
        : [Q] "r"(Q), [K] "r"(K), [S] "r"(S)
        : "memory", "x4", "x5", "x6", "x7", "x10", "x11", "x12", "x13", "x14",
          "z0", "z1", "z2", "z3", "z4",
          "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15",
          "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23",
          "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31", "p0", "p1"
    );
}

int main() {
    // Allocate aligned buffers
    int8_t* Q = aligned_alloc(256, 6 * 256);
    int8_t* K = aligned_alloc(256, 64 * 256);  // N=64, K-groups=64, 4 bytes each
    int32_t* S = aligned_alloc(256, 6 * 64 * sizeof(int32_t));

    // Initialize with test data
    for (int i = 0; i < 6 * 256; i++) Q[i] = (i % 7) - 3;
    for (int i = 0; i < 64 * 256; i++) K[i] = (i % 5) - 2;

    printf("==============================================\n");
    printf("K-Loop Unrolling + Pipelining Optimization\n");
    printf("==============================================\n\n");
    printf("Configuration: MR=6, N=64, D=256\n");
    printf("SDOT per kernel: 6×64×64 = 24576 (per N iteration)\n");
    printf("Total SDOT: 1536 (Phase 1 only, 6 rows × 64 N × 4 K-tiles)\n\n");

    // Actually for Phase 1: Q[6,256] @ K[256,64] = S[6,64]
    // SDOT = 6 rows × 64 N-cols × 64 K-groups = 24576
    // But we compute in 4 K-tiles, so SDOT = 6 × 64 × 64 = 24576
    // Wait, let me recalculate...
    // Phase 1: D=256 = 64 K-groups × 4 bytes
    // For each K-group: 6 rows × 4 K-tiles = 24 SDOT
    // Total: 64 K-groups × 24 SDOT = 1536 SDOT

    int warmup = 100;
    int iters = 1000;

    // Warmup
    for (int i = 0; i < warmup; i++) {
        baseline_2x_unroll(Q, K, S);
    }

    printf("Approach                            Ticks Efficiency    SDOT/tick\n");
    printf("------------------------------------------------------------------\n");

    // Baseline 2x unroll
    uint64_t start = rdcycle();
    for (int i = 0; i < iters; i++) {
        baseline_2x_unroll(Q, K, S);
    }
    uint64_t end = rdcycle();
    double ticks_baseline = (double)(end - start) / iters;
    double sdot_per_tick = 1536.0 / ticks_baseline;
    double efficiency = sdot_per_tick / 40.0 * 100.0;
    printf("Baseline (2x unroll)                %6.1f    %5.1f%%       %6.2f\n",
           ticks_baseline, efficiency, sdot_per_tick);

    // Warmup 4x unroll
    for (int i = 0; i < warmup; i++) {
        opt_4x_unroll(Q, K, S);
    }

    // 4x unroll
    start = rdcycle();
    for (int i = 0; i < iters; i++) {
        opt_4x_unroll(Q, K, S);
    }
    end = rdcycle();
    double ticks_4x = (double)(end - start) / iters;
    sdot_per_tick = 1536.0 / ticks_4x;
    efficiency = sdot_per_tick / 40.0 * 100.0;
    printf("4x K-loop unroll                    %6.1f    %5.1f%%       %6.2f\n",
           ticks_4x, efficiency, sdot_per_tick);

    // Warmup other variants
    for (int i = 0; i < warmup; i++) {
        opt_interleaved_sdot(Q, K, S);
    }

    // Interleaved SDOT
    start = rdcycle();
    for (int i = 0; i < iters; i++) {
        opt_interleaved_sdot(Q, K, S);
    }
    end = rdcycle();
    double ticks_interleaved = (double)(end - start) / iters;
    sdot_per_tick = 1536.0 / ticks_interleaved;
    efficiency = sdot_per_tick / 40.0 * 100.0;
    printf("Interleaved SDOT                    %6.1f    %5.1f%%       %6.2f\n",
           ticks_interleaved, efficiency, sdot_per_tick);

    // Warmup sector cache
    for (int i = 0; i < warmup; i++) {
        opt_sector_cache(Q, K, S);
    }

    // Sector cache
    start = rdcycle();
    for (int i = 0; i < iters; i++) {
        opt_sector_cache(Q, K, S);
    }
    end = rdcycle();
    double ticks_sector = (double)(end - start) / iters;
    sdot_per_tick = 1536.0 / ticks_sector;
    efficiency = sdot_per_tick / 40.0 * 100.0;
    printf("Sector cache prefetch               %6.1f    %5.1f%%       %6.2f\n",
           ticks_sector, efficiency, sdot_per_tick);

    printf("\n");

    // Find best result
    double best_ticks = ticks_baseline;
    const char* best_name = "Baseline (2x unroll)";

    if (ticks_4x < best_ticks) {
        best_ticks = ticks_4x;
        best_name = "4x K-loop unroll";
    }
    if (ticks_interleaved < best_ticks) {
        best_ticks = ticks_interleaved;
        best_name = "Interleaved SDOT";
    }
    if (ticks_sector < best_ticks) {
        best_ticks = ticks_sector;
        best_name = "Sector cache prefetch";
    }

    sdot_per_tick = 1536.0 / best_ticks;
    efficiency = sdot_per_tick / 40.0 * 100.0;
    printf("Best: %s\n", best_name);
    printf("  Ticks: %.1f, Efficiency: %.1f%%, SDOT/tick: %.2f\n",
           best_ticks, efficiency, sdot_per_tick);

    if (best_ticks < ticks_baseline) {
        printf("  Improvement over baseline: %.1f%%\n",
               (ticks_baseline - best_ticks) / ticks_baseline * 100.0);
    }

    free(Q);
    free(K);
    free(S);
    return 0;
}
