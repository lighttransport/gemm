/*
 * Software pipelining for D=256 fused kernel
 *
 * Key changes:
 * 1. Use z4, z5, z6, z7 for Q loads (not just z4) - break RAW hazard
 * 2. Software pipeline: load next K while computing current
 *
 * Register allocation:
 *   z0-z3: K data (current)
 *   z4-z7: Q data (6 rows, reuse z4/z5 for rows 5/6)
 *   z8-z31: Accumulators (24 registers)
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

// Original: single z4 for all Q loads
__attribute__((noinline))
void fused_orig(const int8_t* Q, const int8_t* K, const int8_t* V, int32_t* O) {
    int32_t S[6 * 64] __attribute__((aligned(64)));

    __asm__ volatile(
        "ptrue p0.b\n\t"
        "ptrue p1.s\n\t"
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
        // Load K
        "ld1b {z0.b}, p0/z, [x4]\n\t"
        "ld1b {z1.b}, p0/z, [x4, #1, mul vl]\n\t"
        "ld1b {z2.b}, p0/z, [x4, #2, mul vl]\n\t"
        "ld1b {z3.b}, p0/z, [x4, #3, mul vl]\n\t"

        // All Q loads use z4
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

        // Unroll 1
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

    // Phase 2 (simplified - same as orig)
    __asm__ volatile(
        "ptrue p0.b\n\t"
        "ptrue p1.s\n\t"
        "mov x15, #4\n\t"
        "mov x16, %[V]\n\t"
        "mov x17, %[O]\n\t"
        "2:\n\t"
        "mov z8.s, #0\n\t"  "mov z9.s, #0\n\t"  "mov z10.s, #0\n\t" "mov z11.s, #0\n\t"
        "mov z12.s, #0\n\t" "mov z13.s, #0\n\t" "mov z14.s, #0\n\t" "mov z15.s, #0\n\t"
        "mov z16.s, #0\n\t" "mov z17.s, #0\n\t" "mov z18.s, #0\n\t" "mov z19.s, #0\n\t"
        "mov z20.s, #0\n\t" "mov z21.s, #0\n\t" "mov z22.s, #0\n\t" "mov z23.s, #0\n\t"
        "mov z24.s, #0\n\t" "mov z25.s, #0\n\t" "mov z26.s, #0\n\t" "mov z27.s, #0\n\t"
        "mov z28.s, #0\n\t" "mov z29.s, #0\n\t" "mov z30.s, #0\n\t" "mov z31.s, #0\n\t"
        "mov x5, %[S]\n\t"
        "add x6, x5, #256\n\t"  "add x7, x5, #512\n\t"
        "add x11, x5, #768\n\t" "add x12, x5, #1024\n\t" "add x13, x5, #1280\n\t"
        "mov x4, x16\n\t"
        "mov x10, #8\n\t"
        "3:\n\t"
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
        "add x4, x4, #1024\n\t"
        "ld1b {z0.b}, p0/z, [x4]\n\t"
        "ld1b {z1.b}, p0/z, [x4, #1, mul vl]\n\t"
        "ld1b {z2.b}, p0/z, [x4, #2, mul vl]\n\t"
        "ld1b {z3.b}, p0/z, [x4, #3, mul vl]\n\t"
        "add x4, x4, #1024\n\t"
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
        "b.ne 3b\n\t"
        "add z8.s, z8.s, z9.s\n\t"   "add z10.s, z10.s, z11.s\n\t" "add z8.s, z8.s, z10.s\n\t"
        "add z12.s, z12.s, z13.s\n\t" "add z14.s, z14.s, z15.s\n\t" "add z12.s, z12.s, z14.s\n\t"
        "add z16.s, z16.s, z17.s\n\t" "add z18.s, z18.s, z19.s\n\t" "add z16.s, z16.s, z18.s\n\t"
        "add z20.s, z20.s, z21.s\n\t" "add z22.s, z22.s, z23.s\n\t" "add z20.s, z20.s, z22.s\n\t"
        "add z24.s, z24.s, z25.s\n\t" "add z26.s, z26.s, z27.s\n\t" "add z24.s, z24.s, z26.s\n\t"
        "add z28.s, z28.s, z29.s\n\t" "add z30.s, z30.s, z31.s\n\t" "add z28.s, z28.s, z30.s\n\t"
        "st1w {z8.s}, p1, [x17]\n\t"  "add x17, x17, #1024\n\t"
        "st1w {z12.s}, p1, [x17]\n\t" "add x17, x17, #1024\n\t"
        "st1w {z16.s}, p1, [x17]\n\t" "add x17, x17, #1024\n\t"
        "st1w {z20.s}, p1, [x17]\n\t" "add x17, x17, #1024\n\t"
        "st1w {z24.s}, p1, [x17]\n\t" "add x17, x17, #1024\n\t"
        "st1w {z28.s}, p1, [x17]\n\t"
        "sub x17, x17, #4096\n\t" "sub x17, x17, #1024\n\t"
        "add x17, x17, #64\n\t"
        "add x16, x16, #64\n\t"
        "subs x15, x15, #1\n\t"
        "b.ne 2b\n\t"
        :
        : [S] "r"(S), [V] "r"(V), [O] "r"(O)
        : "memory", "x4", "x5", "x6", "x7", "x10", "x11", "x12", "x13",
          "x15", "x16", "x17",
          "z0", "z1", "z2", "z3", "z4", "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15",
          "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23",
          "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31", "p0", "p1"
    );
}

// Software pipelining: load next K while computing current
// Also use z4-z7 for different Q rows to break dependency
__attribute__((noinline))
void fused_swp(const int8_t* Q, const int8_t* K, const int8_t* V, int32_t* O) {
    int32_t S[6 * 64] __attribute__((aligned(64)));

    // Phase 1: software pipelined with multi-Q-reg
    __asm__ volatile(
        "ptrue p0.b\n\t"
        "ptrue p1.s\n\t"
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

        // Prologue: load first K + all Q for first d_group
        "ld1b {z0.b}, p0/z, [x4]\n\t"
        "ld1b {z1.b}, p0/z, [x4, #1, mul vl]\n\t"
        "ld1b {z2.b}, p0/z, [x4, #2, mul vl]\n\t"
        "ld1b {z3.b}, p0/z, [x4, #3, mul vl]\n\t"
        "ld1rw {z4.s}, p1/z, [x5]\n\t"      // Q row 0
        "ld1rw {z5.s}, p1/z, [x6]\n\t"      // Q row 1
        "ld1rw {z6.s}, p1/z, [x7]\n\t"      // Q row 2
        "ld1rw {z7.s}, p1/z, [x11]\n\t"     // Q row 3

        "1:\n\t"
        // Compute with z4-z7 (rows 0-3), load remaining Q and next K in parallel
        "sdot z8.s, z0.b, z4.b\n\t"
        "sdot z9.s, z1.b, z4.b\n\t"
        "sdot z12.s, z0.b, z5.b\n\t"      // Start row 1 SDOTs
        "sdot z13.s, z1.b, z5.b\n\t"
        "ld1rw {z4.s}, p1/z, [x12]\n\t"   // Load Q row 4 into z4 (done with row 0)
        "sdot z10.s, z2.b, z4.b\n\t"      // oops - this would be wrong - z4 now has row 4
        // Actually this approach has a problem - we overwrite z4 too soon

        // Let me try a different approach: just batch all loads first
        "nop\n\t" // placeholder
        "b.ne 1b\n\t"
        :
        : [Q] "r"(Q), [K] "r"(K), [S] "r"(S)
        : "memory"
    );

    (void)V; (void)O; // suppress unused warnings
}

// Better approach: Double-buffer K loads
// Load K[i+1] while computing K[i]
__attribute__((noinline))
void fused_double_buf(const int8_t* Q, const int8_t* K, const int8_t* V, int32_t* O) {
    int32_t S[6 * 64] __attribute__((aligned(64)));

    __asm__ volatile(
        "ptrue p0.b\n\t"
        "ptrue p1.s\n\t"
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
        "mov x10, #31\n\t"  // One less iteration (prologue handles first)

        // Prologue: load first K tile
        "ld1b {z0.b}, p0/z, [x4]\n\t"
        "ld1b {z1.b}, p0/z, [x4, #1, mul vl]\n\t"
        "ld1b {z2.b}, p0/z, [x4, #2, mul vl]\n\t"
        "ld1b {z3.b}, p0/z, [x4, #3, mul vl]\n\t"
        "add x4, x4, #256\n\t"

        "1:\n\t"
        // Load next K into z4-z7 while computing current K in z0-z3
        "ld1b {z4.b}, p0/z, [x4]\n\t"        // Prefetch next K
        "ld1rw {z7.s}, p1/z, [x5]\n\t"       // Q row 0
        "sdot z8.s, z0.b, z7.b\n\t"
        "sdot z9.s, z1.b, z7.b\n\t"
        "sdot z10.s, z2.b, z7.b\n\t"
        "sdot z11.s, z3.b, z7.b\n\t"

        "ld1b {z5.b}, p0/z, [x4, #1, mul vl]\n\t"  // Next K[1]
        "ld1rw {z7.s}, p1/z, [x6]\n\t"       // Q row 1
        "sdot z12.s, z0.b, z7.b\n\t"
        "sdot z13.s, z1.b, z7.b\n\t"
        "sdot z14.s, z2.b, z7.b\n\t"
        "sdot z15.s, z3.b, z7.b\n\t"

        "ld1b {z6.b}, p0/z, [x4, #2, mul vl]\n\t"  // Next K[2]
        "ld1rw {z7.s}, p1/z, [x7]\n\t"       // Q row 2
        "sdot z16.s, z0.b, z7.b\n\t"
        "sdot z17.s, z1.b, z7.b\n\t"
        "sdot z18.s, z2.b, z7.b\n\t"
        "sdot z19.s, z3.b, z7.b\n\t"

        "ld1rw {z7.s}, p1/z, [x11]\n\t"      // Q row 3
        "sdot z20.s, z0.b, z7.b\n\t"
        "sdot z21.s, z1.b, z7.b\n\t"
        "sdot z22.s, z2.b, z7.b\n\t"
        "sdot z23.s, z3.b, z7.b\n\t"

        "ld1rw {z7.s}, p1/z, [x12]\n\t"      // Q row 4
        "sdot z24.s, z0.b, z7.b\n\t"
        "sdot z25.s, z1.b, z7.b\n\t"
        "sdot z26.s, z2.b, z7.b\n\t"
        "sdot z27.s, z3.b, z7.b\n\t"

        "ld1rw {z7.s}, p1/z, [x13]\n\t"      // Q row 5
        "sdot z28.s, z0.b, z7.b\n\t"
        "sdot z29.s, z1.b, z7.b\n\t"
        "sdot z30.s, z2.b, z7.b\n\t"
        "ld1b {z0.b}, p0/z, [x4, #3, mul vl]\n\t"  // Next K[3] -> z0 (done with old z0)
        "sdot z31.s, z3.b, z7.b\n\t"

        // Swap: z4-z6,z0 -> z0-z3 for next iteration
        "mov z1.d, z4.d\n\t"
        "mov z2.d, z5.d\n\t"
        "mov z3.d, z6.d\n\t"
        // z0 already has next K[3]... wait, that's K[3] not K[0]
        // Need to reorganize

        "add x4, x4, #256\n\t"
        "add x5, x5, #4\n\t"
        "add x6, x6, #4\n\t"
        "add x7, x7, #4\n\t"
        "add x11, x11, #4\n\t"
        "add x12, x12, #4\n\t"
        "add x13, x13, #4\n\t"
        "subs x10, x10, #1\n\t"
        "b.ne 1b\n\t"

        // Epilogue: compute last tile (already in z4-z6, z0 has K[3])
        "ld1rw {z7.s}, p1/z, [x5]\n\t"
        "sdot z8.s, z4.b, z7.b\n\t"   // oops z4 is K[0], z1 should have it

        // This is getting complicated. Let me simplify.
        :
        : [Q] "r"(Q), [K] "r"(K), [S] "r"(S)
        : "memory"
    );

    (void)V; (void)O;
}

// Cleanest approach: Pre-load all Q values into z4-z7 (reusing for rows 4,5)
// Then just load K sequentially and compute
// This removes the interleaved ld1rw that might be causing stalls
__attribute__((noinline))
void fused_preload_q(const int8_t* Q, const int8_t* K, const int8_t* V, int32_t* O) {
    int32_t S[6 * 64] __attribute__((aligned(64)));

    __asm__ volatile(
        "ptrue p0.b\n\t"
        "ptrue p1.s\n\t"
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
        "mov x10, #64\n\t"  // 64 d_groups, no unroll

        "1:\n\t"
        // Load all 6 Q values first (before K to hide latency)
        "ld1rw {z4.s}, p1/z, [x5]\n\t"
        "ld1rw {z5.s}, p1/z, [x6]\n\t"
        "ld1rw {z6.s}, p1/z, [x7]\n\t"
        "ld1rw {z7.s}, p1/z, [x11]\n\t"
        // z4-z7 now have Q rows 0-3

        // Load K
        "ld1b {z0.b}, p0/z, [x4]\n\t"
        "ld1b {z1.b}, p0/z, [x4, #1, mul vl]\n\t"
        "ld1b {z2.b}, p0/z, [x4, #2, mul vl]\n\t"
        "ld1b {z3.b}, p0/z, [x4, #3, mul vl]\n\t"

        // Compute rows 0-3 using z4-z7
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

        // Now load Q rows 4-5 (reuse z4-z5)
        "ld1rw {z4.s}, p1/z, [x12]\n\t"
        "ld1rw {z5.s}, p1/z, [x13]\n\t"

        // Compute rows 4-5
        "sdot z24.s, z0.b, z4.b\n\t"
        "sdot z25.s, z1.b, z4.b\n\t"
        "sdot z26.s, z2.b, z4.b\n\t"
        "sdot z27.s, z3.b, z4.b\n\t"
        "sdot z28.s, z0.b, z5.b\n\t"
        "sdot z29.s, z1.b, z5.b\n\t"
        "sdot z30.s, z2.b, z5.b\n\t"
        "sdot z31.s, z3.b, z5.b\n\t"

        "add x4, x4, #256\n\t"
        "add x5, x5, #4\n\t"
        "add x6, x6, #4\n\t"
        "add x7, x7, #4\n\t"
        "add x11, x11, #4\n\t"
        "add x12, x12, #4\n\t"
        "add x13, x13, #4\n\t"
        "subs x10, x10, #1\n\t"
        "b.ne 1b\n\t"

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

    // Phase 2 with preload
    __asm__ volatile(
        "ptrue p0.b\n\t"
        "ptrue p1.s\n\t"
        "mov x15, #4\n\t"
        "mov x16, %[V]\n\t"
        "mov x17, %[O]\n\t"
        "2:\n\t"
        "mov z8.s, #0\n\t"  "mov z9.s, #0\n\t"  "mov z10.s, #0\n\t" "mov z11.s, #0\n\t"
        "mov z12.s, #0\n\t" "mov z13.s, #0\n\t" "mov z14.s, #0\n\t" "mov z15.s, #0\n\t"
        "mov z16.s, #0\n\t" "mov z17.s, #0\n\t" "mov z18.s, #0\n\t" "mov z19.s, #0\n\t"
        "mov z20.s, #0\n\t" "mov z21.s, #0\n\t" "mov z22.s, #0\n\t" "mov z23.s, #0\n\t"
        "mov z24.s, #0\n\t" "mov z25.s, #0\n\t" "mov z26.s, #0\n\t" "mov z27.s, #0\n\t"
        "mov z28.s, #0\n\t" "mov z29.s, #0\n\t" "mov z30.s, #0\n\t" "mov z31.s, #0\n\t"
        "mov x5, %[S]\n\t"
        "add x6, x5, #256\n\t"  "add x7, x5, #512\n\t"
        "add x11, x5, #768\n\t" "add x12, x5, #1024\n\t" "add x13, x5, #1280\n\t"
        "mov x4, x16\n\t"
        "mov x10, #16\n\t"  // 16 N_groups

        "3:\n\t"
        // Preload S values first
        "ld1rw {z4.s}, p1/z, [x5]\n\t"
        "ld1rw {z5.s}, p1/z, [x6]\n\t"
        "ld1rw {z6.s}, p1/z, [x7]\n\t"
        "ld1rw {z7.s}, p1/z, [x11]\n\t"

        // Load V
        "ld1b {z0.b}, p0/z, [x4]\n\t"
        "ld1b {z1.b}, p0/z, [x4, #1, mul vl]\n\t"
        "ld1b {z2.b}, p0/z, [x4, #2, mul vl]\n\t"
        "ld1b {z3.b}, p0/z, [x4, #3, mul vl]\n\t"

        // Compute rows 0-3
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

        // Load S rows 4-5
        "ld1rw {z4.s}, p1/z, [x12]\n\t"
        "ld1rw {z5.s}, p1/z, [x13]\n\t"

        // Compute rows 4-5
        "sdot z24.s, z0.b, z4.b\n\t"
        "sdot z25.s, z1.b, z4.b\n\t"
        "sdot z26.s, z2.b, z4.b\n\t"
        "sdot z27.s, z3.b, z4.b\n\t"
        "sdot z28.s, z0.b, z5.b\n\t"
        "sdot z29.s, z1.b, z5.b\n\t"
        "sdot z30.s, z2.b, z5.b\n\t"
        "sdot z31.s, z3.b, z5.b\n\t"

        "add x4, x4, #1024\n\t"  // V stride
        "add x5, x5, #4\n\t"
        "add x6, x6, #4\n\t"
        "add x7, x7, #4\n\t"
        "add x11, x11, #4\n\t"
        "add x12, x12, #4\n\t"
        "add x13, x13, #4\n\t"
        "subs x10, x10, #1\n\t"
        "b.ne 3b\n\t"

        "add z8.s, z8.s, z9.s\n\t"   "add z10.s, z10.s, z11.s\n\t" "add z8.s, z8.s, z10.s\n\t"
        "add z12.s, z12.s, z13.s\n\t" "add z14.s, z14.s, z15.s\n\t" "add z12.s, z12.s, z14.s\n\t"
        "add z16.s, z16.s, z17.s\n\t" "add z18.s, z18.s, z19.s\n\t" "add z16.s, z16.s, z18.s\n\t"
        "add z20.s, z20.s, z21.s\n\t" "add z22.s, z22.s, z23.s\n\t" "add z20.s, z20.s, z22.s\n\t"
        "add z24.s, z24.s, z25.s\n\t" "add z26.s, z26.s, z27.s\n\t" "add z24.s, z24.s, z26.s\n\t"
        "add z28.s, z28.s, z29.s\n\t" "add z30.s, z30.s, z31.s\n\t" "add z28.s, z28.s, z30.s\n\t"

        "st1w {z8.s}, p1, [x17]\n\t"  "add x17, x17, #1024\n\t"
        "st1w {z12.s}, p1, [x17]\n\t" "add x17, x17, #1024\n\t"
        "st1w {z16.s}, p1, [x17]\n\t" "add x17, x17, #1024\n\t"
        "st1w {z20.s}, p1, [x17]\n\t" "add x17, x17, #1024\n\t"
        "st1w {z24.s}, p1, [x17]\n\t" "add x17, x17, #1024\n\t"
        "st1w {z28.s}, p1, [x17]\n\t"
        "sub x17, x17, #4096\n\t" "sub x17, x17, #1024\n\t"
        "add x17, x17, #64\n\t"
        "add x16, x16, #64\n\t"
        "subs x15, x15, #1\n\t"
        "b.ne 2b\n\t"
        :
        : [S] "r"(S), [V] "r"(V), [O] "r"(O)
        : "memory", "x4", "x5", "x6", "x7", "x10", "x11", "x12", "x13",
          "x15", "x16", "x17",
          "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7",
          "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15",
          "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23",
          "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31", "p0", "p1"
    );
}

int main() {
    printf("==============================================\n");
    printf("Software Pipelining / Multi-Q-reg D=256\n");
    printf("==============================================\n\n");

    int D = 256;
    int N = 64;
    int D_groups = D / 4;

    int8_t* Q = aligned_alloc(256, 6 * D);
    int8_t* K = aligned_alloc(256, D_groups * N * 4);
    int8_t* V = aligned_alloc(256, N * D * 4);
    int32_t* O = aligned_alloc(256, 6 * D * 4);

    memset(Q, 1, 6 * D);
    memset(K, 1, D_groups * N * 4);
    memset(V, 1, N * D * 4);

    int iters = 5000;
    int total_sdot = 3072;

    printf("Testing different Q-loading strategies:\n");
    printf("  orig:       ld1rw z4 interleaved with SDOT (current)\n");
    printf("  preload_q:  batch Q loads, then compute all SDOTs\n\n");

    printf("%-20s %10s %12s %8s\n", "Version", "Ticks", "SDOT/tick", "Eff");
    printf("-------------------------------------------------------------\n");

    // Warmup
    for (int i = 0; i < 100; i++) {
        fused_orig(Q, K, V, O);
        fused_preload_q(Q, K, V, O);
    }

    // Original
    uint64_t start = rdcycle();
    for (int i = 0; i < iters; i++) {
        fused_orig(Q, K, V, O);
    }
    uint64_t end = rdcycle();
    double ticks1 = (double)(end - start) / iters;
    printf("%-20s %10.1f %12.2f %7.1f%%\n",
           "orig (z4 only)", ticks1, total_sdot/ticks1, (total_sdot/ticks1)/40*100);

    // Preload Q
    start = rdcycle();
    for (int i = 0; i < iters; i++) {
        fused_preload_q(Q, K, V, O);
    }
    end = rdcycle();
    double ticks2 = (double)(end - start) / iters;
    printf("%-20s %10.1f %12.2f %7.1f%%\n",
           "preload_q (z4-z7)", ticks2, total_sdot/ticks2, (total_sdot/ticks2)/40*100);

    double improvement = (ticks1 - ticks2) / ticks1 * 100;
    printf("\nImprovement: %.1f%% %s\n",
           improvement > 0 ? improvement : -improvement,
           improvement > 0 ? "faster" : "slower");

    free(Q); free(K); free(V); free(O);
    return 0;
}
