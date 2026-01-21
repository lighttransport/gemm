/*
 * Avoid MOV z.s, #0 for accumulator reset
 *
 * Problem: Phase 2 has 4 D-tiles × 24 MOV = 96 MOV instructions
 * MOV uses EXA pipe, competing with FPU
 *
 * Solutions:
 * 1. Use EOR z.d, z.d, z.d instead of MOV (different pipe)
 * 2. Ring approach: process all 4 D-tiles with 6 acc each, no reset needed
 * 3. Fused approach: compute O directly, accumulating across D-tiles
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

// Baseline: MOV z.s, #0 for accumulator reset
__attribute__((noinline))
void phase2_mov(const int32_t* S, const int8_t* V, int32_t* O) {
    __asm__ volatile(
        "ptrue p0.b\n\t"
        "ptrue p1.s\n\t"
        "mov x15, #4\n\t"
        "mov x16, %[V]\n\t"
        "mov x17, %[O]\n\t"
        "2:\n\t"
        // 24 MOV instructions to zero accumulators
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

// EOR instead of MOV for zeroing (uses different pipe)
__attribute__((noinline))
void phase2_eor(const int32_t* S, const int8_t* V, int32_t* O) {
    __asm__ volatile(
        "ptrue p0.b\n\t"
        "ptrue p1.s\n\t"
        "mov x15, #4\n\t"
        "mov x16, %[V]\n\t"
        "mov x17, %[O]\n\t"
        "2:\n\t"
        // Use EOR to zero accumulators (different pipe than MOV)
        "eor z8.d, z8.d, z8.d\n\t"   "eor z9.d, z9.d, z9.d\n\t"
        "eor z10.d, z10.d, z10.d\n\t" "eor z11.d, z11.d, z11.d\n\t"
        "eor z12.d, z12.d, z12.d\n\t" "eor z13.d, z13.d, z13.d\n\t"
        "eor z14.d, z14.d, z14.d\n\t" "eor z15.d, z15.d, z15.d\n\t"
        "eor z16.d, z16.d, z16.d\n\t" "eor z17.d, z17.d, z17.d\n\t"
        "eor z18.d, z18.d, z18.d\n\t" "eor z19.d, z19.d, z19.d\n\t"
        "eor z20.d, z20.d, z20.d\n\t" "eor z21.d, z21.d, z21.d\n\t"
        "eor z22.d, z22.d, z22.d\n\t" "eor z23.d, z23.d, z23.d\n\t"
        "eor z24.d, z24.d, z24.d\n\t" "eor z25.d, z25.d, z25.d\n\t"
        "eor z26.d, z26.d, z26.d\n\t" "eor z27.d, z27.d, z27.d\n\t"
        "eor z28.d, z28.d, z28.d\n\t" "eor z29.d, z29.d, z29.d\n\t"
        "eor z30.d, z30.d, z30.d\n\t" "eor z31.d, z31.d, z31.d\n\t"
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

// Ring approach: 4 D-tiles × 6 accumulators, NO reset in loop
// Process 1 N-vector per D-tile, store after each N_group
// z8-z13:  D-tile 0 (6 rows)
// z14-z19: D-tile 1 (6 rows)
// z20-z25: D-tile 2 (6 rows)
// z26-z31: D-tile 3 (6 rows)
__attribute__((noinline))
void phase2_ring(const int32_t* S, const int8_t* V, int32_t* O) {
    __asm__ volatile(
        "ptrue p0.b\n\t"
        "ptrue p1.s\n\t"

        // Zero ALL accumulators ONCE at start (not per D-tile!)
        "eor z8.d, z8.d, z8.d\n\t"   "eor z9.d, z9.d, z9.d\n\t"
        "eor z10.d, z10.d, z10.d\n\t" "eor z11.d, z11.d, z11.d\n\t"
        "eor z12.d, z12.d, z12.d\n\t" "eor z13.d, z13.d, z13.d\n\t"
        "eor z14.d, z14.d, z14.d\n\t" "eor z15.d, z15.d, z15.d\n\t"
        "eor z16.d, z16.d, z16.d\n\t" "eor z17.d, z17.d, z17.d\n\t"
        "eor z18.d, z18.d, z18.d\n\t" "eor z19.d, z19.d, z19.d\n\t"
        "eor z20.d, z20.d, z20.d\n\t" "eor z21.d, z21.d, z21.d\n\t"
        "eor z22.d, z22.d, z22.d\n\t" "eor z23.d, z23.d, z23.d\n\t"
        "eor z24.d, z24.d, z24.d\n\t" "eor z25.d, z25.d, z25.d\n\t"
        "eor z26.d, z26.d, z26.d\n\t" "eor z27.d, z27.d, z27.d\n\t"
        "eor z28.d, z28.d, z28.d\n\t" "eor z29.d, z29.d, z29.d\n\t"
        "eor z30.d, z30.d, z30.d\n\t" "eor z31.d, z31.d, z31.d\n\t"

        // Setup S pointers (6 rows)
        "mov x5, %[S]\n\t"
        "add x6, x5, #256\n\t"  "add x7, x5, #512\n\t"
        "add x11, x5, #768\n\t" "add x12, x5, #1024\n\t" "add x13, x5, #1280\n\t"

        // V pointers for 4 D-tiles (offset by 64 bytes = 1 vector)
        "mov x4, %[V]\n\t"           // D-tile 0: V + 0
        "add x8, x4, #64\n\t"        // D-tile 1: V + 64
        "add x9, x8, #64\n\t"        // D-tile 2: V + 128
        "add x14, x9, #64\n\t"       // D-tile 3: V + 192

        // O pointers for 4 D-tiles
        "mov x17, %[O]\n\t"          // D-tile 0
        "add x18, x17, #64\n\t"      // D-tile 1 (next 64-byte block)
        "add x19, x18, #64\n\t"      // D-tile 2
        "add x20, x19, #64\n\t"      // D-tile 3

        "mov x10, #16\n\t"  // 16 N_groups

        "1:\n\t"
        // Load V for all 4 D-tiles (1 vector each = 64 bytes)
        "ld1b {z0.b}, p0/z, [x4]\n\t"    // V D-tile 0
        "ld1b {z1.b}, p0/z, [x8]\n\t"    // V D-tile 1
        "ld1b {z2.b}, p0/z, [x9]\n\t"    // V D-tile 2
        "ld1b {z3.b}, p0/z, [x14]\n\t"   // V D-tile 3

        // Load S[n_group] for all 6 rows (4 bytes broadcast)
        "ld1rw {z4.s}, p1/z, [x5]\n\t"   // S row 0
        "sdot z8.s, z0.b, z4.b\n\t"      // D-tile 0, row 0
        "sdot z14.s, z1.b, z4.b\n\t"     // D-tile 1, row 0
        "sdot z20.s, z2.b, z4.b\n\t"     // D-tile 2, row 0
        "sdot z26.s, z3.b, z4.b\n\t"     // D-tile 3, row 0

        "ld1rw {z4.s}, p1/z, [x6]\n\t"   // S row 1
        "sdot z9.s, z0.b, z4.b\n\t"
        "sdot z15.s, z1.b, z4.b\n\t"
        "sdot z21.s, z2.b, z4.b\n\t"
        "sdot z27.s, z3.b, z4.b\n\t"

        "ld1rw {z4.s}, p1/z, [x7]\n\t"   // S row 2
        "sdot z10.s, z0.b, z4.b\n\t"
        "sdot z16.s, z1.b, z4.b\n\t"
        "sdot z22.s, z2.b, z4.b\n\t"
        "sdot z28.s, z3.b, z4.b\n\t"

        "ld1rw {z4.s}, p1/z, [x11]\n\t"  // S row 3
        "sdot z11.s, z0.b, z4.b\n\t"
        "sdot z17.s, z1.b, z4.b\n\t"
        "sdot z23.s, z2.b, z4.b\n\t"
        "sdot z29.s, z3.b, z4.b\n\t"

        "ld1rw {z4.s}, p1/z, [x12]\n\t"  // S row 4
        "sdot z12.s, z0.b, z4.b\n\t"
        "sdot z18.s, z1.b, z4.b\n\t"
        "sdot z24.s, z2.b, z4.b\n\t"
        "sdot z30.s, z3.b, z4.b\n\t"

        "ld1rw {z4.s}, p1/z, [x13]\n\t"  // S row 5
        "sdot z13.s, z0.b, z4.b\n\t"
        "sdot z19.s, z1.b, z4.b\n\t"
        "sdot z25.s, z2.b, z4.b\n\t"
        "sdot z31.s, z3.b, z4.b\n\t"

        // Advance pointers
        "add x4, x4, #1024\n\t"   // V stride = D * 4 = 1024
        "add x8, x8, #1024\n\t"
        "add x9, x9, #1024\n\t"
        "add x14, x14, #1024\n\t"
        "add x5, x5, #4\n\t"      // S stride = 4 bytes
        "add x6, x6, #4\n\t"
        "add x7, x7, #4\n\t"
        "add x11, x11, #4\n\t"
        "add x12, x12, #4\n\t"
        "add x13, x13, #4\n\t"

        "subs x10, x10, #1\n\t"
        "b.ne 1b\n\t"

        // Store all results (no reduction needed - 1 vector per row per D-tile)
        // D-tile 0
        "st1w {z8.s}, p1, [x17]\n\t"      "add x17, x17, #1024\n\t"
        "st1w {z9.s}, p1, [x17]\n\t"      "add x17, x17, #1024\n\t"
        "st1w {z10.s}, p1, [x17]\n\t"     "add x17, x17, #1024\n\t"
        "st1w {z11.s}, p1, [x17]\n\t"     "add x17, x17, #1024\n\t"
        "st1w {z12.s}, p1, [x17]\n\t"     "add x17, x17, #1024\n\t"
        "st1w {z13.s}, p1, [x17]\n\t"
        // D-tile 1
        "st1w {z14.s}, p1, [x18]\n\t"     "add x18, x18, #1024\n\t"
        "st1w {z15.s}, p1, [x18]\n\t"     "add x18, x18, #1024\n\t"
        "st1w {z16.s}, p1, [x18]\n\t"     "add x18, x18, #1024\n\t"
        "st1w {z17.s}, p1, [x18]\n\t"     "add x18, x18, #1024\n\t"
        "st1w {z18.s}, p1, [x18]\n\t"     "add x18, x18, #1024\n\t"
        "st1w {z19.s}, p1, [x18]\n\t"
        // D-tile 2
        "st1w {z20.s}, p1, [x19]\n\t"     "add x19, x19, #1024\n\t"
        "st1w {z21.s}, p1, [x19]\n\t"     "add x19, x19, #1024\n\t"
        "st1w {z22.s}, p1, [x19]\n\t"     "add x19, x19, #1024\n\t"
        "st1w {z23.s}, p1, [x19]\n\t"     "add x19, x19, #1024\n\t"
        "st1w {z24.s}, p1, [x19]\n\t"     "add x19, x19, #1024\n\t"
        "st1w {z25.s}, p1, [x19]\n\t"
        // D-tile 3
        "st1w {z26.s}, p1, [x20]\n\t"     "add x20, x20, #1024\n\t"
        "st1w {z27.s}, p1, [x20]\n\t"     "add x20, x20, #1024\n\t"
        "st1w {z28.s}, p1, [x20]\n\t"     "add x20, x20, #1024\n\t"
        "st1w {z29.s}, p1, [x20]\n\t"     "add x20, x20, #1024\n\t"
        "st1w {z30.s}, p1, [x20]\n\t"     "add x20, x20, #1024\n\t"
        "st1w {z31.s}, p1, [x20]\n\t"
        :
        : [S] "r"(S), [V] "r"(V), [O] "r"(O)
        : "memory", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12", "x13", "x14",
          "x17", "x18", "x19", "x20",
          "z0", "z1", "z2", "z3", "z4",
          "z8", "z9", "z10", "z11", "z12", "z13",
          "z14", "z15", "z16", "z17", "z18", "z19",
          "z20", "z21", "z22", "z23", "z24", "z25",
          "z26", "z27", "z28", "z29", "z30", "z31", "p0", "p1"
    );
}

int main() {
    printf("==============================================\n");
    printf("Avoid MOV Overhead in Phase 2\n");
    printf("==============================================\n\n");

    int D = 256;
    int N = 64;

    int32_t* S = aligned_alloc(256, 6 * N * 4);
    int8_t* V = aligned_alloc(256, N * D * 4);
    int32_t* O = aligned_alloc(256, 6 * D * 4);

    memset(S, 1, 6 * N * 4);
    memset(V, 1, N * D * 4);

    int iters = 5000;

    // Different SDOT counts:
    // mov/eor versions: 4 D-tiles × 16 N_groups × 24 SDOT = 1536
    // ring version: 16 N_groups × 24 SDOT = 384 (but covers all 4 D-tiles!)
    // Actually ring does: 16 N_groups × (6 rows × 4 D-tiles) = 16 × 24 = 384... wait
    // Let me recalculate: ring does 1 vector per D-tile per N_group
    // = 16 N_groups × 4 D-tiles × 6 rows × 1 SDOT = 384 SDOTs
    // That's only 1/4 of the original!

    // For fair comparison, ring needs 4× more iterations to match original SDOTs
    // Or we compute SDOTs differently

    // Actually the ring version processes the SAME total work:
    // Original: For each D-tile, accumulate 4 N-vectors across 16 N_groups
    //   = 4 D-tiles × 16 N_groups × 6 rows × 4 SDOTs = 1536 SDOTs
    // Ring: For each N_group, 1 SDOT per (D-tile, row) pair
    //   = 16 N_groups × 4 D-tiles × 6 rows × 1 SDOT = 384 SDOTs

    // The ring version computes LESS because it only uses 1 N-vector per D-tile
    // To match, we'd need 4× the loop iterations... but that defeats the purpose

    // Let's just compare timing and note the SDOT difference

    int p2_sdot_baseline = 1536;
    int p2_sdot_ring = 384;  // 1/4 the SDOTs but NO accumulator reset

    printf("Comparing approaches:\n");
    printf("  MOV baseline: 4 D-tiles × (24 MOV + 384 SDOT) = 96 MOV + 1536 SDOT\n");
    printf("  EOR version:  4 D-tiles × (24 EOR + 384 SDOT) = 96 EOR + 1536 SDOT\n");
    printf("  Ring version: 24 EOR once + 384 SDOT total (different algorithm!)\n\n");

    printf("%-20s %10s %12s %8s\n", "Version", "Ticks", "SDOT/tick", "Notes");
    printf("------------------------------------------------------------------\n");

    // Warmup
    for (int i = 0; i < 100; i++) {
        phase2_mov(S, V, O);
        phase2_eor(S, V, O);
        phase2_ring(S, V, O);
    }

    // MOV baseline
    uint64_t start = rdcycle();
    for (int i = 0; i < iters; i++) {
        phase2_mov(S, V, O);
    }
    uint64_t end = rdcycle();
    double ticks1 = (double)(end - start) / iters;
    printf("%-20s %10.1f %12.2f %s\n",
           "MOV zero", ticks1, p2_sdot_baseline/ticks1, "baseline");

    // EOR version
    start = rdcycle();
    for (int i = 0; i < iters; i++) {
        phase2_eor(S, V, O);
    }
    end = rdcycle();
    double ticks2 = (double)(end - start) / iters;
    printf("%-20s %10.1f %12.2f %s\n",
           "EOR zero", ticks2, p2_sdot_baseline/ticks2,
           ticks2 < ticks1 ? "better!" : "same");

    // Ring version (different algorithm)
    start = rdcycle();
    for (int i = 0; i < iters; i++) {
        phase2_ring(S, V, O);
    }
    end = rdcycle();
    double ticks3 = (double)(end - start) / iters;
    printf("%-20s %10.1f %12.2f %s\n",
           "Ring (no reset)", ticks3, p2_sdot_ring/ticks3, "1/4 SDOTs");

    printf("\nAnalysis:\n");
    printf("  MOV vs EOR: %.1f%% %s\n",
           (ticks1 - ticks2) / ticks1 * 100,
           ticks2 < ticks1 ? "faster" : "slower");
    printf("  Ring normalized: %.1f ticks for equiv work (×4 = %.1f)\n",
           ticks3, ticks3 * 4);
    printf("  Ring vs MOV (normalized): %.1f%% %s\n",
           (ticks1 - ticks3*4) / ticks1 * 100,
           ticks3*4 < ticks1 ? "faster" : "slower");

    free(S); free(V); free(O);
    return 0;
}
