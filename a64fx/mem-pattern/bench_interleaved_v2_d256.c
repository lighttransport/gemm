/*
 * Interleaved v2: MR=6 throughout, but overlap Phase 2 D-tiles
 *
 * Idea: For Phase 2, we have 4 D-tiles. Can we overlap:
 *   - D-tile 0: compute while D-tile 1 data prefetches
 *   - Store D-tile 0 O, start D-tile 1 compute
 *   - etc.
 *
 * Also try: Process 2 D-tiles in parallel with 2×MR=3
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

// Baseline MR=6 (same as before)
extern void fused_mr6_baseline(const int8_t* Q, const int8_t* K, const int8_t* V, int32_t* O);

// Phase 2 with 2 D-tiles interleaved (MR=3 per tile)
// Process D-tile 0 rows 0-2 and D-tile 1 rows 0-2 in parallel registers
__attribute__((noinline))
void fused_interleaved_dtile(
    const int8_t* Q,
    const int8_t* K,
    const int8_t* V,
    int32_t* O
) {
    int32_t S[6 * 64] __attribute__((aligned(64)));

    // Phase 1: Standard MR=6
    __asm__ volatile(
        "ptrue p0.b\n\t"
        "ptrue p1.s\n\t"

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
        "add x7, x6, #256\n\t"
        "add x11, x7, #256\n\t"
        "add x12, x11, #256\n\t"
        "add x13, x12, #256\n\t"

        "mov x4, %[K]\n\t"
        "mov x10, #32\n\t"

        "1:\n\t"
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

        "ld1b {z0.b}, p0/z, [x4, #4, mul vl]\n\t"
        "ld1b {z1.b}, p0/z, [x4, #5, mul vl]\n\t"
        "ld1b {z2.b}, p0/z, [x4, #6, mul vl]\n\t"
        "ld1b {z3.b}, p0/z, [x4, #7, mul vl]\n\t"
        "add x4, x4, #512\n\t"

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

        :
        : [Q] "r"(Q), [K] "r"(K), [S] "r"(S)
        : "memory", "x4", "x5", "x6", "x7", "x10", "x11", "x12", "x13", "x14",
          "z0", "z1", "z2", "z3", "z4",
          "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15",
          "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23",
          "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31",
          "p0", "p1"
    );

    // Phase 2: Process 2 D-tiles at a time with interleaved stores
    // D-tile 0+1, then D-tile 2+3
    // For each pair: compute tile 0, store, compute tile 1, store
    // This keeps V data hot in L1 while alternating between tiles
    __asm__ volatile(
        "ptrue p0.b\n\t"
        "ptrue p1.s\n\t"

        "mov x15, #2\n\t"  // 2 pairs of D-tiles
        "mov x16, %[V]\n\t"
        "mov x17, %[O]\n\t"

        "20:\n\t"
        // ===== D-tile A (even) =====
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

        "mov x4, x16\n\t"  // V for D-tile A
        "mov x10, #8\n\t"

        "21:\n\t"
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
        "b.ne 21b\n\t"

        // Merge and store D-tile A
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

        // Store D-tile A O (keep in z-regs for later, no we can't - need regs for B)
        // Actually store immediately
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
        "add x17, x17, #64\n\t"  // Move to next D-tile column

        // ===== D-tile B (odd) =====
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

        "add x4, x16, #64\n\t"  // V for D-tile B (offset by 64)
        "mov x10, #8\n\t"

        "22:\n\t"
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
        "b.ne 22b\n\t"

        // Merge and store D-tile B
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

        // Advance V base for next pair
        "add x16, x16, #128\n\t"

        "subs x15, x15, #1\n\t"
        "b.ne 20b\n\t"

        :
        : [S] "r"(S), [V] "r"(V), [O] "r"(O)
        : "memory", "x4", "x5", "x6", "x7", "x10", "x11", "x12", "x13",
          "x15", "x16", "x17",
          "z0", "z1", "z2", "z3", "z4",
          "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15",
          "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23",
          "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31",
          "p0", "p1"
    );
}

int main() {
    printf("==============================================\n");
    printf("Interleaved v2 D=256 (D-tile pairs)\n");
    printf("==============================================\n\n");

    int D = 256;
    int N = 64;
    int D_groups = D / 4;

    int8_t* Q = aligned_alloc(256, 6 * D);
    int8_t* K = aligned_alloc(256, D_groups * N * 4);
    int8_t* V = aligned_alloc(256, N * D * 4);
    int32_t* O1 = aligned_alloc(256, 6 * D * 4);
    int32_t* O2 = aligned_alloc(256, 6 * D * 4);

    memset(Q, 1, 6 * D);
    memset(K, 1, D_groups * N * 4);
    memset(V, 1, N * D * 4);

    int iters = 5000;
    int total_sdot = 3072;

    printf("%-30s %10s %12s %8s\n", "Version", "Ticks", "SDOT/tick", "Eff");
    printf("-------------------------------------------------------------------\n");

    // Warmup
    for (int i = 0; i < 100; i++) {
        fused_interleaved_dtile(Q, K, V, O2);
    }

    // D-tile interleaved
    uint64_t start = rdcycle();
    for (int i = 0; i < iters; i++) {
        fused_interleaved_dtile(Q, K, V, O2);
    }
    uint64_t end = rdcycle();
    double ticks2 = (double)(end - start) / iters;
    printf("%-30s %10.1f %12.2f %7.1f%%\n",
           "D-tile pairs (MR=6)", ticks2, total_sdot/ticks2, (total_sdot/ticks2)/40*100);

    printf("\nNote: Processing D-tiles in A-B pairs to overlap stores\n");

    free(Q); free(K); free(V); free(O1); free(O2);
    return 0;
}
