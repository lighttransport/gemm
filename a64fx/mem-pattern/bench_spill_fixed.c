/*
 * Register spilling with STR/LDR - FIXED VERSION
 *
 * Bug in v1: After first loop, V pointers were incremented past correct positions
 * Fix: Save V base pointers and compute correct offsets for second half
 *
 * Strategy: Process 2 D-tiles per outer iteration
 * - D-tile 0: accumulate in z8-z19 (2 K-tiles × 6 rows = 12 regs)
 * - D-tile 1: accumulate in z20-z31 (2 K-tiles × 6 rows = 12 regs)
 * - First half: K-tiles 0-1, spill results
 * - Second half: K-tiles 2-3, load spilled, add, reduce, store
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

// Baseline: 4 D-tiles sequential
__attribute__((noinline))
void phase2_baseline(const int32_t* S, const int8_t* V, int32_t* O) {
    __asm__ volatile(
        "ptrue p0.b\n\t"
        "ptrue p1.s\n\t"
        "mov x15, #4\n\t"
        "mov x16, %[V]\n\t"
        "mov x17, %[O]\n\t"
        "2:\n\t"
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

// Spill version: 2 D-tiles per iteration, split K-tiles with spilling
// Process D-tiles 0&1 together, then 2&3 together
__attribute__((noinline))
void phase2_spill(const int32_t* S, const int8_t* V, int32_t* O, void* spill_buf) {
    __asm__ volatile(
        "ptrue p0.b\n\t"
        "ptrue p1.s\n\t"
        "mov x15, #2\n\t"              // 2 outer iterations (2 D-tiles each)
        "mov x16, %[V]\n\t"            // V base
        "mov x17, %[O]\n\t"            // O base
        "mov x18, %[spill]\n\t"        // Spill buffer (64-byte aligned)

        "2:\n\t"  // Outer D-tile pair loop
        // ===== FIRST HALF: K-tiles 0-1 for both D-tiles =====
        // Zero 24 accumulators (12 per D-tile, 2 K-tiles × 6 rows)
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

        // Setup S pointers (reset for each D-tile pair)
        "mov x5, %[S]\n\t"
        "add x6, x5, #256\n\t"  "add x7, x5, #512\n\t"
        "add x11, x5, #768\n\t" "add x12, x5, #1024\n\t" "add x13, x5, #1280\n\t"

        // V pointers: D-tile 0 at x16, D-tile 1 at x16+64
        "mov x4, x16\n\t"              // D-tile 0 V pointer
        "add x19, x16, #64\n\t"        // D-tile 1 V pointer

        // Inner loop: 16 N values (8 iterations with 2x unroll)
        "mov x10, #8\n\t"
        "3:\n\t"
        // Load K-tiles 0-1 for both D-tiles
        "ld1b {z0.b}, p0/z, [x4]\n\t"           // D-tile 0, K-tile 0
        "ld1b {z1.b}, p0/z, [x4, #1, mul vl]\n\t"   // D-tile 0, K-tile 1
        "ld1b {z2.b}, p0/z, [x19]\n\t"          // D-tile 1, K-tile 0
        "ld1b {z3.b}, p0/z, [x19, #1, mul vl]\n\t"  // D-tile 1, K-tile 1

        // Row 0: D-tile 0 → z8,z9, D-tile 1 → z20,z21
        "ld1rw {z4.s}, p1/z, [x5]\n\t"
        "sdot z8.s, z0.b, z4.b\n\t"   "sdot z9.s, z1.b, z4.b\n\t"
        "sdot z20.s, z2.b, z4.b\n\t"  "sdot z21.s, z3.b, z4.b\n\t"
        // Row 1
        "ld1rw {z4.s}, p1/z, [x6]\n\t"
        "sdot z10.s, z0.b, z4.b\n\t"  "sdot z11.s, z1.b, z4.b\n\t"
        "sdot z22.s, z2.b, z4.b\n\t"  "sdot z23.s, z3.b, z4.b\n\t"
        // Row 2
        "ld1rw {z4.s}, p1/z, [x7]\n\t"
        "sdot z12.s, z0.b, z4.b\n\t"  "sdot z13.s, z1.b, z4.b\n\t"
        "sdot z24.s, z2.b, z4.b\n\t"  "sdot z25.s, z3.b, z4.b\n\t"
        // Row 3
        "ld1rw {z4.s}, p1/z, [x11]\n\t"
        "sdot z14.s, z0.b, z4.b\n\t"  "sdot z15.s, z1.b, z4.b\n\t"
        "sdot z26.s, z2.b, z4.b\n\t"  "sdot z27.s, z3.b, z4.b\n\t"
        // Row 4
        "ld1rw {z4.s}, p1/z, [x12]\n\t"
        "sdot z16.s, z0.b, z4.b\n\t"  "sdot z17.s, z1.b, z4.b\n\t"
        "sdot z28.s, z2.b, z4.b\n\t"  "sdot z29.s, z3.b, z4.b\n\t"
        // Row 5
        "ld1rw {z4.s}, p1/z, [x13]\n\t"
        "sdot z18.s, z0.b, z4.b\n\t"  "sdot z19.s, z1.b, z4.b\n\t"
        "sdot z30.s, z2.b, z4.b\n\t"  "sdot z31.s, z3.b, z4.b\n\t"

        // Move to next N (stride 1024)
        "add x4, x4, #1024\n\t"
        "add x19, x19, #1024\n\t"

        // Second N of unroll (S offset +4)
        "ld1b {z0.b}, p0/z, [x4]\n\t"
        "ld1b {z1.b}, p0/z, [x4, #1, mul vl]\n\t"
        "ld1b {z2.b}, p0/z, [x19]\n\t"
        "ld1b {z3.b}, p0/z, [x19, #1, mul vl]\n\t"

        "ld1rw {z4.s}, p1/z, [x5, #4]\n\t"
        "sdot z8.s, z0.b, z4.b\n\t"   "sdot z9.s, z1.b, z4.b\n\t"
        "sdot z20.s, z2.b, z4.b\n\t"  "sdot z21.s, z3.b, z4.b\n\t"
        "ld1rw {z4.s}, p1/z, [x6, #4]\n\t"
        "sdot z10.s, z0.b, z4.b\n\t"  "sdot z11.s, z1.b, z4.b\n\t"
        "sdot z22.s, z2.b, z4.b\n\t"  "sdot z23.s, z3.b, z4.b\n\t"
        "ld1rw {z4.s}, p1/z, [x7, #4]\n\t"
        "sdot z12.s, z0.b, z4.b\n\t"  "sdot z13.s, z1.b, z4.b\n\t"
        "sdot z24.s, z2.b, z4.b\n\t"  "sdot z25.s, z3.b, z4.b\n\t"
        "ld1rw {z4.s}, p1/z, [x11, #4]\n\t"
        "sdot z14.s, z0.b, z4.b\n\t"  "sdot z15.s, z1.b, z4.b\n\t"
        "sdot z26.s, z2.b, z4.b\n\t"  "sdot z27.s, z3.b, z4.b\n\t"
        "ld1rw {z4.s}, p1/z, [x12, #4]\n\t"
        "sdot z16.s, z0.b, z4.b\n\t"  "sdot z17.s, z1.b, z4.b\n\t"
        "sdot z28.s, z2.b, z4.b\n\t"  "sdot z29.s, z3.b, z4.b\n\t"
        "ld1rw {z4.s}, p1/z, [x13, #4]\n\t"
        "sdot z18.s, z0.b, z4.b\n\t"  "sdot z19.s, z1.b, z4.b\n\t"
        "sdot z30.s, z2.b, z4.b\n\t"  "sdot z31.s, z3.b, z4.b\n\t"

        "add x4, x4, #1024\n\t"
        "add x19, x19, #1024\n\t"
        "add x5, x5, #8\n\t"   "add x6, x6, #8\n\t"   "add x7, x7, #8\n\t"
        "add x11, x11, #8\n\t" "add x12, x12, #8\n\t" "add x13, x13, #8\n\t"
        "subs x10, x10, #1\n\t"
        "b.ne 3b\n\t"

        // Spill 24 accumulators (store-fetch-bypass with 64-byte alignment)
        "str z8, [x18]\n\t"         "str z9, [x18, #1, mul vl]\n\t"
        "str z10, [x18, #2, mul vl]\n\t"  "str z11, [x18, #3, mul vl]\n\t"
        "str z12, [x18, #4, mul vl]\n\t"  "str z13, [x18, #5, mul vl]\n\t"
        "str z14, [x18, #6, mul vl]\n\t"  "str z15, [x18, #7, mul vl]\n\t"
        "str z16, [x18, #8, mul vl]\n\t"  "str z17, [x18, #9, mul vl]\n\t"
        "str z18, [x18, #10, mul vl]\n\t" "str z19, [x18, #11, mul vl]\n\t"
        "str z20, [x18, #12, mul vl]\n\t" "str z21, [x18, #13, mul vl]\n\t"
        "str z22, [x18, #14, mul vl]\n\t" "str z23, [x18, #15, mul vl]\n\t"
        "str z24, [x18, #16, mul vl]\n\t" "str z25, [x18, #17, mul vl]\n\t"
        "str z26, [x18, #18, mul vl]\n\t" "str z27, [x18, #19, mul vl]\n\t"
        "str z28, [x18, #20, mul vl]\n\t" "str z29, [x18, #21, mul vl]\n\t"
        "str z30, [x18, #22, mul vl]\n\t" "str z31, [x18, #23, mul vl]\n\t"

        // ===== SECOND HALF: K-tiles 2-3 for both D-tiles =====
        // Zero accumulators
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

        // Reset S pointers
        "mov x5, %[S]\n\t"
        "add x6, x5, #256\n\t"  "add x7, x5, #512\n\t"
        "add x11, x5, #768\n\t" "add x12, x5, #1024\n\t" "add x13, x5, #1280\n\t"

        // Reset V pointers for K-tiles 2-3 (offset +128 from D-tile base)
        "add x4, x16, #128\n\t"        // D-tile 0, K-tile 2 start
        "add x19, x16, #192\n\t"       // D-tile 1, K-tile 2 start (64+128)

        "mov x10, #8\n\t"
        "4:\n\t"
        // Load K-tiles 2-3 for both D-tiles
        "ld1b {z0.b}, p0/z, [x4]\n\t"           // D-tile 0, K-tile 2
        "ld1b {z1.b}, p0/z, [x4, #1, mul vl]\n\t"   // D-tile 0, K-tile 3
        "ld1b {z2.b}, p0/z, [x19]\n\t"          // D-tile 1, K-tile 2
        "ld1b {z3.b}, p0/z, [x19, #1, mul vl]\n\t"  // D-tile 1, K-tile 3

        "ld1rw {z4.s}, p1/z, [x5]\n\t"
        "sdot z8.s, z0.b, z4.b\n\t"   "sdot z9.s, z1.b, z4.b\n\t"
        "sdot z20.s, z2.b, z4.b\n\t"  "sdot z21.s, z3.b, z4.b\n\t"
        "ld1rw {z4.s}, p1/z, [x6]\n\t"
        "sdot z10.s, z0.b, z4.b\n\t"  "sdot z11.s, z1.b, z4.b\n\t"
        "sdot z22.s, z2.b, z4.b\n\t"  "sdot z23.s, z3.b, z4.b\n\t"
        "ld1rw {z4.s}, p1/z, [x7]\n\t"
        "sdot z12.s, z0.b, z4.b\n\t"  "sdot z13.s, z1.b, z4.b\n\t"
        "sdot z24.s, z2.b, z4.b\n\t"  "sdot z25.s, z3.b, z4.b\n\t"
        "ld1rw {z4.s}, p1/z, [x11]\n\t"
        "sdot z14.s, z0.b, z4.b\n\t"  "sdot z15.s, z1.b, z4.b\n\t"
        "sdot z26.s, z2.b, z4.b\n\t"  "sdot z27.s, z3.b, z4.b\n\t"
        "ld1rw {z4.s}, p1/z, [x12]\n\t"
        "sdot z16.s, z0.b, z4.b\n\t"  "sdot z17.s, z1.b, z4.b\n\t"
        "sdot z28.s, z2.b, z4.b\n\t"  "sdot z29.s, z3.b, z4.b\n\t"
        "ld1rw {z4.s}, p1/z, [x13]\n\t"
        "sdot z18.s, z0.b, z4.b\n\t"  "sdot z19.s, z1.b, z4.b\n\t"
        "sdot z30.s, z2.b, z4.b\n\t"  "sdot z31.s, z3.b, z4.b\n\t"

        "add x4, x4, #1024\n\t"
        "add x19, x19, #1024\n\t"

        // Second N of unroll
        "ld1b {z0.b}, p0/z, [x4]\n\t"
        "ld1b {z1.b}, p0/z, [x4, #1, mul vl]\n\t"
        "ld1b {z2.b}, p0/z, [x19]\n\t"
        "ld1b {z3.b}, p0/z, [x19, #1, mul vl]\n\t"

        "ld1rw {z4.s}, p1/z, [x5, #4]\n\t"
        "sdot z8.s, z0.b, z4.b\n\t"   "sdot z9.s, z1.b, z4.b\n\t"
        "sdot z20.s, z2.b, z4.b\n\t"  "sdot z21.s, z3.b, z4.b\n\t"
        "ld1rw {z4.s}, p1/z, [x6, #4]\n\t"
        "sdot z10.s, z0.b, z4.b\n\t"  "sdot z11.s, z1.b, z4.b\n\t"
        "sdot z22.s, z2.b, z4.b\n\t"  "sdot z23.s, z3.b, z4.b\n\t"
        "ld1rw {z4.s}, p1/z, [x7, #4]\n\t"
        "sdot z12.s, z0.b, z4.b\n\t"  "sdot z13.s, z1.b, z4.b\n\t"
        "sdot z24.s, z2.b, z4.b\n\t"  "sdot z25.s, z3.b, z4.b\n\t"
        "ld1rw {z4.s}, p1/z, [x11, #4]\n\t"
        "sdot z14.s, z0.b, z4.b\n\t"  "sdot z15.s, z1.b, z4.b\n\t"
        "sdot z26.s, z2.b, z4.b\n\t"  "sdot z27.s, z3.b, z4.b\n\t"
        "ld1rw {z4.s}, p1/z, [x12, #4]\n\t"
        "sdot z16.s, z0.b, z4.b\n\t"  "sdot z17.s, z1.b, z4.b\n\t"
        "sdot z28.s, z2.b, z4.b\n\t"  "sdot z29.s, z3.b, z4.b\n\t"
        "ld1rw {z4.s}, p1/z, [x13, #4]\n\t"
        "sdot z18.s, z0.b, z4.b\n\t"  "sdot z19.s, z1.b, z4.b\n\t"
        "sdot z30.s, z2.b, z4.b\n\t"  "sdot z31.s, z3.b, z4.b\n\t"

        "add x4, x4, #1024\n\t"
        "add x19, x19, #1024\n\t"
        "add x5, x5, #8\n\t"   "add x6, x6, #8\n\t"   "add x7, x7, #8\n\t"
        "add x11, x11, #8\n\t" "add x12, x12, #8\n\t" "add x13, x13, #8\n\t"
        "subs x10, x10, #1\n\t"
        "b.ne 4b\n\t"

        // Load spilled first-half results and add (store-fetch-bypass: 11 cycles)
        "ldr z0, [x18]\n\t"         "ldr z1, [x18, #1, mul vl]\n\t"
        "ldr z2, [x18, #2, mul vl]\n\t"   "ldr z3, [x18, #3, mul vl]\n\t"
        "add z8.s, z8.s, z0.s\n\t"    "add z9.s, z9.s, z1.s\n\t"
        "add z10.s, z10.s, z2.s\n\t"  "add z11.s, z11.s, z3.s\n\t"

        "ldr z0, [x18, #4, mul vl]\n\t"   "ldr z1, [x18, #5, mul vl]\n\t"
        "ldr z2, [x18, #6, mul vl]\n\t"   "ldr z3, [x18, #7, mul vl]\n\t"
        "add z12.s, z12.s, z0.s\n\t"  "add z13.s, z13.s, z1.s\n\t"
        "add z14.s, z14.s, z2.s\n\t"  "add z15.s, z15.s, z3.s\n\t"

        "ldr z0, [x18, #8, mul vl]\n\t"   "ldr z1, [x18, #9, mul vl]\n\t"
        "ldr z2, [x18, #10, mul vl]\n\t"  "ldr z3, [x18, #11, mul vl]\n\t"
        "add z16.s, z16.s, z0.s\n\t"  "add z17.s, z17.s, z1.s\n\t"
        "add z18.s, z18.s, z2.s\n\t"  "add z19.s, z19.s, z3.s\n\t"

        "ldr z0, [x18, #12, mul vl]\n\t"  "ldr z1, [x18, #13, mul vl]\n\t"
        "ldr z2, [x18, #14, mul vl]\n\t"  "ldr z3, [x18, #15, mul vl]\n\t"
        "add z20.s, z20.s, z0.s\n\t"  "add z21.s, z21.s, z1.s\n\t"
        "add z22.s, z22.s, z2.s\n\t"  "add z23.s, z23.s, z3.s\n\t"

        "ldr z0, [x18, #16, mul vl]\n\t"  "ldr z1, [x18, #17, mul vl]\n\t"
        "ldr z2, [x18, #18, mul vl]\n\t"  "ldr z3, [x18, #19, mul vl]\n\t"
        "add z24.s, z24.s, z0.s\n\t"  "add z25.s, z25.s, z1.s\n\t"
        "add z26.s, z26.s, z2.s\n\t"  "add z27.s, z27.s, z3.s\n\t"

        "ldr z0, [x18, #20, mul vl]\n\t"  "ldr z1, [x18, #21, mul vl]\n\t"
        "ldr z2, [x18, #22, mul vl]\n\t"  "ldr z3, [x18, #23, mul vl]\n\t"
        "add z28.s, z28.s, z0.s\n\t"  "add z29.s, z29.s, z1.s\n\t"
        "add z30.s, z30.s, z2.s\n\t"  "add z31.s, z31.s, z3.s\n\t"

        // Final reduction (1 ADD per row per D-tile)
        // D-tile 0: z8+z9, z10+z11, z12+z13, z14+z15, z16+z17, z18+z19
        "add z8.s, z8.s, z9.s\n\t"
        "add z10.s, z10.s, z11.s\n\t"
        "add z12.s, z12.s, z13.s\n\t"
        "add z14.s, z14.s, z15.s\n\t"
        "add z16.s, z16.s, z17.s\n\t"
        "add z18.s, z18.s, z19.s\n\t"
        // D-tile 1: z20+z21, z22+z23, z24+z25, z26+z27, z28+z29, z30+z31
        "add z20.s, z20.s, z21.s\n\t"
        "add z22.s, z22.s, z23.s\n\t"
        "add z24.s, z24.s, z25.s\n\t"
        "add z26.s, z26.s, z27.s\n\t"
        "add z28.s, z28.s, z29.s\n\t"
        "add z30.s, z30.s, z31.s\n\t"

        // Store D-tile 0 results
        "st1w {z8.s}, p1, [x17]\n\t"   "add x17, x17, #1024\n\t"
        "st1w {z10.s}, p1, [x17]\n\t"  "add x17, x17, #1024\n\t"
        "st1w {z12.s}, p1, [x17]\n\t"  "add x17, x17, #1024\n\t"
        "st1w {z14.s}, p1, [x17]\n\t"  "add x17, x17, #1024\n\t"
        "st1w {z16.s}, p1, [x17]\n\t"  "add x17, x17, #1024\n\t"
        "st1w {z18.s}, p1, [x17]\n\t"
        "sub x17, x17, #4096\n\t" "sub x17, x17, #1024\n\t"
        "add x17, x17, #64\n\t"

        // Store D-tile 1 results
        "st1w {z20.s}, p1, [x17]\n\t"  "add x17, x17, #1024\n\t"
        "st1w {z22.s}, p1, [x17]\n\t"  "add x17, x17, #1024\n\t"
        "st1w {z24.s}, p1, [x17]\n\t"  "add x17, x17, #1024\n\t"
        "st1w {z26.s}, p1, [x17]\n\t"  "add x17, x17, #1024\n\t"
        "st1w {z28.s}, p1, [x17]\n\t"  "add x17, x17, #1024\n\t"
        "st1w {z30.s}, p1, [x17]\n\t"
        "sub x17, x17, #4096\n\t" "sub x17, x17, #1024\n\t"
        "add x17, x17, #64\n\t"

        // Move to next D-tile pair
        "add x16, x16, #128\n\t"  // Skip 2 D-tiles (2 × 64 bytes)
        "subs x15, x15, #1\n\t"
        "b.ne 2b\n\t"
        :
        : [S] "r"(S), [V] "r"(V), [O] "r"(O), [spill] "r"(spill_buf)
        : "memory", "x4", "x5", "x6", "x7", "x10", "x11", "x12", "x13",
          "x15", "x16", "x17", "x18", "x19",
          "z0", "z1", "z2", "z3", "z4",
          "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15",
          "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23",
          "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31", "p0", "p1"
    );
}

int main() {
    printf("==============================================\n");
    printf("Register Spilling with STR/LDR - FIXED\n");
    printf("==============================================\n\n");

    int D = 256;
    int N = 64;

    int32_t* S = aligned_alloc(256, 6 * N * 4);
    int8_t* V = aligned_alloc(256, N * D * 4);
    int32_t* O1 = aligned_alloc(256, 6 * D * 4);
    int32_t* O2 = aligned_alloc(256, 6 * D * 4);
    void* spill_buf = aligned_alloc(64, 24 * 64);  // 24 × 64 bytes, 64-byte aligned

    memset(S, 1, 6 * N * 4);
    memset(V, 1, N * D * 4);

    int iters = 10000;
    int phase2_sdot = 1536;
    double peak_sdot_per_tick = 40.0;

    printf("SDOT count: %d (same for both versions)\n", phase2_sdot);
    printf("Spill buffer: 24 × 64 = 1536 bytes (64-byte aligned)\n\n");

    // Warmup
    for (int i = 0; i < 100; i++) {
        phase2_baseline(S, V, O1);
        phase2_spill(S, V, O2, spill_buf);
    }

    // Benchmark baseline
    uint64_t start = rdcycle();
    for (int i = 0; i < iters; i++) {
        phase2_baseline(S, V, O1);
    }
    uint64_t end = rdcycle();
    double ticks_base = (double)(end - start) / iters;
    double sdot_base = phase2_sdot / ticks_base;
    double eff_base = sdot_base / peak_sdot_per_tick * 100;

    // Benchmark spill
    start = rdcycle();
    for (int i = 0; i < iters; i++) {
        phase2_spill(S, V, O2, spill_buf);
    }
    end = rdcycle();
    double ticks_spill = (double)(end - start) / iters;
    double sdot_spill = phase2_sdot / ticks_spill;
    double eff_spill = sdot_spill / peak_sdot_per_tick * 100;

    printf("%-20s %10s %12s %10s\n", "Version", "Ticks", "SDOT/tick", "Efficiency");
    printf("----------------------------------------------------------\n");
    printf("%-20s %10.1f %12.2f %9.1f%%\n", "Baseline (4×seq)", ticks_base, sdot_base, eff_base);
    printf("%-20s %10.1f %12.2f %9.1f%%\n", "Spill (2×par)", ticks_spill, sdot_spill, eff_spill);

    printf("\n");
    double speedup = (ticks_base - ticks_spill) / ticks_base * 100;
    if (speedup > 0) {
        printf("Spilling is %.1f%% faster!\n", speedup);
    } else {
        printf("Baseline is %.1f%% faster.\n", -speedup);
    }

    free(S); free(V); free(O1); free(O2); free(spill_buf);
    return 0;
}
