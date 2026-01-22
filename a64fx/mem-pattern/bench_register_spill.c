/*
 * Register spilling with STR/LDR for store-fetch-bypass
 *
 * Key insight from A64FX:
 *   - STR/LDR (register store/load): 64 bytes each
 *   - 64-byte aligned addresses enable store-fetch-bypass
 *   - Minimal latency: 11 cycles (ld latency)
 *   - Much faster than cache access
 *
 * Strategy:
 *   - Process 2 D-tiles in parallel (12 accumulators each)
 *   - Split K-tile processing: first half, spill, second half
 *   - Use STR to spill intermediate results (64-byte aligned)
 *   - Use LDR to reload and continue accumulation
 *   - Reduces outer D-tile loop from 4 to 2 iterations
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

// Current: 4 D-tiles sequential (baseline)
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

// New: 2 D-tiles in parallel with register spilling
__attribute__((noinline))
void phase2_spill(const int32_t* S, const int8_t* V, int32_t* O, void* spill_buf) {
    __asm__ volatile(
        "ptrue p0.b\n\t"
        "ptrue p1.s\n\t"
        "mov x15, #2\n\t"  // Only 2 outer iterations (process 2 D-tiles each)
        "mov x16, %[V]\n\t"
        "mov x17, %[O]\n\t"
        "mov x18, %[spill]\n\t"  // 64-byte aligned spill buffer
        "2:\n\t"
        // Zero 24 accumulators for 2 D-tiles (12 each)
        // D-tile 0: z8-z19, D-tile 1: z20-z31
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
        "add x19, x16, #64\n\t"  // V pointer for D-tile 1 (64 bytes ahead)

        // First half: process 2 K-tiles for both D-tiles
        "mov x10, #4\n\t"  // Half the iterations
        "3:\n\t"
        // Load K-tiles for both D-tiles
        "ld1b {z0.b}, p0/z, [x4]\n\t"         // D-tile 0, K-tile 0
        "ld1b {z1.b}, p0/z, [x4, #1, mul vl]\n\t"  // D-tile 0, K-tile 1
        "ld1b {z2.b}, p0/z, [x19]\n\t"        // D-tile 1, K-tile 0
        "ld1b {z3.b}, p0/z, [x19, #1, mul vl]\n\t" // D-tile 1, K-tile 1

        // Process D-tile 0
        "ld1rw {z4.s}, p1/z, [x5]\n\t"
        "sdot z8.s, z0.b, z4.b\n\t"   "sdot z9.s, z1.b, z4.b\n\t"
        "ld1rw {z4.s}, p1/z, [x6]\n\t"
        "sdot z10.s, z0.b, z4.b\n\t"  "sdot z11.s, z1.b, z4.b\n\t"
        "ld1rw {z4.s}, p1/z, [x7]\n\t"
        "sdot z12.s, z0.b, z4.b\n\t"  "sdot z13.s, z1.b, z4.b\n\t"
        "ld1rw {z4.s}, p1/z, [x11]\n\t"
        "sdot z14.s, z0.b, z4.b\n\t"  "sdot z15.s, z1.b, z4.b\n\t"
        "ld1rw {z4.s}, p1/z, [x12]\n\t"
        "sdot z16.s, z0.b, z4.b\n\t"  "sdot z17.s, z1.b, z4.b\n\t"
        "ld1rw {z4.s}, p1/z, [x13]\n\t"
        "sdot z18.s, z0.b, z4.b\n\t"  "sdot z19.s, z1.b, z4.b\n\t"

        // Process D-tile 1
        "ld1rw {z4.s}, p1/z, [x5]\n\t"
        "sdot z20.s, z2.b, z4.b\n\t"  "sdot z21.s, z3.b, z4.b\n\t"
        "ld1rw {z4.s}, p1/z, [x6]\n\t"
        "sdot z22.s, z2.b, z4.b\n\t"  "sdot z23.s, z3.b, z4.b\n\t"
        "ld1rw {z4.s}, p1/z, [x7]\n\t"
        "sdot z24.s, z2.b, z4.b\n\t"  "sdot z25.s, z3.b, z4.b\n\t"
        "ld1rw {z4.s}, p1/z, [x11]\n\t"
        "sdot z26.s, z2.b, z4.b\n\t"  "sdot z27.s, z3.b, z4.b\n\t"
        "ld1rw {z4.s}, p1/z, [x12]\n\t"
        "sdot z28.s, z2.b, z4.b\n\t"  "sdot z29.s, z3.b, z4.b\n\t"
        "ld1rw {z4.s}, p1/z, [x13]\n\t"
        "sdot z30.s, z2.b, z4.b\n\t"  "sdot z31.s, z3.b, z4.b\n\t"

        "add x4, x4, #1024\n\t"
        "add x19, x19, #1024\n\t"
        "add x5, x5, #4\n\t"   "add x6, x6, #4\n\t"   "add x7, x7, #4\n\t"
        "add x11, x11, #4\n\t" "add x12, x12, #4\n\t" "add x13, x13, #4\n\t"
        "subs x10, x10, #1\n\t"
        "b.ne 3b\n\t"

        // Spill intermediate results to 64-byte aligned buffer
        // Store all 24 accumulators (12 per D-tile)
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

        // Zero accumulators for second half
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

        // Second half: process remaining 2 K-tiles
        "mov x10, #4\n\t"
        "4:\n\t"
        "ld1b {z0.b}, p0/z, [x4, #2, mul vl]\n\t"      // D-tile 0, K-tile 2
        "ld1b {z1.b}, p0/z, [x4, #3, mul vl]\n\t"      // D-tile 0, K-tile 3
        "ld1b {z2.b}, p0/z, [x19, #2, mul vl]\n\t"     // D-tile 1, K-tile 2
        "ld1b {z3.b}, p0/z, [x19, #3, mul vl]\n\t"     // D-tile 1, K-tile 3

        // Process D-tile 0
        "ld1rw {z4.s}, p1/z, [x5]\n\t"
        "sdot z8.s, z0.b, z4.b\n\t"   "sdot z9.s, z1.b, z4.b\n\t"
        "ld1rw {z4.s}, p1/z, [x6]\n\t"
        "sdot z10.s, z0.b, z4.b\n\t"  "sdot z11.s, z1.b, z4.b\n\t"
        "ld1rw {z4.s}, p1/z, [x7]\n\t"
        "sdot z12.s, z0.b, z4.b\n\t"  "sdot z13.s, z1.b, z4.b\n\t"
        "ld1rw {z4.s}, p1/z, [x11]\n\t"
        "sdot z14.s, z0.b, z4.b\n\t"  "sdot z15.s, z1.b, z4.b\n\t"
        "ld1rw {z4.s}, p1/z, [x12]\n\t"
        "sdot z16.s, z0.b, z4.b\n\t"  "sdot z17.s, z1.b, z4.b\n\t"
        "ld1rw {z4.s}, p1/z, [x13]\n\t"
        "sdot z18.s, z0.b, z4.b\n\t"  "sdot z19.s, z1.b, z4.b\n\t"

        // Process D-tile 1
        "ld1rw {z4.s}, p1/z, [x5]\n\t"
        "sdot z20.s, z2.b, z4.b\n\t"  "sdot z21.s, z3.b, z4.b\n\t"
        "ld1rw {z4.s}, p1/z, [x6]\n\t"
        "sdot z22.s, z2.b, z4.b\n\t"  "sdot z23.s, z3.b, z4.b\n\t"
        "ld1rw {z4.s}, p1/z, [x7]\n\t"
        "sdot z24.s, z2.b, z4.b\n\t"  "sdot z25.s, z3.b, z4.b\n\t"
        "ld1rw {z4.s}, p1/z, [x11]\n\t"
        "sdot z26.s, z2.b, z4.b\n\t"  "sdot z27.s, z3.b, z4.b\n\t"
        "ld1rw {z4.s}, p1/z, [x12]\n\t"
        "sdot z28.s, z2.b, z4.b\n\t"  "sdot z29.s, z3.b, z4.b\n\t"
        "ld1rw {z4.s}, p1/z, [x13]\n\t"
        "sdot z30.s, z2.b, z4.b\n\t"  "sdot z31.s, z3.b, z4.b\n\t"

        "add x4, x4, #1024\n\t"
        "add x19, x19, #1024\n\t"
        "add x5, x5, #4\n\t"   "add x6, x6, #4\n\t"   "add x7, x7, #4\n\t"
        "add x11, x11, #4\n\t" "add x12, x12, #4\n\t" "add x13, x13, #4\n\t"
        "subs x10, x10, #1\n\t"
        "b.ne 4b\n\t"

        // Load spilled results (store-fetch-bypass: 11 cycles)
        "ldr z0, [x18]\n\t"         "ldr z1, [x18, #1, mul vl]\n\t"
        "ldr z2, [x18, #2, mul vl]\n\t"   "ldr z3, [x18, #3, mul vl]\n\t"
        "ldr z4, [x18, #4, mul vl]\n\t"   "ldr z5, [x18, #5, mul vl]\n\t"
        "ldr z6, [x18, #6, mul vl]\n\t"   "ldr z7, [x18, #7, mul vl]\n\t"
        // Add first half to second half
        "add z8.s, z8.s, z0.s\n\t"    "add z9.s, z9.s, z1.s\n\t"
        "add z10.s, z10.s, z2.s\n\t"  "add z11.s, z11.s, z3.s\n\t"
        "add z12.s, z12.s, z4.s\n\t"  "add z13.s, z13.s, z5.s\n\t"
        "add z14.s, z14.s, z6.s\n\t"  "add z15.s, z15.s, z7.s\n\t"

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

        // Reduction (1 ADD per row, 2 D-tiles)
        "add z8.s, z8.s, z9.s\n\t"    "add z10.s, z10.s, z11.s\n\t"
        "add z12.s, z12.s, z13.s\n\t"  "add z14.s, z14.s, z15.s\n\t"
        "add z16.s, z16.s, z17.s\n\t"  "add z18.s, z18.s, z19.s\n\t"
        "add z20.s, z20.s, z21.s\n\t"  "add z22.s, z22.s, z23.s\n\t"
        "add z24.s, z24.s, z25.s\n\t"  "add z26.s, z26.s, z27.s\n\t"
        "add z28.s, z28.s, z29.s\n\t"  "add z30.s, z30.s, z31.s\n\t"

        // Store D-tile 0
        "st1w {z8.s}, p1, [x17]\n\t"  "add x17, x17, #1024\n\t"
        "st1w {z10.s}, p1, [x17]\n\t" "add x17, x17, #1024\n\t"
        "st1w {z12.s}, p1, [x17]\n\t" "add x17, x17, #1024\n\t"
        "st1w {z14.s}, p1, [x17]\n\t" "add x17, x17, #1024\n\t"
        "st1w {z16.s}, p1, [x17]\n\t" "add x17, x17, #1024\n\t"
        "st1w {z18.s}, p1, [x17]\n\t"
        "sub x17, x17, #4096\n\t" "sub x17, x17, #1024\n\t"
        "add x17, x17, #64\n\t"

        // Store D-tile 1
        "st1w {z20.s}, p1, [x17]\n\t"  "add x17, x17, #1024\n\t"
        "st1w {z22.s}, p1, [x17]\n\t"  "add x17, x17, #1024\n\t"
        "st1w {z24.s}, p1, [x17]\n\t"  "add x17, x17, #1024\n\t"
        "st1w {z26.s}, p1, [x17]\n\t"  "add x17, x17, #1024\n\t"
        "st1w {z28.s}, p1, [x17]\n\t"  "add x17, x17, #1024\n\t"
        "st1w {z30.s}, p1, [x17]\n\t"
        "sub x17, x17, #4096\n\t" "sub x17, x17, #1024\n\t"
        "add x17, x17, #64\n\t"

        "add x16, x16, #128\n\t"  // Move to next 2 D-tiles
        "subs x15, x15, #1\n\t"
        "b.ne 2b\n\t"
        :
        : [S] "r"(S), [V] "r"(V), [O] "r"(O), [spill] "r"(spill_buf)
        : "memory", "x4", "x5", "x6", "x7", "x10", "x11", "x12", "x13",
          "x15", "x16", "x17", "x18", "x19",
          "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7",
          "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15",
          "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23",
          "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31", "p0", "p1"
    );
}

int main() {
    printf("==============================================\n");
    printf("Register Spilling with STR/LDR (Store-Fetch-Bypass)\n");
    printf("==============================================\n\n");

    int D = 256;
    int N = 64;

    int32_t* S = aligned_alloc(256, 6 * N * 4);
    int8_t* V = aligned_alloc(256, N * D * 4);
    int32_t* O1 = aligned_alloc(256, 6 * D * 4);
    int32_t* O2 = aligned_alloc(256, 6 * D * 4);
    void* spill_buf = aligned_alloc(64, 24 * 64);  // 64-byte aligned for store-fetch-bypass

    memset(S, 1, 6 * N * 4);
    memset(V, 1, N * D * 4);

    int iters = 10000;
    int phase2_sdot = 1536;
    double peak_sdot_per_tick = 40.0;

    printf("Phase 2 optimization with register spilling\n\n");
    printf("Baseline (4 D-tiles sequential):\n");
    printf("  - 4 outer loop iterations\n");
    printf("  - 24 accumulators per D-tile\n");
    printf("  - Overhead: 24 EOR + 18 ADD + stores per D-tile\n\n");

    printf("Spill version (2 D-tiles in parallel):\n");
    printf("  - 2 outer loop iterations (halved!)\n");
    printf("  - Process 2 D-tiles simultaneously\n");
    printf("  - 12 accumulators per D-tile (2 K-tile groups)\n");
    printf("  - STR intermediate results (24×64 bytes)\n");
    printf("  - LDR with store-fetch-bypass (11 cycles)\n");
    printf("  - ADD spilled + current values\n\n");

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
    if (ticks_spill < ticks_base) {
        printf("Spilling wins! %.1f%% faster\n",
               (ticks_base - ticks_spill) / ticks_base * 100);
        printf("Store-fetch-bypass reduces D-tile loop overhead.\n");
    } else {
        printf("Baseline still better (spill %.1f%% slower).\n",
               (ticks_spill - ticks_base) / ticks_base * 100);
        printf("STR/LDR overhead outweighs loop reduction benefits.\n");
    }

    free(S); free(V); free(O1); free(O2); free(spill_buf);
    return 0;
}
