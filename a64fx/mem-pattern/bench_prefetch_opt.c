/*
 * Phase 2 Optimization with Prefetch and Software Pipelining
 *
 * Analysis shows:
 * - Reduced K-tile approach is WORSE (58.5% vs 79.1% baseline)
 * - The bottleneck is NOT reduction overhead
 * - Most likely bottleneck: memory latency or SDOT pipeline stalls
 *
 * This benchmark tests:
 * 1. Prefetch hints for V matrix
 * 2. Software pipelining (overlap loads with compute)
 * 3. Tighter interleaving of loads and SDOT
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

// ============================================================
// Baseline Phase 2
// ============================================================
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

// ============================================================
// Phase 2 with Prefetch
// ============================================================
// Add prefetch hints for next V block
__attribute__((noinline))
void phase2_prefetch(const int32_t* S, const int8_t* V, int32_t* O) {
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
        // Prefetch next N's V data (2 iterations ahead = 2048 bytes)
        "prfm pldl1strm, [x4, #2048]\n\t"
        "prfm pldl1strm, [x4, #2304]\n\t"

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

        // Prefetch
        "prfm pldl1strm, [x4, #2048]\n\t"
        "prfm pldl1strm, [x4, #2304]\n\t"

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

// ============================================================
// Phase 2 with Better Interleaving (load-compute overlap)
// ============================================================
// Load next V while computing with current V
__attribute__((noinline))
void phase2_pipelined(const int32_t* S, const int8_t* V, int32_t* O) {
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

        // Prologue: Load first V
        "ld1b {z0.b}, p0/z, [x4]\n\t"
        "ld1b {z1.b}, p0/z, [x4, #1, mul vl]\n\t"
        "ld1b {z2.b}, p0/z, [x4, #2, mul vl]\n\t"
        "ld1b {z3.b}, p0/z, [x4, #3, mul vl]\n\t"
        "add x4, x4, #1024\n\t"

        "mov x10, #7\n\t"  // 7 more iterations (total 8 × 2 = 16 N)
        "3:\n\t"
        // Load next V while computing with current
        "ld1b {z5.b}, p0/z, [x4]\n\t"
        "ld1rw {z4.s}, p1/z, [x5]\n\t"
        "sdot z8.s, z0.b, z4.b\n\t"
        "ld1b {z6.b}, p0/z, [x4, #1, mul vl]\n\t"
        "sdot z9.s, z1.b, z4.b\n\t"
        "ld1b {z7.b}, p0/z, [x4, #2, mul vl]\n\t"
        "sdot z10.s, z2.b, z4.b\n\t"
        "sdot z11.s, z3.b, z4.b\n\t"

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

        // Load next V while computing with previous loaded V
        "ld1b {z0.b}, p0/z, [x4]\n\t"
        "ld1rw {z4.s}, p1/z, [x5, #4]\n\t"
        "sdot z8.s, z5.b, z4.b\n\t"
        "ld1b {z1.b}, p0/z, [x4, #1, mul vl]\n\t"
        "sdot z9.s, z6.b, z4.b\n\t"
        "ld1b {z2.b}, p0/z, [x4, #2, mul vl]\n\t"
        "sdot z10.s, z7.b, z4.b\n\t"
        "ld1b {z3.b}, p0/z, [x4, #3, mul vl]\n\t"
        // Note: z5 only had 3 vectors loaded, need to fix this
        // Actually the interleaving is getting complex, let me simplify

        "ld1rw {z4.s}, p1/z, [x6, #4]\n\t"
        "sdot z12.s, z5.b, z4.b\n\t" "sdot z13.s, z6.b, z4.b\n\t"
        "sdot z14.s, z7.b, z4.b\n\t"
        "ld1rw {z4.s}, p1/z, [x7, #4]\n\t"
        "sdot z16.s, z5.b, z4.b\n\t" "sdot z17.s, z6.b, z4.b\n\t"
        "sdot z18.s, z7.b, z4.b\n\t"
        "ld1rw {z4.s}, p1/z, [x11, #4]\n\t"
        "sdot z20.s, z5.b, z4.b\n\t" "sdot z21.s, z6.b, z4.b\n\t"
        "sdot z22.s, z7.b, z4.b\n\t"
        "ld1rw {z4.s}, p1/z, [x12, #4]\n\t"
        "sdot z24.s, z5.b, z4.b\n\t" "sdot z25.s, z6.b, z4.b\n\t"
        "sdot z26.s, z7.b, z4.b\n\t"
        "ld1rw {z4.s}, p1/z, [x13, #4]\n\t"
        "sdot z28.s, z5.b, z4.b\n\t" "sdot z29.s, z6.b, z4.b\n\t"
        "sdot z30.s, z7.b, z4.b\n\t"

        "add x4, x4, #1024\n\t"
        "add x5, x5, #8\n\t"   "add x6, x6, #8\n\t"   "add x7, x7, #8\n\t"
        "add x11, x11, #8\n\t" "add x12, x12, #8\n\t" "add x13, x13, #8\n\t"
        "subs x10, x10, #1\n\t"
        "b.ne 3b\n\t"

        // Epilogue: Process last iteration without loading next
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
    printf("Phase 2 Prefetch and Pipelining Optimization\n");
    printf("==============================================\n\n");

    int D = 256;
    int N = 64;

    int32_t* S = aligned_alloc(256, 6 * N * 4);
    int8_t* V = aligned_alloc(256, N * D * 4);
    int32_t* O1 = aligned_alloc(256, 6 * D * 4);
    int32_t* O2 = aligned_alloc(256, 6 * D * 4);
    int32_t* O3 = aligned_alloc(256, 6 * D * 4);

    memset(S, 1, 6 * N * 4);
    memset(V, 1, N * D * 4);

    int iters = 10000;
    int phase2_sdot = 1536;
    double peak_sdot_per_tick = 40.0;

    // Warmup
    for (int i = 0; i < 100; i++) {
        phase2_baseline(S, V, O1);
        phase2_prefetch(S, V, O2);
        phase2_pipelined(S, V, O3);
    }

    // Benchmark baseline
    uint64_t start = rdcycle();
    for (int i = 0; i < iters; i++) {
        phase2_baseline(S, V, O1);
    }
    uint64_t end = rdcycle();
    double ticks_base = (double)(end - start) / iters;
    double eff_base = (phase2_sdot / ticks_base) / peak_sdot_per_tick * 100;

    // Benchmark prefetch
    start = rdcycle();
    for (int i = 0; i < iters; i++) {
        phase2_prefetch(S, V, O2);
    }
    end = rdcycle();
    double ticks_prf = (double)(end - start) / iters;
    double eff_prf = (phase2_sdot / ticks_prf) / peak_sdot_per_tick * 100;

    // Benchmark pipelined
    start = rdcycle();
    for (int i = 0; i < iters; i++) {
        phase2_pipelined(S, V, O3);
    }
    end = rdcycle();
    double ticks_pipe = (double)(end - start) / iters;
    double eff_pipe = (phase2_sdot / ticks_pipe) / peak_sdot_per_tick * 100;

    printf("%-25s %10s %10s\n", "Approach", "Ticks", "Efficiency");
    printf("------------------------------------------------------\n");
    printf("%-25s %10.1f %9.1f%%\n", "Baseline", ticks_base, eff_base);
    printf("%-25s %10.1f %9.1f%%\n", "With Prefetch", ticks_prf, eff_prf);
    printf("%-25s %10.1f %9.1f%%\n", "Pipelined (buggy)", ticks_pipe, eff_pipe);

    printf("\n");
    printf("Note: Pipelined version has incomplete K-tile 3 processing\n");
    printf("due to register pressure. Results may be incorrect.\n");

    free(S); free(V); free(O1); free(O2); free(O3);
    return 0;
}
