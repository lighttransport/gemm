// bench_fused_optimized.c
// Optimized fused attention with:
// 1. Interleaved S storage for efficient ld1rw P loads
// 2. Sector cache hints for K/V streaming

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <arm_sve.h>

static inline uint64_t rdtsc(void) {
    uint64_t val;
    __asm__ volatile("mrs %0, cntvct_el0" : "=r"(val));
    return val;
}

// Baseline: Row-major S, standard loads
void fused_baseline(
    const int8_t* Q,
    const int8_t* K_int,
    int32_t* S,           // [4, 64] row-major
    const int8_t* V_t,
    int32_t* O
) {
    __asm__ volatile(
        "ptrue p0.b\n\t"
        "ptrue p1.s\n\t"

        // Phase 1: Q@K^T -> S (row-major)
        "mov x0, %[q]\n\t"
        "add x1, x0, #256\n\t"
        "add x2, x0, #512\n\t"
        "add x3, x0, #768\n\t"
        "mov x4, %[k]\n\t"

        "dup z24.s, #0\n\t"
        "dup z25.s, #0\n\t"
        "dup z26.s, #0\n\t"
        "dup z27.s, #0\n\t"

        "mov x5, #16\n\t"
        "1:\n\t"
        "ld1b {z0.b}, p0/z, [x4]\n\t"
        "ld1b {z1.b}, p0/z, [x4, #1, mul vl]\n\t"
        "ld1b {z2.b}, p0/z, [x4, #2, mul vl]\n\t"
        "ld1b {z3.b}, p0/z, [x4, #3, mul vl]\n\t"
        "ld1rw {z4.s}, p1/z, [x0]\n\t"
        "ld1rw {z5.s}, p1/z, [x1]\n\t"
        "ld1rw {z6.s}, p1/z, [x2]\n\t"
        "ld1rw {z7.s}, p1/z, [x3]\n\t"

        "ld1b {z8.b}, p0/z, [x4, #4, mul vl]\n\t"
        "ld1b {z9.b}, p0/z, [x4, #5, mul vl]\n\t"
        "ld1b {z10.b}, p0/z, [x4, #6, mul vl]\n\t"
        "ld1b {z11.b}, p0/z, [x4, #7, mul vl]\n\t"
        "ld1rw {z12.s}, p1/z, [x0, #4]\n\t"
        "ld1rw {z13.s}, p1/z, [x1, #4]\n\t"
        "ld1rw {z14.s}, p1/z, [x2, #4]\n\t"
        "ld1rw {z15.s}, p1/z, [x3, #4]\n\t"

        "add x4, x4, #512\n\t"

        "ld1b {z0.b}, p0/z, [x4]\n\t"
        "ld1b {z1.b}, p0/z, [x4, #1, mul vl]\n\t"
        "ld1b {z2.b}, p0/z, [x4, #2, mul vl]\n\t"
        "ld1b {z3.b}, p0/z, [x4, #3, mul vl]\n\t"
        "ld1rw {z4.s}, p1/z, [x0, #8]\n\t"
        "ld1rw {z5.s}, p1/z, [x1, #8]\n\t"
        "ld1rw {z6.s}, p1/z, [x2, #8]\n\t"
        "ld1rw {z7.s}, p1/z, [x3, #8]\n\t"

        "ld1b {z8.b}, p0/z, [x4, #4, mul vl]\n\t"
        "ld1b {z9.b}, p0/z, [x4, #5, mul vl]\n\t"
        "ld1b {z10.b}, p0/z, [x4, #6, mul vl]\n\t"
        "ld1b {z11.b}, p0/z, [x4, #7, mul vl]\n\t"
        "ld1rw {z12.s}, p1/z, [x0, #12]\n\t"
        "ld1rw {z13.s}, p1/z, [x1, #12]\n\t"
        "ld1rw {z14.s}, p1/z, [x2, #12]\n\t"
        "ld1rw {z15.s}, p1/z, [x3, #12]\n\t"

        "add x4, x4, #512\n\t"
        "add x0, x0, #16\n\t"
        "add x1, x1, #16\n\t"
        "add x2, x2, #16\n\t"
        "add x3, x3, #16\n\t"
        "subs x5, x5, #1\n\t"
        "b.ne 1b\n\t"

        // Store S row-major: [4, 64] = 4 x 256 bytes
        "mov x6, %[s]\n\t"
        "st1w {z24.s}, p1, [x6]\n\t"
        "add x6, x6, #256\n\t"
        "st1w {z25.s}, p1, [x6]\n\t"
        "add x6, x6, #256\n\t"
        "st1w {z26.s}, p1, [x6]\n\t"
        "add x6, x6, #256\n\t"
        "st1w {z27.s}, p1, [x6]\n\t"

        // Phase 2: P@V with strided P loads
        "mov x7, %[v]\n\t"
        "mov x8, %[o]\n\t"
        "mov x9, #4\n\t"

        "2:\n\t"
        "mov x0, %[s]\n\t"
        "add x1, x0, #256\n\t"
        "add x2, x0, #512\n\t"
        "add x3, x0, #768\n\t"

        "dup z24.s, #0\n\t"
        "dup z25.s, #0\n\t"
        "dup z26.s, #0\n\t"
        "dup z27.s, #0\n\t"

        "mov x10, x7\n\t"
        "mov x5, #4\n\t"

        "3:\n\t"
        "ld1b {z0.b}, p0/z, [x10]\n\t"
        "ld1b {z1.b}, p0/z, [x10, #1, mul vl]\n\t"
        "ld1b {z2.b}, p0/z, [x10, #2, mul vl]\n\t"
        "ld1b {z3.b}, p0/z, [x10, #3, mul vl]\n\t"
        "ld1rw {z4.s}, p1/z, [x0]\n\t"
        "ld1rw {z5.s}, p1/z, [x1]\n\t"
        "ld1rw {z6.s}, p1/z, [x2]\n\t"
        "ld1rw {z7.s}, p1/z, [x3]\n\t"

        "add x10, x10, #1024\n\t"

        "ld1b {z8.b}, p0/z, [x10]\n\t"
        "ld1b {z9.b}, p0/z, [x10, #1, mul vl]\n\t"
        "ld1b {z10.b}, p0/z, [x10, #2, mul vl]\n\t"
        "ld1b {z11.b}, p0/z, [x10, #3, mul vl]\n\t"
        "ld1rw {z12.s}, p1/z, [x0, #4]\n\t"
        "ld1rw {z13.s}, p1/z, [x1, #4]\n\t"
        "ld1rw {z14.s}, p1/z, [x2, #4]\n\t"
        "ld1rw {z15.s}, p1/z, [x3, #4]\n\t"

        "add x10, x10, #1024\n\t"

        "ld1b {z0.b}, p0/z, [x10]\n\t"
        "ld1b {z1.b}, p0/z, [x10, #1, mul vl]\n\t"
        "ld1b {z2.b}, p0/z, [x10, #2, mul vl]\n\t"
        "ld1b {z3.b}, p0/z, [x10, #3, mul vl]\n\t"
        "ld1rw {z4.s}, p1/z, [x0, #8]\n\t"
        "ld1rw {z5.s}, p1/z, [x1, #8]\n\t"
        "ld1rw {z6.s}, p1/z, [x2, #8]\n\t"
        "ld1rw {z7.s}, p1/z, [x3, #8]\n\t"

        "add x10, x10, #1024\n\t"

        "ld1b {z8.b}, p0/z, [x10]\n\t"
        "ld1b {z9.b}, p0/z, [x10, #1, mul vl]\n\t"
        "ld1b {z10.b}, p0/z, [x10, #2, mul vl]\n\t"
        "ld1b {z11.b}, p0/z, [x10, #3, mul vl]\n\t"
        "ld1rw {z12.s}, p1/z, [x0, #12]\n\t"
        "ld1rw {z13.s}, p1/z, [x1, #12]\n\t"
        "ld1rw {z14.s}, p1/z, [x2, #12]\n\t"
        "ld1rw {z15.s}, p1/z, [x3, #12]\n\t"

        "add x10, x10, #1024\n\t"
        "add x0, x0, #16\n\t"
        "add x1, x1, #16\n\t"
        "add x2, x2, #16\n\t"
        "add x3, x3, #16\n\t"
        "subs x5, x5, #1\n\t"
        "b.ne 3b\n\t"

        "st1w {z24.s}, p1, [x8]\n\t"
        "add x8, x8, #1024\n\t"
        "st1w {z25.s}, p1, [x8]\n\t"
        "add x8, x8, #1024\n\t"
        "st1w {z26.s}, p1, [x8]\n\t"
        "add x8, x8, #1024\n\t"
        "st1w {z27.s}, p1, [x8]\n\t"
        "sub x8, x8, #3008\n\t"

        "add x7, x7, #256\n\t"
        "subs x9, x9, #1\n\t"
        "b.ne 2b\n\t"
        :
        : [q] "r"(Q), [k] "r"(K_int), [s] "r"(S), [v] "r"(V_t), [o] "r"(O)
        : "x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10",
          "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7",
          "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15",
          "z24", "z25", "z26", "z27",
          "p0", "p1", "memory"
    );
}

// Optimized: Interleaved S storage + optimized ld1rw for P
// S layout: [col, row] = S[0:64, 0:4] interleaved
// So P[row, col] = S_interleaved[col*4 + row]
void fused_interleaved_s(
    const int8_t* Q,
    const int8_t* K_int,
    int32_t* S,           // [64, 4] interleaved (col-major)
    const int8_t* V_t,
    int32_t* O
) {
    __asm__ volatile(
        "ptrue p0.b\n\t"
        "ptrue p1.s\n\t"

        // Phase 1: Q@K^T -> S interleaved
        "mov x0, %[q]\n\t"
        "add x1, x0, #256\n\t"
        "add x2, x0, #512\n\t"
        "add x3, x0, #768\n\t"
        "mov x4, %[k]\n\t"

        "dup z24.s, #0\n\t"
        "dup z25.s, #0\n\t"
        "dup z26.s, #0\n\t"
        "dup z27.s, #0\n\t"

        "mov x5, #16\n\t"
        "1:\n\t"
        "ld1b {z0.b}, p0/z, [x4]\n\t"
        "ld1b {z1.b}, p0/z, [x4, #1, mul vl]\n\t"
        "ld1b {z2.b}, p0/z, [x4, #2, mul vl]\n\t"
        "ld1b {z3.b}, p0/z, [x4, #3, mul vl]\n\t"
        "ld1rw {z4.s}, p1/z, [x0]\n\t"
        "ld1rw {z5.s}, p1/z, [x1]\n\t"
        "ld1rw {z6.s}, p1/z, [x2]\n\t"
        "ld1rw {z7.s}, p1/z, [x3]\n\t"

        "ld1b {z8.b}, p0/z, [x4, #4, mul vl]\n\t"
        "ld1b {z9.b}, p0/z, [x4, #5, mul vl]\n\t"
        "ld1b {z10.b}, p0/z, [x4, #6, mul vl]\n\t"
        "ld1b {z11.b}, p0/z, [x4, #7, mul vl]\n\t"
        "ld1rw {z12.s}, p1/z, [x0, #4]\n\t"
        "ld1rw {z13.s}, p1/z, [x1, #4]\n\t"
        "ld1rw {z14.s}, p1/z, [x2, #4]\n\t"
        "ld1rw {z15.s}, p1/z, [x3, #4]\n\t"

        "add x4, x4, #512\n\t"

        "ld1b {z0.b}, p0/z, [x4]\n\t"
        "ld1b {z1.b}, p0/z, [x4, #1, mul vl]\n\t"
        "ld1b {z2.b}, p0/z, [x4, #2, mul vl]\n\t"
        "ld1b {z3.b}, p0/z, [x4, #3, mul vl]\n\t"
        "ld1rw {z4.s}, p1/z, [x0, #8]\n\t"
        "ld1rw {z5.s}, p1/z, [x1, #8]\n\t"
        "ld1rw {z6.s}, p1/z, [x2, #8]\n\t"
        "ld1rw {z7.s}, p1/z, [x3, #8]\n\t"

        "ld1b {z8.b}, p0/z, [x4, #4, mul vl]\n\t"
        "ld1b {z9.b}, p0/z, [x4, #5, mul vl]\n\t"
        "ld1b {z10.b}, p0/z, [x4, #6, mul vl]\n\t"
        "ld1b {z11.b}, p0/z, [x4, #7, mul vl]\n\t"
        "ld1rw {z12.s}, p1/z, [x0, #12]\n\t"
        "ld1rw {z13.s}, p1/z, [x1, #12]\n\t"
        "ld1rw {z14.s}, p1/z, [x2, #12]\n\t"
        "ld1rw {z15.s}, p1/z, [x3, #12]\n\t"

        "add x4, x4, #512\n\t"
        "add x0, x0, #16\n\t"
        "add x1, x1, #16\n\t"
        "add x2, x2, #16\n\t"
        "add x3, x3, #16\n\t"
        "subs x5, x5, #1\n\t"
        "b.ne 1b\n\t"

        // Store S interleaved: z24=row0, z25=row1, z26=row2, z27=row3
        // We need to transpose and store as [col, row]
        // Use ZIP to interleave: zip1/zip2 pairs of vectors
        // z24 = [r0c0, r0c1, r0c2, ...r0c15]
        // z25 = [r1c0, r1c1, r1c2, ...r1c15]
        // Want: [r0c0, r1c0, r2c0, r3c0, r0c1, r1c1, r2c1, r3c1, ...]

        // Interleave row 0,1 and row 2,3
        "zip1 z0.s, z24.s, z25.s\n\t"  // [r0c0,r1c0,r0c1,r1c1,r0c2,r1c2,...]
        "zip2 z1.s, z24.s, z25.s\n\t"  // [r0c8,r1c8,r0c9,r1c9,...]
        "zip1 z2.s, z26.s, z27.s\n\t"  // [r2c0,r3c0,r2c1,r3c1,...]
        "zip2 z3.s, z26.s, z27.s\n\t"  // [r2c8,r3c8,...]

        // Now interleave pairs to get final layout
        "zip1 z4.d, z0.d, z2.d\n\t"   // [r0c0,r1c0,r2c0,r3c0,r0c1,r1c1,r2c1,r3c1]
        "zip2 z5.d, z0.d, z2.d\n\t"   // [r0c2,r1c2,r2c2,r3c2,...]
        "zip1 z6.d, z1.d, z3.d\n\t"   // [r0c8,...]
        "zip2 z7.d, z1.d, z3.d\n\t"   // [r0c10,...]

        "mov x6, %[s]\n\t"
        "st1w {z4.s}, p1, [x6]\n\t"
        "st1w {z5.s}, p1, [x6, #1, mul vl]\n\t"
        "st1w {z6.s}, p1, [x6, #2, mul vl]\n\t"
        "st1w {z7.s}, p1, [x6, #3, mul vl]\n\t"

        // Phase 2: P@V with optimized P loads
        // P is now interleaved: P[col*4+row] = S[row, col]
        // Single base pointer, consecutive ld1rw
        "mov x7, %[v]\n\t"
        "mov x8, %[o]\n\t"
        "mov x9, #4\n\t"

        "2:\n\t"
        // P base - single pointer, consecutive access
        "mov x0, %[s]\n\t"

        "dup z24.s, #0\n\t"
        "dup z25.s, #0\n\t"
        "dup z26.s, #0\n\t"
        "dup z27.s, #0\n\t"

        "mov x10, x7\n\t"
        "mov x5, #4\n\t"

        "3:\n\t"
        // V loads
        "ld1b {z0.b}, p0/z, [x10]\n\t"
        "ld1b {z1.b}, p0/z, [x10, #1, mul vl]\n\t"
        "ld1b {z2.b}, p0/z, [x10, #2, mul vl]\n\t"
        "ld1b {z3.b}, p0/z, [x10, #3, mul vl]\n\t"
        // P loads - consecutive with single base!
        "ld1rw {z4.s}, p1/z, [x0]\n\t"
        "ld1rw {z5.s}, p1/z, [x0, #4]\n\t"
        "ld1rw {z6.s}, p1/z, [x0, #8]\n\t"
        "ld1rw {z7.s}, p1/z, [x0, #12]\n\t"

        "add x10, x10, #1024\n\t"

        "ld1b {z8.b}, p0/z, [x10]\n\t"
        "ld1b {z9.b}, p0/z, [x10, #1, mul vl]\n\t"
        "ld1b {z10.b}, p0/z, [x10, #2, mul vl]\n\t"
        "ld1b {z11.b}, p0/z, [x10, #3, mul vl]\n\t"
        "ld1rw {z12.s}, p1/z, [x0, #16]\n\t"
        "ld1rw {z13.s}, p1/z, [x0, #20]\n\t"
        "ld1rw {z14.s}, p1/z, [x0, #24]\n\t"
        "ld1rw {z15.s}, p1/z, [x0, #28]\n\t"

        "add x10, x10, #1024\n\t"

        "ld1b {z0.b}, p0/z, [x10]\n\t"
        "ld1b {z1.b}, p0/z, [x10, #1, mul vl]\n\t"
        "ld1b {z2.b}, p0/z, [x10, #2, mul vl]\n\t"
        "ld1b {z3.b}, p0/z, [x10, #3, mul vl]\n\t"
        "ld1rw {z4.s}, p1/z, [x0, #32]\n\t"
        "ld1rw {z5.s}, p1/z, [x0, #36]\n\t"
        "ld1rw {z6.s}, p1/z, [x0, #40]\n\t"
        "ld1rw {z7.s}, p1/z, [x0, #44]\n\t"

        "add x10, x10, #1024\n\t"

        "ld1b {z8.b}, p0/z, [x10]\n\t"
        "ld1b {z9.b}, p0/z, [x10, #1, mul vl]\n\t"
        "ld1b {z10.b}, p0/z, [x10, #2, mul vl]\n\t"
        "ld1b {z11.b}, p0/z, [x10, #3, mul vl]\n\t"
        "ld1rw {z12.s}, p1/z, [x0, #48]\n\t"
        "ld1rw {z13.s}, p1/z, [x0, #52]\n\t"
        "ld1rw {z14.s}, p1/z, [x0, #56]\n\t"
        "ld1rw {z15.s}, p1/z, [x0, #60]\n\t"

        "add x10, x10, #1024\n\t"
        "add x0, x0, #64\n\t"
        "subs x5, x5, #1\n\t"
        "b.ne 3b\n\t"

        "st1w {z24.s}, p1, [x8]\n\t"
        "add x8, x8, #1024\n\t"
        "st1w {z25.s}, p1, [x8]\n\t"
        "add x8, x8, #1024\n\t"
        "st1w {z26.s}, p1, [x8]\n\t"
        "add x8, x8, #1024\n\t"
        "st1w {z27.s}, p1, [x8]\n\t"
        "sub x8, x8, #3008\n\t"

        "add x7, x7, #256\n\t"
        "subs x9, x9, #1\n\t"
        "b.ne 2b\n\t"
        :
        : [q] "r"(Q), [k] "r"(K_int), [s] "r"(S), [v] "r"(V_t), [o] "r"(O)
        : "x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10",
          "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7",
          "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15",
          "z24", "z25", "z26", "z27",
          "p0", "p1", "memory"
    );
}

// With sector cache hints for K/V streaming
// Use PRFM with PSTL1STRM hint for streaming data
void fused_sector_cache(
    const int8_t* Q,
    const int8_t* K_int,
    int32_t* S,
    const int8_t* V_t,
    int32_t* O
) {
    __asm__ volatile(
        "ptrue p0.b\n\t"
        "ptrue p1.s\n\t"

        // Phase 1: Q@K^T with K streaming prefetch
        "mov x0, %[q]\n\t"
        "add x1, x0, #256\n\t"
        "add x2, x0, #512\n\t"
        "add x3, x0, #768\n\t"
        "mov x4, %[k]\n\t"

        "dup z24.s, #0\n\t"
        "dup z25.s, #0\n\t"
        "dup z26.s, #0\n\t"
        "dup z27.s, #0\n\t"

        "mov x5, #16\n\t"
        "1:\n\t"
        // Prefetch next K block as streaming (non-temporal)
        "prfm pldl1strm, [x4, #1024]\n\t"
        "prfm pldl1strm, [x4, #1280]\n\t"

        "ld1b {z0.b}, p0/z, [x4]\n\t"
        "ld1b {z1.b}, p0/z, [x4, #1, mul vl]\n\t"
        "ld1b {z2.b}, p0/z, [x4, #2, mul vl]\n\t"
        "ld1b {z3.b}, p0/z, [x4, #3, mul vl]\n\t"
        "ld1rw {z4.s}, p1/z, [x0]\n\t"
        "ld1rw {z5.s}, p1/z, [x1]\n\t"
        "ld1rw {z6.s}, p1/z, [x2]\n\t"
        "ld1rw {z7.s}, p1/z, [x3]\n\t"

        "ld1b {z8.b}, p0/z, [x4, #4, mul vl]\n\t"
        "ld1b {z9.b}, p0/z, [x4, #5, mul vl]\n\t"
        "ld1b {z10.b}, p0/z, [x4, #6, mul vl]\n\t"
        "ld1b {z11.b}, p0/z, [x4, #7, mul vl]\n\t"
        "ld1rw {z12.s}, p1/z, [x0, #4]\n\t"
        "ld1rw {z13.s}, p1/z, [x1, #4]\n\t"
        "ld1rw {z14.s}, p1/z, [x2, #4]\n\t"
        "ld1rw {z15.s}, p1/z, [x3, #4]\n\t"

        "add x4, x4, #512\n\t"

        "ld1b {z0.b}, p0/z, [x4]\n\t"
        "ld1b {z1.b}, p0/z, [x4, #1, mul vl]\n\t"
        "ld1b {z2.b}, p0/z, [x4, #2, mul vl]\n\t"
        "ld1b {z3.b}, p0/z, [x4, #3, mul vl]\n\t"
        "ld1rw {z4.s}, p1/z, [x0, #8]\n\t"
        "ld1rw {z5.s}, p1/z, [x1, #8]\n\t"
        "ld1rw {z6.s}, p1/z, [x2, #8]\n\t"
        "ld1rw {z7.s}, p1/z, [x3, #8]\n\t"

        "ld1b {z8.b}, p0/z, [x4, #4, mul vl]\n\t"
        "ld1b {z9.b}, p0/z, [x4, #5, mul vl]\n\t"
        "ld1b {z10.b}, p0/z, [x4, #6, mul vl]\n\t"
        "ld1b {z11.b}, p0/z, [x4, #7, mul vl]\n\t"
        "ld1rw {z12.s}, p1/z, [x0, #12]\n\t"
        "ld1rw {z13.s}, p1/z, [x1, #12]\n\t"
        "ld1rw {z14.s}, p1/z, [x2, #12]\n\t"
        "ld1rw {z15.s}, p1/z, [x3, #12]\n\t"

        "add x4, x4, #512\n\t"
        "add x0, x0, #16\n\t"
        "add x1, x1, #16\n\t"
        "add x2, x2, #16\n\t"
        "add x3, x3, #16\n\t"
        "subs x5, x5, #1\n\t"
        "b.ne 1b\n\t"

        // Store S interleaved
        "zip1 z0.s, z24.s, z25.s\n\t"
        "zip2 z1.s, z24.s, z25.s\n\t"
        "zip1 z2.s, z26.s, z27.s\n\t"
        "zip2 z3.s, z26.s, z27.s\n\t"
        "zip1 z4.d, z0.d, z2.d\n\t"
        "zip2 z5.d, z0.d, z2.d\n\t"
        "zip1 z6.d, z1.d, z3.d\n\t"
        "zip2 z7.d, z1.d, z3.d\n\t"

        "mov x6, %[s]\n\t"
        "st1w {z4.s}, p1, [x6]\n\t"
        "st1w {z5.s}, p1, [x6, #1, mul vl]\n\t"
        "st1w {z6.s}, p1, [x6, #2, mul vl]\n\t"
        "st1w {z7.s}, p1, [x6, #3, mul vl]\n\t"

        // Phase 2: P@V with V streaming prefetch
        "mov x7, %[v]\n\t"
        "mov x8, %[o]\n\t"
        "mov x9, #4\n\t"

        "2:\n\t"
        "mov x0, %[s]\n\t"

        "dup z24.s, #0\n\t"
        "dup z25.s, #0\n\t"
        "dup z26.s, #0\n\t"
        "dup z27.s, #0\n\t"

        "mov x10, x7\n\t"
        "mov x5, #4\n\t"

        "3:\n\t"
        // Prefetch V as streaming
        "prfm pldl1strm, [x10, #4096]\n\t"

        "ld1b {z0.b}, p0/z, [x10]\n\t"
        "ld1b {z1.b}, p0/z, [x10, #1, mul vl]\n\t"
        "ld1b {z2.b}, p0/z, [x10, #2, mul vl]\n\t"
        "ld1b {z3.b}, p0/z, [x10, #3, mul vl]\n\t"
        "ld1rw {z4.s}, p1/z, [x0]\n\t"
        "ld1rw {z5.s}, p1/z, [x0, #4]\n\t"
        "ld1rw {z6.s}, p1/z, [x0, #8]\n\t"
        "ld1rw {z7.s}, p1/z, [x0, #12]\n\t"

        "add x10, x10, #1024\n\t"

        "ld1b {z8.b}, p0/z, [x10]\n\t"
        "ld1b {z9.b}, p0/z, [x10, #1, mul vl]\n\t"
        "ld1b {z10.b}, p0/z, [x10, #2, mul vl]\n\t"
        "ld1b {z11.b}, p0/z, [x10, #3, mul vl]\n\t"
        "ld1rw {z12.s}, p1/z, [x0, #16]\n\t"
        "ld1rw {z13.s}, p1/z, [x0, #20]\n\t"
        "ld1rw {z14.s}, p1/z, [x0, #24]\n\t"
        "ld1rw {z15.s}, p1/z, [x0, #28]\n\t"

        "add x10, x10, #1024\n\t"

        "ld1b {z0.b}, p0/z, [x10]\n\t"
        "ld1b {z1.b}, p0/z, [x10, #1, mul vl]\n\t"
        "ld1b {z2.b}, p0/z, [x10, #2, mul vl]\n\t"
        "ld1b {z3.b}, p0/z, [x10, #3, mul vl]\n\t"
        "ld1rw {z4.s}, p1/z, [x0, #32]\n\t"
        "ld1rw {z5.s}, p1/z, [x0, #36]\n\t"
        "ld1rw {z6.s}, p1/z, [x0, #40]\n\t"
        "ld1rw {z7.s}, p1/z, [x0, #44]\n\t"

        "add x10, x10, #1024\n\t"

        "ld1b {z8.b}, p0/z, [x10]\n\t"
        "ld1b {z9.b}, p0/z, [x10, #1, mul vl]\n\t"
        "ld1b {z10.b}, p0/z, [x10, #2, mul vl]\n\t"
        "ld1b {z11.b}, p0/z, [x10, #3, mul vl]\n\t"
        "ld1rw {z12.s}, p1/z, [x0, #48]\n\t"
        "ld1rw {z13.s}, p1/z, [x0, #52]\n\t"
        "ld1rw {z14.s}, p1/z, [x0, #56]\n\t"
        "ld1rw {z15.s}, p1/z, [x0, #60]\n\t"

        "add x10, x10, #1024\n\t"
        "add x0, x0, #64\n\t"
        "subs x5, x5, #1\n\t"
        "b.ne 3b\n\t"

        "st1w {z24.s}, p1, [x8]\n\t"
        "add x8, x8, #1024\n\t"
        "st1w {z25.s}, p1, [x8]\n\t"
        "add x8, x8, #1024\n\t"
        "st1w {z26.s}, p1, [x8]\n\t"
        "add x8, x8, #1024\n\t"
        "st1w {z27.s}, p1, [x8]\n\t"
        "sub x8, x8, #3008\n\t"

        "add x7, x7, #256\n\t"
        "subs x9, x9, #1\n\t"
        "b.ne 2b\n\t"
        :
        : [q] "r"(Q), [k] "r"(K_int), [s] "r"(S), [v] "r"(V_t), [o] "r"(O)
        : "x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10",
          "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7",
          "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15",
          "z24", "z25", "z26", "z27",
          "p0", "p1", "memory"
    );
}

typedef void (*kernel_fn)(const int8_t*, const int8_t*, int32_t*, const int8_t*, int32_t*);

void bench(const char* name, kernel_fn fn,
           const int8_t* Q, const int8_t* K, int32_t* S, const int8_t* V, int32_t* O,
           int iters, int warmup) {
    for (int i = 0; i < warmup; i++) fn(Q, K, S, V, O);

    uint64_t start = rdtsc();
    for (int i = 0; i < iters; i++) fn(Q, K, S, V, O);
    uint64_t end = rdtsc();

    double ticks = (double)(end - start) / iters;
    double cycles = ticks * 20.0;

    size_t bytes_loaded = 16384 + 1024 + 16384 + 256;
    size_t bytes_stored = 256 + 1024;

    double load_bw = bytes_loaded / cycles;
    double eff = 100.0 * load_bw / 128.0;

    printf("%-25s %7.1f cy  %5.1f B/cy  %5.1f%%\n", name, cycles, load_bw, eff);
}

int main() {
    printf("==============================================\n");
    printf("Fused Attention Optimizations\n");
    printf("==============================================\n");
    printf("1. Baseline: row-major S, strided P loads\n");
    printf("2. Interleaved S: optimized ld1rw for P\n");
    printf("3. Sector cache: streaming hints for K/V\n\n");

    printf("Theoretical min: 266 cycles (34KB / 128 B/cy)\n\n");

    int8_t* Q = (int8_t*)aligned_alloc(256, 1024);
    int8_t* K = (int8_t*)aligned_alloc(256, 16384);
    int32_t* S = (int32_t*)aligned_alloc(256, 1024);
    int8_t* V = (int8_t*)aligned_alloc(256, 16384);
    int32_t* O = (int32_t*)aligned_alloc(256, 4096);

    memset(Q, 1, 1024);
    memset(K, 2, 16384);
    memset(S, 0, 1024);
    memset(V, 3, 16384);
    memset(O, 0, 4096);

    printf("%-25s %7s     %5s       %s\n", "Kernel", "Cycles", "BW", "Eff");
    printf("------------------------- -------     -----       -----\n");

    bench("baseline", fused_baseline, Q, K, S, V, O, 10000, 1000);
    bench("interleaved_s", fused_interleaved_s, Q, K, S, V, O, 10000, 1000);
    bench("sector_cache", fused_sector_cache, Q, K, S, V, O, 10000, 1000);

    free(Q); free(K); free(S); free(V); free(O);
    return 0;
}
