/*
 * Fused GEMM benchmark: MR=6 single vs MR=3×2 tiles
 *
 * Fused attention: (Q @ K^T) @ V
 *   Phase 1: Q[MR, D] @ K^T[D, N] -> S[MR, N]
 *   Phase 2: S[MR, N] @ V[N, D] -> O[MR, D]
 *
 * Compare:
 *   1. MR=6 single tile (24 SDOT/d_group)
 *   2. MR=3×2 tiles with shared K (24 SDOT/d_group, interleaved)
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

/*
 * MR=6 single tile fused GEMM
 * Phase 1: Q[6,256] @ K^T[256,64] -> S[6,64]
 * Phase 2: S[6,64] @ V[64,256] -> O[6,256]
 */
void fused_mr6_single(
    const int8_t* Q,      // [6, 256]
    const int8_t* K,      // [64, 64, 4] interleaved for N=64
    const int8_t* V,      // [64, 256, 4] interleaved
    int32_t* O,           // [6, 256]
    int D_groups          // 64 for D=256
) {
    // Stack for S matrix [6, 64] as INT32
    int32_t S[6 * 64] __attribute__((aligned(64)));

    __asm__ volatile(
        "ptrue p0.b\n\t"
        "ptrue p1.s\n\t"

        // ============== Phase 1: Q @ K^T -> S ==============
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

        // Q row pointers (6 rows, stride 256)
        "mov x5, %[Q]\n\t"
        "add x6, x5, #256\n\t"
        "add x7, x5, #512\n\t"
        "add x11, x5, #768\n\t"
        "add x12, x5, #1024\n\t"
        "add x13, x5, #1280\n\t"
        "mov x4, %[K]\n\t"
        "mov x10, %[D_groups]\n\t"

        // Phase 1 loop
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

        // Merge 4 accumulators per row -> 1 result vector
        // Row 0: z8 += z9 + z10 + z11
        "add z8.s, z8.s, z9.s\n\t"
        "add z10.s, z10.s, z11.s\n\t"
        "add z8.s, z8.s, z10.s\n\t"
        // Row 1
        "add z12.s, z12.s, z13.s\n\t"
        "add z14.s, z14.s, z15.s\n\t"
        "add z12.s, z12.s, z14.s\n\t"
        // Row 2
        "add z16.s, z16.s, z17.s\n\t"
        "add z18.s, z18.s, z19.s\n\t"
        "add z16.s, z16.s, z18.s\n\t"
        // Row 3
        "add z20.s, z20.s, z21.s\n\t"
        "add z22.s, z22.s, z23.s\n\t"
        "add z20.s, z20.s, z22.s\n\t"
        // Row 4
        "add z24.s, z24.s, z25.s\n\t"
        "add z26.s, z26.s, z27.s\n\t"
        "add z24.s, z24.s, z26.s\n\t"
        // Row 5
        "add z28.s, z28.s, z29.s\n\t"
        "add z30.s, z30.s, z31.s\n\t"
        "add z28.s, z28.s, z30.s\n\t"

        // Store S[6, 64] - each row is 64 INT32 = 256 bytes = 4 vectors
        // For now store first vector of each row
        "mov x14, %[S]\n\t"
        "st1w {z8.s}, p1, [x14]\n\t"
        "st1w {z12.s}, p1, [x14, #1, mul vl]\n\t"
        "st1w {z16.s}, p1, [x14, #2, mul vl]\n\t"
        "st1w {z20.s}, p1, [x14, #3, mul vl]\n\t"
        "st1w {z24.s}, p1, [x14, #4, mul vl]\n\t"
        "st1w {z28.s}, p1, [x14, #5, mul vl]\n\t"

        // ============== Phase 2: S @ V -> O ==============
        // S[6, 64] @ V[64, 256] -> O[6, 256]
        // Process in 4 D-tiles of 64 columns each

        "mov x15, #4\n\t"  // 4 D-tiles for D=256
        "mov x16, %[V]\n\t"
        "mov x17, %[O]\n\t"

        "2:\n\t"  // D-tile loop
        // Zero accumulators for this D-tile
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

        // S row pointers (6 rows, stride 256 bytes = 64 INT32)
        "mov x5, %[S]\n\t"
        "add x6, x5, #256\n\t"
        "add x7, x5, #512\n\t"
        "add x11, x5, #768\n\t"
        "add x12, x5, #1024\n\t"
        "add x13, x5, #1280\n\t"

        "mov x4, x16\n\t"  // V pointer for this D-tile
        "mov x10, #16\n\t"  // 16 N-groups for N=64

        "3:\n\t"  // N-group loop
        // Load V (4 vectors, interleaved)
        "ld1b {z0.b}, p0/z, [x4]\n\t"
        "ld1b {z1.b}, p0/z, [x4, #1, mul vl]\n\t"
        "ld1b {z2.b}, p0/z, [x4, #2, mul vl]\n\t"
        "ld1b {z3.b}, p0/z, [x4, #3, mul vl]\n\t"
        "add x4, x4, #1024\n\t"  // V stride per N-group

        // Load S broadcasts and compute
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

        // Merge and store O for this D-tile
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

        // Store O[6 rows × 1 vector] for this D-tile
        "st1w {z8.s}, p1, [x17]\n\t"
        "add x17, x17, #1024\n\t"  // O row stride
        "st1w {z12.s}, p1, [x17]\n\t"
        "add x17, x17, #1024\n\t"
        "st1w {z16.s}, p1, [x17]\n\t"
        "add x17, x17, #1024\n\t"
        "st1w {z20.s}, p1, [x17]\n\t"
        "add x17, x17, #1024\n\t"
        "st1w {z24.s}, p1, [x17]\n\t"
        "add x17, x17, #1024\n\t"
        "st1w {z28.s}, p1, [x17]\n\t"

        // Advance to next D-tile
        // Reset O ptr: subtract 5×1024 = 5120 (too large for immediate)
        "sub x17, x17, #4096\n\t"
        "sub x17, x17, #1024\n\t"
        "add x17, x17, #64\n\t"    // Next D-tile column
        "add x16, x16, #64\n\t"    // Next V D-tile

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

/*
 * MR=3×2 tiles with shared K - interleaved processing
 * Same 24 SDOTs per d_group, but two independent tile groups
 */
void fused_mr3x2_tiles(
    const int8_t* Q_A,    // [3, 256] tile A
    const int8_t* Q_B,    // [3, 256] tile B
    const int8_t* K,      // [64, 64, 4] shared
    const int8_t* V,      // [64, 256, 4] shared
    int32_t* O_A,         // [3, 256]
    int32_t* O_B,         // [3, 256]
    int D_groups
) {
    int32_t S_A[3 * 64] __attribute__((aligned(64)));
    int32_t S_B[3 * 64] __attribute__((aligned(64)));

    __asm__ volatile(
        "ptrue p0.b\n\t"
        "ptrue p1.s\n\t"

        // ============== Phase 1: Q @ K^T -> S (both tiles) ==============
        // Accumulators: z8-z19 for tile A (12), z20-z31 for tile B (12)

        // Zero accumulators
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

        // Q_A row pointers
        "mov x5, %[Q_A]\n\t"
        "add x6, x5, #256\n\t"
        "add x7, x5, #512\n\t"
        // Q_B row pointers
        "mov x11, %[Q_B]\n\t"
        "add x12, x11, #256\n\t"
        "add x13, x11, #512\n\t"

        "mov x4, %[K]\n\t"
        "mov x10, %[D_groups]\n\t"

        "1:\n\t"
        // Load K (shared)
        "ld1b {z0.b}, p0/z, [x4]\n\t"
        "ld1b {z1.b}, p0/z, [x4, #1, mul vl]\n\t"
        "ld1b {z2.b}, p0/z, [x4, #2, mul vl]\n\t"
        "ld1b {z3.b}, p0/z, [x4, #3, mul vl]\n\t"
        "add x4, x4, #256\n\t"

        // Tile A row 0
        "ld1rw {z4.s}, p1/z, [x5]\n\t"
        "sdot z8.s, z0.b, z4.b\n\t"
        "sdot z9.s, z1.b, z4.b\n\t"
        "sdot z10.s, z2.b, z4.b\n\t"
        "sdot z11.s, z3.b, z4.b\n\t"

        // Tile B row 0 (interleaved)
        "ld1rw {z5.s}, p1/z, [x11]\n\t"
        "sdot z20.s, z0.b, z5.b\n\t"
        "sdot z21.s, z1.b, z5.b\n\t"
        "sdot z22.s, z2.b, z5.b\n\t"
        "sdot z23.s, z3.b, z5.b\n\t"

        // Tile A row 1
        "ld1rw {z4.s}, p1/z, [x6]\n\t"
        "sdot z12.s, z0.b, z4.b\n\t"
        "sdot z13.s, z1.b, z4.b\n\t"
        "sdot z14.s, z2.b, z4.b\n\t"
        "sdot z15.s, z3.b, z4.b\n\t"

        // Tile B row 1
        "ld1rw {z5.s}, p1/z, [x12]\n\t"
        "sdot z24.s, z0.b, z5.b\n\t"
        "sdot z25.s, z1.b, z5.b\n\t"
        "sdot z26.s, z2.b, z5.b\n\t"
        "sdot z27.s, z3.b, z5.b\n\t"

        // Tile A row 2
        "ld1rw {z4.s}, p1/z, [x7]\n\t"
        "sdot z16.s, z0.b, z4.b\n\t"
        "sdot z17.s, z1.b, z4.b\n\t"
        "sdot z18.s, z2.b, z4.b\n\t"
        "sdot z19.s, z3.b, z4.b\n\t"

        // Tile B row 2
        "ld1rw {z5.s}, p1/z, [x13]\n\t"
        "sdot z28.s, z0.b, z5.b\n\t"
        "sdot z29.s, z1.b, z5.b\n\t"
        "sdot z30.s, z2.b, z5.b\n\t"
        "sdot z31.s, z3.b, z5.b\n\t"

        "add x5, x5, #4\n\t"
        "add x6, x6, #4\n\t"
        "add x7, x7, #4\n\t"
        "add x11, x11, #4\n\t"
        "add x12, x12, #4\n\t"
        "add x13, x13, #4\n\t"

        "subs x10, x10, #1\n\t"
        "b.ne 1b\n\t"

        // Merge accumulators for tile A
        "add z8.s, z8.s, z9.s\n\t"
        "add z10.s, z10.s, z11.s\n\t"
        "add z8.s, z8.s, z10.s\n\t"
        "add z12.s, z12.s, z13.s\n\t"
        "add z14.s, z14.s, z15.s\n\t"
        "add z12.s, z12.s, z14.s\n\t"
        "add z16.s, z16.s, z17.s\n\t"
        "add z18.s, z18.s, z19.s\n\t"
        "add z16.s, z16.s, z18.s\n\t"

        // Merge accumulators for tile B
        "add z20.s, z20.s, z21.s\n\t"
        "add z22.s, z22.s, z23.s\n\t"
        "add z20.s, z20.s, z22.s\n\t"
        "add z24.s, z24.s, z25.s\n\t"
        "add z26.s, z26.s, z27.s\n\t"
        "add z24.s, z24.s, z26.s\n\t"
        "add z28.s, z28.s, z29.s\n\t"
        "add z30.s, z30.s, z31.s\n\t"
        "add z28.s, z28.s, z30.s\n\t"

        // Store S_A and S_B
        "mov x14, %[S_A]\n\t"
        "st1w {z8.s}, p1, [x14]\n\t"
        "st1w {z12.s}, p1, [x14, #1, mul vl]\n\t"
        "st1w {z16.s}, p1, [x14, #2, mul vl]\n\t"

        "mov x14, %[S_B]\n\t"
        "st1w {z20.s}, p1, [x14]\n\t"
        "st1w {z24.s}, p1, [x14, #1, mul vl]\n\t"
        "st1w {z28.s}, p1, [x14, #2, mul vl]\n\t"

        // ============== Phase 2: S @ V -> O (both tiles) ==============
        "mov x15, #4\n\t"  // 4 D-tiles
        "mov x16, %[V]\n\t"
        "mov x17, %[O_A]\n\t"
        "mov x18, %[O_B]\n\t"

        "2:\n\t"
        // Zero accumulators
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

        // S_A/S_B pointers
        "mov x5, %[S_A]\n\t"
        "add x6, x5, #256\n\t"
        "add x7, x5, #512\n\t"
        "mov x11, %[S_B]\n\t"
        "add x12, x11, #256\n\t"
        "add x13, x11, #512\n\t"

        "mov x4, x16\n\t"
        "mov x10, #16\n\t"

        "3:\n\t"
        // Load V (shared)
        "ld1b {z0.b}, p0/z, [x4]\n\t"
        "ld1b {z1.b}, p0/z, [x4, #1, mul vl]\n\t"
        "ld1b {z2.b}, p0/z, [x4, #2, mul vl]\n\t"
        "ld1b {z3.b}, p0/z, [x4, #3, mul vl]\n\t"
        "add x4, x4, #1024\n\t"

        // Tile A
        "ld1rw {z4.s}, p1/z, [x5]\n\t"
        "sdot z8.s, z0.b, z4.b\n\t"
        "sdot z9.s, z1.b, z4.b\n\t"
        "sdot z10.s, z2.b, z4.b\n\t"
        "sdot z11.s, z3.b, z4.b\n\t"

        "ld1rw {z5.s}, p1/z, [x11]\n\t"
        "sdot z20.s, z0.b, z5.b\n\t"
        "sdot z21.s, z1.b, z5.b\n\t"
        "sdot z22.s, z2.b, z5.b\n\t"
        "sdot z23.s, z3.b, z5.b\n\t"

        "ld1rw {z4.s}, p1/z, [x6]\n\t"
        "sdot z12.s, z0.b, z4.b\n\t"
        "sdot z13.s, z1.b, z4.b\n\t"
        "sdot z14.s, z2.b, z4.b\n\t"
        "sdot z15.s, z3.b, z4.b\n\t"

        "ld1rw {z5.s}, p1/z, [x12]\n\t"
        "sdot z24.s, z0.b, z5.b\n\t"
        "sdot z25.s, z1.b, z5.b\n\t"
        "sdot z26.s, z2.b, z5.b\n\t"
        "sdot z27.s, z3.b, z5.b\n\t"

        "ld1rw {z4.s}, p1/z, [x7]\n\t"
        "sdot z16.s, z0.b, z4.b\n\t"
        "sdot z17.s, z1.b, z4.b\n\t"
        "sdot z18.s, z2.b, z4.b\n\t"
        "sdot z19.s, z3.b, z4.b\n\t"

        "ld1rw {z5.s}, p1/z, [x13]\n\t"
        "sdot z28.s, z0.b, z5.b\n\t"
        "sdot z29.s, z1.b, z5.b\n\t"
        "sdot z30.s, z2.b, z5.b\n\t"
        "sdot z31.s, z3.b, z5.b\n\t"

        "add x5, x5, #4\n\t"
        "add x6, x6, #4\n\t"
        "add x7, x7, #4\n\t"
        "add x11, x11, #4\n\t"
        "add x12, x12, #4\n\t"
        "add x13, x13, #4\n\t"

        "subs x10, x10, #1\n\t"
        "b.ne 3b\n\t"

        // Merge and store
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

        // Store O_A
        "st1w {z8.s}, p1, [x17]\n\t"
        "add x17, x17, #1024\n\t"
        "st1w {z12.s}, p1, [x17]\n\t"
        "add x17, x17, #1024\n\t"
        "st1w {z16.s}, p1, [x17]\n\t"
        "sub x17, x17, #2048\n\t"
        "add x17, x17, #64\n\t"

        // Store O_B
        "st1w {z20.s}, p1, [x18]\n\t"
        "add x18, x18, #1024\n\t"
        "st1w {z24.s}, p1, [x18]\n\t"
        "add x18, x18, #1024\n\t"
        "st1w {z28.s}, p1, [x18]\n\t"
        "sub x18, x18, #2048\n\t"
        "add x18, x18, #64\n\t"

        "add x16, x16, #64\n\t"

        "subs x15, x15, #1\n\t"
        "b.ne 2b\n\t"

        :
        : [Q_A] "r"(Q_A), [Q_B] "r"(Q_B), [K] "r"(K), [V] "r"(V),
          [O_A] "r"(O_A), [O_B] "r"(O_B),
          [S_A] "r"(S_A), [S_B] "r"(S_B), [D_groups] "r"((long)D_groups)
        : "memory", "x4", "x5", "x6", "x7", "x10", "x11", "x12", "x13",
          "x14", "x15", "x16", "x17", "x18",
          "z0", "z1", "z2", "z3", "z4", "z5",
          "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15",
          "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23",
          "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31",
          "p0", "p1"
    );
}

int main() {
    printf("==============================================\n");
    printf("Fused GEMM: MR=6 Single vs MR=3×2 Tiles\n");
    printf("==============================================\n\n");

    printf("Configuration:\n");
    printf("  Q@K^T: [MR, 256] @ [256, 64] -> S[MR, 64]\n");
    printf("  S@V:   [MR, 64] @ [64, 256] -> O[MR, 256]\n");
    printf("  MR=6 single: 24 SDOT/d_group × 2 phases\n");
    printf("  MR=3×2: 24 SDOT/d_group (interleaved) × 2 phases\n\n");

    int D_groups = 64;
    int N = 64;
    int D = 256;

    // Allocate buffers
    int8_t* Q = aligned_alloc(256, 6 * D);
    int8_t* Q_B = aligned_alloc(256, 3 * D);
    int8_t* K = aligned_alloc(256, D_groups * N * 4);
    int8_t* V = aligned_alloc(256, N * D * 4);
    int32_t* O = aligned_alloc(256, 6 * D * 4);
    int32_t* O_B = aligned_alloc(256, 3 * D * 4);

    memset(Q, 1, 6 * D);
    memset(Q_B, 1, 3 * D);
    memset(K, 1, D_groups * N * 4);
    memset(V, 1, N * D * 4);

    int iters = 5000;
    uint64_t start, end;
    double ticks;

    printf("%-30s %10s %10s %12s\n", "Kernel", "Ticks", "Total SDOT", "SDOT/tick");
    printf("--------------------------------------------------------------\n");

    // Warmup
    for (int i = 0; i < 100; i++) {
        fused_mr6_single(Q, K, V, O, D_groups);
    }

    // MR=6 single
    start = rdcycle();
    for (int i = 0; i < iters; i++) {
        fused_mr6_single(Q, K, V, O, D_groups);
    }
    end = rdcycle();
    ticks = (double)(end - start) / iters;
    // Phase 1: 64 d_groups × 24 SDOT = 1536
    // Phase 2: 4 D-tiles × 16 N-groups × 24 SDOT = 1536
    int total_sdot_mr6 = 1536 + 1536;
    printf("%-30s %10.1f %10d %12.2f\n",
           "MR=6 single (6 rows)", ticks, total_sdot_mr6, total_sdot_mr6/ticks);

    // Warmup
    for (int i = 0; i < 100; i++) {
        fused_mr3x2_tiles(Q, Q_B, K, V, O, O_B, D_groups);
    }

    // MR=3×2 tiles
    start = rdcycle();
    for (int i = 0; i < iters; i++) {
        fused_mr3x2_tiles(Q, Q_B, K, V, O, O_B, D_groups);
    }
    end = rdcycle();
    ticks = (double)(end - start) / iters;
    // Same total: 2 tiles × (1536 + 1536) / 2 = 3072
    int total_sdot_mr3x2 = 3072;
    printf("%-30s %10.1f %10d %12.2f\n",
           "MR=3×2 tiles (6 rows total)", ticks, total_sdot_mr3x2, total_sdot_mr3x2/ticks);

    printf("\n");
    printf("Analysis:\n");
    printf("  MR=6 single processes 6 rows sequentially\n");
    printf("  MR=3×2 processes 6 rows (2×3) with shared K/V loads\n");
    printf("  Shared loads reduce memory traffic, interleaving hides latency\n");

    free(Q); free(Q_B); free(K); free(V); free(O); free(O_B);
    return 0;
}
