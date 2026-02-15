/*
 * Online Softmax / Chunked Processing for Flash Attention
 *
 * Key insight: Instead of computing all of Phase 1, then Phase 2,
 * process N dimension in chunks:
 *
 * Traditional:
 *   S[6,64] = Q[6,256] @ K[256,64]      // Full Phase 1
 *   P[6,64] = softmax(S)                 // Full softmax
 *   O[6,256] = P[6,64] @ V[64,256]      // Full Phase 2
 *
 * Chunked (N_chunk = 16):
 *   for n_chunk in [0, 16, 32, 48]:
 *     S_chunk[6,16] = Q @ K_chunk[256,16]      // Partial Phase 1
 *     // Update running max/sum for online softmax
 *     P_chunk = online_softmax(S_chunk)
 *     O += P_chunk @ V_chunk[16,256]           // Partial Phase 2
 *
 * Benefits:
 *   - S_chunk fits in registers (6×16 = 96 int32, but we process 6×4=24 at a time)
 *   - No need to store full S matrix
 *   - Better cache utilization (reuse Q, process K/V in streaming fashion)
 *   - Can fuse quantization with computation
 *
 * This benchmark tests the chunked approach for Phase 2 efficiency.
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
// Chunked Phase 2: Process N in chunks of 16
// ============================================================
// Instead of: 4 D-tile iterations, each processing all 16 N values
// Do: 1 N-chunk at a time, processing all 4 D-tiles
//
// This changes the loop order from D-tile major to N-chunk major
// Benefit: Can accumulate across N-chunks without reduction overhead

__attribute__((noinline))
void phase2_nchunk16(const int32_t* S, const int8_t* V, int32_t* O) {
    // Process N=64 as 4 chunks of 16
    // Each chunk: compute partial O for all D-tiles, accumulate

    __asm__ volatile(
        "ptrue p0.b\n\t"
        "ptrue p1.s\n\t"

        // Zero all output accumulators (6 rows × 4 D-tiles × 4 K-tiles = 96 values)
        // But we only have 32 registers, so we'll process D-tiles sequentially
        // within each N-chunk, but accumulate across N-chunks

        // D-tile 0 accumulators: z8-z11 (4 K-tiles)
        // D-tile 1 accumulators: z12-z15
        // D-tile 2 accumulators: z16-z19
        // D-tile 3 accumulators: z20-z23
        // That's only 16 registers for 4 D-tiles × 4 K-tiles for 1 row
        // For 6 rows, need 96 accumulators - doesn't fit!

        // Alternative: Process 1 row at a time, all D-tiles
        // Row accumulators: z8-z23 (4 D-tiles × 4 K-tiles = 16 registers)
        // V vectors: z0-z3 (4 K-tiles)
        // S broadcast: z4
        // Spare: z24-z31 for next row or other use

        // Actually, let's try a different approach:
        // Process 2 rows × 4 D-tiles at once
        // 2 rows × 4 D-tiles × 4 K-tiles = 32 accumulators (exactly fits!)

        // Initialize output to zero (we'll accumulate across N-chunks)
        "mov x17, %[O]\n\t"
        "eor z0.d, z0.d, z0.d\n\t"
        "mov x10, #24\n\t"  // 6 rows × 4 D-tiles = 24 stores
        "1:\n\t"
        "st1w {z0.s}, p1, [x17]\n\t"
        "add x17, x17, #64\n\t"
        "subs x10, x10, #1\n\t"
        "b.ne 1b\n\t"

        // Process N in 4 chunks of 16
        "mov x14, #4\n\t"           // N-chunk counter
        "mov x18, %[V]\n\t"         // V base
        "mov x19, %[S]\n\t"         // S base

        "5:\n\t"  // N-chunk loop
        // For each N-chunk, process all 4 D-tiles for all 6 rows

        // Process rows 2 at a time (fits in registers)
        "mov x20, #3\n\t"           // Row-pair counter (0-1, 2-3, 4-5)
        "mov x5, x19\n\t"           // S pointer for this N-chunk, row 0
        "mov x17, %[O]\n\t"         // O pointer

        "6:\n\t"  // Row-pair loop
        // Zero accumulators for 2 rows × 4 D-tiles × 4 K-tiles = 32 regs
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

        "add x6, x5, #256\n\t"      // S pointer for row 1

        // Inner loop: 16 N values (8 iterations × 2 unroll)
        "mov x4, x18\n\t"           // V pointer for this N-chunk
        "mov x10, #8\n\t"

        "7:\n\t"
        // Load all 4 D-tiles of V (16 vectors total)
        "ld1b {z0.b}, p0/z, [x4]\n\t"
        "ld1b {z1.b}, p0/z, [x4, #1, mul vl]\n\t"
        "ld1b {z2.b}, p0/z, [x4, #2, mul vl]\n\t"
        "ld1b {z3.b}, p0/z, [x4, #3, mul vl]\n\t"

        // Row 0: accumulate into z8-z11, z12-z15, z16-z19, z20-z23
        // Row 1: accumulate into z24-z27, z28-z31 (need more regs!)

        // Actually this doesn't fit. Let me simplify to 1 row at a time.
        // Load S for row 0
        "ld1rw {z4.s}, p1/z, [x5]\n\t"
        "sdot z8.s, z0.b, z4.b\n\t"   "sdot z9.s, z1.b, z4.b\n\t"
        "sdot z10.s, z2.b, z4.b\n\t"  "sdot z11.s, z3.b, z4.b\n\t"
        // Load S for row 1
        "ld1rw {z4.s}, p1/z, [x6]\n\t"
        "sdot z12.s, z0.b, z4.b\n\t"  "sdot z13.s, z1.b, z4.b\n\t"
        "sdot z14.s, z2.b, z4.b\n\t"  "sdot z15.s, z3.b, z4.b\n\t"

        "add x4, x4, #1024\n\t"

        // Second N of unroll
        "ld1b {z0.b}, p0/z, [x4]\n\t"
        "ld1b {z1.b}, p0/z, [x4, #1, mul vl]\n\t"
        "ld1b {z2.b}, p0/z, [x4, #2, mul vl]\n\t"
        "ld1b {z3.b}, p0/z, [x4, #3, mul vl]\n\t"
        "ld1rw {z4.s}, p1/z, [x5, #4]\n\t"
        "sdot z8.s, z0.b, z4.b\n\t"   "sdot z9.s, z1.b, z4.b\n\t"
        "sdot z10.s, z2.b, z4.b\n\t"  "sdot z11.s, z3.b, z4.b\n\t"
        "ld1rw {z4.s}, p1/z, [x6, #4]\n\t"
        "sdot z12.s, z0.b, z4.b\n\t"  "sdot z13.s, z1.b, z4.b\n\t"
        "sdot z14.s, z2.b, z4.b\n\t"  "sdot z15.s, z3.b, z4.b\n\t"

        "add x4, x4, #1024\n\t"
        "add x5, x5, #8\n\t"
        "add x6, x6, #8\n\t"
        "subs x10, x10, #1\n\t"
        "b.ne 7b\n\t"

        // Reduction for 2 rows × 4 D-tiles
        // Row 0 D-tiles: z8+z9, z10+z11 -> need to reduce to 4 values
        "add z8.s, z8.s, z9.s\n\t"    "add z10.s, z10.s, z11.s\n\t"
        "add z12.s, z12.s, z13.s\n\t" "add z14.s, z14.s, z15.s\n\t"

        // Accumulate into output (load existing, add, store)
        "ld1w {z0.s}, p1/z, [x17]\n\t"
        "add z8.s, z8.s, z0.s\n\t"
        "st1w {z8.s}, p1, [x17]\n\t"
        "add x17, x17, #64\n\t"

        "ld1w {z0.s}, p1/z, [x17]\n\t"
        "add z10.s, z10.s, z0.s\n\t"
        "st1w {z10.s}, p1, [x17]\n\t"
        "add x17, x17, #64\n\t"

        // Skip to next row's D-tiles
        "add x17, x17, #192\n\t"  // Skip 3 more D-tiles to get to next row

        "ld1w {z0.s}, p1/z, [x17]\n\t"
        "add z12.s, z12.s, z0.s\n\t"
        "st1w {z12.s}, p1, [x17]\n\t"
        "add x17, x17, #64\n\t"

        "ld1w {z0.s}, p1/z, [x17]\n\t"
        "add z14.s, z14.s, z0.s\n\t"
        "st1w {z14.s}, p1, [x17]\n\t"

        // Move S pointers to next row pair
        "add x5, x5, #448\n\t"  // Skip to row+2 (256*2 - 64 already advanced)

        "subs x20, x20, #1\n\t"
        "b.ne 6b\n\t"

        // Move to next N-chunk
        "add x18, x18, #16384\n\t"  // V stride for 16 N values
        "add x19, x19, #64\n\t"     // S stride for 16 N values (16 × 4 bytes)
        "subs x14, x14, #1\n\t"
        "b.ne 5b\n\t"
        :
        : [S] "r"(S), [V] "r"(V), [O] "r"(O)
        : "memory", "x4", "x5", "x6", "x10", "x14", "x17", "x18", "x19", "x20",
          "z0", "z1", "z2", "z3", "z4",
          "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15",
          "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23",
          "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31", "p0", "p1"
    );
}

// ============================================================
// Baseline Phase 2 for comparison
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
// Simpler approach: Just reduce K-tiles from 4 to 2
// ============================================================
// Process 2 K-tiles at a time, doing 2 inner loops
// This simplifies reduction without changing outer structure

__attribute__((noinline))
void phase2_reduced_ktile(const int32_t* S, const int8_t* V, int32_t* O) {
    // Process D-tiles with only 2 K-tiles per pass
    // First pass: K-tiles 0,1
    // Second pass: K-tiles 2,3

    __asm__ volatile(
        "ptrue p0.b\n\t"
        "ptrue p1.s\n\t"

        // ===== Pass 1: K-tiles 0,1 =====
        "mov x15, #4\n\t"
        "mov x16, %[V]\n\t"
        "mov x17, %[O]\n\t"
        "2:\n\t"
        // Zero 12 accumulators (2 K-tiles × 6 rows)
        "eor z8.d, z8.d, z8.d\n\t"   "eor z9.d, z9.d, z9.d\n\t"
        "eor z10.d, z10.d, z10.d\n\t" "eor z11.d, z11.d, z11.d\n\t"
        "eor z12.d, z12.d, z12.d\n\t" "eor z13.d, z13.d, z13.d\n\t"
        "eor z14.d, z14.d, z14.d\n\t" "eor z15.d, z15.d, z15.d\n\t"
        "eor z16.d, z16.d, z16.d\n\t" "eor z17.d, z17.d, z17.d\n\t"
        "eor z18.d, z18.d, z18.d\n\t" "eor z19.d, z19.d, z19.d\n\t"

        "mov x5, %[S]\n\t"
        "add x6, x5, #256\n\t"  "add x7, x5, #512\n\t"
        "add x11, x5, #768\n\t" "add x12, x5, #1024\n\t" "add x13, x5, #1280\n\t"
        "mov x4, x16\n\t"
        "mov x10, #8\n\t"
        "3:\n\t"
        // Only load K-tiles 0,1
        "ld1b {z0.b}, p0/z, [x4]\n\t"
        "ld1b {z1.b}, p0/z, [x4, #1, mul vl]\n\t"
        "ld1rw {z4.s}, p1/z, [x5]\n\t"
        "sdot z8.s, z0.b, z4.b\n\t"  "sdot z9.s, z1.b, z4.b\n\t"
        "ld1rw {z4.s}, p1/z, [x6]\n\t"
        "sdot z10.s, z0.b, z4.b\n\t" "sdot z11.s, z1.b, z4.b\n\t"
        "ld1rw {z4.s}, p1/z, [x7]\n\t"
        "sdot z12.s, z0.b, z4.b\n\t" "sdot z13.s, z1.b, z4.b\n\t"
        "ld1rw {z4.s}, p1/z, [x11]\n\t"
        "sdot z14.s, z0.b, z4.b\n\t" "sdot z15.s, z1.b, z4.b\n\t"
        "ld1rw {z4.s}, p1/z, [x12]\n\t"
        "sdot z16.s, z0.b, z4.b\n\t" "sdot z17.s, z1.b, z4.b\n\t"
        "ld1rw {z4.s}, p1/z, [x13]\n\t"
        "sdot z18.s, z0.b, z4.b\n\t" "sdot z19.s, z1.b, z4.b\n\t"
        "add x4, x4, #1024\n\t"

        // Second N of unroll
        "ld1b {z0.b}, p0/z, [x4]\n\t"
        "ld1b {z1.b}, p0/z, [x4, #1, mul vl]\n\t"
        "ld1rw {z4.s}, p1/z, [x5, #4]\n\t"
        "sdot z8.s, z0.b, z4.b\n\t"  "sdot z9.s, z1.b, z4.b\n\t"
        "ld1rw {z4.s}, p1/z, [x6, #4]\n\t"
        "sdot z10.s, z0.b, z4.b\n\t" "sdot z11.s, z1.b, z4.b\n\t"
        "ld1rw {z4.s}, p1/z, [x7, #4]\n\t"
        "sdot z12.s, z0.b, z4.b\n\t" "sdot z13.s, z1.b, z4.b\n\t"
        "ld1rw {z4.s}, p1/z, [x11, #4]\n\t"
        "sdot z14.s, z0.b, z4.b\n\t" "sdot z15.s, z1.b, z4.b\n\t"
        "ld1rw {z4.s}, p1/z, [x12, #4]\n\t"
        "sdot z16.s, z0.b, z4.b\n\t" "sdot z17.s, z1.b, z4.b\n\t"
        "ld1rw {z4.s}, p1/z, [x13, #4]\n\t"
        "sdot z18.s, z0.b, z4.b\n\t" "sdot z19.s, z1.b, z4.b\n\t"

        "add x4, x4, #1024\n\t"
        "add x5, x5, #8\n\t"   "add x6, x6, #8\n\t"   "add x7, x7, #8\n\t"
        "add x11, x11, #8\n\t" "add x12, x12, #8\n\t" "add x13, x13, #8\n\t"
        "subs x10, x10, #1\n\t"
        "b.ne 3b\n\t"

        // Simple reduction: 1 ADD per row
        "add z8.s, z8.s, z9.s\n\t"
        "add z10.s, z10.s, z11.s\n\t"
        "add z12.s, z12.s, z13.s\n\t"
        "add z14.s, z14.s, z15.s\n\t"
        "add z16.s, z16.s, z17.s\n\t"
        "add z18.s, z18.s, z19.s\n\t"

        // Store partial results
        "st1w {z8.s}, p1, [x17]\n\t"  "add x17, x17, #1024\n\t"
        "st1w {z10.s}, p1, [x17]\n\t" "add x17, x17, #1024\n\t"
        "st1w {z12.s}, p1, [x17]\n\t" "add x17, x17, #1024\n\t"
        "st1w {z14.s}, p1, [x17]\n\t" "add x17, x17, #1024\n\t"
        "st1w {z16.s}, p1, [x17]\n\t" "add x17, x17, #1024\n\t"
        "st1w {z18.s}, p1, [x17]\n\t"
        "sub x17, x17, #4096\n\t" "sub x17, x17, #1024\n\t"
        "add x17, x17, #64\n\t"
        "add x16, x16, #64\n\t"
        "subs x15, x15, #1\n\t"
        "b.ne 2b\n\t"

        // ===== Pass 2: K-tiles 2,3 =====
        "mov x15, #4\n\t"
        "mov x16, %[V]\n\t"
        "add x16, x16, #128\n\t"  // Start at K-tile 2
        "mov x17, %[O]\n\t"
        "4:\n\t"
        // Zero 12 accumulators
        "eor z8.d, z8.d, z8.d\n\t"   "eor z9.d, z9.d, z9.d\n\t"
        "eor z10.d, z10.d, z10.d\n\t" "eor z11.d, z11.d, z11.d\n\t"
        "eor z12.d, z12.d, z12.d\n\t" "eor z13.d, z13.d, z13.d\n\t"
        "eor z14.d, z14.d, z14.d\n\t" "eor z15.d, z15.d, z15.d\n\t"
        "eor z16.d, z16.d, z16.d\n\t" "eor z17.d, z17.d, z17.d\n\t"
        "eor z18.d, z18.d, z18.d\n\t" "eor z19.d, z19.d, z19.d\n\t"

        "mov x5, %[S]\n\t"
        "add x6, x5, #256\n\t"  "add x7, x5, #512\n\t"
        "add x11, x5, #768\n\t" "add x12, x5, #1024\n\t" "add x13, x5, #1280\n\t"
        "mov x4, x16\n\t"
        "mov x10, #8\n\t"
        "5:\n\t"
        // Load K-tiles 2,3
        "ld1b {z0.b}, p0/z, [x4]\n\t"
        "ld1b {z1.b}, p0/z, [x4, #1, mul vl]\n\t"
        "ld1rw {z4.s}, p1/z, [x5]\n\t"
        "sdot z8.s, z0.b, z4.b\n\t"  "sdot z9.s, z1.b, z4.b\n\t"
        "ld1rw {z4.s}, p1/z, [x6]\n\t"
        "sdot z10.s, z0.b, z4.b\n\t" "sdot z11.s, z1.b, z4.b\n\t"
        "ld1rw {z4.s}, p1/z, [x7]\n\t"
        "sdot z12.s, z0.b, z4.b\n\t" "sdot z13.s, z1.b, z4.b\n\t"
        "ld1rw {z4.s}, p1/z, [x11]\n\t"
        "sdot z14.s, z0.b, z4.b\n\t" "sdot z15.s, z1.b, z4.b\n\t"
        "ld1rw {z4.s}, p1/z, [x12]\n\t"
        "sdot z16.s, z0.b, z4.b\n\t" "sdot z17.s, z1.b, z4.b\n\t"
        "ld1rw {z4.s}, p1/z, [x13]\n\t"
        "sdot z18.s, z0.b, z4.b\n\t" "sdot z19.s, z1.b, z4.b\n\t"
        "add x4, x4, #1024\n\t"

        "ld1b {z0.b}, p0/z, [x4]\n\t"
        "ld1b {z1.b}, p0/z, [x4, #1, mul vl]\n\t"
        "ld1rw {z4.s}, p1/z, [x5, #4]\n\t"
        "sdot z8.s, z0.b, z4.b\n\t"  "sdot z9.s, z1.b, z4.b\n\t"
        "ld1rw {z4.s}, p1/z, [x6, #4]\n\t"
        "sdot z10.s, z0.b, z4.b\n\t" "sdot z11.s, z1.b, z4.b\n\t"
        "ld1rw {z4.s}, p1/z, [x7, #4]\n\t"
        "sdot z12.s, z0.b, z4.b\n\t" "sdot z13.s, z1.b, z4.b\n\t"
        "ld1rw {z4.s}, p1/z, [x11, #4]\n\t"
        "sdot z14.s, z0.b, z4.b\n\t" "sdot z15.s, z1.b, z4.b\n\t"
        "ld1rw {z4.s}, p1/z, [x12, #4]\n\t"
        "sdot z16.s, z0.b, z4.b\n\t" "sdot z17.s, z1.b, z4.b\n\t"
        "ld1rw {z4.s}, p1/z, [x13, #4]\n\t"
        "sdot z18.s, z0.b, z4.b\n\t" "sdot z19.s, z1.b, z4.b\n\t"

        "add x4, x4, #1024\n\t"
        "add x5, x5, #8\n\t"   "add x6, x6, #8\n\t"   "add x7, x7, #8\n\t"
        "add x11, x11, #8\n\t" "add x12, x12, #8\n\t" "add x13, x13, #8\n\t"
        "subs x10, x10, #1\n\t"
        "b.ne 5b\n\t"

        // Reduction and accumulate
        "add z8.s, z8.s, z9.s\n\t"
        "add z10.s, z10.s, z11.s\n\t"
        "add z12.s, z12.s, z13.s\n\t"
        "add z14.s, z14.s, z15.s\n\t"
        "add z16.s, z16.s, z17.s\n\t"
        "add z18.s, z18.s, z19.s\n\t"

        // Load, add, store
        "ld1w {z0.s}, p1/z, [x17]\n\t"
        "add z8.s, z8.s, z0.s\n\t"
        "st1w {z8.s}, p1, [x17]\n\t"  "add x17, x17, #1024\n\t"
        "ld1w {z0.s}, p1/z, [x17]\n\t"
        "add z10.s, z10.s, z0.s\n\t"
        "st1w {z10.s}, p1, [x17]\n\t" "add x17, x17, #1024\n\t"
        "ld1w {z0.s}, p1/z, [x17]\n\t"
        "add z12.s, z12.s, z0.s\n\t"
        "st1w {z12.s}, p1, [x17]\n\t" "add x17, x17, #1024\n\t"
        "ld1w {z0.s}, p1/z, [x17]\n\t"
        "add z14.s, z14.s, z0.s\n\t"
        "st1w {z14.s}, p1, [x17]\n\t" "add x17, x17, #1024\n\t"
        "ld1w {z0.s}, p1/z, [x17]\n\t"
        "add z16.s, z16.s, z0.s\n\t"
        "st1w {z16.s}, p1, [x17]\n\t" "add x17, x17, #1024\n\t"
        "ld1w {z0.s}, p1/z, [x17]\n\t"
        "add z18.s, z18.s, z0.s\n\t"
        "st1w {z18.s}, p1, [x17]\n\t"
        "sub x17, x17, #4096\n\t" "sub x17, x17, #1024\n\t"
        "add x17, x17, #64\n\t"
        "add x16, x16, #64\n\t"
        "subs x15, x15, #1\n\t"
        "b.ne 4b\n\t"
        :
        : [S] "r"(S), [V] "r"(V), [O] "r"(O)
        : "memory", "x4", "x5", "x6", "x7", "x10", "x11", "x12", "x13",
          "x15", "x16", "x17",
          "z0", "z1", "z4", "z8", "z9", "z10", "z11", "z12", "z13",
          "z14", "z15", "z16", "z17", "z18", "z19", "p0", "p1"
    );
}

int main() {
    printf("==============================================\n");
    printf("Phase 2 Optimization: Alternative Approaches\n");
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

    printf("Testing different Phase 2 optimization strategies:\n\n");

    // Warmup
    for (int i = 0; i < 100; i++) {
        phase2_baseline(S, V, O1);
        phase2_reduced_ktile(S, V, O2);
    }

    // Benchmark baseline
    uint64_t start = rdcycle();
    for (int i = 0; i < iters; i++) {
        phase2_baseline(S, V, O1);
    }
    uint64_t end = rdcycle();
    double ticks_base = (double)(end - start) / iters;
    double eff_base = (phase2_sdot / ticks_base) / peak_sdot_per_tick * 100;

    // Benchmark reduced K-tile
    start = rdcycle();
    for (int i = 0; i < iters; i++) {
        phase2_reduced_ktile(S, V, O2);
    }
    end = rdcycle();
    double ticks_rkt = (double)(end - start) / iters;
    double eff_rkt = (phase2_sdot / ticks_rkt) / peak_sdot_per_tick * 100;

    printf("%-30s %10s %10s\n", "Approach", "Ticks", "Efficiency");
    printf("------------------------------------------------------\n");
    printf("%-30s %10.1f %9.1f%%\n", "Baseline (4 K-tiles)", ticks_base, eff_base);
    printf("%-30s %10.1f %9.1f%%\n", "Reduced K-tile (2×2)", ticks_rkt, eff_rkt);

    printf("\n");
    printf("Key insight: The bottleneck may not be reduction overhead.\n");
    printf("Possible true bottlenecks:\n");
    printf("  1. Memory bandwidth (V loads not fully cached)\n");
    printf("  2. SDOT latency not fully hidden\n");
    printf("  3. Branch misprediction or pipeline stalls\n");
    printf("\n");
    printf("Next steps:\n");
    printf("  1. Profile with fapp to identify exact bottleneck\n");
    printf("  2. Test with prefetch hints for V loads\n");
    printf("  3. Consider sector cache for streaming V access\n");

    free(S); free(V); free(O1); free(O2); free(O3);
    return 0;
}
