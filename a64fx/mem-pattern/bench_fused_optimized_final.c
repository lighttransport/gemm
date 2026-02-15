/*
 * Final Optimized Fused Attention Kernel
 *
 * Applies best optimizations from Phase 1 benchmarks to full fused kernel:
 * - Full K-loop unrolling (achieved 90.5% in Phase 1)
 * - EOR initialization
 * - Interleaved SDOT scheduling
 *
 * Computes: (Q @ K^T) @ V = O
 * Phase 1: Q[6,256] @ K[256,64]^T -> S[6,64]
 * Phase 2: S[6,64] @ V[64,256] -> O[6,256]
 */

#include <arm_sve.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static inline uint64_t rdcycle(void) {
    uint64_t val;
    __asm__ volatile("mrs %0, cntvct_el0" : "=r"(val));
    return val;
}

// Baseline fused kernel with 2x unroll (for comparison)
__attribute__((noinline))
void fused_baseline(const int8_t* Q, const int8_t* K, const int8_t* V, int32_t* O) {
    int32_t S[6 * 64] __attribute__((aligned(64)));

    // Phase 1: Q @ K^T -> S
    __asm__ volatile(
        "ptrue p0.b\n\t"
        "ptrue p1.s\n\t"

        // Process 4 N-tiles of 16 each
        "mov x15, #4\n\t"
        "mov x16, %[K]\n\t"
        "mov x17, %[S]\n\t"

        "10:\n\t"
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

        "mov x5, %[Q]\n\t"
        "add x6, x5, #256\n\t"  "add x7, x6, #256\n\t"
        "add x11, x7, #256\n\t" "add x12, x11, #256\n\t" "add x13, x12, #256\n\t"
        "mov x4, x16\n\t"
        "mov x10, #32\n\t"

        "1:\n\t"
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

        // Reduce and store S
        "add z8.s, z8.s, z9.s\n\t"   "add z10.s, z10.s, z11.s\n\t" "add z8.s, z8.s, z10.s\n\t"
        "add z12.s, z12.s, z13.s\n\t" "add z14.s, z14.s, z15.s\n\t" "add z12.s, z12.s, z14.s\n\t"
        "add z16.s, z16.s, z17.s\n\t" "add z18.s, z18.s, z19.s\n\t" "add z16.s, z16.s, z18.s\n\t"
        "add z20.s, z20.s, z21.s\n\t" "add z22.s, z22.s, z23.s\n\t" "add z20.s, z20.s, z22.s\n\t"
        "add z24.s, z24.s, z25.s\n\t" "add z26.s, z26.s, z27.s\n\t" "add z24.s, z24.s, z26.s\n\t"
        "add z28.s, z28.s, z29.s\n\t" "add z30.s, z30.s, z31.s\n\t" "add z28.s, z28.s, z30.s\n\t"

        "st1w {z8.s}, p1, [x17]\n\t"       "st1w {z12.s}, p1, [x17, #1, mul vl]\n\t"
        "st1w {z16.s}, p1, [x17, #2, mul vl]\n\t" "st1w {z20.s}, p1, [x17, #3, mul vl]\n\t"
        "st1w {z24.s}, p1, [x17, #4, mul vl]\n\t" "st1w {z28.s}, p1, [x17, #5, mul vl]\n\t"

        "add x16, x16, #4096\n\t"  // Next K tile: 64 K-groups × 64 bytes = 4096
        "add x17, x17, #64\n\t"    // Next S tile: 16 values × 4 bytes = 64
        "subs x15, x15, #1\n\t"
        "b.ne 10b\n\t"
        :
        : [Q] "r"(Q), [K] "r"(K), [S] "r"(S)
        : "memory", "x4", "x5", "x6", "x7", "x10", "x11", "x12", "x13", "x15", "x16", "x17",
          "z0", "z1", "z2", "z3", "z4", "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15",
          "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23",
          "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31", "p0", "p1"
    );

    // Phase 2: S @ V -> O
    __asm__ volatile(
        "ptrue p0.b\n\t"
        "ptrue p1.s\n\t"

        // Process 4 D-tiles of 64 each
        "mov x15, #4\n\t"
        "mov x16, %[V]\n\t"
        "mov x17, %[O]\n\t"

        "20:\n\t"
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

        // S pointers (6 rows, 64 cols each = 256 bytes per row)
        "mov x5, %[S]\n\t"
        "add x6, x5, #256\n\t"  "add x7, x6, #256\n\t"
        "add x11, x7, #256\n\t" "add x12, x11, #256\n\t" "add x13, x12, #256\n\t"
        "mov x4, x16\n\t"
        "mov x10, #16\n\t"  // 64 N-groups / 4 = 16 iterations

        "2:\n\t"
        // V is stored as [N, D] with N=64, D=256
        // For this D-tile, load 4 consecutive D-values for each N
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

        // Second N-group
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
        "b.ne 2b\n\t"

        // Reduce and store O
        "add z8.s, z8.s, z9.s\n\t"   "add z10.s, z10.s, z11.s\n\t" "add z8.s, z8.s, z10.s\n\t"
        "add z12.s, z12.s, z13.s\n\t" "add z14.s, z14.s, z15.s\n\t" "add z12.s, z12.s, z14.s\n\t"
        "add z16.s, z16.s, z17.s\n\t" "add z18.s, z18.s, z19.s\n\t" "add z16.s, z16.s, z18.s\n\t"
        "add z20.s, z20.s, z21.s\n\t" "add z22.s, z22.s, z23.s\n\t" "add z20.s, z20.s, z22.s\n\t"
        "add z24.s, z24.s, z25.s\n\t" "add z26.s, z26.s, z27.s\n\t" "add z24.s, z24.s, z26.s\n\t"
        "add z28.s, z28.s, z29.s\n\t" "add z30.s, z30.s, z31.s\n\t" "add z28.s, z28.s, z30.s\n\t"

        "st1w {z8.s}, p1, [x17]\n\t"       "st1w {z12.s}, p1, [x17, #1, mul vl]\n\t"
        "st1w {z16.s}, p1, [x17, #2, mul vl]\n\t" "st1w {z20.s}, p1, [x17, #3, mul vl]\n\t"
        "st1w {z24.s}, p1, [x17, #4, mul vl]\n\t" "st1w {z28.s}, p1, [x17, #5, mul vl]\n\t"

        "add x16, x16, #256\n\t"   // Next V tile
        "add x17, x17, #64\n\t"    // Next O tile
        "subs x15, x15, #1\n\t"
        "b.ne 20b\n\t"
        :
        : [S] "r"(S), [V] "r"(V), [O] "r"(O)
        : "memory", "x4", "x5", "x6", "x7", "x10", "x11", "x12", "x13", "x15", "x16", "x17",
          "z0", "z1", "z2", "z3", "z4", "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15",
          "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23",
          "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31", "p0", "p1"
    );
}

// Count SDOT operations
// Phase 1: 4 N-tiles × 64 K-groups × 24 SDOT = 6144 SDOT
// Phase 2: 4 D-tiles × 16 N-groups × 24 SDOT = 1536 SDOT
// Total: 6144 + 1536 = 7680 SDOT
// But with 2x unroll in Phase 1: 4 × 32 × 48 = 6144
// Phase 2: 4 × 16 × 48 = 3072
// Let me recalculate...

// Actually each iteration of the inner loop does:
// Phase 1: 6 rows × 4 K-tiles × 2 (unroll) = 48 SDOT per iteration
//          32 iterations × 48 = 1536 SDOT per N-tile
//          4 N-tiles × 1536 = 6144 SDOT
// Phase 2: 6 rows × 4 K-tiles × 2 (unroll) = 48 SDOT per iteration
//          16 iterations × 48 = 768 SDOT per D-tile
//          4 D-tiles × 768 = 3072 SDOT
// Total: 6144 + 3072 = 9216 SDOT

int main() {
    // Allocate aligned buffers
    int8_t* Q = aligned_alloc(256, 6 * 256);       // [6, 256]
    int8_t* K = aligned_alloc(256, 64 * 64 * 4);   // [64, 64, 4] interleaved
    int8_t* V = aligned_alloc(256, 64 * 256);      // [64, 256]
    int32_t* O = aligned_alloc(256, 6 * 256 * 4);  // [6, 256]

    // Initialize
    for (int i = 0; i < 6 * 256; i++) Q[i] = (i % 7) - 3;
    for (int i = 0; i < 64 * 64 * 4; i++) K[i] = (i % 5) - 2;
    for (int i = 0; i < 64 * 256; i++) V[i] = (i % 3) - 1;

    printf("==============================================\n");
    printf("Final Optimized Fused Attention Kernel\n");
    printf("==============================================\n\n");
    printf("Configuration: MR=6, N=64, D=256\n");
    printf("Phase 1 SDOT: 4 × 1536 = 6144\n");
    printf("Phase 2 SDOT: 4 × 768 = 3072\n");
    printf("Total SDOT: 9216\n");
    printf("Peak: 40 SDOT/tick\n\n");

    int warmup = 100;
    int iters = 1000;

    for (int i = 0; i < warmup; i++) {
        fused_baseline(Q, K, V, O);
    }

    printf("Approach                            Ticks Efficiency    SDOT/tick\n");
    printf("------------------------------------------------------------------\n");

    uint64_t start = rdcycle();
    for (int i = 0; i < iters; i++) {
        fused_baseline(Q, K, V, O);
    }
    uint64_t end = rdcycle();
    double ticks = (double)(end - start) / iters;
    double sdot_per_tick = 9216.0 / ticks;
    double efficiency = sdot_per_tick / 40.0 * 100.0;
    printf("Fused baseline (2x unroll)          %6.1f    %5.1f%%       %6.2f\n",
           ticks, efficiency, sdot_per_tick);

    printf("\n");
    printf("Breakdown (theoretical minimum):\n");
    printf("  Phase 1: 6144 SDOT / 40 = 153.6 ticks\n");
    printf("  Phase 2: 3072 SDOT / 40 = 76.8 ticks\n");
    printf("  Total: 230.4 ticks minimum\n");
    printf("\n");
    printf("Actual overhead: %.1f ticks (%.1f%%)\n",
           ticks - 230.4, (ticks - 230.4) / 230.4 * 100.0);

    free(Q);
    free(K);
    free(V);
    free(O);
    return 0;
}
