// bench_sdot_nodep.c
// SDOT without dependency chains - use separate accumulators per K vector
// Then merge at the end to avoid 9-cycle SDOT latency stalls

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

// Fixed: Use 16 separate accumulators (4 rows × 4 K-vectors)
// Then merge at the end
void qkt_no_dep(
    const int8_t* Q,
    const int8_t* K,
    int32_t* S
) {
    __asm__ volatile(
        "ptrue p0.b\n\t"
        "ptrue p1.s\n\t"

        "mov x0, %[q]\n\t"
        "add x1, x0, #256\n\t"
        "add x2, x0, #512\n\t"
        "add x3, x0, #768\n\t"
        "mov x4, %[k]\n\t"

        // 16 accumulators: row0_k0,k1,k2,k3, row1_k0,k1,k2,k3, ...
        "dup z16.s, #0\n\t"  // row0, K[0]
        "dup z17.s, #0\n\t"  // row0, K[1]
        "dup z18.s, #0\n\t"  // row0, K[2]
        "dup z19.s, #0\n\t"  // row0, K[3]
        "dup z20.s, #0\n\t"  // row1, K[0]
        "dup z21.s, #0\n\t"  // row1, K[1]
        "dup z22.s, #0\n\t"  // row1, K[2]
        "dup z23.s, #0\n\t"  // row1, K[3]
        "dup z24.s, #0\n\t"  // row2, K[0]
        "dup z25.s, #0\n\t"  // row2, K[1]
        "dup z26.s, #0\n\t"  // row2, K[2]
        "dup z27.s, #0\n\t"  // row2, K[3]
        "dup z28.s, #0\n\t"  // row3, K[0]
        "dup z29.s, #0\n\t"  // row3, K[1]
        "dup z30.s, #0\n\t"  // row3, K[2]
        "dup z31.s, #0\n\t"  // row3, K[3]

        "mov x5, #64\n\t"
        "1:\n\t"
        // Load K vectors
        "ld1b {z0.b}, p0/z, [x4]\n\t"
        "ld1b {z1.b}, p0/z, [x4, #1, mul vl]\n\t"
        "ld1b {z2.b}, p0/z, [x4, #2, mul vl]\n\t"
        "ld1b {z3.b}, p0/z, [x4, #3, mul vl]\n\t"

        // Load Q broadcasts
        "ld1rw {z4.s}, p1/z, [x0]\n\t"
        "ld1rw {z5.s}, p1/z, [x1]\n\t"
        "ld1rw {z6.s}, p1/z, [x2]\n\t"
        "ld1rw {z7.s}, p1/z, [x3]\n\t"

        // SDOT: Each to different accumulator - NO DEPENDENCIES!
        // Row 0: z16-z19 accumulate K[0-3]
        "sdot z16.s, z0.b, z4.b\n\t"  // row0 += K[0] * Q[0]
        "sdot z17.s, z1.b, z4.b\n\t"  // row0 += K[1] * Q[0]
        "sdot z18.s, z2.b, z4.b\n\t"  // row0 += K[2] * Q[0]
        "sdot z19.s, z3.b, z4.b\n\t"  // row0 += K[3] * Q[0]

        // Row 1: z20-z23
        "sdot z20.s, z0.b, z5.b\n\t"
        "sdot z21.s, z1.b, z5.b\n\t"
        "sdot z22.s, z2.b, z5.b\n\t"
        "sdot z23.s, z3.b, z5.b\n\t"

        // Row 2: z24-z27
        "sdot z24.s, z0.b, z6.b\n\t"
        "sdot z25.s, z1.b, z6.b\n\t"
        "sdot z26.s, z2.b, z6.b\n\t"
        "sdot z27.s, z3.b, z6.b\n\t"

        // Row 3: z28-z31
        "sdot z28.s, z0.b, z7.b\n\t"
        "sdot z29.s, z1.b, z7.b\n\t"
        "sdot z30.s, z2.b, z7.b\n\t"
        "sdot z31.s, z3.b, z7.b\n\t"

        "add x4, x4, #256\n\t"
        "add x0, x0, #4\n\t"
        "add x1, x1, #4\n\t"
        "add x2, x2, #4\n\t"
        "add x3, x3, #4\n\t"
        "subs x5, x5, #1\n\t"
        "b.ne 1b\n\t"

        // Merge accumulators: row0 = z16+z17+z18+z19, etc.
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

        // Store S
        "st1w {z16.s}, p1, [%[s]]\n\t"
        "st1w {z20.s}, p1, [%[s], #1, mul vl]\n\t"
        "st1w {z24.s}, p1, [%[s], #2, mul vl]\n\t"
        "st1w {z28.s}, p1, [%[s], #3, mul vl]\n\t"
        :
        : [q] "r"(Q), [k] "r"(K), [s] "r"(S)
        : "x0", "x1", "x2", "x3", "x4", "x5",
          "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7",
          "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23",
          "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31",
          "p0", "p1", "memory"
    );
}

// Interleaved version with no dependencies
void qkt_interleaved_nodep(
    const int8_t* Q,
    const int8_t* K,
    int32_t* S
) {
    __asm__ volatile(
        "ptrue p0.b\n\t"
        "ptrue p1.s\n\t"

        "mov x0, %[q]\n\t"
        "add x1, x0, #256\n\t"
        "add x2, x0, #512\n\t"
        "add x3, x0, #768\n\t"
        "mov x4, %[k]\n\t"

        // 16 accumulators
        "dup z16.s, #0\n\t"
        "dup z17.s, #0\n\t"
        "dup z18.s, #0\n\t"
        "dup z19.s, #0\n\t"
        "dup z20.s, #0\n\t"
        "dup z21.s, #0\n\t"
        "dup z22.s, #0\n\t"
        "dup z23.s, #0\n\t"
        "dup z24.s, #0\n\t"
        "dup z25.s, #0\n\t"
        "dup z26.s, #0\n\t"
        "dup z27.s, #0\n\t"
        "dup z28.s, #0\n\t"
        "dup z29.s, #0\n\t"
        "dup z30.s, #0\n\t"
        "dup z31.s, #0\n\t"

        // Preload
        "ld1b {z0.b}, p0/z, [x4]\n\t"
        "ld1rw {z4.s}, p1/z, [x0]\n\t"

        "mov x5, #64\n\t"
        "1:\n\t"
        // Interleave: load, sdot, load, sdot...
        "ld1b {z1.b}, p0/z, [x4, #1, mul vl]\n\t"
        "ld1rw {z5.s}, p1/z, [x1]\n\t"
        "sdot z16.s, z0.b, z4.b\n\t"

        "ld1b {z2.b}, p0/z, [x4, #2, mul vl]\n\t"
        "ld1rw {z6.s}, p1/z, [x2]\n\t"
        "sdot z20.s, z0.b, z5.b\n\t"

        "ld1b {z3.b}, p0/z, [x4, #3, mul vl]\n\t"
        "ld1rw {z7.s}, p1/z, [x3]\n\t"
        "sdot z24.s, z0.b, z6.b\n\t"

        "sdot z28.s, z0.b, z7.b\n\t"
        "sdot z17.s, z1.b, z4.b\n\t"
        "sdot z21.s, z1.b, z5.b\n\t"

        "sdot z25.s, z1.b, z6.b\n\t"
        "sdot z29.s, z1.b, z7.b\n\t"
        "sdot z18.s, z2.b, z4.b\n\t"

        "sdot z22.s, z2.b, z5.b\n\t"
        "sdot z26.s, z2.b, z6.b\n\t"
        "sdot z30.s, z2.b, z7.b\n\t"

        "add x4, x4, #256\n\t"
        "sdot z19.s, z3.b, z4.b\n\t"
        "sdot z23.s, z3.b, z5.b\n\t"
        "sdot z27.s, z3.b, z6.b\n\t"

        "ld1b {z0.b}, p0/z, [x4]\n\t"
        "add x0, x0, #4\n\t"
        "sdot z31.s, z3.b, z7.b\n\t"

        "ld1rw {z4.s}, p1/z, [x0]\n\t"
        "add x1, x1, #4\n\t"
        "add x2, x2, #4\n\t"
        "add x3, x3, #4\n\t"
        "subs x5, x5, #1\n\t"
        "b.ne 1b\n\t"

        // Merge
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

        // Store
        "st1w {z16.s}, p1, [%[s]]\n\t"
        "st1w {z20.s}, p1, [%[s], #1, mul vl]\n\t"
        "st1w {z24.s}, p1, [%[s], #2, mul vl]\n\t"
        "st1w {z28.s}, p1, [%[s], #3, mul vl]\n\t"
        :
        : [q] "r"(Q), [k] "r"(K), [s] "r"(S)
        : "x0", "x1", "x2", "x3", "x4", "x5",
          "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7",
          "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23",
          "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31",
          "p0", "p1", "memory"
    );
}

// Full fused with no dependencies
void fused_nodep(
    const int8_t* Q,
    const int8_t* K,
    int32_t* S,
    const int8_t* V,
    int32_t* O
) {
    __asm__ volatile(
        "ptrue p0.b\n\t"
        "ptrue p1.s\n\t"

        // ============ Phase 1: Q@K^T ============
        "mov x0, %[q]\n\t"
        "add x1, x0, #256\n\t"
        "add x2, x0, #512\n\t"
        "add x3, x0, #768\n\t"
        "mov x4, %[k]\n\t"

        "dup z16.s, #0\n\t"
        "dup z17.s, #0\n\t"
        "dup z18.s, #0\n\t"
        "dup z19.s, #0\n\t"
        "dup z20.s, #0\n\t"
        "dup z21.s, #0\n\t"
        "dup z22.s, #0\n\t"
        "dup z23.s, #0\n\t"
        "dup z24.s, #0\n\t"
        "dup z25.s, #0\n\t"
        "dup z26.s, #0\n\t"
        "dup z27.s, #0\n\t"
        "dup z28.s, #0\n\t"
        "dup z29.s, #0\n\t"
        "dup z30.s, #0\n\t"
        "dup z31.s, #0\n\t"

        "ld1b {z0.b}, p0/z, [x4]\n\t"
        "ld1rw {z4.s}, p1/z, [x0]\n\t"

        "mov x5, #64\n\t"
        "1:\n\t"
        "ld1b {z1.b}, p0/z, [x4, #1, mul vl]\n\t"
        "ld1rw {z5.s}, p1/z, [x1]\n\t"
        "sdot z16.s, z0.b, z4.b\n\t"

        "ld1b {z2.b}, p0/z, [x4, #2, mul vl]\n\t"
        "ld1rw {z6.s}, p1/z, [x2]\n\t"
        "sdot z20.s, z0.b, z5.b\n\t"

        "ld1b {z3.b}, p0/z, [x4, #3, mul vl]\n\t"
        "ld1rw {z7.s}, p1/z, [x3]\n\t"
        "sdot z24.s, z0.b, z6.b\n\t"

        "sdot z28.s, z0.b, z7.b\n\t"
        "sdot z17.s, z1.b, z4.b\n\t"
        "sdot z21.s, z1.b, z5.b\n\t"

        "sdot z25.s, z1.b, z6.b\n\t"
        "sdot z29.s, z1.b, z7.b\n\t"
        "sdot z18.s, z2.b, z4.b\n\t"

        "sdot z22.s, z2.b, z5.b\n\t"
        "sdot z26.s, z2.b, z6.b\n\t"
        "sdot z30.s, z2.b, z7.b\n\t"

        "add x4, x4, #256\n\t"
        "sdot z19.s, z3.b, z4.b\n\t"
        "sdot z23.s, z3.b, z5.b\n\t"
        "sdot z27.s, z3.b, z6.b\n\t"

        "ld1b {z0.b}, p0/z, [x4]\n\t"
        "add x0, x0, #4\n\t"
        "sdot z31.s, z3.b, z7.b\n\t"

        "ld1rw {z4.s}, p1/z, [x0]\n\t"
        "add x1, x1, #4\n\t"
        "add x2, x2, #4\n\t"
        "add x3, x3, #4\n\t"
        "subs x5, x5, #1\n\t"
        "b.ne 1b\n\t"

        // Merge Phase 1
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

        // Store S
        "st1w {z16.s}, p1, [%[s]]\n\t"
        "st1w {z20.s}, p1, [%[s], #1, mul vl]\n\t"
        "st1w {z24.s}, p1, [%[s], #2, mul vl]\n\t"
        "st1w {z28.s}, p1, [%[s], #3, mul vl]\n\t"

        // ============ Phase 2: P@V ============
        "mov x7, %[v]\n\t"
        "mov x8, %[o]\n\t"
        "mov x9, #4\n\t"

        "2:\n\t"
        "mov x0, %[s]\n\t"
        "add x1, x0, #64\n\t"
        "add x2, x0, #128\n\t"
        "add x3, x0, #192\n\t"

        "dup z16.s, #0\n\t"
        "dup z17.s, #0\n\t"
        "dup z18.s, #0\n\t"
        "dup z19.s, #0\n\t"
        "dup z20.s, #0\n\t"
        "dup z21.s, #0\n\t"
        "dup z22.s, #0\n\t"
        "dup z23.s, #0\n\t"
        "dup z24.s, #0\n\t"
        "dup z25.s, #0\n\t"
        "dup z26.s, #0\n\t"
        "dup z27.s, #0\n\t"
        "dup z28.s, #0\n\t"
        "dup z29.s, #0\n\t"
        "dup z30.s, #0\n\t"
        "dup z31.s, #0\n\t"

        "mov x10, x7\n\t"
        "ld1b {z0.b}, p0/z, [x10]\n\t"
        "ld1rw {z4.s}, p1/z, [x0]\n\t"

        "mov x5, #16\n\t"
        "3:\n\t"
        "ld1b {z1.b}, p0/z, [x10, #1, mul vl]\n\t"
        "ld1rw {z5.s}, p1/z, [x1]\n\t"
        "sdot z16.s, z0.b, z4.b\n\t"

        "ld1b {z2.b}, p0/z, [x10, #2, mul vl]\n\t"
        "ld1rw {z6.s}, p1/z, [x2]\n\t"
        "sdot z20.s, z0.b, z5.b\n\t"

        "ld1b {z3.b}, p0/z, [x10, #3, mul vl]\n\t"
        "ld1rw {z7.s}, p1/z, [x3]\n\t"
        "sdot z24.s, z0.b, z6.b\n\t"

        "sdot z28.s, z0.b, z7.b\n\t"
        "sdot z17.s, z1.b, z4.b\n\t"
        "sdot z21.s, z1.b, z5.b\n\t"

        "sdot z25.s, z1.b, z6.b\n\t"
        "sdot z29.s, z1.b, z7.b\n\t"
        "sdot z18.s, z2.b, z4.b\n\t"

        "sdot z22.s, z2.b, z5.b\n\t"
        "sdot z26.s, z2.b, z6.b\n\t"
        "sdot z30.s, z2.b, z7.b\n\t"

        "add x10, x10, #1024\n\t"
        "sdot z19.s, z3.b, z4.b\n\t"
        "sdot z23.s, z3.b, z5.b\n\t"
        "sdot z27.s, z3.b, z6.b\n\t"

        "ld1b {z0.b}, p0/z, [x10]\n\t"
        "add x0, x0, #4\n\t"
        "sdot z31.s, z3.b, z7.b\n\t"

        "ld1rw {z4.s}, p1/z, [x0]\n\t"
        "add x1, x1, #4\n\t"
        "add x2, x2, #4\n\t"
        "add x3, x3, #4\n\t"
        "subs x5, x5, #1\n\t"
        "b.ne 3b\n\t"

        // Merge Phase 2
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

        // Store O
        "st1w {z16.s}, p1, [x8]\n\t"
        "add x8, x8, #1024\n\t"
        "st1w {z20.s}, p1, [x8]\n\t"
        "add x8, x8, #1024\n\t"
        "st1w {z24.s}, p1, [x8]\n\t"
        "add x8, x8, #1024\n\t"
        "st1w {z28.s}, p1, [x8]\n\t"
        "sub x8, x8, #3008\n\t"

        "add x7, x7, #256\n\t"
        "subs x9, x9, #1\n\t"
        "b.ne 2b\n\t"
        :
        : [q] "r"(Q), [k] "r"(K), [s] "r"(S), [v] "r"(V), [o] "r"(O)
        : "x0", "x1", "x2", "x3", "x4", "x5", "x7", "x8", "x9", "x10",
          "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7",
          "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23",
          "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31",
          "p0", "p1", "memory"
    );
}

int main() {
    printf("==============================================\n");
    printf("SDOT Without Dependency Chains\n");
    printf("==============================================\n");
    printf("Use 16 accumulators (4 rows x 4 K-vectors)\n");
    printf("No SDOT->SDOT dependencies within iteration\n\n");

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

    int iters = 10000;
    int warmup = 1000;
    uint64_t start, end;
    double ticks, cycles;

    printf("=== Q@K^T Phase Only ===\n");
    printf("Load: 17KB, Compute: 1024 SDOT\n");
    printf("Theoretical: 136 cy (load) or 256 cy (compute @ 4 SDOT/cy)\n\n");

    printf("%-25s %8s %10s %10s\n", "Kernel", "Cycles", "Ld B/cy", "SDOT/cy");
    printf("----------------------------------------------------------\n");

    for (int i = 0; i < warmup; i++) qkt_no_dep(Q, K, S);
    start = rdtsc();
    for (int i = 0; i < iters; i++) qkt_no_dep(Q, K, S);
    end = rdtsc();
    ticks = (double)(end - start) / iters;
    cycles = ticks * 20.0;
    printf("%-25s %8.1f %10.1f %10.1f\n", "no_dep", cycles, 17408.0/cycles, 1024.0/cycles);

    for (int i = 0; i < warmup; i++) qkt_interleaved_nodep(Q, K, S);
    start = rdtsc();
    for (int i = 0; i < iters; i++) qkt_interleaved_nodep(Q, K, S);
    end = rdtsc();
    ticks = (double)(end - start) / iters;
    cycles = ticks * 20.0;
    printf("%-25s %8.1f %10.1f %10.1f\n", "interleaved_nodep", cycles, 17408.0/cycles, 1024.0/cycles);

    printf("\n=== Full Fused Attention ===\n");
    printf("Load: 34KB, Compute: 2048 SDOT\n");
    printf("Theoretical: 266 cy (load) or 512 cy (compute)\n\n");

    for (int i = 0; i < warmup; i++) fused_nodep(Q, K, S, V, O);
    start = rdtsc();
    for (int i = 0; i < iters; i++) fused_nodep(Q, K, S, V, O);
    end = rdtsc();
    ticks = (double)(end - start) / iters;
    cycles = ticks * 20.0;
    printf("%-25s %8.1f cy  %6.1f ld B/cy  %5.1f SDOT/cy  %.1f%%\n",
           "fused_nodep", cycles, 34816.0/cycles, 2048.0/cycles, 100.0*266.0/cycles);

    free(Q); free(K); free(S); free(V); free(O);
    return 0;
}
