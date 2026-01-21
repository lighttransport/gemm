// bench_interleaved_sdot.c
// Interleave loads with SDOT compute to hide latency and avoid port contention
// Based on reference DGEMM kernel pattern from dgemm.kernel.s

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

// Baseline: All loads, then all computes (sequential)
void qkt_sequential(
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

        // Accumulators for 4 rows
        "dup z16.s, #0\n\t"
        "dup z17.s, #0\n\t"
        "dup z18.s, #0\n\t"
        "dup z19.s, #0\n\t"

        "mov x5, #64\n\t"  // 64 d_groups (D=256, 4 bytes per group)
        "1:\n\t"
        // Load K: 4 vectors (all at once)
        "ld1b {z0.b}, p0/z, [x4]\n\t"
        "ld1b {z1.b}, p0/z, [x4, #1, mul vl]\n\t"
        "ld1b {z2.b}, p0/z, [x4, #2, mul vl]\n\t"
        "ld1b {z3.b}, p0/z, [x4, #3, mul vl]\n\t"

        // Load Q: 4 broadcasts (all at once)
        "ld1rw {z4.s}, p1/z, [x0]\n\t"
        "ld1rw {z5.s}, p1/z, [x1]\n\t"
        "ld1rw {z6.s}, p1/z, [x2]\n\t"
        "ld1rw {z7.s}, p1/z, [x3]\n\t"

        // SDOT: all computes (sequential)
        "sdot z16.s, z0.b, z4.b\n\t"
        "sdot z16.s, z1.b, z4.b\n\t"
        "sdot z16.s, z2.b, z4.b\n\t"
        "sdot z16.s, z3.b, z4.b\n\t"

        "sdot z17.s, z0.b, z5.b\n\t"
        "sdot z17.s, z1.b, z5.b\n\t"
        "sdot z17.s, z2.b, z5.b\n\t"
        "sdot z17.s, z3.b, z5.b\n\t"

        "sdot z18.s, z0.b, z6.b\n\t"
        "sdot z18.s, z1.b, z6.b\n\t"
        "sdot z18.s, z2.b, z6.b\n\t"
        "sdot z18.s, z3.b, z6.b\n\t"

        "sdot z19.s, z0.b, z7.b\n\t"
        "sdot z19.s, z1.b, z7.b\n\t"
        "sdot z19.s, z2.b, z7.b\n\t"
        "sdot z19.s, z3.b, z7.b\n\t"

        "add x4, x4, #256\n\t"
        "add x0, x0, #4\n\t"
        "add x1, x1, #4\n\t"
        "add x2, x2, #4\n\t"
        "add x3, x3, #4\n\t"
        "subs x5, x5, #1\n\t"
        "b.ne 1b\n\t"

        // Store S
        "st1w {z16.s}, p1, [%[s]]\n\t"
        "st1w {z17.s}, p1, [%[s], #1, mul vl]\n\t"
        "st1w {z18.s}, p1, [%[s], #2, mul vl]\n\t"
        "st1w {z19.s}, p1, [%[s], #3, mul vl]\n\t"
        :
        : [q] "r"(Q), [k] "r"(K), [s] "r"(S)
        : "x0", "x1", "x2", "x3", "x4", "x5",
          "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7",
          "z16", "z17", "z18", "z19",
          "p0", "p1", "memory"
    );
}

// Interleaved: Loads and SDOT mixed (like DGEMM reference)
// Pattern: ld1b, ld1rw, sdot, ld1b, ld1rw, sdot, ...
void qkt_interleaved(
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

        // Accumulators
        "dup z16.s, #0\n\t"
        "dup z17.s, #0\n\t"
        "dup z18.s, #0\n\t"
        "dup z19.s, #0\n\t"

        // Preload first K vectors
        "ld1b {z0.b}, p0/z, [x4]\n\t"
        "ld1b {z1.b}, p0/z, [x4, #1, mul vl]\n\t"

        "mov x5, #64\n\t"
        "1:\n\t"
        // Interleaved pattern: load K[2], load Q[0], sdot, load K[3], load Q[1], sdot...
        "ld1b {z2.b}, p0/z, [x4, #2, mul vl]\n\t"
        "ld1rw {z4.s}, p1/z, [x0]\n\t"
        "sdot z16.s, z0.b, z4.b\n\t"

        "ld1b {z3.b}, p0/z, [x4, #3, mul vl]\n\t"
        "ld1rw {z5.s}, p1/z, [x1]\n\t"
        "sdot z17.s, z0.b, z5.b\n\t"

        "ld1rw {z6.s}, p1/z, [x2]\n\t"
        "sdot z18.s, z0.b, z6.b\n\t"
        "sdot z16.s, z1.b, z4.b\n\t"

        "ld1rw {z7.s}, p1/z, [x3]\n\t"
        "sdot z19.s, z0.b, z7.b\n\t"
        "sdot z17.s, z1.b, z5.b\n\t"

        "add x4, x4, #256\n\t"
        "sdot z18.s, z1.b, z6.b\n\t"
        "sdot z19.s, z1.b, z7.b\n\t"

        // Preload next K[0], K[1] while computing with K[2], K[3]
        "ld1b {z0.b}, p0/z, [x4]\n\t"
        "sdot z16.s, z2.b, z4.b\n\t"
        "sdot z17.s, z2.b, z5.b\n\t"

        "ld1b {z1.b}, p0/z, [x4, #1, mul vl]\n\t"
        "sdot z18.s, z2.b, z6.b\n\t"
        "sdot z19.s, z2.b, z7.b\n\t"

        "add x0, x0, #4\n\t"
        "sdot z16.s, z3.b, z4.b\n\t"
        "sdot z17.s, z3.b, z5.b\n\t"

        "add x1, x1, #4\n\t"
        "add x2, x2, #4\n\t"
        "sdot z18.s, z3.b, z6.b\n\t"
        "sdot z19.s, z3.b, z7.b\n\t"

        "add x3, x3, #4\n\t"
        "subs x5, x5, #1\n\t"
        "b.ne 1b\n\t"

        // Store S
        "st1w {z16.s}, p1, [%[s]]\n\t"
        "st1w {z17.s}, p1, [%[s], #1, mul vl]\n\t"
        "st1w {z18.s}, p1, [%[s], #2, mul vl]\n\t"
        "st1w {z19.s}, p1, [%[s], #3, mul vl]\n\t"
        :
        : [q] "r"(Q), [k] "r"(K), [s] "r"(S)
        : "x0", "x1", "x2", "x3", "x4", "x5",
          "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7",
          "z16", "z17", "z18", "z19",
          "p0", "p1", "memory"
    );
}

// Deep interleave with 2-iteration unroll (like DGEMM .L6 loop)
void qkt_deep_interleave(
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

        // Accumulators
        "dup z16.s, #0\n\t"
        "dup z17.s, #0\n\t"
        "dup z18.s, #0\n\t"
        "dup z19.s, #0\n\t"

        // Preload first iteration
        "ld1b {z0.b}, p0/z, [x4]\n\t"
        "ld1rw {z4.s}, p1/z, [x0]\n\t"
        "ld1rw {z5.s}, p1/z, [x1]\n\t"

        "mov x5, #32\n\t"  // 32 x 2 = 64 iterations
        "1:\n\t"
        // Iteration 0
        "ld1b {z1.b}, p0/z, [x4, #1, mul vl]\n\t"
        "ld1rw {z6.s}, p1/z, [x2]\n\t"
        "sdot z16.s, z0.b, z4.b\n\t"
        "sdot z17.s, z0.b, z5.b\n\t"

        "ld1b {z2.b}, p0/z, [x4, #2, mul vl]\n\t"
        "ld1rw {z7.s}, p1/z, [x3]\n\t"
        "sdot z18.s, z0.b, z6.b\n\t"
        "sdot z19.s, z0.b, z7.b\n\t"

        "ld1b {z3.b}, p0/z, [x4, #3, mul vl]\n\t"
        "sdot z16.s, z1.b, z4.b\n\t"
        "sdot z17.s, z1.b, z5.b\n\t"
        "sdot z18.s, z1.b, z6.b\n\t"

        "ld1rw {z8.s}, p1/z, [x0, #4]\n\t"      // Next Q[0]
        "sdot z19.s, z1.b, z7.b\n\t"
        "sdot z16.s, z2.b, z4.b\n\t"
        "sdot z17.s, z2.b, z5.b\n\t"

        "ld1rw {z9.s}, p1/z, [x1, #4]\n\t"      // Next Q[1]
        "sdot z18.s, z2.b, z6.b\n\t"
        "sdot z19.s, z2.b, z7.b\n\t"
        "sdot z16.s, z3.b, z4.b\n\t"

        "ld1b {z0.b}, p0/z, [x4, #4, mul vl]\n\t"  // Next K[0]
        "sdot z17.s, z3.b, z5.b\n\t"
        "sdot z18.s, z3.b, z6.b\n\t"
        "sdot z19.s, z3.b, z7.b\n\t"

        // Iteration 1
        "ld1b {z1.b}, p0/z, [x4, #5, mul vl]\n\t"
        "ld1rw {z10.s}, p1/z, [x2, #4]\n\t"
        "sdot z16.s, z0.b, z8.b\n\t"
        "sdot z17.s, z0.b, z9.b\n\t"

        "ld1b {z2.b}, p0/z, [x4, #6, mul vl]\n\t"
        "ld1rw {z11.s}, p1/z, [x3, #4]\n\t"
        "sdot z18.s, z0.b, z10.b\n\t"
        "sdot z19.s, z0.b, z11.b\n\t"

        "ld1b {z3.b}, p0/z, [x4, #7, mul vl]\n\t"
        "sdot z16.s, z1.b, z8.b\n\t"
        "sdot z17.s, z1.b, z9.b\n\t"
        "sdot z18.s, z1.b, z10.b\n\t"

        "add x4, x4, #512\n\t"
        "sdot z19.s, z1.b, z11.b\n\t"
        "sdot z16.s, z2.b, z8.b\n\t"
        "sdot z17.s, z2.b, z9.b\n\t"

        "add x0, x0, #8\n\t"
        "sdot z18.s, z2.b, z10.b\n\t"
        "sdot z19.s, z2.b, z11.b\n\t"
        "sdot z16.s, z3.b, z8.b\n\t"

        "ld1b {z0.b}, p0/z, [x4]\n\t"           // Next K[0]
        "add x1, x1, #8\n\t"
        "sdot z17.s, z3.b, z9.b\n\t"
        "sdot z18.s, z3.b, z10.b\n\t"

        "ld1rw {z4.s}, p1/z, [x0]\n\t"          // Next Q[0]
        "add x2, x2, #8\n\t"
        "sdot z19.s, z3.b, z11.b\n\t"

        "ld1rw {z5.s}, p1/z, [x1]\n\t"          // Next Q[1]
        "add x3, x3, #8\n\t"
        "subs x5, x5, #1\n\t"
        "b.ne 1b\n\t"

        // Store S
        "st1w {z16.s}, p1, [%[s]]\n\t"
        "st1w {z17.s}, p1, [%[s], #1, mul vl]\n\t"
        "st1w {z18.s}, p1, [%[s], #2, mul vl]\n\t"
        "st1w {z19.s}, p1, [%[s], #3, mul vl]\n\t"
        :
        : [q] "r"(Q), [k] "r"(K), [s] "r"(S)
        : "x0", "x1", "x2", "x3", "x4", "x5",
          "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7",
          "z8", "z9", "z10", "z11",
          "z16", "z17", "z18", "z19",
          "p0", "p1", "memory"
    );
}

// With sector cache hints (like DGEMM: orr addr with sector bits)
void qkt_sector_tagged(
    const int8_t* Q,
    const int8_t* K,
    int32_t* S
) {
    // Tag K with sector 1 (streaming), Q with sector 4 (keep)
    // addr[57:56] = sector_id
    uint64_t k_tagged = (uint64_t)K | (0x1ULL << 56);  // sector 1
    uint64_t q_tagged = (uint64_t)Q | (0x4ULL << 56);  // sector 4

    __asm__ volatile(
        "ptrue p0.b\n\t"
        "ptrue p1.s\n\t"

        "mov x0, %[q]\n\t"
        "add x1, x0, #256\n\t"
        "add x2, x0, #512\n\t"
        "add x3, x0, #768\n\t"
        "mov x4, %[k]\n\t"

        // Accumulators
        "dup z16.s, #0\n\t"
        "dup z17.s, #0\n\t"
        "dup z18.s, #0\n\t"
        "dup z19.s, #0\n\t"

        // Preload
        "ld1b {z0.b}, p0/z, [x4]\n\t"
        "ld1rw {z4.s}, p1/z, [x0]\n\t"
        "ld1rw {z5.s}, p1/z, [x1]\n\t"

        "mov x5, #32\n\t"
        "1:\n\t"
        // Same deep interleave pattern
        "ld1b {z1.b}, p0/z, [x4, #1, mul vl]\n\t"
        "ld1rw {z6.s}, p1/z, [x2]\n\t"
        "sdot z16.s, z0.b, z4.b\n\t"
        "sdot z17.s, z0.b, z5.b\n\t"

        "ld1b {z2.b}, p0/z, [x4, #2, mul vl]\n\t"
        "ld1rw {z7.s}, p1/z, [x3]\n\t"
        "sdot z18.s, z0.b, z6.b\n\t"
        "sdot z19.s, z0.b, z7.b\n\t"

        "ld1b {z3.b}, p0/z, [x4, #3, mul vl]\n\t"
        "sdot z16.s, z1.b, z4.b\n\t"
        "sdot z17.s, z1.b, z5.b\n\t"
        "sdot z18.s, z1.b, z6.b\n\t"

        "ld1rw {z8.s}, p1/z, [x0, #4]\n\t"
        "sdot z19.s, z1.b, z7.b\n\t"
        "sdot z16.s, z2.b, z4.b\n\t"
        "sdot z17.s, z2.b, z5.b\n\t"

        "ld1rw {z9.s}, p1/z, [x1, #4]\n\t"
        "sdot z18.s, z2.b, z6.b\n\t"
        "sdot z19.s, z2.b, z7.b\n\t"
        "sdot z16.s, z3.b, z4.b\n\t"

        "ld1b {z0.b}, p0/z, [x4, #4, mul vl]\n\t"
        "sdot z17.s, z3.b, z5.b\n\t"
        "sdot z18.s, z3.b, z6.b\n\t"
        "sdot z19.s, z3.b, z7.b\n\t"

        // Iteration 1
        "ld1b {z1.b}, p0/z, [x4, #5, mul vl]\n\t"
        "ld1rw {z10.s}, p1/z, [x2, #4]\n\t"
        "sdot z16.s, z0.b, z8.b\n\t"
        "sdot z17.s, z0.b, z9.b\n\t"

        "ld1b {z2.b}, p0/z, [x4, #6, mul vl]\n\t"
        "ld1rw {z11.s}, p1/z, [x3, #4]\n\t"
        "sdot z18.s, z0.b, z10.b\n\t"
        "sdot z19.s, z0.b, z11.b\n\t"

        "ld1b {z3.b}, p0/z, [x4, #7, mul vl]\n\t"
        "sdot z16.s, z1.b, z8.b\n\t"
        "sdot z17.s, z1.b, z9.b\n\t"
        "sdot z18.s, z1.b, z10.b\n\t"

        "add x4, x4, #512\n\t"
        "sdot z19.s, z1.b, z11.b\n\t"
        "sdot z16.s, z2.b, z8.b\n\t"
        "sdot z17.s, z2.b, z9.b\n\t"

        "add x0, x0, #8\n\t"
        "sdot z18.s, z2.b, z10.b\n\t"
        "sdot z19.s, z2.b, z11.b\n\t"
        "sdot z16.s, z3.b, z8.b\n\t"

        "ld1b {z0.b}, p0/z, [x4]\n\t"
        "add x1, x1, #8\n\t"
        "sdot z17.s, z3.b, z9.b\n\t"
        "sdot z18.s, z3.b, z10.b\n\t"

        "ld1rw {z4.s}, p1/z, [x0]\n\t"
        "add x2, x2, #8\n\t"
        "sdot z19.s, z3.b, z11.b\n\t"

        "ld1rw {z5.s}, p1/z, [x1]\n\t"
        "add x3, x3, #8\n\t"
        "subs x5, x5, #1\n\t"
        "b.ne 1b\n\t"

        // Store S
        "st1w {z16.s}, p1, [%[s]]\n\t"
        "st1w {z17.s}, p1, [%[s], #1, mul vl]\n\t"
        "st1w {z18.s}, p1, [%[s], #2, mul vl]\n\t"
        "st1w {z19.s}, p1, [%[s], #3, mul vl]\n\t"
        :
        : [q] "r"(q_tagged), [k] "r"(k_tagged), [s] "r"(S)
        : "x0", "x1", "x2", "x3", "x4", "x5",
          "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7",
          "z8", "z9", "z10", "z11",
          "z16", "z17", "z18", "z19",
          "p0", "p1", "memory"
    );
}

// Full fused with interleaved SDOT
void fused_interleaved(
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

        "ld1b {z0.b}, p0/z, [x4]\n\t"
        "ld1rw {z4.s}, p1/z, [x0]\n\t"
        "ld1rw {z5.s}, p1/z, [x1]\n\t"

        "mov x5, #32\n\t"
        "1:\n\t"
        "ld1b {z1.b}, p0/z, [x4, #1, mul vl]\n\t"
        "ld1rw {z6.s}, p1/z, [x2]\n\t"
        "sdot z16.s, z0.b, z4.b\n\t"
        "sdot z17.s, z0.b, z5.b\n\t"

        "ld1b {z2.b}, p0/z, [x4, #2, mul vl]\n\t"
        "ld1rw {z7.s}, p1/z, [x3]\n\t"
        "sdot z18.s, z0.b, z6.b\n\t"
        "sdot z19.s, z0.b, z7.b\n\t"

        "ld1b {z3.b}, p0/z, [x4, #3, mul vl]\n\t"
        "sdot z16.s, z1.b, z4.b\n\t"
        "sdot z17.s, z1.b, z5.b\n\t"
        "sdot z18.s, z1.b, z6.b\n\t"

        "ld1rw {z8.s}, p1/z, [x0, #4]\n\t"
        "sdot z19.s, z1.b, z7.b\n\t"
        "sdot z16.s, z2.b, z4.b\n\t"
        "sdot z17.s, z2.b, z5.b\n\t"

        "ld1rw {z9.s}, p1/z, [x1, #4]\n\t"
        "sdot z18.s, z2.b, z6.b\n\t"
        "sdot z19.s, z2.b, z7.b\n\t"
        "sdot z16.s, z3.b, z4.b\n\t"

        "ld1b {z0.b}, p0/z, [x4, #4, mul vl]\n\t"
        "sdot z17.s, z3.b, z5.b\n\t"
        "sdot z18.s, z3.b, z6.b\n\t"
        "sdot z19.s, z3.b, z7.b\n\t"

        "ld1b {z1.b}, p0/z, [x4, #5, mul vl]\n\t"
        "ld1rw {z10.s}, p1/z, [x2, #4]\n\t"
        "sdot z16.s, z0.b, z8.b\n\t"
        "sdot z17.s, z0.b, z9.b\n\t"

        "ld1b {z2.b}, p0/z, [x4, #6, mul vl]\n\t"
        "ld1rw {z11.s}, p1/z, [x3, #4]\n\t"
        "sdot z18.s, z0.b, z10.b\n\t"
        "sdot z19.s, z0.b, z11.b\n\t"

        "ld1b {z3.b}, p0/z, [x4, #7, mul vl]\n\t"
        "sdot z16.s, z1.b, z8.b\n\t"
        "sdot z17.s, z1.b, z9.b\n\t"
        "sdot z18.s, z1.b, z10.b\n\t"

        "add x4, x4, #512\n\t"
        "sdot z19.s, z1.b, z11.b\n\t"
        "sdot z16.s, z2.b, z8.b\n\t"
        "sdot z17.s, z2.b, z9.b\n\t"

        "add x0, x0, #8\n\t"
        "sdot z18.s, z2.b, z10.b\n\t"
        "sdot z19.s, z2.b, z11.b\n\t"
        "sdot z16.s, z3.b, z8.b\n\t"

        "ld1b {z0.b}, p0/z, [x4]\n\t"
        "add x1, x1, #8\n\t"
        "sdot z17.s, z3.b, z9.b\n\t"
        "sdot z18.s, z3.b, z10.b\n\t"

        "ld1rw {z4.s}, p1/z, [x0]\n\t"
        "add x2, x2, #8\n\t"
        "sdot z19.s, z3.b, z11.b\n\t"

        "ld1rw {z5.s}, p1/z, [x1]\n\t"
        "add x3, x3, #8\n\t"
        "subs x5, x5, #1\n\t"
        "b.ne 1b\n\t"

        // Store S
        "st1w {z16.s}, p1, [%[s]]\n\t"
        "st1w {z17.s}, p1, [%[s], #1, mul vl]\n\t"
        "st1w {z18.s}, p1, [%[s], #2, mul vl]\n\t"
        "st1w {z19.s}, p1, [%[s], #3, mul vl]\n\t"

        // ============ Phase 2: P@V ============
        "mov x7, %[v]\n\t"
        "mov x8, %[o]\n\t"
        "mov x9, #4\n\t"  // 4 D-tiles

        "2:\n\t"
        "mov x0, %[s]\n\t"
        "add x1, x0, #64\n\t"
        "add x2, x0, #128\n\t"
        "add x3, x0, #192\n\t"

        "dup z16.s, #0\n\t"
        "dup z17.s, #0\n\t"
        "dup z18.s, #0\n\t"
        "dup z19.s, #0\n\t"

        "mov x10, x7\n\t"

        // Preload
        "ld1b {z0.b}, p0/z, [x10]\n\t"
        "ld1rw {z4.s}, p1/z, [x0]\n\t"
        "ld1rw {z5.s}, p1/z, [x1]\n\t"

        "mov x5, #8\n\t"  // 8 x 2 = 16 N_groups
        "3:\n\t"
        "ld1b {z1.b}, p0/z, [x10, #1, mul vl]\n\t"
        "ld1rw {z6.s}, p1/z, [x2]\n\t"
        "sdot z16.s, z0.b, z4.b\n\t"
        "sdot z17.s, z0.b, z5.b\n\t"

        "ld1b {z2.b}, p0/z, [x10, #2, mul vl]\n\t"
        "ld1rw {z7.s}, p1/z, [x3]\n\t"
        "sdot z18.s, z0.b, z6.b\n\t"
        "sdot z19.s, z0.b, z7.b\n\t"

        "ld1b {z3.b}, p0/z, [x10, #3, mul vl]\n\t"
        "sdot z16.s, z1.b, z4.b\n\t"
        "sdot z17.s, z1.b, z5.b\n\t"
        "sdot z18.s, z1.b, z6.b\n\t"

        "ld1rw {z8.s}, p1/z, [x0, #4]\n\t"
        "sdot z19.s, z1.b, z7.b\n\t"
        "sdot z16.s, z2.b, z4.b\n\t"
        "sdot z17.s, z2.b, z5.b\n\t"

        "ld1rw {z9.s}, p1/z, [x1, #4]\n\t"
        "sdot z18.s, z2.b, z6.b\n\t"
        "sdot z19.s, z2.b, z7.b\n\t"
        "sdot z16.s, z3.b, z4.b\n\t"

        "add x10, x10, #1024\n\t"
        "ld1b {z0.b}, p0/z, [x10]\n\t"
        "sdot z17.s, z3.b, z5.b\n\t"
        "sdot z18.s, z3.b, z6.b\n\t"
        "sdot z19.s, z3.b, z7.b\n\t"

        // Iteration 1
        "ld1b {z1.b}, p0/z, [x10, #1, mul vl]\n\t"
        "ld1rw {z10.s}, p1/z, [x2, #4]\n\t"
        "sdot z16.s, z0.b, z8.b\n\t"
        "sdot z17.s, z0.b, z9.b\n\t"

        "ld1b {z2.b}, p0/z, [x10, #2, mul vl]\n\t"
        "ld1rw {z11.s}, p1/z, [x3, #4]\n\t"
        "sdot z18.s, z0.b, z10.b\n\t"
        "sdot z19.s, z0.b, z11.b\n\t"

        "ld1b {z3.b}, p0/z, [x10, #3, mul vl]\n\t"
        "sdot z16.s, z1.b, z8.b\n\t"
        "sdot z17.s, z1.b, z9.b\n\t"
        "sdot z18.s, z1.b, z10.b\n\t"

        "add x10, x10, #1024\n\t"
        "sdot z19.s, z1.b, z11.b\n\t"
        "sdot z16.s, z2.b, z8.b\n\t"
        "sdot z17.s, z2.b, z9.b\n\t"

        "add x0, x0, #8\n\t"
        "sdot z18.s, z2.b, z10.b\n\t"
        "sdot z19.s, z2.b, z11.b\n\t"
        "sdot z16.s, z3.b, z8.b\n\t"

        "ld1b {z0.b}, p0/z, [x10]\n\t"
        "add x1, x1, #8\n\t"
        "sdot z17.s, z3.b, z9.b\n\t"
        "sdot z18.s, z3.b, z10.b\n\t"

        "ld1rw {z4.s}, p1/z, [x0]\n\t"
        "add x2, x2, #8\n\t"
        "sdot z19.s, z3.b, z11.b\n\t"

        "ld1rw {z5.s}, p1/z, [x1]\n\t"
        "add x3, x3, #8\n\t"
        "subs x5, x5, #1\n\t"
        "b.ne 3b\n\t"

        // Store O tile
        "st1w {z16.s}, p1, [x8]\n\t"
        "add x8, x8, #1024\n\t"
        "st1w {z17.s}, p1, [x8]\n\t"
        "add x8, x8, #1024\n\t"
        "st1w {z18.s}, p1, [x8]\n\t"
        "add x8, x8, #1024\n\t"
        "st1w {z19.s}, p1, [x8]\n\t"
        "sub x8, x8, #3008\n\t"

        "add x7, x7, #256\n\t"
        "subs x9, x9, #1\n\t"
        "b.ne 2b\n\t"
        :
        : [q] "r"(Q), [k] "r"(K), [s] "r"(S), [v] "r"(V), [o] "r"(O)
        : "x0", "x1", "x2", "x3", "x4", "x5", "x7", "x8", "x9", "x10",
          "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7",
          "z8", "z9", "z10", "z11",
          "z16", "z17", "z18", "z19",
          "p0", "p1", "memory"
    );
}

int main() {
    printf("==============================================\n");
    printf("Interleaved Load+SDOT Pattern Analysis\n");
    printf("==============================================\n");
    printf("Based on reference DGEMM kernel pattern\n\n");

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
    printf("Load: K(16KB) + Q(1KB) = 17KB\n");
    printf("Compute: 64 d_groups x 4 K_vecs x 4 rows = 1024 SDOT\n");
    printf("Theoretical: 136 cy (load-bound) or 256 cy (compute-bound @ 4 SDOT/cy)\n\n");

    printf("%-25s %8s %10s %10s\n", "Kernel", "Cycles", "Ld B/cy", "SDOT/cy");
    printf("----------------------------------------------------------\n");

    for (int i = 0; i < warmup; i++) qkt_sequential(Q, K, S);
    start = rdtsc();
    for (int i = 0; i < iters; i++) qkt_sequential(Q, K, S);
    end = rdtsc();
    ticks = (double)(end - start) / iters;
    cycles = ticks * 20.0;
    printf("%-25s %8.1f %10.1f %10.1f\n", "sequential", cycles, 17408.0/cycles, 1024.0/cycles);

    for (int i = 0; i < warmup; i++) qkt_interleaved(Q, K, S);
    start = rdtsc();
    for (int i = 0; i < iters; i++) qkt_interleaved(Q, K, S);
    end = rdtsc();
    ticks = (double)(end - start) / iters;
    cycles = ticks * 20.0;
    printf("%-25s %8.1f %10.1f %10.1f\n", "interleaved", cycles, 17408.0/cycles, 1024.0/cycles);

    for (int i = 0; i < warmup; i++) qkt_deep_interleave(Q, K, S);
    start = rdtsc();
    for (int i = 0; i < iters; i++) qkt_deep_interleave(Q, K, S);
    end = rdtsc();
    ticks = (double)(end - start) / iters;
    cycles = ticks * 20.0;
    printf("%-25s %8.1f %10.1f %10.1f\n", "deep_interleave", cycles, 17408.0/cycles, 1024.0/cycles);

    for (int i = 0; i < warmup; i++) qkt_sector_tagged(Q, K, S);
    start = rdtsc();
    for (int i = 0; i < iters; i++) qkt_sector_tagged(Q, K, S);
    end = rdtsc();
    ticks = (double)(end - start) / iters;
    cycles = ticks * 20.0;
    printf("%-25s %8.1f %10.1f %10.1f\n", "sector_tagged", cycles, 17408.0/cycles, 1024.0/cycles);

    printf("\n=== Full Fused Attention ===\n");
    printf("Load: K(16KB) + Q(1KB) + V(16KB) + S(1KB) = 34KB\n");
    printf("Compute: 1024 + 1024 = 2048 SDOT\n");
    printf("Theoretical: 266 cy (load) or 512 cy (compute)\n\n");

    for (int i = 0; i < warmup; i++) fused_interleaved(Q, K, S, V, O);
    start = rdtsc();
    for (int i = 0; i < iters; i++) fused_interleaved(Q, K, S, V, O);
    end = rdtsc();
    ticks = (double)(end - start) / iters;
    cycles = ticks * 20.0;
    printf("%-25s %8.1f cy  %6.1f ld B/cy  %5.1f SDOT/cy  %.1f%%\n",
           "fused_interleaved", cycles, 34816.0/cycles, 2048.0/cycles, 100.0*266.0/cycles);

    free(Q); free(K); free(S); free(V); free(O);
    return 0;
}
