#include <arm_sve.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// A64FX: 2 FPU pipes, peak 2 SDOT/cycle = 40 SDOT/tick
// MR=6 rows, N=64, D=256
// Phase 1: Q[6,256] @ K[256,64]^T = S[6,64] -> 6*64*64 = 24576 SDOT (4 bytes each) = 6144 SDOT
// Phase 2: S[6,64] @ V[64,256] = O[6,256] -> 6*64*64 = 24576 SDOT = 3072 SDOT (4 D-tiles)
// Total: 9216 SDOT per fused kernel

static inline uint64_t rdtick(void) {
    uint64_t val;
    __asm__ volatile("mrs %0, cntvct_el0" : "=r"(val));
    return val;
}

// K-group macro for Phase 1 (Q@K^T) with full unrolling
// Each K-group: 4 K loads + 4 Q broadcasts + 16 SDOT
#define PHASE1_K_ITER(k_off, q_off) \
    "ld1b {z24.b}, p0/z, [x5, #" #k_off ", mul vl]\n\t" \
    "ld1b {z25.b}, p0/z, [x5, #" #k_off "+1, mul vl]\n\t" \
    "ld1b {z26.b}, p0/z, [x5, #" #k_off "+2, mul vl]\n\t" \
    "ld1b {z27.b}, p0/z, [x5, #" #k_off "+3, mul vl]\n\t" \
    "ld1rw {z28.s}, p1/z, [x4, #" #q_off "]\n\t" \
    "ld1rw {z29.s}, p1/z, [x17, #" #q_off "]\n\t" \
    "ld1rw {z30.s}, p1/z, [x18, #" #q_off "]\n\t" \
    "ld1rw {z31.s}, p1/z, [x19, #" #q_off "]\n\t" \
    "sdot z0.s, z24.b, z28.b\n\t" \
    "sdot z1.s, z25.b, z28.b\n\t" \
    "sdot z2.s, z26.b, z28.b\n\t" \
    "sdot z3.s, z27.b, z28.b\n\t" \
    "sdot z4.s, z24.b, z29.b\n\t" \
    "sdot z5.s, z25.b, z29.b\n\t" \
    "sdot z6.s, z26.b, z29.b\n\t" \
    "sdot z7.s, z27.b, z29.b\n\t" \
    "sdot z8.s, z24.b, z30.b\n\t" \
    "sdot z9.s, z25.b, z30.b\n\t" \
    "sdot z10.s, z26.b, z30.b\n\t" \
    "sdot z11.s, z27.b, z30.b\n\t" \
    "sdot z12.s, z24.b, z31.b\n\t" \
    "sdot z13.s, z25.b, z31.b\n\t" \
    "sdot z14.s, z26.b, z31.b\n\t" \
    "sdot z15.s, z27.b, z31.b\n\t"

// Additional Q pointers for rows 4-5
#define PHASE1_K_ITER_6ROW(k_off, q_off) \
    "ld1b {z24.b}, p0/z, [x5, #" #k_off ", mul vl]\n\t" \
    "ld1b {z25.b}, p0/z, [x5, #" #k_off "+1, mul vl]\n\t" \
    "ld1b {z26.b}, p0/z, [x5, #" #k_off "+2, mul vl]\n\t" \
    "ld1b {z27.b}, p0/z, [x5, #" #k_off "+3, mul vl]\n\t" \
    "ld1rw {z28.s}, p1/z, [x4, #" #q_off "]\n\t" \
    "ld1rw {z29.s}, p1/z, [x17, #" #q_off "]\n\t" \
    "ld1rw {z30.s}, p1/z, [x18, #" #q_off "]\n\t" \
    "ld1rw {z31.s}, p1/z, [x19, #" #q_off "]\n\t" \
    "sdot z0.s, z24.b, z28.b\n\t" \
    "sdot z4.s, z24.b, z29.b\n\t" \
    "sdot z8.s, z24.b, z30.b\n\t" \
    "sdot z12.s, z24.b, z31.b\n\t" \
    "ld1rw {z28.s}, p1/z, [x20, #" #q_off "]\n\t" \
    "ld1rw {z29.s}, p1/z, [x21, #" #q_off "]\n\t" \
    "sdot z1.s, z25.b, z28.b\n\t" \
    "sdot z5.s, z25.b, z29.b\n\t" \
    "sdot z2.s, z26.b, z28.b\n\t" \
    "sdot z6.s, z26.b, z29.b\n\t" \
    "sdot z3.s, z27.b, z28.b\n\t" \
    "sdot z7.s, z27.b, z29.b\n\t" \
    "sdot z9.s, z25.b, z30.b\n\t" \
    "sdot z13.s, z25.b, z31.b\n\t" \
    "sdot z10.s, z26.b, z30.b\n\t" \
    "sdot z14.s, z26.b, z31.b\n\t" \
    "sdot z11.s, z27.b, z30.b\n\t" \
    "sdot z15.s, z27.b, z31.b\n\t" \
    "ld1rw {z30.s}, p1/z, [x20, #" #q_off "]\n\t" \
    "ld1rw {z31.s}, p1/z, [x21, #" #q_off "]\n\t" \
    "sdot z16.s, z24.b, z30.b\n\t" \
    "sdot z20.s, z24.b, z31.b\n\t" \
    "sdot z17.s, z25.b, z30.b\n\t" \
    "sdot z21.s, z25.b, z31.b\n\t" \
    "sdot z18.s, z26.b, z30.b\n\t" \
    "sdot z22.s, z26.b, z31.b\n\t" \
    "sdot z19.s, z27.b, z30.b\n\t" \
    "sdot z23.s, z27.b, z31.b\n\t"

// Simplified 4-row version with full unroll for Phase 1 (one N-tile)
__attribute__((noinline))
void fused_attention_full_unroll(
    const int8_t* __restrict__ Q,     // [6, 256] row-major
    const int8_t* __restrict__ K,     // [64, 256] interleaved as [64, 64, 4]
    const int8_t* __restrict__ V,     // [64, 256] interleaved
    int32_t* __restrict__ O,          // [6, 256] output
    int MR                            // 4 or 6
) {
    // Process 4 N-tiles in Phase 1, 4 D-tiles in Phase 2
    // Use MR=4 for this implementation

    int8_t S_quant[4 * 64] __attribute__((aligned(64)));

    // For each N-tile (4 tiles covering N=64)
    for (int n_tile = 0; n_tile < 4; n_tile++) {
        const int8_t* k_ptr = K + n_tile * 64 * 4;  // Each N-tile has 16 columns

        __asm__ volatile(
            "ptrue p0.b\n\t"
            "ptrue p1.s\n\t"

            // Initialize accumulators with EOR
            "eor z0.d, z0.d, z0.d\n\t"
            "eor z1.d, z1.d, z1.d\n\t"
            "eor z2.d, z2.d, z2.d\n\t"
            "eor z3.d, z3.d, z3.d\n\t"
            "eor z4.d, z4.d, z4.d\n\t"
            "eor z5.d, z5.d, z5.d\n\t"
            "eor z6.d, z6.d, z6.d\n\t"
            "eor z7.d, z7.d, z7.d\n\t"
            "eor z8.d, z8.d, z8.d\n\t"
            "eor z9.d, z9.d, z9.d\n\t"
            "eor z10.d, z10.d, z10.d\n\t"
            "eor z11.d, z11.d, z11.d\n\t"
            "eor z12.d, z12.d, z12.d\n\t"
            "eor z13.d, z13.d, z13.d\n\t"
            "eor z14.d, z14.d, z14.d\n\t"
            "eor z15.d, z15.d, z15.d\n\t"

            // Setup Q row pointers (rows 0-3)
            "add x17, x4, #256\n\t"     // Q row 1
            "add x18, x4, #512\n\t"     // Q row 2
            "add x19, x4, #768\n\t"     // Q row 3

            // K-loop: 64 iterations, fully unrolled in groups of 8
            // Group 0: k=0-7 (offset 0-28)
            PHASE1_K_ITER(0, 0)
            PHASE1_K_ITER(4, 4)
            "add x5, x5, #512\n\t"  // Advance K ptr
            "add x4, x4, #32\n\t"   // Advance Q ptr
            "add x17, x17, #32\n\t"
            "add x18, x18, #32\n\t"
            "add x19, x19, #32\n\t"

            // Group 1: k=8-15
            PHASE1_K_ITER(0, 0)
            PHASE1_K_ITER(4, 4)
            "add x5, x5, #512\n\t"
            "add x4, x4, #32\n\t"
            "add x17, x17, #32\n\t"
            "add x18, x18, #32\n\t"
            "add x19, x19, #32\n\t"

            // Group 2: k=16-23
            PHASE1_K_ITER(0, 0)
            PHASE1_K_ITER(4, 4)
            "add x5, x5, #512\n\t"
            "add x4, x4, #32\n\t"
            "add x17, x17, #32\n\t"
            "add x18, x18, #32\n\t"
            "add x19, x19, #32\n\t"

            // Group 3: k=24-31
            PHASE1_K_ITER(0, 0)
            PHASE1_K_ITER(4, 4)
            "add x5, x5, #512\n\t"
            "add x4, x4, #32\n\t"
            "add x17, x17, #32\n\t"
            "add x18, x18, #32\n\t"
            "add x19, x19, #32\n\t"

            // Group 4: k=32-39
            PHASE1_K_ITER(0, 0)
            PHASE1_K_ITER(4, 4)
            "add x5, x5, #512\n\t"
            "add x4, x4, #32\n\t"
            "add x17, x17, #32\n\t"
            "add x18, x18, #32\n\t"
            "add x19, x19, #32\n\t"

            // Group 5: k=40-47
            PHASE1_K_ITER(0, 0)
            PHASE1_K_ITER(4, 4)
            "add x5, x5, #512\n\t"
            "add x4, x4, #32\n\t"
            "add x17, x17, #32\n\t"
            "add x18, x18, #32\n\t"
            "add x19, x19, #32\n\t"

            // Group 6: k=48-55
            PHASE1_K_ITER(0, 0)
            PHASE1_K_ITER(4, 4)
            "add x5, x5, #512\n\t"
            "add x4, x4, #32\n\t"
            "add x17, x17, #32\n\t"
            "add x18, x18, #32\n\t"
            "add x19, x19, #32\n\t"

            // Group 7: k=56-63
            PHASE1_K_ITER(0, 0)
            PHASE1_K_ITER(4, 4)

            // Store S scores to scratch (quantize to int8 - simplified)
            "st1w {z0.s}, p1, [x6]\n\t"
            "st1w {z1.s}, p1, [x6, #1, mul vl]\n\t"
            "st1w {z2.s}, p1, [x6, #2, mul vl]\n\t"
            "st1w {z3.s}, p1, [x6, #3, mul vl]\n\t"
            "add x6, x6, #256\n\t"
            "st1w {z4.s}, p1, [x6]\n\t"
            "st1w {z5.s}, p1, [x6, #1, mul vl]\n\t"
            "st1w {z6.s}, p1, [x6, #2, mul vl]\n\t"
            "st1w {z7.s}, p1, [x6, #3, mul vl]\n\t"
            "add x6, x6, #256\n\t"
            "st1w {z8.s}, p1, [x6]\n\t"
            "st1w {z9.s}, p1, [x6, #1, mul vl]\n\t"
            "st1w {z10.s}, p1, [x6, #2, mul vl]\n\t"
            "st1w {z11.s}, p1, [x6, #3, mul vl]\n\t"
            "add x6, x6, #256\n\t"
            "st1w {z12.s}, p1, [x6]\n\t"
            "st1w {z13.s}, p1, [x6, #1, mul vl]\n\t"
            "st1w {z14.s}, p1, [x6, #2, mul vl]\n\t"
            "st1w {z15.s}, p1, [x6, #3, mul vl]\n\t"

            :
            : [Q]"r"(Q), [K]"r"(k_ptr), [S]"r"(S_quant + n_tile * 16)
            : "x4", "x5", "x6", "x17", "x18", "x19",
              "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7",
              "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15",
              "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31",
              "p0", "p1", "memory"
        );
    }
}

// Simplified Phase 1 only benchmark with full unroll
__attribute__((noinline))
void phase1_full_unroll_4row(
    const int8_t* __restrict__ Q,     // [4, 256]
    const int8_t* __restrict__ K,     // [256, 64] interleaved
    int32_t* __restrict__ S           // [4, 64] output
) {
    __asm__ volatile(
        "ptrue p0.b\n\t"
        "ptrue p1.s\n\t"

        // Initialize 16 accumulators with EOR
        "eor z0.d, z0.d, z0.d\n\t"
        "eor z1.d, z1.d, z1.d\n\t"
        "eor z2.d, z2.d, z2.d\n\t"
        "eor z3.d, z3.d, z3.d\n\t"
        "eor z4.d, z4.d, z4.d\n\t"
        "eor z5.d, z5.d, z5.d\n\t"
        "eor z6.d, z6.d, z6.d\n\t"
        "eor z7.d, z7.d, z7.d\n\t"
        "eor z8.d, z8.d, z8.d\n\t"
        "eor z9.d, z9.d, z9.d\n\t"
        "eor z10.d, z10.d, z10.d\n\t"
        "eor z11.d, z11.d, z11.d\n\t"
        "eor z12.d, z12.d, z12.d\n\t"
        "eor z13.d, z13.d, z13.d\n\t"
        "eor z14.d, z14.d, z14.d\n\t"
        "eor z15.d, z15.d, z15.d\n\t"

        // Setup pointers
        "mov x4, %[Q]\n\t"          // Q ptr
        "mov x5, %[K]\n\t"          // K ptr
        "add x17, x4, #256\n\t"     // Q row 1
        "add x18, x4, #512\n\t"     // Q row 2
        "add x19, x4, #768\n\t"     // Q row 3

        // ===== FULLY UNROLLED K-LOOP (64 iterations) =====
        // Group 0: k=0,1 (offsets 0,4)
        PHASE1_K_ITER(0, 0)
        PHASE1_K_ITER(4, 4)
        "add x5, x5, #512\n\t"
        "add x4, x4, #32\n\t"
        "add x17, x17, #32\n\t"
        "add x18, x18, #32\n\t"
        "add x19, x19, #32\n\t"

        // Group 1: k=2,3
        PHASE1_K_ITER(0, 0)
        PHASE1_K_ITER(4, 4)
        "add x5, x5, #512\n\t"
        "add x4, x4, #32\n\t"
        "add x17, x17, #32\n\t"
        "add x18, x18, #32\n\t"
        "add x19, x19, #32\n\t"

        // Group 2: k=4,5
        PHASE1_K_ITER(0, 0)
        PHASE1_K_ITER(4, 4)
        "add x5, x5, #512\n\t"
        "add x4, x4, #32\n\t"
        "add x17, x17, #32\n\t"
        "add x18, x18, #32\n\t"
        "add x19, x19, #32\n\t"

        // Group 3: k=6,7
        PHASE1_K_ITER(0, 0)
        PHASE1_K_ITER(4, 4)
        "add x5, x5, #512\n\t"
        "add x4, x4, #32\n\t"
        "add x17, x17, #32\n\t"
        "add x18, x18, #32\n\t"
        "add x19, x19, #32\n\t"

        // Group 4-7: k=8-15
        PHASE1_K_ITER(0, 0)
        PHASE1_K_ITER(4, 4)
        "add x5, x5, #512\n\t"
        "add x4, x4, #32\n\t"
        "add x17, x17, #32\n\t"
        "add x18, x18, #32\n\t"
        "add x19, x19, #32\n\t"

        PHASE1_K_ITER(0, 0)
        PHASE1_K_ITER(4, 4)
        "add x5, x5, #512\n\t"
        "add x4, x4, #32\n\t"
        "add x17, x17, #32\n\t"
        "add x18, x18, #32\n\t"
        "add x19, x19, #32\n\t"

        PHASE1_K_ITER(0, 0)
        PHASE1_K_ITER(4, 4)
        "add x5, x5, #512\n\t"
        "add x4, x4, #32\n\t"
        "add x17, x17, #32\n\t"
        "add x18, x18, #32\n\t"
        "add x19, x19, #32\n\t"

        PHASE1_K_ITER(0, 0)
        PHASE1_K_ITER(4, 4)
        "add x5, x5, #512\n\t"
        "add x4, x4, #32\n\t"
        "add x17, x17, #32\n\t"
        "add x18, x18, #32\n\t"
        "add x19, x19, #32\n\t"

        // Group 8-15: k=16-31
        PHASE1_K_ITER(0, 0)
        PHASE1_K_ITER(4, 4)
        "add x5, x5, #512\n\t"
        "add x4, x4, #32\n\t"
        "add x17, x17, #32\n\t"
        "add x18, x18, #32\n\t"
        "add x19, x19, #32\n\t"

        PHASE1_K_ITER(0, 0)
        PHASE1_K_ITER(4, 4)
        "add x5, x5, #512\n\t"
        "add x4, x4, #32\n\t"
        "add x17, x17, #32\n\t"
        "add x18, x18, #32\n\t"
        "add x19, x19, #32\n\t"

        PHASE1_K_ITER(0, 0)
        PHASE1_K_ITER(4, 4)
        "add x5, x5, #512\n\t"
        "add x4, x4, #32\n\t"
        "add x17, x17, #32\n\t"
        "add x18, x18, #32\n\t"
        "add x19, x19, #32\n\t"

        PHASE1_K_ITER(0, 0)
        PHASE1_K_ITER(4, 4)
        "add x5, x5, #512\n\t"
        "add x4, x4, #32\n\t"
        "add x17, x17, #32\n\t"
        "add x18, x18, #32\n\t"
        "add x19, x19, #32\n\t"

        PHASE1_K_ITER(0, 0)
        PHASE1_K_ITER(4, 4)
        "add x5, x5, #512\n\t"
        "add x4, x4, #32\n\t"
        "add x17, x17, #32\n\t"
        "add x18, x18, #32\n\t"
        "add x19, x19, #32\n\t"

        PHASE1_K_ITER(0, 0)
        PHASE1_K_ITER(4, 4)
        "add x5, x5, #512\n\t"
        "add x4, x4, #32\n\t"
        "add x17, x17, #32\n\t"
        "add x18, x18, #32\n\t"
        "add x19, x19, #32\n\t"

        PHASE1_K_ITER(0, 0)
        PHASE1_K_ITER(4, 4)
        "add x5, x5, #512\n\t"
        "add x4, x4, #32\n\t"
        "add x17, x17, #32\n\t"
        "add x18, x18, #32\n\t"
        "add x19, x19, #32\n\t"

        PHASE1_K_ITER(0, 0)
        PHASE1_K_ITER(4, 4)
        "add x5, x5, #512\n\t"
        "add x4, x4, #32\n\t"
        "add x17, x17, #32\n\t"
        "add x18, x18, #32\n\t"
        "add x19, x19, #32\n\t"

        // Group 16-23: k=32-47
        PHASE1_K_ITER(0, 0)
        PHASE1_K_ITER(4, 4)
        "add x5, x5, #512\n\t"
        "add x4, x4, #32\n\t"
        "add x17, x17, #32\n\t"
        "add x18, x18, #32\n\t"
        "add x19, x19, #32\n\t"

        PHASE1_K_ITER(0, 0)
        PHASE1_K_ITER(4, 4)
        "add x5, x5, #512\n\t"
        "add x4, x4, #32\n\t"
        "add x17, x17, #32\n\t"
        "add x18, x18, #32\n\t"
        "add x19, x19, #32\n\t"

        PHASE1_K_ITER(0, 0)
        PHASE1_K_ITER(4, 4)
        "add x5, x5, #512\n\t"
        "add x4, x4, #32\n\t"
        "add x17, x17, #32\n\t"
        "add x18, x18, #32\n\t"
        "add x19, x19, #32\n\t"

        PHASE1_K_ITER(0, 0)
        PHASE1_K_ITER(4, 4)
        "add x5, x5, #512\n\t"
        "add x4, x4, #32\n\t"
        "add x17, x17, #32\n\t"
        "add x18, x18, #32\n\t"
        "add x19, x19, #32\n\t"

        PHASE1_K_ITER(0, 0)
        PHASE1_K_ITER(4, 4)
        "add x5, x5, #512\n\t"
        "add x4, x4, #32\n\t"
        "add x17, x17, #32\n\t"
        "add x18, x18, #32\n\t"
        "add x19, x19, #32\n\t"

        PHASE1_K_ITER(0, 0)
        PHASE1_K_ITER(4, 4)
        "add x5, x5, #512\n\t"
        "add x4, x4, #32\n\t"
        "add x17, x17, #32\n\t"
        "add x18, x18, #32\n\t"
        "add x19, x19, #32\n\t"

        PHASE1_K_ITER(0, 0)
        PHASE1_K_ITER(4, 4)
        "add x5, x5, #512\n\t"
        "add x4, x4, #32\n\t"
        "add x17, x17, #32\n\t"
        "add x18, x18, #32\n\t"
        "add x19, x19, #32\n\t"

        PHASE1_K_ITER(0, 0)
        PHASE1_K_ITER(4, 4)
        "add x5, x5, #512\n\t"
        "add x4, x4, #32\n\t"
        "add x17, x17, #32\n\t"
        "add x18, x18, #32\n\t"
        "add x19, x19, #32\n\t"

        // Group 24-31: k=48-63
        PHASE1_K_ITER(0, 0)
        PHASE1_K_ITER(4, 4)
        "add x5, x5, #512\n\t"
        "add x4, x4, #32\n\t"
        "add x17, x17, #32\n\t"
        "add x18, x18, #32\n\t"
        "add x19, x19, #32\n\t"

        PHASE1_K_ITER(0, 0)
        PHASE1_K_ITER(4, 4)
        "add x5, x5, #512\n\t"
        "add x4, x4, #32\n\t"
        "add x17, x17, #32\n\t"
        "add x18, x18, #32\n\t"
        "add x19, x19, #32\n\t"

        PHASE1_K_ITER(0, 0)
        PHASE1_K_ITER(4, 4)
        "add x5, x5, #512\n\t"
        "add x4, x4, #32\n\t"
        "add x17, x17, #32\n\t"
        "add x18, x18, #32\n\t"
        "add x19, x19, #32\n\t"

        PHASE1_K_ITER(0, 0)
        PHASE1_K_ITER(4, 4)
        "add x5, x5, #512\n\t"
        "add x4, x4, #32\n\t"
        "add x17, x17, #32\n\t"
        "add x18, x18, #32\n\t"
        "add x19, x19, #32\n\t"

        PHASE1_K_ITER(0, 0)
        PHASE1_K_ITER(4, 4)
        "add x5, x5, #512\n\t"
        "add x4, x4, #32\n\t"
        "add x17, x17, #32\n\t"
        "add x18, x18, #32\n\t"
        "add x19, x19, #32\n\t"

        PHASE1_K_ITER(0, 0)
        PHASE1_K_ITER(4, 4)
        "add x5, x5, #512\n\t"
        "add x4, x4, #32\n\t"
        "add x17, x17, #32\n\t"
        "add x18, x18, #32\n\t"
        "add x19, x19, #32\n\t"

        PHASE1_K_ITER(0, 0)
        PHASE1_K_ITER(4, 4)
        "add x5, x5, #512\n\t"
        "add x4, x4, #32\n\t"
        "add x17, x17, #32\n\t"
        "add x18, x18, #32\n\t"
        "add x19, x19, #32\n\t"

        PHASE1_K_ITER(0, 0)
        PHASE1_K_ITER(4, 4)
        // Last group - no pointer advance needed

        // Store results
        "mov x6, %[S]\n\t"
        "st1w {z0.s}, p1, [x6]\n\t"
        "st1w {z1.s}, p1, [x6, #1, mul vl]\n\t"
        "st1w {z2.s}, p1, [x6, #2, mul vl]\n\t"
        "st1w {z3.s}, p1, [x6, #3, mul vl]\n\t"
        "add x6, x6, #256\n\t"
        "st1w {z4.s}, p1, [x6]\n\t"
        "st1w {z5.s}, p1, [x6, #1, mul vl]\n\t"
        "st1w {z6.s}, p1, [x6, #2, mul vl]\n\t"
        "st1w {z7.s}, p1, [x6, #3, mul vl]\n\t"
        "add x6, x6, #256\n\t"
        "st1w {z8.s}, p1, [x6]\n\t"
        "st1w {z9.s}, p1, [x6, #1, mul vl]\n\t"
        "st1w {z10.s}, p1, [x6, #2, mul vl]\n\t"
        "st1w {z11.s}, p1, [x6, #3, mul vl]\n\t"
        "add x6, x6, #256\n\t"
        "st1w {z12.s}, p1, [x6]\n\t"
        "st1w {z13.s}, p1, [x6, #1, mul vl]\n\t"
        "st1w {z14.s}, p1, [x6, #2, mul vl]\n\t"
        "st1w {z15.s}, p1, [x6, #3, mul vl]\n\t"

        :
        : [Q]"r"(Q), [K]"r"(K), [S]"r"(S)
        : "x4", "x5", "x6", "x17", "x18", "x19",
          "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7",
          "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15",
          "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31",
          "p0", "p1", "memory"
    );
}

// Phase 2 N-group iteration macro
#define PHASE2_N_ITER(v_off, s_off) \
    "ld1b {z24.b}, p0/z, [x5, #" #v_off ", mul vl]\n\t" \
    "ld1b {z25.b}, p0/z, [x5, #" #v_off "+1, mul vl]\n\t" \
    "ld1b {z26.b}, p0/z, [x5, #" #v_off "+2, mul vl]\n\t" \
    "ld1b {z27.b}, p0/z, [x5, #" #v_off "+3, mul vl]\n\t" \
    "ld1rw {z28.s}, p1/z, [x4, #" #s_off "]\n\t" \
    "ld1rw {z29.s}, p1/z, [x17, #" #s_off "]\n\t" \
    "ld1rw {z30.s}, p1/z, [x18, #" #s_off "]\n\t" \
    "ld1rw {z31.s}, p1/z, [x19, #" #s_off "]\n\t" \
    "sdot z0.s, z24.b, z28.b\n\t" \
    "sdot z1.s, z25.b, z28.b\n\t" \
    "sdot z2.s, z26.b, z28.b\n\t" \
    "sdot z3.s, z27.b, z28.b\n\t" \
    "sdot z4.s, z24.b, z29.b\n\t" \
    "sdot z5.s, z25.b, z29.b\n\t" \
    "sdot z6.s, z26.b, z29.b\n\t" \
    "sdot z7.s, z27.b, z29.b\n\t" \
    "sdot z8.s, z24.b, z30.b\n\t" \
    "sdot z9.s, z25.b, z30.b\n\t" \
    "sdot z10.s, z26.b, z30.b\n\t" \
    "sdot z11.s, z27.b, z30.b\n\t" \
    "sdot z12.s, z24.b, z31.b\n\t" \
    "sdot z13.s, z25.b, z31.b\n\t" \
    "sdot z14.s, z26.b, z31.b\n\t" \
    "sdot z15.s, z27.b, z31.b\n\t"

// Phase 2: S[4,64] @ V[64,D] = O[4,D]
// For one D-tile (D=64): 16 N-groups, each with 16 SDOT = 256 SDOT
__attribute__((noinline))
void phase2_full_unroll_4row(
    const int8_t* __restrict__ S,     // [4, 64] quantized scores
    const int8_t* __restrict__ V,     // [64, 64] interleaved as [16, 64, 4]
    int32_t* __restrict__ O           // [4, 64] output
) {
    __asm__ volatile(
        "ptrue p0.b\n\t"
        "ptrue p1.s\n\t"

        // Initialize 16 accumulators
        "eor z0.d, z0.d, z0.d\n\t"
        "eor z1.d, z1.d, z1.d\n\t"
        "eor z2.d, z2.d, z2.d\n\t"
        "eor z3.d, z3.d, z3.d\n\t"
        "eor z4.d, z4.d, z4.d\n\t"
        "eor z5.d, z5.d, z5.d\n\t"
        "eor z6.d, z6.d, z6.d\n\t"
        "eor z7.d, z7.d, z7.d\n\t"
        "eor z8.d, z8.d, z8.d\n\t"
        "eor z9.d, z9.d, z9.d\n\t"
        "eor z10.d, z10.d, z10.d\n\t"
        "eor z11.d, z11.d, z11.d\n\t"
        "eor z12.d, z12.d, z12.d\n\t"
        "eor z13.d, z13.d, z13.d\n\t"
        "eor z14.d, z14.d, z14.d\n\t"
        "eor z15.d, z15.d, z15.d\n\t"

        // Setup pointers
        "mov x4, %[S]\n\t"          // S ptr (row 0)
        "mov x5, %[V]\n\t"          // V ptr
        "add x17, x4, #64\n\t"      // S row 1
        "add x18, x4, #128\n\t"     // S row 2
        "add x19, x4, #192\n\t"     // S row 3

        // V stride per N-group = 64*4 = 256 bytes
        // 16 N-groups fully unrolled

        // N-group 0-1
        PHASE2_N_ITER(0, 0)
        PHASE2_N_ITER(4, 4)
        "add x5, x5, #512\n\t"
        "add x4, x4, #32\n\t"
        "add x17, x17, #32\n\t"
        "add x18, x18, #32\n\t"
        "add x19, x19, #32\n\t"

        // N-group 2-3
        PHASE2_N_ITER(0, 0)
        PHASE2_N_ITER(4, 4)
        "add x5, x5, #512\n\t"
        "add x4, x4, #32\n\t"
        "add x17, x17, #32\n\t"
        "add x18, x18, #32\n\t"
        "add x19, x19, #32\n\t"

        // N-group 4-5
        PHASE2_N_ITER(0, 0)
        PHASE2_N_ITER(4, 4)
        "add x5, x5, #512\n\t"
        "add x4, x4, #32\n\t"
        "add x17, x17, #32\n\t"
        "add x18, x18, #32\n\t"
        "add x19, x19, #32\n\t"

        // N-group 6-7
        PHASE2_N_ITER(0, 0)
        PHASE2_N_ITER(4, 4)
        "add x5, x5, #512\n\t"
        "add x4, x4, #32\n\t"
        "add x17, x17, #32\n\t"
        "add x18, x18, #32\n\t"
        "add x19, x19, #32\n\t"

        // N-group 8-9
        PHASE2_N_ITER(0, 0)
        PHASE2_N_ITER(4, 4)
        "add x5, x5, #512\n\t"
        "add x4, x4, #32\n\t"
        "add x17, x17, #32\n\t"
        "add x18, x18, #32\n\t"
        "add x19, x19, #32\n\t"

        // N-group 10-11
        PHASE2_N_ITER(0, 0)
        PHASE2_N_ITER(4, 4)
        "add x5, x5, #512\n\t"
        "add x4, x4, #32\n\t"
        "add x17, x17, #32\n\t"
        "add x18, x18, #32\n\t"
        "add x19, x19, #32\n\t"

        // N-group 12-13
        PHASE2_N_ITER(0, 0)
        PHASE2_N_ITER(4, 4)
        "add x5, x5, #512\n\t"
        "add x4, x4, #32\n\t"
        "add x17, x17, #32\n\t"
        "add x18, x18, #32\n\t"
        "add x19, x19, #32\n\t"

        // N-group 14-15 (last)
        PHASE2_N_ITER(0, 0)
        PHASE2_N_ITER(4, 4)

        // Store output
        "mov x6, %[O]\n\t"
        "st1w {z0.s}, p1, [x6]\n\t"
        "st1w {z1.s}, p1, [x6, #1, mul vl]\n\t"
        "st1w {z2.s}, p1, [x6, #2, mul vl]\n\t"
        "st1w {z3.s}, p1, [x6, #3, mul vl]\n\t"
        "add x6, x6, #256\n\t"
        "st1w {z4.s}, p1, [x6]\n\t"
        "st1w {z5.s}, p1, [x6, #1, mul vl]\n\t"
        "st1w {z6.s}, p1, [x6, #2, mul vl]\n\t"
        "st1w {z7.s}, p1, [x6, #3, mul vl]\n\t"
        "add x6, x6, #256\n\t"
        "st1w {z8.s}, p1, [x6]\n\t"
        "st1w {z9.s}, p1, [x6, #1, mul vl]\n\t"
        "st1w {z10.s}, p1, [x6, #2, mul vl]\n\t"
        "st1w {z11.s}, p1, [x6, #3, mul vl]\n\t"
        "add x6, x6, #256\n\t"
        "st1w {z12.s}, p1, [x6]\n\t"
        "st1w {z13.s}, p1, [x6, #1, mul vl]\n\t"
        "st1w {z14.s}, p1, [x6, #2, mul vl]\n\t"
        "st1w {z15.s}, p1, [x6, #3, mul vl]\n\t"

        :
        : [S]"r"(S), [V]"r"(V), [O]"r"(O)
        : "x4", "x5", "x6", "x17", "x18", "x19",
          "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7",
          "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15",
          "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31",
          "p0", "p1", "memory"
    );
}

int main() {
    printf("==============================================\n");
    printf("Fused Attention Full Unroll Optimization\n");
    printf("==============================================\n\n");

    const int MR = 4;
    const int N = 64;
    const int D = 256;

    // Allocate aligned buffers
    int8_t* Q = aligned_alloc(64, MR * D);
    int8_t* K = aligned_alloc(64, D * N);      // [256, 64] -> interleaved [64, 64, 4]
    int8_t* V = aligned_alloc(64, N * D);      // [64, 256] -> interleaved
    int32_t* S = aligned_alloc(64, MR * N * sizeof(int32_t));
    int8_t* S_quant = aligned_alloc(64, MR * N);
    int32_t* O = aligned_alloc(64, MR * D * sizeof(int32_t));

    // Initialize
    memset(Q, 1, MR * D);
    memset(K, 1, D * N);
    memset(V, 1, N * D);
    memset(S_quant, 1, MR * N);

    // Phase 1 SDOT count: MR * N * D / 4 = 4 * 64 * 256 / 4 = 16384 / 4 = 4096
    // But per N-tile (16 cols): MR * 16 * D / 4 = 4 * 16 * 256 / 4 = 4096
    // Total for 4 N-tiles: 4096 (just one tile of N=64)
    // Actually: Per iteration: 16 SDOT. 64 K iterations = 1024 SDOT per N-tile
    // 4 N-tiles = 4096 SDOT... wait
    // Let me recalculate:
    // Phase 1: Q[4,256] @ K[256,64]
    // Each SDOT processes 4 int8 values
    // Total multiplications = 4 * 256 * 64 = 65536
    // SDOT count = 65536 / 4 = 16384? No...
    // Actually MR=4 rows, each producing N=64 outputs
    // Each output = dot product of D=256 values = 256/4 = 64 SDOT calls
    // Total = 4 * 64 * 64 = 16384 SDOT... that's too high

    // Let me re-check the kernel structure:
    // Per K iteration (d_group): 4 K loads (for 4 N-tile columns) x 4 Q broadcasts = 16 SDOT
    // 64 K iterations = 64 * 16 = 1024 SDOT for one N-tile (16 columns)
    // For full N=64: 4 N-tiles * 1024 = 4096 SDOT... but that's for 4-row output

    // Hmm, I think I had it wrong. Let me use the numbers from the previous benchmark:
    // Phase 1: 6144 SDOT (for MR=6)
    // For MR=4: 4096 SDOT

    // Phase 2: For MR=4, D=256 (4 D-tiles of 64):
    // Per D-tile: MR * 64 * N/4 = 4 * 64 * 64/4 = 4 * 64 * 16 = 4096/4 = 1024?
    // Actually: Per N-group (4 N values): 16 SDOT. 16 N-groups = 256 SDOT per D-tile
    // 4 D-tiles = 1024 SDOT for Phase 2

    int phase1_sdot = 64 * 16;  // 64 K-iters * 16 SDOT each = 1024 per N-tile
    int n_tiles = 4;
    int total_phase1 = phase1_sdot * n_tiles;  // 4096 SDOT

    int phase2_sdot = 16 * 16;  // 16 N-iters * 16 SDOT each = 256 per D-tile
    int d_tiles = 4;
    int total_phase2 = phase2_sdot * d_tiles;  // 1024 SDOT

    int total_sdot = total_phase1 + total_phase2;  // 5120 SDOT

    printf("Configuration: MR=%d, N=%d, D=%d\n", MR, N, D);
    printf("Phase 1 SDOT: %d N-tiles x %d = %d\n", n_tiles, phase1_sdot, total_phase1);
    printf("Phase 2 SDOT: %d D-tiles x %d = %d\n", d_tiles, phase2_sdot, total_phase2);
    printf("Total SDOT: %d\n", total_sdot);
    printf("Peak: 40 SDOT/tick\n\n");

    // Warmup
    for (int i = 0; i < 100; i++) {
        phase1_full_unroll_4row(Q, K, S);
        phase2_full_unroll_4row(S_quant, V, O);
    }

    // Benchmark Phase 1 only (full unroll)
    int iters = 10000;
    uint64_t start = rdtick();
    for (int i = 0; i < iters; i++) {
        phase1_full_unroll_4row(Q, K, S);
    }
    uint64_t end = rdtick();
    double ticks_p1 = (double)(end - start) / iters;
    double eff_p1 = (double)phase1_sdot / ticks_p1 / 40.0 * 100.0;

    // Benchmark Phase 2 only (full unroll)
    start = rdtick();
    for (int i = 0; i < iters; i++) {
        phase2_full_unroll_4row(S_quant, V, O);
    }
    end = rdtick();
    double ticks_p2 = (double)(end - start) / iters;
    double eff_p2 = (double)phase2_sdot / ticks_p2 / 40.0 * 100.0;

    // Benchmark combined (fused)
    start = rdtick();
    for (int i = 0; i < iters; i++) {
        phase1_full_unroll_4row(Q, K, S);
        phase2_full_unroll_4row(S_quant, V, O);
    }
    end = rdtick();
    double ticks_fused = (double)(end - start) / iters;
    double total_fused_sdot = phase1_sdot + phase2_sdot;  // 1024 + 256 = 1280 per tile
    double eff_fused = total_fused_sdot / ticks_fused / 40.0 * 100.0;

    printf("Approach                            Ticks Efficiency    SDOT/tick\n");
    printf("------------------------------------------------------------------\n");
    printf("Phase 1 full unroll (1 N-tile)   %8.1f     %5.1f%%     %8.2f\n",
           ticks_p1, eff_p1, (double)phase1_sdot / ticks_p1);
    printf("Phase 2 full unroll (1 D-tile)   %8.1f     %5.1f%%     %8.2f\n",
           ticks_p2, eff_p2, (double)phase2_sdot / ticks_p2);
    printf("Fused P1+P2 full unroll          %8.1f     %5.1f%%     %8.2f\n",
           ticks_fused, eff_fused, total_fused_sdot / ticks_fused);

    printf("\nTheoretical minimum (100%% efficiency):\n");
    printf("  Phase 1: %.1f ticks\n", (double)phase1_sdot / 40.0);
    printf("  Phase 2: %.1f ticks\n", (double)phase2_sdot / 40.0);
    printf("  Total: %.1f ticks\n", total_fused_sdot / 40.0);

    free(Q); free(K); free(V); free(S); free(S_quant); free(O);
    return 0;
}
