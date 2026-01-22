/*
 * Phase Fusion Optimization Analysis
 *
 * Current bottleneck: Phase 2 at 87% efficiency
 *
 * Optimization ideas:
 * 1. Fuse quantization into Phase 1 output (reduce memory traffic)
 * 2. Compute row max during Phase 1 (for softmax preparation)
 * 3. Reduce Phase 2 K-tiles from 4 to 2 (simpler reduction)
 * 4. Online softmax with chunked processing
 *
 * This benchmark analyzes the overhead breakdown and tests fusion approaches.
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
 * Current Phase 2 structure (87% efficiency):
 *   - 4 D-tile iterations
 *   - Per D-tile: 24 EOR + 384 SDOT + 18 ADD + 6 ST1W
 *   - Total: 96 EOR + 1536 SDOT + 72 ADD + 24 ST1W
 *
 * Overhead analysis:
 *   - EOR uses EXA pipe (doesn't compete with FPU but adds cycles)
 *   - ADD uses FLA/FLB pipe (competes with SDOT)
 *   - Reduction: 18 ADD per D-tile = 72 ADD total (36 cycles at 2/cycle)
 *
 * Key insight: Reduction overhead scales with K-tiles
 *   - 4 K-tiles → need 3 ADD to reduce to 1 output
 *   - 2 K-tiles → need only 1 ADD to reduce to 1 output
 */

// ============================================================
// Approach 1: Reduce K-tiles from 4 to 2
// ============================================================
// If we process 2 K-tiles at a time instead of 4:
//   - Each D-tile: 12 accumulators (2 K-tiles × 6 rows)
//   - Final reduction: only 6 ADD per D-tile (was 18)
//   - But: need 2 passes over V instead of 1
//
// Trade-off:
//   - Saves: 48 ADD (72 → 24)
//   - Costs: 2x V memory reads, 2x zeroing
//
// Verdict: Probably not worth it due to memory overhead

// Measure 2-Ktile approach
__attribute__((noinline))
void phase2_2ktile(const int32_t* S, const int8_t* V, int32_t* O) {
    // First pass: K-tiles 0-1
    __asm__ volatile(
        "ptrue p0.b\n\t"
        "ptrue p1.s\n\t"
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
        // Load only K-tiles 0-1
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
        // Second N iteration
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
        // Simpler reduction: only 1 ADD per row
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
        :
        : [S] "r"(S), [V] "r"(V), [O] "r"(O)
        : "memory", "x4", "x5", "x6", "x7", "x10", "x11", "x12", "x13",
          "x15", "x16", "x17",
          "z0", "z1", "z4", "z8", "z9", "z10", "z11", "z12", "z13",
          "z14", "z15", "z16", "z17", "z18", "z19", "p0", "p1"
    );

    // Second pass: K-tiles 2-3, accumulate into O
    __asm__ volatile(
        "ptrue p0.b\n\t"
        "ptrue p1.s\n\t"
        "mov x15, #4\n\t"
        "mov x16, %[V]\n\t"
        "add x16, x16, #128\n\t"  // Start at K-tile 2
        "mov x17, %[O]\n\t"
        "2:\n\t"
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
        "3:\n\t"
        // Load K-tiles 2-3
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
        "b.ne 3b\n\t"
        // Reduction and accumulate with previous results
        "add z8.s, z8.s, z9.s\n\t"
        "add z10.s, z10.s, z11.s\n\t"
        "add z12.s, z12.s, z13.s\n\t"
        "add z14.s, z14.s, z15.s\n\t"
        "add z16.s, z16.s, z17.s\n\t"
        "add z18.s, z18.s, z19.s\n\t"
        // Load previous results and add
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
        "b.ne 2b\n\t"
        :
        : [S] "r"(S), [V] "r"(V), [O] "r"(O)
        : "memory", "x4", "x5", "x6", "x7", "x10", "x11", "x12", "x13",
          "x15", "x16", "x17",
          "z0", "z1", "z4", "z8", "z9", "z10", "z11", "z12", "z13",
          "z14", "z15", "z16", "z17", "z18", "z19", "p0", "p1"
    );
}

// ============================================================
// Approach 2: Fuse row max computation into Phase 1
// ============================================================
// During Phase 1, track row maximum for softmax preparation
// This helps Phase 2 by pre-computing softmax scaling

__attribute__((noinline))
void phase1_with_max(const int8_t* Q, const int8_t* K, int32_t* S, int32_t* row_max) {
    // Phase 1 with row max tracking
    // Compute Q @ K^T and track max per row
    __asm__ volatile(
        "ptrue p0.b\n\t"
        "ptrue p1.s\n\t"
        // Initialize row max to minimum
        "mov w3, #0x80000000\n\t"  // INT32_MIN
        "mov z28.s, w3\n\t"
        "mov z29.s, w3\n\t"
        "mov z30.s, w3\n\t"
        "mov z31.s, w3\n\t"

        // Standard Phase 1 computation with max tracking
        "mov x4, %[Q]\n\t"
        "mov x5, %[K]\n\t"
        "mov x6, %[S]\n\t"
        "mov x10, #16\n\t"  // 64 K values / 4 = 16 groups

        "eor z0.d, z0.d, z0.d\n\t"
        "eor z1.d, z1.d, z1.d\n\t"
        "eor z2.d, z2.d, z2.d\n\t"
        "eor z3.d, z3.d, z3.d\n\t"

        "1:\n\t"
        "ld1b {z8.b}, p0/z, [x5]\n\t"
        "ld1rw {z16.s}, p1/z, [x4]\n\t"
        "sdot z0.s, z8.b, z16.b\n\t"
        "ld1rw {z16.s}, p1/z, [x4, #256]\n\t"
        "sdot z1.s, z8.b, z16.b\n\t"
        "ld1rw {z16.s}, p1/z, [x4, #512]\n\t"
        "sdot z2.s, z8.b, z16.b\n\t"
        "ld1rw {z16.s}, p1/z, [x4, #768]\n\t"
        "sdot z3.s, z8.b, z16.b\n\t"
        "add x4, x4, #4\n\t"
        "add x5, x5, #64\n\t"
        "subs x10, x10, #1\n\t"
        "b.ne 1b\n\t"

        // Update row max
        "smax z28.s, p1/m, z28.s, z0.s\n\t"
        "smax z29.s, p1/m, z29.s, z1.s\n\t"
        "smax z30.s, p1/m, z30.s, z2.s\n\t"
        "smax z31.s, p1/m, z31.s, z3.s\n\t"

        // Store S values
        "st1w {z0.s}, p1, [x6]\n\t"
        "st1w {z1.s}, p1, [x6, #256]\n\t"
        "st1w {z2.s}, p1, [x6, #512]\n\t"
        "st1w {z3.s}, p1, [x6, #768]\n\t"

        // Reduce and store row max (horizontal max)
        "smaxv s28, p1, z28.s\n\t"
        "smaxv s29, p1, z29.s\n\t"
        "smaxv s30, p1, z30.s\n\t"
        "smaxv s31, p1, z31.s\n\t"
        "mov x7, %[row_max]\n\t"
        "str s28, [x7]\n\t"
        "str s29, [x7, #4]\n\t"
        "str s30, [x7, #8]\n\t"
        "str s31, [x7, #12]\n\t"
        :
        : [Q] "r"(Q), [K] "r"(K), [S] "r"(S), [row_max] "r"(row_max)
        : "memory", "x3", "x4", "x5", "x6", "x7", "x10",
          "z0", "z1", "z2", "z3", "z8", "z16",
          "z28", "z29", "z30", "z31", "p0", "p1"
    );
}

// ============================================================
// Approach 3: Baseline Phase 2 for reference
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
// Approach 4: Chunked N processing (online softmax style)
// ============================================================
// Process N in smaller chunks, fusing Phase 1 and Phase 2
// This reduces intermediate storage and improves cache efficiency

__attribute__((noinline))
void fused_chunked_n4(const int8_t* Q, const int8_t* K, const int8_t* V,
                      int32_t* O, int N, int D) {
    // Process N in chunks of 4
    // Each chunk: compute S_chunk[6,4], quantize, compute O_chunk[6,D]
    // This is a simplified version - real online softmax would track running max/sum

    int32_t S_chunk[6 * 4] __attribute__((aligned(64)));
    int8_t P_chunk[6 * 4] __attribute__((aligned(64)));

    // Initialize O to zero
    memset(O, 0, 6 * D * sizeof(int32_t));

    for (int n_start = 0; n_start < N; n_start += 4) {
        // Phase 1 chunk: Q[6,D] @ K[D,4] -> S_chunk[6,4]
        // Simplified: just compute the chunk
        for (int row = 0; row < 6; row++) {
            for (int col = 0; col < 4; col++) {
                int32_t sum = 0;
                for (int k = 0; k < D; k++) {
                    sum += (int32_t)Q[row * D + k] * (int32_t)K[(n_start + col) * D + k];
                }
                S_chunk[row * 4 + col] = sum;
            }
        }

        // Quantize S_chunk -> P_chunk (simplified)
        for (int i = 0; i < 24; i++) {
            P_chunk[i] = (int8_t)(S_chunk[i] >> 8);  // Simple shift quantization
        }

        // Phase 2 chunk: P_chunk[6,4] @ V[4,D] -> accumulate into O[6,D]
        for (int row = 0; row < 6; row++) {
            for (int d = 0; d < D; d++) {
                int32_t sum = 0;
                for (int n = 0; n < 4; n++) {
                    sum += (int32_t)P_chunk[row * 4 + n] * (int32_t)V[(n_start + n) * D + d];
                }
                O[row * D + d] += sum;
            }
        }
    }
}

int main() {
    printf("==============================================\n");
    printf("Phase Fusion Optimization Analysis\n");
    printf("==============================================\n\n");

    int D = 256;
    int N = 64;

    int32_t* S = aligned_alloc(256, 6 * N * 4);
    int8_t* V = aligned_alloc(256, N * D * 4);
    int32_t* O1 = aligned_alloc(256, 6 * D * 4);
    int32_t* O2 = aligned_alloc(256, 6 * D * 4);

    memset(S, 1, 6 * N * 4);
    memset(V, 1, N * D * 4);

    int iters = 10000;
    int phase2_sdot = 1536;
    double peak_sdot_per_tick = 40.0;

    printf("Optimization approaches for 95%%+ efficiency:\n\n");

    printf("Current bottleneck: Phase 2 at 87%% efficiency\n");
    printf("  - 1536 SDOT operations\n");
    printf("  - 4 D-tile iterations with overhead per iteration\n\n");

    // Warmup
    for (int i = 0; i < 100; i++) {
        phase2_baseline(S, V, O1);
        phase2_2ktile(S, V, O2);
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

    // Benchmark 2-Ktile approach
    start = rdcycle();
    for (int i = 0; i < iters; i++) {
        phase2_2ktile(S, V, O2);
    }
    end = rdcycle();
    double ticks_2kt = (double)(end - start) / iters;
    double sdot_2kt = phase2_sdot / ticks_2kt;
    double eff_2kt = sdot_2kt / peak_sdot_per_tick * 100;

    printf("%-25s %10s %12s %10s\n", "Approach", "Ticks", "SDOT/tick", "Efficiency");
    printf("----------------------------------------------------------\n");
    printf("%-25s %10.1f %12.2f %9.1f%%\n", "Baseline (4 K-tiles)", ticks_base, sdot_base, eff_base);
    printf("%-25s %10.1f %12.2f %9.1f%%\n", "2-Ktile (2 passes)", ticks_2kt, sdot_2kt, eff_2kt);

    printf("\n");
    printf("Analysis:\n");
    printf("=========\n");
    if (eff_2kt > eff_base) {
        printf("2-Ktile approach is %.1f%% more efficient!\n", eff_2kt - eff_base);
        printf("Simpler reduction (6 ADD vs 18 ADD) helps.\n");
    } else {
        printf("2-Ktile approach is %.1f%% less efficient.\n", eff_base - eff_2kt);
        printf("Extra memory passes hurt more than simpler reduction helps.\n");
    }

    printf("\n");
    printf("Alternative optimization strategies:\n");
    printf("=====================================\n");
    printf("1. Online softmax (Flash Attention style):\n");
    printf("   - Process N in small chunks (e.g., N=4 at a time)\n");
    printf("   - Fuse Phase 1 quantize and Phase 2 in same loop\n");
    printf("   - Track running max/sum for incremental softmax\n");
    printf("   - Benefit: Reduces intermediate storage, better cache use\n\n");

    printf("2. Pre-compute row statistics in Phase 1:\n");
    printf("   - Track row max during Phase 1 computation\n");
    printf("   - Use for softmax scaling without extra pass\n");
    printf("   - Minimal overhead (SMAX is 1 cycle)\n\n");

    printf("3. Packed V layout for 2 K-tiles:\n");
    printf("   - Repack V as [N, D/2, 2, 4] instead of [N, D, 4]\n");
    printf("   - Each D-tile needs only 2 K-tiles\n");
    printf("   - Halves reduction overhead\n\n");

    printf("4. MR=4 instead of MR=6:\n");
    printf("   - 4 rows × 4 K-tiles = 16 accumulators (vs 24)\n");
    printf("   - Could process 2 D-tiles in parallel\n");
    printf("   - Trade-off: More outer iterations for rows\n");

    free(S); free(V); free(O1); free(O2);
    return 0;
}
