/*
 * Eliminate reduction overhead by incremental accumulation
 *
 * Current approach (91% eff):
 *   - Accumulate into 4 registers per row (4 K-tiles)
 *   - Reduce: 3 ADD per row × 6 rows = 18 ADD per phase
 *   - Total: 36 ADD for both phases
 *
 * New approach:
 *   - Use only 1 register per row
 *   - Accumulate incrementally (load each K-tile, SDOT, repeat)
 *   - Zero reduction overhead
 *
 * Trade-off: More sequential loads vs parallel accumulation
 */

#include <arm_sve.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static inline uint64_t rdcycle(void) {
    uint64_t val;
    __asm__ volatile("mrs %0, cntvct_el0" : "=r"(val));
    return val;
}

// Current approach: parallel accumulation + reduction
__attribute__((noinline))
void phase1_parallel(const int8_t* Q, const int8_t* K, int32_t* S) {
    __asm__ volatile(
        "ptrue p0.b\n\t"
        "ptrue p1.s\n\t"
        // Zero all 24 accumulators with EOR
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
        "mov x4, %[K]\n\t"
        "mov x10, #32\n\t"
        "1:\n\t"
        // Load 4 K-tiles in parallel
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
        "add x4, x4, #256\n\t"
        "add x5, x5, #4\n\t"   "add x6, x6, #4\n\t"   "add x7, x7, #4\n\t"
        "add x11, x11, #4\n\t" "add x12, x12, #4\n\t" "add x13, x13, #4\n\t"
        "subs x10, x10, #1\n\t"
        "b.ne 1b\n\t"
        // Reduce: 18 ADD instructions (3 per row)
        "add z8.s, z8.s, z9.s\n\t"   "add z10.s, z10.s, z11.s\n\t" "add z8.s, z8.s, z10.s\n\t"
        "add z12.s, z12.s, z13.s\n\t" "add z14.s, z14.s, z15.s\n\t" "add z12.s, z12.s, z14.s\n\t"
        "add z16.s, z16.s, z17.s\n\t" "add z18.s, z18.s, z19.s\n\t" "add z16.s, z16.s, z18.s\n\t"
        "add z20.s, z20.s, z21.s\n\t" "add z22.s, z22.s, z23.s\n\t" "add z20.s, z20.s, z22.s\n\t"
        "add z24.s, z24.s, z25.s\n\t" "add z26.s, z26.s, z27.s\n\t" "add z24.s, z24.s, z26.s\n\t"
        "add z28.s, z28.s, z29.s\n\t" "add z30.s, z30.s, z31.s\n\t" "add z28.s, z28.s, z30.s\n\t"
        "mov x14, %[S]\n\t"
        "st1w {z8.s}, p1, [x14]\n\t"       "st1w {z12.s}, p1, [x14, #1, mul vl]\n\t"
        "st1w {z16.s}, p1, [x14, #2, mul vl]\n\t" "st1w {z20.s}, p1, [x14, #3, mul vl]\n\t"
        "st1w {z24.s}, p1, [x14, #4, mul vl]\n\t" "st1w {z28.s}, p1, [x14, #5, mul vl]\n\t"
        :
        : [Q] "r"(Q), [K] "r"(K), [S] "r"(S)
        : "memory", "x4", "x5", "x6", "x7", "x10", "x11", "x12", "x13", "x14",
          "z0", "z1", "z2", "z3", "z4", "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15",
          "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23",
          "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31", "p0", "p1"
    );
}

// New approach: incremental accumulation (no reduction)
__attribute__((noinline))
void phase1_incremental(const int8_t* Q, const int8_t* K, int32_t* S) {
    __asm__ volatile(
        "ptrue p0.b\n\t"
        "ptrue p1.s\n\t"
        // Zero only 6 accumulators (one per row)
        "eor z8.d, z8.d, z8.d\n\t"   "eor z9.d, z9.d, z9.d\n\t"
        "eor z10.d, z10.d, z10.d\n\t" "eor z11.d, z11.d, z11.d\n\t"
        "eor z12.d, z12.d, z12.d\n\t" "eor z13.d, z13.d, z13.d\n\t"
        "mov x5, %[Q]\n\t"
        "add x6, x5, #256\n\t"  "add x7, x6, #256\n\t"
        "add x11, x7, #256\n\t" "add x12, x11, #256\n\t" "add x13, x12, #256\n\t"
        "mov x4, %[K]\n\t"
        "mov x10, #32\n\t"
        "1:\n\t"
        // Process all 4 K-tiles sequentially, accumulating to same register
        // K-tile 0
        "ld1b {z0.b}, p0/z, [x4]\n\t"
        "ld1rw {z4.s}, p1/z, [x5]\n\t"
        "sdot z8.s, z0.b, z4.b\n\t"
        "ld1rw {z4.s}, p1/z, [x6]\n\t"
        "sdot z9.s, z0.b, z4.b\n\t"
        "ld1rw {z4.s}, p1/z, [x7]\n\t"
        "sdot z10.s, z0.b, z4.b\n\t"
        "ld1rw {z4.s}, p1/z, [x11]\n\t"
        "sdot z11.s, z0.b, z4.b\n\t"
        "ld1rw {z4.s}, p1/z, [x12]\n\t"
        "sdot z12.s, z0.b, z4.b\n\t"
        "ld1rw {z4.s}, p1/z, [x13]\n\t"
        "sdot z13.s, z0.b, z4.b\n\t"
        // K-tile 1
        "ld1b {z0.b}, p0/z, [x4, #1, mul vl]\n\t"
        "ld1rw {z4.s}, p1/z, [x5]\n\t"
        "sdot z8.s, z0.b, z4.b\n\t"
        "ld1rw {z4.s}, p1/z, [x6]\n\t"
        "sdot z9.s, z0.b, z4.b\n\t"
        "ld1rw {z4.s}, p1/z, [x7]\n\t"
        "sdot z10.s, z0.b, z4.b\n\t"
        "ld1rw {z4.s}, p1/z, [x11]\n\t"
        "sdot z11.s, z0.b, z4.b\n\t"
        "ld1rw {z4.s}, p1/z, [x12]\n\t"
        "sdot z12.s, z0.b, z4.b\n\t"
        "ld1rw {z4.s}, p1/z, [x13]\n\t"
        "sdot z13.s, z0.b, z4.b\n\t"
        // K-tile 2
        "ld1b {z0.b}, p0/z, [x4, #2, mul vl]\n\t"
        "ld1rw {z4.s}, p1/z, [x5]\n\t"
        "sdot z8.s, z0.b, z4.b\n\t"
        "ld1rw {z4.s}, p1/z, [x6]\n\t"
        "sdot z9.s, z0.b, z4.b\n\t"
        "ld1rw {z4.s}, p1/z, [x7]\n\t"
        "sdot z10.s, z0.b, z4.b\n\t"
        "ld1rw {z4.s}, p1/z, [x11]\n\t"
        "sdot z11.s, z0.b, z4.b\n\t"
        "ld1rw {z4.s}, p1/z, [x12]\n\t"
        "sdot z12.s, z0.b, z4.b\n\t"
        "ld1rw {z4.s}, p1/z, [x13]\n\t"
        "sdot z13.s, z0.b, z4.b\n\t"
        // K-tile 3
        "ld1b {z0.b}, p0/z, [x4, #3, mul vl]\n\t"
        "ld1rw {z4.s}, p1/z, [x5]\n\t"
        "sdot z8.s, z0.b, z4.b\n\t"
        "ld1rw {z4.s}, p1/z, [x6]\n\t"
        "sdot z9.s, z0.b, z4.b\n\t"
        "ld1rw {z4.s}, p1/z, [x7]\n\t"
        "sdot z10.s, z0.b, z4.b\n\t"
        "ld1rw {z4.s}, p1/z, [x11]\n\t"
        "sdot z11.s, z0.b, z4.b\n\t"
        "ld1rw {z4.s}, p1/z, [x12]\n\t"
        "sdot z12.s, z0.b, z4.b\n\t"
        "ld1rw {z4.s}, p1/z, [x13]\n\t"
        "sdot z13.s, z0.b, z4.b\n\t"
        "add x4, x4, #256\n\t"
        "add x5, x5, #4\n\t"   "add x6, x6, #4\n\t"   "add x7, x7, #4\n\t"
        "add x11, x11, #4\n\t" "add x12, x12, #4\n\t" "add x13, x13, #4\n\t"
        "subs x10, x10, #1\n\t"
        "b.ne 1b\n\t"
        // No reduction! Direct store
        "mov x14, %[S]\n\t"
        "st1w {z8.s}, p1, [x14]\n\t"       "st1w {z9.s}, p1, [x14, #1, mul vl]\n\t"
        "st1w {z10.s}, p1, [x14, #2, mul vl]\n\t" "st1w {z11.s}, p1, [x14, #3, mul vl]\n\t"
        "st1w {z12.s}, p1, [x14, #4, mul vl]\n\t" "st1w {z13.s}, p1, [x14, #5, mul vl]\n\t"
        :
        : [Q] "r"(Q), [K] "r"(K), [S] "r"(S)
        : "memory", "x4", "x5", "x6", "x7", "x10", "x11", "x12", "x13", "x14",
          "z0", "z4", "z8", "z9", "z10", "z11", "z12", "z13", "p0", "p1"
    );
}

// Hybrid: 2x parallel + 1 reduction per row
__attribute__((noinline))
void phase1_hybrid(const int8_t* Q, const int8_t* K, int32_t* S) {
    __asm__ volatile(
        "ptrue p0.b\n\t"
        "ptrue p1.s\n\t"
        // Zero 12 accumulators (2 K-tile groups per row)
        "eor z8.d, z8.d, z8.d\n\t"   "eor z9.d, z9.d, z9.d\n\t"
        "eor z10.d, z10.d, z10.d\n\t" "eor z11.d, z11.d, z11.d\n\t"
        "eor z12.d, z12.d, z12.d\n\t" "eor z13.d, z13.d, z13.d\n\t"
        "eor z14.d, z14.d, z14.d\n\t" "eor z15.d, z15.d, z15.d\n\t"
        "eor z16.d, z16.d, z16.d\n\t" "eor z17.d, z17.d, z17.d\n\t"
        "eor z18.d, z18.d, z18.d\n\t" "eor z19.d, z19.d, z19.d\n\t"
        "mov x5, %[Q]\n\t"
        "add x6, x5, #256\n\t"  "add x7, x6, #256\n\t"
        "add x11, x7, #256\n\t" "add x12, x11, #256\n\t" "add x13, x12, #256\n\t"
        "mov x4, %[K]\n\t"
        "mov x10, #32\n\t"
        "1:\n\t"
        // Load K-tiles 0-1 (first group)
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
        // Load K-tiles 2-3 (second group) - accumulate to same registers
        "ld1b {z0.b}, p0/z, [x4, #2, mul vl]\n\t"
        "ld1b {z1.b}, p0/z, [x4, #3, mul vl]\n\t"
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
        "add x4, x4, #256\n\t"
        "add x5, x5, #4\n\t"   "add x6, x6, #4\n\t"   "add x7, x7, #4\n\t"
        "add x11, x11, #4\n\t" "add x12, x12, #4\n\t" "add x13, x13, #4\n\t"
        "subs x10, x10, #1\n\t"
        "b.ne 1b\n\t"
        // Reduced reduction: 1 ADD per row (6 total vs 18)
        "add z8.s, z8.s, z9.s\n\t"
        "add z10.s, z10.s, z11.s\n\t"
        "add z12.s, z12.s, z13.s\n\t"
        "add z14.s, z14.s, z15.s\n\t"
        "add z16.s, z16.s, z17.s\n\t"
        "add z18.s, z18.s, z19.s\n\t"
        "mov x14, %[S]\n\t"
        "st1w {z8.s}, p1, [x14]\n\t"       "st1w {z10.s}, p1, [x14, #1, mul vl]\n\t"
        "st1w {z12.s}, p1, [x14, #2, mul vl]\n\t" "st1w {z14.s}, p1, [x14, #3, mul vl]\n\t"
        "st1w {z16.s}, p1, [x14, #4, mul vl]\n\t" "st1w {z18.s}, p1, [x14, #5, mul vl]\n\t"
        :
        : [Q] "r"(Q), [K] "r"(K), [S] "r"(S)
        : "memory", "x4", "x5", "x6", "x7", "x10", "x11", "x12", "x13", "x14",
          "z0", "z1", "z4", "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15",
          "z16", "z17", "z18", "z19", "p0", "p1"
    );
}

int main() {
    printf("==============================================\n");
    printf("Reduction Overhead Elimination Experiments\n");
    printf("==============================================\n\n");

    int D = 256;
    int N = 64;
    int D_groups = D / 4;

    int8_t* Q = aligned_alloc(256, 6 * D);
    int8_t* K = aligned_alloc(256, D_groups * N * 4);
    int32_t* S = aligned_alloc(256, 6 * N * 4);

    memset(Q, 1, 6 * D);
    memset(K, 1, D_groups * N * 4);

    int iters = 10000;
    int phase1_sdot = 1536;
    double peak_sdot_per_tick = 40.0;

    printf("Phase 1 only test (Q@K^T, 1536 SDOTs)\n");
    printf("Comparing accumulation strategies:\n\n");
    printf("1. Parallel (current): 4 K-tiles parallel → 18 ADD reduction\n");
    printf("   - 24 accumulators (4 per row × 6 rows)\n");
    printf("   - Reduction: 3 ADD per row × 6 = 18 ADD\n\n");
    printf("2. Incremental: Process K-tiles sequentially → 0 ADD reduction\n");
    printf("   - 6 accumulators (1 per row)\n");
    printf("   - No reduction needed\n");
    printf("   - Trade-off: More sequential loads\n\n");
    printf("3. Hybrid: 2 K-tile groups → 6 ADD reduction\n");
    printf("   - 12 accumulators (2 per row × 6 rows)\n");
    printf("   - Reduction: 1 ADD per row × 6 = 6 ADD\n\n");

    // Warmup
    for (int i = 0; i < 100; i++) {
        phase1_parallel(Q, K, S);
        phase1_incremental(Q, K, S);
        phase1_hybrid(Q, K, S);
    }

    // Benchmark parallel (current best)
    uint64_t start = rdcycle();
    for (int i = 0; i < iters; i++) {
        phase1_parallel(Q, K, S);
    }
    uint64_t end = rdcycle();
    double ticks_par = (double)(end - start) / iters;
    double sdot_par = phase1_sdot / ticks_par;
    double eff_par = sdot_par / peak_sdot_per_tick * 100;

    // Benchmark incremental
    start = rdcycle();
    for (int i = 0; i < iters; i++) {
        phase1_incremental(Q, K, S);
    }
    end = rdcycle();
    double ticks_inc = (double)(end - start) / iters;
    double sdot_inc = phase1_sdot / ticks_inc;
    double eff_inc = sdot_inc / peak_sdot_per_tick * 100;

    // Benchmark hybrid
    start = rdcycle();
    for (int i = 0; i < iters; i++) {
        phase1_hybrid(Q, K, S);
    }
    end = rdcycle();
    double ticks_hyb = (double)(end - start) / iters;
    double sdot_hyb = phase1_sdot / ticks_hyb;
    double eff_hyb = sdot_hyb / peak_sdot_per_tick * 100;

    printf("%-20s %10s %12s %10s %12s\n",
           "Method", "Ticks", "SDOT/tick", "Efficiency", "Reduction");
    printf("-------------------------------------------------------------------------\n");
    printf("%-20s %10.1f %12.2f %9.1f%% %12s\n",
           "Parallel (4-way)", ticks_par, sdot_par, eff_par, "18 ADD");
    printf("%-20s %10.1f %12.2f %9.1f%% %12s\n",
           "Incremental (1-way)", ticks_inc, sdot_inc, eff_inc, "0 ADD");
    printf("%-20s %10.1f %12.2f %9.1f%% %12s\n",
           "Hybrid (2-way)", ticks_hyb, sdot_hyb, eff_hyb, "6 ADD");

    printf("\n");
    printf("Speedup vs Parallel:\n");
    printf("  Incremental: %.1f%% %s\n",
           fabs((ticks_par - ticks_inc) / ticks_par * 100),
           ticks_inc < ticks_par ? "faster" : "slower");
    printf("  Hybrid:      %.1f%% %s\n",
           fabs((ticks_par - ticks_hyb) / ticks_par * 100),
           ticks_hyb < ticks_par ? "faster" : "slower");

    if (ticks_inc < ticks_par || ticks_hyb < ticks_par) {
        printf("\nReduction elimination helps!\n");
        if (ticks_inc < ticks_hyb) {
            printf("Best: Incremental (no reduction)\n");
        } else {
            printf("Best: Hybrid (reduced reduction)\n");
        }
    } else {
        printf("\nParallel accumulation + reduction is still best.\n");
        printf("Instruction-level parallelism outweighs reduction overhead.\n");
    }

    free(Q); free(K); free(S);
    return 0;
}
