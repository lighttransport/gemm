/*
 * Full K-Loop Unrolling Benchmark
 *
 * Goal: Eliminate ALL loop overhead to reach 95%+ SDOT efficiency
 *
 * Strategy:
 * - Fully unroll the K-loop (no branches in hot path)
 * - Use a code generator approach with macros
 * - Compare against baseline with loop
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

// Macro for one K-group iteration (4 K-tiles, 6 rows = 24 SDOT)
#define K_GROUP_ITER(k_offset, q_offset) \
    "ld1b {z0.b}, p0/z, [x4, #" #k_offset ", mul vl]\n\t" \
    "ld1b {z1.b}, p0/z, [x4, #" #k_offset "+1, mul vl]\n\t" \
    "ld1b {z2.b}, p0/z, [x4, #" #k_offset "+2, mul vl]\n\t" \
    "ld1b {z3.b}, p0/z, [x4, #" #k_offset "+3, mul vl]\n\t" \
    "ld1rw {z4.s}, p1/z, [x5, #" #q_offset "]\n\t" \
    "sdot z8.s, z0.b, z4.b\n\t"  "sdot z9.s, z1.b, z4.b\n\t" \
    "sdot z10.s, z2.b, z4.b\n\t" "sdot z11.s, z3.b, z4.b\n\t" \
    "ld1rw {z4.s}, p1/z, [x6, #" #q_offset "]\n\t" \
    "sdot z12.s, z0.b, z4.b\n\t" "sdot z13.s, z1.b, z4.b\n\t" \
    "sdot z14.s, z2.b, z4.b\n\t" "sdot z15.s, z3.b, z4.b\n\t" \
    "ld1rw {z4.s}, p1/z, [x7, #" #q_offset "]\n\t" \
    "sdot z16.s, z0.b, z4.b\n\t" "sdot z17.s, z1.b, z4.b\n\t" \
    "sdot z18.s, z2.b, z4.b\n\t" "sdot z19.s, z3.b, z4.b\n\t" \
    "ld1rw {z4.s}, p1/z, [x11, #" #q_offset "]\n\t" \
    "sdot z20.s, z0.b, z4.b\n\t" "sdot z21.s, z1.b, z4.b\n\t" \
    "sdot z22.s, z2.b, z4.b\n\t" "sdot z23.s, z3.b, z4.b\n\t" \
    "ld1rw {z4.s}, p1/z, [x12, #" #q_offset "]\n\t" \
    "sdot z24.s, z0.b, z4.b\n\t" "sdot z25.s, z1.b, z4.b\n\t" \
    "sdot z26.s, z2.b, z4.b\n\t" "sdot z27.s, z3.b, z4.b\n\t" \
    "ld1rw {z4.s}, p1/z, [x13, #" #q_offset "]\n\t" \
    "sdot z28.s, z0.b, z4.b\n\t" "sdot z29.s, z1.b, z4.b\n\t" \
    "sdot z30.s, z2.b, z4.b\n\t" "sdot z31.s, z3.b, z4.b\n\t"

// Fully unrolled kernel for first 8 K-groups (uses mul vl offsets 0-7)
// Then advance pointer and do next 8 K-groups
__attribute__((noinline))
void fully_unrolled_kernel(const int8_t* Q, const int8_t* K, int32_t* S) {
    __asm__ volatile(
        "ptrue p0.b\n\t"
        "ptrue p1.s\n\t"
        // Zero accumulators
        "mov z8.s, #0\n\t"  "mov z9.s, #0\n\t"  "mov z10.s, #0\n\t" "mov z11.s, #0\n\t"
        "mov z12.s, #0\n\t" "mov z13.s, #0\n\t" "mov z14.s, #0\n\t" "mov z15.s, #0\n\t"
        "mov z16.s, #0\n\t" "mov z17.s, #0\n\t" "mov z18.s, #0\n\t" "mov z19.s, #0\n\t"
        "mov z20.s, #0\n\t" "mov z21.s, #0\n\t" "mov z22.s, #0\n\t" "mov z23.s, #0\n\t"
        "mov z24.s, #0\n\t" "mov z25.s, #0\n\t" "mov z26.s, #0\n\t" "mov z27.s, #0\n\t"
        "mov z28.s, #0\n\t" "mov z29.s, #0\n\t" "mov z30.s, #0\n\t" "mov z31.s, #0\n\t"

        // Q row pointers
        "mov x5, %[Q]\n\t"
        "add x6, x5, #256\n\t"  "add x7, x6, #256\n\t"
        "add x11, x7, #256\n\t" "add x12, x11, #256\n\t" "add x13, x12, #256\n\t"
        "mov x4, %[K]\n\t"

        // K-groups 0-1 (q_offset 0, 4)
        K_GROUP_ITER(0, 0)
        K_GROUP_ITER(4, 4)

        // Advance K pointer
        "add x4, x4, #512\n\t"

        // K-groups 2-3 (q_offset 8, 12)
        K_GROUP_ITER(0, 8)
        K_GROUP_ITER(4, 12)

        "add x4, x4, #512\n\t"

        // K-groups 4-5 (q_offset 16, 20)
        K_GROUP_ITER(0, 16)
        K_GROUP_ITER(4, 20)

        "add x4, x4, #512\n\t"

        // K-groups 6-7 (q_offset 24, 28)
        K_GROUP_ITER(0, 24)
        K_GROUP_ITER(4, 28)

        "add x4, x4, #512\n\t"

        // K-groups 8-9 (q_offset 32, 36)
        K_GROUP_ITER(0, 32)
        K_GROUP_ITER(4, 36)

        "add x4, x4, #512\n\t"

        // K-groups 10-11 (q_offset 40, 44)
        K_GROUP_ITER(0, 40)
        K_GROUP_ITER(4, 44)

        "add x4, x4, #512\n\t"

        // K-groups 12-13 (q_offset 48, 52)
        K_GROUP_ITER(0, 48)
        K_GROUP_ITER(4, 52)

        "add x4, x4, #512\n\t"

        // K-groups 14-15 (q_offset 56, 60)
        K_GROUP_ITER(0, 56)
        K_GROUP_ITER(4, 60)

        "add x4, x4, #512\n\t"

        // K-groups 16-17 (q_offset 64, 68)
        K_GROUP_ITER(0, 64)
        K_GROUP_ITER(4, 68)

        "add x4, x4, #512\n\t"

        // K-groups 18-19 (q_offset 72, 76)
        K_GROUP_ITER(0, 72)
        K_GROUP_ITER(4, 76)

        "add x4, x4, #512\n\t"

        // K-groups 20-21 (q_offset 80, 84)
        K_GROUP_ITER(0, 80)
        K_GROUP_ITER(4, 84)

        "add x4, x4, #512\n\t"

        // K-groups 22-23 (q_offset 88, 92)
        K_GROUP_ITER(0, 88)
        K_GROUP_ITER(4, 92)

        "add x4, x4, #512\n\t"

        // K-groups 24-25 (q_offset 96, 100)
        K_GROUP_ITER(0, 96)
        K_GROUP_ITER(4, 100)

        "add x4, x4, #512\n\t"

        // K-groups 26-27 (q_offset 104, 108)
        K_GROUP_ITER(0, 104)
        K_GROUP_ITER(4, 108)

        "add x4, x4, #512\n\t"

        // K-groups 28-29 (q_offset 112, 116)
        K_GROUP_ITER(0, 112)
        K_GROUP_ITER(4, 116)

        "add x4, x4, #512\n\t"

        // K-groups 30-31 (q_offset 120, 124)
        K_GROUP_ITER(0, 120)
        K_GROUP_ITER(4, 124)

        "add x4, x4, #512\n\t"

        // K-groups 32-33 (q_offset 128, 132)
        K_GROUP_ITER(0, 128)
        K_GROUP_ITER(4, 132)

        "add x4, x4, #512\n\t"

        // K-groups 34-35 (q_offset 136, 140)
        K_GROUP_ITER(0, 136)
        K_GROUP_ITER(4, 140)

        "add x4, x4, #512\n\t"

        // K-groups 36-37 (q_offset 144, 148)
        K_GROUP_ITER(0, 144)
        K_GROUP_ITER(4, 148)

        "add x4, x4, #512\n\t"

        // K-groups 38-39 (q_offset 152, 156)
        K_GROUP_ITER(0, 152)
        K_GROUP_ITER(4, 156)

        "add x4, x4, #512\n\t"

        // K-groups 40-41 (q_offset 160, 164)
        K_GROUP_ITER(0, 160)
        K_GROUP_ITER(4, 164)

        "add x4, x4, #512\n\t"

        // K-groups 42-43 (q_offset 168, 172)
        K_GROUP_ITER(0, 168)
        K_GROUP_ITER(4, 172)

        "add x4, x4, #512\n\t"

        // K-groups 44-45 (q_offset 176, 180)
        K_GROUP_ITER(0, 176)
        K_GROUP_ITER(4, 180)

        "add x4, x4, #512\n\t"

        // K-groups 46-47 (q_offset 184, 188)
        K_GROUP_ITER(0, 184)
        K_GROUP_ITER(4, 188)

        "add x4, x4, #512\n\t"

        // K-groups 48-49 (q_offset 192, 196)
        K_GROUP_ITER(0, 192)
        K_GROUP_ITER(4, 196)

        "add x4, x4, #512\n\t"

        // K-groups 50-51 (q_offset 200, 204)
        K_GROUP_ITER(0, 200)
        K_GROUP_ITER(4, 204)

        "add x4, x4, #512\n\t"

        // K-groups 52-53 (q_offset 208, 212)
        K_GROUP_ITER(0, 208)
        K_GROUP_ITER(4, 212)

        "add x4, x4, #512\n\t"

        // K-groups 54-55 (q_offset 216, 220)
        K_GROUP_ITER(0, 216)
        K_GROUP_ITER(4, 220)

        "add x4, x4, #512\n\t"

        // K-groups 56-57 (q_offset 224, 228)
        K_GROUP_ITER(0, 224)
        K_GROUP_ITER(4, 228)

        "add x4, x4, #512\n\t"

        // K-groups 58-59 (q_offset 232, 236)
        K_GROUP_ITER(0, 232)
        K_GROUP_ITER(4, 236)

        "add x4, x4, #512\n\t"

        // K-groups 60-61 (q_offset 240, 244)
        K_GROUP_ITER(0, 240)
        K_GROUP_ITER(4, 244)

        "add x4, x4, #512\n\t"

        // K-groups 62-63 (q_offset 248, 252) - max offset is 252!
        K_GROUP_ITER(0, 248)
        K_GROUP_ITER(4, 252)

        // Reduce 4 K-tiles to 1
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
        : "memory", "x4", "x5", "x6", "x7", "x11", "x12", "x13", "x14",
          "z0", "z1", "z2", "z3", "z4", "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15",
          "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23",
          "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31", "p0", "p1"
    );
}

// Baseline with loop for comparison
__attribute__((noinline))
void baseline_with_loop(const int8_t* Q, const int8_t* K, int32_t* S) {
    __asm__ volatile(
        "ptrue p0.b\n\t"
        "ptrue p1.s\n\t"
        "mov z8.s, #0\n\t"  "mov z9.s, #0\n\t"  "mov z10.s, #0\n\t" "mov z11.s, #0\n\t"
        "mov z12.s, #0\n\t" "mov z13.s, #0\n\t" "mov z14.s, #0\n\t" "mov z15.s, #0\n\t"
        "mov z16.s, #0\n\t" "mov z17.s, #0\n\t" "mov z18.s, #0\n\t" "mov z19.s, #0\n\t"
        "mov z20.s, #0\n\t" "mov z21.s, #0\n\t" "mov z22.s, #0\n\t" "mov z23.s, #0\n\t"
        "mov z24.s, #0\n\t" "mov z25.s, #0\n\t" "mov z26.s, #0\n\t" "mov z27.s, #0\n\t"
        "mov z28.s, #0\n\t" "mov z29.s, #0\n\t" "mov z30.s, #0\n\t" "mov z31.s, #0\n\t"

        "mov x5, %[Q]\n\t"
        "add x6, x5, #256\n\t"  "add x7, x6, #256\n\t"
        "add x11, x7, #256\n\t" "add x12, x11, #256\n\t" "add x13, x12, #256\n\t"
        "mov x4, %[K]\n\t"
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

int main() {
    int8_t* Q = aligned_alloc(256, 6 * 256);
    int8_t* K = aligned_alloc(256, 64 * 256);
    int32_t* S = aligned_alloc(256, 6 * 64 * sizeof(int32_t));

    for (int i = 0; i < 6 * 256; i++) Q[i] = (i % 7) - 3;
    for (int i = 0; i < 64 * 256; i++) K[i] = (i % 5) - 2;

    printf("==============================================\n");
    printf("Full K-Loop Unrolling Benchmark\n");
    printf("==============================================\n\n");
    printf("Configuration: MR=6, N=64, D=256\n");
    printf("Total SDOT: 1536 (6 rows × 64 K-groups × 4 K-tiles)\n");
    printf("Peak: 40 SDOT/tick\n\n");

    int warmup = 100;
    int iters = 1000;

    // Warmup
    for (int i = 0; i < warmup; i++) {
        baseline_with_loop(Q, K, S);
    }

    printf("Approach                            Ticks Efficiency    SDOT/tick\n");
    printf("------------------------------------------------------------------\n");

    // Baseline
    uint64_t start = rdcycle();
    for (int i = 0; i < iters; i++) {
        baseline_with_loop(Q, K, S);
    }
    uint64_t end = rdcycle();
    double ticks_baseline = (double)(end - start) / iters;
    double sdot_per_tick = 1536.0 / ticks_baseline;
    double efficiency = sdot_per_tick / 40.0 * 100.0;
    printf("Baseline (with loop)                %6.1f    %5.1f%%       %6.2f\n",
           ticks_baseline, efficiency, sdot_per_tick);

    // Warmup fully unrolled
    for (int i = 0; i < warmup; i++) {
        fully_unrolled_kernel(Q, K, S);
    }

    // Fully unrolled
    start = rdcycle();
    for (int i = 0; i < iters; i++) {
        fully_unrolled_kernel(Q, K, S);
    }
    end = rdcycle();
    double ticks_unrolled = (double)(end - start) / iters;
    sdot_per_tick = 1536.0 / ticks_unrolled;
    efficiency = sdot_per_tick / 40.0 * 100.0;
    printf("Fully unrolled (no loop)            %6.1f    %5.1f%%       %6.2f\n",
           ticks_unrolled, efficiency, sdot_per_tick);

    printf("\n");

    if (ticks_unrolled < ticks_baseline) {
        printf("Fully unrolled is %.1f%% faster than baseline.\n",
               (ticks_baseline - ticks_unrolled) / ticks_baseline * 100.0);
    } else {
        printf("Fully unrolled is %.1f%% slower than baseline.\n",
               (ticks_unrolled - ticks_baseline) / ticks_baseline * 100.0);
    }

    printf("\nAnalysis:\n");
    printf("  Loop overhead saved: 32 iterations × (1 subs + 1 branch + 7 adds) = ~288 instructions\n");
    printf("  But fully unrolled adds: 31 K-pointer adds = 31 instructions\n");
    printf("  Net savings: ~257 instructions\n");

    free(Q);
    free(K);
    free(S);
    return 0;
}
