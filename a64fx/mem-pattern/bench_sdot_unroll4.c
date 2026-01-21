// bench_sdot_unroll4.c
// 4x unrolled SDOT with no dependencies and deep pipelining

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

// 4x unrolled, interleaved loads+SDOT, no dependencies
void qkt_unroll4(
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

        // 16 accumulators (4 rows × 4 K-vectors)
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

        // Preload iteration 0
        "ld1b {z0.b}, p0/z, [x4]\n\t"
        "ld1b {z1.b}, p0/z, [x4, #1, mul vl]\n\t"
        "ld1rw {z4.s}, p1/z, [x0]\n\t"
        "ld1rw {z5.s}, p1/z, [x1]\n\t"

        "mov x5, #16\n\t"  // 16 x 4 = 64 iterations
        "1:\n\t"
        // === Iteration 0 ===
        "ld1b {z2.b}, p0/z, [x4, #2, mul vl]\n\t"
        "ld1rw {z6.s}, p1/z, [x2]\n\t"
        "sdot z16.s, z0.b, z4.b\n\t"
        "sdot z20.s, z0.b, z5.b\n\t"

        "ld1b {z3.b}, p0/z, [x4, #3, mul vl]\n\t"
        "ld1rw {z7.s}, p1/z, [x3]\n\t"
        "sdot z24.s, z0.b, z6.b\n\t"
        "sdot z28.s, z0.b, z7.b\n\t"

        "ld1rw {z8.s}, p1/z, [x0, #4]\n\t"
        "sdot z17.s, z1.b, z4.b\n\t"
        "sdot z21.s, z1.b, z5.b\n\t"
        "sdot z25.s, z1.b, z6.b\n\t"

        "ld1rw {z9.s}, p1/z, [x1, #4]\n\t"
        "sdot z29.s, z1.b, z7.b\n\t"
        "sdot z18.s, z2.b, z4.b\n\t"
        "sdot z22.s, z2.b, z5.b\n\t"

        "ld1b {z0.b}, p0/z, [x4, #4, mul vl]\n\t"
        "sdot z26.s, z2.b, z6.b\n\t"
        "sdot z30.s, z2.b, z7.b\n\t"
        "sdot z19.s, z3.b, z4.b\n\t"

        "ld1b {z1.b}, p0/z, [x4, #5, mul vl]\n\t"
        "sdot z23.s, z3.b, z5.b\n\t"
        "sdot z27.s, z3.b, z6.b\n\t"
        "sdot z31.s, z3.b, z7.b\n\t"

        // === Iteration 1 ===
        "ld1b {z2.b}, p0/z, [x4, #6, mul vl]\n\t"
        "ld1rw {z10.s}, p1/z, [x2, #4]\n\t"
        "sdot z16.s, z0.b, z8.b\n\t"
        "sdot z20.s, z0.b, z9.b\n\t"

        "ld1b {z3.b}, p0/z, [x4, #7, mul vl]\n\t"
        "ld1rw {z11.s}, p1/z, [x3, #4]\n\t"
        "sdot z24.s, z0.b, z10.b\n\t"
        "sdot z28.s, z0.b, z11.b\n\t"

        "add x4, x4, #512\n\t"  // K += 8 vectors (move here)
        "ld1rw {z4.s}, p1/z, [x0, #8]\n\t"
        "sdot z17.s, z1.b, z8.b\n\t"
        "sdot z21.s, z1.b, z9.b\n\t"
        "sdot z25.s, z1.b, z10.b\n\t"

        "ld1rw {z5.s}, p1/z, [x1, #8]\n\t"
        "sdot z29.s, z1.b, z11.b\n\t"
        "sdot z18.s, z2.b, z8.b\n\t"
        "sdot z22.s, z2.b, z9.b\n\t"

        "ld1b {z0.b}, p0/z, [x4]\n\t"          // Now offset 0
        "sdot z26.s, z2.b, z10.b\n\t"
        "sdot z30.s, z2.b, z11.b\n\t"
        "sdot z19.s, z3.b, z8.b\n\t"

        "ld1b {z1.b}, p0/z, [x4, #1, mul vl]\n\t"  // Now offset 1
        "sdot z23.s, z3.b, z9.b\n\t"
        "sdot z27.s, z3.b, z10.b\n\t"
        "sdot z31.s, z3.b, z11.b\n\t"

        // === Iteration 2 ===
        "ld1b {z2.b}, p0/z, [x4, #2, mul vl]\n\t"
        "ld1rw {z6.s}, p1/z, [x2, #8]\n\t"
        "sdot z16.s, z0.b, z4.b\n\t"
        "sdot z20.s, z0.b, z5.b\n\t"

        "ld1b {z3.b}, p0/z, [x4, #3, mul vl]\n\t"
        "ld1rw {z7.s}, p1/z, [x3, #8]\n\t"
        "sdot z24.s, z0.b, z6.b\n\t"
        "sdot z28.s, z0.b, z7.b\n\t"

        "ld1rw {z8.s}, p1/z, [x0, #12]\n\t"
        "sdot z17.s, z1.b, z4.b\n\t"
        "sdot z21.s, z1.b, z5.b\n\t"
        "sdot z25.s, z1.b, z6.b\n\t"

        "ld1rw {z9.s}, p1/z, [x1, #12]\n\t"
        "sdot z29.s, z1.b, z7.b\n\t"
        "sdot z18.s, z2.b, z4.b\n\t"
        "sdot z22.s, z2.b, z5.b\n\t"

        "ld1b {z0.b}, p0/z, [x4, #4, mul vl]\n\t"
        "sdot z26.s, z2.b, z6.b\n\t"
        "sdot z30.s, z2.b, z7.b\n\t"
        "sdot z19.s, z3.b, z4.b\n\t"

        "ld1b {z1.b}, p0/z, [x4, #5, mul vl]\n\t"
        "sdot z23.s, z3.b, z5.b\n\t"
        "sdot z27.s, z3.b, z6.b\n\t"
        "sdot z31.s, z3.b, z7.b\n\t"

        // === Iteration 3 ===
        "ld1b {z2.b}, p0/z, [x4, #6, mul vl]\n\t"
        "ld1rw {z10.s}, p1/z, [x2, #12]\n\t"
        "sdot z16.s, z0.b, z8.b\n\t"
        "sdot z20.s, z0.b, z9.b\n\t"

        "ld1b {z3.b}, p0/z, [x4, #7, mul vl]\n\t"
        "ld1rw {z11.s}, p1/z, [x3, #12]\n\t"
        "sdot z24.s, z0.b, z10.b\n\t"
        "sdot z28.s, z0.b, z11.b\n\t"

        "add x4, x4, #512\n\t"  // K += 8 vectors
        "sdot z17.s, z1.b, z8.b\n\t"
        "sdot z21.s, z1.b, z9.b\n\t"
        "sdot z25.s, z1.b, z10.b\n\t"

        "add x0, x0, #16\n\t"
        "sdot z29.s, z1.b, z11.b\n\t"
        "sdot z18.s, z2.b, z8.b\n\t"
        "sdot z22.s, z2.b, z9.b\n\t"

        "ld1b {z0.b}, p0/z, [x4]\n\t"
        "add x1, x1, #16\n\t"
        "sdot z26.s, z2.b, z10.b\n\t"
        "sdot z30.s, z2.b, z11.b\n\t"
        "sdot z19.s, z3.b, z8.b\n\t"

        "ld1b {z1.b}, p0/z, [x4, #1, mul vl]\n\t"
        "add x2, x2, #16\n\t"
        "sdot z23.s, z3.b, z9.b\n\t"
        "sdot z27.s, z3.b, z10.b\n\t"
        "sdot z31.s, z3.b, z11.b\n\t"

        "ld1rw {z4.s}, p1/z, [x0]\n\t"
        "ld1rw {z5.s}, p1/z, [x1]\n\t"
        "add x3, x3, #16\n\t"
        "subs x5, x5, #1\n\t"
        "b.ne 1b\n\t"

        // Merge accumulators
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
          "z8", "z9", "z10", "z11",
          "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23",
          "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31",
          "p0", "p1", "memory"
    );
}

int main() {
    printf("==============================================\n");
    printf("4x Unrolled SDOT - No Dependencies\n");
    printf("==============================================\n\n");

    int8_t* Q = (int8_t*)aligned_alloc(256, 1024);
    int8_t* K = (int8_t*)aligned_alloc(256, 16384);
    int32_t* S = (int32_t*)aligned_alloc(256, 1024);

    memset(Q, 1, 1024);
    memset(K, 2, 16384);
    memset(S, 0, 1024);

    int iters = 10000;
    int warmup = 1000;
    uint64_t start, end;
    double ticks, cycles;

    printf("Q@K^T: Load 17KB, 1024 SDOT\n");
    printf("Theoretical: 512 cy (compute @ 2 SDOT/cy) or 136 cy (load @ 128 B/cy)\n\n");

    printf("%-25s %8s %10s %10s\n", "Kernel", "Cycles", "Ld B/cy", "SDOT/cy");
    printf("----------------------------------------------------------\n");

    for (int i = 0; i < warmup; i++) qkt_unroll4(Q, K, S);
    start = rdtsc();
    for (int i = 0; i < iters; i++) qkt_unroll4(Q, K, S);
    end = rdtsc();
    ticks = (double)(end - start) / iters;
    cycles = ticks * 20.0;
    printf("%-25s %8.1f %10.1f %10.1f\n", "unroll4", cycles, 17408.0/cycles, 1024.0/cycles);

    printf("\nPer-iteration stats (4 d_groups per iteration):\n");
    printf("  Loads: 16 ld1b + 16 ld1rw = %d bytes\n", 16*64 + 16*4);
    printf("  Compute: 64 SDOT\n");
    printf("  Theoretical: %.1f cy (64 SDOT / 2)\n", 64.0/2.0);
    printf("  Actual: %.1f cy\n", cycles / 16.0);

    free(Q); free(K); free(S);
    return 0;
}
