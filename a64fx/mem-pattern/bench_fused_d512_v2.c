/*
 * Fused GEMM D=512 using separate Phase 1 and Phase 2 functions
 * This avoids the overhead of the single-function implementation
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

// Phase 1: Q[6,D] @ K^T[D,64] -> S[6,64]
__attribute__((noinline))
void phase1_mr6(
    const int8_t* Q,
    const int8_t* K,
    int32_t* S,
    int D_groups
) {
    __asm__ volatile(
        "ptrue p0.b\n\t"
        "ptrue p1.s\n\t"

        // Zero 24 accumulators
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

        // Q row stride = D bytes = D_groups * 4
        "mov x5, %[Q]\n\t"
        "mov x14, %[D_groups]\n\t"
        "lsl x14, x14, #2\n\t"
        "add x6, x5, x14\n\t"
        "add x7, x6, x14\n\t"
        "add x11, x7, x14\n\t"
        "add x12, x11, x14\n\t"
        "add x13, x12, x14\n\t"

        "mov x4, %[K]\n\t"
        "mov x10, %[D_groups]\n\t"
        "lsr x10, x10, #1\n\t"

        "1:\n\t"
        "ld1b {z0.b}, p0/z, [x4]\n\t"
        "ld1b {z1.b}, p0/z, [x4, #1, mul vl]\n\t"
        "ld1b {z2.b}, p0/z, [x4, #2, mul vl]\n\t"
        "ld1b {z3.b}, p0/z, [x4, #3, mul vl]\n\t"

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

        "ld1b {z0.b}, p0/z, [x4, #4, mul vl]\n\t"
        "ld1b {z1.b}, p0/z, [x4, #5, mul vl]\n\t"
        "ld1b {z2.b}, p0/z, [x4, #6, mul vl]\n\t"
        "ld1b {z3.b}, p0/z, [x4, #7, mul vl]\n\t"
        "add x4, x4, #512\n\t"

        "ld1rw {z4.s}, p1/z, [x5, #4]\n\t"
        "sdot z8.s, z0.b, z4.b\n\t"
        "sdot z9.s, z1.b, z4.b\n\t"
        "sdot z10.s, z2.b, z4.b\n\t"
        "sdot z11.s, z3.b, z4.b\n\t"

        "ld1rw {z4.s}, p1/z, [x6, #4]\n\t"
        "sdot z12.s, z0.b, z4.b\n\t"
        "sdot z13.s, z1.b, z4.b\n\t"
        "sdot z14.s, z2.b, z4.b\n\t"
        "sdot z15.s, z3.b, z4.b\n\t"

        "ld1rw {z4.s}, p1/z, [x7, #4]\n\t"
        "sdot z16.s, z0.b, z4.b\n\t"
        "sdot z17.s, z1.b, z4.b\n\t"
        "sdot z18.s, z2.b, z4.b\n\t"
        "sdot z19.s, z3.b, z4.b\n\t"

        "ld1rw {z4.s}, p1/z, [x11, #4]\n\t"
        "sdot z20.s, z0.b, z4.b\n\t"
        "sdot z21.s, z1.b, z4.b\n\t"
        "sdot z22.s, z2.b, z4.b\n\t"
        "sdot z23.s, z3.b, z4.b\n\t"

        "ld1rw {z4.s}, p1/z, [x12, #4]\n\t"
        "sdot z24.s, z0.b, z4.b\n\t"
        "sdot z25.s, z1.b, z4.b\n\t"
        "sdot z26.s, z2.b, z4.b\n\t"
        "sdot z27.s, z3.b, z4.b\n\t"

        "ld1rw {z4.s}, p1/z, [x13, #4]\n\t"
        "sdot z28.s, z0.b, z4.b\n\t"
        "sdot z29.s, z1.b, z4.b\n\t"
        "sdot z30.s, z2.b, z4.b\n\t"
        "sdot z31.s, z3.b, z4.b\n\t"

        "add x5, x5, #8\n\t"
        "add x6, x6, #8\n\t"
        "add x7, x7, #8\n\t"
        "add x11, x11, #8\n\t"
        "add x12, x12, #8\n\t"
        "add x13, x13, #8\n\t"

        "subs x10, x10, #1\n\t"
        "b.ne 1b\n\t"

        // Merge accumulators
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

        // Store S
        "mov x14, %[S]\n\t"
        "st1w {z8.s}, p1, [x14]\n\t"
        "st1w {z12.s}, p1, [x14, #1, mul vl]\n\t"
        "st1w {z16.s}, p1, [x14, #2, mul vl]\n\t"
        "st1w {z20.s}, p1, [x14, #3, mul vl]\n\t"
        "st1w {z24.s}, p1, [x14, #4, mul vl]\n\t"
        "st1w {z28.s}, p1, [x14, #5, mul vl]\n\t"

        :
        : [Q] "r"(Q), [K] "r"(K), [S] "r"(S), [D_groups] "r"((long)D_groups)
        : "memory", "x4", "x5", "x6", "x7", "x10", "x11", "x12", "x13", "x14",
          "z0", "z1", "z2", "z3", "z4",
          "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15",
          "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23",
          "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31",
          "p0", "p1"
    );
}

// Phase 2: S[6,64] @ V[64,D] -> O[6,D]
__attribute__((noinline))
void phase2_mr6(
    const int32_t* S,
    const int8_t* V,
    int32_t* O,
    int D,
    int V_stride  // D * 4
) {
    int D_tiles = D / 64;

    __asm__ volatile(
        "ptrue p0.b\n\t"
        "ptrue p1.s\n\t"

        "mov x15, %[D_tiles]\n\t"
        "mov x16, %[V]\n\t"
        "mov x17, %[O]\n\t"
        "mov x18, %[V_stride]\n\t"
        "mov x19, %[O_stride]\n\t"

        "2:\n\t"
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

        "mov x5, %[S]\n\t"
        "add x6, x5, #256\n\t"
        "add x7, x5, #512\n\t"
        "add x11, x5, #768\n\t"
        "add x12, x5, #1024\n\t"
        "add x13, x5, #1280\n\t"

        "mov x4, x16\n\t"
        "mov x10, #8\n\t"

        "3:\n\t"
        "ld1b {z0.b}, p0/z, [x4]\n\t"
        "ld1b {z1.b}, p0/z, [x4, #1, mul vl]\n\t"
        "ld1b {z2.b}, p0/z, [x4, #2, mul vl]\n\t"
        "ld1b {z3.b}, p0/z, [x4, #3, mul vl]\n\t"

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

        "add x4, x4, x18\n\t"  // V stride
        "ld1b {z0.b}, p0/z, [x4]\n\t"
        "ld1b {z1.b}, p0/z, [x4, #1, mul vl]\n\t"
        "ld1b {z2.b}, p0/z, [x4, #2, mul vl]\n\t"
        "ld1b {z3.b}, p0/z, [x4, #3, mul vl]\n\t"
        "add x4, x4, x18\n\t"

        "ld1rw {z4.s}, p1/z, [x5, #4]\n\t"
        "sdot z8.s, z0.b, z4.b\n\t"
        "sdot z9.s, z1.b, z4.b\n\t"
        "sdot z10.s, z2.b, z4.b\n\t"
        "sdot z11.s, z3.b, z4.b\n\t"

        "ld1rw {z4.s}, p1/z, [x6, #4]\n\t"
        "sdot z12.s, z0.b, z4.b\n\t"
        "sdot z13.s, z1.b, z4.b\n\t"
        "sdot z14.s, z2.b, z4.b\n\t"
        "sdot z15.s, z3.b, z4.b\n\t"

        "ld1rw {z4.s}, p1/z, [x7, #4]\n\t"
        "sdot z16.s, z0.b, z4.b\n\t"
        "sdot z17.s, z1.b, z4.b\n\t"
        "sdot z18.s, z2.b, z4.b\n\t"
        "sdot z19.s, z3.b, z4.b\n\t"

        "ld1rw {z4.s}, p1/z, [x11, #4]\n\t"
        "sdot z20.s, z0.b, z4.b\n\t"
        "sdot z21.s, z1.b, z4.b\n\t"
        "sdot z22.s, z2.b, z4.b\n\t"
        "sdot z23.s, z3.b, z4.b\n\t"

        "ld1rw {z4.s}, p1/z, [x12, #4]\n\t"
        "sdot z24.s, z0.b, z4.b\n\t"
        "sdot z25.s, z1.b, z4.b\n\t"
        "sdot z26.s, z2.b, z4.b\n\t"
        "sdot z27.s, z3.b, z4.b\n\t"

        "ld1rw {z4.s}, p1/z, [x13, #4]\n\t"
        "sdot z28.s, z0.b, z4.b\n\t"
        "sdot z29.s, z1.b, z4.b\n\t"
        "sdot z30.s, z2.b, z4.b\n\t"
        "sdot z31.s, z3.b, z4.b\n\t"

        "add x5, x5, #8\n\t"
        "add x6, x6, #8\n\t"
        "add x7, x7, #8\n\t"
        "add x11, x11, #8\n\t"
        "add x12, x12, #8\n\t"
        "add x13, x13, #8\n\t"

        "subs x10, x10, #1\n\t"
        "b.ne 3b\n\t"

        // Merge
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

        // Store O
        "st1w {z8.s}, p1, [x17]\n\t"
        "add x17, x17, x19\n\t"
        "st1w {z12.s}, p1, [x17]\n\t"
        "add x17, x17, x19\n\t"
        "st1w {z16.s}, p1, [x17]\n\t"
        "add x17, x17, x19\n\t"
        "st1w {z20.s}, p1, [x17]\n\t"
        "add x17, x17, x19\n\t"
        "st1w {z24.s}, p1, [x17]\n\t"
        "add x17, x17, x19\n\t"
        "st1w {z28.s}, p1, [x17]\n\t"

        // Reset O ptr for next D-tile
        "mov x20, x19\n\t"
        "lsl x20, x20, #2\n\t"
        "add x20, x20, x19\n\t"
        "sub x17, x17, x20\n\t"
        "add x17, x17, #64\n\t"
        "add x16, x16, #64\n\t"

        "subs x15, x15, #1\n\t"
        "b.ne 2b\n\t"

        :
        : [S] "r"(S), [V] "r"(V), [O] "r"(O),
          [D_tiles] "r"((long)(D/64)), [V_stride] "r"((long)V_stride),
          [O_stride] "r"((long)(D*4))
        : "memory", "x4", "x5", "x6", "x7", "x10", "x11", "x12", "x13",
          "x15", "x16", "x17", "x18", "x19", "x20",
          "z0", "z1", "z2", "z3", "z4",
          "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15",
          "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23",
          "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31",
          "p0", "p1"
    );
}

int main() {
    printf("==============================================\n");
    printf("Fused GEMM D=512 V2 (Separate Functions)\n");
    printf("==============================================\n\n");

    int D = 512;
    int N = 64;
    int D_groups = D / 4;  // 128

    int8_t* Q = aligned_alloc(256, 6 * D);
    int8_t* K = aligned_alloc(256, D_groups * N * 4);
    int8_t* V = aligned_alloc(256, N * D * 4);
    int32_t* S = aligned_alloc(256, 6 * N * 4);
    int32_t* O = aligned_alloc(256, 6 * D * 4);

    memset(Q, 1, 6 * D);
    memset(K, 1, D_groups * N * 4);
    memset(V, 1, N * D * 4);

    int iters = 5000;
    int total_sdot = D_groups * 24 + (D / 64) * 16 * 24;  // 3072 + 3072 = 6144

    printf("Configuration:\n");
    printf("  Phase 1: Q[6,%d] @ K^T[%d,64] -> S[6,64] (%d SDOTs)\n",
           D, D, D_groups * 24);
    printf("  Phase 2: S[6,64] @ V[64,%d] -> O[6,%d] (%d SDOTs)\n",
           D, D, (D/64) * 16 * 24);
    printf("  Total: %d SDOTs\n\n", total_sdot);

    // Warmup
    for (int i = 0; i < 100; i++) {
        phase1_mr6(Q, K, S, D_groups);
        phase2_mr6(S, V, O, D, D * 4);
    }

    // Benchmark
    uint64_t start = rdcycle();
    for (int i = 0; i < iters; i++) {
        phase1_mr6(Q, K, S, D_groups);
        phase2_mr6(S, V, O, D, D * 4);
    }
    uint64_t end = rdcycle();

    double ticks = (double)(end - start) / iters;
    double sdot_tick = total_sdot / ticks;
    double eff = sdot_tick / 40.0 * 100;

    printf("Results:\n");
    printf("  Ticks: %.1f\n", ticks);
    printf("  SDOT/tick: %.2f\n", sdot_tick);
    printf("  Efficiency: %.1f%% (vs ~40 SDOT/tick peak)\n\n", eff);

    printf("Working set:\n");
    printf("  Q: %.1f KB, K: %.1f KB, V: %.1f KB, S: %.1f KB, O: %.1f KB\n",
           6.0*D/1024, D_groups*64.0*4/1024, N*D*4.0/1024, 6.0*N*4/1024, 6.0*D*4/1024);
    printf("  Total: %.1f KB (L1D = 64KB)\n",
           (6*D + D_groups*64*4 + N*D*4 + 6*N*4 + 6*D*4) / 1024.0);

    free(Q); free(K); free(V); free(S); free(O);
    return 0;
}
