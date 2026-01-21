/*
 * Isolate Phase 1 vs Phase 2 timing for D=256
 *
 * Total 3072 SDOTs = 1536 (Phase 1) + 1536 (Phase 2)
 * If both equally efficient: 50% time each
 * If Phase 1 bottleneck: >50% time
 * If Phase 2 bottleneck: >50% time
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

// Phase 1 only: Q@K^T
__attribute__((noinline))
void phase1_only(const int8_t* Q, const int8_t* K, int32_t* S) {
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

        // Unroll 1
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

// Phase 2 only: S@V (1 D-tile)
__attribute__((noinline))
void phase2_only(const int32_t* S, const int8_t* V, int32_t* O) {
    __asm__ volatile(
        "ptrue p0.b\n\t"
        "ptrue p1.s\n\t"
        "mov x15, #4\n\t"  // 4 D-tiles
        "mov x16, %[V]\n\t"
        "mov x17, %[O]\n\t"
        "2:\n\t"
        "mov z8.s, #0\n\t"  "mov z9.s, #0\n\t"  "mov z10.s, #0\n\t" "mov z11.s, #0\n\t"
        "mov z12.s, #0\n\t" "mov z13.s, #0\n\t" "mov z14.s, #0\n\t" "mov z15.s, #0\n\t"
        "mov z16.s, #0\n\t" "mov z17.s, #0\n\t" "mov z18.s, #0\n\t" "mov z19.s, #0\n\t"
        "mov z20.s, #0\n\t" "mov z21.s, #0\n\t" "mov z22.s, #0\n\t" "mov z23.s, #0\n\t"
        "mov z24.s, #0\n\t" "mov z25.s, #0\n\t" "mov z26.s, #0\n\t" "mov z27.s, #0\n\t"
        "mov z28.s, #0\n\t" "mov z29.s, #0\n\t" "mov z30.s, #0\n\t" "mov z31.s, #0\n\t"
        "mov x5, %[S]\n\t"
        "add x6, x5, #256\n\t"  "add x7, x5, #512\n\t"
        "add x11, x5, #768\n\t" "add x12, x5, #1024\n\t" "add x13, x5, #1280\n\t"
        "mov x4, x16\n\t"
        "mov x10, #8\n\t"  // 16 N_groups / 2 = 8 (2x unroll)

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

        "add x4, x4, #1024\n\t"  // V stride = D * 4 = 1024
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

// Full fused kernel (both phases)
__attribute__((noinline))
void fused_full(const int8_t* Q, const int8_t* K, const int8_t* V, int32_t* O) {
    int32_t S[6 * 64] __attribute__((aligned(64)));
    phase1_only(Q, K, S);
    phase2_only(S, V, O);
}

int main() {
    printf("==============================================\n");
    printf("Phase 1 vs Phase 2 Isolation D=256\n");
    printf("==============================================\n\n");

    int D = 256;
    int N = 64;
    int D_groups = D / 4;

    int8_t* Q = aligned_alloc(256, 6 * D);
    int8_t* K = aligned_alloc(256, D_groups * N * 4);
    int8_t* V = aligned_alloc(256, N * D * 4);
    int32_t* O = aligned_alloc(256, 6 * D * 4);
    int32_t* S = aligned_alloc(256, 6 * N * 4);

    memset(Q, 1, 6 * D);
    memset(K, 1, D_groups * N * 4);
    memset(V, 1, N * D * 4);

    int iters = 5000;
    int p1_sdot = 1536;  // Phase 1: 6 * 4 * 64 = 1536
    int p2_sdot = 1536;  // Phase 2: 6 * 4 * 64 = 1536
    int total_sdot = 3072;

    printf("SDOTs: Phase1=%d, Phase2=%d, Total=%d\n\n", p1_sdot, p2_sdot, total_sdot);
    printf("%-15s %10s %12s %8s %10s\n", "Phase", "Ticks", "SDOT/tick", "Eff", "% Total");
    printf("----------------------------------------------------------------\n");

    // Warmup
    for (int i = 0; i < 100; i++) {
        phase1_only(Q, K, S);
        phase2_only(S, V, O);
        fused_full(Q, K, V, O);
    }

    // Phase 1 only
    uint64_t start = rdcycle();
    for (int i = 0; i < iters; i++) {
        phase1_only(Q, K, S);
    }
    uint64_t end = rdcycle();
    double ticks_p1 = (double)(end - start) / iters;
    double eff_p1 = (p1_sdot / ticks_p1) / 40 * 100;
    printf("%-15s %10.1f %12.2f %7.1f%%\n",
           "Phase 1 (Q@K)", ticks_p1, p1_sdot/ticks_p1, eff_p1);

    // Phase 2 only
    start = rdcycle();
    for (int i = 0; i < iters; i++) {
        phase2_only(S, V, O);
    }
    end = rdcycle();
    double ticks_p2 = (double)(end - start) / iters;
    double eff_p2 = (p2_sdot / ticks_p2) / 40 * 100;
    printf("%-15s %10.1f %12.2f %7.1f%%\n",
           "Phase 2 (S@V)", ticks_p2, p2_sdot/ticks_p2, eff_p2);

    // Full fused
    start = rdcycle();
    for (int i = 0; i < iters; i++) {
        fused_full(Q, K, V, O);
    }
    end = rdcycle();
    double ticks_full = (double)(end - start) / iters;
    double eff_full = (total_sdot / ticks_full) / 40 * 100;
    printf("%-15s %10.1f %12.2f %7.1f%%\n",
           "Full Fused", ticks_full, total_sdot/ticks_full, eff_full);

    printf("\n");
    printf("Phase breakdown:\n");
    printf("  Phase 1: %.1f ticks (%.1f%%)\n", ticks_p1, ticks_p1/(ticks_p1+ticks_p2)*100);
    printf("  Phase 2: %.1f ticks (%.1f%%)\n", ticks_p2, ticks_p2/(ticks_p1+ticks_p2)*100);
    printf("  Sum:     %.1f ticks\n", ticks_p1 + ticks_p2);
    printf("  Fused:   %.1f ticks (overhead: %.1f ticks = %.1f%%)\n",
           ticks_full, ticks_full - (ticks_p1 + ticks_p2),
           (ticks_full - (ticks_p1 + ticks_p2)) / ticks_full * 100);

    free(Q); free(K); free(V); free(O); free(S);
    return 0;
}
