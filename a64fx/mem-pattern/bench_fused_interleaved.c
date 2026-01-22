/*
 * Fused Attention with Proper Interleaved Q/SDOT Pattern
 *
 * Key insight from successful 90.5% kernel:
 * - Load 4 K vectors
 * - For each row: load Q broadcast, then 4 SDOT (pipeline overlap)
 * - Don't batch all Q loads before SDOT
 */

#include <arm_sve.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static inline uint64_t rdtick(void) {
    uint64_t val;
    __asm__ volatile("mrs %0, cntvct_el0" : "=r"(val));
    return val;
}

// Correct interleaved K iteration: load K, then interleave Q loads with SDOT
// MR=6: 24 accumulators (z8-z31), K in z0-z3, Q broadcast in z4
#define K_ITER_INTERLEAVED(k_off, q_off) \
    "ld1b {z0.b}, p0/z, [x4, #" #k_off ", mul vl]\n\t" \
    "ld1b {z1.b}, p0/z, [x4, #" #k_off "+1, mul vl]\n\t" \
    "ld1b {z2.b}, p0/z, [x4, #" #k_off "+2, mul vl]\n\t" \
    "ld1b {z3.b}, p0/z, [x4, #" #k_off "+3, mul vl]\n\t" \
    /* Row 0 */ \
    "ld1rw {z4.s}, p1/z, [x5, #" #q_off "]\n\t" \
    "sdot z8.s, z0.b, z4.b\n\t"  "sdot z9.s, z1.b, z4.b\n\t" \
    "sdot z10.s, z2.b, z4.b\n\t" "sdot z11.s, z3.b, z4.b\n\t" \
    /* Row 1 */ \
    "ld1rw {z4.s}, p1/z, [x6, #" #q_off "]\n\t" \
    "sdot z12.s, z0.b, z4.b\n\t" "sdot z13.s, z1.b, z4.b\n\t" \
    "sdot z14.s, z2.b, z4.b\n\t" "sdot z15.s, z3.b, z4.b\n\t" \
    /* Row 2 */ \
    "ld1rw {z4.s}, p1/z, [x7, #" #q_off "]\n\t" \
    "sdot z16.s, z0.b, z4.b\n\t" "sdot z17.s, z1.b, z4.b\n\t" \
    "sdot z18.s, z2.b, z4.b\n\t" "sdot z19.s, z3.b, z4.b\n\t" \
    /* Row 3 */ \
    "ld1rw {z4.s}, p1/z, [x11, #" #q_off "]\n\t" \
    "sdot z20.s, z0.b, z4.b\n\t" "sdot z21.s, z1.b, z4.b\n\t" \
    "sdot z22.s, z2.b, z4.b\n\t" "sdot z23.s, z3.b, z4.b\n\t" \
    /* Row 4 */ \
    "ld1rw {z4.s}, p1/z, [x12, #" #q_off "]\n\t" \
    "sdot z24.s, z0.b, z4.b\n\t" "sdot z25.s, z1.b, z4.b\n\t" \
    "sdot z26.s, z2.b, z4.b\n\t" "sdot z27.s, z3.b, z4.b\n\t" \
    /* Row 5 */ \
    "ld1rw {z4.s}, p1/z, [x13, #" #q_off "]\n\t" \
    "sdot z28.s, z0.b, z4.b\n\t" "sdot z29.s, z1.b, z4.b\n\t" \
    "sdot z30.s, z2.b, z4.b\n\t" "sdot z31.s, z3.b, z4.b\n\t"

// Phase 1: Q[6,256] @ K[256,64] -> S[6,64]
// 64 K iterations, 24 SDOT each = 1536 SDOT
__attribute__((noinline))
void phase1_interleaved_6row(const int8_t* Q, const int8_t* K, int32_t* S) {
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

        // Q row pointers (x5=row0, x6=row1, x7=row2, x11=row3, x12=row4, x13=row5)
        "mov x5, %[Q]\n\t"
        "add x6, x5, #256\n\t"
        "add x7, x6, #256\n\t"
        "add x11, x7, #256\n\t"
        "add x12, x11, #256\n\t"
        "add x13, x12, #256\n\t"
        "mov x4, %[K]\n\t"

        // === FULLY UNROLLED K-LOOP (64 iterations in 32 pairs) ===
        // K-groups 0-1 (q_offset 0, 4)
        K_ITER_INTERLEAVED(0, 0)
        K_ITER_INTERLEAVED(4, 4)
        "add x4, x4, #512\n\t"

        // K-groups 2-3 (q_offset 8, 12)
        K_ITER_INTERLEAVED(0, 8)
        K_ITER_INTERLEAVED(4, 12)
        "add x4, x4, #512\n\t"

        // K-groups 4-5
        K_ITER_INTERLEAVED(0, 16)
        K_ITER_INTERLEAVED(4, 20)
        "add x4, x4, #512\n\t"

        // K-groups 6-7
        K_ITER_INTERLEAVED(0, 24)
        K_ITER_INTERLEAVED(4, 28)
        "add x4, x4, #512\n\t"

        // K-groups 8-9
        K_ITER_INTERLEAVED(0, 32)
        K_ITER_INTERLEAVED(4, 36)
        "add x4, x4, #512\n\t"

        // K-groups 10-11
        K_ITER_INTERLEAVED(0, 40)
        K_ITER_INTERLEAVED(4, 44)
        "add x4, x4, #512\n\t"

        // K-groups 12-13
        K_ITER_INTERLEAVED(0, 48)
        K_ITER_INTERLEAVED(4, 52)
        "add x4, x4, #512\n\t"

        // K-groups 14-15
        K_ITER_INTERLEAVED(0, 56)
        K_ITER_INTERLEAVED(4, 60)
        "add x4, x4, #512\n\t"

        // K-groups 16-17
        K_ITER_INTERLEAVED(0, 64)
        K_ITER_INTERLEAVED(4, 68)
        "add x4, x4, #512\n\t"

        // K-groups 18-19
        K_ITER_INTERLEAVED(0, 72)
        K_ITER_INTERLEAVED(4, 76)
        "add x4, x4, #512\n\t"

        // K-groups 20-21
        K_ITER_INTERLEAVED(0, 80)
        K_ITER_INTERLEAVED(4, 84)
        "add x4, x4, #512\n\t"

        // K-groups 22-23
        K_ITER_INTERLEAVED(0, 88)
        K_ITER_INTERLEAVED(4, 92)
        "add x4, x4, #512\n\t"

        // K-groups 24-25
        K_ITER_INTERLEAVED(0, 96)
        K_ITER_INTERLEAVED(4, 100)
        "add x4, x4, #512\n\t"

        // K-groups 26-27
        K_ITER_INTERLEAVED(0, 104)
        K_ITER_INTERLEAVED(4, 108)
        "add x4, x4, #512\n\t"

        // K-groups 28-29
        K_ITER_INTERLEAVED(0, 112)
        K_ITER_INTERLEAVED(4, 116)
        "add x4, x4, #512\n\t"

        // K-groups 30-31
        K_ITER_INTERLEAVED(0, 120)
        K_ITER_INTERLEAVED(4, 124)
        "add x4, x4, #512\n\t"

        // K-groups 32-33
        K_ITER_INTERLEAVED(0, 128)
        K_ITER_INTERLEAVED(4, 132)
        "add x4, x4, #512\n\t"

        // K-groups 34-35
        K_ITER_INTERLEAVED(0, 136)
        K_ITER_INTERLEAVED(4, 140)
        "add x4, x4, #512\n\t"

        // K-groups 36-37
        K_ITER_INTERLEAVED(0, 144)
        K_ITER_INTERLEAVED(4, 148)
        "add x4, x4, #512\n\t"

        // K-groups 38-39
        K_ITER_INTERLEAVED(0, 152)
        K_ITER_INTERLEAVED(4, 156)
        "add x4, x4, #512\n\t"

        // K-groups 40-41
        K_ITER_INTERLEAVED(0, 160)
        K_ITER_INTERLEAVED(4, 164)
        "add x4, x4, #512\n\t"

        // K-groups 42-43
        K_ITER_INTERLEAVED(0, 168)
        K_ITER_INTERLEAVED(4, 172)
        "add x4, x4, #512\n\t"

        // K-groups 44-45
        K_ITER_INTERLEAVED(0, 176)
        K_ITER_INTERLEAVED(4, 180)
        "add x4, x4, #512\n\t"

        // K-groups 46-47
        K_ITER_INTERLEAVED(0, 184)
        K_ITER_INTERLEAVED(4, 188)
        "add x4, x4, #512\n\t"

        // K-groups 48-49
        K_ITER_INTERLEAVED(0, 192)
        K_ITER_INTERLEAVED(4, 196)
        "add x4, x4, #512\n\t"

        // K-groups 50-51
        K_ITER_INTERLEAVED(0, 200)
        K_ITER_INTERLEAVED(4, 204)
        "add x4, x4, #512\n\t"

        // K-groups 52-53
        K_ITER_INTERLEAVED(0, 208)
        K_ITER_INTERLEAVED(4, 212)
        "add x4, x4, #512\n\t"

        // K-groups 54-55
        K_ITER_INTERLEAVED(0, 216)
        K_ITER_INTERLEAVED(4, 220)
        "add x4, x4, #512\n\t"

        // K-groups 56-57
        K_ITER_INTERLEAVED(0, 224)
        K_ITER_INTERLEAVED(4, 228)
        "add x4, x4, #512\n\t"

        // K-groups 58-59
        K_ITER_INTERLEAVED(0, 232)
        K_ITER_INTERLEAVED(4, 236)
        "add x4, x4, #512\n\t"

        // K-groups 60-61
        K_ITER_INTERLEAVED(0, 240)
        K_ITER_INTERLEAVED(4, 244)
        "add x4, x4, #512\n\t"

        // K-groups 62-63 (last pair)
        K_ITER_INTERLEAVED(0, 248)
        K_ITER_INTERLEAVED(4, 252)

        // Reduce 4 K-tiles to 1 (sum across K dimension)
        "add z8.s, z8.s, z9.s\n\t"   "add z10.s, z10.s, z11.s\n\t" "add z8.s, z8.s, z10.s\n\t"
        "add z12.s, z12.s, z13.s\n\t" "add z14.s, z14.s, z15.s\n\t" "add z12.s, z12.s, z14.s\n\t"
        "add z16.s, z16.s, z17.s\n\t" "add z18.s, z18.s, z19.s\n\t" "add z16.s, z16.s, z18.s\n\t"
        "add z20.s, z20.s, z21.s\n\t" "add z22.s, z22.s, z23.s\n\t" "add z20.s, z20.s, z22.s\n\t"
        "add z24.s, z24.s, z25.s\n\t" "add z26.s, z26.s, z27.s\n\t" "add z24.s, z24.s, z26.s\n\t"
        "add z28.s, z28.s, z29.s\n\t" "add z30.s, z30.s, z31.s\n\t" "add z28.s, z28.s, z30.s\n\t"

        // Store 6 rows × 16 int32 = 6 × 64 bytes
        "mov x14, %[S]\n\t"
        "st1w {z8.s}, p1, [x14]\n\t"
        "st1w {z12.s}, p1, [x14, #1, mul vl]\n\t"
        "st1w {z16.s}, p1, [x14, #2, mul vl]\n\t"
        "st1w {z20.s}, p1, [x14, #3, mul vl]\n\t"
        "st1w {z24.s}, p1, [x14, #4, mul vl]\n\t"
        "st1w {z28.s}, p1, [x14, #5, mul vl]\n\t"

        :
        : [Q]"r"(Q), [K]"r"(K), [S]"r"(S)
        : "memory", "x4", "x5", "x6", "x7", "x11", "x12", "x13", "x14",
          "z0", "z1", "z2", "z3", "z4",
          "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15",
          "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23",
          "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31",
          "p0", "p1"
    );
}

// Phase 2 N-group iteration (same interleaved pattern)
#define N_ITER_INTERLEAVED(v_off, s_off) \
    "ld1b {z0.b}, p0/z, [x4, #" #v_off ", mul vl]\n\t" \
    "ld1b {z1.b}, p0/z, [x4, #" #v_off "+1, mul vl]\n\t" \
    "ld1b {z2.b}, p0/z, [x4, #" #v_off "+2, mul vl]\n\t" \
    "ld1b {z3.b}, p0/z, [x4, #" #v_off "+3, mul vl]\n\t" \
    /* Row 0 */ \
    "ld1rw {z4.s}, p1/z, [x5, #" #s_off "]\n\t" \
    "sdot z8.s, z0.b, z4.b\n\t"  "sdot z9.s, z1.b, z4.b\n\t" \
    "sdot z10.s, z2.b, z4.b\n\t" "sdot z11.s, z3.b, z4.b\n\t" \
    /* Row 1 */ \
    "ld1rw {z4.s}, p1/z, [x6, #" #s_off "]\n\t" \
    "sdot z12.s, z0.b, z4.b\n\t" "sdot z13.s, z1.b, z4.b\n\t" \
    "sdot z14.s, z2.b, z4.b\n\t" "sdot z15.s, z3.b, z4.b\n\t" \
    /* Row 2 */ \
    "ld1rw {z4.s}, p1/z, [x7, #" #s_off "]\n\t" \
    "sdot z16.s, z0.b, z4.b\n\t" "sdot z17.s, z1.b, z4.b\n\t" \
    "sdot z18.s, z2.b, z4.b\n\t" "sdot z19.s, z3.b, z4.b\n\t" \
    /* Row 3 */ \
    "ld1rw {z4.s}, p1/z, [x11, #" #s_off "]\n\t" \
    "sdot z20.s, z0.b, z4.b\n\t" "sdot z21.s, z1.b, z4.b\n\t" \
    "sdot z22.s, z2.b, z4.b\n\t" "sdot z23.s, z3.b, z4.b\n\t" \
    /* Row 4 */ \
    "ld1rw {z4.s}, p1/z, [x12, #" #s_off "]\n\t" \
    "sdot z24.s, z0.b, z4.b\n\t" "sdot z25.s, z1.b, z4.b\n\t" \
    "sdot z26.s, z2.b, z4.b\n\t" "sdot z27.s, z3.b, z4.b\n\t" \
    /* Row 5 */ \
    "ld1rw {z4.s}, p1/z, [x13, #" #s_off "]\n\t" \
    "sdot z28.s, z0.b, z4.b\n\t" "sdot z29.s, z1.b, z4.b\n\t" \
    "sdot z30.s, z2.b, z4.b\n\t" "sdot z31.s, z3.b, z4.b\n\t"

// Phase 2: S[6,64] @ V[64,64] -> O[6,64] for one D-tile
// 16 N iterations, 24 SDOT each = 384 SDOT per D-tile
__attribute__((noinline))
void phase2_interleaved_6row(const int8_t* S_quant, const int8_t* V, int32_t* O) {
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

        // S row pointers (S has 64 columns per row for attention scores)
        "mov x5, %[S]\n\t"
        "add x6, x5, #64\n\t"    // S row stride = 64 bytes
        "add x7, x6, #64\n\t"
        "add x11, x7, #64\n\t"
        "add x12, x11, #64\n\t"
        "add x13, x12, #64\n\t"
        "mov x4, %[V]\n\t"

        // 16 N-groups, V stride = 64*4 = 256 bytes per N-group
        // N-groups 0-1
        N_ITER_INTERLEAVED(0, 0)
        N_ITER_INTERLEAVED(4, 4)
        "add x4, x4, #512\n\t"

        // N-groups 2-3
        N_ITER_INTERLEAVED(0, 8)
        N_ITER_INTERLEAVED(4, 12)
        "add x4, x4, #512\n\t"

        // N-groups 4-5
        N_ITER_INTERLEAVED(0, 16)
        N_ITER_INTERLEAVED(4, 20)
        "add x4, x4, #512\n\t"

        // N-groups 6-7
        N_ITER_INTERLEAVED(0, 24)
        N_ITER_INTERLEAVED(4, 28)
        "add x4, x4, #512\n\t"

        // N-groups 8-9
        N_ITER_INTERLEAVED(0, 32)
        N_ITER_INTERLEAVED(4, 36)
        "add x4, x4, #512\n\t"

        // N-groups 10-11
        N_ITER_INTERLEAVED(0, 40)
        N_ITER_INTERLEAVED(4, 44)
        "add x4, x4, #512\n\t"

        // N-groups 12-13
        N_ITER_INTERLEAVED(0, 48)
        N_ITER_INTERLEAVED(4, 52)
        "add x4, x4, #512\n\t"

        // N-groups 14-15 (last)
        N_ITER_INTERLEAVED(0, 56)
        N_ITER_INTERLEAVED(4, 60)

        // Reduce 4 N-tiles to 1
        "add z8.s, z8.s, z9.s\n\t"   "add z10.s, z10.s, z11.s\n\t" "add z8.s, z8.s, z10.s\n\t"
        "add z12.s, z12.s, z13.s\n\t" "add z14.s, z14.s, z15.s\n\t" "add z12.s, z12.s, z14.s\n\t"
        "add z16.s, z16.s, z17.s\n\t" "add z18.s, z18.s, z19.s\n\t" "add z16.s, z16.s, z18.s\n\t"
        "add z20.s, z20.s, z21.s\n\t" "add z22.s, z22.s, z23.s\n\t" "add z20.s, z20.s, z22.s\n\t"
        "add z24.s, z24.s, z25.s\n\t" "add z26.s, z26.s, z27.s\n\t" "add z24.s, z24.s, z26.s\n\t"
        "add z28.s, z28.s, z29.s\n\t" "add z30.s, z30.s, z31.s\n\t" "add z28.s, z28.s, z30.s\n\t"

        // Store output
        "mov x14, %[O]\n\t"
        "st1w {z8.s}, p1, [x14]\n\t"
        "st1w {z12.s}, p1, [x14, #1, mul vl]\n\t"
        "st1w {z16.s}, p1, [x14, #2, mul vl]\n\t"
        "st1w {z20.s}, p1, [x14, #3, mul vl]\n\t"
        "st1w {z24.s}, p1, [x14, #4, mul vl]\n\t"
        "st1w {z28.s}, p1, [x14, #5, mul vl]\n\t"

        :
        : [S]"r"(S_quant), [V]"r"(V), [O]"r"(O)
        : "memory", "x4", "x5", "x6", "x7", "x11", "x12", "x13", "x14",
          "z0", "z1", "z2", "z3", "z4",
          "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15",
          "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23",
          "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31",
          "p0", "p1"
    );
}

int main() {
    printf("==============================================\n");
    printf("Fused Attention Interleaved Pattern (MR=6)\n");
    printf("==============================================\n\n");

    const int MR = 6;
    const int N = 64;  // Attention sequence length
    const int D = 256; // Head dimension

    // Allocate aligned buffers
    int8_t* Q = aligned_alloc(256, MR * D);
    int8_t* K = aligned_alloc(256, D * N);  // [256, 64] interleaved
    int8_t* V = aligned_alloc(256, N * D);  // [64, 256] interleaved
    int32_t* S = aligned_alloc(256, MR * N * sizeof(int32_t));  // Scores [6, 64]
    int8_t* S_quant = aligned_alloc(256, MR * N);  // Quantized scores
    int32_t* O = aligned_alloc(256, MR * D * sizeof(int32_t));  // Output [6, 256]

    // Initialize
    memset(Q, 1, MR * D);
    memset(K, 1, D * N);
    memset(V, 1, N * D);
    memset(S_quant, 1, MR * N);

    // SDOT counts
    // Phase 1: 64 K-iters × 24 SDOT = 1536 SDOT (one N-tile of 16 cols)
    // For full N=64: 4 N-tiles × 1536 = 6144 SDOT
    // Phase 2: 16 N-iters × 24 SDOT = 384 SDOT (one D-tile of 64 cols)
    // For full D=256: 4 D-tiles × 384 = 1536 SDOT

    int phase1_sdot_per_tile = 64 * 24;  // 1536
    int phase2_sdot_per_tile = 16 * 24;  // 384

    printf("Configuration: MR=%d, N=%d, D=%d\n", MR, N, D);
    printf("Phase 1 SDOT per N-tile: %d\n", phase1_sdot_per_tile);
    printf("Phase 2 SDOT per D-tile: %d\n", phase2_sdot_per_tile);
    printf("Peak: 40 SDOT/tick\n\n");

    // Warmup
    for (int i = 0; i < 100; i++) {
        phase1_interleaved_6row(Q, K, S);
        phase2_interleaved_6row(S_quant, V, O);
    }

    int iters = 10000;

    // Benchmark Phase 1
    uint64_t start = rdtick();
    for (int i = 0; i < iters; i++) {
        phase1_interleaved_6row(Q, K, S);
    }
    uint64_t end = rdtick();
    double ticks_p1 = (double)(end - start) / iters;
    double eff_p1 = (double)phase1_sdot_per_tile / ticks_p1 / 40.0 * 100.0;

    // Benchmark Phase 2
    start = rdtick();
    for (int i = 0; i < iters; i++) {
        phase2_interleaved_6row(S_quant, V, O);
    }
    end = rdtick();
    double ticks_p2 = (double)(end - start) / iters;
    double eff_p2 = (double)phase2_sdot_per_tile / ticks_p2 / 40.0 * 100.0;

    // Benchmark fused (P1 + P2)
    start = rdtick();
    for (int i = 0; i < iters; i++) {
        phase1_interleaved_6row(Q, K, S);
        phase2_interleaved_6row(S_quant, V, O);
    }
    end = rdtick();
    double ticks_fused = (double)(end - start) / iters;
    double total_sdot = phase1_sdot_per_tile + phase2_sdot_per_tile;  // 1920
    double eff_fused = total_sdot / ticks_fused / 40.0 * 100.0;

    printf("Approach                            Ticks Efficiency    SDOT/tick\n");
    printf("------------------------------------------------------------------\n");
    printf("Phase 1 interleaved (1 N-tile)   %8.1f     %5.1f%%     %8.2f\n",
           ticks_p1, eff_p1, (double)phase1_sdot_per_tile / ticks_p1);
    printf("Phase 2 interleaved (1 D-tile)   %8.1f     %5.1f%%     %8.2f\n",
           ticks_p2, eff_p2, (double)phase2_sdot_per_tile / ticks_p2);
    printf("Fused P1+P2 interleaved          %8.1f     %5.1f%%     %8.2f\n",
           ticks_fused, eff_fused, total_sdot / ticks_fused);

    printf("\nTarget: 95%% efficiency = 38.0 SDOT/tick\n");
    printf("Phase 1 min (100%%): %.1f ticks\n", (double)phase1_sdot_per_tile / 40.0);
    printf("Phase 2 min (100%%): %.1f ticks\n", (double)phase2_sdot_per_tile / 40.0);

    free(Q); free(K); free(V); free(S); free(S_quant); free(O);
    return 0;
}
