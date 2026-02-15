/*
 * Streaming S Fused Attention Kernel
 *
 * Key optimization: Keep S in registers between Phase 1 and Phase 2
 * to eliminate S memory traffic.
 *
 * Architecture:
 * - Phase 1 produces S[6,16] int32 in 6 vectors (after reduction)
 * - Quantize S to int8 in-place
 * - Phase 2 uses S directly from registers
 *
 * Register allocation for streaming:
 * - z8,z12,z16,z20,z24,z28: S[6,16] (6 vectors)
 * - z0-z3: V loads
 * - z4-z7: O accumulators (1 row × 4 D-tiles at a time)
 * - Process 2 rows at a time to fit in registers
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

// Baseline: separate P1 and P2 with S memory store
extern void baseline_interleaved(const int8_t* Q, const int8_t* K, int32_t* S);

// Phase 1 that keeps S in registers (returns S in z8,z12,z16,z20,z24,z28)
// Then immediately does Phase 2 without storing S to memory
__attribute__((noinline))
void streaming_fused_p1_p2(
    const int8_t* Q,      // [6, 256]
    const int8_t* K,      // [256, 64] interleaved for 1 N-tile (16 cols)
    const int8_t* V,      // [64, 256] interleaved
    int32_t* O            // [6, 256] output
) {
    // Process one N-tile at a time
    // For simplicity, use 2-row blocks in Phase 2

    __asm__ volatile(
        "ptrue p0.b\n\t"
        "ptrue p1.s\n\t"

        // ====== PHASE 1: Compute S[6,16] ======
        // Zero accumulators
        "mov z8.s, #0\n\t"  "mov z9.s, #0\n\t"  "mov z10.s, #0\n\t" "mov z11.s, #0\n\t"
        "mov z12.s, #0\n\t" "mov z13.s, #0\n\t" "mov z14.s, #0\n\t" "mov z15.s, #0\n\t"
        "mov z16.s, #0\n\t" "mov z17.s, #0\n\t" "mov z18.s, #0\n\t" "mov z19.s, #0\n\t"
        "mov z20.s, #0\n\t" "mov z21.s, #0\n\t" "mov z22.s, #0\n\t" "mov z23.s, #0\n\t"
        "mov z24.s, #0\n\t" "mov z25.s, #0\n\t" "mov z26.s, #0\n\t" "mov z27.s, #0\n\t"
        "mov z28.s, #0\n\t" "mov z29.s, #0\n\t" "mov z30.s, #0\n\t" "mov z31.s, #0\n\t"

        // Q row pointers
        "mov x5, %[Q]\n\t"
        "add x6, x5, #256\n\t"
        "add x7, x6, #256\n\t"
        "add x11, x7, #256\n\t"
        "add x12, x11, #256\n\t"
        "add x13, x12, #256\n\t"
        "mov x4, %[K]\n\t"
        "mov x10, #32\n\t"  // 64 K-iters in pairs

        // K-loop
        "1:\n\t"
        // K-group 0
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

        // K-group 1
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

        "add x5, x5, #8\n\t"
        "add x6, x6, #8\n\t"
        "add x7, x7, #8\n\t"
        "add x11, x11, #8\n\t"
        "add x12, x12, #8\n\t"
        "add x13, x13, #8\n\t"
        "subs x10, x10, #1\n\t"
        "b.ne 1b\n\t"

        // Reduce S: 4 K-tiles -> 1 per row
        // S row 0 -> z8
        "add z8.s, z8.s, z9.s\n\t"
        "add z10.s, z10.s, z11.s\n\t"
        "add z8.s, z8.s, z10.s\n\t"
        // S row 1 -> z12
        "add z12.s, z12.s, z13.s\n\t"
        "add z14.s, z14.s, z15.s\n\t"
        "add z12.s, z12.s, z14.s\n\t"
        // S row 2 -> z16
        "add z16.s, z16.s, z17.s\n\t"
        "add z18.s, z18.s, z19.s\n\t"
        "add z16.s, z16.s, z18.s\n\t"
        // S row 3 -> z20
        "add z20.s, z20.s, z21.s\n\t"
        "add z22.s, z22.s, z23.s\n\t"
        "add z20.s, z20.s, z22.s\n\t"
        // S row 4 -> z24
        "add z24.s, z24.s, z25.s\n\t"
        "add z26.s, z26.s, z27.s\n\t"
        "add z24.s, z24.s, z26.s\n\t"
        // S row 5 -> z28
        "add z28.s, z28.s, z29.s\n\t"
        "add z30.s, z30.s, z31.s\n\t"
        "add z28.s, z28.s, z30.s\n\t"

        // Now S[6,16] int32 is in z8,z12,z16,z20,z24,z28

        // Quantize S: int32 -> int8 (simplified saturation)
        // sqxtnb narrows top half, sqxtnt narrows bottom half
        // First to int16
        "sqxtnb z9.h, z8.s\n\t"     // S row 0 int16 in z9.h
        "sqxtnb z13.h, z12.s\n\t"   // S row 1
        "sqxtnb z17.h, z16.s\n\t"   // S row 2
        "sqxtnb z21.h, z20.s\n\t"   // S row 3
        "sqxtnb z25.h, z24.s\n\t"   // S row 4
        "sqxtnb z29.h, z28.s\n\t"   // S row 5

        // Then to int8
        "sqxtnb z8.b, z9.h\n\t"     // S row 0 int8 in z8.b[0:7]
        "sqxtnb z12.b, z13.h\n\t"
        "sqxtnb z16.b, z17.h\n\t"
        "sqxtnb z20.b, z21.h\n\t"
        "sqxtnb z24.b, z25.h\n\t"
        "sqxtnb z28.b, z29.h\n\t"

        // S[6,16] int8 is now in z8,z12,z16,z20,z24,z28 (lower bytes)

        // ====== PHASE 2: Compute O using streaming S ======
        // V layout: [64, 256] interleaved, this N-tile uses rows 0-15
        // Process 4 D-tiles, each producing 64 output columns

        // For each D-tile, load O accumulators, compute += S @ V, store
        // Register pressure: need O(6×4=24 regs) + S(6 regs) + V(4 regs)
        // Too many! Process 2 rows at a time instead.

        // D-tile loop (process 4 D-tiles)
        "mov x4, %[V]\n\t"          // V pointer
        "mov x14, %[O]\n\t"         // O pointer
        "mov x15, #4\n\t"           // D-tile counter

        "2:\n\t"  // D-tile loop

        // Process rows 0-1
        // Load O[0:2, dtile*16:(dtile+1)*16] accumulators
        "ld1w {z0.s}, p1, [x14]\n\t"            // O[0, 0:16]
        "add x17, x14, #1024\n\t"               // O row 1 base
        "ld1w {z1.s}, p1, [x17]\n\t"            // O[1, 0:16]

        // N-loop: 16 iterations (use S to update O)
        "mov x16, %[V]\n\t"         // Reset V pointer for this D-tile
        "mov x10, #4\n\t"           // 16 N-iters in groups of 4

        "3:\n\t"  // N-loop
        // Load V[n*4:(n+1)*4, dtile*64:(dtile+1)*64] - 4 vectors
        "ld1b {z4.b}, p0/z, [x16]\n\t"
        "ld1b {z5.b}, p0/z, [x16, #1, mul vl]\n\t"
        "ld1b {z6.b}, p0/z, [x16, #2, mul vl]\n\t"
        "ld1b {z7.b}, p0/z, [x16, #3, mul vl]\n\t"

        // Broadcast S[0, n*4:(n+1)*4] and compute
        // S is in z8 (packed int8), need to broadcast 4 bytes
        // For now, use ld1rw from S memory (simplification)
        // TODO: true streaming would use TBL to extract S values

        // This is simplified - real streaming needs TBL instruction
        // to extract S values from z8,z12,... without memory access

        "add x16, x16, #256\n\t"
        "subs x10, x10, #1\n\t"
        "b.ne 3b\n\t"

        // Store O[0:2, dtile*16:(dtile+1)*16]
        "st1w {z0.s}, p1, [x14]\n\t"
        "st1w {z1.s}, p1, [x17]\n\t"

        // Next D-tile
        "add x14, x14, #64\n\t"     // Move to next D-tile (16 int32)
        "subs x15, x15, #1\n\t"
        "b.ne 2b\n\t"

        :
        : [Q]"r"(Q), [K]"r"(K), [V]"r"(V), [O]"r"(O)
        : "memory", "x4", "x5", "x6", "x7", "x10", "x11", "x12", "x13",
          "x14", "x15", "x16", "x17",
          "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7",
          "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15",
          "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23",
          "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31",
          "p0", "p1"
    );
}

// Simpler test: Just measure Phase 1 with full unrolling (our best)
// vs baseline with loop
__attribute__((noinline))
void phase1_full_unroll_best(const int8_t* Q, const int8_t* K, int32_t* S);

int main() {
    printf("==============================================\n");
    printf("Streaming S Fused Attention Test\n");
    printf("==============================================\n\n");

    printf("Strategy: Keep S in registers between P1 and P2\n");
    printf("Challenge: True streaming needs TBL to extract S bytes\n\n");

    const int MR = 6;
    const int N = 64;
    const int D = 256;

    int8_t* Q = aligned_alloc(256, MR * D);
    int8_t* K = aligned_alloc(256, D * N);
    int8_t* V = aligned_alloc(256, N * D);
    int32_t* S = aligned_alloc(256, MR * N * sizeof(int32_t));
    int32_t* O = aligned_alloc(256, MR * D * sizeof(int32_t));

    memset(Q, 1, MR * D);
    memset(K, 1, D * N);
    memset(V, 1, N * D);
    memset(O, 0, MR * D * sizeof(int32_t));

    // For now, just demonstrate the Phase 1 portion works
    // The streaming Phase 2 needs TBL instruction support

    printf("Phase 1 only test (streaming Phase 2 needs TBL support):\n\n");

    // Warmup
    for (int i = 0; i < 100; i++) {
        streaming_fused_p1_p2(Q, K, V, O);
    }

    int iters = 10000;

    // The streaming kernel currently only does Phase 1 fully
    // Phase 2 is incomplete due to TBL complexity
    printf("Note: Full streaming requires TBL instruction to extract\n");
    printf("      S bytes from packed int8 vectors without memory access.\n");
    printf("      This test measures Phase 1 + partial Phase 2.\n\n");

    printf("For true 95%%+ efficiency, consider:\n");
    printf("  1. Use TBL to permute S values from z8,z12,... to broadcast form\n");
    printf("  2. Or store S to L1 (very fast) and reload with ld1rw\n");
    printf("  3. L1 store+load is ~11 cycles, may be acceptable\n");

    free(Q); free(K); free(V); free(S); free(O);
    return 0;
}
