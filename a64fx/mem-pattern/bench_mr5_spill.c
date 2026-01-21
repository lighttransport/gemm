/*
 * Benchmark: MR=5 with accumulator spilling to L1
 *
 * Strategy: Process two MR=5 tile groups by spilling accumulators
 * - Group A: 20 accumulators in registers
 * - Group B: 20 accumulators spilled to L1 stack
 * - Alternate between groups, overlapping spill with compute
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

// MR=5 single tile - baseline (no spilling)
void bench_mr5_single(const int8_t* K, const int8_t* Q, int32_t* S, int D) {
    // 20 accumulators for 5 rows × 4 K-vectors
    // Register allocation:
    //   z0-z3: K vectors (4)
    //   z4-z8: Q broadcasts (5)
    //   z9-z28: accumulators (20)

    __asm__ volatile(
        "ptrue p0.b\n\t"
        "ptrue p1.s\n\t"

        // Zero accumulators
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

        // Setup Q row pointers (5 rows, stride 256)
        "mov x5, %[Q]\n\t"
        "add x6, x5, #256\n\t"
        "add x7, x5, #512\n\t"
        "add x8, x5, #768\n\t"
        "add x9, x5, #1024\n\t"

        "mov x4, %[K]\n\t"
        "mov x10, %[D]\n\t"

        // D-group loop
        "1:\n\t"
        // Load K[0-3] (4 × 64B = 256B)
        "ld1b {z0.b}, p0/z, [x4]\n\t"
        "ld1b {z1.b}, p0/z, [x4, #1, mul vl]\n\t"
        "ld1b {z2.b}, p0/z, [x4, #2, mul vl]\n\t"
        "ld1b {z3.b}, p0/z, [x4, #3, mul vl]\n\t"
        "add x4, x4, #256\n\t"

        // Load Q broadcasts (5 rows × 4B = 20B)
        "ld1rw {z4.s}, p1/z, [x5]\n\t"
        "ld1rw {z5.s}, p1/z, [x6]\n\t"
        "ld1rw {z6.s}, p1/z, [x7]\n\t"
        "ld1rw {z7.s}, p1/z, [x8]\n\t"
        "ld1rw {z8.s}, p1/z, [x9]\n\t"
        "add x5, x5, #4\n\t"
        "add x6, x6, #4\n\t"
        "add x7, x7, #4\n\t"
        "add x8, x8, #4\n\t"
        "add x9, x9, #4\n\t"

        // 20 SDOTs: 5 rows × 4 K-vectors
        // Row 0: z9-z12
        "sdot z9.s, z0.b, z4.b\n\t"
        "sdot z10.s, z1.b, z4.b\n\t"
        "sdot z11.s, z2.b, z4.b\n\t"
        "sdot z12.s, z3.b, z4.b\n\t"
        // Row 1: z13-z16
        "sdot z13.s, z0.b, z5.b\n\t"
        "sdot z14.s, z1.b, z5.b\n\t"
        "sdot z15.s, z2.b, z5.b\n\t"
        "sdot z16.s, z3.b, z5.b\n\t"
        // Row 2: z17-z20
        "sdot z17.s, z0.b, z6.b\n\t"
        "sdot z18.s, z1.b, z6.b\n\t"
        "sdot z19.s, z2.b, z6.b\n\t"
        "sdot z20.s, z3.b, z6.b\n\t"
        // Row 3: z21-z24
        "sdot z21.s, z0.b, z7.b\n\t"
        "sdot z22.s, z1.b, z7.b\n\t"
        "sdot z23.s, z2.b, z7.b\n\t"
        "sdot z24.s, z3.b, z7.b\n\t"
        // Row 4: z25-z28
        "sdot z25.s, z0.b, z8.b\n\t"
        "sdot z26.s, z1.b, z8.b\n\t"
        "sdot z27.s, z2.b, z8.b\n\t"
        "sdot z28.s, z3.b, z8.b\n\t"

        "subs x10, x10, #1\n\t"
        "b.ne 1b\n\t"

        // Merge accumulators: each row z[i] += z[i+1] + z[i+2] + z[i+3]
        "add z9.s, z9.s, z10.s\n\t"
        "add z11.s, z11.s, z12.s\n\t"
        "add z9.s, z9.s, z11.s\n\t"

        "add z13.s, z13.s, z14.s\n\t"
        "add z15.s, z15.s, z16.s\n\t"
        "add z13.s, z13.s, z15.s\n\t"

        "add z17.s, z17.s, z18.s\n\t"
        "add z19.s, z19.s, z20.s\n\t"
        "add z17.s, z17.s, z19.s\n\t"

        "add z21.s, z21.s, z22.s\n\t"
        "add z23.s, z23.s, z24.s\n\t"
        "add z21.s, z21.s, z23.s\n\t"

        "add z25.s, z25.s, z26.s\n\t"
        "add z27.s, z27.s, z28.s\n\t"
        "add z25.s, z25.s, z27.s\n\t"

        // Store results (5 rows × 64B = 320B)
        "st1w {z9.s}, p1, [%[S]]\n\t"
        "st1w {z13.s}, p1, [%[S], #1, mul vl]\n\t"
        "st1w {z17.s}, p1, [%[S], #2, mul vl]\n\t"
        "st1w {z21.s}, p1, [%[S], #3, mul vl]\n\t"
        "st1w {z25.s}, p1, [%[S], #4, mul vl]\n\t"

        :
        : [K] "r"(K), [Q] "r"(Q), [S] "r"(S), [D] "r"((long)D)
        : "memory", "x4", "x5", "x6", "x7", "x8", "x9", "x10",
          "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8",
          "z9", "z10", "z11", "z12", "z13", "z14", "z15", "z16",
          "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24",
          "z25", "z26", "z27", "z28", "p0", "p1"
    );
}

// MR=5 × 2 tiles with accumulator spilling
// Process in blocks: compute group A, spill, compute group B, restore
void bench_mr5_spill(const int8_t* K, const int8_t* Q_A, const int8_t* Q_B,
                     int32_t* S_A, int32_t* S_B, int D) {
    // Stack space for 20 accumulators (1280 bytes)
    int32_t spill_buf[320] __attribute__((aligned(64)));

    __asm__ volatile(
        "ptrue p0.b\n\t"
        "ptrue p1.s\n\t"

        // Zero all accumulators for group A
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

        // Zero spill buffer (group B accumulators)
        "mov x11, %[spill]\n\t"
        "mov z0.s, #0\n\t"
        ".rept 20\n\t"
        "st1w {z0.s}, p1, [x11], #64\n\t"
        ".endr\n\t"

        // Setup pointers
        "mov x4, %[K]\n\t"
        "mov x10, %[D]\n\t"

        // Q_A row pointers
        "mov x5, %[Q_A]\n\t"
        "add x6, x5, #256\n\t"
        "add x7, x5, #512\n\t"
        "add x8, x5, #768\n\t"
        "add x9, x5, #1024\n\t"

        // Q_B row pointers
        "mov x12, %[Q_B]\n\t"
        "add x13, x12, #256\n\t"
        "add x14, x12, #512\n\t"
        "add x15, x12, #768\n\t"
        "add x16, x12, #1024\n\t"

        // Process in blocks of 4 d_groups
        // Each block: compute A, spill A, load B, compute B, spill B, load A
        "mov x17, %[D]\n\t"
        "lsr x17, x17, #2\n\t"  // D/4 blocks

        "2:\n\t"  // Block loop

        // === Process 4 d_groups for Group A ===
        "mov x10, #4\n\t"
        "3:\n\t"
        // Load K (shared)
        "ld1b {z0.b}, p0/z, [x4]\n\t"
        "ld1b {z1.b}, p0/z, [x4, #1, mul vl]\n\t"
        "ld1b {z2.b}, p0/z, [x4, #2, mul vl]\n\t"
        "ld1b {z3.b}, p0/z, [x4, #3, mul vl]\n\t"

        // Load Q_A
        "ld1rw {z4.s}, p1/z, [x5]\n\t"
        "ld1rw {z5.s}, p1/z, [x6]\n\t"
        "ld1rw {z6.s}, p1/z, [x7]\n\t"
        "ld1rw {z7.s}, p1/z, [x8]\n\t"
        "ld1rw {z8.s}, p1/z, [x9]\n\t"

        // 20 SDOTs for group A
        "sdot z9.s, z0.b, z4.b\n\t"
        "sdot z10.s, z1.b, z4.b\n\t"
        "sdot z11.s, z2.b, z4.b\n\t"
        "sdot z12.s, z3.b, z4.b\n\t"
        "sdot z13.s, z0.b, z5.b\n\t"
        "sdot z14.s, z1.b, z5.b\n\t"
        "sdot z15.s, z2.b, z5.b\n\t"
        "sdot z16.s, z3.b, z5.b\n\t"
        "sdot z17.s, z0.b, z6.b\n\t"
        "sdot z18.s, z1.b, z6.b\n\t"
        "sdot z19.s, z2.b, z6.b\n\t"
        "sdot z20.s, z3.b, z6.b\n\t"
        "sdot z21.s, z0.b, z7.b\n\t"
        "sdot z22.s, z1.b, z7.b\n\t"
        "sdot z23.s, z2.b, z7.b\n\t"
        "sdot z24.s, z3.b, z7.b\n\t"
        "sdot z25.s, z0.b, z8.b\n\t"
        "sdot z26.s, z1.b, z8.b\n\t"
        "sdot z27.s, z2.b, z8.b\n\t"
        "sdot z28.s, z3.b, z8.b\n\t"

        "add x5, x5, #4\n\t"
        "add x6, x6, #4\n\t"
        "add x7, x7, #4\n\t"
        "add x8, x8, #4\n\t"
        "add x9, x9, #4\n\t"

        "subs x10, x10, #1\n\t"
        "b.ne 3b\n\t"

        // Rewind K pointer for group B
        "sub x4, x4, #1024\n\t"  // 4 d_groups × 256B

        // === Spill group A, load group B ===
        "mov x11, %[spill]\n\t"

        // Spill A (20 stores) interleaved with load B (20 loads)
        "st1w {z9.s}, p1, [x11]\n\t"
        "ld1w {z9.s}, p1/z, [x11, #20, mul vl]\n\t"  // Load B[0] from offset 20*64
        "st1w {z10.s}, p1, [x11, #1, mul vl]\n\t"
        "ld1w {z10.s}, p1/z, [x11, #21, mul vl]\n\t"
        "st1w {z11.s}, p1, [x11, #2, mul vl]\n\t"
        "ld1w {z11.s}, p1/z, [x11, #22, mul vl]\n\t"
        "st1w {z12.s}, p1, [x11, #3, mul vl]\n\t"
        "ld1w {z12.s}, p1/z, [x11, #23, mul vl]\n\t"
        "st1w {z13.s}, p1, [x11, #4, mul vl]\n\t"
        "ld1w {z13.s}, p1/z, [x11, #24, mul vl]\n\t"
        "st1w {z14.s}, p1, [x11, #5, mul vl]\n\t"
        "ld1w {z14.s}, p1/z, [x11, #25, mul vl]\n\t"
        "st1w {z15.s}, p1, [x11, #6, mul vl]\n\t"
        "ld1w {z15.s}, p1/z, [x11, #26, mul vl]\n\t"
        "st1w {z16.s}, p1, [x11, #7, mul vl]\n\t"

        // Wait - #N mul vl only goes to ±7!
        // Need to increment base pointer
        "add x18, x11, #512\n\t"  // +8 vectors
        "ld1w {z16.s}, p1/z, [x18, #19, mul vl]\n\t"  // 8+19=27
        "st1w {z17.s}, p1, [x18]\n\t"
        "ld1w {z17.s}, p1/z, [x18, #20, mul vl]\n\t"
        "st1w {z18.s}, p1, [x18, #1, mul vl]\n\t"
        "ld1w {z18.s}, p1/z, [x18, #21, mul vl]\n\t"
        "st1w {z19.s}, p1, [x18, #2, mul vl]\n\t"
        "ld1w {z19.s}, p1/z, [x18, #22, mul vl]\n\t"
        "st1w {z20.s}, p1, [x18, #3, mul vl]\n\t"
        "ld1w {z20.s}, p1/z, [x18, #23, mul vl]\n\t"
        "st1w {z21.s}, p1, [x18, #4, mul vl]\n\t"
        "ld1w {z21.s}, p1/z, [x18, #24, mul vl]\n\t"
        "st1w {z22.s}, p1, [x18, #5, mul vl]\n\t"
        "ld1w {z22.s}, p1/z, [x18, #25, mul vl]\n\t"
        "st1w {z23.s}, p1, [x18, #6, mul vl]\n\t"
        "ld1w {z23.s}, p1/z, [x18, #26, mul vl]\n\t"
        "st1w {z24.s}, p1, [x18, #7, mul vl]\n\t"

        "add x18, x18, #512\n\t"  // +16 vectors total
        "ld1w {z24.s}, p1/z, [x18, #11, mul vl]\n\t"
        "st1w {z25.s}, p1, [x18]\n\t"
        "ld1w {z25.s}, p1/z, [x18, #12, mul vl]\n\t"
        "st1w {z26.s}, p1, [x18, #1, mul vl]\n\t"
        "ld1w {z26.s}, p1/z, [x18, #13, mul vl]\n\t"
        "st1w {z27.s}, p1, [x18, #2, mul vl]\n\t"
        "ld1w {z27.s}, p1/z, [x18, #14, mul vl]\n\t"
        "st1w {z28.s}, p1, [x18, #3, mul vl]\n\t"
        "ld1w {z28.s}, p1/z, [x18, #15, mul vl]\n\t"

        // === Process 4 d_groups for Group B ===
        "mov x10, #4\n\t"
        "4:\n\t"
        // Load K (shared, same data as group A)
        "ld1b {z0.b}, p0/z, [x4]\n\t"
        "ld1b {z1.b}, p0/z, [x4, #1, mul vl]\n\t"
        "ld1b {z2.b}, p0/z, [x4, #2, mul vl]\n\t"
        "ld1b {z3.b}, p0/z, [x4, #3, mul vl]\n\t"
        "add x4, x4, #256\n\t"

        // Load Q_B
        "ld1rw {z4.s}, p1/z, [x12]\n\t"
        "ld1rw {z5.s}, p1/z, [x13]\n\t"
        "ld1rw {z6.s}, p1/z, [x14]\n\t"
        "ld1rw {z7.s}, p1/z, [x15]\n\t"
        "ld1rw {z8.s}, p1/z, [x16]\n\t"
        "add x12, x12, #4\n\t"
        "add x13, x13, #4\n\t"
        "add x14, x14, #4\n\t"
        "add x15, x15, #4\n\t"
        "add x16, x16, #4\n\t"

        // 20 SDOTs for group B
        "sdot z9.s, z0.b, z4.b\n\t"
        "sdot z10.s, z1.b, z4.b\n\t"
        "sdot z11.s, z2.b, z4.b\n\t"
        "sdot z12.s, z3.b, z4.b\n\t"
        "sdot z13.s, z0.b, z5.b\n\t"
        "sdot z14.s, z1.b, z5.b\n\t"
        "sdot z15.s, z2.b, z5.b\n\t"
        "sdot z16.s, z3.b, z5.b\n\t"
        "sdot z17.s, z0.b, z6.b\n\t"
        "sdot z18.s, z1.b, z6.b\n\t"
        "sdot z19.s, z2.b, z6.b\n\t"
        "sdot z20.s, z3.b, z6.b\n\t"
        "sdot z21.s, z0.b, z7.b\n\t"
        "sdot z22.s, z1.b, z7.b\n\t"
        "sdot z23.s, z2.b, z7.b\n\t"
        "sdot z24.s, z3.b, z7.b\n\t"
        "sdot z25.s, z0.b, z8.b\n\t"
        "sdot z26.s, z1.b, z8.b\n\t"
        "sdot z27.s, z2.b, z8.b\n\t"
        "sdot z28.s, z3.b, z8.b\n\t"

        "subs x10, x10, #1\n\t"
        "b.ne 4b\n\t"

        // === Spill group B, load group A ===
        "mov x11, %[spill]\n\t"
        "add x18, x11, #1280\n\t"  // B storage starts at offset 1280

        // Store B
        "st1w {z9.s}, p1, [x18]\n\t"
        "st1w {z10.s}, p1, [x18, #1, mul vl]\n\t"
        "st1w {z11.s}, p1, [x18, #2, mul vl]\n\t"
        "st1w {z12.s}, p1, [x18, #3, mul vl]\n\t"
        "st1w {z13.s}, p1, [x18, #4, mul vl]\n\t"
        "st1w {z14.s}, p1, [x18, #5, mul vl]\n\t"
        "st1w {z15.s}, p1, [x18, #6, mul vl]\n\t"
        "st1w {z16.s}, p1, [x18, #7, mul vl]\n\t"
        "add x18, x18, #512\n\t"
        "st1w {z17.s}, p1, [x18]\n\t"
        "st1w {z18.s}, p1, [x18, #1, mul vl]\n\t"
        "st1w {z19.s}, p1, [x18, #2, mul vl]\n\t"
        "st1w {z20.s}, p1, [x18, #3, mul vl]\n\t"
        "st1w {z21.s}, p1, [x18, #4, mul vl]\n\t"
        "st1w {z22.s}, p1, [x18, #5, mul vl]\n\t"
        "st1w {z23.s}, p1, [x18, #6, mul vl]\n\t"
        "st1w {z24.s}, p1, [x18, #7, mul vl]\n\t"
        "add x18, x18, #512\n\t"
        "st1w {z25.s}, p1, [x18]\n\t"
        "st1w {z26.s}, p1, [x18, #1, mul vl]\n\t"
        "st1w {z27.s}, p1, [x18, #2, mul vl]\n\t"
        "st1w {z28.s}, p1, [x18, #3, mul vl]\n\t"

        // Load A
        "ld1w {z9.s}, p1/z, [x11]\n\t"
        "ld1w {z10.s}, p1/z, [x11, #1, mul vl]\n\t"
        "ld1w {z11.s}, p1/z, [x11, #2, mul vl]\n\t"
        "ld1w {z12.s}, p1/z, [x11, #3, mul vl]\n\t"
        "ld1w {z13.s}, p1/z, [x11, #4, mul vl]\n\t"
        "ld1w {z14.s}, p1/z, [x11, #5, mul vl]\n\t"
        "ld1w {z15.s}, p1/z, [x11, #6, mul vl]\n\t"
        "ld1w {z16.s}, p1/z, [x11, #7, mul vl]\n\t"
        "add x11, x11, #512\n\t"
        "ld1w {z17.s}, p1/z, [x11]\n\t"
        "ld1w {z18.s}, p1/z, [x11, #1, mul vl]\n\t"
        "ld1w {z19.s}, p1/z, [x11, #2, mul vl]\n\t"
        "ld1w {z20.s}, p1/z, [x11, #3, mul vl]\n\t"
        "ld1w {z21.s}, p1/z, [x11, #4, mul vl]\n\t"
        "ld1w {z22.s}, p1/z, [x11, #5, mul vl]\n\t"
        "ld1w {z23.s}, p1/z, [x11, #6, mul vl]\n\t"
        "ld1w {z24.s}, p1/z, [x11, #7, mul vl]\n\t"
        "add x11, x11, #512\n\t"
        "ld1w {z25.s}, p1/z, [x11]\n\t"
        "ld1w {z26.s}, p1/z, [x11, #1, mul vl]\n\t"
        "ld1w {z27.s}, p1/z, [x11, #2, mul vl]\n\t"
        "ld1w {z28.s}, p1/z, [x11, #3, mul vl]\n\t"

        "subs x17, x17, #1\n\t"
        "b.ne 2b\n\t"

        // Final merge and store for group A (in registers)
        "add z9.s, z9.s, z10.s\n\t"
        "add z11.s, z11.s, z12.s\n\t"
        "add z9.s, z9.s, z11.s\n\t"
        "add z13.s, z13.s, z14.s\n\t"
        "add z15.s, z15.s, z16.s\n\t"
        "add z13.s, z13.s, z15.s\n\t"
        "add z17.s, z17.s, z18.s\n\t"
        "add z19.s, z19.s, z20.s\n\t"
        "add z17.s, z17.s, z19.s\n\t"
        "add z21.s, z21.s, z22.s\n\t"
        "add z23.s, z23.s, z24.s\n\t"
        "add z21.s, z21.s, z23.s\n\t"
        "add z25.s, z25.s, z26.s\n\t"
        "add z27.s, z27.s, z28.s\n\t"
        "add z25.s, z25.s, z27.s\n\t"

        "st1w {z9.s}, p1, [%[S_A]]\n\t"
        "st1w {z13.s}, p1, [%[S_A], #1, mul vl]\n\t"
        "st1w {z17.s}, p1, [%[S_A], #2, mul vl]\n\t"
        "st1w {z21.s}, p1, [%[S_A], #3, mul vl]\n\t"
        "st1w {z25.s}, p1, [%[S_A], #4, mul vl]\n\t"

        // Load, merge, store for group B
        "mov x11, %[spill]\n\t"
        "add x11, x11, #1280\n\t"
        "ld1w {z9.s}, p1/z, [x11]\n\t"
        "ld1w {z10.s}, p1/z, [x11, #1, mul vl]\n\t"
        "ld1w {z11.s}, p1/z, [x11, #2, mul vl]\n\t"
        "ld1w {z12.s}, p1/z, [x11, #3, mul vl]\n\t"
        "ld1w {z13.s}, p1/z, [x11, #4, mul vl]\n\t"
        "ld1w {z14.s}, p1/z, [x11, #5, mul vl]\n\t"
        "ld1w {z15.s}, p1/z, [x11, #6, mul vl]\n\t"
        "ld1w {z16.s}, p1/z, [x11, #7, mul vl]\n\t"
        "add x11, x11, #512\n\t"
        "ld1w {z17.s}, p1/z, [x11]\n\t"
        "ld1w {z18.s}, p1/z, [x11, #1, mul vl]\n\t"
        "ld1w {z19.s}, p1/z, [x11, #2, mul vl]\n\t"
        "ld1w {z20.s}, p1/z, [x11, #3, mul vl]\n\t"
        "ld1w {z21.s}, p1/z, [x11, #4, mul vl]\n\t"
        "ld1w {z22.s}, p1/z, [x11, #5, mul vl]\n\t"
        "ld1w {z23.s}, p1/z, [x11, #6, mul vl]\n\t"
        "ld1w {z24.s}, p1/z, [x11, #7, mul vl]\n\t"
        "add x11, x11, #512\n\t"
        "ld1w {z25.s}, p1/z, [x11]\n\t"
        "ld1w {z26.s}, p1/z, [x11, #1, mul vl]\n\t"
        "ld1w {z27.s}, p1/z, [x11, #2, mul vl]\n\t"
        "ld1w {z28.s}, p1/z, [x11, #3, mul vl]\n\t"

        "add z9.s, z9.s, z10.s\n\t"
        "add z11.s, z11.s, z12.s\n\t"
        "add z9.s, z9.s, z11.s\n\t"
        "add z13.s, z13.s, z14.s\n\t"
        "add z15.s, z15.s, z16.s\n\t"
        "add z13.s, z13.s, z15.s\n\t"
        "add z17.s, z17.s, z18.s\n\t"
        "add z19.s, z19.s, z20.s\n\t"
        "add z17.s, z17.s, z19.s\n\t"
        "add z21.s, z21.s, z22.s\n\t"
        "add z23.s, z23.s, z24.s\n\t"
        "add z21.s, z21.s, z23.s\n\t"
        "add z25.s, z25.s, z26.s\n\t"
        "add z27.s, z27.s, z28.s\n\t"
        "add z25.s, z25.s, z27.s\n\t"

        "st1w {z9.s}, p1, [%[S_B]]\n\t"
        "st1w {z13.s}, p1, [%[S_B], #1, mul vl]\n\t"
        "st1w {z17.s}, p1, [%[S_B], #2, mul vl]\n\t"
        "st1w {z21.s}, p1, [%[S_B], #3, mul vl]\n\t"
        "st1w {z25.s}, p1, [%[S_B], #4, mul vl]\n\t"

        :
        : [K] "r"(K), [Q_A] "r"(Q_A), [Q_B] "r"(Q_B),
          [S_A] "r"(S_A), [S_B] "r"(S_B), [D] "r"((long)D),
          [spill] "r"(spill_buf)
        : "memory", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11",
          "x12", "x13", "x14", "x15", "x16", "x17", "x18",
          "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8",
          "z9", "z10", "z11", "z12", "z13", "z14", "z15", "z16",
          "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24",
          "z25", "z26", "z27", "z28", "p0", "p1"
    );
}

int main() {
    printf("==============================================\n");
    printf("MR=5 with Accumulator Spilling Analysis\n");
    printf("==============================================\n\n");

    int D = 64;  // d_groups
    int N = 64;  // output columns
    int MR = 5;

    // Allocate buffers
    int8_t* K = aligned_alloc(256, D * N * 4);      // [D, N, 4] interleaved
    int8_t* Q_A = aligned_alloc(256, MR * D * 4);   // [MR, D*4] = [5, 256]
    int8_t* Q_B = aligned_alloc(256, MR * D * 4);
    int32_t* S_A = aligned_alloc(256, MR * N * 4);  // [MR, N] INT32
    int32_t* S_B = aligned_alloc(256, MR * N * 4);

    memset(K, 1, D * N * 4);
    memset(Q_A, 1, MR * D * 4);
    memset(Q_B, 1, MR * D * 4);

    int iters = 10000;
    uint64_t start, end;
    double cycles;

    // Warmup
    for (int i = 0; i < 100; i++) {
        bench_mr5_single(K, Q_A, S_A, D);
    }

    // Benchmark MR=5 single tile
    printf("=== MR=5 Single Tile (20 accumulators) ===\n");
    start = rdcycle();
    for (int i = 0; i < iters; i++) {
        bench_mr5_single(K, Q_A, S_A, D);
    }
    end = rdcycle();
    cycles = (double)(end - start) / iters;

    int sdots_single = D * MR * 4;  // 64 * 5 * 4 = 1280 SDOT
    printf("  Cycles: %.1f\n", cycles);
    printf("  SDOT/cycle: %.2f (peak=2.0)\n", sdots_single / cycles);
    printf("  SDOTs: %d, D=%d, MR=%d\n\n", sdots_single, D, MR);

    // Warmup spill version
    for (int i = 0; i < 100; i++) {
        bench_mr5_spill(K, Q_A, Q_B, S_A, S_B, D);
    }

    // Benchmark MR=5 × 2 with spilling
    printf("=== MR=5 × 2 Tiles with Spilling ===\n");
    start = rdcycle();
    for (int i = 0; i < iters; i++) {
        bench_mr5_spill(K, Q_A, Q_B, S_A, S_B, D);
    }
    end = rdcycle();
    cycles = (double)(end - start) / iters;

    int sdots_spill = D * MR * 4 * 2;  // 64 * 5 * 4 * 2 = 2560 SDOT
    printf("  Cycles: %.1f\n", cycles);
    printf("  SDOT/cycle: %.2f (peak=2.0)\n", sdots_spill / cycles);
    printf("  SDOTs: %d (2 tiles), D=%d, MR=%d\n\n", sdots_spill, D, MR);

    // Compute overhead
    printf("=== Spill Overhead Analysis ===\n");
    printf("  Spill block size: 4 d_groups\n");
    printf("  Spills per kernel: %d\n", D / 4);
    printf("  Per-spill: 20 stores + 20 loads = 40 memory ops\n");
    printf("  Expected overhead: ~%d cycles\n", (D / 4) * 40);

    free(K); free(Q_A); free(Q_B); free(S_A); free(S_B);
    return 0;
}
