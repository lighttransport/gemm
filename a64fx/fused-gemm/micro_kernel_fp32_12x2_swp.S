// FP32 12×2 Software-Pipelined Micro-kernel
// Key optimization: batch-2 A loads interleaved with batch-1 FMAs.
// Each A register is reloaded immediately after its last use in batch 1,
// giving maximum load-to-use distance for batch 2.
//
// Original: 6 ld1rw → 12 FMLA → 6 ld1rw → 12 FMLA (load-to-use: ~0 cycles for batch 2)
// SWP:      6 ld1rw → [2 FMLA + 1 ld1rw] × 6 → 12 FMLA (load-to-use: ~10-18 insns)
//
// Registers:
//   Z0-Z23:  24 accumulators (12 rows × 2 cols)
//   Z24-Z25: B vectors (2 col vectors)
//   Z26-Z31: A temps (6 regs, shared between batches)
//
// Interface: x0=A[K][12], x1=B[K][32], x2=C, x3=K(even), x5=ldc_bytes

    .arch armv8.2-a+sve
    .text
    .align 4
    .global micro_kernel_fp32_12x2_swp
    .type micro_kernel_fp32_12x2_swp, %function

micro_kernel_fp32_12x2_swp:
    stp d8, d9, [sp, #-64]!
    stp d10, d11, [sp, #16]
    stp d12, d13, [sp, #32]
    stp d14, d15, [sp, #48]

    ptrue p0.s

    ld1w {z24.s}, p0/z, [x1]
    ld1w {z25.s}, p0/z, [x1, #1, mul vl]

    // Zero 24 accumulators
    eor z0.d, z0.d, z0.d
    eor z1.d, z1.d, z1.d
    eor z2.d, z2.d, z2.d
    eor z3.d, z3.d, z3.d
    eor z4.d, z4.d, z4.d
    eor z5.d, z5.d, z5.d
    eor z6.d, z6.d, z6.d
    eor z7.d, z7.d, z7.d
    eor z8.d, z8.d, z8.d
    eor z9.d, z9.d, z9.d
    eor z10.d, z10.d, z10.d
    eor z11.d, z11.d, z11.d
    eor z12.d, z12.d, z12.d
    eor z13.d, z13.d, z13.d
    eor z14.d, z14.d, z14.d
    eor z15.d, z15.d, z15.d
    eor z16.d, z16.d, z16.d
    eor z17.d, z17.d, z17.d
    eor z18.d, z18.d, z18.d
    eor z19.d, z19.d, z19.d
    eor z20.d, z20.d, z20.d
    eor z21.d, z21.d, z21.d
    eor z22.d, z22.d, z22.d
    eor z23.d, z23.d, z23.d

    lsr x6, x3, #1

.L12x2s_loop:
    // ═══ K iteration 0 ═══
    // Load batch 1: rows 0-5
    ld1rw {z26.s}, p0/z, [x0, #0]     // row 0
    ld1rw {z27.s}, p0/z, [x0, #4]     // row 1
    ld1rw {z28.s}, p0/z, [x0, #8]     // row 2
    ld1rw {z29.s}, p0/z, [x0, #12]    // row 3
    ld1rw {z30.s}, p0/z, [x0, #16]    // row 4
    ld1rw {z31.s}, p0/z, [x0, #20]    // row 5

    // Batch 1 FMAs interleaved with Batch 2 A loads
    fmla z0.s, p0/m, z26.s, z24.s     // row0 × col0
    fmla z1.s, p0/m, z26.s, z25.s     // row0 × col1 → z26 free
    ld1rw {z26.s}, p0/z, [x0, #24]    // reload z26 = row 6

    fmla z2.s, p0/m, z27.s, z24.s
    fmla z3.s, p0/m, z27.s, z25.s     // → z27 free
    ld1rw {z27.s}, p0/z, [x0, #28]    // row 7

    fmla z4.s, p0/m, z28.s, z24.s
    fmla z5.s, p0/m, z28.s, z25.s     // → z28 free
    ld1rw {z28.s}, p0/z, [x0, #32]    // row 8

    fmla z6.s, p0/m, z29.s, z24.s
    fmla z7.s, p0/m, z29.s, z25.s     // → z29 free
    ld1rw {z29.s}, p0/z, [x0, #36]    // row 9

    fmla z8.s, p0/m, z30.s, z24.s
    fmla z9.s, p0/m, z30.s, z25.s     // → z30 free
    ld1rw {z30.s}, p0/z, [x0, #40]    // row 10

    fmla z10.s, p0/m, z31.s, z24.s
    fmla z11.s, p0/m, z31.s, z25.s    // → z31 free
    ld1rw {z31.s}, p0/z, [x0, #44]    // row 11

    // Batch 2 FMAs: z26-z31 now hold rows 6-11
    // z26 loaded 18 insns ago, z31 loaded 1 insn ago
    fmla z12.s, p0/m, z26.s, z24.s
    fmla z13.s, p0/m, z26.s, z25.s
    fmla z14.s, p0/m, z27.s, z24.s
    fmla z15.s, p0/m, z27.s, z25.s
    fmla z16.s, p0/m, z28.s, z24.s
    fmla z17.s, p0/m, z28.s, z25.s
    fmla z18.s, p0/m, z29.s, z24.s
    fmla z19.s, p0/m, z29.s, z25.s
    fmla z20.s, p0/m, z30.s, z24.s
    fmla z21.s, p0/m, z30.s, z25.s
    fmla z22.s, p0/m, z31.s, z24.s
    fmla z23.s, p0/m, z31.s, z25.s

    // Load B(k+1)
    ld1w {z24.s}, p0/z, [x1, #2, mul vl]
    ld1w {z25.s}, p0/z, [x1, #3, mul vl]

    // ═══ K iteration 1 ═══
    // Load batch 1: rows 0-5
    ld1rw {z26.s}, p0/z, [x0, #48]
    ld1rw {z27.s}, p0/z, [x0, #52]
    ld1rw {z28.s}, p0/z, [x0, #56]
    ld1rw {z29.s}, p0/z, [x0, #60]
    ld1rw {z30.s}, p0/z, [x0, #64]
    ld1rw {z31.s}, p0/z, [x0, #68]

    // Batch 1 FMAs interleaved with Batch 2 A loads
    fmla z0.s, p0/m, z26.s, z24.s
    fmla z1.s, p0/m, z26.s, z25.s
    ld1rw {z26.s}, p0/z, [x0, #72]    // row 6

    fmla z2.s, p0/m, z27.s, z24.s
    fmla z3.s, p0/m, z27.s, z25.s
    ld1rw {z27.s}, p0/z, [x0, #76]    // row 7

    fmla z4.s, p0/m, z28.s, z24.s
    fmla z5.s, p0/m, z28.s, z25.s
    ld1rw {z28.s}, p0/z, [x0, #80]    // row 8

    fmla z6.s, p0/m, z29.s, z24.s
    fmla z7.s, p0/m, z29.s, z25.s
    ld1rw {z29.s}, p0/z, [x0, #84]    // row 9

    fmla z8.s, p0/m, z30.s, z24.s
    fmla z9.s, p0/m, z30.s, z25.s
    ld1rw {z30.s}, p0/z, [x0, #88]    // row 10

    fmla z10.s, p0/m, z31.s, z24.s
    fmla z11.s, p0/m, z31.s, z25.s
    ld1rw {z31.s}, p0/z, [x0, #92]    // row 11

    // Batch 2 FMAs
    fmla z12.s, p0/m, z26.s, z24.s
    fmla z13.s, p0/m, z26.s, z25.s
    fmla z14.s, p0/m, z27.s, z24.s
    fmla z15.s, p0/m, z27.s, z25.s
    fmla z16.s, p0/m, z28.s, z24.s
    fmla z17.s, p0/m, z28.s, z25.s
    fmla z18.s, p0/m, z29.s, z24.s
    fmla z19.s, p0/m, z29.s, z25.s
    fmla z20.s, p0/m, z30.s, z24.s
    fmla z21.s, p0/m, z30.s, z25.s
    fmla z22.s, p0/m, z31.s, z24.s
    fmla z23.s, p0/m, z31.s, z25.s

    add x0, x0, #96     // A: 2 × 12 × 4
    add x1, x1, #256    // B: 2 × 2 × 64

    subs x6, x6, #1
    beq .L12x2s_store

    ld1w {z24.s}, p0/z, [x1]
    ld1w {z25.s}, p0/z, [x1, #1, mul vl]
    b .L12x2s_loop

.L12x2s_store:
    st1w {z0.s}, p0, [x2]
    st1w {z1.s}, p0, [x2, #1, mul vl]
    add x2, x2, x5
    st1w {z2.s}, p0, [x2]
    st1w {z3.s}, p0, [x2, #1, mul vl]
    add x2, x2, x5
    st1w {z4.s}, p0, [x2]
    st1w {z5.s}, p0, [x2, #1, mul vl]
    add x2, x2, x5
    st1w {z6.s}, p0, [x2]
    st1w {z7.s}, p0, [x2, #1, mul vl]
    add x2, x2, x5
    st1w {z8.s}, p0, [x2]
    st1w {z9.s}, p0, [x2, #1, mul vl]
    add x2, x2, x5
    st1w {z10.s}, p0, [x2]
    st1w {z11.s}, p0, [x2, #1, mul vl]
    add x2, x2, x5
    st1w {z12.s}, p0, [x2]
    st1w {z13.s}, p0, [x2, #1, mul vl]
    add x2, x2, x5
    st1w {z14.s}, p0, [x2]
    st1w {z15.s}, p0, [x2, #1, mul vl]
    add x2, x2, x5
    st1w {z16.s}, p0, [x2]
    st1w {z17.s}, p0, [x2, #1, mul vl]
    add x2, x2, x5
    st1w {z18.s}, p0, [x2]
    st1w {z19.s}, p0, [x2, #1, mul vl]
    add x2, x2, x5
    st1w {z20.s}, p0, [x2]
    st1w {z21.s}, p0, [x2, #1, mul vl]
    add x2, x2, x5
    st1w {z22.s}, p0, [x2]
    st1w {z23.s}, p0, [x2, #1, mul vl]

    ldp d14, d15, [sp, #48]
    ldp d12, d13, [sp, #32]
    ldp d10, d11, [sp, #16]
    ldp d8, d9, [sp], #64
    ret
    .size micro_kernel_fp32_12x2_swp, .-micro_kernel_fp32_12x2_swp
