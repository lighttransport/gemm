// FP16 12×2 B-Preload INIT variant
// Pre-loads B(K=0) into z24/z25 and B(K=1) into z26/z27 at top of 2K block
// to eliminate FMA bubble at K=0→K=1 transition.
//
// Computes: C[12][64](fp32) = A[K][12](fp16) × B[K][64](fp16)
// NR = 64 fp16 = 2 SVE vectors of 32 fp16 each
//
// Registers:
//   z0-z23:  24 fp16 accumulators (12 rows × 2 col-groups)
//   z24-z25: B(K=0) vectors
//   z26-z27: B(K=1) vectors
//   z28-z31: A broadcast temps (4 regs, 3-batch SWP: 4-4-4 rows)
//   p0: ptrue .h (32 fp16 lanes)
//   p1: ptrue .s (16 fp32 lanes for store epilogue)
//
// Args: x0=A(fp16), x1=B(fp16), x2=C(fp32), x3=K(even), x4=unused, x5=ldc_bytes

    .arch armv8.2-a+sve
    .text
    .align 4
    .global micro_kernel_fp16_12x2_bpre
    .type micro_kernel_fp16_12x2_bpre, %function

micro_kernel_fp16_12x2_bpre:
    stp d8, d9, [sp, #-64]!
    stp d10, d11, [sp, #16]
    stp d12, d13, [sp, #32]
    stp d14, d15, [sp, #48]

    ptrue p0.h
    ptrue p1.s

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

    lsr x6, x3, #1          // x6 = K/2 loop iterations

.Lh12bp_loop:
    // ═══ Pre-load B for BOTH K steps ═══
    ld1h {z24.h}, p0/z, [x1]               // B(K=0) col 0
    ld1h {z25.h}, p0/z, [x1, #1, mul vl]   // B(K=0) col 1
    ld1h {z26.h}, p0/z, [x1, #2, mul vl]   // B(K=1) col 0
    ld1h {z27.h}, p0/z, [x1, #3, mul vl]   // B(K=1) col 1

    // ═══ K=0: B in z24/z25 ═══
    // Pre-load A rows 0-3
    ld1rh {z28.h}, p0/z, [x0, #0]     // A[0]
    ld1rh {z29.h}, p0/z, [x0, #2]     // A[1]
    ld1rh {z30.h}, p0/z, [x0, #4]     // A[2]
    ld1rh {z31.h}, p0/z, [x0, #6]     // A[3]

    // Batch 1: rows 0-3, SWP-load rows 4-7
    fmla z0.h, p0/m, z28.h, z24.h     // row 0, col 0
    fmla z1.h, p0/m, z28.h, z25.h     // row 0, col 1
    ld1rh {z28.h}, p0/z, [x0, #8]     // SWP: A[4]

    fmla z2.h, p0/m, z29.h, z24.h     // row 1, col 0
    fmla z3.h, p0/m, z29.h, z25.h     // row 1, col 1
    ld1rh {z29.h}, p0/z, [x0, #10]    // SWP: A[5]

    fmla z4.h, p0/m, z30.h, z24.h     // row 2, col 0
    fmla z5.h, p0/m, z30.h, z25.h     // row 2, col 1
    ld1rh {z30.h}, p0/z, [x0, #12]    // SWP: A[6]

    fmla z6.h, p0/m, z31.h, z24.h     // row 3, col 0
    fmla z7.h, p0/m, z31.h, z25.h     // row 3, col 1
    ld1rh {z31.h}, p0/z, [x0, #14]    // SWP: A[7]

    // Batch 2: rows 4-7, SWP-load rows 8-11
    fmla z8.h, p0/m, z28.h, z24.h     // row 4, col 0
    fmla z9.h, p0/m, z28.h, z25.h     // row 4, col 1
    ld1rh {z28.h}, p0/z, [x0, #16]    // SWP: A[8]

    fmla z10.h, p0/m, z29.h, z24.h    // row 5, col 0
    fmla z11.h, p0/m, z29.h, z25.h    // row 5, col 1
    ld1rh {z29.h}, p0/z, [x0, #18]    // SWP: A[9]

    fmla z12.h, p0/m, z30.h, z24.h    // row 6, col 0
    fmla z13.h, p0/m, z30.h, z25.h    // row 6, col 1
    ld1rh {z30.h}, p0/z, [x0, #20]    // SWP: A[10]

    fmla z14.h, p0/m, z31.h, z24.h    // row 7, col 0
    fmla z15.h, p0/m, z31.h, z25.h    // row 7, col 1
    ld1rh {z31.h}, p0/z, [x0, #22]    // SWP: A[11]

    // Batch 3: rows 8-11 (no SWP needed)
    fmla z16.h, p0/m, z28.h, z24.h    // row 8, col 0
    fmla z17.h, p0/m, z28.h, z25.h    // row 8, col 1
    fmla z18.h, p0/m, z29.h, z24.h    // row 9, col 0
    fmla z19.h, p0/m, z29.h, z25.h    // row 9, col 1
    fmla z20.h, p0/m, z30.h, z24.h    // row 10, col 0
    fmla z21.h, p0/m, z30.h, z25.h    // row 10, col 1
    fmla z22.h, p0/m, z31.h, z24.h    // row 11, col 0
    fmla z23.h, p0/m, z31.h, z25.h    // row 11, col 1

    // ═══ K=1: B in z26/z27 ═══
    // Pre-load A rows 0-3
    ld1rh {z28.h}, p0/z, [x0, #24]    // A[0] for K=1
    ld1rh {z29.h}, p0/z, [x0, #26]    // A[1]
    ld1rh {z30.h}, p0/z, [x0, #28]    // A[2]
    ld1rh {z31.h}, p0/z, [x0, #30]    // A[3]

    // Batch 1: rows 0-3, SWP-load rows 4-7
    fmla z0.h, p0/m, z28.h, z26.h     // row 0, col 0
    fmla z1.h, p0/m, z28.h, z27.h     // row 0, col 1
    ld1rh {z28.h}, p0/z, [x0, #32]    // SWP: A[4]

    fmla z2.h, p0/m, z29.h, z26.h     // row 1, col 0
    fmla z3.h, p0/m, z29.h, z27.h     // row 1, col 1
    ld1rh {z29.h}, p0/z, [x0, #34]    // SWP: A[5]

    fmla z4.h, p0/m, z30.h, z26.h     // row 2, col 0
    fmla z5.h, p0/m, z30.h, z27.h     // row 2, col 1
    ld1rh {z30.h}, p0/z, [x0, #36]    // SWP: A[6]

    fmla z6.h, p0/m, z31.h, z26.h     // row 3, col 0
    fmla z7.h, p0/m, z31.h, z27.h     // row 3, col 1
    ld1rh {z31.h}, p0/z, [x0, #38]    // SWP: A[7]

    // Batch 2: rows 4-7, SWP-load rows 8-11
    fmla z8.h, p0/m, z28.h, z26.h     // row 4, col 0
    fmla z9.h, p0/m, z28.h, z27.h     // row 4, col 1
    ld1rh {z28.h}, p0/z, [x0, #40]    // SWP: A[8]

    fmla z10.h, p0/m, z29.h, z26.h    // row 5, col 0
    fmla z11.h, p0/m, z29.h, z27.h    // row 5, col 1
    ld1rh {z29.h}, p0/z, [x0, #42]    // SWP: A[9]

    fmla z12.h, p0/m, z30.h, z26.h    // row 6, col 0
    fmla z13.h, p0/m, z30.h, z27.h    // row 6, col 1
    ld1rh {z30.h}, p0/z, [x0, #44]    // SWP: A[10]

    fmla z14.h, p0/m, z31.h, z26.h    // row 7, col 0
    fmla z15.h, p0/m, z31.h, z27.h    // row 7, col 1
    ld1rh {z31.h}, p0/z, [x0, #46]    // SWP: A[11]

    // Batch 3: rows 8-11 (no SWP needed)
    fmla z16.h, p0/m, z28.h, z26.h    // row 8, col 0
    fmla z17.h, p0/m, z28.h, z27.h    // row 8, col 1
    fmla z18.h, p0/m, z29.h, z26.h    // row 9, col 0
    fmla z19.h, p0/m, z29.h, z27.h    // row 9, col 1
    fmla z20.h, p0/m, z30.h, z26.h    // row 10, col 0
    fmla z21.h, p0/m, z30.h, z27.h    // row 10, col 1
    fmla z22.h, p0/m, z31.h, z26.h    // row 11, col 0
    fmla z23.h, p0/m, z31.h, z27.h    // row 11, col 1

    // Advance pointers
    add x0, x0, #48          // A: 2K × 12 × 2 bytes
    add x1, x1, #256         // B: 2K × 2 vecs × 64 bytes

    subs x6, x6, #1
    bne .Lh12bp_loop

.Lh12bp_store:
    // ── Convert fp16 accumulators → fp32 and store ──
    // Each row: 2 fp16 regs → 4 fp32 SVE vecs via uunpklo/hi + fcvt

    // Row 0: z0, z1
    uunpklo z24.s, z0.h
    fcvt z24.s, p1/m, z24.h
    st1w {z24.s}, p1, [x2]
    uunpkhi z24.s, z0.h
    fcvt z24.s, p1/m, z24.h
    st1w {z24.s}, p1, [x2, #1, mul vl]
    uunpklo z24.s, z1.h
    fcvt z24.s, p1/m, z24.h
    st1w {z24.s}, p1, [x2, #2, mul vl]
    uunpkhi z24.s, z1.h
    fcvt z24.s, p1/m, z24.h
    st1w {z24.s}, p1, [x2, #3, mul vl]
    add x2, x2, x5

    // Row 1: z2, z3
    uunpklo z24.s, z2.h
    fcvt z24.s, p1/m, z24.h
    st1w {z24.s}, p1, [x2]
    uunpkhi z24.s, z2.h
    fcvt z24.s, p1/m, z24.h
    st1w {z24.s}, p1, [x2, #1, mul vl]
    uunpklo z24.s, z3.h
    fcvt z24.s, p1/m, z24.h
    st1w {z24.s}, p1, [x2, #2, mul vl]
    uunpkhi z24.s, z3.h
    fcvt z24.s, p1/m, z24.h
    st1w {z24.s}, p1, [x2, #3, mul vl]
    add x2, x2, x5

    // Row 2: z4, z5
    uunpklo z24.s, z4.h
    fcvt z24.s, p1/m, z24.h
    st1w {z24.s}, p1, [x2]
    uunpkhi z24.s, z4.h
    fcvt z24.s, p1/m, z24.h
    st1w {z24.s}, p1, [x2, #1, mul vl]
    uunpklo z24.s, z5.h
    fcvt z24.s, p1/m, z24.h
    st1w {z24.s}, p1, [x2, #2, mul vl]
    uunpkhi z24.s, z5.h
    fcvt z24.s, p1/m, z24.h
    st1w {z24.s}, p1, [x2, #3, mul vl]
    add x2, x2, x5

    // Row 3: z6, z7
    uunpklo z24.s, z6.h
    fcvt z24.s, p1/m, z24.h
    st1w {z24.s}, p1, [x2]
    uunpkhi z24.s, z6.h
    fcvt z24.s, p1/m, z24.h
    st1w {z24.s}, p1, [x2, #1, mul vl]
    uunpklo z24.s, z7.h
    fcvt z24.s, p1/m, z24.h
    st1w {z24.s}, p1, [x2, #2, mul vl]
    uunpkhi z24.s, z7.h
    fcvt z24.s, p1/m, z24.h
    st1w {z24.s}, p1, [x2, #3, mul vl]
    add x2, x2, x5

    // Row 4: z8, z9
    uunpklo z24.s, z8.h
    fcvt z24.s, p1/m, z24.h
    st1w {z24.s}, p1, [x2]
    uunpkhi z24.s, z8.h
    fcvt z24.s, p1/m, z24.h
    st1w {z24.s}, p1, [x2, #1, mul vl]
    uunpklo z24.s, z9.h
    fcvt z24.s, p1/m, z24.h
    st1w {z24.s}, p1, [x2, #2, mul vl]
    uunpkhi z24.s, z9.h
    fcvt z24.s, p1/m, z24.h
    st1w {z24.s}, p1, [x2, #3, mul vl]
    add x2, x2, x5

    // Row 5: z10, z11
    uunpklo z24.s, z10.h
    fcvt z24.s, p1/m, z24.h
    st1w {z24.s}, p1, [x2]
    uunpkhi z24.s, z10.h
    fcvt z24.s, p1/m, z24.h
    st1w {z24.s}, p1, [x2, #1, mul vl]
    uunpklo z24.s, z11.h
    fcvt z24.s, p1/m, z24.h
    st1w {z24.s}, p1, [x2, #2, mul vl]
    uunpkhi z24.s, z11.h
    fcvt z24.s, p1/m, z24.h
    st1w {z24.s}, p1, [x2, #3, mul vl]
    add x2, x2, x5

    // Row 6: z12, z13
    uunpklo z24.s, z12.h
    fcvt z24.s, p1/m, z24.h
    st1w {z24.s}, p1, [x2]
    uunpkhi z24.s, z12.h
    fcvt z24.s, p1/m, z24.h
    st1w {z24.s}, p1, [x2, #1, mul vl]
    uunpklo z24.s, z13.h
    fcvt z24.s, p1/m, z24.h
    st1w {z24.s}, p1, [x2, #2, mul vl]
    uunpkhi z24.s, z13.h
    fcvt z24.s, p1/m, z24.h
    st1w {z24.s}, p1, [x2, #3, mul vl]
    add x2, x2, x5

    // Row 7: z14, z15
    uunpklo z24.s, z14.h
    fcvt z24.s, p1/m, z24.h
    st1w {z24.s}, p1, [x2]
    uunpkhi z24.s, z14.h
    fcvt z24.s, p1/m, z24.h
    st1w {z24.s}, p1, [x2, #1, mul vl]
    uunpklo z24.s, z15.h
    fcvt z24.s, p1/m, z24.h
    st1w {z24.s}, p1, [x2, #2, mul vl]
    uunpkhi z24.s, z15.h
    fcvt z24.s, p1/m, z24.h
    st1w {z24.s}, p1, [x2, #3, mul vl]
    add x2, x2, x5

    // Row 8: z16, z17
    uunpklo z24.s, z16.h
    fcvt z24.s, p1/m, z24.h
    st1w {z24.s}, p1, [x2]
    uunpkhi z24.s, z16.h
    fcvt z24.s, p1/m, z24.h
    st1w {z24.s}, p1, [x2, #1, mul vl]
    uunpklo z24.s, z17.h
    fcvt z24.s, p1/m, z24.h
    st1w {z24.s}, p1, [x2, #2, mul vl]
    uunpkhi z24.s, z17.h
    fcvt z24.s, p1/m, z24.h
    st1w {z24.s}, p1, [x2, #3, mul vl]
    add x2, x2, x5

    // Row 9: z18, z19
    uunpklo z24.s, z18.h
    fcvt z24.s, p1/m, z24.h
    st1w {z24.s}, p1, [x2]
    uunpkhi z24.s, z18.h
    fcvt z24.s, p1/m, z24.h
    st1w {z24.s}, p1, [x2, #1, mul vl]
    uunpklo z24.s, z19.h
    fcvt z24.s, p1/m, z24.h
    st1w {z24.s}, p1, [x2, #2, mul vl]
    uunpkhi z24.s, z19.h
    fcvt z24.s, p1/m, z24.h
    st1w {z24.s}, p1, [x2, #3, mul vl]
    add x2, x2, x5

    // Row 10: z20, z21
    uunpklo z24.s, z20.h
    fcvt z24.s, p1/m, z24.h
    st1w {z24.s}, p1, [x2]
    uunpkhi z24.s, z20.h
    fcvt z24.s, p1/m, z24.h
    st1w {z24.s}, p1, [x2, #1, mul vl]
    uunpklo z24.s, z21.h
    fcvt z24.s, p1/m, z24.h
    st1w {z24.s}, p1, [x2, #2, mul vl]
    uunpkhi z24.s, z21.h
    fcvt z24.s, p1/m, z24.h
    st1w {z24.s}, p1, [x2, #3, mul vl]
    add x2, x2, x5

    // Row 11: z22, z23
    uunpklo z24.s, z22.h
    fcvt z24.s, p1/m, z24.h
    st1w {z24.s}, p1, [x2]
    uunpkhi z24.s, z22.h
    fcvt z24.s, p1/m, z24.h
    st1w {z24.s}, p1, [x2, #1, mul vl]
    uunpklo z24.s, z23.h
    fcvt z24.s, p1/m, z24.h
    st1w {z24.s}, p1, [x2, #2, mul vl]
    uunpkhi z24.s, z23.h
    fcvt z24.s, p1/m, z24.h
    st1w {z24.s}, p1, [x2, #3, mul vl]

    ldp d14, d15, [sp, #48]
    ldp d12, d13, [sp, #32]
    ldp d10, d11, [sp, #16]
    ldp d8, d9, [sp], #64
    ret
    .size micro_kernel_fp16_12x2_bpre, .-micro_kernel_fp16_12x2_bpre
