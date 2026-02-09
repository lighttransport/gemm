// FP16 12×2 Dual-SWP INIT variant
// Key insight: BOTH batch-1 and batch-2 use SWP technique.
// - Batch-1 SWP: FMAs interleaved with batch-2 A loads (same as before)
// - Batch-2 SWP: FMAs interleaved with NEXT-K batch-1 A loads (NEW!)
// This eliminates the separate 6-ld1rh A-load phase and hides A load latency.
//
// Flow per K iteration (after first):
//   z26-z31 already hold rows 0-5 (pre-loaded from prev batch-2 SWP)
//   Batch-1 SWP: fmla(z0-z11) with z26-z31, reload z26-z31 as rows 6-11
//   Batch-2 SWP: fmla(z12-z23) with z26-z31, reload z26-z31 as next-K rows 0-5
//   B loads for next K
//
// First K: needs initial 6 ld1rh before entering the SWP pipeline.
//
// Args: x0=A(fp16), x1=B(fp16), x2=C(fp32), x3=K(even), x4=unused, x5=ldc_bytes

    .arch armv8.2-a+sve
    .text
    .align 4
    .global micro_kernel_fp16_12x2_dswp
    .type micro_kernel_fp16_12x2_dswp, %function

micro_kernel_fp16_12x2_dswp:
    stp d8, d9, [sp, #-64]!
    stp d10, d11, [sp, #16]
    stp d12, d13, [sp, #32]
    stp d14, d15, [sp, #48]

    ptrue p0.h
    ptrue p1.s

    ld1h {z24.h}, p0/z, [x1]
    ld1h {z25.h}, p0/z, [x1, #1, mul vl]

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

    lsr x6, x3, #1          // K/2 loop iterations

    // Load initial batch-1 A for K=0 (only needed for first iteration)
    ld1rh {z26.h}, p0/z, [x0, #0]
    ld1rh {z27.h}, p0/z, [x0, #2]
    ld1rh {z28.h}, p0/z, [x0, #4]
    ld1rh {z29.h}, p0/z, [x0, #6]
    ld1rh {z30.h}, p0/z, [x0, #8]
    ld1rh {z31.h}, p0/z, [x0, #10]

.Lh12x2d_loop:
    // ═══ K iteration 0 ═══
    // z26-z31 already hold rows 0-5 (pre-loaded)

    // Batch-1 SWP: FMAs with rows 0-5, reload as rows 6-11
    fmla z0.h, p0/m, z26.h, z24.h
    fmla z1.h, p0/m, z26.h, z25.h
    ld1rh {z26.h}, p0/z, [x0, #12]    // row 6

    fmla z2.h, p0/m, z27.h, z24.h
    fmla z3.h, p0/m, z27.h, z25.h
    ld1rh {z27.h}, p0/z, [x0, #14]    // row 7

    fmla z4.h, p0/m, z28.h, z24.h
    fmla z5.h, p0/m, z28.h, z25.h
    ld1rh {z28.h}, p0/z, [x0, #16]    // row 8

    fmla z6.h, p0/m, z29.h, z24.h
    fmla z7.h, p0/m, z29.h, z25.h
    ld1rh {z29.h}, p0/z, [x0, #18]    // row 9

    fmla z8.h, p0/m, z30.h, z24.h
    fmla z9.h, p0/m, z30.h, z25.h
    ld1rh {z30.h}, p0/z, [x0, #20]    // row 10

    fmla z10.h, p0/m, z31.h, z24.h
    fmla z11.h, p0/m, z31.h, z25.h
    ld1rh {z31.h}, p0/z, [x0, #22]    // row 11

    // Batch-2 SWP: FMAs with rows 6-11, reload as K=1 rows 0-5
    fmla z12.h, p0/m, z26.h, z24.h
    fmla z13.h, p0/m, z26.h, z25.h
    ld1rh {z26.h}, p0/z, [x0, #24]    // next K, row 0

    fmla z14.h, p0/m, z27.h, z24.h
    fmla z15.h, p0/m, z27.h, z25.h
    ld1rh {z27.h}, p0/z, [x0, #26]    // next K, row 1

    fmla z16.h, p0/m, z28.h, z24.h
    fmla z17.h, p0/m, z28.h, z25.h
    ld1rh {z28.h}, p0/z, [x0, #28]    // next K, row 2

    fmla z18.h, p0/m, z29.h, z24.h
    fmla z19.h, p0/m, z29.h, z25.h
    ld1rh {z29.h}, p0/z, [x0, #30]    // next K, row 3

    fmla z20.h, p0/m, z30.h, z24.h
    fmla z21.h, p0/m, z30.h, z25.h
    ld1rh {z30.h}, p0/z, [x0, #32]    // next K, row 4

    fmla z22.h, p0/m, z31.h, z24.h
    fmla z23.h, p0/m, z31.h, z25.h
    ld1rh {z31.h}, p0/z, [x0, #34]    // next K, row 5

    // B loads for K=1
    ld1h {z24.h}, p0/z, [x1, #2, mul vl]
    ld1h {z25.h}, p0/z, [x1, #3, mul vl]

    // ═══ K iteration 1 ═══
    // z26-z31 already hold K=1 rows 0-5 (pre-loaded in batch-2 SWP above)

    // Batch-1 SWP
    fmla z0.h, p0/m, z26.h, z24.h
    fmla z1.h, p0/m, z26.h, z25.h
    ld1rh {z26.h}, p0/z, [x0, #36]    // K=1 row 6

    fmla z2.h, p0/m, z27.h, z24.h
    fmla z3.h, p0/m, z27.h, z25.h
    ld1rh {z27.h}, p0/z, [x0, #38]    // K=1 row 7

    fmla z4.h, p0/m, z28.h, z24.h
    fmla z5.h, p0/m, z28.h, z25.h
    ld1rh {z28.h}, p0/z, [x0, #40]    // K=1 row 8

    fmla z6.h, p0/m, z29.h, z24.h
    fmla z7.h, p0/m, z29.h, z25.h
    ld1rh {z29.h}, p0/z, [x0, #42]    // K=1 row 9

    fmla z8.h, p0/m, z30.h, z24.h
    fmla z9.h, p0/m, z30.h, z25.h
    ld1rh {z30.h}, p0/z, [x0, #44]    // K=1 row 10

    fmla z10.h, p0/m, z31.h, z24.h
    fmla z11.h, p0/m, z31.h, z25.h
    ld1rh {z31.h}, p0/z, [x0, #46]    // K=1 row 11

    // Batch-2 SWP: reload as next-loop K=0 rows 0-5
    // Note: after add x0, offsets for next K=0 are 0,2,4,6,8,10
    // We must use the UPDATED x0, so we do the add FIRST
    add x0, x0, #48      // advance A pointer by 2K × 12 × 2 bytes

    fmla z12.h, p0/m, z26.h, z24.h
    fmla z13.h, p0/m, z26.h, z25.h
    ld1rh {z26.h}, p0/z, [x0, #0]     // next loop K=0 row 0

    fmla z14.h, p0/m, z27.h, z24.h
    fmla z15.h, p0/m, z27.h, z25.h
    ld1rh {z27.h}, p0/z, [x0, #2]     // row 1

    fmla z16.h, p0/m, z28.h, z24.h
    fmla z17.h, p0/m, z28.h, z25.h
    ld1rh {z28.h}, p0/z, [x0, #4]     // row 2

    fmla z18.h, p0/m, z29.h, z24.h
    fmla z19.h, p0/m, z29.h, z25.h
    ld1rh {z29.h}, p0/z, [x0, #6]     // row 3

    fmla z20.h, p0/m, z30.h, z24.h
    fmla z21.h, p0/m, z30.h, z25.h
    ld1rh {z30.h}, p0/z, [x0, #8]     // row 4

    fmla z22.h, p0/m, z31.h, z24.h
    fmla z23.h, p0/m, z31.h, z25.h
    ld1rh {z31.h}, p0/z, [x0, #10]    // row 5

    add x1, x1, #256     // B: 2 × 2 × 64 bytes

    subs x6, x6, #1
    beq .Lh12x2d_store

    // B loads for next loop iteration K=0
    // z26-z31 already hold next K=0 rows 0-5
    ld1h {z24.h}, p0/z, [x1]
    ld1h {z25.h}, p0/z, [x1, #1, mul vl]
    b .Lh12x2d_loop

.Lh12x2d_store:
    // ── Convert fp16 accumulators → fp32 and store ──
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
    .size micro_kernel_fp16_12x2_dswp, .-micro_kernel_fp16_12x2_dswp
