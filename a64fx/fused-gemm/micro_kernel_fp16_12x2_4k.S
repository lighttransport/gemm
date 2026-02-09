// FP16 12×2 SWP 4K-unrolled INIT variant
// 4 K-steps per loop iteration to reduce loop overhead (branch + ptr advance)
// Requires K to be a multiple of 4.
//
// B offsets per K: K=0 #0/#1, K=1 #2/#3, K=2 #4/#5, K=3 #6/#7 (all valid mul vl)
// A pointer bumped by #48 after K=1 and after K=3 (2K × MR=12 × 2 bytes each)
//
// Args: x0=A(fp16), x1=B(fp16), x2=C(fp32), x3=K(mult of 4), x4=unused, x5=ldc_bytes

    .arch armv8.2-a+sve
    .text
    .align 4
    .global micro_kernel_fp16_12x2_4k
    .type micro_kernel_fp16_12x2_4k, %function

micro_kernel_fp16_12x2_4k:
    stp d8, d9, [sp, #-64]!
    stp d10, d11, [sp, #16]
    stp d12, d13, [sp, #32]
    stp d14, d15, [sp, #48]

    ptrue p0.h
    ptrue p1.s

    ld1h {z24.h}, p0/z, [x1]
    ld1h {z25.h}, p0/z, [x1, #1, mul vl]

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

    lsr x6, x3, #2          // K/4 loop iterations

.Lh12x2_4k_loop:
    // ═══ K=0: B at [x1,#0], [x1,#1,mul vl] ═══
    ld1rh {z26.h}, p0/z, [x0, #0]
    ld1rh {z27.h}, p0/z, [x0, #2]
    ld1rh {z28.h}, p0/z, [x0, #4]
    ld1rh {z29.h}, p0/z, [x0, #6]
    ld1rh {z30.h}, p0/z, [x0, #8]
    ld1rh {z31.h}, p0/z, [x0, #10]

    fmla z0.h, p0/m, z26.h, z24.h
    fmla z1.h, p0/m, z26.h, z25.h
    ld1rh {z26.h}, p0/z, [x0, #12]
    fmla z2.h, p0/m, z27.h, z24.h
    fmla z3.h, p0/m, z27.h, z25.h
    ld1rh {z27.h}, p0/z, [x0, #14]
    fmla z4.h, p0/m, z28.h, z24.h
    fmla z5.h, p0/m, z28.h, z25.h
    ld1rh {z28.h}, p0/z, [x0, #16]
    fmla z6.h, p0/m, z29.h, z24.h
    fmla z7.h, p0/m, z29.h, z25.h
    ld1rh {z29.h}, p0/z, [x0, #18]
    fmla z8.h, p0/m, z30.h, z24.h
    fmla z9.h, p0/m, z30.h, z25.h
    ld1rh {z30.h}, p0/z, [x0, #20]
    fmla z10.h, p0/m, z31.h, z24.h
    fmla z11.h, p0/m, z31.h, z25.h
    ld1rh {z31.h}, p0/z, [x0, #22]

    fmla z12.h, p0/m, z26.h, z24.h
    fmla z13.h, p0/m, z26.h, z25.h
    fmla z14.h, p0/m, z27.h, z24.h
    fmla z15.h, p0/m, z27.h, z25.h
    fmla z16.h, p0/m, z28.h, z24.h
    fmla z17.h, p0/m, z28.h, z25.h
    fmla z18.h, p0/m, z29.h, z24.h
    fmla z19.h, p0/m, z29.h, z25.h
    fmla z20.h, p0/m, z30.h, z24.h
    fmla z21.h, p0/m, z30.h, z25.h
    fmla z22.h, p0/m, z31.h, z24.h
    fmla z23.h, p0/m, z31.h, z25.h

    // ═══ K=1: B at [x1,#2,mul vl], [x1,#3,mul vl] ═══
    ld1h {z24.h}, p0/z, [x1, #2, mul vl]
    ld1h {z25.h}, p0/z, [x1, #3, mul vl]

    ld1rh {z26.h}, p0/z, [x0, #24]
    ld1rh {z27.h}, p0/z, [x0, #26]
    ld1rh {z28.h}, p0/z, [x0, #28]
    ld1rh {z29.h}, p0/z, [x0, #30]
    ld1rh {z30.h}, p0/z, [x0, #32]
    ld1rh {z31.h}, p0/z, [x0, #34]

    fmla z0.h, p0/m, z26.h, z24.h
    fmla z1.h, p0/m, z26.h, z25.h
    ld1rh {z26.h}, p0/z, [x0, #36]
    fmla z2.h, p0/m, z27.h, z24.h
    fmla z3.h, p0/m, z27.h, z25.h
    ld1rh {z27.h}, p0/z, [x0, #38]
    fmla z4.h, p0/m, z28.h, z24.h
    fmla z5.h, p0/m, z28.h, z25.h
    ld1rh {z28.h}, p0/z, [x0, #40]
    fmla z6.h, p0/m, z29.h, z24.h
    fmla z7.h, p0/m, z29.h, z25.h
    ld1rh {z29.h}, p0/z, [x0, #42]
    fmla z8.h, p0/m, z30.h, z24.h
    fmla z9.h, p0/m, z30.h, z25.h
    ld1rh {z30.h}, p0/z, [x0, #44]
    fmla z10.h, p0/m, z31.h, z24.h
    fmla z11.h, p0/m, z31.h, z25.h
    ld1rh {z31.h}, p0/z, [x0, #46]

    fmla z12.h, p0/m, z26.h, z24.h
    fmla z13.h, p0/m, z26.h, z25.h
    fmla z14.h, p0/m, z27.h, z24.h
    fmla z15.h, p0/m, z27.h, z25.h
    fmla z16.h, p0/m, z28.h, z24.h
    fmla z17.h, p0/m, z28.h, z25.h
    fmla z18.h, p0/m, z29.h, z24.h
    fmla z19.h, p0/m, z29.h, z25.h
    fmla z20.h, p0/m, z30.h, z24.h
    fmla z21.h, p0/m, z30.h, z25.h
    fmla z22.h, p0/m, z31.h, z24.h
    fmla z23.h, p0/m, z31.h, z25.h

    add x0, x0, #48      // advance A past K=0,K=1

    // ═══ K=2: B at [x1,#4,mul vl], [x1,#5,mul vl] ═══
    ld1h {z24.h}, p0/z, [x1, #4, mul vl]
    ld1h {z25.h}, p0/z, [x1, #5, mul vl]

    ld1rh {z26.h}, p0/z, [x0, #0]
    ld1rh {z27.h}, p0/z, [x0, #2]
    ld1rh {z28.h}, p0/z, [x0, #4]
    ld1rh {z29.h}, p0/z, [x0, #6]
    ld1rh {z30.h}, p0/z, [x0, #8]
    ld1rh {z31.h}, p0/z, [x0, #10]

    fmla z0.h, p0/m, z26.h, z24.h
    fmla z1.h, p0/m, z26.h, z25.h
    ld1rh {z26.h}, p0/z, [x0, #12]
    fmla z2.h, p0/m, z27.h, z24.h
    fmla z3.h, p0/m, z27.h, z25.h
    ld1rh {z27.h}, p0/z, [x0, #14]
    fmla z4.h, p0/m, z28.h, z24.h
    fmla z5.h, p0/m, z28.h, z25.h
    ld1rh {z28.h}, p0/z, [x0, #16]
    fmla z6.h, p0/m, z29.h, z24.h
    fmla z7.h, p0/m, z29.h, z25.h
    ld1rh {z29.h}, p0/z, [x0, #18]
    fmla z8.h, p0/m, z30.h, z24.h
    fmla z9.h, p0/m, z30.h, z25.h
    ld1rh {z30.h}, p0/z, [x0, #20]
    fmla z10.h, p0/m, z31.h, z24.h
    fmla z11.h, p0/m, z31.h, z25.h
    ld1rh {z31.h}, p0/z, [x0, #22]

    fmla z12.h, p0/m, z26.h, z24.h
    fmla z13.h, p0/m, z26.h, z25.h
    fmla z14.h, p0/m, z27.h, z24.h
    fmla z15.h, p0/m, z27.h, z25.h
    fmla z16.h, p0/m, z28.h, z24.h
    fmla z17.h, p0/m, z28.h, z25.h
    fmla z18.h, p0/m, z29.h, z24.h
    fmla z19.h, p0/m, z29.h, z25.h
    fmla z20.h, p0/m, z30.h, z24.h
    fmla z21.h, p0/m, z30.h, z25.h
    fmla z22.h, p0/m, z31.h, z24.h
    fmla z23.h, p0/m, z31.h, z25.h

    // ═══ K=3: B at [x1,#6,mul vl], [x1,#7,mul vl] ═══
    ld1h {z24.h}, p0/z, [x1, #6, mul vl]
    ld1h {z25.h}, p0/z, [x1, #7, mul vl]

    ld1rh {z26.h}, p0/z, [x0, #24]
    ld1rh {z27.h}, p0/z, [x0, #26]
    ld1rh {z28.h}, p0/z, [x0, #28]
    ld1rh {z29.h}, p0/z, [x0, #30]
    ld1rh {z30.h}, p0/z, [x0, #32]
    ld1rh {z31.h}, p0/z, [x0, #34]

    fmla z0.h, p0/m, z26.h, z24.h
    fmla z1.h, p0/m, z26.h, z25.h
    ld1rh {z26.h}, p0/z, [x0, #36]
    fmla z2.h, p0/m, z27.h, z24.h
    fmla z3.h, p0/m, z27.h, z25.h
    ld1rh {z27.h}, p0/z, [x0, #38]
    fmla z4.h, p0/m, z28.h, z24.h
    fmla z5.h, p0/m, z28.h, z25.h
    ld1rh {z28.h}, p0/z, [x0, #40]
    fmla z6.h, p0/m, z29.h, z24.h
    fmla z7.h, p0/m, z29.h, z25.h
    ld1rh {z29.h}, p0/z, [x0, #42]
    fmla z8.h, p0/m, z30.h, z24.h
    fmla z9.h, p0/m, z30.h, z25.h
    ld1rh {z30.h}, p0/z, [x0, #44]
    fmla z10.h, p0/m, z31.h, z24.h
    fmla z11.h, p0/m, z31.h, z25.h
    ld1rh {z31.h}, p0/z, [x0, #46]

    fmla z12.h, p0/m, z26.h, z24.h
    fmla z13.h, p0/m, z26.h, z25.h
    fmla z14.h, p0/m, z27.h, z24.h
    fmla z15.h, p0/m, z27.h, z25.h
    fmla z16.h, p0/m, z28.h, z24.h
    fmla z17.h, p0/m, z28.h, z25.h
    fmla z18.h, p0/m, z29.h, z24.h
    fmla z19.h, p0/m, z29.h, z25.h
    fmla z20.h, p0/m, z30.h, z24.h
    fmla z21.h, p0/m, z30.h, z25.h
    fmla z22.h, p0/m, z31.h, z24.h
    fmla z23.h, p0/m, z31.h, z25.h

    add x0, x0, #48      // advance A past K=2,K=3
    add x1, x1, #512     // B: 4K × 2 vecs × 64 bytes

    subs x6, x6, #1
    beq .Lh12x2_4k_store

    ld1h {z24.h}, p0/z, [x1]
    ld1h {z25.h}, p0/z, [x1, #1, mul vl]
    b .Lh12x2_4k_loop

.Lh12x2_4k_store:
    // ── Convert fp16 accumulators → fp32 and store (same as 2K variant) ──
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
    .size micro_kernel_fp16_12x2_4k, .-micro_kernel_fp16_12x2_4k
