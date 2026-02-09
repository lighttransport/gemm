// FP16 12×2 NOEPI 4K-unrolled ACCUM — fp16 C, no conversion, 4K per iteration
// K must be multiple of 4.
// Args: x0=A(fp16), x1=B(fp16), x2=C(fp16), x3=K, x4=unused, x5=ldc_bytes

    .arch armv8.2-a+sve
    .text
    .align 4
    .global micro_kernel_fp16_12x2_noepi4_accum
    .type micro_kernel_fp16_12x2_noepi4_accum, %function

micro_kernel_fp16_12x2_noepi4_accum:
    stp d8, d9, [sp, #-64]!
    stp d10, d11, [sp, #16]
    stp d12, d13, [sp, #32]
    stp d14, d15, [sp, #48]

    ptrue p0.h

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

    lsr x6, x3, #2

.Lne4a_loop:
    // ═══ K=0 ═══
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
    ld1h {z24.h}, p0/z, [x1, #2, mul vl]
    ld1h {z25.h}, p0/z, [x1, #3, mul vl]

    // ═══ K=1 ═══
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
    ld1h {z24.h}, p0/z, [x1, #4, mul vl]
    ld1h {z25.h}, p0/z, [x1, #5, mul vl]

    // ═══ K=2 ═══
    ld1rh {z26.h}, p0/z, [x0, #48]
    ld1rh {z27.h}, p0/z, [x0, #50]
    ld1rh {z28.h}, p0/z, [x0, #52]
    ld1rh {z29.h}, p0/z, [x0, #54]
    ld1rh {z30.h}, p0/z, [x0, #56]
    ld1rh {z31.h}, p0/z, [x0, #58]
    fmla z0.h, p0/m, z26.h, z24.h
    fmla z1.h, p0/m, z26.h, z25.h
    ld1rh {z26.h}, p0/z, [x0, #60]
    fmla z2.h, p0/m, z27.h, z24.h
    fmla z3.h, p0/m, z27.h, z25.h
    ld1rh {z27.h}, p0/z, [x0, #62]
    fmla z4.h, p0/m, z28.h, z24.h
    fmla z5.h, p0/m, z28.h, z25.h
    ld1rh {z28.h}, p0/z, [x0, #64]
    fmla z6.h, p0/m, z29.h, z24.h
    fmla z7.h, p0/m, z29.h, z25.h
    ld1rh {z29.h}, p0/z, [x0, #66]
    fmla z8.h, p0/m, z30.h, z24.h
    fmla z9.h, p0/m, z30.h, z25.h
    ld1rh {z30.h}, p0/z, [x0, #68]
    fmla z10.h, p0/m, z31.h, z24.h
    fmla z11.h, p0/m, z31.h, z25.h
    ld1rh {z31.h}, p0/z, [x0, #70]
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
    ld1h {z24.h}, p0/z, [x1, #6, mul vl]
    ld1h {z25.h}, p0/z, [x1, #7, mul vl]

    // ═══ K=3 ═══
    ld1rh {z26.h}, p0/z, [x0, #72]
    ld1rh {z27.h}, p0/z, [x0, #74]
    ld1rh {z28.h}, p0/z, [x0, #76]
    ld1rh {z29.h}, p0/z, [x0, #78]
    ld1rh {z30.h}, p0/z, [x0, #80]
    ld1rh {z31.h}, p0/z, [x0, #82]
    fmla z0.h, p0/m, z26.h, z24.h
    fmla z1.h, p0/m, z26.h, z25.h
    ld1rh {z26.h}, p0/z, [x0, #84]
    fmla z2.h, p0/m, z27.h, z24.h
    fmla z3.h, p0/m, z27.h, z25.h
    ld1rh {z27.h}, p0/z, [x0, #86]
    fmla z4.h, p0/m, z28.h, z24.h
    fmla z5.h, p0/m, z28.h, z25.h
    ld1rh {z28.h}, p0/z, [x0, #88]
    fmla z6.h, p0/m, z29.h, z24.h
    fmla z7.h, p0/m, z29.h, z25.h
    ld1rh {z29.h}, p0/z, [x0, #90]
    fmla z8.h, p0/m, z30.h, z24.h
    fmla z9.h, p0/m, z30.h, z25.h
    ld1rh {z30.h}, p0/z, [x0, #92]
    fmla z10.h, p0/m, z31.h, z24.h
    fmla z11.h, p0/m, z31.h, z25.h
    ld1rh {z31.h}, p0/z, [x0, #94]
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

    add x0, x0, #96
    add x1, x1, #512

    subs x6, x6, #1
    beq .Lne4a_store

    ld1h {z24.h}, p0/z, [x1]
    ld1h {z25.h}, p0/z, [x1, #1, mul vl]
    b .Lne4a_loop

.Lne4a_store:
    // Load existing fp16 C, add, store
    ld1h {z24.h}, p0/z, [x2]
    ld1h {z25.h}, p0/z, [x2, #1, mul vl]
    fadd z0.h, z0.h, z24.h
    fadd z1.h, z1.h, z25.h
    st1h {z0.h}, p0, [x2]
    st1h {z1.h}, p0, [x2, #1, mul vl]
    add x2, x2, x5

    ld1h {z24.h}, p0/z, [x2]
    ld1h {z25.h}, p0/z, [x2, #1, mul vl]
    fadd z2.h, z2.h, z24.h
    fadd z3.h, z3.h, z25.h
    st1h {z2.h}, p0, [x2]
    st1h {z3.h}, p0, [x2, #1, mul vl]
    add x2, x2, x5

    ld1h {z24.h}, p0/z, [x2]
    ld1h {z25.h}, p0/z, [x2, #1, mul vl]
    fadd z4.h, z4.h, z24.h
    fadd z5.h, z5.h, z25.h
    st1h {z4.h}, p0, [x2]
    st1h {z5.h}, p0, [x2, #1, mul vl]
    add x2, x2, x5

    ld1h {z24.h}, p0/z, [x2]
    ld1h {z25.h}, p0/z, [x2, #1, mul vl]
    fadd z6.h, z6.h, z24.h
    fadd z7.h, z7.h, z25.h
    st1h {z6.h}, p0, [x2]
    st1h {z7.h}, p0, [x2, #1, mul vl]
    add x2, x2, x5

    ld1h {z24.h}, p0/z, [x2]
    ld1h {z25.h}, p0/z, [x2, #1, mul vl]
    fadd z8.h, z8.h, z24.h
    fadd z9.h, z9.h, z25.h
    st1h {z8.h}, p0, [x2]
    st1h {z9.h}, p0, [x2, #1, mul vl]
    add x2, x2, x5

    ld1h {z24.h}, p0/z, [x2]
    ld1h {z25.h}, p0/z, [x2, #1, mul vl]
    fadd z10.h, z10.h, z24.h
    fadd z11.h, z11.h, z25.h
    st1h {z10.h}, p0, [x2]
    st1h {z11.h}, p0, [x2, #1, mul vl]
    add x2, x2, x5

    ld1h {z24.h}, p0/z, [x2]
    ld1h {z25.h}, p0/z, [x2, #1, mul vl]
    fadd z12.h, z12.h, z24.h
    fadd z13.h, z13.h, z25.h
    st1h {z12.h}, p0, [x2]
    st1h {z13.h}, p0, [x2, #1, mul vl]
    add x2, x2, x5

    ld1h {z24.h}, p0/z, [x2]
    ld1h {z25.h}, p0/z, [x2, #1, mul vl]
    fadd z14.h, z14.h, z24.h
    fadd z15.h, z15.h, z25.h
    st1h {z14.h}, p0, [x2]
    st1h {z15.h}, p0, [x2, #1, mul vl]
    add x2, x2, x5

    ld1h {z24.h}, p0/z, [x2]
    ld1h {z25.h}, p0/z, [x2, #1, mul vl]
    fadd z16.h, z16.h, z24.h
    fadd z17.h, z17.h, z25.h
    st1h {z16.h}, p0, [x2]
    st1h {z17.h}, p0, [x2, #1, mul vl]
    add x2, x2, x5

    ld1h {z24.h}, p0/z, [x2]
    ld1h {z25.h}, p0/z, [x2, #1, mul vl]
    fadd z18.h, z18.h, z24.h
    fadd z19.h, z19.h, z25.h
    st1h {z18.h}, p0, [x2]
    st1h {z19.h}, p0, [x2, #1, mul vl]
    add x2, x2, x5

    ld1h {z24.h}, p0/z, [x2]
    ld1h {z25.h}, p0/z, [x2, #1, mul vl]
    fadd z20.h, z20.h, z24.h
    fadd z21.h, z21.h, z25.h
    st1h {z20.h}, p0, [x2]
    st1h {z21.h}, p0, [x2, #1, mul vl]
    add x2, x2, x5

    ld1h {z24.h}, p0/z, [x2]
    ld1h {z25.h}, p0/z, [x2, #1, mul vl]
    fadd z22.h, z22.h, z24.h
    fadd z23.h, z23.h, z25.h
    st1h {z22.h}, p0, [x2]
    st1h {z23.h}, p0, [x2, #1, mul vl]

    ldp d14, d15, [sp, #48]
    ldp d12, d13, [sp, #32]
    ldp d10, d11, [sp, #16]
    ldp d8, d9, [sp], #64
    ret
    .size micro_kernel_fp16_12x2_noepi4_accum, .-micro_kernel_fp16_12x2_noepi4_accum
