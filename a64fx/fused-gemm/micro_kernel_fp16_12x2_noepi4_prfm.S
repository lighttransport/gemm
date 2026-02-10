// FP16 12x2 NOEPI 4K-unrolled INIT with PRFM — fp16 C, no conversion, 4K per iteration
// Prefetches B data 2 iterations ahead during batch-2 FMAs (free decode slots)
//
// K must be multiple of 4.
// Args: x0=A(fp16), x1=B(fp16), x2=C(fp16), x3=K, x4=unused, x5=ldc_bytes

    .arch armv8.2-a+sve
    .text
    .align 4
    .global micro_kernel_fp16_12x2_noepi4_prfm
    .type micro_kernel_fp16_12x2_noepi4_prfm, %function

micro_kernel_fp16_12x2_noepi4_prfm:
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

.Lne4p_loop:
    // === K=0 (offsets 0-22) ===
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
    prfm pldl1keep, [x1, #1024]      // prefetch B 2 iterations ahead (line 0)
    fmla z18.h, p0/m, z29.h, z24.h
    fmla z19.h, p0/m, z29.h, z25.h
    fmla z20.h, p0/m, z30.h, z24.h
    fmla z21.h, p0/m, z30.h, z25.h
    fmla z22.h, p0/m, z31.h, z24.h
    fmla z23.h, p0/m, z31.h, z25.h

    ld1h {z24.h}, p0/z, [x1, #2, mul vl]
    ld1h {z25.h}, p0/z, [x1, #3, mul vl]

    // === K=1 (offsets 24-46) ===
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

    // === K=2 (offsets 48-70) ===
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
    prfm pldl1keep, [x1, #1280]      // prefetch B 2 iterations ahead (line 1)
    fmla z18.h, p0/m, z29.h, z24.h
    fmla z19.h, p0/m, z29.h, z25.h
    fmla z20.h, p0/m, z30.h, z24.h
    fmla z21.h, p0/m, z30.h, z25.h
    fmla z22.h, p0/m, z31.h, z24.h
    fmla z23.h, p0/m, z31.h, z25.h

    ld1h {z24.h}, p0/z, [x1, #6, mul vl]
    ld1h {z25.h}, p0/z, [x1, #7, mul vl]

    // === K=3 (offsets 72-94) ===
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
    beq .Lne4p_store

    ld1h {z24.h}, p0/z, [x1]
    ld1h {z25.h}, p0/z, [x1, #1, mul vl]
    b .Lne4p_loop

.Lne4p_store:
    st1h {z0.h}, p0, [x2]
    st1h {z1.h}, p0, [x2, #1, mul vl]
    add x2, x2, x5
    st1h {z2.h}, p0, [x2]
    st1h {z3.h}, p0, [x2, #1, mul vl]
    add x2, x2, x5
    st1h {z4.h}, p0, [x2]
    st1h {z5.h}, p0, [x2, #1, mul vl]
    add x2, x2, x5
    st1h {z6.h}, p0, [x2]
    st1h {z7.h}, p0, [x2, #1, mul vl]
    add x2, x2, x5
    st1h {z8.h}, p0, [x2]
    st1h {z9.h}, p0, [x2, #1, mul vl]
    add x2, x2, x5
    st1h {z10.h}, p0, [x2]
    st1h {z11.h}, p0, [x2, #1, mul vl]
    add x2, x2, x5
    st1h {z12.h}, p0, [x2]
    st1h {z13.h}, p0, [x2, #1, mul vl]
    add x2, x2, x5
    st1h {z14.h}, p0, [x2]
    st1h {z15.h}, p0, [x2, #1, mul vl]
    add x2, x2, x5
    st1h {z16.h}, p0, [x2]
    st1h {z17.h}, p0, [x2, #1, mul vl]
    add x2, x2, x5
    st1h {z18.h}, p0, [x2]
    st1h {z19.h}, p0, [x2, #1, mul vl]
    add x2, x2, x5
    st1h {z20.h}, p0, [x2]
    st1h {z21.h}, p0, [x2, #1, mul vl]
    add x2, x2, x5
    st1h {z22.h}, p0, [x2]
    st1h {z23.h}, p0, [x2, #1, mul vl]

    ldp d14, d15, [sp, #48]
    ldp d12, d13, [sp, #32]
    ldp d10, d11, [sp, #16]
    ldp d8, d9, [sp], #64
    ret
    .size micro_kernel_fp16_12x2_noepi4_prfm, .-micro_kernel_fp16_12x2_noepi4_prfm
