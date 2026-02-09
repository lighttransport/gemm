// FP32 6x4 broadcast microkernel - ACCUMULATE variant
// Same as micro_kernel_fp32_6x4_bcast but loads C into accumulators
// Used for K-blocking: subsequent Kc blocks accumulate into existing O

    .arch armv8.2-a+sve
    .text
    .align 4
    .global micro_kernel_fp32_6x4_bcast_accum
    .type micro_kernel_fp32_6x4_bcast_accum, %function

micro_kernel_fp32_6x4_bcast_accum:
    // Save callee-saved D8-D15
    stp d8, d9, [sp, #-64]!
    stp d10, d11, [sp, #16]
    stp d12, d13, [sp, #32]
    stp d14, d15, [sp, #48]

    ptrue p0.s

    // Load existing C into 24 accumulators (6 rows × 4 col vectors)
    // Row 0
    ld1w {z0.s}, p0/z, [x2]
    ld1w {z1.s}, p0/z, [x2, #1, mul vl]
    ld1w {z2.s}, p0/z, [x2, #2, mul vl]
    ld1w {z3.s}, p0/z, [x2, #3, mul vl]
    add x7, x2, x5
    // Row 1
    ld1w {z4.s}, p0/z, [x7]
    ld1w {z5.s}, p0/z, [x7, #1, mul vl]
    ld1w {z6.s}, p0/z, [x7, #2, mul vl]
    ld1w {z7.s}, p0/z, [x7, #3, mul vl]
    add x7, x7, x5
    // Row 2
    ld1w {z8.s}, p0/z, [x7]
    ld1w {z9.s}, p0/z, [x7, #1, mul vl]
    ld1w {z10.s}, p0/z, [x7, #2, mul vl]
    ld1w {z11.s}, p0/z, [x7, #3, mul vl]
    add x7, x7, x5
    // Row 3
    ld1w {z12.s}, p0/z, [x7]
    ld1w {z13.s}, p0/z, [x7, #1, mul vl]
    ld1w {z14.s}, p0/z, [x7, #2, mul vl]
    ld1w {z15.s}, p0/z, [x7, #3, mul vl]
    add x7, x7, x5
    // Row 4
    ld1w {z16.s}, p0/z, [x7]
    ld1w {z17.s}, p0/z, [x7, #1, mul vl]
    ld1w {z18.s}, p0/z, [x7, #2, mul vl]
    ld1w {z19.s}, p0/z, [x7, #3, mul vl]
    add x7, x7, x5
    // Row 5
    ld1w {z20.s}, p0/z, [x7]
    ld1w {z21.s}, p0/z, [x7, #1, mul vl]
    ld1w {z22.s}, p0/z, [x7, #2, mul vl]
    ld1w {z23.s}, p0/z, [x7, #3, mul vl]

    // Load initial B
    ld1w {z24.s}, p0/z, [x1]
    ld1w {z25.s}, p0/z, [x1, #1, mul vl]
    ld1w {z26.s}, p0/z, [x1, #2, mul vl]
    ld1w {z27.s}, p0/z, [x1, #3, mul vl]

    lsr x6, x3, #1

.L6x4a_loop:
    // K iteration 0
    ld1rw {z28.s}, p0/z, [x0, #0]
    ld1rw {z29.s}, p0/z, [x0, #4]
    ld1rw {z30.s}, p0/z, [x0, #8]
    ld1rw {z31.s}, p0/z, [x0, #12]

    fmla z0.s, p0/m, z28.s, z24.s
    fmla z1.s, p0/m, z28.s, z25.s
    fmla z2.s, p0/m, z28.s, z26.s
    fmla z3.s, p0/m, z28.s, z27.s

    fmla z4.s, p0/m, z29.s, z24.s
    fmla z5.s, p0/m, z29.s, z25.s
    fmla z6.s, p0/m, z29.s, z26.s
    fmla z7.s, p0/m, z29.s, z27.s

    fmla z8.s, p0/m, z30.s, z24.s
    fmla z9.s, p0/m, z30.s, z25.s
    fmla z10.s, p0/m, z30.s, z26.s
    fmla z11.s, p0/m, z30.s, z27.s

    fmla z12.s, p0/m, z31.s, z24.s
    fmla z13.s, p0/m, z31.s, z25.s
    fmla z14.s, p0/m, z31.s, z26.s
    fmla z15.s, p0/m, z31.s, z27.s

    ld1rw {z28.s}, p0/z, [x0, #16]
    ld1rw {z29.s}, p0/z, [x0, #20]

    fmla z16.s, p0/m, z28.s, z24.s
    fmla z17.s, p0/m, z28.s, z25.s
    fmla z18.s, p0/m, z28.s, z26.s
    fmla z19.s, p0/m, z28.s, z27.s

    fmla z20.s, p0/m, z29.s, z24.s
    fmla z21.s, p0/m, z29.s, z25.s
    fmla z22.s, p0/m, z29.s, z26.s
    fmla z23.s, p0/m, z29.s, z27.s

    ld1w {z24.s}, p0/z, [x1, #4, mul vl]
    ld1w {z25.s}, p0/z, [x1, #5, mul vl]
    ld1w {z26.s}, p0/z, [x1, #6, mul vl]
    ld1w {z27.s}, p0/z, [x1, #7, mul vl]

    // K iteration 1
    ld1rw {z28.s}, p0/z, [x0, #24]
    ld1rw {z29.s}, p0/z, [x0, #28]
    ld1rw {z30.s}, p0/z, [x0, #32]
    ld1rw {z31.s}, p0/z, [x0, #36]

    fmla z0.s, p0/m, z28.s, z24.s
    fmla z1.s, p0/m, z28.s, z25.s
    fmla z2.s, p0/m, z28.s, z26.s
    fmla z3.s, p0/m, z28.s, z27.s

    fmla z4.s, p0/m, z29.s, z24.s
    fmla z5.s, p0/m, z29.s, z25.s
    fmla z6.s, p0/m, z29.s, z26.s
    fmla z7.s, p0/m, z29.s, z27.s

    fmla z8.s, p0/m, z30.s, z24.s
    fmla z9.s, p0/m, z30.s, z25.s
    fmla z10.s, p0/m, z30.s, z26.s
    fmla z11.s, p0/m, z30.s, z27.s

    fmla z12.s, p0/m, z31.s, z24.s
    fmla z13.s, p0/m, z31.s, z25.s
    fmla z14.s, p0/m, z31.s, z26.s
    fmla z15.s, p0/m, z31.s, z27.s

    ld1rw {z28.s}, p0/z, [x0, #40]
    ld1rw {z29.s}, p0/z, [x0, #44]

    fmla z16.s, p0/m, z28.s, z24.s
    fmla z17.s, p0/m, z28.s, z25.s
    fmla z18.s, p0/m, z28.s, z26.s
    fmla z19.s, p0/m, z28.s, z27.s

    fmla z20.s, p0/m, z29.s, z24.s
    fmla z21.s, p0/m, z29.s, z25.s
    fmla z22.s, p0/m, z29.s, z26.s
    fmla z23.s, p0/m, z29.s, z27.s

    add x0, x0, #48
    add x1, x1, #512

    subs x6, x6, #1
    beq .L6x4a_store

    ld1w {z24.s}, p0/z, [x1]
    ld1w {z25.s}, p0/z, [x1, #1, mul vl]
    ld1w {z26.s}, p0/z, [x1, #2, mul vl]
    ld1w {z27.s}, p0/z, [x1, #3, mul vl]

    b .L6x4a_loop

.L6x4a_store:
    st1w {z0.s}, p0, [x2]
    st1w {z1.s}, p0, [x2, #1, mul vl]
    st1w {z2.s}, p0, [x2, #2, mul vl]
    st1w {z3.s}, p0, [x2, #3, mul vl]
    add x2, x2, x5

    st1w {z4.s}, p0, [x2]
    st1w {z5.s}, p0, [x2, #1, mul vl]
    st1w {z6.s}, p0, [x2, #2, mul vl]
    st1w {z7.s}, p0, [x2, #3, mul vl]
    add x2, x2, x5

    st1w {z8.s}, p0, [x2]
    st1w {z9.s}, p0, [x2, #1, mul vl]
    st1w {z10.s}, p0, [x2, #2, mul vl]
    st1w {z11.s}, p0, [x2, #3, mul vl]
    add x2, x2, x5

    st1w {z12.s}, p0, [x2]
    st1w {z13.s}, p0, [x2, #1, mul vl]
    st1w {z14.s}, p0, [x2, #2, mul vl]
    st1w {z15.s}, p0, [x2, #3, mul vl]
    add x2, x2, x5

    st1w {z16.s}, p0, [x2]
    st1w {z17.s}, p0, [x2, #1, mul vl]
    st1w {z18.s}, p0, [x2, #2, mul vl]
    st1w {z19.s}, p0, [x2, #3, mul vl]
    add x2, x2, x5

    st1w {z20.s}, p0, [x2]
    st1w {z21.s}, p0, [x2, #1, mul vl]
    st1w {z22.s}, p0, [x2, #2, mul vl]
    st1w {z23.s}, p0, [x2, #3, mul vl]

    // Restore callee-saved D8-D15
    ldp d14, d15, [sp, #48]
    ldp d12, d13, [sp, #32]
    ldp d10, d11, [sp, #16]
    ldp d8, d9, [sp], #64
    ret
    .size micro_kernel_fp32_6x4_bcast_accum, .-micro_kernel_fp32_6x4_bcast_accum
