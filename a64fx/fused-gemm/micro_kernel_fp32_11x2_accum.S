// FP32 11×2 ACCUMULATE variant

    .arch armv8.2-a+sve
    .text
    .align 4
    .global micro_kernel_fp32_11x2_accum
    .type micro_kernel_fp32_11x2_accum, %function

micro_kernel_fp32_11x2_accum:
    stp d8, d9, [sp, #-64]!
    stp d10, d11, [sp, #16]
    stp d12, d13, [sp, #32]
    stp d14, d15, [sp, #48]

    ptrue p0.s

    // Load existing C into 22 accumulators
    ld1w {z0.s}, p0/z, [x2]
    ld1w {z1.s}, p0/z, [x2, #1, mul vl]
    add x7, x2, x5
    ld1w {z2.s}, p0/z, [x7]
    ld1w {z3.s}, p0/z, [x7, #1, mul vl]
    add x7, x7, x5
    ld1w {z4.s}, p0/z, [x7]
    ld1w {z5.s}, p0/z, [x7, #1, mul vl]
    add x7, x7, x5
    ld1w {z6.s}, p0/z, [x7]
    ld1w {z7.s}, p0/z, [x7, #1, mul vl]
    add x7, x7, x5
    ld1w {z8.s}, p0/z, [x7]
    ld1w {z9.s}, p0/z, [x7, #1, mul vl]
    add x7, x7, x5
    ld1w {z10.s}, p0/z, [x7]
    ld1w {z11.s}, p0/z, [x7, #1, mul vl]
    add x7, x7, x5
    ld1w {z12.s}, p0/z, [x7]
    ld1w {z13.s}, p0/z, [x7, #1, mul vl]
    add x7, x7, x5
    ld1w {z14.s}, p0/z, [x7]
    ld1w {z15.s}, p0/z, [x7, #1, mul vl]
    add x7, x7, x5
    ld1w {z16.s}, p0/z, [x7]
    ld1w {z17.s}, p0/z, [x7, #1, mul vl]
    add x7, x7, x5
    ld1w {z18.s}, p0/z, [x7]
    ld1w {z19.s}, p0/z, [x7, #1, mul vl]
    add x7, x7, x5
    ld1w {z20.s}, p0/z, [x7]
    ld1w {z21.s}, p0/z, [x7, #1, mul vl]

    // Load initial B
    ld1w {z22.s}, p0/z, [x1]
    ld1w {z23.s}, p0/z, [x1, #1, mul vl]

    lsr x6, x3, #1

.L11x2a_loop:
    // ═══ K iteration 0 ═══
    ld1rw {z24.s}, p0/z, [x0, #0]
    ld1rw {z25.s}, p0/z, [x0, #4]
    ld1rw {z26.s}, p0/z, [x0, #8]
    ld1rw {z27.s}, p0/z, [x0, #12]
    ld1rw {z28.s}, p0/z, [x0, #16]
    ld1rw {z29.s}, p0/z, [x0, #20]

    fmla z0.s, p0/m, z24.s, z22.s
    fmla z1.s, p0/m, z24.s, z23.s
    fmla z2.s, p0/m, z25.s, z22.s
    fmla z3.s, p0/m, z25.s, z23.s
    fmla z4.s, p0/m, z26.s, z22.s
    fmla z5.s, p0/m, z26.s, z23.s
    fmla z6.s, p0/m, z27.s, z22.s
    fmla z7.s, p0/m, z27.s, z23.s
    fmla z8.s, p0/m, z28.s, z22.s
    fmla z9.s, p0/m, z28.s, z23.s
    fmla z10.s, p0/m, z29.s, z22.s
    fmla z11.s, p0/m, z29.s, z23.s

    ld1rw {z24.s}, p0/z, [x0, #24]
    ld1rw {z25.s}, p0/z, [x0, #28]
    ld1rw {z26.s}, p0/z, [x0, #32]
    ld1rw {z27.s}, p0/z, [x0, #36]
    ld1rw {z28.s}, p0/z, [x0, #40]

    fmla z12.s, p0/m, z24.s, z22.s
    fmla z13.s, p0/m, z24.s, z23.s
    fmla z14.s, p0/m, z25.s, z22.s
    fmla z15.s, p0/m, z25.s, z23.s
    fmla z16.s, p0/m, z26.s, z22.s
    fmla z17.s, p0/m, z26.s, z23.s
    fmla z18.s, p0/m, z27.s, z22.s
    fmla z19.s, p0/m, z27.s, z23.s
    fmla z20.s, p0/m, z28.s, z22.s
    fmla z21.s, p0/m, z28.s, z23.s

    ld1w {z22.s}, p0/z, [x1, #2, mul vl]
    ld1w {z23.s}, p0/z, [x1, #3, mul vl]

    // ═══ K iteration 1 ═══
    ld1rw {z24.s}, p0/z, [x0, #44]
    ld1rw {z25.s}, p0/z, [x0, #48]
    ld1rw {z26.s}, p0/z, [x0, #52]
    ld1rw {z27.s}, p0/z, [x0, #56]
    ld1rw {z28.s}, p0/z, [x0, #60]
    ld1rw {z29.s}, p0/z, [x0, #64]

    fmla z0.s, p0/m, z24.s, z22.s
    fmla z1.s, p0/m, z24.s, z23.s
    fmla z2.s, p0/m, z25.s, z22.s
    fmla z3.s, p0/m, z25.s, z23.s
    fmla z4.s, p0/m, z26.s, z22.s
    fmla z5.s, p0/m, z26.s, z23.s
    fmla z6.s, p0/m, z27.s, z22.s
    fmla z7.s, p0/m, z27.s, z23.s
    fmla z8.s, p0/m, z28.s, z22.s
    fmla z9.s, p0/m, z28.s, z23.s
    fmla z10.s, p0/m, z29.s, z22.s
    fmla z11.s, p0/m, z29.s, z23.s

    ld1rw {z24.s}, p0/z, [x0, #68]
    ld1rw {z25.s}, p0/z, [x0, #72]
    ld1rw {z26.s}, p0/z, [x0, #76]
    ld1rw {z27.s}, p0/z, [x0, #80]
    ld1rw {z28.s}, p0/z, [x0, #84]

    fmla z12.s, p0/m, z24.s, z22.s
    fmla z13.s, p0/m, z24.s, z23.s
    fmla z14.s, p0/m, z25.s, z22.s
    fmla z15.s, p0/m, z25.s, z23.s
    fmla z16.s, p0/m, z26.s, z22.s
    fmla z17.s, p0/m, z26.s, z23.s
    fmla z18.s, p0/m, z27.s, z22.s
    fmla z19.s, p0/m, z27.s, z23.s
    fmla z20.s, p0/m, z28.s, z22.s
    fmla z21.s, p0/m, z28.s, z23.s

    add x0, x0, #88
    add x1, x1, #256

    subs x6, x6, #1
    beq .L11x2a_store

    ld1w {z22.s}, p0/z, [x1]
    ld1w {z23.s}, p0/z, [x1, #1, mul vl]
    b .L11x2a_loop

.L11x2a_store:
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

    ldp d14, d15, [sp, #48]
    ldp d12, d13, [sp, #32]
    ldp d10, d11, [sp, #16]
    ldp d8, d9, [sp], #64
    ret
    .size micro_kernel_fp32_11x2_accum, .-micro_kernel_fp32_11x2_accum
