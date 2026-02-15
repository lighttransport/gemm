// FP32 FMLA Micro-kernel: 10×2 tile, 2× K-unroll
// AI = 3.81, d=256 exact (no padding), L1 = 43.3 KB
//
// Registers:
//   Z0-Z19:  20 accumulators (10 rows × 2 cols)
//   Z20-Z21: B vectors (2 col vectors, 32 FP32)
//   Z22-Z26: A temps batch 1 (rows 0-4)
//   Z27-Z31: A temps batch 2 (rows 5-9)
//
// Interface: x0=A[K][10], x1=B[K][32], x2=C, x3=K(even), x5=ldc_bytes

    .arch armv8.2-a+sve
    .text
    .align 4
    .global micro_kernel_fp32_10x2
    .type micro_kernel_fp32_10x2, %function

micro_kernel_fp32_10x2:
    stp d8, d9, [sp, #-64]!
    stp d10, d11, [sp, #16]
    stp d12, d13, [sp, #32]
    stp d14, d15, [sp, #48]

    ptrue p0.s

    // Load initial B
    ld1w {z20.s}, p0/z, [x1]
    ld1w {z21.s}, p0/z, [x1, #1, mul vl]

    // Zero 20 accumulators
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

    lsr x6, x3, #1

.L10x2_loop:
    // ═══ K iteration 0 ═══
    ld1rw {z22.s}, p0/z, [x0, #0]
    ld1rw {z23.s}, p0/z, [x0, #4]
    ld1rw {z24.s}, p0/z, [x0, #8]
    ld1rw {z25.s}, p0/z, [x0, #12]
    ld1rw {z26.s}, p0/z, [x0, #16]

    fmla z0.s, p0/m, z22.s, z20.s
    fmla z1.s, p0/m, z22.s, z21.s
    fmla z2.s, p0/m, z23.s, z20.s
    fmla z3.s, p0/m, z23.s, z21.s
    fmla z4.s, p0/m, z24.s, z20.s
    fmla z5.s, p0/m, z24.s, z21.s
    fmla z6.s, p0/m, z25.s, z20.s
    fmla z7.s, p0/m, z25.s, z21.s
    fmla z8.s, p0/m, z26.s, z20.s
    fmla z9.s, p0/m, z26.s, z21.s

    ld1rw {z27.s}, p0/z, [x0, #20]
    ld1rw {z28.s}, p0/z, [x0, #24]
    ld1rw {z29.s}, p0/z, [x0, #28]
    ld1rw {z30.s}, p0/z, [x0, #32]
    ld1rw {z31.s}, p0/z, [x0, #36]

    fmla z10.s, p0/m, z27.s, z20.s
    fmla z11.s, p0/m, z27.s, z21.s
    fmla z12.s, p0/m, z28.s, z20.s
    fmla z13.s, p0/m, z28.s, z21.s
    fmla z14.s, p0/m, z29.s, z20.s
    fmla z15.s, p0/m, z29.s, z21.s
    fmla z16.s, p0/m, z30.s, z20.s
    fmla z17.s, p0/m, z30.s, z21.s
    fmla z18.s, p0/m, z31.s, z20.s
    fmla z19.s, p0/m, z31.s, z21.s

    // Load B(k+1)
    ld1w {z20.s}, p0/z, [x1, #2, mul vl]
    ld1w {z21.s}, p0/z, [x1, #3, mul vl]

    // ═══ K iteration 1 ═══
    ld1rw {z22.s}, p0/z, [x0, #40]
    ld1rw {z23.s}, p0/z, [x0, #44]
    ld1rw {z24.s}, p0/z, [x0, #48]
    ld1rw {z25.s}, p0/z, [x0, #52]
    ld1rw {z26.s}, p0/z, [x0, #56]

    fmla z0.s, p0/m, z22.s, z20.s
    fmla z1.s, p0/m, z22.s, z21.s
    fmla z2.s, p0/m, z23.s, z20.s
    fmla z3.s, p0/m, z23.s, z21.s
    fmla z4.s, p0/m, z24.s, z20.s
    fmla z5.s, p0/m, z24.s, z21.s
    fmla z6.s, p0/m, z25.s, z20.s
    fmla z7.s, p0/m, z25.s, z21.s
    fmla z8.s, p0/m, z26.s, z20.s
    fmla z9.s, p0/m, z26.s, z21.s

    ld1rw {z27.s}, p0/z, [x0, #60]
    ld1rw {z28.s}, p0/z, [x0, #64]
    ld1rw {z29.s}, p0/z, [x0, #68]
    ld1rw {z30.s}, p0/z, [x0, #72]
    ld1rw {z31.s}, p0/z, [x0, #76]

    fmla z10.s, p0/m, z27.s, z20.s
    fmla z11.s, p0/m, z27.s, z21.s
    fmla z12.s, p0/m, z28.s, z20.s
    fmla z13.s, p0/m, z28.s, z21.s
    fmla z14.s, p0/m, z29.s, z20.s
    fmla z15.s, p0/m, z29.s, z21.s
    fmla z16.s, p0/m, z30.s, z20.s
    fmla z17.s, p0/m, z30.s, z21.s
    fmla z18.s, p0/m, z31.s, z20.s
    fmla z19.s, p0/m, z31.s, z21.s

    add x0, x0, #80     // A: 2 × 10 × 4
    add x1, x1, #256    // B: 2 × 2 × 64

    subs x6, x6, #1
    beq .L10x2_store

    ld1w {z20.s}, p0/z, [x1]
    ld1w {z21.s}, p0/z, [x1, #1, mul vl]
    b .L10x2_loop

.L10x2_store:
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

    ldp d14, d15, [sp, #48]
    ldp d12, d13, [sp, #32]
    ldp d10, d11, [sp, #16]
    ldp d8, d9, [sp], #64
    ret
    .size micro_kernel_fp32_10x2, .-micro_kernel_fp32_10x2
