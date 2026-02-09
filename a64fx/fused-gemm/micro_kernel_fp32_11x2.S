// FP32 FMLA Micro-kernel: 11×2 tile, 2× K-unroll
// AI = 4.09, d=256 exact, L1 = 44.4 KB
//
// Registers:
//   Z0-Z21:  22 accumulators (11 rows × 2 cols)
//   Z22-Z23: B vectors (2 col vectors)
//   Z24-Z29: A temps batch 1 (rows 0-5, 6 regs)
//   Z24-Z28: A temps batch 2 (rows 6-10, 5 regs, reuse)
//
// Interface: x0=A[K][11], x1=B[K][32], x2=C, x3=K(even), x5=ldc_bytes

    .arch armv8.2-a+sve
    .text
    .align 4
    .global micro_kernel_fp32_11x2
    .type micro_kernel_fp32_11x2, %function

micro_kernel_fp32_11x2:
    stp d8, d9, [sp, #-64]!
    stp d10, d11, [sp, #16]
    stp d12, d13, [sp, #32]
    stp d14, d15, [sp, #48]

    ptrue p0.s

    ld1w {z22.s}, p0/z, [x1]
    ld1w {z23.s}, p0/z, [x1, #1, mul vl]

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

    lsr x6, x3, #1

.L11x2_loop:
    // ═══ K iteration 0 ═══
    // Batch 1: rows 0-5
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

    // Batch 2: rows 6-10
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

    // Load B(k+1)
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

    add x0, x0, #88     // A: 2 × 11 × 4
    add x1, x1, #256    // B: 2 × 2 × 64

    subs x6, x6, #1
    beq .L11x2_store

    ld1w {z22.s}, p0/z, [x1]
    ld1w {z23.s}, p0/z, [x1, #1, mul vl]
    b .L11x2_loop

.L11x2_store:
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
    .size micro_kernel_fp32_11x2, .-micro_kernel_fp32_11x2
