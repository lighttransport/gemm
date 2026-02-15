/*
 * FP8 GEMM Kernel with Fused Conversion - Version 3
 *
 * Key optimization: K-unroll by 2 to pipeline gather latency
 * - Start gather for k+1 while computing k
 * - Use mov z.s, z.s[i] (it's actually faster than st+ld roundtrip)
 *
 * Arguments:
 *   x0: A_fp8 pointer ([K][MR] packed)
 *   x1: B_fp32 pointer ([K][N])
 *   x2: C pointer
 *   x3: ldc in bytes
 *   x4: K (must be even)
 *   x5: LUT pointer
 */

    .arch armv8.2-a+sve
    .text
    .align 4
    .global fp8_fused_kernel_v3
    .type fp8_fused_kernel_v3, %function

fp8_fused_kernel_v3:
    ptrue p0.s

    // Initialize 24 accumulators
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

    // Predicate for 8 elements
    mov x6, #8
    whilelt p1.s, xzr, x6

    // K loop counter (process 2 K per iteration)
    lsr x6, x4, #1
    cbz x6, .Lv3_store

    // Load first B (k=0)
    ld1w {z24.s}, p0/z, [x1]
    ld1w {z25.s}, p0/z, [x1, #1, mul vl]
    ld1w {z26.s}, p0/z, [x1, #2, mul vl]

    // Load first A[k=0] and gather
    ld1b {z31.s}, p1/z, [x0]
    ld1w {z27.s}, p1/z, [x5, z31.s, uxtw #2]  // A[0..7] for k=0

.Lv3_loop:
    // ═══ First K (k=0) ═══
    // Start loading next A (k+1) early to hide gather latency
    add x7, x0, #8
    ld1b {z31.s}, p1/z, [x7]

    // Compute rows 0-3 using A[k=0] values
    mov z28.s, z27.s[0]
    fmla z0.s, p0/m, z28.s, z24.s
    fmla z1.s, p0/m, z28.s, z25.s
    fmla z2.s, p0/m, z28.s, z26.s

    mov z28.s, z27.s[1]
    fmla z3.s, p0/m, z28.s, z24.s
    fmla z4.s, p0/m, z28.s, z25.s
    fmla z5.s, p0/m, z28.s, z26.s

    // Start gather for A[k+1]
    ld1w {z30.s}, p1/z, [x5, z31.s, uxtw #2]

    mov z28.s, z27.s[2]
    fmla z6.s, p0/m, z28.s, z24.s
    fmla z7.s, p0/m, z28.s, z25.s
    fmla z8.s, p0/m, z28.s, z26.s

    mov z28.s, z27.s[3]
    fmla z9.s, p0/m, z28.s, z24.s
    fmla z10.s, p0/m, z28.s, z25.s
    fmla z11.s, p0/m, z28.s, z26.s

    // Compute rows 4-7 using A[k=0]
    mov z28.s, z27.s[4]
    fmla z12.s, p0/m, z28.s, z24.s
    fmla z13.s, p0/m, z28.s, z25.s
    fmla z14.s, p0/m, z28.s, z26.s

    mov z28.s, z27.s[5]
    fmla z15.s, p0/m, z28.s, z24.s
    fmla z16.s, p0/m, z28.s, z25.s
    fmla z17.s, p0/m, z28.s, z26.s

    mov z28.s, z27.s[6]
    fmla z18.s, p0/m, z28.s, z24.s
    fmla z19.s, p0/m, z28.s, z25.s
    fmla z20.s, p0/m, z28.s, z26.s

    mov z28.s, z27.s[7]
    fmla z21.s, p0/m, z28.s, z24.s
    fmla z22.s, p0/m, z28.s, z25.s
    fmla z23.s, p0/m, z28.s, z26.s

    // Load next B (k+1)
    ld1w {z24.s}, p0/z, [x1, #3, mul vl]
    ld1w {z25.s}, p0/z, [x1, #4, mul vl]
    ld1w {z26.s}, p0/z, [x1, #5, mul vl]

    // ═══ Second K (k+1) ═══
    // z30 has A values for k+1 from earlier gather

    // Start loading A[k+2] for next iteration
    add x8, x0, #16
    ld1b {z31.s}, p1/z, [x8]

    mov z28.s, z30.s[0]
    fmla z0.s, p0/m, z28.s, z24.s
    fmla z1.s, p0/m, z28.s, z25.s
    fmla z2.s, p0/m, z28.s, z26.s

    mov z28.s, z30.s[1]
    fmla z3.s, p0/m, z28.s, z24.s
    fmla z4.s, p0/m, z28.s, z25.s
    fmla z5.s, p0/m, z28.s, z26.s

    // Start gather for A[k+2]
    ld1w {z27.s}, p1/z, [x5, z31.s, uxtw #2]

    mov z28.s, z30.s[2]
    fmla z6.s, p0/m, z28.s, z24.s
    fmla z7.s, p0/m, z28.s, z25.s
    fmla z8.s, p0/m, z28.s, z26.s

    mov z28.s, z30.s[3]
    fmla z9.s, p0/m, z28.s, z24.s
    fmla z10.s, p0/m, z28.s, z25.s
    fmla z11.s, p0/m, z28.s, z26.s

    mov z28.s, z30.s[4]
    fmla z12.s, p0/m, z28.s, z24.s
    fmla z13.s, p0/m, z28.s, z25.s
    fmla z14.s, p0/m, z28.s, z26.s

    mov z28.s, z30.s[5]
    fmla z15.s, p0/m, z28.s, z24.s
    fmla z16.s, p0/m, z28.s, z25.s
    fmla z17.s, p0/m, z28.s, z26.s

    mov z28.s, z30.s[6]
    fmla z18.s, p0/m, z28.s, z24.s
    fmla z19.s, p0/m, z28.s, z25.s
    fmla z20.s, p0/m, z28.s, z26.s

    mov z28.s, z30.s[7]
    fmla z21.s, p0/m, z28.s, z24.s
    fmla z22.s, p0/m, z28.s, z25.s
    fmla z23.s, p0/m, z28.s, z26.s

    // Advance pointers
    add x0, x0, #16    // A: 2 * 8 bytes
    add x1, x1, #384   // B: 2 * 3 * 64 bytes

    subs x6, x6, #1
    beq .Lv3_store

    // Load next B (k+2)
    ld1w {z24.s}, p0/z, [x1]
    ld1w {z25.s}, p0/z, [x1, #1, mul vl]
    ld1w {z26.s}, p0/z, [x1, #2, mul vl]

    b .Lv3_loop

.Lv3_store:
    // Store results
    st1w {z0.s}, p0, [x2]
    st1w {z1.s}, p0, [x2, #1, mul vl]
    st1w {z2.s}, p0, [x2, #2, mul vl]
    add x2, x2, x3

    st1w {z3.s}, p0, [x2]
    st1w {z4.s}, p0, [x2, #1, mul vl]
    st1w {z5.s}, p0, [x2, #2, mul vl]
    add x2, x2, x3

    st1w {z6.s}, p0, [x2]
    st1w {z7.s}, p0, [x2, #1, mul vl]
    st1w {z8.s}, p0, [x2, #2, mul vl]
    add x2, x2, x3

    st1w {z9.s}, p0, [x2]
    st1w {z10.s}, p0, [x2, #1, mul vl]
    st1w {z11.s}, p0, [x2, #2, mul vl]
    add x2, x2, x3

    st1w {z12.s}, p0, [x2]
    st1w {z13.s}, p0, [x2, #1, mul vl]
    st1w {z14.s}, p0, [x2, #2, mul vl]
    add x2, x2, x3

    st1w {z15.s}, p0, [x2]
    st1w {z16.s}, p0, [x2, #1, mul vl]
    st1w {z17.s}, p0, [x2, #2, mul vl]
    add x2, x2, x3

    st1w {z18.s}, p0, [x2]
    st1w {z19.s}, p0, [x2, #1, mul vl]
    st1w {z20.s}, p0, [x2, #2, mul vl]
    add x2, x2, x3

    st1w {z21.s}, p0, [x2]
    st1w {z22.s}, p0, [x2, #1, mul vl]
    st1w {z23.s}, p0, [x2, #2, mul vl]

    ret
    .size fp8_fused_kernel_v3, .-fp8_fused_kernel_v3
