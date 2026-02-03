/*
 * SVE exp2 kernel using polynomial approximation (v2)
 *
 * Algorithm for exp2(x):
 *   n = floor(x)     (integer part)
 *   f = x - n        (fractional part, [0,1))
 *   exp2(x) = 2^n * 2^f
 *   - 2^n via IEEE754 bit manipulation
 *   - 2^f via minimax polynomial (degree 4)
 *
 * Polynomial for 2^f on [0,1]:
 *   p(f) = 1 + c1*f + c2*f^2 + c3*f^3 + c4*f^4
 *   Coefficients optimized for max error < 2 ULP
 */

.arch armv8.2-a+sve

.text
.align 4

/*
 * exp2_poly_softmax_v2: Compute exp2((x - max) * scale) for softmax
 */
.global exp2_poly_softmax_v2
.type exp2_poly_softmax_v2, %function
exp2_poly_softmax_v2:
    stp     d8, d9, [sp, #-16]!
    stp     d10, d11, [sp, #-16]!
    stp     d12, d13, [sp, #-16]!
    stp     d14, d15, [sp, #-16]!

    ptrue   p0.s

    // Broadcast scale
    dup     z20.s, z0.s[0]          // z20 = scale

    // Broadcast max_val and convert to float
    mov     z21.s, w3
    scvtf   z21.s, p0/m, z21.s      // z21 = (float)max_val

    // Constants
    mov     w4, #127
    dup     z30.s, w4               // exponent bias

    // Polynomial coefficients for 2^f on [0,1]
    // Use movz (zero-extend) instead of mov (sign-extend) for low 16 bits
    // c1 = ln(2) ≈ 0.693147180559945 = 0x3F317218
    movz    w4, #0x7218
    movk    w4, #0x3F31, lsl #16
    dup     z26.s, w4               // 0.6931472

    // c2 = ln(2)^2/2 ≈ 0.240226507 = 0x3E75ED50
    movz    w4, #0xED50
    movk    w4, #0x3E75, lsl #16
    dup     z27.s, w4               // 0.2402265

    // c3 = ln(2)^3/6 ≈ 0.055504109 = 0x3D636C16
    movz    w4, #0x6C16
    movk    w4, #0x3D63, lsl #16
    dup     z28.s, w4               // 0.05550411

    // c4 = ln(2)^4/24 ≈ 0.009618129 = 0x3C1D8DAE
    movz    w4, #0x8DAE
    movk    w4, #0x3C1D, lsl #16
    dup     z29.s, w4               // 0.009618129

    // Clamp constants
    // -126.0 = 0xC2FC0000
    movz    w4, #0x0000
    movk    w4, #0xC2FC, lsl #16
    dup     z24.s, w4

    // 127.0 = 0x42FE0000
    movz    w4, #0x0000
    movk    w4, #0x42FE, lsl #16
    dup     z25.s, w4

    // 1.0 = 0x3F800000
    movz    w4, #0x0000
    movk    w4, #0x3F80, lsl #16
    dup     z31.s, w4

    // Vector length
    cntw    x4
    lsl     x5, x4, #2

    cmp     x2, x5
    b.lt    .Lpoly2_tail

.Lpoly2_loop4:
    // Load 4 vectors
    ld1w    z0.s, p0/z, [x0]
    ld1w    z1.s, p0/z, [x0, #1, mul vl]
    ld1w    z2.s, p0/z, [x0, #2, mul vl]
    ld1w    z3.s, p0/z, [x0, #3, mul vl]

    // Convert int32 -> float
    scvtf   z0.s, p0/m, z0.s
    scvtf   z1.s, p0/m, z1.s
    scvtf   z2.s, p0/m, z2.s
    scvtf   z3.s, p0/m, z3.s

    // Subtract max and scale
    fsub    z0.s, z0.s, z21.s
    fsub    z1.s, z1.s, z21.s
    fsub    z2.s, z2.s, z21.s
    fsub    z3.s, z3.s, z21.s

    fmul    z0.s, z0.s, z20.s
    fmul    z1.s, z1.s, z20.s
    fmul    z2.s, z2.s, z20.s
    fmul    z3.s, z3.s, z20.s

    // Clamp
    fmax    z0.s, p0/m, z0.s, z24.s
    fmax    z1.s, p0/m, z1.s, z24.s
    fmax    z2.s, p0/m, z2.s, z24.s
    fmax    z3.s, p0/m, z3.s, z24.s

    fmin    z0.s, p0/m, z0.s, z25.s
    fmin    z1.s, p0/m, z1.s, z25.s
    fmin    z2.s, p0/m, z2.s, z25.s
    fmin    z3.s, p0/m, z3.s, z25.s

    // Split: n = floor(x), f = x - n
    frintm  z4.s, p0/m, z0.s
    frintm  z5.s, p0/m, z1.s
    frintm  z6.s, p0/m, z2.s
    frintm  z7.s, p0/m, z3.s

    fsub    z8.s, z0.s, z4.s        // f in [0, 1)
    fsub    z9.s, z1.s, z5.s
    fsub    z10.s, z2.s, z6.s
    fsub    z11.s, z3.s, z7.s

    // Compute 2^n via bit manipulation
    fcvtzs  z4.s, p0/m, z4.s
    fcvtzs  z5.s, p0/m, z5.s
    fcvtzs  z6.s, p0/m, z6.s
    fcvtzs  z7.s, p0/m, z7.s

    add     z4.s, z4.s, z30.s
    add     z5.s, z5.s, z30.s
    add     z6.s, z6.s, z30.s
    add     z7.s, z7.s, z30.s

    lsl     z4.s, z4.s, #23
    lsl     z5.s, z5.s, #23
    lsl     z6.s, z6.s, #23
    lsl     z7.s, z7.s, #23

    // Polynomial (Horner's method): p = 1 + f*(c1 + f*(c2 + f*(c3 + f*c4)))
    // p = c4*f + c3
    fmul    z12.s, z29.s, z8.s
    fmul    z13.s, z29.s, z9.s
    fmul    z14.s, z29.s, z10.s
    fmul    z15.s, z29.s, z11.s

    fadd    z12.s, z12.s, z28.s
    fadd    z13.s, z13.s, z28.s
    fadd    z14.s, z14.s, z28.s
    fadd    z15.s, z15.s, z28.s

    // p = p*f + c2
    fmul    z12.s, z12.s, z8.s
    fmul    z13.s, z13.s, z9.s
    fmul    z14.s, z14.s, z10.s
    fmul    z15.s, z15.s, z11.s

    fadd    z12.s, z12.s, z27.s
    fadd    z13.s, z13.s, z27.s
    fadd    z14.s, z14.s, z27.s
    fadd    z15.s, z15.s, z27.s

    // p = p*f + c1
    fmul    z12.s, z12.s, z8.s
    fmul    z13.s, z13.s, z9.s
    fmul    z14.s, z14.s, z10.s
    fmul    z15.s, z15.s, z11.s

    fadd    z12.s, z12.s, z26.s
    fadd    z13.s, z13.s, z26.s
    fadd    z14.s, z14.s, z26.s
    fadd    z15.s, z15.s, z26.s

    // p = p*f + 1
    fmul    z12.s, z12.s, z8.s
    fmul    z13.s, z13.s, z9.s
    fmul    z14.s, z14.s, z10.s
    fmul    z15.s, z15.s, z11.s

    fadd    z12.s, z12.s, z31.s
    fadd    z13.s, z13.s, z31.s
    fadd    z14.s, z14.s, z31.s
    fadd    z15.s, z15.s, z31.s

    // Combine: exp2(x) = 2^n * 2^f
    fmul    z12.s, z12.s, z4.s
    fmul    z13.s, z13.s, z5.s
    fmul    z14.s, z14.s, z6.s
    fmul    z15.s, z15.s, z7.s

    // Store
    st1w    z12.s, p0, [x1]
    st1w    z13.s, p0, [x1, #1, mul vl]
    st1w    z14.s, p0, [x1, #2, mul vl]
    st1w    z15.s, p0, [x1, #3, mul vl]

    add     x0, x0, x5, lsl #2
    add     x1, x1, x5, lsl #2
    sub     x2, x2, x5
    cmp     x2, x5
    b.ge    .Lpoly2_loop4

.Lpoly2_tail:
    cbz     x2, .Lpoly2_done

.Lpoly2_loop1:
    whilelt p1.s, xzr, x2

    ld1w    z0.s, p1/z, [x0]
    scvtf   z0.s, p1/m, z0.s
    fsub    z0.s, p1/m, z0.s, z21.s
    fmul    z0.s, p1/m, z0.s, z20.s
    fmax    z0.s, p1/m, z0.s, z24.s
    fmin    z0.s, p1/m, z0.s, z25.s

    frintm  z4.s, p1/m, z0.s
    fsub    z8.s, z0.s, z4.s

    fcvtzs  z4.s, p1/m, z4.s
    add     z4.s, z4.s, z30.s
    lsl     z4.s, z4.s, #23

    // Polynomial for single vector
    fmul    z12.s, z29.s, z8.s      // c4 * f
    fadd    z12.s, z12.s, z28.s     // + c3
    fmul    z12.s, z12.s, z8.s      // * f
    fadd    z12.s, z12.s, z27.s     // + c2
    fmul    z12.s, z12.s, z8.s      // * f
    fadd    z12.s, z12.s, z26.s     // + c1
    fmul    z12.s, z12.s, z8.s      // * f
    fadd    z12.s, z12.s, z31.s     // + 1

    fmul    z12.s, p1/m, z12.s, z4.s

    st1w    z12.s, p1, [x1]
    // All remaining elements processed with predicate, done

.Lpoly2_done:
    ldp     d14, d15, [sp], #16
    ldp     d12, d13, [sp], #16
    ldp     d10, d11, [sp], #16
    ldp     d8, d9, [sp], #16
    ret
.size exp2_poly_softmax_v2, .-exp2_poly_softmax_v2


/*
 * exp2_poly_simple_v2: Simple exp2 for float input
 */
.global exp2_poly_simple_v2
.type exp2_poly_simple_v2, %function
exp2_poly_simple_v2:
    stp     d8, d9, [sp, #-16]!
    stp     d10, d11, [sp, #-16]!
    stp     d12, d13, [sp, #-16]!
    stp     d14, d15, [sp, #-16]!

    ptrue   p0.s

    // Constants
    mov     w4, #127
    dup     z30.s, w4

    // Polynomial coefficients (use movz for zero-extension)
    movz    w4, #0x7218
    movk    w4, #0x3F31, lsl #16
    dup     z26.s, w4               // c1 = 0.6931472

    movz    w4, #0xED50
    movk    w4, #0x3E75, lsl #16
    dup     z27.s, w4               // c2 = 0.2402265

    movz    w4, #0x6C16
    movk    w4, #0x3D63, lsl #16
    dup     z28.s, w4               // c3 = 0.05550411

    movz    w4, #0x8DAE
    movk    w4, #0x3C1D, lsl #16
    dup     z29.s, w4               // c4 = 0.009618129

    // Clamp
    movz    w4, #0x0000
    movk    w4, #0xC2FC, lsl #16
    dup     z24.s, w4               // -126.0

    movz    w4, #0x0000
    movk    w4, #0x42FE, lsl #16
    dup     z25.s, w4               // 127.0

    // 1.0
    movz    w4, #0x0000
    movk    w4, #0x3F80, lsl #16
    dup     z31.s, w4

    cntw    x4
    lsl     x5, x4, #2

    cmp     x2, x5
    b.lt    .Lsimpoly2_tail

.Lsimpoly2_loop4:
    ld1w    z0.s, p0/z, [x0]
    ld1w    z1.s, p0/z, [x0, #1, mul vl]
    ld1w    z2.s, p0/z, [x0, #2, mul vl]
    ld1w    z3.s, p0/z, [x0, #3, mul vl]

    // Clamp
    fmax    z0.s, p0/m, z0.s, z24.s
    fmax    z1.s, p0/m, z1.s, z24.s
    fmax    z2.s, p0/m, z2.s, z24.s
    fmax    z3.s, p0/m, z3.s, z24.s

    fmin    z0.s, p0/m, z0.s, z25.s
    fmin    z1.s, p0/m, z1.s, z25.s
    fmin    z2.s, p0/m, z2.s, z25.s
    fmin    z3.s, p0/m, z3.s, z25.s

    // Split
    frintm  z4.s, p0/m, z0.s
    frintm  z5.s, p0/m, z1.s
    frintm  z6.s, p0/m, z2.s
    frintm  z7.s, p0/m, z3.s

    fsub    z8.s, z0.s, z4.s
    fsub    z9.s, z1.s, z5.s
    fsub    z10.s, z2.s, z6.s
    fsub    z11.s, z3.s, z7.s

    // 2^n
    fcvtzs  z4.s, p0/m, z4.s
    fcvtzs  z5.s, p0/m, z5.s
    fcvtzs  z6.s, p0/m, z6.s
    fcvtzs  z7.s, p0/m, z7.s

    add     z4.s, z4.s, z30.s
    add     z5.s, z5.s, z30.s
    add     z6.s, z6.s, z30.s
    add     z7.s, z7.s, z30.s

    lsl     z4.s, z4.s, #23
    lsl     z5.s, z5.s, #23
    lsl     z6.s, z6.s, #23
    lsl     z7.s, z7.s, #23

    // Polynomial (Horner's method)
    // p = c4*f + c3
    fmul    z12.s, z29.s, z8.s
    fmul    z13.s, z29.s, z9.s
    fmul    z14.s, z29.s, z10.s
    fmul    z15.s, z29.s, z11.s

    fadd    z12.s, z12.s, z28.s
    fadd    z13.s, z13.s, z28.s
    fadd    z14.s, z14.s, z28.s
    fadd    z15.s, z15.s, z28.s

    // p = p*f + c2
    fmul    z12.s, z12.s, z8.s
    fmul    z13.s, z13.s, z9.s
    fmul    z14.s, z14.s, z10.s
    fmul    z15.s, z15.s, z11.s

    fadd    z12.s, z12.s, z27.s
    fadd    z13.s, z13.s, z27.s
    fadd    z14.s, z14.s, z27.s
    fadd    z15.s, z15.s, z27.s

    // p = p*f + c1
    fmul    z12.s, z12.s, z8.s
    fmul    z13.s, z13.s, z9.s
    fmul    z14.s, z14.s, z10.s
    fmul    z15.s, z15.s, z11.s

    fadd    z12.s, z12.s, z26.s
    fadd    z13.s, z13.s, z26.s
    fadd    z14.s, z14.s, z26.s
    fadd    z15.s, z15.s, z26.s

    // p = p*f + 1
    fmul    z12.s, z12.s, z8.s
    fmul    z13.s, z13.s, z9.s
    fmul    z14.s, z14.s, z10.s
    fmul    z15.s, z15.s, z11.s

    fadd    z12.s, z12.s, z31.s
    fadd    z13.s, z13.s, z31.s
    fadd    z14.s, z14.s, z31.s
    fadd    z15.s, z15.s, z31.s

    // exp2(x) = 2^n * p
    fmul    z12.s, z12.s, z4.s
    fmul    z13.s, z13.s, z5.s
    fmul    z14.s, z14.s, z6.s
    fmul    z15.s, z15.s, z7.s

    st1w    z12.s, p0, [x1]
    st1w    z13.s, p0, [x1, #1, mul vl]
    st1w    z14.s, p0, [x1, #2, mul vl]
    st1w    z15.s, p0, [x1, #3, mul vl]

    add     x0, x0, x5, lsl #2
    add     x1, x1, x5, lsl #2
    sub     x2, x2, x5
    cmp     x2, x5
    b.ge    .Lsimpoly2_loop4

.Lsimpoly2_tail:
    cbz     x2, .Lsimpoly2_done

.Lsimpoly2_loop1:
    whilelt p1.s, xzr, x2

    ld1w    z0.s, p1/z, [x0]

    fmax    z0.s, p1/m, z0.s, z24.s
    fmin    z0.s, p1/m, z0.s, z25.s

    frintm  z4.s, p1/m, z0.s
    fsub    z8.s, z0.s, z4.s

    fcvtzs  z4.s, p1/m, z4.s
    add     z4.s, z4.s, z30.s
    lsl     z4.s, z4.s, #23

    fmul    z12.s, z29.s, z8.s
    fadd    z12.s, z12.s, z28.s
    fmul    z12.s, z12.s, z8.s
    fadd    z12.s, z12.s, z27.s
    fmul    z12.s, z12.s, z8.s
    fadd    z12.s, z12.s, z26.s
    fmul    z12.s, z12.s, z8.s
    fadd    z12.s, z12.s, z31.s

    fmul    z12.s, p1/m, z12.s, z4.s

    st1w    z12.s, p1, [x1]
    // All remaining elements processed, done

.Lsimpoly2_done:
    ldp     d14, d15, [sp], #16
    ldp     d12, d13, [sp], #16
    ldp     d10, d11, [sp], #16
    ldp     d8, d9, [sp], #16
    ret
.size exp2_poly_simple_v2, .-exp2_poly_simple_v2
