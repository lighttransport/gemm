/*
 * SVE exp2 kernel using Estrin's polynomial scheme
 *
 * Estrin's method reduces dependency chain by evaluating polynomial terms in parallel.
 * For p(f) = 1 + c1*f + c2*f² + c3*f³ + c4*f⁴:
 *
 *   f2 = f * f           \
 *   q0 = 1 + c1*f         > parallel
 *   q1 = c2 + c3*f       /
 *   f4 = f2 * f2         \
 *   r = q0 + q1*f2        > parallel
 *   p = r + c4*f4        /
 *
 * This allows 2 FMLAs to execute in parallel vs Horner's serial chain.
 */

.arch armv8.2-a+sve

.text
.align 4

/*
 * exp2_estrin_simple: exp2 with Estrin polynomial (float input)
 */
.global exp2_estrin_simple
.type exp2_estrin_simple, %function
exp2_estrin_simple:
    stp     d8, d9, [sp, #-16]!
    stp     d10, d11, [sp, #-16]!
    stp     d12, d13, [sp, #-16]!
    stp     d14, d15, [sp, #-16]!

    ptrue   p0.s

    // Constants
    mov     w4, #127
    dup     z30.s, w4               // exponent bias

    // Polynomial coefficients (use movz for zero-extension)
    // c1 = ln(2) = 0.6931472 = 0x3F317218
    movz    w4, #0x7218
    movk    w4, #0x3F31, lsl #16
    dup     z26.s, w4

    // c2 = 0.2402265 = 0x3E75ED50
    movz    w4, #0xED50
    movk    w4, #0x3E75, lsl #16
    dup     z27.s, w4

    // c3 = 0.0555041 = 0x3D636C16
    movz    w4, #0x6C16
    movk    w4, #0x3D63, lsl #16
    dup     z28.s, w4

    // c4 = 0.0096181 = 0x3C1D8DAE
    movz    w4, #0x8DAE
    movk    w4, #0x3C1D, lsl #16
    dup     z29.s, w4

    // 1.0 = 0x3F800000
    movz    w4, #0x0000
    movk    w4, #0x3F80, lsl #16
    dup     z31.s, w4

    // Clamp constants
    movz    w4, #0x0000
    movk    w4, #0xC2FC, lsl #16    // -126.0
    dup     z24.s, w4

    movz    w4, #0x0000
    movk    w4, #0x42FE, lsl #16    // 127.0
    dup     z25.s, w4

    cntw    x4
    lsl     x5, x4, #2              // 4 * vl

    cmp     x2, x5
    b.lt    .Lestrin_tail

.Lestrin_loop4:
    // Load 4 vectors
    ld1w    z0.s, p0/z, [x0]
    ld1w    z1.s, p0/z, [x0, #1, mul vl]
    ld1w    z2.s, p0/z, [x0, #2, mul vl]
    ld1w    z3.s, p0/z, [x0, #3, mul vl]

    // Clamp to valid range
    fmax    z0.s, p0/m, z0.s, z24.s
    fmax    z1.s, p0/m, z1.s, z24.s
    fmax    z2.s, p0/m, z2.s, z24.s
    fmax    z3.s, p0/m, z3.s, z24.s

    fmin    z0.s, p0/m, z0.s, z25.s
    fmin    z1.s, p0/m, z1.s, z25.s
    fmin    z2.s, p0/m, z2.s, z25.s
    fmin    z3.s, p0/m, z3.s, z25.s

    // Split: n = floor(x), f = x - n
    frintm  z4.s, p0/m, z0.s        // n0
    frintm  z5.s, p0/m, z1.s        // n1
    frintm  z6.s, p0/m, z2.s        // n2
    frintm  z7.s, p0/m, z3.s        // n3

    fsub    z8.s, z0.s, z4.s        // f0
    fsub    z9.s, z1.s, z5.s        // f1
    fsub    z10.s, z2.s, z6.s       // f2
    fsub    z11.s, z3.s, z7.s       // f3

    // Compute 2^n via bit manipulation
    fcvtzs  z4.s, p0/m, z4.s
    fcvtzs  z5.s, p0/m, z5.s
    fcvtzs  z6.s, p0/m, z6.s
    fcvtzs  z7.s, p0/m, z7.s

    add     z4.s, z4.s, z30.s
    add     z5.s, z5.s, z30.s
    add     z6.s, z6.s, z30.s
    add     z7.s, z7.s, z30.s

    lsl     z4.s, z4.s, #23         // 2^n0
    lsl     z5.s, z5.s, #23         // 2^n1
    lsl     z6.s, z6.s, #23         // 2^n2
    lsl     z7.s, z7.s, #23         // 2^n3

    // ============ Estrin's scheme for 2^f ============
    // Step 1: f² and parallel q0, q1 computation

    // f² = f * f (for all 4 vectors)
    fmul    z12.s, z8.s, z8.s       // f0²
    fmul    z13.s, z9.s, z9.s       // f1²
    fmul    z14.s, z10.s, z10.s     // f2²
    fmul    z15.s, z11.s, z11.s     // f3²

    // q0 = 1 + c1*f (can execute in parallel with f²)
    mov     z16.d, z31.d            // q0_0 = 1.0
    mov     z17.d, z31.d            // q0_1 = 1.0
    mov     z18.d, z31.d            // q0_2 = 1.0
    mov     z19.d, z31.d            // q0_3 = 1.0
    fmla    z16.s, p0/m, z8.s, z26.s    // q0_0 = 1 + c1*f0
    fmla    z17.s, p0/m, z9.s, z26.s    // q0_1 = 1 + c1*f1
    fmla    z18.s, p0/m, z10.s, z26.s   // q0_2 = 1 + c1*f2
    fmla    z19.s, p0/m, z11.s, z26.s   // q0_3 = 1 + c1*f3

    // q1 = c2 + c3*f (parallel with q0)
    mov     z20.d, z27.d            // q1_0 = c2
    mov     z21.d, z27.d            // q1_1 = c2
    mov     z22.d, z27.d            // q1_2 = c2
    mov     z23.d, z27.d            // q1_3 = c2
    fmla    z20.s, p0/m, z8.s, z28.s    // q1_0 = c2 + c3*f0
    fmla    z21.s, p0/m, z9.s, z28.s    // q1_1 = c2 + c3*f1
    fmla    z22.s, p0/m, z10.s, z28.s   // q1_2 = c2 + c3*f2
    fmla    z23.s, p0/m, z11.s, z28.s   // q1_3 = c2 + c3*f3

    // Step 2: f⁴ and r = q0 + q1*f²

    // f⁴ = f² * f²
    fmul    z0.s, z12.s, z12.s      // f0⁴
    fmul    z1.s, z13.s, z13.s      // f1⁴
    fmul    z2.s, z14.s, z14.s      // f2⁴
    fmul    z3.s, z15.s, z15.s      // f3⁴

    // r = q0 + q1*f² (parallel with f⁴)
    fmla    z16.s, p0/m, z20.s, z12.s   // r0 = q0_0 + q1_0*f0²
    fmla    z17.s, p0/m, z21.s, z13.s   // r1 = q0_1 + q1_1*f1²
    fmla    z18.s, p0/m, z22.s, z14.s   // r2 = q0_2 + q1_2*f2²
    fmla    z19.s, p0/m, z23.s, z15.s   // r3 = q0_3 + q1_3*f3²

    // Step 3: p = r + c4*f⁴
    fmla    z16.s, p0/m, z0.s, z29.s    // p0 = r0 + c4*f0⁴
    fmla    z17.s, p0/m, z1.s, z29.s    // p1 = r1 + c4*f1⁴
    fmla    z18.s, p0/m, z2.s, z29.s    // p2 = r2 + c4*f2⁴
    fmla    z19.s, p0/m, z3.s, z29.s    // p3 = r3 + c4*f3⁴

    // Combine: exp2(x) = 2^n * 2^f
    fmul    z16.s, z16.s, z4.s
    fmul    z17.s, z17.s, z5.s
    fmul    z18.s, z18.s, z6.s
    fmul    z19.s, z19.s, z7.s

    // Store
    st1w    z16.s, p0, [x1]
    st1w    z17.s, p0, [x1, #1, mul vl]
    st1w    z18.s, p0, [x1, #2, mul vl]
    st1w    z19.s, p0, [x1, #3, mul vl]

    add     x0, x0, x5, lsl #2
    add     x1, x1, x5, lsl #2
    sub     x2, x2, x5
    cmp     x2, x5
    b.ge    .Lestrin_loop4

.Lestrin_tail:
    cbz     x2, .Lestrin_done

    // Handle remaining with predicate
    whilelt p1.s, xzr, x2

    ld1w    z0.s, p1/z, [x0]
    fmax    z0.s, p1/m, z0.s, z24.s
    fmin    z0.s, p1/m, z0.s, z25.s

    frintm  z4.s, p1/m, z0.s
    fsub    z8.s, z0.s, z4.s

    fcvtzs  z4.s, p1/m, z4.s
    add     z4.s, z4.s, z30.s
    lsl     z4.s, z4.s, #23

    // Estrin for single vector
    fmul    z12.s, z8.s, z8.s       // f²

    mov     z16.d, z31.d            // q0 = 1
    fmla    z16.s, p1/m, z8.s, z26.s    // q0 = 1 + c1*f

    mov     z20.d, z27.d            // q1 = c2
    fmla    z20.s, p1/m, z8.s, z28.s    // q1 = c2 + c3*f

    fmul    z0.s, z12.s, z12.s      // f⁴
    fmla    z16.s, p1/m, z20.s, z12.s   // r = q0 + q1*f²
    fmla    z16.s, p1/m, z0.s, z29.s    // p = r + c4*f⁴

    fmul    z16.s, p1/m, z16.s, z4.s

    st1w    z16.s, p1, [x1]

.Lestrin_done:
    ldp     d14, d15, [sp], #16
    ldp     d12, d13, [sp], #16
    ldp     d10, d11, [sp], #16
    ldp     d8, d9, [sp], #16
    ret
.size exp2_estrin_simple, .-exp2_estrin_simple


/*
 * exp2_estrin_softmax: Softmax exp2 with Estrin polynomial
 */
.global exp2_estrin_softmax
.type exp2_estrin_softmax, %function
exp2_estrin_softmax:
    stp     d8, d9, [sp, #-16]!
    stp     d10, d11, [sp, #-16]!
    stp     d12, d13, [sp, #-16]!
    stp     d14, d15, [sp, #-16]!

    ptrue   p0.s

    // Broadcast scale and max
    dup     z30.s, z0.s[0]          // scale
    mov     z29.s, w3
    scvtf   z29.s, p0/m, z29.s      // (float)max_val

    // Exponent bias
    mov     w4, #127
    dup     z28.s, w4

    // Polynomial coefficients
    movz    w4, #0x7218
    movk    w4, #0x3F31, lsl #16
    dup     z24.s, w4               // c1

    movz    w4, #0xED50
    movk    w4, #0x3E75, lsl #16
    dup     z25.s, w4               // c2

    movz    w4, #0x6C16
    movk    w4, #0x3D63, lsl #16
    dup     z26.s, w4               // c3

    movz    w4, #0x8DAE
    movk    w4, #0x3C1D, lsl #16
    dup     z27.s, w4               // c4

    // 1.0
    movz    w4, #0x0000
    movk    w4, #0x3F80, lsl #16
    dup     z31.s, w4

    // Clamp constants
    movz    w4, #0x0000
    movk    w4, #0xC2FC, lsl #16
    dup     z22.s, w4               // -126.0

    movz    w4, #0x0000
    movk    w4, #0x42FE, lsl #16
    dup     z23.s, w4               // 127.0

    cntw    x4
    lsl     x5, x4, #2

    cmp     x2, x5
    b.lt    .Lestrin_sm_tail

.Lestrin_sm_loop4:
    // Load 4 vectors of int32
    ld1w    z0.s, p0/z, [x0]
    ld1w    z1.s, p0/z, [x0, #1, mul vl]
    ld1w    z2.s, p0/z, [x0, #2, mul vl]
    ld1w    z3.s, p0/z, [x0, #3, mul vl]

    // Convert to float
    scvtf   z0.s, p0/m, z0.s
    scvtf   z1.s, p0/m, z1.s
    scvtf   z2.s, p0/m, z2.s
    scvtf   z3.s, p0/m, z3.s

    // (x - max) * scale
    fsub    z0.s, z0.s, z29.s
    fsub    z1.s, z1.s, z29.s
    fsub    z2.s, z2.s, z29.s
    fsub    z3.s, z3.s, z29.s

    fmul    z0.s, z0.s, z30.s
    fmul    z1.s, z1.s, z30.s
    fmul    z2.s, z2.s, z30.s
    fmul    z3.s, z3.s, z30.s

    // Clamp
    fmax    z0.s, p0/m, z0.s, z22.s
    fmax    z1.s, p0/m, z1.s, z22.s
    fmax    z2.s, p0/m, z2.s, z22.s
    fmax    z3.s, p0/m, z3.s, z22.s

    fmin    z0.s, p0/m, z0.s, z23.s
    fmin    z1.s, p0/m, z1.s, z23.s
    fmin    z2.s, p0/m, z2.s, z23.s
    fmin    z3.s, p0/m, z3.s, z23.s

    // Split n, f
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

    add     z4.s, z4.s, z28.s
    add     z5.s, z5.s, z28.s
    add     z6.s, z6.s, z28.s
    add     z7.s, z7.s, z28.s

    lsl     z4.s, z4.s, #23
    lsl     z5.s, z5.s, #23
    lsl     z6.s, z6.s, #23
    lsl     z7.s, z7.s, #23

    // Estrin's scheme
    // f²
    fmul    z12.s, z8.s, z8.s
    fmul    z13.s, z9.s, z9.s
    fmul    z14.s, z10.s, z10.s
    fmul    z15.s, z11.s, z11.s

    // q0 = 1 + c1*f
    mov     z16.d, z31.d
    mov     z17.d, z31.d
    mov     z18.d, z31.d
    mov     z19.d, z31.d
    fmla    z16.s, p0/m, z8.s, z24.s
    fmla    z17.s, p0/m, z9.s, z24.s
    fmla    z18.s, p0/m, z10.s, z24.s
    fmla    z19.s, p0/m, z11.s, z24.s

    // q1 = c2 + c3*f
    mov     z20.d, z25.d
    mov     z21.d, z25.d
    // Reuse z0-z3 for q1 (they're no longer needed)
    mov     z0.d, z25.d
    mov     z1.d, z25.d
    fmla    z20.s, p0/m, z8.s, z26.s
    fmla    z21.s, p0/m, z9.s, z26.s
    fmla    z0.s, p0/m, z10.s, z26.s
    fmla    z1.s, p0/m, z11.s, z26.s

    // f⁴
    fmul    z2.s, z12.s, z12.s
    fmul    z3.s, z13.s, z13.s
    // Reuse z8-z11 for f⁴ of vectors 2,3
    fmul    z8.s, z14.s, z14.s
    fmul    z9.s, z15.s, z15.s

    // r = q0 + q1*f²
    fmla    z16.s, p0/m, z20.s, z12.s
    fmla    z17.s, p0/m, z21.s, z13.s
    fmla    z18.s, p0/m, z0.s, z14.s
    fmla    z19.s, p0/m, z1.s, z15.s

    // p = r + c4*f⁴
    fmla    z16.s, p0/m, z2.s, z27.s
    fmla    z17.s, p0/m, z3.s, z27.s
    fmla    z18.s, p0/m, z8.s, z27.s
    fmla    z19.s, p0/m, z9.s, z27.s

    // 2^n * 2^f
    fmul    z16.s, z16.s, z4.s
    fmul    z17.s, z17.s, z5.s
    fmul    z18.s, z18.s, z6.s
    fmul    z19.s, z19.s, z7.s

    // Store
    st1w    z16.s, p0, [x1]
    st1w    z17.s, p0, [x1, #1, mul vl]
    st1w    z18.s, p0, [x1, #2, mul vl]
    st1w    z19.s, p0, [x1, #3, mul vl]

    add     x0, x0, x5, lsl #2
    add     x1, x1, x5, lsl #2
    sub     x2, x2, x5
    cmp     x2, x5
    b.ge    .Lestrin_sm_loop4

.Lestrin_sm_tail:
    cbz     x2, .Lestrin_sm_done

    whilelt p1.s, xzr, x2

    ld1w    z0.s, p1/z, [x0]
    scvtf   z0.s, p1/m, z0.s
    fsub    z0.s, p1/m, z0.s, z29.s
    fmul    z0.s, p1/m, z0.s, z30.s
    fmax    z0.s, p1/m, z0.s, z22.s
    fmin    z0.s, p1/m, z0.s, z23.s

    frintm  z4.s, p1/m, z0.s
    fsub    z8.s, z0.s, z4.s

    fcvtzs  z4.s, p1/m, z4.s
    add     z4.s, z4.s, z28.s
    lsl     z4.s, z4.s, #23

    // Estrin
    fmul    z12.s, z8.s, z8.s
    mov     z16.d, z31.d
    fmla    z16.s, p1/m, z8.s, z24.s
    mov     z20.d, z25.d
    fmla    z20.s, p1/m, z8.s, z26.s
    fmul    z0.s, z12.s, z12.s
    fmla    z16.s, p1/m, z20.s, z12.s
    fmla    z16.s, p1/m, z0.s, z27.s

    fmul    z16.s, p1/m, z16.s, z4.s
    st1w    z16.s, p1, [x1]

.Lestrin_sm_done:
    ldp     d14, d15, [sp], #16
    ldp     d12, d13, [sp], #16
    ldp     d10, d11, [sp], #16
    ldp     d8, d9, [sp], #16
    ret
.size exp2_estrin_softmax, .-exp2_estrin_softmax
