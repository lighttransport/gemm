// exp_bench.S
// SVE exp() benchmark using range reduction + fexpa + polynomial correction
//
// Algorithm:
//   1. Range reduction: x = n * ln(2) + r, where n = round(x / ln(2))
//   2. exp(x) = 2^n * exp(r)
//   3. exp(r) ≈ fexpa(r) * correction_poly(r)
//
// fexpa gives a 6-7 bit accurate approximation of 2^x for x in [0,1)
// We use polynomial correction to improve accuracy

    .arch armv8.2-a+sve
    .text

// Constants (will be loaded into registers)
// 1/ln(2) = 1.4426950408889634 = 0x3fb8aa3b
// ln(2) = 0.6931471805599453 = 0x3f317218

// ============================================
// exp_vec_unroll4: Process N floats with unroll 4
// Args: x0 = input array, x1 = output array, x2 = count
// ============================================
    .align 6
    .global exp_vec_unroll4
    .type exp_vec_unroll4, %function
exp_vec_unroll4:
    ptrue   p0.s

    // Load constants
    movz    w10, #0xaa3b
    movk    w10, #0x3fb8, lsl #16   // 1/ln(2)
    dup     z28.s, w10

    movz    w10, #0x7218
    movk    w10, #0x3f31, lsl #16   // ln(2)
    dup     z29.s, w10

    // Correction polynomial coefficients for fexpa
    // exp(r) / fexpa(r) ≈ 1 + c1*r + c2*r^2 where r is fractional part
    // c1 ≈ 0.0 (fexpa handles linear term)
    // c2 ≈ 0.5 for quadratic correction
    fmov    z30.s, #0.5            // c2

    cntw    x3                      // elements per vector (16 for 512-bit)
    lsl     x4, x3, #2              // x4 = 4 * vl (unroll factor 4)

    mov     x5, x2                  // remaining count

.Lloop4:
    cmp     x5, x4
    b.lt    .Ltail4

    // Load 4 vectors
    ld1w    z0.s, p0/z, [x0]
    ld1w    z1.s, p0/z, [x0, #1, mul vl]
    ld1w    z2.s, p0/z, [x0, #2, mul vl]
    ld1w    z3.s, p0/z, [x0, #3, mul vl]

    // Range reduction: n = round(x / ln(2))
    fmul    z4.s, z0.s, z28.s
    fmul    z5.s, z1.s, z28.s
    fmul    z6.s, z2.s, z28.s
    fmul    z7.s, z3.s, z28.s

    frintn  z4.s, p0/m, z4.s       // n0
    frintn  z5.s, p0/m, z5.s       // n1
    frintn  z6.s, p0/m, z6.s       // n2
    frintn  z7.s, p0/m, z7.s       // n3

    // r = x - n * ln(2)
    fmls    z0.s, p0/m, z4.s, z29.s  // r0 = x0 - n0 * ln2
    fmls    z1.s, p0/m, z5.s, z29.s  // r1
    fmls    z2.s, p0/m, z6.s, z29.s  // r2
    fmls    z3.s, p0/m, z7.s, z29.s  // r3

    // Convert n to integer for scaling
    fcvtzs  z4.s, p0/m, z4.s
    fcvtzs  z5.s, p0/m, z5.s
    fcvtzs  z6.s, p0/m, z6.s
    fcvtzs  z7.s, p0/m, z7.s

    // Compute 2^n by manipulating IEEE754 exponent
    // 2^n = ((n + 127) << 23) reinterpreted as float
    mov     z31.s, #127
    add     z4.s, z4.s, z31.s
    add     z5.s, z5.s, z31.s
    add     z6.s, z6.s, z31.s
    add     z7.s, z7.s, z31.s

    lsl     z4.s, z4.s, #23        // 2^n0
    lsl     z5.s, z5.s, #23        // 2^n1
    lsl     z6.s, z6.s, #23        // 2^n2
    lsl     z7.s, z7.s, #23        // 2^n3

    // Polynomial for exp(r): 1 + r(1 + r(0.5 + r(1/6 + r/24)))
    // Start with p = c4 = 1/24
    movz    w10, #0xaaab
    movk    w10, #0x3d2a, lsl #16   // 1/24
    dup     z8.s, w10
    dup     z9.s, w10
    dup     z10.s, w10
    dup     z11.s, w10

    // p = p * r + c3  ->  fmad z8, p0/m, z0, z31 means z8 = z8*z0 + z31
    movz    w10, #0xaaab
    movk    w10, #0x3e2a, lsl #16   // 1/6
    dup     z31.s, w10

    fmad    z8.s, p0/m, z0.s, z31.s    // p0 = c4*r0 + c3
    fmad    z9.s, p0/m, z1.s, z31.s
    fmad    z10.s, p0/m, z2.s, z31.s
    fmad    z11.s, p0/m, z3.s, z31.s

    // p = p * r + 0.5
    fmov    z31.s, #0.5
    fmad    z8.s, p0/m, z0.s, z31.s
    fmad    z9.s, p0/m, z1.s, z31.s
    fmad    z10.s, p0/m, z2.s, z31.s
    fmad    z11.s, p0/m, z3.s, z31.s

    // p = p * r + 1.0
    fmov    z31.s, #1.0
    fmad    z8.s, p0/m, z0.s, z31.s
    fmad    z9.s, p0/m, z1.s, z31.s
    fmad    z10.s, p0/m, z2.s, z31.s
    fmad    z11.s, p0/m, z3.s, z31.s

    // p = p * r + 1.0
    fmad    z8.s, p0/m, z0.s, z31.s
    fmad    z9.s, p0/m, z1.s, z31.s
    fmad    z10.s, p0/m, z2.s, z31.s
    fmad    z11.s, p0/m, z3.s, z31.s

    // exp(x) = 2^n * exp(r) = z4 * z8 (treating z4 as float)
    fmul    z8.s, z8.s, z4.s
    fmul    z9.s, z9.s, z5.s
    fmul    z10.s, z10.s, z6.s
    fmul    z11.s, z11.s, z7.s

    // Store results
    st1w    z8.s, p0, [x1]
    st1w    z9.s, p0, [x1, #1, mul vl]
    st1w    z10.s, p0, [x1, #2, mul vl]
    st1w    z11.s, p0, [x1, #3, mul vl]

    add     x0, x0, x4, lsl #2      // advance input by 4*vl floats
    add     x1, x1, x4, lsl #2      // advance output
    sub     x5, x5, x4
    b       .Lloop4

.Ltail4:
    // Handle remaining elements with predicated loads
    cbz     x5, .Ldone4

    whilelt p1.s, xzr, x5
    ld1w    z0.s, p1/z, [x0]

    // Same computation for single vector
    fmul    z4.s, z0.s, z28.s
    frintn  z4.s, p0/m, z4.s
    fmls   z0.s, p0/m, z4.s, z29.s
    fcvtzs  z4.s, p0/m, z4.s
    mov     z31.s, #127
    add     z4.s, z4.s, z31.s
    lsl     z4.s, z4.s, #23

    movz    w10, #0xaaab
    movk    w10, #0x3d2a, lsl #16
    dup     z8.s, w10
    movz    w10, #0xaaab
    movk    w10, #0x3e2a, lsl #16
    dup     z31.s, w10
    fmad    z8.s, p0/m, z0.s, z31.s
    fmov    z31.s, #0.5
    fmad    z8.s, p0/m, z0.s, z31.s
    fmov    z31.s, #1.0
    fmad    z8.s, p0/m, z0.s, z31.s
    fmad    z8.s, p0/m, z0.s, z31.s
    fmul    z8.s, z8.s, z4.s

    st1w    z8.s, p1, [x1]

.Ldone4:
    ret
    .size exp_vec_unroll4, .-exp_vec_unroll4


// ============================================
// exp_vec_unroll5: Process N floats with unroll 5
// ============================================
    .align 6
    .global exp_vec_unroll5
    .type exp_vec_unroll5, %function
exp_vec_unroll5:
    ptrue   p0.s

    movz    w10, #0xaa3b
    movk    w10, #0x3fb8, lsl #16
    dup     z28.s, w10              // 1/ln(2)

    movz    w10, #0x7218
    movk    w10, #0x3f31, lsl #16
    dup     z29.s, w10              // ln(2)

    cntw    x3
    mov     x4, #5
    mul     x4, x3, x4              // 5 * vl

    mov     x5, x2

.Lloop5:
    cmp     x5, x4
    b.lt    .Ltail5

    // Load 5 vectors
    ld1w    z0.s, p0/z, [x0]
    ld1w    z1.s, p0/z, [x0, #1, mul vl]
    ld1w    z2.s, p0/z, [x0, #2, mul vl]
    ld1w    z3.s, p0/z, [x0, #3, mul vl]
    ld1w    z4.s, p0/z, [x0, #4, mul vl]

    // Range reduction
    fmul    z5.s, z0.s, z28.s
    fmul    z6.s, z1.s, z28.s
    fmul    z7.s, z2.s, z28.s
    fmul    z16.s, z3.s, z28.s
    fmul    z17.s, z4.s, z28.s

    frintn  z5.s, p0/m, z5.s
    frintn  z6.s, p0/m, z6.s
    frintn  z7.s, p0/m, z7.s
    frintn  z16.s, p0/m, z16.s
    frintn  z17.s, p0/m, z17.s

    fmls    z0.s, p0/m, z5.s, z29.s
    fmls    z1.s, p0/m, z6.s, z29.s
    fmls    z2.s, p0/m, z7.s, z29.s
    fmls    z3.s, p0/m, z16.s, z29.s
    fmls    z4.s, p0/m, z17.s, z29.s

    // 2^n
    fcvtzs  z5.s, p0/m, z5.s
    fcvtzs  z6.s, p0/m, z6.s
    fcvtzs  z7.s, p0/m, z7.s
    fcvtzs  z16.s, p0/m, z16.s
    fcvtzs  z17.s, p0/m, z17.s

    mov     z31.s, #127
    add     z5.s, z5.s, z31.s
    add     z6.s, z6.s, z31.s
    add     z7.s, z7.s, z31.s
    add     z16.s, z16.s, z31.s
    add     z17.s, z17.s, z31.s

    lsl     z5.s, z5.s, #23
    lsl     z6.s, z6.s, #23
    lsl     z7.s, z7.s, #23
    lsl     z16.s, z16.s, #23
    lsl     z17.s, z17.s, #23

    // Polynomial for exp(r)
    movz    w10, #0xaaab
    movk    w10, #0x3d2a, lsl #16   // 1/24
    dup     z18.s, w10
    dup     z19.s, w10
    dup     z20.s, w10
    dup     z21.s, w10
    dup     z22.s, w10

    movz    w10, #0xaaab
    movk    w10, #0x3e2a, lsl #16   // 1/6
    dup     z31.s, w10

    fmad    z18.s, p0/m, z0.s, z31.s
    fmad    z19.s, p0/m, z1.s, z31.s
    fmad    z20.s, p0/m, z2.s, z31.s
    fmad    z21.s, p0/m, z3.s, z31.s
    fmad    z22.s, p0/m, z4.s, z31.s

    fmov    z31.s, #0.5
    fmad    z18.s, p0/m, z0.s, z31.s
    fmad    z19.s, p0/m, z1.s, z31.s
    fmad    z20.s, p0/m, z2.s, z31.s
    fmad    z21.s, p0/m, z3.s, z31.s
    fmad    z22.s, p0/m, z4.s, z31.s

    fmov    z31.s, #1.0
    fmad    z18.s, p0/m, z0.s, z31.s
    fmad    z19.s, p0/m, z1.s, z31.s
    fmad    z20.s, p0/m, z2.s, z31.s
    fmad    z21.s, p0/m, z3.s, z31.s
    fmad    z22.s, p0/m, z4.s, z31.s

    fmad    z18.s, p0/m, z0.s, z31.s
    fmad    z19.s, p0/m, z1.s, z31.s
    fmad    z20.s, p0/m, z2.s, z31.s
    fmad    z21.s, p0/m, z3.s, z31.s
    fmad    z22.s, p0/m, z4.s, z31.s

    // Multiply by 2^n
    fmul    z18.s, z18.s, z5.s
    fmul    z19.s, z19.s, z6.s
    fmul    z20.s, z20.s, z7.s
    fmul    z21.s, z21.s, z16.s
    fmul    z22.s, z22.s, z17.s

    st1w    z18.s, p0, [x1]
    st1w    z19.s, p0, [x1, #1, mul vl]
    st1w    z20.s, p0, [x1, #2, mul vl]
    st1w    z21.s, p0, [x1, #3, mul vl]
    st1w    z22.s, p0, [x1, #4, mul vl]

    add     x0, x0, x4, lsl #2
    add     x1, x1, x4, lsl #2
    sub     x5, x5, x4
    b       .Lloop5

.Ltail5:
    cbz     x5, .Ldone5
    // Simple scalar fallback for tail
    whilelt p1.s, xzr, x5
    ld1w    z0.s, p1/z, [x0]

    fmul    z4.s, z0.s, z28.s
    frintn  z4.s, p0/m, z4.s
    fmls   z0.s, p0/m, z4.s, z29.s
    fcvtzs  z4.s, p0/m, z4.s
    mov     z31.s, #127
    add     z4.s, z4.s, z31.s
    lsl     z4.s, z4.s, #23

    movz    w10, #0xaaab
    movk    w10, #0x3d2a, lsl #16
    dup     z8.s, w10
    movz    w10, #0xaaab
    movk    w10, #0x3e2a, lsl #16
    dup     z31.s, w10
    fmad    z8.s, p0/m, z0.s, z31.s
    fmov    z31.s, #0.5
    fmad    z8.s, p0/m, z0.s, z31.s
    fmov    z31.s, #1.0
    fmad    z8.s, p0/m, z0.s, z31.s
    fmad    z8.s, p0/m, z0.s, z31.s
    fmul    z8.s, z8.s, z4.s

    st1w    z8.s, p1, [x1]

.Ldone5:
    ret
    .size exp_vec_unroll5, .-exp_vec_unroll5


// ============================================
// exp_vec_unroll8: Process N floats with unroll 8
// ============================================
    .align 6
    .global exp_vec_unroll8
    .type exp_vec_unroll8, %function
exp_vec_unroll8:
    ptrue   p0.s

    movz    w10, #0xaa3b
    movk    w10, #0x3fb8, lsl #16
    dup     z28.s, w10              // 1/ln(2)

    movz    w10, #0x7218
    movk    w10, #0x3f31, lsl #16
    dup     z29.s, w10              // ln(2)

    cntw    x3
    lsl     x4, x3, #3              // 8 * vl

    mov     x5, x2

.Lloop8:
    cmp     x5, x4
    b.lt    .Ltail8

    // Load 8 vectors
    ld1w    z0.s, p0/z, [x0]
    ld1w    z1.s, p0/z, [x0, #1, mul vl]
    ld1w    z2.s, p0/z, [x0, #2, mul vl]
    ld1w    z3.s, p0/z, [x0, #3, mul vl]
    ld1w    z4.s, p0/z, [x0, #4, mul vl]
    ld1w    z5.s, p0/z, [x0, #5, mul vl]
    ld1w    z6.s, p0/z, [x0, #6, mul vl]
    ld1w    z7.s, p0/z, [x0, #7, mul vl]

    // Range reduction for all 8
    fmul    z8.s, z0.s, z28.s
    fmul    z9.s, z1.s, z28.s
    fmul    z10.s, z2.s, z28.s
    fmul    z11.s, z3.s, z28.s
    fmul    z12.s, z4.s, z28.s
    fmul    z13.s, z5.s, z28.s
    fmul    z14.s, z6.s, z28.s
    fmul    z15.s, z7.s, z28.s

    frintn  z8.s, p0/m, z8.s
    frintn  z9.s, p0/m, z9.s
    frintn  z10.s, p0/m, z10.s
    frintn  z11.s, p0/m, z11.s
    frintn  z12.s, p0/m, z12.s
    frintn  z13.s, p0/m, z13.s
    frintn  z14.s, p0/m, z14.s
    frintn  z15.s, p0/m, z15.s

    fmls   z0.s, p0/m, z8.s, z29.s
    fmls   z1.s, p0/m, z9.s, z29.s
    fmls   z2.s, p0/m, z10.s, z29.s
    fmls   z3.s, p0/m, z11.s, z29.s
    fmls   z4.s, p0/m, z12.s, z29.s
    fmls   z5.s, p0/m, z13.s, z29.s
    fmls   z6.s, p0/m, z14.s, z29.s
    fmls   z7.s, p0/m, z15.s, z29.s

    // 2^n
    fcvtzs  z8.s, p0/m, z8.s
    fcvtzs  z9.s, p0/m, z9.s
    fcvtzs  z10.s, p0/m, z10.s
    fcvtzs  z11.s, p0/m, z11.s
    fcvtzs  z12.s, p0/m, z12.s
    fcvtzs  z13.s, p0/m, z13.s
    fcvtzs  z14.s, p0/m, z14.s
    fcvtzs  z15.s, p0/m, z15.s

    mov     z31.s, #127
    add     z8.s, z8.s, z31.s
    add     z9.s, z9.s, z31.s
    add     z10.s, z10.s, z31.s
    add     z11.s, z11.s, z31.s
    add     z12.s, z12.s, z31.s
    add     z13.s, z13.s, z31.s
    add     z14.s, z14.s, z31.s
    add     z15.s, z15.s, z31.s

    lsl     z8.s, z8.s, #23
    lsl     z9.s, z9.s, #23
    lsl     z10.s, z10.s, #23
    lsl     z11.s, z11.s, #23
    lsl     z12.s, z12.s, #23
    lsl     z13.s, z13.s, #23
    lsl     z14.s, z14.s, #23
    lsl     z15.s, z15.s, #23

    // Polynomial: need to reload r values since we overwrote z0-z7
    // Actually we need to save r values before computing 2^n
    // Let me restructure to use different registers...

    // Reload inputs and recompute r (trading compute for registers)
    ld1w    z16.s, p0/z, [x0]
    ld1w    z17.s, p0/z, [x0, #1, mul vl]
    ld1w    z18.s, p0/z, [x0, #2, mul vl]
    ld1w    z19.s, p0/z, [x0, #3, mul vl]
    ld1w    z20.s, p0/z, [x0, #4, mul vl]
    ld1w    z21.s, p0/z, [x0, #5, mul vl]
    ld1w    z22.s, p0/z, [x0, #6, mul vl]
    ld1w    z23.s, p0/z, [x0, #7, mul vl]

    // Recompute n and r
    fmul    z0.s, z16.s, z28.s
    fmul    z1.s, z17.s, z28.s
    fmul    z2.s, z18.s, z28.s
    fmul    z3.s, z19.s, z28.s
    fmul    z4.s, z20.s, z28.s
    fmul    z5.s, z21.s, z28.s
    fmul    z6.s, z22.s, z28.s
    fmul    z7.s, z23.s, z28.s

    frintn  z0.s, p0/m, z0.s
    frintn  z1.s, p0/m, z1.s
    frintn  z2.s, p0/m, z2.s
    frintn  z3.s, p0/m, z3.s
    frintn  z4.s, p0/m, z4.s
    frintn  z5.s, p0/m, z5.s
    frintn  z6.s, p0/m, z6.s
    frintn  z7.s, p0/m, z7.s

    fmls   z16.s, p0/m, z0.s, z29.s   // r0
    fmls   z17.s, p0/m, z1.s, z29.s   // r1
    fmls   z18.s, p0/m, z2.s, z29.s   // r2
    fmls   z19.s, p0/m, z3.s, z29.s   // r3
    fmls   z20.s, p0/m, z4.s, z29.s   // r4
    fmls   z21.s, p0/m, z5.s, z29.s   // r5
    fmls   z22.s, p0/m, z6.s, z29.s   // r6
    fmls   z23.s, p0/m, z7.s, z29.s   // r7

    // Polynomial for exp(r) - 4-term approximation
    // p = 1/24
    movz    w10, #0xaaab
    movk    w10, #0x3d2a, lsl #16
    dup     z0.s, w10
    dup     z1.s, w10
    dup     z2.s, w10
    dup     z3.s, w10
    dup     z4.s, w10
    dup     z5.s, w10
    dup     z6.s, w10
    dup     z7.s, w10

    // p = p*r + 1/6
    movz    w10, #0xaaab
    movk    w10, #0x3e2a, lsl #16
    dup     z31.s, w10

    fmad    z0.s, p0/m, z16.s, z31.s
    fmad    z1.s, p0/m, z17.s, z31.s
    fmad    z2.s, p0/m, z18.s, z31.s
    fmad    z3.s, p0/m, z19.s, z31.s
    fmad    z4.s, p0/m, z20.s, z31.s
    fmad    z5.s, p0/m, z21.s, z31.s
    fmad    z6.s, p0/m, z22.s, z31.s
    fmad    z7.s, p0/m, z23.s, z31.s

    // p = p*r + 0.5
    fmov    z31.s, #0.5
    fmad    z0.s, p0/m, z16.s, z31.s
    fmad    z1.s, p0/m, z17.s, z31.s
    fmad    z2.s, p0/m, z18.s, z31.s
    fmad    z3.s, p0/m, z19.s, z31.s
    fmad    z4.s, p0/m, z20.s, z31.s
    fmad    z5.s, p0/m, z21.s, z31.s
    fmad    z6.s, p0/m, z22.s, z31.s
    fmad    z7.s, p0/m, z23.s, z31.s

    // p = p*r + 1.0
    fmov    z31.s, #1.0
    fmad    z0.s, p0/m, z16.s, z31.s
    fmad    z1.s, p0/m, z17.s, z31.s
    fmad    z2.s, p0/m, z18.s, z31.s
    fmad    z3.s, p0/m, z19.s, z31.s
    fmad    z4.s, p0/m, z20.s, z31.s
    fmad    z5.s, p0/m, z21.s, z31.s
    fmad    z6.s, p0/m, z22.s, z31.s
    fmad    z7.s, p0/m, z23.s, z31.s

    // p = p*r + 1.0
    fmad    z0.s, p0/m, z16.s, z31.s
    fmad    z1.s, p0/m, z17.s, z31.s
    fmad    z2.s, p0/m, z18.s, z31.s
    fmad    z3.s, p0/m, z19.s, z31.s
    fmad    z4.s, p0/m, z20.s, z31.s
    fmad    z5.s, p0/m, z21.s, z31.s
    fmad    z6.s, p0/m, z22.s, z31.s
    fmad    z7.s, p0/m, z23.s, z31.s

    // Multiply by 2^n
    fmul    z0.s, z0.s, z8.s
    fmul    z1.s, z1.s, z9.s
    fmul    z2.s, z2.s, z10.s
    fmul    z3.s, z3.s, z11.s
    fmul    z4.s, z4.s, z12.s
    fmul    z5.s, z5.s, z13.s
    fmul    z6.s, z6.s, z14.s
    fmul    z7.s, z7.s, z15.s

    st1w    z0.s, p0, [x1]
    st1w    z1.s, p0, [x1, #1, mul vl]
    st1w    z2.s, p0, [x1, #2, mul vl]
    st1w    z3.s, p0, [x1, #3, mul vl]
    st1w    z4.s, p0, [x1, #4, mul vl]
    st1w    z5.s, p0, [x1, #5, mul vl]
    st1w    z6.s, p0, [x1, #6, mul vl]
    st1w    z7.s, p0, [x1, #7, mul vl]

    add     x0, x0, x4, lsl #2
    add     x1, x1, x4, lsl #2
    sub     x5, x5, x4
    b       .Lloop8

.Ltail8:
    cbz     x5, .Ldone8
    whilelt p1.s, xzr, x5
    ld1w    z0.s, p1/z, [x0]

    fmul    z4.s, z0.s, z28.s
    frintn  z4.s, p0/m, z4.s
    fmls   z0.s, p0/m, z4.s, z29.s
    fcvtzs  z4.s, p0/m, z4.s
    mov     z31.s, #127
    add     z4.s, z4.s, z31.s
    lsl     z4.s, z4.s, #23

    movz    w10, #0xaaab
    movk    w10, #0x3d2a, lsl #16
    dup     z8.s, w10
    movz    w10, #0xaaab
    movk    w10, #0x3e2a, lsl #16
    dup     z31.s, w10
    fmad    z8.s, p0/m, z0.s, z31.s
    fmov    z31.s, #0.5
    fmad    z8.s, p0/m, z0.s, z31.s
    fmov    z31.s, #1.0
    fmad    z8.s, p0/m, z0.s, z31.s
    fmad    z8.s, p0/m, z0.s, z31.s
    fmul    z8.s, z8.s, z4.s

    st1w    z8.s, p1, [x1]

.Ldone8:
    ret
    .size exp_vec_unroll8, .-exp_vec_unroll8


// ============================================
// exp_vec_unroll10: Process N floats with unroll 10
// ============================================
    .align 6
    .global exp_vec_unroll10
    .type exp_vec_unroll10, %function
exp_vec_unroll10:
    stp     x19, x20, [sp, #-16]!

    ptrue   p0.s

    movz    w10, #0xaa3b
    movk    w10, #0x3fb8, lsl #16
    dup     z28.s, w10              // 1/ln(2)

    movz    w10, #0x7218
    movk    w10, #0x3f31, lsl #16
    dup     z29.s, w10              // ln(2)

    cntw    x3
    mov     x4, #10
    mul     x4, x3, x4              // 10 * vl

    mov     x5, x2

.Lloop10:
    cmp     x5, x4
    b.lt    .Ltail10

    // Process 10 vectors in two batches of 5 to manage register pressure
    // First batch: vectors 0-4
    ld1w    z0.s, p0/z, [x0]
    ld1w    z1.s, p0/z, [x0, #1, mul vl]
    ld1w    z2.s, p0/z, [x0, #2, mul vl]
    ld1w    z3.s, p0/z, [x0, #3, mul vl]
    ld1w    z4.s, p0/z, [x0, #4, mul vl]

    // Compute exp for first 5
    fmul    z5.s, z0.s, z28.s
    fmul    z6.s, z1.s, z28.s
    fmul    z7.s, z2.s, z28.s
    fmul    z8.s, z3.s, z28.s
    fmul    z9.s, z4.s, z28.s

    frintn  z5.s, p0/m, z5.s
    frintn  z6.s, p0/m, z6.s
    frintn  z7.s, p0/m, z7.s
    frintn  z8.s, p0/m, z8.s
    frintn  z9.s, p0/m, z9.s

    fmls   z0.s, p0/m, z5.s, z29.s
    fmls   z1.s, p0/m, z6.s, z29.s
    fmls   z2.s, p0/m, z7.s, z29.s
    fmls   z3.s, p0/m, z8.s, z29.s
    fmls   z4.s, p0/m, z9.s, z29.s

    // 2^n for first 5
    fcvtzs  z5.s, p0/m, z5.s
    fcvtzs  z6.s, p0/m, z6.s
    fcvtzs  z7.s, p0/m, z7.s
    fcvtzs  z8.s, p0/m, z8.s
    fcvtzs  z9.s, p0/m, z9.s

    mov     z31.s, #127
    add     z5.s, z5.s, z31.s
    add     z6.s, z6.s, z31.s
    add     z7.s, z7.s, z31.s
    add     z8.s, z8.s, z31.s
    add     z9.s, z9.s, z31.s

    lsl     z5.s, z5.s, #23
    lsl     z6.s, z6.s, #23
    lsl     z7.s, z7.s, #23
    lsl     z8.s, z8.s, #23
    lsl     z9.s, z9.s, #23

    // Polynomial for r0-r4 (r values in z0-z4)
    movz    w10, #0xaaab
    movk    w10, #0x3d2a, lsl #16
    dup     z10.s, w10
    dup     z11.s, w10
    dup     z12.s, w10
    dup     z13.s, w10
    dup     z14.s, w10

    movz    w10, #0xaaab
    movk    w10, #0x3e2a, lsl #16
    dup     z31.s, w10

    fmad    z10.s, p0/m, z0.s, z31.s
    fmad    z11.s, p0/m, z1.s, z31.s
    fmad    z12.s, p0/m, z2.s, z31.s
    fmad    z13.s, p0/m, z3.s, z31.s
    fmad    z14.s, p0/m, z4.s, z31.s

    fmov    z31.s, #0.5
    fmad    z10.s, p0/m, z0.s, z31.s
    fmad    z11.s, p0/m, z1.s, z31.s
    fmad    z12.s, p0/m, z2.s, z31.s
    fmad    z13.s, p0/m, z3.s, z31.s
    fmad    z14.s, p0/m, z4.s, z31.s

    fmov    z31.s, #1.0
    fmad    z10.s, p0/m, z0.s, z31.s
    fmad    z11.s, p0/m, z1.s, z31.s
    fmad    z12.s, p0/m, z2.s, z31.s
    fmad    z13.s, p0/m, z3.s, z31.s
    fmad    z14.s, p0/m, z4.s, z31.s

    fmad    z10.s, p0/m, z0.s, z31.s
    fmad    z11.s, p0/m, z1.s, z31.s
    fmad    z12.s, p0/m, z2.s, z31.s
    fmad    z13.s, p0/m, z3.s, z31.s
    fmad    z14.s, p0/m, z4.s, z31.s

    fmul    z10.s, z10.s, z5.s
    fmul    z11.s, z11.s, z6.s
    fmul    z12.s, z12.s, z7.s
    fmul    z13.s, z13.s, z8.s
    fmul    z14.s, z14.s, z9.s

    st1w    z10.s, p0, [x1]
    st1w    z11.s, p0, [x1, #1, mul vl]
    st1w    z12.s, p0, [x1, #2, mul vl]
    st1w    z13.s, p0, [x1, #3, mul vl]
    st1w    z14.s, p0, [x1, #4, mul vl]

    // Second batch: vectors 5-9
    ld1w    z0.s, p0/z, [x0, #5, mul vl]
    ld1w    z1.s, p0/z, [x0, #6, mul vl]
    ld1w    z2.s, p0/z, [x0, #7, mul vl]

    // Need to compute offset for vectors 8,9
    cntw    x19
    lsl     x19, x19, #5            // 8 * vl * 4 bytes
    add     x20, x0, x19
    ld1w    z3.s, p0/z, [x20]
    cntw    x19
    lsl     x19, x19, #2            // vl * 4
    add     x20, x20, x19
    ld1w    z4.s, p0/z, [x20]

    fmul    z5.s, z0.s, z28.s
    fmul    z6.s, z1.s, z28.s
    fmul    z7.s, z2.s, z28.s
    fmul    z8.s, z3.s, z28.s
    fmul    z9.s, z4.s, z28.s

    frintn  z5.s, p0/m, z5.s
    frintn  z6.s, p0/m, z6.s
    frintn  z7.s, p0/m, z7.s
    frintn  z8.s, p0/m, z8.s
    frintn  z9.s, p0/m, z9.s

    fmls   z0.s, p0/m, z5.s, z29.s
    fmls   z1.s, p0/m, z6.s, z29.s
    fmls   z2.s, p0/m, z7.s, z29.s
    fmls   z3.s, p0/m, z8.s, z29.s
    fmls   z4.s, p0/m, z9.s, z29.s

    fcvtzs  z5.s, p0/m, z5.s
    fcvtzs  z6.s, p0/m, z6.s
    fcvtzs  z7.s, p0/m, z7.s
    fcvtzs  z8.s, p0/m, z8.s
    fcvtzs  z9.s, p0/m, z9.s

    mov     z31.s, #127
    add     z5.s, z5.s, z31.s
    add     z6.s, z6.s, z31.s
    add     z7.s, z7.s, z31.s
    add     z8.s, z8.s, z31.s
    add     z9.s, z9.s, z31.s

    lsl     z5.s, z5.s, #23
    lsl     z6.s, z6.s, #23
    lsl     z7.s, z7.s, #23
    lsl     z8.s, z8.s, #23
    lsl     z9.s, z9.s, #23

    movz    w10, #0xaaab
    movk    w10, #0x3d2a, lsl #16
    dup     z10.s, w10
    dup     z11.s, w10
    dup     z12.s, w10
    dup     z13.s, w10
    dup     z14.s, w10

    movz    w10, #0xaaab
    movk    w10, #0x3e2a, lsl #16
    dup     z31.s, w10

    fmad    z10.s, p0/m, z0.s, z31.s
    fmad    z11.s, p0/m, z1.s, z31.s
    fmad    z12.s, p0/m, z2.s, z31.s
    fmad    z13.s, p0/m, z3.s, z31.s
    fmad    z14.s, p0/m, z4.s, z31.s

    fmov    z31.s, #0.5
    fmad    z10.s, p0/m, z0.s, z31.s
    fmad    z11.s, p0/m, z1.s, z31.s
    fmad    z12.s, p0/m, z2.s, z31.s
    fmad    z13.s, p0/m, z3.s, z31.s
    fmad    z14.s, p0/m, z4.s, z31.s

    fmov    z31.s, #1.0
    fmad    z10.s, p0/m, z0.s, z31.s
    fmad    z11.s, p0/m, z1.s, z31.s
    fmad    z12.s, p0/m, z2.s, z31.s
    fmad    z13.s, p0/m, z3.s, z31.s
    fmad    z14.s, p0/m, z4.s, z31.s

    fmad    z10.s, p0/m, z0.s, z31.s
    fmad    z11.s, p0/m, z1.s, z31.s
    fmad    z12.s, p0/m, z2.s, z31.s
    fmad    z13.s, p0/m, z3.s, z31.s
    fmad    z14.s, p0/m, z4.s, z31.s

    fmul    z10.s, z10.s, z5.s
    fmul    z11.s, z11.s, z6.s
    fmul    z12.s, z12.s, z7.s
    fmul    z13.s, z13.s, z8.s
    fmul    z14.s, z14.s, z9.s

    st1w    z10.s, p0, [x1, #5, mul vl]
    st1w    z11.s, p0, [x1, #6, mul vl]
    st1w    z12.s, p0, [x1, #7, mul vl]

    cntw    x19
    lsl     x19, x19, #5            // 8 * vl * 4
    add     x20, x1, x19
    st1w    z13.s, p0, [x20]
    cntw    x19
    lsl     x19, x19, #2
    add     x20, x20, x19
    st1w    z14.s, p0, [x20]

    add     x0, x0, x4, lsl #2
    add     x1, x1, x4, lsl #2
    sub     x5, x5, x4
    b       .Lloop10

.Ltail10:
    cbz     x5, .Ldone10
    whilelt p1.s, xzr, x5
    ld1w    z0.s, p1/z, [x0]

    fmul    z4.s, z0.s, z28.s
    frintn  z4.s, p0/m, z4.s
    fmls   z0.s, p0/m, z4.s, z29.s
    fcvtzs  z4.s, p0/m, z4.s
    mov     z31.s, #127
    add     z4.s, z4.s, z31.s
    lsl     z4.s, z4.s, #23

    movz    w10, #0xaaab
    movk    w10, #0x3d2a, lsl #16
    dup     z8.s, w10
    movz    w10, #0xaaab
    movk    w10, #0x3e2a, lsl #16
    dup     z31.s, w10
    fmad    z8.s, p0/m, z0.s, z31.s
    fmov    z31.s, #0.5
    fmad    z8.s, p0/m, z0.s, z31.s
    fmov    z31.s, #1.0
    fmad    z8.s, p0/m, z0.s, z31.s
    fmad    z8.s, p0/m, z0.s, z31.s
    fmul    z8.s, z8.s, z4.s

    st1w    z8.s, p1, [x1]

.Ldone10:
    ldp     x19, x20, [sp], #16
    ret
    .size exp_vec_unroll10, .-exp_vec_unroll10


// ============================================
// FP64 (double precision) exp kernels
// ============================================

// Constants for FP64:
// 1/ln(2) = 1.4426950408889634 = 0x3ff71547652b82fe
// ln(2)   = 0.6931471805599453 = 0x3fe62e42fefa39ef
// 1/24    = 0.041666666666666664 = 0x3fa5555555555555
// 1/6     = 0.16666666666666666 = 0x3fc5555555555555
// IEEE754 double: exponent bias = 1023, mantissa bits = 52

// ============================================
// exp_vec_f64_unroll4: Process N doubles with unroll 4
// Args: x0 = input array, x1 = output array, x2 = count
// ============================================
    .align 6
    .global exp_vec_f64_unroll4
    .type exp_vec_f64_unroll4, %function
exp_vec_f64_unroll4:
    ptrue   p0.d

    // Load constants: 1/ln(2)
    movz    x10, #0x82fe
    movk    x10, #0x652b, lsl #16
    movk    x10, #0x1547, lsl #32
    movk    x10, #0x3ff7, lsl #48
    dup     z28.d, x10

    // ln(2)
    movz    x10, #0x39ef
    movk    x10, #0xfefa, lsl #16
    movk    x10, #0x2e42, lsl #32
    movk    x10, #0x3fe6, lsl #48
    dup     z29.d, x10

    cntd    x3                      // elements per vector (8 for 512-bit)
    lsl     x4, x3, #2              // x4 = 4 * vl

    mov     x5, x2                  // remaining count

.Lloop_f64_4:
    cmp     x5, x4
    b.lt    .Ltail_f64_4

    // Load 4 vectors
    ld1d    z0.d, p0/z, [x0]
    ld1d    z1.d, p0/z, [x0, #1, mul vl]
    ld1d    z2.d, p0/z, [x0, #2, mul vl]
    ld1d    z3.d, p0/z, [x0, #3, mul vl]

    // Range reduction: n = round(x / ln(2))
    fmul    z4.d, z0.d, z28.d
    fmul    z5.d, z1.d, z28.d
    fmul    z6.d, z2.d, z28.d
    fmul    z7.d, z3.d, z28.d

    frintn  z4.d, p0/m, z4.d
    frintn  z5.d, p0/m, z5.d
    frintn  z6.d, p0/m, z6.d
    frintn  z7.d, p0/m, z7.d

    // r = x - n * ln(2)
    fmls    z0.d, p0/m, z4.d, z29.d
    fmls    z1.d, p0/m, z5.d, z29.d
    fmls    z2.d, p0/m, z6.d, z29.d
    fmls    z3.d, p0/m, z7.d, z29.d

    // Convert n to integer
    fcvtzs  z4.d, p0/m, z4.d
    fcvtzs  z5.d, p0/m, z5.d
    fcvtzs  z6.d, p0/m, z6.d
    fcvtzs  z7.d, p0/m, z7.d

    // 2^n via IEEE754: ((n + 1023) << 52)
    mov     z31.d, #1023
    add     z4.d, z4.d, z31.d
    add     z5.d, z5.d, z31.d
    add     z6.d, z6.d, z31.d
    add     z7.d, z7.d, z31.d

    lsl     z4.d, z4.d, #52
    lsl     z5.d, z5.d, #52
    lsl     z6.d, z6.d, #52
    lsl     z7.d, z7.d, #52

    // Polynomial for exp(r): 1 + r(1 + r(0.5 + r(1/6 + r(1/24 + r/120))))
    // Using 5-term polynomial for better FP64 accuracy
    // Start with p = 1/120
    movz    x10, #0x1111
    movk    x10, #0x1111, lsl #16
    movk    x10, #0x1111, lsl #32
    movk    x10, #0x3f81, lsl #48   // 1/120 ≈ 0.008333
    dup     z8.d, x10
    dup     z9.d, x10
    dup     z10.d, x10
    dup     z11.d, x10

    // p = p*r + 1/24
    movz    x10, #0x5555
    movk    x10, #0x5555, lsl #16
    movk    x10, #0x5555, lsl #32
    movk    x10, #0x3fa5, lsl #48
    dup     z31.d, x10
    fmad    z8.d, p0/m, z0.d, z31.d
    fmad    z9.d, p0/m, z1.d, z31.d
    fmad    z10.d, p0/m, z2.d, z31.d
    fmad    z11.d, p0/m, z3.d, z31.d

    // p = p*r + 1/6
    movz    x10, #0x5555
    movk    x10, #0x5555, lsl #16
    movk    x10, #0x5555, lsl #32
    movk    x10, #0x3fc5, lsl #48
    dup     z31.d, x10
    fmad    z8.d, p0/m, z0.d, z31.d
    fmad    z9.d, p0/m, z1.d, z31.d
    fmad    z10.d, p0/m, z2.d, z31.d
    fmad    z11.d, p0/m, z3.d, z31.d

    // p = p*r + 0.5
    fmov    z31.d, #0.5
    fmad    z8.d, p0/m, z0.d, z31.d
    fmad    z9.d, p0/m, z1.d, z31.d
    fmad    z10.d, p0/m, z2.d, z31.d
    fmad    z11.d, p0/m, z3.d, z31.d

    // p = p*r + 1.0
    fmov    z31.d, #1.0
    fmad    z8.d, p0/m, z0.d, z31.d
    fmad    z9.d, p0/m, z1.d, z31.d
    fmad    z10.d, p0/m, z2.d, z31.d
    fmad    z11.d, p0/m, z3.d, z31.d

    // p = p*r + 1.0
    fmad    z8.d, p0/m, z0.d, z31.d
    fmad    z9.d, p0/m, z1.d, z31.d
    fmad    z10.d, p0/m, z2.d, z31.d
    fmad    z11.d, p0/m, z3.d, z31.d

    // exp(x) = 2^n * exp(r)
    fmul    z8.d, z8.d, z4.d
    fmul    z9.d, z9.d, z5.d
    fmul    z10.d, z10.d, z6.d
    fmul    z11.d, z11.d, z7.d

    // Store results
    st1d    z8.d, p0, [x1]
    st1d    z9.d, p0, [x1, #1, mul vl]
    st1d    z10.d, p0, [x1, #2, mul vl]
    st1d    z11.d, p0, [x1, #3, mul vl]

    add     x0, x0, x4, lsl #3      // advance by 4*vl doubles
    add     x1, x1, x4, lsl #3
    sub     x5, x5, x4
    b       .Lloop_f64_4

.Ltail_f64_4:
    cbz     x5, .Ldone_f64_4
    whilelt p1.d, xzr, x5
    ld1d    z0.d, p1/z, [x0]

    fmul    z4.d, z0.d, z28.d
    frintn  z4.d, p0/m, z4.d
    fmls    z0.d, p0/m, z4.d, z29.d
    fcvtzs  z4.d, p0/m, z4.d
    mov     z31.d, #1023
    add     z4.d, z4.d, z31.d
    lsl     z4.d, z4.d, #52

    movz    x10, #0x1111
    movk    x10, #0x1111, lsl #16
    movk    x10, #0x1111, lsl #32
    movk    x10, #0x3f81, lsl #48
    dup     z8.d, x10

    movz    x10, #0x5555
    movk    x10, #0x5555, lsl #16
    movk    x10, #0x5555, lsl #32
    movk    x10, #0x3fa5, lsl #48
    dup     z31.d, x10
    fmad    z8.d, p0/m, z0.d, z31.d

    movz    x10, #0x5555
    movk    x10, #0x5555, lsl #16
    movk    x10, #0x5555, lsl #32
    movk    x10, #0x3fc5, lsl #48
    dup     z31.d, x10
    fmad    z8.d, p0/m, z0.d, z31.d

    fmov    z31.d, #0.5
    fmad    z8.d, p0/m, z0.d, z31.d
    fmov    z31.d, #1.0
    fmad    z8.d, p0/m, z0.d, z31.d
    fmad    z8.d, p0/m, z0.d, z31.d
    fmul    z8.d, z8.d, z4.d

    st1d    z8.d, p1, [x1]

.Ldone_f64_4:
    ret
    .size exp_vec_f64_unroll4, .-exp_vec_f64_unroll4


// ============================================
// exp_vec_f64_unroll8: Process N doubles with unroll 8
// Args: x0 = input array, x1 = output array, x2 = count
// ============================================
    .align 6
    .global exp_vec_f64_unroll8
    .type exp_vec_f64_unroll8, %function
exp_vec_f64_unroll8:
    ptrue   p0.d

    // Load constants
    movz    x10, #0x82fe
    movk    x10, #0x652b, lsl #16
    movk    x10, #0x1547, lsl #32
    movk    x10, #0x3ff7, lsl #48
    dup     z28.d, x10              // 1/ln(2)

    movz    x10, #0x39ef
    movk    x10, #0xfefa, lsl #16
    movk    x10, #0x2e42, lsl #32
    movk    x10, #0x3fe6, lsl #48
    dup     z29.d, x10              // ln(2)

    cntd    x3
    lsl     x4, x3, #3              // 8 * vl

    mov     x5, x2

.Lloop_f64_8:
    cmp     x5, x4
    b.lt    .Ltail_f64_8

    // Load 8 vectors and process in two batches
    // First batch: 0-3
    ld1d    z0.d, p0/z, [x0]
    ld1d    z1.d, p0/z, [x0, #1, mul vl]
    ld1d    z2.d, p0/z, [x0, #2, mul vl]
    ld1d    z3.d, p0/z, [x0, #3, mul vl]

    fmul    z4.d, z0.d, z28.d
    fmul    z5.d, z1.d, z28.d
    fmul    z6.d, z2.d, z28.d
    fmul    z7.d, z3.d, z28.d

    frintn  z4.d, p0/m, z4.d
    frintn  z5.d, p0/m, z5.d
    frintn  z6.d, p0/m, z6.d
    frintn  z7.d, p0/m, z7.d

    fmls    z0.d, p0/m, z4.d, z29.d
    fmls    z1.d, p0/m, z5.d, z29.d
    fmls    z2.d, p0/m, z6.d, z29.d
    fmls    z3.d, p0/m, z7.d, z29.d

    fcvtzs  z4.d, p0/m, z4.d
    fcvtzs  z5.d, p0/m, z5.d
    fcvtzs  z6.d, p0/m, z6.d
    fcvtzs  z7.d, p0/m, z7.d

    mov     z31.d, #1023
    add     z4.d, z4.d, z31.d
    add     z5.d, z5.d, z31.d
    add     z6.d, z6.d, z31.d
    add     z7.d, z7.d, z31.d

    lsl     z4.d, z4.d, #52
    lsl     z5.d, z5.d, #52
    lsl     z6.d, z6.d, #52
    lsl     z7.d, z7.d, #52

    // Polynomial
    movz    x10, #0x1111
    movk    x10, #0x1111, lsl #16
    movk    x10, #0x1111, lsl #32
    movk    x10, #0x3f81, lsl #48
    dup     z8.d, x10
    dup     z9.d, x10
    dup     z10.d, x10
    dup     z11.d, x10

    movz    x10, #0x5555
    movk    x10, #0x5555, lsl #16
    movk    x10, #0x5555, lsl #32
    movk    x10, #0x3fa5, lsl #48
    dup     z31.d, x10
    fmad    z8.d, p0/m, z0.d, z31.d
    fmad    z9.d, p0/m, z1.d, z31.d
    fmad    z10.d, p0/m, z2.d, z31.d
    fmad    z11.d, p0/m, z3.d, z31.d

    movz    x10, #0x5555
    movk    x10, #0x5555, lsl #16
    movk    x10, #0x5555, lsl #32
    movk    x10, #0x3fc5, lsl #48
    dup     z31.d, x10
    fmad    z8.d, p0/m, z0.d, z31.d
    fmad    z9.d, p0/m, z1.d, z31.d
    fmad    z10.d, p0/m, z2.d, z31.d
    fmad    z11.d, p0/m, z3.d, z31.d

    fmov    z31.d, #0.5
    fmad    z8.d, p0/m, z0.d, z31.d
    fmad    z9.d, p0/m, z1.d, z31.d
    fmad    z10.d, p0/m, z2.d, z31.d
    fmad    z11.d, p0/m, z3.d, z31.d

    fmov    z31.d, #1.0
    fmad    z8.d, p0/m, z0.d, z31.d
    fmad    z9.d, p0/m, z1.d, z31.d
    fmad    z10.d, p0/m, z2.d, z31.d
    fmad    z11.d, p0/m, z3.d, z31.d

    fmad    z8.d, p0/m, z0.d, z31.d
    fmad    z9.d, p0/m, z1.d, z31.d
    fmad    z10.d, p0/m, z2.d, z31.d
    fmad    z11.d, p0/m, z3.d, z31.d

    fmul    z8.d, z8.d, z4.d
    fmul    z9.d, z9.d, z5.d
    fmul    z10.d, z10.d, z6.d
    fmul    z11.d, z11.d, z7.d

    st1d    z8.d, p0, [x1]
    st1d    z9.d, p0, [x1, #1, mul vl]
    st1d    z10.d, p0, [x1, #2, mul vl]
    st1d    z11.d, p0, [x1, #3, mul vl]

    // Second batch: 4-7
    ld1d    z0.d, p0/z, [x0, #4, mul vl]
    ld1d    z1.d, p0/z, [x0, #5, mul vl]
    ld1d    z2.d, p0/z, [x0, #6, mul vl]
    ld1d    z3.d, p0/z, [x0, #7, mul vl]

    fmul    z4.d, z0.d, z28.d
    fmul    z5.d, z1.d, z28.d
    fmul    z6.d, z2.d, z28.d
    fmul    z7.d, z3.d, z28.d

    frintn  z4.d, p0/m, z4.d
    frintn  z5.d, p0/m, z5.d
    frintn  z6.d, p0/m, z6.d
    frintn  z7.d, p0/m, z7.d

    fmls    z0.d, p0/m, z4.d, z29.d
    fmls    z1.d, p0/m, z5.d, z29.d
    fmls    z2.d, p0/m, z6.d, z29.d
    fmls    z3.d, p0/m, z7.d, z29.d

    fcvtzs  z4.d, p0/m, z4.d
    fcvtzs  z5.d, p0/m, z5.d
    fcvtzs  z6.d, p0/m, z6.d
    fcvtzs  z7.d, p0/m, z7.d

    mov     z31.d, #1023
    add     z4.d, z4.d, z31.d
    add     z5.d, z5.d, z31.d
    add     z6.d, z6.d, z31.d
    add     z7.d, z7.d, z31.d

    lsl     z4.d, z4.d, #52
    lsl     z5.d, z5.d, #52
    lsl     z6.d, z6.d, #52
    lsl     z7.d, z7.d, #52

    movz    x10, #0x1111
    movk    x10, #0x1111, lsl #16
    movk    x10, #0x1111, lsl #32
    movk    x10, #0x3f81, lsl #48
    dup     z8.d, x10
    dup     z9.d, x10
    dup     z10.d, x10
    dup     z11.d, x10

    movz    x10, #0x5555
    movk    x10, #0x5555, lsl #16
    movk    x10, #0x5555, lsl #32
    movk    x10, #0x3fa5, lsl #48
    dup     z31.d, x10
    fmad    z8.d, p0/m, z0.d, z31.d
    fmad    z9.d, p0/m, z1.d, z31.d
    fmad    z10.d, p0/m, z2.d, z31.d
    fmad    z11.d, p0/m, z3.d, z31.d

    movz    x10, #0x5555
    movk    x10, #0x5555, lsl #16
    movk    x10, #0x5555, lsl #32
    movk    x10, #0x3fc5, lsl #48
    dup     z31.d, x10
    fmad    z8.d, p0/m, z0.d, z31.d
    fmad    z9.d, p0/m, z1.d, z31.d
    fmad    z10.d, p0/m, z2.d, z31.d
    fmad    z11.d, p0/m, z3.d, z31.d

    fmov    z31.d, #0.5
    fmad    z8.d, p0/m, z0.d, z31.d
    fmad    z9.d, p0/m, z1.d, z31.d
    fmad    z10.d, p0/m, z2.d, z31.d
    fmad    z11.d, p0/m, z3.d, z31.d

    fmov    z31.d, #1.0
    fmad    z8.d, p0/m, z0.d, z31.d
    fmad    z9.d, p0/m, z1.d, z31.d
    fmad    z10.d, p0/m, z2.d, z31.d
    fmad    z11.d, p0/m, z3.d, z31.d

    fmad    z8.d, p0/m, z0.d, z31.d
    fmad    z9.d, p0/m, z1.d, z31.d
    fmad    z10.d, p0/m, z2.d, z31.d
    fmad    z11.d, p0/m, z3.d, z31.d

    fmul    z8.d, z8.d, z4.d
    fmul    z9.d, z9.d, z5.d
    fmul    z10.d, z10.d, z6.d
    fmul    z11.d, z11.d, z7.d

    st1d    z8.d, p0, [x1, #4, mul vl]
    st1d    z9.d, p0, [x1, #5, mul vl]
    st1d    z10.d, p0, [x1, #6, mul vl]
    st1d    z11.d, p0, [x1, #7, mul vl]

    add     x0, x0, x4, lsl #3
    add     x1, x1, x4, lsl #3
    sub     x5, x5, x4
    b       .Lloop_f64_8

.Ltail_f64_8:
    cbz     x5, .Ldone_f64_8
    whilelt p1.d, xzr, x5
    ld1d    z0.d, p1/z, [x0]

    fmul    z4.d, z0.d, z28.d
    frintn  z4.d, p0/m, z4.d
    fmls    z0.d, p0/m, z4.d, z29.d
    fcvtzs  z4.d, p0/m, z4.d
    mov     z31.d, #1023
    add     z4.d, z4.d, z31.d
    lsl     z4.d, z4.d, #52

    movz    x10, #0x1111
    movk    x10, #0x1111, lsl #16
    movk    x10, #0x1111, lsl #32
    movk    x10, #0x3f81, lsl #48
    dup     z8.d, x10

    movz    x10, #0x5555
    movk    x10, #0x5555, lsl #16
    movk    x10, #0x5555, lsl #32
    movk    x10, #0x3fa5, lsl #48
    dup     z31.d, x10
    fmad    z8.d, p0/m, z0.d, z31.d

    movz    x10, #0x5555
    movk    x10, #0x5555, lsl #16
    movk    x10, #0x5555, lsl #32
    movk    x10, #0x3fc5, lsl #48
    dup     z31.d, x10
    fmad    z8.d, p0/m, z0.d, z31.d

    fmov    z31.d, #0.5
    fmad    z8.d, p0/m, z0.d, z31.d
    fmov    z31.d, #1.0
    fmad    z8.d, p0/m, z0.d, z31.d
    fmad    z8.d, p0/m, z0.d, z31.d
    fmul    z8.d, z8.d, z4.d

    st1d    z8.d, p1, [x1]

.Ldone_f64_8:
    ret
    .size exp_vec_f64_unroll8, .-exp_vec_f64_unroll8


// ============================================
// FP16 (half precision) exp kernels
// ============================================

// Constants for FP16:
// IEEE754 half: exponent bias = 15, mantissa bits = 10
// We use simpler 3-term polynomial due to limited precision

// ============================================
// exp_vec_f16_unroll4: Process N halfs with unroll 4
// Args: x0 = input array, x1 = output array, x2 = count
// ============================================
    .align 6
    .global exp_vec_f16_unroll4
    .type exp_vec_f16_unroll4, %function
exp_vec_f16_unroll4:
    ptrue   p0.h

    // Load constants (as FP16)
    // 1/ln(2) = 1.4426950... ≈ 0x3dc5 in FP16
    mov     w10, #0x3dc5
    dup     z28.h, w10

    // ln(2) = 0.6931471... ≈ 0x398c in FP16
    mov     w10, #0x398c
    dup     z29.h, w10

    cnth    x3                      // elements per vector (32 for 512-bit)
    lsl     x4, x3, #2              // x4 = 4 * vl

    mov     x5, x2

.Lloop_f16_4:
    cmp     x5, x4
    b.lt    .Ltail_f16_4

    // Load 4 vectors
    ld1h    z0.h, p0/z, [x0]
    ld1h    z1.h, p0/z, [x0, #1, mul vl]
    ld1h    z2.h, p0/z, [x0, #2, mul vl]
    ld1h    z3.h, p0/z, [x0, #3, mul vl]

    // Range reduction
    fmul    z4.h, z0.h, z28.h
    fmul    z5.h, z1.h, z28.h
    fmul    z6.h, z2.h, z28.h
    fmul    z7.h, z3.h, z28.h

    frintn  z4.h, p0/m, z4.h
    frintn  z5.h, p0/m, z5.h
    frintn  z6.h, p0/m, z6.h
    frintn  z7.h, p0/m, z7.h

    // r = x - n * ln(2)
    fmls    z0.h, p0/m, z4.h, z29.h
    fmls    z1.h, p0/m, z5.h, z29.h
    fmls    z2.h, p0/m, z6.h, z29.h
    fmls    z3.h, p0/m, z7.h, z29.h

    // Convert n to integer
    fcvtzs  z4.h, p0/m, z4.h
    fcvtzs  z5.h, p0/m, z5.h
    fcvtzs  z6.h, p0/m, z6.h
    fcvtzs  z7.h, p0/m, z7.h

    // 2^n via IEEE754: ((n + 15) << 10)
    mov     z31.h, #15
    add     z4.h, z4.h, z31.h
    add     z5.h, z5.h, z31.h
    add     z6.h, z6.h, z31.h
    add     z7.h, z7.h, z31.h

    lsl     z4.h, z4.h, #10
    lsl     z5.h, z5.h, #10
    lsl     z6.h, z6.h, #10
    lsl     z7.h, z7.h, #10

    // Polynomial for exp(r): 1 + r(1 + r(0.5 + r/6))
    // Using 3-term for FP16 (sufficient precision)
    // Start with p = 1/6 ≈ 0x3155 in FP16
    mov     w10, #0x3155
    dup     z8.h, w10
    dup     z9.h, w10
    dup     z10.h, w10
    dup     z11.h, w10

    // p = p*r + 0.5
    fmov    z31.h, #0.5
    fmad    z8.h, p0/m, z0.h, z31.h
    fmad    z9.h, p0/m, z1.h, z31.h
    fmad    z10.h, p0/m, z2.h, z31.h
    fmad    z11.h, p0/m, z3.h, z31.h

    // p = p*r + 1.0
    fmov    z31.h, #1.0
    fmad    z8.h, p0/m, z0.h, z31.h
    fmad    z9.h, p0/m, z1.h, z31.h
    fmad    z10.h, p0/m, z2.h, z31.h
    fmad    z11.h, p0/m, z3.h, z31.h

    // p = p*r + 1.0
    fmad    z8.h, p0/m, z0.h, z31.h
    fmad    z9.h, p0/m, z1.h, z31.h
    fmad    z10.h, p0/m, z2.h, z31.h
    fmad    z11.h, p0/m, z3.h, z31.h

    // exp(x) = 2^n * exp(r)
    fmul    z8.h, z8.h, z4.h
    fmul    z9.h, z9.h, z5.h
    fmul    z10.h, z10.h, z6.h
    fmul    z11.h, z11.h, z7.h

    // Store results
    st1h    z8.h, p0, [x1]
    st1h    z9.h, p0, [x1, #1, mul vl]
    st1h    z10.h, p0, [x1, #2, mul vl]
    st1h    z11.h, p0, [x1, #3, mul vl]

    add     x0, x0, x4, lsl #1      // advance by 4*vl halfs
    add     x1, x1, x4, lsl #1
    sub     x5, x5, x4
    b       .Lloop_f16_4

.Ltail_f16_4:
    cbz     x5, .Ldone_f16_4
    whilelt p1.h, xzr, x5
    ld1h    z0.h, p1/z, [x0]

    fmul    z4.h, z0.h, z28.h
    frintn  z4.h, p0/m, z4.h
    fmls    z0.h, p0/m, z4.h, z29.h
    fcvtzs  z4.h, p0/m, z4.h
    mov     z31.h, #15
    add     z4.h, z4.h, z31.h
    lsl     z4.h, z4.h, #10

    mov     w10, #0x3155
    dup     z8.h, w10
    fmov    z31.h, #0.5
    fmad    z8.h, p0/m, z0.h, z31.h
    fmov    z31.h, #1.0
    fmad    z8.h, p0/m, z0.h, z31.h
    fmad    z8.h, p0/m, z0.h, z31.h
    fmul    z8.h, z8.h, z4.h

    st1h    z8.h, p1, [x1]

.Ldone_f16_4:
    ret
    .size exp_vec_f16_unroll4, .-exp_vec_f16_unroll4


// ============================================
// exp_vec_f16_unroll8: Process N halfs with unroll 8
// Args: x0 = input array, x1 = output array, x2 = count
// ============================================
    .align 6
    .global exp_vec_f16_unroll8
    .type exp_vec_f16_unroll8, %function
exp_vec_f16_unroll8:
    ptrue   p0.h

    mov     w10, #0x3dc5
    dup     z28.h, w10              // 1/ln(2)

    mov     w10, #0x398c
    dup     z29.h, w10              // ln(2)

    cnth    x3
    lsl     x4, x3, #3              // 8 * vl

    mov     x5, x2

.Lloop_f16_8:
    cmp     x5, x4
    b.lt    .Ltail_f16_8

    // Process 8 vectors
    ld1h    z0.h, p0/z, [x0]
    ld1h    z1.h, p0/z, [x0, #1, mul vl]
    ld1h    z2.h, p0/z, [x0, #2, mul vl]
    ld1h    z3.h, p0/z, [x0, #3, mul vl]
    ld1h    z16.h, p0/z, [x0, #4, mul vl]
    ld1h    z17.h, p0/z, [x0, #5, mul vl]
    ld1h    z18.h, p0/z, [x0, #6, mul vl]
    ld1h    z19.h, p0/z, [x0, #7, mul vl]

    // Range reduction for all 8
    fmul    z4.h, z0.h, z28.h
    fmul    z5.h, z1.h, z28.h
    fmul    z6.h, z2.h, z28.h
    fmul    z7.h, z3.h, z28.h
    fmul    z20.h, z16.h, z28.h
    fmul    z21.h, z17.h, z28.h
    fmul    z22.h, z18.h, z28.h
    fmul    z23.h, z19.h, z28.h

    frintn  z4.h, p0/m, z4.h
    frintn  z5.h, p0/m, z5.h
    frintn  z6.h, p0/m, z6.h
    frintn  z7.h, p0/m, z7.h
    frintn  z20.h, p0/m, z20.h
    frintn  z21.h, p0/m, z21.h
    frintn  z22.h, p0/m, z22.h
    frintn  z23.h, p0/m, z23.h

    fmls    z0.h, p0/m, z4.h, z29.h
    fmls    z1.h, p0/m, z5.h, z29.h
    fmls    z2.h, p0/m, z6.h, z29.h
    fmls    z3.h, p0/m, z7.h, z29.h
    fmls    z16.h, p0/m, z20.h, z29.h
    fmls    z17.h, p0/m, z21.h, z29.h
    fmls    z18.h, p0/m, z22.h, z29.h
    fmls    z19.h, p0/m, z23.h, z29.h

    // 2^n
    fcvtzs  z4.h, p0/m, z4.h
    fcvtzs  z5.h, p0/m, z5.h
    fcvtzs  z6.h, p0/m, z6.h
    fcvtzs  z7.h, p0/m, z7.h
    fcvtzs  z20.h, p0/m, z20.h
    fcvtzs  z21.h, p0/m, z21.h
    fcvtzs  z22.h, p0/m, z22.h
    fcvtzs  z23.h, p0/m, z23.h

    mov     z31.h, #15
    add     z4.h, z4.h, z31.h
    add     z5.h, z5.h, z31.h
    add     z6.h, z6.h, z31.h
    add     z7.h, z7.h, z31.h
    add     z20.h, z20.h, z31.h
    add     z21.h, z21.h, z31.h
    add     z22.h, z22.h, z31.h
    add     z23.h, z23.h, z31.h

    lsl     z4.h, z4.h, #10
    lsl     z5.h, z5.h, #10
    lsl     z6.h, z6.h, #10
    lsl     z7.h, z7.h, #10
    lsl     z20.h, z20.h, #10
    lsl     z21.h, z21.h, #10
    lsl     z22.h, z22.h, #10
    lsl     z23.h, z23.h, #10

    // Polynomial
    mov     w10, #0x3155            // 1/6
    dup     z8.h, w10
    dup     z9.h, w10
    dup     z10.h, w10
    dup     z11.h, w10
    dup     z24.h, w10
    dup     z25.h, w10
    dup     z26.h, w10
    dup     z27.h, w10

    fmov    z31.h, #0.5
    fmad    z8.h, p0/m, z0.h, z31.h
    fmad    z9.h, p0/m, z1.h, z31.h
    fmad    z10.h, p0/m, z2.h, z31.h
    fmad    z11.h, p0/m, z3.h, z31.h
    fmad    z24.h, p0/m, z16.h, z31.h
    fmad    z25.h, p0/m, z17.h, z31.h
    fmad    z26.h, p0/m, z18.h, z31.h
    fmad    z27.h, p0/m, z19.h, z31.h

    fmov    z31.h, #1.0
    fmad    z8.h, p0/m, z0.h, z31.h
    fmad    z9.h, p0/m, z1.h, z31.h
    fmad    z10.h, p0/m, z2.h, z31.h
    fmad    z11.h, p0/m, z3.h, z31.h
    fmad    z24.h, p0/m, z16.h, z31.h
    fmad    z25.h, p0/m, z17.h, z31.h
    fmad    z26.h, p0/m, z18.h, z31.h
    fmad    z27.h, p0/m, z19.h, z31.h

    fmad    z8.h, p0/m, z0.h, z31.h
    fmad    z9.h, p0/m, z1.h, z31.h
    fmad    z10.h, p0/m, z2.h, z31.h
    fmad    z11.h, p0/m, z3.h, z31.h
    fmad    z24.h, p0/m, z16.h, z31.h
    fmad    z25.h, p0/m, z17.h, z31.h
    fmad    z26.h, p0/m, z18.h, z31.h
    fmad    z27.h, p0/m, z19.h, z31.h

    fmul    z8.h, z8.h, z4.h
    fmul    z9.h, z9.h, z5.h
    fmul    z10.h, z10.h, z6.h
    fmul    z11.h, z11.h, z7.h
    fmul    z24.h, z24.h, z20.h
    fmul    z25.h, z25.h, z21.h
    fmul    z26.h, z26.h, z22.h
    fmul    z27.h, z27.h, z23.h

    st1h    z8.h, p0, [x1]
    st1h    z9.h, p0, [x1, #1, mul vl]
    st1h    z10.h, p0, [x1, #2, mul vl]
    st1h    z11.h, p0, [x1, #3, mul vl]
    st1h    z24.h, p0, [x1, #4, mul vl]
    st1h    z25.h, p0, [x1, #5, mul vl]
    st1h    z26.h, p0, [x1, #6, mul vl]
    st1h    z27.h, p0, [x1, #7, mul vl]

    add     x0, x0, x4, lsl #1
    add     x1, x1, x4, lsl #1
    sub     x5, x5, x4
    b       .Lloop_f16_8

.Ltail_f16_8:
    cbz     x5, .Ldone_f16_8
    whilelt p1.h, xzr, x5
    ld1h    z0.h, p1/z, [x0]

    fmul    z4.h, z0.h, z28.h
    frintn  z4.h, p0/m, z4.h
    fmls    z0.h, p0/m, z4.h, z29.h
    fcvtzs  z4.h, p0/m, z4.h
    mov     z31.h, #15
    add     z4.h, z4.h, z31.h
    lsl     z4.h, z4.h, #10

    mov     w10, #0x3155
    dup     z8.h, w10
    fmov    z31.h, #0.5
    fmad    z8.h, p0/m, z0.h, z31.h
    fmov    z31.h, #1.0
    fmad    z8.h, p0/m, z0.h, z31.h
    fmad    z8.h, p0/m, z0.h, z31.h
    fmul    z8.h, z8.h, z4.h

    st1h    z8.h, p1, [x1]

.Ldone_f16_8:
    ret
    .size exp_vec_f16_unroll8, .-exp_vec_f16_unroll8
