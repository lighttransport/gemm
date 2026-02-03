/*
 * SVE exp2 kernel using Estrin's polynomial with 8x unroll
 *
 * 8x unrolling processes 8 SVE vectors (128 floats on A64FX) per iteration
 * by doing two sequential batches of 4 vectors.
 * This maintains simple register allocation while increasing throughput.
 */

.arch armv8.2-a+sve

.text
.align 4

/*
 * exp2_estrin_u8: exp2 with Estrin polynomial, 8x unrolled
 *
 * Arguments:
 *   x0: input pointer (float*)
 *   x1: output pointer (float*)
 *   x2: count (int)
 *
 * Register allocation:
 *   z24: clamp low (-126.0)
 *   z25: clamp high (127.0)
 *   z26: c1 = ln(2) = 0.6931472
 *   z27: c2 = 0.2402265
 *   z28: c3 = 0.0555041
 *   z29: c4 = 0.0096181
 *   z30: bias = 127
 *   z31: 1.0
 *   p0: all-true predicate
 */
.global exp2_estrin_u8
.type exp2_estrin_u8, %function
exp2_estrin_u8:
    // Save callee-saved registers
    stp     x19, x20, [sp, #-16]!
    stp     d8, d9, [sp, #-16]!
    stp     d10, d11, [sp, #-16]!
    stp     d12, d13, [sp, #-16]!
    stp     d14, d15, [sp, #-16]!

    ptrue   p0.s

    // ========== Load constants into z24-z31 ==========
    // bias = 127
    mov     w19, #127
    dup     z30.s, w19

    // c1 = 0.6931472 (0x3F317218)
    movz    w19, #0x7218
    movk    w19, #0x3F31, lsl #16
    dup     z26.s, w19

    // c2 = 0.2402265 (0x3E75ED50)
    movz    w19, #0xED50
    movk    w19, #0x3E75, lsl #16
    dup     z27.s, w19

    // c3 = 0.0555041 (0x3D636C16)
    movz    w19, #0x6C16
    movk    w19, #0x3D63, lsl #16
    dup     z28.s, w19

    // c4 = 0.0096181 (0x3C1D8DAE)
    movz    w19, #0x8DAE
    movk    w19, #0x3C1D, lsl #16
    dup     z29.s, w19

    // 1.0 (0x3F800000)
    movz    w19, #0x0000
    movk    w19, #0x3F80, lsl #16
    dup     z31.s, w19

    // Clamp: -126.0 (0xC2FC0000)
    movz    w19, #0x0000
    movk    w19, #0xC2FC, lsl #16
    dup     z24.s, w19

    // Clamp: 127.0 (0x42FE0000)
    movz    w19, #0x0000
    movk    w19, #0x42FE, lsl #16
    dup     z25.s, w19

    // Calculate vector length and 8x batch size
    cntw    x4                          // x4 = vector length (16 for A64FX)
    lsl     x5, x4, #3                  // x5 = 8 * vl = 128 elements per iteration

    // Check if we have at least 8 vectors
    cmp     x2, x5
    b.lt    .Lu8_tail

    // ========== Main loop: process 8 vectors per iteration ==========
.Lu8_loop:
    // ===== First batch of 4 vectors =====
    ld1w    z0.s, p0/z, [x0]
    ld1w    z1.s, p0/z, [x0, #1, mul vl]
    ld1w    z2.s, p0/z, [x0, #2, mul vl]
    ld1w    z3.s, p0/z, [x0, #3, mul vl]

    // Clamp to [-126, 127]
    fmax    z0.s, p0/m, z0.s, z24.s
    fmax    z1.s, p0/m, z1.s, z24.s
    fmax    z2.s, p0/m, z2.s, z24.s
    fmax    z3.s, p0/m, z3.s, z24.s
    fmin    z0.s, p0/m, z0.s, z25.s
    fmin    z1.s, p0/m, z1.s, z25.s
    fmin    z2.s, p0/m, z2.s, z25.s
    fmin    z3.s, p0/m, z3.s, z25.s

    // n = floor(x)
    frintm  z4.s, p0/m, z0.s
    frintm  z5.s, p0/m, z1.s
    frintm  z6.s, p0/m, z2.s
    frintm  z7.s, p0/m, z3.s

    // f = x - n (fractional part)
    fsub    z8.s, z0.s, z4.s
    fsub    z9.s, z1.s, z5.s
    fsub    z10.s, z2.s, z6.s
    fsub    z11.s, z3.s, z7.s

    // Compute 2^n via IEEE754 bit manipulation
    fcvtzs  z4.s, p0/m, z4.s
    fcvtzs  z5.s, p0/m, z5.s
    fcvtzs  z6.s, p0/m, z6.s
    fcvtzs  z7.s, p0/m, z7.s
    add     z4.s, z4.s, z30.s           // n + 127
    add     z5.s, z5.s, z30.s
    add     z6.s, z6.s, z30.s
    add     z7.s, z7.s, z30.s
    lsl     z4.s, z4.s, #23             // shift to exponent position
    lsl     z5.s, z5.s, #23
    lsl     z6.s, z6.s, #23
    lsl     z7.s, z7.s, #23

    // Estrin's scheme for 2^f polynomial
    // f^2 (all 4 in parallel)
    fmul    z12.s, z8.s, z8.s
    fmul    z13.s, z9.s, z9.s
    fmul    z14.s, z10.s, z10.s
    fmul    z15.s, z11.s, z11.s

    // q0 = 1 + c1*f
    mov     z16.d, z31.d
    mov     z17.d, z31.d
    mov     z18.d, z31.d
    mov     z19.d, z31.d
    fmla    z16.s, p0/m, z8.s, z26.s
    fmla    z17.s, p0/m, z9.s, z26.s
    fmla    z18.s, p0/m, z10.s, z26.s
    fmla    z19.s, p0/m, z11.s, z26.s

    // q1 = c2 + c3*f
    mov     z20.d, z27.d
    mov     z21.d, z27.d
    mov     z22.d, z27.d
    mov     z23.d, z27.d
    fmla    z20.s, p0/m, z8.s, z28.s
    fmla    z21.s, p0/m, z9.s, z28.s
    fmla    z22.s, p0/m, z10.s, z28.s
    fmla    z23.s, p0/m, z11.s, z28.s

    // f^4 = f^2 * f^2
    fmul    z0.s, z12.s, z12.s
    fmul    z1.s, z13.s, z13.s
    fmul    z2.s, z14.s, z14.s
    fmul    z3.s, z15.s, z15.s

    // r = q0 + q1*f^2
    fmla    z16.s, p0/m, z20.s, z12.s
    fmla    z17.s, p0/m, z21.s, z13.s
    fmla    z18.s, p0/m, z22.s, z14.s
    fmla    z19.s, p0/m, z23.s, z15.s

    // p = r + c4*f^4
    fmla    z16.s, p0/m, z0.s, z29.s
    fmla    z17.s, p0/m, z1.s, z29.s
    fmla    z18.s, p0/m, z2.s, z29.s
    fmla    z19.s, p0/m, z3.s, z29.s

    // result = 2^n * 2^f
    fmul    z16.s, z16.s, z4.s
    fmul    z17.s, z17.s, z5.s
    fmul    z18.s, z18.s, z6.s
    fmul    z19.s, z19.s, z7.s

    // Store first batch results
    st1w    z16.s, p0, [x1]
    st1w    z17.s, p0, [x1, #1, mul vl]
    st1w    z18.s, p0, [x1, #2, mul vl]
    st1w    z19.s, p0, [x1, #3, mul vl]

    // ===== Second batch of 4 vectors =====
    ld1w    z0.s, p0/z, [x0, #4, mul vl]
    ld1w    z1.s, p0/z, [x0, #5, mul vl]
    ld1w    z2.s, p0/z, [x0, #6, mul vl]
    ld1w    z3.s, p0/z, [x0, #7, mul vl]

    fmax    z0.s, p0/m, z0.s, z24.s
    fmax    z1.s, p0/m, z1.s, z24.s
    fmax    z2.s, p0/m, z2.s, z24.s
    fmax    z3.s, p0/m, z3.s, z24.s
    fmin    z0.s, p0/m, z0.s, z25.s
    fmin    z1.s, p0/m, z1.s, z25.s
    fmin    z2.s, p0/m, z2.s, z25.s
    fmin    z3.s, p0/m, z3.s, z25.s

    frintm  z4.s, p0/m, z0.s
    frintm  z5.s, p0/m, z1.s
    frintm  z6.s, p0/m, z2.s
    frintm  z7.s, p0/m, z3.s

    fsub    z8.s, z0.s, z4.s
    fsub    z9.s, z1.s, z5.s
    fsub    z10.s, z2.s, z6.s
    fsub    z11.s, z3.s, z7.s

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

    fmul    z12.s, z8.s, z8.s
    fmul    z13.s, z9.s, z9.s
    fmul    z14.s, z10.s, z10.s
    fmul    z15.s, z11.s, z11.s

    mov     z16.d, z31.d
    mov     z17.d, z31.d
    mov     z18.d, z31.d
    mov     z19.d, z31.d
    fmla    z16.s, p0/m, z8.s, z26.s
    fmla    z17.s, p0/m, z9.s, z26.s
    fmla    z18.s, p0/m, z10.s, z26.s
    fmla    z19.s, p0/m, z11.s, z26.s

    mov     z20.d, z27.d
    mov     z21.d, z27.d
    mov     z22.d, z27.d
    mov     z23.d, z27.d
    fmla    z20.s, p0/m, z8.s, z28.s
    fmla    z21.s, p0/m, z9.s, z28.s
    fmla    z22.s, p0/m, z10.s, z28.s
    fmla    z23.s, p0/m, z11.s, z28.s

    fmul    z0.s, z12.s, z12.s
    fmul    z1.s, z13.s, z13.s
    fmul    z2.s, z14.s, z14.s
    fmul    z3.s, z15.s, z15.s

    fmla    z16.s, p0/m, z20.s, z12.s
    fmla    z17.s, p0/m, z21.s, z13.s
    fmla    z18.s, p0/m, z22.s, z14.s
    fmla    z19.s, p0/m, z23.s, z15.s

    fmla    z16.s, p0/m, z0.s, z29.s
    fmla    z17.s, p0/m, z1.s, z29.s
    fmla    z18.s, p0/m, z2.s, z29.s
    fmla    z19.s, p0/m, z3.s, z29.s

    fmul    z16.s, z16.s, z4.s
    fmul    z17.s, z17.s, z5.s
    fmul    z18.s, z18.s, z6.s
    fmul    z19.s, z19.s, z7.s

    st1w    z16.s, p0, [x1, #4, mul vl]
    st1w    z17.s, p0, [x1, #5, mul vl]
    st1w    z18.s, p0, [x1, #6, mul vl]
    st1w    z19.s, p0, [x1, #7, mul vl]

    // Advance pointers (8 vectors = x5 elements * 4 bytes)
    add     x0, x0, x5, lsl #2
    add     x1, x1, x5, lsl #2
    sub     x2, x2, x5

    cmp     x2, x5
    b.ge    .Lu8_loop

    // ========== Tail handling ==========
.Lu8_tail:
    // Check if we have at least 4 vectors
    lsl     x6, x4, #2                  // x6 = 4 * vl = 64 elements
    cmp     x2, x6
    b.lt    .Lu8_tail1

.Lu8_tail4:
    // Process 4 vectors
    ld1w    z0.s, p0/z, [x0]
    ld1w    z1.s, p0/z, [x0, #1, mul vl]
    ld1w    z2.s, p0/z, [x0, #2, mul vl]
    ld1w    z3.s, p0/z, [x0, #3, mul vl]

    fmax    z0.s, p0/m, z0.s, z24.s
    fmax    z1.s, p0/m, z1.s, z24.s
    fmax    z2.s, p0/m, z2.s, z24.s
    fmax    z3.s, p0/m, z3.s, z24.s
    fmin    z0.s, p0/m, z0.s, z25.s
    fmin    z1.s, p0/m, z1.s, z25.s
    fmin    z2.s, p0/m, z2.s, z25.s
    fmin    z3.s, p0/m, z3.s, z25.s

    frintm  z4.s, p0/m, z0.s
    frintm  z5.s, p0/m, z1.s
    frintm  z6.s, p0/m, z2.s
    frintm  z7.s, p0/m, z3.s

    fsub    z8.s, z0.s, z4.s
    fsub    z9.s, z1.s, z5.s
    fsub    z10.s, z2.s, z6.s
    fsub    z11.s, z3.s, z7.s

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

    fmul    z12.s, z8.s, z8.s
    fmul    z13.s, z9.s, z9.s
    fmul    z14.s, z10.s, z10.s
    fmul    z15.s, z11.s, z11.s

    mov     z16.d, z31.d
    mov     z17.d, z31.d
    mov     z18.d, z31.d
    mov     z19.d, z31.d
    fmla    z16.s, p0/m, z8.s, z26.s
    fmla    z17.s, p0/m, z9.s, z26.s
    fmla    z18.s, p0/m, z10.s, z26.s
    fmla    z19.s, p0/m, z11.s, z26.s

    mov     z20.d, z27.d
    mov     z21.d, z27.d
    mov     z22.d, z27.d
    mov     z23.d, z27.d
    fmla    z20.s, p0/m, z8.s, z28.s
    fmla    z21.s, p0/m, z9.s, z28.s
    fmla    z22.s, p0/m, z10.s, z28.s
    fmla    z23.s, p0/m, z11.s, z28.s

    fmul    z0.s, z12.s, z12.s
    fmul    z1.s, z13.s, z13.s
    fmul    z2.s, z14.s, z14.s
    fmul    z3.s, z15.s, z15.s

    fmla    z16.s, p0/m, z20.s, z12.s
    fmla    z17.s, p0/m, z21.s, z13.s
    fmla    z18.s, p0/m, z22.s, z14.s
    fmla    z19.s, p0/m, z23.s, z15.s

    fmla    z16.s, p0/m, z0.s, z29.s
    fmla    z17.s, p0/m, z1.s, z29.s
    fmla    z18.s, p0/m, z2.s, z29.s
    fmla    z19.s, p0/m, z3.s, z29.s

    fmul    z16.s, z16.s, z4.s
    fmul    z17.s, z17.s, z5.s
    fmul    z18.s, z18.s, z6.s
    fmul    z19.s, z19.s, z7.s

    st1w    z16.s, p0, [x1]
    st1w    z17.s, p0, [x1, #1, mul vl]
    st1w    z18.s, p0, [x1, #2, mul vl]
    st1w    z19.s, p0, [x1, #3, mul vl]

    add     x0, x0, x6, lsl #2
    add     x1, x1, x6, lsl #2
    sub     x2, x2, x6

    cmp     x2, x6
    b.ge    .Lu8_tail4

.Lu8_tail1:
    // Handle remaining elements with predicate
    cbz     x2, .Lu8_done

    whilelt p1.s, xzr, x2
    ld1w    z0.s, p1/z, [x0]

    fmax    z0.s, p1/m, z0.s, z24.s
    fmin    z0.s, p1/m, z0.s, z25.s

    frintm  z4.s, p1/m, z0.s
    fsub    z8.s, z0.s, z4.s

    fcvtzs  z4.s, p1/m, z4.s
    add     z4.s, z4.s, z30.s
    lsl     z4.s, z4.s, #23

    fmul    z12.s, z8.s, z8.s

    mov     z16.d, z31.d
    fmla    z16.s, p1/m, z8.s, z26.s

    mov     z20.d, z27.d
    fmla    z20.s, p1/m, z8.s, z28.s

    fmul    z0.s, z12.s, z12.s
    fmla    z16.s, p1/m, z20.s, z12.s
    fmla    z16.s, p1/m, z0.s, z29.s

    fmul    z16.s, p1/m, z16.s, z4.s
    st1w    z16.s, p1, [x1]

.Lu8_done:
    // Restore callee-saved registers
    ldp     d14, d15, [sp], #16
    ldp     d12, d13, [sp], #16
    ldp     d10, d11, [sp], #16
    ldp     d8, d9, [sp], #16
    ldp     x19, x20, [sp], #16
    ret
.size exp2_estrin_u8, .-exp2_estrin_u8
