/*
 * Fixed SVE exp2 kernel using FEXPA instruction
 *
 * FEXPA: Takes float input directly (bits 22:16 are used as table index)
 * For f in [0, 1), FEXPA(f) returns 2^f approximation
 *
 * Algorithm for exp2(x):
 *   n = floor(x)     (integer part)
 *   f = x - n        (fractional part, [0,1))
 *   exp2(x) = 2^n * 2^f
 *   - 2^n computed via bit manipulation (IEEE754 exponent)
 *   - 2^f computed using FEXPA directly on f
 */

.arch armv8.2-a+sve

.text
.align 4

/*
 * exp2_fexpa_softmax_fixed: Compute exp2((x - max) * scale) for softmax
 *
 * void exp2_fexpa_softmax_fixed(
 *     const int32_t* in,   // x0: input int32 values
 *     float* out,          // x1: output exp2 values
 *     int n,               // x2: number of elements
 *     float scale,         // s0: scale factor
 *     int32_t max_val      // w3: max value for stability
 * );
 */
.global exp2_fexpa_softmax_fixed
.type exp2_fexpa_softmax_fixed, %function
exp2_fexpa_softmax_fixed:
    stp     d8, d9, [sp, #-16]!
    stp     d10, d11, [sp, #-16]!
    stp     d12, d13, [sp, #-16]!
    stp     d14, d15, [sp, #-16]!

    ptrue   p0.s

    // Broadcast scale to vector
    dup     z20.s, z0.s[0]          // z20 = scale

    // Broadcast max_val and convert to float
    mov     z21.s, w3
    scvtf   z21.s, p0/m, z21.s      // z21 = (float)max_val

    // Constants for IEEE754 manipulation
    mov     w4, #127
    dup     z22.s, w4               // z22 = 127 (exponent bias)

    // Clamp constants
    mov     w4, #0x0000
    movk    w4, #0xC2FC, lsl #16    // -126.0
    dup     z24.s, w4

    mov     w4, #0x0000
    movk    w4, #0x42FE, lsl #16    // 127.0
    dup     z25.s, w4

    // SVE vector length
    cntw    x4
    lsl     x5, x4, #2              // 4 * vl

    cmp     x2, x5
    b.lt    .Lfix_tail

.Lfix_loop4:
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

    add     z4.s, z4.s, z22.s
    add     z5.s, z5.s, z22.s
    add     z6.s, z6.s, z22.s
    add     z7.s, z7.s, z22.s

    lsl     z4.s, z4.s, #23         // 2^n as IEEE754
    lsl     z5.s, z5.s, #23
    lsl     z6.s, z6.s, #23
    lsl     z7.s, z7.s, #23

    // FEXPA: apply directly to fractional part f (as float)
    // FEXPA uses bits [22:16] of the float to index lookup table
    fexpa   z12.s, z8.s
    fexpa   z13.s, z9.s
    fexpa   z14.s, z10.s
    fexpa   z15.s, z11.s

    // Combine: 2^x = 2^n * 2^f
    // Note: z4-z7 contain IEEE754 bits for 2^n, treated as float in fmul
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
    b.ge    .Lfix_loop4

.Lfix_tail:
    cbz     x2, .Lfix_done

.Lfix_loop1:
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
    add     z4.s, z4.s, z22.s
    lsl     z4.s, z4.s, #23

    fexpa   z12.s, z8.s
    fmul    z12.s, p1/m, z12.s, z4.s

    st1w    z12.s, p1, [x1]

    incw    x0
    incw    x1
    decw    x2
    cbnz    x2, .Lfix_loop1

.Lfix_done:
    ldp     d14, d15, [sp], #16
    ldp     d12, d13, [sp], #16
    ldp     d10, d11, [sp], #16
    ldp     d8, d9, [sp], #16
    ret
.size exp2_fexpa_softmax_fixed, .-exp2_fexpa_softmax_fixed


/*
 * exp2_fexpa_simple_fixed: Simple exp2 for float input (fixed version)
 */
.global exp2_fexpa_simple_fixed
.type exp2_fexpa_simple_fixed, %function
exp2_fexpa_simple_fixed:
    stp     d8, d9, [sp, #-16]!
    stp     d10, d11, [sp, #-16]!
    stp     d12, d13, [sp, #-16]!
    stp     d14, d15, [sp, #-16]!

    ptrue   p0.s

    // Constants
    mov     w4, #127
    dup     z22.s, w4               // bias

    mov     w4, #0x0000
    movk    w4, #0xC2FC, lsl #16    // -126.0
    dup     z24.s, w4

    mov     w4, #0x0000
    movk    w4, #0x42FE, lsl #16    // 127.0
    dup     z25.s, w4

    cntw    x4
    lsl     x5, x4, #2

    cmp     x2, x5
    b.lt    .Lsimfix_tail

.Lsimfix_loop4:
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

    add     z4.s, z4.s, z22.s
    add     z5.s, z5.s, z22.s
    add     z6.s, z6.s, z22.s
    add     z7.s, z7.s, z22.s

    lsl     z4.s, z4.s, #23
    lsl     z5.s, z5.s, #23
    lsl     z6.s, z6.s, #23
    lsl     z7.s, z7.s, #23

    // FEXPA directly on fractional part
    fexpa   z12.s, z8.s
    fexpa   z13.s, z9.s
    fexpa   z14.s, z10.s
    fexpa   z15.s, z11.s

    // Combine
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
    b.ge    .Lsimfix_loop4

.Lsimfix_tail:
    cbz     x2, .Lsimfix_done

.Lsimfix_loop1:
    whilelt p1.s, xzr, x2

    ld1w    z0.s, p1/z, [x0]

    fmax    z0.s, p1/m, z0.s, z24.s
    fmin    z0.s, p1/m, z0.s, z25.s

    frintm  z4.s, p1/m, z0.s
    fsub    z8.s, z0.s, z4.s

    fcvtzs  z4.s, p1/m, z4.s
    add     z4.s, z4.s, z22.s
    lsl     z4.s, z4.s, #23

    fexpa   z12.s, z8.s
    fmul    z12.s, p1/m, z12.s, z4.s

    st1w    z12.s, p1, [x1]

    incw    x0
    incw    x1
    decw    x2
    cbnz    x2, .Lsimfix_loop1

.Lsimfix_done:
    ldp     d14, d15, [sp], #16
    ldp     d12, d13, [sp], #16
    ldp     d10, d11, [sp], #16
    ldp     d8, d9, [sp], #16
    ret
.size exp2_fexpa_simple_fixed, .-exp2_fexpa_simple_fixed
