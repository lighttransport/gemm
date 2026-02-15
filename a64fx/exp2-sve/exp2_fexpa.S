/*
 * SVE exp2 kernel using FEXPA instruction
 *
 * FEXPA provides ~6-7 bit accurate approximation of 2^(fractional part)
 * We add polynomial correction for better accuracy.
 *
 * For exp2(x) where x is any float:
 *   n = floor(x)     (integer part)
 *   f = x - n        (fractional part, [0,1))
 *   exp2(x) = 2^n * 2^f
 *   - 2^n is computed via bit manipulation (IEEE754 exponent)
 *   - 2^f uses FEXPA + polynomial correction
 *
 * Target: ~14 instructions per vector (amortized with 4x unroll)
 */

.arch armv8.2-a+sve

.text
.align 4

/*
 * exp2_fexpa_softmax: Compute exp2((x - max) * scale) for softmax
 *
 * void exp2_fexpa_softmax(
 *     const int32_t* in,   // x0: input int32 values
 *     float* out,          // x1: output exp2 values
 *     int n,               // x2: number of elements
 *     float scale,         // s0: scale factor (e.g., 1/sqrt(D) * log2(e))
 *     int32_t max_val      // w3: max value for stability
 * );
 *
 * Computes: out[i] = exp2((in[i] - max_val) * scale)
 *
 * Used for softmax: exp(x - max) = exp2((x - max) * log2(e))
 */
.global exp2_fexpa_softmax
.type exp2_fexpa_softmax, %function
exp2_fexpa_softmax:
    // Save callee-saved registers
    stp     d8, d9, [sp, #-16]!
    stp     d10, d11, [sp, #-16]!
    stp     d12, d13, [sp, #-16]!
    stp     d14, d15, [sp, #-16]!

    // Setup predicates and constants
    ptrue   p0.s                    // All-true predicate

    // Broadcast scale to vector
    dup     z20.s, z0.s[0]          // z20 = scale

    // Broadcast max_val and convert to float
    mov     z21.s, w3               // z21 = max_val (as int)
    scvtf   z21.s, p0/m, z21.s      // z21 = (float)max_val

    // Constants for IEEE754 manipulation
    mov     w4, #127                // IEEE754 exponent bias
    dup     z22.s, w4               // z22 = 127

    // Polynomial correction coefficient (for better accuracy)
    // exp2(f) ≈ fexpa(f) + c * f^2, where c ≈ 0.000125
    mov     w4, #0x0000             // low 16 bits
    movk    w4, #0x3900, lsl #16    // ~0.000122 in float bits
    dup     z23.s, w4               // z23 = correction coeff

    // Clamp constants: -126.0 = 0xC2FC0000, 127.0 = 0x42FE0000
    mov     w4, #0x0000
    movk    w4, #0xC2FC, lsl #16    // -126.0 in IEEE754
    dup     z24.s, w4               // z24 = -126.0 (min exponent)
    
    mov     w4, #0x0000
    movk    w4, #0x42FE, lsl #16    // 127.0 in IEEE754
    dup     z25.s, w4               // z25 = 127.0 (max exponent)

    // 64.0 for FEXPA scaling: 0x42800000
    mov     w4, #0x0000
    movk    w4, #0x4280, lsl #16
    dup     z26.s, w4               // z26 = 64.0

    // SVE vector length in 32-bit elements
    cntw    x4                      // x4 = elements per vector

    // Check if we can use 4x unrolled loop
    lsl     x5, x4, #2              // x5 = 4 * vl
    cmp     x2, x5
    b.lt    .Lsoftmax_tail

.Lsoftmax_loop4:
    // Load 4 vectors of int32 input
    ld1w    z0.s, p0/z, [x0]
    ld1w    z1.s, p0/z, [x0, #1, mul vl]
    ld1w    z2.s, p0/z, [x0, #2, mul vl]
    ld1w    z3.s, p0/z, [x0, #3, mul vl]

    // Convert int32 -> float
    scvtf   z0.s, p0/m, z0.s
    scvtf   z1.s, p0/m, z1.s
    scvtf   z2.s, p0/m, z2.s
    scvtf   z3.s, p0/m, z3.s

    // Subtract max and scale: x = (in - max) * scale
    fsub    z0.s, z0.s, z21.s
    fsub    z1.s, z1.s, z21.s
    fsub    z2.s, z2.s, z21.s
    fsub    z3.s, z3.s, z21.s

    fmul    z0.s, z0.s, z20.s
    fmul    z1.s, z1.s, z20.s
    fmul    z2.s, z2.s, z20.s
    fmul    z3.s, z3.s, z20.s

    // Clamp to valid exp2 range
    fmax    z0.s, p0/m, z0.s, z24.s
    fmax    z1.s, p0/m, z1.s, z24.s
    fmax    z2.s, p0/m, z2.s, z24.s
    fmax    z3.s, p0/m, z3.s, z24.s

    fmin    z0.s, p0/m, z0.s, z25.s
    fmin    z1.s, p0/m, z1.s, z25.s
    fmin    z2.s, p0/m, z2.s, z25.s
    fmin    z3.s, p0/m, z3.s, z25.s

    // Split into integer and fractional parts
    // n = floor(x), f = x - n
    frintm  z4.s, p0/m, z0.s        // n0 = floor(x0)
    frintm  z5.s, p0/m, z1.s        // n1 = floor(x1)
    frintm  z6.s, p0/m, z2.s        // n2 = floor(x2)
    frintm  z7.s, p0/m, z3.s        // n3 = floor(x3)

    fsub    z8.s, z0.s, z4.s        // f0 = x0 - n0
    fsub    z9.s, z1.s, z5.s        // f1 = x1 - n1
    fsub    z10.s, z2.s, z6.s       // f2 = x2 - n2
    fsub    z11.s, z3.s, z7.s       // f3 = x3 - n3

    // Compute 2^n via IEEE754 bit manipulation
    // Convert n to int, add bias, shift to exponent position
    fcvtzs  z4.s, p0/m, z4.s        // n0 as int
    fcvtzs  z5.s, p0/m, z5.s        // n1 as int
    fcvtzs  z6.s, p0/m, z6.s        // n2 as int
    fcvtzs  z7.s, p0/m, z7.s        // n3 as int

    add     z4.s, z4.s, z22.s       // n0 + 127
    add     z5.s, z5.s, z22.s       // n1 + 127
    add     z6.s, z6.s, z22.s       // n2 + 127
    add     z7.s, z7.s, z22.s       // n3 + 127

    lsl     z4.s, z4.s, #23         // 2^n0 as IEEE754 float bits
    lsl     z5.s, z5.s, #23         // 2^n1 as IEEE754 float bits
    lsl     z6.s, z6.s, #23         // 2^n2 as IEEE754 float bits
    lsl     z7.s, z7.s, #23         // 2^n3 as IEEE754 float bits

    // FEXPA: get exp2 approximation for fractional part
    // Scale fractional part for FEXPA: f * 64
    fmul    z12.s, z8.s, z26.s      // f0 * 64
    fmul    z13.s, z9.s, z26.s      // f1 * 64
    fmul    z14.s, z10.s, z26.s     // f2 * 64
    fmul    z15.s, z11.s, z26.s     // f3 * 64

    // Convert to unsigned for FEXPA
    fcvtzu  z12.s, p0/m, z12.s
    fcvtzu  z13.s, p0/m, z13.s
    fcvtzu  z14.s, p0/m, z14.s
    fcvtzu  z15.s, p0/m, z15.s

    // Apply FEXPA
    fexpa   z12.s, z12.s            // exp2 approx for f0
    fexpa   z13.s, z13.s            // exp2 approx for f1
    fexpa   z14.s, z14.s            // exp2 approx for f2
    fexpa   z15.s, z15.s            // exp2 approx for f3

    // Polynomial correction: exp2 += c * f^2
    fmul    z16.s, z8.s, z8.s       // f0^2
    fmul    z17.s, z9.s, z9.s       // f1^2
    fmul    z18.s, z10.s, z10.s     // f2^2
    fmul    z19.s, z11.s, z11.s     // f3^2

    fmla    z12.s, p0/m, z16.s, z23.s   // exp2_0 += c * f0^2
    fmla    z13.s, p0/m, z17.s, z23.s   // exp2_1 += c * f1^2
    fmla    z14.s, p0/m, z18.s, z23.s   // exp2_2 += c * f2^2
    fmla    z15.s, p0/m, z19.s, z23.s   // exp2_3 += c * f3^2

    // Combine 2^n * 2^f
    fmul    z12.s, z12.s, z4.s
    fmul    z13.s, z13.s, z5.s
    fmul    z14.s, z14.s, z6.s
    fmul    z15.s, z15.s, z7.s

    // Store results
    st1w    z12.s, p0, [x1]
    st1w    z13.s, p0, [x1, #1, mul vl]
    st1w    z14.s, p0, [x1, #2, mul vl]
    st1w    z15.s, p0, [x1, #3, mul vl]

    // Advance pointers and decrement count
    add     x0, x0, x5, lsl #2      // x0 += 4 * vl * sizeof(int32)
    add     x1, x1, x5, lsl #2      // x1 += 4 * vl * sizeof(float)
    sub     x2, x2, x5              // n -= 4 * vl
    cmp     x2, x5
    b.ge    .Lsoftmax_loop4

.Lsoftmax_tail:
    // Handle remaining elements one vector at a time
    cbz     x2, .Lsoftmax_done

.Lsoftmax_loop1:
    // Create predicate for remaining elements
    whilelt p1.s, xzr, x2

    // Load input
    ld1w    z0.s, p1/z, [x0]

    // Convert int32 -> float
    scvtf   z0.s, p1/m, z0.s

    // Subtract max and scale
    fsub    z0.s, p1/m, z0.s, z21.s
    fmul    z0.s, p1/m, z0.s, z20.s

    // Clamp
    fmax    z0.s, p1/m, z0.s, z24.s
    fmin    z0.s, p1/m, z0.s, z25.s

    // Split into integer and fractional parts
    frintm  z4.s, p1/m, z0.s
    fsub    z8.s, z0.s, z4.s

    // 2^n via bit manipulation
    fcvtzs  z4.s, p1/m, z4.s
    add     z4.s, z4.s, z22.s
    lsl     z4.s, z4.s, #23

    // FEXPA for fractional part
    fmul    z12.s, z8.s, z26.s
    fcvtzu  z12.s, p1/m, z12.s
    fexpa   z12.s, z12.s

    // Correction
    fmul    z16.s, z8.s, z8.s
    fmla    z12.s, p1/m, z16.s, z23.s

    // Combine
    fmul    z12.s, p1/m, z12.s, z4.s

    // Store
    st1w    z12.s, p1, [x1]

    // Advance
    incw    x0
    incw    x1
    decw    x2
    cbnz    x2, .Lsoftmax_loop1

.Lsoftmax_done:
    // Restore callee-saved registers
    ldp     d14, d15, [sp], #16
    ldp     d12, d13, [sp], #16
    ldp     d10, d11, [sp], #16
    ldp     d8, d9, [sp], #16
    ret

.size exp2_fexpa_softmax, .-exp2_fexpa_softmax


/*
 * exp2_fexpa_simple: Simple exp2 for float input
 *
 * void exp2_fexpa_simple(
 *     const float* in,     // x0: input float values
 *     float* out,          // x1: output exp2 values
 *     int n                // x2: number of elements
 * );
 *
 * Computes: out[i] = exp2(in[i])
 */
.global exp2_fexpa_simple
.type exp2_fexpa_simple, %function
exp2_fexpa_simple:
    stp     d8, d9, [sp, #-16]!
    stp     d10, d11, [sp, #-16]!
    stp     d12, d13, [sp, #-16]!
    stp     d14, d15, [sp, #-16]!

    ptrue   p0.s

    // Constants
    mov     w4, #127
    dup     z22.s, w4               // bias

    mov     w4, #0x0000
    movk    w4, #0x3900, lsl #16
    dup     z23.s, w4               // correction coeff

    mov     w4, #0x0000
    movk    w4, #0xC2FC, lsl #16    // -126.0
    dup     z24.s, w4
    
    mov     w4, #0x0000
    movk    w4, #0x42FE, lsl #16    // 127.0
    dup     z25.s, w4

    mov     w4, #0x0000
    movk    w4, #0x4280, lsl #16    // 64.0
    dup     z26.s, w4

    cntw    x4
    lsl     x5, x4, #2

    cmp     x2, x5
    b.lt    .Lsimple_tail

.Lsimple_loop4:
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

    // FEXPA
    fmul    z12.s, z8.s, z26.s
    fmul    z13.s, z9.s, z26.s
    fmul    z14.s, z10.s, z26.s
    fmul    z15.s, z11.s, z26.s

    fcvtzu  z12.s, p0/m, z12.s
    fcvtzu  z13.s, p0/m, z13.s
    fcvtzu  z14.s, p0/m, z14.s
    fcvtzu  z15.s, p0/m, z15.s

    fexpa   z12.s, z12.s
    fexpa   z13.s, z13.s
    fexpa   z14.s, z14.s
    fexpa   z15.s, z15.s

    // Correction
    fmul    z16.s, z8.s, z8.s
    fmul    z17.s, z9.s, z9.s
    fmul    z18.s, z10.s, z10.s
    fmul    z19.s, z11.s, z11.s

    fmla    z12.s, p0/m, z16.s, z23.s
    fmla    z13.s, p0/m, z17.s, z23.s
    fmla    z14.s, p0/m, z18.s, z23.s
    fmla    z15.s, p0/m, z19.s, z23.s

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
    b.ge    .Lsimple_loop4

.Lsimple_tail:
    cbz     x2, .Lsimple_done

.Lsimple_loop1:
    whilelt p1.s, xzr, x2

    ld1w    z0.s, p1/z, [x0]

    fmax    z0.s, p1/m, z0.s, z24.s
    fmin    z0.s, p1/m, z0.s, z25.s

    frintm  z4.s, p1/m, z0.s
    fsub    z8.s, z0.s, z4.s

    fcvtzs  z4.s, p1/m, z4.s
    add     z4.s, z4.s, z22.s
    lsl     z4.s, z4.s, #23

    fmul    z12.s, z8.s, z26.s
    fcvtzu  z12.s, p1/m, z12.s
    fexpa   z12.s, z12.s

    fmul    z16.s, z8.s, z8.s
    fmla    z12.s, p1/m, z16.s, z23.s

    fmul    z12.s, p1/m, z12.s, z4.s

    st1w    z12.s, p1, [x1]

    incw    x0
    incw    x1
    decw    x2
    cbnz    x2, .Lsimple_loop1

.Lsimple_done:
    ldp     d14, d15, [sp], #16
    ldp     d12, d13, [sp], #16
    ldp     d10, d11, [sp], #16
    ldp     d8, d9, [sp], #16
    ret

.size exp2_fexpa_simple, .-exp2_fexpa_simple
