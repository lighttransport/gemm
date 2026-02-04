/*
 * Optimized exp2f using FEXPA - 4x unroll
 * Based on glibc algorithm
 *
 * FEXPA for FP32:
 * - Input: z = x + shift (shift = 0x48481fc0)
 * - FEXPA reads bits from z to index lookup table T[i] = 2^(i/64)
 * - Output: 2^n where n = round(x)
 *
 * Algorithm:
 *   z = x + shift
 *   n = z - shift       (rounded integer)
 *   r = x - n           (small remainder)
 *   scale = fexpa(z)    (≈ 2^n)
 *   poly = r * c0       (first-order Taylor: 2^r - 1 ≈ ln(2)*r)
 *   result = scale + scale * poly
 *
 * Register allocation (all caller-saved):
 *   z0-z3:   input x, then reused for scale, then result
 *   z4-z7:   z = x + shift, then reused for poly
 *   z16-z19: n = z - shift
 *   z20-z23: r = x - n
 *   z30:     shift constant
 *   z31:     c0 = ln(2)
 *
 * NOTE: z8-z15 are CALLEE-SAVED (d8-d15), must not be used without saving!
 */

.arch armv8.2-a+sve

.text
.align 4

.global exp2_fexpa_opt
.type exp2_fexpa_opt, %function
exp2_fexpa_opt:
    // x0 = input, x1 = output, x2 = n elements

    ptrue   p0.s

    // shift = 0x48481fc0 (magic constant for FEXPA)
    mov     w3, #0x1fc0
    movk    w3, #0x4848, lsl #16
    dup     z30.s, w3

    // c0 = ln(2) = 0x3f317211
    mov     w3, #0x7211
    movk    w3, #0x3f31, lsl #16
    dup     z31.s, w3

    cntw    x4                      // VL (16 for 512-bit)
    lsl     x5, x4, #2              // 4 * VL = 64 elements

    cmp     x2, x5
    b.lt    .Lopt_small

.Lopt_loop:
    // Load 4 vectors of input x
    ld1w    z0.s, p0/z, [x0]
    ld1w    z1.s, p0/z, [x0, #1, mul vl]
    ld1w    z2.s, p0/z, [x0, #2, mul vl]
    ld1w    z3.s, p0/z, [x0, #3, mul vl]

    // z = x + shift (positions bits for FEXPA)
    fadd    z4.s, z0.s, z30.s
    fadd    z5.s, z1.s, z30.s
    fadd    z6.s, z2.s, z30.s
    fadd    z7.s, z3.s, z30.s

    // n = z - shift (rounded integer part)
    fsub    z16.s, z4.s, z30.s
    fsub    z17.s, z5.s, z30.s
    fsub    z18.s, z6.s, z30.s
    fsub    z19.s, z7.s, z30.s

    // r = x - n (small remainder for polynomial)
    fsub    z20.s, z0.s, z16.s
    fsub    z21.s, z1.s, z17.s
    fsub    z22.s, z2.s, z18.s
    fsub    z23.s, z3.s, z19.s

    // scale = fexpa(z) - gives 2^n directly!
    // Reuse z0-z3 for scale (input x no longer needed)
    fexpa   z0.s, z4.s
    fexpa   z1.s, z5.s
    fexpa   z2.s, z6.s
    fexpa   z3.s, z7.s

    // poly = r * c0 (first-order: 2^r - 1 ≈ ln(2)*r)
    // Reuse z4-z7 for poly (z values no longer needed)
    fmul    z4.s, z20.s, z31.s
    fmul    z5.s, z21.s, z31.s
    fmul    z6.s, z22.s, z31.s
    fmul    z7.s, z23.s, z31.s

    // result = scale + scale * poly = scale * (1 + poly)
    fmla    z0.s, p0/m, z0.s, z4.s
    fmla    z1.s, p0/m, z1.s, z5.s
    fmla    z2.s, p0/m, z2.s, z6.s
    fmla    z3.s, p0/m, z3.s, z7.s

    // Store results
    st1w    z0.s, p0, [x1]
    st1w    z1.s, p0, [x1, #1, mul vl]
    st1w    z2.s, p0, [x1, #2, mul vl]
    st1w    z3.s, p0, [x1, #3, mul vl]

    add     x0, x0, x5, lsl #2      // input += 64 * 4 bytes
    add     x1, x1, x5, lsl #2      // output += 64 * 4 bytes
    sub     x2, x2, x5              // n -= 64

    cmp     x2, x5
    b.ge    .Lopt_loop

.Lopt_small:
    // Process remaining elements, 1 vector at a time
    cmp     x2, x4
    b.lt    .Lopt_tail

.Lopt_small_loop:
    ld1w    z0.s, p0/z, [x0]

    fadd    z4.s, z0.s, z30.s
    fsub    z16.s, z4.s, z30.s
    fsub    z20.s, z0.s, z16.s
    fexpa   z0.s, z4.s
    fmul    z4.s, z20.s, z31.s
    fmla    z0.s, p0/m, z0.s, z4.s

    st1w    z0.s, p0, [x1]

    add     x0, x0, x4, lsl #2
    add     x1, x1, x4, lsl #2
    sub     x2, x2, x4

    cmp     x2, x4
    b.ge    .Lopt_small_loop

.Lopt_tail:
    cbz     x2, .Lopt_done

    whilelt p1.s, xzr, x2
    ld1w    z0.s, p1/z, [x0]

    fadd    z4.s, z0.s, z30.s
    fsub    z16.s, z4.s, z30.s
    fsub    z20.s, z0.s, z16.s
    fexpa   z0.s, z4.s
    fmul    z4.s, z20.s, z31.s
    fmla    z0.s, p1/m, z0.s, z4.s

    st1w    z0.s, p1, [x1]

.Lopt_done:
    ret
.size exp2_fexpa_opt, .-exp2_fexpa_opt
