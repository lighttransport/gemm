/*
 * FEXPA exp2 with degree-2 Taylor polynomial
 * Using glibc-style Taylor coefficients
 */
.arch armv8.2-a+sve
.text
.align 4

.global exp2_fexpa_accurate
.type exp2_fexpa_accurate, %function
exp2_fexpa_accurate:
    ptrue   p0.s

    // shift = 0x48481fc0
    mov     w3, #0x1fc0
    movk    w3, #0x4848, lsl #16
    dup     z30.s, w3

    // c0 = ln(2) = 0x3f317218 (Taylor)
    mov     w3, #0x7218
    movk    w3, #0x3f31, lsl #16
    dup     z28.s, w3

    // c1 = ln(2)^2/2 = 0x3e75fdf0 (Taylor)
    mov     w3, #0xfdf0
    movk    w3, #0x3e75, lsl #16
    dup     z29.s, w3

    cntw    x4
    lsl     x5, x4, #2

    cmp     x2, x5
    b.lt    .Ld2t_small

.Ld2t_loop:
    ld1w    z0.s, p0/z, [x0]
    ld1w    z1.s, p0/z, [x0, #1, mul vl]
    ld1w    z2.s, p0/z, [x0, #2, mul vl]
    ld1w    z3.s, p0/z, [x0, #3, mul vl]

    fadd    z4.s, z0.s, z30.s
    fadd    z5.s, z1.s, z30.s
    fadd    z6.s, z2.s, z30.s
    fadd    z7.s, z3.s, z30.s

    fsub    z16.s, z4.s, z30.s
    fsub    z17.s, z5.s, z30.s
    fsub    z18.s, z6.s, z30.s
    fsub    z19.s, z7.s, z30.s

    fsub    z20.s, z0.s, z16.s
    fsub    z21.s, z1.s, z17.s
    fsub    z22.s, z2.s, z18.s
    fsub    z23.s, z3.s, z19.s

    fexpa   z0.s, z4.s
    fexpa   z1.s, z5.s
    fexpa   z2.s, z6.s
    fexpa   z3.s, z7.s

    // poly = r*(c0 + r*c1) using Horner
    movprfx z4, z28
    fmla    z4.s, p0/m, z20.s, z29.s
    movprfx z5, z28
    fmla    z5.s, p0/m, z21.s, z29.s
    movprfx z6, z28
    fmla    z6.s, p0/m, z22.s, z29.s
    movprfx z7, z28
    fmla    z7.s, p0/m, z23.s, z29.s

    fmul    z4.s, z4.s, z20.s
    fmul    z5.s, z5.s, z21.s
    fmul    z6.s, z6.s, z22.s
    fmul    z7.s, z7.s, z23.s

    fmla    z0.s, p0/m, z0.s, z4.s
    fmla    z1.s, p0/m, z1.s, z5.s
    fmla    z2.s, p0/m, z2.s, z6.s
    fmla    z3.s, p0/m, z3.s, z7.s

    st1w    z0.s, p0, [x1]
    st1w    z1.s, p0, [x1, #1, mul vl]
    st1w    z2.s, p0, [x1, #2, mul vl]
    st1w    z3.s, p0, [x1, #3, mul vl]

    add     x0, x0, x5, lsl #2
    add     x1, x1, x5, lsl #2
    sub     x2, x2, x5
    cmp     x2, x5
    b.ge    .Ld2t_loop

.Ld2t_small:
    cmp     x2, x4
    b.lt    .Ld2t_tail

.Ld2t_small_loop:
    ld1w    z0.s, p0/z, [x0]
    fadd    z4.s, z0.s, z30.s
    fsub    z16.s, z4.s, z30.s
    fsub    z20.s, z0.s, z16.s
    fexpa   z0.s, z4.s
    movprfx z5, z28
    fmla    z5.s, p0/m, z20.s, z29.s
    fmul    z5.s, z5.s, z20.s
    fmla    z0.s, p0/m, z0.s, z5.s
    st1w    z0.s, p0, [x1]
    add     x0, x0, x4, lsl #2
    add     x1, x1, x4, lsl #2
    sub     x2, x2, x4
    cmp     x2, x4
    b.ge    .Ld2t_small_loop

.Ld2t_tail:
    cbz     x2, .Ld2t_done
    whilelt p1.s, xzr, x2
    ld1w    z0.s, p1/z, [x0]
    fadd    z4.s, z0.s, z30.s
    fsub    z16.s, z4.s, z30.s
    fsub    z20.s, z0.s, z16.s
    fexpa   z0.s, z4.s
    movprfx z5, z28
    fmla    z5.s, p1/m, z20.s, z29.s
    fmul    z5.s, z5.s, z20.s
    fmla    z0.s, p1/m, z0.s, z5.s
    st1w    z0.s, p1, [x1]

.Ld2t_done:
    ret
.size exp2_fexpa_accurate, .-exp2_fexpa_accurate
