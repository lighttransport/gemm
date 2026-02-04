/*
 * exp2f using FEXPA - 8x unroll
 *
 * A64FX latencies (all 9 cycles): fadd, fsub, fmul, fmla
 * fexpa: 4 cycles
 *
 * Critical path: fadd→fsub→fsub→fmul→fmla = 45 cycles
 * Throughput: 6 FLA ops / 2 pipes = 3 cycles/vector
 * Need 45/3 = 15 vectors in flight for peak throughput
 *
 * 8x unroll: 8 vectors, should help but not optimal
 */

.arch armv8.2-a+sve
.text
.align 4

.global exp2_fexpa_u8
.type exp2_fexpa_u8, %function
exp2_fexpa_u8:
    ptrue   p0.s

    mov     w3, #0x1fc0
    movk    w3, #0x4848, lsl #16
    dup     z30.s, w3

    mov     w3, #0x7211
    movk    w3, #0x3f31, lsl #16
    dup     z31.s, w3

    cntw    x4
    lsl     x5, x4, #3              // 8 * VL

    cmp     x2, x5
    b.lt    .Lu8_small

.Lu8_loop:
    // Load 8 vectors
    ld1w    z0.s, p0/z, [x0]
    ld1w    z1.s, p0/z, [x0, #1, mul vl]
    ld1w    z2.s, p0/z, [x0, #2, mul vl]
    ld1w    z3.s, p0/z, [x0, #3, mul vl]
    ld1w    z4.s, p0/z, [x0, #4, mul vl]
    ld1w    z5.s, p0/z, [x0, #5, mul vl]
    ld1w    z6.s, p0/z, [x0, #6, mul vl]
    ld1w    z7.s, p0/z, [x0, #7, mul vl]

    // z = x + shift
    fadd    z16.s, z0.s, z30.s
    fadd    z17.s, z1.s, z30.s
    fadd    z18.s, z2.s, z30.s
    fadd    z19.s, z3.s, z30.s
    fadd    z20.s, z4.s, z30.s
    fadd    z21.s, z5.s, z30.s
    fadd    z22.s, z6.s, z30.s
    fadd    z23.s, z7.s, z30.s

    // n = z - shift
    fsub    z24.s, z16.s, z30.s
    fsub    z25.s, z17.s, z30.s
    fsub    z26.s, z18.s, z30.s
    fsub    z27.s, z19.s, z30.s
    fsub    z28.s, z20.s, z30.s
    fsub    z29.s, z21.s, z30.s

    // r = x - n (first 6)
    fsub    z0.s, z0.s, z24.s
    fsub    z1.s, z1.s, z25.s
    fsub    z2.s, z2.s, z26.s
    fsub    z3.s, z3.s, z27.s
    fsub    z4.s, z4.s, z28.s
    fsub    z5.s, z5.s, z29.s

    // n6, n7 and r6, r7
    fsub    z24.s, z22.s, z30.s
    fsub    z25.s, z23.s, z30.s
    fsub    z6.s, z6.s, z24.s
    fsub    z7.s, z7.s, z25.s

    // scale = fexpa(z)
    fexpa   z24.s, z16.s
    fexpa   z25.s, z17.s
    fexpa   z26.s, z18.s
    fexpa   z27.s, z19.s
    fexpa   z28.s, z20.s
    fexpa   z29.s, z21.s
    fexpa   z16.s, z22.s
    fexpa   z17.s, z23.s

    // poly = r * c0
    fmul    z0.s, z0.s, z31.s
    fmul    z1.s, z1.s, z31.s
    fmul    z2.s, z2.s, z31.s
    fmul    z3.s, z3.s, z31.s
    fmul    z4.s, z4.s, z31.s
    fmul    z5.s, z5.s, z31.s
    fmul    z6.s, z6.s, z31.s
    fmul    z7.s, z7.s, z31.s

    // result = scale + scale * poly
    fmla    z24.s, p0/m, z24.s, z0.s
    fmla    z25.s, p0/m, z25.s, z1.s
    fmla    z26.s, p0/m, z26.s, z2.s
    fmla    z27.s, p0/m, z27.s, z3.s
    fmla    z28.s, p0/m, z28.s, z4.s
    fmla    z29.s, p0/m, z29.s, z5.s
    fmla    z16.s, p0/m, z16.s, z6.s
    fmla    z17.s, p0/m, z17.s, z7.s

    // Store
    st1w    z24.s, p0, [x1]
    st1w    z25.s, p0, [x1, #1, mul vl]
    st1w    z26.s, p0, [x1, #2, mul vl]
    st1w    z27.s, p0, [x1, #3, mul vl]
    st1w    z28.s, p0, [x1, #4, mul vl]
    st1w    z29.s, p0, [x1, #5, mul vl]
    st1w    z16.s, p0, [x1, #6, mul vl]
    st1w    z17.s, p0, [x1, #7, mul vl]

    add     x0, x0, x5, lsl #2
    add     x1, x1, x5, lsl #2
    sub     x2, x2, x5
    cmp     x2, x5
    b.ge    .Lu8_loop

.Lu8_small:
    cmp     x2, x4
    b.lt    .Lu8_tail

.Lu8_small_loop:
    ld1w    z0.s, p0/z, [x0]
    fadd    z4.s, z0.s, z30.s
    fsub    z16.s, z4.s, z30.s
    fsub    z0.s, z0.s, z16.s
    fexpa   z16.s, z4.s
    fmul    z0.s, z0.s, z31.s
    fmla    z16.s, p0/m, z16.s, z0.s
    st1w    z16.s, p0, [x1]
    add     x0, x0, x4, lsl #2
    add     x1, x1, x4, lsl #2
    sub     x2, x2, x4
    cmp     x2, x4
    b.ge    .Lu8_small_loop

.Lu8_tail:
    cbz     x2, .Lu8_done
    whilelt p1.s, xzr, x2
    ld1w    z0.s, p1/z, [x0]
    fadd    z4.s, z0.s, z30.s
    fsub    z16.s, z4.s, z30.s
    fsub    z0.s, z0.s, z16.s
    fexpa   z16.s, z4.s
    fmul    z0.s, z0.s, z31.s
    fmla    z16.s, p1/m, z16.s, z0.s
    st1w    z16.s, p1, [x1]

.Lu8_done:
    ret
.size exp2_fexpa_u8, .-exp2_fexpa_u8
