/*
 * exp2f using FEXPA with full degree-4 polynomial (glibc algorithm)
 * Achieves ~1 ULP accuracy
 *
 * Register allocation (avoiding callee-saved z8-z15):
 *   z0-z3:   input x, then scale, then result
 *   z4-z7:   z = x + shift, then polynomial temps
 *   z16-z19: n = z - shift
 *   z20-z23: r = x - n
 *   z24:     r^2
 *   z25-z29: polynomial coefficients c0-c4
 *   z30:     shift constant
 *   z31:     scratch
 */

.arch armv8.2-a+sve

.data
.align 4
const_shift:    .word 0x48481fc0      // shift for FEXPA
const_c0:       .word 0x3f317211      // ln(2) = 0x1.62e422p-1
const_c1:       .word 0x3e75fcde      // 0x1.ebf9bcp-3
const_c2:       .word 0x3de35e99      // 0x1.c6bd32p-5
const_c3:       .word 0x3c9e74f2      // 0x1.3ce9e4p-7
const_c4:       .word 0x3a4ccbbd      // 0x1.59977ap-10

.text
.align 4

.global exp2_fexpa_glibc
.type exp2_fexpa_glibc, %function
exp2_fexpa_glibc:
    // x0 = input, x1 = output, x2 = n elements

    ptrue   p0.s

    // Load constants
    adrp    x3, const_shift
    add     x3, x3, :lo12:const_shift
    ld1rw   z30.s, p0/z, [x3]       // shift

    adrp    x3, const_c0
    add     x3, x3, :lo12:const_c0
    ld1rw   z25.s, p0/z, [x3]       // c0

    adrp    x3, const_c1
    add     x3, x3, :lo12:const_c1
    ld1rw   z26.s, p0/z, [x3]       // c1

    adrp    x3, const_c2
    add     x3, x3, :lo12:const_c2
    ld1rw   z27.s, p0/z, [x3]       // c2

    adrp    x3, const_c3
    add     x3, x3, :lo12:const_c3
    ld1rw   z28.s, p0/z, [x3]       // c3

    adrp    x3, const_c4
    add     x3, x3, :lo12:const_c4
    ld1rw   z29.s, p0/z, [x3]       // c4

    cntw    x4                      // VL (16)
    lsl     x5, x4, #2              // 4 * VL

    cmp     x2, x5
    b.lt    .Lglibc_small

.Lglibc_loop:
    // Load 4 vectors
    ld1w    z0.s, p0/z, [x0]
    ld1w    z1.s, p0/z, [x0, #1, mul vl]
    ld1w    z2.s, p0/z, [x0, #2, mul vl]
    ld1w    z3.s, p0/z, [x0, #3, mul vl]

    // z = x + shift
    fadd    z4.s, z0.s, z30.s
    fadd    z5.s, z1.s, z30.s
    fadd    z6.s, z2.s, z30.s
    fadd    z7.s, z3.s, z30.s

    // n = z - shift
    fsub    z16.s, z4.s, z30.s
    fsub    z17.s, z5.s, z30.s
    fsub    z18.s, z6.s, z30.s
    fsub    z19.s, z7.s, z30.s

    // r = x - n
    fsub    z20.s, z0.s, z16.s
    fsub    z21.s, z1.s, z17.s
    fsub    z22.s, z2.s, z18.s
    fsub    z23.s, z3.s, z19.s

    // scale = fexpa(z)
    fexpa   z0.s, z4.s
    fexpa   z1.s, z5.s
    fexpa   z2.s, z6.s
    fexpa   z3.s, z7.s

    // Polynomial evaluation using Horner's scheme
    // poly = r*(c0 + r*(c1 + r*(c2 + r*(c3 + r*c4))))
    // Evaluated as: poly = r*c0 + r^2*(c1 + r*(c2 + r*(c3 + r*c4)))

    // Process each vector sequentially (register pressure)
    // Vector 0:
    fmul    z24.s, z20.s, z20.s     // r^2
    movprfx z31, z28
    fmla    z31.s, p0/m, z20.s, z29.s   // c3 + r*c4
    movprfx z4, z27
    fmla    z4.s, p0/m, z20.s, z31.s    // c2 + r*(c3+r*c4)
    movprfx z31, z26
    fmla    z31.s, p0/m, z20.s, z4.s    // c1 + r*(c2+...)
    fmul    z4.s, z20.s, z25.s          // r*c0
    fmla    z4.s, p0/m, z24.s, z31.s    // poly = r*c0 + r^2*(c1+...)
    fmla    z0.s, p0/m, z0.s, z4.s      // result = scale + scale*poly

    // Vector 1:
    fmul    z24.s, z21.s, z21.s
    movprfx z31, z28
    fmla    z31.s, p0/m, z21.s, z29.s
    movprfx z4, z27
    fmla    z4.s, p0/m, z21.s, z31.s
    movprfx z31, z26
    fmla    z31.s, p0/m, z21.s, z4.s
    fmul    z4.s, z21.s, z25.s
    fmla    z4.s, p0/m, z24.s, z31.s
    fmla    z1.s, p0/m, z1.s, z4.s

    // Vector 2:
    fmul    z24.s, z22.s, z22.s
    movprfx z31, z28
    fmla    z31.s, p0/m, z22.s, z29.s
    movprfx z4, z27
    fmla    z4.s, p0/m, z22.s, z31.s
    movprfx z31, z26
    fmla    z31.s, p0/m, z22.s, z4.s
    fmul    z4.s, z22.s, z25.s
    fmla    z4.s, p0/m, z24.s, z31.s
    fmla    z2.s, p0/m, z2.s, z4.s

    // Vector 3:
    fmul    z24.s, z23.s, z23.s
    movprfx z31, z28
    fmla    z31.s, p0/m, z23.s, z29.s
    movprfx z4, z27
    fmla    z4.s, p0/m, z23.s, z31.s
    movprfx z31, z26
    fmla    z31.s, p0/m, z23.s, z4.s
    fmul    z4.s, z23.s, z25.s
    fmla    z4.s, p0/m, z24.s, z31.s
    fmla    z3.s, p0/m, z3.s, z4.s

    // Store
    st1w    z0.s, p0, [x1]
    st1w    z1.s, p0, [x1, #1, mul vl]
    st1w    z2.s, p0, [x1, #2, mul vl]
    st1w    z3.s, p0, [x1, #3, mul vl]

    add     x0, x0, x5, lsl #2
    add     x1, x1, x5, lsl #2
    sub     x2, x2, x5

    cmp     x2, x5
    b.ge    .Lglibc_loop

.Lglibc_small:
    cmp     x2, x4
    b.lt    .Lglibc_tail

.Lglibc_small_loop:
    ld1w    z0.s, p0/z, [x0]

    fadd    z4.s, z0.s, z30.s
    fsub    z16.s, z4.s, z30.s
    fsub    z20.s, z0.s, z16.s
    fexpa   z0.s, z4.s

    fmul    z24.s, z20.s, z20.s
    movprfx z31, z28
    fmla    z31.s, p0/m, z20.s, z29.s
    movprfx z4, z27
    fmla    z4.s, p0/m, z20.s, z31.s
    movprfx z31, z26
    fmla    z31.s, p0/m, z20.s, z4.s
    fmul    z4.s, z20.s, z25.s
    fmla    z4.s, p0/m, z24.s, z31.s
    fmla    z0.s, p0/m, z0.s, z4.s

    st1w    z0.s, p0, [x1]

    add     x0, x0, x4, lsl #2
    add     x1, x1, x4, lsl #2
    sub     x2, x2, x4

    cmp     x2, x4
    b.ge    .Lglibc_small_loop

.Lglibc_tail:
    cbz     x2, .Lglibc_done

    whilelt p1.s, xzr, x2
    ld1w    z0.s, p1/z, [x0]

    fadd    z4.s, z0.s, z30.s
    fsub    z16.s, z4.s, z30.s
    fsub    z20.s, z0.s, z16.s
    fexpa   z0.s, z4.s

    fmul    z24.s, z20.s, z20.s
    movprfx z31, z28
    fmla    z31.s, p1/m, z20.s, z29.s
    movprfx z4, z27
    fmla    z4.s, p1/m, z20.s, z31.s
    movprfx z31, z26
    fmla    z31.s, p1/m, z20.s, z4.s
    fmul    z4.s, z20.s, z25.s
    fmla    z4.s, p1/m, z24.s, z31.s
    fmla    z0.s, p1/m, z0.s, z4.s

    st1w    z0.s, p1, [x1]

.Lglibc_done:
    ret
.size exp2_fexpa_glibc, .-exp2_fexpa_glibc
