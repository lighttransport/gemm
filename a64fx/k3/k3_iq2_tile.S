/* IQ2_XS packed 16-row tile microkernel.
 *
 * Arguments:
 *   x0  float out[16]
 *   x1  packed byte weights, [nb][8][8][64]
 *   x2  compact int8 scales, [nb][8][32] (s0[16], s1[16])
 *   x3  FP16 row scales, [nb][16]
 *   x4  q8 activation
 *   w5  number of 256-element blocks
 *   s0  activation dequant scale
 *
 * The C path remains the default.  This routine is deliberately standalone
 * while its ABI and numerical behavior are being gated. */
    .arch armv8.2-a+sve
    .text
    .align 6
    .global k3_iq2_xs_tile_asm
    .type k3_iq2_xs_tile_asm, @function

k3_iq2_xs_tile_asm:
    ptrue   p0.b
    ptrue   p1.s
    mov     z16.s, #0               // block accumulator
    mov     z17.s, #0               // output accumulator
    fmov    w15, s0                 // activation scale is a FP ABI argument
    dup     z15.s, w15
    mov     x6, x1
    mov     x7, x2
    mov     x8, x3
    mov     x9, x4

.Liq2_block:
    mov     z16.s, #0
    mov     x10, x6
    mov     x11, x7
    mov     x12, x9
    mov     w13, #8

.Liq2_group:
    ld1rw   { z24.s }, p1/z, [x12]
    ld1rw   { z25.s }, p1/z, [x12, #4]
    ld1rw   { z26.s }, p1/z, [x12, #8]
    ld1rw   { z27.s }, p1/z, [x12, #12]
    ld1rw   { z28.s }, p1/z, [x12, #16]
    ld1rw   { z29.s }, p1/z, [x12, #20]
    ld1rw   { z30.s }, p1/z, [x12, #24]
    ld1rw   { z31.s }, p1/z, [x12, #28]
    ld1b    { z0.b }, p0/z, [x10, #0, mul vl]
    ld1b    { z1.b }, p0/z, [x10, #1, mul vl]
    ld1b    { z2.b }, p0/z, [x10, #2, mul vl]
    ld1b    { z3.b }, p0/z, [x10, #3, mul vl]
    ld1b    { z4.b }, p0/z, [x10, #4, mul vl]
    ld1b    { z5.b }, p0/z, [x10, #5, mul vl]
    ld1b    { z6.b }, p0/z, [x10, #6, mul vl]
    ld1b    { z7.b }, p0/z, [x10, #7, mul vl]
    mov     z18.s, #0
    mov     z19.s, #0
    sdot    z18.s, z0.b, z24.b
    sdot    z18.s, z1.b, z25.b
    sdot    z18.s, z2.b, z26.b
    sdot    z18.s, z3.b, z27.b
    sdot    z19.s, z4.b, z28.b
    sdot    z19.s, z5.b, z29.b
    sdot    z19.s, z6.b, z30.b
    sdot    z19.s, z7.b, z31.b
    ld1sb   { z20.s }, p1/z, [x11]
    add     x14, x11, #16
    ld1sb   { z21.s }, p1/z, [x14]
    scvtf   z18.s, p1/m, z18.s
    scvtf   z19.s, p1/m, z19.s
    scvtf   z20.s, p1/m, z20.s
    scvtf   z21.s, p1/m, z21.s
    fmla    z16.s, p1/m, z18.s, z20.s
    fmla    z16.s, p1/m, z19.s, z21.s
    add     x10, x10, #512
    add     x11, x11, #32
    add     x12, x12, #32
    subs    w13, w13, #1
    b.ne    .Liq2_group

    ld1h    { z18.h }, p1/z, [x8]
    fcvt    z18.s, p1/m, z18.h
    fmla    z17.s, p1/m, z16.s, z18.s
    add     x6, x6, #4096
    add     x7, x7, #256
    add     x8, x8, #32
    add     x9, x9, #256
    subs    w5, w5, #1
    b.ne    .Liq2_block

    fmul    z17.s, p1/m, z17.s, z15.s
    fmov    z14.s, #0.125
    fmul    z17.s, p1/m, z17.s, z14.s
    st1w    { z17.s }, p1, [x0]
    ret
    .size k3_iq2_xs_tile_asm, .-k3_iq2_xs_tile_asm
