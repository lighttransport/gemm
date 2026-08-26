/* Lossless semantic-nibble IQ2_XS Q8 SDOT kernel, 32 output rows.
 *
 * x0 out, x1 paired semantic nibbles, x2 four scale vectors/group,
 * x3 FP32 block scales, x4 Q8 activation, x5 unused, w6 K256 blocks,
 * s0 activation dequant scale.
 */
    .arch armv8.2-a+sve
    .text
    .align 6
    .global k3_iq2_pair32_asm
    .type k3_iq2_pair32_asm, @function

    .macro IQ2_DOT_PAIR wr, xa, lane, alo, ahi
    mov     z22.d, z\wr\().d
    lsr     z\wr\().b, p0/m, z\wr\().b, #4
    and     z22.b, p0/m, z22.b, z23.b
    tbl     z\wr\().b, { z31.b }, z\wr\().b
    tbl     z22.b, { z31.b }, z22.b
    sdot    z\alo\().s, z22.b, z\xa\().b[\lane]
    sdot    z\ahi\().s, z\wr\().b, z\xa\().b[\lane]
    .endm

k3_iq2_pair32_asm:
    ptrue   p0.b
    ptrue   p1.s
    mov     z28.s, #0
    mov     z29.s, #0
    mov     z23.b, #15
    fmov    w9, s0
    dup     z27.s, w9
    fmov    z30.s, #0.125
    adrp    x9, .Liq2_semantic_table
    add     x9, x9, :lo12:.Liq2_semantic_table
    ld1b    { z31.b }, p0/z, [x9]

.Liq2_block:
    mov     z26.s, #0
    mov     z19.s, #0
    mov     w8, #8
    ld1rqb  { z0.b }, p0/z, [x4]
    ld1rqb  { z1.b }, p0/z, [x4, #16]
    ld1b    { z2.b }, p0/z, [x1, #0, mul vl]
    ld1b    { z3.b }, p0/z, [x1, #1, mul vl]
    ld1b    { z4.b }, p0/z, [x1, #2, mul vl]
    ld1b    { z5.b }, p0/z, [x1, #3, mul vl]
    ld1b    { z6.b }, p0/z, [x1, #4, mul vl]
    ld1b    { z7.b }, p0/z, [x1, #5, mul vl]
    ld1b    { z16.b }, p0/z, [x1, #6, mul vl]
    ld1b    { z17.b }, p0/z, [x1, #7, mul vl]

.Liq2_group:
    mov     z24.s, #0
    mov     z25.s, #0
    mov     z20.s, #0
    mov     z21.s, #0
    IQ2_DOT_PAIR 2, 0, 0, 24, 25
    IQ2_DOT_PAIR 3, 0, 1, 24, 25
    IQ2_DOT_PAIR 4, 0, 2, 24, 25
    IQ2_DOT_PAIR 5, 0, 3, 24, 25
    IQ2_DOT_PAIR 6, 1, 0, 20, 21
    IQ2_DOT_PAIR 7, 1, 1, 20, 21
    IQ2_DOT_PAIR 16, 1, 2, 20, 21
    IQ2_DOT_PAIR 17, 1, 3, 20, 21

    /* Once the current operands have fed SDOT, reuse their registers to
     * fetch the next group before scale/accumulation work.  Those independent
     * instructions cover the load-to-use latency without a second bank. */
    subs    w8, w8, #1
    add     x1, x1, #512
    add     x4, x4, #32
    b.eq    .Liq2_no_preload
    ld1rqb  { z0.b }, p0/z, [x4]
    ld1rqb  { z1.b }, p0/z, [x4, #16]
    ld1b    { z2.b }, p0/z, [x1, #0, mul vl]
    ld1b    { z3.b }, p0/z, [x1, #1, mul vl]
    ld1b    { z4.b }, p0/z, [x1, #2, mul vl]
    ld1b    { z5.b }, p0/z, [x1, #3, mul vl]
    ld1b    { z6.b }, p0/z, [x1, #4, mul vl]
    ld1b    { z7.b }, p0/z, [x1, #5, mul vl]
    ld1b    { z16.b }, p0/z, [x1, #6, mul vl]
    ld1b    { z17.b }, p0/z, [x1, #7, mul vl]
.Liq2_no_preload:

    ld1sb   { z18.s }, p1/z, [x2]
    mla     z26.s, p1/m, z24.s, z18.s
    add     x10, x2, #16
    ld1sb   { z18.s }, p1/z, [x10]
    mla     z26.s, p1/m, z20.s, z18.s
    add     x10, x2, #32
    ld1sb   { z18.s }, p1/z, [x10]
    mla     z19.s, p1/m, z25.s, z18.s
    add     x10, x2, #48
    ld1sb   { z18.s }, p1/z, [x10]
    mla     z19.s, p1/m, z21.s, z18.s

    add     x2, x2, #64
    b.ne    .Liq2_group

    scvtf   z24.s, p1/m, z26.s
    scvtf   z25.s, p1/m, z19.s
    ld1w    { z16.s }, p1/z, [x3, #0, mul vl]
    ld1w    { z17.s }, p1/z, [x3, #1, mul vl]
    fmul    z16.s, p1/m, z16.s, z30.s
    fmul    z17.s, p1/m, z17.s, z30.s
    fmla    z28.s, p1/m, z24.s, z16.s
    fmla    z29.s, p1/m, z25.s, z17.s
    add     x3, x3, #128
    subs    w6, w6, #1
    b.ne    .Liq2_block

    fmul    z28.s, p1/m, z28.s, z27.s
    fmul    z29.s, p1/m, z29.s, z27.s
    st1w    { z28.s }, p1, [x0, #0, mul vl]
    st1w    { z29.s }, p1, [x0, #1, mul vl]
    ret
    .size k3_iq2_pair32_asm, .-k3_iq2_pair32_asm

    .section .rodata
    .align 6
.Liq2_semantic_table:
    .byte 8, 25, 43, -8, -25, -43
    .zero 58
