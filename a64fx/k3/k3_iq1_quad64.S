/* Lossless IQ1_S signed-2bit Q8 SDOT kernel for 64 output rows.
 * One packed byte contains the same K value for four 16-row groups.
 */
    .arch armv8.2-a+sve
    .text
    .align 6
    .global k3_iq1_quad64_asm
    .type k3_iq1_quad64_asm, @function

    .macro IQ1_QUAD_DOT wr, xa, lane
    dup     z23.s, z\xa\().s[\lane]
    mov     z16.d, z\wr\().d
    mov     z17.d, z\wr\().d
    mov     z22.d, z\wr\().d
    lsl     z16.b, p0/m, z16.b, #6
    lsl     z17.b, p0/m, z17.b, #4
    lsl     z22.b, p0/m, z22.b, #2
    asr     z\wr\().b, p0/m, z\wr\().b, #6
    asr     z16.b, p0/m, z16.b, #6
    asr     z17.b, p0/m, z17.b, #6
    asr     z22.b, p0/m, z22.b, #6
    sdot    z24.s, z16.b, z23.b
    sdot    z25.s, z17.b, z23.b
    sdot    z20.s, z22.b, z23.b
    sdot    z21.s, z\wr\().b, z23.b
    .endm

k3_iq1_quad64_asm:
    sub     sp, sp, #64
    stp     q8, q9, [sp]
    stp     q10, q11, [sp, #32]
    ptrue   p0.b
    ptrue   p1.s
    mov     z28.s, #0
    mov     z29.s, #0
    mov     z8.s, #0
    mov     z9.s, #0
    fmov    w10, s0
    dup     z31.s, w10
    fmov    z30.s, #0.125

.Liq1q_block:
    mov     z26.s, #0
    mov     z27.s, #0
    mov     z18.s, #0
    mov     z19.s, #0
    mov     w12, #8
    ld1rqb  { z0.b }, p0/z, [x4]
    ld1rqb  { z1.b }, p0/z, [x4, #16]
    ld1b    { z2.b }, p0/z, [x1, #0, mul vl]
    ld1b    { z3.b }, p0/z, [x1, #1, mul vl]
    ld1b    { z4.b }, p0/z, [x1, #2, mul vl]
    ld1b    { z5.b }, p0/z, [x1, #3, mul vl]
    ld1b    { z6.b }, p0/z, [x1, #4, mul vl]
    ld1b    { z7.b }, p0/z, [x1, #5, mul vl]
    ld1b    { z10.b }, p0/z, [x1, #6, mul vl]
    ld1b    { z11.b }, p0/z, [x1, #7, mul vl]

.Liq1q_group:
    mov     z24.s, #0
    mov     z25.s, #0
    mov     z20.s, #0
    mov     z21.s, #0
    IQ1_QUAD_DOT 2, 0, 0
    IQ1_QUAD_DOT 3, 0, 1
    IQ1_QUAD_DOT 4, 0, 2
    IQ1_QUAD_DOT 5, 0, 3
    IQ1_QUAD_DOT 6, 1, 0
    IQ1_QUAD_DOT 7, 1, 1
    IQ1_QUAD_DOT 10, 1, 2
    IQ1_QUAD_DOT 11, 1, 3

    subs    w12, w12, #1
    add     x1, x1, #512
    add     x4, x4, #32
    b.eq    .Liq1q_no_preload
    ld1rqb  { z0.b }, p0/z, [x4]
    ld1rqb  { z1.b }, p0/z, [x4, #16]
    ld1b    { z2.b }, p0/z, [x1, #0, mul vl]
    ld1b    { z3.b }, p0/z, [x1, #1, mul vl]
    ld1b    { z4.b }, p0/z, [x1, #2, mul vl]
    ld1b    { z5.b }, p0/z, [x1, #3, mul vl]
    ld1b    { z6.b }, p0/z, [x1, #4, mul vl]
    ld1b    { z7.b }, p0/z, [x1, #5, mul vl]
    ld1b    { z10.b }, p0/z, [x1, #6, mul vl]
    ld1b    { z11.b }, p0/z, [x1, #7, mul vl]
.Liq1q_no_preload:

    ld1rw   { z23.s }, p1/z, [x5]
    ld1sb   { z16.s }, p1/z, [x2]
    abs     z17.s, p1/m, z16.s
    lsl     z24.s, p1/m, z24.s, #3
    mla     z26.s, p1/m, z24.s, z17.s
    mla     z26.s, p1/m, z23.s, z16.s
    add     x10, x2, #16
    ld1sb   { z16.s }, p1/z, [x10]
    abs     z17.s, p1/m, z16.s
    lsl     z25.s, p1/m, z25.s, #3
    mla     z27.s, p1/m, z25.s, z17.s
    mla     z27.s, p1/m, z23.s, z16.s
    add     x10, x2, #32
    ld1sb   { z16.s }, p1/z, [x10]
    abs     z17.s, p1/m, z16.s
    lsl     z20.s, p1/m, z20.s, #3
    mla     z18.s, p1/m, z20.s, z17.s
    mla     z18.s, p1/m, z23.s, z16.s
    add     x10, x2, #48
    ld1sb   { z16.s }, p1/z, [x10]
    abs     z17.s, p1/m, z16.s
    lsl     z21.s, p1/m, z21.s, #3
    mla     z19.s, p1/m, z21.s, z17.s
    mla     z19.s, p1/m, z23.s, z16.s
    add     x2, x2, #64
    add     x5, x5, #4
    b.ne    .Liq1q_group

    scvtf   z24.s, p1/m, z26.s
    scvtf   z25.s, p1/m, z27.s
    scvtf   z20.s, p1/m, z18.s
    scvtf   z21.s, p1/m, z19.s
    ld1w    { z16.s }, p1/z, [x3, #0, mul vl]
    ld1w    { z17.s }, p1/z, [x3, #1, mul vl]
    ld1w    { z22.s }, p1/z, [x3, #2, mul vl]
    ld1w    { z23.s }, p1/z, [x3, #3, mul vl]
    fmul    z16.s, p1/m, z16.s, z30.s
    fmul    z17.s, p1/m, z17.s, z30.s
    fmul    z22.s, p1/m, z22.s, z30.s
    fmul    z23.s, p1/m, z23.s, z30.s
    fmla    z28.s, p1/m, z24.s, z16.s
    fmla    z29.s, p1/m, z25.s, z17.s
    fmla    z8.s, p1/m, z20.s, z22.s
    fmla    z9.s, p1/m, z21.s, z23.s
    add     x3, x3, #256
    subs    w6, w6, #1
    b.ne    .Liq1q_block

    fmul    z28.s, p1/m, z28.s, z31.s
    fmul    z29.s, p1/m, z29.s, z31.s
    fmul    z8.s, p1/m, z8.s, z31.s
    fmul    z9.s, p1/m, z9.s, z31.s
    st1w    { z28.s }, p1, [x0, #0, mul vl]
    st1w    { z29.s }, p1, [x0, #1, mul vl]
    st1w    { z8.s }, p1, [x0, #2, mul vl]
    st1w    { z9.s }, p1, [x0, #3, mul vl]
    ldp     q10, q11, [sp, #32]
    ldp     q8, q9, [sp]
    add     sp, sp, #64
    ret
    .size k3_iq1_quad64_asm, .-k3_iq1_quad64_asm
