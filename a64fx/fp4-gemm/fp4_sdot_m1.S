    .arch armv8.2-a+sve
    .text
    .align 4
/* args: q, activation, fp16 row scales, activation scales, output,
 *       K, weight K group, activation K group. */
    .global fp4_i8_sdot_m1_asm
    .type fp4_i8_sdot_m1_asm, %function
fp4_i8_sdot_m1_asm:
    sub     sp, sp, #128
    stp     q8, q9, [sp]
    stp     q10, q11, [sp, #32]
    stp     q12, q13, [sp, #64]
    stp     q14, q15, [sp, #96]
    ldp     x1, x2, [x0]
    ldp     x3, x4, [x0, #16]
    ldr     x5, [x0, #32]
    ldr     w6, [x0, #40]
    ldr     w14, [x0, #44]
    ldr     w15, [x0, #48]
    ptrue   p0.b
    ptrue   p1.s
    eor     z8.d, z8.d, z8.d
    eor     z9.d, z9.d, z9.d
    eor     z10.d, z10.d, z10.d
    eor     z11.d, z11.d, z11.d
    eor     z12.d, z12.d, z12.d
    eor     z13.d, z13.d, z13.d
    eor     z14.d, z14.d, z14.d
    eor     z15.d, z15.d, z15.d
    udiv    w6, w6, w14
.Lfp4sdot_wblock:
    udiv    w8, w14, w15
.Lfp4sdot_agroup:
    eor     z0.d, z0.d, z0.d
    eor     z1.d, z1.d, z1.d
    eor     z2.d, z2.d, z2.d
    eor     z3.d, z3.d, z3.d
    eor     z4.d, z4.d, z4.d
    eor     z5.d, z5.d, z5.d
    eor     z6.d, z6.d, z6.d
    eor     z7.d, z7.d, z7.d
    lsr     w7, w15, #2
.Lfp4sdot_k4:
    ld1b    z16.b, p0/z, [x1]
    ld1b    z17.b, p0/z, [x1, #1, mul vl]
    ld1b    z18.b, p0/z, [x1, #2, mul vl]
    ld1b    z19.b, p0/z, [x1, #3, mul vl]
    ld1b    z20.b, p0/z, [x1, #4, mul vl]
    ld1b    z21.b, p0/z, [x1, #5, mul vl]
    ld1b    z22.b, p0/z, [x1, #6, mul vl]
    ld1b    z23.b, p0/z, [x1, #7, mul vl]
    ld1rw   z24.s, p0/z, [x2]
    sdot    z0.s, z16.b, z24.b
    sdot    z1.s, z17.b, z24.b
    sdot    z2.s, z18.b, z24.b
    sdot    z3.s, z19.b, z24.b
    sdot    z4.s, z20.b, z24.b
    sdot    z5.s, z21.b, z24.b
    sdot    z6.s, z22.b, z24.b
    sdot    z7.s, z23.b, z24.b
    add     x1, x1, #512
    add     x2, x2, #4
    subs    w7, w7, #1
    b.ne    .Lfp4sdot_k4
    ld1rw   z24.s, p1/z, [x4]
    scvtf   z0.s, p1/m, z0.s
    scvtf   z1.s, p1/m, z1.s
    scvtf   z2.s, p1/m, z2.s
    scvtf   z3.s, p1/m, z3.s
    scvtf   z4.s, p1/m, z4.s
    scvtf   z5.s, p1/m, z5.s
    scvtf   z6.s, p1/m, z6.s
    scvtf   z7.s, p1/m, z7.s
    ld1h    z16.h, p0/z, [x3]
    ld1h    z19.h, p0/z, [x3, #1, mul vl]
    ld1h    z22.h, p0/z, [x3, #2, mul vl]
    ld1h    z25.h, p0/z, [x3, #3, mul vl]
    uunpklo z17.s, z16.h
    uunpkhi z18.s, z16.h
    uunpklo z20.s, z19.h
    uunpkhi z21.s, z19.h
    uunpklo z23.s, z22.h
    uunpkhi z26.s, z22.h
    uunpklo z27.s, z25.h
    uunpkhi z28.s, z25.h
    fcvt    z17.s, p1/m, z17.h
    fcvt    z18.s, p1/m, z18.h
    fcvt    z20.s, p1/m, z20.h
    fcvt    z21.s, p1/m, z21.h
    fcvt    z23.s, p1/m, z23.h
    fcvt    z26.s, p1/m, z26.h
    fcvt    z27.s, p1/m, z27.h
    fcvt    z28.s, p1/m, z28.h
    fmul    z17.s, p1/m, z17.s, z24.s
    fmul    z18.s, p1/m, z18.s, z24.s
    fmul    z20.s, p1/m, z20.s, z24.s
    fmul    z21.s, p1/m, z21.s, z24.s
    fmul    z23.s, p1/m, z23.s, z24.s
    fmul    z26.s, p1/m, z26.s, z24.s
    fmul    z27.s, p1/m, z27.s, z24.s
    fmul    z28.s, p1/m, z28.s, z24.s
    fmla    z8.s, p1/m, z0.s, z17.s
    fmla    z9.s, p1/m, z1.s, z18.s
    fmla    z10.s, p1/m, z2.s, z20.s
    fmla    z11.s, p1/m, z3.s, z21.s
    fmla    z12.s, p1/m, z4.s, z23.s
    fmla    z13.s, p1/m, z5.s, z26.s
    fmla    z14.s, p1/m, z6.s, z27.s
    fmla    z15.s, p1/m, z7.s, z28.s
    add     x4, x4, #4
    subs    w8, w8, #1
    b.ne    .Lfp4sdot_agroup
    add     x3, x3, #256
    subs    w6, w6, #1
    b.ne    .Lfp4sdot_wblock
    st1w    z8.s, p1, [x5]
    st1w    z9.s, p1, [x5, #1, mul vl]
    st1w    z10.s, p1, [x5, #2, mul vl]
    st1w    z11.s, p1, [x5, #3, mul vl]
    st1w    z12.s, p1, [x5, #4, mul vl]
    st1w    z13.s, p1, [x5, #5, mul vl]
    st1w    z14.s, p1, [x5, #6, mul vl]
    st1w    z15.s, p1, [x5, #7, mul vl]
    ldp     q8, q9, [sp]
    ldp     q10, q11, [sp, #32]
    ldp     q12, q13, [sp, #64]
    ldp     q14, q15, [sp, #96]
    add     sp, sp, #128
    ret
    .size fp4_i8_sdot_m1_asm, .-fp4_i8_sdot_m1_asm
    .section .note.GNU-stack,"",%progbits
