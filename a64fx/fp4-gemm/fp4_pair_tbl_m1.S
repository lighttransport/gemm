    .arch armv8.2-a+sve
    .text
    .align 4
/* One N128 group. Packed byte = E2M1(k) | E2M1(k+1)<<4. The two
 * activation-dependent 16-entry INT16 tables are duplicated to fill a Z
 * register. INT16 accumulation is safe through K16. */
    .global fp4_pair_tbl_m1_asm
    .type fp4_pair_tbl_m1_asm, %function
fp4_pair_tbl_m1_asm:
    sub     sp, sp, #128
    stp     q8, q9, [sp]
    stp     q10, q11, [sp, #32]
    stp     q12, q13, [sp, #64]
    stp     q14, q15, [sp, #96]
    ldp     x1, x2, [x0]          /* packed codes, row scales */
    ldp     x3, x4, [x0, #16]    /* nibble tables, activation scales */
    ldr     x5, [x0, #32]        /* output */
    ldr     w6, [x0, #40]        /* N128 groups */
    ldr     w7, [x0, #44]        /* this group */
    ldr     w8, [x0, #52]        /* weight blocks */
    ldr     w9, [x0, #56]        /* K pairs/block */
    ldr     w10, [x0, #60]       /* activation group */
    ptrue   p0.h
    ptrue   p1.s
    mov     x11, #16
    whilelo p2.h, xzr, x11
    dup     z29.h, #15
    eor     z8.d, z8.d, z8.d
    eor     z9.d, z9.d, z9.d
    eor     z10.d, z10.d, z10.d
    eor     z11.d, z11.d, z11.d
    eor     z12.d, z12.d, z12.d
    eor     z13.d, z13.d, z13.d
    eor     z14.d, z14.d, z14.d
    eor     z15.d, z15.d, z15.d
    lsl     w11, w7, #9
    add     x5, x5, w11, uxtw
    mov     w12, wzr               /* block index */
.Lpt_block:
    /* q = base + ((block*ng + group)*pairs)*128 */
    madd    w11, w12, w6, w7
    mul     w11, w11, w9
    lsl     w11, w11, #7
    add     x13, x1, w11, uxtw
    madd    w11, w12, w6, w7
    lsl     w11, w11, #8
    add     x14, x2, w11, uxtw
    mul     w11, w12, w9
    lsl     w11, w11, #7          /* 2 K tables * 64 bytes per pair */
    add     x15, x3, w11, uxtw
    lsl     w11, w9, #1
    udiv    w11, w11, w10         /* activation groups/block */
    mul     w11, w11, w12
    lsl     w11, w11, #2
    add     x16, x4, w11, uxtw

    ld1h    z24.h, p0/z, [x14]
    ld1h    z26.h, p0/z, [x14, #1, mul vl]
    ld1h    z28.h, p0/z, [x14, #2, mul vl]
    ld1h    z30.h, p0/z, [x14, #3, mul vl]
    uunpklo z16.s, z24.h
    uunpkhi z17.s, z24.h
    uunpklo z18.s, z26.h
    uunpkhi z19.s, z26.h
    uunpklo z20.s, z28.h
    uunpkhi z21.s, z28.h
    uunpklo z22.s, z30.h
    uunpkhi z23.s, z30.h
    fcvt    z16.s, p1/m, z16.h
    fcvt    z17.s, p1/m, z17.h
    fcvt    z18.s, p1/m, z18.h
    fcvt    z19.s, p1/m, z19.h
    fcvt    z20.s, p1/m, z20.h
    fcvt    z21.s, p1/m, z21.h
    fcvt    z22.s, p1/m, z22.h
    fcvt    z23.s, p1/m, z23.h
    lsl     w11, w9, #1
    udiv    w11, w11, w10
.Lpt_agroup:
    eor     z0.d, z0.d, z0.d
    eor     z1.d, z1.d, z1.d
    eor     z2.d, z2.d, z2.d
    eor     z3.d, z3.d, z3.d
    eor     z4.d, z4.d, z4.d
    eor     z5.d, z5.d, z5.d
    eor     z6.d, z6.d, z6.d
    eor     z7.d, z7.d, z7.d
    lsr     w14, w10, #1
.Lpt_pair:
    ld1h    z30.h, p0/z, [x15]
    ld1h    z31.h, p0/z, [x15, #1, mul vl]
    add     x15, x15, #128
.macro PT_DECODE2 ACC0, ACC1
    ld1b    z24.h, p2/z, [x13]
    add     x13, x13, #16
    ld1b    z26.h, p2/z, [x13]
    add     x13, x13, #16
    and     z25.d, z24.d, z29.d
    and     z27.d, z26.d, z29.d
    lsr     z24.h, z24.h, #4
    lsr     z26.h, z26.h, #4
    tbl     z25.h, {z30.h}, z25.h
    tbl     z27.h, {z30.h}, z27.h
    tbl     z24.h, {z31.h}, z24.h
    tbl     z26.h, {z31.h}, z26.h
    add     z\ACC0\().h, z\ACC0\().h, z25.h
    add     z\ACC1\().h, z\ACC1\().h, z27.h
    add     z\ACC0\().h, z\ACC0\().h, z24.h
    add     z\ACC1\().h, z\ACC1\().h, z26.h
.endm
    PT_DECODE2 0, 1
    PT_DECODE2 2, 3
    PT_DECODE2 4, 5
    PT_DECODE2 6, 7
    subs    w14, w14, #1
    b.ne    .Lpt_pair
    ld1rw   z31.s, p1/z, [x16]
    add     x16, x16, #4
.macro PT_PROMOTE IACC, FACC, WS
    sunpklo z24.s, z\IACC\().h
    scvtf   z24.s, p1/m, z24.s
    fmul    z24.s, p1/m, z24.s, z31.s
    fmla    z\FACC\().s, p1/m, z24.s, z\WS\().s
.endm
    PT_PROMOTE 0, 8, 16
    PT_PROMOTE 1, 9, 17
    PT_PROMOTE 2, 10, 18
    PT_PROMOTE 3, 11, 19
    PT_PROMOTE 4, 12, 20
    PT_PROMOTE 5, 13, 21
    PT_PROMOTE 6, 14, 22
    PT_PROMOTE 7, 15, 23
    subs    w11, w11, #1
    b.ne    .Lpt_agroup
    add     w12, w12, #1
    cmp     w12, w8
    b.lt    .Lpt_block
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
    .size fp4_pair_tbl_m1_asm, .-fp4_pair_tbl_m1_asm
    .section .note.GNU-stack,"",%progbits
