/* MXFP4 M1 x eight N32 tiles, block-major interleaved packed stream. */
    .arch armv8.2-a+sve
    .text
    .align 4
/* args: q, scales, activation-scaled tables, fp16 output, k_count */
    .global fp4_mx_m1_t8_asm
    .type fp4_mx_m1_t8_asm, %function
fp4_mx_m1_t8_asm:
    sub     sp, sp, #128
    stp     q8, q9, [sp]
    stp     q10, q11, [sp, #32]
    stp     q12, q13, [sp, #64]
    stp     q14, q15, [sp, #96]
    ldp     x1, x2, [x0]
    ldp     x3, x4, [x0, #16]
    ldr     w5, [x0, #32]
    ptrue   p0.h
    mov     x6, #16
    whilelo p1.h, xzr, x6
    dup     z1.h, #15
    eor     z2.d, z2.d, z2.d
    eor     z3.d, z3.d, z3.d
    eor     z4.d, z4.d, z4.d
    eor     z5.d, z5.d, z5.d
    eor     z6.d, z6.d, z6.d
    eor     z7.d, z7.d, z7.d
    eor     z8.d, z8.d, z8.d
    eor     z9.d, z9.d, z9.d
    lsr     w5, w5, #5
.Lt8_block:
    ld1h    z10.h, p0/z, [x2]
    ld1h    z11.h, p0/z, [x2, #1, mul vl]
    ld1h    z12.h, p0/z, [x2, #2, mul vl]
    ld1h    z13.h, p0/z, [x2, #3, mul vl]
    ld1h    z14.h, p0/z, [x2, #4, mul vl]
    ld1h    z15.h, p0/z, [x2, #5, mul vl]
    ld1h    z16.h, p0/z, [x2, #6, mul vl]
    ld1h    z17.h, p0/z, [x2, #7, mul vl]
    add     x2, x2, #512
    mov     w6, #32
.Lt8_k:
    ld1h    z26.h, p0/z, [x3]
    add     x3, x3, #64
.macro T8_BATCH A0,A1,A2,A3,S0,S1,S2,S3
    ld1b    z18.h, p1/z, [x1]
    add     x1, x1, #16
    ld1b    z19.h, p1/z, [x1]
    add     x1, x1, #16
    ld1b    z20.h, p1/z, [x1]
    add     x1, x1, #16
    ld1b    z21.h, p1/z, [x1]
    add     x1, x1, #16
    and     z22.d, z18.d, z1.d
    and     z23.d, z19.d, z1.d
    and     z24.d, z20.d, z1.d
    and     z25.d, z21.d, z1.d
    lsr     z18.h, p0/m, z18.h, #4
    lsr     z19.h, p0/m, z19.h, #4
    lsr     z20.h, p0/m, z20.h, #4
    lsr     z21.h, p0/m, z21.h, #4
    zip1    z22.h, z22.h, z18.h
    zip1    z23.h, z23.h, z19.h
    zip1    z24.h, z24.h, z20.h
    zip1    z25.h, z25.h, z21.h
    tbl     z22.h, {z26.h}, z22.h
    tbl     z23.h, {z26.h}, z23.h
    tbl     z24.h, {z26.h}, z24.h
    tbl     z25.h, {z26.h}, z25.h
    fmla    z\A0\().h, p0/m, z22.h, z\S0\().h
    fmla    z\A1\().h, p0/m, z23.h, z\S1\().h
    fmla    z\A2\().h, p0/m, z24.h, z\S2\().h
    fmla    z\A3\().h, p0/m, z25.h, z\S3\().h
.endm
    T8_BATCH 2,3,4,5,10,11,12,13
    T8_BATCH 6,7,8,9,14,15,16,17
    subs    w6, w6, #1
    b.ne    .Lt8_k
    subs    w5, w5, #1
    b.ne    .Lt8_block
    st1h    z2.h, p0, [x4]
    st1h    z3.h, p0, [x4, #1, mul vl]
    st1h    z4.h, p0, [x4, #2, mul vl]
    st1h    z5.h, p0, [x4, #3, mul vl]
    st1h    z6.h, p0, [x4, #4, mul vl]
    st1h    z7.h, p0, [x4, #5, mul vl]
    st1h    z8.h, p0, [x4, #6, mul vl]
    st1h    z9.h, p0, [x4, #7, mul vl]
    ldp     q8, q9, [sp]
    ldp     q10, q11, [sp, #32]
    ldp     q12, q13, [sp, #64]
    ldp     q14, q15, [sp, #96]
    add     sp, sp, #128
    ret
    .size fp4_mx_m1_t8_asm, .-fp4_mx_m1_t8_asm
    .section .note.GNU-stack,"",%progbits
