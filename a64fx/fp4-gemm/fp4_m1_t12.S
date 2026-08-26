/* MXFP4 M1 x twelve N32 tiles, block-major interleaved packed stream. */
    .arch armv8.2-a+sve
    .text
    .align 4
/* args: q, scales, activation-scaled tables, fp16 output, k_count */
    .global fp4_mx_m1_t12_asm
    .type fp4_mx_m1_t12_asm, %function
fp4_mx_m1_t12_asm:
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
    eor     z10.d, z10.d, z10.d
    eor     z11.d, z11.d, z11.d
    eor     z12.d, z12.d, z12.d
    eor     z13.d, z13.d, z13.d
    lsr     w5, w5, #5
.Lt12_block:
    ld1h    z14.h, p0/z, [x2]
    ld1h    z15.h, p0/z, [x2, #1, mul vl]
    ld1h    z16.h, p0/z, [x2, #2, mul vl]
    ld1h    z17.h, p0/z, [x2, #3, mul vl]
    ld1h    z18.h, p0/z, [x2, #4, mul vl]
    ld1h    z19.h, p0/z, [x2, #5, mul vl]
    ld1h    z20.h, p0/z, [x2, #6, mul vl]
    ld1h    z21.h, p0/z, [x2, #7, mul vl]
    add     x2, x2, #512
    ld1h    z22.h, p0/z, [x2]
    ld1h    z23.h, p0/z, [x2, #1, mul vl]
    ld1h    z24.h, p0/z, [x2, #2, mul vl]
    ld1h    z25.h, p0/z, [x2, #3, mul vl]
    add     x2, x2, #256
    mov     w6, #32
.Lt12_k:
    ld1h    z30.h, p0/z, [x3]
    add     x3, x3, #64
.macro T12_PAIR A0,A1,S0,S1
    ld1b    z26.h, p1/z, [x1]
    add     x1, x1, #16
    ld1b    z27.h, p1/z, [x1]
    add     x1, x1, #16
    and     z28.d, z26.d, z1.d
    and     z29.d, z27.d, z1.d
    lsr     z26.h, p0/m, z26.h, #4
    lsr     z27.h, p0/m, z27.h, #4
    zip1    z28.h, z28.h, z26.h
    zip1    z29.h, z29.h, z27.h
    tbl     z28.h, {z30.h}, z28.h
    tbl     z29.h, {z30.h}, z29.h
    fmla    z\A0\().h, p0/m, z28.h, z\S0\().h
    fmla    z\A1\().h, p0/m, z29.h, z\S1\().h
.endm
    T12_PAIR 2,3,14,15
    T12_PAIR 4,5,16,17
    T12_PAIR 6,7,18,19
    T12_PAIR 8,9,20,21
    T12_PAIR 10,11,22,23
    T12_PAIR 12,13,24,25
    subs    w6, w6, #1
    b.ne    .Lt12_k
    subs    w5, w5, #1
    b.ne    .Lt12_block
    st1h    z2.h, p0, [x4]
    st1h    z3.h, p0, [x4, #1, mul vl]
    st1h    z4.h, p0, [x4, #2, mul vl]
    st1h    z5.h, p0, [x4, #3, mul vl]
    st1h    z6.h, p0, [x4, #4, mul vl]
    st1h    z7.h, p0, [x4, #5, mul vl]
    st1h    z8.h, p0, [x4, #6, mul vl]
    st1h    z9.h, p0, [x4, #7, mul vl]
    add     x4, x4, #512
    st1h    z10.h, p0, [x4]
    st1h    z11.h, p0, [x4, #1, mul vl]
    st1h    z12.h, p0, [x4, #2, mul vl]
    st1h    z13.h, p0, [x4, #3, mul vl]
    ldp     q8, q9, [sp]
    ldp     q10, q11, [sp, #32]
    ldp     q12, q13, [sp, #64]
    ldp     q14, q15, [sp, #96]
    add     sp, sp, #128
    ret
    .size fp4_mx_m1_t12_asm, .-fp4_mx_m1_t12_asm
    .section .note.GNU-stack,"",%progbits
