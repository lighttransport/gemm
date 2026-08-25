/* MXFP4 M1 x four N32 tiles, pipe-scheduled A64FX SVE kernel. */
    .arch armv8.2-a+sve
    .text
    .align 4
/* args: q[4] 0..31, s[4] 32..63, a 64, out 72, k_count 80, bs 84 */
    .global fp4_mx_m1_t4_asm
    .type fp4_mx_m1_t4_asm, %function
fp4_mx_m1_t4_asm:
    sub     sp, sp, #64
    stp     q8, q9, [sp]
    stp     q10, q11, [sp, #32]
    ldp     x1, x2, [x0]
    ldp     x3, x4, [x0, #16]
    ldp     x5, x6, [x0, #32]
    ldp     x7, x8, [x0, #48]
    ldr     x9, [x0, #64]
    ldr     x10, [x0, #72]
    ldr     w11, [x0, #80]
    ldr     w12, [x0, #84]
    ptrue   p0.h
    mov     x14, #16
    whilelo p1.h, xzr, x14
    adrp    x13, .Le2m1_t4
    add     x13, x13, :lo12:.Le2m1_t4
    ld1h    z0.h, p0/z, [x13]
    dup     z1.h, #15
    eor     z2.d, z2.d, z2.d
    eor     z3.d, z3.d, z3.d
    eor     z4.d, z4.d, z4.d
    eor     z5.d, z5.d, z5.d
    udiv    w11, w11, w12
.Lt4_block:
    ld1h    z6.h, p0/z, [x5]
    ld1h    z7.h, p0/z, [x6]
    ld1h    z16.h, p0/z, [x7]
    ld1h    z17.h, p0/z, [x8]
    add     x5, x5, #64
    add     x6, x6, #64
    add     x7, x7, #64
    add     x8, x8, #64
    mov     w13, w12
.Lt4_k:
    /* Two K steps in flight: the second decode covers ZIP1/TBL latency. */
    ld1b    z18.h, p1/z, [x1]
    ld1b    z19.h, p1/z, [x2]
    ld1b    z20.h, p1/z, [x3]
    ld1b    z21.h, p1/z, [x4]
    add     x1, x1, #16
    add     x2, x2, #16
    add     x3, x3, #16
    add     x4, x4, #16
    and     z22.d, z18.d, z1.d
    and     z23.d, z19.d, z1.d
    and     z24.d, z20.d, z1.d
    and     z25.d, z21.d, z1.d
    lsr     z18.h, z18.h, #4
    lsr     z19.h, z19.h, #4
    lsr     z20.h, z20.h, #4
    lsr     z21.h, z21.h, #4
    zip1    z22.h, z22.h, z18.h
    zip1    z23.h, z23.h, z19.h
    zip1    z24.h, z24.h, z20.h
    zip1    z25.h, z25.h, z21.h
    ld1b    z18.h, p1/z, [x1]
    ld1b    z19.h, p1/z, [x2]
    ld1b    z20.h, p1/z, [x3]
    ld1b    z21.h, p1/z, [x4]
    and     z8.d, z18.d, z1.d
    and     z9.d, z19.d, z1.d
    and     z10.d, z20.d, z1.d
    and     z11.d, z21.d, z1.d
    lsr     z18.h, z18.h, #4
    lsr     z19.h, z19.h, #4
    lsr     z20.h, z20.h, #4
    lsr     z21.h, z21.h, #4
    zip1    z8.h, z8.h, z18.h
    zip1    z9.h, z9.h, z19.h
    zip1    z10.h, z10.h, z20.h
    zip1    z11.h, z11.h, z21.h
    tbl     z22.h, {z0.h}, z22.h
    tbl     z23.h, {z0.h}, z23.h
    tbl     z24.h, {z0.h}, z24.h
    tbl     z25.h, {z0.h}, z25.h
    tbl     z8.h, {z0.h}, z8.h
    tbl     z9.h, {z0.h}, z9.h
    tbl     z10.h, {z0.h}, z10.h
    tbl     z11.h, {z0.h}, z11.h
    fmul    z22.h, p0/m, z22.h, z6.h
    fmul    z23.h, p0/m, z23.h, z7.h
    fmul    z24.h, p0/m, z24.h, z16.h
    fmul    z25.h, p0/m, z25.h, z17.h
    fmul    z8.h, p0/m, z8.h, z6.h
    fmul    z9.h, p0/m, z9.h, z7.h
    fmul    z10.h, p0/m, z10.h, z16.h
    fmul    z11.h, p0/m, z11.h, z17.h
    ld1rh   z26.h, p0/z, [x9]
    ld1rh   z27.h, p0/z, [x9, #2]
    add     x9, x9, #4
    fmla    z2.h, p0/m, z22.h, z26.h
    fmla    z3.h, p0/m, z23.h, z26.h
    fmla    z4.h, p0/m, z24.h, z26.h
    fmla    z5.h, p0/m, z25.h, z26.h
    fmla    z2.h, p0/m, z8.h, z27.h
    fmla    z3.h, p0/m, z9.h, z27.h
    fmla    z4.h, p0/m, z10.h, z27.h
    fmla    z5.h, p0/m, z11.h, z27.h
    add     x1, x1, #16
    add     x2, x2, #16
    add     x3, x3, #16
    add     x4, x4, #16
    subs    w13, w13, #2
    b.ne    .Lt4_k
    subs    w11, w11, #1
    b.ne    .Lt4_block
    st1h    z2.h, p0, [x10]
    st1h    z3.h, p0, [x10, #1, mul vl]
    st1h    z4.h, p0, [x10, #2, mul vl]
    st1h    z5.h, p0, [x10, #3, mul vl]
    ldp     q8, q9, [sp]
    ldp     q10, q11, [sp, #32]
    add     sp, sp, #64
    ret
    .size fp4_mx_m1_t4_asm, .-fp4_mx_m1_t4_asm

/* MXFP4 M1 x four N32 tiles, one byte per code (no ZIP1). Same args ABI. */
    .global fp4_mx_u8_m1_t4_asm
    .type fp4_mx_u8_m1_t4_asm, %function
fp4_mx_u8_m1_t4_asm:
    sub     sp, sp, #64
    stp     q8, q9, [sp]
    stp     q10, q11, [sp, #32]
    ldp     x1, x2, [x0]
    ldp     x3, x4, [x0, #16]
    ldp     x5, x6, [x0, #32]
    ldp     x7, x8, [x0, #48]
    ldr     x9, [x0, #64]
    ldr     x10, [x0, #72]
    ldr     w11, [x0, #80]
    ldr     w12, [x0, #84]
    ptrue   p0.h
    adrp    x13, .Le2m1_t4
    add     x13, x13, :lo12:.Le2m1_t4
    ld1h    z0.h, p0/z, [x13]
    eor     z2.d, z2.d, z2.d
    eor     z3.d, z3.d, z3.d
    eor     z4.d, z4.d, z4.d
    eor     z5.d, z5.d, z5.d
    udiv    w11, w11, w12
.Lu8_block:
    ld1h    z6.h, p0/z, [x5]
    ld1h    z7.h, p0/z, [x6]
    ld1h    z16.h, p0/z, [x7]
    ld1h    z17.h, p0/z, [x8]
    add     x5, x5, #64
    add     x6, x6, #64
    add     x7, x7, #64
    add     x8, x8, #64
    mov     w13, w12
.Lu8_k:
    ld1b    z18.h, p0/z, [x1]
    ld1b    z19.h, p0/z, [x2]
    ld1b    z20.h, p0/z, [x3]
    ld1b    z21.h, p0/z, [x4]
    add     x1, x1, #128
    add     x2, x2, #128
    add     x3, x3, #128
    add     x4, x4, #128
    ld1b    z8.h, p0/z, [x1]
    ld1b    z9.h, p0/z, [x2]
    ld1b    z10.h, p0/z, [x3]
    ld1b    z11.h, p0/z, [x4]
    tbl     z18.h, {z0.h}, z18.h
    tbl     z19.h, {z0.h}, z19.h
    tbl     z20.h, {z0.h}, z20.h
    tbl     z21.h, {z0.h}, z21.h
    tbl     z8.h, {z0.h}, z8.h
    tbl     z9.h, {z0.h}, z9.h
    tbl     z10.h, {z0.h}, z10.h
    tbl     z11.h, {z0.h}, z11.h
    fmul    z18.h, p0/m, z18.h, z6.h
    fmul    z19.h, p0/m, z19.h, z7.h
    fmul    z20.h, p0/m, z20.h, z16.h
    fmul    z21.h, p0/m, z21.h, z17.h
    fmul    z8.h, p0/m, z8.h, z6.h
    fmul    z9.h, p0/m, z9.h, z7.h
    fmul    z10.h, p0/m, z10.h, z16.h
    fmul    z11.h, p0/m, z11.h, z17.h
    ld1rh   z26.h, p0/z, [x9]
    ld1rh   z27.h, p0/z, [x9, #2]
    add     x9, x9, #4
    fmla    z2.h, p0/m, z18.h, z26.h
    fmla    z3.h, p0/m, z19.h, z26.h
    fmla    z4.h, p0/m, z20.h, z26.h
    fmla    z5.h, p0/m, z21.h, z26.h
    fmla    z2.h, p0/m, z8.h, z27.h
    fmla    z3.h, p0/m, z9.h, z27.h
    fmla    z4.h, p0/m, z10.h, z27.h
    fmla    z5.h, p0/m, z11.h, z27.h
    add     x1, x1, #128
    add     x2, x2, #128
    add     x3, x3, #128
    add     x4, x4, #128
    subs    w13, w13, #2
    b.ne    .Lu8_k
    subs    w11, w11, #1
    b.ne    .Lu8_block
    st1h    z2.h, p0, [x10]
    st1h    z3.h, p0, [x10, #1, mul vl]
    st1h    z4.h, p0, [x10, #2, mul vl]
    st1h    z5.h, p0, [x10, #3, mul vl]
    ldp     q8, q9, [sp]
    ldp     q10, q11, [sp, #32]
    add     sp, sp, #64
    ret
    .size fp4_mx_u8_m1_t4_asm, .-fp4_mx_u8_m1_t4_asm

    .section .rodata
    .align 6
.Le2m1_t4:
    .hword 0x0000,0x3800,0x3c00,0x3e00,0x4000,0x4200,0x4400,0x4600
    .hword 0x8000,0xb800,0xbc00,0xbe00,0xc000,0xc200,0xc400,0xc600
    .hword 0x0000,0x3800,0x3c00,0x3e00,0x4000,0x4200,0x4400,0x4600
    .hword 0x8000,0xb800,0xbc00,0xbe00,0xc000,0xc200,0xc400,0xc600
    .section .note.GNU-stack,"",%progbits
