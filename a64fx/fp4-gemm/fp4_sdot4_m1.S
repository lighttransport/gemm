    .arch armv8.2-a+sve
    .text
    .align 4
/* True four-bit M1 SDOT: each 64-byte load holds 32 rows x K4. */
    .global fp4_i8_sdot4_m1_asm
    .type fp4_i8_sdot4_m1_asm, %function
fp4_i8_sdot4_m1_asm:
    sub     sp, sp, #128
    stp     q8, q9, [sp]
    stp     q10, q11, [sp, #32]
    stp     q12, q13, [sp, #64]
    stp     q14, q15, [sp, #96]
    ldp     x1, x2, [x0]
    ldp     x3, x4, [x0, #16]
    ldr     x5, [x0, #32]
    ldr     w6, [x0, #40]
    ldr     w7, [x0, #44]
    ldr     w8, [x0, #48]
    ldr     w9, [x0, #52]
    ptrue   p0.b
    ptrue   p1.s
    adrp    x15, .Le2m1_i8
    add     x15, x15, :lo12:.Le2m1_i8
    ld1b    z30.b, p0/z, [x15]
    dup     z29.b, #15
    eor     z8.d, z8.d, z8.d
    eor     z9.d, z9.d, z9.d
    eor     z10.d, z10.d, z10.d
    eor     z11.d, z11.d, z11.d
    eor     z12.d, z12.d, z12.d
    eor     z13.d, z13.d, z13.d
    eor     z14.d, z14.d, z14.d
    eor     z15.d, z15.d, z15.d
    mov     w10, wzr
.Ls4_block:
    madd    w11, w10, w6, w7
    mul     w11, w11, w9
    lsl     w11, w11, #8
    add     x11, x1, w11, uxtw
    mul     w12, w10, w9
    lsl     w12, w12, #2
    add     x12, x2, w12, uxtw
    madd    w13, w10, w6, w7
    lsl     w13, w13, #9
    add     x13, x3, w13, uxtw
    mov     x15, x13
    mul     w14, w10, w9
    lsl     w14, w14, #2
    add     x14, x4, w14, uxtw
    mov     w13, w9
.Ls4_k4:
    eor     z0.d, z0.d, z0.d
    eor     z1.d, z1.d, z1.d
    eor     z2.d, z2.d, z2.d
    eor     z3.d, z3.d, z3.d
    eor     z4.d, z4.d, z4.d
    eor     z5.d, z5.d, z5.d
    eor     z6.d, z6.d, z6.d
    eor     z7.d, z7.d, z7.d
    ld1rw   z31.s, p1/z, [x12]
    add     x12, x12, #4
.macro S4_LOAD Z
    ld1b    z\Z\().b, p0/z, [x11]
    add     x11, x11, #64
.endm
    S4_LOAD 16
    S4_LOAD 17
    S4_LOAD 18
    S4_LOAD 19
    and     z20.d, z16.d, z29.d
    and     z21.d, z17.d, z29.d
    and     z22.d, z18.d, z29.d
    and     z23.d, z19.d, z29.d
    lsr     z16.b, p0/m, z16.b, #4
    lsr     z17.b, p0/m, z17.b, #4
    lsr     z18.b, p0/m, z18.b, #4
    lsr     z19.b, p0/m, z19.b, #4
    tbl     z20.b, {z30.b}, z20.b
    tbl     z21.b, {z30.b}, z21.b
    tbl     z22.b, {z30.b}, z22.b
    tbl     z23.b, {z30.b}, z23.b
    tbl     z16.b, {z30.b}, z16.b
    tbl     z17.b, {z30.b}, z17.b
    tbl     z18.b, {z30.b}, z18.b
    tbl     z19.b, {z30.b}, z19.b
    zip1    z24.b, z20.b, z16.b
    zip1    z25.b, z21.b, z17.b
    zip1    z26.b, z22.b, z18.b
    zip1    z27.b, z23.b, z19.b
    zip2    z20.b, z20.b, z16.b
    zip2    z21.b, z21.b, z17.b
    zip2    z22.b, z22.b, z18.b
    zip2    z23.b, z23.b, z19.b
    sdot    z0.s, z24.b, z31.b
    sdot    z1.s, z20.b, z31.b
    sdot    z2.s, z25.b, z31.b
    sdot    z3.s, z21.b, z31.b
    sdot    z4.s, z26.b, z31.b
    sdot    z5.s, z22.b, z31.b
    sdot    z6.s, z27.b, z31.b
    sdot    z7.s, z23.b, z31.b
    ld1rw   z28.s, p1/z, [x14]
    add     x14, x14, #4
    scvtf   z0.s, p1/m, z0.s
    scvtf   z1.s, p1/m, z1.s
    scvtf   z2.s, p1/m, z2.s
    scvtf   z3.s, p1/m, z3.s
    scvtf   z4.s, p1/m, z4.s
    scvtf   z5.s, p1/m, z5.s
    scvtf   z6.s, p1/m, z6.s
    scvtf   z7.s, p1/m, z7.s
    fmul    z0.s, p1/m, z0.s, z28.s
    fmul    z1.s, p1/m, z1.s, z28.s
    fmul    z2.s, p1/m, z2.s, z28.s
    fmul    z3.s, p1/m, z3.s, z28.s
    fmul    z4.s, p1/m, z4.s, z28.s
    fmul    z5.s, p1/m, z5.s, z28.s
    fmul    z6.s, p1/m, z6.s, z28.s
    fmul    z7.s, p1/m, z7.s, z28.s
    ld1w    z16.s, p1/z, [x15]
    ld1w    z17.s, p1/z, [x15, #1, mul vl]
    ld1w    z18.s, p1/z, [x15, #2, mul vl]
    ld1w    z19.s, p1/z, [x15, #3, mul vl]
    ld1w    z20.s, p1/z, [x15, #4, mul vl]
    ld1w    z21.s, p1/z, [x15, #5, mul vl]
    ld1w    z22.s, p1/z, [x15, #6, mul vl]
    ld1w    z23.s, p1/z, [x15, #7, mul vl]
    fmla    z8.s, p1/m, z0.s, z16.s
    fmla    z9.s, p1/m, z1.s, z17.s
    fmla    z10.s, p1/m, z2.s, z18.s
    fmla    z11.s, p1/m, z3.s, z19.s
    fmla    z12.s, p1/m, z4.s, z20.s
    fmla    z13.s, p1/m, z5.s, z21.s
    fmla    z14.s, p1/m, z6.s, z22.s
    fmla    z15.s, p1/m, z7.s, z23.s
    subs    w13, w13, #1
    b.ne    .Ls4_k4
    add     w10, w10, #1
    cmp     w10, w8
    b.lt    .Ls4_block
    lsl     w11, w7, #9
    add     x5, x5, w11, uxtw
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
    .size fp4_i8_sdot4_m1_asm, .-fp4_i8_sdot4_m1_asm
    .section .rodata
    .align 6
.Le2m1_i8:
    .byte 0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12
    .byte 0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12
    .byte 0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12
    .byte 0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12
    .section .note.GNU-stack,"",%progbits
