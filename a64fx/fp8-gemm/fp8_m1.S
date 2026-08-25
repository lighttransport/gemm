    .arch armv8.2-a+sve
    .text
    .align 4
/* args: q 0, a 8, scales 16, out 24, k 32 */
    .global fp8_e4m3fn_m1_asm
    .type fp8_e4m3fn_m1_asm, %function
fp8_e4m3fn_m1_asm:
    sub     sp, sp, #64
    stp     q8, q9, [sp]
    stp     q10, q11, [sp, #32]
    ldp     x1, x2, [x0]
    ldp     x3, x4, [x0, #16]
    ldr     w5, [x0, #32]
    ptrue   p0.s
    eor     z0.d, z0.d, z0.d
    eor     z1.d, z1.d, z1.d
    eor     z2.d, z2.d, z2.d
    eor     z3.d, z3.d, z3.d
    eor     z4.d, z4.d, z4.d
    eor     z5.d, z5.d, z5.d
    eor     z6.d, z6.d, z6.d
    eor     z7.d, z7.d, z7.d
    mov     w6, #0x3c000000
    dup     z8.s, w6
    mov     w6, #0x80
    dup     z9.s, w6
    mov     z10.s, #15
    mov     x7, #16
    mov     x8, #32
    mov     x9, #48
    mov     x10, #64
    mov     x11, #80
    mov     x12, #96
    mov     x13, #112
    lsr     w5, w5, #7
.Lfp8_block:
    ld1rw   z11.s, p0/z, [x3]
    add     x3, x3, #4
    mov     w6, #128
.Lfp8_k:
    ld1b    z16.s, p0/z, [x1]
    ld1b    z17.s, p0/z, [x1, x7]
    ld1b    z18.s, p0/z, [x1, x8]
    ld1b    z19.s, p0/z, [x1, x9]
    ld1b    z20.s, p0/z, [x1, x10]
    ld1b    z21.s, p0/z, [x1, x11]
    ld1b    z22.s, p0/z, [x1, x12]
    ld1b    z23.s, p0/z, [x1, x13]
    and     z24.d, z16.d, z9.d
    and     z25.d, z17.d, z9.d
    and     z26.d, z18.d, z9.d
    and     z27.d, z19.d, z9.d
    and     z28.d, z20.d, z9.d
    and     z29.d, z21.d, z9.d
    and     z30.d, z22.d, z9.d
    and     z31.d, z23.d, z9.d
    lsl     z16.s, z16.s, #20
    lsl     z17.s, z17.s, #20
    lsl     z18.s, z18.s, #20
    lsl     z19.s, z19.s, #20
    lsl     z20.s, z20.s, #20
    lsl     z21.s, z21.s, #20
    lsl     z22.s, z22.s, #20
    lsl     z23.s, z23.s, #20
    lsl     z24.s, z24.s, #20
    lsl     z25.s, z25.s, #20
    lsl     z26.s, z26.s, #20
    lsl     z27.s, z27.s, #20
    lsl     z28.s, z28.s, #20
    lsl     z29.s, z29.s, #20
    lsl     z30.s, z30.s, #20
    lsl     z31.s, z31.s, #20
    add     z16.s, z16.s, z8.s
    add     z17.s, z17.s, z8.s
    add     z18.s, z18.s, z8.s
    add     z19.s, z19.s, z8.s
    add     z20.s, z20.s, z8.s
    add     z21.s, z21.s, z8.s
    add     z22.s, z22.s, z8.s
    add     z23.s, z23.s, z8.s
    mla     z16.s, p0/m, z24.s, z10.s
    mla     z17.s, p0/m, z25.s, z10.s
    mla     z18.s, p0/m, z26.s, z10.s
    mla     z19.s, p0/m, z27.s, z10.s
    mla     z20.s, p0/m, z28.s, z10.s
    mla     z21.s, p0/m, z29.s, z10.s
    mla     z22.s, p0/m, z30.s, z10.s
    mla     z23.s, p0/m, z31.s, z10.s
    ld1rw   z31.s, p0/z, [x2]
    fmul    z31.s, p0/m, z31.s, z11.s
    fmla    z0.s, p0/m, z16.s, z31.s
    fmla    z1.s, p0/m, z17.s, z31.s
    fmla    z2.s, p0/m, z18.s, z31.s
    fmla    z3.s, p0/m, z19.s, z31.s
    fmla    z4.s, p0/m, z20.s, z31.s
    fmla    z5.s, p0/m, z21.s, z31.s
    fmla    z6.s, p0/m, z22.s, z31.s
    fmla    z7.s, p0/m, z23.s, z31.s
    add     x1, x1, #128
    add     x2, x2, #4
    subs    w6, w6, #1
    b.ne    .Lfp8_k
    subs    w5, w5, #1
    b.ne    .Lfp8_block
    st1w    z0.s, p0, [x4]
    st1w    z1.s, p0, [x4, #1, mul vl]
    st1w    z2.s, p0, [x4, #2, mul vl]
    st1w    z3.s, p0, [x4, #3, mul vl]
    st1w    z4.s, p0, [x4, #4, mul vl]
    st1w    z5.s, p0, [x4, #5, mul vl]
    st1w    z6.s, p0, [x4, #6, mul vl]
    st1w    z7.s, p0, [x4, #7, mul vl]
    ldp     q8, q9, [sp]
    ldp     q10, q11, [sp, #32]
    add     sp, sp, #64
    ret
    .size fp8_e4m3fn_m1_asm, .-fp8_e4m3fn_m1_asm

/* Signed INT8 re-encoding, FP32 FMA/accumulation. Two K rows per 256B line. */
    .global fp8_i8_m1_asm
    .type fp8_i8_m1_asm, %function
fp8_i8_m1_asm:
    sub     sp, sp, #64
    stp     q12, q13, [sp]
    stp     q14, q15, [sp, #32]
    ldp     x1, x2, [x0]
    ldp     x3, x4, [x0, #16]
    ldr     w5, [x0, #32]
    ldr     w15, [x0, #36]
    ptrue   p0.s
    eor     z0.d, z0.d, z0.d
    eor     z1.d, z1.d, z1.d
    eor     z2.d, z2.d, z2.d
    eor     z3.d, z3.d, z3.d
    eor     z4.d, z4.d, z4.d
    eor     z5.d, z5.d, z5.d
    eor     z6.d, z6.d, z6.d
    eor     z7.d, z7.d, z7.d
    mov     x7, #16
    mov     x8, #32
    mov     x9, #48
    mov     x10, #64
    mov     x11, #80
    mov     x12, #96
    mov     x13, #112
    udiv    w5, w5, w15
    lsr     w15, w15, #1
.Li8_block:
    ld1rw   z13.s, p0/z, [x3]
    add     x3, x3, #4
    mov     w6, w15
.Li8_k2:
    add     x14, x1, #128
    ld1sb   z16.s, p0/z, [x1]
    ld1sb   z17.s, p0/z, [x1, x7]
    ld1sb   z18.s, p0/z, [x1, x8]
    ld1sb   z19.s, p0/z, [x1, x9]
    ld1sb   z20.s, p0/z, [x1, x10]
    ld1sb   z21.s, p0/z, [x1, x11]
    ld1sb   z22.s, p0/z, [x1, x12]
    ld1sb   z23.s, p0/z, [x1, x13]
    ld1sb   z24.s, p0/z, [x14]
    ld1sb   z25.s, p0/z, [x14, x7]
    ld1sb   z26.s, p0/z, [x14, x8]
    ld1sb   z27.s, p0/z, [x14, x9]
    ld1sb   z28.s, p0/z, [x14, x10]
    ld1sb   z29.s, p0/z, [x14, x11]
    ld1sb   z30.s, p0/z, [x14, x12]
    ld1sb   z31.s, p0/z, [x14, x13]
    scvtf   z16.s, p0/m, z16.s
    scvtf   z17.s, p0/m, z17.s
    scvtf   z18.s, p0/m, z18.s
    scvtf   z19.s, p0/m, z19.s
    scvtf   z20.s, p0/m, z20.s
    scvtf   z21.s, p0/m, z21.s
    scvtf   z22.s, p0/m, z22.s
    scvtf   z23.s, p0/m, z23.s
    scvtf   z24.s, p0/m, z24.s
    scvtf   z25.s, p0/m, z25.s
    scvtf   z26.s, p0/m, z26.s
    scvtf   z27.s, p0/m, z27.s
    scvtf   z28.s, p0/m, z28.s
    scvtf   z29.s, p0/m, z29.s
    scvtf   z30.s, p0/m, z30.s
    scvtf   z31.s, p0/m, z31.s
    ld1rw   z14.s, p0/z, [x2]
    ld1rw   z15.s, p0/z, [x2, #4]
    fmul    z14.s, p0/m, z14.s, z13.s
    fmul    z15.s, p0/m, z15.s, z13.s
    fmla    z0.s, p0/m, z16.s, z14.s
    fmla    z1.s, p0/m, z17.s, z14.s
    fmla    z2.s, p0/m, z18.s, z14.s
    fmla    z3.s, p0/m, z19.s, z14.s
    fmla    z4.s, p0/m, z20.s, z14.s
    fmla    z5.s, p0/m, z21.s, z14.s
    fmla    z6.s, p0/m, z22.s, z14.s
    fmla    z7.s, p0/m, z23.s, z14.s
    fmla    z0.s, p0/m, z24.s, z15.s
    fmla    z1.s, p0/m, z25.s, z15.s
    fmla    z2.s, p0/m, z26.s, z15.s
    fmla    z3.s, p0/m, z27.s, z15.s
    fmla    z4.s, p0/m, z28.s, z15.s
    fmla    z5.s, p0/m, z29.s, z15.s
    fmla    z6.s, p0/m, z30.s, z15.s
    fmla    z7.s, p0/m, z31.s, z15.s
    add     x1, x1, #256
    add     x2, x2, #8
    subs    w6, w6, #1
    b.ne    .Li8_k2
    subs    w5, w5, #1
    b.ne    .Li8_block
    st1w    z0.s, p0, [x4]
    st1w    z1.s, p0, [x4, #1, mul vl]
    st1w    z2.s, p0, [x4, #2, mul vl]
    st1w    z3.s, p0, [x4, #3, mul vl]
    st1w    z4.s, p0, [x4, #4, mul vl]
    st1w    z5.s, p0, [x4, #5, mul vl]
    st1w    z6.s, p0, [x4, #6, mul vl]
    st1w    z7.s, p0, [x4, #7, mul vl]
    ldp     q12, q13, [sp]
    ldp     q14, q15, [sp, #32]
    add     sp, sp, #64
    ret
    .size fp8_i8_m1_asm, .-fp8_i8_m1_asm

/* Four scales per K group, one for each 32-output sub-tile. */
    .global fp8_i8x4_m1_asm
    .type fp8_i8x4_m1_asm, %function
fp8_i8x4_m1_asm:
    ldp     x1, x2, [x0]
    ldp     x3, x4, [x0, #16]
    ldr     w5, [x0, #32]
    ldr     w15, [x0, #36]
    ptrue   p0.s
    eor     z0.d, z0.d, z0.d
    eor     z1.d, z1.d, z1.d
    eor     z2.d, z2.d, z2.d
    eor     z3.d, z3.d, z3.d
    eor     z4.d, z4.d, z4.d
    eor     z5.d, z5.d, z5.d
    eor     z6.d, z6.d, z6.d
    eor     z7.d, z7.d, z7.d
    mov     x7, #16
    mov     x8, #32
    mov     x9, #48
    mov     x10, #64
    mov     x11, #80
    mov     x12, #96
    mov     x13, #112
    udiv    w5, w5, w15
.Li8x4_group:
    mov     w6, w15
.Li8x4_k:
    ld1sb   z16.s, p0/z, [x1]
    ld1sb   z17.s, p0/z, [x1, x7]
    ld1sb   z18.s, p0/z, [x1, x8]
    ld1sb   z19.s, p0/z, [x1, x9]
    ld1sb   z20.s, p0/z, [x1, x10]
    ld1sb   z21.s, p0/z, [x1, x11]
    ld1sb   z22.s, p0/z, [x1, x12]
    ld1sb   z23.s, p0/z, [x1, x13]
    scvtf   z16.s, p0/m, z16.s
    scvtf   z17.s, p0/m, z17.s
    scvtf   z18.s, p0/m, z18.s
    scvtf   z19.s, p0/m, z19.s
    scvtf   z20.s, p0/m, z20.s
    scvtf   z21.s, p0/m, z21.s
    scvtf   z22.s, p0/m, z22.s
    scvtf   z23.s, p0/m, z23.s
    ld1rw   z24.s, p0/z, [x3]
    ld1rw   z25.s, p0/z, [x3, #4]
    ld1rw   z26.s, p0/z, [x3, #8]
    ld1rw   z27.s, p0/z, [x3, #12]
    ld1rw   z28.s, p0/z, [x2]
    fmul    z24.s, p0/m, z24.s, z28.s
    fmul    z25.s, p0/m, z25.s, z28.s
    fmul    z26.s, p0/m, z26.s, z28.s
    fmul    z27.s, p0/m, z27.s, z28.s
    fmla    z0.s, p0/m, z16.s, z24.s
    fmla    z1.s, p0/m, z17.s, z24.s
    fmla    z2.s, p0/m, z18.s, z25.s
    fmla    z3.s, p0/m, z19.s, z25.s
    fmla    z4.s, p0/m, z20.s, z26.s
    fmla    z5.s, p0/m, z21.s, z26.s
    fmla    z6.s, p0/m, z22.s, z27.s
    fmla    z7.s, p0/m, z23.s, z27.s
    add     x1, x1, #128
    add     x2, x2, #4
    subs    w6, w6, #1
    b.ne    .Li8x4_k
    add     x3, x3, #16
    subs    w5, w5, #1
    b.ne    .Li8x4_group
    st1w    z0.s, p0, [x4]
    st1w    z1.s, p0, [x4, #1, mul vl]
    st1w    z2.s, p0, [x4, #2, mul vl]
    st1w    z3.s, p0, [x4, #3, mul vl]
    st1w    z4.s, p0, [x4, #4, mul vl]
    st1w    z5.s, p0, [x4, #5, mul vl]
    st1w    z6.s, p0, [x4, #6, mul vl]
    st1w    z7.s, p0, [x4, #7, mul vl]
    ret
    .size fp8_i8x4_m1_asm, .-fp8_i8x4_m1_asm

/* Exact E4M3FN via a 1 KiB L1-resident FP32 LUT. args adds LUT at 40. */
    .global fp8_e4m3fn_lut_m1_asm
    .type fp8_e4m3fn_lut_m1_asm, %function
fp8_e4m3fn_lut_m1_asm:
    ldp     x1, x2, [x0]
    ldp     x3, x4, [x0, #16]
    ldr     w5, [x0, #32]
    ldr     x14, [x0, #48]
    ptrue   p0.s
    eor     z0.d, z0.d, z0.d
    eor     z1.d, z1.d, z1.d
    eor     z2.d, z2.d, z2.d
    eor     z3.d, z3.d, z3.d
    eor     z4.d, z4.d, z4.d
    eor     z5.d, z5.d, z5.d
    eor     z6.d, z6.d, z6.d
    eor     z7.d, z7.d, z7.d
    mov     x7, #16
    mov     x8, #32
    mov     x9, #48
    mov     x10, #64
    mov     x11, #80
    mov     x12, #96
    mov     x13, #112
    lsr     w5, w5, #7
.Lfp8lut_block:
    ld1rw   z24.s, p0/z, [x3]
    add     x3, x3, #4
    mov     w6, #128
.Lfp8lut_k:
    ld1b    z16.s, p0/z, [x1]
    ld1b    z17.s, p0/z, [x1, x7]
    ld1b    z18.s, p0/z, [x1, x8]
    ld1b    z19.s, p0/z, [x1, x9]
    ld1b    z20.s, p0/z, [x1, x10]
    ld1b    z21.s, p0/z, [x1, x11]
    ld1b    z22.s, p0/z, [x1, x12]
    ld1b    z23.s, p0/z, [x1, x13]
    ld1w    z16.s, p0/z, [x14, z16.s, uxtw #2]
    ld1w    z17.s, p0/z, [x14, z17.s, uxtw #2]
    ld1w    z18.s, p0/z, [x14, z18.s, uxtw #2]
    ld1w    z19.s, p0/z, [x14, z19.s, uxtw #2]
    ld1w    z20.s, p0/z, [x14, z20.s, uxtw #2]
    ld1w    z21.s, p0/z, [x14, z21.s, uxtw #2]
    ld1w    z22.s, p0/z, [x14, z22.s, uxtw #2]
    ld1w    z23.s, p0/z, [x14, z23.s, uxtw #2]
    ld1rw   z31.s, p0/z, [x2]
    fmul    z31.s, p0/m, z31.s, z24.s
    fmla    z0.s, p0/m, z16.s, z31.s
    fmla    z1.s, p0/m, z17.s, z31.s
    fmla    z2.s, p0/m, z18.s, z31.s
    fmla    z3.s, p0/m, z19.s, z31.s
    fmla    z4.s, p0/m, z20.s, z31.s
    fmla    z5.s, p0/m, z21.s, z31.s
    fmla    z6.s, p0/m, z22.s, z31.s
    fmla    z7.s, p0/m, z23.s, z31.s
    add     x1, x1, #128
    add     x2, x2, #4
    subs    w6, w6, #1
    b.ne    .Lfp8lut_k
    subs    w5, w5, #1
    b.ne    .Lfp8lut_block
    st1w    z0.s, p0, [x4]
    st1w    z1.s, p0, [x4, #1, mul vl]
    st1w    z2.s, p0, [x4, #2, mul vl]
    st1w    z3.s, p0, [x4, #3, mul vl]
    st1w    z4.s, p0, [x4, #4, mul vl]
    st1w    z5.s, p0, [x4, #5, mul vl]
    st1w    z6.s, p0, [x4, #6, mul vl]
    st1w    z7.s, p0, [x4, #7, mul vl]
    ret
    .size fp8_e4m3fn_lut_m1_asm, .-fp8_e4m3fn_lut_m1_asm
    .section .note.GNU-stack,"",%progbits
