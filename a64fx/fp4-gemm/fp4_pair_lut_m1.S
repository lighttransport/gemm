    .arch armv8.2-a+sve
    .text
    .align 4
/* Packed FP4 M=1 kernel.  Each table entry is the exact INT32 dot product
 * of one packed E2M1 weight byte and two quantized activation values.
 * Eight independent gather chains cover 128 output rows. */
    .global fp4_pair_lut_m1_asm
    .type fp4_pair_lut_m1_asm, %function
fp4_pair_lut_m1_asm:
    sub     sp, sp, #208
    stp     x19, x20, [sp]
    stp     x21, x22, [sp, #16]
    stp     x23, x24, [sp, #32]
    stp     x25, x26, [sp, #48]
    stp     x27, x28, [sp, #64]
    stp     q8, q9, [sp, #80]
    stp     q10, q11, [sp, #112]
    stp     q12, q13, [sp, #144]
    stp     q14, q15, [sp, #176]
    ldp     x19, x20, [x0]
    ldp     x21, x22, [x0, #16]
    ldr     x23, [x0, #32]
    ldr     w24, [x0, #40]
    ldr     w25, [x0, #44]
    ldr     w26, [x0, #48]
    ldr     w27, [x0, #52]
    ldr     w28, [x0, #56]
    ldr     w5, [x0, #60]
    ptrue   p0.h
    ptrue   p1.s

    /* Start at group g0 in block zero. */
    mul     w6, w25, w28
    lsl     w6, w6, #7
    add     x6, x19, w6, uxtw
    lsl     w7, w25, #8
    add     x7, x20, w7, uxtw
    lsl     w10, w25, #9
    add     x10, x23, w10, uxtw
    mov     x8, x21
    mov     x9, x22
    mov     w11, wzr
.Lpair_block:
    mov     w12, w26
    lsl     w10, w25, #9
    add     x10, x23, w10, uxtw
.Lpair_group:
    cbnz    w11, .Lpair_load_output
    eor     z8.d, z8.d, z8.d
    eor     z9.d, z9.d, z9.d
    eor     z10.d, z10.d, z10.d
    eor     z11.d, z11.d, z11.d
    eor     z12.d, z12.d, z12.d
    eor     z13.d, z13.d, z13.d
    eor     z14.d, z14.d, z14.d
    eor     z15.d, z15.d, z15.d
    b       .Lpair_output_ready
.Lpair_load_output:
    ld1w    z8.s, p1/z, [x10]
    ld1w    z9.s, p1/z, [x10, #1, mul vl]
    ld1w    z10.s, p1/z, [x10, #2, mul vl]
    ld1w    z11.s, p1/z, [x10, #3, mul vl]
    ld1w    z12.s, p1/z, [x10, #4, mul vl]
    ld1w    z13.s, p1/z, [x10, #5, mul vl]
    ld1w    z14.s, p1/z, [x10, #6, mul vl]
    ld1w    z15.s, p1/z, [x10, #7, mul vl]
.Lpair_output_ready:
    /* Convert 128 FP16 row scales once per weight block/output group. */
    ld1h    z24.h, p0/z, [x7]
    ld1h    z26.h, p0/z, [x7, #1, mul vl]
    ld1h    z28.h, p0/z, [x7, #2, mul vl]
    ld1h    z30.h, p0/z, [x7, #3, mul vl]
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

    mov     x15, x8
    mov     x16, x9
    lsl     w13, w28, #1
    udiv    w13, w13, w5
.Lpair_agroup:
    eor     z0.d, z0.d, z0.d
    eor     z1.d, z1.d, z1.d
    eor     z2.d, z2.d, z2.d
    eor     z3.d, z3.d, z3.d
    eor     z4.d, z4.d, z4.d
    eor     z5.d, z5.d, z5.d
    eor     z6.d, z6.d, z6.d
    eor     z7.d, z7.d, z7.d
    lsr     w14, w5, #1
.Lpair_one:
    ld1b    z24.s, p1/z, [x6]
    add     x6, x6, #16
    ld1b    z25.s, p1/z, [x6]
    add     x6, x6, #16
    ld1b    z26.s, p1/z, [x6]
    add     x6, x6, #16
    ld1b    z27.s, p1/z, [x6]
    add     x6, x6, #16
    ld1b    z28.s, p1/z, [x6]
    add     x6, x6, #16
    ld1b    z29.s, p1/z, [x6]
    add     x6, x6, #16
    ld1b    z30.s, p1/z, [x6]
    add     x6, x6, #16
    ld1b    z31.s, p1/z, [x6]
    add     x6, x6, #16
    ld1sh   z24.s, p1/z, [x15, z24.s, uxtw #1]
    ld1sh   z25.s, p1/z, [x15, z25.s, uxtw #1]
    ld1sh   z26.s, p1/z, [x15, z26.s, uxtw #1]
    ld1sh   z27.s, p1/z, [x15, z27.s, uxtw #1]
    ld1sh   z28.s, p1/z, [x15, z28.s, uxtw #1]
    ld1sh   z29.s, p1/z, [x15, z29.s, uxtw #1]
    ld1sh   z30.s, p1/z, [x15, z30.s, uxtw #1]
    ld1sh   z31.s, p1/z, [x15, z31.s, uxtw #1]
    add     z0.s, z0.s, z24.s
    add     z1.s, z1.s, z25.s
    add     z2.s, z2.s, z26.s
    add     z3.s, z3.s, z27.s
    add     z4.s, z4.s, z28.s
    add     z5.s, z5.s, z29.s
    add     z6.s, z6.s, z30.s
    add     z7.s, z7.s, z31.s
    add     x15, x15, #512
    subs    w14, w14, #1
    b.ne    .Lpair_one
    ld1rw   z31.s, p1/z, [x16]
    add     x16, x16, #4
    scvtf   z0.s, p1/m, z0.s
    scvtf   z1.s, p1/m, z1.s
    scvtf   z2.s, p1/m, z2.s
    scvtf   z3.s, p1/m, z3.s
    scvtf   z4.s, p1/m, z4.s
    scvtf   z5.s, p1/m, z5.s
    scvtf   z6.s, p1/m, z6.s
    scvtf   z7.s, p1/m, z7.s
    fmul    z0.s, p1/m, z0.s, z31.s
    fmul    z1.s, p1/m, z1.s, z31.s
    fmul    z2.s, p1/m, z2.s, z31.s
    fmul    z3.s, p1/m, z3.s, z31.s
    fmul    z4.s, p1/m, z4.s, z31.s
    fmul    z5.s, p1/m, z5.s, z31.s
    fmul    z6.s, p1/m, z6.s, z31.s
    fmul    z7.s, p1/m, z7.s, z31.s
    fmla    z8.s, p1/m, z0.s, z16.s
    fmla    z9.s, p1/m, z1.s, z17.s
    fmla    z10.s, p1/m, z2.s, z18.s
    fmla    z11.s, p1/m, z3.s, z19.s
    fmla    z12.s, p1/m, z4.s, z20.s
    fmla    z13.s, p1/m, z5.s, z21.s
    fmla    z14.s, p1/m, z6.s, z22.s
    fmla    z15.s, p1/m, z7.s, z23.s
    subs    w13, w13, #1
    b.ne    .Lpair_agroup
    st1w    z8.s, p1, [x10]
    st1w    z9.s, p1, [x10, #1, mul vl]
    st1w    z10.s, p1, [x10, #2, mul vl]
    st1w    z11.s, p1, [x10, #3, mul vl]
    st1w    z12.s, p1, [x10, #4, mul vl]
    st1w    z13.s, p1, [x10, #5, mul vl]
    st1w    z14.s, p1, [x10, #6, mul vl]
    st1w    z15.s, p1, [x10, #7, mul vl]
    add     x7, x7, #256
    add     x10, x10, #512
    subs    w12, w12, #1
    b.ne    .Lpair_group

    /* Skip groups owned by other threads and advance block-local LUT/scales. */
    sub     w17, w24, w26
    mul     w17, w17, w28
    lsl     w17, w17, #7
    add     x6, x6, w17, uxtw
    sub     w17, w24, w26
    lsl     w17, w17, #8
    add     x7, x7, w17, uxtw
    lsl     w17, w28, #9
    add     x8, x8, w17, uxtw
    lsl     w17, w28, #1
    udiv    w17, w17, w5
    lsl     w17, w17, #2
    add     x9, x9, w17, uxtw
    add     w11, w11, #1
    cmp     w11, w27
    b.lt    .Lpair_block

    ldp     x19, x20, [sp]
    ldp     x21, x22, [sp, #16]
    ldp     x23, x24, [sp, #32]
    ldp     x25, x26, [sp, #48]
    ldp     x27, x28, [sp, #64]
    ldp     q8, q9, [sp, #80]
    ldp     q10, q11, [sp, #112]
    ldp     q12, q13, [sp, #144]
    ldp     q14, q15, [sp, #176]
    add     sp, sp, #208
    ret
    .size fp4_pair_lut_m1_asm, .-fp4_pair_lut_m1_asm
    .section .note.GNU-stack,"",%progbits
