/* Four-way software-pipelined MXFP4 N32 panel decoder. */
    .arch armv8.2-a+sve
    .text
    .align 4
    .global fp4_dequant_mx_n32_asm
    .type fp4_dequant_mx_n32_asm, %function
/* void fn(fp16 *out, const u8 *codes, const fp16 *scales, int k_count); */
fp4_dequant_mx_n32_asm:
    ptrue   p0.h
    mov     x10, #16
    whilelo p1.h, xzr, x10
    adrp    x8, .Le2m1_sve
    add     x8, x8, :lo12:.Le2m1_sve
    ld1h    z0.h, p0/z, [x8]
    dup     z18.h, #15
    lsr     w3, w3, #5             // K32 blocks
.Lmx_block:
    ld1h    z1.h, p0/z, [x2]
    add     x2, x2, #64
    mov     w9, #8                 // eight groups of four K values
.Lmx_four:
    ld1b    z2.h, p1/z, [x1]
    add     x1, x1, #16
    ld1b    z3.h, p1/z, [x1]
    add     x1, x1, #16
    ld1b    z4.h, p1/z, [x1]
    add     x1, x1, #16
    ld1b    z5.h, p1/z, [x1]
    add     x1, x1, #16
    and     z6.d, z2.d, z18.d
    and     z7.d, z3.d, z18.d
    and     z16.d, z4.d, z18.d
    and     z17.d, z5.d, z18.d
    lsr     z2.h, z2.h, #4
    lsr     z3.h, z3.h, #4
    lsr     z4.h, z4.h, #4
    lsr     z5.h, z5.h, #4
    and     z2.d, z2.d, z18.d
    and     z3.d, z3.d, z18.d
    and     z4.d, z4.d, z18.d
    and     z5.d, z5.d, z18.d
    zip1    z6.h, z6.h, z2.h
    zip1    z7.h, z7.h, z3.h
    zip1    z16.h, z16.h, z4.h
    zip1    z17.h, z17.h, z5.h
    tbl     z6.h, {z0.h}, z6.h
    tbl     z7.h, {z0.h}, z7.h
    tbl     z16.h, {z0.h}, z16.h
    tbl     z17.h, {z0.h}, z17.h
    fmul    z6.h, p0/m, z6.h, z1.h
    fmul    z7.h, p0/m, z7.h, z1.h
    fmul    z16.h, p0/m, z16.h, z1.h
    fmul    z17.h, p0/m, z17.h, z1.h
    st1h    z6.h, p0, [x0]
    st1h    z7.h, p0, [x0, #1, mul vl]
    st1h    z16.h, p0, [x0, #2, mul vl]
    st1h    z17.h, p0, [x0, #3, mul vl]
    add     x0, x0, #256
    subs    w9, w9, #1
    b.ne    .Lmx_four
    subs    w3, w3, #1
    b.ne    .Lmx_block
    ret
    .size fp4_dequant_mx_n32_asm, .-fp4_dequant_mx_n32_asm
    .section .rodata
    .align 6
.Le2m1_sve:
    .hword 0x0000,0x3800,0x3c00,0x3e00,0x4000,0x4200,0x4400,0x4600
    .hword 0x8000,0xb800,0xbc00,0xbe00,0xc000,0xc200,0xc400,0xc600
    .hword 0x0000,0x3800,0x3c00,0x3e00,0x4000,0x4200,0x4400,0x4600
    .hword 0x8000,0xb800,0xbc00,0xbe00,0xc000,0xc200,0xc400,0xc600
    .section .note.GNU-stack,"",%progbits
