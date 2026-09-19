    .arch armv8.2-a+sve
    .text
    .align 6

/*
 * M=1, N-panel=16 MXFP4 kernel.
 * x0: SDOT-native packed codes [groups][8][32]
 * x1: E8M0 scales [groups][16]
 * x2: quantized activation [groups][32]
 * x3: activation scales [groups]
 * x4: group count
 * x5: output [16]
 */
    .global swfp4_mxfp4_sdot_panel16
    .type swfp4_mxfp4_sdot_panel16, %function
swfp4_mxfp4_sdot_panel16:
    ptrue   p0.s
    ptrue   p2.b
    mov     x6, #32
    whilelo p1.b, xzr, x6
    adrp    x6, .Lfp4_i8x2_table
    add     x6, x6, :lo12:.Lfp4_i8x2_table
    ld1b    {z31.b}, p2/z, [x6]
    mov     z0.s, #0
    cbz     x4, 3f
1:
    mov     z18.s, #0
    mov     z19.s, #0
    mov     z20.s, #0
    mov     z21.s, #0
    .macro SDOT_Q4 a0, a1, a2, a3
    ld1b    {z2.b}, p1/z, [x0]
    add     x0, x0, #32
    ld1b    {z3.b}, p1/z, [x0]
    add     x0, x0, #32
    ld1b    {z4.b}, p1/z, [x0]
    add     x0, x0, #32
    ld1b    {z5.b}, p1/z, [x0]
    add     x0, x0, #32
    ld1rw   {z10.s}, p0/z, [x2]
    add     x2, x2, #4
    ld1rw   {z11.s}, p0/z, [x2]
    add     x2, x2, #4
    ld1rw   {z12.s}, p0/z, [x2]
    add     x2, x2, #4
    ld1rw   {z13.s}, p0/z, [x2]
    add     x2, x2, #4
    lsr     z6.b, z2.b, #4
    lsr     z7.b, z3.b, #4
    lsr     z8.b, z4.b, #4
    lsr     z9.b, z5.b, #4
    and     z2.b, z2.b, #0x0f
    and     z3.b, z3.b, #0x0f
    and     z4.b, z4.b, #0x0f
    and     z5.b, z5.b, #0x0f
    zip1    z14.b, z2.b, z6.b
    zip1    z15.b, z3.b, z7.b
    zip1    z16.b, z4.b, z8.b
    zip1    z17.b, z5.b, z9.b
    tbl     z14.b, {z31.b}, z14.b
    tbl     z15.b, {z31.b}, z15.b
    tbl     z16.b, {z31.b}, z16.b
    tbl     z17.b, {z31.b}, z17.b
    sdot    \a0\().s, z14.b, z10.b
    sdot    \a1\().s, z15.b, z11.b
    sdot    \a2\().s, z16.b, z12.b
    sdot    \a3\().s, z17.b, z13.b
    .endm
    SDOT_Q4 z18, z19, z20, z21
    SDOT_Q4 z18, z19, z20, z21
    add     z18.s, z18.s, z19.s
    add     z20.s, z20.s, z21.s
    add     z1.s, z18.s, z20.s
    ld1b    {z2.s}, p0/z, [x1]
    lsl     z2.s, z2.s, #23
    scvtf   z1.s, p0/m, z1.s
    ld1rw   {z3.s}, p0/z, [x3]
    fmul    z2.s, p0/m, z2.s, z3.s
    fmla    z0.s, p0/m, z1.s, z2.s
    add     x1, x1, #16
    add     x3, x3, #4
    subs    x4, x4, #1
    nop
    nop
    b.ne    1b
3:
    st1w    {z0.s}, p0, [x5]
    ret
    .size swfp4_mxfp4_sdot_panel16, .-swfp4_mxfp4_sdot_panel16

    .section .rodata
    .align 6
.Lfp4_i8x2_table:
    .byte 0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12
    .zero 48
