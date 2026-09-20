/* Biased-byte E4M3FN candidate; ordinary K-major layout. */
.text
.p2align 6
.global w8_e4m3_bias_f16
.type w8_e4m3_bias_f16, %function
w8_e4m3_bias_f16:
    stp d8, d9, [sp, #-16]!
    stp d10, d11, [sp, #-16]!
    stp d12, d13, [sp, #-16]!
    stp d14, d15, [sp, #-16]!
    ptrue p0.h
    eor z0.d, z0.d, z0.d
    eor z1.d, z1.d, z1.d
    eor z2.d, z2.d, z2.d
    eor z3.d, z3.d, z3.d
    eor z4.d, z4.d, z4.d
    eor z5.d, z5.d, z5.d
    eor z6.d, z6.d, z6.d
    eor z7.d, z7.d, z7.d
    mov z29.h, #127
    mov z30.h, #8064
    adr x9, .Le4_bias
    ld1h {z31.h}, p0/z, [x9]
    ld1sb {z8.h}, p0/z, [x0, #0, mul vl]
    ld1sb {z9.h}, p0/z, [x0, #1, mul vl]
    ld1sb {z10.h}, p0/z, [x0, #2, mul vl]
    ld1sb {z11.h}, p0/z, [x0, #3, mul vl]
    ld1sb {z12.h}, p0/z, [x0, #4, mul vl]
    ld1sb {z13.h}, p0/z, [x0, #5, mul vl]
    ld1sb {z14.h}, p0/z, [x0, #6, mul vl]
    ld1sb {z15.h}, p0/z, [x0, #7, mul vl]
    ld1rh {z24.h}, p0/z, [x1]
    and z26.d, z8.d, z29.d
    tbl z26.h, {z31.h}, z26.h
    lsl z8.h, z8.h, #7
    and z8.h, z8.h, #0xbfff
    add z8.h, z8.h, z30.h
    add z8.h, z8.h, z26.h
    and z26.d, z9.d, z29.d
    tbl z26.h, {z31.h}, z26.h
    lsl z9.h, z9.h, #7
    and z9.h, z9.h, #0xbfff
    add z9.h, z9.h, z30.h
    add z9.h, z9.h, z26.h
    and z26.d, z10.d, z29.d
    tbl z26.h, {z31.h}, z26.h
    lsl z10.h, z10.h, #7
    and z10.h, z10.h, #0xbfff
    add z10.h, z10.h, z30.h
    add z10.h, z10.h, z26.h
    and z26.d, z11.d, z29.d
    tbl z26.h, {z31.h}, z26.h
    lsl z11.h, z11.h, #7
    and z11.h, z11.h, #0xbfff
    add z11.h, z11.h, z30.h
    add z11.h, z11.h, z26.h
    and z26.d, z12.d, z29.d
    tbl z26.h, {z31.h}, z26.h
    lsl z12.h, z12.h, #7
    and z12.h, z12.h, #0xbfff
    add z12.h, z12.h, z30.h
    add z12.h, z12.h, z26.h
    and z26.d, z13.d, z29.d
    tbl z26.h, {z31.h}, z26.h
    lsl z13.h, z13.h, #7
    and z13.h, z13.h, #0xbfff
    add z13.h, z13.h, z30.h
    add z13.h, z13.h, z26.h
    and z26.d, z14.d, z29.d
    tbl z26.h, {z31.h}, z26.h
    lsl z14.h, z14.h, #7
    and z14.h, z14.h, #0xbfff
    add z14.h, z14.h, z30.h
    add z14.h, z14.h, z26.h
    and z26.d, z15.d, z29.d
    tbl z26.h, {z31.h}, z26.h
    lsl z15.h, z15.h, #7
    and z15.h, z15.h, #0xbfff
    add z15.h, z15.h, z30.h
    add z15.h, z15.h, z26.h
    mov x10, #63
.p2align 6
1:
    add x0, x0, #256
    add x1, x1, #2
    ld1sb {z16.h}, p0/z, [x0, #0, mul vl]
    ld1sb {z17.h}, p0/z, [x0, #1, mul vl]
    ld1sb {z18.h}, p0/z, [x0, #2, mul vl]
    ld1sb {z19.h}, p0/z, [x0, #3, mul vl]
    ld1sb {z20.h}, p0/z, [x0, #4, mul vl]
    ld1sb {z21.h}, p0/z, [x0, #5, mul vl]
    ld1sb {z22.h}, p0/z, [x0, #6, mul vl]
    ld1sb {z23.h}, p0/z, [x0, #7, mul vl]
    and z25.d, z16.d, z29.d
    and z26.d, z17.d, z29.d
    and z27.d, z18.d, z29.d
    and z28.d, z19.d, z29.d
    tbl z25.h, {z31.h}, z25.h
    tbl z26.h, {z31.h}, z26.h
    tbl z27.h, {z31.h}, z27.h
    tbl z28.h, {z31.h}, z28.h
    lsl z16.h, z16.h, #7
    and z16.h, z16.h, #0xbfff
    lsl z17.h, z17.h, #7
    and z17.h, z17.h, #0xbfff
    lsl z18.h, z18.h, #7
    and z18.h, z18.h, #0xbfff
    lsl z19.h, z19.h, #7
    and z19.h, z19.h, #0xbfff
    fmla z0.h, p0/m, z8.h, z24.h
    fmla z1.h, p0/m, z9.h, z24.h
    fmla z2.h, p0/m, z10.h, z24.h
    fmla z3.h, p0/m, z11.h, z24.h
    add z16.h, z16.h, z30.h
    add z16.h, z16.h, z25.h
    add z17.h, z17.h, z30.h
    add z17.h, z17.h, z26.h
    add z18.h, z18.h, z30.h
    add z18.h, z18.h, z27.h
    add z19.h, z19.h, z30.h
    add z19.h, z19.h, z28.h
    and z25.d, z20.d, z29.d
    and z26.d, z21.d, z29.d
    and z27.d, z22.d, z29.d
    and z28.d, z23.d, z29.d
    tbl z25.h, {z31.h}, z25.h
    tbl z26.h, {z31.h}, z26.h
    tbl z27.h, {z31.h}, z27.h
    tbl z28.h, {z31.h}, z28.h
    lsl z20.h, z20.h, #7
    and z20.h, z20.h, #0xbfff
    lsl z21.h, z21.h, #7
    and z21.h, z21.h, #0xbfff
    lsl z22.h, z22.h, #7
    and z22.h, z22.h, #0xbfff
    lsl z23.h, z23.h, #7
    and z23.h, z23.h, #0xbfff
    fmla z4.h, p0/m, z12.h, z24.h
    fmla z5.h, p0/m, z13.h, z24.h
    fmla z6.h, p0/m, z14.h, z24.h
    fmla z7.h, p0/m, z15.h, z24.h
    add z20.h, z20.h, z30.h
    add z20.h, z20.h, z25.h
    add z21.h, z21.h, z30.h
    add z21.h, z21.h, z26.h
    add z22.h, z22.h, z30.h
    add z22.h, z22.h, z27.h
    add z23.h, z23.h, z30.h
    add z23.h, z23.h, z28.h
    ld1rh {z24.h}, p0/z, [x1]
    add x0, x0, #256
    add x1, x1, #2
    ld1sb {z8.h}, p0/z, [x0, #0, mul vl]
    ld1sb {z9.h}, p0/z, [x0, #1, mul vl]
    ld1sb {z10.h}, p0/z, [x0, #2, mul vl]
    ld1sb {z11.h}, p0/z, [x0, #3, mul vl]
    ld1sb {z12.h}, p0/z, [x0, #4, mul vl]
    ld1sb {z13.h}, p0/z, [x0, #5, mul vl]
    ld1sb {z14.h}, p0/z, [x0, #6, mul vl]
    ld1sb {z15.h}, p0/z, [x0, #7, mul vl]
    and z25.d, z8.d, z29.d
    and z26.d, z9.d, z29.d
    and z27.d, z10.d, z29.d
    and z28.d, z11.d, z29.d
    tbl z25.h, {z31.h}, z25.h
    tbl z26.h, {z31.h}, z26.h
    tbl z27.h, {z31.h}, z27.h
    tbl z28.h, {z31.h}, z28.h
    lsl z8.h, z8.h, #7
    and z8.h, z8.h, #0xbfff
    lsl z9.h, z9.h, #7
    and z9.h, z9.h, #0xbfff
    lsl z10.h, z10.h, #7
    and z10.h, z10.h, #0xbfff
    lsl z11.h, z11.h, #7
    and z11.h, z11.h, #0xbfff
    fmla z0.h, p0/m, z16.h, z24.h
    fmla z1.h, p0/m, z17.h, z24.h
    fmla z2.h, p0/m, z18.h, z24.h
    fmla z3.h, p0/m, z19.h, z24.h
    add z8.h, z8.h, z30.h
    add z8.h, z8.h, z25.h
    add z9.h, z9.h, z30.h
    add z9.h, z9.h, z26.h
    add z10.h, z10.h, z30.h
    add z10.h, z10.h, z27.h
    add z11.h, z11.h, z30.h
    add z11.h, z11.h, z28.h
    and z25.d, z12.d, z29.d
    and z26.d, z13.d, z29.d
    and z27.d, z14.d, z29.d
    and z28.d, z15.d, z29.d
    tbl z25.h, {z31.h}, z25.h
    tbl z26.h, {z31.h}, z26.h
    tbl z27.h, {z31.h}, z27.h
    tbl z28.h, {z31.h}, z28.h
    lsl z12.h, z12.h, #7
    and z12.h, z12.h, #0xbfff
    lsl z13.h, z13.h, #7
    and z13.h, z13.h, #0xbfff
    lsl z14.h, z14.h, #7
    and z14.h, z14.h, #0xbfff
    lsl z15.h, z15.h, #7
    and z15.h, z15.h, #0xbfff
    fmla z4.h, p0/m, z20.h, z24.h
    fmla z5.h, p0/m, z21.h, z24.h
    fmla z6.h, p0/m, z22.h, z24.h
    fmla z7.h, p0/m, z23.h, z24.h
    add z12.h, z12.h, z30.h
    add z12.h, z12.h, z25.h
    add z13.h, z13.h, z30.h
    add z13.h, z13.h, z26.h
    add z14.h, z14.h, z30.h
    add z14.h, z14.h, z27.h
    add z15.h, z15.h, z30.h
    add z15.h, z15.h, z28.h
    ld1rh {z24.h}, p0/z, [x1]
    subs x10, x10, #1
    b.ne 1b
    add x0, x0, #256
    add x1, x1, #2
    ld1sb {z16.h}, p0/z, [x0, #0, mul vl]
    ld1sb {z17.h}, p0/z, [x0, #1, mul vl]
    ld1sb {z18.h}, p0/z, [x0, #2, mul vl]
    ld1sb {z19.h}, p0/z, [x0, #3, mul vl]
    ld1sb {z20.h}, p0/z, [x0, #4, mul vl]
    ld1sb {z21.h}, p0/z, [x0, #5, mul vl]
    ld1sb {z22.h}, p0/z, [x0, #6, mul vl]
    ld1sb {z23.h}, p0/z, [x0, #7, mul vl]
    and z25.d, z16.d, z29.d
    and z26.d, z17.d, z29.d
    and z27.d, z18.d, z29.d
    and z28.d, z19.d, z29.d
    tbl z25.h, {z31.h}, z25.h
    tbl z26.h, {z31.h}, z26.h
    tbl z27.h, {z31.h}, z27.h
    tbl z28.h, {z31.h}, z28.h
    lsl z16.h, z16.h, #7
    and z16.h, z16.h, #0xbfff
    lsl z17.h, z17.h, #7
    and z17.h, z17.h, #0xbfff
    lsl z18.h, z18.h, #7
    and z18.h, z18.h, #0xbfff
    lsl z19.h, z19.h, #7
    and z19.h, z19.h, #0xbfff
    fmla z0.h, p0/m, z8.h, z24.h
    fmla z1.h, p0/m, z9.h, z24.h
    fmla z2.h, p0/m, z10.h, z24.h
    fmla z3.h, p0/m, z11.h, z24.h
    add z16.h, z16.h, z30.h
    add z16.h, z16.h, z25.h
    add z17.h, z17.h, z30.h
    add z17.h, z17.h, z26.h
    add z18.h, z18.h, z30.h
    add z18.h, z18.h, z27.h
    add z19.h, z19.h, z30.h
    add z19.h, z19.h, z28.h
    and z25.d, z20.d, z29.d
    and z26.d, z21.d, z29.d
    and z27.d, z22.d, z29.d
    and z28.d, z23.d, z29.d
    tbl z25.h, {z31.h}, z25.h
    tbl z26.h, {z31.h}, z26.h
    tbl z27.h, {z31.h}, z27.h
    tbl z28.h, {z31.h}, z28.h
    lsl z20.h, z20.h, #7
    and z20.h, z20.h, #0xbfff
    lsl z21.h, z21.h, #7
    and z21.h, z21.h, #0xbfff
    lsl z22.h, z22.h, #7
    and z22.h, z22.h, #0xbfff
    lsl z23.h, z23.h, #7
    and z23.h, z23.h, #0xbfff
    fmla z4.h, p0/m, z12.h, z24.h
    fmla z5.h, p0/m, z13.h, z24.h
    fmla z6.h, p0/m, z14.h, z24.h
    fmla z7.h, p0/m, z15.h, z24.h
    add z20.h, z20.h, z30.h
    add z20.h, z20.h, z25.h
    add z21.h, z21.h, z30.h
    add z21.h, z21.h, z26.h
    add z22.h, z22.h, z30.h
    add z22.h, z22.h, z27.h
    add z23.h, z23.h, z30.h
    add z23.h, z23.h, z28.h
    ld1rh {z24.h}, p0/z, [x1]
    fmla z0.h, p0/m, z16.h, z24.h
    fmla z1.h, p0/m, z17.h, z24.h
    fmla z2.h, p0/m, z18.h, z24.h
    fmla z3.h, p0/m, z19.h, z24.h
    fmla z4.h, p0/m, z20.h, z24.h
    fmla z5.h, p0/m, z21.h, z24.h
    fmla z6.h, p0/m, z22.h, z24.h
    fmla z7.h, p0/m, z23.h, z24.h
    st1h {z0.h}, p0, [x2, #0, mul vl]
    st1h {z1.h}, p0, [x2, #1, mul vl]
    st1h {z2.h}, p0, [x2, #2, mul vl]
    st1h {z3.h}, p0, [x2, #3, mul vl]
    st1h {z4.h}, p0, [x2, #4, mul vl]
    st1h {z5.h}, p0, [x2, #5, mul vl]
    st1h {z6.h}, p0, [x2, #6, mul vl]
    st1h {z7.h}, p0, [x2, #7, mul vl]
    ldp d14, d15, [sp], #16
    ldp d12, d13, [sp], #16
    ldp d10, d11, [sp], #16
    ldp d8, d9, [sp], #16
    ret
.size w8_e4m3_bias_f16, .-w8_e4m3_bias_f16

.section .rodata
.p2align 6
.Le4_bias:
.hword 0x5e80,0xe000,0xf780,0xfb00,0xfc80,0xfe00,0xfe80,0xff00,0xff80,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0


.text
.p2align 6
/* P9 sign planes are transposed within each 64 columns; see e4_pack.h.
 * Shift exact magnitude bits, apply the predicate sign, and preserve K order. */
.global w8_e4m3_p9_f16
.type w8_e4m3_p9_f16, %function
w8_e4m3_p9_f16:
    stp d8, d9, [sp, #-16]!
    stp d10, d11, [sp, #-16]!
    stp d12, d13, [sp, #-16]!
    stp d14, d15, [sp, #-16]!
    ptrue p0.h
    eor z0.d, z0.d, z0.d
    eor z1.d, z1.d, z1.d
    eor z2.d, z2.d, z2.d
    eor z3.d, z3.d, z3.d
    eor z4.d, z4.d, z4.d
    eor z5.d, z5.d, z5.d
    eor z6.d, z6.d, z6.d
    eor z7.d, z7.d, z7.d
    add x0, x0, #64
    ld1b {z8.h}, p0/z, [x0, #0, mul vl]
    ld1b {z9.h}, p0/z, [x0, #1, mul vl]
    ld1b {z10.h}, p0/z, [x0, #2, mul vl]
    ld1b {z11.h}, p0/z, [x0, #3, mul vl]
    ld1b {z12.h}, p0/z, [x0, #4, mul vl]
    ld1b {z13.h}, p0/z, [x0, #5, mul vl]
    ld1b {z14.h}, p0/z, [x0, #6, mul vl]
    ld1b {z15.h}, p0/z, [x0, #7, mul vl]
    ld1rh {z24.h}, p0/z, [x1]
    add x3, x0, #256
    ldr p1, [x3, #0, mul vl]
    trn2 p3.b, p1.b, p1.b
    lsl z8.h, z8.h, #7
    fneg z8.h, p1/m, z8.h
    lsl z9.h, z9.h, #7
    fneg z9.h, p3/m, z9.h
    ldr p1, [x3, #1, mul vl]
    trn2 p3.b, p1.b, p1.b
    lsl z10.h, z10.h, #7
    fneg z10.h, p1/m, z10.h
    lsl z11.h, z11.h, #7
    fneg z11.h, p3/m, z11.h
    ldr p1, [x3, #2, mul vl]
    trn2 p3.b, p1.b, p1.b
    lsl z12.h, z12.h, #7
    fneg z12.h, p1/m, z12.h
    lsl z13.h, z13.h, #7
    fneg z13.h, p3/m, z13.h
    ldr p1, [x3, #3, mul vl]
    trn2 p3.b, p1.b, p1.b
    lsl z14.h, z14.h, #7
    fneg z14.h, p1/m, z14.h
    lsl z15.h, z15.h, #7
    fneg z15.h, p3/m, z15.h
    mov x10, #63
.p2align 6
1:
    add x0, x0, #288
    add x1, x1, #2
    ld1b {z16.h}, p0/z, [x0, #0, mul vl]
    ld1b {z17.h}, p0/z, [x0, #1, mul vl]
    ld1b {z18.h}, p0/z, [x0, #2, mul vl]
    ld1b {z19.h}, p0/z, [x0, #3, mul vl]
    ld1b {z20.h}, p0/z, [x0, #4, mul vl]
    ld1b {z21.h}, p0/z, [x0, #5, mul vl]
    ld1b {z22.h}, p0/z, [x0, #6, mul vl]
    ld1b {z23.h}, p0/z, [x0, #7, mul vl]
    ld1rh {z25.h}, p0/z, [x1]
    add x3, x0, #256
    ldr p1, [x3, #0, mul vl]
    trn2 p3.b, p1.b, p1.b
    fmla z0.h, p0/m, z8.h, z24.h
    lsl z16.h, z16.h, #7
    fneg z16.h, p1/m, z16.h
    lsl z17.h, z17.h, #7
    fmla z1.h, p0/m, z9.h, z24.h
    fneg z17.h, p3/m, z17.h
    ldr p1, [x3, #1, mul vl]
    trn2 p3.b, p1.b, p1.b
    fmla z2.h, p0/m, z10.h, z24.h
    lsl z18.h, z18.h, #7
    fneg z18.h, p1/m, z18.h
    lsl z19.h, z19.h, #7
    fmla z3.h, p0/m, z11.h, z24.h
    fneg z19.h, p3/m, z19.h
    ldr p1, [x3, #2, mul vl]
    trn2 p3.b, p1.b, p1.b
    fmla z4.h, p0/m, z12.h, z24.h
    lsl z20.h, z20.h, #7
    fneg z20.h, p1/m, z20.h
    lsl z21.h, z21.h, #7
    fmla z5.h, p0/m, z13.h, z24.h
    fneg z21.h, p3/m, z21.h
    ldr p1, [x3, #3, mul vl]
    trn2 p3.b, p1.b, p1.b
    fmla z6.h, p0/m, z14.h, z24.h
    lsl z22.h, z22.h, #7
    fneg z22.h, p1/m, z22.h
    lsl z23.h, z23.h, #7
    fmla z7.h, p0/m, z15.h, z24.h
    fneg z23.h, p3/m, z23.h
    add x0, x0, #288
    add x1, x1, #2
    ld1b {z8.h}, p0/z, [x0, #0, mul vl]
    ld1b {z9.h}, p0/z, [x0, #1, mul vl]
    ld1b {z10.h}, p0/z, [x0, #2, mul vl]
    ld1b {z11.h}, p0/z, [x0, #3, mul vl]
    ld1b {z12.h}, p0/z, [x0, #4, mul vl]
    ld1b {z13.h}, p0/z, [x0, #5, mul vl]
    ld1b {z14.h}, p0/z, [x0, #6, mul vl]
    ld1b {z15.h}, p0/z, [x0, #7, mul vl]
    ld1rh {z24.h}, p0/z, [x1]
    add x3, x0, #256
    ldr p1, [x3, #0, mul vl]
    trn2 p3.b, p1.b, p1.b
    fmla z0.h, p0/m, z16.h, z25.h
    lsl z8.h, z8.h, #7
    fneg z8.h, p1/m, z8.h
    lsl z9.h, z9.h, #7
    fmla z1.h, p0/m, z17.h, z25.h
    fneg z9.h, p3/m, z9.h
    ldr p1, [x3, #1, mul vl]
    trn2 p3.b, p1.b, p1.b
    fmla z2.h, p0/m, z18.h, z25.h
    lsl z10.h, z10.h, #7
    fneg z10.h, p1/m, z10.h
    lsl z11.h, z11.h, #7
    fmla z3.h, p0/m, z19.h, z25.h
    fneg z11.h, p3/m, z11.h
    ldr p1, [x3, #2, mul vl]
    trn2 p3.b, p1.b, p1.b
    fmla z4.h, p0/m, z20.h, z25.h
    lsl z12.h, z12.h, #7
    fneg z12.h, p1/m, z12.h
    lsl z13.h, z13.h, #7
    fmla z5.h, p0/m, z21.h, z25.h
    fneg z13.h, p3/m, z13.h
    ldr p1, [x3, #3, mul vl]
    trn2 p3.b, p1.b, p1.b
    fmla z6.h, p0/m, z22.h, z25.h
    lsl z14.h, z14.h, #7
    fneg z14.h, p1/m, z14.h
    lsl z15.h, z15.h, #7
    fmla z7.h, p0/m, z23.h, z25.h
    fneg z15.h, p3/m, z15.h
    subs x10, x10, #1
    b.ne 1b
    add x0, x0, #288
    add x1, x1, #2
    ld1b {z16.h}, p0/z, [x0, #0, mul vl]
    ld1b {z17.h}, p0/z, [x0, #1, mul vl]
    ld1b {z18.h}, p0/z, [x0, #2, mul vl]
    ld1b {z19.h}, p0/z, [x0, #3, mul vl]
    ld1b {z20.h}, p0/z, [x0, #4, mul vl]
    ld1b {z21.h}, p0/z, [x0, #5, mul vl]
    ld1b {z22.h}, p0/z, [x0, #6, mul vl]
    ld1b {z23.h}, p0/z, [x0, #7, mul vl]
    ld1rh {z25.h}, p0/z, [x1]
    add x3, x0, #256
    ldr p1, [x3, #0, mul vl]
    trn2 p3.b, p1.b, p1.b
    fmla z0.h, p0/m, z8.h, z24.h
    lsl z16.h, z16.h, #7
    fneg z16.h, p1/m, z16.h
    lsl z17.h, z17.h, #7
    fmla z1.h, p0/m, z9.h, z24.h
    fneg z17.h, p3/m, z17.h
    ldr p1, [x3, #1, mul vl]
    trn2 p3.b, p1.b, p1.b
    fmla z2.h, p0/m, z10.h, z24.h
    lsl z18.h, z18.h, #7
    fneg z18.h, p1/m, z18.h
    lsl z19.h, z19.h, #7
    fmla z3.h, p0/m, z11.h, z24.h
    fneg z19.h, p3/m, z19.h
    ldr p1, [x3, #2, mul vl]
    trn2 p3.b, p1.b, p1.b
    fmla z4.h, p0/m, z12.h, z24.h
    lsl z20.h, z20.h, #7
    fneg z20.h, p1/m, z20.h
    lsl z21.h, z21.h, #7
    fmla z5.h, p0/m, z13.h, z24.h
    fneg z21.h, p3/m, z21.h
    ldr p1, [x3, #3, mul vl]
    trn2 p3.b, p1.b, p1.b
    fmla z6.h, p0/m, z14.h, z24.h
    lsl z22.h, z22.h, #7
    fneg z22.h, p1/m, z22.h
    lsl z23.h, z23.h, #7
    fmla z7.h, p0/m, z15.h, z24.h
    fneg z23.h, p3/m, z23.h
    fmla z0.h, p0/m, z16.h, z25.h
    fmla z1.h, p0/m, z17.h, z25.h
    fmla z2.h, p0/m, z18.h, z25.h
    fmla z3.h, p0/m, z19.h, z25.h
    fmla z4.h, p0/m, z20.h, z25.h
    fmla z5.h, p0/m, z21.h, z25.h
    fmla z6.h, p0/m, z22.h, z25.h
    fmla z7.h, p0/m, z23.h, z25.h
    st1h {z0.h}, p0, [x2, #0, mul vl]
    st1h {z1.h}, p0, [x2, #1, mul vl]
    st1h {z2.h}, p0, [x2, #2, mul vl]
    st1h {z3.h}, p0, [x2, #3, mul vl]
    st1h {z4.h}, p0, [x2, #4, mul vl]
    st1h {z5.h}, p0, [x2, #5, mul vl]
    st1h {z6.h}, p0, [x2, #6, mul vl]
    st1h {z7.h}, p0, [x2, #7, mul vl]
    ldp d14, d15, [sp], #16
    ldp d12, d13, [sp], #16
    ldp d10, d11, [sp], #16
    ldp d8, d9, [sp], #16
    ret
.size w8_e4m3_p9_f16, .-w8_e4m3_p9_f16
.p2align 6
/* Finite P9 magnitudes shifted by 20 encode w * 2^-112 as normal FP32.
 * Reusable activations are scaled by 2^112 exactly by e4_p9_prepare. */
.global w8_e4m3_p9_f32
.type w8_e4m3_p9_f32, %function
w8_e4m3_p9_f32:
    ldr w9, [x0, #8]
    cbnz w9, e4_p9_f32_fallback
    ldr w9, [x1, #1024]
    cbz w9, e4_p9_f32_fallback
    stp d8, d9, [sp, #-16]!
    stp d10, d11, [sp, #-16]!
    stp d12, d13, [sp, #-16]!
    stp d14, d15, [sp, #-16]!
    mov z31.s, #0x80000000
    ptrue p0.s
    eor z0.d, z0.d, z0.d
    eor z1.d, z1.d, z1.d
    eor z2.d, z2.d, z2.d
    eor z3.d, z3.d, z3.d
    eor z4.d, z4.d, z4.d
    eor z5.d, z5.d, z5.d
    eor z6.d, z6.d, z6.d
    eor z7.d, z7.d, z7.d
    add x0, x0, #64
    add x3, x0, #128
    ldr p1, [x3]
    ldr p2, [x3, #1, mul vl]
    ld1b {z8.s}, p0/z, [x0, #0, mul vl]
    ld1b {z9.s}, p0/z, [x0, #1, mul vl]
    ld1b {z10.s}, p0/z, [x0, #2, mul vl]
    ld1b {z11.s}, p0/z, [x0, #3, mul vl]
    ld1b {z12.s}, p0/z, [x0, #4, mul vl]
    ld1b {z13.s}, p0/z, [x0, #5, mul vl]
    ld1b {z14.s}, p0/z, [x0, #6, mul vl]
    ld1b {z15.s}, p0/z, [x0, #7, mul vl]
    lsl z8.s, z8.s, #20
    lsl z9.s, z9.s, #20
    lsl z10.s, z10.s, #20
    lsl z11.s, z11.s, #20
    lsl z12.s, z12.s, #20
    lsl z13.s, z13.s, #20
    lsl z14.s, z14.s, #20
    lsl z15.s, z15.s, #20
    ld1rw {z24.s}, p0/z, [x1]
    trn2 p5.b, p1.b, p1.b
    trn2 p6.h, p1.h, p1.h
    trn2 p7.h, p5.h, p5.h
    eor z8.s, p1/m, z8.s, z31.s
    eor z9.s, p5/m, z9.s, z31.s
    eor z10.s, p6/m, z10.s, z31.s
    eor z11.s, p7/m, z11.s, z31.s
    trn2 p5.b, p2.b, p2.b
    trn2 p6.h, p2.h, p2.h
    trn2 p7.h, p5.h, p5.h
    eor z12.s, p2/m, z12.s, z31.s
    eor z13.s, p5/m, z13.s, z31.s
    eor z14.s, p6/m, z14.s, z31.s
    eor z15.s, p7/m, z15.s, z31.s
    /* Eight K transitions per iteration; after 15 iterations, K=120
     * remains prepared. Drain seven transitions, then retire K=127. */
.macro E4_P9_F32_PAIR
    add x0, x0, #144
    add x1, x1, #4
    add x3, x0, #128
    ldr p1, [x3]
    ldr p2, [x3, #1, mul vl]
    ld1b {z16.s}, p0/z, [x0, #0, mul vl]
    ld1b {z17.s}, p0/z, [x0, #1, mul vl]
    ld1b {z18.s}, p0/z, [x0, #2, mul vl]
    ld1b {z19.s}, p0/z, [x0, #3, mul vl]
    ld1b {z20.s}, p0/z, [x0, #4, mul vl]
    ld1b {z21.s}, p0/z, [x0, #5, mul vl]
    ld1b {z22.s}, p0/z, [x0, #6, mul vl]
    ld1b {z23.s}, p0/z, [x0, #7, mul vl]
    ld1rw {z25.s}, p0/z, [x1]
    trn2 p5.b, p1.b, p1.b
    trn2 p6.h, p1.h, p1.h
    trn2 p7.h, p5.h, p5.h
    lsl z16.s, z16.s, #20
    lsl z17.s, z17.s, #20
    lsl z18.s, z18.s, #20
    lsl z19.s, z19.s, #20
    fmla z0.s, p0/m, z8.s, z24.s
    eor z16.s, p1/m, z16.s, z31.s
    fmla z1.s, p0/m, z9.s, z24.s
    eor z17.s, p5/m, z17.s, z31.s
    fmla z2.s, p0/m, z10.s, z24.s
    eor z18.s, p6/m, z18.s, z31.s
    fmla z3.s, p0/m, z11.s, z24.s
    eor z19.s, p7/m, z19.s, z31.s
    trn2 p5.b, p2.b, p2.b
    trn2 p6.h, p2.h, p2.h
    trn2 p7.h, p5.h, p5.h
    lsl z20.s, z20.s, #20
    lsl z21.s, z21.s, #20
    lsl z22.s, z22.s, #20
    lsl z23.s, z23.s, #20
    fmla z4.s, p0/m, z12.s, z24.s
    eor z20.s, p2/m, z20.s, z31.s
    fmla z5.s, p0/m, z13.s, z24.s
    eor z21.s, p5/m, z21.s, z31.s
    fmla z6.s, p0/m, z14.s, z24.s
    eor z22.s, p6/m, z22.s, z31.s
    fmla z7.s, p0/m, z15.s, z24.s
    eor z23.s, p7/m, z23.s, z31.s
    add x0, x0, #144
    add x1, x1, #4
    add x3, x0, #128
    ldr p1, [x3]
    ldr p2, [x3, #1, mul vl]
    ld1b {z8.s}, p0/z, [x0, #0, mul vl]
    ld1b {z9.s}, p0/z, [x0, #1, mul vl]
    ld1b {z10.s}, p0/z, [x0, #2, mul vl]
    ld1b {z11.s}, p0/z, [x0, #3, mul vl]
    ld1b {z12.s}, p0/z, [x0, #4, mul vl]
    ld1b {z13.s}, p0/z, [x0, #5, mul vl]
    ld1b {z14.s}, p0/z, [x0, #6, mul vl]
    ld1b {z15.s}, p0/z, [x0, #7, mul vl]
    ld1rw {z24.s}, p0/z, [x1]
    trn2 p5.b, p1.b, p1.b
    trn2 p6.h, p1.h, p1.h
    trn2 p7.h, p5.h, p5.h
    lsl z8.s, z8.s, #20
    lsl z9.s, z9.s, #20
    lsl z10.s, z10.s, #20
    lsl z11.s, z11.s, #20
    fmla z0.s, p0/m, z16.s, z25.s
    eor z8.s, p1/m, z8.s, z31.s
    fmla z1.s, p0/m, z17.s, z25.s
    eor z9.s, p5/m, z9.s, z31.s
    fmla z2.s, p0/m, z18.s, z25.s
    eor z10.s, p6/m, z10.s, z31.s
    fmla z3.s, p0/m, z19.s, z25.s
    eor z11.s, p7/m, z11.s, z31.s
    trn2 p5.b, p2.b, p2.b
    trn2 p6.h, p2.h, p2.h
    trn2 p7.h, p5.h, p5.h
    lsl z12.s, z12.s, #20
    lsl z13.s, z13.s, #20
    lsl z14.s, z14.s, #20
    lsl z15.s, z15.s, #20
    fmla z4.s, p0/m, z20.s, z25.s
    eor z12.s, p2/m, z12.s, z31.s
    fmla z5.s, p0/m, z21.s, z25.s
    eor z13.s, p5/m, z13.s, z31.s
    fmla z6.s, p0/m, z22.s, z25.s
    eor z14.s, p6/m, z14.s, z31.s
    fmla z7.s, p0/m, z23.s, z25.s
    eor z15.s, p7/m, z15.s, z31.s
.endm
    mov x10, #15
.p2align 6
1:
.rept 4
    E4_P9_F32_PAIR
.endr
    subs x10, x10, #1
    b.ne 1b
.rept 3
    E4_P9_F32_PAIR
.endr
    add x0, x0, #144
    add x1, x1, #4
    add x3, x0, #128
    ldr p1, [x3]
    ldr p2, [x3, #1, mul vl]
    ld1b {z16.s}, p0/z, [x0, #0, mul vl]
    ld1b {z17.s}, p0/z, [x0, #1, mul vl]
    ld1b {z18.s}, p0/z, [x0, #2, mul vl]
    ld1b {z19.s}, p0/z, [x0, #3, mul vl]
    ld1b {z20.s}, p0/z, [x0, #4, mul vl]
    ld1b {z21.s}, p0/z, [x0, #5, mul vl]
    ld1b {z22.s}, p0/z, [x0, #6, mul vl]
    ld1b {z23.s}, p0/z, [x0, #7, mul vl]
    ld1rw {z25.s}, p0/z, [x1]
    trn2 p5.b, p1.b, p1.b
    trn2 p6.h, p1.h, p1.h
    trn2 p7.h, p5.h, p5.h
    lsl z16.s, z16.s, #20
    lsl z17.s, z17.s, #20
    lsl z18.s, z18.s, #20
    lsl z19.s, z19.s, #20
    fmla z0.s, p0/m, z8.s, z24.s
    eor z16.s, p1/m, z16.s, z31.s
    fmla z1.s, p0/m, z9.s, z24.s
    eor z17.s, p5/m, z17.s, z31.s
    fmla z2.s, p0/m, z10.s, z24.s
    eor z18.s, p6/m, z18.s, z31.s
    fmla z3.s, p0/m, z11.s, z24.s
    eor z19.s, p7/m, z19.s, z31.s
    trn2 p5.b, p2.b, p2.b
    trn2 p6.h, p2.h, p2.h
    trn2 p7.h, p5.h, p5.h
    lsl z20.s, z20.s, #20
    lsl z21.s, z21.s, #20
    lsl z22.s, z22.s, #20
    lsl z23.s, z23.s, #20
    fmla z4.s, p0/m, z12.s, z24.s
    eor z20.s, p2/m, z20.s, z31.s
    fmla z5.s, p0/m, z13.s, z24.s
    eor z21.s, p5/m, z21.s, z31.s
    fmla z6.s, p0/m, z14.s, z24.s
    eor z22.s, p6/m, z22.s, z31.s
    fmla z7.s, p0/m, z15.s, z24.s
    eor z23.s, p7/m, z23.s, z31.s
    fmla z0.s, p0/m, z16.s, z25.s
    fmla z1.s, p0/m, z17.s, z25.s
    fmla z2.s, p0/m, z18.s, z25.s
    fmla z3.s, p0/m, z19.s, z25.s
    fmla z4.s, p0/m, z20.s, z25.s
    fmla z5.s, p0/m, z21.s, z25.s
    fmla z6.s, p0/m, z22.s, z25.s
    fmla z7.s, p0/m, z23.s, z25.s
    st1w {z0.s}, p0, [x2, #0, mul vl]
    st1w {z1.s}, p0, [x2, #1, mul vl]
    st1w {z2.s}, p0, [x2, #2, mul vl]
    st1w {z3.s}, p0, [x2, #3, mul vl]
    st1w {z4.s}, p0, [x2, #4, mul vl]
    st1w {z5.s}, p0, [x2, #5, mul vl]
    st1w {z6.s}, p0, [x2, #6, mul vl]
    st1w {z7.s}, p0, [x2, #7, mul vl]
    ldp d14, d15, [sp], #16
    ldp d12, d13, [sp], #16
    ldp d10, d11, [sp], #16
    ldp d8, d9, [sp], #16
    ret
.size w8_e4m3_p9_f32, .-w8_e4m3_p9_f32
.section .note.GNU-stack,"",%progbits
