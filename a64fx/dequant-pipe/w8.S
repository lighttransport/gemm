/* Fixed SVE512, K-major byte weights; see w8.h.
 * INT8 / E5M2: prologue prepares K=0; 63 two-step iterations retire
 * K=0..125, then a final transition and load-free drain retire K=126/127.
 * z0-z7 retain eight ascending-K output chains. z8-z15 / z16-z23
 * alternate decoded weights, and z24/z25 alternate shared activations.
 * E5M2 shifts directly into IEEE half bits; FCVT widens exactly for FP32.
 * FPCR is not changed: no flush-to-zero or special-value shortcut.
 *
 * E4M3 FP16: signed widening supplies the sign. Shifting by 7 and masking
 * 0xbfff forms a sign-preserving biased encoding; +0x2000 fixes the bias.
 * The wrapped (code+1)&127 lookup returns corrections only for subnormals
 * and NaNs (indices 1..8 and 0), and zero for every other code.
 * E4M3 FP32: two independent 64-byte streams build exact BF16 high/low
 * bytes with byte tables and corrections, then ZIP with zero low halfwords
 * forms FP32 bits. BF16 exactly represents every E4M3 finite value.
 * No layout changes or expanded weight buffers are used. */
.text
.p2align 6
.global w8_i8_f16
.type w8_i8_f16, %function
w8_i8_f16:
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
    ld1sb {z8.h}, p0/z, [x0, #0, mul vl]
    ld1sb {z9.h}, p0/z, [x0, #1, mul vl]
    ld1sb {z10.h}, p0/z, [x0, #2, mul vl]
    ld1sb {z11.h}, p0/z, [x0, #3, mul vl]
    ld1sb {z12.h}, p0/z, [x0, #4, mul vl]
    ld1sb {z13.h}, p0/z, [x0, #5, mul vl]
    ld1sb {z14.h}, p0/z, [x0, #6, mul vl]
    ld1sb {z15.h}, p0/z, [x0, #7, mul vl]
    ld1rh {z24.h}, p0/z, [x1]
    scvtf z8.h, p0/m, z8.h
    scvtf z9.h, p0/m, z9.h
    scvtf z10.h, p0/m, z10.h
    scvtf z11.h, p0/m, z11.h
    scvtf z12.h, p0/m, z12.h
    scvtf z13.h, p0/m, z13.h
    scvtf z14.h, p0/m, z14.h
    scvtf z15.h, p0/m, z15.h
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
    ld1rh {z25.h}, p0/z, [x1]
    fmla z0.h, p0/m, z8.h, z24.h
    scvtf z16.h, p0/m, z16.h
    fmla z1.h, p0/m, z9.h, z24.h
    scvtf z17.h, p0/m, z17.h
    fmla z2.h, p0/m, z10.h, z24.h
    scvtf z18.h, p0/m, z18.h
    fmla z3.h, p0/m, z11.h, z24.h
    scvtf z19.h, p0/m, z19.h
    fmla z4.h, p0/m, z12.h, z24.h
    scvtf z20.h, p0/m, z20.h
    fmla z5.h, p0/m, z13.h, z24.h
    scvtf z21.h, p0/m, z21.h
    fmla z6.h, p0/m, z14.h, z24.h
    scvtf z22.h, p0/m, z22.h
    fmla z7.h, p0/m, z15.h, z24.h
    scvtf z23.h, p0/m, z23.h
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
    ld1rh {z24.h}, p0/z, [x1]
    fmla z0.h, p0/m, z16.h, z25.h
    scvtf z8.h, p0/m, z8.h
    fmla z1.h, p0/m, z17.h, z25.h
    scvtf z9.h, p0/m, z9.h
    fmla z2.h, p0/m, z18.h, z25.h
    scvtf z10.h, p0/m, z10.h
    fmla z3.h, p0/m, z19.h, z25.h
    scvtf z11.h, p0/m, z11.h
    fmla z4.h, p0/m, z20.h, z25.h
    scvtf z12.h, p0/m, z12.h
    fmla z5.h, p0/m, z21.h, z25.h
    scvtf z13.h, p0/m, z13.h
    fmla z6.h, p0/m, z22.h, z25.h
    scvtf z14.h, p0/m, z14.h
    fmla z7.h, p0/m, z23.h, z25.h
    scvtf z15.h, p0/m, z15.h
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
    ld1rh {z25.h}, p0/z, [x1]
    fmla z0.h, p0/m, z8.h, z24.h
    scvtf z16.h, p0/m, z16.h
    fmla z1.h, p0/m, z9.h, z24.h
    scvtf z17.h, p0/m, z17.h
    fmla z2.h, p0/m, z10.h, z24.h
    scvtf z18.h, p0/m, z18.h
    fmla z3.h, p0/m, z11.h, z24.h
    scvtf z19.h, p0/m, z19.h
    fmla z4.h, p0/m, z12.h, z24.h
    scvtf z20.h, p0/m, z20.h
    fmla z5.h, p0/m, z13.h, z24.h
    scvtf z21.h, p0/m, z21.h
    fmla z6.h, p0/m, z14.h, z24.h
    scvtf z22.h, p0/m, z22.h
    fmla z7.h, p0/m, z15.h, z24.h
    scvtf z23.h, p0/m, z23.h
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
.size w8_i8_f16, .-w8_i8_f16
.p2align 6
.global w8_i8_f32
.type w8_i8_f32, %function
w8_i8_f32:
    stp d8, d9, [sp, #-16]!
    stp d10, d11, [sp, #-16]!
    stp d12, d13, [sp, #-16]!
    stp d14, d15, [sp, #-16]!
    ptrue p0.s
    eor z0.d, z0.d, z0.d
    eor z1.d, z1.d, z1.d
    eor z2.d, z2.d, z2.d
    eor z3.d, z3.d, z3.d
    eor z4.d, z4.d, z4.d
    eor z5.d, z5.d, z5.d
    eor z6.d, z6.d, z6.d
    eor z7.d, z7.d, z7.d
    ld1sb {z8.s}, p0/z, [x0, #0, mul vl]
    ld1sb {z9.s}, p0/z, [x0, #1, mul vl]
    ld1sb {z10.s}, p0/z, [x0, #2, mul vl]
    ld1sb {z11.s}, p0/z, [x0, #3, mul vl]
    ld1sb {z12.s}, p0/z, [x0, #4, mul vl]
    ld1sb {z13.s}, p0/z, [x0, #5, mul vl]
    ld1sb {z14.s}, p0/z, [x0, #6, mul vl]
    ld1sb {z15.s}, p0/z, [x0, #7, mul vl]
    ld1rw {z24.s}, p0/z, [x1]
    scvtf z8.s, p0/m, z8.s
    scvtf z9.s, p0/m, z9.s
    scvtf z10.s, p0/m, z10.s
    scvtf z11.s, p0/m, z11.s
    scvtf z12.s, p0/m, z12.s
    scvtf z13.s, p0/m, z13.s
    scvtf z14.s, p0/m, z14.s
    scvtf z15.s, p0/m, z15.s
    mov x10, #63
.p2align 6
1:
    add x0, x0, #128
    add x1, x1, #4
    ld1sb {z16.s}, p0/z, [x0, #0, mul vl]
    ld1sb {z17.s}, p0/z, [x0, #1, mul vl]
    ld1sb {z18.s}, p0/z, [x0, #2, mul vl]
    ld1sb {z19.s}, p0/z, [x0, #3, mul vl]
    ld1sb {z20.s}, p0/z, [x0, #4, mul vl]
    ld1sb {z21.s}, p0/z, [x0, #5, mul vl]
    ld1sb {z22.s}, p0/z, [x0, #6, mul vl]
    ld1sb {z23.s}, p0/z, [x0, #7, mul vl]
    ld1rw {z25.s}, p0/z, [x1]
    fmla z0.s, p0/m, z8.s, z24.s
    scvtf z16.s, p0/m, z16.s
    fmla z1.s, p0/m, z9.s, z24.s
    scvtf z17.s, p0/m, z17.s
    fmla z2.s, p0/m, z10.s, z24.s
    scvtf z18.s, p0/m, z18.s
    fmla z3.s, p0/m, z11.s, z24.s
    scvtf z19.s, p0/m, z19.s
    fmla z4.s, p0/m, z12.s, z24.s
    scvtf z20.s, p0/m, z20.s
    fmla z5.s, p0/m, z13.s, z24.s
    scvtf z21.s, p0/m, z21.s
    fmla z6.s, p0/m, z14.s, z24.s
    scvtf z22.s, p0/m, z22.s
    fmla z7.s, p0/m, z15.s, z24.s
    scvtf z23.s, p0/m, z23.s
    add x0, x0, #128
    add x1, x1, #4
    ld1sb {z8.s}, p0/z, [x0, #0, mul vl]
    ld1sb {z9.s}, p0/z, [x0, #1, mul vl]
    ld1sb {z10.s}, p0/z, [x0, #2, mul vl]
    ld1sb {z11.s}, p0/z, [x0, #3, mul vl]
    ld1sb {z12.s}, p0/z, [x0, #4, mul vl]
    ld1sb {z13.s}, p0/z, [x0, #5, mul vl]
    ld1sb {z14.s}, p0/z, [x0, #6, mul vl]
    ld1sb {z15.s}, p0/z, [x0, #7, mul vl]
    ld1rw {z24.s}, p0/z, [x1]
    fmla z0.s, p0/m, z16.s, z25.s
    scvtf z8.s, p0/m, z8.s
    fmla z1.s, p0/m, z17.s, z25.s
    scvtf z9.s, p0/m, z9.s
    fmla z2.s, p0/m, z18.s, z25.s
    scvtf z10.s, p0/m, z10.s
    fmla z3.s, p0/m, z19.s, z25.s
    scvtf z11.s, p0/m, z11.s
    fmla z4.s, p0/m, z20.s, z25.s
    scvtf z12.s, p0/m, z12.s
    fmla z5.s, p0/m, z21.s, z25.s
    scvtf z13.s, p0/m, z13.s
    fmla z6.s, p0/m, z22.s, z25.s
    scvtf z14.s, p0/m, z14.s
    fmla z7.s, p0/m, z23.s, z25.s
    scvtf z15.s, p0/m, z15.s
    subs x10, x10, #1
    b.ne 1b
    add x0, x0, #128
    add x1, x1, #4
    ld1sb {z16.s}, p0/z, [x0, #0, mul vl]
    ld1sb {z17.s}, p0/z, [x0, #1, mul vl]
    ld1sb {z18.s}, p0/z, [x0, #2, mul vl]
    ld1sb {z19.s}, p0/z, [x0, #3, mul vl]
    ld1sb {z20.s}, p0/z, [x0, #4, mul vl]
    ld1sb {z21.s}, p0/z, [x0, #5, mul vl]
    ld1sb {z22.s}, p0/z, [x0, #6, mul vl]
    ld1sb {z23.s}, p0/z, [x0, #7, mul vl]
    ld1rw {z25.s}, p0/z, [x1]
    fmla z0.s, p0/m, z8.s, z24.s
    scvtf z16.s, p0/m, z16.s
    fmla z1.s, p0/m, z9.s, z24.s
    scvtf z17.s, p0/m, z17.s
    fmla z2.s, p0/m, z10.s, z24.s
    scvtf z18.s, p0/m, z18.s
    fmla z3.s, p0/m, z11.s, z24.s
    scvtf z19.s, p0/m, z19.s
    fmla z4.s, p0/m, z12.s, z24.s
    scvtf z20.s, p0/m, z20.s
    fmla z5.s, p0/m, z13.s, z24.s
    scvtf z21.s, p0/m, z21.s
    fmla z6.s, p0/m, z14.s, z24.s
    scvtf z22.s, p0/m, z22.s
    fmla z7.s, p0/m, z15.s, z24.s
    scvtf z23.s, p0/m, z23.s
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
.size w8_i8_f32, .-w8_i8_f32
.p2align 6
.global w8_e4m3_f16
.type w8_e4m3_f16, %function
w8_e4m3_f16:
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
    mov z28.h, #127
    mov z29.h, #1
    mov z30.h, #8192
    adr x9, .Le4_sub_f16
    ld1h {z31.h}, p0/z, [x9]
    mov x10, #128
.p2align 6
1:
    ld1rh {z24.h}, p0/z, [x1]
    ld1sb {z8.h}, p0/z, [x0, #0, mul vl]
    ld1sb {z9.h}, p0/z, [x0, #1, mul vl]
    ld1sb {z10.h}, p0/z, [x0, #2, mul vl]
    ld1sb {z11.h}, p0/z, [x0, #3, mul vl]
    ld1sb {z12.h}, p0/z, [x0, #4, mul vl]
    ld1sb {z13.h}, p0/z, [x0, #5, mul vl]
    ld1sb {z14.h}, p0/z, [x0, #6, mul vl]
    ld1sb {z15.h}, p0/z, [x0, #7, mul vl]
    add z16.h, z8.h, z29.h
    and z16.d, z16.d, z28.d
    add z17.h, z9.h, z29.h
    and z17.d, z17.d, z28.d
    add z18.h, z10.h, z29.h
    and z18.d, z18.d, z28.d
    add z19.h, z11.h, z29.h
    and z19.d, z19.d, z28.d
    add z20.h, z12.h, z29.h
    and z20.d, z20.d, z28.d
    add z21.h, z13.h, z29.h
    and z21.d, z21.d, z28.d
    add z22.h, z14.h, z29.h
    and z22.d, z22.d, z28.d
    add z23.h, z15.h, z29.h
    and z23.d, z23.d, z28.d
    tbl z16.h, {z31.h}, z16.h
    tbl z17.h, {z31.h}, z17.h
    tbl z18.h, {z31.h}, z18.h
    tbl z19.h, {z31.h}, z19.h
    tbl z20.h, {z31.h}, z20.h
    tbl z21.h, {z31.h}, z21.h
    tbl z22.h, {z31.h}, z22.h
    tbl z23.h, {z31.h}, z23.h
    lsl z8.h, z8.h, #7
    and z8.h, z8.h, #0xbfff
    lsl z9.h, z9.h, #7
    and z9.h, z9.h, #0xbfff
    lsl z10.h, z10.h, #7
    and z10.h, z10.h, #0xbfff
    lsl z11.h, z11.h, #7
    and z11.h, z11.h, #0xbfff
    lsl z12.h, z12.h, #7
    and z12.h, z12.h, #0xbfff
    lsl z13.h, z13.h, #7
    and z13.h, z13.h, #0xbfff
    lsl z14.h, z14.h, #7
    and z14.h, z14.h, #0xbfff
    lsl z15.h, z15.h, #7
    and z15.h, z15.h, #0xbfff
    add z8.h, z8.h, z30.h
    add z8.h, z8.h, z16.h
    add z9.h, z9.h, z30.h
    add z9.h, z9.h, z17.h
    add z10.h, z10.h, z30.h
    add z10.h, z10.h, z18.h
    add z11.h, z11.h, z30.h
    add z11.h, z11.h, z19.h
    add z12.h, z12.h, z30.h
    add z12.h, z12.h, z20.h
    add z13.h, z13.h, z30.h
    add z13.h, z13.h, z21.h
    add z14.h, z14.h, z30.h
    add z14.h, z14.h, z22.h
    add z15.h, z15.h, z30.h
    add z15.h, z15.h, z23.h
    fmla z0.h, p0/m, z8.h, z24.h
    fmla z1.h, p0/m, z9.h, z24.h
    fmla z2.h, p0/m, z10.h, z24.h
    fmla z3.h, p0/m, z11.h, z24.h
    fmla z4.h, p0/m, z12.h, z24.h
    fmla z5.h, p0/m, z13.h, z24.h
    fmla z6.h, p0/m, z14.h, z24.h
    fmla z7.h, p0/m, z15.h, z24.h
    add x0, x0, #256
    add x1, x1, #2
    subs x10, x10, #1
    b.ne 1b
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
.size w8_e4m3_f16, .-w8_e4m3_f16
.p2align 6
.global w8_e4m3_f32
.type w8_e4m3_f32, %function
w8_e4m3_f32:
    stp d8, d9, [sp, #-16]!
    stp d10, d11, [sp, #-16]!
    stp d12, d13, [sp, #-16]!
    stp d14, d15, [sp, #-16]!
    ptrue p0.b
    adr x9, .Le4_base32
    ld1b {z24.b}, p0/z, [x9]
    adr x9, .Le4_hi32
    ld1b {z25.b}, p0/z, [x9]
    adr x9, .Le4_lo32
    ld1b {z26.b}, p0/z, [x9]
    mov z27.b, #1
    mov z28.b, #127
    mov z29.b, #0
    eor z0.d, z0.d, z0.d
    eor z1.d, z1.d, z1.d
    eor z2.d, z2.d, z2.d
    eor z3.d, z3.d, z3.d
    eor z4.d, z4.d, z4.d
    eor z5.d, z5.d, z5.d
    eor z6.d, z6.d, z6.d
    eor z7.d, z7.d, z7.d
    mov x10, #128
.p2align 6
1:
    ld1rw {z30.s}, p0/z, [x1]
    ld1b {z8.b}, p0/z, [x0, #0, mul vl]
    ld1b {z16.b}, p0/z, [x0, #1, mul vl]
    add z11.b, z8.b, z27.b
    add z19.b, z16.b, z27.b
    and z11.d, z11.d, z28.d
    and z19.d, z19.d, z28.d
    tbl z12.b, {z25.b}, z11.b
    tbl z20.b, {z25.b}, z19.b
    tbl z13.b, {z26.b}, z11.b
    tbl z21.b, {z26.b}, z19.b
    lsr z9.b, z8.b, #4
    lsr z17.b, z16.b, #4
    tbl z9.b, {z24.b}, z9.b
    tbl z17.b, {z24.b}, z17.b
    lsl z10.b, z8.b, #4
    lsl z18.b, z16.b, #4
    add z9.b, z9.b, z12.b
    add z17.b, z17.b, z20.b
    add z10.b, z10.b, z13.b
    add z18.b, z18.b, z21.b
    zip1 z14.b, z10.b, z9.b
    zip1 z22.b, z18.b, z17.b
    zip2 z15.b, z10.b, z9.b
    zip2 z23.b, z18.b, z17.b
    zip1 z8.h, z29.h, z14.h
    zip1 z16.h, z29.h, z22.h
    zip2 z9.h, z29.h, z14.h
    zip2 z17.h, z29.h, z22.h
    zip1 z10.h, z29.h, z15.h
    zip1 z18.h, z29.h, z23.h
    zip2 z11.h, z29.h, z15.h
    zip2 z19.h, z29.h, z23.h
    fmla z0.s, p0/m, z8.s, z30.s
    fmla z4.s, p0/m, z16.s, z30.s
    fmla z1.s, p0/m, z9.s, z30.s
    fmla z5.s, p0/m, z17.s, z30.s
    fmla z2.s, p0/m, z10.s, z30.s
    fmla z6.s, p0/m, z18.s, z30.s
    fmla z3.s, p0/m, z11.s, z30.s
    fmla z7.s, p0/m, z19.s, z30.s
    add x0, x0, #128
    add x1, x1, #4
    subs x10, x10, #1
    b.ne 1b
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
.size w8_e4m3_f32, .-w8_e4m3_f32
.p2align 6
.global w8_e5m2_f16_native
.type w8_e5m2_f16_native, %function
w8_e5m2_f16_native:
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
    ld1b {z8.h}, p0/z, [x0, #0, mul vl]
    ld1b {z9.h}, p0/z, [x0, #1, mul vl]
    ld1b {z10.h}, p0/z, [x0, #2, mul vl]
    ld1b {z11.h}, p0/z, [x0, #3, mul vl]
    ld1b {z12.h}, p0/z, [x0, #4, mul vl]
    ld1b {z13.h}, p0/z, [x0, #5, mul vl]
    ld1b {z14.h}, p0/z, [x0, #6, mul vl]
    ld1b {z15.h}, p0/z, [x0, #7, mul vl]
    ld1rh {z24.h}, p0/z, [x1]
    lsl z8.h, z8.h, #8
    lsl z9.h, z9.h, #8
    lsl z10.h, z10.h, #8
    lsl z11.h, z11.h, #8
    lsl z12.h, z12.h, #8
    lsl z13.h, z13.h, #8
    lsl z14.h, z14.h, #8
    lsl z15.h, z15.h, #8
    mov x10, #63
.p2align 6
1:
    add x0, x0, #256
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
    lsl z16.h, z16.h, #8
    lsl z17.h, z17.h, #8
    lsl z18.h, z18.h, #8
    lsl z19.h, z19.h, #8
    lsl z20.h, z20.h, #8
    lsl z21.h, z21.h, #8
    lsl z22.h, z22.h, #8
    lsl z23.h, z23.h, #8
    fmla z0.h, p0/m, z8.h, z24.h
    fmla z1.h, p0/m, z9.h, z24.h
    fmla z2.h, p0/m, z10.h, z24.h
    fmla z3.h, p0/m, z11.h, z24.h
    fmla z4.h, p0/m, z12.h, z24.h
    fmla z5.h, p0/m, z13.h, z24.h
    fmla z6.h, p0/m, z14.h, z24.h
    fmla z7.h, p0/m, z15.h, z24.h
    add x0, x0, #256
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
    lsl z8.h, z8.h, #8
    lsl z9.h, z9.h, #8
    lsl z10.h, z10.h, #8
    lsl z11.h, z11.h, #8
    lsl z12.h, z12.h, #8
    lsl z13.h, z13.h, #8
    lsl z14.h, z14.h, #8
    lsl z15.h, z15.h, #8
    fmla z0.h, p0/m, z16.h, z25.h
    fmla z1.h, p0/m, z17.h, z25.h
    fmla z2.h, p0/m, z18.h, z25.h
    fmla z3.h, p0/m, z19.h, z25.h
    fmla z4.h, p0/m, z20.h, z25.h
    fmla z5.h, p0/m, z21.h, z25.h
    fmla z6.h, p0/m, z22.h, z25.h
    fmla z7.h, p0/m, z23.h, z25.h
    subs x10, x10, #1
    b.ne 1b
    add x0, x0, #256
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
    lsl z16.h, z16.h, #8
    lsl z17.h, z17.h, #8
    lsl z18.h, z18.h, #8
    lsl z19.h, z19.h, #8
    lsl z20.h, z20.h, #8
    lsl z21.h, z21.h, #8
    lsl z22.h, z22.h, #8
    lsl z23.h, z23.h, #8
    fmla z0.h, p0/m, z8.h, z24.h
    fmla z1.h, p0/m, z9.h, z24.h
    fmla z2.h, p0/m, z10.h, z24.h
    fmla z3.h, p0/m, z11.h, z24.h
    fmla z4.h, p0/m, z12.h, z24.h
    fmla z5.h, p0/m, z13.h, z24.h
    fmla z6.h, p0/m, z14.h, z24.h
    fmla z7.h, p0/m, z15.h, z24.h
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
.size w8_e5m2_f16_native, .-w8_e5m2_f16_native
.p2align 6
.global w8_e5m2_f32
.type w8_e5m2_f32, %function
w8_e5m2_f32:
    stp d8, d9, [sp, #-16]!
    stp d10, d11, [sp, #-16]!
    stp d12, d13, [sp, #-16]!
    stp d14, d15, [sp, #-16]!
    ptrue p0.s
    eor z0.d, z0.d, z0.d
    eor z1.d, z1.d, z1.d
    eor z2.d, z2.d, z2.d
    eor z3.d, z3.d, z3.d
    eor z4.d, z4.d, z4.d
    eor z5.d, z5.d, z5.d
    eor z6.d, z6.d, z6.d
    eor z7.d, z7.d, z7.d
    ld1b {z8.s}, p0/z, [x0, #0, mul vl]
    ld1b {z9.s}, p0/z, [x0, #1, mul vl]
    ld1b {z10.s}, p0/z, [x0, #2, mul vl]
    ld1b {z11.s}, p0/z, [x0, #3, mul vl]
    ld1b {z12.s}, p0/z, [x0, #4, mul vl]
    ld1b {z13.s}, p0/z, [x0, #5, mul vl]
    ld1b {z14.s}, p0/z, [x0, #6, mul vl]
    ld1b {z15.s}, p0/z, [x0, #7, mul vl]
    ld1rw {z24.s}, p0/z, [x1]
    lsl z8.s, z8.s, #8
    lsl z9.s, z9.s, #8
    lsl z10.s, z10.s, #8
    lsl z11.s, z11.s, #8
    lsl z12.s, z12.s, #8
    lsl z13.s, z13.s, #8
    lsl z14.s, z14.s, #8
    lsl z15.s, z15.s, #8
    fcvt z8.s, p0/m, z8.h
    fcvt z9.s, p0/m, z9.h
    fcvt z10.s, p0/m, z10.h
    fcvt z11.s, p0/m, z11.h
    fcvt z12.s, p0/m, z12.h
    fcvt z13.s, p0/m, z13.h
    fcvt z14.s, p0/m, z14.h
    fcvt z15.s, p0/m, z15.h
    mov x10, #63
.p2align 6
1:
    add x0, x0, #128
    add x1, x1, #4
    ld1b {z16.s}, p0/z, [x0, #0, mul vl]
    ld1b {z17.s}, p0/z, [x0, #1, mul vl]
    ld1b {z18.s}, p0/z, [x0, #2, mul vl]
    ld1b {z19.s}, p0/z, [x0, #3, mul vl]
    ld1b {z20.s}, p0/z, [x0, #4, mul vl]
    ld1b {z21.s}, p0/z, [x0, #5, mul vl]
    ld1b {z22.s}, p0/z, [x0, #6, mul vl]
    ld1b {z23.s}, p0/z, [x0, #7, mul vl]
    ld1rw {z25.s}, p0/z, [x1]
    lsl z16.s, z16.s, #8
    lsl z17.s, z17.s, #8
    lsl z18.s, z18.s, #8
    lsl z19.s, z19.s, #8
    lsl z20.s, z20.s, #8
    lsl z21.s, z21.s, #8
    lsl z22.s, z22.s, #8
    lsl z23.s, z23.s, #8
    fmla z0.s, p0/m, z8.s, z24.s
    fcvt z16.s, p0/m, z16.h
    fmla z1.s, p0/m, z9.s, z24.s
    fcvt z17.s, p0/m, z17.h
    fmla z2.s, p0/m, z10.s, z24.s
    fcvt z18.s, p0/m, z18.h
    fmla z3.s, p0/m, z11.s, z24.s
    fcvt z19.s, p0/m, z19.h
    fmla z4.s, p0/m, z12.s, z24.s
    fcvt z20.s, p0/m, z20.h
    fmla z5.s, p0/m, z13.s, z24.s
    fcvt z21.s, p0/m, z21.h
    fmla z6.s, p0/m, z14.s, z24.s
    fcvt z22.s, p0/m, z22.h
    fmla z7.s, p0/m, z15.s, z24.s
    fcvt z23.s, p0/m, z23.h
    add x0, x0, #128
    add x1, x1, #4
    ld1b {z8.s}, p0/z, [x0, #0, mul vl]
    ld1b {z9.s}, p0/z, [x0, #1, mul vl]
    ld1b {z10.s}, p0/z, [x0, #2, mul vl]
    ld1b {z11.s}, p0/z, [x0, #3, mul vl]
    ld1b {z12.s}, p0/z, [x0, #4, mul vl]
    ld1b {z13.s}, p0/z, [x0, #5, mul vl]
    ld1b {z14.s}, p0/z, [x0, #6, mul vl]
    ld1b {z15.s}, p0/z, [x0, #7, mul vl]
    ld1rw {z24.s}, p0/z, [x1]
    lsl z8.s, z8.s, #8
    lsl z9.s, z9.s, #8
    lsl z10.s, z10.s, #8
    lsl z11.s, z11.s, #8
    lsl z12.s, z12.s, #8
    lsl z13.s, z13.s, #8
    lsl z14.s, z14.s, #8
    lsl z15.s, z15.s, #8
    fmla z0.s, p0/m, z16.s, z25.s
    fcvt z8.s, p0/m, z8.h
    fmla z1.s, p0/m, z17.s, z25.s
    fcvt z9.s, p0/m, z9.h
    fmla z2.s, p0/m, z18.s, z25.s
    fcvt z10.s, p0/m, z10.h
    fmla z3.s, p0/m, z19.s, z25.s
    fcvt z11.s, p0/m, z11.h
    fmla z4.s, p0/m, z20.s, z25.s
    fcvt z12.s, p0/m, z12.h
    fmla z5.s, p0/m, z21.s, z25.s
    fcvt z13.s, p0/m, z13.h
    fmla z6.s, p0/m, z22.s, z25.s
    fcvt z14.s, p0/m, z14.h
    fmla z7.s, p0/m, z23.s, z25.s
    fcvt z15.s, p0/m, z15.h
    subs x10, x10, #1
    b.ne 1b
    add x0, x0, #128
    add x1, x1, #4
    ld1b {z16.s}, p0/z, [x0, #0, mul vl]
    ld1b {z17.s}, p0/z, [x0, #1, mul vl]
    ld1b {z18.s}, p0/z, [x0, #2, mul vl]
    ld1b {z19.s}, p0/z, [x0, #3, mul vl]
    ld1b {z20.s}, p0/z, [x0, #4, mul vl]
    ld1b {z21.s}, p0/z, [x0, #5, mul vl]
    ld1b {z22.s}, p0/z, [x0, #6, mul vl]
    ld1b {z23.s}, p0/z, [x0, #7, mul vl]
    ld1rw {z25.s}, p0/z, [x1]
    lsl z16.s, z16.s, #8
    lsl z17.s, z17.s, #8
    lsl z18.s, z18.s, #8
    lsl z19.s, z19.s, #8
    lsl z20.s, z20.s, #8
    lsl z21.s, z21.s, #8
    lsl z22.s, z22.s, #8
    lsl z23.s, z23.s, #8
    fmla z0.s, p0/m, z8.s, z24.s
    fcvt z16.s, p0/m, z16.h
    fmla z1.s, p0/m, z9.s, z24.s
    fcvt z17.s, p0/m, z17.h
    fmla z2.s, p0/m, z10.s, z24.s
    fcvt z18.s, p0/m, z18.h
    fmla z3.s, p0/m, z11.s, z24.s
    fcvt z19.s, p0/m, z19.h
    fmla z4.s, p0/m, z12.s, z24.s
    fcvt z20.s, p0/m, z20.h
    fmla z5.s, p0/m, z13.s, z24.s
    fcvt z21.s, p0/m, z21.h
    fmla z6.s, p0/m, z14.s, z24.s
    fcvt z22.s, p0/m, z22.h
    fmla z7.s, p0/m, z15.s, z24.s
    fcvt z23.s, p0/m, z23.h
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
.size w8_e5m2_f32, .-w8_e5m2_f32
.text
.p2align 6
.global w8_e5m2_f16
.type w8_e5m2_f16, %function
/* Exact rescaling for subnormal E5M2 operands: w*4 and a/4 have the
 * identical real product. Every finite nonzero activation must have half
 * exponent >=3, so division by four remains normal and exactly representable.
 * Unsafe activations or a nondefault FPCR take the native, unchanged path.
 * Table corrections map the three magnitudes to normal half bits without
 * executing floating-point arithmetic on a subnormal weight. */
w8_e5m2_f16:
    mrs x9, fpcr
    cbnz x9, w8_e5m2_f16_native
    ptrue p0.h
    ld1h {z0.h}, p0/z, [x1, #0, mul vl]
    and z0.h, z0.h, #0x7fff
    mov z1.h, #0x0c00
    cmpge p1.h, p0/z, z0.h, z1.h
    cmpeq p2.h, p0/z, z0.h, #0
    orr p1.b, p0/z, p1.b, p2.b
    cntp x9, p0, p1.h
    cmp x9, #32
    b.ne w8_e5m2_f16_native
    ld1h {z0.h}, p0/z, [x1, #1, mul vl]
    and z0.h, z0.h, #0x7fff
    mov z1.h, #0x0c00
    cmpge p1.h, p0/z, z0.h, z1.h
    cmpeq p2.h, p0/z, z0.h, #0
    orr p1.b, p0/z, p1.b, p2.b
    cntp x9, p0, p1.h
    cmp x9, #32
    b.ne w8_e5m2_f16_native
    ld1h {z0.h}, p0/z, [x1, #2, mul vl]
    and z0.h, z0.h, #0x7fff
    mov z1.h, #0x0c00
    cmpge p1.h, p0/z, z0.h, z1.h
    cmpeq p2.h, p0/z, z0.h, #0
    orr p1.b, p0/z, p1.b, p2.b
    cntp x9, p0, p1.h
    cmp x9, #32
    b.ne w8_e5m2_f16_native
    ld1h {z0.h}, p0/z, [x1, #3, mul vl]
    and z0.h, z0.h, #0x7fff
    mov z1.h, #0x0c00
    cmpge p1.h, p0/z, z0.h, z1.h
    cmpeq p2.h, p0/z, z0.h, #0
    orr p1.b, p0/z, p1.b, p2.b
    cntp x9, p0, p1.h
    cmp x9, #32
    b.ne w8_e5m2_f16_native
    stp d8, d9, [sp, #-16]!
    stp d10, d11, [sp, #-16]!
    stp d12, d13, [sp, #-16]!
    stp d14, d15, [sp, #-16]!
    eor z0.d, z0.d, z0.d
    eor z1.d, z1.d, z1.d
    eor z2.d, z2.d, z2.d
    eor z3.d, z3.d, z3.d
    eor z4.d, z4.d, z4.d
    eor z5.d, z5.d, z5.d
    eor z6.d, z6.d, z6.d
    eor z7.d, z7.d, z7.d
    fmov z26.h, #0.25
    mov z27.h, #127
    adr x9, .Le5_rescale
    ld1h {z28.h}, p0/z, [x9]
    mov x10, #128
.p2align 6
1:
    ld1rh {z24.h}, p0/z, [x1]
    mov z25.d, z24.d
    fmul z25.h, p0/m, z25.h, z26.h
    ld1b {z8.h}, p0/z, [x0, #0, mul vl]
    ld1b {z9.h}, p0/z, [x0, #1, mul vl]
    ld1b {z10.h}, p0/z, [x0, #2, mul vl]
    ld1b {z11.h}, p0/z, [x0, #3, mul vl]
    ld1b {z12.h}, p0/z, [x0, #4, mul vl]
    ld1b {z13.h}, p0/z, [x0, #5, mul vl]
    ld1b {z14.h}, p0/z, [x0, #6, mul vl]
    ld1b {z15.h}, p0/z, [x0, #7, mul vl]
    and z16.d, z8.d, z27.d
    cmplo p1.h, p0/z, z16.h, #4
    and z17.d, z9.d, z27.d
    cmplo p2.h, p0/z, z17.h, #4
    and z18.d, z10.d, z27.d
    cmplo p3.h, p0/z, z18.h, #4
    and z19.d, z11.d, z27.d
    cmplo p4.h, p0/z, z19.h, #4
    and z20.d, z12.d, z27.d
    cmplo p5.h, p0/z, z20.h, #4
    and z21.d, z13.d, z27.d
    cmplo p6.h, p0/z, z21.h, #4
    and z22.d, z14.d, z27.d
    cmplo p7.h, p0/z, z22.h, #4
    and z23.d, z15.d, z27.d
    cmplo p8.h, p0/z, z23.h, #4
    tbl z16.h, {z28.h}, z16.h
    lsl z8.h, z8.h, #8
    tbl z17.h, {z28.h}, z17.h
    lsl z9.h, z9.h, #8
    tbl z18.h, {z28.h}, z18.h
    lsl z10.h, z10.h, #8
    tbl z19.h, {z28.h}, z19.h
    lsl z11.h, z11.h, #8
    tbl z20.h, {z28.h}, z20.h
    lsl z12.h, z12.h, #8
    tbl z21.h, {z28.h}, z21.h
    lsl z13.h, z13.h, #8
    tbl z22.h, {z28.h}, z22.h
    lsl z14.h, z14.h, #8
    tbl z23.h, {z28.h}, z23.h
    lsl z15.h, z15.h, #8
    add z8.h, z8.h, z16.h
    sel z16.h, p1, z25.h, z24.h
    add z9.h, z9.h, z17.h
    sel z17.h, p2, z25.h, z24.h
    add z10.h, z10.h, z18.h
    sel z18.h, p3, z25.h, z24.h
    add z11.h, z11.h, z19.h
    sel z19.h, p4, z25.h, z24.h
    add z12.h, z12.h, z20.h
    sel z20.h, p5, z25.h, z24.h
    add z13.h, z13.h, z21.h
    sel z21.h, p6, z25.h, z24.h
    add z14.h, z14.h, z22.h
    sel z22.h, p7, z25.h, z24.h
    add z15.h, z15.h, z23.h
    sel z23.h, p8, z25.h, z24.h
    fmla z0.h, p0/m, z8.h, z16.h
    fmla z1.h, p0/m, z9.h, z17.h
    fmla z2.h, p0/m, z10.h, z18.h
    fmla z3.h, p0/m, z11.h, z19.h
    fmla z4.h, p0/m, z12.h, z20.h
    fmla z5.h, p0/m, z13.h, z21.h
    fmla z6.h, p0/m, z14.h, z22.h
    fmla z7.h, p0/m, z15.h, z23.h
    add x0, x0, #256
    add x1, x1, #2
    subs x10, x10, #1
    b.ne 1b
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
.size w8_e5m2_f16, .-w8_e5m2_f16
.section .rodata
.p2align 6
.Le5_rescale:
.hword 0,0x0300,0x0600,0x0700
.zero 56

.p2align 6
.Le4_base32:
.byte 60,61,62,63,64,65,66,67,188,189,190,191,192,193,194,195,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
.p2align 6
.Le4_hi32:
.byte 60,196,255,255,255,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
.p2align 6
.Le4_lo32:
.byte 208,0,240,96,144,192,208,224,240,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0


.p2align 6
.Le4_sub_f16:
.hword 0x1e80,0xe000,0xf780,0xfb00,0xfc80,0xfe00,0xfe80,0xff00,0xff80,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0
.section .note.GNU-stack,"",%progbits
