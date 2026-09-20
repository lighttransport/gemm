/* A64FX fixed 512-bit SVE W4A16 kernels. See fused_opt.h for layouts. */
.text
.macro SAVE
    stp d8, d9, [sp, #-16]!
    stp d10, d11, [sp, #-16]!
    stp d12, d13, [sp, #-16]!
    stp d14, d15, [sp, #-16]!
.endm
.macro RESTORE
    ldp d14, d15, [sp], #16
    ldp d12, d13, [sp], #16
    ldp d10, d11, [sp], #16
    ldp d8, d9, [sp], #16
.endm
.p2align 6
/* x0: 8192 packed bytes + 256 sum bytes, x1: biased digits, x2: INT32 output.
 * z0/z1, z2/z3, ... are low/high digit accumulator pairs.
 * z16-z23 hold both blocks' low/high nibble vectors independently.
 * At loop exit x0 addresses the sum trailer. Its widening loads consume
 * 32 source bytes each; the epilogue computes low + (high<<8) + (sum<<7). */
.global fused_i4_i16x8_full_sve
.type fused_i4_i16x8_full_sve, %function
fused_i4_i16x8_full_sve:
    SAVE
    ptrue p0.b
    eor z0.d, z0.d, z0.d
    eor z1.d, z1.d, z1.d
    eor z2.d, z2.d, z2.d
    eor z3.d, z3.d, z3.d
    eor z4.d, z4.d, z4.d
    eor z5.d, z5.d, z5.d
    eor z6.d, z6.d, z6.d
    eor z7.d, z7.d, z7.d
    eor z8.d, z8.d, z8.d
    eor z9.d, z9.d, z9.d
    eor z10.d, z10.d, z10.d
    eor z11.d, z11.d, z11.d
    eor z12.d, z12.d, z12.d
    eor z13.d, z13.d, z13.d
    eor z14.d, z14.d, z14.d
    eor z15.d, z15.d, z15.d
    add x3, x1, #128
    add x4, x1, #256
    add x5, x1, #384
    mov x10, #32
.p2align 6
1:
    ld1b {z16.b}, p0/z, [x0, #0, mul vl]
    ld1b {z17.b}, p0/z, [x0, #1, mul vl]
    ld1b {z18.b}, p0/z, [x0, #2, mul vl]
    ld1b {z19.b}, p0/z, [x0, #3, mul vl]
    asr z20.b, z16.b, #4
    lsl z16.b, z16.b, #4
    asr z16.b, z16.b, #4
    asr z21.b, z17.b, #4
    lsl z17.b, z17.b, #4
    asr z17.b, z17.b, #4
    asr z22.b, z18.b, #4
    lsl z18.b, z18.b, #4
    asr z18.b, z18.b, #4
    asr z23.b, z19.b, #4
    lsl z19.b, z19.b, #4
    asr z19.b, z19.b, #4
    ld1rw {z24.s}, p0/z, [x1]
    ld1rw {z25.s}, p0/z, [x3]
    ld1rw {z26.s}, p0/z, [x4]
    ld1rw {z27.s}, p0/z, [x5]
    sdot z0.s, z16.b, z24.b
    sdot z1.s, z16.b, z25.b
    sdot z2.s, z20.b, z24.b
    sdot z3.s, z20.b, z25.b
    sdot z4.s, z17.b, z24.b
    sdot z5.s, z17.b, z25.b
    sdot z6.s, z21.b, z24.b
    sdot z7.s, z21.b, z25.b
    sdot z8.s, z18.b, z26.b
    sdot z9.s, z18.b, z27.b
    sdot z10.s, z22.b, z26.b
    sdot z11.s, z22.b, z27.b
    sdot z12.s, z19.b, z26.b
    sdot z13.s, z19.b, z27.b
    sdot z14.s, z23.b, z26.b
    sdot z15.s, z23.b, z27.b
    add x0, x0, #256
    add x1, x1, #4
    add x3, x3, #4
    add x4, x4, #4
    add x5, x5, #4
    subs x10, x10, #1
    b.ne 1b
    ptrue p1.s
    ld1sh {z16.s}, p1/z, [x0, #0, mul vl]
    ld1sh {z17.s}, p1/z, [x0, #1, mul vl]
    ld1sh {z18.s}, p1/z, [x0, #2, mul vl]
    ld1sh {z19.s}, p1/z, [x0, #3, mul vl]
    ld1sh {z20.s}, p1/z, [x0, #4, mul vl]
    ld1sh {z21.s}, p1/z, [x0, #5, mul vl]
    ld1sh {z22.s}, p1/z, [x0, #6, mul vl]
    ld1sh {z23.s}, p1/z, [x0, #7, mul vl]
    lsl z1.s, z1.s, #8
    lsl z16.s, z16.s, #7
    add z0.s, z0.s, z1.s
    add z0.s, z0.s, z16.s
    lsl z3.s, z3.s, #8
    lsl z17.s, z17.s, #7
    add z2.s, z2.s, z3.s
    add z2.s, z2.s, z17.s
    lsl z5.s, z5.s, #8
    lsl z18.s, z18.s, #7
    add z4.s, z4.s, z5.s
    add z4.s, z4.s, z18.s
    lsl z7.s, z7.s, #8
    lsl z19.s, z19.s, #7
    add z6.s, z6.s, z7.s
    add z6.s, z6.s, z19.s
    lsl z9.s, z9.s, #8
    lsl z20.s, z20.s, #7
    add z8.s, z8.s, z9.s
    add z8.s, z8.s, z20.s
    lsl z11.s, z11.s, #8
    lsl z21.s, z21.s, #7
    add z10.s, z10.s, z11.s
    add z10.s, z10.s, z21.s
    lsl z13.s, z13.s, #8
    lsl z22.s, z22.s, #7
    add z12.s, z12.s, z13.s
    add z12.s, z12.s, z22.s
    lsl z15.s, z15.s, #8
    lsl z23.s, z23.s, #7
    add z14.s, z14.s, z15.s
    add z14.s, z14.s, z23.s
    st1w {z0.s}, p1, [x2, #0, mul vl]
    st1w {z2.s}, p1, [x2, #1, mul vl]
    st1w {z4.s}, p1, [x2, #2, mul vl]
    st1w {z6.s}, p1, [x2, #3, mul vl]
    st1w {z8.s}, p1, [x2, #4, mul vl]
    st1w {z10.s}, p1, [x2, #5, mul vl]
    st1w {z12.s}, p1, [x2, #6, mul vl]
    st1w {z14.s}, p1, [x2, #7, mul vl]
    RESTORE
    ret
.size fused_i4_i16x8_full_sve, .-fused_i4_i16x8_full_sve
.p2align 6
/* Same full-range ABI; FP4 table entries are the exact doubled E2M1 lattice.
 * The resident register mask avoids an immediate-mask copy/dependency. */
.global fused_fp4_i16x8_full_sve
.type fused_fp4_i16x8_full_sve, %function
fused_fp4_i16x8_full_sve:
    SAVE
    ptrue p0.b
    adr x9, .Lfp4_i8
    ld1b {z31.b}, p0/z, [x9]
    mov z30.b, #15
    eor z0.d, z0.d, z0.d
    eor z1.d, z1.d, z1.d
    eor z2.d, z2.d, z2.d
    eor z3.d, z3.d, z3.d
    eor z4.d, z4.d, z4.d
    eor z5.d, z5.d, z5.d
    eor z6.d, z6.d, z6.d
    eor z7.d, z7.d, z7.d
    eor z8.d, z8.d, z8.d
    eor z9.d, z9.d, z9.d
    eor z10.d, z10.d, z10.d
    eor z11.d, z11.d, z11.d
    eor z12.d, z12.d, z12.d
    eor z13.d, z13.d, z13.d
    eor z14.d, z14.d, z14.d
    eor z15.d, z15.d, z15.d
    add x3, x1, #128
    add x4, x1, #256
    add x5, x1, #384
    mov x10, #32
.p2align 6
1:
    ld1b {z16.b}, p0/z, [x0, #0, mul vl]
    ld1b {z17.b}, p0/z, [x0, #1, mul vl]
    ld1b {z18.b}, p0/z, [x0, #2, mul vl]
    ld1b {z19.b}, p0/z, [x0, #3, mul vl]
    lsr z20.b, z16.b, #4
    and z16.d, z16.d, z30.d
    lsr z21.b, z17.b, #4
    and z17.d, z17.d, z30.d
    lsr z22.b, z18.b, #4
    and z18.d, z18.d, z30.d
    lsr z23.b, z19.b, #4
    and z19.d, z19.d, z30.d
    tbl z16.b, {z31.b}, z16.b
    tbl z17.b, {z31.b}, z17.b
    tbl z18.b, {z31.b}, z18.b
    tbl z19.b, {z31.b}, z19.b
    tbl z20.b, {z31.b}, z20.b
    tbl z21.b, {z31.b}, z21.b
    tbl z22.b, {z31.b}, z22.b
    tbl z23.b, {z31.b}, z23.b
    ld1rw {z24.s}, p0/z, [x1]
    ld1rw {z25.s}, p0/z, [x3]
    ld1rw {z26.s}, p0/z, [x4]
    ld1rw {z27.s}, p0/z, [x5]
    sdot z0.s, z16.b, z24.b
    sdot z1.s, z16.b, z25.b
    sdot z2.s, z20.b, z24.b
    sdot z3.s, z20.b, z25.b
    sdot z4.s, z17.b, z24.b
    sdot z5.s, z17.b, z25.b
    sdot z6.s, z21.b, z24.b
    sdot z7.s, z21.b, z25.b
    sdot z8.s, z18.b, z26.b
    sdot z9.s, z18.b, z27.b
    sdot z10.s, z22.b, z26.b
    sdot z11.s, z22.b, z27.b
    sdot z12.s, z19.b, z26.b
    sdot z13.s, z19.b, z27.b
    sdot z14.s, z23.b, z26.b
    sdot z15.s, z23.b, z27.b
    add x0, x0, #256
    add x1, x1, #4
    add x3, x3, #4
    add x4, x4, #4
    add x5, x5, #4
    subs x10, x10, #1
    b.ne 1b
    ptrue p1.s
    ld1sh {z16.s}, p1/z, [x0, #0, mul vl]
    ld1sh {z17.s}, p1/z, [x0, #1, mul vl]
    ld1sh {z18.s}, p1/z, [x0, #2, mul vl]
    ld1sh {z19.s}, p1/z, [x0, #3, mul vl]
    ld1sh {z20.s}, p1/z, [x0, #4, mul vl]
    ld1sh {z21.s}, p1/z, [x0, #5, mul vl]
    ld1sh {z22.s}, p1/z, [x0, #6, mul vl]
    ld1sh {z23.s}, p1/z, [x0, #7, mul vl]
    lsl z1.s, z1.s, #8
    lsl z16.s, z16.s, #7
    add z0.s, z0.s, z1.s
    add z0.s, z0.s, z16.s
    lsl z3.s, z3.s, #8
    lsl z17.s, z17.s, #7
    add z2.s, z2.s, z3.s
    add z2.s, z2.s, z17.s
    lsl z5.s, z5.s, #8
    lsl z18.s, z18.s, #7
    add z4.s, z4.s, z5.s
    add z4.s, z4.s, z18.s
    lsl z7.s, z7.s, #8
    lsl z19.s, z19.s, #7
    add z6.s, z6.s, z7.s
    add z6.s, z6.s, z19.s
    lsl z9.s, z9.s, #8
    lsl z20.s, z20.s, #7
    add z8.s, z8.s, z9.s
    add z8.s, z8.s, z20.s
    lsl z11.s, z11.s, #8
    lsl z21.s, z21.s, #7
    add z10.s, z10.s, z11.s
    add z10.s, z10.s, z21.s
    lsl z13.s, z13.s, #8
    lsl z22.s, z22.s, #7
    add z12.s, z12.s, z13.s
    add z12.s, z12.s, z22.s
    lsl z15.s, z15.s, #8
    lsl z23.s, z23.s, #7
    add z14.s, z14.s, z15.s
    add z14.s, z14.s, z23.s
    st1w {z0.s}, p1, [x2, #0, mul vl]
    st1w {z2.s}, p1, [x2, #1, mul vl]
    st1w {z4.s}, p1, [x2, #2, mul vl]
    st1w {z6.s}, p1, [x2, #3, mul vl]
    st1w {z8.s}, p1, [x2, #4, mul vl]
    st1w {z10.s}, p1, [x2, #5, mul vl]
    st1w {z12.s}, p1, [x2, #6, mul vl]
    st1w {z14.s}, p1, [x2, #7, mul vl]
    RESTORE
    ret
.size fused_fp4_i16x8_full_sve, .-fused_fp4_i16x8_full_sve
.p2align 6
/* FP16: four blocks, eight output vectors, one K step per loop.
 * LD1B .h performs widening in the load; TBL .h returns final FP16 bits.
 * z30 is the halfword nibble mask, z31 the table, z24-z27 activations. */
.global fused_i4_f16_opt1_sve
.type fused_i4_f16_opt1_sve, %function
fused_i4_f16_opt1_sve:
    SAVE
    ptrue p0.h
    adr x9, .Li4_f16
    ld1h {z31.h}, p0/z, [x9]
    mov z30.h, #15
    eor z0.d, z0.d, z0.d
    eor z1.d, z1.d, z1.d
    eor z2.d, z2.d, z2.d
    eor z3.d, z3.d, z3.d
    eor z4.d, z4.d, z4.d
    eor z5.d, z5.d, z5.d
    eor z6.d, z6.d, z6.d
    eor z7.d, z7.d, z7.d
    add x3, x1, #256
    add x4, x1, #512
    add x5, x1, #768
    mov x10, #128
.p2align 6
1:
    ld1b {z8.h}, p0/z, [x0, #0, mul vl]
    ld1b {z10.h}, p0/z, [x0, #1, mul vl]
    ld1b {z12.h}, p0/z, [x0, #2, mul vl]
    ld1b {z14.h}, p0/z, [x0, #3, mul vl]
    ld1rh {z24.h}, p0/z, [x1]
    ld1rh {z25.h}, p0/z, [x3]
    ld1rh {z26.h}, p0/z, [x4]
    ld1rh {z27.h}, p0/z, [x5]
    lsr z9.h, z8.h, #4
    and z8.d, z8.d, z30.d
    lsr z11.h, z10.h, #4
    and z10.d, z10.d, z30.d
    lsr z13.h, z12.h, #4
    and z12.d, z12.d, z30.d
    lsr z15.h, z14.h, #4
    and z14.d, z14.d, z30.d
    tbl z8.h, {z31.h}, z8.h
    tbl z9.h, {z31.h}, z9.h
    tbl z10.h, {z31.h}, z10.h
    tbl z11.h, {z31.h}, z11.h
    tbl z12.h, {z31.h}, z12.h
    tbl z13.h, {z31.h}, z13.h
    tbl z14.h, {z31.h}, z14.h
    tbl z15.h, {z31.h}, z15.h
    fmla z0.h, p0/m, z8.h, z24.h
    fmla z1.h, p0/m, z9.h, z24.h
    fmla z2.h, p0/m, z10.h, z25.h
    fmla z3.h, p0/m, z11.h, z25.h
    fmla z4.h, p0/m, z12.h, z26.h
    fmla z5.h, p0/m, z13.h, z26.h
    fmla z6.h, p0/m, z14.h, z27.h
    fmla z7.h, p0/m, z15.h, z27.h
    add x0, x0, #128
    add x1, x1, #2
    add x3, x3, #2
    add x4, x4, #2
    add x5, x5, #2
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
    RESTORE
    ret
.size fused_i4_f16_opt1_sve, .-fused_i4_f16_opt1_sve
.p2align 6
/* Selected FP16 schedule: two K steps with distinct z8-z15 / z16-z23
 * temporaries. Each accumulator sees the same ascending-K FMA order as opt1.
 * There are no partial sums or floating-point reassociation. */
.global fused_i4_f16_opt2_sve
.type fused_i4_f16_opt2_sve, %function
fused_i4_f16_opt2_sve:
    SAVE
    ptrue p0.h
    adr x9, .Li4_f16
    ld1h {z31.h}, p0/z, [x9]
    mov z30.h, #15
    eor z0.d, z0.d, z0.d
    eor z1.d, z1.d, z1.d
    eor z2.d, z2.d, z2.d
    eor z3.d, z3.d, z3.d
    eor z4.d, z4.d, z4.d
    eor z5.d, z5.d, z5.d
    eor z6.d, z6.d, z6.d
    eor z7.d, z7.d, z7.d
    add x3, x1, #256
    add x4, x1, #512
    add x5, x1, #768
    mov x10, #64
.p2align 6
1:
    ld1b {z8.h}, p0/z, [x0, #0, mul vl]
    ld1b {z10.h}, p0/z, [x0, #1, mul vl]
    ld1b {z12.h}, p0/z, [x0, #2, mul vl]
    ld1b {z14.h}, p0/z, [x0, #3, mul vl]
    ld1rh {z24.h}, p0/z, [x1]
    ld1rh {z25.h}, p0/z, [x3]
    ld1rh {z26.h}, p0/z, [x4]
    ld1rh {z27.h}, p0/z, [x5]
    lsr z9.h, z8.h, #4
    and z8.d, z8.d, z30.d
    lsr z11.h, z10.h, #4
    and z10.d, z10.d, z30.d
    lsr z13.h, z12.h, #4
    and z12.d, z12.d, z30.d
    lsr z15.h, z14.h, #4
    and z14.d, z14.d, z30.d
    tbl z8.h, {z31.h}, z8.h
    tbl z9.h, {z31.h}, z9.h
    tbl z10.h, {z31.h}, z10.h
    tbl z11.h, {z31.h}, z11.h
    tbl z12.h, {z31.h}, z12.h
    tbl z13.h, {z31.h}, z13.h
    tbl z14.h, {z31.h}, z14.h
    tbl z15.h, {z31.h}, z15.h
    fmla z0.h, p0/m, z8.h, z24.h
    fmla z1.h, p0/m, z9.h, z24.h
    fmla z2.h, p0/m, z10.h, z25.h
    fmla z3.h, p0/m, z11.h, z25.h
    fmla z4.h, p0/m, z12.h, z26.h
    fmla z5.h, p0/m, z13.h, z26.h
    fmla z6.h, p0/m, z14.h, z27.h
    fmla z7.h, p0/m, z15.h, z27.h
    add x0, x0, #128
    add x1, x1, #2
    add x3, x3, #2
    add x4, x4, #2
    add x5, x5, #2
    ld1b {z16.h}, p0/z, [x0, #0, mul vl]
    ld1b {z18.h}, p0/z, [x0, #1, mul vl]
    ld1b {z20.h}, p0/z, [x0, #2, mul vl]
    ld1b {z22.h}, p0/z, [x0, #3, mul vl]
    ld1rh {z24.h}, p0/z, [x1]
    ld1rh {z25.h}, p0/z, [x3]
    ld1rh {z26.h}, p0/z, [x4]
    ld1rh {z27.h}, p0/z, [x5]
    lsr z17.h, z16.h, #4
    and z16.d, z16.d, z30.d
    lsr z19.h, z18.h, #4
    and z18.d, z18.d, z30.d
    lsr z21.h, z20.h, #4
    and z20.d, z20.d, z30.d
    lsr z23.h, z22.h, #4
    and z22.d, z22.d, z30.d
    tbl z16.h, {z31.h}, z16.h
    tbl z17.h, {z31.h}, z17.h
    tbl z18.h, {z31.h}, z18.h
    tbl z19.h, {z31.h}, z19.h
    tbl z20.h, {z31.h}, z20.h
    tbl z21.h, {z31.h}, z21.h
    tbl z22.h, {z31.h}, z22.h
    tbl z23.h, {z31.h}, z23.h
    fmla z0.h, p0/m, z16.h, z24.h
    fmla z1.h, p0/m, z17.h, z24.h
    fmla z2.h, p0/m, z18.h, z25.h
    fmla z3.h, p0/m, z19.h, z25.h
    fmla z4.h, p0/m, z20.h, z26.h
    fmla z5.h, p0/m, z21.h, z26.h
    fmla z6.h, p0/m, z22.h, z27.h
    fmla z7.h, p0/m, z23.h, z27.h
    add x0, x0, #128
    add x1, x1, #2
    add x3, x3, #2
    add x4, x4, #2
    add x5, x5, #2
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
    RESTORE
    ret
.size fused_i4_f16_opt2_sve, .-fused_i4_f16_opt2_sve
.p2align 6
.global fused_fp4_f16_opt1_sve
.type fused_fp4_f16_opt1_sve, %function
fused_fp4_f16_opt1_sve:
    SAVE
    ptrue p0.h
    adr x9, .Lfp4_f16
    ld1h {z31.h}, p0/z, [x9]
    mov z30.h, #15
    eor z0.d, z0.d, z0.d
    eor z1.d, z1.d, z1.d
    eor z2.d, z2.d, z2.d
    eor z3.d, z3.d, z3.d
    eor z4.d, z4.d, z4.d
    eor z5.d, z5.d, z5.d
    eor z6.d, z6.d, z6.d
    eor z7.d, z7.d, z7.d
    add x3, x1, #256
    add x4, x1, #512
    add x5, x1, #768
    mov x10, #128
.p2align 6
1:
    ld1b {z8.h}, p0/z, [x0, #0, mul vl]
    ld1b {z10.h}, p0/z, [x0, #1, mul vl]
    ld1b {z12.h}, p0/z, [x0, #2, mul vl]
    ld1b {z14.h}, p0/z, [x0, #3, mul vl]
    ld1rh {z24.h}, p0/z, [x1]
    ld1rh {z25.h}, p0/z, [x3]
    ld1rh {z26.h}, p0/z, [x4]
    ld1rh {z27.h}, p0/z, [x5]
    lsr z9.h, z8.h, #4
    and z8.d, z8.d, z30.d
    lsr z11.h, z10.h, #4
    and z10.d, z10.d, z30.d
    lsr z13.h, z12.h, #4
    and z12.d, z12.d, z30.d
    lsr z15.h, z14.h, #4
    and z14.d, z14.d, z30.d
    tbl z8.h, {z31.h}, z8.h
    tbl z9.h, {z31.h}, z9.h
    tbl z10.h, {z31.h}, z10.h
    tbl z11.h, {z31.h}, z11.h
    tbl z12.h, {z31.h}, z12.h
    tbl z13.h, {z31.h}, z13.h
    tbl z14.h, {z31.h}, z14.h
    tbl z15.h, {z31.h}, z15.h
    fmla z0.h, p0/m, z8.h, z24.h
    fmla z1.h, p0/m, z9.h, z24.h
    fmla z2.h, p0/m, z10.h, z25.h
    fmla z3.h, p0/m, z11.h, z25.h
    fmla z4.h, p0/m, z12.h, z26.h
    fmla z5.h, p0/m, z13.h, z26.h
    fmla z6.h, p0/m, z14.h, z27.h
    fmla z7.h, p0/m, z15.h, z27.h
    add x0, x0, #128
    add x1, x1, #2
    add x3, x3, #2
    add x4, x4, #2
    add x5, x5, #2
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
    RESTORE
    ret
.size fused_fp4_f16_opt1_sve, .-fused_fp4_f16_opt1_sve
.p2align 6
.global fused_fp4_f16_opt2_sve
.type fused_fp4_f16_opt2_sve, %function
fused_fp4_f16_opt2_sve:
    SAVE
    ptrue p0.h
    adr x9, .Lfp4_f16
    ld1h {z31.h}, p0/z, [x9]
    mov z30.h, #15
    eor z0.d, z0.d, z0.d
    eor z1.d, z1.d, z1.d
    eor z2.d, z2.d, z2.d
    eor z3.d, z3.d, z3.d
    eor z4.d, z4.d, z4.d
    eor z5.d, z5.d, z5.d
    eor z6.d, z6.d, z6.d
    eor z7.d, z7.d, z7.d
    add x3, x1, #256
    add x4, x1, #512
    add x5, x1, #768
    mov x10, #64
.p2align 6
1:
    ld1b {z8.h}, p0/z, [x0, #0, mul vl]
    ld1b {z10.h}, p0/z, [x0, #1, mul vl]
    ld1b {z12.h}, p0/z, [x0, #2, mul vl]
    ld1b {z14.h}, p0/z, [x0, #3, mul vl]
    ld1rh {z24.h}, p0/z, [x1]
    ld1rh {z25.h}, p0/z, [x3]
    ld1rh {z26.h}, p0/z, [x4]
    ld1rh {z27.h}, p0/z, [x5]
    lsr z9.h, z8.h, #4
    and z8.d, z8.d, z30.d
    lsr z11.h, z10.h, #4
    and z10.d, z10.d, z30.d
    lsr z13.h, z12.h, #4
    and z12.d, z12.d, z30.d
    lsr z15.h, z14.h, #4
    and z14.d, z14.d, z30.d
    tbl z8.h, {z31.h}, z8.h
    tbl z9.h, {z31.h}, z9.h
    tbl z10.h, {z31.h}, z10.h
    tbl z11.h, {z31.h}, z11.h
    tbl z12.h, {z31.h}, z12.h
    tbl z13.h, {z31.h}, z13.h
    tbl z14.h, {z31.h}, z14.h
    tbl z15.h, {z31.h}, z15.h
    fmla z0.h, p0/m, z8.h, z24.h
    fmla z1.h, p0/m, z9.h, z24.h
    fmla z2.h, p0/m, z10.h, z25.h
    fmla z3.h, p0/m, z11.h, z25.h
    fmla z4.h, p0/m, z12.h, z26.h
    fmla z5.h, p0/m, z13.h, z26.h
    fmla z6.h, p0/m, z14.h, z27.h
    fmla z7.h, p0/m, z15.h, z27.h
    add x0, x0, #128
    add x1, x1, #2
    add x3, x3, #2
    add x4, x4, #2
    add x5, x5, #2
    ld1b {z16.h}, p0/z, [x0, #0, mul vl]
    ld1b {z18.h}, p0/z, [x0, #1, mul vl]
    ld1b {z20.h}, p0/z, [x0, #2, mul vl]
    ld1b {z22.h}, p0/z, [x0, #3, mul vl]
    ld1rh {z24.h}, p0/z, [x1]
    ld1rh {z25.h}, p0/z, [x3]
    ld1rh {z26.h}, p0/z, [x4]
    ld1rh {z27.h}, p0/z, [x5]
    lsr z17.h, z16.h, #4
    and z16.d, z16.d, z30.d
    lsr z19.h, z18.h, #4
    and z18.d, z18.d, z30.d
    lsr z21.h, z20.h, #4
    and z20.d, z20.d, z30.d
    lsr z23.h, z22.h, #4
    and z22.d, z22.d, z30.d
    tbl z16.h, {z31.h}, z16.h
    tbl z17.h, {z31.h}, z17.h
    tbl z18.h, {z31.h}, z18.h
    tbl z19.h, {z31.h}, z19.h
    tbl z20.h, {z31.h}, z20.h
    tbl z21.h, {z31.h}, z21.h
    tbl z22.h, {z31.h}, z22.h
    tbl z23.h, {z31.h}, z23.h
    fmla z0.h, p0/m, z16.h, z24.h
    fmla z1.h, p0/m, z17.h, z24.h
    fmla z2.h, p0/m, z18.h, z25.h
    fmla z3.h, p0/m, z19.h, z25.h
    fmla z4.h, p0/m, z20.h, z26.h
    fmla z5.h, p0/m, z21.h, z26.h
    fmla z6.h, p0/m, z22.h, z27.h
    fmla z7.h, p0/m, z23.h, z27.h
    add x0, x0, #128
    add x1, x1, #2
    add x3, x3, #2
    add x4, x4, #2
    add x5, x5, #2
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
    RESTORE
    ret
.size fused_fp4_f16_opt2_sve, .-fused_fp4_f16_opt2_sve
.p2align 6
/* Native halfword SDOT: low nibbles remain signed byte values times 16.
 * The low-nibble INT64 accumulators are divided exactly once after K=128.
 * The high nibbles retain their original scale. No sum metadata is needed. */
.global fused_i4_i16_opt_sve
.type fused_i4_i16_opt_sve, %function
fused_i4_i16_opt_sve:
    SAVE
    ptrue p0.b
    ptrue p1.d
    add x3, x1, #256
    eor z0.d, z0.d, z0.d
    eor z1.d, z1.d, z1.d
    eor z2.d, z2.d, z2.d
    eor z3.d, z3.d, z3.d
    eor z4.d, z4.d, z4.d
    eor z5.d, z5.d, z5.d
    eor z6.d, z6.d, z6.d
    eor z7.d, z7.d, z7.d
    eor z8.d, z8.d, z8.d
    eor z9.d, z9.d, z9.d
    eor z10.d, z10.d, z10.d
    eor z11.d, z11.d, z11.d
    eor z12.d, z12.d, z12.d
    eor z13.d, z13.d, z13.d
    eor z14.d, z14.d, z14.d
    eor z15.d, z15.d, z15.d
    mov x10, #32
.p2align 6
1:
    ld1b {z16.b}, p0/z, [x0, #0, mul vl]
    ld1b {z17.b}, p0/z, [x0, #1, mul vl]
    ld1b {z18.b}, p0/z, [x0, #2, mul vl]
    ld1b {z19.b}, p0/z, [x0, #3, mul vl]
    lsl z20.b, z16.b, #4
    asr z16.b, z16.b, #4
    lsl z21.b, z17.b, #4
    asr z17.b, z17.b, #4
    lsl z22.b, z18.b, #4
    asr z18.b, z18.b, #4
    lsl z23.b, z19.b, #4
    asr z19.b, z19.b, #4
    ld1rd {z30.d}, p0/z, [x1]
    ld1rd {z31.d}, p0/z, [x3]
    sunpklo z24.h, z20.b
    sunpkhi z25.h, z20.b
    sunpklo z26.h, z16.b
    sunpkhi z27.h, z16.b
    sunpklo z28.h, z21.b
    sunpkhi z29.h, z21.b
    sdot z0.d, z24.h, z30.h
    sdot z1.d, z25.h, z30.h
    sdot z2.d, z26.h, z30.h
    sdot z3.d, z27.h, z30.h
    sdot z4.d, z28.h, z30.h
    sdot z5.d, z29.h, z30.h
    sunpklo z24.h, z17.b
    sunpkhi z25.h, z17.b
    sunpklo z26.h, z22.b
    sunpkhi z27.h, z22.b
    sunpklo z28.h, z18.b
    sunpkhi z29.h, z18.b
    sdot z6.d, z24.h, z30.h
    sdot z7.d, z25.h, z30.h
    sdot z8.d, z26.h, z31.h
    sdot z9.d, z27.h, z31.h
    sdot z10.d, z28.h, z31.h
    sdot z11.d, z29.h, z31.h
    sunpklo z24.h, z23.b
    sunpkhi z25.h, z23.b
    sunpklo z26.h, z19.b
    sunpkhi z27.h, z19.b
    sdot z12.d, z24.h, z31.h
    sdot z13.d, z25.h, z31.h
    sdot z14.d, z26.h, z31.h
    sdot z15.d, z27.h, z31.h
    add x0, x0, #256
    add x1, x1, #8
    add x3, x3, #8
    subs x10, x10, #1
    b.ne 1b
    asr z0.d, z0.d, #4
    asr z1.d, z1.d, #4
    asr z4.d, z4.d, #4
    asr z5.d, z5.d, #4
    asr z8.d, z8.d, #4
    asr z9.d, z9.d, #4
    asr z12.d, z12.d, #4
    asr z13.d, z13.d, #4
    st1d {z0.d}, p1, [x2, #0, mul vl]
    st1d {z1.d}, p1, [x2, #1, mul vl]
    st1d {z2.d}, p1, [x2, #2, mul vl]
    st1d {z3.d}, p1, [x2, #3, mul vl]
    st1d {z4.d}, p1, [x2, #4, mul vl]
    st1d {z5.d}, p1, [x2, #5, mul vl]
    st1d {z6.d}, p1, [x2, #6, mul vl]
    st1d {z7.d}, p1, [x2, #7, mul vl]
    add x2, x2, #512
    st1d {z8.d}, p1, [x2, #0, mul vl]
    st1d {z9.d}, p1, [x2, #1, mul vl]
    st1d {z10.d}, p1, [x2, #2, mul vl]
    st1d {z11.d}, p1, [x2, #3, mul vl]
    st1d {z12.d}, p1, [x2, #4, mul vl]
    st1d {z13.d}, p1, [x2, #5, mul vl]
    st1d {z14.d}, p1, [x2, #6, mul vl]
    st1d {z15.d}, p1, [x2, #7, mul vl]
    RESTORE
    ret
.size fused_i4_i16_opt_sve, .-fused_i4_i16_opt_sve
.p2align 6
/* Native FP4: LD1B .h + TBL .h replaces byte lookup followed by SUNPK.
 * Each batch alternates the two blocks to expose independent dot chains. */
.global fused_fp4_i16_opt_sve
.type fused_fp4_i16_opt_sve, %function
fused_fp4_i16_opt_sve:
    SAVE
    ptrue p0.b
    ptrue p1.d
    add x3, x1, #256
    adr x9, .Lfp4_i16
    ld1h {z31.h}, p0/z, [x9]
    mov z30.h, #15
    eor z0.d, z0.d, z0.d
    eor z1.d, z1.d, z1.d
    eor z2.d, z2.d, z2.d
    eor z3.d, z3.d, z3.d
    eor z4.d, z4.d, z4.d
    eor z5.d, z5.d, z5.d
    eor z6.d, z6.d, z6.d
    eor z7.d, z7.d, z7.d
    eor z8.d, z8.d, z8.d
    eor z9.d, z9.d, z9.d
    eor z10.d, z10.d, z10.d
    eor z11.d, z11.d, z11.d
    eor z12.d, z12.d, z12.d
    eor z13.d, z13.d, z13.d
    eor z14.d, z14.d, z14.d
    eor z15.d, z15.d, z15.d
    mov x10, #32
.p2align 6
1:
    ld1rd {z28.d}, p0/z, [x1]
    ld1rd {z29.d}, p0/z, [x3]
    ld1b {z16.h}, p0/z, [x0, #0, mul vl]
    ld1b {z17.h}, p0/z, [x0, #4, mul vl]
    ld1b {z18.h}, p0/z, [x0, #1, mul vl]
    ld1b {z19.h}, p0/z, [x0, #5, mul vl]
    lsr z20.h, z16.h, #4
    and z16.d, z16.d, z30.d
    lsr z21.h, z17.h, #4
    and z17.d, z17.d, z30.d
    lsr z22.h, z18.h, #4
    and z18.d, z18.d, z30.d
    lsr z23.h, z19.h, #4
    and z19.d, z19.d, z30.d
    tbl z16.h, {z31.h}, z16.h
    tbl z17.h, {z31.h}, z17.h
    tbl z18.h, {z31.h}, z18.h
    tbl z19.h, {z31.h}, z19.h
    tbl z20.h, {z31.h}, z20.h
    tbl z21.h, {z31.h}, z21.h
    tbl z22.h, {z31.h}, z22.h
    tbl z23.h, {z31.h}, z23.h
    sdot z0.d, z16.h, z28.h
    sdot z2.d, z20.h, z28.h
    sdot z8.d, z17.h, z29.h
    sdot z10.d, z21.h, z29.h
    sdot z1.d, z18.h, z28.h
    sdot z3.d, z22.h, z28.h
    sdot z9.d, z19.h, z29.h
    sdot z11.d, z23.h, z29.h
    ld1b {z16.h}, p0/z, [x0, #2, mul vl]
    ld1b {z17.h}, p0/z, [x0, #6, mul vl]
    ld1b {z18.h}, p0/z, [x0, #3, mul vl]
    ld1b {z19.h}, p0/z, [x0, #7, mul vl]
    lsr z20.h, z16.h, #4
    and z16.d, z16.d, z30.d
    lsr z21.h, z17.h, #4
    and z17.d, z17.d, z30.d
    lsr z22.h, z18.h, #4
    and z18.d, z18.d, z30.d
    lsr z23.h, z19.h, #4
    and z19.d, z19.d, z30.d
    tbl z16.h, {z31.h}, z16.h
    tbl z17.h, {z31.h}, z17.h
    tbl z18.h, {z31.h}, z18.h
    tbl z19.h, {z31.h}, z19.h
    tbl z20.h, {z31.h}, z20.h
    tbl z21.h, {z31.h}, z21.h
    tbl z22.h, {z31.h}, z22.h
    tbl z23.h, {z31.h}, z23.h
    sdot z4.d, z16.h, z28.h
    sdot z6.d, z20.h, z28.h
    sdot z12.d, z17.h, z29.h
    sdot z14.d, z21.h, z29.h
    sdot z5.d, z18.h, z28.h
    sdot z7.d, z22.h, z28.h
    sdot z13.d, z19.h, z29.h
    sdot z15.d, z23.h, z29.h
    add x0, x0, #256
    add x1, x1, #8
    add x3, x3, #8
    subs x10, x10, #1
    b.ne 1b
    st1d {z0.d}, p1, [x2, #0, mul vl]
    st1d {z1.d}, p1, [x2, #1, mul vl]
    st1d {z2.d}, p1, [x2, #2, mul vl]
    st1d {z3.d}, p1, [x2, #3, mul vl]
    st1d {z4.d}, p1, [x2, #4, mul vl]
    st1d {z5.d}, p1, [x2, #5, mul vl]
    st1d {z6.d}, p1, [x2, #6, mul vl]
    st1d {z7.d}, p1, [x2, #7, mul vl]
    add x2, x2, #512
    st1d {z8.d}, p1, [x2, #0, mul vl]
    st1d {z9.d}, p1, [x2, #1, mul vl]
    st1d {z10.d}, p1, [x2, #2, mul vl]
    st1d {z11.d}, p1, [x2, #3, mul vl]
    st1d {z12.d}, p1, [x2, #4, mul vl]
    st1d {z13.d}, p1, [x2, #5, mul vl]
    st1d {z14.d}, p1, [x2, #6, mul vl]
    st1d {z15.d}, p1, [x2, #7, mul vl]
    RESTORE
    ret
.size fused_fp4_i16_opt_sve, .-fused_fp4_i16_opt_sve
.section .rodata
.p2align 6
.Lfp4_i16:
.hword 0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12
.zero 32
.p2align 6
.Lfp4_i8:
.byte 0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12
.zero 48
.Li4_f16:
.hword 0x0000,0x3c00,0x4000,0x4200,0x4400,0x4500,0x4600,0x4700
.hword 0xc800,0xc700,0xc600,0xc500,0xc400,0xc200,0xc000,0xbc00
.zero 32
.Lfp4_f16:
.hword 0x0000,0x3c00,0x4000,0x4200,0x4400,0x4600,0x4800,0x4a00
.hword 0x0000,0xbc00,0xc000,0xc200,0xc400,0xc600,0xc800,0xca00
.zero 32
.section .note.GNU-stack,"",%progbits
