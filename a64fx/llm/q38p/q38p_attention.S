.text
.p2align 4
.global q38p_qk6x4
.type q38p_qk6x4, %function
// x0 Q[6][256], x1 K[4][256], x2 scores, x3 score head stride; s0 scale.
// Same sixteen-lane accumulations/reduction tree as qk_4h + qk_2h.
q38p_qk6x4:
    stp d8, d9, [sp, #-64]!
    stp d10, d11, [sp, #16]
    stp d12, d13, [sp, #32]
    stp d14, d15, [sp, #48]
    fmov w12, s0
    lsl x13, x3, #2
    add x3, x1, #1024
    add x4, x1, #2048
    add x5, x1, #3072
    add x6, x0, #1024
    add x7, x0, #2048
    add x8, x0, #3072
    add x9, x6, #3072
    add x10, x7, #3072
    ptrue p0.s
    dup z0.s, #0
    dup z1.s, #0
    dup z2.s, #0
    dup z3.s, #0
    dup z4.s, #0
    dup z5.s, #0
    dup z6.s, #0
    dup z7.s, #0
    dup z8.s, #0
    dup z9.s, #0
    dup z10.s, #0
    dup z11.s, #0
    dup z12.s, #0
    dup z13.s, #0
    dup z14.s, #0
    dup z15.s, #0
    dup z16.s, #0
    dup z17.s, #0
    dup z18.s, #0
    dup z19.s, #0
    dup z20.s, #0
    dup z21.s, #0
    dup z22.s, #0
    dup z23.s, #0
    mov x11, #16
.p2align 4
.Lqk_loop:
    ld1w {z24.s}, p0/z, [x1]
    add x1, x1, #64
    ld1w {z25.s}, p0/z, [x3]
    add x3, x3, #64
    ld1w {z26.s}, p0/z, [x4]
    add x4, x4, #64
    ld1w {z27.s}, p0/z, [x5]
    add x5, x5, #64
    ld1w {z28.s}, p0/z, [x0]
    add x0, x0, #64
    fmla z0.s, p0/m, z28.s, z24.s
    fmla z1.s, p0/m, z28.s, z25.s
    fmla z2.s, p0/m, z28.s, z26.s
    fmla z3.s, p0/m, z28.s, z27.s
    ld1w {z28.s}, p0/z, [x6]
    add x6, x6, #64
    fmla z4.s, p0/m, z28.s, z24.s
    fmla z5.s, p0/m, z28.s, z25.s
    fmla z6.s, p0/m, z28.s, z26.s
    fmla z7.s, p0/m, z28.s, z27.s
    ld1w {z28.s}, p0/z, [x7]
    add x7, x7, #64
    fmla z8.s, p0/m, z28.s, z24.s
    fmla z9.s, p0/m, z28.s, z25.s
    fmla z10.s, p0/m, z28.s, z26.s
    fmla z11.s, p0/m, z28.s, z27.s
    ld1w {z28.s}, p0/z, [x8]
    add x8, x8, #64
    fmla z12.s, p0/m, z28.s, z24.s
    fmla z13.s, p0/m, z28.s, z25.s
    fmla z14.s, p0/m, z28.s, z26.s
    fmla z15.s, p0/m, z28.s, z27.s
    ld1w {z28.s}, p0/z, [x9]
    add x9, x9, #64
    fmla z16.s, p0/m, z28.s, z24.s
    fmla z17.s, p0/m, z28.s, z25.s
    fmla z18.s, p0/m, z28.s, z26.s
    fmla z19.s, p0/m, z28.s, z27.s
    ld1w {z28.s}, p0/z, [x10]
    add x10, x10, #64
    fmla z20.s, p0/m, z28.s, z24.s
    fmla z21.s, p0/m, z28.s, z25.s
    fmla z22.s, p0/m, z28.s, z26.s
    fmla z23.s, p0/m, z28.s, z27.s
    subs x11, x11, #1
    b.ne .Lqk_loop
    uzp1 z24.s, z0.s, z1.s
    uzp2 z25.s, z0.s, z1.s
    fadd z0.s, z24.s, z25.s
    uzp1 z24.s, z2.s, z3.s
    uzp2 z25.s, z2.s, z3.s
    fadd z2.s, z24.s, z25.s
    uzp1 z24.s, z4.s, z5.s
    uzp2 z25.s, z4.s, z5.s
    fadd z4.s, z24.s, z25.s
    uzp1 z24.s, z6.s, z7.s
    uzp2 z25.s, z6.s, z7.s
    fadd z6.s, z24.s, z25.s
    uzp1 z24.s, z8.s, z9.s
    uzp2 z25.s, z8.s, z9.s
    fadd z8.s, z24.s, z25.s
    uzp1 z24.s, z10.s, z11.s
    uzp2 z25.s, z10.s, z11.s
    fadd z10.s, z24.s, z25.s
    uzp1 z24.s, z12.s, z13.s
    uzp2 z25.s, z12.s, z13.s
    fadd z12.s, z24.s, z25.s
    uzp1 z24.s, z14.s, z15.s
    uzp2 z25.s, z14.s, z15.s
    fadd z14.s, z24.s, z25.s
    uzp1 z24.s, z0.s, z2.s
    uzp2 z25.s, z0.s, z2.s
    fadd z0.s, z24.s, z25.s
    uzp1 z24.s, z4.s, z6.s
    uzp2 z25.s, z4.s, z6.s
    fadd z4.s, z24.s, z25.s
    uzp1 z24.s, z8.s, z10.s
    uzp2 z25.s, z8.s, z10.s
    fadd z8.s, z24.s, z25.s
    uzp1 z24.s, z12.s, z14.s
    uzp2 z25.s, z12.s, z14.s
    fadd z12.s, z24.s, z25.s
    uzp1 z24.s, z0.s, z4.s
    uzp2 z25.s, z0.s, z4.s
    fadd z0.s, z24.s, z25.s
    uzp1 z24.s, z8.s, z12.s
    uzp2 z25.s, z8.s, z12.s
    fadd z8.s, z24.s, z25.s
    uzp1 z24.s, z0.s, z8.s
    uzp2 z25.s, z0.s, z8.s
    fadd z0.s, z24.s, z25.s
    uzp1 z24.s, z16.s, z17.s
    uzp2 z25.s, z16.s, z17.s
    fadd z16.s, z24.s, z25.s
    uzp1 z24.s, z18.s, z19.s
    uzp2 z25.s, z18.s, z19.s
    fadd z18.s, z24.s, z25.s
    uzp1 z24.s, z20.s, z21.s
    uzp2 z25.s, z20.s, z21.s
    fadd z20.s, z24.s, z25.s
    uzp1 z24.s, z22.s, z23.s
    uzp2 z25.s, z22.s, z23.s
    fadd z22.s, z24.s, z25.s
    uzp1 z24.s, z16.s, z18.s
    uzp2 z25.s, z16.s, z18.s
    fadd z16.s, z24.s, z25.s
    uzp1 z24.s, z20.s, z22.s
    uzp2 z25.s, z20.s, z22.s
    fadd z20.s, z24.s, z25.s
    uzp1 z24.s, z16.s, z20.s
    uzp2 z25.s, z16.s, z20.s
    fadd z16.s, z24.s, z25.s
    uzp1 z24.s, z16.s, z16.s
    uzp2 z25.s, z16.s, z16.s
    fadd z16.s, z24.s, z25.s
    dup z26.s, w12
    fmul z0.s, p0/m, z0.s, z26.s
    fmul z16.s, p0/m, z16.s, z26.s
    ptrue p1.s, vl4
    st1w {z0.s}, p1, [x2]
    add x2, x2, x13
    ext z0.b, z0.b, z0.b, #16
    st1w {z0.s}, p1, [x2]
    add x2, x2, x13
    ext z0.b, z0.b, z0.b, #16
    st1w {z0.s}, p1, [x2]
    add x2, x2, x13
    ext z0.b, z0.b, z0.b, #16
    st1w {z0.s}, p1, [x2]
    add x2, x2, x13
    st1w {z16.s}, p1, [x2]
    add x2, x2, x13
    ext z16.b, z16.b, z16.b, #16
    st1w {z16.s}, p1, [x2]
    ldp d8, d9, [sp]
    ldp d10, d11, [sp, #16]
    ldp d12, d13, [sp, #32]
    ldp d14, d15, [sp, #48]
    add sp, sp, #64
    ret
.size q38p_qk6x4, .-q38p_qk6x4

.p2align 4
.global q38p_sgemm12x32
.type q38p_sgemm12x32, %function
// A[K+1][12], B[K+1][32], C[12][stride]. One padded input row required.
// x0 A, x1 B, x2 C, x3 K, x4 C stride in floats, x5 reset.
// Six rotating broadcasts hide load latency; each output uses sequential FMAs.
q38p_sgemm12x32:
    stp d8, d9, [sp, #-64]!
    stp d10, d11, [sp, #16]
    stp d12, d13, [sp, #32]
    stp d14, d15, [sp, #48]
    ptrue p0.s
    lsl x4, x4, #2
    cbnz x5, .Lsg_zero
    mov x6, x2
    ld1w {z0.s}, p0/z, [x6, #0, mul vl]
    ld1w {z1.s}, p0/z, [x6, #1, mul vl]
    add x6, x6, x4
    ld1w {z2.s}, p0/z, [x6, #0, mul vl]
    ld1w {z3.s}, p0/z, [x6, #1, mul vl]
    add x6, x6, x4
    ld1w {z4.s}, p0/z, [x6, #0, mul vl]
    ld1w {z5.s}, p0/z, [x6, #1, mul vl]
    add x6, x6, x4
    ld1w {z6.s}, p0/z, [x6, #0, mul vl]
    ld1w {z7.s}, p0/z, [x6, #1, mul vl]
    add x6, x6, x4
    ld1w {z8.s}, p0/z, [x6, #0, mul vl]
    ld1w {z9.s}, p0/z, [x6, #1, mul vl]
    add x6, x6, x4
    ld1w {z10.s}, p0/z, [x6, #0, mul vl]
    ld1w {z11.s}, p0/z, [x6, #1, mul vl]
    add x6, x6, x4
    ld1w {z12.s}, p0/z, [x6, #0, mul vl]
    ld1w {z13.s}, p0/z, [x6, #1, mul vl]
    add x6, x6, x4
    ld1w {z14.s}, p0/z, [x6, #0, mul vl]
    ld1w {z15.s}, p0/z, [x6, #1, mul vl]
    add x6, x6, x4
    ld1w {z16.s}, p0/z, [x6, #0, mul vl]
    ld1w {z17.s}, p0/z, [x6, #1, mul vl]
    add x6, x6, x4
    ld1w {z18.s}, p0/z, [x6, #0, mul vl]
    ld1w {z19.s}, p0/z, [x6, #1, mul vl]
    add x6, x6, x4
    ld1w {z20.s}, p0/z, [x6, #0, mul vl]
    ld1w {z21.s}, p0/z, [x6, #1, mul vl]
    add x6, x6, x4
    ld1w {z22.s}, p0/z, [x6, #0, mul vl]
    ld1w {z23.s}, p0/z, [x6, #1, mul vl]
    b .Lsg_start
.Lsg_zero:
    dup z0.s, #0
    dup z1.s, #0
    dup z2.s, #0
    dup z3.s, #0
    dup z4.s, #0
    dup z5.s, #0
    dup z6.s, #0
    dup z7.s, #0
    dup z8.s, #0
    dup z9.s, #0
    dup z10.s, #0
    dup z11.s, #0
    dup z12.s, #0
    dup z13.s, #0
    dup z14.s, #0
    dup z15.s, #0
    dup z16.s, #0
    dup z17.s, #0
    dup z18.s, #0
    dup z19.s, #0
    dup z20.s, #0
    dup z21.s, #0
    dup z22.s, #0
    dup z23.s, #0
.Lsg_start:
    cbz x3, .Lsg_store
    ld1w {z24.s}, p0/z, [x1]
    ld1w {z25.s}, p0/z, [x1, #1, mul vl]
    ld1rw {z26.s}, p0/z, [x0, #0]
    ld1rw {z27.s}, p0/z, [x0, #4]
    ld1rw {z28.s}, p0/z, [x0, #8]
    ld1rw {z29.s}, p0/z, [x0, #12]
    ld1rw {z30.s}, p0/z, [x0, #16]
    ld1rw {z31.s}, p0/z, [x0, #20]
.p2align 4
.Lsg_loop:
    fmla z0.s, p0/m, z24.s, z26.s
    fmla z1.s, p0/m, z25.s, z26.s
    ld1rw {z26.s}, p0/z, [x0, #24]
    fmla z2.s, p0/m, z24.s, z27.s
    fmla z3.s, p0/m, z25.s, z27.s
    ld1rw {z27.s}, p0/z, [x0, #28]
    fmla z4.s, p0/m, z24.s, z28.s
    fmla z5.s, p0/m, z25.s, z28.s
    ld1rw {z28.s}, p0/z, [x0, #32]
    fmla z6.s, p0/m, z24.s, z29.s
    fmla z7.s, p0/m, z25.s, z29.s
    ld1rw {z29.s}, p0/z, [x0, #36]
    fmla z8.s, p0/m, z24.s, z30.s
    fmla z9.s, p0/m, z25.s, z30.s
    ld1rw {z30.s}, p0/z, [x0, #40]
    fmla z10.s, p0/m, z24.s, z31.s
    fmla z11.s, p0/m, z25.s, z31.s
    ld1rw {z31.s}, p0/z, [x0, #44]
    fmla z12.s, p0/m, z24.s, z26.s
    fmla z13.s, p0/m, z25.s, z26.s
    ld1rw {z26.s}, p0/z, [x0, #48]
    fmla z14.s, p0/m, z24.s, z27.s
    fmla z15.s, p0/m, z25.s, z27.s
    ld1rw {z27.s}, p0/z, [x0, #52]
    fmla z16.s, p0/m, z24.s, z28.s
    fmla z17.s, p0/m, z25.s, z28.s
    ld1rw {z28.s}, p0/z, [x0, #56]
    fmla z18.s, p0/m, z24.s, z29.s
    fmla z19.s, p0/m, z25.s, z29.s
    ld1rw {z29.s}, p0/z, [x0, #60]
    fmla z20.s, p0/m, z24.s, z30.s
    fmla z21.s, p0/m, z25.s, z30.s
    ld1rw {z30.s}, p0/z, [x0, #64]
    fmla z22.s, p0/m, z24.s, z31.s
    fmla z23.s, p0/m, z25.s, z31.s
    ld1rw {z31.s}, p0/z, [x0, #68]
    add x0, x0, #48
    add x1, x1, #128
    ld1w {z24.s}, p0/z, [x1]
    ld1w {z25.s}, p0/z, [x1, #1, mul vl]
    subs x3, x3, #1
    b.ne .Lsg_loop
.Lsg_store:
    mov x6, x2
    st1w {z0.s}, p0, [x6, #0, mul vl]
    st1w {z1.s}, p0, [x6, #1, mul vl]
    add x6, x6, x4
    st1w {z2.s}, p0, [x6, #0, mul vl]
    st1w {z3.s}, p0, [x6, #1, mul vl]
    add x6, x6, x4
    st1w {z4.s}, p0, [x6, #0, mul vl]
    st1w {z5.s}, p0, [x6, #1, mul vl]
    add x6, x6, x4
    st1w {z6.s}, p0, [x6, #0, mul vl]
    st1w {z7.s}, p0, [x6, #1, mul vl]
    add x6, x6, x4
    st1w {z8.s}, p0, [x6, #0, mul vl]
    st1w {z9.s}, p0, [x6, #1, mul vl]
    add x6, x6, x4
    st1w {z10.s}, p0, [x6, #0, mul vl]
    st1w {z11.s}, p0, [x6, #1, mul vl]
    add x6, x6, x4
    st1w {z12.s}, p0, [x6, #0, mul vl]
    st1w {z13.s}, p0, [x6, #1, mul vl]
    add x6, x6, x4
    st1w {z14.s}, p0, [x6, #0, mul vl]
    st1w {z15.s}, p0, [x6, #1, mul vl]
    add x6, x6, x4
    st1w {z16.s}, p0, [x6, #0, mul vl]
    st1w {z17.s}, p0, [x6, #1, mul vl]
    add x6, x6, x4
    st1w {z18.s}, p0, [x6, #0, mul vl]
    st1w {z19.s}, p0, [x6, #1, mul vl]
    add x6, x6, x4
    st1w {z20.s}, p0, [x6, #0, mul vl]
    st1w {z21.s}, p0, [x6, #1, mul vl]
    add x6, x6, x4
    st1w {z22.s}, p0, [x6, #0, mul vl]
    st1w {z23.s}, p0, [x6, #1, mul vl]
    ldp d8, d9, [sp]
    ldp d10, d11, [sp, #16]
    ldp d12, d13, [sp, #32]
    ldp d14, d15, [sp, #48]
    add sp, sp, #64
    ret
.size q38p_sgemm12x32, .-q38p_sgemm12x32

.section .note.GNU-stack,"",%progbits
