/* Lossless Q8_0 K-major 128-output by 128-input tile, SVE512.
 * x0: four 128-lane FP16 scale vectors, then K-major signed bytes.
 * x1: 128 FP32 activations; x2: 128 FP32 accumulators, updated in place.
 * Eight ascending-K accumulation chains and two weight banks preserve the
 * canonical packed block arithmetic while hiding conversion/FMA latency.
 * Output accumulators are updated after each original 32-element Q8 block. */
.text
.p2align 6
.global q8_k128_f32
.type q8_k128_f32, %function
q8_k128_f32:
    stp d8, d9, [sp, #-16]!
    stp d10, d11, [sp, #-16]!
    stp d12, d13, [sp, #-16]!
    stp d14, d15, [sp, #-16]!
    mov x3, x2
    mov x2, x1
    ptrue p0.s
    add x5, x0, #1024
    mov x9, #4
.Lpair_block:
    eor z8.d, z8.d, z8.d
    eor z9.d, z9.d, z9.d
    eor z10.d, z10.d, z10.d
    eor z11.d, z11.d, z11.d
    eor z12.d, z12.d, z12.d
    eor z13.d, z13.d, z13.d
    eor z14.d, z14.d, z14.d
    eor z15.d, z15.d, z15.d
    ld1sb {z16.s}, p0/z, [x5, #0, mul vl]
    ld1sb {z17.s}, p0/z, [x5, #1, mul vl]
    ld1sb {z18.s}, p0/z, [x5, #2, mul vl]
    ld1sb {z19.s}, p0/z, [x5, #3, mul vl]
    ld1sb {z20.s}, p0/z, [x5, #4, mul vl]
    ld1sb {z21.s}, p0/z, [x5, #5, mul vl]
    ld1sb {z22.s}, p0/z, [x5, #6, mul vl]
    ld1sb {z23.s}, p0/z, [x5, #7, mul vl]
    ld1rw {z0.s}, p0/z, [x2]
    scvtf z16.s, p0/m, z16.s
    scvtf z17.s, p0/m, z17.s
    scvtf z18.s, p0/m, z18.s
    scvtf z19.s, p0/m, z19.s
    scvtf z20.s, p0/m, z20.s
    scvtf z21.s, p0/m, z21.s
    scvtf z22.s, p0/m, z22.s
    scvtf z23.s, p0/m, z23.s
    mov x8, #15
.p2align 6
.Lpair_kloop:
    /* One hint per 256 weight bytes; L2 avoids displacing hot activations. */
    prfm pldl2keep, [x5, #4096]
    add x5, x5, #128
    ld1rw {z1.s}, p0/z, [x2, #4]
    ld1sb {z24.s}, p0/z, [x5, #0, mul vl]
    ld1sb {z25.s}, p0/z, [x5, #1, mul vl]
    ld1sb {z26.s}, p0/z, [x5, #2, mul vl]
    ld1sb {z27.s}, p0/z, [x5, #3, mul vl]
    ld1sb {z28.s}, p0/z, [x5, #4, mul vl]
    ld1sb {z29.s}, p0/z, [x5, #5, mul vl]
    ld1sb {z30.s}, p0/z, [x5, #6, mul vl]
    ld1sb {z31.s}, p0/z, [x5, #7, mul vl]
    fmla z8.s, p0/m, z16.s, z0.s
    scvtf z24.s, p0/m, z24.s
    fmla z9.s, p0/m, z17.s, z0.s
    scvtf z25.s, p0/m, z25.s
    fmla z10.s, p0/m, z18.s, z0.s
    scvtf z26.s, p0/m, z26.s
    fmla z11.s, p0/m, z19.s, z0.s
    scvtf z27.s, p0/m, z27.s
    fmla z12.s, p0/m, z20.s, z0.s
    scvtf z28.s, p0/m, z28.s
    fmla z13.s, p0/m, z21.s, z0.s
    scvtf z29.s, p0/m, z29.s
    fmla z14.s, p0/m, z22.s, z0.s
    scvtf z30.s, p0/m, z30.s
    fmla z15.s, p0/m, z23.s, z0.s
    scvtf z31.s, p0/m, z31.s
    add x5, x5, #128
    ld1rw {z0.s}, p0/z, [x2, #8]
    ld1sb {z16.s}, p0/z, [x5, #0, mul vl]
    ld1sb {z17.s}, p0/z, [x5, #1, mul vl]
    ld1sb {z18.s}, p0/z, [x5, #2, mul vl]
    ld1sb {z19.s}, p0/z, [x5, #3, mul vl]
    ld1sb {z20.s}, p0/z, [x5, #4, mul vl]
    ld1sb {z21.s}, p0/z, [x5, #5, mul vl]
    ld1sb {z22.s}, p0/z, [x5, #6, mul vl]
    ld1sb {z23.s}, p0/z, [x5, #7, mul vl]
    fmla z8.s, p0/m, z24.s, z1.s
    scvtf z16.s, p0/m, z16.s
    fmla z9.s, p0/m, z25.s, z1.s
    scvtf z17.s, p0/m, z17.s
    fmla z10.s, p0/m, z26.s, z1.s
    scvtf z18.s, p0/m, z18.s
    fmla z11.s, p0/m, z27.s, z1.s
    scvtf z19.s, p0/m, z19.s
    fmla z12.s, p0/m, z28.s, z1.s
    scvtf z20.s, p0/m, z20.s
    fmla z13.s, p0/m, z29.s, z1.s
    scvtf z21.s, p0/m, z21.s
    fmla z14.s, p0/m, z30.s, z1.s
    scvtf z22.s, p0/m, z22.s
    fmla z15.s, p0/m, z31.s, z1.s
    scvtf z23.s, p0/m, z23.s
    add x2, x2, #8
    subs x8, x8, #1
    b.ne .Lpair_kloop
    add x5, x5, #128
    ld1rw {z1.s}, p0/z, [x2, #4]
    ld1sb {z24.s}, p0/z, [x5, #0, mul vl]
    ld1sb {z25.s}, p0/z, [x5, #1, mul vl]
    ld1sb {z26.s}, p0/z, [x5, #2, mul vl]
    ld1sb {z27.s}, p0/z, [x5, #3, mul vl]
    ld1sb {z28.s}, p0/z, [x5, #4, mul vl]
    ld1sb {z29.s}, p0/z, [x5, #5, mul vl]
    ld1sb {z30.s}, p0/z, [x5, #6, mul vl]
    ld1sb {z31.s}, p0/z, [x5, #7, mul vl]
    fmla z8.s, p0/m, z16.s, z0.s
    scvtf z24.s, p0/m, z24.s
    fmla z9.s, p0/m, z17.s, z0.s
    scvtf z25.s, p0/m, z25.s
    fmla z10.s, p0/m, z18.s, z0.s
    scvtf z26.s, p0/m, z26.s
    fmla z11.s, p0/m, z19.s, z0.s
    scvtf z27.s, p0/m, z27.s
    fmla z12.s, p0/m, z20.s, z0.s
    scvtf z28.s, p0/m, z28.s
    fmla z13.s, p0/m, z21.s, z0.s
    scvtf z29.s, p0/m, z29.s
    fmla z14.s, p0/m, z22.s, z0.s
    scvtf z30.s, p0/m, z30.s
    fmla z15.s, p0/m, z23.s, z0.s
    scvtf z31.s, p0/m, z31.s
    fmla z8.s, p0/m, z24.s, z1.s
    fmla z9.s, p0/m, z25.s, z1.s
    fmla z10.s, p0/m, z26.s, z1.s
    fmla z11.s, p0/m, z27.s, z1.s
    fmla z12.s, p0/m, z28.s, z1.s
    fmla z13.s, p0/m, z29.s, z1.s
    fmla z14.s, p0/m, z30.s, z1.s
    fmla z15.s, p0/m, z31.s, z1.s
    add x5, x5, #128
    add x2, x2, #8
    ld1h {z16.s}, p0/z, [x0, #0, mul vl]
    ld1h {z17.s}, p0/z, [x0, #1, mul vl]
    ld1h {z18.s}, p0/z, [x0, #2, mul vl]
    ld1h {z19.s}, p0/z, [x0, #3, mul vl]
    ld1h {z20.s}, p0/z, [x0, #4, mul vl]
    ld1h {z21.s}, p0/z, [x0, #5, mul vl]
    ld1h {z22.s}, p0/z, [x0, #6, mul vl]
    ld1h {z23.s}, p0/z, [x0, #7, mul vl]
    ld1w {z24.s}, p0/z, [x3, #0, mul vl]
    fcvt z16.s, p0/m, z16.h
    ld1w {z25.s}, p0/z, [x3, #1, mul vl]
    fcvt z17.s, p0/m, z17.h
    ld1w {z26.s}, p0/z, [x3, #2, mul vl]
    fcvt z18.s, p0/m, z18.h
    ld1w {z27.s}, p0/z, [x3, #3, mul vl]
    fcvt z19.s, p0/m, z19.h
    ld1w {z28.s}, p0/z, [x3, #4, mul vl]
    fcvt z20.s, p0/m, z20.h
    ld1w {z29.s}, p0/z, [x3, #5, mul vl]
    fcvt z21.s, p0/m, z21.h
    ld1w {z30.s}, p0/z, [x3, #6, mul vl]
    fcvt z22.s, p0/m, z22.h
    ld1w {z31.s}, p0/z, [x3, #7, mul vl]
    fcvt z23.s, p0/m, z23.h
    fmla z24.s, p0/m, z8.s, z16.s
    fmla z25.s, p0/m, z9.s, z17.s
    fmla z26.s, p0/m, z10.s, z18.s
    fmla z27.s, p0/m, z11.s, z19.s
    fmla z28.s, p0/m, z12.s, z20.s
    fmla z29.s, p0/m, z13.s, z21.s
    fmla z30.s, p0/m, z14.s, z22.s
    fmla z31.s, p0/m, z15.s, z23.s
    st1w {z24.s}, p0, [x3, #0, mul vl]
    st1w {z25.s}, p0, [x3, #1, mul vl]
    st1w {z26.s}, p0, [x3, #2, mul vl]
    st1w {z27.s}, p0, [x3, #3, mul vl]
    st1w {z28.s}, p0, [x3, #4, mul vl]
    st1w {z29.s}, p0, [x3, #5, mul vl]
    st1w {z30.s}, p0, [x3, #6, mul vl]
    st1w {z31.s}, p0, [x3, #7, mul vl]
    add x0, x0, #256
    subs x9, x9, #1
    b.ne .Lpair_block
    ldp d14, d15, [sp], #16
    ldp d12, d13, [sp], #16
    ldp d10, d11, [sp], #16
    ldp d8, d9, [sp], #16
    ret
.size q8_k128_f32, .-q8_k128_f32

/* Read a 256-byte-aligned/multiple stream with the same four-load loop used by
 * the standalone HBM roofline benchmark.  This provides a reduction-free
 * resident-arena control; callers handle unaligned edges separately. */
.p2align 6
.global q8_stream_256_sve
.type q8_stream_256_sve, %function
q8_stream_256_sve:
    ptrue p0.b
    cbz x1, .Lq8_stream_done
.Lq8_stream_loop:
    ld1b {z0.b}, p0/z, [x0]
    ld1b {z1.b}, p0/z, [x0, #1, mul vl]
    ld1b {z2.b}, p0/z, [x0, #2, mul vl]
    ld1b {z3.b}, p0/z, [x0, #3, mul vl]
    add x0, x0, #256
    subs x1, x1, #256
    b.ne .Lq8_stream_loop
.Lq8_stream_done:
    ret
.size q8_stream_256_sve, .-q8_stream_256_sve

/* Opt-in W8A15: q = 128*hi + lo, exact integer dot per Q8 block. */
.p2align 6
.global q8_k128_a15_dot
.type q8_k128_a15_dot, %function
q8_k128_a15_dot:
    stp d8, d9, [sp, #-16]!
    stp d10, d11, [sp, #-16]!
    stp d12, d13, [sp, #-16]!
    stp d14, d15, [sp, #-16]!
    ptrue p0.s
    ptrue p1.b
    add x5, x0, #1024
    mov x6, #4
.La15_block:
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
    mov x7, #8
.La15_k:
    prfm pldl2keep, [x5, #8192]
    prfm pldl2keep, [x5, #8448]
    ld1rw {z24.s}, p0/z, [x1]
    ld1rw {z25.s}, p0/z, [x2]
    ld1b {z16.b}, p1/z, [x5, #0, mul vl]
    ld1b {z17.b}, p1/z, [x5, #1, mul vl]
    ld1b {z18.b}, p1/z, [x5, #2, mul vl]
    ld1b {z19.b}, p1/z, [x5, #3, mul vl]
    ld1b {z20.b}, p1/z, [x5, #4, mul vl]
    ld1b {z21.b}, p1/z, [x5, #5, mul vl]
    ld1b {z22.b}, p1/z, [x5, #6, mul vl]
    ld1b {z23.b}, p1/z, [x5, #7, mul vl]
    sdot z0.s, z16.b, z24.b
    sdot z8.s, z16.b, z25.b
    sdot z1.s, z17.b, z24.b
    sdot z9.s, z17.b, z25.b
    sdot z2.s, z18.b, z24.b
    sdot z10.s, z18.b, z25.b
    sdot z3.s, z19.b, z24.b
    sdot z11.s, z19.b, z25.b
    sdot z4.s, z20.b, z24.b
    sdot z12.s, z20.b, z25.b
    sdot z5.s, z21.b, z24.b
    sdot z13.s, z21.b, z25.b
    sdot z6.s, z22.b, z24.b
    sdot z14.s, z22.b, z25.b
    sdot z7.s, z23.b, z24.b
    sdot z15.s, z23.b, z25.b
    add x5, x5, #512
    add x1, x1, #4
    add x2, x2, #4
    subs x7, x7, #1
    b.ne .La15_k
    ld1rw {z24.s}, p0/z, [x3]
    lsl z0.s, z0.s, #7
    add z0.s, z0.s, z8.s
    scvtf z0.s, p0/m, z0.s
    ld1h {z16.s}, p0/z, [x0, #0, mul vl]
    fcvt z16.s, p0/m, z16.h
    fmul z16.s, z16.s, z24.s
    ld1w {z8.s}, p0/z, [x4, #0, mul vl]
    fmla z8.s, p0/m, z0.s, z16.s
    st1w {z8.s}, p0, [x4, #0, mul vl]
    lsl z1.s, z1.s, #7
    add z1.s, z1.s, z9.s
    scvtf z1.s, p0/m, z1.s
    ld1h {z17.s}, p0/z, [x0, #1, mul vl]
    fcvt z17.s, p0/m, z17.h
    fmul z17.s, z17.s, z24.s
    ld1w {z9.s}, p0/z, [x4, #1, mul vl]
    fmla z9.s, p0/m, z1.s, z17.s
    st1w {z9.s}, p0, [x4, #1, mul vl]
    lsl z2.s, z2.s, #7
    add z2.s, z2.s, z10.s
    scvtf z2.s, p0/m, z2.s
    ld1h {z18.s}, p0/z, [x0, #2, mul vl]
    fcvt z18.s, p0/m, z18.h
    fmul z18.s, z18.s, z24.s
    ld1w {z10.s}, p0/z, [x4, #2, mul vl]
    fmla z10.s, p0/m, z2.s, z18.s
    st1w {z10.s}, p0, [x4, #2, mul vl]
    lsl z3.s, z3.s, #7
    add z3.s, z3.s, z11.s
    scvtf z3.s, p0/m, z3.s
    ld1h {z19.s}, p0/z, [x0, #3, mul vl]
    fcvt z19.s, p0/m, z19.h
    fmul z19.s, z19.s, z24.s
    ld1w {z11.s}, p0/z, [x4, #3, mul vl]
    fmla z11.s, p0/m, z3.s, z19.s
    st1w {z11.s}, p0, [x4, #3, mul vl]
    lsl z4.s, z4.s, #7
    add z4.s, z4.s, z12.s
    scvtf z4.s, p0/m, z4.s
    ld1h {z20.s}, p0/z, [x0, #4, mul vl]
    fcvt z20.s, p0/m, z20.h
    fmul z20.s, z20.s, z24.s
    ld1w {z12.s}, p0/z, [x4, #4, mul vl]
    fmla z12.s, p0/m, z4.s, z20.s
    st1w {z12.s}, p0, [x4, #4, mul vl]
    lsl z5.s, z5.s, #7
    add z5.s, z5.s, z13.s
    scvtf z5.s, p0/m, z5.s
    ld1h {z21.s}, p0/z, [x0, #5, mul vl]
    fcvt z21.s, p0/m, z21.h
    fmul z21.s, z21.s, z24.s
    ld1w {z13.s}, p0/z, [x4, #5, mul vl]
    fmla z13.s, p0/m, z5.s, z21.s
    st1w {z13.s}, p0, [x4, #5, mul vl]
    lsl z6.s, z6.s, #7
    add z6.s, z6.s, z14.s
    scvtf z6.s, p0/m, z6.s
    ld1h {z22.s}, p0/z, [x0, #6, mul vl]
    fcvt z22.s, p0/m, z22.h
    fmul z22.s, z22.s, z24.s
    ld1w {z14.s}, p0/z, [x4, #6, mul vl]
    fmla z14.s, p0/m, z6.s, z22.s
    st1w {z14.s}, p0, [x4, #6, mul vl]
    lsl z7.s, z7.s, #7
    add z7.s, z7.s, z15.s
    scvtf z7.s, p0/m, z7.s
    ld1h {z23.s}, p0/z, [x0, #7, mul vl]
    fcvt z23.s, p0/m, z23.h
    fmul z23.s, z23.s, z24.s
    ld1w {z15.s}, p0/z, [x4, #7, mul vl]
    fmla z15.s, p0/m, z7.s, z23.s
    st1w {z15.s}, p0, [x4, #7, mul vl]
    add x0, x0, #256
    add x3, x3, #4
    subs x6, x6, #1
    b.ne .La15_block
    ldp d14, d15, [sp], #16
    ldp d12, d13, [sp], #16
    ldp d10, d11, [sp], #16
    ldp d8, d9, [sp], #16
    ret
.size q8_k128_a15_dot, .-q8_k128_a15_dot
