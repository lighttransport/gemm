/* Lossless Q8_0 dequantization, K-major 64x128 tile, SVE512.
 * x0: four 64-lane FP16 scale vectors followed by K-major signed bytes
 * x1: 128 FP32 activations; x2: 64 FP32 accumulators (read/modify/write).
 * Each 32-term Q8 block is accumulated in FP32 before its original scale is
 * applied. Two weight register banks hide A64FX load/conversion latency. */
.text
.p2align 6
.global q8_k128_f32
.type q8_k128_f32, %function
q8_k128_f32:
    stp d8, d9, [sp, #-16]!
    stp d10, d11, [sp, #-16]!
    stp d12, d13, [sp, #-16]!
    stp d14, d15, [sp, #-16]!
    ptrue p0.s
    ld1w {z0.s}, p0/z, [x2, #0, mul vl]
    ld1w {z1.s}, p0/z, [x2, #1, mul vl]
    ld1w {z2.s}, p0/z, [x2, #2, mul vl]
    ld1w {z3.s}, p0/z, [x2, #3, mul vl]
    mov x3, x0
    add x4, x0, #512
    mov x7, #4

.Lq8k_block:
    ld1h {z16.s}, p0/z, [x3, #0, mul vl]
    ld1h {z17.s}, p0/z, [x3, #1, mul vl]
    ld1h {z18.s}, p0/z, [x3, #2, mul vl]
    ld1h {z19.s}, p0/z, [x3, #3, mul vl]
    fcvt z16.s, p0/m, z16.h
    fcvt z17.s, p0/m, z17.h
    fcvt z18.s, p0/m, z18.h
    fcvt z19.s, p0/m, z19.h
    add x3, x3, #128
    eor z4.d, z4.d, z4.d
    eor z5.d, z5.d, z5.d
    eor z6.d, z6.d, z6.d
    eor z7.d, z7.d, z7.d
    ld1sb {z8.s}, p0/z, [x4, #0, mul vl]
    ld1sb {z9.s}, p0/z, [x4, #1, mul vl]
    ld1sb {z10.s}, p0/z, [x4, #2, mul vl]
    ld1sb {z11.s}, p0/z, [x4, #3, mul vl]
    ld1rw {z24.s}, p0/z, [x1]
    scvtf z8.s, p0/m, z8.s
    scvtf z9.s, p0/m, z9.s
    scvtf z10.s, p0/m, z10.s
    scvtf z11.s, p0/m, z11.s
    mov x6, #15

.Lq8k_pair:
    add x4, x4, #64
    add x1, x1, #4
    ld1sb {z12.s}, p0/z, [x4, #0, mul vl]
    ld1sb {z13.s}, p0/z, [x4, #1, mul vl]
    ld1sb {z14.s}, p0/z, [x4, #2, mul vl]
    ld1sb {z15.s}, p0/z, [x4, #3, mul vl]
    ld1rw {z25.s}, p0/z, [x1]
    fmla z4.s, p0/m, z8.s, z24.s
    scvtf z12.s, p0/m, z12.s
    fmla z5.s, p0/m, z9.s, z24.s
    scvtf z13.s, p0/m, z13.s
    fmla z6.s, p0/m, z10.s, z24.s
    scvtf z14.s, p0/m, z14.s
    fmla z7.s, p0/m, z11.s, z24.s
    scvtf z15.s, p0/m, z15.s
    add x4, x4, #64
    add x1, x1, #4
    ld1sb {z8.s}, p0/z, [x4, #0, mul vl]
    ld1sb {z9.s}, p0/z, [x4, #1, mul vl]
    ld1sb {z10.s}, p0/z, [x4, #2, mul vl]
    ld1sb {z11.s}, p0/z, [x4, #3, mul vl]
    ld1rw {z24.s}, p0/z, [x1]
    fmla z4.s, p0/m, z12.s, z25.s
    scvtf z8.s, p0/m, z8.s
    fmla z5.s, p0/m, z13.s, z25.s
    scvtf z9.s, p0/m, z9.s
    fmla z6.s, p0/m, z14.s, z25.s
    scvtf z10.s, p0/m, z10.s
    fmla z7.s, p0/m, z15.s, z25.s
    scvtf z11.s, p0/m, z11.s
    subs x6, x6, #1
    b.ne .Lq8k_pair

    add x4, x4, #64
    add x1, x1, #4
    ld1sb {z12.s}, p0/z, [x4, #0, mul vl]
    ld1sb {z13.s}, p0/z, [x4, #1, mul vl]
    ld1sb {z14.s}, p0/z, [x4, #2, mul vl]
    ld1sb {z15.s}, p0/z, [x4, #3, mul vl]
    ld1rw {z25.s}, p0/z, [x1]
    fmla z4.s, p0/m, z8.s, z24.s
    scvtf z12.s, p0/m, z12.s
    fmla z5.s, p0/m, z9.s, z24.s
    scvtf z13.s, p0/m, z13.s
    fmla z6.s, p0/m, z10.s, z24.s
    scvtf z14.s, p0/m, z14.s
    fmla z7.s, p0/m, z11.s, z24.s
    scvtf z15.s, p0/m, z15.s
    fmla z4.s, p0/m, z12.s, z25.s
    fmla z5.s, p0/m, z13.s, z25.s
    fmla z6.s, p0/m, z14.s, z25.s
    fmla z7.s, p0/m, z15.s, z25.s
    add x4, x4, #64
    add x1, x1, #4

    fmla z0.s, p0/m, z4.s, z16.s
    fmla z1.s, p0/m, z5.s, z17.s
    fmla z2.s, p0/m, z6.s, z18.s
    fmla z3.s, p0/m, z7.s, z19.s
    subs x7, x7, #1
    b.ne .Lq8k_block

    st1w {z0.s}, p0, [x2, #0, mul vl]
    st1w {z1.s}, p0, [x2, #1, mul vl]
    st1w {z2.s}, p0, [x2, #2, mul vl]
    st1w {z3.s}, p0, [x2, #3, mul vl]
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
