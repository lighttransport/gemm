// FP16 8x64 A64FX SVE tail microkernel
// C[8][64] = A[K][0:8 of stride 12] * B[K][64]
// Args: x0=A[K][12], x1=B[K][64], x2=C, x3=K, x4=unused, x5=ldc_bytes
    .arch armv8.2-a+sve

    .text
    .align 4
    .global fp4_dense_m8n64_init
    .type fp4_dense_m8n64_init, %function

fp4_dense_m8n64_init:
    stp d8, d9, [sp, #-64]!
    stp d10, d11, [sp, #16]
    stp d12, d13, [sp, #32]
    stp d14, d15, [sp, #48]
    ptrue p0.h
    ptrue p1.s
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

.Lfp4_dense_m8n64_init_k:
    ld1h {z16.h}, p0/z, [x1]
    ld1h {z17.h}, p0/z, [x1, #1, mul vl]
    ld1rh {z18.h}, p0/z, [x0, #0]
    ld1rh {z19.h}, p0/z, [x0, #2]
    ld1rh {z20.h}, p0/z, [x0, #4]
    ld1rh {z21.h}, p0/z, [x0, #6]
    ld1rh {z22.h}, p0/z, [x0, #8]
    ld1rh {z23.h}, p0/z, [x0, #10]
    ld1rh {z24.h}, p0/z, [x0, #12]
    ld1rh {z25.h}, p0/z, [x0, #14]
    fmla z0.h, p0/m, z18.h, z16.h
    fmla z1.h, p0/m, z18.h, z17.h
    fmla z2.h, p0/m, z19.h, z16.h
    fmla z3.h, p0/m, z19.h, z17.h
    fmla z4.h, p0/m, z20.h, z16.h
    fmla z5.h, p0/m, z20.h, z17.h
    fmla z6.h, p0/m, z21.h, z16.h
    fmla z7.h, p0/m, z21.h, z17.h
    fmla z8.h, p0/m, z22.h, z16.h
    fmla z9.h, p0/m, z22.h, z17.h
    fmla z10.h, p0/m, z23.h, z16.h
    fmla z11.h, p0/m, z23.h, z17.h
    fmla z12.h, p0/m, z24.h, z16.h
    fmla z13.h, p0/m, z24.h, z17.h
    fmla z14.h, p0/m, z25.h, z16.h
    fmla z15.h, p0/m, z25.h, z17.h
    add x0, x0, #24
    add x1, x1, #128
    subs x3, x3, #1
    b.ne .Lfp4_dense_m8n64_init_k

    uunpklo z16.s, z0.h
    fcvt z16.s, p1/m, z16.h
    st1w {z16.s}, p1, [x2]
    uunpkhi z16.s, z0.h
    fcvt z16.s, p1/m, z16.h
    st1w {z16.s}, p1, [x2, #1, mul vl]
    uunpklo z16.s, z1.h
    fcvt z16.s, p1/m, z16.h
    st1w {z16.s}, p1, [x2, #2, mul vl]
    uunpkhi z16.s, z1.h
    fcvt z16.s, p1/m, z16.h
    st1w {z16.s}, p1, [x2, #3, mul vl]
    add x2, x2, x5
    uunpklo z16.s, z2.h
    fcvt z16.s, p1/m, z16.h
    st1w {z16.s}, p1, [x2]
    uunpkhi z16.s, z2.h
    fcvt z16.s, p1/m, z16.h
    st1w {z16.s}, p1, [x2, #1, mul vl]
    uunpklo z16.s, z3.h
    fcvt z16.s, p1/m, z16.h
    st1w {z16.s}, p1, [x2, #2, mul vl]
    uunpkhi z16.s, z3.h
    fcvt z16.s, p1/m, z16.h
    st1w {z16.s}, p1, [x2, #3, mul vl]
    add x2, x2, x5
    uunpklo z16.s, z4.h
    fcvt z16.s, p1/m, z16.h
    st1w {z16.s}, p1, [x2]
    uunpkhi z16.s, z4.h
    fcvt z16.s, p1/m, z16.h
    st1w {z16.s}, p1, [x2, #1, mul vl]
    uunpklo z16.s, z5.h
    fcvt z16.s, p1/m, z16.h
    st1w {z16.s}, p1, [x2, #2, mul vl]
    uunpkhi z16.s, z5.h
    fcvt z16.s, p1/m, z16.h
    st1w {z16.s}, p1, [x2, #3, mul vl]
    add x2, x2, x5
    uunpklo z16.s, z6.h
    fcvt z16.s, p1/m, z16.h
    st1w {z16.s}, p1, [x2]
    uunpkhi z16.s, z6.h
    fcvt z16.s, p1/m, z16.h
    st1w {z16.s}, p1, [x2, #1, mul vl]
    uunpklo z16.s, z7.h
    fcvt z16.s, p1/m, z16.h
    st1w {z16.s}, p1, [x2, #2, mul vl]
    uunpkhi z16.s, z7.h
    fcvt z16.s, p1/m, z16.h
    st1w {z16.s}, p1, [x2, #3, mul vl]
    add x2, x2, x5
    uunpklo z16.s, z8.h
    fcvt z16.s, p1/m, z16.h
    st1w {z16.s}, p1, [x2]
    uunpkhi z16.s, z8.h
    fcvt z16.s, p1/m, z16.h
    st1w {z16.s}, p1, [x2, #1, mul vl]
    uunpklo z16.s, z9.h
    fcvt z16.s, p1/m, z16.h
    st1w {z16.s}, p1, [x2, #2, mul vl]
    uunpkhi z16.s, z9.h
    fcvt z16.s, p1/m, z16.h
    st1w {z16.s}, p1, [x2, #3, mul vl]
    add x2, x2, x5
    uunpklo z16.s, z10.h
    fcvt z16.s, p1/m, z16.h
    st1w {z16.s}, p1, [x2]
    uunpkhi z16.s, z10.h
    fcvt z16.s, p1/m, z16.h
    st1w {z16.s}, p1, [x2, #1, mul vl]
    uunpklo z16.s, z11.h
    fcvt z16.s, p1/m, z16.h
    st1w {z16.s}, p1, [x2, #2, mul vl]
    uunpkhi z16.s, z11.h
    fcvt z16.s, p1/m, z16.h
    st1w {z16.s}, p1, [x2, #3, mul vl]
    add x2, x2, x5
    uunpklo z16.s, z12.h
    fcvt z16.s, p1/m, z16.h
    st1w {z16.s}, p1, [x2]
    uunpkhi z16.s, z12.h
    fcvt z16.s, p1/m, z16.h
    st1w {z16.s}, p1, [x2, #1, mul vl]
    uunpklo z16.s, z13.h
    fcvt z16.s, p1/m, z16.h
    st1w {z16.s}, p1, [x2, #2, mul vl]
    uunpkhi z16.s, z13.h
    fcvt z16.s, p1/m, z16.h
    st1w {z16.s}, p1, [x2, #3, mul vl]
    add x2, x2, x5
    uunpklo z16.s, z14.h
    fcvt z16.s, p1/m, z16.h
    st1w {z16.s}, p1, [x2]
    uunpkhi z16.s, z14.h
    fcvt z16.s, p1/m, z16.h
    st1w {z16.s}, p1, [x2, #1, mul vl]
    uunpklo z16.s, z15.h
    fcvt z16.s, p1/m, z16.h
    st1w {z16.s}, p1, [x2, #2, mul vl]
    uunpkhi z16.s, z15.h
    fcvt z16.s, p1/m, z16.h
    st1w {z16.s}, p1, [x2, #3, mul vl]
    add x2, x2, x5
    ldp d14, d15, [sp, #48]
    ldp d12, d13, [sp, #32]
    ldp d10, d11, [sp, #16]
    ldp d8, d9, [sp], #64
    ret
    .size fp4_dense_m8n64_init, .-fp4_dense_m8n64_init

