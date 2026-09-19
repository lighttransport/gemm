    .arch armv8.2-a+sve
    .text
    .align 6

/*
 * Read exactly one 256-byte A64FX cache line per iteration.  Four full-vector
 * loads are used because SVE VL is 64 bytes on A64FX.  The source address and
 * byte count must both be 256-byte aligned/multiples.
 */
    .global hbm_read_256_sve
    .type hbm_read_256_sve, %function
hbm_read_256_sve:
    ptrue p0.b
    cbz x1, 2f
1:
    ld1b {z0.b}, p0/z, [x0]
    ld1b {z1.b}, p0/z, [x0, #1, mul vl]
    ld1b {z2.b}, p0/z, [x0, #2, mul vl]
    ld1b {z3.b}, p0/z, [x0, #3, mul vl]
    add x0, x0, #256
    subs x1, x1, #256
    b.ne 1b
2:
    ret
    .size hbm_read_256_sve, .-hbm_read_256_sve

/* Same stream with an L1 keep prefetch eight 256-byte lines ahead. */
    .global hbm_read_256_pf_sve
    .type hbm_read_256_pf_sve, %function
hbm_read_256_pf_sve:
    ptrue p0.b
    cbz x1, 2f
1:
    prfm pldl1keep, [x0, #2048]
    ld1b {z0.b}, p0/z, [x0]
    ld1b {z1.b}, p0/z, [x0, #1, mul vl]
    ld1b {z2.b}, p0/z, [x0, #2, mul vl]
    ld1b {z3.b}, p0/z, [x0, #3, mul vl]
    add x0, x0, #256
    subs x1, x1, #256
    b.ne 1b
2:
    ret
    .size hbm_read_256_pf_sve, .-hbm_read_256_pf_sve

/* HBM roofline probes: four 64-byte loads plus N independent arithmetic ops. */
    .altmacro
    .macro COMPUTE_OP kind, acc, src
    .ifc \kind,sdot
    sdot z\acc\().s, z\src\().b, z28.b
    .else
    fmla z\acc\().s, p0/m, z\src\().s, z28.s
    .endif
    .endm

    .macro GEN_HBM_COMPUTE kind, count
    .global hbm_read_256_\kind\()_\count\()_sve
    .type hbm_read_256_\kind\()_\count\()_sve, %function
hbm_read_256_\kind\()_\count\()_sve:
    ptrue p0.b
    mov z28.b, #1
    cbz x1, 2f
1:
    ld1b {z24.b}, p0/z, [x0]
    ld1b {z25.b}, p0/z, [x0, #1, mul vl]
    ld1b {z26.b}, p0/z, [x0, #2, mul vl]
    ld1b {z27.b}, p0/z, [x0, #3, mul vl]
    .set n, 0
    .rept \count
    COMPUTE_OP \kind, %(n % 24), %(24 + (n % 4))
    .set n, n + 1
    .endr
    add x0, x0, #256
    subs x1, x1, #256
    b.ne 1b
2:
    ret
    .size hbm_read_256_\kind\()_\count\()_sve, .-hbm_read_256_\kind\()_\count\()_sve
    .endm

    GEN_HBM_COMPUTE sdot, 4
    GEN_HBM_COMPUTE sdot, 8
    GEN_HBM_COMPUTE sdot, 12
    GEN_HBM_COMPUTE sdot, 16
    GEN_HBM_COMPUTE sdot, 24
    GEN_HBM_COMPUTE sdot, 32
    GEN_HBM_COMPUTE sdot, 40
    GEN_HBM_COMPUTE sdot, 48
    GEN_HBM_COMPUTE sdot, 52
    GEN_HBM_COMPUTE sdot, 56
    GEN_HBM_COMPUTE sdot, 60
    GEN_HBM_COMPUTE sdot, 64
    GEN_HBM_COMPUTE fmla, 4
    GEN_HBM_COMPUTE fmla, 8
    GEN_HBM_COMPUTE fmla, 12
    GEN_HBM_COMPUTE fmla, 16
    GEN_HBM_COMPUTE fmla, 24
    GEN_HBM_COMPUTE fmla, 32
    GEN_HBM_COMPUTE fmla, 40
    GEN_HBM_COMPUTE fmla, 48
    GEN_HBM_COMPUTE fmla, 52
    GEN_HBM_COMPUTE fmla, 56
    GEN_HBM_COMPUTE fmla, 60
    GEN_HBM_COMPUTE fmla, 64
    .noaltmacro

/* Pure load/store controls for the HBM -> L2 -> L2 transport benchmark. */
    .global null_expand1_sve
    .type null_expand1_sve, %function
null_expand1_sve:
    ptrue p0.b
    cbz x2, 2f
1:
    ld1b {z0.b}, p0/z, [x0]
    ld1b {z1.b}, p0/z, [x0, #1, mul vl]
    ld1b {z2.b}, p0/z, [x0, #2, mul vl]
    ld1b {z3.b}, p0/z, [x0, #3, mul vl]
    st1b {z0.b}, p0, [x1]
    st1b {z1.b}, p0, [x1, #1, mul vl]
    st1b {z2.b}, p0, [x1, #2, mul vl]
    st1b {z3.b}, p0, [x1, #3, mul vl]
    add x0, x0, #256
    add x1, x1, #256
    subs x2, x2, #256
    b.ne 1b
2:
    ret
    .size null_expand1_sve, .-null_expand1_sve

    .global null_expand2_sve
    .type null_expand2_sve, %function
null_expand2_sve:
    ptrue p0.b
    cbz x2, 2f
1:
    ld1b {z0.b}, p0/z, [x0]
    ld1b {z1.b}, p0/z, [x0, #1, mul vl]
    st1b {z0.b}, p0, [x1]
    st1b {z0.b}, p0, [x1, #1, mul vl]
    st1b {z1.b}, p0, [x1, #2, mul vl]
    st1b {z1.b}, p0, [x1, #3, mul vl]
    add x0, x0, #128
    add x1, x1, #256
    subs x2, x2, #128
    b.ne 1b
2:
    ret
    .size null_expand2_sve, .-null_expand2_sve

    .global null_expand4_sve
    .type null_expand4_sve, %function
null_expand4_sve:
    ptrue p0.b
    cbz x2, 2f
1:
    ld1b {z0.b}, p0/z, [x0]
    st1b {z0.b}, p0, [x1]
    st1b {z0.b}, p0, [x1, #1, mul vl]
    st1b {z0.b}, p0, [x1, #2, mul vl]
    st1b {z0.b}, p0, [x1, #3, mul vl]
    add x0, x0, #64
    add x1, x1, #256
    subs x2, x2, #64
    b.ne 1b
2:
    ret
    .size null_expand4_sve, .-null_expand4_sve

    .global null_copy_sve
    .type null_copy_sve, %function
null_copy_sve:
    ptrue p0.b
    cbz x2, 2f
1:
    ld1b {z0.b}, p0/z, [x0]
    ld1b {z1.b}, p0/z, [x0, #1, mul vl]
    ld1b {z2.b}, p0/z, [x0, #2, mul vl]
    ld1b {z3.b}, p0/z, [x0, #3, mul vl]
    st1b {z0.b}, p0, [x1]
    st1b {z1.b}, p0, [x1, #1, mul vl]
    st1b {z2.b}, p0, [x1, #2, mul vl]
    st1b {z3.b}, p0, [x1, #3, mul vl]
    add x0, x0, #256
    add x1, x1, #256
    subs x2, x2, #256
    b.ne 1b
2:
    ret
    .size null_copy_sve, .-null_copy_sve

/* Expand consecutive low/high nibbles to signed int8 values. */
    .global dequant_i4_i8_sve
    .type dequant_i4_i8_sve, %function
dequant_i4_i8_sve:
    ptrue   p0.b
    cbz     x2, 2f
1:
    ld1b    {z0.b}, p0/z, [x0]
    mov     z1.d, z0.d
    and     z1.b, z1.b, #0x0f
    lsr     z2.b, z0.b, #4
    lsl     z1.b, z1.b, #4
    asr     z1.b, z1.b, #4
    lsl     z2.b, z2.b, #4
    asr     z2.b, z2.b, #4
    zip1    z3.b, z1.b, z2.b
    zip2    z4.b, z1.b, z2.b
    st1b    {z3.b}, p0, [x1]
    st1b    {z4.b}, p0, [x1, #1, mul vl]
    add     x0, x0, #64
    add     x1, x1, #128
    subs    x2, x2, #64
    b.ne    1b
2:
    ret
    .size dequant_i4_i8_sve, .-dequant_i4_i8_sve

/* E2M1 code -> signed integer lattice scaled by two. */
    .global dequant_fp4_i8_sve
    .type dequant_fp4_i8_sve, %function
dequant_fp4_i8_sve:
    ptrue   p0.b
    adr     x3, .Lfp4_lut
    ld1b    {z5.b}, p0/z, [x3]
    cbz     x2, 2f
1:
    ld1b    {z0.b}, p0/z, [x0]
    mov     z1.d, z0.d
    and     z1.b, z1.b, #0x0f
    lsr     z2.b, z0.b, #4
    zip1    z3.b, z1.b, z2.b
    zip2    z4.b, z1.b, z2.b
    tbl     z3.b, {z5.b}, z3.b
    tbl     z4.b, {z5.b}, z4.b
    st1b    {z3.b}, p0, [x1]
    st1b    {z4.b}, p0, [x1, #1, mul vl]
    add     x0, x0, #64
    add     x1, x1, #128
    subs    x2, x2, #64
    b.ne    1b
2:
    ret
    .size dequant_fp4_i8_sve, .-dequant_fp4_i8_sve

/* Direct nibble expansion to signed int16. */
    .macro DEQUANT_I16 name, fp4
    .global \name
    .type \name, %function
\name:
    ptrue   p0.b
    ptrue   p1.h
    .if \fp4
    adr     x3, .Lfp4_lut
    ld1b    {z7.b}, p0/z, [x3]
    .endif
    cbz     x2, 2f
1:
    ld1b    {z0.b}, p0/z, [x0]
    mov     z1.d, z0.d
    and     z1.b, z1.b, #0x0f
    lsr     z2.b, z0.b, #4
    .if \fp4
    zip1    z3.b, z1.b, z2.b
    zip2    z4.b, z1.b, z2.b
    tbl     z3.b, {z7.b}, z3.b
    tbl     z4.b, {z7.b}, z4.b
    .else
    lsl     z1.b, z1.b, #4
    asr     z1.b, z1.b, #4
    lsl     z2.b, z2.b, #4
    asr     z2.b, z2.b, #4
    zip1    z3.b, z1.b, z2.b
    zip2    z4.b, z1.b, z2.b
    .endif
    sunpklo z0.h, z3.b
    sunpkhi z1.h, z3.b
    sunpklo z2.h, z4.b
    sunpkhi z3.h, z4.b
    st1h    {z0.h}, p1, [x1]
    st1h    {z1.h}, p1, [x1, #1, mul vl]
    st1h    {z2.h}, p1, [x1, #2, mul vl]
    st1h    {z3.h}, p1, [x1, #3, mul vl]
    add     x0, x0, #64
    add     x1, x1, #256
    subs    x2, x2, #64
    b.ne    1b
2:
    ret
    .size \name, .-\name
    .endm


    DEQUANT_I16 dequant_i4_i16_sve, 0
    DEQUANT_I16 dequant_fp4_i16_sve, 1

/*
 * FP16 block expansion. Source is k-major: 32 packed bytes for each of 128
 * K positions and 64 N values. scales contains 64 FP16 values. FP4 uses its
 * x2 integer lattice and therefore receives scale/2 from the caller.
 */
    .macro DEQUANT_F16_BLOCK name, fp4
    .global \name
    .type \name, %function
\name:
    ptrue   p0.b
    mov     x4, #32
    whilelt p1.b, xzr, x4
    ptrue   p2.h
    ld1h    {z30.h}, p2/z, [x2]
    ld1h    {z31.h}, p2/z, [x2, #1, mul vl]
    .if \fp4
    adr     x5, .Lfp4_lut
    ld1b    {z29.b}, p0/z, [x5]
    .endif
    mov     x3, #128
1:
    ld1b    {z0.b}, p1/z, [x0]
    mov     z1.d, z0.d
    and     z1.b, z1.b, #0x0f
    lsr     z2.b, z0.b, #4
    zip1    z3.b, z1.b, z2.b
    .if \fp4
    tbl     z3.b, {z29.b}, z3.b
    .else
    lsl     z3.b, z3.b, #4
    asr     z3.b, z3.b, #4
    .endif
    sunpklo z4.h, z3.b
    sunpkhi z5.h, z3.b
    scvtf   z4.h, p2/m, z4.h
    scvtf   z5.h, p2/m, z5.h
    fmul    z4.h, p2/m, z4.h, z30.h
    fmul    z5.h, p2/m, z5.h, z31.h
    st1h    {z4.h}, p2, [x1]
    st1h    {z5.h}, p2, [x1, #1, mul vl]
    add     x0, x0, #32
    add     x1, x1, #128
    subs    x3, x3, #1
    b.ne    1b
    ret
    .size \name, .-\name
    .endm

    DEQUANT_F16_BLOCK dequant_i4_f16_block_sve, 0
    DEQUANT_F16_BLOCK dequant_fp4_f16_block_sve, 1

/* Save the ABI-preserved low 64 bits of z8-z15 used as accumulators. */
    .macro SAVE_D8_D15
    stp     d8, d9, [sp, #-16]!
    stp     d10, d11, [sp, #-16]!
    stp     d12, d13, [sp, #-16]!
    stp     d14, d15, [sp, #-16]!
    .endm
    .macro RESTORE_D8_D15
    ldp     d14, d15, [sp], #16
    ldp     d12, d13, [sp], #16
    ldp     d10, d11, [sp], #16
    ldp     d8, d9, [sp], #16
    .endm

/* Mx64 by K=128, packed for four 16-output SDOT vectors per K group. */
    .macro GEMM_SDOT name, rows, halfword
    .global \name
    .type \name, %function
\name:
    SAVE_D8_D15
    ptrue   p0.b
    .if \halfword
    mov     x10, #32
    mov     x11, #256
    .else
    mov     x10, #32
    mov     x11, #128
    .endif
    mov     x3, x1
    .if \rows >= 2
    add     x4, x3, x11
    .endif
    .if \rows >= 3
    add     x5, x4, x11
    .endif
    .if \rows >= 4
    add     x6, x5, x11
    .endif
    .if \rows >= 5
    add     x7, x6, x11
    .endif
    .if \rows >= 6
    add     x8, x7, x11
    .endif
    eor     z0.d, z0.d, z0.d
    eor     z1.d, z1.d, z1.d
    eor     z2.d, z2.d, z2.d
    eor     z3.d, z3.d, z3.d
    .if \rows >= 2
    eor     z4.d, z4.d, z4.d
    eor     z5.d, z5.d, z5.d
    eor     z6.d, z6.d, z6.d
    eor     z7.d, z7.d, z7.d
    .endif
    .if \rows >= 3
    eor     z8.d, z8.d, z8.d
    eor     z9.d, z9.d, z9.d
    eor     z10.d, z10.d, z10.d
    eor     z11.d, z11.d, z11.d
    .endif
    .if \rows >= 4
    eor     z12.d, z12.d, z12.d
    eor     z13.d, z13.d, z13.d
    eor     z14.d, z14.d, z14.d
    eor     z15.d, z15.d, z15.d
    .endif
    .if \rows >= 5
    eor     z16.d, z16.d, z16.d
    eor     z17.d, z17.d, z17.d
    eor     z18.d, z18.d, z18.d
    eor     z19.d, z19.d, z19.d
    .endif
    .if \rows >= 6
    eor     z20.d, z20.d, z20.d
    eor     z21.d, z21.d, z21.d
    eor     z22.d, z22.d, z22.d
    eor     z23.d, z23.d, z23.d
    .endif
1:
    .if \halfword
    ld1h    {z24.h}, p0/z, [x0]
    ld1h    {z25.h}, p0/z, [x0, #1, mul vl]
    ld1h    {z26.h}, p0/z, [x0, #2, mul vl]
    ld1h    {z27.h}, p0/z, [x0, #3, mul vl]
    .else
    ld1b    {z24.b}, p0/z, [x0]
    ld1b    {z25.b}, p0/z, [x0, #1, mul vl]
    ld1b    {z26.b}, p0/z, [x0, #2, mul vl]
    ld1b    {z27.b}, p0/z, [x0, #3, mul vl]
    .endif
    .if \halfword
    ld1rd   {z28.d}, p0/z, [x3]
    .else
    ld1rw   {z28.s}, p0/z, [x3]
    .endif
    .if \halfword
    sdot    z0.d, z24.h, z28.h
    sdot    z1.d, z25.h, z28.h
    sdot    z2.d, z26.h, z28.h
    sdot    z3.d, z27.h, z28.h
    .else
    sdot    z0.s, z24.b, z28.b
    sdot    z1.s, z25.b, z28.b
    sdot    z2.s, z26.b, z28.b
    sdot    z3.s, z27.b, z28.b
    .endif
    .if \rows >= 2
    .if \halfword
    ld1rd {z28.d}, p0/z, [x4]
    .else
    ld1rw {z28.s}, p0/z, [x4]
    .endif
    .if \halfword
    sdot z4.d, z24.h, z28.h
    sdot z5.d, z25.h, z28.h
    sdot z6.d, z26.h, z28.h
    sdot z7.d, z27.h, z28.h
    .else
    sdot z4.s, z24.b, z28.b
    sdot z5.s, z25.b, z28.b
    sdot z6.s, z26.b, z28.b
    sdot z7.s, z27.b, z28.b
    .endif
    .endif
    .if \rows >= 3
    .if \halfword
    ld1rd {z28.d}, p0/z, [x5]
    .else
    ld1rw {z28.s}, p0/z, [x5]
    .endif
    .if \halfword
    sdot z8.d, z24.h, z28.h
    sdot z9.d, z25.h, z28.h
    sdot z10.d, z26.h, z28.h
    sdot z11.d, z27.h, z28.h
    .else
    sdot z8.s, z24.b, z28.b
    sdot z9.s, z25.b, z28.b
    sdot z10.s, z26.b, z28.b
    sdot z11.s, z27.b, z28.b
    .endif
    .endif
    .if \rows >= 4
    .if \halfword
    ld1rd {z28.d}, p0/z, [x6]
    .else
    ld1rw {z28.s}, p0/z, [x6]
    .endif
    .if \halfword
    sdot z12.d, z24.h, z28.h
    sdot z13.d, z25.h, z28.h
    sdot z14.d, z26.h, z28.h
    sdot z15.d, z27.h, z28.h
    .else
    sdot z12.s, z24.b, z28.b
    sdot z13.s, z25.b, z28.b
    sdot z14.s, z26.b, z28.b
    sdot z15.s, z27.b, z28.b
    .endif
    .endif
    .if \rows >= 5
    .if \halfword
    ld1rd {z28.d}, p0/z, [x7]
    .else
    ld1rw {z28.s}, p0/z, [x7]
    .endif
    .if \halfword
    sdot z16.d, z24.h, z28.h
    sdot z17.d, z25.h, z28.h
    sdot z18.d, z26.h, z28.h
    sdot z19.d, z27.h, z28.h
    .else
    sdot z16.s, z24.b, z28.b
    sdot z17.s, z25.b, z28.b
    sdot z18.s, z26.b, z28.b
    sdot z19.s, z27.b, z28.b
    .endif
    .endif
    .if \rows >= 6
    .if \halfword
    ld1rd {z28.d}, p0/z, [x8]
    .else
    ld1rw {z28.s}, p0/z, [x8]
    .endif
    .if \halfword
    sdot z20.d, z24.h, z28.h
    sdot z21.d, z25.h, z28.h
    sdot z22.d, z26.h, z28.h
    sdot z23.d, z27.h, z28.h
    .else
    sdot z20.s, z24.b, z28.b
    sdot z21.s, z25.b, z28.b
    sdot z22.s, z26.b, z28.b
    sdot z23.s, z27.b, z28.b
    .endif
    .endif
    .if \halfword
    add x0, x0, #512
    .else
    add x0, x0, #256
    .endif
    .if \halfword
    add x3, x3, #8
    .if \rows >= 2
    add x4, x4, #8
    .endif
    .if \rows >= 3
    add x5, x5, #8
    .endif
    .if \rows >= 4
    add x6, x6, #8
    .endif
    .if \rows >= 5
    add x7, x7, #8
    .endif
    .if \rows >= 6
    add x8, x8, #8
    .endif
    .else
    add x3, x3, #4
    .if \rows >= 2
    add x4, x4, #4
    .endif
    .if \rows >= 3
    add x5, x5, #4
    .endif
    .if \rows >= 4
    add x6, x6, #4
    .endif
    .if \rows >= 5
    add x7, x7, #4
    .endif
    .if \rows >= 6
    add x8, x8, #4
    .endif
    .endif
    subs    x10, x10, #1
    b.ne    1b
    .if \halfword
    ptrue p1.d
    .else
    ptrue p1.s
    .endif
    .if \halfword
    st1d    {z0.d}, p1, [x2]
    st1d    {z1.d}, p1, [x2, #1, mul vl]
    st1d    {z2.d}, p1, [x2, #2, mul vl]
    st1d    {z3.d}, p1, [x2, #3, mul vl]
    .else
    st1w    {z0.s}, p1, [x2]
    st1w    {z1.s}, p1, [x2, #1, mul vl]
    st1w    {z2.s}, p1, [x2, #2, mul vl]
    st1w    {z3.s}, p1, [x2, #3, mul vl]
    .endif
    .if \rows >= 2
    add x2, x2, #256
    .if \halfword
    st1d {z4.d}, p1, [x2]
    st1d {z5.d}, p1, [x2, #1, mul vl]
    st1d {z6.d}, p1, [x2, #2, mul vl]
    st1d {z7.d}, p1, [x2, #3, mul vl]
    .else
    st1w {z4.s}, p1, [x2]
    st1w {z5.s}, p1, [x2, #1, mul vl]
    st1w {z6.s}, p1, [x2, #2, mul vl]
    st1w {z7.s}, p1, [x2, #3, mul vl]
    .endif
    .endif
    .if \rows >= 3
    add x2, x2, #256
    .if \halfword
    st1d {z8.d}, p1, [x2]
    st1d {z9.d}, p1, [x2, #1, mul vl]
    st1d {z10.d}, p1, [x2, #2, mul vl]
    st1d {z11.d}, p1, [x2, #3, mul vl]
    .else
    st1w {z8.s}, p1, [x2]
    st1w {z9.s}, p1, [x2, #1, mul vl]
    st1w {z10.s}, p1, [x2, #2, mul vl]
    st1w {z11.s}, p1, [x2, #3, mul vl]
    .endif
    .endif
    .if \rows >= 4
    add x2, x2, #256
    .if \halfword
    st1d {z12.d}, p1, [x2]
    st1d {z13.d}, p1, [x2, #1, mul vl]
    st1d {z14.d}, p1, [x2, #2, mul vl]
    st1d {z15.d}, p1, [x2, #3, mul vl]
    .else
    st1w {z12.s}, p1, [x2]
    st1w {z13.s}, p1, [x2, #1, mul vl]
    st1w {z14.s}, p1, [x2, #2, mul vl]
    st1w {z15.s}, p1, [x2, #3, mul vl]
    .endif
    .endif
    .if \rows >= 5
    add x2, x2, #256
    .if \halfword
    st1d {z16.d}, p1, [x2]
    st1d {z17.d}, p1, [x2, #1, mul vl]
    st1d {z18.d}, p1, [x2, #2, mul vl]
    st1d {z19.d}, p1, [x2, #3, mul vl]
    .else
    st1w {z16.s}, p1, [x2]
    st1w {z17.s}, p1, [x2, #1, mul vl]
    st1w {z18.s}, p1, [x2, #2, mul vl]
    st1w {z19.s}, p1, [x2, #3, mul vl]
    .endif
    .endif
    .if \rows >= 6
    add x2, x2, #256
    .if \halfword
    st1d {z20.d}, p1, [x2]
    st1d {z21.d}, p1, [x2, #1, mul vl]
    st1d {z22.d}, p1, [x2, #2, mul vl]
    st1d {z23.d}, p1, [x2, #3, mul vl]
    .else
    st1w {z20.s}, p1, [x2]
    st1w {z21.s}, p1, [x2, #1, mul vl]
    st1w {z22.s}, p1, [x2, #2, mul vl]
    st1w {z23.s}, p1, [x2, #3, mul vl]
    .endif
    .endif
    RESTORE_D8_D15
    ret
    .size \name, .-\name
    .endm

    GEMM_SDOT gemm_i8_m1_sve, 1, 0
    GEMM_SDOT gemm_i8_m2_basic_sve, 2, 0
    GEMM_SDOT gemm_i8_m4_basic_sve, 4, 0
    GEMM_SDOT gemm_i8_m6_basic_sve, 6, 0
    GEMM_SDOT gemm_i16_m1_sve, 1, 1
    GEMM_SDOT gemm_i16_m2_sve, 2, 1
    GEMM_SDOT gemm_i16_m4_sve, 4, 1
    GEMM_SDOT gemm_i16_m6_sve, 6, 1

/*
 * M=1 fused across four consecutive K=128 blocks.  Sixteen independent
 * accumulator vectors hide SDOT latency while each consumer streams four
 * adjacent expanded blocks from its L2 handoff ring.
 */
    .macro M1K4_STEP wp, ap, c0, c1, c2, c3
    ld1b {z16.b}, p0/z, [\wp]
    ld1b {z17.b}, p0/z, [\wp, #1, mul vl]
    ld1b {z18.b}, p0/z, [\wp, #2, mul vl]
    ld1b {z19.b}, p0/z, [\wp, #3, mul vl]
    ld1rw {z20.s}, p0/z, [\ap]
    sdot \c0\().s, z16.b, z20.b
    sdot \c1\().s, z17.b, z20.b
    sdot \c2\().s, z18.b, z20.b
    sdot \c3\().s, z19.b, z20.b
    .endm

    .global gemm_i8_m1_k4_sve
    .type gemm_i8_m1_k4_sve, %function
gemm_i8_m1_k4_sve:
    SAVE_D8_D15
    ptrue p0.b
    mov x11, #8192
    add x4, x0, x11
    add x5, x4, x11
    add x6, x5, x11
    add x7, x1, #128
    add x8, x7, #128
    add x9, x8, #128
    mov x10, #32
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
    .p2align 6
1:
    M1K4_STEP x0, x1, z0, z1, z2, z3
    M1K4_STEP x4, x7, z4, z5, z6, z7
    M1K4_STEP x5, x8, z8, z9, z10, z11
    M1K4_STEP x6, x9, z12, z13, z14, z15
    add x0, x0, #256
    add x4, x4, #256
    add x5, x5, #256
    add x6, x6, #256
    add x1, x1, #4
    add x7, x7, #4
    add x8, x8, #4
    add x9, x9, #4
    subs x10, x10, #1
    b.ne 1b
    ptrue p1.s
    st1w {z0.s}, p1, [x2]
    st1w {z1.s}, p1, [x2, #1, mul vl]
    st1w {z2.s}, p1, [x2, #2, mul vl]
    st1w {z3.s}, p1, [x2, #3, mul vl]
    add x2, x2, #256
    st1w {z4.s}, p1, [x2]
    st1w {z5.s}, p1, [x2, #1, mul vl]
    st1w {z6.s}, p1, [x2, #2, mul vl]
    st1w {z7.s}, p1, [x2, #3, mul vl]
    add x2, x2, #256
    st1w {z8.s}, p1, [x2]
    st1w {z9.s}, p1, [x2, #1, mul vl]
    st1w {z10.s}, p1, [x2, #2, mul vl]
    st1w {z11.s}, p1, [x2, #3, mul vl]
    add x2, x2, #256
    st1w {z12.s}, p1, [x2]
    st1w {z13.s}, p1, [x2, #1, mul vl]
    st1w {z14.s}, p1, [x2, #2, mul vl]
    st1w {z15.s}, p1, [x2, #3, mul vl]
    RESTORE_D8_D15
    ret
    .size gemm_i8_m1_k4_sve, .-gemm_i8_m1_k4_sve

/*
 * True fused INT4 kernels.  Packed weights are read directly, sign-extended
 * in registers, and consumed by SDOT without an expanded-weight store/load.
 * K-block fusion supplies enough independent accumulators to cover SDOT
 * latency while preserving the existing tile-major packed layout.
 */
    .macro FUSED_I4_I8_BLOCK wp, ap, c0, c1, c2, c3
    ld1b    {z16.b}, p0/z, [\wp]
    ld1b    {z17.b}, p0/z, [\wp, #1, mul vl]
    mov     z18.d, z16.d
    mov     z19.d, z17.d
    lsl     z16.b, z16.b, #4
    lsl     z17.b, z17.b, #4
    asr     z16.b, z16.b, #4
    asr     z17.b, z17.b, #4
    asr     z18.b, z18.b, #4
    asr     z19.b, z19.b, #4
    zip1    z21.b, z16.b, z18.b
    zip2    z22.b, z16.b, z18.b
    zip1    z23.b, z17.b, z19.b
    zip2    z24.b, z17.b, z19.b
    ld1rw   {z20.s}, p0/z, [\ap]
    sdot    \c0\().s, z21.b, z20.b
    sdot    \c1\().s, z22.b, z20.b
    sdot    \c2\().s, z23.b, z20.b
    sdot    \c3\().s, z24.b, z20.b
    .endm

    .global fused_i4_i8_m1_k4_sve
    .type fused_i4_i8_m1_k4_sve, %function
fused_i4_i8_m1_k4_sve:
    SAVE_D8_D15
    ptrue   p0.b
    mov     x9, #4096
    add     x3, x0, x9
    add     x4, x3, x9
    add     x5, x4, x9
    add     x6, x1, #128
    add     x7, x6, #128
    add     x8, x7, #128
    eor     z0.d, z0.d, z0.d
    eor     z1.d, z1.d, z1.d
    eor     z2.d, z2.d, z2.d
    eor     z3.d, z3.d, z3.d
    eor     z4.d, z4.d, z4.d
    eor     z5.d, z5.d, z5.d
    eor     z6.d, z6.d, z6.d
    eor     z7.d, z7.d, z7.d
    eor     z8.d, z8.d, z8.d
    eor     z9.d, z9.d, z9.d
    eor     z10.d, z10.d, z10.d
    eor     z11.d, z11.d, z11.d
    eor     z12.d, z12.d, z12.d
    eor     z13.d, z13.d, z13.d
    eor     z14.d, z14.d, z14.d
    eor     z15.d, z15.d, z15.d
    mov     x10, #32
    .p2align 6
1:
    FUSED_I4_I8_BLOCK x0, x1, z0, z1, z2, z3
    FUSED_I4_I8_BLOCK x3, x6, z4, z5, z6, z7
    FUSED_I4_I8_BLOCK x4, x7, z8, z9, z10, z11
    FUSED_I4_I8_BLOCK x5, x8, z12, z13, z14, z15
    add     x0, x0, #128
    add     x3, x3, #128
    add     x4, x4, #128
    add     x5, x5, #128
    add     x1, x1, #4
    add     x6, x6, #4
    add     x7, x7, #4
    add     x8, x8, #4
    subs    x10, x10, #1
    b.ne    1b
    ptrue   p1.s
    st1w    {z0.s}, p1, [x2]
    st1w    {z1.s}, p1, [x2, #1, mul vl]
    st1w    {z2.s}, p1, [x2, #2, mul vl]
    st1w    {z3.s}, p1, [x2, #3, mul vl]
    add     x2, x2, #256
    st1w    {z4.s}, p1, [x2]
    st1w    {z5.s}, p1, [x2, #1, mul vl]
    st1w    {z6.s}, p1, [x2, #2, mul vl]
    st1w    {z7.s}, p1, [x2, #3, mul vl]
    add     x2, x2, #256
    st1w    {z8.s}, p1, [x2]
    st1w    {z9.s}, p1, [x2, #1, mul vl]
    st1w    {z10.s}, p1, [x2, #2, mul vl]
    st1w    {z11.s}, p1, [x2, #3, mul vl]
    add     x2, x2, #256
    st1w    {z12.s}, p1, [x2]
    st1w    {z13.s}, p1, [x2, #1, mul vl]
    st1w    {z14.s}, p1, [x2, #2, mul vl]
    st1w    {z15.s}, p1, [x2, #3, mul vl]
    RESTORE_D8_D15
    ret
    .size fused_i4_i8_m1_k4_sve, .-fused_i4_i8_m1_k4_sve

/* Signed-nibble lookup variant, retaining the consecutive-nibble layout. */
    .macro FUSED_I4_I8_LUT_BLOCK wp, ap, c0, c1, c2, c3
    ld1b    {z16.b}, p0/z, [\wp]
    ld1b    {z17.b}, p0/z, [\wp, #1, mul vl]
    mov     z18.d, z16.d
    mov     z19.d, z17.d
    and     z18.b, z18.b, #0x0f
    and     z19.b, z19.b, #0x0f
    lsr     z16.b, z16.b, #4
    lsr     z17.b, z17.b, #4
    zip1    z20.b, z18.b, z16.b
    zip2    z21.b, z18.b, z16.b
    zip1    z22.b, z19.b, z17.b
    zip2    z23.b, z19.b, z17.b
    tbl     z20.b, {z31.b}, z20.b
    tbl     z21.b, {z31.b}, z21.b
    tbl     z22.b, {z31.b}, z22.b
    tbl     z23.b, {z31.b}, z23.b
    ld1rw   {z24.s}, p0/z, [\ap]
    sdot    \c0\().s, z20.b, z24.b
    sdot    \c1\().s, z21.b, z24.b
    sdot    \c2\().s, z22.b, z24.b
    sdot    \c3\().s, z23.b, z24.b
    .endm

    .global fused_i4_i8_m1_k4_lut_sve
    .type fused_i4_i8_m1_k4_lut_sve, %function
fused_i4_i8_m1_k4_lut_sve:
    SAVE_D8_D15
    ptrue   p0.b
    adr     x11, .Li4_lut
    ld1b    {z31.b}, p0/z, [x11]
    mov     x11, #4096
    add     x3, x0, x11
    add     x4, x3, x11
    add     x5, x4, x11
    add     x6, x1, #128
    add     x7, x6, #128
    add     x8, x7, #128
    eor     z0.d, z0.d, z0.d
    eor     z1.d, z1.d, z1.d
    eor     z2.d, z2.d, z2.d
    eor     z3.d, z3.d, z3.d
    eor     z4.d, z4.d, z4.d
    eor     z5.d, z5.d, z5.d
    eor     z6.d, z6.d, z6.d
    eor     z7.d, z7.d, z7.d
    eor     z8.d, z8.d, z8.d
    eor     z9.d, z9.d, z9.d
    eor     z10.d, z10.d, z10.d
    eor     z11.d, z11.d, z11.d
    eor     z12.d, z12.d, z12.d
    eor     z13.d, z13.d, z13.d
    eor     z14.d, z14.d, z14.d
    eor     z15.d, z15.d, z15.d
    mov     x10, #32
    .p2align 6
1:
    FUSED_I4_I8_LUT_BLOCK x0, x1, z0, z1, z2, z3
    FUSED_I4_I8_LUT_BLOCK x3, x6, z4, z5, z6, z7
    FUSED_I4_I8_LUT_BLOCK x4, x7, z8, z9, z10, z11
    FUSED_I4_I8_LUT_BLOCK x5, x8, z12, z13, z14, z15
    add     x0, x0, #128
    add     x3, x3, #128
    add     x4, x4, #128
    add     x5, x5, #128
    add     x1, x1, #4
    add     x6, x6, #4
    add     x7, x7, #4
    add     x8, x8, #4
    subs    x10, x10, #1
    b.ne    1b
    ptrue   p1.s
    st1w    {z0.s}, p1, [x2]
    st1w    {z1.s}, p1, [x2, #1, mul vl]
    st1w    {z2.s}, p1, [x2, #2, mul vl]
    st1w    {z3.s}, p1, [x2, #3, mul vl]
    add     x2, x2, #256
    st1w    {z4.s}, p1, [x2]
    st1w    {z5.s}, p1, [x2, #1, mul vl]
    st1w    {z6.s}, p1, [x2, #2, mul vl]
    st1w    {z7.s}, p1, [x2, #3, mul vl]
    add     x2, x2, #256
    st1w    {z8.s}, p1, [x2]
    st1w    {z9.s}, p1, [x2, #1, mul vl]
    st1w    {z10.s}, p1, [x2, #2, mul vl]
    st1w    {z11.s}, p1, [x2, #3, mul vl]
    add     x2, x2, #256
    st1w    {z12.s}, p1, [x2]
    st1w    {z13.s}, p1, [x2, #1, mul vl]
    st1w    {z14.s}, p1, [x2, #2, mul vl]
    st1w    {z15.s}, p1, [x2, #3, mul vl]
    RESTORE_D8_D15
    ret
    .size fused_i4_i8_m1_k4_lut_sve, .-fused_i4_i8_m1_k4_lut_sve

/* Two-block expansion bank: expose independent shift/ZIP chains together. */
    .macro FUSED_I4_I8_PAIR wa, aa, a0, a1, a2, a3, wb, ab, b0, b1, b2, b3
    ld1b    {z16.b}, p0/z, [\wa]
    ld1b    {z17.b}, p0/z, [\wa, #1, mul vl]
    ld1b    {z18.b}, p0/z, [\wb]
    ld1b    {z19.b}, p0/z, [\wb, #1, mul vl]
    mov     z20.d, z16.d
    mov     z21.d, z17.d
    mov     z22.d, z18.d
    mov     z23.d, z19.d
    lsl     z16.b, z16.b, #4
    lsl     z17.b, z17.b, #4
    lsl     z18.b, z18.b, #4
    lsl     z19.b, z19.b, #4
    asr     z16.b, z16.b, #4
    asr     z17.b, z17.b, #4
    asr     z18.b, z18.b, #4
    asr     z19.b, z19.b, #4
    asr     z20.b, z20.b, #4
    asr     z21.b, z21.b, #4
    asr     z22.b, z22.b, #4
    asr     z23.b, z23.b, #4
    zip1    z24.b, z16.b, z20.b
    zip2    z25.b, z16.b, z20.b
    zip1    z26.b, z17.b, z21.b
    zip2    z27.b, z17.b, z21.b
    zip1    z28.b, z18.b, z22.b
    zip2    z29.b, z18.b, z22.b
    zip1    z30.b, z19.b, z23.b
    zip2    z31.b, z19.b, z23.b
    ld1rw   {z16.s}, p0/z, [\aa]
    ld1rw   {z17.s}, p0/z, [\ab]
    sdot    \a0\().s, z24.b, z16.b
    sdot    \a1\().s, z25.b, z16.b
    sdot    \a2\().s, z26.b, z16.b
    sdot    \a3\().s, z27.b, z16.b
    sdot    \b0\().s, z28.b, z17.b
    sdot    \b1\().s, z29.b, z17.b
    sdot    \b2\().s, z30.b, z17.b
    sdot    \b3\().s, z31.b, z17.b
    .endm

    .global fused_i4_i8_m1_k4_pipe_sve
    .type fused_i4_i8_m1_k4_pipe_sve, %function
fused_i4_i8_m1_k4_pipe_sve:
    SAVE_D8_D15
    ptrue   p0.b
    mov     x11, #4096
    add     x3, x0, x11
    add     x4, x3, x11
    add     x5, x4, x11
    add     x6, x1, #128
    add     x7, x6, #128
    add     x8, x7, #128
    eor     z0.d, z0.d, z0.d
    eor     z1.d, z1.d, z1.d
    eor     z2.d, z2.d, z2.d
    eor     z3.d, z3.d, z3.d
    eor     z4.d, z4.d, z4.d
    eor     z5.d, z5.d, z5.d
    eor     z6.d, z6.d, z6.d
    eor     z7.d, z7.d, z7.d
    eor     z8.d, z8.d, z8.d
    eor     z9.d, z9.d, z9.d
    eor     z10.d, z10.d, z10.d
    eor     z11.d, z11.d, z11.d
    eor     z12.d, z12.d, z12.d
    eor     z13.d, z13.d, z13.d
    eor     z14.d, z14.d, z14.d
    eor     z15.d, z15.d, z15.d
    mov     x10, #32
    .p2align 6
1:
    FUSED_I4_I8_PAIR x0, x1, z0, z1, z2, z3, x3, x6, z4, z5, z6, z7
    FUSED_I4_I8_PAIR x4, x7, z8, z9, z10, z11, x5, x8, z12, z13, z14, z15
    add     x0, x0, #128
    add     x3, x3, #128
    add     x4, x4, #128
    add     x5, x5, #128
    add     x1, x1, #4
    add     x6, x6, #4
    add     x7, x7, #4
    add     x8, x8, #4
    subs    x10, x10, #1
    b.ne    1b
    ptrue   p1.s
    st1w    {z0.s}, p1, [x2]
    st1w    {z1.s}, p1, [x2, #1, mul vl]
    st1w    {z2.s}, p1, [x2, #2, mul vl]
    st1w    {z3.s}, p1, [x2, #3, mul vl]
    add     x2, x2, #256
    st1w    {z4.s}, p1, [x2]
    st1w    {z5.s}, p1, [x2, #1, mul vl]
    st1w    {z6.s}, p1, [x2, #2, mul vl]
    st1w    {z7.s}, p1, [x2, #3, mul vl]
    add     x2, x2, #256
    st1w    {z8.s}, p1, [x2]
    st1w    {z9.s}, p1, [x2, #1, mul vl]
    st1w    {z10.s}, p1, [x2, #2, mul vl]
    st1w    {z11.s}, p1, [x2, #3, mul vl]
    add     x2, x2, #256
    st1w    {z12.s}, p1, [x2]
    st1w    {z13.s}, p1, [x2, #1, mul vl]
    st1w    {z14.s}, p1, [x2, #2, mul vl]
    st1w    {z15.s}, p1, [x2, #3, mul vl]
    RESTORE_D8_D15
    ret
    .size fused_i4_i8_m1_k4_pipe_sve, .-fused_i4_i8_m1_k4_pipe_sve

/* Four-block K-major supertile: one contiguous 512-byte stream per K quartet. */
    .global fused_i4_i8_m1_k4_super_sve
    .type fused_i4_i8_m1_k4_super_sve, %function
fused_i4_i8_m1_k4_super_sve:
    SAVE_D8_D15
    ptrue   p0.b
    add     x3, x0, #128
    add     x4, x0, #256
    add     x5, x0, #384
    add     x6, x1, #128
    add     x7, x6, #128
    add     x8, x7, #128
    eor     z0.d, z0.d, z0.d
    eor     z1.d, z1.d, z1.d
    eor     z2.d, z2.d, z2.d
    eor     z3.d, z3.d, z3.d
    eor     z4.d, z4.d, z4.d
    eor     z5.d, z5.d, z5.d
    eor     z6.d, z6.d, z6.d
    eor     z7.d, z7.d, z7.d
    eor     z8.d, z8.d, z8.d
    eor     z9.d, z9.d, z9.d
    eor     z10.d, z10.d, z10.d
    eor     z11.d, z11.d, z11.d
    eor     z12.d, z12.d, z12.d
    eor     z13.d, z13.d, z13.d
    eor     z14.d, z14.d, z14.d
    eor     z15.d, z15.d, z15.d
    mov     x10, #32
    .p2align 6
1:
    FUSED_I4_I8_BLOCK x0, x1, z0, z1, z2, z3
    FUSED_I4_I8_BLOCK x3, x6, z4, z5, z6, z7
    FUSED_I4_I8_BLOCK x4, x7, z8, z9, z10, z11
    FUSED_I4_I8_BLOCK x5, x8, z12, z13, z14, z15
    add     x0, x0, #512
    add     x3, x3, #512
    add     x4, x4, #512
    add     x5, x5, #512
    add     x1, x1, #4
    add     x6, x6, #4
    add     x7, x7, #4
    add     x8, x8, #4
    subs    x10, x10, #1
    b.ne    1b
    ptrue   p1.s
    st1w    {z0.s}, p1, [x2]
    st1w    {z1.s}, p1, [x2, #1, mul vl]
    st1w    {z2.s}, p1, [x2, #2, mul vl]
    st1w    {z3.s}, p1, [x2, #3, mul vl]
    add     x2, x2, #256
    st1w    {z4.s}, p1, [x2]
    st1w    {z5.s}, p1, [x2, #1, mul vl]
    st1w    {z6.s}, p1, [x2, #2, mul vl]
    st1w    {z7.s}, p1, [x2, #3, mul vl]
    add     x2, x2, #256
    st1w    {z8.s}, p1, [x2]
    st1w    {z9.s}, p1, [x2, #1, mul vl]
    st1w    {z10.s}, p1, [x2, #2, mul vl]
    st1w    {z11.s}, p1, [x2, #3, mul vl]
    add     x2, x2, #256
    st1w    {z12.s}, p1, [x2]
    st1w    {z13.s}, p1, [x2, #1, mul vl]
    st1w    {z14.s}, p1, [x2, #2, mul vl]
    st1w    {z15.s}, p1, [x2, #3, mul vl]
    RESTORE_D8_D15
    ret
    .size fused_i4_i8_m1_k4_super_sve, .-fused_i4_i8_m1_k4_super_sve

    .macro FUSED_I4_I16_HALF wp, ap, c0, c1, c2, c3
    ld1b    {z16.b}, p0/z, [\wp]
    mov     z17.d, z16.d
    lsl     z16.b, z16.b, #4
    asr     z16.b, z16.b, #4
    asr     z17.b, z17.b, #4
    sunpklo z20.h, z16.b
    sunpkhi z21.h, z16.b
    sunpklo z22.h, z17.b
    sunpkhi z23.h, z17.b
    ld1rd   {z24.d}, p0/z, [\ap]
    sdot    \c0\().d, z20.h, z24.h
    sdot    \c1\().d, z21.h, z24.h
    sdot    \c2\().d, z22.h, z24.h
    sdot    \c3\().d, z23.h, z24.h
    .endm

    .global fused_i4_i16_m1_k2_sve
    .type fused_i4_i16_m1_k2_sve, %function
fused_i4_i16_m1_k2_sve:
    SAVE_D8_D15
    ptrue   p0.b
    mov     x5, #4096
    add     x3, x0, x5
    add     x4, x1, #256
    eor     z0.d, z0.d, z0.d
    eor     z1.d, z1.d, z1.d
    eor     z2.d, z2.d, z2.d
    eor     z3.d, z3.d, z3.d
    eor     z4.d, z4.d, z4.d
    eor     z5.d, z5.d, z5.d
    eor     z6.d, z6.d, z6.d
    eor     z7.d, z7.d, z7.d
    eor     z8.d, z8.d, z8.d
    eor     z9.d, z9.d, z9.d
    eor     z10.d, z10.d, z10.d
    eor     z11.d, z11.d, z11.d
    eor     z12.d, z12.d, z12.d
    eor     z13.d, z13.d, z13.d
    eor     z14.d, z14.d, z14.d
    eor     z15.d, z15.d, z15.d
    mov     x10, #32
    .p2align 6
1:
    FUSED_I4_I16_HALF x0, x1, z0, z1, z2, z3
    add     x5, x0, #64
    FUSED_I4_I16_HALF x5, x1, z4, z5, z6, z7
    FUSED_I4_I16_HALF x3, x4, z8, z9, z10, z11
    add     x5, x3, #64
    FUSED_I4_I16_HALF x5, x4, z12, z13, z14, z15
    add     x0, x0, #128
    add     x3, x3, #128
    add     x1, x1, #8
    add     x4, x4, #8
    nop
    nop
    subs    x10, x10, #1
    b.ne    1b
    ptrue   p1.d
    st1d    {z0.d}, p1, [x2]
    st1d    {z1.d}, p1, [x2, #1, mul vl]
    st1d    {z2.d}, p1, [x2, #2, mul vl]
    st1d    {z3.d}, p1, [x2, #3, mul vl]
    add     x2, x2, #256
    st1d    {z4.d}, p1, [x2]
    st1d    {z5.d}, p1, [x2, #1, mul vl]
    st1d    {z6.d}, p1, [x2, #2, mul vl]
    st1d    {z7.d}, p1, [x2, #3, mul vl]
    add     x2, x2, #256
    st1d    {z8.d}, p1, [x2]
    st1d    {z9.d}, p1, [x2, #1, mul vl]
    st1d    {z10.d}, p1, [x2, #2, mul vl]
    st1d    {z11.d}, p1, [x2, #3, mul vl]
    add     x2, x2, #256
    st1d    {z12.d}, p1, [x2]
    st1d    {z13.d}, p1, [x2, #1, mul vl]
    st1d    {z14.d}, p1, [x2, #2, mul vl]
    st1d    {z15.d}, p1, [x2, #3, mul vl]
    RESTORE_D8_D15
    ret
    .size fused_i4_i16_m1_k2_sve, .-fused_i4_i16_m1_k2_sve

/*
 * M=6 INT8 schedule: fetch activation scalars into GPRs, then broadcast four
 * at once. This separates the L1 scalar-load latency from SDOT consumption
 * while retaining four resident weight vectors and 24 accumulators.
 */
    .global gemm_i8_m6_gpr_sve
    .type gemm_i8_m6_gpr_sve, %function
gemm_i8_m6_gpr_sve:
    SAVE_D8_D15
    ptrue p0.b
    mov x3, x1
    add x4, x3, #128
    add x5, x4, #128
    add x6, x5, #128
    add x7, x6, #128
    add x8, x7, #128
    mov x10, #32
    eor z8.d, z8.d, z8.d
    eor z9.d, z9.d, z9.d
    eor z10.d, z10.d, z10.d
    eor z11.d, z11.d, z11.d
    eor z12.d, z12.d, z12.d
    eor z13.d, z13.d, z13.d
    eor z14.d, z14.d, z14.d
    eor z15.d, z15.d, z15.d
    eor z16.d, z16.d, z16.d
    eor z17.d, z17.d, z17.d
    eor z18.d, z18.d, z18.d
    eor z19.d, z19.d, z19.d
    eor z20.d, z20.d, z20.d
    eor z21.d, z21.d, z21.d
    eor z22.d, z22.d, z22.d
    eor z23.d, z23.d, z23.d
1:
    ldr w12, [x3]
    ldr w13, [x4]
    ldr w14, [x5]
    ldr w15, [x6]
    ldr w16, [x7]
    ldr w17, [x8]
    dup z28.s, w12
    dup z29.s, w13
    dup z30.s, w14
    dup z31.s, w15
    ld1b {z24.b}, p0/z, [x0]
    ld1b {z25.b}, p0/z, [x0, #1, mul vl]
    ld1b {z26.b}, p0/z, [x0, #2, mul vl]
    ld1b {z27.b}, p0/z, [x0, #3, mul vl]
    sdot z0.s, z24.b, z28.b
    sdot z1.s, z25.b, z28.b
    sdot z2.s, z26.b, z28.b
    sdot z3.s, z27.b, z28.b
    sdot z4.s, z24.b, z29.b
    sdot z5.s, z25.b, z29.b
    sdot z6.s, z26.b, z29.b
    sdot z7.s, z27.b, z29.b
    sdot z8.s, z24.b, z30.b
    sdot z9.s, z25.b, z30.b
    sdot z10.s, z26.b, z30.b
    sdot z11.s, z27.b, z30.b
    sdot z12.s, z24.b, z31.b
    sdot z13.s, z25.b, z31.b
    sdot z14.s, z26.b, z31.b
    sdot z15.s, z27.b, z31.b
    dup z28.s, w16
    dup z29.s, w17
    sdot z16.s, z24.b, z28.b
    sdot z17.s, z25.b, z28.b
    sdot z18.s, z26.b, z28.b
    sdot z19.s, z27.b, z28.b
    sdot z20.s, z24.b, z29.b
    sdot z21.s, z25.b, z29.b
    sdot z22.s, z26.b, z29.b
    sdot z23.s, z27.b, z29.b
    add x0, x0, #256
    add x3, x3, #4
    add x4, x4, #4
    add x5, x5, #4
    add x6, x6, #4
    add x7, x7, #4
    add x8, x8, #4
    subs x10, x10, #1
    b.ne 1b
    ptrue p1.s
    st1w {z0.s}, p1, [x2]
    st1w {z1.s}, p1, [x2, #1, mul vl]
    st1w {z2.s}, p1, [x2, #2, mul vl]
    st1w {z3.s}, p1, [x2, #3, mul vl]
    add x2, x2, #256
    st1w {z4.s}, p1, [x2]
    st1w {z5.s}, p1, [x2, #1, mul vl]
    st1w {z6.s}, p1, [x2, #2, mul vl]
    st1w {z7.s}, p1, [x2, #3, mul vl]
    add x2, x2, #256
    st1w {z8.s}, p1, [x2]
    st1w {z9.s}, p1, [x2, #1, mul vl]
    st1w {z10.s}, p1, [x2, #2, mul vl]
    st1w {z11.s}, p1, [x2, #3, mul vl]
    add x2, x2, #256
    st1w {z12.s}, p1, [x2]
    st1w {z13.s}, p1, [x2, #1, mul vl]
    st1w {z14.s}, p1, [x2, #2, mul vl]
    st1w {z15.s}, p1, [x2, #3, mul vl]
    add x2, x2, #256
    st1w {z16.s}, p1, [x2]
    st1w {z17.s}, p1, [x2, #1, mul vl]
    st1w {z18.s}, p1, [x2, #2, mul vl]
    st1w {z19.s}, p1, [x2, #3, mul vl]
    add x2, x2, #256
    st1w {z20.s}, p1, [x2]
    st1w {z21.s}, p1, [x2, #1, mul vl]
    st1w {z22.s}, p1, [x2, #2, mul vl]
    st1w {z23.s}, p1, [x2, #3, mul vl]
    RESTORE_D8_D15
    ret
    .size gemm_i8_m6_gpr_sve, .-gemm_i8_m6_gpr_sve

    .macro SDOT_M4 wa, wb, wc, wd, a0, a1, a2, a3
    sdot z0.s, \wa\().b, \a0\().b
    sdot z1.s, \wb\().b, \a0\().b
    sdot z2.s, \wc\().b, \a0\().b
    sdot z3.s, \wd\().b, \a0\().b
    sdot z4.s, \wa\().b, \a1\().b
    sdot z5.s, \wb\().b, \a1\().b
    sdot z6.s, \wc\().b, \a1\().b
    sdot z7.s, \wd\().b, \a1\().b
    sdot z8.s, \wa\().b, \a2\().b
    sdot z9.s, \wb\().b, \a2\().b
    sdot z10.s, \wc\().b, \a2\().b
    sdot z11.s, \wd\().b, \a2\().b
    sdot z12.s, \wa\().b, \a3\().b
    sdot z13.s, \wb\().b, \a3\().b
    sdot z14.s, \wc\().b, \a3\().b
    sdot z15.s, \wd\().b, \a3\().b
    .endm

    .global gemm_i8_m4_sve
    .type gemm_i8_m4_sve, %function
gemm_i8_m4_sve:
    SAVE_D8_D15
    ptrue p0.b
    mov x3, x1
    add x4, x3, #128
    add x5, x4, #128
    add x6, x5, #128
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
    ld1b {z16.b}, p0/z, [x0]
    ld1b {z17.b}, p0/z, [x0, #1, mul vl]
    ld1b {z18.b}, p0/z, [x0, #2, mul vl]
    ld1b {z19.b}, p0/z, [x0, #3, mul vl]
    ld1rw {z24.s}, p0/z, [x3]
    ld1rw {z25.s}, p0/z, [x4]
    ld1rw {z26.s}, p0/z, [x5]
    ld1rw {z27.s}, p0/z, [x6]
    mov x10, #15
1:
    add x9, x0, #256
    ld1b {z20.b}, p0/z, [x9]
    ld1b {z21.b}, p0/z, [x9, #1, mul vl]
    ld1b {z22.b}, p0/z, [x9, #2, mul vl]
    ld1b {z23.b}, p0/z, [x9, #3, mul vl]
    ld1rw {z28.s}, p0/z, [x3, #4]
    ld1rw {z29.s}, p0/z, [x4, #4]
    ld1rw {z30.s}, p0/z, [x5, #4]
    ld1rw {z31.s}, p0/z, [x6, #4]
    SDOT_M4 z16, z17, z18, z19, z24, z25, z26, z27
    add x9, x0, #512
    ld1b {z16.b}, p0/z, [x9]
    ld1b {z17.b}, p0/z, [x9, #1, mul vl]
    ld1b {z18.b}, p0/z, [x9, #2, mul vl]
    ld1b {z19.b}, p0/z, [x9, #3, mul vl]
    ld1rw {z24.s}, p0/z, [x3, #8]
    ld1rw {z25.s}, p0/z, [x4, #8]
    ld1rw {z26.s}, p0/z, [x5, #8]
    ld1rw {z27.s}, p0/z, [x6, #8]
    SDOT_M4 z20, z21, z22, z23, z28, z29, z30, z31
    add x0, x0, #512
    add x3, x3, #8
    add x4, x4, #8
    add x5, x5, #8
    add x6, x6, #8
    subs x10, x10, #1
    b.ne 1b
    add x9, x0, #256
    ld1b {z20.b}, p0/z, [x9]
    ld1b {z21.b}, p0/z, [x9, #1, mul vl]
    ld1b {z22.b}, p0/z, [x9, #2, mul vl]
    ld1b {z23.b}, p0/z, [x9, #3, mul vl]
    ld1rw {z28.s}, p0/z, [x3, #4]
    ld1rw {z29.s}, p0/z, [x4, #4]
    ld1rw {z30.s}, p0/z, [x5, #4]
    ld1rw {z31.s}, p0/z, [x6, #4]
    SDOT_M4 z16, z17, z18, z19, z24, z25, z26, z27
    SDOT_M4 z20, z21, z22, z23, z28, z29, z30, z31
    ptrue p1.s
    st1w {z0.s}, p1, [x2]
    st1w {z1.s}, p1, [x2, #1, mul vl]
    st1w {z2.s}, p1, [x2, #2, mul vl]
    st1w {z3.s}, p1, [x2, #3, mul vl]
    add x2, x2, #256
    st1w {z4.s}, p1, [x2]
    st1w {z5.s}, p1, [x2, #1, mul vl]
    st1w {z6.s}, p1, [x2, #2, mul vl]
    st1w {z7.s}, p1, [x2, #3, mul vl]
    add x2, x2, #256
    st1w {z8.s}, p1, [x2]
    st1w {z9.s}, p1, [x2, #1, mul vl]
    st1w {z10.s}, p1, [x2, #2, mul vl]
    st1w {z11.s}, p1, [x2, #3, mul vl]
    add x2, x2, #256
    st1w {z12.s}, p1, [x2]
    st1w {z13.s}, p1, [x2, #1, mul vl]
    st1w {z14.s}, p1, [x2, #2, mul vl]
    st1w {z15.s}, p1, [x2, #3, mul vl]
    RESTORE_D8_D15
    ret
    .size gemm_i8_m4_sve, .-gemm_i8_m4_sve

    .macro SDOT_M2 wa, wb, wc, wd, a0, a1
    sdot z0.s, \wa\().b, \a0\().b
    sdot z1.s, \wb\().b, \a0\().b
    sdot z2.s, \wc\().b, \a0\().b
    sdot z3.s, \wd\().b, \a0\().b
    sdot z4.s, \wa\().b, \a1\().b
    sdot z5.s, \wb\().b, \a1\().b
    sdot z6.s, \wc\().b, \a1\().b
    sdot z7.s, \wd\().b, \a1\().b
    .endm

    .global gemm_i8_m2_sve
    .type gemm_i8_m2_sve, %function
gemm_i8_m2_sve:
    SAVE_D8_D15
    ptrue p0.b
    mov x3, x1
    add x4, x3, #128
    eor z0.d, z0.d, z0.d
    eor z1.d, z1.d, z1.d
    eor z2.d, z2.d, z2.d
    eor z3.d, z3.d, z3.d
    eor z4.d, z4.d, z4.d
    eor z5.d, z5.d, z5.d
    eor z6.d, z6.d, z6.d
    eor z7.d, z7.d, z7.d
    ld1b {z8.b}, p0/z, [x0]
    ld1b {z9.b}, p0/z, [x0, #1, mul vl]
    ld1b {z10.b}, p0/z, [x0, #2, mul vl]
    ld1b {z11.b}, p0/z, [x0, #3, mul vl]
    ld1rw {z16.s}, p0/z, [x3]
    ld1rw {z17.s}, p0/z, [x4]
    mov x10, #15
1:
    add x9, x0, #256
    ld1b {z12.b}, p0/z, [x9]
    ld1b {z13.b}, p0/z, [x9, #1, mul vl]
    ld1b {z14.b}, p0/z, [x9, #2, mul vl]
    ld1b {z15.b}, p0/z, [x9, #3, mul vl]
    ld1rw {z18.s}, p0/z, [x3, #4]
    ld1rw {z19.s}, p0/z, [x4, #4]
    SDOT_M2 z8, z9, z10, z11, z16, z17
    add x9, x0, #512
    ld1b {z8.b}, p0/z, [x9]
    ld1b {z9.b}, p0/z, [x9, #1, mul vl]
    ld1b {z10.b}, p0/z, [x9, #2, mul vl]
    ld1b {z11.b}, p0/z, [x9, #3, mul vl]
    ld1rw {z16.s}, p0/z, [x3, #8]
    ld1rw {z17.s}, p0/z, [x4, #8]
    SDOT_M2 z12, z13, z14, z15, z18, z19
    add x0, x0, #512
    add x3, x3, #8
    add x4, x4, #8
    subs x10, x10, #1
    b.ne 1b
    add x9, x0, #256
    ld1b {z12.b}, p0/z, [x9]
    ld1b {z13.b}, p0/z, [x9, #1, mul vl]
    ld1b {z14.b}, p0/z, [x9, #2, mul vl]
    ld1b {z15.b}, p0/z, [x9, #3, mul vl]
    ld1rw {z18.s}, p0/z, [x3, #4]
    ld1rw {z19.s}, p0/z, [x4, #4]
    SDOT_M2 z8, z9, z10, z11, z16, z17
    SDOT_M2 z12, z13, z14, z15, z18, z19
    ptrue p1.s
    st1w {z0.s}, p1, [x2]
    st1w {z1.s}, p1, [x2, #1, mul vl]
    st1w {z2.s}, p1, [x2, #2, mul vl]
    st1w {z3.s}, p1, [x2, #3, mul vl]
    add x2, x2, #256
    st1w {z4.s}, p1, [x2]
    st1w {z5.s}, p1, [x2, #1, mul vl]
    st1w {z6.s}, p1, [x2, #2, mul vl]
    st1w {z7.s}, p1, [x2, #3, mul vl]
    RESTORE_D8_D15
    ret
    .size gemm_i8_m2_sve, .-gemm_i8_m2_sve

    .global gemm_i8_m6_split_sve
    .type gemm_i8_m6_split_sve, %function
gemm_i8_m6_split_sve:
    stp x19, x20, [sp, #-32]!
    stp x21, x30, [sp, #16]
    mov x19, x0
    mov x20, x1
    mov x21, x2
    bl gemm_i8_m4_sve
    mov x0, x19
    add x1, x20, #512
    add x2, x21, #1024
    bl gemm_i8_m2_sve
    ldp x21, x30, [sp, #16]
    ldp x19, x20, [sp], #32
    ret
    .size gemm_i8_m6_split_sve, .-gemm_i8_m6_split_sve

/*
 * M=6 INT8 schedule. Activations are interleaved by K quartet.  Each aligned
 * iteration handles two K groups and overlaps the next four weight loads with
 * the last row of current SDOTs.  The final pair does not read past the tile.
 */
    .macro M6_ROWS0_4 abase, a0, a1, a2, a3, a4, a5
    ld1rw {z28.s}, p0/z, [\abase, #\a0]
    ld1rw {z29.s}, p0/z, [\abase, #\a1]
    ld1rw {z30.s}, p0/z, [\abase, #\a4]
    ld1rw {z31.s}, p0/z, [\abase, #\a5]
    sdot z0.s, z24.b, z28.b
    sdot z1.s, z25.b, z28.b
    sdot z2.s, z26.b, z28.b
    sdot z3.s, z27.b, z28.b
    ld1rw {z28.s}, p0/z, [\abase, #\a2]
    sdot z4.s, z24.b, z29.b
    sdot z5.s, z25.b, z29.b
    sdot z6.s, z26.b, z29.b
    sdot z7.s, z27.b, z29.b
    ld1rw {z29.s}, p0/z, [\abase, #\a3]
    sdot z8.s, z24.b, z28.b
    sdot z9.s, z25.b, z28.b
    sdot z10.s, z26.b, z28.b
    sdot z11.s, z27.b, z28.b
    sdot z12.s, z24.b, z29.b
    sdot z13.s, z25.b, z29.b
    sdot z14.s, z26.b, z29.b
    sdot z15.s, z27.b, z29.b
    sdot z16.s, z24.b, z30.b
    sdot z17.s, z25.b, z30.b
    sdot z18.s, z26.b, z30.b
    sdot z19.s, z27.b, z30.b
    .endm

    .macro M6_ROW5_NEXT wbase, w0, w1, w2, w3
    sdot z20.s, z24.b, z31.b
    ld1b {z24.b}, p0/z, [\wbase, #\w0, mul vl]
    sdot z21.s, z25.b, z31.b
    ld1b {z25.b}, p0/z, [\wbase, #\w1, mul vl]
    sdot z22.s, z26.b, z31.b
    ld1b {z26.b}, p0/z, [\wbase, #\w2, mul vl]
    sdot z23.s, z27.b, z31.b
    ld1b {z27.b}, p0/z, [\wbase, #\w3, mul vl]
    .endm

    .macro M6_ROW5_FINAL
    sdot z20.s, z24.b, z31.b
    sdot z21.s, z25.b, z31.b
    sdot z22.s, z26.b, z31.b
    sdot z23.s, z27.b, z31.b
    .endm

    .global gemm_i8_m6_sve
    .type gemm_i8_m6_sve, %function
gemm_i8_m6_sve:
    mov x3, #32
    b gemm_i8_m6_stream_sve
    .size gemm_i8_m6_sve, .-gemm_i8_m6_sve

    .global gemm_i8_m6_stream_sve
    .type gemm_i8_m6_stream_sve, %function
gemm_i8_m6_stream_sve:
    SAVE_D8_D15
    ptrue p0.b
    lsr x10, x3, #1
    subs x10, x10, #1
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
    eor z16.d, z16.d, z16.d
    eor z17.d, z17.d, z17.d
    eor z18.d, z18.d, z18.d
    eor z19.d, z19.d, z19.d
    eor z20.d, z20.d, z20.d
    eor z21.d, z21.d, z21.d
    eor z22.d, z22.d, z22.d
    eor z23.d, z23.d, z23.d
    ld1b {z24.b}, p0/z, [x0]
    ld1b {z25.b}, p0/z, [x0, #1, mul vl]
    ld1b {z26.b}, p0/z, [x0, #2, mul vl]
    ld1b {z27.b}, p0/z, [x0, #3, mul vl]
    .p2align 6
1:
    M6_ROWS0_4 x1, 0, 4, 8, 12, 16, 20
    M6_ROW5_NEXT x0, 4, 5, 6, 7
    M6_ROWS0_4 x1, 24, 28, 32, 36, 40, 44
    add x0, x0, #512
    add x1, x1, #48
    M6_ROW5_NEXT x0, 0, 1, 2, 3
    subs x10, x10, #1
    b.ne 1b

    /* Final pair: preload its second quartet, then stop before over-reading. */
    M6_ROWS0_4 x1, 0, 4, 8, 12, 16, 20
    M6_ROW5_NEXT x0, 4, 5, 6, 7
    M6_ROWS0_4 x1, 24, 28, 32, 36, 40, 44
    M6_ROW5_FINAL
    ptrue p1.s
    st1w {z0.s}, p1, [x2]
    st1w {z1.s}, p1, [x2, #1, mul vl]
    st1w {z2.s}, p1, [x2, #2, mul vl]
    st1w {z3.s}, p1, [x2, #3, mul vl]
    add x2, x2, #256
    st1w {z4.s}, p1, [x2]
    st1w {z5.s}, p1, [x2, #1, mul vl]
    st1w {z6.s}, p1, [x2, #2, mul vl]
    st1w {z7.s}, p1, [x2, #3, mul vl]
    add x2, x2, #256
    st1w {z8.s}, p1, [x2]
    st1w {z9.s}, p1, [x2, #1, mul vl]
    st1w {z10.s}, p1, [x2, #2, mul vl]
    st1w {z11.s}, p1, [x2, #3, mul vl]
    add x2, x2, #256
    st1w {z12.s}, p1, [x2]
    st1w {z13.s}, p1, [x2, #1, mul vl]
    st1w {z14.s}, p1, [x2, #2, mul vl]
    st1w {z15.s}, p1, [x2, #3, mul vl]
    add x2, x2, #256
    st1w {z16.s}, p1, [x2]
    st1w {z17.s}, p1, [x2, #1, mul vl]
    st1w {z18.s}, p1, [x2, #2, mul vl]
    st1w {z19.s}, p1, [x2, #3, mul vl]
    add x2, x2, #256
    st1w {z20.s}, p1, [x2]
    st1w {z21.s}, p1, [x2, #1, mul vl]
    st1w {z22.s}, p1, [x2, #2, mul vl]
    st1w {z23.s}, p1, [x2, #3, mul vl]
    RESTORE_D8_D15
    ret
    .size gemm_i8_m6_stream_sve, .-gemm_i8_m6_stream_sve

/* Mx64 by K=128 FP16 FMLA; output remains FP16 for the block reduction. */
    .macro GEMM_F16 name, rows
    .global \name
    .type \name, %function
\name:
    SAVE_D8_D15
    ptrue p0.h
    mov x10, #128
    mov x3, x1
    .if \rows >= 2
    add x4, x3, #256
    .endif
    .if \rows >= 3
    add x5, x4, #256
    .endif
    .if \rows >= 4
    add x6, x5, #256
    .endif
    .if \rows >= 5
    add x7, x6, #256
    .endif
    .if \rows >= 6
    add x8, x7, #256
    .endif
    eor z0.d, z0.d, z0.d
    eor z1.d, z1.d, z1.d
    .if \rows >= 2
    eor z2.d, z2.d, z2.d
    eor z3.d, z3.d, z3.d
    .endif
    .if \rows >= 3
    eor z4.d, z4.d, z4.d
    eor z5.d, z5.d, z5.d
    .endif
    .if \rows >= 4
    eor z6.d, z6.d, z6.d
    eor z7.d, z7.d, z7.d
    .endif
    .if \rows >= 5
    eor z8.d, z8.d, z8.d
    eor z9.d, z9.d, z9.d
    .endif
    .if \rows >= 6
    eor z10.d, z10.d, z10.d
    eor z11.d, z11.d, z11.d
    .endif
1:
    ld1h {z24.h}, p0/z, [x0]
    ld1h {z25.h}, p0/z, [x0, #1, mul vl]
    ld1rh {z28.h}, p0/z, [x3]
    fmla z0.h, p0/m, z24.h, z28.h
    fmla z1.h, p0/m, z25.h, z28.h
    .if \rows >= 2
    ld1rh {z28.h}, p0/z, [x4]
    fmla z2.h, p0/m, z24.h, z28.h
    fmla z3.h, p0/m, z25.h, z28.h
    .endif
    .if \rows >= 3
    ld1rh {z28.h}, p0/z, [x5]
    fmla z4.h, p0/m, z24.h, z28.h
    fmla z5.h, p0/m, z25.h, z28.h
    .endif
    .if \rows >= 4
    ld1rh {z28.h}, p0/z, [x6]
    fmla z6.h, p0/m, z24.h, z28.h
    fmla z7.h, p0/m, z25.h, z28.h
    .endif
    .if \rows >= 5
    ld1rh {z28.h}, p0/z, [x7]
    fmla z8.h, p0/m, z24.h, z28.h
    fmla z9.h, p0/m, z25.h, z28.h
    .endif
    .if \rows >= 6
    ld1rh {z28.h}, p0/z, [x8]
    fmla z10.h, p0/m, z24.h, z28.h
    fmla z11.h, p0/m, z25.h, z28.h
    .endif
    add x0, x0, #128
    add x3, x3, #2
    .if \rows >= 2
    add x4, x4, #2
    .endif
    .if \rows >= 3
    add x5, x5, #2
    .endif
    .if \rows >= 4
    add x6, x6, #2
    .endif
    .if \rows >= 5
    add x7, x7, #2
    .endif
    .if \rows >= 6
    add x8, x8, #2
    .endif
    subs x10, x10, #1
    b.ne 1b
    st1h {z0.h}, p0, [x2]
    st1h {z1.h}, p0, [x2, #1, mul vl]
    .if \rows >= 2
    add x2, x2, #128
    st1h {z2.h}, p0, [x2]
    st1h {z3.h}, p0, [x2, #1, mul vl]
    .endif
    .if \rows >= 3
    add x2, x2, #128
    st1h {z4.h}, p0, [x2]
    st1h {z5.h}, p0, [x2, #1, mul vl]
    .endif
    .if \rows >= 4
    add x2, x2, #128
    st1h {z6.h}, p0, [x2]
    st1h {z7.h}, p0, [x2, #1, mul vl]
    .endif
    .if \rows >= 5
    add x2, x2, #128
    st1h {z8.h}, p0, [x2]
    st1h {z9.h}, p0, [x2, #1, mul vl]
    .endif
    .if \rows >= 6
    add x2, x2, #128
    st1h {z10.h}, p0, [x2]
    st1h {z11.h}, p0, [x2, #1, mul vl]
    .endif
    RESTORE_D8_D15
    ret
    .size \name, .-\name
    .endm

    GEMM_F16 gemm_f16_m1_sve, 1
    GEMM_F16 gemm_f16_m2_sve, 2
    GEMM_F16 gemm_f16_m4_sve, 4
    GEMM_F16 gemm_f16_m6_sve, 6

    .section .rodata
    .align 6
.Lfp4_lut:
    .byte 0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12
    .rept 3
    .byte 0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12
    .endr
.Li4_lut:
    .rept 4
    .byte 0, 1, 2, 3, 4, 5, 6, 7, -8, -7, -6, -5, -4, -3, -2, -1
    .endr

    .section .note.GNU-stack,"",%progbits
