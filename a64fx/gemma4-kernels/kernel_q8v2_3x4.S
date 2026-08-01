// kernel_q8v2_3x4.S - per-block (Q8v2) int8 GEMM microkernel for A64FX SVE.
// Tile: MR=3 token rows x NR=64 weight cols (4 SVE vectors of 16).
//
//   C[r,c] = sum_b  ad[b,r] * dw[b,c] * < qa[b,r,:], qw[b,c,:] >   (BLK=32 per b)
//
// Per 32-wide K block: accumulate int32 partials (z12..z23) via SDOT, then
// scvtf + scale by dw (per-col vec) and ad (per-row scalar) into fp32
// accumulators (z0..z11) that persist across all blocks. 3 A-row broadcasts
// (z24..z26) reused across all 4 B-vecs (z27..z30).
//
// Register map:
//   z0..z11  : 12 fp32 accumulators  f[row][vec]  (persist across K)
//   z12..z23 : 12 int32 partials     p[row][vec]  (per block)
//   z24..z26 : A row broadcasts / (reused as da broadcasts in scale-out)
//   z27..z30 : B col vectors        / (reused as dw vectors in scale-out)
//   z31      : scratch
//   p0.b all-true (bytes), p1.s all-true (words)
//
// Args:
//   x0 = aq  : [nb][3][32] int8  activations (row-major per block)
//   x1 = ad  : [nb][3]     f32   per-row per-block act scale
//   x2 = bq  : [nb][8][4][64] int8 weights (kgroup,vec,col*4byte)
//   x3 = bd  : [nb][64]    f32   per-col per-block weight scale (4 vec x 16)
//   x4 = nb  : number of 32-wide K blocks
//   x5 = C   : f32 output tile base (row 0)
//   x6 = ldc : C row stride in BYTES

    .arch armv8.2-a+sve
    .text
    .align 6
    .global kernel_q8v2_3x4
    .type kernel_q8v2_3x4, @function

kernel_q8v2_3x4:
    // save callee-saved d8-d15 (low 64b of z8-z15)
    stp     d8, d9,  [sp, #-16]!
    stp     d10, d11, [sp, #-16]!
    stp     d12, d13, [sp, #-16]!
    stp     d14, d15, [sp, #-16]!

    ptrue   p0.b
    ptrue   p1.s

    // zero fp32 accumulators z0..z11
    dup     z0.s, #0
    dup     z1.s, #0
    dup     z2.s, #0
    dup     z3.s, #0
    dup     z4.s, #0
    dup     z5.s, #0
    dup     z6.s, #0
    dup     z7.s, #0
    dup     z8.s, #0
    dup     z9.s, #0
    dup     z10.s, #0
    dup     z11.s, #0

    // running pointers
    mov     x7,  x0                 // aq running (row0)
    mov     x9,  x2                 // bq running
    mov     x10, x3                 // bd running
    mov     x11, x1                 // ad running

.Lblock:
    // zero int32 partials z12..z23
    dup     z12.s, #0
    dup     z13.s, #0
    dup     z14.s, #0
    dup     z15.s, #0
    dup     z16.s, #0
    dup     z17.s, #0
    dup     z18.s, #0
    dup     z19.s, #0
    dup     z20.s, #0
    dup     z21.s, #0
    dup     z22.s, #0
    dup     z23.s, #0

    // A row pointers for this block: row stride = 32 bytes
    mov     x12, x7                 // row0
    add     x13, x7, #32            // row1
    add     x14, x7, #64            // row2
    mov     x15, x9                 // bq for this block

    mov     x16, #8                 // g3 counter (8 kgroups of 4)
.Lkg:
    ld1rw   { z24.s }, p1/z, [x12]
    ld1rw   { z25.s }, p1/z, [x13]
    ld1rw   { z26.s }, p1/z, [x14]
    ld1b    { z27.b }, p0/z, [x15, #0, mul vl]
    ld1b    { z28.b }, p0/z, [x15, #1, mul vl]
    ld1b    { z29.b }, p0/z, [x15, #2, mul vl]
    ld1b    { z30.b }, p0/z, [x15, #3, mul vl]

    sdot    z12.s, z24.b, z27.b
    sdot    z13.s, z24.b, z28.b
    sdot    z14.s, z24.b, z29.b
    sdot    z15.s, z24.b, z30.b
    sdot    z16.s, z25.b, z27.b
    sdot    z17.s, z25.b, z28.b
    sdot    z18.s, z25.b, z29.b
    sdot    z19.s, z25.b, z30.b
    sdot    z20.s, z26.b, z27.b
    sdot    z21.s, z26.b, z28.b
    sdot    z22.s, z26.b, z29.b
    sdot    z23.s, z26.b, z30.b

    add     x12, x12, #4
    add     x13, x13, #4
    add     x14, x14, #4
    add     x15, x15, #256          // 4 vecs * 64 bytes
    subs    x16, x16, #1
    b.gt    .Lkg

    // ---- scale-out: f[r][v] += scvtf(p[r][v]) * dw[v] * ad[r] ----
    ld1w    { z27.s }, p1/z, [x10, #0, mul vl]   // dw vec0 (16 cols)
    ld1w    { z28.s }, p1/z, [x10, #1, mul vl]   // dw vec1
    ld1w    { z29.s }, p1/z, [x10, #2, mul vl]   // dw vec2
    ld1w    { z30.s }, p1/z, [x10, #3, mul vl]   // dw vec3
    ld1rw   { z24.s }, p1/z, [x11]               // ad row0
    ld1rw   { z25.s }, p1/z, [x11, #4]           // ad row1
    ld1rw   { z26.s }, p1/z, [x11, #8]           // ad row2

    // row0
    scvtf   z12.s, p1/m, z12.s
    fmul    z12.s, z12.s, z27.s
    fmla    z0.s, p1/m, z12.s, z24.s
    scvtf   z13.s, p1/m, z13.s
    fmul    z13.s, z13.s, z28.s
    fmla    z1.s, p1/m, z13.s, z24.s
    scvtf   z14.s, p1/m, z14.s
    fmul    z14.s, z14.s, z29.s
    fmla    z2.s, p1/m, z14.s, z24.s
    scvtf   z15.s, p1/m, z15.s
    fmul    z15.s, z15.s, z30.s
    fmla    z3.s, p1/m, z15.s, z24.s
    // row1
    scvtf   z16.s, p1/m, z16.s
    fmul    z16.s, z16.s, z27.s
    fmla    z4.s, p1/m, z16.s, z25.s
    scvtf   z17.s, p1/m, z17.s
    fmul    z17.s, z17.s, z28.s
    fmla    z5.s, p1/m, z17.s, z25.s
    scvtf   z18.s, p1/m, z18.s
    fmul    z18.s, z18.s, z29.s
    fmla    z6.s, p1/m, z18.s, z25.s
    scvtf   z19.s, p1/m, z19.s
    fmul    z19.s, z19.s, z30.s
    fmla    z7.s, p1/m, z19.s, z25.s
    // row2
    scvtf   z20.s, p1/m, z20.s
    fmul    z20.s, z20.s, z27.s
    fmla    z8.s, p1/m, z20.s, z26.s
    scvtf   z21.s, p1/m, z21.s
    fmul    z21.s, z21.s, z28.s
    fmla    z9.s, p1/m, z21.s, z26.s
    scvtf   z22.s, p1/m, z22.s
    fmul    z22.s, z22.s, z29.s
    fmla    z10.s, p1/m, z22.s, z26.s
    scvtf   z23.s, p1/m, z23.s
    fmul    z23.s, z23.s, z30.s
    fmla    z11.s, p1/m, z23.s, z26.s

    // advance per-block pointers
    add     x7,  x7,  #96           // aq: 3 rows * 32
    add     x9,  x9,  #2048         // bq: 8*4*64
    add     x10, x10, #256          // bd: 64 floats
    add     x11, x11, #12           // ad: 3 floats
    subs    x4, x4, #1
    b.gt    .Lblock

    // ---- store C: 3 rows x 4 vecs ----
    mov     x12, x5
    st1w    { z0.s }, p1, [x12, #0, mul vl]
    st1w    { z1.s }, p1, [x12, #1, mul vl]
    st1w    { z2.s }, p1, [x12, #2, mul vl]
    st1w    { z3.s }, p1, [x12, #3, mul vl]
    add     x12, x12, x6
    st1w    { z4.s }, p1, [x12, #0, mul vl]
    st1w    { z5.s }, p1, [x12, #1, mul vl]
    st1w    { z6.s }, p1, [x12, #2, mul vl]
    st1w    { z7.s }, p1, [x12, #3, mul vl]
    add     x12, x12, x6
    st1w    { z8.s }, p1, [x12, #0, mul vl]
    st1w    { z9.s }, p1, [x12, #1, mul vl]
    st1w    { z10.s }, p1, [x12, #2, mul vl]
    st1w    { z11.s }, p1, [x12, #3, mul vl]

    ldp     d14, d15, [sp], #16
    ldp     d12, d13, [sp], #16
    ldp     d10, d11, [sp], #16
    ldp     d8, d9,  [sp], #16
    ret
    .size kernel_q8v2_3x4, .-kernel_q8v2_3x4


// ===========================================================================
// kernel_q8v2_3x4_arow: per-ROW activation scale variant.
// da[r] is constant across blocks -> factor it out of the per-block loop:
//   f[r][v] = da[r] * sum_b dw[b,v] * scvtf(p[b,r,v])
// Per-block scale-out is scvtf + fmla (2 ops) instead of scvtf+fmul+fmla (3).
// da[] is applied once in the epilogue. ad arg = [3] floats (per-row, block 0).
// Same arg signature; ad is read as 3 per-row scales (stride 0 across blocks).
// ===========================================================================
    .align 6
    .global kernel_q8v2_3x4_arow
    .type kernel_q8v2_3x4_arow, @function
kernel_q8v2_3x4_arow:
    stp     d8, d9,  [sp, #-16]!
    stp     d10, d11, [sp, #-16]!
    stp     d12, d13, [sp, #-16]!
    stp     d14, d15, [sp, #-16]!
    ptrue   p0.b
    ptrue   p1.s
    dup     z0.s, #0
    dup     z1.s, #0
    dup     z2.s, #0
    dup     z3.s, #0
    dup     z4.s, #0
    dup     z5.s, #0
    dup     z6.s, #0
    dup     z7.s, #0
    dup     z8.s, #0
    dup     z9.s, #0
    dup     z10.s, #0
    dup     z11.s, #0
    mov     x7,  x0
    mov     x9,  x2
    mov     x10, x3
.Lblock_ar:
    dup     z12.s, #0
    dup     z13.s, #0
    dup     z14.s, #0
    dup     z15.s, #0
    dup     z16.s, #0
    dup     z17.s, #0
    dup     z18.s, #0
    dup     z19.s, #0
    dup     z20.s, #0
    dup     z21.s, #0
    dup     z22.s, #0
    dup     z23.s, #0
    mov     x12, x7
    add     x13, x7, #32
    add     x14, x7, #64
    mov     x15, x9
    mov     x16, #8
.Lkg_ar:
    ld1rw   { z24.s }, p1/z, [x12]
    ld1rw   { z25.s }, p1/z, [x13]
    ld1rw   { z26.s }, p1/z, [x14]
    ld1b    { z27.b }, p0/z, [x15, #0, mul vl]
    ld1b    { z28.b }, p0/z, [x15, #1, mul vl]
    ld1b    { z29.b }, p0/z, [x15, #2, mul vl]
    ld1b    { z30.b }, p0/z, [x15, #3, mul vl]
    sdot    z12.s, z24.b, z27.b
    sdot    z13.s, z24.b, z28.b
    sdot    z14.s, z24.b, z29.b
    sdot    z15.s, z24.b, z30.b
    sdot    z16.s, z25.b, z27.b
    sdot    z17.s, z25.b, z28.b
    sdot    z18.s, z25.b, z29.b
    sdot    z19.s, z25.b, z30.b
    sdot    z20.s, z26.b, z27.b
    sdot    z21.s, z26.b, z28.b
    sdot    z22.s, z26.b, z29.b
    sdot    z23.s, z26.b, z30.b
    add     x12, x12, #4
    add     x13, x13, #4
    add     x14, x14, #4
    add     x15, x15, #256
    subs    x16, x16, #1
    b.gt    .Lkg_ar
    // scale-out: f += scvtf(p) * dw   (2 ops, da deferred)
    ld1w    { z27.s }, p1/z, [x10, #0, mul vl]
    ld1w    { z28.s }, p1/z, [x10, #1, mul vl]
    ld1w    { z29.s }, p1/z, [x10, #2, mul vl]
    ld1w    { z30.s }, p1/z, [x10, #3, mul vl]
    scvtf   z12.s, p1/m, z12.s
    fmla    z0.s, p1/m, z12.s, z27.s
    scvtf   z13.s, p1/m, z13.s
    fmla    z1.s, p1/m, z13.s, z28.s
    scvtf   z14.s, p1/m, z14.s
    fmla    z2.s, p1/m, z14.s, z29.s
    scvtf   z15.s, p1/m, z15.s
    fmla    z3.s, p1/m, z15.s, z30.s
    scvtf   z16.s, p1/m, z16.s
    fmla    z4.s, p1/m, z16.s, z27.s
    scvtf   z17.s, p1/m, z17.s
    fmla    z5.s, p1/m, z17.s, z28.s
    scvtf   z18.s, p1/m, z18.s
    fmla    z6.s, p1/m, z18.s, z29.s
    scvtf   z19.s, p1/m, z19.s
    fmla    z7.s, p1/m, z19.s, z30.s
    scvtf   z20.s, p1/m, z20.s
    fmla    z8.s, p1/m, z20.s, z27.s
    scvtf   z21.s, p1/m, z21.s
    fmla    z9.s, p1/m, z21.s, z28.s
    scvtf   z22.s, p1/m, z22.s
    fmla    z10.s, p1/m, z22.s, z29.s
    scvtf   z23.s, p1/m, z23.s
    fmla    z11.s, p1/m, z23.s, z30.s
    add     x7,  x7,  #96
    add     x9,  x9,  #2048
    add     x10, x10, #256
    subs    x4, x4, #1
    b.gt    .Lblock_ar
    // epilogue: f[r][v] *= da[r]; store
    ld1rw   { z24.s }, p1/z, [x1]
    ld1rw   { z25.s }, p1/z, [x1, #4]
    ld1rw   { z26.s }, p1/z, [x1, #8]
    fmul    z0.s, z0.s, z24.s
    fmul    z1.s, z1.s, z24.s
    fmul    z2.s, z2.s, z24.s
    fmul    z3.s, z3.s, z24.s
    fmul    z4.s, z4.s, z25.s
    fmul    z5.s, z5.s, z25.s
    fmul    z6.s, z6.s, z25.s
    fmul    z7.s, z7.s, z25.s
    fmul    z8.s, z8.s, z26.s
    fmul    z9.s, z9.s, z26.s
    fmul    z10.s, z10.s, z26.s
    fmul    z11.s, z11.s, z26.s
    mov     x12, x5
    st1w    { z0.s }, p1, [x12, #0, mul vl]
    st1w    { z1.s }, p1, [x12, #1, mul vl]
    st1w    { z2.s }, p1, [x12, #2, mul vl]
    st1w    { z3.s }, p1, [x12, #3, mul vl]
    add     x12, x12, x6
    st1w    { z4.s }, p1, [x12, #0, mul vl]
    st1w    { z5.s }, p1, [x12, #1, mul vl]
    st1w    { z6.s }, p1, [x12, #2, mul vl]
    st1w    { z7.s }, p1, [x12, #3, mul vl]
    add     x12, x12, x6
    st1w    { z8.s }, p1, [x12, #0, mul vl]
    st1w    { z9.s }, p1, [x12, #1, mul vl]
    st1w    { z10.s }, p1, [x12, #2, mul vl]
    st1w    { z11.s }, p1, [x12, #3, mul vl]
    ldp     d14, d15, [sp], #16
    ldp     d12, d13, [sp], #16
    ldp     d10, d11, [sp], #16
    ldp     d8, d9,  [sp], #16
    ret
    .size kernel_q8v2_3x4_arow, .-kernel_q8v2_3x4_arow
