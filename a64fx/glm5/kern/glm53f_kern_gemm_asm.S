/* gk_gemm_tile6x4_asm: one 64-row x 6-token prefill GEMM tile (see
 * glm53f_kern_gemm.c for the panel format).  Fixed registers: z0-z23 int32
 * accumulators (token t, panel p -> z(4t+p)), z24-z27 weight vectors,
 * z28-z31 token broadcasts.  The k loop is unrolled 4x with immediate
 * offsets (10 pointer ADDs per 96 SDOTs); the f32 accumulators live in the
 * L1-resident y tile, updated once per scale block.
 *
 *   void gk_gemm_tile6x4_asm(const int8_t *w, size_t panel_bytes,
 *                            long nblocks, long kgroups4,   // sb/16
 *                            const int8_t *x, size_t ldx,
 *                            const float *xs, size_t ldxs_bytes,
 *                            float *y, size_t ldy_bytes);   // y, ldy on stack
 * y must be zeroed by the caller. */
    .text
    .globl gk_gemm_tile6x4_asm
    .type gk_gemm_tile6x4_asm, %function
    .p2align 6
gk_gemm_tile6x4_asm:
    stp x19, x20, [sp, #-144]!
    stp x21, x22, [sp, #16]
    stp x23, x24, [sp, #32]
    stp x25, x26, [sp, #48]
    stp x27, x28, [sp, #64]
    stp d8, d9, [sp, #80]
    stp d10, d11, [sp, #96]
    stp d12, d13, [sp, #112]
    stp d14, d15, [sp, #128]
    ldr x8, [sp, #144]
    ldr x9, [sp, #152]
    mov x10, x0
    add x11, x10, x1
    add x12, x11, x1
    add x13, x12, x1
    mov x14, x4
    add x15, x14, x5
    add x16, x15, x5
    add x17, x16, x5
    add x19, x17, x5
    add x20, x19, x5
    mov x21, x6
    add x22, x21, x7
    add x23, x22, x7
    add x24, x23, x7
    add x25, x24, x7
    add x26, x25, x7
    ptrue p0.b
.Lblock:
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
    mov x27, x3
.Lk:
    ld1b {z24.b}, p0/z, [x10, #0, mul vl]
    ld1b {z25.b}, p0/z, [x11, #0, mul vl]
    ld1b {z26.b}, p0/z, [x12, #0, mul vl]
    ld1b {z27.b}, p0/z, [x13, #0, mul vl]
    ld1rw {z28.s}, p0/z, [x14, #0]
    sdot z0.s, z24.b, z28.b
    sdot z1.s, z25.b, z28.b
    sdot z2.s, z26.b, z28.b
    sdot z3.s, z27.b, z28.b
    ld1rw {z29.s}, p0/z, [x15, #0]
    sdot z4.s, z24.b, z29.b
    sdot z5.s, z25.b, z29.b
    sdot z6.s, z26.b, z29.b
    sdot z7.s, z27.b, z29.b
    ld1rw {z30.s}, p0/z, [x16, #0]
    sdot z8.s, z24.b, z30.b
    sdot z9.s, z25.b, z30.b
    sdot z10.s, z26.b, z30.b
    sdot z11.s, z27.b, z30.b
    ld1rw {z31.s}, p0/z, [x17, #0]
    sdot z12.s, z24.b, z31.b
    sdot z13.s, z25.b, z31.b
    sdot z14.s, z26.b, z31.b
    sdot z15.s, z27.b, z31.b
    ld1rw {z28.s}, p0/z, [x19, #0]
    sdot z16.s, z24.b, z28.b
    sdot z17.s, z25.b, z28.b
    sdot z18.s, z26.b, z28.b
    sdot z19.s, z27.b, z28.b
    ld1rw {z29.s}, p0/z, [x20, #0]
    sdot z20.s, z24.b, z29.b
    sdot z21.s, z25.b, z29.b
    sdot z22.s, z26.b, z29.b
    sdot z23.s, z27.b, z29.b
    ld1b {z24.b}, p0/z, [x10, #1, mul vl]
    ld1b {z25.b}, p0/z, [x11, #1, mul vl]
    ld1b {z26.b}, p0/z, [x12, #1, mul vl]
    ld1b {z27.b}, p0/z, [x13, #1, mul vl]
    ld1rw {z28.s}, p0/z, [x14, #4]
    sdot z0.s, z24.b, z28.b
    sdot z1.s, z25.b, z28.b
    sdot z2.s, z26.b, z28.b
    sdot z3.s, z27.b, z28.b
    ld1rw {z29.s}, p0/z, [x15, #4]
    sdot z4.s, z24.b, z29.b
    sdot z5.s, z25.b, z29.b
    sdot z6.s, z26.b, z29.b
    sdot z7.s, z27.b, z29.b
    ld1rw {z30.s}, p0/z, [x16, #4]
    sdot z8.s, z24.b, z30.b
    sdot z9.s, z25.b, z30.b
    sdot z10.s, z26.b, z30.b
    sdot z11.s, z27.b, z30.b
    ld1rw {z31.s}, p0/z, [x17, #4]
    sdot z12.s, z24.b, z31.b
    sdot z13.s, z25.b, z31.b
    sdot z14.s, z26.b, z31.b
    sdot z15.s, z27.b, z31.b
    ld1rw {z28.s}, p0/z, [x19, #4]
    sdot z16.s, z24.b, z28.b
    sdot z17.s, z25.b, z28.b
    sdot z18.s, z26.b, z28.b
    sdot z19.s, z27.b, z28.b
    ld1rw {z29.s}, p0/z, [x20, #4]
    sdot z20.s, z24.b, z29.b
    sdot z21.s, z25.b, z29.b
    sdot z22.s, z26.b, z29.b
    sdot z23.s, z27.b, z29.b
    ld1b {z24.b}, p0/z, [x10, #2, mul vl]
    ld1b {z25.b}, p0/z, [x11, #2, mul vl]
    ld1b {z26.b}, p0/z, [x12, #2, mul vl]
    ld1b {z27.b}, p0/z, [x13, #2, mul vl]
    ld1rw {z28.s}, p0/z, [x14, #8]
    sdot z0.s, z24.b, z28.b
    sdot z1.s, z25.b, z28.b
    sdot z2.s, z26.b, z28.b
    sdot z3.s, z27.b, z28.b
    ld1rw {z29.s}, p0/z, [x15, #8]
    sdot z4.s, z24.b, z29.b
    sdot z5.s, z25.b, z29.b
    sdot z6.s, z26.b, z29.b
    sdot z7.s, z27.b, z29.b
    ld1rw {z30.s}, p0/z, [x16, #8]
    sdot z8.s, z24.b, z30.b
    sdot z9.s, z25.b, z30.b
    sdot z10.s, z26.b, z30.b
    sdot z11.s, z27.b, z30.b
    ld1rw {z31.s}, p0/z, [x17, #8]
    sdot z12.s, z24.b, z31.b
    sdot z13.s, z25.b, z31.b
    sdot z14.s, z26.b, z31.b
    sdot z15.s, z27.b, z31.b
    ld1rw {z28.s}, p0/z, [x19, #8]
    sdot z16.s, z24.b, z28.b
    sdot z17.s, z25.b, z28.b
    sdot z18.s, z26.b, z28.b
    sdot z19.s, z27.b, z28.b
    ld1rw {z29.s}, p0/z, [x20, #8]
    sdot z20.s, z24.b, z29.b
    sdot z21.s, z25.b, z29.b
    sdot z22.s, z26.b, z29.b
    sdot z23.s, z27.b, z29.b
    ld1b {z24.b}, p0/z, [x10, #3, mul vl]
    ld1b {z25.b}, p0/z, [x11, #3, mul vl]
    ld1b {z26.b}, p0/z, [x12, #3, mul vl]
    ld1b {z27.b}, p0/z, [x13, #3, mul vl]
    ld1rw {z28.s}, p0/z, [x14, #12]
    sdot z0.s, z24.b, z28.b
    sdot z1.s, z25.b, z28.b
    sdot z2.s, z26.b, z28.b
    sdot z3.s, z27.b, z28.b
    ld1rw {z29.s}, p0/z, [x15, #12]
    sdot z4.s, z24.b, z29.b
    sdot z5.s, z25.b, z29.b
    sdot z6.s, z26.b, z29.b
    sdot z7.s, z27.b, z29.b
    ld1rw {z30.s}, p0/z, [x16, #12]
    sdot z8.s, z24.b, z30.b
    sdot z9.s, z25.b, z30.b
    sdot z10.s, z26.b, z30.b
    sdot z11.s, z27.b, z30.b
    ld1rw {z31.s}, p0/z, [x17, #12]
    sdot z12.s, z24.b, z31.b
    sdot z13.s, z25.b, z31.b
    sdot z14.s, z26.b, z31.b
    sdot z15.s, z27.b, z31.b
    ld1rw {z28.s}, p0/z, [x19, #12]
    sdot z16.s, z24.b, z28.b
    sdot z17.s, z25.b, z28.b
    sdot z18.s, z26.b, z28.b
    sdot z19.s, z27.b, z28.b
    ld1rw {z29.s}, p0/z, [x20, #12]
    sdot z20.s, z24.b, z29.b
    sdot z21.s, z25.b, z29.b
    sdot z22.s, z26.b, z29.b
    sdot z23.s, z27.b, z29.b
    add x10, x10, #256
    add x11, x11, #256
    add x12, x12, #256
    add x13, x13, #256
    add x14, x14, #16
    add x15, x15, #16
    add x16, x16, #16
    add x17, x17, #16
    add x19, x19, #16
    add x20, x20, #16
    subs x27, x27, #1
    b.ne .Lk
    ld1w {z24.s}, p0/z, [x10]
    add x10, x10, #64
    ld1w {z25.s}, p0/z, [x11]
    add x11, x11, #64
    ld1w {z26.s}, p0/z, [x12]
    add x12, x12, #64
    ld1w {z27.s}, p0/z, [x13]
    add x13, x13, #64
    mov x28, x8
    ld1rw {z28.s}, p0/z, [x21]
    scvtf z0.s, p0/m, z0.s
    fmul z29.s, z24.s, z28.s
    ld1w {z30.s}, p0/z, [x28, #0, mul vl]
    fmla z30.s, p0/m, z0.s, z29.s
    st1w {z30.s}, p0, [x28, #0, mul vl]
    scvtf z1.s, p0/m, z1.s
    fmul z29.s, z25.s, z28.s
    ld1w {z30.s}, p0/z, [x28, #1, mul vl]
    fmla z30.s, p0/m, z1.s, z29.s
    st1w {z30.s}, p0, [x28, #1, mul vl]
    scvtf z2.s, p0/m, z2.s
    fmul z29.s, z26.s, z28.s
    ld1w {z30.s}, p0/z, [x28, #2, mul vl]
    fmla z30.s, p0/m, z2.s, z29.s
    st1w {z30.s}, p0, [x28, #2, mul vl]
    scvtf z3.s, p0/m, z3.s
    fmul z29.s, z27.s, z28.s
    ld1w {z30.s}, p0/z, [x28, #3, mul vl]
    fmla z30.s, p0/m, z3.s, z29.s
    st1w {z30.s}, p0, [x28, #3, mul vl]
    add x28, x28, x9
    ld1rw {z28.s}, p0/z, [x22]
    scvtf z4.s, p0/m, z4.s
    fmul z29.s, z24.s, z28.s
    ld1w {z30.s}, p0/z, [x28, #0, mul vl]
    fmla z30.s, p0/m, z4.s, z29.s
    st1w {z30.s}, p0, [x28, #0, mul vl]
    scvtf z5.s, p0/m, z5.s
    fmul z29.s, z25.s, z28.s
    ld1w {z30.s}, p0/z, [x28, #1, mul vl]
    fmla z30.s, p0/m, z5.s, z29.s
    st1w {z30.s}, p0, [x28, #1, mul vl]
    scvtf z6.s, p0/m, z6.s
    fmul z29.s, z26.s, z28.s
    ld1w {z30.s}, p0/z, [x28, #2, mul vl]
    fmla z30.s, p0/m, z6.s, z29.s
    st1w {z30.s}, p0, [x28, #2, mul vl]
    scvtf z7.s, p0/m, z7.s
    fmul z29.s, z27.s, z28.s
    ld1w {z30.s}, p0/z, [x28, #3, mul vl]
    fmla z30.s, p0/m, z7.s, z29.s
    st1w {z30.s}, p0, [x28, #3, mul vl]
    add x28, x28, x9
    ld1rw {z28.s}, p0/z, [x23]
    scvtf z8.s, p0/m, z8.s
    fmul z29.s, z24.s, z28.s
    ld1w {z30.s}, p0/z, [x28, #0, mul vl]
    fmla z30.s, p0/m, z8.s, z29.s
    st1w {z30.s}, p0, [x28, #0, mul vl]
    scvtf z9.s, p0/m, z9.s
    fmul z29.s, z25.s, z28.s
    ld1w {z30.s}, p0/z, [x28, #1, mul vl]
    fmla z30.s, p0/m, z9.s, z29.s
    st1w {z30.s}, p0, [x28, #1, mul vl]
    scvtf z10.s, p0/m, z10.s
    fmul z29.s, z26.s, z28.s
    ld1w {z30.s}, p0/z, [x28, #2, mul vl]
    fmla z30.s, p0/m, z10.s, z29.s
    st1w {z30.s}, p0, [x28, #2, mul vl]
    scvtf z11.s, p0/m, z11.s
    fmul z29.s, z27.s, z28.s
    ld1w {z30.s}, p0/z, [x28, #3, mul vl]
    fmla z30.s, p0/m, z11.s, z29.s
    st1w {z30.s}, p0, [x28, #3, mul vl]
    add x28, x28, x9
    ld1rw {z28.s}, p0/z, [x24]
    scvtf z12.s, p0/m, z12.s
    fmul z29.s, z24.s, z28.s
    ld1w {z30.s}, p0/z, [x28, #0, mul vl]
    fmla z30.s, p0/m, z12.s, z29.s
    st1w {z30.s}, p0, [x28, #0, mul vl]
    scvtf z13.s, p0/m, z13.s
    fmul z29.s, z25.s, z28.s
    ld1w {z30.s}, p0/z, [x28, #1, mul vl]
    fmla z30.s, p0/m, z13.s, z29.s
    st1w {z30.s}, p0, [x28, #1, mul vl]
    scvtf z14.s, p0/m, z14.s
    fmul z29.s, z26.s, z28.s
    ld1w {z30.s}, p0/z, [x28, #2, mul vl]
    fmla z30.s, p0/m, z14.s, z29.s
    st1w {z30.s}, p0, [x28, #2, mul vl]
    scvtf z15.s, p0/m, z15.s
    fmul z29.s, z27.s, z28.s
    ld1w {z30.s}, p0/z, [x28, #3, mul vl]
    fmla z30.s, p0/m, z15.s, z29.s
    st1w {z30.s}, p0, [x28, #3, mul vl]
    add x28, x28, x9
    ld1rw {z28.s}, p0/z, [x25]
    scvtf z16.s, p0/m, z16.s
    fmul z29.s, z24.s, z28.s
    ld1w {z30.s}, p0/z, [x28, #0, mul vl]
    fmla z30.s, p0/m, z16.s, z29.s
    st1w {z30.s}, p0, [x28, #0, mul vl]
    scvtf z17.s, p0/m, z17.s
    fmul z29.s, z25.s, z28.s
    ld1w {z30.s}, p0/z, [x28, #1, mul vl]
    fmla z30.s, p0/m, z17.s, z29.s
    st1w {z30.s}, p0, [x28, #1, mul vl]
    scvtf z18.s, p0/m, z18.s
    fmul z29.s, z26.s, z28.s
    ld1w {z30.s}, p0/z, [x28, #2, mul vl]
    fmla z30.s, p0/m, z18.s, z29.s
    st1w {z30.s}, p0, [x28, #2, mul vl]
    scvtf z19.s, p0/m, z19.s
    fmul z29.s, z27.s, z28.s
    ld1w {z30.s}, p0/z, [x28, #3, mul vl]
    fmla z30.s, p0/m, z19.s, z29.s
    st1w {z30.s}, p0, [x28, #3, mul vl]
    add x28, x28, x9
    ld1rw {z28.s}, p0/z, [x26]
    scvtf z20.s, p0/m, z20.s
    fmul z29.s, z24.s, z28.s
    ld1w {z30.s}, p0/z, [x28, #0, mul vl]
    fmla z30.s, p0/m, z20.s, z29.s
    st1w {z30.s}, p0, [x28, #0, mul vl]
    scvtf z21.s, p0/m, z21.s
    fmul z29.s, z25.s, z28.s
    ld1w {z30.s}, p0/z, [x28, #1, mul vl]
    fmla z30.s, p0/m, z21.s, z29.s
    st1w {z30.s}, p0, [x28, #1, mul vl]
    scvtf z22.s, p0/m, z22.s
    fmul z29.s, z26.s, z28.s
    ld1w {z30.s}, p0/z, [x28, #2, mul vl]
    fmla z30.s, p0/m, z22.s, z29.s
    st1w {z30.s}, p0, [x28, #2, mul vl]
    scvtf z23.s, p0/m, z23.s
    fmul z29.s, z27.s, z28.s
    ld1w {z30.s}, p0/z, [x28, #3, mul vl]
    fmla z30.s, p0/m, z23.s, z29.s
    st1w {z30.s}, p0, [x28, #3, mul vl]
    add x28, x28, x9
    add x21, x21, #4
    add x22, x22, #4
    add x23, x23, #4
    add x24, x24, #4
    add x25, x25, #4
    add x26, x26, #4
    subs x2, x2, #1
    b.ne .Lblock
    ldp d14, d15, [sp, #128]
    ldp d12, d13, [sp, #112]
    ldp d10, d11, [sp, #96]
    ldp d8, d9, [sp, #80]
    ldp x27, x28, [sp, #64]
    ldp x25, x26, [sp, #48]
    ldp x23, x24, [sp, #32]
    ldp x21, x22, [sp, #16]
    ldp x19, x20, [sp], #144
    ret
    .size gk_gemm_tile6x4_asm, .-gk_gemm_tile6x4_asm

/* gk_gemm_tile6x4p_asm: 64-row x 6-token prefill GEMM tile on fully packed
 * operands (one stream each, like a64fx/int8-cmg's 94% 6x4 kernel):
 *   w:  64-row super-panel, per scale block of sb columns:
 *       (sb/4) k-groups x 4 vectors x 64 B, then 64 f32 row scales (256 B)
 *   xp: packed activations, per k-group 6 tokens x 4 bytes (24 B)
 *   xsp: packed token scales, per block 6 f32
 * Registers as in gk_gemm_tile6x4_asm.  The k loop is unrolled 4x.
 *
 *   void gk_gemm_tile6x4p_asm(const int8_t *w, long nblocks, long kgroups4,
 *                             const int8_t *xp, const float *xsp,
 *                             float *y, size_t ldy_bytes);
 * y must be zeroed by the caller. */
    .text
    .globl gk_gemm_tile6x4p_asm
    .type gk_gemm_tile6x4p_asm, %function
    .p2align 6
gk_gemm_tile6x4p_asm:
    stp d8, d9, [sp, #-64]!
    stp d10, d11, [sp, #16]
    stp d12, d13, [sp, #32]
    stp d14, d15, [sp, #48]
    ptrue p0.b
.Lpblock:
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
    mov x8, x2
.Lpk:
    ld1b {z24.b}, p0/z, [x0, #0, mul vl]
    ld1b {z25.b}, p0/z, [x0, #1, mul vl]
    ld1b {z26.b}, p0/z, [x0, #2, mul vl]
    ld1b {z27.b}, p0/z, [x0, #3, mul vl]
    ld1rw {z28.s}, p0/z, [x3, #0]
    sdot z0.s, z24.b, z28.b
    sdot z1.s, z25.b, z28.b
    sdot z2.s, z26.b, z28.b
    sdot z3.s, z27.b, z28.b
    ld1rw {z29.s}, p0/z, [x3, #4]
    sdot z4.s, z24.b, z29.b
    sdot z5.s, z25.b, z29.b
    sdot z6.s, z26.b, z29.b
    sdot z7.s, z27.b, z29.b
    ld1rw {z30.s}, p0/z, [x3, #8]
    sdot z8.s, z24.b, z30.b
    sdot z9.s, z25.b, z30.b
    sdot z10.s, z26.b, z30.b
    sdot z11.s, z27.b, z30.b
    ld1rw {z31.s}, p0/z, [x3, #12]
    sdot z12.s, z24.b, z31.b
    sdot z13.s, z25.b, z31.b
    sdot z14.s, z26.b, z31.b
    sdot z15.s, z27.b, z31.b
    ld1rw {z28.s}, p0/z, [x3, #16]
    sdot z16.s, z24.b, z28.b
    sdot z17.s, z25.b, z28.b
    sdot z18.s, z26.b, z28.b
    sdot z19.s, z27.b, z28.b
    ld1rw {z29.s}, p0/z, [x3, #20]
    sdot z20.s, z24.b, z29.b
    sdot z21.s, z25.b, z29.b
    sdot z22.s, z26.b, z29.b
    sdot z23.s, z27.b, z29.b
    ld1b {z24.b}, p0/z, [x0, #4, mul vl]
    ld1b {z25.b}, p0/z, [x0, #5, mul vl]
    ld1b {z26.b}, p0/z, [x0, #6, mul vl]
    ld1b {z27.b}, p0/z, [x0, #7, mul vl]
    ld1rw {z28.s}, p0/z, [x3, #24]
    sdot z0.s, z24.b, z28.b
    sdot z1.s, z25.b, z28.b
    sdot z2.s, z26.b, z28.b
    sdot z3.s, z27.b, z28.b
    ld1rw {z29.s}, p0/z, [x3, #28]
    sdot z4.s, z24.b, z29.b
    sdot z5.s, z25.b, z29.b
    sdot z6.s, z26.b, z29.b
    sdot z7.s, z27.b, z29.b
    ld1rw {z30.s}, p0/z, [x3, #32]
    sdot z8.s, z24.b, z30.b
    sdot z9.s, z25.b, z30.b
    sdot z10.s, z26.b, z30.b
    sdot z11.s, z27.b, z30.b
    ld1rw {z31.s}, p0/z, [x3, #36]
    sdot z12.s, z24.b, z31.b
    sdot z13.s, z25.b, z31.b
    sdot z14.s, z26.b, z31.b
    sdot z15.s, z27.b, z31.b
    ld1rw {z28.s}, p0/z, [x3, #40]
    sdot z16.s, z24.b, z28.b
    sdot z17.s, z25.b, z28.b
    sdot z18.s, z26.b, z28.b
    sdot z19.s, z27.b, z28.b
    ld1rw {z29.s}, p0/z, [x3, #44]
    sdot z20.s, z24.b, z29.b
    sdot z21.s, z25.b, z29.b
    sdot z22.s, z26.b, z29.b
    sdot z23.s, z27.b, z29.b
    add x0, x0, #512
    ld1b {z24.b}, p0/z, [x0, #0, mul vl]
    ld1b {z25.b}, p0/z, [x0, #1, mul vl]
    ld1b {z26.b}, p0/z, [x0, #2, mul vl]
    ld1b {z27.b}, p0/z, [x0, #3, mul vl]
    ld1rw {z28.s}, p0/z, [x3, #48]
    sdot z0.s, z24.b, z28.b
    sdot z1.s, z25.b, z28.b
    sdot z2.s, z26.b, z28.b
    sdot z3.s, z27.b, z28.b
    ld1rw {z29.s}, p0/z, [x3, #52]
    sdot z4.s, z24.b, z29.b
    sdot z5.s, z25.b, z29.b
    sdot z6.s, z26.b, z29.b
    sdot z7.s, z27.b, z29.b
    ld1rw {z30.s}, p0/z, [x3, #56]
    sdot z8.s, z24.b, z30.b
    sdot z9.s, z25.b, z30.b
    sdot z10.s, z26.b, z30.b
    sdot z11.s, z27.b, z30.b
    ld1rw {z31.s}, p0/z, [x3, #60]
    sdot z12.s, z24.b, z31.b
    sdot z13.s, z25.b, z31.b
    sdot z14.s, z26.b, z31.b
    sdot z15.s, z27.b, z31.b
    ld1rw {z28.s}, p0/z, [x3, #64]
    sdot z16.s, z24.b, z28.b
    sdot z17.s, z25.b, z28.b
    sdot z18.s, z26.b, z28.b
    sdot z19.s, z27.b, z28.b
    ld1rw {z29.s}, p0/z, [x3, #68]
    sdot z20.s, z24.b, z29.b
    sdot z21.s, z25.b, z29.b
    sdot z22.s, z26.b, z29.b
    sdot z23.s, z27.b, z29.b
    ld1b {z24.b}, p0/z, [x0, #4, mul vl]
    ld1b {z25.b}, p0/z, [x0, #5, mul vl]
    ld1b {z26.b}, p0/z, [x0, #6, mul vl]
    ld1b {z27.b}, p0/z, [x0, #7, mul vl]
    ld1rw {z28.s}, p0/z, [x3, #72]
    sdot z0.s, z24.b, z28.b
    sdot z1.s, z25.b, z28.b
    sdot z2.s, z26.b, z28.b
    sdot z3.s, z27.b, z28.b
    ld1rw {z29.s}, p0/z, [x3, #76]
    sdot z4.s, z24.b, z29.b
    sdot z5.s, z25.b, z29.b
    sdot z6.s, z26.b, z29.b
    sdot z7.s, z27.b, z29.b
    ld1rw {z30.s}, p0/z, [x3, #80]
    sdot z8.s, z24.b, z30.b
    sdot z9.s, z25.b, z30.b
    sdot z10.s, z26.b, z30.b
    sdot z11.s, z27.b, z30.b
    ld1rw {z31.s}, p0/z, [x3, #84]
    sdot z12.s, z24.b, z31.b
    sdot z13.s, z25.b, z31.b
    sdot z14.s, z26.b, z31.b
    sdot z15.s, z27.b, z31.b
    ld1rw {z28.s}, p0/z, [x3, #88]
    sdot z16.s, z24.b, z28.b
    sdot z17.s, z25.b, z28.b
    sdot z18.s, z26.b, z28.b
    sdot z19.s, z27.b, z28.b
    ld1rw {z29.s}, p0/z, [x3, #92]
    sdot z20.s, z24.b, z29.b
    sdot z21.s, z25.b, z29.b
    sdot z22.s, z26.b, z29.b
    sdot z23.s, z27.b, z29.b
    add x0, x0, #512
    add x3, x3, #96
    subs x8, x8, #1
    b.ne .Lpk
    ld1w {z24.s}, p0/z, [x0, #0, mul vl]
    ld1w {z25.s}, p0/z, [x0, #1, mul vl]
    ld1w {z26.s}, p0/z, [x0, #2, mul vl]
    ld1w {z27.s}, p0/z, [x0, #3, mul vl]
    add x0, x0, #256
    mov x9, x5
    ld1rw {z28.s}, p0/z, [x4, #0]
    scvtf z0.s, p0/m, z0.s
    fmul z29.s, z24.s, z28.s
    ld1w {z30.s}, p0/z, [x9, #0, mul vl]
    fmla z30.s, p0/m, z0.s, z29.s
    st1w {z30.s}, p0, [x9, #0, mul vl]
    scvtf z1.s, p0/m, z1.s
    fmul z29.s, z25.s, z28.s
    ld1w {z30.s}, p0/z, [x9, #1, mul vl]
    fmla z30.s, p0/m, z1.s, z29.s
    st1w {z30.s}, p0, [x9, #1, mul vl]
    scvtf z2.s, p0/m, z2.s
    fmul z29.s, z26.s, z28.s
    ld1w {z30.s}, p0/z, [x9, #2, mul vl]
    fmla z30.s, p0/m, z2.s, z29.s
    st1w {z30.s}, p0, [x9, #2, mul vl]
    scvtf z3.s, p0/m, z3.s
    fmul z29.s, z27.s, z28.s
    ld1w {z30.s}, p0/z, [x9, #3, mul vl]
    fmla z30.s, p0/m, z3.s, z29.s
    st1w {z30.s}, p0, [x9, #3, mul vl]
    add x9, x9, x6
    ld1rw {z28.s}, p0/z, [x4, #4]
    scvtf z4.s, p0/m, z4.s
    fmul z29.s, z24.s, z28.s
    ld1w {z30.s}, p0/z, [x9, #0, mul vl]
    fmla z30.s, p0/m, z4.s, z29.s
    st1w {z30.s}, p0, [x9, #0, mul vl]
    scvtf z5.s, p0/m, z5.s
    fmul z29.s, z25.s, z28.s
    ld1w {z30.s}, p0/z, [x9, #1, mul vl]
    fmla z30.s, p0/m, z5.s, z29.s
    st1w {z30.s}, p0, [x9, #1, mul vl]
    scvtf z6.s, p0/m, z6.s
    fmul z29.s, z26.s, z28.s
    ld1w {z30.s}, p0/z, [x9, #2, mul vl]
    fmla z30.s, p0/m, z6.s, z29.s
    st1w {z30.s}, p0, [x9, #2, mul vl]
    scvtf z7.s, p0/m, z7.s
    fmul z29.s, z27.s, z28.s
    ld1w {z30.s}, p0/z, [x9, #3, mul vl]
    fmla z30.s, p0/m, z7.s, z29.s
    st1w {z30.s}, p0, [x9, #3, mul vl]
    add x9, x9, x6
    ld1rw {z28.s}, p0/z, [x4, #8]
    scvtf z8.s, p0/m, z8.s
    fmul z29.s, z24.s, z28.s
    ld1w {z30.s}, p0/z, [x9, #0, mul vl]
    fmla z30.s, p0/m, z8.s, z29.s
    st1w {z30.s}, p0, [x9, #0, mul vl]
    scvtf z9.s, p0/m, z9.s
    fmul z29.s, z25.s, z28.s
    ld1w {z30.s}, p0/z, [x9, #1, mul vl]
    fmla z30.s, p0/m, z9.s, z29.s
    st1w {z30.s}, p0, [x9, #1, mul vl]
    scvtf z10.s, p0/m, z10.s
    fmul z29.s, z26.s, z28.s
    ld1w {z30.s}, p0/z, [x9, #2, mul vl]
    fmla z30.s, p0/m, z10.s, z29.s
    st1w {z30.s}, p0, [x9, #2, mul vl]
    scvtf z11.s, p0/m, z11.s
    fmul z29.s, z27.s, z28.s
    ld1w {z30.s}, p0/z, [x9, #3, mul vl]
    fmla z30.s, p0/m, z11.s, z29.s
    st1w {z30.s}, p0, [x9, #3, mul vl]
    add x9, x9, x6
    ld1rw {z28.s}, p0/z, [x4, #12]
    scvtf z12.s, p0/m, z12.s
    fmul z29.s, z24.s, z28.s
    ld1w {z30.s}, p0/z, [x9, #0, mul vl]
    fmla z30.s, p0/m, z12.s, z29.s
    st1w {z30.s}, p0, [x9, #0, mul vl]
    scvtf z13.s, p0/m, z13.s
    fmul z29.s, z25.s, z28.s
    ld1w {z30.s}, p0/z, [x9, #1, mul vl]
    fmla z30.s, p0/m, z13.s, z29.s
    st1w {z30.s}, p0, [x9, #1, mul vl]
    scvtf z14.s, p0/m, z14.s
    fmul z29.s, z26.s, z28.s
    ld1w {z30.s}, p0/z, [x9, #2, mul vl]
    fmla z30.s, p0/m, z14.s, z29.s
    st1w {z30.s}, p0, [x9, #2, mul vl]
    scvtf z15.s, p0/m, z15.s
    fmul z29.s, z27.s, z28.s
    ld1w {z30.s}, p0/z, [x9, #3, mul vl]
    fmla z30.s, p0/m, z15.s, z29.s
    st1w {z30.s}, p0, [x9, #3, mul vl]
    add x9, x9, x6
    ld1rw {z28.s}, p0/z, [x4, #16]
    scvtf z16.s, p0/m, z16.s
    fmul z29.s, z24.s, z28.s
    ld1w {z30.s}, p0/z, [x9, #0, mul vl]
    fmla z30.s, p0/m, z16.s, z29.s
    st1w {z30.s}, p0, [x9, #0, mul vl]
    scvtf z17.s, p0/m, z17.s
    fmul z29.s, z25.s, z28.s
    ld1w {z30.s}, p0/z, [x9, #1, mul vl]
    fmla z30.s, p0/m, z17.s, z29.s
    st1w {z30.s}, p0, [x9, #1, mul vl]
    scvtf z18.s, p0/m, z18.s
    fmul z29.s, z26.s, z28.s
    ld1w {z30.s}, p0/z, [x9, #2, mul vl]
    fmla z30.s, p0/m, z18.s, z29.s
    st1w {z30.s}, p0, [x9, #2, mul vl]
    scvtf z19.s, p0/m, z19.s
    fmul z29.s, z27.s, z28.s
    ld1w {z30.s}, p0/z, [x9, #3, mul vl]
    fmla z30.s, p0/m, z19.s, z29.s
    st1w {z30.s}, p0, [x9, #3, mul vl]
    add x9, x9, x6
    ld1rw {z28.s}, p0/z, [x4, #20]
    scvtf z20.s, p0/m, z20.s
    fmul z29.s, z24.s, z28.s
    ld1w {z30.s}, p0/z, [x9, #0, mul vl]
    fmla z30.s, p0/m, z20.s, z29.s
    st1w {z30.s}, p0, [x9, #0, mul vl]
    scvtf z21.s, p0/m, z21.s
    fmul z29.s, z25.s, z28.s
    ld1w {z30.s}, p0/z, [x9, #1, mul vl]
    fmla z30.s, p0/m, z21.s, z29.s
    st1w {z30.s}, p0, [x9, #1, mul vl]
    scvtf z22.s, p0/m, z22.s
    fmul z29.s, z26.s, z28.s
    ld1w {z30.s}, p0/z, [x9, #2, mul vl]
    fmla z30.s, p0/m, z22.s, z29.s
    st1w {z30.s}, p0, [x9, #2, mul vl]
    scvtf z23.s, p0/m, z23.s
    fmul z29.s, z27.s, z28.s
    ld1w {z30.s}, p0/z, [x9, #3, mul vl]
    fmla z30.s, p0/m, z23.s, z29.s
    st1w {z30.s}, p0, [x9, #3, mul vl]
    add x9, x9, x6
    add x4, x4, #24
    subs x1, x1, #1
    b.ne .Lpblock
    ldp d14, d15, [sp, #48]
    ldp d12, d13, [sp, #32]
    ldp d10, d11, [sp, #16]
    ldp d8, d9, [sp], #64
    ret
    .size gk_gemm_tile6x4p_asm, .-gk_gemm_tile6x4p_asm
