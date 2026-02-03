/*
 * Baseline GEMM for comparison
 * O[4][64] += P[4] * V[64] per K iteration
 * Uses LD1RW for broadcasting P values
 */

    .arch armv8.2-a+sve
    .text
    .align 4

/*
 * gemm_baseline_4x4(P, V, O, Nc, ld_v, ld_o)
 *   x0: P[4][Nc] - pre-computed exp2 values
 *   x1: V[Nc][64] - value matrix
 *   x2: O[4][64] - output
 *   x3: Nc - loop count
 *   x4: ld_v - V stride in bytes
 *   x5: ld_o - O stride in bytes
 */
    .global gemm_baseline_4x4
    .type gemm_baseline_4x4, %function
gemm_baseline_4x4:
    stp     d8, d9, [sp, #-64]!
    stp     d10, d11, [sp, #16]
    stp     d12, d13, [sp, #32]
    stp     d14, d15, [sp, #48]

    ptrue   p0.s

    /* Zero accumulators */
    fmov    z0.s, #0
    fmov    z1.s, #0
    fmov    z2.s, #0
    fmov    z3.s, #0
    fmov    z4.s, #0
    fmov    z5.s, #0
    fmov    z6.s, #0
    fmov    z7.s, #0
    fmov    z8.s, #0
    fmov    z9.s, #0
    fmov    z10.s, #0
    fmov    z11.s, #0
    fmov    z12.s, #0
    fmov    z13.s, #0
    fmov    z14.s, #0
    fmov    z15.s, #0

    cbz     x3, .Lgemm_store

    /* P row pointers (stride = Nc * 4 bytes) */
    lsl     x6, x3, #2                  // Nc * 4
    mov     x7, x0                      // P row 0
    add     x8, x0, x6                  // P row 1
    add     x9, x8, x6                  // P row 2
    add     x10, x9, x6                 // P row 3

    mov     x11, x3                     // loop counter

.Lgemm_loop:
    /* LD1RW: Broadcast P values (runs on LD/ST pipe) */
    ld1rw   {z24.s}, p0/z, [x7]         // Broadcast P[0,k]
    ld1rw   {z25.s}, p0/z, [x8]         // Broadcast P[1,k]
    ld1rw   {z26.s}, p0/z, [x9]         // Broadcast P[2,k]
    ld1rw   {z27.s}, p0/z, [x10]        // Broadcast P[3,k]
    add     x7, x7, #4
    add     x8, x8, #4
    add     x9, x9, #4
    add     x10, x10, #4

    /* Load V[k, :] - 4 vectors */
    ld1w    {z28.s}, p0/z, [x1, #0, mul vl]
    ld1w    {z29.s}, p0/z, [x1, #1, mul vl]
    ld1w    {z30.s}, p0/z, [x1, #2, mul vl]
    ld1w    {z31.s}, p0/z, [x1, #3, mul vl]
    add     x1, x1, x4                  // V += ld_v

    /* 16 FMLAs */
    fmla    z0.s, p0/m, z24.s, z28.s
    fmla    z1.s, p0/m, z24.s, z29.s
    fmla    z2.s, p0/m, z24.s, z30.s
    fmla    z3.s, p0/m, z24.s, z31.s
    fmla    z4.s, p0/m, z25.s, z28.s
    fmla    z5.s, p0/m, z25.s, z29.s
    fmla    z6.s, p0/m, z25.s, z30.s
    fmla    z7.s, p0/m, z25.s, z31.s
    fmla    z8.s, p0/m, z26.s, z28.s
    fmla    z9.s, p0/m, z26.s, z29.s
    fmla    z10.s, p0/m, z26.s, z30.s
    fmla    z11.s, p0/m, z26.s, z31.s
    fmla    z12.s, p0/m, z27.s, z28.s
    fmla    z13.s, p0/m, z27.s, z29.s
    fmla    z14.s, p0/m, z27.s, z30.s
    fmla    z15.s, p0/m, z27.s, z31.s

    subs    x11, x11, #1
    b.gt    .Lgemm_loop

.Lgemm_store:
    /* Store output */
    mov     x6, x2
    st1w    {z0.s}, p0, [x6, #0, mul vl]
    st1w    {z1.s}, p0, [x6, #1, mul vl]
    st1w    {z2.s}, p0, [x6, #2, mul vl]
    st1w    {z3.s}, p0, [x6, #3, mul vl]
    add     x6, x6, x5
    st1w    {z4.s}, p0, [x6, #0, mul vl]
    st1w    {z5.s}, p0, [x6, #1, mul vl]
    st1w    {z6.s}, p0, [x6, #2, mul vl]
    st1w    {z7.s}, p0, [x6, #3, mul vl]
    add     x6, x6, x5
    st1w    {z8.s}, p0, [x6, #0, mul vl]
    st1w    {z9.s}, p0, [x6, #1, mul vl]
    st1w    {z10.s}, p0, [x6, #2, mul vl]
    st1w    {z11.s}, p0, [x6, #3, mul vl]
    add     x6, x6, x5
    st1w    {z12.s}, p0, [x6, #0, mul vl]
    st1w    {z13.s}, p0, [x6, #1, mul vl]
    st1w    {z14.s}, p0, [x6, #2, mul vl]
    st1w    {z15.s}, p0, [x6, #3, mul vl]

    ldp     d14, d15, [sp, #48]
    ldp     d12, d13, [sp, #32]
    ldp     d10, d11, [sp, #16]
    ldp     d8, d9, [sp], #64
    ret

    .size gemm_baseline_4x4, .-gemm_baseline_4x4
