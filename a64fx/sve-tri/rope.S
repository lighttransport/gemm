/*
 * Fused RoPE kernels — compute sincos + apply rotation in one pass
 *
 * For each pair (x[2i], x[2i+1]) with angle theta[i]:
 *   x'[2i]   = x[2i]*cos(theta[i]) - x[2i+1]*sin(theta[i])
 *   x'[2i+1] = x[2i]*sin(theta[i]) + x[2i+1]*cos(theta[i])
 *
 * Uses LD2W/ST2W for deinterleaving pairs.
 * With VL=512: processes 8 pairs (16 floats) per iteration.
 *
 * void sve_rope_poly_f32(float *x, const float *theta, int64_t dim)
 *   x0 = x (dim floats, in-place), x1 = theta (dim/2 floats), x2 = dim
 *
 * void sve_rope_ftmad_f32(float *x, const float *theta, int64_t dim)
 *   same signature
 */

    .text

/* ============================================================
 * Polynomial version (degree-9 sin / degree-8 cos)
 * ============================================================ */
    .global sve_rope_poly_f32
    .type   sve_rope_poly_f32, %function
    .align  6

sve_rope_poly_f32:
    lsr     x3, x2, #1                 /* n_pairs = dim / 2     */
    cbz     x3, .Lrpoly_done

    ptrue   p0.s

    /* Constants */
    adr     x4, .Lrpoly_consts
    ld1rw   {z16.s}, p0/z, [x4]        /* inv_pio2              */
    ld1rw   {z17.s}, p0/z, [x4, #4]    /* pio2_hi               */
    ld1rw   {z18.s}, p0/z, [x4, #8]    /* pio2_lo               */
    ld1rw   {z19.s}, p0/z, [x4, #12]   /* S1                    */
    ld1rw   {z20.s}, p0/z, [x4, #16]   /* S2                    */
    ld1rw   {z21.s}, p0/z, [x4, #20]   /* S3                    */
    ld1rw   {z22.s}, p0/z, [x4, #24]   /* S4                    */
    ld1rw   {z23.s}, p0/z, [x4, #28]   /* C0                    */
    ld1rw   {z24.s}, p0/z, [x4, #32]   /* C1                    */
    ld1rw   {z25.s}, p0/z, [x4, #36]   /* C2                    */
    ld1rw   {z26.s}, p0/z, [x4, #40]   /* C3                    */
    ld1rw   {z27.s}, p0/z, [x4, #44]   /* one = 1.0f            */

    mov     x5, #0                      /* pair index            */

.Lrpoly_loop:
    whilelt p0.s, x5, x3
    b.none  .Lrpoly_done

    /* Load theta[i] */
    ld1w    {z0.s}, p0/z, [x1, x5, lsl #2]

    /* ---- Range reduction ---- */
    fmul    z1.s, z0.s, z16.s
    frintn  z1.s, p0/m, z1.s
    fcvtzs  z2.s, p0/m, z1.s           /* n (int32)             */

    movprfx z3, z0
    fmls    z3.s, p0/m, z1.s, z17.s
    fmls    z3.s, p0/m, z1.s, z18.s    /* r                     */

    fmul    z4.s, z3.s, z3.s           /* r2                    */

    /* ---- Sin polynomial (4 terms) ---- */
    movprfx z5, z22                     /* S4                    */
    fmad    z5.s, p0/m, z4.s, z21.s    /* S4*r2 + S3            */
    fmad    z5.s, p0/m, z4.s, z20.s    /* ..*r2 + S2            */
    fmad    z5.s, p0/m, z4.s, z19.s    /* ..*r2 + S1            */

    /* ---- Cos polynomial (4 terms) ---- */
    movprfx z6, z26                     /* C3                    */
    fmad    z6.s, p0/m, z4.s, z25.s    /* C3*r2 + C2            */
    fmad    z6.s, p0/m, z4.s, z24.s    /* ..*r2 + C1            */
    fmad    z6.s, p0/m, z4.s, z23.s    /* ..*r2 + C0            */

    /* sin_r = r + r*r2*ps */
    fmul    z5.s, z5.s, z4.s
    fmad    z5.s, p0/m, z3.s, z3.s     /* sin_r                 */

    /* cos_r = 1 + r2*pc */
    mov     z7.d, z27.d
    fmla    z7.s, p0/m, z4.s, z6.s     /* cos_r                 */

    /* ---- Quadrant adjustment ---- */
    mov     z0.d, z2.d
    and     z0.s, z0.s, #1
    cmpne   p1.s, p0/z, z0.s, #0

    sel     z28.s, p1, z7.s, z5.s      /* sin_out               */
    sel     z29.s, p1, z5.s, z7.s      /* cos_out               */

    mov     z0.d, z2.d
    and     z0.s, z0.s, #2
    cmpne   p2.s, p0/z, z0.s, #0
    fneg    z28.s, p2/m, z28.s

    mov     z0.d, z2.d
    add     z0.s, z0.s, #1
    and     z0.s, z0.s, #2
    cmpne   p3.s, p0/z, z0.s, #0
    fneg    z29.s, p3/m, z29.s

    /* z28 = sin(theta), z29 = cos(theta) */

    /* ---- Load x pairs (deinterleaved) ---- */
    lsl     x6, x5, #3                 /* byte offset = i*8     */
    add     x7, x0, x6
    ld2w    {z30.s, z31.s}, p0/z, [x7] /* z30=even, z31=odd     */

    /* ---- Apply rotation ---- */
    fmul    z0.s, z30.s, z29.s
    fmls    z0.s, p0/m, z31.s, z28.s   /* x'_even               */

    fmul    z1.s, z31.s, z29.s
    fmla    z1.s, p0/m, z30.s, z28.s   /* x'_odd                */

    st2w    {z0.s, z1.s}, p0, [x7]

    incw    x5
    b       .Lrpoly_loop

.Lrpoly_done:
    ret

    .size   sve_rope_poly_f32, . - sve_rope_poly_f32

/* ============================================================
 * FTMAD version
 * ============================================================ */
    .global sve_rope_ftmad_f32
    .type   sve_rope_ftmad_f32, %function
    .align  6

sve_rope_ftmad_f32:
    lsr     x3, x2, #1
    cbz     x3, .Lrftmad_done

    ptrue   p0.s

    adr     x4, .Lrftmad_consts
    ld1rw   {z16.s}, p0/z, [x4]
    ld1rw   {z17.s}, p0/z, [x4, #4]
    ld1rw   {z18.s}, p0/z, [x4, #8]

    mov     x5, #0

.Lrftmad_loop:
    whilelt p0.s, x5, x3
    b.none  .Lrftmad_done

    ld1w    {z0.s}, p0/z, [x1, x5, lsl #2]

    fmul    z1.s, z0.s, z16.s
    frintn  z1.s, p0/m, z1.s
    fcvtzs  z2.s, p0/m, z1.s

    movprfx z3, z0
    fmls    z3.s, p0/m, z1.s, z17.s
    fmls    z3.s, p0/m, z1.s, z18.s

    /* Sin setup */
    ftssel  z4.s, z3.s, z2.s
    ftsmul  z5.s, z3.s, z2.s

    /* Cos setup (n+1) */
    mov     z6.d, z2.d
    add     z6.s, z6.s, #1
    ftssel  z7.s, z3.s, z6.s
    ftsmul  z26.s, z3.s, z6.s

    /* FTMAD chains (interleaved) */
    mov     z0.s, #0
    mov     z1.s, #0

    ftmad   z0.s, z0.s, z5.s, #7
    ftmad   z1.s, z1.s, z26.s, #7
    ftmad   z0.s, z0.s, z5.s, #6
    ftmad   z1.s, z1.s, z26.s, #6
    ftmad   z0.s, z0.s, z5.s, #5
    ftmad   z1.s, z1.s, z26.s, #5
    ftmad   z0.s, z0.s, z5.s, #4
    ftmad   z1.s, z1.s, z26.s, #4
    ftmad   z0.s, z0.s, z5.s, #3
    ftmad   z1.s, z1.s, z26.s, #3
    ftmad   z0.s, z0.s, z5.s, #2
    ftmad   z1.s, z1.s, z26.s, #2
    ftmad   z0.s, z0.s, z5.s, #1
    ftmad   z1.s, z1.s, z26.s, #1
    ftmad   z0.s, z0.s, z5.s, #0
    ftmad   z1.s, z1.s, z26.s, #0

    fmul    z28.s, z4.s, z0.s          /* sin(theta)            */
    fmul    z29.s, z7.s, z1.s          /* cos(theta)            */

    lsl     x6, x5, #3
    add     x7, x0, x6
    ld2w    {z30.s, z31.s}, p0/z, [x7]

    fmul    z0.s, z30.s, z29.s
    fmls    z0.s, p0/m, z31.s, z28.s

    fmul    z1.s, z31.s, z29.s
    fmla    z1.s, p0/m, z30.s, z28.s

    st2w    {z0.s, z1.s}, p0, [x7]

    incw    x5
    b       .Lrftmad_loop

.Lrftmad_done:
    ret

    .size   sve_rope_ftmad_f32, . - sve_rope_ftmad_f32

/* ---- Constants ---- */
    .section .rodata
    .align  4
.Lrpoly_consts:
    .word   0x3F22F983      /* inv_pio2                            */
    .word   0x3FC90FDB      /* pio2_hi                             */
    .word   0xB33BBD2E      /* pio2_lo                             */
    .word   0xBE2AAAA4      /* S1                                  */
    .word   0x3C0887A2      /* S2                                  */
    .word   0xB94FB975      /* S3                                  */
    .word   0x362DD514      /* S4                                  */
    .word   0xBF000000      /* C0                                  */
    .word   0x3D2AAA9E      /* C1                                  */
    .word   0xBAB6077D      /* C2                                  */
    .word   0x37CCF5CE      /* C3                                  */
    .word   0x3F800000      /* one                                 */

.Lrftmad_consts:
    .word   0x3F22F983
    .word   0x3FC90FDB
    .word   0xB33BBD2E
