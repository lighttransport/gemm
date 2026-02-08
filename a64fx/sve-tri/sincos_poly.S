/*
 * SVE polynomial sin/cos for fp32 — A64FX optimized
 *
 * Algorithm:
 *   1. Cody-Waite range reduction: r = x - round(x*2/pi) * pi/2
 *   2. Minimax polynomial (musl-derived, degree-9 sin / degree-8 cos):
 *        sin(r) = r + r^3*(S1 + r^2*(S2 + r^2*(S3 + r^2*S4)))
 *        cos(r) = 1 + r^2*(C0 + r^2*(C1 + r^2*(C2 + r^2*C3)))
 *   3. Quadrant adjustment via predicated select + negate
 *
 * Sin and cos polynomials are interleaved for 2-pipeline ILP.
 *
 * void sve_sincos_poly_f32(const float *theta, float *sin_out,
 *                          float *cos_out, int64_t n)
 *   x0 = theta, x1 = sin_out, x2 = cos_out, x3 = n
 */

    .text
    .global sve_sincos_poly_f32
    .type   sve_sincos_poly_f32, %function
    .align  6

sve_sincos_poly_f32:
    cbz     x3, .Lpoly_done

    /* p0 is caller-saved; must init before ld1rw broadcast */
    ptrue   p0.s

    /* Load constants into z16-z27 (caller-saved) */
    adr     x4, .Lpoly_consts
    ld1rw   {z16.s}, p0/z, [x4]        /* inv_pio2 = 2/pi       */
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

    mov     x5, #0                      /* index i = 0 */

.Lpoly_loop:
    whilelt p0.s, x5, x3               /* p0 = active mask */
    b.none  .Lpoly_done

    /* ---- Load theta ---- */
    ld1w    {z0.s}, p0/z, [x0, x5, lsl #2]

    /* ---- Range reduction ---- */
    fmul    z1.s, z0.s, z16.s          /* x * inv_pio2          */
    frintn  z1.s, p0/m, z1.s           /* n_f = round           */
    fcvtzs  z2.s, p0/m, z1.s           /* n = (int32)n_f        */

    movprfx z3, z0
    fmls    z3.s, p0/m, z1.s, z17.s    /* z3 = x - n_f*pio2_hi  */
    fmls    z3.s, p0/m, z1.s, z18.s    /* z3 = r (reduced)      */

    fmul    z4.s, z3.s, z3.s           /* z4 = r2               */

    /* ---- Sin polynomial (Horner, 4 terms): ---- */
    /* r + r^3*(S1 + r2*(S2 + r2*(S3 + r2*S4))) */
    movprfx z5, z22                     /* z5 = S4               */
    fmad    z5.s, p0/m, z4.s, z21.s    /* z5 = S4*r2 + S3      */
    fmad    z5.s, p0/m, z4.s, z20.s    /* z5 = (..)*r2 + S2    */
    fmad    z5.s, p0/m, z4.s, z19.s    /* z5 = (..)*r2 + S1    */

    /* ---- Cos polynomial (Horner, 4 terms): ---- */
    /* 1 + r2*(C0 + r2*(C1 + r2*(C2 + r2*C3))) */
    movprfx z6, z26                     /* z6 = C3               */
    fmad    z6.s, p0/m, z4.s, z25.s    /* z6 = C3*r2 + C2      */
    fmad    z6.s, p0/m, z4.s, z24.s    /* z6 = (..)*r2 + C1    */
    fmad    z6.s, p0/m, z4.s, z23.s    /* z6 = (..)*r2 + C0    */

    /* sin: sin_r = r + r*r2*ps */
    fmul    z5.s, z5.s, z4.s           /* z5 = r2 * ps          */
    fmad    z5.s, p0/m, z3.s, z3.s     /* z5 = z5*r + r = sin_r */

    /* cos: cos_r = 1 + r2*pc */
    mov     z7.d, z27.d                 /* z7 = 1.0              */
    fmla    z7.s, p0/m, z4.s, z6.s     /* z7 = 1 + r2*pc=cos_r */

    /* z5 = sin(r), z7 = cos(r) */

    /* ---- Quadrant adjustment ---- */
    mov     z0.d, z2.d
    and     z0.s, z0.s, #1
    cmpne   p1.s, p0/z, z0.s, #0       /* p1 = odd quadrant     */

    sel     z0.s, p1, z7.s, z5.s       /* sin_out               */
    sel     z1.s, p1, z5.s, z7.s       /* cos_out               */

    mov     z6.d, z2.d
    and     z6.s, z6.s, #2
    cmpne   p2.s, p0/z, z6.s, #0
    fneg    z0.s, p2/m, z0.s           /* negate sin if n&2     */

    mov     z6.d, z2.d
    add     z6.s, z6.s, #1
    and     z6.s, z6.s, #2
    cmpne   p3.s, p0/z, z6.s, #0
    fneg    z1.s, p3/m, z1.s           /* negate cos if (n+1)&2 */

    /* ---- Store results ---- */
    st1w    {z0.s}, p0, [x1, x5, lsl #2]
    st1w    {z1.s}, p0, [x2, x5, lsl #2]

    incw    x5
    b       .Lpoly_loop

.Lpoly_done:
    ret

    .size   sve_sincos_poly_f32, . - sve_sincos_poly_f32

/* ---- Constants ---- */
    .section .rodata
    .align  4
.Lpoly_consts:
    .word   0x3F22F983      /* inv_pio2 = 2/pi                    */
    .word   0x3FC90FDB      /* pio2_hi  = 0x1.921fb6p0            */
    .word   0xB33BBD2E      /* pio2_lo  = -0x1.777a5cp-25         */
    .word   0xBE2AAAA4      /* S1 = -1.6666654611e-01             */
    .word   0x3C0887A2      /* S2 =  8.3321608736e-03             */
    .word   0xB94FB975      /* S3 = -1.9515295891e-04             */
    .word   0x362DD514      /* S4 =  2.5903036289e-06             */
    .word   0xBF000000      /* C0 = -5.0000000000e-01             */
    .word   0x3D2AAA9E      /* C1 =  4.1666664568e-02             */
    .word   0xBAB6077D      /* C2 = -1.3887316250e-03             */
    .word   0x37CCF5CE      /* C3 =  2.4433156796e-05             */
    .word   0x3F800000      /* one = 1.0f                          */
