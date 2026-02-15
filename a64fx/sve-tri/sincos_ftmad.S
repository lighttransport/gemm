/*
 * SVE trig-accelerator sin/cos for fp32 — A64FX
 *
 * Uses SVE trigonometric helper instructions:
 *   FTSSEL — select starting value (r or 1.0) with sign from quadrant
 *   FTSMUL — compute r^2 with sin/cos path bit in LSB
 *   FTMAD  — polynomial step with built-in minimax coefficients
 *
 * Algorithm:
 *   1. Cody-Waite range reduction (same as poly version)
 *   2. For sin: FTSSEL/FTSMUL with quadrant n, then FTMAD chain
 *   3. For cos: FTSSEL/FTSMUL with quadrant n+1, then FTMAD chain
 *      (because cos(x) = sin(x + pi/2), so use next quadrant)
 *   4. Sin and cos FTMAD chains are interleaved for ILP
 *
 * FTMAD has 8 built-in coefficient indices (imm 0-7).
 *
 * void sve_sincos_ftmad_f32(const float *theta, float *sin_out,
 *                           float *cos_out, int64_t n)
 *   x0 = theta, x1 = sin_out, x2 = cos_out, x3 = n
 */

    .text
    .global sve_sincos_ftmad_f32
    .type   sve_sincos_ftmad_f32, %function
    .align  6

sve_sincos_ftmad_f32:
    cbz     x3, .Lftmad_done

    ptrue   p0.s

    /* Load range-reduction constants */
    adr     x4, .Lftmad_consts
    ld1rw   {z16.s}, p0/z, [x4]        /* inv_pio2 = 2/pi       */
    ld1rw   {z17.s}, p0/z, [x4, #4]    /* pio2_hi               */
    ld1rw   {z18.s}, p0/z, [x4, #8]    /* pio2_lo               */

    mov     x5, #0                      /* index */

.Lftmad_loop:
    whilelt p0.s, x5, x3
    b.none  .Lftmad_done

    /* ---- Load theta ---- */
    ld1w    {z0.s}, p0/z, [x0, x5, lsl #2]

    /* ---- Range reduction ---- */
    fmul    z1.s, z0.s, z16.s          /* x * 2/pi              */
    frintn  z1.s, p0/m, z1.s           /* n_f = round           */
    fcvtzs  z2.s, p0/m, z1.s           /* n = (int32)n_f        */

    /* r = x - n_f * pio2_hi - n_f * pio2_lo */
    movprfx z3, z0
    fmls    z3.s, p0/m, z1.s, z17.s
    fmls    z3.s, p0/m, z1.s, z18.s    /* z3 = r                */

    /* ---- Sin path: quadrant n ---- */
    /* FTSSEL: starting value
     *   n&1==0 → start = r  (sin polynomial path)
     *   n&1==1 → start = 1  (cos polynomial path)
     *   n&2    → negate                               */
    ftssel  z4.s, z3.s, z2.s           /* z4 = sin starting val */

    /* FTSMUL: r^2 with sin/cos path marker in LSB */
    ftsmul  z5.s, z3.s, z2.s           /* z5 = sin_x2           */

    /* ---- Cos path: quadrant n+1 ---- */
    mov     z6.d, z2.d
    add     z6.s, z6.s, #1             /* z6 = n + 1            */
    ftssel  z7.s, z3.s, z6.s           /* z7 = cos starting val */
    ftsmul  z26.s, z3.s, z6.s          /* z26 = cos_x2          */

    /* ---- FTMAD polynomial chain (interleaved sin/cos) ---- */
    /* FTMAD: Zdn = coeff[imm] + Zdn * Zm
     * Coefficient selected by imm AND by path bit (LSB of Zm)
     * Initialize to 0, then chain from imm=7 down to imm=0    */

    mov     z0.s, #0                    /* sin accumulator       */
    mov     z1.s, #0                    /* cos accumulator       */

    /* 8-step polynomial (sin and cos interleaved for ILP) */
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

    /* Final: result = starting_value * polynomial */
    fmul    z0.s, z4.s, z0.s           /* sin = start * poly    */
    fmul    z1.s, z7.s, z1.s           /* cos = start * poly    */

    /* ---- Store ---- */
    st1w    {z0.s}, p0, [x1, x5, lsl #2]
    st1w    {z1.s}, p0, [x2, x5, lsl #2]

    incw    x5
    b       .Lftmad_loop

.Lftmad_done:
    ret

    .size   sve_sincos_ftmad_f32, . - sve_sincos_ftmad_f32

/* ---- Constants ---- */
    .section .rodata
    .align  4
.Lftmad_consts:
    .word   0x3F22F983      /* inv_pio2                            */
    .word   0x3FC90FDB      /* pio2_hi                             */
    .word   0xB33BBD2E      /* pio2_lo                             */
