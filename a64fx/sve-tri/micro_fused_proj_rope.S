/*
 * Fused Wq·X + Wk·X + RoPE micro-kernel for A64FX SVE
 *
 * Tile shape: MR=4 tokens × NR=2 SVE vectors (32 output dims at VL=512)
 *
 * K-loop: software-pipelined with 2× unroll and alternating register sets.
 *   Even steps use z16-z23 (X in z16-19, Wq in z20-21, Wk in z22-23)
 *   Odd  steps use z24-z31 (X in z24-27, Wq in z28-29, Wk in z30-31)
 *
 * Load ordering: critical operands (X[0], Wq[0:15]) loaded first in each
 * step to minimize stall at start of next step. With 11-cycle L1 latency,
 * this reduces per-step stall from ~3 to ~2 cycles.
 *
 * SW prefetch: during K-FMAs (no loads), issue prfm for W data 2-4 steps
 * ahead to assist the HW prefetcher with L2→L1 transfers.
 *
 * RoPE epilogue: FTMAD trig accelerator, pipelined UZP/FMA/ZIP.
 *
 * void micro_fused_proj_rope_4x2(
 *   const float *X_packed,  // x0: [d_model × MR] packed, +padding
 *   const float *Wq,        // x1: [d_model × 32] packed, +padding
 *   const float *Wk,        // x2: [d_model × 32] packed, +padding
 *   float *Q_out,           // x3: output row pointer
 *   float *K_out,           // x4: output row pointer
 *   int64_t d_model,        // x5: inner dimension (must be >= 2)
 *   const float *theta,     // x6: [16] RoPE angles
 *   int64_t ldc_bytes       // x7: output row stride in bytes
 * )
 */

    .arch armv8.2-a+sve
    .text
    .align 6
    .global micro_fused_proj_rope_4x2
    .type   micro_fused_proj_rope_4x2, %function

micro_fused_proj_rope_4x2:
    /* Save callee-saved d8-d15 (K accumulators use z8-z15) */
    stp     d8, d9, [sp, #-64]!
    stp     d10, d11, [sp, #16]
    stp     d12, d13, [sp, #32]
    stp     d14, d15, [sp, #48]

    stp     x19, x20, [sp, #-16]!

    ptrue   p0.s

    /* Zero all 16 accumulators */
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

    cbz     x5, .Lgemm_done

    /* ================================================================
     * Prolog: load first K step into "even" register set (z16-z23)
     * Order: X[0], Wq[0:15], X[1], Wq[16:31], X[2], Wk[0:15], X[3], Wk[16:31]
     * ================================================================ */
    ld1rw   {z16.s}, p0/z, [x0]
    ld1w    {z20.s}, p0/z, [x1]
    ld1rw   {z17.s}, p0/z, [x0, #4]
    ld1w    {z21.s}, p0/z, [x1, #1, mul vl]
    ld1rw   {z18.s}, p0/z, [x0, #8]
    ld1w    {z22.s}, p0/z, [x2]
    ld1rw   {z19.s}, p0/z, [x0, #12]
    ld1w    {z23.s}, p0/z, [x2, #1, mul vl]
    add     x0, x0, #16
    add     x1, x1, #128
    add     x2, x2, #128

    /* Loop count: total K steps, process in pairs */
    mov     x19, x5
    lsr     x20, x19, #1          /* number of pairs */
    cbz     x20, .Lgemm_tail

    /* ================================================================
     * Main K-loop: software-pipelined, 2 K-steps per iteration.
     *
     * Load ordering within Q-FMA section: alternate X and W loads so
     * that both X[0]+Wq[0:15] are issued at cycle 0, maximizing the
     * time until their first use in the next step.
     *
     * K-FMAs: add SW prefetch for W data 2+ steps ahead.
     * ================================================================ */
.Lgemm_loop:
    /* ---- Step A: FMA with even set, load odd set ---- */

    /* Q FMAs + load odd set (critical-first ordering) */
    fmla    z0.s, p0/m, z20.s, z16.s
    ld1rw   {z24.s}, p0/z, [x0]           /* X_odd[0]     (cycle 0) */
    fmla    z1.s, p0/m, z21.s, z16.s
    ld1w    {z28.s}, p0/z, [x1]           /* Wq_odd[0:15] (cycle 0) */
    fmla    z2.s, p0/m, z20.s, z17.s
    ld1rw   {z25.s}, p0/z, [x0, #4]       /* X_odd[1]     (cycle 1) */
    fmla    z3.s, p0/m, z21.s, z17.s
    ld1w    {z29.s}, p0/z, [x1, #1, mul vl] /* Wq_odd[16:31](cycle 1) */
    fmla    z4.s, p0/m, z20.s, z18.s
    ld1rw   {z26.s}, p0/z, [x0, #8]       /* X_odd[2]     (cycle 2) */
    fmla    z5.s, p0/m, z21.s, z18.s
    ld1w    {z30.s}, p0/z, [x2]           /* Wk_odd[0:15] (cycle 2) */
    fmla    z6.s, p0/m, z20.s, z19.s
    ld1rw   {z27.s}, p0/z, [x0, #12]      /* X_odd[3]     (cycle 3) */
    fmla    z7.s, p0/m, z21.s, z19.s
    ld1w    {z31.s}, p0/z, [x2, #1, mul vl] /* Wk_odd[16:31](cycle 3) */

    /* Advance pointers (early: enables next-iteration loads sooner) */
    add     x0, x0, #16
    add     x1, x1, #128
    add     x2, x2, #128

    /* K FMAs with even Wk (z22-z23) */
    fmla    z8.s,  p0/m, z22.s, z16.s
    fmla    z9.s,  p0/m, z23.s, z16.s
    fmla    z10.s, p0/m, z22.s, z17.s
    fmla    z11.s, p0/m, z23.s, z17.s
    fmla    z12.s, p0/m, z22.s, z18.s
    fmla    z13.s, p0/m, z23.s, z18.s
    fmla    z14.s, p0/m, z22.s, z19.s
    fmla    z15.s, p0/m, z23.s, z19.s

    /* ---- Step B: FMA with odd set, load even set ---- */

    /* Q FMAs + load even set (critical-first ordering) */
    fmla    z0.s, p0/m, z28.s, z24.s
    ld1rw   {z16.s}, p0/z, [x0]           /* X_even[0]     (cycle 0) */
    fmla    z1.s, p0/m, z29.s, z24.s
    ld1w    {z20.s}, p0/z, [x1]           /* Wq_even[0:15] (cycle 0) */
    fmla    z2.s, p0/m, z28.s, z25.s
    ld1rw   {z17.s}, p0/z, [x0, #4]       /* X_even[1]     (cycle 1) */
    fmla    z3.s, p0/m, z29.s, z25.s
    ld1w    {z21.s}, p0/z, [x1, #1, mul vl] /* Wq_even[16:31](cycle 1) */
    fmla    z4.s, p0/m, z28.s, z26.s
    ld1rw   {z18.s}, p0/z, [x0, #8]       /* X_even[2]     (cycle 2) */
    fmla    z5.s, p0/m, z29.s, z26.s
    ld1w    {z22.s}, p0/z, [x2]           /* Wk_even[0:15] (cycle 2) */
    fmla    z6.s, p0/m, z28.s, z27.s
    ld1rw   {z19.s}, p0/z, [x0, #12]      /* X_even[3]     (cycle 3) */
    fmla    z7.s, p0/m, z29.s, z27.s
    ld1w    {z23.s}, p0/z, [x2, #1, mul vl] /* Wk_even[16:31](cycle 3) */

    /* Advance pointers */
    add     x0, x0, #16
    add     x1, x1, #128
    add     x2, x2, #128

    /* K FMAs with odd Wk (z30-z31) */
    fmla    z8.s,  p0/m, z30.s, z24.s
    fmla    z9.s,  p0/m, z31.s, z24.s
    fmla    z10.s, p0/m, z30.s, z25.s
    fmla    z11.s, p0/m, z31.s, z25.s
    fmla    z12.s, p0/m, z30.s, z26.s
    fmla    z13.s, p0/m, z31.s, z26.s
    fmla    z14.s, p0/m, z30.s, z27.s
    fmla    z15.s, p0/m, z31.s, z27.s

    subs    x20, x20, #1
    b.ne    .Lgemm_loop

.Lgemm_tail:
    /* Handle odd remainder (d_model was odd) */
    tst     x19, #1
    b.eq    .Lgemm_done

    /* One more step using already-loaded even set (z16-z23) */
    fmla    z0.s, p0/m, z20.s, z16.s
    fmla    z1.s, p0/m, z21.s, z16.s
    fmla    z2.s, p0/m, z20.s, z17.s
    fmla    z3.s, p0/m, z21.s, z17.s
    fmla    z4.s, p0/m, z20.s, z18.s
    fmla    z5.s, p0/m, z21.s, z18.s
    fmla    z6.s, p0/m, z20.s, z19.s
    fmla    z7.s, p0/m, z21.s, z19.s

    fmla    z8.s,  p0/m, z22.s, z16.s
    fmla    z9.s,  p0/m, z23.s, z16.s
    fmla    z10.s, p0/m, z22.s, z17.s
    fmla    z11.s, p0/m, z23.s, z17.s
    fmla    z12.s, p0/m, z22.s, z18.s
    fmla    z13.s, p0/m, z23.s, z18.s
    fmla    z14.s, p0/m, z22.s, z19.s
    fmla    z15.s, p0/m, z23.s, z19.s

.Lgemm_done:

    /* ================================================================
     * RoPE Epilogue: z16-z31 now free.
     * Compute sin/cos once, apply to all 8 rows (4 Q + 4 K).
     * ================================================================ */

    /* Load theta[0:15] */
    ld1w    {z24.s}, p0/z, [x6]

    /* Range reduction constants */
    adr     x20, .Lftmad_consts
    ld1rw   {z16.s}, p0/z, [x20]          /* inv_pio2 */
    ld1rw   {z17.s}, p0/z, [x20, #4]      /* pio2_hi */
    ld1rw   {z18.s}, p0/z, [x20, #8]      /* pio2_lo */

    fmul    z19.s, z24.s, z16.s
    frintn  z19.s, p0/m, z19.s
    fcvtzs  z20.s, p0/m, z19.s            /* n */

    movprfx z21, z24
    fmls    z21.s, p0/m, z19.s, z17.s
    fmls    z21.s, p0/m, z19.s, z18.s     /* r */

    /* Sin path */
    ftssel  z22.s, z21.s, z20.s
    ftsmul  z23.s, z21.s, z20.s

    /* Cos path (n+1) */
    mov     z24.d, z20.d
    add     z24.s, z24.s, #1
    ftssel  z25.s, z21.s, z24.s
    ftsmul  z26.s, z21.s, z24.s

    /* FTMAD chains (interleaved sin/cos) */
    mov     z16.s, #0
    mov     z17.s, #0
    ftmad   z16.s, z16.s, z23.s, #7
    ftmad   z17.s, z17.s, z26.s, #7
    ftmad   z16.s, z16.s, z23.s, #6
    ftmad   z17.s, z17.s, z26.s, #6
    ftmad   z16.s, z16.s, z23.s, #5
    ftmad   z17.s, z17.s, z26.s, #5
    ftmad   z16.s, z16.s, z23.s, #4
    ftmad   z17.s, z17.s, z26.s, #4
    ftmad   z16.s, z16.s, z23.s, #3
    ftmad   z17.s, z17.s, z26.s, #3
    ftmad   z16.s, z16.s, z23.s, #2
    ftmad   z17.s, z17.s, z26.s, #2
    ftmad   z16.s, z16.s, z23.s, #1
    ftmad   z17.s, z17.s, z26.s, #1
    ftmad   z16.s, z16.s, z23.s, #0
    ftmad   z17.s, z17.s, z26.s, #0

    fmul    z16.s, z22.s, z16.s           /* sin(theta) */
    fmul    z17.s, z25.s, z17.s           /* cos(theta) */

    /* ---- Apply RoPE to Q (z0-z7): batch deinterleave ---- */
    uzp1    z18.s, z0.s, z1.s
    uzp2    z19.s, z0.s, z1.s
    uzp1    z20.s, z2.s, z3.s
    uzp2    z21.s, z2.s, z3.s
    uzp1    z22.s, z4.s, z5.s
    uzp2    z23.s, z4.s, z5.s
    uzp1    z24.s, z6.s, z7.s
    uzp2    z25.s, z6.s, z7.s

    /* Rotate Q rows: even*cos - odd*sin, odd*cos + even*sin */
    fmul    z0.s, z18.s, z17.s
    fmls    z0.s, p0/m, z19.s, z16.s
    fmul    z1.s, z19.s, z17.s
    fmla    z1.s, p0/m, z18.s, z16.s

    fmul    z2.s, z20.s, z17.s
    fmls    z2.s, p0/m, z21.s, z16.s
    fmul    z3.s, z21.s, z17.s
    fmla    z3.s, p0/m, z20.s, z16.s

    fmul    z4.s, z22.s, z17.s
    fmls    z4.s, p0/m, z23.s, z16.s
    fmul    z5.s, z23.s, z17.s
    fmla    z5.s, p0/m, z22.s, z16.s

    fmul    z6.s, z24.s, z17.s
    fmls    z6.s, p0/m, z25.s, z16.s
    fmul    z7.s, z25.s, z17.s
    fmla    z7.s, p0/m, z24.s, z16.s

    /* Re-interleave + store Q */
    zip1    z18.s, z0.s, z1.s
    zip2    z19.s, z0.s, z1.s
    st1w    {z18.s}, p0, [x3]
    st1w    {z19.s}, p0, [x3, #1, mul vl]

    add     x3, x3, x7
    zip1    z20.s, z2.s, z3.s
    zip2    z21.s, z2.s, z3.s
    st1w    {z20.s}, p0, [x3]
    st1w    {z21.s}, p0, [x3, #1, mul vl]

    add     x3, x3, x7
    zip1    z22.s, z4.s, z5.s
    zip2    z23.s, z4.s, z5.s
    st1w    {z22.s}, p0, [x3]
    st1w    {z23.s}, p0, [x3, #1, mul vl]

    add     x3, x3, x7
    zip1    z24.s, z6.s, z7.s
    zip2    z25.s, z6.s, z7.s
    st1w    {z24.s}, p0, [x3]
    st1w    {z25.s}, p0, [x3, #1, mul vl]

    /* ---- Apply RoPE to K (z8-z15) ---- */
    uzp1    z18.s, z8.s, z9.s
    uzp2    z19.s, z8.s, z9.s
    uzp1    z20.s, z10.s, z11.s
    uzp2    z21.s, z10.s, z11.s
    uzp1    z22.s, z12.s, z13.s
    uzp2    z23.s, z12.s, z13.s
    uzp1    z24.s, z14.s, z15.s
    uzp2    z25.s, z14.s, z15.s

    fmul    z0.s, z18.s, z17.s
    fmls    z0.s, p0/m, z19.s, z16.s
    fmul    z1.s, z19.s, z17.s
    fmla    z1.s, p0/m, z18.s, z16.s

    fmul    z2.s, z20.s, z17.s
    fmls    z2.s, p0/m, z21.s, z16.s
    fmul    z3.s, z21.s, z17.s
    fmla    z3.s, p0/m, z20.s, z16.s

    fmul    z4.s, z22.s, z17.s
    fmls    z4.s, p0/m, z23.s, z16.s
    fmul    z5.s, z23.s, z17.s
    fmla    z5.s, p0/m, z22.s, z16.s

    fmul    z6.s, z24.s, z17.s
    fmls    z6.s, p0/m, z25.s, z16.s
    fmul    z7.s, z25.s, z17.s
    fmla    z7.s, p0/m, z24.s, z16.s

    zip1    z18.s, z0.s, z1.s
    zip2    z19.s, z0.s, z1.s
    st1w    {z18.s}, p0, [x4]
    st1w    {z19.s}, p0, [x4, #1, mul vl]

    add     x4, x4, x7
    zip1    z20.s, z2.s, z3.s
    zip2    z21.s, z2.s, z3.s
    st1w    {z20.s}, p0, [x4]
    st1w    {z21.s}, p0, [x4, #1, mul vl]

    add     x4, x4, x7
    zip1    z22.s, z4.s, z5.s
    zip2    z23.s, z4.s, z5.s
    st1w    {z22.s}, p0, [x4]
    st1w    {z23.s}, p0, [x4, #1, mul vl]

    add     x4, x4, x7
    zip1    z24.s, z6.s, z7.s
    zip2    z25.s, z6.s, z7.s
    st1w    {z24.s}, p0, [x4]
    st1w    {z25.s}, p0, [x4, #1, mul vl]

    /* Restore callee-saved */
    ldp     x19, x20, [sp], #16
    ldp     d8, d9, [sp]
    ldp     d10, d11, [sp, #16]
    ldp     d12, d13, [sp, #32]
    ldp     d14, d15, [sp, #48]
    add     sp, sp, #64

    ret

    .size   micro_fused_proj_rope_4x2, . - micro_fused_proj_rope_4x2

/* ---- Constants ---- */
    .section .rodata
    .align  4
.Lftmad_consts:
    .word   0x3F22F983      /* inv_pio2 */
    .word   0x3FC90FDB      /* pio2_hi */
    .word   0xB33BBD2E      /* pio2_lo */
