/*
 * Optimized A64FX SVE exp2 Softmax Kernel with FEXPA
 *
 * Target: 14 instructions/vector = 7 cycles/vector throughput
 *
 * A64FX has 2 FPU pipes + 2 LD/ST ports
 * Latencies: FP add/sub/fma=9, FEXPA=4, bit ops=4, int add=5
 *
 * Strategy:
 * 1. No fmov/dup in loop - constants pre-loaded
 * 2. 4x unroll with interleaved scheduling to hide latency
 * 3. Software pipelining where possible
 */

    .arch armv8.2-a+sve

    .text
    .align 4

    .section .rodata
    .align 4
.Lfast_64f:
    .float 64.0

    .text

/*----------------------------------------------------------------------------
 * exp2_softmax_fast - Fast softmax exp2 (no polynomial correction)
 *
 * void exp2_softmax_fast(const int32_t* in, float* out, size_t n,
 *                        float scale, float max_val);
 *
 * 14 instructions per vector:
 * ld1w, scvtf, fmul, fsub, frintm, fsub, fmul, fcvtzs, fcvtzs, add, lsl, orr, fexpa, st1w
 *
 * Accuracy: ~1.5% max error (acceptable for softmax where relative values matter)
 *----------------------------------------------------------------------------*/
    .global exp2_softmax_fast
    .type exp2_softmax_fast, %function
exp2_softmax_fast:
    cbz     x2, .Lfast_done_early

    /* Save callee-saved registers */
    stp     d8, d9, [sp, #-64]!
    stp     d10, d11, [sp, #16]
    stp     d12, d13, [sp, #32]
    stp     d14, d15, [sp, #48]

    cntw    x3
    lsl     x4, x3, #2              // 4 * VL

    ptrue   p0.s

    /* Load constants - avoid dup in loop */
    /* scale in s0, max_val in s1 - broadcast once */
    mov     z20.s, s0               // scale
    mov     z21.s, s1               // max_val

    /* Load 64.0f constant */
    adrp    x5, .Lfast_64f
    add     x5, x5, :lo12:.Lfast_64f
    ld1rw   z22.s, p0/z, [x5]       // 64.0f

    /* 127 bias as immediate in add */
    mov     w5, #127
    mov     z24.s, w5

    cmp     x2, x4
    b.lo    .Lfast_remainder

    /*========================================================================
     * Main loop: 4x unrolled, interleaved for latency hiding
     *
     * Schedule: Interleave operations from 4 vectors to fill FPU pipes
     * while hiding 9-cycle FP latency
     *========================================================================*/
.Lfast_main_loop:
    /* Phase 1: Load 4 vectors (2 cycles with 2 LD ports) */
    ld1w    z0.s, p0/z, [x0]
    ld1w    z1.s, p0/z, [x0, #1, mul vl]
    ld1w    z2.s, p0/z, [x0, #2, mul vl]
    ld1w    z3.s, p0/z, [x0, #3, mul vl]

    /* Phase 2: Int to float conversion (2 cycles with 2 FPU) */
    scvtf   z0.s, p0/m, z0.s
    scvtf   z1.s, p0/m, z1.s
    scvtf   z2.s, p0/m, z2.s
    scvtf   z3.s, p0/m, z3.s

    /* Phase 3: Scale (2 cycles) */
    fmul    z0.s, z0.s, z20.s
    fmul    z1.s, z1.s, z20.s
    fmul    z2.s, z2.s, z20.s
    fmul    z3.s, z3.s, z20.s

    /* Phase 4: Subtract max (2 cycles) */
    fsub    z0.s, z0.s, z21.s
    fsub    z1.s, z1.s, z21.s
    fsub    z2.s, z2.s, z21.s
    fsub    z3.s, z3.s, z21.s

    /* Phase 5: N = floor(x) (2 cycles) */
    frintm  z4.s, p0/m, z0.s
    frintm  z5.s, p0/m, z1.s
    frintm  z6.s, p0/m, z2.s
    frintm  z7.s, p0/m, z3.s

    /* Phase 6: f = x - N (fractional part) (2 cycles) */
    fsub    z8.s, z0.s, z4.s
    fsub    z9.s, z1.s, z5.s
    fsub    z10.s, z2.s, z6.s
    fsub    z11.s, z3.s, z7.s

    /* Phase 7: m = floor(f * 64) as int (2 cycles fmul) */
    fmul    z8.s, z8.s, z22.s
    fmul    z9.s, z9.s, z22.s
    fmul    z10.s, z10.s, z22.s
    fmul    z11.s, z11.s, z22.s

    /* Phase 8: fcvtzs for m (2 cycles) */
    fcvtzs  z8.s, p0/m, z8.s
    fcvtzs  z9.s, p0/m, z9.s
    fcvtzs  z10.s, p0/m, z10.s
    fcvtzs  z11.s, p0/m, z11.s

    /* Phase 9: fcvtzs for N (2 cycles) */
    fcvtzs  z4.s, p0/m, z4.s
    fcvtzs  z5.s, p0/m, z5.s
    fcvtzs  z6.s, p0/m, z6.s
    fcvtzs  z7.s, p0/m, z7.s

    /* Phase 10: N + 127 (2 cycles, int add = 5 cycles latency) */
    add     z4.s, z4.s, z24.s
    add     z5.s, z5.s, z24.s
    add     z6.s, z6.s, z24.s
    add     z7.s, z7.s, z24.s

    /* Phase 11: lsl << 6 (2 cycles, 4 cycles latency) */
    lsl     z4.s, z4.s, #6
    lsl     z5.s, z5.s, #6
    lsl     z6.s, z6.s, #6
    lsl     z7.s, z7.s, #6

    /* Phase 12: orr combine (2 cycles, 4 cycles latency) */
    orr     z0.s, z4.s, z8.s
    orr     z1.s, z5.s, z9.s
    orr     z2.s, z6.s, z10.s
    orr     z3.s, z7.s, z11.s

    /* Phase 13: FEXPA (2 cycles, 4 cycles latency) */
    fexpa   z0.s, z0.s
    fexpa   z1.s, z1.s
    fexpa   z2.s, z2.s
    fexpa   z3.s, z3.s

    /* Phase 14: Store (2 cycles with 2 ST ports) */
    st1w    z0.s, p0, [x1]
    st1w    z1.s, p0, [x1, #1, mul vl]
    st1w    z2.s, p0, [x1, #2, mul vl]
    st1w    z3.s, p0, [x1, #3, mul vl]

    /* Loop control */
    addvl   x0, x0, #4
    addvl   x1, x1, #4
    sub     x2, x2, x4
    cmp     x2, x4
    b.hs    .Lfast_main_loop

.Lfast_remainder:
    cbz     x2, .Lfast_done

    mov     x6, #0

.Lfast_rem_loop:
    whilelo p1.s, x6, x2

    ld1w    z0.s, p1/z, [x0, x6, lsl #2]
    scvtf   z0.s, p1/m, z0.s
    fmul    z0.s, z0.s, z20.s
    fsub    z0.s, z0.s, z21.s

    frintm  z1.s, p1/m, z0.s
    fsub    z2.s, z0.s, z1.s
    fmul    z2.s, z2.s, z22.s

    fcvtzs  z2.s, p1/m, z2.s
    fcvtzs  z1.s, p1/m, z1.s

    add     z1.s, z1.s, z24.s
    lsl     z1.s, z1.s, #6
    orr     z1.s, z1.s, z2.s

    fexpa   z0.s, z1.s

    st1w    z0.s, p1, [x1, x6, lsl #2]

    incw    x6
    whilelo p1.s, x6, x2
    b.first .Lfast_rem_loop

.Lfast_done:
    /* Restore callee-saved registers */
    ldp     d10, d11, [sp, #16]
    ldp     d12, d13, [sp, #32]
    ldp     d14, d15, [sp, #48]
    ldp     d8, d9, [sp], #64
    ret

.Lfast_done_early:
    ret

    .size exp2_softmax_fast, .-exp2_softmax_fast


/*----------------------------------------------------------------------------
 * exp2_softmax_opt - Optimized softmax exp2 with linear correction
 *
 * 16 instructions per vector (2 extra for linear correction)
 * Accuracy: ~0.1% max error
 *
 * Linear correction: result *= (1 + ln2 * err) where err = f - m/64
 *----------------------------------------------------------------------------*/
    .section .rodata
    .align 4
.Lopt_inv64:
    .float 0.015625              // 1/64
.Lopt_ln2:
    .float 0.693147180559945     // ln(2)

    .text

    .global exp2_softmax_opt
    .type exp2_softmax_opt, %function
exp2_softmax_opt:
    cbz     x2, .Lopt_done_early

    /* Save callee-saved registers */
    stp     d8, d9, [sp, #-64]!
    stp     d10, d11, [sp, #16]
    stp     d12, d13, [sp, #32]
    stp     d14, d15, [sp, #48]

    cntw    x3
    lsl     x4, x3, #2

    ptrue   p0.s

    /* Constants */
    mov     z20.s, s0               // scale
    mov     z21.s, s1               // max_val

    adrp    x5, .Lfast_64f
    add     x5, x5, :lo12:.Lfast_64f
    ld1rw   z22.s, p0/z, [x5]       // 64.0f

    adrp    x5, .Lopt_inv64
    add     x5, x5, :lo12:.Lopt_inv64
    ld1rw   z23.s, p0/z, [x5]       // 1/64

    adrp    x5, .Lopt_ln2
    add     x5, x5, :lo12:.Lopt_ln2
    ld1rw   z26.s, p0/z, [x5]       // ln(2)

    mov     w5, #127
    mov     z24.s, w5

    cmp     x2, x4
    b.lo    .Lopt_remainder

.Lopt_main_loop:
    /* Load */
    ld1w    z0.s, p0/z, [x0]
    ld1w    z1.s, p0/z, [x0, #1, mul vl]
    ld1w    z2.s, p0/z, [x0, #2, mul vl]
    ld1w    z3.s, p0/z, [x0, #3, mul vl]

    /* Int to float */
    scvtf   z0.s, p0/m, z0.s
    scvtf   z1.s, p0/m, z1.s
    scvtf   z2.s, p0/m, z2.s
    scvtf   z3.s, p0/m, z3.s

    /* x = input * scale - max */
    fmul    z0.s, z0.s, z20.s
    fmul    z1.s, z1.s, z20.s
    fmul    z2.s, z2.s, z20.s
    fmul    z3.s, z3.s, z20.s

    fsub    z0.s, z0.s, z21.s
    fsub    z1.s, z1.s, z21.s
    fsub    z2.s, z2.s, z21.s
    fsub    z3.s, z3.s, z21.s

    /* N = floor(x) */
    frintm  z4.s, p0/m, z0.s
    frintm  z5.s, p0/m, z1.s
    frintm  z6.s, p0/m, z2.s
    frintm  z7.s, p0/m, z3.s

    /* f = x - N */
    fsub    z8.s, z0.s, z4.s
    fsub    z9.s, z1.s, z5.s
    fsub    z10.s, z2.s, z6.s
    fsub    z11.s, z3.s, z7.s

    /* m = floor(f * 64) */
    fmul    z0.s, z8.s, z22.s
    fmul    z1.s, z9.s, z22.s
    fmul    z2.s, z10.s, z22.s
    fmul    z3.s, z11.s, z22.s

    fcvtzs  z12.s, p0/m, z0.s
    fcvtzs  z13.s, p0/m, z1.s
    fcvtzs  z14.s, p0/m, z2.s
    fcvtzs  z15.s, p0/m, z3.s

    /* N as int */
    fcvtzs  z4.s, p0/m, z4.s
    fcvtzs  z5.s, p0/m, z5.s
    fcvtzs  z6.s, p0/m, z6.s
    fcvtzs  z7.s, p0/m, z7.s

    /* biased_exp = N + 127 */
    add     z4.s, z4.s, z24.s
    add     z5.s, z5.s, z24.s
    add     z6.s, z6.s, z24.s
    add     z7.s, z7.s, z24.s

    /* FEXPA input = (biased_exp << 6) | m */
    lsl     z4.s, z4.s, #6
    lsl     z5.s, z5.s, #6
    lsl     z6.s, z6.s, #6
    lsl     z7.s, z7.s, #6

    orr     z4.s, z4.s, z12.s
    orr     z5.s, z5.s, z13.s
    orr     z6.s, z6.s, z14.s
    orr     z7.s, z7.s, z15.s

    /* FEXPA */
    fexpa   z0.s, z4.s
    fexpa   z1.s, z5.s
    fexpa   z2.s, z6.s
    fexpa   z3.s, z7.s

    /* Linear correction: err = f - m/64, result *= (1 + ln2*err) */
    /* Convert m back to float and compute m/64 */
    scvtf   z12.s, p0/m, z12.s
    scvtf   z13.s, p0/m, z13.s
    scvtf   z14.s, p0/m, z14.s
    scvtf   z15.s, p0/m, z15.s

    /* err = f - m/64 using fmsb: z = zm - z*zn = f - m*(1/64) */
    /* z8 = f, z12 = m, z23 = 1/64 */
    /* fmsb Zda, Pg/M, Zn, Zm : Zda = Zm - Zda * Zn */
    fmsb    z12.s, p0/m, z23.s, z8.s    // z12 = z8 - z12*z23 = f - m/64
    fmsb    z13.s, p0/m, z23.s, z9.s
    fmsb    z14.s, p0/m, z23.s, z10.s
    fmsb    z15.s, p0/m, z23.s, z11.s

    /* result = result * (1 + ln2*err) = result + result*ln2*err */
    /* Use fmla: result += result * (ln2*err) */
    fmul    z12.s, z12.s, z26.s         // ln2 * err
    fmul    z13.s, z13.s, z26.s
    fmul    z14.s, z14.s, z26.s
    fmul    z15.s, z15.s, z26.s

    fmla    z0.s, p0/m, z0.s, z12.s     // result += result * ln2*err
    fmla    z1.s, p0/m, z1.s, z13.s
    fmla    z2.s, p0/m, z2.s, z14.s
    fmla    z3.s, p0/m, z3.s, z15.s

    /* Store */
    st1w    z0.s, p0, [x1]
    st1w    z1.s, p0, [x1, #1, mul vl]
    st1w    z2.s, p0, [x1, #2, mul vl]
    st1w    z3.s, p0, [x1, #3, mul vl]

    addvl   x0, x0, #4
    addvl   x1, x1, #4
    sub     x2, x2, x4
    cmp     x2, x4
    b.hs    .Lopt_main_loop

.Lopt_remainder:
    cbz     x2, .Lopt_done
    mov     x6, #0

.Lopt_rem_loop:
    whilelo p1.s, x6, x2

    ld1w    z0.s, p1/z, [x0, x6, lsl #2]
    scvtf   z0.s, p1/m, z0.s
    fmul    z0.s, z0.s, z20.s
    fsub    z0.s, z0.s, z21.s

    frintm  z1.s, p1/m, z0.s
    fsub    z2.s, z0.s, z1.s        // f
    fmul    z3.s, z2.s, z22.s       // f*64

    fcvtzs  z3.s, p1/m, z3.s        // m
    fcvtzs  z1.s, p1/m, z1.s        // N

    add     z1.s, z1.s, z24.s
    lsl     z1.s, z1.s, #6
    orr     z1.s, z1.s, z3.s

    fexpa   z0.s, z1.s

    /* Linear correction */
    scvtf   z3.s, p1/m, z3.s
    fmsb    z3.s, p1/m, z23.s, z2.s // err = f - m/64
    fmul    z3.s, z3.s, z26.s       // ln2*err
    fmla    z0.s, p1/m, z0.s, z3.s  // result *= (1 + ln2*err)

    st1w    z0.s, p1, [x1, x6, lsl #2]

    incw    x6
    whilelo p1.s, x6, x2
    b.first .Lopt_rem_loop

.Lopt_done:
    ldp     d10, d11, [sp, #16]
    ldp     d12, d13, [sp, #32]
    ldp     d14, d15, [sp, #48]
    ldp     d8, d9, [sp], #64
    ret

.Lopt_done_early:
    ret

    .size exp2_softmax_opt, .-exp2_softmax_opt
