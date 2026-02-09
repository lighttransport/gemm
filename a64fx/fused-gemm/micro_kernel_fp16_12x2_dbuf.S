// FP16 12×2 with B double-buffering — INIT variant
// Eliminates B load-use stall by preloading next B set during current K's FMAs
//
// Key insight: z24-z25 (B0) are free during K=1, z26-z27 (B1) free during K=0
// Preload B0 at START of K=1 → used next K=0 → 12+ cy distance → latency hidden
// Preload B1 in loop tail → used next K=1 → 12+ cy distance → latency hidden
//
// Registers:
//   Z0-Z23:  24 fp16 accumulators (12 rows × 2 col-groups)
//   Z24-Z25: B set 0 (K even)
//   Z26-Z27: B set 1 (K odd)
//   Z28-Z31: A broadcast temps (4 per sub-batch, 3 sub-batches per K)
//   p0: ptrue .h, p1: ptrue .s
//
// Args: x0=A(fp16), x1=B(fp16), x2=C(fp32), x3=K(even), x4=unused, x5=ldc_bytes

    .arch armv8.2-a+sve
    .text
    .align 4
    .global micro_kernel_fp16_12x2_dbuf
    .type micro_kernel_fp16_12x2_dbuf, %function

micro_kernel_fp16_12x2_dbuf:
    stp d8, d9, [sp, #-64]!
    stp d10, d11, [sp, #16]
    stp d12, d13, [sp, #32]
    stp d14, d15, [sp, #48]

    ptrue p0.h
    ptrue p1.s

    // Zero 24 accumulators
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

    // Load B0 for K=0
    ld1h {z24.h}, p0/z, [x1]
    ld1h {z25.h}, p0/z, [x1, #1, mul vl]
    // Preload B1 for K=1
    ld1h {z26.h}, p0/z, [x1, #2, mul vl]
    ld1h {z27.h}, p0/z, [x1, #3, mul vl]

    lsr x6, x3, #1          // x6 = K/2 loop iterations

.Ldbuf_loop:
    // ═══ K=0: uses B0 (z24,z25). B1 (z26,z27) already preloaded ═══

    // Sub-batch 0: rows 0-3
    ld1rh {z28.h}, p0/z, [x0, #0]
    ld1rh {z29.h}, p0/z, [x0, #2]
    ld1rh {z30.h}, p0/z, [x0, #4]
    ld1rh {z31.h}, p0/z, [x0, #6]
    fmla z0.h, p0/m, z28.h, z24.h
    fmla z1.h, p0/m, z28.h, z25.h
    fmla z2.h, p0/m, z29.h, z24.h
    fmla z3.h, p0/m, z29.h, z25.h
    fmla z4.h, p0/m, z30.h, z24.h
    fmla z5.h, p0/m, z30.h, z25.h
    fmla z6.h, p0/m, z31.h, z24.h
    fmla z7.h, p0/m, z31.h, z25.h

    // Sub-batch 1: rows 4-7
    ld1rh {z28.h}, p0/z, [x0, #8]
    ld1rh {z29.h}, p0/z, [x0, #10]
    ld1rh {z30.h}, p0/z, [x0, #12]
    ld1rh {z31.h}, p0/z, [x0, #14]
    fmla z8.h, p0/m, z28.h, z24.h
    fmla z9.h, p0/m, z28.h, z25.h
    fmla z10.h, p0/m, z29.h, z24.h
    fmla z11.h, p0/m, z29.h, z25.h
    fmla z12.h, p0/m, z30.h, z24.h
    fmla z13.h, p0/m, z30.h, z25.h
    fmla z14.h, p0/m, z31.h, z24.h
    fmla z15.h, p0/m, z31.h, z25.h

    // Sub-batch 2: rows 8-11
    ld1rh {z28.h}, p0/z, [x0, #16]
    ld1rh {z29.h}, p0/z, [x0, #18]
    ld1rh {z30.h}, p0/z, [x0, #20]
    ld1rh {z31.h}, p0/z, [x0, #22]
    fmla z16.h, p0/m, z28.h, z24.h
    fmla z17.h, p0/m, z28.h, z25.h
    fmla z18.h, p0/m, z29.h, z24.h
    fmla z19.h, p0/m, z29.h, z25.h
    fmla z20.h, p0/m, z30.h, z24.h
    fmla z21.h, p0/m, z30.h, z25.h
    fmla z22.h, p0/m, z31.h, z24.h
    fmla z23.h, p0/m, z31.h, z25.h

    // ═══ K=1: uses B1 (z26,z27). Preload B0 for next iter ═══

    // Preload B0 (z24,z25) for next iteration's K=0
    // Distance to first use: all of K=1 (24 FMAs=12cy) + loop tail → >>11cy
    ld1h {z24.h}, p0/z, [x1, #4, mul vl]
    ld1h {z25.h}, p0/z, [x1, #5, mul vl]

    // Sub-batch 0: rows 0-3
    ld1rh {z28.h}, p0/z, [x0, #24]
    ld1rh {z29.h}, p0/z, [x0, #26]
    ld1rh {z30.h}, p0/z, [x0, #28]
    ld1rh {z31.h}, p0/z, [x0, #30]
    fmla z0.h, p0/m, z28.h, z26.h
    fmla z1.h, p0/m, z28.h, z27.h
    fmla z2.h, p0/m, z29.h, z26.h
    fmla z3.h, p0/m, z29.h, z27.h
    fmla z4.h, p0/m, z30.h, z26.h
    fmla z5.h, p0/m, z30.h, z27.h
    fmla z6.h, p0/m, z31.h, z26.h
    fmla z7.h, p0/m, z31.h, z27.h

    // Sub-batch 1: rows 4-7
    ld1rh {z28.h}, p0/z, [x0, #32]
    ld1rh {z29.h}, p0/z, [x0, #34]
    ld1rh {z30.h}, p0/z, [x0, #36]
    ld1rh {z31.h}, p0/z, [x0, #38]
    fmla z8.h, p0/m, z28.h, z26.h
    fmla z9.h, p0/m, z28.h, z27.h
    fmla z10.h, p0/m, z29.h, z26.h
    fmla z11.h, p0/m, z29.h, z27.h
    fmla z12.h, p0/m, z30.h, z26.h
    fmla z13.h, p0/m, z30.h, z27.h
    fmla z14.h, p0/m, z31.h, z26.h
    fmla z15.h, p0/m, z31.h, z27.h

    // Sub-batch 2: rows 8-11
    ld1rh {z28.h}, p0/z, [x0, #40]
    ld1rh {z29.h}, p0/z, [x0, #42]
    ld1rh {z30.h}, p0/z, [x0, #44]
    ld1rh {z31.h}, p0/z, [x0, #46]
    fmla z16.h, p0/m, z28.h, z26.h
    fmla z17.h, p0/m, z28.h, z27.h
    fmla z18.h, p0/m, z29.h, z26.h
    fmla z19.h, p0/m, z29.h, z27.h
    fmla z20.h, p0/m, z30.h, z26.h
    fmla z21.h, p0/m, z30.h, z27.h
    fmla z22.h, p0/m, z31.h, z26.h
    fmla z23.h, p0/m, z31.h, z27.h

    // Loop tail: advance pointers, preload B1 for next iter
    add x0, x0, #48       // A: 2K × 12 × 2 bytes
    add x1, x1, #256      // B: 2K × 2 × 64 bytes

    subs x6, x6, #1
    beq .Ldbuf_store

    // Preload B1 (z26,z27) for next iteration's K=1
    // Distance to first use: all of K=0 (24 FMAs=12cy) → >>11cy
    ld1h {z26.h}, p0/z, [x1, #2, mul vl]
    ld1h {z27.h}, p0/z, [x1, #3, mul vl]
    b .Ldbuf_loop

.Ldbuf_store:
    // ── Convert fp16 accumulators → fp32 and store ──
    // Row 0: z0 (cols 0-31), z1 (cols 32-63)
    uunpklo z24.s, z0.h
    fcvt z24.s, p1/m, z24.h
    st1w {z24.s}, p1, [x2]
    uunpkhi z24.s, z0.h
    fcvt z24.s, p1/m, z24.h
    st1w {z24.s}, p1, [x2, #1, mul vl]
    uunpklo z24.s, z1.h
    fcvt z24.s, p1/m, z24.h
    st1w {z24.s}, p1, [x2, #2, mul vl]
    uunpkhi z24.s, z1.h
    fcvt z24.s, p1/m, z24.h
    st1w {z24.s}, p1, [x2, #3, mul vl]
    add x2, x2, x5

    // Row 1
    uunpklo z24.s, z2.h
    fcvt z24.s, p1/m, z24.h
    st1w {z24.s}, p1, [x2]
    uunpkhi z24.s, z2.h
    fcvt z24.s, p1/m, z24.h
    st1w {z24.s}, p1, [x2, #1, mul vl]
    uunpklo z24.s, z3.h
    fcvt z24.s, p1/m, z24.h
    st1w {z24.s}, p1, [x2, #2, mul vl]
    uunpkhi z24.s, z3.h
    fcvt z24.s, p1/m, z24.h
    st1w {z24.s}, p1, [x2, #3, mul vl]
    add x2, x2, x5

    // Row 2
    uunpklo z24.s, z4.h
    fcvt z24.s, p1/m, z24.h
    st1w {z24.s}, p1, [x2]
    uunpkhi z24.s, z4.h
    fcvt z24.s, p1/m, z24.h
    st1w {z24.s}, p1, [x2, #1, mul vl]
    uunpklo z24.s, z5.h
    fcvt z24.s, p1/m, z24.h
    st1w {z24.s}, p1, [x2, #2, mul vl]
    uunpkhi z24.s, z5.h
    fcvt z24.s, p1/m, z24.h
    st1w {z24.s}, p1, [x2, #3, mul vl]
    add x2, x2, x5

    // Row 3
    uunpklo z24.s, z6.h
    fcvt z24.s, p1/m, z24.h
    st1w {z24.s}, p1, [x2]
    uunpkhi z24.s, z6.h
    fcvt z24.s, p1/m, z24.h
    st1w {z24.s}, p1, [x2, #1, mul vl]
    uunpklo z24.s, z7.h
    fcvt z24.s, p1/m, z24.h
    st1w {z24.s}, p1, [x2, #2, mul vl]
    uunpkhi z24.s, z7.h
    fcvt z24.s, p1/m, z24.h
    st1w {z24.s}, p1, [x2, #3, mul vl]
    add x2, x2, x5

    // Row 4
    uunpklo z24.s, z8.h
    fcvt z24.s, p1/m, z24.h
    st1w {z24.s}, p1, [x2]
    uunpkhi z24.s, z8.h
    fcvt z24.s, p1/m, z24.h
    st1w {z24.s}, p1, [x2, #1, mul vl]
    uunpklo z24.s, z9.h
    fcvt z24.s, p1/m, z24.h
    st1w {z24.s}, p1, [x2, #2, mul vl]
    uunpkhi z24.s, z9.h
    fcvt z24.s, p1/m, z24.h
    st1w {z24.s}, p1, [x2, #3, mul vl]
    add x2, x2, x5

    // Row 5
    uunpklo z24.s, z10.h
    fcvt z24.s, p1/m, z24.h
    st1w {z24.s}, p1, [x2]
    uunpkhi z24.s, z10.h
    fcvt z24.s, p1/m, z24.h
    st1w {z24.s}, p1, [x2, #1, mul vl]
    uunpklo z24.s, z11.h
    fcvt z24.s, p1/m, z24.h
    st1w {z24.s}, p1, [x2, #2, mul vl]
    uunpkhi z24.s, z11.h
    fcvt z24.s, p1/m, z24.h
    st1w {z24.s}, p1, [x2, #3, mul vl]
    add x2, x2, x5

    // Row 6
    uunpklo z24.s, z12.h
    fcvt z24.s, p1/m, z24.h
    st1w {z24.s}, p1, [x2]
    uunpkhi z24.s, z12.h
    fcvt z24.s, p1/m, z24.h
    st1w {z24.s}, p1, [x2, #1, mul vl]
    uunpklo z24.s, z13.h
    fcvt z24.s, p1/m, z24.h
    st1w {z24.s}, p1, [x2, #2, mul vl]
    uunpkhi z24.s, z13.h
    fcvt z24.s, p1/m, z24.h
    st1w {z24.s}, p1, [x2, #3, mul vl]
    add x2, x2, x5

    // Row 7
    uunpklo z24.s, z14.h
    fcvt z24.s, p1/m, z24.h
    st1w {z24.s}, p1, [x2]
    uunpkhi z24.s, z14.h
    fcvt z24.s, p1/m, z24.h
    st1w {z24.s}, p1, [x2, #1, mul vl]
    uunpklo z24.s, z15.h
    fcvt z24.s, p1/m, z24.h
    st1w {z24.s}, p1, [x2, #2, mul vl]
    uunpkhi z24.s, z15.h
    fcvt z24.s, p1/m, z24.h
    st1w {z24.s}, p1, [x2, #3, mul vl]
    add x2, x2, x5

    // Row 8
    uunpklo z24.s, z16.h
    fcvt z24.s, p1/m, z24.h
    st1w {z24.s}, p1, [x2]
    uunpkhi z24.s, z16.h
    fcvt z24.s, p1/m, z24.h
    st1w {z24.s}, p1, [x2, #1, mul vl]
    uunpklo z24.s, z17.h
    fcvt z24.s, p1/m, z24.h
    st1w {z24.s}, p1, [x2, #2, mul vl]
    uunpkhi z24.s, z17.h
    fcvt z24.s, p1/m, z24.h
    st1w {z24.s}, p1, [x2, #3, mul vl]
    add x2, x2, x5

    // Row 9
    uunpklo z24.s, z18.h
    fcvt z24.s, p1/m, z24.h
    st1w {z24.s}, p1, [x2]
    uunpkhi z24.s, z18.h
    fcvt z24.s, p1/m, z24.h
    st1w {z24.s}, p1, [x2, #1, mul vl]
    uunpklo z24.s, z19.h
    fcvt z24.s, p1/m, z24.h
    st1w {z24.s}, p1, [x2, #2, mul vl]
    uunpkhi z24.s, z19.h
    fcvt z24.s, p1/m, z24.h
    st1w {z24.s}, p1, [x2, #3, mul vl]
    add x2, x2, x5

    // Row 10
    uunpklo z24.s, z20.h
    fcvt z24.s, p1/m, z24.h
    st1w {z24.s}, p1, [x2]
    uunpkhi z24.s, z20.h
    fcvt z24.s, p1/m, z24.h
    st1w {z24.s}, p1, [x2, #1, mul vl]
    uunpklo z24.s, z21.h
    fcvt z24.s, p1/m, z24.h
    st1w {z24.s}, p1, [x2, #2, mul vl]
    uunpkhi z24.s, z21.h
    fcvt z24.s, p1/m, z24.h
    st1w {z24.s}, p1, [x2, #3, mul vl]
    add x2, x2, x5

    // Row 11
    uunpklo z24.s, z22.h
    fcvt z24.s, p1/m, z24.h
    st1w {z24.s}, p1, [x2]
    uunpkhi z24.s, z22.h
    fcvt z24.s, p1/m, z24.h
    st1w {z24.s}, p1, [x2, #1, mul vl]
    uunpklo z24.s, z23.h
    fcvt z24.s, p1/m, z24.h
    st1w {z24.s}, p1, [x2, #2, mul vl]
    uunpkhi z24.s, z23.h
    fcvt z24.s, p1/m, z24.h
    st1w {z24.s}, p1, [x2, #3, mul vl]

    ldp d14, d15, [sp, #48]
    ldp d12, d13, [sp, #32]
    ldp d10, d11, [sp, #16]
    ldp d8, d9, [sp], #64
    ret
    .size micro_kernel_fp16_12x2_dbuf, .-micro_kernel_fp16_12x2_dbuf
