// FP16 12×2 Software-Pipelined 4K-Unrolled ACCUMULATE variant
// Processes 4 K-steps per loop iteration to reduce loop overhead by 2×
// K must be a multiple of 4
//
// Computes: C[12][64](fp32) += A[K][12](fp16) × B[K][64](fp16)
// Epilogue: uunpklo/hi + fcvt + ld1w + fadd + st1w
//
// Args: x0=A(fp16), x1=B(fp16), x2=C(fp32), x3=K(mult of 4), x4=unused, x5=ldc_bytes

    .arch armv8.2-a+sve
    .text
    .align 4
    .global micro_kernel_fp16_12x2_swp4_accum
    .type micro_kernel_fp16_12x2_swp4_accum, %function

micro_kernel_fp16_12x2_swp4_accum:
    stp d8, d9, [sp, #-64]!
    stp d10, d11, [sp, #16]
    stp d12, d13, [sp, #32]
    stp d14, d15, [sp, #48]

    ptrue p0.h
    ptrue p1.s

    // Load initial B
    ld1h {z24.h}, p0/z, [x1]
    ld1h {z25.h}, p0/z, [x1, #1, mul vl]

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

    lsr x6, x3, #2          // x6 = K/4 loop iterations

.Lh12x2s4a_loop:
    // ═══════════════════════════════════
    // K iteration 0 (A offsets 0-22)
    // ═══════════════════════════════════
    ld1rh {z26.h}, p0/z, [x0, #0]
    ld1rh {z27.h}, p0/z, [x0, #2]
    ld1rh {z28.h}, p0/z, [x0, #4]
    ld1rh {z29.h}, p0/z, [x0, #6]
    ld1rh {z30.h}, p0/z, [x0, #8]
    ld1rh {z31.h}, p0/z, [x0, #10]

    fmla z0.h, p0/m, z26.h, z24.h
    fmla z1.h, p0/m, z26.h, z25.h
    ld1rh {z26.h}, p0/z, [x0, #12]
    fmla z2.h, p0/m, z27.h, z24.h
    fmla z3.h, p0/m, z27.h, z25.h
    ld1rh {z27.h}, p0/z, [x0, #14]
    fmla z4.h, p0/m, z28.h, z24.h
    fmla z5.h, p0/m, z28.h, z25.h
    ld1rh {z28.h}, p0/z, [x0, #16]
    fmla z6.h, p0/m, z29.h, z24.h
    fmla z7.h, p0/m, z29.h, z25.h
    ld1rh {z29.h}, p0/z, [x0, #18]
    fmla z8.h, p0/m, z30.h, z24.h
    fmla z9.h, p0/m, z30.h, z25.h
    ld1rh {z30.h}, p0/z, [x0, #20]
    fmla z10.h, p0/m, z31.h, z24.h
    fmla z11.h, p0/m, z31.h, z25.h
    ld1rh {z31.h}, p0/z, [x0, #22]

    fmla z12.h, p0/m, z26.h, z24.h
    fmla z13.h, p0/m, z26.h, z25.h
    fmla z14.h, p0/m, z27.h, z24.h
    fmla z15.h, p0/m, z27.h, z25.h
    fmla z16.h, p0/m, z28.h, z24.h
    fmla z17.h, p0/m, z28.h, z25.h
    fmla z18.h, p0/m, z29.h, z24.h
    fmla z19.h, p0/m, z29.h, z25.h
    fmla z20.h, p0/m, z30.h, z24.h
    fmla z21.h, p0/m, z30.h, z25.h
    fmla z22.h, p0/m, z31.h, z24.h
    fmla z23.h, p0/m, z31.h, z25.h

    // Load B(k+1)
    ld1h {z24.h}, p0/z, [x1, #2, mul vl]
    ld1h {z25.h}, p0/z, [x1, #3, mul vl]

    // ═══════════════════════════════════
    // K iteration 1 (A offsets 24-46)
    // ═══════════════════════════════════
    ld1rh {z26.h}, p0/z, [x0, #24]
    ld1rh {z27.h}, p0/z, [x0, #26]
    ld1rh {z28.h}, p0/z, [x0, #28]
    ld1rh {z29.h}, p0/z, [x0, #30]
    ld1rh {z30.h}, p0/z, [x0, #32]
    ld1rh {z31.h}, p0/z, [x0, #34]

    fmla z0.h, p0/m, z26.h, z24.h
    fmla z1.h, p0/m, z26.h, z25.h
    ld1rh {z26.h}, p0/z, [x0, #36]
    fmla z2.h, p0/m, z27.h, z24.h
    fmla z3.h, p0/m, z27.h, z25.h
    ld1rh {z27.h}, p0/z, [x0, #38]
    fmla z4.h, p0/m, z28.h, z24.h
    fmla z5.h, p0/m, z28.h, z25.h
    ld1rh {z28.h}, p0/z, [x0, #40]
    fmla z6.h, p0/m, z29.h, z24.h
    fmla z7.h, p0/m, z29.h, z25.h
    ld1rh {z29.h}, p0/z, [x0, #42]
    fmla z8.h, p0/m, z30.h, z24.h
    fmla z9.h, p0/m, z30.h, z25.h
    ld1rh {z30.h}, p0/z, [x0, #44]
    fmla z10.h, p0/m, z31.h, z24.h
    fmla z11.h, p0/m, z31.h, z25.h
    ld1rh {z31.h}, p0/z, [x0, #46]

    fmla z12.h, p0/m, z26.h, z24.h
    fmla z13.h, p0/m, z26.h, z25.h
    fmla z14.h, p0/m, z27.h, z24.h
    fmla z15.h, p0/m, z27.h, z25.h
    fmla z16.h, p0/m, z28.h, z24.h
    fmla z17.h, p0/m, z28.h, z25.h
    fmla z18.h, p0/m, z29.h, z24.h
    fmla z19.h, p0/m, z29.h, z25.h
    fmla z20.h, p0/m, z30.h, z24.h
    fmla z21.h, p0/m, z30.h, z25.h
    fmla z22.h, p0/m, z31.h, z24.h
    fmla z23.h, p0/m, z31.h, z25.h

    // Load B(k+2)
    ld1h {z24.h}, p0/z, [x1, #4, mul vl]
    ld1h {z25.h}, p0/z, [x1, #5, mul vl]

    // ═══════════════════════════════════
    // K iteration 2 (A offsets 48-70)
    // ═══════════════════════════════════
    ld1rh {z26.h}, p0/z, [x0, #48]
    ld1rh {z27.h}, p0/z, [x0, #50]
    ld1rh {z28.h}, p0/z, [x0, #52]
    ld1rh {z29.h}, p0/z, [x0, #54]
    ld1rh {z30.h}, p0/z, [x0, #56]
    ld1rh {z31.h}, p0/z, [x0, #58]

    fmla z0.h, p0/m, z26.h, z24.h
    fmla z1.h, p0/m, z26.h, z25.h
    ld1rh {z26.h}, p0/z, [x0, #60]
    fmla z2.h, p0/m, z27.h, z24.h
    fmla z3.h, p0/m, z27.h, z25.h
    ld1rh {z27.h}, p0/z, [x0, #62]
    fmla z4.h, p0/m, z28.h, z24.h
    fmla z5.h, p0/m, z28.h, z25.h
    ld1rh {z28.h}, p0/z, [x0, #64]
    fmla z6.h, p0/m, z29.h, z24.h
    fmla z7.h, p0/m, z29.h, z25.h
    ld1rh {z29.h}, p0/z, [x0, #66]
    fmla z8.h, p0/m, z30.h, z24.h
    fmla z9.h, p0/m, z30.h, z25.h
    ld1rh {z30.h}, p0/z, [x0, #68]
    fmla z10.h, p0/m, z31.h, z24.h
    fmla z11.h, p0/m, z31.h, z25.h
    ld1rh {z31.h}, p0/z, [x0, #70]

    fmla z12.h, p0/m, z26.h, z24.h
    fmla z13.h, p0/m, z26.h, z25.h
    fmla z14.h, p0/m, z27.h, z24.h
    fmla z15.h, p0/m, z27.h, z25.h
    fmla z16.h, p0/m, z28.h, z24.h
    fmla z17.h, p0/m, z28.h, z25.h
    fmla z18.h, p0/m, z29.h, z24.h
    fmla z19.h, p0/m, z29.h, z25.h
    fmla z20.h, p0/m, z30.h, z24.h
    fmla z21.h, p0/m, z30.h, z25.h
    fmla z22.h, p0/m, z31.h, z24.h
    fmla z23.h, p0/m, z31.h, z25.h

    // Load B(k+3)
    ld1h {z24.h}, p0/z, [x1, #6, mul vl]
    ld1h {z25.h}, p0/z, [x1, #7, mul vl]

    // ═══════════════════════════════════
    // K iteration 3 (A offsets 72-94)
    // ═══════════════════════════════════
    ld1rh {z26.h}, p0/z, [x0, #72]
    ld1rh {z27.h}, p0/z, [x0, #74]
    ld1rh {z28.h}, p0/z, [x0, #76]
    ld1rh {z29.h}, p0/z, [x0, #78]
    ld1rh {z30.h}, p0/z, [x0, #80]
    ld1rh {z31.h}, p0/z, [x0, #82]

    fmla z0.h, p0/m, z26.h, z24.h
    fmla z1.h, p0/m, z26.h, z25.h
    ld1rh {z26.h}, p0/z, [x0, #84]
    fmla z2.h, p0/m, z27.h, z24.h
    fmla z3.h, p0/m, z27.h, z25.h
    ld1rh {z27.h}, p0/z, [x0, #86]
    fmla z4.h, p0/m, z28.h, z24.h
    fmla z5.h, p0/m, z28.h, z25.h
    ld1rh {z28.h}, p0/z, [x0, #88]
    fmla z6.h, p0/m, z29.h, z24.h
    fmla z7.h, p0/m, z29.h, z25.h
    ld1rh {z29.h}, p0/z, [x0, #90]
    fmla z8.h, p0/m, z30.h, z24.h
    fmla z9.h, p0/m, z30.h, z25.h
    ld1rh {z30.h}, p0/z, [x0, #92]
    fmla z10.h, p0/m, z31.h, z24.h
    fmla z11.h, p0/m, z31.h, z25.h
    ld1rh {z31.h}, p0/z, [x0, #94]

    fmla z12.h, p0/m, z26.h, z24.h
    fmla z13.h, p0/m, z26.h, z25.h
    fmla z14.h, p0/m, z27.h, z24.h
    fmla z15.h, p0/m, z27.h, z25.h
    fmla z16.h, p0/m, z28.h, z24.h
    fmla z17.h, p0/m, z28.h, z25.h
    fmla z18.h, p0/m, z29.h, z24.h
    fmla z19.h, p0/m, z29.h, z25.h
    fmla z20.h, p0/m, z30.h, z24.h
    fmla z21.h, p0/m, z30.h, z25.h
    fmla z22.h, p0/m, z31.h, z24.h
    fmla z23.h, p0/m, z31.h, z25.h

    // Advance pointers
    add x0, x0, #96      // A: 4 × 12 × 2 bytes
    add x1, x1, #512     // B: 4 × 2 × 64 bytes

    subs x6, x6, #1
    beq .Lh12x2s4a_store

    ld1h {z24.h}, p0/z, [x1]
    ld1h {z25.h}, p0/z, [x1, #1, mul vl]
    b .Lh12x2s4a_loop

.Lh12x2s4a_store:
    // ── Convert fp16 → fp32, add to existing C, store ──

    // Row 0: z0, z1
    uunpklo z24.s, z0.h
    fcvt z24.s, p1/m, z24.h
    ld1w {z25.s}, p1/z, [x2]
    fadd z24.s, z24.s, z25.s
    st1w {z24.s}, p1, [x2]
    uunpkhi z24.s, z0.h
    fcvt z24.s, p1/m, z24.h
    ld1w {z25.s}, p1/z, [x2, #1, mul vl]
    fadd z24.s, z24.s, z25.s
    st1w {z24.s}, p1, [x2, #1, mul vl]
    uunpklo z24.s, z1.h
    fcvt z24.s, p1/m, z24.h
    ld1w {z25.s}, p1/z, [x2, #2, mul vl]
    fadd z24.s, z24.s, z25.s
    st1w {z24.s}, p1, [x2, #2, mul vl]
    uunpkhi z24.s, z1.h
    fcvt z24.s, p1/m, z24.h
    ld1w {z25.s}, p1/z, [x2, #3, mul vl]
    fadd z24.s, z24.s, z25.s
    st1w {z24.s}, p1, [x2, #3, mul vl]
    add x2, x2, x5

    // Row 1: z2, z3
    uunpklo z24.s, z2.h
    fcvt z24.s, p1/m, z24.h
    ld1w {z25.s}, p1/z, [x2]
    fadd z24.s, z24.s, z25.s
    st1w {z24.s}, p1, [x2]
    uunpkhi z24.s, z2.h
    fcvt z24.s, p1/m, z24.h
    ld1w {z25.s}, p1/z, [x2, #1, mul vl]
    fadd z24.s, z24.s, z25.s
    st1w {z24.s}, p1, [x2, #1, mul vl]
    uunpklo z24.s, z3.h
    fcvt z24.s, p1/m, z24.h
    ld1w {z25.s}, p1/z, [x2, #2, mul vl]
    fadd z24.s, z24.s, z25.s
    st1w {z24.s}, p1, [x2, #2, mul vl]
    uunpkhi z24.s, z3.h
    fcvt z24.s, p1/m, z24.h
    ld1w {z25.s}, p1/z, [x2, #3, mul vl]
    fadd z24.s, z24.s, z25.s
    st1w {z24.s}, p1, [x2, #3, mul vl]
    add x2, x2, x5

    // Row 2: z4, z5
    uunpklo z24.s, z4.h
    fcvt z24.s, p1/m, z24.h
    ld1w {z25.s}, p1/z, [x2]
    fadd z24.s, z24.s, z25.s
    st1w {z24.s}, p1, [x2]
    uunpkhi z24.s, z4.h
    fcvt z24.s, p1/m, z24.h
    ld1w {z25.s}, p1/z, [x2, #1, mul vl]
    fadd z24.s, z24.s, z25.s
    st1w {z24.s}, p1, [x2, #1, mul vl]
    uunpklo z24.s, z5.h
    fcvt z24.s, p1/m, z24.h
    ld1w {z25.s}, p1/z, [x2, #2, mul vl]
    fadd z24.s, z24.s, z25.s
    st1w {z24.s}, p1, [x2, #2, mul vl]
    uunpkhi z24.s, z5.h
    fcvt z24.s, p1/m, z24.h
    ld1w {z25.s}, p1/z, [x2, #3, mul vl]
    fadd z24.s, z24.s, z25.s
    st1w {z24.s}, p1, [x2, #3, mul vl]
    add x2, x2, x5

    // Row 3: z6, z7
    uunpklo z24.s, z6.h
    fcvt z24.s, p1/m, z24.h
    ld1w {z25.s}, p1/z, [x2]
    fadd z24.s, z24.s, z25.s
    st1w {z24.s}, p1, [x2]
    uunpkhi z24.s, z6.h
    fcvt z24.s, p1/m, z24.h
    ld1w {z25.s}, p1/z, [x2, #1, mul vl]
    fadd z24.s, z24.s, z25.s
    st1w {z24.s}, p1, [x2, #1, mul vl]
    uunpklo z24.s, z7.h
    fcvt z24.s, p1/m, z24.h
    ld1w {z25.s}, p1/z, [x2, #2, mul vl]
    fadd z24.s, z24.s, z25.s
    st1w {z24.s}, p1, [x2, #2, mul vl]
    uunpkhi z24.s, z7.h
    fcvt z24.s, p1/m, z24.h
    ld1w {z25.s}, p1/z, [x2, #3, mul vl]
    fadd z24.s, z24.s, z25.s
    st1w {z24.s}, p1, [x2, #3, mul vl]
    add x2, x2, x5

    // Row 4: z8, z9
    uunpklo z24.s, z8.h
    fcvt z24.s, p1/m, z24.h
    ld1w {z25.s}, p1/z, [x2]
    fadd z24.s, z24.s, z25.s
    st1w {z24.s}, p1, [x2]
    uunpkhi z24.s, z8.h
    fcvt z24.s, p1/m, z24.h
    ld1w {z25.s}, p1/z, [x2, #1, mul vl]
    fadd z24.s, z24.s, z25.s
    st1w {z24.s}, p1, [x2, #1, mul vl]
    uunpklo z24.s, z9.h
    fcvt z24.s, p1/m, z24.h
    ld1w {z25.s}, p1/z, [x2, #2, mul vl]
    fadd z24.s, z24.s, z25.s
    st1w {z24.s}, p1, [x2, #2, mul vl]
    uunpkhi z24.s, z9.h
    fcvt z24.s, p1/m, z24.h
    ld1w {z25.s}, p1/z, [x2, #3, mul vl]
    fadd z24.s, z24.s, z25.s
    st1w {z24.s}, p1, [x2, #3, mul vl]
    add x2, x2, x5

    // Row 5: z10, z11
    uunpklo z24.s, z10.h
    fcvt z24.s, p1/m, z24.h
    ld1w {z25.s}, p1/z, [x2]
    fadd z24.s, z24.s, z25.s
    st1w {z24.s}, p1, [x2]
    uunpkhi z24.s, z10.h
    fcvt z24.s, p1/m, z24.h
    ld1w {z25.s}, p1/z, [x2, #1, mul vl]
    fadd z24.s, z24.s, z25.s
    st1w {z24.s}, p1, [x2, #1, mul vl]
    uunpklo z24.s, z11.h
    fcvt z24.s, p1/m, z24.h
    ld1w {z25.s}, p1/z, [x2, #2, mul vl]
    fadd z24.s, z24.s, z25.s
    st1w {z24.s}, p1, [x2, #2, mul vl]
    uunpkhi z24.s, z11.h
    fcvt z24.s, p1/m, z24.h
    ld1w {z25.s}, p1/z, [x2, #3, mul vl]
    fadd z24.s, z24.s, z25.s
    st1w {z24.s}, p1, [x2, #3, mul vl]
    add x2, x2, x5

    // Row 6: z12, z13
    uunpklo z24.s, z12.h
    fcvt z24.s, p1/m, z24.h
    ld1w {z25.s}, p1/z, [x2]
    fadd z24.s, z24.s, z25.s
    st1w {z24.s}, p1, [x2]
    uunpkhi z24.s, z12.h
    fcvt z24.s, p1/m, z24.h
    ld1w {z25.s}, p1/z, [x2, #1, mul vl]
    fadd z24.s, z24.s, z25.s
    st1w {z24.s}, p1, [x2, #1, mul vl]
    uunpklo z24.s, z13.h
    fcvt z24.s, p1/m, z24.h
    ld1w {z25.s}, p1/z, [x2, #2, mul vl]
    fadd z24.s, z24.s, z25.s
    st1w {z24.s}, p1, [x2, #2, mul vl]
    uunpkhi z24.s, z13.h
    fcvt z24.s, p1/m, z24.h
    ld1w {z25.s}, p1/z, [x2, #3, mul vl]
    fadd z24.s, z24.s, z25.s
    st1w {z24.s}, p1, [x2, #3, mul vl]
    add x2, x2, x5

    // Row 7: z14, z15
    uunpklo z24.s, z14.h
    fcvt z24.s, p1/m, z24.h
    ld1w {z25.s}, p1/z, [x2]
    fadd z24.s, z24.s, z25.s
    st1w {z24.s}, p1, [x2]
    uunpkhi z24.s, z14.h
    fcvt z24.s, p1/m, z24.h
    ld1w {z25.s}, p1/z, [x2, #1, mul vl]
    fadd z24.s, z24.s, z25.s
    st1w {z24.s}, p1, [x2, #1, mul vl]
    uunpklo z24.s, z15.h
    fcvt z24.s, p1/m, z24.h
    ld1w {z25.s}, p1/z, [x2, #2, mul vl]
    fadd z24.s, z24.s, z25.s
    st1w {z24.s}, p1, [x2, #2, mul vl]
    uunpkhi z24.s, z15.h
    fcvt z24.s, p1/m, z24.h
    ld1w {z25.s}, p1/z, [x2, #3, mul vl]
    fadd z24.s, z24.s, z25.s
    st1w {z24.s}, p1, [x2, #3, mul vl]
    add x2, x2, x5

    // Row 8: z16, z17
    uunpklo z24.s, z16.h
    fcvt z24.s, p1/m, z24.h
    ld1w {z25.s}, p1/z, [x2]
    fadd z24.s, z24.s, z25.s
    st1w {z24.s}, p1, [x2]
    uunpkhi z24.s, z16.h
    fcvt z24.s, p1/m, z24.h
    ld1w {z25.s}, p1/z, [x2, #1, mul vl]
    fadd z24.s, z24.s, z25.s
    st1w {z24.s}, p1, [x2, #1, mul vl]
    uunpklo z24.s, z17.h
    fcvt z24.s, p1/m, z24.h
    ld1w {z25.s}, p1/z, [x2, #2, mul vl]
    fadd z24.s, z24.s, z25.s
    st1w {z24.s}, p1, [x2, #2, mul vl]
    uunpkhi z24.s, z17.h
    fcvt z24.s, p1/m, z24.h
    ld1w {z25.s}, p1/z, [x2, #3, mul vl]
    fadd z24.s, z24.s, z25.s
    st1w {z24.s}, p1, [x2, #3, mul vl]
    add x2, x2, x5

    // Row 9: z18, z19
    uunpklo z24.s, z18.h
    fcvt z24.s, p1/m, z24.h
    ld1w {z25.s}, p1/z, [x2]
    fadd z24.s, z24.s, z25.s
    st1w {z24.s}, p1, [x2]
    uunpkhi z24.s, z18.h
    fcvt z24.s, p1/m, z24.h
    ld1w {z25.s}, p1/z, [x2, #1, mul vl]
    fadd z24.s, z24.s, z25.s
    st1w {z24.s}, p1, [x2, #1, mul vl]
    uunpklo z24.s, z19.h
    fcvt z24.s, p1/m, z24.h
    ld1w {z25.s}, p1/z, [x2, #2, mul vl]
    fadd z24.s, z24.s, z25.s
    st1w {z24.s}, p1, [x2, #2, mul vl]
    uunpkhi z24.s, z19.h
    fcvt z24.s, p1/m, z24.h
    ld1w {z25.s}, p1/z, [x2, #3, mul vl]
    fadd z24.s, z24.s, z25.s
    st1w {z24.s}, p1, [x2, #3, mul vl]
    add x2, x2, x5

    // Row 10: z20, z21
    uunpklo z24.s, z20.h
    fcvt z24.s, p1/m, z24.h
    ld1w {z25.s}, p1/z, [x2]
    fadd z24.s, z24.s, z25.s
    st1w {z24.s}, p1, [x2]
    uunpkhi z24.s, z20.h
    fcvt z24.s, p1/m, z24.h
    ld1w {z25.s}, p1/z, [x2, #1, mul vl]
    fadd z24.s, z24.s, z25.s
    st1w {z24.s}, p1, [x2, #1, mul vl]
    uunpklo z24.s, z21.h
    fcvt z24.s, p1/m, z24.h
    ld1w {z25.s}, p1/z, [x2, #2, mul vl]
    fadd z24.s, z24.s, z25.s
    st1w {z24.s}, p1, [x2, #2, mul vl]
    uunpkhi z24.s, z21.h
    fcvt z24.s, p1/m, z24.h
    ld1w {z25.s}, p1/z, [x2, #3, mul vl]
    fadd z24.s, z24.s, z25.s
    st1w {z24.s}, p1, [x2, #3, mul vl]
    add x2, x2, x5

    // Row 11: z22, z23
    uunpklo z24.s, z22.h
    fcvt z24.s, p1/m, z24.h
    ld1w {z25.s}, p1/z, [x2]
    fadd z24.s, z24.s, z25.s
    st1w {z24.s}, p1, [x2]
    uunpkhi z24.s, z22.h
    fcvt z24.s, p1/m, z24.h
    ld1w {z25.s}, p1/z, [x2, #1, mul vl]
    fadd z24.s, z24.s, z25.s
    st1w {z24.s}, p1, [x2, #1, mul vl]
    uunpklo z24.s, z23.h
    fcvt z24.s, p1/m, z24.h
    ld1w {z25.s}, p1/z, [x2, #2, mul vl]
    fadd z24.s, z24.s, z25.s
    st1w {z24.s}, p1, [x2, #2, mul vl]
    uunpkhi z24.s, z23.h
    fcvt z24.s, p1/m, z24.h
    ld1w {z25.s}, p1/z, [x2, #3, mul vl]
    fadd z24.s, z24.s, z25.s
    st1w {z24.s}, p1, [x2, #3, mul vl]

    ldp d14, d15, [sp, #48]
    ldp d12, d13, [sp, #32]
    ldp d10, d11, [sp, #16]
    ldp d8, d9, [sp], #64
    ret
    .size micro_kernel_fp16_12x2_swp4_accum, .-micro_kernel_fp16_12x2_swp4_accum
