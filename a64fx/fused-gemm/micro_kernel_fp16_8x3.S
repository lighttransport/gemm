// FP16 8×3 INIT: MR=8, NR=96 (3 SVE vecs × 32 fp16), 24 accumulators
// 2K-unrolled, SWP: rows 5-7 A reloaded during rows 0-2 FMAs
//
// Register map:
//   z0-z23:  24 accumulators [row i, col j] = z[i*3 + j]
//   z24-z26: B vectors (3 columns)
//   z27-z31: A temp registers (5 of 8 rows pre-loaded; rows 5-7 SWP-reloaded)
//
// Key advantage over 12×2: all 24 FMAs share same B vectors (no batch transition)
// → 24 FMAs = 12 cycles at 2/cy, vs 11cy L1 latency → can fully hide load latency
//
// B tile: 96 × Kc × 2 bytes (Kc=256 → 48KB)
// A per K: 8 × 2 = 16 bytes
//
// Args: x0=A(fp16), x1=B(fp16), x2=C(fp32), x3=K(even), x4=unused, x5=ldc_bytes

    .arch armv8.2-a+sve
    .text
    .align 4
    .global micro_kernel_fp16_8x3
    .type micro_kernel_fp16_8x3, %function

micro_kernel_fp16_8x3:
    stp d8, d9, [sp, #-64]!
    stp d10, d11, [sp, #16]
    stp d12, d13, [sp, #32]
    stp d14, d15, [sp, #48]

    ptrue p0.h
    ptrue p1.s

    // Load initial B (3 vectors for K=0)
    ld1h {z24.h}, p0/z, [x1]
    ld1h {z25.h}, p0/z, [x1, #1, mul vl]
    ld1h {z26.h}, p0/z, [x1, #2, mul vl]

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

    lsr x6, x3, #1          // K/2 loop iterations

.Lh8x3_loop:
    // ═══ K=0: B at [x1,#0/#1/#2 mul vl], A at [x0,#0..#14] ═══
    // Pre-load A rows 0-4
    ld1rh {z27.h}, p0/z, [x0, #0]
    ld1rh {z28.h}, p0/z, [x0, #2]
    ld1rh {z29.h}, p0/z, [x0, #4]
    ld1rh {z30.h}, p0/z, [x0, #6]
    ld1rh {z31.h}, p0/z, [x0, #8]

    // Row 0 (z27) + SWP reload row 5
    fmla z0.h, p0/m, z27.h, z24.h
    fmla z1.h, p0/m, z27.h, z25.h
    fmla z2.h, p0/m, z27.h, z26.h
    ld1rh {z27.h}, p0/z, [x0, #10]      // row 5

    // Row 1 (z28) + SWP reload row 6
    fmla z3.h, p0/m, z28.h, z24.h
    fmla z4.h, p0/m, z28.h, z25.h
    fmla z5.h, p0/m, z28.h, z26.h
    ld1rh {z28.h}, p0/z, [x0, #12]      // row 6

    // Row 2 (z29) + SWP reload row 7
    fmla z6.h, p0/m, z29.h, z24.h
    fmla z7.h, p0/m, z29.h, z25.h
    fmla z8.h, p0/m, z29.h, z26.h
    ld1rh {z29.h}, p0/z, [x0, #14]      // row 7

    // Rows 3-4 (z30, z31) — no reload needed
    fmla z9.h, p0/m, z30.h, z24.h
    fmla z10.h, p0/m, z30.h, z25.h
    fmla z11.h, p0/m, z30.h, z26.h

    fmla z12.h, p0/m, z31.h, z24.h
    fmla z13.h, p0/m, z31.h, z25.h
    fmla z14.h, p0/m, z31.h, z26.h

    // Rows 5-7 (z27, z28, z29 — SWP-reloaded)
    fmla z15.h, p0/m, z27.h, z24.h
    fmla z16.h, p0/m, z27.h, z25.h
    fmla z17.h, p0/m, z27.h, z26.h

    fmla z18.h, p0/m, z28.h, z24.h
    fmla z19.h, p0/m, z28.h, z25.h
    fmla z20.h, p0/m, z28.h, z26.h

    fmla z21.h, p0/m, z29.h, z24.h
    fmla z22.h, p0/m, z29.h, z25.h
    fmla z23.h, p0/m, z29.h, z26.h

    // ═══ K=1: B at [x1,#3/#4/#5 mul vl], A at [x0,#16..#30] ═══
    ld1h {z24.h}, p0/z, [x1, #3, mul vl]
    ld1h {z25.h}, p0/z, [x1, #4, mul vl]
    ld1h {z26.h}, p0/z, [x1, #5, mul vl]

    // Pre-load A rows 0-4
    ld1rh {z27.h}, p0/z, [x0, #16]
    ld1rh {z28.h}, p0/z, [x0, #18]
    ld1rh {z29.h}, p0/z, [x0, #20]
    ld1rh {z30.h}, p0/z, [x0, #22]
    ld1rh {z31.h}, p0/z, [x0, #24]

    // Row 0 + SWP row 5
    fmla z0.h, p0/m, z27.h, z24.h
    fmla z1.h, p0/m, z27.h, z25.h
    fmla z2.h, p0/m, z27.h, z26.h
    ld1rh {z27.h}, p0/z, [x0, #26]      // row 5

    // Row 1 + SWP row 6
    fmla z3.h, p0/m, z28.h, z24.h
    fmla z4.h, p0/m, z28.h, z25.h
    fmla z5.h, p0/m, z28.h, z26.h
    ld1rh {z28.h}, p0/z, [x0, #28]      // row 6

    // Row 2 + SWP row 7
    fmla z6.h, p0/m, z29.h, z24.h
    fmla z7.h, p0/m, z29.h, z25.h
    fmla z8.h, p0/m, z29.h, z26.h
    ld1rh {z29.h}, p0/z, [x0, #30]      // row 7

    // Rows 3-4
    fmla z9.h, p0/m, z30.h, z24.h
    fmla z10.h, p0/m, z30.h, z25.h
    fmla z11.h, p0/m, z30.h, z26.h

    fmla z12.h, p0/m, z31.h, z24.h
    fmla z13.h, p0/m, z31.h, z25.h
    fmla z14.h, p0/m, z31.h, z26.h

    // Rows 5-7
    fmla z15.h, p0/m, z27.h, z24.h
    fmla z16.h, p0/m, z27.h, z25.h
    fmla z17.h, p0/m, z27.h, z26.h

    fmla z18.h, p0/m, z28.h, z24.h
    fmla z19.h, p0/m, z28.h, z25.h
    fmla z20.h, p0/m, z28.h, z26.h

    fmla z21.h, p0/m, z29.h, z24.h
    fmla z22.h, p0/m, z29.h, z25.h
    fmla z23.h, p0/m, z29.h, z26.h

    // Advance pointers
    add x0, x0, #32         // 2K × MR=8 × 2 bytes
    add x1, x1, #384        // 2K × 3 vecs × 64 bytes

    subs x6, x6, #1
    beq .Lh8x3_store

    // Pre-load next B
    ld1h {z24.h}, p0/z, [x1]
    ld1h {z25.h}, p0/z, [x1, #1, mul vl]
    ld1h {z26.h}, p0/z, [x1, #2, mul vl]
    b .Lh8x3_loop

.Lh8x3_store:
    // Convert fp16 accumulators → fp32 and store
    // Each row: 3 fp16 vecs → 6 fp32 vecs

    // Row 0: z0, z1, z2
    uunpklo z27.s, z0.h
    fcvt z27.s, p1/m, z27.h
    st1w {z27.s}, p1, [x2]
    uunpkhi z27.s, z0.h
    fcvt z27.s, p1/m, z27.h
    st1w {z27.s}, p1, [x2, #1, mul vl]
    uunpklo z27.s, z1.h
    fcvt z27.s, p1/m, z27.h
    st1w {z27.s}, p1, [x2, #2, mul vl]
    uunpkhi z27.s, z1.h
    fcvt z27.s, p1/m, z27.h
    st1w {z27.s}, p1, [x2, #3, mul vl]
    uunpklo z27.s, z2.h
    fcvt z27.s, p1/m, z27.h
    st1w {z27.s}, p1, [x2, #4, mul vl]
    uunpkhi z27.s, z2.h
    fcvt z27.s, p1/m, z27.h
    st1w {z27.s}, p1, [x2, #5, mul vl]
    add x2, x2, x5

    // Row 1: z3, z4, z5
    uunpklo z27.s, z3.h
    fcvt z27.s, p1/m, z27.h
    st1w {z27.s}, p1, [x2]
    uunpkhi z27.s, z3.h
    fcvt z27.s, p1/m, z27.h
    st1w {z27.s}, p1, [x2, #1, mul vl]
    uunpklo z27.s, z4.h
    fcvt z27.s, p1/m, z27.h
    st1w {z27.s}, p1, [x2, #2, mul vl]
    uunpkhi z27.s, z4.h
    fcvt z27.s, p1/m, z27.h
    st1w {z27.s}, p1, [x2, #3, mul vl]
    uunpklo z27.s, z5.h
    fcvt z27.s, p1/m, z27.h
    st1w {z27.s}, p1, [x2, #4, mul vl]
    uunpkhi z27.s, z5.h
    fcvt z27.s, p1/m, z27.h
    st1w {z27.s}, p1, [x2, #5, mul vl]
    add x2, x2, x5

    // Row 2: z6, z7, z8
    uunpklo z27.s, z6.h
    fcvt z27.s, p1/m, z27.h
    st1w {z27.s}, p1, [x2]
    uunpkhi z27.s, z6.h
    fcvt z27.s, p1/m, z27.h
    st1w {z27.s}, p1, [x2, #1, mul vl]
    uunpklo z27.s, z7.h
    fcvt z27.s, p1/m, z27.h
    st1w {z27.s}, p1, [x2, #2, mul vl]
    uunpkhi z27.s, z7.h
    fcvt z27.s, p1/m, z27.h
    st1w {z27.s}, p1, [x2, #3, mul vl]
    uunpklo z27.s, z8.h
    fcvt z27.s, p1/m, z27.h
    st1w {z27.s}, p1, [x2, #4, mul vl]
    uunpkhi z27.s, z8.h
    fcvt z27.s, p1/m, z27.h
    st1w {z27.s}, p1, [x2, #5, mul vl]
    add x2, x2, x5

    // Row 3: z9, z10, z11
    uunpklo z27.s, z9.h
    fcvt z27.s, p1/m, z27.h
    st1w {z27.s}, p1, [x2]
    uunpkhi z27.s, z9.h
    fcvt z27.s, p1/m, z27.h
    st1w {z27.s}, p1, [x2, #1, mul vl]
    uunpklo z27.s, z10.h
    fcvt z27.s, p1/m, z27.h
    st1w {z27.s}, p1, [x2, #2, mul vl]
    uunpkhi z27.s, z10.h
    fcvt z27.s, p1/m, z27.h
    st1w {z27.s}, p1, [x2, #3, mul vl]
    uunpklo z27.s, z11.h
    fcvt z27.s, p1/m, z27.h
    st1w {z27.s}, p1, [x2, #4, mul vl]
    uunpkhi z27.s, z11.h
    fcvt z27.s, p1/m, z27.h
    st1w {z27.s}, p1, [x2, #5, mul vl]
    add x2, x2, x5

    // Row 4: z12, z13, z14
    uunpklo z27.s, z12.h
    fcvt z27.s, p1/m, z27.h
    st1w {z27.s}, p1, [x2]
    uunpkhi z27.s, z12.h
    fcvt z27.s, p1/m, z27.h
    st1w {z27.s}, p1, [x2, #1, mul vl]
    uunpklo z27.s, z13.h
    fcvt z27.s, p1/m, z27.h
    st1w {z27.s}, p1, [x2, #2, mul vl]
    uunpkhi z27.s, z13.h
    fcvt z27.s, p1/m, z27.h
    st1w {z27.s}, p1, [x2, #3, mul vl]
    uunpklo z27.s, z14.h
    fcvt z27.s, p1/m, z27.h
    st1w {z27.s}, p1, [x2, #4, mul vl]
    uunpkhi z27.s, z14.h
    fcvt z27.s, p1/m, z27.h
    st1w {z27.s}, p1, [x2, #5, mul vl]
    add x2, x2, x5

    // Row 5: z15, z16, z17
    uunpklo z27.s, z15.h
    fcvt z27.s, p1/m, z27.h
    st1w {z27.s}, p1, [x2]
    uunpkhi z27.s, z15.h
    fcvt z27.s, p1/m, z27.h
    st1w {z27.s}, p1, [x2, #1, mul vl]
    uunpklo z27.s, z16.h
    fcvt z27.s, p1/m, z27.h
    st1w {z27.s}, p1, [x2, #2, mul vl]
    uunpkhi z27.s, z16.h
    fcvt z27.s, p1/m, z27.h
    st1w {z27.s}, p1, [x2, #3, mul vl]
    uunpklo z27.s, z17.h
    fcvt z27.s, p1/m, z27.h
    st1w {z27.s}, p1, [x2, #4, mul vl]
    uunpkhi z27.s, z17.h
    fcvt z27.s, p1/m, z27.h
    st1w {z27.s}, p1, [x2, #5, mul vl]
    add x2, x2, x5

    // Row 6: z18, z19, z20
    uunpklo z27.s, z18.h
    fcvt z27.s, p1/m, z27.h
    st1w {z27.s}, p1, [x2]
    uunpkhi z27.s, z18.h
    fcvt z27.s, p1/m, z27.h
    st1w {z27.s}, p1, [x2, #1, mul vl]
    uunpklo z27.s, z19.h
    fcvt z27.s, p1/m, z27.h
    st1w {z27.s}, p1, [x2, #2, mul vl]
    uunpkhi z27.s, z19.h
    fcvt z27.s, p1/m, z27.h
    st1w {z27.s}, p1, [x2, #3, mul vl]
    uunpklo z27.s, z20.h
    fcvt z27.s, p1/m, z27.h
    st1w {z27.s}, p1, [x2, #4, mul vl]
    uunpkhi z27.s, z20.h
    fcvt z27.s, p1/m, z27.h
    st1w {z27.s}, p1, [x2, #5, mul vl]
    add x2, x2, x5

    // Row 7: z21, z22, z23
    uunpklo z27.s, z21.h
    fcvt z27.s, p1/m, z27.h
    st1w {z27.s}, p1, [x2]
    uunpkhi z27.s, z21.h
    fcvt z27.s, p1/m, z27.h
    st1w {z27.s}, p1, [x2, #1, mul vl]
    uunpklo z27.s, z22.h
    fcvt z27.s, p1/m, z27.h
    st1w {z27.s}, p1, [x2, #2, mul vl]
    uunpkhi z27.s, z22.h
    fcvt z27.s, p1/m, z27.h
    st1w {z27.s}, p1, [x2, #3, mul vl]
    uunpklo z27.s, z23.h
    fcvt z27.s, p1/m, z27.h
    st1w {z27.s}, p1, [x2, #4, mul vl]
    uunpkhi z27.s, z23.h
    fcvt z27.s, p1/m, z27.h
    st1w {z27.s}, p1, [x2, #5, mul vl]

    ldp d14, d15, [sp, #48]
    ldp d12, d13, [sp, #32]
    ldp d10, d11, [sp, #16]
    ldp d8, d9, [sp], #64
    ret
    .size micro_kernel_fp16_8x3, .-micro_kernel_fp16_8x3
