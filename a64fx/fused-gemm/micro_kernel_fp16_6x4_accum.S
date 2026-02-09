// FP16 6x4 ACCUMULATE variant
// Mixed precision: fp16 FMLA loop -> convert to fp32, add to existing C, store
//
// Computes: C[6][128](fp32) += A[K][6](fp16) x B[K][128](fp16)
// MR = 6 rows, NR = 128 fp16 = 4 SVE vectors of 32 fp16 each
// 24 accumulators, 2K-unrolled with SWP A reloads for rows 4-5
//
// Same K-loop as init variant. Epilogue: uunpklo/hi + fcvt + ld1w + fadd + st1w
//
// Registers:
//   Z0-Z23:  24 fp16 accumulators (6 rows x 4 col-groups of 32 fp16)
//            Row i, col j -> z[i*4 + j]
//            Row 0: z0,z1,z2,z3    Row 1: z4,z5,z6,z7
//            Row 2: z8,z9,z10,z11  Row 3: z12,z13,z14,z15
//            Row 4: z16,z17,z18,z19  Row 5: z20,z21,z22,z23
//   Z24-Z27: B vectors (4 columns of 32 fp16)
//   Z28-Z31: A broadcast temps (rows 0-3 pre-loaded; rows 4-5 SWP-reloaded)
//   p0: ptrue .h (32 fp16 lanes for FMLA)
//   p1: ptrue .s (16 fp32 lanes for store epilogue)
//
// Args: x0=A(fp16), x1=B(fp16), x2=C(fp32), x3=K(even), x4=unused, x5=ldc_bytes

    .arch armv8.2-a+sve
    .text
    .align 4
    .global micro_kernel_fp16_6x4_accum
    .type micro_kernel_fp16_6x4_accum, %function

micro_kernel_fp16_6x4_accum:
    stp d8, d9, [sp, #-64]!
    stp d10, d11, [sp, #16]
    stp d12, d13, [sp, #32]
    stp d14, d15, [sp, #48]

    ptrue p0.h
    ptrue p1.s

    // Load initial B (4 SVE vecs of 32 fp16)
    ld1h {z24.h}, p0/z, [x1]
    ld1h {z25.h}, p0/z, [x1, #1, mul vl]
    ld1h {z26.h}, p0/z, [x1, #2, mul vl]
    ld1h {z27.h}, p0/z, [x1, #3, mul vl]

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

    lsr x6, x3, #1          // x6 = K/2 loop iterations

.Lh6x4a_loop:
    // =========================================================
    // K iteration 0: B at [x1, #0..#3, mul vl], A at x0+0..10
    // =========================================================

    // Pre-load A rows 0-3
    ld1rh {z28.h}, p0/z, [x0, #0]     // A[0,k]
    ld1rh {z29.h}, p0/z, [x0, #2]     // A[1,k]
    ld1rh {z30.h}, p0/z, [x0, #4]     // A[2,k]
    ld1rh {z31.h}, p0/z, [x0, #6]     // A[3,k]

    // Row 0 (z28): 4 FMAs, then SWP reload row 4
    fmla z0.h, p0/m, z28.h, z24.h
    fmla z1.h, p0/m, z28.h, z25.h
    fmla z2.h, p0/m, z28.h, z26.h
    fmla z3.h, p0/m, z28.h, z27.h
    ld1rh {z28.h}, p0/z, [x0, #8]     // A[4,k] (SWP reload)

    // Row 1 (z29): 4 FMAs, then SWP reload row 5
    fmla z4.h, p0/m, z29.h, z24.h
    fmla z5.h, p0/m, z29.h, z25.h
    fmla z6.h, p0/m, z29.h, z26.h
    fmla z7.h, p0/m, z29.h, z27.h
    ld1rh {z29.h}, p0/z, [x0, #10]    // A[5,k] (SWP reload)

    // Row 2 (z30): 4 FMAs (no reload)
    fmla z8.h, p0/m, z30.h, z24.h
    fmla z9.h, p0/m, z30.h, z25.h
    fmla z10.h, p0/m, z30.h, z26.h
    fmla z11.h, p0/m, z30.h, z27.h

    // Row 3 (z31): 4 FMAs (no reload)
    fmla z12.h, p0/m, z31.h, z24.h
    fmla z13.h, p0/m, z31.h, z25.h
    fmla z14.h, p0/m, z31.h, z26.h
    fmla z15.h, p0/m, z31.h, z27.h

    // Row 4 (z28, SWP-reloaded): 4 FMAs
    fmla z16.h, p0/m, z28.h, z24.h
    fmla z17.h, p0/m, z28.h, z25.h
    fmla z18.h, p0/m, z28.h, z26.h
    fmla z19.h, p0/m, z28.h, z27.h

    // Row 5 (z29, SWP-reloaded): 4 FMAs
    fmla z20.h, p0/m, z29.h, z24.h
    fmla z21.h, p0/m, z29.h, z25.h
    fmla z22.h, p0/m, z29.h, z26.h
    fmla z23.h, p0/m, z29.h, z27.h

    // =========================================================
    // K iteration 1: B at [x1, #4..#7, mul vl], A at x0+12..22
    // =========================================================

    // Load B(k+1)
    ld1h {z24.h}, p0/z, [x1, #4, mul vl]
    ld1h {z25.h}, p0/z, [x1, #5, mul vl]
    ld1h {z26.h}, p0/z, [x1, #6, mul vl]
    ld1h {z27.h}, p0/z, [x1, #7, mul vl]

    // Pre-load A rows 0-3 for k+1
    ld1rh {z28.h}, p0/z, [x0, #12]    // A[0,k+1]
    ld1rh {z29.h}, p0/z, [x0, #14]    // A[1,k+1]
    ld1rh {z30.h}, p0/z, [x0, #16]    // A[2,k+1]
    ld1rh {z31.h}, p0/z, [x0, #18]    // A[3,k+1]

    // Row 0 (z28): 4 FMAs, then SWP reload row 4
    fmla z0.h, p0/m, z28.h, z24.h
    fmla z1.h, p0/m, z28.h, z25.h
    fmla z2.h, p0/m, z28.h, z26.h
    fmla z3.h, p0/m, z28.h, z27.h
    ld1rh {z28.h}, p0/z, [x0, #20]    // A[4,k+1] (SWP reload)

    // Row 1 (z29): 4 FMAs, then SWP reload row 5
    fmla z4.h, p0/m, z29.h, z24.h
    fmla z5.h, p0/m, z29.h, z25.h
    fmla z6.h, p0/m, z29.h, z26.h
    fmla z7.h, p0/m, z29.h, z27.h
    ld1rh {z29.h}, p0/z, [x0, #22]    // A[5,k+1] (SWP reload)

    // Row 2 (z30): 4 FMAs
    fmla z8.h, p0/m, z30.h, z24.h
    fmla z9.h, p0/m, z30.h, z25.h
    fmla z10.h, p0/m, z30.h, z26.h
    fmla z11.h, p0/m, z30.h, z27.h

    // Row 3 (z31): 4 FMAs
    fmla z12.h, p0/m, z31.h, z24.h
    fmla z13.h, p0/m, z31.h, z25.h
    fmla z14.h, p0/m, z31.h, z26.h
    fmla z15.h, p0/m, z31.h, z27.h

    // Row 4 (z28, SWP-reloaded): 4 FMAs
    fmla z16.h, p0/m, z28.h, z24.h
    fmla z17.h, p0/m, z28.h, z25.h
    fmla z18.h, p0/m, z28.h, z26.h
    fmla z19.h, p0/m, z28.h, z27.h

    // Row 5 (z29, SWP-reloaded): 4 FMAs
    fmla z20.h, p0/m, z29.h, z24.h
    fmla z21.h, p0/m, z29.h, z25.h
    fmla z22.h, p0/m, z29.h, z26.h
    fmla z23.h, p0/m, z29.h, z27.h

    // Advance pointers: A += 2K * 6 * 2 = 24, B += 2K * 4 * 64 = 512
    add x0, x0, #24
    add x1, x1, #512

    subs x6, x6, #1
    beq .Lh6x4a_store

    // Pre-load next B for next iteration
    ld1h {z24.h}, p0/z, [x1]
    ld1h {z25.h}, p0/z, [x1, #1, mul vl]
    ld1h {z26.h}, p0/z, [x1, #2, mul vl]
    ld1h {z27.h}, p0/z, [x1, #3, mul vl]
    b .Lh6x4a_loop

.Lh6x4a_store:
    // Convert fp16 accumulators -> fp32, add to existing C, store
    // Pattern per 16-col group: uunpklo/hi + fcvt + ld1w + fadd + st1w

    // Row 0: z0 (cols 0-31), z1 (cols 32-63), z2 (cols 64-95), z3 (cols 96-127)
    uunpklo z28.s, z0.h
    fcvt z28.s, p1/m, z28.h
    ld1w {z29.s}, p1/z, [x2]
    fadd z28.s, z28.s, z29.s
    st1w {z28.s}, p1, [x2]
    uunpkhi z28.s, z0.h
    fcvt z28.s, p1/m, z28.h
    ld1w {z29.s}, p1/z, [x2, #1, mul vl]
    fadd z28.s, z28.s, z29.s
    st1w {z28.s}, p1, [x2, #1, mul vl]
    uunpklo z28.s, z1.h
    fcvt z28.s, p1/m, z28.h
    ld1w {z29.s}, p1/z, [x2, #2, mul vl]
    fadd z28.s, z28.s, z29.s
    st1w {z28.s}, p1, [x2, #2, mul vl]
    uunpkhi z28.s, z1.h
    fcvt z28.s, p1/m, z28.h
    ld1w {z29.s}, p1/z, [x2, #3, mul vl]
    fadd z28.s, z28.s, z29.s
    st1w {z28.s}, p1, [x2, #3, mul vl]
    uunpklo z28.s, z2.h
    fcvt z28.s, p1/m, z28.h
    ld1w {z29.s}, p1/z, [x2, #4, mul vl]
    fadd z28.s, z28.s, z29.s
    st1w {z28.s}, p1, [x2, #4, mul vl]
    uunpkhi z28.s, z2.h
    fcvt z28.s, p1/m, z28.h
    ld1w {z29.s}, p1/z, [x2, #5, mul vl]
    fadd z28.s, z28.s, z29.s
    st1w {z28.s}, p1, [x2, #5, mul vl]
    uunpklo z28.s, z3.h
    fcvt z28.s, p1/m, z28.h
    ld1w {z29.s}, p1/z, [x2, #6, mul vl]
    fadd z28.s, z28.s, z29.s
    st1w {z28.s}, p1, [x2, #6, mul vl]
    uunpkhi z28.s, z3.h
    fcvt z28.s, p1/m, z28.h
    ld1w {z29.s}, p1/z, [x2, #7, mul vl]
    fadd z28.s, z28.s, z29.s
    st1w {z28.s}, p1, [x2, #7, mul vl]
    add x2, x2, x5

    // Row 1: z4, z5, z6, z7
    uunpklo z28.s, z4.h
    fcvt z28.s, p1/m, z28.h
    ld1w {z29.s}, p1/z, [x2]
    fadd z28.s, z28.s, z29.s
    st1w {z28.s}, p1, [x2]
    uunpkhi z28.s, z4.h
    fcvt z28.s, p1/m, z28.h
    ld1w {z29.s}, p1/z, [x2, #1, mul vl]
    fadd z28.s, z28.s, z29.s
    st1w {z28.s}, p1, [x2, #1, mul vl]
    uunpklo z28.s, z5.h
    fcvt z28.s, p1/m, z28.h
    ld1w {z29.s}, p1/z, [x2, #2, mul vl]
    fadd z28.s, z28.s, z29.s
    st1w {z28.s}, p1, [x2, #2, mul vl]
    uunpkhi z28.s, z5.h
    fcvt z28.s, p1/m, z28.h
    ld1w {z29.s}, p1/z, [x2, #3, mul vl]
    fadd z28.s, z28.s, z29.s
    st1w {z28.s}, p1, [x2, #3, mul vl]
    uunpklo z28.s, z6.h
    fcvt z28.s, p1/m, z28.h
    ld1w {z29.s}, p1/z, [x2, #4, mul vl]
    fadd z28.s, z28.s, z29.s
    st1w {z28.s}, p1, [x2, #4, mul vl]
    uunpkhi z28.s, z6.h
    fcvt z28.s, p1/m, z28.h
    ld1w {z29.s}, p1/z, [x2, #5, mul vl]
    fadd z28.s, z28.s, z29.s
    st1w {z28.s}, p1, [x2, #5, mul vl]
    uunpklo z28.s, z7.h
    fcvt z28.s, p1/m, z28.h
    ld1w {z29.s}, p1/z, [x2, #6, mul vl]
    fadd z28.s, z28.s, z29.s
    st1w {z28.s}, p1, [x2, #6, mul vl]
    uunpkhi z28.s, z7.h
    fcvt z28.s, p1/m, z28.h
    ld1w {z29.s}, p1/z, [x2, #7, mul vl]
    fadd z28.s, z28.s, z29.s
    st1w {z28.s}, p1, [x2, #7, mul vl]
    add x2, x2, x5

    // Row 2: z8, z9, z10, z11
    uunpklo z28.s, z8.h
    fcvt z28.s, p1/m, z28.h
    ld1w {z29.s}, p1/z, [x2]
    fadd z28.s, z28.s, z29.s
    st1w {z28.s}, p1, [x2]
    uunpkhi z28.s, z8.h
    fcvt z28.s, p1/m, z28.h
    ld1w {z29.s}, p1/z, [x2, #1, mul vl]
    fadd z28.s, z28.s, z29.s
    st1w {z28.s}, p1, [x2, #1, mul vl]
    uunpklo z28.s, z9.h
    fcvt z28.s, p1/m, z28.h
    ld1w {z29.s}, p1/z, [x2, #2, mul vl]
    fadd z28.s, z28.s, z29.s
    st1w {z28.s}, p1, [x2, #2, mul vl]
    uunpkhi z28.s, z9.h
    fcvt z28.s, p1/m, z28.h
    ld1w {z29.s}, p1/z, [x2, #3, mul vl]
    fadd z28.s, z28.s, z29.s
    st1w {z28.s}, p1, [x2, #3, mul vl]
    uunpklo z28.s, z10.h
    fcvt z28.s, p1/m, z28.h
    ld1w {z29.s}, p1/z, [x2, #4, mul vl]
    fadd z28.s, z28.s, z29.s
    st1w {z28.s}, p1, [x2, #4, mul vl]
    uunpkhi z28.s, z10.h
    fcvt z28.s, p1/m, z28.h
    ld1w {z29.s}, p1/z, [x2, #5, mul vl]
    fadd z28.s, z28.s, z29.s
    st1w {z28.s}, p1, [x2, #5, mul vl]
    uunpklo z28.s, z11.h
    fcvt z28.s, p1/m, z28.h
    ld1w {z29.s}, p1/z, [x2, #6, mul vl]
    fadd z28.s, z28.s, z29.s
    st1w {z28.s}, p1, [x2, #6, mul vl]
    uunpkhi z28.s, z11.h
    fcvt z28.s, p1/m, z28.h
    ld1w {z29.s}, p1/z, [x2, #7, mul vl]
    fadd z28.s, z28.s, z29.s
    st1w {z28.s}, p1, [x2, #7, mul vl]
    add x2, x2, x5

    // Row 3: z12, z13, z14, z15
    uunpklo z28.s, z12.h
    fcvt z28.s, p1/m, z28.h
    ld1w {z29.s}, p1/z, [x2]
    fadd z28.s, z28.s, z29.s
    st1w {z28.s}, p1, [x2]
    uunpkhi z28.s, z12.h
    fcvt z28.s, p1/m, z28.h
    ld1w {z29.s}, p1/z, [x2, #1, mul vl]
    fadd z28.s, z28.s, z29.s
    st1w {z28.s}, p1, [x2, #1, mul vl]
    uunpklo z28.s, z13.h
    fcvt z28.s, p1/m, z28.h
    ld1w {z29.s}, p1/z, [x2, #2, mul vl]
    fadd z28.s, z28.s, z29.s
    st1w {z28.s}, p1, [x2, #2, mul vl]
    uunpkhi z28.s, z13.h
    fcvt z28.s, p1/m, z28.h
    ld1w {z29.s}, p1/z, [x2, #3, mul vl]
    fadd z28.s, z28.s, z29.s
    st1w {z28.s}, p1, [x2, #3, mul vl]
    uunpklo z28.s, z14.h
    fcvt z28.s, p1/m, z28.h
    ld1w {z29.s}, p1/z, [x2, #4, mul vl]
    fadd z28.s, z28.s, z29.s
    st1w {z28.s}, p1, [x2, #4, mul vl]
    uunpkhi z28.s, z14.h
    fcvt z28.s, p1/m, z28.h
    ld1w {z29.s}, p1/z, [x2, #5, mul vl]
    fadd z28.s, z28.s, z29.s
    st1w {z28.s}, p1, [x2, #5, mul vl]
    uunpklo z28.s, z15.h
    fcvt z28.s, p1/m, z28.h
    ld1w {z29.s}, p1/z, [x2, #6, mul vl]
    fadd z28.s, z28.s, z29.s
    st1w {z28.s}, p1, [x2, #6, mul vl]
    uunpkhi z28.s, z15.h
    fcvt z28.s, p1/m, z28.h
    ld1w {z29.s}, p1/z, [x2, #7, mul vl]
    fadd z28.s, z28.s, z29.s
    st1w {z28.s}, p1, [x2, #7, mul vl]
    add x2, x2, x5

    // Row 4: z16, z17, z18, z19
    uunpklo z28.s, z16.h
    fcvt z28.s, p1/m, z28.h
    ld1w {z29.s}, p1/z, [x2]
    fadd z28.s, z28.s, z29.s
    st1w {z28.s}, p1, [x2]
    uunpkhi z28.s, z16.h
    fcvt z28.s, p1/m, z28.h
    ld1w {z29.s}, p1/z, [x2, #1, mul vl]
    fadd z28.s, z28.s, z29.s
    st1w {z28.s}, p1, [x2, #1, mul vl]
    uunpklo z28.s, z17.h
    fcvt z28.s, p1/m, z28.h
    ld1w {z29.s}, p1/z, [x2, #2, mul vl]
    fadd z28.s, z28.s, z29.s
    st1w {z28.s}, p1, [x2, #2, mul vl]
    uunpkhi z28.s, z17.h
    fcvt z28.s, p1/m, z28.h
    ld1w {z29.s}, p1/z, [x2, #3, mul vl]
    fadd z28.s, z28.s, z29.s
    st1w {z28.s}, p1, [x2, #3, mul vl]
    uunpklo z28.s, z18.h
    fcvt z28.s, p1/m, z28.h
    ld1w {z29.s}, p1/z, [x2, #4, mul vl]
    fadd z28.s, z28.s, z29.s
    st1w {z28.s}, p1, [x2, #4, mul vl]
    uunpkhi z28.s, z18.h
    fcvt z28.s, p1/m, z28.h
    ld1w {z29.s}, p1/z, [x2, #5, mul vl]
    fadd z28.s, z28.s, z29.s
    st1w {z28.s}, p1, [x2, #5, mul vl]
    uunpklo z28.s, z19.h
    fcvt z28.s, p1/m, z28.h
    ld1w {z29.s}, p1/z, [x2, #6, mul vl]
    fadd z28.s, z28.s, z29.s
    st1w {z28.s}, p1, [x2, #6, mul vl]
    uunpkhi z28.s, z19.h
    fcvt z28.s, p1/m, z28.h
    ld1w {z29.s}, p1/z, [x2, #7, mul vl]
    fadd z28.s, z28.s, z29.s
    st1w {z28.s}, p1, [x2, #7, mul vl]
    add x2, x2, x5

    // Row 5: z20, z21, z22, z23
    uunpklo z28.s, z20.h
    fcvt z28.s, p1/m, z28.h
    ld1w {z29.s}, p1/z, [x2]
    fadd z28.s, z28.s, z29.s
    st1w {z28.s}, p1, [x2]
    uunpkhi z28.s, z20.h
    fcvt z28.s, p1/m, z28.h
    ld1w {z29.s}, p1/z, [x2, #1, mul vl]
    fadd z28.s, z28.s, z29.s
    st1w {z28.s}, p1, [x2, #1, mul vl]
    uunpklo z28.s, z21.h
    fcvt z28.s, p1/m, z28.h
    ld1w {z29.s}, p1/z, [x2, #2, mul vl]
    fadd z28.s, z28.s, z29.s
    st1w {z28.s}, p1, [x2, #2, mul vl]
    uunpkhi z28.s, z21.h
    fcvt z28.s, p1/m, z28.h
    ld1w {z29.s}, p1/z, [x2, #3, mul vl]
    fadd z28.s, z28.s, z29.s
    st1w {z28.s}, p1, [x2, #3, mul vl]
    uunpklo z28.s, z22.h
    fcvt z28.s, p1/m, z28.h
    ld1w {z29.s}, p1/z, [x2, #4, mul vl]
    fadd z28.s, z28.s, z29.s
    st1w {z28.s}, p1, [x2, #4, mul vl]
    uunpkhi z28.s, z22.h
    fcvt z28.s, p1/m, z28.h
    ld1w {z29.s}, p1/z, [x2, #5, mul vl]
    fadd z28.s, z28.s, z29.s
    st1w {z28.s}, p1, [x2, #5, mul vl]
    uunpklo z28.s, z23.h
    fcvt z28.s, p1/m, z28.h
    ld1w {z29.s}, p1/z, [x2, #6, mul vl]
    fadd z28.s, z28.s, z29.s
    st1w {z28.s}, p1, [x2, #6, mul vl]
    uunpkhi z28.s, z23.h
    fcvt z28.s, p1/m, z28.h
    ld1w {z29.s}, p1/z, [x2, #7, mul vl]
    fadd z28.s, z28.s, z29.s
    st1w {z28.s}, p1, [x2, #7, mul vl]

    ldp d14, d15, [sp, #48]
    ldp d12, d13, [sp, #32]
    ldp d10, d11, [sp, #16]
    ldp d8, d9, [sp], #64
    ret
    .size micro_kernel_fp16_6x4_accum, .-micro_kernel_fp16_6x4_accum
