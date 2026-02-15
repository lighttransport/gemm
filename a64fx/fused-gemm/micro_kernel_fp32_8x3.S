// FP32 FMLA Micro-kernel: 8×3 tile with 2× K-loop unroll
// BEST arithmetic intensity: 3.43 FLOPS/byte (vs 2.74 for 6×4)
//
// Why 8×3 is superior:
//   - Memory per K: 8×4 (A) + 3×64 (B) = 32 + 192 = 224 bytes
//   - FLOPs per K: 24 FMLAs × 32 = 768
//   - Arithmetic intensity: 768/224 = 3.43 (25% better than 6×4!)
//
// Register allocation:
//   Z0-Z23:  24 accumulators (8 rows × 3 columns)
//   Z24-Z26: 3 B vectors
//   Z27-Z30: 4 A temps (for 4 rows at a time)
//   Z31:     spare
//
// Input format:
//   A: packed as [K][MR] = [K][8] - 32 bytes per k
//   B: packed as [K][NR] = [K][48] - 192 bytes per k (3 vectors)
//   C: [MR][NR] = [8][48] stored with ldc stride
//   K: must be even (K % 2 == 0)

    .arch armv8.2-a+sve
    .text
    .align 4
    .global micro_kernel_fp32_8x3
    .type micro_kernel_fp32_8x3, %function

micro_kernel_fp32_8x3:
    // Arguments:
    // x0: A (float *)
    // x1: B (float *)
    // x2: C (float *)
    // x3: K (must be even)
    // x4: unused
    // x5: ldc in bytes

    // Save callee-saved D8-D15
    stp d8, d9, [sp, #-64]!
    stp d10, d11, [sp, #16]
    stp d12, d13, [sp, #32]
    stp d14, d15, [sp, #48]

    // Setup predicates
    ptrue p0.s

    // Load B for first iteration (3 columns = 48 floats)
    ld1w {z24.s}, p0/z, [x1]              // B[k, 0:15]
    ld1w {z25.s}, p0/z, [x1, #1, mul vl]  // B[k, 16:31]
    ld1w {z26.s}, p0/z, [x1, #2, mul vl]  // B[k, 32:47]

    // Initialize 24 accumulators with EOR (fast zeroing)
    // Row 0: Z0-Z2
    eor z0.d, z0.d, z0.d
    eor z1.d, z1.d, z1.d
    eor z2.d, z2.d, z2.d
    // Row 1: Z3-Z5
    eor z3.d, z3.d, z3.d
    eor z4.d, z4.d, z4.d
    eor z5.d, z5.d, z5.d
    // Row 2: Z6-Z8
    eor z6.d, z6.d, z6.d
    eor z7.d, z7.d, z7.d
    eor z8.d, z8.d, z8.d
    // Row 3: Z9-Z11
    eor z9.d, z9.d, z9.d
    eor z10.d, z10.d, z10.d
    eor z11.d, z11.d, z11.d
    // Row 4: Z12-Z14
    eor z12.d, z12.d, z12.d
    eor z13.d, z13.d, z13.d
    eor z14.d, z14.d, z14.d
    // Row 5: Z15-Z17
    eor z15.d, z15.d, z15.d
    eor z16.d, z16.d, z16.d
    eor z17.d, z17.d, z17.d
    // Row 6: Z18-Z20
    eor z18.d, z18.d, z18.d
    eor z19.d, z19.d, z19.d
    eor z20.d, z20.d, z20.d
    // Row 7: Z21-Z23
    eor z21.d, z21.d, z21.d
    eor z22.d, z22.d, z22.d
    eor z23.d, z23.d, z23.d

    // K loop counter (process 2 K per iteration)
    lsr x6, x3, #1

.Lloop_start:
    // ═══════════════════════════════════════════════════════════════════
    // First K iteration (k=0)
    // ═══════════════════════════════════════════════════════════════════

    // Load A[0..3][k] with broadcast
    ld1rw {z27.s}, p0/z, [x0, #0]   // A[0,k]
    ld1rw {z28.s}, p0/z, [x0, #4]   // A[1,k]
    ld1rw {z29.s}, p0/z, [x0, #8]   // A[2,k]
    ld1rw {z30.s}, p0/z, [x0, #12]  // A[3,k]

    // FMA rows 0-3 (12 FMLAs)
    fmla z0.s, p0/m, z27.s, z24.s   // C[0,0:15] += A[0,k] * B[k,0:15]
    fmla z1.s, p0/m, z27.s, z25.s   // C[0,16:31]
    fmla z2.s, p0/m, z27.s, z26.s   // C[0,32:47]

    fmla z3.s, p0/m, z28.s, z24.s   // C[1,0:15]
    fmla z4.s, p0/m, z28.s, z25.s   // C[1,16:31]
    fmla z5.s, p0/m, z28.s, z26.s   // C[1,32:47]

    fmla z6.s, p0/m, z29.s, z24.s   // C[2,0:15]
    fmla z7.s, p0/m, z29.s, z25.s   // C[2,16:31]
    fmla z8.s, p0/m, z29.s, z26.s   // C[2,32:47]

    fmla z9.s, p0/m, z30.s, z24.s   // C[3,0:15]
    fmla z10.s, p0/m, z30.s, z25.s  // C[3,16:31]
    fmla z11.s, p0/m, z30.s, z26.s  // C[3,32:47]

    // Load A[4..7][k] with broadcast
    ld1rw {z27.s}, p0/z, [x0, #16]  // A[4,k]
    ld1rw {z28.s}, p0/z, [x0, #20]  // A[5,k]
    ld1rw {z29.s}, p0/z, [x0, #24]  // A[6,k]
    ld1rw {z30.s}, p0/z, [x0, #28]  // A[7,k]

    // FMA rows 4-7 (12 FMLAs)
    fmla z12.s, p0/m, z27.s, z24.s  // C[4,0:15]
    fmla z13.s, p0/m, z27.s, z25.s  // C[4,16:31]
    fmla z14.s, p0/m, z27.s, z26.s  // C[4,32:47]

    fmla z15.s, p0/m, z28.s, z24.s  // C[5,0:15]
    fmla z16.s, p0/m, z28.s, z25.s  // C[5,16:31]
    fmla z17.s, p0/m, z28.s, z26.s  // C[5,32:47]

    fmla z18.s, p0/m, z29.s, z24.s  // C[6,0:15]
    fmla z19.s, p0/m, z29.s, z25.s  // C[6,16:31]
    fmla z20.s, p0/m, z29.s, z26.s  // C[6,32:47]

    fmla z21.s, p0/m, z30.s, z24.s  // C[7,0:15]
    fmla z22.s, p0/m, z30.s, z25.s  // C[7,16:31]
    fmla z23.s, p0/m, z30.s, z26.s  // C[7,32:47]

    // Load next B (k+1) - 3 vectors
    ld1w {z24.s}, p0/z, [x1, #3, mul vl]  // B[k+1, 0:15]
    ld1w {z25.s}, p0/z, [x1, #4, mul vl]  // B[k+1, 16:31]
    ld1w {z26.s}, p0/z, [x1, #5, mul vl]  // B[k+1, 32:47]

    // ═══════════════════════════════════════════════════════════════════
    // Second K iteration (k+1)
    // ═══════════════════════════════════════════════════════════════════

    // Load A[0..3][k+1] with broadcast
    ld1rw {z27.s}, p0/z, [x0, #32]  // A[0,k+1]
    ld1rw {z28.s}, p0/z, [x0, #36]  // A[1,k+1]
    ld1rw {z29.s}, p0/z, [x0, #40]  // A[2,k+1]
    ld1rw {z30.s}, p0/z, [x0, #44]  // A[3,k+1]

    // FMA rows 0-3 (12 FMLAs)
    fmla z0.s, p0/m, z27.s, z24.s
    fmla z1.s, p0/m, z27.s, z25.s
    fmla z2.s, p0/m, z27.s, z26.s

    fmla z3.s, p0/m, z28.s, z24.s
    fmla z4.s, p0/m, z28.s, z25.s
    fmla z5.s, p0/m, z28.s, z26.s

    fmla z6.s, p0/m, z29.s, z24.s
    fmla z7.s, p0/m, z29.s, z25.s
    fmla z8.s, p0/m, z29.s, z26.s

    fmla z9.s, p0/m, z30.s, z24.s
    fmla z10.s, p0/m, z30.s, z25.s
    fmla z11.s, p0/m, z30.s, z26.s

    // Load A[4..7][k+1] with broadcast
    ld1rw {z27.s}, p0/z, [x0, #48]  // A[4,k+1]
    ld1rw {z28.s}, p0/z, [x0, #52]  // A[5,k+1]
    ld1rw {z29.s}, p0/z, [x0, #56]  // A[6,k+1]
    ld1rw {z30.s}, p0/z, [x0, #60]  // A[7,k+1]

    // FMA rows 4-7 (12 FMLAs)
    fmla z12.s, p0/m, z27.s, z24.s
    fmla z13.s, p0/m, z27.s, z25.s
    fmla z14.s, p0/m, z27.s, z26.s

    fmla z15.s, p0/m, z28.s, z24.s
    fmla z16.s, p0/m, z28.s, z25.s
    fmla z17.s, p0/m, z28.s, z26.s

    fmla z18.s, p0/m, z29.s, z24.s
    fmla z19.s, p0/m, z29.s, z25.s
    fmla z20.s, p0/m, z29.s, z26.s

    fmla z21.s, p0/m, z30.s, z24.s
    fmla z22.s, p0/m, z30.s, z25.s
    fmla z23.s, p0/m, z30.s, z26.s

    // Advance pointers (2 K iterations)
    add x0, x0, #64   // A: 2 * 8 floats = 64 bytes
    add x1, x1, #384  // B: 2 * 3 vectors * 64 bytes = 384 bytes

    subs x6, x6, #1
    beq .Lstoreout

    // Load next B (k+2) for next loop iteration
    ld1w {z24.s}, p0/z, [x1]
    ld1w {z25.s}, p0/z, [x1, #1, mul vl]
    ld1w {z26.s}, p0/z, [x1, #2, mul vl]

    b .Lloop_start

.Lstoreout:
    // Store results to C (8 rows × 3 columns = 48 floats per row)
    // Row 0
    st1w {z0.s}, p0, [x2]
    st1w {z1.s}, p0, [x2, #1, mul vl]
    st1w {z2.s}, p0, [x2, #2, mul vl]
    add x2, x2, x5

    // Row 1
    st1w {z3.s}, p0, [x2]
    st1w {z4.s}, p0, [x2, #1, mul vl]
    st1w {z5.s}, p0, [x2, #2, mul vl]
    add x2, x2, x5

    // Row 2
    st1w {z6.s}, p0, [x2]
    st1w {z7.s}, p0, [x2, #1, mul vl]
    st1w {z8.s}, p0, [x2, #2, mul vl]
    add x2, x2, x5

    // Row 3
    st1w {z9.s}, p0, [x2]
    st1w {z10.s}, p0, [x2, #1, mul vl]
    st1w {z11.s}, p0, [x2, #2, mul vl]
    add x2, x2, x5

    // Row 4
    st1w {z12.s}, p0, [x2]
    st1w {z13.s}, p0, [x2, #1, mul vl]
    st1w {z14.s}, p0, [x2, #2, mul vl]
    add x2, x2, x5

    // Row 5
    st1w {z15.s}, p0, [x2]
    st1w {z16.s}, p0, [x2, #1, mul vl]
    st1w {z17.s}, p0, [x2, #2, mul vl]
    add x2, x2, x5

    // Row 6
    st1w {z18.s}, p0, [x2]
    st1w {z19.s}, p0, [x2, #1, mul vl]
    st1w {z20.s}, p0, [x2, #2, mul vl]
    add x2, x2, x5

    // Row 7
    st1w {z21.s}, p0, [x2]
    st1w {z22.s}, p0, [x2, #1, mul vl]
    st1w {z23.s}, p0, [x2, #2, mul vl]

    // Restore callee-saved D8-D15
    ldp d14, d15, [sp, #48]
    ldp d12, d13, [sp, #32]
    ldp d10, d11, [sp, #16]
    ldp d8, d9, [sp], #64
    ret
    .size micro_kernel_fp32_8x3, .-micro_kernel_fp32_8x3
