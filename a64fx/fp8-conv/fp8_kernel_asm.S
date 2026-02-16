/*
 * FP8 GEMM Micro-kernel for A64FX SVE (Assembly)
 *
 * 8×3 tile: 8 rows × 3 FP32 vectors = 24 accumulators
 * Uses ld1rw for broadcast loads of A (much faster than load + MOV)
 * K-unrolled by 2 for pipelining
 *
 * Register allocation:
 *   Z0-Z23:  24 accumulators (8 rows × 3 columns)
 *   Z24-Z26: 3 B vectors
 *   Z27-Z30: 4 A temps (broadcast values)
 *   Z31:     spare
 *
 *   p0: all true for full vector ops
 *
 * Arguments:
 *   x0: Ap_f32 (packed A, 8 floats per K = 32 bytes)
 *   x1: Bp_f32 (packed B, 48 floats per K = 192 bytes = 3 vectors)
 *   x2: C (output, row-major)
 *   x3: ldc (leading dimension of C in elements)
 *   x4: K (inner dimension, must be even)
 *
 * Memory layout:
 *   A: [K][8] - 32 bytes per k
 *   B: [K][48] - 192 bytes per k (3 vectors × 16 floats × 4 bytes)
 */

   .text
   .align 4
   .global fp8_gemm_kernel_asm
   .type fp8_gemm_kernel_asm, @function

fp8_gemm_kernel_asm:
    // Setup predicate
    ptrue p0.s

    // Convert ldc to bytes
    lsl x3, x3, #2

    // Load first B (3 vectors)
    ld1w {z24.s}, p0/z, [x1]
    ld1w {z25.s}, p0/z, [x1, #1, mul vl]
    ld1w {z26.s}, p0/z, [x1, #2, mul vl]

    // Initialize 24 accumulators with EOR (fast zeroing)
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

    // K loop counter (process 2 K per iteration)
    lsr x5, x4, #1
    cbz x5, .Lepilogue

.Lloop_start:
    // ═══════════════════════════════════════════════════════════════════
    // First K iteration (k=0)
    // ═══════════════════════════════════════════════════════════════════

    // Load A[0..3][k] with broadcast
    ld1rw {z27.s}, p0/z, [x0, #0]   // A[0,k] broadcast
    ld1rw {z28.s}, p0/z, [x0, #4]   // A[1,k] broadcast
    ld1rw {z29.s}, p0/z, [x0, #8]   // A[2,k] broadcast
    ld1rw {z30.s}, p0/z, [x0, #12]  // A[3,k] broadcast

    // FMA rows 0-3 (12 FMLAs)
    fmla z0.s, p0/m, z27.s, z24.s   // C[0,0:15] += A[0,k] * B[k,0:15]
    fmla z1.s, p0/m, z27.s, z25.s   // C[0,16:31]
    fmla z2.s, p0/m, z27.s, z26.s   // C[0,32:47]

    fmla z3.s, p0/m, z28.s, z24.s
    fmla z4.s, p0/m, z28.s, z25.s
    fmla z5.s, p0/m, z28.s, z26.s

    fmla z6.s, p0/m, z29.s, z24.s
    fmla z7.s, p0/m, z29.s, z25.s
    fmla z8.s, p0/m, z29.s, z26.s

    fmla z9.s, p0/m, z30.s, z24.s
    fmla z10.s, p0/m, z30.s, z25.s
    fmla z11.s, p0/m, z30.s, z26.s

    // Load A[4..7][k] with broadcast
    ld1rw {z27.s}, p0/z, [x0, #16]  // A[4,k]
    ld1rw {z28.s}, p0/z, [x0, #20]  // A[5,k]
    ld1rw {z29.s}, p0/z, [x0, #24]  // A[6,k]
    ld1rw {z30.s}, p0/z, [x0, #28]  // A[7,k]

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

    // Load next B (k+1) - 3 vectors
    ld1w {z24.s}, p0/z, [x1, #3, mul vl]
    ld1w {z25.s}, p0/z, [x1, #4, mul vl]
    ld1w {z26.s}, p0/z, [x1, #5, mul vl]

    // ═══════════════════════════════════════════════════════════════════
    // Second K iteration (k+1)
    // ═══════════════════════════════════════════════════════════════════

    // Load A[0..3][k+1] with broadcast
    ld1rw {z27.s}, p0/z, [x0, #32]  // A[0,k+1]
    ld1rw {z28.s}, p0/z, [x0, #36]  // A[1,k+1]
    ld1rw {z29.s}, p0/z, [x0, #40]  // A[2,k+1]
    ld1rw {z30.s}, p0/z, [x0, #44]  // A[3,k+1]

    // FMA rows 0-3
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

    // FMA rows 4-7
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
    add x0, x0, #64    // A: 2 * 8 floats = 64 bytes
    add x1, x1, #384   // B: 2 * 3 vectors * 64 bytes = 384 bytes

    subs x5, x5, #1
    beq .Lepilogue

    // Load next B (k+2) for next loop iteration
    ld1w {z24.s}, p0/z, [x1]
    ld1w {z25.s}, p0/z, [x1, #1, mul vl]
    ld1w {z26.s}, p0/z, [x1, #2, mul vl]

    b .Lloop_start

.Lepilogue:
    // Store results to C (8 rows × 3 columns = 48 floats per row)
    // Row 0
    st1w {z0.s}, p0, [x2]
    st1w {z1.s}, p0, [x2, #1, mul vl]
    st1w {z2.s}, p0, [x2, #2, mul vl]
    add x2, x2, x3

    // Row 1
    st1w {z3.s}, p0, [x2]
    st1w {z4.s}, p0, [x2, #1, mul vl]
    st1w {z5.s}, p0, [x2, #2, mul vl]
    add x2, x2, x3

    // Row 2
    st1w {z6.s}, p0, [x2]
    st1w {z7.s}, p0, [x2, #1, mul vl]
    st1w {z8.s}, p0, [x2, #2, mul vl]
    add x2, x2, x3

    // Row 3
    st1w {z9.s}, p0, [x2]
    st1w {z10.s}, p0, [x2, #1, mul vl]
    st1w {z11.s}, p0, [x2, #2, mul vl]
    add x2, x2, x3

    // Row 4
    st1w {z12.s}, p0, [x2]
    st1w {z13.s}, p0, [x2, #1, mul vl]
    st1w {z14.s}, p0, [x2, #2, mul vl]
    add x2, x2, x3

    // Row 5
    st1w {z15.s}, p0, [x2]
    st1w {z16.s}, p0, [x2, #1, mul vl]
    st1w {z17.s}, p0, [x2, #2, mul vl]
    add x2, x2, x3

    // Row 6
    st1w {z18.s}, p0, [x2]
    st1w {z19.s}, p0, [x2, #1, mul vl]
    st1w {z20.s}, p0, [x2, #2, mul vl]
    add x2, x2, x3

    // Row 7
    st1w {z21.s}, p0, [x2]
    st1w {z22.s}, p0, [x2, #1, mul vl]
    st1w {z23.s}, p0, [x2, #2, mul vl]

    ret

    .size fp8_gemm_kernel_asm, .-fp8_gemm_kernel_asm
