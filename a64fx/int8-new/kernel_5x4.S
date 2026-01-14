// kernel_5x4.S
// INT8 SDOT GEMM Microkernel for A64FX SVE
// MR=5 rows × NR=64 columns (4 SVE vectors)
//
// Computes: C[5×64] += A[5×256] × B[64×256]^T (with B packed K-major)
//
// Register allocation:
//   z0-z19:  20 accumulators (5 rows × 4 vectors)
//   z20-z24: A row buffers (5 vectors, loaded with ld1rqb)
//   z25-z28: B column buffers (4 vectors, loaded with ld1b)
//   z29-z31: Spare (unused in simple version)
//   p0:      All-true predicate for byte ops
//   p1:      All-true predicate for word ops

    .arch armv8.2-a+sve
    .text
    .align 6

// ============================================================================
// kernel_5x4_256
// Simple, correct version with scalar output (SADDV reduction)
//
// Arguments:
//   x0 = Apack pointer [5×256] int8 row-major
//   x1 = Bpack pointer [256×64] int8 K-major
//   x2 = C pointer [5×64] int32 row-major output
//   x3 = ldc in bytes (typically 64 * 4 = 256)
// ============================================================================
    .global kernel_5x4_256
    .type kernel_5x4_256, @function

kernel_5x4_256:
    // Save callee-saved registers
    stp     x19, x20, [sp, #-16]!
    stp     x21, x22, [sp, #-16]!
    stp     d8, d9, [sp, #-16]!
    stp     d10, d11, [sp, #-16]!

    // Setup predicates
    ptrue   p0.b                    // All int8 lanes active
    ptrue   p1.s                    // All int32 lanes active

    // ========================================
    // Initialize 20 accumulators to zero
    // z0-z19: 5 rows × 4 vectors
    // ========================================
    fmov    z0.s, #0
    fmov    z1.s, #0
    fmov    z2.s, #0
    fmov    z3.s, #0
    fmov    z4.s, #0
    fmov    z5.s, #0
    fmov    z6.s, #0
    fmov    z7.s, #0
    fmov    z8.s, #0
    fmov    z9.s, #0
    fmov    z10.s, #0
    fmov    z11.s, #0
    fmov    z12.s, #0
    fmov    z13.s, #0
    fmov    z14.s, #0
    fmov    z15.s, #0
    fmov    z16.s, #0
    fmov    z17.s, #0
    fmov    z18.s, #0
    fmov    z19.s, #0

    // Setup pointers and counter
    mov     x4, x0                  // Apack row 0
    add     x10, x0, #256           // Apack row 1
    add     x11, x10, #256          // Apack row 2
    add     x12, x11, #256          // Apack row 3
    add     x13, x12, #256          // Apack row 4
    mov     x5, x1                  // Bpack base
    mov     x6, #256                // K counter (fixed K=256)

    .align 6
.Lk_loop_5x4:
    // ========================================
    // Load A rows (5 vectors using ld1rw + dup pattern)
    // Need to load 4 bytes and replicate across entire vector
    // ld1rqb loads 16 bytes which causes wrong replication pattern
    // Solution: Load 4 bytes as int32, then extract and dup as bytes
    // ========================================
    // Load 4 bytes from each row into scalar
    ldr     w14, [x4]                // Load A[row0, k:k+3] as int32
    ldr     w15, [x10]               // Load A[row1, k:k+3] as int32
    ldr     w16, [x11]               // Load A[row2, k:k+3] as int32
    ldr     w17, [x12]               // Load A[row3, k:k+3] as int32
    ldr     w19, [x13]               // Load A[row4, k:k+3] as int32

    // Move to vector and duplicate the 4-byte pattern
    mov     z20.s, w14               // Broadcast int32 to all lanes
    mov     z21.s, w15
    mov     z22.s, w16
    mov     z23.s, w17
    mov     z24.s, w19

    // ========================================
    // Load B vectors (4 full 64-byte vectors)
    // Each vector contains 64 int8 values (one per lane)
    // ========================================
    ld1b    z25.b, p0/z, [x5, #0, mul vl]   // B col 0
    ld1b    z26.b, p0/z, [x5, #1, mul vl]   // B col 1
    ld1b    z27.b, p0/z, [x5, #2, mul vl]   // B col 2
    ld1b    z28.b, p0/z, [x5, #3, mul vl]   // B col 3

    // ========================================
    // 20 SDOT instructions (5 rows × 4 cols)
    // SDOT groups 4 consecutive int8 into 1 int32
    // Each SDOT: z_acc.s += z_a.b * z_b.b
    // ========================================

    // Row 0: C[0, 0:4] += A[0] · B[0:4]
    sdot    z0.s, z20.b, z25.b
    sdot    z1.s, z20.b, z26.b
    sdot    z2.s, z20.b, z27.b
    sdot    z3.s, z20.b, z28.b

    // Row 1: C[1, 0:4] += A[1] · B[0:4]
    sdot    z4.s, z21.b, z25.b
    sdot    z5.s, z21.b, z26.b
    sdot    z6.s, z21.b, z27.b
    sdot    z7.s, z21.b, z28.b

    // Row 2: C[2, 0:4] += A[2] · B[0:4]
    sdot    z8.s, z22.b, z25.b
    sdot    z9.s, z22.b, z26.b
    sdot    z10.s, z22.b, z27.b
    sdot    z11.s, z22.b, z28.b

    // Row 3: C[3, 0:4] += A[3] · B[0:4]
    sdot    z12.s, z23.b, z25.b
    sdot    z13.s, z23.b, z26.b
    sdot    z14.s, z23.b, z27.b
    sdot    z15.s, z23.b, z28.b

    // Row 4: C[4, 0:4] += A[4] · B[0:4]
    sdot    z16.s, z24.b, z25.b
    sdot    z17.s, z24.b, z26.b
    sdot    z18.s, z24.b, z27.b
    sdot    z19.s, z24.b, z28.b

    // ========================================
    // Advance pointers
    // SDOT groups 4 bytes per iteration
    // ========================================
    add     x4, x4, #4              // Apack row 0 += 4 bytes
    add     x10, x10, #4            // Apack row 1 += 4 bytes
    add     x11, x11, #4            // Apack row 2 += 4 bytes
    add     x12, x12, #4            // Apack row 3 += 4 bytes
    add     x13, x13, #4            // Apack row 4 += 4 bytes
    add     x5, x5, #256            // Bpack += 4 vectors × 64 bytes
    subs    x6, x6, #4              // K -= 4
    b.gt    .Lk_loop_5x4

    // ========================================
    // Store results (full vectors, 20 vectors × 16 int32 lanes each)
    // Output: C[5×64] int32 row-major
    // Each accumulator holds 16 int32 values (one SVE vector)
    // Row layout: z0-z3 for row 0 (64 values), z4-z7 for row 1, etc.
    // ========================================

    mov     x7, x2                  // C base pointer
    mov     x8, x3                  // ldc in bytes

    // Row 0: Store z0-z3 (64 int32 values)
    st1w    z0.s, p1, [x7, #0, mul vl]    // C[0, 0:15]
    st1w    z1.s, p1, [x7, #1, mul vl]    // C[0, 16:31]
    st1w    z2.s, p1, [x7, #2, mul vl]    // C[0, 32:47]
    st1w    z3.s, p1, [x7, #3, mul vl]    // C[0, 48:63]

    // Row 1: Store z4-z7
    add     x7, x7, x8                    // Advance to next row
    st1w    z4.s, p1, [x7, #0, mul vl]
    st1w    z5.s, p1, [x7, #1, mul vl]
    st1w    z6.s, p1, [x7, #2, mul vl]
    st1w    z7.s, p1, [x7, #3, mul vl]

    // Row 2: Store z8-z11
    add     x7, x7, x8
    st1w    z8.s, p1, [x7, #0, mul vl]
    st1w    z9.s, p1, [x7, #1, mul vl]
    st1w    z10.s, p1, [x7, #2, mul vl]
    st1w    z11.s, p1, [x7, #3, mul vl]

    // Row 3: Store z12-z15
    add     x7, x7, x8
    st1w    z12.s, p1, [x7, #0, mul vl]
    st1w    z13.s, p1, [x7, #1, mul vl]
    st1w    z14.s, p1, [x7, #2, mul vl]
    st1w    z15.s, p1, [x7, #3, mul vl]

    // Row 4: Store z16-z19
    add     x7, x7, x8
    st1w    z16.s, p1, [x7, #0, mul vl]
    st1w    z17.s, p1, [x7, #1, mul vl]
    st1w    z18.s, p1, [x7, #2, mul vl]
    st1w    z19.s, p1, [x7, #3, mul vl]

    // Restore callee-saved registers
    ldp     d10, d11, [sp], #16
    ldp     d8, d9, [sp], #16
    ldp     x21, x22, [sp], #16
    ldp     x19, x20, [sp], #16

    ret

    .size kernel_5x4_256, .-kernel_5x4_256
