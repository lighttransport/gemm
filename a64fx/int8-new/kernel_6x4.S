// kernel_6x4.S
// INT8 SDOT GEMM Microkernel for A64FX SVE
// MR=6 rows × NR=64 columns (4 SVE vectors)
//
// Computes: C[6×64] += A[6×256] × B[64×256]^T (with B packed K-major)
//
// Register allocation:
//   z0-z23:  24 accumulators (6 rows × 4 vectors)
//   z24-z29: A row buffers (6 vectors, loaded with ld1rqb)
//   z30-z31: B column buffers (2 vectors, reloaded twice per iteration)
//   p0:      All-true predicate for byte ops
//   p1:      All-true predicate for word ops

    .arch armv8.2-a+sve
    .text
    .align 6

// ============================================================================
// kernel_6x4_256
// Simple, correct version with full vector output
//
// Arguments:
//   x0 = Apack pointer [6×256] int8 row-major
//   x1 = Bpack pointer [256×64] int8 K-major
//   x2 = C pointer [6×64] int32 row-major output
//   x3 = ldc in bytes (typically 64 * 4 = 256)
// ============================================================================
    .global kernel_6x4_256
    .type kernel_6x4_256, @function

kernel_6x4_256:
    // Save callee-saved registers
    stp     x19, x20, [sp, #-16]!
    stp     x21, x22, [sp, #-16]!
    stp     d8, d9, [sp, #-16]!
    stp     d10, d11, [sp, #-16]!
    stp     d12, d13, [sp, #-16]!

    // Setup predicates
    ptrue   p0.b                    // All int8 lanes active
    ptrue   p1.s                    // All int32 lanes active

    // ========================================
    // Initialize 24 accumulators to zero
    // z0-z23: 6 rows × 4 vectors
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
    fmov    z20.s, #0
    fmov    z21.s, #0
    fmov    z22.s, #0
    fmov    z23.s, #0

    // Setup pointers and counter (6 row pointers)
    mov     x4, x0                  // Apack row 0
    add     x10, x0, #256           // Apack row 1
    add     x11, x10, #256          // Apack row 2
    add     x12, x11, #256          // Apack row 3
    add     x13, x12, #256          // Apack row 4
    add     x14, x13, #256          // Apack row 5
    mov     x5, x1                  // Bpack base
    mov     x6, #256                // K counter (fixed K=256)

    .align 6
.Lk_loop_6x4:
    // ========================================
    // Load A rows (6 vectors using scalar load + broadcast)
    // Load 4 bytes as int32 and broadcast to all lanes
    // This ensures all lanes get the same A values for SDOT
    // ========================================
    ldr     w15, [x4]                // Load A[row0, k:k+3] as int32
    ldr     w16, [x10]               // Load A[row1, k:k+3] as int32
    ldr     w17, [x11]               // Load A[row2, k:k+3] as int32
    ldr     w19, [x12]               // Load A[row3, k:k+3] as int32
    ldr     w20, [x13]               // Load A[row4, k:k+3] as int32
    ldr     w21, [x14]               // Load A[row5, k:k+3] as int32

    // Broadcast to all lanes of each vector
    mov     z24.s, w15               // Broadcast row 0
    mov     z25.s, w16               // Broadcast row 1
    mov     z26.s, w17               // Broadcast row 2
    mov     z27.s, w19               // Broadcast row 3
    mov     z28.s, w20               // Broadcast row 4
    mov     z29.s, w21               // Broadcast row 5

    // ========================================
    // Load B cols 0-1 and compute 12 SDOTs
    // ========================================
    ld1b    z30.b, p0/z, [x5, #0, mul vl]   // B col 0
    ld1b    z31.b, p0/z, [x5, #1, mul vl]   // B col 1

    // Row 0: C[0, 0:1] += A[0] · B[0:1]
    sdot    z0.s, z24.b, z30.b
    sdot    z1.s, z24.b, z31.b

    // Row 1: C[1, 0:1] += A[1] · B[0:1]
    sdot    z4.s, z25.b, z30.b
    sdot    z5.s, z25.b, z31.b

    // Row 2: C[2, 0:1] += A[2] · B[0:1]
    sdot    z8.s, z26.b, z30.b
    sdot    z9.s, z26.b, z31.b

    // Row 3: C[3, 0:1] += A[3] · B[0:1]
    sdot    z12.s, z27.b, z30.b
    sdot    z13.s, z27.b, z31.b

    // Row 4: C[4, 0:1] += A[4] · B[0:1]
    sdot    z16.s, z28.b, z30.b
    sdot    z17.s, z28.b, z31.b

    // Row 5: C[5, 0:1] += A[5] · B[0:1]
    sdot    z20.s, z29.b, z30.b
    sdot    z21.s, z29.b, z31.b

    // ========================================
    // Load B cols 2-3 and compute 12 SDOTs
    // ========================================
    ld1b    z30.b, p0/z, [x5, #2, mul vl]   // B col 2
    ld1b    z31.b, p0/z, [x5, #3, mul vl]   // B col 3

    // Row 0: C[0, 2:3] += A[0] · B[2:3]
    sdot    z2.s, z24.b, z30.b
    sdot    z3.s, z24.b, z31.b

    // Row 1: C[1, 2:3] += A[1] · B[2:3]
    sdot    z6.s, z25.b, z30.b
    sdot    z7.s, z25.b, z31.b

    // Row 2: C[2, 2:3] += A[2] · B[2:3]
    sdot    z10.s, z26.b, z30.b
    sdot    z11.s, z26.b, z31.b

    // Row 3: C[3, 2:3] += A[3] · B[2:3]
    sdot    z14.s, z27.b, z30.b
    sdot    z15.s, z27.b, z31.b

    // Row 4: C[4, 2:3] += A[4] · B[2:3]
    sdot    z18.s, z28.b, z30.b
    sdot    z19.s, z28.b, z31.b

    // Row 5: C[5, 2:3] += A[5] · B[2:3]
    sdot    z22.s, z29.b, z30.b
    sdot    z23.s, z29.b, z31.b

    // ========================================
    // Advance pointers
    // SDOT groups 4 bytes per iteration
    // ========================================
    add     x4, x4, #4              // Apack row 0 += 4 bytes
    add     x10, x10, #4            // Apack row 1 += 4 bytes
    add     x11, x11, #4            // Apack row 2 += 4 bytes
    add     x12, x12, #4            // Apack row 3 += 4 bytes
    add     x13, x13, #4            // Apack row 4 += 4 bytes
    add     x14, x14, #4            // Apack row 5 += 4 bytes
    add     x5, x5, #256            // Bpack += 4 vectors × 64 bytes
    subs    x6, x6, #4              // K -= 4
    b.gt    .Lk_loop_6x4

    // ========================================
    // Store results (full vectors, 24 vectors × 16 int32 lanes each)
    // Output: C[6×64] int32 row-major
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
    add     x7, x7, x8
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

    // Row 5: Store z20-z23
    add     x7, x7, x8
    st1w    z20.s, p1, [x7, #0, mul vl]
    st1w    z21.s, p1, [x7, #1, mul vl]
    st1w    z22.s, p1, [x7, #2, mul vl]
    st1w    z23.s, p1, [x7, #3, mul vl]

    // Restore callee-saved registers
    ldp     d12, d13, [sp], #16
    ldp     d10, d11, [sp], #16
    ldp     d8, d9, [sp], #16
    ldp     x21, x22, [sp], #16
    ldp     x19, x20, [sp], #16

    ret

    .size kernel_6x4_256, .-kernel_6x4_256
