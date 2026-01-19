// kernel_ffn_6row_gemm.S - 6-row INT8 GEMM with split loading for FFN
// Computes: C[6, N] = A[6, K] @ B[K, N] (INT8 -> INT32)
//
// Register allocation with split loading:
// - z0-z23: 24 accumulators (6 rows x 4 N-vectors)
// - z24-z27: 4 B vectors
// - z28-z31: A vectors (rows 0-3 first, then reuse z28-z29 for rows 4-5)
//
// Parameters:
// x0: A (input matrix, [6, K])
// x1: B (weight matrix, packed [K/4, N, 4])
// x2: C (output matrix, [6, N])
// x3: K (must be multiple of 4)
// x4: N (must be multiple of 64, processed in chunks)

.global kernel_ffn_6row_gemm_d512
.type kernel_ffn_6row_gemm_d512, %function

// Specialized for D_in=512, D_ff=2048
// A: [6, 512]
// B: [512/4=128, 2048, 4] with 256-byte stride
// C: [6, 2048]
kernel_ffn_6row_gemm_d512:
    // Save registers
    stp     x19, x20, [sp, #-80]!
    stp     x21, x22, [sp, #16]
    stp     x23, x24, [sp, #32]
    stp     x25, x26, [sp, #48]
    stp     x27, x28, [sp, #64]

    mov     x19, x0                 // Save A
    mov     x20, x1                 // Save B
    mov     x21, x2                 // Save C
    mov     w22, #512               // K = 512
    mov     w23, #2048              // N = 2048
    mov     w24, #128               // K_groups = 512/4 = 128

    // Setup predicates
    ptrue   p0.b, ALL               // All bytes
    ptrue   p1.s, ALL               // All 32-bit elements

    // N-chunk loop: process 64 columns at a time
    mov     w25, #0                 // n_chunk = 0
    mov     w26, #32                // N_chunks = 2048/64 = 32

.Ln_chunk_loop:
    // Initialize accumulators to zero
    dup     z0.s, #0
    dup     z1.s, #0
    dup     z2.s, #0
    dup     z3.s, #0
    dup     z4.s, #0
    dup     z5.s, #0
    dup     z6.s, #0
    dup     z7.s, #0
    dup     z8.s, #0
    dup     z9.s, #0
    dup     z10.s, #0
    dup     z11.s, #0
    dup     z12.s, #0
    dup     z13.s, #0
    dup     z14.s, #0
    dup     z15.s, #0
    dup     z16.s, #0
    dup     z17.s, #0
    dup     z18.s, #0
    dup     z19.s, #0
    dup     z20.s, #0
    dup     z21.s, #0
    dup     z22.s, #0
    dup     z23.s, #0

    // Calculate B base for this N-chunk
    mov     x5, x20                 // B base
    lsl     x6, x25, #8             // n_chunk * 256 (offset to this chunk)
    add     x5, x5, x6              // B[0, n_chunk*64, 0]

    // K-group loop: iterate over 128 groups of 4
    mov     x6, x5                  // B pointer
    mov     x7, x19                 // A pointer
    mov     w27, w24                // k_group counter = 128

.Lk_group_loop:
    // Load B[k, :, 0:3] - 4 vectors of 64 elements each
    ld1b    z24.b, p0/z, [x6]
    ld1b    z25.b, p0/z, [x6, #1, MUL VL]
    ld1b    z26.b, p0/z, [x6, #2, MUL VL]
    ld1b    z27.b, p0/z, [x6, #3, MUL VL]
    add     x6, x6, #256            // Next K-group (stride 256)

    // Load A[0:3, k] - broadcast 4 elements to each row
    ld1rw   {z28.s}, p1/z, [x7]
    add     x8, x7, #512            // Row 1
    ld1rw   {z29.s}, p1/z, [x8]
    add     x8, x8, #512            // Row 2
    ld1rw   {z30.s}, p1/z, [x8]
    add     x8, x8, #512            // Row 3
    ld1rw   {z31.s}, p1/z, [x8]

    // Compute rows 0-3: 16 SDOTs
    sdot    z0.s, z24.b, z28.b
    sdot    z1.s, z25.b, z28.b
    sdot    z2.s, z26.b, z28.b
    sdot    z3.s, z27.b, z28.b

    sdot    z4.s, z24.b, z29.b
    sdot    z5.s, z25.b, z29.b
    sdot    z6.s, z26.b, z29.b
    sdot    z7.s, z27.b, z29.b

    sdot    z8.s, z24.b, z30.b
    sdot    z9.s, z25.b, z30.b
    sdot    z10.s, z26.b, z30.b
    sdot    z11.s, z27.b, z30.b

    sdot    z12.s, z24.b, z31.b
    sdot    z13.s, z25.b, z31.b
    sdot    z14.s, z26.b, z31.b
    sdot    z15.s, z27.b, z31.b

    // Load A[4:5, k] - reuse z28-z29
    add     x8, x8, #512            // Row 4
    ld1rw   {z28.s}, p1/z, [x8]
    add     x8, x8, #512            // Row 5
    ld1rw   {z29.s}, p1/z, [x8]

    // Compute rows 4-5: 8 SDOTs
    sdot    z16.s, z24.b, z28.b
    sdot    z17.s, z25.b, z28.b
    sdot    z18.s, z26.b, z28.b
    sdot    z19.s, z27.b, z28.b

    sdot    z20.s, z24.b, z29.b
    sdot    z21.s, z25.b, z29.b
    sdot    z22.s, z26.b, z29.b
    sdot    z23.s, z27.b, z29.b

    // Next K-group
    add     x7, x7, #4              // A += 4 elements
    subs    w27, w27, #1
    b.ne    .Lk_group_loop

    // Store results C[0:5, n_chunk*64:(n_chunk+1)*64]
    mov     x7, x21                 // C base
    lsl     x8, x25, #8             // n_chunk * 256 (64 elements * 4 bytes)
    add     x7, x7, x8              // C[0, n_chunk*64]

    // Store row 0
    st1w    z0.s, p1, [x7]
    st1w    z1.s, p1, [x7, #1, MUL VL]
    st1w    z2.s, p1, [x7, #2, MUL VL]
    st1w    z3.s, p1, [x7, #3, MUL VL]

    // Store row 1
    add     x8, x7, #8192           // Row stride: 2048 * 4 bytes
    st1w    z4.s, p1, [x8]
    st1w    z5.s, p1, [x8, #1, MUL VL]
    st1w    z6.s, p1, [x8, #2, MUL VL]
    st1w    z7.s, p1, [x8, #3, MUL VL]

    // Store row 2
    add     x8, x8, #8192
    st1w    z8.s, p1, [x8]
    st1w    z9.s, p1, [x8, #1, MUL VL]
    st1w    z10.s, p1, [x8, #2, MUL VL]
    st1w    z11.s, p1, [x8, #3, MUL VL]

    // Store row 3
    add     x8, x8, #8192
    st1w    z12.s, p1, [x8]
    st1w    z13.s, p1, [x8, #1, MUL VL]
    st1w    z14.s, p1, [x8, #2, MUL VL]
    st1w    z15.s, p1, [x8, #3, MUL VL]

    // Store row 4
    add     x8, x8, #8192
    st1w    z16.s, p1, [x8]
    st1w    z17.s, p1, [x8, #1, MUL VL]
    st1w    z18.s, p1, [x8, #2, MUL VL]
    st1w    z19.s, p1, [x8, #3, MUL VL]

    // Store row 5
    add     x8, x8, #8192
    st1w    z20.s, p1, [x8]
    st1w    z21.s, p1, [x8, #1, MUL VL]
    st1w    z22.s, p1, [x8, #2, MUL VL]
    st1w    z23.s, p1, [x8, #3, MUL VL]

    // Next N-chunk
    add     w25, w25, #1
    cmp     w25, w26
    b.lt    .Ln_chunk_loop

    // Restore registers
    ldp     x27, x28, [sp, #64]
    ldp     x25, x26, [sp, #48]
    ldp     x23, x24, [sp, #32]
    ldp     x21, x22, [sp, #16]
    ldp     x19, x20, [sp], #80
    ret

.size kernel_ffn_6row_gemm_d512, .-kernel_ffn_6row_gemm_d512
