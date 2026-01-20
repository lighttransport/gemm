// K-tiled 6-row INT8 GEMM kernel for D=512 with L1 cache optimization
// Cache-optimized for A64FX: L1=11 cycles, L2=27-36 cycles
//
// C = A @ B (INT8)
// A: [M=6, K=512] INT8
// B: [K=512, N=2048] INT8 (row-major in memory)
// C: [M=6, N=2048] INT32 accumulator
//
// Strategy:
// - Process K in 4 tiles of 32 K-groups each (128 K-elements per tile)
// - Each tile: 32 K-groups × 8KB = 256 KB working set (fits L2 better)
// - Better temporal locality: reuse B data across K-tiles
// - 6-row split loading within each tile

.global gemm_6row_int8_d512_ktile
.type gemm_6row_int8_d512_ktile, @function

// void gemm_6row_int8_d512_ktile(
//     const int8_t* A,      // x0: [6, 512] row-major
//     const int8_t* B,      // x1: [512, 2048] row-major
//     int32_t* C,           // x2: [6, 2048] row-major
//     int M,                // x3: (unused, always 6)
//     int K,                // x4: (unused, always 512)
//     int N                 // x5: (unused, always 2048)
// )

gemm_6row_int8_d512_ktile:
    // Save callee-saved registers
    stp     x19, x20, [sp, #-96]!
    stp     x21, x22, [sp, #16]
    stp     x23, x24, [sp, #32]
    stp     x25, x26, [sp, #48]
    stp     x27, x28, [sp, #64]
    str     x29, [sp, #80]

    // Constants
    mov     x19, #512               // K dimension
    mov     x20, #2048              // N dimension
    mov     x21, #4                 // K-group size
    mov     x22, #64                // N-chunk size

    // Predicate setup
    ptrue   p0.b, ALL               // Full predicate for 64-byte loads
    ptrue   p1.s, ALL               // Full predicate for 32-bit operations

    // Calculate strides
    mov     x23, #8192              // C row stride: 2048 * 4 bytes
    mov     x24, #256               // B N-chunk stride: 64 * 4 bytes

    // Zero accumulators (z0-z23)
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

    // K-tile loop: 4 tiles (128 K-groups / 32 per tile)
    mov     x29, #0                 // k_tile counter

k_tile_loop:
    // K-tile base: k_tile * 32 K-groups
    lsl     x10, x29, #5            // k_tile * 32

    // K-group loop within tile: 32 iterations
    mov     x11, #0                 // k_group_in_tile counter

k_group_loop:
    // Absolute K-group index
    add     x25, x10, x11           // k_group = k_tile*32 + k_group_in_tile

    // Calculate B base for this K-group
    // B_k = B + k_group * 4 * 2048 = B + k_group * 8192
    lsl     x26, x25, #13           // k_group * 8192
    add     x6, x1, x26             // B base pointer

    // N-chunk loop: 32 iterations (2048/64)
    mov     x27, #0                 // n_chunk counter

n_chunk_loop:
    // Calculate N-chunk offset
    // B_kn = B_k + n_chunk * 64 * 4 = B_k + n_chunk * 256
    lsl     x28, x27, #8            // n_chunk * 256
    add     x7, x6, x28             // B pointer for this chunk

    // Prefetch next N-chunk (256 bytes ahead)
    add     x9, x7, #256
    prfm    pldl1keep, [x9]

    // Calculate A pointers for this K-group
    // A[row, k_group*4:(k_group+1)*4]
    lsl     x9, x25, #2             // k_group * 4
    add     x8, x0, x9              // A row 0

    // Load K vectors from B (4 elements per K-group)
    ld1b    {z24.b}, p0/z, [x7]
    ld1b    {z25.b}, p0/z, [x7, #1, MUL VL]
    ld1b    {z26.b}, p0/z, [x7, #2, MUL VL]
    ld1b    {z27.b}, p0/z, [x7, #3, MUL VL]

    // PHASE 1: Load Q[0-3] and compute rows 0-3
    // Row 0 Q vectors (broadcast 4 INT8 elements as INT32)
    ld1rw   {z28.s}, p1/z, [x8]

    // Row 1
    add     x9, x8, x19             // A row 1 = A + 512
    ld1rw   {z29.s}, p1/z, [x9]

    // Row 2
    add     x9, x9, x19             // A row 2 = A + 1024
    ld1rw   {z30.s}, p1/z, [x9]

    // Row 3
    add     x9, x9, x19             // A row 3 = A + 1536
    ld1rw   {z31.s}, p1/z, [x9]

    // Compute rows 0-3 (16 SDOTs)
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

    // PHASE 2: Reuse z28-z29 for Q[4-5] and compute rows 4-5
    // Row 4
    add     x9, x9, x19             // A row 4 = A + 2048
    ld1rw   {z28.s}, p1/z, [x9]

    // Row 5
    add     x9, x9, x19             // A row 5 = A + 2560
    ld1rw   {z29.s}, p1/z, [x9]

    // Compute rows 4-5 (8 SDOTs)
    sdot    z16.s, z24.b, z28.b
    sdot    z17.s, z25.b, z28.b
    sdot    z18.s, z26.b, z28.b
    sdot    z19.s, z27.b, z28.b

    sdot    z20.s, z24.b, z29.b
    sdot    z21.s, z25.b, z29.b
    sdot    z22.s, z26.b, z29.b
    sdot    z23.s, z27.b, z29.b

    // Next N-chunk
    add     x27, x27, #1
    cmp     x27, #32
    b.lt    n_chunk_loop

    // Next K-group within tile
    add     x11, x11, #1
    cmp     x11, #32
    b.lt    k_group_loop

    // Next K-tile
    add     x29, x29, #1
    cmp     x29, #4
    b.lt    k_tile_loop

    // Store results to C
    mov     x8, x2                  // C base

    // Store row 0
    st1w    {z0.s}, p1, [x8]
    st1w    {z1.s}, p1, [x8, #1, MUL VL]
    st1w    {z2.s}, p1, [x8, #2, MUL VL]
    st1w    {z3.s}, p1, [x8, #3, MUL VL]

    // Store row 1
    add     x8, x8, x23             // C + row_stride
    st1w    {z4.s}, p1, [x8]
    st1w    {z5.s}, p1, [x8, #1, MUL VL]
    st1w    {z6.s}, p1, [x8, #2, MUL VL]
    st1w    {z7.s}, p1, [x8, #3, MUL VL]

    // Store row 2
    add     x8, x8, x23
    st1w    {z8.s}, p1, [x8]
    st1w    {z9.s}, p1, [x8, #1, MUL VL]
    st1w    {z10.s}, p1, [x8, #2, MUL VL]
    st1w    {z11.s}, p1, [x8, #3, MUL VL]

    // Store row 3
    add     x8, x8, x23
    st1w    {z12.s}, p1, [x8]
    st1w    {z13.s}, p1, [x8, #1, MUL VL]
    st1w    {z14.s}, p1, [x8, #2, MUL VL]
    st1w    {z15.s}, p1, [x8, #3, MUL VL]

    // Store row 4
    add     x8, x8, x23
    st1w    {z16.s}, p1, [x8]
    st1w    {z17.s}, p1, [x8, #1, MUL VL]
    st1w    {z18.s}, p1, [x8, #2, MUL VL]
    st1w    {z19.s}, p1, [x8, #3, MUL VL]

    // Store row 5
    add     x8, x8, x23
    st1w    {z20.s}, p1, [x8]
    st1w    {z21.s}, p1, [x8, #1, MUL VL]
    st1w    {z22.s}, p1, [x8, #2, MUL VL]
    st1w    {z23.s}, p1, [x8, #3, MUL VL]

    // Restore callee-saved registers
    ldr     x29, [sp, #80]
    ldp     x27, x28, [sp, #64]
    ldp     x25, x26, [sp, #48]
    ldp     x23, x24, [sp, #32]
    ldp     x21, x22, [sp, #16]
    ldp     x19, x20, [sp], #96

    ret

.size gemm_6row_int8_d512_ktile, .-gemm_6row_int8_d512_ktile
