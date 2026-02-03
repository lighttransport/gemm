/*
 * FlashAttention-style Tiled exp2 + GEMM
 *
 * Key insight: Use LD1RW (load-replicate) instead of DUP
 * - DUP runs on FLA pipe (competes with FMLA)
 * - LD1RW runs on load/store pipe (parallel with FMLA)
 *
 * Tiling strategy:
 * - Tile Nc into Kc blocks that fit exp2 output in L1
 * - Tile D into Dc blocks for output accumulation
 * - Inner loop: exp2 + GEMM with LD1RW broadcast
 */

    .arch armv8.2-a+sve
    .text
    .align 4

/*============================================================================
 * exp2_flash_tiled_4x4 - FlashAttention-style fused exp2 + GEMM
 *
 * void exp2_flash_tiled_4x4(
 *     const int32_t* S,   // x0: [4][Nc] scores
 *     const float* V,     // x1: [Nc][64] values
 *     float* O,           // x2: [4][64] output
 *     float* P,           // x3: [4][Nc] exp2 buffer (L1 resident)
 *     int Nc,             // x4: sequence length
 *     float scale,        // s0
 *     float max_val,      // s1
 *     int ld_s,           // x5: S stride in elements
 *     int ld_v,           // x6: V stride in bytes (64*4=256)
 *     int ld_o            // x7: O stride in bytes
 * );
 *
 * Key: Uses LD1RW to broadcast P values (runs on LD/ST pipe, not FLA)
 *============================================================================*/

    .global exp2_flash_tiled_4x4
    .type exp2_flash_tiled_4x4, %function
exp2_flash_tiled_4x4:
    stp     d8, d9, [sp, #-64]!
    stp     d10, d11, [sp, #16]
    stp     d12, d13, [sp, #32]
    stp     d14, d15, [sp, #48]

    ptrue   p0.s

    /* Setup constants */
    mov     z20.s, s0                   // scale
    mov     z21.s, s1                   // max_val

    /* exp2 constants */
    mov     w8, #0x42800000             // 64.0f
    mov     z22.s, w8
    mov     w8, #127
    mov     z23.s, w8                   // IEEE754 bias

    /* Zero output accumulators */
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

    cbz     x4, .Lflash_store

    /* S row pointers (stride = ld_s * 4 bytes) */
    lsl     x5, x5, #2
    mov     x8, x0                      // S row 0
    add     x9, x0, x5                  // S row 1
    add     x10, x9, x5                 // S row 2
    add     x11, x10, x5                // S row 3

    /* P row pointers (stride = Nc * 4 bytes) */
    lsl     x12, x4, #2                 // Nc * 4
    mov     x13, x3                     // P row 0
    add     x14, x3, x12                // P row 1
    add     x15, x14, x12               // P row 2
    add     x16, x15, x12               // P row 3

    mov     x17, x4                     // loop counter

/*----------------------------------------------------------------------------
 * Main loop: Per-K iteration with LD1RW broadcast
 *
 * Pipeline:
 *   Cycle 0-5: exp2 computation (uses z24-z28)
 *   Cycle 2-7: Store P (overlaps with exp2, uses store pipe)
 *   Cycle 6-9: LD1RW broadcast + FMLA (LD1RW on LD pipe, FMLA on FLA)
 *----------------------------------------------------------------------------*/
.Lflash_loop:
    /* Load S[row, k] for all 4 rows into NEON v24 */
    /* Use only w18 as scratch (x18 is platform register, caller-saved) */
    ldr     w18, [x8], #4               // S[0, k]
    fmov    s24, w18                    // v24.s[0] = S[0,k]
    ldr     w18, [x9], #4               // S[1, k]
    mov     v24.s[1], w18               // v24.s[1] = S[1,k]
    ldr     w18, [x10], #4              // S[2, k]
    mov     v24.s[2], w18               // v24.s[2] = S[2,k]
    ldr     w18, [x11], #4              // S[3, k]
    mov     v24.s[3], w18               // v24.s[3] = S[3,k]

    /* Now z24[0:3] contains the 4 int32 S values */
    /* Convert to float and compute x = S * scale - max */
    scvtf   z24.s, p0/m, z24.s          // int32 -> float
    fmul    z24.s, z24.s, z20.s         // * scale
    fsub    z24.s, z24.s, z21.s         // - max

    /* FEXPA-based exp2: encode (N+127)<<6 | m into FEXPA input */
    frintm  z25.s, p0/m, z24.s          // N = floor(x)
    fsub    z26.s, z24.s, z25.s         // f = x - N (0 <= f < 1)
    fmul    z26.s, z26.s, z22.s         // f * 64
    fcvtzs  z26.s, p0/m, z26.s          // m = floor(f * 64) as signed int
    fcvtzs  z25.s, p0/m, z25.s          // N as signed int
    add     z25.s, z25.s, z23.s         // N + 127
    lsl     z25.s, z25.s, #6            // (N + 127) << 6
    orr     z24.s, z25.s, z26.s         // ((N+127) << 6) | m
    fexpa   z24.s, z24.s                // exp2 result in z24[0:3]

    /* Store exp2 to P buffer (uses store pipe, overlaps with FMLA below) */
    str     s24, [x13], #4              // P[0, k]
    mov     w18, v24.s[1]
    str     w18, [x14], #4              // P[1, k]
    mov     w18, v24.s[2]
    str     w18, [x15], #4              // P[2, k]
    mov     w18, v24.s[3]
    str     w18, [x16], #4              // P[3, k]

    /* LD1RW: Load and replicate (runs on LOAD pipe, not FLA!) */
    /* This is the key optimization - no DUP needed */
    /* Use x18 for all temp addresses (avoid callee-saved registers) */
    sub     x18, x13, #4                // P[0, k] address
    ld1rw   {z24.s}, p0/z, [x18]        // Broadcast P[0,k] to all lanes
    sub     x18, x14, #4                // P[1, k] address
    ld1rw   {z25.s}, p0/z, [x18]        // Broadcast P[1,k]
    sub     x18, x15, #4
    ld1rw   {z26.s}, p0/z, [x18]        // Broadcast P[2,k]
    sub     x18, x16, #4
    ld1rw   {z27.s}, p0/z, [x18]        // Broadcast P[3,k]

    /* Load V[k, :] - 4 vectors for 64 elements */
    ld1w    {z28.s}, p0/z, [x1, #0, mul vl]
    ld1w    {z29.s}, p0/z, [x1, #1, mul vl]
    ld1w    {z30.s}, p0/z, [x1, #2, mul vl]
    ld1w    {z31.s}, p0/z, [x1, #3, mul vl]
    add     x1, x1, x6                  // V += ld_v

    /* 16 FMLAs - all on FLA pipes, no conflict with LD1RW */
    fmla    z0.s, p0/m, z24.s, z28.s    // O[0,:] += P[0,k] * V[k,:]
    fmla    z1.s, p0/m, z24.s, z29.s
    fmla    z2.s, p0/m, z24.s, z30.s
    fmla    z3.s, p0/m, z24.s, z31.s
    fmla    z4.s, p0/m, z25.s, z28.s    // O[1,:] += P[1,k] * V[k,:]
    fmla    z5.s, p0/m, z25.s, z29.s
    fmla    z6.s, p0/m, z25.s, z30.s
    fmla    z7.s, p0/m, z25.s, z31.s
    fmla    z8.s, p0/m, z26.s, z28.s    // O[2,:] += P[2,k] * V[k,:]
    fmla    z9.s, p0/m, z26.s, z29.s
    fmla    z10.s, p0/m, z26.s, z30.s
    fmla    z11.s, p0/m, z26.s, z31.s
    fmla    z12.s, p0/m, z27.s, z28.s   // O[3,:] += P[3,k] * V[k,:]
    fmla    z13.s, p0/m, z27.s, z29.s
    fmla    z14.s, p0/m, z27.s, z30.s
    fmla    z15.s, p0/m, z27.s, z31.s

    subs    x17, x17, #1
    b.gt    .Lflash_loop

.Lflash_store:
    /* Store output */
    mov     x8, x2
    st1w    {z0.s}, p0, [x8, #0, mul vl]
    st1w    {z1.s}, p0, [x8, #1, mul vl]
    st1w    {z2.s}, p0, [x8, #2, mul vl]
    st1w    {z3.s}, p0, [x8, #3, mul vl]
    add     x8, x8, x7
    st1w    {z4.s}, p0, [x8, #0, mul vl]
    st1w    {z5.s}, p0, [x8, #1, mul vl]
    st1w    {z6.s}, p0, [x8, #2, mul vl]
    st1w    {z7.s}, p0, [x8, #3, mul vl]
    add     x8, x8, x7
    st1w    {z8.s}, p0, [x8, #0, mul vl]
    st1w    {z9.s}, p0, [x8, #1, mul vl]
    st1w    {z10.s}, p0, [x8, #2, mul vl]
    st1w    {z11.s}, p0, [x8, #3, mul vl]
    add     x8, x8, x7
    st1w    {z12.s}, p0, [x8, #0, mul vl]
    st1w    {z13.s}, p0, [x8, #1, mul vl]
    st1w    {z14.s}, p0, [x8, #2, mul vl]
    st1w    {z15.s}, p0, [x8, #3, mul vl]

    ldp     d14, d15, [sp, #48]
    ldp     d12, d13, [sp, #32]
    ldp     d10, d11, [sp, #16]
    ldp     d8, d9, [sp], #64
    ret

    .size exp2_flash_tiled_4x4, .-exp2_flash_tiled_4x4


/*============================================================================
 * exp2_flash_ld1rw_4x4 - Optimized version with better pipelining
 *
 * Key optimization: Pre-compute P addresses, use LD1RW directly after store
 * The LD1RW will hit store buffer or L1 cache (same address just stored)
 *============================================================================*/

    .global exp2_flash_ld1rw_4x4
    .type exp2_flash_ld1rw_4x4, %function
exp2_flash_ld1rw_4x4:
    stp     d8, d9, [sp, #-64]!
    stp     d10, d11, [sp, #16]
    stp     d12, d13, [sp, #32]
    stp     d14, d15, [sp, #48]

    ptrue   p0.s

    /* Setup constants */
    mov     z20.s, s0                   // scale
    mov     z21.s, s1                   // max_val
    mov     w8, #0x42800000
    mov     z22.s, w8                   // 64.0f
    mov     w8, #127
    mov     z23.s, w8                   // bias

    /* Zero accumulators */
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

    cbz     x4, .Lld1rw_store

    /* S row pointers */
    lsl     x5, x5, #2
    mov     x8, x0
    add     x9, x0, x5
    add     x10, x9, x5
    add     x11, x10, x5

    /* P row pointers */
    lsl     x12, x4, #2
    mov     x13, x3
    add     x14, x3, x12
    add     x15, x14, x12
    add     x16, x15, x12

    mov     x17, x4

/*----------------------------------------------------------------------------
 * Main loop with software pipelining
 *
 * Instruction scheduling for A64FX:
 * - 2 FLA pipes: FMLA, FEXPA, FRINTM, FCVT, FMUL, FSUB
 * - 2 LD/ST pipes: LD1W, ST1W, LD1RW (KEY!)
 * - 2 EXA pipes: ADD, LSL, ORR
 *
 * LD1RW runs on LD/ST pipe -> parallel with FMLA!
 *----------------------------------------------------------------------------*/
.Lld1rw_loop:
    /* === Phase 1: Load 4 S values into NEON v28 === */
    ldr     w12, [x8], #4               // S[0, k]
    ldr     w18, [x9], #4               // S[1, k]
    fmov    s28, w12                    // v28.s[0] = S[0,k]
    mov     v28.s[1], w18               // v28.s[1] = S[1,k]
    ldr     w12, [x10], #4              // S[2, k]
    ldr     w18, [x11], #4              // S[3, k]
    mov     v28.s[2], w12               // v28.s[2] = S[2,k]
    mov     v28.s[3], w18               // v28.s[3] = S[3,k]

    /* Convert to float and compute x = S * scale - max */
    scvtf   z28.s, p0/m, z28.s
    fmul    z28.s, z28.s, z20.s
    fsub    z28.s, z28.s, z21.s

    /* === Phase 2: exp2 via FEXPA on z28 === */
    frintm  z29.s, p0/m, z28.s          // N = floor(x)
    fsub    z30.s, z28.s, z29.s         // f = x - N
    fmul    z30.s, z30.s, z22.s         // f * 64
    fcvtzs  z30.s, p0/m, z30.s          // m = floor(f*64)
    fcvtzs  z29.s, p0/m, z29.s          // N as int
    add     z29.s, z29.s, z23.s         // N + 127
    lsl     z29.s, z29.s, #6            // (N + 127) << 6
    orr     z28.s, z29.s, z30.s         // combine
    fexpa   z28.s, z28.s                // exp2 result in z28[0:3]

    /* === Phase 3: Store to P and immediately LD1RW === */
    /* Store P[0,k] then LD1RW from same address */
    str     s28, [x13]
    ld1rw   {z24.s}, p0/z, [x13]      // Broadcast P[0,k]
    add     x13, x13, #4

    mov     w18, v28.s[1]
    str     w18, [x14]
    ld1rw   {z25.s}, p0/z, [x14]      // Broadcast P[1,k]
    add     x14, x14, #4

    mov     w18, v28.s[2]
    str     w18, [x15]
    ld1rw   {z26.s}, p0/z, [x15]      // Broadcast P[2,k]
    add     x15, x15, #4

    mov     w18, v28.s[3]
    str     w18, [x16]
    ld1rw   {z27.s}, p0/z, [x16]      // Broadcast P[3,k]
    add     x16, x16, #4

    /* === Phase 4: Load V[k,:] === */
    /* z24-z27 = P broadcast, so load V to z28-z31 */
    ld1w    {z28.s}, p0/z, [x1, #0, mul vl]
    ld1w    {z29.s}, p0/z, [x1, #1, mul vl]
    ld1w    {z30.s}, p0/z, [x1, #2, mul vl]
    ld1w    {z31.s}, p0/z, [x1, #3, mul vl]
    add     x1, x1, x6

    /* === Phase 5: 16 FMLAs === */
    /* z24-z27 = broadcast P values, z28-z31 = V row */
    fmla    z0.s, p0/m, z24.s, z28.s
    fmla    z1.s, p0/m, z24.s, z29.s
    fmla    z2.s, p0/m, z24.s, z30.s
    fmla    z3.s, p0/m, z24.s, z31.s
    fmla    z4.s, p0/m, z25.s, z28.s
    fmla    z5.s, p0/m, z25.s, z29.s
    fmla    z6.s, p0/m, z25.s, z30.s
    fmla    z7.s, p0/m, z25.s, z31.s
    fmla    z8.s, p0/m, z26.s, z28.s
    fmla    z9.s, p0/m, z26.s, z29.s
    fmla    z10.s, p0/m, z26.s, z30.s
    fmla    z11.s, p0/m, z26.s, z31.s
    fmla    z12.s, p0/m, z27.s, z28.s
    fmla    z13.s, p0/m, z27.s, z29.s
    fmla    z14.s, p0/m, z27.s, z30.s
    fmla    z15.s, p0/m, z27.s, z31.s

    subs    x17, x17, #1
    b.gt    .Lld1rw_loop

.Lld1rw_store:
    mov     x8, x2
    st1w    {z0.s}, p0, [x8, #0, mul vl]
    st1w    {z1.s}, p0, [x8, #1, mul vl]
    st1w    {z2.s}, p0, [x8, #2, mul vl]
    st1w    {z3.s}, p0, [x8, #3, mul vl]
    add     x8, x8, x7
    st1w    {z4.s}, p0, [x8, #0, mul vl]
    st1w    {z5.s}, p0, [x8, #1, mul vl]
    st1w    {z6.s}, p0, [x8, #2, mul vl]
    st1w    {z7.s}, p0, [x8, #3, mul vl]
    add     x8, x8, x7
    st1w    {z8.s}, p0, [x8, #0, mul vl]
    st1w    {z9.s}, p0, [x8, #1, mul vl]
    st1w    {z10.s}, p0, [x8, #2, mul vl]
    st1w    {z11.s}, p0, [x8, #3, mul vl]
    add     x8, x8, x7
    st1w    {z12.s}, p0, [x8, #0, mul vl]
    st1w    {z13.s}, p0, [x8, #1, mul vl]
    st1w    {z14.s}, p0, [x8, #2, mul vl]
    st1w    {z15.s}, p0, [x8, #3, mul vl]

    ldp     d14, d15, [sp, #48]
    ldp     d12, d13, [sp, #32]
    ldp     d10, d11, [sp, #16]
    ldp     d8, d9, [sp], #64
    ret

    .size exp2_flash_ld1rw_4x4, .-exp2_flash_ld1rw_4x4

