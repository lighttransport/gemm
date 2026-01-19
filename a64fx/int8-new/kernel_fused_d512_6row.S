// kernel_fused_d512_6row.S
// Fused Q@K^T -> Quantize -> P@V for D=512 with 6 rows
//
// 6 rows × 4 N-tiles = 24 accumulators
// Register reuse: load Q[0-3], compute rows 0-3, then load Q[4-5] into z28-z29
//
// With 11-cycle load latency:
// - Load K[0-3] and Q[0-3] together (8 loads)
// - Compute rows 0-3 (16 SDOTs) while Q[4-5] loads
// - Compute rows 4-5 (8 SDOTs)
//
// 24 SDOTs / 10 loads = 2.4 SDOT/load (better than 5-row's 2.22)

    .arch armv8.2-a+sve
    .text
    .align 6

    .global kernel_fused_d512_6row
    .type kernel_fused_d512_6row, @function

// Arguments:
//   x0 = Q pointer [6, 512] row-major
//   x1 = K_int pointer [128, 64, 4]
//   x2 = V_t_int pointer [8, 16, 64, 4]
//   x3 = O output pointer [6, 512] int32

kernel_fused_d512_6row:
    stp     x19, x20, [sp, #-128]!
    stp     x21, x22, [sp, #16]
    stp     x23, x24, [sp, #32]
    stp     x25, x26, [sp, #48]
    stp     x27, x28, [sp, #64]
    stp     d8, d9, [sp, #80]
    stp     d10, d11, [sp, #96]
    stp     d12, d13, [sp, #112]

    ptrue   p0.b
    ptrue   p1.s

    mov     x19, x0                     // Q
    mov     x20, x1                     // K_int
    mov     x21, x2                     // V_t_int
    mov     x22, x3                     // O

    // P buffer: [6, 64] = 384 bytes
    sub     sp, sp, #384
    mov     x24, sp
    mov     x14, #256                   // stride

    // Row stride for Q: D=512 bytes, for O: D*4=2048 bytes
    mov     x25, #512
    mov     x26, #2048

    // ========================================================================
    // PHASE 1: Q @ K^T -> S[6, 64]
    // 128 d_groups, split Q loading to fit in registers
    // ========================================================================

    // Zero 24 accumulators
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

    mov     x5, x19                     // Q base
    mov     x6, x20                     // K_int base
    mov     x7, #128                    // 128 d_groups

.Lqkt_6row_loop:
    // Load K[0-3] into z24-z27
    ld1b    z24.b, p0/z, [x6]
    add     x8, x6, #64
    ld1b    z25.b, p0/z, [x8]
    add     x8, x8, #64
    ld1b    z26.b, p0/z, [x8]
    add     x8, x8, #64
    ld1b    z27.b, p0/z, [x8]
    add     x6, x6, x14

    // Load Q[0-3] into z28-z31 (rows 0-3)
    mov     x8, x5
    ld1rw   {z28.s}, p1/z, [x8]
    add     x8, x8, x25                 // +512
    ld1rw   {z29.s}, p1/z, [x8]
    add     x8, x8, x25
    ld1rw   {z30.s}, p1/z, [x8]
    add     x8, x8, x25
    ld1rw   {z31.s}, p1/z, [x8]

    // Rows 0-3: 16 SDOTs (K reused across all rows)
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

    // Load Q[4-5] into z28-z29 (reusing after rows 0-1 done)
    add     x8, x5, x25, lsl #2         // Q + 4*512 = row 4
    ld1rw   {z28.s}, p1/z, [x8]
    add     x8, x8, x25                 // row 5
    ld1rw   {z29.s}, p1/z, [x8]

    // Rows 4-5: 8 SDOTs
    sdot    z16.s, z24.b, z28.b
    sdot    z17.s, z25.b, z28.b
    sdot    z18.s, z26.b, z28.b
    sdot    z19.s, z27.b, z28.b

    sdot    z20.s, z24.b, z29.b
    sdot    z21.s, z25.b, z29.b
    sdot    z22.s, z26.b, z29.b
    sdot    z23.s, z27.b, z29.b

    add     x5, x5, #4                  // Q advances by 4 bytes

    subs    x7, x7, #1
    b.gt    .Lqkt_6row_loop

    // ========================================================================
    // PHASE 2: Quantize S[6, 64] -> P[6, 64]
    // ========================================================================

    // Helper constants for clamp
    dup     z30.s, #0
    dup     z31.s, #127

    // Row 0
    asr     z24.s, z0.s, #8
    asr     z25.s, z1.s, #8
    asr     z26.s, z2.s, #8
    asr     z27.s, z3.s, #8
    smax    z24.s, p1/m, z24.s, z30.s
    smax    z25.s, p1/m, z25.s, z30.s
    smax    z26.s, p1/m, z26.s, z30.s
    smax    z27.s, p1/m, z27.s, z30.s
    smin    z24.s, p1/m, z24.s, z31.s
    smin    z25.s, p1/m, z25.s, z31.s
    smin    z26.s, p1/m, z26.s, z31.s
    smin    z27.s, p1/m, z27.s, z31.s
    uzp1    z28.h, z24.h, z25.h
    uzp1    z29.h, z26.h, z27.h
    uzp1    z28.b, z28.b, z29.b
    st1b    z28.b, p0, [x24]

    // Row 1
    asr     z24.s, z4.s, #8
    asr     z25.s, z5.s, #8
    asr     z26.s, z6.s, #8
    asr     z27.s, z7.s, #8
    smax    z24.s, p1/m, z24.s, z30.s
    smax    z25.s, p1/m, z25.s, z30.s
    smax    z26.s, p1/m, z26.s, z30.s
    smax    z27.s, p1/m, z27.s, z30.s
    smin    z24.s, p1/m, z24.s, z31.s
    smin    z25.s, p1/m, z25.s, z31.s
    smin    z26.s, p1/m, z26.s, z31.s
    smin    z27.s, p1/m, z27.s, z31.s
    uzp1    z28.h, z24.h, z25.h
    uzp1    z29.h, z26.h, z27.h
    uzp1    z28.b, z28.b, z29.b
    add     x8, x24, #64
    st1b    z28.b, p0, [x8]

    // Row 2
    asr     z24.s, z8.s, #8
    asr     z25.s, z9.s, #8
    asr     z26.s, z10.s, #8
    asr     z27.s, z11.s, #8
    smax    z24.s, p1/m, z24.s, z30.s
    smax    z25.s, p1/m, z25.s, z30.s
    smax    z26.s, p1/m, z26.s, z30.s
    smax    z27.s, p1/m, z27.s, z30.s
    smin    z24.s, p1/m, z24.s, z31.s
    smin    z25.s, p1/m, z25.s, z31.s
    smin    z26.s, p1/m, z26.s, z31.s
    smin    z27.s, p1/m, z27.s, z31.s
    uzp1    z28.h, z24.h, z25.h
    uzp1    z29.h, z26.h, z27.h
    uzp1    z28.b, z28.b, z29.b
    add     x8, x24, #128
    st1b    z28.b, p0, [x8]

    // Row 3
    asr     z24.s, z12.s, #8
    asr     z25.s, z13.s, #8
    asr     z26.s, z14.s, #8
    asr     z27.s, z15.s, #8
    smax    z24.s, p1/m, z24.s, z30.s
    smax    z25.s, p1/m, z25.s, z30.s
    smax    z26.s, p1/m, z26.s, z30.s
    smax    z27.s, p1/m, z27.s, z30.s
    smin    z24.s, p1/m, z24.s, z31.s
    smin    z25.s, p1/m, z25.s, z31.s
    smin    z26.s, p1/m, z26.s, z31.s
    smin    z27.s, p1/m, z27.s, z31.s
    uzp1    z28.h, z24.h, z25.h
    uzp1    z29.h, z26.h, z27.h
    uzp1    z28.b, z28.b, z29.b
    add     x8, x24, #192
    st1b    z28.b, p0, [x8]

    // Row 4
    asr     z24.s, z16.s, #8
    asr     z25.s, z17.s, #8
    asr     z26.s, z18.s, #8
    asr     z27.s, z19.s, #8
    smax    z24.s, p1/m, z24.s, z30.s
    smax    z25.s, p1/m, z25.s, z30.s
    smax    z26.s, p1/m, z26.s, z30.s
    smax    z27.s, p1/m, z27.s, z30.s
    smin    z24.s, p1/m, z24.s, z31.s
    smin    z25.s, p1/m, z25.s, z31.s
    smin    z26.s, p1/m, z26.s, z31.s
    smin    z27.s, p1/m, z27.s, z31.s
    uzp1    z28.h, z24.h, z25.h
    uzp1    z29.h, z26.h, z27.h
    uzp1    z28.b, z28.b, z29.b
    add     x8, x24, #256
    st1b    z28.b, p0, [x8]

    // Row 5
    asr     z24.s, z20.s, #8
    asr     z25.s, z21.s, #8
    asr     z26.s, z22.s, #8
    asr     z27.s, z23.s, #8
    smax    z24.s, p1/m, z24.s, z30.s
    smax    z25.s, p1/m, z25.s, z30.s
    smax    z26.s, p1/m, z26.s, z30.s
    smax    z27.s, p1/m, z27.s, z30.s
    smin    z24.s, p1/m, z24.s, z31.s
    smin    z25.s, p1/m, z25.s, z31.s
    smin    z26.s, p1/m, z26.s, z31.s
    smin    z27.s, p1/m, z27.s, z31.s
    uzp1    z28.h, z24.h, z25.h
    uzp1    z29.h, z26.h, z27.h
    uzp1    z28.b, z28.b, z29.b
    add     x8, x24, #320
    st1b    z28.b, p0, [x8]

    // ========================================================================
    // PHASE 3: P @ V -> O[6, 512]
    // 8 D-tiles, same split approach for P loading
    // ========================================================================

    mov     x9, #8                      // 8 D-tiles
    mov     x15, #4096                  // D_tile stride

.Lpv_6row_d_tile_loop:
    // Zero 24 accumulators
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

    mov     x5, x24                     // P base
    mov     x6, x21                     // V_t_int base
    mov     x7, #16                     // 16 N_groups

.Lpv_6row_n_group_loop:
    // Load V[0-3] into z24-z27
    ld1b    z24.b, p0/z, [x6]
    add     x8, x6, #64
    ld1b    z25.b, p0/z, [x8]
    add     x8, x8, #64
    ld1b    z26.b, p0/z, [x8]
    add     x8, x8, #64
    ld1b    z27.b, p0/z, [x8]
    add     x6, x6, x14

    // Load P[0-3] into z28-z31 (P row stride = 64)
    mov     x8, x5
    ld1rw   {z28.s}, p1/z, [x8]
    add     x8, x8, #64
    ld1rw   {z29.s}, p1/z, [x8]
    add     x8, x8, #64
    ld1rw   {z30.s}, p1/z, [x8]
    add     x8, x8, #64
    ld1rw   {z31.s}, p1/z, [x8]

    // Rows 0-3: 16 SDOTs
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

    // Load P[4-5] into z28-z29
    add     x8, x5, #256                // P + 4*64 = row 4
    ld1rw   {z28.s}, p1/z, [x8]
    add     x8, x8, #64                 // row 5
    ld1rw   {z29.s}, p1/z, [x8]

    // Rows 4-5: 8 SDOTs
    sdot    z16.s, z24.b, z28.b
    sdot    z17.s, z25.b, z28.b
    sdot    z18.s, z26.b, z28.b
    sdot    z19.s, z27.b, z28.b

    sdot    z20.s, z24.b, z29.b
    sdot    z21.s, z25.b, z29.b
    sdot    z22.s, z26.b, z29.b
    sdot    z23.s, z27.b, z29.b

    add     x5, x5, #4                  // P advances

    subs    x7, x7, #1
    b.gt    .Lpv_6row_n_group_loop

    // Store O[6, 64] with accumulate
    mov     x8, x22
    ld1w    z30.s, p1/z, [x8]
    add     z0.s, z0.s, z30.s
    st1w    z0.s, p1, [x8]
    add     x8, x8, #64
    ld1w    z30.s, p1/z, [x8]
    add     z1.s, z1.s, z30.s
    st1w    z1.s, p1, [x8]
    add     x8, x8, #64
    ld1w    z30.s, p1/z, [x8]
    add     z2.s, z2.s, z30.s
    st1w    z2.s, p1, [x8]
    add     x8, x8, #64
    ld1w    z30.s, p1/z, [x8]
    add     z3.s, z3.s, z30.s
    st1w    z3.s, p1, [x8]

    add     x8, x22, x26                // +2048 (row 1)
    ld1w    z30.s, p1/z, [x8]
    add     z4.s, z4.s, z30.s
    st1w    z4.s, p1, [x8]
    add     x8, x8, #64
    ld1w    z30.s, p1/z, [x8]
    add     z5.s, z5.s, z30.s
    st1w    z5.s, p1, [x8]
    add     x8, x8, #64
    ld1w    z30.s, p1/z, [x8]
    add     z6.s, z6.s, z30.s
    st1w    z6.s, p1, [x8]
    add     x8, x8, #64
    ld1w    z30.s, p1/z, [x8]
    add     z7.s, z7.s, z30.s
    st1w    z7.s, p1, [x8]

    add     x8, x22, x26, lsl #1        // +4096 (row 2)
    ld1w    z30.s, p1/z, [x8]
    add     z8.s, z8.s, z30.s
    st1w    z8.s, p1, [x8]
    add     x8, x8, #64
    ld1w    z30.s, p1/z, [x8]
    add     z9.s, z9.s, z30.s
    st1w    z9.s, p1, [x8]
    add     x8, x8, #64
    ld1w    z30.s, p1/z, [x8]
    add     z10.s, z10.s, z30.s
    st1w    z10.s, p1, [x8]
    add     x8, x8, #64
    ld1w    z30.s, p1/z, [x8]
    add     z11.s, z11.s, z30.s
    st1w    z11.s, p1, [x8]

    add     x10, x26, x26, lsl #1       // 3 * 2048 = 6144
    add     x8, x22, x10                // row 3
    ld1w    z30.s, p1/z, [x8]
    add     z12.s, z12.s, z30.s
    st1w    z12.s, p1, [x8]
    add     x8, x8, #64
    ld1w    z30.s, p1/z, [x8]
    add     z13.s, z13.s, z30.s
    st1w    z13.s, p1, [x8]
    add     x8, x8, #64
    ld1w    z30.s, p1/z, [x8]
    add     z14.s, z14.s, z30.s
    st1w    z14.s, p1, [x8]
    add     x8, x8, #64
    ld1w    z30.s, p1/z, [x8]
    add     z15.s, z15.s, z30.s
    st1w    z15.s, p1, [x8]

    add     x8, x22, x26, lsl #2        // 4 * 2048 = 8192 (row 4)
    ld1w    z30.s, p1/z, [x8]
    add     z16.s, z16.s, z30.s
    st1w    z16.s, p1, [x8]
    add     x8, x8, #64
    ld1w    z30.s, p1/z, [x8]
    add     z17.s, z17.s, z30.s
    st1w    z17.s, p1, [x8]
    add     x8, x8, #64
    ld1w    z30.s, p1/z, [x8]
    add     z18.s, z18.s, z30.s
    st1w    z18.s, p1, [x8]
    add     x8, x8, #64
    ld1w    z30.s, p1/z, [x8]
    add     z19.s, z19.s, z30.s
    st1w    z19.s, p1, [x8]

    add     x10, x26, x26, lsl #2       // 5 * 2048 = 10240
    add     x8, x22, x10                // row 5
    ld1w    z30.s, p1/z, [x8]
    add     z20.s, z20.s, z30.s
    st1w    z20.s, p1, [x8]
    add     x8, x8, #64
    ld1w    z30.s, p1/z, [x8]
    add     z21.s, z21.s, z30.s
    st1w    z21.s, p1, [x8]
    add     x8, x8, #64
    ld1w    z30.s, p1/z, [x8]
    add     z22.s, z22.s, z30.s
    st1w    z22.s, p1, [x8]
    add     x8, x8, #64
    ld1w    z30.s, p1/z, [x8]
    add     z23.s, z23.s, z30.s
    st1w    z23.s, p1, [x8]

    // Next D-tile
    add     x22, x22, #256
    add     x21, x21, x15

    subs    x9, x9, #1
    b.gt    .Lpv_6row_d_tile_loop

    // Cleanup
    add     sp, sp, #384
    ldp     d12, d13, [sp, #112]
    ldp     d10, d11, [sp, #96]
    ldp     d8, d9, [sp, #80]
    ldp     x27, x28, [sp, #64]
    ldp     x25, x26, [sp, #48]
    ldp     x23, x24, [sp, #32]
    ldp     x21, x22, [sp, #16]
    ldp     x19, x20, [sp], #128
    ret

    .size kernel_fused_d512_6row, .-kernel_fused_d512_6row
