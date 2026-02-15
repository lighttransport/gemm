// kernel_fused_d512_5row.S
// Fused Q@K^T -> Quantize -> P@V for D=512 with 5 rows
//
// D=512 = 8 SVE vectors, giving more compute per K/V load
// K_int: [D_group=128, N=64, 4] - 128 groups of 4 D elements
// V_t:   [D_tile=8, N_group=16, 64, 4] - 8 D-tiles of 64 elements
//
// With 11-cycle load latency, more compute per load helps hide latency

    .arch armv8.2-a+sve
    .text
    .align 6

    .global kernel_fused_d512_5row
    .type kernel_fused_d512_5row, @function

// Arguments:
//   x0 = Q pointer [5, 512] row-major
//   x1 = K_int pointer [128, 64, 4] interleaved
//   x2 = V_t_int pointer [8, 16, 64, 4] interleaved
//   x3 = O output pointer [5, 512] int32

kernel_fused_d512_5row:
    stp     x19, x20, [sp, #-112]!
    stp     x21, x22, [sp, #16]
    stp     x23, x24, [sp, #32]
    stp     x25, x26, [sp, #48]
    stp     d8, d9, [sp, #64]
    stp     d10, d11, [sp, #80]
    stp     d12, d13, [sp, #96]

    ptrue   p0.b
    ptrue   p1.s

    mov     x19, x0                     // Q
    mov     x20, x1                     // K_int
    mov     x21, x2                     // V_t_int
    mov     x22, x3                     // O

    // P buffer: [5, 64] = 320 bytes, round to 384
    sub     sp, sp, #384
    mov     x24, sp
    mov     x14, #256                   // N_group/D_group stride

    // ========================================================================
    // PHASE 1: Q @ K^T -> S[5, 64]
    // D=512 means 128 d_groups (512/4 = 128)
    // ========================================================================

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

    mov     x5, x19                     // Q base
    mov     x6, x20                     // K_int base
    mov     x7, #128                    // 128 d_groups for D=512

.Lqkt_d512_loop:
    // Load K for 4 N-tiles
    ld1b    z20.b, p0/z, [x6]
    add     x8, x6, #64
    ld1b    z21.b, p0/z, [x8]
    add     x8, x8, #64
    ld1b    z22.b, p0/z, [x8]
    add     x8, x8, #64
    ld1b    z23.b, p0/z, [x8]
    add     x6, x6, x14

    // Load Q broadcasts for 5 rows (Q row stride = 512 for D=512)
    mov     x8, x5
    ld1rw   {z24.s}, p1/z, [x8]
    add     x8, x8, #512
    ld1rw   {z25.s}, p1/z, [x8]
    add     x8, x8, #512
    ld1rw   {z26.s}, p1/z, [x8]
    add     x8, x8, #512
    ld1rw   {z27.s}, p1/z, [x8]
    add     x8, x8, #512
    ld1rw   {z28.s}, p1/z, [x8]

    // 20 SDOTs: 5 rows × 4 N-tiles
    sdot    z0.s, z20.b, z24.b
    sdot    z1.s, z21.b, z24.b
    sdot    z2.s, z22.b, z24.b
    sdot    z3.s, z23.b, z24.b

    sdot    z4.s, z20.b, z25.b
    sdot    z5.s, z21.b, z25.b
    sdot    z6.s, z22.b, z25.b
    sdot    z7.s, z23.b, z25.b

    sdot    z8.s, z20.b, z26.b
    sdot    z9.s, z21.b, z26.b
    sdot    z10.s, z22.b, z26.b
    sdot    z11.s, z23.b, z26.b

    sdot    z12.s, z20.b, z27.b
    sdot    z13.s, z21.b, z27.b
    sdot    z14.s, z22.b, z27.b
    sdot    z15.s, z23.b, z27.b

    sdot    z16.s, z20.b, z28.b
    sdot    z17.s, z21.b, z28.b
    sdot    z18.s, z22.b, z28.b
    sdot    z19.s, z23.b, z28.b

    add     x5, x5, #4

    subs    x7, x7, #1
    b.gt    .Lqkt_d512_loop

    // ========================================================================
    // PHASE 2: Quantize S[5, 64] -> P[5, 64]
    // ========================================================================

    dup     z29.s, #0
    dup     z30.s, #127

    // Row 0
    asr     z20.s, z0.s, #8
    asr     z21.s, z1.s, #8
    asr     z22.s, z2.s, #8
    asr     z23.s, z3.s, #8
    smax    z20.s, p1/m, z20.s, z29.s
    smax    z21.s, p1/m, z21.s, z29.s
    smax    z22.s, p1/m, z22.s, z29.s
    smax    z23.s, p1/m, z23.s, z29.s
    smin    z20.s, p1/m, z20.s, z30.s
    smin    z21.s, p1/m, z21.s, z30.s
    smin    z22.s, p1/m, z22.s, z30.s
    smin    z23.s, p1/m, z23.s, z30.s
    uzp1    z24.h, z20.h, z21.h
    uzp1    z25.h, z22.h, z23.h
    uzp1    z31.b, z24.b, z25.b
    st1b    z31.b, p0, [x24]

    // Row 1
    asr     z20.s, z4.s, #8
    asr     z21.s, z5.s, #8
    asr     z22.s, z6.s, #8
    asr     z23.s, z7.s, #8
    smax    z20.s, p1/m, z20.s, z29.s
    smax    z21.s, p1/m, z21.s, z29.s
    smax    z22.s, p1/m, z22.s, z29.s
    smax    z23.s, p1/m, z23.s, z29.s
    smin    z20.s, p1/m, z20.s, z30.s
    smin    z21.s, p1/m, z21.s, z30.s
    smin    z22.s, p1/m, z22.s, z30.s
    smin    z23.s, p1/m, z23.s, z30.s
    uzp1    z24.h, z20.h, z21.h
    uzp1    z25.h, z22.h, z23.h
    uzp1    z31.b, z24.b, z25.b
    add     x8, x24, #64
    st1b    z31.b, p0, [x8]

    // Row 2
    asr     z20.s, z8.s, #8
    asr     z21.s, z9.s, #8
    asr     z22.s, z10.s, #8
    asr     z23.s, z11.s, #8
    smax    z20.s, p1/m, z20.s, z29.s
    smax    z21.s, p1/m, z21.s, z29.s
    smax    z22.s, p1/m, z22.s, z29.s
    smax    z23.s, p1/m, z23.s, z29.s
    smin    z20.s, p1/m, z20.s, z30.s
    smin    z21.s, p1/m, z21.s, z30.s
    smin    z22.s, p1/m, z22.s, z30.s
    smin    z23.s, p1/m, z23.s, z30.s
    uzp1    z24.h, z20.h, z21.h
    uzp1    z25.h, z22.h, z23.h
    uzp1    z31.b, z24.b, z25.b
    add     x8, x24, #128
    st1b    z31.b, p0, [x8]

    // Row 3
    asr     z20.s, z12.s, #8
    asr     z21.s, z13.s, #8
    asr     z22.s, z14.s, #8
    asr     z23.s, z15.s, #8
    smax    z20.s, p1/m, z20.s, z29.s
    smax    z21.s, p1/m, z21.s, z29.s
    smax    z22.s, p1/m, z22.s, z29.s
    smax    z23.s, p1/m, z23.s, z29.s
    smin    z20.s, p1/m, z20.s, z30.s
    smin    z21.s, p1/m, z21.s, z30.s
    smin    z22.s, p1/m, z22.s, z30.s
    smin    z23.s, p1/m, z23.s, z30.s
    uzp1    z24.h, z20.h, z21.h
    uzp1    z25.h, z22.h, z23.h
    uzp1    z31.b, z24.b, z25.b
    add     x8, x24, #192
    st1b    z31.b, p0, [x8]

    // Row 4
    asr     z20.s, z16.s, #8
    asr     z21.s, z17.s, #8
    asr     z22.s, z18.s, #8
    asr     z23.s, z19.s, #8
    smax    z20.s, p1/m, z20.s, z29.s
    smax    z21.s, p1/m, z21.s, z29.s
    smax    z22.s, p1/m, z22.s, z29.s
    smax    z23.s, p1/m, z23.s, z29.s
    smin    z20.s, p1/m, z20.s, z30.s
    smin    z21.s, p1/m, z21.s, z30.s
    smin    z22.s, p1/m, z22.s, z30.s
    smin    z23.s, p1/m, z23.s, z30.s
    uzp1    z24.h, z20.h, z21.h
    uzp1    z25.h, z22.h, z23.h
    uzp1    z31.b, z24.b, z25.b
    add     x8, x24, #256
    st1b    z31.b, p0, [x8]

    // ========================================================================
    // PHASE 3: P @ V -> O[5, 512]
    // 8 D-tiles of 64 elements each
    // ========================================================================

    mov     x9, #8                      // 8 D-tiles for D=512
    mov     x15, #4096                  // D_tile stride

.Lpv_d512_d_tile_loop:
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

    mov     x5, x24                     // P base
    mov     x6, x21                     // V_t_int base
    mov     x7, #16                     // 16 N_groups

.Lpv_d512_n_group_loop:
    // Load V for 4 D-elements within this D-tile
    ld1b    z20.b, p0/z, [x6]
    add     x8, x6, #64
    ld1b    z21.b, p0/z, [x8]
    add     x8, x8, #64
    ld1b    z22.b, p0/z, [x8]
    add     x8, x8, #64
    ld1b    z23.b, p0/z, [x8]
    add     x6, x6, x14

    // Load P broadcasts for 5 rows
    mov     x8, x5
    ld1rw   {z24.s}, p1/z, [x8]
    add     x8, x8, #64
    ld1rw   {z25.s}, p1/z, [x8]
    add     x8, x8, #64
    ld1rw   {z26.s}, p1/z, [x8]
    add     x8, x8, #64
    ld1rw   {z27.s}, p1/z, [x8]
    add     x8, x8, #64
    ld1rw   {z28.s}, p1/z, [x8]

    // 20 SDOTs
    sdot    z0.s, z20.b, z24.b
    sdot    z1.s, z21.b, z24.b
    sdot    z2.s, z22.b, z24.b
    sdot    z3.s, z23.b, z24.b

    sdot    z4.s, z20.b, z25.b
    sdot    z5.s, z21.b, z25.b
    sdot    z6.s, z22.b, z25.b
    sdot    z7.s, z23.b, z25.b

    sdot    z8.s, z20.b, z26.b
    sdot    z9.s, z21.b, z26.b
    sdot    z10.s, z22.b, z26.b
    sdot    z11.s, z23.b, z26.b

    sdot    z12.s, z20.b, z27.b
    sdot    z13.s, z21.b, z27.b
    sdot    z14.s, z22.b, z27.b
    sdot    z15.s, z23.b, z27.b

    sdot    z16.s, z20.b, z28.b
    sdot    z17.s, z21.b, z28.b
    sdot    z18.s, z22.b, z28.b
    sdot    z19.s, z23.b, z28.b

    add     x5, x5, #4

    subs    x7, x7, #1
    b.gt    .Lpv_d512_n_group_loop

    // Store O[5, 64] with accumulate
    // O row stride = D * 4 = 512 * 4 = 2048 bytes
    mov     x10, #2048

    mov     x8, x22
    ld1w    z29.s, p1/z, [x8]
    add     z0.s, z0.s, z29.s
    st1w    z0.s, p1, [x8]
    add     x8, x8, #64
    ld1w    z29.s, p1/z, [x8]
    add     z1.s, z1.s, z29.s
    st1w    z1.s, p1, [x8]
    add     x8, x8, #64
    ld1w    z29.s, p1/z, [x8]
    add     z2.s, z2.s, z29.s
    st1w    z2.s, p1, [x8]
    add     x8, x8, #64
    ld1w    z29.s, p1/z, [x8]
    add     z3.s, z3.s, z29.s
    st1w    z3.s, p1, [x8]

    add     x8, x22, x10
    ld1w    z29.s, p1/z, [x8]
    add     z4.s, z4.s, z29.s
    st1w    z4.s, p1, [x8]
    add     x8, x8, #64
    ld1w    z29.s, p1/z, [x8]
    add     z5.s, z5.s, z29.s
    st1w    z5.s, p1, [x8]
    add     x8, x8, #64
    ld1w    z29.s, p1/z, [x8]
    add     z6.s, z6.s, z29.s
    st1w    z6.s, p1, [x8]
    add     x8, x8, #64
    ld1w    z29.s, p1/z, [x8]
    add     z7.s, z7.s, z29.s
    st1w    z7.s, p1, [x8]

    add     x8, x22, x10, lsl #1
    ld1w    z29.s, p1/z, [x8]
    add     z8.s, z8.s, z29.s
    st1w    z8.s, p1, [x8]
    add     x8, x8, #64
    ld1w    z29.s, p1/z, [x8]
    add     z9.s, z9.s, z29.s
    st1w    z9.s, p1, [x8]
    add     x8, x8, #64
    ld1w    z29.s, p1/z, [x8]
    add     z10.s, z10.s, z29.s
    st1w    z10.s, p1, [x8]
    add     x8, x8, #64
    ld1w    z29.s, p1/z, [x8]
    add     z11.s, z11.s, z29.s
    st1w    z11.s, p1, [x8]

    add     x11, x10, x10, lsl #1       // 3 * stride
    add     x8, x22, x11
    ld1w    z29.s, p1/z, [x8]
    add     z12.s, z12.s, z29.s
    st1w    z12.s, p1, [x8]
    add     x8, x8, #64
    ld1w    z29.s, p1/z, [x8]
    add     z13.s, z13.s, z29.s
    st1w    z13.s, p1, [x8]
    add     x8, x8, #64
    ld1w    z29.s, p1/z, [x8]
    add     z14.s, z14.s, z29.s
    st1w    z14.s, p1, [x8]
    add     x8, x8, #64
    ld1w    z29.s, p1/z, [x8]
    add     z15.s, z15.s, z29.s
    st1w    z15.s, p1, [x8]

    add     x8, x22, x10, lsl #2        // 4 * stride
    ld1w    z29.s, p1/z, [x8]
    add     z16.s, z16.s, z29.s
    st1w    z16.s, p1, [x8]
    add     x8, x8, #64
    ld1w    z29.s, p1/z, [x8]
    add     z17.s, z17.s, z29.s
    st1w    z17.s, p1, [x8]
    add     x8, x8, #64
    ld1w    z29.s, p1/z, [x8]
    add     z18.s, z18.s, z29.s
    st1w    z18.s, p1, [x8]
    add     x8, x8, #64
    ld1w    z29.s, p1/z, [x8]
    add     z19.s, z19.s, z29.s
    st1w    z19.s, p1, [x8]

    // Next D-tile
    add     x22, x22, #256              // O advances by 64 int32s
    add     x21, x21, x15               // V advances by D_tile stride

    subs    x9, x9, #1
    b.gt    .Lpv_d512_d_tile_loop

    // Cleanup
    add     sp, sp, #384
    ldp     d12, d13, [sp, #96]
    ldp     d10, d11, [sp, #80]
    ldp     d8, d9, [sp, #64]
    ldp     x25, x26, [sp, #48]
    ldp     x23, x24, [sp, #32]
    ldp     x21, x22, [sp, #16]
    ldp     x19, x20, [sp], #112
    ret

    .size kernel_fused_d512_5row, .-kernel_fused_d512_5row
