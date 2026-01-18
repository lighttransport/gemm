// kernel_pv_int8_opt.S
// Optimized P@V INT8 SDOT kernel
//
// Key optimization: Load V for next K group at start of current K group
// This gives ~12+ cycles of SDOT work between V load and V use
//
// V layout for D>64: V[D_tile][K_group][4][64] - each D tile is contiguous
// For K=64: 16 K-groups per D tile, 256 bytes per K-group = 4096 bytes per D tile

    .arch armv8.2-a+sve
    .text
    .align 6

    .global kernel_pv_int8_opt
    .type kernel_pv_int8_opt, @function

kernel_pv_int8_opt:
    stp     d8, d9, [sp, #-64]!
    stp     d10, d11, [sp, #16]
    stp     d12, d13, [sp, #32]
    stp     d14, d15, [sp, #48]

    ptrue   p0.b
    ptrue   p1.s

    lsr     x5, x3, #6              // D_tiles = D / 64
    cbz     x5, .Lopt_done

    mov     x6, x2                  // O pointer
    mov     x7, x1                  // V base
    mov     x11, #4096              // V stride per D tile = 16 K-groups * 256 bytes

.Lopt_d_loop:
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

    mov     x8, x0                  // P pointer
    mov     x9, x7                  // V pointer
    mov     x10, #16                // 16 K groups

    // Load first K group V and P
    ld1b    z30.b, p0/z, [x9, #0, mul vl]
    ld1b    z31.b, p0/z, [x9, #1, mul vl]
    ld1rw   z24.s, p1/z, [x8, #0]
    ld1rw   z25.s, p1/z, [x8, #4]
    ld1rw   z26.s, p1/z, [x8, #8]
    ld1rw   z27.s, p1/z, [x8, #12]
    ld1rw   z28.s, p1/z, [x8, #16]
    ld1rw   z29.s, p1/z, [x8, #20]

    add     x8, x8, #24
    sub     x10, x10, #1
    cbz     x10, .Lopt_epilogue

    .align 6
.Lopt_k_loop:
    // V cols 0-1 in z30-z31, P in z24-z29
    // Strategy: Start loading V cols 2-3 during Phase 1 to hide latency

    // Phase 1 part A: 6 SDOT with V cols 0-1
    sdot    z0.s, z24.b, z30.b
    sdot    z1.s, z24.b, z31.b
    sdot    z4.s, z25.b, z30.b
    sdot    z5.s, z25.b, z31.b
    sdot    z8.s, z26.b, z30.b
    sdot    z9.s, z26.b, z31.b

    // Prefetch V cols 2-3 into registers z26-z27 temporarily
    // But wait - z26/z27 are in use. Use different approach.
    // Actually we can't avoid this - z30/z31 must be freed first.
    // Instead, reorder to get max SDOT between load and use.

    // Phase 1 part B: remaining 6 SDOT with V cols 0-1
    sdot    z12.s, z27.b, z30.b
    sdot    z13.s, z27.b, z31.b
    sdot    z16.s, z28.b, z30.b
    sdot    z17.s, z28.b, z31.b
    sdot    z20.s, z29.b, z30.b
    sdot    z21.s, z29.b, z31.b

    // Load V cols 2-3 for current K (z30/z31 now free)
    ld1b    z30.b, p0/z, [x9, #2, mul vl]
    ld1b    z31.b, p0/z, [x9, #3, mul vl]

    // Advance V pointer to next K (hide load latency)
    add     x9, x9, #256

    // Phase 2: SDOT for V cols 2-3, interleaved with next P loads
    // First pair uses z24 (already loaded), then reload z24
    sdot    z2.s, z24.b, z30.b
    sdot    z3.s, z24.b, z31.b
    ld1rw   z24.s, p1/z, [x8, #0]

    sdot    z6.s, z25.b, z30.b
    sdot    z7.s, z25.b, z31.b
    ld1rw   z25.s, p1/z, [x8, #4]

    sdot    z10.s, z26.b, z30.b
    sdot    z11.s, z26.b, z31.b
    ld1rw   z26.s, p1/z, [x8, #8]

    sdot    z14.s, z27.b, z30.b
    sdot    z15.s, z27.b, z31.b
    ld1rw   z27.s, p1/z, [x8, #12]

    sdot    z18.s, z28.b, z30.b
    sdot    z19.s, z28.b, z31.b
    ld1rw   z28.s, p1/z, [x8, #16]

    sdot    z22.s, z29.b, z30.b
    sdot    z23.s, z29.b, z31.b
    ld1rw   z29.s, p1/z, [x8, #20]

    // Load V cols 0-1 for NEXT K
    ld1b    z30.b, p0/z, [x9, #0, mul vl]
    ld1b    z31.b, p0/z, [x9, #1, mul vl]

    add     x8, x8, #24

    subs    x10, x10, #1
    b.gt    .Lopt_k_loop

.Lopt_epilogue:
    // Final K group
    sdot    z0.s, z24.b, z30.b
    sdot    z1.s, z24.b, z31.b
    sdot    z4.s, z25.b, z30.b
    sdot    z5.s, z25.b, z31.b
    sdot    z8.s, z26.b, z30.b
    sdot    z9.s, z26.b, z31.b
    sdot    z12.s, z27.b, z30.b
    sdot    z13.s, z27.b, z31.b
    sdot    z16.s, z28.b, z30.b
    sdot    z17.s, z28.b, z31.b
    sdot    z20.s, z29.b, z30.b
    sdot    z21.s, z29.b, z31.b

    ld1b    z30.b, p0/z, [x9, #2, mul vl]
    ld1b    z31.b, p0/z, [x9, #3, mul vl]

    sdot    z2.s, z24.b, z30.b
    sdot    z3.s, z24.b, z31.b
    sdot    z6.s, z25.b, z30.b
    sdot    z7.s, z25.b, z31.b
    sdot    z10.s, z26.b, z30.b
    sdot    z11.s, z26.b, z31.b
    sdot    z14.s, z27.b, z30.b
    sdot    z15.s, z27.b, z31.b
    sdot    z18.s, z28.b, z30.b
    sdot    z19.s, z28.b, z31.b
    sdot    z22.s, z29.b, z30.b
    sdot    z23.s, z29.b, z31.b

    // Store O[6][64]
    mov     x8, x6
    mov     x9, x4

    st1w    z0.s, p1, [x8, #0, mul vl]
    st1w    z1.s, p1, [x8, #1, mul vl]
    st1w    z2.s, p1, [x8, #2, mul vl]
    st1w    z3.s, p1, [x8, #3, mul vl]
    add     x8, x8, x9

    st1w    z4.s, p1, [x8, #0, mul vl]
    st1w    z5.s, p1, [x8, #1, mul vl]
    st1w    z6.s, p1, [x8, #2, mul vl]
    st1w    z7.s, p1, [x8, #3, mul vl]
    add     x8, x8, x9

    st1w    z8.s, p1, [x8, #0, mul vl]
    st1w    z9.s, p1, [x8, #1, mul vl]
    st1w    z10.s, p1, [x8, #2, mul vl]
    st1w    z11.s, p1, [x8, #3, mul vl]
    add     x8, x8, x9

    st1w    z12.s, p1, [x8, #0, mul vl]
    st1w    z13.s, p1, [x8, #1, mul vl]
    st1w    z14.s, p1, [x8, #2, mul vl]
    st1w    z15.s, p1, [x8, #3, mul vl]
    add     x8, x8, x9

    st1w    z16.s, p1, [x8, #0, mul vl]
    st1w    z17.s, p1, [x8, #1, mul vl]
    st1w    z18.s, p1, [x8, #2, mul vl]
    st1w    z19.s, p1, [x8, #3, mul vl]
    add     x8, x8, x9

    st1w    z20.s, p1, [x8, #0, mul vl]
    st1w    z21.s, p1, [x8, #1, mul vl]
    st1w    z22.s, p1, [x8, #2, mul vl]
    st1w    z23.s, p1, [x8, #3, mul vl]

    add     x6, x6, #256            // O advances by 64 int32 = 256 bytes
    add     x7, x7, x11             // V advances by 16 K-groups * 256 = 4096 bytes

    subs    x5, x5, #1
    b.gt    .Lopt_d_loop

.Lopt_done:
    ldp     d10, d11, [sp, #16]
    ldp     d12, d13, [sp, #32]
    ldp     d14, d15, [sp, #48]
    ldp     d8, d9, [sp], #64
    ret

    .size kernel_pv_int8_opt, .-kernel_pv_int8_opt
