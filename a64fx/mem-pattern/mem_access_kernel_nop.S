// mem_access_kernel_nop.S
// Memory-access-only fused GEMM kernel for Flash Attention pattern analysis
//
// SDOT instructions replaced with NOPs to isolate load/store pipeline behavior.
// This allows analyzing cache hit/miss patterns without compute interference.
//
// Original: kernel_fused_d256_4row.S
// Processes:
//   1. Q@K^T: [4, 256] x [256, 64] -> S[4, 64] int32
//   2. Quantize: S[4, 64] int32 -> P[4, 64] int8
//   3. P@V: [4, 64] x [64, 256] -> O[4, 256] int32
//
// Memory traffic analysis:
//   Phase 1 (Q@K^T): 64 iters x (256 + 16) bytes = 17,408 bytes loaded
//   Phase 2 (Quant): 256 bytes stored
//   Phase 3 (P@V):   4 D-tiles x 16 N_groups x 272 bytes = 17,408 loaded
//                    4 D-tiles x 1024 bytes = 4,096 stored
//   Total: ~34,816 bytes loaded, ~4,352 bytes stored

    .arch armv8.2-a+sve
    .text
    .align 6

    .global mem_access_kernel_nop
    .type mem_access_kernel_nop, @function

// Arguments:
//   x0 = Q pointer [4, 256] row-major
//   x1 = K_int pointer [64, 64, 4] interleaved
//   x2 = V_t_int pointer [16, 256, 4] interleaved
//   x3 = O output pointer [4, 256] int32

mem_access_kernel_nop:
    stp     x19, x20, [sp, #-96]!
    stp     x21, x22, [sp, #16]
    stp     x23, x24, [sp, #32]
    stp     d8, d9, [sp, #48]
    stp     d10, d11, [sp, #64]
    stp     d12, d13, [sp, #80]

    ptrue   p0.b
    ptrue   p1.s

    // Save arguments
    mov     x19, x0                     // Q
    mov     x20, x1                     // K_int
    mov     x21, x2                     // V_t_int
    mov     x22, x3                     // O

    // Allocate stack space for P[4, 64] = 256 bytes
    sub     sp, sp, #256
    mov     x24, sp                     // P buffer

    // K_int stride per d_group: 64 * 4 = 256
    mov     x14, #256
    // V_t_int stride per N_group: 256 * 4 = 1024
    mov     x15, #1024

    // ========================================================================
    // PHASE 1: Q @ K^T -> S[4, 64]
    // Memory access pattern: ld1b K (4x64B), ld1rw Q (4x4B), NOPs for compute
    // ========================================================================

    // Zero accumulators (kept for register pressure simulation)
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

    mov     x5, x19                     // Q base
    mov     x6, x20                     // K_int base

    // Loop over 64 d_groups
    mov     x7, #64

.Lqkt_d_group_loop:
    // Load K for all 4 N-tiles: 4 x 64 bytes = 256 bytes
    ld1b    z16.b, p0/z, [x6]           // K[d, 0:16]
    add     x8, x6, #64
    ld1b    z17.b, p0/z, [x8]           // K[d, 16:32]
    add     x8, x8, #64
    ld1b    z18.b, p0/z, [x8]           // K[d, 32:48]
    add     x8, x8, #64
    ld1b    z19.b, p0/z, [x8]           // K[d, 48:64]
    add     x6, x6, x14                 // Next d_group

    // Load Q broadcasts for all 4 rows: 4 x 4 bytes = 16 bytes
    mov     x8, x5
    ld1rw   {z20.s}, p1/z, [x8]
    add     x8, x8, #256                // Q row stride = D = 256
    ld1rw   {z21.s}, p1/z, [x8]
    add     x8, x8, #256
    ld1rw   {z22.s}, p1/z, [x8]
    add     x8, x8, #256
    ld1rw   {z23.s}, p1/z, [x8]

    // *** SDOT replaced with NOP ***
    // 24 NOPs to hide L1 latency (11 cycles) + margin
    // A64FX can issue 4 instructions/cycle, so 24 NOPs = 6 cycles
    // Combined with load latency overlap, this simulates compute hiding
    // Row 0 x all N-tiles (6 NOP)
    nop
    nop
    nop
    nop
    nop
    nop

    // Row 1 x all N-tiles (6 NOP)
    nop
    nop
    nop
    nop
    nop
    nop

    // Row 2 x all N-tiles (6 NOP)
    nop
    nop
    nop
    nop
    nop
    nop

    // Row 3 x all N-tiles (6 NOP)
    nop
    nop
    nop
    nop
    nop
    nop

    add     x5, x5, #4                  // Q advances by 4 bytes

    subs    x7, x7, #1
    b.gt    .Lqkt_d_group_loop

    // ========================================================================
    // PHASE 2: Quantize S[4, 64] int32 -> P[4, 64] int8
    // Keep quantize logic (it's small overhead and needed for realistic P)
    // ========================================================================

    // Prepare constants
    dup     z24.s, #0                   // min = 0
    dup     z25.s, #127                 // max = 127

    // Quantize and store row 0
    asr     z16.s, z0.s, #8
    asr     z17.s, z1.s, #8
    asr     z18.s, z2.s, #8
    asr     z19.s, z3.s, #8
    smax    z16.s, p1/m, z16.s, z24.s
    smax    z17.s, p1/m, z17.s, z24.s
    smax    z18.s, p1/m, z18.s, z24.s
    smax    z19.s, p1/m, z19.s, z24.s
    smin    z16.s, p1/m, z16.s, z25.s
    smin    z17.s, p1/m, z17.s, z25.s
    smin    z18.s, p1/m, z18.s, z25.s
    smin    z19.s, p1/m, z19.s, z25.s

    // Narrow to int8 and store (row 0)
    mov     x8, x24
    uzp1    z26.h, z16.h, z17.h
    uzp1    z27.h, z18.h, z19.h
    uzp1    z28.b, z26.b, z27.b
    st1b    z28.b, p0, [x8]             // Store 64 bytes

    // Quantize and store row 1
    asr     z16.s, z4.s, #8
    asr     z17.s, z5.s, #8
    asr     z18.s, z6.s, #8
    asr     z19.s, z7.s, #8
    smax    z16.s, p1/m, z16.s, z24.s
    smax    z17.s, p1/m, z17.s, z24.s
    smax    z18.s, p1/m, z18.s, z24.s
    smax    z19.s, p1/m, z19.s, z24.s
    smin    z16.s, p1/m, z16.s, z25.s
    smin    z17.s, p1/m, z17.s, z25.s
    smin    z18.s, p1/m, z18.s, z25.s
    smin    z19.s, p1/m, z19.s, z25.s

    add     x8, x24, #64
    uzp1    z26.h, z16.h, z17.h
    uzp1    z27.h, z18.h, z19.h
    uzp1    z28.b, z26.b, z27.b
    st1b    z28.b, p0, [x8]

    // Quantize and store row 2
    asr     z16.s, z8.s, #8
    asr     z17.s, z9.s, #8
    asr     z18.s, z10.s, #8
    asr     z19.s, z11.s, #8
    smax    z16.s, p1/m, z16.s, z24.s
    smax    z17.s, p1/m, z17.s, z24.s
    smax    z18.s, p1/m, z18.s, z24.s
    smax    z19.s, p1/m, z19.s, z24.s
    smin    z16.s, p1/m, z16.s, z25.s
    smin    z17.s, p1/m, z17.s, z25.s
    smin    z18.s, p1/m, z18.s, z25.s
    smin    z19.s, p1/m, z19.s, z25.s

    add     x8, x24, #128
    uzp1    z26.h, z16.h, z17.h
    uzp1    z27.h, z18.h, z19.h
    uzp1    z28.b, z26.b, z27.b
    st1b    z28.b, p0, [x8]

    // Quantize and store row 3
    asr     z16.s, z12.s, #8
    asr     z17.s, z13.s, #8
    asr     z18.s, z14.s, #8
    asr     z19.s, z15.s, #8
    smax    z16.s, p1/m, z16.s, z24.s
    smax    z17.s, p1/m, z17.s, z24.s
    smax    z18.s, p1/m, z18.s, z24.s
    smax    z19.s, p1/m, z19.s, z24.s
    smin    z16.s, p1/m, z16.s, z25.s
    smin    z17.s, p1/m, z17.s, z25.s
    smin    z18.s, p1/m, z18.s, z25.s
    smin    z19.s, p1/m, z19.s, z25.s

    add     x8, x24, #192
    uzp1    z26.h, z16.h, z17.h
    uzp1    z27.h, z18.h, z19.h
    uzp1    z28.b, z26.b, z27.b
    st1b    z28.b, p0, [x8]

    // ========================================================================
    // PHASE 3: P @ V -> O[4, 256]
    // Memory access pattern: ld1b V (4x64B), ld1rw P (4x4B), NOPs, st1w O
    // ========================================================================

    mov     x9, #4                      // 4 D-tile groups

.Lpv_d_group_loop:
    // Zero accumulators for O[4, 64]
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

    mov     x5, x24                     // P base
    mov     x6, x21                     // V_t_int base

    // Loop over 16 N_groups
    mov     x7, #16

.Lpv_n_group_loop:
    // Load V for 4 D-tiles: 4 x 64 bytes = 256 bytes
    ld1b    z16.b, p0/z, [x6]
    add     x8, x6, #64
    ld1b    z17.b, p0/z, [x8]
    add     x8, x8, #64
    ld1b    z18.b, p0/z, [x8]
    add     x8, x8, #64
    ld1b    z19.b, p0/z, [x8]
    add     x6, x6, x15                 // Next N_group

    // Load P broadcasts for all 4 rows: 4 x 4 bytes = 16 bytes
    mov     x8, x5
    ld1rw   {z20.s}, p1/z, [x8]
    add     x8, x8, #64                 // P row stride = 64
    ld1rw   {z21.s}, p1/z, [x8]
    add     x8, x8, #64
    ld1rw   {z22.s}, p1/z, [x8]
    add     x8, x8, #64
    ld1rw   {z23.s}, p1/z, [x8]

    // *** SDOT replaced with NOP ***
    // 24 NOPs to hide L1 latency (11 cycles) + margin
    // Row 0 x all D-tiles (6 NOP)
    nop
    nop
    nop
    nop
    nop
    nop

    // Row 1 x all D-tiles (6 NOP)
    nop
    nop
    nop
    nop
    nop
    nop

    // Row 2 x all D-tiles (6 NOP)
    nop
    nop
    nop
    nop
    nop
    nop

    // Row 3 x all D-tiles (6 NOP)
    nop
    nop
    nop
    nop
    nop
    nop

    add     x5, x5, #4                  // P advances

    subs    x7, x7, #1
    b.gt    .Lpv_n_group_loop

    // Store O[4, 64]: 16 x 64 bytes = 1024 bytes
    mov     x8, x22
    st1w    z0.s, p1, [x8]
    add     x8, x8, #64
    st1w    z1.s, p1, [x8]
    add     x8, x8, #64
    st1w    z2.s, p1, [x8]
    add     x8, x8, #64
    st1w    z3.s, p1, [x8]

    mov     x10, #1024                  // O row stride
    add     x8, x22, x10
    st1w    z4.s, p1, [x8]
    add     x8, x8, #64
    st1w    z5.s, p1, [x8]
    add     x8, x8, #64
    st1w    z6.s, p1, [x8]
    add     x8, x8, #64
    st1w    z7.s, p1, [x8]

    add     x8, x22, x10, lsl #1
    st1w    z8.s, p1, [x8]
    add     x8, x8, #64
    st1w    z9.s, p1, [x8]
    add     x8, x8, #64
    st1w    z10.s, p1, [x8]
    add     x8, x8, #64
    st1w    z11.s, p1, [x8]

    add     x11, x10, x10, lsl #1       // 3 * stride
    add     x8, x22, x11
    st1w    z12.s, p1, [x8]
    add     x8, x8, #64
    st1w    z13.s, p1, [x8]
    add     x8, x8, #64
    st1w    z14.s, p1, [x8]
    add     x8, x8, #64
    st1w    z15.s, p1, [x8]

    // Next D-tile group
    add     x22, x22, #256              // O advances by 64 int32s
    add     x21, x21, #256              // V_t_int advances

    subs    x9, x9, #1
    b.gt    .Lpv_d_group_loop

    // Cleanup
    add     sp, sp, #256
    ldp     d12, d13, [sp, #80]
    ldp     d10, d11, [sp, #64]
    ldp     d8, d9, [sp, #48]
    ldp     x23, x24, [sp, #32]
    ldp     x21, x22, [sp, #16]
    ldp     x19, x20, [sp], #96
    ret

    .size mem_access_kernel_nop, .-mem_access_kernel_nop
