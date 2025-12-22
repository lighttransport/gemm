// gemm_raw.S
// Raw GEMM kernels for Flash Attention without softmax/exp
// Used to measure pure matrix multiply performance
//
// Two versions:
// 1. Fused: O = (Q @ K^T) @ V  - single pass, no intermediate storage
// 2. Two-pass: S = Q @ K^T, then O = S @ V
//
// Dimensions:
//   Q[BR, D]   = [4, 64]
//   K[BC, D]   = [64, 64]  (K^T is [D, BC] = [64, 64])
//   V[BC, D]   = [64, 64]
//   S[BR, BC]  = [4, 64]   (intermediate, only for 2-pass)
//   O[BR, D]   = [4, 64]

    .arch armv8.2-a+sve
    .text

// ============================================
// FADDV-Free Version: Pass 1 - S = Q @ K^T
// ============================================
// Uses transposed K (Kt[D, BC] = [64, 64]) to avoid horizontal reductions
//
// Key insight: Instead of computing dot products requiring faddv,
// we accumulate S[4, 64] directly using outer-product style:
//
//   for k in 0..D:
//       S[:, :] += Q[:, k] * Kt[k, :]
//
// This means:
//   - S[4, 64] kept in registers (4 rows * 4 vectors = 16 regs)
//   - Q[:, k] is 4 scalars → broadcast to 4 vectors
//   - Kt[k, :] is 64 floats = 4 vectors
//   - Pure FMLA, no faddv needed!
//
// Args: x0=Q, x1=Kt (transposed K), x2=S
// Q[4, 64], Kt[64, 64], S[4, 64]
//
// Register allocation:
//   z0-z3:   S[0, 0:64]
//   z4-z7:   S[1, 0:64]
//   z8-z11:  S[2, 0:64]
//   z12-z15: S[3, 0:64]
//   z16-z19: Kt[k, 0:64]
//   z20-z23: Q[:,k] broadcasts

    .align 6
    .global gemm_qkt_nofaddv
    .type gemm_qkt_nofaddv, %function

gemm_qkt_nofaddv:
    ptrue   p0.s

    mov     x4, #256            // row stride = 64 * 4 bytes

    // ========================================
    // Initialize S[4, 64] = 0
    // ========================================
    fmov    z0.s, #0.0
    fmov    z1.s, #0.0
    fmov    z2.s, #0.0
    fmov    z3.s, #0.0
    fmov    z4.s, #0.0
    fmov    z5.s, #0.0
    fmov    z6.s, #0.0
    fmov    z7.s, #0.0
    fmov    z8.s, #0.0
    fmov    z9.s, #0.0
    fmov    z10.s, #0.0
    fmov    z11.s, #0.0
    fmov    z12.s, #0.0
    fmov    z13.s, #0.0
    fmov    z14.s, #0.0
    fmov    z15.s, #0.0

    // Q row pointers
    mov     x5, x0              // Q[0]
    add     x6, x0, x4          // Q[1]
    add     x7, x6, x4          // Q[2]
    add     x8, x7, x4          // Q[3]

    // Kt pointer
    mov     x9, x1

    // Loop over k = 0..63
    mov     x10, #64

.Lqkt_nofaddv_loop:
    // Load Kt[k, 0:64]
    ld1w    z16.s, p0/z, [x9]
    ld1w    z17.s, p0/z, [x9, #1, mul vl]
    ld1w    z18.s, p0/z, [x9, #2, mul vl]
    ld1w    z19.s, p0/z, [x9, #3, mul vl]
    add     x9, x9, x4

    // Load and broadcast Q[0:4, k]
    ld1rw   z20.s, p0/z, [x5]   // Q[0, k] broadcast
    ld1rw   z21.s, p0/z, [x6]   // Q[1, k] broadcast
    ld1rw   z22.s, p0/z, [x7]   // Q[2, k] broadcast
    ld1rw   z23.s, p0/z, [x8]   // Q[3, k] broadcast

    // Advance Q pointers
    add     x5, x5, #4
    add     x6, x6, #4
    add     x7, x7, #4
    add     x8, x8, #4

    // S[0, :] += Q[0, k] * Kt[k, :]
    fmla    z0.s, p0/m, z20.s, z16.s
    fmla    z1.s, p0/m, z20.s, z17.s
    fmla    z2.s, p0/m, z20.s, z18.s
    fmla    z3.s, p0/m, z20.s, z19.s

    // S[1, :] += Q[1, k] * Kt[k, :]
    fmla    z4.s, p0/m, z21.s, z16.s
    fmla    z5.s, p0/m, z21.s, z17.s
    fmla    z6.s, p0/m, z21.s, z18.s
    fmla    z7.s, p0/m, z21.s, z19.s

    // S[2, :] += Q[2, k] * Kt[k, :]
    fmla    z8.s, p0/m, z22.s, z16.s
    fmla    z9.s, p0/m, z22.s, z17.s
    fmla    z10.s, p0/m, z22.s, z18.s
    fmla    z11.s, p0/m, z22.s, z19.s

    // S[3, :] += Q[3, k] * Kt[k, :]
    fmla    z12.s, p0/m, z23.s, z16.s
    fmla    z13.s, p0/m, z23.s, z17.s
    fmla    z14.s, p0/m, z23.s, z18.s
    fmla    z15.s, p0/m, z23.s, z19.s

    subs    x10, x10, #1
    b.ne    .Lqkt_nofaddv_loop

    // ========================================
    // Store S[4, 64]
    // ========================================
    mov     x11, x2
    st1w    z0.s, p0, [x11]
    st1w    z1.s, p0, [x11, #1, mul vl]
    st1w    z2.s, p0, [x11, #2, mul vl]
    st1w    z3.s, p0, [x11, #3, mul vl]
    add     x11, x11, x4

    st1w    z4.s, p0, [x11]
    st1w    z5.s, p0, [x11, #1, mul vl]
    st1w    z6.s, p0, [x11, #2, mul vl]
    st1w    z7.s, p0, [x11, #3, mul vl]
    add     x11, x11, x4

    st1w    z8.s, p0, [x11]
    st1w    z9.s, p0, [x11, #1, mul vl]
    st1w    z10.s, p0, [x11, #2, mul vl]
    st1w    z11.s, p0, [x11, #3, mul vl]
    add     x11, x11, x4

    st1w    z12.s, p0, [x11]
    st1w    z13.s, p0, [x11, #1, mul vl]
    st1w    z14.s, p0, [x11, #2, mul vl]
    st1w    z15.s, p0, [x11, #3, mul vl]

    ret
    .size gemm_qkt_nofaddv, .-gemm_qkt_nofaddv


// ============================================
// Two-Pass Version: Pass 1 - S = Q @ K^T
// ============================================
// This is essentially the same as pass1_qkt_rowmax but without rowmax
//
// Args: x0=Q, x1=K, x2=S
// Q[4, 64], K[64, 64], S[4, 64]
//
// Register allocation:
//   z0-z3:   Q[0, 0:64]
//   z4-z7:   Q[1, 0:64]
//   z8-z11:  Q[2, 0:64]
//   z12-z15: Q[3, 0:64]
//   z16-z19: K[j, 0:64] current K row
//   z20-z23: dot product accumulators

    .align 6
    .global gemm_qkt_2pass
    .type gemm_qkt_2pass, %function

gemm_qkt_2pass:
    ptrue   p0.s

    mov     x4, #256            // row stride = 64 * 4

    // ========================================
    // Load Q[4, 64] into z0-z15
    // ========================================
    mov     x10, x0
    ld1w    z0.s, p0/z, [x10]
    ld1w    z1.s, p0/z, [x10, #1, mul vl]
    ld1w    z2.s, p0/z, [x10, #2, mul vl]
    ld1w    z3.s, p0/z, [x10, #3, mul vl]
    add     x10, x10, x4

    ld1w    z4.s, p0/z, [x10]
    ld1w    z5.s, p0/z, [x10, #1, mul vl]
    ld1w    z6.s, p0/z, [x10, #2, mul vl]
    ld1w    z7.s, p0/z, [x10, #3, mul vl]
    add     x10, x10, x4

    ld1w    z8.s, p0/z, [x10]
    ld1w    z9.s, p0/z, [x10, #1, mul vl]
    ld1w    z10.s, p0/z, [x10, #2, mul vl]
    ld1w    z11.s, p0/z, [x10, #3, mul vl]
    add     x10, x10, x4

    ld1w    z12.s, p0/z, [x10]
    ld1w    z13.s, p0/z, [x10, #1, mul vl]
    ld1w    z14.s, p0/z, [x10, #2, mul vl]
    ld1w    z15.s, p0/z, [x10, #3, mul vl]

    // ========================================
    // Compute S = Q @ K^T
    // Process K rows in groups of 4
    // ========================================
    mov     x5, #16             // 64 / 4 = 16 groups
    mov     x6, x1              // K ptr
    mov     x7, x2              // S ptr

.Lqkt_group_loop:
    // Process 4 consecutive K rows

.macro compute_sj_nomax offset
    // Load K[j, 0:64]
    ld1w    z16.s, p0/z, [x6]
    ld1w    z17.s, p0/z, [x6, #1, mul vl]
    ld1w    z18.s, p0/z, [x6, #2, mul vl]
    ld1w    z19.s, p0/z, [x6, #3, mul vl]
    add     x6, x6, x4

    // S[0, j] = Q[0, :] · K[j, :]
    fmul    z20.s, z0.s, z16.s
    fmla    z20.s, p0/m, z1.s, z17.s
    fmla    z20.s, p0/m, z2.s, z18.s
    fmla    z20.s, p0/m, z3.s, z19.s

    // S[1, j]
    fmul    z21.s, z4.s, z16.s
    fmla    z21.s, p0/m, z5.s, z17.s
    fmla    z21.s, p0/m, z6.s, z18.s
    fmla    z21.s, p0/m, z7.s, z19.s

    // S[2, j]
    fmul    z22.s, z8.s, z16.s
    fmla    z22.s, p0/m, z9.s, z17.s
    fmla    z22.s, p0/m, z10.s, z18.s
    fmla    z22.s, p0/m, z11.s, z19.s

    // S[3, j]
    fmul    z23.s, z12.s, z16.s
    fmla    z23.s, p0/m, z13.s, z17.s
    fmla    z23.s, p0/m, z14.s, z18.s
    fmla    z23.s, p0/m, z15.s, z19.s

    // Horizontal reduction
    faddv   s24, p0, z20.s
    faddv   s25, p0, z21.s
    faddv   s26, p0, z22.s
    faddv   s27, p0, z23.s

    // Store S[0:4, j]
    str     s24, [x7, #(\offset * 4)]
    str     s25, [x7, #(\offset * 4 + 256)]      // S[1] row offset
    str     s26, [x7, #(\offset * 4 + 512)]      // S[2] row offset
    str     s27, [x7, #(\offset * 4 + 768)]      // S[3] row offset
.endm

    compute_sj_nomax 0
    compute_sj_nomax 1
    compute_sj_nomax 2
    compute_sj_nomax 3

    add     x7, x7, #16         // advance S ptr by 4 floats

    subs    x5, x5, #1
    b.ne    .Lqkt_group_loop

    ret
    .size gemm_qkt_2pass, .-gemm_qkt_2pass


// ============================================
// Two-Pass Version: Pass 2 - O = S @ V
// ============================================
// Args: x0=S, x1=V, x2=O
// S[4, 64], V[64, 64], O[4, 64]
//
// Strategy: Process S in chunks of 16 columns
// For each chunk, broadcast S values and accumulate with V rows
//
// Register allocation:
//   z0-z3:   O[0, 0:64]
//   z4-z7:   O[1, 0:64]
//   z8-z11:  O[2, 0:64]
//   z12-z15: O[3, 0:64]
//   z16-z19: S[0:4, jj:jj+16] (16 values per row, need 1 vector each)
//   z20-z23: V[j, 0:64]
//   z24-z27: broadcast S values

    .align 6
    .global gemm_sv_2pass
    .type gemm_sv_2pass, %function

gemm_sv_2pass:
    ptrue   p0.s

    mov     x4, #256            // row stride = 64 * 4
    mov     x5, #64             // VL bytes = 16 * 4

    // ========================================
    // Initialize O[4, 64] = 0
    // ========================================
    fmov    z0.s, #0.0
    fmov    z1.s, #0.0
    fmov    z2.s, #0.0
    fmov    z3.s, #0.0
    fmov    z4.s, #0.0
    fmov    z5.s, #0.0
    fmov    z6.s, #0.0
    fmov    z7.s, #0.0
    fmov    z8.s, #0.0
    fmov    z9.s, #0.0
    fmov    z10.s, #0.0
    fmov    z11.s, #0.0
    fmov    z12.s, #0.0
    fmov    z13.s, #0.0
    fmov    z14.s, #0.0
    fmov    z15.s, #0.0

    // S row pointers
    mov     x6, x0              // S[0]
    add     x7, x0, x4          // S[1]
    add     x8, x7, x4          // S[2]
    add     x9, x8, x4          // S[3]

    // V base
    mov     x10, x1

    // ========================================
    // Main loop: process S in chunks of 16
    // ========================================
    mov     x11, #4             // 4 chunks

.Lsv_chunk_loop:
    // Load S[0:4, jj:jj+16]
    ld1w    z16.s, p0/z, [x6]
    ld1w    z17.s, p0/z, [x7]
    ld1w    z18.s, p0/z, [x8]
    ld1w    z19.s, p0/z, [x9]

    // Advance S pointers
    add     x6, x6, x5
    add     x7, x7, x5
    add     x8, x8, x5
    add     x9, x9, x5

    // Process 16 j values
.macro process_sv_j lane
    // Broadcast S[0:4, jj+lane]
    dup     z24.s, z16.s[\lane]
    dup     z25.s, z17.s[\lane]
    dup     z26.s, z18.s[\lane]
    dup     z27.s, z19.s[\lane]

    // Load V[jj+lane, 0:64]
    ld1w    z20.s, p0/z, [x10]
    ld1w    z21.s, p0/z, [x10, #1, mul vl]
    ld1w    z22.s, p0/z, [x10, #2, mul vl]
    ld1w    z23.s, p0/z, [x10, #3, mul vl]
    add     x10, x10, x4

    // O[0, :] += S[0, j] * V[j, :]
    fmla    z0.s, p0/m, z24.s, z20.s
    fmla    z1.s, p0/m, z24.s, z21.s
    fmla    z2.s, p0/m, z24.s, z22.s
    fmla    z3.s, p0/m, z24.s, z23.s

    // O[1, :] += S[1, j] * V[j, :]
    fmla    z4.s, p0/m, z25.s, z20.s
    fmla    z5.s, p0/m, z25.s, z21.s
    fmla    z6.s, p0/m, z25.s, z22.s
    fmla    z7.s, p0/m, z25.s, z23.s

    // O[2, :] += S[2, j] * V[j, :]
    fmla    z8.s, p0/m, z26.s, z20.s
    fmla    z9.s, p0/m, z26.s, z21.s
    fmla    z10.s, p0/m, z26.s, z22.s
    fmla    z11.s, p0/m, z26.s, z23.s

    // O[3, :] += S[3, j] * V[j, :]
    fmla    z12.s, p0/m, z27.s, z20.s
    fmla    z13.s, p0/m, z27.s, z21.s
    fmla    z14.s, p0/m, z27.s, z22.s
    fmla    z15.s, p0/m, z27.s, z23.s
.endm

    process_sv_j 0
    process_sv_j 1
    process_sv_j 2
    process_sv_j 3
    process_sv_j 4
    process_sv_j 5
    process_sv_j 6
    process_sv_j 7
    process_sv_j 8
    process_sv_j 9
    process_sv_j 10
    process_sv_j 11
    process_sv_j 12
    process_sv_j 13
    process_sv_j 14
    process_sv_j 15

    subs    x11, x11, #1
    b.ne    .Lsv_chunk_loop

    // ========================================
    // Store O[4, 64]
    // ========================================
    mov     x12, x2
    st1w    z0.s, p0, [x12]
    st1w    z1.s, p0, [x12, #1, mul vl]
    st1w    z2.s, p0, [x12, #2, mul vl]
    st1w    z3.s, p0, [x12, #3, mul vl]
    add     x12, x12, x4

    st1w    z4.s, p0, [x12]
    st1w    z5.s, p0, [x12, #1, mul vl]
    st1w    z6.s, p0, [x12, #2, mul vl]
    st1w    z7.s, p0, [x12, #3, mul vl]
    add     x12, x12, x4

    st1w    z8.s, p0, [x12]
    st1w    z9.s, p0, [x12, #1, mul vl]
    st1w    z10.s, p0, [x12, #2, mul vl]
    st1w    z11.s, p0, [x12, #3, mul vl]
    add     x12, x12, x4

    st1w    z12.s, p0, [x12]
    st1w    z13.s, p0, [x12, #1, mul vl]
    st1w    z14.s, p0, [x12, #2, mul vl]
    st1w    z15.s, p0, [x12, #3, mul vl]

    ret
    .size gemm_sv_2pass, .-gemm_sv_2pass


// ============================================
// Fused Version: O = (Q @ K^T) @ V
// ============================================
// Single-pass computation without storing intermediate S
//
// Strategy:
// - Keep Q[4, 64] in registers
// - For each K row j:
//   - Compute S[0:4, j] = Q @ K[j, :] (4 scalar results)
//   - Immediately accumulate O += S[0:4, j] * V[j, :]
// - No intermediate storage needed
//
// Args: x0=Q, x1=K, x2=V, x3=O
//
// Register allocation:
//   z0-z3:   Q[0, 0:64]
//   z4-z7:   Q[1, 0:64]
//   z8-z11:  Q[2, 0:64]
//   z12-z15: Q[3, 0:64]
//   z16-z19: K[j, 0:64] / V[j, 0:64] (time-multiplexed)
//   z20-z23: O[0, 0:64]
//   z24-z27: O[1, 0:64]
//   z28-z31: Temps for S values and O[2:4]
//
// Note: We need 16 regs for O (4 rows * 4 vectors), but only have z20-z31 = 12
// Solution: Store O[2:4] to stack and reload, or process in batches

    .align 6
    .global gemm_fused
    .type gemm_fused, %function

gemm_fused:
    // Save callee-saved registers and allocate stack for O[2:4]
    stp     x19, x20, [sp, #-16]!
    stp     x21, x22, [sp, #-16]!
    sub     sp, sp, #768        // Space for O[2,64] and O[3,64] = 2 * 64 * 4 = 512
                                // Plus some extra alignment

    ptrue   p0.s

    mov     x4, #256            // row stride = 64 * 4

    // ========================================
    // Load Q[4, 64] into z0-z15
    // ========================================
    mov     x10, x0
    ld1w    z0.s, p0/z, [x10]
    ld1w    z1.s, p0/z, [x10, #1, mul vl]
    ld1w    z2.s, p0/z, [x10, #2, mul vl]
    ld1w    z3.s, p0/z, [x10, #3, mul vl]
    add     x10, x10, x4

    ld1w    z4.s, p0/z, [x10]
    ld1w    z5.s, p0/z, [x10, #1, mul vl]
    ld1w    z6.s, p0/z, [x10, #2, mul vl]
    ld1w    z7.s, p0/z, [x10, #3, mul vl]
    add     x10, x10, x4

    ld1w    z8.s, p0/z, [x10]
    ld1w    z9.s, p0/z, [x10, #1, mul vl]
    ld1w    z10.s, p0/z, [x10, #2, mul vl]
    ld1w    z11.s, p0/z, [x10, #3, mul vl]
    add     x10, x10, x4

    ld1w    z12.s, p0/z, [x10]
    ld1w    z13.s, p0/z, [x10, #1, mul vl]
    ld1w    z14.s, p0/z, [x10, #2, mul vl]
    ld1w    z15.s, p0/z, [x10, #3, mul vl]

    // ========================================
    // Initialize O[4, 64] = 0
    // O[0] in z20-z23, O[1] in z24-z27
    // O[2], O[3] on stack
    // ========================================
    fmov    z20.s, #0.0
    fmov    z21.s, #0.0
    fmov    z22.s, #0.0
    fmov    z23.s, #0.0
    fmov    z24.s, #0.0
    fmov    z25.s, #0.0
    fmov    z26.s, #0.0
    fmov    z27.s, #0.0

    // Initialize O[2], O[3] on stack
    fmov    z28.s, #0.0
    st1w    z28.s, p0, [sp]
    st1w    z28.s, p0, [sp, #1, mul vl]
    st1w    z28.s, p0, [sp, #2, mul vl]
    st1w    z28.s, p0, [sp, #3, mul vl]
    st1w    z28.s, p0, [sp, #4, mul vl]
    st1w    z28.s, p0, [sp, #5, mul vl]
    st1w    z28.s, p0, [sp, #6, mul vl]
    st1w    z28.s, p0, [sp, #7, mul vl]

    mov     x5, x1              // K ptr
    mov     x6, x2              // V ptr
    mov     x7, #64             // 64 K/V rows

    // ========================================
    // Main loop: for each j in 0..64
    // ========================================
.Lfused_loop:
    // Load K[j, 0:64]
    ld1w    z16.s, p0/z, [x5]
    ld1w    z17.s, p0/z, [x5, #1, mul vl]
    ld1w    z18.s, p0/z, [x5, #2, mul vl]
    ld1w    z19.s, p0/z, [x5, #3, mul vl]
    add     x5, x5, x4

    // Compute S[0:4, j] = Q[i, :] · K[j, :]
    // S[0, j]
    fmul    z28.s, z0.s, z16.s
    fmla    z28.s, p0/m, z1.s, z17.s
    fmla    z28.s, p0/m, z2.s, z18.s
    fmla    z28.s, p0/m, z3.s, z19.s
    faddv   s28, p0, z28.s      // s28 = S[0, j]

    // S[1, j]
    fmul    z29.s, z4.s, z16.s
    fmla    z29.s, p0/m, z5.s, z17.s
    fmla    z29.s, p0/m, z6.s, z18.s
    fmla    z29.s, p0/m, z7.s, z19.s
    faddv   s29, p0, z29.s      // s29 = S[1, j]

    // S[2, j]
    fmul    z30.s, z8.s, z16.s
    fmla    z30.s, p0/m, z9.s, z17.s
    fmla    z30.s, p0/m, z10.s, z18.s
    fmla    z30.s, p0/m, z11.s, z19.s
    faddv   s30, p0, z30.s      // s30 = S[2, j]

    // S[3, j]
    fmul    z31.s, z12.s, z16.s
    fmla    z31.s, p0/m, z13.s, z17.s
    fmla    z31.s, p0/m, z14.s, z18.s
    fmla    z31.s, p0/m, z15.s, z19.s
    faddv   s31, p0, z31.s      // s31 = S[3, j]

    // Load V[j, 0:64] (reuse z16-z19)
    ld1w    z16.s, p0/z, [x6]
    ld1w    z17.s, p0/z, [x6, #1, mul vl]
    ld1w    z18.s, p0/z, [x6, #2, mul vl]
    ld1w    z19.s, p0/z, [x6, #3, mul vl]
    add     x6, x6, x4

    // Broadcast S values
    dup     z28.s, z28.s[0]     // S[0, j] broadcast
    dup     z29.s, z29.s[0]     // S[1, j] broadcast
    dup     z30.s, z30.s[0]     // S[2, j] broadcast
    dup     z31.s, z31.s[0]     // S[3, j] broadcast

    // O[0, :] += S[0, j] * V[j, :]
    fmla    z20.s, p0/m, z28.s, z16.s
    fmla    z21.s, p0/m, z28.s, z17.s
    fmla    z22.s, p0/m, z28.s, z18.s
    fmla    z23.s, p0/m, z28.s, z19.s

    // O[1, :] += S[1, j] * V[j, :]
    fmla    z24.s, p0/m, z29.s, z16.s
    fmla    z25.s, p0/m, z29.s, z17.s
    fmla    z26.s, p0/m, z29.s, z18.s
    fmla    z27.s, p0/m, z29.s, z19.s

    // O[2, :] += S[2, j] * V[j, :] (on stack)
    ld1w    z28.s, p0/z, [sp]
    fmla    z28.s, p0/m, z30.s, z16.s
    st1w    z28.s, p0, [sp]

    ld1w    z28.s, p0/z, [sp, #1, mul vl]
    fmla    z28.s, p0/m, z30.s, z17.s
    st1w    z28.s, p0, [sp, #1, mul vl]

    ld1w    z28.s, p0/z, [sp, #2, mul vl]
    fmla    z28.s, p0/m, z30.s, z18.s
    st1w    z28.s, p0, [sp, #2, mul vl]

    ld1w    z28.s, p0/z, [sp, #3, mul vl]
    fmla    z28.s, p0/m, z30.s, z19.s
    st1w    z28.s, p0, [sp, #3, mul vl]

    // O[3, :] += S[3, j] * V[j, :] (on stack)
    ld1w    z28.s, p0/z, [sp, #4, mul vl]
    fmla    z28.s, p0/m, z31.s, z16.s
    st1w    z28.s, p0, [sp, #4, mul vl]

    ld1w    z28.s, p0/z, [sp, #5, mul vl]
    fmla    z28.s, p0/m, z31.s, z17.s
    st1w    z28.s, p0, [sp, #5, mul vl]

    ld1w    z28.s, p0/z, [sp, #6, mul vl]
    fmla    z28.s, p0/m, z31.s, z18.s
    st1w    z28.s, p0, [sp, #6, mul vl]

    ld1w    z28.s, p0/z, [sp, #7, mul vl]
    fmla    z28.s, p0/m, z31.s, z19.s
    st1w    z28.s, p0, [sp, #7, mul vl]

    subs    x7, x7, #1
    b.ne    .Lfused_loop

    // ========================================
    // Store O[4, 64]
    // ========================================
    mov     x10, x3

    // O[0]
    st1w    z20.s, p0, [x10]
    st1w    z21.s, p0, [x10, #1, mul vl]
    st1w    z22.s, p0, [x10, #2, mul vl]
    st1w    z23.s, p0, [x10, #3, mul vl]
    add     x10, x10, x4

    // O[1]
    st1w    z24.s, p0, [x10]
    st1w    z25.s, p0, [x10, #1, mul vl]
    st1w    z26.s, p0, [x10, #2, mul vl]
    st1w    z27.s, p0, [x10, #3, mul vl]
    add     x10, x10, x4

    // O[2] from stack
    ld1w    z28.s, p0/z, [sp]
    ld1w    z29.s, p0/z, [sp, #1, mul vl]
    ld1w    z30.s, p0/z, [sp, #2, mul vl]
    ld1w    z31.s, p0/z, [sp, #3, mul vl]
    st1w    z28.s, p0, [x10]
    st1w    z29.s, p0, [x10, #1, mul vl]
    st1w    z30.s, p0, [x10, #2, mul vl]
    st1w    z31.s, p0, [x10, #3, mul vl]
    add     x10, x10, x4

    // O[3] from stack
    ld1w    z28.s, p0/z, [sp, #4, mul vl]
    ld1w    z29.s, p0/z, [sp, #5, mul vl]
    ld1w    z30.s, p0/z, [sp, #6, mul vl]
    ld1w    z31.s, p0/z, [sp, #7, mul vl]
    st1w    z28.s, p0, [x10]
    st1w    z29.s, p0, [x10, #1, mul vl]
    st1w    z30.s, p0, [x10, #2, mul vl]
    st1w    z31.s, p0, [x10, #3, mul vl]

    // Restore stack and callee-saved registers
    add     sp, sp, #768
    ldp     x21, x22, [sp], #16
    ldp     x19, x20, [sp], #16

    ret
    .size gemm_fused, .-gemm_fused
