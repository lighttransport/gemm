// kernel_5x4_dblbuf.S
// Double-buffered 5x4 kernel optimized for 11-cycle SVE load latency
//
// Key insight: 6x4 uses all 32 registers, no room for double-buffering
// Solution: 5x4 tile allows double-buffering A
//
// Register allocation:
//   z0-z19:  20 accumulators (5 rows × 4 cols)
//   z20-z24: A[k] values (5 rows) - current iteration
//   z25-z29: A[k+1] values (5 rows) - next iteration (prefetched)
//   z30-z31: B values (2 vectors, reload for cols 2-3)
//
// Pipeline structure:
//   Prologue: Load A[0] into z20-z24
//   Loop iteration k:
//     - Load B[k,0:1], compute 10 SDOT with A[k] (z20-z24)
//     - Load A[k+1] into z25-z29 (starts here, 11 cycles to ready)
//     - Load B[k,2:3], compute 10 SDOT with A[k] (z20-z24)
//     - Swap A buffers: z20-z24 <-> z25-z29 (just pointer/naming)
//     - A[k+1] now ready in z20-z24 for next iteration
//   Epilogue: Final computation with last A
//
// Latency hiding:
//   - A[k+1] loaded during A[k] compute (10 SDOT = 5 cycles + B loads)
//   - 11 cycle latency partially hidden
//   - Remaining latency hidden across loop boundary

    .arch armv8.2-a+sve
    .text
    .align 6

    .global kernel_5x4_dblbuf
    .type kernel_5x4_dblbuf, @function

// void kernel_5x4_dblbuf(const int8_t* A, const int8_t* B, int32_t* C, int K, int ldc)
// A packed as [K/4][5][4] = 20 bytes per K-group
// B packed as [K/4][4][64] = 256 bytes per K-group
// K must be multiple of 8 (for K-unroll by 2)

kernel_5x4_dblbuf:
    stp     x19, x20, [sp, #-16]!
    stp     x21, x22, [sp, #-16]!
    stp     d8, d9, [sp, #-16]!
    stp     d10, d11, [sp, #-16]!
    stp     d12, d13, [sp, #-16]!
    stp     d14, d15, [sp, #-16]!

    ptrue   p0.b

    // Zero 20 accumulators
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

    // K/4 then /2 for double-buffer = K/8 iterations
    lsr     x5, x3, #3

    // ============================================================
    // Prologue: Load first A[0] into primary buffer (z20-z24)
    // ============================================================
    ld1rw   {z20.s}, p0/z, [x0, #0]     // A row 0
    ld1rw   {z21.s}, p0/z, [x0, #4]     // A row 1
    ld1rw   {z22.s}, p0/z, [x0, #8]     // A row 2
    ld1rw   {z23.s}, p0/z, [x0, #12]    // A row 3
    ld1rw   {z24.s}, p0/z, [x0, #16]    // A row 4
    add     x0, x0, #20                  // Advance A to next K-group

    cbz     x5, .Lepilogue_5x4

    .align 6
.Lk_loop_5x4:
    // ============================================================
    // Phase 1: Load B[k,0:1], compute 10 SDOT with A[k] (z20-z24)
    //          Start loading A[k+1] into secondary buffer (z25-z29)
    // ============================================================
    ld1b    {z30.b}, p0/z, [x1, #0, mul vl]     // B col 0
    ld1b    {z31.b}, p0/z, [x1, #1, mul vl]     // B col 1

    // Compute with A[k], interleave A[k+1] loads
    sdot    z0.s, z20.b, z30.b          // C[0,0]
    ld1rw   {z25.s}, p0/z, [x0, #0]     // A[k+1] row 0 -> z25
    sdot    z1.s, z20.b, z31.b          // C[0,1]
    ld1rw   {z26.s}, p0/z, [x0, #4]     // A[k+1] row 1 -> z26
    sdot    z4.s, z21.b, z30.b          // C[1,0]
    ld1rw   {z27.s}, p0/z, [x0, #8]     // A[k+1] row 2 -> z27
    sdot    z5.s, z21.b, z31.b          // C[1,1]
    ld1rw   {z28.s}, p0/z, [x0, #12]    // A[k+1] row 3 -> z28
    sdot    z8.s, z22.b, z30.b          // C[2,0]
    ld1rw   {z29.s}, p0/z, [x0, #16]    // A[k+1] row 4 -> z29
    sdot    z9.s, z22.b, z31.b          // C[2,1]
    sdot    z12.s, z23.b, z30.b         // C[3,0]
    sdot    z13.s, z23.b, z31.b         // C[3,1]
    sdot    z16.s, z24.b, z30.b         // C[4,0]
    sdot    z17.s, z24.b, z31.b         // C[4,1]

    // ============================================================
    // Phase 2: Load B[k,2:3], compute 10 SDOT with A[k] (z20-z24)
    //          A[k+1] loads issued, wait for them to be ready
    // ============================================================
    ld1b    {z30.b}, p0/z, [x1, #2, mul vl]     // B col 2
    ld1b    {z31.b}, p0/z, [x1, #3, mul vl]     // B col 3

    sdot    z2.s, z20.b, z30.b          // C[0,2]
    sdot    z3.s, z20.b, z31.b          // C[0,3]
    sdot    z6.s, z21.b, z30.b          // C[1,2]
    sdot    z7.s, z21.b, z31.b          // C[1,3]
    sdot    z10.s, z22.b, z30.b         // C[2,2]
    sdot    z11.s, z22.b, z31.b         // C[2,3]
    sdot    z14.s, z23.b, z30.b         // C[3,2]
    sdot    z15.s, z23.b, z31.b         // C[3,3]
    sdot    z18.s, z24.b, z30.b         // C[4,2]
    sdot    z19.s, z24.b, z31.b         // C[4,3]

    // Advance B pointer
    add     x1, x1, #256                // B += 4 vectors × 64 bytes

    // ============================================================
    // Phase 3: Load B[k+1,0:1], compute 20 SDOT with A[k+1] (z25-z29)
    //          By now, z25-z29 should be ready (11 cycles elapsed)
    //          Start loading A[k+2] into primary buffer (z20-z24)
    // ============================================================
    ld1b    {z30.b}, p0/z, [x1, #0, mul vl]     // B[k+1] col 0
    ld1b    {z31.b}, p0/z, [x1, #1, mul vl]     // B[k+1] col 1

    // Compute with A[k+1], interleave A[k+2] loads
    sdot    z0.s, z25.b, z30.b          // C[0,0]
    ld1rw   {z20.s}, p0/z, [x0, #20]    // A[k+2] row 0 -> z20
    sdot    z1.s, z25.b, z31.b          // C[0,1]
    ld1rw   {z21.s}, p0/z, [x0, #24]    // A[k+2] row 1 -> z21
    sdot    z4.s, z26.b, z30.b          // C[1,0]
    ld1rw   {z22.s}, p0/z, [x0, #28]    // A[k+2] row 2 -> z22
    sdot    z5.s, z26.b, z31.b          // C[1,1]
    ld1rw   {z23.s}, p0/z, [x0, #32]    // A[k+2] row 3 -> z23
    sdot    z8.s, z27.b, z30.b          // C[2,0]
    ld1rw   {z24.s}, p0/z, [x0, #36]    // A[k+2] row 4 -> z24
    sdot    z9.s, z27.b, z31.b          // C[2,1]
    sdot    z12.s, z28.b, z30.b         // C[3,0]
    sdot    z13.s, z28.b, z31.b         // C[3,1]
    sdot    z16.s, z29.b, z30.b         // C[4,0]
    sdot    z17.s, z29.b, z31.b         // C[4,1]

    // ============================================================
    // Phase 4: Load B[k+1,2:3], compute 10 SDOT with A[k+1] (z25-z29)
    // ============================================================
    ld1b    {z30.b}, p0/z, [x1, #2, mul vl]     // B[k+1] col 2
    ld1b    {z31.b}, p0/z, [x1, #3, mul vl]     // B[k+1] col 3

    sdot    z2.s, z25.b, z30.b          // C[0,2]
    sdot    z3.s, z25.b, z31.b          // C[0,3]
    sdot    z6.s, z26.b, z30.b          // C[1,2]
    sdot    z7.s, z26.b, z31.b          // C[1,3]
    sdot    z10.s, z27.b, z30.b         // C[2,2]
    sdot    z11.s, z27.b, z31.b         // C[2,3]
    sdot    z14.s, z28.b, z30.b         // C[3,2]
    sdot    z15.s, z28.b, z31.b         // C[3,3]
    sdot    z18.s, z29.b, z30.b         // C[4,2]
    sdot    z19.s, z29.b, z31.b         // C[4,3]

    // Advance pointers
    add     x0, x0, #40                 // A += 2 K-groups × 20 bytes
    add     x1, x1, #256                // B += 4 vectors × 64 bytes

    subs    x5, x5, #1
    b.gt    .Lk_loop_5x4

.Lepilogue_5x4:
    // Store results (5 rows × 64 columns)
    ptrue   p0.s
    st1w    {z0.s}, p0, [x2, #0, mul vl]
    st1w    {z1.s}, p0, [x2, #1, mul vl]
    st1w    {z2.s}, p0, [x2, #2, mul vl]
    st1w    {z3.s}, p0, [x2, #3, mul vl]
    add     x2, x2, x4
    st1w    {z4.s}, p0, [x2, #0, mul vl]
    st1w    {z5.s}, p0, [x2, #1, mul vl]
    st1w    {z6.s}, p0, [x2, #2, mul vl]
    st1w    {z7.s}, p0, [x2, #3, mul vl]
    add     x2, x2, x4
    st1w    {z8.s}, p0, [x2, #0, mul vl]
    st1w    {z9.s}, p0, [x2, #1, mul vl]
    st1w    {z10.s}, p0, [x2, #2, mul vl]
    st1w    {z11.s}, p0, [x2, #3, mul vl]
    add     x2, x2, x4
    st1w    {z12.s}, p0, [x2, #0, mul vl]
    st1w    {z13.s}, p0, [x2, #1, mul vl]
    st1w    {z14.s}, p0, [x2, #2, mul vl]
    st1w    {z15.s}, p0, [x2, #3, mul vl]
    add     x2, x2, x4
    st1w    {z16.s}, p0, [x2, #0, mul vl]
    st1w    {z17.s}, p0, [x2, #1, mul vl]
    st1w    {z18.s}, p0, [x2, #2, mul vl]
    st1w    {z19.s}, p0, [x2, #3, mul vl]

    ldp     d14, d15, [sp], #16
    ldp     d12, d13, [sp], #16
    ldp     d10, d11, [sp], #16
    ldp     d8, d9, [sp], #16
    ldp     x21, x22, [sp], #16
    ldp     x19, x20, [sp], #16
    ret

    .size kernel_5x4_dblbuf, .-kernel_5x4_dblbuf


// ============================================================
// Same-address test version for pure compute benchmarking
// ============================================================
    .global kernel_5x4_dblbuf_same
    .type kernel_5x4_dblbuf_same, @function

kernel_5x4_dblbuf_same:
    stp     x19, x20, [sp, #-16]!
    stp     d8, d9, [sp, #-16]!
    stp     d10, d11, [sp, #-16]!
    stp     d12, d13, [sp, #-16]!
    stp     d14, d15, [sp, #-16]!

    ptrue   p0.b

    // Zero 20 accumulators
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

    mov     x5, x3                      // K_iters (not K/4)

    // Pre-load A into primary buffer
    ld1rw   {z20.s}, p0/z, [x0, #0]
    ld1rw   {z21.s}, p0/z, [x0, #4]
    ld1rw   {z22.s}, p0/z, [x0, #8]
    ld1rw   {z23.s}, p0/z, [x0, #12]
    ld1rw   {z24.s}, p0/z, [x0, #16]

    .align 6
.Lloop_5x4_same:
    // Phase 1: B[0:1], compute with A (z20-z24), load A into z25-z29
    ld1b    {z30.b}, p0/z, [x1, #0, mul vl]
    ld1b    {z31.b}, p0/z, [x1, #1, mul vl]

    sdot    z0.s, z20.b, z30.b
    ld1rw   {z25.s}, p0/z, [x0, #0]     // Same address
    sdot    z1.s, z20.b, z31.b
    ld1rw   {z26.s}, p0/z, [x0, #4]
    sdot    z4.s, z21.b, z30.b
    ld1rw   {z27.s}, p0/z, [x0, #8]
    sdot    z5.s, z21.b, z31.b
    ld1rw   {z28.s}, p0/z, [x0, #12]
    sdot    z8.s, z22.b, z30.b
    ld1rw   {z29.s}, p0/z, [x0, #16]
    sdot    z9.s, z22.b, z31.b
    sdot    z12.s, z23.b, z30.b
    sdot    z13.s, z23.b, z31.b
    sdot    z16.s, z24.b, z30.b
    sdot    z17.s, z24.b, z31.b

    // Phase 2: B[2:3], compute with A (z20-z24)
    ld1b    {z30.b}, p0/z, [x1, #2, mul vl]
    ld1b    {z31.b}, p0/z, [x1, #3, mul vl]

    sdot    z2.s, z20.b, z30.b
    sdot    z3.s, z20.b, z31.b
    sdot    z6.s, z21.b, z30.b
    sdot    z7.s, z21.b, z31.b
    sdot    z10.s, z22.b, z30.b
    sdot    z11.s, z22.b, z31.b
    sdot    z14.s, z23.b, z30.b
    sdot    z15.s, z23.b, z31.b
    sdot    z18.s, z24.b, z30.b
    sdot    z19.s, z24.b, z31.b

    // Phase 3: B[0:1], compute with A (z25-z29), load A into z20-z24
    ld1b    {z30.b}, p0/z, [x1, #0, mul vl]
    ld1b    {z31.b}, p0/z, [x1, #1, mul vl]

    sdot    z0.s, z25.b, z30.b
    ld1rw   {z20.s}, p0/z, [x0, #0]
    sdot    z1.s, z25.b, z31.b
    ld1rw   {z21.s}, p0/z, [x0, #4]
    sdot    z4.s, z26.b, z30.b
    ld1rw   {z22.s}, p0/z, [x0, #8]
    sdot    z5.s, z26.b, z31.b
    ld1rw   {z23.s}, p0/z, [x0, #12]
    sdot    z8.s, z27.b, z30.b
    ld1rw   {z24.s}, p0/z, [x0, #16]
    sdot    z9.s, z27.b, z31.b
    sdot    z12.s, z28.b, z30.b
    sdot    z13.s, z28.b, z31.b
    sdot    z16.s, z29.b, z30.b
    sdot    z17.s, z29.b, z31.b

    // Phase 4: B[2:3], compute with A (z25-z29)
    ld1b    {z30.b}, p0/z, [x1, #2, mul vl]
    ld1b    {z31.b}, p0/z, [x1, #3, mul vl]

    sdot    z2.s, z25.b, z30.b
    sdot    z3.s, z25.b, z31.b
    sdot    z6.s, z26.b, z30.b
    sdot    z7.s, z26.b, z31.b
    sdot    z10.s, z27.b, z30.b
    sdot    z11.s, z27.b, z31.b
    sdot    z14.s, z28.b, z30.b
    sdot    z15.s, z28.b, z31.b
    sdot    z18.s, z29.b, z30.b
    sdot    z19.s, z29.b, z31.b

    // No pointer advance - same address test
    subs    x5, x5, #1
    b.gt    .Lloop_5x4_same

    // Store
    ptrue   p0.s
    st1w    {z0.s}, p0, [x2, #0, mul vl]
    st1w    {z1.s}, p0, [x2, #1, mul vl]
    st1w    {z2.s}, p0, [x2, #2, mul vl]
    st1w    {z3.s}, p0, [x2, #3, mul vl]
    add     x2, x2, x4
    st1w    {z4.s}, p0, [x2, #0, mul vl]
    st1w    {z5.s}, p0, [x2, #1, mul vl]
    st1w    {z6.s}, p0, [x2, #2, mul vl]
    st1w    {z7.s}, p0, [x2, #3, mul vl]
    add     x2, x2, x4
    st1w    {z8.s}, p0, [x2, #0, mul vl]
    st1w    {z9.s}, p0, [x2, #1, mul vl]
    st1w    {z10.s}, p0, [x2, #2, mul vl]
    st1w    {z11.s}, p0, [x2, #3, mul vl]
    add     x2, x2, x4
    st1w    {z12.s}, p0, [x2, #0, mul vl]
    st1w    {z13.s}, p0, [x2, #1, mul vl]
    st1w    {z14.s}, p0, [x2, #2, mul vl]
    st1w    {z15.s}, p0, [x2, #3, mul vl]
    add     x2, x2, x4
    st1w    {z16.s}, p0, [x2, #0, mul vl]
    st1w    {z17.s}, p0, [x2, #1, mul vl]
    st1w    {z18.s}, p0, [x2, #2, mul vl]
    st1w    {z19.s}, p0, [x2, #3, mul vl]

    ldp     d14, d15, [sp], #16
    ldp     d12, d13, [sp], #16
    ldp     d10, d11, [sp], #16
    ldp     d8, d9, [sp], #16
    ldp     x19, x20, [sp], #16
    ret

    .size kernel_5x4_dblbuf_same, .-kernel_5x4_dblbuf_same
