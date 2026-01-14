// kernel_6x4_pipe.S
// Software-pipelined INT8 SDOT GEMM Microkernel for A64FX SVE
// Target: 90%+ of 512 GOPS peak
//
// MR=6 rows × NR=64 columns (4 SVE vectors), K=256 fixed
//
// Key optimizations:
// 1. Software pipelining: Load next iteration while computing current
// 2. 4× K-loop unrolling (16 iterations instead of 64)
// 3. Interleaved A packing [K/4][MR][4] for sequential ld1rw
// 4. Better instruction scheduling for dual-pipe A64FX
//
// Register allocation (aggressive):
//   z0-z23:  24 accumulators (6 rows × 4 vectors)
//   z24-z27: A values current iteration (4 of 6 rows)
//   z28-z29: A values current iteration (2 of 6 rows)
//   z30-z31: B column values (rotated for double-buffering)
//   p0:      All-true predicate for bytes
//   p1:      All-true predicate for words

    .arch armv8.2-a+sve
    .text
    .align 6

    .global kernel_6x4_pipe
    .type kernel_6x4_pipe, @function

kernel_6x4_pipe:
    // Save callee-saved registers (need x19-x21 for pointers)
    stp     x19, x20, [sp, #-16]!
    stp     x21, x22, [sp, #-16]!

    // Setup predicates
    ptrue   p0.b
    ptrue   p1.s

    // Zero all 24 accumulators
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

    // Setup pointers
    mov     x4, x0                  // Apack pointer
    mov     x5, x1                  // Bpack pointer

    // K-loop counter: 16 iterations (4 K-groups per iteration)
    mov     x6, #16

    // Prefetch first block
    prfm    pldl1keep, [x4]
    prfm    pldl1keep, [x5]
    prfm    pldl1keep, [x5, #256]

    .align 6
.Lk_loop_pipe:
    // ================================================================
    // Process 4 K-groups (16 bytes of K) per iteration
    // Each K-group: 6 ld1rw A loads + 4 ld1b B loads + 24 SDOTs
    // Total per iteration: 24 A loads + 16 B loads + 96 SDOTs
    // ================================================================

    // Prefetch next iteration's data
    prfm    pldl1strm, [x4, #96]
    prfm    pldl1strm, [x5, #1024]

    // ----------------------------------------
    // K-group 0: k+0,1,2,3
    // ----------------------------------------
    ld1rw   {z24.s}, p0/z, [x4, #0]     // A row 0
    ld1rw   {z25.s}, p0/z, [x4, #4]     // A row 1
    ld1b    {z30.b}, p0/z, [x5, #0, mul vl]   // B col 0
    ld1b    {z31.b}, p0/z, [x5, #1, mul vl]   // B col 1
    ld1rw   {z26.s}, p0/z, [x4, #8]     // A row 2
    ld1rw   {z27.s}, p0/z, [x4, #12]    // A row 3

    sdot    z0.s, z24.b, z30.b          // C[0,0]
    sdot    z1.s, z24.b, z31.b          // C[0,1]
    sdot    z4.s, z25.b, z30.b          // C[1,0]
    sdot    z5.s, z25.b, z31.b          // C[1,1]

    ld1rw   {z28.s}, p0/z, [x4, #16]    // A row 4
    ld1rw   {z29.s}, p0/z, [x4, #20]    // A row 5

    sdot    z8.s, z26.b, z30.b          // C[2,0]
    sdot    z9.s, z26.b, z31.b          // C[2,1]
    sdot    z12.s, z27.b, z30.b         // C[3,0]
    sdot    z13.s, z27.b, z31.b         // C[3,1]
    sdot    z16.s, z28.b, z30.b         // C[4,0]
    sdot    z17.s, z28.b, z31.b         // C[4,1]
    sdot    z20.s, z29.b, z30.b         // C[5,0]
    sdot    z21.s, z29.b, z31.b         // C[5,1]

    ld1b    {z30.b}, p0/z, [x5, #2, mul vl]   // B col 2
    ld1b    {z31.b}, p0/z, [x5, #3, mul vl]   // B col 3

    sdot    z2.s, z24.b, z30.b          // C[0,2]
    sdot    z3.s, z24.b, z31.b          // C[0,3]
    sdot    z6.s, z25.b, z30.b          // C[1,2]
    sdot    z7.s, z25.b, z31.b          // C[1,3]
    sdot    z10.s, z26.b, z30.b         // C[2,2]
    sdot    z11.s, z26.b, z31.b         // C[2,3]
    sdot    z14.s, z27.b, z30.b         // C[3,2]
    sdot    z15.s, z27.b, z31.b         // C[3,3]
    sdot    z18.s, z28.b, z30.b         // C[4,2]
    sdot    z19.s, z28.b, z31.b         // C[4,3]
    sdot    z22.s, z29.b, z30.b         // C[5,2]
    sdot    z23.s, z29.b, z31.b         // C[5,3]

    // ----------------------------------------
    // K-group 1: k+4,5,6,7
    // ----------------------------------------
    ld1rw   {z24.s}, p0/z, [x4, #24]
    ld1rw   {z25.s}, p0/z, [x4, #28]
    ld1b    {z30.b}, p0/z, [x5, #4, mul vl]
    ld1b    {z31.b}, p0/z, [x5, #5, mul vl]
    ld1rw   {z26.s}, p0/z, [x4, #32]
    ld1rw   {z27.s}, p0/z, [x4, #36]

    sdot    z0.s, z24.b, z30.b
    sdot    z1.s, z24.b, z31.b
    sdot    z4.s, z25.b, z30.b
    sdot    z5.s, z25.b, z31.b

    ld1rw   {z28.s}, p0/z, [x4, #40]
    ld1rw   {z29.s}, p0/z, [x4, #44]

    sdot    z8.s, z26.b, z30.b
    sdot    z9.s, z26.b, z31.b
    sdot    z12.s, z27.b, z30.b
    sdot    z13.s, z27.b, z31.b
    sdot    z16.s, z28.b, z30.b
    sdot    z17.s, z28.b, z31.b
    sdot    z20.s, z29.b, z30.b
    sdot    z21.s, z29.b, z31.b

    ld1b    {z30.b}, p0/z, [x5, #6, mul vl]
    ld1b    {z31.b}, p0/z, [x5, #7, mul vl]

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

    // ----------------------------------------
    // K-group 2: k+8,9,10,11
    // ----------------------------------------
    ld1rw   {z24.s}, p0/z, [x4, #48]
    ld1rw   {z25.s}, p0/z, [x4, #52]
    // Note: mul vl offset limited to [-8,7], so we need base update
    add     x19, x5, #512           // Temp base for K-groups 2-3
    ld1b    {z30.b}, p0/z, [x19, #0, mul vl]
    ld1b    {z31.b}, p0/z, [x19, #1, mul vl]
    ld1rw   {z26.s}, p0/z, [x4, #56]
    ld1rw   {z27.s}, p0/z, [x4, #60]

    sdot    z0.s, z24.b, z30.b
    sdot    z1.s, z24.b, z31.b
    sdot    z4.s, z25.b, z30.b
    sdot    z5.s, z25.b, z31.b

    ld1rw   {z28.s}, p0/z, [x4, #64]
    ld1rw   {z29.s}, p0/z, [x4, #68]

    sdot    z8.s, z26.b, z30.b
    sdot    z9.s, z26.b, z31.b
    sdot    z12.s, z27.b, z30.b
    sdot    z13.s, z27.b, z31.b
    sdot    z16.s, z28.b, z30.b
    sdot    z17.s, z28.b, z31.b
    sdot    z20.s, z29.b, z30.b
    sdot    z21.s, z29.b, z31.b

    ld1b    {z30.b}, p0/z, [x19, #2, mul vl]
    ld1b    {z31.b}, p0/z, [x19, #3, mul vl]

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

    // ----------------------------------------
    // K-group 3: k+12,13,14,15
    // ----------------------------------------
    ld1rw   {z24.s}, p0/z, [x4, #72]
    ld1rw   {z25.s}, p0/z, [x4, #76]
    ld1b    {z30.b}, p0/z, [x19, #4, mul vl]
    ld1b    {z31.b}, p0/z, [x19, #5, mul vl]
    ld1rw   {z26.s}, p0/z, [x4, #80]
    ld1rw   {z27.s}, p0/z, [x4, #84]

    sdot    z0.s, z24.b, z30.b
    sdot    z1.s, z24.b, z31.b
    sdot    z4.s, z25.b, z30.b
    sdot    z5.s, z25.b, z31.b

    ld1rw   {z28.s}, p0/z, [x4, #88]
    ld1rw   {z29.s}, p0/z, [x4, #92]

    sdot    z8.s, z26.b, z30.b
    sdot    z9.s, z26.b, z31.b
    sdot    z12.s, z27.b, z30.b
    sdot    z13.s, z27.b, z31.b
    sdot    z16.s, z28.b, z30.b
    sdot    z17.s, z28.b, z31.b
    sdot    z20.s, z29.b, z30.b
    sdot    z21.s, z29.b, z31.b

    ld1b    {z30.b}, p0/z, [x19, #6, mul vl]
    ld1b    {z31.b}, p0/z, [x19, #7, mul vl]

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

    // Advance pointers for next iteration (4 K-groups = 96 A bytes, 1024 B bytes)
    add     x4, x4, #96
    add     x5, x5, #1024

    // Loop control
    subs    x6, x6, #1
    b.gt    .Lk_loop_pipe

    // ================================================================
    // Store results: 6 rows × 4 vectors = 24 stores
    // ================================================================

    // Row 0
    st1w    {z0.s}, p1, [x2, #0, mul vl]
    st1w    {z1.s}, p1, [x2, #1, mul vl]
    st1w    {z2.s}, p1, [x2, #2, mul vl]
    st1w    {z3.s}, p1, [x2, #3, mul vl]
    add     x2, x2, x3

    // Row 1
    st1w    {z4.s}, p1, [x2, #0, mul vl]
    st1w    {z5.s}, p1, [x2, #1, mul vl]
    st1w    {z6.s}, p1, [x2, #2, mul vl]
    st1w    {z7.s}, p1, [x2, #3, mul vl]
    add     x2, x2, x3

    // Row 2
    st1w    {z8.s}, p1, [x2, #0, mul vl]
    st1w    {z9.s}, p1, [x2, #1, mul vl]
    st1w    {z10.s}, p1, [x2, #2, mul vl]
    st1w    {z11.s}, p1, [x2, #3, mul vl]
    add     x2, x2, x3

    // Row 3
    st1w    {z12.s}, p1, [x2, #0, mul vl]
    st1w    {z13.s}, p1, [x2, #1, mul vl]
    st1w    {z14.s}, p1, [x2, #2, mul vl]
    st1w    {z15.s}, p1, [x2, #3, mul vl]
    add     x2, x2, x3

    // Row 4
    st1w    {z16.s}, p1, [x2, #0, mul vl]
    st1w    {z17.s}, p1, [x2, #1, mul vl]
    st1w    {z18.s}, p1, [x2, #2, mul vl]
    st1w    {z19.s}, p1, [x2, #3, mul vl]
    add     x2, x2, x3

    // Row 5
    st1w    {z20.s}, p1, [x2, #0, mul vl]
    st1w    {z21.s}, p1, [x2, #1, mul vl]
    st1w    {z22.s}, p1, [x2, #2, mul vl]
    st1w    {z23.s}, p1, [x2, #3, mul vl]

    // Restore registers
    ldp     x21, x22, [sp], #16
    ldp     x19, x20, [sp], #16

    ret

    .size kernel_6x4_pipe, .-kernel_6x4_pipe
