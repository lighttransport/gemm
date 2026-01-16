// kernel_6x4_pipeV2.S
// Software-pipelined INT8 SDOT GEMM Microkernel for A64FX SVE
// Variable K support with true software pipelining
//
// Strategy: Load next iteration's A while computing current iteration's SDOTs
// - Prologue: Load first A
// - Main loop: Compute with A[k], load A[k+1], interleave B loads
// - Epilogue: Last iteration without next load
//
// Register allocation:
//   z0-z23:  24 accumulators
//   z24-z29: A values (current iteration)
//   z30-z31: B values (reloaded)
//   p0: All-true predicate

    .arch armv8.2-a+sve
    .text
    .align 6

    .global kernel_6x4_pipeV2
    .type kernel_6x4_pipeV2, @function

// void kernel_6x4_pipeV2(const int8_t* A, const int8_t* B, int32_t* C, int K, int ldc)
kernel_6x4_pipeV2:
    stp     x19, x20, [sp, #-16]!
    stp     x21, x22, [sp, #-16]!

    // Setup predicate
    ptrue   p0.b

    // Zero accumulators
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

    // K iterations = K/4, then -1 for epilogue
    lsr     x5, x3, #2              // x5 = K/4
    sub     x5, x5, #1              // Main loop does K/4-1 iterations

    // Prefetch
    prfm    pldl1strm, [x0, #192]
    prfm    pldl1strm, [x1, #2048]

    // ================================================================
    // Prologue: Load first iteration's A
    // ================================================================
    ld1rw   {z24.s}, p0/z, [x0, #0]
    ld1rw   {z25.s}, p0/z, [x0, #4]
    ld1rw   {z26.s}, p0/z, [x0, #8]
    ld1rw   {z27.s}, p0/z, [x0, #12]
    ld1rw   {z28.s}, p0/z, [x0, #16]
    ld1rw   {z29.s}, p0/z, [x0, #20]

    cbz     x5, .Lepilogue          // If K=4, skip main loop

    .align 6
.Lmain_loop:
    // ================================================================
    // Compute with current A, interleave B loads and next A loads
    // ================================================================

    // Load B cols 0-1
    ld1b    {z30.b}, p0/z, [x1, #0, mul vl]
    ld1b    {z31.b}, p0/z, [x1, #1, mul vl]

    // SDOTs rows 0-1 with B[0,1], interleave next A loads
    sdot    z0.s, z24.b, z30.b      // C[0,0]
    sdot    z4.s, z25.b, z30.b      // C[1,0]
    sdot    z1.s, z24.b, z31.b      // C[0,1]
    sdot    z5.s, z25.b, z31.b      // C[1,1]

    // SDOTs rows 2-5 with B[0,1]
    sdot    z8.s, z26.b, z30.b      // C[2,0]
    sdot    z12.s, z27.b, z30.b     // C[3,0]
    sdot    z9.s, z26.b, z31.b      // C[2,1]
    sdot    z13.s, z27.b, z31.b     // C[3,1]
    sdot    z16.s, z28.b, z30.b     // C[4,0]
    sdot    z20.s, z29.b, z30.b     // C[5,0]
    sdot    z17.s, z28.b, z31.b     // C[4,1]
    sdot    z21.s, z29.b, z31.b     // C[5,1]

    // Load B cols 2-3
    ld1b    {z30.b}, p0/z, [x1, #2, mul vl]
    ld1b    {z31.b}, p0/z, [x1, #3, mul vl]

    // SDOTs all rows with B[2,3]
    sdot    z2.s, z24.b, z30.b      // C[0,2]
    sdot    z6.s, z25.b, z30.b      // C[1,2]
    sdot    z3.s, z24.b, z31.b      // C[0,3]
    sdot    z7.s, z25.b, z31.b      // C[1,3]
    sdot    z10.s, z26.b, z30.b     // C[2,2]
    sdot    z14.s, z27.b, z30.b     // C[3,2]
    sdot    z11.s, z26.b, z31.b     // C[2,3]
    sdot    z15.s, z27.b, z31.b     // C[3,3]
    sdot    z18.s, z28.b, z30.b     // C[4,2]
    sdot    z22.s, z29.b, z30.b     // C[5,2]
    sdot    z19.s, z28.b, z31.b     // C[4,3]
    sdot    z23.s, z29.b, z31.b     // C[5,3]

    // Advance pointers
    add     x0, x0, #24
    add     x1, x1, #256

    // Load next iteration's A (pipelined)
    ld1rw   {z24.s}, p0/z, [x0, #0]
    ld1rw   {z25.s}, p0/z, [x0, #4]
    ld1rw   {z26.s}, p0/z, [x0, #8]
    ld1rw   {z27.s}, p0/z, [x0, #12]
    ld1rw   {z28.s}, p0/z, [x0, #16]
    ld1rw   {z29.s}, p0/z, [x0, #20]

    // Loop
    subs    x5, x5, #1
    b.gt    .Lmain_loop

    // ================================================================
    // Epilogue: Last iteration (A already loaded, no next A to load)
    // ================================================================
.Lepilogue:
    // Load B cols 0-1
    ld1b    {z30.b}, p0/z, [x1, #0, mul vl]
    ld1b    {z31.b}, p0/z, [x1, #1, mul vl]

    sdot    z0.s, z24.b, z30.b
    sdot    z4.s, z25.b, z30.b
    sdot    z1.s, z24.b, z31.b
    sdot    z5.s, z25.b, z31.b
    sdot    z8.s, z26.b, z30.b
    sdot    z12.s, z27.b, z30.b
    sdot    z9.s, z26.b, z31.b
    sdot    z13.s, z27.b, z31.b
    sdot    z16.s, z28.b, z30.b
    sdot    z20.s, z29.b, z30.b
    sdot    z17.s, z28.b, z31.b
    sdot    z21.s, z29.b, z31.b

    // Load B cols 2-3
    ld1b    {z30.b}, p0/z, [x1, #2, mul vl]
    ld1b    {z31.b}, p0/z, [x1, #3, mul vl]

    sdot    z2.s, z24.b, z30.b
    sdot    z6.s, z25.b, z30.b
    sdot    z3.s, z24.b, z31.b
    sdot    z7.s, z25.b, z31.b
    sdot    z10.s, z26.b, z30.b
    sdot    z14.s, z27.b, z30.b
    sdot    z11.s, z26.b, z31.b
    sdot    z15.s, z27.b, z31.b
    sdot    z18.s, z28.b, z30.b
    sdot    z22.s, z29.b, z30.b
    sdot    z19.s, z28.b, z31.b
    sdot    z23.s, z29.b, z31.b

    // ================================================================
    // Store results
    // ================================================================
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
    add     x2, x2, x4

    st1w    {z20.s}, p0, [x2, #0, mul vl]
    st1w    {z21.s}, p0, [x2, #1, mul vl]
    st1w    {z22.s}, p0, [x2, #2, mul vl]
    st1w    {z23.s}, p0, [x2, #3, mul vl]

    ldp     x21, x22, [sp], #16
    ldp     x19, x20, [sp], #16
    ret

    .size kernel_6x4_pipeV2, .-kernel_6x4_pipeV2
