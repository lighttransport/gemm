// kernel_6x4_pipe2.S
// Software-pipelined fully unrolled INT8 SDOT GEMM Microkernel for A64FX
// Target: 95%+ of 512 GOPS peak
//
// Key optimization: Load A[k+1] while computing SDOTs with A[k]
// This hides ld1rw latency (4 cycles) behind SDOT computation (24 ops)
//
// MR=6 rows × NR=64 columns (4 SVE vectors), K=256 fixed

    .arch armv8.2-a+sve
    .text
    .align 6

    .global kernel_6x4_pipe2
    .type kernel_6x4_pipe2, @function

// Macro for one K-group with pipelined A loads for next iteration
// Assumes A[current] already in z24-z29
// Loads A[next] into z24-z29 AFTER starting SDOTs (overlap)
.macro KGROUP_PIPE A_NEXT, A_OFF, B_BASE, B_IDX
    // B loads for columns 0-1
    ld1b    {z30.b}, p0/z, [\B_BASE, #\B_IDX, mul vl]
    ld1b    {z31.b}, p0/z, [\B_BASE, #\B_IDX+1, mul vl]

    // First 12 SDOTs (rows 0-2 × cols 0-1, rows 3-5 × cols 0-1)
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

    // B loads for columns 2-3
    ld1b    {z30.b}, p0/z, [\B_BASE, #\B_IDX+2, mul vl]
    ld1b    {z31.b}, p0/z, [\B_BASE, #\B_IDX+3, mul vl]

    // Remaining 12 SDOTs (rows 0-2 × cols 2-3, rows 3-5 × cols 2-3)
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

    // Load A for NEXT K-group (overlapped with dependent SDOTs above)
    ld1rw   {z24.s}, p0/z, [\A_NEXT, #\A_OFF]
    ld1rw   {z25.s}, p0/z, [\A_NEXT, #\A_OFF+4]
    ld1rw   {z26.s}, p0/z, [\A_NEXT, #\A_OFF+8]
    ld1rw   {z27.s}, p0/z, [\A_NEXT, #\A_OFF+12]
    ld1rw   {z28.s}, p0/z, [\A_NEXT, #\A_OFF+16]
    ld1rw   {z29.s}, p0/z, [\A_NEXT, #\A_OFF+20]
.endm

// Last K-group - no next A to load
.macro KGROUP_LAST B_BASE, B_IDX
    ld1b    {z30.b}, p0/z, [\B_BASE, #\B_IDX, mul vl]
    ld1b    {z31.b}, p0/z, [\B_BASE, #\B_IDX+1, mul vl]

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

    ld1b    {z30.b}, p0/z, [\B_BASE, #\B_IDX+2, mul vl]
    ld1b    {z31.b}, p0/z, [\B_BASE, #\B_IDX+3, mul vl]

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
.endm

kernel_6x4_pipe2:
    // Setup predicate
    ptrue   p0.b

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

    // Precompute B base pointers to reduce per-K-group overhead
    // B layout: [K/4][4 vectors][16 lanes][4] = 256 bytes per K-group
    add     x4, x1, #256        // B + 256
    add     x5, x1, #512        // B + 512
    add     x6, x1, #768        // B + 768
    add     x7, x1, #1024       // B + 1024
    add     x8, x1, #1280       // B + 1280
    add     x9, x1, #1536       // B + 1536
    add     x10, x1, #1792      // B + 1792

    // Load initial A (K-group 0)
    ld1rw   {z24.s}, p0/z, [x0, #0]
    ld1rw   {z25.s}, p0/z, [x0, #4]
    ld1rw   {z26.s}, p0/z, [x0, #8]
    ld1rw   {z27.s}, p0/z, [x0, #12]
    ld1rw   {z28.s}, p0/z, [x0, #16]
    ld1rw   {z29.s}, p0/z, [x0, #20]

    // Section 0: K-groups 0-7 (A offsets 0-168, B offsets 0-1792)
    KGROUP_PIPE x0, 24, x1, 0       // K=0, load A for K=1
    KGROUP_PIPE x0, 48, x4, 0       // K=1, load A for K=2
    KGROUP_PIPE x0, 72, x5, 0       // K=2, load A for K=3
    KGROUP_PIPE x0, 96, x6, 0       // K=3, load A for K=4
    KGROUP_PIPE x0, 120, x7, 0      // K=4, load A for K=5
    KGROUP_PIPE x0, 144, x8, 0      // K=5, load A for K=6
    KGROUP_PIPE x0, 168, x9, 0      // K=6, load A for K=7

    // Update A base for section 1
    add     x11, x0, #192

    // Update B bases for section 1 (K-groups 8-15)
    add     x4, x1, #2048       // B + 2048
    add     x5, x1, #2304       // B + 2304
    add     x6, x1, #2560       // B + 2560
    add     x7, x1, #2816       // B + 2816
    add     x8, x1, #3072       // B + 3072
    add     x9, x1, #3328       // B + 3328
    add     x10, x1, #3584      // B + 3584
    add     x12, x1, #3840      // B + 3840

    KGROUP_PIPE x11, 0, x10, 0      // K=7, load A for K=8
    KGROUP_PIPE x11, 24, x4, 0      // K=8, load A for K=9
    KGROUP_PIPE x11, 48, x5, 0      // K=9, load A for K=10
    KGROUP_PIPE x11, 72, x6, 0      // K=10, load A for K=11
    KGROUP_PIPE x11, 96, x7, 0      // K=11, load A for K=12
    KGROUP_PIPE x11, 120, x8, 0     // K=12, load A for K=13
    KGROUP_PIPE x11, 144, x9, 0     // K=13, load A for K=14
    KGROUP_PIPE x11, 168, x12, 0    // K=14, load A for K=15

    // Section 2: K-groups 16-23
    add     x11, x0, #384

    add     x4, x1, #4096           // B + 4096
    add     x5, x4, #256            // B + 4352
    add     x6, x4, #512            // B + 4608
    add     x7, x4, #768            // B + 4864
    add     x8, x4, #1024           // B + 5120
    add     x9, x4, #1280           // B + 5376
    add     x10, x4, #1536          // B + 5632
    add     x12, x4, #1792          // B + 5888

    KGROUP_PIPE x11, 0, x4, 0
    KGROUP_PIPE x11, 24, x5, 0
    KGROUP_PIPE x11, 48, x6, 0
    KGROUP_PIPE x11, 72, x7, 0
    KGROUP_PIPE x11, 96, x8, 0
    KGROUP_PIPE x11, 120, x9, 0
    KGROUP_PIPE x11, 144, x10, 0
    KGROUP_PIPE x11, 168, x12, 0

    // Section 3: K-groups 24-31
    add     x11, x0, #576

    add     x4, x4, #2048           // B + 6144
    add     x5, x4, #256            // B + 6400
    add     x6, x4, #512            // B + 6656
    add     x7, x4, #768            // B + 6912
    add     x8, x4, #1024           // B + 7168
    add     x9, x4, #1280           // B + 7424
    add     x10, x4, #1536          // B + 7680
    add     x12, x4, #1792          // B + 7936

    KGROUP_PIPE x11, 0, x4, 0
    KGROUP_PIPE x11, 24, x5, 0
    KGROUP_PIPE x11, 48, x6, 0
    KGROUP_PIPE x11, 72, x7, 0
    KGROUP_PIPE x11, 96, x8, 0
    KGROUP_PIPE x11, 120, x9, 0
    KGROUP_PIPE x11, 144, x10, 0
    KGROUP_PIPE x11, 168, x12, 0

    // Section 4: K-groups 32-39
    add     x11, x0, #768

    add     x4, x1, #8192
    add     x5, x4, #256
    add     x6, x4, #512
    add     x7, x4, #768
    add     x8, x4, #1024
    add     x9, x4, #1280
    add     x10, x4, #1536
    add     x12, x4, #1792

    KGROUP_PIPE x11, 0, x4, 0
    KGROUP_PIPE x11, 24, x5, 0
    KGROUP_PIPE x11, 48, x6, 0
    KGROUP_PIPE x11, 72, x7, 0
    KGROUP_PIPE x11, 96, x8, 0
    KGROUP_PIPE x11, 120, x9, 0
    KGROUP_PIPE x11, 144, x10, 0
    KGROUP_PIPE x11, 168, x12, 0

    // Section 5: K-groups 40-47
    add     x11, x0, #960

    add     x4, x4, #2048           // B + 10240 (x4 was B+8192)
    add     x5, x4, #256
    add     x6, x4, #512
    add     x7, x4, #768
    add     x8, x4, #1024
    add     x9, x4, #1280
    add     x10, x4, #1536
    add     x12, x4, #1792

    KGROUP_PIPE x11, 0, x4, 0
    KGROUP_PIPE x11, 24, x5, 0
    KGROUP_PIPE x11, 48, x6, 0
    KGROUP_PIPE x11, 72, x7, 0
    KGROUP_PIPE x11, 96, x8, 0
    KGROUP_PIPE x11, 120, x9, 0
    KGROUP_PIPE x11, 144, x10, 0
    KGROUP_PIPE x11, 168, x12, 0

    // Section 6: K-groups 48-55
    add     x11, x0, #1152

    add     x4, x4, #2048           // B + 12288 (x4 was B+10240)
    add     x5, x4, #256
    add     x6, x4, #512
    add     x7, x4, #768
    add     x8, x4, #1024
    add     x9, x4, #1280
    add     x10, x4, #1536
    add     x12, x4, #1792

    KGROUP_PIPE x11, 0, x4, 0
    KGROUP_PIPE x11, 24, x5, 0
    KGROUP_PIPE x11, 48, x6, 0
    KGROUP_PIPE x11, 72, x7, 0
    KGROUP_PIPE x11, 96, x8, 0
    KGROUP_PIPE x11, 120, x9, 0
    KGROUP_PIPE x11, 144, x10, 0
    KGROUP_PIPE x11, 168, x12, 0

    // Section 7: K-groups 56-63
    add     x11, x0, #1344

    add     x4, x4, #2048           // B + 14336 (x4 was B+12288)
    add     x5, x4, #256
    add     x6, x4, #512
    add     x7, x4, #768
    add     x8, x4, #1024
    add     x9, x4, #1280
    add     x10, x4, #1536
    add     x12, x4, #1792

    KGROUP_PIPE x11, 0, x4, 0
    KGROUP_PIPE x11, 24, x5, 0
    KGROUP_PIPE x11, 48, x6, 0
    KGROUP_PIPE x11, 72, x7, 0
    KGROUP_PIPE x11, 96, x8, 0
    KGROUP_PIPE x11, 120, x9, 0
    KGROUP_PIPE x11, 144, x10, 0
    KGROUP_LAST x12, 0              // Last K-group, no next A load

    // Store results
    ptrue   p0.s

    st1w    {z0.s}, p0, [x2, #0, mul vl]
    st1w    {z1.s}, p0, [x2, #1, mul vl]
    st1w    {z2.s}, p0, [x2, #2, mul vl]
    st1w    {z3.s}, p0, [x2, #3, mul vl]
    add     x2, x2, x3

    st1w    {z4.s}, p0, [x2, #0, mul vl]
    st1w    {z5.s}, p0, [x2, #1, mul vl]
    st1w    {z6.s}, p0, [x2, #2, mul vl]
    st1w    {z7.s}, p0, [x2, #3, mul vl]
    add     x2, x2, x3

    st1w    {z8.s}, p0, [x2, #0, mul vl]
    st1w    {z9.s}, p0, [x2, #1, mul vl]
    st1w    {z10.s}, p0, [x2, #2, mul vl]
    st1w    {z11.s}, p0, [x2, #3, mul vl]
    add     x2, x2, x3

    st1w    {z12.s}, p0, [x2, #0, mul vl]
    st1w    {z13.s}, p0, [x2, #1, mul vl]
    st1w    {z14.s}, p0, [x2, #2, mul vl]
    st1w    {z15.s}, p0, [x2, #3, mul vl]
    add     x2, x2, x3

    st1w    {z16.s}, p0, [x2, #0, mul vl]
    st1w    {z17.s}, p0, [x2, #1, mul vl]
    st1w    {z18.s}, p0, [x2, #2, mul vl]
    st1w    {z19.s}, p0, [x2, #3, mul vl]
    add     x2, x2, x3

    st1w    {z20.s}, p0, [x2, #0, mul vl]
    st1w    {z21.s}, p0, [x2, #1, mul vl]
    st1w    {z22.s}, p0, [x2, #2, mul vl]
    st1w    {z23.s}, p0, [x2, #3, mul vl]

    ret

    .size kernel_6x4_pipe2, .-kernel_6x4_pipe2
