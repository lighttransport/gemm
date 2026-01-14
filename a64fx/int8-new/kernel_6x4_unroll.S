// kernel_6x4_unroll.S
// Fully unrolled INT8 SDOT GEMM Microkernel for A64FX SVE
// Target: 95%+ of 512 GOPS peak
//
// MR=6 rows × NR=64 columns (4 SVE vectors), K=256 fixed
// All 64 K-groups fully unrolled - NO loop overhead

    .arch armv8.2-a+sve
    .text
    .align 6

    .global kernel_6x4_unroll
    .type kernel_6x4_unroll, @function

// Macro for one K-group
.macro KGROUP A_BASE, A_OFF, B_BASE, B_IDX
    ld1rw   {z24.s}, p0/z, [\A_BASE, #\A_OFF]
    ld1rw   {z25.s}, p0/z, [\A_BASE, #\A_OFF+4]
    ld1rw   {z26.s}, p0/z, [\A_BASE, #\A_OFF+8]
    ld1rw   {z27.s}, p0/z, [\A_BASE, #\A_OFF+12]
    ld1rw   {z28.s}, p0/z, [\A_BASE, #\A_OFF+16]
    ld1rw   {z29.s}, p0/z, [\A_BASE, #\A_OFF+20]

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

kernel_6x4_unroll:
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

    // Section 0: K-groups 0-7 (A offsets 0-168)
    KGROUP x0, 0, x1, 0
    KGROUP x0, 24, x1, 4

    add     x4, x1, #512
    KGROUP x0, 48, x4, 0
    KGROUP x0, 72, x4, 4

    add     x4, x1, #1024
    KGROUP x0, 96, x4, 0
    KGROUP x0, 120, x4, 4

    add     x4, x1, #1536
    KGROUP x0, 144, x4, 0
    KGROUP x0, 168, x4, 4

    // Section 1: K-groups 8-15 (A offsets 192-360, need new A base)
    add     x5, x0, #192

    add     x4, x1, #2048
    KGROUP x5, 0, x4, 0
    KGROUP x5, 24, x4, 4

    add     x4, x1, #2560
    KGROUP x5, 48, x4, 0
    KGROUP x5, 72, x4, 4

    add     x4, x1, #3072
    KGROUP x5, 96, x4, 0
    KGROUP x5, 120, x4, 4

    add     x4, x1, #3584
    KGROUP x5, 144, x4, 0
    KGROUP x5, 168, x4, 4

    // Section 2: K-groups 16-23
    add     x5, x0, #384

    add     x4, x1, #4096
    KGROUP x5, 0, x4, 0

    add     x6, x4, #256           // x4 + 256 = x1 + 4352
    KGROUP x5, 24, x6, 0

    add     x6, x4, #512           // x1 + 4608
    KGROUP x5, 48, x6, 0

    add     x6, x4, #768           // x1 + 4864
    KGROUP x5, 72, x6, 0

    add     x6, x4, #1024          // x1 + 5120
    KGROUP x5, 96, x6, 0

    add     x6, x4, #1280          // x1 + 5376
    KGROUP x5, 120, x6, 0

    add     x6, x4, #1536          // x1 + 5632
    KGROUP x5, 144, x6, 0

    add     x6, x4, #1792          // x1 + 5888
    KGROUP x5, 168, x6, 0

    // Section 3: K-groups 24-31
    add     x5, x0, #576

    add     x4, x4, #2048          // x4 now = x1 + 6144
    KGROUP x5, 0, x4, 0

    add     x6, x4, #256
    KGROUP x5, 24, x6, 0

    add     x6, x4, #512
    KGROUP x5, 48, x6, 0

    add     x6, x4, #768
    KGROUP x5, 72, x6, 0

    add     x6, x4, #1024
    KGROUP x5, 96, x6, 0

    add     x6, x4, #1280
    KGROUP x5, 120, x6, 0

    add     x6, x4, #1536
    KGROUP x5, 144, x6, 0

    add     x6, x4, #1792
    KGROUP x5, 168, x6, 0

    // Section 4: K-groups 32-39
    add     x5, x0, #768

    add     x4, x4, #2048          // x4 now = x1 + 8192
    KGROUP x5, 0, x4, 0

    add     x6, x4, #256
    KGROUP x5, 24, x6, 0

    add     x6, x4, #512
    KGROUP x5, 48, x6, 0

    add     x6, x4, #768
    KGROUP x5, 72, x6, 0

    add     x6, x4, #1024
    KGROUP x5, 96, x6, 0

    add     x6, x4, #1280
    KGROUP x5, 120, x6, 0

    add     x6, x4, #1536
    KGROUP x5, 144, x6, 0

    add     x6, x4, #1792
    KGROUP x5, 168, x6, 0

    // Section 5: K-groups 40-47
    add     x5, x0, #960

    add     x4, x4, #2048          // x4 now = x1 + 10240
    KGROUP x5, 0, x4, 0

    add     x6, x4, #256
    KGROUP x5, 24, x6, 0

    add     x6, x4, #512
    KGROUP x5, 48, x6, 0

    add     x6, x4, #768
    KGROUP x5, 72, x6, 0

    add     x6, x4, #1024
    KGROUP x5, 96, x6, 0

    add     x6, x4, #1280
    KGROUP x5, 120, x6, 0

    add     x6, x4, #1536
    KGROUP x5, 144, x6, 0

    add     x6, x4, #1792
    KGROUP x5, 168, x6, 0

    // Section 6: K-groups 48-55
    add     x5, x0, #1152

    add     x4, x4, #2048          // x4 now = x1 + 12288
    KGROUP x5, 0, x4, 0

    add     x6, x4, #256
    KGROUP x5, 24, x6, 0

    add     x6, x4, #512
    KGROUP x5, 48, x6, 0

    add     x6, x4, #768
    KGROUP x5, 72, x6, 0

    add     x6, x4, #1024
    KGROUP x5, 96, x6, 0

    add     x6, x4, #1280
    KGROUP x5, 120, x6, 0

    add     x6, x4, #1536
    KGROUP x5, 144, x6, 0

    add     x6, x4, #1792
    KGROUP x5, 168, x6, 0

    // Section 7: K-groups 56-63
    add     x5, x0, #1344

    add     x4, x4, #2048          // x4 now = x1 + 14336
    KGROUP x5, 0, x4, 0

    add     x6, x4, #256
    KGROUP x5, 24, x6, 0

    add     x6, x4, #512
    KGROUP x5, 48, x6, 0

    add     x6, x4, #768
    KGROUP x5, 72, x6, 0

    add     x6, x4, #1024
    KGROUP x5, 96, x6, 0

    add     x6, x4, #1280
    KGROUP x5, 120, x6, 0

    add     x6, x4, #1536
    KGROUP x5, 144, x6, 0

    add     x6, x4, #1792
    KGROUP x5, 168, x6, 0

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

    .size kernel_6x4_unroll, .-kernel_6x4_unroll
