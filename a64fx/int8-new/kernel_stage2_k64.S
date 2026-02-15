// kernel_stage2_k64.S
// Second stage kernel for fused GEMM: O += S @ C
// MR=6 rows, NR=64 columns, K=64 (fixed)
//
// S_tile: [6, 64] int8 packed as [K/4][MR][4] = [16][6][4]
// Cpack:  [64, 64] int8 packed as [K/4][4 vectors][16 lanes][4]
// O_acc:  [6, 64] int32 output (accumulated, not zeroed)

    .arch armv8.2-a+sve
    .text
    .align 6

    .global kernel_stage2_k64
    .type kernel_stage2_k64, @function

// Arguments:
//   x0 = S_tile pointer [6, 64] int8, packed [16][6][4]
//   x1 = Cpack pointer [64, 64] int8, packed [16][4][16][4]
//   x2 = O_acc pointer [6, 64] int32 output
//   x3 = ldo in bytes (stride for O)

// Macro for one K-group (processes 4 K values)
.macro KGROUP_K64 S_BASE, S_OFF, C_BASE, C_IDX
    // Load 6 rows of S (4 bytes each, broadcast to all lanes)
    ld1rw   {z24.s}, p0/z, [\S_BASE, #\S_OFF]
    ld1rw   {z25.s}, p0/z, [\S_BASE, #\S_OFF+4]
    ld1rw   {z26.s}, p0/z, [\S_BASE, #\S_OFF+8]
    ld1rw   {z27.s}, p0/z, [\S_BASE, #\S_OFF+12]
    ld1rw   {z28.s}, p0/z, [\S_BASE, #\S_OFF+16]
    ld1rw   {z29.s}, p0/z, [\S_BASE, #\S_OFF+20]

    // Load C vectors (4 vectors × 64 bytes each)
    ld1b    {z30.b}, p0/z, [\C_BASE, #\C_IDX, mul vl]
    ld1b    {z31.b}, p0/z, [\C_BASE, #\C_IDX+1, mul vl]

    // 12 SDOTs for columns 0-1
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

    // Load next 2 C vectors
    ld1b    {z30.b}, p0/z, [\C_BASE, #\C_IDX+2, mul vl]
    ld1b    {z31.b}, p0/z, [\C_BASE, #\C_IDX+3, mul vl]

    // 12 SDOTs for columns 2-3
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

kernel_stage2_k64:
    // Setup predicate
    ptrue   p0.b

    // Load existing accumulators (O_acc is accumulated, not zeroed)
    ptrue   p1.s
    ld1w    {z0.s}, p1/z, [x2, #0, mul vl]
    ld1w    {z1.s}, p1/z, [x2, #1, mul vl]
    ld1w    {z2.s}, p1/z, [x2, #2, mul vl]
    ld1w    {z3.s}, p1/z, [x2, #3, mul vl]
    add     x4, x2, x3
    ld1w    {z4.s}, p1/z, [x4, #0, mul vl]
    ld1w    {z5.s}, p1/z, [x4, #1, mul vl]
    ld1w    {z6.s}, p1/z, [x4, #2, mul vl]
    ld1w    {z7.s}, p1/z, [x4, #3, mul vl]
    add     x4, x4, x3
    ld1w    {z8.s}, p1/z, [x4, #0, mul vl]
    ld1w    {z9.s}, p1/z, [x4, #1, mul vl]
    ld1w    {z10.s}, p1/z, [x4, #2, mul vl]
    ld1w    {z11.s}, p1/z, [x4, #3, mul vl]
    add     x4, x4, x3
    ld1w    {z12.s}, p1/z, [x4, #0, mul vl]
    ld1w    {z13.s}, p1/z, [x4, #1, mul vl]
    ld1w    {z14.s}, p1/z, [x4, #2, mul vl]
    ld1w    {z15.s}, p1/z, [x4, #3, mul vl]
    add     x4, x4, x3
    ld1w    {z16.s}, p1/z, [x4, #0, mul vl]
    ld1w    {z17.s}, p1/z, [x4, #1, mul vl]
    ld1w    {z18.s}, p1/z, [x4, #2, mul vl]
    ld1w    {z19.s}, p1/z, [x4, #3, mul vl]
    add     x4, x4, x3
    ld1w    {z20.s}, p1/z, [x4, #0, mul vl]
    ld1w    {z21.s}, p1/z, [x4, #1, mul vl]
    ld1w    {z22.s}, p1/z, [x4, #2, mul vl]
    ld1w    {z23.s}, p1/z, [x4, #3, mul vl]

    // K=64 means 16 K-groups (64/4 = 16)
    // S layout: [16][6][4] = 384 bytes total
    // Each K-group: 6 rows × 4 bytes = 24 bytes S offset
    // C layout: [16][4 vectors] = 16 × 256 bytes = 4096 bytes total
    // Each K-group: 4 vectors × 64 bytes = 256 bytes C offset

    // K-groups 0-7 (S offsets 0-168, C offsets 0-1792)
    KGROUP_K64 x0, 0, x1, 0
    add     x4, x1, #256
    KGROUP_K64 x0, 24, x4, 0
    add     x4, x1, #512
    KGROUP_K64 x0, 48, x4, 0
    add     x4, x1, #768
    KGROUP_K64 x0, 72, x4, 0
    add     x4, x1, #1024
    KGROUP_K64 x0, 96, x4, 0
    add     x4, x1, #1280
    KGROUP_K64 x0, 120, x4, 0
    add     x4, x1, #1536
    KGROUP_K64 x0, 144, x4, 0
    add     x4, x1, #1792
    KGROUP_K64 x0, 168, x4, 0

    // K-groups 8-15 (S offsets 192-360, C offsets 2048-3840)
    add     x5, x0, #192
    add     x4, x1, #2048
    KGROUP_K64 x5, 0, x4, 0
    add     x4, x1, #2304
    KGROUP_K64 x5, 24, x4, 0
    add     x4, x1, #2560
    KGROUP_K64 x5, 48, x4, 0
    add     x4, x1, #2816
    KGROUP_K64 x5, 72, x4, 0
    add     x4, x1, #3072
    KGROUP_K64 x5, 96, x4, 0
    add     x4, x1, #3328
    KGROUP_K64 x5, 120, x4, 0
    add     x4, x1, #3584
    KGROUP_K64 x5, 144, x4, 0
    add     x4, x1, #3840
    KGROUP_K64 x5, 168, x4, 0

    // Store results back
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

    .size kernel_stage2_k64, .-kernel_stage2_k64
