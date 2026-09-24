/* Software-pipelined pair kernels for q38d groups (A64FX, 512-bit SVE).
 *
 * void q38d_asm_<fmt>_<arith>(const uint8_t *code, const uint8_t *high,
 *                             const uint8_t *scale, const int8_t *actq,
 *                             const float *actsc, long np, float *acc16,
 *                             const int8_t *lut);
 * Writes the 16-lane FP32 accumulator (lane 2r+u) to acc16; the caller adds
 * lane pairs and applies the format exponent. np >= 3.
 *
 * Four stages, one pair per iteration, in program order
 *   S4(i-3): combine partial dots, SCVTF, scale decode, FMLA
 *   S3(i-2): activation broadcasts, SDOT partials (MOVPRFX from zero)
 *   S2(i-1): nibble split (vector-mask AND), 6-bit assembly, TBL decode
 *   S1(i)  : code (and high-plane) loads
 * so every instruction's inputs were produced one iteration earlier.
 *
 * Registers
 *   z31 lut, z30 0x0f mask, z29 zero, z28 0x30 mask, z27 0xc0 mask, z26 acc
 *   S1->S2: z0,z1 codes, z2 high plane (F6: 64 bytes for both vectors)
 *   S2->S3: z4,z5,z6,z7 = l0,h0,l1,h1
 *   S3->S4: z8..z11 partials, z12 scale bytes
 *   temporaries: z16..z23
 *   pointers: x0 code/high(i), x1 high(i), x2 scale(i-2), x3 act(i-2),
 *             x4 actsc(i-3), x5 counter
 */
#define ZL z31
#define M0F z30
#define ZERO z29
#define M30 z28
#define MC0 z27
#define ACC z26

    .text
    .arch armv8.2-a+sve

#ifndef PF1_DIST
#define PF1_DIST 1024
#endif
#ifndef PF2_DIST
#define PF2_DIST 16384
#endif
.macro S1 f6, c0=z0, c1=z1, hp=z2, c3=z28
.if \f6 == 4
#if PF1_DIST
    prfm    pldl1keep, [x0, #PF1_DIST]
#endif
#if PF2_DIST
    prfm    pldl2keep, [x0, x9]
#endif
    ld1b    {\c0\().b}, p0/z, [x0]
    ld1b    {\c1\().b}, p0/z, [x0, #1, mul vl]
    ld1b    {\hp\().b}, p0/z, [x0, #2, mul vl]
    ld1b    {\c3\().b}, p0/z, [x0, #3, mul vl]
    add     x0, x0, #256
.exitm
.endif
#if PF1_DIST
    prfm    pldl1keep, [x0, #PF1_DIST]
#endif
#if PF2_DIST
    prfm    pldl2keep, [x0, x9]
#endif
    ld1b    {\c0\().b}, p0/z, [x0]
    ld1b    {\c1\().b}, p0/z, [x0, #1, mul vl]
.if \f6 == 1 || \f6 == 2
    /* 6-bit formats: {low 128 | high 64} per pair in one stream */
    ld1b    {\hp\().b}, p0/z, [x0, #2, mul vl]
    add     x0, x0, #192
.else
    add     x0, x0, #128
.endif
.endm

.macro S2 f6, c0=z0, c1=z1, hp=z2, c3=z28
.if \f6 == 4
    mov     z4.d, \c0\().d
    mov     z5.d, \c1\().d
    mov     z6.d, \hp\().d
    mov     z7.d, \c3\().d
.exitm
.endif
.if \f6 == 1 || \f6 == 2
    /* one 64-byte high plane per pair: fields for vector 0 at [1:0]/[3:2],
     * vector 1 at [5:4]/[7:6]; one shift and a 0x30 mask each */
    lsl     z16.b, \hp\().b, #4
    lsl     z17.b, \hp\().b, #2
    lsr     z18.b, \hp\().b, #2
    and     z16.d, z16.d, M30.d
    and     z17.d, z17.d, M30.d
    and     z19.d, \hp\().d, M30.d
    and     z18.d, z18.d, M30.d
    and     z20.d, \c0\().d, M0F.d
    lsr     z21.b, \c0\().b, #4
    and     z22.d, \c1\().d, M0F.d
    lsr     z23.b, \c1\().b, #4
    orr     z20.d, z20.d, z16.d
    orr     z21.d, z21.d, z17.d
    orr     z22.d, z22.d, z19.d
    orr     z23.d, z23.d, z18.d
    tbl     z4.b, {ZL.b}, z20.b
    tbl     z5.b, {ZL.b}, z21.b
    tbl     z6.b, {ZL.b}, z22.b
    tbl     z7.b, {ZL.b}, z23.b
.else
    and     z19.d, \c0\().d, M0F.d
    lsr     z20.b, \c0\().b, #4
    and     z21.d, \c1\().d, M0F.d
    lsr     z22.b, \c1\().b, #4
#ifdef ABL_NOTBL
    mov     z4.d, z19.d
    mov     z5.d, z20.d
    mov     z6.d, z21.d
    mov     z7.d, z22.d
#else
    tbl     z4.b, {ZL.b}, z19.b
    tbl     z5.b, {ZL.b}, z20.b
    tbl     z6.b, {ZL.b}, z21.b
    tbl     z7.b, {ZL.b}, z22.b
#endif
.endif
.endm

.macro S3 f6, a16
.if \f6 == 3
    ld1w    {z12.s}, p3/z, [x2]             /* 8 float scales (one per row) */
    ld1w    {z13.s}, p3/z, [x15]            /* 8 float offsets */
    add     x2, x2, #32
    add     x15, x15, #32
.endif
#ifndef ABL_NOACT
    ld1rd   {z16.d}, p2/z, [x3]
    ld1rd   {z17.d}, p2/z, [x3, #8]
    ld1rd   {z18.d}, p2/z, [x3, #16]
    ld1rd   {z19.d}, p2/z, [x3, #24]
#endif
.if \a16
    ld1rd   {z20.d}, p2/z, [x3, #32]
    ld1rd   {z21.d}, p2/z, [x3, #40]
    ld1rd   {z22.d}, p2/z, [x3, #48]
    ld1rd   {z23.d}, p2/z, [x3, #56]
    movprfx z8, ZERO
    sdot    z8.s, z4.b, z16.b
    movprfx z9, ZERO
    sdot    z9.s, z5.b, z17.b
    movprfx z10, ZERO
    sdot    z10.s, z4.b, z20.b
    movprfx z11, ZERO
    sdot    z11.s, z5.b, z21.b
    sdot    z8.s, z6.b, z18.b
    sdot    z9.s, z7.b, z19.b
    sdot    z10.s, z6.b, z22.b
    sdot    z11.s, z7.b, z23.b
    add     x3, x3, #64
.else
    movprfx z8, ZERO
    sdot    z8.s, z4.b, z16.b
    movprfx z9, ZERO
    sdot    z9.s, z5.b, z17.b
    movprfx z10, ZERO
    sdot    z10.s, z6.b, z18.b
    movprfx z11, ZERO
    sdot    z11.s, z7.b, z19.b
    add     x3, x3, #32
.endif
#if PF2_DIST
    prfm    pldl2keep, [x2, x11]
#endif
.if \f6 == 3
.elseif \f6 == 2 || \f6 == 4
    ld1sb   {z12.s}, p1/z, [x2]         /* 16 int8 Q6_K sub-scales, lane 2r+u */
    add     x2, x2, #16
    lsr     x15, x14, #3
    add     x14, x14, #1
    lsl     x15, x15, #3
    ld1w    {z13.s}, p3/z, [x13, x15, lsl #2]   /* d[r] for this 256-column block */
.elseif \f6
    ld1b    {z12.d}, p2/z, [x2]         /* 8 E8M0 bytes, one per 64-bit lane */
    add     x2, x2, #8
.else
    ld1b    {z12.s}, p1/z, [x2]         /* 16 UE4M3 bytes, lane 2r+u */
    add     x2, x2, #16
.endif
.endm

.macro S4 f6, a16
.if \a16
    add     z8.s, z8.s, z9.s
    add     z10.s, z10.s, z11.s
    lsl     z10.s, z10.s, #8
    add     z8.s, z8.s, z10.s
.else
    add     z8.s, z8.s, z9.s
    add     z10.s, z10.s, z11.s
    add     z8.s, z8.s, z10.s
.endif
    scvtf   z8.s, p1/m, z8.s
#ifdef ABL_NOSCALE
    fmla    ACC.s, p1/m, z8.s, z8.s
.endm
.macro S4_unused f6, a16
#endif
    ld1rd   {z9.d}, p2/z, [x4]
    add     x4, x4, #8
.if \f6 == 2
    zip1    z13.s, z13.s, z13.s
    scvtf   z12.s, p1/m, z12.s
    fmul    z12.s, z12.s, z13.s
.elseif \f6
    /* both 32-bit halves of each 64-bit lane get the E8M0 byte << 23 */
    lsl     z10.d, z12.d, #23
    lsl     z12.d, z12.d, #55
    orr     z12.d, z12.d, z10.d
.else
    lsl     z12.s, z12.s, #20
.endif
    fmul    z12.s, z12.s, z9.s
    fmla    ACC.s, p1/m, z8.s, z12.s
.endm

.macro KERNEL name, f6, a16
    .global \name
    .type \name, %function
    .p2align 6
\name:
    stp     d8, d9, [sp, #-64]!
    stp     d10, d11, [sp, #16]
    stp     d12, d13, [sp, #32]
    stp     d14, d15, [sp, #48]
    ptrue   p0.b
    ptrue   p1.s
    ptrue   p2.d
    ptrue   p3.s, vl8
    ld1b    {ZL.b}, p0/z, [x7]
.if \f6 == 2
    add     x13, x2, x5, lsl #4         /* d stream follows np*16 scale bytes */
    mov     x14, #0
.endif
    dup     M0F.b, #15
    dup     ZERO.s, #0
    dup     M30.b, #0x30
    dup     MC0.b, #-64
    dup     ACC.s, #0
    sub     x5, x5, #3
    mov     x9, #PF2_DIST
    lsr     x10, x9, #1
.if \f6 == 1
    lsr     x11, x9, #4
.else
    lsr     x11, x9, #3
.endif
    /* prologue */
    S1 \f6
    S2 \f6
    S1 \f6
    S3 \f6, \a16
    S2 \f6
    S1 \f6
    cbz     x5, 2f
1:
    S4 \f6, \a16
    S3 \f6, \a16
    S2 \f6
    S1 \f6
    subs    x5, x5, #1
    b.ne    1b
2:
    /* epilogue */
    S4 \f6, \a16
    S3 \f6, \a16
    S2 \f6
    S4 \f6, \a16
    S3 \f6, \a16
    S4 \f6, \a16
    st1w    {ACC.s}, p1, [x6]
    ldp     d10, d11, [sp, #16]
    ldp     d12, d13, [sp, #32]
    ldp     d14, d15, [sp, #48]
    ldp     d8, d9, [sp], #64
    ret
    .size \name, .-\name
.endm

    KERNEL q38d_asm_f4_a8, 0, 0
    KERNEL q38d_asm_f4_a16, 0, 1
    KERNEL q38d_asm_f6_a8, 1, 0
    KERNEL q38d_asm_f6_a16, 1, 1
    KERNEL q38d_asm_q6k_a8, 2, 0
    KERNEL q38d_asm_q6k_a16, 2, 1


.macro KERNEL2 name, f6, a16
    .global \name
    .type \name, %function
    .p2align 6
\name:
    stp     d8, d9, [sp, #-64]!
    stp     d10, d11, [sp, #16]
    stp     d12, d13, [sp, #32]
    stp     d14, d15, [sp, #48]
    ptrue   p0.b
    ptrue   p1.s
    ptrue   p2.d
    ptrue   p3.s, vl8
    ld1b    {ZL.b}, p0/z, [x7]
.if \f6 == 2
    add     x13, x2, x5, lsl #4
    mov     x14, #0
.endif
    dup     M0F.b, #15
    dup     ZERO.s, #0
    dup     M30.b, #0x30
    dup     MC0.b, #-64
    dup     ACC.s, #0
    mov     x9, #PF2_DIST
    lsr     x10, x9, #1
.if \f6 == 1
    lsr     x11, x9, #4
.else
    lsr     x11, x9, #3
.endif
    /* slots j: S4(j-4) S3(j-3) S2(j-2) S1(j); even j uses z0/z1/z2, odd z14/z15/z24 */
    S1 \f6, z0, z1, z2                       /* j=0 */
    S1 \f6, z14, z15, z24                    /* j=1 */
    S2 \f6, z0, z1, z2                       /* j=2: S2(0) */
    S1 \f6, z0, z1, z2
    S3 \f6, \a16                             /* j=3: S3(0) S2(1) S1(3) */
    S2 \f6, z14, z15, z24
    S1 \f6, z14, z15, z24
    sub     x5, x5, #4
    lsr     x5, x5, #1
    cbz     x5, 2f
1:
    S4 \f6, \a16                             /* even slot */
    S3 \f6, \a16
    S2 \f6, z0, z1, z2
    S1 \f6, z0, z1, z2
    S4 \f6, \a16                             /* odd slot */
    S3 \f6, \a16
    S2 \f6, z14, z15, z24
    S1 \f6, z14, z15, z24
    subs    x5, x5, #1
    b.ne    1b
2:
    S4 \f6, \a16
    S3 \f6, \a16
    S2 \f6, z0, z1, z2
    S4 \f6, \a16
    S3 \f6, \a16
    S2 \f6, z14, z15, z24
    S4 \f6, \a16
    S3 \f6, \a16
    S4 \f6, \a16
    st1w    {ACC.s}, p1, [x6]
    ldp     d10, d11, [sp, #16]
    ldp     d12, d13, [sp, #32]
    ldp     d14, d15, [sp, #48]
    ldp     d8, d9, [sp], #64
    ret
    .size \name, .-\name
.endm


/* F4 only: codes loaded four pairs ahead (four register sets).
 * Slot t runs S4(t-6) S3(t-5) S2(t-4) S1(t); pair t uses set t%4:
 * set0 z0/z1, set1 z14/z15, set2 z24/z25, set3 z3/z13. np % 4 == 0, np >= 8. */
#define SET0 z0, z1, z2
#define SET1 z14, z15, z2
#define SET2 z24, z25, z2
#define SET3 z3, z13, z2
.macro SLOT a16, c0, c1, hp
    S4 0, \a16
    S3 0, \a16
    S2 0, \c0, \c1, \hp
    S1 0, \c0, \c1, \hp
.endm
.macro KERNEL4 name, a16
    .global \name
    .type \name, %function
    .p2align 6
\name:
    stp     d8, d9, [sp, #-64]!
    stp     d10, d11, [sp, #16]
    stp     d12, d13, [sp, #32]
    stp     d14, d15, [sp, #48]
    ptrue   p0.b
    ptrue   p1.s
    ptrue   p2.d
    ptrue   p3.s, vl8
    ld1b    {ZL.b}, p0/z, [x7]
    dup     M0F.b, #15
    dup     ZERO.s, #0
    dup     M30.b, #0x30
    dup     MC0.b, #-64
    dup     ACC.s, #0
    mov     x9, #PF2_DIST
    lsr     x10, x9, #1
    lsr     x11, x9, #3
    S1 0, SET0                              /* t=0 */
    S1 0, SET1                              /* t=1 */
    S1 0, SET2                              /* t=2 */
    S1 0, SET3                              /* t=3 */
    S2 0, SET0                              /* t=4: S2(0) S1(4) */
    S1 0, SET0
    S3 0, \a16                              /* t=5: S3(0) S2(1) S1(5) */
    S2 0, SET1
    S1 0, SET1
    sub     x5, x5, #8
    lsr     x5, x5, #2
    cbz     x5, 2f
1:
    SLOT \a16, SET2
    SLOT \a16, SET3
    SLOT \a16, SET0
    SLOT \a16, SET1
    subs    x5, x5, #1
    b.ne    1b
2:
    SLOT \a16, SET2                         /* t=np-2 */
    SLOT \a16, SET3                         /* t=np-1 */
    S4 0, \a16                              /* t=np */
    S3 0, \a16
    S2 0, SET0
    S4 0, \a16                              /* t=np+1 */
    S3 0, \a16
    S2 0, SET1
    S4 0, \a16                              /* t=np+2 */
    S3 0, \a16
    S2 0, SET2
    S4 0, \a16                              /* t=np+3 */
    S3 0, \a16
    S2 0, SET3
    S4 0, \a16                              /* t=np+4 */
    S3 0, \a16
    S4 0, \a16                              /* t=np+5 */
    st1w    {ACC.s}, p1, [x6]
    ldp     d10, d11, [sp, #16]
    ldp     d12, d13, [sp, #32]
    ldp     d14, d15, [sp, #48]
    ldp     d8, d9, [sp], #64
    ret
    .size \name, .-\name
.endm

    KERNEL4 q38d_asm4_f4_a8, 0
    KERNEL4 q38d_asm4_f4_a16, 1
    KERNEL2 q38d_asm2_f4_a8, 0, 0
    KERNEL2 q38d_asm2_f4_a16, 0, 1
    KERNEL2 q38d_asm2_f6_a8, 1, 0
    KERNEL2 q38d_asm2_f6_a16, 1, 1
    KERNEL2 q38d_asm2_q6k_a8, 2, 0
    KERNEL2 q38d_asm2_q6k_a16, 2, 1


#define OFFACC z2   /* Q4K only: 4-bit formats never load a high plane into z2 */
/* Split combine: S4X (integer combine + scale) and S5X (SCVTF + FMLA) so no
 * stage carries the 30-cycle add->lsl->add->scvtf->fmla chain at the commit
 * head. d -> z3, s -> z25. Slot t: S5X(t-5) S4X(t-4) S3(t-3) S2(t-2) S1(t),
 * codes loaded two pairs ahead (sets as in KERNEL2). */
.macro S4X f6, a16
.if \a16
    add     z8.s, z8.s, z9.s
    add     z10.s, z10.s, z11.s
    lsl     z10.s, z10.s, #8
    add     z3.s, z8.s, z10.s
.else
    add     z8.s, z8.s, z9.s
    add     z10.s, z10.s, z11.s
    add     z3.s, z8.s, z10.s
.endif
    ld1rd   {z9.d}, p2/z, [x4]
    add     x4, x4, #8
.if \f6 == 3
    zip1    z12.s, z12.s, z12.s
    zip1    z13.s, z13.s, z13.s
    ld1rd   {z10.d}, p2/z, [x16]            /* scaled digit sums of this pair */
    add     x16, x16, #8
    fmla    OFFACC.s, p1/m, z13.s, z10.s
.elseif \f6 == 2 || \f6 == 4
    zip1    z13.s, z13.s, z13.s
    scvtf   z12.s, p1/m, z12.s
    fmul    z12.s, z12.s, z13.s
.elseif \f6
    lsl     z10.d, z12.d, #23
    lsl     z12.d, z12.d, #55
    orr     z12.d, z12.d, z10.d
.else
    lsl     z12.s, z12.s, #20
.endif
    fmul    z25.s, z12.s, z9.s
.endm
.macro S5X
    scvtf   z3.s, p1/m, z3.s
    fmla    ACC.s, p1/m, z3.s, z25.s
.endm
.macro KERNEL5 name, f6, a16
    .global \name
    .type \name, %function
    .p2align 6
\name:
.if \f6 == 3
    ldr     x16, [sp]                       /* 9th argument: activation digit sums */
    add     x15, x2, x5, lsl #5             /* offsets follow np*8 float scales */
.endif
    stp     d8, d9, [sp, #-64]!
    stp     d10, d11, [sp, #16]
    stp     d12, d13, [sp, #32]
    stp     d14, d15, [sp, #48]
    ptrue   p0.b
    ptrue   p1.s
    ptrue   p2.d
    ptrue   p3.s, vl8
    ld1b    {ZL.b}, p0/z, [x7]
.if \f6 == 2 || \f6 == 4
    add     x13, x2, x5, lsl #4
    mov     x14, #0
.endif
    dup     M0F.b, #15
    dup     ZERO.s, #0
.if \f6 == 1 || \f6 == 2
    dup     M30.b, #0x30
    dup     MC0.b, #-64
.endif
    dup     ACC.s, #0
.if \f6 == 3
    dup     OFFACC.s, #0
.endif
    mov     x9, #PF2_DIST
.if \f6 == 4
    add     x9, x9, x9, lsr #1              /* 256 code bytes per pair: prefetch further */
.endif
    lsr     x10, x9, #1
.if \f6 == 1 || \f6 == 4
    lsr     x11, x9, #4
.elseif \f6 == 3
    lsr     x11, x9, #2
.else
    lsr     x11, x9, #3
.endif
    S1 \f6, z0, z1, z2, z28                       /* t=0 */
    S1 \f6, z14, z15, z24, z27                    /* t=1 */
    S2 \f6, z0, z1, z2, z28                       /* t=2 */
    S1 \f6, z0, z1, z2, z28
    S3 \f6, \a16                             /* t=3 */
    S2 \f6, z14, z15, z24, z27
    S1 \f6, z14, z15, z24, z27
    S4X \f6, \a16                            /* t=4 */
    S3 \f6, \a16
    S2 \f6, z0, z1, z2, z28
    S1 \f6, z0, z1, z2, z28
    S5X                                      /* t=5 */
    S4X \f6, \a16
    S3 \f6, \a16
    S2 \f6, z14, z15, z24, z27
    S1 \f6, z14, z15, z24, z27
    sub     x5, x5, #6
    lsr     x5, x5, #1
    cbz     x5, 2f
1:
    S5X
    S4X \f6, \a16
    S3 \f6, \a16
    S2 \f6, z0, z1, z2, z28
    S1 \f6, z0, z1, z2, z28
    S5X
    S4X \f6, \a16
    S3 \f6, \a16
    S2 \f6, z14, z15, z24, z27
    S1 \f6, z14, z15, z24, z27
    subs    x5, x5, #1
    b.ne    1b
2:
    S5X                                      /* t=np */
    S4X \f6, \a16
    S3 \f6, \a16
    S2 \f6, z0, z1, z2, z28
    S5X                                      /* t=np+1 */
    S4X \f6, \a16
    S3 \f6, \a16
    S2 \f6, z14, z15, z24, z27
    S5X                                      /* t=np+2 */
    S4X \f6, \a16
    S3 \f6, \a16
    S5X                                      /* t=np+3 */
    S4X \f6, \a16
    S5X                                      /* t=np+4 */
.if \f6 == 3
    fsub    ACC.s, ACC.s, OFFACC.s
.endif
    st1w    {ACC.s}, p1, [x6]
    ldp     d10, d11, [sp, #16]
    ldp     d12, d13, [sp, #32]
    ldp     d14, d15, [sp, #48]
    ldp     d8, d9, [sp], #64
    ret
    .size \name, .-\name
.endm

    KERNEL5 q38d_asm5_f4_a8, 0, 0
    KERNEL5 q38d_asm5_f4_a16, 0, 1
    KERNEL5 q38d_asm5_f6_a8, 1, 0
    KERNEL5 q38d_asm5_f6_a16, 1, 1
    KERNEL5 q38d_asm5_q6k_a8, 2, 0
    KERNEL5 q38d_asm5_q6k_a16, 2, 1
    KERNEL5 q38d_asm5_q4k_a8, 3, 0
    KERNEL5 q38d_asm5_q4k_a16, 3, 1
    KERNEL5 q38d_asm5_q8k_a8, 4, 0
    KERNEL5 q38d_asm5_q8k_a16, 4, 1


/* F4, two pairs (A, B) interleaved per slot. Slot t: S4D(t-3) S3D(t-2)
 * S2D(t-1) S1D(t). Registers:
 *   codes A z0,z1  B z2,z3 | decoded A z4-z7  B z14,z15,z24,z25
 *   partials A z8-z11  B z16-z19 | scale bytes A z12  B z13
 *   activation temps z20-z23 | ZL z31, M0F z30, ZERO z29, ACC z26, ACC2 z27 */
#define ACC2 z27
.macro S1D
#if PF1_DIST
    prfm    pldl1keep, [x0, #PF1_DIST]
#endif
#if PF2_DIST
    prfm    pldl2keep, [x0, x9]
#endif
    ld1b    {z0.b}, p0/z, [x0]
    ld1b    {z1.b}, p0/z, [x0, #1, mul vl]
    ld1b    {z2.b}, p0/z, [x0, #2, mul vl]
    ld1b    {z3.b}, p0/z, [x0, #3, mul vl]
    add     x0, x0, #256
.endm
.macro S2D
    and     z4.d, z0.d, M0F.d
    lsr     z5.b, z0.b, #4
    and     z14.d, z2.d, M0F.d
    lsr     z15.b, z2.b, #4
    and     z6.d, z1.d, M0F.d
    lsr     z7.b, z1.b, #4
    and     z24.d, z3.d, M0F.d
    lsr     z25.b, z3.b, #4
    tbl     z4.b, {ZL.b}, z4.b
    tbl     z14.b, {ZL.b}, z14.b
    tbl     z5.b, {ZL.b}, z5.b
    tbl     z15.b, {ZL.b}, z15.b
    tbl     z6.b, {ZL.b}, z6.b
    tbl     z24.b, {ZL.b}, z24.b
    tbl     z7.b, {ZL.b}, z7.b
    tbl     z25.b, {ZL.b}, z25.b
.endm
.macro S3D a16
#if PF2_DIST
    prfm    pldl2keep, [x2, x11]
#endif
.if \a16
    /* lo digits: first halves of A and B */
    ld1rd   {z20.d}, p2/z, [x3]
    ld1rd   {z21.d}, p2/z, [x3, #8]
    ld1rd   {z22.d}, p2/z, [x3, #64]
    ld1rd   {z23.d}, p2/z, [x3, #72]
    movprfx z8, ZERO
    sdot    z8.s, z4.b, z20.b
    movprfx z9, ZERO
    sdot    z9.s, z5.b, z21.b
    movprfx z16, ZERO
    sdot    z16.s, z14.b, z22.b
    movprfx z17, ZERO
    sdot    z17.s, z15.b, z23.b
    /* hi digits: first halves */
    ld1rd   {z20.d}, p2/z, [x3, #32]
    ld1rd   {z21.d}, p2/z, [x3, #40]
    ld1rd   {z22.d}, p2/z, [x3, #96]
    ld1rd   {z23.d}, p2/z, [x3, #104]
    movprfx z10, ZERO
    sdot    z10.s, z4.b, z20.b
    movprfx z11, ZERO
    sdot    z11.s, z5.b, z21.b
    movprfx z18, ZERO
    sdot    z18.s, z14.b, z22.b
    movprfx z19, ZERO
    sdot    z19.s, z15.b, z23.b
    /* lo digits: second halves */
    ld1rd   {z20.d}, p2/z, [x3, #16]
    ld1rd   {z21.d}, p2/z, [x3, #24]
    ld1rd   {z22.d}, p2/z, [x3, #80]
    ld1rd   {z23.d}, p2/z, [x3, #88]
    sdot    z8.s, z6.b, z20.b
    sdot    z9.s, z7.b, z21.b
    sdot    z16.s, z24.b, z22.b
    sdot    z17.s, z25.b, z23.b
    /* hi digits: second halves */
    ld1rd   {z20.d}, p2/z, [x3, #48]
    ld1rd   {z21.d}, p2/z, [x3, #56]
    ld1rd   {z22.d}, p2/z, [x3, #112]
    ld1rd   {z23.d}, p2/z, [x3, #120]
    sdot    z10.s, z6.b, z20.b
    sdot    z11.s, z7.b, z21.b
    sdot    z18.s, z24.b, z22.b
    sdot    z19.s, z25.b, z23.b
    add     x3, x3, #128
.else
    ld1rd   {z20.d}, p2/z, [x3]
    ld1rd   {z21.d}, p2/z, [x3, #8]
    ld1rd   {z22.d}, p2/z, [x3, #32]
    ld1rd   {z23.d}, p2/z, [x3, #40]
    movprfx z8, ZERO
    sdot    z8.s, z4.b, z20.b
    movprfx z9, ZERO
    sdot    z9.s, z5.b, z21.b
    movprfx z16, ZERO
    sdot    z16.s, z14.b, z22.b
    movprfx z17, ZERO
    sdot    z17.s, z15.b, z23.b
    ld1rd   {z20.d}, p2/z, [x3, #16]
    ld1rd   {z21.d}, p2/z, [x3, #24]
    ld1rd   {z22.d}, p2/z, [x3, #48]
    ld1rd   {z23.d}, p2/z, [x3, #56]
    movprfx z10, ZERO
    sdot    z10.s, z6.b, z20.b
    movprfx z11, ZERO
    sdot    z11.s, z7.b, z21.b
    movprfx z18, ZERO
    sdot    z18.s, z24.b, z22.b
    movprfx z19, ZERO
    sdot    z19.s, z25.b, z23.b
    add     x3, x3, #64
.endif
    ld1b    {z12.s}, p1/z, [x2]
    ld1b    {z13.s}, p1/z, [x2, #1, mul vl]
    add     x2, x2, #32
.endm
.macro S4D a16
    add     z8.s, z8.s, z9.s
    add     z16.s, z16.s, z17.s
    add     z10.s, z10.s, z11.s
    add     z18.s, z18.s, z19.s
.if \a16
    lsl     z10.s, z10.s, #8
    lsl     z18.s, z18.s, #8
.endif
    add     z8.s, z8.s, z10.s
    add     z16.s, z16.s, z18.s
    ld1rd   {z9.d}, p2/z, [x4]
    ld1rd   {z17.d}, p2/z, [x4, #8]
    add     x4, x4, #16
    scvtf   z8.s, p1/m, z8.s
    scvtf   z16.s, p1/m, z16.s
    lsl     z12.s, z12.s, #20
    lsl     z13.s, z13.s, #20
    fmul    z12.s, z12.s, z9.s
    fmul    z13.s, z13.s, z17.s
    fmla    ACC.s, p1/m, z8.s, z12.s
    fmla    ACC2.s, p1/m, z16.s, z13.s
.endm
.macro KERNEL6 name, a16
    .global \name
    .type \name, %function
    .p2align 6
\name:
    stp     d8, d9, [sp, #-64]!
    stp     d10, d11, [sp, #16]
    stp     d12, d13, [sp, #32]
    stp     d14, d15, [sp, #48]
    ptrue   p0.b
    ptrue   p1.s
    ptrue   p2.d
    ld1b    {ZL.b}, p0/z, [x7]
    dup     M0F.b, #15
    dup     ZERO.s, #0
    dup     ACC.s, #0
    dup     ACC2.s, #0
    mov     x9, #PF2_DIST
    lsr     x11, x9, #3
    lsr     x5, x5, #1                       /* slots of two pairs */
    S1D                                      /* t=0 */
    S2D                                      /* t=1 */
    S1D
    S3D \a16                                 /* t=2 */
    S2D
    S1D
    sub     x5, x5, #3
    cbz     x5, 2f
1:
    S4D \a16
    S3D \a16
    S2D
    S1D
    subs    x5, x5, #1
    b.ne    1b
2:
    S4D \a16
    S3D \a16
    S2D
    S4D \a16
    S3D \a16
    S4D \a16
    fadd    ACC.s, ACC.s, ACC2.s
    st1w    {ACC.s}, p1, [x6]
    ldp     d10, d11, [sp, #16]
    ldp     d12, d13, [sp, #32]
    ldp     d14, d15, [sp, #48]
    ldp     d8, d9, [sp], #64
    ret
    .size \name, .-\name
.endm

    KERNEL6 q38d_asm6_f4_a8, 0
    KERNEL6 q38d_asm6_f4_a16, 1


/* F4, two pairs per slot with the combine/convert chain split:
 * slot t: S5E(t-4) S4E(t-3) S3E(t-2) S2D(t-1) S1D(t).
 *   S3E: activation broadcasts (two temps z20/z21) + SDOT partials
 *   S4E: integer combine -> d (z22 A, z23 B); scale -> s (z12 A, z13 B)
 *   S5E: SCVTF + FMLA */
.macro S3E a16
.if \a16
    ld1rd   {z20.d}, p2/z, [x3]
    ld1rd   {z21.d}, p2/z, [x3, #64]
    movprfx z8, ZERO
    sdot    z8.s, z4.b, z20.b
    movprfx z16, ZERO
    sdot    z16.s, z14.b, z21.b
    ld1rd   {z20.d}, p2/z, [x3, #8]
    ld1rd   {z21.d}, p2/z, [x3, #72]
    movprfx z9, ZERO
    sdot    z9.s, z5.b, z20.b
    movprfx z17, ZERO
    sdot    z17.s, z15.b, z21.b
    ld1rd   {z20.d}, p2/z, [x3, #32]
    ld1rd   {z21.d}, p2/z, [x3, #96]
    movprfx z10, ZERO
    sdot    z10.s, z4.b, z20.b
    movprfx z18, ZERO
    sdot    z18.s, z14.b, z21.b
    ld1rd   {z20.d}, p2/z, [x3, #40]
    ld1rd   {z21.d}, p2/z, [x3, #104]
    movprfx z11, ZERO
    sdot    z11.s, z5.b, z20.b
    movprfx z19, ZERO
    sdot    z19.s, z15.b, z21.b
    ld1rd   {z20.d}, p2/z, [x3, #16]
    ld1rd   {z21.d}, p2/z, [x3, #80]
    sdot    z8.s, z6.b, z20.b
    sdot    z16.s, z24.b, z21.b
    ld1rd   {z20.d}, p2/z, [x3, #24]
    ld1rd   {z21.d}, p2/z, [x3, #88]
    sdot    z9.s, z7.b, z20.b
    sdot    z17.s, z25.b, z21.b
    ld1rd   {z20.d}, p2/z, [x3, #48]
    ld1rd   {z21.d}, p2/z, [x3, #112]
    sdot    z10.s, z6.b, z20.b
    sdot    z18.s, z24.b, z21.b
    ld1rd   {z20.d}, p2/z, [x3, #56]
    ld1rd   {z21.d}, p2/z, [x3, #120]
    sdot    z11.s, z7.b, z20.b
    sdot    z19.s, z25.b, z21.b
    add     x3, x3, #128
.else
    ld1rd   {z20.d}, p2/z, [x3]
    ld1rd   {z21.d}, p2/z, [x3, #32]
    movprfx z8, ZERO
    sdot    z8.s, z4.b, z20.b
    movprfx z16, ZERO
    sdot    z16.s, z14.b, z21.b
    ld1rd   {z20.d}, p2/z, [x3, #8]
    ld1rd   {z21.d}, p2/z, [x3, #40]
    movprfx z9, ZERO
    sdot    z9.s, z5.b, z20.b
    movprfx z17, ZERO
    sdot    z17.s, z15.b, z21.b
    ld1rd   {z20.d}, p2/z, [x3, #16]
    ld1rd   {z21.d}, p2/z, [x3, #48]
    movprfx z10, ZERO
    sdot    z10.s, z6.b, z20.b
    movprfx z18, ZERO
    sdot    z18.s, z24.b, z21.b
    ld1rd   {z20.d}, p2/z, [x3, #24]
    ld1rd   {z21.d}, p2/z, [x3, #56]
    movprfx z11, ZERO
    sdot    z11.s, z7.b, z20.b
    movprfx z19, ZERO
    sdot    z19.s, z25.b, z21.b
    add     x3, x3, #64
.endif
.endm
.macro S4E a16
#if PF2_DIST
    prfm    pldl2keep, [x2, x11]
#endif
    ld1b    {z12.s}, p1/z, [x2]
    ld1b    {z13.s}, p1/z, [x2, #1, mul vl]
    add     x2, x2, #32
    add     z8.s, z8.s, z9.s
    add     z16.s, z16.s, z17.s
    add     z10.s, z10.s, z11.s
    add     z18.s, z18.s, z19.s
.if \a16
    lsl     z10.s, z10.s, #8
    lsl     z18.s, z18.s, #8
.endif
    add     z22.s, z8.s, z10.s
    add     z23.s, z16.s, z18.s
    ld1rd   {z9.d}, p2/z, [x4]
    ld1rd   {z17.d}, p2/z, [x4, #8]
    add     x4, x4, #16
    lsl     z12.s, z12.s, #20
    lsl     z13.s, z13.s, #20
    fmul    z12.s, z12.s, z9.s
    fmul    z13.s, z13.s, z17.s
.endm
.macro S5E
    scvtf   z22.s, p1/m, z22.s
    scvtf   z23.s, p1/m, z23.s
    fmla    ACC.s, p1/m, z22.s, z12.s
    fmla    ACC2.s, p1/m, z23.s, z13.s
.endm
.macro KERNEL7 name, a16
    .global \name
    .type \name, %function
    .p2align 6
\name:
    stp     d8, d9, [sp, #-64]!
    stp     d10, d11, [sp, #16]
    stp     d12, d13, [sp, #32]
    stp     d14, d15, [sp, #48]
    ptrue   p0.b
    ptrue   p1.s
    ptrue   p2.d
    ld1b    {ZL.b}, p0/z, [x7]
    dup     M0F.b, #15
    dup     ZERO.s, #0
    dup     ACC.s, #0
    dup     ACC2.s, #0
    mov     x9, #PF2_DIST
    lsr     x11, x9, #3
    lsr     x5, x5, #1                       /* slots of two pairs */
    S1D                                      /* t=0 */
    S2D                                      /* t=1 */
    S1D
    S3E \a16                                 /* t=2 */
    S2D
    S1D
    S4E \a16                                 /* t=3 */
    S3E \a16
    S2D
    S1D
    sub     x5, x5, #4
    cbz     x5, 2f
1:
    S5E
    S4E \a16
    S3E \a16
    S2D
    S1D
    subs    x5, x5, #1
    b.ne    1b
2:
    S5E                                      /* t=ns */
    S4E \a16
    S3E \a16
    S2D
    S5E                                      /* t=ns+1 */
    S4E \a16
    S3E \a16
    S5E                                      /* t=ns+2 */
    S4E \a16
    S5E                                      /* t=ns+3 */
    fadd    ACC.s, ACC.s, ACC2.s
    st1w    {ACC.s}, p1, [x6]
    ldp     d10, d11, [sp, #16]
    ldp     d12, d13, [sp, #32]
    ldp     d14, d15, [sp, #48]
    ldp     d8, d9, [sp], #64
    ret
    .size \name, .-\name
.endm

    KERNEL7 q38d_asm7_f4_a8, 0
    KERNEL7 q38d_asm7_f4_a16, 1


/* Dual-matrix F4: one pair of matrix A and the same column pair of matrix B
 * per slot, sharing every activation broadcast. x0 codes A, x1 codes B,
 * x2 scales A, x8 scales B (9th argument), output acc32 = [A 16 | B 16]. */
.macro S1F
#if PF1_DIST
    prfm    pldl1keep, [x0, #PF1_DIST]
    prfm    pldl1keep, [x1, #PF1_DIST]
#endif
#if PF2_DIST
    prfm    pldl2keep, [x0, x9]
    prfm    pldl2keep, [x1, x9]
#endif
    ld1b    {z0.b}, p0/z, [x0]
    ld1b    {z1.b}, p0/z, [x0, #1, mul vl]
    ld1b    {z2.b}, p0/z, [x1]
    ld1b    {z3.b}, p0/z, [x1, #1, mul vl]
    add     x0, x0, #128
    add     x1, x1, #128
.endm
.macro S3F a16
.if \a16
    ld1rd   {z20.d}, p2/z, [x3]
    ld1rd   {z21.d}, p2/z, [x3, #8]
    movprfx z8, ZERO
    sdot    z8.s, z4.b, z20.b
    movprfx z16, ZERO
    sdot    z16.s, z14.b, z20.b
    movprfx z9, ZERO
    sdot    z9.s, z5.b, z21.b
    movprfx z17, ZERO
    sdot    z17.s, z15.b, z21.b
    ld1rd   {z20.d}, p2/z, [x3, #32]
    ld1rd   {z21.d}, p2/z, [x3, #40]
    movprfx z10, ZERO
    sdot    z10.s, z4.b, z20.b
    movprfx z18, ZERO
    sdot    z18.s, z14.b, z20.b
    movprfx z11, ZERO
    sdot    z11.s, z5.b, z21.b
    movprfx z19, ZERO
    sdot    z19.s, z15.b, z21.b
    ld1rd   {z20.d}, p2/z, [x3, #16]
    ld1rd   {z21.d}, p2/z, [x3, #24]
    sdot    z8.s, z6.b, z20.b
    sdot    z16.s, z24.b, z20.b
    sdot    z9.s, z7.b, z21.b
    sdot    z17.s, z25.b, z21.b
    ld1rd   {z20.d}, p2/z, [x3, #48]
    ld1rd   {z21.d}, p2/z, [x3, #56]
    sdot    z10.s, z6.b, z20.b
    sdot    z18.s, z24.b, z20.b
    sdot    z11.s, z7.b, z21.b
    sdot    z19.s, z25.b, z21.b
    add     x3, x3, #64
.else
    ld1rd   {z20.d}, p2/z, [x3]
    ld1rd   {z21.d}, p2/z, [x3, #8]
    movprfx z8, ZERO
    sdot    z8.s, z4.b, z20.b
    movprfx z16, ZERO
    sdot    z16.s, z14.b, z20.b
    movprfx z9, ZERO
    sdot    z9.s, z5.b, z21.b
    movprfx z17, ZERO
    sdot    z17.s, z15.b, z21.b
    ld1rd   {z20.d}, p2/z, [x3, #16]
    ld1rd   {z21.d}, p2/z, [x3, #24]
    movprfx z10, ZERO
    sdot    z10.s, z6.b, z20.b
    movprfx z18, ZERO
    sdot    z18.s, z24.b, z20.b
    movprfx z11, ZERO
    sdot    z11.s, z7.b, z21.b
    movprfx z19, ZERO
    sdot    z19.s, z25.b, z21.b
    add     x3, x3, #32
.endif
.endm
.macro S4F a16
#if PF2_DIST
    prfm    pldl2keep, [x2, x11]
    prfm    pldl2keep, [x8, x11]
#endif
    ld1b    {z12.s}, p1/z, [x2]
    ld1b    {z13.s}, p1/z, [x8]
    add     x2, x2, #16
    add     x8, x8, #16
    add     z8.s, z8.s, z9.s
    add     z16.s, z16.s, z17.s
    add     z10.s, z10.s, z11.s
    add     z18.s, z18.s, z19.s
.if \a16
    lsl     z10.s, z10.s, #8
    lsl     z18.s, z18.s, #8
.endif
    add     z22.s, z8.s, z10.s
    add     z23.s, z16.s, z18.s
    ld1rd   {z9.d}, p2/z, [x4]
    add     x4, x4, #8
    lsl     z12.s, z12.s, #20
    lsl     z13.s, z13.s, #20
    fmul    z12.s, z12.s, z9.s
    fmul    z13.s, z13.s, z9.s
.endm
.macro KERNEL8 name, a16
    .global \name
    .type \name, %function
    .p2align 6
\name:
    ldr     x8, [sp]
    stp     d8, d9, [sp, #-64]!
    stp     d10, d11, [sp, #16]
    stp     d12, d13, [sp, #32]
    stp     d14, d15, [sp, #48]
    ptrue   p0.b
    ptrue   p1.s
    ptrue   p2.d
    ld1b    {ZL.b}, p0/z, [x7]
    dup     M0F.b, #15
    dup     ZERO.s, #0
    dup     ACC.s, #0
    dup     ACC2.s, #0
    mov     x9, #PF2_DIST
    lsr     x11, x9, #3
    S1F                                      /* t=0 */
    S2D                                      /* t=1 */
    S1F
    S3F \a16                                 /* t=2 */
    S2D
    S1F
    S4F \a16                                 /* t=3 */
    S3F \a16
    S2D
    S1F
    sub     x5, x5, #4
    cbz     x5, 2f
1:
    S5E
    S4F \a16
    S3F \a16
    S2D
    S1F
    subs    x5, x5, #1
    b.ne    1b
2:
    S5E
    S4F \a16
    S3F \a16
    S2D
    S5E
    S4F \a16
    S3F \a16
    S5E
    S4F \a16
    S5E
    st1w    {ACC.s}, p1, [x6]
    st1w    {ACC2.s}, p1, [x6, #1, mul vl]
    ldp     d10, d11, [sp, #16]
    ldp     d12, d13, [sp, #32]
    ldp     d14, d15, [sp, #48]
    ldp     d8, d9, [sp], #64
    ret
    .size \name, .-\name
.endm

    KERNEL8 q38d_asm8_f4_a8, 0
    KERNEL8 q38d_asm8_f4_a16, 1


/* KERNEL7 with one SDOT chain per digit: partials z8 (A lo), z10 (A hi),
 * z16 (B lo), z18 (B hi); the per-pair combine is lsl+add (A16) or nothing
 * (A8). Chains are interleaved round-robin across the four accumulators. */
.macro S3H a16
    ld1rd   {z20.d}, p2/z, [x3]
    ld1rd   {z21.d}, p2/z, [x3, #(\a16 * 64 + (1 - \a16) * 32)]
    movprfx z8, ZERO
    sdot    z8.s, z4.b, z20.b
    movprfx z16, ZERO
    sdot    z16.s, z14.b, z21.b
.if \a16
    ld1rd   {z20.d}, p2/z, [x3, #32]
    ld1rd   {z21.d}, p2/z, [x3, #96]
    movprfx z10, ZERO
    sdot    z10.s, z4.b, z20.b
    movprfx z18, ZERO
    sdot    z18.s, z14.b, z21.b
.endif
    ld1rd   {z20.d}, p2/z, [x3, #8]
    ld1rd   {z21.d}, p2/z, [x3, #(\a16 * 72 + (1 - \a16) * 40)]
    sdot    z8.s, z5.b, z20.b
    sdot    z16.s, z15.b, z21.b
.if \a16
    ld1rd   {z20.d}, p2/z, [x3, #40]
    ld1rd   {z21.d}, p2/z, [x3, #104]
    sdot    z10.s, z5.b, z20.b
    sdot    z18.s, z15.b, z21.b
.endif
    ld1rd   {z20.d}, p2/z, [x3, #16]
    ld1rd   {z21.d}, p2/z, [x3, #(\a16 * 80 + (1 - \a16) * 48)]
    sdot    z8.s, z6.b, z20.b
    sdot    z16.s, z24.b, z21.b
.if \a16
    ld1rd   {z20.d}, p2/z, [x3, #48]
    ld1rd   {z21.d}, p2/z, [x3, #112]
    sdot    z10.s, z6.b, z20.b
    sdot    z18.s, z24.b, z21.b
.endif
    ld1rd   {z20.d}, p2/z, [x3, #24]
    ld1rd   {z21.d}, p2/z, [x3, #(\a16 * 88 + (1 - \a16) * 56)]
    sdot    z8.s, z7.b, z20.b
    sdot    z16.s, z25.b, z21.b
.if \a16
    ld1rd   {z20.d}, p2/z, [x3, #56]
    ld1rd   {z21.d}, p2/z, [x3, #120]
    sdot    z10.s, z7.b, z20.b
    sdot    z18.s, z25.b, z21.b
    add     x3, x3, #128
.else
    add     x3, x3, #64
.endif
.endm
.macro S4H a16
#if PF2_DIST
    prfm    pldl2keep, [x2, x11]
#endif
    ld1b    {z12.s}, p1/z, [x2]
    ld1b    {z13.s}, p1/z, [x2, #1, mul vl]
    add     x2, x2, #32
.if \a16
    lsl     z10.s, z10.s, #8
    lsl     z18.s, z18.s, #8
    add     z22.s, z8.s, z10.s
    add     z23.s, z16.s, z18.s
.else
    mov     z22.d, z8.d
    mov     z23.d, z16.d
.endif
    ld1rd   {z9.d}, p2/z, [x4]
    ld1rd   {z17.d}, p2/z, [x4, #8]
    add     x4, x4, #16
    lsl     z12.s, z12.s, #20
    lsl     z13.s, z13.s, #20
    fmul    z12.s, z12.s, z9.s
    fmul    z13.s, z13.s, z17.s
.endm
.macro KERNEL7C name, a16
    .global \name
    .type \name, %function
    .p2align 6
\name:
    stp     d8, d9, [sp, #-64]!
    stp     d10, d11, [sp, #16]
    stp     d12, d13, [sp, #32]
    stp     d14, d15, [sp, #48]
    ptrue   p0.b
    ptrue   p1.s
    ptrue   p2.d
    ld1b    {ZL.b}, p0/z, [x7]
    dup     M0F.b, #15
    dup     ZERO.s, #0
    dup     ACC.s, #0
    dup     ACC2.s, #0
    mov     x9, #PF2_DIST
    lsr     x11, x9, #3
    lsr     x5, x5, #1
    S1D
    S2D
    S1D
    S3H \a16
    S2D
    S1D
    S4H \a16
    S3H \a16
    S2D
    S1D
    sub     x5, x5, #4
    cbz     x5, 2f
1:
    S5E
    S4H \a16
    S3H \a16
    S2D
    S1D
    subs    x5, x5, #1
    b.ne    1b
2:
    S5E
    S4H \a16
    S3H \a16
    S2D
    S5E
    S4H \a16
    S3H \a16
    S5E
    S4H \a16
    S5E
    fadd    ACC.s, ACC.s, ACC2.s
    st1w    {ACC.s}, p1, [x6]
    ldp     d10, d11, [sp, #16]
    ldp     d12, d13, [sp, #32]
    ldp     d14, d15, [sp, #48]
    ldp     d8, d9, [sp], #64
    ret
    .size \name, .-\name
.endm

    KERNEL7C q38d_asm7c_f4_a8, 0
    KERNEL7C q38d_asm7c_f4_a16, 1


/* F6, two pairs (A, B) per slot. Slot t: S5E(t-3) S46(t-2) S3E(t-1) S12F(t).
 * S12F loads and decodes in one stage (L1 prefetch keeps codes resident);
 * temporaries z0-z3, z20, z21. Decoded A z4-z7, B z14,z15,z24,z25. */
.macro DEC6 dl, dh, dl2, dh2
    lsl     z3.b, z2.b, #4
    and     z3.d, z3.d, M30.d
    and     z20.d, z0.d, M0F.d
    orr     z20.d, z20.d, z3.d
    lsl     z3.b, z2.b, #2
    and     z3.d, z3.d, M30.d
    lsr     z21.b, z0.b, #4
    orr     z21.d, z21.d, z3.d
    tbl     \dl\().b, {ZL.b}, z20.b
    tbl     \dh\().b, {ZL.b}, z21.b
    and     z3.d, z2.d, M30.d
    and     z20.d, z1.d, M0F.d
    orr     z20.d, z20.d, z3.d
    lsr     z3.b, z2.b, #2
    and     z3.d, z3.d, M30.d
    lsr     z21.b, z1.b, #4
    orr     z21.d, z21.d, z3.d
    tbl     \dl2\().b, {ZL.b}, z20.b
    tbl     \dh2\().b, {ZL.b}, z21.b
.endm
.macro S12F
#if PF1_DIST
    prfm    pldl1keep, [x0, #PF1_DIST]
#endif
#if PF2_DIST
    prfm    pldl2keep, [x0, x9]
#endif
    ld1b    {z0.b}, p0/z, [x0]
    ld1b    {z1.b}, p0/z, [x0, #1, mul vl]
    ld1b    {z2.b}, p0/z, [x0, #2, mul vl]
    DEC6    z4, z5, z6, z7
    ld1b    {z0.b}, p0/z, [x0, #3, mul vl]
    ld1b    {z1.b}, p0/z, [x0, #4, mul vl]
    ld1b    {z2.b}, p0/z, [x0, #5, mul vl]
    DEC6    z14, z15, z24, z25
    add     x0, x0, #384
.endm
.macro S46 a16
#if PF2_DIST
    prfm    pldl2keep, [x2, x11]
#endif
    ld1b    {z12.d}, p2/z, [x2]
    ld1b    {z13.d}, p2/z, [x2, #1, mul vl]
    add     x2, x2, #16
    add     z8.s, z8.s, z9.s
    add     z16.s, z16.s, z17.s
    add     z10.s, z10.s, z11.s
    add     z18.s, z18.s, z19.s
.if \a16
    lsl     z10.s, z10.s, #8
    lsl     z18.s, z18.s, #8
.endif
    add     z22.s, z8.s, z10.s
    add     z23.s, z16.s, z18.s
    ld1rd   {z9.d}, p2/z, [x4]
    ld1rd   {z17.d}, p2/z, [x4, #8]
    add     x4, x4, #16
    lsl     z10.d, z12.d, #23
    lsl     z12.d, z12.d, #55
    lsl     z11.d, z13.d, #23
    lsl     z13.d, z13.d, #55
    orr     z12.d, z12.d, z10.d
    orr     z13.d, z13.d, z11.d
    fmul    z12.s, z12.s, z9.s
    fmul    z13.s, z13.s, z17.s
.endm
.macro KERNEL9 name, a16
    .global \name
    .type \name, %function
    .p2align 6
\name:
    stp     d8, d9, [sp, #-64]!
    stp     d10, d11, [sp, #16]
    stp     d12, d13, [sp, #32]
    stp     d14, d15, [sp, #48]
    ptrue   p0.b
    ptrue   p1.s
    ptrue   p2.d
    ld1b    {ZL.b}, p0/z, [x7]
    dup     M0F.b, #15
    dup     M30.b, #0x30
    dup     ZERO.s, #0
    dup     ACC.s, #0
    dup     ACC2.s, #0
    mov     x9, #PF2_DIST
    lsr     x10, x9, #1
    lsr     x11, x9, #4
    lsr     x5, x5, #1                       /* slots of two pairs */
    S12F                                     /* t=0 */
    S3E \a16                                 /* t=1 */
    S12F
    S46 \a16                                 /* t=2 */
    S3E \a16
    S12F
    sub     x5, x5, #3
    cbz     x5, 2f
1:
    S5E
    S46 \a16
    S3E \a16
    S12F
    subs    x5, x5, #1
    b.ne    1b
2:
    S5E
    S46 \a16
    S3E \a16
    S5E
    S46 \a16
    S5E
    fadd    ACC.s, ACC.s, ACC2.s
    st1w    {ACC.s}, p1, [x6]
    ldp     d10, d11, [sp, #16]
    ldp     d12, d13, [sp, #32]
    ldp     d14, d15, [sp, #48]
    ldp     d8, d9, [sp], #64
    ret
    .size \name, .-\name
.endm

    KERNEL9 q38d_asm9_f6_a8, 0
    KERNEL9 q38d_asm9_f6_a16, 1


/* Dual-matrix F6 (e.g. FFN gate+up): one pair of A and of B per slot, codes
 * loaded and decoded in the same stage. Slot t: S5E(t-3) S4G(t-2) S3F(t-1)
 * S12D(t). x0 codes A, x1 codes B (192-byte records), x2 scales A, x8 scales
 * B (9th argument), output [A 16 | B 16]. */
.macro S12D
#if PF1_DIST
    prfm    pldl1keep, [x0, #PF1_DIST]
    prfm    pldl1keep, [x1, #PF1_DIST]
#endif
#if PF2_DIST
    prfm    pldl2keep, [x0, x9]
    prfm    pldl2keep, [x1, x9]
#endif
    ld1b    {z0.b}, p0/z, [x0]
    ld1b    {z1.b}, p0/z, [x0, #1, mul vl]
    ld1b    {z2.b}, p0/z, [x0, #2, mul vl]
    DEC6    z4, z5, z6, z7
    ld1b    {z0.b}, p0/z, [x1]
    ld1b    {z1.b}, p0/z, [x1, #1, mul vl]
    ld1b    {z2.b}, p0/z, [x1, #2, mul vl]
    DEC6    z14, z15, z24, z25
    add     x0, x0, #192
    add     x1, x1, #192
.endm
.macro S4G a16
#if PF2_DIST
    prfm    pldl2keep, [x2, x11]
    prfm    pldl2keep, [x8, x11]
#endif
    ld1b    {z12.d}, p2/z, [x2]
    ld1b    {z13.d}, p2/z, [x8]
    add     x2, x2, #8
    add     x8, x8, #8
    add     z8.s, z8.s, z9.s
    add     z16.s, z16.s, z17.s
    add     z10.s, z10.s, z11.s
    add     z18.s, z18.s, z19.s
.if \a16
    lsl     z10.s, z10.s, #8
    lsl     z18.s, z18.s, #8
.endif
    add     z22.s, z8.s, z10.s
    add     z23.s, z16.s, z18.s
    ld1rd   {z9.d}, p2/z, [x4]
    add     x4, x4, #8
    lsl     z10.d, z12.d, #23
    lsl     z12.d, z12.d, #55
    lsl     z11.d, z13.d, #23
    lsl     z13.d, z13.d, #55
    orr     z12.d, z12.d, z10.d
    orr     z13.d, z13.d, z11.d
    fmul    z12.s, z12.s, z9.s
    fmul    z13.s, z13.s, z9.s
.endm
.macro KERNEL8F6 name, a16
    .global \name
    .type \name, %function
    .p2align 6
\name:
    ldr     x8, [sp]
    stp     d8, d9, [sp, #-64]!
    stp     d10, d11, [sp, #16]
    stp     d12, d13, [sp, #32]
    stp     d14, d15, [sp, #48]
    ptrue   p0.b
    ptrue   p1.s
    ptrue   p2.d
    ld1b    {ZL.b}, p0/z, [x7]
    dup     M0F.b, #15
    dup     M30.b, #0x30
    dup     ZERO.s, #0
    dup     ACC.s, #0
    dup     ACC2.s, #0
    mov     x9, #PF2_DIST
    lsr     x11, x9, #4
    S12D                                     /* t=0 */
    S3F \a16                                 /* t=1 */
    S12D
    S4G \a16                                 /* t=2 */
    S3F \a16
    S12D
    sub     x5, x5, #3
    cbz     x5, 2f
1:
    S5E
    S4G \a16
    S3F \a16
    S12D
    subs    x5, x5, #1
    b.ne    1b
2:
    S5E
    S4G \a16
    S3F \a16
    S5E
    S4G \a16
    S5E
    st1w    {ACC.s}, p1, [x6]
    st1w    {ACC2.s}, p1, [x6, #1, mul vl]
    ldp     d10, d11, [sp, #16]
    ldp     d12, d13, [sp, #32]
    ldp     d14, d15, [sp, #48]
    ldp     d8, d9, [sp], #64
    ret
    .size \name, .-\name
.endm

    KERNEL8F6 q38d_asm8_f6_a8, 0
    KERNEL8F6 q38d_asm8_f6_a16, 1

    .section .note.GNU-stack,"",%progbits
