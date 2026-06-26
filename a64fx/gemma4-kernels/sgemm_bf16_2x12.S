// sgemm_bf16_2x12.S — All-BF16 GEMM microkernel for A64FX SVE (512-bit)
//
// BF16 A × BF16 B → FP32 accumulation → FP32 output.
// ZERO conversion overhead via p_odd predicate + interleaved A packing.
//
// The p_odd trick: ld1h {z.h}, p_odd/z loads from positional addresses
// (base + j*2 for active odd j=1,3,...,31), placing BF16 values in the
// UPPER 16 bits of each .s container = valid FP32.
//
// For A contiguous loads: A is packed INTERLEAVED [R0_0,R1_0,R0_1,R1_1,...].
// Two loads at offsets 0 and -2 sample even/odd source elements:
//   ld1h {z24.h}, p_odd/z, [A_ptr, x_neg1, lsl #1]  → row 0
//   ld1h {z25.h}, p_odd/z, [A_ptr]                    → row 1
// Both hit the SAME cache line.  Zero lsl.  Zero extra FP ops.
//
// For B broadcast loads: ld1rh {z.h}, p_odd/z works directly (verified).
//
// void sgemm_bf16_2x12(int64_t K,             // x0
//                       const void *A,          // x1 (packed 32×K BF16, INTERLEAVED)
//                       const void *B,          // x2 (packed K×12 BF16, row-major)
//                       float *C,               // x3 (col-major FP32, stride ldc)
//                       int64_t ldc);           // x4 (in FP32 elements)
//
// A packing: INTERLEAVED per k-step (64 bytes):
//   [row0[0], row1[0], row0[1], row1[1], ..., row0[15], row1[15]]
//   Pack once, reuse across all NR column passes.
//
// Tile: MR=32, NR=12, 24 FP32 accumulators
// Pipeline: 24 FMLA / 2 = 12 cycles per k-step (IDENTICAL to pure FP32!)
// Memory: A=64 bytes (BF16) + B=24 bytes (BF16) = 88 bytes/k-step
// AI = 768 / 88 = 8.73 FLOP/byte (2× the pure FP32 kernel)
//
// Registers:
//   z0-z23:  24 FP32 accumulators
//   z24,z25: A panel (FP32 via p_odd positional load)
//   z26-z31: B broadcast (FP32 via p_odd ld1rh)
//   p0: ptrue .s (FMA + store)
//   p1: p_odd (odd .h positions, for BF16 loads)
//   x16: -1  (A row-0 offset: -1 elements = -2 bytes from interleave base)
//   x17: 31  (A[k+1] row-0 offset)
//   x18: 32  (A[k+1] row-1 offset)

    .arch armv8.2-a+sve
    .text
    .align 6
    .global sgemm_bf16_2x12
    .type sgemm_bf16_2x12, %function

sgemm_bf16_2x12:
    sub     sp, sp, #96
    stp     x19, x20, [sp]
    stp     x21, x22, [sp, #16]
    stp     d8, d9, [sp, #32]
    stp     d10, d11, [sp, #48]
    stp     d12, d13, [sp, #64]
    stp     d14, d15, [sp, #80]

    ptrue   p0.s

    // Generate p_odd: odd .h positions {1,3,5,...,31}
    index   z24.h, #0, #1
    ptrue   p2.h
    and     z24.h, z24.h, #1
    cmpne   p1.h, p2/z, z24.h, #0

    // A offset constants (element offsets, LSL #1 → byte offsets)
    mov     x16, #-1              // row-0 base: A_ptr - 2 bytes
    mov     x17, #31              // A[k+1] row-0: (A_ptr+64) - 2 = A_ptr + 62
    mov     x18, #32              // A[k+1] row-1: A_ptr + 64
    mov     x21, #63              // A[k+2] row-0: A_ptr + 126
    mov     x22, #64              // A[k+2] row-1: A_ptr + 128

    // Column byte offsets for FP32 C store
    lsl     x5, x4, #2
    lsl     x6, x5, #1
    add     x7, x6, x5
    lsl     x8, x6, #1
    add     x9, x8, x5
    add     x10, x8, x6
    add     x11, x8, x7
    lsl     x12, x8, #1
    add     x13, x12, x5
    add     x14, x12, x6
    add     x15, x12, x7

    // Zero 24 accumulators
    eor     z0.d, z0.d, z0.d
    eor     z1.d, z1.d, z1.d
    eor     z2.d, z2.d, z2.d
    eor     z3.d, z3.d, z3.d
    eor     z4.d, z4.d, z4.d
    eor     z5.d, z5.d, z5.d
    eor     z6.d, z6.d, z6.d
    eor     z7.d, z7.d, z7.d
    eor     z8.d, z8.d, z8.d
    eor     z9.d, z9.d, z9.d
    eor     z10.d, z10.d, z10.d
    eor     z11.d, z11.d, z11.d
    eor     z12.d, z12.d, z12.d
    eor     z13.d, z13.d, z13.d
    eor     z14.d, z14.d, z14.d
    eor     z15.d, z15.d, z15.d
    eor     z16.d, z16.d, z16.d
    eor     z17.d, z17.d, z17.d
    eor     z18.d, z18.d, z18.d
    eor     z19.d, z19.d, z19.d
    eor     z20.d, z20.d, z20.d
    eor     z21.d, z21.d, z21.d
    eor     z22.d, z22.d, z22.d
    eor     z23.d, z23.d, z23.d

    cmp     x0, #0
    ble     .Lbf_store

    // Prefetch A (BF16 interleaved: 64 bytes/k-step)
    prfm    pldl1keep, [x1, #0]
    prfm    pldl1keep, [x1, #64]
    prfm    pldl1keep, [x1, #128]
    prfm    pldl1keep, [x1, #192]
    prfm    pldl1keep, [x1, #256]
    prfm    pldl1keep, [x1, #320]
    prfm    pldl1keep, [x1, #384]
    prfm    pldl1keep, [x1, #448]
    prfm    pldl1keep, [x1, #512]

    // Pre-load A[0]: interleaved → two offset loads via p_odd
    ld1h    {z24.h}, p1/z, [x1, x16, lsl #1]      // row 0 (base = x1-2)
    ld1h    {z25.h}, p1/z, [x1]                    // row 1 (base = x1)
    // Pre-load B[0,0:5]: broadcast via p_odd
    ld1rh   {z26.h}, p1/z, [x2]
    ld1rh   {z27.h}, p1/z, [x2, #2]
    ld1rh   {z28.h}, p1/z, [x2, #4]
    ld1rh   {z29.h}, p1/z, [x2, #6]
    ld1rh   {z30.h}, p1/z, [x2, #8]
    ld1rh   {z31.h}, p1/z, [x2, #10]

    mov     x19, x0
    cmp     x19, #2
    blt     .Lbf_k_tail

    // ================================================================
    // 2× K-unrolled main loop — ALL BF16, ZERO conversion overhead
    // ================================================================
    .align 6
.Lbf_k_loop:

    // ---- Step k: group-1 FMAs (cols 0-5) + B[k,6:11] loads ----
    fmla    z0.s, p0/m, z24.s, z26.s
    fmla    z1.s, p0/m, z25.s, z26.s
    ld1rh   {z26.h}, p1/z, [x2, #12]
    fmla    z2.s, p0/m, z24.s, z27.s
    fmla    z3.s, p0/m, z25.s, z27.s
    ld1rh   {z27.h}, p1/z, [x2, #14]
    fmla    z4.s, p0/m, z24.s, z28.s
    fmla    z5.s, p0/m, z25.s, z28.s
    ld1rh   {z28.h}, p1/z, [x2, #16]
    fmla    z6.s, p0/m, z24.s, z29.s
    fmla    z7.s, p0/m, z25.s, z29.s
    ld1rh   {z29.h}, p1/z, [x2, #18]
    fmla    z8.s, p0/m, z24.s, z30.s
    fmla    z9.s, p0/m, z25.s, z30.s
    ld1rh   {z30.h}, p1/z, [x2, #20]
    fmla    z10.s, p0/m, z24.s, z31.s
    fmla    z11.s, p0/m, z25.s, z31.s
    ld1rh   {z31.h}, p1/z, [x2, #22]

    // ---- Step k: group-2 FMAs (cols 6-11) + B[k+1,0:5] + A[k+1] ----
    fmla    z12.s, p0/m, z24.s, z26.s
    fmla    z13.s, p0/m, z25.s, z26.s
    ld1rh   {z26.h}, p1/z, [x2, #24]
    fmla    z14.s, p0/m, z24.s, z27.s
    fmla    z15.s, p0/m, z25.s, z27.s
    ld1rh   {z27.h}, p1/z, [x2, #26]
    fmla    z16.s, p0/m, z24.s, z28.s
    fmla    z17.s, p0/m, z25.s, z28.s
    ld1rh   {z28.h}, p1/z, [x2, #28]
    fmla    z18.s, p0/m, z24.s, z29.s
    fmla    z19.s, p0/m, z25.s, z29.s
    ld1rh   {z29.h}, p1/z, [x2, #30]
    fmla    z20.s, p0/m, z24.s, z30.s
    fmla    z21.s, p0/m, z25.s, z30.s
    ld1rh   {z30.h}, p1/z, [x2, #32]
    fmla    z22.s, p0/m, z24.s, z31.s
    fmla    z23.s, p0/m, z25.s, z31.s       // z24,z25 dead
    // A[k+1] load: offset trick from interleaved data
    ld1h    {z24.h}, p1/z, [x1, x17, lsl #1]  // row 0 (x1 + 31*2 = x1+62)
    ld1h    {z25.h}, p1/z, [x1, x18, lsl #1]  // row 1 (x1 + 32*2 = x1+64)
    ld1rh   {z31.h}, p1/z, [x2, #34]
    prfm    pldl1keep, [x1, #1152]

    // ---- Step k+1: group-1 FMAs + B[k+1,6:11] loads ----
    fmla    z0.s, p0/m, z24.s, z26.s
    fmla    z1.s, p0/m, z25.s, z26.s
    ld1rh   {z26.h}, p1/z, [x2, #36]
    fmla    z2.s, p0/m, z24.s, z27.s
    fmla    z3.s, p0/m, z25.s, z27.s
    ld1rh   {z27.h}, p1/z, [x2, #38]
    fmla    z4.s, p0/m, z24.s, z28.s
    fmla    z5.s, p0/m, z25.s, z28.s
    ld1rh   {z28.h}, p1/z, [x2, #40]
    fmla    z6.s, p0/m, z24.s, z29.s
    fmla    z7.s, p0/m, z25.s, z29.s
    ld1rh   {z29.h}, p1/z, [x2, #42]
    fmla    z8.s, p0/m, z24.s, z30.s
    fmla    z9.s, p0/m, z25.s, z30.s
    ld1rh   {z30.h}, p1/z, [x2, #44]
    fmla    z10.s, p0/m, z24.s, z31.s
    fmla    z11.s, p0/m, z25.s, z31.s
    ld1rh   {z31.h}, p1/z, [x2, #46]

    // ---- Step k+1: group-2 FMAs + B[k+2,0:5] + A[k+2] pre-load ----
    fmla    z12.s, p0/m, z24.s, z26.s
    fmla    z13.s, p0/m, z25.s, z26.s
    ld1rh   {z26.h}, p1/z, [x2, #48]
    fmla    z14.s, p0/m, z24.s, z27.s
    fmla    z15.s, p0/m, z25.s, z27.s
    ld1rh   {z27.h}, p1/z, [x2, #50]
    fmla    z16.s, p0/m, z24.s, z28.s
    fmla    z17.s, p0/m, z25.s, z28.s
    ld1rh   {z28.h}, p1/z, [x2, #52]
    fmla    z18.s, p0/m, z24.s, z29.s
    fmla    z19.s, p0/m, z25.s, z29.s
    ld1rh   {z29.h}, p1/z, [x2, #54]
    fmla    z20.s, p0/m, z24.s, z30.s
    fmla    z21.s, p0/m, z25.s, z30.s
    ld1rh   {z30.h}, p1/z, [x2, #56]
    fmla    z22.s, p0/m, z24.s, z31.s
    fmla    z23.s, p0/m, z25.s, z31.s       // z24,z25 dead
    // A[k+2] pre-load for next iteration
    ld1h    {z24.h}, p1/z, [x1, x21, lsl #1]  // row 0 (x1 + 63*2 = x1+126)
    ld1h    {z25.h}, p1/z, [x1, x22, lsl #1]  // row 1 (x1 + 64*2 = x1+128)
    ld1rh   {z31.h}, p1/z, [x2, #58]

    // Advance: A += 2 × 64 bytes, B += 2 × 24 bytes
    add     x1, x1, #128
    add     x2, x2, #48
    sub     x19, x19, #2
    prfm    pldl1keep, [x1, #1152]
    cmp     x19, #2
    nop
    bge     .Lbf_k_loop

    // ================================================================
    // K remainder (0 or 1 step)
    // ================================================================
.Lbf_k_tail:
    cmp     x19, #1
    blt     .Lbf_store

    fmla    z0.s, p0/m, z24.s, z26.s
    fmla    z1.s, p0/m, z25.s, z26.s
    ld1rh   {z26.h}, p1/z, [x2, #12]
    fmla    z2.s, p0/m, z24.s, z27.s
    fmla    z3.s, p0/m, z25.s, z27.s
    ld1rh   {z27.h}, p1/z, [x2, #14]
    fmla    z4.s, p0/m, z24.s, z28.s
    fmla    z5.s, p0/m, z25.s, z28.s
    ld1rh   {z28.h}, p1/z, [x2, #16]
    fmla    z6.s, p0/m, z24.s, z29.s
    fmla    z7.s, p0/m, z25.s, z29.s
    ld1rh   {z29.h}, p1/z, [x2, #18]
    fmla    z8.s, p0/m, z24.s, z30.s
    fmla    z9.s, p0/m, z25.s, z30.s
    ld1rh   {z30.h}, p1/z, [x2, #20]
    fmla    z10.s, p0/m, z24.s, z31.s
    fmla    z11.s, p0/m, z25.s, z31.s
    ld1rh   {z31.h}, p1/z, [x2, #22]
    fmla    z12.s, p0/m, z24.s, z26.s
    fmla    z13.s, p0/m, z25.s, z26.s
    fmla    z14.s, p0/m, z24.s, z27.s
    fmla    z15.s, p0/m, z25.s, z27.s
    fmla    z16.s, p0/m, z24.s, z28.s
    fmla    z17.s, p0/m, z25.s, z28.s
    fmla    z18.s, p0/m, z24.s, z29.s
    fmla    z19.s, p0/m, z25.s, z29.s
    fmla    z20.s, p0/m, z24.s, z30.s
    fmla    z21.s, p0/m, z25.s, z30.s
    fmla    z22.s, p0/m, z24.s, z31.s
    fmla    z23.s, p0/m, z25.s, z31.s

.Lbf_store:
    st1w    {z0.s}, p0, [x3]
    st1w    {z1.s}, p0, [x3, #1, mul vl]
    add     x20, x3, x5
    st1w    {z2.s}, p0, [x20]
    st1w    {z3.s}, p0, [x20, #1, mul vl]
    add     x20, x3, x6
    st1w    {z4.s}, p0, [x20]
    st1w    {z5.s}, p0, [x20, #1, mul vl]
    add     x20, x3, x7
    st1w    {z6.s}, p0, [x20]
    st1w    {z7.s}, p0, [x20, #1, mul vl]
    add     x20, x3, x8
    st1w    {z8.s}, p0, [x20]
    st1w    {z9.s}, p0, [x20, #1, mul vl]
    add     x20, x3, x9
    st1w    {z10.s}, p0, [x20]
    st1w    {z11.s}, p0, [x20, #1, mul vl]
    add     x20, x3, x10
    st1w    {z12.s}, p0, [x20]
    st1w    {z13.s}, p0, [x20, #1, mul vl]
    add     x20, x3, x11
    st1w    {z14.s}, p0, [x20]
    st1w    {z15.s}, p0, [x20, #1, mul vl]
    add     x20, x3, x12
    st1w    {z16.s}, p0, [x20]
    st1w    {z17.s}, p0, [x20, #1, mul vl]
    add     x20, x3, x13
    st1w    {z18.s}, p0, [x20]
    st1w    {z19.s}, p0, [x20, #1, mul vl]
    add     x20, x3, x14
    st1w    {z20.s}, p0, [x20]
    st1w    {z21.s}, p0, [x20, #1, mul vl]
    add     x20, x3, x15
    st1w    {z22.s}, p0, [x20]
    st1w    {z23.s}, p0, [x20, #1, mul vl]

    ldp     d14, d15, [sp, #80]
    ldp     d12, d13, [sp, #64]
    ldp     d10, d11, [sp, #48]
    ldp     d8, d9, [sp, #32]
    ldp     x21, x22, [sp, #16]
    ldp     x19, x20, [sp]
    add     sp, sp, #96
    ret

    .size sgemm_bf16_2x12, .-sgemm_bf16_2x12
