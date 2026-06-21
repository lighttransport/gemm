// micro_kernel_bf16B_8x3.S
// Mixed-precision SVE GEMM microkernel for A64FX:
//   A = FP32 (broadcast, packed [K/4][MR][4])
//   B = BF16 (vector,    packed [K/4][NR][4])
//   C = FP32 (output, row-major with ldc_bytes stride)
//
// Tile: MR=8 rows × NR=48 cols, K-unroll 4. Mirror of micro_kernel_fp16B_8x3
// but BF16 → FP32 is just LSL #16 (BF16 occupies the upper 16 bits of FP32).
// A64FX has no BFMMLA/BFDOT, so the win here is purely memory bandwidth.
//
// Register allocation:
//   Z0-Z23:  24 accumulators (8 rows × 3 SVE-vec columns, all fp32)
//   Z24-Z26: 3 B vectors (loaded as bf16 → LSL #16 → fp32)
//   Z27-Z30: 4 A temps (fp32 broadcast)
//   Z31:     spare
//
// Arguments:
//   x0 = A_packed pointer (broadcast format: [K/4][MR][4])
//   x1 = B_packed pointer (bf16 vector format: [K/4][NR][4])
//   x2 = C output pointer (fp32)
//   x3 = K (must be divisible by 4)
//   x4 = unused
//   x5 = ldc_bytes
//
// B advance per K4: 4 * 48 * 2 = 384 bytes (half of fp32's 768).
// For ld1h {z.s}, "MUL VL" scales by VL_bytes/2 = 32 on a64fx (SVE 512).

    .arch armv8.2-a+sve
    .text
    .align 4
    .global micro_kernel_bf16B_8x3_unroll4
    .type micro_kernel_bf16B_8x3_unroll4, %function

micro_kernel_bf16B_8x3_unroll4:
    stp x19, x20, [sp, #-80]!
    stp d8, d9,  [sp, #16]
    stp d10, d11, [sp, #32]
    stp d12, d13, [sp, #48]
    stp d14, d15, [sp, #64]

    ptrue p0.s

    prfm pldl1strm, [x0]
    prfm pldl1strm, [x0, #64]
    prfm pldl1strm, [x1]
    prfm pldl1strm, [x1, #64]

    // Load first 3 B vectors and shift bf16 → fp32 (bf16 lives in upper 16 bits).
    ld1h {z24.s}, p0/z, [x1]
    lsl  z24.s, z24.s, #16
    ld1h {z25.s}, p0/z, [x1, #1, mul vl]
    lsl  z25.s, z25.s, #16
    ld1h {z26.s}, p0/z, [x1, #2, mul vl]
    lsl  z26.s, z26.s, #16

    eor z0.d,  z0.d,  z0.d
    eor z1.d,  z1.d,  z1.d
    eor z2.d,  z2.d,  z2.d
    eor z3.d,  z3.d,  z3.d
    eor z4.d,  z4.d,  z4.d
    eor z5.d,  z5.d,  z5.d
    eor z6.d,  z6.d,  z6.d
    eor z7.d,  z7.d,  z7.d
    eor z8.d,  z8.d,  z8.d
    eor z9.d,  z9.d,  z9.d
    eor z10.d, z10.d, z10.d
    eor z11.d, z11.d, z11.d
    eor z12.d, z12.d, z12.d
    eor z13.d, z13.d, z13.d
    eor z14.d, z14.d, z14.d
    eor z15.d, z15.d, z15.d
    eor z16.d, z16.d, z16.d
    eor z17.d, z17.d, z17.d
    eor z18.d, z18.d, z18.d
    eor z19.d, z19.d, z19.d
    eor z20.d, z20.d, z20.d
    eor z21.d, z21.d, z21.d
    eor z22.d, z22.d, z22.d
    eor z23.d, z23.d, z23.d

    lsr x6, x3, #2     // K/4 iterations

    mov x19, #256      // A prefetch distance
    mov x20, #384      // B prefetch distance (one K4 block ahead)

.Lloop_bf16B:
    add x7, x0, x19
    prfm pldl1strm, [x7]
    add x7, x1, x20
    prfm pldl1strm, [x7]

    // ─── K iter 0 ─────────────────────────────────────────────────
    ld1rw {z27.s}, p0/z, [x0, #0]
    ld1rw {z28.s}, p0/z, [x0, #4]
    ld1rw {z29.s}, p0/z, [x0, #8]
    ld1rw {z30.s}, p0/z, [x0, #12]

    fmla z0.s,  p0/m, z27.s, z24.s
    fmla z1.s,  p0/m, z27.s, z25.s
    fmla z2.s,  p0/m, z27.s, z26.s
    fmla z3.s,  p0/m, z28.s, z24.s
    fmla z4.s,  p0/m, z28.s, z25.s
    fmla z5.s,  p0/m, z28.s, z26.s
    fmla z6.s,  p0/m, z29.s, z24.s
    fmla z7.s,  p0/m, z29.s, z25.s
    fmla z8.s,  p0/m, z29.s, z26.s
    fmla z9.s,  p0/m, z30.s, z24.s
    fmla z10.s, p0/m, z30.s, z25.s
    fmla z11.s, p0/m, z30.s, z26.s

    ld1rw {z27.s}, p0/z, [x0, #16]
    ld1rw {z28.s}, p0/z, [x0, #20]
    ld1rw {z29.s}, p0/z, [x0, #24]
    ld1rw {z30.s}, p0/z, [x0, #28]

    fmla z12.s, p0/m, z27.s, z24.s
    fmla z13.s, p0/m, z27.s, z25.s
    fmla z14.s, p0/m, z27.s, z26.s
    fmla z15.s, p0/m, z28.s, z24.s
    fmla z16.s, p0/m, z28.s, z25.s
    fmla z17.s, p0/m, z28.s, z26.s
    fmla z18.s, p0/m, z29.s, z24.s
    fmla z19.s, p0/m, z29.s, z25.s
    fmla z20.s, p0/m, z29.s, z26.s
    fmla z21.s, p0/m, z30.s, z24.s
    fmla z22.s, p0/m, z30.s, z25.s
    fmla z23.s, p0/m, z30.s, z26.s

    // Load B for k+1
    ld1h {z24.s}, p0/z, [x1, #3, mul vl]
    lsl  z24.s, z24.s, #16
    ld1h {z25.s}, p0/z, [x1, #4, mul vl]
    lsl  z25.s, z25.s, #16
    ld1h {z26.s}, p0/z, [x1, #5, mul vl]
    lsl  z26.s, z26.s, #16

    // ─── K iter 1 ─────────────────────────────────────────────────
    ld1rw {z27.s}, p0/z, [x0, #32]
    ld1rw {z28.s}, p0/z, [x0, #36]
    ld1rw {z29.s}, p0/z, [x0, #40]
    ld1rw {z30.s}, p0/z, [x0, #44]

    fmla z0.s,  p0/m, z27.s, z24.s
    fmla z1.s,  p0/m, z27.s, z25.s
    fmla z2.s,  p0/m, z27.s, z26.s
    fmla z3.s,  p0/m, z28.s, z24.s
    fmla z4.s,  p0/m, z28.s, z25.s
    fmla z5.s,  p0/m, z28.s, z26.s
    fmla z6.s,  p0/m, z29.s, z24.s
    fmla z7.s,  p0/m, z29.s, z25.s
    fmla z8.s,  p0/m, z29.s, z26.s
    fmla z9.s,  p0/m, z30.s, z24.s
    fmla z10.s, p0/m, z30.s, z25.s
    fmla z11.s, p0/m, z30.s, z26.s

    ld1rw {z27.s}, p0/z, [x0, #48]
    ld1rw {z28.s}, p0/z, [x0, #52]
    ld1rw {z29.s}, p0/z, [x0, #56]
    ld1rw {z30.s}, p0/z, [x0, #60]

    fmla z12.s, p0/m, z27.s, z24.s
    fmla z13.s, p0/m, z27.s, z25.s
    fmla z14.s, p0/m, z27.s, z26.s
    fmla z15.s, p0/m, z28.s, z24.s
    fmla z16.s, p0/m, z28.s, z25.s
    fmla z17.s, p0/m, z28.s, z26.s
    fmla z18.s, p0/m, z29.s, z24.s
    fmla z19.s, p0/m, z29.s, z25.s
    fmla z20.s, p0/m, z29.s, z26.s
    fmla z21.s, p0/m, z30.s, z24.s
    fmla z22.s, p0/m, z30.s, z25.s
    fmla z23.s, p0/m, z30.s, z26.s

    // Load B for k+2: offsets #6,#7 mul vl reach 192,224; #8 spills → use x7=x1+256
    ld1h {z24.s}, p0/z, [x1, #6, mul vl]
    lsl  z24.s, z24.s, #16
    ld1h {z25.s}, p0/z, [x1, #7, mul vl]
    lsl  z25.s, z25.s, #16
    add x7, x1, #256
    ld1h {z26.s}, p0/z, [x7]
    lsl  z26.s, z26.s, #16

    // ─── K iter 2 ─────────────────────────────────────────────────
    ld1rw {z27.s}, p0/z, [x0, #64]
    ld1rw {z28.s}, p0/z, [x0, #68]
    ld1rw {z29.s}, p0/z, [x0, #72]
    ld1rw {z30.s}, p0/z, [x0, #76]

    fmla z0.s,  p0/m, z27.s, z24.s
    fmla z1.s,  p0/m, z27.s, z25.s
    fmla z2.s,  p0/m, z27.s, z26.s
    fmla z3.s,  p0/m, z28.s, z24.s
    fmla z4.s,  p0/m, z28.s, z25.s
    fmla z5.s,  p0/m, z28.s, z26.s
    fmla z6.s,  p0/m, z29.s, z24.s
    fmla z7.s,  p0/m, z29.s, z25.s
    fmla z8.s,  p0/m, z29.s, z26.s
    fmla z9.s,  p0/m, z30.s, z24.s
    fmla z10.s, p0/m, z30.s, z25.s
    fmla z11.s, p0/m, z30.s, z26.s

    ld1rw {z27.s}, p0/z, [x0, #80]
    ld1rw {z28.s}, p0/z, [x0, #84]
    ld1rw {z29.s}, p0/z, [x0, #88]
    ld1rw {z30.s}, p0/z, [x0, #92]

    fmla z12.s, p0/m, z27.s, z24.s
    fmla z13.s, p0/m, z27.s, z25.s
    fmla z14.s, p0/m, z27.s, z26.s
    fmla z15.s, p0/m, z28.s, z24.s
    fmla z16.s, p0/m, z28.s, z25.s
    fmla z17.s, p0/m, z28.s, z26.s
    fmla z18.s, p0/m, z29.s, z24.s
    fmla z19.s, p0/m, z29.s, z25.s
    fmla z20.s, p0/m, z29.s, z26.s
    fmla z21.s, p0/m, z30.s, z24.s
    fmla z22.s, p0/m, z30.s, z25.s
    fmla z23.s, p0/m, z30.s, z26.s

    // Load B for k+3 from x7 (=x1+256): offsets 288,320,352 = #1,#2,#3 mul vl
    ld1h {z24.s}, p0/z, [x7, #1, mul vl]
    lsl  z24.s, z24.s, #16
    ld1h {z25.s}, p0/z, [x7, #2, mul vl]
    lsl  z25.s, z25.s, #16
    ld1h {z26.s}, p0/z, [x7, #3, mul vl]
    lsl  z26.s, z26.s, #16

    // ─── K iter 3 ─────────────────────────────────────────────────
    ld1rw {z27.s}, p0/z, [x0, #96]
    ld1rw {z28.s}, p0/z, [x0, #100]
    ld1rw {z29.s}, p0/z, [x0, #104]
    ld1rw {z30.s}, p0/z, [x0, #108]

    fmla z0.s,  p0/m, z27.s, z24.s
    fmla z1.s,  p0/m, z27.s, z25.s
    fmla z2.s,  p0/m, z27.s, z26.s
    fmla z3.s,  p0/m, z28.s, z24.s
    fmla z4.s,  p0/m, z28.s, z25.s
    fmla z5.s,  p0/m, z28.s, z26.s
    fmla z6.s,  p0/m, z29.s, z24.s
    fmla z7.s,  p0/m, z29.s, z25.s
    fmla z8.s,  p0/m, z29.s, z26.s
    fmla z9.s,  p0/m, z30.s, z24.s
    fmla z10.s, p0/m, z30.s, z25.s
    fmla z11.s, p0/m, z30.s, z26.s

    ld1rw {z27.s}, p0/z, [x0, #112]
    ld1rw {z28.s}, p0/z, [x0, #116]
    ld1rw {z29.s}, p0/z, [x0, #120]
    ld1rw {z30.s}, p0/z, [x0, #124]

    fmla z12.s, p0/m, z27.s, z24.s
    fmla z13.s, p0/m, z27.s, z25.s
    fmla z14.s, p0/m, z27.s, z26.s
    fmla z15.s, p0/m, z28.s, z24.s
    fmla z16.s, p0/m, z28.s, z25.s
    fmla z17.s, p0/m, z28.s, z26.s
    fmla z18.s, p0/m, z29.s, z24.s
    fmla z19.s, p0/m, z29.s, z25.s
    fmla z20.s, p0/m, z29.s, z26.s
    fmla z21.s, p0/m, z30.s, z24.s
    fmla z22.s, p0/m, z30.s, z25.s
    fmla z23.s, p0/m, z30.s, z26.s

    // Advance pointers (4 K iterations)
    add x0, x0, #128   // A: 4 * 8 * 4 bytes
    add x1, x1, #384   // B: 4 * 48 * 2 bytes (half of fp32 stride)

    subs x6, x6, #1
    beq .Lstoreout_bf16B

    ld1h {z24.s}, p0/z, [x1]
    lsl  z24.s, z24.s, #16
    ld1h {z25.s}, p0/z, [x1, #1, mul vl]
    lsl  z25.s, z25.s, #16
    ld1h {z26.s}, p0/z, [x1, #2, mul vl]
    lsl  z26.s, z26.s, #16

    b .Lloop_bf16B

.Lstoreout_bf16B:
    st1w {z0.s},  p0, [x2]
    st1w {z1.s},  p0, [x2, #1, mul vl]
    st1w {z2.s},  p0, [x2, #2, mul vl]
    add x2, x2, x5
    st1w {z3.s},  p0, [x2]
    st1w {z4.s},  p0, [x2, #1, mul vl]
    st1w {z5.s},  p0, [x2, #2, mul vl]
    add x2, x2, x5
    st1w {z6.s},  p0, [x2]
    st1w {z7.s},  p0, [x2, #1, mul vl]
    st1w {z8.s},  p0, [x2, #2, mul vl]
    add x2, x2, x5
    st1w {z9.s},  p0, [x2]
    st1w {z10.s}, p0, [x2, #1, mul vl]
    st1w {z11.s}, p0, [x2, #2, mul vl]
    add x2, x2, x5
    st1w {z12.s}, p0, [x2]
    st1w {z13.s}, p0, [x2, #1, mul vl]
    st1w {z14.s}, p0, [x2, #2, mul vl]
    add x2, x2, x5
    st1w {z15.s}, p0, [x2]
    st1w {z16.s}, p0, [x2, #1, mul vl]
    st1w {z17.s}, p0, [x2, #2, mul vl]
    add x2, x2, x5
    st1w {z18.s}, p0, [x2]
    st1w {z19.s}, p0, [x2, #1, mul vl]
    st1w {z20.s}, p0, [x2, #2, mul vl]
    add x2, x2, x5
    st1w {z21.s}, p0, [x2]
    st1w {z22.s}, p0, [x2, #1, mul vl]
    st1w {z23.s}, p0, [x2, #2, mul vl]

    ldp d14, d15, [sp, #64]
    ldp d12, d13, [sp, #48]
    ldp d10, d11, [sp, #32]
    ldp d8, d9,  [sp, #16]
    ldp x19, x20, [sp], #80
    ret
    .size micro_kernel_bf16B_8x3_unroll4, .-micro_kernel_bf16B_8x3_unroll4
