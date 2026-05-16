// micro_kernel_bf16B_8x3_pv.S
// Pair-interleaved BF16 variant of micro_kernel_bf16B_8x3_unroll4.
//
// B is packed with two consecutive K-rows interleaved at halfword
// granularity within each 16-col chunk:
//   chunk (64 bytes / 32 HW), cols [16c .. 16c+15]:
//     HW[2i+0] = B[k_even][16c+i]
//     HW[2i+1] = B[k_odd ][16c+i]
//
// A predicated ld1h{.h} with p_odd (HW lanes 1,3,...,31 active, /z zeroes
// the rest) puts BF16 into the upper 16 bits of every .s lane = valid
// FP32. No LSL on the FLA pipe.
//   k_odd  vector: ld1h {z.h}, p_odd/z, [base       ]
//   k_even vector: ld1h {z.h}, p_odd/z, [base - 2HW ]   (-2 bytes shift)
//
// Per K-pair, per N-block (NR=48): 3 chunks × 64 bytes = 192 bytes.
// K unroll = 4 ⇒ per outer iter: 2 K-pairs = 384 bytes (same B stride
// as the LSL variant, just rearranged).
//
// Arguments (identical to micro_kernel_bf16B_8x3_unroll4):
//   x0 = A_packed pointer
//   x1 = B_packed (PV layout) pointer
//   x2 = C output pointer
//   x3 = K (must be divisible by 4 and even)
//   x4 = unused
//   x5 = ldc_bytes
//
// Caller MUST ensure B[-2 bytes] is dereferenceable (1 HW of pad before
// the buffer is sufficient). pack_B_bf16_pv produces a layout that
// satisfies this when the BTP allocation includes a 1-HW prefix.

    .arch armv8.2-a+sve
    .text
    .align 4
    .global micro_kernel_bf16B_8x3_unroll4_pv
    .type micro_kernel_bf16B_8x3_unroll4_pv, %function

micro_kernel_bf16B_8x3_unroll4_pv:
    stp x19, x20, [sp, #-80]!
    stp d8, d9,  [sp, #16]
    stp d10, d11, [sp, #32]
    stp d12, d13, [sp, #48]
    stp d14, d15, [sp, #64]

    ptrue p0.s
    ptrue p2.h
    index z31.h, #0, #1
    and   z31.h, z31.h, #1
    cmpne p1.h, p2/z, z31.h, #0       // p_odd: HW lanes 1,3,...,31 active

    sub  x9, x1, #2                    // k_even base (= k_odd base - 2 bytes)

    prfm pldl1strm, [x0]
    prfm pldl1strm, [x0, #64]
    prfm pldl1strm, [x1]
    prfm pldl1strm, [x1, #64]

    // Load first K-pair (k=0,1). Per chunk: k_even from x9, k_odd from x1.
    ld1h {z24.h}, p1/z, [x9]                       // k+0, cols 0..15
    ld1h {z25.h}, p1/z, [x9, #1, mul vl]           // k+0, cols 16..31
    ld1h {z26.h}, p1/z, [x9, #2, mul vl]           // k+0, cols 32..47

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
    mov x20, #384      // B prefetch distance

.Lloop_bf16B_pv:
    add x7, x0, x19
    prfm pldl1strm, [x7]
    add x7, x1, x20
    prfm pldl1strm, [x7]

    // ─── K iter 0 (k_even of pair 0) ─────────────────────────────
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

    // Load B for k+1 (k_odd of pair 0): same chunk offsets, from x1
    ld1h {z24.h}, p1/z, [x1]
    ld1h {z25.h}, p1/z, [x1, #1, mul vl]
    ld1h {z26.h}, p1/z, [x1, #2, mul vl]

    // ─── K iter 1 (k_odd of pair 0) ──────────────────────────────
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

    // Load B for k+2 (k_even of pair 1): offset +3, +4, +5 MUL VL from x9
    ld1h {z24.h}, p1/z, [x9, #3, mul vl]
    ld1h {z25.h}, p1/z, [x9, #4, mul vl]
    ld1h {z26.h}, p1/z, [x9, #5, mul vl]

    // ─── K iter 2 (k_even of pair 1) ─────────────────────────────
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

    // Load B for k+3 (k_odd of pair 1): offset +3, +4, +5 MUL VL from x1
    ld1h {z24.h}, p1/z, [x1, #3, mul vl]
    ld1h {z25.h}, p1/z, [x1, #4, mul vl]
    ld1h {z26.h}, p1/z, [x1, #5, mul vl]

    // ─── K iter 3 (k_odd of pair 1) ──────────────────────────────
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

    // Advance pointers — same stride as LSL variant: 384 B
    add x0, x0, #128
    add x1, x1, #384
    add x9, x9, #384

    subs x6, x6, #1
    beq .Lstoreout_bf16B_pv

    // Preload first K-pair of next outer iter (k_even of pair 0)
    ld1h {z24.h}, p1/z, [x9]
    ld1h {z25.h}, p1/z, [x9, #1, mul vl]
    ld1h {z26.h}, p1/z, [x9, #2, mul vl]

    b .Lloop_bf16B_pv

.Lstoreout_bf16B_pv:
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
    .size micro_kernel_bf16B_8x3_unroll4_pv, .-micro_kernel_bf16B_8x3_unroll4_pv
