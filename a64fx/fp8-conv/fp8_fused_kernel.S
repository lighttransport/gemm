/*
 * FP8 GEMM Kernel with Fused Conversion
 *
 * Instead of pre-converting FP8 to FP32, we convert inline in the kernel.
 * This allows gather latency to be hidden by compute.
 *
 * Key insight:
 *   - SVE gather has ~11 cycle latency
 *   - But A64FX can have 2 in-flight gathers
 *   - We interleave gather + conversion with FMLA compute
 *
 * Tile: 8×3 (8 rows, 3 columns = 48 output elements)
 * Per K iteration:
 *   - Load 8 FP8 bytes from A (gather from LUT for conversion)
 *   - Load 3 FP32 vectors from B (already converted)
 *   - 24 FMLA operations
 *
 * Register allocation:
 *   Z0-Z23:  24 accumulators
 *   Z24-Z26: 3 B vectors
 *   Z27-Z30: 4 A values (broadcast FP32)
 *   Z31:     temp for A conversion
 *
 * Arguments:
 *   x0: A_fp8 pointer (FP8 packed as [K][MR])
 *   x1: B_fp32 pointer (FP32 packed as [K][3*VL])
 *   x2: C pointer
 *   x3: ldc in bytes
 *   x4: K
 *   x5: LUT pointer (fp8_to_fp32_lut)
 */

    .arch armv8.2-a+sve
    .text
    .align 4
    .global fp8_fused_kernel_8x3
    .type fp8_fused_kernel_8x3, %function

fp8_fused_kernel_8x3:
    // Setup
    ptrue p0.s

    // Save callee-saved registers if needed
    // For now, we use only caller-saved regs

    // Initialize 24 accumulators
    eor z0.d, z0.d, z0.d
    eor z1.d, z1.d, z1.d
    eor z2.d, z2.d, z2.d
    eor z3.d, z3.d, z3.d
    eor z4.d, z4.d, z4.d
    eor z5.d, z5.d, z5.d
    eor z6.d, z6.d, z6.d
    eor z7.d, z7.d, z7.d
    eor z8.d, z8.d, z8.d
    eor z9.d, z9.d, z9.d
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

    // Predicate for 8 elements (MR=8)
    mov x6, #8
    whilelt p1.s, xzr, x6

    // K loop counter
    mov x6, x4
    cbz x6, .Lfused_store

    // Load first B (3 vectors)
    ld1w {z24.s}, p0/z, [x1]
    ld1w {z25.s}, p0/z, [x1, #1, mul vl]
    ld1w {z26.s}, p0/z, [x1, #2, mul vl]

    // Preload first A[0..7] FP8 indices
    ld1b {z31.s}, p1/z, [x0]       // Load 8 bytes as 32-bit (zero-extend)

    // Gather first FP32 values from LUT
    ld1w {z27.s}, p1/z, [x5, z31.s, uxtw #2]

.Lfused_loop:
    // ═══ Process K iteration ═══

    // Broadcast A[0] and compute row 0
    mov z28.s, z27.s[0]            // Broadcast element 0
    fmla z0.s, p0/m, z28.s, z24.s
    fmla z1.s, p0/m, z28.s, z25.s
    fmla z2.s, p0/m, z28.s, z26.s

    // Broadcast A[1] and compute row 1
    mov z28.s, z27.s[1]
    fmla z3.s, p0/m, z28.s, z24.s
    fmla z4.s, p0/m, z28.s, z25.s
    fmla z5.s, p0/m, z28.s, z26.s

    // Broadcast A[2] and compute row 2
    mov z28.s, z27.s[2]
    fmla z6.s, p0/m, z28.s, z24.s
    fmla z7.s, p0/m, z28.s, z25.s
    fmla z8.s, p0/m, z28.s, z26.s

    // Broadcast A[3] and compute row 3
    mov z28.s, z27.s[3]
    fmla z9.s, p0/m, z28.s, z24.s
    fmla z10.s, p0/m, z28.s, z25.s
    fmla z11.s, p0/m, z28.s, z26.s

    // Broadcast A[4] and compute row 4
    mov z28.s, z27.s[4]
    fmla z12.s, p0/m, z28.s, z24.s
    fmla z13.s, p0/m, z28.s, z25.s
    fmla z14.s, p0/m, z28.s, z26.s

    // Broadcast A[5] and compute row 5
    mov z28.s, z27.s[5]
    fmla z15.s, p0/m, z28.s, z24.s
    fmla z16.s, p0/m, z28.s, z25.s
    fmla z17.s, p0/m, z28.s, z26.s

    // Broadcast A[6] and compute row 6
    mov z28.s, z27.s[6]
    fmla z18.s, p0/m, z28.s, z24.s
    fmla z19.s, p0/m, z28.s, z25.s
    fmla z20.s, p0/m, z28.s, z26.s

    // Broadcast A[7] and compute row 7
    mov z28.s, z27.s[7]
    fmla z21.s, p0/m, z28.s, z24.s
    fmla z22.s, p0/m, z28.s, z25.s
    fmla z23.s, p0/m, z28.s, z26.s

    // Advance pointers
    add x0, x0, #8              // A: 8 FP8 bytes per K
    add x1, x1, #192            // B: 3 vectors × 64 bytes

    // Decrement and check loop
    subs x6, x6, #1
    beq .Lfused_store

    // Load next B
    ld1w {z24.s}, p0/z, [x1]
    ld1w {z25.s}, p0/z, [x1, #1, mul vl]
    ld1w {z26.s}, p0/z, [x1, #2, mul vl]

    // Load next A FP8 indices
    ld1b {z31.s}, p1/z, [x0]

    // Gather FP32 from LUT (while B loads are in flight)
    ld1w {z27.s}, p1/z, [x5, z31.s, uxtw #2]

    b .Lfused_loop

.Lfused_store:
    // Store results
    st1w {z0.s}, p0, [x2]
    st1w {z1.s}, p0, [x2, #1, mul vl]
    st1w {z2.s}, p0, [x2, #2, mul vl]
    add x2, x2, x3

    st1w {z3.s}, p0, [x2]
    st1w {z4.s}, p0, [x2, #1, mul vl]
    st1w {z5.s}, p0, [x2, #2, mul vl]
    add x2, x2, x3

    st1w {z6.s}, p0, [x2]
    st1w {z7.s}, p0, [x2, #1, mul vl]
    st1w {z8.s}, p0, [x2, #2, mul vl]
    add x2, x2, x3

    st1w {z9.s}, p0, [x2]
    st1w {z10.s}, p0, [x2, #1, mul vl]
    st1w {z11.s}, p0, [x2, #2, mul vl]
    add x2, x2, x3

    st1w {z12.s}, p0, [x2]
    st1w {z13.s}, p0, [x2, #1, mul vl]
    st1w {z14.s}, p0, [x2, #2, mul vl]
    add x2, x2, x3

    st1w {z15.s}, p0, [x2]
    st1w {z16.s}, p0, [x2, #1, mul vl]
    st1w {z17.s}, p0, [x2, #2, mul vl]
    add x2, x2, x3

    st1w {z18.s}, p0, [x2]
    st1w {z19.s}, p0, [x2, #1, mul vl]
    st1w {z20.s}, p0, [x2, #2, mul vl]
    add x2, x2, x3

    st1w {z21.s}, p0, [x2]
    st1w {z22.s}, p0, [x2, #1, mul vl]
    st1w {z23.s}, p0, [x2, #2, mul vl]

    ret
    .size fp8_fused_kernel_8x3, .-fp8_fused_kernel_8x3
