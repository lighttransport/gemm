// Pure FMLA throughput microbenchmark
// Tests fp16 vs fp32 FMLA issue rate on independent accumulators

    .arch armv8.2-a+sve
    .text
    .align 4

// ─── FP32 FMLA throughput: 24 independent accumulators ───
// x0 = iteration count
    .global bench_fmla_fp32
    .type bench_fmla_fp32, %function
bench_fmla_fp32:
    stp d8, d9, [sp, #-64]!
    stp d10, d11, [sp, #16]
    stp d12, d13, [sp, #32]
    stp d14, d15, [sp, #48]
    ptrue p0.s
    // Zero all accumulators
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
    // Use z24,z25 as constant multipliers
    fdup z24.s, #1.0
    fdup z25.s, #1.0

.Lfp32_loop:
    // 24 independent FMLAs per iteration
    fmla z0.s, p0/m, z24.s, z25.s
    fmla z1.s, p0/m, z24.s, z25.s
    fmla z2.s, p0/m, z24.s, z25.s
    fmla z3.s, p0/m, z24.s, z25.s
    fmla z4.s, p0/m, z24.s, z25.s
    fmla z5.s, p0/m, z24.s, z25.s
    fmla z6.s, p0/m, z24.s, z25.s
    fmla z7.s, p0/m, z24.s, z25.s
    fmla z8.s, p0/m, z24.s, z25.s
    fmla z9.s, p0/m, z24.s, z25.s
    fmla z10.s, p0/m, z24.s, z25.s
    fmla z11.s, p0/m, z24.s, z25.s
    fmla z12.s, p0/m, z24.s, z25.s
    fmla z13.s, p0/m, z24.s, z25.s
    fmla z14.s, p0/m, z24.s, z25.s
    fmla z15.s, p0/m, z24.s, z25.s
    fmla z16.s, p0/m, z24.s, z25.s
    fmla z17.s, p0/m, z24.s, z25.s
    fmla z18.s, p0/m, z24.s, z25.s
    fmla z19.s, p0/m, z24.s, z25.s
    fmla z20.s, p0/m, z24.s, z25.s
    fmla z21.s, p0/m, z24.s, z25.s
    fmla z22.s, p0/m, z24.s, z25.s
    fmla z23.s, p0/m, z24.s, z25.s
    subs x0, x0, #1
    bne .Lfp32_loop

    ldp d14, d15, [sp, #48]
    ldp d12, d13, [sp, #32]
    ldp d10, d11, [sp, #16]
    ldp d8, d9, [sp], #64
    ret
    .size bench_fmla_fp32, .-bench_fmla_fp32

// ─── FP16 FMLA throughput: 24 independent accumulators ───
// x0 = iteration count
    .global bench_fmla_fp16
    .type bench_fmla_fp16, %function
bench_fmla_fp16:
    stp d8, d9, [sp, #-64]!
    stp d10, d11, [sp, #16]
    stp d12, d13, [sp, #32]
    stp d14, d15, [sp, #48]
    ptrue p0.h
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
    fdup z24.h, #1.0
    fdup z25.h, #1.0

.Lfp16_loop:
    fmla z0.h, p0/m, z24.h, z25.h
    fmla z1.h, p0/m, z24.h, z25.h
    fmla z2.h, p0/m, z24.h, z25.h
    fmla z3.h, p0/m, z24.h, z25.h
    fmla z4.h, p0/m, z24.h, z25.h
    fmla z5.h, p0/m, z24.h, z25.h
    fmla z6.h, p0/m, z24.h, z25.h
    fmla z7.h, p0/m, z24.h, z25.h
    fmla z8.h, p0/m, z24.h, z25.h
    fmla z9.h, p0/m, z24.h, z25.h
    fmla z10.h, p0/m, z24.h, z25.h
    fmla z11.h, p0/m, z24.h, z25.h
    fmla z12.h, p0/m, z24.h, z25.h
    fmla z13.h, p0/m, z24.h, z25.h
    fmla z14.h, p0/m, z24.h, z25.h
    fmla z15.h, p0/m, z24.h, z25.h
    fmla z16.h, p0/m, z24.h, z25.h
    fmla z17.h, p0/m, z24.h, z25.h
    fmla z18.h, p0/m, z24.h, z25.h
    fmla z19.h, p0/m, z24.h, z25.h
    fmla z20.h, p0/m, z24.h, z25.h
    fmla z21.h, p0/m, z24.h, z25.h
    fmla z22.h, p0/m, z24.h, z25.h
    fmla z23.h, p0/m, z24.h, z25.h
    subs x0, x0, #1
    bne .Lfp16_loop

    ldp d14, d15, [sp, #48]
    ldp d12, d13, [sp, #32]
    ldp d10, d11, [sp, #16]
    ldp d8, d9, [sp], #64
    ret
    .size bench_fmla_fp16, .-bench_fmla_fp16

// ─── FP16 FMLA with ld1rh (like GEMM kernel) ───
// x0 = iteration count, x1 = A pointer (dummy)
    .global bench_fmla_fp16_withld
    .type bench_fmla_fp16_withld, %function
bench_fmla_fp16_withld:
    stp d8, d9, [sp, #-64]!
    stp d10, d11, [sp, #16]
    stp d12, d13, [sp, #32]
    stp d14, d15, [sp, #48]
    ptrue p0.h
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
    fdup z24.h, #1.0
    fdup z25.h, #1.0

.Lfp16ld_loop:
    // 12 FMLAs + 6 ld1rh interleaved (like batch 1 of kernel)
    ld1rh {z26.h}, p0/z, [x1, #0]
    ld1rh {z27.h}, p0/z, [x1, #2]
    ld1rh {z28.h}, p0/z, [x1, #4]
    ld1rh {z29.h}, p0/z, [x1, #6]
    ld1rh {z30.h}, p0/z, [x1, #8]
    ld1rh {z31.h}, p0/z, [x1, #10]

    fmla z0.h, p0/m, z26.h, z24.h
    fmla z1.h, p0/m, z26.h, z25.h
    fmla z2.h, p0/m, z27.h, z24.h
    fmla z3.h, p0/m, z27.h, z25.h
    fmla z4.h, p0/m, z28.h, z24.h
    fmla z5.h, p0/m, z28.h, z25.h
    fmla z6.h, p0/m, z29.h, z24.h
    fmla z7.h, p0/m, z29.h, z25.h
    fmla z8.h, p0/m, z30.h, z24.h
    fmla z9.h, p0/m, z30.h, z25.h
    fmla z10.h, p0/m, z31.h, z24.h
    fmla z11.h, p0/m, z31.h, z25.h

    // 12 more FMLAs (batch 2, reuse same A values)
    fmla z12.h, p0/m, z26.h, z24.h
    fmla z13.h, p0/m, z26.h, z25.h
    fmla z14.h, p0/m, z27.h, z24.h
    fmla z15.h, p0/m, z27.h, z25.h
    fmla z16.h, p0/m, z28.h, z24.h
    fmla z17.h, p0/m, z28.h, z25.h
    fmla z18.h, p0/m, z29.h, z24.h
    fmla z19.h, p0/m, z29.h, z25.h
    fmla z20.h, p0/m, z30.h, z24.h
    fmla z21.h, p0/m, z30.h, z25.h
    fmla z22.h, p0/m, z31.h, z24.h
    fmla z23.h, p0/m, z31.h, z25.h

    subs x0, x0, #1
    bne .Lfp16ld_loop

    ldp d14, d15, [sp, #48]
    ldp d12, d13, [sp, #32]
    ldp d10, d11, [sp, #16]
    ldp d8, d9, [sp], #64
    ret
    .size bench_fmla_fp16_withld, .-bench_fmla_fp16_withld

// ─── FP32 FMLA with ld1rw (like GEMM kernel) ───
// x0 = iteration count, x1 = A pointer (dummy)
    .global bench_fmla_fp32_withld
    .type bench_fmla_fp32_withld, %function
bench_fmla_fp32_withld:
    stp d8, d9, [sp, #-64]!
    stp d10, d11, [sp, #16]
    stp d12, d13, [sp, #32]
    stp d14, d15, [sp, #48]
    ptrue p0.s
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
    fdup z24.s, #1.0
    fdup z25.s, #1.0

.Lfp32ld_loop:
    ld1rw {z26.s}, p0/z, [x1, #0]
    ld1rw {z27.s}, p0/z, [x1, #4]
    ld1rw {z28.s}, p0/z, [x1, #8]
    ld1rw {z29.s}, p0/z, [x1, #12]
    ld1rw {z30.s}, p0/z, [x1, #16]
    ld1rw {z31.s}, p0/z, [x1, #20]

    fmla z0.s, p0/m, z26.s, z24.s
    fmla z1.s, p0/m, z26.s, z25.s
    fmla z2.s, p0/m, z27.s, z24.s
    fmla z3.s, p0/m, z27.s, z25.s
    fmla z4.s, p0/m, z28.s, z24.s
    fmla z5.s, p0/m, z28.s, z25.s
    fmla z6.s, p0/m, z29.s, z24.s
    fmla z7.s, p0/m, z29.s, z25.s
    fmla z8.s, p0/m, z30.s, z24.s
    fmla z9.s, p0/m, z30.s, z25.s
    fmla z10.s, p0/m, z31.s, z24.s
    fmla z11.s, p0/m, z31.s, z25.s

    fmla z12.s, p0/m, z26.s, z24.s
    fmla z13.s, p0/m, z26.s, z25.s
    fmla z14.s, p0/m, z27.s, z24.s
    fmla z15.s, p0/m, z27.s, z25.s
    fmla z16.s, p0/m, z28.s, z24.s
    fmla z17.s, p0/m, z28.s, z25.s
    fmla z18.s, p0/m, z29.s, z24.s
    fmla z19.s, p0/m, z29.s, z25.s
    fmla z20.s, p0/m, z30.s, z24.s
    fmla z21.s, p0/m, z30.s, z25.s
    fmla z22.s, p0/m, z31.s, z24.s
    fmla z23.s, p0/m, z31.s, z25.s

    subs x0, x0, #1
    bne .Lfp32ld_loop

    ldp d14, d15, [sp, #48]
    ldp d12, d13, [sp, #32]
    ldp d10, d11, [sp, #16]
    ldp d8, d9, [sp], #64
    ret
    .size bench_fmla_fp32_withld, .-bench_fmla_fp32_withld
