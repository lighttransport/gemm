// FMLA latency measurement via dependency chains
    .arch armv8.2-a+sve
    .text
    .align 4

// ─── FP32 FMLA: 1 accumulator (= latency) ───
    .global bench_fp32_lat1
    .type bench_fp32_lat1, %function
bench_fp32_lat1:
    ptrue p0.s
    fdup z24.s, #1.0
    eor z0.d, z0.d, z0.d
.Lfp32_l1:
    fmla z0.s, p0/m, z24.s, z24.s
    fmla z0.s, p0/m, z24.s, z24.s
    fmla z0.s, p0/m, z24.s, z24.s
    fmla z0.s, p0/m, z24.s, z24.s
    fmla z0.s, p0/m, z24.s, z24.s
    fmla z0.s, p0/m, z24.s, z24.s
    fmla z0.s, p0/m, z24.s, z24.s
    fmla z0.s, p0/m, z24.s, z24.s
    subs x0, x0, #1
    bne .Lfp32_l1
    ret
    .size bench_fp32_lat1, .-bench_fp32_lat1

// ─── FP16 FMLA: 1 accumulator (= latency) ───
    .global bench_fp16_lat1
    .type bench_fp16_lat1, %function
bench_fp16_lat1:
    ptrue p0.h
    fdup z24.h, #1.0
    eor z0.d, z0.d, z0.d
.Lfp16_l1:
    fmla z0.h, p0/m, z24.h, z24.h
    fmla z0.h, p0/m, z24.h, z24.h
    fmla z0.h, p0/m, z24.h, z24.h
    fmla z0.h, p0/m, z24.h, z24.h
    fmla z0.h, p0/m, z24.h, z24.h
    fmla z0.h, p0/m, z24.h, z24.h
    fmla z0.h, p0/m, z24.h, z24.h
    fmla z0.h, p0/m, z24.h, z24.h
    subs x0, x0, #1
    bne .Lfp16_l1
    ret
    .size bench_fp16_lat1, .-bench_fp16_lat1

// ─── FP32 FMLA: 9 accumulators (should saturate if lat=9) ───
    .global bench_fp32_lat9
    .type bench_fp32_lat9, %function
bench_fp32_lat9:
    stp d8, d9, [sp, #-16]!
    ptrue p0.s
    fdup z24.s, #1.0
    eor z0.d, z0.d, z0.d
    eor z1.d, z1.d, z1.d
    eor z2.d, z2.d, z2.d
    eor z3.d, z3.d, z3.d
    eor z4.d, z4.d, z4.d
    eor z5.d, z5.d, z5.d
    eor z6.d, z6.d, z6.d
    eor z7.d, z7.d, z7.d
    eor z8.d, z8.d, z8.d
.Lfp32_l9:
    // 18 FMLAs cycling through 9 accumulators (each used 2x)
    fmla z0.s, p0/m, z24.s, z24.s
    fmla z1.s, p0/m, z24.s, z24.s
    fmla z2.s, p0/m, z24.s, z24.s
    fmla z3.s, p0/m, z24.s, z24.s
    fmla z4.s, p0/m, z24.s, z24.s
    fmla z5.s, p0/m, z24.s, z24.s
    fmla z6.s, p0/m, z24.s, z24.s
    fmla z7.s, p0/m, z24.s, z24.s
    fmla z8.s, p0/m, z24.s, z24.s
    fmla z0.s, p0/m, z24.s, z24.s
    fmla z1.s, p0/m, z24.s, z24.s
    fmla z2.s, p0/m, z24.s, z24.s
    fmla z3.s, p0/m, z24.s, z24.s
    fmla z4.s, p0/m, z24.s, z24.s
    fmla z5.s, p0/m, z24.s, z24.s
    fmla z6.s, p0/m, z24.s, z24.s
    fmla z7.s, p0/m, z24.s, z24.s
    fmla z8.s, p0/m, z24.s, z24.s
    subs x0, x0, #1
    bne .Lfp32_l9
    ldp d8, d9, [sp], #16
    ret
    .size bench_fp32_lat9, .-bench_fp32_lat9

// ─── FP16 FMLA: 9 accumulators ───
    .global bench_fp16_lat9
    .type bench_fp16_lat9, %function
bench_fp16_lat9:
    stp d8, d9, [sp, #-16]!
    ptrue p0.h
    fdup z24.h, #1.0
    eor z0.d, z0.d, z0.d
    eor z1.d, z1.d, z1.d
    eor z2.d, z2.d, z2.d
    eor z3.d, z3.d, z3.d
    eor z4.d, z4.d, z4.d
    eor z5.d, z5.d, z5.d
    eor z6.d, z6.d, z6.d
    eor z7.d, z7.d, z7.d
    eor z8.d, z8.d, z8.d
.Lfp16_l9:
    fmla z0.h, p0/m, z24.h, z24.h
    fmla z1.h, p0/m, z24.h, z24.h
    fmla z2.h, p0/m, z24.h, z24.h
    fmla z3.h, p0/m, z24.h, z24.h
    fmla z4.h, p0/m, z24.h, z24.h
    fmla z5.h, p0/m, z24.h, z24.h
    fmla z6.h, p0/m, z24.h, z24.h
    fmla z7.h, p0/m, z24.h, z24.h
    fmla z8.h, p0/m, z24.h, z24.h
    fmla z0.h, p0/m, z24.h, z24.h
    fmla z1.h, p0/m, z24.h, z24.h
    fmla z2.h, p0/m, z24.h, z24.h
    fmla z3.h, p0/m, z24.h, z24.h
    fmla z4.h, p0/m, z24.h, z24.h
    fmla z5.h, p0/m, z24.h, z24.h
    fmla z6.h, p0/m, z24.h, z24.h
    fmla z7.h, p0/m, z24.h, z24.h
    fmla z8.h, p0/m, z24.h, z24.h
    subs x0, x0, #1
    bne .Lfp16_l9
    ldp d8, d9, [sp], #16
    ret
    .size bench_fp16_lat9, .-bench_fp16_lat9

// ─── FP16 FMLA: 12 accumulators ───
    .global bench_fp16_lat12
    .type bench_fp16_lat12, %function
bench_fp16_lat12:
    stp d8, d9, [sp, #-64]!
    stp d10, d11, [sp, #16]
    stp d12, d13, [sp, #32]
    stp d14, d15, [sp, #48]
    ptrue p0.h
    fdup z24.h, #1.0
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
.Lfp16_l12:
    // 24 FMLAs cycling through 12 accumulators (each used 2x)
    fmla z0.h, p0/m, z24.h, z24.h
    fmla z1.h, p0/m, z24.h, z24.h
    fmla z2.h, p0/m, z24.h, z24.h
    fmla z3.h, p0/m, z24.h, z24.h
    fmla z4.h, p0/m, z24.h, z24.h
    fmla z5.h, p0/m, z24.h, z24.h
    fmla z6.h, p0/m, z24.h, z24.h
    fmla z7.h, p0/m, z24.h, z24.h
    fmla z8.h, p0/m, z24.h, z24.h
    fmla z9.h, p0/m, z24.h, z24.h
    fmla z10.h, p0/m, z24.h, z24.h
    fmla z11.h, p0/m, z24.h, z24.h
    fmla z0.h, p0/m, z24.h, z24.h
    fmla z1.h, p0/m, z24.h, z24.h
    fmla z2.h, p0/m, z24.h, z24.h
    fmla z3.h, p0/m, z24.h, z24.h
    fmla z4.h, p0/m, z24.h, z24.h
    fmla z5.h, p0/m, z24.h, z24.h
    fmla z6.h, p0/m, z24.h, z24.h
    fmla z7.h, p0/m, z24.h, z24.h
    fmla z8.h, p0/m, z24.h, z24.h
    fmla z9.h, p0/m, z24.h, z24.h
    fmla z10.h, p0/m, z24.h, z24.h
    fmla z11.h, p0/m, z24.h, z24.h
    subs x0, x0, #1
    bne .Lfp16_l12
    ldp d14, d15, [sp, #48]
    ldp d12, d13, [sp, #32]
    ldp d10, d11, [sp, #16]
    ldp d8, d9, [sp], #64
    ret
    .size bench_fp16_lat12, .-bench_fp16_lat12

// ─── FP32 FMLA: 12 accumulators ───
    .global bench_fp32_lat12
    .type bench_fp32_lat12, %function
bench_fp32_lat12:
    stp d8, d9, [sp, #-64]!
    stp d10, d11, [sp, #16]
    stp d12, d13, [sp, #32]
    stp d14, d15, [sp, #48]
    ptrue p0.s
    fdup z24.s, #1.0
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
.Lfp32_l12:
    fmla z0.s, p0/m, z24.s, z24.s
    fmla z1.s, p0/m, z24.s, z24.s
    fmla z2.s, p0/m, z24.s, z24.s
    fmla z3.s, p0/m, z24.s, z24.s
    fmla z4.s, p0/m, z24.s, z24.s
    fmla z5.s, p0/m, z24.s, z24.s
    fmla z6.s, p0/m, z24.s, z24.s
    fmla z7.s, p0/m, z24.s, z24.s
    fmla z8.s, p0/m, z24.s, z24.s
    fmla z9.s, p0/m, z24.s, z24.s
    fmla z10.s, p0/m, z24.s, z24.s
    fmla z11.s, p0/m, z24.s, z24.s
    fmla z0.s, p0/m, z24.s, z24.s
    fmla z1.s, p0/m, z24.s, z24.s
    fmla z2.s, p0/m, z24.s, z24.s
    fmla z3.s, p0/m, z24.s, z24.s
    fmla z4.s, p0/m, z24.s, z24.s
    fmla z5.s, p0/m, z24.s, z24.s
    fmla z6.s, p0/m, z24.s, z24.s
    fmla z7.s, p0/m, z24.s, z24.s
    fmla z8.s, p0/m, z24.s, z24.s
    fmla z9.s, p0/m, z24.s, z24.s
    fmla z10.s, p0/m, z24.s, z24.s
    fmla z11.s, p0/m, z24.s, z24.s
    subs x0, x0, #1
    bne .Lfp32_l12
    ldp d14, d15, [sp, #48]
    ldp d12, d13, [sp, #32]
    ldp d10, d11, [sp, #16]
    ldp d8, d9, [sp], #64
    ret
    .size bench_fp32_lat12, .-bench_fp32_lat12

// ─── FP16 FMLA: 18 accumulators ───
    .global bench_fp16_lat18
    .type bench_fp16_lat18, %function
bench_fp16_lat18:
    stp d8, d9, [sp, #-64]!
    stp d10, d11, [sp, #16]
    stp d12, d13, [sp, #32]
    stp d14, d15, [sp, #48]
    ptrue p0.h
    fdup z24.h, #1.0
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
.Lfp16_l18:
    // 18 FMLAs, one per accumulator (each used 1x per iter)
    fmla z0.h, p0/m, z24.h, z24.h
    fmla z1.h, p0/m, z24.h, z24.h
    fmla z2.h, p0/m, z24.h, z24.h
    fmla z3.h, p0/m, z24.h, z24.h
    fmla z4.h, p0/m, z24.h, z24.h
    fmla z5.h, p0/m, z24.h, z24.h
    fmla z6.h, p0/m, z24.h, z24.h
    fmla z7.h, p0/m, z24.h, z24.h
    fmla z8.h, p0/m, z24.h, z24.h
    fmla z9.h, p0/m, z24.h, z24.h
    fmla z10.h, p0/m, z24.h, z24.h
    fmla z11.h, p0/m, z24.h, z24.h
    fmla z12.h, p0/m, z24.h, z24.h
    fmla z13.h, p0/m, z24.h, z24.h
    fmla z14.h, p0/m, z24.h, z24.h
    fmla z15.h, p0/m, z24.h, z24.h
    fmla z16.h, p0/m, z24.h, z24.h
    fmla z17.h, p0/m, z24.h, z24.h
    subs x0, x0, #1
    bne .Lfp16_l18
    ldp d14, d15, [sp, #48]
    ldp d12, d13, [sp, #32]
    ldp d10, d11, [sp, #16]
    ldp d8, d9, [sp], #64
    ret
    .size bench_fp16_lat18, .-bench_fp16_lat18

// ─── FP32 FMLA: 18 accumulators ───
    .global bench_fp32_lat18
    .type bench_fp32_lat18, %function
bench_fp32_lat18:
    stp d8, d9, [sp, #-64]!
    stp d10, d11, [sp, #16]
    stp d12, d13, [sp, #32]
    stp d14, d15, [sp, #48]
    ptrue p0.s
    fdup z24.s, #1.0
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
.Lfp32_l18:
    fmla z0.s, p0/m, z24.s, z24.s
    fmla z1.s, p0/m, z24.s, z24.s
    fmla z2.s, p0/m, z24.s, z24.s
    fmla z3.s, p0/m, z24.s, z24.s
    fmla z4.s, p0/m, z24.s, z24.s
    fmla z5.s, p0/m, z24.s, z24.s
    fmla z6.s, p0/m, z24.s, z24.s
    fmla z7.s, p0/m, z24.s, z24.s
    fmla z8.s, p0/m, z24.s, z24.s
    fmla z9.s, p0/m, z24.s, z24.s
    fmla z10.s, p0/m, z24.s, z24.s
    fmla z11.s, p0/m, z24.s, z24.s
    fmla z12.s, p0/m, z24.s, z24.s
    fmla z13.s, p0/m, z24.s, z24.s
    fmla z14.s, p0/m, z24.s, z24.s
    fmla z15.s, p0/m, z24.s, z24.s
    fmla z16.s, p0/m, z24.s, z24.s
    fmla z17.s, p0/m, z24.s, z24.s
    subs x0, x0, #1
    bne .Lfp32_l18
    ldp d14, d15, [sp, #48]
    ldp d12, d13, [sp, #32]
    ldp d10, d11, [sp, #16]
    ldp d8, d9, [sp], #64
    ret
    .size bench_fp32_lat18, .-bench_fp32_lat18
