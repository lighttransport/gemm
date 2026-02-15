// int16_sdot_peak.S
// Minimal INT16 SDOT kernel to measure peak throughput
// Tests if A64FX can sustain 4 SDOT/cycle for INT16

    .arch armv8.2-a+sve
    .text

// ============================================
// Peak throughput test: 40 SDOT per iteration
// If 4 SDOT/cycle, should take 10 cycles
// ============================================

    .align 6
    .global int16_sdot_peak_40
    .type int16_sdot_peak_40, %function

int16_sdot_peak_40:
    // x0: A pointer
    // x1: B pointer
    // x2: C pointer
    // x3: iterations

    stp     d8, d9, [sp, #-16]!
    stp     d10, d11, [sp, #-16]!
    stp     d12, d13, [sp, #-16]!
    stp     d14, d15, [sp, #-16]!

    ptrue   p0.h
    ptrue   p1.d

    // Initialize 20 accumulators
    fmov    z0.d, #0
    fmov    z1.d, #0
    fmov    z2.d, #0
    fmov    z3.d, #0
    fmov    z4.d, #0
    fmov    z5.d, #0
    fmov    z6.d, #0
    fmov    z7.d, #0
    fmov    z8.d, #0
    fmov    z9.d, #0
    fmov    z10.d, #0
    fmov    z11.d, #0
    fmov    z12.d, #0
    fmov    z13.d, #0
    fmov    z14.d, #0
    fmov    z15.d, #0
    fmov    z16.d, #0
    fmov    z17.d, #0
    fmov    z18.d, #0
    fmov    z19.d, #0

    // Load A (5 rows) and B (4 vectors) once
    ld1rqh  z20.h, p0/z, [x0, #0]
    ld1rqh  z21.h, p0/z, [x0, #16]
    ld1rqh  z22.h, p0/z, [x0, #32]
    ld1rqh  z23.h, p0/z, [x0, #48]
    ld1rqh  z24.h, p0/z, [x0, #64]

    ld1h    z28.h, p0/z, [x1, #0, mul vl]
    ld1h    z29.h, p0/z, [x1, #1, mul vl]
    ld1h    z30.h, p0/z, [x1, #2, mul vl]
    ld1h    z31.h, p0/z, [x1, #3, mul vl]

    cbz     x3, .Lpeak40_done

    .align 6
.Lpeak40_loop:
    // 40 SDOT: 20 unique + 20 repeated (to test 4-way issue)
    // Group 1: 4 independent SDOT
    sdot    z0.d, z20.h, z28.h
    sdot    z1.d, z20.h, z29.h
    sdot    z2.d, z20.h, z30.h
    sdot    z3.d, z20.h, z31.h

    // Group 2: 4 independent SDOT
    sdot    z4.d, z21.h, z28.h
    sdot    z5.d, z21.h, z29.h
    sdot    z6.d, z21.h, z30.h
    sdot    z7.d, z21.h, z31.h

    // Group 3
    sdot    z8.d, z22.h, z28.h
    sdot    z9.d, z22.h, z29.h
    sdot    z10.d, z22.h, z30.h
    sdot    z11.d, z22.h, z31.h

    // Group 4
    sdot    z12.d, z23.h, z28.h
    sdot    z13.d, z23.h, z29.h
    sdot    z14.d, z23.h, z30.h
    sdot    z15.d, z23.h, z31.h

    // Group 5
    sdot    z16.d, z24.h, z28.h
    sdot    z17.d, z24.h, z29.h
    sdot    z18.d, z24.h, z30.h
    sdot    z19.d, z24.h, z31.h

    // Repeat 20 more to get 40 total
    sdot    z0.d, z20.h, z28.h
    sdot    z1.d, z20.h, z29.h
    sdot    z2.d, z20.h, z30.h
    sdot    z3.d, z20.h, z31.h

    sdot    z4.d, z21.h, z28.h
    sdot    z5.d, z21.h, z29.h
    sdot    z6.d, z21.h, z30.h
    sdot    z7.d, z21.h, z31.h

    sdot    z8.d, z22.h, z28.h
    sdot    z9.d, z22.h, z29.h
    sdot    z10.d, z22.h, z30.h
    sdot    z11.d, z22.h, z31.h

    sdot    z12.d, z23.h, z28.h
    sdot    z13.d, z23.h, z29.h
    sdot    z14.d, z23.h, z30.h
    sdot    z15.d, z23.h, z31.h

    sdot    z16.d, z24.h, z28.h
    sdot    z17.d, z24.h, z29.h
    sdot    z18.d, z24.h, z30.h
    sdot    z19.d, z24.h, z31.h

    subs    x3, x3, #1
    b.ne    .Lpeak40_loop

.Lpeak40_done:
    // Store one result to prevent optimization
    st1d    z0.d, p1, [x2]

    ldp     d14, d15, [sp], #16
    ldp     d12, d13, [sp], #16
    ldp     d10, d11, [sp], #16
    ldp     d8, d9, [sp], #16
    ret
    .size int16_sdot_peak_40, .-int16_sdot_peak_40
