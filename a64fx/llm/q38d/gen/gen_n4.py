#!/usr/bin/env python3
"""Generate multi-token F4 A16 verification kernels (q38d_kern_n4.S).

Interface (as q38d_asmn4_f4_a16 in q38d_kern_sve.S):
  void name(const uint8_t *code, const uint8_t *unused, const uint8_t *scale,
            const int8_t *actq, const float *actsc, long np, float *acc,
            const int8_t *lut);
Activations interleaved per pair: actq[np][NT][64] (per token 32 low, then
32 high digits), actsc[np][NT][2]; acc[NT][16] (lane 2r+u). np >= 4.

Pipeline, one pair per slot: S4(s-3) S3(s-2) S2(s-1) S1(s).
Token t uses four depth-2 SDOT chains: low digits (l0,l1) (h0,h1), high
digits (l0,l1) (h0,h1).
"""
import sys

def kernel(name, nt):
    assert nt * 4 + nt + 12 <= 32, "register budget"
    L = []
    e = L.append
    chains = lambda t: [8 + 4 * t + k for k in range(4)]
    accs = [8 + 4 * nt + t for t in range(nt)]
    tmp = 8 + 5 * nt           # S4 temp
    e(f"    .global {name}\n    .type {name}, %function\n    .p2align 6\n{name}:")
    e("    stp d8, d9, [sp, #-64]!\n    stp d10, d11, [sp, #16]\n    stp d12, d13, [sp, #32]\n    stp d14, d15, [sp, #48]")
    e("    ptrue p0.b\n    ptrue p1.s\n    ptrue p2.d\n    ld1b {z31.b}, p0/z, [x7]\n    dup z30.b, #15\n    dup z29.s, #0")
    for a in accs: e(f"    dup z{a}.s, #0")
    e("    mov x9, #PF2_DIST")
    def S1():
        e("    prfm pldl1keep, [x0, #PF1_DIST]\n    prfm pldl2keep, [x0, x9]")
        e("    ld1b {z0.b}, p0/z, [x0]\n    ld1b {z1.b}, p0/z, [x0, #1, mul vl]\n    add x0, x0, #128")
    def S2():
        e("    and z4.d, z0.d, z30.d\n    lsr z5.b, z0.b, #4\n    and z6.d, z1.d, z30.d\n    lsr z7.b, z1.b, #4")
        e("    tbl z4.b, {z31.b}, z4.b\n    tbl z5.b, {z31.b}, z5.b\n    tbl z6.b, {z31.b}, z6.b\n    tbl z7.b, {z31.b}, z7.b")
    def S3():
        for step, (w, wh, o) in enumerate([(4, 5, 0), (6, 7, 16)]):
            for t in range(nt):
                c = chains(t)
                for k, (wr, off) in enumerate([(w, o), (wh, o + 8), (w, o + 32), (wh, o + 40)]):
                    e(f"    ld1rd {{z3.d}}, p2/z, [x3, #{64 * t + off}]")
                    if step == 0: e(f"    movprfx z{c[k]}, z29")
                    e(f"    sdot z{c[k]}.s, z{wr}.b, z3.b")
        e(f"    add x3, x3, #{64 * nt}")
    def S4():
        e("    ld1b {z2.s}, p1/z, [x2]\n    add x2, x2, #16\n    lsl z2.s, z2.s, #20")
        for t in range(nt):
            c = chains(t)
            e(f"    add z{c[0]}.s, z{c[0]}.s, z{c[1]}.s\n    add z{c[2]}.s, z{c[2]}.s, z{c[3]}.s")
            e(f"    ld1rd {{z{tmp}.d}}, p2/z, [x4, #{8 * t}]")
            e(f"    lsl z{c[2]}.s, z{c[2]}.s, #8\n    fmul z{tmp}.s, z{tmp}.s, z2.s")
            e(f"    add z{c[0]}.s, z{c[0]}.s, z{c[2]}.s\n    scvtf z{c[0]}.s, p1/m, z{c[0]}.s")
            e(f"    fmla z{accs[t]}.s, p1/m, z{c[0]}.s, z{tmp}.s")
        e(f"    add x4, x4, #{8 * nt}")
    S1(); S2(); S1(); S3(); S2(); S1()
    e("    sub x5, x5, #3\n1:")
    S4(); S3(); S2(); S1()
    e("    subs x5, x5, #1\n    b.ne 1b")
    S4(); S3(); S2(); S4(); S3(); S4()
    for t, a in enumerate(accs): e(f"    st1w {{z{a}.s}}, p1, [x6, #{t}, mul vl]")
    e("    ldp d10, d11, [sp, #16]\n    ldp d12, d13, [sp, #32]\n    ldp d14, d15, [sp, #48]\n    ldp d8, d9, [sp], #64\n    ret")
    e(f"    .size {name}, .-{name}\n")
    return "\n".join(L)

def kernel_g2(name, nt, pipelined=True):
    """nt tokens x two 8-row groups (A = code, B = code + gstride) per slot.
    void name(const uint8_t *code, long gstride, const uint8_t *scale,
              const int8_t *actq, const float *actsc, long np, float *acc,
              const int8_t *lut)   -- acc[2][nt][16]; scale B = scale + gstride.
    Chains per (token, group): high digits (movprfx, 4 SDOT), LSL 8, then the
    low-digit SDOTs continue the same register."""
    L = []
    e = L.append
    G = 2
    dec = {0: [4, 5, 6, 7], 1: [8, 9, 10, 11]}
    ch = {(t, g): 12 + 2 * t + g for t in range(nt) for g in range(G)}
    acc = {(t, g): 12 + 2 * nt + 2 * t + g for t in range(nt) for g in range(G)}
    nxt = 12 + 4 * nt
    ws = [nxt, nxt + 1]; tmp = nxt + 2
    assert tmp < 29, "register budget"
    e(f"    .global {name}\n    .type {name}, %function\n    .p2align 6\n{name}:")
    e("    stp d8, d9, [sp, #-64]!\n    stp d10, d11, [sp, #16]\n    stp d12, d13, [sp, #32]\n    stp d14, d15, [sp, #48]")
    e("    ptrue p0.b\n    ptrue p1.s\n    ptrue p2.d\n    ld1b {z31.b}, p0/z, [x7]\n    dup z30.b, #15\n    dup z29.s, #0")
    for k in acc.values(): e(f"    dup z{k}.s, #0")
    e("    mov x9, #PF2_DIST\n    add x10, x0, x1\n    add x12, x2, x1")
    def S12():   # load + decode both groups
        e("    prfm pldl1keep, [x0, #PF1_DIST]\n    prfm pldl1keep, [x10, #PF1_DIST]")
        e("    prfm pldl2keep, [x0, x9]\n    prfm pldl2keep, [x10, x9]")
        e("    ld1b {z0.b}, p0/z, [x0]\n    ld1b {z1.b}, p0/z, [x0, #1, mul vl]")
        e("    ld1b {z2.b}, p0/z, [x10]\n    ld1b {z3.b}, p0/z, [x10, #1, mul vl]")
        e("    add x0, x0, #128\n    add x10, x10, #128")
        for g, (c0, c1) in enumerate([(0, 1), (2, 3)]):
            d = dec[g]
            e(f"    and z{d[0]}.d, z{c0}.d, z30.d\n    lsr z{d[1]}.b, z{c0}.b, #4\n    and z{d[2]}.d, z{c1}.d, z30.d\n    lsr z{d[3]}.b, z{c1}.b, #4")
        for g in range(G):
            for r in dec[g]: e(f"    tbl z{r}.b, {{z31.b}}, z{r}.b")
    def S3():
        # high digits: weight vector j (l0 h0 l1 h1) at act offset 32 + 8j
        for half, base in ((0, 32), (1, 0)):
            for j in range(4):
                for t in range(nt):
                    b = [0, 1, 2, 3][(t + j) % 4]   # rotate broadcast temps (renamed anyway)
                    e(f"    ld1rd {{z{b}.d}}, p2/z, [x3, #{64 * t + base + 8 * j}]")
                    for g in range(G):
                        c = ch[(t, g)]
                        if half == 0 and j == 0: e(f"    movprfx z{c}, z29")
                        e(f"    sdot z{c}.s, z{dec[g][j]}.b, z{b}.b")
            if half == 0:
                for t in range(nt):
                    for g in range(G):
                        c = ch[(t, g)]
                        e(f"    lsl z{c}.s, z{c}.s, #8")
        e(f"    add x3, x3, #{64 * nt}")
    def S4():
        e(f"    ld1b {{z{ws[0]}.s}}, p1/z, [x2]\n    ld1b {{z{ws[1]}.s}}, p1/z, [x12]\n    add x2, x2, #16\n    add x12, x12, #16")
        e(f"    lsl z{ws[0]}.s, z{ws[0]}.s, #20\n    lsl z{ws[1]}.s, z{ws[1]}.s, #20")
        for t in range(nt):
            e(f"    ld1rd {{z{tmp}.d}}, p2/z, [x4, #{8 * t}]")
            for g in range(G):
                c = ch[(t, g)]
                e(f"    scvtf z{c}.s, p1/m, z{c}.s")
                e(f"    fmul z28.s, z{tmp}.s, z{ws[g]}.s")
                e(f"    fmla z{acc[(t, g)]}.s, p1/m, z{c}.s, z28.s")
        e(f"    add x4, x4, #{8 * nt}")
    # two-stage pipeline: slot s = S4(s-1) S12(s) S3(s)
    S12(); S3()
    e("    sub x5, x5, #1\n1:")
    S4(); S12(); S3()
    e("    subs x5, x5, #1\n    b.ne 1b")
    S4()
    for g in range(G):
        for t in range(nt):
            e(f"    st1w {{z{acc[(t, g)]}.s}}, p1, [x6, #{g * nt + t}, mul vl]")
    e("    ldp d10, d11, [sp, #16]\n    ldp d12, d13, [sp, #32]\n    ldp d14, d15, [sp, #48]\n    ldp d8, d9, [sp], #64\n    ret")
    e(f"    .size {name}, .-{name}\n")
    return "\n".join(L)

def kernel_c(name, nt=4):
    """Four tokens, one group, depth-4 chains (token t: low z(b+2t), high
    z(b+2t+1)) in two alternating register sets (b = 8 / 16) so the combine
    stage of pair s-3 interleaves with the SDOT steps of pair s-2.
    Same interface as q38d_asmn4_f4_a16. np even, np >= 6."""
    assert nt == 4
    L = []
    e = L.append
    ACC = [24, 25, 26, 27]
    def chains(par): return [(8 + 8 * par + 2 * t, 9 + 8 * par + 2 * t) for t in range(nt)]
    e(f"    .global {name}\n    .type {name}, %function\n    .p2align 6\n{name}:")
    e("    stp d8, d9, [sp, #-64]!\n    stp d10, d11, [sp, #16]\n    stp d12, d13, [sp, #32]\n    stp d14, d15, [sp, #48]")
    e("    ptrue p0.b\n    ptrue p1.s\n    ptrue p2.d\n    ld1b {z31.b}, p0/z, [x7]\n    dup z30.b, #15\n    dup z29.s, #0")
    for a in ACC: e(f"    dup z{a}.s, #0")
    e("    mov x9, #PF2_DIST")
    def S1():
        e("    prfm pldl1keep, [x0, #PF1_DIST]\n    prfm pldl2keep, [x0, x9]")
        e("    ld1b {z0.b}, p0/z, [x0]\n    ld1b {z1.b}, p0/z, [x0, #1, mul vl]\n    add x0, x0, #128")
    def S2():
        e("    and z4.d, z0.d, z30.d\n    lsr z5.b, z0.b, #4\n    and z6.d, z1.d, z30.d\n    lsr z7.b, z1.b, #4")
        e("    tbl z4.b, {z31.b}, z4.b\n    tbl z5.b, {z31.b}, z5.b\n    tbl z6.b, {z31.b}, z6.b\n    tbl z7.b, {z31.b}, z7.b")
    def S3step(par, j):
        w = 4 + j
        for t, (lo, hi) in enumerate(chains(par)):
            for reg, off in ((lo, 64 * t + 8 * j), (hi, 64 * t + 32 + 8 * j)):
                e(f"    ld1rd {{z3.d}}, p2/z, [x3, #{off}]")
                if j == 0: e(f"    movprfx z{reg}, z29")
                e(f"    sdot z{reg}.s, z{w}.b, z3.b")
    def S4pre():
        e("    ld1b {z2.s}, p1/z, [x2]\n    add x2, x2, #16\n    lsl z2.s, z2.s, #20")
    def S4tok(par, t):
        lo, hi = chains(par)[t]
        e(f"    ld1rd {{z28.d}}, p2/z, [x4, #{8 * t}]")
        e(f"    lsl z{hi}.s, z{hi}.s, #8\n    add z{lo}.s, z{lo}.s, z{hi}.s")
        e(f"    fmul z28.s, z28.s, z2.s\n    scvtf z{lo}.s, p1/m, z{lo}.s")
        e(f"    fmla z{ACC[t]}.s, p1/m, z{lo}.s, z28.s")
    def slot(s3par, s4par, do_s1=True, do_s2=True, do_s3=True, do_s4=True):
        # interleave: S3 steps with S4 tokens; S2, S1 late
        if do_s4: S4pre()
        for j in range(4):
            if do_s3: S3step(s3par, j)
            if do_s4: S4tok(s4par, j)
            if j == 2 and do_s2: S2()
        if do_s3: e("    add x3, x3, #256")
        if do_s4: e("    add x4, x4, #32")
        if do_s1: S1()
    # prologue: s=0 S1(0); s=1 S2(0) S1(1); s=2 S3(0) S2(1) S1(2)
    S1()
    S2(); S1()
    slot(0, None, do_s4=False)
    # loop: slots s=3..np-1 (np-3 odd): body = slot(s odd) + slot(s even)
    # slot s: S3 pair s-2 (parity s%2), S4 pair s-3 (parity (s-1)%2)
    e("    sub x5, x5, #4\n    lsr x5, x5, #1\n1:")
    slot(1, 0)
    slot(0, 1)
    e("    subs x5, x5, #1\n    b.ne 1b")
    slot(1, 0)                 # s = np-1
    slot(0, 1, do_s1=False)    # s = np: S4(np-3) S3(np-2) S2(np-1)
    slot(1, 0, do_s1=False, do_s2=False)   # s = np+1: S4(np-2) S3(np-1)
    slot(None, 1, do_s1=False, do_s2=False, do_s3=False)  # S4(np-1)
    for t, a in enumerate(ACC): e(f"    st1w {{z{a}.s}}, p1, [x6, #{t}, mul vl]")
    e("    ldp d10, d11, [sp, #16]\n    ldp d12, d13, [sp, #32]\n    ldp d14, d15, [sp, #48]\n    ldp d8, d9, [sp], #64\n    ret")
    e(f"    .size {name}, .-{name}\n")
    return "\n".join(L)

def kernel_g2n4(name, nt=4, share_as=False):
    """nt tokens x two groups; one depth-9 chain per (token, group): high-digit
    SDOTs (movprfx from zero), LSL 8, low-digit SDOTs on the same register.
    Slot s (program order): S4(s-1) S3(s) S1(s+1) S2(s+1).
    Interface as q38d_asmg2n*: code, gstride, scale, actq[np][nt][64],
    actsc[np][nt][2], np, acc[2][nt][16], lut. np >= 2.
      z0,z1 codes | z2 bcast | z3 ws | z4-z7 dec A | z8-z11 dec B
      z12.. chains (2*nt) | accumulators (2*nt) | z28 as temp | z29 zero | z30 0x0f | z31 lut"""
    L = []
    e = L.append
    dec = {0: [4, 5, 6, 7], 1: [8, 9, 10, 11]}
    ch = {(t, g): 12 + 2 * t + g for t in range(nt) for g in range(2)}
    acc = {(t, g): 12 + 2 * nt + 2 * t + g for t in range(nt) for g in range(2)}
    assert 12 + 4 * nt <= 28
    e(f"    .global {name}\n    .type {name}, %function\n    .p2align 6\n{name}:")
    e("    stp d8, d9, [sp, #-64]!\n    stp d10, d11, [sp, #16]\n    stp d12, d13, [sp, #32]\n    stp d14, d15, [sp, #48]")
    e("    ptrue p0.b\n    ptrue p1.s\n    ptrue p2.d\n    ld1b {z31.b}, p0/z, [x7]\n    dup z30.b, #15\n    dup z29.s, #0")
    for k in acc.values(): e(f"    dup z{k}.s, #0")
    e("    mov x9, #PF2_DIST\n    add x10, x0, x1\n    add x12, x2, x1")
    def S12():
        for g, base in ((0, "x0"), (1, "x10")):
            d = dec[g]
            e(f"    prfm pldl1keep, [{base}, #PF1_DIST]\n    prfm pldl2keep, [{base}, x9]")
            e(f"    ld1b {{z0.b}}, p0/z, [{base}]\n    ld1b {{z1.b}}, p0/z, [{base}, #1, mul vl]\n    add {base}, {base}, #128")
            e(f"    and z{d[0]}.d, z0.d, z30.d\n    lsr z{d[1]}.b, z0.b, #4\n    and z{d[2]}.d, z1.d, z30.d\n    lsr z{d[3]}.b, z1.b, #4")
            for r in d: e(f"    tbl z{r}.b, {{z31.b}}, z{r}.b")
    def S3():
        for half, base in ((0, 32), (1, 0)):
            for j in range(4):
                for t in range(nt):
                    e(f"    ld1rd {{z2.d}}, p2/z, [x3, #{64 * t + base + 8 * j}]")
                    for g in range(2):
                        c = ch[(t, g)]
                        if half == 0 and j == 0: e(f"    movprfx z{c}, z29")
                        e(f"    sdot z{c}.s, z{dec[g][j]}.b, z2.b")
            if half == 0:
                for t in range(nt):
                    for g in range(2):
                        c = ch[(t, g)]
                        e(f"    lsl z{c}.s, z{c}.s, #8")
        e(f"    add x3, x3, #{64 * nt}")
    def S4():
        if share_as:
            # weight scales of both groups in z3 / z2 (z2 is free before S3),
            # one activation-scale load per token, products in z0 / z1
            e("    ld1b {z3.s}, p1/z, [x2]\n    ld1b {z2.s}, p1/z, [x12]\n    add x2, x2, #16\n    add x12, x12, #16")
            e("    lsl z3.s, z3.s, #20\n    lsl z2.s, z2.s, #20")
            for t in range(nt):
                e(f"    ld1rd {{z28.d}}, p2/z, [x4, #{8 * t}]")
                e("    fmul z0.s, z28.s, z3.s\n    fmul z1.s, z28.s, z2.s")
                cA, cB = ch[(t, 0)], ch[(t, 1)]
                e(f"    scvtf z{cA}.s, p1/m, z{cA}.s\n    scvtf z{cB}.s, p1/m, z{cB}.s")
                e(f"    fmla z{acc[(t, 0)]}.s, p1/m, z{cA}.s, z0.s\n    fmla z{acc[(t, 1)]}.s, p1/m, z{cB}.s, z1.s")
            e(f"    add x4, x4, #{8 * nt}")
            return
        for g, sp in ((0, "x2"), (1, "x12")):
            e(f"    ld1b {{z3.s}}, p1/z, [{sp}]\n    add {sp}, {sp}, #16\n    lsl z3.s, z3.s, #20")
            for t in range(nt):
                c = ch[(t, g)]
                e(f"    ld1rd {{z28.d}}, p2/z, [x4, #{8 * t}]")
                e(f"    scvtf z{c}.s, p1/m, z{c}.s\n    fmul z28.s, z28.s, z3.s")
                e(f"    fmla z{acc[(t, g)]}.s, p1/m, z{c}.s, z28.s")
        e(f"    add x4, x4, #{8 * nt}")
    S12()                      # decode pair 0
    e("    sub x5, x5, #1")
    S3(); S12()                # slot 0: S3(0) S12(1)
    e("    cbz x5, 2f\n1:")
    S4(); S3()
    e("    subs x5, x5, #1\n    b.eq 3f")
    S12()
    e("    b 1b\n3:\n2:")
    S4()
    for g in range(2):
        for t in range(nt):
            e(f"    st1w {{z{acc[(t, g)]}.s}}, p1, [x6, #{g * nt + t}, mul vl]")
    e("    ldp d10, d11, [sp, #16]\n    ldp d12, d13, [sp, #32]\n    ldp d14, d15, [sp, #48]\n    ldp d8, d9, [sp], #64\n    ret")
    e(f"    .size {name}, .-{name}\n")
    return "\n".join(L)

def kernel_g2p(name, nt):
    """As kernel_g2n4(share_as=True) but activations are per-token pointers in
    the standard single-token layout:
    void name(const uint8_t *code, long gstride, const uint8_t *scale,
              const int8_t *const *aq, const float *const *asc, long np,
              float *acc, const int8_t *lut)   -- acc[2][nt][16].
    B codes/scales = A + gstride (gstride 0: the same group twice).
    Activation pointers per token: x13.. (q), x19.. (scale pairs)."""
    L = []
    e = L.append
    dec = {0: [4, 5, 6, 7], 1: [8, 9, 10, 11]}
    ch = {(t, g): 12 + 2 * t + g for t in range(nt) for g in range(2)}
    acc = {(t, g): 12 + 2 * nt + 2 * t + g for t in range(nt) for g in range(2)}
    AQ = ["x13", "x14", "x15", "x16"][:nt]
    AS = ["x19", "x20", "x21", "x22"][:nt]
    e(f"    .global {name}\n    .type {name}, %function\n    .p2align 6\n{name}:")
    e("    stp d8, d9, [sp, #-96]!\n    stp d10, d11, [sp, #16]\n    stp d12, d13, [sp, #32]\n    stp d14, d15, [sp, #48]")
    e("    stp x19, x20, [sp, #64]\n    stp x21, x22, [sp, #80]")
    for t in range(nt):
        e(f"    ldr {AQ[t]}, [x3, #{8 * t}]\n    ldr {AS[t]}, [x4, #{8 * t}]")
    e("    ptrue p0.b\n    ptrue p1.s\n    ptrue p2.d\n    ld1b {z31.b}, p0/z, [x7]\n    dup z30.b, #15\n    dup z29.s, #0")
    for k in acc.values(): e(f"    dup z{k}.s, #0")
    e("    mov x9, #PF2_DIST\n    add x10, x0, x1\n    add x12, x2, x1")
    def S12():
        for g, base in ((0, "x0"), (1, "x10")):
            d = dec[g]
            e(f"    prfm pldl1keep, [{base}, #PF1_DIST]\n    prfm pldl2keep, [{base}, x9]")
            e(f"    ld1b {{z0.b}}, p0/z, [{base}]\n    ld1b {{z1.b}}, p0/z, [{base}, #1, mul vl]\n    add {base}, {base}, #128")
            e(f"    and z{d[0]}.d, z0.d, z30.d\n    lsr z{d[1]}.b, z0.b, #4\n    and z{d[2]}.d, z1.d, z30.d\n    lsr z{d[3]}.b, z1.b, #4")
            for r in d: e(f"    tbl z{r}.b, {{z31.b}}, z{r}.b")
    def S3():
        for half, base in ((0, 32), (1, 0)):
            for j in range(4):
                for t in range(nt):
                    e(f"    ld1rd {{z2.d}}, p2/z, [{AQ[t]}, #{base + 8 * j}]")
                    for g in range(2):
                        c = ch[(t, g)]
                        if half == 0 and j == 0: e(f"    movprfx z{c}, z29")
                        e(f"    sdot z{c}.s, z{dec[g][j]}.b, z2.b")
            if half == 0:
                for t in range(nt):
                    for g in range(2):
                        c = ch[(t, g)]
                        e(f"    lsl z{c}.s, z{c}.s, #8")
        for t in range(nt): e(f"    add {AQ[t]}, {AQ[t]}, #64")
    def S4():
        e("    ld1b {z3.s}, p1/z, [x2]\n    ld1b {z2.s}, p1/z, [x12]\n    add x2, x2, #16\n    add x12, x12, #16")
        e("    lsl z3.s, z3.s, #20\n    lsl z2.s, z2.s, #20")
        for t in range(nt):
            e(f"    ld1rd {{z28.d}}, p2/z, [{AS[t]}]\n    add {AS[t]}, {AS[t]}, #8")
            e("    fmul z0.s, z28.s, z3.s\n    fmul z1.s, z28.s, z2.s")
            cA, cB = ch[(t, 0)], ch[(t, 1)]
            e(f"    scvtf z{cA}.s, p1/m, z{cA}.s\n    scvtf z{cB}.s, p1/m, z{cB}.s")
            e(f"    fmla z{acc[(t, 0)]}.s, p1/m, z{cA}.s, z0.s\n    fmla z{acc[(t, 1)]}.s, p1/m, z{cB}.s, z1.s")
    S12()
    e("    sub x5, x5, #1")
    S3(); S12()
    e("    cbz x5, 2f\n1:")
    S4(); S3()
    e("    subs x5, x5, #1\n    b.eq 3f")
    S12()
    e("    b 1b\n3:\n2:")
    S4()
    for g in range(2):
        for t in range(nt):
            e(f"    st1w {{z{acc[(t, g)]}.s}}, p1, [x6, #{g * nt + t}, mul vl]")
    e("    ldp x19, x20, [sp, #64]\n    ldp x21, x22, [sp, #80]")
    e("    ldp d10, d11, [sp, #16]\n    ldp d12, d13, [sp, #32]\n    ldp d14, d15, [sp, #48]\n    ldp d8, d9, [sp], #96\n    ret")
    e(f"    .size {name}, .-{name}\n")
    return "\n".join(L)

def kernel_q8k_g2p(name, nt):
    """Q8K (Q6_K expanded to int8), nt tokens x two groups, per-token pointers:
    void name(const uint8_t *code, long gstride, const uint8_t *sc,
              const int8_t *const *aq, const float *const *asc, long np,
              float *acc, const float *d)  -- acc[2][nt][16], np % 8 == 0.
    Weights per pair: l0 h0 l1 h1 (4 x 64 int8); sc[np][16] int8 sub-scales
    (lane 2r+u); d[np/8][8] floats per row. Slot for pair p:
    S4(p-1) [d reload after S4 at p % 8 == 0] S3(p) L(p+1).
      z0/z1 d vectors A/B | z2 bcast / scale B | z3 scale A | z4-z11 weights
      z12.. chains, accumulators | z28 act scale | z29 zero | z30/z31 products"""
    L = []
    e = L.append
    W = {0: [4, 5, 6, 7], 1: [8, 9, 10, 11]}
    ch = {(t, g): 12 + 2 * t + g for t in range(nt) for g in range(2)}
    acc = {(t, g): 12 + 2 * nt + 2 * t + g for t in range(nt) for g in range(2)}
    AQ = ["x13", "x14", "x15", "x16"][:nt]
    AS = ["x19", "x20", "x21", "x22"][:nt]
    e(f"    .global {name}\n    .type {name}, %function\n    .p2align 6\n{name}:")
    e("    stp d8, d9, [sp, #-96]!\n    stp d10, d11, [sp, #16]\n    stp d12, d13, [sp, #32]\n    stp d14, d15, [sp, #48]")
    e("    stp x19, x20, [sp, #64]\n    stp x21, x22, [sp, #80]")
    for t in range(nt):
        e(f"    ldr {AQ[t]}, [x3, #{8 * t}]\n    ldr {AS[t]}, [x4, #{8 * t}]")
    e("    ptrue p0.b\n    ptrue p1.s\n    ptrue p2.d\n    ptrue p3.s, vl8\n    dup z29.s, #0")
    for k in acc.values(): e(f"    dup z{k}.s, #0")
    e("    mov x9, #PF2_DIST\n    add x10, x0, x1\n    add x12, x2, x1\n    add x17, x7, x1\n    lsr x5, x5, #3")
    def L_():
        for g, base in ((0, "x0"), (1, "x10")):
            w = W[g]
            e(f"    prfm pldl1keep, [{base}, #PF1_DIST]\n    prfm pldl2keep, [{base}, x9]")
            for i in range(4): e(f"    ld1b {{z{w[i]}.b}}, p0/z, [{base}, #{i}, mul vl]")
            e(f"    add {base}, {base}, #256")
    def D():
        e("    ld1w {z0.s}, p3/z, [x7]\n    ld1w {z1.s}, p3/z, [x17]\n    add x7, x7, #32\n    add x17, x17, #32")
        e("    zip1 z0.s, z0.s, z0.s\n    zip1 z1.s, z1.s, z1.s")
    def S3():
        for half, base in ((0, 32), (1, 0)):
            for j in range(4):
                for t in range(nt):
                    e(f"    ld1rd {{z2.d}}, p2/z, [{AQ[t]}, #{base + 8 * j}]")
                    for g in range(2):
                        c = ch[(t, g)]
                        if half == 0 and j == 0: e(f"    movprfx z{c}, z29")
                        e(f"    sdot z{c}.s, z{W[g][j]}.b, z2.b")
            if half == 0:
                for t in range(nt):
                    for g in range(2):
                        c = ch[(t, g)]
                        e(f"    lsl z{c}.s, z{c}.s, #8")
        for t in range(nt): e(f"    add {AQ[t]}, {AQ[t]}, #64")
    def S4():
        e("    ld1sb {z3.s}, p1/z, [x2]\n    ld1sb {z2.s}, p1/z, [x12]\n    add x2, x2, #16\n    add x12, x12, #16")
        e("    scvtf z3.s, p1/m, z3.s\n    scvtf z2.s, p1/m, z2.s\n    fmul z3.s, z3.s, z0.s\n    fmul z2.s, z2.s, z1.s")
        for t in range(nt):
            e(f"    ld1rd {{z28.d}}, p2/z, [{AS[t]}]\n    add {AS[t]}, {AS[t]}, #8")
            e("    fmul z30.s, z28.s, z3.s\n    fmul z31.s, z28.s, z2.s")
            cA, cB = ch[(t, 0)], ch[(t, 1)]
            e(f"    scvtf z{cA}.s, p1/m, z{cA}.s\n    scvtf z{cB}.s, p1/m, z{cB}.s")
            e(f"    fmla z{acc[(t, 0)]}.s, p1/m, z{cA}.s, z30.s\n    fmla z{acc[(t, 1)]}.s, p1/m, z{cB}.s, z31.s")
    # block 0 (no S4 before pair 0)
    L_(); D()
    for j in range(8):
        if j: S4()
        S3(); L_()
    e("    subs x5, x5, #1\n    b.eq 3f\n1:")
    for j in range(8):
        S4()
        if j == 0: D()
        S3(); L_()
    e("    subs x5, x5, #1\n    b.ne 1b\n3:")
    S4()
    for g in range(2):
        for t in range(nt):
            e(f"    st1w {{z{acc[(t, g)]}.s}}, p1, [x6, #{g * nt + t}, mul vl]")
    e("    ldp x19, x20, [sp, #64]\n    ldp x21, x22, [sp, #80]")
    e("    ldp d10, d11, [sp, #16]\n    ldp d12, d13, [sp, #32]\n    ldp d14, d15, [sp, #48]\n    ldp d8, d9, [sp], #96\n    ret")
    e(f"    .size {name}, .-{name}\n")
    return "\n".join(L)

out = ["/* Generated by gen/gen_n4.py -- do not edit. */",
       "#ifndef PF1_DIST\n#define PF1_DIST 1024\n#endif\n#ifndef PF2_DIST\n#define PF2_DIST 16384\n#endif",
       "    .text\n    .arch armv8.2-a+sve\n"]
for nt in (2, 3, 4):
    out.append(kernel(f"q38d_asmnb{nt}_f4_a16", nt))
for nt in (1, 2, 3):
    out.append(kernel_g2(f"q38d_asmg2n{nt}_f4_a16", nt))
out.append(kernel_c("q38d_asmn4c_f4_a16"))
for nt in (1, 2, 4):
    out.append(kernel_g2n4(f"q38d_asmg2c{nt}_f4_a16", nt))
for nt in (2, 4):
    out.append(kernel_g2n4(f"q38d_asmg2s{nt}_f4_a16", nt, share_as=True))
for nt in (2, 3, 4):
    out.append(kernel_g2p(f"q38d_asmg2p{nt}_f4_a16", nt))
for nt in (2, 3, 4):
    out.append(kernel_q8k_g2p(f"q38d_asmg2p{nt}_q8k_a16", nt))
out.append('    .section .note.GNU-stack,"",%progbits\n')
open(sys.argv[1], "w").write("\n".join(out))
