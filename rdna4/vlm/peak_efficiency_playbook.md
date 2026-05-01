# Generic ~90 % peak-efficiency GEMM playbook

A 10-step recipe distilled from the mm0 case study (174 TFLOP/s, 89 % peak,
RDNA4 BF16). Aimed at a single GEMM kernel on a single GPU; works for
HIP/Tensile-style codegen and is informative for hand-asm.

**Inputs you start with:**

- Problem: `(M, N, K, layout, dtype_in, dtype_acc)`
- HW spec: `peak_TFLOPs(dtype)`, `peak_DRAM_BW`, `LDS_bytes/CU`,
  `VGPR/SIMD`, `wave_size`, `WMMA_latency_cycles`, number of CUs/SIMDs

---

## Step 1 — Compute the ceilings *before* you touch anything

```
total_FLOP   = 2 · M · N · K
arith_int    = total_FLOP / (sizeof(in)·(M·K + K·N) + sizeof(out)·M·N)   # bytes per FLOP^-1
roof_compute = peak_TFLOPs
roof_BW      = peak_DRAM_BW · arith_int
roof         = min(roof_compute, roof_BW)
target_90    = 0.9 · roof
```

If `roof_BW < roof_compute` you are **memory-bound** — stop reading this
playbook and switch to a BW-optimization recipe (split-K, persistent
kernel, batched fusion). Everything below assumes compute-bound.

For mm0: AI = 2·1024·4608·4608 / (2·(1024·4608 + 4608·4608) + 4·1024·4608)
= 43.5 GFLOP / 65.6 MB = 663 FLOP/B → roof_BW @ 640 GB/s = 424 TFLOP/s ≫
roof_compute = 195 TFLOP/s. Compute-bound. Target = 175 TFLOP/s.

---

## Step 2 — Pick the macro tile (MT) and wave-tile (MIWT)

Constraints:

1. **LDS fit:** `(MT_M·MT_K + MT_K·MT_N) · sizeof(in) · pad_factor ≤ LDS_bytes/CU` (or just one operand if you go DTV).
2. **VGPR budget:** `MIWT_M · MIWT_N · sizeof(acc)/4 + prefetch_regs ≤ VGPR/SIMD`. For BF16→FP32 with MIWT4×4 the C-tile alone is 4·4·16·16 = 4096 elts / 32 lanes = 128 VGPRs, leaving 128 for everything else.
3. **WMMA amortization:** `MIWT_M · MIWT_N · K_per_iter / WMMA_K ≥ 32` so each loop body issues ≥32 WMMAs (loop overhead < 5 %).
4. **WG occupancy target:** at least 2 waves per WG (typically 4 for
   wave32 GPUs). MT_M / (MIWT_M · WMMA_M) × MT_N / (MIWT_N · WMMA_N) =
   waves/WG.

Default starting point on RDNA4: `MT=128×128×32, MIWT=4×4, WG32_4_1` (4
waves/WG, 1 WG/SIMD). Adjust upward if VGPR budget allows.

---

## Step 3 — Choose DTV (direct-to-VGPR) operand

DTV bypasses LDS for one operand: load global → register → WMMA. Saves
LDS round-trip and one stage of pipelining.

- **DTVB** (B direct): use when N is large and B is read once per K-iter
  (typical NT GEMM with N as the contiguous output dim). This is what mm0
  uses.
- **DTVA**: use when M dominates and A is the streamed operand.
- **No DTV**: use when both operands need LDS sharing across waves (e.g.
  TT layout with both operands bank-conflict-prone).

Cost: DTV operand needs `MIWT_X · K_per_iter / GRVW_X` VGPRs reserved
per K-iter. Tightens VGPR budget; may force lower MIWT.

---

## Step 4 — Set prefetch depth (PGR)

Prefetch depth = how many K-tiles' worth of VMEM are in flight before the
WMMA on tile 0 starts.

```
PGR ≈ ceil( VMEM_round_trip_cycles / WMMA_cycles_per_K_iter )
```

- `VMEM_round_trip ≈ 200–400 cycles` on consumer RDNA4.
- `WMMA_cycles_per_K_iter = MIWT_M·MIWT_N · (K_per_iter/WMMA_K) · WMMA_lat`.

For mm0: 4·4·2·16 = 512 WMMA cycles/iter → ratio ≈ 0.5 → PGR1 *technically*
suffices for steady-state. **But PGR2 still wins +5 TFLOP/s** because:
1. The first iter has nothing to overlap with → PGR2 hides cold-start.
2. Deeper VMEM queue → better DRAM channel utilization.
3. PGR2 enables more aggressive WMMA dual-issue scheduling.

**Rule:** start at PGR2 if the VGPR budget allows (it costs 2× the
prefetch register window). Drop to PGR1 only if VGPR-pressure forces it.

---

## Step 5 — Vectorize loads and stores aggressively

| Knob | What it controls | Default |
|---|---|---|
| `GRVW_A`, `GRVW_B` | Global-read vector width (= b128 = 16 B for BF16/FP16/FP8 → load 8/8/16 elts per inst) | **max** allowed by alignment (8 for BF16) |
| `VWA`, `VWB` | How many K-tiles packed into contiguous VGPR block per wave (controls WMMA register-bank layout) | **4** if VGPR allows, never 1 |
| `SVW` | Store vector width (epilogue) | **4** for FP32 output (= b128 store) |
| `LRVW` | LDS read vector width | **8** for BF16 |

**The single biggest lever in the mm0 study was VWA/VWB 1→4** (+12 TFLOP/s
alone). Reason: WMMA inputs read from VGPR banks; VWA=4 packs 4 K-tiles
into a contiguous block so consecutive WMMAs read from different sub-banks
→ no read-port stalls → WMMAs can dual-issue. **Never accept VWA=1 unless
forced by a tiny K dimension.**

---

## Step 6 — LDS padding (LBSPPx)

**Failure mode this fixes:** when an LDS row stride is an exact multiple of
the bank-page size (32 banks × 4 B = 128 B), all lanes of a wave hit the
same bank → bank conflicts → LDS read takes 4–8 cycles instead of 1.

**Recipe:**

```
LDS_row_stride_bytes = MT_K · sizeof(in) · LDSB_factor
if LDS_row_stride_bytes % 128 == 0:
    add padding via LBSPPA / LBSPPB until row stride is *not* a multiple of 128
```

For mm0: MT_K=32, BF16, no LDSB → row = 64 B → already off-page; LBSPPA128
*adds* a 128 B pad making rows 192 B. But the actual bank-page boundary
matters between *consecutive rows*: with 4 waves reading rows 0,1,2,3, the
LDS bank these rows hit must differ. **LBSPPA256 spaces rows by 256 B,
shifting the bank phase across the wave** → ~600 cycles/wave saved
(~3.5 TFLOP/s).

If the gfx1201 `SQC_LDS_BANK_CONFLICT` counter doesn't help (it returned
0 in our study despite real conflicts), trust the perf delta from a
LBSPPx sweep — sweep 0 / 64 / 128 / 256 / 512 and take the winner.

---

## Step 7 — Slot schedule (SIA3 vs SIA1)

- **RDNA4 / gfx12: SIA3** — interleave 1 VMEM + 1 LDS + 1 WMMA per slot.
  RDNA4's dual-issue across the SIMD pair makes this work.
- **CDNA3/4: SIA1** — bunch each class. Different SIMD architecture.

If you're on RDNA4 always start at SIA3. The mm0 win used SIA3 in *both*
OLD and NEW kernels — it's not a knob you tune, it's a knob you *set
correctly for the architecture* and never touch again.

---

## Step 8 — Validate with three counters

```
pmc: SQ_WAVES SQ_BUSY_CYCLES SQC_LDS_BANK_CONFLICT
```

Targets:

```
cycles_per_wave           = SQ_BUSY_CYCLES / SQ_WAVES
cycles_per_wave_floor     = (WMMAs_per_wave · WMMA_lat) / dual_issue_factor
utilization               = floor / cycles_per_wave    # aim ≥ 0.85
ldsbc_per_wave            = SQC_LDS_BANK_CONFLICT / SQ_WAVES   # ideally 0
```

For mm0 WIN: cycles/wave = 20,363, ms = 0.233, peak ratio = 89 % ✓.
On gfx1201 the bank-conflict counter is unreliable — fall back to perf
deltas from LBSPPx sweep.

---

## Step 9 — When to stop

| Achieved peak | Action |
|---|---|
| ≥ 87 % | **Ship.** Going further is diminishing returns. |
| 80–87 % | Try ±1 PGR, ±1 SVW, ±256 LBSPPA in a 3×3 sweep. Stop if no win. |
| 70–80 % | Likely missing one of the big levers (VW, DTV, prefetch). Re-derive Steps 3–5 from scratch. |
| < 70 % | Structural issue: wrong MT, wrong DTV operand, or you're memory-bound (re-check Step 1). |

The mm0 study found **17 % peak across two structurally identical kernels**
just from second-order tunables. Always sweep them; never trust a default.

---

## Step 10 — Sweep as a fallback

If the target HW ships with a Tensile/hipBLASLt kernel zoo (`.co` files
under `/opt/rocm-*/lib/hipblaslt/library/`), it likely contains hundreds of
pre-compiled variants for your shape's macro tile. Filter to your `(MT,
MIWT, ISA, layout, dtype)` and sweep with a launcher that loads the .co
directly (see `mm0_extracted_launcher.cpp`).

This is **how the mm0 +19 TFLOP/s was actually found** — the playbook
above lets you predict the right region of knobs, but a 50-variant sweep
in that region is faster than reasoning from first principles for the last
few percent. Cost: ~5 min wall-clock.

---

## Appendix: knob → mechanism cheat sheet

| Knob | What it does | When to try |
|---|---|---|
| MT | macro tile size | Step 2 |
| MIWT | per-wave WMMA tile | Step 2 |
| WG dim | waves per WG | Step 2 |
| DTVA / DTVB | direct-to-VGPR for one operand | Step 3 |
| PGR | global-read prefetch depth | Step 4 |
| PLR | LDS-read prefetch depth (1 vs 0) | Almost always 1 |
| GRVW_A / GRVW_B | global-read vec width | Step 5 (max) |
| VWA / VWB | VGPR pack width = WMMA bank layout | **Step 5 — biggest lever** |
| SVW | store vec width | Step 5 |
| LBSPPA / LBSPPB | LDS row pad | Step 6 |
| LDSB | LDS double-buffer for B | enable when LDS allows |
| TLDS | LDS-init / pipeline hint | platform-default |
| CLR | clear LDS at iter end | usually 0 |
| SS | scheduler split | platform-default |
| SIA | issue interleave | Step 7 (architecture) |
| ONLL | optimized no-load-loop | always 1 |
| K-loop unroll (PLR effect) | how many K-tiles per body iter | tied to PLR |

If a knob isn't listed in Steps 2–7, leave it at the Tensile default.
