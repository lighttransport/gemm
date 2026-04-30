# mm0 BF16 17% gap — logical decomposition (pre-implementation)

> **Update (after Tensile metadata inspection):** the kernel descriptor in the
> hipBLASLt `.co` exposes its full parameter set as a Tensile name string. Several
> assumptions in the original draft below were wrong. The corrected facts are
> in §0; the rest of the doc is preserved for reference and largely still
> holds, with the differences called out.

## 0. Ground truth from the kernel descriptor

Pulled from `llvm-readobj --notes /tmp/hipblaslt_bf16_alik_bljk.gfx1201.o` for
the kernel that contains `label_LoopBeginL` at file offset 0x34ec40. **This is
algo 73624 — the 166.6 TFLOP/s reference.**

```
Symbol: Cijk_Alik_Bljk_BSS_BH_Bias_HA_S_SAV_UserArgs
        _MT128x128x32 _MI16x16x1 _SN _LDSB0
        _GRVWA8 _GRVWB8 _LRVW8 _MIWT4_4
        _NLCA1 _NLCB2 _PGR1 _PLR1 _SIA3 _TLDS2
        _WS32 _WG32_4_1 _LBSPPA128 _LPA16

vgpr_count:           256       ← MORE than ours (241)
sgpr_count:            80       ← MORE than ours (12)
kernarg_segment_size: 144       ← UserArgs ABI; ours is 32 (4-pointer)
max_flat_workgroup_size: 128    ← same as ours (4 waves × 32 lanes)
wavefront_size:        32       ← same as ours
```

### What this overturns from the prior draft

| Prior assumption | Truth from descriptor |
|---|---|
| "PGR2 (PrefetchGlobalRead=2)" | **PGR1.** The whole "port PGR2" plan title was a misread. |
| "MIWaveTile = 2×2 with InnerUnroll=4 → 16 WMMAs/iter" | **MIWaveTile = 4×4, InnerUnroll=1 → 16 WMMAs/iter.** Same count, different tile shape. |
| "VGPR ≤192 to unlock 2 WG/WGP" | **hipBLASLt uses 256 VGPRs and still beats us.** Occupancy is *not* the lever. Both kernels are at 1 WG/WGP. |
| "144 vs 160 LDS row stride is the gap" | LDS read offsets show 2560 stride (= 160 b128) only in the inner loop, but parameters say `LBSPPA128 + LPA16 = 144`. The 2560 likely comes from `LBSPPA × MIWT[0]` packing, not raw row stride. **The "stride" framing was misleading.** |

### Direct kernel-descriptor comparison

| Metric | Ours `barriersig-early` | hipBLASLt 73624 | Δ |
|---|---|---|---|
| `vgpr_count` | 238 | **256** | +18 (theirs uses more) |
| `sgpr_count` | 14 | **80** | +66 (per-load SGPRs + SRDs) |
| `group_segment_fixed_size` (LDS) | 36 864 B | **26 624 B** | **−10 240 B (−28%)** |
| `kernarg_segment_size` | 32 B (4 ptrs) | 144 B (UserArgs ABI) | structural |
| `max_flat_workgroup_size` | 128 | 128 | same |

The 28% smaller LDS allocation is the smoking gun. Where does it come from?

### `DTVB1` — the architectural divergence

Decoding the kernel symbol: **`DTVB1` = `DirectToVgprB = 1`.** hipBLASLt's
kernel **does not stage B through LDS at all** — B's `buffer_load_b128` writes
directly into VGPRs that feed WMMA, bypassing the ds_store → ds_load round-trip
that our kernel uses. Only A goes through LDS.

LDS-budget breakdown (BF16 / DepthU=32 / MT128x128):
- A buffer: 128 × 32 × 2 = 8 192 B, double-buffered = 16 384 B + LPA16 padding
- B buffer (ours only): 128 × 32 × 2 = 8 192 B, double-buffered = 16 384 B + pad
- Ours total ≈ 36 864 B (matches measurement) — A and B both staged
- hipBLASLt total ≈ 26 624 B — only A staged, B routed VMEM→VGPR→WMMA

**This is a fundamentally different algorithm**, not a schedule tweak.
Implications:
- Their critical path for B is `buffer_load → wait → WMMA` (2 stages).
- Ours is `global_load → ds_store → barrier → ds_load → WMMA` (5 stages).
- For B operand, that's 2–3 fewer pipeline stages per WMMA. The SIA3
  schedule's "1 buffer_load + 1 ds_load + 1 WMMA per slot" pattern is the
  *expression* of this — half the memory ops per WMMA are B's direct-load.

### What it confirms

- **SIA3** scheduling — the dominant lever. We do not have an SIA3-equivalent
  schedule; we have batched ds_load + WMMA stream + batched ds_store.
- **TLDS2** (Thread-Level Data Scheduling = 2) — this is the per-MFMA
  distribution machinery. Pairs with SIA3.
- **NLCA1, NLCB2** (NumLoadsCoalescedA=1, NumLoadsCoalescedB=2) — A loads in
  one coalesced burst, B in two. Matches the disasm pattern (4 X loads + 8 W
  loads in inner loop).
- **GRVW=8, LRVW=8** — same as ours (1 b128 per memory op).
- **MIWT4_4** — confirms 4×4 = 16 MFMAs per K-slice per wave; with LoopIters=2
  that's 32 WMMAs per K-stage. Matches our count.

### The 4-vs-8 W-load mystery resolved

Earlier draft said hipBLASLt does "8 b128 loads for W per thread vs our 4".
Actually: with NLCB2 they do **2 separate coalesced groups of 4 W b128
each = 8 total per outer iter**, but each group is in a different scheduling
slot. The "structural" difference isn't the per-thread tile width — it's that
the two coalesced groups straddle the LDS-write boundary in the WMMA stream,
giving the scheduler more freedom. Our kernel has 4 W-loads bunched in one
group, so the scheduler has fewer slots to spread them.

### Implications for the lever-attribution table (§5)

**Lever ranking after the descriptor evidence:**

| Lever | Rough share of the 17% gap | Rationale |
|---|---|---|
| **DTVB (Direct-To-VGPR for B)** — bypass LDS for B | **~7–10%** | Removes 2 pipeline stages per B operand; halves LDS pressure; matches the 28% LDS-budget delta and the SIA3 "1 bload + 1 dsload per WMMA" rhythm |
| **SIA3 + TLDS2 schedule** (per-MFMA distribution) | ~3–5% | Falls out partially from DTVB (with B not in LDS, the "interleave" pattern is forced — there's no batched ds_load to do) |
| **Per-load SGPR `soffset`** (s66–s75) | ~1–2% | Codegen artifact, real but small |
| **2× outer unroll + tighter wait countdown** | ~1–2% | Reduces branch overhead; visible in cycle counts |
| **Smaller WG-launch overhead** | ~1–2% | Falls out of the smaller epilogue |

**Total: ~13–21%, centered on ~17%** — consistent with the measured gap.

### Re-ordered attack plan (replaces the earlier ordering)

1. **Implement DTVB-style B routing** (the dominant lever). This is the structural
   change: B's `buffer_load_b128` writes directly into a VGPR window that's
   consumed by the next 4 WMMAs without ds_store/ds_load. Costs ~150 LOC of
   generator code: kernarg parsing, buffer-descriptor SRD, K+1 prefetch ring
   on VGPRs, and recomputing wait counters. Removes ~10 KB of LDS allocation.
2. **SIA3 schedule rhythm falls out** of (1) — once B is direct-load, the
   "1 buffer_load + 1 ds_load + 1 WMMA per slot" pattern is structurally
   simpler to emit than what we have now (which has to coordinate 16 ds_loads
   for a mix of A and B).
3. **Per-load SGPR offsets** are a small codegen polish, only worth doing
   inside the new emitter, not as a standalone patch.

**Revised best-realistic outcome**: a generator producing PGR1+SIA3+DTVB
should hit ~155–160 TFLOP/s (80–82% peak). The remaining 3–5% to hipBLASLt's
85% is encoding-level (per-load SGPR, instruction packing, exact wait
countdown sequence) and unlikely worth chasing once we have ≥80%.

### Why prior patches all regressed (now explainable)

Every patch in the 11-attempt series tried to **rearrange ops within the
existing A+B-both-in-LDS algorithm**. The dominant lever (DTVB) requires a
*different algorithm*, not a different schedule of the same ops. Any
"interleave VMEM and LDS" pattern fails when both operands still flow through
LDS — there isn't enough VMEM-port work in the loop body to actually
benefit from the interleave. hipBLASLt's interleave works because **half its
WMMA inputs come straight from VMEM**, doubling the VMEM-port issue rate.



**Setting:** M=1024, N=4608, K=4608, BF16 in / F32 out, RX 9070 XT (gfx1201).
**Baseline:** `mm0_bf16_asm_barriersig_early.co` — **142.0 TFLOP/s** (72.9% peak).
**Reference:** hipBLASLt algo 73624 — **166.6 TFLOP/s** (85.4% peak).
**Gap:** 17.4% wall-time, kernels otherwise identical (bit-exact output).

This document estimates *where* the 17% lives, from the assembly diff alone, before
committing engineering hours to a regenerator.

---

## 1. Sanity check — is this kernel compute-bound or memory-bound?

| Quantity | Value |
|---|---|
| Bytes streamed per WG | 16 KB × 144 outer K-iters = 2.25 MB |
| WGs total | (1024/128) × (4608/128) = 8 × 36 = 288 |
| Total HBM traffic | ≤ 162 MB (with WG-level reuse) |
| Wall time @ 166 TFLOP/s | 0.261 ms |
| HBM bandwidth × wall time | 640 GB/s × 0.261 ms = **167 MB** |

→ Even hipBLASLt sits ~near memory-bandwidth ceiling, but still **compute-bound**:
the 195 TFLOP/s peak corresponds to 0.232 ms; both kernels are 12–32% off peak,
so the WMMA pipe is what we're feeding, not HBM.

→ **The 17% gap must be in scheduling efficiency inside the WG**, not in
DRAM traffic. Cache reuse and tile shape are already correct.

---

## 2. Same WMMA count, same memory volume — so compare the *issue stream*

Per-K-iter inner-loop instruction census (one full K-stage = 32 WMMAs):

| Class | Ours `barriersig-early` | hipBLASLt 73624 | Δ |
|---|---|---|---|
| `v_wmma_f32_16x16x16_bf16` | 32 | 32 | — |
| Global / buffer load (K+1) | 8 × `global_load_b128` | 12 × `buffer_load_b128` | hipBLASLt has 4 W loads × 2 + 4 X loads (i.e. **W tile is wider per thread**) |
| `ds_load_b128` | 16 | 16 | — |
| `ds_store_b128` (K+1 LDS write) | 8 | 8 | — |
| `s_wait_*cnt` | 21 | 18 | **−3** |
| `s_barrier_signal/wait` | 2 | 2 | — |
| `v_add_co_u32 / v_addc_co_u32` (address math) | **7** | 0 | −7 |
| Inline scalar prep (per-load offsets, branch logic) | 4 | ~12 | +8 (in their favor — front-loads SGPR work) |
| **Total instructions / K-iter** | **~124** | **~90** | **−27%** |

**Key observation**: hipBLASLt has fewer instructions per K-iter despite doing
*more* W loads. The savings come from:
- No per-load address arithmetic (`v_add_co_u32` per iter is gone — replaced
  by buffer-descriptor SRD set up once in the prologue)
- Per-load SGPR `soffset` constants (s66–s75) → buffer_load uses only one of
  these instead of an `offset:N` immediate field (single-cycle issue)
- Tighter wait countdown (one wait per load rather than batched waits)

---

## 3. Cycle-budget decomposition

Per-WG SQ_BUSY_CYCLES (already measured, `rdna4_gemm_optimization_log.md`):

| Metric | Ours | hipBLASLt | Δ |
|---|---|---|---|
| Per-wave busy cycles | 24,100 | 21,979 | **−2,121 (−8.8%)** |
| Wall time | 0.3066 ms | 0.2613 ms | −14.8% |

**The wall-time gap (14.8%) is bigger than the per-wave gap (8.8%).** That
extra ~6% comes from **WG-level parallelism**, not per-wave efficiency:

- Both kernels saturate **1 WG/WGP** in steady state (LDS 36 KB + VGPRs both
  >128 → 1 WG/WGP for both per resource limits, *but* …)
- hipBLASLt's tighter loop body lets the **next WG launch overlap with the
  previous WG's epilogue store sequence** more effectively. Our `bb.5` is
  longer (more `v_add_co_u32` boilerplate before the final stores) so the
  WGP serialization at WG boundaries is worse.

VGPR check (refutes prior hypothesis from the optimization log):

| Kernel | Max VGPR | Allocated VGPRs |
|---|---|---|
| Ours | v237 (loop body) | **241** |
| hipBLASLt | v245 (loop body) | **246** (+5 more) |

→ **hipBLASLt uses MORE VGPRs and still wins.** The "VGPR ≤192 to unlock 2
WG/WGP" suggestion in the log is **wrong** — neither kernel hits 2 WG/WGP from
VGPR/LDS alone. The win is in the schedule, not occupancy.

---

## 4. Estimating each lever's contribution

Allocation of the 8.8% per-wave gap (2,121 cycles per wave):

### A. Per-load SGPR `soffset` (s66–s75 pattern) — **est. 3–4%**
hipBLASLt issues `buffer_load_b128 v[…], v_off, s[srd], s66 offen` where
`s66`–`s75` are pre-computed offset SGPRs. Our kernel uses `global_load_b128
v[…], v[157:158], off offset:64` with v_add_co address math.

- 8 loads × 144 iters × ~2 cycles saved per load = **~2,300 cycles**
- Caveat: this is an upper bound; actual savings depend on whether VMEM was
  the bottleneck issue port at those cycles. Probably ~half realized = ~1,150
  cycles ≈ **5% of slack**.

### B. Buffer-descriptor + SRD vs global_load address math — **est. 2–3%**
The 7 `v_add_co_u32 / v_addc_co_u32` per K-iter consume 7×1 = 7 cycles VALU
issue per iter × 144 = 1,008 cycles per wave. Buffer-load eliminates these
entirely.

But VALU is dual-issued with WMMA on RDNA4, so the realized cost is lower
(maybe 30%): **~300 cycles ≈ 1.5% of slack**.

### C. Interleave VMEM + LDS in WMMA stream (the SIA3-style) — **est. 3–4%**
hipBLASLt fires 1 `buffer_load` + 1 `ds_load` per WMMA in slots 0–3
(interleaving both memory ports), then `buffer_load`-only in slots 4–15.
Our kernel batches all 16 ds_loads at bb.4 entry, then all WMMAs in a row.

This was tried in `dsload-interleave-bse` and regressed −6%. The likely reason:
**we interleaved only ds_loads, not buffer_loads**. hipBLASLt interleaves both
because they hit separate ports (LDS vs VMEM). Without VMEM interleaved, our
patch increased serialization on a single port.

→ This requires the buffer-load conversion (item B) **first** to be testable.

### D. Loop unroll 2x — **est. 1–2%**
hipBLASLt's body covers 2 outer iterations (lines 1116–1296), halving branch
overhead. Branch + s_cmp + s_sub = ~3 cycles per iter saved by unrolling = 432
cycles per wave = **2% of slack**.

### E. Tighter wait countdown — **est. 1%**
hipBLASLt has 18 wait instructions vs our 21. Each `s_wait_*cnt` that
collapses saves a stall window (variable). Conservative: 200 cycles per wave.

### F. WG-launch overlap (the missing 6% in wall time) — **est. 5–6%**
Smaller WG epilogue (no per-load address math hangover) → next WG can launch
sooner on the WGP after barrier release. **This is "free" once items A+B land**.

---

## 5. Estimated decomposition table

| Lever | Estimated gain | Implementation cost | Risk |
|---|---|---|---|
| **A.** Per-load SGPR offsets s66–s75 | 4–5% | ~50 lines: prologue + 8 buffer_load patches | Low |
| **B.** SRD/buffer_load conversion | 1.5–2% | ~30 lines: prologue setup, already partly in `pgr2_bufload_noguard_bse` | Low (already tested) |
| **C.** VMEM+LDS interleave | 3–4% | ~80 lines: re-emit bb.4 with combined slots | Medium (must redo wait counters) |
| **D.** 2x unroll | 1–2% | ~150 lines: emit loop body twice with reg renaming | Medium (VGPR pressure risk) |
| **E.** Wait collapse | ~1% | falls out of A–D naturally | None |
| **F.** WG-launch overlap | 5–6% | falls out of A–B naturally | None |
| **Sum (best case)** | **15–20%** | ~250–300 LOC of generator | — |
| **Sum (60% realization)** | **9–12%** | same | — |
| **What was missing from prior log** | LDS stride 144 → 160 was a **distractor** (offset:2304 appears in hipBLASLt epilogues too); VGPR ≤192 hypothesis was wrong | — | — |

**Predicted outcome**: 60% realization of all levers ≈ **155–158 TFLOP/s
(80% peak)**, hitting the project's 80% target but ~5% short of hipBLASLt.
The remaining 5% is likely instruction-encoding nuances we won't replicate
without literally copying their generator output.

---

## 6. Recommended order of attack

1. **Item A first (per-load SGPR offsets)** — biggest single lever, lowest risk,
   reuses our existing `mm0_bf16_asm_pgr2_bufload_noguard_bse` codebase. Patch
   the existing buffer_load variant to add 8 SGPR offsets in prologue and use
   them in the 8 mainloop loads.
2. **Item C only if A passes correctness** — combined VMEM+LDS interleave with
   tightened wait counters. This is where the prior `dsload-interleave-bse`
   patch failed; with item A in place, the failure mode (VMEM-port serialization
   from missing buffer_load interleave) goes away.
3. **Items D+E later** — diminishing returns; only worth it if 1+2 land us
   inside 5% of target.

**Stop condition**: if item A alone yields <2% gain on the bench, the cycle
model above is wrong and a full Tensile-style rewrite is required. If item A
yields 3%+, items B+C+E will likely close to 80% peak.

---

## 7. What this analysis does NOT cover

- **Tensile-internal `numMfmaPerIter` heuristic value** — we assume it matches
  ours (16). If the disasm shows a different value, all percentages shift.
- **RDNA4-specific WMMA throughput** (16 vs 8 cycles per WMMA in wave32) — the
  *ratio* of our gap to hipBLASLt's is what matters; absolute cycle floor is
  not needed for this estimate.
- **Hardware perf counter validation** — the `rocprofv3` runs got killed at
  30+ min waiting for full PMC sweep. A targeted single-pass profile (just
  SQ_INST_CYCLES_VMEM and SQ_WAIT_INST_ANY) would confirm whether items A+B
  actually save VMEM-port cycles. **Should be run before committing to the
  generator rewrite.**
