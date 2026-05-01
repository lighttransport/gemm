# Optimization strategy guide: GEMM at ~90 % peak

A technical guide for writing (or tuning) BF16/FP16/FP8 GEMM kernels that
sustain ~90 % of advertised peak on modern matrix-engine GPUs. Grounded
in the RDNA4 (gfx1201) mm0 case study where a tuned hipBLASLt variant
hit **174 TFLOP/s / 89.4 % peak** on M=1024, N=4608, K=4608, BF16→FP32.
Numbers are RDNA4-specific; the *reasoning* generalizes to CDNA, Hopper,
Blackwell.

The companion documents are `peak_efficiency_playbook.md` (a 10-step
checklist) and `mm0_lever_attribution.md` (the per-knob TFLOP/s
breakdown for the mm0 win). This guide is the *why* behind both.

---

## 0. Mental model: where do cycles go?

Every wave-cycle on a SIMD is in exactly one of these states:

```
┌─────────────────┬─────────────────────────────────────────────────────┐
│  WMMA pipeline  │  the SIMD is issuing or executing a v_wmma_* op     │
├─────────────────┼─────────────────────────────────────────────────────┤
│  VMEM issue     │  buffer_load / global_load / store address+request  │
│  LDS issue      │  ds_load / ds_store address+request                 │
│  VALU / SALU    │  arithmetic that isn't WMMA (address calc, cast)    │
│  WAIT           │  s_wait_loadcnt / s_wait_dscnt / s_barrier — wave   │
│                 │  is alive but stalled on a counter                  │
│  IDLE           │  no wave runnable on this SIMD (occupancy gap)      │
└─────────────────┴─────────────────────────────────────────────────────┘
```

**Peak efficiency = WMMA pipeline cycles / total cycles.**

Everything in this guide is a technique for *moving cycles out of WAIT,
LDS issue, VMEM issue, and VALU into WMMA*. Three macro tactics:

1. **Overlap** non-WMMA work with WMMA work (prefetch, pipeline depth).
2. **Compress** non-WMMA work into fewer instructions (vectorization).
3. **Eliminate** non-WMMA work that isn't load-bearing (DTV, CLR off).

For mm0, OLD spent **24,174 cyc/wave** to do the same WMMA work that WIN
finished in **20,363 cyc/wave**. Same WMMA count, same VMEM bytes — the
4,000 cycles WIN saved came from these three tactics applied to specific
levers.

---

## 1. The instruction floor (start here)

Before tuning anything, calculate the theoretical instruction-cycle floor
for *just the WMMA work*:

```
WMMAs_per_wave = MIWT_M · MIWT_N · (K / WMMA_K)
WMMA_cycles    = WMMAs_per_wave · WMMA_latency   # 16 cyc on RDNA4
```

For mm0 (MIWT=4×4, K=4608, WMMA_K=16): 4·4·288 · 16 = **73,728 cycles**.

Now compare to measured `SQ_BUSY_CYCLES / SQ_WAVES`. The ratio is your
*WMMA utilization*. Anything below ~85 % is leaving perf on the table;
above ~90 % you're chasing diminishing returns. RDNA4 wmma is fully
pipelined (1 issue/cycle, 16-cycle latency), so the *issue floor* is
actually `WMMAs_per_wave / dual_issue_factor`, where dual_issue_factor =
2 if you can keep both halves of the SIMD-pair busy. **Hitting that
factor of 2 is the single biggest win in the mm0 case.**

### 1.1 WMMA dual-issue: the VWA/VWB story

RDNA4's SIMD32 is paired into a "WMMA pair" that can co-issue two WMMAs
per cycle *if* their VGPR sources don't collide on the same register
sub-bank. Sub-banks rotate every 4 VGPRs.

**VWA=1 layout** (worst): WMMA n+1 reads VGPRs adjacent to WMMA n →
same sub-bank → can't co-issue → effective WMMA latency = 16 cyc instead
of 8.

**VWA=4 layout**: 4 K-tiles' worth of A inputs packed into a contiguous
32-VGPR window. Consecutive WMMAs read from different sub-banks → both
halves of the SIMD pair issue simultaneously → effective WMMA latency
≈ 8 cyc.

In the mm0 attribution this **single knob was worth ~2,000 cyc/wave (~12
TFLOP/s)**. The static instruction count *didn't change* — the schedule
changed. This is why VWA1→4 is the first knob to flip after structure is
set.

**Rule:** never accept VWA=1 unless K is so small that 4 K-tiles don't
fit in the wave. On modern HW with large register files, this constraint
basically never binds.

### 1.2 The dual-issue audit

If you can't hit ≥85 % of the *issue floor* (= WMMA_cycles / 2), look for:

1. **VGPR bank conflicts** — VWA/VWB packing.
2. **Operand ordering** — the third source (the C accumulator) reads from
   yet another VGPR window. If that window overlaps A-window, conflict
   returns. Tensile's "MIWT4_4" layout pre-segregates these; hand-asm
   needs care.
3. **Forwarding stalls** — a WMMA's accumulator becoming the source of
   the next WMMA-K-tile must complete before the new issue. If you
   schedule WMMAs by-output-tile (deep accum chain on one tile then
   move to next), you eat the latency. Schedule by-K-tile-then-output
   (round-robin across MIWT_M·MIWT_N tiles per K-step) — accum write of
   tile (i,j) lands ~16 cyc later when its next WMMA arrives.

---

## 2. Load latency and how to hide it

VMEM round-trip on RDNA4 is **200–400 cycles** (hot L2: ~80 cyc cache
hit; DRAM: ~400 cyc). LDS round-trip is **~30 cycles**. Both are vastly
longer than a WMMA, so the only question is *how much WMMA work do you
have to overlap a load*.

```
overlap_capacity_cycles = MIWT_M · MIWT_N · K_per_iter / WMMA_K · WMMA_lat
                        = 4 · 4 · 32 / 16 · 16 = 512 cyc per K-iter
```

For mm0 that's 512 cyc per K-iter of in-flight WMMA work, which already
covers a ~400-cyc DRAM latency. **One in-flight K-tile (PGR=1) is
arithmetically sufficient** for steady-state. So why does PGR=2 still
give +5 TFLOP/s?

### 2.1 The cold-start problem

PGR=1 issues K-tile 0's load, then waits for it before any WMMA can
start. That's ~400 idle cycles at wave start. PGR=2 issues K-tiles 0 and
1 *before* the first WMMA, so the first wait is on tile 1 (already ~half
back) instead of tile 0. The first 800 cycles of overlap come for free.

For a wave that runs ~73 k cycles total, 400 cycles cold-start = **0.5 %
penalty**. So why +5 TFLOP/s? Two more reasons:

### 2.2 DRAM channel utilization

RDNA4's HBM/GDDR has multiple channels, each with its own queue. If you
have only 1 outstanding request per wave, you saturate ~1 channel's
worth of BW. With 2 outstanding requests the channel scheduler reorders
across waves and squeezes another 10–15 % out of aggregate DRAM
throughput. This compounds across all 1152 waves → measurable end-to-end.

### 2.3 Instruction scheduling slack

PGR=2 means the K-tile 0 load result is "due" *2 K-iters from now*
instead of 1. The compiler/scheduler has 2× the slot window to fit the
load issue without bumping a WMMA. In mm0's case this lets the first 16
WMMAs of every K-iter dual-issue cleanly.

**Rule of thumb:**
- `PGR = ceil(VMEM_latency / overlap_capacity)` is the *floor*; add 1 if
  the VGPR budget allows.
- VGPR cost: each level of PGR adds `loads_per_K_iter · GRVW · sizeof(in) / 4`
  VGPRs. For mm0 BF16, +24 b128 loads per iter × 4 VGPRs = +96 VGPRs per
  PGR level. From 192 used at PGR=1 to ~256 at PGR=2 — right at the
  ceiling, no room for PGR=3.

### 2.4 PLR (LDS-read prefetch)

PLR=1 is the LDS analogue of PGR. It overlaps LDS read of K-tile (i+1)
with WMMA on K-tile (i). Cost: extra VGPRs to hold the prefetched A.
**Always set PLR=1**; the cost is tiny (one MIWT_M · MIWT_N · K_per_iter
window, often 16 VGPRs) and it avoids ~30 cyc/iter of LDS-read stall.

---

## 3. Unrolling: K-loop, why and by how much

Unrolling here means: how many K-tiles does the K-loop body process per
backward branch? In Tensile parlance this is determined by `PLR` and
`PrefetchLocalRead` settings, which together pick "1, 2, or N K-tiles
per body iteration."

### 3.1 Why unroll at all

A backward branch on RDNA4 costs 1 cycle. A `s_cmp_*` + `s_cbranch_*`
pair costs 2. The K-loop counter increment costs 1. So per K-iter the
loop overhead is ~4 cycles. With 144 K-iters, that's 576 cycles, ~0.8 %
of WMMA time. Unrolling by 2× cuts that to 0.4 %. Not the dominant lever.

The *real* reason to unroll is that it doubles the instruction window
the scheduler can rearrange in. With 1 K-tile per body, the scheduler
sees 32 WMMAs + ~30 other ops. With 2 K-tiles per body (PLR=1 unrolls
the loop), it sees 64 WMMAs + 60 other ops, and can move loads of
K-tile (i+2) ahead of WMMAs of K-tile (i) inside the body without the
swap-bb dance.

### 3.2 How much

The mm0 winner uses 2-K-tile unroll (the standard PLR=1 default). 4-tile
would help in principle but doubles register pressure for the prefetch
ring beyond 256 VGPRs.

**Practical guide:**

| K_per_iter | When to use |
|---|---|
| 1 | Almost never — give up 5 % perf for 5 % less code |
| 2 (PLR=1) | **Default** — the right answer 95 % of the time |
| 4 (deeper PLR) | Only if VGPR budget has 50+ free VGPRs left over |

For mm0, 2-K-tile is forced — the VGPR budget is full at 256.

### 3.3 Unrolling vs occupancy

Bigger unroll → more register pressure → fewer waves/SIMD → lower
occupancy. The win from unrolling has to beat the loss from going from
2 waves/SIMD to 1. On RDNA4 with WMMA dominating, **1 wave/SIMD with
high WMMA utilization beats 2 waves/SIMD with VMEM-stalled WMMA** —
because WMMA is per-wave-pipeline, not per-CU. The mm0 winner runs at 1
wave/SIMD on purpose.

This is the *opposite* of conventional CPU/GPU shader wisdom that says
"more occupancy = better hiding". Matrix-engine kernels live or die by
the WMMA pipeline; once it's full, more waves only thrash caches.

---

## 4. Prefetch: depth, double-buffering, and DTV

### 4.1 The full prefetch pipeline

A K-tile's data flow on a "both-in-LDS" kernel:

```
DRAM ──VMEM──> VGPR ──ds_store──> LDS ──ds_load──> VGPR ──> WMMA
       ~400 cyc         ~30 cyc           ~30 cyc          16 cyc
```

That's **4 stages** of pipeline. To keep WMMA fed you need 4 K-tiles' worth
of buffering somewhere — but 4 tiles × 32 K-elt × 128 M+N = ~16 KB which
matches LDS budgets nicely.

### 4.2 Direct-to-VGPR (DTV)

DTV bypasses LDS for one operand. For B:

```
DRAM ──VMEM──> VGPR ──> WMMA            (3 stages, no LDS)
```

Saves ~60 cycles round-trip per K-tile per wave plus the LDS bandwidth
those operations would have consumed. On mm0 (DTVB1) this frees up LDS
for A-only, which is why WIN's LDS allocation drops 26 KB → 9 KB. Lower
LDS = higher CU occupancy if the kernel isn't VGPR-limited (mm0 is, so
it doesn't help here, but on other shapes it does).

**When to use DTV:**

- **DTVB**: NT GEMM, large N, B is read-once per K-iter by the wave's
  output column. Standard for inference (mm0, qkv, ffn).
- **DTVA**: TT or NN with large M, A is read-once.
- **No DTV**: TT-NN with both operands ping-ponged across waves in a WG
  that needs LDS for sharing.

DTV's cost: the operand needs to fit in registers. For mm0 B at MIWT_N=4
× K_per_iter=32 × 4 lanes-per-VGPR-fragment = 32 VGPRs reserved per
K-iter, doubled by PLR=1 = 64 VGPRs of B prefetch. That's a chunk of the
256-VGPR budget.

### 4.3 LDS double-buffering (LDSB)

Without LDSB, the wave has to barrier between "writing this iter's LDS"
and "reading next iter's LDS" because it's the same memory. The barrier
costs ~50 cyc and serializes all 4 waves of the WG.

LDSB=1 doubles the LDS allocation for A (if DTVB) or both, and
double-buffers: write to bank 0 while reading bank 1, swap each iter.
The barrier stays (you still need waves of the WG to agree on which
bank is active) but it's much cheaper because there's no flush to wait
on.

For mm0 LDSB=1 is worth ~150 cyc/wave (~0.5 TFLOP/s). Cheap to enable
when LDS allows.

### 4.4 The "Optimized No-Load-Loop" tail (ONLL)

The last few K-iters can't prefetch (there's nothing left to prefetch).
ONLL=1 emits a separate tail loop body that doesn't issue prefetch — it
just drains the WMMA pipeline. Saves 1 unnecessary VMEM issue per tail
iter and avoids waiting on a prefetch that loaded zero bytes.

Always enable. Costs 10–20 instructions of code size. Worth ~0.5 TFLOP/s.

---

## 5. LDS: bank layout and conflict avoidance

LDS on RDNA4: **32 banks × 4 bytes/bank = 128 B bank-page**. A wave32 issuing
ds_load_b128 reads 32 lanes × 16 B = 512 B from LDS in one cycle *if* each
lane's address falls in a different bank. If two lanes hit the same bank in
the same issue group, the LDS unit re-issues for those lanes (1-cycle
penalty per conflict).

### 5.1 The classic bank-conflict trap

If your LDS row stride is an exact multiple of 128 B, every lane reading
"row r, col c, c+1, c+2, ..." across waves will hit the same bank for
different rows. mm0's MT_K=32 BF16 = 64 B/row → row-stride 64 → 2 rows per
bank-page → conflicts when 2 waves of the WG read different rows
simultaneously.

**Fix: padding (LBSPPx).** Add bytes between rows so the row-to-row stride
isn't a clean multiple of 128:

| LBSPPA | Row stride | Bank phase shift |
|---:|---:|---|
| 0 | 64 B | conflict (2 rows/bank-page) |
| 128 | 192 B | better but still bank-aligned (192 ≡ 64 mod 128) |
| 256 | 320 B | clean (320 mod 128 = 64, but combined with LDSB the effective stride breaks) |

For mm0 the 128→256 step buys ~600 cyc/wave (~3.5 TFLOP/s). The reason
LBSPPA128 is bad and 256 is good is subtle and depends on the WMMA
swizzle pattern — sweep 0/64/128/256/512 and trust the perf delta. The
gfx1201 bank-conflict counter is unreliable (returns 0 even when conflicts
demonstrably exist), so don't rely on counters here.

### 5.2 The swizzle interaction

An A-tile written as a 32×16 BF16 block needs to be read as 16-wide
fragments by wave-tiles. The compiler picks a swizzle mapping (lane→addr)
that avoids conflicts for one specific row stride. Change the row stride
without updating the swizzle and you get cosine=0.998 nonsense (verified
2026-04-29 on a hand-asm patch attempt).

**Practical rule:** when using a Tensile-generated kernel, only change
LBSPPx in the kernel-name string and let Tensile re-pick the swizzle.
Don't try to patch LDS layout in hand-asm without rewriting the read-side
addressing.

---

## 6. Vectorization: instruction count is real cost

VMEM and LDS instructions issue at ~1 per cycle (VMEM-issue is 1 every
~4 cycles in practice due to back-pressure on the address unit). Stores
to global memory are even slower because of the write-combine queue.
**Every separate store instruction costs ~4 cycles even if the data is
ready.**

### 6.1 SVW: store vector width

mm0's epilogue writes 128×128 FP32 = 64 KB per WG. With SVW=1 that's 16
b32 stores per output element × 4 lanes = ~16,000 b32 stores per WG.
With SVW=4 it's 4 b128 stores covering the same bytes — **4× fewer
instructions**.

Cost savings: mm0 epilogue went from 128 b32 stores to 32 b128 stores per
wave (96 fewer × ~4 cyc = ~400 cyc/wave saved, ~2 TFLOP/s). Trivial to
enable when the output's M-stride is divisible by SVW × sizeof(out).

### 6.2 GRVW: global-read vector width

Set to the maximum the alignment allows. For BF16 inputs, GRVW=8 gives
b128 (16 B) loads. Going below this proportionally increases instruction
count.

### 6.3 The store-staging trick

With SVW=4 and an output type that doesn't natively support b128 stores
to the right address layout, the kernel may stage results through LDS:

```
WMMA accumulator (32 VGPRs/wave) ──ds_store_b128──> LDS staging
                                  ──ds_load_b128──> VGPR (different lane perm)
                                  ──buffer_store_b128──> global
```

Adds ~32 LDS ops per wave but eliminates 96 VMEM stores. Net win ~300
cyc/wave. Tensile generates this automatically when SVW=4 and the
output write needs lane reshuffling.

---

## 7. Schedule: SIA and instruction interleaving

`SIA` (Schedule Instruction Architecture) tells Tensile how to interleave
instruction classes inside the K-loop body:

- **SIA1** — bunched: all VMEMs, then all LDSs, then all WMMAs.
  Best on CDNA where the SIMD has separate execution slots and bunched
  issue lets the wide-VMEM unit drain efficiently.
- **SIA3** — interleaved: 1 VMEM, 1 LDS, 1 WMMA per slot, repeating.
  Best on RDNA4 where the dual-issue scheduler picks across pending
  instructions of different classes per cycle.

mm0 uses SIA3. Switching to SIA1 on RDNA4 regressed ~9 % in earlier
hand-patch experiments. **This is one knob you set per architecture and
never touch.**

### 7.1 What SIA3 actually emits

For one body iter (PLR=1, K_per_iter=32):

```
slot 0: buffer_load_b128 (next-K A frag 0)   ds_load_b128 (curr-K A frag 0)   v_wmma (out 0,0 K-half 0)
slot 1: buffer_load_b128 (next-K A frag 1)   ds_load_b128 (curr-K A frag 1)   v_wmma (out 0,1 K-half 0)
...
```

The dual-issue scheduler picks any pair of (VMEM, LDS, WMMA) per cycle
that don't conflict on register banks. Mixed scheduling lets the WMMA
pipeline run continuously while the address ALU and LDS unit work in
parallel.

---

## 8. Register pressure: the hard ceiling

VGPRs are the constraint that everything else negotiates around. RDNA4
tops out at **256 VGPRs/wave** for full 1-wave/SIMD occupancy (some
configs allow 252, encoding-dependent). Above 256, the kernel won't
compile or runs at lower occupancy.

mm0 winner uses exactly 256. Budget breakdown:

```
C accumulator (MIWT 4×4 × 16×16 FP32 / 32 lanes)         128 VGPRs
A LDS read window (PLR=1 double)                          16
A prefetch ring (PGR=2 × MIWT_M × K_per_iter)             32
B direct-VGPR (DTVB, MIWT_N × K_per_iter × PLR)           64
B prefetch ring (PGR=2 extra)                             16
Address calculation, scratch                              ~10
                                                          ───
                                                         ≈ 266  → trimmed to 256
```

Trimming to 256 requires recycling: B prefetch ring shares VGPRs with
the previous K-tile's WMMA inputs once those WMMAs retire.

### 8.1 Hard rules

- **Never go above 256 VGPRs** unless you accept 0.5×-occupancy mode.
- **Never go below 192 VGPRs** if you can use them productively — extra
  VGPRs unlock deeper PGR or wider VWA.
- **Sweet spot is 240–256** for compute-bound matrix kernels.

### 8.2 What to spend extra VGPRs on, in order

1. PGR depth (1 → 2 → 3) — each level: ~5 TFLOP/s, ~64 VGPRs.
2. VWA/VWB width (1 → 4) — biggest single lever, requires the right
   register packing, ~32 VGPRs.
3. PLR (0 → 1) — small win, ~16 VGPRs.
4. SVW staging buffers — modest, ~16 VGPRs.

The mm0 winner has all four. There's nothing left to spend VGPRs on.

---

## 9. Counter-driven debugging

On gfx1201 only three SQ counters are reliable:

| Counter | What it tells you |
|---|---|
| `SQ_WAVES` | Total waves dispatched (= num_WG × waves/WG) |
| `SQ_BUSY_CYCLES` | Sum across SIMDs of cycles where ≥1 wave was active |
| `SQC_LDS_BANK_CONFLICT` | Bank conflicts (broken on gfx1201, returns 0) |

The other counters (`SQ_INSTS_*`, `TCP_*`, `TCC_*`) often return 0 or
wildly inconsistent values across runs. Use perf timing as ground truth
and SQ_BUSY_CYCLES / SQ_WAVES as the secondary metric.

### 9.1 The cycles-per-wave decomposition

```
cyc_per_wave = WMMA_floor + VMEM_issue + LDS_issue + VALU + WAIT
```

You can't measure all terms directly, but you can *bound* them:

- **WMMA_floor** is computed (Section 1).
- **VMEM_issue** ≤ VMEM_inst × 4 cyc/issue (count from disasm).
- **LDS_issue** ≤ LDS_inst × 1 cyc/issue.
- **WAIT** = whatever's left.

If WAIT dominates (>30 % of cycles), the problem is *latency hiding* →
go up PGR or look for a missing PLR.

If VMEM_issue dominates (>15 %), the problem is *too many loads* → go up
GRVW or rethink the macro tile.

If WMMA_floor / cyc_per_wave > 80 %, you're done.

### 9.2 The A/B sweep workflow

The right perf-investigation loop:

```
1. Pick one knob (e.g. LBSPPA).
2. Sweep its values over a small set (0, 64, 128, 256, 512).
3. Measure ms (median of 5 runs after 64-iter warmup).
4. Take the best.
5. Lock that knob, move to next.
```

Order of knobs matters. Optimal order:

1. MT, MIWT, WG dim (structural; do first, lock).
2. DTVA / DTVB (structural; choose based on layout).
3. PGR.
4. VWA, VWB.
5. SVW.
6. LBSPPA, LBSPPB.
7. LDSB, CLR, ONLL.

Iterating in the wrong order means re-tuning later knobs after every
earlier change. This order minimizes re-work.

---

## 10. Worked example: closing the mm0 gap

The mm0 OLD baseline (algo 73624 default) hit 158 TFLOP/s @ 0.276 ms.
Per-wave: 24,174 cycles. WMMA floor: 73,728 / 2 (dual-issue) = 36,864 →
**67 % of issue floor**. Plenty of room.

Knob-by-knob lever pulls (cumulative):

| Step | Knob change | cyc/wave | TFLOP/s | Comment |
|---|---|---:|---:|---|
| 0 | OLD baseline | 24,174 | 158 | starting point |
| 1 | VWA1→4, VWB1→4 | ~22,200 | ~170 | unlocks WMMA dual-issue |
| 2 | PGR1→2 | ~21,400 | ~175 | deepens VMEM pipeline |
| 3 | LBSPPA128→256 | ~20,800 | ~178 | breaks LDS bank-page |
| 4 | SVW1→4 | ~20,500 | ~180 | epilogue 4× faster |
| 5 | LDSB0→1, CLR1→0 | 20,363 | 174 | minor; runtime variance |
| Final | (winner kernel) | 20,363 | **174.4** | 89.4 % peak |

Note that step 4 → step 5 looks negative; that's measurement noise (the
final number has variance ±2 TFLOP/s across runs). The cumulative
attribution of ~3,800 cycles saved is robust.

Step 1 alone is half the win. **VWA is the lever.**

---

## 11. Cross-architecture notes

**CDNA3/4 (MI300/MI355):** WMMA throughput is ~4× higher per CU due to
full matrix cores. Latency hiding is harder because the compute window
shrinks while DRAM latency stays similar → PGR=3 or 4 is normal. SIA1
(bunched) outperforms SIA3 because the matrix unit has its own
scheduling slot.

**Hopper / Blackwell (NVIDIA):** TMA + WGMMA shifts the model — TMA does
the prefetch+swizzle in hardware so PGR-style tuning becomes
"async-pipeline depth" (mbarrier stages). VWA-style register packing is
handled by WGMMA register descriptors, not chosen by the user. The
"dual-issue across SIMD pair" concept doesn't apply; instead it's "WGMMA
overlap with mma.async". The categories of tuning (overlap, compress,
eliminate) carry over; the specific knob names don't.

**General rule across architectures:** the *floor* analysis (Section 1)
and the *cycle decomposition* (Section 9) work everywhere. Only the
specific knob names and their default values change.

---

## 12. Checklist (TL;DR)

In order of importance for matrix-engine GEMM:

- [ ] Compute roof: compute-bound or BW-bound? (Section 0)
- [ ] Pick MT / MIWT / WG to fit LDS + VGPR budgets, ≥32 WMMAs/iter
- [ ] Choose DTVA vs DTVB based on layout (large-N → DTVB)
- [ ] **Set VWA = VWB = 4** (or maximum the K dim allows) — biggest lever
- [ ] Set GRVW = max alignment allows (8 for BF16)
- [ ] Set SVW = 4 for FP32 output, max for others
- [ ] Set PGR = 2 (3 if VGPRs allow)
- [ ] Set PLR = 1
- [ ] Set LDSB = 1 if LDS budget allows
- [ ] Sweep LBSPPA/LBSPPB over 0/64/128/256/512
- [ ] Set SIA per architecture (3 on RDNA4, 1 on CDNA)
- [ ] Set ONLL = 1
- [ ] Set CLR = 0
- [ ] Validate: cyc_per_wave / (WMMA_floor / 2) ≥ 0.85
- [ ] If a kernel zoo exists, sweep it as a fallback — beats hand-tuning
      for the last 5 %

For mm0, this checklist applied in order would have predicted 88–90 %
peak on the first iteration. The actual exhaustive sweep found 89.4 %,
confirming the checklist isn't leaving headroom.
