# Research: fused attention kernels and the achievable peak fraction on RDNA4 (gfx1201)

**Question.** Can a "fused attention" kernel push qimg's self-attention from ~10% of the
195 TF/s BF16 WMMA peak toward a high fraction (the 70% target)?

**Bottom line.** No — not on this silicon, and not as a coding effort. The published
result that reaches a high peak fraction (FlashAttention-3, ~75% of Hopper peak) is built on
hardware mechanisms RDNA4 does not have. Across this repo's *entire* body of RDNA4 attention
work the empirical ceiling for online-softmax attention is ~7–13% of WMMA peak, and the most
developed line declared the fusion design space exhausted. The realistic ceiling for a hard,
genuinely-untried RDNA4-native fusion (compute-stage software pipelining) is — optimistically —
the high-teens-to-~25%-of-peak band: a meaningful 2–3× over today, but a hardware-generation gap
short of 70%. This note records the research so the target can be calibrated.

## 1. Why online-softmax attention is structurally far below GEMM peak

Pure GEMM hits 89% of peak here (the mm0 effort). Attention layers an *irreducible* non-WMMA
tax on the same matmuls: an online softmax (cross-lane max/sum shuffle chains + `expf` on the
transcendental unit) and a layout transpose — the QK-WMMA produces P in the accumulator lane
layout but the PV-WMMA needs P in the A-fragment layout, forcing a P→LDS→back round-trip with a
wait. The matrix pipe idles through all of it. And within a query tile the chain is strictly
serial: QK → softmax → P-transpose → PV, with a tile-to-tile dependency through the running
softmax statistics (`m_i`, `l_i`). So the WMMA pipe alternates *burst → stall → burst*.

The repo's cycle-budget model (`rdna4/vlm/optimized-gemm-guide.md`) makes this concrete: peak
efficiency = WMMA-pipeline-cycles / total-cycles. In attention the softmax/transpose cycles are
a large, non-overlappable share of the total, so the WMMA-busy fraction is low by construction.

## 2. The published frontier (FlashAttention-3) and why it does not transfer

FlashAttention-3 (Hopper, 2024) lifted attention from FA2's ~35% to ~75% of H100 tensor-core
peak. The mechanisms (general knowledge, not in-repo):

- **Asynchronous warpgroup MMA (`wgmma`)** — the tensor core runs a matmul *in the background*
  while the issuing warpgroup proceeds to do softmax; a later wait collects the result. This is
  what lets softmax overlap matmul — the single most important lever.
- **Warp specialization + ping-pong scheduling** — producer warps stream operands (via TMA), and
  two consumer warpgroups alternate so one does `wgmma` while the other does softmax.
- **TMA** — hardware async bulk copies that issue without occupying the warp.

**RDNA4 gfx1201 has none of these.** Its WMMA (`v_wmma_f32_16x16x16_*`) is a *synchronous*
per-wave instruction issued on the SIMD's own pipe (16-cycle latency, dual-issue when register
windows don't bank-collide — `optimized-gemm-guide.md`). There is no async-MMA: the wave that
issues a WMMA *is* busy doing it; it cannot also be computing softmax. There is no TMA. The exact
asynchrony FA3 relies on to hide softmax behind matmul is absent. This is an architectural-
generation gap (Hopper/Blackwell-class tensor cores + TMA), not something a kernel rewrite closes.

## 3. The repo's own empirical ceiling (triangulation)

Every RDNA4 attention effort in this tree converges on the same low fraction — strong evidence
the ceiling is structural, not a missed trick:

- **VLM vision encoder** (`rdna4/vlm/brief.md`): the optimized BF16-WMMA flash attention reached
  **~13 TF/s ≈ 7% of the 195 TF/s peak**; the projected ceiling with *all* listed micro-opts
  landed (BQ64-4wave, b128 LDS vectorization, V pre-transpose, bank-pad, fused cast) was ~3 ms/call
  from 4.6 — still on the order of ~10% of peak. Its single biggest win was removing an integer
  divide from the hot loop (−30%), i.e. instruction-stream pressure, not WMMA throughput.
- **trellis2** — the most-developed self-attention family: a documented "floor at 346 ms" with the
  conclusion that the quick-fusion design space is exhausted and further wins need an FA softmax
  rewrite. Its **deferred-sum softmax** (the one in-repo idea that shortens the softmax dependency
  chain) bought only **~2%**. Its **double-buffered wide tile** (`bc32_db`) was a *net occupancy
  loss* — the doubled LDS cut CTAs/WGP from 3 to 2, costing more than the prefetch recovered, so it
  ships default-OFF.
- **qimg** itself: BKV32 regressed −4.5%; bf16 K/V pre-pack gave +1.6% (L2 already absorbed the
  re-reads — not bandwidth-bound). The recent persistent-Q + double-buffer kernel (v2,
  `flash_attn_sa_wmma_pq_f32`, gated `QIMG_ATTN_V2=1`) is the first attention change to move e2e the
  right way: −9.6% at 1024², lossless — taking attention from ~8% to ~10% of peak by hiding the K/V
  *load* latency. It explicitly does not touch the serial softmax/transpose chain.

No kernel in the repo overlaps the softmax/transpose of one tile with the WMMA of another; all
overlap only the *K/V load* with compute. And no vendor fused-attention library (Composable Kernel,
hipBLASLt-FA) is wired in — hipBLASLt is used for GEMM only.

## 4. Technique-transfer assessment (to gfx1201 self-attention, head_dim=128)

| Technique | Transfers to RDNA4? | Mechanism / why | Plausible payoff |
|---|---|---|---|
| Online softmax (already done) | yes (baseline) | one-pass, the existing kernel | — |
| K/V load double-buffering (v2, done) | yes | hides global-load latency behind compute | realized ~−10% e2e |
| Persistent-Q in registers (v2, done) | yes | frees 16 KB smQ, removes inner-loop LDS read | folded into the above |
| Deferred-sum softmax | yes (trivial port) | moves the per-tile sum reduction to the end; ~half the shuffle work | ~2% (trellis2-measured) |
| Wide-Bc amortization | yes — *newly* viable | v2's freed LDS lets Bc grow so the fixed softmax/transpose cost amortizes over more keys; the old BKV32 failed only because smQ pinned LDS | single-digit to low-double-digit %, occupancy-bounded |
| WMMA dual-issue VWA=4 register packing | yes (hard) | the top GEMM lever (+12 TF/s); makes the QK/PV WMMA *bursts* run at ~8-cyc effective latency. Helps the matmul portion only, not the softmax tax | bounded by the WMMA-burst share (~40% of attention) |
| **Compute-stage software pipelining** | yes (research-grade) | issue tile (t+1)'s QK WMMAs and fill their *latency shadows* with tile t's softmax VALU/transcendental work — overlapping the chain within a single wave. The genuinely-untried RDNA4-native frontier; fights the LLVM scheduler, no async hardware to lean on | the only intra-design ceiling-raiser; partial overlap → still well short of FA3 |
| Warp specialization (FA3-style producer/consumer) | partial / weak | dedicate some of the 4 waves to WMMA and others to softmax, handing off P through LDS — a manual emulation of FA3. RDNA4's *synchronous* WMMA means the WMMA waves still stall at hand-offs; far weaker payoff than Hopper's async version, very intricate | speculative; high risk |
| Async warpgroup-MMA / TMA pipelining (the FA3 ceiling-raiser) | **no** | hardware features RDNA4 lacks | n/a |

## 5. The realistic frontier and recommendation

The only fusion ideas that could lift the *ceiling* (rather than shave constants) are compute-stage
software pipelining and warp specialization — both attempts to overlap softmax with WMMA, which is
precisely what the absent async-MMA hardware would have done for free. Single-wave compute-stage
pipelining is the more tractable and more RDNA4-appropriate of the two (no cross-wave LDS handoff),
and is the genuinely-novel research direction; it leans on the dual-issue WMMA latency shadows that
the GEMM playbook exploits, applied across attention tiles. Even so, the overlap is partial and the
softmax/transpose tax does not vanish — a compounded optimistic outcome (wide-Bc + deferred-sum +
WMMA dual-issue packing + partial pipelining) plausibly lands in the high-teens-to-~25%-of-peak band.
That is a worthwhile 2–3× over today's ~10%, roughly halving attention's share of the denoise step —
but it is not 70%.

70%-of-WMMA-peak fused attention is a hardware-generation capability (async tensor-core MMA + TMA),
not a kernel that can be written for gfx1201 with online softmax. The conservative reading of this
repo's exhaustive prior attempts agrees.

**Recommendation:** treat the v2 kernel (lossless −9.6%, ready) as the shippable win. If pushing
attention further is wanted, the realistic, low-risk increments are deferred-sum softmax and a
wide-Bc variant (both gated A/B against v2, both proven-shape in trellis2) toward the ~15–25% band.
Compute-stage software pipelining is the high-risk research bet with the only chance at a larger
step; it should be scoped as an explicit experiment with a sober expectation, and is at the edge of
what hand-tuned kernel work can reliably deliver. Note also the broader picture: at 1024² qimg is
block-streaming-bound and attention is one term — even halving attention is a single-digit-percent
wall change. The bigger e2e levers remain structural (model size vs VRAM), not the attention kernel.

## Empirical addendum (2026-05-23): the software-pipelined "FA3-idea" attempt is a measured no-op

The most tractable RDNA4-native embodiment of FA3's overlap idea was built and benchmarked:
`flash_attn_sa_wmma_sp_f32` (gated `QIMG_ATTN_V3=1`) — a single-wave software pipeline that
computes the *next* tile's QK scores (register-carried `S_next`) in the same loop body as the
current tile's softmax + P·V, with triple-buffered K/V (three live tiles: PV'd / QK-lookahead'd /
loading). Intent: let the compiler fill the QK/PV WMMA latency shadows with the softmax VALU work.

Result: numerically correct (256² 52.27 dB / cos 0.999975 — within FP-reorder noise of the 51.70 dB
baseline) but **1024² 7.10 s/step — identical to v2 (7.10), i.e. zero gain over v2's load
double-buffering** (v1 = 7.86). Source-level instruction reordering produced no overlap, for two
structural reasons confirmed by the experiment: (1) RDNA4's single per-wave matrix pipe runs the
lookahead QK and the current P·V serially whatever the issue order — they contend for the one
functional unit, so issuing QK early creates no overlap; (2) the `s_waitcnt(0)` of the P-transpose
round-trip pins the softmax in place. This is exactly the prediction above — the FA3 overlap is a
hardware capability (asynchronous tensor-core MMA), not a schedulable property of a synchronous-WMMA
kernel. The only paths that *could* overlap softmax with matmul on RDNA4 are cross-wave warp
specialization (heavier, the matmul waves still stall at LDS hand-offs) — and the repo evidence
makes a meaningful win there unlikely. v3 was kept gated/default-OFF as a recorded negative result;
v2 (the lossless −9.6%) remains the shippable attention win.

## References
- `rdna4/vlm/optimized-gemm-guide.md`, `peak_efficiency_playbook.md`, `mm0_lever_attribution.md`,
  `brief.md` — the RDNA4 cycle-budget model, dual-issue WMMA / VWA packing, and the VLM flash-attn
  narrative (~7–13% of peak; integer-divide and load-prefetch wins).
- `rdna4/trellis2/hip_trellis2_kernels.h` — the bc32 / bc32_db / b16_db / deferred-sum (v2) attention
  family and its occupancy/perf comments; trellis2 project memory ("floor at 346 ms; fusion exhausted").
- `rdna4/qimg/hip_qimg_kernels.h` — the v1 and v2 (`flash_attn_sa_wmma_pq_f32`) qimg kernels.
- FlashAttention-3 (Shah, Dao, et al., 2024), Hopper warp-specialized async-MMA attention — general
  knowledge, the published ~75%-of-peak frontier; cited for the hardware-feature contrast.
