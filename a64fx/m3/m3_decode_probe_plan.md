# MiniMax-M3 bf16 decode-perf optimization probes

Small, node-economical probe jobs to push M3 **bf16 decode throughput past the current best**. Prepared,
not submitted. Sizing/context: `a64fx/m3/m3.md` (decode roofline), `a64fx/m3/m3_sim.py` (node fit).

## Where decode stands (from m3.md, all measured)

- **Single-stream ceiling ~3.3 tok/s is STRUCTURAL** — the `M3_DUMMY=1` probe removed the M=1 FMA and
  gained only ~5%. The bottleneck is **dispatch (hundreds of matvec parallel-regions/token) + comm
  (~23–42%) + serial non-matvec work**, NOT compute or HBM bandwidth (~1700 tok/s HW ceiling).
- **Multi-stream is the working lever:** `M3_MSTREAM` N=1→3.13, N=4→8.82, **N=8→12.33 tok/s aggregate**;
  comm grows 25%→42% as compute shrinks. **+ expert grouping → 14.18 tok/s @N=8 (~4.5× single-stream).**
- **Dead ends:** full-48/pinned-pool (hangs, no speedup — 1 CMG/12 threads is best), per-layer
  comm-overlap (net-negative; only cross-layer pipelining would help — deferred). `tok/s ~flat in N`.
- **Best today: ~14.2 tok/s aggregate @N=8, @96n (or any N — flat).** bf16 min node count = 48n.

## The three open levers → three probes

| # | probe | script | nodes | ~node-h | question |
|---|---|---|---|---|---|
| **P1** | mstream × comm-AR | `pjsub_m3_bf16_mstream_sweep_48n.sh` | 48 (synth) | ~48 | does throughput scale **past N=8** (16/32)? does **bf16-AR** cut the growing comm? |
| **P2** | int4-KV high-M enabler | `pjsub_m3_bf16_kv_highm_1n.sh` | 1 (synth) | ~0.4 | does **int4-KV** shrink the arena enough to deploy high-M at min nodes, w/o hurting tok/s? |
| **P3** | real-weight confirm | `pjsub_m3_bf16_confirm_48n.sh` | 48 (real) | ~72 | does the P1/P2 winner load real bf16, fit 48n, stay coherent, hit projected tok/s? |

Core = **P1 + P2 (~48 node-h)**; P3 is the optional expensive real-weight validation (~72 node-h).
All use `LLM_THREADS=12` (1 CMG — the established sweet spot), TP on, expert grouping on.

### P1 — mstream × AR (the primary throughput probe)
Synthetic full-60L bf16 @ 48n (bf16 deployment scale; tok/s flat in N so cheaper than 96n). Sweeps
`M3_MSTREAM ∈ {8,16,32}` × `TP_AR_BF16 ∈ {0,1}`. Reports agg tok/s + comm% + arena per config.
- **If tok/s keeps climbing 8→16→32** → batching hasn't plateaued; the serving throughput target rises
  (and P2 makes the high-M config fit). **If it plateaus/regresses** → comm-bound; bf16-AR (or higher N
  is pointless) is the lever.
- **If `TP_AR_BF16=1` lifts the high-M passes** → halving the [N,hidden] reduce payload cuts the 42%
  comm — a free win (watch: lockstep argmax must hold; coherent, not bit-identical to f32-AR).
- A pass that OOMs at M=32 (per-stream KV × M) just proves P2's point; the sweep continues.

### P2 — int4-KV high-M enabler (cheap, 1 node)
Small synthetic (12L/16E) on 1 node — KV is per-rank replicated, so memory scaling is
node-count-independent. Sweeps `M ∈ {8,32,64}` × KV `∈ {bf16, int4}`. The bf16→int4 **arena delta at
fixed M** is the KV saving (extrapolate ×60/12); tok/s should be ~unchanged. Answers whether P1's
high-M winner can run at the **24n/48n minimum** instead of needing more nodes for KV headroom.

### P3 — real-weight confirm (optional, expensive)
Set the knobs to the P1/P2 winner (`M3_MSTREAM`, `TP_AR_BF16`, `M3_INT4_KV`), stage the real 796 GB
bf16 model to `/local/m3` @ 48n, gen the coherence prompt. Confirms real-weight load + 48n fit
(**MemFree, not RSS**) + coherent "Paris" output + aggregate tok/s.

## Submit order & decisions

1. **P2 first** (near-free) — establishes the int4-KV memory envelope so P1's high-M passes are known to
   fit (avoids an OOM in P1 that would waste the 48n alloc; note: an OOM SIGKILL degrades PMIx and can
   cost the whole alloc).
2. **P1** — the throughput sweep. Read off the best (M, AR).
3. **P3** (optional) — confirm the winner on real bf16 weights before declaring a new best config.

Keep concurrency ≤2 jobs. Fold results back into `m3.md` (the decode-roofline bullets) and, if a new
best config emerges, update the recommended serving config there and in `pjsub_m3_mxfp8_gen_24n.sh` /
the gen launchers.

## Results

### BEST CONFIG: `M3_MSTREAM=48 TP_AR_BF16=1` — **17.75 tok/s synth / 15.25 tok/s REAL @48n**
Progression (synthetic): prior 14.2 (M=8) → P1 M=32 **16.68** → P1b M=48 **17.75** (**+25%**). Plateau at
M≈node count; M=64 regresses. bf16-AR is a free +2–3% (lockstep-preserving). **P3 real-weight confirm
(DONE):** coherent "Paris" ✓, M=48 FITS 48n (no OOM), but **real-weight AGG = 15.25 tok/s** (comm
27%→37% from real-router expert-load imbalance) → **synthetic overpredicts ~14% at high M**. Shippable
real-weight best is **15.25 tok/s** (+7% over the prior config).

### P3 — real-weight confirm (job 49441549, 48n, real bf16, bf16-AR) — DONE
- **Pass A (coherence, mstream=1):** "The capital of France is" → **" Paris. (Paris is the capital of
  France.)"** — bf16-AR preserves REAL-weight output (was argmax-identical on synth). NaN=0, arena 21.25
  GB (≈ synth 21.46), decode 3.33 tok/s single-stream.
- **Pass B (throughput, M=48):** **AGG 15.25 tok/s**, comm **37.4%**, arena 21.28 GB — **M=48 fits real
  weights @48n, no OOM** (memory question resolved for short ctx; int4-KV not needed at M=48/short-ctx).
  Real 15.25 < synth 17.75 → real-router imbalance raises comm; **m3_decode_sim is optimistic at high M.**

### P1b — higher-M plateau (job 49441449, 48n, full-60L synth bf16, bf16-AR, maxpos=512) — DONE
| M | AGG tok/s | comm% | out0 |
|---|---|---|---|
| 32 (control) | 16.70 | 26.3% | 694 |
| 48 | **17.75** | 26.7% | 2429 |
| 64 | 17.35 | 25.3% | 3090 |

- M=32@512 (16.70) ≈ P1's M=32@1024 (16.68) → **maxpos-independent**, P1b comparable to P1.
- **Peak at M=48 (17.75), M=64 regresses (−2.3%)** — batching tops out ~M=node-count; extra per-stream
  overhead at M=64 beats comm amortization (matches P2's ~17.6 single-node compute ceiling). NaN=0.
- M=48@maxpos=512 fit (arena 21.46 + KV ~6 GB); at maxpos=1024 M=48 is tight (~27.5 GB) -> needs
  int4-KV or more nodes for long ctx.

### P1 — mstream × AR (job 49441386, 48n, full-60L synth bf16, max_pos=1024, 32 steps) — DONE
| M | AR | AGG tok/s | comm% | out0 |
|---|---|---|---|---|
| 8  | f32  | 12.89 | 31.6% | 2847 |
| 16 | f32  | 13.96 | 25.7% | 3967 |
| 32 | f32  | **16.15** | 28.7% | 694 |
| 8  | bf16 | 13.19 | 29.2% | 2847 |
| 16 | bf16 | 14.29 | 24.5% | 3967 |
| 32 | bf16 | **16.68** | 26.0% | 694 |

- **Batching scales past N=8** at real node counts: +25% M=8→32 (f32), still climbing at M=32 (the
  1-node P2 plateaued ~17.6 only because it had no comm; at 48n the per-layer reduce amortizes with M).
- **bf16-AR (`TP_AR_BF16=1`) is a free +2–3%** at every M, **out0 bit-identical** to f32-AR (lockstep
  holds, no argmax flip on synth).
- **NEW BEST: M=32 + bf16-AR = 16.68 tok/s aggregate @48n** vs prior ~14.2 @N=8 → **+17%**. arena 21.53
  GB (weight only; per-stream KV not in the metric — M=32 fit with no OOM; M>32 headroom uncharacterized).
- **Next:** a P1b (M=48/64 @48n, bf16-AR) would find the true plateau (compute ceiling ~17.6 from P2),
  but needs the int4-KV memory question (P2 caveat) settled first if M=64 KV overflows.

### P2 — int4-KV high-M (job 49441305, 1n, 12L/16E synth, max_pos=1024) — DONE
(First submit 49441301 failed: the runner `read_topo()` exit(1)s without `tofu_topo.txt`, and the
1-node template skipped the topo helper; also perf goes to the rank files, not stdout. Fixed in
`e9354b13`.)

| M | KV | AGG tok/s | per-stream | out0 | arena_used |
|---|---|---|---|---|---|
| 8 | bf16 / int4 | 15.76 / 15.82 | 1.97 | 138162 (both) | 26.27 / 26.25 GB |
| 32 | bf16 / int4 | 17.11 / 17.08 | 0.53 | 177681 (both) | 26.27 / 26.25 GB |
| 64 | bf16 / int4 | 17.59 / 17.60 | 0.27 | 28337 (both) | 26.27 / 26.25 GB |

- **int4-KV is free** (tok/s identical, NaN=0). **Batching plateaus on compute** (no comm at 1n):
  +8.6% M=8→32, +2.8% M=32→64 → sweep M≤32 (as P1 is written).
- **CAVEATS:** (1) `arena_used` is the WEIGHT arena only — per-stream KV is allocated separately and
  not reported, so the int4-KV **memory saving was NOT measured** (rely on P1 MemFree @48n).
  (2) int4-KV out0 was **bit-identical** to bf16 → likely **not engaging on the batched/mstream
  path** (single-stream KV-quality probe does see int4 diffs on real weights). **Verify int4-KV
  actually changes the batch-decode output before treating it as the high-M memory enabler** — else
  P1's high-M passes must fit with bf16-KV (more nodes / shorter max_pos) or int4-KV needs wiring.
