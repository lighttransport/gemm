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
