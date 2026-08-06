# K3 roofline vs measured

Measured on the live 12-node allocation, 2026-08-06 (job 50008795).  Companion
to `logs/quant-bench-live12-50000128/SUMMARY.md`, which records the kernel
optimization passes this analysis was written to check.

Artifacts: `a64fx/k3/logs/roofline-50008795/` — `decode_bw.txt` (node bandwidth),
`tp_ar_ack_result.txt` (collective latency), `layer3_profile.txt` (per-phase).

## Measured hardware constants

`a64fx/llm/ds4f_decode_bw_bench.c` (`make CC=fcc ds4f_decode_bw`) reproduces the
documented curve almost exactly, so **R = 726 GB/s/node** stands:

| threads | 1 | 12 | 24 | 48 |
|---|---|---|---|---|
| documented (`ds4f.md:596`) | 42 | 466 | 714 | 719 |
| measured now | 42.5 | 466.1 | 709 | **726** |

An earlier probe in this session read 193 GB/s and was **wrong** — four
accumulators with no prefetch is latency-bound, and `aligned_alloc` recycles
faulted pages onto CMG0. Use the existing bench, not a fresh one.

## The number that reframes the quantization work

Every kernel on one axis (quant rates from this session's layer bench at 47
threads on real weights; bf16/mxfp4 from `decode_bw.txt` at 48 threads):

| kernel | Gmac/s | GB/s | % of R |
|---|---|---|---|
| bf16 matvec | **326** | 652 | **89.8%** |
| mxfp4 (the original's experts) | **212** | 106 | 14.6% |
| IQ1_S `ffn_down_exps` (ours) | 82 | 17.7 | 2.4% |
| IQ2_XXS `ffn_gate_exps` (ours) | 79 | 17.2 | 2.4% |
| Q8_0 `output.weight` (ours) | 74 | 78 | 10.8% |
| IQ2_XS `ffn_down_exps` (ours) | 58 | 17.1 | 2.4% |

**The bf16 kernel is at 90% of the memory roofline; every quantized kernel is at
2-11%.** The quantized path is compute-bound by a wide margin, and the original
MXFP4 expert kernel is **2.6x faster per mac than our IQ1_S** after six
optimization passes. IQ formats are codebook/grid designs that assume cheap L1
gathers; A64FX's gather is slow, and that is structural, not a tuning gap.

Consequence for decode at equal node count (96n, per-token per-node macs:
591.7e6 dense + 506.5e6 expert):

| package | dense ms | expert ms | compute ms | +comm | **peak tok/s** |
|---|---|---|---|---|---|
| original (bf16 + MXFP4) | 1.82 | 2.39 | **4.20** | ~20 | **42-48** |
| IQ1 (Q8_0 + IQ1_S) | 8.04 | 6.18 | 14.2 | ~20 | 29-32 |
| Q2 (Q8_0 + IQ2_XS) | 8.04 | 8.72 | 16.8 | ~20 | 27-30 |

**The original is the fastest format for decode, not the slowest.** IQ1's value
is fitting in 24-32 nodes instead of 56-96 — a capacity argument, not a speed
one. It only becomes a speed win if its kernels reach MXFP4's 212 Gmac/s.

## Collectives are latency-bound

`tp_ar_ack_test`, 12 nodes, real MoE payload (HIDDEN+LATENT = 10752 f32 = 43 KB):

| variant | us/reduce |
|---|---|
| flat | 106.9 |
| 2D A=3xB=4 | **88.6** |
| 2D A=2xB=6 | 93.9 |
| bf16 payload (half the bytes) | 104.9 |
| a2a | 106.8 |

**Halving the payload changes latency by 1.9%.** 43 KB at the documented
6.36 GB/s is 6.8 us of wire time against 107 us measured, so ~94% is protocol
and synchronization. Smaller payloads will not help; fewer, larger, or
better-overlapped collectives will. Hierarchical 2D is worth 1.21x for free.

This also settles an inconsistency in `network_estimate.txt`: its "collective=20
us/step" calibration is wrong and its 24.7 ms comm column (186 x 133 us) is
right. 186 collectives x 88.6 us = **16.5 ms/token of pure comm**.

## The first per-phase profile

No `K3FULL_PROFILE` had ever been emitted. `K3_PROFILE=1 ./run_k3_full_12n.sh
--mode layer12 --layer-index 3 --expert-tp` produces one. Layer 3 (MLA), 12
nodes, expert-TP, rank-max means over 32 samples:

| phase | ms | share |
|---|---|---|
| **layer total** | **4.918** | vs the 1.08 ms budget at `k3_full_runner.c:415` |
| attention | 2.137 | largest single phase |
| moe (total) | 3.264 | |
| — moe_shared | 1.136 | **1.8x the routed experts** |
| — moe_finish | 0.938 | |
| — moe_collective | 0.669 | |
| — **moe_expert** | **0.632** | everything optimized this session |
| — moe_dispatch | 0.398 | |

Sub-phases do not sum to their parent because each is a rank-max, and the
sample maxima are 4-6x the means (attention 12.5 vs 2.14, moe 20.0 vs 3.26),
i.e. large straggler tails.

4.918 ms x 93 layers = 457 ms/token = **2.19 tok/s**, which brackets the 96-node
measured 1.699. **The full-model slowness is per-layer and reproducible on 12
nodes** — it does not need the queue to iterate on.

## Where the gap actually is

| | practical peak | measured |
|---|---|---|
| decode, original, 96n | 42-48 tok/s | 1.699 |

~20-25x, and the profile says it is **not** the expert kernels: `moe_expert` is
0.632 of 4.918 ms, so even making it free buys 15%. Ranked by headroom:

1. **attention, 2.137 ms (43%)** — untouched all session.
2. **moe_shared, 1.136 ms** — 2 always-active shared experts costing more than
   16 routed ones. Suspicious enough to be a bug, not just slowness.
3. **moe_finish 0.938 + moe_collective 0.669** — comm and its epilogue; the 2D
   variant above is a free 1.21x on part of this.
4. moe_expert 0.632 — the part six passes have already been spent on.

Also free: the straggler tails (sample max 4-6x mean) and, for prefill, the
measured run's `chunk 64` on a 256-token prompt, the worst case for GEMM
efficiency.

## Follow-up: the per-layer prefetch thread is a net loss

`full_prefetch_start` (`k3_full_runner.c:354`) does a `pthread_create` per MoE
layer whose worker issues one `__builtin_prefetch` per 64 bytes over 16 MiB —
262144 of them on a single core — and it is joined inside the `moe_shared`
timed region, which also mis-attributed the wait to that phase.

Layer 3, 12 nodes, 512 samples per run:

| prefetch | rep1 | rep2 |
|---|---|---|
| off | **4.146** | **4.138** |
| 16 MiB | 4.894 | 5.413 |

1.18-1.31x, and note the off case is reproducible where the on case is not —
the rogue thread adds jitter as well as time. Default changed to 0 in
`k3_full_runner.c` and `run_k3_full_12n.sh`; the production 96-node script
already passed 0 explicitly.

**A caution on measuring this.** A single on/off pair suggested 2x. Three
repeats at the default 32 samples gave overlapping distributions (off
3.87/3.63/5.47, on 4.16/5.42/4.46) — i.e. the 2x was noise. Only at 512 samples
(`--prefill-tokens 512`) does the effect resolve. Use 512 samples for anything
under ~30% on this node.

## Clean baseline and what to attack next

Layer 3 (MLA), 12 nodes, expert-TP, prefetch off, 512 samples — **4.303 ms**:

| phase | ms | share |
|---|---|---|
| attention | **1.855** | 43% |
| moe_collective | 0.933 | 22% |
| moe_shared | 0.711 | 17% |
| moe_finish | 0.583 | 14% |
| moe_dispatch | 0.499 | 12% |
| moe_expert | 0.479 | 11% |

Sub-phases are rank-maxima and do not sum to the parent; every phase's sample
peak is 12-20 ms against means under 2 ms, so straggler tails are large.

**`moe_shared` is overhead, not compute.** With TP over 12 nodes it is 512 local
rows: gate 512x7168, up 512x7168, down 7168x512 = 11.0e6 macs on 22 MB of bf16
weights. At the measured bf16 rate (326 Gmac/s) that is 0.034 ms; it takes
0.711 ms, i.e. **15.5 Gmac/s, 21x off the kernel's own rate**. The block is two
OpenMP regions (`full_bf16_many` for gate+up, then `full_bf16_matvec` for down,
separated by a SiTU dependency) plus the activation. At this size the region
entry dominates — the same `__kmp_fork_barrier` cost that was 12-14% of the
quant bench. The fix is fewer, larger parallel regions per layer, not a faster
kernel.

The same argument likely applies to `attention` and to `moe_dispatch` /
`dispatch_proj`, which are also small projections in their own regions. That is
the next piece of work, and it is a restructuring of where parallel regions
begin and end across a layer, not kernel tuning.

## The parallel-region hypothesis was wrong; the schedule was the bug

Premise going in: the small phases are dominated by OpenMP region-entry cost, so
a layer should hold one persistent team. **Measured, that premise is false.** An
empty parallel region at 48 threads costs 10.6 us back-to-back and 18.5 us when
threads idle between regions (8.9 / 12.2 with `OMP_WAIT_POLICY=active`). The
shared-expert block uses two regions — ~37 us against its 0.711 ms. Region entry
is a few percent, not the problem, and the restructuring was not done.

Timing the three shared-expert calls standalone at their staged shapes
(`local_shared = 6144/12 = 512`, so gate/up are 512x7168 and down is 7168x512)
found the real defect — and it was self-inflicted:

| schedule, 48 threads | gate+up | down | total |
|---|---|---|---|
| **static** | 0.025 ms (577 GB/s) | 0.025 (296 GB/s) | **0.052** |
| guided | 0.074 | 0.311 (23.6 GB/s) | 0.390 |
| dynamic | 0.075 | 0.105 | 0.184 |

The commit that made `full_bf16_many` use `schedule(guided)` was measured on a
*heterogeneous* batch — a whole layer's projections, columns spanning
512..12288 — where guided is 1.36x. For a *homogeneous* batch of equal-cost
8 KB tasks, guided's chunk arithmetic and shared counter dominate and it is
**7.5x slower** than static. The shared expert is exactly that case.

Fix: choose the schedule from the batch's shape — static when every `cols[i]`
matches, guided otherwise, selected with `omp_set_schedule` + `schedule(runtime)`
so there is still one loop body.

Layer 3, 12 nodes, 512 samples, two reps:

| phase | guided everywhere | per-batch choice |
|---|---|---|
| **layer** | 4.303 | **2.86** (1.51x) |
| attention | 1.855 | 1.34 |
| moe_collective | 0.933 | 0.56 |
| moe_shared | 0.711 | 0.35 |
| moe_finish | 0.583 | 0.30 |
| moe_expert | 0.479 | 0.37 |

The layer12 output hash is byte-identical (`02e4dc4b1746567a`), as it must be —
scheduling cannot change results here.

Extrapolated, 2.86 ms x 93 = 266 ms/token = 3.76 tok/s, against 2.19 before and
the 96-node measured 1.699. Attention remains the largest phase at 47%.

## Attention: KDA is the target, and the cost is OpenMP barriers

Layer 3 is MLA, but only 24 of 93 layers are. Profiling both (12 nodes, 512
samples, after the schedule fix):

| layer type | count | layer ms | attention ms | attention per token |
|---|---|---|---|---|
| KDA | 69 | 2.319 | 0.883 | **60.9 ms** |
| MLA | 24 | 2.799 | 1.331 | 31.9 ms |

So **KDA attention is twice the target MLA is**, and a token is
69x2.319 + 24x2.799 = 227 ms => 4.40 tok/s extrapolated.

KDA's weight traffic is small: two `full_bf16_many` batches (5 projections of
4224 rows x 7168, then f_b 1024x128 and b_proj 8x7168) plus o_proj 7168x1024 =
~75 MB, about 130 us at the measured static rate. The phase costs 883 us.

`perf record` on rank 0 of a 12-rank KDA run says where the rest goes:

| symbol | share |
|---|---|
| `__kmp_fork_barrier` | **37.0%** |
| `full_bf16_run_task` | 19.9% |
| kernel (unresolved) | 8.9 + 5.6% |
| `__sched_yield` | 5.2% |
| `k3_mxfp4_group_batch` | 4.4% |
| `__kmp_hyper_barrier_release` | 4.0% |

**~46% of runtime is OpenMP team synchronization**, and `__sched_yield` says
workers are sleeping between regions and paying a wakeup each time. Note this
does *not* contradict the earlier "region entry is 10.6 us" measurement — that
was a warm team looping back-to-back. In the runner the regions are separated by
serial work, so threads sleep and the real cost is the wakeup.

What the available knobs buy (KDA layer, 2 reps each):

| threads | default | `OMP_WAIT_POLICY=active` + `KMP_BLOCKTIME=infinite` |
|---|---|---|
| 24 | 2.165 / 2.127 | **2.072 / 2.072** |
| 48 | 2.116 / 2.507 | 2.052 / 2.346 |

3-5%, and it removes most of the run-to-run variance; 48 threads is erratic
either way. Added to the three launcher scripts.

**The remaining ~35% needs fewer parallel regions per layer**, i.e. one
persistent team per layer with the inner `parallel for` becoming `omp for`.
That is now justified by data rather than by the assumption it was dismissed on,
but it touches `k3_full_runner.c`, `k3_kernels.h`, `k3_moe.h` and
`common/ggml_dequant.h` together and should be its own piece of work.

## The persistent-team restructuring: built, measured, reverted

Built it for the KDA path (the 69-layer majority): task lists for all three
projection batches constructed before one `#pragma omp parallel`, with `omp for`
per batch, `omp single` for the scalar stretches (conv step, L2 normalise,
log-decay, the expf loop, gated RMS norm) and
`k3_kda_step_decay_team_sve` — which already exists as an `omp for` designed to
reuse an outer team — in the middle. Correct: the layer-2 output hash stayed
`02e4dc4b1746567a`.

Three reps each, layer 2, 12 nodes, 512 samples:

| | layer ms | attention ms |
|---|---|---|
| persistent team | 2.195 / 1.904 / 2.041 | 0.862 / 0.703 / 0.778 |
| pre-team | 1.851 / 2.353 / 2.013 | 0.583 / 0.881 / 0.668 |

Means 2.047 vs 2.072 — 1.2% apart inside a ±12% spread. **No effect.** Reverted.

Why the perf data pointed the wrong way: `__kmp_fork_barrier` at 37% is threads
*waiting*, and a persistent team does not remove the waiting. Every `omp for`
and `omp single` carries an implicit barrier, so ~4 region entries became ~6
in-region barriers across the same 48 threads. The fork/join went away; the
synchronization did not, and the synchronization was the cost.

**What this implies for the next attempt.** The lever is the *number of
synchronization points* per layer and the *team size*, not the team's lifetime.
Two things already point that way and are cheap to test:

- 24 threads with `OMP_WAIT_POLICY=active` measured 2.072/2.072 — the same mean
  as 48 threads but reproducible, where 48 swings 2.05-2.35. Barriers get
  cheaper with fewer threads, and KDA has only 8 local heads to spread anyway.
- The KDA layer has ~6 dependency-ordered stages. Merging stages that do not
  actually depend on each other (the decay batch needs only the qkv batch, not
  the conv step) would remove barriers outright rather than relocating them.

## 48 threads is a cliff; 47 is optimal

`run_k3_full_12n.sh` defaulted to `THREADS=48`, i.e. all 48 compute cores with
none left for the runtime. Every layer measurement earlier in this document was
taken there. Two reps each, after the schedule fix and the active wait policy:

| threads | 24 | 32 | 40 | 44 | **47** | 48 |
|---|---|---|---|---|---|---|
| KDA layer ms | 2.070 | 1.768 | 1.606 | 1.542 | **1.508** | 2.180 |
| MLA layer ms | — | 2.083 | 1.996 | — | **1.965** | 2.503 |

Monotonic improvement from 24 to 47 and then a 45% jump at 48 — a cliff, not a
curve, and it reproduces the note already in this repo that 48 pinned OMP
threads on 48 compute cores costs ~40%. The earlier reading that "24 and 32 beat
48" was really "anything beats 48"; the true optimum is 47.

Default changed to 47. The production 96-node scripts already used 47.

Corrected extrapolation: 69 x 1.508 + 24 x 1.965 = **151 ms/token = 6.6 tok/s**,
against 4.40 before this and 2.19 at the start of the day.

## Not attempted: merging the independent KDA stages

The decay batch depends only on the qkv batch, not on the conv step that
currently sits between them, so that barrier is removable by reordering (and
`b_proj` reads only `x`, so it could move into the first batch outright). Worth
doing, not attempted here. Given that the persistent-team restructuring above
turned out to be worth 1.2% and the thread count was worth 45%, the ordering
question deserves the same "measure a small change first" treatment rather than
being assumed to matter.

## KDA stage merge: landed, and the reason is the schedule again

Moved `b_proj` from the decay batch into the qkv batch. It reads only `x`, so
the dependency allowed it, but the payoff is not the ordering — it is that both
batches become homogeneous in columns. The old decay batch mixed f_b's 128
columns with b_proj's 7168, so it took `guided`, and f_b is 128 eight-row tasks
of 128 columns: exactly the tiny-homogeneous-task case measured at 7.5x slower
than static earlier in this document.

Three reps, layer 2, 47 threads:

| | layer ms | attention ms |
|---|---|---|
| merged | 1.500 / 1.498 / 1.492 | 0.457 / 0.455 / 0.450 |
| baseline | 1.757 / 1.972 / 1.508 | 0.610 / 0.677 / 0.467 |

1.17x on the layer and 1.29x on attention, with the merged worst case beating
the baseline best case. Output hash unchanged (`02e4dc4b1746567a`).

## Where <1 ms/layer stands

Phase profile at 47 threads, before the merge:

| phase | KDA (l2) | MLA (l3) |
|---|---|---|
| **layer** | **1.524** | **1.958** |
| attention | 0.481 | 0.909 |
| moe | 0.935 | 0.939 |
| — moe_dispatch | 0.244 | 0.247 |
| — moe_expert | 0.212 | 0.212 |
| — moe_shared | 0.190 | 0.192 |
| — dispatch_proj | 0.175 | 0.176 |
| — moe_finish | 0.171 | 0.169 |
| — moe_collective | 0.126 | 0.124 |
| reduce | 0.119 | 0.119 |

After the merge KDA is ~1.49. **The target is not met and is not one change
away.** The layer is now broadly balanced: MoE is 61% of a KDA layer and no
sub-phase inside it exceeds 26% of MoE. Getting KDA under 1 ms means removing
~0.5 ms spread across six sub-phases that are each 0.12-0.24 ms, and MLA needs
~0.96 ms removed with attention (0.909) as the only large single item.

Two observations for whoever continues:

- Six MoE sub-stages at 47 threads carry roughly 18 us of barrier each, so
  ~110 us of the 935 us is synchronization. Real, but it does not get you to
  1 ms on its own — and the persistent-team attempt above shows barriers do not
  disappear by merging regions, only by removing sync points.
- MLA attention at 0.909 ms against KDA's 0.481 for the same projection volume
  is the largest unexplained single number left in the profile, and has never
  been broken down. That is where I would look next.

## MLA attention: the KV scan was serial over heads

Breaking MLA down by reading it rather than instrumenting it: of its five
stages, four are batched projections and elementwise work, but the KV-cache scan
ran `k3_attention_sve` in a **serial loop over the local heads**. KDA has no
such scan, which is the whole of the 0.909 vs 0.481 difference.

`k3_attention_heads_parallel_sve` already exists in `k3_kernels.h` for exactly
this -- it splits each head's token range across the team and merges with
log-sum-exp -- and `k3_ep_runner.c:715` already uses it. The full runner did
not. Swapped in, with `mla_scratch`/`mla_stats` sized as at
`k3_ep_runner.c:869-870`.

Three reps, layer 3, 47 threads (first parallel run discarded as cold, the
pattern throughout this document):

| | layer ms | attention ms |
|---|---|---|
| parallel | 1.746 / 1.709 | 0.689 / 0.658 |
| serial | 1.954 / 1.973 / 1.953 | 0.903 / 0.915 / 0.903 |

**Attention 1.35x**, layer 1.13x.

**A caveat on the correctness gate.** `layer12` mode reports `tokens=0`, so the
output hash it prints does not cover the attention result -- it is a weak gate,
weaker than this document implied for the earlier changes. Those were
scheduling-only and unchanged by construction; this one changes the summation
order, so it rests on `make test`'s `[mla-parallel]` case (max_abs 2.794e-09
against the reference) plus the call-site mapping: the kernel indexes
`keys + h*cache_tokens*qk_dim`, which equals the caller's `h*key_stride` when
`cache_tokens = max_seq`, and likewise for values and the 128-wide output.
A generation-mode run would be a stronger check and has not been done.

## Standing position on <1 ms/layer

KDA ~1.49 ms, MLA ~1.73 ms. Extrapolated 69 x 1.49 + 24 x 1.73 = 144 ms/token
= 6.9 tok/s, from 457 ms at the start of the day.

Still not met, and still not one change away: MoE is ~0.93 ms of both layer
types with no sub-phase above 26% of it.

## Session 2026-08-06 05:26-06:00, job 50008795 (12n, layer 2 KDA / layer 3 MLA)

Method throughout: 47 threads, `--prefill-tokens 512`, expert-TP, first run
discarded as cold, reps reported individually. Every run below carried the
identical output hash `14650fb0739d0383`, so nothing here changes arithmetic.

### The framing correction: there is no bf16 expert path to build

`config.json` gives `routed_expert_hidden_size=3584`, `moe_intermediate_size=3072`,
896 experts x 92 MoE layers = **2.72 T expert params**. At bf16 that is 5.4 TB;
the checkpoint is 1.5 TB. So the experts are **natively MXFP4** and only the
dense/attention weights are bf16. The "non-quantized 1.6 TB model" *is* the
"original (bf16 + MXFP4)" row of the table above -- already the fastest decode
format in this tree. **No quantization work is on the decode critical path.**

### The win: `--moe-shard-layout row-aligned` instead of `replicated`

Under `replicated`, `routed_down` (3584x7168) and `routed_up` (7168x3584) are
staged whole on every rank -- 51 MB each, streamed **12x redundantly**, and
together the two largest weight reads in the layer. `row-aligned` shards them
and pays for it with two extra allreduces.

| layer | replicated (3 reps) | row-aligned (3 reps) | |
|---|---|---|---|
| 2 (KDA, x69) | 1.4955 / 1.5272 / 1.5097 | **1.4139 / 1.4429 / 1.3997** | 1.065x |
| 3 (MLA, x24) | 1.7107 / 1.7164 | **1.6210 / 1.6199** | 1.057x |

Distributions do not overlap: the worst row-aligned run beats the best
replicated run on both layer types.

Phase deltas, layer 2, showing where it comes from and what it costs:

| phase | replicated | row-aligned |
|---|---|---|
| dispatch_proj | 0.176 | **0.058** (3.0x -- the redundant read) |
| latent_reduce | 0.000 | 0.132 (the price) |
| moe_dispatch (parent) | 0.246 | 0.190 |
| moe_finish | 0.166 | 0.136 |
| moe (parent) | 0.932 | 0.834 |

Note the shape of the trade: sharding buys **118 us** of streaming and gives
**132 us** back as a collective, and still nets a win because `moe_finish` and
the attention `reduce` also improve. **A layout that shards without adding a
collective would be worth ~118 us/layer more** -- that is the argument for the
sharded-residual-stream restructuring, now with a measured price tag.

Extrapolated: 69 x 1.4188 + 24 x 1.6205 = **136.8 ms/token = 7.31 tok/s**,
from 6.9. Defaults changed to `row-aligned` in `run_k3_full_12n.sh`,
`pjsub_k3_full_96n.sh`, `pjsub_k3_full_96n_short_1h.sh`.

### Four things measured and rejected, all cheap, all negative

The 9 s/run turnaround at `layer12` makes one-line hypotheses nearly free to
kill. These cost about 15 minutes in total.

| attempt | layer ms | verdict |
|---|---|---|
| `--ar-groups` 2 / 3 / 4 / 6 | 1.498 / 1.520 / 1.511 / 1.500 | **noise.** The 88.6 vs 93.9 us microbenchmark gap is real but there are only ~2 collectives/layer, so ~10 us lands under a +/-5% run-to-run spread. |
| `K3_COMM_POLL_SPINS` 1 / 2 / 8 / 32 / 128 | 1.500 / 1.498 / 1.484 / 1.495 / 1.516 | **noise.** Never swept before; now it has been, and the default 4 is fine. |
| threads 45 / 46 / 47 | 1.654 / 1.514 / 1.524 | 46 == 47. **A core is free for a comm thread at no measured cost** -- relevant to any overlap work. |
| `moe_shared` task-quantization theory (64 8-row tasks over 47 threads => `ceil`=2, so 32 threads should tie 47) | shared 0.289 / 0.208 / 0.191 at 16 / 32 / 47 | **refuted.** It scales monotonically with threads, so it is not critical-path-quantized. |

**Collectives are not the 12-node lever.** `reduce` (0.117) + `moe_collective`
(0.125) = 242 us of a 1495 us layer, 16%. This kills the working hypothesis that
a 4-collectives-per-layer count was the dominant cost -- with
`moe_shard_layout=replicated` the profile shows `latent_reduce=0` and
`router_reduce=0`, i.e. the layer was already running **2** collectives, not 4.

### NUMA interleave is load-bearing (2.34x)

`k3_apply_numa_interleave` (`k3_runtime.h:75`) is now gated on
`K3_NUMA_INTERLEAVE` (default on) so it can be A/B'd in one run:

| policy | layer ms |
|---|---|
| `MPOL_INTERLEAVE` (default) | 1.500 / 1.653 |
| first-touch (`K3_NUMA_INTERLEAVE=0`) | **3.505 / 3.513** |

Every phase degrades together (attention 0.456 -> 1.145, dispatch 0.244 ->
0.821, expert 0.212 -> 0.306). First-touch puts the whole blob on the loader's
CMG, so this measures cross-CMG contention rather than anything subtle -- but it
establishes that **memory placement has 2.3x of leverage here**, which is more
than any other knob tested today. The untested variant is CMG-*local*
partitioning with a CMG-aware task->thread mapping, which is a different thing
from either arm above.

### Where the remaining time is (layer 2, row-aligned, 1.419 ms)

Per-rank active bytes vs the 652 GB/s bf16 / 106 GB/s MXFP4 measured rates:

| phase | ms | bytes/rank | roofline ms | ratio |
|---|---|---|---|---|
| attention | 0.445 | ~75 MB bf16 | 0.115 | **3.9x** |
| moe_expert | 0.211 | 23 MB mxfp4 | 0.220 | **1.0x -- at roofline** |
| moe_shared | 0.190 | 22 MB bf16 | 0.034 | **5.6x** |
| moe_dispatch | 0.190 | 5 MB + collective | -- | |
| moe_finish | 0.136 | 4 MB + collective | -- | |
| collectives | 0.242 | 2 x 43 KB | latency-bound | |

**`moe_expert` is at its kernel's measured rate and cannot be improved without a
faster MXFP4 kernel** -- and MXFP4 at 14.6% of the memory roofline is the single
largest structural inefficiency left in the model. The two phases that are off
their own roofline by 3.9x and 5.6x, `attention` and `moe_shared`, are together
0.635 ms of a 1.419 ms layer and are the next target. Both are batches of small
matvecs in their own OpenMP regions; note the persistent-team restructuring
already failed against exactly this shape, and the thread-quantization theory
above failed too, so the mechanism is still unidentified. **An `fapp` counter run
(TLB, L1D/L2 miss, CMG-remote access) on the attention phase is the measurement
that has never been taken and is the obvious next step.**

### The attention gap decomposed: the bf16 kernel is the problem, not sync

KDA layer-2 attention weights, read exactly off the staged manifest:
q/k/v/g/o_proj 5 x 14,680,064 B + f_a 1.835 MB + f_b 0.262 + b_proj 0.115 =
**75.6 MB/rank**. Thread sweep (layer 2, row-aligned, 128 samples):

| threads | 1 | 2 | 4 | 8 | 12 | 24 | 47 |
|---|---|---|---|---|---|---|---|
| attention ms | 5.941 | 3.124 | 1.694 | 1.023 | 0.864 | 0.604 | **0.464** |
| GB/s | 12.7 | 24.2 | 44.6 | 73.9 | 87.5 | 125.2 | **162.9** |
| speedup | 1.00 | 1.90 | 3.51 | 5.81 | 6.88 | 9.84 | 12.80 |

Two independent deficits, both now quantified:

1. **Single-thread efficiency: 12.7 GB/s against the 42 GB/s single-thread node
   bandwidth = 30%.** The kernel is 3.3x off before any thread interacts with
   any other.
2. **Parallel efficiency: 12.80x on 47 threads = 27%.** An Amdahl fit gives a
   serial fraction of **5.8%**, which caps speedup at 17x — so most of the
   parallel loss is real serial work (the conv step, L2 normalise, log-decay,
   the expf loop, the gated RMS norm), not barrier overhead.

**This overturns the standing assumption that the kernels are finished and only
synchronization is left.** The "bf16 matvec = 326 Gmac/s = 652 GB/s = 90% of R"
headline at the top of this document was measured on a large standalone matrix;
at the runner's actual shapes the same kernel delivers **162.9 GB/s, 22% of R**.
That number does not transfer, and every plan built on it was mis-aimed —
including the collective-count and OpenMP-restructuring work, both of which
measured as noise today.

Ranked by what each would return on the 0.464 ms attention phase:

- **Fix the single-thread bf16 matvec (3.3x available, ~0.32 ms/layer).** By far
  the largest identified win in the model, and it is ordinary kernel work.
  The 8-row form (`matvec_bf16_8row`, called from `full_bf16_run_task`,
  `k3_full_runner.c:815-825`) runs 8 strided row streams plus the shared input
  vector per thread — 9 streams x 47 threads against a finite prefetch resource,
  and A64FX is documented as prefetch-stream-limited. Test row counts 2/4/8/16
  and explicit `prfm` distance before anything else.
- Attack the 5.8% serial fraction (caps at 17x vs the 12.8x realised, so worth
  ~1.3x on the parallel part) — this is the `omp single` scalar work, and it is
  the one thing the failed persistent-team experiment was actually near.

Note the same measurement should be repeated for `moe_shared` (5.6x off) before
assuming its mechanism is identical; and note that `moe_expert` sits at its
MXFP4 kernel's measured rate, so the same "is the standalone rate real at the
model's shapes?" question should be asked of MXFP4 too.

### A faster bf16 matvec already exists in this tree and K3 does not call it

`matvec_bf16_8row_pv` (`common/ggml_dequant.h:1531`) is a p_odd
predicated-load, pair-interleaved variant documented in `common/ds4f.h:209` and
`:436` as **"+22..28% over plain bf16, BYTE-IDENTICAL result"** — same column
order, same 8-row `svaddv` reduction. DS4F selects it with `DS4F_BF16_PV=1` and
applies it to *all* bf16 matvecs.

K3 calls plain `matvec_bf16_8row` (`k3_full_runner.c:815-825`). Since bf16
matvecs are attention + moe_shared + dispatch_proj + moe_finish — i.e. most of
the layer outside `moe_expert` — 22-28% there is roughly **0.2 ms/layer**, and
byte-identical output means the layer hash must not move.

The cost is that the layout is tied to the tensor type: `k3_full_stage.py` must
write the interleaved layout for exactly the tensors read through the pv matvec
and never for flat-read norms/embeddings (`ds4f.h:210-212` is explicit that
mixing the two is the failure mode). That is a stage-format change plus a
re-stage, not a one-liner — but it is the same "a kernel that already existed
but was not being called" pattern that was worth 35% earlier in this session's
history, and it should be tried before any new kernel is written.

Note this is *complementary* to the single-thread deficit above, not a
substitute: 22-28% against a 3.3x gap still leaves the kernel far off
single-thread bandwidth.

### Two attempts at the single-thread bf16 deficit, both negative

Both are behind env knobs in `k3_full_runner.c` (default off), both kept the
hash at `14650fb0739d0383`:

| attempt | layer ms | attention ms |
|---|---|---|
| baseline (8-row) | 1.395 / 1.408 | 0.445 / 0.452 |
| `K3_BF16_ROWS=4` (two 4-row calls: 5 streams and 8 accumulators instead of 9 and 16) | 1.463 / 1.462 | 0.485 / 0.486 |
| `K3_BF16_PREFETCH=` 128 / 256 / 512 / 1024 / 2048 elements | 1.419 / 1.391 / 1.420 / 1.412 / 1.423 | 0.455 / 0.440 / 0.468 / 0.450 / 0.456 |

So the 12.7 GB/s single-thread rate is **not** prefetch-stream pressure, **not**
SVE register spill, and **not** missing software prefetch — the hardware
prefetcher already handles eight sequential streams, and fewer/wider rows is
strictly worse. Three plausible mechanisms eliminated in about ten minutes.

What is left is instruction issue. Per `2*VL` iteration the kernel executes
~16 FMA + ~16 zip/widen + 10 loads = ~42 SVE instructions to consume 512 B of
weights; at 2 issue/cycle and 2.0 GHz that is ~49 GB/s, still **3.8x above the
measured 12.7**. Either the `SVE_BF16_ZIP` widen is far more expensive than
assumed or the loop is latency-bound on something not yet identified.

**This is the highest-value open question in the model, and it needs no
allocation**: it is a single-threaded kernel, so it can be microbenchmarked with
`fapp`/`perf` on one core of any node during the queue wait. Do that before
writing another variant — three hypotheses have now been killed by measurement
and none by reasoning.

### The control that isolates the defect: MXFP4 scales, bf16 does not

Same layer, same run, same OpenMP runtime, same node — thread sweep of the two
MoE phases side by side (layer 2, row-aligned, 128 samples):

| threads | 1 | 2 | 4 | 8 | 12 | 24 | 47 | efficiency |
|---|---|---|---|---|---|---|---|---|
| `moe_expert` (MXFP4) ms | 8.034 | 4.030 | 2.036 | 1.040 | 0.711 | 0.380 | 0.211 | **x38.0 of 47 = 81%** |
| `moe_shared` (bf16) ms | 2.038 | 1.113 | 0.641 | 0.399 | 0.332 | 0.244 | 0.192 | **x10.6 of 47 = 23%** |
| `attention` (bf16) — from the sweep above | 5.941 | 3.124 | 1.694 | 1.023 | 0.864 | 0.604 | 0.464 | **x12.8 of 47 = 27%** |

**This is the result that matters.** The MXFP4 expert kernel scales at 81% on the
same 47 threads, in the same layer, through the same OpenMP runtime, on the same
allocation. So the poor scaling of the bf16 phases is **not** OpenMP
synchronization, not barrier cost, not thread wakeup, and not the node's memory
system — all of those would hit MXFP4 equally.

It is specific to the bf16 matvec path (`matvec_bf16_8row` /
`full_bf16_many`), which shows the same two deficits everywhere it appears:
**~11-13 GB/s single-threaded** (attention 12.7, shared 10.8, against 42 GB/s
single-thread node bandwidth) and **23-27% parallel efficiency**.

This retires the standing interpretation of the earlier `perf` profile
(`__kmp_fork_barrier` 37%, "~46% of runtime is OpenMP team synchronization").
That reading is why the persistent-team restructuring was attempted and why it
measured 1.2%; the barrier time is a *symptom* of threads waiting on a bf16
kernel that does not scale, not the cause. Three subsequent experiments today —
collective count, poll spins, thread count — were aimed at the same wrong target
and all measured as noise.

**Everything now points at one function.** It is single-threaded work to
diagnose, so it needs no allocation and can be done entirely during the queue
wait.

### Standalone single-thread microbenchmark: the widen is the bottleneck

`k3_bf16_bench.c`, one thread on this A64FX node, no allocation needed.
Same buffer for all three: `matvec_bf16_8row`, a pure SVE streaming read of the
identical bytes (the achievable single-thread ceiling), and
`matvec_bf16_8row_pv`. Best of 5, `rows=4224`:

| cols | matvec_bf16_8row | pure stream read | matvec_bf16_8row_pv |
|---|---|---|---|
| **7168** (q/k/v/g/o_proj) | 21.4 GB/s | **58.7 GB/s** | **33.6 GB/s (1.57x)** |
| 3584 | 15.3 | 55.8 | 18.2 (1.19x) |
| 512 | 17.9 | 60.4 | 19.6 (1.09x) |

Two conclusions, both firm:

1. **The kernel is issue-bound, not memory-bound.** Reading the same bytes with
   no arithmetic runs at 56-60 GB/s single-threaded; the matvec manages 15-21.
   A 2.7-3.7x gap on identical traffic cannot be the memory system. This is why
   prefetch distance, row count and OpenMP knobs all measured as noise — none of
   them touches instruction issue.
2. **The cost is the bf16->f32 widen, and the fix already exists.**
   `matvec_bf16_8row_pv` removes it (p_odd predicated load: 2 instructions where
   `SVE_BF16_ZIP` needs 4) and measures **1.57x at cols=7168** — better than the
   +22..28% documented in `ds4f.h`, because the documented figure is an average
   over shapes and 7168 is the best case. 7168 is also exactly the column count
   of the five attention projections that dominate the layer.

Caveat on the pv row: the timing is valid (identical access pattern and volume)
but it was fed row-major bytes, so this measures speed only. pv requires the
pair-interleaved layout, which `k3_full_stage.py` must write for exactly the
tensors read through it and never for flat-read norms/embeddings.

**Projected:** attention 0.445 -> ~0.30 ms/layer, with smaller gains on
`dispatch_proj`, `moe_finish` and `moe_shared`. Roughly 0.15-0.20 ms/layer, on
top of the 0.09 already banked from `row-aligned`. That would put KDA near
1.2 ms — still short of 1 ms, but it is the clearest remaining step and the
measurement supporting it is now on record rather than inferred.

## The pv structural change: landed, correct, and almost worthless at 47 threads

`K3_BF16_PV=1` repacks BF16 projection weights in place at load into the
pair-interleaved layout and routes them through `matvec_bf16_8row_pv`
(`full_bf16_pv_repack` + the `task->pv` branch in `k3_full_runner.c`). Repacking
at load rather than at stage time leaves `k3_full_stage.py` and the blob format
untouched, and the layout is recorded per tensor, so any tensor not on the
explicit list keeps the row-major kernel and stays correct.

Layout, derived from the kernel rather than from the prose: for an 8-row group
the block is four pair-planes of `2*cols` halfwords, `plane[2*c] = rowEven[c]`,
`plane[2*c+1] = rowOdd[c]`. The group base stays `w + r*cols`, so
`full_bf16_many`'s task construction needed no change.

**The output hash stayed `14650fb0739d0383` on the first run**, which validates
the derivation — a wrong interleave would scramble every projection.

| threads | attention ms, pv off | pv on | speedup |
|---|---|---|---|
| **1** | 5.831 / 5.832 | **4.368 / 4.367** | **1.335x** |
| **47** | 0.453 / 0.451 | 0.435 / 0.435 | **1.038x** |

Layer totals at 47 threads: 1.424 / 1.418 / 1.413 off vs 1.397 / 1.395 / 1.408
on — about **1.3%**, consistent in sign but marginal.

**So the kernel is genuinely 1.34x faster in the model, and 96% of that win
disappears at 47 threads.** That is the important result, and it corrects the
conclusion of the previous section. Attention cannot be issue-bound at 47
threads, because making the instruction stream shorter does almost nothing
there. Something shared saturates first — but it is **not** raw DRAM bandwidth
either: 75.6 MB in 0.435 ms is 174 GB/s against a 726 GB/s node ceiling.

The earlier inference "MXFP4 scales at 81%, therefore the bf16 problem is not the
memory system" was too strong. MXFP4 does far more arithmetic per byte, so it can
scale while a leaner kernel on the same fabric does not.

**The prime suspect is now cross-CMG traffic.** `MPOL_INTERLEAVE` spreads every
weight over 4 CMGs, so ~75% of reads are remote, and remote concurrency — not
aggregate bandwidth — would cap exactly this pattern while leaving MXFP4 (which
is not bandwidth-limited) untouched. Note today's NUMA A/B did **not** test this:
first-touch puts everything on one CMG and is worse than interleave, so both arms
were bad. The untested arm is **CMG-local partitioning with a CMG-aware
task->thread mapping**, and it is now the top candidate for the 4x.

**Disposition:** `K3_BF16_PV` defaults **off**. It is correct and free of risk,
but 1.3% does not justify a load-time repack of every projection until the
cross-CMG question is settled — at which point the pv win may reappear, since it
is real whenever the phase is issue-bound rather than fabric-bound.

## CMG placement: the interconnect is the cap, and R=726 was measured wrong

Topology (`/sys/devices/system/node/`): **4 CMGs = NUMA nodes 4-7, cores 12-23 /
24-35 / 36-47 / 48-59**, ~7.5 GB each. `node4/distance` is `10 20 30 30`, so even
the three remote CMGs are not equidistant.

### Local vs remote bandwidth (`k3_cmg_bw_bench.c`)

Buffer `mbind`-ed to one CMG, threads pinned to one CMG, pure SVE streaming read:

| | mem4 | mem5 | mem6 | mem7 |
|---|---|---|---|---|
| **1 thread** — cpu4 | **43.9** | 35.8 | 33.7 | 33.7 |
| **12 threads** — cpu4 | **226.2** | 118.8 | 119.9 | 119.9 |
| 12 threads — cpu5 | 118.7 | **227.1** | 120.1 | 119.8 |
| 12 threads — cpu6 | 119.9 | 119.8 | **227.2** | 118.9 |
| 12 threads — cpu7 | 119.9 | 119.8 | 118.8 | **226.4** |

**CMG-local 226 GB/s, inter-CMG ~119 GB/s.** At one thread the penalty is only
1.3x — latency is hidden — so this is a *bandwidth* ceiling that only appears
under concurrency, which is why no single-threaded experiment could have found
it.

### The whole-node consequence (`k3_cmg_matvec_bench.c`, streaming arm)

| threads | interleaved (what K3 does) | CMG-local partitioned | |
|---|---|---|---|
| 12 | 160.1 | 889.3 | 5.55x |
| 24 | 272.4 | 876.2 | 3.22x |
| 47 | 455.2 | 852.4 | **1.87x** |
| 48 | 460.9 | 854.1 | 1.85x |

**The node's real read ceiling is ~854-890 GB/s, not the 726 GB/s recorded at the
top of this document.** That 726 came from `ds4f_decode_bw_bench` under
interleave; it is not a hardware constant, it is a placement artifact. Under the
interleave `k3_apply_numa_interleave` actually applies, the node delivers ~460.
**Every roofline percentage earlier in this file is computed against the wrong
denominator** — the quantized kernels are further from the ceiling than stated,
and bf16's "90% of R" was measured on a first-touch (accidentally CMG-local)
buffer.

### On the real kernel

`matvec_bf16_8row`, 4224x7168, 47 threads, best of 5:

| shape | interleaved | CMG-local | |
|---|---|---|---|
| 7168 cols (q/k/v/g/o_proj) | 261.3 GB/s | **373.5** | **1.43x** |
| 3584 cols | 201.9 | 252.9 | 1.25x |
| 7168 cols, 48 threads | 273.2 | 358.5 | 1.31x |

So **CMG-local placement is worth ~1.43x on the dominant attention shape** —
larger than anything else currently on the table, and it composes with pv rather
than competing (pv's issue win should partly reappear once the fabric is not the
cap).

### Two traps this measurement walked into, both worth remembering

1. **The subnormal-FMUL trap, again.** Filling the weight buffer with
   `memset(buf,1,n)` makes every bf16 `0x0101` ~ 2.4e-38. A64FX traps to
   microcode on subnormal FMUL and the *entire* benchmark read 0.6 GB/s across
   every allocator and policy — a flat, plausible-looking table that was pure
   artifact. Fill benchmark weights with realistic values (`0x3f80 ^ (i & 0x7f)`).
2. **Partial-CMG coverage.** With `cmg = id/12`, any thread count below 47 leaves
   whole CMGs unvisited, so the CMG-local arm silently does 1/4 or 1/2 of the
   work and reports a beautiful 4.00x. **Only 47/48-thread rows of that table are
   valid.** A speedup that lands on exactly 4.00x/1.99x is a work-accounting bug,
   not a result.

### Implementation sketch

- **Placement:** for each 2-D BF16 matvec weight, `mbind` its row-blocks to the
  4 CMGs (page-aligned; a 1056-row block at 7168 cols is 14.8 MB, so alignment is
  cheap). Do it per tensor after load, so the blob stays one allocation.
- **Task mapping:** with `OMP_PROC_BIND=close`/`OMP_PLACES=cores`, thread `t`
  sits on core `12+t`, i.e. CMG `t/12`. Ordering `full_bf16_many`'s task array by
  CMG and using `schedule(static)` makes the mapping fall out for free.
- **The 47/48 tension:** CMG-local wants 12 threads per CMG (=48, the cliff), so
  at 47 the split is 12/12/12/11 and the row-blocks must be sized to match, not
  cut into equal quarters.
- Keep it behind `K3_CMG_LOCAL=1` next to `K3_NUMA_INTERLEAVE`, and gate on the
  layer-2 hash `14650fb0739d0383` — placement cannot change arithmetic.

## CMG placement, done properly: per-CMG replication wins, routing does not

The `k3_cmg_matvec_bench.c` result above (1.43x for CMG-local) **overstated the
in-model opportunity**, and finding out why produced the actual fix.

### The page size is 2 MB, and that changes the whole picture

`sysconf(_SC_PAGESIZE)` reports **64 KB**, but the heap this runner allocates
from is backed by **2 MB large pages**. Standalone test (a standalone mbind probe):

| target | mbind, 64 KB-aligned range | mbind, 2 MB-aligned |
|---|---|---|
| `posix_memalign` (the blob) | **EINVAL** | OK |
| `mmap` | OK | OK |

Smallest working alignment on the heap: **2097152 bytes**. Two consequences:

1. **The first CMG implementation was a silent no-op.** It aligned to
   `sysconf(_SC_PAGESIZE)`, so every `mbind` returned EINVAL, `cmg_split` stayed
   zero and the CMG path never ran. The A/B that "measured" it (1.41 vs 1.46) was
   pure noise. `mbind` also returns 0 without moving anything unless
   `MPOL_MF_MOVE` is set, so *two* silent-failure modes had to be excluded before
   any number meant anything. Verify placement with `get_mempolicy`, never assume.
2. **`MPOL_INTERLEAVE` round-robins 2 MB blocks, not 4 KB ones.** Probing
   consecutive large pages gives `5 6 7 4 5 6 7 4`. A 7168-column bf16 row is
   14 KB and an 8-row task 114 KB, so a task's weights already sit on **one** CMG.
   The `k3_cmg_matvec_bench.c` arm used `mmap` (small pages), so every task there
   straddled all four CMGs -- a harsher pattern than the runner's, which is why it
   promised 1.43x that the model could not deliver.

### What is actually wrong, and the 96-node prediction

Tasks are already CMG-clustered; what is uncontrolled is *which* CMG runs them.
Two fixes were built and measured (layer 2, 47 threads, 512 samples, all runs
hash `14650fb0739d0383`):

**(a) CMG routing** — map each task to the CMG owning its pages, run it there.
**Net loss, ~2.5%**: 1.453 / 1.455 vs 1.418 / 1.419. `dispatch_proj` improved 8%
but `attention` lost 4.5% and `moe_shared` 10%. At 12 nodes a projection is only
3-7 large pages, so the per-CMG split is 2/1/2/2 and the imbalance costs more
than the locality gains.

**(b) Per-CMG replication** — give every CMG its own bound copy, keep the normal
schedule so any thread can still take any task. **This is the win:**

| | base (3 reps) | replicated (3 reps) | |
|---|---|---|---|
| layer 2 (KDA) | 1.4150 / 1.4187 / 1.4205 | **1.3054 / 1.3234 / 1.3060** | **1.081x** |
| — attention | 0.4565 | 0.3888 | 1.17x |
| — moe_shared | 0.1998 | 0.1770 | 1.13x |
| — dispatch_proj | 0.0586 | 0.0458 | 1.28x |
| layer 3 (MLA) | 1.6469 / 1.6414 | **1.5393 / 1.5428** | 1.068x |

Distributions do not overlap on either layer. Extrapolated:
69 x 1.3116 + 24 x 1.5411 = **127.5 ms/token = 7.84 tok/s**, from 7.28.

**And it should be worth far more at 96 nodes.** At 96n a per-rank attention
projection is 96 heads/96 nodes = 128 rows x 7168 x 2 = **1.835 MB -- smaller
than one 2 MB page**, so each tensor lands wholly on one CMG and 36 of 47 threads
read it across the ~119 GB/s interconnect. That condition cannot occur at 12
nodes, so `K3_CMG_FORCE=<cmg>` was added to simulate it:

| layer 2 | ms | attention |
|---|---|---|
| interleaved (12n today) | 1.416 / 1.417 | 0.457 |
| **forced onto one CMG (= the 96n condition)** | **2.026** | **0.931 (2.04x)** |
| forced + replicated | **1.320** | 0.385 |

**Replication completely neutralises a 1.53x penalty that 96 nodes gets for
free.** This is a concrete, quantified contribution to the unexplained gap
between the 12n extrapolation and the 1.699 tok/s measured at 96 nodes, and
unlike most of that gap it now has a fix attached.

### Cost and disposition

Only bf16 projections are replicated -- the MXFP4 experts, which are the bulk of
the model, are not. At 12 nodes that is 121.7 MB/layer/rank replicated, so 3
extra copies = +365 MB. Scaled to 96 nodes: **+45.6 MB/layer/rank = 4.14 GB/rank
over 93 layers**, against ~15.6 GB of weights -- a 27% increase, which should fit
32 GB/node but has not been verified against the KV cache at the target context.

`K3_CMG_REPLICATE=1`, default **off**. Recommended for the 96-node job once the
memory headroom is checked; the `K3_CMG_LOCAL` routing path is retained but
should not be used.

### Combined: replication + pv

pv was parked at 1.038x because the phase was fabric-bound; with replication
removing that cap its issue-side win partly returns, as predicted. Final A/B,
`K3_CMG_REPLICATE=1 K3_BF16_PV=1` against stock:

| | base | replicate + pv | |
|---|---|---|---|
| layer 2 (KDA, x69) | 1.4253 / 1.4549 / 1.4340 | **1.2841 / 1.2851 / 1.3167** | **1.110x** |
| layer 3 (MLA, x24) | 1.6427 / 1.6632 | **1.5224 / 1.5282** | **1.084x** |

Distributions do not overlap on either layer; all runs hash `14650fb0739d0383`.

69 x 1.2953 + 24 x 1.5253 = **126.0 ms/token = 7.94 tok/s**, from 7.28 at the
start of this session and 6.9 before `row-aligned`.

pv contributes ~1.5-2% of that, replication the rest. pv is worth keeping on
only because it is free once the weights are replicated; on its own it is still
noise.

## The SiTU activation: 3 libm calls per element, on the wrong path

Instrumenting `full_kda_forward` and the shared-expert block (new profile phases
`kda_*` and `shared_*`) found the largest single remaining item, and it was not
where any of the bandwidth analysis pointed.

### KDA attention breakdown (replicate + pv, layer 2, attention 0.370 ms)

| stage | ms | |
|---|---|---|
| **kda_qkv** | **0.100** | 73.4 MB ⇒ **734 GB/s = 86% of the 854 GB/s CMG-local ceiling** |
| kda_out | 0.064 | o_proj, 14 MB ⇒ 219 GB/s |
| kda_serial | 0.047 | L2-norm + sigmoid + log-decay + a 1024-iteration scalar `expf`, one thread |
| kda_step | 0.021 | |
| kda_decay_proj | 0.018 | |
| kda_conv | 0.017 | |

The five big projections are **done** -- 86% of roofline after replication. What
is left in attention is the serial block and the small stages.

### The shared-expert block: half of it was one activation

| stage | ms |
|---|---|
| shared_gateup (2 x 512x7168) | 0.037 |
| **shared_situ (512 elements)** | **0.089** |
| shared_down (7168x512) | 0.056 |

**0.089 ms for an activation over 512 values** -- ~350 cycles per element.
`k3_situ_sve` (`k3_kernels.h:129`) is a one-line forward to `k3_situ_ref`, which
computes `4*tanh(g/4)*sigmoid(g)*25*tanh(u/25)`: **three libm transcendentals per
element**, 1536 calls for 512 lanes.

`k3_situ_fast_sve` (`k3_kernels.h:157`) implements the same function on FEXPA and
sits directly below it. **The 16 routed experts already call it** -- `k3_moe.h`
lines 183, 616, 662, 747, 830 -- while the 2 shared experts and the dense layer
did not. The model was already approximating this activation for the routed path
and paying full libm price for the shared one.

`K3_SITU_FAST=1` routes them the same way:

| | off | on | |
|---|---|---|---|
| shared_situ | 0.0889 / 0.0888 | **0.0020 / 0.0020** | **44x** |
| moe_shared | 0.1876 / 0.1820 | **0.0953 / 0.0955** | **1.94x** |
| layer | 1.319 / 1.297 | 1.246 / 1.209 | |

**This is an accuracy change, not a free one.** The gate is `make test`'s
`[situ-fexpa]` case, which measures **max_abs 1.457e-03 against a 2e-3
tolerance**, plus the argument that it makes the two expert paths consistent.
The `layer12` hash does *not* move, but that hash reports `tokens=0` and is
already documented above as a weak gate -- do not read it as verification here.

## Session total

Stock defaults vs `K3_CMG_REPLICATE=1 K3_BF16_PV=1 K3_SITU_FAST=1`:

| | stock | all three | |
|---|---|---|---|
| layer 2 (KDA, x69) | 1.4303 / 1.4326 | **1.2083 / 1.2076 / 1.2073** | |
| layer 3 (MLA, x24) | 1.6528 / 1.6493 | **1.4408 / 1.4415** | |
| **token** | 138.4 ms = **7.23 tok/s** | **117.9 ms = 8.48 tok/s** | **1.174x** |

The all-flags runs are reproducible to four digits where stock swings 1.43-1.59.
From 6.9 tok/s at the start of the day (144 ms) via `row-aligned`, CMG
replication, pv and the SiTU fix. Recommended launch:

```
K3_MOE_SHARD_LAYOUT=row-aligned K3_CMG_REPLICATE=1 K3_BF16_PV=1 K3_SITU_FAST=1
```

All three new flags default **off**: replication costs +4.14 GB/rank at 96n, and
`K3_SITU_FAST` is an accuracy decision that belongs to whoever owns the model's
output quality.

### Where the remaining 1.21 ms sits (layer 2, all flags)

| phase | ms | assessment |
|---|---|---|
| attention | ~0.37 | qkv at 86% of roofline; `kda_serial` 0.047 and `kda_out` 0.064 are what is left |
| moe_expert | 0.211 | at the MXFP4 kernel's own rate; MXFP4 is 14.6% of memory roofline, so the headroom is real but needs kernel work |
| 3 collectives | ~0.35 | see below |
| moe_shared | 0.095 | was 0.199 |
| moe_finish | 0.131 | |
| dispatch_proj | 0.039 | |

**Reducing the collective *count* is not the lever it looked like.** Switching to
`moe_shard_layout=replicated` removes the `latent_reduce` (0.133 ms) outright,
and the layer total is unchanged (1.2827 row-aligned vs 1.2797 replicated, 3 reps
each). The saving is absorbed by `dispatch_proj` growing 0.039 -> 0.107. Note
this also means the two layouts are now *equivalent* at 12 nodes where
`row-aligned` was 1.065x ahead this morning -- a clean demonstration that
comm-vs-compute trades flip with kernel speed. Keep `row-aligned` for 96 nodes,
where `replicated` would stream 102 MB/rank/layer that does not shrink with node
count.

## CORRECTION: `output_hash` is not a gate, and never was

**Every "hash unchanged" claim in this document and in `k3-resume.md` before this
section is vacuous.** `layer12` runs with `generated_tokens=0`, and
`output_hash=14650fb0739d0383` is emitted **identically for layer 2 (KDA) and
layer 3 (MLA)** — two different layer types, different weights, different code
paths, same hash. It does not depend on the layer computation at all.

The real gate is **`hidden_hash`**, also in `output.txt`, which does vary by
layer (L2 `6e94844067bbd6a1`, L3 `129ef3c996499aa9`). Re-validating every change
in this session against it:

| change | hidden_hash | verdict |
|---|---|---|
| baseline (stock, row-aligned) | `6e94844067bbd6a1` | — |
| `K3_CMG_REPLICATE=1` | `6e94844067bbd6a1` | **bit-identical** — placement only, as claimed |
| `K3_CMG_LOCAL=1` (routing) | `6e94844067bbd6a1` | bit-identical |
| `K3_CMG_FORCE=0` | `6e94844067bbd6a1` | bit-identical |
| threads 16/32/47, `K3_BF16_ROWS=4`, prefetch, ar_groups, poll spins | `6e94844067bbd6a1` | bit-identical |
| **`K3_BF16_PV=1`** | `99c89d76b4452d7d` | **changes numerics** |
| **`moe_shard_layout=replicated` vs `row-aligned`** | `6bd20d172c2bead7` vs `6e94844067bbd6a1` | **changes numerics** |
| `K3_SITU_FAST=1` | `6cf26fdcdbaf52af` | changes numerics (intended) |
| `K3_FAST_EXP=1` | `b1ffa8f58b4010bc` | changes numerics (intended) |

So the CMG work is exactly what it claimed to be, but **two changes reported this
session as "hash-identical" are not bit-identical**:

- **pv.** `ds4f.h:209` advertises "BYTE-IDENTICAL", and that does not hold for
  this integration: `matvec_bf16_8row` accumulates in a lo/hi pair and adds them
  at the end, while `matvec_bf16_8row_pv` uses a single accumulator per row, so
  the reduction order differs. Quantified against an f64 reference at
  cols=7168 (`k3_bf16_pv_error_test.c`): **`8row` max abs error 4.768e-07, `pv`
  4.172e-07, max relative difference between them 2.539e-06.** Both sit at f32
  epsilon and pv is marginally *closer* to exact, so this is a benign
  reassociation — but it is a reassociation, not an identity.
- **`row-aligned`.** Sharding `routed_down`/`routed_up` on the contraction
  dimension changes the summation order, exactly the reassociation class already
  recorded for ds4f TP x Q8. Benign, but it means the layout switch was never
  validated by the hash it was reported with.

**Method going forward:** gate on `hidden_hash`, and for anything that moves it,
state a measured error bound rather than a hash. `output_hash` should be deleted
or fixed; as it stands it invites exactly this mistake.

## Open defect: MLA layer output is not reproducible run to run

Found while re-validating against `hidden_hash`. This is **pre-existing, present
at stock defaults, and not caused by anything in this session.**

Layer 3 (MLA), stock, `--prefill-tokens 512`, three consecutive runs:
`76383adb7d54fff3`, `c85401fcc4bf9abc`, `a26597673c8165b9` — three different
answers. Layer 2 (KDA), same conditions, twice: `6e94844067bbd6a1` both times.

Narrowing (layer 3, `--prefill-tokens 64`, two runs each):

| threads | `parts` = (threads+heads-1)/heads | reproducible? |
|---|---|---|
| 1 | 1 | **yes** — `fda4a12872c3851e` twice |
| 8 | 1 | no |
| 16 | 2 | no |
| 47 | 6 | no |

**It is not the log-sum-exp merge.** At 8 threads `parts=1`, so
`k3_attention_heads_parallel_sve` does no token-range splitting and no merge at
all, and the output is still non-reproducible. What was excluded:

- Scratch sizing: `mla_scratch`/`mla_stats` are **over**-allocated (48128 and 768
  floats against 6144 and 112 needed at heads=8/parts=6), so not an overflow.
- Buffer overrun on `m->q`/`m->k`: correctly sized via `max_q_channels` /
  `max_k_channels` (`k3_full_runner.c:2502-2505`); MLA's 1536/2048/1024 writes
  all fit.
- OpenMP reductions: there are none in `k3_full_runner.c` or `k3_kernels.h`.
- The merge order in `k3_attention_heads_parallel_sve` is fixed (`schedule(static)`,
  `p` iterated in order, double accumulator).

**ROOT CAUSE FOUND AND FIXED — see the next section.** The text above is kept
because the *narrowing* is what led to it. Reproduce the old behaviour with:

```
K3_MOE_SHARD_LAYOUT=row-aligned ./run_k3_full_12n.sh --mode layer12 \
  --layer-index 3 --expert-tp --prefill-tokens 64 --stage-dir <l3 stage>
# compare hidden_hash in <result>/output.txt across runs
```

**Why this matters beyond reproducibility.** 24 of 93 layers are MLA.
`--comm-deterministic` defaults to 1 (`k3_full_runner.c:1875`) precisely because
"fixed-root reductions are required for rank-identical full decode" — but if a
rank's own MLA arithmetic varies run to run, that guarantee does not hold
regardless of the collectives. It also means **no MLA change can be validated by
hash**, which is the gate the outstanding TODO on the MLA parallel-attention
change was going to rely on. Fix this before trusting any MLA correctness claim.

## The MLA defect was a data race on the output buffer

Adding `K3_MLA_TRACE=1` (FNV hash of each MLA intermediate) localised it in one
run pair. Every intermediate was identical across runs — `latent(tmp,q_a)`,
`latent(tmp2,kv_a)`, `q_b`, `kv_b`, **`attn_out` all matched** — and only
`layer_out` differed. That leaves exactly one operation.

`full_forward_token` calls `full_mla_forward(m, l, m->normed, m->attn)`, so
inside the function **`out` and `m->attn` are the same buffer**. The final
projection was:

```c
for (i < 1024) m->attn[i] *= k3_sigmoidf(m->gate[i]);
full_bf16_matvec(out, &l->mla_o_proj, K3_HIDDEN, 1024, m->attn, m->threads);
```

`o_proj` is 7168x1024, so the matvec **writes `out[0..7167]` while reading
`m->attn[0..1023]` — the same memory**. Across 47 threads, whichever threads own
output rows 0..1023 clobber the input before other threads have read it.

This is a genuine correctness bug, not merely a reordering: the values consumed
were partly finished output, so **MLA layers were computing intermittently wrong
results** — 24 of 93 layers, on every run, including the 96-node job.

`full_kda_forward` is immune purely by accident of buffer choice: its gated RMS
norm writes to `m->tmp` and `o_proj` reads from there, never from `out`. That is
the entire reason KDA was reproducible and MLA was not, and why the bug survived
this long.

**Fix:** stage the gated attention in `m->tmp` (dead at that point — it held
`q_a`, already consumed by the `attn_proj` batch) and feed `o_proj` from it,
exactly as KDA does.

| layer 3 (MLA), 47 threads | before | after |
|---|---|---|
| hidden_hash, 3 runs | `76383adb…` / `c85401fc…` / `a2659767…` | **`0d8160ab3188f259` x3** |
| all-flags, 2 runs | `fd42aeef…` / `da4b7d6a…` | **`71f573c950576aa5` x2** |
| layer ms | 1.6528 / 1.6493 | 1.6429 / 1.6439 |

No cost — the extra 1024-element copy is free and the timings are, if anything,
slightly better and now stable. KDA is unchanged (`6e94844067bbd6a1`).

A 1-thread run still gives a different hash (`f28e6ea2…`) from a 47-thread run.
That is expected and benign: `parts = (threads+heads-1)/heads` is 1 vs 6, so the
log-sum-exp partitioning differs. Thread-count-dependent reassociation, not a
race — the same thread count now always reproduces.

**Two lessons.** The bug was found only because `hidden_hash` replaced the
vacuous `output_hash`; with the old gate every one of these runs "passed".
And the narrowing that mattered was per-buffer tracing, after two plausible
hypotheses (the log-sum-exp merge, then buffer sizing) had both been measured
false — the serial-scan bisect exonerated the attention kernel entirely.

## Session total, final

| | stock | all four flags |
|---|---|---|
| layer 2 (KDA, x69) | 1.4278 / 1.4303 / 1.4326 | **1.1617 / 1.1610 / 1.1608** |
| layer 3 (MLA, x24) | 1.6429 / 1.6439 | **1.4317 / 1.4318** |
| **token** | 138.1 ms = **7.24 tok/s** | **114.5 ms = 8.73 tok/s** |

**1.206x**, from 6.9 tok/s at the start of the day, and MLA now deterministic.

## Collectives: four per layer, at the floor, and no knob moves them

Splitting `moe_finish` and `kda_out` (new phases `finish_matvec`/`finish_reduce`,
`kda_grmsnorm`/`kda_oproj`) completes the accounting. Layer 2, all flags,
1.16 ms:

| collective | ms |
|---|---|
| `reduce` (attention out, 28 KB) | 0.093 |
| `latent_reduce` (14 KB) | 0.134 |
| `moe_collective` (43 KB) | 0.122 |
| **`finish_reduce`** (28 KB) — hidden inside `moe_finish` until now | 0.099 |
| **total** | **0.448 = 38% of the layer** |

There are **four** collectives per layer under `row-aligned`, not the two or
three assumed earlier in this document. At ~112 us each they sit exactly on the
`tp_ar_ack_test` floor (88.6-106.9 us), so there is no arrival skew to recover —
only latency or count.

Every remaining knob, measured with all flags on (control 1.1608 / 1.1659):

| knob | layer ms |
|---|---|
| `K3_COMM_A2A=1` | 1.178 |
| `K3_COMM_ROBUST=0` / `=1` | 1.163 / 1.162 |
| `K3_COMM_POLL_SPINS=16` | 1.162 |
| `K3_AR_GROUPS=3` | **1.263** (collectives 540 us) |
| `K3_AR_GROUPS=6` | 1.162 |

All neutral or worse. **`ar_groups=3` is actively harmful in the runner** despite
being 1.21x faster in the standalone `tp_ar_ack_test` — a microbenchmark result
that does not survive contact with the real arrival pattern. Keep the default 2.

The one knob that does work is payload size:

| | layer ms (3 reps) | hidden_hash |
|---|---|---|
| `K3_COMM_BF16=0` | 1.1615 / 1.1706 / 1.1594 | `b1ffa8f58b4010bc` |
| `K3_COMM_BF16=1` | **1.1215 / 1.1215 / 1.1520** | `61c09c5fb54f15a5` |

~2.8%, every bf16 run beating every f32 run. **Not recommended without a quality
gate**: it transports the residual-stream allreduce in bf16, i.e. ~8 mantissa
bits instead of 24, on every one of 93 layers. That is a far more aggressive
approximation than the FEXPA activations, and unlike them it has no error test.

### Why the count cannot simply be cut at 12 nodes

`moe_shard_layout=replicated` runs **two** collectives instead of four (0.235 vs
0.448) and the layer total is unchanged (1.1616 vs 1.1576-1.1605). Replication
makes `routed_down`/`routed_up` full 51 MB tensors instead of sharded, adding
~94 MB/rank/layer of streaming — about 128 us at the CMG-local rate — which eats
the 213 us of saved collective time.

**This flips at 96 nodes** and is the reason to keep `row-aligned` there: the
sharded weights shrink 8x with node count while the replicated ones do not, so
replication's streaming cost stays fixed at ~128 us while its collective saving
stays roughly constant. Sharded wins as soon as compute is small.

### Standing budget at 12 nodes

Layer 1.16 ms = **0.45 comm + 0.71 compute**. Reaching <1 ms/layer here needs
compute under 0.55, i.e. ~0.16 ms removed from: `moe_expert` 0.210,
`kda_oproj` 0.056, `shared_down` 0.056, `finish_matvec` 0.026.

`moe_expert` is the only item large enough to matter and it is **not** badly
structured: `k3_expert_tp_forward_selected_mxfp4` opens one team for all 48
selected expert matrices, and the decode kernel `k3_matvec_mxfp4_8row` is already
unrolled with a prefetch macro. At 212 Gmac/s it is ~14% of a core's FMA peak,
which is about what a nibble-unpack-plus-scale dequant costs per FMA. Improving
it means a different dequant strategy, not tuning — real work, not a knob.

Note the 12-node budget **understates 96 nodes**: compute there is ~8x smaller
while comm is similar, so the layer would be comm-dominated at ~0.5 ms.

## MXFP4: the e8m0 scale was crossing GPR->FPR eight times per block

`moe_expert` (0.210 ms) is the largest compute item left. It is **not** badly
structured — `k3_expert_tp_forward_selected_mxfp4` opens one team for all 48
selected expert matrices, and `k3_matvec_mxfp4_8row` already folds the shared
scale into the accumulator once per 32-value block rather than per value.

Single-thread, cache-resident microbenchmark (`k3_mxfp4_scale_bench.c`, k=3584,
256 rows; note the first attempt was dead-code-eliminated and reported 1e6
Gmac/s — the outputs must be sunk):

| variant | Gmac/s |
|---|---|
| current: `ggml_e8m0_to_fp32(S[b])` | 20.26 |
| scales pre-converted to f32 in memory | 27.07 (1.34x) |
| no scale at all (upper bound) | 26.83 (1.32x) |
| **e8m0 via a 256-entry f32 LUT, same bytes on the wire** | **26.98 (1.33x)** |

The broadcast is free — the pre-converted and no-scale variants tie. The cost is
that `ggml_e8m0_to_fp32` computes the right bits with a shift but leaves them in
a **general-purpose register**, so `svmla_n_f32_x` pays a GPR->FPR domain
crossing eight times per block. Indexing a 1 KB table instead lands the scale
directly in an FP register and recovers the whole 1.33x **with no format change
and no extra memory traffic**.

Bit-identical by construction (entry `i` is `(uint32)i << 23`); verified against
`ggml_e8m0_to_fp32` for all 256 codes, 0 mismatches.

In-model (layer 2, all flags, 3 reps): `moe_expert` **0.2101 -> 0.1978, 1.06x**,
`hidden_hash` unchanged at `b1ffa8f58b4010bc` as required.

**1.33x standalone, 1.06x in the model** — which is itself the useful result:
`moe_expert` runs at ~22% of this kernel's own compute rate, so it is bound by
memory access pattern (48 small matrices of ~486 KB, 16 short row-streams per
thread), not by dequant arithmetic. That also retires the earlier claim in this
document that `moe_expert` "is at the MXFP4 kernel's own rate" — that 212 Gmac/s
figure was measured *in this same suboptimal in-model condition*, so it was
circular. The real headroom is in placement and streaming, not the dequant.

Kept unconditionally (bit-identical, no cost). Note a latent bug found on the
way: `k3_moe.h:702` references an undeclared `local` in the non-OpenMP branch of
`k3_expert_tp_forward_selected_mxfp4`; it only compiles because `_OPENMP` is
always defined in the runner.

## Final standing, 2026-08-06

| | stock | all flags |
|---|---|---|
| layer 2 (KDA, x69) | 1.4111 / 1.4089 | **1.1604 / 1.1535 / 1.1463** |
| layer 3 (MLA, x24) | ~1.632 | **1.4203 / 1.4190** |
| **token** | 136.5 ms = **7.33 tok/s** | **113.7 ms = 8.80 tok/s** |

From 144 ms / 6.9 tok/s at the start of the day. `make test` 46 OK / 0 FAIL,
`run_k3_ep.sh` checksum `+3.650037202e+02`.
