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
