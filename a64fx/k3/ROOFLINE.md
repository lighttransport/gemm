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
