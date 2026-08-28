# GLM-5.3F A64FX 12-node decode-first plan

## Measured anchors (job 51040571)

- Full 34-layer/64-head KDA recurrent update, 136 MiB replicated test state:
  **1.483 ms/token** best at 24 threads. The production head-TP layout owns only
  5--6 heads/rank, so KDA state arithmetic is not the primary limiter.
- Production uTofu all-reduce, 12 ranks, 4096 f32 values: **105.9 us**.
- 78 back-to-back reductions: **5.02 ms/token**. Robust modes 0/1/2 are equal;
  changing completion mode is not a useful optimization.
- Physical 1M CP cache allocation: 12/12 ranks pass with 1.28 GiB/rank committed.
- Real staged F8_E4M3 quarter-expert (1024x4096 gate/up + 4096x512 down),
  anonymous resident weights: **0.438 ms/task mean** at 24 threads with the
  exact gather-free decoder, versus 0.477 ms/task with the LUT-gather decoder.
  A four-task/four-CMG kernel reduces the measured batch critical path to
  **1.13 ms mean / 0.424 ms best** while the background full-model stager is
  active. Repeat the steady-state number after staging exits.
- Full 42-layer, 12-rank four-way expert decode with 23.631 GiB anonymous
  weights/rank and one real 4096-float MPI combine/layer: **47.39 tok/s** over
  200 tokens (**21.104 ms/token**). Compute is 12.615 ms/token; combine plus
  arrival wait is 8.617 ms/token. An unloaded MPI baseline is 73.2 us/call,
  or 3.075 ms/token, leaving **5.56 ms/token of rank-arrival skew**. Resident
  MemAvailable is 6.0--6.45 GiB/rank.

## Decode decomposition

Use one process per A64FX node and all 12 ranks as the expert group.

- Routed experts: split every expert four ways over its intermediate dimension.
  Part `p` belongs to `(e % 12 + p*3) % 12`; each rank holds 96 quarter-experts.
  A synthetic occupancy simulation plus exact-shape GEMVs reduced the expected
  slowest-rank expert critical path from ~0.47 to ~0.29 ms/layer versus whole
  experts. An exhaustive search of all translated four-owner offset sets confirms
  `{0,3,6,9}` has the lowest top-8 slowest-rank occupancy: mean 4.061 tasks,
  p95/p99 6/6. Execute local hits concurrently as independent CMG teams; the
  existing MLP hidden-vector sum combines the partial outputs.
- Quantization-aligned 12-way slicing is the next decode experiment. Partition
  the 16 FP8 intermediate block rows, not 2048 raw elements: each expert has
  eight 128-wide and four 256-wide rank shards. This preserves the compressed
  128x128 scale grid, gives every rank all eight routed tasks, and reduces the
  simulated slowest-rank work from 16.244 to **12.061 block rows/layer**
  (p95/p99 14/14). The full 12-way stage must beat the verified four-way
  47.39 tok/s result before replacing the default.
- KDA layers: partition 64 heads as balanced contiguous ranges (5 or 6/rank).
  Q/K/V, gates, convolution channels, and recurrent state follow head ownership.
  `o_proj` is column-parallel; one hidden-vector sum completes attention.
- MLP: router logits are replicated. Routed and shared expert outputs are local
  partial hidden vectors; one sum completes the MLP.
- mHC is local because its four hidden streams are replicated immediately after
  each attention/MLP reduction.
- Sparse layers, short context: head-TP with replicated latent/index cache. This
  is the fastest decode mode and remains active only while its explicit HBM cap fits.
- Sparse layers, long context: transition to block-CP latent/index cache. Gather
  the small query representation, select/score rank-owned blocks, and merge online
  softmax statistics. This adds communication but keeps 256K--1M memory bounded.

The short-context target is two hidden-vector reductions/layer: 90 reductions for
45 layers plus the head argmax. The measured bare-wire floor is about **9.5 ms/token**.
Real decode must profile rank arrival skew; historical GLM-5.2 real-weight runs saw
effective collective latency around 0.66 ms when expert work was imbalanced.

## Optimization order

1. Decode-only rank-owned stager; do not stage vision or unused MTP tensors.
2. BF16 KDA projection GEMVs and FP8 routed/shared-expert GEMVs. Keep checkpoint
   scales compressed at one F32 value per 128x128 block; use the exact SVE
   bit-decode path rather than LUT gathers.
3. Short-context head-TP forward with exactly two reductions/layer.
4. Expert scheduling sorted by local hit count to reduce rank skew.
5. Continuous batch after single-stream correctness; decode FP8 weights once per
   active expert and reuse them across tokens.
6. Add the long-context CP transition after the fast path is stable.

Every benchmark must report kernel time, collective time, arrival-wait time,
weight bytes/rank, and end-to-end tokens/s. A tight-loop collective number alone
is not an end-to-end communication claim.
