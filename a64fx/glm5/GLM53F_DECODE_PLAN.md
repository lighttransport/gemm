# GLM-5.3F A64FX 12-node decode-first plan

## Measured anchors (job 51040571)

- Full 34-layer/64-head KDA recurrent update, 136 MiB replicated test state:
  **1.483 ms/token** best at 24 threads. The production head-TP layout owns only
  5--6 heads/rank, so KDA state arithmetic is not the primary limiter.
- Production uTofu all-reduce, 12 ranks, 4096 f32 values: **105.9 us**.
- 78 back-to-back reductions: **5.02 ms/token**. Robust modes 0/1/2 are equal;
  changing completion mode is not a useful optimization.
- Physical 1M CP cache allocation: 12/12 ranks pass with 1.28 GiB/rank committed.

## Decode decomposition

Use one process per A64FX node and all 12 ranks as the expert group.

- Routed experts: expert `e` belongs to `e % 12`; each rank holds 24 experts.
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
2. BF16 KDA projection GEMVs and FP8 routed/shared-expert GEMVs.
3. Short-context head-TP forward with exactly two reductions/layer.
4. Expert scheduling sorted by local hit count to reduce rank skew.
5. Continuous batch after single-stream correctness; decode FP8 weights once per
   active expert and reuse them across tokens.
6. Add the long-context CP transition after the fast path is stable.

Every benchmark must report kernel time, collective time, arrival-wait time,
weight bytes/rank, and end-to-end tokens/s. A tight-loop collective number alone
is not an end-to-end communication claim.
