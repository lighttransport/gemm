# GLM-5.3F A64FX 12-node decode-first plan

## Measured anchors (job 51040571)

- Full 34-layer/64-head KDA recurrent update, 136 MiB replicated test state:
  **1.483 ms/token** best at 24 threads. The production head-TP layout owns only
  5--6 heads/rank. The measured six-head/rank shape is **0.047 ms best /
  0.049 ms mean** at 48 threads, confirming KDA state arithmetic is not the
  primary limiter.
- Production uTofu all-reduce, 12 ranks, 4096 f32 values: **105.9 us**.
- 78 back-to-back reductions: **5.02 ms/token**. Robust modes 0/1/2 are equal;
  changing completion mode is not a useful optimization.
- Physical 1M CP cache allocation: 12/12 ranks pass with 1.28 GiB/rank committed.
- Conservative checkpoint planner (2 GiB scratch + 1 GiB OS reserve) fits all
  requested contexts: **28.986 GiB/rank at 256K** (3.014 GiB headroom),
  **29.273 GiB at 512K** (2.727 GiB), and **29.845 GiB at 1M** (2.155 GiB).
- Real staged F8_E4M3 quarter-expert (1024x4096 gate/up + 4096x512 down),
  anonymous resident weights: **0.438 ms/task mean** at 24 threads with the
  exact gather-free decoder, versus 0.477 ms/task with the LUT-gather decoder.
  A four-task/four-CMG kernel reduces the measured batch critical path to
  **1.13 ms mean / 0.424 ms best** while the background full-model stager is
  active; the distributed steady-state result below supersedes this isolated
  estimate.
- Full 42-layer, 12-rank four-way expert decode with 23.631 GiB anonymous
  weights/rank and one real 4096-float MPI combine/layer: **47.39 tok/s** over
  200 tokens (**21.104 ms/token**). Compute is 12.615 ms/token; combine plus
  arrival wait is 8.617 ms/token. An unloaded MPI baseline is 73.2 us/call,
  or 3.075 ms/token, leaving **5.56 ms/token of rank-arrival skew**. Resident
  MemAvailable is 6.0--6.45 GiB/rank.
- Aligned 12-way routed-only decode reaches **55.778 tok/s** over 200 tokens:
  17.928 ms/token wall, 14.114 ms compute, and 0.803 ms arrival skew. This is
  17.7% faster than the original four-way routed-only result.
- The checkpoint shared expert is 2048-wide (not the early synthetic 171-wide
  assumption). Block-TP across 12 ranks gives 128/256-wide shards with 63.0 /
  126.0 MiB weights/rank. The real 42-layer persistent-team stream costs
  **2.833 ms/token** for a narrow shard and **4.234 ms/token** for a wide shard.
  Add its partial hidden output to the routed partial before the existing MLP
  combine; it does not require another collective.
- Adding the otherwise-unloaded attention hidden-vector reduction to the matched
  aligned routed+shared run gives **45.788 tok/s (21.840 ms/token)** over 200
  tokens. Expert/shared compute is 15.758 ms/token, the MLP combine is 4.906 ms,
  and the added attention combine is **2.209 ms**. The measured 84-call bare-wire
  contribution is 6.122 ms/token and total arrival overhead is only 0.993 ms.
  This communication-inclusive number is the current decode ceiling; it still
  excludes attention projection GEMVs, KDA/DSA math, router, norms, mHC, and the
  final vocabulary projection.
- The checkpoint contains no E4M3 NaN payloads (a 0.1 GiB staged sample also
  measured 0.00077% zeros and 0.01071% subnormals). Removing the redundant NaN
  compares/select from the exact inner-loop decoder reduces expert/shared
  compute from 15.758 to **14.49--14.57 ms/token**. Two matched 200-token runs
  deliver **48.513 and 47.023 tok/s**, both with checksum `1.41924829e-05`.
  The stager now rejects `0x7f/0xff`, making this a checked payload contract.
  The INT8 SDOT alternative is not a single-token win: 4096x4096 SDOT measured
  434 Gop/s versus 688 Gop/s for W8A16. Its 64-token register-blocked kernel is
  2.37x faster, so retain it as a speculative/batched verification candidate.

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
- Quantization-aligned 12-way slicing is the decode default. Partition
  the 16 FP8 intermediate block rows, not 2048 raw elements: each expert has
  eight 128-wide and four 256-wide rank shards. This preserves the compressed
  128x128 scale grid, gives every rank all eight routed tasks, and reduces the
  simulated slowest-rank work from 16.244 to **12.061 block rows/layer**
  (p95/p99 14/14). With the real shared shard fused into the same MLP combine,
  **12-way delivers 50.776 tok/s (19.694 ms/token)** versus **44.466 tok/s
  (22.489 ms/token)** for the matched four-way control, a 14.2% gain. Compute
  is 15.763 vs 14.096 ms/token, but arrival skew falls from 5.881 to 1.781 ms.
  Both layouts produce the identical benchmark checksum `1.41924829e-05`.
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

## 100 tok/s assessment

Strict single-token decode cannot reach 100 tok/s with the current FP8 graph:
84 measured hidden-vector reductions already cost 6.12 ms/token, and the
routed/shared path alone costs about 14.5 ms/token. Perfectly overlapping those
two terms still caps this partial graph near 69 tok/s before attention
projections, router/norm/mHC, and the vocabulary head. A credible 100+ delivered
token/s target therefore requires multi-token/speculative verification (where
the measured INT8 register-blocked kernel amortizes activation quantization),
or a lower-bit expert representation plus fewer/overlapped collectives. It is
not a scheduler-only target.

### Real layer-45 MTP/speculative probe

Layer 45 is a complete independent sparse-attention+MoE draft block with 288
routed experts, a shared expert, router, `eh_proj [4096,8192]`, and the shared
vocabulary head. It is not merely an auxiliary logits head. The layer-selectable
stager and distributed benchmark measured its real 12-way routed/shared expert
path over 5,000 drafts:

- 0.564 GiB weights/rank;
- **0.524 ms/draft** wall;
- 0.369 ms expert/shared compute;
- 0.138 ms MLP combine and 0.049 ms unloaded attention combine;
- 1,908 partial drafts/s.

This was a lower bound on draft cost at the time of the partial probe. The
token-correct integrated runner described below supersedes it; sparse attention,
`eh_proj`, normalization, vocabulary projection, cache update, greedy sampling,
target verification, and rollback are now connected.

Cold-HBM A64FX measurements put MXFP4 expert throughput at 102--155 GB/s versus
148--174 GB/s for FP8 magic. Accounting for half-sized MXFP4 weights gives about
a **1.45x**, not 2x, effective expert speedup. `glm53f_spec_ceiling.py` combines
that result with measured routed/shared, wire, and partial MTP costs. It makes
deliberately impossible-best assumptions: all shared weights are read once per
verification batch, one collective sequence serves the whole batch, and every
unimplemented graph operation costs zero. Even then:

- K=1: 84.2 tok/s at impossible-perfect alpha=1.0;
- K=2: 95.6 tok/s at alpha=1.0;
- K=3: 102.5 tok/s only at alpha=1.0, but 88.2 tok/s at alpha=0.9;
- K=3 at alpha=0.8: 75.7 tok/s.

Therefore 100+ single-stream delivered tok/s is rejected for this 12-node
FP8/MXFP4+MTP design unless real chained acceptance is effectively perfect and
the omitted graph work is somehow free. The practical 100+ route is continuous
batching across independent requests; MTP may still improve latency/throughput,
but must first be evaluated in a token-correct full forward runner.

### MTP numerical stability and quality gate

Two independent 10,000-step, 12-rank layer-45 runs completed without NaNs,
OOM, or collective failures. Both produced the bit-identical checksum
`-0.000212714513`; expert compute was 0.362 ms/draft in both runs and wall
throughput was 2,005.5 / 1,956.7 partial drafts/s. Rank-0 MemAvailable stayed
near 29.05 GiB. The stager's full-payload scan rejects E4M3 `0x7f/0xff`, and all
12 routed/shared stages completed that contract.

`glm53f_mtp_expert_check.c` validates a real staged rank-0 layer-45 expert shard
against the scalar FP8 reference. The 48-thread SVE path is finite and passes at
`max_abs=4.002e-11`, `rel_l2=3.618e-7` for the full gate/up/SiLU/down operation.

This establishes payload and expert-kernel numerical stability. It remains a
useful isolated check, but the integrated measurements below are the quality
gate for speculative decode.

### Token-correct integrated target/MTP runner (job 51077354)

The 12-rank runner now connects the real embedding, 45 target layers, 12-way
KDA and sparse-attention projections/state, mHC, routed and shared experts,
normalization, shared vocabulary head, and the independent layer-45 MTP graph.
Verification snapshots contain every KDA recurrent/convolution state and every
sparse-attention cache length. Rejection restores the selected target snapshot;
the committed MTP suffix is replayed from exact target hidden states.

Correctness and stability evidence:

- scalar versus five-position batched KDA output has relative L2
  `6.93943709e-08`; all captured recurrent and convolution snapshots are
  bit-exact;
- a sustained 128-token target run produces the same token trajectory through
  final token `271` before and after the decode optimizations;
- G1 after a 128-token warmup repeatedly produces the identical 16-cycle
  acceptance pattern, `10/16` accepted drafts (`alpha=0.625000`), 42 delivered
  tokens, and final token `40591`;
- target plus MTP weights and capacity-177 caches leave 3.70 GiB
  `MemAvailable` on rank 0, with no NaNs, OOM, or collective failure.

The validated KDA changes fuse Q/K/V projection launches, parallelize the three
depthwise convolution channel sets, and parallelize independent local-head
normalization/decay and gated RMSNorm. On the controlled 128-token target test,
latency improved from 68.500 to **62.413 ms/token**, or **14.599 to 16.022
tok/s** (+9.75%), with the token trajectory unchanged. The best matching G1
run is **11.936 delivered tok/s** with unchanged acceptance and final token.

The optimized scalar target profile is 25.696 ms attention, 22.579 ms FFN,
12.596 ms mHC, 1.178 ms vocabulary head, and 0.373 ms embedding per position
(component maxima are reduced independently across ranks and need not sum to
the end-to-end maximum). This makes attention/FFN the next optimization targets;
MTP remains latency-negative at the measured alpha because target verification
dominates.

Latest 12-node Fugaku reruns (job 51086028) measured 17.21 tok/s unprofiled
(58.107 ms/token) and 16.72 tok/s with profiling enabled. A cross-token routed
expert scheduler restored exact logits (92/92 probes) but reduced batch speedup
to 1.25x and MTP-3 to 8.86 tok/s; it is retained only as infrastructure. The
performance path uses one full OpenMP team per token and measures MTP-3 at 9.63
tok/s, alpha 0.4167, PASS. `OMP_PROC_BIND=spread` was tested at 15.62 tok/s and
is slower than the default close binding.

Runtime context allocation tests establish **256K as the minimum-safe target
and 512K as the preferred maximum** for the current 32 GiB/rank layout. The
planner's theoretical 1M estimate above does not satisfy the runtime 2 GiB
headroom guard once the complete integrated graph and working buffers are
resident, so 1M is not a supported launch configuration in this implementation.
