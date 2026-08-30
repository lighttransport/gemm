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

Eliminating empty remainder OpenMP regions in the KDA `mv`/`mv3` projection
helpers is exact and improves the 16-token profiled target run from 59.812 to
58.657 ms/token (16.719 to 17.048 tok/s); attention falls from 25.761 to
24.689 ms/token.

The same build improves the 8-cycle MTP-3 run from 9.634 to **9.683 delivered
tok/s** (alpha 0.4167, 26 delivered tokens, PASS); target/verify phases are
66.022/247.587 ms per cycle.

Routed verifier scratch (`up`, activation, and per-route output) is now
persistent per MoE context instead of allocated per layer/cycle. Exact batch
verification remains 92/92 PASS, and MTP-3 rises to **9.728 delivered tok/s**
(alpha 0.4167; verify 245.824 ms/cycle).

Runtime context allocation tests establish **256K as the minimum-safe target
and 512K as the preferred maximum** for the current 32 GiB/rank layout. The
planner's theoretical 1M estimate above does not satisfy the runtime 2 GiB
headroom guard once the complete integrated graph and working buffers are
resident, so 1M is not a supported launch configuration in this implementation.

FP8 8-row matvec prefetching (next input vector plus two weight rows) is
arithmetic-neutral and produced the best sustained result so far: 16-token
target decode **17.654 tok/s** (56.643 ms/token, final token 432, PASS), with
attention 23.394 ms and FFN 21.061 ms. MTP-3 improved to **9.933 delivered
tok/s**, alpha 0.4167, 26/26 committed tokens, PASS; target/verify phases are
64.298/240.950 ms per cycle.

With the same build, MTP-2 is the better throughput/quality point: **11.904
delivered tok/s**, alpha 0.625 (10/16 draft matches), target/verify phases
64.498/190.441 ms per cycle, final token 3669, PASS. Keep `drafts=2` for the
current best speculative-decode setting; the runner still accepts explicit
draft counts for experiments.

On the dedicated 12-node allocation, `OMP_WAIT_POLICY=active` improves MTP-2
slightly to **11.948 delivered tok/s** (target 64.265 ms, verify 189.649 ms per
cycle; alpha 0.625, final token 3669, PASS) versus 11.904 tok/s with the default
wait policy. Use active waiting when CPU isolation is guaranteed.

A longer 16-cycle MTP-2 run confirms stability: **12.337 delivered tok/s**,
23/32 draft matches (alpha 0.71875), final token 16320, PASS. Mean cycle phases
are target 66.202 ms, draft 9.580 ms, verify 192.990 ms, and rebase 10.219 ms.

For sustained deployment, the 16-cycle MTP-1 run is faster: **12.749 delivered
tok/s**, 10/16 matches (alpha 0.625), final token 40591, PASS. Its phases are
target 65.354 ms, draft 5.486 ms, verify 128.563 ms, and rebase 6.867 ms.
Use the runner's default `drafts=1` for maximum delivered throughput; MTP-2 is
useful when measuring multi-token verification scaling.

A 32-token sustained scalar run with the same two-row/c+64 prefetch and
48-thread close binding reaches **18.094 tok/s** (55.268 ms/token, final token
25, PASS). Profile maxima are mHC 9.698 ms, attention 23.990 ms, FFN 21.018 ms,
and head 1.185 ms per token.

The next 12-node rerun is job **51094320** (6-hour interactive allocation,
ports offset by +10). Its target routed stage is being rebuilt under
`/local/glm53f-target-routed-51094320`; the MPI stage is deliberately allowed
to finish before launching decode benchmarks. Two arithmetic-neutral MLA
overhead reductions are in the current tree: remove the unused `vacc` buffer
and keep the per-call query/logit scratch on the stack (`6f596b70`, `b6907db6`).
They must be rechecked against greedy-exact output and the 16/32-token target
profiles after staging.

The first foreground staging attempt was interrupted by the bash-over-HTTP
30-minute request limit (SIGINT at 16.4 GiB, with no rank status files). The
replacement stage is detached under the same allocation so MPI can run to
completion independently of bridge request lifetime. The integrated build
script now passes explicit `-I.` and `-I../../common` paths for Fujitsu
compiler local-header lookup; the corrected build completed successfully.

Fresh job-51094320 validation completed the target stage on all 12 ranks. The
target batch gate remains exact (`probe=92/92 PASS`, matching logits) and reports
5-token batch speedup 1.356x. A 32-token profiled scalar decode reaches
**18.140 tok/s** (55.128 ms/token, final token 25, PASS); profile is embed 0.287,
mHC 10.113, attention 24.216, FFN 20.785, and head 1.168 ms/token. This is
effectively unchanged from the prior 18.094 tok/s result, so attention and mHC
remain the primary optimization targets. MTP staging has been launched next
under the same 12-node allocation; its quality/performance result is pending.

The follow-on MTP stage completed all 24 rank checks. Standard 16-cycle
speculative validation is exact and stable (`accepted=10/16`, alpha 0.625,
final token 40591, PASS), delivering **12.803 tok/s**. Mean phases are target
64.863 ms, draft 5.532 ms, verify 128.106 ms, and rebase 6.898 ms per cycle.
This is a small improvement over the previous 12.749 tok/s baseline; verify
plus target latency still prevents the 30+ tok/s target, so further work should
focus on attention/KDA and verification batching rather than draft quality.

The same allocation's 2-draft comparison is also exact and stable: **23/32
accepted (alpha 0.71875), final token 16320, PASS**, at **12.482 tok/s**. Its
mean phases are target 65.567 ms, draft 9.468 ms, verify 190.468 ms, and rebase
10.251 ms. Thus 2 drafts improve agreement but reduce delivered throughput;
retain MTP-1 as the deployment default until verification batching is optimized.

A controlled 36-thread scalar rerun on job 51094320 is slower: **17.003 tok/s**
(58.813 ms/token, greedy PASS) versus 48-thread close binding at 18.140
tok/s. Its profile shifts FFN to 26.923 ms/token
(versus 20.785 ms at 48 threads), confirming 48 threads/close binding as the
current target configuration.

A detailed 12-node KDA layer-44 run (`GLM53F_KDA_DETAIL=1`) is exact and
stable. The slowest rank takes 1.266 ms/layer: 0.518 ms local graph, 0.066 ms
output projection, and **0.934 ms all-reduce** (rank variation is 0.30--0.52 ms
for the local graph). This identifies collective latency/overlap as the next
KDA optimization target; further standalone projection micro-optimizations are
unlikely to move end-to-end decode materially.

An optional `GLM53F_FAST_MATH=1` build was tested on job 51094320. The build
completed, but the 12-node 32-token target run failed the lockstep stability
gate: after about six minutes one rank remained active without synchronized
token output (the baseline completes in about 3.5 minutes). The isolated run
was terminated and fast-math is rejected for deployment; the default strict
floating-point build remains required.

The real-weight sparse layer-43 benchmark is also exact (`BIT_EXACT PASS`):
1.437 ms/layer at 128 cached tokens, comprising 0.460 ms indexer/front,
0.221 ms MLA, 0.105 ms output projection, and **0.800 ms all-reduce**. Thus
the sparse-attention side independently confirms that communication, rather
than MLA arithmetic, is the limiting component.

An opt-in `GLM53F_SPLIT_AR=1` prototype (MPI reduce-scatter plus allgatherv)
was tested on real KDA weights. It remains numerically exact (`finite=YES
PASS`) and trims the nominal reduction portion from 0.934 to 0.928 ms, but
two-collective overhead raises total layer time from 1.266 to 1.288 ms. It is
therefore rejected for deployment and left disabled by default.

The allocation-specific 12-rank ToFu topology was regenerated before testing
the existing uTofu collective path.  With `GLM53F_UTOFU=1`, the real-weight
layer-44 callback is exact (`rel_l2=0`, `state=BIT_EXACT`, `PASS`); five-token
batch latency is 1.861 ms and the measured all-reduce portion is 0.252 ms,
versus 0.934 ms through MPI on the same layer.  The standalone ToFu diagnostic
measures a 35.18 us warm 16 KiB reduction floor (12 ranks), confirming that the
earlier initialization failures were caused by stale/incomplete topology files,
not by the collective implementation.  An integrated target decode using this
path is running on job 51094320; its greedy-exact and end-to-end timing result
must be recorded before enabling ToFu by default.

Scalar (4096-float) callback measurements further isolate the benefit: ToFu
reduces the layer-44 all-reduce from 1.163 ms (MPI) to 0.271 ms while remaining
bit-exact.  A paired 16-token integrated run is also greedy-exact (`final_token`
432): ToFu measured 15.637 tok/s and MPI 15.581 tok/s.  This small 0.36% delta
is within run variance, so ToFu remains an explicit opt-in pending repeated
long-run measurements; the strict MPI path remains the deployment baseline.

An opt-in 2-D ToFu hierarchy (`GLM53F_UTOFU_2D=1`, two groups of six ranks)
was then tested. It is exact and lowers isolated scalar KDA reduction to
0.180 ms, but the five-token callback reduction is unchanged at 0.248 ms and
the integrated 16-token target is slower at **15.374 tok/s** (final token 432).
The extra row/column synchronization outweighs the microbenchmark win, so the
2-D mode remains disabled and the flat MPI path remains the deployment default.

The existing direct all-to-all ToFu option (`TP_AR_A2A=1`) was also checked.
It is exact (`PASS`) and reduces the isolated scalar KDA reduction to 0.151 ms,
but five-token verification remains 0.255 ms (flat ToFu 0.248 ms).  Because
the full target run is dominated by verification-sized and non-KDA collectives,
this option is retained for scalar experiments only and is not enabled by
default without a paired end-to-end win.

The paired integrated all-to-all target run is exact (`final_token=432`) but
slower at **13.609 tok/s** for 16 tokens, versus 15.637 tok/s for flat ToFu and
15.581 tok/s for MPI. The additional peer puts and cache traffic dominate in
the full graph; `TP_AR_A2A` is therefore rejected for deployment.

ToFu's robustness overhead was isolated with `TP_AR_ROBUST=0`: scalar KDA
all-reduce falls to 0.171 ms (exact), but five-token verification remains
0.269 ms versus 0.248 ms for flat robust-ToFu. A guarded integrated run is in
progress; this mode is only suitable when the allocation is dedicated and
long-run MRQ-overflow stability is demonstrated.

The guarded integrated non-robust run remained exact (`final_token=432`) but
measured only **14.369 tok/s** for 16 tokens, slower than robust flat ToFu
(15.637 tok/s) and MPI (15.581 tok/s). It is rejected for deployment; the
MRQ-drain behavior remains enabled whenever ToFu is selected.

BF16-compressed ToFu reduction (`TP_AR_BF16=1`) also preserved the 16-token
greedy result (`final_token=432 PASS`) but measured **15.122 tok/s**, slower than
FP32 flat ToFu (15.637 tok/s). The layer-44 output statistics differ at the
fourth decimal place, so BF16 reduction is rejected for deployment despite its
lower payload size.

An opt-in in-place MPI reduction (`GLM53F_MPI_INPLACE=1`) is bit-exact and
reduces scalar layer-44 all-reduce from 1.163 to 0.971 ms. Its integrated
16-token target run reaches **15.888 tok/s** (`final_token=432 PASS`) versus
15.581 tok/s for the out-of-place MPI control. The five-token callback path is
slower (0.304 ms reduction), so retain the switch for scalar decode only until
longer mixed scalar/batch validation is complete.

The longer 32-token in-place run remained greedy-exact (`final_token=25 PASS`)
but measured **15.273 tok/s**. This does not beat the established strict
baseline (17.443--18.140 tok/s), so the in-place reduction remains an opt-in
diagnostic rather than a deployment default despite its isolated scalar-layer
benefit.

An MLA head-dimension parallelization experiment (commit `58dee7e5`) preserved
the exact token stream but regressed the 12-node target to **17.624 tok/s**
(56.742 ms/token); attention rose to 25.084 ms/token. The additional OpenMP
regions and working-set effects outweigh the extra parallelism for 5--6 local
heads, so the experiment is reverted and the 18.140 tok/s implementation stays
as the performance baseline.

After restoring the strict build, a repeat 32-token run remained greedy-exact
(`final_token=25`, PASS) but measured **17.443 tok/s** (57.331 ms/token;
attention 25.739 ms, FFN 21.592 ms). The token stream is identical to the
18.140 tok/s run, so this is retained as a run-to-run A64FX variance datapoint,
not a replacement for the established best result.

The optional `GLM53F_NO_MATH_ERRNO=1` build was repeated: it remained
greedy-exact (`final_token=25`, PASS) but measured **17.752 tok/s** (56.330
ms/token), versus 18.191 tok/s on the first run. The spread matches the
observed A64FX run variance, so the flag is not claimed as a reliable gain and
is left disabled by default.

The KDA output-projection row-batching experiment (8-row SVE kernel replacing
4,096 one-row OpenMP iterations) is bit-exact and improves the isolated scalar
callback to 1.003 ms, but the integrated 16-token target measures **15.391
tok/s** (`final_token=432 PASS`). It therefore does not beat the current
baseline and remains experimental.

The opt-in `GLM53F_MHC_FUSED=1` implementation (commit `369d986c`) fuses the
mHC RMS reduction and 24-row BF16 projection into one OpenMP team, removing a
fork/join from each scalar mHC pre-step. The A64FX callback and full target
binaries compile successfully; the callback remains bit-exact, but an
end-to-end decode comparison is pending completion of the fresh allocation's
node-local expert staging. The validated default (`GLM53F_MHC_FUSED=0`) is
unchanged until that gate reports a wall-time improvement.

The first full 12-node comparison is complete: baseline target decode measured
**14.529 tok/s** (`final_token=432 PASS`) and the fused build measured **14.435
tok/s** (`final_token=432 PASS`) for the same 16-token workload. The fused
region is therefore ~0.65% slower in this run and remains opt-in/diagnostic;
the strict default is retained.

The opt-in `GLM53F_MHC_POST_FLOAT=1` post-mix path is bit-exact in both short
and stability runs: 16 tokens measured **14.987 tok/s** (`final_token=432
PASS`) and 32 tokens measured **15.756 tok/s** (`final_token=25 PASS`). The
16-token result is 3.2% above the same-allocation strict 16-token baseline
(14.529 tok/s). A same-allocation strict 32-token control is still pending;
the float path remains opt-in until that control is recorded.

The same-allocation 32-token strict control then measured **15.694 tok/s**
(`final_token=25 PASS`) versus **15.756 tok/s** (`final_token=25 PASS`) for
the float post-mix build, a modest **0.4%** gain. Both streams are exact; the
float path remains opt-in because this delta is close to observed run variance.
