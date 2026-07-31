# Kimi K3 on A64FX

## Goal

Build an exact text-only Kimi K3 inference path for 96 A64FX nodes. The implementation will live in this directory and reuse the proven GLM-5.2 runner structure for uTofu bootstrap, robust collectives, rank-local staging, profiling, and generation.

The current interactive allocation has twelve 32 GB nodes, so development uses synthetic tests and carefully selected real-weight slices. A complete 96-node run is out of scope for the interactive validation stage.

## Checkpoint and architecture

The checkpoint at `~/models/kimi-k3` is a 1.56086 TB, 96-shard safetensors release. Its text model contains:

- 93 decoder layers: 69 Kimi Delta Attention (KDA) and 24 gated MLA layers.
- Hidden size 7168 and 96 attention heads.
- One dense FFN layer followed by 92 LatentMoE layers.
- 896 routed experts, with exactly 16 selected per token.
- LatentMoE width 3584, routed-expert width 3072, and two shared experts.
- SiTU-GLU activations and Attention Residuals with a 12-layer block size.
- Native MXFP4 E2M1 expert weights with E8M0 group-32 scales; other text tensors are BF16/F32.
- Vocabulary size 163840 and a nominal 1,048,576-token context window.

The vision tower and multimodal projector are not part of the first implementation.

## Numerical contract

The first runner will preserve the checkpoint graph:

- Native MXFP4 weights are consumed directly by W4A16 A64FX kernels.
- Activations remain BF16/F32, with FP32 reductions and recurrent state where required.
- Routing uses the full BF16 router, correction bias, sigmoid scores, exact top-16 selection, renormalization, and all 896 experts.
- No reduced-expert or approximate default is permitted.
- MLA follows the supplied NoPE graph and does not add an untrained rotary transform.
- KDA follows the official recurrent delta rule, including q/k L2 normalization, the lower-bounded decay gate, sigmoid beta, causal depthwise convolution, and sigmoid-gated RMSNorm output.

## 96-node decomposition

Routed experts account for about 1.446 TB of the checkpoint. With ownership `expert_id % 96`, ranks 0–31 own ten experts per MoE layer (about 16.14 GB total) and the remaining ranks own nine (about 14.53 GB).

The remaining roughly 114 GB cannot be replicated. The intended layout is:

- Attention is head tensor-parallel; at 96 nodes each rank owns one head. The output projection is reduced to a replicated 7168-wide hidden state.
- The router and LatentMoE down-projection are replicated to avoid another latency-sensitive collective.
- Routed expert outputs are accumulated in the 3584-wide latent space and all-reduced once per MoE layer.
- Latent-up and shared-expert output projections are tensor-parallel and folded into a second 7168-wide all-reduce.
- The dense layer splits its intermediate dimension and reduces its output.
- Embedding and LM-head vocabulary rows are sharded.
- KDA recurrent and convolution state exists only for locally owned heads.
- Exact MLA K/V exists only for locally owned heads; startup must reject contexts that do not fit safely in HBM.

This layout is expected to keep the busiest rank near 23 GB of weights before caches and scratch space. The simulator and runtime preflight must calculate the actual allocation rather than relying on that estimate.

## Modules to implement

The project will provide:

- Scalar references and A64FX/SVE kernels for RMSNorm, SiTU-GLU, short convolution, KDA recurrence, gated RMSNorm, MLA attention, AttnRes, and router top-k.
- Reuse of the existing split-layout MXFP4 matvec and batched GEMM kernels from the DS4F work.
- K3 configuration, tensor, layer, cache, model, and checked-arena types.
- Rank-local safetensor staging with tensor slicing and an aligned blob/manifest format.
- Token decode and chunked-prefill forward paths.
- A GLM-5.2-style uTofu EP/TP runner with synthetic and real-weight modes, durable per-rank logs, profiling, prompt-ID input, and greedy generation.
- A dependency-free helper for the K3 `tiktoken.model` and XTML text chat template. Prompt-ID files remain the canonical reproducible input.

Planned core interfaces are `k3_default_config`, `k3_load_real`, `k3_forward_token`, `k3_forward_prefill_chunk`, and `k3_free`.

## Memory-safe staging

The stager must never copy or read the entire checkpoint into memory. It will:

- Validate all 96 safetensor headers and required shapes without faulting tensor data.
- Keep only locally owned experts and tensor-parallel slices of dense tensors.
- Support layer ranges or explicit layer lists for partial tests.
- Copy in bounded chunks, periodically `fdatasync`, and drop source and destination page cache with `POSIX_FADV_DONTNEED`.
- Preflight blob, arena, cache, and scratch sizes before allocating.
- Abort partial runs before allocation if `MemAvailable` is below 6 GB.

## Six-node validation

The interactive validation will stage layers 0–3 only. This covers the dense KDA layer, KDA+MoE, MLA+MoE, shared experts, and the initial AttnRes block while keeping each rank well below the HBM limit.

Validation gates are:

1. Compare every SVE primitive with an independent scalar C implementation using deterministic inputs.
2. Confirm token-serial KDA and chunked KDA produce equivalent outputs and final recurrent state.
3. Compare optimized MLA with an expanded scalar attention reference, including cache continuation and output gating.
4. Decode real checkpoint MXFP4 expert tensors and compare matvec/GEMM results with scalar dequantization.
5. Run isolated real-weight layer 0, layer 1, and layer 3 tests on one rank and six ranks, comparing outputs within numerical tolerance.
6. Run a four-layer truncated prefill/decode smoke test and require deterministic routing, stable checksums, finite outputs, and no NaNs.

Truncated generation is not expected to produce meaningful text; its purpose is loader, graph, collective, and numerical validation.

## Interactive 12-node full-runner debug

The C11 full runner has three explicit modes. `full96` remains strict: it requires
96 MPI ranks and the complete rank-local image. `layer12` stages one checkpoint
layer, including all six tensors for experts owned by `expert_id % 12`, and runs
that real layer on the twelve physical ranks. `synthetic12` traverses all 93
layers with deterministic shape-preserving operators and replaces one selected
layer with its real staged image. Synthetic output is a structural/debug oracle,
not a model-quality result.

The direct interactive harness does not submit a job:

```sh
./run_k3_full_12n.sh --mode layer12 --layer-index 1 \
  --prefill-tokens 32 --new-tokens 0 --prefill-chunk 1
./run_k3_full_12n.sh --mode synthetic12 --layer-index 3 \
  --prefill-tokens 32 --new-tokens 16 --prefill-chunk 8
```

Layer indices are checkpoint indices `0..92`; existing `run_k3_ep.sh` keeps its
older one-based layer convention. Results are retained under
`logs/full-debug-12n-$PJM_JOBID`, while rank-local images remain under
`/local/$USER/k3-full-debug-12n-$PJM_JOBID`.

## Performance simulator

`k3_sim.py` will be based on `a64fx/llm/ds4f_sim.py` and will derive model byte counts from the K3 manifest. It will model:

- Ragged expert ownership, tensor-parallel weights, KDA state, MLA cache, AttnRes scratch, and activation memory.
- Decode active-weight traffic, MXFP4/BF16 kernel rates, KDA work, MLA context scanning, routing imbalance, and collective latency.
- Prefill projection and expert GEMMs, linear KDA recurrence, quadratic MLA attention, chunk efficiency, and communication/synchronization.
- Measured lower bounds separately from calibrated predictions and unmeasured assumptions.

The default report will cover decode batches 1, 8, and 32 at 4K, 128K, and 1M context, plus prefill prompts of 1K, 8K, 128K, and 1M tokens with chunk sizes 64, 256, and 1024. Six-node kernel and collective measurements will replace inherited GLM/DS4F calibration constants where available.

Exact 1M-context runtime is not part of v1. Context-parallel MLA can distribute the
scan work, but cannot reduce the cluster-average KV bytes once all 96 ranks participate.
The 1M design therefore also needs validated cache compression or lower weight memory;
query gathers, distributed attention statistics, flash-combine, and output reduction
remain relevant performance work after capacity is solved.

## Deliverables

- Native kernel and correctness tests.
- K3 stager and manifest validator.
- Partial real-weight test harness.
- 96-node EP/TP runner and launch script, prepared but not submitted.
- Decode/prefill memory and performance simulator.
- Measured six-node results and known limitations recorded in this directory.

## Implemented milestone: graph kernels and simulator

The first self-contained implementation unit is now present:

- `k3_kernels.h` defines the K3 dimensions and scalar/SVE paths for dot products,
  L2 normalization, RMSNorm, sigmoid-gated RMSNorm, SiTU-GLU, causal depthwise
  convolution state, the official lower-bounded KDA decay gate and recurrent delta
  update, stable online MLA attention, AttnRes mixing, and exact corrected top-k routing.
- `k3_kernel_test.c` exercises those paths with deterministic synthetic data and reuses
  `common/ggml_dequant.h` for the native split-layout MXFP4/E8M0 expert matvec.
- `k3_sim.py` is a Python-3.6-compatible, dependency-free network memory/decode/prefill
  model. It reads only the eight-byte prefix and JSON header of each safetensor shard;
  it never reads, maps, or copies tensor payloads.
- `k3_stage.py` validates the complete checkpoint using headers only and stages one
  explicitly selected rank-owned expert with bounded `pread` calls, 256-byte alignment,
  atomic output replacement, `fsync`, page-cache eviction, and a 6 GiB `MemAvailable`
  guard.
- `k3_real_mxfp4_test.c` reads only eight selected rows per matrix from a partial staged
  blob and compares the real checkpoint's `w1`, `w2`, and `w3` against scalar MXFP4
  dequantization.
- `Makefile` builds the test with Fujitsu clang and A64FX SVE.

Run the native checks and the 96-node report with:

```sh
make -C a64fx/k3 clean test
make -C a64fx/k3 validate
make -C a64fx/k3 real-test
make -C a64fx/k3 probe-kda
make -C a64fx/k3 probe-kda-mpi
python3 a64fx/k3/k3_sim.py
```

The current native test passes all primitives on 512-bit SVE. A representative run on
the interactive A64FX node measured 3.86 GB/s per core for the 8-row, K=3584 MXFP4
matvec. Hoisting KDA decay exponentiation out of the 128 value-row loop reduced one
synthetic 128x128 head step from roughly 348 to 23.5 microseconds on one core, about a
15x graph-level improvement. These are microkernel measurements, not full-node or
end-to-end runner results. MXFP4 scaling to 48 cores remains a simulator assumption;
collective constants remain inherited until the distributed runner measures them.

The manifest scan validates 497,220 tensor entries across 96 shards and assigns the
text checkpoint as follows (decimal GB):

| Placement | GB |
|---|---:|
| Routed experts, EP-sharded | 1446.456 |
| Latency-sensitive replicated tensors | 6.089 |
| Head/vocabulary/intermediate TP tensors | 107.513 |
| Total text weights | 1560.058 |

At 96 ranks the fullest rank is modeled at 23.35 GB of weights. With expanded BF16 MLA
K/V and a 27 GB usable budget, 4K context fits through batch 32, and 128K fits only at
batch 1. One-million-token context does not fit even at batch 1 (39.81 GB total), which
confirms that a cache/weight capacity reduction is required; context sharding alone
cannot change the average bytes per rank.

With the current default assumptions (336 GB/s large-matrix bandwidth, measured
131.8 GB/s cache-evicted head-slice bandwidth, 180 GB/s 48-core MXFP4 bandwidth,
measured eight-thread KDA, and 20 microseconds per recursive-doubling collective step),
the original bandwidth-average estimate was 14.17 token/s at 4K. The measured
expert-service model below supersedes it: random top-16 ownership collisions make the
current 4K batch-one estimate 10.06 token/s.
The 1M result remains compute-only and is not runnable under the v1 cache layout.

Known implementation limits of this milestone:

- The release checkpoint stores all 69 KDA `A_log` tensors as `[128]`, shared by
  key channel across heads, although the bundled Python constructor declares
  `[num_heads]` (`[96]`). The C kernel follows the actual checkpoint. The stager must
  validate `[128]` and keep this discrepancy visible until upstream clarifies it.
- The complete header validator passes: 96 shards, 497,220 tensors, 69 KDA layers,
  24 MLA layers, and all 82,432 layer/expert groups containing six exact-shape tensors.
- A bounded real-weight test staged layer 1/expert 0 (six tensors, 16.734 MiB) and tested
  rows 0--7 and 1024--1031. All `w1/w2/w3` SVE results passed; the largest absolute
  row-test error was `7.153e-7`. The composed full expert test also checks all 3,072
  `w1/w3` outputs, SiTU, and all 3,584 `w2` outputs; its output checksum was
  `-2.924505058e-01`, L2 norm `4.545992581`, with all values finite.
- Partial staging can be reproduced without reading unrelated payloads:

  ```sh
  python3 a64fx/k3/k3_stage.py --output-dir /tmp/k3-l1-e0 --layer 1 --expert 0
  a64fx/k3/k3_real_mxfp4_test /tmp/k3-l1-e0/layer01_expert000.blob \
    /tmp/k3-l1-e0/layer01_expert000.manifest
  ```

- The harness does not yet execute complete layer 0/1/3 graphs.
- Transcendental functions use the scalar libm accuracy contract. SVE currently speeds
  reductions and vector algebra; SiTU, convolution bookkeeping, and sigmoid evaluation
  still need tuned vector approximations or restructuring.
- The real KDA probe has a row-parallel recurrence, but it creates an OpenMP workshare
  per step. The runner should retain a persistent team and use 8 threads per local
  head; using all 48 threads on one 128-row recurrence is counterproductive.
- The simulator's collective latency, full-node bandwidth scaling, GEMM rate, routing
  distribution, and imbalance factor are surfaced assumptions, not measured K3 data.
- The GLM-5.2-derived stager, full layer graph, uTofu runner, and launch scripts remain
  to be implemented. No full checkpoint load or multi-node job was attempted here.

## Estimated performance on 64--96 A64FX nodes

The simulator's strict usable-memory limit is 27 GB/node. With the current placement,
64 nodes require 30.86 GB/node, 72 require 29.05 GB, and 80 require 27.29 GB. The first
configuration that fits a 4K, batch-one decode is 82 nodes. At 96 nodes, exactly one of
the 96 attention heads resides on each rank; below 96 the critical rank owns two heads,
which creates a visible performance step.

| Nodes | 4K memory/node | Fits | Measured-KDA decode estimate |
|---:|---:|:---:|---:|
| 64 | 30.86 GB | no | compute-only; not deployable |
| 72 | 29.05 GB | no | compute-only; not deployable |
| 80 | 27.29 GB | no, narrowly | compute-only; not deployable |
| 84 | 25.61 GB | yes | 10.19 token/s |
| 92 | 23.89 GB | yes | 10.32 token/s |
| 96 | 23.77 GB | yes | 10.43 token/s |

The KDA calibration now comes from real layer-0/head-0 activations on all twelve nodes:
8 threads are optimal at 6.67 microseconds/head-step and 14.73 GOP/s. Cache-evicted
one-head BF16 projections deliver a robust 131.8 GB/s mean with 24 threads. For batch-one 128K decode,
only 96 nodes fit the expanded BF16 MLA cache; the revised estimate is 13.10 token/s.

Projected prefill with chunk size 256 is:

| Nodes | 1K prompt | 8K prompt | 128K prompt |
|---:|---:|---:|---:|
| 84 | 111 token/s | 108 token/s | does not fit |
| 92 | 113 token/s | 110 token/s | does not fit |
| 96 | 122 token/s | 121 token/s | 96 token/s |

The projection uses 336 GB/s for large dense matrices, the measured 131.8 GB/s for
small head-TP attention slices, 180 GB/s full-node MXFP4 bandwidth, 1.25 TFLOP/s
BF16-equivalent GEMM per node, 20 microseconds per
recursive-doubling collective step, and a 1.20 routed-expert imbalance factor. The
modeled 96-node 1M prefill rate is about 37 token/s, but it cannot run with the v1 cache
layout; exact 1M additionally requires cache compression or lower weight residency.

## Real partial KDA probe

`k3_kda_stage.py` stages thirteen tensors for one real KDA head: local `q/k/v`, output
gate, latent decay projections, beta row, convolution weights, `dt_bias`, `A_log`, and
output norm. Layer 0/head 0 occupies only 8.802 MiB. The test executes the projection,
causal convolution with SiLU, q/k normalization, safe decay gate, sigmoid beta,
recurrent delta update, and gated RMSNorm. All twelve nodes produced the identical
checksum `+4.685916041e-05`, beta `0.526281`, and finite output.

Twelve-node mean scaling, with one MPI process per node, is:

| Kernel | Threads | Time | Rate | Efficiency vs 1 thread |
|---|---:|---:|---:|---:|
| Four BF16 projections, resident | 1 | 343.85 us | 21.35 GB/s | 100% |
| Four BF16 projections, resident | 24 | 20.52 us | 363.16 GB/s | 69.8% |
| Four BF16 projections, resident | 48 | 17.56 us | 418.73 GB/s | 40.8% |
| Four BF16 projections, cache-evicted | 16 | 60.29 us | 121.82 GB/s | -- |
| Four BF16 projections, cache-evicted | 24 | 55.68 us | 131.84 GB/s | -- |
| Four BF16 projections, cache-evicted | 48 | 59.59 us | 140.06 GB/s mean (outliers) | -- |
| KDA recurrence | 1 | 25.20 us | 3.90 GOP/s | 100% |
| KDA recurrence | 4 | 9.21 us | 10.68 GOP/s | 68.4% |
| KDA recurrence | 8 | 6.67 us | 14.73 GOP/s | 47.7% |
| KDA recurrence | 12 | 9.13 us | 10.76 GOP/s | 23.0% |
| KDA recurrence | 48 | 11.76 us | 8.38 GOP/s | 4.5% |

The earlier six-node allocation reached 6.53 microseconds at 12 threads while retaining
the same 25.3-microsecond single-thread result, demonstrating allocation-sensitive
OpenMP synchronization. The eight-thread calibration is the more conservative and
robust choice. Hoisting `exp(log_decay)` out of the value-row loop remains the dominant
optimization: it changed the synthetic single-core recurrence from about 348 to
23--26 microseconds.

## Real partial MoE dispatch and MXFP4 probe

`k3_moe.h` now contains exact top-k local bucketing, gather, weighted scatter-add,
single-expert MXFP4 execution, and a fused multi-expert scheduler. The batched SVE path
dequantizes each 8-row MXFP4 tile once to an 8 KiB BF16 pair-vector panel and reuses it
through a three-token microkernel. On real K3 weights the tiled path differs from the
decode `svtbl` reference by at most `1.42e-7`; scalar-batch and fused-dispatch checks are
bit exact.

All performance probes first read their bounded blobs into anonymous memory under
`MPOL_INTERLEAVE` over NUMA nodes 4--7 (`mask=0xf0`). The launchers set
`XOS_MMM_L_PAGING_POLICY=demand:demand:demand`; file-backed first-touch is not used for
reported rates. `run_moe_probe_mpi.sh` stages four distinct layer-1 experts per node
(66.94 MiB/node), runs all ranks, and requires an explicit PASS record from every rank.

The decode kernel now prefetches packed rows eight MXFP4 blocks ahead. The tiled batch
kernel uses a 512-column dequant tile below M=24 and 3,072 columns at M>=24; this keeps
M=16 from regressing while improving M=32. A 12-row decode experiment was unstable and
slower and was not retained. A persistent OpenMP team across the three expert stages
also regressed M=1/M=2 and was removed.

SiTU was the remaining local M=32 hotspot: scalar `tanhf`/`expf` cost about 0.37 ms
of a 3.25 ms expert. The default path now uses SVE FEXPA with residual correction and
one reciprocal refinement. It takes 0.02--0.03 ms. The scalar contract remains
available at compile time with `K3_SITU_FEXPA=0`. Across twelve nodes, comparison with
libm followed by the real MXFP4 down projection had maximum absolute error `9.62e-5`
and relative L2 `3.05e-4`. The 3,072-column tile differs from the `svtbl` reference by
at most `7.14e-7` at M=32.

Measured on all twelve nodes at 48 threads after these changes:

| Real expert workload | Time/rate |
|---|---:|
| One expert, M=1 | median 0.167 ms (0.163--0.169 ms) |
| One expert, M=8 tiled | median 0.946 ms (0.942--0.951 ms) |
| One expert, M=16 tiled | median 1.690 ms (1.678--1.701 ms) |
| One expert, M=32 tiled | median 2.692 ms (2.675--3.079 ms), 11.9k assignments/s normally |
| Two distinct M=1 experts, shared workshare | median 0.292 ms (0.279--0.295 ms) |
| Four distinct M=1 experts, CMG-partitioned | median 0.575 ms (0.557--0.582 ms) |
| Eight distinct M=1 experts, shared workshare | median 1.032 ms (1.008--1.036 ms) |
| Sixteen distinct M=1 experts, shared workshare | median 2.024 ms (1.997--2.027 ms) |

The exactly-four-expert sparse path pins one bucket per 12-core CMG subgroup. Two and
three active experts retain the global workshare because a two-way CMG split measured
slower. The four-way path improves the normal four-expert result from about 0.89 ms to
0.77 ms.

Reproduce the distributed bounded test inside an allocation with:

```sh
K3_KEEP_RESULTS=1 make -C a64fx/k3 probe-moe-mpi
```

## 4K decode target audit

The simulator now uses a deterministic routing Monte Carlo and the measured real-expert
service curve instead of dividing active expert bytes evenly over all ranks. This matters
at batch one: 16 experts mapped onto 96 owners collide often enough that the mean
slowest-rank MoE service is about 0.37 ms/layer, or 34.0 ms over 92 layers, rather than
the impossible bandwidth-average value of 1.8 ms for the whole stack.

`k3_dense.h` adds the runner-facing fused router plus routed-latent-down BF16 workshare.
All twelve nodes passed the real row check. At 47 workers its median is 226.4 us/layer
slice and 283.7 GB/s (223.4--231.3 us); 48 workers regress to 239.5 GB/s median. The
47-worker default deliberately reserves the highest cpuset core for uTofu progress.

For 96 nodes, 4K context, current exact placement (93 attention collectives, one dense
FFN collective, and latent plus hidden collectives for 92 MoE layers: 278 total), the
revised estimates are:

| Decode batch | Dense/attention | Routed experts | KDA + KV | Collectives | Aggregate token/s |
|---:|---:|---:|---:|---:|---:|
| 1 | 28.3 ms | 24.0 ms | 0.7 ms | 39.7 ms | 10.79 |
| 8 | 28.3 ms | 62.3 ms | 3.9 ms | 45.5 ms | 57.17 |
| 16 | 28.3 ms | 88.6 ms | 7.6 ms | 52.1 ms | 90.66 |
| 32 | 28.3 ms | 130.3 ms | 14.9 ms | 65.3 ms | 134.03 |

The production uTofu probe was also rerun on the twelve-node allocation. Exact FP32 SUM
and MAX checks passed. For 7,168 floats, flat recursive doubling takes 76.3 us and the
existing 3x4 hierarchical path takes 65.6 us (1.16x). At batch-sized payloads the gain
is 1.30x: 114,688 floats take 1.164/0.896 ms flat/hierarchical and 229,376 floats take
2.328/1.782 ms. `--hierarchical-ar` applies the measured payload-dependent speedup;
carrying that ratio to 96 nodes is an explicit extrapolation, not a 96-node result.

With hierarchical reduction extrapolated and the exact graph retained, plus the bounded
latent overlap, the optimistic projection is 11.57 token/s at M=1 and 143.71 aggregate
token/s at M=32. The latter now exceeds the 128 token/s target in the calibrated model.
The scheduler model is no longer a serialization assumption: bounded probes staged
16 distinct real experts per node (267.7 MiB/node), and all twelve nodes passed. Eight
and sixteen distinct experts cost 0.775x and 0.760x their serialized service; these
measured factors are interpolated by active critical-rank bucket count.

Overlapping the latent reduce with the TP-local shared expert remains a minor lever:
the whole shared-expert stack is 24.310 GB, only 0.253 GB per 96-node rank, so the model
caps hidden work at about 0.75 ms for the entire stack. There is no legitimate large
overlap window. The 12-node large-payload measurements also warn that the simulator's
8 GB/s wire lower bound is optimistic for M=32; 143.71 is therefore a ceiling-oriented
estimate, not a demonstrated 128 token/s result.

The current exact design still misses the single-decode target: 10.79 token/s with flat
collectives and 11.57 under the hierarchical/overlap projection, versus 15. Eliminating
the latent collective exactly would require
replicating the 4.727 GB BF16 routed-up stack per rank (or a new 2-D placement); that
exceeds the 27 GB memory budget on the fullest ranks. Mode `--moe-collectives 0` means
*no MoE collective* (94 attention/dense calls), not one collective per MoE layer, and is
only an impossible upper bound: 15.05 token/s at M=1 and 162.2 at M=32. These are
projections from bounded real weights, not full-layer end-to-end validation.

## Intermediate-TP MoE architecture

The decode MoE path now supports intermediate tensor parallelism instead of whole-expert
ownership. At 96 ranks, rank `r` owns channels `[32*r, 32*r+32)` from every expert:
the matching `w1/w3` rows and `w2` columns. Thirty-two channels are exactly one native
MXFP4 scale group. `k3_stage.py --expert-tp --tp-size 96 --tp-rank R` creates these
slices with bounded reads and without requantization.

`k3_expert_tp_forward_selected_mxfp4` schedules all 16 selected slices in one OpenMP
team. `k3_moe_pack_reduce` concatenates routed latent and shared hidden partials into
10,752 floats for one sum-allreduce. The BF16 or quality-gated Q8 completion then applies
routed RMSNorm, replicated routed-up, and the shared residual locally. This changes the
MoE graph from two collectives to one.

Twelve real 256-channel slices reconstructed layer-1/expert-0 with max absolute error
`6.054e-8` and relative L2 `2.196e-7`. The distributed slice probe passed on 12/12
nodes. With 16 real selected experts, the 12-way path measured 0.204--0.233 ms/layer.
A true 96-way 32-channel slice measured locally at 0.096 ms/layer, or 8.81 ms over 92
MoE layers. Run the 12-node check with `make -C a64fx/k3 probe-expert-tp-mpi`.

## Routed projection Q8 gate

`k3_dense.h` now has dynamic per-vector activation quantization, symmetric per-row
weights, and a 24-row SVE `sdot` matvec. The real routed-down matrix occupies 24.51 MiB.
Q8 down measured up to 140.2 GB/s; the mixed BF16-router/Q8-down stage measured about
0.231 ms, versus 0.227 ms for BF16 router+down. Q8 therefore saves memory but not time.

The real result had relative L2 `9.969e-3` and cosine `0.99995032`. It fails the chosen
relative-L2 limit of `5e-3`, so the probe reports `GATE-REJECT(BF16 fallback)` and router
Q8 remains disabled. Both BF16 and experimental Q8 routed-up completion APIs exist.
The separately staged real routed-up projection reached only 97.5 GB/s and also failed
the gate (relative L2 `9.432e-3`, cosine `0.99995553`). Replicated routed-up is therefore
the tighter Q8 bottleneck.

The simulator now models the placement directly via `--expert-tp`, `--fused-moe-ar`,
`--dense-q8`, and `--attention-rsag`. With measured expert (`0.096 ms/layer`) and Q8
(`140/97 GB/s` down/up) calibration, the full Q8 candidate estimates about 12.4 tok/s
at 4K, M=1, and
96 nodes. It does not meet the 15 tok/s gate. The former 20 tok/s projection depended
on unmeasured 240--336 GB/s Q8 and 0.045 ms expert rates and is retracted. M=32 remains
above 128 tok/s in the model, pending end-to-end large-payload collective measurement.

### Continued projection tuning

The routed-up sweep found 44 workers optimal for row-Q8: 166.6 GB/s, compared with
about 252 GB/s for row-major BF16. Group-64 and group-32 packed Q8 improved cosine but
still missed the relative-L2 gate (`7.170e-3` and `6.491e-3` respectively); neither was
consistently faster than row-Q8. Pair-vector BF16 was also slower than row-major BF16.

For KDA, fusing the independent q/k/v/g/f_a projections and using four-row tasks fixes
the 64-task load imbalance of the old eight-row schedule. Real head-0 cold-weight
bandwidth reaches 172.5 GB/s at 40 workers, up from the prior 131.8 GB/s calibration.
The TP=96 selected-expert sweep remains best at 48 workers; fusing routed `w2` weighting
into the output-row workshare reduces scratch and measures 0.094 ms/layer.

With these measured kernel rates, the quality-experimental Q8 architecture projects
about 14.5 tok/s at 4K M=1 and 96 nodes, and over 180 aggregate tok/s at M=32. It is
within roughly 2.5 ms/token of the 15 tok/s requirement, but cannot be promoted while
the Q8 projection gate fails. The exact BF16 path remains slower and slightly exceeds
the strict 27 GB working budget when routed-up is replicated.

Keeping router and routed-down in BF16 while quantizing only replicated routed-up is
the fastest mixed placement. A 16-row K-blocked Q8 layout raises routed-up to 172.9
GB/s at 44 workers, and the shared hidden residual is now added inside the same
workshare. It estimates 14.67 tok/s and 25.05 GB/rank at M=1. At
M=32 it reaches 185.6 aggregate tok/s but models 27.15 GB/rank, narrowly beyond the
strict 27 GB guard. Use `--q8-up-only`; this remains experimental because routed-up
alone still fails the projection-level relative-L2 gate. These rates used the former
wire-only attention-RSAG projection and are retracted by the real-sum measurement below.

A 12-node multi-TNI probe used the fused MoE payload size (21,504 bytes in BF16).
One TNI was fastest at 4.598 us/hop and 4.68 GB/s. Two through six TNIs regressed
monotonically to 4.702--4.980 us/hop, so same-peer payload striping is rejected. Any
further collective improvement must reduce synchronization depth or select better
topological peers; adding VCQs for byte striping will not close the remaining gap.

### 20 token/s stretch work

The K3-sized attention collective was measured separately on the current 12-node
allocation. The original wire-only probe reported 9.59 us for six-TNI scatter plus
gather versus 20.73 us for the tree, but it omitted the sum and used an unsafe trailing
sequence word. Real payload reads exposed incomplete/overwritten tail cache lines.
The corrected protocol uses ordered completion puts and disjoint scatter/gather landing
areas. Its SVE BF16 receiver sum takes 2.33 us and passes full numerical validation.
End-to-end scatter + sum + gather is 21.46 us versus 20.80 us for the tree (1.03x):
RSAG is rejected. `--attention-rsag` now models this measured regression rather than
the retracted 0.52 factor.

Several additional real routed-up and KDA probes delimit the useful kernel space:

- K-blocked Q8 does not help routed-down: 144.7 GB/s versus about 140 GB/s for the
  row-Q8 kernel, with the same 0.997% relative projection error.
- Fusing five Q8 KDA projections quantizes the activation only once, but SDOT is
  compute-limited at 46.9 GB/s. It takes 97.8 us versus 53--56 us for BF16 and has
  1.42% relative error, so it is rejected.
- Routed-up MXFP4 reaches 112 us/layer (457 GB/s BF16-equivalent), but 11.7%
  relative error and cosine 0.9932 make it unusable.
- Group-32 weight-only Q8 preserves the FP32 activation and improves error to
  0.539%, narrowly missing the 0.5% gate. Clipping the group scale makes accuracy
  worse. Group-16 weight-only Q8 has worst relative error 0.478% and minimum cosine
  0.9999886 across eight independent activations, passing the projection gate. Its
  best isolated real-weight result is 143.6 us at 47 workers and 223.6 GB/s of stored
  weights; the simulator conservatively uses 218 GB/s. `k3_moe_finish_reduce_q8w16`
  fuses the shared residual into the same workshare. Rerun only this calibration with
  `./k3_dense_probe --only q8w16 BLOB MANIFEST`; shared-node HBM contention
  caused slower outliers, so the modeled rate is an optimistic clean-run calibration.

The quality-gated simulator mode is `--q8w16-up`. With expert TP, fused MoE reduction,
and hierarchical tree collectives, it estimates 13.77 token/s at 4K M=1 and 180.3
aggregate token/s at M=32. M=1 fits at 25.64 GB/rank; M=32 is 27.74 GB and misses the
strict 27 GB guard. The M=1 breakdown is 40.3 ms weights, 8.6 ms experts, 0.7 ms
KV/KDA, and 23.0 ms communication. Reaching 20 token/s requires about 22.6 ms more
than the validated model and cannot be obtained from routed-up optimization alone;
it requires a genuinely lower-depth collective or cross-layer pipeline overlap.

Q8W16 is also quality-safe for routed-down: across eight independent activations the
real matrix has worst relative L2 0.478% and minimum cosine 0.9999886. The fused
BF16-router/Q8W16-down workshare reaches 207.5 us at 40 workers, or 216.7 GB/s of
mixed stored traffic. `--q8w16-dense` enables group-16 weight-only Q8 for both routed
projections while retaining the router in BF16. It estimates 14.11 token/s at 4K M=1
and 182.1 aggregate token/s at M=32. The smaller weights fit both cases: 23.87 GB/rank
at M=1 and 25.97 GB/rank at M=32. Its M=1 compute-only lower bound is 47.9 ms, or
20.9 token/s with free communication; the current 23.0 ms collective term must fall
to roughly 2 ms to reach 20 token/s without further kernel gains.

### 18 token/s attack

The first pass toward 18 token/s removes two runner/model discrepancies and one
TP=96 expert bottleneck without relaxing the Q8W16 quality gate.

The runner now accepts `--ar-groups A`.  Zero retains the flat tree; a divisor
greater than one factors `N=A*B` into contiguous row groups followed by a strided
column reduction.  Both communication regions come from the NUMA-aware 256-byte
aligned runner pool, and SUM/MAX use the same checked error path as the flat tree.
On the current 12-node 2x3x2 allocation, 8,192 exact 10,754-float reductions gave:

| Mapping | Flat | Hierarchical | Speedup |
|---:|---:|---:|---:|
| 2x6 | 112.7 us | 96.4 us | 1.17x |
| 3x4 | 112.7 us | 91.1 us | 1.24x |
| 4x3 | 112.9 us | 92.0 us | 1.23x |
| 6x2 | 112.7 us | 96.8 us | 1.16x |

The transport also overlaps local completion of a contiguous payload+trailer Put
with the reciprocal receive.  A 16,384-reduction rerun improved the flat fused
payload from 112.7 to 106.8 us and 3x4 from 91.1 to 89.0 us, with zero mismatches.
Separate 3x4 calibration is 1.16x at 3,584 floats, 1.19x at 7,168, and 1.24x at
10,752; the simulator uses those payload-dependent values.  Direct all-to-all
(74.6 to 113.0 us for 7,168 floats), BF16 transport (at most 1.6%), and lean
polling were measured and rejected.  The 96-node group factor remains a launch
parameter until a real 96-node topology sweep chooses it.

At TP=96, routed expert down previously horizontally reduced eight rows for each
of 16 experts and then combined 128 scalars.  The new SVE path applies routing
weights to vector accumulators and performs only eight final reductions.  Sixteen
real layer-1 32-channel slices match the unfused reference to `9.313e-10` maximum
absolute error.  In a same-node A/B build, the selected-expert layer falls from
0.081 to 0.062 ms (23.5%); the initial conservative simulator default was 0.065
ms and the stable-18 calibration below uses 0.063 ms.

The initial real Q8W16 reruns reached 232--237 GB/s for routed-up and 241 GB/s for the
BF16-router/Q8W16-down pair. Those write-eviction measurements and their 230/235
GB/s defaults are superseded by the stable calibration below. A
16-row tile regressed to 186 GB/s, while activation-Q8 SDOT regressed to 170 GB/s
and failed quality at 0.605% relative L2; neither rejected kernel remains enabled.

With fused expert TP, fused MoE reduction, hierarchical collectives, and Q8W16
dense projections, the revised 96-node 4K M=1 estimate is **15.32 token/s**:
36.3 ms weights, 6.0 ms experts, 0.9 ms KV/KDA, and 22.1 ms communication.
At a 10 us 96-node tree-step calibration it becomes **18.33 token/s**.  Therefore
18 token/s is not yet claimed: it requires the unavailable 96-node run to show
roughly 10 us/step or another reduction of the 186-collective stack.  The new
hierarchical path did pass 32,768 sequential real-weight layer steps on 12/12
ranks in 14.45 seconds, with bounded KDA state and no collective failure.

### Stable 15 token/s calibration

The earlier 225--235 GB/s Q8W16 calibration was artificially low. The dense
probe evicted cache by writing 192 MiB immediately before each timed call, so
outstanding writeback competed with the weight stream. The probe now performs a
read-only sweep of every cache line and consumes the reduction through a volatile
sink. This still makes the real weights cold without injecting traffic that the
runner does not generate. `--stable-reps N` reports both mean bandwidth and the
p95-latency bandwidth floor; `--threads N` isolates one worker count without
OpenMP team-resize or sweep-order effects.

Five independent 64-sample runs on the current 12-node interactive allocation
gave routed-up p95 floors of 330--392 GB/s at 44 workers in the clean runs, and
the mixed BF16-router/Q8W16-down path gave 288--334 GB/s at 47 workers. The mixed
task order interleaves one router group with four down groups, raising its median
mean throughput to 354 GB/s versus 340 GB/s for the concatenated schedule; median
p95 bandwidth was essentially tied near 319 GB/s. Both real projections retain
their eight-activation quality gates: worst relative L2 is 0.478% and minimum
cosine is 0.9999886. Occasional interactive-host scheduling stalls remain visible
and are not represented as kernel bandwidth; a production rank must be exclusively
pinned. Reproduce the selected measurements with:

```sh
OMP_DYNAMIC=false OMP_PROC_BIND=close OMP_PLACES=cores \
  ./k3_dense_probe --only q8w16 --threads 44 --stable-reps 64 BLOB MANIFEST
OMP_DYNAMIC=false OMP_PROC_BIND=close OMP_PLACES=cores \
  ./k3_dense_probe --only q8w16down --threads 47 --stable-reps 64 BLOB MANIFEST
```

The stable-15 calibration discounted those clean p95 results to 300 GB/s for
routed-up and 285 GB/s for the router/down pair; it is superseded by the
quality-gated router and overlap path below. With 96 nodes, expert TP, fused
MoE reduction, hierarchical collectives, and dense Q8W16, the 4K M=1 model is
**16.89 token/s**: 59.19 ms/token against the stable-15 budget of 66.67 ms, a
7.47 ms margin. M=32 is **224.98 aggregate token/s** and fits the modeled HBM
budget at 25.93 GB/rank. The Q8W16 finish path caps its compute team at 44 workers
so application cores remain available for communication progress.

This establishes a partial-real-weight kernel floor and a conservative full-model
estimate, not a measured 96-node end-to-end result. The 22.1 ms collective term
and full-layer scheduling still require validation in an exclusive 96-node job;
the simulator prints an explicit PASS/FAIL line via `--decode-target-tps` so future
measurements cannot silently weaken the stable target.

### Stable 18 token/s attack

Two changes close the remaining 3.64 ms gap from stable 15. First, the router is
no longer left in BF16. Group-16 Q8 narrowly fails on the real layer-1 router
(0.5298% relative L2), and scale clipping at 0.98/0.96 degrades it to 1.23%/2.36%.
A router-only group-8 layout passes eight independent activations at 0.4343%
relative L2 and 0.9999906 minimum cosine. Two adjacent eight-weight groups are
packed into one 16-lane SVE block with separate lower/upper scales, avoiding the
half-vector execution cost of a naive group-8 kernel. The fused stage assigns a
fixed router CMG share and runs group-8 router with group-16 routed-down. Five
64-sample real-weight runs at 47 workers gave p95 stored-bandwidth floors of
270--310 GB/s; the model uses 280 GB/s.

Second, the first 16 MiB of each replicated Q8W16 routed-up matrix is pulled into
L2 while its fused MoE reduction is in flight. A cache-line sampling kernel is
sufficient because an A64FX fill allocates the complete 256-byte line. Its real
partial-weight p95 is 64.01 us and 262.1 GB/s, below the measured 12-node MoE
collective window. The runner has a persistent asynchronous reduction worker,
uses AArch64 WFE/SEV when idle, and reserves application core 59 while 47 OpenMP
workers are confined to cores 12--58. Without explicit core ownership the helper
can alias an OpenMP worker and latency rises by orders of magnitude, so
`--prefetch-mib` rejects 48-worker launches and the launcher installs the safe
A64FX place list automatically.

The 12-node hierarchical runner completed 4,096 sequential overlapped layer
steps on all ranks with the staged real expert slices, identical checksums, and
no communication failure. The 16 MiB prefetch raised the dummy A/B collective
stage from 97.6 to 118.1 us; the real partial run measured 122.9 us. The simulator
therefore charges a conservative 22 us/layer launch penalty instead of assuming
free overlap. The partial runner intentionally uses a proxy weight window and
does not consume it as routed-up, so its total loop rate is not an end-to-end
speed measurement; it validates concurrent HBM fill, real expert execution,
uTofu progress, numerical lockstep, and long-run transport behavior.

With the measured 0.062 ms expert kernel represented as 0.063 ms, 300 GB/s
routed-up, 280 GB/s router/down, 248 GB/s prefetch, and the 22 us overlap charge,
the real 96-shard manifest model predicts **18.15 token/s** at 4K M=1:
55.09 ms/token against the 55.56 ms target budget, a **0.46 ms margin**. M=32 is
**234.31 aggregate token/s** and uses 25.63 GB/rank. Reproduce the overlap probe
inside the 12-node allocation with:

```sh
./run_k3_ep.sh --mode dummy --nodes 12 --layers 1 --tokens 4096 \
  --layer 1 --threads 47 --profile --ar-groups 3 --prefetch-mib 16 \
  --result-dir logs/attack18-overlap
```

This is a stable partial-runner result plus a full-model estimate. The 0.46 ms
margin is intentionally small: 18 token/s is not an end-to-end 96-node claim
until the real routed-up buffers replace the proxy window and the 96-node
collective calibration confirms the modeled 24.1 ms communication stack.

### 20 token/s attack

The decode transport now defaults to `--comm-robust 2`. It still drains the
uTofu MRQ at receive entry and completion, but invalidates the RDMA trailer cache
line once every eight spins instead of executing `dc civac; dsb sy` plus an MRQ
poll on every spin. `--comm-robust 1` retains the eager recovery path for A/B and
diagnosis. On the restarted 12-node 2x3x2 allocation, 8,192 hierarchical layer
steps reduced rank-maximum allreduce time from **124.8 us to 99.2 us** (1.258x),
with identical checksums. A 16,384-step flat run also passed on all ranks without
MRQ growth or transport failure. A later same-weight real A/B was stricter:
95.8 us versus 86.7 us, or **1.105x**. The simulator uses this smaller factor.

The polling choice is phase-specific. During concurrent HBM prefetch, the pinned
WFE/SEV worker benefits from eager trailer invalidation, so the runner temporarily
uses robust mode 1 only between async submit and completion and restores the
requested mode immediately afterward. With 47 compute workers, three hierarchy
groups, and a 16 MiB window, this hybrid path measured **112.1 us** versus
**101.6 us** without prefetch in the dummy run. The repeated real-weight path
measured 115.2--119.2 us versus an 86.7 us baseline, so the simulator uses a
33 us typical-run charge. A subsequent concurrency sweep selected 32 prefetch
workers: four repeated stages were 123.9, 128.4, 123.9, and 137.4 us, while 28
workers reached 216.0 us and 36 reached 169.8 us. Auto mode now uses 32 workers,
and the stable simulator uses the conservative **51 us** rank-max charge.
`--prefetch-threads N` permits explicit profiling overrides. Two alternatives
were measured and rejected:

- keeping communication and prefetch inside the existing OpenMP team cost
  139.7 us for 16 MiB because uTofu and HBM fills interfered;
- issuing L2 hints before the expert cost 84--123 us because the A64FX prefetch
  queue throttled before expert execution.

The bounded real-weight test restaged layer 1, experts 0--15 on all twelve nodes
and completed 4,096 hybrid steps. All 12 ranks passed, checksum disagreement was
at most `6.463e-08`, peak runner-pool use was 40.14 MiB/rank, and the real collective
stage measured 115.2--119.2 us. This is a real MXFP4 kernel/collective validation, not a
whole-network throughput measurement; the prefetch window remains a proxy until
the full runner exposes the next layer's Q8W16 routed-up allocation.

With the real 96-shard manifest, 0.063 ms selected-expert kernel, 300 GB/s
routed-up, 280 GB/s router/down, 248 GB/s cache-line fill, 1.105x lean decode
collectives, and the conservative 51 us hybrid charge, the model predicts
**55.66 ms/token = 17.97 token/s** for 4K M=1 on 96 nodes. The typical 33 us
charge gives about **18.5 token/s**. Memory is 23.53
GB/rank. The earlier 20.11 token/s number used the faster dummy-only 1.258x/13 us
pair and is retracted by the real A/B. The 20 token/s stretch remains open; it
requires another roughly 4 ms/token from router/down bandwidth or a 96-node
collective improvement, followed by an end-to-end full-layer run.

The follow-on real router/down sweep did not justify a higher bandwidth
assumption. Across five 64-sample trials, the existing 11-router/36-down worker
split had a 270.9 GB/s minimum p95 floor; 10/37, 14/33, and 16/31 bottomed at
259.3, 269.6, and 267.8 GB/s. A 16-row kernel that reused each activation SVE
load across two adjacent output groups was also rejected: its 16 vector
accumulators spilled and reduced the floor to 205.5--210.0 GB/s. Production
therefore retains the eight-row kernel and conservative 280 GB/s model value.
`k3_dense_probe --only q8pair` now isolates this quality/performance gate for
future work without running unrelated quantizers.

Compressing Q8 scales also failed to improve M=1. BF16 scales reduced the
routed-down layout from 0.625x to 0.5625x BF16 bytes but narrowly failed quality
(0.5046% relative L2). FP16 scales passed at 0.4778%, yet converting scales in
the inner loop raised p95 from roughly 151 to 161 us. Compressing only router
scales and retuning to 16 router workers recovered about 154 us but did not beat
the original kernel. Both compressed layouts were removed from production.

Windows larger than 16 MiB are likewise rejected: real 24 and 32 MiB runs raised
the collective stage to 212.4 and 245.0 us, more than the additional cached
weight bytes can repay. The stable runner therefore keeps a 16 MiB window.

The robust-2 trailer invalidation cadence is now exposed as
`--comm-poll-spins N` (a power of two in `[1,1024]`) instead of being buried in
the transport. A 12-node sweep initially favored 32
spins (93--99 us versus 105--108 us in clean 16K dummy trials), and 32 completed
65,536 consecutive hierarchical layer steps without MRQ growth or a transport
failure. A same-stage real-MXFP4 series, however, overlapped after system-wide
slow runs were excluded: 8-spin clean runs were 98--103 us and 32-spin runs were
97--114 us. After allocation restart, alternating real-weight 2K runs measured a
2,539 layer-step/s median with four spins and 2,503 with eight; every four-spin
run had exact rank agreement, as did a separate 4K trial. The default is therefore
four spins, while the option remains a 96-node tuning control rather than a claimed
full-model gain.

Two further overlap experiments were rejected. Splitting the expert OpenMP team
at the TP=96 local shape (32 channels/rank, 16 selected experts) raised the
selected-expert critical path from 77--86 us to 202--282 us for 4--16 prefetch
workers. Spreading the existing 32 prefetch workers across all four CMGs raised
the 12-node overlapped stage from 118.4 to 338.3 us because the extra HBM pressure
interfered with uTofu. Production retains the close-bound asynchronous path.

The runner's outer bootstrap/final barrier now has the same RDMA cache-coherency
envelope as the allreduce. Receive slots are cleaned before uTofu registration,
flag polls periodically invalidate the A64FX cache line, and MRQ notices are
drained at entry, progress, and completion. This closes a long-generation failure
mode in which every token could finish but the final barrier could spin on a stale
cached sequence value.

Reproduce the real partial validation in a 12-node allocation with:

```sh
./run_k3_ep.sh --mode real --nodes 12 --layers 1 --tokens 4096 \
  --layer 1 --experts 0-15 --threads 47 --fused-threads 47 \
  --profile --heartbeat-tokens 1024 --ar-groups 3 --comm-robust 2 \
  --prefetch-mib 16 --stage-dir /local/$USER/k3-runner-profile-$PJM_JOBID \
  --result-dir logs/attack20-real-hybrid-$PJM_JOBID

python3 k3_sim.py --nodes 96 --contexts 4096 --batches 1 \
  --expert-tp --fused-moe-ar --hierarchical-ar --q8w16-dense
```

### Stable decode handoff and TP96 prefill

The 4K M=1 decode configuration is now frozen as the accepted 18 token/s
engineering point. The conservative model remains 17.97 token/s (55.66 ms),
only 0.03 token/s or 0.10 ms outside the nominal value. `k3_sim.py` therefore
defaults to an 18 token/s target with an explicit 0.5% calibration tolerance;
it reports the signed latency margin and does not alter any kernel calibration.

Prefill no longer reuses the generic 1.25 TFLOP/s assumption for routed experts.
`k3_expert_tp_prefill_mxfp4` implements the actual TP96 routing shape:

- W1/W3 retain expert buckets, reusing each real MXFP4 slice across the tokens
  routed to that expert.
- Buckets below eight tokens read latent rows through compact dispatch indices;
  larger buckets use the existing lossless dequantize-once BF16 tile. This avoids
  copying every routed latent row for the common sparse-bucket case.
- Routed-down is fused by token and top-k while its eight outputs are still SVE
  accumulators. It never materializes or rereads the otherwise enormous
  `[chunk*16,3584]` expert-output tensor.
- Gather, W1/W3, SiTU, and routed-down execute in one persistent OpenMP team.
  Managed scratch is 0.016/0.063/0.254 GB at chunks 64/256/1024.

Sixteen real layer-1 TP96 slices were staged per rank in one safetensor-header
scan and aliased across a deterministic 896-expert routing population. All 12
ranks matched the per-route reference (`1.397e-9` maximum error) and passed at
all chunk sizes. Rank-maximum mean layer times, used by the simulator, were:

| Chunk | Active experts | Assignments | TP96 expert layer | Layer-local tokens/s |
|---:|---:|---:|---:|---:|
| 64 | 620 | 1,024 | 1.730 ms | 37.0k |
| 256 | 891 | 4,096 | 6.621 ms | 38.7k |
| 1,024 | 896 | 16,384 | 23.492 ms | 43.6k |

The indexed small-bucket path improves the same-node mean from 1.811 to 1.705 ms
at chunk 64 and 6.768 to 6.302 ms at chunk 256. A staged routed-down alternative
was rejected: materializing expert outputs made chunks 64/256/1024 2.09x/1.79x/
1.81x slower. Row-major and 64-token-tiled routed-down schedules were also slower
on the bounded real slices, so production retains token-major output locality.

With expert TP, fused MoE reduction, hierarchical collectives, and the accepted
decode layout, the revised prefill estimates are deliberately lower than the old
generic-GEMM projection:

| Prompt | Chunk 64 | Chunk 256 | Chunk 1,024 |
|---:|---:|---:|---:|
| 1K | 103.4 tok/s | 107.2 tok/s | 111.1 tok/s |
| 8K | 102.1 tok/s | 105.7 tok/s | 109.5 tok/s |
| 128K | 83.7 tok/s | 86.1 tok/s | 88.6 tok/s |

These are partial-real expert measurements plus modeled dense GEMM, MLA, and
96-node communication. Only 16 distinct slices are resident in the bounded probe;
the full 896-slice HBM stream and the assumed 1.25 TFLOP/s dense prefill GEMM remain
the next calibration gates. Reproduce the distributed expert test with:

```sh
K3_KEEP_RESULTS=1 ./run_expert_tp_probe_mpi.sh \
  --nodes 12 --experts 16 --threads 48 --prefill
```

### Dense BF16 prefill calibration

Dense prefill now has a real A64FX path rather than a simulator-only FLOP
assumption. `k3_prefill_gemm_bf16_pv` reuses the proven 8x48 SVE microkernel,
but owns the K3-specific execution contract:

- Real row-major K3 weights are packed directly into pair-interleaved Kx48
  panels. No full transpose is constructed.
- Packed weights and activation workspace are supplied by `k3_pool`; the GEMM
  performs no allocation and all buffers remain 256-byte aligned.
- Prefill is internally split into 256-token panels. This keeps the long-K
  packed activation panel within the four CMG L2 caches and bounds scratch,
  while callers may still submit larger chunks.
- M and N tails use a bounded local micro-tile and are checked against an FP64
  sampled reference before performance is reported.

The partial-real test stages the layer-1 router, routed-latent down, and
routed-latent up weights (110.25 MiB total). Correctness passed with sampled
relative L2 error below `1.7e-6`. Representative 48-core mean throughput is:

| Projection | M64 | M256 | M1024 |
|---|---:|---:|---:|
| Router 896x7168 | 1.60 TFLOP/s | 2.21 TFLOP/s | 2.10 TFLOP/s |
| Down 3584x7168 | 2.05 TFLOP/s | 4.93 TFLOP/s | 4.74 TFLOP/s |
| Up 7168x3584 | 2.98 TFLOP/s | 4.94 TFLOP/s | 4.78 TFLOP/s |

Without paneling, M1024 routed-down reached only 2.16 TFLOP/s. The 256-token
panel raises it to 4.74 TFLOP/s and reduces probe peak active memory from
243.47 to 208.47 MiB. The simulator uses a conservative 2.0 TFLOP/s dense
floor instead of the measured 4.7 TFLOP/s large-projection rate. MLA has a
separate 0.4375 TFLOP/s calibration, so dense improvements do not incorrectly
accelerate long-context attention. Revised 96-node estimates are:

| Prompt | Chunk 64 | Chunk 256 | Chunk 1,024 |
|---:|---:|---:|---:|
| 1K | 133.2 tok/s | 139.5 tok/s | 146.2 tok/s |
| 8K | 131.0 tok/s | 137.1 tok/s | 143.5 tok/s |
| 128K | 102.1 tok/s | 105.8 tok/s | 109.6 tok/s |

Reproduce the dense probe after safely staging one layer:

```sh
python3 k3_dense_stage.py --output-dir /local/u14346/k3-dense-prefill \
  --layer 1 --include-up
OMP_PROC_BIND=close OMP_PLACES=cores ./k3_prefill_probe --threads 48 \
  /local/u14346/k3-dense-prefill/layer01_dense.blob \
  /local/u14346/k3-dense-prefill/layer01_dense.manifest
```

## Runner runtime and command-line contract

The K3 dense, KDA, and MoE probes now allocate through `k3_pool` in
`k3_runtime.h`; runner code does not call raw `malloc`, `calloc`, or `free`.
Every returned address is 256-byte aligned. The pool applies anonymous-memory
interleave over NUMA nodes visible in the rank's CPU affinity, caches released blocks,
tracks active/reserved/peak bytes, and trims cached blocks before retrying a failed
allocation. Partial blobs are loaded with bounded `pread` calls and source pages are
dropped as they are consumed. Errors include the operation, requested and pool bytes,
`MemAvailable`, path, and failing file offset where applicable. Manifest counts,
required tensor names, and every tensor extent are validated before a kernel runs.

Operational settings are command-line arguments. Examples:

```sh
./k3_dense_probe --only q8w16 BLOB MANIFEST
./k3_moe_probe --threads 48 --tile-threshold 8 --tp-selected BLOB MANIFEST ...
./run_kda_probe.sh --model-dir "$HOME/models/kimi-k3" --layer 0 --head 0
./run_moe_probe_mpi.sh --nodes 12 --layer 1 --experts-per-rank 4 --threads 48
./run_expert_tp_probe_mpi.sh --nodes 12 --layer 1 --experts 16 --threads 48
```

The launchers reject missing/unknown arguments, invalid ranges, and pre-existing
result directories. Cleanup only removes directories created by that invocation.
Environment variables are limited to scheduler/rank discovery, OpenMP/XOS profiling
and binding controls, and the debugging retention switches `K3_KEEP_PROBE=1` and
`K3_KEEP_RESULTS=1`; they no longer select model, layer, kernel, or topology behavior.

## Distributed K3 runner

`k3_ep_runner` is the executable uTofu bring-up path for the intermediate-TP MoE
architecture. `--nodes N` is the physical allocation and `--tp-nodes T` is the ranks
assigned to each independent context; `T` must divide `N` and is at most 96. The
default is `min(N,96)`, so 72 and 96 nodes run one ragged TP72/TP96 context while 192
nodes run two independent TP96 contexts for aggregate prefill or batched decode. The
96 native 32-channel MXFP4 scale groups are distributed raggedly inside each context:
at TP72, 24 ranks own 64 channels and 48 own 32; at TP96 every rank owns 32 channels.
Attention heads use the same balanced rule. Physical-rank status files remain unique,
and each context has an independent progress file, collective, state, and synthetic
input stream.
The selected-expert SVE kernel shares one OpenMP team, route-weights the local `w2`
partials, packs the 3,584-float routed latent and 7,168-float hidden partial, and issues
one 10,752-float sum-allreduce per layer step. All ranks then advance the same residual
state. A final checksum reduction detects lost lockstep.

Both execution modes use the same kernels, scratch buffers, and collective:

- `--mode dummy` creates deterministic MXFP4 tensors through the 256-byte-aligned,
  NUMA-aware pool. It checks topology, registration, allreduce, computation, status
  reporting, and teardown without filesystem traffic.
- `--mode real` loads 16 rank-local intermediate-TP slices produced by `k3_stage.py`.
  `--experts 0-15` stages all selected experts with one scan of the 96 safetensor
  headers. At TP=96 this reads about 2.79 MiB/rank, or about 45 MiB per 16-node SIO
  group. Loading is collective-safe: one rank's failure is reduced to every rank,
  producing `load-failed` status rather than stranding peers in the next collective.
- `--ar-groups auto|A` controls the pool-backed hierarchical all-reduce. `auto` is
  the default and chooses six-rank rows: two contiguous groups on 12 nodes and 16 on
  TP96 ranks. An explicit divisor of `--tp-nodes` selects that many groups; `0`
  remains the diagnostic flat override.

Inside an allocation, use the orchestration wrapper:

```sh
./run_k3_ep.sh --mode dummy --nodes 96 --layers 2 --tokens 2 --ar-groups auto
./run_k3_ep.sh --mode real --nodes 96 --layer 1 --experts 0-15 \
  --model-dir "$HOME/models/kimi-k3" --layers 1 --tokens 2
./run_k3_ep.sh --mode real --nodes 192 --tp-nodes 96 --layer 1 \
  --experts 0-15 --model-dir "$HOME/models/kimi-k3" --layers 1 --tokens 256
```

The grouping path was validated on job `49868333` by splitting twelve physical ranks
into two TP6 contexts. Dummy and real-weight runs both passed 12/12 status checks;
each context completed 32 steps with zero rank disagreement and distinct final
checksums. A logical TP72 sweep then validated all 72 real slices: every M=2 prefill
reference passed at roughly `1e-7` relative L2, including all 24 64-channel ranks.
The generalized fused routed-down path measured 0.065 ms median on those 64-channel
ranks and 0.060 ms on the 32-channel ranks; their p95 values were 0.082 and 0.069 ms.
Prefill medians for M=64/256/1024 were 3.178/12.502/43.233 ms on 64-channel ranks and
1.762/6.779/24.282 ms on 32-channel ranks. These measured ragged-rank critical times,
not the old TP96 constants, must calibrate TP72 projections.

The wrapper builds the runner and topology helper, retries topology discovery, stages
only for real mode, captures per-rank output on the shared filesystem, and requires one
durable `state=pass` marker per rank. It refuses pre-existing result and rank-local
stage directories. Rank-local data is intentionally not recursively deleted by the
launcher; Fugaku wipes `/local` when the allocation ends, while retention during an
interactive job makes failures recoverable.

The auto hierarchy was selected from a fresh 12-node sweep. Flat reduction measured
4.912 ms per layer in the four-layer smoke graph. Six-rank rows plus a two-rank column
measured 0.108 ms/layer over 128 consecutive reductions with robust-2/poll-8. Robust-1
and robust-2 poll cadences 1/4/8 all passed 12/12 ranks with identical checksums. The
runner retains robust-2/poll-8 rather than selecting the slightly faster robust-1 path.
A 256-token longevity test then passed 12/12 ranks, published four heartbeats, completed
1,024 layer steps and 1,033 collectives, and kept MemAvailable above 29.8 GiB. The
launcher reports the resolved topology (`ar_groups=2` on this allocation) in `K3_RUN`.
The existing partial-real TP stage also completed 256 KDA/expert layer steps on all
12 ranks with `2.413e-7` checksum disagreement, 24.15 MiB peak pool use, and 0.590 ms
rank-maximum collective-plus-arrival-skew time per layer. Its first 64-token interval
was the slowest; cumulative throughput rose from 668 to 865 layer steps/s, confirming
that the short real-weight spike was startup/jitter rather than persistent transport
failure.

### Fixed-team accuracy, tail profiling, and recovery

Fujitsu OpenMP dynamic teams are now disabled both in `k3_ep_runner` and the launch
wrapper. This is an accuracy invariant: a 1,024-token partial-real sweep at fused-team
size 32 produced checksum `-3.565328441`, while the unfused reference and team sizes
16/24/40/48 all produced `+0.668299676`. With `OMP_DYNAMIC=false`, team 32 returns the
reference checksum exactly. The runner also creates its full OpenMP team before the
timed token loop. At team 32 this reduced worst observed KDA startup from 46.8 to
1.07 ms; the production team remains 48 because its clean 2,048-step run sustained
2,531 layer steps/s versus 2,071 at team 40.

`--profile` now reports `K3_PROFILE_MAX` with the worst single KDA, MLA, expert, pack,
collective, and residual occurrence in addition to rank-maximum averages. This keeps
rare scheduler/arrival tails visible instead of hiding them in the mean. Progress
files remain atomically renamed but are advisory and no longer `fsync` on the hot
path; final per-rank status remains fsync-durable.

The final agreement check now reduces FP32 checksum max/min bounds and compares them
to the local FP32 checksum. The previous FP64-to-FP32 comparison created a false
`4.254e-6` disagreement at a checksum near 443 even though every rank printed the
same value. Final `tokens_completed` and `last_layer` are distributed minima, so a
peer numeric failure produces one consistent recovery point. Injecting NaN at rank 3,
token 2 caused all 12 ranks to publish `numeric-failed`, `reason=non-finite`, and
`tokens_completed=2` without hanging.

The communication reliability modes are explicit:

- `--comm-ack 0` is the default clean-traffic path. A partial-real 2,048-step run
  measured 2,531 layer steps/s and 0.103 ms rank-maximum mean all-reduce time.
- `--comm-ack 1` enables bounded ACK/retransmit. With every seventh payload Put
  deliberately dropped, all 12 ranks completed 128 steps with `2.365e-10` checksum
  disagreement and a 5.061 ms worst recovered collective. On clean traffic ACK mode
  measured 2,228 layer steps/s, about 12% below the default, so it remains opt-in.
- `--comm-deterministic 1` exposes the fixed-root reduction for diagnostic accuracy
  comparisons; normal execution remains `0` because fixed OpenMP teams and the
  corrected verifier already preserve the observed rank agreement.

A final 4,096-token dummy run covered three KDA layers and one growing-context MLA
layer per token: 16,384 layer steps and 16,399 checked collectives passed on 12/12
ranks at 2,298 layer steps/s. The BF16 MLA cache reached 20 MiB/rank, pool peak was
45.15 MiB, checksum disagreement was exactly zero, and all latent/KDA/cache maxima
remained finite.

Short 96-node batch smoke scripts are `pjsub_k3_dummy_96n.sh` (five minutes, no model
I/O), `pjsub_k3_real_96n.sh` (ten minutes, bounded partial weights), and
`pjsub_k3_smoke_96n.sh` (ten minutes; dummy must pass before real staging begins). They use the
`small` resource group because `small-s2` has a scheduler-enforced minimum elapsed time
of 3,601 seconds. The PJM allocation remains 96-node-specific; within an allocation the
runner and wrapper node count are configurable.

The 12-node preflight on job `49849632` passed all bring-up gates:

- A representative dummy range `[2,5)` exercised two KDA layers, one MLA layer,
  three MoE collectives per token, and produced 12/12 pass markers with checksum
  disagreement `4.65e-11`.
- One dummy token traversed the full 93-layer schedule (69 KDA, 24 MLA, dense layer
  0 and 92 MoE layers) with 12/12 pass markers. Managed peak memory was 70.09 MiB/rank.
- The real layer-1 preflight staged all 16 experts in one header scan, read 22.31
  MiB/rank (about 268 MiB for the allocation's SIO group), and passed 12/12 ranks.
  Two layer steps took 71.26 ms total; managed peak memory was 23.20 MiB/rank and
  checksum disagreement was `4.25e-10`.

This is a distributed graph/loader/collective runner, not yet a quality-generation
claim: dummy mode uses synthetic attention projections, route weights, dense/shared
projection, embeddings, and head; bounded real mode replaces the 16 expert slices
only. Full checkpoint staging and tokenizer-driven generation remain the next runner
increment after the 96-node graph gate.

### Twelve-node runner tuning and profiling

The runner now accepts `--profile` and reports rank-maximum time for KDA, MLA,
expert execution, reduction packing, allreduce, and residual update. These separately
reduced maxima form an upper bound and can slightly exceed wall time; the output labels
them `measured_upper` rather than implying that they are additive critical-path time.
`--reuse-stage` skips storage I/O only after an MPI-wide check confirms that every
rank-local marker matches the requested node count, layer, and expert list. This is
useful for tuning repeatedly from one bounded real-weight stage:

```sh
./run_k3_ep.sh --mode real --nodes 12 --layer 1 --experts 0-15 \
  --layers 1 --tokens 64 --threads 48 --kda-threads 8 --profile \
  --reuse-stage --stage-dir /local/$USER/k3-runner-profile-$PJM_JOBID
```

At TP=12, eight KDA heads are local to each rank. The old outer-head workshare could
therefore occupy at most eight cores. The new exact SVE recurrence flattens independent
`[head][value-row]` work into one team. The deterministic eight-head kernel test is
bit exact against the old SVE result and measured 15.23 us versus 29.82 us, a 1.96x
isolated speedup. Runner tuning keeps 48 threads for the 16 selected MXFP4 slices but
uses an independent eight-thread KDA team; one team size is not optimal for both.

Job `49852817` reused a 22.31 MiB/rank real layer-1 stage and passed all 12 ranks for
64 layer steps. Two complete runs measured 69.42--69.75 ms, or 917.6--921.9 partial
layer-steps/s, with checksum disagreement `2.84e-7` and managed peak memory 23.78
MiB/rank. Rank-maximum phase time per layer was about 0.339 ms KDA, 0.601 ms selected
experts, 0.133 ms allreduce, and 0.015 ms residual update. A robust 48-thread single-team baseline was 595.1
layer-steps/s, so phase-specific team sizing improved this bounded workload by 1.55x.
The earlier two-step 28.1 layer-steps/s number was dominated by OpenMP and process
startup and should not be used as steady-state throughput.

MLA head parallelism has the same TP scaling problem: TP=12 leaves eight local heads,
and TP=96 leaves one. `k3_attention_heads_parallel_sve` divides the context into token
blocks, computes stable online-softmax triples independently, and combines them with
the exact log-sum-exp identity. Its three-head correctness test differs from serial SVE
by at most `2.8e-9`. The runner retains cheaper outer-head attention below 128 context
tokens and switches to token-block parallelism afterward. On the 12-node synthetic MLA
probe, a full 4,096-step context sweep averaged 0.190 ms of MLA work per layer step and
passed 12/12 ranks; the 64-step short-context path retained its prior 1.110 ms figure
(dominated by repeated OpenMP startup). The full dummy 93-layer schedule also passed
12/12 ranks after these changes.

The residual harness no longer applies scalar `tanhf` to every latent element. It now
uses an SVE residual add followed by RMS rescaling, which keeps the synthetic recurrent
state bounded while representing the graph's residual behavior more faithfully. The
runner owns all new MLA scratch through the 256-byte-aligned, NUMA-aware memory pool;
no raw runner allocation was introduced.

The next scheduling pass removes the OpenMP team-size transition between KDA and MoE
on KDA+MoE layers. Both kernels now expose orphaned workshares that can execute inside
one existing team: flattened KDA rows complete first, the synthetic attention partial
is projected, and the same workers immediately execute the three selected-expert
stages. Standalone wrappers retain the original APIs for tests and non-runner users.
`--fused-threads N` controls this team and defaults to `--threads` (48 on A64FX).
`--no-fused-team` retains the independently sized `--kda-threads` path as a diagnostic
and recovery fallback.

An A/B test over 512 real layer-1 steps on job `49852817` produced identical checksums
and passed all 12 ranks:

| Schedule | Layer-steps/s | KDA ms/layer | Expert ms/layer | Wall ms/layer |
|---|---:|---:|---:|---:|
| Separate 8-thread KDA + 48-thread expert teams | 1,496.0 | 0.139 | 0.418 | 0.668 |
| Fused 48-thread KDA + expert team | 1,853.1 | 0.188 | 0.245 | 0.540 |

The fused path is 1.24x faster end to end. Three additional 512-step runs measured
1,816--1,859 layer-steps/s at 48 fused threads; 40 threads measured
1,767--1,776, so the full 48-core team is the robust default despite its slightly
higher KDA phase time. A 16-token traversal of the complete 93-layer dummy schedule
passed 12/12 ranks at 2,365 layer-steps/s after the fusion. This optimization addresses
local scheduling only; the allreduce remains outside the OpenMP region and continues
to cost approximately 0.11--0.12 ms per partial layer step on 12 nodes.

The selected-expert workshare now fuses each eight-channel W1 group, matching W3
group, and SiTU activation into one task. This removes a full-team barrier and avoids
rereading the gate/up tile before W2 while retaining 512 independent tasks at TP=12.
The real 16-expert slice reference remained exact (`max_abs=0`). A 4K dummy sweep
reduced expert time from 0.2217 to 0.2142 ms/layer and increased total throughput from
2,416 to 2,426 layer-steps/s. Four 4K real-weight distributed runs passed 12/12 ranks;
the three non-outlier expert times were 0.2131--0.2199 ms/layer with identical final
checksums. One run experienced a simultaneous collective/system outlier and is not
used as a kernel-speed claim.

### Long-context stability and compact state

Runtime state is now indexed by compact layer-type slots rather than the decoder-layer
index. A full network allocates 69 recurrent KDA slots and 24 MLA cache slots; the old
layout incorrectly reserved MLA cache for all 93 layers. In the 16-token full-network
control this reduced managed peak memory from 83.69 to 60.91 MiB/rank while preserving
the checksum and passing all ranks. At 4K context the change avoids approximately
2.9 GiB of unnecessary FP32 cache allocation per TP=12 rank. A KDA-only range now
allocates no MLA cache, so `--tokens` accepts values through 1,048,576 for bounded
recurrent stress testing without an unrelated context allocation.

Before allocating recurrent state or K/V cache, every rank compares the exact byte
requirement with `MemAvailable` and retains a 6 GiB reserve. Failure is reduced across
all ranks before entering the token loop. A deliberately impossible 1M-token,
93-layer request required 245,794 MiB/rank and was rejected by all 12 ranks with
`alloc-failed`; no large allocation was attempted and no peer entered a collective
alone. `K3_HEALTH` reports compact slot counts, state/cache sizes, global latent maximum,
KDA-state maximum, and MLA-cache maximum after every successful run.

The parallel MLA combine now computes each block's log-sum-exp weight once and reuses
it for all 128 value dimensions. Previously it evaluated the same exponential 128
times per block. A double accumulator is used for the small final block sum before
rounding the output to FP32. The independent three-head correctness test remains within
`2.8e-9` of serial online attention. On 12 nodes, the 8K synthetic MLA phase improved
from 0.262 to 0.239 ms/step and end-to-end throughput rose from 1,589 to 1,665 partial
layer-steps/s. The optimized 16K run used 160 MiB cache/rank and passed 12/12 ranks at
1,327 layer-steps/s with `mla_cache_max=0.139` and stable latent RMS.

The runner now stores MLA K/V cache in BF16 by default, matching the modeled Kimi K3
activation/cache format while retaining FP32 accumulation and stable online-softmax
statistics. `--mla-cache-fp32` selects the former cache representation for numerical
diagnosis; `--mla-cache-bf16` is also accepted explicitly. The BF16 SVE dot product
unpacks vectors in registers and does not materialize an FP32 cache copy. Correctness
tests measured `1.9e-9` maximum difference between serial and token-block-parallel BF16
attention and `2.4e-5` between BF16-cache and FP32-cache outputs.

On job `49852817`, one synthetic MLA layer over a complete 8K context passed all 12
ranks with 40 MiB cache/rank versus 80 MiB for FP32. The MLA phase improved from
0.2393 to 0.2298 ms/step and end-to-end throughput from 1,665 to 1,703 layer-steps/s.
At 16K, cache fell from 160 to 80 MiB and throughput was effectively neutral
(1,327 FP32 versus 1,330 BF16 layer-steps/s), showing that expert and collective time
mask the saved cache traffic at that scale. The BF16 and FP32 16K checksums differed by
only `5.5e-5`. A 16-token traversal of the default BF16 full 93-layer schedule passed
12/12 ranks at 2,439 layer-steps/s, allocated exactly 69 KDA and 24 MLA state slots,
and reported 59.03 MiB peak managed memory. Commands used for the long-context A/B were:

```sh
a64fx/k3/run_k3_ep.sh --mode dummy --nodes 12 --layer 3 --layers 1 \
  --tokens 16384 --threads 48 --kda-threads 8 --mla-cache-bf16 --profile
a64fx/k3/run_k3_ep.sh --mode dummy --nodes 12 --layer 3 --layers 1 \
  --tokens 16384 --threads 48 --kda-threads 8 --mla-cache-fp32 --profile
```

### Exact 128K runner hardening

Online MLA now branches on the running maximum so exactly one exponential is evaluated
per cached token; the other online-softmax weight is mathematically one. Token-block
parallelism uses `ceil(threads/local_heads)` parts per head instead of assigning every
thread to every head. TP=12 therefore uses six parts for each of eight local heads,
while TP=96 retains 48 parts for its one local head. All serial/parallel, FP32/BF16,
and extreme-logit tests retain their previous tolerances. At 16K the BF16 MLA phase
fell from 0.3829 to 0.3535 ms/step and total throughput rose from 1,330 to 1,377
layer-steps/s. A strided interleaved K/V kernel is covered by correctness tests, but
was rejected for the runner: it measured 0.3711 versus 0.3535 ms at 16K and 0.6349
versus 0.6287 ms at 32K, with identical checksums.

`--heartbeat-tokens N` defaults to 1024. Rank 0 atomically publishes
`k3_progress.status` with completed tokens, elapsed rate, global latent/KDA/cache
maxima, minimum `MemAvailable`, maximum process RSS/HWM, and collective sequence.
Cache maxima are accumulated
while appending K/V, avoiding a growing-cache scan at every heartbeat. SIGINT,
SIGTERM, and non-finite state use two control floats appended to the existing fused
layer allreduce, so no per-token collective was added. Completion records now include
the reason, completed token count, last layer, and sequence. A rank-0 NaN injection
produced 12/12 `numeric-failed` records at sequence 380; SIGTERM delivered to one rank
produced 12/12 stopped records at sequence 1311 without stranding peers.

Managed-pool accounting is now supplemented by `/proc/self/status` RSS and high-water
telemetry. Every final per-rank status records RSS, HWM, and system `MemAvailable`;
heartbeats and `K3_HEALTH` reduce the process values to rank maxima. This makes libc,
OpenMP, and communication-library allocations visible in postmortems without replacing
the pool's exact active/peak accounting. On restarted job `49862159`, a bounded 64-step
dummy run passed 12/12 ranks with 25.46 MiB managed peak and 10.19 MiB reported process
HWM; the procfs figure is treated as OS telemetry, not an A64FX HBM ownership total.
The heartbeat also enforces `--min-available-mib` (2,048 MiB by default). If any node
crosses that floor, all ranks finish the current token, coordinate a stop, and publish
`state=stopped reason=memory-pressure` instead of letting one process disappear into
the OOM killer. Set the guard to zero only for controlled diagnostics. The existing
6 GiB state-allocation reserve remains the earlier, stricter admission check.
An injected 32 GiB floor on job `49862159` stopped 12/12 ranks after token one with the
expected status and no stranded peer. With the normal 2 GiB floor, the retained real
layer-1 weights completed 4,096 steps on 12/12 ranks at 2,456 layer-steps/s, exact rank
agreement, 0.2185 ms expert time, and 0.1145 ms collective time; minimum reported
`MemAvailable` remained 29,904 MiB.

The allreduce region is sized explicitly and allocated by the runner's 256-byte-aligned,
NUMA-aware pool. K3 passes an explicit robust communication configuration rather than
operational environment variables. Checked SUM/MAX entry points unwind uTofu errors
and timeouts to the runner, allowing `comm-failed` status publication; legacy callers
retain the original terminating wrappers.
With every debug Put dropped and a 200 ms profiling timeout, all 12 ranks unwound the
first readiness collective and atomically reported `comm-failed` at sequence 1; no
process called the legacy hard-exit path or waited for the production timeout.

Two long real-TNI tests completed on job `49852817`:

- Real layer-1 KDA+MoE ran 131,072 steps in 54.06 seconds (2,425 layer-steps/s),
  reached collective sequence 131,080, kept `kda_state_max=0.110`, and had rank
  disagreement `1.6e-7`.
- One exact BF16 MLA+MoE layer ran the complete 128K sequential sweep in 346.84
  seconds, reached sequence 131,084, used 640 MiB KV/rank, kept
  `mla_cache_max=0.1387`, and had rank disagreement `6.9e-6`. Its average MLA phase
  was 2.262 ms/step over the quadratic sweep.

The 128K sweep implies 148.3 GB/s effective per-rank exact-MLA scan throughput, now
modeled separately from dense-weight bandwidth by `k3_sim.py --mla-bw-gbps`. At 96
nodes the revised model reports 9.56 token/s for 128K M=1 versus 10.93 token/s at 4K.
A full TP=12 1M request is rejected collectively before allocation (122,914 MiB/rank
required). At 96 nodes, exact 1M BF16 KV plus the fullest weight shard is still modeled
at 39.81 GB/rank; context sharding alone does not reduce this average, so 1M remains
out of scope pending a validated capacity reduction.

The real layer-1 KDA+MoE path also completed 65,536 recurrent steps on all 12 ranks in
27.12 seconds (2,416 partial layer-steps/s). Latent RMS remained 7.4833,
`kda_state_max` was 0.107 versus 0.110 at 8K, and checksum disagreement was below
`1e-8`, showing no slow recurrent-state growth in this bounded test. These stress runs
still use synthetic attention inputs and only real MXFP4 expert slices; they validate
state evolution, cache indexing, collectives, and failure handling rather than
full-model generation quality.

### Recovery-marker hardening

Rank completion records and rank-local stage-ready records are now published through a
same-directory temporary file followed by an atomic rename. Runner status additionally
flushes and `fsync`s the complete record before rename; stage markers use `sync -f`
after the Python stager has already atomically published and fsynced every blob and
manifest. A killed process can therefore leave either no marker or an ignored temporary
file, but cannot leave a prefix containing `state=pass` that the wrapper accepts.

Retained-stage validation compares the complete expected line, including rank, node
count, layer, and exact expert-list spelling. The former substring check could accept
`experts=0-15` for a request of `experts=0-1`; the 12-node negative test now rejects
that case with exit code 4, while exact `0-15` reuse passes all ranks. Pass counting
likewise requires the complete space-delimited `state=pass` field. An adversarial
257-token MLA test with logits large enough to force softmax underflow remains finite
and is bit exact between serial and parallel online attention.

Real-weight loading additionally binds all six manifest tensor names to the requested
layer and expert instead of accepting suffixes alone. It requires U8 payloads,
256-byte-aligned non-overlapping extents, exact containment, and an exact final blob
size before constructing matrix views. The retained 16-expert layer-1 stage passes the
new checks on all 12 ranks. Topology parsing now rejects coordinates outside the
stored byte range and duplicate physical coordinates before uTofu peer construction.

Newly staged expert blobs use the `K3EXPERTV2` manifest header with a CRC32 over the
complete aligned blob. The stager computes it through bounded reads of its already
local temporary file, then drops those pages; it does not reread checkpoint payloads
from shared storage. The runner verifies CRC32 before exposing matrix views. Legacy V1
stages remain readable with one warning per rank so an interactive allocation is not
invalidated unexpectedly, but must be restaged before production use.

A fresh 12-node layer-1 stage read 22.31 MiB/rank and verified all 16 checksums before
passing decode on 12/12 ranks. Changing only rank 0's test-manifest CRC produced the
expected mismatch, reduced readiness to 11/12, wrote no pass markers, and terminated
collectively without stranding the other ranks. The test manifest was restored after
the negative test.

Each rank now builds a new stage beneath a uniquely named same-filesystem temporary
root. Only after every expert, V2 manifest, and ready marker is durable does the rank
atomically rename the complete directory to the requested stage path. An interrupted
attempt may leave a clearly named `.tmp.rank...` diagnostic directory, but it cannot
poison the final path or satisfy `--reuse-stage`; an interactive retry can select the
same final path safely. A one-expert TP=12 publication test left one complete final
directory and no temporary root.

Queued 96-node smoke job `49852287` never started and produced no runner logs. It was
held by the scheduler with `RSCGRP STOP`, then explicitly canceled on 2026-07-30. The
replacement 10-minute small-group smoke job `49860807` ran on 2026-07-30 at 14:51.
All 96 ranks completed the two-token, three-layer dummy graph with finite health,
identical printed checksums, 4.06 MiB peak pool use, and the resolved 6x16 hierarchy.
The then-current SUM/96 checksum verifier nevertheless produced rank-dependent
rounding and falsely marked every rank `numeric-failed`; the dummy gate correctly
prevented the real stage. The max/min FP32 verifier in commit `958defee` directly
fixes this observed 96-node failure. A replacement run has not been submitted.

All figures in this section are partial-runner measurements: real mode supplies real
MXFP4 expert slices but still uses synthetic attention projections and omits the full
dense/shared completion, embedding, tokenizer, and LM head. They are useful for kernel
scheduling and collective diagnosis, not a full-model token/s claim.

### Combined TP96 decode and prefill calibration job

`pjsub_k3_profile_96n.sh` requests 96 small-group nodes for one hour and stages the
real layer-1 expert slices exactly once. It first runs an eight-step dummy transport
gate, then a 256-step real-weight decode profile, reuses the same rank-local slices for
M=64/256/1024 expert-prefill probes, and feeds the critical-rank measurements into the
whole-network simulator. The modeled workload shapes are a 1,024-token C++ codegen
context and an 8,192-token C++ code-analysis prompt. Prompt text and the important
caveat that activations remain deterministic/synthetic are saved in `workloads.txt`.

Both `run_k3_ep.sh` and the outer batch script write tab-separated stage timing files.
They preserve start/end epochs, elapsed seconds, and return codes for build, topology,
weight staging, decode, prefill, result validation, and the whole-network estimate.
An EXIT trap records an interrupted active stage, making the logs suitable for sizing
later checkpoint-heavy allocations even when a job hits its elapsed-time limit.

### TP72 non-contiguous decode probe

`pjsub_k3_probe_72n.sh` is the first crowded-system fallback probe. It requests the
`small` group for one hour with scalar `node=72` placement rather than a torus shape,
allowing a non-contiguous allocation. The named `small-s4` group is disabled for this
project and must not be requested explicitly. The job runs an eight-token dummy
transport gate followed by a 256-token partial-real layer-1 decode profile. Automatic
hierarchy selection resolves to twelve six-rank groups.

TP72 uses the runner's balanced ragged ownership: ranks 0--23 own 64 expert channels
and two attention heads, while ranks 24--71 own 32 channels and one head. Routed-down
decode and prefill fuse any number of native 32-channel groups, so the job also runs
real M=64/256/1024 expert-prefill calibration on both rank shapes. The current
whole-model simulator reports about 28.9 GiB on the fullest TP72 rank, above the
strict 27 GiB target; this allocation calibrates transport and partial-real kernels
rather than claiming an end-to-end TP72 serving configuration.

Submit from the repository root with:

```sh
pjsub --no-check-directory a64fx/k3/pjsub_k3_probe_72n.sh
```

Results are retained under `a64fx/k3/logs/probe-72n-$PJM_JOBID`. Success requires
72/72 pass markers from decode and prefill, the expected 12x6 hierarchy, all 256 real
steps, and `K3_PROFILE`/`K3_PROFILE_MAX` output in `summary.txt`.

### HTTP and llmgr control

The generic `a64fx/llmgr` HTTP supervisor now has a `k3` adapter and a dedicated
`pjsub_llmgr_k3_96n.sh` wrapper. It exposes authenticated build, true stage-only,
bounded partial-run, stop, log, stage-status, and fapp-profile operations. The K3
launcher honors llmgr's unique `MPIEXEC_OF_PROC` prefix so rank logs are collected
and detached MPI processes remain recoverable. Its rank-local stage survives runner
restarts, allowing a source edit through `/bash`, followed by stop, rebuild, and a
new bounded run without repeating checkpoint I/O.

This is a control/measurement HTTP interface, not an OpenAI-compatible completion
server. The partial K3 runner still lacks tokenizer, embedding, complete dense/shared
execution, and LM head, so llmgr advertises `supports_serve=false` and does not
pretend its synthetic latent steps are generated text. Running machine code is not
hot-patched: applying a live fix means a coordinated stop and restart against the
retained stage.

The authenticated HTTP path was exercised on job `49862159`: a stage-only request
validated the retained real slice on all 12 ranks and exited zero. A subsequent
65,536-step managed child was stopped after 7,481 steps; every rank completed the
coordinated `signal-term` path, no runner/`mpiexec`/`plexec` process survived, and a
fresh two-step distributed launch passed immediately. This specifically validates the
edit/build/stop/restart loop needed for later TP96 development allocations.

### Logical TP96 estimate on 12 physical nodes

`run_expert_tp_probe_mpi.sh --logical-tp 96 --logical-waves 8` maps logical rank
`physical_rank + 12*wave` and therefore samples all 96 distinct group-32 weight
slices using the 12-node allocation. Each wave stages only its bounded TP slice,
runs the real selected-expert or prefill kernel, then advances to the next logical
rank. This is stronger than repeating one weight eight times because it captures
the complete checkpoint-dependent critical-rank distribution.

On job `49862159`, all 96/96 decode samples and all 96/96 prefill samples passed
their local numerical references. Decode selected-expert latency had 0.061 ms
median, 0.062 ms p95, and 0.063 ms maximum. Real expert-prefill results were:

| M | median | p95 | maximum |
|---:|---:|---:|---:|
| 64 | 1.724 ms | 1.783 ms | 2.356 ms |
| 256 | 6.611 ms | 6.742 ms | 7.327 ms |
| 1,024 | 23.491 ms | 23.929 ms | 25.818 ms |

The p95 values are now the default `--expert-prefill-ms` simulator calibration.
With the measured 0.062 ms decode expert p95, the 96-node whole-network estimate is
18.10 tok/s at 1K context and 18.00 tok/s at 4K. An 8K prompt estimates 129.7,
136.3, and 142.7 tok/s for chunks 64, 256, and 1,024. Substituting the isolated
0.070 ms cold decode sample and maximum prefill values gives a conservative
17.86/17.76 tok/s decode and 117.2/132.5/139.4 tok/s prefill range.

Communication remains the extrapolated part. An eight-layer, 4,096-token stress run
completed 32,768 real K3-payload hierarchical reductions on 12 nodes at 0.1183 ms
per layer and exact rank agreement. The simulator then applies its 96-node hierarchy
model, producing 24.7 ms communication per decode stack. Twelve nodes cannot reproduce
96-node link contention, failure probability, or physical topology; these figures are
planning estimates until queued job `49863795` runs, not a substitute for its result.

Reproduce both real-weight sweeps with:

```sh
K3_KEEP_RESULTS=1 a64fx/k3/run_expert_tp_probe_mpi.sh --nodes 12 \
  --logical-tp 96 --logical-waves 8 --layer 1 --experts 16 --threads 48 \
  --result-dir a64fx/k3/logs/logical96-decode-$PJM_JOBID
K3_KEEP_RESULTS=1 a64fx/k3/run_expert_tp_probe_mpi.sh --nodes 12 \
  --logical-tp 96 --logical-waves 8 --layer 1 --experts 16 --threads 48 \
  --prefill --result-dir a64fx/k3/logs/logical96-prefill-$PJM_JOBID
```
