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

## Performance simulator

`k3_sim.py` will be based on `a64fx/llm/ds4f_sim.py` and will derive model byte counts from the K3 manifest. It will model:

- Ragged expert ownership, tensor-parallel weights, KDA state, MLA cache, AttnRes scratch, and activation memory.
- Decode active-weight traffic, MXFP4/BF16 kernel rates, KDA work, MLA context scanning, routing imbalance, and collective latency.
- Prefill projection and expert GEMMs, linear KDA recurrence, quadratic MLA attention, chunk efficiency, and communication/synchronization.
- Measured lower bounds separately from calibrated predictions and unmeasured assumptions.

The default report will cover decode batches 1, 8, and 32 at 4K, 128K, and 1M context, plus prefill prompts of 1K, 8K, 128K, and 1M tokens with chunk sizes 64, 256, and 1024. Six-node kernel and collective measurements will replace inherited GLM/DS4F calibration constants where available.

Exact 1M-context runtime is not part of v1. The simulator will describe the required follow-up context-parallel MLA design: context-sharded cache, query/gate gathers, distributed attention statistics, flash-combine, and output reduction.

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
confirms that context-parallel MLA is required rather than optional.

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
layout; context-parallel MLA is required.

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
kernel uses a 512-column dequant tile below M=24 and 1,024 columns at M>=24; this keeps
M=16 from regressing while improving M=32. A 12-row decode experiment was unstable and
slower and was not retained. A persistent OpenMP team across the three expert stages
also regressed M=1/M=2 and was removed.

Measured on all twelve nodes at 48 threads after these changes:

| Real expert workload | Time/rate |
|---|---:|
| One expert, M=1 | median 0.176 ms (0.172--0.298 ms) |
| One expert, M=8 tiled | median 1.030 ms (1.025--1.034 ms) |
| One expert, M=16 tiled | median 1.863 ms (1.849--1.875 ms) |
| One expert, M=32 tiled | median 3.237 ms (3.217--3.258 ms), 9.89k assignments/s |
| Two distinct M=1 experts, shared workshare | median 0.334 ms (one 0.473 ms outlier) |
| Four distinct M=1 experts, CMG-partitioned | median 0.629 ms (0.625--0.632 ms) |

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
| 1 | 28.3 ms | 27.2 ms | 0.7 ms | 39.7 ms | 10.43 |
| 8 | 28.3 ms | 72.4 ms | 3.9 ms | 45.5 ms | 53.32 |
| 16 | 28.3 ms | 113.5 ms | 7.6 ms | 52.1 ms | 79.44 |
| 32 | 28.3 ms | 174.8 ms | 14.9 ms | 65.3 ms | 112.96 |

The production uTofu probe was also rerun on the twelve-node allocation. Exact FP32 SUM
and MAX checks passed. For 7,168 floats, flat recursive doubling takes 76.3 us and the
existing 3x4 hierarchical path takes 65.6 us (1.16x). At batch-sized payloads the gain
is 1.30x: 114,688 floats take 1.164/0.896 ms flat/hierarchical and 229,376 floats take
2.328/1.782 ms. `--hierarchical-ar` applies the measured payload-dependent speedup;
carrying that ratio to 96 nodes is an explicit extrapolation, not a 96-node result.

With hierarchical reduction extrapolated and the exact graph retained, the optimistic
projection is 11.06 token/s at M=1 and 119.42 aggregate token/s at M=32. Overlapping the
latent reduce with the TP-local shared expert only raises these to 11.16 and 119.76:
the whole shared-expert stack is 24.310 GB, only 0.253 GB per 96-node rank, so the model
caps hidden work at about 0.75 ms for the entire stack. There is no legitimate large
overlap window. The 12-node large-payload measurements also warn that the simulator's
8 GB/s wire lower bound is optimistic for M=32; 119.76 is therefore a ceiling-oriented
estimate, not a demonstrated 128 token/s result.

The current exact design does not meet either new target: single decode remains well
below 15 token/s and M=32 remains below 128 token/s even before applying the conservative
large-payload correction. Eliminating the latent collective exactly would require
replicating the 4.727 GB BF16 routed-up stack per rank (or a new 2-D placement); that
exceeds the 27 GB memory budget on the fullest ranks. Mode `--moe-collectives 0` means
*no MoE collective* (94 attention/dense calls), not one collective per MoE layer, and is
only an impossible upper bound: 14.35 token/s at M=1 and 132.3 at M=32. These are
projections from bounded real weights, not full-layer end-to-end validation.
