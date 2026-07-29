# Kimi K3 on A64FX

## Goal

Build an exact text-only Kimi K3 inference path for 96 A64FX nodes. The implementation will live in this directory and reuse the proven GLM-5.2 runner structure for uTofu bootstrap, robust collectives, rank-local staging, profiling, and generation.

The current interactive allocation has six 32 GB nodes, so development must use synthetic tests and carefully selected real-weight layers. A complete 96-node run is out of scope for the interactive validation stage.

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
python3 a64fx/k3/k3_sim.py
```

The current native test passes all primitives on 512-bit SVE. A representative run on
the interactive A64FX node measured 3.86 GB/s per core for the 8-row, K=3584 MXFP4
matvec and 0.279 GOP/s per core (352 microseconds) for one 128x128 KDA head step. These
are microkernel measurements, not full-node or end-to-end runner results. MXFP4 scaling
to 48 cores is an explicit simulator assumption; collective and dense-kernel constants
remain inherited calibration assumptions until the six-node runner measures them.

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

With the current default assumptions (336 GB/s dense bandwidth, 180 GB/s 48-core
MXFP4 bandwidth, measured single-core KDA, and 20 microseconds per recursive-doubling
collective step), the conservative 96-node decode estimates are 11.5 token/s at 4K,
10.8 token/s at 128K, and 7.4 token/s at 1M for batch 1. The 1M result is a modeled
compute/communication value only and is not runnable under the v1 cache layout.

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
- KDA is correctness-first and single-threaded in this harness. A persistent 48-thread
  value-row schedule is the next performance task.
- The simulator's collective latency, full-node bandwidth scaling, GEMM rate, routing
  distribution, and imbalance factor are surfaced assumptions, not measured K3 data.
- The GLM-5.2-derived stager, full layer graph, uTofu runner, and launch scripts remain
  to be implemented. No full checkpoint load or multi-node job was attempted here.
