# Qwen3.8-27B Q8 on one A64FX node

## Measured baseline

Measurements used the second node (`a35-1110s`) of job 50639255 so the Codex
process did not consume the model node's HBM. Both model runs used 48 cores,
`max_seq=8`, one prompt token, four measured decode tokens, and no NextN draft.

| Path | Load | Decode | Notes |
| --- | ---: | ---: | --- |
| Q4_K_XL anonymous, NUMA distributed | 609.6 s | 0.164 tok/s | 24.398 s / 4 tokens; 16.7 GB resident weights |
| Q8_0 lazy mmap from `/local` | 0.53 s | 0.020 tok/s | 204.607 s / 4 tokens; repeated file faults/storage traffic |
| Q8_0 full anonymous | OOM | n/a | killed during load even with `max_seq=8` |

The old decode profiler sees only the separate logits projection because the
Qwen trunk executes inside one persistent pool dispatch. It also charges two
bytes per Q8 weight instead of the GGUF Q8_0 size of 34/32 bytes. Its reported
bandwidth and dispatch count are therefore not valid for the trunk. Dividing
the Q4 resident tensor bytes by the measured trunk time gives only about
2.7 GB/s, consistent with the observed 4 GB/s-class behavior.

The Q8 file contains 29.036 GB of tensors: 0.451 GB belongs to block 64/NextN,
and the token embedding is 1.351 GB. Ordinary decode can leave the embedding
file-backed (only one row is read per token) and omit NextN, reducing required
anonymous weight residency to about 27.23 GB.

The existing DS4F cold-pool probe on the same node measured:

| Kernel/layout | Weight bandwidth | Accuracy observation |
| --- | ---: | --- |
| packed BF16 | 730.4 GB/s | hardware/placement control |
| block-scaled Q8 SDOT | 431.4 GB/s | retains per-block scaling |
| row-weight/per-token-activation Q8 | 549.3 GB/s | up to 10.6% relative error on an activation-spike case |
| row-weight/per-64-activation Q8 | 406.3 GB/s | safer spike behavior |

The 800 GB/s HBM roofline is consequently not the Q8 kernel roofline: SDOT,
scale conversion, and scale application are material costs. A 20 tok/s result
requires roughly 540 GB/s over 27 GB before non-matvec work, so the aggressive
row-scaled path must remain gated by end-to-end quality.

## Implemented measurements

The first implementation pass fixed the serialized SSM projections and split
the 48 independent recurrent heads across the 48 persistent workers. It also
added selective Q8 residency and two explicitly selected experimental SDOT
formats. Four-token measurements on `a35-1110s` are:

| Mode | Resident decode weights | Load | Decode | Greedy output for prompt `x` |
| --- | ---: | ---: | ---: | --- |
| Q4 reference | 17.9 GB | 16.094 s | **1.195 tok/s** | unchanged: `ĊThá»©ĠBa,` |
| Q8 `reference` | 27.223 GB | 61.127 s | **0.843 tok/s** (8 tokens, profiling off) | `ĊThá»©Ġhai,Ġ19/` |
| Q8 `row` | 25.639 GB | 70.443 s | **0.822 tok/s** | differs: `ĊThá»©ĠBa,` |
| Q8 `block64` | 26.423 GB | 76.857 s | **0.868 tok/s** | matches reference on this short prompt |

For Q4, cooperative SSM execution reduced `ssm_core` from 89.4 to 45.9
ms/token and improved the earlier 1.144 tok/s intermediate result to 1.195
tok/s. For resident reference Q8, the measured stage costs were 10.8 ms
attention QKV, 4.9 ms attention output, 36.7 ms SSM input projections, 300.6
ms SSM core, 15.6 ms SSM output, 102.5 ms FFN gate/up, and 53.8 ms FFN down per
token. The remaining unclassified time, including the very large vocabulary
head, norms, barriers, and serial SSM preparation, is now the dominant part of
the 1.056 s/token total.

Stage instrumentation is measurably intrusive on this machine: a final
instrumented reference run was 0.773 tok/s versus 0.843 tok/s without it. The
table therefore uses the uninstrumented timing and the stage numbers only for
attribution. A final Q4 mmap correctness rerun also reproduced
`ĊThá»©ĠBa,`; as expected for faulting the 17.9 GB file every token, it achieved
only 0.031 tok/s. A second anonymous Q4 load was killed by the batch node, so
the earlier successful 1.195 tok/s anonymous result is retained as its speed
measurement.

Neither experimental SDOT representation passes the performance gate. `row`
also fails even the short greedy-output gate. `auto` therefore remains the
resident GGUF Q8 reference path; the experimental modes require explicit
`--q8-mode row` or `--q8-mode block64`.

## Original root cause (addressed)

The persistent Qwen worker parallelized attention and dense FFN rows, but for
each of the 48 SSM layers thread 0 disabled the pool and called the complete SSM
forward routine while the other 47 cores waited. This serialized the large QKV,
gate, and output projections as well as convolution and recurrence. The first
implementation pass fixed this execution structure before further kernel tuning.

The projection and recurrence serialization described above is now fixed. The
remaining work is dominated by the still-unclassified half of token time,
serial SSM preparation, the 1.35 GB vocabulary head, repeated activation
quantization, and kernel/NUMA behavior that falls far below the cold-pool
microbenchmark.

## Next tasks and implementation plan

Work in this order. Do not make either experimental format the default until it
passes all numerical gates.

### 1. Obtain a low-overhead token-time ledger

The current per-stage profiler perturbs decode by about 8–20%, and roughly half
of reference-Q8 token time is not assigned to a stage. Replace hundreds of
per-layer clock calls with per-thread cycle counters accumulated in registers
and sampled only at coarse boundaries.

Implementation:

1. Add counters for final RMSNorm, LM head, residual/norm work, barriers, SSM
   alpha/beta projections, convolution, Q/K normalization and expansion,
   recurrence, and SSM output normalization/gating.
2. Read the A64FX virtual counter (`cntvct_el0`) or one monotonic timestamp once
   around each whole category, not each individual matrix. Accumulate locally
   and publish after the token dispatch.
3. Count actual resident bytes from the active layout (`reference`, `row`, or
   `block64`) and report effective GB/s per category.
4. Add a no-print sampling mode that profiles one token after at least eight
   warm tokens. Require less than 2% timing difference from profiling disabled.

Deliverable: a ledger whose categories sum to at least 95% of wall time and a
32-token uninstrumented reference baseline.

### 2. Remove serial SSM preparation

`ssm_core` is still about 300 ms/token for Q8 despite parallel recurrence, far
above Q4. Split and measure it before changing arithmetic.

Implementation:

1. Pre-dequantize the depthwise convolution weights once during model loading
   into the transposed `[conv_k][qkv_dim]` layout; stop rebuilding it for every
   layer and token.
2. Compute alpha and beta cooperatively. Their output has only 48 rows, so use
   one worker per row and avoid a nested pool dispatch.
3. Channel-split convolution and SiLU across all workers. Give each worker a
   fixed channel range and update the circular history for that range.
4. Parallelize Q/K L2 normalization and head expansion by group/head. Preserve
   the current per-head recurrence assignment and first-touch recurrent state
   from the same worker that will update it.
5. Fuse recurrence output RMSNorm and gate SiLU into the recurrence worker so
   the head output remains hot.

Gate: identical Q8-reference logits within the existing F32 reduction tolerance,
identical 128-token greedy output, and `ssm_core < 75 ms/token`.

### 3. Optimize the vocabulary head separately

The output matrix is 1.35 GB and is read once per generated token. It must be
reported independently instead of being hidden in serial time.

Implementation:

1. Add a standalone LM-head benchmark using the actual `[248320, 5120]` tensor,
   the production worker mapping, and warm resident pages.
2. Partition rows on eight-row boundaries and first-touch each partition from
   its consuming CMG. Compare static contiguous, CMG-striped, and work-stealing
   mappings.
3. Fuse argmax into each worker's row sweep and reduce 48 local maxima after the
   matvec. Do not write or reread the full logits array during greedy decode.
4. Keep an opt-in full-logits path for callers that need sampling or logprobs.

Gate: greedy token identical to the full-logits reference and at least 400 GB/s
effective LM-head bandwidth. Record the time saved by fused argmax separately.

### 4. Build an exact Q8_0 multi-row kernel before further lossy packing

The current `block64` format re-quantizes two original 32-value blocks and is
both slower and less exact than desired. Implement a kernel that consumes the
resident GGUF Q8_0 bytes directly, retaining both original FP16 scales.

Implementation:

1. Quantize the F32 activation once per 32-value block and cache its int8 values
   and FP32 scales for reuse by every projection consuming that activation.
2. Process eight output rows together. Load each row's original 32 Q8 bytes,
   issue eight SVE `sdot` operations per activation block, convert the lane
   accumulators once, and apply the eight independent weight scales.
3. Unroll two or four 32-value blocks while keeping row accumulators live.
   Benchmark scalar scale conversion, vector FP16 conversion, and pre-expanded
   FP32 scale sidecars. A sidecar is acceptable only if total single-node
   residency stays above the 2 GB reserve.
4. Add software prefetch two block groups ahead and compare CMG-local tensor
   placement against process-wide interleave.
5. Route only large dense projections through this kernel initially; retain the
   reference F32 path for narrow alpha/beta and convolution tensors.

Gate: projection cosine at least 0.9999, relative L2 at most 1%, 128-token
greedy agreement, and at least 500 GB/s on the cold-pool and real-shape tests.

### 5. Reuse activation quantization and fuse paired projections

The current experimental paths quantize the same activation independently in
each worker and again for gate/up or Q/K/V consumers.

Implementation:

1. Add per-token, per-source activation caches for normalized hidden state,
   attention input, SSM input, and FFN input. Store block32 int8 values plus
   FP32 scales, generation-tagged so no stale cache can be used.
2. Have workers cooperatively quantize disjoint blocks once, followed by one
   barrier.
3. Fuse FFN gate/up row panels in one dispatch and interleave their weight
   streams only if the real-shape benchmark shows a gain. Reuse the same cached
   activation for attention Q/K/V and SSM QKV/gate pairs.
4. Count activation-quantization time explicitly and ensure caching does not
   add more barriers than it removes.

Gate: at least 1.5x over the exact-Q8 single-matrix production kernel for the
FFN gate/up pair, with unchanged numerical gates.

### 6. Fix placement and measure sustained model bandwidth

The isolated kernels reach 400–550 GB/s, while end-to-end decode implies only
tens of GB/s. Verify placement rather than assuming interleave is effective.

Implementation:

1. Record page residency per CMG for representative tensors after selective
   loading and after eight decode tokens.
2. Change selective materialization so each worker first-touches exactly the
   rows it later consumes. Keep row partitions stable across all tokens.
3. Pin the four 12-thread worker groups to their CMGs and use the A64FX hardware
   barrier only for cross-CMG synchronization points.
4. Add hardware-counter collection for HBM bytes, L2 misses, SVE instructions,
   and barrier wait cycles around one warmed token.

Gate: real dense projections sustain at least 70% of their cold-pool bandwidth;
otherwise document the counter evidence before changing the kernel again.

### 7. Close modes and then return to Q4

Run a minimum of 32 warm decode tokens with `max_seq=64` for all surviving Q8
paths. Validate ordinary English, multilingual text, code, long-context input,
and the activation-spike prompt. Promote a mode to `auto` only if it passes all
quality gates and is faster than reference by at least 10%.

If exact Q8 remains below 20 tok/s after tasks 1–6, publish the measured
single-node ceiling and its bandwidth/serial-time decomposition rather than
weakening accuracy. Then apply the proven execution, placement, fused-argmax,
and activation-reuse changes to Q4; do not carry over a Q8 packing format merely
because it exists.

## Implementation design

1. Add persistent-forward timings for QKV, SSM projections/recurrence,
   attention output, FFN gate/up/down, logits, barriers/serial work, and exact
   bytes for each tensor format.
2. Parallelize the SSM stages within the already-running persistent workers:
   row-split QKV/gate projections, channel-split convolution, head-split
   normalization/recurrence, and row-split output projection. Pre-dequantize
   the small depthwise-convolution weights once per layer.
3. Open Q8 lazily and selectively materialize decode tensors. Keep embeddings
   mmap-backed, skip NextN for `spec_k=0`, discard source pages after each
   tensor, and abort loading before `MemAvailable` falls below 2 GB.
4. Provide three Q8 modes:
   - `reference`: resident GGUF Q8_0 with F32 activations.
   - `block64`: eight-row, 64-column panels re-quantized to one weight scale per
     64 values, with per-64 activation quantization and SVE SDOT.
   - `row`: eight-row int8 panels with one weight and activation scale, full-K
     int32 accumulation, and scale application once per output row.
5. Quantize each source activation once and reuse it for gate/up and Q/K/V.
   Split panels on eight-row boundaries and align the worker mapping with the
   four 12-core CMGs.
6. Expose `--q8-mode reference|block64|row|auto`. `auto` may select a fast mode
   only after its numerical and greedy-token gates pass; reference always
   remains available.

## Gates

- Packing tests cover every Qwen matrix shape, tails, exact source decoding,
  finite output, and int32 overflow bounds.
- Projection comparisons require cosine at least 0.9999 and relative L2 at
  most 1%.
- A fast mode must reproduce 128 greedy tokens across ordinary, multilingual,
  code, and activation-spike prompts, including the known synthetic next token
  3165.
- Performance uses at least 32 warmed tokens with `max_seq=64`, reports peak
  memory and stage timings, and runs only on the non-Codex node.
- Target: at least 600 GB/s for the row kernel and 20 tok/s end to end. If the
  row mode fails quality, `auto` falls back and the quality-preserving ceiling
  is reported instead of weakening the gate.

Q4-specific packing is deferred until Q8 is closed, although Q4 receives the
shared persistent-SSM parallelism and is remeasured for regression coverage.
