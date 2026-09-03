# Qwen3.8-27B Q8 on one A64FX node

## Current model location and safe staging procedure

The Qwen3.8 27B model directory is:

```text
/home/u14346/models/qwen38/27b
```

Available files on the model filesystem are:

| File | Size | Intended use |
| --- | ---: | --- |
| `Qwen3.8-27B-Q8_0.gguf` | 29,047,086,048 bytes | Single-node Q8 decode target |
| `Qwen3.8-27B-UD-Q4_K_XL.gguf` | 17,923,394,624 bytes | Q4 comparison path |
| `bf16/Qwen3.8-27B-BF16-00001-of-00002.gguf` | 49,986,159,616 bytes | BF16 reference shards; too large for this single-node Q8 run |
| `bf16/Qwen3.8-27B-BF16-00002-of-00002.gguf` | 4,671,576,000 bytes | BF16 reference shard |

`/local` has approximately 60 GB free in the current allocation. Stage the Q8
file into `/local/u14346/qwen38` with the repository helper:

```sh
MODEL=/home/u14346/models/qwen38/27b/Qwen3.8-27B-Q8_0.gguf
STAGE=/local/u14346/qwen38
sh a64fx/llm/stage_gguf_shards.sh "$MODEL" "$STAGE"
```

The helper copies aligned 1 MiB blocks with direct I/O, fsyncs the tail, checks
the final size, and publishes atomically. It is idempotent and reuses a staged
file whose size already matches. Do not use `cp` or an unbounded whole-file
read for this 29 GB model: the node has 32 GB HBM and page-cache accumulation
can stall or kill the interactive session.

The staged model is then:

```text
/local/u14346/qwen38/Qwen3.8-27B-Q8_0.gguf
```

## Single-node staged decode benchmark

Build and run the selective-resident Q8 reference path from the repository
root. This keeps the embedding and unused NextN tensors file-backed while
materializing ordinary decode tensors anonymously:

```sh
make -C a64fx/llm qwen38_runner CC=fcc OPENMP=1
LLM_THREADS=48 OMP_NUM_THREADS=48 OMP_PROC_BIND=close OMP_PLACES=cores \
  numactl --interleave=all sh a64fx/llm/run_qwen38.sh \
  --model /home/u14346/models/qwen38/27b/Qwen3.8-27B-Q8_0.gguf \
  --mode single --nodes 1 --stage-dir /local/u14346/qwen38 \
  --prompt x --max-gen 32 --max-seq 64 --q8-mode reference
```

For a profiling comparison, add `TF_DPROF=1`; use the uninstrumented run for
the headline throughput. Monitor `MemAvailable` during loading and decode and
  terminate the run before it falls below 2 GB. The expected correctness gate is
the same greedy output as the Q8 reference and stable decode over at least 32
generated tokens. Record load time, resident bytes, decode tokens, tok/s, and
the stage ledger in this document after each benchmark.

On the current node, `numactl --interleave=all` is important for the staged
anonymous Q8 buffers. A warmed 32-token corrected reference run measured
3.627 tok/s with it versus 2.820 tok/s without it, with identical output.
Explicit `NUMA_DISTRIBUTE=1 NUMA_N_CMGS=4` pinning measured 3.387 tok/s and
was slower than interleaving. The process-local `NUMA_INTERLEAVE=1` policy
also works, but measured 3.203 tok/s in one repeat; use the external
`numactl` launch for the current headline.

## Measured baseline

## Latest staged benchmark (2026-08-20)

The Q8 model was staged from
`/home/u14346/models/qwen38/27b/Qwen3.8-27B-Q8_0.gguf` to
`/local/u14346/qwen38/Qwen3.8-27B-Q8_0.gguf` using
`stage_gguf_shards.sh`. The corrected persistent-worker build was run with 48
threads, `max_seq=64`, prompt `x`, no NextN draft, and `--q8-mode reference`.

| Run | Load | Decode | Throughput | Resident Q8 | Greedy output prefix |
| --- | ---: | ---: | ---: | ---: | --- |
| 32-token uninstrumented | 66.787 s | 11.405 s / 32 | **2.806 tok/s** | 27.223 GB | `ĊThá»© Ġhai, Ġ19/09/2022 Ġ14:30` |
| 8-token uninstrumented repeat 1 | 71.410 s | 3.908 s / 8 | 2.047 tok/s | 27.223 GB | `ĊThá»© Ġhai, Ġ19/` |
| 8-token uninstrumented repeat 2 | 70.294 s | 4.002 s / 8 | 1.999 tok/s | 27.223 GB | `ĊThá»© Ġhai, Ġ19/` |

The 32-token run reported `MemAvailable=3948.7 MB` immediately after
materialization and completed without OOM or memory thrashing. Its output
matches the established Q8-reference prefix. A corrected profiled 8-token run
reported 2.229 tok/s and the following approximate stage ledger (profiling is
intrusive and is not the headline rate):

```text
attn_qkv=14.4 ms  attn_out=6.2 ms  ssm_in=47.6 ms
ssm_prepare=53.9 ms  ssm_core=18.7 ms  ssm_out=17.8 ms
ffn_gateup=134.6 ms  ffn_down=74.7 ms  lm_head=12.6 ms
```

The persistent Qwen trunk does not update the older pooled-matvec profiler;
its `matvec=0` report is expected. The stage values above are the useful
ledger for this path. An earlier 2.774 tok/s measurement from the first
cooperative-Q/K implementation was discarded because a missing barrier caused
nondeterministic output; the normalization-to-expansion barrier is now present,
and repeated runs are deterministic.

### Exact fused gate/up follow-up (2026-08-20)

The persistent dense FFN now uses an exact four-row fused gate/up Q8 kernel.
It streams the gate and up matrices together and loads the shared F32
activation vector once, while retaining the reference Q8_0 dequantization and
accumulation order. Other tensor types and block64/row experimental formats
fall back to the previous path automatically; no activation quantization is
enabled.

Build and short validation command:

```sh
make -C a64fx/llm qwen38_runner CC=fcc OPENMP=1
OMP_NUM_THREADS=48 OMP_PROC_BIND=close OMP_PLACES=cores \
  ./a64fx/llm/build/qwen38_runner /local/u14346/qwen38/Qwen3.8-27B-Q8_0.gguf \
  --threads 48 --prompt x --max-gen 8 --max-seq 64 --q8-mode reference
```

The post-change run produced the same greedy prefix `ĊThá»©Ġhai,Ġ19/` and
measured 1.903 tok/s. An intrusive profiled run measured 1.982 tok/s and gave:

```text
attn_qkv=13.8 ms  attn_out=6.1 ms  ssm_in=48.9 ms
ssm_prepare=75.3 ms  ssm_core=25.7 ms  ssm_out=17.6 ms
ffn_gateup=121.7 ms  ffn_down=68.2 ms  lm_head=11.6 ms
```

Compared with the earlier profiled ledger, gate/up fell from 134.6 to 121.7
ms/token and down fell from 74.7 to 68.2 ms/token, but the short uninstrumented
rate remains within the observed ~2 tok/s variance. This is therefore kept as
an accuracy-preserving optimization, not yet claimed as a headline throughput
gain. A warmed 32-token run is still required before promotion.

The warmed 32-token post-change run completed at 2.820 tok/s
(11.349 s / 32 tokens), with resident Q8 decode weights still 27.223 GB,
\`MemAvailable=3935.2 MB\` after materialization, and the same output prefix
through `ĊThá»©Ġ19/09/2022Ġ14:30`. This is effectively tied with the earlier
2.806 tok/s 32-token result, so the fused kernel is retained for its lower
projection time but not presented as a large end-to-end gain.

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

Under the current interleaved-HBM launch, the original packed SDOT reached
5.729 tok/s for 32 tokens and 4.981 tok/s for 128 tokens, but diverged from
exact Q8 after the shared 32-token prefix. The packed decode layout has since
been hardened to preserve the original Q8_0 weight bytes and both FP16
scales; activation quantization remains approximate. This fidelity-preserving
layout measured 4.335 tok/s for 32 tokens and 4.931 tok/s for 128 tokens, and
still diverged at 128 tokens. FFN-only packing reached 4.596 tok/s for 32
tokens and 4.643 tok/s for 128 tokens, with the same divergence. The row mode
now routes correctly through its int8 kernel (the previous dispatch bug could
SIGSEGV), but produces corrupted text at 3.892 tok/s. `auto` therefore
remains the resident GGUF Q8 reference path; all packed modes require explicit
selection and fail the quality gate.

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
4. Provide Q8 modes:
   - `reference`: resident GGUF Q8_0 with F32 activations.
   - `block64`: eight-row, 64-column panels retaining original Q8_0 weight
     bytes/scales, with per-64 activation quantization and SVE SDOT.
   - `block64-ffn`: packed SDOT for FFN gate/up/down only; exact elsewhere.
   - `block64-exact`: packed original Q8_0 bytes/scales with F32 activations;
     layout-only quality-preserving experiment.
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

## Null-GEMM bandwidth probe (2026-08-20)

Before judging the exact Q8 kernels against the 20 tok/s goal, use the guarded
`TF_NULL_GEMM=1` mode. It preserves the persistent decode schedule and worker
partitioning, reads every resident Q8 weight cache line, skips dequantization
and dot products, and writes zero projection outputs. This is a decode-shaped
streaming lower-compute bound; RMSNorm, recurrent/attention bookkeeping, and
barriers still remain in the runner.

Build after changing `common/transformer.h` or the runner (the Makefile does not
track all common-header dependencies):

```sh
make -B -C a64fx/llm qwen38_runner CC=fcc OPENMP=1
```

Run against the staged model with interleaved HBM placement:

```sh
numactl --interleave=all env TF_NULL_GEMM=1 \
  OMP_NUM_THREADS=48 OMP_PROC_BIND=close OMP_PLACES=cores \
  ./a64fx/llm/build/qwen38_runner \
  /local/u14346/qwen38/Qwen3.8-27B-Q8_0.gguf \
  --threads 48 --prompt x --max-gen 32 --max-seq 64 --q8-mode reference
```

Observed output on the A64FX node:

```text
q8 resident: 27.223GB decode weights (reference-q8)
decode=32 tokens 6.647s 4.814 tok/s
null-stream=27.223 GB/tok 131.1 GB/s
20 tok/s requires 544.5 GB/s (equiv 4.81 tok/s)
```

Thus the current decode-shaped memory-only ceiling is about 4.8 tok/s, or
131 GB/s sustained by this path. The 20 tok/s goal requires roughly 4.15x more
sustained bandwidth than the probe delivered. This result should be treated as
the baseline for placement and stream-efficiency work; `TF_NULL_GEMM` is
diagnostic only and is never selected by `auto`.

### CMG-local placement follow-up

The original `NUMA_DISTRIBUTE` worker affinity was round-robin (`tid % 4`),
so contiguous row slices were consumed by remote CMGs. It now uses contiguous
groups:

```text
tid  0..11 -> CMG0 / NUMA node 4 / cores 12..23
tid 12..23 -> CMG1 / NUMA node 5 / cores 24..35
tid 24..35 -> CMG2 / NUMA node 6 / cores 36..47
tid 36..47 -> CMG3 / NUMA node 7 / cores 48..59
```

The Q8 materializer splits each tensor into the same contiguous row ranges, so
destination pages are first-touched by the workers that later read them. Use
the documented A64FX placement and large-page settings:

```sh
numactl -C 12-59 -m 4-7 env \
  XOS_MMM_L_HPAGE_TYPE=hugetlbfs \
  XOS_MMM_L_PAGING_POLICY=demand:demand:demand \
  XOS_MMM_L_ARENA_FREE=2 XOS_MMM_L_HUGETLB_FALLBACK=1 \
  NUMA_DISTRIBUTE=1 NUMA_N_CMGS=4 NUMA_CMG_BUDGET_GB=7 \
  NUMA_ALIGNMENT=2097152 TF_NULL_GEMM=1 \
  OMP_NUM_THREADS=48 OMP_PROC_BIND=close OMP_PLACES=cores \
  ./a64fx/llm/build/qwen38_runner \
  /local/u14346/qwen38/Qwen3.8-27B-Q8_0.gguf \
  --threads 48 --prompt x --max-gen 32 --max-seq 64 --q8-mode reference
```

Set `NUMA_REPORT=1` for a non-invasive `/proc/self/numa_maps` check. The
verified large tensors were 2 MiB hugepage mappings distributed over all four
nodes; a representative report was `N4=11 N5=13 N6=11 N7=10`. Small tensors
can legitimately occupy one node because they are smaller than one hugepage.

The null scanner first used one SVE 512-bit load per cache line and measured
162.1 GB/s (5.96 equivalent tok/s). It now uses eight independent SVE load
streams, matching the A64FX raw-stream benchmark structure; the CMG-local run
measured:

```text
null-stream=27.223 GB/tok 307.2 GB/s; 20 tok/s requires 544.5 GB/s
equivalent throughput: 11.28 tok/s
```

The earlier interleaved baseline was 131.1 GB/s with the older scalar scanner,
so the full progression reflects both correct CMG placement and sufficient
outstanding loads. Hugepage settings alone changed little (151.9 GB/s in one
repeat), so they are retained for TLB stability but are not the primary
bandwidth fix. The remaining gap to 544.5 GB/s is now a decode-shaped stream
and projection-scheduling problem rather than the original round-robin NUMA
placement problem.

For a pure hardware ceiling, use `TF_NULL_GEMM=2`. This skips transformer
forward after loading and scans all 497 resident Q8 tensors as one batched pool
stream, with the same CMG row ownership:

```sh
numactl -C 12-59 -m 4-7 env \
  XOS_MMM_L_HPAGE_TYPE=hugetlbfs \
  XOS_MMM_L_PAGING_POLICY=demand:demand:demand \
  XOS_MMM_L_ARENA_FREE=2 XOS_MMM_L_HUGETLB_FALLBACK=1 \
  NUMA_DISTRIBUTE=1 NUMA_N_CMGS=4 NUMA_CMG_BUDGET_GB=7 \
  NUMA_ALIGNMENT=2097152 TF_NULL_GEMM=2 \
  OMP_NUM_THREADS=48 OMP_PROC_BIND=close OMP_PLACES=cores \
  ./a64fx/llm/build/qwen38_runner \
  /local/u14346/qwen38/Qwen3.8-27B-Q8_0.gguf \
  --threads 48 --prompt x --max-gen 32 --max-seq 64 --q8-mode reference
```

Measured result:

```text
null-batched tensors=497 27.223 GB/pass 761.3 GB/s (3 passes)
```

This reaches the expected near-800 GB/s hardware ceiling. The 307.2 GB/s
decode-shaped result is therefore caused by per-projection scheduling and exact
Q8 compute/dequant work, not an inter-CMG placement failure.

### Packed exact-layout experiment

`--q8-mode block64-exact` reorders each eligible Q8 tensor into contiguous
8-row × 64-column groups while retaining every original Q8_0 scale and byte.
Its F32-activation kernel is numerically quality-preserving in principle and
matched the reference output prefix in the 32-token probe, but it measured only
2.363 tok/s versus 3.857 tok/s for the current reference layout under the same
CMG-local launch. It therefore remains explicit-only and is not a candidate for
`auto`.

### Exact FFN gate/up register-pressure optimization

The production exact Q8 path now uses a two-row-per-projection fused gate/up
kernel by default. It keeps the original Q8_0 scales, F32 activations, and two
partial sums per output, but reduces live SVE state versus the four-row pair.
Set `TF_Q8_FUSED_2ROW=0` to restore the previous kernel for an A/B check.

Under the CMG-local launch, a 128-token comparison measured 4.210 tok/s with
the new kernel versus 4.075 tok/s before it. The generated stdout was
byte-identical (`cmp=0`). The 32-token run produced the same output prefix;
short runs have more timing noise. This is now the default exact gate/up path.

An A64FX SVE SiLU×mul replacement for the persistent BF16 path was also
tested with the same four-rank launch. It measured 46.91 ms/token versus
39.82 ms/token for the reference `expf` path and was removed; exact BF16
decode therefore keeps the scalar activation.

An FP32 sidecar for all Q8 block scales was also tested to remove repeated
FP16-to-F32 conversion. It exceeded the 32 GB HBM budget during materialization
(the process was OOM-killed), so no sidecar allocation remains in production.

A two-SDOT split-int16 activation experiment was also rejected: it produced
corrupted output and only 1.433 tok/s at 32 tokens. Its code was removed.

Native NextN selective residency was tested separately. With `--spec-k 4`, the
target-model greedy check accepted only 9/31 drafts and throughput fell to
2.604 tok/s, so selective Q8 mode continues to require `--spec-k 0`.

### Compiler scheduling

`qwen38_runner` now adds Fujitsu `-Kfast` only for the `fcc` build. A 128-token
run measured 4.295 tok/s versus 4.075 tok/s with the prior flags, and generated
stdout was byte-identical to the reference (`cmp=0`). Other Makefile targets do
not inherit this flag.

### Four-CMG ownership and barrier check

The production launch uses four CMGs explicitly:

```text
numactl -C 12-59 -m 4-7 env NUMA_DISTRIBUTE=1 NUMA_N_CMGS=4 \\
  NUMA_CMG_BUDGET_GB=7 NUMA_ALIGNMENT=2097152 \\
  XOS_MMM_L_HPAGE_TYPE=hugetlbfs \\
  XOS_MMM_L_PAGING_POLICY=demand:demand:demand \\
  XOS_MMM_L_HUGETLB_FALLBACK=1 ...
```

With 48 workers, IDs 0--11 bind to cores 12--23 (CMG0), 12--23 to
cores 24--35 (CMG1), 24--35 to cores 36--47 (CMG2), and 36--47 to cores
48--59 (CMG3). Large tensors are split into contiguous row ranges and each
range is loaded/first-touched by its owning worker. This avoids the previous
round-robin worker affinity, where each contiguous tensor slice was commonly
remote from its consumer. Small tensors below the worker split threshold may
remain on CMG0; they are negligible compared with the 27.223 GB Q8 decode
working set.

The resulting four-CMG placement was checked with `NUMA_REPORT=1`: representative
large mappings used 2 MiB pages across N4--N7 (`N4=11 N5=13 N6=11 N7=10`). The
batched null scan reached 761.3 GB/s, or about 190 GB/s per CMG when divided
evenly, so the memory system is no longer limited to a single inter-CMG stream.

An opt-in A64FX W0 hardware-barrier variant was tested (`TF_HW_BARRIER=1`) but
deadlocked before the first decode token with workers spanning four CMGs. It is
not part of the production path; the default WFE/SEV software barrier remains
enabled. The barrier experiment did not change model data or output, and the
source was reverted after the hang.

### Exact Q8 row blocking

Individual Q8 projections now default to an exact four-row SVE kernel. The
previous eight-row kernel kept 16 vector accumulators live and was slower on
A64FX, likely from register pressure. Both kernels preserve the original
Q8_0 block order, FP16 scale conversion, and FP32 accumulation structure.
`TF_Q8_ROWS=8` restores the old kernel for comparison.

On identical 32-token runs with separate stdout capture:

```text
TF_Q8_ROWS=4   4.022 tok/s
TF_Q8_ROWS=8   3.606 tok/s
stdout cmp     0 (identical SHA-256)
```

The four-row kernel is therefore the production default; this optimization
changes scheduling/register pressure only and does not change model quality.
The two-row experimental kernel also passed the 64-token output check but
measured 4.308 tok/s versus 4.361 tok/s for four rows in the longer A/B, so
`TF_Q8_ROWS=2` remains available for hardware-specific testing and is not the
default.

### Hierarchical four-CMG barrier

The persistent decoder now uses a software hierarchical barrier by default.
Each CMG first synchronizes its 12 workers on a cache-line-local counter; only
the four CMG leaders then exchange the global sense. This removes the
48-thread inter-CMG atomic counter from every layer stage. Set
`TF_HIER_BARRIER=0` to restore the old barrier for an A/B check.

With the same four-row exact Q8 kernel and 48-core/four-CMG launch:

```text
old barrier, 32 tokens     3.975 tok/s
hierarchical, 32 tokens    4.465 tok/s
hierarchical, 64 tokens    4.463 tok/s
stdout cmp                 0 (32- and 64-token checks)
```

The hierarchical path is now production default. It changes only worker
synchronization; the generated output remained byte-identical.

An exact scale-after arithmetic variant was also tested. Applying each
Q8_0 block scale after its local dot reduction changed the floating-point
rounding order and required four horizontal reductions per block; it measured
only 2.764 tok/s and failed the stdout comparison. It was removed from the
production source.

A distance-8 software-prefetch variant of the four-row kernel was also tested
at 4.333 tok/s with identical output. It was within normal run noise of the
default and added instruction overhead, so it was removed.

### Packed FFN default

The `block64-ffn` mode packs only FFN gate/up weights into 8-row × 64-column
groups and uses the A64FX SDOT path with per-64 activation quantization.
Attention, SSM, FFN down, and the output head remain on the reference exact
Q8 path. Greedy stdout was byte-identical to `reference` for 32 generated
tokens from three probes (`x`, `Hello`, and `The quick brown fox jumps over
the lazy dog.`); the 64-token `x` probe also matched.

Measured decode rates:

```text
prompt                  reference     block64-ffn
x, 64 tokens              4.463          5.448 tok/s
Hello, 32 tokens          4.372          4.699 tok/s
fox prompt, 32 tokens     4.868          6.042 tok/s
```

`--q8-mode auto` remains the reference exact path. `--q8-mode block64-ffn`
is explicit-only: it passed the 32/64-token probes above but diverged from
reference at 128 generated tokens (first output difference around byte 327).
The packed mode is therefore a useful speed experiment, not a quality-safe
default; its activation quantization must not be used when accuracy is the
requirement.

Finer per-32 and per-16 activation-scale variants were tested while chasing
the 128-token divergence. Per-32 remained divergent; per-16 restored the
128-token output but fell to 3.073 tok/s. Both experiments were removed, and
the explicit packed mode is back to its faster per-64 scale implementation.

An A64FX sector-1 pointer-tag experiment for Q8 weight streams was also
rejected: with the documented `FLIB_SCCR_*` settings it produced identical
output but only 4.020 tok/s, below the untagged exact path. No sector tagging
remains in production.

Compiler reassociation was tested with an additional `-Ofast` flag. It kept
the 32-token greedy output identical but measured only 4.189 tok/s, below the
normal `-O3 -Kfast` build, so the production flags were restored.

## Four-node BF16 tensor parallel decode (2026-08-20)

The split Qwen3.8-27B BF16 model is now staged as final rank-local TP4 slices
rather than mapping the 54.66 GB source on every node.  The stage contains 497
decode tensors per rank and deliberately omits the embedding and NextN weights.
The exact rank payload is 14.321 GB (14,323,023,872-byte file including the
2 MiB header), broken down as approximately 2.013 GB of replicated SSM Q/K and
one quarter of the remaining decode weights.  Runtime dimensions are six Q
heads, one KV head, FFN 4352, SSM dt-rank 12, and vocabulary 62080 per rank.

Build, stage, verify, and measure with:

```sh
make -B -C a64fx/llm qwen38_tp_stage tp_runner CC=fcc OPENMP=1
./a64fx/llm/run_qwen38_bf16_tp4.sh plan
./a64fx/llm/run_qwen38_bf16_tp4.sh stage
TP_STAGE_VERIFY=1 TP_NULL_STREAM_PASSES=1 \
  ./a64fx/llm/run_qwen38_bf16_tp4.sh stream
TP_NULL_STREAM_PASSES=10 ./a64fx/llm/run_qwen38_bf16_tp4.sh stream
TP_MAXGEN=64 TP_PERF_WARMUP=32 \
  ./a64fx/llm/run_qwen38_bf16_tp4.sh bench
```

Staging uses bounded 64 MiB reads, periodic `fdatasync`, source-page
`POSIX_FADV_DONTNEED`, and an atomic partial-file rename.  A valid same-size TP
file is reused.  The loader verifies the header and optional full tensor hashes,
aborts below 2 GB `MemAvailable`, and leaves approximately 16 GB available after
the resident load on these nodes.

### Placement and memory ceiling

The first TP2 arena used the Fujitsu large-page allocation path and achieved
only 180--189 GB/s, effectively one CMG.  TP4 uses a demand-paged anonymous
mapping with `MADV_NOHUGEPAGE`; the 48 pinned loader workers first-touch the same
contiguous ranges they consume (workers 0--11 through 36--47 own CMGs 4--7).
All four full-file checksums passed.  The ten-pass batched stream measured:

```text
rank 0  828.2 GB/s
rank 1  839.2 GB/s
rank 2  841.7 GB/s
rank 3  843.1 GB/s
```

Thus 700 GB/s per node is not merely a hardware estimate; the real 14.321 GB
rank shards sustain at least 828 GB/s.  At 700 GB/s the weight-only floor is
20.46 ms/token (48.9 tok/s), and at the measured slow-rank stream it is
17.29 ms/token (57.8 tok/s).  Each node performs about 7.16 GMAC or 14.32 GFLOP
per token, so an exact FP32-activation/BF16-weight kernel at 700 GB/s needs only
about 700 GFLOP/s and remains bandwidth-bound.  Eight independent row results
provide the instruction-level overlap between weight loads, BF16 widening, and
FMA; a separate compute/load pipeline is not required.

The live four-node uTofu topology is a compact 2x2 group.  Its 16 KiB FP32
all-reduce measured 13.88 us warm and 25.40 us after a 32 MiB eviction.  The
model performs 129 reductions/token, giving a 1.8--3.3 ms transport floor.  The
accepted long run spends 5.55--7.91 ms/token in communication including rank
arrival skew.  Communication cannot generally cross a layer's residual/norm
dependency, so reducing skew and projection time is more useful than pretending
all uTofu time can be hidden behind the next layer.

### TP4/TP6/TP12 sweep after the 12-node restart (2026-08-21)

The launcher now accepts `TP_SIZE=4`, `6`, or `12` and stages each configuration
under `/local/u14346/qwen38-bf16-tp${TP_SIZE}`.  All three rank-local sets fit in
the 87 GiB node-local filesystem.  The measured rank-0 files, including their
headers, were:

| configuration | rank file | attention layout | runtime FFN | steady decode |
|---|---:|---|---:|---:|
| TP4 | 14.323 GB | 6 Q / 1 KV head per rank | 4352 | 21.95 tok/s |
| TP6 | 13.352 GB | replicated attention (4 KV heads) | 3072 | 21.12 tok/s |
| TP12 | 9.362 GB | replicated attention (4 KV heads) | 1536 | 18.47 tok/s |

The benchmark used the same synthetic one-token prompt, `TP_MAXSEQ=128`,
`TP_PERF_WARMUP=8`, and `TP_MAXGEN=16`.  Rank-0 steady timings were 45.47 ms
(TP4), 47.26 ms (TP6), and 54.04 ms (TP12) per token.  The corresponding
compute/communication splits were 39.21/6.27, 38.07/9.19, and 40.82/13.22
ms/token.  TP6 and TP12 cannot shard the four KV heads evenly, so the runtime
keeps attention replicated; this explains both their larger per-rank attention
payload and the extra all-reduce cost.  On this live topology TP4 remains the
best BF16 decode configuration; neither wider group reaches the 40 tok/s target.

Reproduce the sweep with:

```sh
for n in 4 6 12; do TP_SIZE=$n ./a64fx/llm/run_qwen38_bf16_tp4.sh stage; done
for n in 4 6 12; do TP_SIZE=$n TP_MAXGEN=16 TP_PERF_WARMUP=8 \
  ./a64fx/llm/run_qwen38_bf16_tp4.sh bench; done
```

### Decode implementation and result

An experimental pair-interleaved eight-row BF16 layout and SVE kernel were also
implemented.  The initial single-accumulator version diverged from row-major at
generated token 14.  The kernel now mirrors the row-major low/high accumulation
order; matched 64-token dumps are identical, while the live run improved from
39.05 to 36.55 ms/token (25.61 to 27.36 tok/s).  Therefore
`TP_STAGE_BF16_PV=1` is now the launcher default, with `PV=0` retained as the
source-equivalent rollback.  Software prefetch was slightly slower end to end
and remains off.

The larger win was scheduling: `transformer_forward_partial()` previously used
the legacy path and issued 578 pool dispatches/token.  Full-range TP decode now
uses one persistent worker dispatch, with rank all-reduces after SSM/attention
output and FFN down.  The hierarchical CMG barrier caused seconds of rank skew
in this TP path, so the TP4 launcher fixes `TF_HIER_BARRIER=0`; the ordinary
SEV/WFE barrier is stable.  `TP_PERF_WARMUP` excludes first-use recurrent-state
faults from measured decode statistics.

Final safe row-major `-O3` result, 32 warm tokens followed by 64 measured
tokens:

```text
rank 0  40.72 ms/token  compute 34.66 ms  comm 6.06 ms  24.55 tok/s
rank 1  40.81 ms/token  compute 34.37 ms  comm 6.44 ms
rank 2  40.81 ms/token  compute 32.18 ms  comm 8.62 ms
rank 3  40.81 ms/token  compute 33.35 ms  comm 7.45 ms
```

This clears the required 20 tok/s exact-path performance target.  The 40 tok/s
stretch goal is not yet reached: 25 ms/token would require moving projection
execution from the current effective 413--445 GB/s toward the 828 GB/s resident
stream ceiling while also cutting the 6.1--8.6 ms synchronization component.
The next targets are the short-K SSM output projection, FFN gate/up/down, and
rank-0 SSM serial work.  `TP_AR_ROBUST=2` was neutral.  `TP_AR_A2A=1` was
slightly slower and changed the token stream at token 14.  Fujitsu `-Kfast` was
also slower and diverged at token 14, so all three remain rejected.

Correctness caveat: accepted TP4 uses row-major staged tensors whose hashes
match the source slices.  It starts with synthetic second-token ID 5840, not
the historical TP2/TP12 control ID 3165.  A deterministic reduction still
produced 5840 and was much slower.  Therefore 24.55 tok/s is a valid performance
result for the current source-equivalent TP4 execution, but the historical
cross-topology greedy gate remains open and must be resolved before calling TP4
model quality production-validated.

### Four-node BF16 follow-up optimization sweep

The safe 24.55 tok/s result is not improved by the following measured changes:

- `TF_PODD_MV=1`: 42.95 ms/token (23.28 tok/s); row-major predicated loads
  lost bandwidth despite removing widening instructions.
- 44 rather than 48 workers: 43.39 ms/token (23.05 tok/s).
- `TP_AR_BF16=1`: 45.69 ms/token (21.89 tok/s) and a different token stream;
  conversion and rank-arrival overhead exceed the payload saving.
- `-mcpu=a64fx`: 42.53 ms/token (23.51 tok/s), slower than the generic
  Armv8.2+SVE build.
- Sparse software L2 prefetch over the eight row streams caused a rank-0
  memory-queue stall and 138.68 ms/token, so it was removed.
- An exact-order pair-packed prototype retained two accumulators per row but
  failed the token gate; it was removed rather than exposed as a selectable
  mode.

The 40 tok/s target is 25 ms/token.  With 6--9 ms of unavoidable current uTofu
synchronization, exact single-token decode would need to execute 14.321 GB of
BF16 projections in roughly 16--19 ms, or 754--895 GB/s including widening and
FMA.  That is at or above the 828 GB/s load-only ceiling.  Therefore 40 tok/s on
four nodes is not reachable by another small M=1 kernel or environment tweak.
The remaining algorithmic route is multi-token MTP speculative verification:
batch several drafted positions through the trunk so one weight scan verifies
multiple tokens.  The existing `TP_SPEC_K` path only measures draft agreement
and still performs one trunk scan per accepted token; it must be extended with
batched state checkpoint/rollback and greedy acceptance before it can raise
exact accepted-token throughput.

### Live TP4 follow-up rejection tests (2026-08-21)

The compact BF16-to-W8A8 SDOT prepack is valid only from row-major stage data.
Running it with `TP_STAGE_BF16_PV=0` fixed the earlier packed-layout misuse and
produced finite tokens, but measured only **8.0 tok/s** (124.62 ms/token,
120 GB/s) for 8 tokens. Its one-time prepack converted 401 tensors to 7.378
GB. Activation quantization, SDOT conversion, and scale application outweigh
the reduced weight traffic, so it remains rejected.

The resident W8A8 path now caches each thread's per-block activation
quantization while the same activation vector is reused by adjacent
projections. In a matched 16-token/8-warmup TP4 probe, disabling the cache
measured 119.44 ms/token (8.37 tok/s), while the default cache measured
107.91 ms/token (9.27 tok/s). The cache reuses the exact int8 values and
scales, so it does not change the SDOT result; `TF_W8_ACT_CACHE=0` restores
the uncached A/B path.

An SVE vectorized replacement for the per-64 activation quantizer was rejected:
its first TP4 token changed from the exact reference (`5840` to `23`), so the
reference scalar rounding remains in production.

An `svld2` interleaved-load variant of the exact BF16 PV kernel was also
compiled and tested. The A64FX compiler lowered it poorly: it diverged on the
first token and ran at 104.32 ms/token. It was removed. The restored exact
`svld1` PV8 path reproduced the accepted prefix
`5840,22456,2228,9867,72452,5840,174427,174342,198,58024,220,248046`.

The SSM SVE recurrence was also given an opt-in two-row dot path
(`TF_SSM_SVE2=1`) that shares Q/K vector loads while preserving each row's
reduction order. It reproduced 32/32 tokens and measured 36.81 ms/token
(27.17 tok/s), effectively tied with the 27.36 tok/s PV8 reference, so it is
retained as an opt-in A/B path rather than made default. A fused greedy-only
LM-head argmax was then rechecked under the current TP4 run and diverged at
token 14; it remains disabled and full logits remain the default.

The live `TP_SPEC_K=1` probe remains unusable for throughput: with NextN
materialization it measured 1.76 tok/s for four generated tokens, issued 132
all-reduce calls/token, and reported `MTP greedy match=0/4 alpha=0`. Since the
current driver still runs the full trunk once per token and does not checkpoint
or batch-verify the draft sequence, speculative mode is not enabled by the
BF16 launcher.

The TP4 launcher now defaults to `OMP_PROC_BIND=spread` rather than `close`.
On the live four-node session, the exact PV8 binary measured 37.37 ms/token
(26.76 tok/s) with spread and 51.73 ms/token (19.33 tok/s) with close; both
streams matched the saved 32-token greedy output.  Spread kept rank compute
times within about 3 ms, while close left ranks 1--3 waiting roughly 20--22
ms/token for a rank-0 compute outlier.  The setting remains overrideable.

I also tested `TP_AR_BATCH=1` to make the 5120-float decode reduction use a
single contiguous payload-plus-trailer Put.  Although this removes one Put per
round, the live run slowed to 42.75 ms/token and diverged after the shared
prefix.  The uTofu combined-Put ordering is therefore not safe for this
trailer protocol; TP4 keeps the validated split payload/trailer transfers.

An exact compiler A/B with `-funroll-loops` was also rejected: matched 64-token
streams measured 38.31 ms/token with unrolling versus 37.48 ms/token for the
normal `-Kfast` build.

An exact BF16 gate/up fused microkernel was prototyped to share activation
loads, but its opt-in run diverged immediately under the longer-context test.
It was removed rather than exposed as a selectable path.

The live TP runner also now inherits the existing Fujitsu `-Kfast` Qwen build
flag. It was previously applied only to the single-node Qwen target, not
`tp_runner`; a four-node BF16 PV A/B preserved 32/32 tokens and measured 36.59
ms/token (31.99 ms compute), versus approximately 36.8 ms/token without it.
The repository's older `-Kocl,hpctag` suggestion is incompatible with this
`-Nclang` compiler (`unknown argument: -Khpctag`), while `-Kocl` alone built
successfully but slowed the live run to 43.02 ms/token, so neither is enabled.

## Q8 TP4 staging implementation (2026-08-20)

The TP stage format now supports both BF16 and native Q8_0 tensors. Version 2
records source and local row byte sizes, preserves Q8_0's 32-value blocks and
FP16 scales, and validates those layouts when loading a rank blob. Single-file
GGUF inputs are supported in addition to split GGUF inputs.

The Q8 source was staged safely to `/local/u14346/qwen38-q8-source` and rank
shards were generated under `/local/u14346/qwen38-q8-tp4`:

```text
source: 29,047,086,048 bytes
rank00..rank03: 7,607,992,320-byte payload each
entries per rank: 497
stage format: q38tp-v2
```

After the live-session restart, the native Q8 TP4 stage was regenerated from
`/home/u14346/models/qwen38/27b/Qwen3.8-27B-Q8_0.gguf` and measured 173.5
ms/token (5.76 tok/s) with spread affinity.  The exact block64 repack was also
tested; it reached 170.75 ms/token (5.86 tok/s), but changed the greedy stream
and is therefore not an exact replacement.  Neither Q8 layout is competitive
with the exact BF16 PV8 path, so no Q8 default was changed.

The launcher is `a64fx/llm/run_qwen38_q8_tp4.sh`. It uses exact resident Q8_0
weights, disables panel repacking, and provides `plan`, `stage`, `stream`,
`null`, `check`, `bench`, and `profile` modes. A four-node uTofu decode run is
still required to establish the TP4 correctness and throughput baseline.

### BF16 NextN materialization and TP vocabulary fix (2026-08-20)

The TP4 launcher now accepts `TP_SPEC_K` instead of forcing it to zero. When
drafting is enabled, the auxiliary NextN tensors are copied once into
NUMA-distributed anonymous memory; the staged shared vocabulary shard is reused
without duplication. This reduced the live four-node BF16 `K=1` probe from
approximately 0.006 tok/s to 6.69 tok/s for four generated tokens.

The TP NextN head also now computes only its local vocabulary rows and uses the
same TP argmax reduction as the trunk. The current probe reports `MTP greedy
match=0/4`, so batched accepted-token verification and the model-specific draft
quality gate remain open.

The standalone BF16-PV UF2 kernel candidate was also checked at K=5120: the
current kernel sustained 800.78 GB/s with zero error, while UF2 sustained
712.59 GB/s and differed by 1.76e-5 absolute (2.94e-7 relative). It was not
integrated into TP4.

NextN attention and FFN projections were then TP-sharded across the four ranks
and their partial outputs were all-reduced. The live `K=1` probe measured 6.40
tok/s versus 6.69 tok/s before sharding; the extra reductions offset the saved
auxiliary weight traffic. The sharding is retained for correctness and as a
foundation for multi-position verification, but it is not an end-to-end gain
for single-draft decoding.

### TP4 BF16 reduction topology check (2026-08-21)

An opt-in two-level TP all-reduce was tested on the live four-node BF16 path,
using rank groups `{0,1},{2,3}` followed by `{0,2},{1,3}` and routing both
hidden-state sums and vocab argmax through the same hierarchy. It preserved
the checked greedy prefix, but was slower in the matched runs: flat TP was
36.07 ms/token (27.72 tok/s) versus 37.10 ms/token (26.95 tok/s); an earlier
pair was 38.09 versus 37.90 ms/token. The experimental runner integration
was removed; the flat reduction remains the production path.

The existing `TP_OVERLAP=1` tiled projection experiment was also rechecked.
With 1024-row tiles it changed the first token and increased reductions from
129 to 641 per token, measuring 80.90 ms/token (12.36 tok/s). It is rejected:
the tile-level reductions do not preserve the live TP4 lockstep/stream and
their extra synchronization overwhelms any attempted compute/communication
overlap.

### Persistent SSM projection barrier removal (2026-08-21)

The persistent BF16 decoder now lets the independent SSM QKV/gate and
alpha/beta projections proceed without an intermediate worker barrier. The
existing barrier after alpha/beta still protects convolution and recurrent
state preparation. The first 13 token IDs remained identical; matched live
32-token probes were 39.48/41.53 ms/token with the change versus
53.44/37.09 ms/token with the original barrier. A longer 64-token/32-warmup
pair measured 36.51 and 36.93 ms/token with the change, compared with the
previous flat-path 36.07 and 38.09 ms/token. It is retained as a safe,
small BF16 improvement; a 128-token/64-warmup run remained on the expected
prefix and measured 39.04 ms/token. Live-run jitter remains significant.

An attempted TP4 SSM Q/K traffic reduction was rejected. Each rank was made to
project only the Q/K groups used by its local dt-head interval while retaining
all local V rows. It matched the first 13 IDs but diverged at token 14 and
measured 46.04 ms/token, so the reference replicated-Q/K projection remains.

Distributing the post-convolution QKV/state copies across the persistent
workers also passed the 32-token prefix check, but was neutral at 64 tokens
(36.92 ms/token versus 36.51/36.93 ms/token for the scalar-copy baseline).
It was reverted to keep the SSM path simpler.

The runner now exposes an opt-in `TP_INT8_MODE=block64-ffn` experiment that
packs only BF16 FFN gate/up/down projections into compact W8A8 SDOT weights;
SSM, attention, and the output head remain BF16. It requires
`TP_STAGE_BF16_PV=0` because the packer consumes row-major staged weights.
This mode is implemented but awaits a TP4 live allocation for correctness and
throughput validation.

### Qwen3.8 BF16 batched prefill and TP handoff (2026-08-21)

The TP runner now has a Qwen hybrid batched-prefill path. SSM state updates and
causal attention remain in prompt order, while the large BF16 projections use
token-major GEMMs. Prefill forces the row-major staged-weight layout (`PV=0`);
the packed decode layout is not valid for these GEMMs.

On TP12, the live measurements were 19.94 tok/s for 128 prompt tokens and
25.35 tok/s for 512 prompt tokens. The path preserved the checked TP12
row-major greedy result (2005) against the token-loop implementation. This is
well below the 120--200 tok/s target; the remaining serial causal SSM/attention
work and replicated attention on TP12 are the dominant limitation.

TP12 per-rank checkpoint state now includes SSM convolution and recurrent
state, and a 12-to-4 repartitioner reconstructs all 64 layer records. The
repartition operation completes, but the first end-to-end continuation did not
yet match direct TP4 prefill (71930 versus 2918 in the smoke test), so the
handoff is retained as an experimental path and is not claimed numerically
equivalent.

### Twelve-node PP3 x TP4 prefill prototype (2026-08-21)

An additive single-sequence pipeline runner, `qwen38_prefill_runner`, now uses
three pipeline stages with four TP lanes each and layer cuts `[0,21)`,
`[21,43)`, and `[43,64)`. Each physical node reuses the normal TP4 blob chosen
by `world_rank % 4`; explicit logical stager rank/size overrides prepare all
twelve node-local copies in one bounded, page-cache-safe launch.

The range-prefill API exchanges token-major FP32 hidden blocks between stages.
Stage-owned BF16 projections use the existing 8-token x 48-row A64FX packed
kernel with FP32 activations/accumulation. SSM preparation remains in prompt
order, followed by one head-parallel scan over each chunk while preserving
sequential recurrence within a head.

All ranks completed the live sweep in lockstep. PV48 and row-major execution
gave the same distributed next token on the checked 128-token case (`98709`).

| prompt | chunk | elapsed | throughput | next token |
|---:|---:|---:|---:|---:|
| 128 | 64 | 7.041 s | 18.18 tok/s | 98709 |
| 512 | 128 | 9.710 s | 52.73 tok/s | 98709 |
| 1024 | 256 | 12.318 s | 83.13 tok/s | 74723 |
| 4096 | 256 | 30.665 s | **133.57 tok/s** | 23145 |
| 4096 | 1024 | 27.834 s | **147.16 tok/s** | 23145 |

This improves substantially over the TP12 512-token result but does **not**
meet the full 120--200+ tok/s sweep target: it reaches the lower bound only at
4096 tokens. Parallel `(token,head)` attention cut the 4096-token result from
73.685 seconds / 55.59 tok/s to 30.665 seconds / 133.57 tok/s without changing
the next token. Stage compute at 1024 tokens was approximately
8.76/7.12/6.78 seconds; at 4096 it was 22.96/20.19/25.06 seconds. The next
required work is a packed/tiled QK/PV kernel, uTofu subgroup collectives instead
of MPI all-reduce, and measured layer cuts. TP4 decode-state handoff is not yet
implemented for this prototype and remains a production correctness gate.

#### 4K profiling and quantized experiments (2026-08-22)

The range API now accumulates phase timers and the runner writes them to each
rank log.  On the exact BF16 4096/1024 run, stage 0 spent 17.29 seconds in
compute and stage 2 spent 17.19 seconds; end-to-end time was 27.834 seconds.
The remaining gap is dominated by single-sequence pipeline fill/drain plus TP
collectives, not causal attention (about 0.94--1.11 seconds per stage).  A
double-buffered `MPI_Isend`/`MPI_Irecv` test preserved `next=23145` but produced
the same 147.16 tok/s because this MPI configuration made no useful progress
during compute, so it was removed.

`Q38_PREFILL_QUANT=int8|int16` now converts only the PP-owned BF16 projections
in place. `int8` is W8A8 SDOT; `int16` is an accurate-at-short-context W8A16
mimic (INT8 weights, INT16 activations, INT64 SDOT accumulation). They are
explicit experiments, not defaults:

| mode | prompt/chunk | throughput | next token | result |
|---|---:|---:|---:|---|
| BF16 PV48 | 4096/1024 | **147.16 tok/s** | 23145 | accepted |
| W8A8 | 128/64 | 17.79 tok/s | 33014 | divergent |
| W8A16 mimic | 128/64 | 16.50 tok/s | 98709 | short check matches |
| W8A8 | 4096/1024 | 87.38 tok/s | 42161 | slower, divergent |
| W8A16 mimic | 4096/1024 | 51.16 tok/s | 100393 | slower, divergent |

The current row-outer quantized GEMMs are rejected for performance: at 4K the
W8A8 FFN projection time grows to roughly 13.4 seconds versus 5.0 seconds for
PV48 BF16. A competitive W8 path needs an 8-row by 4--5-token register-blocked
kernel and group scales, followed by long-context quality validation. Reaching
250 tok/s with PP3 also requires eliminating the single-prompt pipeline bubble
(or changing the topology) and replacing the high-latency MPI TP reductions.

### Native Q8 12-node prefill experiment (2026-08-22)

`run_qwen38_q8_prefill_12n.sh` stages the native Q8_0 model as PP3 x TP4.
The old A64FX Q8 token-major GEMM was a row-by-token loop around the decode
dot product, repeating Q8-to-F32 conversion for every token. An exact 1-row x
4-token SVE tile now amortizes that conversion while retaining original Q8_0
block scales and F32 activations.

The launcher defaults to `Q38_PREFILL_Q8=w8a8`: each PP-owned Q8 projection is
repacked to 8-row x 64-column blocks, activations are quantized per 64 values,
and the 8-row x 3-token SVE SDOT kernel is used.

| path | prompt/chunk | elapsed | throughput | next token |
|---|---:|---:|---:|---:|
| native Q8, F32 activation | 128/64 | 100.37 s | 1.28 tok/s | 1293 |
| Q8-derived W8A8, SDOT 8x3 | 128/64 | 6.94 s | **18.45 tok/s** | 1293 |
| Q8-derived W8A8, SDOT 8x3 | 4096/1024 | 42.33 s | **96.77 tok/s** | 62842 |
| rejected SDOT 1x16 | 128/64 | 6.52 s | 19.64 tok/s | 1293 |
| rejected SDOT 1x16 | 4096/1024 | 47.90 s | 85.50 tok/s | 62842 |

The 1x16 intrinsic tile was removed because compiler register spills outweighed
its nominal weight reuse at large N. The retained 8x3 path improves the native
Q8 short run 14.4x and matches its checked next token. Long-context quality is
not established because a native-Q8 4K oracle would take too long.

The two-V100 result is not a direct aggregate-bandwidth prediction for this
implementation. PP3 has single-sequence pipeline fill/drain, TP4 performs
latency-sensitive reductions, and the current SDOT tile reloads weights once
per three tokens. A 500 tok/s A64FX design needs a spill-free assembly GEMM
with a wider token tile plus a topology that removes the single-sequence PP
bubble, rather than Q8 storage alone.

#### Native per-block Q8v2 assembly integration

The Q8 prefill launcher now defaults to `Q38_PREFILL_Q8=q8v2`. Native Q8_0
bytes are repacked without requantization into the existing A64FX
`kernel_q8v2_3x4.S` layout (3 tokens x 64 outputs); FP16 block scales are
expanded to FP32 and activations are quantized per 32 values. Original Q8_0
storage remains available for tails and A/B checks. The runner links the
assembly explicitly with `TF_HAVE_Q8V2`.

Activation packs are cached across projections sharing the same input matrix:
Q/K/V, SSM QKV/gate/alpha/beta, and FFN gate/up. This removed repeated
quantization and allocator churn without changing checked tokens.

| path | prompt/chunk | elapsed | throughput | next token |
|---|---:|---:|---:|---:|
| Q8v2 before activation reuse | 1024/1024 | 16.67 s | 61.45 tok/s | 1293 |
| Q8v2 + activation reuse | 1024/1024 | 12.77 s | **80.17 tok/s** | 1293 |
| Q8v2 + activation reuse | 4096/1024 | 26.04 s | **157.31 tok/s** | 62842 |
| Q8v2, smaller pipeline tile | 4096/512 | 36.18 s | 113.21 tok/s | 62842 |

The accepted 4K result is 62.6% faster than the retained compact-W8A8 result
(96.77 tok/s), but it does not meet 250 tok/s. Profiling still attributes most
time to projections plus TP collectives. A wider spill-free assembly token tile
(for example 6x32) is the next kernel step; 3x64 reloads a weight panel for
every three tokens.

The runner and stager accept PP3xTP4, PP2xTP6, and PP1xTP12 via
`Q38_PREFILL_TP_SIZE=4|6|12`. TP4 remains the default.

#### Wider-kernel and topology follow-up

Two additional assembly schedules were implemented and measured:

- A per-block 6-token x 32-output Q8v2 kernel is numerically correct
  (`maxrel=2.07e-5`) but reaches 187.0 GIOPS/core versus 196.6 GIOPS/core for
  3x64. Extra activation broadcasts outweigh the reduced weight-panel reads.
- `q8blk6` reuses `int8-cmg/micro_kernel_6x4_no_sector.S` once per 64-wide K
  block and applies validated per-64 scales outside the integer kernel. It
  preserves checked tokens but reaches 151.02 tok/s at 4096/1024, below Q8v2.

The existing `kernel_q8v2_3x4_arow` path is exposed as `q8v2_arow`. It reaches
162.33 tok/s at 4096/1024 but changes the long-context next token from 62842 to
1293, so the coarser activation scale is rejected by the quality gate.

TP12 initially corrupted the heap at its first attention layer. Replicated
attention has `qdim=5120`, while its temporary gate buffer was incorrectly
sized to the rank-local FFN width. Sizing scratch to `max(qdim, local_ff,
local_ssm)` fixes the abort. Query heads and output-projection columns are now
sharded even when the four KV heads must be replicated; the batched attention
worker also receives the global query-head offset.

Live corrected topology results for Q8v2 at 4096/1024:

| topology | throughput | attention time/rank | next token |
|---|---:|---:|---:|
| PP3 x TP4 | **157.31 tok/s** | 0.95--1.14 s | 62842 |
| PP2 x TP6 | 141.96 tok/s | 3.23--3.25 s | 9587 |
| PP1 x TP12 | 126.14 tok/s | 3.48 s | 1 |

Native Q8 and Q8v2 agree within TP12 on the 128-token check (`next=1`), so the
packed kernel passes the within-topology quality comparison. TP6/TP12 differ
from TP4 because their replicated-KV partition and reduction order are not
greedy-equivalent. TP4 remains both the fastest and the accepted topology.

#### Padded assembly tails and 250+ tok/s result

The largest remaining regression was not the main assembly loop. Q8v2 used
`N/3` complete microkernel tiles and sent every one- or two-token remainder to
the native Q8/F32 GEMM. Thus common chunk sizes 256, 512, 896, and 1024 invoked
the extremely slow fallback in every projection and every layer. Multiples of
three such as 768 appeared anomalously fast.

The dispatcher now rounds the tile count up, zero-pads missing activation rows,
runs one final `kernel_q8v2_3x4` tile, and copies only valid output rows from a
small temporary tile. Quantization and valid-row arithmetic are unchanged.
This preserves both checked tokens: 1293 at 128 and 62842 at 4096.

The scale-out schedule was also reordered from twelve adjacent
`scvtf -> fmul -> fmla` chains into conversion/multiply/accumulate waves. The
L1-resident microbenchmark improves from 1835 to 1784 cycles (2.8%); K=5120 is
approximately flat because panel traffic dominates.

Final PP3 x TP4 Q8v2 sweep at 4096 tokens:

| chunk | elapsed | throughput | next token |
|---:|---:|---:|---:|
| 1024, old native tail | 26.04 s | 157.31 tok/s | 62842 |
| 768, padded tail | 19.13 s | 214.14 tok/s | 62842 |
| 512, padded tail | 16.25 s | 252.09 tok/s | 62842 |
| 384, padded tail | 15.38 s | 266.31 tok/s | 62842 |
| **256, padded tail** | **14.74 s** | **277.80 tok/s** | **62842** |
| 192, padded tail | 14.84 s | 275.93 tok/s | 62842 |

Chunk 256 is the measured optimum and is now the Q8 launcher default. It is
2.87x the earlier compact-W8A8 result (96.77 tok/s), 1.77x the first accepted
Q8v2 result (157.31 tok/s), and exceeds the requested 250 tok/s target.

#### Post-restart SSM optimization and practical ceiling

After restaging the wiped node-local TP4 shards, the 4096/256 baseline reproduced
at 278.44 tok/s once the session was warm. Two quality-preserving SSM changes
then reduced non-GEMM time:

- Depthwise causal convolution is evaluated as one channel-parallel batch per
  SSM layer. Each worker walks prompt time in strict order for its channels and
  writes the final circular history back in the original layout. This removes
  the single-core token-by-token convolution without changing recurrence order.
- Convolution SiLU and the recurrent output gate use the existing A64FX SVE
  FEXPA sigmoid approximation instead of scalar `expf` loops.

Both 128-token (`next=1293`) and 4096-token (`next=62842`) gates remain stable.
The accepted live result is now:

| prompt/chunk | elapsed | throughput | next token |
|---:|---:|---:|---:|
| **4096/256** | **13.924 s** | **294.17 tok/s** | **62842** |

Stage compute is balanced at 11.73/11.97/11.53 seconds. Chunk 224 was slower
(288.67 tok/s) because increased send/range-call overhead outweighed its smaller
pipeline bubble. `OMP_PROC_BIND=close` was neutral versus `spread`.

Based on the measured phase floor, the practical ceiling of the current exact
PP3xTP4/Q8v2 dataflow is approximately 310--330 tok/s. Reaching 350--450 tok/s
requires lower-cost TP collectives and/or a wider microkernel that retains
per-group activation scales. 500 tok/s remains an aggressive redesign target,
not a plausible outcome from chunk or thread-placement tuning alone.

#### Batched DeltaNet preparation and BF16 follow-up

The post-convolution DeltaNet preparation no longer round-trips every prompt
row through the model's single-token scratch buffers.  Alpha/beta transforms,
Q/K normalization, head expansion, and V extraction are token-parallel, while
the recurrent scan remains sequential within each head.  Both Q8 gates remain
unchanged (`next=1293` at 128 and `next=62842` at 4096).

On the restarted allocation the warmed Q8 MPI baseline was 283.75 tok/s.  The
batched preparation reduced critical-stage `ssm_prepare` from 1.5--1.7 seconds
to 0.72 seconds and produced **310.96 tok/s** at 4096/chunk256.  Nearby chunks
192, 224, 288, and 320 reached 293.14, 290.03, 305.28, and 300.20 tok/s;
chunk256 remains the accepted default.

Two uTofu prefill collectives were implemented behind `Q38_PREFILL_COMM`:
the existing full-buffer recursive-doubling tree and a TP4 direct
reduce-scatter/all-gather using three TNIs.  Both preserved the Q8 tokens, but
were slower than Fujitsu MPI for the 5 MiB payload: 272.90 and 269.46 tok/s,
respectively, versus 283.75 for the matched pre-change MPI run.  MPI therefore
remains the default; `utofu-tree` and `utofu` are retained for transport A/Bs.

The safely restaged exact BF16 PV48 path improved from the old 147.16 tok/s to
**205.53 tok/s** at 4096/chunk256, preserving `next=23145`.  Chunk1024 now
reaches only 136.67 tok/s because its large MPI reductions cost 6.5 seconds on
the critical stage.  Exact BF16 is still bounded by FP32 activation/accumulation
and did not approach 300 tok/s.

An explicitly experimental `Q38_PREFILL_BF16=bf16-act` mode links the existing
Clair `sgemm_bf16_2x12.S`, pre-packs only the owned pipeline range in place, and
uses BF16 activations with FP32 accumulation.  Its natural 12-token tile favors
chunk252 and reaches **305.98 tok/s** at 4096 tokens.  It produces `next=35349`
instead of the exact BF16 `23145`, so it is not an accepted exact path and is
never selected by default.  Use `Q38_PREFILL_BF16=exact` for the canonical
model and `bf16-act` only for labeled throughput/quality experiments.

#### Eight-node PP2 x TP4 topology

The runner and launcher also accept `Q38_PREFILL_NODES=8`, retaining TP4 and
using two pipeline stages.  This is numerically preferable to inventing a TP3
partition: the native Q8 short and 4K gates remain `1293` and `62842`.

At 4096 tokens, the Q8 chunk sweep with the initial 32/32 cut measured 227.68,
219.25, and 211.58 tok/s for chunks 256, 384, and 512.  Sweeping the layer cut
at chunk256 found 31/33 best:

| nodes/topology | cut | chunk | throughput | next |
|---|---:|---:|---:|---:|
| 8 / PP2xTP4 | 30/34 | 256 | 223.19 tok/s | 62842 |
| **8 / PP2xTP4** | **31/33** | **256** | **231.15 tok/s** | **62842** |
| 8 / PP2xTP4 | 32/32 | 256 | 227.68 tok/s | 62842 |
| 8 / PP2xTP4 | 33/31 | 256 | 228.81 tok/s | 62842 |
| 8 / PP2xTP4 | 34/30 | 256 | 220.75 tok/s | 62842 |

The PP2 default cut is therefore 31.  It is more node-efficient than the
12-node 310.96 tok/s result (28.89 versus 25.91 tok/s per node), but slower in
absolute single-prompt throughput because each stage executes about 50% more
layers.  Exact BF16 at the same eight-node 31/33, chunk256 configuration was
120.22 tok/s with `next=23145`, so BF16 does not reverse the conclusion.

#### Fused DeltaNet state dots

The A64FX recurrent scan can now calculate each state row's `state*K` and
`state*Q` reductions together, sharing the state load.  After the unchanged
rank-one state update, the output is formed by the equivalent identity
`old_state*Q + delta*(K*Q)`.  The Q8 launcher enables this with
`TF_SSM_FUSED_DOTS=1`; setting it to zero restores the previous two-pass scan.

The 128-token gate remains `next=1293`, and repeated 4096-token runs retain
`next=62842`.  Scan time on rank 0 fell from roughly 1.33 seconds to
1.08--1.09 seconds.  The best live 4096/chunk256 result is **321.36 tok/s**
(12.746 seconds), up from the accepted 310.96 tok/s result.  End-to-end repeats
vary with TP collective and pipeline handoff time (one repeat was 307.76
tok/s), so 321.36 is a best observed result rather than a sustained floor.

Forcing Open MPI's generic tuned collectives did not improve the 5 MiB TP
reduction: Rabenseifner reached 241.24 tok/s, while ring and recursive doubling
reached 296.47 and 296.03 tok/s.  Fujitsu's default mtofu collective remains
selected.  The approximate scalar SSM transforms (`TF_SSM_FAST_SCALARS=1`)
also provided no repeatable end-to-end gain and remain opt-in.

#### Post-restart bulk SiLU optimization

Allocation 50725516 was restaged with the bounded Q8 TP4 builder after `/local`
was wiped. The exact 128-token gate reproduced at `next=1293`, and the warm
4096/chunk256 fused-SSM baseline reproduced at 317.51 tok/s with `next=62842`.

The A64FX FFN SiLU-multiply pass previously fell through to a scalar `expf`
loop. `TF_SILU_SVE=1` now evaluates the full contiguous pass with SVE FEXPA,
one reciprocal estimate plus Newton refinement, and parallel static chunks.
The Q8 launcher enables it by default. Rank-0 FFN activation time fell from
about 533 ms to **53--58 ms** while retaining both exact next-token gates.

| prompt/chunk | elapsed | throughput | next token |
|---:|---:|---:|---:|
| 4096/256 | **12.288 s** | **333.33 tok/s** | **62842** |
| 4096/256 repeat | 12.616 s | 324.68 tok/s | 62842 |

An attempted gate/up macro-fusion was rejected. Although it removed the
separate activation pass, interleaving the two large packed weight panels
destroyed the static schedule's cache locality: rank-0 FFN projection time rose
from about 1.6 seconds to 16.1 seconds and throughput fell to 140.24 tok/s.
The implementation was removed rather than retained as a misleading option.

#### Exact-path follow-up after 333 tok/s

The next two non-projection experiments did not justify changing the default:

- A fixed-scale signed-16 TP allreduce reduced the 4K collective phase from
  roughly 1.49 seconds to 1.15 seconds and reached 343.91 tok/s. Scale 4096
  already failed the short gate (`next=67577`); scale 16384 passed the short
  gate but failed the long gate (`next=141330`). The implementation was removed.
- Moving DeltaNet decay `expf` from the head-serial scan into token-parallel
  preparation preserves `1293`/`62842`, but scan time remained 1.08 seconds and
  throughput was 331.25 tok/s. It remains opt-in as `TF_SSM_PREEXP=1`.
- Explicit SVE for the attention output sigmoid also preserves both gates. A
  335.23 tok/s run was observed, but the measured attention phase was unchanged
  at about 0.96 seconds, so this is treated as run variance and remains opt-in
  as `TF_ATTN_GATE_SVE=1`.

The exact 12-node default therefore remains Q8v2 + fused DeltaNet dots + bulk
SVE FFN SiLU, with **333.33 tok/s best accepted** and 324.68 tok/s repeat.

#### BF16 prefill optimization on the restaged allocation

The BF16 TP4 shards were rebuilt on allocation 50725516 with the bounded
rank-local stager. Exact BF16 retains FP32 activations and accumulators and
continues to produce `next=23145` at 4096 tokens (`next=98709` at 128).

The generic BF16 launcher now enables the datatype-independent fused DeltaNet
dots and bulk SVE SiLU defaults. At 4096/chunk256 this raised the best exact run
from the cold 171.39 tok/s baseline to **231.15 tok/s**:

| BF16 mode | prompt/chunk | throughput | next token | status |
|---|---:|---:|---:|---|
| exact PV48, cold baseline | 4096/256 | 171.39 tok/s | 23145 | exact |
| **exact PV48, optimized best** | **4096/256** | **231.15 tok/s** | **23145** | **accepted** |
| exact PV48, later repeat | 4096/256 | 205.71 tok/s | 23145 | exact; MPI noisy |
| BF16-activation `2x12` asm | 4096/252 | **329.56 tok/s** | 35349 | experimental |

PV48 now reuses its persistent activation pack for consecutive Q/K/V and
gate/up projections when `N<=256`. This preserves multiplication and
accumulation order. On a balanced run it reduced rank-0 projection compute by
about 0.66 seconds, although end-to-end repeats were dominated by collective
variance: rank collective time ranged from roughly 1.84 to 3.85 seconds and
could move the whole run between 183.91 and 231.15 tok/s.

The `Q38_PREFILL_BF16=bf16-act` path benefits from the same SSM/SiLU work and
improves over its previous 305.98 tok/s result to 329.56 tok/s. It quantizes
activations to BF16 for `sgemm_bf16_2x12.S`, changes the 4K next token, and is
therefore reported separately rather than replacing exact PV48.

#### Decode revisit on allocation 50725516

The restaged BF16 TP4 decode remains the fastest real single-stream
configuration. With `TP_STAGE_BF16_PV=1`, 48 spread workers, ordinary barriers,
`TP_MAXSEQ=128`, eight warmup tokens, and sixteen measured tokens, the live run
reported **25.21 tok/s** including all 24 generated tokens. The post-warmup
forward interval was 36.09 ms/token: 31.63 ms compute plus 4.46 ms communication,
or **27.71 forward tok/s**. The phase profile attributes about 15.37 ms to FFN,
9.95 ms to SSM projections, and 2.12 ms to attention projections.

No exact 40+ tok/s configuration was found:

- TP6 and TP12 remain slower because four KV heads cannot be evenly sharded;
  their recorded rates are 21.12 and 18.47 tok/s.
- Native Q8 TP4 remains around 6--7 tok/s and is not competitive with BF16.
- Source-row-major BF16 with the existing exact 8-row SVE kernel reached only
  23.14 tok/s overall and 40.91 ms/token post-warmup, so the integration was
  removed and PV8 retained.
- `TF_SSM_SVE2=1` reached 24.94 tok/s overall and 38.93 ms/token post-warmup,
  below the default.
- Current NextN drafting has measured alpha zero and still performs one trunk
  scan per generated token, so `TP_SPEC_K` is not a speculative speedup.

Forty tok/s requires at most 25 ms/token. The live default already spends
4.46 ms in 129 reductions and streams 14.31 GB of weights per rank at an
effective 452 GB/s. Meeting 25 ms would require roughly 697 GB/s for all
projection work after communication, close to the 828 GB/s load-only stream
ceiling and well above the measured compute kernel. The viable route to 40+
accepted tokens/s is therefore batched speculative verification with a useful
draft model, not another existing single-token environment configuration.
### Single-token BF16 PV8 bandwidth attack (2026-08-22, job 50725516)

The production-exact low/high reduction kernel was added to
`a64fx/tools/mv_bench_bf16_8row.c` as variant 4.  With four-CMG first-touch and
48 pinned workers, the no-prefetch kernel reached 593/720/772 GB/s at
K=4352/5120/6144.  L2 software prefetch lifted these to 759/833/851 GB/s at
distance 8 chunks (distance 12 was slightly better only at K=4352).  The raw
stream roof in the same harness was 919 GB/s at K=5120.

`TF_BF16PV_PREFETCH` now accepts a chunk distance (legacy value `1` maps to 8),
and the TP4 BF16 launcher defaults to the measured distance 8.  Exact TP4 A/B,
16 measured tokens after 8 warmups:

- old default: 36.09 ms forward, 31.63 ms compute, 4.46 ms communication,
  452 GB/s effective, 25.21 tok/s overall;
- distance 8: 31.64 ms forward, 26.65 ms compute, 4.98 ms communication,
  537 GB/s effective, 30.58 tok/s overall.

The emitted token stream was identical.  Distance 4 was slower (32.40 ms,
516 GB/s), and alternating gate/up PV groups was also slower (32.03 ms,
519 GB/s), so the latter experiment was removed.  The isolated kernel exceeds
700 GB/s without token batching; the remaining model-level loss comes from 129
short projection/collective phases per token rather than the steady-state
microkernel.

### TP4 BF16 decode and MTP follow-up (2026-08-23, job 50782148)

The four BF16 shards were restaged and verified on fresh node-local storage.
A 128-token exact baseline produced token SHA256
`38630a0a8022e55e8424680855f0dcdb44f7550207ea1788f0211260e0cd4bd1`.
It reached 27.24 tok/s cold; after 32 warmup tokens the default distance-8 run
reported 34.02 ms/token on rank 0 (28.33 ms compute plus 5.69 ms communication),
or 29.20 tok/s including warmup.  The requested 33 tok/s sustained decode target
was therefore not met.

The exact low/high reduction microkernel remains bandwidth-limited by the
projection shape.  At 2.0 GHz, distances 8/10/12 measured 754/772/755 GB/s for
K=4352, 826/836/824 GB/s for K=5120, and 844/850/845 GB/s for K=6144.  The best
result was **850.36 GB/s**, below the 880 GB/s target but 92.6% of the harness's
919 GB/s stream ceiling.  Model-level prefetch and collective sweeps sometimes
changed the greedy stream, so no faster setting replaced the exact distance-8
default.  In particular, the fastest short distance-10 run reached 32.31
ms/token but failed the token gate.

The TP NextN path now slices its optional private vocabulary head per rank,
applies the trunk output RMSNorm when a private draft-head norm is absent, and
right-shifts prompt hidden states while warming the draft cache.  This cuts
NextN materialization from 2.774 GB to 0.866 GB per node and prevents the draft
head from collapsing to tiny local IDs.  However, the current draft remains
misaligned with the trunk (`MTP greedy match=0/4` in the traced validation), so
there is no honest accepted-token speedup and the 50 tok/s MTP target is not
met.  `run_qwen38_bf16_tp4.sh mtp-check` reproduces the greedy-agreement probe;
normal `bench` keeps `TP_SPEC_K=0`.  Batched verification must remain disabled
until this probe reports nonzero, stable agreement against the exact stream.

### TP4 BF16 MTP status and remaining work (2026-08-24)

This section supersedes the zero-agreement conclusion immediately above.  The
NextN alignment, recurrent rollback, and batched verifier are now functional
and greedy-exact on four A64FX nodes.  Rank-local BF16 shards are staged under
`/local/u14346/qwen38-bf16-tp4` and uploaded into anonymous HBM2; no weight mmap
is used.  Each rank holds about 17.72 GB of staged weights, remaining safely
below the 32 GB/node limit.

The long-context performance gate duplicates `tmp/qwen38_mtp_prompt.txt` for a
359-token prompt and generates 64 tokens with `TP_IGNORE_EOS=1`.  The accepted
oracle is `/local/u14346/build-tmp/q38-longctx-oracle64.tokens`, SHA256
`17b19d8bed7200ee15f147858e55841c468d0343635077b13acd219b13b8e701`.
All accepted MTP results below match that file byte-for-byte.

| mode | draft agreement | verifier | draft | result |
|---|---:|---:|---:|---:|
| trunk, K=0 | n/a | n/a | n/a | 28.35 tok/s |
| MTP K=3 | 42/44 (0.955) | 59.51 ms/round | 9.21 ms/round | 39.05 tok/s |
| MTP K=4 | 49/51 (0.961) | 65.95 ms/round | 13.91 ms/round | **42.70 tok/s** |
| MTP K=5, representative | 53/56 (0.946) | 75.13 ms/round | 19.75 ms/round | 42.55 tok/s |
| MTP K=5, best profiled | 53/56 (0.946) | 72.56 ms/round | 17.98 ms/round | **44.57 tok/s wall; 50.48 forward tok/s** |

K=5 horizon agreement was `13/14, 13/14, 14/14, 13/14`; the draft is useful
and non-degenerate.  The best profiled run reached 826 GB/s/node aggregate
effective bandwidth.  Runs vary enough that 50.48 forward tok/s is a best
observation, not a sustained 50 tok/s wall result.  The requested 50 tok/s
decode gate and 70 tok/s final gate therefore remain open.

Implemented foundations include:

- snapshot-free selection of the committed recurrent state and direct writes
  into rollback slots;
- removal of redundant initial/final NextN proposals;
- native exact BF16 PV verifier kernels for K=2 through K=5, including split
  4-row x 3-token and native 4-row x 4/5-token SVE kernels;
- fused DeltaNet state/output dots and persistent-pool NextN attention;
- parking the OpenMP verifier team while the pthread NextN pool drafts;
- one batched TP argmax collective for all verifier columns;
- optional `TP_BUFFER_OUTPUT=1` to avoid synchronous filesystem flushes in a
  benchmark while preserving interactive streaming by default.

#### Remaining tasks, in priority order

1. **Replace the serial linear draft critical path.**  K=5 performs four
   dependent NextN calls and spends about 18--20 ms/round drafting.  Implement
   tree or asynchronous drafting so independent candidates/heads run together,
   or overlap useful draft work with verifier completion.  First inspect the
   GGUF NextN tensors to determine whether multiple independent prediction
   heads exist; do not emulate a tree by duplicating the same recurrent head.
   Preserve exact verification and rollback for every selected branch.

2. **Remove verifier/draft synchronization bubbles.**  The K=5 verifier is
   about 73 ms/round and its collective phase is about 11 ms/round.  Pipeline
   rank-local projection completion with TP reductions where dependencies
   permit, retain the batched verifier argmax, and reduce persistent-pool
   handoffs between trunk and NextN execution.  Draft argmax reductions remain
   sequential when each predicted token feeds the next NextN call.

3. **Raise batched PV bandwidth from 826 toward 880 GB/s/node.**  Inspect the
   compiler output of `matvec_bf16_4x5_pv` for spills and load/FMA scheduling,
   then benchmark a hand-scheduled assembly version under `/local`, not in the
   repository or `/tmp`.  Sweep per-shape prefetch carefully: distance 16 was
   best for K=4/K=5; 0, 8, and 24 were slower.  Validate the full token oracle,
   because numerically different packing/reduction orders can sharply reduce
   draft acceptance even when final verified tokens remain exact.

4. **Measure decode wall time at the actual decode boundaries.**  The current
   `t_total - t_prefill` report includes setup, final barrier/reporting, and
   token-output costs that are outside `t_fwd`.  Add explicit decode-loop start
   and stop timestamps and report both compute/communication throughput and
   user-visible streaming throughput.  This is measurement cleanup, not a
   substitute for meeting the wall target.  `TP_BUFFER_OUTPUT=1` improved one
   comparable wall run only from 42.55 to 43.26 tok/s.

5. **Validate sustained 50 tok/s before pursuing 70.**  Use at least 256
   generated tokens, repeat three times, and require identical greedy tokens.
   For the current K=5 acceptance (64 tokens in 14 rounds), 50 tok/s needs at
   most 1.28 seconds total, roughly 11 ms/round less than the best 1.436-second
   wall run.  At the same 4.57 emitted tokens/round, 70 tok/s requires a round
   below about 65 ms, versus the current roughly 91 ms.  Reaching 70 therefore
   needs substantial verifier/draft overlap or a wider accurate tree, not a
   small prefetch adjustment.

6. **Only deepen linear K after a native-kernel model.**  K=4 and K=5 have
   nearly identical wall throughput despite excellent horizon-4 accuracy;
   added verifier compute and another serial draft cancel the extra accepted
   token.  Do not extend arrays to K=6+ without first predicting round cost and
   supplying a native no-reread verifier kernel.

#### Rejected or non-default experiments

- Packing the replicated NextN matrices into the trunk PV layout preserved the
  verified output but changed draft numerics: agreement fell from about 0.95 to
  0.58 and throughput to 28.49 tok/s.  It was reverted.
- Active OpenMP workers competing with the NextN pthread pool made drafting
  catastrophically slow.  Keep the verifier-team parking protocol and the
  launcher defaults `OMP_WAIT_POLICY=active`, `KMP_BLOCKTIME=1`, and
  `TP_MTP_OMP_PARK=1` unless a complete scheduler replacement is measured.
- Approximate SSM scalar/pre-exp combinations can produce degenerate, non-exact
  streams and misleading throughput.  They are not accepted optimizations.
- Increasing linear depth alone is not the route to 70 tok/s; K=5 already
  demonstrates the diminishing-return point.

Relevant commits, newest first: `9d169b59` (buffered benchmark output),
`73643277` (batched verifier argmax), `18fe8553` (K=4/K=5 verifier kernels),
`f8a9b471` (parallel NextN attention), `65c1fd65` (fused MTP SSM dots), and
`79ff4e37` (split K=3 verifier kernel).

### Sustained 50 tok/s milestone and post-restart MTP work (2026-08-24)

The first target is now met.  Commit `a746072a` moved the decode wall timer to
the actual loop boundaries, enabled the exact SVE SiLU path, and retained only
the useful packed NextN hidden-fusion projection.  With the same 359-token
long-context prompt, K=5 produced three exact 64-token repeats at
52.13/52.12/51.42 tok/s.  The stronger sustained gate generated 256 tokens in
4.790 seconds, or **53.43 tok/s**, byte-identical to the K=0 oracle:

- oracle SHA256: `7b86e9830096198c4066689d487ad18b3cd6efbad02626494a0d3fb9460d2f14`;
- 55 verifier rounds, 68.79 ms verify plus 18.26 ms draft per round;
- greedy agreement 205/220, alpha 0.9318;
- horizon agreement 53/55, 52/55, 53/55, 47/55;
- rank-0 forward 4.7888 seconds and reported 868 GB/s/node effective bandwidth.

After the session restart, the four rank-local blobs were restaged under
`/local/u14346/qwen38-bf16-tp4`.  Each node uploads 17.724 GB into anonymous
HBM2 from metadata-only GGUF input; rank 0 retained 12.67 GB `MemAvailable`
after upload.  No weight mmap or llama.cpp path is involved.  A clean rebuild
reproduced the 256-token oracle at 53.45 tok/s and 870 GB/s/node.  A later noisy
64-token default run was exact at 49.86 tok/s; the sustained result above remains
the headline rather than selecting the best short repeat.

This work also adds a weights-sharing NextN runtime context.  It owns private
NextN KV, hidden/fusion buffers, transformer scratch, logits, and a pthread
pool, while sharing the immutable anonymous staged weights and RoPE tables.
`TP_MTP_SHADOW_THREADS` selects it for the sequential correctness path, and
`TP_MTP_SHADOW_CORE_OFFSET` can place a future background pool on a reserved
A64FX CMG.  The isolated 48-thread context reproduced the exact 64-token oracle,
53/56 agreement, and 51.00 tok/s.  It adds only runtime scratch/state, not a
second 17.7 GB weight arena.

#### New rejected experiments

- A separate second-TNI/tag-8 communicator plus background five-step lookahead
  retained the rank-0 exact output, but concurrent traffic/compute raised the
  verifier from about 69 ms to 282.68 ms and reduced throughput to 12.12 tok/s.
  Prefetched tails were not lockstep-identical across ranks.  The async runner
  path and communicator were removed; only the safe context isolation remains.
- Scoring the already-replicated `token_embd.weight` locally avoided the draft
  collective, but Qwen3.8 has a distinct trained output head.  Agreement
  collapsed to zero and rank-local numerical drift produced different draft
  chains.  This path was removed.
- The safe generic BF16 PV K=6 verifier was greedy-exact, unlike the earlier
  rejected specialized 4x6 kernel, but reached only 45.54 tok/s: 93.97 ms
  verification plus 23.08 ms drafting per round.  K=6 exposure was reverted.
- Selective packed or W8A8 NextN attention/output/down/QKV/FFN variants were
  neutral, slower, or damaged agreement.  Only packed `nextn.eh_proj.weight`
  remains enabled.  Prefetch distance 18 was also noisy and lost to the default
  distance 16 on the sustained gate.

#### Remaining decode and MTP tasks

1. **Design overlap around the one real recurrent NextN head.**  GGUF reports
   `nextn_predict_layers=1`; there are no independent native tree heads.  Keep
   the new private runtime, but do not duplicate the same recurrence and call it
   a tree.  A useful pipeline must precompute beyond the current queue and reuse
   it only after both the verified prefix and predicted correction match.

2. **Give background drafting deterministic tokens without a concurrent
   all-reduce.**  The most promising design is to stage the trained full
   `output.weight` only where needed (the embedding matrix is not equivalent),
   let a designated draft rank select tokens, and send each tiny token on an
   isolated one-way channel.  Other ranks must consume the same token before
   advancing their replicated NextN state.  Measure this transport alone before
   reconnecting it to verification.

3. **Partition cores/CMGs instead of oversubscribing 48+draft workers.**  Use
   `TP_MTP_SHADOW_CORE_OFFSET=36` for a 12-core CMG3 draft pool and benchmark the
   verifier with 36 reserved trunk workers.  Sweep 8/12/16 draft workers and
   32/36/40 verifier workers.  Require rank-identical pending queues and compare
   verifier time against the 68.79 ms sustained baseline; overlap that slows the
   verifier more than the hidden draft time is a rejection.

4. **Reduce verifier time below its current theoretical ceiling.**  K=5 emits
   256/55 = 4.65 tokens/round.  Even zero-cost drafting with a 68.79 ms verifier
   yields only about 67.6 tok/s, so the 70 tok/s gate also needs roughly 2--3 ms
   removed from verification or a slightly higher emitted-token ratio.  Profile
   the 4x5 PV kernel for spills, preserve its exact accumulation order, and
   attack the 10.2 ms/round collective phase without changing greedy tokens.

5. **Validate every scheduler change at 256 tokens before acceptance.**  Require
   the SHA above, identical pending tokens on all four ranks, at least three
   repeats, 55-ish rounds with alpha near 0.93, and memory below 32 GB/node.
   Report decode-loop wall time separately from setup/final barriers.  The final
   goal remains sustained **70+ tok/s**; the current accepted result is
   **53.43 tok/s** and 870 GB/s/node.

6. **Further improvement opportunities after overlap is stable.**  Fuse the
   local NextN head projection with greedy maximum reduction to avoid writing
   62,080 logits; pipeline verifier projection completion with its TP reduction;
   batch or piggyback the tiny draft-token messages; and test a hand-scheduled
   exact 4x5 assembly kernel using build scratch under `/local`, never `/tmp`.

### BF16 256-token revalidation and MTP opportunity probes (2026-08-24)

The sustained gate was rerun after the Q8 cleanup using the anonymous BF16 TP4
stage (`17.724 GB/rank`) and a fresh K=0 oracle.  The shared runner's
`TP_PROMPT_REPEAT` option was restored because the BF16 long-context gate uses
it; removing it as part of a Q8-only cleanup had been a regression.  The
apparent 367-token prompt and low agreement reported initially were caused by
omitting `TP_RAW_PROMPT=1`, which wrapped the benchmark paragraph in the chat
template.  They were not evidence of prompt drift.

For reference, the obsolete chat-wrapped diagnostic's K=0 oracle and all three
deterministic K=5 runs had SHA256
`ddc0237f13a7d712e523ac97ef67ffe76564f5a31a1098f07dff0d3e6e94df6d`.
Its MTP runs reached 37.42, 38.19, and 37.09 tok/s, using 72 rounds with alpha
0.715--0.719.  The plain K=0 run was 29.06 tok/s.  All ranks loaded the complete
stage into anonymous HBM; worst reported post-load `MemAvailable` was 11.41 GB.
The draft token reductions themselves are global, so every rank consumes the
same pending queue.  Logs and token files are retained under
`/local/u14346/q38-bf16-mtp-gate`.

Two item-8 candidates were measured and rejected:

- A BF16 NextN head-projection/greedy-max fusion avoided writing and rescanning
  62,080 local logits, but disturbed the initial draft/verifier handoff.  It
  inserted a wrong first token, reduced alpha to 0.695, and slowed to 36.55
  tok/s.  The code was fully reverted.
- The robust non-deterministic reduction topology reduced accumulated
  collective time from about 1.11 s to 0.74 s and reached 39.17/38.99/39.16
  tok/s.  Only the first run matched the oracle; the next two agreed with each
  other but diverged later in the verified stream.  Deterministic reduction
  remains mandatory for the BF16 exact path.

The 4x5 BF16-PV kernel was also compiled to assembly under `/local`.  It has a
144-byte scalar/callee-save frame but no SVE accumulator spills: all twenty
accumulators remain in registers.  A hand assembly rewrite therefore has no
obvious spill-removal win; its remaining opportunity is instruction scheduling
and prefetch, with the existing exact distance 16 as the baseline.  The next
meaningful scheduler attack must preserve deterministic reduction ordering and
fix the draft/verifier queue boundary before attempting fused local argmax.

The reproducible raw-prompt gate is now a named launcher mode and its prompt is
tracked as `a64fx/llm/qwen38_mtp_prompt.txt`:

```sh
TP_SIZE=4 ./a64fx/llm/run_qwen38_bf16_tp4.sh stage
TP_SIZE=4 ./a64fx/llm/run_qwen38_bf16_tp4.sh mtp-sustained
```

It sets raw prompting, repeats the paragraph twice, and uses K=5, batch verify,
`TP_MAXSEQ=768`, and 256 generated tokens.  A 2026-08-24 rebuild of the exact
source produced a K=0 oracle at 27.83 tok/s and three K=5 runs at 52.66, 52.56,
and 52.80 tok/s.  Every run produced token SHA256
`7b86e9830096198c4066689d487ad18b3cd6efbad02626494a0d3fb9460d2f14`;
K=5 used 55 rounds with alpha 0.932--0.936.  Post-load `MemAvailable` remained
11.8 GB or higher.  An experimental K=8 verifier was rejected: even with a
dedicated 2x8 PV kernel it changed the 64-token target stream and reached only
20.55 tok/s, so the supported exact maximum remains K=5.

### BF16 MTP coding-agent quality gate (2026-08-24)

The accepted BF16 TP4/K=5 path was exercised with the tracked
`a64fx/llm/qwen38_coding_quality_prompt.txt`: a request for one self-contained
C++20 durable key-value database, its CLI, and extensive recovery/randomized
tests.  The run used anonymous HBM weights, `TP_MAXSEQ=16660`, and generated
exactly 16,384 tokens.  HBM headroom stayed stable at about 10.2 GB.

The workload is substantially harder for the draft model than the repeated
performance paragraph.  The complete run used 4,298 verification rounds,
alpha 0.7926, 95.19 ms verification and 40.12 ms drafting per round, and took
581.816 seconds: **28.16 tok/s**.  A like-for-like 512-token screen measured
28.02 tok/s at K=0 and 34.89 tok/s at K=5 (alpha 0.6966).  Both 512-token files
had SHA256 `c74323766ad0844800abf7b24af78ccbe8f7e39fbfa71a56bf1f887cd7ea0704`,
proving that verified MTP did not alter coding output.

Quality did not pass.  The model emitted a coherent 1,573-line/54,045-byte
partial implementation with WAL, checksums, ordered versioned index,
compaction, snapshots, and many tests, but exhausted 16,384 tokens in the
middle of `test_prefix_scan_deleted`; it never emitted `main` or
`CEDARDB_COMPLETE`, and it never produced EOS.  Compilation also found defects
that precede truncation: missing `FileDescriptor::fd_`, an invalid generic
`Result<void>`, and move-only values returned through a const accessor.  The
artifact therefore does not compile and is not suitable as a one-shot 16K
coding-agent result.  Logs, extracted source, token IDs, and compiler output are
under `/local/u14346/q38-coding-quality`.

This test also exposed a deployment hazard: `/local` is node-local, so an
unstaged prompt file can tokenize differently on each TP rank and desynchronize
collectives.  The runner now checks prompt length and a token hash across ranks
before prefill and fails coherently on mismatch.  Shared repository prompt
paths or explicit per-node staging are required.

### Q8 TP2--TP4 decode and MTP attack (2026-08-24)

Qwen3.8 Q8_0 now has complete anonymous-HBM rank stages for TP2, TP3, and TP4.
The launcher reads model metadata only, uploads the selected rank blob into a
System-V anonymous arena, and never mmaps weights.  The source GGUF is
`/home/u14346/models/qwen38/27b/Qwen3.8-27B-Q8_0.gguf`; session-local rank
stages are under `/local/u14346/qwen38-q8-tp{2,3,4}`.

| topology | rank-0 stage | short native-Q8 decode | result |
|---|---:|---:|---|
| TP2 | 15.959 GB | 8.44 tok/s | too much weight traffic per rank |
| TP3 | 11.808 GB | 11.17 tok/s | replicated four-head KV attention |
| TP4 | 9.421 GB | **14.09 tok/s** | best topology in the requested range |

The TP3 staged path keeps the small NextN block replicated, so its four KV
heads no longer incorrectly reject a three-rank launch.  Empty `TP_Q8_MODE`
also no longer materializes a second Q8 copy; that bug was the cause of the
earlier TP2 OOM.  All three stages and runs remained below 32 GB/node.  TP4 is
the decisive 2--4-node choice: adding ranks reduces the per-node streamed
weight set more than the extra collectives cost.

On the 188-token long-context prompt, native Q8 TP4 generated 64 exact tokens
at **13.39 tok/s**.  The token SHA256 is
`03fdeecf744db006497cd1bc12ca5a065a95d61c3c38f570293641f3941afd7d`.
The same stream is used as the Q8 oracle below.  A 32-token warm run reached
14.09 tok/s with SHA256
`79c41776ee6b7caf5fd210dff14aa9e9375f914c0b0d669bbf3d626940b80498`.

#### Native-Q8 MTP result

The generic small-N verifier used to create and join 48 pthreads for every Q8
projection.  Reusing the hot pinned OpenMP team reduced a 16-token K=5 run from
49.75 seconds to 1.42 seconds without changing its SHA.  Packing the trunk for
the existing exact Q8v2 3x4 verifier then produced:

- 64 exact tokens in 2.990 seconds, or **21.40 tok/s**;
- 15 rounds, 162.13 ms verification and 37.15 ms drafting per round;
- agreement 52/60 (`alpha=0.8667`), horizons 15/15, 14/15, 13/15, 10/15;
- 7.691 GB of additional Q8v2 verifier panels, still below 32 GB/node.

This is the best accepted result that executes the trunk directly as Q8.  It
does not beat the BF16 baselines, so it must not be reported as meeting the
30 tok/s plain or 50+/53.43 tok/s MTP targets.

#### Q8 checkpoint with guarded BF16-PV runtime expansion

An opt-in bridge, `TP_Q8_EXPAND_BF16=1`, dequantizes the resident Q8 rank stage
once into the pair-interleaved BF16 layout used by the proven A64FX kernels.
This still loads and stages only the Q8 checkpoint, but the active dense
decode weights are BF16 after startup; it is therefore a Q8-storage result,
not a native-Q8-compute result.

The expansion is deliberately not a malloc or weight mmap.  It allocates one
anonymous System-V arena, then the model's pinned 48-thread pool first-touches
the same row partitions that decode will consume.  A malloc-backed prototype
placed the 14.309 GB arena poorly and achieved only 5.81--5.96 tok/s.  The
System-V arena reached 27.99 tok/s, and enabling the validated PV prefetch
distance 8 raised the 32-token run to **31.37 tok/s**.  Its SHA was the exact
Q8 32-token oracle above.  This clears the approximately 30 tok/s BF16 plain
baseline by 4.6%, although it does not clear the earlier 33 tok/s stretch goal.

Before allocating, the converter sums every eligible tensor and checks
`MemAvailable`; the default reserve is 3 GB and cannot be configured below
2 GB.  Plain decode expands 401 tensors / 14.309 GB in addition to the 9.421 GB
Q8 stage.  Rank 1, the tightest node in the measured run, retained 5.9 GB after
expansion.  No mmap, `/tmp`, llama.cpp, or filesystem-backed runtime weights
are involved.

For MTP, selected draft tensors can share the same arena.  Mask 53 expands the
EH fusion projection, attention output, FFN down projection, and local draft
LM head.  Those four components preserved the oracle stream and the original
draft agreement.  With the ordinary replicated NextN stage, K=4 reached
43.98 tok/s.

The stage builder now also supports `TP_NEXTN_SHARD=1`.  It slices NextN Q/K/V
and gate/up rows plus attention-output/FFN-down columns across the four ranks;
the existing runtime performs the two required draft all-reduces.  The stage
shrinks from 9.421 to 9.124 GB/rank, and the direct all-to-all small-buffer
collective remains exact.  The best 188-token-prompt result was:

- **47.77 tok/s** wall for 64 exact tokens with K=4;
- SHA256 `03fdeecf744db006497cd1bc12ca5a065a95d61c3c38f570293641f3941afd7d`;
- 18 rounds, agreement 48/54 (`alpha=0.8889`), horizons 18/18, 15/18, 15/18;
- 15.110 GB expanded arena plus the 9.124 GB sharded-NextN Q8 stage; observed
  `MemAvailable` remained at least 5.47 GB/rank after expansion.

K=5 retained agreement 52/60 and reached 46.78 tok/s, so K=4 is the current
selection.  This improves substantially over native-Q8 MTP and confirms that
more nodes help when the draft block is actually sharded, but remains below
both the 50 tok/s requested gate and the sustained BF16 result of 53.43 tok/s.
The remaining gaps are 2.23 and 5.66 tok/s respectively.

The reproducible optimized commands are:

```sh
# Plain Q8-storage / BF16-PV runtime decode.  The named profile supplies the
# accepted expansion, sequence-length, generation, and warmup settings.
TP_SIZE=4 ./a64fx/llm/run_qwen38_q8_tp4.sh bf16-bench

# Build the separate stage whose NextN transformer is also TP-sharded.
TP_SIZE=4 TP_NEXTN_SHARD=1 ./a64fx/llm/run_qwen38_q8_tp4.sh stage

# Exact K=4 MTP on the long-context Q8 oracle.  bf16-mtp selects the sharded
# stage, mask 53, PV prefetch 12, direct small-buffer all-to-all, and K=4.
TP_SIZE=4 TP_PROMPT_FILE="$PWD/tmp/qwen38_mtp_prompt.txt" \
  ./a64fx/llm/run_qwen38_q8_tp4.sh bf16-mtp

# Native-Q8 verifier path (no BF16 runtime expansion).
TP_SIZE=4 TP_Q8_VERIFY=q8v2 TP_SPEC_K=5 \
  TP_PROMPT_FILE="$PWD/tmp/qwen38_mtp_prompt.txt" TP_MAXSEQ=512 \
  ./a64fx/llm/run_qwen38_q8_tp4.sh mtp-check
```

`bf16-mtp` deliberately rejects TP2/TP3: the polished performance profile is
the measured TP4 configuration.  The generic `mtp-check` mode remains
available for diagnostic topology, mask, and K sweeps.  BF16 expansion now
also requires a complete TP stage and rejects unknown NextN-mask bits, so it
cannot silently fall back to source-file-backed weights or accept a mistyped
profile.  The launcher uses ordinary incremental builds rather than forcing a
full rebuild for every benchmark.

#### Rejected experiments and remaining work

- Hierarchical barriers preserved tokens but reduced native TP4 from 14.09 to
  7.78 tok/s.  Flat barriers remain the default.
- `block64-ffn` W8A8 changed later tokens and fell to 9.74 tok/s.
- The Q8v2 6x2 verifier, K=6/Q8B6 verifier, and a Q8v2 M=1 decode path were
  slower; K=6 and M=1 also changed the accepted stream.
- Expanding all NextN projections produced an invalid draft: agreement
  collapsed to 2/248.  Q, K, V, FFN gate, and FFN up are individually exposed
  for diagnostics but are not in accepted mask 53.  A corrected aligned
  BF16-PV fused gate/up dispatch preserved agreement but slowed sharded MTP to
  46.07 tok/s, so it too was reverted.
- Exact four-row and fused-two-matrix Q8 kernels in the standalone NextN pool
  were 1--3% slower at model level and were reverted.
- Repeating the prompt to 367 tokens did not reproduce BF16's higher agreement;
  Q8 agreement fell to 55/68, confirming that checkpoint quantization, not
  merely short context, limits this draft.

The highest-value remaining tasks are:

1. Build a native M=1 Q8 panel/kernel that keeps Q8 bytes in HBM, shares each
   activation load across rows, and preserves the current two-accumulator
   reduction exactly.  The native plain gap is still 13.39 to 30+ tok/s.
2. Replace the 7.691 GB Q8v2 MTP verifier panels with a compact no-reread
   K=4/K=5 layout.  It must beat 162 ms/round without changing the Q8 oracle.
3. Accelerate the remaining sharded NextN QKV and FFN gate/up directly in Q8.
   Straight BF16-PV gate/up conversion is exact after aligned dispatch but is
   slower at model level; Q8 needs a lower-byte kernel to reduce draft time.
4. Raise draft agreement for the quantized checkpoint.  At alpha 0.889, K=4
   emits only 64/18=3.56 tokens/round, so even reduced verifier time has a hard
   ceiling.  Test quantization-aware calibration of only the draft block and
   head against the Q8 trunk oracle, without borrowing BF16 model weights.
5. Compact the few tensors that must remain Q8 after runtime expansion, then
   detach the original 9.421 GB stage arena.  This would increase HBM headroom
   and permit safer verifier/draft experiments, but it is a storage-mode
   optimization rather than native-Q8 compute.
6. Run a 256-token, three-repeat sustained gate for every future accepted path.
   Require the K=0 SHA, stable rank-local pending queues, memory below 32 GB,
   and separate setup/prefill/decode timing.  The open targets remain native
   Q8 above BF16 for both modes and **50+ tok/s MTP** (then 53.43+).

### Resident PP3xTP4 prefill to three concurrent TP4 decoders (2026-08-24)

`qwen38_mixed_runner` implements a single-process-lifetime topology switch on
12 nodes.  Every node loads its complete 9.421 GB TP4 Q8 stage once.  The
runner builds only its PP-owned Q8v2 prefill panels, prefills three independent
requests, serializes mutable KV/SSM state into memory, transposes those blobs
across equal TP lanes, releases the prefill panels, expands the retained Q8
stage to the 14.309 GB BF16-PV decode layout, and then runs three TP4 decode
replicas.  No weight is reread from `/local` at the transition.

The in-memory state API is `transformer_runtime_state_size/pack/unpack` in
`common/transformer.h`.  State transfer is range-addressed by PP layer ownership
and contains no tensor weights or scratch buffers.  The launcher is:

```sh
./a64fx/llm/run_qwen38_q8_mixed_12n.sh
```

A 12-node smoke with three synthetic eight-token prompts validated the complete
lifecycle.  Per node it retained 9.421 GB of Q8, temporarily held 2.508--2.654
GB of Q8v2 panels, released those panels, and expanded 14.309 GB for decode.
The state transpose moved 41.258 MB/rank in 19 ms.  All four ranks within each
decode replica produced identical token hashes.

The sustained 32-token concurrent result was:

| replica | decode time | throughput | token hash |
|---:|---:|---:|---:|
| 0 | 3.107 s | 10.30 tok/s | `a7f258fc628d5923` |
| 1 | 3.003 s | 10.66 tok/s | `6865a015d2e5b483` |
| 2 | 3.019 s | 10.60 tok/s | `7c1af6a9cc995633` |

This is **30.9 aggregate tok/s**, not the expected approximately 94 tok/s from
three isolated 31 tok/s replicas.  Decode collectives account for only
0.64--0.81 seconds of each 32-token run; most of the regression is local
forward bandwidth while all 12 nodes decode.  Compact torus rectangle grouping
and parking the OpenMP prefill team did not improve aggregate throughput.  The
non-compact grouping remains the default because it preserved the established
state/request mapping and hashes.

One transition hazard was isolated: immediately decoding transferred hybrid
state without a full state walk produced a severe first-touch/placement stall.
The runner now validates and warms every transferred convolution/recurrent
state page before rebuilding the persistent decode pool.  No IEEE subnormals
were present in the accepted run.  Dropping the transferred state was used only
as a diagnostic and is not enabled by the launcher.

Next work should measure HBM counters and page residency during simultaneous
decode, then compare three independently launched TP4 controls in the same
12-node allocation.  Until aggregate bandwidth scales, this topology is a
correct resident-weight prototype rather than a throughput win.

### Native-BF16 PP3xTP4 prefill and independent TP4 decode (2026-08-25)

`run_qwen38_bf16_mixed_12n.sh` is the accepted 12-node launcher for the BF16
weights.  It stages one native TP4 shard per node, builds exact BF16 PV48
panels only for the PP-owned layers, prefills three requests, transposes their
runtime state in memory, releases the transient panels, and repacks the
resident row-major BF16 weights in place to the PV8 decode layout.  It never
rereads or quantizes the weights at the phase boundary.

```sh
./a64fx/llm/run_qwen38_bf16_mixed_12n.sh stage
./a64fx/llm/run_qwen38_bf16_mixed_12n.sh direct # 3 independent TP4 K=0 streams
./a64fx/llm/run_qwen38_bf16_mixed_12n.sh bench  # PP3xTP4, 4096 then decode
```

The 4096-token prefill measured 221.6--225.4 tok/s per request.  Releasing
4.49--4.76 GB/rank of PP-local PV48 panels and repacking 14.309 GB/rank took
0.17--0.18 seconds; the warm weight stream sustained 856--869 GB/s.  The
independent 256-token K=0 decode streams measured 31.81, 32.42, and 31.61
tok/s, with identical hashes on all four ranks in each replica.  These are
per-context rates, not an aggregate number.

Flat worker barriers (`TF_HIER_BARRIER=0`) are required.  Omitting that setting
was the cause of the earlier 10--11 tok/s mixed-runner result; it was not HBM
contention between the three replicas.  `OMP_WAIT_POLICY=passive` and
`KMP_BLOCKTIME=0` keep the prefill OpenMP team from competing with decode.

Decode that continues with the complete 4096-token KV cache is a different,
attention-bound workload.  Splitting QK positions over all 48 cores preserves
the token hashes and raises it from 21.05--23.54 to 24.74--25.06 tok/s, but it
does not meet 30 tok/s.  A value-dimension split was tested and rejected: it
fell to 21.77--22.83 tok/s and one replica diverged.  The accepted launcher
therefore meets the 30+ target for the requested independent decode contexts
and the 120+ target for each prefill request, while reporting long-context
continuation separately.

### Twelve-node prefill ceilings and 500/1000 attack (2026-08-25)

At the allocation's 2.0 GHz setting, the conservative dense-compute roofs are:

| path | 12-node arithmetic roof | 27B dense-token roof | requested target |
|---|---:|---:|---:|
| BF16 storage, FP32 FMA | 73.728 TFLOP/s | 1365 tok/s | 500 tok/s (37% of peak) |
| Q8 SDOT, one vector issue/cycle | 147.456 TOP/s | 2731 tok/s | 1000 tok/s (37% of peak) |
| measured Q8v2 kernel, 187 GIOPS/core | 107.7 TOP/s | 1994 tok/s | 1000 tok/s (50% of kernel roof) |

The calculation uses two operations per parameter, 48 cores/node, 16 FP32 SVE
lanes, two FP FMA pipes, and a conservative single SDOT issue rate.  The prior
36.864 TFLOP/s figure accidentally omitted the second operation in each FMA.
It is an arithmetic upper bound: causal attention, DeltaNet recurrence, activation
packing, pipeline bubbles, and communication all lower application throughput.
At chunk 256 the weight-stream bandwidth roof is far higher, because each
packed panel is reused by 256 tokens; prefill is compute/communication bound,
unlike M=1 decode.

The exact BF16 4096/chunk256 profile measured 221.95 tok/s.  On stage 0 its
18.455 seconds comprised 15.633 seconds of range compute and 0.626 seconds of
pipeline send.  The accumulated phase costs were 9.356 seconds in projection
GEMMs, 3.407 seconds in TP reductions, and about 2.57 seconds in recurrent,
attention, normalization, and activation work.  Consequently, perfecting the
GEMM alone cannot reach 500 in PP3xTP4.  uTofu RSAG4 reduced the reported
collective phase to 2.720 seconds but did not improve end-to-end throughput
(222.20 tok/s); chunk 512 increased pipeline bubbles and fell to 205.45 tok/s.

The runner now exposes experimental PP12xTP1 (`Q38_PREFILL_TP_SIZE=1`) using
the transformer's layer-range loader.  It eliminates TP reductions and is a
useful topology diagnostic, but the first 4096/chunk128 run reached only 62.04
tok/s: the stages were imbalanced, full-width attention was costly, and the
last stage faulted the tied LM head from the shared GGUF.  This is not an
accepted performance path.  The measured results establish the next kernel
work: BF16 needs substantially higher projection efficiency plus cheaper
non-GEMM phases; Q8 needs a wider/reused-activation SDOT schedule and reductions
below the present PP3xTP4 critical path.

The first BF16-activation driver pass now stores complete 12-token p-odd tiles
directly to their final token-major destination and reuses the packed activation
for consecutive matrices with the same input.  The old temporary tile remains
available with `TF_PODD_DIRECT_OUT=0`, and activation reuse can be disabled with
`TF_PODD_REUSE_X=0`.  The current short gate remains `next=192550`; the 4096
gate remains `next=226343`.  On this allocation the full run is still about
230--232 tok/s, so these lossless driver changes alone do not approach 500.

A one-region grouped projection dispatcher reduced short-run time but changed
attention and FFN results and was rejected.  BF16 pipeline payloads are retained
behind `Q38_PREFILL_PIPE_BF16=1`; they preserved the measured gates but were
end-to-end neutral.  The launcher now leaves the decode pthread pool off after
the first chunk.  An actual-shape `252x5120x3840` p-odd microbenchmark measured
2.56 TFLOP/s/node including its legacy weight pack and output transpose,
confirming driver headroom while also showing that communication and non-GEMM
phases must be reduced alongside GEMM.

#### Corrected BF16 prefill layout and 424 tok/s (2026-08-25)

The earlier 230--232 tok/s BF16-activation ceiling was invalidated by a layout
initialization bug.  `transformer_tp_load_stage` defaults native BF16 tensors to
the decode-only PV8 layout, while the prefill runner set `TF_PODD` only after
loading and then treated those PV8 bytes as row-major input to the p-odd packer.
The dispatcher selected PV8 before p-odd, so the advertised `2x12` projection
path was not actually running.  The runner and launcher now force
`TP_STAGE_BF16_PV=0` before the stage load for both PV48 and BF16-activation
prefill.

With genuine row-major-to-p-odd weights, the 128-token gate changes from the
corrupted-path `192550` to **`1293`**, matching native Q8.  At 4096/chunk252 the
MPI path reaches **408.10 tok/s**, with projection phases roughly halved.  A
deterministic BF16-wire uTofu RSAG truncates rank partials to BF16, sums ranks
0..3 in FP32, and all-gathers BF16; it preserves `1293`/`62842` and reaches
**423.97 tok/s**:

| phase, rank 0 | corrected MPI | BF16-wire uTofu |
|---|---:|---:|
| projection + output + FFN GEMMs | 4.12 s | 4.10 s |
| TP collectives | 1.77 s | 1.36 s |
| attention kernel | 0.67 s | 0.67 s |
| end-to-end | 10.037 s | **9.661 s** |

SVE FEXPA softmax is enabled for prefill and reduced the attention kernel from
about 0.89 s to 0.67--0.71 s without changing either corrected gate.  CMG-local
activation replication (227.53 vs 225.60 tok/s on the old broken layout), a
four-worker-per-head DeltaNet scan (1.68 vs 1.04 s scan), and direct FFN
up/SiLU/down packing (405.65 vs 408.10 tok/s corrected MPI) were all neutral or
slower and remain opt-in diagnostics.  Chunk192 reached 422.21 tok/s, so
chunk252 remains the default.

#### Barrier-free 48-core DeltaNet scan and 479 tok/s (2026-08-25)

The original four-worker-per-head experiment synchronized each group twice per
token so lane 0 could compute the output RMS norm.  That synchronization is not
required for recurrence: normalized/gated output is consumed only by the SSM
output projection and never feeds the recurrent state.  The accepted schedule
therefore assigns four workers disjoint 32-row state slices for the complete
token sequence, then normalizes and gates all completed token/head outputs in a
separate 48-thread pass.  It preserves recurrence order within every state row
without any scan-loop barriers.  `TF_SSM_PREEXP=1` also precomputes the scalar
decay once; the scan honors that representation rather than exponentiating it
again.

On 12 nodes, PP3xTP4, BF16 activation, BF16-wire uTofu, 4096/chunk252, this
reduces the accumulated DeltaNet scan from about **1.01 s to 0.38 s** and reaches
**479.17 tok/s** (`8.548 s`, `next=62842`).  The 128-token correctness gate is
still `next=1293`.  The launcher enables the barrier-free scan, fast scalar
preparation, and precomputed decay by default for `bf16-act`; exact BF16 retains
the conservative paths.  Query-blocked attention (404.93 tok/s), chunk280
(452.52), chunk256 (456.12), and MPI nonblocking pipeline sends (no asynchronous
progress, 413.47) were measured and rejected.

The remaining preparation bottleneck was the batched depthwise convolution.
Its scalar channel-parallel schedule walked token-major rows at a roughly
10K-float stride and used only one float from each fetched cache line.  The
Qwen kernel-size-4 path now advances an SVE vector of adjacent channels through
time, retaining its three history vectors in registers and preserving the
circular-state update exactly.  SSM preparation falls from about **0.52 s to
0.097 s**.  Combined with the barrier-free scan, the accepted 12-node BF16 run
reaches **500.93 tok/s** (4096/chunk252, 8.1769 s, `next=62842`); the short gate
remains `next=1293`.

#### TP4 grouped projections and 600 tok/s BF16 prefill (2026-08-25)

The TP4 prefill schedule now packs RMS-normalized activations directly into the
p-odd input layout.  Attention Q/K/V and FFN gate/up projections share one
barrier-free OpenMP task queue, eliminating the projection-local team barriers
and the unused FP32 normalized buffer.  The packed-only path is guarded by all
of the destination weights being p-odd packed; exact BF16/PV48 continues through
the original FP32 normalization and GEMM path.

TP output reductions can add directly into the residual.  The accepted narrow
wire is block-scaled signed INT8 (256 values per FP32 scale): all four outgoing
TP shards are quantized in one parallel region, owners sum ranks 0..3 in FP32,
and all four gathered shards are dequantized and residual-added in one region.
This retains the checked tokens (`1293` at 128 and `62842` at 4096) while reducing
rank-0 accumulated collective time from about 1.17 seconds to 0.75--0.80 seconds.

The 12-node BF16-activation launcher therefore defaults to PP3 x TP4, layer cuts
20/42, chunk252, grouped packed projections, fused residual reductions, and the
INT8 uTofu wire.  Two consecutive 4096-token runs measured **603.57 tok/s**
(6.7863 s) and **600.35 tok/s** (6.8227 s), both with `next=62842`.  An exact
BF16/MPI 128-token fallback check remained `next=1293` with unpacked PV48
weights.

#### BF16 750 tok/s follow-up (2026-08-25)

The block-scaled INT8 TP collective now retains one OpenMP team across outgoing
quantization, owner reduction, reduced-shard quantization, and gathered residual
addition. Network operations execute in single-thread regions between work
sharing phases. This preserves `1293`/`62842`, reduces the accumulated
collective phase to about 0.79--0.81 seconds, and measures about 602--607 tok/s
end to end; it is accepted as the default implementation.

Two larger redesigns remain opt-in diagnostics. `TF_QWEN_ATTN_BLOCK=1` groups
causal queries to reuse GQA K/V rows, but score-workspace traffic leaves the
attention phase neutral at about 0.66 seconds. `Q38_PREFILL_PP_COMM=utofu` uses
directional double-buffered uTofu payload slots, sequence trailers, and reverse
credits. It cuts explicit PP send time from 0.5--0.7 seconds to about 0.02
seconds, but only removes backpressure accounting: the slowest pipeline stage
still determines the 6.75--6.80 second wall time, so MPI remains the default.

The existing FP16 kernel was evaluated as a route to twice-width arithmetic.
Full-K FP16 accumulation reaches 276.9 GFLOP/s/core but has roughly 6.2% relative
error at K=5120. Splitting K into 64--1024 element FP16 partials and accumulating
them in FP32 reaches only 49--108 GFLOP/s/core, below the 137 GFLOP/s/core BF16
p-odd kernel, while retaining unstable error. It was rejected. The two-run 750
tok/s gate remains open; reaching it requires a new mixed-precision kernel or a
materially different parallel decomposition, not additional PP handoff tuning.

## Two- and four-node decode topology recommendation (2026-08-31)

For one autoregressive Qwen3.8-27B decode stream, use tensor parallelism on all
available nodes: **TP2 on two nodes and TP4 on four nodes**.  Layer/pipeline
parallelism reduces the number of collectives, but it executes each layer range
serially for a single token.  The saved communication is smaller than the
weight-streaming time saved by running each projection concurrently across the
TP ranks.

Qwen3.8-27B reduces a 5120-element FP32 residual buffer 129 times per generated
token.  On the compact four-node topology, the earlier 16 KiB probe measured
13.88 us warm and 25.40 us cold per collective, corresponding to a 1.8--3.3 ms
transport floor.  Real TP4 decode spends approximately 4.5--8 ms/token in the
collective phase once cache effects and rank-arrival skew are included.  This is
only about 15--20% of a representative 34--40 ms BF16 token, so eliminating the
collectives does not compensate for serializing the layer stages.

The dedicated `a64fx/utofu-tests/qwen38_allreduce_bench` measures the exact
20 KiB decode payload.  It runs TP2 as two concurrent pairs followed by TP4 on
all four nodes, uses the production recursive-doubling collective, and reports
the projection for 129 reductions/token.  Its payload, barrier, send, and receive
regions are 256-byte aligned and strictly `MPOL_BIND`-bound to the CMG containing
the pinned communication thread.  `wire_GB/s` and `link_peak_pct` count actual
recursive-doubling wire bytes against one Tofu-D TNI's 6.8 GB/s peak.  The 20 KiB
synchronous collective is latency- and reduction-bound and must not be expected
to attain the large-message raw-Put peak.

### Recommended configurations

| workload | two nodes | four nodes |
|---|---|---|
| one decode stream, lowest latency | **PP1 x TP2** | **PP1 x TP4** |
| latency-sensitive serving | **TP2** | **TP4** |
| several continuously busy streams | TP2 or PP2 after measurement | evaluate **PP2 x TP2** or PP4 |
| long-prompt prefill | TP2 | TP4 for four nodes; mixed PP x TP becomes useful with more nodes/chunks |

The measured native-Q8 decode sweep confirms the single-stream choice: TP2
reached 8.44 tok/s, while TP4 reached 14.09 tok/s.  Adding ranks reduced the
per-node streamed weight set more than the extra collective round cost.  The
four-node BF16 path similarly reaches roughly 25--31 tok/s plain decode, with
representative profiles around 28--34 ms compute plus 4.5--8 ms communication.
The accepted TP4 speculative path reaches 53.43 tok/s, which further favors
retaining TP4 as the decode topology.

A four-node **PP2 x TP2** layout is not preferred for a single stream.  Its two
half-model stages are sequential, so their summed compute resembles a full TP2
decode rather than TP4; the smaller TP2 collective cannot recover the lost
parallel weight bandwidth.  Pure PP4 has the same issue across four serial
stages.  Both layouts can become useful for aggregate serving throughput when
independent requests keep every pipeline stage occupied.  In that case PP2 x
TP2 is the balanced first configuration to measure: it retains two-way tensor
parallelism within each stage and permits two requests to overlap.  PP4 is a
throughput-oriented alternative when at least four independent sequences remain
ready and per-request latency is secondary.

For combined prefill and decode on larger allocations, use mixed parallelism
for token-chunked prefill and switch to independent TP4 decode groups when the
runtime supports a correct state handoff.  The established twelve-node example
is PP3 x TP4 for prefill.  This does not make pipeline parallelism preferable for
single-token decode: prefill has enough token chunks to fill the stages, whereas
one decode dependency chain does not.
## 2026-08-31: Qwen3.8-27B TP4 mixed-Q4 bring-up

The native TP stage now accepts the complete mixed `Qwen3.8-27B-UD-Q4_K_XL.gguf`
layout rather than silently retaining only its F32 tensors.  The model contains
866 tensors: F32 360, Q4_K 97, Q5_K 325, Q6_K 19, and IQ4_XS 65.  Column slices
are required to start and end on each format's GGML block boundary.  Every file
entry and the registered uTofu collective regions remain 256-byte aligned.

TP4 stages 5.756 GB per rank.  A cold, one-token end-to-end check loaded all 866
tensors on all four nodes and produced token 198 in lockstep.  Strict MPOL_BIND
placed the barrier and all-reduce source/landing regions on persistent worker
0's CMG (reported NUMA node 4 on this allocation), preventing inter-CMG access
in the communication path.

The first compact-Q4 correctness baseline is 315.58 ms/token (3.15 tok/s):
295.08 ms compute and 20.50 ms communication across 129 reductions.  This is a
bring-up baseline, not an optimized result.  Q5_K, Q6_K, and IQ4_XS currently
fall through the generic full-row F32 dequantizer; native compact SVE/SDOT
kernels are therefore the gating work for the 100 tok/s plain and 120 tok/s MTP
targets.  The existing weight-byte estimator reports 14.31 GB/token for this
mixed stage and must be corrected before using its derived 48 GB/s figure.

Direct compact SVE dots now cover Q5_K, Q6_K, and IQ4_XS in addition to the
existing Q4_K kernel. They decode packed bitplanes and scales inside registers
and never allocate or materialize an F32 row. A first-token TP4 check improved
forward time from 315.58 to 171.76 ms (3.15 to 5.76 tok/s) while retaining token
198. The opt-in `TF_COMPACT_K_CHECK=1` oracle compared the first row of each
format against GGML dequantization plus a double-precision dot on every rank;
the worst relative error was 9.22e-7. These are exact compact SVE/FMA kernels.
An SDOT variant requires activation/weight requantization and remains behind
the quantized quality gate rather than silently changing the exact Q4 path.

### TP4 BF16/Q8 SDOT comparison (2026-08-31)

Short identical 16-token measured regions (`TP_AR_BATCH=1`, four warm-up
tokens) gave the following directional results. Exact BF16 remained fastest at
32.26 ms/token. Converting BF16 weights in place to row INT8 took 49.57
ms/token; using INT16 activations with compact INT8 weights took 49.88 ms/token.
The INT16 stream matched all 20 exact-BF16 token IDs, while INT8 first diverged
at token 8. Thus H-to-D SDOT is retained as an accuracy experiment, not a speed
profile: it is about 55% slower than the optimized BF16 path.

For the native Q8 stage, row INT8 and INT16-activation SDOT measured 101.26 and
98.20 ms/token respectively. Native block64 Q8 measured 107.82 ms/token. A new
four-row row-INT8 kernel shares each activation load across four weight rows;
it improved the Q8 row probe to 96.60 ms/token, but did not improve the BF16
conversion path outside run-to-run noise. These short probes are not the final
three-repeat 256-token acceptance gate, and neither Q8 token stream has yet
passed the teacher-forced quality threshold.

### Exact BF16 follow-up profile (2026-08-31)

The current TP4 exact-PV path measures 32.36 ms/token with the validated
default collective and retains the 20-token reference stream. Of that, 27.51
ms is compute/rank-arrival time and 4.85 ms is 129 FP32 reductions. Profiling
rank 0 attributes 8.27 ms to FFN gate/up, 5.74 ms to FFN down, 5.11 ms to SSM
input, 4.09 ms to SSM output, 1.97 ms to attention projections, and 2.66 ms to
SSM preparation/core.

BF16-PV prefetch distances 4, 12, and 16 measured 32.51, 32.18, and 33.24
ms/token in short A/B runs, so the launcher now uses 12. Hierarchical barriers
regressed to 103.42 ms/token; projection/communication overlap regressed to
52.79 ms/token and increased reductions from 129 to 641. A single-accumulator
BF16 reduction tree was neutral once given equal prefetching and was removed.
These results put the remaining 40 tok/s gap in projection scheduling and
resident bandwidth, not scalar activation work or collective-buffer tuning.

After the four-node allocation restarted, the rank-local BF16 shards were
restaged and all 866 tensors loaded within the HBM guard.  A 32-token warmed
profile reproduced the exact 48-token reference stream.  Production-kernel
microbenchmarks then showed that distance 8 is the stronger general setting:
at K=4352 it reached 766.56 GB/s versus 764.77 GB/s at distance 12, and at
K=5120 it reached 815.94 versus 805.98 GB/s.  Same-session end-to-end A/B runs
measured 33.78 ms/token at distance 8 and 34.75 ms/token at distance 12, so the
launcher default is now 8.  A separate K=1536 specialization was rejected:
although its isolated kernel reached 588.87 GB/s at distance 8, it was neutral
end to end and added no useful model-level speedup.

### BF16 K=5 MTP runtime fix (2026-09-01)

The restarted sustained-MTP baseline exposed an OpenMP runtime failure rather
than a math-kernel limit: K=5 verification took about 9.4 seconds per round and
delivered only 0.23--0.29 tok/s.  The batch verifier opens many short OpenMP
regions, while the launcher's `KMP_BLOCKTIME=1` repeatedly put the 48-worker
team to sleep.  Holding the team warm globally fixed verification but slowed
the ordinary prompt pass, so `tp_runner` now changes Fujitsu's exported
`kmp_set_blocktime` dynamically: 200 ms during batched verification and zero
before the pthread NextN draft.

The K=5 BF16 dispatch also replaces the register-heavy 4-row x 5-token kernel
with compact 4x3 plus exact 8x2 kernels.  On the 180-token sustained prompt,
the accepted combination reduced verification to 89.34 ms/round and preserved
the normal 17.79 tok/s prompt pass.  Six rounds generated 16 tokens at 17.21
tok/s with nondegenerate greedy agreement (13/24, alpha 0.5417), versus 0.29
tok/s before the runtime fix.  Draft generation is now the dominant cost at
65.51 ms/round; verification is no longer catastrophically stalled.

An initial independently staged TP4-sharded NextN test reduced draft time to
56.25 ms but failed correctness (`teacher match=0/178`, alpha zero).  The cause
and corrected sharded result are documented below.

### Correct BF16 NextN TP4 and K=2 selection

The sharded-stage failure was a layout/scheduler mismatch.  The BF16 loader
automatically PV-packed every sliced tensor, including NextN K/V.  Each local
K/V matrix has 256 rows, and the fused QKV scheduler splits it across 48
workers in 5--6-row ranges; the PV kernel requires 8-row-aligned ranges.  The
loader now leaves NextN row-major unless `TP_NEXTN_PV_MASK` explicitly selects
a scheduler-safe tensor.  This restored the replicated reference gate exactly:
teacher match 55/178 and the same offset histogram.

Individual quality sweeps selected mask 5 (EH fusion plus attention output).
PV Q changed draft argmaxes, while attention output retained 55/178.  K/V and
gate/up intentionally have no PV mask bit.  With direct small-buffer all-to-all
enabled, the 64-token K sweep selected K=2:

- K=4: 16.51 tok/s, 48/90 greedy matches;
- K=3: 20.21 tok/s, 40/62 greedy matches;
- K=2: **26.24 tok/s**, 26/39 greedy matches and alpha 0.6667.

The accepted K=2 profile uses 39 rounds, 50.51 ms verification and 11.97 ms
draft time per round.  It preserves teacher match 55/178, processes the
180-token prompt at 19.40 tok/s, and reduces the effective generated-token
forward time to 38.08 ms (33.02 ms compute plus 5.06 ms communication).  This
is a large correction over the 0.29 tok/s stalled baseline, though it remains
below plain BF16 decode and the requested 80 tok/s MTP target.

Use `run_qwen38_bf16_tp4.sh stage-mtp` after each allocation restart to build
the separate `/local/...-nextnshard` image, then run `mtp-sustained`.  The MTP
mode now selects TP4 NextN sharding, mask 5, direct all-to-all, and K=2 by
default; ordinary BF16 stage/decode settings are unchanged.

#### Post-restart BF16 sweep (2026-09-01)

The `stage-mtp` workflow was restaged on a fresh four-node allocation and
reproduced the accepted teacher gate (55/178).  The short K=2 profile measured
52.56 ms verification, 13.25 ms draft, and 26.99 tok/s.  Raw target hidden
states slightly increased the offline teacher score to 58/178 but left runtime
acceptance at 26/39 and reduced the 64-token result to 25.42 tok/s; hidden-first
fusion failed completely at 0/178.  Both remain disabled.

Verifier blocktimes 100 and 400 ms, MTP2 prefetch distances 8 and 16, poll-spin
32, a 16,384-float all-to-all cutoff, and adding FFN-down PV all regressed from
the accepted defaults (200 ms, distance 12, poll-spin 8, cutoff 8192, mask 5).
A PV-aware NextN fused gate/up implementation preserved 55/178 but raised draft
time to 14.58 ms and was removed.

Direct all-to-all was also retested for exact plain BF16 decode.  The initially
recorded standard-collective token hash could not be reproduced: an adjacent
control run with recursive doubling produced the same 96-token SHA256 as
all-to-all, `737b132892e98a47c9f69f10779d0075d1c071579b94c4a16045d7d385239f94`.
An explicit four-rank arithmetic-tree implementation produced that same hash
as well, ruling out the A2A fold order as the cause of the earlier sequence.
It was removed rather than retaining unnecessary code.

In the final same-build A/B, recursive doubling measured 34.40 ms/token and
the simple direct A2A path measured 33.49 ms/token over the 64-token measured
region, a 0.91 ms/token (2.6%) reduction with identical token hashes.  TP4 now
enables A2A for reductions up to 8192 floats in both ordinary decode and MTP;
set `TP_AR_A2A=0` for a recursive-doubling control.  Larger reductions continue
to use the tree collective.

A finer same-session BF16-PV prefetch sweep then found distance 6 preferable to
the prior distance 8 default.  Distance 6 measured 32.59 ms/token over the
32-token screen, while distance 4 measured 33.54 ms/token.  Its 64-token
acceptance run measured 32.65 ms/token (27.10 ms compute and 5.55 ms
communication on rank 1) and retained the control SHA256 above.  The launcher
therefore uses distance 6 for single-token BF16 kernels; the independently
tuned two-token MTP kernel remains at distance 12.

Production-shape microbenchmarks show the exact BF16-PV kernel is already close
to the local streaming limit: 773.79 GB/s at K=4352 (84.3% of a 918.17 GB/s
read ceiling) and 832.54 GB/s at K=5120 (91.0% of 915.02 GB/s).  The remaining
model-level gap is therefore primarily scheduling, barriers, and rank skew.  A
two-chunk SVE unroll retained the token hash but regressed decode to 35.19
ms/token through added register pressure.  Vectorizing the persistent worker's
small per-thread SiLU slices also retained the long hash but regressed to 36.10
ms/token and made rank 0 the straggler.  Both kernel experiments were removed.
BF16 collective transport was also rejected: although its 96-token hash matched
the current FP32 control, conversion overhead and disabling direct A2A raised
decode to 36.86 ms/token, with roughly 10 ms/token charged to communication on
the waiting ranks.  FP32 direct A2A remains the TP4 decode transport.

#### Resident INT8/INT16 SDOT follow-up

The row-major BF16 stage was quantized in place to compact per-row INT8 weights
and tested with both INT8 and INT16 activations.  Full-projection W8A8 measured
51.28 ms/token (about 19.5 tok/s); H-to-D W8A16 measured 51.15 ms/token.  Both
were slower than the 32.65 ms/token BF16-PV path and produced quantized token
streams.  A four-row W8A16 kernel retained its stream but regressed to 56.30
ms/token from SVE register pressure, so it was removed; the existing four-row
W8A8 kernel remains active.

`TP_INT8_MODE=row-ffn` now provides a memory-neutral diagnostic that quantizes
only FFN gate/up/down in place while leaving attention, SSM, and the head BF16.
One-token screens measured 50.56 ms for INT8 SDOT and 48.77 ms for INT16 SDOT,
still well behind BF16.  The allocating block64 FFN pack requested another
4.412 GB while retaining the 17.165 GB stage and was killed on rank 0; it is not
safe for this 32 GB interactive configuration.  Decode P-at-V is not converted:
it would require a scaled, transposed INT8 value cache, and its small current
context cost cannot recover the projection/FFN regression.

#### BF16 MTP 50 tok/s target

On the repeated 359-token production prompt, runtime draft agreement is much
higher than the earlier short-context sweep.  K=2 accepted 32/33 second drafts
and measured 30.16 tok/s (52.01 ms verify plus 12.24 ms draft per round).  A
same-session K sweep measured 32.35 tok/s at K=3, **34.90 tok/s at K=4**, and
32.53 tok/s at K=5.  K=4 retained 49/51 draft matches (alpha 0.9608), so the
sustained launcher now defaults to K=4.

The 50 tok/s target is not yet met.  K=4 spends about 71 ms verifying and 37 ms
building its three sequential NextN drafts.  A forced-accept diagnostic with
draft generation removed reached **55.26 tok/s**, establishing that the batched
verifier is fast enough but leaving only about 7 ms/round for a production
draft path.  A private 48-thread shadow pool reduced draft time only to 34.94
ms and did not improve end-to-end throughput.  MTP4 prefetch distances 0, 4,
and 12 did not beat distance 8 in the 64-token acceptance run.  Reaching 50+
therefore requires a persistent/fused NextN implementation or a different
near-zero-cost proposer, not another trunk-verifier prefetch adjustment.

`TF_NEXTN_PROFILE=1` now reports per-call draft phases.  Stable mask-5 calls
take roughly 8--10 ms each: EH 0.8--1.5 ms, QKV 0.8--1.4 ms, attention about
0.8--1.3 ms, output 0.9--1.8 ms, FFN 1.9--3.0 ms, and the local vocabulary
head 1.4--2.3 ms.  Thus three autoregressive calls have a measured 24--30 ms
floor even after dispatch jitter is removed.  Shadow pools with 24 and 12
threads measured 34.41 and 37.73 ms/round and did not help.

A separately staged mask-37 experiment added BF16-PV packing for the NextN
vocabulary head.  It retained 25/27 runtime matches but regressed the 32-token
screen to 30.41 tok/s (76.69 ms verify, 40.17 ms draft); head time remained
about 1.6--2.0 ms.  Mask 5 remains the accepted layout.  The result reinforces
that 50+ needs speculative lookahead overlapped with verification (and a
separate collective stream), rather than another per-call layout change.

#### Asynchronous K=4 drafting experiment

`TP_MTP_ASYNC=1` implements the proposed continuation pipeline.  At the start
of a round, a background thread predicts the next four-token queue from the
current bonus token while the trunk verifies `[input,draft0,draft1,draft2]`.
The continuation is adopted only when all three drafts and the bonus token
match.  A rejected chain is discarded and regenerated from the selected target
hidden state, so speculative state never changes the committed trunk state.

The draft uses an independent NextN scratch/KV context and pthread pool.  Its
collectives use a second uTofu VCQ on TNI 1, a separate stag, and a 256-byte
aligned 405 KiB communication region.  The default 36/12 split reserves three
cores in every CMG for drafting; the striped affinity keeps each weight row on
its owning CMG and avoids inter-CMG reads.  The verifier's batch helpers now
consistently honor the reduced team size.  The SSM scan maps an arbitrary equal
number of lanes to each of the 12 local recurrent heads, supporting both the
48-thread control and reduced verifier teams without the snapshot-heavy scalar
fallback.

Two correctness/performance faults found during bring-up are now guarded in
the implementation.  A completed worker waits for the consumer instead of
executing the same request repeatedly, and blocked attention, SSM convolution,
SSM preparation, projections, and FFN all use the same verifier thread count.
Before those fixes, stale draft requests flooded the second TNI and mixed
36/48-thread OpenMP regions took 10--22 seconds per verification round.  After
the fixes, the 36-thread verifier returned to 73.52 ms/round.

The experiment does not improve sustained throughput on A64FX.  With the
359-token prompt, 36 verifier plus 12 draft cores measured 73.52 ms verification
but 207.63 ms for the concurrent four-step continuation, including 150.01 ms
of exposed wait; the 32-token screen reached 9.65 tok/s.  A 24/24 split measured
91.14 ms verification, 227.97 ms continuation, and 154.69 ms exposed wait,
reaching 10.00 tok/s over 64 tokens.  Giving the draft more cores did not help:
the verifier and NextN projections contend for the same HBM bandwidth in every
CMG.  A second TNI removes collective serialization, but cannot remove this
weight-stream contention.

Therefore asynchronous full-NextN drafting is retained as an opt-in diagnostic,
not enabled by the production launcher.  The measured break-even requires the
four-step continuation to finish within roughly the 72 ms verifier window and
the verifier itself to fall toward 55 ms/round; current concurrent continuation
is about three times that budget.  Reaching 50+ BF16 MTP needs a materially
smaller proposer or reuse/fusion that avoids rereading the full NextN weights,
plus the planned 25--30% verifier reduction.  Merely repartitioning the 48 cores
or adding a second communication stream is insufficient on this memory-bound
node.

#### Post-async verifier and proposer probes

K=3 cannot replace K=4/K=5 as the route to 50 tok/s.  A 128-token forced-full-
accept run measured 68.99 ms/round; even three emitted tokens per round cap at
43.5 tok/s before proposer cost.  A current exact K=5/256 run retained the
established oracle SHA256
`7b86e9830096198c4066689d487ad18b3cd6efbad02626494a0d3fb9460d2f14`,
but the present allocation delivered only 531--561 GB/s/node effective local
bandwidth versus the earlier accepted 868--870 GB/s/node.  Its 93.09 ms verify,
48.57 ms draft, and 32.84 tok/s result should therefore not replace the
previous sustained 53.43 tok/s headline.

Two further algorithmic probes were rejected.  A snapshot-free vector SSM
ceiling reduced K=5 verification by about 8% (roughly 94 to 86.94 ms and 55.05
tok/s with drafting removed).  Full lazy rollback/replay was exact at both 128
and 256 tokens, including the established `7b86e983...` oracle, but eight
rejected-round replays raised batch calls from 55 to 63, cost 79.37 ms/round,
and reduced throughput to 24.17 tok/s.  Direct per-token recurrent snapshots
remain substantially faster.  Connecting the existing BF16-PV fused local-argmax helper to the
NextN vocabulary head did preserve the complete 256-token oracle, but raised
draft time from 48.57 to 57.50 ms and reduced throughput from 32.84 to 30.02
tok/s.  Both code experiments were fully removed.

Extending the direct TP4 all-to-all cutoff from 8192 to 32768 floats was also
rejected.  It moved the 25,600-float K=5 verifier reductions off the two-round
tree, but the 128-token SHA256 changed from the established `6b136ca0...` to
`9b71fe7c...`.  The direct peer-put fold is therefore retained only for the
validated small-buffer range; K5 residual reductions remain on the exact tree.
