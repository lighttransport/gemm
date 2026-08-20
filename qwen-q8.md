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

### Decode implementation and result

An experimental pair-interleaved eight-row BF16 layout and SVE kernel were also
implemented.  It reuses each FP32 activation load across eight output rows, but
a final comparison found divergence from row-major at generated token 14.
Consequently `TP_STAGE_BF16_PV=0` is the launcher default and all accepted
numbers below use the source-equivalent row-major layout.  The experimental
path remains available only for diagnosis.  Software prefetch was slightly
slower end to end and also defaults off.

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
