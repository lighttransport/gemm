# DeepSeek-V4.1-Flash A64FX memory design

Latest continuation: [20/30/40 tokens/s implementation plan](#203040-tokenss-implementation-plan).
The earlier pause was superseded by the request to pursue INT8 SDOT decode.

## 20/30/40 tokens/s implementation plan

Agreed 2026-09-12, recorded before implementation. Keep 12 A64FX nodes and
single-request decode around 1K history, all 40 layers, 64 attention heads,
top-6 routed experts, top-512 selected rows and the 128-token window. Maintain
two separately reported tracks: numerically validated and speed-first
approximate. Exact-output speculation is a third, separately measured path.
Current unprofiled INT8 runs reach 10.36–10.39 tokens/s but fail the numerical
gate; the FP8 regression control reaches 6.92 tokens/s. Neither is official
GPU parity. Preserve the original math path as the comparison control.

### Milestone budgets

These mutually exclusive milliseconds/token are engineering targets, not
predictions or measured results. Apply the budgets independently to each
quality track; never promote a failing approximate run as validated.

| Component | 20+ target | 30+ target | Ordinary 40+ stretch |
| --- | ---: | ---: | ---: |
| Attention projections and preparation | 14 | 8 | 6 |
| Sparse attention and index | 9 | 5.5 | 4 |
| Routed and shared experts | 12 | 8 | 6 |
| mHC mixing | 4 | 2.5 | 2 |
| Synchronization | 4 | 3 | 2 |
| Other | 6 | 5 | 4 |
| **Total ms/token** | **49** | **32** | **24** |
| **Implied tokens/s** | **20.4** | **31.3** | **41.7** |

The 800 GB/s / 16 GB single-node weight-only estimate is 50 tokens/s. The
profile counts 13.88 GB of weight operands globally and about 9.81 GB on the
current serial dense-owner/EP critical path. Neither is a hardware-counter
HBM measurement. Other nodes wait during owner-only attention; multiplying
bandwidth by twelve is inappropriate. Eliminating the entire 21.58 ms INT8
kernel aggregate alone cannot take the present roughly 98 ms step to 20+.

### Stage 1: profile, attention kernels, projection reuse, mHC and transport

1. Split sparse attention into QK, softmax and PV, and separate packing,
   activation quantization, OpenMP launches and collective arrival skew.
   Reconstruct the critical path without summing nested spans or rank waits.
2. Add FP32 QK tiles of four heads by two keys while preserving the existing
   two-accumulator dot order. Benchmark PV tiles of 2/4/6 heads by 64 channels
   and retain the fastest complete operator passing its quality track.
   Decode selected rows once into bounded scratch; do not expand full history.
3. Adapt ideas from
   `~/work/clair/a64fx/a64fx/llm-guided-opt/attention_decode_a64fx.c`:
   reuse K across heads and V across head/channel tiles. Its 12-head, D=256
   geometry and single-valid-key shortcut are not DS41F semantics: preserve
   the DS41F sink in the maximum and denominator even with a single key.
4. Test SDOT QK using exact E2M1*2 integer values with original compressed-KV
   group-16 scales, and quantized queries/raw keys. Rescale each dot block to
   FP32 before combining scores; block-dependent scales preclude one global
   integer coefficient. Initially retain FP32 PV.
5. Compare libm, corrected FEXPA, integer polynomial-2 and integer affine
   softmax independently. Reference
   `~/work/clair/a64fx/a64fx/llm-guided-opt/int_exp2_sdot_a64fx.s`
   and `integer-exp2-sdot.md`. Their quoted timing is QLAIR simulation, not
   native evidence; affine/poly2 error bounds are about 2.98%/0.375%.
   Apply stable maximum subtraction including the sink, multiply by log2(e),
   extend the original [-16,0] exponent domain to [-31,0], underflow to zero,
   zero masked entries and use 64-bit Q31 denominator sums. A -16 clamp would
   add a substantial floor over 640 rows. Check overflow, tails, ABI register
   preservation, all-masked/sink-only cases and real model inputs.
6. Reuse activation quantization across compatible projections, shared W1/W3
   and routed experts without moving FP8/BF16 rounding boundaries. Keep fresh
   anonymous INT8 pages. Measure cold complete operators including conversion
   and packing. Fuse MXFP4 W1/W3 with original group-32 scales; the CLAIR
   `fp4_i8_sdot_a64fx.s`/`sdot_quant.h` integer-grid format, nibble order and
   ties-away rounding differ from this checkpoint and need adaptation.
7. Retain the ordered mHC control; gate register-split FP32/FP64 matrix sums
   and Sinkhorn variants on full-model fixed-history logits. Local norm tests
   alone did not detect the earlier split-K model regression.
8. Replace world residual broadcast with owner-to-next-owner handoff (last
   layer to head rank 11). Pack already-BF16 residual/FFN vectors losslessly;
   retain FP32 mixing coefficients and route weights, and signed-zero
   normalization. Publish source-cache bytes directly instead of float codes.
   Keep robust uTofu ACK reductions initially and all MPI on the main thread
   under MPI_THREAD_FUNNELED. Do not disable acknowledgements to hide waits.
9. If this does not meet the 49 ms budget, proceed to TP2 then TP4 without
   changing the fixed backbone or loosening the validated-track gate.

### Stage 2: dense tensor parallelism for 30+, ordinary 40+ stretch

Add TP=1/2/4, with layer owner `layer % 12` and contiguous group base
`(owner / TP) * TP`. Fixed rank offsets select 32/16 heads and 4/2 whole
WO-A groups for TP2/TP4. The original dense and shared payload is balanced
so these layouts retain the same per-rank total original weight bytes as
TP1; verify actual converted buffers and peak loading memory separately.

- Initially keep owner QA, KV compression and index. Publish group inputs;
  shard QB, RoPE, sparse attention and whole WO-A groups. Allgather the
  already-BF16 8192-element projected vector, row-shard WO-B with its full
  K dot order, then gather outputs to the owner for mHC and expert routing.
  Keep replicated packed source caches in the first TP implementation.
- Distribute index work by whole eight-row candidate blocks, then merge with
  original score/ID tie rules, selected-ID ordering and forced newest block.
- Row-shard shared W1/W3, gather its 2304-element hidden vector, and row-shard
  W2. Distribute the vocabulary head's 4040 blocks of 32 rows over all 12
  ranks; reduce max with smallest-global-ID ties. Gather full logits only
  for diagnostics. Avoid duplicating the shared backbone embedding/head.
- Introduce explicit persistent OpenMP team kernels with main-thread MPI;
  preserve the existing control and measure launch savings. For the 24 ms
  stretch, test column-sharded WO-B/shared W2 and combined communication only
  behind independent numerical gates because their reductions change order.
- Expose explicit runner arguments for kernels/math/TP and an INT8 tensor
  allowlist. Version stage metadata with global shapes, local ranges, group
  mapping and source hashes; retain TP1 compatibility. Stage new dense
  layouts separately while reusing expert and Engram shards.
- Convert after reading `/local` when preprocessing is negligible, as with
  the current 0.27–0.50 s INT8 conversion versus 53–57 s loading. If new
  preparation is material, write versioned INT8 artifacts beside the shared
  original weights, then stage them. Account for every workspace, cache and
  MTP buffer with a 2 GiB minimum MemAvailable; no full-model duplicates.

### Stage 3: exact-output speculative decode for 40+

The checkpoint contains three DSpark/MTP stages, fixed draft block size five,
noise ID 128799, main hidden taps at layers 37/38/39 and Markov rank 256.
MTP tensors total 7,932,874,632 bytes; draft MoE uses 128 experts/top-3 while
the backbone remains 384/top-6. Implement checkpoint behavior from local
`inference/model.py`, including tap positions/dtypes, main projection,
stage-specific attention masks, mHC, Markov bias and confidence output.

Shard draft experts with EP and dense weights with TP, reuse backbone
embedding/head, and admit all state before loading. Always calculate the
checkpoint's full five-position draft block; benchmark verified prefixes of
2/4/5. Add verifier microbatches of 1–6 with weight reuse and original per-token
dot, reduction and BF16 boundaries. Existing batched GEMM changes reduction
order and is not automatically an exact verifier; use bounded packed tiles.

Given known seed x0, draft x1..xd and verify inputs x0..xd. Accept the matching
greedy prefix of length a, emit a accepted tokens plus the verifier bonus,
and commit a+1 input states; the bonus is the next uncached seed. Confidence
never bypasses verification. Preserve per-token causal index/cache views.
Journal or shadow overwritten window slots, compressed append counts,
candidate and pool state, Engram history and prefetch generations; roll back
the rejected suffix. MTP main-KV receives every committed main hidden position;
draft noise-KV stays temporary. Handle wrap, EOS and output limits explicitly.

Measure actual emitted tokens divided by draft+verify+commit time, separately
from ordinary decode. Three emitted tokens need a cycle below 75 ms for 40+.
Select the fastest measured prefix, preferring fewer drafts on a tie. Default
speculation off until emitted tokens and committed state match the selected
validated sequential verifier, including forced rejection at every position.

### Validation and execution order

- Kernel tests use references and real inputs, masks/sinks/tails/canaries,
  conversion costs and overflow bounds. Test TP1/2/4 across every owner,
  delayed-rank collectives, BF16 transport, ties, window/compression and top-k
  boundaries. Finish the corrected independent nine-position reference.
- Compare identical input histories at early positions and positions
  1000..1008, plus chat/code prompts. Report cosine, relative RMS, argmax and
  every approximate-track failure; token agreement alone is insufficient.
- Run three uninstrumented 1100-output repeats for each retained milestone,
  measuring positions 1000..1104. Every repeat must exceed the claimed target.
  Report p95, minimum memory, prompt dependence, binary/staging hashes and a
  separate profile. Allocating 1M cache is not running at 1M history.
- Execute and commit coherent stages in this order: this document; detailed
  profile/kernels/transport; TP2; TP4/shared/head; ordinary 40+ experiments;
  exact speculative 40+. Update this document with results and remaining work.
  Use one MPI program per allocation, detached immutable binary/script
  snapshots, remote SHA verification, `/local` or repository `tmp/`, and check
  allocation lifetime before each long experiment. Do not edit active scripts.

## Implementation continuation: attention tiles and dense TP, 2026-09-12

The plan above was committed as `470c45a1` before kernel changes. Experimental
options remain explicit; none of these measurements establishes 20 tokens/s.
The active allocation is still 51569201 (12 nodes, 2 GHz, ends 00:00:29 JST
September 13). Original TP1 staging remains intact; TP2/TP4 use separate
`/local/u14346/ds41f-51569201/tp{2,4}/rank<R>` directories and reuse local
expert/Engram files. `stage_tp.py` records versioned ranges, checkpoint header
hashes and SHA256 sidecars for copied shards. Runtime validates TP/rank and
weight row-range metadata before conversion.

Implemented stages and initial evidence:

- The finer original attention profile reports maximum worker work durations
  of 5.185 ms QK, 1.012 ms softmax and 5.066 ms PV per token around 1K.
  These maxima are nested work measurements, not three additive wall spans.
  Tiled profiles measure elapsed phases including their barriers.
- `--sparse-tile 1/2/4/6` adds split-phase attention; 2/4/6 use four-head,
  two-key QK and corresponding multi-head PV. Zero retains the original path.
  Tile four reduced the complete sparse span 12.283 to 8.818 ms/token, and
  full profiled latency 98.596 to 94.418 ms (10.591 tokens/s). Fifty geometry,
  mask, sink, tail and canary cases pass bitwise against the existing control
  on A64FX and native scalar builds. All 1105 token triples and nine saved
  logits also match the INT8 control exactly (`sparse-tile4-full-v1`).
- `--compact-comm` uses byte source publication, mixed BF16/FP32 FFN packets,
  and owner-to-next-owner residual handoff (layer 39 to head rank 11).
  Packing rejects non-BF16 inputs; route weights and mHC coefficients stay
  FP32. MPI stays on the main thread and robust uTofu ACK sums are retained.
  All-owner delayed-rank, self-handoff, nonparticipant, signed-zero and canary
  tests pass. The long run retains all 1105 triples/nine logits exactly and
  improves total inference 101.311 to 99.851 seconds (`compact-full-v1`).
- RoPE rounds only its modified 64-coordinate suffix, since every caller has
  already rounded the untouched coordinates to BF16. The combined exact
  changes preserve all nine FP8 control logits and all 1105 INT8 token triples
  plus nine logits (`stage1-fp8-exact9-v1`, `stage1-int8-exact-full-v1`).
- `--hc-matvec 1/2` tests register-accumulated FP32/FP64 sums; zero retains the
  ordered FP32 control. FP64 accumulates rounded FP32 products. Both pass
  bounded local tests but **fail full-model numerical gates**: early nine
  positions versus INT8 control have minimum cosine 0.993941/0.990968 and
  maximum relative RMS 11.935%/15.405%. FP32 at fixed 1K history reaches
  cosine 0.943039 and RMS 33.972%; its mHC span falls to 3.40 ms/token. Keep
  both clearly marked approximate; local norm agreement is insufficient.
- `--dense-tp 2/4` distributes QB, whole WO-A groups, row-sharded WO-B and
  shared W1/W3/W2 within contiguous groups. BF16 allgathers retain complete
  input K ranges for WO-B/W2, and gathers return outputs to the owner. QA,
  compression and index remain owner operations. Vocabulary rows are split
  in blocks of 32 across all 12 ranks; MAXLOC preserves smallest-ID ties,
  and only diagnostic runs gather full logits. TP requires shared overlap.
  Both TP sizes pass every-owner transport tests and reproduce all nine FP8
  and INT8 control logits bit-for-bit. TP2's first long INT8 run also matches
  all 1105 token triples and nine logits, reaching 89.923 ms/token around 1K
  (11.121 tokens/s), 97.700 seconds total, minimum final memory 3.861 GB.
  The head falls to 0.260 ms/token, while attention still takes 40.032 ms.
  Further TP4 performance and 1K logit checks are running.
- `--sparse-math 1/2/3` independently selects corrected FEXPA, Q31 integer
  polynomial-2 or affine softmax with a nonzero sparse tile. Zero retains
  libm. Integer kernels extend the CLAIR domain to -31 and underflow smaller
  scores to zero; the denominator is uint64. Exhaustive Q16 domain and
  extreme-value tests match the scalar integer oracle, respect error bounds,
  and pass monotonicity, masks/sink, single-key, tail and canary checks on
  A64FX and native builds. Full-model numerical/performance gates are pending.

Evidence is under `tmp/ds41f/job51569201/`, including `sparse-profile-v1`,
`sparse-tiles-v1`, `compact-check-v1`, `stage1-runs-v1`, `exp-check-v1`,
`tp-staging-v1` and `tp-short-runs-v1`. Every remote runner uses an immutable
snapshot and verified binary hash. Minimum final memory in the completed long
runs remains above 3.98 GB. The corrected independent nine-position NumPy
reference is running in `reference9-v1`; do not claim it has passed yet.

Build/check commands for this stage:

```sh
TMPDIR="$PWD/tmp/ds41f" make -C a64fx/ds41f a64fx \
  A64FX_CC=fccpx A64FX_MPICC=mpifccpx \
  A64FX_CFLAGS='-Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast -Wall -Wextra -Wpedantic'
python3 a64fx/ds41f/test_stage_tp.py
# Inside the allocation, with one program at a time:
OMP_NUM_THREADS=48 OMP_PROC_BIND=close OMP_PLACES=cores ./test_sparse_tiles
./test_exp2
mpiexec -np 12 ./test_broadcast
mpiexec -np 12 ./test_tp_comm 2
mpiexec -np 12 ./test_tp_comm 4
```

Remaining planned work includes activation-quantization reuse, fused expert
kernels, SDOT sparse QK with complete packing costs, distributed index work,
persistent OpenMP teams, any justified changes to projection reduction order,
and the separately validated DSpark speculative path. MTP/INT8 batched GEMM
and speculative rollback are not implemented by this continuation yet.

## Implementation continuation: index, experts and input reuse, 2026-09-12

The following results supersede the pending TP4 results above. All rates in
this table are instrumented samples at positions 1000..1104 on job 51569201,
2 GHz, twelve nodes, TP4, the fixed six-token capital prompt and 1100 generated
outputs. Independent uninstrumented repeats are running; these are not 20+
claims. Every exact-control row below preserves the corresponding 1105 token
triples and nine saved logits bit-for-bit.

| Configuration | ms/token | tokens/s | Control / limitation |
| --- | ---: | ---: | --- |
| FP8, TP4 initial | 105.794 | 9.452 | Nine fixed-history 1K logits match FP8 |
| INT8, TP4 initial | 87.143 | 11.475 | INT8 control |
| INT8, active OpenMP wait policy | 81.166 | 12.320 | INT8 control |
| INT8, vector index heads + fused expert pair | 76.466 | 13.078 | INT8 control |
| Above + fresh attention scratch pages | 76.958 | 12.994 | No clear gain |
| Above + input quantization cache | 75.632 | 13.222 | INT8 control |
| Cache path + approximate mHC mode 1 | 72.244 | 13.842 | Fails mHC numerical gate |

The INT8 control itself still fails the FP8 numerical gate. Bitwise regression
checks of the new transformations do not make INT8 numerically validated.
TP4 fixed-history replay also preserves nine INT8 logits at positions
1000..1008 exactly. The retained FP8 path keeps original checkpoint weights
and rounding. The completed corrected independent nine-position reference
will be recorded separately when available.

- `--index-head-tiles` transposes the 32x128 query once and evaluates heads in
  SVE lanes. Each lane retains the 128-term ordered FP32 sum, BF16 score and
  weighted-score boundaries; ordered head accumulation retains selection
  ties. Twenty-four native/A64FX mask, tail and canary cases pass bitwise.
  At 1105 candidate rows the standalone score kernel falls from 460.85 to
  42.82 microseconds. Full-model index scoring falls to 0.281 ms/token.
  Distributing this now-small score phase would likely add more communication
  than it removes at 1K; reconsider only for longer measured histories.
- Top-k selection skips heap construction when every valid row fits and uses
  bounded membership flags for ascending selected IDs below 4096 rows.
  Larger histories retain the sorted fallback. Existing cache/selection
  boundary tests pass.
- `--expert-fused 1/2` shares input loads for routed W1/W3 while preserving
  their FP32 accumulation order and original MXFP4 group scales. Eighteen
  geometries with both tiles pass bitwise. Cold 2304x5120 paired projections
  take 130.84 us originally, 106.18 us for tile one and 127.59 us for tile two;
  tile one is retained. Packing and SDOT experiments remain separate.
- `--linear-input-cache` retains up to four prepared activation vectors keyed
  by complete input contents, length, INT8 block size and FP8 activation
  rounding boundary. Reused addresses with changed values miss correctly;
  grouped WO-A keeps its original absence of FP8 activation rounding. The
  cache stays below 1 MB/rank. Raw/FP8/grouped inputs, changed contents,
  repeated hits, changing block sizes and nonfinite rejection pass native
  and A64FX tests. Prepared and ordinary GEMV match in all 162 INT8 cases.
- `--attention-local-pages` places only bounded selected-row scratch on fresh
  anonymous pages. It passes the long exact regression but showed no clear
  speed gain and is omitted from the retained uninstrumented repeats.
- Corrected FEXPA, integer polynomial-2 and affine softmax all fail the early
  nine-position full-model gate versus the INT8 control: minimum cosine
  0.994960/0.994632/0.992309 and maximum relative RMS
  10.066%/10.451%/12.862%. Keep `--sparse-math 0` for the retained control.
- The separate `ds41f_int8_matmul[_prepared]` kernel accepts 1..6 tokens and
  reuses each four-row weight tile while preserving sequential GEMV lane
  and reduction order. All 378 batch/shape/group/stride/tail/canary cases pass
  bitwise on native and A64FX builds. Cold complete INT8 operators, including
  input quantization, improve 1.2–1.5x for batches two through four on TP4
  WO-B/QB/WO-A geometries; five/six gain little. FP8 activation rounding is a
  caller boundary outside this microbenchmark. This is verifier groundwork,
  not an integrated batched verifier or speculative token-rate result.

The cached full profile still spends 29.600 ms in attention, 20.887 ms in
routed/shared experts and 9.914 ms in mHC mixing. Attention's nested sparse
span is 4.330 ms; index preparation plus scoring/selection is 3.115 ms.
The slowest routed rank spends 7.664 ms in W1/W3, 4.648 ms in W2 and 2.214 ms
in activation quantization. These measurements motivate packed expert SDOT,
quantization scheduling/reuse and further projection work before speculation.
Minimum final MemAvailable for the cached run is 3,884,056,576 bytes.

Evidence: `tmp/ds41f/job51569201/{index-pair-check-v1,cache-check-v1,batch-check-v1}`
and the `tp4-*` run directories, including `comparison.json`, `quality.json`
and `profile-1k.json`. Builds use the warning flags above; remote checks run
with `OMP_NUM_THREADS=48 OMP_PROC_BIND=close OMP_PLACES=cores OMP_WAIT_POLICY=active`.
The immutable retained driver is `retained-runs-v1/run-v1.sh`. The next
12-node allocation is **51575979**, running 22:50:08–04:50:08 JST, bridge
42395/32395/21266; safe bounded original/TP4 staging is running in
`tmp/ds41f/job51575979/staging-v1`. Do not overlap MPI programs per allocation.

Still outstanding: SDOT sparse QK, packed expert experiments, persistent
OpenMP teams, broader chat/code quality checks and the full DSpark/MTP
staging, draft execution, causal batched verifier and state rollback. No MTP
inference or speculative speed claim is implemented by the batch kernel alone.

## Implementation status, 2026-09-12 (job 51562789)

The 12-node runner now executes all 40 layers with real resident weights and
Engram offload. The six-token prompt `The capital of France is` (including
BOS) generated ` Paris. In the` across four outputs. All ranks completed.
The completed independent NumPy reference agrees on all nine next-token choices
in `infer-v2`. Logit cosine similarity ranges from 0.994041139 to 0.999988026
over those positions, and five fail the 0.999 comparison gate. Thus full numerical
validation is **not passed**; this is neither bit-exact nor official GPU parity.
The continuation below corrects an mHC arithmetic boundary and passes a fresh
three-position full-model reference check. A complete nine-position rerun with
that corrected reference remains outstanding; no reference process is left running.

`infer-v1` and `infer-v2` logs under `tmp/ds41f/job51562789/` record the same
nine input positions with tracing and logits enabled: 11.339 s versus 3.129 s.
The improvement comes from bitwise FP8 conversion and an SVE BF16-weight /
FP32-input path. Resident loading took 52–69 s/rank in v1. Subsequent allocator
demand-paging runs use `XOS_MMM_L_PAGING_POLICY=demand:demand:demand` so parallel
first-touch is effective; default prepopulation defeated NUMA placement.

A 160-output run (`infer-v3`) completed 165 input positions in 58.245 s,
including window wraparound. It committed the configured 1M-token cache pages
up front; minimum final MemAvailable exceeded 3.8 GB. This proves cache
allocation plus bounded execution, **not execution at 1M-token history**.
The longer `infer-v4` and `infer-v5` runs each completed 1,105 input positions
and generated 1,100 outputs with EOS stopping explicitly disabled. All 12 ranks
finished, and all 1,105 input/next-token triples were identical between runs.
Time fell from 530.402 s to 357.689 s after exact packed-row decoding and
parallel selected-row unpacking: 1.48x faster, approximately 3.08 generated
tokens/s including the six-token prefill but **excluding resident loading**.
V5 loading took 51.4–66.8 s/rank; minimum final MemAvailable was 3.809 GB.
The completed `infer-v6` repeat took 319.984 s (3.44 generated tokens/s), and
all 1,105 input/next-token triples match v5 exactly. All 12 ranks finished.
These runs cross window wraparound and the 512-entry index top-k boundary for
both compression ratios. They still do not establish actual 1M-history speed.

`infer-math` additionally completed 42 positions / 32 outputs in 11.910 s and
answered `5` to a plain-completion arithmetic prompt (then continued its
question/answer pattern). This is not a chat-template quality test.
`infer-chat` completed a checkpoint-formatted chat prompt with 45 generated
tokens, stopping at EOS, in 16.453 s. It explained the blue sky through
wavelength-dependent scattering; all 12 ranks finished. CPU performance
targets are not yet met by the complete quantized model kernels.

The exact residency plan is generated by `a64fx/ds41f/plan_residency.py` and
saved, with corrected index-key accounting, in
`tmp/ds41f/job51562789/residency-v2.json`.
Experts use `expert_id % 12`, dense layers use `layer % 12`, embeddings rank
zero, head/final norm rank eleven. Compressed HBM weights range from
24,574,984,584 to 26,080,018,272 bytes/node. Engram consumes approximately
16,896,502,920 bytes/node on disk. Dense ownership distribution and rank-local
resident indexes completed on all 12 nodes (`phase2/dense.rank*.log`).

Corrected packed batch-one, 1M-token KV accounting is 935,936,000 bytes
**globally**, or that much per rank if conservatively replicated:

- four compressed KV sources: 3 at ratio two, 1 at ratio one;
  512 values/row, FP4 plus one E4M3 scale per 16 values;
- four index-key owners (the KV sources): 3 at ratio two, 1 at ratio one;
  128 values/row, FP4 plus one E8M0 scale per 32 values. The other four index
  query sources reuse these keys; they do not allocate additional key caches;
- all 40 sliding windows: 128 rows of FP8 512-wide KV plus group-32 scales.

The runner stores compressed KV/index rows packed, but retains FP32 sliding
windows as an execution buffer. Its corresponding allocation is 943,718,400
bytes plus candidate masks and small pooling/selection state. The index top-k
of 512 is not a key dimension. Other FP32/BF16 buffers,
candidate lists, pool state, allocator/communication buffers and temporary
dequantization must be accounted separately. Use actual node MemAvailable,
not nominal 32 GiB alone, when admitting resident weights and cache.

Validated on the initial A64FX compute node:

- SVE512 BF16/RMSNorm dispatch with tails and canaries;
- exact FP8 scale-block and finite-code conversion regressions;
- real layer-zero expert projection: max absolute error 1.13249e-6 against
  scalar decoding/FP64 accumulation;
- real expert w1/w3/SwiGLU/w2 chain: max absolute error zero against the CPU
  reference for the tested input, including dynamic FP8 activations and BF16
  boundaries (not a comparison with the official GPU model);
- routing, mHC, RoPE and sparse-attention unit cases;
- Engram hash compressed history/padding/bucket offsets; tokenizer metadata
  has 129280 entries, 99092 compressed IDs, and exact checkpoint bucket totals.

Representative 48-thread dummy GEMV measurements during background staging:

| Format and matrix | Time | Useful GFLOP/s | Effective weight GB/s |
| --- | ---: | ---: | ---: |
| MXFP4 2304x5120 | 59.806 us | 394.494 | 104.787 |
| MXFP4 32768x1280 | 200.950 us | 417.448 | 110.884 |
| FP8 2304x5120 | 125.698 us | 187.696 | 93.940 |
| FP8 32768x1280 | 475.835 us | 176.292 | 88.232 |

Logs: `tmp/ds41f/job51562789/quant-bench-inline.log`. Same tensors are reused
50 times; effective bandwidth is bytes/time, not an HBM counter measurement.
Inlining E8M0 scale conversion removed a major bottleneck. A direct bitwise
FP8 SVE decoder was slower (193.711 us at 2304x5120) and was removed in favor
of gathers. The 80% compute/90% bandwidth targets are **not achieved**. GEMV
has low arithmetic intensity; a separate batched GEMM benchmark is needed to
evaluate a compute-utilization target meaningfully.

Post-staging rank-local suites, staged Engram smoke tests, the six-expert
uTofu combine versus an MPI oracle, and dense distribution passed on all 12
nodes. Evidence is in `tmp/ds41f/job51562789/phase2/`. The original waiting
script failed after being edited while active; the stable
`run_bringup_phase2.sh` completed the sequence. Do not edit active scripts.

Remaining gates are broader numerical/token validation, efficient batched
prefill integration, and the kernel utilization targets. KV persistence and
streaming are not implemented. The independent NumPy reference is not an
official GPU execution oracle.

Hardware controls on the initial node (48 threads, 2 GHz, demand-paged NUMA
first-touch): up to 6.122 TFLOP/s with 24 independent SVE FMA chains (99.6%
nominal), and 900.080 GB/s streaming reads from 2 GiB (87.9% of nominal
1024 GB/s). Larger streams amortize launch/reduction overhead; 512 MiB measured
855 GB/s with the same eight-accumulator loop. Manual prefetching did not help.
These are controls, **not model throughput**. The quantized batched GEMM
prototype now reaches 4.10–4.22 TFLOP/s including quantization/packing at
batch 768 (66.7–68.7% nominal). Its packed microkernel phase alone reaches
approximately 5.3 TFLOP/s for M=2304/K=5120, but that excludes required packing
and is not the complete operator. Sparse attention at 64 heads, dimension 512, 640 selected rows
improved from 3.250 ms to 0.628 ms in warmed best-of-repeat measurements,
with max absolute error 1.49e-6 versus the FP64-score reference.

Additional validation/optimization evidence:

- `batch-shared-quant.log`: SVE group-32 activation quantization matches the
  scalar path exactly over 12,288 groups spanning BF16 codes, random FP32
  inputs, scaled FP8 midpoints, signed zeros and rejected nonfinite inputs.
  The same helper is now used by decode; the completed `infer-v6` long repeat
  agrees with v5 on all 1,105 input/next-token triples.
- `quant-grouped.log`: grouped FP8 output projection matches eight separate
  calls exactly, improving 407.408 us to 296.219 us (1.38x) in the dummy case.
  The alternative direct-bit FP8 GEMV decoder remained slower than lookup
  gathers and is not enabled.
- `attention-source-tests/`: six positions for each source layer 0/2/8/14/20
  completed. Independent NumPy comparisons for compressed sources 2 and 20
  give cosine 0.999999962 and 0.999999754, maximum absolute BF16 output
  differences 0.015625 and 0.03125, respectively.
- `attention-candidate-v2/`: real query projections with **synthetic** 32K
  history retain 2,048/4,096 candidate blocks at layer 20, including the newest;
  layer 24 selects exactly the 64 rows permitted by eight injected blocks.
  Later query layers now skip decoding/scoring masked-out rows. This is not
  a full-model 32K-history run.
- `cache-final.log`: top-2048 selection over 32,771 scores agrees with a full
  sort, including ties, NaNs, masked negatives and positive infinity.

Publish versioned test scripts as well as binaries: an in-place frontend edit
was not visible in the compute node's cached shared script. The first candidate
test launch therefore ran the old wrapper; only `attention-candidate-v2/`
contains the candidate tests. Compare frontend/compute SHA256 before execution.

### Continuation: mHC arithmetic and bounded operator replay

Job 51562789 was resumed through the existing bridge at local port 42393,
login reverse port 32393, on service node `a25-4009c`. Its scheduled end is
2026-09-12 17:57:45 JST. This allocation retains the previously staged rank
directories; do not reuse those paths after the allocation ends.

The mHC post operation now sums residual terms before adding `post * x`,
matching the checkpoint's `inference/model.py` expression. FP32 products are
rounded separately, rather than contracted into that sum. The former ordering
lost a small residual under cancellation: the new regression expected 1 and
returned 0 on the old implementation. Cancellation, product rounding and
in-place updates pass on A64FX; the old/new cancellation comparison also
reproduced on the frontend. The NumPy reference now uses the same explicit
multiply/reduce/add boundaries instead of a BLAS matrix multiply.

`--dump-prefix` / `--dump-count` provide bounded owner-written intermediate
records (at most 64 positions). `replay_intermediates.py` reads original
safetensors and recomputes operators with the recorded inputs. It checks exact
routing IDs plus cosine and relative RMS gates, rebuilding attention history
from the source-layer inputs. This isolates individual operators from upstream
rounding drift; it does not replace an autoregressive comparison. Nine positions
across all 40 layers produce 120,318,480 dump bytes, and their shared-storage
writes must be excluded from speed measurements.

Evidence under `tmp/ds41f/job51562789/`:

- `resume-numerics-v1/replay-early.log`: 408/408 checks passed over three
  positions and layers 0–8, with minimum cosine 0.999997493. This initial
  replay used the earlier NumPy mHC expression and the cosine-only gate.
- `resume-mhc-v2/ops-before-x86.log`, `ops-after-x86.log`, and
  `resume-packets-v4/ops.log`: the cancellation regression fails before the
  fix and the updated operator suite passes on frontend and A64FX.
- `resume-packets-v4/comparison.json`: merging each layer's input/route and
  residual/pre-mix broadcasts removes 80 collectives per token. All nine
  logit dumps and all 360 intermediate files are bitwise equal to the
  corrected mHC baseline; both runs finished on all 12 ranks.
- `resume-packets-v4/replay.log` and `replay-summary.json`: the updated replay
  passed 726/726 checks over three positions and 16 layers
  (0–8, 14, 20, 24, 28, 32, 36, 39). Minimum cosine is 0.999998019 and maximum
  relative RMS error is 0.001992132 (0.199%). All 1,966,080 values in 96 mHC
  post comparisons match exactly with identical inputs. These checks use
  the explicit NumPy mHC expression and both numerical gates.

The new runner and replay workflow are documented in `a64fx/ds41f/README.md`.
Source snapshots and SHA256 manifests are retained beside the run binaries.

The detached `resume-long-comparison-v1.sh` ran the corrected mHC baseline and
combined-broadcast version sequentially, at 48 threads/node with demand paging,
2 GHz, `--generate 1100 --ignore-eos --max-context 1048576`. Both completed on
all 12 ranks. These times include the six-token prefill and exclude resident
loading; diagnostics were limited to nine logit dumps, with no layer dumps.

| Run | Elapsed seconds | Generated tokens/s | Minimum final MemAvailable |
| --- | ---: | ---: | ---: |
| `resume-long-mhc-v2` | 321.126 | 3.425 | 3.860 GB |
| `resume-long-packets-v4` | 309.411 | 3.555 | 3.858 GB |

This pair measured a 1.03786x improvement (3.8%). All 1,105 input/next-token
triples and the nine logit dumps match exactly; the logits also match the
corresponding runs with intermediate dumping enabled. Evidence:
`resume-long-comparison.json`. The run crosses both compression ratios'
512-row selection boundary; its history is 1,105 positions, not 1M positions.

Cross-build command for the changed runner and operator regressions:

```sh
TMPDIR="$PWD/tmp/ds41f" make -C a64fx/ds41f ds41f_run test_ops \
  A64FX_CC=fccpx A64FX_MPICC=mpifccpx
```

The validated combined-broadcast runner SHA256 is
`3263ec1f7c0c70883e9cd6aba2f8be9c8993074038750219ce26813d56c43e2a`.

The same binary subsequently passed two bounded chat smoke runs, each on all
12 ranks with `--max-context 4096 --generate 96` and normal EOS stopping:

- `resume-chat-sky-v4`: 16 prompt tokens, 47 generated tokens including EOS,
  16.645 s after loading. It correctly explained shorter blue wavelengths
  scattering more strongly than red wavelengths.
- `resume-chat-math-v4`: 18 prompt tokens, 2 generated tokens including EOS,
  4.951 s after loading. The response to `What is 17 times 23? Reply with
  just the integer.` was exactly `391` followed by EOS.

Both prompts use the checkpoint's chat formatter. The immutable launcher is
`resume-chat-comparison-v1.sh`; decoded responses and completion records are
saved in `resume-chat-summary.json`. These are smoke checks, not a broad
quality evaluation.

The corrected independent reference completed all 40 layers for the first
three prompt positions, starting from the embedding and carrying its own
state. `resume-packets-v4/compare-corrected-reference.log` passes the unchanged
0.999 cosine plus argmax gate:

| Position | Reference / A64FX next token | Logit cosine |
| --- | ---: | ---: |
| 0 | 5 / 5 | 0.999988319 |
| 1 | 41079 / 41079 | 0.999947498 |
| 2 | 294 / 294 | 0.999578117 |

The reference used `resume-mhc-v2/source/reference_numpy.py`,
`resume-mhc-v2/prompt-first3.ids`, and `--generate 1`, with four BLAS threads
on the frontend. Its log and logits are under `resume-mhc-v2/reference*`.
Comparison command:

```sh
OPENBLAS_NUM_THREADS=2 python3 a64fx/ds41f/compare_logits.py \
  --reference-prefix tmp/ds41f/job51562789/resume-mhc-v2/reference \
  --actual-prefix tmp/ds41f/job51562789/resume-packets-v4/logits --positions 3
```

This is a full-graph check for three positions, distinct from the replay that
injects recorded inputs into individual operators. It does not establish the
complete nine-position numerical gate, GPU parity, batched prefill performance,
or actual 1M-history execution. All continuation tests finished; job 51562789
was left idle with its staged shards and working bridge available.

## Single-request profiling at 1K history (job 51562789)

The performance target is **20+ decode tokens/s at approximately 1K actual
history**, clarified on 2026-09-12. This requires less than 50 ms/token.
The six-token capital prompt with `--generate 1100 --ignore-eos` supplies the
same 1,105 input positions as the earlier baseline. Report positions
1000–1104 (105 samples); retain `--max-context 1048576` for the existing
memory-admission check. This is not a 1M-history speed measurement.

The bounded profiler records main-thread spans per position/layer/rank with
`--profile-start 16 --profile-count 1089`; it writes binary arrays and JSON
metadata after the timed run. The report uses producer-owner spans and the
slowest parallel expert work. Collective remainders include rendezvous/skew,
not just transport. No new barriers or in-loop profile writes are introduced.

Initial measured critical path (`tmp/ds41f/job51562789/profile-v1/`):

| Stage | ms/token at positions 1000–1104 |
| --- | ---: |
| Attention, including projections/index/RoPE | 145.796 |
| Routed experts, slowest rank per layer | 34.458 |
| Shared experts, including output sum/round | 29.312 |
| mHC mixes/pre/post | 46.615 |
| Engram local fetch/decode + owner projection | 17.102 |
| Gate | 4.388 |
| Head norm/matvec/selection | 3.094 |
| Broadcasts + reduction rendezvous | 9.099 |
| Measured token total | **289.802 (3.451 tok/s)** |

The reconstruction differs from token timing by only -0.118 ms/token.
Nested FP8 GEMVs consume **70.485 ms/token**, reading 6.843 GB/token at an
effective 97.1 GB/s. The head matvec is only 1.885 ms/token. Scalar output
rounding alone costs 20.640 ms inside dense linears, plus substantial rounding
time inside RoPE, mHC and expert spans. Sparse attention costs 24.691 ms.
Six routed experts activate an average 4.871 of 12 ranks per layer; the busiest
rank owns 1.910 experts on average. Summing all ranks' collective wait would
misidentify producer/straggler waits as network traffic.

The profile run took 310.127 s versus 309.411 s for the uninstrumented baseline
(0.23% longer overall). All 12 ranks finished, all 1,105 input/next-token
triples and the first nine logit arrays were bitwise unchanged. Minimum final
MemAvailable was 3,871,997,952 bytes. Binary SHA256:
`62c3234211e87da23517e011f55492cc7d3c60b7a173a2b16e1716c4953fcf27`.

The first optimization replaces scalar BF16 round/conversion loops with SVE
integer rounding, preserving ties-to-even, NaN sign/payload handling and tails.
It also reuses that routine for routed-expert gate/up/output rounding.
`round-sve-v1/sve-test.log` passes 426,112 bit-exact cases, every BF16 upper
16-bit pattern at six rounding boundaries, and lengths 0–256 with canaries.
The full 12-node run takes 219.066 s; positions 1000–1104 average **206.638
ms/token (4.839 tok/s)**. All 1,105 token triples and the first nine logit
arrays remain bitwise identical. Minimum final MemAvailable is 3,860,201,472
bytes. This removes 83.165 ms/token at 1K, a 28.7% latency reduction.

Artifacts contain versioned source snapshots, binaries, SHA checks, launch
scripts, per-rank logs, `comparison.json`, `profile-1k.json` and
`report-1k.txt`. Reproduce the report on the frontend:

```sh
OPENBLAS_NUM_THREADS=2 tmp/ds41f/metadata-venv/bin/python \
  a64fx/ds41f/profile_report.py tmp/ds41f/job51562789/profile-v1 \
  --start 1000 --stop 1105 --json tmp/ds41f/job51562789/profile-v1/profile-1k.json
```

The parallel SwiGLU and mHC post updates (`pointwise-v1`) pass independent
bit-exact checks at lengths 1, 511, 512, 513, 2304, 5120 and 5123, including
in-place mHC and separate product rounding. Their full run takes 198.105 s;
positions 1000–1104 average **187.531 ms/token (5.332 tok/s)**. All 1,105 token
triples and nine logit arrays match the initial profile run exactly.

Dense FP8-to-BF16 expansion was tested and **rejected**. Although small
microbenchmarks improved, the complete runner regressed: `dense-bf16-v2`
took 261.109 s and `dense-adaptive-v1` took 240.602 s, versus 198.105 s for
the compressed pointwise runner. The expansion option and kernels were removed;
versioned experiment sources and logs remain under the job directory.

### Engram prefetch and receive-slot correctness

Fine timing attributes 11.920 of 11.980 ms of the Engram fetch stage to
`pread`, with only 0.051 ms for row conversion (`dense-bf16-v2`, same token
sequence). `--engram-prefetch` uses one worker/rank and two 24-by-256 FP32
buffers (49,152 bytes total) to fetch both layers' rows once their IDs are known.
The main thread waits at layers 1 and 14. The worker does not use MPI/uTofu;
MPI requests `MPI_THREAD_FUNNELED`, and profiler cursors are thread-local.

The initial `prefetch-v1` trial is **invalid**: it diverged at position 57 and
aborted with a nonfinite check after position 526. Prefetch exposed a latent
receive-slot reuse race in the no-ack recursive-doubling transport. A bounded
reproducer in `comm-skew-v1` delays rank 10 by 3 ms after the receive trailer
arrives, before reading its payload. Without acknowledgments, iteration 0
reads 866 instead of 66; with acknowledgments, all 300 reductions of 20,484
floats pass on all 12 ranks. Exact evidence is in:

- `comm-skew-v1/output.51562789/0/41/stderr.41.10`
- `comm-skew-v1/output.51562789/0/42/stdout.42.0`

The DS4.1 runner now enables the existing receive-acknowledgment path; the
shared all-reduce header and its bounded retry policy are unchanged. The
3 ms skew test validates this case, not arbitrary receiver delays.
`test_prefetch` checks 96 generations
against synchronous reads, zero fill for non-owned rows, short-read errors,
joining with pending work, and profiler isolation. It passes on A64FX and
the frontend. The corrected `prefetch-v2` run completes in 196.071 s, with
**186.392 ms/token (5.365 tok/s)** at positions 1000–1104. All 1,105 triples
and nine logits files are bitwise identical to `profile-v1`; minimum final
MemAvailable is 3,882,418,176 bytes. The remaining critical-path Engram wait
is 5.551 ms/token. This comparison includes the required acknowledgment cost.

### Sparse-attention loop

The weighted-value loop loads four adjacent SVE vectors from each selected
KV row, consuming an A64FX cache line while preserving the original per-lane
FMA order. The predicated fallback handles the remaining dimensions.
`test_ops` passes 48 reference cases around vector/block boundaries, including
empty, masked and duplicate selections with output canaries. The 64-head,
512-dimensional, 640-selection component benchmark reports max absolute error
1.49012e-6 versus the independent reference and 0.284 ms for the SVE kernel
(`sparse-loop-v1/attention-bench.log`).

The full `sparse-loop-v1` run completes in **184.819 s** with Engram prefetch
enabled. Positions 1000–1104 average **172.438 ms/token (5.799 tok/s)**,
with p50 172.695 ms and p95 175.904 ms. The sparse-attention span falls from
25.852 to **11.107 ms/token** versus `prefetch-v2` (57.0% less), reducing
total token latency by 7.5%. All 12 ranks finish, all 1,105 token triples and
nine saved logit arrays match the initial profile run bitwise, and minimum
final MemAvailable is 3,833,987,072 bytes. Runner SHA256:
`96e4571732ea471beb4c9c3c76319556fa5191b4165e0f0802cfdc1ad3ae837e`.

The final `sparse-no-prefetch-v1` control retains the same acknowledgment and
sparse-loop changes but omits `--engram-prefetch`. It completes in 189.008 s;
the 1K window averages **175.922 ms/token (5.684 tok/s)**, p95 181.767 ms.
All 1,105 triples and nine logit arrays match the uninstrumented baseline
bitwise; all 12 ranks finish with minimum final MemAvailable 3,863,412,736
bytes. Prefetch reduces exposed Engram I/O from 11.951 to 5.985 ms/token and
improves overall throughput by 2.0% in this sequential pair. These are single
full-run comparisons, not repeated-trial confidence intervals. The final
warning-clean source differs from the prefetch-on snapshot only by an
explicit const-array pointer cast in the runner call, plus test cleanup.
Final runner SHA256:
`58af7ef72cd965f1c215ea08eed406d2a24fa076811f1559b7a64b0ba640ced4`.

### Completed optimization checkpoint (2026-09-12)

**Historical pause:** Engram prefetch and sparse-attention loop optimization
were complete at the requested boundary, with no further runs queued then.
Job 51562789 ended at **17:57:45 JST on 2026-09-12** and its `/local` data expired.
The later INT8 request resumed work on job 51569201, described below. Follow
`a64fx/remote-dev-procedure.md` to reconnect or allocate again; do not assume
old `/local` paths survive.

| Valid run | 1K ms/token | 1K tok/s | Whole inference loop, seconds |
| --- | ---: | ---: | ---: |
| Initial fine profile | 289.802 | 3.451 | 310.127 |
| SVE BF16 rounding | 206.638 | 4.839 | 219.066 |
| Parallel pointwise operations | 187.531 | 5.332 | 198.105 |
| Engram prefetch + receive acknowledgments | 186.392 | 5.365 | 196.071 |
| Sparse loop + Engram prefetch | **172.438** | **5.799** | **184.819** |
| Sparse loop, prefetch disabled (control) | 175.922 | 5.684 | 189.008 |

The final optimized run improves throughput by **68.1%** and reduces latency
by **40.5%** relative to the initial fine profile. The **20+ tok/s target is
not reached**: 172.438 ms/token still needs to fall below 50 ms. Attention
remains the largest stage at 84.035 ms/token, including 11.107 ms of sparse
attention. Across dense projections, compressed FP8 GEMVs take 71.419 ms/token
for 6.843 GB of weights, approximately 95.8 GB/s. Other large stages are
shared experts (19.929 ms), routed experts on the slowest rank (19.751 ms),
and the two mHC mixes (17.014 ms). Within the mixes, serial normalization
costs 7.319 ms and F32 matvecs 8.323 ms. These nested timings overlap their
parent stages; they must not be added together. They identify the remaining
bottlenecks for a later session, without reopening tuning now.

Exact build and validation commands, from the repository root on the frontend:

```sh
TMPDIR="$PWD/tmp/ds41f" make -C a64fx/ds41f \
  ds41f_run ds41f_sve_test test_ops test_pointwise test_prefetch \
  bench_attention_kernel A64FX_CC=fccpx A64FX_MPICC=mpifccpx
TMPDIR="$PWD/tmp/ds41f" make -C a64fx/ds41f test \
  CFLAGS='-O2 -Wall -Wextra -Wpedantic -std=c11 -fopenmp'
```

The cross-build uses `-Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast`,
OpenMP, and pthreads for the runner/prefetch test. Versioned compute launch
scripts are `prefetch-sparse-v1.sh` and `sparse-no-prefetch-v1/run-v1.sh` under
`tmp/ds41f/job51562789/`; source snapshots are in each results directory.
The optimized
run uses the following command from its fresh shared results directory:

```sh
env XOS_MMM_L_PAGING_POLICY=demand:demand:demand \
  OMP_NUM_THREADS=48 OMP_PROC_BIND=close OMP_PLACES=cores \
  mpiexec -np 12 ./ds41f_run \
  --stage-root /local/u14346/ds41f-51562789 \
  --prompt-ids /vol0006/mdt0/data/hp250467/work/gemm/ds4f/tmp/ds41f/job51562789/prompt-capital.ids \
  --generate 1100 --ignore-eos --max-context 1048576 \
  --logits-prefix logits --logits-count 9 \
  --profile-start 16 --profile-count 1089 --engram-prefetch
```

Saved logits cover only positions 0–8, outside the profiling window.
`comparison.json`, `profile-1k.json`, and `report-1k.txt` in each final run
directory contain the correctness checks and timing summaries. Component
logs record `BF16_ROUND PASS bit_exact=426112`, `POINTWISE PASS bit_exact`,
`SPARSE_TAILS PASS reference_cases=48`, and `PREFETCH PASS ... generations=96`.
These optimization regressions establish agreement with the existing runner;
the earlier independent-reference and actual-1M-history limitations still
apply.

### Future tasks after the pause

Resume from commit `dbce50bf` and the `sparse-loop-v1` / `sparse-no-prefetch-v1`
artifacts. The target remains **single-request decode above 20 tok/s at
approximately 1K actual history**. Engram prefetch and the sparse value-loop
update are complete; the tasks below are pending, with no runs queued.

Performance work, in priority order:

1. **Reproduce the final baseline.** Check allocation/bridge health and staged
   manifests using the remote development procedure; restage if `/local` has
   expired. Run the final committed binary with prefetch enabled and the same
   prompt/options, then repeat sequential prefetch-on/off comparisons to
   distinguish the measured 2.0% gain from run variability. Keep each binary,
   source snapshot, SHA256 and log in a new results directory. Report positions
   1000–1104, including mean and p95; also measure with profiling disabled.
2. **Optimize compressed FP8 projections first.** Their 71.419 ms/token already
   exceeds the entire 50 ms target budget. Profile the actual `wq_b`, grouped
   `wo_a`, `wo_b` and shared-expert shapes in `ds41f_sve.c`, separating weight
   traffic, FP8/scale decoding and arithmetic. Inspect generated SVE code and
   measure CMG/thread placement and row blocking with realistic weight working
   sets. Test bounded conversion reuse or fused scale/decode loops. Keep FP8
   weights resident in compressed form; the rejected full BF16 expansion must
   not be restored based only on small, warm microbenchmarks.
3. **Reduce mHC mix overhead.** In `ds41f_run.c:mixes`, investigate the serial
   normalization (7.319 ms/token) and the 24-row F32 matvecs (8.323 ms/token).
   Measure vectorization, work distribution and parallel-region overhead.
   Changes to reduction order require explicit numerical comparison; preserve
   the separately rounded products in mHC post processing.
4. **Reduce expert critical-path time.** Shared experts cost 19.929 ms/token
   and the slowest routed rank costs 19.751 ms/token. Profile MXFP4 decoding,
   activation quantization, scratch reuse and the observed expert imbalance
   before changing placement. Evaluate overlap of shared and routed work only
   with an explicit core budget: concurrent OpenMP teams can compete for the
   same 48 cores. Keep owner synchronization and receive acknowledgments valid.
5. **Evaluate dense tensor parallelism if kernel tuning is insufficient.**
   Dense layers currently execute on one owner at a time. Estimate the extra
   communication and per-node memory before prototyping projection sharding
   across a small rank group. Require a measured end-to-end gain; distributing
   work must not replicate the full dense model or exceed the existing HBM
   admission budget. This is a larger architecture experiment, not an assumed
   route to 20 tok/s.

Correctness and acceptance work:

- **Complete independent-reference validation.** Rerun the corrected NumPy
  full-graph reference for all nine positions; only the corrected three-position
  check is complete. Retain the cosine >= 0.999 and matching-argmax gates.
  Extend bounded logit capture to selected positions near 1K in a separate
  correctness run; current saved arrays cover positions 0–8 only. Continue
  comparing all 1,105 token triples and component references after math changes.
- **Harden transport under longer receiver stalls.** Turn the temporary
  `comm-skew-v1` reproducer into a maintained regression and sweep delayed ranks,
  payload sizes and delays around/beyond the existing approximately 64 ms ACK
  retry budget. The shared transport can currently proceed optimistically
  after retry exhaustion. Design a bounded error/abort or proven receive-slot
  ownership scheme before claiming correctness under arbitrary scheduling skew;
  preserve the current 3 ms / 300-reduction passing case.
- **Require whole-run evidence for each retained optimization.** Keep compressed
  residency and memory guards, pass the affected component checks, and compare
  identical prompts and history windows on all 12 ranks. Re-profile the full
  runner and record MemAvailable, mean/p95 latency and numerical differences.
  Treat the 20 tok/s target as achieved only with repeated single-request
  measurements below 50 ms/token at actual 1K history.

Actual long-history/1M execution, KV checkpoint/restore and batched prefill
integration remain separate deferred tasks. Allocating the 1M cache does not
validate those paths; use the memory and acceptance sections below when that
scope is resumed.

## Resumed INT8 optimization (job 51569201)

The user resumed optimization toward **20+ tok/s for a single request around
1K actual history**, specifically requesting INT8 requantization and SDOT for
FP8 matrices. The earlier pause no longer applies. Job **51569201** has 12
nodes in normal 2000 MHz mode, from **18:00:29 JST September 12** until
**00:00:29 JST September 13**. The initial node is `d29-2208c`; its bridge uses
local port 42394, reverse port 32394 and server port 21265. Tunnel control
state is in `tmp/i8`. Port 42393/job 51562789 is expired.

Exact original weights were restaged from `/home/u14346/models/ds41f` into
`/local/u14346/ds41f-51569201/rank<R>` using bounded, page-cache-evicting copies.
Original Engram metadata was reused after checking its hash. All 12 staging
ranks finished. Results and immutable source/binary snapshots are under
`tmp/ds41f/job51569201/`. Launchers are serialized; do not start another MPI
program while one is running, and do not edit active scripts or binaries.

### Measured progress

All full runs below generate 1,100 outputs from the same six-token prompt,
allocate the 1M-token cache, and measure positions 1000–1104 (105 samples).
They do not establish execution at an actual 1M history. FP8 and INT8 may
generate different histories; fixed-history quality comparisons are separate.

| Run | ms/token near 1K | tok/s | Whole loop, s | Validation (control stated) |
| --- | ---: | ---: | ---: | --- |
| `baseline-v1` | 173.463 | 5.765 | 186.497 | 1,105 token triples + nine logits bit-exact |
| `fp8-overlap-full-v1` | 147.852 | 6.764 | 159.811 | 1,105 token triples + nine logits bit-exact |
| `int8-overlap-full-v1` | 122.272 | 8.178 | 131.791 | Experimental INT8; original malloc pack allocation |
| `int8-retained-full-v1` | 121.064 | 8.260 | 129.734 | Same INT8 arithmetic, warning cleanup |
| `int8-mmap-probe-v1` | 101.761 | 9.827 | 108.991 | Bit-exact to retained INT8 control |
| `int8-local-pages-full-v1` | 99.939 | 10.006 | 107.998 | Bit-exact to retained INT8 control |
| `fp8-local-pages-full-v1` | 147.127 | 6.797 | 158.196 | 1,105 token triples + nine logits bit-exact |
| `int8-mpi-bcast-full-v1` | 98.448 | 10.158 | 105.687 | Bit-exact to retained INT8 control |
| `int8-selection-full-v1` | 97.965 | 10.208 | 104.442 | Bit-exact to retained INT8 control |
| `fp8-selection-full-v1` | 144.516 | 6.920 | 154.926 | 1,105 token triples + nine logits bit-exact |

The exact FP8 improvement combines the contiguous-head RoPE loop, SVE FP64
mHC normalization, Engram scale caching and shared/routed expert overlap.
Minimum final MemAvailable is **4,195,811,328 bytes**, with all 12 ranks
finished. Its binary SHA256 is
`3d7db1d2d484d82dceef2a7f79bde0df9559367e053676e199c06b2c0d93467a`.
The norm-only nine-position control `fp8-norm9-v1` also matched bit-exactly.
The 20+ tok/s target is **not reached**.

The reproduced baseline profile attributes 72.243 ms/token to dense FP8
GEMVs, reading 6.843 GB/token (95 GB/s effective). Attention takes 85.298 ms,
shared experts 20.296 ms, routed experts on the slowest rank 19.611 ms, and
mHC mixes 17.285 ms. After the exact changes, attention is 82.893 ms and the
combined routed/shared stage is 26.577 ms. The norm drops from 7.430 to about
0.31 ms/token; the original mHC matrix-reduction order is retained.
Nested timings overlap their parent stages.

### INT8 format, conversion cost and validation

`--fp8-int8-block 32|64|128|256` enables a row/block-scaled signed INT8 SDOT
GEMV path. Default zero preserves FP8. The converter applies the original
FP8 E4M3 and E8M0 scales, finds each row/block maximum, rounds symmetrically
to [-127,127], and packs groups of rows for paired SVE SDOT accumulators.
Each row/block has one FP32 scale. Activations retain the existing FP8
quantization first, then use matching blockwise INT8 quantization; output
BF16 rounding stays in place. Grouped `wo_a` is supported. The large grouped
activation buffers are quantized in parallel. Batched INT8 GEMM remains
unimplemented; this continuation targets batch-one decode GEMV.

Conversion occurs **once after reading the original files from `/local`**,
releasing each original FP8 matrix after its replacement is ready. Block 32
adds 12.5% scale storage plus row padding, while admission accounts for both
representations of the current tensor when source mappings can be released.
With pooled source allocations, admission conservatively reserves the original
store plus all packed replacements because the pool can retain freed pages. Full-model conversion measured
**0.2727–0.4608 seconds per rank**, versus **52.688–57.028 seconds** of resident
startup. The later fresh-allocation run took at most 0.4914 s/rank.
This remains below 1% of startup, satisfying the user's inexpensive
online-requantization condition. No offline converted files were written;
the original shared safetensors and staged weight files remain authoritative.

The first real-matrix probes (`int8-components-v1`) measured query projection
32768x1280 at 0.444 -> 0.104 ms (4.27x) and grouped output 8192x4096 at
0.336 -> 0.157 ms (2.14x), block 32. Parallel activation quantization later
reduced the grouped probe to 0.115 ms (`overlap-runs-v1/components.log`). These
are reused-tensor component timings, not a full-run speed claim. Cold-weight
probes (`int8-tile-v1`) selected the original four-row format: four/eight/sixteen
rows take 0.0998/0.1083/0.1065 ms for `wq_b`, 0.1219/0.1203/0.1247 ms for
`wo_a`, and 0.1408/0.1517/0.1309 ms for `wo_b`. The mixed larger-tile gains
were insufficient to replace the four-row format; both prototypes passed the
162 arithmetic cases. No larger tile is retained.

Numerical validation distinguishes kernel arithmetic from quantization loss:

- `test_int8` passes **162** reference cases covering blocks 32/64/128/256,
  padded rows, grouped inputs, the parallel input-quantization branch,
  canaries, zeros, nonfinite rejection and subnormal scales. SVE SDOT is
  checked against integer dots with double rescaling; native warning-clean
  builds also pass. AddressSanitizer was unavailable on this frontend.
- `test_int8_attention` replays 54 real attention inputs (nine positions in
  layers 0/1/2/8/14/20). Block 32 passes its cosine >= 0.999 exit gate, but
  relative RMS reaches 4.23%; this is **not** a pass of the stricter 1% gate.
  Block 128 fails 11 of 54 cosine checks. Limiting quantization to query/output
  projections improves this local replay, but not full-model agreement.
- Full fixed-token replay `int8-all32-v1` matches **9/9 argmax choices**, but
  minimum logit cosine is **0.994185** and maximum relative RMS **10.77%**.
  The projection-only control reaches minimum cosine **0.952806** and maximum
  relative RMS **31.05%**, also with 9/9 argmax choices. Both fail the existing
  numerical gates and remain experimental; token-choice agreement alone does
  not establish acceptable model quality.
- Fixed-history replay at **positions 1000–1008** is complete in
  `fp8-replay1k-v1` and `int8-replay1k-v1`. All 1,105 input tokens are identical;
  INT8 next-token choices match **1,065/1,105 overall** and **104/105 at
  positions 1000–1104**. The nine saved argmax choices agree, but logit cosine
  falls to **0.902769873** and relative RMS reaches **0.439981917**. All nine
  fail the numerical gates. `comparison-1k.json` and `token-agreement.json`
  retain the evidence. These dump-enabled replays are quality runs, not speed
  measurements. `compare_run_logits.py` rejects missing/divergent histories.
- A second INT8 residual plane lowers individual matvec relative error to
  about 5e-5, but the full replay still reaches cosine 0.994099 / relative RMS
  12.82%, and costs more time/memory. This experiment is **not retained**.
  Its sources and evidence are in `refined-runs-v1` and
  `int8-refined32-quality-v1`.

`--fp8-int8-scope projections` selects only `attn.wq_b`, `attn.wo_a`,
`attn.wo_b` and shared experts; default scope `all` converts every FP8 matrix.
INT8 remains explicitly opt-in and must not be described as an accepted
numerical replacement for the FP8 path.

### Resident memory placement

The large gap between component and full-run SDOT timings was traced to
allocation reuse. In `resident-int8-probe-v1`, rank-zero `wo_a` and `wo_b` took
about 0.44/0.45 ms even on repeated real inputs, while `wq_b` reached 0.0985 ms.
The packed `wo_a`/`wo_b` pointers landed at the end of the malloc arena, and
`int8-retained-full-v1/numa-rank0.txt` shows their full mappings on NUMA node 7.
The original weight scales were normal (E8M0 codes 114–121), excluding a
subnormal arithmetic explanation for these tensors.

Fresh anonymous allocations bypass the Fugaku `libmpg` malloc pool and let
parallel conversion place new pages. Changing only the packed INT8 allocations
improved **8.260 -> 9.827 tok/s**, with identical 1,105 token triples and nine
logit arrays. `ds41f_alloc.h` uses Linux LP64 raw anonymous mmap/munmap, with
posix_memalign/free fallback elsewhere. The weight files are still read into
resident anonymous buffers using bounded pread plus fadvise; this is not
file-backed model mmap.

`--weights-local-pages` applies fresh allocation to the original tensors too.
Together with reusing a single 640x512 attention row workspace, it reached
**10.006 tok/s** and minimum final MemAvailable **3,993,108,480 bytes**.
Rank-zero MemAvailable stayed close to its post-load value (4.059 -> 3.993 GB).
The combined comparison does not isolate the workspace from source allocation.
The FP8 control remained bit-exact and reached **6.797 tok/s**. The loader
fixture passes six cases covering ordinary/fresh allocation, page boundaries,
size rejection and independence from subsequent source-file changes.
A later `int8-local-pages-full-v1/numa-rank0.txt` capture occurred during
teardown and must not be used as proof of full-resident placement.

The native MPI broadcast prototype then reached **10.158 tok/s**. Residual
broadcast time fell from about 5.6 to 3.5 ms/token; other synchronization spans
changed with rank skew. INT8 token triples and nine saved logits stayed exact.
It is retained as `--mpi-broadcast`; uTofu still performs reductions. Native
broadcast normalizes signed zero as the previous sum-based broadcast did.
The component test checks both modes, 12 owners, seven lengths including a
32768-float chunk boundary, canaries and delayed receivers.

At this point the critical path is attention **45.511 ms/token**, combined
experts **20.909 ms**, mHC mixes **10.233 ms**, gate **4.024 ms** and residual/FFN
broadcasts **6.076 ms**. Attention includes sparse attention **11.600 ms**,
indexing **6.289 ms**, and query/output INT8 projections **15.362 ms**.
INT8 kernels read 7.691 GB/token at an aggregate effective **358 GB/s**.
These nested measurements overlap parent stages. Simply improving INT8 GEMV
cannot remove the remaining roughly 48 ms needed for the 50 ms/token target.

Vocabulary argmax now uses SVE finite/max scans followed by the first matching
index. Its measured cost falls from **1.230 to 0.034 ms/token**. Gate score
calculation uses parallel independent scalar math, retaining the existing
selection and normalization order; total gate time falls from **4.024 to
3.498 ms/token**. The combined INT8 run reaches **10.208 tok/s**, with all
1,105 token triples and nine saved logit arrays unchanged. `test_selection`
passes 130 argmax cases and 20 bit-exact gate cases, including ties, tails,
nonfinite rejection and the parallel threshold, on native and SVE builds.
Its binary SHA256 is
`45aa28c1b0e27d720f647f8a68b7206c1033a06ff506dcbbe85886dfe54eff87`.
The same binary's FP8 run reaches **6.920 tok/s** (144.516 ms/token),
20.0% above the same-allocation baseline. All 1,105 token triples and nine
saved logits remain bit-exact to original FP8; all 12 ranks finish with minimum
final MemAvailable **4,044,226,560 bytes**. INT8 minimum final MemAvailable
is **3,968,729,088 bytes**. These runs use profiling; final unprofiled repeats
are recorded separately.

With profiling and logit dumps disabled, `int8-final-unprofiled-v1` measures
**96.494 ms/token / 10.363 tok/s**, p95 **98.467 ms**, over the same 105 positions.
The whole loop takes **102.940 s**. All 1,105 token triples match the profiled
INT8 run, all 12 ranks finish, and minimum final MemAvailable is
**3,997,958,144 bytes**. Per-token timings are the runner's rank-zero wall clock;
p95 uses linear interpolation. The `summary.json` records each unprofiled run.
`int8-final-unprofiled-v2` confirms **96.279 ms/token / 10.386 tok/s**,
p95 **98.463 ms**, whole loop **102.578 s**, and minimum final MemAvailable
**3,999,596,544 bytes**. It also matches all 1,105 token triples. Thus repeated
uninstrumented INT8 throughput is **10.36–10.39 tok/s**, still below 20 tok/s.
Both binaries have the same SHA256 given above.

The matched `int8-source-pool-control-v1` omits only `--weights-local-pages`
while retaining fresh INT8 allocations and the reused row workspace. It reaches
**98.015 ms/token / 10.203 tok/s**, p95 **99.889 ms**, whole loop **104.492 s**,
and minimum final MemAvailable **3,576,233,984 bytes**. All 1,105 token triples
match. Fresh original-weight allocations therefore add roughly 1.7% throughput
and about 0.42 GB final headroom in these runs; the explicit flag remains in the
benchmark configuration, while its default stays off.

All recorded launchers have finished successfully, with no further inference
runs queued. Job **51569201** remains available until the end time above; its
original staged `/local` weights can be reused while that allocation lives.
The 20+ tok/s target and full INT8 numerical acceptance remain **open**.

### Other changes and rejected probes

`--engram-scale-cache` reads each rank's two raw scale shards (about 512 MB)
into HBM with bounded reads and page-cache eviction. It removes one of the two
reads per Engram row and retains the 2 GiB admission floor. The prefetch fixture
checks budget rejection, bit-exact cached/uncached rows, closing the scale FDs
before cached reads, short-read propagation, pending close and profiler TLS.

`--hc-mix-sve` uses an explicitly vectorized FP64 sum of squares. It passes
352 bit-exact norm cases with tails. Splitting the 24-row mHC matrix across
48 K-partitions halved its component time but caused full-model logit drift
(minimum cosine 0.989646 in `fp8-extras9-v1`); that split was removed.
The RoPE head/angle loop swap passes 360 bit-exact layout/tail/position cases
and roughly halves the RoPE component time.

`--shared-overlap` computes the owner's shared expert before the routed
reduction using the same OpenMP team, while other ranks finish their routed
experts. It adds the shared output after reduction in the original order.
The profiler reconstructs `EXPERTS_AND_SHARED` from each rank's combined
local work, avoiding double-counting the overlap.

Other rejected temporary probes: power-of-two INT8 scales (worse attention
agreement), INT8 SDOT MXFP4 decoding (slower, especially down projection),
four-row floating MXFP4 interleaving (slower), and FEXPA sparse softmax
(only a small gain with extra approximation). None is enabled in the runner.

Retained component checks are archived in `retained-checks-v1/components.log`,
`local-pages-runs-v1/components.log` and `selection-runs-v1/components.log`
(with MPI stdout under that run's `output.51569201/`). Representative output:

```text
DS41F_KERNEL_TEST PASS
SPARSE_TAILS PASS reference_cases=48 masked duplicate_ids empty canaries
ROPE_LAYOUT PASS bit_exact=360 canaries inverse long_positions
MHC_MIX PASS norm_bit_exact=352 tails
INT8 PASS cases=162 blocks=32,64,128,256 padded_rows grouped canaries zero nonfinite subnormal_scales
PREFETCH PASS bit_exact generations=96 remote_zeros short_read_error pending_close profiler_TLS scale_cache budget cache_only_reads
TENSOR_LOCAL PASS cases=6 malloc fresh_pages boundary_sizes source_independence size_rejection
SELECTION PASS argmax=130 gate=20 first_ties nonfinite_rejection bit_exact_weights
BROADCAST PASS modes=2 owners=12 sizes=7 chunk_boundary canary signed_zero delayed_receivers
```

### Reproduction and future tasks

Build on the frontend (no `/tmp`):

```sh
mkdir -p tmp/ds41f
TMPDIR="$PWD/tmp/ds41f" make -C a64fx/ds41f ds41f_run test_int8 \
  test_int8_attention test_tensor_local test_selection test_broadcast \
  test_rope_layout test_mhc_mix test_prefetch \
  A64FX_CC=fccpx A64FX_MPICC=mpifccpx \
  A64FX_CFLAGS='-Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast -Wall -Wextra -Wpedantic'
```

Snapshot the binary into a new shared results directory and verify its SHA256
on the compute node. Inside job 51569201, with no other MPI program running:

```sh
export TMPDIR=/local/u14346/ds41f-51569201
export XOS_MMM_L_PAGING_POLICY=demand:demand:demand
export OMP_NUM_THREADS=48 OMP_PROC_BIND=close OMP_PLACES=cores
mpiexec -np 12 ./ds41f_run --stage-root /local/u14346/ds41f-51569201 \
  --prompt-ids /absolute/repo/tmp/ds41f/job51562789/prompt-capital.ids \
  --generate 1100 --ignore-eos --max-context 1048576 \
  --engram-prefetch --engram-scale-cache --hc-mix-sve --shared-overlap \
  --weights-local-pages --mpi-broadcast --fp8-int8-block 32
```

Omit `--fp8-int8-block 32` for FP8. For an instrumented comparison add
`--profile-start 16 --profile-count 1089 --logits-prefix logits --logits-count 9`,
then run `profile_report.py RESULTS --start 1000 --stop 1105` on the frontend.
For near-1K numerical comparison use the same fixed 1,105-token input file in
both runs and `--generate 1 --logits-start 1000 --logits-count 9`.

Remaining tasks, in priority order:

1. Resolve accumulated INT8 quality loss. Local cosine/argmax agreement is
   insufficient; retain the full fixed-history cosine >= 0.999 / relative RMS
   <= 1% gates. Isolate activation versus weight quantization and sensitive
   layers with bounded same-input captures. The second-plane and projection-only
   experiments already failed to resolve the full-model error.
2. Reduce the remaining attention and expert critical paths. Evaluate dense
   tensor parallelism with a concrete staging, communication and HBM budget;
   most ranks currently wait while one owner runs attention. No tensor-parallel
   implementation is present. Optimize routed MXFP4 using actual cold weights
   and realistic expert placement before revisiting rejected SDOT probes.
3. Improve mHC matrix execution while controlling accumulation error. Generated
   code uses ordered SVE `fadda` within each 20,480-column row; changing the
   reduction order already caused full-model drift. Benchmark any new layout
   against both component arithmetic and the full fixed-history replay.
4. Repeat actual-1K unprofiled performance measurements after every retained
   structural change; do not claim 20+ tok/s until repeated runs exceed it.
   Repeat the matched source-allocation control on another allocation before
   changing the current opt-in default.
5. Complete the corrected nine-position independent NumPy reference and retain
   transport skew/ACK regression checks. Actual-1M execution, KV persistence,
   batched prefill and INT8 GEMM remain separate unvalidated future work.


## Checkpoint accounting and active staging (job 51562789)

Header inventory supersedes the rough per-node fit estimate below:

| Category | Exact bytes |
| --- | ---: |
| 40-layer routed experts | 288,777,830,400 |
| Engram tables including scales | 202,758,032,400 |
| Dense text weights | 9,846,748,608 |
| Excluded auxiliary/vision/MTP | 8,903,411,592 |

Expert shards plus fully replicated dense weights consume 31.58 GiB/node,
before runtime state. Pure EP with replicated dense weights therefore does
not meet the memory budget. Initial inference must distribute dense layer
ownership as well as routed experts. Keep `expert_id % 12` ownership; assign
dense layers round-robin (`layer % 12`), with embeddings/head separately
accounted. Broadcast owner-produced activations for EP and combine routed
outputs using uTofu. The runner now shares packed KV/index source rows and
selection state using that transport.

`stage_backbone.py` stages source bytes unchanged to
`/local/$USER/ds41f-51562789/rank<R>` on the corresponding node. Engram rows
use contiguous ceil-div ownership. Dense tensors are staged once on rank 0
as a canonical disk source before the now-completed redistribution; this is not an instruction
to load all dense tensors into rank 0 HBM. Disk use is 50.808 GB on rank 0
and about 40.961 GB elsewhere. Total transferred payload is 501.383 GB,
roughly 28 minutes at an aggregate 300 MB/s excluding metadata/fsync overhead.

The stager uses 8 MiB bounded buffers, fsync and page-cache eviction, SHA256
sidecars and per-rank manifests. It skips completed size/stamp pairs on
restart; it does not rehash existing files, so a separate checksum audit is
required before production admission. Shared progress logs are under
`tmp/ds41f/job51562789/staging/`. Node-local files expire with the job.

MXFP4 checkpoint packing was verified against the local official
`inference/convert.py`: low/high nibbles represent adjacent columns, E2M1
has maximum 6, and scales are one E8M0 byte per row per 32 columns. The
legacy GGML split-half packing cannot be used on these source bytes directly.

## Recommended 12-node layout

Keep the FP8/FP4 model backbone resident in HBM2 and keep Engram on each
node's private `/local` filesystem, accessed through the uTofu owner path.
Keep the active 1M-token KV cache resident in HBM2. Use `/local` as a
persistent checkpoint and cold-session backing store, not as the decode-time
KV store.

```text
HBM2 on each node
  FP8/FP4 backbone shard       ~24-25 GiB
  active batch-1 KV cache      ~0.9 GiB at 1M context
  dequant/GEMM/attention work  remaining headroom

/local on each node
  Engram owner shard            ~16 GiB
  persistent KV checkpoint     optional
  inactive-session spill       optional
```

The current checkpoint is about 510.3 GB (475.2 GiB). Removing the Engram
tables leaves approximately 307-322 GB of backbone weights, depending on
whether Engram scales and metadata are included in the accounting. That is
about 24-25 GiB per node across 12 ranks. This fits 32 GiB HBM2 only if the
FP8/FP4 storage representation is retained; expanding the weights to BF16
does not fit.

## 1M-token KV cache

For batch 1 and the released 1M-token limit, the approximate cache budget is:

| Component | Approximate size |
| --- | ---: |
| Compressed FP4 KV latents | 0.63 GiB |
| FP8 compression scales | 0.08 GiB |
| Index KV and scales | 0.17 GiB |
| 40-layer 128-token sliding windows | negligible |
| **Total** | **~0.9 GiB** |

The model's compressed KV sources are layers 2, 8, 14, and 20. The other
layers reuse those caches. The reference implementation stores compressed
KV at `max_seq_len / compress_ratio` and quantizes it with FP4 plus scales:

<https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/inference/model.py>

The cache should be rank-local or replicated according to the attention
implementation, with no per-token disk round trip. Reserve at least 2 GiB
per node for KV and cache-management overhead even though the raw estimate is
under 1 GiB.

## Persistence and `/local` streaming policy

Use a two-tier KV manager:

1. Decode from the HBM2-resident active cache.
2. Append completed cache blocks to a per-session checkpoint on `/local`.
3. On restart or session migration, prefetch the checkpoint sequentially into
   HBM2 before resuming decode.
4. Spill inactive sessions from HBM2 to `/local`; restore them as whole blocks
   when they become active.

Do not fetch individual KV rows from `/local` for every decode token. A query
may select hundreds of compressed positions, turning random local-storage or
uTofu reads into the decode bottleneck. The measured Engram path is roughly
20.9k row lookups/s at QD1 across 12 nodes, which corresponds to only about
40 tokens/s if 512 KV positions had to be fetched remotely for every token.

If HBM pressure eventually requires KV paging, page aligned blocks rather than
rows. Recommended starting points are 16-64 KiB blocks, double-buffered with
an asynchronous prefetch queue, with LRU admission for inactive sessions.
Measure block hit rate and restore bandwidth before enabling decode-time KV
spill.

## Implementation priorities

- Keep backbone weights compressed in HBM2 and dequantize into bounded scratch
  buffers.
- Keep the active batch-1 1M-token KV cache entirely in HBM2.
- Stage Engram shards to `/local` and use uTofu for owner communication.
- Add checkpoint metadata containing model revision, tokenizer/hash revision,
  rank topology, KV format, context length, and checksum.
- Checkpoint KV asynchronously at block boundaries; never pause decode for a
  full-cache synchronous write.
- Guard memory with a per-rank HBM budget and reject new sessions before
  evicting the active session's KV cache.

## Acceptance measurements

Before using persistent KV in production, measure:

- HBM resident bytes per rank at 1M context;
- prefill time and checkpoint write bandwidth;
- restore time from `/local`;
- active-cache HBM hit rate;
- decode tokens/s with zero spill;
- decode tokens/s with cold-session spill;
- p50/p95/p99 latency for KV block prefetch;
- correctness after checkpoint/restore at fixed-token replay.
