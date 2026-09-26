# Qwen3.8-27B A64FX prefill pipeline

The MPI pipeline assigns contiguous mixer/FFN units to 2–128 nodes (one rank
per mixer/FFN unit, the 128-unit maximum). Each
process uses 48 A64FX workers and keeps the full FP4 decode image resident;
only its assigned prefill matrix descriptors are built. A stage sends the
FP32 residual for each prompt chunk to the next stage. Successive prompts
can occupy different stages at the same time. The benchmark repeats the same
tokenized prompt with fresh KV and SSM state for each prompt.

The stage cuts minimize the largest stage cost with dynamic programming.
Default unit costs are calibrated from an 8-prompt, 1024-token, 12-node run;
`Q38P_PP_ATTN_COST` supplies a long-context attention cost. A mixer and its FFN can
land on different nodes. The 3-node run on job 51917132 exercised cuts at
units 43 and 85 and produced the same final residual hash as the single-node
build.

## Build and run on Fugaku

Stage the FP4 image at `/local/q38/fp4.image` on every allocated node and
make `/local/q38/tmp` on the first node. The 4-node stage hook used for job
51917132 is `tmp/q38-fast/stage_hook_4n.sh`. From the repository root on the
first compute node:

```sh
bash a64fx/llm/q38p/build_pp.sh
bash a64fx/llm/q38p/run_pp.sh 4 8 1024 160 1
```

The runner arguments are `NODES PROMPTS TOKENS CHUNK DECODE`. `DECODE=1`
gathers the final prompt's KV cache and SSM/conv state to rank 0, then runs
256 decode steps there and compares them with `/local/q38/ref-f32.log`.
For another prompt length, set `Q38P_REF` to its F32 reference log. Rank logs
are written under `tmp/q38p/pp_runs/`.

The standard scale sweep uses 8 independent prompts of 1024 tokens and chunk
160. The per-node target is 150 tok/s, giving targets of 900, 1200, and 1800
tok/s for 6, 8, and 12 nodes, and 3600 tok/s for 24 nodes. From an allocation
of at least 12 nodes, run the 6/8/12 configurations with:

```sh
bash a64fx/llm/q38p/run_pp.sh 6 8 1024 160 1
bash a64fx/llm/q38p/run_pp.sh 8 8 1024 160 1
bash a64fx/llm/q38p/run_pp.sh 12 8 1024 160 1
```

Ranks above 12 use the same command, up to 128 ranks. For example, a
24-node allocation runs `bash a64fx/llm/q38p/run_pp.sh 24 8 1024 160 1`.
An allocation must provide at least as many nodes as the first argument.
The partitioner keeps at least one mixer/FFN unit per rank and balances the
estimated MAC cost; throughput above 12 nodes still needs measurement.

Submit the bundled scale jobs from the Fugaku frontend:

```sh
pjsub --no-check-directory a64fx/llm/q38p/pjsub_pp12_v2.sh  # 6, 8, 12
pjsub --no-check-directory a64fx/llm/q38p/pjsub_pp24.sh      # 24
```

The MPI build uses `-O2 -mcpu=a64fx` without `-ffp-contract=fast` to retain
the established decode correctness. An `-O3 -ffp-contract=fast` build on job
51917132 diverged from F32 at generated token 5 even on one node; the
baseline-precision MPI build matched 256/256 on one through four nodes.
Its single-node 1024-token prefill rate was 155.2 tok/s at chunk 480.
An `int16_t` activation panel packing alias violation has since been fixed
with `memcpy`. With that fix, a single-node `-O3` build without fast math
passes 256/256 decode but remains near 154 tok/s; the earlier apparent
162 tok/s `-O3` result came from an incorrect panel. Keep the MPI build at
its validated `-O2` setting until the pipeline is retested with the fix.

## Measurements

FP4 model, 1024-token prompt, eight repeated independent prompts. `steady` is
the seven inter-completion intervals at the final stage. `end-to-end` includes
pipeline fill and drain; `latency` is the first prompt's completion time.
State transfer and decode follow those timings when `DECODE=1`.

| Nodes | Chunk | Steady tok/s | End-to-end tok/s | Latency s | Decode vs F32 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 160 | 275.1 | 272.5 | 4.01 | 256/256 |
| 3 | 160 | 411.3 | 400.3 | 3.04 | 256/256 |
| 4 | 480 | 482.7 | 442.7 | 3.66 | 256/256 at chunk 480, one prompt |
| 4 | 320 | 523.4 | 489.2 | 3.05 | residual hash stable |
| 4 | 240 | 541.5 | 511.8 | 2.77 | residual hash stable |
| 4 | 200 | 526.8 | 502.4 | 2.70 | residual hash stable |
| 4 | 160 | **544.9** | **521.7** | **2.55** | **256/256** |
| 4 | 120 | 535.8 | 515.8 | 2.50 | residual hash stable |

Six-, eight-, and twelve-node results from interactive job 51918651, using
chunk 160 and the calibrated partition, were:

| Nodes | Steady tok/s | End-to-end tok/s | Steady tok/s/node | First prompt s | Decode vs F32 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 6 | 802.7 | 743.0 | 133.8 | 2.10 | 256/256 |
| 8 | 1074.8 | 964.2 | 134.3 | 1.83 | 256/256 |
| 12 | 1562.8 | 1316.6 | 130.2 | 1.64 | 256/256 |

The target is 150 tok/s/node (900/1200/1800 steady tok/s at 6/8/12 nodes),
so these measurements are still below target. The initial 12-node partition
measured 1555.1 steady tok/s; fitting relative FFN, SSM-mixer, and attention-
mixer costs to per-rank compute times adjusted the cuts and raised the repeat
to 1562.8 tok/s. This 0.5% gain is small and does not explain the remaining
gap.

The 12-node chunk sweep (original partition, decode disabled) measured steady
rates of 1517.8, 1555.1, 1506.5, 1547.6, 1503.5, and 1388.3 tok/s for chunks
120, 160, 200, 240, 320, and 480. Chunk 160 remains the best setting. A
four-node smoke run after raising the rank limit measured 543.3 steady
(135.8 tok/s/node), 520.1 end-to-end, and matched 256/256 decode tokens.

On four nodes at chunk 480, each stage spent about 12.3 s computing eight
prompts; upstream stages also spent about 4.7 s in blocking sends. Smaller
chunks reduce this pipeline wait but increase GEMM work. The best measured
chunk was 160. MPI nonblocking sends without a progress thread serialized the
single-prompt path and were removed.

The pipeline currently replicates the full model on each node. The original
runner gathers state to rank 0 for single-node decode; the handoff runner below
exports a canonical snapshot for TP2/TP4 decode. Stage-sharded weight images,
uTofu handoff, and concurrent TP4 decode groups remain future work.
Throughput above 12 nodes remains unmeasured.

For long FP4 prompts, `Q38P_PREEXPAND_F4_MIB=2048` stores selected stage-owned
F4 matrices as exact INT16 GEMM panels in HBM, removing repeated expansion
on every prompt chunk. The budget is per rank, defaults to zero, and is capped
at 2048 MiB because larger unrestricted sets triggered HBM NUMA out-of-memory
kills in 12-node experiments. `Q38P_PP_ATTN_COST=1500` selects measured
32K-context stage cuts. These settings are specific to the 27B NVFP4 image
and 32K input; profile other models and lengths before adopting them.

## Experimental 12-node prefill → TP4/TP2 decode

`run_handoff.sh` uses the existing **PP12 pipeline** for prefill, then imports
its state into a separate TP4 or TP2 decode process group. PP12 is pipeline
parallelism, not Q38D's replicated-mixer TP12 path. The latter performs
sequential prefill and has very different throughput.

From an existing interactive 12-node allocation, with the FP4 image staged on
all nodes and `tofu_topo.txt` generated for this allocation:

```sh
export TMPDIR=/local/q38/tmp
bash a64fx/llm/q38p/build_pp.sh
make -C a64fx/llm q38d CC=fcc Q38D_TP=1
HANDOFF_TPS='1 4 2' bash a64fx/llm/q38p/run_handoff.sh tmp/handoff-small 65 256 160
python3 a64fx/llm/q38p/check_handoff.py tmp/handoff-small --require-exact
# Fixed 32572-token corpus; replay its last 64 tokens on each decoder:
HANDOFF_PREFIX_TOKENS=32508 HANDOFF_TPS='4 2' \
  bash a64fx/llm/q38p/run_handoff.sh tmp/handoff-code32k 0 8192 160 \
  tmp/q38p/review-c-source.txt
```

Use a new run directory each time. `--prompt-file` selects the first
`--prompt-tokens` tokens without repeating them; shorter files are rejected.
Use `--prompt-tokens 0` (runner argument `0`) to consume the complete file,
including its trailing chat framing.
The existing `--prompt` synthetic benchmark retains its repetition behavior.
Do not interpret truncated code-corpus runs as an assessment of review quality.

The versioned snapshot carries the final residual, first output token,
global-head SSM state (including its deferred update), convolution rings, and
chronological FP32 KV caches. Consumers load only their assigned heads and
redistribute the KV positions across their CMGs. Image metadata (including
weight payload checksums), prompt token hash, format, and arithmetic must
match. Per-record checksums reject damaged state. Export requires an empty,
new snapshot directory; metadata is published after all layers finish.

Timers are independent:

- `q38p_prefill`: prefill end-to-end rate, aggregate and divided by 12;
  excludes state export, model load and descriptor initialization.
- `q38d_state: producer_nodes`: shared-filesystem export wall time.
- `q38d_state: TP...`: import/reshard time and readiness including the output
  head collective and optional replay; use the maximum rank readiness for the
  group. The final implementation synchronizes ranks before import timing.
- `replay_tokens` / `replay`: known prompt suffix evaluation after importing
  `--state-prefix-tokens N`; excluded from generation throughput.
- `q38d: ... decode` / `RESULT`: generation-only rate; includes growing context,
  excludes import and model load. Runs of 1024+ outputs also report each block.

This first implementation reloads decoder weights between processes. Phase
rates are valid, but their sum is a **warm-service estimate**, not measured
interactive end-to-end latency. Shared-filesystem snapshots and cold process
startup remain overheads; persistent loaded engines and direct network state
transfer are future work. Multiple simultaneous decode groups are not enabled.
The 150 prefill tok/s/node target remains a target, not a claimed result.

Portable layout/integrity test (no model or A64FX required):

```sh
mkdir -p tmp/state-test
cc -O2 -Wall -Wextra -Wno-unused-function -Wno-unused-variable \
  a64fx/llm/q38d/test_q38d_state.c -o tmp/state-test/test_state
./tmp/state-test/test_state tmp/state-test/snapshot
```

### September 26 measured results

For the fixed 32,572-token source-review prompt and 8,192 outputs, PP12
processes a 32,508-token prefix at **590.459 tok/s (49.205/node)**. Replaying
64 prompt tokens takes 1.013 s on TP4 / 1.685 s on TP2. Generation-only
throughput is **57.958 tok/s TP4** and **34.831 tok/s TP2**, up from 50.059 /
29.356 before PV cache blocking. Both match all 8,192 baseline TP2 IDs;
this is not a full-generation F32 reference check.

The complete 32,572-token prefill improves from 376.812 to 588.820 tok/s;
all 2,434 snapshot headers/record checksums agree. Selected defaults are SVE
maximum reduction, CMG-local query scheduling and PV block 256. Controls
`Q38D_SCORE_MAX=0`, `Q38P_ATTN_CMG=0`, `Q38D_ATT_PV_BLOCK=0` restore the
respective old paths. Eight independent 1024-token prompts reach 1577.164
steady tok/s (131.430/node), with 256/256 gathered-decode F32 agreement.
Neither prefill rate meets the 150 tok/s/node target.

Synthetic TP4 handoff without replay diverges at output token 5; 64-token
replay restores 256/256 F32 agreement. Use replay explicitly and validate new
prompts. See `resume-prefill.md` for exact logs, timings and remaining work.

## Prefill at a specified context depth

`run_depth.sh` measures new suffix tokens at a fixed context depth. Its default
`tokens` fill evaluates a deterministic random-token prefix **once**, outside
the measured interval. It retains prefix KV and checkpoints each stage's SSM
and convolution state; each repetition restores that state and overwrites only
the suffix KV. This avoids reevaluating the full prefix for every sample.

```sh
# Existing 12-node allocation; build and stage the same FP4 image first.
bash a64fx/llm/q38p/run_depth.sh tmp/q38p/depth/NEW_RUN \
  32768 1920 6 tokens 480 12
python3 a64fx/llm/q38p/summarize_depth.py tmp/q38p/depth/NEW_RUN
# Faster screening: populate prefix KV directly, leave recurrent state zero.
bash a64fx/llm/q38p/run_depth.sh tmp/q38p/depth/NEW_FAST_RUN \
  32768 1920 6 synthetic-kv 480 12
```

Arguments are `NEW_RUN_DIR DEPTH SUFFIX REPEATS FILL CHUNK NODES`.
The engine exposes `--bench-depth`, `--bench-fill tokens|synthetic-kv`, and
`--bench-seed`; `--prompt-tokens` is the measured suffix length in this mode.
Random token IDs are deterministic for a seed. Prefix setup, model loading,
and prefill descriptor creation are excluded. End-to-end throughput counts
`REPEATS * SUFFIX`; sustained throughput counts the last `REPEATS - 1`
suffixes after the first finishes, excluding pipeline fill. Neither numerator
includes the prefix. Single-suffix latency includes pipeline traversal and the
output head. All repetitions must produce the same final residual/token.

`synthetic-kv` is a performance screening mode, **not evaluated token history
or a correctness reference**. It fills deterministic finite K/V values directly;
use `tokens` and the fixed coding corpus for confirmation. Neither mode permits
decode or snapshot export. Total depth plus suffix is capped at 49152 tokens
for this full-image, 32 GB HBM prototype.

Attention experiments: `Q38P_ATTN_QTILE=1..16` sets adjacent queries per task;
`Q38P_ATTN_KTILE=16..1024` (multiple of 16) sets the shared KV tile when query
tiling is enabled. The selected defaults are eight queries and 32 KV
positions at context >=4096; shorter contexts use one query per task. Query
tile 1 retains the original task size. `Q38P_ATTN_QK6=0` disables the new
bit-exact six-head QK assembly kernel. Decode retains its existing QK path.
The wider PV candidates were slower and have been removed.
`Q38P_PP_ATTN_COST` overrides the attention-mixer weight used by the contiguous
pipeline stage partitioner; its original short-context value is 326.6. Tune
against per-rank compute/wait times, and verify both residual hashes and
post-prefill generation after selecting a configuration.

Attention kernel regression test on the allocated A64FX node:

```sh
export TMPDIR=/local/q38/tmp
fcc -Nclang -O2 -march=armv8.2-a+sve \
  a64fx/llm/q38d/test_q38d_attention.c a64fx/llm/q38p/q38p_attention.S \
  -lm -o /local/q38/test_attention
/local/q38/test_attention
```

This checks QK against explicit lane-wise FMA and pairwise reduction, plus
PV offsets, tails, score/output strides and repeated cache-block accumulation.

### Long-context results (12 A64FX nodes, job 51930941)

| Workload | Total tok/s | tok/s/node |
| --- | ---: | ---: |
| Full fixed 32,572-token coding prompt, chunk 480 | 1027.275 | 85.606 |
| Sustained 1920-token suffix at depth 32768, evaluated random prefix, six repeats | 821.824 | 68.485 |
| Same depth/suffix with synthetic KV screening | 814.064 | 67.839 |

Full coding-prompt prefill takes 31.707200 s, versus 55.317369 s before query
tiling and the six-head QK kernel (+74.5% throughput). All 2,434 snapshot
headers/record checksums match. Export takes a separate 6.902624 s.
The evaluated-prefix benchmark spends 32.034601 s on prefix setup once;
its six suffixes take 19.064536 s including pipeline fill/drain. Sustained
throughput excludes the first suffix's 7.383 s pipeline traversal. Peak RSS
is about 20.88 GiB/node. The matched evaluated-prefix baseline sustains
364.467 tok/s (30.372/node), giving **2.255x** higher sustained throughput.
These rates do **not** meet sustained 130+/node.

The short regression remains at 1578.182 steady tok/s (131.515/node), with
256/256 generated IDs against the F32 reference. Do not report this short-
context result as the long-context rate. Detailed baseline/configuration logs
and follow-up work are recorded in `resume-prefill.md`.

Prefix-reuse correctness is verified against a direct 34,688-token evaluation:
baseline, optimized cached-prefix, and direct runs have the same final residual
hash (`e4b288d2d51e5f88`). Kernel unit tests and shell syntax checks pass.

## Packed QK and long-context stage balance (job 51931098)

Prefill now uses a 12-row × 32-key FP32 SVE tile, sharing each packed key
panel across two queries and their six GQA heads. Only attention layers owned
by a pipeline stage get a persistent packed-key copy; newly written key panels
are repacked after each chunk. At 32k this adds about 128 MiB per owned
attention layer. Non-pipeline prefill packs keys on demand.

`Q38P_ATTN_GEMM=0` restores the previous QK arithmetic;
`Q38P_ATTN_CACHE=0` retains GEMM but repacks keys per task. The new QK uses
sequential FMAs instead of the previous 16-lane reduction, so snapshot
checksums change. Softmax, PV and decode arithmetic stay unchanged.
`test_gemm_attention.c` checks cached versus uncached scores and a scalar FMA
reference, including odd query counts and partial key panels.

The partitioner now minimizes the largest estimated stage cost. Its portable
`test_q38p_partition.c` compares with exhaustive enumeration and checks
nonempty coverage for all 1–128 rank counts. Use attention cost 3500 for the
32k-depth benchmark, or 2000 for the full coding prompt growing from zero:

```sh
Q38P_PP_ATTN_COST=3500 bash a64fx/llm/q38p/run_depth.sh \
  tmp/q38p/depth/NEW_PACKED 32768 1920 6 tokens 480 12
Q38P_PP_ATTN_COST=2000 HANDOFF_PREFIX_TOKENS=32508 HANDOFF_TPS='4 2' \
  bash a64fx/llm/q38p/run_handoff.sh tmp/q38p/handoff/NEW_PACKED \
  0 8192 480 tmp/q38p/review-c-source.txt
```

Measured on the same 12-node allocation:

| 32k depth, 1920-token suffix, six repeats | Steady tok/s | Per node |
| --- | ---: | ---: |
| Previous kernel, synthetic KV | 823.052 | 68.588 |
| Cached packed QK, previous stage cuts, synthetic KV | 923.151 | 76.929 |
| Cached QK, minimax cuts / cost 3500, synthetic KV | 978.424 | 81.535 |
| Selected configuration, evaluated random-token prefix | 967.713 | 80.643 |

Evaluated-prefix setup takes 31.813683 s once; six measured suffixes take
16.525489 s including pipeline fill/drain. The first suffix takes 6.605 s.
Compared with the previous evaluated-prefix rate of 68.485/node, sustained
throughput improves 17.8%. **The 130/node target remains unmet.**

Full 32,572-token coding prefill takes 29.530393 s, **1102.999 tok/s
(91.917/node)**, up from 1027.275; snapshot export takes another 5.552113 s.
The first 256 TP2 continuation IDs match the existing full-prompt baseline;
selected logits differ by at most 0.0474892. The 1024- and 4096-token gathered
decode checks also pass 256/256 F32 IDs. This does not imply identical state
bits or a full-generation F32 reference check.

Three larger PV layouts were rejected: packing both inputs, twelve separate
probability streams, and packed probabilities with direct V. Padding removes
a severe cache conflict from separate probability streams but does not beat
the retained PV kernel. Query tile 16 also regresses. Chunk 960 gives only
82.321/node and increases first-suffix latency to 11.100 s; keep chunk 480.
Logs are under `tmp/q38p/packed/` remotely and
`tmp/q38-packed/results/` locally.

### Final full 32k-input / 8k-output validation

Final sources rebuilt and tested on job 51931098. The 32,508-token producer
prefix takes **29.486634 s = 1102.466 aggregate prefill tok/s** (91.872/node),
with snapshot export separately timed at 5.438342 s. Each decoder replays the
remaining 64 known prompt tokens before generating 8192 outputs:

| Decode group | Replay s | Generation s | Decode tok/s | Baseline IDs |
| --- | ---: | ---: | ---: | ---: |
| TP4 | 1.015371 | 141.577 | **57.863** | **8192/8192** |
| TP2 | 1.687514 | 235.698 | **34.756** | **8192/8192** |

Both match `code32k-cmg/decode.tp2.rank0.log`, with maximum selected-logit
absolute differences 0.102028 (TP4) and 0.101627 (TP2). This is full-length
agreement with the existing quantized baseline, not a full-length F32 check.
Decode throughput is within 0.3% of the previous 57.958 / 34.831 measurements.
Imports take 7.761373 s (TP4) / 21.817327 s (TP2), outside generation timing;
shared-filesystem snapshot transfer is still not a production handoff.

Validation command:

```sh
python3 a64fx/llm/q38p/check_handoff.py \
  tmp/q38p/handoff/code32k-packed-prefix64 \
  --reference tmp/q38p/handoff/code32k-cmg/decode.tp2.rank0.log --require-exact
```

### Later 12-node PV experiments (job 51931363)

The exact-path defaults above remain selected: cached FP32 QK, FP32 PV,
query tile 8 and key tile 32. A final rebuild measured **970.166 aggregate
steady tok/s (80.847/node)** at depth 32768 with a 1920-token suffix and six
synthetic-KV repeats. The 1024-token F32 continuation check remains 256/256.
The 130 tok/s/node long-context target is still unmet.

`Q38P_ATTN_PV_INT16=1` enables an experimental SDOT PV path. It caches
int16 value panels and uses per-panel probability scaling. Add
`Q38P_ATTN_PV_RESIDUAL=1` for a second value panel that captures rounding
residuals. Both options default to off. **Do not use either for a coding-task
run requiring the validated 8192-token TP2/TP4 IDs.** The single-panel path
reached 99.3 tok/s/node at 32k and 1178.9 aggregate tok/s on the fixed
32,508-token producer prefix, but its TP2 continuation first differed from
the baseline at token 7018. The residual path reached 92.0 tok/s/node and
also differed at token 7018. A temporary 256-key variant differed at 4377.
The flags are retained for numerical and kernel work, not selected by the
runner. `test_pv_int16.c` checks partial panels and causal tails.

Exact FP32 key tiles 16, 64 and 128, and query tiles 4 and 12, all regressed
on the full 12-node 32k benchmark. Local isolated prototypes for alternative
FP32 PV and int16 QK are under `tmp/q38-next/`; none were integrated. See
`resume-prefill.md` for the measured rates, full-length comparisons, and the
projection-GEMM work estimate needed to approach 130/node.

### Probability residual experiment (job 51931991)

`Q38P_ATTN_PV_P_RESIDUAL=1` adds a second int16 probability panel and
requires `Q38P_ATTN_PV_INT16=1`. With
`Q38P_ATTN_PV_RESIDUAL=1`, PV uses three SDOT products: P0V0, P1V0 and
P0V1. The omitted P1V1 term is second order in the two rounding errors.
Probability residual scaling uses the int16 half-step bound, avoiding an
extra row scan. All three flags default to off.

| PV path | 32k steady tok/s/node | Coding prefix tok/s, aggregate | Full TP2 baseline IDs |
| --- | ---: | ---: | --- |
| Validated FP32 default | 80.847 | 1102.466 | 8192/8192 |
| Int16 base, previous fixed P scale | 99.266 | 1178.283 | first mismatch 11 |
| Int16 base, per-512-key P scale | not separately timed | 1178.912 | first mismatch 7018 |
| Int16 + P residual, measured before scan removal | 87.020 | 1125.701 | first mismatch 4377 |
| Int16 + P residual, fixed scale | 87.094 | not measured | not measured |
| Int16 + P and V residual, three SDOT passes | 78.559 | 1041.679 | first mismatch 4377 |

The three-pass 256-token TP2 check passed, but its 8192-token run differed
at token 4377. Generation from the P-residual and three-pass snapshots was
35.047 and 35.093 tok/s respectively; generation timing is independent of
the producer's PV choice. The longest `test_pv_int16.c` case has maximum
unnormalized PV error 2.83e-6 against a double sum with both residuals,
versus 5.64e-5 for the original FP32 FMA order against that sum. Improving
the mathematical PV approximation therefore does not ensure identical
autoregressive IDs. **Keep the validated FP32 default for coding tasks.**

Logs are under `tmp/q38p/presidual2/` and
`tmp/q38p/handoff/code32k-int16-{pres,both3}*` on Fugaku. The 130
tok/s/node 32k target remains unmet; the earlier stage-work profile shows
that PV optimization alone cannot reach it.
