# Qwen3.8-Flash-Next prefill tuning

Measured on RX 9070 XT (gfx1201, 16 GiB) with the Q4_K_XL GGUF,
CPU expert kernels, and 16 OpenMP threads. These are cold-prefill runs:
model loading and kernel initialization are excluded, with no prefill warmup.
The prompt is the first 9,000 bytes of `common/gguf_loader.h` (3,666 tokens),
followed by 64 decode steps. It is source completion, not a chat quality test.

| Standalone configuration | Prefill tok/s | Decode tok/s |
| --- | ---: | ---: |
| Previous 512-token chunks, grouped off | 154.42 | 14.51 |
| 1024-token chunks, grouped off | 197.15 | 35.53 |
| 1024-token chunks, grouped on, LFU deferral fixed | 186.85 | 35.43 |
| Final default, no chunk override, repeat | 196.54 | 35.44 |

The two 1024-token runs produced the same decoded sequence hash:
`8d60d4e54ae1a766`. The 512-token run produced `44f2d710660f40ab`.
The final-default repeat also produced `8d60d4e54ae1a766`.
Approximate decode uses the resident expert set left by prefill, so changing
chunk size can change generated tokens as well as speed. No new expert
omission or quantization is introduced in prefill by these changes.

The Qwen HIP prefill default now uses up to 1024 rows, capped by `LLM_BMAX`.
`LLM_MOE_CHUNK` and `LLM_PREFILL_CHUNK` still override the chunk size.
Other architectures retain their existing defaults.

Grouped cold-expert deferral previously assumed round-robin replacement.
LFU can recycle a slot before its deferred expert is executed, causing an
invalid cache-map lookup and a GPU memory fault. LFU misses now execute
immediately; resident experts may still be grouped. The workload that faulted
before the fix completed afterwards with the same sequence as ungrouped mode.
Ungrouped execution remains the faster standalone setting in this measurement.

## Server memory profile

The server launcher retains 512-row batches, a 7200 MiB expert cache,
grouped prefill, and disabled LFU/copy pipeline. With its 65536-token context
allocation, the same benchmark measured **120.35 tok/s prefill and 32.91 tok/s
decode**. This was a runner benchmark with the server's settings, not an HTTP
latency measurement or a full 65536-token prompt.

Increasing that profile to 1024-row batches exhausted VRAM during load.
Switching to ungrouped pipelined prefill at 512 rows reached 135.21 tok/s,
but reduced decode to 20.05 tok/s. Neither setting is adopted for the server.
The standalone 4096-context profile has room for 1024-row batches and an
8192 MiB expert cache. All measurements use approximate decode; they do not
establish coding or research answer quality.

## Reproduce

From the repository root:

```sh
make -C rdna4/llm -j8 test_hip_llm
rdna4/llm/run_qwen38_flash_next_rocm.sh -s 4096 -n 4096 \
  -t "$(head -c 9000 common/gguf_loader.h)" --decode 64
```

Set `LLM_MOE_CHUNK=512` for the old chunk size, or
`LLM_MOE_GROUPED_PREFILL=1` to exercise the grouped/LFU fix.
Use `LLM_QWEN4_APPROX_DECODE=0 LLM_QWEN4_DEVICE_HITS_ONLY=0` for exact decode;
the throughput numbers above are for the launcher's approximate default.

To reproduce the retained server memory/scheduling profile in the benchmark:

```sh
env QWEN38_MOE_CACHE_MB=7200 LLM_BMAX=512 \
  LLM_MOE_COPY_PIPELINE=0 LLM_MOE_LFU_CACHE=0 LLM_MOE_GROUPED_PREFILL=1 \
  LLM_Q4_2W=0 LLM_Q5_DOWN_2W=1 LLM_HC_GRAPHS=1 \
  OMP_PROC_BIND=close OMP_PLACES=cores \
  rdna4/llm/run_qwen38_flash_next_rocm.sh -s 65536 -n 4096 \
  -t "$(head -c 9000 common/gguf_loader.h)" --decode 64
```

The earlier six-token prefill measurement (~13 tok/s) does not measure
large-prompt throughput. Likewise, ~58 tok/s decode after that tiny prompt
does not establish decode throughput after a long prompt. Compare both
phases using the same prompt, context allocation, cache budget, and settings.

The final short-prompt regression run (`--decode 32 --prefill-len 1`) measured
13.22 tok/s prefill and 57.96 tok/s decode. Its sequence hash
`839cdbe6237ed0c5` matches the pre-change short-prompt run (58.04 tok/s decode).
The build, launcher shell syntax checks, and `git diff --check` passed.

## Current RX 9070 XT validation (September 2026)

The older table above is retained as historical data; it is not reproducible
with the current Qwen4 expert-cache implementation.  The current safe profile
for a 3,666-token prompt is unregistered host memory, grouped prefill disabled,
and a 1,024-row tile.  Recent measured points are:

| Prompt / profile | Prefill tok/s | Decode tok/s | End-to-end tok/s | Status |
| --- | ---: | ---: | ---: | --- |
| 1,275 tokens, BMAX=512, cache=7.2 GiB | 63.08 | 24.02 | 62.76 | PASS |
| 1,275 tokens, BMAX=1024, cache=7.2 GiB | 73.99 | 23.54 | 73.49 | PASS |
| 1,275 tokens, BMAX=2048, cache=5.0 GiB | 86.00 | 22.09 | 85.23 | PASS |
| 3,666 tokens, BMAX=512, cache=8.2 GiB | 68.20 | 31.28 | 67.99 | PASS |
| 3,666 tokens, BMAX=1024, cache=7.2 GiB | 89.08 | 32.39 | 88.38 | PASS |
| 3,666 tokens, BMAX=2048, cache=5.0 GiB | 114.57 | 24.65 | 113.66 | PASS |
| 3,666 tokens, BMAX=2048, cache=7.2 GiB, pinned+async (peak) | 198.40 | 26.20 | 195.60 | PASS |
| 3,666 tokens, BMAX=2048, cache=7.2 GiB, pinned+async (repeat) | 196.79 | 26.68 | 194.10 | PASS |
| 3,666 tokens, BMAX=2048, cache=7.6 GiB, scratch-alias+async | 195.96 | 27.09 | 193.33 | PASS |
| 3,666 tokens, BMAX=2048, cache=7.8 GiB, dual scratch-alias+async | 196.58 | 27.13 | 193.94 | PASS |
| 3,666 tokens, BMAX=2048, cache=7.8 GiB, repeat | 187.46 | 27.39 | 185.10 | PASS |
| 3,666 tokens, BMAX=2048, cache=7.8 GiB, 64-token decode | 196.20 | 32.55 | 180.62 | PASS |
| 3,666 tokens, cache=7.8 GiB, CPU count=3/jobs=160 | 196.13 | 26.66 | 193.45 | PASS |
| 3,666 tokens, GPU router top-k, cache=7.8 GiB | 232.52 | 32.53 | 210.34 | PASS |
| 4,096 tokens, GPU router top-k, cache=7.8 GiB | 235.25 | 31.75 | 214.14 | PASS |
| 4,096 tokens, GPU top-k + exact decode (8 tokens) | 234.26 | 11.87 | 226.01 | PASS |

Host registration (`LLM_MOE_REGISTER_HOST=1`) reached 110.70 tok/s on the
1,275-token test and 133.05 tok/s with BMAX=2048, but full 3,666-token runs
terminated during prefill on this 16-GiB card.  It is therefore diagnostic,
not a stable production setting.  The opt-in grouped Qwen4 prefill path also
still faults in `qwen4_gateup_silu_q4k_grouped`; it remains disabled.

The GPU-top-k router was also tested with grouped prefill enabled.  A single
short run reached the expected prefill rate, but repeated full-depth 3,666-token
runs terminated before the benchmark footer (including an 8-token decode
control), and the following ungrouped control could not complete until the
device recovered.  This is treated as a GPU-reset/stability failure, not a
valid performance result; grouped prefill is therefore still opt-in.

These results do not meet the 200 tok/s 4K target.  The remaining gap is cold
expert execution and upload bandwidth (the safe 1,275-token BMAX=2048 run
still uploaded 39.2 GiB); no decode or quality-changing approximation was
enabled for the measurements above.

The faster BMAX=2048 profile is stable for the full 3,666-token prompt with a
5.0-GiB cache and raises prefill from 89.08 to 114.57 tok/s.  It lowers
decode to 24.65 tok/s, however, so the launcher keeps its decode-safe
BMAX=512/7.2-GiB default; select BMAX=2048 and 5.0 GiB explicitly when prefill
is the priority.  BMAX=4096 fails allocation during weight load on the 16-GiB
card.  The previous 9.2-GiB default also failed to load reliably with batched
scratch buffers.
It also defaults `LLM_MOE_COPY_PIPELINE=0`; the asynchronous copy pipeline is
available as an explicit experiment.  The current fastest stable pinned-host
profile is:

```sh
env QWEN38_MOE_CACHE_MB=7800 LLM_BMAX=2048 LLM_MOE_CHUNK=2048 \
  LLM_MOE_REGISTER_HOST=1 LLM_MOE_COPY_PIPELINE=1 \
  rdna4/llm/run_qwen38_flash_next_rocm.sh -s 8192 -n 4096 \
  -t "$(head -c 9000 common/gguf_loader.h)" --decode 8
```

The same profile can be selected with `QWEN38_FAST_PREFILL=1`; it defaults to
the 7.8-GiB cache after reusing the gathered-input/output scratch allocation.
Explicit cache, `LLM_BMAX`, host-registration, and copy-pipeline overrides
remain honored. An 8.0-GiB cache fails allocation on this 16-GiB card.

The runner now avoids allocating the ~2-GiB grouped-verifier logits matrix
unless `LLM_QWEN4_GROUPED_VERIFY=1`, processes resident experts before cold
experts on the next tile, and defers ordinary prefill cache-map writes until
decode mode.  That reduced measured H2D volume from about 95 GiB to 88.6 GiB
without changing routing or the decoded sequence.  The
`LLM_MOE_LFU_CACHE=0` override is now honored for experiments; LFU remains the
faster default on this prompt.  The
launcher default remains conservative because this fast profile's decode rate
is lower than the smaller-tile decode-safe profile.
The Qwen4 batched MoE path now aliases gathered-input storage as expert-output
storage and reuses the up-output lifetime for down output, reclaiming roughly
250 MiB. The unused mapped-miss staging buffers are also no longer allocated
when `LLM_QWEN4_MAPPED_MISSES=0`.
The fast profile now enables `LLM_QWEN4_PREFILL_GPU_TOPK=1`, which keeps
router top-k/count/fill on the GPU and removes the per-layer router D2H
synchronization. It raised 3,666-token prefill from roughly 196 to 232 tok/s;
the exact 4,096-token boundary measured 235.25 prefill / 214.14 end-to-end
tok/s with 64-token decode. The option remains off for conservative and exact
profiles by default; the explicit exact 4K parity run still measured 234.26
prefill / 226.01 end-to-end tok/s and returned `PASS`.
The 64-token run above preserved the same first token and completed with
`PASS`; an explicit exact-decode run also produced hash
`31f3aa7036a01bca`, confirming the scratch reuse does not alter decode values.
Allowing three-token CPU expert jobs reduced H2D to 84.37 GiB, but raised CPU
time to 1.32 s without improving end-to-end throughput; the default remains
the faster count=2/jobs=160 balance.
An exploratory count=4/jobs=384 setting reached 78.29 GiB H2D but regressed to
193.12 tok/s and produced a different short decode hash, so it is rejected on
both speed and quality grounds.
Repeated runs after deferred map publication reached 197.90–198.40 prefill
tok/s and 195.25–195.60 end-to-end tok/s.  A 64-token decode sample measured
33.00 decode tok/s with a PASS.  GPU clock variance still produces occasional
lower prefill samples, so the 200 tok/s target remains open.
An experimental second H2D copy stream was rejected after measuring 146.12
prefill / 144.75 end-to-end tok/s with no decode or quality gain. The production
path therefore retains one copy stream.

### 256K batched-prefix memory fix

At the full 256K allocation, enabling `LLM_QWEN4_BATCH=1` originally failed
during weight finalization because the batched path eagerly reserved BF16 copies
and per-token scratch for all five SSM projections. The verified Qwen4 schedule
keeps SSM layers on the ordered scalar body (`LLM_QWEN4_BATCH_SSM=0`), so those
copies and scratch arrays are now omitted by default and created only when the
experimental batched-SSM gate is explicitly enabled. Scalar SSM arithmetic and
the KV-cache layout are unchanged.

With I8 KV, `LLM_MOE_CACHE_MB=5000`, `LLM_BMAX=512`, and the verified attention
prefix guard (`LLM_QWEN4_BATCH_ATTN_MAX_LAYER=2`), the batched path now loads at
256K and passes the scalar quality hash. A 128-token control measured 14.39
prefill / 6.32 decode tok/s versus 10.05 / 2.98 for scalar; a longer 512-token
control produced the identical hash (`1b67330f97f98543`) and measured 14.30 /
13.89 versus 14.00 / 14.29. It therefore remains opt-in rather than a
production default. Reproduce it with `bench_qwen38_256k.sh` using
`LLM_QWEN4_BATCH=1 LLM_MOE_CACHE_MB=5000`; the script exposes both batch guards
as environment overrides. Multi-chunk Qwen4 batching is disabled by default:
the 2K/BMAX=1024 experiment faulted the gfx1201 VM on its second chunk, so
requests larger than BMAX take the scalar fallback until recurrent state carry
is validated. `LLM_QWEN4_BATCH_MULTI_CHUNK=1` is retained only for diagnostics.
For the same reason, production batched Qwen4 prefill is limited to
`position_start==0`; subsequent streamed chunks use scalar state carry.
`LLM_QWEN4_BATCH_STATEFUL=1` disables this guard for focused experiments only.

The conservative launcher tile remains BMAX=512; the measured larger-tile
profile is opt-in and reproducible with:

```sh
env QWEN38_MOE_CACHE_MB=7200 LLM_BMAX=1024 LLM_MOE_CHUNK=1024 \
  LLM_MOE_GROUPED_PREFILL=0 LLM_MOE_REGISTER_HOST=0 \
  rdna4/llm/run_qwen38_flash_next_rocm.sh -s 8192 -n 4096 \
  -t "$(head -c 9000 common/gguf_loader.h)" --decode 8
```

Grouped-prefill control buffers now use synchronous H2D copies.  The previous
async copies raced with reuse of the host task/map arrays and could reset the
GPU.  With `LLM_MOE_GROUPED_PREFILL=1 LLM_MOE_LFU_CACHE=1`, a 1,275-token
validation prompt now preserves the baseline sequence hash
`5200a271647abaf3` and measures 75.44 tok/s prefill.  Full 3,666-token grouped
runs still reset the 16-GiB card once resident grouping begins, even after
explicit stream synchronization before each grouped launch, so the grouped
path remains opt-in and is not part of the quality-preserving default profile.
Layer-limited diagnosis is stable through 43 of 48 layers; failures appear
only in the late-layer/full-depth run, consistent with cumulative cache/map
state or a late-layer resource transition.

The production fast profile consequently enables GPU router top-k but keeps
`LLM_MOE_GROUPED_PREFILL=0`.  The former removes the full router-logit D2H
transfer and is covered by the 4,096-token PASS rows above; the latter remains
an experimental path until a repeatable full-depth run survives reset and
quality checks.

The Codex HTTP launcher now matches that production rule: grouped prefill is
off by default, including when `QWEN38_BATCH_PREFILL=1`.  It can still be
enabled explicitly for bounded diagnostics, but a grouped full-depth request
must not be used as a stability or throughput result on the 16-GiB gfx1201
card.

Current 256K smoke results with the scalar I8-KV profile (`cache=5900 MiB`,
`BMAX=512`, graph/plan warmup disabled, direct copies, grouped off) are:

The smoke script reports any existing `/dev/kfd` users. Do not compare
throughput or attribute MES/VRAM failures to a kernel change while another
ROCm client is holding the device; stop the competing workload first.

| Prefill / decode | Prefill tok/s | Decode tok/s | End-to-end tok/s | Hash / status |
| --- | ---: | ---: | ---: | --- |
| 32 / 8, exact | 9.06 | 5.39 | 7.97 | `b08cf8f036d9b0fa`, PASS |
| 8 / 32, exact | 4.91 | 5.21 | 5.15 | `a48257443e52cda2`, PASS |
| 128 / 16, exact | 10.02 | 5.16 | 9.07 | `75aa7fd4f21cfee1`, PASS |
| 8 / 32, approximate, refresh=32 | 4.81 | 25.70 | 13.76 | PASS |
| 128 / 16, exact, decode-cache balance (experimental) | 12.05 | 5.72 | 10.73 | `75aa7fd4f21cfee1`, PASS |
| 128 / 16, approximate, refresh=32 | 15.12 | 41.28 | 16.27 | `84a63e4d612d20bb`, PASS |

The exact 256K decode rate is currently limited by cold routed-expert misses:
the 128/16 run reported 57.4% decode-cache hit and 9.58 GiB of H2D expert
traffic.  Increasing the cache to 6.2 GiB fails during device allocation on
the 16-GiB card, so the exact path is not silently traded for approximation.
The approximate row is an explicit quality/performance experiment and is not
the default coding profile.

The 128/16 approximate run is substantially faster, but its hash and final
token differ from the exact control (`75aa7fd4f21cfee1`), so it is not a
quality-preserving replacement.  Keep `LLM_QWEN4_APPROX_DECODE=0` for exact
coding/research output.

The 256K launcher and smoke benchmark now default to direct expert copies
(`LLM_MOE_COPY_PIPELINE=0`).  Two matched no-copy runs measured 5.99 and
6.73 exact decode tok/s, versus 5.16 and 5.33 with the asynchronous pipeline;
the async mode remains available explicitly for other workloads.

`LLM_QWEN4_DECODE_CACHE_BALANCE=1` selects the experimental allocation used
by the final row.  It preserves the exact sequence hash and gave a small
single-run decode improvement, but the aggregate cache hit rate remained
about 58%.  A matched control completed at 5.33 decode tok/s with the same
hash; a second immediate cold-card repetition stopped during the first
prefill after weight load, so the allocator is not promoted to the default
until repeated runs show a durable benefit without a device reset.

### Long-context coding stress test

The test harness now accepts `--prompt-file`, and its tokenizer buffer scales
with `-s` instead of silently capping prompts at 4,096 tokens.  A generated
coding workload tokenized to 32,068 input tokens.  On the 15.9-GiB RX 9070 XT,
weight loading failed for both a 40,960-token context (needed for 32K + 8K)
and a 32,768-token prefill-only context, even with the expert cache disabled
and BMAX=1.  No quality or 8K-output claim is made from those runs: this is a
VRAM/context-capacity limit, not a decoder correctness result.  The 4K
quality/stability PASS measurements above remain the largest reproducible
production runs on this card.

The stdio/server path was also exercised with `LLM_BMAX=512`: it preserved KV
state across streamed prefill calls and completed a coding control request.
Single-sequence decode remains token-at-a-time because each token depends on
the previous sampled token; llama-server-style decode batching applies across
concurrent sequences.  `LLM_BMAX` is therefore the relevant single-request
prefill microbatch control here.

### KV-cache capacity and quantization

Qwen3.8-Flash-Next has 48 layers with `full_attention_interval=4`, so only 12
layers allocate token KV.  Each has 2 KV heads × 256 head dimensions, and the
runner currently stores separate F16 K and V arrays:

```
bytes = context × 12 × 2(K/V) × 2(KV heads) × 256(head dim) × 2(F16 bytes)
```

At a 256K context this is 6,442,450,944 bytes = 6.000 GiB of KV alone.  An
ideal FP8 cache would be 3.000 GiB, and packed 4-bit KV would be 1.500 GiB,
before per-group scales/zero-points and alignment.  The runner has no validated
FP8/FP4 KV store or attention-dequant kernel today; enabling one without
calibration and a quality comparison would risk silently changing coding
output.  The loader now prints the F16/FP8/FP4 estimates at startup so future
implementations can be checked against the actual layer geometry.
The runtime now rejects `LLM_QWEN4_KV_QUANT=fp8`/`fp4` explicitly rather than
silently allocating an F16 cache; use `none`/`f16` or the validated capacity
mode `i8`.

The experimental scaled-I8 layout uses eight 32-channel scales per KV head;
at 256K this adds about 0.18 GiB, for roughly 3.18 GiB total KV-plus-scales.
The 256K allocation probe now succeeds on the RX 9070 XT: with the routed
expert cache disabled, the loader reported 9.65 GiB free before KV allocation,
reserved 3.000 GiB of I8 K/V, and completed weight loading.  The same max-seq
profile loaded with BMAX=512 and the batched prefill path enabled.  This proves
capacity and startup stability; it is not a claim that a 256K prompt reaches
the short-context 200+ tok/s rate.

`run_qwen38_flash_next_rocm.sh` keeps F16 for ordinary contexts and selects
`LLM_QWEN4_KV_QUANT=i8` automatically for explicit 16-GiB requests at or above
131K tokens (override explicitly to compare either path).  The larger
`24g`/`32g` profiles retain F16 by default.  The path stores
symmetric 8-bit K/V values
with per-token/per-KV-head/group (32-channel) scales.  It is a memory
experiment (not IEEE FP8) with a matching quantized-QSA kernel. On a coding
control prompt, scaled-I8 and F16 both passed and
produced the same greedy sequence hash
`00e1aa748b2006a7`; the 4K cache falls from 0.094 to 0.047 GiB, plus scale
metadata.  This long-context fallback is deliberate: the 4K F16 control
continued coherently in the coding benchmark.  The current scalar control
has since been extended to 512 prompt tokens plus 64 greedy decode tokens;
F16 and scaled-I8 produced the identical sequence hash
`afed1f992f398ac3`, so the remaining quality concern is the experimental
batched arithmetic rather than the scalar I8 KV representation.
The same scalar I8 control with the full 262,144-token allocation produced
hash `88a3e47bf4121b33` for 16 generated tokens and returned `PASS`.

Measured with the full tuned launcher on the RX 9070 XT (BMAX=2048, GPU
router top-k, `LLM_QWEN4_BATCH=1`, 16 generated tokens): F16 at 4K reached
253.1 prefill tok/s and 30.3 decode tok/s; I8 reached 238.5 prefill tok/s and
29.4 decode tok/s.  A
256K-capacity I8 run loaded successfully with the automatic long-context
profile.  The earlier 4.0-GiB/BMAX=2048 profile measured 156.3 prefill tok/s
on a short streamed tile; the current quality-safe default uses 5.0 GiB and
BMAX=1024 because that combination is the one that fits reliably with the
full 256K allocation.
The HTTP server launcher was also started with `QWEN38_CONTEXT=262144`; it
reached `JSONL backend ready` with the automatic I8 profile.

Latest scalar quality regression on the same RX 9070 XT coding control
(F16 KV, `LLM_QWEN4_BATCH=0`, 12 prompt / 4 generated tokens) measured
4.28 prefill tok/s, 4.66 decode tok/s, and 4.37 end-to-end tok/s with
sequence hash `00e1aa748b2006a7` and `PASS`.  These are short-control
quality numbers, not a 256K throughput claim; the production launcher keeps
the scalar route until batched numerical parity is proven.

Qwen4 batched prefill and the experimental warp-per-key I8 attention kernel
remain opt-in (`LLM_QWEN4_BATCH=1` and `LLM_ATTN_PREFILL_I8_WARP=1`).  The
quality-safe launchers leave both disabled because routed-MoE batch parity is
not yet established.  The standalone HC batch verifier passes (`rel_l2≈1.23e-3`),
but full multi-token Qwen4 still diverges, so that local check is not sufficient
to enable the production path.  The experimental path now keeps Qwen4 SSM
recurrences scalar (the batch SSM trace was the first large state mismatch),
while retaining batched dense-attention work for further parity experiments.

The launchers expose the less error-prone alias `QWEN38_BATCH_PREFILL=1`.
It enables the same batched route while defaulting
`LLM_QWEN4_PREFILL_GPU_TOPK=0` (host router grouping); multi-chunk batching
remains diagnostic-only.  `QWEN38_BATCH_PREFILL=0` forces the scalar route.
For the quality-safe scalar 16-GiB profile at 128K+, GPU router top-k is now
enabled by default: a preserved 512-token 256K control matched the scalar
sequence hash `482e10d864607703` and measured 15.23 prefill / 14.47 decode
tok/s versus 11.41 / 11.00 with host grouping.  Explicit
`LLM_QWEN4_PREFILL_GPU_TOPK=0` restores the conservative host route.
At 2,048 padded tokens the same scalar profile measured `10.68` prefill /
`10.66` decode tok/s with the unchanged hash `482e10d864607703` (the host
router control was `9.94` / `8.44`), confirming the gain persists at larger
streamed-prefill scale.

The batched verifier now carries the actual token IDs and the two-token PLE
history into its ordered layer-1 fallback.  Before this fix, batched Qwen4
prefill skipped the token-dependent PLE n-gram state and could diverge by
`rel_l2≈1` after only two layers.  The final HC head also remains scalar by
default; `LLM_QWEN4_BATCH_HEAD=1` is required for the BF16-staged head because
it can move coding argmaxes.  With `LLM_QWEN4_BATCH_SSM=0`, the corrected
scalar-SSM comparator is now exact (`rel_l2=0`, identical argmax) through the
PLE layer.  Full-model batched attention/MoE still accumulates a measurable
logit delta (~7.5% on the short control), and fused SSM recurrence remains
similarly sensitive, so the production launchers continue to keep
`QWEN38_BATCH_PREFILL=0`.

For targeted kernel experiments, `LLM_SSM_BATCH_RECURRENCE=1` enables the
fused M-step DeltaNet recurrence and `LLM_SSM_BATCH_CONV=1` enables the fused
M-step depthwise convolution.  Both are intentionally unset in the quality
profile; the scalar state transitions are the reference used for parity.
`LLM_QWEN4_BATCH_ATTN=0` isolates SSM batching from the non-SSM attention/FFN
batcher; it measured `rel_l2≈0.045` on the short full-model comparator, so it
also remains diagnostic rather than a production setting.

The fused recurrence now has an additional `LLM_SSM_BATCH_PARITY=1` mode.  It
uses the scalar `expf` primitive and performs the decay, delta update, and
output dot product in increasing state-index order while retaining the
register-resident multi-token kernel.  On the RX 9070 XT, a 9-token comparator
with attention batching isolated changed `rel_l2` from `2.99e-2` to
`2.94e-2`; the full batched attention/MoE path measured `3.28e-2` without the
mode and `5.23e-2` with it, showing that its remaining divergence is outside
DeltaNet.  The full batched path is therefore still diagnostic, and the
production launcher keeps all Qwen4 batching disabled.

The HTTP shim now includes `e2e_ms` and `e2e_tok_s` in the completion
`performance` object.  `e2e_tok_s` uses uncached prompt tokens plus generated
tokens and measures the complete runner request, including protocol and
synchronization overhead; `pp_tok_s` and `tg_tok_s` remain component rates.
An actual 256K HTTP coding request reached `JSONL backend ready` with the I8
KV profile and returned `e2e_ms=5341.45`, `e2e_tok_s=6.18`, `pp_tok_s=5.59`,
and `tg_tok_s=26.43` for a 29-token prompt and four generated tokens.

The long-context bottleneck investigation added an explicit
`LLM_QWEN4_BATCH_SSM=1` experiment gate.  On the RX 9070 XT it raised a
512-token F16 prefill from about 12 to 111 tok/s (93 tok/s end-to-end for an
8-token decode), confirming that the 36 recurrent layers dominate the scalar
path.  It is not production-safe: with a 64-token decode the scalar control
hash was `afed1f992f398ac3`, while batched SSM produced
`a14fcdc04e1c5021`.  The gate therefore remains disabled by default and the
quality-safe route keeps Qwen4 SSM recurrence scalar.

With the quality-safe SSM path at a full 256K allocation, a 1,024-token
prefill measured 20.65 tok/s with the implicit 4-GiB expert cache and 21.30
tok/s with GPU router top-k; both returned the same greedy control hash and
`PASS`.  The current 5-GiB cache/BMAX=1024 profile, with launcher LFU
retention enabled, measured 22.87 tok/s and 12.62 decode tok/s with the same
hash; a 5-GiB cache without LFU reduced H2D to 77.3 GiB on a 512-token tile
and measured 21.91 tok/s.  The remaining gap to
short-context rates is therefore recurrent SSM work and expert cold traffic,
not KV capacity; the 5-GiB setting is not selected automatically because a
2K scratch tile cannot coexist with it on a 16-GiB card.

`LLM_MOE_COPY_PIPELINE=1` was also checked at full 256K allocation.  With
four stream slots it passed both 1,024- and 2,048-token controls, preserving
hash `88a3e47bf4121b33`; the 1K run measured 23.31 prefill / 14.89 decode
tok/s, and the 2K run measured 26.19 / 12.98 tok/s.  A 4,096-token control
also passed with the same hash at 28.99 / 11.95 tok/s.  An 8,192-token
control passed at 31.32 / 9.67 tok/s. These are historical pipeline-enabled
measurements; current 16-GiB launchers default to direct copies because newer
exact 256K decode controls measured lower latency without the pipeline.
`LLM_MOE_COPY_PIPELINE=1` remains available explicitly for reproducing these
older long-prefill results.

The resident-expert approximate decode profile (`LLM_QWEN4_APPROX_DECODE=1`,
device hits only) measured 29.41 decode tok/s at full 256K allocation with the
quality-safe 32-token refresh interval, and returned `PASS`.  A 64-token
refresh interval reaches 46.73 tok/s, but changes the 64-token control hash
(`0caab5d8b1da3b3b` versus `8cc86e5bcee4bafd`), so the launcher keeps 32 as
the default; 64 remains an explicit throughput experiment.

The HTTP coding launcher now makes the approximation opt-in: it exports
`LLM_QWEN4_APPROX_DECODE=0` unless the caller explicitly sets it to `1`.
Previously `--qwen4-coding-profile` overrode an explicit zero and could select
resident-hit decode, which produced malformed code in a streamed request.
After making the environment value authoritative, the exact default returned
the coherent control response `int add(int a, int b) { return a + b; }` at
`QWEN38_CONTEXT=4096` (33 prompt / 24 generated tokens), with
`pp_tok_s=6.16`, `tg_tok_s=13.07`, and end-to-end `7.92 tok/s`.  Approximate
decode remains available for measured throughput experiments via
`LLM_QWEN4_APPROX_DECODE=1`; it is not a quality claim.

The post-fix 256K default profile was rerun on the RX 9070 XT with the full
I8 KV allocation.  It reached `JSONL backend ready (max_seq_len=262144)` and
returned the same coherent C-code prefix; with 33 prompt and 16 generated
tokens it measured `pp_tok_s=5.16`, `tg_tok_s=10.58`, and end-to-end
`6.19 tok/s`.  A warm repeat reached `8.49` prefill / `13.00` decode and
`9.57 tok/s` end-to-end.  LFU retention was also tested on the exact route but
reduced decode to `2.94 tok/s`, so the default LRU policy remains selected.

An additional `LLM_ATTN_PREFILL_I8_GQA4=1` kernel reuses each I8 K row across
four query heads (the model has 24 query heads and 2 KV heads).  It compiles
and executes on gfx1201, but its current control hash differs from the scalar
I8 batch kernel (`c7ba9a6ac1abca2f` vs `43288d39a01f6bde`), so it remains
disabled by default pending a numerical-parity pass.

Two exact cold-miss experiments were also rerun at the 256K allocation.  BAR
mapped/direct misses preserved the coding prefix but reached only `11.96`
decode tok/s over an 8-token request, versus `14.08` tok/s for the ordinary
CPU-miss profile on the same card.  Reducing the delayed-cache refill interval
from the launcher default of two tokens to one reduced the same short run to
`11.01` decode tok/s and added host CPU time.  Neither setting is promoted;
the default remains CPU cold-miss handling with interval two.

Raising `LLM_MOE_CPU_PREFILL_MAX_COUNT` from the launcher default of two to
four (with a 256-job cap) was also tested at the full 256K allocation.  A
1,024-token scalar prefill measured `11.56 tok/s` and 8-token decode measured
`12.55 tok/s`; the higher singleton threshold did not improve the path, so the
default remains unchanged.

The exact 256K copy pipeline was compared with four versus eight stream slots.
The short 8-token coding control measured `12.60` versus `12.74` decode tok/s,
respectively, with identical routing statistics and output prefix.  The
difference is within run-to-run variance, so the launcher keeps four slots as
the lower-overhead default; `LLM_MOE_STREAM_SLOTS=8` remains available for
larger concurrent workloads.

The experimental `LLM_SSM_BATCH_Q6K=1` path batches Qwen4 Q6_K/F16 SSM
projections with one exact quantized block per token/output row, avoiding the
BF16 conversion used by the older grouped path.  It preserved the argmax on
the 9-token comparator and reduced its relative logit error from `2.94e-2` to
`2.82e-2`.  The gfx1201-tuned 128-thread reduction measured `15.70 tok/s`
at 128 tokens (versus `15.35 tok/s` with 256 threads and `15.77` without the
path); `LLM_SSM_BATCH_Q6K_THREADS=256` restores the wider block.  It remains
an explicit parity/diagnostic option rather than a production default.

The staged grouped-MoE profile was rechecked at the full 256K allocation with
I8 KV, BMAX=512, a 4-GiB expert cache, and 508 MiB of staging banks.  It
initialized safely and returned `PASS`, but a 512-token run measured only
`15.79 tok/s` prefill and `13.34 tok/s` decode (`5.30 GiB` staged H2D and 34
waves).  The high short-context staging numbers therefore do not carry over
to the 256K allocation; the exact scalar profile remains the production
choice.

`LLM_QWEN4_NATIVE_BATCH_QKV=1` likewise routes batched Q/K/V/O projections
through the native quantized matvec kernels instead of BF16 hipBLASLt staging.
It is useful for profiling dequantization order, but the 12-token control
currently hashes to `60e03495a717c14d`, so it is also opt-in and not a quality
safe production setting.

For an explicit `-s >= 131072`, the 16-GiB launcher now selects a 5900-MiB
implicit expert-cache budget and BMAX=512.  It also disables optional graph
captures and plan pre-warm allocations at full context; those transient
allocations made the older 6100-MiB/BMAX=1024 selection fail on current
gfx1201 VRAM fragmentation.  Explicit `QWEN38_MOE_CACHE_MB`, `LLM_BMAX`,
`LLM_HC_GRAPHS`, `LLM_QWEN_PRE_GRAPHS`, and `LLM_PLAN_PREWARM` settings still
override the production defaults.

The recovered 5900-MiB profile was loaded twice at 256K with I8 KV and passed
the exact scalar smoke gate.  A 128-token prefill / 32-token decode control
measured `12.15` / `6.63` tok/s and `10.42` end-to-end tok/s; the 5000-MiB
same-shape control measured `11.35` / `5.25` and `9.21` end-to-end tok/s.
The greedy output remained coherent.  With graph/plan reservations enabled,
5900 MiB failed in hipBLASLt scratch setup and 6100 MiB failed during the
expert-cache down allocation, so the smaller cache plus disabled transient
allocations is the current reproducible 16-GiB limit.

For the resident 256K coding-server profile, `LLM_MOE_CPU_DECODE_MISSES=0` is
now the launcher default. A 128-token exact control measured `9.22 tok/s`
decode with direct GPU cold misses versus `8.67 tok/s` with CPU misses, while
both produced hash `7a90d592cf2b8cfa`. The override remains available for
systems with different PCIe/CPU characteristics.

The explicit `QWEN38_BATCH_PREFILL=1` profile reserves a 4.0-GiB expert cache
and BMAX=512 when it must coexist with full hybrid scratch at 256K. A padded
direct 2,048-token control streamed as 512-token chunks returned the same greedy
hash `482e10d864607703` as scalar and measured 9.78 prefill / 10.16 decode
tok/s. However, a real heterogeneous 2K HTTP prompt closed the runner after
its first 512-token chunk. Consequently stateful Qwen4 batching now requires
`LLM_QWEN4_BATCH_MULTI_CHUNK_FORCE=1`; without it, the runtime keeps the whole
request scalar and cannot mix a batched first chunk with an unsafe continuation.
The HTTP runner now reports the complete request length to the dispatcher, so
an explicit batch profile may safely use one batch when the request fits BMAX;
longer prompts select scalar from their first chunk. The launchers leave the
force flag unset by default.

The batch implementation has a safety floor even when explicitly enabled:
`LLM_QWEN4_BATCH_MIN_TOKENS` defaults to 128. Lower this only for diagnostics;
the scalar default remains the recommended production setting until real
multi-chunk request parity is proven.

Per-layer 512-token statistics showed the decode-oriented slot table was
overprovisioned for early layers (98--99% hits) and underprovisioned at layers
23, 31, 39, and 40 (33--67% hits).  An explicit
`LLM_QWEN4_PREFILL_CACHE_BALANCE=1` experiment raised the 512-token control
from `25.5` to `29.7 tok/s` and reduced expert H2D traffic from `99.2` to
`69.8 GiB`, while preserving the exact greedy 256K hash
`a2d4f49620d5b663` across a 2,048-token streamed run (`14.75 tok/s` prefill,
`15.07 tok/s` decode).  It did not improve the high-entropy coding control
(`14.73 tok/s`, `943 GiB` H2D), so it remains opt-in; scalar and batch launchers
retain the original decode-balanced table by default.

A longer 256K exact coding control with the promoted profile returned the
complete coherent `int add(int a, int b) { return a + b; }` response
with 24 generated tokens.  It measured `6.11 tok/s` prefill and `12.57 tok/s`
decode over the request; this confirms the cache increase does not alter the
quality-safe output path.

Routing all cold experts to GPU staged copies (`LLM_MOE_CPU_DECODE_MISSES=0`)
was slower: the matched 8-token control fell to `7.70 decode tok/s` with
`3.87 GiB` H2D traffic, versus the CPU-miss profile’s roughly `12.9 tok/s`.
The CPU shadow path remains the exact 16-GiB default.

An exact 256K A/B with `OMP_NUM_THREADS=8` completed coherently but measured
`12.25 decode tok/s` (8 generated tokens, end-to-end API timing), below the
established 16-thread result of roughly `12.6--12.9 tok/s`.  The launcher
therefore retains its 16-thread default; reducing CPU workers is not a win on
the RX 9070 XT.

### Batched-parity isolation at 256K

The opt-in batched route was rechecked with the full 256K allocation.  The
6-GiB cache cannot coexist with its additional scratch buffers (HIP allocation
error 2), so a 4-GiB cache was used for diagnosis.  The route initialized and
ran, but returned an incoherent `user` completion at 13.7 decode tok/s.  Using
scalar MoE evaluation, native quantized QKV/O projections, scalar I8 attention,
or scalar HC mixing did not restore quality.  A 32-token comparator measured
`rel_l2=6.25e-2` (argmax happened to match), so this is not an acceptable
production path.  The new `LLM_QWEN4_BATCH_ATTN_MAX_LAYER` and
`LLM_QWEN4_BATCH_HC_SCALAR` knobs are diagnostic only; the launcher leaves
Qwen4 batching disabled and preserves exact scalar quality.

During this investigation, the batched dispatcher was also corrected to copy
the final `d_hc_batch` row back to the persistent `d_hc` state before the next
decode token.  This state handoff was previously missing.  It is required for
any future batched-quality work, although the remaining projection/FFN drift
still makes the route diagnostic-only today.  Scalar KV-store packing was also
tested and did not change the wrong chat-template first token.
An additional `LLM_QWEN4_BATCH_PROJ_SCALAR=1` run likewise left comparator
drift near `6.0e-2` and reduced decode to `16.1 tok/s`; scalarizing Q/K/V/O
projections is not a useful hybrid setting.
The opt-in `LLM_QWEN4_BATCH_EMBED_SCALAR=1` repeat reduced a 64-token
comparator from `7.94e-2` to `7.12e-2`, but did not restore parity, so it also
remains diagnostic-only.

The layer boundary was isolated on the RX 9070 XT with scalar SSM and scalar
embedding.  Attention batching is bit-identical through the first two eligible
layers (`LLM_QWEN4_BATCH_ATTN_MAX_LAYER=0` and `=2`, `rel_l2=0`).  Allowing the
third eligible layer (`=3`) immediately produces `rel_l2=7.0416e-2`
(`max_abs` about `0.987`), although the per-token and batched argmax still
match.  Enabling the experimental batched SSM path made the result worse
(`rel_l2=1.2619e-1`).  This localizes the remaining quality loss to the
batched-attention body, not SSM, embedding, KV storage, or MoE.  Partial
attention batching therefore remains diagnostic-only.  As a guardrail, when
`LLM_QWEN4_BATCH=1` is explicitly requested and no layer cap is supplied, the
runner now defaults to the verified prefix (`max_layer=2`); a wider experiment
still requires an explicit `LLM_QWEN4_BATCH_ATTN_MAX_LAYER` value.  The new
default was rechecked at 128 tokens: `rel_l2=0`, matching argmax, and 18.15
prefill tok/s on the RX 9070 XT (the 64-token control was 19.55 tok/s).

An opt-in `LLM_ATTN_DECODE_I8_GQA8=1` kernel was added to share K/V work across
Qwen4's 12:1 query/KV grouping.  It is numerically exact on the 32-token and
256-token controls (matching sequence hashes), but measured `25.6 tok/s` and
`25.6 tok/s` decode respectively, versus `42.7` and `52.4 tok/s` for the
existing scalar I8 kernel.  The extra synchronization outweighs the saved
loads on gfx1201, so the optimized kernel remains diagnostic-only.

The exact delayed-refill interval was rechecked at 256K with the 6-GiB cache.
An 8-token sample briefly reached `26.15 tok/s` at interval four, but a
32-token repeat measured `26.29 tok/s` versus `44.35 tok/s` for interval two
under the same run shape (the card has substantial clock/thermal variance).
Both sequences matched; interval two remains the conservative default until a
larger repeated sample justifies changing it.

This choice was rechecked after enabling direct-GPU cold misses (the current
codex-server 16-GiB policy) with the validated 6.1-GiB LRU cache and 128 exact
decode tokens.  Interval one measured `8.69 tok/s` decode and `8.40 tok/s`
end-to-end; interval two measured `9.17` and `8.82`; interval four measured
`9.09` and `8.77`.  All three runs returned sequence hash
`7a90d592cf2b8cfa`, so interval two remains the fastest quality-equivalent
default on the RX 9070 XT.

The explicit approximate decode route was then tested on the same 256K coding
request.  Its old 32-token refresh reached `44.2 tok/s` but produced repeated
malformed code.  A four-token exact refresh returned the complete coherent
function at `28.25 tok/s`; the six-token cadence also returned the complete
function twice, at `34.54` and `40.40 tok/s` decode.  The codex launcher
remains exact by default; when approximate mode is explicitly enabled it now
selects refresh interval six unless the caller overrides it.
Refresh seven also returned the same complete function at `34.29 tok/s`; the
small cadence difference is within run variance, so six remains the documented
default.  Lowering the cold-route threshold from 0.20 to 0.10 produced the
same output and essentially the same `34.32 tok/s`, indicating that the
selected routes in this control are already above both thresholds.
The six-token profile was also requested with a 64-token output budget at
256K; the model stopped naturally after the same 24-token complete function,
with `34.08 tok/s` decode and no malformed suffix.  This confirms the refresh
cadence does not force a fixed short completion.

The launcher keeps `QWEN38_VRAM_PROFILE=16g` as the safe RX 9070 XT default.
For a larger single GPU, `QWEN38_VRAM_PROFILE=24g` selects a 12-GiB expert
cache/BMAX=3072 baseline, while `32g` selects an 18-GiB cache/BMAX=4096
baseline.  These are configuration starting points, not performance claims;
the target card and driver still need benchmark validation.  The same profile
selection is honored by `run_qwen38_codex_server_rocm.sh`; only the 16-GiB
profile automatically switches to I8 KV at 128K+ so larger cards retain the
F16 quality default.

The exact 16-GiB production cache policy is LRU (`LLM_MOE_LFU_CACHE=0`); LFU is
opt-in for A/B experiments.  The flash launcher now matches the server launcher
and defaults to LRU because LFU reduced decode throughput in the 256K control.

A fresh full-allocation smoke run with `max_seq=262144`, I8 KV, a 5.9-GiB
expert cache, BMAX=512, the copy pipeline, and graph/plan reservations
disabled is now the repository baseline.  It keeps the same end-to-end quality
gate while retaining more resident experts than the stable 5.0-GiB fallback.
The corrected benchmark harness now passes `-n` as the prefill length (rather
than accidentally using the decode count), uses `--gpu-only-bench`, and records
the 128+32 control above.  The matching 256K HTTP server reached
`JSONL backend ready`; a short 23-token coding request returned the complete
one-line C function with `5.32` prefill and `5.34` decode tok/s end to end.
The same regression is now available as `make -C rdna4/llm bench-256k` (or
`rdna4/llm/bench_qwen38_256k.sh`); it keeps its log under `rdna4/llm/tmp/` and
fails before model loading when KFD/render access is missing.  Set
`QWEN38_EXPECT_HASH=<greedy sequence hash>` to add an optional output-quality
gate to the smoke run.
The flash launcher now pins `OMP_NUM_THREADS=16` by default, matching the
server launcher and the longer-context measurements; callers can still set an
explicit worker count for A/B tuning.

### SSM batching isolation

The Q6_K batched-projection path was isolated with scalar attention, scalar
recurrence, parity `expf`, scalar convolution, and a 4-GiB expert cache.  A
single recurrent layer measured `rel_l2=2.48e-3`; three layers measured
`2.85e-3`, but the minimum 8-token grouped tile reached `4.20e-2`, and the
64-token tile reached `1.25e-1`.  Argmax still matched, but the error grows
with the recurrent sequence and is not suitable for quality-sensitive serving.
This confirms that the remaining SSM issue is accumulated hidden-state drift,
not a one-off attention-layer failure; the Q6_K SSM path remains opt-in.  The
parity flag only changes the recurrence primitive when the default warp kernel
is retained; forcing the direct scalar-order kernel with
`LLM_SSM_BATCH_WARP=0` was worse (`rel_l2=1.57e-1` on the same 64-token
control), so it is not a production fallback.

Both launchers support `QWEN38_DRY_RUN=1` to print the resolved VRAM/KV/cache/
BMAX profile without requiring `/dev/kfd`; for example, this verifies the
16-GiB 256K selection before granting ROCm access:

```sh
QWEN38_DRY_RUN=1 QWEN38_VRAM_PROFILE=16g \
  rdna4/llm/run_qwen38_flash_next_rocm.sh -s 262144
```

The standalone `test_hip_llm` binary performs the same `/dev/kfd` check before
loading the CPU shadow model, so direct benchmark invocations fail quickly and
consistently when the AMD device is not passed through.

The clean exclusive-GPU 256K smoke was rerun with the production scalar profile
(`I8 KV`, `BMAX=512`, 5.68-GiB expert cache, direct copies, graphs/plan warmup
off, GPU top-k on).  With a 512-token prompt and 8 greedy decode tokens it
completed with `15.67 tok/s` prefill, `17.14 tok/s` decode, and `15.69 tok/s`
end-to-end; the run returned `PASS` and sequence hash
`a2d4f49620d5b663`.  The benchmark is GPU-only (so it is a throughput/stability
smoke, not a CPU-reference parity comparison); the full log is
`tmp/256k_clean_p512.log`.

For sub-32K tuning, the benchmark harness now publishes the Qwen4 request
length before prefill, matching the HTTP server's stateful-batch contract.
Without that setter, every standalone benchmark was conservatively routed to
the scalar fallback even when `LLM_QWEN4_BATCH=1` was requested.

On the RX 9070 XT at an 8K allocation, the experimental full batched profile
(`LLM_QWEN4_BATCH_SSM=1`, `LLM_QWEN4_BATCH_ATTN_MAX_LAYER=47`, native batch QKV,
GPU prefill top-k, 2,048-row tile) measured `181.27 tok/s` prefill and
`19.94 tok/s` exact decode for a 2,048-token prompt plus 64 tokens.  It kept
the simple-control hash, but the coding-prompt scalar-vs-batched comparison
reported `rel_l2=1.32e-1`; it is therefore diagnostic only and is not enabled
by a serving launcher.

The lower-drift all-attention candidate (scalar SSM, exact GPU router top-k)
measured `36.48 tok/s` prefill and `26.55 tok/s` exact decode at 1,024 tokens;
its simple-control hash was preserved, but the coding comparison still showed
`rel_l2=4.88e-2`.  The fully quality-safe scalar route remains below 100
prefill tok/s.  Enabling batched SSM recurrence reaches the throughput target
but changes coding logits, so these switches remain opt-in pending a parity
implementation.

Exact device-resident cache routing (`LLM_QWEN4_DEVICE_CACHE=1`) was also
checked at 8K.  It preserved the simple-control hash and reached `25.74`
decode tok/s, slightly below the ordinary staged-copy route (`26.55`), so it
is not promoted as the default.  BAR-mapped misses were slower still.

The strongest exact control profile so far combines Q6_K SSM batch projection,
scalar-order recurrence parity (`LLM_SSM_BATCH_PARITY=1`, warp path off), and
CPU evaluation of every decode cold miss (`LLM_MOE_CPU_DECODE_MISSES=1`,
`LLM_MOE_CPU_MIN_WEIGHT=0`).  At 8K with a 1,024-token prompt it measured
`169.62` prefill / `30.50` decode / `133.73` end-to-end tok/s and preserved
hash `6d67721190bdaa83`.  This is a control result: on the UTF-8 coding prompt
the same profile measured `161.78` prefill / `19.01` decode and produced hash
`1d0f5941349fc776` versus the scalar reference
`5fdecd9bf8f28901`.  Batched attention/SSM parity therefore still needs a
prompt-diverse numerical pass before it can be a quality-safe serving default.
Repeating with GPU top-k enabled produced hashes `9bdef0951b53ce8b` and
`6d67721190bdaa83` with decode rates from `22.75` to `29.38` tok/s.  The
diagnostic script therefore forces host top-k for deterministic routing; host
top-k repeats preserved `6d67721190bdaa83` and measured `29.55`–`30.30` decode
tok/s.  The coding-prompt hash still differs, so this remains an experimental
target profile rather than a quality-safe serving default.

The corrected `bench_qwen38_sub32_target.sh` (host top-k, Q6_K parity SSM,
exact CPU misses) repeated at `140.28 tok/s` prefill, `29.01 tok/s` decode, and
`114.45 tok/s` end-to-end with the stable control hash.  `OMP_NUM_THREADS=12`
was previously indistinguishable, but the current rebuilt runner produced an
unstable `18.19 tok/s` decode and a different hash with 12 threads.  The
default 16-thread setting remains the reproducible choice; 32 threads caused
oversubscription and dropped decode to `10.33 tok/s`.  The remaining few
percent to a repeatable 30 tok/s is currently the CPU cold-miss/refill cost.

After rebuilding, the restored default profile measured `139.49` prefill /
`29.14` decode tok/s with the stable hash.  An explicit exact GPU-top-k run
measured `140.15` / `29.10` with the same hash, so GPU top-k is exposed as an
experiment but is not promoted as a speed optimization.

An intermediate CPU-miss threshold (`LLM_MOE_CPU_MIN_WEIGHT=0.1`) reached
`34.43` decode tok/s on the simple prompt but fell to `10.98` tok/s and changed
the coding-prompt hash; it remains diagnostic-only.  Raising the cache budget
from 8,500 to 8,550 MiB also triggered a hipBLASLt kernel-symbol failure, so
8,500 MiB remains the maximum stable budget on this card.

A host-thread sweep reached `29.56` and `29.47` decode tok/s on two 17-thread
runs with the stable hash.  The target script now defaults to 17 threads;
18-thread repeats changed the route hash and fell to `23.65` tok/s, so higher
thread counts are not promoted.

An optional `LLM_MOE_CPU_MIN_WEIGHT=0.2` control reduced the same simple
prompt to fewer CPU misses and reached `37.22` decode tok/s while retaining its
control hash.  It is not quality-safe in general: the UTF-8 coding prompt fell
to `7.93` decode tok/s and produced hash `bb602f2deee2e5e7`, so the diagnostic
script keeps the exact `0.0` threshold by default.
