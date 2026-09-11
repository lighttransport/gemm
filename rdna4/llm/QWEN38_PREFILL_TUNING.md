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

The server launcher retains a 512-row scalar prefill tile, a 7200 MiB expert
cache, ungrouped prefill, and disabled LFU/copy pipeline. With its 65536-token context
allocation, the same benchmark measured **120.35 tok/s prefill and 32.91 tok/s
decode**. This was a runner benchmark with the server's settings, not an HTTP
latency measurement or a full 65536-token prompt.

The HTTP launcher now also enables the parity-safe depth-weighted prefill cache
table by default (`LLM_QWEN4_PREFILL_CACHE_BALANCE=1`); `=0` remains an
explicit decode-focused opt-out.

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
before per-group scales/zero-points and alignment.  The runner now has an
explicit scaled-E4M3 FP8 store/dequant attention path; FP4 remains
unimplemented and is rejected rather than silently allocating another format.
The loader prints the F16/FP8/FP4 estimates at startup so implementations can
be checked against the actual layer geometry. FP8 is explicit-only pending
long-context serving qualification; use `none`/`f16` for the quality-safe
default or `i8` for the established capacity experiment.

The experimental scaled-I8 layout uses eight 32-channel scales per KV head;
at 256K this adds about 0.18 GiB, for roughly 3.18 GiB total KV-plus-scales.
However, a fresh RX 9070 XT probe at `-s 4096` hit an HSA memory fault in
`kv_cache_store_i8_devp` before the first benchmark token and left a stale KFD
context. Inspection found that the fixed 256-thread launch let inactive warps
write zero scales beyond the valid 32-channel groups; both single-token and
batched stores now guard the scale write with `group < groups`.
Static inspection found a second independent issue in that probe: exact MTP's
sidecar layer was indexing the target I8 scale table even though its sidecar KV
cache is F16. The runtime now restricts I8 store/attention to the target trunk
and keeps the sidecar on F16. A clean elevated GPU rerun after both fixes
passed: target-only I8 reached `PASS`, and exact-MTP + I8 at 4K reached
`PASS` with sequence hash `454146399ff97e88`.
The 256K allocation probe now succeeds on the RX 9070 XT: with the routed
expert cache disabled, the loader reported 9.65 GiB free before KV allocation,
reserved 3.000 GiB of I8 K/V, and completed weight loading.  The same max-seq
profile loaded with BMAX=512 and the batched prefill path enabled.  This proves
capacity and startup stability; it is not a claim that a 256K prompt reaches
the short-context 200+ tok/s rate.

`run_qwen38_flash_next_rocm.sh` keeps F16 for ordinary contexts and leaves
the I8 choice explicit for 16-GiB requests at or above 131K tokens. The larger
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
The same scalar I8 control with the full 262,144-token allocation now produced
hash `44bc8ad473cbe606` for the one-token capacity smoke and returned `PASS`,
with 370 MiB VRAM free at peak.

The longer exact-MTP coding probe (28-token prompt) returned `PASS` and
coherent C/LRU-cache text. At 16 generated tokens I8/F16 measured 8.20/8.67
decode tok/s and already had different hashes; at 32 tokens they measured
8.05/8.49 tok/s with hashes `b555c433cfd91cf5` and `165eb389c0515b9f`. At 64
tokens I8 measured 8.34 decode / 7.38 end-to-end tok/s (hash
`994e0df7ffcf7854`) and F16 measured 8.36 / 7.34 (hash `4b8937cd7db0e7a7`).
Thus I8 is validated for short exact parity and coherent output, but is not
claimed bit-identical beyond the short horizon.

The new FP8 path passed target-only and exact-MTP 4K smoke tests. On the same
28-token/64-output coding probe it reached 8.61 decode and 7.48 end-to-end
tok/s, returned `PASS`, and retained the F16 control hash `4b8937cd7db0e7a7`.
The E4M3 encoder now uses synchronized round-to-nearest-even mantissa packing
in both the host reference helper and HIPRTC kernel, avoiding systematic
truncation bias; the short exact-MTP hash remains unchanged.
This is an initial quality result; FP8 remains explicit-only until a longer
context sweep is complete.

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

The reproducible `bench_qwen38_256k.sh` smoke was rerun after the I8 store
fixes with the full 262,144-token allocation and an 8-token warm prompt plus
8-token decode: prefill 4.79 tok/s, decode 5.46 tok/s, and end-to-end 5.10
tok/s, `PASS`. This is a short-request stability baseline at 256K capacity,
not a claim of 256K prompt throughput or 8K-output quality.

The same 256K capacity smoke with explicit FP8 also returned `PASS`: 3.000 GiB
FP8 KV, 8.72 decode tok/s, 4.89 end-to-end tok/s, and 370 MiB free at peak.
FP8 therefore provides the expected capacity profile, but this remains a
short-request smoke rather than full-context generation validation.

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
loads on gfx1201, so the optimized kernel remains diagnostic-only. A fresh
exact-MTP 64-token coding control preserved the I8 hash `994e0df7ffcf7854` and
measured 8.39 tok/s versus 8.34 tok/s for scalar I8; the marginal gain is
workload-sensitive and does not justify enabling it by default.

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

For persistent access, the surrounding container/session must be started with
the device nodes and render/video groups passed through (for example
`--device=/dev/kfd --device=/dev/dri --group-add video --group-add render`).
Creating nodes inside one elevated command namespace is only a temporary test
workaround; those nodes are not visible to later ordinary commands.

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

For the true 4K multi-chunk path, disabling exact CPU cold misses moved work to
the GPU but reduced decode from `24.02` to `22.61` tok/s (same hash), confirming
that the remaining decode cost is expert-cache churn/H2D rather than CPU
threading alone.  The script now exposes both CPU-miss gates for controlled
comparisons while retaining exact CPU misses by default.

Reducing BMAX to 512 did not provide a larger stable cache: 8.7–9.0 GiB cache
attempts failed with allocation/segmentation faults on gfx1201.  The stable
8.5 GiB/BMAX=1024 combination remains the practical 16-GiB limit.

At 4K, CPU miss thresholds of 0.2–0.5 removed the 535 ms CPU refill time but
only reached about `28.3` decode tok/s because GPU cache misses still dominate.
GPU top-k and mapped-host misses were also slower (`24.33` and `23.31` tok/s),
so neither is promoted.

A 128-token steady-state decode sample after the 4K prefill measured `28.47`
tok/s, confirming the deficit persists beyond the first-token handoff.

Disabling the MoE copy pipeline and varying stream slots (1/2/4) was neutral:
the 128-token decode stayed at `28.55`–`28.58` tok/s with the same hash.  The
copy stream is therefore not the remaining limiter.

The broader opt-in approximate decode (`LLM_QWEN4_APPROX_DECODE=1`) reached
`32.59` tok/s for a 128-token simple decode after 4K prefill and preserved that
simple hash.  It is not quality-safe: on the coding prompt its hash was
`cd085d716003cbcd` versus the exact `b63472f63a723cc9`.  Keep it explicitly
opt-in; the exact default remains below 30 tok/s.

The CPU-prefill low-count experiment (`LLM_MOE_CPU_PREFILL_MAX_COUNT=4`) did
not activate for the batched path: CPU time remained zero and H2D traffic was
unchanged.  LFU retention was also counterproductive at 4K (`22.35` tok/s,
888 GiB H2D versus 643 GiB for LRU), so LRU remains the prefill policy.

The optional staged expert-bank path was also checked.  At the 8.5-GiB cache
budget its staging allocation was unavailable and transparently fell back to
cache-only prefill; reducing the cache to 7 GiB failed weight allocation before
the stage could be used.  The script exposes `QWEN38_PREFILL_STAGING=1` for
future cards with additional VRAM, but it is not a 16-GiB production setting.

The important long-request gate was then corrected: `forward_batch_logits()`
intentionally falls back to scalar for requests longer than BMAX unless
`LLM_QWEN4_BATCH_MULTI_CHUNK_FORCE=1` is set.  The target script now enables
that explicit stateful multi-chunk path by default.  At 4,096 tokens it measured
`158.71` prefill tok/s with `53.30 GiB` H2D and the same stable hash, versus
`25.34` tok/s and `643 GiB` H2D in the scalar fallback.  Decode was `24.02`
tok/s, so long-context decode still needs separate cache/refill work.

Long-context validation exposed a separate limit: with the current exact
Q6_K/SSM path, a 4,096-token prefill measured only `25.34` tok/s at BMAX=1024
and `22.39` tok/s at BMAX=2048 (7-GiB cache).  The latter uploaded `875 GiB`
of expert data with an 84.6% cache-hit rate.  A 5-GiB BMAX=2048 cache measured
`18.16` tok/s, and disabling cache balancing or enabling grouped prefill did not
improve it.  BMAX, chunk size, cache balance, and grouped-prefill are now
explicit script overrides for further investigation; the safe 1K target
profile remains unchanged.

An optional `LLM_MOE_CPU_MIN_WEIGHT=0.2` control reduced the same simple
prompt to fewer CPU misses and reached `37.22` decode tok/s while retaining its
control hash.  It is not quality-safe in general: the UTF-8 coding prompt fell
to `7.93` decode tok/s and produced hash `bb602f2deee2e5e7`, so the diagnostic
script keeps the exact `0.0` threshold by default.

The exact CPU cold-miss dot-product loops now traverse selected experts
expert-major rather than interleaving every expert on each output row. This
preserves the per-row arithmetic and greedy hash, while improving a matched
4K/64 sample from `23.96` to `24.50` decode tok/s; 128-token runs remain
cache-variance limited.

Exact-mode A/B tests can use `LLM_QWEN4_EXACT_CPU_MIN_WEIGHT=<threshold>`
without changing the default. At 4K, thresholds through `0.5` retained the
simple greedy hash but reached only `28.3` tok/s; `0.8` did not improve it.
No threshold is promoted without coding-prompt parity.

The exact CPU-miss path now skips redundant 4-byte device cache-map updates;
its selected expert kernels consume host slot IDs directly. Cache-copy events
remain unchanged. This is an overhead cleanup only: matched 4K samples were
within run-to-run cache variance and did not establish a 30 tok/s gain.

The benchmark harness now selects `LLM_QWEN4_BATCH_MULTI_CHUNK_FORCE=0` for
requests at or below BMAX and `1` only when the request exceeds BMAX. This
avoids a measurable regression at exactly 1,024 tokens (`29.26` decode tok/s,
stable hash) while retaining the true 4K multi-chunk path (`160.23` prefill
tok/s, stable hash).

The remaining exact-decode gap was rechecked on the 16-GiB RX 9070 XT. An
OpenMP sweep at 4K (24/32/48 threads) reached `24.95/13.29/21.94` tok/s;
the 24-thread result is the best of that matched run and retained the simple
control hash. A 100-MiB cache increase (8,600 MiB total) reached `25.33`
tok/s, so the practical 8.5-GiB cache ceiling remains the right default. The
optional one-warp resident-expert kernel and decode-oriented layer allocation
were also slower (`24.82` and `23.92` tok/s respectively).

Increasing the prefill staging pool from 512 to 768 MiB was allocatable but
did not help: the 4K sample measured `159.95` prefill and `24.29` decode
tok/s. The benchmark now exposes `LLM_QWEN4_PREFILL_STAGE_MB` for cards with
more VRAM, but 512 MiB remains the 16-GiB setting. Per-layer Qwen HIP graph
replay was faster in one simple run, but changed the coding-prompt sequence
hash; it remains disabled for exact production decoding.

A prototype K-parallel Q5_1 down kernel (one warp per selected expert) was
also rejected. It expanded the grid from output-row tiles to one block per
row and measured only `10.64` decode tok/s at 4K, with a changed sequence
hash. The existing row-parallel kernel therefore remains the exact path.

An exact GPU top-k A/B was also rerun on a 128-token/16-output scalar control.
It preserved the host-router sequence hash (`75aa7fd4f21cfee1`) but reduced
decode from `21.80` to `21.08 tok/s`; the extra device route synchronization
was not removed in this configuration. `LLM_QWEN4_EXACT_GPU_TOPK=1` therefore
remains an experiment rather than a production setting.

Validated scaled-I8 KV was also used to reclaim VRAM for the expert cache.
At 8K context, an 8.7-GiB cache loaded successfully but reached only
`24.47` decode tok/s (versus the F16-cache control near `25.3`), with the
same simple hash. A 9.2-GiB request reached weight-loading failure in the
hipBLASLt setup, so I8 does not currently buy a stable decode improvement.

### Opt-in Q8_K routed-kernel prototype

The exact decoder has a diagnostic `LLM_QWEN4_EXACT_Q8K=1` route for the
Q4_K/Q4_K gate-up plus Q5_1 down expert layout. It stages the shared input in
canonical 292-byte Q8_K blocks, stages each SiLU output in 40-byte Q8_1
blocks, and evaluates selected experts with dedicated HIPRTC kernels. The
existing F32 routed kernels remain the default until a matched coding hash
and parity run promotes this path. Scratch allocation is independent of the
expert cache and is only a few KiB for the Qwen3.8 dimensions. Because the
prototype previously triggered a gfx1201 reset on the current driver, and
launching it now additionally requires `LLM_QWEN4_EXACT_Q8K_UNSAFE=1`; this
second gate is intentional and should only be used under a disposable
kernel-debug session. The block-sum race in the Q8_K staging kernel is fixed,
but the path is still numerically non-parity-safe.

A post-fix short smoke compiled and completed successfully (`Result: PASS`)
without a reset, but still produced hash `9a7b3000c5591075` versus the F32
control `96b18100c22040bd` (and was slower, `4.53` versus `6.24` decode
tok/s). This is expected for a first quantized prototype; the switch therefore
remains diagnostic-only pending a same-process parity comparison and coding
quality gate.

The approximate coding sweep is available as `make -C rdna4/llm
approx-coherence`. It tests 1K/4K/8K prefill with refresh intervals 4/6/8,
captures generated text, checks balanced C output, and runs `gcc -fsyntax-only`
on the extracted function. Approximate mode remains explicitly opt-in.

Short-context HTTP coding A/B testing found that full-depth resident
approximation can corrupt conditional syntax even at refresh six. Restricting
approximation to layers 24--47 while keeping the first 24 layers exact returned
the complete compilable `clamp` function at 4K; the measured decode rate was
`21.7 tok/s` versus roughly `31 tok/s` for full-depth approximation. The
launcher selects this layer-range quality profile automatically below 32K;
`LLM_QWEN4_DEVICE_REFRESH_START_LAYER=0` restores the faster diagnostic mode.

The standalone GPU benchmark still has a stability boundary for a single
~4K-prefill dispatch on gfx1201. The supported serving shape is explicit
streamed chunks (512 tokens in the diagnostic); with `LLM_QWEN4_BATCH=0`, the
benchmark bypasses the multi-chunk batch dispatcher and advances the stream
token by token, providing a correctness-first fallback for long contexts.
The scalar streamed path has completed 4K requests with `Result: PASS`; only
the unstreamed/experimental dispatcher remains invalid. Set
`LLM_BENCH_STREAM_CHUNK=0` only for an explicit single-dispatch A/B test.
On a clean RX 9070 XT this fallback completed a 4096-token request with
`Result: PASS` at `14.72 tok/s` (278.3 s total). It is stable but not a
performance solution: the run moved about 1.53 TiB of expert data, so reducing
per-token cold-expert transfers is the next optimization target.

Repeating the same scalar stream with the full 8.5-GiB expert cache raised the
4K prefill rate to `19.80 tok/s` (206.8 s), with an 84.9% cache-hit rate and
861.9 GiB H2D. At 512 tokens, the corresponding comparison was `18.72` versus
`13.94 tok/s` and `115.5` versus `201.8 GiB` H2D. The 8.5-GiB cache remains the
preferred 16-GiB scalar-stream setting; further gains require reducing cold
route churn or retaining experts across requests.
The existing prefill-staging switch was neutral on the 512-token control
(`18.80 tok/s`, identical `115.5 GiB` H2D), so it is not promoted as an
additional optimization for this scalar path.
With 16 generated tokens on the same 512-token request, decode measured
`19.11 tok/s` and end-to-end measured `18.63 tok/s`; the run also returned
`Result: PASS`.

The explicit LFU policy was slower on the same 512-token control (`18.36
tok/s`, 83.1% hit, 120.1 GiB H2D), confirming LRU as the scalar-prefill
default.

Delayed expert-cache refill remains an opt-in exact-decode experiment in the
flash launcher (`LLM_QWEN4_DELAYED_CACHE=0` by default; set `=1` for an A/B
run). The initial matched 64-token/32-output test appeared to improve decode
from `21.27` to `22.49 tok/s`, but the result did not reproduce on longer
streams: at 128 output tokens the delayed/control rates were `22.47`/`24.27
tok/s`, with identical hash `8afbeab2765b52ab`. A coding prompt likewise kept
the same hash (`47b4dceb09c65616`). Keep the switch opt-in until a sustained
workload shows a repeatable gain. The refill remains disabled automatically for
active MTP sidecars and unsafe direct-BAR combinations.

The scalar streamed fallback now avoids the vocabulary projection for every
intermediate prompt token and materializes logits only on the final token. The
512-token control improved from `18.72` to `19.04 tok/s` with unchanged cache
hits/H2D (`Result: PASS`); this optimization is safe for the long-context
fallback as well.
The 4K control improved from `19.80` to `20.21 tok/s` (202.7 s total), again
with 84.9% hit and 861.9 GiB H2D, and returned `Result: PASS`.

With the depth-weighted prefill cache table enabled, a fresh 512-token scalar
control reached `21.64 tok/s` (23.659 s), with an 87.9% hit rate and 85.99 GiB
H2D; it returned `Result: PASS`. This is the new fast-prefill launcher default
(`LLM_QWEN4_PREFILL_CACHE_BALANCE=0` remains an explicit opt-out). A concurrent
4K validation did not reach a benchmark footer after the device became stuck
inside its first streamed chunk, so no 4K speed or quality claim is made for
the rebalanced table yet.

An opt-in scalar prefill copy-stream pipeline then overlapped cold-expert H2D
with resident-expert work on the same 512-token control. It reached
`29.27 tok/s` (17.491 s), with 88.2% cache hit and 83.80 GiB H2D, versus
`21.64 tok/s` without the pipeline. A matching 16-token coding decode control
preserved sequence hash `a895417764461896` exactly (pipeline decode 18.11
tok/s versus 17.56 tok/s without it; both `Result: PASS`). The fast-prefill
launcher keeps this behind explicit `LLM_MOE_COPY_PIPELINE=1` and
`LLM_QWEN4_PREFILL_COPY_PIPELINE=1` gates. The runner additionally refuses
the prefill gate when the published request length exceeds the configured
ceiling. The validated 2K window completes on gfx1201. Direct copies therefore remain
the stable large-request default for larger or unbounded requests.

For a controlled hardware sweep, `LLM_QWEN4_PREFILL_COPY_PIPELINE_MAX_TOKENS`
raises that request-length ceiling (bounded to 4096 by the runner), for example
`LLM_QWEN4_PREFILL_COPY_PIPELINE=1
LLM_QWEN4_PREFILL_COPY_PIPELINE_MAX_TOKENS=1024`. The current launcher and
runner default ceiling is 2,048 tokens; 1K remains a valid conservative
override.
The pipeline remains opt-in and direct copies remain the default; the current
validated explicit ceiling is 2,048 tokens.

The bounded ceiling was exercised at 1,024 tokens on the RX 9070 XT. With
`LLM_QWEN4_PREFILL_COPY_PIPELINE=1`,
`LLM_QWEN4_PREFILL_COPY_PIPELINE_MAX_TOKENS=1024`, and a 1K streamed chunk,
scalar prefill reached `31.45 tok/s` and decode `32.95 tok/s`; the direct-copy
control figures from that historical row are not comparable because the
benchmark harness then forced the pipeline for both cases. The pipeline run
returned simple-control hash `a2d4f49620d5b663` and `Result: PASS`; a corrected
direct-copy comparison is now required for a clean 1K speed claim. The copy
pipeline remains opt-in while its bounded ceiling is 1,024 tokens.

A clean-card pipeline repeat reached the benchmark footer and passed (`28.15`
prefill / `31.25` decode tok/s, hash `1b67330f97f98543`). The corrected direct
control reached `27.31` prefill / `31.12` decode tok/s with the same hash and
`Result: PASS`, establishing a repeatable roughly 3.1% prefill gain.

The corrected benchmark harness was then used for a true 2,048-token A/B with
a 5.0-GiB expert cache. The copy pipeline completed in `102.37 s` (`20.01
tok/s`, `Result: PASS`), while the direct-copy control completed in `107.58 s`
(`19.04 tok/s`, `Result: PASS`), a roughly 5.1% prefill improvement. The
earlier apparent stall was an observation timeout, and the earlier direct-copy
controls were invalid because the harness had forced `LLM_MOE_COPY_PIPELINE=1`.
The harness now honors an explicit `=0`; 2K is still opt-in pending a decoded
quality/hash run, and 4K remains unvalidated for overlap.

A corrected 4K pipeline run with the same 5.0-GiB cache did not reach a footer
within a bounded 3-minute diagnostic window and was terminated after the
runner remained resident. No 4K throughput or quality claim is made from that
attempt; the stable serving path remains direct copies for 4K requests.
The harness now also exposes `LLM_BENCH_STREAM_PUBLISH_CHUNK=1`, which publishes
each 1K chunk separately so the 1K guard can be exercised across a long
request. A four-chunk 4K run still exceeded the 5-minute diagnostic window on
gfx1201, so this mode remains experimental rather than a serving default.
The stdio/HTTP server exposes the corresponding opt-in
`LLM_QWEN4_PREFILL_COPY_PIPELINE_PUBLISH_CHUNK=1`; it is disabled by default
and is intended only for bounded chunking experiments.

The server-shaped 2K run (four 512-token chunks, 5.0-GiB cache, chunk
publication enabled) completed at `20.00 tok/s` prefill and `25.30 tok/s`
decode, with end-to-end `20.01 tok/s`, hash `a2d4f49620d5b663`, and `PASS`.
This is the recommended bounded 2K experiment; larger requests remain
unvalidated.
Splitting that request into two 1,024-token scalar chunks while publishing the
per-chunk ceiling reproduced the stall as well, so the failure is not solely a
single oversized tile; the per-chunk publication experiment was removed.

A minimal-prompt 4K end-to-end control with the 8.5-GiB cache completed with
`19.84 tok/s` prefill, `19.28 tok/s` decode, and `19.83 tok/s` end-to-end
(`Result: PASS`).

The experimental stateful batched dispatcher was also measured at 512 tokens
with a 6-GiB cache: `15.89 tok/s`, 76.2% hit rate, and 169.0 GiB H2D. It is
slower and less resident than the scalar 8.5-GiB path, so it remains disabled
for the stable profile even below the 4K crash boundary. With the full
8.5-GiB cache it reaches `19.22 tok/s` with the same 83.7% hit rate and
115.5 GiB H2D—only a small ~2.7% gain over scalar—so the large-request guard
still selects scalar streaming while short requests may retain batching.

## VRAM measurement

The HIP runner now exposes `hip_llm_get_vram_stats()`.  It reports current
free/total VRAM and a high-water `peak_used_bytes` sampled after completed
forward operations; this is the value to record alongside prefill/decode and
end-to-end tok/s.  The stdio/HTTP server prints the same values as
`free=... total=... peak-used=...` after each request.  Peak usage is observed
runtime telemetry, not the requested expert-cache budget, and remains zero
until the first completed forward.
The server footer also reports `end-to-end=... prompt-added + ... generated`
tok/s, so cached-prefix requests are not confused with decode-only throughput.

A short RX 9070 XT smoke benchmark (`8` prefill + `2` decode, 512-MiB
expert-cache request) reported `8,194 / 16,304 MiB` free and `8,110 MiB`
peak used, with an end-to-end `4.37 tok/s` and `Result: PASS`.

The current repository 256K smoke profile (scaled-I8 KV, 5.9-GiB expert
cache, 8 prefill + 8 decode) completed with `4.97` prefill tok/s, `5.64`
decode tok/s, and `5.29` end-to-end tok/s. It preserved hash
`5e003cee3db52848`, reported `15,960 MiB` peak used / `344 MiB` free, and
returned `Result: PASS`. This is an allocation/end-to-end smoke baseline, not
a 256K-token generation benchmark.

Matched 256K cache-budget sweep (same 8+8 smoke, scaled-I8 KV) shows the
resident-cache trade-off while preserving the same greedy hash:

| Expert cache | Prefill tok/s | Decode tok/s | End-to-end tok/s | Peak VRAM |
| ---: | ---: | ---: | ---: | ---: |
| 512 MiB | 4.32 | 4.48 | 4.40 | 11,566 MiB |
| 2,048 MiB | 4.66 | 4.83 | 4.75 | 12,152 MiB |
| 5,900 MiB | 4.99 | 5.67 | 5.31 | 15,960 MiB |

The 5.9-GiB budget is the fastest measured 16-GiB point but leaves only
344 MiB free; smaller budgets are safer for concurrent allocations.

For selected-attention diagnostics, the exact device selector and warp-per-
head attention kernel are both opt-in:

```sh
make -C rdna4/llm qsa-256k-device-warp
```

The 256K gate passes score and selected-output parity, but no 2K/4K
throughput claim is made yet; those runs require a bounded runner job because
large prefill requests can outlive an external observation timeout.

A persistent-PTY A/B at 2,056 tokens (scaled-I8 KV, 2,048-MiB expert cache,
device selector enabled) completed cleanly. The warp attention path measured
`10.17` prefill / `9.94` decode / `10.17` end-to-end tok/s; the 256-thread
control measured `10.17` / `9.84` / `10.17`. Both returned hash
`9ac18100c593e9bb`, `PASS`, and `8,898 MiB` peak VRAM. The warp kernel is
therefore parity-safe but only a ~1% decode improvement in this MoE-dominated
profile; it remains opt-in.

Increasing only the expert cache from 2,048 to 5,900 MiB on the same
2,056-token run changed the bottleneck materially: prefill rose from `10.17`
to `16.68 tok/s`, decode from `9.94` to `15.48 tok/s`, and end-to-end from
`10.17` to `16.68 tok/s`. The greedy hash stayed `9ac18100c593e9bb`; cache
hit rate rose from `50.4%` to `76.5%` and H2D fell from `1,415.03` to
`668.92 GiB`. Peak VRAM was `12,734 MiB` with `3,752 MiB` free, making this
the preferred 4K selected-attention diagnostic profile on the 16-GiB card.

A second controlled 2,056-token streamed run (BMAX=1, F16 KV, device
selector plus warp attention, two decode tokens) confirms the fast-prefill
cache choice used by `QWEN38_FAST_PREFILL=1`. Raising the resident expert
cache from 5,900 to 7,800 MiB improved prefill from `21.29` to `26.09`
tok/s and decode from `24.87` to `25.82` tok/s. Both runs retained the exact
greedy hash `9ac18100c593e9bb` and returned `PASS`; cache hit rose from
`80.3%` to `87.2%` and prefill H2D fell from `560.56` to `364.94 GiB`.
The trade-off is peak VRAM: `12,678 MiB` (3,780 MiB free) at 5,900 MiB
versus `14,576 MiB` (1,728 MiB free) at 7,800 MiB. Therefore 7.8 GiB stays
an explicit fast-prefill diagnostic setting, while the 5.9 GiB profile remains
the safer long-context default.

A 32-token decode repeat at 7,800 MiB kept the same prefill rate (`26.07`
tok/s), reached `30.23` decode tok/s and `26.13` end-to-end tok/s, and also
returned `PASS`. Its longer-output hash was `482e10d864607703`; the two-token
smoke's `9ac18100c593e9bb` is expected to differ because the generated length
is part of the hash.

The same 7.8-GiB/2,056-token run with the validated scaled-I8 KV cache used
only `14,374 MiB` peak VRAM versus `14,422 MiB` for F16, but measured
`25.93` prefill / `29.81` decode / `25.99` end-to-end tok/s. It preserved the
same 32-token hash and `PASS`. At 4K, KV compression is therefore primarily a
capacity win; the attention kernel is slightly slower than F16. FP8/FP4 remain
unimplemented runtime formats rather than aliases for this I8 path.

### QSA production-default audit

The exact QSA selector/warp-attention path was rechecked with a matched
2,056-token scalar request, scaled-I8 KV, 5.9-GiB expert cache, and `BMAX=1`.
It preserved the exact coding hash, but forced QSA measured only `0.68`
decode tok/s; the QSA-disabled control measured `14.53` decode tok/s with the
same hash and cache statistics. QSA therefore remains explicit diagnostic
coverage (`LLM_QWEN4_QSA_DEVICE_SELECT=1` and/or
`LLM_QWEN4_QSA_WARP_ATTN=1`) and is no longer selected automatically by the
flash or Codex launchers. The remaining QSA work is kernel/synchronization
optimization followed by a matched 8K+ serving benchmark.

The follow-up fix keeps the index-cache update but bypasses selected attention
when it would remove less than 25% of the context. This avoids the near-dense
2K case that caused the earlier `0.68` tok/s result: the same forced-QSA
benchmark now measures `20.01` prefill / `19.13` decode / `20.01` end-to-end
tok/s, with the unchanged hash `c83dbcf03c2f4d7b` and `PASS`. QSA remains
opt-in for long contexts, where a larger sparsity win must still be measured.

The QSA score kernel also now receives a device-resident rotary-frequency table
created once during exact-mode setup, removing per-score-block `powf` calls.
The 2K forced-QSA control remained numerically identical (`20.02` prefill /
`19.13` decode, hash `c83dbcf03c2f4d7b`, `PASS`), so the change is ready for
long-context profiling without altering the production default.

### Scalar streamed chunk sweep

The validated scalar stream was compared at 1,024 and 2,048-token chunk sizes
using the 5.9-GiB scaled-I8 profile. At 2,056 tokens, 1K measured `20.05`
prefill / `19.14` decode tok/s and 2K measured `20.07` / `19.18`; both kept
hash `c83dbcf03c2f4d7b` and returned `PASS`. A full 4,096-token request split
into two 2K scalar chunks completed at `19.01` prefill / `16.75` decode /
`19.00` end-to-end tok/s, with the same hash and `PASS`. The larger chunk is
therefore correctness-safe but not a material throughput win; the serving
default remains the conservative 512-token stream until overlap improves.

The opt-in prefill copy pipeline was then exercised over the same two 2K
chunks with per-chunk publication enabled. It completed at `19.86` prefill /
`17.40` decode / `19.85` end-to-end tok/s, versus `19.01` / `16.75` /
`19.00` for the direct-copy 2K control, with identical hash
`c83dbcf03c2f4d7b` and `PASS`. The validated explicit pipeline ceiling is now
2,048 tokens (`LLM_QWEN4_PREFILL_COPY_PIPELINE_MAX_TOKENS`); the pipeline
itself remains disabled by default.

The HTTP launcher now automatically enables per-chunk publication when the
copy pipeline is explicitly enabled, because the runner uses that publication
to enforce the 2K safety ceiling. An explicit
`LLM_QWEN4_PREFILL_COPY_PIPELINE_PUBLISH_CHUNK=0` still disables it for a
controlled experiment.

### No-padding 4K coding gate

The overlap path was checked with the real 4K coherence harness (no repeated
last-token padding), a ChatML coding request, and 64 generated tokens. Refresh
4 and refresh 6 both produced the same compilable function:

```c
int clamp(int x, int lo, int hi) {
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}
```

Both runs returned hash `a9d9261a7fedaf3a`, `PASS`, and about `12.2` prefill /
`19.1` decode tok/s. Refresh 8 did not reach a footer within the 900-second
sweep window. This is the quality-safe result for heterogeneous 4K input; the
~29 tok/s padded benchmark is not representative of serving throughput.

A focused refresh-6 repeat with a 7.8-GiB expert cache and BMAX=512 raised
prefill to `12.79` tok/s, cache hit rate to `56.2%`, and reduced H2D to
`2,334 GiB`; it preserved the same compilable function and returned `PASS`.
The extra cache consumes nearly all remaining 16-GiB headroom (`1,754 MiB`
free), so this remains an explicit fast-prefill experiment rather than the
default 4K profile.

Enabling `LLM_QWEN4_PREFILL_GPU_TOPK=1` on the same run did not change the
route hash, cache hit rate (`56.2%`), H2D volume, or prefill time (`12.78`
tok/s). It is therefore not an additional 4K optimization; the host-routing
profile remains the quality-safe default.

Raising scalar CPU-prefill singleton handling to
`LLM_MOE_CPU_PREFILL_MAX_COUNT=4` with a 256-job cap was also neutral on this
real prompt: `12.79` prefill tok/s, `56.2%` cache hit, and `2,334 GiB` H2D,
with the same syntax-checked output. The cold-transfer bottleneck therefore
requires better expert residency or transfer reuse rather than router/CPU
threshold tuning.

Increasing `LLM_MOE_STREAM_SLOTS` from two to four on the same uniform-cache
4K overlap workload was neutral (`13.40` vs `13.41` prefill tok/s, identical
hit/H2D statistics and syntax-checked output). Two streams remain the lower
overhead choice for this single-request path.

On the same real 4K prompt, disabling depth-balanced residency for the
explicit overlap path (`LLM_QWEN4_PREFILL_CACHE_BALANCE=0`) raised prefill
from `12.79` to `13.41` tok/s, increased cache hit from `56.2%` to `59.1%`,
and reduced H2D from `2,334` to `2,177 GiB`; the generated function remained
identical and syntax-valid. The flash and HTTP launchers now select this
uniform policy automatically only for explicit pipeline requests at 4K+;
callers can still force balance with `=1`.

An 8K no-padding run with the 5.9-GiB/512-row profile and 2K overlap chunks
completed its refresh-6 benchmark at `10.48` prefill / `15.02` decode tok/s.
The generated fenced C function was coherent after normalizing escaped BPE
newlines and passed syntax checking, but the cache hit rate fell to `46.7%`
with `5,650 GiB` of H2D traffic. This confirms the overlap path is stable at
8K but does not solve the long-context transfer bottleneck; the 8K pipeline
remains diagnostic-only.

An intermediate 6.5-GiB cache improved the same 8K run to `10.85` prefill
tok/s (49.3% hit, 5,368 GiB H2D). The validated 7.2-GiB/512-row run reached
`11.42` prefill / `15.34` decode tok/s, 52.9% hit, and 4,983 GiB H2D while
still producing syntax-valid C. The launchers now select 7.2 GiB for the
validated 8K–<16K approximate profile and retain 5.9 GiB at 16K+ until those
larger contexts are separately validated.

A 16K allocation smoke with the conservative 5.9-GiB/I8 profile also loaded
and completed one prefill plus one decode token (`Result: PASS`, 3.58 GiB
free). This validates startup headroom only; it is not a 16K throughput or
quality claim.

The same 7.2-GiB/512-row allocation was also validated with direct copies
(pipeline disabled): `10.92` prefill / `15.81` decode tok/s, 51.6% hit,
5,120 GiB H2D, and syntax-valid output. This confirms the promoted 8K cache
budget is safe independent of the diagnostic overlap path.

The sub-32K diagnostic runner now wraps each HIP invocation in an internal
`timeout --foreground` (900 seconds by default, configurable with
`QWEN38_SUB32_TIMEOUT`). This prevents stalled parity experiments from leaving
orphaned GPU contexts and stranded VRAM.

Grouped MTP verification now reduces the scalar lm-head logits with one
deterministic `qwen4_argmax_batch` launch for the entire window instead of one
argmax launch and device copy per row. The smallest grouped transaction gate
(draft 4, 8 output tokens, forced reject/rollback coverage) still returned
`PASS`, with identical transaction checks and hash `9243484866657d53`; the
control measured `3.35` decode / `3.61` end-to-end tok/s. This is a launch
overhead reduction only; grouped target-layer parity and throughput remain
experimental.
A 16-token grouped control retained hash `34202a88d2a8906a` and measured
`3.30` decode / `3.46` end-to-end tok/s, up from the earlier `2.79` / `3.05`
control.
Important current-build caveat: grouped verification is only entered when
`LLM_QWEN4_BATCH=1` is explicitly set. Grouped logits can still diverge on
rejected windows (`pred0=1144` versus scalar `2688`), but forced transaction
checks now bypass grouped mode and use the scalar oracle, so the full
reject/EOS suite passes. Grouped throughput remains diagnostic only until its
state/commit parity is fixed.
Repeating that exact grouped control with the tuned 9-GiB expert cache completed
all transaction checks with the same hash and reached `6.08` decode / `5.57`
end-to-end tok/s. Cache telemetry was `55.0%` on prefill and `80.3%` on decode,
with `14.76` GiB peak VRAM. This confirms cache residency is the dominant
limiter for the grouped experiment, but the path remains well below scalar-MTP
throughput and is not promoted.
Grouped resident/deferred task lists now use asynchronous H2D copies ordered on
the compute stream, removing the prior host-visible copy/fence pair. The same
9-GiB exact control remained hash-identical and passed all rollback checks at
`6.07` decode / `5.59` end-to-end tok/s; this is currently a scheduling cleanup
rather than a confirmed throughput improvement.
For a 32-token coding prompt, allowing all attention layers to batch
(`LLM_QWEN4_BATCH_ATTN_MAX_LAYER=47`) preserved hash `48f9514bc5863ce4` and
passed, but was slightly slower (`5.94` vs `6.09` decode tok/s) than the
parity-safe three-layer prefix. Experimental batched SSM projections were also
hash-safe but neutral. Grouped performance is consequently dominated by scalar
SSM recurrence and MoE/cache traffic; the conservative attention limit remains
the default.

### Host-transfer overlap follow-up

The batched MoE router/grouping scratch buffers (`router_batch`, grouped token
indices, and grouped weights) are now allocated with `hipHostMalloc` when the
runtime supports pinned host memory, with a transparent `malloc` fallback.
Their existing asynchronous H2D/D2H copies can therefore overlap GPU work on
ROCm instead of silently synchronizing on pageable memory. Destruction now
releases the buffers through the matching HIP/free path. A real 982-token
coding prompt on the scalar approximate route completed with `PASS` and a
syntax-valid clamp function: `10.90` prefill / `19.87` decode / `11.21`
end-to-end tok/s at 64 generated tokens (hash
`325c17a54f291bb4`). The separate batched coding benchmark was traced to a
gfx1201 hipBLASLt workspace `hipMalloc` crash. Qwen4 batched prefill now
defaults to the self-owned WMMA GEMM backend; explicit `LLM_GEMM=blaslt`
remains available only for guarded A/B diagnostics because it can reproduce
the driver crash. The same 512-token coding benchmark
then completed with `PASS`: `115.19` prefill / `15.43` decode / `104.77`
end-to-end tok/s (hash `0b93d620afb74848`).

A 2,048-token padded batched control also completed with `PASS` at `121.03`
prefill / `14.57` decode / `117.68` end-to-end tok/s (hash
`ae68c4f961193200`), confirming the WMMA default remains stable across the
validated 512–2K tile range.

The coherence harness can now opt into this route with
`QWEN38_APPROX_BATCH=1` (plus the stateful/multi-chunk switches). A real
1,917-token coding prompt completed with syntax-valid output at `10.79`
prefill / `20.22` decode / `10.95` end-to-end tok/s (hash
`a9d9261a7fedaf3`). At 3,798 real tokens, the same 4K streamed batched route
reached `89.49` prefill / `22.40` decode / `85.26` end-to-end tok/s, but
generated repeated non-code text and failed the syntax gate. Batched WMMA is
therefore a useful throughput diagnostic, not a quality-safe 4K serving mode;
the scalar route remains the production quality baseline for long prompts. A
4K batch run with stateful/multi-chunk carry disabled did not reach a terminal
footer, so those controls remain experimental as well.

Using exact decode instead of device-hit approximate decode did not repair the
4K result (`88.56` prefill / `19.67` decode / `83.70` end-to-end tok/s;
repeated `1.0` output). Splitting the same request into two 2K batched chunks
raised prefill to `121.36` and end-to-end to `112.69` tok/s, but produced a
repeated `0` stream. The quality loss is therefore in batched prefill/state
propagation, not the approximate decoder; no batched 4K profile is promoted.
Disabling batched SSM scratch (`QWEN38_SUB32_BATCH_SSM=0`) did not reach a
terminal benchmark footer, so the scalar-SSM hybrid is not currently a safe
fallback either.
Constraining batched attention to the documented parity-safe prefix
(`QWEN38_SUB32_BATCH_ATTN_MAX_LAYER=1`) likewise entered a scalar-heavy
schedule without reaching a footer; it is not a practical 4K optimization on
the 16-GiB card.
