# Qwen3.8/Qwen4 RDNA4 status

Current validation appears first. Earlier investigations are retained below as
history; short-run determinism claims there do not establish scalar F16 parity.
Scalar F16 remains the default; staged prefill is diagnostic.

## Bounded staging implementation (2026-09-12)

The selected bounded-wave manager is implemented. Each of two banks owns its
metadata, quantized weights, Q8_0 repack scratch, and upload/consumption events.
Both host-source lifetime and device-slot reuse are fenced. Direct prefill now
uses independent slot fences, including Q8_0 misses and cache-hit consumers.
The prefill-to-decode map upload completes before its host buffer is freed.

The default pool remains 512 MiB, including device metadata/repack storage.
`LLM_MOE_COPY_PIPELINE=0` serializes waves; `=1` overlaps two banks. Promotion
remains off by default. Runtime errors cannot fall through to a partial-result
fallback. Reset, mode transitions, offload, and free drain outstanding prefill
work before touching owned storage. See [the staging design and commands](QWEN38_MOE_STAGING.md).

`LLM_BENCH_WARMUP=0` now actually disables the optional warmup (previously any
present value enabled it). Reference first-token/hash inputs are paired and
reject a repeatable candidate that differs from the scalar F16 oracle.

Validated:

- Build of the runner and model-free GPU staging test.
- `make -C rdna4/llm moe-stage-test`: profile, delayed-copy lifecycle,
  bank ownership, and benchmark-reference/parser tests all pass.
- CPU ownership test under AddressSanitizer/UndefinedBehaviorSanitizer with
  leak detection: pass. Removing either host-lifetime or slot-reuse fencing
  makes delayed-consumer assertions fail.
- `timeout --foreground 180s ./rdna4/llm/tmp/test_hip_qwen4_moe_stage`: pass
  on RX 9070 XT. Six Q4_K/Q5_K + Q5_1/Q8_0/Q6_K type combinations, both host
  registration modes, resident/cold/mixed workloads, partial waves, overlap,
  promotion, resets, and injected upload failure/recovery match a serialized
  expert oracle bitwise. This does not establish whole-model scalar parity.

Full-model matrix: RX 9070 XT, 9,000-byte header prompt, 4096 prefill / 64
decode, context 8192, BMAX4096, 4000 MiB cache, 512 MiB staging, eight measured
requests per process, warmup disabled, CPU experts/approximation/promotion off.
The common batched hash is `afdf60ceeb4f0103` (first token 16).

| Registered | Overlap | Common hash / 8 | Prefill min / median tok/s | Decode min / median tok/s | Peak MiB |
| --- | --- | ---: | ---: | ---: | ---: |
| No | No | 7 | 86.28 / 117.48 | 9.96 / 17.86 | 14294 |
| No | Yes | 8 | 97.46 / 128.87 | 17.54 / 17.83 | 14296 |
| Yes | No | 5 | 122.93 / 122.99 | 11.42 / 19.07 | 14294 |
| Yes | Yes | 7 | 132.22 / 132.30 | 19.17 / 19.69 | 14296 |
| Yes | Yes, fresh process | 6 | 132.09 / 132.19 | 11.39 / 19.66 | 14296 |
| No | Yes, fresh process | 8 | 96.33 / 128.63 | 16.11 / 17.74 | 14296 |

Four of six processes fail repeatability. Pageable overlap passes 8/8 in
both processes, but this does not establish scalar parity or justify promotion. Full-model nondeterminism remains despite passing ownership tests;
its cause is not established. Neither throughput target is met. Logs:
`tmp/qwen38_stage_matrix_[1-6]_r*_p*.log`. A CPU test compilation overlapped
part of the fourth process; the fresh fifth process had no compiler overlap.

The DIM2048 GPU stress variant also passes the same serial-oracle matrix with
multi-MiB weight transfers (`tmp/gpu_moe_stage_large.log`).

Direct-cache smoke: `bench_qwen38_target.sh` with profile `batch4k`,
512 prefill / 8 decode, two repeats, registered weights, BMAX4096/cache4000,
context8192, the same header prompt, and warmup off passes with copy pipeline
both 0 and 1. Both return first token 18 / hash `75c0ebdb415406cd` (2/2).
Serial median prefill/decode is 58.31/11.70 tok/s, peak13778 MiB; overlap is
54.14/12.21 tok/s, peak13780 MiB. These are bounded regression controls, not
4K determinism or scalar-parity evidence. Logs: `tmp/qwen38_direct_p[01].log`.

Matched F16 quality corpus (`./rdna4/llm/test_qwen38_staging_quality.sh`):
**all five candidates fail scalar parity**. Each scalar and staged result is
repeatable 2/2 within this corpus. The target is 4096/64; the other four are
128/16. All use context8192, cache4000, BMAX4096, registered weights, CPU
experts off, and no approximation. Scalar uses `scalar-exact`, batch0/copy0;
staged uses `batch4k-stage`, batch1/copy1. Exact prompts and remaining profile
settings are in the script; logs are `tmp/staging_quality/*_{scalar,staged}.log`.

| Prompt | Scalar first / hash | Staged first / hash | Parity |
| --- | --- | --- | --- |
| target | 15 / `e3d8bf6d47dc6cc3` | 16 / `afdf60ceeb4f0103` | FAIL |
| coding | 198 / `bbd62d9e3c85af8d` | 47932 / `12ced4d4ee57e7bb` | FAIL |
| arithmetic | 198 / `427a8efc219e9443` | 15 / `ee81d9e0e8e89d42` | FAIL |
| prose | 248068 / `c461a4dabdca797e` | 9619 / `458c70b584329680` | FAIL |
| japanese | 198 / `b5ad0bef0c9a5696` | 44868 / `a099ba1b986b4c43` | FAIL |

The fresh scalar target measured 10.83 prefill / 15.45 decode tok/s (median),
peak 11328 MiB. Its hash differs from the historical `fast` profile with
BMAX2048/7800 MiB cache; the cause of that cross-configuration difference is
not established. These results test repeatability and reference agreement,
not independent model-quality correctness. No throughput or quality target
is claimed, and no serving default is promoted.


Device access: the restricted namespace hides `/dev/kfd` and `/dev/dri`, but
the authorized host execution namespace exposes the RX 9070 XT. Check `fuser`
and VRAM before each standalone benchmark; never run concurrently with a server.

## Earlier measurements (before bounded staging)


- FP8 KV uses real scaled E4M3 encode/decode kernels, not an I8 alias.
- Short exact-MTP parity: I8, FP8, and F16 all returned hash
  `454146399ff97e88` for the 28-token/4-output control.
- FP8 64-token coding control retained the F16 hash
  `4b8937cd7db0e7a7` and returned `PASS`.
- FP8 256K allocation smoke: 3.000 GiB KV, 4.91 prefill, 5.60 decode,
  5.23 end-to-end tok/s, `PASS`.
- FP8 generation under the 256K allocation: 8-token prompt + 64-token greedy
  output, 4.84 prefill, 8.05 decode, 7.50 end-to-end tok/s, hash
  `d6951038d33a3c17`, 15.934 GiB peak used, `PASS`.
- F16 at max 256K allocation requests 6.000 GiB KV and fails finalization on
  the 16-GiB card; no F16 256K result is claimed.
- Profile, lifecycle, protocol, FP8 codec, and shell/static checks pass.
- With temporary `/dev/kfd`/`/dev/dri` access, the 512-token scalar control
  completed end-to-end at 25.71 prefill / 27.24 decode / 25.73 tok/s and
  `PASS` (12.708 GiB peak VRAM).
- The scalar two-chunk 1,024-token control also completed with the same
  sequence hash (`a2d4f49620d5b663`): 23.61 prefill / 23.24 decode / 23.61
  end-to-end tok/s, `PASS` (12.708 GiB peak VRAM).
- The built-in 512-token batched-vs-per-token comparison reported exact
  logits (`rel_l2=0`, `max_abs=0`) and matching argmax token 30.
- The I8 GQA8 selector now correctly recognizes Qwen4's 24/2 (12:1) shape,
  and its reduction bug (only four keys reaching softmax) was fixed. A matched
  512+64 control still produced a different hash with GQA8 enabled
  (`0b3e44e3e7fb5b0c`) versus scalar I8 (`6f231930c119b7e6`), so it remains
  opt-in pending long-horizon quality parity.
- I8 now honors `LLM_QWEN4_DISABLE_QSA=1`, matching the F16 control and
  allowing KV-only drift isolation; the default remains unchanged.
- Matched 512-token/64-decode greedy controls without MTP confirm the format
  boundary: F16 and FP8 both hash `e7e9b5ace7b8f98a` (26.06/24.25 decode
  tok/s respectively), while I8 hashes `6f231930c119b7e6` (15.86 tok/s).
  This isolates the long-horizon drift to INT8 KV quantization/attention, not
  routing or general recurrent state.
- A 16-channel I8/FP8 scale-group experiment was rejected: the I8 control's
  first token changed from 30 to 271. The implementation is restored to the
  validated 32-channel groups; finer grouping needs a separate parity design.
- The explicit batched WMMA path is stable for streamed-shaped requests: 2,048
  tokens at 114.88 prefill / 15.05 decode / 111.99 end-to-end tok/s, and
  4,096 tokens at 112.21 prefill / 14.53 decode / 110.75 end-to-end tok/s;
  both returned `PASS` with 13.448/13.544 GiB peak VRAM respectively.
- Two identical resumed 4,096-token batched runs reproduced throughput
  (111.26/111.49 prefill and 109.39/109.34 end-to-end tok/s, 13.544 GiB
  peak), but produced different hashes (`d973fd2d3e5f6bfc` and
  `42af972770bae7cf`). This is direct evidence that grouped/batched state is
  still nondeterministic on gfx1201 and cannot be promoted to a quality-safe
  default.
- Disabling expert copy pipelining (`LLM_MOE_COPY_PIPELINE=0`) did not remove
  the variance: the 4K run measured 96.67 prefill / 95.17 end-to-end tok/s
  and hash `d9f9ce178c4b4eb1`. The nondeterminism is therefore broader than
  the asynchronous copy overlap; no production switch was changed.
- An opt-in ordered expert-scatter reduction was also tested on the same 2K
  control. Two runs still produced different hashes, so atomic expert
  scatter is not the sole source of the grouped-path variance; the
  experimental kernel was removed rather than adding another unsupported
  production switch.
- The grouped resident/staged task lists previously used asynchronous H2D
  copies from pageable scratch that was overwritten during the subsequent
  cold-expert walk. Those metadata copies now complete synchronously before
  scratch reuse; runtime parity still needs a model-mounted rerun.
- The post-fix 2,048-token grouped controls both passed, at 115.61/109.73
  prefill/end-to-end tok/s and 118.97/113.57 tok/s, but still diverged at the
  first decoded token (17512 vs 220) and sequence hash
  (`70e30279debbe15f` vs `b01463d3c4871377`). The copy-publication race was
  real-risk mitigation, not the complete source of grouped nondeterminism.
- An opt-in scalar-router diagnostic (`LLM_QWEN4_BATCH_ROUTER_SCALAR=1`) also
  failed to stabilize matched 512-token controls: hashes were
  `5c7ed763fe63b0da` and `c6761ef1770a4293` (first tokens 515 and 10586), at
  71.97/66.68 and 71.10/66.07 prefill/end-to-end tok/s. Router reduction
  ordering is therefore not the sole remaining source.
- A stricter 512-token isolation with batched SSM, attention, router,
  projections, grouped MoE, and copy overlap disabled still diverged: hashes
  `ee573f2372785825` and `fbade0a3d53fbd73` (first tokens 17 and 11), at
  62.49/60.34 and 62.49/60.37 prefill/end-to-end tok/s. The remaining bug is
  therefore in broader batched state/stream publication, not one isolated
  grouped kernel; scalar dispatch remains the reference path.
- Per-row fallback and scalar KV diagnostic loops also had an async-copy hazard:
  each queued position transfer referenced a loop-local stack `pos`. Those
  publications are now synchronous. Matched post-fix controls still differed
  (`2aa0ef09c71a181b` vs `fbade0a3d53fbd73`, first tokens 17 vs 11), so this
  was another real race but not the complete source.
- Fresh rebuilt scalar controls remain deterministic: both 512-token runs
  returned first token 30 and hash `a2d4f49620d5b663`, at 25.84/25.86 and
  25.87/25.83 prefill/end-to-end tok/s. The instability is specific to the
  batched dispatcher, not general GPU state or model loading.
- For completeness, forcing `OMP_NUM_THREADS=1` on the scalarized batched
  isolation also failed to stabilize output: hashes `b9ee92610dbbc1cd` and
  `6254e5064f4050f4` (first tokens 17 and 271). Host OpenMP scheduling is not
  the remaining explanation.
- Forcing `LLM_QWEN4_BATCH_HC_SCALAR=1` stabilized the first decoded token
  (`907` in both 512-token controls), but later decode still diverged: hashes
  `3ced02b6184309f7` and `1c91fa80c934253e` at 48.84/48.35 tok/s end-to-end.
  This identifies batched HC/PLE arithmetic as one prefill mismatch source,
  while leaving a separate post-prefill decode-state/KV handoff issue.
- Forcing both `LLM_QWEN4_BATCH_KV_SCALAR=1` and
  `LLM_QWEN4_BATCH_ATTN_SCALAR=1` did not stabilize the batched path either:
  first tokens were 30 and 220, with hashes `525be427d6b320b5` and
  `f5829d6be7fb6728`. KV-store/attention publication is not a standalone fix.
- The HC-scalar result is repeatable as a prefill diagnostic but not a serving
  solution: it agrees on the first token while later decode state diverges;
  scalar KV/attention publication instead changes the first token again. All
  such switches remain opt-in and the scalar dispatcher remains the only
  quality-safe default.
- `HIP_LAUNCH_BLOCKING=1` also failed to stabilize matched 512-token batched
  controls: hashes `7c0b751e708b64fd` and `7ec265c22d7a16f8` (first tokens 47
  and 12920). The variance is not eliminated by global launch serialization.
- Forcing scalar token embedding (`LLM_QWEN4_BATCH_EMBED_SCALAR=1`) likewise
  left the batched path nondeterministic: hashes `fc6397b74e8e9d83` and
  `83452d2dfbc8621b` (first tokens 435 and 5652). Divergence begins after or
  within the first batched layer, not in embedding publication.

## Repeatability gate and batched divergence localization (Phase 0/1)

The runner now has an in-process repeatability gate so a profile cannot be
promoted without passing it:

- `test_hip_llm --bench-repeat N` resets recurrent/KV/PLE state between N
  identical requests in one process (the model loads once). `--bench` footers
  report `First decoded token id` and `sequence hash`.
- `bench_qwen38_target.sh` runs `scalar-exact`, `fast`, `batch`, or `approx`
  profiles for N repeats and fails unless every repeat has the same first token
  and the same full hash. It also reports min/median prefill/decode/end-to-end
  tok/s, peak VRAM, and `rocm-smi` clock/temp before and after.
  `bench_qwen38_256k.sh` (`QWEN38_BENCH_REPEATS`, default 2) and
  `bench_qwen38_sub32_target.sh` (`QWEN38_SUB32_REPEATS`) carry the same gate.
- `make -C rdna4/llm target-gate target-gate-fast target-gate-batch` wraps it.
- `debug_f32_state`/`debug_hc_state` now print a bitwise FNV hash of the full
  state in addition to norm/first under `LLM_DEBUG_LAYERS=1`, exposing one-ULP
  divergence that the 6-decimal print hides.

Matched results on the RX 9070 XT with the real 9,000-byte `gguf_loader.h`
prompt (`tmp/qwen38_target_prompt.txt`), profile `fast` (scalar, pinned host,
BMAX=2048, 7.8-GiB cache, GPU top-k) at 4,096 prefill / 64 decode:

- Three repeats were identical (hash `6d67721190bdaa83`, first token 30):
  23.96--23.98 prefill, 21.15--23.70 decode, 14,818 MiB peak.
- The quality-safe scalar route is therefore repeatable, but only ~24 tok/s
  prefill at 4K; the documented 100--235 tok/s figures are not reachable on the
  current scalar `fast` profile and require the batched route.

Profile `batch` (`LLM_QWEN4_BATCH=1`, `BATCH_SSM=1`, native Q6_K SSM
projections, fused recurrence, `BATCH_ATTN_MAX_LAYER=47`, Q6K/CONV/RECURRENCE/
PARITY, BMAX=1024, 5.9-GiB cache, 1,024-token stream) at 2,048 real tokens
originally reproduced the nondeterminism directly: three identical repeats
produced three different first tokens (32286, 16, 248046) and three different
hashes (`096888097a00e061`, `97574e0f11abcfd3`, `fb57a917f37a253e`).

### Fixes landed

- **Ordered MoE combine.** `moe_scatter_accum` summed the K selected experts
  with order-dependent `atomicAdd` into the token row. `moe_fill_gather` now
  records the expert-grouped slot for each `(token, rank)` in
  `d_moe_assign_pos`, and the new `moe_scatter_accum_ordered` sums the K
  contributions in fixed rank order. This moved the first bitwise divergence
  from layer 1 to layer 3 in the debug trace.
- **Synchronous CPU-result publication.** The CPU expert paths published their
  host results to the device with `hipMemcpyAsync` from `h_moe_output` /
  `h_moe_eout_cpu`, which the next token/layer overwrites; a DMA could race the
  rewrite. Both publications are now synchronous `hipMemcpy`.
- **CPU vs GPU cold-expert arithmetic.** The CPU expert kernels
  (`hllm_cpu_*_jobs`) do not reproduce the GPU kernels bit-for-bit, so which of
  the two evaluates a cold expert changes its contribution. Expert-cache warmth
  selects that path, so back-to-back in-process requests with the CPU path
  enabled hash-differ even though two fresh processes matched
  (`4f21b1d68505eec3` twice at 512/16). The repeatability gate therefore
  defaults CPU expert work off (`bench_qwen38_target.sh` `batch` profile,
  `QWEN38_TARGET_CPU_EXPERTS=0`); `batch-cpu` keeps the mixed path for
  diagnostics.

### Pre-manager observations (RX 9070 XT, real 9,000-byte prompt)

- Single-chunk batched (`prefill <= BMAX`), CPU experts off: **improved but
  still not reliably deterministic**. One real race was fixed: the per-row
  position publication used a blocking `hipMemcpy(r->d_position, &pos, ...)` on
  the null stream, which raced `r->stream` kernels still reading `d_position`
  from the prior row (every row of the forced-scalar layer 1). It now uses a
  stream-ordered `hipMemcpyAsync(r->d_position, &r->h_pos_batch[m], ...)` from a
  stable precomputed host array. With pageable host weights and direct copies
  an 8-repeat run passed, but repeating the same configuration diverged on
  2/8, and pinned host weights or the async pipeline diverge on roughly 1/8.
  `HIP_LAUNCH_BLOCKING=1` largely hides the residual, so at least one more
  ordering dependency remains. Forcing per-token MoE (`LLM_MOE_PREFILL_SCALAR=1`)
  at 512 prefill / 8 decode is deterministic 6/6 (`1ae7b536b9c17b1d`), which
  isolates the residual race to the batched MoE dispatcher
  (`forward_moe_ffn_batched`), not batched attention/SSM. The batched MoE
  kernels themselves have no atomics, and the cause is the cold-expert cache
  H2D not being ordered with the consuming kernel: a plain
  `hipMemcpyAsync(..., r->stream)` did not reliably order on this ROCm stack.
  The direct cold path now copies on `moe_copy_stream` and makes the compute
  stream wait with a `hipEventRecord`/`hipStreamWaitEvent` pair. With that,
  4,096/64 passes 7/8 repeats (hash `afdf60ceeb4f0103`) at ~125 prefill /
  19.7 decode, versus frequent failures before, but one repeat still diverged
  (`989013653e29726e`), so a rarer residual remains. Disabling the
  `d_moe_eout`/gather alias (`LLM_QWEN4_MOE_EOUT_ALIAS=0`) and switching to host
  router top-k (`LLM_QWEN4_PREFILL_GPU_TOPK=0`) each still diverged, so neither
  the alias nor the GPU router is the sole cause. A 4-repeat
  `LLM_DEBUG_LAYERS=1` trace at 1024 did not reproduce it (the per-stage sync
  perturbs timing). The scalar route remains the only quality-safe default.
- Multi-chunk stateful batching (prefill > BMAX, `LLM_QWEN4_BATCH_MULTI_CHUNK_
  FORCE=1`) still diverges: a 4,096-token prompt split at BMAX=1024 produced a
  different third-repeat hash. The inter-chunk state carry has a separate
  remaining race.
- The scalar `fast` route remains repeatable (3/3, hash `6d67721190bdaa83`) but
  only ~24 tok/s prefill at 4K.

### 4K batched profile (`batch4k`)

A single 4,096-token batched dispatch fits on the 16-GiB card with BMAX=4096
and a 5,000-MiB resident cache (peak ~14,900 MiB). `bench_qwen38_target.sh
batch4k` wraps it: GPU router top-k, CPU experts off, and
`LLM_QWEN4_RESET_MOE_CACHE=1`. It usually reproduces hash
`afdf60ceeb4f0103` (first token 16). With pageable host weights and direct
copies (the most-repeatable settings, `reg=0 copy=0`) it measured median
**125 prefill / 19.6 decode / 115.6 end-to-end tok/s**; with pinned host weights
and the async pipeline (`reg=1 copy=1`) it reaches median **149.2 prefill /
21.4 decode / 136.7 end-to-end tok/s** but diverges more often. Neither is fully
repeatable yet.

State-isolation findings that drove the profile:

- Expert-cache residency changes the result even with CPU experts off: the
  first repeat after load differed from later repeats that inherited cache
  residency. `hip_llm_reset_state` now clears the routed-expert cache under
  `LLM_QWEN4_RESET_MOE_CACHE=1`, so every repeat/request starts cold.
- The earlier asynchronous cold-upload pipeline lifted prefill from ~132 to
  ~147 tok/s, but its initial request-isolation fix did not remove the rare
  residual. Overlap remains opt-in; `batch4k` defaults to direct copies.

Multi-chunk prefills (`prefill > BMAX`) still need the scalar fallback or a
separate determinism fix; single-chunk 4K is a diagnostic profile, and these earlier short controls did
not justify a production or scalar-parity claim.

### Why the 200-tok/s prefill target is not reached yet

The expert working set (512 experts x 48 layers, top-10 routing) is far larger
than any cache that fits beside the batched scratch, so the routed-expert hit
rate is routing-limited, not cache-size-limited:

| Config | Prefill tok/s | Cache hit | H2D | Peak MiB |
| --- | ---: | ---: | ---: | ---: |
| 4K, BMAX=4096, cache=5000, pipeline | 147.4 | 32.4% | 124 GiB | 14,788 |
| 4K, BMAX=4096, cache=6200, direct | 136.0 | ~32% | ~124 GiB | 16,010 |
| 2K, BMAX=2048, cache=7800, pipeline | 142.9 | 25.6% | 90.6 GiB | 15,990 |

The larger-cache 2K run has a *lower* hit rate (25.6% vs 32.4%): a shorter
prompt issues fewer tokens per expert, so the same routing diversity fits less
of each expert's assignments. Raising the cache to 6,200 MiB also leaves only
294 MiB free and, with the async pipeline, reintroduces a first-repeat
divergence; it is not adopted. `LLM_MOE_STREAM_SLOTS>2` is nondeterministic
(three different hashes at slots=4), so two slots remain the deterministic
default.

Reaching 200 prefill / 30 decode needs a different expert execution or overlap
strategy, not further tuning of the current knobs. The pipeline is already
overlapping the transfer; the remaining limit is routed-expert transfer volume
plus per-expert compute.

### Grouped routed-expert investigation

Several grouped strategies were measured against the then-apparently-stable
147-tok/s single-dispatch profile; later repeats disproved that determinism claim:

- **Grouped-BF16-WMMA (`gemm_bf16_grouped`) is memory-infeasible here.** The
  all-expert staging buffer is sized `ne*N*K` bf16 (512 experts), ~15 GiB for
  this model, and Qwen4 never allocates it (`d_expw_bf16` is skipped for
  `is_qwen4exp`). Compacting it would require new dequant kernels, and the
  model is transfer-bound, so doubling weight bytes to bf16 cannot win.
- **Per-expert BF16-WMMA** (`LLM_QWEN4_NATIVE_EXPERTS=0`, a new A/B gate) was a
  wash for prefill (146--160 vs 147) and much worse for decode (13.3 vs 21.4
  tok/s, decode cache hit 65% vs 88%). It is also nondeterministic.
- **Grouped resident-only** (`LLM_MOE_GROUPED_PREFILL=1`) gave no prefill gain
  (135--148) because every layer starts with an empty per-layer cache, so there
  is nothing resident to group; it also diverged on the third repeat.
- **Staged grouped cold experts** (`--qwen4-prefill-staging`, cache 4000)
  is the only grouping that helps: median 162--180 prefill / 19.5 decode. It
  copies cold experts into double-buffered staging banks and runs one grouped
  gate-up and one grouped down launch per wave (~350 waves, 0 fallbacks at 4K).
  Two race sources were found and fixed: the per-wave staging map and task
  arrays were published with blocking `hipMemcpy`, which is not ordered against
  kernels on `r->stream` (now `hipMemcpyAsync` on `r->stream`), and the
  staged-to-cache promotion copies are now disabled under the per-request cache
  reset (`LLM_QWEN4_STAGE_PROMOTE=0`). With `HIP_LAUNCH_BLOCKING=1` the staged
  path becomes deterministic and reproduces the non-staged hash
  (`afdf60ceeb4f0103`), confirming its arithmetic is correct and the raced
  results (`ea20ffd2b071b6c3`, `b05a25a0c51bb35c`, ...) were corrupt.
  After the per-row position fix, pageable host weights
  (`LLM_MOE_REGISTER_HOST=0`) make the staged path deterministic 5/5 and it
  reproduces `afdf60ceeb4f0103`, but only at ~119 prefill / 17.7 decode --
  below the non-staged `batch4k` (149/21). With pinned host weights the
  staging-bank async copy still races its consumer (a 5-repeat run diverged),
  so the staged preset remains diagnostic and is not a net win.

The `--qwen4-prefill-staging`, `LLM_QWEN4_STAGE_PROMOTE`, and
`LLM_QWEN4_NATIVE_EXPERTS` switches are exposed for further work; the staged
path's first-request hazard is the next thing to isolate.

The shared-memory `atomicAdd(&head_sq[head], ...)` in
`fused_ssm_out_gated_q6k` is order-dependent but decode-only; the batched SSM
norm uses the deterministic tree reduction in `gated_rmsnorm_silu_batch_f32`.

## Explicitly unresolved

- Long exact-MTP I8 diverges from F16 after roughly 16 generated tokens,
  although output remains coherent. I8 remains explicit-only.
- Grouped Qwen4 prefill still has numerical/state parity failures and remains
  diagnostic-only.
- Scalar 2K/4K runs exceeded the interactive observation window before a
  footer was captured (the 1K two-chunk control is clean); they must not be
  called failures until rerun with a persistent host/container GPU session.
  The batched results above are therefore still an explicit A/B path, not a
  default-quality claim.
- A 32K+ input / 8K+ output streamed coding run has not completed with a
  trustworthy footer and coherence gate.
- The current ROCm image has `libhipblaslt.so` but no hipBLASLt headers, so the
  200+ tok/s accelerated prefill path cannot be rebuilt here. `make -C rdna4/llm
  hipblaslt-status` reports this directly.
- FP4 KV is not implemented or aliased: the current cache formats are F16,
  symmetric I8, and real E4M3 FP8. A useful FP4 implementation would need a
  specified packed FP4 encoding and scale/error policy first; silently
  reusing the I8 byte path would not be a valid assessment.
- Persistent `/dev/kfd` access requires host/container device passthrough;
  elevated command namespaces are temporary and are not persistent access.

## Remaining tasks

- [~] Validate the bounded-wave manager across host-registration/overlap
      combinations, then compare every candidate to scalar F16 greedy hashes.
      Repeated agreement within a batched profile is insufficient for promotion.
      Multi-chunk KV/SSM/HC state carry remains separate unresolved work.
- [ ] Make the CPU cold-expert kernels bit-identical to the GPU kernels (or
      keep CPU experts opt-in only). CPU and GPU expert arithmetic currently
      differ, so expert-cache warmth changes a request's output when the CPU
      path is enabled; `batch-cpu` is diagnostic-only.
- [x] Add a repeatability gate to the streamed 512/2K/4K benchmark: run at
      least two identical requests, compare first token and sequence hash, and
      report prefill, decode, and end-to-end wall-clock tok/s together.
      Delivered as `test_hip_llm --bench-repeat N`,
      `bench_qwen38_target.sh`, the 256K/sub-32K repeat gates, and the
      `target-gate` Makefile targets.
- [ ] Complete a quality-gated 32K+ prompt / 8K+ streamed coding workload
      using 512--2048-token prefill chunks and 64--128-token decode chunks.
      Record coherence, hash/repeatability, peak VRAM, and end-to-end tok/s.
- [ ] Re-run scalar 2K/4K controls in a persistent GPU session and capture a
      complete footer; do not infer their throughput from timed-out runs.
- [ ] Obtain a ROCm image with hipBLASLt development headers, rebuild the
      accelerated prefill path, and compare it against the current scalar and
      WMMA controls without changing the quality gate.
- [ ] Continue FP8 KV long-context quality testing and measure the practical
      256K prompt path. Keep F16/I8/FP8 comparisons separate; no FP4 result is
      valid until a packed encoding and scale/error policy are specified.
- [ ] Make GPU device passthrough persistent for benchmark sessions
      (`/dev/kfd`, `/dev/dri`, `video`, and `render`) so results are not tied to
      temporary elevated namespaces.

Capacity-only results must not be reported as 256K-prompt throughput.
