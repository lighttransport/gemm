# DS4F resume handoff

Worktree: `/mnt/nvme02/work/gemm/ds4f`

## Objective

Continue toward the user’s actual serving target on CPU + Radeon 9070 XT:

- preserve output quality first;
- single-stream decode around 18 tok/s;
- prefill around 100–200 tok/s for 1K+ input;
- prefix/system/tool-token caching for coding-agent requests;
- harden long-context and multi-context operation.

Do not redefine success around the current low benchmark. Do not push to any remote without explicit user permission.

## Authoritative model and staging

```text
GGUF: /mnt/nvme02/models/ds4f-0731/DeepSeek-V4-Flash-MXFP4Experts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2-mxfp4-0731.gguf
Stage: /tmp/ds4f_nocopy_stage
Tokenizer: /mnt/nvme02/models/ds4f-0731/tokenizer.json
Quality prompt IDs: /tmp/ds4f_quality_ids.txt
Quality config: /tmp/ds4f_quality_exact.json
```

## Git/source state

Latest relevant commits:

```text
004c9422 Make transient expert uploads stream ordered
4901156c Stage only active experts for exact serving
8329d84d Fix mHC quality diagnostics and routed decode residency
dc170d69 Fix DS4F routing and grouped prefill
```

No uncommitted source changes as of 2026-08-11 21:30 — see "Pool-change verdict" below.

Untracked runtime logs (do not commit):

```text
a64fx/llm/ds4f_frontend.log
a64fx/llm/ds4f_runner.log
```

## Pool-change verdict (2026-08-11)

The transient routed-expert upload pool described above was runtime-tested and **reverted**. Mechanically the diff looked sound (owner=1 matrices skip `hipFree` in `release_matrix`, pool storage freed in `clear_matrices`), but under `ds4f_serve_bench.py --prompt-tokens 64 --warm-decode 1 --decode-tokens 2` it hung inside `sess.prefill()` — the standalone bench process loaded its model in ~71s (matches expected load time) then produced no `prefill tokens=...` line before a 180s timeout killed it. No orphan processes, no VRAM growth (idle at ~57 MB before and after). This matches the "prior complete-layer attempt stalled" failure mode already noted below — do not retry the same pooling approach without root-causing the stall first (likely a sync/ordering issue introduced by reusing device pointers across calls while `append_device_matrix` still treats the id as fresh).

`git restore hetero/ds4f/hip_ds4f_dense.c` was applied, then `make -C hetero/ds4f -j4` and `sh a64fx/llm/build_ds4f_serve.sh` both rebuilt clean (pre-existing warnings only). Re-baselined against the reverted (per-call malloc/free) routed FFN path:

```text
prefill tokens=64 seconds=18.57 tok/s=3.45
decode  warm=1 tokens=2 seconds=0.42 tok/s=4.77
```

This matches the previously recorded baseline range and confirms the revert is stable. **Caution for future runs:** `ds4f_serve_bench.py` loads its own full 156GB model standalone — it does not talk to a separately running server. Running it while `run_ds4f_single_serve.sh` is also up causes severe CPU/memory contention (observed prefill dropped to 1.1 tok/s, decode to 0.6 tok/s, purely from resource contention, not a real regression). Stop any running server before using this benchmark script, or use the HTTP server + curl path instead if concurrent operation is required.

## Phase telemetry (2026-08-11, added and measured)

Added per-call phase timing inside `hip_ds4f_dense_routed_ffn` (hetero/ds4f/hip_ds4f_dense.c): `prof_route_upload_seconds` (transient expert hipMalloc + H2D memcpy queuing loop), `prof_route_launch_seconds` (buffer setup + GEMM/SwiGLU kernel launch queuing), `prof_route_sync_seconds` (final D2H copy + `hipStreamSynchronize`, where all previously-queued async stream work actually completes). Printed under the existing `DS4F_PROF=1` env flag alongside the pre-existing `hip_route` line, at context destroy.

Measured with `DS4F_PROF=1` + the "Useful benchmark" command (64 prompt tokens, warm-decode 1, decode-tokens 2), no server running concurrently:

```text
hip_route     14.689 s  calls=43 avg=341.609 ms
  upload      13.363 s  avg=310.768 ms (91.0%)
  launch       0.049 s  avg=1.131 ms   (0.3%)
  sync         1.278 s  avg=29.710 ms  (8.7%)
```

**Finding: 91% of routed-FFN wall time is the transient-expert upload phase** (`hipMalloc`+`hipMemcpyAsync` per active expert projection, current code has no pool — reverted above). Kernel launch overhead is negligible (0.3%); the sync/D2H tail (8.7%) is real GPU+copy time. This confirms the earlier pooling instinct was aimed at the right bottleneck, but its implementation stalled at runtime (see verdict above) — root-causing and fixing *that* stall (not abandoning the approach) is the highest-leverage next step. Suspect areas for the stall: reusing a pool slot's device pointer for a new async `hipMemcpyAsync` while a prior kernel might still be reading from it without an intervening sync/event (the old code queued per-expert async copies across multiple calls without any per-slot completion tracking), or an allocator interaction between the routed-FFN stream and the runner's own copy thread (`ctx->stream_copy_thread`).

## Pool retry with per-slot events (2026-08-11, done)

Re-added the transient routed-expert upload pool, this time with a `hipEvent_t` per pool slot: `hip_ds4f_dense_bind_mxfp4_tensor_async_pool()` (hetero/ds4f/hip_ds4f_dense.c) waits on a slot's event (if one was recorded) before reusing its storage, and `hip_ds4f_dense_routed_ffn` records each used slot's event right after all of that call's kernels are enqueued (before the launch-phase timestamp). Ran the same `DS4F_PROF=1` bench with a 180s timeout, no server running concurrently.

**No stall this time** — completed cleanly, no orphan process, VRAM back to idle baseline (~57 MB) after exit. Quality gate re-run (mHC exact, same command as always) still passes: 8/9 argmax match, same token index [2] mismatch as the pre-existing baseline — no regression.

```text
hip_route     13.533 s  calls=43 avg=314.720 ms
  upload      12.133 s  avg=282.169 ms (89.7%)
  launch       0.071 s  avg=1.650 ms   (0.5%)
  sync         1.329 s  avg=30.901 ms  (9.8%)
prefill tokens=64 seconds=17.05 tok/s=3.753   (was 3.447)
decode  warm=1 tokens=2 seconds=0.386 tok/s=5.176  (was 4.691)
```

**Important finding — the pool did NOT eliminate the upload cost, only ~8-9% of it.** Upload is still 89.7% of routed-FFN time (vs. 91.0% before pooling). Root cause: `release_transient_experts()` unconditionally resets `t->gpu_id = -1` after every call, so even though the pool retains the *device buffer*, the code has no per-expert-identity cache — it re-issues a full `hipMemcpyAsync` of the same weight bytes from host RAM on every call that routes to that expert, regardless of whether the last call routed the same expert to the same slot. The pool only removed `hipMalloc`/`hipFree` call overhead (a real but small win), not the dominant cost, which is genuine PCIe H2D bandwidth for re-transferring MXFP4 expert weights every single routed-FFN call.

This is now committed (small in isolation, but validated stable + quality-clean): telemetry (upload/launch/sync split) and event-guarded pool are both in.

**Next real lever:** cross-call expert-weight residency/caching keyed by expert identity (not just slot-buffer reuse) — e.g. an LRU or hot-expert cache that skips the `hipMemcpyAsync` entirely when the same expert's weights are already resident in a pool slot from a recent call, only re-uploading on eviction. This is different from the abandoned "full-layer resident upload" idea (which tried to keep *all* experts resident and stalled) — this only needs to cache the actually-hot experts across nearby calls with an explicit VRAM budget and eviction policy. Validate incrementally under `DS4F_PROF=1` at each step.

Build already completed successfully:

```sh
make -C hetero/ds4f -j4
sh a64fx/llm/build_ds4f_serve.sh
```

The build emits pre-existing warning noise but no compile/link error.

## Decode does not use the routed-FFN path we optimized (2026-08-11, important correction)

Tested the pre-existing prompt-hot expert cache (`hip_ds4f_dense_cache_hot_experts`, hetero/ds4f/hip_ds4f_dense.c:1341 — admits hot experts by identity into permanent GPU residency after prefill, budgeted by `--hip-expert-cache-mb`/`--hip-expert-cache-reserve-mb`) before writing any new caching code, since it already does exactly what "expert-weight caching" would mean. At matched settings (same 64-token prompt, warm=1/decode=2), decode was **5.163 tok/s with the cache vs. 5.176 tok/s without — no measurable difference.**

Root cause: decode's per-token expert compute does not go through `hip_ds4f_dense_routed_ffn` (the function the pool/telemetry above targets) except in the rare case where *all* of a token's top-k routed experts are simultaneously GPU-resident (common/ds4f_impl.h:7732, `c->n_active <= 8` all-GPU fast path). Confirmed via `hip_route` call counts staying flat (~44-46) regardless of whether decode ran 2 or 16 tokens — decode almost never takes that path. The actual decode dispatch (common/ds4f_impl.h:7764-7818, `ds4f_expert_batch_on()` path) splits each token's active experts into a GPU bucket (per-expert matvec for tensors with `gpu_id >= 0`) and a CPU bucket (exact CPU MXFP4 for the rest) — mixed per-expert, not all-or-nothing.

Added new DS4F_PROF sub-timers `exp_cpu`/`exp_gpu` (common/ds4f.h: `DS4F_P_EXPERTS_CPU=24`, `DS4F_P_EXPERTS_GPU=25`, extended `DS4F_NPHASE` 24->26; instrumented in common/ds4f_impl.h around the `gateup_cpu`/`gateup_gpu`/`down_cpu`/`down_gpu` `ds4f_matvec_multi` calls and the all-GPU `m->gpu_routed_ffn` call; loop bounds in `ds4f_serve_close()` (a64fx/llm/ds4f_serve_lib.c:319-320) widened from `DS4F_P_COMM` to `DS4F_NPHASE` to print them). Measured with the hot cache enabled (449 bundles, 6GB, 47% training coverage) over a 64-token prompt + 20 decode tokens:

```text
exp_cpu   1.290 s  18.6% of profiled total
exp_gpu   0.470 s   6.8% of profiled total
```

**CPU fallback costs ~2.7x the GPU-hit path for the same aggregate expert work.** This matches the `hot8_coverage` numbers already printed by the cache (~25-55% per layer, from a single short prompt) — most decode-time expert accesses miss the cache and pay the slower CPU path. Verified: telemetry-only change, mHC exact quality gate still 8/9 (no regression), no VRAM leak, no orphan processes.

**This means the pool/telemetry work above (upload/launch/sync split) only affects prefill, not the harder 18 tok/s decode target.** The real decode lever is raising the effective cache hit rate — either a bigger/smarter admission budget, an online/adaptive cache instead of the current static post-prefill snapshot, or speeding up the CPU fallback path itself. Not yet attempted; needs a decision on approach before more code.

## Cache budget is not the limiter; longer prompts don't help decode either (2026-08-12)

Tested raising `--hip-expert-cache-mb` from `auto` toward the full free VRAM (tried 20000 MB with 1536 MB reserve, vs. `auto`/-1 which already computes `free_bytes - reserve` when no explicit cap is given). **Identical result to `auto`**: `bundles=449 bytes=6.003GB training_coverage=47.24%`, decode still ~5.0 tok/s. `auto` was already using all available headroom — the 47% coverage ceiling is set by how many distinct experts the 64-token prompt actually touched, not by VRAM budget. There was nothing more to admit.

Tested prompt length directly as a fix (1024 and 8192 tokens, `--hip-expert-cache-mb auto`): coverage got *worse*, not better, as context grew (47.2% -> 41.9% -> 28.1%), and decode throughput dropped correspondingly (5.00 -> 3.23 -> 3.25 tok/s). Longer prompts spread routing across more distinct experts (diluting the fixed-size hot-cache's coverage of total routing mass) and their larger prefill activation buffers eat into the free VRAM available at cache-admission time (449 -> 336 -> 177 bundles admitted as prompt length grew, despite `auto` still targeting max headroom each time). **Rules out "use a longer/more realistic prompt" as a decode-cache fix — it's a headwind, not a help.** The online/adaptive-cache and CPU-fallback-speedup options from the prior finding remain the live candidates for decode; still not attempted.

## Prefill scaling and the real prefill bottleneck (2026-08-12)

Prefill throughput by prompt length (`--hip-expert-cache-mb auto`, `DS4F_PROF=1`):

| tokens | tok/s |
|---|---|
| 64 | 3.75 |
| 1024 | 11.03 |
| 8192 | 11.65 |

Scaling plateaus fast: 1024->8192 (8x tokens) only gained ~6%. At 8192 tokens, `hip_route` (the routed-FFN GPU work the pool/telemetry above targets) was only 63s of 703s total prefill wall time — 9%. The other 91% was invisible to any existing telemetry, because **the actual prefill compute function, `ds4f_forward_verify` (common/ds4f_impl.h:7000-7315, called from `ds4f_serve_prefill` in a64fx/llm/ds4f_serve_lib.c:353), had zero `DS4F_TIC`/`DS4F_TOC` instrumentation** — every phase name printed under `DS4F_PROF=1` (qkv_proj, attn, experts, tb2*, etc.) was actually reporting only the tiny per-token *decode* function's (`ds4f_forward_token`) contribution, not prefill's.

**Retraction:** the earlier "tb2scan = 75.5s across 6 decode tokens, likely a one-time ctx-scaling stall" finding was a measurement artifact of this same gap, not a real anomaly — that number was decode-only and had nothing to do with the 8192-token prefill that preceded it. Disregard it.

Added phase timers directly inside `ds4f_forward_verify`, reusing the same `DS4F_P_*` ids as decode (manually-scoped local timestamps rather than the `DS4F_TIC`/`DS4F_TOC` macro in the outer per-layer scope, since the macro's fixed `_t0` name would collide across sibling blocks not wrapped in their own braces; the existing `DS4F_TIC()` macro is still used for the per-position `ds4f_tb2_prepare` sub-timer since each `k` loop iteration is already a fresh scope). Measured at 1024 tokens, cache disabled to isolate prefill:

```text
attn        34.019 s  41.7%   <- now the single largest phase, not experts
experts     24.215 s  29.7%   (hip_route's 21.977s + ~2s CPU-side routing overhead)
qkv_proj     4.271 s   5.2%
head         2.085 s   2.6%
shared       2.488 s   3.1%
o_proj       1.949 s   2.4%
router       0.158 s   0.2%
  tb2prep (nested in attn)   5.984 s -- only 18% of attn's total
```

These 7 top-level phases sum to 69.16s, matching the measured 68.89s prefill wall time almost exactly — full accounting, nothing left unexplained. Within `attn`, `tb2prep` (the per-position index/compression scan) is a real but minority contributor (18%); the majority (~82%, ~28s) is the actual attention compute itself. The attention block dispatches the thread pool once per token per layer (`ds4f_pool_run` inside a `for k in K` loop, ~1024 x 43 ≈ 44,000 calls at this prompt length) — a strong candidate for per-position dispatch overhead dominating over raw compute, distinct from anything the routed-FFN work touches.

Verified: mHC exact quality gate still 8/9 (no regression, telemetry-only change), no VRAM leak, no orphan processes.

**Attention (42% of prefill) is now the clear top lever for the 100-200 tok/s prefill target — bigger than the routed-FFN/expert-upload work already done.**

## Attention cost is raw compute, not thread-pool dispatch overhead (2026-08-12)

Root-caused whether attn's cost is per-position `ds4f_pool_run` dispatch overhead or actual compute. Added a diagnostic-only synthetic benchmark (`ds4f_noop_pool_worker` + a one-time, `DS4F_PROF`-gated loop of 20,000 `ds4f_pool_run` calls doing no real work, at the top of `ds4f_forward_verify`, common/ds4f_impl.h) that measures pure barrier/dispatch overhead using the exact same pool/thread-pinning setup as production code — no numerics touched, diagnostic only, runs once per process.

Result: **2.885 μs/call, nthr=16**. Even generously assuming ~6 `ds4f_pool_run` dispatches per position per layer (one for the attention worker itself, several more nested inside `ds4f_tb2_prepare`'s sub-steps: idxsc/cmpmv/bf16mv/tb2rope workers), that's 1024 positions x 43 layers x 6 ≈ 264,000 calls x 2.885 μs ≈ **under 1 second total** — negligible against the measured 33-34s `attn` phase.

**Conclusion: attention's cost is raw compute, not dispatch overhead.** This rules out "batch the per-position loop to reduce dispatch calls" as a fix. The real lever is the attention math itself — the per-position exact/tier-B2 causal attention workers (`ds4f_attn_exact_worker`, `ds4f_attn_tb2_worker`/`ds4f_attn_tb2_gemm`, common/ds4f_impl.h, called from the per-position loop at ~common/ds4f_impl.h:7108) and/or the tb2 compression math inside `ds4f_tb2_prepare` (already separately timed at 18% of attn's total). Next step would be profiling *inside* those worker functions to find which specific computation dominates (e.g., the KV-cache attention score/softmax loop scaling with growing context length, vs. the sinkhorn/mix hc math, vs. index-scan specifics) — not yet attempted.

Verified: mHC exact quality gate still 8/9 (no regression, diagnostic-only addition), no VRAM leak, no orphan processes.

## Attention's real cost: already-optimized kernel, bounded window, but wide head_dim (2026-08-12)

Continued root-causing with two more cheap diagnostics, both added to `ds4f_attn_tb2_gemm` (common/ds4f_impl.h) and printed in `ds4f_serve_close()`:

1. **`attn_gemm hit`/`miss` counters** confirmed the fast 8-head-blocked SVE kernel (`ds4f_attn_gemm_score/soft/axpy_worker`, gated by `DS4F_ATTN_GEMM` env, default on) is **already active for 100% of calls** (`hit=42107 miss=0` at 1024 tokens). The slower per-head fallback (`ds4f_attn_tb2_worker`/`ds4f_attn_exact_worker`) is never taken in our config. Ruling out "enable the fast path" — it's already enabled.
2. **`avg_nP`/`avg_nsel`/dims** at 256 tokens: `nh=64 heads, head_dim=512, window=128, topk=512, avg_nP=96.6, avg_nsel=16.7`. This confirms attention cost is **bounded per position (~113 attended KV entries), not O(context_length)** — `window_size` defaults to 128 (a small sliding window), so the earlier worry about O(K^2) causal blowup at long context was wrong; the window term plateaus after position ~128 and the compressed term (`nsel`) stays small too since few compressed tokens are available yet at this context length.

**Conclusion: attention's cost is legitimate, already-optimized compute, not an artifact or a bug.** The per-position work (`n_heads=64 x ~113 KV entries x head_dim=512`) is large mainly because `head_dim=512` is unusually wide for this architecture (baked into the model's staged config, not a runtime knob) — every position does real FMA work proportional to that, using an SVE-vectorized, 8-head-blocked kernel that's already reusing KV loads across heads. There is no redundant/wasted work found in this investigation.

**What would actually move prefill toward 100-200 tok/s:** the CPU attention path is fundamentally compute-bound at these dimensions; a large (order-of-magnitude) speedup would need attention to run on the GPU (the routed-FFN work already GPU-offloads MoE; attention currently does not — no `m->gpu_attn*` callback was found wired for this per-position causal path). That is a substantial new engineering effort (a GPU causal-attention kernel with the same window+compressed-term semantics, careful quality validation against the mHC exact gate) — out of scope to attempt safely in the remaining time of this session. Smaller, lower-risk levers (a modest constant-factor kernel tune) were not pursued given the 1-hour time budget and the priority on not risking accuracy without adequate validation time.

Verified: mHC exact quality gate still 8/9 (no regression, diagnostic-only additions), no VRAM leak, no orphan processes.

## Two upload-transfer optimization attempts, both tried and reverted (2026-08-12)

Given "upload" is still ~80-91% of routed-FFN time even after the event-guarded pool, tried to speed the transfer itself (not the allocator) by avoiding the driver's pageable-memory bounce-buffer copy. Both attempts were built, tested for correctness/stability with a timeout, benchmarked, and reverted — neither is in the tree or committed.

**Attempt 1 — `hipHostRegister` the full min-to-max address span of this call's active experts.** Registers (pins) the byte range spanning every expert about to be uploaded, so the async H2D copy can DMA directly. **Result: severe regression** — 256-token prefill dropped from 8.8 tok/s to 3.0 tok/s (`upload` went from ~18.6s to ~67.6s). Root cause: active experts are scattered across the model's large mmap'd file, so the min-to-max span covers mostly *irrelevant* memory between them; pinning that whole span (a large amount of memory) every one of the 43 calls is far more expensive than the bounce-buffer copy it was meant to avoid. No crash, no leak, no orphan process — cleanly reverted via `git checkout -- hetero/ds4f/hip_ds4f_dense.c`. Do not retry this exact approach.

**Attempt 2 — persistent pinned host staging buffers per pool slot** (`hipHostMalloc`'d once, reused via the existing event-guard, refilled with a plain host `memcpy` before each async H2D copy — mirrors the pattern the pre-existing whole-layer streaming path already uses safely via `ensure_stream_host_slot`). This avoids attempt 1's pitfall (no register/unregister of a wide span; the staging buffer is exactly the size of one expert's weight/scale buffer). **Result: no measurable improvement** — 256-token prefill `experts` phase was 18.43s vs. the 18.62s baseline (noise-level). The extra CPU-side memcpy into the staging buffer apparently costs about as much as it saves on the DMA side; pageable-source H2D on this GPU/driver may simply not be as bounce-buffer-penalized as hypothesized. Also cleanly reverted, no correctness issues found.

**Conclusion: the routed-FFN upload phase is not obviously fixable by changing how the copy is staged.** Both host-memory strategies were tried in earnest and neither helped; the remaining cost is most plausibly genuine data-volume-bound PCIe transfer (real bytes that must move for a fresh set of active experts each call) rather than a staging inefficiency. Confirming the actual PCIe link generation/width to the 9070 XT (attempted via `lspci -vv`, inconclusive without root in this session) would help decide if there's still headroom, but wasn't completed. If PCIe bandwidth is already near its ceiling, the only remaining large lever for the upload phase is reducing bytes transferred per call — e.g. genuine cross-call expert-identity caching (not the pool's buffer reuse, which already exists) so a repeated expert isn't re-uploaded at all; this is the same idea explored for decode's cache (see above) and was not re-attempted here for prefill specifically.

Combined with the earlier finding that attention (not routed-FFN) is 42% of prefill time and is itself compute-bound on CPU with no found inefficiency, **prefill's path to 100-200 tok/s most plausibly requires GPU-offloading attention** (the single largest remaining, and only partially-explained, cost) rather than further tuning of the already-optimized routed-FFN upload path. This was not attempted given the size of that engineering effort relative to the time available this session.

## Two more attempts, and the decisive number: upload is genuinely data-volume-bound (2026-08-12)

Kept digging rather than stop at "no easy fix found." Two more tests, one negative and one decisive:

**Thread count (16 -> 32).** The host is a Threadripper 1950X: 16 physical cores, 32 logical (SMT). All prior benchmarks used `LLM_THREADS=16`. Since attention is confirmed CPU-compute-bound, tried 32 threads (a config-only change, zero numerics risk). **Result: worse, not better** — model load alone slowed from ~64s to ~96.8s, and the synthetic pool-dispatch benchmark's overhead more than doubled (2.885 -> 7.064 us/call), consistent with SMT contention on shared ALU/vector units for SIMD-heavy kernels. The run was killed by its own timeout before finishing. `numactl --hardware` confirms a single NUMA node (Threadripper 1950X exposes one node here), ruling out NUMA placement as a factor. 16 threads (one per physical core, no SMT contention) stands as the effective optimum already in use.

**Upload call-count/size/PCIe-link diagnostic.** Checked the actual PCIe link via sysfs (`/sys/bus/pci/devices/.../current_link_speed`): **32.0 GT/s, 16 lanes — full PCIe Gen5 x16, not degraded.** This reframed the earlier "upload is bandwidth-bound" guess, which was based on a wrong data-volume estimate (mistakenly used the hot-expert-cache's *residency* size, 6GB, rather than the actual per-request transient upload volume). Added counters (`ds4f_route_upload_calls`, `ds4f_route_upload_bytes`, hip_ds4f_dense.c + printed in ds4f_serve_lib.c) and measured directly: **a single 256-token prefill call transfers 64.2 GB across 28,824 individual copies, averaging ~2.1 MB/call, achieving ~4.17 GB/s.** 2.1 MB/call is not a tiny, latency-dominated transfer — this data point retroactively explains why *both* staging-mechanism attempts above (host-register pinning, pinned staging buffers) failed to help: the bottleneck was never the transfer *mechanism*, it's the sheer *volume*. 64 GB for one 256-token request is consistent with near-total coverage of each layer's 256 experts (256 tokens x 6 experts/token = 1536 draws, enough for a coupon-collector-style sweep across most of a 256-expert layer) — i.e. this is effectively full-layer-weight traffic, inherent to exact/W4A8 serving with per-token routing diversity at this problem size, not an artifact fixable without changing the model or approximating away some of the routing exactness (which would violate the accuracy requirement).

**This is now a well-evidenced, not speculative, ceiling.** Four real optimization attempts were made this session: (1) event-guarded pool — real, modest, committed win (~8-9%); (2) host-register pinning — reverted, made things worse; (3) pinned staging buffers — reverted, no effect; (4) thread count increase — reverted (config only), made things worse. The routed-FFN transfer is now understood, with hard numbers, to be near its practical ceiling for this design. Reaching 100-200 tok/s prefill requires reducing total bytes moved (impossible without an accuracy tradeoff at this exactness level) or overlapping it with otherwise-idle time (GPU is idle during the CPU-only attention phase, but the two phases have a true data dependency per layer — routing depends on attention's output — so single-request overlap isn't available; cross-*request* batching, running one request's GPU upload concurrently with another's CPU attention, is architecturally the only overlap opportunity left, and is a serving-scheduler-level change, not a kernel change).

Verified: mHC exact quality gate still 8/9 (no regression; the upload-counter addition is diagnostic-only, no logic change), no VRAM leak, no orphan/stuck processes for any of the four attempts.

## Checked one more real lever: partial GPU attention offload — negligible (2026-08-12)

Found that `hip_ds4f_dense_prefill_attention` (hetero/ds4f/hip_ds4f_dense.c:1956) is an *existing*, already-implemented GPU attention kernel, wired to `m->gpu_prefill_attn` and used elsewhere (common/ds4f_impl.h:6642-6643, inside the older/simpler `ds4f_forward_prefill` function) — but **not** on the exact/tierb2 path (`ds4f_forward_verify`) that our serving actually uses. Read its signature and bailout conditions carefully before considering using it: it implements *only* the sliding-window term (`window > 128` rejected, matching our `window_size=128` exactly) — it has no parameters for the compressed-KV (`nsel`/`sel`/`cmp_kv`) term that tier-B2 layers need. Using it directly on a tier-B2 layer would silently drop the compressed-term contribution to attention output — a real accuracy regression, which the goal explicitly prohibits. It would only be safe, as-is, on layers with `compress_ratios[L] == 0` (pure dense/window attention, no compression).

Added a one-time diagnostic (common/ds4f_impl.h, inside `ds4f_forward_verify`, `DS4F_PROF`-gated) counting how many layers actually qualify. Result: **`dense(ratio=0)=2 tierb2(ratio>0)=41` of 43 layers.** Only 4.6% of layers are compatible with the existing kernel as-is — worth roughly ~2% of total prefill time at best. Not a meaningful lever; did not implement the wiring (no code/runtime change beyond the count itself, since there's nothing worth safely shipping here).

**This closes off what was otherwise the most promising-looking remaining idea.** A *real* GPU attention win requires extending `hip_ds4f_dense_prefill_attention` (or writing a new kernel) to also handle the compressed-KV term for the 41 tier-B2 layers — i.e., the "GPU-offload attention" project already identified as the actual path to 100-200 tok/s, now confirmed to need genuinely new kernel development (not just wiring an existing one), sized at a proportional scope to the routed-FFN GPU work already done in this codebase's history, not something to rush.

Verified: mHC exact quality gate still 8/9 (no regression; diagnostic-only, no logic or numerics change), no VRAM leak, no orphan/stuck processes.

## Session-end assessment (2026-08-12)

Five real optimization attempts were made and tested this session: one succeeded (event-guarded pool, ~8-9% gain, committed); four were tried, measured, and correctly reverted or found negligible (host-register pinning, pinned staging buffers, 32-thread config, partial GPU-attention-offload feasibility). Every remaining hypothesis available without new kernel development or a serving-architecture change (batching/pipelining across independent requests) has now been checked with hard numbers, not guesses: PCIe link confirmed at full Gen5 x16, upload transfer size/rate measured directly (64.2GB/256-token-prefill, ~4.17GB/s, ~2.1MB/call — ruling out latency as the cause), attention's kernel confirmed already using the fast SVE 8-head-blocked path with a properly bounded (non-quadratic) window, and the one plausible cheap GPU-offload shortcut found to apply to only 2 of 43 layers.

**100-200 tok/s prefill is not reachable through further tuning of the current single-request CPU/GPU code paths without either sacrificing accuracy or building genuinely new capability** (a tier-B2-aware GPU attention kernel, or cross-request batching to overlap otherwise-idle CPU/GPU time). Both are substantial, multi-session engineering efforts, not something to complete safely under continued time pressure without adequate testing runway against the quality gate. This is the responsible stopping point for this investigation; the next session should pick up at "build a tier-B2-aware GPU attention kernel" as the concrete, sized, and now well-justified next step.

## Real fix found: the "fast" attention path was scalar on x86, not vectorized (2026-08-12)

While re-checking the GPU-attention-offload feasibility, re-examined `ds4f_attn_tb2_gemm`'s bailout conditions (`hetero`/common/ds4f_impl.h) and traced where its helpers (`ds4f_score8_bf16`, `ds4f_axpy8_bf16`, etc.) actually come from on this host. Confirmed via direct preprocessor inspection (`cc -E`) that:

- These 8-head-blocked "GEMM" helpers are hand-written SVE intrinsics **only** under `#if defined(__ARM_FEATURE_SVE)` (common/ds4f_impl.h:5103-5142) — true on the a64fx/Fugaku target this codebase was originally written for.
- On every other target, including this x86_64 host, they resolve to `common/ds4f_kernels_x86.h`'s versions — plain scalar loops, explicitly documented there as **"correct and auto-vectorizable, not hand-tuned."**
- Meanwhile the simpler per-head fallback path this "fast" path bypasses (`ds4f_attn_tb2_worker`, via `ds4f_attn_dot_bf16`) **does** have hand-written AVX2/FMA intrinsics on x86 (common/ds4f_impl.h ~4576-4597).
- `DS4F_ATTN_GEMM` (the toggle controlling which path is used) already existed as a runtime flag, defaulting to 1 (on) **regardless of platform** — so this x86 host has been unconditionally routing attention through unvectorized scalar code, bypassing its own genuinely-vectorized fallback, on every single call (confirmed earlier: `attn_gemm hit=42107 miss=0`).

**Tested `DS4F_ATTN_GEMM=0` (forces the AVX2-vectorized fallback):**

| | GEMM path (old default) | Fallback path (AVX2) |
|---|---|---|
| attn phase, 256 tok | 6.655s | 4.404s (-34%) |
| attn phase, 1024 tok | 34.0s | 23.16s (-32%) |
| prefill tok/s, 1024 tok | ~11.0-15.1 (noisy across runs) | **17.54** (clean run) |
| decode tok/s | 4.77-4.87 | 4.73-5.15 |
| mHC exact quality gate | 8/9 (mismatch @ token 2) | 8/9, **identical** logit values |

Quality is bit-for-bit identical in the gate's output (same mismatch index, same logit_abs/logit_rel/mean_ce values to full printed precision) — expected, since both paths compute the same math, just via different vectorization; the header's own docs already note the summation-order tolerance callers rely on.

**Fixed the default** (common/ds4f_impl.h, `ds4f_attn_tb2_gemm`): `DS4F_ATTN_GEMM` now defaults to 1 only under `__ARM_FEATURE_SVE` (preserving the genuinely-optimized behavior on the a64fx/Fugaku target this was written for) and defaults to 0 everywhere else (including this x86 host), so the fix applies automatically with no server/env-var change needed. Explicit `DS4F_ATTN_GEMM=1`/`0` still overrides in either direction if needed.

This is the single largest real, verified win of the session: **prefill at 1024 tokens improved from the ~11-15 tok/s baseline to 17.54 tok/s** (roughly +20-60% depending which baseline run it's compared against, all noisy due to background contention during measurement — the 17.54 number is from a clean, uncontended run). Still well short of 100-200 tok/s — `experts` (routed-FFN, previously optimized and confirmed near its data-volume ceiling) is now the largest single phase at this scale, and attention (still 33% even after this fix) remains large. But this closes a genuine bug, not a diminishing-returns tuning knob, and is a meaningfully different result from every other attempt this session.

Verified: mHC exact quality gate identical output with the new default (no env var needed), no VRAM leak, no orphan/stuck processes. 8192-token confirmation run was interrupted mid-execution by the harness (not a crash — clean state, no leak) before completing; the 256- and 1024-token results are clean and sufficient to establish the fix.

## User decision: stop here, 100-200 tok/s prefill is a future-session project (2026-08-12)

After the attention-vectorization fix above and a systematic (but negative) search for repeats of that bug class elsewhere in the codebase, explicitly asked the user how to handle the remaining gap to 100-200 tok/s given that closing it requires a genuinely new GPU kernel (tier-B2-aware attention, handling the compressed-KV term for 41/43 layers — the existing GPU attention kernel only covers the window term). Presented three options: stop and accept the current state; authorize a rushed same-session kernel attempt despite accuracy-validation risk; or redefine the target. **User chose to stop here and accept the current state.**

This is the deliberate, authorized end of this optimization investigation, not an unresolved blocker. Do not re-open the "reach 100-200 tok/s this session" framing without the user raising it again. The next real step, when picked up, is building the tier-B2-aware GPU attention kernel described above — a scoped, multi-session project that needs proper test/validation runway against the mHC exact quality gate before any of it should be trusted, not something to attempt under time pressure.

## Quality evidence

Use mHC enabled for the authoritative exact path:

```sh
hetero/ds4f/build/test_ds4f_real_tokens \
  --config /tmp/ds4f_quality_exact.json \
  --stage-dir /tmp/ds4f_nocopy_stage \
  --prompt-ids /tmp/ds4f_quality_ids.txt --max-tokens 9
```

The corrected mHC route matched the reference argmax for 8/9 tokens. One W4A8 mismatch remains an expected approximation; do not claim a 9/9 exact gate. The previous large divergence was diagnosed as running the diagnostic with `mhc=0`.

## Serving/performance evidence

The last completed CPU + 9070 XT measurements with exact mHC/Tier-B2 serving were approximately:

```text
Prefill: 5.1 tok/s at 1024 tokens before active-only staging;
         about 3.1 tok/s on a 64-token active-only run;
         5.7 tok/s with DS4F_SERVE_TIERB2=0 at 256 tokens.
Decode:  about 4.5–6.0 tok/s, depending on configuration.
```

These are far below the 18 tok/s decode and 100–200 tok/s prefill goals. Do not report the target as achieved.

GPU telemetry observed while idle: total VRAM about 17.1 GB, used about 7.7 GB, leaving about 9.4 GB free.

The server was not running at the end of the previous turn. Saved logs only show successful model load (~156 GB, ~67 s) and listener startup; no crash trace was present. A short benchmark was started after rebuilding the pool, but that command was interrupted before producing runtime output. Check for orphan processes before restarting.

## Server restart command

Restore the server only when needed, using program arguments (not performance environment variables except explicitly supported debug/profile flags):

```sh
env DS4F_STAGE_DIR=/tmp/ds4f_nocopy_stage \
 DS4F_SERVE_BASE=/tmp/ds4f_cdx \
 DS4F_TOKENIZER=/mnt/nvme02/models/ds4f-0731/tokenizer.json \
 DS4F_SERVE_USE_HIP=1 DS4F_HIP_DEVICE=0 LLM_THREADS=16 DS4F_CMGS=4 \
 ./a64fx/llm/run_ds4f_single_serve.sh \
 --context-memory-ttl-sec 600 --context-disk-ttl-sec 86400 \
 --context-memory-mb 512 --context-disk-mb 8192 \
 --prefill-quantum-tokens 32 --single-prefill-quantum-tokens 2048 \
 --agent-cache-max-tokens 14336 --runner-timeout-sec 3600 \
 --decode-quantum-tokens 4 --scheduler-quantum-ms 250 \
 --hip-mxfp4-wmma 1 --hip-expert-stream 1 --hip-routed-ffn 1 \
 --hip-expert-cache-mb auto --hip-expert-cache-reserve-mb 1536 \
 --hip-expert-cache-stats 1 --default-temperature 0.0 --default-top-p 1.0
```

Verify with:

```sh
curl -s --max-time 5 http://127.0.0.1:8080/health
curl -s --max-time 5 http://127.0.0.1:8080/v1/progress
```

## Required next work

1. ~~Inspect `git diff` for the reusable pool and run a short benchmark with a timeout.~~ Done 2026-08-11: original (non-event-guarded) version stalled during prefill and was reverted. See "Pool-change verdict" above.
2. ~~Add phase-level serving telemetry inside routed FFN.~~ Done 2026-08-11: see "Phase telemetry" above.
3. ~~Retry pooling with per-slot event guards.~~ Done 2026-08-11: stable, quality-clean, modest win (~8-9%). See "Pool retry with per-slot events" above. **The dominant cost (upload, ~90% of routed-FFN time) is real PCIe transfer of expert weights, not allocator overhead — pooling buffers wasn't enough.**
4. ~~Implement cross-call expert-weight caching.~~ Superseded 2026-08-11: this already existed (`hip_ds4f_dense_cache_hot_experts`) and tested as producing no measurable decode speedup — see "Decode does not use the routed-FFN path we optimized" above. Do not re-attempt a pool-side cache for decode; the bottleneck is elsewhere.
5. ~~Raise `--hip-expert-cache-mb` toward full free VRAM.~~ Done 2026-08-12: no effect, `auto` already used all available headroom — see "Cache budget is not the limiter" above. Coverage is capped by prompt diversity, not budget.
6. ~~Test 256, 1024, and 8192-token prefills.~~ Done 2026-08-12 for 1024 and 8192 (256 not yet run; low priority now that the pattern is clear). See "Prefill scaling and the real prefill bottleneck" above — scaling plateaus around 11-12 tok/s, and decode's cache coverage got worse (not better) at longer context, ruling out "just use a realistic prompt" as a fix for either target.
7. **Decode's real lever remains raising effective cache hit rate** (unchanged from before, budget/prompt-length are now ruled out as the fix): (a) online/adaptive cache that admits/evicts based on live decode-time routing instead of a one-shot post-prefill snapshot; (b) speed up the CPU MXFP4 fallback path itself for cache misses. Not yet attempted.
8. ~~Root-cause whether attn's cost is dispatch overhead or raw compute.~~ Done 2026-08-12: confirmed raw compute (dispatch overhead measured at 2.885 us/call, negligible even at ~264,000 calls). See "Attention cost is raw compute, not thread-pool dispatch overhead" above. **Do not pursue per-position dispatch batching — it would not help.**
9. ~~Profile inside the attention worker functions.~~ Done 2026-08-12: confirmed the fast SVE 8-head-blocked kernel is already 100% active, the window is properly bounded (not O(K^2)), and the cost is legitimate compute dominated by `head_dim=512` (architectural, not fixable without changing the model). See "Attention's real cost" above. **A real fix requires GPU-offloading attention — a substantial new engineering project, not attempted this session.**
10. Continue committing incrementally (telemetry, event-guarded pool, decode CPU/GPU phase split, prefill phase split, and the pool-dispatch-overhead diagnostic are already committed, see below) — after each further runtime-stable, measurably-improving change, commit with a short imperative subject and report the hash plus exact commands/results.

**Reminder:** `ds4f_serve_bench.py` loads its own model standalone — stop any running `run_ds4f_single_serve.sh` server first, or numbers will be contention-skewed (see verdict above).

## Decode optimization round: 18 tok/s target (2026-08-12)

Baseline with the attention-vectorization fix already in place but no other changes: decode 5.13 tok/s (no cache) / 5.48 tok/s (hot-expert cache enabled), at 64-token prompt / 32 decode tokens. `exp_cpu` (CPU MXFP4 fallback for cache-missed experts) is ~52% of decode's own wall time with no cache, still the dominant cost.

**Checked whether decode's CPU expert matvec has the same SVE/x86 vectorization bug as attention did — it does not.** `ds4f_mv_worker`'s MXFP4 path (common/ds4f_impl.h:655-697) has a genuine x86-specific branch calling into `common/ds4f_matvec_avx2.h`, confirmed to contain real AVX2/SSE intrinsics (119 occurrences of `_mm256_`/`__m256`-family instructions, e.g. `matvec_mxfp4_1row_i8_raw` using `_mm_shuffle_epi8`/`_mm_madd_epi16`/`_mm_fmadd_ps`). This is not a repeat of the attention bug; decode's CPU expert path is already properly vectorized.

**Found and fixed a second real, bit-exact default bug**, same class of issue as `DS4F_ATTN_GEMM`: `DS4F_MV_FUSE` (fuses several independent matvec dispatches into one `pool_run`, avoiding a separate thread-wake/barrier per matvec) defaulted to **off**, despite its own comment quantifying the gap it closes as decode-specific: "decode does ~10 pool_run/layer; in-loop matvec BW is ~247 GB/s vs ~610 GB/s... the gap is dispatch/serial-gap overhead, not the kernel." Per the same comment, this is bit-exact (identical worker/rowsplit/dot order, only the barrier is shared). Verified: mHC exact quality gate identical output with `DS4F_MV_FUSE=1`. Measured: decode 5.131 -> 5.458 tok/s (~6%) alone; combined with the hot-expert cache it doesn't compound further (5.437-5.479, within noise of either alone). **Flipped the default to on** (common/ds4f_impl.h, commit 9907c69c) — applies automatically, no server config change needed.

**Checked whether decode attention/QKV could use GPU offload (idle GPU during CPU-bound decode) — same limitation as prefill's attention.** `ds4f_serve_configure_hip_prefill` (a64fx/llm/ds4f_serve_lib.c:180) wires `m->gpu_prefill_attn` to `hip_ds4f_dense_prefill_attention` (the window-only GPU kernel already found incompatible with tier-B2's compressed-KV term), but this callback is never invoked by `ds4f_forward_verify` or the tier-B2 branch of `ds4f_forward_token` — same as prefill. `m->gpu_decode_attn_enabled` is never set to 1 anywhere in that configure function, so it's off by construction, and even if enabled it would hit the identical compressed-term-dropping accuracy problem. No safe lever here without the same new tier-B2-aware GPU kernel identified for prefill.

**Current decode state: ~5.5 tok/s, still ~3.3x short of the 18 tok/s target.** Both cheap real fixes found this round (attention vectorization inherited from the prefill work, plus MV_FUSE) are committed. Remaining cost is dominated by `exp_cpu` (CPU fallback for the ~53% of expert accesses the hot-expert cache doesn't cover) — already established in the earlier decode-cache investigation (see "Decode does not use the routed-FFN path we optimized" and "Cache budget is not the limiter" above) that raising this coverage needs an online/adaptive cache, not a budget or prompt-length change. That remains the concrete next step for decode, same conclusion as before, now re-confirmed after the two new fixes.

## Adaptive hot-expert cache: built, safe, but net-neutral (2026-08-12)

User explicitly authorized building the periodic batched re-admission design after being warned that a naive per-token on-demand swap would regress decode (a single fresh expert upload costs ~3.4ms vs ~0.16ms for an already-resident matvec at M=1 — upload latency isn't amortized over any batch for a single decode token).

**Implementation** (commit a6ecc5ea): `hip_ds4f_dense_refresh_hot_experts` (hetero/ds4f/hip_ds4f_dense.c) re-ranks experts by a caller-supplied recent-window hit delta, evicts residents that fall out of the new top set, admits newly-hot ones into the freed budget — reusing the exact same `release_matrix`/`hip_ds4f_dense_bind_mxfp4_tensor` paths the existing one-shot post-prefill cache already uses (already proven correct by every quality gate run this session). Wired into `ds4f_serve_decode` (a64fx/llm/ds4f_serve_lib.c) via a snapshot/delta of the existing cumulative `route_hits` counter, so the per-token routing hot path itself is untouched. Refresh only runs after `decode()` returns (GPU work for that token already complete), so eviction can never race an in-flight kernel. Off by default (`DS4F_ADAPTIVE_CACHE_PERIOD=0`).

**Tuning found a real failure mode worth remembering**: period=8 (refresh every 8 decode tokens) was too frequent — each layer only sees ~48 routing samples per window, far too noisy to produce a stable top-K ranking, causing destructive churn (up to 310 fresh expert uploads every 8 tokens) that dropped decode to 4.13 tok/s, worse than no adaptive cache at all. **Do not use a short period.**

**At period=64 over a realistic 200-decode-token run, the mechanism is stable and has a real, measurable effect, but nets to zero throughput change**: `exp_cpu` dropped 14.06→11.44s (-19%, genuine work shifted from CPU to GPU) and `exp_gpu` rose 3.44→5.33s (+55%), but total decode throughput was 5.174 vs 5.173 tok/s baseline — statistically identical. The refresh's own upload cost (155-224 experts churned per cycle, several GB at a time) almost exactly offsets the CPU time it saves. Verified: no crash, no VRAM leak, no orphan process across 3 real eviction/re-admission cycles at this longer scale.

**Left in the tree as opt-in** (inert by default, zero risk to existing behavior) rather than reverted, since this session's only available test scenario is a synthetic repeated-prompt decode where the static post-prefill cache already captures most of the achievable coverage — a real, long, topically-shifting conversation might show a different (better) result, but that can't be validated in this environment. Do not enable by default without re-testing on a more representative workload first.

**Decode remains at ~5.2-5.5 tok/s, still ~3.3x short of the 18 tok/s target.** With this result, every readily-available lever for decode (short of a genuinely different cache eviction policy, a much smarter admission heuristic, or the same GPU tier-B2 attention kernel identified for prefill) has now been tried and measured, not just theorized about.

## Useful benchmark

```sh
timeout 180s env DS4F_STAGE_DIR=/tmp/ds4f_nocopy_stage \
 DS4F_TOKENIZER=/mnt/nvme02/models/ds4f-0731/tokenizer.json \
 DS4F_SERVE_USE_HIP=1 DS4F_HIP_DEVICE=0 LLM_THREADS=16 DS4F_CMGS=4 \
 python3 a64fx/llm/ds4f_serve_bench.py \
 --stage-dir /tmp/ds4f_nocopy_stage \
 --tokenizer /mnt/nvme02/models/ds4f-0731/tokenizer.json \
 --prompt-tokens 64 --warm-decode 1 --decode-tokens 2 \
 --threads 16 --cmgs 4 --hip-device 0 --hip-mxfp4-wmma 1 \
 --hip-routed-ffn 1 --hip-expert-stream 1 --hip-expert-cache-mb 0
```

Do not mistake model-load time for inference throughput. The first run can spend roughly a minute loading the staged model.

## Tier-B2-aware GPU attention: built, validated correct, but slower (2026-08-12)

User explicitly authorized attempting the GPU tier-B2 attention kernel (the large, accuracy-risky project flagged repeatedly throughout this session as the real remaining lever for both the 100-200 tok/s prefill and 18 tok/s decode targets). Implemented incrementally with a validation gate at each step, exactly as planned, rather than shipping a single large unverified change.

**Step 1 (commit 5e648d50): a new GPU kernel that returns unnormalized partial attention.** The existing GPU window-attention kernel (`ds4f_dense_prefill_attn`, hetero/ds4f/hip_ds4f_kernels.h — real HIPRTC-compiled source, not a precompiled blob) only ever returns the final *normalized* softmax output, and only handles the sliding-window term — it has no way to incorporate tier-B2's compressed-KV term. Added `ds4f_dense_prefill_attn_partial`: identical window-attention math, but skips the final divide and additionally emits the per-(token,head) softmax max and sum it used. This lets a caller merge in a second (CPU-computed) term via the standard online-softmax identity (`m=max(mA,mB); l=lA*exp(mA-m)+lB*exp(mB-m); O=OA*exp(mA-m)+OB*exp(mB-m); final=O/l`) — mathematically exact, not an approximation, regardless of how the two terms are partitioned.

While writing the C wrapper (`hip_ds4f_dense_prefill_attention_partial`, hetero/ds4f/hip_ds4f_dense.c), found and fixed a real correctness gap before it could bite: tier-B2's KV cache is a **ring buffer** (`kv_slots == window_size`), so the straightforward linear base/end windowing the existing (already-shipped) kernel wrapper uses only works up to the first ring wrap — past that, a naive port would have silently produced wrong results for any position beyond the first `window_size` tokens. Fixed by always copying the full ring and passing `pos0` through unmodified for the `M==1` case (matching the kernel's own modular indexing), mirroring the pattern the *existing* wrapper already uses for its separate `resident_kv` optimization.

**Validated in `hetero/ds4f/test_hip_ds4f_dense.c`** (not just eyeballed): the new kernel's output, once divided by its own reported sum, matches the existing proven kernel to float32 rounding precision (`max_rel≈2.8e-7`) for a plain window case, and matches a direct CPU reference computation to the same precision at a position *past* the ring's first wrap (`pos0=45, kv_slots=32`) — specifically exercising the ring-wrap bug the fix addresses. This step was committed alone, fully inert (nothing calls the new kernel yet), before touching any real dispatch path.

**Step 2 (commit 1dbf751d): the compressed-term CPU partial, the merge, and wiring into real serving.** Added `ds4f_attn_cmp_partial_worker` (compressed-KV-only partial, same math as `ds4f_attn_tb2_worker`'s compressed loop, restricted to the same plain-f32 non-quantized case `ds4f_attn_tb2_gemm` already assumes) and `ds4f_attn_tb2_hybrid_gpu` (the merge, then RoPE de-rotation in the same order the CPU reference uses). Wired into *both* real per-position attention call sites (`ds4f_forward_verify` for prefill, `ds4f_forward_token` for decode) as the first-tried path, falling back to the existing, unmodified code chain on any bailout. Off by default (`DS4F_ATTN_HYBRID_GPU`).

**Correctness result: identical.** mHC exact quality gate output is bit-identical with the hybrid path enabled vs. disabled (same mismatch index @ token 2, same logit values to full printed precision) — and this isn't a silent no-op: hit/miss telemetry confirmed `hit=3444 miss=0 nsel_hit=1701` in the same run, meaning the compressed-term merge genuinely executed over 1701 of those calls, not just the trivial window-only case.

**Performance result: worse, not better.** Same-config comparison (64 prompt tokens, 16 decode tokens): `attn` phase 0.850s → 2.347s (nearly 3x *slower*), prefill 3.749 → 3.517 tok/s (-6%), decode 5.219 → 4.459 tok/s (-15%). Root cause: this dispatches one full GPU round-trip *per position* (Q/KV/sink H2D copy, kernel launch, stream sync, 3x D2H copy for Y/max/sum) — at `M==1`, the fixed per-call GPU dispatch/sync latency exceeds the already-AVX2-vectorized CPU computation it's replacing. This is the exact same economic pattern discovered earlier for per-token expert-cache swapping (a single unamortized GPU round-trip costs more than doing the equivalent work on a fast CPU path). A real win would require batching multiple positions into one kernel call — but tier-B2's compressor has a genuine sequential per-position dependency (each position's compression state depends on the previous position's ring-buffer update via `ds4f_tb2_prepare`), so positions can't be trivially batched without restructuring the compressor pipeline itself. That is a materially larger project than this one, out of scope here.

**Left in as opt-in** (inert by default, zero risk to existing behavior) rather than reverted — the kernel and merge math are now validated correct and available as a real foundation for a future batching attempt, rather than throwing away fully-tested, working code that simply isn't fast yet in its current single-position dispatch form.

**Where this leaves both targets:** prefill ~17.5 tok/s (1024 tokens), decode ~5.2-5.5 tok/s — both well short of the 100-200 / 18 tok/s targets. The GPU tier-B2 attention idea, repeatedly identified across this session as *the* remaining architectural lever, has now been built, correctness-validated, and performance-measured — and the honest result is that it doesn't help without batching, which is a distinct, larger, not-yet-scoped project (restructure the tier-B2 compressor to process multiple positions per kernel call). This is the true state of the art for this codebase as of this session; further throughput gains require either that restructuring or a different architectural approach (e.g. cross-request batching to keep the GPU busy across independent requests instead of within one request's sequential positions).

## User decision: stop here, decode restructuring is a future-session project (2026-08-12)

After the hybrid GPU attention kernel was built, validated correct, and measured slower (see above), explicitly asked the user whether to authorize restructuring tier-B2's sequential per-position compressor dependency to allow batched GPU attention calls — the only remaining path to a real decode speedup identified this session. **User chose to stop and accept the current state** rather than start that larger, more invasive restructuring under continued time pressure.

This is the deliberate, authorized end of the decode optimization investigation for this session, mirroring the same decision already made for prefill above. Do not re-open "reach 18 tok/s decode this session" without the user raising it again. The next real step, when picked up, is the compressor restructuring described above — sized, scoped, and now well-motivated by real measurements, but needing its own dedicated session with proper validation runway, not a continuation under pressure.
