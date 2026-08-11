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
4. Implement cross-call expert-weight caching keyed by expert identity (skip the H2D copy on a cache hit), with an explicit VRAM budget and eviction policy — this is the next real lever per the finding in step 3. Not started. Validate incrementally under `DS4F_PROF=1`; do not attempt full-layer resident uploads (a separate prior attempt at that stalled).
5. Test 256, 1024, and 8192-token prefills (quality gate for the pool retry already passed at 9-token scale). Watch VRAM and abort unsafe runs.
6. Benchmark warmed single decode separately from prefill. Prefix/tool caching is implemented in the server but has not improved throughput; verify cache hit, prompt token count, and processing phase through `/v1/progress` before tuning residency.
7. Continue committing incrementally (telemetry + event-guarded pool are already committed, see below) — after each further runtime-stable, measurably-improving change, commit with a short imperative subject and report the hash plus exact commands/results.

**Reminder:** `ds4f_serve_bench.py` loads its own model standalone — stop any running `run_ds4f_single_serve.sh` server first, or numbers will be contention-skewed (see verdict above).

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
