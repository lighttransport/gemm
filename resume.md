# DS4F resume handoff

Worktree: `/mnt/nvme02/work/gemm/ds4f`

## Objective

Serving target on CPU + Radeon 9070 XT, exact/mHC/tier-B2 quality path:

- preserve output quality first (mHC exact quality gate, see below);
- single-stream decode around 18 tok/s;
- prefill around 100-200 tok/s for 1K+ input;
- prefix/system/tool-token caching for coding-agent requests;
- harden long-context and multi-context operation.

Do not redefine success around the current benchmark numbers. Do not push to any remote without explicit user permission.

## Current status (2026-08-12)

**Prefill: ~17.5 tok/s at 1024 tokens** (up from a ~11-15 tok/s baseline this session via a real bug fix). **Decode: ~5.2-5.5 tok/s** (up from ~5.1 baseline). Both still well short of the 100-200 / 18 tok/s targets. Quality gate holds at 8/9 argmax match throughout (one expected W4A8 approximation mismatch at token index 2 — not a regression, do not chase 9/9).

This session made two genuine, verified, quality-neutral fixes plus a large validated-but-not-yet-beneficial GPU attention kernel:

1. **`DS4F_ATTN_GEMM` platform-default bug (commit `b804fecd`)** — the "fast" 8-head-blocked attention kernel was SVE-only-optimized; on this x86 host it silently fell back to unvectorized scalar code (`common/ds4f_kernels_x86.h`) while bypassing its own genuinely-AVX2-vectorized simpler path. Fixed the default to be platform-conditional. **+~32-34% on the attn phase, prefill baseline 11-15 -> 17.5 tok/s.** This was the single biggest win this session.
2. **`DS4F_MV_FUSE` default bug (commit `9907c69c`)** — batches independent matvec dispatches into one thread-pool barrier instead of one-per-matvec; was off by default despite the code's own comment quantifying the gap as decode-specific. Bit-exact. **+~6% decode.**
3. **Tier-B2-aware GPU attention kernel (commits `5e648d50`, `1dbf751d`)** — built and validated *correct* (mHC quality gate bit-identical, unit-tested including a ring-buffer wraparound edge case found and fixed along the way), but measured **slower** (attn phase ~3x slower, prefill -6%, decode -15%) because it dispatches one GPU round-trip per single sequential position, and that per-call latency exceeds the CPU AVX2 path it replaces. Left in the tree as **opt-in, off by default** (`DS4F_ATTN_HYBRID_GPU=0`) — inert, zero risk, but a real foundation for a future batched-attention rewrite.
4. **Adaptive hot-expert cache (commit `a6ecc5ea`)** — periodic batched re-admission of GPU-resident experts based on a live decode-time routing window. Built, safe, real effect (shifts load from CPU to GPU), but **net-zero throughput** (refresh's own upload cost offsets the CPU time saved). Left in as **opt-in, off by default** (`DS4F_ADAPTIVE_CACHE_PERIOD=0`).

Everything is committed. Working tree is clean except two untracked runtime log files (`a64fx/llm/ds4f_frontend.log`, `a64fx/llm/ds4f_runner.log` — do not commit these). No server or benchmark process is currently running.

## Why the targets aren't reached (the real bottleneck, confirmed with hard numbers)

- **Prefill's `experts` phase (routed-FFN) is data-volume-bound, not a staging inefficiency.** A single 256-token prefill call transfers ~64GB across ~28,824 individual H2D copies at ~4.17 GB/s, on a confirmed-full-speed PCIe Gen5 x16 link. Two different host-memory staging strategies (full-span `hipHostRegister`, persistent pinned staging buffers) were tried and both failed to help — the bottleneck is genuine per-request data volume (near-full per-layer expert-weight coverage from diverse per-token routing), not the transfer mechanism. Not fixable without either violating the accuracy requirement (approximating routing) or reducing repeat uploads via genuine cross-request/cross-call expert-identity caching (see the adaptive-cache result above, which already tried this for decode and found it net-neutral).
- **Attention (both prefill and decode) is legitimately compute-bound**, using an already-optimized, now-correctly-vectorized SVE/AVX2 kernel, with a properly bounded (not O(K^2)) sliding window. The only remaining lever is GPU offload, which now exists (Step 3 above) but is slower in its current single-position dispatch form.
- **The one architectural path to a real win for both targets**: restructure tier-B2's compressor to allow **batching multiple sequential positions into one GPU attention kernel call**, instead of the current one-position-at-a-time dispatch. This requires touching `ds4f_tb2_prepare`'s sequential per-position ring-buffer dependency (each position's compression state depends on the previous position's update) — a materially larger, not-yet-scoped restructuring project. This is the concrete next step, explicitly deferred by user decision twice this session (once for prefill, once for decode) rather than rushed under time pressure.
- **Alternative architectural path**: cross-request batching (keep the GPU busy with one request's routed-FFN upload while another request's CPU attention runs) — not explored this session, would need serving-scheduler-level changes, not kernel changes.

## TODO / next steps, in priority order

1. **Scope and implement tier-B2 compressor restructuring for batched GPU attention.** This is the one remaining lever with real expected payoff for both prefill and decode. Needs its own dedicated session with proper validation runway (unit tests + the mHC exact quality gate at every step, exactly as done for the single-position kernel this session) — do not rush this under time pressure. Once positions can be batched, re-enable and re-benchmark the `DS4F_ATTN_HYBRID_GPU` path built this session (already correct, just needs a batched caller).
2. If pursuing the routed-FFN data-volume problem further: investigate genuine cross-request expert-identity caching (distinct from the adaptive intra-request cache already tried and found net-neutral) or reducing bytes moved via a different serving pattern.
3. Re-run 8192-token prefill scaling once the above changes land (last full run was interrupted mid-execution by the harness, not a crash — 256/1024-token results are the reliable current baseline).
4. Prefix/system/tool-token caching for coding-agent requests, and long-context/multi-context hardening — not investigated this session at all; still open from the original objective.

## Resuming prompt

> Continue DS4F serving optimization on CPU + Radeon 9070 XT. Current state: prefill ~17.5 tok/s @ 1024 tokens, decode ~5.2-5.5 tok/s, both short of the 100-200 / 18 tok/s targets, quality gate holding at 8/9 (expected). This session fixed two real platform-default bugs (`DS4F_ATTN_GEMM`, `DS4F_MV_FUSE` — see resume.md) and built a validated-correct-but-not-yet-fast GPU tier-B2 attention kernel (opt-in, `DS4F_ATTN_HYBRID_GPU`). The confirmed next step is restructuring the tier-B2 compressor (`ds4f_tb2_prepare`) to allow batching multiple sequential positions into one GPU attention call — the current per-position dispatch has too much fixed GPU round-trip latency to win over the CPU path. Read resume.md in full before starting. Validate every change against the mHC exact quality gate (`hetero/ds4f/build/test_ds4f_real_tokens --config /tmp/ds4f_quality_exact.json --stage-dir /tmp/ds4f_nocopy_stage --prompt-ids /tmp/ds4f_quality_ids.txt --max-tokens 9`, expect 8/9) and check for orphan processes / VRAM leaks (`rocm-smi --showmeminfo vram`) before and after every risky test. Do not rush unvalidated numeric kernel changes — this session's pattern of incremental build-then-validate-then-commit worked well; keep using it.

## Authoritative model and staging

```text
GGUF: /mnt/nvme02/models/ds4f-0731/DeepSeek-V4-Flash-MXFP4Experts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2-mxfp4-0731.gguf
Stage: /tmp/ds4f_nocopy_stage
Tokenizer: /mnt/nvme02/models/ds4f-0731/tokenizer.json
Quality prompt IDs: /tmp/ds4f_quality_ids.txt
Quality config: /tmp/ds4f_quality_exact.json
```

## Quality gate (run after every change touching numerics)

```sh
hetero/ds4f/build/test_ds4f_real_tokens \
  --config /tmp/ds4f_quality_exact.json \
  --stage-dir /tmp/ds4f_nocopy_stage \
  --prompt-ids /tmp/ds4f_quality_ids.txt --max-tokens 9
```

Expect 8/9 argmax match (one W4A8 approximation mismatch at token index 2 is the known-expected baseline — not a regression). Never claim 9/9.

## Build

```sh
make -C hetero/ds4f -j4
sh a64fx/llm/build_ds4f_serve.sh
```

Pre-existing warning noise, no errors expected.

## Benchmark (standalone — do not run concurrently with a live server; it loads its own full model and the two will contend for CPU/memory)

```sh
timeout 180s env DS4F_STAGE_DIR=/tmp/ds4f_nocopy_stage \
 DS4F_TOKENIZER=/mnt/nvme02/models/ds4f-0731/tokenizer.json \
 DS4F_SERVE_USE_HIP=1 DS4F_HIP_DEVICE=0 LLM_THREADS=16 DS4F_CMGS=4 DS4F_PROF=1 \
 python3 a64fx/llm/ds4f_serve_bench.py \
 --stage-dir /tmp/ds4f_nocopy_stage \
 --tokenizer /mnt/nvme02/models/ds4f-0731/tokenizer.json \
 --prompt-tokens 1024 --warm-decode 4 --decode-tokens 32 \
 --threads 16 --cmgs 4 --hip-device 0 --hip-mxfp4-wmma 1 \
 --hip-routed-ffn 1 --hip-expert-stream 1 --hip-expert-cache-mb 0
```

Model load takes ~60-70s (156GB staged model) — don't mistake load time for inference throughput. `--hip-expert-cache-mb -1` (or `auto` via the server) enables the static post-prefill hot-expert cache; `DS4F_ADAPTIVE_CACHE_PERIOD=64` additionally enables the (currently net-neutral) periodic adaptive refresh; `DS4F_ATTN_HYBRID_GPU=1` enables the (currently slower) GPU tier-B2 attention path. All default off/safe.

## Server restart command (only when needed)

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

Check for orphan processes (`ps aux | grep ds4f`) before restarting.
