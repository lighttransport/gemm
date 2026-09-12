# Qwen3.8-Flash-Next RDNA4 runner — resume handoff

Worktree: `/mnt/nvme02/work/gemm/main`

## Objective

Stable, quality-safe Qwen3.8-Flash-Next LLM runner on CPU + Radeon RX 9070 XT
(gfx1201, 16 GiB), for `rdna4/llm/`.

- Quality bar: **bit-exact F16 greedy-hash parity** (identical first token +
  full sequence hash vs the scalar F16 reference) on a diverse prompt set.
- Performance targets: **prefill >= 200 tok/s** and **single-stream decode
  >= 30 tok/s**.
- Decode route: exact single-token decode (no MTP).
- Prefill route: fix and promote the batched dispatcher; scalar is the
  quality-safe fallback.

Do not push to any remote without explicit per-action user permission.
Committing freely is allowed once a coherent unit is done.

## Current status (2026-09-12)

The performance targets are **not reached**. The scalar route is deterministic
but slow; the batched route is much faster but has a **rare residual prefill
nondeterminism** (~1 in 8 repeats), so it is best-effort/diagnostic only.

Measured on the RX 9070 XT with the real 9,000-byte `common/gguf_loader.h`
prompt (`rdna4/llm/tmp/qwen38_target_prompt.txt`), 4,096 prefill / 64 decode,
BMAX=4096, 5,000-MiB cache, `LLM_QWEN4_BATCH=1`:

| Profile | Prefill tok/s | Decode tok/s | Repeatability | Hash |
| --- | ---: | ---: | --- | --- |
| scalar `fast` (quality-safe) | ~24 | ~21 | deterministic | `6d67721190bdaa83` |
| `batch4k` pageable + direct copies | 125 (median) | 19.6 | 7/8 | `afdf60ceeb4f0103` |
| `batch4k` pinned host + async pipeline | 149 (median) | 21.4 | ~1/8 divergence | `afdf60ceeb4f0103` |
| `batch4k-stage` (grouped cold, pageable) | ~119 | 17.7 | deterministic | `afdf60ceeb4f0103` |

Batched-prefill cold-expert traffic is the wall: only ~25-33% routed-expert
cache hits and ~90-124 GiB H2D at 2K-4K, because the 512-expert x 48-layer
working set is far larger than any cache that fits beside the BMAX=4096
scratch.

The scalar `fast` route is the only quality-safe deterministic default today.
At 4K it is only ~24 prefill / ~21 decode, so neither target is met.

## What was fixed this session (all committed)

Key commits (newest first):

- `6b31c751` deterministic single-thread `qwen4_renorm_resident_weights`
  (decode-path expert-weight renorm used order-dependent `atomicAdd`).
- `5fdf98e8` `LLM_QWEN4_MOE_EOUT_ALIAS` diagnostic; ruled out the
  `d_moe_eout`/gather alias as the sole residual.
- `62904384` direct cold-expert copies ordered via `moe_copy_stream` +
  `hipEventRecord`/`hipStreamWaitEvent` (a plain `hipMemcpyAsync(..., r->stream)`
  did not reliably order on this ROCm stack).
- `96dc1f95` diagnosed the cold-expert H2D ordering.
- `b933fde2` isolated the residual to `forward_moe_ffn_batched` (per-token MoE
  is deterministic 6/6; batched attention/SSM are fine).
- `058d754a` **stream-ordered per-row position publication** — the biggest fix.
  The forced-scalar layer-1 loop published `r->d_position` with a blocking
  `hipMemcpy` on the null stream once per row, racing `r->stream` kernels.
  Now `hipMemcpyAsync` from a precomputed host array `h_pos_batch`.
- `f2ee146a`, `3b708af8` defaulted `batch4k` to pageable + direct copies and
  corrected determinism claims.
- `8f96e6a8`, `411cc2b8` localized/recorded the residual.
- `6b204f3e`, `a0998941` staged grouped-prefill metadata ordering and its
  residual staging-bank race.
- `628d2663` grouped routed-expert investigation: grouped-BF16-WMMA is
  memory-infeasible (ne*N*K bf16 ~= 15 GiB for 512 experts); grouped
  resident-only gives no gain; staged grouped cold experts help only at
  register=0.
- `70c5a974`, `e6919a79`, `21273851` deterministic single-dispatch 4K profile,
  async-pipeline request isolation, tuning limits.
- `3b228443` multi-chunk state-carry localization (L23 `attn_out`).
- `7d31bc5f` ordered MoE combine (`moe_scatter_accum_ordered` +
  `d_moe_assign_pos`), synchronous CPU-result publication.
- `9e290422` repeatability gate (`test_hip_llm --bench-repeat`,
  `bench_qwen38_target.sh`, 256K/sub-32K gates, `target-gate*`) plus the
  `QWEN38_FAST_PREFILL` precedence fix.

Real bugs fixed: ordered K-expert combine, CPU result publication race,
per-row position null-stream race, cold-expert H2D ordering, staged metadata
ordering, decode renorm atomic order, expert-cache reset between requests.

## The remaining blocker

A **rare residual prefill nondeterminism in the batched MoE dispatcher**.
Evidence and negative results:

- Per-token MoE (`LLM_MOE_PREFILL_SCALAR=1`) is deterministic 6/6 at 512/8, so
  the race is in `forward_moe_ffn_batched`.
- The batched MoE kernels have no atomics, so it is a data-ordering/lifetime
  issue.
- With `--decode 4` the **prefill first token itself varies** (16 / 289 / 9616 /
  37700), so it is not a decode-only issue.
- Disabling the `d_moe_eout` alias, using host router top-k, using pageable
  host weights, and a full `hipStreamSynchronize` after each cold copy all
  still diverge, so no single knob fixes it.
- `HIP_LAUNCH_BLOCKING=1` largely hides it (steady repeats match), confirming an
  ordering bug, not arithmetic.
- A 4-repeat `LLM_DEBUG_LAYERS=1` trace at 1024 does **not** reproduce it (the
  per-stage sync perturbs timing), so it could not be localized to one stage.

Recommended next approach (not yet implemented): a **device-side expert-cache
manager** that eliminates host memcpys during a layer — stage all cold experts
into cache/staging, synchronize once (event), then compute. This removes the
whole host-copy ordering class rather than patching individual copies.

## TODO / next steps, priority order

1. Implement the device-side cold-expert staging manager to remove the
   residual batched-MoE race, then re-run the repeatability gate.
2. Make the async copy pipeline and pinned-host paths repeatable (they reach
   149 prefill / 21 decode but race).
3. Push prefill toward 200: raise routed-expert cache hit rate / overlap.
   Grouped-BF16-WMMA is memory-infeasible at 512 experts; grouped native
   resident-only does not help because each layer's cache starts empty.
4. Push exact single-token decode from ~20 to 30 tok/s (cache sizing, SSM/MoE
   kernel work). Larger cache is VRAM-limited by the BMAX=4096 scratch.
5. Multi-chunk stateful batching (`prefill > BMAX`) still diverges (inter-chunk
   carry, first seen at L23 `attn_out`); needed for >4K prompts.
6. Quality-gated 32K+ prompt workload and 256K capacity path remain open.

## Authoritative paths

```text
Model:      /mnt/nvme01/models/q38nf/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf
MTP:        /mnt/nvme01/models/q38nf/mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf
CPU lib:    /mnt/nvme02/work/llama.cpp/build-codex-hetero-dev2/bin/libggml-cpu.so.0.22.0
Prompt:     rdna4/llm/tmp/qwen38_target_prompt.txt  (9000 bytes of common/gguf_loader.h)
Status doc: rdna4/llm/QWEN38_STATUS.md              (authoritative running log)
Tuning doc: rdna4/llm/QWEN38_PREFILL_TUNING.md
MTP doc:    rdna4/llm/QWEN4_MTP.md
```

## Build and test

```sh
# Build the runner.
make -C rdna4/llm -j8 test_hip_llm

# Static profile/regression tests (no GPU needed).
bash rdna4/llm/test_qwen38_profiles.sh
```

Pre-existing warning noise in `common/gguf_loader.h` / `hip_runner_common.h`;
no errors expected.

## Repeatability gate (the key tool)

```sh
# Deterministic single-dispatch 4K profile (pageable + direct copies).
QWEN38_TARGET_PROFILE=batch4k \
  QWEN38_TARGET_PREFILL=4096 QWEN38_TARGET_DECODE=64 QWEN38_TARGET_CONTEXT=8192 \
  QWEN38_TARGET_REPEATS=8 \
  QWEN38_TARGET_PROMPT_FILE=rdna4/llm/tmp/qwen38_target_prompt.txt \
  QWEN38_TARGET_LOG=rdna4/llm/tmp/qwen38_batch4k.log \
  ./rdna4/llm/bench_qwen38_target.sh
```

`make -C rdna4/llm target-gate target-gate-fast target-gate-batch target-gate-4k`
wrap the same. The gate fails unless every repeat has the same first token and
the same full sequence hash, and reports min/median prefill/decode/e2e tok/s,
peak VRAM, and `rocm-smi` clocks/temps.

Useful env switches (all in `bench_qwen38_target.sh` dry-run output):

```sh
QWEN38_DRY_RUN=1 QWEN38_TARGET_PROFILE=batch4k ./rdna4/llm/bench_qwen38_target.sh
# Profiles: scalar-exact | fast | batch | batch-cpu | batch4k | batch4k-stage | approx
LLM_QWEN4_REGISTER_HOST=0|1      # pageable vs pinned host expert weights
LLM_MOE_COPY_PIPELINE=0|1        # direct vs async expert uploads
LLM_QWEN4_MOE_EOUT_ALIAS=0|1     # disable the gathered-input/output alias
LLM_QWEN4_PREFILL_GPU_TOPK=0|1   # host vs GPU router top-k
LLM_QWEN4_BATCH_MULTI_CHUNK_FORCE=1 LLM_QWEN4_BATCH_STATEFUL=1
LLM_MOE_PREFILL_SCALAR=1         # per-token MoE (deterministic, slow)
LLM_DEBUG_LAYERS=1               # per-layer full-batch bitwise FNV trace
```

## Debug workflow

```sh
# Full-batch per-layer bitwise trace, 2-4 repeats, then diff repeats.
LLM_DEBUG_LAYERS=1 QWEN38_TARGET_PROFILE=batch4k \
  QWEN38_TARGET_PREFILL=1024 QWEN38_TARGET_DECODE=1 QWEN38_TARGET_REPEATS=4 \
  QWEN38_TARGET_PROMPT_FILE=rdna4/llm/tmp/qwen38_target_prompt.txt \
  QWEN38_TARGET_LOG=rdna4/llm/tmp/qwen38_trace.log \
  ./rdna4/llm/bench_qwen38_target.sh
# Then split by "=== Bench repeat k/N ===" and compare the `[L..]` lines.
```

Note: the per-stage `hipStreamSynchronize` in the trace perturbs timing and can
mask the rare race; use it to localize deterministic divergences, and use the
plain gate for repeatability.

## Environment / safety

- AMD device access is available in this session (`/dev/kfd`, `/dev/dri/renderD*`,
  render/video groups). Check exclusivity with `fuser /dev/kfd`; `rocm-smi
  --showmeminfo vram` to gauge peak. A standalone benchmark loads the full
  model, so do not run concurrently with a server.
- gfx1201 has reset under grouped full-depth prefill in the past; use isolated
  runs and a bounded timeout.
- Do not `git push` without explicit permission. Do not commit the stray
  untracked `rdna4/llm/hip_runner_common.h` or `rdna4/llm/tmp/` logs.

## Resuming prompt

> Continue the Qwen3.8-Flash-Next RDNA4 runner work in `/mnt/nvme02/work/gemm/main`.
> Objective: bit-exact F16 greedy-hash parity with prefill >= 200 tok/s and
> single-token decode >= 30 tok/s on CPU + RX 9070 XT (16 GiB). Current state:
> the scalar `fast` route is deterministic but only ~24 prefill / ~21 decode at
> 4K; the batched `batch4k` route reaches ~125-149 prefill / ~20 decode
> (hash `afdf60ceeb4f0103`) but still has a rare (~1 in 8) prefill
> nondeterminism inside `forward_moe_ffn_batched`, so it is best-effort only.
> This session fixed the ordered K-expert combine, CPU-result publication, the
> null-stream per-row position race, the cold-expert H2D ordering, staged
> metadata ordering, the decode renorm atomic order, and per-request
> expert-cache reset (see `rdna4/llm/QWEN38_STATUS.md` and the commit log).
> The confirmed next step is a device-side cold-expert staging manager that
> eliminates host memcpys during a layer (stage cold experts, one event sync,
> then compute) to remove the residual ordering class, then re-run
> `bench_qwen38_target.sh` with `QWEN38_TARGET_REPEATS=8`. Read
> `rdna4/llm/QWEN38_STATUS.md` in full before starting. Build with
> `make -C rdna4/llm -j8 test_hip_llm`; validate with
> `bash rdna4/llm/test_qwen38_profiles.sh` and the repeatability gate; check
> `fuser /dev/kfd` and `rocm-smi --showmeminfo vram` before/after heavy runs;
> use incremental build-then-validate-then-commit. Do not push without
> explicit permission.
