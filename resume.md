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

## Current implementation (2026-09-12)

The approved **bounded-wave staging manager is implemented**. Do not restart
that implementation. Read `rdna4/llm/QWEN38_MOE_STAGING.md` for ownership and
commands, and the first section of `rdna4/llm/QWEN38_STATUS.md` for current
measurements. The earlier history below describes the preceding session.

- Two staging banks own pinned metadata, device metadata, quantized weights,
  Q8_0 scratch, and ready/done events. Host metadata reuse waits for DMA;
  device reuse waits for compute and promotion consumers.
- Direct batched cache misses and hits use independent per-slot fences,
  including Q8_0. Reset/mode changes/offload/free drain prefill streams.
- Fixed a prefill-to-decode map source lifetime bug: synchronize before free.
- `LLM_BENCH_WARMUP=0` now disables warmup. Optional paired
  `QWEN38_TARGET_EXPECTED_FIRST_TOKEN` / `QWEN38_TARGET_EXPECTED_HASH` enforce
  matched scalar references in the target benchmark.
- CPU delayed-copy/lifecycle/failure tests and small/large model-free GPU
  serial-oracle tests pass. Test allocation/error recovery, pinning, overlap,
  Q8 repack, promotions, and repeated bank reuse are covered.

**The full-model issue remains.** At 4096/64, 4000 MiB cache, BMAX4096,
512 MiB staging, warmup off, the first five eight-repeat runs returned the common
batched hash `afdf60ceeb4f0103` in 7/8 (pageable serial), 8/8 (pageable overlap),
5/8 (pinned serial), 7/8 (pinned overlap), and 6/8 (fresh pinned overlap).
A fresh pageable-overlap process also passed 8/8 (16/16 across its two
processes). Pinned overlap measured about 132 prefill / 20 decode tok/s,
peak 14296 MiB.
No staged default is promoted; scalar F16 remains the fallback. Neither the
200/30 performance target nor full-model determinism has been achieved.
The ownership tests alone do not establish where the remaining bug is.

All five matched scalar corpus cases fail staged parity, although each
scalar and staged sequence repeats 2/2. The fresh `scalar-exact` 4K reference
is first token 15 / hash `e3d8bf6d47dc6cc3` (10.83/15.45 tok/s); its difference
from the historical larger-cache `fast` reference is not explained. Full
results are recorded in `QWEN38_STATUS.md` after the matrix. Reproduce with `make -C rdna4/llm moe-stage-quality`. Its source fixes
the exact header/coding/arithmetic/prose/Japanese prompts and generates fresh
references; any mismatch fails the command. Logs remain under
`rdna4/llm/tmp/staging_quality/`.

### Next investigation

Localize the first diverging layer without adding host synchronization:
collect device fingerprints of layer input, routing/assignment buffers,
routed output, shared output, and layer output asynchronously, then read them
at an existing end-of-request barrier. Host-synchronized layer tracing can hide
the timing-sensitive failure. Compare good/bad repeats before changing math.
Do not assert that ownership fencing fixes all races or blame the driver
without evidence. GPU assignment grouping uses integer atomics; ordering alone
is not proof of numerical divergence. Scalar fused routing and batched routing
have separate diagnostic switches, so disabling one does not disable both.

### Working environment and checks

Use repository-local temporary storage (`TMPDIR=$PWD/rdna4/llm/tmp`); never
`/tmp`. `/local` was absent. The ordinary sandbox hides GPU devices; approved
host execution (`require_escalated`) exposes `/dev/kfd` and RX 9070 XT. Check
GPU exclusivity and do not overlap benchmark processes.

```sh
export TMPDIR="$PWD/rdna4/llm/tmp"
make -C rdna4/llm moe-stage-test
make -C rdna4/llm test_hip_llm tmp/test_hip_qwen4_moe_stage tmp/test_hip_qwen4_moe_stage_large
make -C rdna4/llm moe-stage-gpu-test
timeout --foreground 180s ./rdna4/llm/tmp/test_hip_qwen4_moe_stage_large
make -C rdna4/llm moe-stage-quality
```

Preserve unrelated work under `a64fx/glm5` and `common/transformer.h`. Do not
commit the stray untracked `rdna4/llm/hip_runner_common.h`; the runner uses
`../hip_runner_common.h`. Do not push without a new explicit push request.

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

## Historical evidence and remaining work

Read `rdna4/llm/QWEN38_STATUS.md` in full for preceding investigations and
measurements. Historical scalar `fast` at BMAX2048/7800 MiB cache returned
first token 30 / hash `6d67721190bdaa83` in three repeats at about 24/21 tok/s.
Do not substitute it for fresh references with different profile settings.

Earlier per-token-MoE controls passed 6/6 at 512/8. Disabling the gathered
input/output alias, host router top-k, pageable weights, and per-copy stream
synchronization individually did not eliminate full-model divergence. Launch
blocking or host-synchronized layer tracing can mask the failure. These tests
narrow the investigation but do not prove a unique root cause.

The previous request to stage all cold experts before compute was superseded
by the user's explicit choice of bounded waves. The implementation preserves
the 512 MiB pool; full-layer weight staging is not the pending task.

After localization and strict scalar parity, investigate routed cache hit
rate/overlap for prefill and exact single-token kernels for decode. Stateful
multi-chunk prefill, diverse 32K+ quality checks, and the 256K capacity path
remain separate open work. No MTP or approximation promotion is authorized by
this staging milestone.
