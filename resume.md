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

## Latest parity milestone

Native HC + SSM projections + router/shared experts now pass fresh scalar
first-token/full-hash parity on all four128/16 prompts, two requests each
(8/8). `LLM_QWEN4_BATCH_MOE_NATIVE=1` removes BF16 activation rounding
from router/shared batching. Its real-weight `--verify-moe-native` oracle
passes48 layers x8 rows bitwise. Binary:`tmp/test_hip_llm_nativemoe`.
Logs:`tmp/nativemoe_oracle.log`, `tmp/nativemoe_quality_summary.log`.

A fresh scalar4K/64 reference followed by two fully native staged requests
is running via `tmp/run_native_all_4k.sh`. Logs:
`tmp/native_all_4k_summary.log`, `tmp/native_scalar_4k_reference.log`,
`tmp/native_all_4k.log`. Do not substitute the old601167e3b2fb9425 staged
hash for this scalar reference. All200/30 targets remain required.

## Current implementation (2026-09-12 continuation)

The user explicitly asked to keep going until the **200/30** targets are met.
Neither target nor fresh scalar F16 corpus parity is achieved yet. Keep the
scalar fallback as the default. Do not restart the owned staging manager
(committed as `74a68bf2`); details and current measurements are at the top of
`rdna4/llm/QWEN38_STATUS.md`.

Latest follow-up: the corrected-route API profile is in
`rdna4/llm/tmp/rocprof_qwen_api/`. Decode copies30.48 GiB in2.676 s
over64 tokens; kernel time2.200 s. Launch geometry and attention sharding
do not establish a speed gain. Exact-prefix graph replay passes two4K/64
staged hashes but measures only10.87/11.12 decode min/median. Fresh four-prompt
scalar references all disagree with staged prefill; see the new status table.
Native HC batching now passes the real-model bitwise oracle for48 layers,
both phases and8 rows, including F16 injection. Its corpus check fails all four prompts; see
`tmp/nativehc_quality_summary.log`. Do not claim full-model parity from the
HC oracle. Q8 SSM native batching is now implemented:
`LLM_QWEN4_BATCH_SSM_NATIVE=1`, binary `tmp/test_hip_llm_nativessm`.
`--verify-ssm-projections` passes 36 layers x 5 projections x 8 rows bitwise.
Combined HC+SSM still fails all four corpus cases; see
`tmp/nativessm_quality_summary.log`. Next numerical candidates are BF16
router/shared-expert batching. No production default is promoted.

Final validated binary: `rdna4/llm/tmp/test_hip_llm_verified`. Its graph
check captures47 prefixes with zero failures and matches two4K/64 staged
hashes, at170.00/170.42 prefill and13.57/13.59 decode min/median.
Log:`tmp/verified_graph_4k.log`. No GPU jobs remain from this checkpoint.

New work in this continuation:

- Fixed a demonstrated GPU top-K candidate-masking bug. Expanded tests fail
  before the fix and pass afterward. Historical `afdf60ceeb4f0103` performance
  used incorrect routing and is not a current quality baseline.
- Asynchronous fingerprints localized repeat divergence to attention layers
  3/39, before FFN routing. Fixed the shared maximum-buffer reader race in
  F16/I8 prefill/decode attention. Delaying other waves reproduces errors up
  to 0.056 without the barrier; all four corrected tests pass below 6e-8.
- `LLM_QWEN4_BATCH_PLE_FFN=1`: keep layer-1 PLE/SSM attention ordered; batch
  its FFN. Real-weight `--verify-ple-split` matches scalar phase ordering
  bitwise for HC, PLE, and SSM state. Prefill expert H2D drops from132.68 to
  66.24 GiB. This does not establish batched-vs-scalar FFN parity.
- `LLM_QWEN4_PHASE_SCRATCH=1`: one 810 MiB arena for phase-exclusive HC,
  SSM, attention, and MoE intermediates saves1737 MiB at BMAX4096. Persistent
  values and copy-stream banks stay separate. Both full-model requests match
  all baseline layer fingerprints and the complete hash at the same cache.
- `LLM_QWEN4_FINGERPRINT=1`: five stream-ordered fingerprints per layer,
  reported at the existing final tile barrier. Diagnostic, not speed mode.
- Two-token gate/up and down prototypes were rejected and removed after
  one-ULP failures. The test now uses varied non-power-of-two scales.
- `LLM_QWEN4_STAGE_THREADS=128/256/512` changes geometry only. Large bitwise
  oracles pass; the full-model sweep does not show a compelling speed win.
- `LLM_QWEN4_NATIVE_Q8_BATCH=1` wires the existing native Q8 batch kernel into
  attention projections. Model-shaped scalar/batch bitwise tests pass. The
  oracle must initialize outputs on the compute stream; default-stream
  initialization caused an unwritten/NaN output during the first test.
- `LLM_QWEN4_DECODE_ATTN_SHARDS=2/4/8` is an opt-in output-column split with
  unchanged per-output arithmetic. Partial-tile/nonzero-query GPU tests match
  the original bitwise. Full-model tuning is in progress; no default change.

Corrected staged baseline: cache5500/BMAX4096, 4096/64, pinned, overlap1,
PLE split1, promote1, prefill balance0, warmup0 returns first99157 / hash
`601167e3b2fb9425` in4/4 requests with identical fingerprints at all48 layers.
Shared scratch matches it2/2. Cache7200 geometry sweep matches it6/6, but
prefill is only~158–171 tok/s and decode~8–13 tok/s. These are staged-reference
checks, **not scalar F16 parity**. Some geometry128 timing overlapped a CPU
test compilation; do not promote its timing. Later sweeps avoid compilation.

### Live work / next steps

The native-Q8 + attention-shard sweep is running sequentially from
`rdna4/llm/tmp/run_native_shards_sweep.sh` (logs `native_shards_summary.log`,
`native_q8_shards_[1248].log`). Early configurations preserve the corrected
staged hash but do not reach either throughput target. A HIP API + kernel +
copy trace with exact route logging is the next diagnostic: separate CPU
submission and PCIe waits, then inspect cache allocation/policy using the
corrected routing workload. The old cache allocation vectors predate this
routing fix. Previous ROCprof data is in `tmp/rocprof_qwen_split/`.

Refresh the scalar F16 quality corpus after the fixes. The previous corpus
failed all5 staged cases; historical references must not replace a fresh
matched reference. Do not confuse request repeatability with scalar parity.

### Environment and checks

Use `TMPDIR=$PWD/rdna4/llm/tmp`; never `/tmp`. `/local` is absent. GPU devices
are hidden in the sandbox; approved host execution exposes them. GPU jobs must
be exclusive. Do not mutate running scripts/binaries, and avoid compilation
during performance measurements. CPU: Threadripper1950X; GPU: RX9070XT16GiB;
PCIe reports8GT/s x16.

```sh
export TMPDIR="$PWD/rdna4/llm/tmp"
make -C rdna4/llm moe-stage-test
make -C rdna4/llm qwen4-attention-gpu-test
make -C rdna4/llm tmp/test_hip_qwen4_moe_stage tmp/test_hip_qwen4_moe_stage_large
make -C rdna4/llm moe-stage-gpu-test
make -C rdna4/llm moe-stage-quality
```

Preserve unrelated `a64fx/glm5` / `common/transformer.h` work. Do not commit
the stray untracked `rdna4/llm/hip_runner_common.h`; the runner includes
`../hip_runner_common.h`. Commit coherent tested changes and report the hash;
no push without a new explicit push request.

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
