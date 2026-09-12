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

## Resume from here

The quality milestone is complete for the exercised workloads: native HC,
SSM projections, router/shared experts, and exact prefix graphs pass fresh
scalar F16 greedy parity on four short prompts and the 4K/64 staged case. The
best measured native result is about **126 prefill / 22 decode tok/s**; the
requested **200 / 30** target is still open.

Start by reading this file and `rdna4/llm/QWEN38_STATUS.md`, then inspect the
worktree and running jobs:

```sh
cd /mnt/nvme02/work/gemm/main
export TMPDIR="$PWD/rdna4/llm/tmp"
git status --short
ps -eo pid,etime,comm,args | rg 'cc1|test_hip|make|rocprof' || true
```

When the GPU is idle, repeat the checkpoint with:

```sh
sh rdna4/llm/tmp/run_native_ssmtrim_4k.sh
```

Compare first-token/full-hash output with
`rdna4/llm/tmp/native_scalar_4k_reference.log`; expected hash:
`e3d8bf6d47dc6cc3`. Keep temporary artifacts under `rdna4/llm/tmp` and never
use `/tmp`.

The next optimization pass should target routed Q4 gate/up and Q5 down
kernels, then decode transfer and cache overlap. Preserve scalar arithmetic,
rerun the real-weight HC/SSM/MoE oracles and short corpus after each change,
and keep the paired-token routed prototype rejected because its numeric oracle
failed.

### Copy/paste resuming prompt

> Continue the Qwen3.8-Flash-Next RDNA4 performance work in
> `/mnt/nvme02/work/gemm/main`. Read `resume.md` and
> `rdna4/llm/QWEN38_STATUS.md` first. Preserve unrelated worktree changes and
> do not use `/tmp`; set `TMPDIR=$PWD/rdna4/llm/tmp`. The current validated
> native path is scalar-parity exact on the four short prompts and 4K/64,
> with best measured performance about 126 prefill / 22 decode tok/s. The
> required targets remain 200 / 30 tok/s. Check for running GPU/compiler jobs,
> rerun `rdna4/llm/tmp/run_native_ssmtrim_4k.sh` only when the GPU is idle,
> then use `rdna4/llm/tmp/native_profile_analysis.log` to optimize routed Q4
> gate/up, Q5 down, and decode transfer/cache overlap. Preserve scalar
> arithmetic, run the real-weight HC/SSM/MoE oracles and quality corpus after
> each change, and do not promote the rejected paired-token or tiled-attention
> prototypes. Commit coherent tested changes if appropriate; never push
> without explicit per-action authorization.

## Latest performance checkpoint

The current scalar-parity-passing 4K/64 configuration measures
**125.97 prefill / 21.75–21.83 decode tok/s**, below 200/30.
Driver:`rdna4/llm/tmp/run_native_ssmtrim_4k.sh`; log:
`tmp/native_ssmtrim_4k.log`; binary:`tmp/test_hip_llm_ssmtrim`.
Uses all native paths, SSM native warp2, cache balance1, LFU1,
attention shards2 and exact prefix graphs. Both requests match fresh scalar
first15/hash `e3d8bf6d47dc6cc3`. Cache H2D 9.87 GiB, 88.9% hits.
SSM warp modes1/2 each pass the full real-weight bitwise oracle; mode2
adds signed-zero coverage.

A parallel-score attention candidate was bitwise correct but slower in a
fair microbenchmark and was removed. Archives remain in `tmp/`.
The native profile is complete; see `tmp/native_profile_analysis.log`.
No GPU or compiler jobs remain at this checkpoint. Next focus is routed
gate/up and down throughput. All user performance targets remain required.

## Latest parity milestone

Native HC + SSM projections + router/shared experts now pass fresh scalar
first-token/full-hash parity on all four128/16 prompts, two requests each
(8/8). `LLM_QWEN4_BATCH_MOE_NATIVE=1` removes BF16 activation rounding
from router/shared batching. Its real-weight `--verify-moe-native` oracle
passes48 layers x8 rows bitwise. Binary:`tmp/test_hip_llm_nativemoe`.
Logs:`tmp/nativemoe_oracle.log`, `tmp/nativemoe_quality_summary.log`.

The fresh scalar4K/64 reference and both native staged requests PASS:
first15/hash e3d8bf6d47dc6cc3. The qualifying native throughput is
114.38/114.52 prefill and19.78/19.82 decode min/median. Decode cache hits85%,
H2D13.31 GiB; prefill staging65.69 GiB. Peak15890 MiB/free414 MiB.
`tmp/run_native_all_4k.sh` and its three logs capture the exact setup.
Do not use the old601167e3b2fb9425 hash as the scalar reference.

The native API/kernel/copy profile is complete:
`tmp/native_profile_analysis.log` and `tmp/rocprof_qwen_native/`.
Keep GPU jobs exclusive; do not compile during performance measurements.

## Current implementation (2026-09-12 continuation)

The user explicitly asked to keep going until the **200/30** targets are met.
The targets are not met yet, but fresh scalar F16 parity is now established
for the documented short corpus and 4K/64 run. Keep the scalar fallback as the
default for unvalidated workloads. Do not restart the owned staging manager
(committed as `74a68bf2`); details and current measurements are at the top of
`rdna4/llm/QWEN38_STATUS.md`.

Latest follow-up: the corrected-route API profile is in
`rdna4/llm/tmp/rocprof_qwen_api/`. Decode copies30.48 GiB in2.676 s
over64 tokens; kernel time2.200 s. Launch geometry and attention sharding
do not establish a speed gain. Exact-prefix graph replay passes two 4K/64
staged hashes. Native HC batching passes the real-model bitwise oracle for 48
layers, both phases and 8 rows, including F16 injection. Q8 SSM native
batching is implemented:
`LLM_QWEN4_BATCH_SSM_NATIVE=1`, binary `tmp/test_hip_llm_nativessm`.
`--verify-ssm-projections` passes 36 layers x 5 projections x 8 rows bitwise.
The combined native path and native router/shared experts pass the four-prompt
corpus; see `tmp/nativemoe_quality_summary.log`. No production default is
promoted for workloads outside the validated set.

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

No performance sweep is currently required to resume. Begin with the native
SSM-trim checkpoint above, then profile and optimize routed Q4 gate/up and Q5
down throughput. Keep GPU measurements exclusive and do not compile while a
timing run is active. The rejected attention-tile and paired-token prototypes
are archived under `rdna4/llm/tmp/` for reference only.

### Multi-turn coding quality checkpoint (2026-09-12)

The exact-profile HTTP path was exercised with a persistent three-turn C
coding task using `tmp/test_hip_llm_ssmtrim`. Turn 1 compiled and ran
successfully. Turn 2 preserved the prior implementation and added the
requested swapped-bound branch; its test exposed an ambiguous requirement
(`clamp(3,5,1)==5` conflicts with the conventional bound-swap result of 3).
Turn 3 hit the 256-token output limit while adding an array API and was
syntactically incomplete. A larger queue task also hit both 220- and
512-token limits before producing complete code.

The stdio server now saves a state snapshot at each complete prompt boundary
and restores it when the next request still begins with that prompt. This
handles BPE re-tokenization differences in generated text without discarding
the whole recurrent/KV state. The rebuilt GPU retest reports cached tokens
`0 -> 186 -> 231` across three growing turns; turn 3 processed 250 new tokens
out of 481, confirming suffix replay instead of a full reset. The short exact
HTTP turns measured about 6--12 prefill and 9.9--11.4 decode tok/s; these are
cold short-request figures and are not comparable to the 4K staged ~126/22
checkpoint.

The output-budget and prompt-quality issues remain separate: 256 tokens still
truncates a larger array task, and the swapped-bound test requirement must be
made explicit about whether bounds are swapped or the original lower bound is
preferred. Keep those quality checks pending while retaining this cache fix.

The Codex launcher now honors `QWEN38_RUNNER`; before this fix it silently
overrode the requested native runner. A real `CODEX_HOME=/home/syoyo/.codex-local`
probe reached the API, but the full Codex system prompt requires roughly 12K
tokens. At the safe 16 GiB `BMAX=512` setting, native batched prefill measured
12.84--13.20 tok/s, so the bounded probe was stopped before a completed answer.
`BMAX=4096` fails GPU initialization at 16K context from VRAM pressure. This is
an integration/performance limitation, not output-quality evidence.

The explicit `QWEN38_FAST_PREFILL=1` profile now forwards the native Q6K SSM,
native batch QKV, full-layer batch attention, and multi-chunk stateful controls.
With `CODEX_HOME=/home/syoyo/.codex-local`, a real 12,116-token Codex system
prompt ran at 162 tok/s for the first chunk and 137.81 tok/s overall across
seven chunks. The request completed, but the 256-token answer was gibberish;
the multi-chunk recurrent batch carry is therefore still not quality-safe.
Keep this profile diagnostic-only until a fresh scalar output/parity gate passes.
Disabling only fused SSM convolution and recurrence did not fix the result:
the same Codex turn returned only `</think>` at 124.52 tok/s. The remaining
quality defect is therefore in batched attention/MoE or cross-chunk state
handoff, not solely the fused SSM kernels. The launcher also now preserves an
explicit `LLM_QWEN4_NATIVE_BATCH_QKV` override instead of resetting it later.

The documented layer-2 attention cap is exact on the short comparator, but a
4K real-weight run with scalar SSM transitions did not complete within three
minutes, so it misses the requested 50 tok/s floor. It is not a practical
Codex serving profile yet. Quality-first serving should continue using the
scalar route while the batched attention/MoE state handoff is repaired.

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
