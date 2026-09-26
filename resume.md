# Qwen3.8 27B HIP runner vs llama.cpp — resume state

## hipBLASLt-free prefill: self-owned WMMA GEMM + fused dequant (2026-09-26, later)

The runner no longer uses or links hipBLASLt. `make -C rdna4/llm` now defaults
to `HIPBLASLT=0` (no `-lhipblaslt -lhipblas`; `ldd` shows neither). Prefill
projections run `rdna4/llm/gemm_wmma.hip` through `gemm_wmma_dispatch.h`, and
quantized weights are decoded inside the GEMM tile loader instead of being
materialized as BF16 for every 512-token chunk.

Commits `2422e4d8` and `b239cbd6`:

- **Bug fix.** Commit `be2a07f2` (2026-09-03) swapped the K-loop strides of
  `gemm_bf16_own` (skipped half of K) and `gemm_bf16_own_db` (double-counted).
  Every `LLM_GEMM=own` / `HIPBLASLT=0` result since then was numerically wrong,
  including the Qwen4 "batched WMMA → repeated garbage" findings in
  `QWEN38_PREFILL_TUNING.md`. Re-check those conclusions.
- **`gemm_wmma.hip`.** Templated tiled WMMA GEMM: register prefetch,
  double-buffered kslot-major LDS, grid walking M fastest (weight tiles stream
  once), clamped edges, deterministic split-K.
- **Fused-dequant variants.** Cover IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S,
  IQ1_S, IQ1_M, IQ4_XS and Q2_K. Their decoders mirror `dequant_*_to_bf16`.
  `LLM_GEMM_FUSED_CHECK=1` compares every fused GEMM bitwise with dequant +
  GEMM: 2936/2936 were identical on the 4K prefill.
- **Tile rules.** Tuned with `rdna4/llm/bench_llm_gemm`, which compares
  against hipBLASLt when built `HIPBLASLT=1 --blaslt` and checks fused-variant
  bit-equality with `--quant <fmt>`. Rules: dense 128x128/4 waves; fused
  256x128/8 waves for N >= 6144, 128x128/8 waves for N = 5120, 64x64 plus
  split-K for skinny N.
- **Knobs.**
  - `LLM_GEMM_FUSED=0`: dequant + own GEMM.
  - `LLM_GEMM_OWN_LEGACY=1`: old kernels.
  - `LLM_GEMM=blaslt`: A/B, only in a `HIPBLASLT=1` build.
  - `LLM_GEMM_WMMA_{VARIANT,QTILE,SPLITK}`: tuning.

Exact 4K fixture (commands in the next section; `run4k.sh` in
`tmp/qwen38-212w-handoff/` wraps them). 212 W cap, 4096 prompt tokens,
Q8/Q8 KV, ubatch 512, warm pass of `--bench-repeat 2`:

| Backend | Warm prefill | Warm decode | Hash |
| --- | ---: | ---: | --- |
| hipBLASLt (`HIPBLASLT=1 LLM_GEMM=blaslt`) | 614.7 tok/s | 40.79 | `44915ec1039a64c8` |
| own GEMM, dequant + GEMM | 641.7-644.7 | 40.6-40.8 | same |
| own GEMM + fused dequant (`2422e4d8`) | 749.7-753.1 | 40.8 | same |
| + pipelined/slim decode (`b239cbd6`, default) | **764.7-767.3** | 40.75 | same |
| DFlash2 K=7, no-hipBLASLt build (`2422e4d8`) | 744.9 | **87.64** | same, 134/140 accepted |

First-pass prefill no longer pays hipBLASLt plan creation: 508.7 before,
644.7 or more now.

Remaining prefill kernel time at `b239cbd6` (rocprofv3 run before the last
decode tweak, 6.0 s total): fused GEMMs ~2.9 s. The main remaining costs:

- Exact-vector Q8 attention (`qwen35_attention_q8_decode`, 1.17 s). Kept for
  llama.cpp parity up to 4K.
- `deltanet_step_batch_gda_ref` (0.44 s).
- The llama-compatible IQ2_XXS Q8 MMQ (`gemm_iq2_xxs_mmq_wmma`, 0.35 s,
  ~6 TF/s, parity path).

A CPU miner (`xmrig`, ~30 cores) ran during all of these measurements. It
adds launch jitter to short-kernel microbenchmarks; `bench_llm_gemm`
therefore interleaves candidates over rounds and keeps the minimum.

Untested models: other models that previously auto-selected hipBLASLt
(dense n_embd >= 4096, gemma4) now also use the own GEMM and were not
re-validated in this session.

## Current-machine handoff: RX 9070 XT, ROCm 7.14, 212 W (2026-09-26)

This is the latest operational state. Older sections below use model paths and
ROCm installations from another machine. Read `AGENTS.md` first. Use `/local`
for scratch if available, otherwise the repository's `tmp/`; do not use the
system temporary directory. Work from the repository root.

### Setup and local changes

- Branch `main` was at `e45ca254` for these runs. The GPU is an RX 9070 XT
  (`gfx1201`, 16 GiB). The power cap read from the GPU sysfs `power1_cap` was
  `212000000` microwatts (212 W); this session did not change it.
- The target GGUF is
  `/mnt/disk1/models/qwen38/27b/gsq/Qwen3.8-27B-GSQ-RCO-IQ2_XS.gguf`.
  The DFlash2 sidecar is
  `/mnt/disk1/models/qwen38/27b/dflash2/Qwen3.8-27B-DFlash2-Q4_K_M.gguf`.
  Dense NextN is also present at
  `/mnt/disk1/models/qwen38/27b/mtp-Qwen3.8-27B-Q4_0.gguf`.
- `amdrocm-blas-dev7.14`, `amdrocm-runtime-dev7.14`,
  `amdrocm-hipblas-common-dev7.14`, and `amdrocm-llvm-dev7.14` are installed.
  `make -C rdna4/llm -j4 HIPBLASLT=1` succeeds. The executable reports
  `prefill GEMM backend = hipBLASLt` and `Phase-5 graph capture: logits=1
  hidden=1 argmax=1` with the Q8/Q8 graph configuration below. Existing build
  warnings remain; the build exited successfully.
- Two **uncommitted** RDNA4 edits from this session are in the working tree:
  `run_qwen38_gsq_rocm.sh` defaults to the `/mnt/disk1` IQ2 model and sets
  `ROCEW_ROCM_LIB` to `/opt/rocm/lib` unless overridden. Without the latter,
  `rocew` selected the old system `libamdhip64.so.5` and `hipInit` failed with
  `GPU node has an unrecognized id`. `hip_llm_runner.c` has the missing F32/F16
  bridge stubs needed for `HIPBLASLT=0` fallback builds. Preserve other
  pre-existing local changes in `common/`, `cuda/`, and untracked paths.

### Reproduce the exact 4K coding benchmark

`validate_qwen38_reference.py:cpp_prompt()` generates the documented
4,096-token C++17 merge-intervals prompt. The generated file was tokenized by
this GGUF as **exactly 4096 tokens**, beginning with token 248045. Generate it
under a permitted scratch root, then run the ordinary and DFlash2 commands
sequentially so the GPU is not contended:

```bash
if [ -d /local ] && [ -w /local ]; then
  scratch_dir=/local/qwen38-212w-handoff
else
  scratch_dir="$PWD/tmp/qwen38-212w-handoff"
fi
mkdir -p "$scratch_dir"
export TMPDIR="$scratch_dir"
QWEN38_HANDOFF_DIR="$scratch_dir" python3 - <<'PY'
from pathlib import Path
import os
import sys
sys.path.insert(0, 'rdna4/llm')
from validate_qwen38_reference import cpp_prompt
Path(os.environ['QWEN38_HANDOFF_DIR'], 'prompt-4096.txt').write_text(cpp_prompt())
PY
make -C rdna4/llm -j4 HIPBLASLT=1
export QWEN38_MODEL=/mnt/disk1/models/qwen38/27b/gsq/Qwen3.8-27B-GSQ-RCO-IQ2_XS.gguf
bench_args=(--gpu-only-bench --bench
  --prompt-file "$scratch_dir/prompt-4096.txt"
  -n 4096 -s 8192 --ubatch 512 --kv-cache q8q8
  --qwen35-prefill-bf16 --qwen35-decode-graph
  --qwen35-native-q8-prefill --qwen35-native-q2k --qwen35-native-mmvq
  --sampling-profile llama --temp 0 --seed 42
  --decode 256 --bench-repeat 2)
bash rdna4/llm/run_qwen38_gsq_rocm.sh "${bench_args[@]}" \
  > "$scratch_dir/ordinary.log" 2>&1
bash rdna4/llm/run_qwen38_gsq_rocm.sh "${bench_args[@]}" \
  --qwen35-dflash2 /mnt/disk1/models/qwen38/27b/dflash2/Qwen3.8-27B-DFlash2-Q4_K_M.gguf \
  --qwen35-dflash2-draft 7 \
  > "$scratch_dir/dflash7.log" 2>&1
```

The `--bench-repeat 2` first pass creates GEMM plans; compare the second,
warm pass. Both modes stopped at EOS with 154 emitted tokens, reported
`Result: PASS`, and produced the same sequence hash
`44915ec1039a64c8`. DFlash2 verified exact target windows and accepted
134/140 draft tokens on each repeat.

| 212 W mode | First prefill | Warm prefill | First decode | Warm decode |
| --- | ---: | ---: | ---: | ---: |
| Ordinary target | 512.25 tok/s | **613.06 tok/s** | 40.77 tok/s | **40.74 tok/s** |
| DFlash2 K=7 | 508.68 tok/s | **611.13 tok/s** | 86.54 tok/s | **87.48 tok/s** |

This reproduces the documented 600+ tok/s warm prefill and 80+ tok/s
speculative decode on this fixture. It does **not** reproduce 50+ tok/s
ordinary decode; the measured result is about 40.7 tok/s. The exact-4K run
matches the documented sequence hash in `QWEN38_DFLASH2.md`.

The benchmark is workload-sensitive. A padded synthetic 4K prompt reached
619.96 tok/s warm prefill and 41.44 tok/s ordinary decode, but DFlash2
accepted only 43/137 drafts and slowed to 36.65 tok/s. A separate valid
3979-token C coding prompt accepted 61/120 and reached 51.77 tok/s DFlash2
versus 41.25 tok/s ordinary, with identical output hashes. Do not generalize
the 87.48 tok/s result without reporting prompt, accepted drafts, context,
sampling, and output parity.

For comparison, the quality-default Q8 K/Q4 V workload at 53,248 context,
512 prefill tokens, and 64 generated tokens measured 29.08 tok/s prefill and
25.96 tok/s decode with the hipBLASLt build at 212 W. It does not enable BF16
prefill or Q8/Q8 graph decode and is not the same workload as the 4K result.
Before the HIP development packages were installed, a self-owned WMMA build
at a measured 304 W cap reached 30.87/27.38 tok/s for that default workload.
Those measurements differ in cap and backend and are not a controlled A/B.

### Quality and longer-context status

- These September 26 runs checked successful execution and ordinary/DFlash2
  token-hash equality on the exact 4K greedy fixture. `Result: PASS` is the
  runner's internal result; this session did **not** rerun the pinned llama.cpp
  numerical or generated-code quality suite. BF16 projection prefill remains
  an opt-in, approximate arithmetic choice; see `QWEN38_STATUS.md`.
- `QWEN38_DFLASH2.md` records an earlier 16,173-token cross-engine audit in
  which runner and llama.cpp matched only the first 13 generated tokens, and
  the projection/FFN numerical gap remained open. Do not infer whole-model
  parity from the same-hash ordinary/DFlash2 result above.
- `RDNA4_REGRESSION.md` and `test_qwen38_long_context.sh` document 64K
  performance and serving gates. The older 64K target/DFlash fallback suffix
  hashes differed across processes. Recheck this with the current binary and
  the same prompt, KV format, context, and sampling before claiming long
  context equality or promoting a speed path.

### Resume prompt for the next coding agent

> Continue the RDNA4 Qwen3.8-27B IQ2 runner from this handoff. Read
> `AGENTS.md`, this section, `rdna4/llm/QWEN38_DFLASH2.md`,
> `rdna4/llm/QWEN38_STATUS.md`, and `rdna4/llm/RDNA4_REGRESSION.md` first.
> Preserve the uncommitted user and runner work. Use `/local` if available;
> otherwise use repository `tmp/` for all scratch and logs. Verify the 212 W
> power cap, ROCm 7.14 package state, hipBLASLt backend, and exact 4K prompt
> hash. Reproduce the ordinary and DFlash2 numbers above, then profile the
> roughly 40.7 tok/s ordinary decode path against the reported 50+ result,
> identifying workload or configuration differences before changing kernels.
> Improve performance only with stable output hashes and repeated warm runs.
> Test speculative acceptance on meaningful prompts, not only synthetic
> padding. For quality work, run the pinned llama.cpp reference and inspect
> first divergent tokens, layer outputs, and logits at 4K and longer contexts.
> Revalidate 16K/64K target/DFlash parity and serving/cancellation behavior
> before claiming long-context correctness. Report exact commands, power cap,
> prompt token count, KV type, first/warm throughput, accepted drafts, peak
> VRAM, hashes, and any numerical or generated-code checks.

## Long-context, serving-quality, and overlap regression gates (2026-09-22)

Items 7--10 are now represented by reproducible gates. The new
`rdna4/llm/test_qwen38_long_context.sh` runs explicit target, DFlash2, or both
Q8-K/Q8-V modes at a deterministic random 65,536-token offset; its parser
requires complete repeated measurements, stable hashes, 400+ tok/s prefill,
and a configurable decode floor. On gfx1201, two 128-token repeats measured
447.15 tok/s prefill and 35.50 tok/s minimum decode for target-only, and
444.69/35.54 tok/s with DFlash2 configured. Both prefix hashes were
`90178de69a24a76e`. DFlash2 correctly selected target decode beyond its long
context limit, but its suffix hash differed from the target-only process, so
cross-mode byte parity at this boundary remains open.

The resident quality gate now compiles and executes lower-bound, inclusive
clamp, and stable-dedup C++ answers in addition to the existing multi-turn
program. The complete real-GPU DFlash2 suite passes stdio/HTTP, greedy and
seeded sampling, cache restore and LRU eviction, cancellation and recovery,
concurrent identities, and all generated-code cases. Its server response cap
is explicit and defaults to 512 tokens; the earlier implicit 64-token cap
truncated valid coding attempts.

`test_dflash_overlap_lifecycle.py` now covers 27 setup, malformed-layout,
allocation-failure, wait/reset-failure, abort-reuse, and idempotent teardown
cases. Overlap remains opt-in: safety is covered, while prior controlled
measurements did not show a sustained gain. Authoritative commands, thresholds,
results, and promotion rules are in `rdna4/llm/RDNA4_REGRESSION.md`.

## Prefill arithmetic gap isolation and correction probes (2026-09-22)

Implemented opt-in prefill diagnostics that preserve production defaults:

- `LLM_QWEN35_PREFILL_REFERENCE_NORM=1`: use the existing precise reference
  RMSNorm kernel for target prefill only. Scalar decode, draft/verifier norm,
  and BF16 projection selection remain unchanged. Fused DFlash feature/norm
  capture is bypassed so the diagnostic actually controls target prefill norm.
- `LLM_QWEN35_PREFILL_IQ3S_D4=1`: use Q8 activations with FP32 D4 scales for
  IQ3_S SSM QKV/gate projections, keeping the other BF16 paths. Mode `2`
  shares IQ3_S weight decoding across eight token rows. It matches native
  scalar arithmetic but has a different reduction order from mode 1.
- `LLM_QWEN35_PREFILL_IQ4XS_D4=1`: apply the D4 activation contract to the
  IQ4_XS SSM output projection, independently of QKV/gate selection.

These are numerical diagnostics, not production tuning defaults. The generic
fallback remains available for other tensor formats. No token forcing, output
normalization, prompt modification, or sampler bias is used to hide differences.

A single 189-token prefill chunk / decode=1 capture against the pinned llama.cpp
isolates layer zero. Columns below are relative L2 on its last token:

| Stage | Production | Reference norm + IQ3_S D4 | Add IQ4_XS D4 |
| --- | ---: | ---: | ---: |
| Input RMSNorm | 1.27e-7 | **bit-identical** | **bit-identical** |
| SSM QKV | 2.80e-3 | 1.85e-7 | 1.85e-7 |
| SSM gate projection | 3.75e-3 | 2.03e-7 | 2.03e-7 |
| Gated recurrent output | 1.12e-3 | 3.96e-7 | 3.96e-7 |
| SSM output projection | 2.31e-3 | 2.21e-3 | 1.21e-7 |
| FFN input norm | 1.89e-2 | 1.88e-2 | 1.18e-6 |
| Complete layer output | 2.32e-3 | 2.25e-3 | 3.55e-4 |

The alpha projection is already bit-identical and was not changed. These
measurements localize the substantial remaining layer-zero gap to the FFN
once the SSM projection contracts are corrected; they do not establish exact
recurrent-state or whole-model equivalence.

Full 16173-token lower-bound task (same model/Q8-Q8/chunk512/greedy/seed42):

| Diagnostic | Exact initial output tokens | Relative L2, common first 14 logit rows | Prefill tok/s |
| --- | ---: | ---: | ---: |
| Production control | 13 | 0.0452612 | 595.39 |
| Reference norm only | 13 | 0.0452336 | 595.79 |
| IQ3_S D4 mode 1 | 40 | 0.0461942 | 576.90 |
| IQ3_S D4 mode 2 | 13 | 0.0454443 | 588.48 |
| IQ3_S mode 1 + IQ4_XS D4 | 40 | 0.0438669 | 573.10 |

The longer exact prefix alone does not prove lower aggregate error: mode 1
alone slightly worsens the comparable 14-row metric. Combined SSM corrections
improve it about 3.1%, and improve first-selection relative L2 from 0.146984
to 0.136640. None achieves full cross-engine byte parity. Reference norm plus
IQ3_S mode 1 also remains at 13 matching tokens. Norm-only first-selection
relative L2 worsens to 0.151340 despite removing the local norm mismatch.

These are one-pass traced-run prefill measurements, not a repeatability-backed
performance promotion. Decode trace I/O is excluded from throughput claims.
The changes remain opt-in: the faster D4 tile still changes the near-tie and
the combined correction costs roughly 4% prefill in this probe. All completed
coding answers pass the C++17 checker and 30000 ASan/UBSan boundary/random cases.

Sampled seed-42/temperature-0.8 combined-SSM runs also retain exact target-only/
DFlash2 bytes and hash `3dcbfc4759dbed36`, with 124 emitted tokens. Their
prefill measured 571.59 / 568.51 tok/s and decode 38.23 / 49.40 tok/s. Both
answers pass the same 30000-case C++ checker; this is runner/DFlash parity,
not sampled llama.cpp parity.

Final default-path rebuild check retains the original bytes/hash on both
16K repeats and on the 189-token boundary case. Prefill is 595.30 / 587.92
tok/s; decode is 38.87 / 38.86 tok/s. The second prefill result is below the
previous warm 613.74 measurement despite unchanged default dispatch. Treat
this timing variation as unresolved until a controlled binary A/B; do not
claim a production speedup or strict no-regression result from these probes.

Validation: runner build; profile tests; 192 prefill-norm CPU dispatch cases;
24 GPU IQ3_S row-tiling cases bit-identical to the scalar native kernel,
including partial tiles (counts 1/7/8/9/189/512); existing 756 verifier and
54 captured-decode dispatch cases. Added `compare_qwen35_prefill_layer.py`
with five CPU tests: it verifies metadata/layout, compares corresponding
last-token stages, and rejects wrong shapes, nonfinite data, and ambiguous
multi-chunk captures instead of silently comparing unrelated tensor tails.

Artifacts: `rdna4/llm/tmp/parity-layer0/` contains capture scripts, staged
comparisons, and GPU logs; `rdna4/llm/tmp/cross-engine-16k/` contains full-model
logs, comparisons, token/byte evidence, and traces. To conserve disk, norm-only,
combined-norm, and mode-2 traces retain only common-prefix logits in NPZ files;
mode-1 and combined-SSM full traces are retained. Use the new layer comparator
on `runner`, `runner-norm`, `runner-d4`, or `runner-ssm-d4` against `llama`.

Next: match the FFN IQ1_S/IQ1_M/IQ2_XXS prefill arithmetic, then examine the first
attention layer. Preserve MMQ/D4 scale precision and reduction order when
optimizing token reuse; confirm full greedy/sampled reference parity and
controlled prefill/HTTP measurements before promoting any diagnostic.

## Cross-engine exact-parity audit (2026-09-22)

**FAIL for exact tokens/bytes on the 16173-token lower-bound coding task.**
Fresh full-logit traces compare the repaired ordinary runner against pinned
llama.cpp `1859b520910af6f682256fd7299797774111a27a`, using the same IQ2_XS
GGUF, Q8 K and Q8 V, context 32768, chunk 512, greedy sampling, seed 42,
and a 192-token generation limit. All nine reference manifest artifacts
match their recorded SHA-256. Prompt IDs match exactly. Both engines finish
normally at EOS: runner selects 129 tokens (128 emitted), reference 155
(154 emitted). Traced runner output also matches the prior untraced result.

The first 13 generated tokens agree. At zero-based row 13 (token 14):

| Token | Runner logit | llama.cpp logit |
| --- | ---: | ---: |
| 198 (one newline) | 23.9562073 | 24.0388412 |
| 271 (two newlines) | 23.9779339 | 24.0382385 |

Each engine selects its own exact argmax: runner 271, reference 198.
The reference margin is only 0.0006027; the runner prefers 271 by 0.0217266.
The first differing output byte is offset 44, where the reference continues
with `#include <climits>` and the runner starts the function after a blank
line. This is a numerical rank reversal, not a tokenizer or greedy-sampler
mismatch. No output normalization was used.

Logits already differ immediately after prefill (before the first decode
step): max absolute difference 1.84306, relative L2 0.146984. At the first
token divergence the relative L2 is 0.020516. Across the 14 rows with matching
preceding input IDs, relative L2 is 0.045261. Later rows are not compared
numerically because the generated input histories differ. The production
BF16 prefill is approximate, but this check does not isolate which operation
causes the discrepancy or exclude additional decode numerical differences.

Both fresh answers compile and pass 30000 `std::lower_bound` boundary/random
checks with ASan/UBSan. Functional correctness passes; cross-engine exact
parity does not. The earlier target-only/DFlash exact-output checks remain
valid and are a separate claim. This audit covers greedy generation; it does
not establish sampled cross-engine parity or general workload equivalence.

Artifacts under `rdna4/llm/tmp/cross-engine-16k/`: `run.sh` (exact commands),
`reference-audit.json`, `analysis.json`, `comparison.json`, both logs, prompt
IDs, selected IDs including EOS, raw bytes, and full F32 logits. Recheck with:

```sh
python3 rdna4/llm/compare_generation.py rdna4/llm/tmp/cross-engine-16k/runner rdna4/llm/tmp/cross-engine-16k/llama
```

Expected exit status is 1 for the recorded mismatch. Next numerical work:
compare prefill layer outputs at identical positions to isolate the first
operation that drifts; preserve this sensitive fixture rather than changing
the prompt or sampler to hide the rank reversal. No production path changed
for this audit, and trace-I/O timings are not performance measurements.

## Captured target-only attention correctness repair (2026-09-22)

Fixed the quality failure discovered by the broader 16K binary-search task.
`launch_attn_decode_native_q8` previously omitted combine when the host
position was below 256. Decode graphs are captured at position zero, so their
later multi-split replays never combined attention partials. The failed
captured target-only run emitted eight corrupted tokens. This was not caused
by the new five-query verifier, which uses its own launch path.

Graph-enabled launches now always retain combine; the kernel reads the
current device position and returns when there is only one split. Uncaptured
short-context launches retain their existing no-op avoidance. This preserves
capture performance without replaying a short-position host decision.

The corrected 16173-token greedy run emits 128 correct tokens in both repeats,
byte-identical to uncaptured target decode and K=4 DFlash2, hash
`41b94b5f8a941e8d`. Decode is 38.96 / 38.90 tok/s versus 37.92 uncaptured;
prefill is 595.17 / 613.74 tok/s. A 189-token prompt plus 128 generated tokens
crosses the 256-position boundary with exact graph/uncaptured text and token
parity (42.86 versus 41.65 tok/s). The sampled 16K graph run emits 124
tokens at 38.22 tok/s, exactly matching the sampled DFlash2 answer and hash
`3dcbfc4759dbed36`. These are quality-valid measured rates;
the previous corrupted-output timing is not a performance baseline.

`test_decode_attention_launch.py` checks 54 combinations of kernel fallback,
graph selection, and short/long/boundary position. It rejects the old guard
and passes the repaired one. Full-model answers pass the C++ lower-bound
checker. Build and the existing verifier dispatch checks pass.
Artifacts: `rdna4/llm/tmp/lower-bound-target-{uncaptured,graph-fixed}.log`,
`attention-capture-quality.sh`, and `attention-capture-*.log`.

The llama.cpp reference answer still differs in bytes, although all generated
implementations pass the same boundary/random correctness checks. Remaining:
isolate that first cross-engine logit divergence, broaden coding workloads,
and measure a useful injection overlap interval before enabling overlap.
Ordinary target-only throughput optimization remains outside the items 2–5
batch; fixing a demonstrated correctness failure is included.

## Five-query verifier and broader coding quality (2026-09-22)

Full K=4 verifier windows now use a fixed-five-query Q8/Q8 attention kernel
(anchor plus four proposals) when their adaptive split counts agree.
`LLM_QWEN35_VERIFY_ATTN_FIXED5=0` is the diagnostic opt-out. The generic and
fixed-eight kernel bodies are unchanged; short windows and split-count
boundaries retain their fallback. Prefill dispatch is unchanged.

A/B on a 16173-token C++ binary-search prompt, IQ2_XS, Q8/Q8, chunk 512,
K=4, seed 42, two repeats per configuration:

| Sampling | Previous default decode | Fixed-five decode |
| --- | ---: | ---: |
| Greedy (128 tokens) | 55.56 / 55.91 | 56.51 / 56.68 |
| Temperature 0.8 (124 tokens) | 48.58 / 48.91 | 49.52 / 49.79 |

Rates are tok/s. All eight answers preserve control token hashes and bytes:
greedy `41b94b5f8a941e8d`, sampled `3dcbfc4759dbed36`. Prefill measured
590–615 tok/s across these runs; this change does not select a new prefill path.

The prompt requests C++17 `lower_index(const std::vector<int>&, int)`, an
explicit overflow-safe binary search, including empty arrays, duplicates,
and integer extremes. `test_cpp_lower_bound_output.py` compiles each distinct
answer separately from its driver, then checks 30000 boundary/random cases
against `std::lower_bound` with ASan/UBSan. Both runner implementations and
the pinned llama.cpp reference implementation pass. The greedy reference
answer is **not byte-identical** on this broader 16K workload: it adds an
explicit empty-array check, an extra header, and whitespace. This is correct
code, but cross-engine byte parity remains an open quality target; the new
specialization has exact parity with the prior runner. Uncaptured target-only
decode also matches the DFlash answer (128 tokens, same hash, 37.92 tok/s).
A separate captured target-only run emitted a corrupted eight-token answer;
its short-position host guard skips attention combine during graph capture.
Repair and validate that capture guard next; do not use the failed run as a
quality-valid performance result.

Validation: 30 direct GPU comparisons at 4K/16K/64K are bit-identical;
756 CPU dispatch cases pass; runner build and profile checks pass.
The HTTP harness now accepts `--dflash2-draft 4` or `7` (default 7).
The K=4 run with context 8192, snapshot limit 4096 and cached prompt 3072
passes stdio, HTTP, cached-prefix, cancellation, concurrency, sampled
repeatability, and multi-turn C++ compile/run coverage. At 64K/five rows,
the integrated kernel probe measured 3.086 ms generic versus 2.627 ms fixed.
This is a kernel result: normal DFlash still falls back to target-only at 32K.

Reproduction artifacts in `rdna4/llm/tmp/`: `attention-fixed5-quality.sh`,
`lower-bound-long-prompt.txt`, `attention-fixed5-{integrated,quality,http}.log`,
per-configuration `attention-fixed5-16k-*.log`, and `lower-bound-llama.log`.
Run the new checker on these generation logs. The quality script explicitly
sets control/candidate selection, so it remains usable after promotion.

Remaining: fix captured target-only attention combine, then isolate the first
long-context llama.cpp/runner logit divergence;
expand coding-task coverage; investigate a useful injection overlap interval
before changing its default. Keep ordinary 40+ tok/s target-only optimization
outside this goal batch. Slower fusion/projection probes remain opt-in.

## Eight-query verifier attention follow-up (2026-09-22)

The default verifier now uses a fixed-eight-query Q8/Q8 split kernel when
all eight causal rows share the adaptive split count. Other window sizes and
split-count boundaries retain the original generic kernel, whose body is
unchanged. The specialization keeps split parallelism and the exact merge
order. `LLM_QWEN35_VERIFY_ATTN_FIXED8=0` is the diagnostic opt-out.
The slower single-block split/combine fusion remains opt-in.

Warm full-model A/B (IQ2_XS, Q8/Q8, chunk 512, K=7, seed 42):

| Coding context / sampling | Previous default | Fixed-eight attention |
| --- | ---: | ---: |
| 4096 tokens, greedy | 87.01 | 88.18 |
| 4096 tokens, temperature 0.8 | 83.43 | 84.61 |
| 16096 tokens, greedy | 77.56 | 80.40 |
| 16096 tokens, temperature 0.8 | 74.53 | 76.97 |

Rates are tok/s, with two repeats per configuration. All 16 generations
retained hash `15f17d2640c1adfc` and byte-identical text, also matching the saved
pinned llama.cpp reference output. This simple clamp prompt stops after 46
tokens; the HTTP suite separately covers sampled and multi-turn C++ quality.
Prefill dispatch is unchanged.

`test_verifier_attention.py` now tests generic, grouped, and fixed-eight paths:
21 GPU comparisons at 4K/16K/64K were bit-identical, including the staged
source checked independently of uncommitted experiments. At 64K/eight rows,
the integrated isolated test measured 3.722 ms generic versus 2.866 ms fixed.
These are kernel timings. Normal DFlash still falls back to target-only
decode at 32K, so this is not a claim of 64K speculative throughput.
`test_verifier_launch.py` passes 216 cases covering default/on/off selection,
gated/ungated output, short windows, and non-reusable split-boundary windows.
The rebuilt default passes the HTTP/stdio cached-prefix, cancellation,
concurrency, sampled-repeatability, and multi-turn C++ compile/run suite.
Runner builds and profile checks pass.

Further projection probes retained output but did not improve throughput:
control 87.26 tok/s; fixed-eight fused QKV 87.08; fused gate/up 87.03; paired
BF16 alpha/beta 86.87. These remain diagnostic candidates. Injection overlap
also remains opt-in based on the prior neutral timing result.

Artifacts under `rdna4/llm/tmp/`: `dflash-followup.sh` and its logs;
`attention-fixed8-quality.sh` and its eight per-configuration logs;
`attention-fixed8-integrated.log`, `attention-fixed8-index-gpu.log`, and
`attention-fixed8-http.log`. The A/B script explicitly sets the attention
opt-out for controls and remains valid after promotion.

Remaining: evaluate shorter full-window attention specializations, extend
long-context quality beyond the bounded clamp workload, and investigate a
larger useful overlap interval before changing injection defaults. Keep
ordinary 40+ tok/s target-only decode outside this goal batch.

## Items 2–5: goal and validated progress (2026-09-22)

Goal prompt: improve long-context verifier attention; optimize DeltaNet and
checkpoint publication without changing rollback/accepted state; reduce
DFlash2 draft cost; and validate safe per-stream cache-injection overlap.
Require controlled GPU measurements, exact output, seeded K=4/K=7 quality,
long-context checks, and HTTP/stdio lifecycle coverage before changing defaults.
Ordinary 40+ tok/s target-only decode is outside this batch. The goal API
cannot replace its older paused unfinished goal; this is the active scope.

**Promoted defaults:** eight-row Q4_K draft projection and d_state=128
DeltaNet verifier specialization. Diagnostic opt-outs are
`LLM_QWEN35_DFLASH_Q4K_FIXED8=0` and
`LLM_QWEN35_DELTANET_VERIFY_FIXED128=0`. Unsupported shapes retain their
existing paths. This supersedes older pending/opt-in notes for these two paths.
Attention fusion and injection overlap remain opt-in.

Warm 4K coding measurements (Q8/Q8, chunk 512, seed 42; 46 tokens to EOS):

| Window / sampling | Control | Fixed-eight draft | Fixed-128 DeltaNet |
| --- | ---: | ---: | ---: |
| K=4 greedy | 56.94 | 56.94 | 57.74 |
| K=4 temperature 0.8 | 55.26 | 55.40 | 56.06 |
| K=7 greedy | 83.58 | 85.59 | 84.45 |
| K=7 temperature 0.8 | 80.12 | 82.27 | 81.28 |

Rates are tok/s. K=7 fixed-eight draft time fell from 84.75 to 70.01 ms
(greedy) and 80.57 to 66.45 ms (sampled). All 24 generations in this matrix
retained hash `15f17d2640c1adfc` and identical text. This simple prompt
produces the same text at both temperatures; broader sampled/coding behavior
is covered by the separate HTTP suite below.

At 16,096 coding tokens, combined fixed-eight/DeltaNet improved warm decode
75.15 -> 77.44 tok/s with byte-identical output. Prefill ranged 573–612 tok/s
across the matched runs; these specializations do not change prefill dispatch.
The rebuilt default path (no candidate overrides) retained exact output in
three sampled repeats, with warm decode 83.41/83.38 tok/s and prefill
609.13/608.35 tok/s. Log: `rdna4/llm/tmp/dflash-goal-default.log`.
The 64K direct attention test bypasses the normal DFlash >=32K target-only
fallback. All 27 production-kernel comparisons were bit-identical, but fused
split/combine was slower: for eight query rows at 64K, serialized/grouped/fused
measured 3.383/3.734/13.875 ms. These are isolated kernel timings, not full-model
throughput. Do not promote the current fusion design.

Validation:
- `python3 rdna4/llm/test_dflash_fixed8.py`: 15 GPU shapes, all values exact,
  including partial output blocks and untouched guards.
- `python3 rdna4/llm/test_deltanet_verifier.py`: 20 GPU cases; every output and
  rollback checkpoint exact, canonical state unchanged.
- `python3 rdna4/llm/test_verifier_attention.py`: 27 GPU cases at 4K/16K/64K
  in the experimental working tree (the fused candidate remains uncommitted).
- `python3 rdna4/llm/test_verifier_commit_copy.py`: 256 CPU coverage cases.
- `python3 rdna4/llm/test_dflash_overlap_lifecycle.py`: 23 setup/failure/reset
  cases, including HIP's default stream. Reset must fence it too.
- `test_qwen35_dflash2_http.py` with fixed-eight, fixed-128, overlap, and fused
  injection enabled: stdio, cached 3072-token prefix, cancellation, concurrent
  clients, sampled repeatability, and multi-turn C++ compile/run all PASS.
- Runner rebuild and `bash rdna4/llm/test_qwen38_profiles.sh`: PASS.

Reproduction scripts/logs live under `rdna4/llm/tmp/`: `dflash-goal-quality.sh`,
`dflash-goal-long.sh`, `dflash-goal-http.log`, `verifier-attention-goal.log`,
`dflash-fixed8-gpu.log`, and `deltanet-verifier-gpu.log`. The A/B scripts now set
both opt-outs to zero for controls so future runs remain valid after promotion.
GPU tests require host KFD access. No external network/model downloads needed.

Correctness commits: `8d452760` fixes checkpoint copy tails/unaligned rows;
`7c147490` hardens injection failure cleanup, retry suppression, and reset
fencing. Injection overlap retained output and passed lifecycle checks, but
showed no measurable throughput benefit and remains opt-in. Remaining work:
replace the slow attention fusion design; evaluate additional draft projection
candidates; measure sustained overlap benefit before promotion; broaden
long-context quality beyond this bounded coding workload.

## Current goal prompt: items 2–5 (2026-09-22)

Optimize the RDNA4 Qwen3.8 DFlash2 runner by (2) improving long-context
verifier attention while retaining exact Q8/Q8 arithmetic and captured-graph
stability; (3) optimizing DeltaNet/recurrent tails and checkpoint publication
without changing accepted-row, rollback, or restore semantics; (4) reducing
DFlash2 draft cost with measured K=4/K=7 gains; and (5) validating safe
cache-injection overlap with per-stream workspaces, cancellation, and
concurrency. Require controlled GPU A/B measurements, exact target token/text
parity, seeded sampling and coding quality, random-token 64K validation, and
HTTP/stdio lifecycle coverage before promoting defaults. Fix correctness
issues, record gains and rejected candidates, and commit coherent validated
changes. Ordinary 40+ tok/s target-only decode is outside this batch.

The goal tool still holds the older paused unfinished goal and rejects a
replacement; this section records the user's narrowed active scope.

## Prior measurements and implementation state


DFlash2 K=7 GPU check (same 4096-token coding prompt, Q8/Q8, chunk 512,
greedy seed 42, up to 64 generated tokens, two repeats per configuration):

| Configuration | Cold / warm decode tok/s |
| --- | --- |
| Control | 79.38 / 83.57 |
| Grouped combine mode 3 | 81.61 / 83.27 |
| Fused split/combine | 72.76 / 76.22 |
| Fused checkpoint copy | 81.75 / 83.66 |

All eight runs stopped at EOS after 46 emitted tokens, retained sequence hash
`15f17d2640c1adfc`, and produced byte-identical text to a fresh pinned
llama.cpp Q8/Q8 reference run. Text SHA-256 (between generation markers):
`c6c83e1a21f5e487b27bb7bbf08e778e7bc20d3d4d67d1dd05a940178804eef1`.
The generated clamp function compiled as C++17 and passed ten boundary cases.
This validates this bounded greedy workload, not general logit equivalence or
seeded sampling. Earlier 64-token hashes are not directly comparable to these shorter
EOS-aware runs. No candidate merits promotion
from these timings; long-context verifier and sampled K=4/K=7 gates remain.
Commands and logs are in `rdna4/llm/tmp/gpu-access-dflash-ab.sh`,
`gpu-access-dflash-tail.sh`, `gpu-access-dflash-{control,grouped,fused,commit}.log`,
and `gpu-access-llama.log` in that same directory.

GPU access is restored when execution runs outside the sandbox (RX 9070 XT,
`gfx1201`). The rebuilt runner passes the random-token 65536-depth Q8/Q8
baseline: prefill 446.81 tok/s; three 512-token decode repeats 35.42, 35.40,
and 32.21 tok/s, all hash `36a22439594d4e43` (depth hash
`90178de69a24a76e`). Log: `rdna4/llm/tmp/gpu-access-64k-baseline.log`.
The third-repeat slowdown means small performance differences require a
matched recheck; the 40+ tok/s ordinary decode target remains open.

GPU A/B with `LLM_QWEN35_IQ_MIXED_QKV_FUSED=1` and
`LLM_QWEN35_IQ_MIXED_GATEUP_FUSED=1` (default 256 threads) retained depth
hash `90178de69a24a76e` and all three 512-token suffix hashes
`36a22439594d4e43`. Prefill was 445.07 tok/s; decode was
34.78/34.74/34.73 tok/s, below the control's first two 35.42/35.40 runs
(control third run 32.21). This does not justify promotion; retain opt-in.
Log: `rdna4/llm/tmp/gpu-access-64k-mixed256.log`.

Commit `496b5043` records the grouped verifier dispatch fix and its 36-case
regression test; other pending experiments remain separate.

A verifier dispatch audit also fixed the opt-in grouped combine launch:
gated kernels require a gate pointer before positions, but dispatch supplied
the ungated argument array. It now selects the matching argument layout.
`python3 rdna4/llm/test_verifier_launch.py` executes the production dispatch
with a recording launch stub across 36 gated/ungated width-4/8/16 cases.
This validates the host ABI; GPU parity is a separate gate.

Follow-up correctness audit: the opt-in fused commit's unaligned hidden/logit
copy indexed scalar elements in steps of four, leaving three quarters of each
row stale. Both scalar loops now cover every element. The CPU test
`python3 rdna4/llm/test_verifier_commit_copy.py` executes the production kernel
bodies against complete row copies (256 aligned/unaligned, padded, short-tail,
large-logit, and accepted-row cases). The old indexing fails this test. This
proves copy coverage only; GPU parity and throughput must still be checked with the candidate enabled.

1. Raise ordinary random-token 64K decode from the current ~35.8 tok/s toward
   40+ by reducing the dominant mixed IQ projection traffic without changing
   logits or the seeded token stream.
2. Fuse the long-context target verifier's attention split/combine tail while
   keeping the captured graph ABI and exact Q8/Q8 arithmetic stable.
3. Optimize the hybrid DeltaNet tail and accepted-row checkpoint publication
   without changing rollback or committed-state semantics.
4. Reduce DFlash2 draft cost while retaining exact target verification and the
   K=4/K=7 quality gates.
5. Prepare safe sidecar cache-injection overlap with per-stream workspaces;
   production defaults remain unchanged until measured gains are demonstrated.

The sidecar attention merge is landed, and DFlash2 proposal embedding now
decodes one exact anchor row and one exact mask row before copying the repeated
mask row on device. The pinned 4K C++ merge gate is byte-stable: greedy K=7 is
82.65--84.12 tok/s warm with 140 drafted/134 accepted and hash
`44915ec1039a64c8`; seeded sampled K=7 is 70.47 tok/s with 140 drafted/115
accepted and hash `630b7cbc72230e0d`. Draft time is about 263--264 ms (greedy)
or 250 ms (sampled) for 140 proposals. The full HTTP/stdio, cancellation,
cache, concurrency, and multi-turn C++ quality suite passes. The strict
ordinary 64K target, target-tail fusion, and production cache-injection
overlap default remain open.

The overlap candidate now accepts `LLM_QWEN35_DFLASH_INJECT_KV_FUSED=1` at
sidecar load time. For compatible BF16-cached K/V weights it uses one wider
per-layer injection GEMM and keeps the K/V row stride explicit through norm,
RoPE, and cache store; mixed or unsupported layer types retain the two-GEMM
path. The overlap workspace validator now accepts either the fused contiguous
K/V buffer or a complete separate K/V pair, so this candidate can use its
private stream workspace; incomplete layer workspaces still fall back to
serialized injection. The candidate remains opt-in until resident cache
hashes and HTTP/C++ quality gates prove that the changed GEMM shape is exact
and beneficial.

The K=7 DFlash2 window also has an opt-in
`LLM_QWEN35_DFLASH_Q4K_FIXED8=1` Q4_K projection candidate. It specializes
the exact eight-row proposal shape so the packed dot loop no longer tests a
runtime row count, while preserving the multi8 affine correction and warp
reduction order. The serialized multi8 path remains the default pending a
resident K=7 hash and draft-time A/B measurement.

The same fixed-eight activation specialization is available to the opt-in
fused DFlash Q/K/V and gate/up projection path as
`LLM_QWEN35_DFLASH_QKV_FIXED8=1`; it preserves the existing output row ranges
and reduction order.

All DFlash Q8_1 projection and SiLU candidates now apply the same activation
scratch availability guard and fall back to their serialized/native path if a
best-effort allocation is incomplete.

Overlap setup failures now set a runner-local disabled bit, so subsequent
commits take the serialized injection path without repeatedly retrying failed
stream, event, or workspace allocation.

Overlap failure handling now synchronizes and retires the private injection
stream before returning an error, so a partially queued injection cannot leave
scratch marked reusable for the next proposal. The successful path remains
event-ordered and asynchronous. If a future request exceeds the private
eight-row overlap tile, commit now retires any private work and falls back to
the capacity-independent serialized injector instead of failing the request.
The public commit entry point also retires a still-pending prior injection, so
callers that skip the usual proposal boundary cannot reuse private scratch
out of order.
MTP reset and full runner reset now retire pending overlap work too, protecting
snapshot restore and new-request cache initialization from stale sidecar
writes. The reset-only helper synchronizes the target stream after enqueueing
the event wait, making completion host-visible before synchronous restore copies.

The opt-in fused recurrent commit copy now sizes its grid for the largest of
the recurrent state, hidden row, and vocabulary-logits row. This closes a
large-vocabulary tail omission in the diagnostic path; the serialized commit
copy and all production defaults are unchanged. Its aligned-row path now
vector-copies only the full groups and emits the scalar tail once; unaligned
strides stay on the complete scalar path, avoiding unaligned vector loads and
a redundant tail rewrite without changing accepted-row ordering or checkpoint
publication semantics.
It also publishes a final 1--3-float tail for convolution and recurrent
snapshots when a state length is not divisible by four; aligned shapes stay on
the existing vector path. The host grid now uses ceil division as well, so a
sub-four-float state still launches the tail publisher.
An empty state description is clamped to one launch block, preventing a
malformed zero-state configuration from issuing a zero-grid copy.
The fused selector is also limited to configurations with at least one
recurrent layer because its layer-zero branch publishes the accepted hidden
and logits rows; zero-layer models retain the scalar I/O copy fallback.

The recurrent copy kernels now select the vector path only when the source row
stride is four-float aligned. Truly unaligned snapshot strides use a complete
scalar copy, preventing an unaligned `uint4` load from skipping the row prefix;
the aligned path still copies full groups once and then its short tail.

The DFlash2 mask-row broadcast also has an opt-in
`LLM_QWEN35_DFLASH_EMBED_BROADCAST_KERNEL=1` path. It uses the existing exact
row-repeat kernel for all repeated mask rows after the anchor and first mask
row, while the established device-copy loop remains the default.

The sidecar injection path now has an opt-in
`LLM_QWEN35_DFLASH_QKNORM_ROPE_FUSED=1` kernel that combines K RMSNorm and
M-RoPE for each injected row. It keeps the original reduction and rotation
operation order while removing the intermediate global-memory round trip and
one launch per DFlash layer; the established two-launch path remains the
default pending resident hash and draft-time gates.

The DFlash proposal path also exposes
`LLM_QWEN35_DFLASH_QKNORM_ROPE_PAIR_FUSED=1`. It combines the existing paired
Q/K RMSNorm reductions and both M-RoPE transforms in one row/head grid,
retaining the normalized store barrier and per-row arithmetic while removing
three launch boundaries. The four-launch reference path remains the default
until resident K=4/K=7 hashes and draft timing validate the candidate.

The IQ1 Q8_1 batch staging path now skips a redundant Q8x2 quantizer when the
consumer set is IQ1-only, which is the common MTP gate/up reuse case. It
invalidates the shared Q8x2 metadata after writing the IQ1 block-sum layout so
graph replay cannot consume stale second-term bytes. Mixed SSM Q8x2/IQ1 roles
keep an explicit preserving helper and their previous two-format contract.
This is a code-level reduction in draft/verify launch work; resident logits,
hashes, and 64K timing are still required before changing any production
selector.

The DFlash multi-row dispatcher also reuses exact Q8_1 activation staging for
IQ4_XS Q/K/V rows, with the cached source shape and weight type checked before
reuse. This is a sidecar-only launch reduction and remains unmeasured on the
resident device.

An opt-in `LLM_QWEN35_IQ4XS_GATEUP_FUSED=1` target path now applies the same
exact native IQ4_XS Q8_1 staging and row-range fusion to matching FFN gate/up
matrices. It remains diagnostic until resident logits, hashes, and 64K timing
show a quality-safe gain.

The dense FFN path also exposes
`LLM_QWEN35_IQ2XXS_GATEUP_FUSED=1` for matching IQ2_XXS gate/up matrices. It
uses the existing IQ2_XXS Q/K/V kernel with the V range disabled, sharing the
native Q8_1 tile and codebook staging while retaining the standalone integer
dot and reduction order. This remains opt-in pending resident parity and 64K
throughput validation.

Matching IQ3_S gate/up matrices have the corresponding
`LLM_QWEN35_IQ3S_GATEUP_FUSED=1` candidate. It shares native Q8_1 staging and
one row-range launch through the existing IQ3_S fusion kernel, retaining the
standalone code decode and reduction order. Resident parity and sustained 64K
timing are still required before enabling it by default.

Verifier attention now also has an opt-in
`LLM_QWEN35_VERIFY_COMBINE_GROUPED=3` sixteen-row combine candidate. It matches
the verifier's maximum batch, loads all split metadata once, and preserves the
per-row split-order max, numerator, denominator, and gate arithmetic. The
four-row default and eight-row mode remain unchanged pending long-context
resident hashes and timing.

The grouped selector is query-aware: four-row and eight-row captured windows
use the narrowest exact metadata tile even when mode 3 is selected, while a
full sixteen-row window still uses the sixteen-row candidate. This reduces
shared-memory allocation and idle row loops without changing the combine
kernel ABI or per-row arithmetic.

## 2026-09-22 continuation: opt-in native Q2_K Q/K/V projection fusion

The ordinary one-token attention dispatcher now exposes
`LLM_QWEN35_Q2K_QKV_FUSED=1` for layers whose Q, K, and V matrices are all
Q2_K with one input width. The new native kernel keeps the standalone
Q2_K eight-virtual-warp partials, affine correction, and warp reduction order;
it only selects the three output row ranges in one grid after a shared exact
Q8_1 activation quantization. The 128-thread geometry is retained for the
same very-wide shape where the standalone path uses it. Both hybrid and
standard attention dispatches are covered, while the serialized path remains
the default pending resident 4K/64K logits, hashes, and throughput checks.
HIP device syntax, the host build, and profile checks pass; the resident GPU
gate is unavailable in this environment (`/dev/kfd` is absent).

The same native Q2_K kernel now has an opt-in
`LLM_QWEN35_Q2K_GATEUP_FUSED=1` dense gate/up form. It shares the exact Q8_1
activation tile, selects the two output ranges in one grid, and keeps the
standalone 128-thread geometry for the tall 5120-column FFN shape. SiLU stays
as a separate exact stage; the serialized gate/up path remains the default
pending resident 4K/64K logit, hash, and timing validation.

The ordinary attention dispatcher also exposes an opt-in
`LLM_QWEN35_Q4K_QKV_FUSED=1` candidate for all-Q4_K Q/K/V rows. It reuses the
existing Q4_K/Q8_1 multi-output kernel for one decode row and shares one Q8_1
activation quantization launch across the three projections. The serialized
Q4_K path remains the default because the Q8_1 staging changes the arithmetic
contract; resident logits, hashes, and sustained 64K timing are required
before considering it for production. The candidate also checks both batch
activation scratch buffers before enqueueing, so a best-effort scratch
allocation failure falls back to the serialized path without dereferencing a
null staging pointer.
All opt-in one-row Q/K/V and gate/up helpers now reject zero-width shapes before
launching quantization, so malformed metadata falls back to the native path.
They also share a native-Q8_1 readiness check for the quantizer function and
both activation buffers. Partial diagnostic loads therefore fail closed to the
reference dispatcher instead of passing incomplete staging pointers to a fused
kernel; successful launches and production defaults are unchanged.
The hipBLASLt bridge applies the same positive-dimension validation before
building a plan, preventing malformed sidecar or prefill shapes from reaching
library layout creation.
The strided-batch entry point now rejects non-positive batch counts before its
batch-1 fast path, so invalid metadata cannot be silently reinterpreted as a
single-row GEMM.
Its initialization path now rolls back partially-created hipBLAS and hipBLASLt
objects when a handle, preference, or preference-attribute call fails, so an
optional retry cannot inherit stale library state.
The lazy per-stream workspace map is also protected by a narrow mutex; target
and sidecar launches still execute on their own HIP streams, while concurrent
first-use allocation cannot race or duplicate a scratch pointer.
Plan initialization and cache insertion use a separate short-lived lock, so
concurrent shape misses cannot corrupt the shared plan map or duplicate a
heuristic query; matmul submission remains outside both cache locks.
The shared-handle F16 fallback similarly guards stream selection through
enqueue, avoiding cross-context stream substitution while leaving device work
asynchronous.
Mutable bias and epilogue descriptor attributes are protected through their
enqueue transaction as well, preserving per-request bias pointers when target
and sidecar contexts submit concurrently.
Reset and MTP rollback now skip the pre-wait target synchronize when an
injection is pending; the overlap helper inserts the event dependency and one
host-visible fence, while the no-overlap path retains its original fence.

## 2026-09-22 continuation: skip no-op short-window attention combine

Native Q8 decode and prefill now omit the separate split-combine launch when
the causal window has at most one 256-token tile. The attention kernel already
writes the normalized result directly for `splits == 1`; the host guard only
removes the no-op dispatch and leaves the split arithmetic, output buffer, and
captured long-context graph unchanged. The production path remains otherwise
unchanged pending resident hash and throughput measurements.

## 2026-09-22 continuation: sixteen-way DFlash selector candidate

The opt-in `LLM_QWEN35_DFLASH_SELECTOR_WARP16=1` path assigns one warp to
each of the selector's sixteen candidates in a single 512-thread block. It
removes the two eight-candidate batches and their extra shared-memory pass,
while preserving the predecessor decode, each candidate's ordered 256-term
score accumulation, and the existing first-max tie break. The previous
256-thread probe remains available as `LLM_QWEN35_DFLASH_SELECTOR_WARP=1`;
the serialized selector remains the default pending resident K=4/K=7 hash,
acceptance, and draft-time checks. HIP device syntax and the host build pass;
no resident gfx1201 runtime is available here.

The selector tail also has an opt-in `LLM_QWEN35_DFLASH_SELECTOR_FUSED=1`
candidate. It joins the vocabulary-logit and rank-256 hidden Q4_K projections
after one activation quantization, while retaining each range's row layout and
reduction order. Its seven-row K=7 shape uses a fixed activation-count
specialization with the same arithmetic. The serialized pair remains the
default pending resident selector hashes and draft timing.

## 2026-09-22 continuation: five-row Q4_K sidecar projection

K=4 DFlash2 windows have five proposal rows (the anchor plus four mask rows),
so their Q4_K/Q8_1 projection now uses a compile-time five-row HIP kernel. It
keeps the multi8 arithmetic and output layout exact while removing the three
unused accumulators and row-count branches. Two warm K=4 repeats retained
124/124 acceptance and hash `44915ec1039a64c8`; draft time fell to
224.1--226.8 ms and decode reached 56.75--57.06 tok/s, versus the prior
274.5--283.8 ms and 55.51--56.19 tok/s. A K=7 sanity gate still produced
140/134, hash `44915ec1039a64c8`, and 74.79 tok/s, so the specialization is
limited to five-row sidecar projections.

## 2026-09-22 continuation: default GQA reuse launch

The ordinary Q8/Q8 GQA path previously launched both the three-head kernel and
the long-context reuse kernel; one returned immediately based on device
position. The reuse kernel is context-polymorphic when that guard is disabled,
so the default path now launches only it and keeps the three-head kernel as a
fallback when reuse is unavailable. The C++ 4K gate stayed exact
(`96b92d606dde5e28`); three repeats measured 40.24--40.31 tok/s versus
40.09 tok/s for the paired-launch control. A matched random 64K gate retained
prefix hash `90178de69a24a76e`, suffix hash `aed3c962c4a6525d`, and measured
35.47 tok/s versus 33.73 tok/s control. The graph ABI and Q8/Q8 arithmetic
remain unchanged.

## 2026-09-22 continuation: asynchronous accepted-row publication

Speculative commit now enqueues the accepted recurrent/conv checkpoint, hidden
state, logits, and position copies on the target stream and returns without a
host-side `hipStreamSynchronize`. The next same-stream decode or proposal still
observes the copies in order; reset and verifier host-logit boundaries retain
their synchronization. The full DFlash2 HTTP/stdio cancellation, cache,
concurrency, sampled, and multi-turn C++ quality suite passed. On the resident
gate, repeated cached K=7 windows reported commit times around 0.23--0.57 ms;
the authoritative token streams and cache restores remained stable.

The hybrid verifier also has an opt-in
`LLM_QWEN35_DELTANET_VERIFY_FIXED128=1` recurrence kernel for the model's
fixed `d_state=128` shape. It removes runtime row-count and checkpoint bounds
checks while retaining the generic kernel's decay, warp-reduction, update, and
row-major checkpoint order. The argument ABI is unchanged, so captured graph
selection remains stable; the generic kernel remains the default pending a
resident rollback/hash and sustained decode comparison.

Verifier workspace allocation is now transactional on first use. If a device
or host allocation fails, queued pointer-table copies are synchronized before
all partial buffers are released and the capacity returns to zero, so a later
request can retry cleanly. Successful graph and checkpoint paths are
unchanged.

## 2026-09-22 continuation: verifier IQ1 Q8₁ activation reuse

The grouped target verifier's IQ1_S gate and IQ1_M up projections consume the
same normalized row block, but each projection used to launch the exact Q8₁
quantizer independently. The verifier now caches the produced activation bytes
and FP16 block sums by source pointer, width, and row count, so the second
projection reuses them. A source-shape change invalidates the cache; layer
entry also clears it because the normalized scratch buffer is reused for the
next layer. `LLM_QWEN35_MTP_IQ1_Q81_REUSE=0` restores the two-quantizer
control for A/B measurements.

The resident DFlash2 gate passes with reuse enabled and disabled: stdio
window/cache/sampling/cancellation/concurrency, multi-turn C++ compile/run,
and HTTP greedy/sampled repeatability all remain PASS. In matched short
resident runs the 113-token cached request reported 433.458 ms verifier time
with reuse and 432.478 ms with the control, within run-to-run noise, so this is
retained as a launch reduction without claiming an end-to-end throughput win.

A fresh random-token 64K run with both IQ1 roles enabled retained prefix
`90178de69a24a76e`, suffix `aed3c962c4a6525d`, and 443.11 tok/s prefill while
measuring 35.44 tok/s decode. That is within the established 35.5 tok/s band,
so the ordinary 40 tok/s target remains open and no IQ1 approximation default
was changed.

## 2026-09-22 continuation: fixed-head DFlash2 attention candidate

The DFlash2 sidecar has a fixed head dimension of 128. I added an opt-in
`qwen35_dflash2_attention_fused_128` kernel that keeps the generic fused
kernel's per-lane dot order, online-softmax update order, and increasing-split
merge, while removing dynamic head-dimension bounds from the cache walk.
`LLM_QWEN35_DFLASH_FUSED_128=1` selects it; the validated generic fused
kernel remains the default until a resident-device A/B confirms both output
quality and a sustained draft-time gain. HIPRTC device syntax, the C build,
and profile checks pass here. The runtime harness could not be rerun because
`/dev/kfd` and the AMD render node are unavailable, so no throughput or
quality claim is made yet.

## 2026-09-22 continuation: sidecar overlap ordering hardening

The opt-in `LLM_QWEN35_DFLASH_OVERLAP_INJECT=1` path now records a
`target_ready` event on the authoritative stream and makes the injection
stream wait before consuming captured features. The existing `inject_done`
event still orders the next proposal before sidecar scratch is reused. Event
creation failure falls back to the serialized injection path, and production
defaults remain unchanged.

The corrected overlap HTTP/stdio run passed the cancellation, cache,
concurrency, and multi-turn C++ quality checks. A matched 4K K=7 benchmark
kept the exact sequence hash `a36ee81632648a4d`; the overlap path measured
27.03 tok/s (draft/verify/commit `839.592/4756.268/100.127` ms) versus
27.21 tok/s serialized (`807.164/4762.647/89.550` ms). The dependency is
therefore retained as safety hardening, while overlap stays opt-in pending a
measured throughput win.

The overlap path now also allocates bounded per-stream activation workspaces
for the DFlash2 injection stream. Accepted feature rows are copied on the
authoritative stream before `target_ready`; injection consumes private
feature, BF16, normalized, and K/V buffers, while the next target proposal can
reuse its normal scratch. The injection stream still publishes `inject_done`
before any sidecar proposal can read the updated KV cache. The path requires
the loaded BF16 injection plans and falls back to serialized injection if the
workspaces cannot be created. Workspace cleanup now nulls each freed pointer,
so a partial allocation failure can safely retry or fall back without a
double-free during teardown. It remains opt-in pending a resident-GPU
throughput and quality run.

Overlap initialization now tracks which stream and events were created by the
current attempt. A retry after partial setup preserves valid pre-existing
handles, and final teardown uses the same pointer-nulling workspace helper.
The hipBLASLt plan cache now stores heap-stable plan entries, so a concurrent
shape miss cannot rehash the cache underneath an in-flight sidecar enqueue.
This is lifecycle hardening only; scheduling and the opt-in gate are unchanged.

## 2026-09-22 continuation: native IQ3 Q/K/V projection candidate

The ordinary one-token attention path now has an opt-in
`LLM_QWEN35_IQ3_QKV_FUSED=1` candidate for layers whose Q, K, and V matrices
are all IQ3_XXS with the same input width. A single native kernel shares the
staged IQ3 codebook and the already-quantized Q8_1 activation across the three
output ranges while retaining the existing per-row dot and reduction order.
The production path remains on the three-launch dispatch until a resident
gfx1201 run proves identical 4K/64K hashes and a sustained projection win.

For the IQ2_XS target's common IQ1_S gate/IQ1_M up pair, dense FFN decode also
has an opt-in `LLM_QWEN35_IQ1_GATEUP_FUSED=1` Q8_1 kernel. It shares the
quantized activation and one launch while keeping each IQ1 layout, affine
correction, and warp reduction separate. The candidate is enabled only when
both role-level IQ1 Q8_1 switches are already active and MMQ scale overrides
are absent; all defaults and the validated MMQ path remain unchanged.

The same native kernel now has an opt-in
`LLM_QWEN35_IQ3_GATEUP_FUSED=1` gate/up form for dense FFN layers where both
projections are IQ3_XXS with matching shapes. It covers the two output ranges
in one launch, stages the IQ3 codebook once, and leaves the native Q8_1 input
and per-row reduction arithmetic unchanged. It is diagnostic only pending a
resident 4K/64K hash and throughput comparison; the ordinary two-launch path
remains the default.

Mixed-IQ attention layers also expose
`LLM_QWEN35_IQ3S_QKV_FUSED=1` when Q, K, and V are all IQ3_S with a common
input width. The native kernel retains IQ3_S code decoding, scale correction,
and warp reduction order while sharing the Q8_1 activation launch across the
three output ranges. It remains diagnostic until a resident gfx1201 run
confirms exact logits/tokens and a sustained 64K projection improvement;
standalone IQ3_S dispatch remains the default.

The same attention dispatch has an opt-in
`LLM_QWEN35_IQ4XS_QKV_FUSED=1` path for all-IQ4_XS Q/K/V layers with a common
input width. It preserves IQ4_XS's eight virtual warp passes, nibble lookup,
affine correction, and reduction order while sharing native Q8_1 activation
staging. It is diagnostic pending resident logits/tokens and 64K timing; the
standalone IQ4_XS path remains the default.

## 2026-09-22 continuation: opt-in DFlash2 Q/K/V projection fusion

DFlash2 proposal rows now have an opt-in `LLM_QWEN35_DFLASH_QKV_FUSED=1` path
for Q4_K Q/K/V matrices. It quantizes the shared activation tile once and
selects the three independent output ranges inside one Q4_K grid, preserving
the existing per-row dot, affine correction, and reduction order. K=4 uses a
five-row specialization; other proposal widths use the eight-row-capable
kernel. The default sidecar path remains three launches until a resident
gfx1201 run confirms identical K=4/K=7 hashes and a sustained draft-time win.

An opt-in `LLM_QWEN35_DFLASH_GATEUP_FUSED=1` path reuses the same exact fused
kernel for the dense Q4_K gate/up pair, with the V output range disabled. It
preserves the five-row specialization for K=4 and the eight-row arithmetic for
K=7 while removing one gate/up launch. The serialized gate, up, and SiLU path
remains the default pending resident hash and draft-time A/B data.

## 2026-09-22 continuation: opt-in IQ2_XS down/residual fusion

The ordinary FFN down path now has an opt-in
`LLM_QWEN35_FUSED_DOWN_RESIDUAL=1` IQ2_XS kernel for the native Q8_1 route.
It keeps the existing IQ2_XS codebook, per-block scale, dot, and warp
reduction order, then adds the result directly to the live residual instead of
writing `d_xb` followed by a separate add launch. The helper requires the
prepared native Q8_1 activation and otherwise falls back automatically. The
default path is unchanged until a resident random-64K hash/performance A/B
proves the launch and temporary traffic savings.

The opt-in sidecar overlap path now reads captured feature rows directly after
the target-ready event instead of copying them into a second buffer. The
injection stream retains private `x`/norm/K/V workspaces and its inject-done
event still gates the next proposal before target scratch can be reused.

## 2026-09-22 continuation: grouped verifier split-count hoist

The verifier-only four- and eight-row combine candidates now compute each
adaptive split count once during shared metadata setup and reuse it for both
metadata loading and the final merge. The gated recurrent variant has the same
hoist. Split selection, metadata order, packed-F16 accumulation, and output
arithmetic are unchanged; the captured generic combine path is untouched.
Host build and gfx1201 HIP syntax checks pass. Resident hash and timing A/B
results remain pending because this environment has no `/dev/kfd` device.

## 2026-09-22 continuation: DFlash K=4 four-row projection candidate

K=4 DFlash2 selector/output projections have four rows after the anchor is
removed, but the generic Q4_K/Q8_1 sidecar kernel kept eight accumulators live.
An opt-in `LLM_QWEN35_DFLASH_Q4K_MULTI4=1` kernel specializes that exact
arithmetic to four rows, preserving weight decode, affine correction, output
layout, and warp reduction order. The existing five-row K=4 proposal kernel,
K=7 path, and serving defaults are unchanged. Host build, profile tests, and
gfx1201 HIP syntax checks pass; resident hash and draft-time A/B validation is
still required.

## 2026-09-22 continuation: opt-in IQ2_XXS Q/K/V projection fusion

The ordinary one-token path now exposes
`LLM_QWEN35_IQ2XXS_QKV_FUSED=1` for layers whose Q, K, and V matrices are all
IQ2_XXS with a common input width. One native grid shares the 256-entry
codebook staging and prepared Q8₁ activation; each row retains the standalone
IQ2_XXS lookup, scale, integer correction, and warp reduction order. The
serialized dispatch and production defaults remain unchanged pending resident
4K/64K hashes and throughput.

## 2026-09-22 continuation: device-resident accepted-position publication

MTP commit now publishes `d_position` from the accepted row in the verifier's
device position scratch, avoiding a redundant host-to-device upload. The host
`cur_position` value, state-row copies, hidden/logit publication, stream order,
and rollback semantics are unchanged; a host upload remains as a defensive
fallback if position scratch is absent. Host build and profile checks pass.

## 2026-09-22 continuation: opt-in parallel DFlash2 selector

The DFlash2 selector now has an opt-in
`LLM_QWEN35_DFLASH_SELECTOR_WARP=1` geometry. Eight 32-thread candidate groups
decode successor Q4_K rank values in parallel, while each candidate's lane-zero
accumulation still visits all 256 terms in the original order. The selector
tie break and predecessor chain are unchanged; the serialized 32-thread path
remains the default pending resident K=4/K=7 hashes and draft-time A/B data.

The parallel selector now decodes the predecessor's 256-rank Q4_K vector only
in its first candidate group; the other seven groups reuse the same shared
values. This removes duplicated selector loads without changing candidate
scores, tie breaks, or the opt-in/default boundary.

The verifier commit boundary also has an opt-in
`LLM_QWEN35_COMMIT_FUSED_COPY=1` kernel. It retains the existing per-layer
checkpoint pointer-table copies and folds the accepted hidden/logit device
copies into that launch; position publication remains ordered by the existing
host-to-device copy. The reference three-copy path remains the default until
resident accepted-row hashes and commit timing are measured.

The long-context verifier also has an opt-in
`LLM_QWEN35_VERIFY_COMBINE_GROUPED=1` combine geometry. Four adjacent rows
share one verifier block and metadata tile while retaining the existing
per-row split order and arithmetic. The split kernel and captured generic
decode ABI are unchanged; the reference combine remains the default pending
resident verifier hashes and timing.

The grouped combine now loads all four metadata rows before one block barrier
instead of synchronizing after each row. This is an exact barrier-hoist
micro-optimization: the per-row maximum/scale/reduction order is unchanged,
and the opt-in/default boundary remains unchanged.

`LLM_QWEN35_VERIFY_COMBINE_GROUPED=2` now exposes an eight-row verifier-only
combine probe using the same arithmetic and one post-load barrier. Mode `1`
continues to select the four-row candidate; unset or `0` retains the reference
per-row combine. The eight-row mode is intentionally unvalidated and does not
change the serving default. Both grouped launches now pass zero dynamic shared
memory because their metadata tiles are statically allocated; this avoids
reserving an unused duplicate tile in the launch configuration.

## 2026-09-22 continuation: IQ2_XXS block-shape probe

I tested 128- and 512-thread blocks for the native one-row IQ2_XXS kernel
using the same 4K random-token gate and Q8/Q8 graph path. Both shapes kept
the exact prefix/suffix hashes `1c891c2232aa1b7f`/`ab4dd24f5cdf0b2c`; three
repeat decode means were 39.68 tok/s at the retained 256-thread shape and
39.64 tok/s at 512 threads. The alternate launch geometry was removed, so
the mixed projection traffic target remains open.

## 2026-09-22 continuation: long-context GQA reuse probe

A six-query-head Q8 reuse kernel was prototyped for the 6:1 Qwen3.8 GQA
layout. It retained the ordinary 4K and random-token 64K sequence hashes, but
measured about 42.5 tok/s at 4K and 35.66 tok/s at 64K versus the established
three-head path near 43.0 and 35.76 tok/s. The prototype was reverted; the
existing exact three-head reuse path remains the production choice while the
remaining 64K gap is addressed through grouped projection traffic and verifier
tail work.

## 2026-09-22 continuation: fused DFlash2 split merge

DFlash2 attention now has a verifier-only fused path for the production
short-window and twelve-split schedules.  One block assigns a warp to each
split, keeps the partials in LDS, and performs the same increasing-split
online-softmax merge before publishing the output.  This removes the global
partial write/read and the separate combine launch without changing target
verifier state or the captured target graph ABI.  The path is selected by
default for at most twelve splits; `LLM_QWEN35_DFLASH_FUSED_ATTN=0` restores
the previous two-kernel path for A/B testing.

The exact 4K greedy K=7 gate measured 82.76 tok/s fused versus 82.68 tok/s
with the control path and retained sequence hash `44915ec1039a64c8`.  Seeded
sampled K=7 measured 70.89 versus 70.85 tok/s with hash
`72a11474a3a222b5`; K=4 measured 55.90 versus 55.81 tok/s with the same
sampled hash.  The full HTTP/stdio, context-cache, cancellation, concurrency,
and multi-turn C++ quality harness passes.  Ordinary random-token 64K remains
unaffected at 35.76 tok/s with prefix/suffix hashes
`90178de69a24a76e`/`7463f176c9b85ba3`; the strict 40 tok/s ordinary target
and target-verifier long-context fusion remain open.

## 2026-09-22 continuation: DFlash2 long-window split retune

The DFlash2 sidecar now uses twelve attention partitions once its 2,048-token
window reaches 1,024 tokens; the 1/4 split schedule for shorter windows is
unchanged.  An opt-in `LLM_QWEN35_DFLASH_ATTN_SPLITS` override remains for
repeatable A/B runs.  On the fixed 256-token K=7 gate, twelve partitions
measured 56.57 tok/s versus 56.21 with eight; the EOS-limited gate measured
82.57 tok/s versus 82.14.  The K=4 fixed gate measured 47.50 tok/s, and the
normal K=4 gate measured 59.69 tok/s.  Greedy K=7 retained sequence hash
`44915ec1039a64c8`; seeded sampled K=7 and K=4 retained
`630b7cbc72230e0d`.  The production-default K=7 sampled run measured
69.28 tok/s with `DFLASH2 sampled verifier=exact-window`.

A fresh random-token 64K K=7 run with the same native Q8/Q8 target path
confirmed that low acceptance (45/121) made the sidecar reach only 24.02
tok/s.  The generation harness now disables DFlash2 at target position 32,768
and resumes ordinary target decode at the transaction boundary.  The guarded
64K run retains prefix hash `90178de69a24a76e`, emits the same suffix hash
`5821d77a630592cb`, and measures 35.67 tok/s.  Short greedy and sampled K=7
gates remain 83.38 and 69.45 tok/s with hashes `44915ec1039a64c8` and
`630b7cbc72230e0d`; the HTTP/stdio quality harness still passes.

Two other small table-staging probes were rejected.  Staging the native IQ3_S
512-entry grid measured 42.88 tok/s at 4K and 35.74 tok/s after a random 64K
prefix, versus 43.00 and 35.84 controls.  Staging the larger IQ2_S 1,024-entry
grid measured 42.47 tok/s at 4K.  Both retained exact prefix/suffix hashes and
were reverted; their existing occupancy choices remain production defaults.

## 2026-09-22 continuation: IQ2_XS launch-bounds probe

The native one-row IQ2_XS matvec now carries `__launch_bounds__(512, 1)` so
the compiler can budget registers for the production 512-thread shape.  The
change preserves the existing reduction and dequantization order.  A 4K
random-token gate retained prefix `1c891c2232aa1b7f` and suffix
`f44846dacf013e9e`, measuring 43.00 tok/s; the 64K gate retained prefix
`90178de69a24a76e` and suffix `7463f176c9b85ba3`, measuring 35.84 tok/s at
445.82 tok/s prefill.  The pinned C++17 merge-intervals gate remains
byte-identical (`44915ec1039a64c8`, output SHA-256
`4a0cb461966fae9a9d9da3b73c1b0c686ce8ee9ac3895c228bc6a653bc99a354`) and
completed at 41.95 tok/s.  The small long-context gain is within run
variance but the launch contract is quality-safe; keep it while continuing
the larger grouped-projection and verifier-tail work.

I also tried matching launch bounds on the native IQ2_S, IQ3_XXS and IQ3_S
one-row kernels.  Two matched 4K random-token runs were 42.84 tok/s each,
versus the 42.9--43.0 tok/s control, with the exact
`1c891c2232aa1b7f`/`f44846dacf013e9e` hashes.  The bounds were removed; those
formats retain their existing shape-specific occupancy choices.

A separate 128-thread Q8 attention combine A/B kept the 4K suffix exact at
42.97 tok/s and the 64K prefix/suffix hashes exact at 35.78 tok/s versus
35.77 tok/s for the 256-thread merge.  The extra value load made it neutral,
so the kernel and its opt-in dispatch were removed.

Finally, `__launch_bounds__(512, 2)` on IQ2_XS was compared with the retained
`512, 1` bound.  Two 4K runs ranged from 42.82 to 42.91 tok/s with the exact
hash, so the stricter occupancy request was reverted as noise.

The same bounded probe on native IQ2_XXS (`__launch_bounds__(256, 1)`) retained
the 4K hash but measured 42.94 and 42.87 tok/s.  It was reverted; the mixed
projection traffic remains the dominant open target.

Native Q2_K was similarly tested with `__launch_bounds__(256, 1)`.  The 4K
random gate retained `1c891c2232aa1b7f`/`f44846dacf013e9e` at 42.91 tok/s,
which did not exceed the control, so it was reverted.

An opt-in IQ3_S gated-attention K/V pair kernel preserved the pinned C++ hash
and lifted the short random gate to 42.93 tok/s, but the sustained 64K gate
was 35.74 tok/s versus 35.77 tok/s control with the same hashes.  The pair
kernel and dispatch were removed because its launch saving does not survive
the long-context attention cost.

## 2026-09-22 continuation: IQ1_M micro-tuning and fusion audit

The default one-row IQ1_M F32 kernel now marks its output, weight and
activation buffers non-aliasing and explicitly unrolls both four-value FMA
halves.  This leaves the reduction and FMA order unchanged.  A matched 4K
random-token run moved from 43.64 to 43.72 tok/s with the same sequence hash
`8b48e9c489798cfe`; the short exact C++ gate remained byte-identical to the
pinned output (`44915ec1039a64c8`).  The 64K block-size A/B was neutral
(35.77 tok/s at 256 threads versus 35.74--35.80 at the tested alternatives),
so the runner's production launch geometry remains unchanged.

I also tested a one-launch IQ1_S Q8_1 gate plus IQ1_M F32 up fusion.  It was
bit-identical to the non-fused control, but the exact 4K C++ gate measured
41.74 tok/s versus 41.90 tok/s control, so the kernel was removed rather than
adding a slower opt-in path.  The launcher now forwards the existing decode
and prefill profile diagnostics through the production wrapper, making stage
timing reproducible without changing serving defaults.

A separate 64-thread split-combine kernel kept the random 64K prefix/suffix
hashes exact but measured 35.64 tok/s versus 35.77 tok/s for the existing
256-thread merge.  It was removed; the long-context verifier tail still needs
a fused design that reduces global partial traffic without changing the
captured graph ABI.

The scalar DeltaNet recurrence also received a restricted-pointer and
four-way-unroll A/B.  Its 4K random gate stayed exact but slipped to 42.80
tok/s from the 42.90 tok/s control, so the recurrent kernel remains unchanged;
accepted-row checkpoint publication is still the safe optimization boundary.

## 2026-09-22 continuation: bounded overlap and rejected candidates

The ordinary IQ1 audit now reuses the exact Q8_1 activation bytes and FP16
block sums across the IQ1_S gate and IQ1_M up projections while the existing
gate/up reuse scope is open.  The cache is keyed by the producer pointer and
column count and is cleared when the scope closes, so pointer reuse across
layers cannot consume stale data.  This remains selected by
`LLM_QWEN35_FFN_IQ1_Q81=1` because the Q8_1 activation approximation changes
full logits even though it retained the pinned greedy C++ output and token
file byte-for-byte (`44915ec1039a64c8`, output SHA-256
`4a0cb461966fae9a9d9da3b73c1b0c686ce8ee9ac3895c228bc6a653bc99a354`).  On the
same 4K/64-token exact gate it measured 42.24 tok/s versus 39.74 tok/s for
the disabled control, and the random-token 65,536-depth gate retained prefix
hash `90178de69a24a76e`, suffix hash `4f46d4fe27743a5b`, and measured
36.13 tok/s versus 34.18 tok/s.  A seeded temperature-0.6 C++ run also
retained the sampled sequence hash `630b7cbc72230e0d` and output SHA-256
`ddd1752b6c2a44251b659516b5937fdaa0e84f464530607e493abf8bbc37c9ac`.
Production defaults remain unchanged until the broader sampled/logit quality
matrix is rerun.

A DFlash2 proposal now replaces the anchor-plus-mask IQ1_M embedding launches
with one exact row-batched launch. The anchor and mask IDs use selector-owned
scratch, and selector output overwrites that scratch only after the embedding
has consumed it. The pinned greedy K=7 gate is exact at 83.69 tok/s warm
(140 drafted/134 accepted, hash `44915ec1039a64c8`); seeded sampled K=7 is
exact at 70.47 tok/s (140 drafted/115 accepted, hash `630b7cbc72230e0d`).
The full HTTP/stdio and C++ quality matrix passes; larger draft projection
cost remains under measurement.

The sidecar commit path now has an opt-in event-ordered injection stream via
`LLM_QWEN35_DFLASH_OVERLAP_INJECT=1`.  Injection runs on a nonblocking stream
while the authoritative target publishes its accepted-row checkpoints; the
next DFlash proposal waits on the injection event before reusing sidecar
features, K/V scratch, or the private cache.  The default remains serialized
because the matched random-4K K=7 gate was exact but slightly slower with the
extra stream/event work: serial `draft/verify/commit=158.810/1108.634/22.548`
ms and 49.58 tok/s versus overlap `158.912/1110.527/22.636` ms and 49.51
tok/s.  Both retained suffix hash `cd772dbc6c6e4776`.

Two bounded optimization probes were rejected.  A fixed-count Q4_K draft
projection for the eight-row K=7 window measured 158.834 ms versus 158.810 ms
for the generic kernel with the same accepted rows and hash.  A 32-way fixed
Q8 attention split at random 64K retained prefix/suffix hashes
`90178de69a24a76e`/`7463f176c9b85ba3` but fell to 35.39 tok/s from the adaptive
35.80 tok/s baseline.  The DeltaNet alias hint was also neutral at 42.21--42.26
tok/s on the exact 4K gate.  These paths remain reverted or opt-in; the next
high-value work is still a grouped mixed-type projection kernel and a
verifier-only attention tail fusion.

## Q8/Q8 attention split cap at 64K (2026-09-21)

The native gfx1201 Q8/Q8 decode selector now caps the validated 64K serving
window at 64 K/V partitions instead of the adaptive 128-way schedule.  This
reduces split/combine work without changing the short-context schedule or
token stream.  The 4K random-depth gate remains exact (`d7284dcf729e565e`),
with 525.75 tok/s prefill and 42.08 tok/s decode.  The production 64K gate
retains prefix hash `90178de69a24a76e`, suffix hash `051e7338c23a544e`, and
passes at 444.10 tok/s prefill and 35.08 tok/s ordinary decode.  Three repeated
64K runs with the same schedule measured 35.05--35.09 tok/s and the same hash.
The strict ordinary 40 tok/s long-context target remains open; the next
high-value work is still mixed-type projection reuse and verifier-tail fusion.

## Packed Q8/Q8 KV scales (2026-09-21)

The Q8 K/V cache now stores its per-group scales as rounded FP16 values and
the native attention path consumes those packed scales directly.  This halves
scale-cache bandwidth without changing the quantization contract.  The
reference attention gate remains bitwise clean across 49,188,864 comparisons,
including 64K split and multi-query reuse cases.  A captured zero-depth decode
completed 506 generated tokens at 43.1--43.2 tok/s with stable output.  The
full random 64K runner gate now completes with a 443.33 tok/s prefix and
35.52 tok/s ordinary decode suffix, retaining prefix hash `90178de69a24a76e`
and `Result: PASS`.  The ordinary 40 tok/s long-context target remains open;
the attention differential is bitwise clean at 64K.

## Optimized native DFlash2 (2026-09-21)

The runner now loads the IncoAI Qwen3.8-27B DFlash2 Q4_K_M sidecar and runs
its five-layer block-diffusion graph entirely through HIP.  Target inputs from
layers 6/20/34/48/62 seed a private 2048-token draft KV ring.  The rank-256,
top-16 selector proposes up to seven tokens, while the Q8/Q8
target window remains the only source of emitted tokens and committed state.
Greedy, probabilistic, and coding samplers now all use the batched target
window. Every sampled verifier row matches the former ordinary exact-target
trace bit-for-bit, including all 135 stored 248,320-entry logit rows.

On the 4096-token C clamp prompt, K=4 accepted 37/40 drafts and K=7 accepted
41/42.  Both produced the ordinary target's exact 46-token response, EOS and
sequence hash `15f17d2640c1adfc`; the emitted C is coherent, compiles warning
free as C17 and passes `INT_MIN`/`INT_MAX` boundary cases. The final K=7 run
measured **81.05 tok/s decode and 540.43 tok/s prefill**. K=4 measured 54.49
and 537.42. A recent ordinary native baseline measured 39.55 and 533.19.
The upstream llama.cpp server path measured 16.54 tok/s at K=4 with the same
37/40 acceptance, versus its 25.88 baseline.  Native K=4 is 3.29x faster than
upstream DFlash2, and K=7 is 105 percent faster than the recent ordinary native
decode baseline.  Both short-context performance targets are met.

The broader 4096-token C++ merge-intervals gate now passes at K=4 and K=7.
Greedy output matches the pinned llama.cpp bytes and token IDs with SHA-256
`4a0cb461966fae9a9d9da3b73c1b0c686ce8ee9ac3895c228bc6a653bc99a354`;
temperature-0.6 output uses the exact multi-row target window and matches SHA-256
`ddd1752b6c2a44251b659516b5937fdaa0e84f464530607e493abf8bbc37c9ac`.
Both functions pass ASan/UBSan, fixed edge cases and 10,000 randomized cases.
Warm K=7 runs sustain 607.37–608.15 tok/s prefill and 81.68–81.82 tok/s
greedy decode; sampled decode sustains 605.37–605.85 and 68.67–68.78 tok/s.
K=4 sustains 605.43–607.42 tok/s prefill, 59.43–59.50 greedy decode, and
55.79–55.82 sampled decode. The early-context
retrieval gate also emits exactly `ZEPHYR-7319` at K=7 with the pinned token
sequence and EOS.

Three fixes closed sampled verifier parity. Non-FFN IQ1_S projections now use
the scalar path's MMQ-scale arithmetic in one batched launch. Q8/Q8 verifier
attention uses the native query grid so each causal row retains ordinary
decode's adaptive split count. Each captured-graph projection restages the
shared Q8_1 scratch because source-pointer identity did not prove that earlier
scratch contents survived later writes. The exact weight-reuse kernels remain
enabled.

The prefill regression was a fast-math reassociation in the fixed 128-wide
GDA kernel.  A fixed-bound four-iteration loop now matches the generic
operation order while retaining precomputed-decay specialization.  The new
differential checks both raw-alpha scalar and precomputed-decay contracts over
19,537,920 bitwise state/output comparisons.  This restores the pinned sampled
sequence and keeps warmed prefill above 600 tok/s.

The exact verifier now reuses decoded weights across up to eight rows for
Q2_K, IQ1_S, IQ1_M, IQ2/IQ3 and IQ4_XS.  Eight-row IQ kernels specialize the
quantization format at compile time, eliminating runtime codebook branches.
Compact Q2_K, IQ2 and IQ3_S schedules finish one verifier query at a time to
lower accumulator pressure without changing the reference reduction order.
The fixed-eight Q2_K/IQ and IQ4_XS multi-row kernels now use eight-wave,
256-thread blocks. In matched traces this reduced aggregate fixed-eight
projection time from 223.706 to 222.336 ms and IQ4_XS time from 47.068 to
46.624 ms without changing any validated output bit.
RMSNorm and residual-plus-RMSNorm use one batched launch with an independent
block and unchanged reduction per row. The DFlash draft quantizes each row to
Q8_1 and evaluates Q4_K projections with packed gfx1201 integer dots, reusing
decoded weights across eight rows and quantized inputs across adjacent
projections. It also reuses K/V values across four attention rows. Exact target
attention now loads each old Q8 K/V row once while evaluating up to eight
adjacent verifier queries.  It retains the pinned query quantization, online
softmax, packed-F16 accumulation and split-combine order.  K=7 timing for the
final 46-token response is draft 77.239 ms, target verify 478.857 ms and
commit 10.624 ms, for 566.720 ms total. The exact attention differential
passes 46,743,552 values; its eight-query operator takes 213.382 microseconds
at 4K and 3.076784 milliseconds at 64K with eight splits.

Qwen3.5 decode attention preparation now fuses Q/gate deinterleave, Q and K
RMSNorm, Q and K M-RoPE, and Q8/Q8 K/V storage into one exact kernel per
attention layer. This replaces six launches with one across all 16 attention
layers, removing 80 launches per target row while preserving the original
reduction, trigonometric, Q8 scale-rounding and integer-conversion order. The
fresh K=4 and K=7 greedy and sampled C++ gates remain byte-identical to the
pinned llama.cpp fixtures and pass strict compilation, sanitizers, fixed cases
and 10,000 randomized cases. Zero-depth ordinary decode remains effectively
flat at 42.53--42.65 tok/s; the 4K sampled exact-target path now clears 40
tok/s. Artifacts: `tmp/qwen38/dflash2-qkprep-{k4,k7}/`.

The historical fixed-eight-split DFlash K=7 path sustained 49.74 tok/s after a
fully processed 65,536-token random prefix, with suffix hash
`2ddd068dca63669a`.  That run predates exact per-query adaptive split
selection and is not the current output oracle.  The current generic exact
baseline is 33.25 tok/s.  A dedicated captured shared-K/V graph is now chosen
only when every adjacent row has the same adaptive split count; selector
boundaries use the generic per-query graph.  This raises exact random-64K
decode to 39.89--39.93 tok/s while retaining prefix hash
`90178de69a24a76e`, current suffix hash `1c68ea2ff63ba5ab`, and 289 drafted / 213
accepted tokens.  Prefix processing remains 443.45--444.22 tok/s.  Ordinary
scalar decode now stages the IQ2_XXS, IQ2_XS and IQ3_XXS codebooks in LDS. A
512-token zero-depth run sustains 41.90--42.03 tok/s with unchanged hash
`c08c332d32a63532`. The exact three-head attention kernel now computes each
K/Q scale product once per four packed dots while preserving the dot and
accumulation sequence. After the real 64K prefix, a 256-token suffix sustains
34.08 tok/s with retained hash `f4b35758fb99e6db`; the prefix sustains 445.67
tok/s with hash `90178de69a24a76e`. The 40 tok/s long-context target remains
open.

The exact three-head attention kernel now also reuses each head's packed
probability across both 128-dimension value tiles. This removes half of those
LDS reads while retaining every packed-F16 FMA. The 49,188,864-comparison
differential remains bitwise clean; the 64K/128-split operator measures
321.8--323.6 microseconds per layer. A fresh full random-depth run sustains
443.63 tok/s for the 65,536-token prefix and 34.21 tok/s for the 256-token
suffix, with the same `90178de69a24a76e` and `f4b35758fb99e6db` hashes.

The two exact F16 projections that form each recurrent layer's alpha and beta
vectors now share one flattened launch.  Every row retains the
`matvec_f16_llama_f32` FMA and XOR-reduction order.  A 65-row kernel trace
drops these projections from 6,144 launches and 23.102 ms to 3,072 launches
and 12.492 ms, saving about 0.163 ms per decoded row.  Zero-depth decode rises
from the immediate 42.65--42.77 tok/s baseline to 42.83--43.04 tok/s, with the
pinned 256-token hash `3c53b75f283cb9b0` unchanged.  Trace artifact:
`tmp/qwen38/ordinary-decode-profile-f16pair/`.

Dense FFN decode now combines SiLU multiplication with the native Q8_1
staging consumed by 58 of the model's 64 down projections.  Each 32-value
wave keeps the original fast-math SiLU expression, then applies the protected
division and FP16 scale-rounding contract used by the standalone exact
quantizer.  A split-path diagnostic produces bitwise-identical values for all
248,320 final logits, and the pinned 256-token hash remains
`3c53b75f283cb9b0`.  The 64-row trace removes 3,712 quantizer launches, exactly
58 per row; matched three-repeat decode improves from 42.45--42.97 tok/s
(42.79 mean) to 42.81--43.36 tok/s (43.11 mean).  Trace artifact:
`tmp/qwen38/ordinary-decode-profile-siluq81/`.

The gated RMSNorm/SiLU producer for all 48 recurrent output projections now
also stages its exact native Q8_1 input.  Preserving the original 128-thread
reduction and load loop is required: an algebraically equivalent first draft
moved low activation bits and failed the logit gate.  The corrected fused and
split paths are bitwise identical across all 248,320 final logits (SHA-256
`5b5f2f1a334ae644ac5633908e3d447c6d741c764a61dc0e9573addf699553c0`)
and keep the pinned 256-token hash `3c53b75f283cb9b0`.  A 64-row trace removes
3,072 launches, exactly 48 per row.  Matched three-repeat decode is
42.62--43.36 tok/s (43.07 mean) fused versus 41.64--43.14 tok/s (42.60 mean)
split.  Trace artifact: `tmp/qwen38/ordinary-decode-profile-ssmq81/`.

Twelve of the sixteen gated-attention output layers now combine sigmoid
gating with exact native Q8_1 staging; the four IQ1 output layers keep their
specialized quantizer.  The fused/split logit files remain bitwise identical
with SHA-256 `5b5f2f1a334ae644ac5633908e3d447c6d741c764a61dc0e9573addf699553c0`,
and all three 256-token repeats retain `3c53b75f283cb9b0`.  Matched means are
43.33 tok/s fused and 43.21 tok/s split.  The 64-row trace removes exactly 768
launches, 12 per row.  Trace artifact:
`tmp/qwen38/ordinary-decode-profile-attngateq81/`.

The one-row IQ2_XS, tall IQ2_S, and common IQ3_S projections now use measured
shape-specific wave counts while retaining one output row per wave. The
tuned/eight-wave A/B is bitwise identical across all 248,320 logits with
SHA-256 `5b5f2f1a334ae644ac5633908e3d447c6d741c764a61dc0e9573addf699553c0`.
Matched traces save about 0.23 ms per row; a real 65,536-token random-prefix
run keeps prefix/suffix hashes `90178de69a24a76e`/`f4b35758fb99e6db` and
measures 441.44 tok/s prefill plus 34.98 tok/s ordinary decode. Use
`LLM_QWEN35_IQ_SHAPE_THREADS=0` only for the eight-wave diagnostic fallback.

The latest verifier pass batches the target's IQ1_M token embeddings into one
two-dimensional launch and publishes accepted convolution plus recurrent
checkpoints with one kernel across all recurrent layers.  K=4 and K=7 retain
the pinned greedy sequence hash `44915ec1039a64c8`; the seeded-sampled gate
retains sequence hash `630b7cbc72230e0d`, output SHA-256
`ddd1752b6c2a44251b659516b5937fdaa0e84f464530607e493abf8bbc37c9ac`,
and token SHA-256
`fb7d8aeda396cdba5dd65b492a396ed3f91ae4312ea0a52d77be86355b4c7ee0`.
K=7 measures 81.20 tok/s in the post-change exact gate; the three warm
embedding-batch runs measured 81.42--82.49 tok/s.

Prompt feature capture now shares the tapped layer's exact RMSNorm kernel,
removing one standalone capture launch at each of the five taps.  The pinned
4K response stays byte-identical and measures 536.17 tok/s cold,
610.86--612.05 tok/s warm, and 81.12--82.37 tok/s decode.  A fully processed
65,536-token random prefix retained hash `90178de69a24a76e` at 444.17 tok/s,
so the fusion does not reduce the long-context prefill result.

Two broader experiments were rejected.  Extending the generic native
attention kernel ABI for an in-kernel verifier combine caused a graph-time GPU
fault even when its new body was disabled; a future attempt must use a
separate verifier-only kernel.  Skipping the final recurrent checkpoint copy
when every verifier row was processed diverged after 26 generated tokens,
because the live recurrent state is deliberately left at the transaction
origin.  The exact checkpoint publication stays in place.

The exact long-context verifier now owns separate captured graphs for
equal-split and selector-boundary windows.  The shared-K/V graph runs only
when all adjacent causal rows select the same ordinary adaptive split count;
otherwise the generic per-query graph remains authoritative.  At random-64K
depth this raises decode from 33.25 to 39.89--39.93 tok/s while preserving
prefix/suffix hashes `90178de69a24a76e`/`1c68ea2ff63ba5ab`, 289 drafted / 213
accepted tokens, and 443.45--444.22 tok/s prefill.  The 4K K=7 gate retains
82.67 tok/s greedy and 66.46 tok/s sampled with trace I/O; K=4 retains the
greedy hash at 59.75 tok/s.  The sampled output, token, and full-logit SHA-256
values remain pinned.

On 2026-09-22 the DFlash2 sidecar attention schedule was retuned independently
of the target verifier: a full 2,048-token draft window now uses eight splits
instead of sixteen, while shorter windows keep the existing 1/4-split policy.
The exact random-token 64K K=7 gate retained prefix hash
`90178de69a24a76e` and suffix hash `1c68ea2ff63ba5ab`, drafted 289 and
accepted 213, and improved from 39.91 tok/s to 41.01 tok/s.  The matched 4K
K=7 run retained hash `15f17d2640c1adfc`.  This changes only the sidecar draft
attention launch; target verifier arithmetic, graph ABI, and authoritative
output remain unchanged.

The host-side verifier split selector now mirrors the native Q8 kernel's
64-split cap for the 64K--<96K window.  This prevents a shared captured graph
from crossing the 96K transition with a stale 128-split assumption; the
gated HIPRTC combine helper follows the same schedule, and the ordinary 4K
and dense-MTP exact gates remain passing after the change.

The verifier now has an opt-in
`LLM_QWEN35_VERIFY_FUSED_SPLIT_COMBINE=1` candidate. A verifier-only block
owns one head and four adjacent rows, evaluates all split partitions, writes
the existing parts/meta buffers, and performs the exact increasing-split merge
before publishing the gated output. The graph workspace ABI and production
serialized split/combine path are unchanged; resident long-context hash and
timing validation are still required. Its one-split case now publishes the
normalized gated value directly from the local accumulator, avoiding a
partial-buffer round trip for short verifier windows.

Device-guard dual launches, fixed split pinning, shared combine scales, a
one-wave combine, and fused draft/verify synchronization were all measured
and rejected.  They were exact, but none beat the selected-graph result.  The
per-thread verifier length arrays were also moved to LDS to test whether the
eight-query accumulator was register-bound.  That variant kept the random-64K
prefix/suffix hashes exact and raised prefill to 445.21 tok/s, but verifier
time rose to 5927.105 ms and decode fell to 39.26 tok/s, so it was reverted.
The ordinary one-row path remains below 40 tok/s at 64K; the DFlash2 sidecar
now clears 40 tok/s after its separate eight-split retune below.

Remaining optimization items, in measured priority order:

1. Raise ordinary random-64K decode from 35.52 tok/s to 40+ tok/s.  The
   dominant cost is still one-row IQ2/IQ3/IQ1/IQ4 and Q2_K projection traffic;
   evaluate exact grouped or multi-row mixed-type kernels with shared
   activation staging and codebook layout changes.
2. Fuse the long-context verifier attention split/combine tail while keeping
   the pinned logits and sampled output bit-identical.  Equal-split windows
   already use the dedicated captured shared-K/V graph; the remaining work is
   the split-partial/combine boundary.  Keep it in a verifier-only kernel so
   captured generic-decode graph ABIs remain stable.
3. Reduce the remaining DFlash draft cost. Top-k and selector decisions are
   already on the GPU; the selector now shares the predecessor's decoded
   256-rank Q4_K vector across its sixteen candidate lanes without changing
   accumulation order or output hashes. Repeated 4K K=4 runs remain about
   53--55 ms for the draft phase, so projection and selector work remain the
   material cost after the eight-split 64K draft-attention change. Investigate
   position-parallel draft attention and a
   cheaper draft-cache representation. K=7 already clears 60 tok/s, while
   K=4 remains below that target. The new opt-in
   `LLM_QWEN35_DFLASH_SELECTOR_FUSED=1` candidate joins the vocabulary-logit
   and rank-256 selector-hidden Q4_K projections after one Q8_1 activation
   staging; it preserves the serialized path by default pending resident
   selector hashes and timing.
4. Measure the opt-in sidecar cache-injection overlap on resident gfx1201.
   The capture half is complete, the hipBLASLt bridge owns scratch lazily per
   HIP stream, and explicit target-ready/injection-done events protect feature
   and KV reuse.  Fused K/V injection and mask broadcasting are additional
   diagnostics; retain the serial injection path as the fallback until cache
   hashes, HTTP/C++ quality, and throughput all prove a gain.
5. Revisit dense NextN/MTP scheduling. The current exact implementation now
   improves the pinned 4K IQ2 coding fixture from 38.97 tok/s ordinary to
   47.87--47.93 tok/s, but remains below 60 tok/s and falls to 28.82 tok/s at
   real random-token 64K depth.

2026-09-22 continuation: resident DFlash2 A/B measurements used the same
4K/512 Q8/Q8 coding fixture and two repeats.  K=4 control produced
56.25--58.09 tok/s with draft times 85.8--86.5 ms and hash
`15f17d2640c1adfc`; the opt-in selector projection fusion produced
56.57--58.26 tok/s with the same hash, which is within run-to-run noise and
does not justify changing the default.  The opt-in cache-injection overlap
path was slower at 55.46--57.76 tok/s (same hash), so serial injection remains
the production path.  K=7 measured 83.65--88.14 tok/s with draft times
68.9--70.0 ms and the same exact hash, confirming that the remaining K=4
shortfall is draft cost rather than verifier correctness.  A separate opt-in
`fast` target profile run with the GSQ IQ2 model terminated in a segmentation
fault during graph setup; it is not promoted and is now treated as an
unsupported diagnostic combination until the failing setup path is isolated.
The K=4 resident HTTP/stdio run then passed context-cache restore, greedy and
sampled repeatability, cancellation/concurrency, and both multi-turn and
algorithmic C++ compile/run quality checks; its 121-token window measured
59.58 tok/s end-to-end (93/108 accepted).
Follow-up crash isolation mapped the CPU fault to `hllm_qwen35_mtp_verify_impl`
during graph setup.  It reproduced with scalar settings and reduced
cache/BMAX values while VRAM was only 57 MiB in use afterward, pointing to
invalid MTP verifier state or a stale graph pointer rather than capacity.
The root cause was the fused Q/K norm + M-RoPE launch being selected for a
GSQ layer whose runtime kernel path was not safe on this code object.  The
fusion is now explicitly opt-in through `LLM_QWEN35_QK_FUSED=1`; the default
uses the exact separate deinterleave, Q/K norm, and RoPE sequence.  The scalar
control and fast profile both now complete with identical hash
`8a44087a5472a2e2` (29.42 and 29.25 decode tok/s respectively at the 512-token
smoke gate), so the setup crash is fixed without changing output quality.
The launcher now forwards the existing mixed-IQ QKV and gate/up A/B controls.
At randomized 64K, enabling both fused candidates retained suffix hash
`7463f176c9b85ba3` but measured 35.00 tok/s versus 35.59 tok/s for the default
path, so the candidates remain opt-in.
Fresh dense NextN evidence confirms the long-context bottleneck: K=3 at
randomized 64K drafted 77 rows, accepted 36, spent 2.894 s in verification,
and sustained 20.50 tok/s with exact hash `b63380a1a3e5d3b2`. Enabling both
exact MTP pair-batch projection switches was neutral on the 4K fixture
(49.04 versus 49.07 tok/s, identical hash `96b92d606dde5e28`), so those
switches remain opt-in.
The DFlash K=4 sidecar split sweep retained hash `15f17d2640c1adfc` at all
tested geometries: 4 splits reached 55.10 tok/s, 8 reached 56.80 tok/s, and
12 reached 56.94 tok/s, confirming the 12-split default. The exact ordinary
SSM-fused decode candidate also retained the 64K hash `7463f176c9b85ba3` but
measured 35.57 tok/s, so it remains opt-in.
The post-change llama.cpp HIP differential was rerun successfully: all
49,188,864 bitwise Q8/Q8 comparisons passed, including random-like 64K K/V
patterns, adaptive split counts, and multi-query reuse shapes. This keeps the
Q8/Q8 attention and cache contract intact while the ordinary single-query
projection/SSM bottleneck remains open.
The current binary also passes the resident K=4 DFlash quality suite after
the GSQ guard: HTTP/stdio cache restore, greedy and sampled repeatability,
cancellation/concurrency, multi-turn C++, and algorithmic C++ compile/run all
pass. The 121-token window measured 74.23 tok/s end-to-end with 93/108
accepted.
The next exact candidate sweep was also negative: MTP IQ1/Q81 reuse measured
48.99 tok/s versus 49.07 tok/s control, and split residual RMSNorm measured
48.15 tok/s; both retained hash `96b92d606dde5e28`. DFlash K=4 QKV + gate/up
+ SiLU fusion retained hash `15f17d2640c1adfc` but measured 55.61--58.19
tok/s, below the serialized/fused control range, so all remain opt-in.

2026-09-22 continuation: the one-row IQ2/IQ3/IQ4 kernels now declare their
output, weight, activation, and scale buffers non-aliasing.  This preserves
the existing dot and reduction order while giving HIPRTC safe load scheduling.
The final build passes the random 16K gate at 41.47 tok/s with prefix hash
`2cd51a0159d12ee0` and suffix hash `d70e119a6c94bc4c`; a random 64K run remains
exact at 35.80 tok/s (prefill 443.99 tok/s, prefix `90178de69a24a76e`, suffix
`7463f176c9b85ba3`).  The gain is small, so the grouped/mixed projection work
in item 1 remains open; no production tuning defaults changed.

The one-row IQ dispatch now snapshots the immutable MMQ-D4 quantizer and
IQ-shape-thread options once per process instead of re-reading the environment
for every projection. Default selection and arithmetic are unchanged; this
only removes repeated host-side option parsing from the decode launch path.

The opt-in mixed IQ1 gate/up projection now stages its packed 2K-entry IQ1
codebook once per block in LDS before evaluating the IQ1_S gate and IQ1_M up
rows. The per-row dot, affine correction, and warp reduction order are
unchanged; the production two-launch path remains disabled until resident
hash and 64K timing gates are available.

The ordinary one-token IQ2_XS path now has opt-in Q/K/V and dense gate/up
fusion probes (`LLM_QWEN35_IQ2_QKV_FUSED=1` and
`LLM_QWEN35_IQ2_GATEUP_FUSED=1`).  The fused kernel stages the 512-entry
IQ2_XS codebook once and joins only the launch; every row retains the
standalone Q8_1 dot, split-scale rounding, and warp reduction order.  The
independent IQ2_XS launches remain the default pending resident 64K hashes and
timing.

Mixed native IQ2/IQ3/IQ4 attention now has an opt-in
`LLM_QWEN35_IQ_MIXED_QKV_FUSED=1` kernel. It stages one Q8_1 activation and
the supported codebooks per block, then dispatches the exact format-specific
row arithmetic for Q, K, and V in one grid. Unsupported types, unequal input
widths, partial native initialization, or a disabled switch fall back to the
serialized path; production defaults remain unchanged pending resident mixed
format hashes and random-64K timing. Its LDS staging is format-aware, so
unused codebooks are not copied for a given Q/K/V type triple.
The same mixed kernel now accepts a cached diagnostic geometry switch,
`LLM_QWEN35_IQ_MIXED_THREADS=512`; unset or any other value retains the
256-thread default. This only changes block geometry and codebook-staging
amortization, so the resident A/B run can reject it on shapes where occupancy
falls without changing production behavior.

The same exact mixed-IQ kernel now has an opt-in
`LLM_QWEN35_IQ_MIXED_GATEUP_FUSED=1` dense-FFN path. When gate and up use
different supported IQ2/IQ3/IQ4 formats with matching dimensions, it joins
their Q8_1 activation staging and projection launch while passing an empty V
range, so the null V pointers are never touched. Equal-format pairs continue
to use their format-specific candidates; unsupported shapes and the default
configuration retain the serialized dispatcher until resident logits, hashes,
and random-64K timing validate the candidate.

The mixed kernel also covers the common same-format IQ2_S FFN gate/up pair
under `LLM_QWEN35_IQ2S_GATEUP_FUSED=1`. This fills the only frequent native IQ
FFN pair without a dedicated gate/up entry point while keeping the switch
separate from heterogeneous fusion and leaving serialized dispatch as the
default pending resident parity and timing.

Same-format IQ2_S attention Q/K/V projections have the corresponding
`LLM_QWEN35_IQ2S_QKV_FUSED=1` candidate. It joins the three native rows through
the same mixed kernel and Q8_1 activation tile; the regular three-launch path
remains the default until resident attention hashes and random-64K timing are
checked.

The DFlash2 selector's K=4 path now has an exact fixed-count Q4_K kernel
(`qwen35_matvec_q4k_q81_qkv_fixed4`) alongside the existing K=7 specialization.
It removes the runtime proposal-row predicate from the selector's inner dot
loop while preserving the generic vocabulary/rank output ranges and reduction
order. Selection remains opt-in through `LLM_QWEN35_DFLASH_SELECTOR_FUSED=1`
until resident K=4 hashes and draft timing validate it.

The verifier's batched DeltaNet path now has an opt-in exact F16 alpha/beta pair
kernel (`LLM_QWEN35_MTP_F16_PAIR_BATCH=1`). When both matrices have matching
rows and columns it flattens their output rows into one launch over the
verifier window, preserving llama.cpp's half2 FMA and XOR reduction order while
removing one projection launch per recurrent layer. BF16, mismatched, and
unsupported shapes retain the existing independent batch kernels; serving
defaults and target hashes are unchanged until resident timing confirms a gain.

The same verifier boundary now has an opt-in BF16 alpha/beta pair kernel
(`LLM_QWEN35_MTP_BF16_PAIR_BATCH=1`). It uses the llama.cpp BF16 block-size
heuristic and the same BF16 conversion, FMA, and XOR reduction sequence as the
standalone projections, while flattening matching alpha/beta rows into one
launch. F16, mismatched, and unsupported shapes keep their existing paths;
production defaults remain unchanged pending resident parity and timing.

The DFlash2 draft tail also has an opt-in
`LLM_QWEN35_DFLASH_SILU_Q81_FUSED=1` path for Q4_K down projections.  It keeps
the exact SiLU and Q8_1 contracts while writing the row-major batch tile that
the down projection consumes, so the separate SiLU and requantization launches
can be compared independently.  The serialized SiLU path remains the default.

The DFlash2 mask rows now use a safe default fast path. Every non-anchor
proposal row has the same mask token, so the runner embeds one anchor and one
mask row with the exact IQ1_M scalar kernel, then copies the mask row on-device
for the remaining rows. `LLM_QWEN35_DFLASH_EMBED_BROADCAST=0` restores the
older row-batched embedding for A/B checks. Matched 4K K=7 greedy runs kept
140 drafted/134 accepted and hash `44915ec1039a64c8`; draft time fell from
368--370 ms to 263--264 ms and warm decode rose from 78.66--79.88 to
82.65--84.12 tok/s. The full HTTP/stdio, cancellation, cache, concurrency,
sampled-repeatability, and multi-turn C++ harness passed with the broadcast
path. Target verification and selector indexing remain unchanged.

The K=4 greedy gate now measures 55.51--56.19 tok/s across two warm repeats,
with 124/124 accepted rows, `draft_ms` 274.5--283.8, `verify_ms`
2,188.5--2,197.9, and the same `44915ec1039a64c8` hash. A single-launch
device broadcast probe for the repeated mask rows was exact but neutral at the
same range, so it was reverted; the ordered copies keep the simpler stream
contract.

The DeltaNet warp-per-row batch probe was rerun three times under the seeded
sampled 4K gate. It averaged 38.99 tok/s versus 38.94 tok/s for the
reference-order path, with `c6bb94e73050164e` on every run. The difference is
within dispatch noise, so `LLM_SSM_BATCH_WARP=1` remains an explicit diagnostic
option and the reference-order production default is unchanged.

The resident Qwen3.8/DFlash2 server now has request-owned, bounded
multi-context state. `REQ3` carries a hashed cache namespace, and the HTTP
shim derives it from `prompt_cache_key`, conversation/session metadata, or
`X-Prompt-Cache-Key`. A FIFO gate gives queued requests fair access to the
single mutable GPU context. `request_id` and `X-Request-ID` provide targeted
`POST /v1/cancel`; cancelling a queued request cannot signal the active one.
IDs are reserved before SSE headers are committed, duplicate active IDs return
HTTP 409, and idle unscoped cancellation returns 404. Malformed content
lengths/message arrays and oversized stdio frames are rejected without
desynchronizing the resident protocol. Context trimming uses stable original
indices and retains complete user/assistant/tool turn groups, preventing equal
messages from being reordered or tool results from becoming orphaned.
The GPU work remains serialized because target recurrent scratch and DFlash
verification state are still single-context; no decode batching is enabled.

Successful prompt boundaries publish portable snapshots transactionally into
an entry- and byte-bounded LRU (`--context-cache-entries`, default 4;
`--context-cache-max-mib`, default 2048). Failed, cancelled, and incomplete
requests never publish, while earlier committed entries survive. Longest
exact token-prefix reuse is restricted to the same cache identity. The
snapshot includes target Q8/Q8 KV and scales, hybrid convolution/recurrent
state, prompt logits, and DFlash private KV/features. Dense NextN additionally
stores the prompt-boundary target hidden vector needed by its first proposal.
Snapshot position must exactly match the token key before publication.

Nonportable snapshots used by the older Qwen4 path remain eligible only while
their matching device context is resident. They are tagged separately and
discarded before a reset, identity switch, or portable restore, preserving
same-context prefix reuse without treating incomplete KV state as portable.
Exact restored prompts now touch and retain that committed entry instead of
recapturing the same 234--448 MiB state after each response.

This validation exposed two long-context snapshot bugs that immediate repeats
had hidden: batched prefill did not publish its final host position, and Q8/Q8
FP16 scale rows were copied with an FP32 size. The former captured only a
stale short KV prefix; the latter crossed the scale allocation beyond roughly
half context. Both are fixed. A forced A/B/A context switch now restores an
actual 6,535-token prompt with identical greedy output and `cached_tokens=6535`;
the complete host snapshot is 448.3 MiB. The old same-context-only large-prompt
cache observations did not prove portable target KV and are superseded by this
interleaved gate.

The reproducible GPU gate is `test_qwen35_dflash2_http.py`. It drives the
resident JSONL protocol directly, then tests the OpenAI-compatible HTTP shim.
Coverage includes greedy and seeded sampling, targeted and disconnect
cancellation, recovery, malformed cache metadata, concurrent distinct cache
identities, LRU eviction, forced context restoration, and a two-turn C++ task
whose generated programs are compiled and run. The CPU protocol/template/tool
suite now passes 31 tests.
The sampled random-64K target gate now passes 32 suffix tokens with prefix hash
`90178de69a24a76e`, suffix hash `34e2f6bc082bc49f`, and `Result: PASS` after a
445.67 tok/s prefix.

Dense Qwen3.8 NextN is now selectable in the resident HTTP/stdio harness with
`--qwen35-mtp SIDECAR --qwen35-mtp-draft N`. The Python server supplies the
required exact-window flag and validated Q8/Q8 profile. Greedy requests use
exact target windows, while sampled requests remain on ordinary target decode.
Draft KV resets at every request boundary; cancellation and errors discard an
open verifier transaction before another request runs.

The real-GPU resident gate passes direct stdio and HTTP traffic,
ordinary-target byte parity, repeated and A/B/A cache restoration,
targeted/disconnect cancellation, concurrent identities, and compiled
two-turn C++ output. Its retrieval case returns exactly `ZEPHYR-7319` from both
ordinary and Dense NextN serving. Prompt snapshots restore the target hidden
vector as well as logits, KV, and recurrent state, so the first post-restore
proposal retains its normal acceptance behavior.

The dense verifier now fuses Q8 attention split-combine with the per-head gate
for grouped verifier windows.  The ordinary decode path is unchanged.  The
full pinned reference validator preserves greedy and sampled token streams,
EOS, and output bytes; warm DFlash K=7 decode remains above 60 tok/s.

Verifier SSM checkpoint restore now passes explicit row-major strides for both
convolution and recurrent snapshots. DFlash window commit/rollback copies only
the accepted row into live state; the GPU HTTP gate and pinned llama.cpp
comparison retain exact greedy and seeded-sampled token/byte parity.

The batched SSM verifier also uses the existing fused Q/K normalize-and-expand
kernel, removing three intermediate launches per recurrent layer. Exact token
and byte hashes remain unchanged and warm prefill/decode throughput stays above
the established targets.

The attention verifier now pairs Q/K RMS normalization in one launch, retaining
the original per-head reduction order while removing one more preparation
launch per grouped attention layer. The HTTP and llama.cpp gates remain exact.

The Q8 target snapshot token bound remains 16,384 by default and can be changed
with `--qwen35-snapshot-max-tokens N`. Larger token bounds also require enough
`--context-cache-max-mib` host budget. Claims for 10K--60K portable snapshots
must be rerun with the forced interleaved-context gate; earlier tests only hit
the still-live device context and therefore did not validate host restoration.

Verifier SSM alpha softplus/scale and beta sigmoid preparation now share one
batched elementwise launch. The pinned greedy and sampled llama.cpp hashes and
warm throughput targets remain unchanged.

CLI: `--qwen35-dflash2 SIDECAR --qwen35-dflash2-draft 1..7`; it currently
requires benchmark mode, `--qwen35-batched-prefill`, `--qwen35-decode-graph`
and `--kv-cache q8q8`.  `validate_qwen38_reference.py` accepts `--dflash2`
and `--dflash2-draft` for the greedy/sampled C++ gate.  Current artifacts are
under `tmp/qwen38/dflash2-sampled-window-k4/`,
`tmp/qwen38/dflash2-sampled-window-k7/`, and
`tmp/qwen38/dflash2-retrieval-k7.*`.  Details and the reproduction command:
[QWEN38_DFLASH2.md](rdna4/llm/QWEN38_DFLASH2.md).

## Long-context Q8/Q8 prefill (2026-09-20)

The IQ2 runner sustains more than 400 tok/s while processing 65,536 random
tokens in 512-token chunks on RX 9070 XT / gfx1201 / ROCm 10.  The original
optimized ordinary run measured 413.25 tok/s; the current DFlash K=7 run
measures 443.57 tok/s. The previous native Q8/Q8 path took 460.37 seconds at
142.36 tok/s.  Long-context prefill therefore remains above its target after
adding sidecar feature capture and cache injection.

Long-context chunks use a gfx1201 WMMA attention kernel after position 4096.
Eight pairs of waves process 128 queries per block, split the 256-wide head
dimension across each pair, consume resident Q8 K/V directly, and use signed
INT8 QK plus F16 PV matrix instructions. The pinned exact vector kernel remains
the dispatch through 4K, so the validated short-context output path is unchanged.
The optimized kernel is separately compiled with fast math; exact decode and
short prefill retain the precise module.

A separate exact 65,536-token retrieval prompt sustained **412.34 tok/s** and
generated exactly the eleven bytes `ZEPHYR-7319`, retrieving the passphrase
from the prompt's first line after about 440 KB of filler. The output validator
passed. Against the pinned llama.cpp attention kernel, the WMMA path measures
0.000259 relative L2 and 0.01816 maximum absolute error at 4097 positions;
the existing exact path now passes 46,743,552 bitwise comparisons. The
optimized reduction order is approximate, so this is a semantic long-context
check rather than a general byte-parity claim. Artifacts:
`rdna4/llm/tmp/qwen38_gsq_iq2_64k_wmma_i8_local.log` and
`rdna4/llm/tmp/long-retrieval/`.

`bench_qwen38_gsq_decode_64k.sh` now gates depth prefill at 400 tok/s as well as
checking sustained decode, deterministic hashes, and real random-token cache
state. Override the prefill threshold with
`QWEN38_GSQ_64K_PREFILL_FLOOR_TPS` when evaluating other hardware.

## Sustained decode at 64K synthetic depth (2026-09-20)

`--bench-depth 65536` now matches llama-bench depth semantics: it processes a
deterministic random-token prefix through the complete model, then restores the
resulting recurrent state before each timed repeat while retaining the prefix
K/V rows on device. Seed one produces prefix hash `90178de69a24a76e`.

Before the long-prefill WMMA change, the 65,536-token IQ2 prefix took 460.37
seconds at 142.36 tok/s. Three original ordinary 512-token decode repeats
sustain 26.92/26.91/26.90 tok/s and share sequence hash
`b01a17fae16f806d`. Exact three-head GQA K/V reuse reached 32.56 tok/s for a
512-token suffix. Staging the profitable scalar IQ codebooks now reaches 33.31
tok/s after processing the same prefix at 443.44 tok/s; its hash is the
retained `051e7338c23a544e`.
The prior 27.94 tok/s result used zero cache values and is superseded.

The former fixed-eight-split DFlash K=7 path processed the same random prefix
at 443.57 tok/s and reported 49.74 tok/s with suffix hash
`2ddd068dca63669a`.  Exact per-query adaptive split selection supersedes that
result.  The selected shared-K/V graph now preserves the current exact suffix
`1c68ea2ff63ba5ab` and measures 39.89--39.93 tok/s after a 443.45--444.22
tok/s prefix.  It restores most of the generic exact verifier's 33.25 tok/s,
but the 40 tok/s random-depth target remains open by about 0.2%. Ordinary
decode also remains below 40 tok/s. Its exact
three-head K/V-reuse kernel cuts the 128-split attention operator from about
606 to about 348 microseconds per layer after the grouped scale-product
change, leaving about 5.6 ms/token in attention and
25 ms/token in the projection/state path. Dense NextN has now been rerun at
real random-token 64K depth: 408.64 tok/s prefill and 28.82 tok/s decode for a
256-token suffix, with prefix/suffix hashes `90178de69a24a76e` and
`f4b35758fb99e6db` and `Result: PASS`.
Full results and commands:
[QWEN38_64K_DECODE.md](rdna4/llm/QWEN38_64K_DECODE.md).

The pinned-kernel differential test passes 49,188,864 bitwise Q8/Q8 values,
including the adaptive short/long graph path, three-head K/V reuse,
shared-cache verifier cases, and matching split counts at 64K. Fresh
normal-context IQ2 and IQ3
greedy/sampled C++ outputs remain byte-identical to llama.cpp and pass fixed
cases plus 10,000 randomized cases. Artifacts:
`tmp/qwen38/depth64-final-iq2-v2/` and `depth64-final-iq3/`. Preserve unrelated
A64FX/common edits; no push.

## Earlier final decode validation (2026-09-20)

RX 9070 XT / gfx1201 / ROCm 10, 4096 prompt tokens, 512-token chunks,
context 8192, Q8 K and Q8 V. Warm repetitions reset all target/draft state;
timing excludes trace I/O. Sampled mode uses temperature 0.6 and seed 42.

| Model / path / sampling | Warm decode tok/s | Warm prefill tok/s |
|---|---:|---:|
| IQ2 / ordinary / greedy | 37.99–38.02 | 554.57–555.77 |
| IQ2 / ordinary / sampled | 37.20–37.21 | 550.91–551.58 |
| IQ2 / MTP K=3 / greedy | 35.80–35.86 | 550.83–551.39 |
| IQ2 / MTP K=3 / sampled | 35.25–35.27 | 550.61–550.94 |
| IQ3 / ordinary / greedy | 36.12 | 573.38–573.72 |
| IQ3 / ordinary / sampled | 35.19–35.47 | 572.75–573.14 |
| IQ3 / MTP K=3 / greedy | 33.76 | 572.32–573.07 |
| IQ3 / MTP K=3 / sampled | 34.99–35.01 | 571.43–571.59 |

**Ordinary 40 tok/s and dense NextN 60 tok/s remain unmet; DFlash2 clears
60 tok/s at 4K and reaches 39.89--39.93 tok/s at random-token 64K depth.** Ordinary
greedy decode improved from 33.8 to 38.0 tok/s on IQ2 and 32.1 to 36.1 on
IQ3 (about 12–13%). Later verifier scheduling supersedes the IQ2 MTP timing:
the 2026-09-21 pinned run reaches 47.87--47.93 tok/s greedy and
46.25--46.32 tok/s sampled with 609.24--610.77 tok/s warm prefill. It remains
opt-in because it is below 60 tok/s and degrades to 28.82 tok/s at 64K. The
next performance work belongs in multirow projection reuse and reducing
verification/checkpoint overhead.

All eight complete C++ responses match the pinned llama.cpp token IDs, EOS
and output bytes. Each passes C++17 compilation, ASan/UBSan, fixed edge cases
and 10,000 randomized cases. Both models and both sampling modes additionally
match every ordinary-target logit bitwise with MTP enabled. Artifacts:
`tmp/qwen38/decode-final-{iq2,iq3}{,-mtp}/`, including `result.json`, manifests,
and `target-*-parity.json`. Reference outputs/timings are reused from the
hash-checked pinned build; runner timings are fresh.

Exact native operator checks cover 2,948,352 activation values, 13,191,360
matrix outputs, 4,528,128 fused SSM preparation values and 46,743,552 attention
outputs. Parallel greedy selection passes 40 shape/pattern comparisons and
reduces its standalone time from 207.6 to 11.6 microseconds. Retained decode
changes include native IQ4_XS, computed IQ signs, selected packed IQ3_S loads,
shape-specific launch sizes, scoped activation reuse, fused SSM preparation,
and shared-grid multirow IQ/Q2_K projections.

The final retrieval gate repeats the exact 4096-token early-context fixture
on both models with ordinary decode and MTP draft widths 1 and 3. All six
outputs are exactly `ZEPHYR-7319`, including identical selected tokens and
EOS; MTP also preserves every ordinary-target logit bitwise. Artifacts and
commands: `tmp/qwen38/decode-final-retrieval/manifest.json` and adjacent
comparison files.

Dense NextN implementation and reproduction details:
[QWEN38_DENSE_MTP.md](rdna4/llm/QWEN38_DENSE_MTP.md). The benchmark/C API and
resident HTTP/stdio path support
`--qwen35-mtp SIDECAR --qwen35-mtp-draft 3 --qwen35-mtp-window`. Draft state
is independent, all emitted tokens come from exact target verification, and
sampler RNG advances only for consumed target logits. The draft starts at
generation rather than replaying the prompt. A three-step independent
llama.cpp NextN oracle checks the post-output-norm hidden-input contract
(matching top tokens; relative L2 0.01254/0.01218/0.01512), not bitwise
draft-logit parity.

Full-model llama.cpp logits still differ; BF16 prefill remains approximate.
The byte-parity evidence is fixture-specific. Keep MTP opt-in until it
outperforms ordinary decode. Preserve unrelated A64FX/common edits; no push.

## Native Q8/Q8 prefill and reference validation (2026-09-20)

On RX 9070 XT / gfx1201 / ROCm 10, native Q8/Q8 attention now exceeds the
500 tok/s warm prefill target with the existing opt-in BF16 projections:

| Model / sampling | Warm prefill tok/s | Warm decode tok/s |
|---|---:|---:|
| IQ2_XS greedy | 552.66–553.28 | 33.80–33.82 |
| IQ2_XS temperature 0.6, seed 42 | 551.56–551.81 | 33.40–33.42 |
| IQ3_XXS greedy | 575.37–575.62 | 32.07–32.08 |
| IQ3_XXS temperature 0.6, seed 42 | 574.77–575.13 | 31.78–31.79 |

These are full 4096-token, 512-chunk, context-8192 requests, Q8 K **and** V.
Repetitions reset KV and recurrent state and do not reuse prompt results.
Timing excludes trace I/O. Cold passes are 492–515 tok/s. Both sampling modes
match the pinned llama.cpp reference's complete token IDs, EOS and raw output
bytes for the C++ merge task. Every timing repetition reproduces its traced
response. All outputs pass C++17 compilation, ASan/UBSan, fixed edge cases and
10,000 randomized cases. This establishes fixture equivalence, not general
byte parity: full-model logits still differ and BF16 projections are approximate.

The final configuration also retrieves `ZEPHYR-7319` from the first line of
an exact 4096-token prompt on both models. All four runner/reference outputs
are exactly those eleven bytes, with identical selected tokens and EOS. This
checks that the early prefill chunks still affect generation. Artifacts:
`tmp/qwen38/final-native-retrieval-v4/` (reuses the earlier pinned reference).

New controls: `--sampling-profile llama`, `--qwen35-decode-graph`,
`--qwen35-native-q8-attn`, `--qwen35-native-q8-prefill`, and diagnostic
`--qwen35-reference-math`. `--qwen35-native-mmvq` enables native Q2_K,
IQ2_XXS/XS/S and IQ3_XXS/S decode; `--qwen35-native-q2k` isolates Q2_K. The native
matrix-vector kernels and quantizer pass 2,948,352 activation and 13,191,360
output comparisons against actual reference kernels. Normal generation stops at EOS; use the explicit
`--bench-ignore-eos` only for synthetic timing. Native attention eliminates
the repeated Q8-to-F16 expansion and follows the pinned reference's Q8_1
query quantization, half2 arithmetic and split reductions. Its standalone
test passes 46,743,552 bitwise comparisons against actual llama.cpp HIP kernels.
The independent sampler passes 13,801,002 exact comparisons against libllama.
Graph replay before changing attention matches uncaptured logits bitwise.

Reproduction and remaining work: [QWEN38_REFERENCE_VALIDATION.md](rdna4/llm/QWEN38_REFERENCE_VALIDATION.md).
Reference source is pinned to `1859b520910af6f682256fd7299797774111a27a`, exported
and checked without changing the external checkout. Full manifests and traces:
`tmp/qwen38/final-iq2-native-mmvq-v4/` and `final-iq3-native-mmvq-v4/`.
IQ2 reuses the audited reference artifacts from `final-iq2-native-q2k/`,
and IQ3 from `final-iq3-native-mmvq-v2/`. The harness verifies model, prompt,
reference hashes and exact sampler commands. The Q2_K scheduling change
preserves all logged logits bitwise against v3 across both sampling modes
and models (648 selections). Native IQ2_XXS/XS raises IQ3's greedy full-logit
relative L2 from 0.03119 (v2) to 0.03641 (v3/v4), despite exact individual
operator results and identical tested output. This coupled gap remains open.
Attention-only baselines remain in `final-iq2-native-prefill/` and
`final-iq3-native-prefill/` (25.6–26.0 tok/s decode).
The validation script includes the exact 4096-token C++ prompt by default.

Ordinary decode at 40 tok/s, dense NextN/MTP at 60 tok/s, and general byte
parity remain open. DFlash2 separately meets its 60 tok/s short-context goal
and reaches 39.89--39.93 tok/s at random-token 64K depth, just below the strict
40 tok/s threshold.
The existing Qwen4 MoE/HC MTP implementation is incompatible with the dense
27B sidecar. The pinned graph passes the **post-output-norm** hidden vector
to NextN. Do not use the earlier pre-norm assumption.

## Latest validation: matched C++ coding output (2026-09-19)

An exact 4,096-token prompt asked for a C++17 `merge_intervals` function and
supplied the required standard headers. Greedy generation used Q8 K/Q8 V,
512-token chunks, the opt-in BF16 prefill path, and a 256-token ceiling.
Our runner and llama.cpp produced the same 560-byte implementation for both
IQ2_XS and IQ3_XXS (SHA-256 `4a0cb461966fae9a...`). The raw responses are
strict UTF-8 and contain no tokenizer markers, replacement characters,
Markdown fences, or exposed ChatML controls.

All four responses compile under C++17 with warnings as errors, ASan, UBSan,
and libstdc++ assertions. Each passes fixed empty/overlap/nesting/duplicate/
adjacency/INT_MIN/INT_MAX cases and 10,000 randomized comparisons. The new
`rdna4/llm/test_cpp_merge_output.py` reproduces these checks from runner and
llama.cpp generation logs. First-pass prefill was 413.64 tok/s (ours) versus
256.93 (llama.cpp) for IQ2, and 421.97 versus 352.63 for IQ3. Final prompt
argmax matched at token 1771. Full-logit relative L2 remained nonzero:
0.176949976 for IQ2 and 0.072349927 for IQ3, so the result establishes this
coding task's output equivalence rather than full numerical parity.

## Latest continuation: real Q8/Q8 support (2026-09-19)

User clarified Q8_0 for **both K and V**, and selected the opt-in BF16
projection fast path. Implemented `--kv-cache q8q8` and appended public enum
`HIP_LLM_KV_Q8_0_Q8_0`. Existing format defaults remain IQ2 Q8/Q4 and IQ3 F32.

New cache writer matches llama.cpp's HIP quantizer on identical inputs.
Protected reciprocal and quotient refinement is necessary under fast-math;
ordinary division changes integer decisions at half-integer boundaries.
The actual runner kernels pass 14,741,204 code, scale, unpack, and boundary
checks with zero mismatches. Tests: `rdna4/llm/test_kv_q8q8.py` and `.cu`.

Fixed quantized K allocation (previously incorrectly F32-sized), included
context-dependent F16 packing scratch in memory estimates, and restricted
Q4-specific direct kernels to Q8/Q4. Q8/Q8 uses the existing packed-F16
attention path; native llama.cpp Q8/Q8 decode arithmetic is still distinct.

A 40-token BF16 smoke test exposed a separate hipBLASLt workspace crash:
direct `hipMalloc` calls bound to rocew's function-pointer variable. The
bridge now resolves allocation/free/error functions from libamdhip64.

Completed Q8/Q8 4096-token / 512-chunk / capacity-8192 fast runs:

| Model | First pass | Warm 1 | Warm 2 | Peak MiB | Logit rel L2 / max vs Q8/Q8 llama |
|---|---:|---:|---:|---:|---:|
| IQ2_XS | 409.69 | 465.07 | 464.11 | 9782 | 0.088450366 / 1.062415123 |
| IQ3_XXS | 414.20 | 478.35 | 477.66 | 13446 | 0.068600276 / 0.862522125 |

All six generated clamp functions pass syntax and 196 UBSan cases. Each
model repeats its generation hash exactly; reference/runner argmax is 71093.
This is experimental BF16 prefill, not whole-model numerical parity.

Output-integrity follow-up: benchmark generation now byte-decodes every BPE
piece before printing it, so GPT-2 display markers such as `Ġ` and `Ċ` no
longer leak into user-visible text. It also hides ChatML/EOS control pieces
after the first stop token, including duplicate control strings found in
some converted vocabularies. Decode benchmarking still runs the requested
token count, so throughput and sequence hashes retain their old meaning.

Fresh 4K boundary runs cover IQ2 chunks 256/511/512/513 and IQ3 chunks
511/512/513. All seven runs have finite full-vocabulary logits, first token
and argmax 71093, strict UTF-8 output, and a functional clamp implementation
that passes all 196 UBSan cases. BF16 batch-shape arithmetic changes the final
logits modestly (maximum relative L2 0.032505 for IQ2 and 0.018762 for IQ3),
but there is no chunk-boundary corruption.

A stronger exact-4096-token test places `ZEPHYR-7319` in the first prompt
line, fills the intervening context, and asks for the passphrase at the end.
The 512-token Q8/Q8 fast path returns exactly `ZEPHYR-7319` for both models,
at 410.03 tok/s for IQ2 and 423.10 tok/s for IQ3. This verifies that all
eight prefill chunks contribute to the response rather than merely checking
a task located at the prompt tail. Artifacts are under
`tmp/qwen38/prefill-output-fidelity/`; `rdna4/llm/test_prefill_output.py`
checks strict UTF-8, raw tokenizer/control-marker absence, benchmark success,
first-token consistency, requested response text, and generated C behavior.

Scalar Q8/Q8 4K baselines: IQ2 28.18 tok/s, rel L2 0.103469486;
IQ3 23.34 tok/s, rel L2 0.064371208. BF16 improves IQ2's error here, but
raises IQ3 relative L2 by ~6.6% while reducing its maximum error. Both
legacy-format 40-token logit files remain byte-identical to saved baselines.

Artifacts and exact command/environment manifests:
`tmp/qwen38/prefill-q8q8-4k-512/`. `run.py` runs the sequential validation
matrix (completed successful jobs are skipped); `summarize.py` calculates
errors and verifies generated C. All 16 jobs passed: eight finite full-logit
comparisons, 16 generated-C tests (196 UBSan cases each), both default
regressions, and both warm throughput/repeatability gates. The documented
`make -C rdna4/llm kv-q8q8-test HIPCC=/opt/rocm/core-10.0/bin/hipcc` also
passes on final source. Build, profile, CLI, shell syntax, and whitespace
checks pass; existing compiler warnings remain.

Fresh synthetic llama-bench Q8/Q8: IQ2 293.060813 tok/s (293.117 / 293.005),
IQ3 422.700387 (422.799 / 422.601). Matched natural-prompt reference runs,
without warmup: IQ2 250.196 and IQ3 344.217 tok/s. Keep timing methods distinct.
Short (40-token) Q8/Q8 scalar -> BF16 relative L2: IQ2 0.060560054 ->
0.050156043; IQ3 0.041533788 -> 0.061390913. These are limited fixture
checks, not a broad quality equivalence claim. Production defaults remain
unchanged. Full details and the reproduction command are in the final
Q8/Q8 section of `rdna4/llm/QWEN38_STATUS.md`.

No commits or pushes. Preserve the existing unrelated dirty work.

## Latest continuation: 4K / 512 prefill throughput

User requested >=150 tok/s prefill, matching llama.cpp's 4K/512 shape.
The target is exceeded with new opt-in `--qwen35-prefill-bf16`:

| Model, Q8 K / Q4 V | First pass | Second pass | Decode |
|---|---:|---:|---:|
| IQ2_XS default | 28.17 tok/s | — | 19.35 tok/s |
| IQ2_XS BF16 prefill | 409.85 tok/s | 462.89 tok/s | 19.50 / 19.49 tok/s |
| IQ3_XXS BF16 prefill | 421.33 tok/s | 476.50 tok/s | 19.71 / 19.74 tok/s |

All use the meaningful, exactly 4096-token fixture
`tmp/qwen38/prefill-4k-512/coding-4096.txt`, 512-token chunks, BMAX=512,
context capacity 8192, and 80-token --coding generation. Same generation
hash on both fast passes; extracted C passes syntax and 196 UBSan cases.

Fresh llama-bench IQ2, 4K/512: Q8/Q8 KV 293.87 tok/s; Q8/Q4 KV 293.49 tok/s.
The runner's measured configuration is Q8/Q4, NOT newly implemented Q8/Q8.
The matched coding-prompt llama helper measured ~254 IQ2 / 349.80 IQ3 tok/s
without warmup. Do not mix that timing with warmed llama-bench samples.

Matched-cache full-logit relative L2 / max error, argmax 71093 on all paths:
- IQ2 default: 0.170325388 / 2.245854616.
- IQ2 BF16 prefill: 0.171977136 / 2.299321413.
- IQ3 BF16 prefill: 0.132160331 / 1.575695515.

This is an experimental BF16 arithmetic choice, not exact MMQ/MMVQ parity.
Production defaults and decode dispatch remain unchanged. The argument
enables batched SSM as well as attention/FFN GEMMs and bypasses the SSM
per-row output projection. No QWEN38_GSQ_PERF setting is needed.
The main bottleneck was all 48 SSM blocks falling back to per-token compute.

API: `hip_llm_load_options.qwen35_prefill_bf16` (appended field, opt-in).
Diagnostic: `LLM_QWEN35_PROFILE_PREFILL=1` logs synchronized per-layer timings;
disable profiling for throughput measurements. Source also updates CLI help.

Re-run by adding `--qwen35-prefill-bf16` to the launcher and selecting
`--prompt-file tmp/qwen38/prefill-4k-512/coding-4096.txt --ubatch 512
--decode 80 --coding --bench-repeat 2 -s 8192`, with
`LLM_BMAX=512 LLM_BENCH_STREAM_CHUNK=512 QWEN38_GSQ_KV_CACHE=q8q4`.
Use the existing absolute QWEN38_RUNNER_BIN / TMPDIR convention below.

Artifacts and full commands: `tmp/qwen38/prefill-4k-512/` and the final section
of `rdna4/llm/QWEN38_STATUS.md`. `summarize.py` recomputes `summary.json` and
validates all extracted C. Build, profile tests, syntax/help and whitespace
checks pass. No commits or pushes.

Further work should expand quality coverage of this fast mode and reduce
its remaining numerical gap. Keep the prior exact IQ3_S diagnostic below
separate: it is not needed for the prefill speedup.

## Objective and constraints

Continue closing the IQ2/IQ3 numerical gap on RX 9070 XT (RDNA4), then validate
long-context prefill/decode performance. Whole-model parity is NOT achieved.

- Work in /mnt/nvme02/work/gemm/main; use repo-local tmp/, never /tmp.
- Preserve unrelated dirty changes. Use apply_patch for source edits.
- AMD GPU commands require escalated execution.
- Do not push. Commit only when explicitly requested.
- Read rdna4/llm/QWEN38_STATUS.md, especially its final two sections.
- Production defaults remain at the validated hip-prod configuration.

## Current files and new diagnostic

Primary files:
- rdna4/llm/hip_llm_runner.c
- rdna4/llm/run_qwen38_gsq_rocm.sh
- rdna4/llm/QWEN38_STATUS.md
- rdna4/llm/test_iq3s_mmvq_replay.py
- rdna4/llm/test_iq3s_mmvq_replay.cu

The earlier gate/up TODO was stale. GGUF metadata confirms layer 0:
- attn_qkv.weight: IQ3_S, [5120, 10240]
- ffn_gate.weight: IQ1_S, [5120, 17408]
- ffn_up.weight: IQ1_M, [5120, 17408]

The earlier cache-layout fix and Q8_1/SSM/FFN improvements are already present.
Current production IQ2 F32-KV error is 0.046010129, not the older 0.1056/0.0910.

Added opt-in LLM_IQ3S_MMVQ_REF=1. It substitutes
matvec_iq3_s_q81_mmvq_batch for existing IQ3_S Q8_1 routes only.
Unset/zero preserves production. Do not promote it: full IQ2 logits regress.

## Exact local result

The replay includes the actual local llama.cpp vec_dot_iq3_s_q8_1 and extracts
the actual runner kernel from hip_llm_runner.c. Both consume the SAME
GPU-quantized captured input; this isolates the matvec, not the quantizer.

Confirmed IQ3_S schedule: QI=16, VDR=2, nwarps=1, blocks_per_iter=4,
kbx=tid/8, kqs=2*(tid%8). This is identical to the runner's lane-per-Q8-block,
stride-32 schedule. The old QI=8/four-thread description was wrong and is now
corrected in QWEN38_STATUS.md. Do not rewrite this mapping.

The missing contract:
1. Scale the integer dot by the odd scale code before converting to float.
2. Round weight_d * q8_d separately.
3. FMA the scaled integer into the accumulator; XOR warp reduction.

Ordinary expressions (also a __fmul_rn/__fadd_rn attempt) were not sufficient
under fast-math. Explicit v_mul_f32 and v_fma_f32 produce exact replay.
The local llama HIP mmvq build uses -O3 WITHOUT fast-math; the harness compiles
it separately from the runner's -O3 -ffast-math translation unit.

All 10,240 rows, five inputs (captured, three deterministic random, zero):
- Production: 27753/51200 exact, rel 5.76739e-8, max 7.62939e-6.
- Protected mul plus separate add: 26320/51200 exact.
- Protected mul plus FMA: 51200/51200 exact, rel=0, max=0.
- Individual dot contributions: 8192000/8192000 exact.
- Partial row-block test: 13 rows x 5 inputs, 65/65 exact.

## Full-model measurements

Models under /mnt/nvme02/models/qwen38/27b/gsq/:
- Qwen3.8-27B-GSQ-RCO-IQ2_XS.gguf
- Qwen3.8-27B-GSQ-RCO-IQ3_XXS.gguf

Canonical prompt: tmp/qwen38/coding-prompt.txt, exactly 40 tokens.
Preserve its two trailing newlines (final token 271, not 198).

Fresh llama.cpp sequential references from checkout 1859b5209 with local
changes, build-codex-hetero-dev2. Match KV format when comparing.

| Mode | Relative L2 | Max absolute | Argmax |
|---|---:|---:|---:|
| IQ2 production, F32 KV | 0.046010129 | 0.540599227 | 71093 |
| IQ2 diagnostic, F32 KV | 0.073783392 | 0.692957401 | 71093 |
| IQ2 production, Q8/Q4 KV | 0.129714052 | 1.625519276 | 71093 |
| IQ2 diagnostic, Q8/Q4 KV | 0.155747498 | 1.728286982 | 71093 |
| IQ3 production, F32 KV | 0.039762184 | 0.428576946 | 71093 |

All reference argmaxes are also 71093. The local exact fix is not a monotone
end-to-end improvement; other stage differences still perturb Q8 rounding.

Production IQ2 Q8/Q4 logits remain byte-identical to
tmp/qwen38/current-audit-20260919/hip-prod-q8q4.bin.
Production IQ3 logits are byte-identical before/after this session's changes.

40-token prompt, 80-token --coding generation, warm model, -s 256:
- IQ2 production Q8/Q4: prefill 28.86 tok/s; decode 24.78 tok/s.
- IQ2 diagnostic F32: prefill 28.83 tok/s; decode 25.55 tok/s.
- IQ3 production final: prefill 27.77 tok/s; decode 25.40 tok/s.

All produced the complete two-comparison clamp function. Both llama references
and all generated HIP clamp sources passed C11 syntax checking and 196
functional cases, including INT_MIN/INT_MAX, under UBSan.
These short runs do not establish long-context performance.

## Artifacts and commands

Everything from this continuation:
tmp/qwen38/iq3s-replay-20260919/

Important files:
- summary.json / summary.txt; summarize.py recomputes errors and C tests.
- iq2-before.bin / iq2-ref.bin: F32-KV production / diagnostic.
- iq2-default.bin / iq2-ref-q8q4.bin: Q8/Q4 production / diagnostic.
- iq3-before.bin / iq3-final.bin: unchanged IQ3 production.
- iq3-ref.bin: IQ3 diagnostic A/B, byte-identical to production.
- llama-iq2.bin / llama-iq2-q8q4.bin / llama-iq3.bin: fresh references.
- full/result.txt and multi/result.txt: exact isolated GPU replay.
- *-clamp.c, *-clamp-test.c: syntax and functional checks.
- build-final.log: host runner build; existing unrelated warnings remain.

Build:
    make -C rdna4/llm -j2

Replay preparation:
    export TMPDIR="$PWD/tmp"
    export PYTHONPATH=/mnt/nvme02/work/llama.cpp/gguf-py
    python3 rdna4/llm/test_iq3s_mmvq_replay.py \
      /mnt/nvme02/models/qwen38/27b/gsq/Qwen3.8-27B-GSQ-RCO-IQ2_XS.gguf \
      tmp/qwen38/llama-iq2-q8q4-trace2/llama-attn-norm-00.bin \
      --out tmp/qwen38/iq3s-replay-20260919/full --random-vectors 3

Run with AMD access:
    env LD_LIBRARY_PATH=/opt/rocm/core-10.0/lib \
      tmp/qwen38/iq3s-replay-20260919/full/replay \
      tmp/qwen38/iq3s-replay-20260919/full 10240 5120 5

Full runner pattern (absolute /mnt path avoids physical /home alias issues):
    env TMPDIR=/mnt/nvme02/work/gemm/main/tmp \
      QWEN38_RUNNER_BIN=/mnt/nvme02/work/gemm/main/rdna4/llm/test_hip_llm \
      LLM_LOGITS_PATH=/mnt/nvme02/work/gemm/main/tmp/qwen38/NEW.bin \
      bash /mnt/nvme02/work/gemm/main/rdna4/llm/run_qwen38_gsq_rocm.sh \
      --gpu-only-bench --bench \
      --prompt-file /mnt/nvme02/work/gemm/main/tmp/qwen38/coding-prompt.txt \
      --decode 0 -s 256

Set QWEN38_GSQ_KV_CACHE=f32 for F32 comparison; omit for IQ2 production Q8/Q4.
Set QWEN38_MODEL to the IQ3 model for its profile (F32 KV default).
Set LLM_IQ3S_MMVQ_REF=1 only for the diagnostic.
For generation add LLM_GEN_TEXT=1 and --decode 80 --coding.

## Next work

1. Keep production defaults unchanged. The isolated IQ3_S gap is closed, but
   the final gap is not; do not re-run abandoned plain-expression variants.
2. Extend captured-input GPU replay to the next coupled layer-0 stages:
   wide RMSNorm, SSM Q/K normalization and gated RMSNorm, conv/SiLU, and the
   IQ4_XS output projection. Verify each stage independently before combining.
3. Earlier norm ports were reverted because final logits regressed. Revisit
   them with stage equality and the new IQ3_S diagnostic together; do not
   infer local correctness solely from final logits.
4. Preserve matching prompt tokens and KV precision for every A/B.
5. Revalidate IQ2/IQ3 generated C and report numerical and throughput results
   separately. Long-context optimization remains outstanding.
