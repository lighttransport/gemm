# Qwen3.8 27B HIP runner vs llama.cpp — resume state

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
