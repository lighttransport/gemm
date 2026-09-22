# Qwen3.8 DFlash2 on RDNA4

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

## Restored GPU validation (2026-09-22)

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

The RX 9070 XT is accessible outside the execution sandbox. The rebuilt
ordinary Q8/Q8 runner passed the random-token 64K gate at 446.81 tok/s
prefill and 35.42/35.40/32.21 tok/s decode, with identical token hashes.

GPU A/B with `LLM_QWEN35_IQ_MIXED_QKV_FUSED=1` and
`LLM_QWEN35_IQ_MIXED_GATEUP_FUSED=1` (default 256 threads) retained depth
hash `90178de69a24a76e` and all three 512-token suffix hashes
`36a22439594d4e43`. Prefill was 445.07 tok/s; decode was
34.78/34.74/34.73 tok/s, below the control's first two 35.42/35.40 runs
(control third run 32.21). This does not justify promotion; retain opt-in.
Log: `rdna4/llm/tmp/gpu-access-64k-mixed256.log`.

## IQ4_XS sidecar projection staging reuse (2026-09-22)

The multi-row IQ4_XS sidecar dispatcher now reuses its exact Q8_1 activation
tile across Q/K/V projections, matching the existing Q4_K cache. The cache
records the source shape and weight type, so a mixed Q4_K/IQ4_XS sequence
cannot consume a tile produced for the other contract. This removes repeated
quantization launches when an IQ4_XS sidecar is selected; Q4_K behavior and
serving defaults remain unchanged pending resident K=4/K=7 timing and hashes.

## IQ1-only batch staging cleanup (2026-09-22)

The IQ1 Q8_1 batch quantizer no longer launches a Q8x2 producer that is
immediately overwritten. MTP gate/up reuse and standalone IQ1 batch
projections now write only the Q8_1 payload, FP16 scale, and original-input
block sum they consume, then invalidate the shared Q8x2 cache metadata. The
mixed SSM dispatcher uses a separate preserving helper when another projection
still needs the two-term Q8x2 bytes. This removes redundant staging work while
keeping the mixed-format arithmetic and production defaults unchanged; resident
quality/hash and 64K timing validation remain pending.

## Draft embedding launch cleanup (2026-09-22)

Each DFlash2 proposal now decodes one anchor row and one fixed mask-token row
with the exact IQ1_M scalar kernel, then copies that device row for every
remaining mask candidate. The selector overwrites its scratch only after the
embedding and copies have completed on the same stream; setting
`LLM_QWEN35_DFLASH_EMBED_BROADCAST=0` restores the older exact
`embed_iq1_m_batch` A/B path. Target verification, selector state, and captured
graph arguments are unchanged.

The pinned 4K C++ merge benchmark improved greedy K=7 from the previous
79.88 tok/s run to 83.69 tok/s warm (140 drafted, 134 accepted), with stable
hash `44915ec1039a64c8`; draft time fell to about 264 ms for 140 proposals.
Seeded sampled K=7 measured 70.47 tok/s warm (140 drafted, 115 accepted) with
stable hash `630b7cbc72230e0d` and about 250 ms draft time. The full HTTP/stdio
cache, cancellation, concurrency, and multi-turn C++ quality harness passes.

Every non-anchor DFlash2 proposal row is the same mask token. The default
embedding path now decodes one anchor row and one mask row with the exact IQ1_M
scalar kernel, then performs device-to-device copies for the remaining mask
rows. Set `LLM_QWEN35_DFLASH_EMBED_BROADCAST=0` to restore the previous
row-batched A/B path. Matched 4K K=7 greedy runs retained 140 drafted/134
accepted and hash `44915ec1039a64c8`; draft time fell from 368--370 ms to
263--264 ms, and warm decode improved from 78.66--79.88 to 82.65--84.12
tok/s. The complete HTTP/stdio and multi-turn C++ quality suite, including
seeded sampling, passed with the default fast path.

The same default path at K=4 retained 124/124 accepted rows and the greedy
hash, measuring 55.51--56.19 tok/s across two warm repeats (`draft_ms`
274.5--283.8, `verify_ms` 2,188.5--2,197.9). A single-launch device broadcast
probe for the repeated mask rows was byte-stable but neutral at 55.51--56.19
tok/s, so the existing ordered device copies remain the simpler production
path.

## Five-row Q4_K projection specialization (2026-09-22)

K=4 has exactly five sidecar proposal rows. Its Q4_K/Q8_1 projection now uses
a compile-time five-row kernel that preserves the multi8 dot-product and
reduction order while dropping three unused accumulators and row-count tests.
Two warm repeats kept 124/124 accepted rows and hash `44915ec1039a64c8`;
`draft_ms` fell to 224.1--226.8 and decode rose to 56.75--57.06 tok/s,
compared with 274.5--283.8 ms and 55.51--56.19 tok/s for the previous
multi8 path. The K=7 path continues to use multi8: a sanity gate retained
140/134, hash `44915ec1039a64c8`, and 74.79 tok/s.

The K=7 window has a fixed eight-row shape (one anchor plus seven mask rows).
An opt-in `LLM_QWEN35_DFLASH_Q4K_FIXED8=1` candidate now removes the
multi8 kernel's runtime row-count predicates while retaining its Q4_K affine
correction, packed dot order, and warp reduction. The reference multi8 path
remains the default until a resident K=7 hash and draft-time A/B run show a
repeatable gain.

The same fixed-row specialization is available to the opt-in fused Q/K/V and
gate/up projection path as `LLM_QWEN35_DFLASH_QKV_FIXED8=1`. It only changes
the activation-row predicate in the existing fused kernel; output row ranges,
weight decoding, and reduction order are unchanged.

## Fused sidecar attention merge (2026-09-22)

The DFlash2 verifier now uses a fused attention kernel for schedules of up to
twelve splits.  Each warp computes one split and four adjacent rows, writes
its summary to LDS, and warp zero merges summaries in the same increasing
split order as the former attention-plus-combine pair.  This removes global
partial traffic and one launch while leaving target verification and captured
target graph ABIs unchanged.  Set `LLM_QWEN35_DFLASH_FUSED_ATTN=0` to run the
two-kernel control path; schedules above twelve splits always use that
fallback.

On the 4K IQ2_XS/Q8-Q8 gate, fused versus control measured 82.76/82.68 tok/s
for greedy K=7 with hash `44915ec1039a64c8`, 70.89/70.85 tok/s for seeded
sampled K=7 with hash `72a11474a3a222b5`, and 55.90/55.81 tok/s for sampled
K=4 with the same hash.  The HTTP/stdio cache, cancellation, concurrency, and
C++ quality checks all pass.  This is a small but repeatable sidecar win; the
ordinary target's random-token 64K decode remains 35.76 tok/s, so the
long-context 40 tok/s target still requires mixed projection work.

## Fixed-head attention candidate

The sidecar's head dimension is fixed at 128, so an opt-in
`qwen35_dflash2_attention_fused_128` kernel removes dynamic dimension bounds
from the fused cache walk without changing the dot, online-softmax, or
split-merge order. Set `LLM_QWEN35_DFLASH_FUSED_128=1` for an A/B run; the
validated generic fused kernel remains the default until a resident-device
quality and throughput gate is available. HIPRTC syntax and host/profile
checks pass; this environment currently has no `/dev/kfd`, so runtime numbers
are intentionally pending.

## Per-stream cache-injection workspace

When `LLM_QWEN35_DFLASH_OVERLAP_INJECT=1` is selected, accepted feature rows
are first copied on the target stream and then injected through bounded,
request-independent scratch owned by the injection stream. This removes the
previous activation-buffer alias with the next target proposal. The target
and injection streams are ordered by `target_ready`, and `inject_done` still
guards sidecar KV reuse. The overlap path requires the BF16 injection plans;
allocation failure falls back to serialized injection and leaves production
defaults unchanged. Partial workspace cleanup nulls each freed pointer, so a
retry or teardown after allocation failure cannot double-free sidecar buffers.
Initialization now records ownership of handles created during the current
attempt, so a retry after partial stream/event setup does not destroy valid
pre-existing resources. Final teardown uses the same pointer-nulling workspace
helper; this changes lifecycle safety only and leaves overlap opt-in.

## Native IQ3 Q/K/V projection candidate

`LLM_QWEN35_IQ3_QKV_FUSED=1` enables a diagnostic ordinary-decode kernel when
Q, K, and V are all IQ3_XXS with a common input width. It uses the native Q8_1
activation contract and one IQ3 codebook staging load for the combined output
row grid, but preserves the standalone kernel's arithmetic for each row.
Defaults remain unchanged until a resident-device A/B validates exact token and
logit hashes together with a sustained 64K decode improvement.

All native Q2/IQ projection and opt-in Q/K/V or gate/up fusion entry points now
share an explicit Q8_1 readiness guard.  A partial diagnostic load or failed
scratch allocation therefore falls back to the reference projection dispatcher
instead of launching a native kernel with an incomplete activation tile.  This
is a fail-closed robustness change; successful launches and production defaults
are unchanged.

The dense FFN path also exposes `LLM_QWEN35_IQ3_GATEUP_FUSED=1` when gate and
up are matching IQ3_XXS matrices. It reuses the Q/K/V kernel with the V range
disabled, so both one-token projections share codebook staging and one launch
while retaining the standalone row arithmetic. This remains opt-in pending
resident hash, quality, and throughput checks.

Matching IQ2_XXS dense gate/up matrices can similarly opt into
`LLM_QWEN35_IQ2XXS_GATEUP_FUSED=1`. The existing IQ2_XXS fused kernel covers
the two output ranges after one native Q8_1 staging launch, with V disabled and
the standalone integer dot, scale, and reduction order preserved. Production
dispatch remains unchanged until resident hashes and sustained 64K timing are
available.

Matching IQ3_S dense gate/up matrices can opt into
`LLM_QWEN35_IQ3S_GATEUP_FUSED=1` as well. It reuses the IQ3_S fused row-range
kernel and native Q8_1 staging while preserving the standalone code decode and
reduction order. This remains diagnostic pending resident parity and 64K timing.

The verifier combine tail exposes
`LLM_QWEN35_VERIFY_COMBINE_GROUPED=3` for a sixteen-row block. It covers the
full verifier batch in one metadata tile and retains the existing row-wise
split arithmetic, including the gated form. Shorter windows select the four- or eight-row kernel. The serialized combine
remains the production choice until resident parity and timing are measured.
The grouped launch now supplies the gate pointer required by gated kernels;
previously it passed the ungated argument layout.
`python3 rdna4/llm/test_verifier_launch.py` covers 36 host dispatch cases and
rejects the original missing-gate layout; GPU parity is checked separately.

Mixed-IQ attention layers can also opt into
`LLM_QWEN35_IQ3S_QKV_FUSED=1` when Q, K, and V are all IQ3_S with a common
input width. The kernel keeps IQ3_S code decoding, scale correction, and warp
reduction order exact while sharing the Q8_1 activation launch. This remains a
diagnostic candidate until resident gfx1201 token/logit hashes and sustained
64K timing show a gain; standalone IQ3_S launches remain the default.

All-IQ4_XS attention layers can opt into
`LLM_QWEN35_IQ4XS_QKV_FUSED=1`. The candidate keeps the eight virtual warp
passes, nibble decode, affine correction, and reduction order of standalone
IQ4_XS while sharing native Q8_1 activation staging. It remains diagnostic
until resident logits/tokens and sustained 64K timing confirm a gain.

The IQ2_XS target's usual IQ1_S gate/IQ1_M up pair has a corresponding
`LLM_QWEN35_IQ1_GATEUP_FUSED=1` candidate. It shares the exact Q8_1 activation
and dispatches both differing IQ1 layouts in one warp-per-row kernel. The
candidate requires both role-level IQ1 Q8_1 switches, leaves MMQ overrides on
the standalone path, and remains opt-in pending resident parity and 64K
throughput measurements.

Ordinary IQ2_XS attention and dense FFN projections have matching opt-in launch
fusions (`LLM_QWEN35_IQ2_QKV_FUSED=1` and
`LLM_QWEN35_IQ2_GATEUP_FUSED=1`). They stage the native IQ2_XS codebook once
while preserving each row's Q8_1 arithmetic and reduction order; production
dispatch remains unchanged until resident hashes and timing are available.

`LLM_QWEN35_DFLASH_SILU_Q81_FUSED=1` is an additional opt-in tail probe for
Q4_K down projections. It fuses the exact SiLU multiply with row-major Q8_1
staging and lets the down projection reuse those bytes, while retaining the
F32 gate result and the serialized path as the default.

## DFlash2 long-window split retune (2026-09-22)

The sidecar attention launch now uses twelve partitions once its 2,048-token
window reaches 1,024 tokens.  The 1/4 split schedule for shorter windows is
unchanged, and `LLM_QWEN35_DFLASH_ATTN_SPLITS` remains an opt-in A/B override.
On a fixed 256-token K=7 gate, twelve partitions measured 56.57 tok/s versus
56.21 with eight; an EOS-limited K=7 gate measured 82.57 versus 82.14 tok/s.
K=4 measured 47.50 tok/s on the fixed gate and 59.69 tok/s on the normal gate.
Greedy K=7 retained sequence hash `44915ec1039a64c8`; seeded sampled K=7 and
K=4 retained `630b7cbc72230e0d`.  The production-default sampled K=7 run
reported `DFLASH2 sampled verifier=exact-window`, 69.28 tok/s, and the same
seeded hash.  The target verifier and captured graph ABI are unchanged.

At random 64K depth the same Q8/Q8 target path showed low acceptance (45/121)
and measured 24.02 tok/s before sidecar gating.  The generation harness now
disables DFlash2 at target position 32,768 and resumes ordinary target decode
at the transaction boundary.  The guarded run retains prefix hash
`90178de69a24a76e`, emits suffix hash `5821d77a630592cb`, and measures
35.67 tok/s.  Short-window DFlash remains enabled; the HTTP/stdio quality
harness passes after the guard.

## Default GQA reuse launch (2026-09-22)

Ordinary Q8/Q8 GQA decode no longer launches both context-specialized kernels
when one would return immediately. The existing three-head reuse kernel now
runs with its adaptive guard disabled for every context, while the older
three-head kernel remains a fallback if the reuse function is unavailable.
The C++ 4K gate retained hash `96b92d606dde5e28`; three repeats measured
40.24--40.31 tok/s versus 40.09 tok/s for the paired-launch control. A matched
random 64K run retained prefix `90178de69a24a76e` and suffix
`aed3c962c4a6525d`, measuring 35.47 tok/s versus 33.73 tok/s control. The
captured graph arguments and Q8/Q8 arithmetic are unchanged.

## Asynchronous accepted-row publication (2026-09-22)

`hip_llm_qwen35_mtp_commit` now queues the selected recurrent and convolution
state copies, hidden state, logits, and position update on the target stream
without synchronizing the host before returning. Same-stream decode/propose
work remains ordered, while request reset and verifier host-logit boundaries
still synchronize. The complete DFlash2 HTTP/stdio quality matrix passed,
including seeded sampling, cancellation, cache restore, concurrency, and
multi-turn C++ output. Cached K=7 windows reported 0.23--0.57 ms commit times
on the resident gate with unchanged target token streams.

## Parallel selector candidate (opt-in)

`LLM_QWEN35_DFLASH_SELECTOR_WARP=1` selects an eight-group, 256-thread
selector geometry. Successor Q4_K rank values are decoded in parallel, while
lane zero accumulates each candidate's 256 terms in the original order. This
preserves selector tie breaks and the predecessor chain; the serialized
32-thread path remains the default pending resident K=4/K=7 hash and draft-time
A/B measurements.

The verifier commit boundary has an opt-in
`LLM_QWEN35_COMMIT_FUSED_COPY=1` path that copies checkpoint state plus the
accepted hidden/logit rows in one device launch. It leaves position publication
and the default reference ordering unchanged pending resident commit/hash A/B.

The hybrid verifier also exposes
`LLM_QWEN35_DELTANET_VERIFY_FIXED128=1` for Qwen3.8's fixed `d_state=128`
DeltaNet shape. This probe removes runtime row-count and checkpoint bounds
checks while preserving the generic recurrence's decay, warp-reduction, update,
and row-major checkpoint ordering. It has the same launch argument ABI and
remains opt-in until resident rollback/hash and sustained decode tests show a
quality-safe gain.

An opt-in `LLM_QWEN35_VERIFY_COMBINE_GROUPED=1` verifier combine kernel now
handles four adjacent rows per block. It retains the existing split partial
layout and per-row combine arithmetic, so captured generic decode graphs remain
untouched until resident long-context hash and timing A/B validation.
The kernel hoists its metadata synchronization to one barrier after all four
rows are loaded; reduction order and output bits remain unchanged.
Mode `2` of `LLM_QWEN35_VERIFY_COMBINE_GROUPED` additionally exposes an
eight-row block candidate with the same exact reduction order. It remains an
explicit probe until resident long-context hashes and timing justify it. The
grouped launch uses only the kernel's static metadata tile, so no redundant
dynamic shared-memory reservation is requested.

The grouped selector now also uses the captured query count: four-row and
eight-row windows select the narrowest exact metadata tile even when mode `3`
is enabled, while a full sixteen-row window retains the sixteen-row candidate.
This removes unused shared rows and allocation for short K=4/K=7 verifier
graphs without changing the kernel ABI or split-combine arithmetic.

The verifier also exposes an experimental
`LLM_QWEN35_VERIFY_FUSED_SPLIT_COMBINE=1` path. A verifier-only block owns one
head and four adjacent rows, walks each split, writes the existing partial
buffers, and performs the same increasing-split merge before publishing the
gated output. The captured parts/meta workspace ABI remains unchanged. This
path serializes split work inside a block and therefore remains disabled by
default pending resident long-context hashes and timing. A one-split window
publishes directly from the local reduction and applies the gate in place,
avoiding a needless partial-buffer round trip while retaining the same
single-split arithmetic.

Both grouped candidates now calculate each query's adaptive split count once
into shared metadata before loading or merging that query. The gated recurrent
verifier kernels use the same arrangement. This removes duplicate split
selection work without changing split boundaries, metadata ordering, packed-F16
accumulation, or output bytes. The generic captured combine ABI and production
defaults remain unchanged; resident-device hash and timing validation is still
required before enabling either candidate.

MTP commit now copies the accepted position from verifier device scratch into
`d_position` on the same stream. This removes a redundant host upload while
preserving the host position value, state publication order, and rollback
semantics; callers without position scratch use the previous host-copy
fallback.

## K=4 four-row Q4_K projection candidate

After the anchor row is removed, K=4 selector/output projections consume four
rows. `LLM_QWEN35_DFLASH_Q4K_MULTI4=1` selects a four-row Q4_K/Q8_1 kernel that
keeps the multi8 dot, affine correction, output layout, and warp reduction
order while dropping the four unused accumulators. The five-row proposal
projection and K=7 path retain their existing kernels, and the default remains
unchanged pending resident K=4 hash and draft-time measurements.

The ordinary target also has an opt-in
`LLM_QWEN35_IQ2XXS_QKV_FUSED=1` kernel for matching IQ2_XXS Q/K/V matrices. It
shares the native codebook tile and Q8₁ activation while preserving each
standalone row's arithmetic and reduction order. The independent launches stay
the default until resident 4K/64K hash and throughput gates validate it.

The opt-in cache-injection overlap path now consumes captured feature rows
directly after the target-ready event, eliminating a redundant device copy.
Sidecar `x`/norm/K/V workspaces remain stream-private, and the inject-done
dependency still orders the next proposal safely.

## IQ2_XS launch-bounds probe (2026-09-22)

The native one-row IQ2_XS decode kernel is compiled with
`__launch_bounds__(512, 1)` for its existing 512-thread production shape.
The kernel's arithmetic and reduction order are unchanged.  The 4K
random-token gate measured 43.00 tok/s with prefix/suffix hashes
`1c891c2232aa1b7f`/`f44846dacf013e9e`; the 64K random gate measured 35.84
tok/s at 445.82 tok/s prefill with hashes
`90178de69a24a76e`/`7463f176c9b85ba3`.  The exact C++17 gate remained
byte-identical (`44915ec1039a64c8`, SHA-256
`4a0cb461966fae9a9d9da3b73c1b0c686ce8ee9ac3895c228bc6a653bc99a354`) at
41.95 tok/s.  This is a safe small scheduling improvement; the strict 40
tok/s ordinary 64K target remains open.

A matching `__launch_bounds__` probe on the native IQ2_S, IQ3_XXS and IQ3_S
one-row kernels retained the exact 4K hashes but measured 42.84 tok/s on two
runs, slightly below the 42.9--43.0 tok/s control.  It was reverted so those
formats keep their existing shape-specific occupancy choices.

A separate 128-thread Q8 attention combine A/B also retained exact 4K and 64K
hashes but measured 35.78 tok/s at 64K versus 35.77 tok/s for the validated
256-thread merge.  It was removed as neutral; the long-context verifier tail
still needs a fusion that removes partial traffic rather than only changing
the combine width.

The IQ2_XS `__launch_bounds__(512, 2)` variant was also reverted after two
exact 4K runs ranged from 42.82 to 42.91 tok/s; the committed `512, 1` bound
remains the measured choice.

Native IQ2_XXS was tested with `__launch_bounds__(256, 1)` as well.  The exact
4K gate measured 42.94 and 42.87 tok/s across two runs, so that bound was
reverted and the existing dispatch remains unchanged.

The native Q2_K one-row kernel received the same `256, 1` bound for an A/B;
it retained the exact 4K hashes at 42.91 tok/s without exceeding control, so
the change was reverted.

An opt-in IQ3_S K/V pair launch retained the exact C++ response and 4K hash,
but measured 35.74 tok/s at random 64K versus 35.77 tok/s for the validated
separate projections.  It was removed; long-context attention still dominates
the saved small projection launch.

## Packed Q8/Q8 KV scales (2026-09-21)

Q8 K/V cache scales now use the same rounded FP16 contract as the stored
packed-F16 cache values.  The native attention kernels consume the 16-bit
scales directly, halving scale-cache traffic and preserving the existing
Q8/Q8 arithmetic.  The exact differential test passes 49,188,864 bitwise
comparisons, including 64K split counts and the multi-query reuse paths.  A
graph-captured 506-token zero-depth run completed at 43.1--43.2 tok/s with
stable output and `Result: PASS`.  The long-context operator differential is
also clean.  A real random 65,536-token depth run completed with a 443.33
tok/s prefix and 35.52 tok/s ordinary decode suffix, retaining prefix hash
`90178de69a24a76e`; the gate completed with `Result: PASS`.

The Qwen3.8 runner can load the
[IncoAI Qwen3.8-27B DFlash2 GGUF](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2-GGUF)
as an opt-in draft sidecar.  The implementation is native HIP and does not
invoke llama.cpp at runtime.

DFlash2 has a five-layer 5120-wide draft transformer with its own 32-head,
8-KV-head, 128-dimensional attention geometry and a 2048-token sliding
window.  It consumes the target inputs to layers 6, 20, 34, 48 and 62.  The
runner captures those features during each 512-token target prefill tile,
fuses them, and injects the resulting K/V rows into the draft cache.  A draft
step evaluates one non-causal block containing the target anchor followed by
mask tokens.  Its rank-256 selector walks the top-16 candidate lattice.

The target remains authoritative.  Draft tokens are evaluated by the Q8/Q8
multi-row target verifier.  Recurrent states and the target hidden state are
committed only through the accepted row.  The corresponding target features
are then injected into the draft cache, replacing the speculative rows.  A
rejection therefore cannot alter later target output.

The exact multi-row verifier is enabled for greedy, probabilistic, and coding
samplers. Every candidate row follows ordinary decode's projection, recurrent,
Q8/Q8 attention, and rollback arithmetic. The runner prints
`DFLASH2 sampled verifier=exact-window` for a sampled window. The target remains
authoritative: the sampler sees target logits and only accepted target state is
committed.

## Run

```sh
bash rdna4/llm/run_qwen38_gsq_rocm.sh --gpu-only-bench \
  --prompt-file tmp/qwen38/prefill-4k-512/coding-4096.txt \
  -n 4096 -s 8192 --ubatch 512 --kv-cache q8q8 \
  --qwen35-prefill-bf16 --qwen35-decode-graph \
  --qwen35-native-q8-prefill --qwen35-native-mmvq \
  --sampling-profile llama --temp 0 --seed 42 --decode 64 \
  --qwen35-dflash2 \
    /mnt/nvme02/models/qwen38/27b/dflash2/Qwen3.8-27B-DFlash2-Q4_K_M.gguf \
  --qwen35-dflash2-draft 7
```

Draft width may be 1 through 7.  The sidecar currently requires benchmark
mode or the resident stdio server, batched Qwen3.8 prefill, the decode graph,
and Q8 K plus Q8 V.  It is mutually exclusive with dense NextN and Qwen4 MTP.
The HTTP shim enables it with `--qwen35-dflash2 SIDECAR`; each request is
serialized through the exact propose/verify/commit window.  Temperature-zero
requests use the exact argmax window, while sampled requests verify full row
logits with the existing sampler.  Bounded target prompt snapshots include the
sidecar's recurrent K/V cache, target Q8 KV rows, captured features and prompt
logits, so cached requests can restore both target and draft state without
replaying the prompt. Snapshots above the configured token or byte budget are
rejected and replayed.

The opt-in HTTP quality gate covers deterministic greedy and seeded sampled
requests, direct stdio transactions, repeated-request state isolation,
disconnect cancellation after a generated token followed by recovery, and
concurrent request isolation:

```sh
python3 rdna4/llm/test_qwen35_dflash2_http.py \
  --model /mnt/nvme02/models/qwen38/27b/gsq/Qwen3.8-27B-GSQ-RCO-IQ2_XS.gguf \
  --sidecar /mnt/nvme02/models/qwen38/27b/dflash2/Qwen3.8-27B-DFlash2-Q4_K_M.gguf
```

## Multi-context coding-agent serving

The Python shim uses `REQ3`, which adds a SHA-256 cache identity to the stdio
request. HTTP callers may supply `prompt_cache_key`,
`metadata.conversation_id`, `metadata.session_id`, or
`X-Prompt-Cache-Key`. The runner selects the longest exact token prefix only
within that identity. Successful prompt and system-prefix boundaries enter a
transactional host LRU; cancelled or failed work is discarded without
removing earlier committed entries. Configure it with:

```sh
python3 rdna4/llm/codex_server.py TARGET.gguf \
  --qwen35-dflash2 DFLASH2.gguf \
  --context-cache-entries 4 --context-cache-max-mib 2048 \
  --qwen35-snapshot-max-tokens 16384
```

The HTTP scheduler is FIFO. Each request may provide `request_id` or
`X-Request-ID`, and every response echoes `X-Request-ID`. A targeted
`POST /v1/cancel` body of `{"request_id":"..."}` cancels that queued or
active request. Disconnect cancellation carries the same request ownership,
so a disconnected queued client cannot signal the request currently using the
GPU. Execution is deliberately serialized: the target recurrent scratch,
sampler transaction, and DFlash verifier are one mutable device context, and
no exact multi-context decode batch has yet justified changing that contract.
The server reserves a request ID before committing streaming headers;
duplicates return HTTP 409, an idle unscoped cancellation returns 404, and an
exception after SSE headers closes that response instead of appending a JSON
body to the event stream.

Snapshots are accepted only when their recorded position exactly equals the
token-key length. Q8/Q8 scales are copied at their actual FP16 row size. This
check caught a stale batched-prefill position and an oversized scale copy that
short, same-context repeats could hide. The current forced A/B/A GPU gate
restores 6,535 tokens after an unrelated conversation, reports
`cached_tokens=6535`, and reproduces the exact greedy response. That snapshot
uses 448.3 MiB versus about 234 MiB for a 26-token prompt. The harness also
checks LRU eviction, malformed metadata, concurrent identities, deterministic
seeded sampling, cancellation recovery, and a two-turn C++ task by compiling
and running both generated programs. The same harness accepts `--mtp` in place
of `--sidecar` to validate resident Dense NextN, including a separate ordinary
target process for greedy, sampled, and C++ byte parity.

The ordinary target's sampled random-64K quality gate also passes: a
temperature-0.6, seed-42 suffix retains prefix hash `90178de69a24a76e`,
produces suffix hash `34e2f6bc082bc49f`, and completes 32 tokens with
`Result: PASS` after a 445.67 tok/s random prefix.

Grouped verifier attention now combines split partials and applies the Q8
attention gate in one launch.  The pinned greedy and sampled reference gate
still reports identical tokens, EOS, and output bytes, with warm K=7 decode
above 60 tok/s.

The reference validator accepts the sidecar directly:

```sh
python3 rdna4/llm/validate_qwen38_reference.py \
  --model /mnt/nvme02/models/qwen38/27b/gsq/Qwen3.8-27B-GSQ-RCO-IQ2_XS.gguf \
  --out tmp/qwen38/dflash2-quality-k7 \
  --reuse-reference tmp/qwen38/final-iq2-native-q2k \
  --native-q8-prefill --native-mmvq \
  --dflash2 \
    /mnt/nvme02/models/qwen38/27b/dflash2/Qwen3.8-27B-DFlash2-Q4_K_M.gguf \
  --dflash2-draft 7 --decode 256 --repeats 3
```

## Validation and performance

Measured on RX 9070 XT / gfx1201 / ROCm 10 with the IQ2_XS target, a 4096-token
C coding prompt, 512-token chunks, context 8192, greedy sampling, and Q8 for
both K and V:

| Path | Prefill tok/s | Decode tok/s | Accepted drafts | Sequence hash |
|---|---:|---:|---:|---|
| Ordinary target, recent baseline | 533.19 | 39.55 | — | `15f17d2640c1adfc` |
| Native DFlash2, K=4 | 537.42 | 54.49 | 37/40 | `15f17d2640c1adfc` |
| Native DFlash2, K=7 | 540.43 | 81.05 | 41/42 | `15f17d2640c1adfc` |

All three paths emitted the same 46-token response and EOS.  The response is
valid C and implements the requested inclusive clamp without overflow-prone
arithmetic:

```c
int clamp(int x, int lo, int hi) {
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}
```

A broader 4096-token C++17 merge-intervals fixture covers greedy and
temperature-0.6 sampling.  Both K=4 and K=7 match the pinned llama.cpp token
IDs, EOS and output bytes.  The generated function passes fixed edge cases,
ASan/UBSan, and 10,000 randomized cases.  Warm results were:

| Draft width / selection | Prefill tok/s | Decode tok/s | Output SHA-256 |
|---|---:|---:|---|
| K=4 / greedy exact window | 605.43–605.74 | 59.43–59.50 | `4a0cb461966fae9a9d9da3b73c1b0c686ce8ee9ac3895c228bc6a653bc99a354` |
| K=4 / sampled exact window | 606.85–607.42 | 55.79–55.82 | `ddd1752b6c2a44251b659516b5937fdaa0e84f464530607e493abf8bbc37c9ac` |
| K=7 / greedy exact window | 607.37–608.15 | 81.68–81.82 | `4a0cb461966fae9a9d9da3b73c1b0c686ce8ee9ac3895c228bc6a653bc99a354` |
| K=7 / sampled exact window | 605.37–605.85 | 68.67–68.78 | `ddd1752b6c2a44251b659516b5937fdaa0e84f464530607e493abf8bbc37c9ac` |

The 4096-token early-context retrieval fixture also returns exactly
`ZEPHYR-7319` with K=7, including the ordinary target's token sequence and
EOS.  These results are under `tmp/qwen38/dflash2-sampled-window-k4/`,
`tmp/qwen38/dflash2-sampled-window-k7/`, and
`tmp/qwen38/dflash2-retrieval-k7.*`.

The upstream llama.cpp server reference accepted 37/40 drafts at K=4 on the
same prompt, but measured 16.54 tok/s versus its 25.88 tok/s ordinary path.
The native K=4 implementation reproduces that acceptance exactly and is 3.29
times as fast.  K=7 is 105 percent faster than the retained recent ordinary
native baseline on this prompt.  DFlash prefill also clears the 500 tok/s
target.  The feature remains opt-in.

The optimized target verifier decodes IQ and Q2_K weights once for up to eight
candidate rows.  Quantization-format-specific kernels remove runtime codebook
branches; exact Q8_1 IQ1_S/IQ1_M kernels reuse each decoded group; IQ4_XS keeps
the reference's eight virtual sums.  The fixed-eight Q2_K kernel also finishes
one query at a time after decoding a weight block, while compact IQ2 and IQ3_S
schedules reduce live accumulators and retain the reference reduction order.
IQ3_XXS keeps its faster original shared schedule.  RMSNorm and
residual-plus-RMSNorm launch one independent block per candidate row. Q4_K
draft projections quantize each candidate activation in 32-value Q8_1 groups,
reuse one decoded weight chunk across all rows, and use gfx1201 packed integer
dots. Adjacent projections with the same activation reuse the quantized input.
This reduces the 4K K=7 draft phase from 109.739 to 77.239 ms. Draft arithmetic
may change rejected proposals; accepted tokens and state still come only from
the exact target verifier. The exact Q8/Q8 verifier attention
now loads each old K/V row once and evaluates up to eight adjacent causal
queries in the same four-wave block.  It preserves each query's quantization,
online softmax, packed-F16 accumulation and split-combine order.  The draft
also reuses Q4_K weights and holds one K/V vector while evaluating four mask
rows.

The DFlash selector now decodes the predecessor's 256-rank Q4_K vector once
per draft position into shared memory, then reuses it across all sixteen
candidate lanes. This leaves the candidate accumulation order unchanged and
keeps the greedy and seeded sampled verifier hashes identical. Repeated 4K
K=4 runs measure about 53--55 ms for the complete draft phase; the remaining
cost is in the five-layer projections and attention, so the selector change is
kept as a low-risk cleanup rather than counted as a headline speedup.

Sampled parity exposed three verifier-specific hazards. Non-FFN IQ1_S
projections must use the scalar path's MMQ-scale interpretation, so their
batched launch now uses the same MMQ-scale kernel with the candidate row in
the grid Y dimension. Attention split counts must be selected independently
for each causal query, so the native query grid applies the ordinary adaptive
split policy in one launch. Finally, all verifier projections share one Q8_1
scratch allocation inside the captured graph; each projection now restages
that scratch instead of treating source-pointer identity as proof that its
contents are still live. With these fixes, all 135 sampled target rows match
the former ordinary exact-target trace bit-for-bit, including every stored
248,320-entry logit row.

The exact fixed-eight Q2_K/IQ kernels and IQ4_XS multi-row projection use
eight-wave, 256-thread blocks. This changes only the assignment of eight
output rows to a block; each row retains its existing arithmetic. In matched
K=7 traces it reduced aggregate fixed-eight projection time from 223.706 to
222.336 ms and IQ4_XS multi-row time from 47.068 to 46.624 ms. The projection
differential passes 2,948,352 exact activation values and 13,191,360 bitwise
outputs with the production launch geometry.

These changes keep the target sequence unchanged while reducing the K=7
draft/verify/commit split to 77.239/478.857/10.624 ms for the complete
46-token response.  The eight-query attention operator takes 213.382
microseconds at 4K and 3.076784 milliseconds at 64K with eight splits.  The
pinned llama.cpp differential test passes 46,743,552 bitwise Q8/Q8 output
comparisons.  The expanded projection differential passes 2,948,352 exact
activation values and 13,191,360 bitwise Q2_K/IQ outputs, including all fixed
eight-row Q2_K/IQ kernels.  The emitted source passes
`gcc -std=c17 -Wall -Wextra -Wpedantic -Werror` and boundary tests using
`INT_MIN` and `INT_MAX`.

The fixed-shape GDA recurrence has separate contracts for raw scalar alpha and
prefill's precomputed decay.  Its fixed-bound loop preserves the generic
kernel's operation order under the runner's `-ffast-math` HIPRTC mode.  The
dedicated test compares 19,537,920 state/output values bitwise across both
contracts.  This restores the pinned sampled sequence while retaining the
specialized prefill speed; spelling the four rows as separate accumulators
reassociated operations and changed the sampled output.

Decode attention preparation now uses one exact kernel per attention layer for
Q/gate deinterleave, Q and K RMSNorm, Q and K M-RoPE, and Q8/Q8 K/V storage.
It replaces six launches with one in each of the 16 attention layers, removing
80 launches per decoded target row. The fused kernel retains the separate
operators' reduction order, trigonometric operations, Q8 scale rounding and
integer conversion. The complete K=4 and K=7 greedy and sampled C++ gates
remain byte-identical to the pinned llama.cpp fixtures. Both emitted functions
pass ASan/UBSan, fixed edge cases, and 10,000 randomized cases. At K=7, the
exact sampled window sustains 68.67--68.78 tok/s instead of the former
40.17--40.23 tok/s scalar fallback. K=4 sustains 55.79--55.82 tok/s; its
shorter draft does less useful work per verifier launch and remains below the
60 tok/s speculative target.

The earlier fixed-eight-split K=7 path measured 49.74 tok/s after a fully
processed 65,536-token random prefix, with suffix hash `2ddd068dca63669a`.
That result predates exact per-query adaptive split selection and is retained
only as a historical ceiling; it is not the current exact oracle.

The current verifier chooses between two captured graphs.  When every
adjacent causal row selects the same ordinary adaptive split count, the exact
eight-query kernel loads each K/V row once.  At a split-selector boundary it
uses the generic per-query grid.  This removes the fixed-eight arithmetic
shortcut without launching a full generic grid merely to return.  On the
random-64K gate, the generic exact baseline was 33.25 tok/s.  The selected
shared-K/V graph sustains 39.89--39.93 tok/s, with 443.45--444.22 tok/s
prefill, prefix hash `90178de69a24a76e`, and current exact suffix hash
`1c68ea2ff63ba5ab`.  All runs drafted 289 tokens and accepted 213; the final
draft/verify/commit range was 532.352--533.268 / 5818.192--5821.733 /
60.532--62.229 ms.  This is the pre-retune 16-split sidecar baseline; the
eight-split sidecar result below clears 40 tok/s with the same exact hashes.

The earlier DFlash2 draft attention retune used eight splits once its 2,048-token
window was at least half full (shorter windows retained the existing 1/4-split
schedule).  This was separate from the target verifier's adaptive split
selector.  On the random-token 64K K=7 gate, the exact target suffix hash
remained `1c68ea2ff63ba5ab` and the prefix hash remained
`90178de69a24a76e`; decode rose from the matched 16-split baseline of 39.91
tok/s to 41.01 tok/s.  The current production choice is twelve splits based on
the fixed-window comparison above.  The target verifier and its captured graph
ABI are unchanged; only the sidecar proposal attention launch is retuned.

The tested sidecar is
`Qwen3.8-27B-DFlash2-Q4_K_M.gguf`, SHA-256
`1a25c56858e1ebe93f2718ac1d49d1151f9323325c1bbfd6209370f4db131ebd`.

The ordinary target's recurrent alpha and beta F16 projections now use one
flattened launch while retaining the exact per-row FMA and XOR-reduction
order.  In a matched 65-row trace, this replaces 6,144 launches taking 23.102
ms with 3,072 launches taking 12.492 ms, or about 0.163 ms saved per target
row.  Zero-depth decode measures 42.83--43.04 tok/s and keeps the pinned
256-token hash `3c53b75f283cb9b0`.  The profile is under
`tmp/qwen38/ordinary-decode-profile-f16pair/`.

The dense FFN path also fuses SiLU multiplication with native Q8_1 staging
when the following down projection uses Q2_K, IQ2/3, or IQ4_XS.  This covers
58 of 64 layers in the tested IQ2_XS target and removes 58 launches per
ordinary decoded row.  The protected quotient and FP16 scale conversion match
the split quantizer: all 248,320 final logits are bitwise identical in the
fused/split A/B, and the 256-token hash remains `3c53b75f283cb9b0`.
Three-repeat zero-depth decode rises from a 42.79 tok/s split mean to a 43.11
tok/s fused mean.  The kernel trace is under
`tmp/qwen38/ordinary-decode-profile-siluq81/`; set the diagnostic
`LLM_QWEN35_SPLIT_SILU_Q81=1` to restore the two-launch boundary.

For native Q8 attention, the host now skips the separate combine dispatch when
the causal window is no more than one 256-token tile. The decode and prefill
kernels already store the final normalized row for `splits == 1`, so this is a
dispatch-only cleanup: Q8/Q8 arithmetic, output layout, and the captured
long-context graph are unchanged. Long-context windows still use the existing
split/combine path.

The 48 recurrent output layers now fold native Q8_1 staging into their
per-head gated RMSNorm/SiLU kernel.  The fused kernel retains the original
128-thread reduction, load loop, and stored activation boundary before four
independent waves quantize each head.  Its split-path A/B is bitwise identical
for all 248,320 final logits (SHA-256
`5b5f2f1a334ae644ac5633908e3d447c6d741c764a61dc0e9573addf699553c0`),
and three-repeat decode keeps the pinned `3c53b75f283cb9b0` sequence hash at
42.62--43.36 tok/s (43.07 mean), versus 41.64--43.14 tok/s (42.60 mean) for
the matched split path.  A 64-row trace removes exactly 3,072 launches, 48 per
row, and is under `tmp/qwen38/ordinary-decode-profile-ssmq81/`.  Set
`LLM_QWEN35_SPLIT_SSM_NORM_Q81=1` to restore the standalone quantizer.

Gated-attention output now applies sigmoid gating and stages native Q8_1 in
one wave-per-block kernel for 12 of 16 attention layers; the four IQ1 output
layers retain their dedicated quantizer.  The fused/split final logits are
bitwise identical with the same `5b5f2f1a...953c0` SHA-256, and all three
256-token repeats keep `3c53b75f283cb9b0`.  Matched decode means are 43.33
tok/s fused and 43.21 tok/s split.  A 64-row trace replaces 768 standalone
quantizer launches with 768 `sigmoid_mul_q81_f32` launches, reducing the total
by exactly 12 launches per row; it is under
`tmp/qwen38/ordinary-decode-profile-attngateq81/`.  Set
`LLM_QWEN35_SPLIT_ATTN_GATE_Q81=1` for the split path.

The exact one-row IQ scheduler now chooses its block geometry by projection
shape: IQ2_XS uses sixteen waves, the 17408-row IQ2_S shape uses sixteen, and
the frequent 5120/17408-row IQ3_S shapes use four. A wave still owns one
output row, preserving the reduction order. Matched 64-row traces save about
0.23 ms per decoded token; the tuned and eight-wave fallback paths produce
bitwise-identical 248,320-entry logits with SHA-256
`5b5f2f1a334ae644ac5633908e3d447c6d741c764a61dc0e9573addf699553c0`.
The authoritative 64K run keeps prefix/suffix hashes
`90178de69a24a76e`/`f4b35758fb99e6db` at 441.44 prefill and 34.98 decode
tok/s. `LLM_QWEN35_IQ_SHAPE_THREADS=0` restores the diagnostic fallback.

The verifier now embeds all IQ1_M candidate tokens with one two-dimensional
launch.  Accepted convolution and recurrent checkpoints are also published by
one kernel across every recurrent layer, replacing the former pair of commit
launches.  The post-change K=4/K=7 greedy gates retain sequence hash
`44915ec1039a64c8`; seeded K=7 retains sequence hash `630b7cbc72230e0d` and
the pinned output/token SHA-256 values.  K=7 measures 81.20 tok/s in the exact
gate, while the three warm embedding-batch runs measured 81.42--82.49 tok/s.

The grouped target verifier's IQ1_S gate and IQ1_M up projections now share
their exact Q8₁ activation quantization when the normalized source, width, and
row count match. Layer entry clears the source cache because the normalized
scratch buffer is reused for the next layer, and
`LLM_QWEN35_MTP_IQ1_Q81_REUSE=0` restores the independent-quantizer control.
The resident DFlash2 HTTP/stdio and multi-turn C++ quality suite passes in
both modes. The matched 113-token cached request measured 433.458 ms verifier
time with reuse versus 432.478 ms control, so this is currently a safe launch
reduction with no claimed end-to-end gain.

At each of the five target feature taps, capture now shares the existing exact
RMSNorm reduction and output loop.  The 4K gate remains byte-identical and
measures 536.17 tok/s cold, 610.86--612.05 tok/s warm, and 81.12--82.37 tok/s
decode.  A complete 65,536-token random prefix retains hash
`90178de69a24a76e` at 444.17 tok/s.  Overlapping sidecar injection with the
next tile is now stream-safe for the existing per-stream hipBLASLt workspace
cache. The overlap stream has an additional diagnostic,
`LLM_QWEN35_DFLASH_INJECT_KV_FUSED=1`, which must be set before sidecar load.
When the prompt-only K/V weights support the BF16 cache, one wider GEMM writes
the K and V ranges into a private strided tile; the existing exact Q/K norm,
RoPE, and cache-store kernels consume that tile without a split copy. Layers
with unsupported weight types keep the separate BF16 GEMMs. This changes only
the opt-in injection path and still needs resident cache-hash and quality
validation before it can be considered for a serving default. The overlap
workspace check accepts either this fused K/V allocation or a complete
separate K/V pair, while rejecting incomplete layer workspaces before the
private stream is used.

The sidecar also exposes the diagnostic
`LLM_QWEN35_DFLASH_QKNORM_ROPE_FUSED=1` path. It combines K RMSNorm and
M-RoPE after injection, retaining the reference reduction and rotation order
while removing one intermediate read/write and launch per layer. The
two-launch path remains the default until resident cache hashes and draft-time
measurements validate the candidate.

The real-GPU quality harness now runs the resident JSONL protocol directly and
then exposes the same backend through HTTP.  It checks greedy and seeded
sampling, three repeated cache hits, cancellation after a real streamed token,
recovery, and two concurrent callers.  The short gate passes, and a
6,600-token actual long prompt passes greedy and sampled reuse, active-window
cancellation, recovery, and request isolation.

Adaptive shared-K/V verifier graphs retain the complete pinned sampled trace:
sequence hash `630b7cbc72230e0d`, output SHA-256
`ddd1752b6c2a44251b659516b5937fdaa0e84f464530607e493abf8bbc37c9ac`,
token SHA-256
`fb7d8aeda396cdba5dd65b492a396ed3f91ae4312ea0a52d77be86355b4c7ee0`,
and logits SHA-256
`4b3489e92bcaf7f442b0f86722e88a5576cf7e16ea52193d73eecad12835ac3d`.
The 4K K=7 gate measures 82.67 tok/s greedy and 66.46 tok/s sampled with
trace I/O; K=4 remains exact at 59.75 tok/s.  A selector-boundary window uses
the generic graph, so graph reuse never changes a row's split count.

Four follow-up variants were rejected.  Device guards that launched both
attention grids reached only 39.55 tok/s at 64K.  Pinning a captured graph's
split count and sharing combine scales did not improve verifier time.  A
one-wave combine and a fused draft/verify synchronization path were both
byte-identical but slower on the traced 4K gate.  The retained graph selector
is the only measured win.  Moving the eight-query length metadata from
per-thread arrays to LDS also retained exact random-64K prefix/suffix hashes
and raised prefill to 445.21 tok/s, but verifier time increased to 5927.105 ms
and decode dropped to 39.26 tok/s; the register-pressure experiment was
therefore reverted.

## Remaining optimization opportunities

The current 4K gate sustains 66.46 tok/s for traced sampled K=7 and 82.67
tok/s for greedy K=7, clearing the 60 tok/s target.  DFlash reaches
39.89--39.93 tok/s after a real random-token 64K prefix. Ordinary one-token decode now
reaches 34.98 tok/s at 64K after exact GQA reuse, grouped K/Q scale products,
packed-probability reuse and scalar IQ codebook staging, so work that helps
both ordinary and verifier execution remains useful. The following order
reflects the remaining measured costs.

1. **Ordinary one-row target projections.** Scalar IQ2_XXS, IQ2_XS and
   IQ3_XXS now stage their small codebooks in LDS, lifting zero-depth decode
   from about 40.7 to 41.9--42.0 tok/s and 64K decode from 32.56 to 33.31
   tok/s. Hoisting repeated K/Q scale products in the exact three-head
   attention kernel raised the 256-token 64K run to 34.08 tok/s. Reusing each
   packed probability across both value tiles lowers the exact 128-split
   operator to 321.8--323.6 microseconds per layer and raises the full run to
   34.98 tok/s with unchanged prefix and suffix hashes, so the
   remaining gap is dominated by work outside attention. SiLU and Q8_1
   staging for 58 dense down projections now share one exact launch; gated
   RMSNorm and Q8_1 staging do the same for all 48 recurrent output
   projections, and sigmoid gating does the same for 12 attention output
   projections.
   Fixed-eight Q2_K/IQ projections already share decoded weights, but
   ordinary decode still streams weights for one row at a time. Reuse the
   quantized input across gate/up projections and investigate cooperative
   weight staging.  The opt-in IQ1 Q8_1 audit now reuses the activation bytes
   and block sums across the IQ1_S gate and IQ1_M up pair; it raises the
   measured 65,536-depth suffix from 34.18 to 36.13 tok/s with the same
   prefix/suffix hashes.  The seeded temperature-0.6 C++ response also kept
   sequence hash `630b7cbc72230e0d` and output SHA-256
   `ddd1752b6c2a44251b659516b5937fdaa0e84f464530607e493abf8bbc37c9ac`.
   It changes full logits and therefore remains diagnostic until the broader
   logit matrix is complete. A WMMA or reordered
   reduction path needs full output-token and logit validation because the
   current kernels preserve the target arithmetic order.
   The mixed IQ1 gate/up candidate now stages the packed 2K-entry IQ1 codebook
   once per block in LDS before evaluating either output range. This removes
   repeated global codebook reads while preserving the existing IQ1_S and
   IQ1_M dot, affine, and warp-reduction order; it remains behind the same
   diagnostic fusion switch pending resident parity and timing.
   The IQ1_M one-row F32 kernel now uses explicit non-aliasing qualifiers and
   unrolled four-value FMA halves; a matched 4K random gate moved 43.64 to
   43.72 tok/s with the same sequence hash.  A fused IQ1_S Q8_1 gate plus
   IQ1_M F32 up kernel was exact but 41.74 versus 41.90 tok/s on the pinned
   C++ gate, so it was removed.  The 64K ordinary 40 tok/s gap therefore
   remains a grouped mixed-type projection problem.
   A 128- and 512-thread block-shape probe for the native IQ2_XXS one-row
   kernel retained the 4K random prefix/suffix hashes
   `1c891c2232aa1b7f`/`ab4dd24f5cdf0b2c`, but three-repeat decode means were
   39.68 tok/s at 256 threads and 39.64 tok/s at 512. The alternate geometry
   was removed; its extra rows per block did not reduce the long-context
   projection cost.
   An opt-in `LLM_QWEN35_FUSED_DOWN_RESIDUAL=1` IQ2_XS down kernel now adds
   the native Q8_1 projection directly to the live residual, removing the
   temporary output and separate add launch. It preserves the native IQ2_XS
   reduction order and remains disabled until resident random-64K hashes and
   throughput show a quality-safe gain.

   An opt-in `LLM_QWEN35_Q2K_QKV_FUSED=1` candidate now covers ordinary
   attention layers whose Q/K/V matrices are all Q2_K with a common input
   width. It shares the exact Q8_1 activation quantization and selects the
   three output ranges in one native grid while retaining the standalone
   eight-virtual-warp Q2_K partial and reduction order. The host keeps the
   serialized launches as the default until resident 4K/64K logits, hashes,
   and throughput are measured; the kernel and host build pass without a
   resident gfx1201 device in this environment.
   Dense Q2_K gate/up pairs have a matching opt-in
   `LLM_QWEN35_Q2K_GATEUP_FUSED=1` dispatch. It uses the same Q2_K QKV kernel
   with a zero V range, shares one exact Q8_1 activation quantization, and
   retains the tall 128-thread geometry used by standalone 5120-column FFN
   rows. The fused path still runs the ordinary SiLU stage separately and is
   diagnostic until resident logits, hashes, and 64K timing prove a gain.
   Ordinary all-Q4_K attention has a separate diagnostic
   `LLM_QWEN35_Q4K_QKV_FUSED=1` candidate. It reuses the DFlash Q4_K/Q8_1
   multi-output kernel for a single Q/K/V row and one activation-quantization
   launch, removing two projection launch boundaries. It remains opt-in
   because Q8_1 staging changes the serialized Q4_K arithmetic contract and
   needs resident logits, hashes, and sustained 64K validation. The launcher
   checks the Q8_1 batch scratch pointers before selecting it, so allocation
   failure falls back to the serialized projection path safely.
   Mixed native IQ2/IQ3/IQ4 attention has a separate diagnostic
   `LLM_QWEN35_IQ_MIXED_QKV_FUSED=1` path. It shares one Q8_1 activation and
   stages all supported small codebooks once per block, while selecting the
   exact standalone arithmetic for each Q, K, and V row range. The serialized
   dispatcher remains the default until resident mixed-format logits, token
   hashes, and random-64K timing validate the launch reduction. LDS staging is
   format-aware, so each block loads only the codebooks selected by its Q/K/V
   row kinds. `LLM_QWEN35_IQ_MIXED_THREADS=512` is an additional cached
   geometry probe; unset or invalid values retain 256 threads and all
   production defaults remain unchanged.
   Heterogeneous dense FFN gate/up pairs can use the related diagnostic
   `LLM_QWEN35_IQ_MIXED_GATEUP_FUSED=1` path. It uses the same format-aware
   kernel with an empty V range, so only the gate and up row ranges execute;
   equal-format pairs and unsupported metadata retain their existing
   format-specific or serialized dispatch.
   Same-format IQ2_S gate/up pairs have a separate diagnostic
   `LLM_QWEN35_IQ2S_GATEUP_FUSED=1` switch because they have no existing
   dedicated native gate/up entry point.
   Same-format IQ2_S attention rows can use
   `LLM_QWEN35_IQ2S_QKV_FUSED=1` for the analogous one-launch Q/K/V candidate;
   the serialized three-projection path remains the default.
   The fused selector also specializes the common K=4 proposal window with
   `qwen35_matvec_q4k_q81_qkv_fixed4`, matching the existing fixed-seven
   candidate and preserving the generic vocabulary/rank output layout.
   The opt-in one-row projection helpers also reject zero-width metadata before
   launching activation quantization, keeping malformed shapes on fallback.
   The shared hipBLASLt bridge rejects non-positive M/N/K or batch dimensions
   before plan creation, so malformed sidecar shapes cannot reach the library.
   Its strided-batch entry point rejects non-positive batch counts before the
   batch-1 fast path, so invalid metadata cannot be silently reinterpreted as
   a single-row GEMM.
   Its initialization path rolls back partially-created hipBLAS and hipBLASLt
   objects when a handle, preference, or preference-attribute call fails, so an
   optional retry cannot inherit stale library state.
   The lazy per-stream workspace map is protected by a narrow mutex; target
   and sidecar launches still execute on their own HIP streams, while
   concurrent first-use allocation cannot race or duplicate a scratch pointer.
   Plan initialization and cache insertion use a separate short-lived lock, so
   concurrent shape misses cannot corrupt the shared plan map or duplicate a
   heuristic query; heap-stable plan entries prevent a later unordered-map
   rehash from invalidating a plan already being submitted; matmul submission
   remains outside both cache locks.
   The shared-handle F16 fallback similarly guards stream selection through
   enqueue, avoiding cross-context stream substitution while leaving device
   work asynchronous.
   Mutable bias and epilogue descriptor attributes are protected through their
   enqueue transaction as well, preserving per-request bias pointers when
   target and sidecar contexts submit concurrently.
   Reset and MTP rollback now skip the pre-wait target synchronize when an
   injection is pending; the overlap helper inserts the event dependency and
   one host-visible fence, while the no-overlap path retains its original
   fence.
   The other DFlash Q8_1 projection and fused-SiLU candidates use the same
   scratch guard, keeping partial allocator failures on their native fallback.
   The one-row IQ dispatch snapshots the immutable MMQ-D4 quantizer and
   IQ-shape-thread options once per process instead of re-reading the
   environment for every projection. Default selection and arithmetic are
   unchanged; this only removes repeated host-side option parsing.
2. **Verifier attention tail.** The query-grid verifier now selects ordinary
   decode's split count independently for every causal row. Equal-split
   windows now select a dedicated captured shared-K/V graph; split boundaries
   select the generic graph. The kernel still writes split partials for a
   second combine launch, and that pass dominates the long-context verifier
   tail. A prototype that extended the
   generic attention kernel ABI faulted under captured graph replay, even when
   the new body was disabled.  Use a separate verifier-only kernel and fuse
   the combine only if the selected split count and packed-F16 accumulation
   order remain exact.  A 64-thread exact merge probe retained the random
   64K hashes but fell from 35.77 to 35.64 tok/s, so it was removed; reducing
   partial traffic rather than only changing combine occupancy is still open.
3. **Hybrid recurrent tail.** Sequential candidate recurrence and rollback
   checkpoints are already batched and device-local. Alpha/beta F16 work now
   shares one exact launch per recurrent layer, and convolution plus recurrent
   state publication now shares one commit launch across all recurrent layers.
   DeltaNet checkpoint writes and the remaining matrix-vector work remain
   visible.  An attempted final-row copy elision changed target output after
   26 tokens because live state remains at the transaction origin; retain
   explicit accepted-row publication.

   The opt-in fused publication's unaligned hidden/logit scalar loops now
   copy every element, fixing a stride-of-four indexing bug. Run
   `python3 rdna4/llm/test_verifier_commit_copy.py` for 256 CPU cases executing
   the production copy bodies against full-row references and destination
   sentinels; this does not replace resident GPU parity or timing tests.

   The checkpoint copy path now makes the row-major convolution and recurrent
   strides explicit, so a DFlash window commits only its accepted row. GPU
   HTTP and pinned llama.cpp gates retain greedy and seeded-sampled token/byte
   parity after this hardening.
   A restricted-pointer/four-way-unroll DeltaNet scalar probe was exact but
   moved the 4K random gate from 42.90 to 42.80 tok/s, so it was reverted.
   The verifier now also uses the existing fused Q/K normalize-and-expand
   kernel, removing three intermediate launches per recurrent layer while
   preserving the same per-head reduction order.
   Attention verifier Q/K RMS normalization is likewise paired into one launch;
   its independent reductions retain the original order and exact output
   hashes.
   The opt-in fused recurrent commit copy now sizes its grid for the largest
   recurrent state, hidden row, or vocabulary-logits row. This prevents a
   large vocabulary from leaving the tail of the accepted logits row
   unpublished; the ordinary two-launch publication remains unchanged. The
   copy kernel also avoids rewriting the scalar tail after its unaligned-stride
   path has already covered the row, while retaining the same accepted-row
   ordering and checkpoint publication semantics. It also publishes a final
   1--3-float tail for convolution and recurrent snapshots when a state length
   is not divisible by four; aligned shapes stay on the vector path, while
   genuinely unaligned strides use a complete scalar copy so a vector load
never crosses a row boundary. The host grid uses ceil division, keeping
that tail path reachable for sub-four-float states too; an empty state is
clamped to one block rather than issuing a zero-grid launch. The fused
selector now requires at least one recurrent layer because its layer-zero
branch publishes the accepted hidden/logits row; zero-layer configurations
retain the scalar I/O fallback.

   A three-repeat DeltaNet warp-per-row batch probe averaged 38.99 tok/s on
   the seeded sampled 4K gate versus 38.94 tok/s for the reference-order
   implementation, with identical sequence hash `c6bb94e73050164e` on every
   run. The gain is below measurement noise, so the warp path remains opt-in
   (`LLM_SSM_BATCH_WARP=1`) and the reference-order default is retained.

   The HTTP quality harness now forces another conversation between long-prompt
   requests, which proves restoration from host state instead of reuse of the
   still-live GPU context. The validated interleaved prompt is 6,535 actual
   tokens. Q8 target snapshots are bounded to 16,384 tokens by default; larger
   bounds can be requested with `--qwen35-snapshot-max-tokens N`, but they also
   need a sufficient `--context-cache-max-mib` budget. Earlier 10K--60K
   same-context observations must be rerun with this interleaved gate before
   being treated as portable-cache validation.
   Batched verifier SSM alpha/beta preparation now uses one elementwise launch
   for softplus/scale and sigmoid, with exact llama.cpp hashes preserved.
   When both alpha and beta weights are F16 with matching shapes, the verifier
   now also has an exact batched pair matvec behind
   `LLM_QWEN35_MTP_F16_PAIR_BATCH=1`. It flattens both output ranges into one
   launch over the same verifier rows while retaining the standalone half2 FMA
   and XOR reduction order. BF16 or mismatched shapes retain the independent
   kernels, and the new path is covered by the profile guard but remains
   unmeasured on gfx1201 until a resident run is available.
   Matching BF16 alpha/beta projections have the corresponding opt-in
   `LLM_QWEN35_MTP_BF16_PAIR_BATCH=1` path. It retains the standalone BF16
   conversion/FMA/XOR order and llama.cpp block-size heuristic while removing
   the second launch; F16 and mismatched shapes are unchanged, and resident
   parity/timing is still required before changing defaults.
   A verifier-only eight-row combine kernel was also tested against the
   random-token 64K gate.  It remained exact but measured 28.52 tok/s versus
   the retained 28.82 tok/s dense-MTP control, so the generic captured combine
   remains in production.
   First-use verifier workspace allocation is transactional as well: a failed
   device or host allocation synchronizes any queued pointer-table copies,
   releases the partial buffer set, and resets capacity so a later request can
   retry without recreating the runner. This is failure-path hardening only;
   successful checkpoint and graph execution is unchanged.
4. **Kernel and graph count.** Q/gate deinterleave, QK normalization, RoPE,
   Q8/Q8 KV storage, SSM alpha/beta preparation, and IQ1_M verifier embedding
   are now fused or batched exactly. Profile again before joining another
   mixed-type preparation boundary; retain only savings larger than dispatch
   noise.
5. **Remaining draft cost.** Top-k and selector decisions already run on the
   GPU, and packed Q4_K/Q8_1 projections cut draft work substantially.  The
   current exact 64K suffix spends about 562 ms in the drafter after the
   eight-split sidecar attention change; projection and selector work remain
   the material cost.
   Position-parallel attention and
   cheaper draft-cache storage are the next candidates, provided K=4/K=7
   acceptance and authoritative output remain stable.

   The opt-in eight-group selector now lets group zero populate the shared
   256-rank predecessor array; the other groups reuse it after the barrier.
   This removes duplicate Q4_K loads while preserving ordered score arithmetic,
   tie breaks, and the predecessor chain.

   The anchor and fixed mask-token rows now use one scalar IQ1_M mask embedding
   plus device copies for the remaining rows. The pinned greedy K=7 gate is
   exact at 82.65--84.12 tok/s warm (140 drafted/134 accepted, hash
   `44915ec1039a64c8`), and seeded sampled K=7 remains exact at 70.47 tok/s
   (140 drafted/115 accepted, hash `630b7cbc72230e0d`). Projection cost remains
   the material sidecar target after eliminating redundant mask embedding work.
   An opt-in `LLM_QWEN35_DFLASH_QKV_FUSED=1` candidate now combines the
   three Q4_K Q/K/V output ranges into one grid after their shared Q8_1
   activation quantization. K=4 uses a compile-time five-row accumulator and
   wider windows use the eight-row-capable form; both retain the existing
   per-row dot and reduction order. The serialized three-launch path remains
   the default until resident K=4/K=7 hash and draft-time measurements prove
   the launch reduction is beneficial.
   `LLM_QWEN35_DFLASH_GATEUP_FUSED=1` similarly reuses that exact kernel for
   the dense Q4_K gate/up pair with no V range. It is a separate opt-in probe;
   the ordinary gate/up/SiLU sequence remains the serving default.
   `LLM_QWEN35_DFLASH_EMBED_BROADCAST_KERNEL=1` is an additional diagnostic
   for K=4/K=7 windows: after the exact anchor and first mask embedding, one
   row-repeat launch fills all remaining mask rows. The established device
   copy loop remains the default until resident timing and quality checks show
   a benefit.
   The selector also has an opt-in `LLM_QWEN35_DFLASH_SELECTOR_WARP16=1`
   geometry. It assigns one warp to each of the sixteen candidates in a
   single 512-thread block, removing the two eight-candidate batches while
   retaining the predecessor decode, ordered 256-term score accumulation, and
   first-max tie break. The existing 256-thread selector probe remains
   available through `LLM_QWEN35_DFLASH_SELECTOR_WARP=1`; both variants are
   diagnostic until resident K=4/K=7 hashes and draft-time measurements show a
   quality-safe gain.
   An opt-in `LLM_QWEN35_DFLASH_SELECTOR_FUSED=1` candidate combines the
   vocabulary-logit and rank-256 selector-hidden Q4_K projections after one
   Q8_1 activation staging. The two output ranges keep their independent row
   layouts and reduction order; the serialized projections remain the default
   until resident selector hashes and draft timing validate the launch saving.
   The K=7 selector shape selects a fixed-seven activation-count variant of
   the same kernel, removing its final runtime row predicate without changing
   the output layout.
   The proposal Q/K preparation now has a separate
   `LLM_QWEN35_DFLASH_QKNORM_ROPE_PAIR_FUSED=1` candidate. It combines the
   paired Q/K norm reductions and both M-RoPE transforms after the projection
   grid, retaining the normalized global store barrier and rotation order
   while replacing four launches with one. The established four-launch path
   remains the serving default pending resident K=4/K=7 hash and timing gates.
6. **Prompt-cache injection.**  Feature capture now shares the target
   RMSNorm kernel and both 4K and random-64K prefill retain their targets.
   The hipBLASLt bridge now allocates scratch lazily per HIP stream, so a
   target and sidecar stream can safely share a shape without racing a
   workspace.  A two-stream BF16 smoke test passes with identical outputs.
   The opt-in injection path now records a target-ready event before the
   sidecar stream starts and retains an injection-done event before the next
   proposal reuses sidecar scratch.  The corrected overlap quality suite
   passes all HTTP/stdio and C++ checks.  A matched 4K K=7 run measured
   27.03 tok/s overlap versus 27.21 tok/s serialized with the same sequence
   hash, so the serial path remains the production default until overlap
   demonstrates a real throughput gain.  If injection or event recording
   fails after enqueue, the private stream is now synchronized and marked
   idle before the error escapes, preventing a later proposal from reusing
   live scratch.  Permanent overlap setup failures also set a runner-local
   disabled bit, so later commits immediately use serialized injection
   instead of retrying failed stream, event, or workspace allocation.
   A request that exceeds the private eight-row overlap tile also falls back
   to the serialized injector after retiring any pending private work, so the
   optional overlap path cannot turn a larger future batch into a request
   failure. The public commit entry point also waits on any prior pending
   injection, protecting private scratch if a caller skips a proposal
   boundary. MTP and full runner reset paths retire pending overlap work before
   snapshot restore or new-request cache initialization; their reset-only wait
   synchronizes the target stream so completion is host-visible before
   synchronous restore copies.

Each optimization should retain the exact sequence hash and response bytes at
K=4 and K=7, compile the emitted program, and cover non-coding prompts plus
random-token 64K depth.  The HTTP/stdio quality gate now covers direct and
OpenAI-compatible window transactions at short and long prompt lengths.
