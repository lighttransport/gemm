# Qwen3.8 27B HIP runner vs llama.cpp — resume state

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

After a fully processed 65,536-token random prefix, DFlash K=7 now sustains
**49.74 tok/s** for a 256-token suffix. The prefix sustains 443.57 tok/s and
has hash `90178de69a24a76e`; the suffix retains hash `2ddd068dca63669a`.
It drafted 259 tokens, accepted 217, and spent 474.117/4585.996/56.181 ms in
draft/verify/commit.  This meets the random-depth 40 tok/s goal.  Ordinary
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

Prioritize ordinary one-row target projection traffic first, followed by an
exact in-kernel combine for the long-context verifier attention tail and
fusion of the remaining activation/state-preparation launches.
The shared-K/V attention path already uses the ordinary adaptive split policy.
DFlash prompt feature capture and
sidecar-cache injection are secondary prefill targets now that both 4K and
random 64K prefill clear their goals.  Detailed priorities and validation
gates are in the linked DFlash2 document.

Remaining optimization items, in measured priority order:

1. Raise ordinary random-64K decode from 35.52 tok/s to 40+ tok/s.  The
   dominant cost is still one-row IQ2/IQ3/IQ1/IQ4 and Q2_K projection traffic;
   evaluate exact grouped or multi-row mixed-type kernels with shared
   activation staging and codebook layout changes.
2. Fuse the long-context verifier attention split/combine tail while keeping
   the pinned logits and sampled output bit-identical.
3. Batch exact SSM recurrence updates and reduce checkpoint/rollback copies;
   revalidate K=4/K=7 acceptance, retrieval, and random 64K hashes.
4. Fuse remaining verifier preparation launches where mixed-type grouping has
   enough coverage to pay for its dispatch and synchronization cost.
5. Move DFlash top-k/selector work and draft-cache overhead onto the GPU;
   K=7 already clears 60 tok/s, while K=4 remains below that target.
6. Fuse prompt feature capture with target hidden writes and overlap sidecar
   cache injection with the next prefill tile.  This is secondary because
   4K/512 and random-64K prefill already exceed 400 tok/s.
7. Revisit dense NextN/MTP scheduling; the current exact implementation is
   slower than ordinary decode and remains below 60 tok/s.
8. Extend HTTP/stdio serving coverage to DFlash2 window transactions and
   broaden long-context sampled and quality coverage once the performance work
   stabilizes.

The first serving step is now complete for the ordinary Qwen3.8 target, and
the exact DFlash2 window transaction is now available through the same
resident stdio child.  The
`codex_server.py --qwen35-server-profile` option starts the resident stdio
runner with the validated exact Q8/Q8 cache, native prefill, decode graph and
MMVQ routes.  A loopback OpenAI-compatible request returned the exact
`READY` response and stopped on the model's `<|im_end|>` token; the existing
protocol suite passes all 12 tests, including request cancellation, cache
reuse, tool-call framing and diagnostic alignment.  DFlash2 is available with
`--qwen35-dflash2 SIDECAR --qwen35-dflash2-draft 1..7`; its exact
propose/verify/commit window is serialized per request.  A real two-request
loopback test returned coherent output for both greedy and sampled requests;
the DFlash sidecar currently replays each prompt because target snapshots do
not yet include its private recurrent cache.  This favors output correctness
over cache reuse until a sidecar snapshot is implemented.
The reproducible GPU gate is `test_qwen35_dflash2_http.py`; it checks greedy
and seeded sampled repeatability plus a coherent C++ response.
The sampled random-64K target gate now passes 32 suffix tokens with prefix hash
`90178de69a24a76e`, suffix hash `34e2f6bc082bc49f`, and `Result: PASS` after a
445.67 tok/s prefix.

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

The optimized DFlash K=7 path processes the same random prefix in 147.747
seconds at 443.57 tok/s, then sustains 49.74 tok/s for 256 tokens with suffix
hash `2ddd068dca63669a`.  Its exact eight-query shared-K/V attention kernel
takes 3.077 ms per layer at 64K, compared with 8.654 ms for the generic
verifier at the same eight-split schedule.  DFlash therefore meets the 40
tok/s random-depth target. Ordinary decode remains below 40 tok/s. Its exact
three-head K/V-reuse kernel cuts the 128-split attention operator from about
606 to about 348 microseconds per layer after the grouped scale-product
change, leaving about 5.6 ms/token in attention and
25 ms/token in the projection/state path. Dense NextN still needs a real
random-depth rerun.
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

## Final decode validation (2026-09-20)

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

**Ordinary 40 tok/s and dense NextN 60 tok/s remain unmet; DFlash2 now clears
60 tok/s at 4K and 40 tok/s at random-token 64K depth.** Ordinary
greedy decode improved from 33.8 to 38.0 tok/s on IQ2 and 32.1 to 36.1 on
IQ3 (about 12–13%). MTP is opt-in because this verified implementation is
slower on the tested coding response. The next performance work belongs in
multirow projection reuse and reducing verification/checkpoint overhead.

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
[QWEN38_DENSE_MTP.md](rdna4/llm/QWEN38_DENSE_MTP.md). The benchmark/C API
supports `--qwen35-mtp SIDECAR --qwen35-mtp-draft 3 --qwen35-mtp-window`;
HTTP/stdio scheduling is not implemented. Draft state is independent, all
emitted tokens come from exact target verification, and sampler RNG advances
only for consumed target logits. The draft starts at generation rather than
replaying the prompt. A three-step independent llama.cpp NextN oracle checks
the post-output-norm hidden-input contract (matching top tokens; relative L2
0.01254/0.01218/0.01512), not bitwise draft-logit parity.

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
parity remain open.  DFlash2 separately meets its 60 tok/s short-context and
40 tok/s random-depth goals.
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
