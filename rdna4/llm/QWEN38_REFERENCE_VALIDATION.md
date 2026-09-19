# Qwen3.8 reference validation and RDNA4 optimization

The reference is llama.cpp `1859b520910af6f682256fd7299797774111a27a`, built
as one HIP backend for gfx1201 with ROCm 10. The build script exports the clean
checkout and fixes two C++ expressions in its C CPU header. It records that
patch and SHA-256 hashes for the executable, libraries and CMake configuration.
The runner does not link libllama; its sampler and kernels are independent ports.

## Controls

- `--sampling-profile llama`: penalties, top-k, top-p, min-p, temperature,
  then the pinned MT19937/distribution sampler. Temperature zero uses stable
  lowest-ID greedy selection. Defaults are seed 42, top-k 20, top-p 0.95,
  min-p 0, neutral penalties, a 64-token history and temperature zero.
- `--seed`, `--temp`, `--top-k`, `--top-p`, `--min-p`, `--repeat-penalty`,
  `--frequency-penalty`, `--presence-penalty`, `--penalty-last-n` select this
  profile too. Top-k <= 0 disables its filter; there is no legacy 64-token cap.
- HTTP requests with `seed` use the versioned `REQ2` protocol and reset the
  sampler per request. Prefix-cache hits reconstruct penalty history from the
  full prompt. Existing unseeded requests retain their prior sampling behavior.
- `--qwen35-decode-graph`: position-dependent Q8/Q8 stores and reads permit
  graph replay across tokens. The explicit flag overrides the IQ3 launcher's
  old graph-disable default. Greedy generation can return a four-byte argmax.
- `--qwen35-native-q8-attn`: native Q8/Q8 **decode**, D=256, gfx1201. It avoids
  expanding the complete cache to F16. Split selection/reductions follow the
  pinned reference, including its measured occupancy of 11 blocks per SM;
  this port's own occupancy must not replace that value. Prefill is unchanged.
- `--qwen35-native-q8-prefill`: also use native Q8/Q8 for prefill. It follows
  the two-column reference's split schedule (occupancy 9) and uses separate
  prefill scratch so allocation growth cannot invalidate decode graphs.
- `--qwen35-native-q2k`: native Q2_K x Q8_1 scalar projections on gfx1201,
  with the reference quantizer and eight-warp reduction order. For 5120-column
  matrices with at least 5120 rows, one physical warp evaluates eight virtual
  reference warps per row; other shapes retain eight physical warps per row.
  This overrides the old diagnostic Q2_K environment switches.
- `--qwen35-native-mmvq`: also use native IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS
  and IQ3_S x Q8_1 scalar
  projections. It includes native Q2_K and preserves the reference reduction
  order while assigning eight independent IQ rows to each workgroup. It takes
  precedence over diagnostic SSM projection switches and the old fused IQ3
  float projection paths; other SSM fusion remains enabled.
- `--qwen35-reference-math`: diagnostic scalar RMSNorm, IQ3_S SSM projections,
  convolution/SiLU and SSM head normalization. This is incomplete whole-model
  arithmetic parity, forces scalar prefill, and cannot use BF16 prefill.
- `--trace-prefix PATH` writes prompt IDs, selected IDs (including EOS), raw
  output bytes (excluding EOS), and full F32 logits for each selection.

Normal generation stops on EOS and emits at most the requested token count.
It no longer prints the first token twice or performs the unused final
forward. `--bench-ignore-eos` explicitly selects a synthetic benchmark, which
the reference comparator rejects.

## Reproduce

From the repository root (all scratch stays under the repository):

```sh
make -C rdna4/llm HIPCC=/opt/rocm/core-10.0/bin/hipcc
bash rdna4/llm/build_llama_reference.sh
make -C rdna4/llm reference-sampler-test
make -C rdna4/llm reference-attention-test HIPCC=/opt/rocm/core-10.0/bin/hipcc
python3 rdna4/llm/test_reference_norm.py \
  --llama tmp/qwen38/reference-build/source --out tmp/qwen38/reference-norm
env LD_LIBRARY_PATH=/opt/rocm/core-10.0/lib tmp/qwen38/reference-norm/test

env PYTHONDONTWRITEBYTECODE=1 LD_LIBRARY_PATH=/opt/rocm/core-10.0/lib \
python3 rdna4/llm/validate_qwen38_reference.py \
  --model /mnt/nvme02/models/qwen38/27b/gsq/Qwen3.8-27B-GSQ-RCO-IQ2_XS.gguf \
  --out tmp/qwen38/new-iq2-validation --native-q8-prefill --native-mmvq
```

Repeat with IQ3_XXS and a fresh output directory. The validation command runs
both greedy and temperature-0.6/seed-42 sampling, Q8 K/Q8 V, 4096 input tokens,
512-token chunks, context 8192, and a 256-token generation ceiling. It checks
process exit status, complete token/EOS/byte traces and finite logits, then
runs three resident, uncached timing repetitions **without logit trace I/O**.
Repeat zero is cold; repetitions one and two are warm. Each timed response
must match its traced response. Model, prompt, binary and reference-library
hashes and exact commands are saved with the results. The 500/40 tok/s targets
are reported as separate gates; matching one C++ fixture is not general parity.
The default prompt is the built-in 4096-token C++ merge task, checked by
compiling and executing both responses. `--prompt PATH` accepts another prompt;
use `--cpp-merge` only if it requests the same function contract.
For runner-only iterations, `--reuse-reference PREVIOUS_OUTPUT_DIR` verifies
the previous model/prompt/build hashes and exact generation/timing commands,
then rechecks its complete reference traces and timing logs. Reused reference
timing is explicitly marked in the result; it is not a new measurement.

## Validated kernel and output results

- Sampler: **13,801,002** exact token/candidate/logit/probability comparisons
  against libllama, including sorting ties, disabled top-k, non-neutral
  penalties, clone/reset, multiple seeds and the 248,320-token vocabulary.
- Q8/Q8 attention: **39,536,640** bitwise comparisons against the actual HIP
  reference kernels, including zero inputs, 127/128/129, 255/256/257,
  511/512/513, 4096/4097/8192, 1..32 splits and device-side split selection.
  Includes prefill batches of 2, 7 and 512 queries against the two-column
  reference kernel, with causal masking and nonzero prefix positions.
- RMSNorm: **62,145,280** bitwise comparisons, weighted and unweighted.
- Convolution/SiLU: **5,712,768** bitwise comparisons with recurrent history.
- Q2_K: **402,432** bitwise activation-quantization checks and **95,364**
  bitwise matrix-vector output checks, including zero and rounding-boundary
  inputs and the model's 5120/6144/17408-column shapes. Reproduce with
  `test_reference_q2k.py --llama tmp/qwen38/reference-build/source --out tmp/qwen38/reference-q2k`,
  then run `tmp/qwen38/reference-q2k/test` with the ROCm library path.
  The expanded test also covers IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS and IQ3_S with
  32/64/128/256-thread launches and both Q2_K schedules:
  **2,515,200** activation checks and **3,950,820** output comparisons
  across all six types, including 17408-row Q2_K projections.
  Representative 5120x17408 IQ2_S/IQ3_XXS kernels
  take 46.3/53.6 microseconds versus 88.7/93.2 for the pinned reference.
  IQ2_XXS/IQ2_XS take 49.3/52.6 microseconds versus 88.0/88.0; test log:
  `tmp/qwen38/reference-iq2xxs-xs-test.log`.
- Graph replay before changing attention: both IQ2 and IQ3 match uncaptured
  logits bitwise across all 155 selections of the C++ response.
- Native attention: both models emit the same 560-byte greedy C++ response as
  the reference, SHA-256 `4a0cb461966fae9a9d9da3b73c1b0c686ce8ee9ac3895c228bc6a653bc99a354`.
  All four implementations pass edge cases and 10,000 randomized cases under
  ASan/UBSan with warnings treated as errors.

The native-attention traced decode measurements are IQ2 **25.35** versus
**20.03** tok/s before this kernel, and IQ3 **25.59** versus **20.28** tok/s.
These are trace-enabled comparisons; use the separate timing results for
performance reporting.

Native prefill and decode attention, with BF16 projections, achieve:

| Model / sampling | Warm prefill tok/s | Warm decode tok/s |
|---|---:|---:|
| IQ2_XS greedy | 551.70–552.29 | 25.86–25.87 |
| IQ2_XS sampled | 551.73–551.86 | 25.63–25.66 |
| IQ3_XXS greedy | 574.91–575.14 | 26.02–26.04 |
| IQ3_XXS sampled | 573.97–574.20 | 25.82–25.88 |

Artifacts: `tmp/qwen38/final-iq2-native-prefill/` and
`tmp/qwen38/final-iq3-native-prefill/`. Every sampled and greedy response
matches its reference's complete token IDs, raw bytes and EOS; all warm
repetitions reproduce those bytes. Sampled uses temperature 0.6, top-k 20,
top-p 0.95, min-p 0, neutral penalties and seed 42. These measurements do not
enable the separate native Q2_K option.

Before extending native MMVQ to IQ2_XXS/IQ2_XS and tuning Q2_K scheduling,
the warm measurements were:

| Model / sampling | Prefill tok/s | Decode tok/s |
|---|---:|---:|
| IQ2_XS greedy | 551.73–552.31 | 33.25–33.27 |
| IQ2_XS sampled | 551.25–551.64 | 32.80–32.85 |
| IQ3_XXS greedy | 572.40–573.10 | 30.77–30.78 |
| IQ3_XXS sampled | 572.96–573.28 | 30.43–30.48 |

All four complete C++ responses remain token/byte/EOS-identical to the
reference, compile with warnings as errors, and pass ASan/UBSan, fixed edges
and 10,000 randomized cases each. All three timed repetitions per mode match
the traced output. Final artifacts: `tmp/qwen38/final-iq2-native-mmvq-v2/`
and `tmp/qwen38/final-iq3-native-mmvq-v2/`. IQ2 reuses the unchanged pinned
reference from `final-iq2-native-q2k/`; IQ3 reruns the reference. Greedy
full-logit relative L2 is still 0.02710 (IQ2) and 0.03119 (IQ3).

The same final configuration also passes an early-context retrieval check:
an exact 4096-token prompt places `ZEPHYR-7319` on its first line and asks for
it after the intervening context. Both models and both backends emit exactly
those eleven bytes, with identical selected IDs and EOS. Artifacts and commands
are in `tmp/qwen38/final-native-retrieval/`. The final v4 runner also passes
against those audited reference traces; its results and commands are in
`tmp/qwen38/final-native-retrieval-v4/`. This covers retention across all
eight prefill chunks in addition to the C++ task at the prompt tail.

Q2_K alone raises IQ2 decode to 30.9 tok/s, but slightly slows IQ3's existing
Q2_K path. Use the combined MMVQ option for the reported final configuration;
the isolated option remains available for differential experiments.

The final configuration also ports IQ2_XXS and IQ2_XS to the precise native
module and groups Q2_K rows for tall 5120-column projections. The latter
preserves the reference's eight partial sums per row before reduction;
17408-column down projections retain the original schedule. A same-run
17408x5120 microbenchmark improved from 71.5 to 55.6 microseconds.

Final warm measurements (all Q8 K/Q8 V, 4096/512, context 8192):

| Model / sampling | Prefill tok/s | Decode tok/s |
|---|---:|---:|
| IQ2_XS greedy | 552.66–553.28 | 33.80–33.82 |
| IQ2_XS sampled | 551.56–551.81 | 33.40–33.42 |
| IQ3_XXS greedy | 575.37–575.62 | 32.07–32.08 |
| IQ3_XXS sampled | 574.77–575.13 | 31.78–31.79 |

All four complete outputs still match the pinned reference's tokens, bytes
and EOS, pass the C++ execution tests, and repeat identically in the three
uncached timing runs. Artifacts: `tmp/qwen38/final-iq2-native-mmvq-v4/` and
`tmp/qwen38/final-iq3-native-mmvq-v4/`; audited reference runs are reused from
`final-iq2-native-q2k/` and `final-iq3-native-mmvq-v2/`, respectively.
Those reference warm greedy runs measured 292.5/27.2–27.3 tok/s for IQ2 and
420.0–420.3/25.1–25.3 tok/s for IQ3 (prefill/decode).

The Q2_K scheduling change also preserves **every logged full-model logit**
against v3 for all four C++ runs (648 selections total); see each output
directory's `logits-vs-v3-*.json`. This comparison is against the previous
runner, not whole-model llama.cpp numerical parity. Final greedy relative L2
against llama.cpp is 0.02754 (IQ2) and 0.03641 (IQ3); sampled is 0.02548 and
0.04316. The IQ3 greedy error increased from v2's 0.03119 when enabling the
exact IQ2 ports. Local operator parity alone does not close the coupled
model's error, although these complete tested responses remain identical.

## Outstanding requirements

The 500 tok/s warm prefill target is achieved with BF16 projections. The
40 tok/s decode and 60 tok/s MTP targets have **not** been achieved.
Complete model logits still differ from llama.cpp, and BF16
prefill remains an explicitly approximate projection path. Exact native MMQ
prefill, the remaining coupled SSM/projection differences and broader prompt
coverage are required before claiming general byte-identical generation.

Dense Qwen3.8-27B NextN/MTP is still unimplemented. Its sidecar has its own
embedding/output head and one dense block; the existing Qwen4 MoE/HC MTP path
cannot simply be enabled. The pinned Qwen3.5 graph exports NextN hidden input
**after final output normalization**. Transactional target state, KV and RNG
rollback plus exact verification are required before any MTP throughput claim.
