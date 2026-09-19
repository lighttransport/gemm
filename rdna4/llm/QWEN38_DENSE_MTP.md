# Dense Qwen3.8 NextN on RDNA4

This is a benchmark-runner and C API implementation for the dense Qwen3.5/3.8
architecture. It does not use the Qwen4 MoE/HC speculative path. HTTP/stdio
serving does not yet schedule these dense speculative windows.

The target retains Q8 K **and** Q8 V. The draft sidecar owns its fusion weights,
one attention/FFN block, output head and F16 KV. Its embedding stays mapped on
the CPU; only the selected embedding row is uploaded. The initial draft cache
starts at the first generated token, rather than replaying the prompt. Target
hidden states still carry the complete prompt context.

## Run

From the repository root, with ROCm 10 available:

```sh
mkdir -p tmp/qwen38/hiprtc-cache
TMPDIR="$PWD/tmp/qwen38/hiprtc-cache" make -C rdna4/llm test_hip_llm -j2
TMPDIR="$PWD/tmp/qwen38/hiprtc-cache" \
QWEN38_RUNNER_BIN="$PWD/rdna4/llm/test_hip_llm" \
QWEN38_MODEL=/mnt/nvme02/models/qwen38/27b/gsq/Qwen3.8-27B-GSQ-RCO-IQ2_XS.gguf \
bash rdna4/llm/run_qwen38_gsq_rocm.sh --gpu-only-bench \
  --prompt-file tmp/qwen38/final-iq2-native-mmvq-v4/prompt-4096.txt \
  -n 4096 -s 8192 --ubatch 512 --kv-cache q8q8 \
  --qwen35-prefill-bf16 --qwen35-decode-graph \
  --qwen35-native-q8-prefill --qwen35-native-mmvq \
  --sampling-profile llama --temp 0 --seed 42 --decode 256 \
  --qwen35-mtp /mnt/nvme02/models/qwen38/27b/mtp-Qwen3.8-27B-Q4_0.gguf \
  --qwen35-mtp-draft 3 --qwen35-mtp-window --bench-repeat 3
```

Omit the MTP arguments to measure ordinary decode. Omitting just
`--qwen35-mtp-window` provides a sequential target-verification diagnostic; it
does not accelerate generation. Window width is draft count plus one, up to
16 rows. Small windows use shared IQ grid decoding. Unsupported projection
formats retain their scalar math. The first verification allocates fixed
capacity; subsequent calls can use smaller windows.

## Verification contract

NextN consumes the **post-output-norm** hidden state, normalized token
embedding and normalized hidden state, concatenated in that order. Each
proposal is greedy. It neither advances the target state nor consumes target
sampler random numbers.

The target verifies `[anchor, draft_0, ..., draft_K-1]` using scalar-equivalent
projection, attention, normalization and recurrent arithmetic. Projections
are grouped across rows for weight reuse. Recurrent operations remain ordered
within each layer; convolution and recurrent states are checkpointed after
every row. Commit restores the accepted input prefix and its normalized hidden
state. Future KV entries remain masked until overwritten. When all drafts are
accepted, the bonus token's missing draft-cache entry is filled before the
next proposal.

The caller samples each consumed target logit row exactly once with its
ordinary sampler. A different sampled token terminates the window immediately.
This preserves the target's random-number sequence, including temperature,
penalties, EOS and length stops; no unverified draft is emitted. The API requires
commit before another proposal or target forward. Reset is required after an
execution error. Reset also invalidates draft history for a new request.

For four rows, checkpointing costs about 604 MiB, in addition to roughly
816 MiB for the draft and work buffers on the tested 27B model. Large draft
widths require more VRAM and are not implied performance recommendations.

## Reproduce correctness checks

`validate_qwen38_reference.py` supports `--mtp SIDECAR --mtp-draft N`. It checks
the model/sidecar/binary hashes, complete greedy and sampled token streams,
EOS, raw bytes, compiled C++ behavior, and uncached timing repetitions against
the pinned reference. To compare MTP directly with ordinary target logits:

```sh
python3 rdna4/llm/compare_generation.py MTP_TRACE TARGET_TRACE --require-logits
```

The operator checks use actual kernels from the private pinned llama.cpp tree:

```sh
python3 rdna4/llm/test_reference_q2k.py \
  --llama tmp/qwen38/reference-build/source --out tmp/qwen38/test-native-iq
LD_LIBRARY_PATH=/opt/rocm/core-10.0/lib tmp/qwen38/test-native-iq/test
python3 rdna4/llm/test_qwen35_ssm_prep.py --out tmp/qwen38/test-ssm-prep
LD_LIBRARY_PATH=/opt/rocm/core-10.0/lib tmp/qwen38/test-ssm-prep/test
```

`LLM_QWEN35_MTP_TRACE=PREFIX` is a diagnostic that writes the first three draft
inputs and logits. `llama_nextn_reference.cpp`, linked separately against the
private reference build, feeds those exact hidden inputs and tokens into the
reference's MTP context. It is not linked into the production runner. The first
three captured steps matched reference top tokens with relative logit L2
errors 0.01254, 0.01218 and 0.01512. This checks the draft contract; it does not
claim bitwise NextN logit equivalence.

Full-model byte equivalence is established for the tested fixtures. The
target's BF16 prefill remains approximate, and its full logits still differ
from llama.cpp. Exact target verification prevents MTP from adding another
generation difference.

## Independent draft oracle build

```sh
c++ -O3 -std=c++17 -Wall -Wextra -Wpedantic -Werror \
  -Itmp/qwen38/reference-build/source/include \
  -Itmp/qwen38/reference-build/source/src \
  -Itmp/qwen38/reference-build/source/ggml/include \
  rdna4/llm/llama_nextn_reference.cpp \
  -Ltmp/qwen38/reference-build/build/bin \
  -Wl,-rpath,"$PWD/tmp/qwen38/reference-build/build/bin" \
  -Wl,-rpath-link,/opt/rocm/core-10.0/lib \
  -lllama -lggml -lggml-base -o tmp/qwen38/llama_nextn_reference
LD_LIBRARY_PATH=/opt/rocm/core-10.0/lib \
  tmp/qwen38/llama_nextn_reference \
  /mnt/nvme02/models/qwen38/27b/mtp-Qwen3.8-27B-Q4_0.gguf \
  tmp/qwen38/dense-nextn-oracle 3
```

The prefix must contain inputs previously captured with
`LLM_QWEN35_MTP_TRACE=tmp/qwen38/dense-nextn-oracle` on a runner invocation.

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

**The 40 tok/s decode and 60 tok/s MTP targets remain unmet.** Ordinary
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

Exact native operator checks cover 2,948,352 activation values, 9,951,984
matrix outputs, 4,528,128 fused SSM preparation values and 39,536,640 attention
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

After the timing matrix, the CLI's invalid-option error buffer was initialized
and the executable rebuilt. The release binary repeats IQ2 MTP retrieval with
bitwise-identical logits and rejects draft width 17 with a deterministic error.
`release-manifest.json` records this final binary hash; the timing manifests
retain the preceding binary hash. No inference math changed in that rebuild.

## Synthetic 64K depth follow-up

The 64K-offset benchmark and long-context attention scheduling are documented
in [QWEN38_64K_DECODE.md](QWEN38_64K_DECODE.md). The benchmark now processes
random tokens through the model, matching llama-bench depth semantics. The
older zero-cache MTP number is obsolete; MTP remains opt-in pending a new
random-depth measurement.
