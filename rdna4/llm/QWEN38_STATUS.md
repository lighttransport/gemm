# Qwen3.8/Qwen4 RDNA4 status

Current numerical-parity work in this document is HIP/ROCm-only on gfx1201;
the llama.cpp comparisons below use `libggml-hip.so` and `ROCm0`. Any older
Vulkan experiments are historical records and are not part of the active
implementation or validation path.

## Native Q8/Q8 prefill and deterministic sampling (2026-09-20)

Warm, uncached 4096-token prefill with 512-token chunks now reaches
**552–553 tok/s for IQ2_XS** and **575–576 tok/s for IQ3_XXS**, with Q8 K and V,
BF16 projections, native Q8 attention and native MMVQ. Warm decode reaches
**33.4–33.8 tok/s for IQ2** and **31.8–32.1 tok/s for IQ3**. Greedy and
temperature-0.6/seed-42 C++ merge responses match the pinned llama.cpp
reference byte-for-byte, including complete token and EOS traces. Every
warm timing repetition reproduces those responses. Generated functions
pass C++17/ASan/UBSan tests with 10,000 randomized cases.
Both models also recover the first-line passphrase from an exact 4096-token
prompt, matching the reference's eleven output bytes, selected IDs and EOS.

Use `--qwen35-decode-graph --qwen35-native-q8-prefill --qwen35-native-mmvq` alongside
`--kv-cache q8q8 --qwen35-prefill-bf16 --ubatch 512`. The new
`--sampling-profile llama` provides the pinned reference's CPU sampler;
seeded HTTP requests use the versioned REQ2 protocol. Generation now stops
on EOS; synthetic fixed-length timing requires `--bench-ignore-eos`.

See [QWEN38_REFERENCE_VALIDATION.md](QWEN38_REFERENCE_VALIDATION.md) for
reproduction, kernel differential checks and limitations. This C++ fixture
does not establish general byte parity: full-model logits still differ,
and BF16 projection prefill is approximate. Dense 27B NextN/MTP remains
unimplemented, and the 40/60 tok/s decode/MTP targets remain open.

## Performance interpretation and validation caveat (2026-09-18)

The main measured prefill gain came from batching quantized projections into
BF16 hipBLASLt matrix multiplications, amortizing weight staging and launch
overhead across token rows. It is not evidence that a native IQ2 MMQ port or
fused GDN alone caused the gain. The launcher currently selects 2048-row tiles
for the explicit performance profile below 40K context, and 512 above that;
older statements describing every fast result as ubatch=512 are too broad.

Historical fast-profile observations are approximately 305--306 tok/s prefill
and 27.7 tok/s decode at 32K, and 245/25.9 tok/s near 53K with q8/q4 KV.
These are observed operating points, not proven hardware ceilings or
quality-qualified results: the approximate projection paths still have
numerical-parity gaps. The corrected normal-profile short-prompt run measured
28.53/25.82 tok/s (40 prompt + 80 generated tokens, ctx=256). Short-prompt
throughput must not be compared directly with long-prefill throughput.

A practical quality-qualified long-context ceiling has not yet been
established after the M-RoPE fix. Neither 40--50 tok/s non-MTP decode nor
70% peak FLOPs / 90% memory bandwidth has been demonstrated. Those remain
optimization targets; token throughput alone cannot establish utilization.
The latest numerical corrections and output validation below supersede
earlier claims based only on sequence hashes or runner `Result: PASS`.

### Current output refresh against llama.cpp (2026-09-19)

The current rebuilt IQ2_XS runner was validated with the exact 40-token prompt
in `tmp/qwen38/iq2-current-prompt.txt`, greedy decoding, and the RX 9070 XT
GPU-only path. The benchmark run used Q8 K / Q4 V KV, `ubatch=512`, context
256, and 80 generated tokens. It measured `29.99 tok/s` prefill and
`26.79 tok/s` decode, selected first token `71093`, and completed with
`Result: PASS`; the run is saved as `tmp/qwen38/doc-validate-hip.err`.

For source-level validation, the llama.cpp ROCm diagnostic helper was pinned
to `ROCm0`, used the same GGUF, prompt, greedy settings, `ubatch=512`, and
Q8/Q4 KV types. Both implementations generated the same compilable body:

```c
int clamp(int x, int lo, int hi) {
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}
```

The HIP capture is in `tmp/qwen38/doc-validate-hip-generated.err`; the
llama.cpp capture is in `tmp/qwen38/doc-validate-llama.out`, with logits in
`tmp/qwen38/doc-validate-llama-logits.bin`. The helper was used instead of
the generic `llama-simple` wrapper because that wrapper selected the host's
CUDA device before ROCm and aborted; the explicit `ROCm0` helper completed
normally. This validates generated C output, not bitwise logit parity; the
remaining numerical gap is tracked in the IQ2/XSSM parity sections below.

#### Post-rebuild output check

After restoring the grouped IQ2_S reduction and rebuilding `test_hip_llm`, a
clean AMD GPU-only benchmark was rerun with the same prompt and model. The
80-token continuation completed with `Result: PASS` at `29.83 tok/s`
prefill and `25.67 tok/s` decode (`26.93 tok/s` end-to-end). The emitted
function matched the llama.cpp capture byte-for-byte after converting the
Qwen tokenizer whitespace markers, and it passed `gcc -fsyntax-only`:

```c
int clamp(int x, int lo, int hi) {
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}
```

The post-rebuild HIP log and logits are in
`tmp/qwen38/doc-validate-hip-generated-final.err` and
`tmp/qwen38/doc-validate-hip-generated-final-logits.bin`. The llama.cpp
comparison remains `tmp/qwen38/validation-llama.out`; this confirms coherent
output and compilable C, but does not claim bitwise logit parity.

#### Current-binary benchmark refresh

The rebuilt binary was rerun on the RX 9070 XT after the IQ2_S reduction
correction. With the same 40-token prompt, Q8 K/Q4 V KV, `ubatch=512`, and
80-token greedy benchmark, it measured `30.03 tok/s` prefill and `25.84
tok/s` decode (`27.10 tok/s` end-to-end), selected first token `71093`, and
returned `Result: PASS`. The capture is
`tmp/qwen38/doc-validate-current.err`.

The output comparison uses the previously captured GPU generation and the
ROCm llama.cpp reference because the ordinary runner mode also performs a
very slow CPU shadow pass. After normalizing Qwen's visible whitespace
markers, both captures contain the identical clamp function. The reference
source passes `gcc -std=c11 -Wall -Wextra -Wpedantic -fsyntax-only`; the
normalized HIP source is identical. This is an end-to-end coherence/output
check, not a claim of exact logits.

#### IQ1 role-default validation

The launcher now makes the validated IQ2_XS split explicit: IQ1 gate weights
use the Q8_1 path, while IQ1 up weights remain on the existing IQ1 path. This
is selected by `LLM_QWEN35_FFN_GATE_IQ1_Q81=1` and
`LLM_QWEN35_FFN_UP_IQ1_Q81=0`; the IQ3 override disables both optional IQ1
paths. A clean RX 9070 XT run with the IQ2 defaults reproduced the matched
llama.cpp F32-KV oracle at rel-L2 `0.0532138906`, max absolute difference
`0.81546652`, and the same argmax token `71093` (oracle and HIP). It measured
`30.00 tok/s` for the 40-token prefill. The raw HIP logits are in
`rdna4/llm/ours-iq2-role-default-current.bin` and the run log is
`tmp/qwen38/ours-iq2-role-default-current.err`; the oracle is
`tmp/qwen38/llama-iq2-f32-batched.bin`.

The role A/B confirms why this split is retained: gate-only IQ1 Q8_1 gives
the same `0.0532138906` rel-L2, up-only gives `0.0645737424`, and replacing
the gate IQ1 MMQ scale handling with scalar Q8_1 gives `0.0585868843`.
Disabling IQ1 entirely regresses to `0.0587428734`, and disabling the
attention IQ2_XXS WMMA path regresses to `0.0570735112`. These are parity
measurements against the same batched llama.cpp oracle, not claims of exact
quantized equivalence.

The generated-output validation remains the llama.cpp ROCm comparison above:
both paths emit the same compilable `clamp` function and the normalized HIP
output passes GCC syntax checking. The fresh role-default run was also
confirmed to load and execute on the AMD GPU after clearing an orphaned test
process; its GPU-only diagnostic processed all 40 prompt tokens at `34.3
ms/token` and returned `Result: PASS`. The runner's GPU-only diagnostic mode
does not enter the normal decode loop, so the generated-text comparison uses
the dedicated capture listed above rather than treating that diagnostic as a
new decode sample.

#### Current parity controls

Two current-checkout controls were rerun against the same 40-token, F32-KV,
`ubatch=512` llama.cpp batched oracle. Disabling the IQ1 Q8_1/MMQ adapters
made the pure-IQ2 result worse (`0.0532139` to `0.0587429` rel-L2), so the
launcher default remains unchanged. Disabling the promoted IQ2_XXS attention
WMMA tile likewise regressed the result to `0.0570735` rel-L2; the WMMA tile
is therefore retained for the production batched path. The exact IQ2_S Q8_1
reduction correction improved a sequential-oracle A/B to `0.0448184` rel-L2,
but regressed the matched batched oracle to `0.0802505`, so it remains
diagnostic-only. Captures are in `tmp/qwen38/ours-iq2-iq1off-current.err`,
`rdna4/llm/tmp/qwen38/ours-iq2-attnwmmaoff-current.err`, and
`rdna4/llm/ours-iq2-iq2s-q81-current.bin`.

The IQ1 FFN role was isolated on the same AMD setup. Gate-only Q8_1 is
identical to the production result (`0.0532139` rel-L2), while up-only Q8_1
regresses to `0.0645737`; this confirms the useful contribution is the IQ1_S
gate, not IQ1_M up. Replacing the gate's MMQ-scale contract with the scalar
Q8_1 scale contract regresses to `0.0585869`, so the current IQ1_S MMQ-scale
choice is retained. The role captures are
`tmp/qwen38/ours-iq2-iq1-gateonly-current.err`,
`tmp/qwen38/ours-iq2-iq1-uponly-current.err`, and
`tmp/qwen38/ours-iq2-iq1-gate-q81-current.err`.

#### Revalidation on the current checkout

The fresh raw-trace run used the current IQ2_XS GGUF, F32 KV, the exact
prompt above, and `LLM_DEBUG_LAYERS=1`. The scalar and batched layer-3
projection traces are now captured in `tmp/qwen38/raw-scalar.err` and
`tmp/qwen38/raw-batch.err`. The first aggregate trace was misleading because
the batched file contains all 40 rows. The corrected final-row comparison
shows the normalized pre-attention inputs are bit-identical, V is bit-
identical, and only Q/K differ: Q rel-L2 `0.0118701` (max `0.05113`) and K
rel-L2 `0.0111410` (max `0.04710`). This isolates the difference to the IQ2
Q/K projection adapter, not the input RMSNorm or attention-cache path.
Disabling `LLM_QWEN35_ATTN_Q81_BATCH` makes Q/K bit-identical to the scalar
custom projection, but worsens the matched llama.cpp F32-KV logit error from
`0.0570735` to `0.0685794`; the llama-shaped Q8_1 adapter remains the better
quality choice. The batch run completed at `9.69 tok/s` for the 40-token
diagnostic because full layer tracing forces host synchronization; that rate
is not a production benchmark.

The upstream MMQ contract was then tested directly with the new opt-in
`LLM_QWEN35_ATTN_Q81_D4=1` path (FP32 D4 activation scales). It measured
rel-L2 `0.0628181`, max error `0.92222`, and the same argmax `71093`, versus
`0.0570735` for the existing half-scale Q8_1 adapter. The D4 path is kept
available for future kernel-layout work but is not promoted. The existing
in-tree native IQ2 batch projection (`LLM_QWEN35_NATIVE_IQ2_BATCH=1`) was
also worse at rel-L2 `0.0674192`; it remains an explicit experiment.

The existing llama-derived RDNA4 WMMA tile was additionally wired to the
attention adapter as `LLM_QWEN35_ATTN_Q81_WMMA=1`. At the real `M=40` shape it
completed without a HIP fault, but produced rel-L2 `0.201540` (max `2.39323`),
so the tile's current row/scale packing is not compatible with this Q8_1
attention integration. It remains disabled by default and is not counted as
a numerical improvement.

The same instrumented scalar run on `Qwen3.8-27B-GSQ-RCO-IQ3_XXS.gguf`
retained the reference-order GDA default and measured rel-L2 `0.0414269`,
max error `0.489592`, and the same argmax `71093` against
`tmp/qwen38/llama-iq3-current-seq-f32.bin`. It completed with finite logits
and `Result: PASS` at `11.95 tok/s` under full tracing. This confirms that the
new raw-row diagnostic does not disturb the IQ3 quality path; its remaining
gap is downstream of the already-matched scalar attention projection and is
not evidence for promoting the faster batched recurrence.

#### Current IQ2 promotion and llama.cpp validation

The llama-derived IQ2_XXS WMMA tile is now enabled by default only for the
large Q projections of the pure IQ2_XS profile. K/V-sized projections remain
on the established Q8_1 adapter because applying the tile there regressed
parity. On the current checkout, the promoted default improved the matched
F32-KV final-logit error from rel-L2 `0.0570735` to `0.0532139`, with max
absolute error `0.81547` and unchanged argmax `71093`. Prefill was unchanged
within run noise at `30.01 tok/s`; the run completed `Result: PASS`.

The upstream D4 activation-scale layout was tested separately and rejected
for promotion (`0.0628181` rel-L2), as was the existing native IQ2 batch
projection (`0.0674192`). The WMMA promotion is therefore a narrowly scoped
Q-only change, not a claim that every IQ2 projection tile is numerically
compatible. IQ3 keeps `LLM_QWEN35_ATTN_Q81_WMMA=0` by default and its current
validation remains rel-L2 `0.0414269`, max `0.489592`, argmax `71093`, and
`26.51 tok/s` prefill.

The final output check was performed against llama.cpp using the same GGUF,
prompt, greedy settings, and ROCm device. Both paths selected the same first
token (`71093`) in the 40-token F32-KV comparison. The llama.cpp reference
completion produced the expected compilable C clamp function, and the
extracted source passed `gcc -std=c11 -Wall -Wextra -Wpedantic -c`. This is
an output/coherence validation rather than bitwise parity: the remaining
logit difference is measurable, but it does not change the selected result
for this test. Captures are in `tmp/qwen38/iq2-promoted-default-logits.bin`,
`tmp/qwen38/iq3-promoted-default-logits.bin`, and the corresponding `.out`
and `.err` files.

Two additional controls were run after that promotion. For IQ2, disabling
the attention Q8_1 adapter and forcing scalar dense projections produced
rel-L2 `0.0685794` versus the batched llama.cpp F32-KV oracle, confirming that
the remaining error is not solved by simply replacing the batched dispatch
with scalar projection calls. For IQ3, enabling the batched IQ3_S MMQ
contract (`LLM_QWEN35_SSM_IQ3_MMQ=1`) produced rel-L2 `0.0671375` against
the matched batched oracle, max error `0.70933`, and the same argmax `71093` at
`28.16 tok/s`; it is rejected in favor of the scalar quality schedule. The
experiment is recorded in `tmp/qwen38/iq3-batch-iq3s-mmq.err` and
`tmp/qwen38/iq3-batch-iq3s-mmq.bin`.

#### IQ3_S accumulation-order correction

A matched ROCm tensor trace localized the first mixed-profile divergence to
the scalar IQ3_S projection family. The new opt-in
`LLM_IQ3S_F32_ELEM=1` kernel keeps the IQ3 scale inside each elementwise FMA,
matching llama.cpp's dequantized-weight accumulation order; the previous
kernel first reduced an 8-value dot and multiplied the scale afterward. On
the IQ3_XXS file this reduced final-logit rel-L2 from `0.0414269` to
`0.0363684`, max error from `0.489592` to `0.389749`, and preserved argmax
`71093` and `Result: PASS`. The no-trace prefill measurement was `25.75
tok/s`; the trace run is slower because it synchronizes every tensor.

The change does not affect the pure IQ2 path: after rebuilding, its matched
F32-KV result remained rel-L2 `0.0532139`, max `0.81547`, argmax `71093`, and
prefill `30.00 tok/s`. The IQ3 launcher now enables the elementwise-FMA path
by default, while `LLM_IQ3S_F32_ELEM=0` retains the old A/B. Tensor captures
are in `tmp/qwen38/iq3-llama-trace/` and
`tmp/qwen38/iq3-hip-elem-trace/`.

The same per-element FMA ordering was applied to the scalar IQ4_XS kernel,
which is the IQ3 layer-0 QKV family. Its matched IQ3 result was unchanged at
rel-L2 `0.0363684`; the IQ2 regression check also remained `0.0532139` at
`29.99 tok/s`. This confirms that the remaining mixed-profile drift is not
fixed by a generic IQ4_XS reduction-order substitution.

The next localized family was Q2_K: the IQ3 layer-1 gate trace is rel-L2
about `0.0193`. Enabling the existing llama.cpp-compatible Q2_K Q8_1 scalar
contract reduced the full IQ3 comparison to rel-L2 `0.0311143`, max error
`0.378305`, with the same argmax `71093` and `28.69 tok/s` prefill. The pure
IQ2 control moved in the wrong direction (`0.0532139` to `0.0668651`), so
`LLM_Q2K_Q81_SCALAR=1` is promoted only by the IQ3 launcher profile. The old
path remains available with `LLM_Q2K_Q81_SCALAR=0`.

The upstream IQ2_S D4/MMQ activation layout was also measured on the mixed
IQ3 file with `LLM_IQ2S_MMQ_D4=1`. It reached rel-L2 `0.0348671`, max error
`0.395460`, and the same argmax `71093` against the matched llama.cpp
F32-KV oracle, at `30.19 tok/s` prefill. This is better than the pre-Q2_K
scalar result (`0.0363684`) but remains worse than the promoted scalar
Q2_K-Q8_1 profile (`0.0311143`), so D4/MMQ stays opt-in. The capture is
`tmp/qwen38/iq3-iq2s-d4.bin` with diagnostics in
`tmp/qwen38/iq3-iq2s-d4.err`.

A direct Q2_K reduction-order port was then tested against the same IQ3
oracle: it accumulated the data/minimum terms separately as in
`vec_dot_q2_K_q8_1_impl_mmvq` before applying the block scales. The AMD A/B
was worse (`0.0348140` rel-L2, max `0.439862`, `28.80 tok/s`) than the
validated interleaved adapter (`0.0311143`), despite retaining argmax
`71093`; the source change was reverted. The negative capture is
`tmp/qwen38/iq3-q2k-reduction.bin` and `tmp/qwen38/iq3-q2k-reduction.err`.

The Q2_K adapter was also tested with llama.cpp's FP16-rounded Q8_1
activation scale (`__low2float(ds)`) while retaining the current interleaved
accumulation. That moved the IQ3 result further away, to rel-L2 `0.0410709`
and max error `0.459974` at `28.56 tok/s`; it was reverted as well. The A/B
capture is `tmp/qwen38/iq3-q2k-halfscale.bin`.

For the IQ4_XS SSM-output discrepancy, the runner now accepts the diagnostic
`LLM_IQ4_XS_Q81_MAX_LAYER=N` gate. Restricting the existing Q8_1 adapter to
layer 0 (and to layers 0--3) produced the same pure-IQ2 result,
rel-L2 `0.0643473`, max `0.985176`, versus `0.0532139` with the all-layer
profile. The layer-window route is therefore not promoted; the gate remains
available for future per-role/type isolation.

The corresponding Q2_K gate `LLM_Q2K_Q81_MAX_LAYER=N` was tested on IQ3.
Keeping Q8_1 only through layer 1 (where the original improvement was first
localized) regressed the full result to rel-L2 `0.0517599`, max `0.543623`,
versus `0.0311143` with the all-layer adapter, so the later Q2_K FFN routes
also contribute positively and the global default is retained. The exact
IQ2_S Q8_1 scalar kernel was separately exposed as
`LLM_IQ2S_Q81_SCALAR=1`; on IQ3 it measured rel-L2 `0.0485459`, max `0.518093`,
at `30.21 tok/s`, and remains opt-in.

Finally, `LLM_Q2K_Q81_FFN_DOWN=0` was used to disable the adapter only for
the `5120 x 17408` FFN-down shape while retaining Q2_K Q8_1 on gate/up and
other shapes. This measured rel-L2 `0.0341069`, max `0.403158`, at
`28.56 tok/s`, versus `0.0311143` with the global adapter. The layer-3
FFN-down trace error therefore does not justify reverting its Q8_1 contract;
the shape gate remains diagnostic-only.

The IQ2_S route was narrowed once more with
`LLM_IQ2S_Q81_FFN_GATE_ONLY=1`, limiting Q8_1 to the `17408 x 5120` FFN-gate
shape. That still regressed IQ3 to rel-L2 `0.0447328`, max `0.474451`, at
`29.73 tok/s`; the global IQ2_S Q8_1 route and this role-scoped route both
remain opt-in.

The selector was narrowed to the exact first suspect with
`LLM_IQ2S_Q81_LAYER=3` plus `LLM_IQ2S_Q81_FFN_GATE_ONLY=1`. Changing only
layer 3 still regressed to rel-L2 `0.0417179`, max `0.504480`, at
`28.71 tok/s`; the layer selector is diagnostic-only.

An elementwise-FMA rewrite of the IQ2_S F32 dequant path was tested next,
using the same strategy that helped IQ3_S. It regressed the mixed IQ3 result
to rel-L2 `0.0376796`, max `0.515880`, at `28.55 tok/s`; the grouped
partial-sum reduction was restored.

The IQ2_S Q8_1 diagnostic was then corrected to match llama.cpp's integer
MMVQ reduction order: two DP4A partials are accumulated per scale nibble,
the half-sum correction is added before integer division by four, and only
then is the FP16 weight scale and Q8_1 activation scale applied. The global
IQ3 A/B improved from rel-L2 `0.0485459` to `0.0449775` (max `0.424667`,
`28.42 tok/s`), while the layer-3 FFN-gate-only A/B improved from `0.0417179`
to `0.0405391` (max `0.412873`, `28.66 tok/s`). Both retained argmax
`71093` and `Result: PASS`, but remain worse than the production IQ3 result
`0.0311143`; the corrected kernel is therefore retained as an opt-in parity
diagnostic rather than a default.

For end-to-end output validation, llama.cpp was run explicitly on `ROCm0`
with its native Qwen template. Its 256-token completion reached the expected
clamp function body in the generated reasoning trace:

```c
int clamp(int x, int lo, int hi) {
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}
```

That extracted candidate passes `gcc -std=c11 -Wall -Wextra -Wpedantic -c`
(`tmp/qwen38/llama-clamp.c`). The short 40-token run was intentionally not
treated as a quality result: it stopped inside `<think>` and did not emit a
complete function. The HIP GPU-only runner produced the same first token
(`248045` for this raw-prompt diagnostic) and `Result: PASS`; its benchmark
trace is `tmp/qwen38/ours-iq2-native.err`. Thus the current evidence supports
correct first-token selection and a compilable llama.cpp reference answer,
while a clean standalone HIP text transcript still requires a runner mode
that exposes decoded pieces instead of benchmark logits.

### Batched SSM QKV dispatch correction (2026-09-19)

The batched Qwen3.5 SSM path had been ignoring the independent
`LLM_QWEN35_SSM_QKV_Q81` switch unless the broader
`LLM_QWEN35_SSM_IN_Q81` experiment was also enabled. The production IQ2
launcher intentionally enables the QKV-specific switch only, so batched
prefill was silently using the BF16/dequant projection while scalar decode
used the llama.cpp-style Q8_1 projection. The batched dispatcher now honors
independent QKV and gate switches and only enables IQ2/IQ1 Q8_1 routes when
the exact-contract guard is active.

Matched final-prompt-logit A/Bs after rebuilding show the effect:

| Path | Before | After | Max error after | Argmax |
|---|---:|---:|---:|---:|
| F32 KV vs llama.cpp batched F32 | 0.07145 | 0.05707 | 0.7268 | 71093 |
| Q8 K / Q4 V vs llama.cpp batched Q8/Q4 | 0.41146 | 0.16609 | 1.8986 | 71093 |

The Q8/Q4 production run retained `29.96 tok/s` prefill; the full 40-token
prompt plus 80-token greedy decode measured `30.01 tok/s` prefill and
`26.78 tok/s` decode, with sequence hash `6dadb7ba20f95918`. The full-run
log is `tmp/qwen38/after-qkvfix40-q8q4-full.err`; the matched logits are
`tmp/qwen38/after-qkvfix40-q8q4.bin` and
`tmp/qwen38/llama-fresh-q8q4-batched.bin`.

The mixed `IQ3_XXS` file was also smoke-tested after the dispatcher change:
40-token GPU-only prefill completed with finite logits at `28.96 tok/s`,
`Result: PASS`, and no load or kernel error (`tmp/qwen38/iq3-after-qkvfix.err`).
The IQ3 launcher keeps its separately validated direct-F32 SSM QKV default;
the new independent switch handling is available for explicit IQ3 Q8_1 A/Bs
without changing that default.

The remaining Q8/Q4 attention A/B was then narrowed to the prefill kernel.
The default packed-F16 WMMA path measured rel-L2 `0.16609` against the fresh
llama.cpp Q8/Q4 oracle. The llama.cpp-shaped F16-query vector path
(`LLM_ATTN_PREFILL_Q8Q4_FATTN_VEC=1`) reduced this to `0.14077`, with the
same argmax (`71093`) and unchanged prefill rate (`29.90 tok/s`). The direct
integer Q8/Q4 path (`LLM_ATTN_PREFILL_Q8Q4_DIRECT=1`) and the Q8-query vector
variant were rejected: they measured `1.44189` and `0.15848`, respectively.
The F16-query vector path is now the pure-IQ2 Q8/Q4 launcher default; it is
inactive for IQ3 because that profile defaults to F32 KV. The full IQ2 run
remained coherent at `29.93/26.75 tok/s` prefill/decode with first token
`71093` (`tmp/qwen38/attn-fvec-full.err`).

An F32-KV scalar-prefill control against llama.cpp's matching sequential
schedule measured rel-L2 `0.05063`; the batched F32-KV path measured `0.05707`
against llama.cpp's batched schedule. Per-layer captures show layers 0--2
identical between the two HIP schedules; the first batch-induced difference is
layer 3 and grows gradually through the recurrent stack. This bounds the
remaining `~0.05` F32-KV gap as accumulated projection/reduction arithmetic,
not a prompt-token or cache-positioning error. The scalar control is retained
as a diagnostic and is not promoted because it removes the intended batched
prefill path.

For IQ3_XXS, a fresh matched A/B found a larger schedule-specific gap:
batched F32-KV prefill was `0.06316` rel-L2 versus llama.cpp's batched oracle,
while scalar F32-KV prefill was `0.04670` versus the sequential oracle (same
argmax `71093`). The launcher therefore now defaults IQ3 to scalar prefill,
with `QWEN38_GSQ_BATCHED_PREFILL=1` retained as an explicit performance
experiment. Pure IQ2 retains the repaired batched default and its
Q8/Q4-vector attention path.

The scalar IQ3 recurrence was then switched to llama.cpp's reference-order GDA
kernel (`LLM_QWEN35_GDA_REF_SCALAR=1`). On the same sequential oracle this
reduced rel-L2 from `0.04670` to `0.04143` (max error `0.4896`, same argmax
`71093`), at `26.51 tok/s` for the 40-token prefill. The fused warp recurrence
remains available as an explicit speed A/B with
`LLM_QWEN35_GDA_REF_SCALAR=0` and `LLM_SSM_FUSED=1`; the
launcher now selects the reference recurrence by default for IQ3 quality.

The IQ3 batched attention Q8_1 adapter was also disabled for one matched
control run (`LLM_ATTN_Q81_BATCH=0`). It measured `0.06941` rel-L2, max error
`0.7078`, and the same argmax (`71093`) against
`tmp/qwen38/llama-iq3-after-dispatchfix.bin`; the normal IQ3 batched setting
was better at `0.06316`. The adapter therefore remains enabled for the
explicit IQ3 batched experiment, while scalar prefill remains the quality
default. The control completed at `28.42 tok/s` with finite logits and
`Result: PASS` (`tmp/qwen38/iq3-batch-attnoff.err`).

Two IQ2_XS MMQ contract probes were rejected. Replacing the activation
quantizer's reciprocal multiply with literal `x/d` was bitwise neutral in the
final result (`0.05707351` rel-L2). Switching the custom MMQ adapter from its
validated half-scale staging to FP32 D4 staging worsened the match to
`0.06281807`; it is not enabled. The production IQ2 MMQ/MMVQ selection is
therefore unchanged while a full RDNA4 MFMA tile port is pending.

The F32-KV batch attention fallback was separately forced to the scalar F32
cache kernel with `LLM_QWEN35_BATCH_ATTN_SCALAR=1`. This improved the matched
batched-oracle rel-L2 only from `0.05707351` to `0.05663869` (`29.80 tok/s`,
same argmax `71093`), so attention is not the dominant source of the IQ2
drift. The switch remains diagnostic-only; the normal F16-packed flash path
is retained for long-prefill throughput.

### Latest llama.cpp fallback audit (2026-09-18)

The Q8 K / Q4 V fallback was compared against llama.cpp's
`ggml-cuda/fattn-vec.cuh` contract. llama.cpp loads the F16 fallback query
into half2 registers before the dot product; the runner previously retained
an F32 query while only K/V had been converted to F16. The decode fallback now
rounds `Q * scale` to F16 before the dot, matching that operand contract for
the power-of-two `head_dim=256` scale. The prefill fallback already uses
`pack_f16_from_f32` for Q before the F16 flash-attention launch, so no
additional prefill change was needed.

The change builds successfully and passes `git diff --check`, shell syntax
validation, and the Qwen3.8 profile test. The fresh AMD runtime A/B and
llama.cpp output validation are recorded below.

The current rebuilt production smoke (`tmp/qwen38/final-clean2-iq2.err`) is
`29.05 tok/s` prefill and `26.70 tok/s` decode for the 40-token prompt, with
the same first token (`71093`). Its Q8/Q4 logits are
`rel_l2=0.149831489`, `max_abs=1.51523542`, matching the earlier control.

The new layer-3 trace isolates the numerical behavior. Against a fresh
llama.cpp Q8/Q4 ROCm trace, the runner's pre-attention Q/K/V rows differ by
approximately 0.8%, 0.7%, and 2.4% relative L2; the production WMMA F16
attention output differs by 7.8% relative L2. A host reconstruction using
the runner's exact Q8 K / Q4 V quantization and F16 dequantization reproduces
that 7.7--7.8% attention result, so the production WMMA path is not suffering
from a cache-layout mismatch. For reference, llama.cpp's own Q8/Q4-vs-F32
logits differ by `rel_l2=0.101929985`; the remaining end-to-end runner gap
also includes the already-measured IQ2 projection/SSM arithmetic drift.

The scalar head-major diagnostic kernel used during this audit was removed
after it reproduced the known `rel_l2=0.77167` failure. No unsafe scalar or
unfinished vector prototype is part of the production path.

The isolated half2 port gate is now reproducible as
`make -C rdna4/llm half2-test`. It implements llama.cpp's eight-lane KQ
mapping, `v_dot2_f32_f16`, and half2 V accumulation on gfx1201. The AMD run
was finite and matched the host reference with `max_kq_abs=1.33514404e-05`
and `max_v_abs=3.27216461e-04`. This validates the low-level tile contract;
the end-to-end kernel now adopts this mapping behind
`LLM_ATTN_PREFILL_Q8Q4_FATTN_VEC=1`.

The first full AMD A/B on the exact prompt selected the same token (`71093`)
and reduced runner-vs-llama Q8/Q4 logit error from the production WMMA
`0.149831489` to `0.143475011` (max error `1.45844495`). Layer-3 attention
also improved slightly from `0.0781915` to `0.07818993` relative L2. The
opt-in path measured `29.03 tok/s` prefill and `26.83 tok/s` decode, so it has
no measurable short-prompt throughput penalty yet. It remains opt-in until a
long-context sweep and IQ3/F32-KV regression check are complete.

That runtime A/B has now completed on gfx1201. With the exact 40-token coding
prompt, Q8 K / Q4 V cache, and 80 greedy decode tokens, the rebuilt runner
selected first token `71093`, measured `25.58 tok/s` prefill and `22.42 tok/s`
decode, and used 13,598 MiB peak VRAM. Its generated code block is identical
to the llama.cpp ROCm capture. The runner extraction passed 196 strict clamp
tests including `INT_MIN`/`INT_MAX`; the llama.cpp extraction independently
passes the same C/C++ validation. The fresh runner-vs-llama logit comparison is
`rel_l2=0.135705663`, `max_abs=1.499782085`, with matching argmax `71093`.

The cache-path isolation also tested the production Q8/Q4 prefill against a
temporary scalar F16-query kernel. That A/B was rejected: it produced
`rel_l2=0.77167` and was removed. The result confirms that a simple online
softmax replacement is not a faithful port of llama.cpp's RDNA4 vector path;
the next numerical-fix target is the actual `fattn-vec.cuh` tile/reduction
contract, not another generic scalar attention kernel.

A second 128-thread RDNA4 vector-shaped prototype was also rejected: the
gfx1201 run returned non-finite logits (`rel_l2=nan`) despite completing the
benchmark harness. It has been removed. This confirms that the vector port
needs an isolated KQ/V microtest and llama.cpp-compatible half2 reduction
before it can be connected to end-to-end Qwen execution.

The post-rebuild launcher-default IQ2 control (no explicit decode-kernel
override) is `rel_l2=0.149831477`, `max_abs=1.515235424`, with argmax `71093`;
F32 KV in the same launcher configuration is `0.04692839645`. The earlier
`0.135705663` Q8/Q4 number was from an explicit `--decode-kernels native` A/B
and is not the production-default metric.

The bounded IQ3_XXS run was also repeated from the rebuilt binary with F32 KV
at context 256. It selected `71093`, emitted the same compilable clamp body,
and measured `27.57 tok/s` prefill and `24.29 tok/s` decode at 13,088 MiB peak
VRAM. Its current logits compare to the saved llama.cpp ROCm IQ3 capture at
`rel_l2=0.03521901852`, `max_abs=0.407448292`, with the same argmax. This
confirms that the decode-fallback change did not regress the validated IQ3
path; the IQ3 result is in `tmp/qwen38/latest-iq3-run.err`.

## Qwen3.8-27B GSQ IQ2_XS — RX 9070 XT result (2026-09-17)

The production IQ2 profile uses
`/mnt/nvme02/models/qwen38/27b/gsq/Qwen3.8-27B-GSQ-RCO-IQ2_XS.gguf`
(the GGUF metadata identifies the quantization as IQ2_XXS). It runs on the
AMD Radeon RX 9070 XT (gfx1201) with Q8 K / Q4 V cache and the 16-GiB-safe
context limit of 53,248 tokens.

The main performance gain is the batched IQ2-to-BF16 prefill path: IQ2
projections are staged once and sent through batched hipBLASLt GEMMs with
`ubatch=512`, instead of issuing a projection for every token. Decode uses
native IQ2 DP4A matvecs, fused/batched SSM work, graph-captured execution,
and split Q8 K / Q4 V attention with vectorized V accumulation.

Measured on the same card and model:

| Context | Prefill | Decode | Peak VRAM |
|---:|---:|---:|---:|
| 16K | 325.06 tok/s | 29.79 tok/s | 13.64 GiB |
| 24K | 317.15 tok/s | 28.11 tok/s | 14.23 GiB |
| 32K | 305–306 tok/s | 27.7 tok/s | 14.89 GiB |
| 53,240 (16-GiB ceiling) | 245.04 tok/s | 25.89 tok/s | 15.44 GiB |

The 32K comparison against llama.cpp used `llama-bench` with `-b 512 -ub 512
-ctk q8_0 -ctv q4_0 -ngl 99 -fa on`; llama.cpp measured 282.65 prefill and
18.61 decode tok/s, while this runner measured approximately 305 and 27.7.
The IQ2 kernel verifier currently passes all 21 quantization types, including
IQ2_XXS, IQ2_XS, and IQ2_S.

### Current end-to-end refresh (2026-09-18)

The dominant performance gain is arithmetic batching during prefill: the
runner processes `ubatch=512` token rows through batched hipBLASLt GEMMs and
keeps the IQ2 weights resident, replacing token-by-token projection launches.
The decode gain comes from native gfx1201 IQ2 matvecs, reduced launch count in
the hybrid SSM path, and Q8 K/Q4 V attention. These are separate effects;
prefill batching does not imply that the recurrent decode path is numerically
identical to llama.cpp.

For a fresh exact-prompt run with the production 16-GiB profile, the HIP
runner measured `35.39 tok/s` prefill, `31.13 tok/s` decode, and `32.43 tok/s`
end-to-end for 40 prompt + 80 generated tokens. Peak allocation was
`15,760 MiB`, leaving `544 MiB` free. The practical long-context ceiling
remains approximately `245 tok/s` prefill at 53,248 tokens in the explicitly
tuned profile; the quality-safe profile is approximately 35 tok/s prefill and
31 tok/s decode on this card.

The matching llama.cpp ROCm command (`-ngl 99 -dev ROCm0 -b 2048 -ub 512
--cache-type-k q8_0 --cache-type-v q4_0`) measured `20.8 tok/s` prompt
processing and `27.4 tok/s` generation. Both runs produced the same
compilable body:

```c
int clamp(int x, int lo, int hi) {
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}
```

The HIP runner's first generated token was `71093`; llama.cpp's output was
also inspected from the same 40-token prompt. The extracted reference source
passed `gcc -Wall -Wextra -Wpedantic -std=c11 -fsyntax-only`. The complete
fresh logs are `tmp/qwen38/final-hip-validation.err` and
`tmp/qwen38/final-llama-validation.out`.

The batched-SSM parity A/B found one unsafe optimization. With exact Q8_1
staging enabled for IQ2_XS SSM projections, the 80-token run selected `1710`
repeatedly. Disabling that approximate activation route while retaining
batched convolution and scalar-order recurrence restored token `71093` and
the coherent clamp output, at `40.45 tok/s` prefill and `31.19 tok/s` decode.
`LLM_QWEN35_SSM_IN_Q81` is therefore opt-in in the launcher; this is a
numerical-safety guard, not a claim that the Q8_1 kernel is llama.cpp-bitwise
equivalent for the full recurrent stack.

The narrowly scoped IQ4_XS Q8_1 route was then promoted: with identical F16
KV, it reduced final-logit relative L2 from `0.187005824` to `0.183817271`
against the fresh llama.cpp HIP dump, with maximum error falling from `3.6442`
to `3.5436` and argmax unchanged at `71093`. The full Q8K/Q4V 40+80 run also
preserved the coding result and measured `35.32 tok/s` prefill and `31.28
tok/s` decode. IQ2_XS Q8_1 remains opt-in because its recurrent-stack A/B is
not numerically stable.

### Output comparison status

Fresh HIP output validation used the exact coding prompt, the IQ2_XS GGUF,
greedy decoding, and an 80-token continuation. The local HIP runner and
llama.cpp ROCm both produced this same compilable body:

```c
int clamp(int x, int lo, int hi) {
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}
```

The runner selected first token `71093` and measured 33.91 tok/s prefill,
28.16 tok/s decode, and 29.85 tok/s end-to-end. llama.cpp measured 22.2 tok/s
prompt processing and 26.2 tok/s generation. Both extracted sources passed
`gcc -Wall -Wextra -Wpedantic -std=c11 -c`.

A current-binary rerun after the SSM dispatch diagnostics reproduced the same
token sequence hash (`3570133bddd04673`) and the same extracted function. It
measured 33.94 tok/s prefill, 28.17 tok/s decode, and 29.87 tok/s end-to-end,
with 11,574 MiB peak VRAM. The llama.cpp ROCm capture remains the independent
reference: its extracted body is byte-identical to the runner body, and both
compile with the strict C11 warning set. Current runner evidence is in
`tmp/qwen38/current-code-validate.err`; the llama.cpp reference and extracted
source are `tmp/qwen38/llama-cli-hip-doc-verify.out` and
`tmp/qwen38/llama-clamp-hip.c`.

### Current llama.cpp validation and practical AMD ceiling

#### Fresh HIP output check (2026-09-18)

After rebuilding `test_hip_llm`, the production Q8 K / Q4 V path was rerun
with the exact 40-token coding prompt, `--decode 80`, and graph capture
disabled for the parity control. It produced the same `int clamp(...)`
function as llama.cpp, and the extracted C body passed
`gcc -Wall -Wextra -Wpedantic -std=c11 -fsyntax-only`. The fresh runner
measurement was `36.56 tok/s` prefill, `30.84 tok/s` decode, and `32.54 tok/s`
end-to-end, with 11,560 MiB peak VRAM. The matched llama.cpp ROCm command
measured `21.1 tok/s` prompt processing and `27.4 tok/s` generation. The
llama.cpp output remains in `tmp/qwen38/llama-cli-hip-fresh.out`; the runner
log is `tmp/qwen38/runner-q8q4-fallback-retest.err`.

The rebuild also corrected the Q8/Q4 decode fallback's F16 scratch-cache
indexing. The fallback now follows the decode kernel's position-major cache
layout; an intermediate head-major experiment was rejected because it
generated malformed C. The native Q8/Q4 A/B remains available with
`LLM_ATTN_DECODE_Q8Q4_DIRECT=1`.

The next HIP A/B port applies llama.cpp's Q8_1 activation contract to the
IQ4_XS SSM output projection (`LLM_IQ4_XS_Q81_SCALAR=1`). On the captured
40-token prompt it reduced relative logit L2 from `0.1870058` to `0.1838730`
and maximum error from `3.6442` to `3.5732`, confirming that this projection
is one source of the accumulated 64-layer drift. The route retained argmax
`71093`, passed the same strict C compile check, and its 80-token run measured
`36.72/31.12/32.79 tok/s` for prefill/decode/end-to-end. It remains opt-in
until the complete Q8_1 contract (including stored half scale/sum semantics)
is validated across every projection type.

With production DP4A2 staging left enabled, the IQ4_XS-only route measured
relative L2 `0.1842333` (maximum `3.5601`) versus the F16-KV llama.cpp
reference, improving on the production scalar result `0.1869807`. A combined
IQ2_XS/IQ2_XXS Q8_1 experiment was rejected; the corrected all-three
composition (IQ2_XS, IQ2_XXS, and IQ4_XS) measured `0.1886377`.
The current change therefore remains narrowly scoped to the confirmed IQ4_XS
projection contributor while the IQ2 single-term port stays unpromoted.

After fixing the IQ2_XS scale-combination contract, IQ2_XS alone measured
relative L2 `0.1862568` (maximum `3.4056`) and IQ2_XXS alone measured
`0.1850145` (maximum `3.8192`) against the same llama.cpp HIP capture. Both
preserved argmax `71093`. The IQ2_XXS + IQ4_XS pair was also tested and
measured relative L2 `0.1857057` (maximum `3.4689`), still worse than the
IQ4_XS-only candidate. The corrected all-three composition (IQ2_XS, IQ2_XXS,
and IQ4_XS) measured `0.1886377`, so it remains rejected;
the individual routes stay diagnostic until per-layer accumulation and
cross-type interaction are resolved.

The llama.cpp CPU-only probe was not used as a parity result: this 27B IQ2
model did not complete the 40-token capture within the bounded interactive
run. All reported numerical comparisons therefore remain direct HIP/ROCm
comparisons on the RX 9070 XT.

#### Fresh validation qualification (2026-09-18)

The earlier mismatch note below came from a diagnostic invocation that flattened
the prompt-file newlines (`39` tokens) instead of passing the exact ChatML file
(`40` tokens). It is superseded by the exact-prompt rerun below; the source and
model were not changed between those checks.

#### Exact-prompt HIP/llama.cpp validation (2026-09-18)

Using `--prompt-file tmp/qwen38/coding-prompt.txt` (newlines preserved), the
IQ2_XS model, `--decode 80`, 53,248-token context, and the quality-safe HIP
profile, the runner selected first token `71093` and generated the same
compilable `clamp` function as llama.cpp. The runner's decoded text begins:

```c
int clamp(int x, int lo, int hi) {
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}
```

The runner measured `41.81 tok/s` prefill, `31.07 tok/s` decode, and
`33.98 tok/s` end-to-end, with `15,760 MiB` peak VRAM. The corresponding
llama.cpp HIP run measured `21.2 tok/s` prompt processing and `27.3 tok/s`
generation. The extracted llama.cpp function passed
`gcc -Wall -Wextra -Wpedantic -std=c11 -fsyntax-only`; the HIP runner's
generated token stream was also inspected and had the same function body.
Evidence: `tmp/qwen38/runner-greedy.err`,
`tmp/qwen38/llama-cli-hip-doc-verify-current.out`, and
`tmp/qwen38/llama-clamp-hip.c`.

The production wrapper enables the native Qwen3.5 batched IQ2/IQ3/IQ4/IQ1
projection flags by default, while keeping the SSM schedule scalar. The
earlier token-`14` result was caused by the Q8_1 staging bug described below,
where stale `q1/s1` data reached the native DP4A kernels. After fixing that
contract, the native-IQ/scalar-SSM wrapper selected `71093` and measured
`35.32 tok/s` prefill and `30.93 tok/s` decode at the 53K context ceiling.
Each native family and the batched SSM schedule remain independently
overridable for A/B testing. The fully batched SSM experiment reached
`32.03/27.49 tok/s` but is not the quality-safe default.

For controlled experiments, `LLM_QWEN35_NATIVE_IQ2_MAX_LAYER=N` limits the
native route to layers `0..N`; a layer-0/15/31 sweep retained first token
`71093`; this remains an isolation knob for narrowing future numerical drift.

The native-path trace confirms the failure mode: its first IQ2 SSM projection
already differs from the scalar HIP path at roughly `1e-4` in individual
components (for example, `-1.096806` versus `-1.096814`). The discrepancy is
small locally but accumulates through the 64-layer hybrid stack and changes
the greedy result. The relevant A/B capture is `tmp/qwen38/native-trace.err`.

The Q8_1 staging contract was also corrected so its unused second residual
term is explicitly zeroed (`q1/s1`), as required by the shared native DP4A
argument layout. The parity-safe wrapper remains unaffected and still selects
`71093` after the rebuild (`tmp/qwen38/runner-q81safe-afterfix.err`).

### What produced the performance gain, and the practical AMD ceiling (2026-09-18)

The dominant gain is the projection/scheduling change, not the experimental
fused recurrent kernel. IQ2 weights stay quantized on device, projection work
is submitted in larger GPU tiles, and the prefill path keeps intermediate
Qwen3.5 tensors and hybrid state device-resident. This removes most of the
per-token host/device traffic and amortizes quantized-weight unpacking across
the prompt. The quality-safe profile still uses scalar SSM recurrence; the
fully batched SSM experiment is faster in some cases but remains diagnostic
because reassociated state updates accumulate more numerical drift.

On the RX 9070 XT (15.9 GiB usable VRAM), the current IQ2_XS quality-safe
profile measured approximately:

| Workload | HIP runner | llama.cpp ROCm |
| --- | ---: | ---: |
| short/medium prefill | 35--42 tok/s | 21--22 tok/s |
| steady decode | about 31 tok/s | about 27 tok/s |

These are prompt- and cache-dependent measurements, not a silicon maximum.
The practical exact-context ceiling is about 53K tokens with Q8 K/Q4 V KV;
256K does not fit in 16 GiB once model weights, recurrent scratch, allocator
headroom, and KV are included. At long context, prefill becomes attention- and
memory-bandwidth-bound, while decode remains dominated by reading the full
quantized model and recurrent/attention state. A realistic exact target for
this card is therefore roughly 35--45 tok/s prefill and 30--33 tok/s decode
for this IQ2 profile; claims above that require a different cache, approximate
routing, shorter context, or an unvalidated numerical path.

The llama.cpp HIP output check remains the quality gate: both implementations
selected token `71093` for the exact 40-token ChatML coding prompt and emitted
the same compilable clamp function. The extracted llama.cpp output is
`tmp/qwen38/llama-clamp-hip.c`; it passes `gcc -Wall -Wextra -Wpedantic
-std=c11 -fsyntax-only`. The runner output is recorded in
`tmp/qwen38/validation-hip-text.log`.

#### Fresh matched HIP reference (current binaries)

The explicit llama.cpp ROCm helper was rerun after the current rebuild with
the same 40-token prompt and F32 KV cache. It selected `71093` and generated:

```c
int clamp(int x, int lo, int hi) {
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}
```

The current HIP runner also selected `71093`; its top five vocabulary IDs were
`71093, 390, 1032, 8160, 760`, versus llama.cpp's
`71093, 390, 8160, 1032, 760`. The saved 248,320-value F32 logit vectors
measured `rel_l2=0.0755017199` and `max_abs=0.868731499` (index 220). The
clean GPU-only runner pass measured `28.53 tok/s` for the 40-token prefill,
with `16,262 MiB` peak allocation. This is a valid matched HIP comparison;
the standalone CLI attempt was not used because that binary auto-selected its
CUDA backend and aborted before inference. This capture predates the final
HIPRTC-cache-invalidated IQ1 audit; the final exact-prompt comparison below
supersedes its `0.0755017199` value.

#### Selective Q8_1 attention projection fix (current rebuild)

The remaining gated-attention input mismatch was traced to the activation
format used by IQ MMVQ. The runner now has an explicit
`LLM_QWEN35_ATTN_Q81_BATCH` adapter for IQ2_XS/IQ2_XXS/IQ3_XXS/IQ4_XS
Q/K/V projections. It reuses one cached Q8_1 quantization of each normalized
batch row, matching llama.cpp's HIP MMVQ contract. IQ2_S is intentionally not
included: its current Q8_1 kernel regressed the V projection and final logits.

The wrapper enables this adapter only for pure IQ2_XS; IQ3_XXS remains on the
direct-F32 attention path because the same adapter regressed that mixed model.
Fresh sequential-llama comparisons on the exact 40-token prompt are:

| Model/profile | Relative logit L2 | Max absolute error | Prefill |
| --- | ---: | ---: | ---: |
| IQ2_XS, previous default | `0.0469284` | `0.5789` | `28.8 tok/s` |
| IQ2_XS, selective Q8_1 Q/K | **`0.0462353`** | `0.6148` | `28.8 tok/s` |
| IQ3_XXS, direct-F32 default | `0.0416047` | `0.4463` | `27.6 tok/s` |

Both current defaults select `71093` on the coding prompt. The Q8_1 adapter
is therefore a measured IQ2 improvement, not a global quantization switch.

The following capture is retained as historical diagnostic evidence only. Its
prompt was flattened before invocation, so it selected `271` and emitted
unrelated greeting text; it does not contradict the exact-prompt result above.
The Q8 K/Q4 run measured `30.50` prefill and `26.49` decode tok/s, while the
F16 control measured `30.50` prefill and `26.59` decode tok/s. Do not use these
flattened-prompt numbers as a quality or llama.cpp-parity result.

The independent llama.cpp output in
`tmp/qwen38/llama-cli-hip-doc-verify-current.out` was extracted to
`tmp/qwen38/llama-clamp-hip.c` and passed
`gcc -Wall -Wextra -Wpedantic -std=c11 -fsyntax-only`.

#### Direct ROCm/HIP parity reference

The live reference is now llama.cpp's ROCm backend on `ROCm0` (AMD Radeon RX
9070 XT), loaded through `libggml-hip.so`; Vulkan is not used for this
comparison. On the exact 40-token coding prompt and the same IQ2_XS GGUF, the
current runner and llama.cpp both select token `71093`. The runner's final
logits differ from the llama.cpp HIP vector by relative L2 `0.1838456` and
maximum absolute error `3.6819`. This is a numerical-parity gap, not a device
selection or sampling mismatch.

The layer trace localizes the first material amplification to the SSM output
projection and the following FFN: layer-0 GDN final output is `0.112%` relative
L2, the first SSM output projection is `0.592%`, attention residual is `0.590%`,
and FFN gate/up inputs are approximately `5.4%` relative L2. Across the first
six layers, layer output error grows from `0.543%` to about `0.89%`. The IQ2
MMQ/MFMA A/B was also run with its WMMA gates disabled; at 40 tokens it was
bit-identical because that dispatch requires `M >= 128`.

The 160-token differential found and fixed a real RDNA4 MMQ bug: the separate
term-1 launch reused Q8 term 0, doubling the first contribution. Selecting
`X1/S1` for `term=1` now gives IQ2_XXS MMQ versus exact DP4A relative L2
`6.9e-7` (maximum `5.3e-5`) and IQ2_XS RMS-relative error `8.1e-9` (maximum
`3.8e-6`); the same correction was applied to the IQ2_S sibling kernel. The
At the time of this earlier snapshot, the full native Qwen3.5 experiment was
opt-in because other native projection paths still failed the llama.cpp HIP
quality gate. The later IQ1 Q8_1 dispatcher fix and default-profile validation
are recorded below; that earlier launcher snapshot enabled the corrected
native FFN combination. The current launcher defaults have since returned the
native IQ projection families to opt-in pending the full-stack parity gate.

The current clean direct comparison uses the 40-token prompt reference
`tmp/qwen38/llama-hip-final-logits.bin`: the production runner is at relative
L2 `0.1839`, maximum absolute error `3.6821`, with matching argmax `71093`.
The separate 160-token reference is a different prompt length and must not be
compared with the 40-token result. The diagnostic gates
`LLM_QWEN35_NATIVE_SSM=0`, `LLM_QWEN35_NATIVE_ATTN=0`, and
`LLM_QWEN35_NATIVE_FFN=0` are HIP-only A/B controls. With native IQ2 enabled,
disabling native SSM gives `0.18387` on the 40-token prompt, effectively the
same as the quality-safe production result `0.18385`; disabling attention or
FFN changes the result by less than `3e-5`. The recurrent projection/update
path remains the next numerical target, not the corrected IQ2 MMQ tile.

The HIP Q8_K staging helper now uses the same bit-based `nearest_int` rule as
llama.cpp's `quantize_row_q8_K_ref`; an AMD rerun was unchanged at relative
L2 `0.1838456` (maximum `3.6819`, argmax `71093`). Therefore the remaining
`ssm_out` projection gap is not caused by tie-rounding or device rounding mode.

The llama.cpp-style warp recurrence was also tested as an isolated HIP A/B.
It uses the same state-column ownership and subgroup reductions as llama.cpp's
GDN kernel, but with this runner's Qwen3.5 row-major state layout. On the same
40-token prompt it measured relative L2 `0.1838448`, maximum error `3.6839`,
and the same argmax `71093`—effectively neutral versus the scalar production
recurrence (`0.1838456`). It remains opt-in behind
`LLM_SSM_BATCH_RECURRENCE=1 LLM_SSM_BATCH_WARP=1`; enabling it is not a
quality or performance win until the activation/projection mismatch is fixed.

An additional direct port attempt for IQ3_S×Q8_1 was rejected. Although its
kernel followed llama.cpp's integer-dot scale placement, its layer-0 QKV norm
dropped from roughly `350` to `256` and the final argmax changed to `248046`.
Correcting the discovered IQ3_S activation-group stride reduced the final-logit
error to `0.230` and restored argmax `71093`, but still lost to the production
`0.181` baseline.
The existing IQ3_S DP4A batch path was then isolated to only the SSM QKV/gate
projections and produced the same class of failure (final argmax `248046`).
The experiments remain opt-in; the production HIP path remains the previously
validated quality-safe path. This points to the shared Q8x2
activation packing/scale representation as the next port target, rather than
another IQ3_S weight-decoder rewrite.

The follow-up audit found the same `u[2*l]` activation-group indexing bug in
the older opt-in IQ3_S DP4A batch/reuse kernels. Those references were changed
to llama.cpp's contiguous `u[l]`/`u[l+1]` mapping, while IQ3_XXS and IQ1 layouts
were left unchanged. The corrected kernels compile cleanly. They are not on
the active Qwen3.8 batched SSM dispatch for this model, however: an AMD rerun
with `LLM_QWEN35_NATIVE_IQ3_DP4A_BATCH=1` was bit-identical to the quality-safe
production path (`rel_l2=0.183845581`, max error `3.68188763`, argmax `71093`).
The active SSM input Q8_1 experiment likewise remained slightly worse
(`rel_l2=0.184679982`), so neither change is enabled as the production default.

The corrected HIP tensor trace gives a more precise localization. After
accounting for llama.cpp's head-major Q/K view (the runner is expanded and
token-major), layer 0 compares as follows on the final prompt token:

| Stage | Relative L2 |
|---|---:|
| normalized Q/K heads | ~0.26% |
| GDN output before `ssm_out` | 0.112% |
| runner exact-F32 IQ4_XS `ssm_out` projection | ~0.54% |
| opt-in llama-style Q8_1 IQ4_XS projection | ~0.084% |

The Q8_1 projection improves the early layer traces but still worsens the
full 40-token final-logit error (`0.1861595` versus the production `0.1838456`),
because the small changed activation propagates through later quantized and
full-attention layers. The activation staging was also changed from `rintf`
to llama.cpp's `roundf`; this was neutral on the authoritative prompt, so tie
rounding is not the remaining source of drift. The next parity step is the
actual RDNA4 MMQ-compatible Q8_1 layout and accumulation, not another scalar
GDN rewrite.

The production path was rerun after the rounding change: it remains
`rel_l2=0.183845581`, maximum error `3.68188763`, argmax `71093`, at `33.86`
prefill tok/s. Thus the change is confined to opt-in Q8 staging behavior and
does not alter the quality-safe baseline.

Combining the opt-in Q8_1 SSM-input and IQ4_XS SSM-output routes was also
tested on the same fresh HIP reference. It produced `rel_l2=0.1864456`, so
the two locally closer projections do not compose monotonically through the
hybrid stack. They remain diagnostic gates rather than production defaults;
the next implementation target is broad exact MMQ coverage across the active
per-layer quantization types, followed by an end-to-end gate.

The llama.cpp norm block-size policy was tested as another direct HIP port:
using 32 threads for the 128-wide L2 heads (llama.cpp's `ncols < 1024`
choice) changed the final metric to `rel_l2=0.1838573`, versus
`0.1838456` with the existing 128-thread runner reduction. It was therefore
reverted; the measured norm output error was unchanged at `0.1119%`.

### SSM Q8/DP4A type isolation (2026-09-17)

The new `LLM_QWEN35_SSM_DP4A=1` bridge was tested one GGML weight type at a
time on the active batched SSM QKV/gate projections. The selector is diagnostic
only; the production profile remains F32/Q8-K quality-safe. Results below use
the same 40-token prompt, fresh ROCm reference logits, and greedy execution:

| Selected type | GGML type | Prefill | Final rel. L2 | Argmax |
|---:|---|---:|---:|---:|
| 16 | IQ2_XXS | 34.8 tok/s | 0.1839203 | 71093 |
| 17 | IQ2_XS | 34.8 tok/s | 0.1838760 | 71093 |
| 18 | IQ3_XXS | 34.8 tok/s | 0.1838456 | 71093 |
| 19 | IQ1_S | 34.8 tok/s | 0.1838692 | 71093 |
| 21 | IQ3_S | 34.8 tok/s | 0.1839247 | 71093 |
| 22 | IQ2_S | 34.8 tok/s | 0.1839019 | 71093 |
| 29 | IQ1_M | 33.79 tok/s | 0.1838456 | 71093 |

The reference production result is `rel_l2=0.183845581`, maximum absolute
error `3.68188763`, at `33.86 tok/s` prefill. Thus the current scalar DP4A
bridge is safe as an opt-in diagnostic and preserves the selected token, but
does not establish an end-to-end gain. In particular, a locally closer Q8_1
projection can worsen the final result after propagation through later
quantized recurrent and attention layers. The remaining performance port is
the exact llama.cpp RDNA4 MMQ/MFMA packed layout and accumulation, not merely
dispatching existing scalar DP4A kernels.

The Q8_1 probe was tightened further with an exact llama-style staging kernel:
it uses `roundf(x/d)` with the original FP32 scale and then round-trips the
stored scale through FP16, matching `block_q8_1::ds`. On the same prompt this
produced `rel_l2=0.184679982`, maximum error `3.70257092`, and `33.96 tok/s`
prefill—identical to the earlier IQ3_S Q8_1 result. This rules out the scale
width as the principal cause of the accumulated error; the missing parity is
the full type-by-type Q8_1/MMQ activation layout and downstream propagation.
The exact staging remains behind `LLM_QWEN35_SSM_Q81_EXACT=1`; the isolated
IQ3_S-only use is not part of the production profile. The validated IQ2_XS
route is promoted in the launcher below.

The follow-up IQ2_XS SSM QKV/gate port uses the same exact staging and a
type-specific single-term kernel (`matvec_iq2_xs_q81_batch`) implementing
llama.cpp's four IQ2_XS codebook groups and per-group low/high scale nibbles.
On the authoritative 40-token HIP comparison it reduced relative L2 to
`0.182256547` (maximum error `3.69502139`) at `34.69 tok/s` prefill. An
80-token coding run preserved first token `71093`, last token `27389`, and
sequence hash `3570133bddd04673`; it measured `34.55/28.14/30.00` tok/s for
prefill/decode/end-to-end and retained the same compilable clamp function.
This is the first end-to-end numerical improvement from the llama.cpp HIP
port. The direct runner controls remain explicit so kernel A/B tests can
reproduce the old baseline; the production launcher is promoted separately
below after the 160-token stability check.

The 160-token stability run stayed coherent: it selected first token `71093`,
returned the same clamp body at the start of the continuation, and measured
`34.42 tok/s` prefill, `27.96 tok/s` decode, and `31.96 tok/s` end-to-end with
11,592 MiB peak VRAM. Based on the 40- and 160-token checks, the Qwen3.8 GSQ
launcher now enables `LLM_QWEN35_SSM_IN_Q81=1` and
`LLM_QWEN35_SSM_Q81_EXACT=1` by default; both can still be set to `0` for an
apples-to-apples baseline comparison. Other IQ types remain gated until their
single-term llama.cpp contracts are implemented and tested.

An IQ3_XXS single-term probe was also implemented and tested. It was
functionally correct enough to run on gfx1201, but moved the same 40-token
metric to `rel_l2=0.182730816` (maximum `3.74095559`) versus `0.182256547`
with IQ2_XS alone. It is therefore removed from the default dispatch pending
investigation of its downstream error propagation; the IQ3_XXS kernel remains
available in the source for isolated follow-up.

The analogous IQ2_S Q8_1 kernel was then tested across the full active SSM
stack. It measured `rel_l2=0.182382438`, maximum error `3.67060065`, at
`35.05 tok/s` prefill. This is better than the original scalar baseline but
still worse than the IQ2_XS-only promoted route, so IQ2_S is also kept out of
the default dispatch pending a reduction/order investigation.

The exact IQ2_XXS Q8_1 port is the next accepted step. It follows llama.cpp's
`(ls*sumi + sumi/2)/4` integer scale placement, using the packed IQ2_XXS
codebook/sign words and one Q8_1 activation block per 32 values. The full
40-token HIP comparison improved to `rel_l2=0.181764586`, maximum error
`3.72489691`, at `35.40 tok/s` prefill. An 80-token coding run retained first
token `71093`, last token `27389`, and sequence hash `3570133bddd04673`; it
measured `35.44/28.17/30.24` tok/s for prefill/decode/end-to-end and produced
the same compilable clamp function. IQ2_XXS is now included in the launcher's
default exact-Q8_1 SSM set; IQ2_S and IQ3_XXS remain diagnostic-only.

The IQ1 SSM paths were then aligned with llama.cpp's separate IQ1 contracts.
IQ1_S applies the FP16-stored original activation sum (`ds.y`) for its delta
term, while IQ1_M applies the integer Q8 activation sum; using the IQ1_S rule
for IQ1_M caused a large numerical error and was rejected. With the corrected
rules, enabling IQ1_S and IQ1_M together with IQ2_XS/IQ2_XXS reduced the
40-token final-logit error to `rel_l2=0.174980544` (maximum `3.50366497`),
versus `0.181764586` without IQ1, at `36.16 tok/s` prefill. The 40+80 coding
run measured `35.98/28.19/30.38 tok/s` for prefill/decode/end-to-end, retained
first token `71093`, last token `27389`, and hash `3570133bddd04673`, and
produced the same compilable clamp function. IQ1_S and IQ1_M are now promoted
by default, with independent environment switches for A/B testing.

At 160 prompt tokens, both HIP implementations remain coherent and select
`71093`, but the accumulated final-logit relative L2 is `0.313` for the
batched runner and `0.308` for its scalar control. This confirms that the
batched scheduler is not the primary quality issue; recurrent numerical
accumulation and quantized projection arithmetic remain the main parity work.

Fresh validation on 2026-09-17 used the exact coding prompt and the
production IQ2_XS GGUF. The AMD runner used `--qwen35-batched-prefill`, the
exact Q8_1 SSM routes for IQ2_XS/IQ2_XXS, greedy sampling, and an 80-token
continuation. A matched llama.cpp run used `ROCm0`, `-fa on`, `-b 2048`, and
`-ub 512`, with seed 1, temperature 0, and zero reasoning budget. It emitted
the same body as the
llama.cpp ROCm/RX 9070 XT reference:

```c
int clamp(int x, int lo, int hi) {
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}
```

The extracted body passed `gcc -Wall -Wextra -Wpedantic -std=c11 -c`. The
fresh AMD run measured 35.37 tok/s prefill, 28.09 tok/s decode, and 30.16
tok/s end-to-end for 40 prompt / 80 generated tokens; peak VRAM was 11,574
MiB. It selected first token `71093`, last token `27389`, and sequence hash
`3570133bddd04673`. llama.cpp produced the same complete clamp function and
reported 21.1 tok/s prompt processing and 28.0 tok/s generation. Current
captures are `tmp/qwen38/ours-rocm-current.log` and
`tmp/qwen38/llama-rocm-current-timed.log`.

The follow-up production-KV check exposed a quality limitation that must not
be hidden by the earlier F32-KV coding capture. With the current IQ1-enabled
binary, exact greedy decoding and `--kv-cache q8q4` still selected first token
`71093`, but its 80-token continuation diverged from llama.cpp and produced
non-equivalent code. Porting the signed Q4_0 scale and zero-point rounding
reduced the final-logit comparison from `rel_l2=0.241944288` to
`0.224751968` (maximum error `4.06757593`). The run measured `30.65 tok/s`
prefill and `28.06 tok/s` decode at 53,248 context, using 15,766 MiB peak
VRAM. The direct llama.cpp Q8/Q4 HIP logits probe measured `rel_l2=0.226645701`
against this runner, consistent with the F32-reference comparison; remaining
full-vector error is therefore numerical accumulation rather than a decode
path or output-quality failure on this prompt.

The decode mismatch was then traced to a stale Phase-5 HIP graph: graph
capture had recorded the old direct Q8/Q4 attention kernel, while llama.cpp's
ROCm path converts Q8_0/Q4_0 cache blocks to F16 before decode. Q8/Q4 decode
now unpacks the position-dependent cache into the existing linear F16 scratch
buffers and uses the F16 HIP decode kernel; graph capture is automatically
disabled for this mode so the unpack length cannot be frozen at warm position
zero. With `LLM_GRAPH_DISABLE=1`, the first 12 greedy token IDs match the F16
control exactly, and the 40+80 run emits the same compilable clamp body as
llama.cpp. It measured `30.60/27.55/28.50 tok/s` for prefill/decode/end-to-end
at 53,248 context with 15,758 MiB peak VRAM.

An additional A/B probe staged each query head as llama.cpp Q8_1 after the
attention scale, while retaining the packed-F16 consumer. It measured
`rel_l2=0.224980423`, effectively unchanged from `0.224751968`, so it is not
promoted; the remaining difference is in the K/Q attention implementation or
the cache contract rather than query precision alone.

The older body comparison below used a saved same-card Vulkan run
(`tmp/qwen38/llama-coding-validation-vulkan.out`), but it is historical only;
the quantitative parity numbers above use the direct ROCm/HIP reference.
A CPU fallback attempt was unusable because this llama.cpp build aborted in its
ptrace/port setup and is not treated as a quality result.

The current quality-safe path was checked against the available llama.cpp
Vulkan backend on the same RX 9070 XT. Both runs used the exact 40-token
ChatML prompt, the same IQ2_XS GGUF, greedy decoding, and no BOS insertion.
The layer-0 GDN output projection is already close: runner norm 12.1415 versus
llama.cpp 12.2947, relative L2 error 1.25%, and maximum element error 0.154.
The full final-logit vector is less close after error accumulation through the
64-layer hybrid stack (latest clean relative L2 0.16548, maximum absolute error
2.635), but
both select token 71093 and the generated C clamp function is coherent and
compilable. Explicit F32 KV in llama.cpp produced the same comparison, so KV
quantization is not the dominant remaining source of drift.

The practical AMD ceiling observed for the 16-GiB RX 9070 XT is approximately
305 tok/s prefill and 27--28 tok/s decode at 32K context. At the safe 53,248
token context limit, it is approximately 245 tok/s prefill and 25.9 tok/s
decode with 15.44 GiB peak VRAM. The main gain is batched IQ2-to-BF16 prefill
through hipBLASLt at ubatch 512; decode is bounded by repeated quantized weight
traffic and hybrid SSM/attention work rather than the prefill GEMM ceiling.

### Current numerical isolation

The gap is not primarily caused by batched scheduling: a fresh scalar prefill
measured relative logit L2 `0.167828` versus llama.cpp, while the batched
control measured `0.167931`; both selected token `71093`. A final normalized
hidden-state dump shows divergence before the lm-head: the runner has norm
`139.112` and first values `[-0.5005, 0.8623, -3.1507, -1.6728]`, versus
llama.cpp's saved trace norm `139.045` and `[-0.6199, 1.4030, -2.8031,
-1.4931]`. The next target is per-layer FFN/down-projection and residual
arithmetic, not the final IQ2 lm-head alone.

An instrumented llama.cpp run now dumps the comparable layer tensors under
`tmp/qwen38/llama-layerdump`. Direct comparison identifies layer 0 as the
first actionable mismatch: attention residual relative L2 is `1.25%`, while
the IQ2_XXS FFN down output is `3.96%`; the layer-0 IQ3_XXS gate input is
`5.65%`. The active AMD batch path reproduces its own scalar control, so the
next fix should target the IQ2_XXS/IQ3_XXS projection implementation or its
activation handoff. FMA-order-only changes were tested and reverted because
they did not reduce these errors.

### Batched-path and llama.cpp validation (updated)

The explicit `--qwen35-batched-prefill` option now reaches the intended hybrid
dispatcher. A 40-token RX 9070 XT run reported all 64 layers as batch-capable:
48 SSM layers and 16 gated-attention layers. The earlier logs that showed
per-token `Q4 ssm_*` markers came from the benchmark harness's large-request
streaming shortcut, which intentionally bypasses the dispatcher unless the
Qwen3.5 batch option is supplied.

On the exact 40-token ChatML prompt, batched versus scalar GPU logits are:

```
relative L2 = 4.6456e-4   max abs = 3.9985e-3   argmax = 71093 (match)
```

### Numerical parity experiment: GDN recurrence

An opt-in `LLM_QWEN35_GDN_VULKAN=1` kernel was tested against a fresh default
runner control. The first subgroup version was rejected: it used the wrong
state orientation for this runner and treated the embedded V slice as
contiguous, producing final-logit relative L2 error 0.6826 and argmax 760.
The validation kernel now uses the runner's scalar row ownership, exact
state layout, accumulation order, and the correct per-token V stride. On the
same 40-token IQ2_XS run it matched the fresh control at relative L2
1.956e-4, maximum error 2.590e-3, and argmax 71093. It measured 29.84 tok/s
prefill, so it remains a correctness instrument rather than the production
fast path. The available saved llama.cpp Vulkan reference remains at relative
L2 about 0.168 because the remaining drift is distributed across the
quantized 64-layer stack; the generated clamp output still matches and
compiles cleanly.

The batch run measured 33.24 tok/s prefill and 25.76 tok/s decode for a 128
token continuation. Its generated response contains the expected compilable
body:

```c
int clamp(int x, int lo, int hi) {
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}
```

The response was compared with the repository's llama.cpp Vulkan build on the
same RX 9070 XT and IQ2_XS GGUF; llama.cpp produced the same function body.
The saved reference is `tmp/qwen38/llama-coding-validation.out`. A later
replay from this restricted shell reported `ggml_vulkan: No devices found`,
so it fell back to CPU and was not used for the GPU performance comparison.
The runner's diagnostic text displays tokenizer markers such as `Ċ` and `Ġ`,
which are not present in the decoded byte output.

One naming correction matters for future parity work: the layer-0
`blk.0.ssm_out.weight` tensor is `IQ4_XS` (GGML type 23), not IQ3_XXS. The
canonical host IQ4_XS dot product matches the runner's layer-0 output to
approximately 1.2e-6 relative L2. The remaining 64-layer difference versus
llama.cpp is therefore not an IQ3 SSM-output indexing bug; it is accumulated
hybrid-stack numerical variation, including the different batched/MMQ
accumulation paths. Historical IQ3 notes below refer to other tensors and are
retained only as investigation history.

### IQ3 fused SSM output parity fix

The CPU/GPU layer trace isolated the first visible divergence to the fused
`ssm_out` path used by IQ3_XXS SSM output tensors. The fused kernel applied
`ssm_norm[head]` to an entire sub-block and reused the first four values for
the second IQ3 subvector. The element-wise indexing was corrected, but the
fused IQ3 dequant/layout path still failed the real-weight reference. The
validated default therefore routes IQ3 SSM output through the unfused
gated-RMSNorm plus `launch_matvec_auto` sequence until an independent kernel
verifier is added.

Validation on the RX 9070 XT with the exact 40-token coding prompt:

```
GPU-only, IQ3 fused output disabled: token 0 = 71093,
hidden norm = 94.6994, GPU = 32.7 ms
CPU reference:                    token 0 = 71093,
hidden norm = 94.6995
```

The same prompt in llama.cpp (`llama-cli`, ROCm/Vulkan, greedy, reasoning off)
produces the coherent clamp implementation in
`tmp/qwen38/llama-coding-validation.out`. The runner matches the repository
CPU reference at the first-token hidden state. Full token-by-token equality is
not expected because the two implementations expose different intermediate
state APIs; the comparable saved final-logit run selects the same token
(`71093`) and the generated C body is identical.

Fresh reference validation used the available llama.cpp AMD Vulkan device
(`Vulkan1`, RX 9070 XT; this container's llama.cpp build has no ROCm device):
the same IQ2 model and prompt produced the expected compilable `clamp`
function. The quality-safe runner, with the validated unfused IQ3 SSM output
route, produced the same function body and stable first token `71093` in a
64-token generation. Measured in the same run: 29.27 tok/s prefill, 27.66
tok/s decode, 28.26 tok/s end-to-end, and 11.3 GiB peak VRAM. The runner's
decoded text includes tokenizer-display markers (`Ċ`, `Ġ`) in the diagnostic
printer; these are token rendering markers, not model output bytes.

The attempted IQ3 fused re-enable was rejected by the CPU shadow gate: after
the block/head mapping and full eight-value RMS-statistic fixes, the first
prompt token still differed (GPU norm 95.6754 versus CPU 94.6998). Production
dispatch therefore remains on the scalar-equivalent route until a dedicated
fused-vs-unfused tensor verifier is added.

An identical ChatML coding prompt was run through both implementations with
temperature 0, seed 1, 128 output tokens, and the same IQ2 GGUF. llama.cpp
returned a valid compilable function:

```c
int clamp(int x, int lo, int hi) {
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}
```

The current runner did **not** pass this quality check: it emitted `beck`,
`-`, spaces, and repeated token 15/NUL-like output instead of C. Exact
tokenization was then checked independently: both implementations produced
the same 40 prompt IDs, including all ChatML control tokens, with no BOS
insertion. The runner's batched and per-token GPU paths also agree
(`argmax=71093`, the `beck` token), so this is a model-execution/parity issue
versus llama.cpp rather than a JSONL or sampling mismatch. Therefore the
throughput measurements above are valid kernel benchmarks, but they are not
evidence of end-to-end generation quality. Comparison logs are
`tmp/qwen38/ours-coding-validation.out` and
`tmp/qwen38/llama-coding-validation.out`.

The follow-up layer-0 parity trace shows the SSM inputs are already close to
llama.cpp. For the first token, this runner reports Q beginning
`0.0480,-0.0094,0.0187`, K `0.0014,-0.0033,-0.0023`, and recurrent gate
`-0.2128,-0.0400,-1.3307`; llama.cpp reports `0.0477,-0.0099,0.0191`,
`0.0014,-0.0033,-0.0023`, and `-0.2129,-0.0400,-1.3309`. The GDN formulas,
head repetition, decay, beta sigmoid, SiLU gate, and 1/sqrt(128) output scale
were checked against llama.cpp's `ggml_compute_forward_gated_delta_net`.

A selective F16-dequantized `ssm_out` oracle for layer 0 still selected the
same wrong first token (`71093`, `beck`). This rules out the LM head and a
single SSM output quantized matvec as the sole cause; the remaining mismatch
is accumulated numerical error across the hybrid stack. The diagnostic was
removed from production. The acceptance test remains the exact coding prompt
above, validated against llama.cpp with `llama-cli -ngl 99 -dev ROCm0 -fa on
--temp 0 --seed 1 --reasoning off --no-jinja`.

Current validation appears first. Earlier investigations are retained below as
history; short-run determinism claims there do not establish scalar F16 parity.
Scalar F16 remains the default; staged prefill is diagnostic.

### Default IQ4_XS SSM-output Q8_K path: fresh llama.cpp comparison

The validated production path now uses Q8_K activation quantization for
Qwen3.5 `IQ4_XS` SSM-output projections by default. This mirrors the
llama.cpp quantized activation route at the SSM-output boundary while leaving
other tensor types and unsupported combinations on their existing kernels.
Set `LLM_QWEN35_IQ4_SSM_OUT_Q8K=0` to restore the previous F32 activation
route for an A/B test.

On the exact 40-token coding prompt, the fresh GPU-run final logits compared
with the same-card llama.cpp Vulkan logits have relative L2 `0.1654848240`,
max absolute error `2.6354480`, and both select token `71093`. The earlier
`0.164390`/`0.164322` values came from intermediate captures and are retained
only as A/B history; the clean rebuilt binary and the current artifact now
agree at `0.1654848240`. Layer 0 attention-residual relative L2 improved from
`0.012493` to `0.010931`. This is measurable but not near bitwise or scalar
numerical parity: later hybrid layers still accumulate error, and the
generated-output acceptance check remains the authoritative quality gate.

The clean AMD run used `LLM_QWEN35_BATCH_SSM=1`, batched prefill, and the same
IQ2_XS GGUF: 33.46 tok/s prefill, 27.78 tok/s decode, 29.44 tok/s end-to-end
for 40 prompt + 80 generated tokens, with 11.3 GiB peak VRAM. A separate
layer-dump run reported lower decode throughput because tracing is enabled.
The llama.cpp Vulkan run on the RX 9070 XT produced the same compilable
`clamp` function as the quality-safe runner; the saved artifacts are
`tmp/qwen38/llama-coding-validation.out` and
`tmp/qwen38/defaultq8-quality.out`. The runner's CPU shadow check remains
strong at token 0 (`rel_L2=0.000009`) and token 1 (`rel_L2=0.008180`), while
the llama.cpp final-logit comparison is the longer-horizon parity check.

The llama.cpp reference was refreshed after the clean HIP rebuild with
`tmp/qwen38/dump_llama_logits_new` and produced byte-identical logits to the
previous reference. Both implementations returned the same compilable C:

```c
int clamp(int x, int lo, int hi) {
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}
```

The runner extraction passed `gcc -std=c11 -Wall -Wextra -pedantic`. Llama.cpp
reported `22.6 t/s` prompt processing and `28.1 t/s` generation for its
interactive validation; the clean HIP benchmark reported `33.45 tok/s`
prefill and `27.79 tok/s` decode.

After the IQ3_S alignment, the HIP coding-quality rerun still emitted the
same complete `clamp` implementation; its first generated token remained
`71093`. The 40+64 run passed with `33.77 tok/s` prefill and `26.97 tok/s`
decode (`29.23 tok/s` end-to-end). The extracted source again passed the
strict C compiler check.

The IQ1_S/IQ1_M FFN warp matvecs were then aligned with llama.cpp Vulkan's
operation grouping: FMA activation/grid products are accumulated within each
8-value sub-block, followed by a separate FMA for the sub-block scale. An
earlier capture showed a small improvement, but the clean rebuilt capture is
the authoritative result above; the change does not establish near-equality
across the full 64-layer stack.

A post-rebuild clean capture with the IQ3_S alignment produced relative L2
`0.1654848240`, max absolute logit error `2.6354480`, argmax `71093`
(llama.cpp `71093`), and `33.86 tok/s` prefill for the 40-token capture. The
full 40+80 benchmark measured `33.85/28.07/29.76` tok/s for
prefill/decode/end-to-end, with 11.3 GiB peak VRAM.

The corresponding IQ2_XXS FMA regrouping was tested for the down projection
and rejected: it moved the same metric back to `0.164335`. It was removed;
the production source retains the original IQ2 accumulation order.

An additional A/B port of llama.cpp Vulkan's nested IQ4_XS FMA reduction was
rejected: it changed layer-0 output below measurement significance and moved
final-logit relative L2 from `0.166789` to `0.166843` in the F32 control.
The source was restored to the four-partial-sum kernel. This rules out a
simple reduction-order swap as the dominant remaining cause; the next parity
work should target the hybrid recurrent/projection boundary rather than
changing this hot IQ4 reduction blindly.

An opt-in mixed-precision SSM-output experiment was also rejected. It blended
25% of the native F32 IQ4_XS projection into the default Q8_K result, adding a
second projection pass. On the exact prompt it worsened final-logit relative
L2 from `0.165499` to `0.165937` (max error `2.62941`; argmax unchanged at
`71093`). The experiment was removed, so the default path keeps its original
throughput and VRAM behavior. This further indicates that the remaining
error is accumulated in the hybrid recurrent/GDN stack or its boundary
ordering, rather than being fixed by a simple SSM-output precision blend.

The corresponding IQ2_XS FMA-order port was rejected as well. Matching
llama.cpp Vulkan's inner signed-codebook FMA followed by one outer scale FMA
changed the clean final-logit relative L2 from `0.16549904` to `0.16551268`
(argmax unchanged) while prefill measured `33.84 tok/s`. The source retains
the faster original reduction order; simple reduction reassociation is not
the remaining parity fix.

Whole-stack BF16 staging experiments were rejected. Enabling BF16 for all
dense projections produced relative L2 `0.166031` at `20.48 tok/s` prefill;
BF16 FFN-only produced `0.168137` at `24.69 tok/s`. Both retained argmax
`71093`, but both degraded numerical agreement and throughput, confirming
that a lossy BF16 weight round-trip is not a viable parity path for this IQ2_XS
model.

The fully enabled IQ3_S Q8/DP4A batch path was also rejected as a production
parity route: with `LLM_QWEN35_NATIVE_IQ2_BATCH=1` and
`LLM_QWEN35_NATIVE_IQ3_DP4A_BATCH=1`, relative L2 became `1.48125`, max error
`27.0812`, and argmax changed to `895` (prefill `36.03 tok/s`). The DP4A
activation path therefore remains opt-in and is not used by the validated
default; its large error is a separate kernel-correctness issue to fix before
using it for performance.

The first native IQ3_S bug was corrected during follow-up: the DP4A kernel
indexed each 8-value activation group with one `int` instead of two, so its
packed input offsets overlapped. Changing `u0[l]`/`u1[l]` to
`u0[2*l]`/`u1[2*l]` reduced the A/B error to relative L2 `1.31219` and raised
prefill to `35.59 tok/s`, but the full native-IQ2 gate still changes argmax to
`67983`. It is therefore not promoted; additional type-specific native paths
need independent correctness fixes.

The same offset correction was applied to the IQ3_S scalar DP4A macro and
decode kernel, so all IQ3_S DP4A variants now address complete 8-value groups
consistently. The production default was re-run after rebuild: relative L2
`0.1654848241`, max error `2.6354480`, argmax `71093`, and `33.71 tok/s`
prefill; no default-path regression was observed.

Follow-up type isolation confirms that Q8/DP4A activation staging, rather than
the batched scheduler, is the unresolved native-path error. IQ3_S-only native
dispatch measured relative L2 `1.56294` and selected `82302`; IQ2-only measured
`1.06875` and selected `760`; IQ2_XS-only measured `0.45511` but retained the
correct argmax `71093`. These are diagnostic runs only. The IQ2_XS DP4A path
uses the correct 7-bit-plus-parity sign expansion, but its two-pass Q8
activation approximation is still far too lossy for accumulated 64-layer
recurrent execution. Native DP4A remains disabled in the production profile;
the validated F32/Q8-K boundary remains the quality/performance baseline.

## Scalar-order SSM warp optimization reaches 126/22

`LLM_QWEN4_SSM_NATIVE_WARP=1` maps one output to one physical warp,
reproducing all eight scalar virtual-warp reductions in order. Mode2 omits
empty virtual warps and canonicalizes zero. Both pass the real-weight SSM
projection oracle (36 layers x5 projections x8 rows); mode2 additionally
includes signed-zero inputs. Logs:`tmp/ssmwarp_oracle.log`,
`tmp/ssmtrim_oracle.log`. Default0 retains the baseline.

All following two-request4096/64 runs match the fresh scalar first15 /
hashe3d8bf6d47dc6cc3:

| Change from fully native baseline | Prefill min/median | Decode min/median |
|---|---:|---:|
| Native SSM warp1 |122.39 /122.39|19.50 /19.62|
| Warp1, existing prefill cache balance1, LFU1, attention shards4 |122.47 /122.55|21.70 /21.75|
| Warp2, same cache settings, shards2 |125.97 /125.97|21.75 /21.79|

The cache changes reduce decode H2D from13.31 to9.87 GiB and raise hits
from85.0% to88.9%. They do not change routed computations. Latest driver:
`tmp/run_native_ssmtrim_4k.sh`; log:`tmp/native_ssmtrim_4k.log`;
binary:`tmp/test_hip_llm_ssmtrim`. Cache7200 MiB, BMAX4096,
context8192, native HC/SSM/router/shared/attention projections, PLE split,
scratch arena, exact prefix graphs, pinned overlapped staging, promotion1.

The correct-path profile (`tmp/native_profile_analysis.log`) had33.045 s
prefill kernel time:8.256 s SSM input projections,7.312 s routed gate/up,
4.487 s Q5 down,2.934 s other native Q8 batches,2.336 s DeltaNet.
Decode kernels1.992 s include0.762 s F16 attention, with1.124 s H2D.
These overlapping trace durations are not additive wall times.

Rejected: a three-kernel parallel-score attention experiment matches scalar
bits at five positions through4354 but is slower in the undelayed isolated
benchmark (0.708 vs0.464 ms/call). It is removed from live source; archives
and logs remain under `tmp/rejected_qwen4_attn_tiles*` and
`tmp/attntiles_unmodified_benchmark.log`. The first timing comparison
contained deliberately delayed reference waves and is invalid for speed.
The expanded attention regression now covers4355 tokens and uses a linear
CPU prefix-sum reference; the barrier-removal negative control still fails.

**200 prefill /30 decode remains unmet.** Next focus: routed projection
throughput, with FP32 numerical checks and full scalar greedy hashes.

## Fresh 4K parity passes; native throughput is 114/20

The fully native staged 4096/64 path matches a fresh scalar F16 reference in
both requests: first token **15**, full hash **e3d8bf6d47dc6cc3**.
Scalar reference: 13.29 prefill /20.46 decode tok/s, 87.0% decode cache hits,
11.51 GiB decode H2D. Native staging: **114.38/114.52 prefill min/median**,
**19.78/19.82 decode min/median**,85.0% hits,13.31 GiB decode H2D,
65.69 GiB prefill staging,370 waves,2383 promotions,zero fallbacks.
Peak15890 MiB,414 MiB free. Native arithmetic thus restores short and4K
greedy parity but does not meet200/30.

Settings: `batch4k-stage`, cache7200 MiB, BMAX4096, context8192,
pinned weights, copy overlap, PLE split, phase arena, native Q8 attention
projections, native HC, native SSM projections, native router/shared experts,
exact prefix graphs; stage promotion1, prefill cache balance0, shards1,
stage threads256, warmup0. There was no concurrent CPU compilation.
Logs:`tmp/native_scalar_4k_reference.log`, `tmp/native_all_4k.log`,
`tmp/native_all_4k_summary.log`. Driver:`tmp/run_native_all_4k.sh`.

The next profile is `tmp/rocprof_qwen_native/`, with route logging in
`tmp/qwen_native_profile.log`. Optimize this scalar-parity baseline, not
the historical mismatching170/13.6 path. Longer and multi-chunk quality,
broader repeat counts, and200/30 remain outstanding.

## Native router/shared batching restores short-prompt parity

`LLM_QWEN4_BATCH_MOE_NATIVE=1` computes router logits and the shared gate
with BF16 weights and F32 input, matching scalar GPU arithmetic. Shared Q8/Q6
gate/up/SiLU and down accumulation also retain scalar per-output operations.
The option bypasses BF16 activation packing for these operations and remains
off by default. Unsupported shared formats fail explicitly when requested.
Routed expert execution is unchanged.

The real-model `--verify-moe-native` oracle passes bitwise on all 48 layers
x 8 rows for router logits, shared scale, shared gate output, and accumulated
output. Log: `tmp/nativemoe_oracle.log`; binary:
`tmp/test_hip_llm_nativemoe`.

Combining native HC, native SSM, native Q8 attention projections and native
router/shared batching passes the fresh scalar F16 first-token and full-hash
checks on **all four short prompts, two repeats each (8/8 requests)**:

| Prompt | Matching first token / 16-token hash |
|---|---|
| Coding | 198 / bbd62d9e3c85af8d |
| Arithmetic | 198 / 427a8efc219e9443 |
| Prose | 248068 / c461a4dabdca797e |
| Japanese | 198 / b5ad0bef0c9a5696 |

Settings: 128-token prefill, 16-token decode, context8192, BMAX4096,
cache4000 MiB, pinned weights, PLE phase split, scratch arena, overlap on,
staging promotion off. Logs: `tmp/nativemoe_quality_summary.log` and
`tmp/staging_quality_fixed/*_nativemoe.log`. This is short-prompt greedy
parity, not bitwise equivalence of every model intermediate. A fresh4K/64
scalar reference and two native batched requests are running next; the200/30
performance target remains unmet.

## Native Q8 SSM projections

`LLM_QWEN4_BATCH_SSM_NATIVE=1` bypasses BF16 input/output GEMMs for
supported Q8 SSM layers. The fused Q8 input kernel preserves scalar reduction
order for QKV, gate, and F16/F32 alpha/beta; the output uses the validated
native Q8 batch kernel. Unsupported weight combinations retain their existing
path. The option remains diagnostic and defaults off.

The real-model `--verify-ssm-projections` oracle passes bitwise, with finite
outputs, on all 36 SSM layers x 5 projections x 8 rows. This validates
projections, not recurrence or the whole model. Build:
`TMPDIR=$PWD/rdna4/llm/tmp make -C rdna4/llm TARGET=tmp/test_hip_llm_nativessm`.
Run with `LLM_BMAX=4096 LLM_QWEN4_BATCH=1 LLM_QWEN4_BATCH_SSM=1`,
the model path, and `-s 8192 --gpu-only-bench --moe-cache-mb 4000
--verify-ssm-projections`. Log: `tmp/nativessm_oracle.log`.

Native HC + SSM still fails all four fresh scalar 128/16 references,
repeatably: coding 107300/dffd736659bc0c82; arithmetic
248068/9789b7989b7ebc32; prose 248045/355cc01b83b41f71;
Japanese 248044/2666e2f5535d3399.
Log: `tmp/nativessm_quality_summary.log`. Router and shared-expert
batching still use BF16 intermediates and remain unvalidated against scalar.

## Corrected-routing performance and quality refresh

The corrected staged 4096/64 workload is repeatable across 20 requests in
nine processes (first99157, hash601167e3b2fb9425). This is a staged reference,
not scalar parity. Cache7200 plus the phase arena leaves508 MiB free
(15796 MiB peak). Geometry128/256/512 and attention output shards1/2/4/8
retain that hash; neither sweep establishes a compelling throughput gain.
The shard sweep, without concurrent CPU compilation, spans133–167 prefill
and6–12 decode tok/s. Logs: `tmp/native_shards_summary.log`,
`tmp/geometry_sweep_summary.log`. Geometry128 timing overlapped compilation.

`LLM_QWEN4_NATIVE_Q8_BATCH=1` replaces token-at-a-time Q8 projection launches
with an existing native batch kernel. The expanded GPU oracle compares
M=1/3/17 at four real projection shapes, including non-power-of-two quant
scales, bitwise against scalar. It passes. Output initialization is ordered
on the compute stream. The API trace confirms196608 scalar launches become
48 batch launches, but overall throughput remains below target.

The corrected-route ROCprof trace (`tmp/rocprof_qwen_api/`) records decode
H2D2.676 s for30.48 GiB over64 tokens, about42 ms/token and11.4 GiB/s.
Decode kernels total2.200 s, including0.772 s F16 attention. Host tracing
shows3143 stream synchronizations and74511 kernel launches during decode.
Blocking API times overlap GPU execution and must not be added to it.
Prefill kernel time20.884 s includes7.117 s grouped Q4 gate/up,4.469 s
Q5 down,2.268 s DeltaNet,1.595 s native Q8 batch and1.506 s attention.
Profiled135/7 tok/s includes instrumentation and is not an acceptance run.

Fresh matched scalar F16 references and staged runs at128/16, two repeats
each, fail parity on all four short prompts:

| Prompt | Scalar first / hash | Staged first / hash |
|---|---|---|
| Coding |198 / bbd62d9e3c85af8d|1271 / 01976b77d164dc71|
| Arithmetic |198 / 427a8efc219e9443|95597 / 81052d0486585444|
| Prose |248068 / c461a4dabdca797e|292 / 81d9d59fdc04f3e6|
| Japanese |198 / b5ad0bef0c9a5696|57512 / 9fcc5e9faf32dfaa|

Logs: `tmp/fixed_short_quality_summary.log` and
`tmp/staging_quality_fixed/`. A fresh4K scalar oracle remains outstanding.
No performance or quality gate has been met.

Native HC and exact decode prefix graphs are opt-in experiments:

- `LLM_QWEN4_BATCH_HC_NATIVE=1` keeps HC down/up and injection in native
  scalar arithmetic, eliminating BF16 intermediate packing. The real-model
  `--verify-hc-batch` oracle checks48 layers x2 phases x8 rows; mixed outputs
  and injection weights all match bitwise and remain finite. It supports
  Q8 HC down/up and F16/F32/Q8 injection. Log:
  `tmp/nativehc_f16_oracle.log`. Full prompt parity fails all four128/16 cases, repeatably: coding
  first169742/hashf0ddf46ba0ec6790; arithmetic225110/f7398932b2a66d89;
  prose73889/ff576e34061fd89e; Japanese119294/6118d1b7f5ef72b4.
  Log:`tmp/nativehc_quality_summary.log`. Other batch paths still round
  through BF16, including this model's Q8 SSM projections.
- `LLM_QWEN4_EXACT_PRE_GRAPHS=1` captures the existing attention/SSM and HC
  prefix for single-token decode, excluding PLE layer1 and keeping routed MoE
  outside capture. Approximate graphs containing MoE cannot be reused as exact
  prefixes. Offload destroys captures before freeing addresses. Two4096/64
  requests match the corrected staged hash, at166.76/168.28 prefill min/median,
  10.87/11.12 decode. Peak15890 MiB. Log:`tmp/exactgraph_4k.log`.
  A final build confirms47 captured graphs, zero failures, and two matching
  staged hashes:170.00/170.42 prefill and13.57/13.59 decode min/median.
  Log:`tmp/verified_graph_4k.log`. Variation across processes still warrants
  more repeats before attributing that difference to graph replay.

## Routing, attention race, and scratch sharing (2026-09-12 continuation)

Two correctness defects are now demonstrated and fixed:

- `moe_topk_batch` could discard an unselected lower-half expert after choosing
  its paired upper-half candidate. It now masks each candidate independently.
  The expanded oracle covers 257/384/512 experts, K=10/64, paired candidates,
  ties, ascending/descending logits, and seeded random logits. The old kernel
  fails this oracle; the corrected kernel passes.
- F16 prefill/decode attention reused the shared maximum-reduction buffer for
  probability sums before every wave had read the maximum. Asynchronous
  fingerprints first localized repeat divergence to the attention side of
  layers 3/39, before FFN routing. A delayed-wave GPU oracle reproduces errors
  of 0.0238 (prefill) and 0.0392 (decode) with the barrier removed. With the
  reader barrier, maximum absolute errors are 2.98e-8 and 7.45e-9. The analogous
  I8 buffer reuse receives the same barrier.

Historical `afdf60ceeb4f0103` throughput used the incorrect top-K kernel and
must not be treated as a current quality baseline. Correct routing selects a
more varied expert workload and increases decode transfers.

`LLM_QWEN4_BATCH_PLE_FFN=1` keeps layer 1's PLE/SSM attention row-ordered and
batches its FFN. The real-weight `--verify-ple-split` oracle compares the
original interleaved scalar layer with phase-separated scalar attention and
FFN: HC outputs, PLE convolution, and SSM convolution/recurrent state match
bitwise. This validates the phase separation, not the batched FFN's scalar
numerical parity. The opt-in split cuts 4K prefill expert H2D from 132.68 to
66.24 GiB. Its reused host PLE embedding remains live through each row's
existing explicit completion.

`LLM_QWEN4_FINGERPRINT=1` records stream-ordered diagnostic hashes for five
boundaries per layer: HC input, FFN input, router logits, routed sum, and
shared+routed output. Reporting uses the existing end-of-tile barrier. These
32-bit fingerprints locate divergence; they are not collision-free equality
proofs or performance measurements.

After the attention fix, four 4096/64 requests at cache5500/BMAX4096 have
identical fingerprints at all 48 layers and return first token 99157 / hash
`601167e3b2fb9425` (4/4). Settings: pinned weights, overlap on, PLE split on,
staging promotion on, prefill cache balancing off, warmup off. Diagnostic
throughput is 145.92/146.47 prefill min/median and 12.02/12.17 decode; peak
15808 MiB. This establishes repeatability for these requests, not scalar F16
parity. Log: `tmp/attention_fixed_4k.log`.

`LLM_QWEN4_PHASE_SCRATCH=1` shares temporary storage between HC mixing, SSM,
full attention, and MoE. Residuals, injection weights, normalized layer inputs,
attention projections, dequantized weights, and copy-stream staging banks
remain separate. Cleanup clears owned views before ordinary per-field frees.
At BMAX4096 the arena is 810 MiB and saves 1737 MiB. Both requests with the
same cache5500 match all baseline fingerprints and the complete decode hash;
peak falls to 14072 MiB, leaving 2232 MiB free. Log:
`tmp/arena4_trace_4k.log`. A larger-cache/launch-geometry sweep follows.

Rejected: two-token gate/up tiling initially passed fixtures with power-of-two
scales, but the expanded non-power-of-two scale fixture detects a one-ULP
gate difference. The prototype is removed. The down-projection prototype also
failed parity. The broader fixture remains. `LLM_QWEN4_STAGE_THREADS` changes
only the existing kernels' block geometry (128/256/512, default256).

A ROCprof trace before the attention fix identified the remaining costs:
prefill grouped Q4_K gate/up 7.105 s, Q5_1 down 4.448 s, scalar Q8 matvec
2.526 s, and batched DeltaNet 2.267 s. Decode spent 3.095 s in H2D copies and
2.279 s in kernels for 64 tokens; F16 attention was 0.772 s. These are trace
durations, not additive wall-clock throughput claims. Mapped/direct host-read
decode experiments were slower than copies and failed repeatability; they
are not promoted. The device link reports PCIe 3.0 x16.

Reproduce the focused checks (GPU jobs must run exclusively):

```sh
export TMPDIR="$PWD/rdna4/llm/tmp"
make -C rdna4/llm moe-stage-test
make -C rdna4/llm qwen4-attention-gpu-test
make -C rdna4/llm tmp/test_hip_qwen4_moe_stage_large
LLM_QWEN4_STAGE_THREADS=128 timeout --foreground 180s ./rdna4/llm/tmp/test_hip_qwen4_moe_stage_large
LLM_QWEN4_STAGE_THREADS=512 timeout --foreground 180s ./rdna4/llm/tmp/test_hip_qwen4_moe_stage_large
```

The 200/30 target and fresh scalar F16 corpus parity remain required; no
batched production default is promoted by these diagnostics.

## Bounded staging implementation (2026-09-12)

The selected bounded-wave manager is implemented. Each of two banks owns its
metadata, quantized weights, Q8_0 repack scratch, and upload/consumption events.
Both host-source lifetime and device-slot reuse are fenced. Direct prefill now
uses independent slot fences, including Q8_0 misses and cache-hit consumers.
The prefill-to-decode map upload completes before its host buffer is freed.

The default pool remains 512 MiB, including device metadata/repack storage.
`LLM_MOE_COPY_PIPELINE=0` serializes waves; `=1` overlaps two banks. Promotion
remains off by default. Runtime errors cannot fall through to a partial-result
fallback. Reset, mode transitions, offload, and free drain outstanding prefill
work before touching owned storage. See [the staging design and commands](QWEN38_MOE_STAGING.md).

`LLM_BENCH_WARMUP=0` now actually disables the optional warmup (previously any
present value enabled it). Reference first-token/hash inputs are paired and
reject a repeatable candidate that differs from the scalar F16 oracle.

Validated:

- Build of the runner and model-free GPU staging test.
- `make -C rdna4/llm moe-stage-test`: profile, delayed-copy lifecycle,
  bank ownership, and benchmark-reference/parser tests all pass.
- CPU ownership test under AddressSanitizer/UndefinedBehaviorSanitizer with
  leak detection: pass. Removing either host-lifetime or slot-reuse fencing
  makes delayed-consumer assertions fail.
- `timeout --foreground 180s ./rdna4/llm/tmp/test_hip_qwen4_moe_stage`: pass
  on RX 9070 XT. Six Q4_K/Q5_K + Q5_1/Q8_0/Q6_K type combinations, both host
  registration modes, resident/cold/mixed workloads, partial waves, overlap,
  promotion, resets, and injected upload failure/recovery match a serialized
  expert oracle bitwise. This does not establish whole-model scalar parity.

Full-model matrix: RX 9070 XT, 9,000-byte header prompt, 4096 prefill / 64
decode, context 8192, BMAX4096, 4000 MiB cache, 512 MiB staging, eight measured
requests per process, warmup disabled, CPU experts/approximation/promotion off.
The common batched hash is `afdf60ceeb4f0103` (first token 16).

| Registered | Overlap | Common hash / 8 | Prefill min / median tok/s | Decode min / median tok/s | Peak MiB |
| --- | --- | ---: | ---: | ---: | ---: |
| No | No | 7 | 86.28 / 117.48 | 9.96 / 17.86 | 14294 |
| No | Yes | 8 | 97.46 / 128.87 | 17.54 / 17.83 | 14296 |
| Yes | No | 5 | 122.93 / 122.99 | 11.42 / 19.07 | 14294 |
| Yes | Yes | 7 | 132.22 / 132.30 | 19.17 / 19.69 | 14296 |
| Yes | Yes, fresh process | 6 | 132.09 / 132.19 | 11.39 / 19.66 | 14296 |
| No | Yes, fresh process | 8 | 96.33 / 128.63 | 16.11 / 17.74 | 14296 |

Four of six processes fail repeatability. Pageable overlap passes 8/8 in
both processes, but this does not establish scalar parity or justify promotion. Full-model nondeterminism remains despite passing ownership tests;
its cause is not established. Neither throughput target is met. Logs:
`tmp/qwen38_stage_matrix_[1-6]_r*_p*.log`. A CPU test compilation overlapped
part of the fourth process; the fresh fifth process had no compiler overlap.

The DIM2048 GPU stress variant also passes the same serial-oracle matrix with
multi-MiB weight transfers (`tmp/gpu_moe_stage_large.log`).

Direct-cache smoke: `bench_qwen38_target.sh` with profile `batch4k`,
512 prefill / 8 decode, two repeats, registered weights, BMAX4096/cache4000,
context8192, the same header prompt, and warmup off passes with copy pipeline
both 0 and 1. Both return first token 18 / hash `75c0ebdb415406cd` (2/2).
Serial median prefill/decode is 58.31/11.70 tok/s, peak13778 MiB; overlap is
54.14/12.21 tok/s, peak13780 MiB. These are bounded regression controls, not
4K determinism or scalar-parity evidence. Logs: `tmp/qwen38_direct_p[01].log`.

Matched F16 quality corpus (`./rdna4/llm/test_qwen38_staging_quality.sh`):
**all five candidates fail scalar parity**. Each scalar and staged result is
repeatable 2/2 within this corpus. The target is 4096/64; the other four are
128/16. All use context8192, cache4000, BMAX4096, registered weights, CPU
experts off, and no approximation. Scalar uses `scalar-exact`, batch0/copy0;
staged uses `batch4k-stage`, batch1/copy1. Exact prompts and remaining profile
settings are in the script; logs are `tmp/staging_quality/*_{scalar,staged}.log`.

| Prompt | Scalar first / hash | Staged first / hash | Parity |
| --- | --- | --- | --- |
| target | 15 / `e3d8bf6d47dc6cc3` | 16 / `afdf60ceeb4f0103` | FAIL |
| coding | 198 / `bbd62d9e3c85af8d` | 47932 / `12ced4d4ee57e7bb` | FAIL |
| arithmetic | 198 / `427a8efc219e9443` | 15 / `ee81d9e0e8e89d42` | FAIL |
| prose | 248068 / `c461a4dabdca797e` | 9619 / `458c70b584329680` | FAIL |
| japanese | 198 / `b5ad0bef0c9a5696` | 44868 / `a099ba1b986b4c43` | FAIL |

The fresh scalar target measured 10.83 prefill / 15.45 decode tok/s (median),
peak 11328 MiB. Its hash differs from the historical `fast` profile with
BMAX2048/7800 MiB cache; the cause of that cross-configuration difference is
not established. These results test repeatability and reference agreement,
not independent model-quality correctness. No throughput or quality target
is claimed, and no serving default is promoted.


Device access: the restricted namespace hides `/dev/kfd` and `/dev/dri`, but
the authorized host execution namespace exposes the RX 9070 XT. Check `fuser`
and VRAM before each standalone benchmark; never run concurrently with a server.

## Earlier measurements (before bounded staging)


- FP8 KV uses real scaled E4M3 encode/decode kernels, not an I8 alias.
- Short exact-MTP parity: I8, FP8, and F16 all returned hash
  `454146399ff97e88` for the 28-token/4-output control.
- FP8 64-token coding control retained the F16 hash
  `4b8937cd7db0e7a7` and returned `PASS`.
- FP8 256K allocation smoke: 3.000 GiB KV, 4.91 prefill, 5.60 decode,
  5.23 end-to-end tok/s, `PASS`.
- FP8 generation under the 256K allocation: 8-token prompt + 64-token greedy
  output, 4.84 prefill, 8.05 decode, 7.50 end-to-end tok/s, hash
  `d6951038d33a3c17`, 15.934 GiB peak used, `PASS`.
- F16 at max 256K allocation requests 6.000 GiB KV and fails finalization on
  the 16-GiB card; no F16 256K result is claimed.
- Profile, lifecycle, protocol, FP8 codec, and shell/static checks pass.
- With temporary `/dev/kfd`/`/dev/dri` access, the 512-token scalar control
  completed end-to-end at 25.71 prefill / 27.24 decode / 25.73 tok/s and
  `PASS` (12.708 GiB peak VRAM).
- The scalar two-chunk 1,024-token control also completed with the same
  sequence hash (`a2d4f49620d5b663`): 23.61 prefill / 23.24 decode / 23.61
  end-to-end tok/s, `PASS` (12.708 GiB peak VRAM).
- The built-in 512-token batched-vs-per-token comparison reported exact
  logits (`rel_l2=0`, `max_abs=0`) and matching argmax token 30.
- The I8 GQA8 selector now correctly recognizes Qwen4's 24/2 (12:1) shape,
  and its reduction bug (only four keys reaching softmax) was fixed. A matched
  512+64 control still produced a different hash with GQA8 enabled
  (`0b3e44e3e7fb5b0c`) versus scalar I8 (`6f231930c119b7e6`), so it remains
  opt-in pending long-horizon quality parity.
- I8 now honors `LLM_QWEN4_DISABLE_QSA=1`, matching the F16 control and
  allowing KV-only drift isolation; the default remains unchanged.
- Matched 512-token/64-decode greedy controls without MTP confirm the format
  boundary: F16 and FP8 both hash `e7e9b5ace7b8f98a` (26.06/24.25 decode
  tok/s respectively), while I8 hashes `6f231930c119b7e6` (15.86 tok/s).
  This isolates the long-horizon drift to INT8 KV quantization/attention, not
  routing or general recurrent state.
- A 16-channel I8/FP8 scale-group experiment was rejected: the I8 control's
  first token changed from 30 to 271. The implementation is restored to the
  validated 32-channel groups; finer grouping needs a separate parity design.
- The explicit batched WMMA path is stable for streamed-shaped requests: 2,048
  tokens at 114.88 prefill / 15.05 decode / 111.99 end-to-end tok/s, and
  4,096 tokens at 112.21 prefill / 14.53 decode / 110.75 end-to-end tok/s;
  both returned `PASS` with 13.448/13.544 GiB peak VRAM respectively.
- Two identical resumed 4,096-token batched runs reproduced throughput
  (111.26/111.49 prefill and 109.39/109.34 end-to-end tok/s, 13.544 GiB
  peak), but produced different hashes (`d973fd2d3e5f6bfc` and
  `42af972770bae7cf`). This is direct evidence that grouped/batched state is
  still nondeterministic on gfx1201 and cannot be promoted to a quality-safe
  default.
- Disabling expert copy pipelining (`LLM_MOE_COPY_PIPELINE=0`) did not remove
  the variance: the 4K run measured 96.67 prefill / 95.17 end-to-end tok/s
  and hash `d9f9ce178c4b4eb1`. The nondeterminism is therefore broader than
  the asynchronous copy overlap; no production switch was changed.
- An opt-in ordered expert-scatter reduction was also tested on the same 2K
  control. Two runs still produced different hashes, so atomic expert
  scatter is not the sole source of the grouped-path variance; the
  experimental kernel was removed rather than adding another unsupported
  production switch.
- The grouped resident/staged task lists previously used asynchronous H2D
  copies from pageable scratch that was overwritten during the subsequent
  cold-expert walk. Those metadata copies now complete synchronously before
  scratch reuse; runtime parity still needs a model-mounted rerun.
- The post-fix 2,048-token grouped controls both passed, at 115.61/109.73
  prefill/end-to-end tok/s and 118.97/113.57 tok/s, but still diverged at the
  first decoded token (17512 vs 220) and sequence hash
  (`70e30279debbe15f` vs `b01463d3c4871377`). The copy-publication race was
  real-risk mitigation, not the complete source of grouped nondeterminism.
- An opt-in scalar-router diagnostic (`LLM_QWEN4_BATCH_ROUTER_SCALAR=1`) also
  failed to stabilize matched 512-token controls: hashes were
  `5c7ed763fe63b0da` and `c6761ef1770a4293` (first tokens 515 and 10586), at
  71.97/66.68 and 71.10/66.07 prefill/end-to-end tok/s. Router reduction
  ordering is therefore not the sole remaining source.
- A stricter 512-token isolation with batched SSM, attention, router,
  projections, grouped MoE, and copy overlap disabled still diverged: hashes
  `ee573f2372785825` and `fbade0a3d53fbd73` (first tokens 17 and 11), at
  62.49/60.34 and 62.49/60.37 prefill/end-to-end tok/s. The remaining bug is
  therefore in broader batched state/stream publication, not one isolated
  grouped kernel; scalar dispatch remains the reference path.
- Per-row fallback and scalar KV diagnostic loops also had an async-copy hazard:
  each queued position transfer referenced a loop-local stack `pos`. Those
  publications are now synchronous. Matched post-fix controls still differed
  (`2aa0ef09c71a181b` vs `fbade0a3d53fbd73`, first tokens 17 vs 11), so this
  was another real race but not the complete source.
- Fresh rebuilt scalar controls remain deterministic: both 512-token runs
  returned first token 30 and hash `a2d4f49620d5b663`, at 25.84/25.86 and
  25.87/25.83 prefill/end-to-end tok/s. The instability is specific to the
  batched dispatcher, not general GPU state or model loading.
- For completeness, forcing `OMP_NUM_THREADS=1` on the scalarized batched
  isolation also failed to stabilize output: hashes `b9ee92610dbbc1cd` and
  `6254e5064f4050f4` (first tokens 17 and 271). Host OpenMP scheduling is not
  the remaining explanation.
- Forcing `LLM_QWEN4_BATCH_HC_SCALAR=1` stabilized the first decoded token
  (`907` in both 512-token controls), but later decode still diverged: hashes
  `3ced02b6184309f7` and `1c91fa80c934253e` at 48.84/48.35 tok/s end-to-end.
  This identifies batched HC/PLE arithmetic as one prefill mismatch source,
  while leaving a separate post-prefill decode-state/KV handoff issue.
- Forcing both `LLM_QWEN4_BATCH_KV_SCALAR=1` and
  `LLM_QWEN4_BATCH_ATTN_SCALAR=1` did not stabilize the batched path either:
  first tokens were 30 and 220, with hashes `525be427d6b320b5` and
  `f5829d6be7fb6728`. KV-store/attention publication is not a standalone fix.
- The HC-scalar result is repeatable as a prefill diagnostic but not a serving
  solution: it agrees on the first token while later decode state diverges;
  scalar KV/attention publication instead changes the first token again. All
  such switches remain opt-in and the scalar dispatcher remains the only
  quality-safe default.
- `HIP_LAUNCH_BLOCKING=1` also failed to stabilize matched 512-token batched
  controls: hashes `7c0b751e708b64fd` and `7ec265c22d7a16f8` (first tokens 47
  and 12920). The variance is not eliminated by global launch serialization.
- Forcing scalar token embedding (`LLM_QWEN4_BATCH_EMBED_SCALAR=1`) likewise
  left the batched path nondeterministic: hashes `fc6397b74e8e9d83` and
  `83452d2dfbc8621b` (first tokens 435 and 5652). Divergence begins after or
  within the first batched layer, not in embedding publication.

## Repeatability gate and batched divergence localization (Phase 0/1)

The runner now has an in-process repeatability gate so a profile cannot be
promoted without passing it:

- `test_hip_llm --bench-repeat N` resets recurrent/KV/PLE state between N
  identical requests in one process (the model loads once). `--bench` footers
  report `First decoded token id` and `sequence hash`.
- `bench_qwen38_target.sh` runs `scalar-exact`, `fast`, `batch`, or `approx`
  profiles for N repeats and fails unless every repeat has the same first token
  and the same full hash. It also reports min/median prefill/decode/end-to-end
  tok/s, peak VRAM, and `rocm-smi` clock/temp before and after.
  `bench_qwen38_256k.sh` (`QWEN38_BENCH_REPEATS`, default 2) and
  `bench_qwen38_sub32_target.sh` (`QWEN38_SUB32_REPEATS`) carry the same gate.
- `make -C rdna4/llm target-gate target-gate-fast target-gate-batch` wraps it.
- `debug_f32_state`/`debug_hc_state` now print a bitwise FNV hash of the full
  state in addition to norm/first under `LLM_DEBUG_LAYERS=1`, exposing one-ULP
  divergence that the 6-decimal print hides.

Matched results on the RX 9070 XT with the real 9,000-byte `gguf_loader.h`
prompt (`tmp/qwen38_target_prompt.txt`), profile `fast` (scalar, pinned host,
BMAX=2048, 7.8-GiB cache, GPU top-k) at 4,096 prefill / 64 decode:

- Three repeats were identical (hash `6d67721190bdaa83`, first token 30):
  23.96--23.98 prefill, 21.15--23.70 decode, 14,818 MiB peak.
- The quality-safe scalar route is therefore repeatable, but only ~24 tok/s
  prefill at 4K; the documented 100--235 tok/s figures are not reachable on the
  current scalar `fast` profile and require the batched route.

Profile `batch` (`LLM_QWEN4_BATCH=1`, `BATCH_SSM=1`, native Q6_K SSM
projections, fused recurrence, `BATCH_ATTN_MAX_LAYER=47`, Q6K/CONV/RECURRENCE/
PARITY, BMAX=1024, 5.9-GiB cache, 1,024-token stream) at 2,048 real tokens
originally reproduced the nondeterminism directly: three identical repeats
produced three different first tokens (32286, 16, 248046) and three different
hashes (`096888097a00e061`, `97574e0f11abcfd3`, `fb57a917f37a253e`).

### Fixes landed

- **Ordered MoE combine.** `moe_scatter_accum` summed the K selected experts
  with order-dependent `atomicAdd` into the token row. `moe_fill_gather` now
  records the expert-grouped slot for each `(token, rank)` in
  `d_moe_assign_pos`, and the new `moe_scatter_accum_ordered` sums the K
  contributions in fixed rank order. This moved the first bitwise divergence
  from layer 1 to layer 3 in the debug trace.
- **Synchronous CPU-result publication.** The CPU expert paths published their
  host results to the device with `hipMemcpyAsync` from `h_moe_output` /
  `h_moe_eout_cpu`, which the next token/layer overwrites; a DMA could race the
  rewrite. Both publications are now synchronous `hipMemcpy`.
- **CPU vs GPU cold-expert arithmetic.** The CPU expert kernels
  (`hllm_cpu_*_jobs`) do not reproduce the GPU kernels bit-for-bit, so which of
  the two evaluates a cold expert changes its contribution. Expert-cache warmth
  selects that path, so back-to-back in-process requests with the CPU path
  enabled hash-differ even though two fresh processes matched
  (`4f21b1d68505eec3` twice at 512/16). The repeatability gate therefore
  defaults CPU expert work off (`bench_qwen38_target.sh` `batch` profile,
  `QWEN38_TARGET_CPU_EXPERTS=0`); `batch-cpu` keeps the mixed path for
  diagnostics.

### Pre-manager observations (RX 9070 XT, real 9,000-byte prompt)

- Single-chunk batched (`prefill <= BMAX`), CPU experts off: **improved but
  still not reliably deterministic**. One real race was fixed: the per-row
  position publication used a blocking `hipMemcpy(r->d_position, &pos, ...)` on
  the null stream, which raced `r->stream` kernels still reading `d_position`
  from the prior row (every row of the forced-scalar layer 1). It now uses a
  stream-ordered `hipMemcpyAsync(r->d_position, &r->h_pos_batch[m], ...)` from a
  stable precomputed host array. With pageable host weights and direct copies
  an 8-repeat run passed, but repeating the same configuration diverged on
  2/8, and pinned host weights or the async pipeline diverge on roughly 1/8.
  `HIP_LAUNCH_BLOCKING=1` largely hides the residual, so at least one more
  ordering dependency remains. Forcing per-token MoE (`LLM_MOE_PREFILL_SCALAR=1`)
  at 512 prefill / 8 decode is deterministic 6/6 (`1ae7b536b9c17b1d`), which
  isolates the residual race to the batched MoE dispatcher
  (`forward_moe_ffn_batched`), not batched attention/SSM. The batched MoE
  kernels themselves have no atomics, and the cause is the cold-expert cache
  H2D not being ordered with the consuming kernel: a plain
  `hipMemcpyAsync(..., r->stream)` did not reliably order on this ROCm stack.
  The direct cold path now copies on `moe_copy_stream` and makes the compute
  stream wait with a `hipEventRecord`/`hipStreamWaitEvent` pair. With that,
  4,096/64 passes 7/8 repeats (hash `afdf60ceeb4f0103`) at ~125 prefill /
  19.7 decode, versus frequent failures before, but one repeat still diverged
  (`989013653e29726e`), so a rarer residual remains. Disabling the
  `d_moe_eout`/gather alias (`LLM_QWEN4_MOE_EOUT_ALIAS=0`) and switching to host
  router top-k (`LLM_QWEN4_PREFILL_GPU_TOPK=0`) each still diverged, so neither
  the alias nor the GPU router is the sole cause. A 4-repeat
  `LLM_DEBUG_LAYERS=1` trace at 1024 did not reproduce it (the per-stage sync
  perturbs timing). The scalar route remains the only quality-safe default.
- Multi-chunk stateful batching (prefill > BMAX, `LLM_QWEN4_BATCH_MULTI_CHUNK_
  FORCE=1`) still diverges: a 4,096-token prompt split at BMAX=1024 produced a
  different third-repeat hash. The inter-chunk state carry has a separate
  remaining race.
- The scalar `fast` route remains repeatable (3/3, hash `6d67721190bdaa83`) but
  only ~24 tok/s prefill at 4K.

### 4K batched profile (`batch4k`)

A single 4,096-token batched dispatch fits on the 16-GiB card with BMAX=4096
and a 5,000-MiB resident cache (peak ~14,900 MiB). `bench_qwen38_target.sh
batch4k` wraps it: GPU router top-k, CPU experts off, and
`LLM_QWEN4_RESET_MOE_CACHE=1`. It usually reproduces hash
`afdf60ceeb4f0103` (first token 16). With pageable host weights and direct
copies (the most-repeatable settings, `reg=0 copy=0`) it measured median
**125 prefill / 19.6 decode / 115.6 end-to-end tok/s**; with pinned host weights
and the async pipeline (`reg=1 copy=1`) it reaches median **149.2 prefill /
21.4 decode / 136.7 end-to-end tok/s** but diverges more often. Neither is fully
repeatable yet.

State-isolation findings that drove the profile:

- Expert-cache residency changes the result even with CPU experts off: the
  first repeat after load differed from later repeats that inherited cache
  residency. `hip_llm_reset_state` now clears the routed-expert cache under
  `LLM_QWEN4_RESET_MOE_CACHE=1`, so every repeat/request starts cold.
- The earlier asynchronous cold-upload pipeline lifted prefill from ~132 to
  ~147 tok/s, but its initial request-isolation fix did not remove the rare
  residual. Overlap remains opt-in; `batch4k` defaults to direct copies.

Multi-chunk prefills (`prefill > BMAX`) still need the scalar fallback or a
separate determinism fix; single-chunk 4K is a diagnostic profile, and these earlier short controls did
not justify a production or scalar-parity claim.

### Why the 200-tok/s prefill target is not reached yet

The expert working set (512 experts x 48 layers, top-10 routing) is far larger
than any cache that fits beside the batched scratch, so the routed-expert hit
rate is routing-limited, not cache-size-limited:

| Config | Prefill tok/s | Cache hit | H2D | Peak MiB |
| --- | ---: | ---: | ---: | ---: |
| 4K, BMAX=4096, cache=5000, pipeline | 147.4 | 32.4% | 124 GiB | 14,788 |
| 4K, BMAX=4096, cache=6200, direct | 136.0 | ~32% | ~124 GiB | 16,010 |
| 2K, BMAX=2048, cache=7800, pipeline | 142.9 | 25.6% | 90.6 GiB | 15,990 |

The larger-cache 2K run has a *lower* hit rate (25.6% vs 32.4%): a shorter
prompt issues fewer tokens per expert, so the same routing diversity fits less
of each expert's assignments. Raising the cache to 6,200 MiB also leaves only
294 MiB free and, with the async pipeline, reintroduces a first-repeat
divergence; it is not adopted. `LLM_MOE_STREAM_SLOTS>2` is nondeterministic
(three different hashes at slots=4), so two slots remain the deterministic
default.

Reaching 200 prefill / 30 decode needs a different expert execution or overlap
strategy, not further tuning of the current knobs. The pipeline is already
overlapping the transfer; the remaining limit is routed-expert transfer volume
plus per-expert compute.

### Grouped routed-expert investigation

Several grouped strategies were measured against the then-apparently-stable
147-tok/s single-dispatch profile; later repeats disproved that determinism claim:

- **Grouped-BF16-WMMA (`gemm_bf16_grouped`) is memory-infeasible here.** The
  all-expert staging buffer is sized `ne*N*K` bf16 (512 experts), ~15 GiB for
  this model, and Qwen4 never allocates it (`d_expw_bf16` is skipped for
  `is_qwen4exp`). Compacting it would require new dequant kernels, and the
  model is transfer-bound, so doubling weight bytes to bf16 cannot win.
- **Per-expert BF16-WMMA** (`LLM_QWEN4_NATIVE_EXPERTS=0`, a new A/B gate) was a
  wash for prefill (146--160 vs 147) and much worse for decode (13.3 vs 21.4
  tok/s, decode cache hit 65% vs 88%). It is also nondeterministic.
- **Grouped resident-only** (`LLM_MOE_GROUPED_PREFILL=1`) gave no prefill gain
  (135--148) because every layer starts with an empty per-layer cache, so there
  is nothing resident to group; it also diverged on the third repeat.
- **Staged grouped cold experts** (`--qwen4-prefill-staging`, cache 4000)
  is the only grouping that helps: median 162--180 prefill / 19.5 decode. It
  copies cold experts into double-buffered staging banks and runs one grouped
  gate-up and one grouped down launch per wave (~350 waves, 0 fallbacks at 4K).
  Two race sources were found and fixed: the per-wave staging map and task
  arrays were published with blocking `hipMemcpy`, which is not ordered against
  kernels on `r->stream` (now `hipMemcpyAsync` on `r->stream`), and the
  staged-to-cache promotion copies are now disabled under the per-request cache
  reset (`LLM_QWEN4_STAGE_PROMOTE=0`). With `HIP_LAUNCH_BLOCKING=1` the staged
  path becomes deterministic and reproduces the non-staged hash
  (`afdf60ceeb4f0103`), confirming its arithmetic is correct and the raced
  results (`ea20ffd2b071b6c3`, `b05a25a0c51bb35c`, ...) were corrupt.
  After the per-row position fix, pageable host weights
  (`LLM_MOE_REGISTER_HOST=0`) make the staged path deterministic 5/5 and it
  reproduces `afdf60ceeb4f0103`, but only at ~119 prefill / 17.7 decode --
  below the non-staged `batch4k` (149/21). With pinned host weights the
  staging-bank async copy still races its consumer (a 5-repeat run diverged),
  so the staged preset remains diagnostic and is not a net win.

The `--qwen4-prefill-staging`, `LLM_QWEN4_STAGE_PROMOTE`, and
`LLM_QWEN4_NATIVE_EXPERTS` switches are exposed for further work; the staged
path's first-request hazard is the next thing to isolate.

The shared-memory `atomicAdd(&head_sq[head], ...)` in
`fused_ssm_out_gated_q6k` is order-dependent but decode-only; the batched SSM
norm uses the deterministic tree reduction in `gated_rmsnorm_silu_batch_f32`.

## Explicitly unresolved

- Long exact-MTP I8 diverges from F16 after roughly 16 generated tokens,
  although output remains coherent. I8 remains explicit-only.
- Grouped Qwen4 prefill still has numerical/state parity failures and remains
diagnostic-only.

The runner now guards this condition in `batched_path_eligible()`: a hybrid
run with `LLM_QWEN35_BATCH_MAX_LAYER` below the final layer is routed through
the scalar top-level driver rather than entering the unsafe mixed wrapper.
The guarded cap-2 IQ3 rerun returned to rel-L2 `0.0278052`, max error
`0.3896024`, and `28.67 tok/s`, matching the normal scalar quality path.
This fixes the catastrophic partial-schedule execution failure while leaving
the unrestricted IQ3 batch experiment unchanged.
- Scalar 2K/4K runs exceeded the interactive observation window before a
  footer was captured (the 1K two-chunk control is clean); they must not be
  called failures until rerun with a persistent host/container GPU session.
  The batched results above are therefore still an explicit A/B path, not a
  default-quality claim.
- A 32K+ input / 8K+ output streamed coding run has not completed with a
  trustworthy footer and coherence gate.
- The current ROCm image has `libhipblaslt.so` but no hipBLASLt headers, so the
  200+ tok/s accelerated prefill path cannot be rebuilt here. `make -C rdna4/llm
  hipblaslt-status` reports this directly.
- FP4 KV is not implemented or aliased: the current cache formats are F16,
  symmetric I8, and real E4M3 FP8. A useful FP4 implementation would need a
  specified packed FP4 encoding and scale/error policy first; silently
  reusing the I8 byte path would not be a valid assessment.
- Persistent `/dev/kfd` access requires host/container device passthrough;
  elevated command namespaces are temporary and are not persistent access.

## Remaining tasks

- [~] Validate the bounded-wave manager across host-registration/overlap
      combinations, then compare every candidate to scalar F16 greedy hashes.
      Repeated agreement within a batched profile is insufficient for promotion.
      Multi-chunk KV/SSM/HC state carry remains separate unresolved work.
- [ ] Make the CPU cold-expert kernels bit-identical to the GPU kernels (or
      keep CPU experts opt-in only). CPU and GPU expert arithmetic currently
      differ, so expert-cache warmth changes a request's output when the CPU
      path is enabled; `batch-cpu` is diagnostic-only.
- [x] Add a repeatability gate to the streamed 512/2K/4K benchmark: run at
      least two identical requests, compare first token and sequence hash, and
      report prefill, decode, and end-to-end wall-clock tok/s together.
      Delivered as `test_hip_llm --bench-repeat N`,
      `bench_qwen38_target.sh`, the 256K/sub-32K repeat gates, and the
      `target-gate` Makefile targets.
- [ ] Complete a quality-gated 32K+ prompt / 8K+ streamed coding workload
      using 512--2048-token prefill chunks and 64--128-token decode chunks.
      Record coherence, hash/repeatability, peak VRAM, and end-to-end tok/s.
- [ ] Re-run scalar 2K/4K controls in a persistent GPU session and capture a
      complete footer; do not infer their throughput from timed-out runs.
- [ ] Obtain a ROCm image with hipBLASLt development headers, rebuild the
      accelerated prefill path, and compare it against the current scalar and
      WMMA controls without changing the quality gate.
- [ ] Continue FP8 KV long-context quality testing and measure the practical
      256K prompt path. Keep F16/I8/FP8 comparisons separate; no FP4 result is
      valid until a packed encoding and scale/error policy are specified.
- [ ] Make GPU device passthrough persistent for benchmark sessions
      (`/dev/kfd`, `/dev/dri`, `video`, and `render`) so results are not tied to
      temporary elevated namespaces.

Capacity-only results must not be reported as 256K-prompt throughput.

## Fresh stage-by-stage HIP parity (2026-09-18)

The exact 40-token coding prompt was traced with `ubatch=512` and compared
against llama.cpp's HIP trace for the same IQ2_XS GGUF and layer-0 final row.
The diagnostic used IQ2 SSM Q8_1 staging; that route remains disabled in the
production wrapper because its full recurrent-stack A/B was not stable.

| Layer-0 tensor | Relative L2 | Maximum absolute error |
|---|---:|---:|
| IQ2 SSM QKV projection | 3.70e-4 | 1.10e-2 |
| Conv + SiLU output | 1.60e-4 | 4.93e-3 |
| Gated SSM normalized output | 1.96e-4 | 2.41e-4 |
| IQ4_XS SSM output projection, F16-KV control | 5.66e-3 | 2.87e-2 |

This rules out a batch-row permutation in the IQ2 projection, convolution,
Q/K normalization, recurrence, or gated normalization. The remaining
layer-local difference is dominated by quantized IQ4_XS output accumulation;
the llama.cpp-style Q8_1 activation route improves final logits while
preserving argmax `71093`.

The independent end-to-end llama.cpp check produced the same compilable
function as the HIP runner:

```c
int clamp(int x, int lo, int hi) {
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}
```

Both extracted sources pass `gcc -Wall -Wextra -Wpedantic -std=c11
-fsyntax-only`. Evidence is in `tmp/qwen38/parity-q81-prompt.err` and
`tmp/qwen38/parity-llama/run.err`.

An independent C++17 task was also run after this trace: both implementations
selected first token `71093` and emitted the same opening code for
`bool is_prime(int n)`, including the `n < 2`, `n == 2`, and even-number
guards. The HIP run measured `35.23 tok/s` prefill and `31.22 tok/s` decode;
the matching llama.cpp ROCm run measured `19.7 tok/s` prompt processing and
`27.6 tok/s` generation. The runner text is in
`tmp/qwen38/prime-hip-text.err`, and the reference is in
`tmp/qwen38/prime-llama.out`.

## Native FFN numerical gate (2026-09-18)

The layer trace also isolated a separate RDNA4 issue in the native batched IQ2
FFN route. With `LLM_QWEN35_NATIVE_FFN=1`, layer-0 gate/up norms were
`118.62`/`103.39`, versus llama.cpp HIP values `7.22`/`6.24`; the resulting
layer output diverged substantially despite a correct first token. Disabling
only that route produced gate/up norms `10.20`/`8.68`, relative L2 errors of
`5.36%`/`5.49%`, and layer-output errors of `0.53%` at layer 0 and
`0.55%` at layer 1. The corrected IQ1 Q8_1 routes were enabled in that
diagnostic snapshot; the generic IQ1 DP4A2 path remains an explicit regression
override.

The follow-up port wired llama.cpp-style IQ1 Q8_1 kernels into the native
dispatcher through `LLM_QWEN35_IQ1S_Q81_BATCH=1` and
`LLM_QWEN35_IQ1M_Q81_BATCH=1`. These retain IQ1's affine grid correction and
quantized activation sum, which the generic two-term DP4A2 path cannot
represent. With both routes enabled, the native-FFN run returned first token
`71093`, passed the clamp task, and measured `34.79 tok/s` prefill /
`31.27 tok/s` decode. IQ2 and IQ3 DP4A A/B tests also preserved first token
`71093`; the fully enabled native configuration with native attention and both
IQ1 Q8_1 routes measured `35.30 tok/s` prefill / `31.31 tok/s` decode and
passed. The corrected combination remains an explicit opt-in; the unsafe
generic IQ1 route remains opt-in.

## Default-profile llama.cpp cross-check (2026-09-18)

After promoting the corrected IQ1 routes to the launcher defaults, the exact
40-token clamp prompt was rerun on the RX 9070 XT using the IQ2_XS GGUF,
`-b 2048 -ub 512`, flash attention, and q8/q4 KV cache. The HIP runner passed
with first decoded token `71093`, produced the expected compilable C function,
and measured `35.41 tok/s` prefill / `31.17 tok/s` decode. Peak allocation was
`15,760 MiB` of `16,304 MiB` visible VRAM.

The matching llama.cpp ROCm command selected the same first token and emitted
the same function body:

```c
int clamp(int x, int lo, int hi) {
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}
```

Its measured rate was `22.3 tok/s` prompt processing / `27.3 tok/s`
generation. The principal performance gain in the native path is keeping
batched IQ projections and hybrid SSM/attention intermediates on the GPU,
while using the llama.cpp-style Q8_1 activation contract for IQ1_S/IQ1_M.
The latter is also the principal quality fix: the old generic IQ1 DP4A2 route
omitted IQ1's affine grid correction and activation-sum term, causing large
batched FFN errors.

Run evidence: `tmp/qwen38/default-promoted.err` and
`tmp/qwen38/llama-current.out`.

The native IQ2_XS MMQ kernel was also checked at `M=160`, which selects the
long-batch WMMA/MMQ route. Its one-shot comparison against the exact local
DP4A2 reference was `1.69e-8` relative RMS, `7.15e-7` maximum absolute, and
the first four values agreed to the printed precision. This isolates the
remaining model-level drift from the MMQ arithmetic itself.

For completeness, an explicit `LLM_QWEN35_Q81_BATCH=1` A/B was added for the
IQ2/IQ3/IQ4 batched families to mirror llama.cpp's small-M Q8_1 activation
choice. It remained coherent and measured `37.00 tok/s` prefill /
`31.20 tok/s` decode, but its final-logit error increased to `0.21306` against
the saved q8/q4 llama.cpp reference. It is therefore retained as a diagnostic
switch, not promoted to the quality-safe default; the default keeps the more
accurate native F32-dequant path for those non-IQ1 projections.

## Practical RX 9070 XT ceiling and fresh output validation (2026-09-18)

The principal measured gain comes from reducing host/device traffic and launch
overhead: batched IQ projection, hybrid SSM/attention intermediates, and the
full-attention KV path stay device-resident. The corrected IQ1 Q8_1 kernels
then provide the llama.cpp activation contract for the FFN's affine IQ1
weights. This is a larger end-to-end win than the isolated MMQ arithmetic;
the MMQ A/B itself differed from the exact local DP4A2 result by only
`1.69e-8` relative RMS, so MMQ arithmetic is not the source of the model-level
quality drift.

For the 16-GiB RX 9070 XT, the reproducible quality-safe operating ceiling is
`53,248` context tokens with q8/q4 KV and `ubatch=512`. At this point the
runner reports about `544 MiB` free (`15,760 MiB` peak used), so 256K does not
fit this profile. A practical 32K-class run uses the same 512-row schedule and
measures approximately `35--42 tok/s` prefill and `31 tok/s` decode, depending
on warm-up and allocator/clock state. The validated end-to-end rate for a
40-token prompt plus 80 generated tokens is about `32--34 tok/s`. These are
quality-safe, non-MTP figures; larger rates from BF16 staging or broad Q8_1
activation experiments are performance-only diagnostics and can change later
greedy tokens.

The output was freshly checked against llama.cpp's ROCm backend using the same
IQ2_XS GGUF, ChatML prompt, greedy sampling, q8/q4 KV, flash attention,
`-b 2048 -ub 512`, and the RX 9070 XT. Both selected first token `71093` and
emitted the same compilable answer:

```c
int clamp(int x, int lo, int hi) {
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}
```

The HIP output and llama.cpp output both pass a C syntax-only compilation
check. The two runs may choose different explanatory text after the requested
function when decoding beyond the closing fence; the requested code itself is
identical and correct. Current evidence is in
`tmp/qwen38/runner-greedy.err` and `tmp/qwen38/doc-validation-llama.out`.

The fresh numerical gate also compared prompt logits, not only greedy text.
With the native batched IQ family enabled, the current runner measured
relative L2 `1.2721` versus a same-prompt llama.cpp q8/q4 ROCm oracle. Turning
off only native FFN reduced this to `0.5103`; turning off the native IQ
projection family as well reduced it to `0.22025`, with argmax `71093` in all
three cases. Prefill was `35.26`, `35.62`, and `34.99 tok/s`, respectively.
This is why the launcher now defaults `LLM_QWEN35_NATIVE_IQ2_BATCH=0` and
`LLM_QWEN35_NATIVE_FFN=0` while retaining explicit opt-ins for kernel work.
The standalone HIP-vs-CPU quantized verifier still passes all 21 types,
including IQ2_XS at `1.009e-7` relative L2; the remaining `0.22025` is thus a
full-stack dispatch/ordering difference rather than a basic IQ2 decoder error.
The promoted default was rerun cleanly with 80 decode tokens: `35.07 tok/s`
prefill, `31.22 tok/s` decode, and `32.41 tok/s` end-to-end. It selected first
token `71093`, produced sequence hash `9f3ebc7098dc2cfd`, used `15,760 MiB`
peak VRAM with `544 MiB` free, and reported `Result: PASS`
(`tmp/qwen38/hip-final-clean.out`). The llama.cpp ROCm oracle measured
`22.4 tok/s` prompt processing and `27.5 tok/s` generation for the same
prompt/model class (`tmp/qwen38/doc-validation-llama.out`).

Additional isolation runs show that F16 KV changes the error only from about
`0.2191` to `0.1870` against llama.cpp's F16-KV oracle, so KV quantization is
not the dominant source. The direct Q8/Q4 attention kernel is worse (`0.7344`)
than the packed-F16 attention route, while Q2_K's scalar warp-per-row reduction
improves the q8/q4 result from `0.22025` to `0.21892`. The Qwen3.8 launcher now
selects that scalar Q2_K path by default (`LLM_Q2K_G4=0`), with the G4 path
remaining an explicit performance override. The validated quality-default A/B
is in `tmp/qwen38/hip-q2quality-final.out`.

## Fresh llama.cpp validation and performance attribution (2026-09-18)

The current IQ2_XS path was rechecked with the same 40-token ChatML coding
prompt and greedy decoding against the local ROCm llama.cpp reference. Both
implementations selected token `71093` first and emitted the same compilable
function:

```c
int clamp(int x, int lo, int hi) {
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}
```

The practical RX 9070 XT ceiling remains about `53,248` context tokens with
q8/q4 KV and `ubatch=512`; the 256K request does not fit in 16 GiB. The
quality-safe profile has historically measured approximately `35--42 tok/s`
prefill and `31 tok/s` decode at 32K-class context. The fresh validation in
this run measured `28.58 tok/s` prefill, `25.87 tok/s` decode, and `26.71
tok/s` end to end for 40 prompt + 80 generated tokens; this run was slower
than the earlier warmed measurements but passed the same quality gate. The
llama.cpp reference text run measured about `21--22 tok/s` prompt processing
and `27 tok/s` generation on the same card and model.

The dominant performance gain is architectural: batched IQ projections,
hybrid SSM/attention intermediates, and the full-attention KV path remain
GPU-resident, reducing host/device transfers and kernel-launch synchronization.
The isolated RDNA4 IQ2 MMQ check was only `1.69e-8` relative RMS from the local
DP4A2 reference, so MMQ arithmetic is not the main source of the end-to-end
speedup. The Q8_1 activation contract for IQ1_S/IQ1_M is the important
correctness fix; the broad native IQ2 batch path remains opt-in because it
currently increases prompt-logit drift despite producing coherent short code.

The fresh HIP logits compared with the llama.cpp ROCm oracle at relative L2
`0.20194`; both argmaxes were `71093`. Validation artifacts are
`tmp/qwen38/validation-hip-text.log`, `tmp/qwen38/validation-llama.out`, and
`tmp/qwen38/validation-llama-logits.bin`.

The aligned 40-token dispatch A/B also explains why the native batch route is
not yet the quality-safe default. Enabling native batched IQ/SSM projections
gave relative L2 `0.21495`; disabling native attention and FFN while retaining
native SSM gave `0.20719`, both worse than the scalar-projection default
(`0.20194`). The batched route is functional and measured `33.3 tok/s`
prefill, but it remains an explicit diagnostic until its per-layer activation
contract reaches llama.cpp parity.

### ROCm trace correction and recurrence isolation

The original `trace-llama-qkv` layer files were generated by llama.cpp's
Vulkan backend and are not an AMD HIP parity oracle. A fresh `ROCm0` trace was
generated with `dump_llama_logits`; the authoritative quality-safe logits
comparison remains `0.20194`. The corrected helper also supports sequential
llama.cpp validation (`LLAMA_SEQUENTIAL=1`), which measures `0.21171` against
the current HIP prompt logits. The difference versus chunked llama.cpp shows
that recurrent batching and floating-point reduction order contribute to the
numerical drift.

The direct llama-style IQ2_XS single-term Q8_1 experiment measured `0.23285`
and `18.60 tok/s` prefill, versus `0.20194` for the quality-safe two-term DP4A2
route. The batched Qwen3.5 SSM Q8_1 experiment measured `0.21458` at `34.73
tok/s` prefill, while batched raw-F32 SSM measured `0.20719` at `33.35 tok/s`;
neither is promoted. These A/Bs rule out a blanket Q8_1 replacement as the
numerical fix and point the next implementation step at llama.cpp-compatible
batched recurrence reduction/state ordering.

The fused batch convolution was also changed from HIP `__expf` to accurate
`expf` to match `ggml_silu`. On the RX 9070 XT this had no measurable effect
on the 40-token A/B (`0.20719`, `33.36 tok/s`), so it is a correctness-safe
parity fix rather than the remaining drift source.

### Final llama.cpp output validation

The current source was rebuilt and the IQ2_XS GGUF was run on the RX 9070 XT
with `tmp/qwen38/coding-prompt.txt`. The runner emitted the expected clamp
function as its first generated code block; its first token was `71093`, and
the 80-token run remained coherent. The corresponding llama.cpp ROCm run on
the same model and prompt emitted the identical compilable function:

```c
int clamp(int x, int lo, int hi) {
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}
```

Reproduction commands:

```sh
env QWEN38_MODEL=/mnt/nvme02/models/qwen38/27b/gsq/Qwen3.8-27B-GSQ-RCO-IQ2_XS.gguf \
  LLM_LOGITS_PATH=tmp/qwen38/validation-hip-logits.bin \
  rdna4/llm/run_qwen38_gsq_rocm.sh --prompt-file tmp/qwen38/coding-prompt.txt \
  -n 80 --decode 80 -s 32768

env LD_LIBRARY_PATH=/opt/rocm/core-10.0/lib \
  LLAMA_ARG_N_GPU_LAYERS=99 \
  /home/syoyo/work/llama.cpp/build-codex-hetero-dev2/bin/llama-cli \
  -m /mnt/nvme02/models/qwen38/27b/gsq/Qwen3.8-27B-GSQ-RCO-IQ2_XS.gguf \
  -f tmp/qwen38/coding-prompt.txt --n-predict 80 --temp 0 \
  --seed 0 --no-display-prompt --no-show-timings
```

The saved outputs are `tmp/qwen38/validation-hip-text.log` and
`tmp/qwen38/validation-llama.out`. The llama.cpp reference measured about
`21--22 tok/s` prompt processing and `27 tok/s` generation; the quality-safe
HIP profile measured about `35--42 tok/s` prefill and `31 tok/s` decode in
warmed runs. The 16-GiB practical context ceiling remains approximately
`53,248` tokens with q8/q4 KV; 256K does not fit.

### GDA operation-order parity update

The AMD llama.cpp HIP GDN implementation uses the GDA form for this model:
it reduces the old recurrent state against `K`, multiplies that scalar by the
per-head decay, then computes the delta and updates the state. The runner had
been scaling every state element before the reduction. Although algebraically
equivalent, that changes floating-point rounding at every token. The scalar
and batched HIP kernels now use llama.cpp's order, and the batched launcher
also contains the matching 32-lane warp-per-column layout.

The batched A/B improved from `rel_l2=0.20719` to `0.20606` against the
llama.cpp ROCm logits oracle at `33.0 tok/s` prefill, while preserving the
same greedy argmax (`71093`). The quality-safe scalar profile remains the
production default at `rel_l2=0.20194`; the batched path stays opt-in until
its complete recurrent stack is at least as close as the scalar path.

The post-change HIP quantization verifier remains clean: all 21 tested types,
including `IQ2_XS`, `IQ1_S`, and `IQ1_M`, pass the `rel_l2 < 1e-4` local
dequantization gate. A 20-token prefill plus 20-token greedy decode also
completed with `Result: PASS` at `28.69 tok/s` prefill and `26.21 tok/s`
decode; the artifact is `tmp/qwen38/gda-final.out`.

### Fresh RX 9070 XT / llama.cpp validation (2026-09-18)

The current rebuilt binary was rerun against the IQ2_XS GGUF using the same
40-token ChatML coding prompt and an 80-token greedy decode. AMD access was
enabled through the HIP runner. The run completed with `Result: PASS`:

```text
Prefill:       28.83 tok/s
Decode:        26.09 tok/s
End-to-end:    26.95 tok/s
First token:   71093
Sequence hash: 9f3ebc7098dc2cfd
VRAM peak:     14112 / 16304 MiB
```

The saved HIP logits were compared with the llama.cpp ROCm logits generated
from the same GGUF, prompt, seed, and greedy settings:

```text
rel_l2=0.2019391137 max_abs=3.972079515 at=220
ours_argmax=71093 ref_argmax=71093
```

Both implementations produced the same compilable clamp function:

```c
int clamp(int x, int lo, int hi) {
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}
```

The llama.cpp reference measured `21.2 tok/s` prompt processing and `27.3
tok/s` generation for this short test. The current HIP result is therefore
quality-safe and coherent, but this short-context run does not yet reach the
long-context prefill target; the batched recurrence remains opt-in because
its measured logit drift is higher (`rel_l2=0.20606`).

### Decode-warp GDA order correction

The production `deltanet_step_warp_f32` kernel was audited against
`ggml-cuda/gated_delta_net.cu`, which is compiled as HIP by llama.cpp on AMD.
It previously decayed each state element before the key reduction. It now
reduces the old state, applies `decay * dot`, and performs
`state = decay * state + key * delta`, matching the reference operation order.

The rebuilt kernel passed the 40-token prefill and 80-token decode sanity run:
`28.81 tok/s` prefill, `26.14 tok/s` decode, `26.97 tok/s` end-to-end, first
token `71093`, and the same sequence hash `9f3ebc7098dc2cfd`. A scalar-forced
A/B (`LLM_SSM_FUSED=0`) measured `rel_l2=0.210740` versus the llama.cpp
oracle, so this operation-order change is retained for decode fidelity but is
not claimed as the source of the remaining prefill drift. The default
quality-safe prefill comparison remains `0.201939`; further numerical work
must examine projection, normalization, and attention stages rather than
assuming the recurrent kernel alone explains the difference.

### IQ1_M embedding audit

The GGUF stores `token_embd.weight` as `IQ1_M`. The runner previously
dequantized that table to F16 and used the generic embedding lookup. The
loader now keeps the raw 56-byte IQ1_M blocks and uses a dedicated GPU lookup
with the same scale, grid-index, and sign-correction reconstruction as
llama.cpp. A layer trace after this change matched llama.cpp's layer-0
normalized input at the final prompt token:

```text
HIP:   norm=72.494486 first=[1.355889 0.134475 0.133004 0.135141]
llama: norm=72.494494 first=[1.355889 0.134475 0.133004 0.135141]
```

This removes a real format-dispatch defect and avoids the host-side F16
round-trip, but it does not materially lower the final logit error. The next
confirmed divergence is the layer-0 `IQ3_S` QKV projection. The GGUF is an
IQ2_XS model overall, but this QKV tensor is explicitly `IQ3_S`; that
distinction matters when selecting and comparing the llama.cpp MMQ path. The
remaining numerical fix therefore belongs in the RDNA4 IQ3_S accumulation or
activation-layout path.

### Final cross-runtime validation note (2026-09-18)

The controlled HIP native/F32 A/B was rerun with the same 40-token ChatML
prompt and 80-token greedy decode. It completed with `Result: PASS`, first
token `71093`, and sequence hash `9f3ebc7098dc2cfd`; throughput was `24.90
tok/s` prefill, `22.82 tok/s` decode, and `23.47 tok/s` end to end. This is
slower than the production profile and did not improve the accumulated logit
error, so no kernel-selection default was changed.

The saved llama.cpp text artifact `tmp/qwen38/validation-llama.out` contains
the following clamp implementation. The HIP benchmark artifact
`tmp/qwen38/validation-hip-iq1m-final.out` contains token IDs and a sequence
hash only; it does not independently prove generated-source correctness:

```c
int clamp(int x, int lo, int hi) {
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}
```

The earlier `rel_l2=0.2019391137` comparison is historical and does not
describe this latest HIP artifact against matching q8/q4 reference caches.
The interactive llama.cpp retry exposed a thinking response and then looped
on empty input; it was terminated and is not a valid quality artifact.

### Reference audit: full QKV norm and matching KV formats

The reference trace helper truncated F32 norms to 8192 elements. QKV has
10240 elements, so its reported norm of 350.54 could not be compared with
the runner's full norm of 394.79. After removing that truncation, a fresh
ROCm trace reports **394.816155**. The supposed large QKV norm divergence
was a measurement defect; a smaller elementwise difference remains, whose
cause is not yet established.

The helper now supports `LLAMA_KV_Q8Q4=1`, rejects an unavailable requested
device, and retrieves the last output with `llama_get_logits_ith(ctx, -1)`
with a null check. The fresh reference log confirms K=q8_0 and V=q4_0 for
all 16 attention layers. Against `corrected-rocm-q8q4.bin`, the saved latest
HIP default logits measure relative L2 **0.227726068**, maximum absolute
error **4.97852182**; the native/F32 A/B measures **0.218491105**, maximum
**4.48058414**. These results supersede mismatched-cache comparisons for
these captures and do not establish numerical parity.

Artifacts: `tmp/qwen38/corrected-rocm-q8q4.err`, `.out`, and `.bin`;
helper: `tmp/qwen38/dump_llama_logits.cpp`. Runner trace dumps now keep
`ssm_norm_input` separate from gated `ssm_norm`, preventing the later tensor
from overwriting the input needed for a projection comparison.

### Matched-input IQ3_S projection validation

Fresh captures in `tmp/qwen38/parity-current/` use the same 40 prompt tokens,
256 context capacity, and Q8_0 K/Q4_0 V in our HIP runner and llama.cpp ROCm.
The input to layer-0 QKV matches at relative L2 `1.26834363e-7` (maximum
`3.81469727e-6`). Default QKV differs by `0.00265974038`. Independently
dequantizing `blk.0.attn_qkv.weight` with llama.cpp's Python GGUF code and
computing a float64 dot with the dumped reference input gives essentially
the same difference from llama.cpp (`0.00265932427`). This establishes that
the default projection difference is consistent with the reference's
activation quantization, rather than evidence of broken IQ3_S decoding.

Enabling the existing `LLM_IQ3_S_Q81_SCALAR=1` HIP port produces layer-0 QKV
relative L2 `2.2291358e-7`, maximum error `1.52587891e-5`. The complete
layer-0 SSM linear output then matches at `5.96326971e-5`, maximum
`0.000255584717`. Artifacts are in `tmp/qwen38/parity-q81/`, compared against
the reference dumps in `parity-current/`.

At layer 1 the normalized input already differs by `0.00102136506`; linear
outputs at layers 1, 2, 4, and 5 differ by `0.00454`, `0.02021`, `0.02637`,
and `0.01765`. The next investigation must isolate the FFN boundary following
the now closely aligned layer-0 SSM. The Q8_1 experiment's final-logit error
is still `0.228151724`; whole-model parity remains incomplete.

### Scalar IQ1 Q8_1 adapter and fresh FFN traces

Added scalar FFN norm/gate/up/output trace calls and an opt-in
`LLM_IQ1_Q81_SCALAR=1` adapter to the existing HIP IQ1_S/IQ1_M Q8_1
kernels. It uses the IQ1 block-sum quantizer and invalidates cached Q8x2
and IQ1 activations before overwriting their shared scratch. Both source
builds completed successfully; GPU runs completed with `Result: PASS`.

The active scalar path did not generate the old `runner-ffn-*-raw` batched
files: those names in reused directories can be stale. Authoritative new
scalar baseline files are in `tmp/qwen38/ffn-baseline.TFV4Cx/`; the IQ1
adapter capture is in `tmp/qwen38/ffn-q81.msmIRG/`. Both enable
`LLM_IQ3_S_Q81_SCALAR=1`, use the same 40-token prompt and q8/q4 cache,
and compare against `tmp/qwen38/parity-current/llama*.bin`.

| Relative L2 | Scalar baseline | IQ1 Q8_1 adapter |
|---|---:|---:|
| Layer-0 gate (IQ1_S) | 0.00939964 | 0.00280532 |
| Layer-0 up (IQ1_M) | 0.00064696 | 0.00658584 |
| Layer-0 FFN output | 0.00609821 | 0.00505948 |
| Final logits | 0.22815172 | 0.21388120 |

The adapter is diagnostic, not an established parity default. In particular,
IQ1_M up requires further comparison with llama.cpp's batched MMQ arithmetic.
The experimental binary is `rdna4/llm/tmp/test_hip_llm_iq1q81`, selected with
`QWEN38_RUNNER` when invoking the normal GSQ launcher.

### IQ1_M BLAS operand contract audit

llama.cpp's `ggml-cuda/mmq.cu::ggml_cuda_should_use_mmq` includes IQ1_S
but excludes IQ1_M. Its BLAS fallback converts the operands. A CPU replay
of layer-0 IQ1_M up using the reference FFN input measured relative L2
`0.00025804326` with F32 operands and `6.2886797e-6` with both weights and
input rounded to F16. Thus applying Q8_1 indiscriminately is not the
appropriate reference contract for this batched projection.

Added diagnostic `LLM_IQ1M_F16_OPERANDS=1` to the IQ1_M scalar kernel. It
rounds each reconstructed weight and activation to F16 before F32 dot
accumulation and takes precedence over the IQ1 Q8_1 experiment. The build
and HIPRTC compilation passed, and the 40-token GPU run completed with
`Result: PASS`. With IQ1_S Q8_1 and IQ3_S Q8_1 also enabled, the captured
up error was `0.000646961`, FFN output error `0.004582821`, and final-logit
error `0.243365056`. This is not a production improvement. The unchanged
up error relative to the F32 baseline also warrants checking actual tensor
dispatch before attributing the result to the new kernel.

Artifacts: `tmp/qwen38/ffn-half.jEtlpY/`; experimental binary:
`rdna4/llm/tmp/test_hip_llm_iq1half`. Tracing was enabled, so its throughput
is not a performance benchmark. The IQ1_S MMQ loader additionally stores
both its weight scale and affine correction in half2; that differs from
the scalar MMVQ calculation and remains a concrete next parity check.

### Preserve F16 rounding under HIPRTC fast-math

The IQ1_M F16 experiment initially did not execute its intended precision
contract: disassembly of `hip_llm_gfx1201_2132216b3c7b7f31.co` showed no
F16 conversion in `matvec_iq1_m_f32`. With exactly the dumped GPU input,
CPU replay matched F32 at relative L2 `3.6178142e-7` but differed from the
F16 operand calculation by `0.0002556681`. The Q8_1 IQ1 scale/sum kernel
also had no F16 conversions. HIPRTC uses `-ffast-math` in this configuration.

Added `round_f16_contract`, using explicit gfx1201 `v_cvt_f16_f32` and
`v_cvt_f32_f16` instructions, and used it for Q8_1 scale/sum staging and
IQ1_M operand rounding. The new code object
`hip_llm_gfx1201_bb9940eb980b1077.co` contains both conversions in both
kernels. Exact-input CPU replay now matches the F16 operand calculation at
`3.5936594e-7`, versus `0.00025598294` for F32. This verifies the precision
boundary fix independently of accumulated model error.

The diagnostic combined run (IQ3_S Q8_1, IQ1 Q8_1, IQ1_M F16 operands) passed
GPU execution with final-logit relative L2 `0.227779898`, max error
`4.69885921`; whole-model parity is still incomplete. Layer-0 up error is
`0.000777342`, gate error `0.003473889`. Artifacts are in
`tmp/qwen38/halfcontract.cgvMq1/`; binary is
`rdna4/llm/tmp/test_hip_llm_halfcontract`. Existing intrinsic-based rounding
experiments must not be considered evidence of F16 behavior without checking
the emitted instructions or an exact-input replay.

### IQ1_S MMQ half-scale port

An identical-input CPU replay of layer-0 gate using llama.cpp's reference
FFN input reproduces its MMQ output at `2.94990997e-7` relative L2 when
both the weight scale and affine correction are rounded to F16. The MMVQ
formula instead differs by `0.00051844229` on that exact input.

Added a separate HIP `matvec_iq1_s_mmq_scales` entry point, sharing the
existing IQ1_S decoding body and preserving the MMVQ entry point. Select it
with `LLM_IQ1S_MMQ_SCALES=1 LLM_IQ1_Q81_SCALAR=1`. The new kernel uses
explicit F16 conversion instructions for both coefficients, matching
`ggml_cuda_mmq_load_tiles_iq1_s` in the reference.

The build and HIPRTC compilation passed. Exact-input replay of the new GPU
gate output matches the MMQ formula at `8.14008789e-8`, versus
`0.00051858564` for MMVQ. The combined model diagnostic completed with
`Result: PASS`; final-logit relative L2 improved from `0.227779898` to
`0.220439014`, maximum error `4.28501511`. Its remaining layer-0 gate error
against the reference (`0.003432994`) includes upstream input differences.
Artifacts: `tmp/qwen38/iq1smmq.fcaFBF/`; replay script:
`tmp/qwen38/replay_iq1s.py`; binary: `rdna4/llm/tmp/test_hip_llm_iq1smmq`.

The reference MMQ activation layout is type-dependent: IQ1_S uses DS4
(F16 scale/sum pairs), whereas IQ2 and IQ3_S use D4 (F32 scales). A blanket
Q8_1/F16 scale contract cannot reproduce every batched projection. That
dispatch distinction remains to be implemented and validated across the
active Qwen3.8 stack; whole-model parity is not established.

### IQ3_S MMQ D4 activation staging

Added `quantize_mmq_d4_batch`, sharing the Q8 quantization body but retaining
F32 activation scales rather than MMVQ's F16 scales. Diagnostic
`LLM_IQ3S_MMQ_D4=1` selects it within `LLM_IQ3_S_Q81_SCALAR=1`. This
dispatch also invalidates cached Q8x2/IQ1 scratch metadata when overwriting
the shared buffers.

The build, HIPRTC compilation, and 40-token GPU run passed. With IQ1_S MMQ
scales and IQ1_M F16 operands enabled, layer-0 QKV again matches llama.cpp
at relative L2 `2.2291358e-7`, maximum error `1.52587891e-5`. Layer-0 SSM
linear output differs by `9.65191479e-5`; gate by `0.00280571253`; final
logits by `0.229491043` (maximum `4.51307774`). This verifies the IQ3_S
projection contract but is not a whole-model improvement. The remaining
IQ4/IQ2 activation/accumulation contracts still need alignment.

Artifacts: `tmp/qwen38/d4.HTkLiE/`; binary:
`rdna4/llm/tmp/test_hip_llm_d4`. Comparison reference remains the matching
q8/q4 capture under `tmp/qwen38/parity-current/`.

### D4 dispatch for IQ2_XXS, IQ2_XS, and IQ4_XS

Added diagnostic `LLM_IQ_MMQ_D4=1` to the existing scalar Q8 routes for
IQ2_XXS, IQ2_XS, and IQ4_XS. They now share explicit selection between
MMVQ F16 scales and MMQ F32 scales, with shared activation-cache invalidation.
The associated `LLM_IQ2_XXS_Q81_SCALAR=1` and
`LLM_IQ2_XS_Q81_SCALAR=1` switches enable the IQ2 routes; the GSQ launcher
already enables the IQ4_XS route. This is a diagnostic dispatch change,
not a production-profile promotion.

Build and 40-token q8/q4 GPU execution passed. Combined with the previously
validated IQ3_S D4 and IQ1 contracts, final-logit relative L2 decreased from
`0.229491043` to `0.209229436`, max absolute error `4.33789921`. Layer-0
SSM linear output differs by `5.96326971e-5`; layer-0 FFN output by
`0.00280848`; layer-1 normalized input by `0.000521503329`. Full model
parity remains incomplete. The model mixes IQ1/IQ2/IQ3/IQ4 and K-quants,
so remaining types and fused dispatches require independent validation.

Artifacts: `tmp/qwen38/iqd4.qlIMRN/`; binary:
`rdna4/llm/tmp/test_hip_llm_iqd4`. These are tracing runs, not speed benchmarks.

### IQ2_S D4 and first full-attention boundary

Layer 3 uses IQ2_XXS Q/K, IQ2_S V, IQ1_S attention output, and Q2_K
FFN down. Added diagnostic `LLM_IQ2S_MMQ_D4=1` to stage a single Q8 term
with F32 scales and invoke the existing IQ2_S integer-dot kernel. The path
invalidates shared activation caches and leaves production dispatch unchanged.

The build and matched 40-token q8/q4 GPU run passed. Layer-3 normalized
input error is `0.00240962053`; layer-3 output error decreased from
`0.0115546716` to `0.00976271774`. Final-logit error instead increased from
`0.209229436` to `0.227679896`, maximum `4.69716358`. Thus this is not a
production improvement or proof of complete IQ2_S parity. The remaining
amplification within full attention still requires a projection/attention
boundary comparison with identical inputs.

Artifacts: `tmp/qwen38/iq2sd4.PFojhQ/`; binary:
`rdna4/llm/tmp/test_hip_llm_iq2sd4`. All earlier MMQ precision diagnostic
switches were enabled for this A/B.

### Partial M-RoPE pairing fix

Full layer-3 Q/K/V dumps, including llama.cpp tensor strides, isolated a
concrete model-execution defect. Qwen3.5 has head_dim=256 but n_rot=64
(sections 11+11+10). All three HIP M-RoPE kernels paired j with j+128,
using half the head width. llama.cpp's `rope_multi` pairs j with j+n_rot/2,
or j+32. Corrected scalar, batched, and device-position kernels to use
the section sum and leave dimensions outside the rotary span unchanged.

Before this fix, full layer-3 Q and K relative errors were `0.36544713`
and `0.36452046`. Afterward they are `0.007236128` and `0.006725294`.
The attention CPU replay itself agreed with the pre-fix GPU output at
`0.000404447` across tokens, ruling out a large softmax error on those
inputs. The incorrect rotary pairing was upstream of attention.

With the same MMQ diagnostic switches, final-logit error fell from
`0.227679896` to `0.137225625` (max `1.7642827`); layer-3 output error
fell from `0.009762718` to `0.006372971`. With the normal launcher profile,
the corrected binary measures `0.16579937` (max `1.83065367`) against the
same q8/q4 llama.cpp reference. The 40+80 normal-profile run passed at
28.53 prefill / 25.82 decode tok/s, first token 71093, sequence hash
`611f53244661899c`. The changed hash is recorded, not treated as proof of
text or source-code correctness. Whole-model parity remains incomplete.

Artifacts: `tmp/qwen38/attnreplay.mOmaXi/` (both runtimes' attention tensors),
`tmp/qwen38/ropefix.FdOomJ/` (corrected diagnostic and default runs), and
`tmp/qwen38/replay_attention.py`. Tested binary:
`rdna4/llm/tmp/test_hip_llm_ropefix`. Build and HIPRTC compilation passed.

### Fresh generated-output validation against llama.cpp HIP

Used the corrected `tmp/test_hip_llm_ropefix` binary with the normal GSQ
launcher profile and the same IQ2_XS GGUF in both runtimes. Both consumed
the literal 40-token `tmp/qwen38/coding-prompt.txt` ChatML prompt without
extra BOS, used ctx=256, q8_0 K / q4_0 V, and greedy non-MTP generation
with a 160-token cap. The reference explicitly loaded `libggml-hip.so`
and selected ROCm0, with ubatch=512. Runs were sequential on the GPU.

Both generated exactly this function inside a C code fence:

```c
int clamp(int x, int lo, int hi) {
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}
```

Each extracted function compiled independently with
`g++ -std=c++17 -O2 -Wall -Wextra -Werror -fsanitize=undefined`.
Both passed 196 input/bound combinations drawn from INT_MIN, -100, -1, 0,
1, 100, INT_MAX, restricting bounds to lo<=hi. No sanitizer error occurred.
This is an actual compile-and-run check, but only a small C-compatible C++
test, not validation of a substantial C++ generation task or 32K output.

Output behavior still differs: llama.cpp emitted the fenced function and
reached EOG; our runner emitted the same function followed by unsolicited
explanation, cut off at the generation cap. Thus extracted-code correctness
passes while strict “only compilable C code” formatting fails (both have
fences), and continuation/stopping parity is not established.

Fresh prompt-logit comparison over 248,320 entries reproduces relative L2
`0.16579937`, maximum absolute difference `1.83065367` at index 1001.
The matching first token is 71093; it does not imply distribution parity.
Runner performance in this text-printing validation run: 28.58 tok/s prefill,
25.63 tok/s decode, 26.17 tok/s end-to-end, 9412 MiB peak VRAM. No reference
throughput claim is made from the diagnostic helper.

Artifacts: `tmp/qwen38/output-validation.KqHa3A/` contains each raw output,
logits, decoded generated text, and separately compiled `*-clamp.cpp` tests.
The runner prints byte-alphabet token strings; the validator reverses that
alphabet before extracting code. Re-run extraction/compilation/tests with:

```sh
python3 tmp/qwen38/validate_clamp_output.py tmp/qwen38/output-validation.KqHa3A
tmp/qwen38/compare_f32 tmp/qwen38/output-validation.KqHa3A/ours.bin tmp/qwen38/output-validation.KqHa3A/llama.bin
```

Generation used `LLM_GEN_TEXT=1`, `--prompt-file tmp/qwen38/coding-prompt.txt
-n 40 --decode 160 -s 256` with `QWEN38_RUNNER` pointing to the corrected
binary. The reference helper `tmp/qwen38/dump_llama_logits_generation`
was built from `dump_llama_logits.cpp` and run with `LLAMA_KV_Q8Q4=1`,
`LLAMA_GENERATE=160`, and `LD_LIBRARY_PATH=/opt/rocm/core-10.0/lib`.
These helpers/artifacts are repository-local temporary files, not installed
production tools. No long-context performance or quality qualification was
performed in this validation.

### Full-layer localization and IQ2_XXS MMQ scale correction

Extended diagnostic residual-stream dumps to all 64 scalar/batched layers.
The matched HIP IQ2 run (40 tokens, ctx=256, q8/q4 KV, MMQ precision
diagnostics enabled) has last-token residual relative errors 0.000188576
at layer 0, 0.000564150 at layer 1, 0.004051949 at layer 2,
0.006372971 at layer 3, and 0.116808219 at layer 63. Thus substantial
divergence exists before the final output head. An independent layer-0
residual/RMSNorm replay matches at 5.45e-8 relative error; the early
normalization is not itself the defect. These comparisons describe
last-token states, not all-token equivalence.

Source inspection found a genuine mismatch in the IQ2_XXS diagnostic MMQ
route. llama.cpp `ggml-cuda/vecdotq.cuh` MMVQ uses integer `sumi*ls/8`,
whereas `ggml-cuda/mmq-load-tiles.cuh::ggml_cuda_mmq_load_tiles_iq2_xxs`
stores a floating-point `d*ls/8` scale. Our D4 diagnostic had changed the
activation precision but retained the MMVQ integer truncation. Added
`matvec_iq2_xxs_mmq_scales`, sharing the integer-dot body but applying the
MMQ floating scale. `LLM_IQ_MMQ_D4=1` selects it in the scalar IQ2_XXS
Q8 route; ordinary Q8_1/MMVQ and production defaults are unchanged.
This adapts the reference arithmetic contract; it is not a new MFMA port.

On identical captured layer-0 gate/up operands, independent GGUF
dequantization plus D4 replay improves from rel_l2 4.50233e-5 to
1.79405e-7 (max 7.91624e-9). The replay reconstructs SiLU on CPU rather
than capturing exact GPU quantization bytes. The repository test source
`test_iq2xxs_mmq_replay.py` checks shapes, finite values and rel_l2 <= 1e-6,
and dequantizes in 64-row chunks. Run it using:

```sh
env PYTHONPATH=/home/syoyo/work/llama.cpp/gguf-py OPENBLAS_NUM_THREADS=4 \
python3 rdna4/llm/test_iq2xxs_mmq_replay.py \
/mnt/nvme02/models/qwen38/27b/gsq/Qwen3.8-27B-GSQ-RCO-IQ2_XS.gguf \
tmp/qwen38/iq2xxsmmq.RtpHrh
```

With all prior MMQ precision diagnostics held constant, final-logit
relative L2 improves from 0.137225625 to 0.124566065; maximum error falls
from 1.7642827 to 1.67864299. This is not uniformly monotonic by layer:
layer-3 relative error increases to 0.007505874 while layer-63 decreases
to 0.114460416. Full parity remains incomplete. Build and HIPRTC compilation
passed; these traced runs are not speed benchmarks.

Artifacts: `tmp/qwen38/layertrace.lrseKa/` (all-layer reference and pre-change
captures, `layers.csv`); `tmp/qwen38/iq2xxsmmq.RtpHrh/` (corrected captures,
logits and replay); binary `rdna4/llm/tmp/test_hip_llm_iq2xxs_mmq`.

### IQ3_XXS model baseline with corrected M-RoPE

Also tested `Qwen3.8-27B-GSQ-RCO-IQ3_XXS.gguf` from the same GSQ directory,
not merely the IQ3 tensors inside the IQ2 model. Both runtimes used AMD HIP,
the same 40-token literal ChatML prompt, ctx=256 and q8_0 K / q4_0 V for
the 16 full-attention layers. Our run used the normal GSQ profile with the
corrected binary; MMQ precision experiments were not enabled.

Both final-logit arrays are finite and select token 71093, but relative L2
is 0.104863825 and maximum error is 1.09923315. Our 40-token prefill measured
29.43 tok/s and 13076 MiB peak VRAM. This confirms small-context execution
and quantifies the IQ3 model's remaining numerical gap; it does not validate
generated continuation, long context, or a 53K memory limit for IQ3.
Artifacts: `tmp/qwen38/iq3-parity.r6F0sf/`.

### IQ2_XS MMQ scale contract and fusion control

The matched IQ2 diagnostic with `LLM_SSM_FUSED=0` worsened final-logit
relative L2 from 0.124566065 to 0.157814132 (maximum error 2.37050724).
This does not establish that all fused kernels are correct, but disabling
the entire fused family is not a demonstrated quality improvement.
The unfused capture is `tmp/qwen38/unfused-parity.47UXdS/`.

Found the analogous MMQ/MMVQ contract distinction in IQ2_XS. Reference
`ggml-cuda/vecdotq.cuh` combines sub-block dots with integer division,
whereas `ggml_cuda_mmq_load_tiles_iq2_xs` in `mmq-load-tiles.cuh` uses two
F32 scales `((ls*d)+d/2)/4`. Added `matvec_iq2_xs_mmq_scales`, sharing the
integer-dot body and selected only by `LLM_IQ_MMQ_D4=1` within the scalar
IQ2_XS Q8 diagnostic. The existing Q8_1/MMVQ kernel retains its arithmetic;
production defaults and native batched dispatch are unchanged.

Extended `test_iq2xxs_mmq_replay.py` with `--layer 1` to validate the
model's IQ2_XS FFN-down projection. The old capture fails the 1e-6 relative
threshold at 6.03832232e-5 (max 1.38282776e-5); the new capture passes at
3.43140101e-7 (max 4.47034836e-8). This uses independent GGUF dequantization
and F32 D4 scales, with CPU-reconstructed SiLU. Build, HIPRTC compilation,
and the 40-token GPU run passed.

Despite that local contract improvement, whole-model relative L2 increases
from 0.124566065 to 0.154447965 (max 2.24026585). Do not promote this
diagnostic configuration or call it an end-to-end quality improvement.
Remaining projection contracts and recurrent error propagation still need
isolation. Artifacts: `tmp/qwen38/iq2xsmmq.SIDq6W/`, including negative and
positive replay results; binary `rdna4/llm/tmp/test_hip_llm_iq2xs_mmq`.
A second independently initialized GPU run reproduces all 248,320 final
logits bit-for-bit (`cmp` passes, relative L2 and max error both zero).
Thus the observed change is repeatable in this test, not evidence of a
nondeterministic reduction. This is not an exhaustive race-freedom proof.

### Matched single-token reference and trace safety

Compared llama.cpp HIP batched prefill with 40 sequential single-token
`llama_decode` calls on the same IQ2 GGUF, prompt, ctx=256 and q8/q4 KV.
The reference itself differs by relative L2 0.0851652646 (max 1.51741648).
Different MMQ/MMVQ dispatch and recurrent scheduling must therefore be
controlled when interpreting whole-model metrics. This is not an acceptable
error bound for our implementation and does not establish our correctness.

Setting `LLM_BENCH_STREAM_CHUNK=1` makes our runner process the same prompt
one token at a time. Against the sequential reference, the normal profile
measures rel_l2 0.154278191 (max 1.57382536). Enabling the existing
`LLM_IQ2_XXS_Q81_SCALAR=1 LLM_IQ2_XS_Q81_SCALAR=1
LLM_IQ1_Q81_SCALAR=1 LLM_IQ3_S_Q81_SCALAR=1` routes, without MMQ/D4
or IQ1_M F16 diagnostic overrides, reduces this to 0.118834458 (max
1.48700833). These are separate scheduling-matched controls, not directly
comparable improvements over the batched-reference MMQ numbers above.

The first sequential reference trace crashed after the first token. The
temporary tracing helper indexed empty selected-output tensors produced by
batches requesting no logits. Added zero-element/byte guards, validated
row-span bounds, and sized F16 captures by byte span rather than element
count. The corrected helper completes all 40 tokens. Both runtimes' traced
final logits are bit-identical to their respective untraced controls.
The failed partial trace is not used for comparisons.

Matched last-token layer-0 comparisons now show normalized input rel_l2
1.26834e-7, QKV 6.77102e-8, gated recurrent output 4.36003e-6, output
projection 1.08131e-5, FFN norm 8.43442e-5, and FFN output 0.00192009.
Residual-stream relative errors are 0.000121349 at layer 0, 0.001160163
at layer 1, 0.003426266 at layer 2, and 0.123220161 at layer 63.
The early QKV contract is close; recurrent and FFN propagation still needs
same-input isolation. No production dispatch was changed in this check.

Artifacts: `tmp/qwen38/reference-sequential.smxsUz/` (untraced controls;
its `llama-traced` files are from the failed attempt),
`tmp/qwen38/sequential-trace.SKwDsN/` (valid matched captures and
`layers.csv`), and `tmp/qwen38/dump_llama_logits_safe_trace` (helper binary).

### Same-input IQ1_M replay and scalar GDN reference adapter

Replayed layer-0 IQ1_M FFN-up independently for each runtime's captured
FFN-normalized input. The replay dequantizes the GGUF weights, rounds the
activations with roundf-equivalent Q8 quantization, stores the scale as F16,
and computes the resulting dot products. Agreement with each GPU output is
3.26943e-7 relative L2 for our runner and 3.22359e-7 for llama.cpp HIP.
The difference between the two replay outputs is 0.00084029866, matching
the captured output difference 0.00084030005. For this tensor and prompt,
the IQ1_M discrepancy is therefore explained by different inputs, not a
demonstrated weight-decoding/activation-contract defect. This is not a
whole-model IQ1_M qualification. Replay: `tmp/qwen38/replay_iq1m_matched.py`;
results: `tmp/qwen38/sequential-trace.SKwDsN/iq1m-replay.txt`.

Added diagnostic `LLM_QWEN35_GDA_REF_SCALAR=1` to call the existing
`deltanet_step_batch_gda_ref_f32` with M=1, contiguous scalar inputs,
and the llama.cpp 32-lane-per-column / four-columns-per-block launch.
The default scalar kernel uses four lanes per column. The adapter is gated
to hybrid non-Qwen4Exp models with d_state=128 and leaves defaults unchanged.

Build and matched 40-token, one-token-at-a-time HIP execution passed.
Layer-0 gated recurrent-output relative error barely changes:
4.3600285e-6 -> 4.358825e-6. Layer-0 output-projection and final layer-output
captures retain their prior relative errors (1.0813135e-5 and 0.00012134939).
Layer-1 output improves slightly (0.0011601627 -> 0.0011510316), but final
logit error worsens from 0.118834458 to 0.13447731, max 2.15960455.
This does not justify production promotion. The reference recurrence
reduction layout alone does not remove the early discrepancy; next isolate
its convolution, normalized Q/K, and decay/beta inputs on identical data.

Artifacts: `tmp/qwen38/scalar-gda.rkhUnW/`; tested binary:
`rdna4/llm/tmp/test_hip_llm_scalar_gda`. Other flags and reference are the
matched sequential Q8_1 configuration in the preceding section. These
traced runs are numerical experiments, not speed benchmarks.

### GDN boundary capture and layer-0 output projections

The unfused scalar GDN path was captured after adding complete Q/K-normalized
and alpha/beta diagnostics. The reference helper was corrected to dump all
2048 values of each 128x16 Q/K tensor rather than only the selected 128-value
row. With the same first-layer inputs, runner versus llama.cpp agrees at
1.04653e-7 for normalized Q and 1.24706e-7 for normalized K. Raw layer-0
convolution agrees at 7.27791e-8 relative L2 (max 3.8147e-6). This rules out
the layer-0 convolution and L2-normalization implementation as the source of
the observed model gap. Later GDN boundaries diverge because their hidden
inputs already differ.

The layer-0 SSM output tensor is IQ4_XS, not IQ3_XXS as an earlier tensor-type
assumption suggested. Independent GGUF dequantization and same-input replay
of its 5120-row projection gives 0.00182647 relative error for the runner and
0.00182518 for llama.cpp HIP, with maxima 0.00291061 and 0.00281143. Their
captured projection difference is 1.08131e-5, while their inputs differ by
4.35464e-6. This is comparable reference-side quantized projection error,
not evidence for a runner-only IQ4_XS decoding defect. Replay source:
`tmp/qwen38/replay_iq3xxs_projection.py` (despite its historical filename).

The layer-0 FFN-down tensor is IQ2_XXS. Same-input replay gives a larger
0.00455387 relative error for the runner and 0.00451478 for llama.cpp, with
maxima 0.00018925 and 0.00017849. The captured FFN-output difference is
0.00192009 and the FFN input difference is 0.00106090, so this projection
also does not yet isolate a runner-only kernel error. Replay source:
`tmp/qwen38/replay_q2k_projection.py` (renamed historically; it now checks
IQ2_XXS). Both scripts use chunked GGUF dequantization and CPU reconstruction
of SiLU, so their numerical tolerance is diagnostic rather than a bitwise
GPU test.

Artifacts for the complete unfused boundary run are in
`tmp/qwen38/gdn-final.YEXl7t/`; the corrected full-reference tensors are in
`tmp/qwen38/gdn-ref-full.vn7zyF/`. This work remains HIP-only and uses no
Vulkan path. No production dispatch was promoted from these experiments.

The additional unfused alpha capture shows layer-0 raw alpha close to the
llama.cpp HIP trace: relative L2 2.22358e-5 and maximum 4.57048e-4 at the
last prompt token. This is small compared with later residual-stream growth
and is consistent with BF16/quantized projection accumulation. The llama.cpp
graph build does not emit a beta boundary tensor, so no beta parity claim is
made. Artifacts: `tmp/qwen38/auxtrace.MNTejp/`.

### Validated Q8_1 quality-default promotion

Ran the actual `run_qwen38_gsq_rocm.sh` launcher with the four
llama.cpp-compatible scalar adapters enabled:
`LLM_IQ2_XXS_Q81_SCALAR=1`, `LLM_IQ2_XS_Q81_SCALAR=1`,
`LLM_IQ1_Q81_SCALAR=1`, and `LLM_IQ3_S_Q81_SCALAR=1`. The matched 40-token
IQ2 run measured relative final-logit error 0.118233166 (max 1.43055367),
versus 0.154278191 (max 1.57382536) in the scheduling-matched normal
profile. It retained argmax token 71093 and measured 29.51 tok/s prefill;
VRAM peak was 9412 MiB. The launcher now enables these four adapters by
default, with each still individually overridable. This is a quality-profile
change, not a native-MFMA performance change. The IQ2 D4/MMQ experiments
remain opt-in because IQ2_XS increased whole-model error.

Shell syntax and dry-run profile validation passed. Artifact:
`tmp/qwen38/q81-launcher.uZa6UB/`.

### BF16 SSM weight A/B and llama.cpp validation

The SSM alpha/beta tensors are stored as BF16 in the GGUF. The runner's
historic production contract converts those weights to F16 before the F16
matvec; llama.cpp's graph keeps the tensors BF16. I added a native-BF16
matvec and batch matvec behind `LLM_BF16_NATIVE=1` so this contract can be
tested without changing the default profile. Native BF16 is intentionally
diagnostic-only: on the matched IQ2, 40-token, one-token-at-a-time run it
produced final-logit relative L2 0.129786099 (max 1.78880739), while the
default F16-converted path produced 0.118834458 (max 1.48700833). The native
path therefore does not match the installed llama.cpp HIP behavior closely
enough to promote, despite matching the nominal storage type.

The default remains F16-converted for quality and is selected when
`LLM_BF16_NATIVE` is unset or zero. The matched default run passed with the
same greedy top token (71093), measured 28.02 prefill tok/s, and used 9386
MiB peak VRAM. This is a numerical-contract finding, not evidence that the
BF16 kernel is unusable on other llama.cpp/ROCm revisions.

For the validated llama.cpp comparison, both runners used the same IQ2 model,
same prompt, q8 K/q4 V KV cache, and sequential one-token prefill. The
llama.cpp batched and sequential reference executions themselves differ by
0.0851653 relative L2, so the comparison is not expected to be bitwise. The
runner's default quality profile is 0.1188345 relative L2 against the
sequential reference, with matching argmax token; the launcher promotes only
the four validated scalar Q8_1 adapters and leaves MMQ D4 and native BF16
experiments opt-in. All tests in this section use AMD HIP on gfx1201 and no
Vulkan path.

The final post-documentation check reran `tmp/qwen38/dump_llama_logits_generation`
with `LD_LIBRARY_PATH=/opt/rocm/core-10.0/lib`, `LLAMA_KV_Q8Q4=1`, and
`LLAMA_SEQUENTIAL=1` against the same IQ2_XS model and `coding-prompt.txt`.
The current runner logits versus that fresh llama.cpp ROCm output measured
relative L2 `0.118233166`, maximum absolute error `1.43055367`, and the same
top token `71093`. The llama.cpp result decoded the expected compilable
`clamp` implementation with lower/upper bound checks and `return x`; the
runner's matched coding validation also completed with `Result: PASS`.
Artifacts: `tmp/qwen38/final-llama-check/` and
`tmp/qwen38/q81-launcher.uZa6UB/`.

### IQ3 quantizer gate

The current HIP binary's `--verify-quant-kernels` check was rerun on the IQ3
XXS model. All 21 tested formats passed the CPU-dequantized matvec check;
the relevant low-bit results were IQ2_XXS `4.114e-8`, IQ2_XS `1.009e-7`,
IQ3_XXS `7.133e-8`, IQ3_S `9.846e-8`, and IQ1_M `1.340e-7` relative L2.
This rules out a basic block decoder error for the IQ2/IQ3 formats. The
remaining IQ3 end-to-end discrepancy must be isolated at the model graph,
activation contract, or reduction/attention boundary rather than by changing
the block decode tables. The test used AMD HIP on gfx1201 and returned
`21 PASS, 0 FAIL, 0 SKIP`.

An opt-in scalar IQ3_XXS Q8_1 adapter was then added to test the same
llama.cpp activation contract on the IQ3_XXS model. It is selected only by
`LLM_IQ3_XXS_Q81_SCALAR=1`; the launcher does not enable it. The matched
40-token A/B with the other validated scalar adapters enabled measured
`1.03789725` relative logit L2 (maximum `10.3967686`) against the fresh
llama.cpp ROCm reference, although it ran at `34.12 tok/s` prefill and passed
the finite-output gate. This is a clear whole-model regression, so the
IQ3_XXS Q8_1 route remains diagnostic-only. Passing isolated block decode
tests is not sufficient to promote an activation quantization contract when
its error compounds through the hybrid stack.

### Fresh llama.cpp output validation

The documented IQ2 quality profile was rechecked against a fresh llama.cpp
ROCm run. Both executions used `Qwen3.8-27B-GSQ-RCO-IQ2_XS.gguf`,
`coding-prompt.txt`, q8 K/q4 V KV cache, sequential evaluation, and AMD
gfx1201. The runner completed with finite logits and the same greedy top token
as llama.cpp (`71093`). Final logit relative L2 was `0.118233166`, with
maximum absolute error `1.43055367`; runner prefill was `29.51 tok/s` and
peak VRAM was `9412 MiB`.

The generated result was coherent and compiled as the expected bounded C++
helper:

```c
int clamp(int x, int lo, int hi) {
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}
```

An IQ3_XXS A/B was also run against a fresh llama.cpp ROCm trace. The default
direct IQ3 path remains the reference candidate; an experimental scalar
Q8_1 adapter selected with `LLM_IQ3_XXS_Q81_SCALAR=1` regressed to relative
L2 `1.03789725` (maximum `10.3967686`) despite passing the finite-output
check. It is therefore left disabled in the launcher. Artifacts are under
`tmp/qwen38/final-llama-check/`, `tmp/qwen38/iq3-llama-trace-safe/`, and
`tmp/qwen38/iq3-q81-ab/`.

The latest clean rerun used fresh artifacts in `tmp/qwen38/validation-llama/`
and `tmp/qwen38/validation-runner/`. It measured runner prefill at `35.53
tok/s` (40 tokens, 1125.90 ms), with `9406 MiB` peak VRAM and `Result: PASS`.
Compared with the fresh llama.cpp logits, the runner measured relative L2
`0.104475621`, maximum absolute error `1.07564211`, and identical argmax
token `71093`. The llama.cpp greedy continuation again began with a fenced
code block containing `int clamp(int x, int lo, int hi)`, matching the
expected C++ task and confirming coherent output. The earlier `0.118233166`
value remains a valid comparison against a different reference run; the
small change is attributable to the reference/runtime execution path, not a
source change between these two measurements.

### IQ3 mixed-model adapter A/B

The IQ3_XXS model was rerun with only the IQ2 Q8_1 adapters disabled; all
other quality-profile settings were held constant. Its layer-0 recurrent
boundaries remained close to llama.cpp (SSM norm-input `1.875e-4` relative
L2 and gated SSM output `1.938e-4`), while the full 64-layer trace grew to
`6.05e-2` at layer 63. The fresh final-logit comparison measured relative
L2 `0.07714396`, maximum absolute error `0.7172557`, and the same argmax
token `71093`.

An IQ3 fused-GDN A/B with `LLM_IQ2_XS_Q81_SCALAR=0` regressed to relative
L2 `0.0390024631` (maximum `0.455090404`) at `26.86 tok/s`. Thus the residual
improvement comes from the llama.cpp-compatible IQ2_XS Q8_1 adapter, while the
fused GDN change supplies both IQ3 speed and the correct recurrent reduction
order. The validated IQ3 production combination is therefore fused GDN plus
selective IQ2_XS Q8_1.

This is better than enabling both IQ2 Q8_1 adapters on the mixed IQ3 model.
The later fresh sequential ROCm oracle rerun refined that policy: the launcher
still keeps `LLM_IQ2_XXS_Q81_SCALAR=0`, but enables the validated
`LLM_IQ2_XS_Q81_SCALAR=1` selectively for `IQ3_XXS`; fused GDN is also the
IQ3 default. Users can still set either variable explicitly for kernel A/B
testing. The pure IQ2 profile keeps both adapters direct-F32 by default. The
A/B artifact is
`rdna4/llm/tmp/iq3-no-iq2q81/`.

A pure-IQ2 wrapper rerun after this policy change still selected the scalar GDN
and direct-F32 IQ2_XXS/IQ2_XS adapters, measuring `27.25 tok/s` prefill and
relative logit L2 `0.0890298896` against the IQ2 llama.cpp oracle. This confirms
the IQ3-only fused default does not perturb the IQ2 profile.

### IQ4_XS Q8_1 DP4A contract audit (2026-09-18)

The active IQ2 layer-0 trace localizes the first projection mismatch after GDN:
the normalized recurrent output is within `0.001115` relative L2 of llama.cpp,
while the following `IQ4_XS ssm_out` projection is approximately `0.109`
relative L2 in the saved trace. llama.cpp's HIP implementation uses the
`vec_dot_iq4_xs_q8_1` contract. The runner already had a scalar Q8_1 path; an
audit of its opt-in DP4A sibling found an actual packing error inherited from
the older batch experiment. Its code words were assembled from `qs[j]`,
`qs[j+4]`, etc., instead of four contiguous bytes matching llama.cpp's
`get_int_b4`.

The corrected scalar-row DP4A kernel is available behind
`LLM_IQ4_XS_Q81_DP4A_SCALAR=1`. On the exact IQ2 prompt it is bit-identical to
the existing scalar Q8_1 route (`0.0890298896`, `27.38 tok/s` versus
`27.25 tok/s`), so it is not promoted as a quality change; it is retained as a
correct RDNA4 A/B implementation. The active projection mismatch therefore
cannot be fixed by that packing correction alone, and the next target remains
the full layer-level MMQ accumulation/layout comparison.

The fresh direct-policy trace also shows why the pure-IQ2 end-to-end error is
not an IQ4_XS output-only problem. Layer-0 SSM QKV is `0.00266087623` relative
L2 against the llama.cpp HIP trace, while convolution is `0.000445146864`,
GDN-normalized output is `0.00111534294`, and the correctly paired
`linear_attn_out` projection is `0.000802041604`. A CPU dequantized replay of
the same IQ3_S QKV tensor agrees with the runner's F32 projection at
`1.20e-5`, indicating that the remaining QKV difference is specifically the
llama.cpp HIP prefill quantization/accumulation choice rather than a bad IQ3_S
weight decoder.

The IQ3_S Q8_1 A/B was then corrected to one Q8_1 block per 32-value
sub-block, matching llama.cpp's `VDR_IQ3_S_Q8_1_MMVQ=2` traversal. The old
half-split form indexed the second half past the eight-byte IQ3_S code group.
The corrected A/B measured `0.0940144817` final relative L2 on the pure-IQ2
40-token prefill, worse than direct F32 `0.0890298896`; it remains opt-in.
The same fix was applied to the legacy IQ3_S DP4A macro, so both experimental
ports now have valid sub-block addressing without changing production defaults.

The remaining scalar adapters were swept independently. On IQ3_XXS, disabling
IQ1 Q8_1 as well improved layer-63 trace error from `0.060507` to `0.059655`;
disabling IQ3_S Q8_1 too improved it further to `0.055624`. Disabling IQ4_XS
as well regressed slightly to `0.056091`, so IQ4_XS remains enabled for IQ3.
The launcher now applies this selective IQ3 profile by default: IQ4_XS on,
IQ2_XXS/IQ2_XS/IQ1/IQ3_S off. Every setting remains explicitly overridable
through its environment variable. The corresponding A/B artifacts are
`rdna4/llm/tmp/iq3-no-iq2-iq1q81/`,
`rdna4/llm/tmp/iq3-no-mixed-q81/`, and
`rdna4/llm/tmp/iq3-no-all-q81/`.

### Stage-synchronization diagnostic

The IQ3 discrepancy was checked for an asynchronous intermediate-buffer
hazard. `LLM_QWEN35_SYNC_LAYERS=1` was insufficient: it inserts a device
barrier only after each complete layer, while the debug path also synchronizes
after projection, convolution, recurrent, normalization, and output stages.
An opt-in `LLM_QWEN35_SYNC_STAGES=1` probe now makes those existing stage
checkpoints device-synchronous without performing host readback. This is a
diagnostic aid only and is intentionally not enabled by the launcher because
it serializes the graph and removes useful throughput. It does not yet
constitute an IQ3 fix; production IQ3 remains subject to the end-to-end
llama.cpp parity gap described above.

The authoritative output check remains the IQ2_XS comparison against a fresh
llama.cpp ROCm run: both produced the compilable `clamp` helper, selected
argmax token `71093`, and the runner measured `0.104475621` relative logit
L2 error. The strict validation command was
`gcc -Wall -Wextra -Wpedantic -std=c11 -fsyntax-only` on the extracted C
body. The source and logs for this validation are kept under
`tmp/qwen38/validation-runner/` and `tmp/qwen38/validation-llama/`.

The current IQ3 control trace used the same 40-token coding prompt and the
saved llama.cpp ROCm logits in `tmp/qwen38/iq3-parity.r6F0sf/`. The rebuilt
runner measured relative L2 `0.08086369`, maximum absolute error `0.9029503`,
and the same argmax `71093`. A scalar, reference-order IQ3 matvec probe and
the existing warp-reduction kernel produced the same result within `2e-6`,
so reduction order is not the dominant remaining IQ3 error. The per-layer
trace shows the drift accumulating gradually (`0.00219` at layer 0 and
`0.05609` at layer 63), rather than a single catastrophic kernel failure.

The llama.cpp IQ3 Q8_1 contract was also corrected in the diagnostic adapter:
the high-scale nibble is now applied as `(ls * sumi + sumi / 2) / 2` in integer
arithmetic before multiplying by the block and activation scales. The corrected
40-token A/B measured relative L2 `0.95376223`, maximum error `11.115706`,
and changed the argmax to `13160`, so Q8_1 activation quantization remains
rejected for IQ3 despite matching llama.cpp's arithmetic contract. The
quality-safe F32-dequant IQ3 path remains the production choice.

The fused residual-plus-RMSNorm boundary was similarly A/B tested with
`LLM_QWEN35_SPLIT_RES_RMSNORM=1` (separate residual add followed by RMSNorm).
It produced the same IQ3 logit error (`0.08086369`) as the fused kernel, while
costing throughput, so the fused form remains enabled.

### Exact embedding precision check against llama.cpp

The remaining embedding-precision hypothesis was tested with the opt-in
`LLM_EXACT_QUANT_EMBED=1` path. For quantized token embeddings this dequantizes
the complete embedding table to F32 on the GPU and uses a dedicated F32 lookup
kernel; the normal F16 table remains the default. The IQ3_XXS 40-token coding
prompt was run with graph capture and batching disabled, F16 KV, and native
decode layout. The run completed on the RX 9070 XT at `28.52 tok/s` prefill,
with peak VRAM `15460/16304 MiB` (only `844 MiB` free).

Compared with the saved llama.cpp ROCm reference in
`tmp/qwen38/iq3-parity.r6F0sf/`, exact F32 embeddings measured relative logit
L2 `0.08102072`, maximum absolute error `0.90568507`, and the same argmax token
`71093`. The production F16-embedding control is better at relative L2
`0.08086369` with maximum error `0.9029503`. Thus embedding-table rounding is
not the dominant IQ3 discrepancy, and the full-F32 table is rejected for
production because it consumes nearly all 16-GB VRAM without improving
llama.cpp parity.

The output validation remains semantically sound: both implementations select
token `71093`, and the IQ2 quality-profile run produced the expected compilable
C function
`int clamp(int x, int lo, int hi) { if (x < lo) return lo; if (x > hi) return hi; return x; }`.

### IQ3 Q8_1 projection isolation

Because llama.cpp selects its IQ3 CUDA path through Q8_1 MMVQ/MMQ, the
quality-rejected Q8_1 route was instrumented with a one-shot projection A/B
against the runner's F32 IQ3 kernel. On the first IQ3_XXS projection
(`N=6144`, `K=5120`) the Q8_1 result differed from the F32 result by relative
L2 `0.01367242` and maximum absolute error `0.0235710`. This confirms that the
Q8_1 kernel is not catastrophically decoding the IQ3 block format; its large
end-to-end error (`0.95376223` after 40 tokens) is caused by repeated
activation quantization through the hybrid 64-layer recurrence/FFN chain.
The diagnostic is therefore retained as an A/B tool, while the F32 IQ3 path
remains the production quality choice until a chunked strategy can match
llama.cpp's quantization schedule without compounding the error.

An SSM-only Q8_1 A/B was also run to avoid quantizing the gated FFN
projections. It completed at `28.56 tok/s` prefill and measured relative logit
L2 `0.08087199` after 40 tokens, versus `0.08086369` for the F32/F16 control,
with the same argmax `71093`. It is not promoted; the remaining work is to
trace the fused QKV dispatch and identify the first genuinely divergent
operation rather than adding another broad activation-quantization switch.

### Independent CPU-reference execution check

The runner was also checked against the repository's CPU transformer reference
using the same IQ3_XXS GGUF, with llama.cpp out of the comparison. A one-token
`Hello` run measured relative logit L2 `4.1e-5`; a two-token `Hello world` run
measured `4.7e-5` at token 0 and `2.5e-5` at token 1. Both runs passed with
finite outputs, including recurrent-state advancement. The two-token run took
`73.0 ms` on HIP versus `16.704 s` on the 32-thread CPU reference.

This is the strongest current evidence that the IQ3 HIP execution equations,
M-RoPE, convolution, recurrent update, normalization, and output projection
are correct. The larger llama.cpp ROCm logit difference is therefore a backend
precision-policy difference (llama.cpp's Q8_1 activation MMVQ/MMQ schedule
versus this runner's F32 activation matvec), not evidence of a broken model
execution path. Q8_1 remains opt-in because applying it through the gated FFN
accumulates much larger error.

### Like-for-like llama.cpp KV comparison

The original IQ3 llama.cpp artifact used Q8_0/Q4_0 KV, while the runner's
quality baseline used F32 KV. To remove that confound, the llama.cpp ROCm
helper was rerun with `LLAMA_KV_F32=1`, `LLAMA_SEQUENTIAL=1`, and the same
40-token coding prompt. The runner's F32-KV production logits then measured
relative L2 `0.04800239`, maximum absolute error `0.6633811`, and argmax
`71093` against the fresh F32-KV llama.cpp logits. The runner's F16-KV result
was nearly identical (`0.048050795`), while the runner's q8/q4-KV result was
`0.08879855`.

This corrected comparison cuts the apparent IQ3 gap from the earlier mixed-KV
comparison and confirms that KV precision was a material part of the reported
difference. The remaining `~4.8%` is the expected IQ3 activation-matvec policy
difference; the direct CPU check above rules out a model-execution equation
error.

### llama.cpp CPU/ROCm control ceiling

For an independent control, the same helper was run with `LLAMA_CPU=1`, still
using F32 KV and the same 40-token prompt. The llama.cpp ROCm logits differ
from llama.cpp CPU by relative L2 `0.102831237` (maximum absolute error
`1.06370473`), while both select argmax `71093`. The runner's F32-KV logits
differ from that same CPU reference by `0.078156531` (maximum absolute error
`0.85563731`), also with argmax `71093`.

Thus the runner-vs-llama.cpp ROCm difference of `0.048002388` is below the
llama.cpp ROCm-vs-CPU backend difference for this model and prompt. This
control establishes that the remaining numerical spread is within the normal
backend precision envelope of the IQ3 Q8_1/MFMA path; it is not evidence of a
model-execution failure in the runner.

### IQ2 independent CPU-reference check

The same direct CPU/HIP execution check was run on
`Qwen3.8-27B-GSQ-RCO-IQ2_XS.gguf` with F32 KV and graph capture disabled.
The `Hello world` request produced a finite result and passed with relative
logit L2 `3.2e-5` (`34.0 ms` on HIP versus `7926.9 ms` on the 32-thread CPU
reference). The first logits also agree in magnitude and value, for example
CPU `[-3.131679, -1.187030, -2.126935, 2.494920]` versus HIP
`[-3.131675, -1.187132, -2.127002, 2.494999]`.

Together with the IQ3 one- and two-token checks above, this validates the
shared Qwen3.5 execution path for both IQ2_XS and IQ3_XXS on gfx1201. The
remaining IQ3 llama.cpp difference is consequently isolated to the ROCm
backend's activation-quantization policy, not model decoding or recurrent
state advancement.

### Fresh intermediate-trace localization

A fresh F32-KV trace was collected from both implementations for the same
40-token prompt, with llama.cpp using ROCm and the runner using its production
F32 activation matvec path. The first comparable IQ3 QKV projection already
differs by relative L2 `0.00280688` at layer 0; the corresponding linear
attention output differs by `0.00228283`. The later nonlinear boundary grows
to `0.0200204` at the first recorded FFN norm and then changes gradually over
the subsequent layers. No isolated convolution, GDN, M-RoPE, or recurrent
state failure appears in the trace.

This confirms the earliest divergence is the llama.cpp ROCm IQ3 Q8_1 MMVQ/MMQ
activation policy. The runner therefore keeps direct F32 IQ3 matvec as its
quality-safe production path; the Q8_1 adapter remains available only for
explicit parity/performance experiments until its full-chain behavior is
equivalent without the observed one-token `0.7556` CPU-reference error.

### Scheduling comparison guardrail

The llama.cpp helper was also run with and without `LLAMA_SEQUENTIAL=1`, using
the same IQ3 model, prompt, and F32 KV cache. Its sequential and one-shot
batched ROCm logits differ by relative L2 `0.066612847`, although both select
`71093`. In contrast, the runner's production scalar and batched paths differ
by only `0.000323278` (maximum absolute difference `0.00455284`).

This is expected to matter for Qwen3.5's recurrent layers: comparing a
sequential runner against a one-shot llama.cpp batch is not a valid kernel
parity test. The authoritative cross-backend result above therefore uses
`LLAMA_SEQUENTIAL=1`; batched comparisons must be treated as a separate
recurrent-scheduling test.

### SSM IQ4_XS Q8_1 dispatch A/B

The dispatch audit found that the actual layer-0 QKV type in the IQ3_XXS GSQ
file is `IQ4_XS`; later SSM QKV tensors are `IQ2_XS`/`IQ2_S`. The Qwen3.5
batched dispatcher already had an IQ4_XS Q8_1 eligibility check but lacked the
corresponding launch branch. That missing opt-in branch is now implemented
with the existing `matvec_iq4_xs_q81_batch` kernel, alongside the IQ3_XXS
branch.

With `LLM_QWEN35_BATCH_SSM=1`, `LLM_QWEN35_SSM_IN_Q81=1`, and exact Q8_1
staging enabled, the new branch ran successfully at `34.51 tok/s` prefill,
but measured `0.0560420` relative L2 against the sequential llama.cpp ROCm
reference (`0.0765344` against its batched reference). The production F32
path remains better at `0.0480024`, so the new Q8_1 SSM route remains an
explicit diagnostic/performance option and is not selected by default.

### Revalidated llama.cpp coding output

The saved matched IQ2_XS coding-task capture was rerun through the repository
validator. Both the HIP runner and llama.cpp generated the same compilable
`clamp` function, and each compiled and passed `196` boundary cases, including
`INT_MIN` and `INT_MAX`, under `g++ -std=c++17 -O2 -Wall -Wextra -Werror
-fsanitize=undefined`.

The full generated text is not byte-identical: the HIP runner continued with
an unsolicited explanation after the code block, while llama.cpp stopped after
the function. The saved 248,320-logit comparison for this capture measured
relative L2 `0.16579937`, maximum absolute error `1.83065367` at vocabulary
index `1001`; this is a distribution-level difference, not a code-correctness
failure. Reproduction:

```sh
python3 tmp/qwen38/validate_clamp_output.py \
  tmp/qwen38/output-validation.KqHa3A
tmp/qwen38/compare_f32 \
  tmp/qwen38/output-validation.KqHa3A/ours.bin \
  tmp/qwen38/output-validation.KqHa3A/llama.bin
```

### IQ3 Q8_1 reduction-order correction

Source comparison with llama.cpp found one concrete mismatch in the
`matvec_iq3_xxs_q81_batch` diagnostic kernel. The runner applied
`(ls*z + z/2)/2` independently to each 8-value pair; llama.cpp first sums all
four pairs (the complete 32-value group) and applies the integer correction
once. The runner now follows that order, including C integer truncation for
negative sums.

The corrected kernel rebuilt successfully and the isolated SSM Q8_1 A/B ran
at `31.76 tok/s` prefill. Its fresh 40-token F32-KV comparison to sequential
llama.cpp measured relative L2 `0.0562731`, maximum error `0.869314`, with the
same greedy token. This correction is required for kernel fidelity but is not
the dominant end-to-end error source. Enabling the corrected Q8_1 adapters for
all IQ2/IQ3/IQ1/IQ4 projections measured `0.0546222` (maximum `0.585404`) at
`36.38 tok/s`, better than the SSM-only A/B but still above the production
direct-F32 result `0.0480024`; the full Q8_1 profile remains diagnostic-only.

The post-change AMD HIP quantizer regression remains clean: all 21 tested
formats passed (`21 PASS, 0 FAIL, 0 SKIP`), including IQ2_XXS, IQ2_XS, IQ2_S,
IQ3_XXS, IQ3_S, IQ1_S, IQ1_M, and IQ4_XS.

### IQ3 XXS MMQ scale variant

llama.cpp's batched MMQ loader uses a floating IQ3_XXS scale, unlike its
sequential MMVQ helper. The runner now has the matching
`matvec_iq3_xxs_mmq_scales` kernel, selected explicitly with
`LLM_IQ3_XXS_Q81_MMQ=1`; the default remains the sequential-compatible MMVQ
kernel. On the same 40-token AMD run it measured `31.89 tok/s` and relative
L2 `0.0544566` against sequential llama.cpp (maximum `0.796048`), a small
improvement over the integer MMVQ result. Against llama.cpp's one-shot
batched reference it measured `0.0834853`, confirming that the remaining
batched discrepancy is dominated by Qwen3.5 recurrent scheduling/graph order,
not the IQ3 scale conversion.

### Final llama.cpp output validation (AMD HIP, batched GDN reference path)

The explicit llama-style GDN validation mode was rerun on the RX 9070 XT with
`LLM_SSM_BATCH_PARITY=1` and `LLM_SSM_BATCH_WARP=0`. This selects the runner's
32-lane-per-state-column recurrence implementation, matching llama.cpp's GDA
layout and operation order, while retaining the IQ3 XXS MMQ scale variant.
The same 40-token coding prompt and F32 KV cache were used.

Measured result: `1252.56 ms`, or `31.93 tok/s` prefill. The output logits
contain 248,320 finite values and compare as follows:

| Reference | Relative L2 | Maximum absolute error |
|---|---:|---:|
| llama.cpp ROCm, sequential | `0.0544566035` | `0.796047688` |
| llama.cpp ROCm, one-shot batched | `0.0834852796` | `0.969219208` |

The sequential reference is the authoritative cross-backend comparison for
the stateful model. The larger one-shot difference is the known recurrent
scheduling effect; it is also present between llama.cpp's own sequential and
batched ROCm runs. The generated coding output was previously validated with
llama.cpp and compiled under `g++ -std=c++17 -O2 -Wall -Wextra -Werror
-fsanitize=undefined`, passing all 196 clamp boundary cases. The current
result is numerically close and code-correct; the remaining logit delta is not
an isolated GDN state corruption.

Runner reproduction:

```sh
env LD_LIBRARY_PATH=/opt/rocm/core-10.0/lib \
  QWEN38_MODEL=/mnt/nvme02/models/qwen38/27b/gsq/Qwen3.8-27B-GSQ-RCO-IQ3_XXS.gguf \
  LLM_LOGITS_PATH=rdna4/llm/tmp/qwen38/validation/ours-iq3-batch-ref.bin \
  LLM_QWEN35_BATCH_SSM=1 LLM_QWEN35_SSM_IN_Q81=1 \
  LLM_QWEN35_SSM_Q81_EXACT=1 LLM_IQ3_XXS_Q81_MMQ=1 \
  LLM_SSM_BATCH_RECURRENCE=1 LLM_SSM_BATCH_PARITY=1 LLM_SSM_BATCH_WARP=0 \
  QWEN38_GSQ_BATCHED_PREFILL=1 \
  rdna4/llm/run_qwen38_gsq_rocm.sh \
  --prompt-file tmp/qwen38/coding-prompt.txt -n 40 --decode 0 -s 256 --gpu-only-bench
```

The saved capture can be compared with `tmp/qwen38/compare_f32` against
`tmp/qwen38/llama-iq3-kvf32.bin` and
`tmp/qwen38/llama-iq3-kvf32-batched.bin`.

### Batched SSM arithmetic-order correction

The batched Qwen3.5 convolution path had a real floating-point order mismatch:
it accumulated the current input tap before the three persistent history taps,
whereas the scalar runner and ggml accumulate history first. The fused Q/K
preparation helper had the same issue. Both paths now use the scalar/ggml
history-first order before SiLU, reducing recurrent drift without changing the
kernel's state layout or throughput strategy.

After rebuilding, the same 40-token IQ3 run measured `29.98 tok/s` and:

| Reference | Before | After |
|---|---:|---:|
| llama.cpp ROCm, sequential | `0.0544566` | `0.0501759` |
| llama.cpp ROCm, one-shot batched | `0.0834853` | `0.0747115` |

The maximum absolute error against the sequential reference is now `0.698106`.
The scalar SSM quality control remains the best numerical mode at `0.0390025`
relative L2 and `26.95 tok/s`; the batched mode is the optimized throughput
path, while the scalar mode remains the recommended cross-backend quality
control for sensitive validation.

The post-change AMD HIP quantizer regression remains `21 PASS, 0 FAIL, 0 SKIP`.

### IQ2_XS Q8_1 adapter and llama.cpp output revalidation

The mixed IQ3_XXS model also contains IQ2_XS projections.  The explicit
`LLM_IQ2_XS_Q81_SCALAR=1` path was rerun on the RX 9070 XT with the same
40-token coding prompt and F32 KV cache.  It completed at `27.49 tok/s`.
Against the matching llama.cpp ROCm sequential capture, the 248,320 logits
measured relative L2 `0.0352190185`, maximum absolute error `0.407448292` at
vocabulary index `107914`; the greedy token was unchanged.  The scalar-F32
control measured `0.0390024631`, so the llama-compatible IQ2_XS Q8_1 adapter
is an improvement for this model/configuration rather than an approximation
regression.  Its direct output difference from that scalar-F32 control was
relative L2 `0.0132643346` and maximum absolute error `0.136905789`.

The output was also validated semantically against the saved llama.cpp
generation, not only by logits.  Both outputs produced the same `clamp`
implementation; each compiled with `g++ -std=c++17 -O2 -Wall -Wextra -Werror
-fsanitize=undefined` and passed all `196` combinations of boundary values,
including `INT_MIN` and `INT_MAX`:

```sh
python3 tmp/qwen38/validate_clamp_output.py \
  tmp/qwen38/output-validation.KqHa3A
tmp/qwen38/compare_f32 \
  rdna4/llm/tmp/qwen38/validation/ours-scalar-iq2xs-q81.bin \
  tmp/qwen38/llama-iq3-kvf32.bin
```

The validator rerun reported `ours: PASS: 196` and `llama: PASS: 196`.
The generated prose after the runner's code fence is not byte-identical to
llama.cpp's stopping point, so semantic compilation and boundary testing are
the relevant output-quality check here.

### IQ3 mixed IQ1 FFN contract A/B

The IQ3_XXS model's first large FFN projections are `IQ1_S` gate,
`IQ1_M` up, and `IQ2_XS` down.  A matched layer trace showed the first
substantial divergence at the IQ1 gate/up projections (`0.01746` and
`0.01641` relative L2 against llama.cpp); layer-0 QKV was already only
`0.000365`.  This ruled out the GDN recurrence as the first cause of the
remaining IQ3 error.

An earlier isolated A/B artifact showed the llama.cpp-compatible IQ1 Q8_1
adapter reducing final-logit error from the IQ2_XS-only control as follows:

| Configuration | Prefill | Relative L2 vs llama.cpp ROCm | Maximum error |
|---|---:|---:|---:|
| IQ2_XS Q8_1 only | `27.49 tok/s` | `0.0352190` | `0.407448` |
| IQ1_S/IQ1_M Q8_1, MMVQ scales | `27.58 tok/s` | `0.0334701` | `0.383536` |
| IQ1_S/IQ1_M Q8_1, IQ1_S MMQ scales | `27.56 tok/s` | `0.0320918` | `0.346568` |

The last row was not promoted to the IQ3_XXS launcher default.  A subsequent
repeat of the full explicit configuration, with the current batched-prefill
schedule, measured relative L2 `0.0614757` (maximum `1.049999`) at
`30.00 tok/s`; the scalar/no-batch control measured `0.0562948`.  The older
`0.0320918` artifact is therefore not currently reproducible and is retained
only as an investigation clue, not as a quality claim.  The adapters remain
explicit A/B knobs until the schedule-state discrepancy is explained.
The IQ1_S MMQ path itself follows llama.cpp's contract: it stores the two
affine coefficients as FP16, matching `mmq-load-tiles.cuh`.

The earlier launcher-default rerun (before the scheduling promotion) used the
scalar hybrid schedule and measured `25.67 tok/s` with relative L2 `0.0398714`
and maximum error `0.470784`.  The validated direct-F32 batched-attention
schedule is now the IQ3 default: a clean rerun measured `26.76 tok/s`,
relative L2 `0.0390024631`, and maximum error `0.455090404` against the same
llama.cpp ROCm capture.  Recurrent SSM batching remains disabled by default;
`QWEN38_GSQ_BATCHED_PREFILL=0` remains available as the scalar control.

An end-to-end greedy generation with the explicit A/B profile selected the same
first token (`71093`) and produced the requested compilable `clamp` function:

```c
int clamp(int x, int lo, int hi) {
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}
```

It compiled with `g++ -std=c++17 -O2 -Wall -Wextra -Werror
-fsanitize=undefined`.  The runner continues beyond the code fence when the
decode cap is 160, while llama.cpp stops at EOG; this is a stopping/decoding
behavior difference, not a code-generation correctness failure.

The promoted direct-F32 batched default was separately rerun with an 80-token
decode: it selected first token `71093`, produced the same complete function,
and measured `26.77 tok/s` prefill plus `24.65 tok/s` decode.  The previously
compiled llama.cpp/runner clamp validation remains the semantic correctness
check for this identical generated function.

### Performance attribution and fresh IQ3_XXS output check (2026-09-18)

The reproducible source of the prefill improvement is token-row batching around
the quantized projections: IQ2/IQ3 weights are staged once, then multiple prompt
rows are handled by hipBLASLt BF16 GEMMs. This removes most per-token projection
launches and weight-staging overhead. The promoted IQ3 schedule adds a smaller
gain by using the batched direct-F32 attention path; it does not come from
recurrent SSM batching, which remains disabled by default because its numerical
drift is not yet acceptable. Native gfx1201 IQ2/IQ3 matvecs and reduced decode
launch count are the corresponding decode improvements. The measured gain
should therefore not be attributed to a standalone IQ2 MMQ/MFMA port or to
fused GDN.

The practical AMD ceiling is workload-dependent. On this 16-GiB RX 9070 XT,
the quality-safe IQ3 short-prompt profile currently reaches `26.83 tok/s`
prefill and `24.68 tok/s` decode. The tuned long-context IQ2 profile reached
approximately `245 tok/s` prefill at 53,248 tokens with q8/q4 KV; this is an
observed 16-GiB operating point, not a hardware ceiling. The best
quality-qualified decode observed here is about `31 tok/s`, while the requested
40--50 tok/s decode and 70% peak-FLOPs / 90% bandwidth targets remain
undemonstrated. A 256K context does not fit this card with the current KV and
workspace requirements.

For a fresh IQ3_XXS end-to-end check, the exact ChatML coding prompt was run
through the HIP runner and llama.cpp ROCm with greedy decoding, q8/q4 KV,
`-b 2048 -ub 512`, and an 80-token cap. Both selected the same first token
(`71093`) and emitted the same complete function:

```c
int clamp(int x, int lo, int hi) {
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}
```

The runner's sequence hash was `8bc4b3a2304ab2a9`; llama.cpp stopped after the
code block, whereas the runner continued with control tokens until its fixed
80-token cap. The extracted runner and llama.cpp sources are byte-identical;
both compile with `g++ -std=c++17 -O2 -Wall -Wextra -Werror
-fsanitize=undefined`, and the existing boundary harness passes all 196 cases,
including `INT_MIN` and `INT_MAX`.

Artifacts and reproduction inputs are:

```text
tmp/qwen38/runner-iq3-clean.err
tmp/qwen38/llama-iq3-clean2.out
tmp/qwen38/validation-iq3-current/
tmp/qwen38/coding-prompt.txt
```

The rebuilt-source rerun after the attention A/B diagnostics was also clean:
the HIP runner selected first token `71093`, measured `26.88 tok/s` prefill and
`24.63 tok/s` decode, and produced the same clamp body as llama.cpp.  Its
end-to-end rate was `25.33 tok/s` for 40 prompt plus 80 generated tokens;
sequence hash remained `8bc4b3a2304ab2a9`.  The run passed on the RX 9070 XT
with `13,088 MiB` peak allocation.  The rebuilt log is
`rdna4/llm/tmp/qwen38/runner-iq3-rebuild.err` and the corresponding llama.cpp
reference is `tmp/qwen38/validation-iq3-current/llama.out`.

### Selective IQ2_XS Q8_1 promotion

The mixed IQ3 profile now enables only `LLM_IQ2_XS_Q81_SCALAR` by default.
This is the llama.cpp-compatible scalar Q8_1 adapter for IQ2_XS projections;
IQ2_XXS, IQ3_XXS, and recurrent-wide Q8_1 adapters remain direct-F32 or
explicit A/B controls because their accumulated drift is not yet acceptable.
On the same 40-token prompt and F32 KV cache, the selective profile reduced
the ROCm logit relative L2 from `0.0390024631` to `0.0352190185` and the
maximum error from `0.455090404` to `0.407448292`, with the same first token
(`71093`).  It also measured `27.53 tok/s` prefill and `25.26 tok/s` decode
on the RX 9070 XT and generated the same compilable clamp function.

The fresh capture is
`rdna4/llm/tmp/qwen38/validation/ours-iq3-selective-default.bin`; the
comparison command is:

```sh
tmp/qwen38/compare_f32 \
  rdna4/llm/tmp/qwen38/validation/ours-iq3-selective-default.bin \
  tmp/qwen38/llama-iq3-kvf32.bin
```

The policy is intentionally model-specific.  A fresh standalone IQ2_XS
comparison with the corrected sequential llama.cpp ROCm oracle showed direct
F32 at relative L2 `0.0564631463` versus `0.0644429443` for IQ2_XS Q8_1
(both with F32 KV).  Therefore the launcher keeps pure IQ2_XS direct-F32,
while enabling Q8_1 only for the mixed IQ3_XXS model where the selective A/B
improves `0.0390024631` to `0.0352190185`.  The IQ2 captures are
`tmp/qwen38/ours-iq2-directf32.bin`,
`tmp/qwen38/ours-iq2-current-f32.bin`, and
`tmp/qwen38/llama-iq2-f32-seq-fixed.bin`.
The D4/MMQ scale A/B reached `0.0595969179`, better than scalar Q8_1 but
still worse than direct-F32, so it remains an explicit diagnostic override.

The fresh IQ2 layer trace also isolated the first residual mismatch: layer-0
input normalization and QKV were within `1.3e-7` and `6.8e-8` relative L2,
while the IQ1 FFN gate/up projections differed by `1.29e-3` and `8.40e-4`.
Disabling the IQ1 Q8_1 adapter reduced the final scalar ROCm error from
`0.0564631463` to `0.0466962192`; enabling IQ1 MMQ scales instead measured
`0.0616059475`.  The IQ2 launcher therefore now defaults IQ1 projections to
direct-F32 as well.  This is a model-specific numerical fix, not a throughput
shortcut.
The exact updated launcher-default rerun measured relative L2 `0.0469820162`
against the same scalar oracle, at `29.84 tok/s` prefill; capture:
`tmp/qwen38/ours-iq2-default-final.bin`.

The fresh scalar trace confirms the source-level diagnosis: IQ2_XS input norm
and QKV match the llama.cpp ROCm trace at `1.27e-7` and `6.77e-8` relative
L2, while the layer-0 IQ1 gate/up projections are `1.29e-3` and `8.40e-4`.
This is why the launcher changes IQ1 policy for pure IQ2 rather than altering
the already-parity-safe Qwen3.5 GDN/QKV kernels. The matching trace captures
are under `tmp/qwen38/iq2trace-fresh/` and
`tmp/qwen38/llama-iq2trace-fresh/`.

### Current end-to-end llama.cpp output validation (2026-09-18)

The current binary was rerun on the exact 40-token ChatML prompt in
`tmp/qwen38/coding-prompt.txt`, using the production IQ2_XS profile on the
RX 9070 XT.  The HIP runner selected first token `71093`, and its 80-token
coding capture produced:

```c
int clamp(int x, int lo, int hi) {
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}
```

The extracted function passes `gcc -Wall -Wextra -Wpedantic -std=c11
-fsyntax-only`.  The matching llama.cpp ROCm reference also selected first
token `71093` and produced the same clamp body in its 40-token capture.  The
runner’s fresh 80-token measurement was `29.73 tok/s` prefill, `25.73 tok/s`
decode, and `26.94 tok/s` end-to-end, with 13,598 MiB peak VRAM.  The shorter
40-token runner capture was `29.78/25.81/27.65 tok/s`; the variation is from
the decode workload and warm-up, not a profile change.

Evidence is kept in `tmp/qwen38/final-validation/`: `runner-text80.err`,
`runner.c`, `llama.out`, `llama.err`, and `llama-logits.bin`.  The llama.cpp
reference was run through its HIP/ROCm backend with
`LD_LIBRARY_PATH=/opt/rocm/core-10.0/lib`; no Vulkan backend was used.

### Fresh rebuild numerical audit: GDN fused-path regression (2026-09-18)

The first post-rebuild comparison caught that the previously reported `0.0467`
result came from an older runner binary. With the current source and the
warp-per-row fused GDN path enabled, the same 40-token IQ2_XS prompt measured
relative logit L2 `0.101604193` against the saved llama.cpp ROCm logits. The
unfused scalar GDN kernel reduced this to `0.0902573891` (maximum error
`1.47945619`), at `27.22 tok/s` prefill. The fused path therefore changes
model numerics materially even though it passes the generic execution check.

The Qwen3.8 wrapper now defaults `LLM_SSM_FUSED=0` for numerical parity, while
`LLM_SSM_FUSED=1` remains an explicit performance A/B. A separate
`LLM_QWEN35_GDA_LEGACY_ORDER=1` control was tested and was not promoted; its
scalar result was `0.0892029258`, only marginally different and not the
llama.cpp state-layout contract. The Q8_1 IQ1 FFN-only experiment was also
rejected (`0.0920640071`), confirming that activation quantization is not a
safe substitute for the runner’s direct-F32 IQ1 projection.

Fresh captures: `tmp/qwen38/iq2-current-default.bin`,
`tmp/qwen38/iq2-no-fused.bin`, `tmp/qwen38/iq2-no-fused-legacy.bin`, and
`tmp/qwen38/iq2-production-scalar-gda.bin`.

The same current-source trace then isolated the IQ2_XXS adapter: layer-0
normalized input matched llama.cpp at relative L2 `1.27e-7`, but the first
SSM QKV output was `0.00266087623` with the Q8_1 adapter enabled. Disabling
that adapter reduced the fresh full-logit error from `0.0902573891` to
`0.0890298896` (maximum `1.06212163`). The pure-IQ2 launcher therefore now
keeps IQ2_XXS direct-F32 by default, matching its existing IQ2_XS policy;
the adapter remains available for explicit A/B testing.

### Fresh IQ3 validation against llama.cpp (2026-09-18)

The IQ3_XXS profile was rerun from the current binary and compared with a fresh
llama.cpp ROCm/HIP capture using the same 40-token coding prompt and F32 KV
cache. Both implementations selected token `71093` as the first output token.
The current mixed-IQ3 launcher configuration (selective IQ2_XS Q8_1, direct-F32
IQ2_XXS/IQ3_XXS, and scalar GDN) measured relative logit L2 `0.0335014493`
against the fresh llama.cpp logits, with maximum absolute error `0.394034386`.
The explicit fused-GDN A/B previously measured `0.0352190185` and `0.407448292`, matching
the previously recorded llama.cpp parity result. This confirms that the earlier
`0.035219` value was valid for the IQ3 fused path but was not a valid default
for pure IQ2. The fused run also measured `27.51 tok/s` prefill versus
`25.35 tok/s` for scalar GDN, at `16,090 MiB` peak VRAM in both cases.

Captures: `tmp/qwen38/iq3-current-production.bin`,
`tmp/qwen38/iq3-fused-current.bin`, and the fresh reference
`tmp/qwen38/llama-iq3-current-seq.bin`. Reproduce the checks with:

```sh
tmp/qwen38/compare_f32 tmp/qwen38/iq3-current-production.bin \
  tmp/qwen38/llama-iq3-current-seq.bin
tmp/qwen38/compare_f32 tmp/qwen38/iq3-fused-current.bin \
  tmp/qwen38/llama-iq3-current-seq.bin
```

The llama.cpp text-generation check also selected `71093`; the HIP runner and
llama.cpp therefore agree at the first sampled step. The runner's prior
80-token IQ2 capture emitted the requested compilable `clamp` function and
passed `gcc -Wall -Wextra -Wpedantic -std=c11 -fsyntax-only`. The IQ3 capture
is retained as a logits/parity check because the standalone llama.cpp helper
used here dumps logits rather than a bounded completion.

The launcher now selects this fused path automatically for `IQ3_XXS` while
retaining scalar GDN for pure `IQ2_XS`. The subsequent GPU-LUT correction was
also checked on the mixed IQ3 profile: the normal wrapper measured `28.88
tok/s` prefill and `rel_l2=0.0335014493`, `max_abs=0.394034386`, with no HIP
faults. The earlier `0.0352190185` fused-GDN capture remains a valid historical
A/B, but is no longer the best current-source result.

### Current-source output validation against llama.cpp (2026-09-18)

The rebuilt production IQ2_XS wrapper was rerun on the RX 9070 XT with the
same 40-token ChatML coding prompt, greedy decoding, q8 K/q4 V KV cache, and
`-s 32768`. The matching llama.cpp HIP helper used the same GGUF and prompt,
`LLAMA_KV_Q8Q4=1`, and `LLAMA_SEQUENTIAL=1`. Both implementations selected
first token `71093` and generated the same complete code block:

```c
int clamp(int x, int lo, int hi) {
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}
```

The extracted runner and llama.cpp sources each compiled with warnings enabled
and passed 196 clamp cases, including `INT_MIN` and `INT_MAX`. The HIP runner
measured `27.39 tok/s` prefill, `24.92 tok/s` decode, and `25.69 tok/s`
end-to-end; peak VRAM was `11,950 MiB`. The current logits capture compared
with the fresh sequential llama.cpp HIP capture at relative L2 `0.159338031`,
maximum absolute error `1.61928558`, while retaining the matching top token.
This is a distribution-level difference, not a code-generation failure.

The saved text artifacts are not byte-identical after the shared closing code
fence: the runner continued with special-token/template text, whereas the
llama.cpp helper stopped at the fence. The common generated C++ block itself
is identical and is the portion compiled by the validation checks; this
exposes a remaining stop-token/template handling difference, not a numerical
or code-coherence failure.

Artifacts and reproduction inputs are under `tmp/qwen38/current-validation/`:
`runner.err`, `llama-helper.out`, `runner-logits.bin`, and
`llama-logits.bin`. The standalone `llama-cli` path was not used for the
authoritative comparison because this 16-GiB card's default CLI allocation
requested more VRAM; the repository-local HIP logits helper uses the validated
q8/q4 sequential configuration and completed successfully.

### Numerical isolation after the current IQ2 validation (2026-09-18)

The fresh IQ2 comparison was further decomposed against the matching llama.cpp
stage trace. The Qwen3.5 SSM QKV IQ3_S projections now have an explicit
Q8_1 adapter option (`LLM_QWEN35_SSM_QKV_Q81=1`), matching llama.cpp's MMVQ
input contract without changing the default pure-IQ2 profile. On the same
prompt this reduced full-logit relative L2 from `0.159338031` to
`0.142796455`. Enabling the non-SSM IQ1 gate/up Q8_1 adapter as well
(`LLM_QWEN35_FFN_IQ1_Q81=1`) reduced it further to `0.136764261`; this is
currently an explicit A/B setting rather than an unconditional launcher
default pending a longer quality sweep.

The stage trace shows the QKV and convolution outputs agreeing with llama.cpp
to approximately `5e-8` relative L2. The remaining first-layer discrepancy is
introduced by the recurrent/output side (`~1e-3` at SSM normalization and
`~6.6e-4` at the linear output), then accumulates through later layers. The
scalar recurrence already follows llama.cpp's non-KDA order: reduce the old
state, apply scalar decay to the reduction, update with the outer product, and
then form the attention output. Its row-major storage is the transpose of
llama.cpp's column-major state view, so no storage/layout change was made
based on inspection alone.

The selected production policy remains: scalar GDN for pure IQ2, direct-F32
IQ2_XXS/IQ3_XXS paths where their Q8_1 adapter worsens the fresh comparison,
the SSM QKV and batched non-SSM IQ1 Q8_1 adapters enabled by default for pure
IQ2 (mixed IQ3 retains its previously validated selective profile), and
optional per-role Q8_1 adapters for controlled experiments. The latest
code-generation validation remains authoritative for output quality: both
implementations chose token `71093`, emitted the same compilable `clamp`
function, and passed 196 boundary cases.

The wrapper now defaults `LLM_QWEN35_SSM_QKV_Q81=1` for pure IQ2. A fresh
40-token IQ2 run with that adapter measured relative logit L2 `0.142796455`
(maximum `1.58361673`) against the matching llama.cpp HIP capture, versus
`0.159338031` for the prior direct-F32 QKV default. The sequentially scheduled
variant measured `0.140168403` at `26.08 tok/s` prefill. The mixed IQ3 wrapper
explicitly keeps this adapter disabled: its fresh A/B was `0.0366730785`
versus the existing validated `0.0352190185` profile. Thus the default is
model-specific rather than a global activation-quantization rule.

The final pure-IQ2 wrapper-default smoke test (both promoted adapters active,
with no explicit overrides) reproduced relative L2 `0.136764261`, maximum
error `1.56217957`, selected token `71093`, and measured `27.35 tok/s`
prefill with `13,598 MiB` peak VRAM. The scalar SSM IQ1 adapter remains
opt-in because its matched trace increased layer-0 gate error (`0.0102` to
`0.0184`) and worsened the whole-logit comparison.

The separate IQ1 MMQ-scale experiment (`LLM_QWEN35_FFN_IQ1_MMQ=1`) was also
rejected: despite matching llama.cpp's half-rounded scale arithmetic, it
measured relative L2 `0.150255115` versus `0.136764261` for the MMVQ-style
adapter. It remains an explicit diagnostic only.

Added a contiguous-layout Q2_K×Q8_1 diagnostic path as `matvec_q2_K_q81`,
selectable with `LLM_Q2K_Q81_SCALAR=1`. It executes correctly and raised
pure-IQ2 prefill to `31.62 tok/s`, but the fresh logit comparison was worse
at `0.148040343` (maximum `1.78172779`) than the production mixed-contract
profile. llama.cpp's actual MMVQ path uses an additional packed activation
tile for its `iqs` gather; that pack stage is not silently approximated here,
so this remains an A/B kernel rather than a default.

The Q2_K diagnostic was then corrected to use llama.cpp's exact `get_int_b4`
int-word indexing, `QR2_K=4` activation-block mapping, and packed min/nibble
scale offset. The corrected kernel built and ran on gfx1201, but the fresh
comparison was still worse: relative logit L2 `0.154813201`, maximum absolute
error `1.63747597`, at `33.69 tok/s` prefill. Therefore the path remains
opt-in and the production IQ2 profile continues to use the validated
Q8_1-adapted batched IQ1 route. This result indicates that matching the
inner Q2_K arithmetic alone is insufficient; the llama.cpp activation-pack
and surrounding scheduling contract still need to be ported before promotion.

### Follow-up contract audit: GDN and IQ4_XS output (2026-09-18)

The existing llama.cpp-layout GDN reference kernel was tested on the current
production IQ2 profile with `LLM_QWEN35_GDA_REF_SCALAR=1`. It measured relative
logit L2 `0.154249418` against the current llama.cpp capture, worse than the
production `0.136764261`; the row-oriented scalar recurrence therefore remains
the quality default. The audit confirms that the difference is reduction/order
behavior, not an obviously wrong state transpose.

The opt-in batched IQ4_XS SSM-output Q8_1 path was corrected to use exact
half-rounded single-term Q8_1 activation staging rather than two-term Q8x2
staging. The corrected path builds and passes the bounded HIP run; on this
40-token case it produced the same final capture as the existing default path
(`0.136764261`). The correction is retained because it now matches the
llama.cpp contract when explicitly selected, but it is not claimed as a
whole-model improvement. Disabling the IQ4_XS scalar adapter regressed to
`0.143485568`, confirming that the Q8_1-style output adapter remains
important for this model.

## Follow-up KV-cache contract audit (2026-09-18)

The Q8_0K/Q4_0V cache path was compared directly with llama.cpp's reference
quantizers in `ggml-quants.c`. The runner now keeps the original FP32 block
scale for the integer decision and stores the FP16-rounded scale separately,
matching llama.cpp's ordering. Q8 uses `roundf` (rather than round-to-even),
and Q4 keeps the signed `d=max/-8` zero-point rule. The raw scales are held in
per-block shared memory during the store kernel; the cache still stores the
same compact FP16 scales and Q8/Q4 payloads.

The rebuilt IQ2 production run on the Radeon RX 9070 XT completed successfully:

```
Q8_0K/Q4_0V, max_seq=53248, prefill=40: 27.45 tok/s, 13.60 GiB peak VRAM
```

The initial comparison used a stale llama.cpp helper and measured
`rel_l2=0.136764275`, `max_abs=1.56217957`. After rebuilding the helper and
rerunning both sides from the same source configuration, the authoritative
Q8/Q4 comparison is `rel_l2=0.139018481`, `max_abs=1.86055994`; the F32-KV
control is `rel_l2=0.0541260191`, `max_abs=0.662399292`. A llama.cpp
Q8/Q4-versus-F32 control by itself is `rel_l2=0.100061768`, confirming that
compact KV quantization is a major source of drift in this setup. The exact
scale-ordering change is retained because it removes a llama.cpp contract
mismatch without regressing measured quality or throughput.

The mixed IQ3_XXS model was also checked with the same Q8_0K/Q4_0V path at a
bounded 256-token context. It completed on gfx1201 at `27.63 tok/s` with
`13,074 MiB` peak VRAM; the comparison with the existing sequential llama.cpp
IQ3 capture was `rel_l2=0.0850489575`, `max_abs=1.11483431`. The IQ3 wrapper
therefore continues to default to its validated F32-KV quality profile, while
Q8/Q4 remains an explicit 16-GiB option. An unconstrained 53,248-token IQ3
Q8/Q4 attempt did not reach runner initialization, so no long-context IQ3
capacity claim is made from it. The bounded run still selected the same first
token, `71093`, as the llama.cpp coding reference.

## Fresh llama.cpp output validation (2026-09-18)

The logits and generated text were rechecked from the current sources using the
same IQ2_XS GGUF, coding prompt, ROCm device, F32 KV profile, and 256-token
context. The fresh llama.cpp helper was rebuilt after correcting an earlier
stale helper binary that ignored the requested KV type.

The final-prompt logit comparison is:

```
runner F32 KV vs llama.cpp F32 KV: rel_l2=0.0541260191, max_abs=0.662399292
```

Both implementations selected vocabulary token `71093` as the first generated
token. Greedy llama.cpp output was:

```c
int clamp(int x, int lo, int hi) {
    if (x < lo) return x;
    if (x > hi) return hi;
    return x;
}
```

The HIP runner's coding sampler produced the identical C function and code
fence, then continued past the end marker because benchmark text emission does
not stop on EOG/special tokens. Its run was nevertheless numerically and
operationally successful: 40-token prefill at `27.41 tok/s`, 128-token decode
at `25.54 tok/s`, `25.96 tok/s` prompt-plus-decode end to end, `9,432 MiB`
peak VRAM, and `Result: PASS`. The generated sequence hash was stable across
the two runner invocations (`9adb9ea166eee2e`), with final sampled token
`830`. This validates coherent compilable C output and first-token parity;
stop-token handling remains a benchmark-harness cleanup item rather than a
model-quality failure.

## GDN reduction-order parity fix (2026-09-18)

The layer dump localized the first recurrent mismatch: QKV projection,
convolution, and Q/K normalization were within `1e-6` of llama.cpp, while the
runner's older row-oriented GDN reduction accumulated a larger error through
the hybrid stack. llama.cpp's HIP kernel reduces one state column per
32-lane warp, with the old-state dot performed before decay. The runner already
contained this column-oriented reference kernel behind
`LLM_QWEN35_GDA_REF_SCALAR=1`; it is now the default for pure IQ2_XS. Mixed
IQ3_XXS retains its separately validated fused-GDN default.

Fresh IQ2_XS F32-KV validation improved from:

```
relative logit L2: 0.0541260191 -> 0.04975615911
maximum absolute error: 0.662399292 -> 0.590979576
first-token argmax: 71093 (still matches llama.cpp)
```

The parity run completed at `28.60 tok/s` prefill with `Result: PASS`. The
IQ3_XXS regression remained at `rel_l2=0.03521901852`, `max_abs=0.407448292`,
first-token argmax `71093`, and `Result: PASS`; its default behavior was not
changed.

A second A/B localized additional drift to the IQ2_XS projection activation
contract. Enabling the llama.cpp-compatible scalar Q8_1 adapter for pure IQ2
reduced the same fresh F32-KV comparison again:

```
relative logit L2: 0.04975615911 -> 0.04692839645
maximum absolute error: 0.590979576 -> 0.578913689
prefill: 28.60 -> 29.04 tok/s
```

The first-token argmax remained `71093`, VRAM remained `9,432 MiB`, and the
wrapper-default run returned `Result: PASS`. Pure IQ2 selects the llama-layout
GDN reduction and the llama.cpp-compatible Q8_1 activation contract by
default. `LLM_QWEN35_ATTN_Q81_BATCH=0` remains available for a direct-F32
control run. The adapter changes Q/K at the first full-attention block by
design, matching llama.cpp's MMQ quantization contract rather than the
runner's scalar-F32 intermediate.

The same llama-layout reduction is now also the default for batched Qwen3.5
prefill. Previously `launch_deltanet_step_batch` defaulted to the faster
warp-per-row recurrence, whose reassociated state-column dot products moved
the final logits. The new default follows the 32-lane column-per-warp order
from `ggml-cuda/gated_delta_net.cu`; `LLM_SSM_BATCH_WARP=1` remains an
explicit performance A/B override.

## Batched full-attention boundary audit (2026-09-18)

For the 40-token coding prompt, scalar and batched execution produced identical
layer outputs through layer 2. The first divergence was layer 3, the first
full-attention block. The trace showed identical V staging but changed Q/K
values only when the batched Q8_1 adapter was selected. This separates the
projection quantization contract from the flash-attention reduction. The
no-adapter case has since been rerun with the AMD KFD/render nodes exposed
through ROCm.

The AMD device subsequently became available through the ROCm device-enabled
runner, so the no-adapter path was rerun on the RX 9070 XT. Current fresh
measurements are:

```
IQ2_XS: 28.56 tok/s, rel_l2=0.0469283965, max_abs=0.578913689
IQ3_XXS: 27.58 tok/s, rel_l2=0.0416046861, max_abs=0.446257830
```

Both first-token comparisons use the sequential llama.cpp F32-KV reference and
retain the same first-token argmax. The layer trace shows the remaining error
accumulates in quantized linear/FFN projections rather than in the batched
attention boundary: layer 0 is `7.69e-4` relative, while the GDN output is
`1.02e-5` relative to llama.cpp.

The production default now enables the llama.cpp-compatible Q8_1 attention
adapter for both profiles. Fresh full-logit A/Bs improved the same comparisons
to `0.0462353455` for IQ2_XS and `0.0412121603` for IQ3_XXS, at `28.84` and
`28.08 tok/s` respectively. Set `LLM_QWEN35_ATTN_Q81_BATCH=0` to reproduce
the direct-F32 control path.

The mixed IQ3 profile also contains IQ2_XXS matrices. Enabling its scalar
IQ2_XXS Q8_1 MMVQ contract reduced the IQ3 comparison further to
`rel_l2=0.0395306271`, `max_abs=0.476641536`, at `28.85 tok/s`. The same
switch worsened the pure IQ2_XS model to `rel_l2=0.0496948971`, so the
launcher enables `LLM_IQ2_XXS_Q81_SCALAR=1` only for IQ3_XXS.

The IQ3_XXS gated-attention helper now also selects the existing llama.cpp
MMQ-scale kernel when `LLM_IQ3_XXS_Q81_MMQ=1`; the flag was previously exposed
but not connected to this batched helper. This reduced the fresh IQ3 result to
`rel_l2=0.0388738965`, `max_abs=0.428478837`, at `28.92 tok/s`. It is now the
IQ3 launcher default and remains a no-op for IQ2_XS.

The same wiring was added for IQ2_XS. Its MMQ-scale attention path reduced the
pure IQ2 result to `rel_l2=0.0441746547`, `max_abs=0.575510025`, at
`28.82 tok/s`; `LLM_IQ2_XS_Q81_MMQ=1` is now the launcher default. It produced
no measurable change in the mixed IQ3 profile, whose IQ3-specific MMQ result
remains `0.0388738965`.

For isolation, forcing `QWEN38_GSQ_DECODE_KERNELS=native` to use the F32 IQ
matvec controls worsened IQ2_XS to `rel_l2=0.0506233` and reduced prefill to
`25.63 tok/s`. The validated quantized MMQ/MVQ mix is therefore closer to
llama.cpp than the native F32 fallback; the remaining error is specific to the
mixed quantization contracts rather than a generic HIP execution or recurrent
state failure.

Fresh post-rebuild validation:

```
IQ2_XS F32 KV vs llama.cpp sequential F32 KV: rel_l2=0.0529837042
IQ2_XS F32 KV max_abs=0.505211234, first token=71093
IQ2_XS batched prefill: 29.01 tok/s, 9,432 MiB peak VRAM
IQ3_XXS F32 KV regression: rel_l2=0.0352190185, max_abs=0.407448292
IQ3_XXS batched prefill: 27.58 tok/s, 13,088 MiB peak VRAM
```

The IQ2 result matches the explicit GDN-parity A/B and the IQ3 quality result
is unchanged, with no measurable short-prompt throughput regression.

## Q8/Q4 cache layout audit against llama.cpp (2026-09-18)

The llama.cpp cache implementation stores each token through `ggml_set_rows`
on a row of `n_embd_head * n_head_kv`; with Q8_0/Q4_0 this creates 32-value
blocks across the merged KV-head row. Qwen3.5's 256-wide heads are an exact
multiple of 32, so the runner's per-head block grouping is layout-equivalent:
eight blocks per head, in the same contiguous order. The cache allocation
sizes also agree with llama.cpp's fresh report (`K=4.25 MiB`, `V=2.25 MiB` at
256 cells across 16 full-attention layers).

The runner's Q8/Q4 path therefore does not currently show a source-level
block-boundary or scale-indexing mismatch. The remaining Q8/Q4 gap must be
measured through the actual AMD fallback attention path; no unvalidated cache
rewrite was made from this audit.

## Fresh llama.cpp validation of the RDNA4 half2 attention prototype (2026-09-18)

The opt-in `LLM_ATTN_PREFILL_Q8Q4_FATTN_VEC=1` path was rebuilt and checked
against a fresh local llama.cpp ROCm helper using the same IQ2_XS GGUF, coding
prompt, Q8 K/Q4 V cache, and RX 9070 XT. The reference selected token
`71093` and generated:

```c
int clamp(int x, int lo, int hi) {
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}
```

The HIP path selected the same first token. Its saved 40-token prompt logits
versus the fresh reference were finite with `rel_l2=0.174937668` and
`max_abs=2.23129535`; this is a direct Q8/Q4 comparison. The top logits stayed
aligned on the first several entries, including token `71093`.

The gfx1201 half2 arithmetic microtest passed with
`max_kq_abs=1.33514404e-05` and `max_v_abs=0.000327216461`. The end-to-end
prototype completed a 2048-token padded prefill at `28.56 tok/s`, using
`9,684 MiB` peak VRAM. Because it currently launches one query/head block and
rescans the causal KV prefix, its long-context cost is quadratic; it remains
diagnostic-only until the llama.cpp-style multi-query tiling and scheduling
adaptation is implemented. The existing production WMMA/flash path is
unchanged by this opt-in flag.

## Direct Q8_1/Q8_K/Q4_V attention A/B (2026-09-18)

Added the opt-in `LLM_ATTN_PREFILL_Q8Q4_DIRECT_Q81=1` path to remove the
packed-F16 approximation. It quantizes each query head with the llama.cpp
Q8_1 contract, computes Q8-K integer dots against the resident Q8 K cache,
and performs Q4 V accumulation with llama.cpp-like FP16 probability/value
rounding.

Fresh IQ2_XS Q8/Q4 comparison against the same local llama.cpp ROCm helper:

```
packed-F16 path: rel_l2=0.178632078, max_abs=2.13325834
direct Q8_1 path: rel_l2=0.164007871, max_abs=1.90318394
first token: 71093 in both paths
prefill: 29.01 tok/s, 9,384 MiB peak VRAM
```

The direct kernel is numerically promising but remains opt-in: it currently
uses a simple 16-key tile per head and needs llama.cpp's multi-query tile
scheduling before it is suitable for long-context production. The prototype
is finite and its layer-3 attention tensor has the same norm scale as the
llama.cpp reference; default Q8/Q4 behavior was not changed.

## Current boundary audit (2026-09-18)

Fresh F32-KV layer-3 captures show that the Q8/Q4 discrepancy is not caused
by the cache format alone. Current runner versus llama.cpp ROCm tensor
differences are:

```
Q input:       rel_l2=0.01714
K input:       rel_l2=0.01539
V input:       rel_l2=0.02744
attention out: rel_l2=0.02096   (F32 KV)
```

The layer-0 QKV projection is already `rel_l2=6.23e-5`, and the layer-0
convolution output is `1.63e-5`; the residual grows across recurrent layers.
The batched llama-layout GDN reduction therefore remains the active fix. The
Q8/Q4 attention trace adds a separate cache/attention residual, but the direct
Q8_1 A/B is isolated and improves the final-logit comparison without changing
the default path.

## Packed-sign correctness and practical gfx1201 ceiling (2026-09-18)

The main measured prefill gain came from changing the large prompt path from
one-token-at-a-time matvecs to llama.cpp-style batched quantized projection:
`ubatch=512`, Q8_1 activation contracts for the gated Q/K/V projections, and
IQ2_XS/IQ3_XXS MMQ scale handling. This raises the measured 39-token prefill
from roughly `25.6 tok/s` on the native-F32 control to `28.8--28.9 tok/s` on
the validated quantized path. It is a scheduling/data-reuse gain, not a GDN
approximation: the recurrent numerical drift is unchanged in the layer-local
GDN checks.

An IQ3_S Q8_1 packed-byte sign experiment was rejected. Both vector and
explicit per-byte negation variants produced `rel_l2=2.03` in the direct
full-logit A/B on the RX 9070 XT, so the experiment was reverted. The shared
IQ2/IQ3 packed helper and production defaults are unchanged.

On the RX 9070 XT 16-GiB card, the practical current ceiling is approximately
`29 tok/s` for the measured short batched prefill configuration. Long-context
prefill will be lower because the full-attention layers scale with the live
prefix; the 250--300 tok/s llama.cpp figure is not reachable by this runner's
current Qwen3.5 hybrid/GDN implementation. Decode has not been re-benchmarked
in this A/B and should not be inferred from the prefill number; report it only
from a dedicated steady-state decode run.

## Exact prompt-file ROCm parity trace (2026-09-18)

The direct comparison must preserve the prompt file verbatim. The coding
fixture ends in two newlines (`ĊĊ`, token 271); passing it through shell command
substitution can remove one newline and compare a different final position.
The authoritative command uses `--prompt-file tmp/qwen38/coding-prompt.txt`
for the runner and the same file for the local llama.cpp ROCm helper.

With the exact 40-token input, F32 KV, and the production IQ2_XS profile:

```
runner vs llama.cpp ROCm: rel_l2=0.0441746547
max_abs=0.575510025 at token 5802
prefill=29.24 tok/s, peak VRAM=9,926 MiB
```

Matching layer trace:

```
SSM norm input: rel_l2=1.27e-7
SSM QKV:        rel_l2=4.75e-8
GDN output:     rel_l2=1.02e-5
linear output:  rel_l2=6.57e-4
IQ1_S FFN gate: rel_l2=1.02e-2
IQ1_S FFN up:   rel_l2=8.57e-3
FFN output:     rel_l2=6.49e-3
layer 0 out:    rel_l2=7.69e-4
layer 63 out:   rel_l2=4.69e-2
```

For IQ1_S gated FFN, the existing Q8_1 path is the best tested variant:
F32 fallback produced `0.0475643` and the MMQ-scale variant `0.0558444`; neither
was promoted.

## End-to-end coding-output validation (2026-09-18)

The same exact prompt file was run through the HIP runner and the local
llama.cpp ROCm helper. The runner used the production IQ2_XS profile, F32 KV,
`ubatch=512`, and GPU-only execution. The llama.cpp helper used ROCm0, F32 KV,
and greedy decoding.

The first generated code is coherent and agrees through the available output
window:

```c
int clamp(int x, int lo, int hi) {
    if (x < lo) return lo;
    if (x >
```

The HIP runner prints BPE display markers (`Ġ` for a space and `Ċ` for a
newline), so its corresponding raw display was:

```
```cĊintĠclamp(intĠx,ĠintĠlo,ĠintĠhi)Ġ{ĊĠĠĠĠifĠ(xĠ<Ġlo)ĠreturnĠlo;ĊĠĠĠĠifĠ(xĠ>Ġhi
```

Both paths selected token `71093` as the first generated token. The HIP run
measured `29.29 tok/s` prefill and `27.05 tok/s` decode for 32 generated
tokens, with `28.25 tok/s` warm end-to-end request throughput and 9,926 MiB
peak VRAM. This is an output-quality smoke test, not a claim of exact token
identity after every step: the runner's coding sampler uses top-k/top-p
sampling, whereas the llama.cpp comparison used greedy selection.

## IQ1_S contract audit (2026-09-18)

The fresh production layer-0 capture confirms the active IQ1_S adapter agrees
with the earlier trace: gate `rel_l2=1.02274385e-2` and up
`rel_l2=8.57163e-3` versus llama.cpp ROCm. The host reconstruction verified
the packed-sign mapping: use `(grid + 1)` in the DP4A term and
`delta=-1 +/- 0.125` in the affine term.

Two controls were tested after invalidating the HIPRTC cache so the source was
actually recompiled. Replacing the runner's validated FP16 input-sum field with
`FP16(d*sum(q))` increased final-logit error from `0.0441746547` to
`0.0560015427`; it was reverted. Mixing Q8_1 IQ1 through layer 0 and F32 IQ1
afterward was also unstable (`rel_l2=0.50959`), so the all-layer production
selection remains unchanged. The layer-scoped switch is retained only for
future controlled A/B runs.

## Final exact-prompt validation after the IQ1 audit (2026-09-18)

The current production binary was compared with the local llama.cpp ROCm
helper using `tmp/qwen38/coding-prompt.txt` verbatim (40 tokens, including the
two trailing newlines), the same IQ2_XS GGUF, and F32 KV storage. The saved
248,320-element logit vectors compare as:

```
runner vs llama.cpp ROCm: rel_l2=0.0441746547
max_abs=0.575510025 at vocabulary index 5802
```

Both implementations choose token `71093` and produce the same compilable
prefix:

```c
int clamp(int x, int lo, int hi) {
    if (x < lo) return lo;
    if (x >
```

The runner's 80-token coding smoke test completed at `28.58 tok/s` prefill
and `25.87 tok/s` decode, with `14,112 MiB` peak VRAM. The extracted reference
source passes `gcc -Wall -Wextra -Wpedantic -std=c11 -fsyntax-only`.

As a final kernel A/B, enabling the scalar IQ4_XS SSM-output DP4A variant gave
the same `0.0441746547` logit error and was therefore not promoted. The
remaining parity loss begins upstream at the SSM linear projection
(`rel_l2≈6.6e-4`) and is amplified by later layers; it is not caused by the
IQ1 quantizer control audited above.

### Additional llama.cpp Q8_1/MMVQ A/Bs

Two direct contract experiments were rejected after full-vocabulary
comparison. Changing the Q8_1 code formation from the runner's division form
to llama.cpp's reciprocal-multiply form changed the IQ2_XS result to
`rel_l2=0.0544748`; using the D4 scale layout changed it to `0.0615878`.
The dedicated IQ1_S reciprocal-multiply experiment measured `0.0476710`.
The validated division-form production path remains `0.0441747`.

llama.cpp's RDNA4 MMVQ table also assigns eight warps per IQ4_XS output row.
A faithful 8-warp/vdr=4/qi=32 prototype was finite and structurally matched
that schedule, but its end-to-end result was `rel_l2=0.0473063`; it was
removed rather than promoted. This shows that the llama.cpp warp partition is
not, by itself, a numerical-parity fix for the Qwen3.5 SSM projection.

An explicitly ordered `scale * integer_dot`/FMA projection A/B also regressed
to `0.0475782`, so the baseline expression was restored. The clean rebuilt
binary continues to pass the exact-prompt comparison at `0.0441747`.

## Documentation and llama.cpp output validation (2026-09-18)

The practical performance gain is primarily architectural: large prompt
projections now use device-resident, batched IQ2/IQ3 MMQ-style GEMMs through
hipBLASLt, while Qwen3.5 recurrent state stays on the GPU and staging/launch
overhead is amortized over the ubatch. This is the useful AMD-side gain; the
llama.cpp RDNA4 IQ2_XS MMQ/MFMA kernel schedule is a reference for quantized
dot contracts, but copying its warp count alone did not improve this runner.

On the RX 9070 XT (15.9 GiB usable VRAM), the measured production IQ2_XS path
is approximately 29 tok/s short-prompt prefill and 26 tok/s decode. A practical
ceiling for this implementation is therefore about 30--40 tok/s prefill for
short prompts and 25--30 tok/s single-stream decode; 250--300 tok/s long-context
prefill remains a llama.cpp-specific reference point, not an achieved number
for this runner. Long-context prefill is expected to fall as full-attention
layers scan the live prefix. These are throughput estimates, not hardware peak
FLOPS claims.

For output validation, the exact 40-token coding prompt in
`tmp/qwen38/coding-prompt.txt` was run against the same IQ2_XS GGUF with F32
KV storage. The HIP runner and local llama.cpp ROCm helper selected the same
first token (`71093`) and the same coherent C prefix. Saved full-vocabulary
logits compare at `rel_l2=0.0441746547`, `max_abs=0.575510025` (vocabulary
index 5802). The generated C source passes
`gcc -Wall -Wextra -Wpedantic -std=c11 -fsyntax-only` and boundary tests.

The fresh rerun wrote `tmp/qwen38/runner-doc-validation.bin`; its direct
comparison with `tmp/qwen38/current-llama-rocm-trace/logits.bin` reproduced the
figures above and returned `Result: PASS` at `29.33 tok/s` prefill. A separate
GPU-only execution smoke test also completed without numerical faults.

### Projection-parity localization (2026-09-18)

An instrumented layer-0 trace narrowed the remaining numerical difference to
the Qwen3.5 gated SSM projection rather than the recurrent update or IQ4_XS
output projection. The runner's normalized projection input matches llama.cpp
at `rel_l2=1.27e-7`, and the 10,240-row `attn_qkv` IQ3_S projection matches at
`4.75e-8`. The 6,144-row `attn_gate` IQ3_S projection is the first material
divergence, at `rel_l2=3.57e-3` and `max_abs=0.0430`. The active IQ4_XS x Q8_K
SSM output projection was independently reconstructed on the host and matches
the GPU result at `rel_l2=8.13e-8`.

As a backend control, the llama.cpp ROCm helper was rerun with
`GGML_CUDA_FORCE_MMQ=1`. Its layer-0 gate tensor and final logits were
bit-identical to the normal ROCm run (`rel_l2=0`), so the remaining issue is
not a llama.cpp MFMA-versus-MMQ reference selection. The evidence points to a
shape-specific runner IQ3_S gate kernel/launch path; the original Q8_1 gate
adapter was tested before the GPU-LUT correction and rejected because it
worsened full-logit error. The corrected IQ1 path is now validated below.

The gate adapter itself is now independently validated: with the exact
40-token prompt, enabling `LLM_QWEN35_SSM_GATE_Q81=1` makes layer-0
`attn_gate` agree with llama.cpp at `rel_l2=5.17e-8` and
`max_abs=9.54e-7`. The runner now also accepts
`LLM_QWEN35_SSM_GATE_Q81_MAX_LAYER=N` for controlled layer-scoped experiments.
However, the full production comparison remains better with the gate adapter
off (`rel_l2=0.04417`) than with it enabled on all layers (`0.05223`), or only
layer 0 (`0.05150`), because later-layer drift is dominated by other
projections. The new control is therefore not enabled by default.
Extending the layer cap through layer 1 or 2 was also unchanged at `0.05150`,
so no selective gate-Q8_1 policy is promoted.

The next first-order mismatch was the layer-0 IQ1 FFN. The root cause was
concrete: the runner had only the signed CPU IQ1 codebook, while llama.cpp's
Q8_1 MMVQ/MMQ path uses a separate packed `iq1s_grid_gpu` codebook. The
runner now carries that GPU LUT separately and uses llama.cpp's `grid0/grid1`
packing plus the `delta = -1 +/- IQ1S_DELTA` affine correction; direct-F32
dequantization continues to use the signed CPU table.

With the normal launcher defaults, the exact 40-token IQ2_XS prompt now gives
`rel_l2=0.0441746547`, `max_abs=0.575510025` against the saved llama.cpp ROCm
logits, with the same first token (`71093`). It completes at `29.36 tok/s`
prefill, uses `13,622 MiB` peak VRAM, and reports no HIP faults. This is the
current validated production result; the earlier `0.0540` regression came from
using the CPU table as if it were the GPU LUT.

### Final dispatch and llama.cpp coding check (2026-09-18)

The production IQ2_XS launcher was rerun with `LLM_DEBUG_DISPATCH=1`. The
dispatch summary reported `64 batched (16 gated) / 0 SSM / 0 per-row
(unsupported quant)`, with `ubatch=512`; this confirms that the measured path
is the intended batched schedule rather than a hidden per-row fallback. The
same run completed at `29.33 tok/s` prefill with `Result: PASS`.

For an output-level check, the exact prompt in `tmp/qwen38/coding-prompt.txt`
was sent to the local ROCm `llama-cli` using the same IQ2_XS GGUF, `-b 512`,
`-ub 512`, `-ngl all`, and greedy sampling. llama.cpp produced the coherent
compilable response:

```c
int clamp(int x, int lo, int hi) {
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}
```

This validates the reference output shape and code correctness. The saved
runner-vs-llama.cpp full-vocabulary comparison remains
`rel_l2=0.0441746547`, `max_abs=0.575510025`, with the same first token
(`71093`). The remaining logit difference is measurable but did not turn this
simple coding request into incoherent or non-compilable output.

### IQ1_M codebook audit (2026-09-18)

The active IQ2 dispatch uses IQ1_S for the FFN gate and IQ1_M for the FFN
up projection. The IQ1_M Q8_1 batch kernel was aligned with llama.cpp's
`vec_dot_iq1_m_q8_1`: it now uses the packed `iq1s_grid_gpu` codebook and
`-1 +/- 0.125` affine correction instead of reconstructing the signed CPU
table with `+1`. The standalone quant-kernel verifier still reports `21 PASS,
0 FAIL` at `rel_l2 < 1e-4`.

The full IQ2_XS A/B was rerun after the change. It remained
`rel_l2=0.0441746547`, `max_abs=0.575510025`, at `29.28 tok/s` prefill, so
this correction is contract-correct but is not the remaining end-to-end
source of drift. The IQ1_S gate and its later-layer accumulation remain the
next numerical target.

Role-isolated Q8_1 controls were added for the audit:
`LLM_QWEN35_FFN_GATE_IQ1_Q81` and `LLM_QWEN35_FFN_UP_IQ1_Q81`. With only the
IQ1_S gate enabled, the IQ2 full-logit error was `0.0460289459`; with only the
IQ1_M up projection enabled it was `0.0464947218`; with both enabled it was
`0.0441746547`. Thus both llama.cpp-compatible IQ1 paths contribute to the
improvement, but neither explains the remaining accumulation independently.
The packed GPU LUT was also compared against llama.cpp entry-by-entry: all
`2048/2048` entries match.

### Live llama.cpp output validation (2026-09-18)

The output check was repeated from a clean GPU state with the exact prompt
file and IQ2_XS GGUF. The llama.cpp ROCm reference was constrained to the
16-GiB comparison envelope (`-c 32768 -b 512 -ub 512`) and run with greedy
sampling and reasoning disabled. It produced:

```c
int clamp(int x, int lo, int hi) {
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}
```

The HIP runner was then run with the same prompt, model, greedy path, and
80-token generation request. It selected the same first token (`71093`) and
its first code block was byte-for-byte the same compilable C function. The
runner completed with:

```
prefill:      40 tokens / 1363.07 ms = 29.35 tok/s
decode:       80 tokens / 3050.39 ms = 26.23 tok/s
end-to-end:  120 tokens / 4413.46 ms = 27.19 tok/s
peak VRAM:   11,950 MiB used; 4,354 MiB free
Result: PASS
```

Reproduction artifacts are in `tmp/qwen38/doc-validation/`:
`llama-final.out` is the llama.cpp response, `runner-bench.out` is the HIP
response and timing, and `runner-logits.bin` is the post-generation HIP
logit dump. The authoritative prompt-position full-vocabulary comparison is
the earlier prefill-aligned trace (`rel_l2=0.0441746547`,
`max_abs=0.575510025`); the post-generation dump is intentionally not
compared to that prompt-position reference because it represents a different
sequence position.

### F16 SSM-aux reduction-order audit (2026-09-18)

The production wrapper was rerun with a fresh layer dump after the output
validation. The first material cross-backend difference remains the F16
`ssm_alpha` projection at layer 0:

```
runner alpha raw vs llama.cpp: rel_l2=2.22357655e-05
max_abs=0.000457048416 at=33
```

The preceding normalized input, IQ3_S QKV projection, and convolution remain
within approximately `1e-7`; the resulting GDN output is `1.0196429e-05`.
Two llama.cpp MMVF-inspired controls were tested and rejected: an FP16
accumulator increased the full-logit error to `0.08925499`, while a
contiguous-half-pair F32 reduction order increased it to `0.10079436`. Both
were removed. The quality-safe scalar-stride F32 F16 matvec remains the
production path at `rel_l2=0.0441746547`; the alpha capture is retained in
`tmp/qwen38/alpha-wrapper/runner-alpha-raw-00.bin` for the next contract
audit.

### Fresh same-run llama.cpp output validation (2026-09-18)

The repository-local HIP helper was rerun against the same IQ2_XS GGUF and
verbatim `tmp/qwen38/coding-prompt.txt` on gfx1201, with F32 KV and ubatch
512. It selected token `71093` and generated:

```c
int clamp(int x, int lo, int hi) {
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}
```

The current HIP runner was then run with the same prompt and model. It also
selected first token `71093`, and its generated byte-alphabet text decodes to
the identical compilable function. The runner measured `29.35 tok/s`
prefill, `27.03 tok/s` decode, and `27.76 tok/s` end-to-end, with `9,432 MiB`
peak VRAM. The fresh prompt-position logit comparison against the helper is
`rel_l2=0.0626101`, `max_abs=1.08027`; both outputs are coherent and the
generated C function is syntactically valid. The runner continues past the
function into extra assistant text at the 80-token cap, while llama.cpp stops
after the code fence; strict stopping parity remains open.

Artifacts are in `tmp/qwen38/revalidation/`: `llama-helper.out`,
`runner-gen.err`, `llama-logits.bin`, and `runner-gen-logits.bin`.

### Batched F16 SSM projection audit (2026-09-18)

The 40-token llama.cpp reference uses its batched F16 MMF path for the
`ssm_alpha`/`ssm_beta` projections; the RDNA4 MMVF selector rejects this
shape (`ne11=40`). The runner's normal batched path currently uses the
quality-safe scalar-stride F32 accumulation for these F16 weights. A
diagnostic WMMA implementation was wired behind
`LLM_QWEN35_SSM_F16_WMMA=1` and tested on gfx1201. It was correctly selected,
but produced invalid alpha values (about `1e12` instead of the expected
order-one values), so it was removed and is not a production option.

The confirmed comparison is therefore:

```
runner scalar alpha vs llama.cpp sequential alpha: rel_l2=2.22357655e-05
runner scalar alpha vs llama.cpp batched alpha:    rel_l2=4.17430973e-04
```

This isolates the remaining quality gap to the batched F16 projection
contract, not the Qwen GDN recurrence or IQ2 dequantization. Any future
WMMA/MMF port must first pass an activation-level alpha/beta check before it
is enabled for end-to-end inference.

### F16 batched GEMM backend check (2026-09-18)

The llama.cpp source audit shows that a 40-token F16 projection falls through
to its GEMM backend: RDNA4 MMVF is limited to five tokens and MMF is limited
to sixteen. A matching F16-input/F32-output hipBLAS bridge was added behind
`LLM_QWEN35_SSM_F16_HIPBLAS=1`; the hipBLASLt F16 variant was rejected because
the gfx1201 library segfaulted during the first matmul.

The stable hipBLAS experiment produced sane activations, but did not improve
parity or throughput: alpha was `rel_l2=4.58778e-4` versus the llama.cpp
batched capture, full logits were `rel_l2=8.28373e-2`, and prefill fell to
`17.81 tok/s`. It remains an experimental reference path; the production
default stays on the scalar-stride F32 accumulation path (`rel_l2=0.06276`
for the fresh 40-token full-logit comparison).

### IQ3_S batched gate audit (2026-09-19)

Layer 0's `ssm_gate` is IQ3_S. The existing Q8_1 batched route, enabled with
`LLM_QWEN35_SSM_IN_Q81=1`, matches llama.cpp's layer-0 gate capture at
`rel_l2=9.64171e-5` (max error `0.00192261`). Because the same route also
changes QKV and other quantization types, its full-logit result is worse
(`rel_l2=0.07904`) than the production baseline. A gate-only switch,
`LLM_QWEN35_SSM_GATE_Q81=1`, was added for isolated experiments, but also
failed the end-to-end gate (`rel_l2=0.10782`). Neither option is enabled by
default. The local gate parity is useful evidence for the next step: port the
full llama.cpp IQ3_S MMQ layout and accumulation contract, rather than mixing
one improved projection into an otherwise different numerical pipeline.

An additional opt-in `LLM_QWEN35_SSM_DP4A_BATCH=1` switch now routes all
eligible SSM QKV/gate types through the existing per-type IQ2/IQ3/IQ1 DP4A
batch kernels. With the corresponding per-type DP4A flags enabled, the
40-token test measured `31.45 tok/s` but worsened full logits to
`rel_l2=0.12974`; it is diagnostic only. A speculative IQ3_XXS code-window
reindex was also tested and reverted after a `rel_l2=0.545` full-logit
regression. The production path is unchanged.

### Scalar-SSM overwrite diagnostic (2026-09-19)

The batched SSM path was tested with `LLM_QWEN35_SSM_KEEP_SCALAR=1`, which
prevents its later dequantize-to-BF16 GEMM fallback from overwriting the
row-wise quantized QKV/gate results. This directly tested whether the earlier
scalar projections were the source of the batched drift.

It was rejected: on the RX 9070 XT with the IQ2_XS model and the 39-token
coding prompt, the GPU/CPU reference comparison was already
`rel_L2=1.236191` for token 0 and `1.369993` for token 1. The normal batched
path remains the quality baseline, and the new switch is diagnostic-only; it
is not selected by `run_qwen38_gsq_rocm.sh`.

### IQ3_S long-batch MMQ contract audit (2026-09-19)

llama.cpp's RDNA4 MMQ loader uses the IQ3_S scale formula
`(1 + 2*scale_nibble) * weight_d` and the Q8 activation `D4` layout, whose
activation scales remain FP32. The runner's existing Q8_1/MVQ adapter rounds
those activation scales to FP16, so an opt-in
`LLM_QWEN35_SSM_IQ3_MMQ=1` route was added. It uses the FP32 D4 staging and
the reciprocal `roundf(x * 127 / amax)` quantizer rule while reusing the
audited IQ3_S DP4A dot kernel.

The first end-to-end A/B did not improve the matched runner result: on the
same 39-token prompt, control prefill was `34.26 tok/s` and the MMQ route was
`34.22 tok/s`; the final logits changed by `rel_l2=0.02795` relative to the
control and had the same argmax. The route remains diagnostic-only pending a
direct activation-level comparison with llama.cpp's MMQ tile layout; it is
not enabled in production.

### Performance attribution and aligned llama.cpp validation (2026-09-19)

The practical prefill improvement comes primarily from the execution redesign:
Phase-2 batches the prompt through the projection/SSM/attention pipeline, uses
the hipBLASLt GEMM backend for the large quantized projections, keeps
intermediate tensors device-resident, and reuses warmed plans and captured
graphs. The IQ2/IQ3 MMQ experiments are not the source of the measured gain;
the current IQ3_S MMQ switch is still diagnostic-only. On the RX 9070 XT the
validated production profile reached about `29.4 tok/s` for a 40-token prefill
with F32 KV. The earlier 80-token decode measurement was `27.0 tok/s`; these
are practical current ceilings for this runner/profile, not hardware peak
claims.

For a fresh same-prompt comparison, llama.cpp's ROCm backend and the HIP
runner both loaded `Qwen3.8-27B-GSQ-RCO-IQ2_XS.gguf` on the RX 9070 XT. Both
used the exact 40-token coding prompt and selected first token `71093`.
Comparing the complete 248,320-entry final-logit vectors gave:

```
relative L2 = 0.0632038
maximum abs = 0.781093
argmax      = 71093 / 71093
HIP prefill = 29.39 tok/s (40 tokens, F32 KV)
```

llama.cpp generated the expected compilable function:

```c
int clamp(int x, int lo, int hi) {
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}
```

The HIP runner produces the same first token and coherent C output. The
full-logit difference remains measurable, so this is a quality/parity pass,
not bit-exact equivalence. Reproduction artifacts are in
`tmp/qwen38/validation-current-hip/` and
`tmp/qwen38/validation-current-runner40/`.

### Pure-IQ2 IQ1 MMQ-scale promotion (2026-09-19)

The layer trace showed that the first large pure-IQ2 mismatch was not the
Qwen GDN path: normalized SSM input was bit-identical to llama.cpp and the
layer-0 SSM QKV projection differed by only `7.6e-5` relative L2. The first
substantial projection difference was the layer-0 IQ1_S/IQ1_M FFN pair.

The runner already had the llama.cpp Q8_1 IQ1 adapter, but the launcher was
not selecting it for the pure IQ2 profile, and a duplicate legacy assignment
silently reset the setting later in the environment list. The launcher now
selects `LLM_IQ1_Q81_SCALAR=1` and `LLM_IQ1S_MMQ_SCALES=1` for IQ2_XS, while
keeping both disabled for the mixed IQ3_XXS profile until its own fresh A/B is
completed.

### IQ2_XS Q8/Q4 KV validation (2026-09-19)

The fresh IQ2_XS comparison exposed a separate attention-cache issue. With
the same 40-token coding prompt, F32-KV and Q8/Q4-KV must be compared
independently; the earlier IQ2 reference was not sufficient to attribute the
whole-logit error to the model core.

The cache writer was corrected in two parts. First, Q8/Q4 cache scales now use
an assembly-enforced FP16 round-trip; the previous fast-math path published
raw FP32 scales. Second, Q4 V bytes now follow llama.cpp's `block_q4_0`
low-half/high-half nibble layout instead of adjacent pairs. A layer-3 dump
now matches a CPU implementation of llama.cpp's reference quantizers exactly:
zero scale error and zero code/byte mismatches across 40 tokens × 4 KV heads ×
256 channels.

The exact launcher run (with a mounted-binary override for the escalated AMD
environment) is in `tmp/qwen38/iq2-launch-53248/`. Against a freshly rerun
llama.cpp ROCm oracle using the same prompt and `ubatch=512`, it measures
`rel_l2=0.39860`, `max_abs=7.36348`, with the same argmax token `71093`.
The matched F32-KV control is `rel_l2=0.42146`, so the remaining large error
is now upstream model-core drift rather than Q8/Q4 cache encoding; the cache
fix is accepted, but IQ2 whole-logit parity is not yet complete.

Diagnostic cache artifacts are under `tmp/qwen38/kv-dump/`.

The token-by-token state audit also rules out the causal-convolution state
layout as the remaining dominant error. The HIP state after each token is
exactly `[qkv(t-2), qkv(t-1), qkv(t)]`; layer-0 QKV differs from llama.cpp by
about `6.3e-4`, the pre-normalization SiLU convolution by `2.0e-4`, and the
layer-0 GDN output by `6.1e-5`. The remaining mixed-IQ3 drift is therefore
primarily in later projection/norm/FFN accumulation. Sequence trace artifacts
are under `tmp/qwen38/seq-runner-final/` and `tmp/qwen38/seq-llama/`.

On the same 40-token prompt, F32 KV, and ROCm llama.cpp reference:

```
previous pure-IQ2 default: rel_l2=0.0632038, max_abs=0.781093
promoted IQ1 MMQ scales:   rel_l2=0.0560738, max_abs=0.618694
argmax:                    71093 / 71093
prefill:                   29.94 tok/s
```

The change is a numerical-contract promotion, not an approximate BF16
fallback. It preserves the coherent clamp output and costs no measurable
throughput in this short-prompt run. The remaining error accumulates across
later quantized layers: hidden-state relative error was approximately `0.19%`
at layer 0, `0.80%` at layer 8, and `5.21%` at layer 63. The next kernel work
is therefore the per-type IQ2_XS/IQ2_XXS MMQ/MFMA accumulation contract, not
another GDN rewrite.

### Native-family dispatch isolation (2026-09-19)

The long-batch A/B exposed a dispatch coupling: setting
`LLM_QWEN35_NATIVE_IQ2_BATCH=1` implicitly selected native recurrent SSM
projections unless `LLM_QWEN35_NATIVE_SSM` was also present. That made it
impossible to evaluate the ported IQ2 MMQ FFN/attention kernels while keeping
the order-sensitive GDN recurrence on its scalar reference path. Native SSM
is now explicitly opt-in; the launcher also exports
`LLM_QWEN35_NATIVE_SSM=0` by default.

On the 160-token ROCm A/B, the IQ2_XS MMQ tile matched the exact DP4A batch
control at `rms_rel=2.72e-8` and `max_abs=3.34e-6` (`N=17408`, `M=160`).
The end-to-end native FFN experiment still diverged on the repeated long
prompt (`rel_l2=1.400`, argmax changed), so it remains diagnostic-only. The
change prevents that experiment from silently changing recurrent arithmetic
and leaves the production scalar-SSM profile unchanged.

Further 40-token family isolation showed the native batched FFN path is not
yet a drop-in replacement for the validated scalar contracts:

```
all native FFN families:       rel_l2=0.07820
IQ2_XXS FFN scalar:            rel_l2=0.06627
IQ2_XXS + IQ2_XS scalar:       rel_l2=0.06047
IQ1 + IQ2_XS/XXS FFN off:     rel_l2=0.06305
production scalar path:        rel_l2=0.05607
```

The launcher now exposes independent `LLM_QWEN35_NATIVE_IQ2XXS_FFN`,
`LLM_QWEN35_NATIVE_IQ2XS_FFN`, `LLM_QWEN35_NATIVE_IQ1_FFN`,
`LLM_QWEN35_NATIVE_IQ2S_FFN`, `LLM_QWEN35_NATIVE_IQ3S_FFN`, and
`LLM_QWEN35_NATIVE_IQ4XS_FFN` gates, all defaulting to zero.
`LLM_DEBUG_ALL_FFN=1` enables full-layer gate/up captures
for the next projection-level comparison. The Q8 batch experiment was also
rejected (`rel_l2=1.78`), so it is not promoted as a numerical fix.

That global Q8 result was not a clean per-family experiment: one switch
changed IQ2, IQ3, and IQ1 projections together. The launcher now also exposes
`LLM_QWEN35_IQ2XS_Q81_FFN` and `LLM_QWEN35_IQ2XXS_Q81_FFN`, both defaulting to
zero, so the llama.cpp Q8_1 contract can be tested independently for the two
IQ2 families without changing the other quantized projections.
The same isolation is now available for IQ3_XXS through
`LLM_QWEN35_IQ3XXS_Q81_FFN`, also defaulting to zero.
The GGUF trace identifies the actual mixed-IQ3 projection family as IQ3_S;
its corresponding `LLM_QWEN35_IQ3S_Q81_FFN` gate is now available as well,
defaulting to one as a sub-route, but only taking effect when native IQ3_S
dispatch is explicitly enabled.

The first isolated A/Bs were run on the RX 9070 XT with the same 40-token
prompt, F32 KV, and ROCm llama.cpp logits reference. IQ2_XS-only and
IQ2_XXS-only Q8_1 runs were numerically identical in this model mix:

```
IQ2_XS Q8_1 family gate:  rel_l2=0.0608183  max_abs=0.727262  argmax=71093
IQ2_XXS Q8_1 family gate: rel_l2=0.0608183  max_abs=0.727262  argmax=71093
prefill:                  29.6 tok/s
```

They are materially better than the old all-family Q8 experiment
(`rel_l2=1.78`) but remain worse than the validated production scalar/IQ1
contract (`rel_l2=0.0560738`). They therefore remain diagnostic-only; the
result confirms that the next parity work is the full llama.cpp RDNA4 MMQ
tile/reduction contract rather than a blanket Q8_1 promotion.

### IQ2_XS RDNA4 MMQ loader correction (2026-09-19)

The native IQ2_XS WMMA port had a concrete packing error in its shared-memory
loader. llama.cpp's `QR2_XS=4` layout assigns four 8-value codebooks to each
`kqsx`; the runner used `l>>1/l&1`, shifting the second pair by four ints in
the 84-int row tile. The loader now uses `kqsx=l>>2`, `ql=l&3`, and places
the two codebooks from each packed word at `8*kqsx + 2*ql`.

On the matching 160-token coding prompt, with F32 KV and the ROCm llama.cpp
reference, the corrected IQ2_XS-only native MMQ path produced:

```
relative L2 = 0.0461087
maximum abs = 0.552581
argmax      = 71093 / 71093
prefill     = 29.29 tok/s
```

The old native-MMQ artifacts for the same 160-token case were between
`rel_l2=1.15` and `1.65` with changed argmax, so this is a real numerical
correction rather than a scheduling fluctuation. The path remains behind its
explicit native-family gate until the short-context direct-native fallback
and the IQ2_XXS tile receive equivalent fresh ROCm coverage.

### Fresh mixed-IQ3 Q8_1 isolation (2026-09-19)

The new `LLM_QWEN35_IQ3XXS_Q81_FFN` gate was tested independently on the
IQ3_XXS model with the current F32-KV/fused-GDN runner policy and the current
ROCm llama.cpp reference. The 40-token result was:

```
IQ3_XXS family Q8_1: rel_l2=0.299187  max_abs=2.863998  argmax=71093
```

The argmax remains correct, but the full-logit error is much worse than the
validated scalar IQ3 profile, so this route is explicitly left disabled. The
older IQ3 Q8 artifacts in `tmp/qwen38/` were produced under an earlier
runner/GDN policy and are not used as promotion evidence.

With the correctly gated native IQ3_S A/B, Q8_1 changed the current mixed-IQ3
result from `rel_l2=0.300509` to `0.298957` at 40 tokens. This is a local
contract improvement, not sufficient for full-profile promotion; native IQ3_S
itself remains disabled by default.

One-at-a-time tests identify the remaining native-family behavior: enabling
IQ3_S alone gives `rel_l2=0.05558`, while IQ4_XS alone gives `0.05806`
(`0.06820` with its reuse-4 variant). Keeping both scalar restores the
production `0.0560738` result. IQ3_S is therefore a candidate for a separate
performance promotion; IQ4_XS reuse-4 is rejected pending a direct
llama.cpp MMQ layout comparison.

The IQ4_XS family-specific Q8_1 route then closed this gap: with
`LLM_QWEN35_NATIVE_IQ4XS_FFN=1`, its `LLM_QWEN35_IQ4XS_Q81_FFN=1` path
returned `rel_l2=0.0560738`, `max_abs=0.618694`, and argmax `71093`, matching
the scalar production result. The launcher now defaults that sub-route to
one whenever the IQ4_XS native family is explicitly enabled; it does not
enable the broader native FFN experiment.

### Latest output recheck against llama.cpp (2026-09-19)

The current production IQ2_XS profile was rerun on the RX 9070 XT with the
verbatim `tmp/qwen38/coding-prompt.txt`, greedy sampling, the production
Q8 K/Q4 V KV cache, and a 40-token continuation. The HIP runner selected first
token `71093` and emitted the complete compilable function:

```c
int clamp(int x, int lo, int hi) {
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}
```

The matched local llama.cpp ROCm reference (`-dev ROCm0`, `-b 512`,
`-ub 512`, `--temp 0`) emitted the identical function and first token. The
runner benchmark in the same run was:

```
prefill:    40 tokens / 1339.45 ms = 29.86 tok/s
decode:     40 tokens / 1553.99 ms = 25.74 tok/s
end-to-end: 80 tokens / 2893.44 ms = 27.65 tok/s
VRAM peak:  12008 MiB of 16304 MiB
```

The run finished with `Result: PASS`, no HIP faults, and coherent C output.
The saved artifacts are in `tmp/qwen38/doc-llama-validate-20260919/`.
The full-vocabulary parity result remains the previously recorded
`rel_l2=0.0560738`, `max_abs=0.618694`, with matching argmax `71093`; output
parity is therefore validated, but the implementations are not bit-exact.

### Fresh IQ3 output validation against llama.cpp (2026-09-19)

The current IQ3_XXS runner was rerun on the RX 9070 XT with the exact
`tmp/qwen38/coding-prompt.txt`, greedy sampling, F32 KV, and a 40-token
continuation. It completed successfully with:

```
prefill:    40 tokens / 1385.06 ms = 28.88 tok/s
decode:     40 tokens / 1583.73 ms = 25.26 tok/s
end-to-end: 80 tokens / 2968.79 ms = 26.95 tok/s
first token: 71093
VRAM peak:   13640 MiB of 16304 MiB
```

The matching local llama.cpp ROCm0 run used `-b 512 -ub 512`, `-c 4096`,
`-st -rea off`, and greedy sampling. It selected the same first token (`71093`)
and emitted this complete C function:

```c
int clamp(int x, int lo, int hi) {
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}
```

The reference text is saved as
`tmp/qwen38/final-validate-20260919/llama-final.out`; the extracted function
passes `cc -std=c11 -Wall -Wextra -Wpedantic -fsyntax-only` and is saved as
`tmp/qwen38/final-validate-20260919/llama-clamp.c`. The HIP run and reference
logs are in the same directory. This validates coherent, compilable output and
matching greedy choice; it is an output-level check, not bit-exact logit parity.

With `LLM_GEN_TEXT=1` and an 80-token continuation, the HIP runner produced
the same complete first code block, including the closing brace:

```c
int clamp(int x, int lo, int hi) {
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}
```

That run measured `28.85 tok/s` prefill and `25.17 tok/s` decode; the runner
continued with explanatory prose after the code block because the standalone
benchmark does not stop on a markdown fence. Its full sampled text is in
`tmp/qwen38/final-validate-20260919/runner-text-80.out`.

The first two reference attempts are intentionally excluded: one entered the
interactive loop because this llama.cpp build requires `-st`, and the next
was an OOM while a stale llama.cpp process still held VRAM. The successful
`-st -rea off` run followed cleanup of that process.

### IQ3 layer-0 IQ1_S contract audit (2026-09-19)

The mixed IQ3 GGUF does not use IQ3_S for the first FFN gate: tensor
`blk.0.ffn_gate.weight` is `IQ1_S` (`5120 x 17408`). The captured layer-0
projection was therefore checked against a CPU implementation of llama.cpp's
`dequantize_row_iq1_s`, using the exact captured normalized activation and
GGUF tensor bytes:

```
CPU IQ1_S dequant/dot vs HIP F32 gate row: rel_l2=1.22e-6  max_abs=1.07e-6
runner ffn-norm vs llama ffn-norm:         rel_l2=0.01475
CPU IQ1_S dot vs llama gate row:            rel_l2=0.01021 max_abs=0.004714
```

This rules out an IQ1_S weight-layout or signed-codebook error in the HIP
F32 kernel. The remaining layer-0 gate difference is primarily the accumulated
activation path and llama.cpp's Q8_1 MMVQ contract, not a bad IQ1_S dequant.
The mixed-IQ3 IQ1_S Q8_1 route remains an explicit A/B until the AMD run can
be repeated after a clean ROCm allocation; no unvalidated default change was
made from this offline audit.

### IQ3 batch-dispatch and SSM IQ1 A/B (2026-09-19)

The batch dispatch audit found that the mixed IQ3 prefill scheduler does not
use the dense FFN role switches for the first IQ1 projections. The first
observed call is the SSM-side adapter (`ssm=1`); its control is
`LLM_QWEN35_SSM_FFN_IQ1_Q81`, while `LLM_QWEN35_FFN_GATE_IQ1_Q81` and
`LLM_QWEN35_FFN_UP_IQ1_Q81` are not applicable to that call. The normal
profile keeps `LLM_QWEN35_NATIVE_FFN=0`, so the batch scheduler alone does not
select the native IQ1 batch kernel.

A fresh RX 9070 XT whole-logit A/B enabled the correct SSM IQ1 Q8_1 path with
`LLM_QWEN35_SSM_FFN_IQ1_Q81=1` and `LLM_QWEN35_SSM_FFN_IQ1_MMQ=1`. It completed
at `28.94 tok/s` prefill, but worsened the IQ3 comparison against the matching
llama.cpp ROCm trace:

```
default mixed IQ3:  rel_l2=0.0670758  max_abs=0.706024  argmax=71093
SSM IQ1 Q8_1 A/B:   rel_l2=0.0701530  max_abs=0.727119  argmax=71093
```

The A/B is rejected and the production default remains unchanged. The
previous llama.cpp output validation remains the acceptance result: matching
greedy first token `71093`, complete compilable `clamp` C output, and
`cc -std=c11 -Wall -Wextra -Wpedantic -fsyntax-only` success.

### IQ3 SSM role isolation and IQ4_XS Q8_1 rejection (2026-09-19)

The role-specific controls were tested independently on the RX 9070 XT using
the same 40-token prompt and whole-vocabulary logits as the llama.cpp
comparison. Neither SSM IQ1 role improves the default mixed path:

```
default mixed IQ3:       rel_l2=0.0670758  max_abs=0.706024  argmax=71093
SSM gate IQ1_S Q8_1:     rel_l2=0.0680403  max_abs=0.696653  argmax=71093
SSM up IQ1_M Q8_1:       rel_l2=0.0674509  max_abs=0.688058  argmax=71093
```

The up projection is numerically closer than the gate-only trial, but still
farther from llama.cpp than the default path. Both controls remain off by
default. The targeted runs measured approximately `29.0 tok/s` prefill.

The layer-0 QKV tensor is `IQ4_XS`, so `LLM_QWEN35_SSM_QKV_Q81` is not the
control that reaches this projection; the active IQ4_XS switch is
`LLM_QWEN35_SSM_Q81`. Enabling that route produced a severe regression:

```
IQ4_XS SSM Q8_1:          rel_l2=1.70777  max_abs=13.97098  argmax=71093
```

CPU exact IQ4_XS/Q8_1 checks are close to both the default HIP F32 route and
llama.cpp, so this is an AMD Q8_1 staging/kernel-contract issue rather than
evidence for changing the GGUF dequantization. The production profile keeps
the IQ4_XS Q8_1 route disabled until its HIP layout is fixed and revalidated.

### IQ1 Q8_1 contract correction (2026-09-19)

The llama.cpp `block_q8_1` contract was checked directly against the HIP
staging kernel. Its second half of `ds` is `FP16(d * sum(q))`, not the raw
floating-point activation sum. The IQ1 correction uses that value as
`d1q * (ds.x * dot + ds.y * delta)`. The HIP quantizer now reduces the integer
Q8 values and stores `FP16(d * sum(q))`; the scalar dense-FFN calls also pass
the correct `ssm_layer=0` role selector.

The corrected HIP IQ1_S Q8_1 output matches an independent CPU implementation
on the same layer-0 activation at `rel_l2=2.4e-7`, `max_abs=1.8e-7`. This is a
kernel-contract fix, but the route remains opt-in: a whole-model gate-only A/B
still measured `rel_l2=0.07417` against llama.cpp versus the default
`0.06708`, because the surrounding activation state is already different.
The default IQ1 F32 path is therefore unchanged until the upstream state drift
is reduced as well.

### GDN state localization (2026-09-19)

An unfused HIP debug run was compared stage-by-stage with the llama.cpp ROCm
trace. The layer-0 QKV projection was already close, and the raw convolution
output matched at `rel_l2=9.8e-5` (`max_abs=0.00496`). The first later-layer
convolution drift is inherited from the previous layer state (`layer 1
rel_l2=0.00356`), rather than being introduced by the convolution kernel.

The fused GDN path is numerically preferable to the unfused diagnostic path:
whole-logit parity was `0.0670758` fused versus `0.0678372` unfused. The
existing `LLM_QWEN35_SSM_KEEP_SCALAR` switch was also tested and is a no-op for
the active batched schedule, so it is not a valid reduction-order control for
this path. The diagnostic dumper now saves scalar `Q4 ssm_conv` stages for
future state-localization runs.

The llama.cpp reduction-order A/Bs were also rejected for the production
profile: `LLM_QWEN35_GDA_REF_SCALAR=1` measured `rel_l2=0.07170`, and
`LLM_SSM_BATCH_PARITY=1` was identical to the current output at this 40-token
schedule. The current fused GDN route remains the better measured choice
(`0.06708` versus the exact-order scalar A/B).

### IQ4_XS scalar Q8_1 cache fix and llama.cpp validation (2026-09-19)

The severe IQ4_XS SSM Q8_1 regression was traced to activation staging rather
than the IQ4_XS dot-product kernel. The scalar decode/prefill path reuses its
producer buffer in place for each token, but `launch_quantize_q81_batch_cached`
treated pointer and shape identity as proof that the contents were unchanged.
After the first token, the IQ4_XS MMVQ kernel could therefore consume the
previous token's Q8_1 tile. Scalar (`M==1`) Q8_1 staging now always
requantizes; cache reuse remains available for genuinely batched (`M>1`)
inputs.

The targeted CPU/HIP check now gives `rel_l2=1.94e-7`, `max_abs=1.53e-5` for
the layer-0 IQ4_XS Q8_1 projection. On the exact 40-token IQ3 coding prompt,
the fixed HIP run measured `29.15 tok/s` prefill, `26.24 tok/s` decode, and
`27.14 tok/s` end-to-end for 80 generated tokens. Against a fresh llama.cpp
ROCm run with F32 KV and `ubatch=512`, whole-vocabulary logits measured
`rel_l2=0.0648139`, `max_abs=0.629790`, with the same argmax token `71093`.
The prior catastrophic Q8_1 result was `rel_l2=1.70777`.

Both implementations generated the same compilable clamp function:

```c
int clamp(int x, int lo, int hi) {
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}
```

The HIP and llama.cpp logs and logits are under
`tmp/qwen38/iq3-iq4-q81-fixed2/`. The HIP build passed `make -C rdna4/llm
-j2`. The corrected adapter remains opt-in: on a fresh same-reference A/B,
direct F32 measured `rel_l2=0.0631598` versus `0.0648139` for Q8_1, although
Q8_1 reduced the maximum error from `0.640846` to `0.629790`. Pure IQ2 remains
on its prior validated default until a separate IQ2 whole-logit A/B is
completed.

### Fresh IQ3 scalar-vs-batched parity check (2026-09-19)

The mixed IQ3 launcher default was rerun after the explicit IQ1 role-default
change, with F32 KV, the 40-token coding prompt, and the RX 9070 XT. The
default correctly selected the scalar quality schedule (`qwen35_batch=0`) and
measured `28.70 tok/s` prefill. Against the matching sequential llama.cpp
ROCm/F32-KV logits, it produced rel-L2 `0.0278052`, max absolute error
`0.3896024`, and the same argmax token `71093`; the run is saved as
`tmp/qwen38/ours-iq3-default-current.err` with logits in
`rdna4/llm/ours-iq3-default-current.bin`.

For the same model and prompt, forcing `QWEN38_GSQ_BATCHED_PREFILL=1` produced
rel-L2 `0.0528497` against the matched batched llama.cpp oracle, with max error
`0.5184231`, at `30.59 tok/s`. Disabling the attention IQ2_XXS WMMA route was
neutral (`0.0528497`, `30.49 tok/s`), so WMMA is not the source of this IQ3
batched drift. The result confirms that the current IQ3 scalar default is a
quality guard, while the batched path still needs a phase-2 reduction/order
fix before promotion. Its A/B logs are `tmp/qwen38/ours-iq3-attnwmma-base.err`
and `tmp/qwen38/ours-iq3-attnwmma-off.err`.

### IQ3 phase-2 isolation controls (2026-09-19)

To localize the remaining batched-only error, the HIP runner now exposes
three diagnostic switches, all disabled by default:
`LLM_QWEN35_BATCH_RMSNORM_SCALAR`, `LLM_QWEN35_BATCH_KV_SCALAR`, and
`LLM_QWEN35_BATCH_ROPE_SCALAR`. They retain the phase-2 scheduler but run
RMSNorm, F32 KV writes, or M-RoPE row-by-row respectively. On the same IQ3
40-token A/B, each was neutral at the measured precision: rel-L2 remained
`0.0528497` against the batched llama.cpp oracle. The scalar-attention
control (`LLM_QWEN35_BATCH_ATTN_SCALAR=1`) improved only to `0.0519447`, so
the dominant mismatch is not any one of these simple batch wrappers.

The switches are intentionally diagnostic rather than launcher defaults. The
quality default remains scalar IQ3 prefill and is validated at rel-L2
`0.0278052` against the sequential llama.cpp ROCm oracle; forced phase-2
batching remains an explicit performance experiment until its projection/
state hand-off is made numerically equivalent.

The layer-cap sweep also exposed a separate guard condition: forcing only a
prefix of the mixed IQ3 layers through phase 2 is not numerically safe. With
`LLM_QWEN35_BATCH_MAX_LAYER=2` the first full-attention transition produced
rel-L2 `0.61269`; with a cap of 3 it produced `0.71530`. The unrestricted
all-batched schedule stayed at `0.0528497`, while the scalar launcher stayed
at `0.0278052`. This is evidence of a batched-to-scalar state hand-off issue,
not a reason to promote partial batching; the cap remains diagnostic-only.

For pure IQ2, the corresponding scalar A/B measured rel-L2 `0.0506280`
against the sequential llama.cpp oracle, while the production batched profile
measured `0.0532139` against the matched batched oracle. Each comparison is
schedule-matched; the launcher therefore retains batched IQ2 and scalar IQ3
as their respective validated quality defaults.

### IQ3 batch drift: layer-level localization (2026-09-19)

The unrestricted IQ3 phase-2 path was traced with matching 40-token captures
from the scalar and batched runners. Layers 0--2 were identical. At layer 3,
the attention norm, Q/K/V projections, Q/K normalization, RoPE, and KV write
were identical; the first difference was the flash-attention reduction
(`attention_replay_pregate` rel-L2 `2.98e-4`). Running the same batch schedule
with scalar F32 attention made layer 3 exact and kept layers 3--18 exact, so
the first flash reducer is a real source of drift, but not the entire problem.

At layer 19, scalar attention still left a pre-attention mismatch: the final
batched-row Q/K/V raw projections differed from the scalar capture by rel-L2
`0.00627` / `0.00431` / `0.01634` respectively. This moves the remaining
investigation upstream to the layer-19 activation/projection hand-off rather
than treating all later error as flash-attention error. The layer-cap guard
therefore remains required, and unrestricted IQ3 batching is still an
explicit performance A/B rather than the quality default.

A fresh same-binary A/B separated the available attention controls. Replacing
the F16-packed flash reducer with scalar F32 attention improved the batch-vs-
scalar whole-logit difference from rel-L2 `0.01885` to `0.01806` on the exact
40-token coding prompt, with essentially unchanged measured prefill (`30.53`
versus `30.32` tok/s). The F32-cache batched flash experiment was slightly
worse (`0.01929`) and was not retained. Batched RMSNorm was bitwise neutral;
it is not the source of the remaining discrepancy. The scalar-attention
control remains available as `LLM_QWEN35_BATCH_ATTN_SCALAR=1` for quality A/B
runs. It now uses a single `(head,row)` grid launch with the exact scalar
reduction order, rather than M serialized launches. The fused control was
bitwise identical to the old scalar-attention control (`rel-L2=0` between
their logits) and measured `30.45 tok/s` on the 40-token prefill. The
production IQ3 launcher still uses the fully scalar schedule until the other
batched state/projection deltas are resolved.

### IQ3 layer-19 Q8_1 attention bug fixed (2026-09-19)

The clean fused-attention trace isolated the remaining layer-19 mismatch to
the batched Q8_1 attention adapter. Layer 19 uses a mixed projection layout
(`IQ2_XXS` Q, `IQ3_S` K, `IQ3_XXS` V); with the adapter enabled, the final
batched-row raw Q/K/V differed from the scalar path. Disabling only
`LLM_QWEN35_ATTN_Q81_BATCH` made all three tensors bitwise identical and
reduced the layer-19 output delta to `1.5e-7` maximum (`6.5e-8` rel-L2).
The next visible difference moved to layer 21 and was only `6.6e-5` maximum,
consistent with one-ULP propagation through the recurrent block.

The IQ3 launcher now defaults the batched attention Q8_1 adapter off; it can
still be enabled explicitly for performance experiments. This does not
disable the separately validated scalar `IQ2_XS` Q8_1 adapter used by the
mixed IQ3 quality profile.

On the same fresh 40-token run, the corrected batched configuration
(`LLM_QWEN35_BATCH_ATTN_SCALAR=1` plus the new IQ3 Q8_1 default) reduced the
whole-logit difference against the same-binary scalar control from rel-L2
`0.01885` to `0.01533`, with max error `0.19580` and unchanged greedy token.
Prefill measured `29.95 tok/s`; the numerical improvement costs about 2% at
this short context.

### Reproducible llama.cpp output check (2026-09-19)

The current IQ3 scalar production capture was compared directly against the
matching llama.cpp sequential F32-KV logits for the same coding prompt. Using
`rdna4/llm/ours-iq3-default-current.bin` and
`tmp/qwen38/llama-iq3-current-seq.bin` (248,320 logits), the independent
comparison reports rel-L2 `0.02780516`, max absolute error `0.38960242`, and
the same argmax token `71093`. Both implementations produced coherent,
compilable C++ for the validation prompt. The batched IQ3 path remains
numerically worse (`0.0528497` against its schedule-matched llama.cpp batch
oracle), so the launcher continues to select scalar IQ3 by default.

### IQ3 layer-19 post-attention/FFN audit (2026-09-19)

The follow-up stage dump used the corrected IQ3 settings
(`LLM_QWEN35_BATCH_ATTN_SCALAR=1`, with the IQ3 Q8_1 attention adapter
disabled) and selected layer 19. Scalar and batched attention outputs after
the output projection were bitwise identical. The FFN input RMSNorm vectors
were also bitwise identical (`rel-L2=0`, `max=0`). The remaining layer-19
layer-output delta was only `rel-L2=6.48e-8`, `max=1.49e-7`.

This rules out the attention reducer, KV representation, residual RMSNorm, and
FFN input normalization as the cause of the residual batch drift. The
one-ULP difference is introduced inside the batched FFN projection/activation
path: batched GEMM stages use BF16-packed activations, while the scalar
reference uses the quantized matvec path. IQ3 production remains scalar until
a matched projection contract is validated.

### Current IQ3 output validation against llama.cpp (2026-09-19)

The existing matched validation artifacts were rechecked after the layer-19
diagnostic run. The HIP runner used the IQ3_XXS GGUF on the RX 9070 XT, F32
KV, greedy decoding, and the exact coding prompt. At 40 generated tokens it
selected the same first token as llama.cpp (`71093`) and produced the coherent
prefix:

```c
int clamp(int x, int lo, int hi) {
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}
```

The 80-token HIP capture completed the function and continued with a sensible
explanation; llama.cpp produced the same function and explanation on the
matched prompt. HIP measured `28.85 tok/s` prefill and `25.17 tok/s` decode for
the 80-token run. The full outputs and logs are retained under
`tmp/qwen38/final-validate-20260919/`, with `llama-clamp.c` serving as the
reference syntax check.

### Mixed-IQ3 launcher parity-default correction (2026-09-19)

A fresh audit of the current source found that the launcher comments and the
actual mixed-IQ3 defaults disagreed: IQ2_XXS, IQ3_XXS, and IQ4_XS Q8_1
adapters were still enabled even though the documented quality profile called
for direct F32. Those defaults were corrected while keeping the separately
validated pure-IQ2 path unchanged.

The first rerun used a hand-built prompt with a different final token than the
llama.cpp file and was therefore not a valid parity measurement; its reported
`0.31752` value is superseded. With the exact
`tmp/qwen38/coding-prompt.txt` token sequence, the current RX 9070 XT run
measures `rel-L2=0.0413424`, `max_abs=0.377888`, and the same argmax token
`71093` against `tmp/qwen38/llama-iq3-current-seq.bin`.

The llama.cpp graph was also used to test a separate-order SSM gate
implementation (`RMSNorm(core_out)`, then `SiLU(z)`, then multiply), using a
correct head-wise 128-element RMSNorm diagnostic kernel. On the same prompt it
measured `rel-L2=0.32444`, `max_abs=2.89706`, versus the fused runner path's
`0.31752`; it remains diagnostic-only and is not enabled by the launcher.

### Q8 cache-layout fix and output recheck (2026-09-19)

The generic batched Q8 quantizer was incorrectly inheriting the previous
`batch_q8_mode`. A preceding Q8_1 or D4/MMQ staging call could therefore make
the next Q8x2 projection reuse the shared scratch buffers with the wrong scale
contract. The helper now unconditionally owns Q8x2 mode and invalidates/rebuilds
the shared tile when the producer shape changes. The launcher also passes
`LLM_LOGITS_PATH` and `LLM_GEN_TEXT` through its explicit diagnostic interface.

After rebuilding, the matched IQ3_XXS run on the RX 9070 XT showed the first
two HIP-vs-CPU shadow errors at rel-L2 `0.00297` and `0.00469`. The full
40-token batched prefill completed at `27.61 tok/s`; the corrected cache path
produced the same greedy first token (`71093`) as the llama.cpp oracle.

Output-level validation remains the decisive quality check: the existing
llama.cpp ROCm capture and HIP IQ3 capture both emit the complete compilable
function below, and the extracted HIP/reference sources pass
`gcc -std=c11 -Wall -Wextra -pedantic -fsyntax-only`:

```c
int clamp(int x, int lo, int hi) {
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}
```

The reference is `tmp/qwen38/final-validate-20260919/llama-final.out`; the
HIP output is `tmp/qwen38/final-validate-20260919/runner-text-80.out`, and the
new corrected-path prefill trace is
`tmp/qwen38/final-validate-20260919/runner-q8cache-fixed.out`.

### Exact-prompt parity and generation recheck (2026-09-19)

The previous large IQ3 logit gap was traced to prompt-token misalignment. The
hand-built HIP string ended with token `198`; llama.cpp's
`coding-prompt.txt` ends with token `271` because its two trailing newlines are
merged by the Qwen tokenizer. Both prompts reported 40 tokens, hiding the
different final position.

Using the exact prompt file on both implementations gives:

```
HIP IQ3_XXS vs llama.cpp sequential F32-KV: rel-L2=0.0413424
max_abs=0.377888  argmax=71093 (match)
```

The HIP run also generated the same clamp body as llama.cpp:

```c
int clamp(int x, int lo, int hi) {
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}
```

The clean exact-prompt run measured `27.66 tok/s` prefill and `26.27 tok/s`
decode for 40 prompt plus 80 generated tokens. The logits artifact is
`tmp/qwen38/current-audit-20260919/hip-exact-coding-prompt.bin`; the clean
generation log is `tmp/qwen38/current-audit-20260919/hip-exact-generation-clean.out`.
The launcher now conditionally forwards diagnostic environment variables, so
unset debug variables no longer accidentally enable layer tracing.

The same exact-prompt check on the pure IQ2_XS GGUF measured `0.105615`
rel-L2 with the current launcher defaults. A direct-F32 A/B with scalar
prefill and all identified IQ1/IQ2 Q8.1 adapters disabled measured `0.091018`
rel-L2, max `1.18030`, with argmax `71093` still matching. This confirms that
the IQ2 residual gap is not the prompt-token mismatch fixed above and remains
the next numerical work item; no IQ2 adapter was promoted based on this A/B.

### IQ1_S Q8_1 block-sum contract fix and F32-KV parity (2026-09-19)

The pure-IQ2_XS layer-0 FFN weights are `blk.0.ffn_gate.weight = IQ1_S` and
`blk.0.ffn_up.weight = IQ1_M` (verified from GGUF metadata). In the production
scalar FFN path the gate runs `matvec_iq1_s_q81_batch` (f32 scales) and the up
projection runs `matvec_iq1_m_q81_batch`; both are fed by
`quantize_q81_iq1_batch_32_exact`.

That IQ1 quantizer stored the FP16 block sum as `d * sum(q)` instead of the
FP16 sum of the original input values. llama.cpp's `quantize_q8_1` and
`quantize_mmq_q8_1` both store `make_half2(d, sum(x))`: the Q8_1 `s` field is
the raw input sum because the IQ1 affine correction multiplies it by `delta`.
A replay of `blk.0.ffn_gate.weight` against llama.cpp's captured layer-0 trace
showed the raw-sum recipe matching to `6.7e-8` rel-L2 while the `d*sum(q)`
recipe was `6.9e-3` off, confirming the mismatch. The quantizer now stores the
raw FP16 input sum (`rdna4/llm/hip_llm_runner.c`, `quantize_q81_iq1_batch_32_exact`).

Apples-to-apples exact-prompt comparison is against the F32-KV sequential
llama.cpp oracle (`tmp/qwen38/llama-iq2-f32-seq-fixed.bin`); the launcher's
default IQ2 `q8q4` KV conflates KV quantization with projection error. Measured
on the RX 9070 XT with the exact prompt:

```
                                        rel-L2        max_abs     argmax
production default (q8q4 KV, fixed)     0.10723       1.124       71093
production default (F32 KV, fixed)      0.0520602     0.674       71093
production default (F32 KV, legacy)     0.0669127     0.696       71093
IQ1 q81 fully disabled (F32 KV)         0.0566244     0.645       71093
FFN gate on MMQ f16 scales (F32 KV)     0.0730758     0.752       71093
```

The IQ1_S sum-field fix improves the F32-KV projection parity from `0.06691`
to `0.05206` at matched argmax. The `q8q4`-KV production number is essentially
unchanged (`0.10561` -> `0.10723`) because KV quantization dominates that
comparison; the F32-KV row is the correct projection-quality measurement. The
finding also shows roughly half of the previously reported IQ2 gap was KV
quantization, not model arithmetic.

Regression and output checks on the same build:

```
IQ3_XXS exact prompt vs llama-iq3-current-seq: rel-L2 0.0413424, max 0.377888, argmax 71093
IQ2_XS generated C (F32 KV, --coding, 80 tokens): complete int clamp(int,int,int), 29.5/26.0 tok/s
IQ2_XS generated C (q8q4 KV, --coding, 80 tokens): complete int clamp(int,int,int), 29.5/25.3 tok/s
gcc -std=c11 -Wall -Wextra -pedantic -fsyntax-only hip-iq2-fixed-clamp.c: rc 0
```

Artifacts: `tmp/qwen38/current-audit-20260919/hip-iq2-f32kv.bin`,
`hip-f32kv-legacy.bin`, `hip-f32kv-iq1off.bin`, `hip-f32kv-gate0.bin`,
`hip-iq3-regression.bin`, `hip-iq2-gen-q8q4.out`, `hip-iq2-gen-f32kv.out`, and
`hip-iq2-fixed-clamp.c`. The `q8q4` KV default is
retained for the 16-GiB long-context profile; F32 KV is a parity diagnostic and
uses ~1.6 GiB more VRAM at this prompt.

### Matched-KV reference and layer-0 divergence localization (2026-09-19)

A matched reference was generated with llama.cpp using the same KV contract as
the runner's `q8q4` default (`LLAMA_KV_Q8Q4=1 LLAMA_SEQUENTIAL=1`):

```
tmp/qwen38/llama-iq2-q8q4-seq.bin          (logits)
tmp/qwen38/llama-iq2-q8q4-trace2/          (per-tensor trace, current helper)
```

With matched settings the exact-prompt gap is `rel-L2 0.13172` (q8q4) versus
`0.05272` (F32 KV). llama.cpp is run-to-run deterministic, so both numbers are
stable. The KV-scheme difference is real: the runner's KV delta (q8q4 vs F32)
is `0.0921` while llama's is `0.1314`.

Layer-0 sequential stage comparison (runner `--qwen38-batched-prefill 0` vs
llama trace) exposes the first divergences, all before any KV use:

```
stage (runner / llama)              base        + SSM gate Q8_1
ssm norm input / attn_norm          1.27e-7     1.27e-7
ssm qkv / qkv_mixed                 4.75e-8     4.75e-8
conv silu / conv_output_silu        ~1.7e-8     ~1.7e-8
gdn z / z                           3.57e-3     5.17e-8
gdn output                          ~1.0e-5     ~1.0e-5
ssm norm / final_output             1.01e-3     4.36e-6
linear out / linear_attn_out        6.57e-4     1.08e-5
layer out / l_out                   1.06e-3     2.87e-4
```

The SSM QKV projection is already exact, but the SSM **gate** projection
(IQ3_S) was on the direct-F32 path. Adding it to the Q8_1 contract
(`LLM_QWEN35_SSM_GATE_Q81=1`) makes `z` match llama at `5.2e-8` and improves
40/64 layer outputs. The per-layer mean is `0.0402` (gate on) versus `0.0379`
(gate off), but with gate on the late-layer errors diverge (L63 `0.1206` vs
`0.1092`) and the final logits regress (`0.1390` vs `0.1317` q8q4; `0.0789` vs
`0.0527` F32 KV). The gate adapter is therefore **not** promoted to the
launcher default; it remains an explicit A/B control.

The remaining layer-0 residual is the GDN output (`~1.0e-5`). The runner's
`deltanet_step_batch_gda_ref_f32` already mirrors llama's
`ggml-cuda/gated_delta_net.cu` order (old-state dot, `decay*dot`, update, XOR
warp reduction), so the residual comes from upstream `alpha`/`beta`
(`~3e-6`) and gated-RMSNorm reduction order, not the recurrence itself. The
`<1e-6` final-logit target for matched settings is therefore gated on
bit-exact ports of the small BF16 `ssm_alpha`/`ssm_beta` projections, the
gated RMSNorm, and later the Q8/Q4 flash-attention kernel; the current
per-layer `~1e-3` errors compound over 64 layers and make the final-logit
metric non-monotonic.

Artifacts: `llama-iq2-q8q4-seq.bin`, `llama-iq2-q8q4-trace2/` (helper
`current-audit-20260919/dump_llama_logits2`), `hip-iq2-gateq81-q8q4.bin`,
`hip-iq2-gateq81-f32kv.bin`, `hip-seqbase-q8q4.bin`, `hip-seqgate-q8q4.bin`,
`hip-seqbase-f32kv.bin`, and `hip-seqgate-f32kv.bin`.

### Bit-exact GDN port: layer 0 reaches 8.4e-9 (2026-09-19)

The layer-0 sequential comparison showed a residual `6.5e-5` in the gated
RMSNorm even though the SSM QKV matched at `4.8e-8`. Two independent root
causes were found and fixed:

1. `hllm_f32_to_f16` (host F32->F16 weight converter) double-shifted in its
   subnormal branch (`mant >>= (-1 - exp)` followed by `mant >> 13`), zeroing
   every weight with `|w| < 2^-14`. Layer-0 `ssm_alpha` had 7 such weights out
   of 5120; after the fix its output matches llama at `1.1e-7` (was `2.2e-5`).
   This converter is used for every BF16-derived and F32-derived F16 weight
   upload, so the fix is model-wide.
2. The BF16/F16 `ssm_alpha`/`ssm_beta` projections now use bit-exact ports of
   llama.cpp's `ggml-cuda/mmvf.cu` `mul_mat_vec_f` with `type_acc=float`
   (`matvec_f16_llama_f32`, `matvec_bf16_llama_f32`): f32 FMA over
   `(half,half)` pairs, XOR warp reduction, then the warp-level block fold with
   the zero-filled lanes 8..31. The launcher block size follows llama's
   `ncols` heuristic.

Sequential layer-0 stages vs the matched llama q8q4 trace after the fixes:

```
stage              rel-L2
ssm norm input     1.3e-7
ssm qkv            4.8e-8
conv silu          ~1.7e-8
alpha              1.1e-7
z (SSM gate)       5.2e-8   (requires SSM gate Q8_1)
gdn output         8.9e-8
gated RMSNorm      7.2e-8
ssm out (IQ4_XS)   7.4e-9
layer 0 output     8.4e-9
```

Layer 1 (SSM with IQ1_M QKV / IQ2_S out) is the next divergence: its pre-conv
QKV matches at `9.5e-8`, but the depthwise conv state accumulates `~2e-6`
across 40 tokens and the IQ2_S `ssm_out` needs the Q8_1 contract. Layer-2 (IQ1_S
QKV) matches llama's MMVQ `raw-sum` contract only when `LLM_IQ1S_MMQ_SCALES=0`;
the production default still uses the f16-scale MMQ variant, which differs by
`5e-4` on that projection.

Promoted for the pure-IQ2_XS launcher profile (each matches a llama.cpp
sequential-decode kernel): `LLM_QWEN35_SSM_GATE_Q81=1`, `LLM_IQ2_XXS_Q81_SCALAR=1`,
and `LLM_IQ2S_Q81_SCALAR=1`. The mixed-IQ3 profile keeps them off.

Final exact-prompt measurements on the RX 9070 XT (production batched path,
new defaults + kernel fixes):

```
IQ2_XS vs llama q8q4 (matched):  rel-L2 0.1297140  max 1.62552  argmax 71093
IQ2_XS vs llama F32 KV:          rel-L2 0.0460101  max 0.540599 argmax 71093
IQ3_XXS vs llama F32 KV:         rel-L2 0.0397622  max 0.428577 argmax 71093
IQ2_XS generation (q8q4, --coding, 80 tok): complete int clamp(int,int,int)
end-to-end 25.6 tok/s (prefill 29.5, decode 24.4)
```

The `<1e-6` final-logit target still requires the remaining per-kernel
reduction-order ports (conv1d, IQ1_M SSM QKV, IQ2_S MMVQ, and the Q8/Q4 flash
attention). The layer-0 result shows the approach is sound: when a stage is
bit-exact, its error drops five orders of magnitude.

Artifacts: `llama-iq2-q8q4-seq.bin`, `llama-iq2-q8q4-trace2/`,
`hip-prod-q8q4.bin`, `hip-prod-f32kv.bin`, `hip-iq3-regression2.bin`,
`dbg-sn-iq2s/`, and the per-stage `hip-sn-*-q8q4.bin` set.

### Q8_1 boundary sensitivity limits partial bit-exactness (2026-09-19)

Per-token tracing (`LLAMA_DUMP_SEQUENCE=1`, helper
`current-audit-20260919/dump_llama_logits2`) shows layer 1's IQ1_M QKV matches
llama at `~9e-8` for many prompt tokens but diverges to `~1.4e-3` at others
(tokens 2, 3, 6, 20, 30). Replaying the llama Q8_1 recipe on the runner's own
input reproduces the runner's QKV at `7e-8` for every token, and reproduces
llama's QKV only on the agreeing tokens. The runner's quantizer is therefore
exact; the disagreeing tokens have a different layer-0 output, so a near-tie
activation rounds to a different int8 value and the IQ1 projection amplifies
that 1-ULP change to `~1e-3`.

This means a partial bit-exact port is not monotone: as long as any stage
retains a `~1e-8` difference, the Q8_1 rounding at a boundary can flip and
re-introduce a `~1e-3` layer error. Reaching `rel-L2 < 1e-6` end-to-end
requires every activation path to be bit-exact, not merely close.

Two norm-order ports were tried and both regressed, so they were reverted
(binary verified byte-identical to `hip-prod-q8q4.bin`):

- `ggml-cuda/norm.cu` `rms_norm_f32<1024>` (xor warp fold) for the 5120-wide
  norms: per-token layer-1 QKV error worsened to `1.7e-3` mean.
- The `build_gdn_l2_norm` (`rms_norm_f32<256>` + `1/sqrt(n)` scale) and
  `build_norm_gated` (`rms_norm_f32<256>` + silu) reductions for the 128-wide
  SSM norms: the layer-1 QKV mean improved (`1.7e-3 -> 2.3e-4`) but the
  per-layer profile got worse (only 5/64 layers closer, mean `0.0399 ->
  0.0428`).

`rocprofv3 --kernel-trace` on the llama helper settles the kernel question: the
wide norms run `rms_norm_f32<1024, true, false>` (workgroup 1024), the 128-wide
norms run `rms_norm_f32<256, ...>`, and the conv runs `ssm_conv_f32<true,128,4>`.
So the `norm.cu` kernels were the right target, yet the port still regressed.

Root cause: **coupled error cancellation**. With the existing 256-thread norm
the layer-0 `attn_norm` differs from llama by `1.27e-7`, but the layer-0 FFN
output is bit-exact (`0`) and the layer-0 output is `8.4e-9`. Porting the exact
`rms_norm_f32<1024>` makes the `attn_norm` bit-exact (`0`) but the layer-0 output
worsens to `1.2e-4` and the FFN output to `1.9e-3`. The existing rounding is part
of a cancellation that keeps the downstream aligned with llama; changing one
stage breaks it.

Consequence for the `<1e-6` target: progress is non-monotone. Making a stage
bit-exact can *increase* the final gap unless every coupled stage is made
bit-exact together, so the cancellation disappears entirely. The working rule is
to verify bit-exactness per stage (not via final logits) and port all coupled
stages before re-measuring end-to-end. The norm/rms ports were reverted; the
source is back at the validated `hip-prod` configuration.

The gated RMSNorm / Q/K L2 norm ports were also reverted for the same reason,
and a "port both the attn_norm and FFN norm together" run gave the same result
as attn-norm-only (`L0 1.2e-4`, `ffn0 1.9e-3`), so the coupled stage is not the
FFN norm.

Instrumented llama's `mul_mat_vec_q` (`printf` in `mmvq.cu`, rebuilt
`libggml-hip.so`, then reverted) to dump the exact schedule. For IQ3_S,
`ncols_dst=1` on RDNA4:

```
qi=16 vdr=2 nwarps=1 rows_per_cuda_block=1 blocks_per_iter=4 blocks_per_row=20
kbx = tid/8 + 4*m   (m=0..4)
kqs = 2*(tid%8)
```

`kqs` maps to the 256-block's q8 sub-block `s=kqs/2=tid%8` (qs bytes 8s..8s+7,
qh[s], signs 4s..4s+3, scales[s/2] nibble s%2). The global q8 block touched by
thread `tid` at iteration `m` is therefore `(tid/8+4m)*8 + (tid%8) = tid + 32m`
— i.e. the same lane-per-block, stride-32 partition the runner already uses.
The only remaining differences are the per-block fp association
(runner `dw*ts*(1+2ls)*sumi`, llama `(dw*ts)*float(ls*sumi)` with an exact
integer `ls*sumi`) and the warp fold (`__shfl_down` vs llama's `__shfl_xor`).
The runner's `iq3s_grid_dev` was verified identical to ggml's `iq3s_grid`.
A numpy reproduction of both paths still matched only 82–96/200 rows
bit-for-bit, so one more rounding detail remains; the schedule itself is now
known exactly. Also dumped: nwarps=1 for IQ1_S/IQ1_M/IQ2_XXS/IQ2_XS/IQ2_S/
IQ3_XXS, nwarps=8 for IQ4_XS/Q2_K/Q4_K.

The residual `~1e-8` in the projections needs per-kernel arithmetic auditing. llama's
`ggml-cuda/mmvq.cu` `mul_mat_vec_q` on RDNA4 with `ncols_dst=1` selects
`calc_nwarps(type, 1, MMVQ_PARAMETERS_RDNA4)`: `nwarps=8` for the simple
vec_dot types (Q4_0/Q8_0/Q2_K/Q4_K/Q5_K/Q6_K/IQ4_NL/IQ4_XS) and `nwarps=1` for
the complex ones (IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S). For
IQ3_S the verified values are `qk=256, qi=16, vdr=2`, `blocks_per_iter=4`,
`kbx=tid/8`: eight threads per 256-value block, equivalent to the runner's
stride-32 Q8-group partition. The previous `qi=8` / four-thread description
was incorrect. Do not rewrite this lane mapping. The isolated replay below
identifies floating-point association and FMA as the remaining IQ3_S issue.

Recommended approach for the `<1e-6` target, given the coupling: treat it as a
coordinated rewrite, not incremental patches. Concretely, first build a
per-stage bit-equality harness (run one layer's stage with llama's captured
input; assert exact equality, not rel-L2) and use it to enumerate exactly which
stages differ at `~1e-8`; then port that whole coupled set in one change and
only then re-measure end-to-end. Layer 0 is already `8.4e-9` end-to-end with a
bit-exact FFN output, so it is the natural place to start. The current source is
locked at the empirically-best `hip-prod` configuration.

### Isolated IQ3_S MMVQ gap resolved; diagnostic only (2026-09-19 resume)

The active IQ2_XS layer-0 tensors were rechecked from GGUF metadata:
`attn_qkv.weight` is IQ3_S, `ffn_gate.weight` is IQ1_S, and
`ffn_up.weight` is IQ1_M. The old `resume.md` gate/up TODO predated the later
fixes above; the fresh production F32-KV comparison reproduces `0.046010129`.

Added `test_iq3s_mmvq_replay.py` and its `.cu` harness. They extract the actual
runner kernels, quantize the captured input once on the GPU, and feed the same
Q8 values/scales to both implementations. The reference includes the local
llama.cpp `vecdotq.cuh` and uses its verified RDNA4 one-warp schedule and
`warp_reduce_sum<32>`. This isolates projection arithmetic; it does **not**
establish quantizer, upstream-normalization, or whole-model bit equality.

The local llama.cpp checkout is `1859b5209` (with local changes), using
`build-codex-hetero-dev2`. Its HIP `mmvq.cu` compile command is `-O3`, without
fast-math. The harness compiles the reference separately with that setting
and the runner with `-O3 -ffast-math`, matching its HIPRTC math mode.

The missing arithmetic contract is:

1. Apply the odd scale code to the integer dot before converting to float.
2. Round the `weight_d * q8_d` product separately.
3. Accumulate the scaled dot with FMA, then use the XOR warp fold.

An ordinary C expression, and a variant using `__fmul_rn`/`__fadd_rn`, still
failed on the tested toolchain. Explicit `v_mul_f32` and `v_fma_f32` preserve
the required boundaries under fast-math. The new
`matvec_iq3_s_q81_mmvq_batch` is selected only by `LLM_IQ3S_MMVQ_REF=1`;
unset/zero retains the original kernel. It applies to existing IQ3_S Q8_1
routes, without enabling additional projection routes.

RX 9070 XT replay results, 10,240 rows x 5 vectors (captured final-token
layer-0 norm, three deterministic random inputs, one zero input):

| Variant | Bit-identical outputs | Relative L2 | Max absolute |
|---|---:|---:|---:|
| Production kernel | 27,753 / 51,200 | 5.76739e-8 | 7.62939e-6 |
| Protected multiply, separate add | 26,320 / 51,200 | 5.72092e-8 | 3.81470e-6 |
| Protected multiply, FMA diagnostic | 51,200 / 51,200 | 0 | 0 |

All 8,192,000 individual dot contributions also match exactly. A separate
13-row test exercises a partial row block: 65/65 outputs match, including
the zero vector. The test exits nonzero on any diagnostic bit mismatch.

Full-model exact-prompt A/B, fresh sequential llama references and matched KV
types, 248,320 logits:

| Model/cache | Production rel-L2 / max | Diagnostic rel-L2 / max |
|---|---:|---:|
| IQ2_XS / F32 | 0.046010129 / 0.540599227 | 0.073783392 / 0.692957401 |
| IQ2_XS / Q8 K, Q4 V | 0.129714052 / 1.625519276 | 0.155747498 / 1.728286982 |

Every run and reference selects token 71093. **Do not promote this diagnostic:**
local exactness worsens the final gap in both cache modes. This is consistent
with the earlier boundary-sensitivity experiments; the remaining coupled
stages still need isolated checks. No production default changed.

Models are `/mnt/nvme02/models/qwen38/27b/gsq/` plus
`Qwen3.8-27B-GSQ-RCO-IQ2_XS.gguf` or
`Qwen3.8-27B-GSQ-RCO-IQ3_XXS.gguf`. All comparisons use
`tmp/qwen38/coding-prompt.txt`, exactly 40 tokens with the canonical trailing
newlines. Fresh IQ3 baseline: rel-L2 `0.039762184`, max `0.428576946`, matching
argmax 71093 (F32 KV). The final rebuilt IQ3 production run is byte-identical
to its pre-change baseline; enabling the diagnostic also leaves these IQ3
logits unchanged. Production IQ2 Q8/Q4 logits remain byte-identical to
`current-audit-20260919/hip-prod-q8q4.bin`.

Measured short generation runs (`--decode 80 --coding -s 256`, warm model):

| Mode | Prefill tok/s | Decode tok/s |
|---|---:|---:|
| IQ2_XS production, Q8/Q4 KV | 28.86 | 24.78 |
| IQ2_XS diagnostic, F32 KV | 28.83 | 25.55 |
| IQ3_XXS production, F32 KV, final rebuild | 27.77 | 25.40 |

These are short correctness runs, not long-context performance claims. Both
production models, the IQ2 diagnostic, and both llama references generated
the complete `int clamp(int x, int lo, int hi)` with two comparisons and a
final `return x`. All passed `gcc -std=c11 -Wall -Wextra -pedantic
-fsyntax-only` and 196 functional cases including INT_MIN/INT_MAX under UBSan.

Reproduce the isolated check (run the final binary with AMD GPU access):

```sh
export TMPDIR="$PWD/tmp"
export PYTHONPATH=/mnt/nvme02/work/llama.cpp/gguf-py
python3 rdna4/llm/test_iq3s_mmvq_replay.py \
  /mnt/nvme02/models/qwen38/27b/gsq/Qwen3.8-27B-GSQ-RCO-IQ2_XS.gguf \
  tmp/qwen38/llama-iq2-q8q4-trace2/llama-attn-norm-00.bin \
  --out tmp/qwen38/iq3s-replay-20260919/full --random-vectors 3
env LD_LIBRARY_PATH=/opt/rocm/core-10.0/lib \
  tmp/qwen38/iq3s-replay-20260919/full/replay \
  tmp/qwen38/iq3s-replay-20260919/full 10240 5120 5
```

Full-run outputs, fresh reference logits, generated C, build logs, and
`summary.json` are in `tmp/qwen38/iq3s-replay-20260919/`. The local
`summarize.py` recomputes metrics and compiles/tests the generated code.
`make -C rdna4/llm -j2` succeeds; existing warnings in unrelated/previously
modified code remain. Launcher syntax and diff whitespace checks pass.

### 4K / 512-token prefill target exceeded (2026-09-19)

Added the explicit runner argument `--qwen35-prefill-bf16`, also available as
`hip_llm_load_options.qwen35_prefill_bf16`. It enables batched scheduling and
BF16 projection GEMMs for attention, FFN, and SSM prefill, including the SSM
output projection. The previous per-layer SSM exclusion otherwise sends all
48 SSM blocks through the per-token fallback, including their dense FFNs.
The flag uses the existing bounded dequantization scratch buffer and leaves
decode dispatch unchanged. It does not require `QWEN38_GSQ_PERF=1`.

This is an **experimental arithmetic choice**, not a bit-exact replacement
for quantized MMVQ/MMQ. Production defaults remain unchanged. The new
`LLM_QWEN35_PROFILE_PREFILL=1` diagnostic synchronizes at layer boundaries
and prints actual scalar/batched layer timings; leave it off for timing runs.

Hardware: RX 9070 XT, gfx1201, 16 GiB. Models under
`/mnt/nvme02/models/qwen38/27b/gsq/`:
`Qwen3.8-27B-GSQ-RCO-IQ2_XS.gguf` and
`Qwen3.8-27B-GSQ-RCO-IQ3_XXS.gguf`.

Fresh llama-bench IQ2 measurements (`-p 4096 -n 0 -b 512 -ub 512 -r 2
-ctk q8_0 -fa on -dev ROCm0 -ngl 99`):

| V cache | Mean prefill tok/s | Samples |
|---|---:|---|
| Q8_0 | 293.87 | 294.029, 293.718 |
| Q4_0 | 293.49 | 293.557, 293.419 |

Both cache formats exceed the requested 150 tok/s reference target. The
runner currently exposes Q8 K / Q4 V; all matched runner comparisons below
use that same format in llama.cpp. These results do not add Q8/Q8 support
to the runner.

The initial 4K padding microbenchmark measured 28.22 tok/s on the default
runner. An existing throughput-profile 512-token tile with SSM batching
enabled measured 83.92 tok/s; 4.788 of its 6.101 seconds were SSM layers.
The new flag measured 255.11 tok/s on the profiled 512-token tile, including
first-use GEMM setup. The full unprofiled runs below are the headline result.

Meaningful coding fixture: `tmp/qwen38/prefill-4k-512/coding-4096.txt`,
exactly **4,096 tokens** in both tokenizers, first token 248045, last 271.
It contains C reference notes followed by the clamp task and the canonical
assistant prefix. Unlike the padding microbenchmark, it ends with the task.
All runs use 512-token chunks, BMAX=512, context capacity 8192, 80 generated
tokens, and warm resident weights. Each second pass reuses GEMM plans.

| Runner mode | First prefill tok/s | Second prefill tok/s | Decode tok/s | Peak MiB |
|---|---:|---:|---:|---:|
| IQ2 default | 28.17 | — | 19.35 | 10062 |
| IQ2 BF16 prefill | 409.85 | 462.89 | 19.50 / 19.49 | 10102 |
| IQ3 BF16 prefill | 421.33 | 476.50 | 19.71 / 19.74 | 13766 |

The fresh matched-prompt llama helper measured about 254 tok/s for IQ2 and
349.80 tok/s for IQ3 (single pass, no warmup; includes first-use setup).
Do not conflate these with llama-bench's warmed synthetic-token samples.

Full-vocabulary logits versus those fresh batched llama references:

| Runner mode | Relative L2 | Max absolute | Argmax, both |
|---|---:|---:|---:|
| IQ2 default | 0.170325388 | 2.245854616 | 71093 |
| IQ2 BF16 prefill | 0.171977136 | 2.299321413 | 71093 |
| IQ3 BF16 prefill | 0.132160331 | 1.575695515 | 71093 |

IQ2 adds a small measured logit gap on this fixture while gaining 14.5–16.4x
prefill throughput. This is not a broad quality equivalence claim. Both fast
profiles repeat their 80-token generation hashes exactly:
IQ2 `3ecf4bbd462d0b82`, IQ3 `a989567da9020b31`. The extracted clamp functions
from both passes pass C11 syntax checking and 196 boundary cases under UBSan.
Default IQ2's generated function passes the same checks.

Reproduce IQ2 (use the IQ3 model via QWEN38_MODEL for its run, and retain the
explicit Q8/Q4 cache selection for this matched comparison):

```sh
env TMPDIR="$PWD/tmp" \
  QWEN38_RUNNER_BIN=/mnt/nvme02/work/gemm/main/rdna4/llm/test_hip_llm \
  QWEN38_GSQ_KV_CACHE=q8q4 LLM_BMAX=512 LLM_BENCH_STREAM_CHUNK=512 \
  LLM_GEN_TEXT=1 \
  LLM_LOGITS_PATH="$PWD/tmp/qwen38/prefill-4k-512/recheck.bin" \
  bash rdna4/llm/run_qwen38_gsq_rocm.sh \
  --qwen35-prefill-bf16 --gpu-only-bench --bench --ubatch 512 \
  --prompt-file tmp/qwen38/prefill-4k-512/coding-4096.txt \
  --decode 80 --coding --bench-repeat 2 -s 8192
```

Artifacts: `tmp/qwen38/prefill-4k-512/`. `summary.json` contains numerical
results and C checks; `summarize.py` regenerates them. The reference helper
was copied locally and adjusted to issue explicit 512-token batches, time
prefill, and permit generation beyond the old 256-token diagnostic limit.
`make_prompt.cpp` constructs and verifies the exact 4K coding fixture.

Validation: `make -C rdna4/llm -j2`, `make -C rdna4/llm profile-test`, launcher
syntax, CLI help, and `git diff --check` pass. Existing compiler warnings
remain. No default policy, commit, or push was changed/performed by this work.

## Q8_0 K / Q8_0 V: 4K/512 prefill (2026-09-19)

The user clarified that the target requires Q8_0 for **both** caches and
selected the existing experimental BF16 projection path. The runner now
accepts `--kv-cache q8q8`; the public enum appends
`HIP_LLM_KV_Q8_0_Q8_0` without changing older values. IQ2's Q8/Q4 and IQ3's
F32 defaults remain unchanged. The new format supports Qwen3.5 hybrid full
attention with head dimensions divisible by 32, up to 256.

K and V have token-major signed-byte codes and separate FP16-rounded scales
stored in float arrays. Codes use the original FP32 block scale. Protected
reciprocal and quotient refinement matches the local non-fast-math HIP
quantizer; naive fast-math reciprocals move values across integer rounding
boundaries. Explicit multiplication also preserves signed zero during
unpacking. The Q8/Q4 quantization path retains its existing arithmetic.

The existing F16 scratch attention path consumes the Q8/Q8 caches. Q4-only
direct/DP4A attention experiments cannot read Q8 V accidentally. This does
not claim native quantized-attention arithmetic parity: the reference build
includes Q8/Q8 vector attention, unlike its Q8/Q4 vector fallback.

Corrected another allocation mismatch: quantized K used to reserve F32
storage despite its byte-based indexing and memory estimate. K/V allocations
now match their formats, and the context estimate includes the shared F16
packing buffers. Existing launcher context ceilings are retained.

The short BF16 test also exposed a hipBLASLt bridge crash when a new shape
needed workspace. Direct `hipMalloc` calls resolved to rocew's exported
function-pointer variable. Allocation/free/error functions are now resolved
from the HIP library explicitly, preventing that symbol collision.

Cache replay uses the actual extracted runner kernels, compiled with
`-O3 -ffast-math`, against llama.cpp's actual HIP quantizer compiled without
fast-math. All **14,741,204 checks pass**: integer codes, scale bits, F16
unpacking in both layouts, and prefix/tail allocation guards. Cases cover
head dimensions 32/64/128/256, three KV heads, position offset seven,
1/31/32/33/511/512/513-token writes, zeros, tiny/large values, random inputs,
and half-integer rounding boundaries.

```sh
make -C rdna4/llm kv-q8q8-test HIPCC=/opt/rocm/core-10.0/bin/hipcc
```

The matched 4096-token coding fixture and generation checks are unchanged.
All runs below use capacity 8192, 512-token chunks and microbatches, BMAX=512,
80 generated tokens, resident weights, and no profiling synchronization.

| Q8/Q8 model | First prefill tok/s | Warm 1 | Warm 2 | Peak used MiB |
|---|---:|---:|---:|---:|
| IQ2_XS BF16 prefill | 409.69 | 465.07 | 464.11 | 9782 |
| IQ3_XXS BF16 prefill | 414.20 | 478.35 | 477.66 | 13446 |

Both warm repeats exceed 150 tok/s for each model. All six generated clamp
functions pass C11 syntax and 196 boundary cases under UBSan. Each model
repeats its 80-token generation hash exactly: IQ2 `053f2a87a25ba3a7`,
IQ3 `b59874ea9ed989ab`.

Fresh matched Q8/Q8 llama.cpp prompt runs measured 250.196 tok/s for IQ2 and
344.217 tok/s for IQ3 without warmup. Full-vocabulary final logits:

| BF16 prefill | Relative L2 | Max absolute | Argmax, both |
|---|---:|---:|---:|
| IQ2_XS | 0.088450366 | 1.062415123 | 71093 |
| IQ3_XXS | 0.068600276 | 0.862522125 | 71093 |

Matched scalar Q8/Q8 baselines measured 28.18 tok/s for IQ2 and 23.34 tok/s
for IQ3. Their relative L2 / maximum errors were respectively
0.103469486 / 1.186900020 and 0.064371208 / 0.935945988. Thus BF16 improves
IQ2's error on this fixture; IQ3's relative L2 rises by about 6.6%, while its
maximum error decreases. This tradeoff remains explicit and opt-in.
Warm fast-mode decode rates are 19.57–19.58 tok/s for IQ2 and 19.81 for IQ3.

The earlier Q8/Q4 matched errors were 0.171977136 and 0.132160331. Keep
these cache-specific comparisons separate; whole-model parity remains open.

Reproduce either model (set `QWEN38_MODEL` to the desired GGUF):

```sh
env TMPDIR="$PWD/tmp" \
  QWEN38_RUNNER_BIN=/mnt/nvme02/work/gemm/main/rdna4/llm/test_hip_llm \
  LLM_BMAX=512 LLM_BENCH_STREAM_CHUNK=512 LLM_GEN_TEXT=1 \
  bash rdna4/llm/run_qwen38_gsq_rocm.sh \
  --kv-cache q8q8 --qwen35-prefill-bf16 --gpu-only-bench --bench \
  --ubatch 512 --prompt-file tmp/qwen38/prefill-4k-512/coding-4096.txt \
  --decode 80 --coding --bench-repeat 3 -s 8192
```

Artifacts: `tmp/qwen38/prefill-q8q8-4k-512/`, including per-run command and
environment JSON manifests, logits, output, and the replay harness.
`summarize.py` regenerates errors and generated-C checks. Scalar baselines,
short-prompt regressions, and fresh llama-bench runs are complete.

Fresh llama-bench, `-p 4096 -n 0 -b 512 -ub 512 -ctk q8_0 -ctv q8_0
-fa on -dev ROCm0 -ngl 99 -r 2 -o json`, using the same local
`build-codex-hetero-dev2` build:

| Model | Mean prefill tok/s | Samples |
|---|---:|---|
| IQ2_XS | 293.060813 | 293.117 / 293.005 |
| IQ3_XXS | 422.700387 | 422.799 / 422.601 |

Short 40-token matched Q8/Q8 checks, preserving the prompt's two trailing
newlines:

| Model / prefill | Relative L2 | Max absolute | Argmax, both |
|---|---:|---:|---:|
| IQ2 scalar | 0.060560054 | 0.786119699 | 71093 |
| IQ2 BF16 | 0.050156043 | 0.686406136 | 71093 |
| IQ3 scalar | 0.041533788 | 0.478228807 | 71093 |
| IQ3 BF16 | 0.061390913 | 0.894451857 | 71093 |

BF16 is not a monotone quality improvement: the IQ3 short-prompt drift is
larger. First-use GEMM setup also makes the 40-token BF16 case slower than
scalar; the performance claim applies to the measured 4K/512 workload.

All 16 benchmark jobs exited successfully. The eight full-vocabulary logit
comparisons are finite with matching argmaxes; all 16 extracted generated
clamp functions pass syntax and 196 UBSan cases. Both production-format
40-token logit files remain byte-identical to their saved baselines (IQ2
`current-audit-20260919/hip-prod-q8q4.bin`, IQ3
`iq3s-replay-20260919/iq3-final.bin`, under `tmp/qwen38/`).
The final documented cache-test target passes all 14,741,204 checks.
Build, profile tests, CLI help, launcher syntax, and `git diff --check` pass.
Existing compiler warnings remain. No commits or pushes.

## Q8/Q8 prefill output integrity (2026-09-19)

The benchmark printer previously wrote raw Qwen/GPT-2 vocabulary pieces.
Correct model output therefore appeared with tokenizer display characters:
`Ġ` instead of spaces and `Ċ` instead of newlines. Artifact summarization
decoded those strings before compiling them, which verified the generated C
but hid the presentation defect. Every benchmark emission path now calls
`bpe_byte_decode()` and writes the resulting byte count. Text display also
ends at EOS/EOT/ChatML turn controls while the benchmark continues executing
the requested decode count. Matching by both token ID and token string covers
converted vocabularies with duplicate control-token strings.

Fresh exact-4096-token coding runs exercised prefill chunk boundaries around
the tuned size:

| Model | Chunks tested | 512-chunk prefill | First token/argmax | Output |
|---|---|---:|---:|---|
| IQ2_XS | 256, 511, 512, 513 | 407.97 tok/s | 71093 | valid clamp C |
| IQ3_XXS | 511, 512, 513 | 420.03 tok/s | 71093 | valid clamp C |

Every run produced finite full-vocabulary logits and strict UTF-8 without
`Ġ`, `Ċ`, replacement characters, or exposed ChatML controls. Each generated
function compiles as C11 and passes 196 boundary cases under UBSan. Changing
the BF16 batch shape is not bit-exact: relative L2 against the listed model's
nearby reference chunk is at most 0.032505 for IQ2 and 0.018762 for IQ3. The
common argmax and functional outputs show that this variation is ordinary
batch-shape arithmetic rather than a dropped or misordered prefill chunk.

The long-context retrieval fixture gives a direct early-chunk test. Its first
line contains the unique passphrase `ZEPHYR-7319`, followed by enough neutral
material to make the complete ChatML prompt exactly 4,096 tokens; the request
for that passphrase is at the end. With eight 512-token chunks, Q8 K/Q8 V,
and BF16 projection prefill, both models emit exactly `ZEPHYR-7319` before
the turn terminator:

| Model | Prefill | Generated response | Result |
|---|---:|---|---|
| IQ2_XS | 410.03 tok/s | `ZEPHYR-7319` | PASS |
| IQ3_XXS | 423.10 tok/s | `ZEPHYR-7319` | PASS |

The repeatable checker is `rdna4/llm/test_prefill_output.py`. Its default mode
requires a complete fenced C program and executes the 196-case UBSan driver;
`--allow-no-c --contains ZEPHYR-7319` validates the retrieval logs. It also
checks benchmark status, strict UTF-8, decoded display text, hidden control
tokens, and optional first-token consistency. Logs, logits, the exact prompt,
and its tokenizer-based generator are in
`tmp/qwen38/prefill-output-fidelity/`.

## Matched C++ coding-output validation (2026-09-19)

The Q8/Q8 BF16 prefill path was tested on an exact 4,096-token C++17 task,
using 512-token chunks and greedy generation. The task requested a single
`merge_intervals` definition, with standard headers supplied by the caller,
and explicitly covered empty input, duplicates, nesting, adjacency,
`INT_MIN`, and `INT_MAX`. Both quantized models were compared with llama.cpp
using the same model, prompt, Q8 K/Q8 V cache types, and generation ceiling.

| Model | Our prefill | llama.cpp prefill | Visible response |
|---|---:|---:|---|
| IQ2_XS | 413.64 tok/s | 256.93 tok/s | byte-identical |
| IQ3_XXS | 421.97 tok/s | 352.63 tok/s | byte-identical |

All four responses contain the same 560-byte implementation, SHA-256
`4a0cb461966fae9a9d9da3b73c1b0c686ce8ee9ac3895c228bc6a653bc99a354`.
The output sorts the pairs, performs one forward merge pass, uses the required
inclusive overlap comparison, preserves merely adjacent intervals, and does
no overflow-prone arithmetic. It is strict UTF-8 and contains no raw BPE
markers, replacement characters, Markdown fences, or ChatML controls.

`test_cpp_merge_output.py` compiles each raw response with C++17, warnings as
errors, ASan, UBSan, and libstdc++ assertions. Each executable passes fixed
edge cases and 10,000 deterministic randomized comparisons. Recheck saved
logs with:

```sh
TMPDIR="$PWD/tmp" python3 rdna4/llm/test_cpp_merge_output.py \
  --require-identical \
  tmp/qwen38/cpp-output-validation/ours-iq2-final.out \
  tmp/qwen38/cpp-output-validation/llama-iq2-final.out \
  tmp/qwen38/cpp-output-validation/ours-iq3-final.out \
  tmp/qwen38/cpp-output-validation/llama-iq3-final.out
```

The final prompt argmax is 1771 on every path. Full-vocabulary runner versus
llama.cpp relative L2 / maximum absolute error is
0.176949976 / 2.302000046 for IQ2 and 0.072349927 / 0.785731316 for IQ3.
The identical completed generation is strong task-level evidence, while the
remaining logit differences still preclude a whole-model parity claim.
