# HunyuanVideo 1.5 reference validation

These tools validate the experimental native CUDA runner against the pinned
[official implementation](https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5).
The reference must use the same Google SigLIP weights as the native runner.
FLUX.1-Redux is not used. Full capture and parity require staged video weights.

Install the official requirements in a separate environment and stage its exact
checkpoint layout. `capture_ref.py` checks upstream commit
`60783e704160023913bee78f0b47036d393d4dfa`, wraps component methods, and runs the
upstream `generate.py` with the arguments following `--`. Use its documented
480p recipe with offloading, no prompt rewriting, and no sparse attention/cache.
Select original or step distilled weights explicitly. Keep dumps under `tmp/`.

```sh
python capture_ref.py --upstream ../../tmp/hunyuan-video15-upstream \
  --dump-dir ../../tmp/hv15-reference \
  --vision-model ../../tmp/hunyuan-video15-model/google_siglip/reference_vision -- \
  --model_path CKPTS --image_path portrait.png --resolution 480p \
  --video_length 81 --seed 42 --offloading true --rewrite false
```

Run this from `ref/hunyuan_video15/`; adjust interpreter and model paths. Fast12
also needs the official step-distillation option. Use the native artifact
`input.png` as the reference image so crop and alpha compositing match. The official environment may
need a separate requirements lock from other diffusion runners.

The capture writes `noise_input.npy` and little-endian `noise_input.f32`.
Native Philox and the reference's CPU RNG do not produce the same noise merely
because the seed matches. For a matched-noise native run:

```sh
mkdir -p tmp/hv15-native
HV15_DUMP_DIR="$PWD/tmp/hv15-native" \
HV15_NOISE_F32="$PWD/tmp/hv15-reference/noise_input.f32" \
python3 cuda/hunyuan_video15/native_generate.py [generation arguments]
python3 ref/hunyuan_video15/convert_native.py tmp/hv15-native
python3 ref/hunyuan_video15/compare.py tmp/hv15-reference tmp/hv15-native
```

Run these from repository root. The environment variables are diagnostic only;
production generation options are arguments. Native dumps reverse tensor axes
into canonical batch/channel/time/height/width or batch/token/hidden order.
The comparator checks all seven named components by default. `--components`
selects a bounded subset for staged bring-up. It never passes an empty or missing
comparison set.

The first DiT dump is the positive CFG branch before guidance. Qwen captures
remove masked padding. Empty ByT5 embeddings are zero and excluded from native
attention; their capture does not prove a quoted-glyph ByT5 encoder path. T2V
has no encoded portrait; choose an explicit subset when validating T2V. Capture
hook execution, shape conventions, dtype effects, and VAE tiling parity still
need real checkpoint validation.

## Google SigLIP component check

Build the bounded encoder probe and compare it with Transformers, using the
public Google vision tensors exported by the downloader:

```sh
make -C cuda/hunyuan_video15 probe
python3 ref/hunyuan_video15/verify_siglip.py --model tmp/hunyuan-video15-model \
  --image portrait.png --out tmp/hv15-siglip-parity
```

The output directory must be new. This verifies preprocessing, 729 token
features and finite values, then writes `parity.json` with explicit dtypes.
The default FP32 reference matches the native activation precision; both load
the same exported FP16 weights. Native patch extraction remains FP32 and
vision inference explicitly disables the backend's default TF32 cuBLAS rounding.
The scoped settings are restored before subsequent components run.
Use `--reference-dtype float16` to test the
upstream production activation dtype.

The updated IEEE FP32 native path passed the production prepared female portrait
against the CPU FP32 reference: cosine **0.9999999953**, relative L2
**0.0000974552**. Its earlier TF32 output had relative L2 **0.0061160358**.
These compare identical exported FP16 weights and the same prepared pixels.

Previously, on RTX 5060 Ti 16GB, two vhuman portraits at 480×848 passed against FP32
reference activations: cosine 0.99999743 / 0.99999465, relative L2
0.00226786 / 0.00327150. Against FP16 reference activations, the first passed
(cosine 0.99993719, relative L2 0.01121211) and the second failed
(cosine 0.99964932, relative L2 0.02648826). The strict threshold is unchanged.
This validates the encoder graph at native precision, not FP16 equivalence or
the denoiser, VAE, complete video quality or peak video VRAM.

## Bounded VAE graph check

Build `make -C cuda/hunyuan_video15 vae-probe` and run `verify_vae.py` with
`--model`, `--upstream`, `--image`, a new `--out`, and `--config` pointing to the
official `vae/config.json`. The tested config comes from Tencent model revision
`9b49404b3f5df2a8f0b31df27a0c7ab872e7b038`; the script checks the source pin.
It uses one 128×128 frame without tiling, equal FP16 weights and FP32 reference
activations by default. On RTX 5060 Ti it passed encode (cosine 0.9999993111,
relative L2 0.0011887254) and decode (cosine 0.9999997321, relative L2
0.0007354493). This does not establish full-resolution spatial/temporal tiling
parity. The FP16-reference run also passed: encode cosine 0.9999990820 /
relative L2 0.0013563158, decode cosine 0.9999997047 / relative L2
0.0007720890. `--reference-dtype float16` reproduces that dtype check.


The five-frame moving-input test (`--frames 5 --reference-dtype float16`)
exposed a missing frame-causal mask in the native VAE attention block. The overlay
adds it. After that fix, encode passed (cosine 0.9999972943, relative L2
0.0023552031) and decode passed (cosine 0.9999996590, relative L2 0.0008284218).
This still uses bounded untiled inputs; full-resolution tiling needs validation.

## Bounded Qwen conditioning check

Build `make -C cuda/hunyuan_video15 qwen-probe`, then run `verify_qwen.py` with
`--model`, a pinned Qwen config directory as `--config`, and a new `--out`.
The default prompt is `A person smiles naturally.`; the probe bounds prompts to
256 bytes. Stage `config.json` from the downloader's pinned Qwen repository
revision. The FP32 reference runs on CPU to avoid placing a 7B FP32 model in
16GB VRAM; native uses FP32 activations and equal FP16 weights with host offload.
It compares hidden-state index 26 after removing the exact system prefix.
On RTX 5060 Ti this passed: cosine **0.9999949127**, relative L2 **0.0031899215**.
This bounded result does not prove FP16 reference equivalence, long-prompt or
quoted-glyph conditioning parity.

## Full-resolution decode diagnostic

`make -C cuda/hunyuan_video15 vae-decode-probe` builds a decode-only diagnostic.
Its arguments are `VAE_CHECKPOINT LATENT_FINAL_F32 NEW_OUTPUT_DIR [1|5|81]`.
Input layout is 480×848, with `(frames-1)/4+1` latent frames and 32 channels.
The default is 81 frames; use a canonical little-endian F32 dump rather than a
NumPy file. Create the new output directory before invoking the probe. It writes
PPM frames and an FP32 decoded tensor. No denoiser or text encoder is loaded.

A saved fast12 latent was decoded with full temporal context, dense causal VAE
attention and 128-pixel spatial tiles/25% overlap. This took 631 seconds with
2,870 MiB sampled VRAM. Inspected frames no longer showed the grid artifacts
seen with the initial temporal-window/flash-attention decode. Both settings
changed, so this does not isolate which change caused the improvement or
establish reference parity. The continuous/dense profile is now the default.

The native Hunyuan VAE now follows the official fixed-stride spatial tiler:
clipped edge tiles, vertical then horizontal linear overlap blending, and
cropping to non-overlap prefixes. A CPU comparison against the official method
with tile-local patterns, partial edges and corners was exact (maximum error 0).
Build `spatial-tiling-probe` and run `verify_vae_tiling.py --upstream UPSTREAM
--out NEW_DIR` to reproduce that check.

`verify_vae_decode.py` compares every decoded pixel at 480×848 using the same
saved latent, official FP16 VAE and 128-pixel tiles/25% overlap:

```sh
python3 ref/hunyuan_video15/verify_vae_decode.py \
  --model tmp/hunyuan-video15-model --upstream tmp/hunyuan-video15-upstream \
  --config tmp/hunyuan-video15-model/vae/config.json \
  --latent tmp/hv15-native --out tmp/hv15-vae-full-parity --frames 81
```

On RTX 5060 Ti all 81 frames passed: cosine **0.9999997635**, relative L2
**0.0006881319**. The worst frame's relative L2 was **0.0006982480**. Native
decode took 586 seconds; the reference took 70 seconds, with 6,005 MiB peak
Torch allocation and 8,736 MiB peak reservation. The full-size single-frame
check improved from relative L2 0.002253 to 0.000643 after aligning tiling
(the latter also matches official FP16 pixel postprocessing).

`verify_vae.py --portrait` checks full-size portrait encode/reconstruction with
the same tiling. A second portrait passed encode (relative L2 **0.0012887090**)
and reconstruction (**0.0009443797**) against FP16 reference outputs.
`--encode-only --actual DUMP_DIR` compares saved production conditioning.

## Independent pipeline and expression review

Create a new capture directory with `mkdir -p` before running the native job,
then set `HV15_DUMP_DIR` to that directory.
The diagnostic capture includes the actual sampler schedule. Then run:

```sh
python3 ref/hunyuan_video15/verify_pipeline.py \
  --model tmp/hunyuan-video15-model --upstream tmp/hunyuan-video15-upstream \
  --native tmp/hv15-native --generation-manifest tmp/hv15-clip/manifest.json \
  --out tmp/hv15-pipeline-parity
```

This recomputes Qwen and Google SigLIP conditioning, portrait VAE encoding,
every Euler step and VAE decoding independently, sharing only initial noise.
It compares six components including final decoded pixels and fails closed.
Every decoded frame must also meet cosine ≥ 0.9999 and relative L2 ≤ 0.02;
a passing video-wide average cannot override a failed frame. A regression test
demonstrates a single failing motion frame hidden by the global average.
The reference uses FP32 Qwen/SigLIP activations to match the native profile,
FP16 VAE/DiT weights, and FP32 conditioning and latent inputs to DiT. This is
an assembled official component chain; it does not invoke `generate.py` or
establish equivalence to the upstream all-FP16 encoder configuration. It
supports fast12 portrait I2V with 81 or 121 frames and ordinary unquoted prompts.
Pass `--encoder-dtype float16` for an explicitly separate FP16 Qwen/SigLIP
reference run. This option and 121-frame checking are implemented; GPU runs
for those combinations are still pending. VAE decoding receives the requested
frame count and rejects truncated temporal captures.

### Broader coverage and repository backend

The historical measured results in this document apply to the legacy
`cuda/hunyuan_video15` backend. The new `cuda/hunyuan_video15_native` backend
requires its own captures and comparisons. It currently emits 81 frames; its
GPU parity and 16 GB fit are unverified. The vhuman server selects it with
`--video-backend repo` and forces repository GEMM with fallback disabled.

```sh
# Three distinct portrait files; creates 144 required combinations.
python -m ref.hunyuan_video15.validation_matrix create \
  --portraits first.png second.png third.png --out tmp/hv15-matrix/plan.json
# Each runs/CASE_ID/ must contain manifest.json and independent parity.json.
python -m ref.hunyuan_video15.validation_matrix audit \
  --matrix tmp/hv15-matrix/plan.json --runs tmp/hv15-matrix/runs \
  --encoder-dtype float16 --backend hv15n_cuda_experimental \
  --out tmp/hv15-matrix/coverage.json
```

The audit checks backend, portrait hash, expression intent, seed, preset,
dimensions, encoder precision, generation receipt and all six component/all
frame gates. Missing or failed cases return a nonzero status. It never counts
legacy parity as evidence for the repository port. The plan includes quality
profiles, whose independent CFG reference chain still needs extension; current
`verify_pipeline.py` deliberately accepts only fast12. Visual expression review
is a separate outstanding gate.

Before running the full oracle on a repository-backend capture, record its
actual sampler schedule as `sigmas.json` alongside the tensor dumps and retain
`steps`, `cfg` and `flow_shift` in the generation manifest. The server adapter
adds those manifest fields; the standalone repository runner does not yet emit
the schedule receipt. The reference checker requires it and will not infer
missing evidence. That capture integration remains outstanding.

`convert_native.py` accepts both legacy `.shape.json` captures and canonical
`{dtype,layout,shape}` metadata from the repository backend. Global comparisons
process bounded chunks instead of copying entire videos into FP64 arrays.
A saved legacy 81-frame regression passed six components and every frame after
this change: decoded relative L2 0.0017763566, maximum difference from the prior
comparison metrics below 3e-12. This is a saved-capture recheck, not a new GPU run.

`audit_pipeline.py` reuses an already computed independent reference for a
matched-input regression run. Supply `--reference-run`, its
`--reference-generation-manifest`, the new `--generation-manifest`, `--native`
capture directory and a new `--out`. It checks both manifests' model/profile,
prompt and dimensions, the prepared portrait, captured noise and each recorded
reference tensor checksum before comparing components and all frames.
It does not run reference inference. The earlier TF32 blink clip passed global
comparisons but failed seven per-frame checks around its second closure;
its worst frame relative L2 was 0.02302685. That clip is not a strict pipeline
parity pass.

After explicit IEEE FP32 SigLIP math, the full native blink run passed all six
components and all 81 frames against the same independent reference. Final
latent cosine was **0.9999971140**, relative L2 **0.0024037050**; decoded video
cosine **0.9999984386**, relative L2 **0.0017763566**. Worst frame 39 had cosine
**0.9999929102**, relative L2 **0.0037721860**. The audit verified the unchanged
reference tensor checksums and matched inputs before reuse. Native generation
took 1,925.90 seconds with sampled peak VRAM 13,300 MiB and host RSS 33,234 MiB.
This is one female portrait, fast12/81, Google vision, ordinary blink prompt.
It does not establish the full expression matrix or upstream all-FP16 parity.

`review_expression.py --video CLIP --out NEW_DIR --eye-box left,top,right,bottom`
exports every eye crop, an all-frame portrait sheet and nine larger full-frame
samples. `--mouth-box` adds all lip crops.
Use those sheets to review closure duration, teeth, identity and seams; the tool
does not assign an automatic expression/identity quality score.

## First-step transformer check

The downloader's `--reference-configs` flag stages the official JSON configs.
`verify_dit.py` compares the pinned official transformer with a saved fast12
native first step, reusing the native noise, Qwen, SigLIP and encoded portrait.
This isolates transformer math; it does not prove end-to-end encoder or pipeline
parity. Fused repacked QKV weights are split into exact views and loaded strictly.
Reference weights are staged one block at a time on CUDA.

```sh
python3 ref/hunyuan_video15/verify_dit.py \
  --model tmp/hunyuan-video15-model --upstream tmp/hunyuan-video15-upstream \
  --config tmp/hunyuan-video15-model/transformer/480p_i2v_step_distilled/config.json \
  --native tmp/hv15-native --generation-manifest tmp/hv15-clip/manifest.json \
  --out tmp/hv15-dit-parity --reference-dtype float32
```

`--matmul-precision highest` is the default; `high` enables ordinary tensor-core
FP32 arithmetic. `--attention-precision tf32x3` explicitly selects higher-precision
tensor-core arithmetic for the official dense flex attention kernel. Each choice
is recorded separately; no thresholds are relaxed. Run GPU comparisons
sequentially on a 16GB card. Exact full-sized FP32 attention can be very slow.

`make -C cuda/hunyuan_video15 dit-probe` builds a native saved-input diagnostic.
Arguments: `DIT_CHECKPOINT INPUT_DUMP_DIR EXISTING_OUTPUT_DIR
[--f32-acc|--f32-weights|--first-block|--full-denoise]`.
It reads canonical `noise_input.shape.json` plus native F32 noise, Qwen, SigLIP
and encoded portrait dumps. Supported diagnostic shapes are batch 1, channels
32, up to 31 latent frames, 80 per spatial axis, and 33,390 total image tokens.
Qwen is capped at 1,000 active tokens. The probe uses a 4 GiB managed budget.
Its optional FP32 accumulation override is diagnostic; it does not select a
production runner profile. `--f32-weights` also changes native parameter storage
to FP32 and needs substantially more host RAM. `--first-block` retains only the
first transformer block plus the original output projection; match it with the
reference's `--first-block` option. Use reference `--actual OTHER_DUMP_DIR` to
compare another native prediction while retaining the same canonical inputs.
`--kv-precision float16` is an explicitly labelled FP32-reference diagnostic
that rounds attention K/V to FP16; it is not a production reference setting.

For bounded checks, crop noise and encoded portrait to one latent frame and
8×8 spatial positions, retaining the same text/vision inputs. A copied input
manifest must explicitly identify `validation_stage` as
`bounded_dit_from_saved_native_inputs` and its diagnostic width/height/frames.
These dimensions are supported by the probe, not the production video API.

On RTX 5060 Ti, the original full-sized first step failed against FP16 reference
(relative L2 0.0374). The repo-owned CUDA attention kernel retains the value sum
in FP32 instead of the pinned NVIDIA backend's half2 accumulator. With 54 kernel
calls, all 54 blocks at latent shape `[1,32,21,53,30]` passed against the official
FP32/TF32x3 reference: cosine **0.9999982506**, relative L2 **0.0018723230**.
The same output passed against FP16: cosine **0.9999971836**, relative L2
**0.0023859726**. The tiled cuBLAS implementation took 101.18 seconds with the
probe's 4 GiB managed budget; the earlier WMMA implementation took 167.81
seconds under the same budget. These comparisons reuse saved native conditioning and noise;
complete pipeline parity is checked separately in the section above.

## Complete fast12 denoising check

Pass `--full-denoise` to the native DiT probe to run the actual native Euler
sampler for all 12 steps from saved conditioning and noise. It writes each
velocity prediction, updated latent, final latent and the float32 shift-7
schedule. Use a new output directory, then compare it with the official
transformer and `FlowMatchDiscreteScheduler`:

```sh
mkdir -p tmp/hv15-denoise-native
cuda/hunyuan_video15/test_cuda_hunyuan_video15_dit DIT_CHECKPOINT \
  tmp/hv15-native tmp/hv15-denoise-native --full-denoise
python3 ref/hunyuan_video15/verify_dit.py \
  --model tmp/hunyuan-video15-model --upstream tmp/hunyuan-video15-upstream \
  --config tmp/hunyuan-video15-model/transformer/480p_i2v_step_distilled/config.json \
  --native tmp/hv15-native --actual tmp/hv15-denoise-native \
  --generation-manifest tmp/hv15-clip/manifest.json \
  --out tmp/hv15-denoise-parity --reference-dtype float16 --full-denoise
```

The reference checks the saved schedule against the official recipe before
using identical float32 timestep values. Every prediction and updated latent
must pass the existing thresholds. The reference receives the same initial
inputs, then advances its own latent through its own predictions; it does not
reuse native intermediate latents. Conditioning and decoding are outside this
comparison's scope. Run the native and reference GPU jobs sequentially.

The one-latent-frame, 8×8 spatial diagnostic passed all 26 comparisons against
the FP16 reference across the 12 steps. Final latent cosine was
**0.9999988586**, relative L2 **0.0015339099**. This bounded result does not
establish portrait-resolution denoising parity.

The full 480×848, 81-frame diagnostic (latent `[1,32,21,53,30]`) also passed
all 26 comparisons against the official FP16 reference. Final latent cosine
was **0.9999970074**, relative L2 **0.0024469009**. The worst comparison was
the step-12 prediction: cosine **0.9999928382**, relative L2 **0.0037846448**.
Native execution took 1,365 seconds with a 4 GiB managed budget, 648 precise
attention calls and 16.2 GiB peak host RSS. The official reference took 563
seconds, with 3,133 MiB peak Torch allocation and 4,328 MiB peak reservation.
These memory figures cover the diagnostic, not complete video generation.
The report records the matched schedule, every native output's SHA256 and
the native build receipt's checksum. Encoder inputs remain native captures;
full pipeline parity remains separate. Full-resolution VAE parity is measured
in the VAE decode section above.

`make -C cuda/hunyuan_video15 attention-probe` builds a direct GPU kernel check.
Run `cuda/hunyuan_video15/test_cuda_hunyuan_video15_attention`. It compares two
heads, 8,193 queries and 129 keys against a double-accumulated CPU reference,
including partial query/key tiles. Measured relative L2 was 0.0008067978 and
maximum absolute error 0.0000513573. The check requires an actual hooked kernel
call and fails on non-finite output or excessive error.
