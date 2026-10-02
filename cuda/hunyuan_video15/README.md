# HunyuanVideo 1.5 CUDA runner

Native HunyuanVideo-1.5 generation for vhuman portrait expression previews. The
initial implementation uses a pinned stable-diffusion.cpp CUDA backend behind a
repo-owned C API. Neural inference stays native; Python prepares pixels and
packages artifacts with FFmpeg. It does not use a Python diffusion pipeline.

**Status:** CUDA compilation and request/job/packaging tests pass. Google SigLIP
graph parity passes against an FP32-activation reference on RTX 5060 Ti.
The explicit IEEE FP32 path gives relative L2 **0.00009746** on the second
portrait (the earlier TF32 path gave 0.006116 for this prepared image). Native uses FP32
activations with FP16 weights. Strict parity against FP16 reference activations
passes one portrait and fails another; it is not established.
The bounded untiled VAE encode/decode passes for one and five frames against
FP16 reference outputs after correcting its missing frame-causal attention mask.
Full 480×848/81-frame VAE decode also passes against the official FP16 reference
with identical latents (relative L2 0.00068813). Native spatial tiling now matches
the official fixed strides, clipped edges and linear overlap blends. Full-size
portrait encode/reconstruction passes on a second subject.
Qwen selected-layer conditioning passes a bounded FP32 CPU reference check.
A full fast12 portrait I2V run passed all six component comparisons and all
81 per-frame checks against the assembled official component chain with
independently recomputed conditioning and matched noise. This uses FP32
Qwen/SigLIP reference activations and FP16 DiT/VAE weights. Equivalence to
upstream `generate.py` and its all-FP16 encoders remains unverified, as do
other profiles and the expression/identity quality matrix.
Per the requested model choice, the vision encoder is public Google SigLIP SO400M/14 (Apache-2.0).
FLUX.1-Redux is not downloaded or used. The manifest identifies the Google
vision profile explicitly; identical architecture does not establish identical
weights to the upstream recipe. Generation requires experimental enablement.

## Build and checkpoints

From repository root:

```sh
python3 cuda/hunyuan_video15/setup_native.py --nvcc /usr/local/cuda-13.2/bin/nvcc
make -C cuda/hunyuan_video15 test
```

The setup script uses `tmp/`, pins native revision
`3f8527a46c54ecf4cb4ed6003da8e8982283c73c` and its ggml submodule, and applies the
owned overlay. It recreates overlay target files in that ignored dependency;
do not put independent edits in that build tree. CUDA architecture 120 is the
default for RTX 5060 Ti. The dependency is MIT licensed; retain its license when
distributing a linked build.

Install `huggingface_hub`, `numpy` and `safetensors` in a selected interpreter,
then download one public profile:

```sh
python3 cuda/hunyuan_video15/download_models.py \
  --out tmp/hunyuan-video15-model --checkpoint fast12_i2v --reference-configs
```

The downloader pins public sources, requires
40 GiB free disk for a fresh profile, and writes `model.json` with pinned
component revisions and SHA256 receipts. Other selectable checkpoints are
`quality_i2v` and `quality_t2v`. Download only the profiles needed; each additional
denoiser is about 16.7 GB. The downloader does not fetch the entire upstream model
repository. It uses full precision repackaged weights, not scaled FP8 weights.
`--reference-configs` also stages the pinned official VAE/transformer and Qwen
JSON configurations for component validation; their paths and receipts are
recorded in `model.json`.

## Generate a clip

```sh
python3 cuda/hunyuan_video15/native_generate.py \
  --model tmp/hunyuan-video15-model --task i2v --preset fast12 \
  --image portrait.png --prompt 'The person smiles naturally, then relaxes.' \
  --frames 81 --seed 42 --out tmp/hunyuan-video15-smile \
  --allow-experimental
```

Outputs: `clip.mp4`, `poster.png`, `manifest.json`, `metrics.json`, and a runner
log. `--keep-frames` retains RGB PPM frames. The manifest is published only after
successful encoding. Failed and cancelled runs remove their unpublished output.
Existing output directories are refused. The C executable emits frames and can
validate requests without loading weights; see its `--help` and
[hunyuan_video15.h](hunyuan_video15.h) for the C API.

| Profile | Task | Steps | CFG | Flow shift |
| --- | --- | ---: | ---: | ---: |
| quality | I2V or T2V, matching original checkpoint | 50 | 6 | 5 |
| fast12 | I2V, step distilled checkpoint | 12 | 1 | 7 |

Both support 81 or 121 frames at 24 fps. Allowed 480p buckets are portrait
480×848, landscape 848×480, and square 640×640. The vhuman preview defaults to
portrait. A fixed crop can change composition; bucket selection is explicit.

The runtime uses CPU parameter storage, mmap, block segmentation through explicit transformer graph boundaries,
flash attention for video, sequential CFG, no diffusion cache, and spatial
VAE tiling with 128-pixel tiles and 25% overlap. VAE decoding retains the full
temporal context and uses dense causal attention. The 14,336 MiB ceiling
reserves 3,072 MiB for allocations outside the backend's managed budget,
including the tiled attention workspace. This is
a requested budget, not a proven hardware memory bound. Metrics sample the owned
native PID every 200 ms; short peaks may be missed. Host RAM and PCIe transfers
can strongly affect latency. The FP16-weight smoke run exceeded 32 GiB host
RSS; use a host with more than 32 GiB RAM (64 GiB provides practical headroom).

The overlay adds the missing SigLIP SO400M/14 graph and I2V conditioning, corrects
the video system template and crop offset, restores causal video VAE attention,
and fixes the one-based next timestep
index for meanflow distilled checkpoints. A repo-owned CUDA attention kernel
uses tiled cuBLAS GEMMs with FP32 accumulators and disallows reduced-precision
reductions for unmasked 128-channel sequences with at
least 8,192 queries. It attaches through a backend hook; the dependency's ggml
sources remain unchanged. The 128-key tiles require workspace linear in query
count (about 805 MiB for the 81-frame portrait bucket). Short or masked
attention uses the original backend.
Preprocessing uses Pillow Lanczos and
bicubic, and writes normalized SigLIP pixels for native encoding. Transparent portrait
pixels are composited over neutral gray. SigLIP uses FP32 patch extraction and
explicitly scoped IEEE FP32 cuBLAS math with FP16 weights. Its previous FP32
accumulation flag still allowed TF32 input rounding. The scope initializes all
CUDA stream handles and restores their previous math modes after vision
inference. The owned CUDA helper compiles with the pinned backend's
definitions/includes for its private ABI; ggml sources are unchanged. T2V excludes
masked vision tokens, as the official transformer's mask does. Ordinary prompts
exclude masked empty ByT5 tokens. Quoted glyph prompts still use the bootstrap
backend's formatting and have not been brought to official parity.

## Measured smoke run

The current default blink prompt completed with the corrected spatial tiling
and IEEE FP32 vision path: **1,925.90 seconds (32.1 minutes)**, sampled peak
VRAM **13,300 MiB**, host RSS **33,234 MiB**, and 648 precise-attention calls.
At 480×848/81 frames, all six components and every frame passed against the
independent official component reference. Reference inputs were reused only
after checksums confirmed identical model, prompt, prepared portrait and noise.
Final latent relative L2 was **0.00240370**; decoded pixels **0.00177636**;
worst frame **0.00377219**, cosine **0.99999291**. The earlier TF32 clip failed
seven frame checks, with worst relative L2 **0.02302685**.

Inspection of all 81 eye/mouth crops shows two complete eyelid closures despite
the prompt asking for one. Approximate fully closed holds are 250 and 333 ms;
the earlier longer default prompt's second hold was about 542 ms. These are
manual frame-based estimates, not total blink durations. Lips remain closed;
identity appears stable and no tile grid is visible in the all-frame thumbnail
sheet and nine larger samples. Blink count and natural timing remain unreliable.

The following earlier run used the longer default blink prompt:

The combined runner completed a second portrait's blink job through the vhuman
CLI: fast12 I2V, seed 42, 81 frames at 480×848/24 fps. Wall time was **1,781
seconds (29.7 minutes)**, sampled peak VRAM **12,624 MiB**, and host RSS
**33,065 MiB (32.3 GiB)**. The run used the default continuous temporal VAE
profile and made 648 precise-attention calls (54 blocks × 12 steps). Denoising
took 1,177 seconds and VAE decoding 583 seconds. FFmpeg decoded all 81 frames
without errors; the MP4 duration is 3.375 seconds.

Inspection of all 81 eye crops shows two closure motions: the first is partial,
and the second holds the eyes closed longer than a typical blink. Identity
appears stable in nine full-frame samples, with no visible grid seams there.
Natural expression timing and fidelity remain experimental.

An earlier male smile run used temporal VAE tiles and showed grid artifacts.
Continuous temporal decoding removed those artifacts in inspected frames; the
combined configuration above has now been tested end to end.

These measurements cover two portraits and fast12/81 only. Quality50, T2V,
121 frames, other buckets and the expression matrix remain untested. The
200 ms sampler may miss brief GPU peaks. A 64 GiB host is recommended.

## vhuman integration

```sh
python3 -m server.vhuman.app --video-model tmp/hunyuan-video15-model --video-experimental
python3 -m server.vhuman.cli --work tmp/vhuman-independent --backend cuda video \
  --head HEAD_ID --expression smile --preset fast12 \
  --model tmp/hunyuan-video15-model --allow-experimental
```

`video_generate` jobs accept `head_id`, `expression`, optional `prompt`, `preset`,
`frames`, and `seed`. Expressions: smile, laugh, surprise, sad, angry, blink.
They use the existing shared device lock and require 14,336 MiB free VRAM.
Artifacts live in `heads/HEAD_ID/videos/RUN_ID/`. The head page queues, cancels,
and plays clips; HTTP supports MP4 byte ranges and HEAD requests. Mock mode
creates a clearly labelled static portrait fixture. Video outputs are not
submitted automatically to rig fitting or training.

## Validation still required

`make -C cuda/hunyuan_video15 test` checks request validation and separates
actual denoising progress from loader/tiling callbacks. The latter can have the
same total count as a 12-step sample; the dedicated callback avoids premature
90% updates. The GPU attention check is separate: build `attention-probe`, then
run `cuda/hunyuan_video15/test_cuda_hunyuan_video15_attention`.

Use the tools in [ref/hunyuan_video15](../../ref/hunyuan_video15/README.md) to
capture official and native tensors with identical noise. Require finite values,
cosine ≥ 0.9999 and relative L2 ≤ 0.02. The comparison fails for missing, empty,
wrong-shaped or non-finite components. The Google SigLIP encoder has passed;
bounded VAE and Qwen checks also pass. The original full-sized first DiT step
failed: relative L2 was 0.0374 against the official FP16 reference, exceeding
0.02. After replacing the NVIDIA half-precision value accumulator, the full-sized
54-block first step passed against the official FP32/TF32x3 reference (cosine
0.99999825, relative L2 0.00187232) and FP16 reference (0.99999718, 0.00238597).
Both references reuse identical saved native conditioning and noise to isolate
the transformer. The complete 12-step denoising chain at 480×848/81 frames
also passed all 26 prediction/latent comparisons against the official FP16
transformer and Euler scheduler: final cosine 0.99999701, relative L2
0.00244690; worst relative L2 0.00378464. This uses identical saved native
conditioning and noise. Pipeline parity with independently recomputed
conditioning now passes for the single fast12/81 portrait clip described above,
including all per-frame gates. Full-resolution VAE decode with identical
latents also passes separately. See the reference
README for reproducible checks.

Compare complete pipelines and decoded clips. Measure both frame counts and presets on the
16GB GPU. Review three portraits × six expressions × two seeds for identity,
eyes, teeth, motion, and temporal/tile seams. Native FP8 kernels and replacing
ggml with the repository's own CUDA operators are subsequent stages.

## Model selection and terms

[HunyuanVideo-1.5](https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5) is the
chosen quality-first model. Its official minimum is 14GB with offloading; that
is not evidence that this native implementation fits. The model uses the Tencent
Hunyuan Community License, including territorial and model-improvement
restrictions. Read the actual license before using outputs in any training.
Google's [SigLIP checkpoint](https://huggingface.co/google/siglip-so400m-patch14-384)
uses Apache-2.0. The comparison reference loads this same explicit component.


[Wan2.2 TI2V-5B](https://github.com/Wan-Video/Wan2.2) remains a potential smaller
alternative, but its official 720p recipe requires 24GB. Wan A14B's official
single-GPU recipe requires 80GB. [Mochi](https://github.com/genmoai/models) is
Apache-2.0 and research friendly, but its official recipe requires 60GB and is
T2V only. [LTX](https://github.com/Lightricks/LTX-Video) is a separate alternative;
current LTX-2.x audio/video checkpoints should not be confused with the smaller
legacy LTX-Video 2B distilled model. None is implemented in this module.
