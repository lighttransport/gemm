# FLUX 3 Image Klein on ROCm / RDNA4

Native C/HIPRTC transformer and VAE inference for the distilled FLUX 3 Image
Klein flow-map checkpoint. The DiT is implemented here; VAE decoding reuses
`../flux2/hip_flux2_runner.c`. No hipcc, CUDA, vLLM, or PyTorch GPU runtime is
required. Prompt encoding currently uses Qwen3-VL on **CPU** through Transformers,
then exports features to the native runner. Prepared features can be reused. CPU encoding can take a few minutes on older
processors without BF16 instructions; the generation wrapper caches the result.

The implementation follows [BFL's reference](https://github.com/black-forest-labs/flux-3-image-klein)
revision `bfc8889a4497f846416f253a3972a1941ac698b9`, specifically `dit.py`,
`sampling.py`, and `text_encoder.py`. This is a separate architecture from FLUX2:

- Image, condition, and text streams each have 5 early blocks, followed by 28
  shared blocks. Hidden size 3072, 24 heads, SwiGLU width 9216.
- Four 32-dimensional RoPE axes, including reference-image index on axis four.
- Parallel attention/MLP with a shared shift, scale, and gate per stream.
- Current and target timestep embeddings, 16 flow-map steps, schedule alpha 6.93.
- Qwen3-VL-4B hidden layers 4/8/12/16/20/24/28/32, concatenated to 20480 channels.
  Right padding to a multiple of 80; padded query features are retained.

## Download, detached and resumable

The user-requested model directory is `/mnt/nvme02/model/flux3-klein` (singular
`model`). Override it as the first argument or with `--model-dir` on the runners.
The Hugging Face gate must be accepted and `hf auth login` completed already.

From the repository root:

```sh
mkdir -p tmp/flux3-download
nohup setsid sh rdna4/flux3/download.sh /mnt/nvme02/model/flux3-klein \
  > tmp/flux3-download/download.log 2>&1 < /dev/null &
tail -f tmp/flux3-download/download.log
```

The script pins model revision `337337621b8526e120f2275d250242f4a6bedd75`, uses an
isolated `hf` 0.36 client with stable resumable `.incomplete` files, disables Xet,
locks against duplicate supervisors, and retries failed transfers after 30 seconds.
It downloads FP8 first, then the full repository, including BF16 and NVFP4 assets.
Restart with the same command to resume. `tmp/flux3-download/download.pid` identifies
the supervisor, and `complete.revision` is written only after the entire download
succeeds. No credentials are written to logs. Download and runtime caches use repo
`tmp/`, never `/tmp`.

## Build and generate

```sh
make -C rdna4/flux3
TMPDIR="$PWD/tmp" rdna4/flux3/test_hip_flux3 --selftest
# Optional isolated CPU text encoder environment (requires uv):
sh rdna4/flux3/setup.sh

tmp/flux3-text-venv/bin/python rdna4/flux3/generate.py \
  --prompt 'A red ceramic teapot on a wooden table, soft daylight' \
  --width 256 --height 256 --steps 16 --seed 42 \
  --output tmp/flux3/teapot.ppm
```

The generator caches prompt features in `tmp/flux3/`, runs the native executable,
and saves the PPM plus the final packed latent. It does not launch or stop other
GPU jobs. The native DiT checks free VRAM before loading weights and allocating
workspace, with a configurable 1024 MiB reserve. VAE decoding applies a conservative
workspace estimate and the same reserve. Memory checks cannot reserve
space against another process allocating concurrently. Initial bring-up uses the
FP8 checkpoint (`variants/fp8r/model.safetensors`); BF16 is also supported via
`--dit`. Weights stay in their original GPU dtype. FP8 uses per-output-row F32
weight scales and dynamic per-token E4M3FN activation quantization. BF16 WMMA
accumulates in F32 with BF16 rounding between model operations.

For explicit feature preparation and native inference:

```sh
python rdna4/flux3/prepare_text.py --prompt 'A red ceramic teapot' \
  --output tmp/flux3/teapot-text.f32
# Read the tokens field from teapot-text.f32.json (normally 80 for a short prompt).
rdna4/flux3/test_hip_flux3 --text tmp/flux3/teapot-text.f32 --ntxt 80 \
  --width 256 --height 256 --steps 16 \
  --dump tmp/flux3/teapot-latent.f32 --output tmp/flux3/teapot.ppm
```

The VAE path must use `vae/diffusion_pytorch_model.safetensors`, whose keys match
the reused decoder. Root `ae.safetensors` has different keys. Packed-channel
batch normalization is applied once inside the decoder after unpatching.

`--step T TARGET --latent initial.f32 --dump velocity.f32` evaluates one velocity
without decoding. `--no-vae` runs denoising only. Image dimensions must be multiples
of 16. Raw files are little-endian row-major F32; latent tokens have 128 channels,
text features have 20480. Image token IDs are `(0,row/2,column/2,0)`, text IDs are
`(0,0,0,position)`. Optional prepared condition tokens and IDs are accepted with
`--condition`, `--condition-ids`, `--ncond`; no reference-image encoder is included
in the CLI yet. One sample per call is supported. NVFP4 is not supported by this
runner. Initial noise is rounded to BF16 before FP32 latent integration. The native seeded noise generator differs from PyTorch's generator;
use `--latent` for identical reference inputs.

VAE convolutions default to F32 for reference accuracy. `--vae-wmma 1` explicitly
selects the faster BF16 convolution path. The reused VAE retains its existing diagnostic environment variables. Run with
`FLUX2_SKIP_VAE_BN` unset to apply normalization correctly. This runner supplies
no performance tuning through environment variables.

## Reference validation

With BFL's source checked out independently and CPU dependencies installed:

```sh
python rdna4/flux3/verify_reference.py --reference-dir tmp/flux3-reference \
  --text tmp/flux3/teapot-text.f32 --size 256
python rdna4/flux3/verify_reference.py --reference-dir tmp/flux3-reference \
  --text tmp/flux3/teapot-text.f32 --size 256 \
  --no-condition --output-dir tmp/flux3/verify-t2i
```

The verifier feeds real prepared text features and identical deterministic BF16
image/condition inputs into the official DiT and
the HIP runner, exercising all three streams, current/target timesteps, and
reference IDs -1/-2. It saves both velocity arrays and error metrics. CPU reference
FP8 linear layers emulate quantized dot products in F32 because CUDA `_scaled_mm`
is unavailable there; other layers use BF16. The native kernel selftest separately
checks BF16 and FP8 GEMM against scalar CPU dot products, including tails,
subnormals, exponent 15, and varying row scales. Full model comparisons are
numerical comparisons across CPU and GPU reductions. Small rounding differences
can grow through the FP8 blocks, especially with synthetic text features or only
four image tokens. Validate at a meaningful image size with real prompt features;
`--trace-dir DIR` saves intermediate reference/native activations for diagnosis.
The native CLI also accepts `--trace-dir` (the directory must exist).

The sampler retains FP32 latent state and rounds the timestep-times-velocity
product to BF16 before adding it, matching the reference's tensor operations.

The VAE can also be checked independently (including packed-channel normalization):

```sh
python rdna4/flux3/verify_vae.py --reference-dir tmp/flux3-reference
```

## Validated bring-up (2026-10-17)

RX 9070 XT (`gfx1201`, 16 GiB), while an existing GPU process remained active.
FP8 checkpoint loaded 5.84 GiB. CPU reference used Torch 2.14.1+cpu,
Transformers 5.8.1 and safetensors 0.8.0.

| Check | Result |
|---|---|
| BF16 / row-scaled FP8 GEMM, attention-mean and four-axis RoPE selftests | Passed; exact on the small fixtures |
| Impossible VRAM reserve | Rejected before workspace allocation |
| Official VAE vs native F32 decoder, 64×64 | Relative RMSE 0.00000742; max absolute error 0.0000120 |
| Official DiT vs native FP8, real text, 256 image tokens, no condition | Relative RMSE 0.03405; cosine 0.999421 |
| Same with two condition tokens, reference IDs -1/-2 | Relative RMSE 0.03407; cosine 0.999419 |
| Real prompt, 256×256, 16 steps + F32 VAE | Generated a red ceramic teapot; 91.4 s DiT + 0.8 s decode under concurrent GPU load |
| Download interruption / supervisor restart | Resumed stable partial files at the saved byte offsets |

The tiny synthetic 4-image-token fixture was numerically unstable across CPU/GPU
FP8 reductions (relative RMSE 0.143); using real text and a meaningful image grid
reduced it to the measured values above. Bitwise CPU/CUDA/ROCm equivalence is not
claimed. The verification script reports its metrics and fails above 4% relative
RMSE by default. These are single-velocity checks plus a full native generation;
BF16 full-checkpoint inference and NVFP4 inference were not exercised.
