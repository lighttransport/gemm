# Independent native-port references

`verify.py` reads only the clean upstream revision
`60783e704160023913bee78f0b47036d393d4dfa`, pinned model receipts, and the native
float32 captures. It uses FP32 CPU references for Qwen, SigLIP, and ByT5, and
block-offloaded FP16 CUDA references for the official DiT/VAE. SigLIP uses the
PIL processor matching its pinned resize/normalization configuration. Qwen uses
hidden state 26, including the assistant suffix after the system-prefix crop.
The ByT5 reference uses Transformers' byte tokenizer and encoder independently.

Generation never imports these PyTorch modules. Each validation phase writes its
own receipt and exits nonzero on failure. A bounded diagnostic is labeled as such
and cannot satisfy the pipeline acceptance check. Reference tensor hashes are
bound to the exact generation manifest, upstream revision, and verified weight
and configuration hashes. Reusing a reference from another run fails validation.

After a native run with `--dump-dir CAPTURES`, run the following phases under the
existing CUDA device lock. Use a reference environment with torch, transformers,
diffusers, safetensors, and the pinned upstream's dependencies. Set `TMPDIR` to
an existing repository scratch directory before invoking it.

```sh
MODEL=tmp/hv15-native/model
ACTUAL=tmp/hv15-native/i2v-captures
OUT=tmp/hv15-native/i2v-reference
RUN=tmp/hv15-native/i2v-run/manifest.json
PY=tmp/qimg21-ref-venv/bin/python
mkdir -p tmp/hv15-native/reference-tmp
export TMPDIR=$PWD/tmp/hv15-native/reference-tmp

$PY ref/hunyuan_video15_native/verify.py --model "$MODEL" --actual "$ACTUAL" --out "$OUT" \
  --generation-manifest "$RUN" --phase encoders \
  --components qwen_hidden qwen_negative_hidden byt5_hidden siglip_hidden vae_encoded \
  --image tmp/hv15-native/i2v-run/input.png --vision-pixels tmp/hv15-native/i2v-run/vision_pixels.f32

flock -w 900 tmp/pixal3d/device-locks/cuda-0.lock $PY ref/hunyuan_video15_native/verify.py \
  --model "$MODEL" --actual "$ACTUAL" --out "$OUT" --generation-manifest "$RUN" --phase denoise
flock -w 900 tmp/pixal3d/device-locks/cuda-0.lock $PY ref/hunyuan_video15_native/verify.py \
  --model "$MODEL" --actual "$ACTUAL" --out "$OUT" --generation-manifest "$RUN" --phase decode
$PY ref/hunyuan_video15_native/verify.py --model "$MODEL" --actual "$ACTUAL" --out "$OUT" \
  --generation-manifest "$RUN" --phase compare
```

Acquire the device lock for `encoders` too when it includes `vae_encoded`.
For T2V omit `siglip_hidden`/`vae_encoded`, `--image`, and `--vision-pixels`.
For fast12 omit `qwen_negative_hidden`. The denoiser consumes the independently
computed reference conditioning and matched native noise, uses the official
Euler scheduler, and compares every saved step. The decoder consumes the
independent reference's final latent. The final `compare` phase requires every
component for the chosen task/preset and all 81 decoded frames.
It also requires all 12 or 50 intermediate latents, the prescribed generation
recipe, and a measured memory-budget pass. Only this final phase has pipeline
acceptance scope; individual phases remain component checks.

`component_probe MODEL qwen|byt5 PROMPT OUT` captures complete encoders;
`component_probe MODEL siglip PIXELS.f32 OUT` captures the vision encoder.
`component_probe MODEL vae_encode|vae_decode THWC.f32 OUT T H W` exercises VAE
graphs with smaller shapes. `bounded-decode` compares a diagnostic decoded
capture to `OUT/reference/latent_final.npy` without claiming full-video parity.
`component_probe MODEL dit INPUT_DIR OUT T H W [PROFILE]` executes all 54
blocks with smaller spatial/temporal dimensions and zero image conditioning.
Its input directory contains `latent_thwc.f32`, Qwen captures, and optional
ByT5/SigLIP captures. `bounded-dit` compares it to the official graph using
independent conditioning previously saved in `OUT/reference/`; omit vision
captures for the zero-vision diagnostic. These tests are useful for bring-up,
but production acceptance still requires both complete quality pipelines.
