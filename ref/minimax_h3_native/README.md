# Independent MiniMax H3 checks

These programs implement the pinned ComfyUI H3 formulas with PyTorch, without
calling native HIP kernels to produce expected results. `verify.py components`
checks checkpoint projections against CPU INT32 sums and a dense Hadamard
rotation. `verify.py qwen` runs all 50 Qwen layers on CPU or an explicitly
selected PyTorch GPU. `reference.py` runs the two refiner blocks, all 50 DiT
blocks per Euler update, joint stereo audio denoising and the 36-block VAE.
Only initial video/audio noise is shared with native captures.

Install NumPy, PyTorch, safetensors and tokenizers in an existing environment.
The GPU reference requires ROCm PyTorch supporting gfx1200/gfx1201 and integer
matmul. The native runner itself does not depend on PyTorch.

Generate native captures first using the commands in
[the H3 runner documentation](../../rdna4/minimax_h3/README.md). For a complete
40-point run, use the same prompt in the following commands:

```sh
python3 ref/minimax_h3_native/verify.py qwen \
  --model /mnt/disk01/models/h3/weights \
  --prompt 'A red ball rolling on a wooden table, cinematic lighting.' \
  --native tmp/video-rocm/h3-dump --out tmp/video-rocm/h3-qwen-reference
# With ROCm PyTorch, --device cuda selects the independent GPU text reference.

flock tmp/pixal3d/device-locks/rocm-0.lock \
  python3 ref/minimax_h3_native/reference.py \
  --manifest tmp/video-rocm/h3-video/manifest.json \
  --native tmp/video-rocm/h3-dump \
  --qwen-reference tmp/video-rocm/h3-qwen-reference \
  --out tmp/video-rocm/h3-reference

python3 ref/minimax_h3_native/verify.py pipeline \
  --manifest tmp/video-rocm/h3-video/manifest.json \
  --native tmp/video-rocm/h3-dump --reference tmp/video-rocm/h3-reference \
  --out tmp/video-rocm/h3-parity.json
```

Use `verify.py diagnostic` instead of `pipeline` for reduced geometries or
schedules. A passing diagnostic cannot certify the default video profile.
`pipeline` requires 1344×768, 124 frames, 24 fps, seed 42, 40 grid points, video
shift 12 and audio shift 3. It compares every latent update and decoded frame,
requiring cosine ≥0.9999, relative L2 ≤0.02 and measured native process VRAM
within its budget. Failure exits nonzero and retains the comparison report.

Qwen provenance binds the prompt, checkpoint/tokenizer hashes, output hash and
reference source. GPU reference receipts bind each array, canonical shape,
upstream revision and reference source to the native generation manifest.
Checkpoint hashes are verified before reference execution. Source changes
require regenerating references. The upstream revisions are ComfyUI
`2472a20bd291451acc303917059ab14dfc380478` and comfy-kitchen
`be003b7c23c5b01328657955b8bc5d3f073d868e`.

To isolate block error with shared component inputs:

```sh
tmp/video-rocm/h3-build/component_probe /mnt/disk01/models/h3/weights \
  dit-block tmp/video-rocm/h3-dump tmp/video-rocm/h3-block
python3 ref/minimax_h3_native/components.py dit-block \
  --input tmp/video-rocm/h3-dump --native tmp/video-rocm/h3-block \
  --out tmp/video-rocm/h3-block-parity.json
```

The native `vae INPUT_DIR OUTPUT_DIR FRAME_COUNT` probe reads
`latent_video.json` (`dtype: float32`, THWC `shape`) and `latent_video.f32`.
Compare its frames with `components.py vae --input INPUT_DIR --native OUTPUT_DIR
--frames FRAME_COUNT --out REPORT.json`. The latent must have 24 channels and
`5*((FRAME_COUNT-5)/17)+2` temporal tokens. This mode explicitly shares the
component input and never counts as independent end-to-end acceptance.

To trace a complete DiT evaluation at the first timestep with identical
component inputs, use `dit-step` instead of `dit-block` in both commands above.
It compares the attention and FFN residual outputs of all 50 blocks. This
probe requires at most 256 combined text/audio/video tokens, bounds diagnostic
disk use, and retains the `component_shared_inputs` scope.

For disk-limited validation, add `--compress-dumps` to native generation and
`--compress-output` to `reference.py`. Both use lossless gzip compression and
retain F32 values. The verifier reads either native format; reference receipts
record the actual compressed filename and hash, and bind the capture reader
source. These switches do not change the model computation or acceptance
thresholds. Source changes still require fresh independent receipts.

For the default profile when storage is limited, `verify_streaming.py` runs the
independent reference and compares each receipt-bound capture as it is written:

```sh
flock tmp/pixal3d/device-locks/rocm-0.lock \
  python3 ref/minimax_h3_native/verify_streaming.py \
  --manifest tmp/video-rocm/h3-video/manifest.json \
  --native tmp/video-rocm/h3-dump \
  --qwen-reference tmp/video-rocm/h3-qwen-reference \
  --reference tmp/video-rocm/h3-streamed-reference \
  --out tmp/video-rocm/h3-streamed-parity.json
```

This mode requires a fresh reference directory and output report. It publishes
an atomic witness containing each independent capture hash, provenance receipt,
native capture hash and comparison before pruning that reference intermediate.
It retains noise, text, final video/audio latents, first/last frames and the first
failed capture. Native captures are retained. On completion it checks all 206
arrays, source hashes, receipts and retained bytes again. Its report scope is
`full_pipeline_streamed_reference`; acceptance thresholds match `pipeline`.
The ordinary `pipeline` command requires the complete reference capture set.
