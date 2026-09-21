# Qwen-Image 2.1 CUDA runner

This directory adds Qwen-Image 2.1 support for the local Hugging Face
snapshot. The runner uses the official Diffusers CUDA implementation with
sequential CPU offload, which fits the 16 GB RTX 5060 Ti. A native C/NVRTC
transformer path is included for kernel bring-up and direct PyTorch comparison.

The model is never copied. Pass the existing snapshot directly:

```sh
make -C cuda/qimg21 setup
make -C cuda/qimg21
cuda/qimg21/test_cuda_qimg21 --generate \
  --model /mnt/nvme01/models/qimg-21 \
  --prompt "a red apple on a white table" \
  --height 1024 --width 1024 --steps 40 --seed 42 \
  --out tmp/qwen_image21.png
```

For a PyTorch fixture and runner trace:

```sh
cuda/qimg21/reference.py --model /mnt/nvme01/models/qimg-21 \
  --height 256 --width 256 --steps 2 \
  --dump-dir tmp/qimg21-reference
cuda/qimg21/test_cuda_qimg21 --generate --model /mnt/nvme01/models/qimg-21 \
  --height 256 --width 256 --steps 2 --dump-dir tmp/qimg21-runner \
  --out tmp/qimg21-runner.png
tmp/qimg21-ref-venv/bin/python cuda/qimg21/compare.py \
  --reference-dir tmp/qimg21-reference --runner-dir tmp/qimg21-runner
```

`compare.py` treats parity as an acceptance test: every matched denoising
checkpoint must reach cosine `>= 0.99996` for the non-quantized BF16/FP16
weights. It exits non-zero on a missing checkpoint, shape mismatch, non-finite
value, or threshold failure. For a quantized experiment, pass `--quantized`;
that selects the provisional `0.995` gate until a quantizer-specific
calibration set establishes a tighter measured threshold. Use
`--cosine-threshold X` to record an explicit threshold in benchmark logs.
The regression driver also records `initial_latents.npy`, preserving the exact
PyTorch-packed noise input for native denoiser comparisons.

Current native BF16 arithmetic explicitly rounds the text projection before
GELU and Q/K normalization before multiplication by the learned RMS weights.
On the saved 256x256/seed42 two-step reference, matched-input denoiser cosine
is 0.999981133 at timestep 1 and 0.999955009 at timestep 0.02001953125.
The latter **fails** the 0.99996 acceptance gate; full native parity remains
unfinished. These are direct denoiser comparisons, not free-running trajectory
or end-to-end quality guarantees.

Timestep embedding, shared modulation, and final modulation use the same
two-row `[real timestep, zero timestep]` GEMM shapes as PyTorch. Separate
single-row GEMVs select different reduction paths: on the low-timestep
fixture, batching the rows makes both timestep linear outputs and the
shared modulation tensor bit-exact with the saved PyTorch stages. This also
initializes the final scale for text rows, which previously reused unrelated
scratch contents. Final normalization on identical hidden states and scales
matches PyTorch at cosine 0.99999999919; remaining denoiser error accumulates
upstream across the transformer blocks.

The F32-to-BF16 weight loader also has a guarded-memory regression:

```sh
make -C cuda/qimg21 test_weight_upload
./cuda/qimg21/test_weight_upload
```

It checks both input dtypes and BF16 ties-to-even conversion. F32 element
counts must use four bytes per element; using the BF16 divisor reads past
the source tensor, which the guard page detects.

To isolate attention error from upstream projection/normalization drift,
set `QIMG21_STAGE_DIR` and `QIMG21_STAGE_BLOCK` when running a native denoiser
fixture. The dump includes `rope_q.npy`, `rope_k.npy`, `v.npy`, and
`attn_raw.npy`. Replay those exact inputs with:

```sh
tmp/qimg21-ref-venv/bin/python cuda/qimg21/attention_probe.py \
  --stage-dir tmp/qimg21-attn-native17
```

`--backend math` or `--backend flash` selects a diagnostic PyTorch SDPA
backend explicitly. The default preserves PyTorch's normal selection.
On block 17 of the low-timestep fixture, the native attention cosine was
0.99999928391 against default SDPA and 0.99999999884 against math SDPA
(PyTorch 2.14.0+cu130). These isolated operator checks do not replace the
full denoiser acceptance gate.
The same matched-input low-timestep denoiser compared with a PyTorch math
SDPA reference scored 0.99994985006, still below 0.99996. Changing the
attention backend alone therefore does not resolve the parity failure.

The native Euler update rounds its BF16 prediction-times-step product and
casts the updated sample back to BF16, matching the CUDA PyTorch scheduler.
The model timestep separately follows BF16 rounding of `sigma*1000` and
division by 1000. True CFG preserves the BF16 subtraction, multiplication,
and addition boundaries before passing the prediction to Euler.

Run the independent scheduler arithmetic regression with:

```sh
make -C cuda/qimg21 test_scheduler
tmp/qimg21-ref-venv/bin/python cuda/qimg21/scheduler_regression.py
```

Three seeded tests cover sigma transitions 1→0.02, 0.63→0.37, and 0.02→0.
All match CUDA PyTorch bit-for-bit with finite outputs. This isolates the
scheduler update from denoiser errors and does not validate a complete
trajectory. CUDA is the acceptance reference: CPU PyTorch converts the
scalar step to BF16 before multiplying, producing different rounding.

The regression also checks dynamic-shift schedules against the pipeline's
`linspace(1, 1/steps, steps)` convention before terminal stretching. Four
cases (2/4/40 steps at 256/1024/2048/4096 image tokens) match within 1.2e-7
maximum absolute sigma error. A zero endpoint before shifting is incorrect
for interior timesteps and is no longer used.

After these corrections, the native 256x256/seed42 two-step trajectory scored
0.9999691665 and 0.9999702302 against the saved PyTorch scheduler checkpoints,
passing the 0.99996 gate with finite outputs. The isolated matched-input
low-timestep denoiser test still fails; trajectory agreement does not waive
that requirement or establish multi-seed/full-resolution acceptance.

Run the deterministic smoke matrix with:

```sh
tmp/qimg21-ref-venv/bin/python cuda/qimg21/regression.py \
  --model /mnt/nvme01/models/qimg-21
```

The default matrix covers 256x256, 256x512, and 512x512. Add
`--include-full` for the 1024x1024/40-step acceptance run, or pass repeatable
`--case HEIGHTxWIDTH:STEPS:SEED` values to define a custom matrix. Each case
gets isolated fixtures under `tmp/qimg21-regression/`.

Once the native binary is built, run the native denoiser parity matrix with:

```sh
tmp/qimg21-ref-venv/bin/python cuda/qimg21/regression.py \
  --native --model /mnt/nvme01/models/qimg-21 \
  --case 256x256:2:42
```

Native mode captures the exact PyTorch transformer input and BF16 timestep for
each denoising iteration, then replays that single denoiser call in the native
C/NVRTC runner. The acceptance gate is applied to every matched
`pred_NNN.npy` output (`cosine >= 0.99996` for these non-quantized weights).
It also runs a complete native trajectory from the shared initial latent
and checks every scheduler checkpoint at the same 0.99996 gate. Both
comparisons run even if the denoiser gate fails, and either failure fails
the case. Native mode requires BF16 activation boundaries. Intermediate
files from isolated denoiser replays are not trajectory checkpoints; the
actual free-running sequence is saved under `native/trajectory/`.

The native path is batch-1 text-to-image. It accepts either the model’s
recommended no-guidance path or a second negative embedding fixture for true
CFG; the latter runs two native denoiser passes per step and applies the same
linear blend as the Qwen-Image 2.1 pipeline. The native executable takes the
text encoder output as an F32 `.npy` fixture; tokenisation and the Qwen3-VL
text encoder remain at that Python boundary. The scheduler loop is native now,
while the official Qwen-Image 2.1 VAE is used as a separate decode stage.

For native true CFG, pass `--negative-prompt` and `--true-cfg-scale` to
`native_generate.py` (the negative embedding is exported beside the positive
fixture):

```sh
tmp/qimg21-ref-venv/bin/python cuda/qimg21/native_generate.py \
  --model /mnt/nvme01/models/qimg-21 \
  --prompt "a red apple on a white table" \
  --negative-prompt "blurry, distorted" --true-cfg-scale 4.0 \
  --height 256 --width 256 --steps 2 \
  --work-dir tmp/qimg21-native-cfg --out tmp/qimg21-native-cfg.png
```

The Python reference/runner also expose the model’s editing and true-CFG
controls for baseline comparisons:

```sh
tmp/qimg21-ref-venv/bin/python cuda/qimg21/test_cuda_qimg21.py \
  --generate --model /mnt/nvme01/models/qimg-21 \
  --prompt "change the background to a sunset beach" \
  --image input.png --negative-prompt "blurry, distorted" \
  --true-cfg-scale 4.0 --height 512 --width 512 \
  --steps 40 --out tmp/qimg21-edit.png
```

Editing remains a Python baseline: native condition-image tokens and the VAE
image-conditioning hand-off still need to be ported and accepted against the
same checkpoint gate. Quantized native weights likewise remain a separate
calibration task; use the provisional `--quantized` comparator threshold until
that work is measured.

## Native transformer step

Build the C runner, dump prompt embeddings with the already validated Python
runner, and make a deterministic latent fixture:

```sh
make -C cuda/qimg21 native
tmp/qimg21-ref-venv/bin/python cuda/qimg21/test_cuda_qimg21.py --test-text \
  --model /mnt/nvme01/models/qimg-21 --dump-dir tmp/qimg21-native-fixture
tmp/qimg21-ref-venv/bin/python cuda/qimg21/make_native_fixture.py \
  --prompt-embeds tmp/qimg21-native-fixture/prompt_embeds.npy \
  --dtype bf16 --torch-rng --out-dir tmp/qimg21-native-fixture
cuda/qimg21/test_cuda_qimg21_native \
  --model /mnt/nvme01/models/qimg-21 \
  --prompt-embeds tmp/qimg21-native-fixture/prompt_embeds.npy \
  --latents tmp/qimg21-native-fixture/latents.npy \
  --height-tokens 16 --width-tokens 16 --steps 4 \
  --dump-dir tmp/qimg21-native-steps \
  --out tmp/qimg21-native-fixture/native_latents.npy
```

The executable opens the two transformer safetensors shards directly and
executes all 32 blocks for every denoising step using custom NVRTC kernels for
zero-centred RMSNorm, Ada modulation, 3-axis RoPE, block-causal attention,
residual gates and SwiGLU, with BF16 cuBLAS GEMMs for the large matrix
products. Block weights are uploaded and released one block at a time, keeping
the transformer resident set appropriate for a 12–16 GB GPU. The native
scheduler mirrors FlowMatch Euler dynamic shifting from the local scheduler
config, and activation boundaries are rounded to BF16 to match the PyTorch
reference numerics. `--torch-rng` makes the fixture use the same CUDA RNG and
packing as the Diffusers pipeline; omit it only for a standalone NumPy smoke
input. Add `--verbose` for finite-value stage probes.

## Hybrid native image generation

For an end-to-end smoke image, the orchestration script uses the validated
Python text encoder, runs all denoising steps in the native executable, then
decodes the final normalized latents with `AutoencoderKLQwenImage21` after that
native subprocess exits. This process boundary prevents the transformer
allocations from competing with the VAE on the 16 GB RTX 5060 Ti:

```sh
make -C cuda/qimg21 native
tmp/qimg21-ref-venv/bin/python cuda/qimg21/native_generate.py \
  --model /mnt/nvme01/models/qimg-21 \
  --prompt "a red apple on a white table" \
  --height 256 --width 256 --steps 2 --seed 42 \
  --work-dir tmp/qimg21-native-generate \
  --out tmp/qimg21-native-generate.png
```

The work directory contains `prompt/prompt_embeds.npy`, the deterministic
initial `latents.npy`, one `steps/step_XXX.npy` file per Euler update, and the
final `native_latents.npy`. These arrays are the hand-off points for comparing
the native transformer/scheduler against the PyTorch reference.

Add `--native-vae` to use the native F32 CUDA decoder. It reads the original
VAE safetensors, applies latent denormalization and the learned post-quant
convolution, then runs the residual/attention/upsampling graph and clamps the
RGBA result. Text encoding still uses Python. This decoder supports single
images; encoding/editing and larger-resolution memory validation remain work
in progress. Its kernels and residual copies share the default CUDA stream
to avoid races with the shared VAE helpers' synchronous device copies.

Reproduce the decoder comparison with:

```sh
make -C cuda/qimg21 native-vae
tmp/qimg21-ref-venv/bin/python cuda/qimg21/vae_regression.py
```

The deterministic cases are 128x128/seed17, 256x256/seed42, and
128x256/seed123. On the RTX 5060 Ti, their minimum output cosine against the
official F32 PyTorch VAE with TF32 disabled was 0.99999999935, with maximum
mean absolute error 0.00000332 and finite outputs throughout. The script
enforces cosine >= 0.99996 and writes inputs, outputs, and results under
`tmp/qimg21-vae-regression`. These decoder results do not establish denoiser
or full-pipeline acceptance. `--case HEIGHTxWIDTH:SEED` is repeatable for
additional coverage.

The additional 512x512/seed42 F32 decoder comparison passed with cosine
0.99999999951 (native subprocess including startup: 12.04 seconds). Native
1024x1024 decoding completed, but the untiled F32 PyTorch reference ran out
of VRAM on this GPU; its 1024 parity is therefore unverified. This is not
the requested 1024x1024/40-step generation benchmark, which remains pending.
