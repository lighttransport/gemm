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

An optional **experimental** row-scaled INT8 transformer package can be
exported without modifying the original snapshot:

```sh
tmp/qimg21-ref-venv/bin/python cuda/qimg21/quantize_weights.py \
  --model /mnt/nvme01/models/qimg-21 --out tmp/qimg21-int8-weights
tmp/qimg21-ref-venv/bin/python cuda/qimg21/regression.py --native \
  --model /mnt/nvme01/models/qimg-21 --quantized-transformer tmp/qimg21-int8-weights \
  --case 256x256:2:42 --work-dir tmp/qimg21-int8-regression
```

The output directory must be new and needs space for the complete compressed
transformer. Each matrix stores symmetric `[-127,127]` INT8 rows with F32
scales; vectors remain in the original checkpoint. The native C loader
dequantizes one matrix at a time to BF16 before upload, retaining the existing
BF16 GEMM path. This compresses weights on disk; it is not an INT8 GEMM
acceleration or a reduction in the uploaded matrix's GPU size. Both the native
executable and `native_generate.py` accept `--quantized-transformer DIR`.
Missing/invalid matrices fail rather than silently falling back to BF16.

CPU tests compare native reconstruction bit-for-bit with PyTorch BF16 and
reject invalid scales, shapes, and payloads:
`make -C cuda/qimg21 test_quant_weights && tmp/qimg21-ref-venv/bin/python cuda/qimg21/test_quant_weights.py`.
Full-checkpoint export, GPU model parity, image quality, and quantizer-specific
threshold calibration remain unverified. The `0.995` quantized model-output
gate is still provisional, not measured acceptance. The regression driver
selects it only when an actual quantized package or quantize-on-load mode is supplied; `--quantized`
alone cannot relabel a BF16 run as a quantized experiment.

When disk space is limited, pass `--quantize-on-load int8-row` instead of
`--quantized-transformer DIR` to the native executable, `native_generate.py`,
or `regression.py --native`. This applies the identical symmetric row-INT8
quantization/reconstruction to one original matrix at a time, without a
second model copy. CPU tests verify bit-exact equivalence to exported packages
for BF16/F32 inputs, zero rows, and half-integer rounding ties. This avoids
disk storage, not quantization error or CPU quantization cost; matrices are
re-quantized on each load. Package export checks available disk space before
creating its output directory. On the saved 256x256/seed42 matched-input
fixture, streamed INT8 denoiser cosine is **0.9995792302** at the low timestep
and **0.9999601303** at timestep 1, with finite outputs (relative L2 0.02904
and 0.00896). Both clear the provisional 0.995 gate. A tighter 0.999 candidate
gate is being tested on additional seeds/resolutions; these first two cases
alone do not establish calibration or image quality. Full exported-model GPU
loading remains untested because the workspace lacks space for the copy.

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

CPU softmax-order replay gives a more specific lead: reverse-order 64-key
tiles with BF16-rounded unnormalized probabilities match the saved block-17
PyTorch image-attention output at cosine 0.9999999904 (99.8369% exact), versus
0.9999992812 for the full-F32 calculation. Reproduce with
`OMP_NUM_THREADS=2 python attention_order_probe.py --stage-dir DIR` using a
stage directory containing `pytorch_attn_matched.npy` from the CUDA probe.
This is an inference from saved outputs, not proof of PyTorch's dispatch.
The native executable exposes an experimental `--attention reverse64` kernel
for full-denoiser validation; default `--attention math` remains unchanged.
Native GPU replay of the saved 256x256/seed42 fixture scores 0.9999551703
at the low timestep (**fails**) and 0.9999807856 at timestep 1 (passes).
The much closer isolated attention arithmetic does not resolve accumulated
full-denoiser error, so this remains an experiment, not the default.
Both `regression.py` and `native_generate.py` accept
`--native-attention reverse64`; regression records the choice in
`native_config.json` and uses it for isolated denoiser and trajectory tests.
Matched-input regression now requires every captured `timestep_NNN.npy`
exactly: missing, extra, or invalid timesteps fail instead of falling back
to a reconstructed, potentially differently rounded schedule.

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

Native true-CFG regression uses an explicit negative prompt (including an
empty string) and scale greater than one:

```sh
tmp/qimg21-ref-venv/bin/python cuda/qimg21/regression.py \
  --native --model /mnt/nvme01/models/qimg-21 --case 256x256:2:42 \
  --negative-prompt "" --true-cfg-scale 3 \
  --work-dir tmp/qimg21-cfg-regression
```

The reference records positive and negative branches separately, plus one
combined `pred_NNN.npy` per scheduler step using the pipeline's BF16 CFG
arithmetic. Native replay receives both embedding fixtures. Fixture indexing
and empty-negative-prompt handling have a CPU-only test:
`tmp/qimg21-ref-venv/bin/python cuda/qimg21/test_reference_capture.py`.
That bookkeeping test is not model parity evidence; the CUDA CFG regression
still needs to meet both numerical gates.

The native path is batch-1 text-to-image. It accepts either the model’s
recommended no-guidance path or a second negative embedding fixture for true
CFG; the latter runs two native denoiser passes per step and applies the same
linear blend as the Qwen-Image 2.1 pipeline. The native executable takes the
text encoder output as an F32 `.npy` fixture; tokenisation and the Qwen3-VL
text encoder remain at that Python boundary. The scheduler loop is native now,
while the official Qwen-Image 2.1 VAE is used as a separate decode stage.

Editing bring-up has a CPU-native joint-layout builder in `joint_layout.h`:
four-token image-slot expansion, interleaved text/image scatter indices,
separate condition-image attention blocks, target modulation boundary, and
centered three-axis RoPE positions. Four layout tests match official Diffusers
metadata and RoPE exactly, including adjacent images and rectangular grids:

```sh
make -C cuda/qimg21 test_joint_layout
OMP_NUM_THREADS=2 tmp/qimg21-ref-venv/bin/python cuda/qimg21/test_joint_layout.py
```

Reference prediction captures also save each CFG branch's `img_mask`, text
key-validity mask, and `img_shapes` layout. This is preparatory work only:
the native denoiser does not yet consume editing layouts, and native VAE
encoding/vision-conditioned text encoding remain unimplemented.

`edit_kernels.h` adds experimental CUDA primitives for that layout:
interleaved text/image scatter, Q/K RMSNorm plus layout-driven three-axis
RoPE, and block-causal attention that keeps adjacent images separate.
They are not selected by the denoiser yet. A synthetic, checkpoint-free
GPU comparison is available:

```sh
make -C cuda/qimg21 test_edit_kernels
OMP_NUM_THREADS=2 tmp/qimg21-ref-venv/bin/python cuda/qimg21/editing_kernel_regression.py
```

The harness checks exact scatter and applies the 0.99996 gate to RoPE and
attention against official Diffusers metadata/RoPE and PyTorch math. Host
build and sm_120 NVRTC compilation pass; GPU comparison is queued and remains
unverified. Passing this synthetic test would not establish model editing
parity or implement the still-missing conditioning pipeline.

For native text-encoder bring-up, `test_cuda_qimg21.py --test-text
--dump-text-stages --dump-dir DIR` additionally records `text_positive/`
(and `text_negative/` when requested). Each directory contains exact integer
input IDs/masks, the full padded `hidden_prenorm.npy`, and `capture.json`
with the system-prefix crop index. Compare before final RMSNorm: the official
2.1 pipeline bypasses that normalization before extracting prompt embeddings.
The existing generic Qwen3 CUDA loader is not a drop-in replacement: this
checkpoint uses Qwen3-VL tensor names, 5,000,000 RoPE theta, and a pre-norm
output boundary.
The CPU fixture test is
`tmp/qimg21-ref-venv/bin/python cuda/qimg21/test_text_capture.py`; it verifies
integer preservation, pre-norm capture, unchanged outputs, and hook cleanup.

An experimental streamed native encoder is now available separately:

```sh
make -C cuda/qimg21 test_cuda_qimg21_text
tmp/qimg21-ref-venv/bin/python cuda/qimg21/native_text.py \
  --model /mnt/nvme01/models/qimg-21 --prompt "a red apple on a white table"
```

It gathers only requested embedding rows, streams one BF16 weight matrix at
a time, and implements all 36 text blocks with native RMSNorm, split-half
text RoPE, causal grouped-query attention, SwiGLU, and BF16 residual updates.
It omits the final RMSNorm and LM head. The C executable requires unpadded,
batch-one integer token IDs, rejects vision tokens, and limits inputs to
4096 tokens. Python handles only processor/tokenization and system-prefix
cropping; `--prepare-only` runs that stage without CUDA or model weights.
The original checkpoint configuration is required; this is not a general
Qwen3-VL loader. Host build, sm_120 NVRTC compilation, and GPU execution pass,
but the first English-prompt GPU comparison **fails numerical acceptance**:
full pre-norm cosine 0.9632453344, cropped-prompt cosine 0.9211241964.
Per-layer diagnostics exposed nondeterministic weight-upload ordering:
cuBLAS uses a nonblocking stream, so it must wait for staged host-to-device
copies before consuming each matrix. Explicit upload synchronization makes
two native runs bit-exact and improves cosine to 0.9993278549 / 0.9994377151
(full / cropped). BF16 rounding of unnormalized attention probabilities
improves these further to **0.9994088823 / 0.9995524853**, still below the
0.99996 gate. First-layer operator capture then identified a GEMM dispatch
difference: BF16-output GEMM matches PyTorch Q projection and Q normalization
bit-for-bit, whereas F32-output GEMM followed by rounding does not. The
experimental encoder now defaults to BF16-output GEMM; full/cropped cosine
is **0.9997771787 / 0.9994480992**, still failing. `--f32-gemm-output` on the
C executable retains the old path for diagnostics. Further arithmetic
diagnosis remains necessary; improving one operator does not establish
end-to-end acceptance.
It is deliberately
not the default generation encoder until comparison against captured
`hidden_prenorm.npy` and cropped prompt embeddings meets the strict gate.
CPU tokenization checks against the installed official pipeline cover an
English prompt, an empty negative prompt, and Unicode/newline text:

```sh
tmp/qimg21-ref-venv/bin/python cuda/qimg21/test_native_text.py \
  --model /mnt/nvme01/models/qimg-21
```

The native text acceptance driver captures the official BF16 encoder first,
then releases that process before running the streamed C encoder:

```sh
tmp/qimg21-ref-venv/bin/python cuda/qimg21/text_regression.py \
  --model /mnt/nvme01/models/qimg-21 --negative-prompt "" \
  --work-dir tmp/qimg21-text-regression
```

Both the complete pre-final-RMSNorm hidden state and the cropped prompt
embedding must independently meet cosine `>= 0.99996`. The driver saves
cosine, relative L2, MAE, and gate results in `results.json`. It rejects
non-finite outputs, shape mismatches, padded/vision inputs, and inconsistent
reference cropping. Use a fresh work directory for each run; an explicit
`--reference-dir` may reuse previously captured official fixtures, but native
outputs must still be new. `test_text_regression.py` exercises the gate and
fixture rejection paths without CUDA; passing that unit test does not prove
the native model meets the gate.
Text-stage reference captures now include `layer_NN.npy`; the native text
executable can save matching boundaries with `--dump-dir DIR`.
First-layer `stage_*.npy` files expose normalization, Q/K/V projections,
Q/K normalization, attention output, and MLP projections for matched-stage
comparisons.
RoPE-specific diagnosis shows precise CUDA math matches the first layer's
rotary Q/K values bit-for-bit; default fast math differs in 36 Q and 2 K
elements on the 29-token prompt. A full precise-math text run still fails
(full cosine 0.9997824312, cropped 0.9994433137), so this is not a solution
to encoder acceptance. Unlike the text projections, changing all denoiser
GEMMs to BF16 outputs left both low-timestep attention-mode results unchanged;
that unsuccessful denoiser experiment was removed.

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
1024x1024 native decoding completed, but the initial untiled F32 PyTorch
reference ran out of VRAM while retaining unused temporal caches. The
cache-bounded reference below now completes and validates 1024 parity.

For memory-bounded untiled PyTorch comparison, `vae_regression.py` has an
opt-in `--discard-frame-cache` mode. It executes the official decoder with
the same arithmetic and first-chunk branches, but discards temporal cache
writes that only a later video frame could use. Checked cache access fails
if any discarded slot is read again. This is not spatial tiling, and does
not change temporal upsampling behavior by setting `feat_cache=None`.
Results explicitly record this reference mode. CPU tests are bit-exact for
individual official modules and a reduced-width complete five-stage decoder:
`OMP_NUM_THREADS=2 tmp/qimg21-ref-venv/bin/python cuda/qimg21/test_vae_reference.py`.
Full-checkpoint GPU validation now passes. At 256x256/seed42, this reference
is bit-identical to the saved original cached PyTorch decode (maximum
absolute difference zero). Native-vs-reference cosine is 0.9999999996157.
At 1024x1024/seed42, the untiled cache-bounded reference completes without
OOM; native cosine is **0.9999999996970**, MAE 1.8036e-6, with finite outputs.
Native decode subprocess time was 45.8903 seconds. Reproduce with:

```sh
tmp/qimg21-ref-venv/bin/python cuda/qimg21/vae_regression.py \
  --discard-frame-cache --case 256x256:42 --case 1024x1024:42 \
  --work-dir tmp/qimg21-vae-cachebounded-regression
```

The reference model remains F32 with TF32 disabled. These are seeded
single-frame decoder checks, not encoder/editing or denoiser parity evidence.

## Completed 1024x1024/40-step generation benchmark

Command (RTX 5060 Ti, 15.5 GiB reported by CUDA):

```sh
/usr/bin/time -v tmp/qimg21-ref-venv/bin/python cuda/qimg21/native_generate.py \
  --model /mnt/nvme01/models/qimg-21 --prompt "a red apple on a white table" \
  --height 1024 --width 1024 --steps 40 --seed 42 --native-vae \
  --work-dir tmp/qimg21-native-1024-40-benchmark \
  --out tmp/qimg21-native-1024-40-benchmark.png
```

Completed with exit status 0 in **44m 08.05s** wall time. This is end-to-end
time including the Python text encoder, native BF16 denoiser (default `math`
attention), and native F32 VAE; it is not denoiser-only timing. Maximum host
RSS was 18,612,956 KiB, with zero swaps. CPU development/tests ran concurrently
on the host, but no other GPU workload overlapped this run. One mid-denoising
GPU snapshot showed 6,019 MiB used; this was **not** a peak-VRAM measurement
and does not establish a hard 12 GiB limit for the entire pipeline.

All 40 saved `[4096,64]` latent checkpoints are finite. The decoded
`[4,1024,1024]` tensor is finite and within `[-1,1]`; the saved image is a
1024x1024 RGBA PNG. Visual inspection shows a coherent red apple on a white
surface, consistent with the prompt. Fixtures, image, and timing log remain
under ignored `tmp/`, not in Git. No corresponding 40-step PyTorch trajectory
was compared: successful generation does not establish the strict parity
acceptance gate, which still fails on the small low-timestep denoiser case.
