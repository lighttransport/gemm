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
that selects the row-INT8 MRE `<= 0.10` gate described below, while cosine is
still reported as a diagnostic. Other quantizers require their own
calibration. Use
`--cosine-threshold X` to record an explicit threshold in benchmark logs.
The regression driver also records `initial_latents.npy`, preserving the exact
PyTorch-packed noise input for native denoiser comparisons.

The non-quantized acceptance target is **0.99996**. Historical sections below
retain the threshold and pass/fail language used when those experiments were
recorded; current harness results use
`compare.NONQUANTIZED_COSINE_THRESHOLD == 0.99996`.

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
Full-checkpoint export, editing/CFG validation, and 40-step image-quality
validation now pass (below).
Streamed row-INT8 GPU calibration establishes a normalized mean absolute error
(`mean(abs(error)) / mean(abs(reference))`) gate of `<= 0.10`. The original
16-point text-to-image calibration peaked at `0.036572`; a subsequent editing
case peaked at `0.045095`, and true-CFG editing peaked at `0.083975`, leaving
about 19% headroom. The regression driver
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
and 0.00896). Both clear the measured 0.999 gate. Full exported-model GPU
loading was initially deferred for lack of disk space; the following check
now covers that path.

After space recovery, a complete original-checkpoint export produced **232
matrices**, 7,121,043,712 payload bytes (about 6.7 GiB including files), with
the completion marker written last. The package and streaming paths produce
**bit-identical** predictions on the saved 256x256/seed42 low-timestep input.
Both score **0.999569753579132** against PyTorch, relative L2
**0.029346483906742**, finite and above the calibrated 0.999 INT8 gate.
This measurement uses the current ordered-GELU implementation and default
math attention, not the historical pre-GELU numbers above. Reproduce with:

```sh
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 tmp/qimg21-ref-venv/bin/python cuda/qimg21/quantize_weights.py \
  --model /mnt/nvme01/models/qimg-21 --out tmp/qimg21-int8-package
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 tmp/qimg21-ref-venv/bin/python cuda/qimg21/quant_package_regression.py \
  --model /mnt/nvme01/models/qimg-21 --package tmp/qimg21-int8-package \
  --reference-dir tmp/qimg21-blockref --step 1 --height-tokens 16 --width-tokens 16 \
  --work-dir tmp/qimg21-int8-package-low
```

Export/output directories must be new. The checker requires both original-
reference parity and package/stream bit equality; it rejects editing and CFG
fixtures. Results are recorded in `tmp/qimg21-int8-package-low/results.json`.
This proves full package loading and one matched prediction, not 40-step
quality or broad packaged-weight regression. Original model files are
unchanged and the generated package remains outside Git.

The additional calibration matrix passed all eight matched-input predictions
and eight free-running trajectory checkpoints, all finite:

| Case | Minimum prediction cosine | Maximum prediction relative L2 | Minimum trajectory cosine |
| --- | ---: | ---: | ---: |
| 256x256, 2 steps, seed 7 | 0.999476820 | 0.032350584 | 0.999919228 |
| 256x512, 2 steps, seed 123 | 0.999172219 | 0.040695879 | 0.999931849 |
| 512x512, 4 steps, seed 42 | 0.999486525 | 0.032055460 | 0.999937582 |

```sh
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 tmp/qimg21-ref-venv/bin/python cuda/qimg21/regression.py \
  --native --model /mnt/nvme01/models/qimg-21 --quantize-on-load int8-row \
  --cosine-threshold 0.999 --case 256x256:2:7 --case 256x512:2:123 \
  --case 512x512:4:42 --work-dir tmp/qimg21-int8-calibration
```

This initial matrix established the row-INT8 error scale; the editing, CFG,
and 40-step validations below extend it. It does not calibrate other
quantization schemes.
`quant_quality.py` decodes paired final latents with the same validated native
F32 VAE, saves reference/quantized PNGs, and reports RGB error for visual review.
For the 512x512/four-step seed42 pair, both decoded images are finite: RGB
cosine **0.999994245**, MAE **0.00162609** on [0,1], PSNR **50.62 dB**,
alpha MAE **0.0000219955**. Visual review found the same red apple and white
background without an obvious quantization-specific artifact. Both are soft
four-step outputs, so this is not evidence of production 40-step image quality.

```sh
OMP_NUM_THREADS=2 tmp/qimg21-ref-venv/bin/python cuda/qimg21/quant_quality.py \
  --model /mnt/nvme01/models/qimg-21 --height 512 --width 512 \
  --reference-latents tmp/qimg21-int8-calibration/512x512-s4-seed42/reference/step_003.npy \
  --quantized-latents tmp/qimg21-int8-calibration/512x512-s4-seed42/native/trajectory/final_latents.npy \
  --out-dir tmp/qimg21-int8-quality-512-v2
```

Exact efficient-SDPA editing with the exported row-INT8 package passes the
MRE gate: the two prediction MREs are `0.028851` and `0.045095`, and the two
trajectory MREs are `0.033041` and `0.032932`. True-CFG scale 4 editing also
passes, with worst prediction MRE `0.077929` and worst trajectory MRE
`0.083975`. Reproduce either capture with `editing_regression.py` plus
`--quantized-transformer tmp/qimg21-int8-package`; the driver detects and
validates negative prompt fixtures automatically.

The full 1024x1024, 40-step, seed-42 row-INT8 trajectory also passes all 40
checkpoints. MRE increases smoothly from `0.000297` to a maximum of `0.017865`;
final latent cosine is `0.999805733`. Transformer runtime was `17:35.90` on
the RTX 5060 Ti, with four persistent CUTLASS workspace allocations. A
200-millisecond `nvidia-smi` sample during a full-resolution transformer step
measured peak process VRAM at `2010 MiB`. Decoding
the reference and quantized final latents through the same native F32 VAE
gave RGB cosine `0.999982435`, relative L2 `0.00592812`, MAE `0.00202228`,
PSNR `46.58 dB`, and alpha MAE `0.00002860`. Both finite images show the same
photorealistic red apple and table, with no visually apparent
quantization-specific artifact.

### Native W8A8 tensor-core execution

Pass `--int8-tensor-core` with an exported row-INT8 package to keep weights as
INT8 on device, dynamically quantize each BF16 activation row to INT8, execute
signed `mma.sync.m16n8k32.s32.s8.s8.s32` dot products, and fuse the activation
and weight row scales into the F32 output epilogue. This is genuine W8A8
tensor-core execution; it does not reconstruct BF16 weights before GEMM. The
option is supported by the native executable, `native_generate.py`,
`regression.py`, and `editing_regression.py`.

```sh
tmp/qimg21-ref-venv/bin/python cuda/qimg21/regression.py --native \
  --model /mnt/nvme01/models/qimg-21 \
  --quantized-transformer tmp/qimg21-int8-package --int8-tensor-core \
  --native-attention cutlass-efficient --native-normalization vector4 \
  --native-rope host-table-exact --include-full \
  --work-dir tmp/qimg21-w8a8-regression
```

W8A8 has a separate calibrated MRE gate of `<= 0.25`; dynamic activation
quantization is materially noisier than row-INT8 weights with BF16 compute.
In exact two-step editing, prediction MREs are `0.072323` and `0.202099`, and
trajectory MREs are `0.082492` and `0.081736`. True-CFG scale 4 trajectory MRE
peaks at `0.196597`; its isolated low-timestep guided prediction reaches
`0.333089` with every block in W8A8. For strict CFG prediction quality, pass
`--int8-bf16-tail-blocks 16`. This reconstructs the final 16 transformer
blocks from the row-INT8 package for BF16 GEMM while retaining the custom
fused-scale INT8 tensor-core path in the first 16 blocks and the surrounding
projections. The calibrated true-CFG prediction MREs are `0.177098` and
`0.249895`, and trajectory MREs are `0.190641` and `0.189244`, all within the
`<= 0.25` gate. Each CFG prediction still executes 230 custom INT8 MMA GEMMs.
The option is explicit because the accuracy/speed tradeoff is specific to
high-guidance editing; ordinary generation remains fully W8A8 by default.

The full 1024x1024, 40-step W8A8 trajectory passes all checkpoints, with MRE
rising smoothly to `0.041948` and final latent cosine `0.998980616`. It ran in
`8:12.02`, versus `17:35.90` for BF16-compute row-INT8 (2.15x faster), executed
9,000 custom INT8 MMA GEMMs, and sampled at `2148 MiB` peak process VRAM.
Shared native-VAE decoding gave RGB cosine `0.999871005`, MAE `0.00668258`,
PSNR `37.87 dB`, and no visually apparent change to the generated red apple.
Compute Sanitizer reports zero errors for the complete 227-GEMM editing step.

Current native BF16 arithmetic explicitly rounds the text projection before
GELU and Q/K normalization before multiplication by the learned RMS weights.
An older automatic-SDPA fixture reached 0.999981133 at timestep 1 and
0.999955009 at timestep 0.02001953125. That diagnostic result is superseded by
the pinned efficient-SDPA acceptance path documented below; it remains useful
evidence that CUDA's automatic backend mix is not a stable exact oracle.

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

For disk-bounded capture, set `QIMG21_STAGE_KEYS=rope_q,rope_k,v,attn_raw`
(exact comma-separated labels). A text-to-image probe also needs `txt_input`;
editing probes instead take `--editing-reference CAPTURE_DIR` and reproduce
the official per-segment attention calls from its image layout. This separates
attention arithmetic from accumulated projection/normalization errors.

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
key-validity mask, and `img_shapes` layout. An omitted official key mask is
captured as all-valid, matching the transformer semantics. Native VAE
encoding/vision-conditioned text encoding remain unimplemented.

`prepare_edit_fixture.py` converts a captured editing call into a native-ready
fixture without re-encoding or renormalizing its inputs:

```sh
tmp/qimg21-ref-venv/bin/python cuda/qimg21/prepare_edit_fixture.py \
  --reference-dir CAPTURE_DIR --out-dir NEW_FIXTURE_DIR --step 0 --branch positive
```

It writes the packed condition/target latents separately, prompt embeddings,
an exact timestep manifest, and `layout.txt` for `q21_layout_read`. It rejects
padding, malformed image blocks, nonfinite/unrepresentable inputs, and existing
output directories. Four additional CPU tests cover round-trip loading and
failure cases. Its manifest explicitly marks editing unvalidated.

The native denoiser now accepts `--editing-layout layout.txt` together with
`--condition-latents condition_latents.npy`, while `--latents` contains only
the target latents. It assembles interleaved text/image tokens, applies the
zero-timestep modulation to the condition prefix, predicts only the target
tail, and leaves condition latents unchanged through Euler steps. The default
text-to-image path is unchanged. `--attention reverse64` is now available as
an opt-in layout-aware arithmetic experiment. Its timestep-1 editing cosine is
**0.999915311**, worse than default attention, so it remains nondefault.
This is experimental:
the first full-model editing prediction (256x256 target, 1024x1024 condition,
seed42, captured timestep 1) is finite with cosine **0.999932378** and relative
L2 **0.011906675**. It **fails** the 0.99996 gate. A mid-run total GPU-memory
snapshot was 5201 MiB; this is not a measured peak. All 20 CPU tests pass.

```sh
cuda/qimg21/test_cuda_qimg21_native --model /mnt/nvme01/models/qimg-21 \
  --prompt-embeds FIXTURE/prompt_embeds.npy --latents FIXTURE/target_latents.npy \
  --condition-latents FIXTURE/condition_latents.npy --editing-layout FIXTURE/layout.txt \
  --height-tokens 16 --width-tokens 16 --steps 1 --timestep 1 \
  --pred-dir NEW_PRED_DIR --out NEW_PRED_DIR/latents.npy
```

Use the exact captured timestep, not a guessed value. The installed official
one-step editing scheduler produced nonfinite output; two-step capture is
used instead. A native `--steps 1 --timestep T` call above is a matched-input
prediction diagnostic, not the official one-step generation trajectory.

Experimental native editing CFG accepts `--negative-editing-layout` together
with `--negative-prompt-embeds` and `--guidance-scale S` (S > 1). Each branch
gets a separately validated layout and its own token count; both reuse the
same fixed condition latents and target state, and combine target predictions
with the existing BF16 CFG rounding. Missing branch-layout pairs are rejected.
This branch-aware path builds and has CLI guard coverage, but guided editing
GPU parity is not yet accepted. The editing regression driver below reads the
captured CFG scale, validates paired branch dimensions/latents, and supplies
each branch's own layout and embeddings. Guided editing validation completed
and fails the strict gate: prediction cosines **0.999770114 / 0.999445503**,
trajectory cosines **0.999757280 / 0.999763197** (256x256 target, seed42,
empty negative prompt, scale4). Outputs are finite. CLI and metadata guards
are covered by CPU tests.

The separate text-to-image true-CFG regression (256x256, two steps, seed42,
empty negative prompt, scale4) runs to completion with finite outputs but
**fails** the strict gate: prediction cosines **0.999959521 / 0.999387980**,
trajectory cosines **0.999956809 / 0.999956741**. Reproduce with
`regression.py --native --model /mnt/nvme01/models/qimg-21 --negative-prompt ''
--true-cfg-scale 4 --case 256x256:2:42 --work-dir tmp/qimg21-truecfg-regression`.

For full fixture-driven editing validation, `editing_regression.py` checks
every captured timestep at matched inputs, then runs an independent Euler
trajectory from the first target latent. It requires unchanged conditioning
and prompt/layout across calls, validates CFG capture metadata, writes per-checkpoint
results, and returns nonzero if any cosine is below 0.99996:

```sh
OMP_NUM_THREADS=2 tmp/qimg21-ref-venv/bin/python cuda/qimg21/editing_regression.py \
  --model /mnt/nvme01/models/qimg-21 \
  --reference-dir tmp/qimg21-edit-reference-256-s2-mask \
  --work-dir tmp/qimg21-edit-regression-256
```

After editing integration, the original 256x256 text-to-image timestep-1
prediction still has cosine 0.9999811334183142 versus its saved reference,
the same measured result as before integration. The independent editing
rerun reproduces its failing timestep-1 cosine exactly (0.9999323775131732).
The complete two-step editing regression fails all four checks: prediction
cosines **0.999932378 / 0.999845282**, trajectory cosines
**0.999911016 / 0.999911662**. All outputs are finite. Separately, disabling
CUDA fast math and FMA for the original text-to-image low-timestep fixture
gives **0.999953708**, also below the gate; neither diagnostic changes defaults.

First-block editing attention replay on identical native Q/K/V yields cosine
**0.999999823** versus PyTorch default segmented SDPA (93.7835% exact), and
**0.999999983** versus math SDPA (99.7820% exact). Math target-only cosine is
**0.999999997**. These measurements isolate attention from upstream drift;
they do not establish full-model parity. The attention probe also reports
prefix and target errors separately to avoid hiding target drift in a large
condition prefix.

`test_bf16_attention` is a diagnostic replay using the existing custom
`flash_attn_bf16_xq` tensor-core kernel (not PyTorch). It processes each image
block with its permitted key prefix and each causal text query separately,
then rounds output to BF16. This checks whether tensor-core accumulation is
closer to default SDPA before changing production dispatch:

```sh
make -C cuda/qimg21 test_bf16_attention
cuda/qimg21/test_bf16_attention tmp/qimg21-edit-attn0 \
  tmp/qimg21-edit-fixture-256-s2-mask/layout.txt \
  tmp/qimg21-edit-attn0/mma_attention.npy
```

The original forward-32 tensor-core replay passes isolated attention: aggregate
cosine 0.999999898, target 0.999999496. The standalone reverse-64 variant in
`mma64_kernels.h` improves these to **0.99999999708 / 0.99999999704** and
**99.9478% exact overall**, using 64-key tiles, reversed key traversal, and
BF16 probability fragments. Add `--reverse64` to the replay command to test it.

The full native runner exposes this as opt-in `--attention mma64` (Python
wrappers: `--native-attention mma64`). It reuses the BF16 MLP scratch space
for packed Q/K/V and handles each image block and causal text query separately.
It is not the default: the original-weight text-to-image low-timestep cosine
is **0.999953858**, still below the required 0.99996. Improved isolated attention
therefore does not establish full-model acceptance.
The editing timestep-1 prediction is **0.999918015**, also failing and worse
than default scalar attention despite better first-block replay parity.
Both full-model experiments completed; no default was changed.

Normalization is separately isolated with `test_norm_vector` and `norm_probe.py`.
The experimental `--normalization vector4` native option uses vector-four,
four-warp Welford ordering based on [PyTorch's CUDA layer-normalization
implementation](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/layer_norm_kernel.cu).
On identical saved first-block hidden/modulation tensors it is bit-exact with
PyTorch across **1,110,016 values** (the default differs in 37 values). It is
not the default: full low-timestep cosine is **0.999953908**, still failing.

```sh
make -C cuda/qimg21 test_norm_vector
cuda/qimg21/test_norm_vector tmp/qimg21-batch2-low tmp/qimg21-norm-vector.npy
OMP_NUM_THREADS=2 tmp/qimg21-ref-venv/bin/python cuda/qimg21/norm_probe.py \
  --stage-dir tmp/qimg21-batch2-low --candidate tmp/qimg21-norm-vector.npy \
  --out tmp/qimg21-norm-vector-results.json
```

This probe requires exact equality, not just a passing cosine. The native
option changes block and final normalization; combined attention/normalization
full-model validation remains necessary despite isolated exactness.
The combined `--normalization vector4 --attention mma64` low-timestep run
completed at cosine **0.999954275**, still below acceptance. On the same saved
first-block inputs, the native SwiGLU activation is bit-exact with PyTorch.

`rope_probe.py` isolates real-weight Q RMSNorm/complex RoPE on matching inputs:

```sh
OMP_NUM_THREADS=2 tmp/qimg21-ref-venv/bin/python cuda/qimg21/rope_probe.py \
  --model /mnt/nvme01/models/qimg-21 --stage-dir tmp/qimg21-batch2-low \
  --height-tokens 16 --width-tokens 16
```

The saved first-block Q RoPE has 201 mismatched values, cosine
**0.999999997836**. Replaying the native pairwise RMS reduction with official
RoPE finds 9 normalization mismatches and 11 post-RoPE mismatches; 190 values
differ between native output and that replay. This points to rotary frequency
or complex arithmetic as the larger local discrepancy, not SwiGLU. These
operator diagnostics do not establish whole-model parity.

`rope_probe.py --export-replay DIR` also exports normalized Q and official
complex frequencies for `test_rotary_arithmetic`. On the saved first-block
fixture, fused arithmetic with official frequencies differs in just five
values (max absolute error 0.0000152588); non-fused arithmetic differs in
eight (max 0.0078125). A C host table using F32 `powf`/reciprocal/`cosf`/`sinf`
reproduces the same five small output mismatches as official frequencies.

```sh
make -C cuda/qimg21 test_rotary_arithmetic
cuda/qimg21/test_rotary_arithmetic tmp/qimg21-rotary-arithmetic 0
cuda/qimg21/test_rotary_arithmetic tmp/qimg21-rotary-arithmetic 0 15 16 16
```

The second invocation regenerates frequencies for 15 text tokens and a
16x16 target grid; modes 0/1/2 select fused/non-fused/double-intermediate math.
The native denoiser exposes host frequencies as opt-in `--rope host-table`,
supporting both ordinary and editing layouts without Python at runtime.
Full low-timestep cosine is **0.999954723**, still failing 0.99996. Defaults
remain unchanged; isolated rotary improvement is not model acceptance.

The denoiser text projection now uses ordered cubic evaluation and a
double-evaluated tanh rounded to F32 before the BF16 activation boundary.
This matches all four isolated stages (zero-centered RMSNorm, input GEMM,
GELU, output GEMM) bit-for-bit with PyTorch on the saved 15-token fixture.
The earlier fast GELU differed in 3,002 values, propagating to 5,896 output
projection mismatches. The full low-timestep cosine improves from 0.999955009
to **0.999955965**, but still **fails** 0.99996. This fixes a demonstrated
operator discrepancy; it does not close model acceptance.

```sh
make -C cuda/qimg21 test_text_projection
cuda/qimg21/test_text_projection /mnt/nvme01/models/qimg-21 \
  tmp/qimg21-batch2-low/txt_input.npy NEW_PROJECTION_DIR
OMP_NUM_THREADS=2 tmp/qimg21-ref-venv/bin/python cuda/qimg21/text_projection_probe.py \
  --model /mnt/nvme01/models/qimg-21 --input tmp/qimg21-batch2-low/txt_input.npy \
  --native-dir NEW_PROJECTION_DIR
```

The C replay accepts `--legacy-gelu` for the previous arithmetic. It tests
the denoiser's projection, not the separate Qwen3-VL text encoder. The prior
host-table/vector-normalization/MMA64 combination (before this GELU fix)
completed at cosine 0.999952815 and did not meet acceptance either.

With the GELU fix, the combined `--attention mma64 --normalization vector4
--rope host-table` path passes the previously failing saved low-timestep
checkpoint: cosine **0.9999618066543818**, relative L2 **0.0087566643**.
The corrected GELU/default-attention high-timestep check is **0.999987734990**.
These are bounded checkpoint results, not complete acceptance or a default
configuration change. A refreshed combined first-block capture has bit-exact
text projection, modulation, Q and V; block output is 98.7822% exact with
cosine 0.999999999837.

Both Python wrappers now accept `--native-normalization` and `--native-rope`
alongside `--native-attention`; regression records them in `native_config.json`.
Broader validation was run with:

```sh
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 tmp/qimg21-ref-venv/bin/python cuda/qimg21/regression.py \
  --native --native-attention mma64 --native-normalization vector4 --native-rope host-table \
  --model /mnt/nvme01/models/qimg-21 --case 256x256:2:42 --case 256x512:2:123 \
  --case 512x512:4:7 --work-dir tmp/qimg21-accurate-regression
```

The 256x256/seed42 case passes all four checks: predictions
**0.999992279 / 0.999961807**, trajectory **0.999985893 / 0.999986316**.
The original matrix process subsequently exited during the rectangular case
when the filesystem filled. Its partial remaining cases are not acceptance
evidence. After space recovery, the two unfinished cases were restarted in
`tmp/qimg21-accurate-regression-remaining` with fresh outputs.
The restarted cases completed but **failed** the unchanged 0.99996 gate:
256x512/seed123 predictions were 0.999991815 / **0.999937836**, while
512x512/seed7 predictions were 0.999988945 / 0.999992314 / 0.999991200 /
**0.999946099**. All six trajectory checkpoints passed (minimum 0.999980070).
The combined configuration therefore remains experimental; passing trajectories
does not override failed matched-input predictions.
Completed large replay outputs may be compressed as `.npy.gz`; decompress
before using the replay commands. The original forward-MMA candidate can be
regenerated from its preserved Q/K/V and layout; its metrics remain recorded.

`editing_regression.py` also accepts and records the same three native
accuracy options. Positive-only editing validation with this combination
completed in `tmp/qimg21-edit-accurate-regression`: the high prediction passes
at **0.9999689764854415**, but the low prediction **0.999887702401766** and
trajectory **0.999957597830316 / 0.9999571642207834** fail. Editing is not
accepted. Native output now propagates header, payload and close/flush errors
to a nonzero runner status, including requested diagnostic captures; incomplete
files must not be used as reference evidence. CPU coverage exercises successful
NPY round-trip, missing output parent, and delayed `/dev/full` flush failure:

```sh
TMPDIR="$PWD/tmp" make -C cuda/qimg21 test_scheduler
tmp/qimg21-ref-venv/bin/python cuda/qimg21/test_native_output.py
```

An additional opt-in `--rope host-table-vector4` (Python wrappers:
`--native-rope host-table-vector4`) retains host frequencies and changes only
Q/K RMS reduction to PyTorch's four-adjacent-values-per-lane, descending
32-lane sum. `rope_probe.py` independently reproduces both reduction trees.
On saved first-block Q, the old RMS tree differs in 9 values while the vector
tree matches official RMSNorm exactly. Native combined RMS/RoPE leaves 5 tiny
complex-arithmetic differences among 1,110,016 values (relative L2 9.58e-9).
Original weights, `mma64` attention and `vector4` layer normalization give:

| Matched low-timestep prediction | Host table | Host table + vector RMS | Gate |
| --- | ---: | ---: | --- |
| 256x256, seed42 | 0.999961807 | 0.999962479 | pass |
| 256x512, seed123 | 0.999937836 | 0.999944498 | fail |
| 512x512, seed7 | 0.999946099 | 0.999948000 | fail |

Artifacts are `tmp/qimg21-rms-vector-{low,rect-low,512-low}`. These are isolated
matched-input predictions, not a rerun of all trajectories or editing/CFG.
No defaults or acceptance thresholds change. CPU tests remain 25/25 passing.

The additional `--attention mma64-flash` option (Python wrappers:
`--native-attention mma64-flash`) tests base-2 softmax with separately rounded
score/max scaling and lane-local denominator sums reduced only at the end.
This follows the numerical structure in
[upstream FlashAttention softmax](https://github.com/Dao-AILab/flash-attention/blob/main/csrc/flash_attn/src/softmax.h),
including its documented PyTorch unfused-scaling variant; it is not a claim
of a bit-exact port of the installed backend. Replay with
`test_bf16_attention STAGES LAYOUT OUT.npy --flash-softmax`.
Saved editing-block Q/K/V target-region cosine improves from 0.999999997039
to 0.999999998629, relative L2 from 7.70e-5 to 5.24e-5, and exact fraction
from 99.9435% to 99.9739%. Original `--reverse64` replay remains bit-identical
to its saved baseline. Text-only rows remain less exact (cosine 0.999999772809,
92.6904% elementwise equality); scalar math is worse on these same inputs.
Full matched low-timestep predictions with vector normalization and vector
RMS improve to **0.999948668832** at 256x512/seed123 and **0.999948430099** at
512x512/seed7, but both still fail 0.99996. Artifacts are under
`tmp/qimg21-flash-softmax-{rect-low,512-low}` and `tmp/qimg21-edit-attn0`.
All 25 CPU tests pass; broader trajectory/editing acceptance remains unproven.
CPU dispatch profiling of this GPU replay confirms that both masked text
segments use `aten::_scaled_dot_product_efficient_attention`, while condition
and target image segments use `aten::_scaled_dot_product_flash_attention`.
The trace is `tmp/qimg21-attention-backends.log`. Matching the masked-text
backend's arithmetic is therefore a separate remaining parity task.

Forward 64-key traversal is now available in the standalone attention replay:
`--forward64` (ordinary softmax) and `--forward64-flash` (Flash-style softmax).
On the same saved editing Q/K/V, ordinary forward traversal raises text-row
cosine from 0.999999772809 to **0.999999997115** and elementwise equality from
92.6904% to **99.8853%**. The leading 8 text rows match exactly; the 12 text
rows after the condition image are 99.8088% exact. Forward Flash-style
softmax is slightly worse on text. Forward traversal worsens target image
cosine to 0.999999502640, so it is not suitable for replacing image attention.

The experimental `--attention mma64-mixed` (Python: `--native-attention
mma64-mixed`) selects ordinary forward traversal only for text rows and
reverse Flash-style traversal for condition/target images. Both paths remain
native BF16 tensor-core kernels, reusing the same buffers. Defaults and the
0.99996 acceptance target are unchanged. All 25 CPU tests pass. Full saved
editing validation was run with:

```sh
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 tmp/qimg21-ref-venv/bin/python cuda/qimg21/editing_regression.py \
  --model /mnt/nvme01/models/qimg-21 \
  --reference-dir tmp/qimg21-edit-reference-256-s2-mask \
  --work-dir tmp/qimg21-edit-mixed-regression \
  --native-attention mma64-mixed --native-normalization vector4 --native-rope host-table-vector4
```

It completed but failed all four gates: predictions **0.999923131554 /
0.999884239326**, trajectory **0.999899042085 / 0.999899960032**. All outputs
are finite. This combination is not accepted; operator replay improvements
alone do not establish full denoiser acceptance. Compared with the earlier
editing combination, this run also changes RMS and image softmax arithmetic,
so its differences cannot be attributed solely to text traversal order.

A controlled high-timestep editing ablation now separates these changes.
Every row uses the same saved input, original weights and vector layer norm:

| Attention | Q/K RMS | Prediction cosine | Gate |
| --- | --- | ---: | --- |
| MMA64 ordinary reverse | original tree | 0.999968976485 | pass |
| MMA64 ordinary reverse | vector order | 0.999935711069 | fail |
| MMA64 Flash-style reverse | original tree | 0.999954412383 | fail |
| MMA64 Flash-style reverse | vector order | 0.999929380655 | fail |
| Mixed forward text / Flash-style image | vector order | 0.999923131554 | fail |

New artifacts are `tmp/qimg21-edit-{mma-vector,flash-tree,flash-vector}-high`.
Both isolated RMS and softmax changes worsen this complete checkpoint despite
their same-input operator improvements. Do not promote them based solely on
operator similarity or the smaller text-to-image case. The earlier accepted
high checkpoint still does not establish editing acceptance: its low checkpoint
fails. A replay-only forward exp2 variant modeled on the masked backend's
separate score/log2 scaling left text equality unchanged at 99.8853%; it was
removed rather than adding another runner option. Its output remains in
`tmp/qimg21-edit-attn0/efficient_softmax.npy` for diagnostics.

Two further checks rule out specific explanations for the editing drift:

- A separate diagnostic build changed every denoiser GEMM output to BF16
  before expanding it back to F32. On the editing low-timestep fixture,
  prediction is bit-identical to the ordinary F32-output GEMM path, including
  the failing cosine **0.999887702401766**. The diagnostic production-source
  change was removed; output is `tmp/qimg21-edit-bf16-output-low`.
- Native fast `tanhf`, followed by the existing BF16 rounding operation,
  matches PyTorch BF16 `tanh` on **all 65,280 finite BF16 bit patterns**,
  including signed zeros: zero bit mismatches and zero maximum absolute error.
  This checks the gate activation only, not the complete residual update.
  Reproduce the GPU regression (a fresh output directory is required):

```sh
TMPDIR="$PWD/tmp" make -C cuda/qimg21 test_tanh_bf16
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 tmp/qimg21-ref-venv/bin/python cuda/qimg21/tanh_regression.py
```

Results are recorded in `tmp/qimg21-tanh-regression/results.json` with Torch
version and GPU identity. These checks do not waive the failing model gates.

### Matched-input transformer-block replay

To separate accumulated drift from a block's own arithmetic, set
`QIMG21_REPLAY_HIDDEN=hidden.npy`, `QIMG21_STAGE_DIR`, and an explicit
`QIMG21_STAGE_BLOCK` from 0 through 31. This diagnostic requires exactly one
manual-timestep step and no CFG. It validates a finite BF16-valued F32 fixture
of shape `[N,4096]` or `[1,N,4096]`, initializes normal prompt/timestep
conditioning, skips preceding transformer blocks, and runs only the selected
block from the supplied state. A successful replay intentionally exits with
**status 3** and writes only stage captures—no denoiser prediction, scheduler
update, or final latent. Capture/write failures return failure instead.
Never use injected-state outputs as full-model acceptance evidence.

On the saved low-timestep PyTorch block-17 input, with vector normalization,
host-table/vector RMS and Flash-style MMA attention, native modulation, Q and
V projections match exactly. Subsequent matched-block measurements are:

| Stage | Cosine | Elementwise equality |
| --- | ---: | ---: |
| Q after RMS/RoPE | 0.999999999980 | 99.9993% |
| Attention before output projection | 0.999999997024 | 99.9488% |
| Attention after output projection | 0.999999950513 | 97.8368% |
| MLP output | 0.999999778503 | 79.9929% |
| Block output | 0.999999998057 | 88.9708% |

Mixed text/image attention changes these results only slightly (block cosine
0.999999998059). This narrows the remaining local discrepancy to the
RMS/RoPE-attention boundary and its downstream propagation, rather than the
already exact input projections. Artifacts are
`tmp/qimg21-replay-block17` and `tmp/qimg21-replay-block17-mixed`.
CLI guard coverage brings the CPU suite to 26 passing tests.

`QIMG21_REPLAY_ATTENTION=attention.npy` optionally injects a finite,
BF16-valued F32 `[N,4096]` or `[1,N,4096]` tensor before attention's output
projection. It requires the guarded hidden-state replay above and retains
status 3/no-prediction semantics. This isolates the residual/MLP tail without
changing normal execution. For the same low-timestep block 17, injecting
`tmp/qimg21-pytorch-low17/b17_attn_raw.npy` alongside `b17_hidden.npy`
produces **bit-exact** attention output projection, MLP gate/projection,
MLP output, and final block output. The saved `b17_mod_ln2` reference hook is
before modulation, unlike native `mod_ln2`; those boundaries must not be
directly compared. Results: `tmp/qimg21-replay-exact-attention17/results.json`.
This isolates the discrepancy in this block to attention/its inputs; it does
not establish parity for other blocks or the full denoiser.

```sh
QIMG21_REPLAY_HIDDEN=tmp/qimg21-pytorch-low17/b17_hidden.npy \
QIMG21_REPLAY_ATTENTION=tmp/qimg21-pytorch-low17/b17_attn_raw.npy \
QIMG21_STAGE_DIR=tmp/qimg21-replay-exact-attention17 QIMG21_STAGE_BLOCK=17 \
QIMG21_STAGE_KEYS=attn_raw,attn_out,mlp_gate,mlp_proj,mlp_out,block_17 \
cuda/qimg21/test_cuda_qimg21_native --model /mnt/nvme01/models/qimg-21 \
  --prompt-embeds tmp/qimg21-blockref/prompt_embeds.npy \
  --latents tmp/qimg21-blockref/input_001.npy --height-tokens 16 --width-tokens 16 \
  --steps 1 --timestep 0.02001953125 --normalization vector4 \
  --rope host-table-vector4 --attention mma64-flash \
  --out tmp/qimg21-replay-exact-attention17/forbidden.npy
# Expected diagnostic exit status: 3; forbidden.npy must not exist.
OPENBLAS_NUM_THREADS=1 tmp/qimg21-ref-venv/bin/python cuda/qimg21/block_replay_compare.py \
  --reference tmp/qimg21-pytorch-low17 --candidate tmp/qimg21-replay-exact-attention17 \
  --block 17 --require-exact --out tmp/qimg21-replay-exact-attention17/results.json
```

A subsequent exact-Q/K/V block-17 diagnostic reversed the eight 16-element
Q/K contraction fragments while retaining reverse-64 Flash-style softmax.
This worsened aggregate relative L2 from 5.8743e-5 to 1.0927e-4 and elementwise
equality from 99.9712% to 99.8971% (target cosine 0.999999993574).
Artifacts are `tmp/qimg21-exact-attention17/reverse_k.{npy,json,log}`.
The experiment was removed; ordinary ascending contraction order remains.
This rules out simple contraction reversal as a local improvement, not other
accumulation layouts or the unresolved full-denoiser parity failure.

The saved exact-Q/K/V case now also has a real CUDA dispatch trace (not just
CPU operator names). With PyTorch 2.14.0+cu130 on the RTX 5060 Ti:

- Masked text: `fmha_cutlassF_bf16_aligned_64x128_rf_sm80`, grid `[1,32,1]`,
  block `[32,8,1]`.
- Image: `flash_fwd_kernel` with head dimension 128, query tile 128, key tile
  64 and four warps, grid `[2,1,32]`, block `[128,1,1]`.

The profiled output is bit-identical to the saved PyTorch block-17 attention.
Reproduce this diagnostic with a fresh output directory:

```sh
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 tmp/qimg21-ref-venv/bin/python cuda/qimg21/attention_dispatch_probe.py \
  --stage-dir tmp/qimg21-exact-attention17 --prefix-tokens 15 \
  --out-dir tmp/qimg21-dispatch-block17
```

The tool saves `trace.json`, `dispatch.json`, and `attention.npy`; it fails
if no CUDA kernel events were captured. It profiles text-to-image segmentation
only, not arbitrary editing layouts. For head dimension 128, float32 and
double-intermediate construction of the exp2 softmax scale both round to
`0x3e0293ee`, ruling out that constant difference. A separate native replay
with `CUDA_RUNNER_NO_FMAD=1` retains 99.9712% equality and relative L2
5.8744e-5 (versus 5.8743e-5 normally), so disabling scalar FMA is not a useful
fix either. Artifacts: `tmp/qimg21-exact-attention17/no_fmad.{npy,json,log}`.

`attention_tile_regression.py` further isolates intra-tile versus online
softmax discrepancies. It crops saved model Q/K/V to 32, 64, 68, 128, 132 and
256 tokens, rounds inputs to BF16, treats the sequence as one unmasked image,
and compares ordinary reverse-64 and Flash-style native attention against
explicit PyTorch FlashAttention. It is diagnostic, not denoiser acceptance.

```sh
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 tmp/qimg21-ref-venv/bin/python cuda/qimg21/attention_tile_regression.py \
  --stage-dir tmp/qimg21-exact-attention17 --work-dir tmp/qimg21-attention-tile-matrix
```

All 12 comparisons pass the isolated 0.99996 gate, but differences exist
before any cross-tile update. Flash-style equality is 99.9924% at 32 tokens
(relative L2 1.6996e-5) and 99.9756% at 64 tokens (relative L2 5.0615e-5).
At 256 tokens it is 99.9720% (relative L2 6.0026e-5). Therefore the online
rescaling recurrence is not the sole cause; the single-tile QK/softmax/PV
path must also be examined. Results: `tmp/qimg21-attention-tile-matrix/results.json`.

The subsequent **GPU** scale check found what the CPU-only comparison above
missed: on sm_120, `rsqrtf(128)` returns `0x3db504f2`, one ULP below the
host-computed float scale. `mma64_kernels.h` now constructs the inverse square
root in double and casts to float, then computes the exp2 scale with a double
intermediate, matching the reference's host setup. This changes all MMA64
variants; the scalar `math` and original `reverse64` kernels are unchanged.
With the correction, the Flash-style 32/64/68/132-token tile tests are
bit-exact; only one element differs at 128 tokens and three at 256. On saved
exact block-17 Q/K/V, target equality improves from 99.9718% to **99.9997%**
and target relative L2 from 6.0919e-5 to **5.2386e-6**. Artifacts:
`tmp/qimg21-attention-tile-host-scale/results.json` and
`tmp/qimg21-exact-attention17/host_scale.json`.

The 256x512/seed123 low-timestep full-denoiser prediction improves from
0.999948668832 to **0.999951535120**, passing the revised 0.99995 gate. This uses
`--attention mma64-flash --normalization vector4 --rope host-table-vector4`.
Artifact: `tmp/qimg21-host-scale-rect-low`. This isolated checkpoint does not
by itself establish full-model acceptance. The CPU suite remains 35/35 passing.
The 512x512/seed7 final checkpoint likewise improves from 0.999948430099 to
**0.999952904755**, also passing the revised gate; see
`tmp/qimg21-host-scale-512-low`.
Using `mma64-mixed` with the corrected scale on the rectangular checkpoint
instead gives **0.999947326171**, so switching the text branch to the current
forward variant does not close the gap (`tmp/qimg21-host-scale-mixed-rect-low`).

Fresh two-step 256x256 editing validation under the revised gate still fails.
With original-tree normalization and host-table RoPE, corrected-scale
`mma64-flash` predictions are **0.999900373 / 0.999866826**, while ordinary
`mma64` gives **0.999927830 / 0.999862317**. Their trajectories also remain
below 0.99995. Artifacts are `tmp/qimg21-edit-host-scale-flash-tree` and
`tmp/qimg21-edit-host-scale-mma-tree`. Thus text-to-image checkpoints clearing
the revised gate do not establish editing acceptance.

The stronger combination of vector4 LayerNorm, ordinary-tree Q/K RMS and
ordinary `mma64` raises the high-timestep editing prediction to
**0.999957879** (pass), but its low-timestep prediction is **0.999881718** and
trajectory cosines are **0.999942440 / 0.999943021** (fail). Artifact:
`tmp/qimg21-edit-host-scale-mma-vector-ln/results.json`. This is the current
best reverse-traversal two-step editing result under the revised target.

`mma64-forward-flash` is an explicit experimental mode that applies
Flash-style exp2 softmax while traversing 64-key tiles in forward order. On an
exact low-timestep editing block-17 Q/K/V replay, forward traversal reduces
target attention relative L2 from **8.2169e-4** to **4.4080e-4** (cosine
0.999999902850); prefix relative L2 is 4.9129e-4. The reusable
`capture_edit_block.py` captures a selected official transformer block directly
from existing editing fixtures, without rerunning VAE or text/vision encoding.
These injected-state comparisons are diagnostic only.

In the complete two-step editing regression, forward-Flash with vector4
LayerNorm and host-table/original-tree Q/K RMS passes the high prediction
(**0.999963623**) and both scheduler trajectory checkpoints
(**0.999951105 / 0.999950821**). The independently matched low prediction is
only **0.999868168**, so the run still fails overall. Artifacts:
`tmp/qimg21-edit-low-block17-v3` and
`tmp/qimg21-edit-forward-flash-vector-ln/results.json`.

Current upstream FlashAttention source reduces each 64-key tile's probability
sum across its four-lane row group before updating the running denominator. An
experimental native implementation of that ordering made the exact-Q/K/V
block-17 target replay only marginally closer (relative L2
**4.40802e-4 -> 4.40712e-4**, exact fraction **98.0930% -> 98.0945%**), but
regressed the complete two-step editing run: predictions became
**0.999951407 / 0.999846603** and trajectories became
**0.999934909 / 0.999935189**. The kernel change was therefore removed. This
also cautions against assuming the installed PyTorch 2.14 kernel has identical
rounding to the current upstream implementation. Diagnostic artifact:
`tmp/qimg21-edit-forward-flash-tile-reduce/results.json`.

The same low-step block replay with its matched PyTorch hidden state shows
where long-sequence error enters: target Q/K after RMS+RoPE have relative L2
1.15e-5 / 2.15e-6, while ordinary MMA64 attention reaches 8.22e-4 and the
block output reaches 1.96e-3. Replaying the exact PyTorch Q/K/V yields the same
ordinary-attention error (8.2169e-4), proving that the dominant local error is
inside long segmented attention rather than its Q/K inputs.

A bounded depth-switch experiment used forward-Flash for early blocks and
reverse-Flash after blocks 8, 16, or 24. Low-prediction cosines were
0.999862188, 0.999844693, and 0.999870276 respectively, all worse than uniform
reverse MMA64 at 0.999881718. The switch implementation was removed; artifacts
are `tmp/qimg21-edit-hybrid-cut{8,16,24}-low`.

Forward/reverse output blending was also evaluated on exact Q/K/V captures at
blocks 0, 17, and 31. Least-squares forward weights were 0.918, 0.803, and
0.860, and improved local target-attention relative L2 versus forward alone.
A native dual-kernel 0.875 blend improved the full low prediction only from
0.999881718 to **0.999887621**; a 0.5 blend fell to **0.999870922**. This is
far short of 0.99995 while doubling image-attention work and adding an extra
activation buffer, so the blend mode was removed. Artifacts:
`tmp/qimg21-edit-low-block{0,31}`, `tmp/qimg21-edit-blend-low`, and
`tmp/qimg21-edit-blend-half-low`.

### Revised-gate true-CFG editing status

A fresh scale-4 two-step editing run with forward-Flash, vector4 LayerNorm and
host-table/original-tree Q/K RMS still fails: combined predictions are
**0.999795049 / 0.999448447**, and trajectory cosines are
**0.999784429 / 0.999786137**. At the failing low timestep, isolated positive
and negative branches independently pass at **0.999969063** and
**0.999965500**. PyTorch reconstruction from the saved branch tensors using
BF16 `negative + 4 * (positive - negative)` is bit-exact to the saved combined
reference, matching the native operation order. The CFG failure is therefore
error amplification between two slightly different branch residuals, not an
incorrect guidance formula. Artifacts: `tmp/qimg21-edit-cfg-forward-flash-095`,
`tmp/qimg21-edit-cfg-positive-forward-low`, and
`tmp/qimg21-edit-cfg-negative-forward-low`.

Two accuracy fallbacks were checked on the non-CFG low editing checkpoint.
The native scalar/math attention path with vector4 LayerNorm and host-table
RoPE scores **0.999875346**. Forward-Flash with vector4 Q/K RMS scores
**0.999873028**, worse than its original-tree Q/K RMS result. Neither clears
0.99995, so no default-mode change was made. Artifacts are
`tmp/qimg21-edit-math-vector-low` and
`tmp/qimg21-edit-forward-vector-rms-low`.

An explicit non-fused multiply/add RoPE experiment was rejected: matched
block-17 Q/K mismatches rose to 13/11 elements and block-output relative L2
rose to 5.0646e-5. No production RoPE change was retained; diagnostic artifacts
are in `tmp/qimg21-replay-rope-unfused17`.

Replaying attention directly from the saved **PyTorch** block-17 Q/K/V removes
native RMS/RoPE from the comparison. Flash-style MMA attention still differs:
cosine 0.999999998275, relative L2 5.87e-5, elementwise equality 99.9712%
(target equality 99.9718%). Thus RoPE is not the sole source of the local
error. Disabling NVRTC fast math gives bit-identical output on this fixture.
A fused score-scaling variant worsens equality to 99.9668% and was removed.
Inputs, candidates and region metrics are under `tmp/qimg21-exact-attention17`;
these operator results do not establish full-model acceptance.

Backend-forced replay corrected the earlier long-sequence dispatch diagnosis.
For the 256-query target segment with 4,372 keys, PyTorch
`EFFICIENT_ATTENTION` reproduces the captured official `attn_raw` tensor
**bit-exactly** (relative L2 0, exact fraction 1.0). Forced Flash and default
SDPA both give target cosine **0.999999663446**, relative L2 **8.20448e-4**,
and exact fraction **88.4286%**; forced cuDNN gives relative L2
**4.44236e-4** and exact fraction **98.0833%**. Profiling identifies the exact
target kernel as `fmha_cutlassF_bf16_aligned_64x128_rf_sm80`, specifically
PyTorch's `AttentionKernel<bf16, Sm80, aligned, 64 queries, 128 keys, max-K
128>`. Editing parity therefore requires the memory-efficient CUTLASS
accumulation order; matching FlashAttention tile order cannot close this gap.
`attention_probe.py` accepts `--backend efficient` and `--backend cudnn` and
forces only the unmasked target call, leaving masked prefix dispatch
unchanged. Artifacts are in `tmp/qimg21-edit-low-block17-v3`.

The native `mma128-efficient` experiment mirrors the identified 128-key tile
shape using one 69.6-KiB K/V shared-memory buffer. On the same exact block-17
Q/K/V replay it reaches cosine **0.999999991281**, relative L2
**1.32071e-4**, and **99.8283%** BF16 element equality after matching CUTLASS's
pre-scaled score state and per-tile denominator reduction. This is a useful local
advance but not an accepted full-model path: the best three-way dispatch
(forward text, reverse-64 masked image prefix, forward-128 target) scores
predictions **0.999946048 / 0.999853441** and trajectories
**0.999927566 / 0.999928472**. It remains explicit and experimental while the
strict non-quantized gate is 0.99996. Artifact:
`tmp/qimg21-edit-mma128-cutlass-softmax/results.json`.

Compare the replay without loading PyTorch or allocating GPU memory:

```sh
OPENBLAS_NUM_THREADS=1 tmp/qimg21-ref-venv/bin/python cuda/qimg21/attention_replay_compare.py \
  --reference tmp/qimg21-edit-attn0/pytorch_attn_matched_default.npy \
  --candidate tmp/qimg21-edit-attn0/mma_attention.npy --target-tokens 256 \
  --out tmp/qimg21-edit-attn0/mma_comparison.json
```

Both replay comparators require the whole tensor, prefix, and target region
to clear 0.99996, and reject nonfinite or mismatched inputs. Two CPU tests
cover invalid inputs and a failing target hidden inside a passing aggregate
(23 CPU tests pass total). The existing scalar editing kernel's target-only
cosine versus default SDPA is **0.999999561**, compared with aggregate
**0.999999823**; both pass this isolated first-block attention check.

`edit_kernels.h` adds experimental CUDA primitives for that layout:
interleaved text/image scatter, Q/K RMSNorm plus layout-driven three-axis
RoPE, and block-causal attention that keeps adjacent images separate.
They are selected only by the explicit editing-layout path. A synthetic,
checkpoint-free GPU comparison is available:

```sh
make -C cuda/qimg21 test_edit_kernels
OMP_NUM_THREADS=2 tmp/qimg21-ref-venv/bin/python cuda/qimg21/editing_kernel_regression.py
```

The harness checks exact scatter and applies the 0.99996 gate to RoPE and
attention against official Diffusers metadata/RoPE and PyTorch math. Host
build and sm_120 NVRTC compilation pass. The RTX 5060 Ti synthetic GPU run
passed with bit-exact scatter, Q/K RoPE, and attention versus the reference.
This synthetic test does not establish model editing
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
make -C cuda/qimg21 native-text-exact
tmp/qimg21-ref-venv/bin/python cuda/qimg21/native_text.py \
  --model /mnt/nvme01/models/qimg-21 --prompt "a red apple on a white table"
```

It gathers only requested embedding rows, streams one BF16 weight matrix at
a time, and implements all 36 text blocks with native RMSNorm, split-half
text RoPE, causal grouped-query attention, SwiGLU, and BF16 residual updates.
It omits the final RMSNorm and LM head. The C executable requires unpadded,
batch-one integer token IDs and limits inputs to 4096 tokens. Vision tokens
require the explicitly supplied merger, deep-stack, and MRoPE fixtures
described below. Python handles only processor/tokenization and system-prefix
cropping; `--prepare-only` runs that stage without CUDA or model weights.
The original checkpoint configuration is required; this is not a general
Qwen3-VL loader. The validated default uses the pinned FlashAttention forward
specialization selected by PyTorch on sm_120, an exported exact BF16 text-RoPE
table, BF16-output GEMMs, and PyTorch-matched RMS reduction topologies. The
positive English prompt and empty negative prompt are bit-exact at all 36
saved layer boundaries. Full pre-norm and cropped prompt embeddings have zero
relative L2 and MAE; their reported cosines are `1.0` (the empty cropped
branch prints `0.9999999999999998` from the F64 metric calculation). All four
comparisons pass the `0.99996` gate. `--native-attention custom` and
`cutlass-efficient` remain diagnostic alternatives; `flash-exact` is the
acceptance default. `--f32-gemm-output` likewise retains the older diagnostic
GEMM path.
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
Text-stage reference captures include `layer_NN.npy`. Use
`--text-stage-layer N` on the PyTorch fixture and `--dump-layer N` with
`--dump-dir DIR` on the native executable to expose matching normalization,
Q/K/V projection, RoPE, attention and MLP boundaries for any layer. This
diagnosis found that formula-based fast RoPE differed in 36 Q and 2 K values,
the generic 256-thread head RMS reduction first differed at layer 3, and the
short empty prompt selects a vector-four wide RMS reduction at layer 34.
The exact table and matched reduction paths eliminate those differences.

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

Editing remains an experimental fixture-driven native path: full-model parity
and the native VAE/image-conditioned text hand-off still need to be accepted
against the same checkpoint gate. Row-INT8 and W8A8 editing, true-CFG, and
40-step generation use the calibrated MRE gates and are validated above; that
quantized denoiser evidence does not waive the original-weight conditioning
handoff gate.

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

For a text-to-image smoke image, the orchestration script uses the native
Hugging Face JSON tokenizer and validated CUDA text encoder, runs all
denoising steps in the native executable, then
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

The native tokenizer reproduces the processor's Qwen ChatML template and
system-prefix crop. For `a red apple on a white table`, all 29 pre-crop token
IDs match the official processor exactly and the 15-token native embedding is
bit-identical to the pinned PyTorch reference (cosine 1.0, maximum absolute
error 0). The empty negative prompt likewise matches all 23 token IDs and its
9-token embedding bit-for-bit. A CPU-only token check is available without
loading model weights:

```sh
cuda/qimg21/test_cuda_qimg21_text \
  --model /mnt/nvme01/models/qimg-21 \
  --prompt "a red apple on a white table" \
  --dump-tokens tmp/qimg21-native-tokens.txt
diff -u tmp/qimg21-token-ref/tokens.txt tmp/qimg21-native-tokens.txt
```

The native Qwen3-VL vision front end is available as
`test_cuda_qimg21_vision`. It consumes either the processor's flattened F32
patch matrix or an RGB-convertible image whose sides are divisible by 32.
The `--image` path normalizes pixels to [-1,1] and emits the exact merged-patch
ordering natively. Its default `--patch-projection cudnn-engine23` recreates
the pinned PyTorch cuDNN Conv3d plan (engine 23, tile 7, four stages and
split-K 3); `cublas` retains the earlier diagnostic path. The custom CUDA
epilogue and interpolated learned 48x48 position table remain native. At full
1024x1024 conditioning resolution, all 4,718,592 patch-projection values and
the complete 27-block output are bit-identical to the official capture.
The executable also contains the 27 custom CUDA vision blocks, three
deep-stack mergers, and the final merger. Its acceptance default is the
pinned FlashAttention forward specialization selected by PyTorch Flash SDPA
for BF16 head dimension 72; `--attention cutlass` and `math` retain diagnostic
alternatives. Vision linears use BF16-output cuBLAS-LT bias epilogues, including
the PyTorch-selected algo 21 tile/stage and split-K configurations. The
full-resolution FC2 and merger shapes use the captured split-K 6 and 5 plans,
respectively. LayerNorm defaults to the literal vector-four,
four-warp topology used by pinned ATen. With the exact attention scale,
teacher-forced block 0 is bit-identical. Vision RoPE uses a checked CUDA-generated
cosine/sine artifact and separate multiply/add operations matching PyTorch;
correcting its table, the FlashAttention scale, and ATen's exact Welford
expression structure makes every one of the 27 recurrent block outputs
elementwise bit-identical. On the 4096-token full-resolution capture, the three
deep-stack merger cosines are 0.999999994689, 0.999999997178, and
0.999999997502; the final merger cosine is 0.999999999999. All vision
checkpoints therefore pass the 0.99996 gate. The
accepted path is wired into the
experimental editing integration to exercise the complete native path. Use `--hidden ... --block-index N
--max-blocks 1` for isolated block replay, or `--max-blocks 0 --merged-out ...`
for merger-only replay, and `--dump-dir DIR` to save every executed block.
`--layer-norm nvrtc` retains the in-module Welford implementation for
diagnosis while `nvcc` retains the earlier eight-warp implementation.
`nvcc-pytorch` is the default and selects the literal `dim3(32,4)`
128-thread/four-warp launch used by pinned ATen: its block-0 LayerNorm rstd is
bit-exact and the complete recurrent block stack is bit-exact. With the earlier FlashAttention path,
the all-NVRTC run improves final block cosine from 0.999461003 to 0.999620834
but still fails the gate. A greedy per-block oracle over both implementations
first failed at block 5 (0.999954165); this historical result preceded the
exact Welford expression-order fix.

`--dump-dir` also records `norm1_stats.npy` and `norm2_stats.npy` as
`[tokens,2]` mean/rstd pairs when the pinned FlashAttention plugin is active.
`text_capture.py --dump-text-stages` records the corresponding PyTorch
`vision_stage_norm{1,2}_{mean,rstd}.npy` arrays. On exact block-0 input, native
means are bit-identical for 130/256 rows and differ by at most 7.45e-9;
rstd is identical for 203/256 rows and differs by at most 2.38e-7. The
initial four-warp clone made rstd bit-exact for all rows and means exact for
157/256 rows. These F32 reduction differences explained the sparse LayerNorm
BF16 discrepancies.
With the corrected attention scale, block 0 is bit-exact and therefore gives
block 1 an exact input.  Direct BF16 ATen replay of block-1 Norm1 differs from
the four-warp native output in only 4 of 294,912 values (maximum 0.001953125).
Injecting that exact Norm1 raises the isolated block-1 cosine from
0.999999321935 to 0.999999999602 and leaves only 21 differing output values,
confirming that sparse LayerNorm rounding was the next recurrence source.
On the matched post-attention state, block-1 Norm2 differs in only 1 of
294,912 values (maximum 0.000244140625). Injecting both exact normalization
outputs makes the complete block bit-identical. The native diagnostic accepts
`--norm2-override` alongside `--norm1-override` for this boundary proof.
The final fix preserves ATen's source-level online/combine expression structure;
the compiler otherwise schedules the post-shuffle mean and variance dependency
chains differently despite mathematically equivalent C++. Testing unfused
online/combine arithmetic, native BF16 conversion, PyTorch's
2D thread indexing, O2 compilation, and CUDA 12.9 code generation did not improve
the full recurrence, so the accepted kernel retains the literal pinned
PyTorch Welford expressions. Recompiling with the wheel's recorded CUDA 13.0
flags (`-O2`, C++20, extended lambdas, and the half/BF16 conversion-disable
macros) left the same four block-1 Norm1 output differences before the
expression-order fix.
`--layer-norm nvcc-pytorch-bf16` additionally casts hidden states to actual
BF16 storage and runs BF16-vector input/weight/bias kernels. It produces the
same four differences and is retained only as a diagnostic; the default avoids
its extra cast and buffer.

For boundary isolation, `--norm1-override NORM.npy` replaces the first
executed block's computed normalization output after still running the native
kernel, and `--flash-plugin PATH` selects an alternate ABI-compatible pinned
attention build. Replaying block 0 with the captured PyTorch normalization
makes all 884,736 QKV values and both rotary Q/K tensors bit-exact. Before the
scale correction, the first remaining difference was the FlashAttention output:
56 of 294,912 BF16 values differed by one representable step (maximum absolute
error 0.0009765625).
Official FlashAttention `--use_fast_math`, fused softmax FMA, and CUDA 12.9
build variants do not improve that result; the default unfused CUDA 13.1 build
has the fewest mismatches.
Forced PyTorch backend replay confirms that default and
`SDPBackend.FLASH_ATTENTION` are both bit-exact with the captured tensor;
efficient attention, cuDNN, and math are not.  Profiling identifies the exact
`Flash_fwd_kernel_traits<96,128,64,4>` specialization, with a 2x1x16 grid,
128 threads, 49,152 dynamic shared-memory bytes, and the same eight boolean
template arguments as the native plugin.  A locally extracted NVCC 13.0.88
toolchain produces the same 56 mismatches as NVCC 12.9 and 13.1.  Rebuilding
with the upstream PyTorch flags (`--use_fast_math`, normal FMA contraction,
half/BF16 macro overrides, and extended lambdas) produces 61 mismatches, so
neither backend dispatch, launch geometry, nor CUDA compiler minor version
explains the residual difference.  The final cause was the wrapper's
unqualified host `sqrt` expression and separately compiled kernel arithmetic.
PyTorch's BF16 output is stable for adjacent scale bits `0x3df15bef` and
`0x3df15bf0`, while the native specialization requires the upper value to
reproduce it.  Using that checked constant makes all 294,912 block-0
attention values bit-identical on matched Q/K/V.

```sh
make -C cuda/qimg21 test_cuda_qimg21_vision
cuda/qimg21/test_cuda_qimg21_vision \
  --model /mnt/nvme01/models/qimg-21 \
  --pixel-values tmp/qimg21-vision-boundaries2/text_positive/pixel_values.npy \
  --grid-height 16 --grid-width 16 \
  --patch-out tmp/qimg21-native-vision-patch.npy \
  --out tmp/qimg21-native-vision-block-input.npy
```

The work directory contains `prompt/prompt_embeds.npy`, the deterministic
initial `latents.npy`, one `steps/step_XXX.npy` file per Euler update, and the
final `native_latents.npy`. These arrays are the hand-off points for comparing
the native transformer/scheduler against the PyTorch reference.

The native text executable can now consume multimodal fixtures directly:
`--vision-merged` replaces image-pad token embeddings, and
`--vision-deepstack-dir` adds the three visual features after language layers
0, 1, and 2. `--rope-table` supplies the corresponding MRoPE cosine/sine table.
The visual feature upload is explicitly synchronized with the CUDA execution
stream; without that dependency, successive layers could consume overwritten
deep-stack storage. PyTorch's generic reduction selects different contiguous
vector widths at sensitive input and post-attention RMSNorm boundaries; the
native exact default mirrors those layer-specific topologies and retains
`--rms`/`--post-rms` overrides for diagnosis. On the captured
96-token/16x16-patch case, all 36 language layers, all three deep-stack
additions, and the cropped prompt embedding are bit-exact: cosine
0.9999999999999998, zero MAE, and 100% elementwise equality. The accepted
29-token text-only path remains bit-exact as well. Multimodal `--prompt` now
constructs the image-aware ChatML sequence itself from the native tokenizer;
`--image-grid-height/--image-grid-width` generate the 3D MRoPE layout by
recomposing rows from the checked exact text-RoPE artifact, without Python or
runtime trigonometry. For the same editing fixture, all 96 token IDs and the
entire MRoPE table are bit-exact, and the resulting prompt embedding retains
cosine 0.9999999999999998, zero MAE, and 100% equality.

```sh
cuda/qimg21/test_cuda_qimg21_text \
  --model /mnt/nvme01/models/qimg-21 --prompt "make the apple blue" \
  --vision-merged tmp/qimg21-vision-boundaries2/text_positive/vision_merger.npy \
  --vision-deepstack-dir tmp/qimg21-vision-boundaries2/text_positive \
  --image-grid-height 16 --image-grid-width 16 --attention flash-exact \
  --out tmp/qimg21-native-multimodal-text/native-full-exact.npy
```

Add `--native-vae` to use the native F32 CUDA decoder. It reads the original
VAE safetensors, applies latent denormalization and the learned post-quant
convolution, then runs the residual/attention/upsampling graph and clamps the
RGBA result. Text-only and image-editing prompt conditioning now use the native
text and vision executables. This decoder supports
single images; editing parity and peak-memory validation remain work in progress.
The separate single-frame encoder is described below. Decoder kernels and residual copies share the default CUDA stream
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

## Native single-frame VAE encoder

`test_cuda_qimg21_vae_encode` implements the original-weight F32 encoder:
input convolution, five residual down blocks with first-frame averaging
shortcuts, mid-block attention/residuals, output normalization/convolution,
and the learned quant convolution. It reuses the validated decoder's native
convolution, RMS normalization and attention kernels. First-chunk temporal
padding is handled explicitly; video/temporal continuation is not supported.

```sh
TMPDIR="$PWD/tmp" make -C cuda/qimg21 native-vae
cuda/qimg21/test_cuda_qimg21_vae_encode \
  --model /mnt/nvme01/models/qimg-21/vae --image normalized_rgba.npy --out moments.npy \
  --normalized-latents condition_tokens.npy
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 tmp/qimg21-ref-venv/bin/python cuda/qimg21/vae_encoder_regression.py \
  --model /mnt/nvme01/models/qimg-21/vae --case 128x128:42 --case 64x128:123 \
  --case 256x256:7 --work-dir tmp/qimg21-encoder-matrix
```

Input may be finite F32 `[4,H,W]` RGBA already normalized to `[-1,1]`, or an
image file supplied with `--input-image` and `--resolution`. The native image
path decodes with stb, reproduces Pillow's premultiplied-alpha fixed-point
Lanczos resize and Diffusers normalization, and can emit `--resized-out` plus
`--preprocessed-out`. For the 1024-to-256 editing fixture, the native F32
tensor and resulting normalized VAE latents are bit-identical to the former
Python/Pillow path (maximum error 0). `--preprocess-only` needs no model or GPU.
Dimensions are divisible by 16 and at most 1024. Output is raw F32 posterior
parameters `[128,H/16,W/16]`: mean channels first, then log-variance channels.
It does not sample the posterior or clamp log variance. The optional
`--normalized-latents` output selects the deterministic
posterior mean, applies `(mean - latents_mean) / latents_std`, and packs F32
tokens as `[H/16 * W/16,64]`. Encoder and decoder share the original model's
mean/std constants. This token file has the native denoiser's condition-latent
format and can also be read directly by the native decoder; raw 128-channel
posterior moments cannot. Image-conditioned text embeddings and matching
joint-layout metadata are still required for an editing request.

Against official PyTorch F32 encoding with TF32 disabled, all four synthetic
RGBA cases pass: 64x64/seed17, 128x128/seed42, 64x128/seed123, and
256x256/seed7. Minimum posterior cosine is **0.999999999998848**; minimum
latent-mean cosine is **0.999999999989884**, and all outputs are finite. Both
cosines are gated independently at 0.99996. Maximum absolute error is
0.00044835 on the 256x256 case. Results are under `tmp/qimg21-encoder-first`
and `tmp/qimg21-encoder-matrix`. A decoder smoke test after the shared-helper
refactor also passes at cosine 0.999999999347549 (128x128/seed17).

CPU input guards reject wrong channel counts, unsupported dimensions,
non-finite pixels and out-of-range pixels before CUDA initialization; the
CPU suite is now 35/35 passing. Deterministic BF16 editing and true-CFG
denoiser/scheduler parity is verified in the exact-backend section below;
the process-isolated 1024-square pipeline memory measurement follows.
The encoder downsamplers use a native stride-2 CUDA convolution. It evaluates
only the retained odd-position outputs and no longer allocates a full-resolution
convolution result before sampling. The optimized path is bit-identical to the
previous native output at both 256x256 and 1024x1024. At 1024x1024 on the RTX
5060 Ti, the complete encoder subprocess takes 7.74 seconds (700,320 KiB host
maximum RSS); posterior cosine against the pinned PyTorch reference is
0.999999999999038 with MAE 9.41e-6 and maximum error 4.80e-4.

The updated encoder regression independently gates normalized token cosine
against PyTorch normalization and the official pipeline packing function.
128x128/seed42, 64x128/seed123 and 256x256/seed7 all pass, minimum cosine
**0.999999999990019**, maximum relative L2 4.47e-6. Artifacts are in
`tmp/qimg21-encoder-handoff`. A direct native encoder-token -> native decoder
round trip at 128x128/seed42 matches official PyTorch posterior-mode
reconstruction at cosine **0.999999999429927** (relative L2 3.38e-5).
The old decoder 128x128/seed17 output remains bit-identical after sharing the
statistics helper. These are F32 VAE checks, not BF16 editing-pipeline parity.

The additional 512x512/seed42 encoder regression also passes: posterior cosine
0.999999999998892, mean cosine 0.999999999985253, normalized-token cosine
0.999999999987389 (relative L2 5.03e-6). Artifacts:
`tmp/qimg21-encoder-512`.

Full-resolution 1024x1024/seed42 encoding also passes against the unchanged
official F32 encoder (no cache-discard workaround or tiling): posterior cosine
**0.9999999999990085**, mean cosine **0.999999999994387**, normalized-token
cosine **0.9999999999953705**, token relative L2 **3.0388e-6**. All outputs are
finite. Reproduce with:

```sh
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 tmp/qimg21-ref-venv/bin/python cuda/qimg21/vae_encoder_regression.py \
  --model /mnt/nvme01/models/qimg-21/vae --case 1024x1024:42 \
  --work-dir tmp/qimg21-encoder-1024
```

A native-only repeat was bit-identical for both posterior moments and
normalized tokens, taking **9.28 seconds** wall time with **700424 KiB** maximum
host RSS and zero swaps. An already-running `nvidia-smi` 100 ms process-memory
sampler observed a maximum **1860 MiB** for that native encoder PID across
90 samples. This is a sampled process-memory maximum, not a hard allocation
bound or evidence for the complete pipeline's 12 GB requirement. The original
comparison's sampler started late; only the fully covered native repeat is
used for this memory figure. Artifacts: `tmp/qimg21-encoder-1024/results.json`,
`tmp/qimg21-encoder-1024-repeat.log`, and
`tmp/qimg21-encoder-1024-memory.csv` (native-repeat PID 1304832).

### Experimental single-image editing integration

`native_generate.py --image IMAGE --condition-resolution 256 --native-vae`
now connects native CPU RGBA decoding/Pillow-compatible preprocessing, native F32 VAE encoding and
normalization, native image patchification and vision encoding, native ChatML
tokenization/MRoPE and text encoding, checked native joint layouts, native
denoising, and native decoding. Negative prompts get their own native tokens,
embeddings, masks, and layout.
Only one condition image is supported; resized sides must be within 32..1024.
Use a fresh work directory for each editing request.

```sh
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 tmp/qimg21-ref-venv/bin/python cuda/qimg21/native_generate.py \
  --model /mnt/nvme01/models/qimg-21 \
  --image tmp/qimg21-native-1024-40-benchmark.png --condition-resolution 256 \
  --prompt 'change the apple to green' --height 128 --width 128 --steps 2 --seed 42 \
  --native-vae --native-attention mma64 --native-normalization vector4 --native-rope host-table \
  --work-dir tmp/qimg21-native-edit-smoke-v2 --out tmp/qimg21-native-edit-smoke-v2.png
```

The original two-step 128x128 smoke test completed with finite condition
tokens, both Euler checkpoints, final latents and decoded RGBA pixels. A newer
one-step 256x256 run exercised native VAE encode, all 27 vision blocks, native
multimodal text, denoising, and native VAE decode and produced finite
`(82,4096)` prompt embeddings, `(64,4096)` merged vision features, and
`(256,64)` output latents plus a 256x256 PNG.

Full-resolution conditioning exposed additional parity boundaries. The
native image loader now matches the official processor's integer RGBA-over-white
composition and F32 normalization exactly; `--pixels-out` dumps this boundary,
and all 6,291,456 values match the official 1024x1024 capture bit-for-bit.
The cuDNN patch plan and long-shape cuBLASLt dispatch now make the patch and
all 27 recurrent vision blocks bit-exact. The merger LayerNorm and first GEMM
were already exact, but CUDA 13.1's exact-GELU code generation differed from
the CUDA 13.0 PyTorch wheel at sparse negative inputs. `libq21_gelu.so` is
therefore built with the wheel-matched compiler at `PYTORCH_CUDA_HOME`
(default `tmp/cuda130`); all three deep-stack mergers and the final merger are
now bit-exact through GELU and FC2. The 1058-token text path uses the literal
contiguous ATen F32 mean-reduction topology for both wide RMSNorm boundaries:
16 independent warps per CTA, four vector-lane accumulators, and one output
row per warp. With the standard CUDA 13.1 FlashAttention plugin, the resulting
full-resolution cropped prompt embedding is bit-identical to the pinned
PyTorch capture both with reference vision features and with the native vision
encoder (cosine 1.0, zero relative L2 and MAE, 100% exact elements). The
29-token positive and 23-token negative branches also remain bit-identical.

### Exact BF16 editing parity

The strict non-quantized comparison pins PyTorch SDPA to its efficient
backend. CUDA's automatic SDPA choice mixes memory-efficient attention for
masked text segments with Flash Attention for unmasked image segments; the
last-bit choice is hardware dependent and is therefore not a reproducible
kernel oracle. The native path uses the exact PyTorch 2.14 CUTLASS
memory-efficient specialization plus PyTorch's CPU-generated RoPE table.

Install the fully pinned Python reference environment and source checkouts,
then build the exact native components:

```sh
make -C cuda/qimg21 setup
make -C cuda/qimg21 setup-cuda130
make -C cuda/qimg21 setup-exact-sources
make -C cuda/qimg21 native-exact
```

`setup-cuda130` downloads NVIDIA's five redistributable compiler components,
checks pinned SHA-256 digests, and assembles CUDA 13.0.88 under
`tmp/cuda130` without modifying the system toolkit. This wheel-matched NVCC is
used only for exact GELU code generation; set `PYTORCH_CUDA_HOME` to install
or use the pinned toolkit elsewhere. Once the five verified archives are in
`tmp/cuda130-download`, a clean machine can require cache-only installation:

```sh
cuda/qimg21/setup_cuda130.sh --offline
```

Use `--cache-dir DIR` and `--dest DIR` to stage or consume a portable cache.
Downloads use a `.part` file and are renamed only after completion; every
online or offline install verifies the pinned SHA-256 digest before unpacking.

`setup_exact_sources.sh` pins PyTorch at
`08187d9e0fba026dc8217405802ab5381dc88d90`, FlashAttention at
`14c377950125c70b7a9dabf9c561fca53715ac7d`, and its CUTLASS checkout at
`e05f953a5b3d38adc240df2ff928e0421c2abba3` (the SM120-capable revision used
for validation). The plugin build verifies all three revisions before
compiling. Override `PYTORCH_SOURCE` or `FLASH_ATTN_SOURCE` when keeping the
checkouts outside the repository-local `tmp/` directory.

The generated transformer, text, and vision RoPE tables are also pinned byte-for-byte:

```sh
make -C cuda/qimg21 verify-generated-artifacts
```

Capture and compare a deterministic two-step editing case:

```sh
tmp/qimg21-ref-venv/bin/python cuda/qimg21/reference.py \
  --model /mnt/nvme01/models/qimg-21 --image input.png \
  --prompt 'change the apple color to green' --height 256 --width 256 \
  --steps 2 --seed 42 --sdpa-backend efficient --dump-initial-latents \
  --dump-pred-dir tmp/qimg21-edit-reference --dump-dir tmp/qimg21-edit-reference

tmp/qimg21-ref-venv/bin/python cuda/qimg21/editing_regression.py \
  --model /mnt/nvme01/models/qimg-21 \
  --reference-dir tmp/qimg21-edit-reference \
  --work-dir tmp/qimg21-edit-native \
  --native-attention cutlass-efficient --native-normalization vector4 \
  --native-rope host-table-exact
```

On the RTX 5060 Ti, the measured prediction cosines were
`0.9999999818465785` and `0.9999999912661135`; the matched two-step Euler
trajectory cosines were `0.999998675096587` and `0.9999982127383727`. All
comparisons clear the `0.99996` non-quantized gate. The CUTLASS plugin retains
and grows one output/LSE workspace per process; a 4,352-token editing step now
uses four setup allocations across all 60 attention calls (instead of 120
per-call allocations), and releases them before the CUDA context is closed.
Generated tables, libraries, and fixtures remain ignored.

The same two-step case with negative prompt `red apple, blurry` and true-CFG
scale 4 also passes: prediction cosines `0.9999998485751089` and
`0.9999994896000577`, with trajectory cosines `0.9999985850203311` and
`0.9999975521106901`.

The full single-image native path is covered by
`native_edit_e2e_regression.py`.  Generate the PyTorch trajectory with an
explicit pre-final-RMSNorm prompt fixture, then run the native image loader,
BF16 VAE encoder, vision encoder, multimodal text encoder, denoiser, and VAE
decoder in one regression:

```sh
tmp/qimg21-ref-venv/bin/python cuda/qimg21/reference.py \
  --model /mnt/nvme01/models/qimg-21 --image input.png \
  --prompt 'change the apple color to green' \
  --prompt-fixture-dir TEXT_CAPTURE --height 256 --width 256 --steps 2 \
  --seed 42 --sdpa-backend efficient --dump-initial-latents \
  --dump-pred-dir REFERENCE --dump-dir REFERENCE

tmp/qimg21-ref-venv/bin/python cuda/qimg21/native_edit_e2e_regression.py \
  --model /mnt/nvme01/models/qimg-21 --image input.png \
  --reference-dir REFERENCE --text-reference-dir TEXT_CAPTURE \
  --work-dir NATIVE_WORK
```

The RTX 5060 Ti acceptance run was exact at the condition-latent,
multimodal-prompt, and initial-noise boundaries.  Two-step trajectory cosines
were **0.9999986390 / 0.9999981952**, and native decoded RGBA cosine was
**0.9999974887**.  Every boundary clears the `0.99996` non-quantized gate.
The explicit prompt fixture avoids Transformers 5 output-capturing aliasing:
the fixture itself is independently captured from the input of the final text
RMSNorm, which is the conditioning boundary consumed by Qwen-Image 2.1.

The deterministic text-to-image matrix also passes all matched predictions
and free-running checkpoints:

| Case | Minimum prediction cosine | Minimum trajectory cosine |
| --- | ---: | ---: |
| 256x256, 2 steps, seed 7 | 1.000000000 | 0.999998204 |
| 256x512, 2 steps, seed 123 | 0.999997464 | 0.999994666 |
| 512x512, 4 steps, seed 42 | 0.999996859 | 0.999990533 |

Run this matrix with `regression.py --native --reference-sdpa-backend
efficient --native-attention cutlass-efficient --native-normalization vector4
--native-rope host-table-exact` and the desired repeatable `--case` values.

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
GPU snapshot showed 6,019 MiB used; this was not a peak measurement.

A subsequent one-step 1024x1024 run sampled per-process GPU memory every 50 ms,
with the sampler active before orchestration launch. Observed maxima were:

| Stage (separate process) | Samples | Maximum MiB |
| --- | ---: | ---: |
| Python text/vision helper | 238 | 1,422 |
| PyTorch noise-fixture helper | 12 | 158 |
| Native BF16 denoiser | 1,262 | 1,948 |
| Native F32 decoder | 908 | 3,112 |

No pipeline stages overlap by design, so the largest observed stage is
**3,112 MiB**, comfortably under the requested ~12 GB limit. The complete run
finished in **2m 17.24s**, with finite `[4096,64]` step/final latents, finite
`[4,1024,1024]` decoded pixels, and a 1024x1024 RGBA output. The native-only
denoiser was also measured separately from pre-launch through exit: 1,343
samples, **1,948 MiB** maximum, 69.78 seconds, 14,419,764 KiB host RSS and zero
swaps. Artifacts are `tmp/qimg21-native-1024-memory{.csv,-run}` and
`tmp/qimg21-native-1024-pipeline-memory.csv` plus
`tmp/qimg21-native-1024-memory-pipeline`. These are dense sampled observations,
not an allocator-enforced hard cap; the earlier successful 40-step run provides
the long-run stability evidence.

All 40 saved `[4096,64]` latent checkpoints are finite. The decoded
`[4,1024,1024]` tensor is finite and within `[-1,1]`; the saved image is a
1024x1024 RGBA PNG. Visual inspection shows a coherent red apple on a white
surface, consistent with the prompt. Fixtures, image, and timing log remain
under ignored `tmp/`, not in Git.

## Exact 1024x1024/40-step PyTorch parity

The corresponding full-resolution strict parity run now passes against the
pinned efficient-SDPA PyTorch reference. It uses exact CUTLASS attention,
vector-four normalization and the exact host-generated RoPE table:

```sh
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 /usr/bin/time -v \
  tmp/qimg21-ref-venv/bin/python cuda/qimg21/regression.py \
  --native --model /mnt/nvme01/models/qimg-21 \
  --reference-sdpa-backend efficient \
  --native-attention cutlass-efficient --native-normalization vector4 \
  --native-rope host-table-exact --cosine-threshold 0.99996 \
  --case 1024x1024:40:42 --work-dir tmp/qimg21-exact-1024-40
```

All 40 matched denoiser predictions pass: minimum cosine is
**0.999971784** and maximum relative L2 is **0.0075120298**. All 40
free-running Euler checkpoints also pass: minimum cosine is **0.999989908**
and maximum relative L2 is **0.0045665863**. The complete reference,
40 independent matched native invocations, native trajectory and comparisons
finished in **28m 32.15s**, with zero swaps.

Decoding the PyTorch and native final latents through the same validated native
F32 VAE gives RGB cosine **0.9999990451**, RGB relative L2 **0.0015630461**,
MAE **0.000802789**, PSNR **58.16 dB**, and alpha MAE
**0.0000116778**. Both images are finite and visual inspection shows no
obvious difference. Reproduce the paired decode with:

```sh
OMP_NUM_THREADS=2 tmp/qimg21-ref-venv/bin/python cuda/qimg21/quant_quality.py \
  --model /mnt/nvme01/models/qimg-21 --height 1024 --width 1024 \
  --reference-latents tmp/qimg21-exact-1024-40/1024x1024-s40-seed42/reference/step_039.npy \
  --quantized-latents tmp/qimg21-exact-1024-40/1024x1024-s40-seed42/native/trajectory/final_latents.npy \
  --out-dir tmp/qimg21-exact-1024-40-image-quality
```

The acceptance invocation was not sampled for a new GPU-memory peak. The
same native 1024x1024 denoiser executable and streamed-weight layout was
previously sampled at **1,948 MiB**; the shared native F32 decoder peaked at
**3,112 MiB**, as recorded above. These are observed process peaks, not an
allocator-enforced cap. Fixtures, decoded images and logs remain ignored.
