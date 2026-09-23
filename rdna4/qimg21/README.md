# Qwen Image 2.1 HIP runner

This directory contains the RDNA4 native Qwen Image 2.1 denoiser. It is a
separate runner from `rdna4/qimg`: Qwen Image 2.1 uses 32 transformer blocks,
hidden size 4096, 32 attention heads, head dimension 128, and BF16 weights.

The runner keeps the validated Qwen 2.1 host graph and replaces its CUDA driver
and cuBLAS boundary with the project-local dynamic HIP/HIPRTC layer. GEMMs use
the gfx1201 BF16 WMMA path when available, with a HIPRTC F32-accumulate BF16
fallback. The checkpoint is streamed one transformer block at a time, so the
16-GiB RX 9070 XT does not need a second full model copy.

Build:

```sh
make -C rdna4/qimg21
```

The executable consumes the same deterministic fixture ABI as the CUDA 2.1
runner:

```sh
rdna4/qimg21/test_hip_qimg21_native \
  --model /mnt/disk2/models/qimg-21 \
  --prompt-embeds prompt_embeds.npy --latents latents.npy \
  --height-tokens 16 --width-tokens 16 --steps 2 \
  --attention wmma-fused --out native_latents.npy
```

The RDNA4 runner uses WMMA GEMMs by default when the GPU supports them.
`--attention wmma-fused` adds the shared Pixal3D/TRELLIS gfx12 WMMA attention
kernel for image queries and a separate causal text-prefix correction. Editing
is supported experimentally by processing image groups in reverse order and
correcting interleaved causal text rows; the editing default remains scalar.
`--attention wmma` and `--attention math` use WMMA GEMMs with scalar attention.
The legacy `mma64*`, `mma128-efficient`, and `cutlass-efficient` attention
names are rejected on RDNA4 because they select CUDA-specific kernel source;
use `math`, `reverse64`, `wmma`, or `wmma-fused` instead. The text and vision
helpers likewise reject their CUDA-only `cutlass`/`flash` modes.
The runner covers the 32-layer
denoiser ABI, CFG inputs, editing layout inputs, stage dumps, and FlowMatch
latent updates. Matching native text, vision, VAE decoder, and VAE encoder
executables are built by the same Makefile. ROCm web generation uses the
native VAE path. A separate PyTorch reference mode is available when a ROCm
PyTorch environment with Qwen Image 2.1 Diffusers support is configured.
Health reports each native component separately.

Component smoke checks for the ROCm 10 RX 9070 XT are:

```sh
make -C rdna4/qimg21 all
ROCEW_ROCM_LIB=/opt/rocm/core-10.0/lib rdna4/qimg21/test_hip_qimg21_wmma
ROCEW_ROCM_LIB=/opt/rocm/core-10.0/lib rdna4/qimg21/test_hip_qimg21_text \
  --model /mnt/disk2/models/qimg-21 --prompt 'a red apple' --max-layers 1 --out text.npy
ROCEW_ROCM_LIB=/opt/rocm/core-10.0/lib rdna4/qimg21/test_hip_qimg21_vision \
  --model /mnt/disk2/models/qimg-21 --pixel-values PATCHES.npy \
  --grid-height 64 --grid-width 64 --max-blocks 1 --out vision.npy
ROCEW_ROCM_LIB=/opt/rocm/core-10.0/lib rdna4/qimg21/test_hip_qimg21_vae \
  --model /mnt/disk2/models/qimg-21/vae --latents LATENTS.npy \
  --height-tokens 64 --width-tokens 64 --out image.npy --quiet
ROCEW_ROCM_LIB=/opt/rocm/core-10.0/lib rdna4/qimg21/test_hip_qimg21_vae_encode \
  --model /mnt/disk2/models/qimg-21/vae --input-image IMAGE.png \
  --resolution 256 --out moments.npy --normalized-latents latents.npy
```

The WMMA test reports `incorrect outputs=0`; the text and vision checks
produce finite `[tokens,4096]` and `[4096,1152]` outputs, the decoder produces
finite `[4,1024,1024]` output, and the encoder produces finite `[128,16,16]`
moments plus `[256,64]` normalized latents.

For standalone generation with only NumPy and Pillow installed, run
`python3 cuda/qimg21/native_generate.py --backend rocm --model MODEL_ROOT`.
The ROCm path defaults to native VAE decode and uses a deterministic NumPy
latent seed. CUDA keeps its PyTorch CUDA RNG seed path, so identical seed
numbers across the two backends do not produce identical initial noise.
Both text-to-image and image-editing paths passed initial 256x256 one-step
smoke tests. The scalar path's matched low-timestep 256x256 denoiser prediction
has cosine about 0.99995 against the saved PyTorch reference, below the CUDA
runner's 0.99996 regression gate.

On the RX 9070 XT with ROCm 10.0.0 installed at `/opt/rocm/core-10.0`
(the HIPRTC component reports version 9.0), a matched-input
256x256 two-step denoiser run takes 6.76 seconds (14,102,172 KiB peak host RSS) and
is byte-identical to the earlier ROCm implementation. The earlier run took
30.26 seconds: direct H2D upload from the safetensors mmap eliminates a
redundant host copy for each BF16 matrix. Set `QIMG21_PROFILE=1` for diagnostic
per-block upload, compute, and release timing. At 1024x1024, one step takes
37.74 seconds and its first matched prediction has cosine 0.9999915 and
relative L2 0.004273 against the saved PyTorch reference. This single-step
result was followed by a complete 1024x1024/40-step run from the same saved
PyTorch initial latents and prompt embedding. All 40 free-running checkpoints
were finite and passed the 0.99996 cosine gate: minimum cosine 0.99999049 and
maximum relative L2 0.0043614, both at the final step. The denoiser took
1375.75 seconds (22m56s) and peaked at 14,102,656 KiB host RSS. Native ROCm
VAE decoding took another 19.74 seconds and yielded a coherent 1024x1024
apple image. Compared with the saved PyTorch reference PNG, the decoded RGB
has cosine 0.99999435 and mean absolute channel error 0.373/255. Artifacts
are under `tmp/qimg21-rdna4-1024-40/`.

With `--attention wmma-fused`, the same 1024x1024/40-step denoiser run took
238.50 seconds (3m59s) and peaked at 14,103,924 KiB host RSS. Every saved
checkpoint is finite and passes the 0.99996 cosine gate; minimum cosine is
0.99998877 and maximum relative L2 is 0.004739 at the final step. Native VAE
decode took 17.79 seconds. The resulting RGB image has cosine 0.99999437 and
mean absolute channel error 0.385/255 versus the saved PyTorch reference PNG.
This is faster than the repository's 5060 Ti W8A8 transformer measurement of
8m12s, though the AMD run uses BF16 weights and a different attention kernel.
Artifacts are under `tmp/qimg21-rdna4-fused-1024-40/`.

The public `native_generate.py --backend rocm` path now selects fused WMMA
attention for text-to-image and scalar mask-aware attention for editing. A
standalone 1024x1024/40-step generation with native text encoding, NumPy
seed-42 noise, fused denoising, native VAE decode, and PNG output completed in
265.88 seconds (4m26s), peak host RSS 14,103,212 KiB, on the RX 9070 XT.
The resulting image is a coherent red apple on a white table; it is not
pixel-comparable to CUDA seed 42 because the initial-noise RNG differs.
Artifacts are under `tmp/qimg21-rdna4-fused-e2e-1024-40/`.

For editing, pass `--native-attention wmma-fused` explicitly to the standalone
generator. On a two-step 256x256 target with a 1024-condition image, this took
16.89 seconds versus 30.47 seconds for scalar ROCm attention. Final latent
cosine versus a saved CUDA native run was 0.99992565 (scalar ROCm: 0.99990237),
below the strict 0.99996 gate. A 40-step 256x256 fused edit produced a visually
similar apple to scalar editing, but final latent cosine between the two was
0.99680; therefore fused editing is opt-in while the matched PyTorch parity
gate remains unmet. It is not part of the validated text-to-image speed claim.

The matched efficient-SDPA editing gate can be run with the same reference
regression as CUDA, selecting the RDNA4 binary:

```sh
ROCEW_ROCM_LIB=/opt/rocm/core/lib python3 cuda/qimg21/editing_regression.py \
  --native-binary rdna4/qimg21/test_hip_qimg21_native \
  --model MODEL_ROOT --reference-dir REFERENCE_CAPTURE \
  --work-dir tmp/qimg21-rdna4-edit-regression \
  --native-attention wmma-fused --native-normalization vector4 \
  --native-rope host-table-exact
```

For the saved two-step 1024-condition/256-target capture, prediction cosines
were 0.999891917 and 0.999873082; the trajectory minimum was 0.999863032.
All are below the 0.99996 gate, so editing parity is not yet established.
For a same-GPU PyTorch reference, `pytorch_rocm_reference.py` replays the
saved transformer inputs directly with sequential CPU offload. It uses the
project's existing ROCm Torch/torchvision and the pinned Diffusers Python
package files from the CUDA reference environment; it does not load CUDA
Torch into the ROCm process:

```sh
ref/pixal3d/run.sh rocm rdna4/qimg21/pytorch_rocm_reference.py \
  --model /mnt/disk2/models/qimg-21 \
  --capture-dir tmp/qimg21-edit-reference-efficient \
  --out-dir tmp/qimg21-edit-reference-rocm-direct \
  --diffusers-site-packages tmp/qimg21-ref-venv/lib/python3.12/site-packages \
  --sdpa-backend efficient
```

Add `--capture-block0` with a fresh `--out-dir` to save the selected step's
`block0/{time2,time2_silu,mod,hidden0,mod_ln,q,k,v,attn_raw,attn_out,post_attn_hidden,mod_ln2,mlp_gate,mlp_proj,mlp_act,mlp_out,block_00}.npy`
alongside the predictions. The default is step 0; `--capture-step 1` selects
the second editing step. The hooks leave the two predictions byte-identical
to the run without stage capture. This makes the PyTorch ROCm reference
directly comparable to `QIMG21_STAGE_BLOCK=0` dumps from the native runner.
`--capture-all-blocks --capture-step 1` also writes all 32 target block
outputs under `all_blocks/`. `--free-run` saves a complete regression fixture:
it uses the pinned Diffusers scheduler to update the ROCm target latent and
feeds that latent into the next step while retaining the captured condition
and prompt. The reconstructed model timestep is checked against every
captured timestep.
It currently requires a capture without true CFG. For this two-step fixture,
the same update applied to the saved CUDA predictions reproduced both CUDA
trajectory checkpoints exactly. The scheduler-backed ROCm capture is
byte-identical to the earlier BF16 Euler reconstruction on this fixture.

```sh
ref/pixal3d/run.sh rocm rdna4/qimg21/pytorch_rocm_reference.py \
  --model /mnt/nvme01/models/qimg-21 \
  --capture-dir tmp/qimg21-edit-reference-efficient \
  --out-dir tmp/qimg21-edit-reference-rocm-scheduler-20260923 \
  --diffusers-site-packages tmp/qimg21-ref-venv/lib/python3.12/site-packages \
  --sdpa-backend efficient --free-run
ROCEW_ROCM_LIB=/opt/rocm/core-10.0/lib python3 cuda/qimg21/editing_regression.py \
  --native-binary rdna4/qimg21/test_hip_qimg21_native \
  --model /mnt/nvme01/models/qimg-21 \
  --reference-dir tmp/qimg21-edit-reference-rocm-scheduler-20260923 \
  --work-dir tmp/qimg21-edit-rocm-scheduler-regression-20260923 \
  --native-attention math --native-normalization vector4 \
  --native-rope host-table-exact
```

The original `tmp/qimg21-edit-reference-rocm-expected/` fixture has ROCm
predictions but CUDA trajectory files and CUDA `run.json` metadata. Do not use
its trajectory numbers as a same-GPU ROCm comparison. A fresh independent
free-running capture is under
`tmp/qimg21-edit-reference-rocm-free-bundle-20260923/`; its native regression
is under `tmp/qimg21-edit-rocm-free-regression-20260923/`. Against PyTorch ROCm,
the native scalar-attention path reaches prediction cosines 0.999974046 and
0.999876137, and trajectory cosines 0.999964614 and 0.999964570. Thus both
trajectory checkpoints pass the unchanged 0.99996 gate, while the second
matched-input prediction still fails. The free-running CUDA/ROCm trajectory
cosines are 0.999917801 and 0.999917953. With identical captured inputs,
pinned CUDA PyTorch 2.14 versus ROCm PyTorch 2.11 prediction cosines are
0.999936229 and 0.999861802; these are a cross-framework reference floor,
not native acceptance results. A defensible cross-platform criterion would
retain 0.99996 for native versus same-GPU predictions and trajectory, and
report CUDA versus ROCm reference drift separately. Do not relax the
nonquantized gate to the reference floor; the second native prediction still
needs a fix.
An optional, separately named cross-platform floor tier would require each
native-versus-PyTorch-ROCm prediction cosine to be at least the corresponding
pinned PyTorch-CUDA-versus-PyTorch-ROCm cosine on identical inputs, and both
native trajectory checkpoints to retain the 0.99996 requirement. The current
native predictions exceed those reference floors by about 0.0000378 and
0.0000143, and the trajectories pass. This tier is a proposal for review;
the unchanged strict same-GPU prediction gate remains open.

On the free-running second-step ROCm input, the native and PyTorch ROCm target
block outputs start at cosine 0.999999816 after block 0, first fall below
0.99996 after block 9, and end at 0.999939892 after block 31. All 32 values
are saved in `tmp/qimg21-edit-rocm-block-compare-20260923.json`. This is
gradual accumulated drift; the trace does not identify a single failed block.
On identical native Q/K/V, PyTorch ROCm attention and HIP scalar attention
reach target cosine 0.999999625. Replaying exact PyTorch ROCm hidden state and
modulation into native block 0 barely changes its target output cosine
(0.999999812); replaying PyTorch attention as well raises it to 0.999999981.
After attention replay, target output projection differs in 472 of 1,048,576
BF16 values, and the MLP output differs in 30,404 values. The detailed
`--capture-block0` hooks leave both reference predictions and checkpoints
byte-identical to the uncaptured run. These are guarded local diagnostics;
injected-state runs do not emit a model prediction.
With identical ROCm `hidden0` and `mod` replayed into native HIP, vector4
`mod_ln` differs in 575 of 17,907,712 BF16 values; the free-running native
state differs in 5,702. On identical `mod_ln` input, hipBLAS BF16-output QKV
matches PyTorch ROCm bit-for-bit, but substituting hipBLAS QKV throughout
the native model does not pass both editing steps. FP64 normalization also
improves the first step while worsening the second. Neither experiment is
selected for production; the remaining divergence is amplified across blocks.

On the saved two-step fixture, PyTorch ROCm predictions have cosine
0.999936229 and 0.999861802 against the PyTorch CUDA capture. Native HIP
predictions have cosine 0.999939325 and 0.999882018 against PyTorch ROCm.
Both native comparisons still miss 0.99996. The PyTorch comparison also
includes framework-version differences (ROCm 2.11 versus CUDA 2.14) and
does not, by itself, isolate GPU hardware rounding. Diagnostic FP64 timestep
accumulation reaches 0.999960013 against PyTorch ROCm on step 0 but drops
to 0.999873083 on step 1; it therefore remains non-production.
The production scalar editing-attention path has saved-prediction cosine
0.999930810 and 0.999860295 against CUDA, or 0.999974046 and 0.999884473
against PyTorch ROCm. Its CUDA trajectory cosines are 0.999909817 and
0.999910201, so it also misses the gate. A step-1 block-0 capture from the
same ROCm inputs shows 603 differing BF16 values in the initial 17,907,712
hidden values, then 5,728 at modulated normalization, 195,397 at Q, and
3,200,496 after block 0. This is numerical amplification, not a GPU launch
failure. The reference capture hooks leave both predictions byte-identical
to the no-capture reference.
PyTorch ROCm's default and forced efficient SDPA produce byte-identical
predictions on this fixture, so choosing between them does not resolve the
editing miss.
The regression harness accepts `--native-gemm scalar` as a diagnostic
comparison. On the saved ROCm fixture, scalar GEMM produced cosines
0.999920042 and 0.999871362, versus WMMA's 0.999974046 and 0.999884473;
its trajectory was 0.999898372/0.999899996. WMMA therefore remains the
production choice and scalar GEMM is not enabled by default.
The matched first-step trace has byte-identical BF16 timestep SiLU inputs on
CUDA and ROCm; the second projection differs in 78 of 8192 BF16 values.
Those differences propagate into 24 image-row and 233 text-row modulation
values. For diagnosis only, `QIMG21_REPLAY_TIME2=PATH` and
`QIMG21_REPLAY_MOD=PATH` accept saved F32 `.npy` stage tensors containing
BF16 values; `QIMG21_REPLAY_MOD_ROW=0|1` limits modulation replay to one row.
On the first editing prediction, replaying the CUDA timestep state raised
cosine from 0.999891917 to 0.999924535; replaying CUDA modulation raised it
to 0.999950087, still below the gate. Scalar HIP and hipBLAS two-row GEMM
diagnostics also missed the gate, so they are not production paths.

For a precision probe, `QIMG21_DIAG_TIME2_F64=1` runs the two-row timestep
projection with FP64 accumulation; `QIMG21_DIAG_MOD_F64=1` does the same for
the modulation projection. Both are diagnostic only and leave the default
WMMA path unchanged. On the saved first editing step, the input to timestep
SiLU is byte-identical between CUDA and HIP, but the pre-round WMMA output
has a small downward error relative to an FP64 dot product. FP64 timestep
accumulation reproduces the FP64-rounded BF16 state exactly and improves the
first prediction cosine from 0.999891917 to 0.999932841. Adding FP64
modulation instead lowers it to 0.999903025. Neither clears the 0.99996
editing gate, so broader numeric or attention differences remain. Set
`QIMG21_STAGE_KEYS=time2_pre_round` to dump the FP32 projection before its
BF16 rounding for further diagnosis.

`QIMG21_STAGE_KEYS=time2_silu` dumps the BF16 SiLU output immediately before
the modulation GEMM. With CUDA's BF16 `time2.npy` replayed into HIP, all 8192
SiLU values are byte-identical; modulation still differs in 12 image-row and
21 text-row values. Thus the remaining matched-input discrepancy at that
boundary is GEMM accumulation, not SiLU. FP64 modulation with the replayed
CUDA timestep yields 16/13 differing values and first-prediction cosine
0.999934041; FP64 modulation alone yields 0.999915156. Both miss the gate,
and neither is selected by default.

Matched block-0 replay further separates accumulated input drift from local
kernel error. Injecting the saved CUDA `hidden0.npy` and `mod.npy` with
`QIMG21_REPLAY_HIDDEN`, `QIMG21_REPLAY_MOD`, and `QIMG21_STAGE_BLOCK=0` into
the HIP runner gives block-0 attention cosine 0.999999902 overall and
0.999999787 on the 256 target rows. The sampled block-output relative L2 is
0.0004035. Replaying CUDA `attn_raw.npy` as well lowers that sampled
block-output relative L2 to 0.0002883, so attention contributes, but the
output projection and MLP still introduce BF16 differences. These are
diagnostic injected-state comparisons, not model acceptance; the free-running
editing predictions above remain below the gate.

The saved first-step, free-running target-hidden trace (`QIMG21_STAGE_ALL_BLOCKS=1`)
shows gradual amplification rather than one failing layer. CUDA/HIP target
cosine is 0.999999597 after block 0, first drops below 0.99996 after block 9,
reaches 0.997721522 at block 29, then recovers to 0.999673598 after block 31.
This trace compares native CUTLASS-efficient CUDA with HIP fused WMMA using
the identical saved fixture. It identifies accumulated BF16/kernel-order
sensitivity, not a single bad block or a passing end-to-end edit.

To validate attention against PyTorch ROCm without installing Diffusers into
the Pixal3D reference environment, use the saved native editing layout:

```sh
ref/pixal3d/run.sh rocm cuda/qimg21/attention_probe.py \
  --stage-dir tmp/qimg21-rdna4-edit-efficient-stage0 \
  --editing-layout tmp/qimg21-edit-efficient-native-exact/fixture-000/layout.txt
```

On the free-running first-block trace, native HIP versus PyTorch ROCm default
SDPA has cosine 0.999999990 overall and 0.999999945 on target rows. The same
probe with CUDA hidden/modulation replay gives 0.999999991 and 0.999999951.
These are same-Q/K/V local attention checks; they pass the 0.99996 stage gate
but do not establish end-to-end editing parity.

The 1024-condition native vision path converts BF16 checkpoint biases to F32
for its F32 linear epilogue and rounds the patch GEMM output to BF16 before
adding the bias, matching CUDA's two activation boundaries. On the same
1024x1024 input, patch-projection relative L2 versus CUDA fell from 0.002067
to 0.000076, and block-0 QKV cosine rose to 0.99999999. After 27 blocks,
merged vision cosine rose from 0.99545964 to 0.99686712; multimodal prompt
cosine rose from 0.93467182 to 0.95263400. Before either vision fix,
merged-vision and prompt cosines were only 0.17442450 and 0.18132024.
These are CUDA comparisons, not a strict text-encoder parity result. CUDA's
own flash-versus-scalar vision merged-feature cosine is 0.99794674 on this
input, so those attention algorithms also differ numerically. The RDNA4
vision executable defaults to HIP GEMM and scalar HIP attention and does not
require CUDA/cuDNN plugins; a default-options first-block smoke test is
byte-identical to the explicit `--attention math` run. Feeding the same
post-bias-fix ROCm vision features to both native text encoders yields prompt
cosine 0.99789171. Keeping CUDA text execution but swapping CUDA vision
features for ROCm vision features yields 0.93445386, locating most of the
remaining multimodal difference in the vision stack.

For a pinned same-input vision oracle, `pytorch_rocm_vision_reference.py`
loads only the Qwen3-VL visual weights. It either replays saved
`pixel_values.npy`/`image_grid_thw.npy` or uses the Qwen Image 2.1 pipeline's
resize, white alpha composite, prompt template, and processor to prepare an
image. On the house fixture, generate the PyTorch ROCm stages with:

```
ref/pixal3d/run.sh rocm rdna4/qimg21/pytorch_rocm_vision_reference.py \
  --model /mnt/nvme01/models/qimg-21 \
  --image ref/pixal3d/upstream/assets/images/1_img.png \
  --out-dir tmp/qimg21-rocm-vision-house-oracle-20260924 \
  --diffusers-site-packages tmp/qimg21-ref-venv/lib/python3.12/site-packages
```

The 4096×1536 normalized patch tensor is byte-identical to the native
`--image` preprocessing of the same resized house (6,291,456 values). Replaying
those exact patches through native HIP scalar attention yields cosine
0.999996367 after block 0, 0.999877620 after block 8, 0.996742602 after block
26, and 0.996581803 after the merger against PyTorch ROCm. First-block norm1
and QKV cosines are 0.999999999 and 0.999999999; attention projection is
0.999998301. The attention result before projection is 0.999994906: the first
measurable block-0 divergence occurs after near-identical QKV, in attention.
The GELU output is identical wherever its BF16 input is identical. The
PyTorch CUDA versus PyTorch ROCm reference floor on the same
patch tensor is 0.999994394 after block 0 and 0.996302817 after the merger.
The pinned reference implementations therefore differ well beyond the
0.99996 prediction gate at the vision merger. This is stage evidence, not an
editing prediction pass. The unchanged end-to-end editing gate remains open.
The replayed native merged and three deepstack arrays are byte-identical to
the outputs of the full 1024x1024/40-step house edit rerun.
Forcing PyTorch ROCm's math SDPA on those same patches gives a merger cosine
of 0.998161784 against its default SDPA result, but native HIP scalar
attention versus that math oracle is 0.995929653 at the merger. A matched
attention algorithm alone therefore does not remove the accumulated vision
drift; the math replay is under
`tmp/qimg21-rocm-vision-house-math-oracle-20260924/`.

For the same 1024x1024 condition and target dimensions, a two-step standalone
edit took 62.84 seconds end-to-end on the RX 9070 XT with the corrected
vision encoder, versus 72.87 seconds on
the RTX 5060 Ti (CUDA `cutlass-efficient` attention). This is a throughput
comparison, not a same-noise output comparison: CUDA uses PyTorch's seed-42
noise while ROCm uses NumPy's seed-42 noise.

With both vision arithmetic fixes, a standalone 1024x1024/40-step edit
with native vision, text, denoiser, and VAE completed in 521.28 seconds,
peaking at 14,123,724 KiB host RSS. All 40 saved latent checkpoints are finite
and the 1024x1024 PNG is valid. With the house source image and the prompt
"a red apple on a white table", the edit retains the house and adds a small
apple beside it. This is an execution/visual check, not a matched PyTorch
editing-parity pass; artifacts are under
`tmp/qimg21-rdna4-fused-edit-40-1024-patchround/`.

This acceptance run uses the saved reference text embedding, so it verifies
the ROCm denoiser and VAE, not the full native text-encoder accuracy. Native
text-to-image and image-editing orchestration have separate 256x256 one-step
execution smoke tests. Both also completed 40-step 256x256 runs using native
text conditioning: text-to-image denoising took 68.04 seconds and image-editing
denoising took 77.24 seconds. Native VAE decode produced coherent red-apple
images in both cases. The editing run reused the native VAE-encoded condition,
27-block vision features, and 36-layer multimodal text embedding from the
one-step end-to-end run. These are functional and visual checks, not matched
40-step PyTorch editing-parity tests. The ROCm row-INT8 package also executes
through BF16 dequantization and WMMA, but its measured 256x256 one-step latency is 23.12
seconds versus 5.16 seconds for BF16 weights; it is a compatibility path, not
yet a performance optimization.

## 1024x1024 repeated RDNA4 revalidation (2026-09-24)

Using model `/mnt/nvme01/models/qimg-21`, prompt `a red apple on a white
table`, BF16, seed 42, 40 steps, native F32 VAE, and `wmma-fused` attention,
`native_generate.py --backend rocm` completed three text-to-image runs in
4:23.70, 4:23.66, and 4:23.98, and two house-image edits in 8:40.74 and
8:39.67. Repeat runs reused the exact F32 initial-noise file through
`--initial-latents tmp/qimg21-revalidate-t2i-20260924/latents.npy`. All
repeated PNGs, final latents, and native VAE tensors are byte-identical within
each mode; all 40 saved steps are finite. Full-run VRAM sampling observed
3,360,542,720 bytes at peak for text-to-image and 4,566,437,888 bytes for
editing, including a 59,912,192-byte device baseline. The valid 1024x1024
outputs and their memory traces use the prefixes
`tmp/qimg21-rocm-bench-t2i-r3-20260924` and
`tmp/qimg21-rocm-bench-edit-r2-20260924`. A synchronized one-step profile on
the same text-to-image inputs
measured 4.6422 seconds across the 32 block compute sections, 3.0921 seconds
for block weight uploads, and 0.0146 seconds for release.

On the RTX 5060 Ti, the same model, prompt, BF16 precision, exact initial
noise, 1024x1024 resolution, 40 steps, and native F32 VAE completed in
17:43.98 with `cutlass-efficient` attention. The monitor measured
3,263,168,512 bytes peak for this invocation's CUDA process group; another
GPU process held about 7.6 GiB throughout, so the wall-time comparison is
contention affected. Native text embeddings differ across backends (cosine
0.999032835), despite the identical prompt and noise; this is a matched-input
throughput run, not an output-parity result. The CUDA PNG and memory trace are
under `tmp/qimg21-cuda-bench-t2i-r2-20260924*`.
The same CUDA command with a new work directory repeated in 18:08.06; the
prompt embedding, initial/final latents, native VAE output, all 40 saved
steps, and PNG are byte-identical to the first CUDA run. Its measured process
peak was again 3,263,168,512 bytes. Across ROCm and CUDA, the matched-input
text-to-image PNGs have pixel cosine 0.999993255 and PSNR 50.11 dB. The
second CUDA run uses prefix `tmp/qimg21-cuda-bench-t2i-r3-20260924`.

The paired house edit on CUDA used the same source PNG (including a
byte-identical native resized image), exact initial noise, model, prompt,
BF16 precision, 1024x1024 target, 40 steps, and native F32 VAE. It completed
in 24:24.27 with 40 finite steps, finite condition/final latents and decoder
tensor, a valid PNG, and 3,263,168,512 bytes peak CUDA process-group memory.
Native CUDA and ROCm output PNGs have pixel cosine 0.9999897319, mean absolute
pixel difference 0.785/255, and PSNR 47.09 dB; the final latent cosine is
0.9999025564. The image similarity is a visual/full-path result and does not
replace the still-failing two-step prediction gate. The CUDA edit artifacts
use prefix `tmp/qimg21-cuda-bench-edit-r1-20260924`.

On a separate identical-tensor one-step native denoiser probe, using the
ROCm text embedding and initial noise on both backends, native wall times
were 9.45 seconds (ROCm WMMA fused) and 26.89 seconds (CUDA CUTLASS
efficient); resulting latent cosine was 0.999981750. These invocation times
include weight loading and launch setup. The ROCm block compute subtotal
above is a narrower kernel section; CUDA was still contended by the other
GPU process.

The unchanged two-step same-GPU editing gate still fails. `reverse64`
attention gave prediction cosines 0.999932210/0.999879014, and fused WMMA
gave 0.999939325/0.999864240. Plain WMMA gave the same
0.999974046/0.999876137 as scalar `math` on this fixture: both trajectory
checkpoints pass, but the second prediction does not. These diagnostics are
under `tmp/qimg21-edit-rocm-{reverse64,wmma-fused,wmma}-20260924/`.
Default normalization with exact RoPE gave 0.999954939/0.999889537 and
failed both trajectory checkpoints. Default RoPE with vector4 normalization
gave 0.999963776/0.999874586 and also failed both trajectory checkpoints.
Neither variant replaces vector4 plus exact RoPE.
Combining the diagnostic FP64 timestep and modulation projections gave
0.999945683/0.999876798 predictions and failed both trajectory checks on
the same ROCm reference; it is not selected for production.
