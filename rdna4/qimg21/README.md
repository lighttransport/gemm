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
The runner covers the 32-layer
denoiser ABI, CFG inputs, editing layout inputs, stage dumps, and FlowMatch
latent updates. Matching native text, vision, VAE decoder, and VAE encoder
executables are built by the same Makefile. ROCm web generation uses the
native VAE path. A separate PyTorch reference mode is available when a ROCm
PyTorch environment with Qwen Image 2.1 Diffusers support is configured.
Health reports each native component separately.

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
