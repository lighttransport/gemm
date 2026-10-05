# MiniMax H3 INT8 on RDNA4

Native C++ generation using MiniMax H3 **Ref2VA or FL2VA pruned INT8
ConvRot** checkpoints. HIPRTC compiles gfx1200/gfx1201 kernels; signed INT8 WMMA
accumulates into INT32. The inference runner has no PyTorch dependency. Image
conditioning uses bounded upstream PyTorch ROCm visual and VAE encoders before
the native Qwen language model, packed DiT, and decoder run.
The Python wrapper packages frames into a silent MP4 and records provenance.

Build from the repository root:

```sh
make -C rdna4/minimax_h3 -j4
make -C rdna4/minimax_h3 test
make -C rdna4/hunyuan_video15_native test-gpu
```

GPU tests need access to `/dev/kfd` and the AMD render node. Build products and
runtime scratch are under `tmp/video-rocm/`; no hipcc is needed. Runtime requires
ROCm HIP/HIPRTC libraries and HIP headers available at `/opt/rocm/include`.
The default BF16 ConvRot uses hipBLAS for its dense 256×256 Hadamard rotation;
all checkpoint INT8 projections still use the repository's signed INT8 WMMA
kernels. `--convrot-hipblas 0` selects the native factorized rotation, whose
reduction order can change INT8 rounding. Dense BF16 projections preserve
BF16 weights and default to hipBLAS to match reference accumulation.
`--bf16-hipblas 0` selects the native BF16 WMMA path; the video VAE retains FP16.
FP32 projections use standard hipBLAS with FP32 inputs and accumulation by
default. `--fp32-hipblas 0` selects the native IEEE kernels. The additive C API
`h3_set_fp32_hipblas()` configures an idle context; the configuration struct
remains 40 bytes. An actual-weight 257×5376 final-projection fixture matched
all 24,672 PyTorch FP32 output values exactly through hipBLAS. Rotary values are
computed on the GPU to match the reference math library. Metrics record separate
INT8/BF16 WMMA, FP32/BF16 hipBLAS and dense-ConvRot call counts.
`--vae-hipblas 1` optionally selects FP16 hipBLAS GEMM during VAE decoding;
the DiT keeps its configured kernels, including IEEE FP32 islands. The default
decoder uses native FP16 WMMA. The hipBLAS decoder passed all 39 frames of a
288×288 shared-latent probe (maximum relative L2 0.000604), with a 36.5-second
decode time on this system.

Python packaging requires Pillow and ffmpeg. Independent checks also require
NumPy, PyTorch, safetensors and tokenizers; the DiT/VAE reference uses ROCm PyTorch.

An optional standalone AOTriton bridge supplies long BF16 attention. Build it
against an AOTriton SDK with headers, `libaotriton_v2.so` and its kernel images:

```sh
make -C rdna4/minimax_h3 aotriton AOTRITON_ROOT=/path/to/aotriton-sdk
# Add this argument to generation:
# --aotriton-bridge tmp/video-rocm/h3-build/libvideo_aotriton.so
```

The bridge was built with AOTriton 0.11.2 and has no PyTorch library dependency.
It loads only when explicitly selected; the normal runner build needs no
AOTriton SDK. INT8 GEMMs retain signed WMMA. The wrapper records the bridge hash,
and metrics count provider attention calls. At 37,722 tokens, its single-head
output matched all 4,828,416 reference BF16 values exactly; the full 64-head
attention took 2.80 seconds. A fresh 128×128, 22-frame, one-update diagnostic
passed all 28 independent comparisons, with video/audio latent relative L2
below 5e-7 and frame relative L2 below 0.00094. The full-resolution one-update
result is recorded below; default 39-update acceptance is still required.

```sh
python3 rdna4/minimax_h3/generate.py \
  --model /mnt/disk01/models/h3/weights \
  --prompt 'A red ball rolling on a wooden table, cinematic lighting.' \
  --out tmp/video-rocm/h3-video --dump-dir tmp/video-rocm/h3-dump \
  --allow-experimental
```

Defaults are 1344×768, 124 frames, 24 fps, seed 42, 40 sigma-grid points
(**39 Euler updates**), video shift 12 and audio shift 3. The model retains both
stereo audio streams during every denoising step; audio decoding and muxing are
omitted. Text-only generation applies no reference inputs, guidance branch or LoRA.

The default 12,288 MiB process budget reserves 3,072 MiB for runtime overhead.
`--vram-budget-mib 14336` permits extra headroom on a 16 GB device. Block
weights stream from immutable mmap files. QKV projections stream 512 rows at
a time to avoid a second full-size packed projection. FFNs process 256 rows at a time;
attention uses bounded online softmax; VAE decoding uses 256-pixel spatial tiles
and seven-latent temporal windows. A 64 GB host is required by the wrapper.
The decoder retains its bounded FP16 weight cache across tiles. A 39-frame
288×288 probe passed all frames with 4.85 GB uploaded (previously 38.8 GB),
5,289 MiB managed VRAM and 27.7 seconds including runtime compilation.
`--compress-dumps` losslessly stores diagnostic F32 captures as gzip files.
The AMD device lock is shared with the Hunyuan wrapper. Cancellation cleans
owned partial output and releases the lock.

The native executable and `libh3_rocm.so` expose the API in [h3.h](h3.h).
Callbacks publish transient RGB frame buffers and denoising progress.
`--noise-file` reads F32 NCTHW video noise; `--audio-noise-file` reads F32 NC2T
stereo noise. `--dump-dir` captures F32 states, each Euler update, and decoded
frames for reference comparison. Diagnostic geometries start at 64×64 and five
frames; dimensions must be multiples of 32, and frame counts must be `17*n+5`.

## Image conditioning on ROCm

Ref2VA accepts up to nine `--reference-image` inputs. Name references in the
prompt as `<Picture 1>`, `<Picture 2>`, etc. FL2VA accepts `--first-frame`,
`--last-frame`, or both, using its separate FL2VA checkpoint. Reference video
and audio inputs are not implemented. Output is silent video.

Install [requirements-conditioning.txt](requirements-conditioning.txt) in a
PyTorch ROCm environment. The default interpreter is
`tmp/vhuman-rocm-venv/bin/python`; override with `--conditioning-python`.
The encoder imports ComfyUI revision
`2472a20bd291451acc303917059ab14dfc380478`; check it out at
`tmp/video-rocm/pytorch-bench-comfy`, or select it with `--conditioning-comfy`.
Build the standalone AOTriton bridge as described above. The image wrapper
selects `tmp/video-rocm/h3-build/libvideo_aotriton.so` by default, including
short refiner attention to preserve upstream numerical behavior.

The H3 bridge also provides cached hipBLASLt BF16 ConvRot plans. It matches
PyTorch's `X @ rotation` dispatch, BF16 output type, and non-transposed right
operand. Using a transpose on the symmetric Hadamard matrix changes the
selected reduction and can move a BF16 rounding tie across an INT8 boundary.
The native allocator budgets its 32 MiB workspace; all checkpoint INT8
projections retain signed WMMA. Metrics include `convrot_hipblaslt_calls`.
Build requires the ROCm hipBLASLt headers and library. AOTriton headers and
library shipped with this ROCm PyTorch can be selected with
`AOTRITON_ROOT=/mnt/disk01/vhuman-rocm/venv/lib/python3.12/site-packages/torch`;
the compiled bridge calls standalone GPU libraries and does not call PyTorch.

```sh
python3 rdna4/minimax_h3/generate.py \
  --variant ref2va --reference-image portrait.png \
  --prompt 'The person in <Picture 1> smiles, fixed camera.' \
  --width 480 --height 832 --frames 22 --steps 6 \
  --out tmp/video-rocm/h3-reference --allow-experimental

python3 rdna4/minimax_h3/generate.py \
  --variant fl2va --first-frame first.png --last-frame last.png \
  --prompt 'The person smiles, fixed camera.' \
  --width 480 --height 832 --frames 22 --steps 6 \
  --out tmp/video-rocm/h3-keyframes --allow-experimental
```

Keep all three weight components and the tokenizer in the existing H3 model
directory. FL2VA additionally needs
`diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors` from
Comfy-Org/MiniMax-H3 revision `e5eb578a89295337b8ff433a035929ce0279e0b6`;
SHA256 `e889202c41dafb67b10d67b97f0d8541508036a6090af23425a5c2615d03c47a`.

The visual tower and VAE encoder load sequentially and exit before native
generation. Reference scaling uses equal shares of the target canvas area,
rounded to 32-pixel dimensions; Qwen visual
tokens are capped at 1536 total. FL2VA keyframes use the target canvas. Packed
reference/keyframe tokens remain fixed during every Euler update and use the
upstream visual-conditioning timestep and rotary coordinates. The bundle
records encoder input/output hashes, frame anchors, geometry, seed and model
provenance. `--conditioning-dir` reuses a matching bundle, copying it into
the new generation directory and validating required files and checksums.
Bundles bind all model component hashes; a bundle from different weights is
rejected. Encoder weights are checked for modification during preprocessing.

Check conditioned captures using the independent references:

```sh
python3 ref/minimax_h3_native/verify.py qwen \
  --prompt 'The person in <Picture 1> smiles, fixed camera.' \
  --conditioning-dir tmp/video-rocm/h3-reference/conditioning \
  --native tmp/video-rocm/h3-reference-dump --out tmp/video-rocm/h3-reference-qwen
python3 ref/minimax_h3_native/conditioning.py \
  --generation tmp/video-rocm/h3-reference --native tmp/video-rocm/h3-reference-dump \
  --qwen-reference tmp/video-rocm/h3-reference-qwen --out tmp/video-rocm/h3-reference-check.json
```

Capture generation with `--dump-dir` when using these checks. Diagnostic
similarity does not certify portrait identity, expression quality, or the
default 39-update trajectory; experimental opt-in remains required.

On RX 9070 XT, both 64x64, five-frame Ref2VA and two-keyframe FL2VA diagnostics
passed all 18 comparisons through five Euler updates. Conditioned Qwen,
refiner and every video/audio update were exact. Decoded-frame relative L2
was <=0.00149 for Ref2VA and <=0.00106 for FL2VA. Both final runs had 5005 MiB
sampled peak, with 1350 native INT8 WMMA and hipBLASLt ConvRot calls.
A full-resolution 1344x768, 124-frame single-block memory probe included
4096 text/conditioning-equivalent rows and peaked at 9456 MiB under the
12,288 MiB budget, including the corrected ConvRot provider. This is a memory probe,
not a complete full-resolution generation or full-trajectory parity claim.

The vhuman Ref2VA smile candidate at 480x832, 22 frames and five updates
completed with 5129 MiB sampled peak and 309.914 seconds including image
preprocessing, native inference and packaging, excluding model verification.
DiT time was 121.446 seconds. MediaPipe found a face in all 22 frames and
selected smile weights 0.949/0.950. These remain unreviewed candidate assets.
The final FL2VA first-frame candidate at the same resolution, frame count
and update count completed in 202.508 seconds, with 5215 MiB sampled peak,
109.533 seconds in the DiT, and one visible face in all 22 frames. Its selected
smile weights were 0.910/0.870. These single-run timings are not a comparative
PyTorch benchmark. Direct packing into one buffer preserves all 21 native
diagnostic captures exactly and avoids intermediate concatenations.

The BF16 rounding-boundary GPU regression is separate from host tests:

```sh
H3_GPU_TESTS=1 LD_LIBRARY_PATH=/opt/rocm/core-7.14/lib \
  tmp/vhuman-rocm-venv/bin/python -m unittest rdna4.minimax_h3.test_convrot_gpu
```

## Numerical validation

```sh
python3 ref/minimax_h3_native/verify.py components \
  --out tmp/video-rocm/h3-projection-check
python3 ref/minimax_h3_native/verify.py qwen \
  --prompt 'A red ball rolling on a wooden table, cinematic lighting.' \
  --native tmp/video-rocm/h3-dump --out tmp/video-rocm/h3-qwen-reference
```

The reference uses dense Kronecker Hadamard rotation and CPU INT32 accumulation
for checkpoint projections. On the RX 9070 XT, the tested Qwen and H3 INT8
projections matched exactly. The complete Qwen layer-50 comparison achieved
cosine **1.0**, relative L2 **0.0** using the independent PyTorch
GPU text reference with activation-dtype BF16 ConvRot. Qwen RMSNorm matches
the pinned PyTorch reduction order, and short causal attention applies split
Q/K scaling before FP32 softmax. The token refiner also matches exactly
with the default dense BF16 hipBLAS path. Tests also cover signed values,
tails, row scales, causal grouped-query attention at 1/17/67/257/512 tokens,
an adversarial BF16 RMSNorm rounding boundary, and FP16 VAE projection.

A `256×21504×5376` INT8 projection measured **26.25 ms / 2.26 TOPS** with the
small tile and approximately **5.5 ms / 10.7 TOPS** with the pipelined 128×128 tile.
These are component measurements, not end-to-end throughput.

A receipt-bound **64×64, five-frame, 39-update** diagnostic passed **87/87**
independent array comparisons. Qwen, the refiner and all **78 video/audio latent
updates matched exactly**. All five decoded frames passed; maximum relative L2
was **0.00103**. The run completed in **313.1 seconds**, executed **8,150 INT8
WMMA projections** and nine dense BF16 hipBLAS projections, and measured
**1,096.5 MiB** peak process VRAM (**833.7 MiB** managed allocation).

A full-resolution single-block probe used **10,811.7 MiB** managed allocation,
with **10,050.5 MiB** sampled process VRAM, and spent **13.3 seconds** in attention
on this system's existing manual clock configuration. This measures a single
block; it does not certify a complete default-resolution video. On identical
full-length Q/K/V for one head, the 64-key base-2 attention path passed against
ROCm FlashAttention (cosine **0.999999987**, relative L2 **0.000163**).
The measured 32-key alternative was slower (about **14.0 seconds**) and was not
selected.

A full-resolution **one-update, 124-frame** native diagnostic completed in
**45.6 minutes**, with **11,448.6 MiB** sampled process VRAM and **11,170.5 MiB**
managed allocation. Memory fit passed, but independent numerical parity failed:
video latent relative L2 was **0.01077**, audio relative L2 was **0.02920**, and
decoded-frame relative L2 reached **0.02440**. This run does not certify the
default schedule. The optional AOTriton bridge addresses the long-attention
rounding discrepancy, as verified by the subsequent one-update diagnostic.

The standalone AOTriton/hipBLAS decoder path subsequently passed a fresh
full-resolution one-update diagnostic: **130/130** arrays, including both latent
streams and all **124 frames**. Video/audio relative L2 was **1.50e-6 / 2.31e-6**;
maximum frame relative L2 was **0.00118**. Native runtime was **17.2 minutes**,
with **11,411 MiB** sampled process VRAM. This establishes the full geometry
for a reduced schedule, not the default 39-update acceptance.

A complete native 39-update run produced all 124 frames in **10,940.7 seconds**
with **11,410.7 MiB** sampled process VRAM and **11,133.7 MiB** managed VRAM.
Its independent reference failed at update 28: video cosine **0.9998948** and
relative L2 **0.01450**. Reference execution was stopped after the failure;
this run does not establish acceptance. The subsequent FP32 projection fix
matches the independent component exactly. Fresh full-profile validation with
that correction is pending.

Full-video acceptance requires independent references for every video/audio
latent update and all 124 decoded frames, cosine ≥0.9999 and relative L2 ≤0.02,
and measured process VRAM within budget:

```sh
python3 ref/minimax_h3_native/verify.py pipeline \
  --manifest tmp/video-rocm/h3-video/manifest.json \
  --native tmp/video-rocm/h3-dump --reference INDEPENDENT_REFERENCE_DIR \
  --out tmp/video-rocm/h3-parity.json
```

Generate the independent DiT/VAE arrays using
[the reference workflow](../../ref/minimax_h3_native/README.md).
Reference receipts bind each independent `.npy` file to the generation manifest.
Component checks and reduced diagnostic runs cannot establish full-video parity.
**Full-resolution end-to-end acceptance has not passed.** Generation remains
experimental until that acceptance passes. The architecture and ConvRot behavior
were checked against ComfyUI revision `2472a20bd291451acc303917059ab14dfc380478`
and comfy-kitchen revision `be003b7c23c5b01328657955b8bc5d3f073d868e`.
H3 uses raw prompt tokens and the unnormalized layer-50 Qwen hidden state.

Additional component checks matched all **100 attention/FFN residual outputs**
across all **50 DiT blocks exactly**, and passed every frame of a **39-frame,
288×288** VAE probe exercising both spatial overlap and temporal stitching
(maximum relative L2 **0.00060**). The full-geometry DiT rotary table matches
all **3,622,080** values exactly. Factorized ConvRot and native dense BF16 WMMA
remain explicit alternatives; their reduction orders can change quantization.
