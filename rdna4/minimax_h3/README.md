# MiniMax H3 INT8 on RDNA4

Native C++ text-only generation using the local MiniMax H3 **Ref2VA pruned INT8
ConvRot** checkpoint. HIPRTC compiles gfx1200/gfx1201 kernels; signed INT8 WMMA
accumulates into INT32. There is no PyTorch dependency in the inference runner.
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
omitted. No reference inputs, guidance branch or LoRA are applied.

The 14,336 MiB process budget reserves 3,072 MiB for runtime overhead. Block
weights stream from immutable mmap files. FFNs process 256 rows at a time;
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
