# Native Pixal3D inference

Single-view inference for the **main release** of [TencentARC/Pixal3D](https://github.com/TencentARC/Pixal3D).
The README's “main” release lives on the upstream `master` branch at the pinned
revision in [sources.json](../../ref/pixal3d/sources.json). The paper version using
Direct3D-S2 is deferred.

The CLI and [C API](../../common/pixal3d.h) run in C/C++ without Python or PyTorch.
CPU uses OpenBLAS; CUDA and ROCm use separately built native plugins. Python is
confined to the [reference/validation environment](../../ref/pixal3d/README.md).

## Resident GPU execution

See [validation and measured performance](OPTIMIZATION.md).

GPU CLI runs default to `--gpu-execution resident --gpu-kernels auto` to retain
NN activations and packed weights on the GPU. CPU runs default to legacy
execution. `auto` uses vendor
BF16/FP16 GEMM, CUDA FA2 or gfx12 WMMA for BF16 self-attention, and a dedicated
five-key cross-attention kernel. `blas` disables matrix-instruction attention;
`mma` additionally forces the experimental explicit MMA/WMMA GEMMs, which can
be slower than vendor GEMM. FP32 conditioning and boundary layers stay FP32;
there is no FP8, INT8 or TF32 precision reduction.

Resident execution covers flow blocks, sparse/dense decoders, DINO and NAF.
Coordinate construction, subdivision decisions, Euler updates, and mesh/PBR
postprocessing remain on the CPU. DINO outputs are reused for identical images
within a generation. Weights are cached in their compute type; activation
buffers hold FP32 values rounded at the model's BF16/FP16 boundaries. Device
scratch is reused within a stage and released at stage changes and before
postprocessing. The existing `--vram-budget-mib` cap applies to resident
allocations as well as legacy workspaces.

`--profile-json tmp/pixal3d/profile.json` records conditioning, diffusion,
decoder and postprocessing times, transfers, allocations, dispatch counts and
reserved device peak. `resident_command_ms` measures resident device commands
with GPU events; enabling it synchronizes each command and adds overhead.
Generation time excludes GLB serialization. External process/device VRAM can
be measured with `ref/pixal3d/run_fixture.py`.

The C API uses a separate versioned `pixal3d_gpu_options` structure initialized
with `pixal3d_default_gpu_options`, then applied with `pixal3d_configure_gpu`
before generation. The original options/result layouts are unchanged. Rebuild
both the host library and GPU plugins together for resident execution.

## CPU fallback

The CPU path uses OpenBLAS plus runtime-dispatched AVX2/F16C pointwise kernels.
Immutable BF16/F16 weights are converted once and consumed directly without a
second no-op rounding copy. Attention score storage is bounded to one query
tile and reused across heads. The retained all-30-block benchmark improves the
recorded 256-token shape-1024 median by 12.5% with byte-identical output; see
[the CPU measurements](OPTIMIZATION.md#cpu-fallback).

## Build

Run from the repository root. Requirements: Linux x86-64, GCC 12+ or a Clang compiler with `_Float16`, C++17, OpenMP,
OpenBLAS development files, Boost JSON headers, and OpenCV core/imgproc/photo
headers with core/imgproc libraries. Shared repository xatlas, FDG and BVH code
is built directly. No CuMesh, spconv, FlexGEMM or NATTEN extension is needed for
native inference.

```sh
mkdir -p tmp/pixal3d
# Ubuntu 24.04: stage OpenCV headers/libraries inside ref/, without sudo.
ref/pixal3d/setup_native.sh
make -C cpu/pixal3d -j4
make -C cuda/pixal3d                 # CUDA 13.2, sm_120 by default
make -C cuda/pixal3d GPU_ARCH=sm_89  # Ada; sm_80 and sm_120 are also supported
make -C cuda/pixal3d check-architectures # compile-check all supported CUDA ISAs
make -C rdna4/pixal3d               # hipcc, gfx1201 by default
make -C cpu/pixal3d test            # C API / analytic math; no weights or GPU needed
```

For system OpenCV development packages, use `make -C cpu/pixal3d OPENCV_ROOT=/usr`.
CUDA toolkit location is selected with `CUDA_PATH` and its target with
`GPU_ARCH=sm_80|sm_89|sm_120`. The explicit BF16/FP16 MMA path requires
Ampere (sm_80) or newer. ROCm uses `HIPCC`, `ROCM_LIB`, and its own
`GPU_ARCH`. The tested AMD installation uses `/opt/rocm/core/lib`. Plugins
are loaded relative to `cpu/pixal3d/libpixal3d.so`, preserving the repository's
`cpu/`, `cuda/` and `rdna4/` directory layout when installing elsewhere.

### Windows CUDA kernel build

Native Windows currently builds and validates the standalone CUDA resident
plugin. From a PowerShell prompt in the repository root, create the local uv
environment and build for an RTX 3070 (`sm_86`) with:

```powershell
powershell -ExecutionPolicy Bypass -File ref\pixal3d\setup_windows.ps1
powershell -ExecutionPolicy Bypass -File ref\pixal3d\build_windows.ps1 `
  -Architecture 86
```

The setup installs PyTorch 2.7.1 CUDA 12.8 and the NVIDIA compiler-support pip
package inside `.venv-pixal3d-cuda`. NVIDIA's CUDA 13.3 compiler, runtime, NVVM,
and cuBLAS redistributable archives are installed without administrator access
under `.cuda\13.3`; this version matches the tested 610.62 driver's reported
maximum CUDA level. Pass `-SkipToolkit` to use an existing complete toolkit and
`-CudaRoot` to select it. The build script compiles `pixal3d_cuda.dll` with MSVC
and runs the resident GEMM, attention, RoPE-cache, allocation, and
error-recovery checks against PyTorch.
The complete CPU host pipeline remains a Linux build because it depends on the
project's Linux OpenBLAS/OpenCV and dynamic-loader path.

This path was validated on `titanv-win10` with an RTX 3070 8 GB, driver 610.62,
MSVC 19.44, CUDA compiler 13.3.73, and PyTorch 2.7.1+cu128. The generated Ninja
rule contains only `compute_86,sm_86`. Resident FP32/BF16/FP16 GEMM and
attention, cached RoPE, bounded allocation reuse, invalid-command recovery, and
pool trimming all passed against PyTorch. Median 12-head BF16 attention times
were 0.373 ms at 1024 tokens and 3.730 ms at 4096 tokens for the retained MMA
kernel, versus 3.781 ms and 25.116 ms for the diagnostic BLAS path.

The complete host pipeline is tested on Windows through WSL2 Ubuntu 24.04. The
PowerShell wrapper stages Ubuntu development packages, pins uv and its Python
packages, and assembles NVIDIA CUDA 13.3 artifacts under the repository without
administrator access. It then builds an `sm_86` Linux plugin and the native host:

```powershell
powershell -ExecutionPolicy Bypass -File ref\pixal3d\setup_windows_wsl.ps1 `
  -Distribution ubu2404
```

Weights remain explicit and are not downloaded by the setup script. The tested
layout was `tmp/pixal3d/models/Pixal3D`,
`tmp/pixal3d/models/dinov3-vitl16/model.safetensors`, and
`tmp/pixal3d/models/naf/naf_release.safetensors`. A monitored generation can be
launched from PowerShell with:

```powershell
wsl.exe -d ubu2404 -- bash /mnt/d/work/gemm/ref/pixal3d/run_windows_wsl.sh `
  --backend cuda --input tmp/pixal3d/windows-generation-inputs/1_img.png `
  --output-dir tmp/pixal3d/windows-3070/house-smoke `
  --model-dir tmp/pixal3d/models/Pixal3D `
  --dinov3 tmp/pixal3d/models/dinov3-vitl16/model.safetensors `
  --naf tmp/pixal3d/models/naf/naf_release.safetensors `
  --fov 0.857556 --seed 42 --threads 8 `
  --gpu-execution resident --gpu-kernels auto `
  --gpu-flow-precision mixed --vram-budget-mib 7168 `
  --texture-size 1024 --triangle-target 250000
```

Three complete single-view runs were retained on the 8 GB RTX 3070 using those
settings. All outputs passed GLB structure, finite geometry, unit-normal, UV,
embedded PBR texture, and connected-component validation:

| Asset | Native / wall time | Shape tokens | Vertices / triangles | Native peak / total-device peak / host RSS | GLB SHA-256 |
|---|---:|---:|---:|---:|---|
| house | 443.379 / 446.730 s | 10,432 | 217,011 / 234,108 | 6.38 / 7.56 / 6.30 GiB | `2868a4d4...730a2a2` |
| crab | 1169.443 / 1174.504 s | 18,274 | 225,875 / 237,990 | 4.54 / 5.80 / 14.17 GiB | `fb6a5859...6452439` |
| jester | 438.041 / 442.508 s | 8,866 | 201,339 / 245,826 | 5.98 / 7.04 / 6.24 GiB | `71303947...dbc2bb2` |

After deleting and reproducibly reassembling the repository-local CUDA 13.3
tree, rebuilding the `sm_86` plugin, and rebuilding the host, the house fixture
completed again in 466.056 / 469.483 seconds. Its GLB was byte-identical to the
first run (`2868a4d4...730a2a2`) and passed the complete validator again. The
native allocator peak remained 6.38 GiB; total device use peaked at 6.60 GiB.

The total-device measurement includes roughly 1 GiB used by the Windows WDDM
desktop. WSL reports per-process NVML memory as unavailable, so the fixture
retains total-device and native allocator peaks separately. The WSL crab run
above predates the memory-bounded resident decoder: at that time inputs above
16,384 shape tokens at an effective budget of 7 GiB or less used the tiled
host-offloaded decoder. The resident decoder now scatters conv1 tiles directly
to their children, accumulates conv2 tiles into the skip buffer, chunks the
ConvNeXt MLP, and normalizes in place, so dense shapes stay resident at 7 GiB
with byte-identical output. The tiled decoder remains only as an automatic
retry when a resident allocation exceeds the budget or device memory.

The inpainting implementation is pinned to OpenCV 4.12 to match the Python
reference. It uses its stable priority heap even when the system libraries are
4.6; the old 4.6 photo implementation is prohibitively slow on some 4096 atlases.

## Weights

The default model paths on this machine are:

- `/mnt/disk2/models/Pixal3D/pipeline.json` and its seven `ckpts/*.safetensors` files;
- `/mnt/disk2/models/dinov3-vitl16/model.safetensors` (timm ViT-L/16 naming);
- `ref/pixal3d/weights/naf_release.safetensors`.

Use `--model-dir`, `--dinov3` and `--naf` for weights under `/mnt/disk1/models` or
another location. The runtime never downloads weights. Prepare the small official
NAF checkpoint and pinned reference sources with the project uv environment:

```sh
ref/pixal3d/setup.sh cpu
ref/pixal3d/run.sh cpu ref/pixal3d/prepare_refs.py
```

The learned flow weights in this release are stored as F32 despite the `bf16`
checkpoint names. The runtime applies BF16 rounding in the flow torso and FP16
rounding in the decoders, preserving F32 input/output layers and normalization.
It does not quantize to INT8 or FP8. Host elementwise operations dispatch to AVX2/F16C
when available, with scalar fallbacks; optional glibc vector math accelerates GELU. C64 RoPE buffers are not learned parameters;
RoPE is reconstructed from coordinates.

## Run

Supply an RGBA image, or an RGB image plus a same-size foreground mask. Supply
horizontal FOV in radians for the cropped input. Foreground cropping, Lanczos
resizing, black compositing and camera projection follow upstream. The native API
keeps these inputs explicit. The project reference helper and web server can add
RMBG-2.0 alpha and estimate FOV/distance with MoGe-2 before invoking the native CLI.

```sh
cpu/pixal3d/pixal3d --backend cuda \
  --input ref/pixal3d/upstream/assets/images/1_img.png \
  --fov 0.857556 --seed 1 --output tmp/pixal3d/house.glb
```

Use a `.ply` output path for a binary little-endian geometry export containing
positions, normals, UVs and indexed triangles. GLB remains the default choice
for rendered assets because it also embeds the generated PBR textures.

Posed multiview inference uses Tencent's `pipeline_mv.json` and `_mv`
checkpoints. Supply a directory containing RGBA frames and a NeRF/Blender-style
`transforms.json`; frame zero defines the output orientation. Each view is
projected with `F * inverse(C0) * Ci`, then global and DINO/NAF grid features
are averaged like the upstream multiview extractor. Calibrated framing is
preserved, so multiview inputs are resized without foreground cropping.

```sh
cpu/pixal3d/pixal3d --backend cuda \
  --views-dir ref/pixal3d/upstream/assets/mv_images/example \
  --output tmp/pixal3d/multiview.glb --seed 42 \
  --gpu-execution resident --gpu-flow-precision mixed --vram-budget-mib 7168
```

Use `--num-views N` to select the first N frames. The C API exposes the same
path through `pixal3d_generate_multiview` and `pixal3d_view`.

The four-view upstream example was validated on the RTX 5060 Ti with resident
execution, mixed precision, seed 42, 4096 textures and the one-million-triangle
target. Under desktop GPU/CPU contention, the 7168 MiB path took 506.207 s in
the native generation timer and 512.737 s wall clock including GLB serialization;
peak native reservation was 6.99 GiB. The 12288 MiB path took 484.915 s and
490.727 s respectively, with an 8.29 GiB native reserved peak. Both produced
the byte-identical `a5a22a90...c700383` GLB with 655,071 vertices and 961,142
triangles and passed mesh, normal, and material validation. These are functional
observations under contention, not isolated throughput measurements.
For 12 GB and larger cards, pass `--vram-budget-mib 12288`. The runtime still
clamps the effective budget to free VRAM minus its safety reserve, so the same
command retains the lower-memory fallback on smaller or contended devices.

Select `--backend rocm` for the RX 9070 XT or `--backend cpu` for CPU execution.
`--device` selects the backend-local device ordinal. `--threads` controls host
OpenMP/OpenBLAS threads. `--distance` overrides the FOV-derived camera distance;
`--mesh-scale` defaults to 1. `--texture-size` selects 1024, 2048, or the
quality-preserving 4096 default; `--triangle-target` accepts 10000–5000000 and
defaults to one million. All execution choices are explicit arguments.

The pipeline executes DINOv3/NAF conditioning, 12-step sparse structure diffusion,
structure decoding, 512 shape diffusion, four learned subdivisions, the 1024
shape/texture cascade, guided sparse decoders, FDG extraction, hole filling,
CuMesh-style UDF remeshing, midpoint QEM simplification, UV charting, sparse PBR
baking and GLB export. Output uses 4096 base-color/alpha and metallic/roughness
textures embedded as PNG. Metal is B and roughness G in the material texture;
GLB coordinates apply the combined upstream rotation `(-x,y,-z)`. Vertex normals
are area-weighted; exactly cancelling contributions use an incident face direction,
normalized for glTF output.

## Memory and numerical behavior

The default workspace budget is 12 GiB (`--vram-budget-mib 12288`). At creation,
it is reduced to available device memory minus 512 MiB. GEMMs stream rows and
attention tiles queries while retaining every key. Resident execution keeps
NN weights and activations on the GPU within each model stage, then releases
stage allocations before loading the next model. Geometry and PBR processing
run on CPU. The historical `legacy` path instead transfers operation inputs and
outputs across PCIe to bound device workspaces; its 28–97 minute functional runs
under concurrent validation load, including GLB serialization, are retained in
the [historical execution table](../../ref/pixal3d/README.md#historical-host-offloaded-full-pipeline-runs)
only to document that tradeoff.

No resolution fallback or token truncation is used. A workspace allocation that
cannot fit fails with a diagnostic. The reported `peak_device_bytes` covers
native scratch allocations over the context lifetime; `peak_host_bytes` uses
process-lifetime peak RSS. The generation timer excludes final GLB serialization.
Driver/BLAS overhead and complete invocation time are measured separately by
`ref/pixal3d/run_fixture.py`. Host RAM use depends strongly on decoded voxel count.

The seed controls a fixed native Box-Muller generator shared by all backends.
PyTorch uses a different RNG, so comparison scripts replay native noise tensors.
Floating-point reductions differ between BLAS backends. Values close to zero
can change occupancy/subdivision decisions: seeds do not guarantee bit-identical
meshes across CPU/CUDA/ROCm. Reference tests separately measure continuous feature
error and exact coordinate expansion with shared subdivision decisions.
Isolated checkpoint-stage tests pass their 2% NRMSE bound. The recommended
mixed mode also passes complete twelve-step comparisons against pinned FP32
PyTorch for all four stages: NRMSE ranges from `1.09e-6` to `7.11e-5`. Pure
BF16 accumulated structure/shape trajectories remain diagnostic and can differ
substantially across PyTorch CUDA/ROCm backends. See the measured comparisons in
the [reference validation record](../../ref/pixal3d/README.md).

Use `--dump-dir DIR` to save preprocessing, conditioning, noise, every diffusion
step, decoded outputs, and the FDG/remeshed/simplified meshes as safetensors for reference replay. Dumps can occupy
several GiB and are disabled by default. Standard output contains final JSON
counts, timing and peak memory; progress and errors go to standard error.

## C API

Initialize options with `pixal3d_default_options`, set backend/model paths, create
one context, then call `pixal3d_generate` and optionally `pixal3d_write_glb` or
`pixal3d_write_ply`.
Release results with `pixal3d_result_free`, and contexts with `pixal3d_destroy`.
Input data is borrowed for the duration of the call; results own their buffers.
Zero-initialize each result, and free its buffers before passing it to another generation.
Use separate contexts serially: OpenBLAS/OpenMP thread counts are process-global.
`pixal3d_last_error(NULL)` reports creation errors; other errors use the context.

See [reference validation](../../ref/pixal3d/README.md) and
[third-party notices](../../ref/pixal3d/THIRD_PARTY.md).
