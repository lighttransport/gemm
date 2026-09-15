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

Add `--gpu-execution resident --gpu-kernels auto` to retain NN activations and
packed weights on the GPU. The default remains `legacy`. `auto` uses vendor
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
make -C rdna4/pixal3d               # hipcc, gfx1201 by default
make -C cpu/pixal3d test            # C API / analytic math; no weights or GPU needed
```

For system OpenCV development packages, use `make -C cpu/pixal3d OPENCV_ROOT=/usr`.
CUDA toolkit location is selected with `CUDA_PATH`; ROCm with `HIPCC`, `ROCM_LIB`
and `GPU_ARCH`. The tested AMD installation uses `/opt/rocm/core/lib`. Plugins
are loaded relative to `cpu/pixal3d/libpixal3d.so`, preserving the repository's
`cpu/`, `cuda/` and `rdna4/` directory layout when installing elsewhere.

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
resizing, black compositing and camera projection follow upstream. Automatic
BiRefNet masking and MoGe camera estimation are outside this first implementation.

```sh
cpu/pixal3d/pixal3d --backend cuda \
  --input ref/pixal3d/upstream/assets/images/1_img.png \
  --fov 0.857556 --seed 1 --output tmp/pixal3d/house.glb
```

Select `--backend rocm` for the RX 9070 XT or `--backend cpu` for CPU execution.
`--device` selects the backend-local device ordinal. `--threads` controls host
OpenMP/OpenBLAS threads. `--distance` overrides the FOV-derived camera distance;
`--mesh-scale` defaults to 1. All execution choices are explicit arguments.

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

The default workspace budget is 14 GiB (`--vram-budget-mib 14336`). At creation,
it is reduced to available device memory minus 512 MiB. GEMMs stream rows and
attention tiles queries while retaining every key. Weights/activations live in
host memory between operations, and stages release model allocations before the
next stage. Geometry and PBR processing run on CPU. This trades host RAM and PCIe
traffic for bounded VRAM; it is not a fully device-resident fast path.
Recorded complete GPU invocations took 28–97 minutes under concurrent validation
load; the narrow CPU pencil fixture took 63 minutes. Observed host RSS ranged
from 6.05 to 13.80 GiB. See the [execution table](../../ref/pixal3d/README.md#full-pipeline-runs)
for fixture sizes, memory counters and measurement conditions.

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
Isolated checkpoint-stage tests pass their 2% NRMSE bound, but accumulated BF16
structure/shape trajectories exceed it. Original PyTorch CUDA/ROCm trajectories
also differ; full-trajectory equivalence is not established. See the measured
comparisons in the [reference validation record](../../ref/pixal3d/README.md).

Use `--dump-dir DIR` to save preprocessing, conditioning, noise, every diffusion
step, decoded outputs, and the FDG/remeshed/simplified meshes as safetensors for reference replay. Dumps can occupy
several GiB and are disabled by default. Standard output contains final JSON
counts, timing and peak memory; progress and errors go to standard error.

## C API

Initialize options with `pixal3d_default_options`, set backend/model paths, create
one context, then call `pixal3d_generate` and optionally `pixal3d_write_glb`.
Release results with `pixal3d_result_free`, and contexts with `pixal3d_destroy`.
Input data is borrowed for the duration of the call; results own their buffers.
Zero-initialize each result, and free its buffers before passing it to another generation.
Use separate contexts serially: OpenBLAS/OpenMP thread counts are process-global.
`pixal3d_last_error(NULL)` reports creation errors; other errors use the context.

See [reference validation](../../ref/pixal3d/README.md) and
[third-party notices](../../ref/pixal3d/THIRD_PARTY.md).
