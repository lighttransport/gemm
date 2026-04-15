# Hunyuan3D-2.1 Texture Gen — CUDA port plan

The shape-gen stage (`cuda/hy3d/`) is feature-complete and matches the
PyTorch reference within F32 noise. The texture-gen stage (`hy3dpaint`
upstream) is orders of magnitude more work — multiple non-trivial neural
networks, a custom triangle rasterizer, UV unwrapping, texture baking,
and PBR post-processing.

The plan here is a **hybrid, staged port**: native C/C++ for the mesh /
geometry / rasterization bits that are small and well-specified, and the
existing PyTorch reference for the heavy ML components until we have time
to port them. Each piece is listed with its target backend and a
"Python first" vs "CUDA first" label.

## Pipeline overview

Upstream (`hy3dpaint/textureGenPipeline.py`):

```
shape mesh (+ style image)
   │
   ▼
┌──────────────────────────────┐
│ mesh remesh (trimesh +       │   [Python, skip if already decimated]
│ fast_simplification)         │
└──────────────────────────────┘
   │
   ▼
┌──────────────────────────────┐
│ UV unwrap (xatlas)           │   [C++ NATIVE, vendored]
└──────────────────────────────┘
   │
   ▼
┌──────────────────────────────┐
│ select N camera views        │   [C++ native, trivial]
│ render normal + position     │   [CUDA native — rewrite of
│ maps per view                │    upstream custom_rasterizer]
└──────────────────────────────┘
   │
   ▼
┌──────────────────────────────┐
│ DINOv2-giant image encoder   │   [Python first → CUDA later,
│ (condition)                  │    same ops as hy3d shape-gen DINOv2]
└──────────────────────────────┘
   │
   ▼
┌──────────────────────────────┐
│ Multiview SD-2.1 UNet        │   [Python first → CUDA later,
│ 15 UniPC steps, 6 views,     │    this is the biggest chunk]
│ produces albedo + MR         │
└──────────────────────────────┘
   │
   ▼
┌──────────────────────────────┐
│ VAE decoder (SD-2.1)         │   [Python first → CUDA later]
└──────────────────────────────┘
   │
   ▼
┌──────────────────────────────┐
│ Real-ESRGAN x4 SR            │   [Python first → CUDA later,
│ per view                     │    small CNN, eventually CUDA]
└──────────────────────────────┘
   │
   ▼
┌──────────────────────────────┐
│ back-projection bake         │   [CUDA native — new kernel]
│ (camera-space -> UV space)   │
└──────────────────────────────┘
   │
   ▼
┌──────────────────────────────┐
│ texture inpaint              │   [C++/CUDA native — rewrite of
│ (fill uncovered UV islands)  │    mesh_inpaint_processor]
└──────────────────────────────┘
   │
   ▼
OBJ + MTL + albedo/metal/rough JPGs
```

## Vendored dependencies (native)

| Lib | Location | License | Status |
|---|---|---|---|
| [xatlas](https://github.com/jpcy/xatlas) | `common/xatlas.{h,cc}` | MIT | vendored |
| `custom_rasterizer` (upstream) | `cuda/hy3d_paint/rasterizer.cu` | Tencent non-commercial | to rewrite (native) |
| `mesh_inpaint_processor.cpp` | `cuda/hy3d_paint/inpaint.cu` | Tencent non-commercial | to rewrite (native) |

### xatlas

Header-only-ish (one `xatlas.h` + one `xatlas.cc`). Vendored as-is at
`common/xatlas.h` + `common/xatlas.cc`. No external deps. API:

```c++
xatlas::Atlas *atlas = xatlas::Create();
xatlas::AddMeshError e = xatlas::AddMesh(atlas, mesh_decl);
xatlas::Generate(atlas);
// Atlas->width, height, meshes[...].chartArray, meshes[...].vertexArray
// Each vertex has .uv[2], .xref (original index).
xatlas::Destroy(atlas);
```

First binary consumer: `cuda/hy3d_paint/test_uv_unwrap.c(c)` which loads an
OBJ from disk and writes an OBJ with unwrapped UVs.

### Why not reuse `custom_rasterizer` directly

The upstream `custom_rasterizer` (cuda/hy3d_paint/custom_rasterizer_kernel)
is built as a PyTorch `CUDAExtension`. Every entry point consumes and
returns `torch::Tensor`. Using it from native C/C++ would require linking
libtorch (~500 MB) and wrapping every call through a Python shim — which
defeats the purpose of the port.

The actual rasterization kernel is ~100 lines of `calculateBarycentric` +
per-pixel Z-test. We'll lift that into a plain CUDA C kernel compiled via
NVRTC (same pattern as `cuda/hy3d/cuda_hy3d_kernels.h`), consuming raw
`CUdeviceptr` inputs. The Python `custom_rasterizer` installed in the uv
venv remains available as the PyTorch-reference baseline for numerical
comparison.

## Python deps for the ML-heavy parts

`ref/hy3d/pyproject.toml` already has everything needed for the upstream
reference runner:

- `torch`, `diffusers`, `transformers`, `pytorch-lightning`,
  `accelerate`, `omegaconf`, `safetensors`
- `xatlas` (Python bindings — used by the reference; native C++ is the
  vendored copy in `common/`)
- `basicsr`, `realesrgan`, `fast_simplification`, `trimesh`, `pymeshlab`,
  `opencv-python`, `Pillow`, `einops`, `timm`, `pygltflib`
- `pybind11` (for the in-place `mesh_inpaint_processor` build)

The upstream `custom_rasterizer` + `mesh_inpaint_processor` C extensions
are built in-place into the repo by `ref/hy3d/setup_paint_native.sh`
(see `ref/hy3d/run_texturegen.py` header for build notes).

`run_texturegen.py` already supports `--trace-dir`, dumping per-stage
tensors (DINO input/output, 15 UNet steps, VAE decoder in/out) for
layer-by-layer diffing against a future CUDA port.

## CUDA port sequence

Earliest → hardest. Each step has a small, testable slice.

1. **xatlas UV unwrap, native C++ binary** — `cuda/hy3d_paint/test_uv_unwrap.cc`
   - Loads OBJ via a minimal reader (or `common/stb_*.h` equivalent)
   - Calls `xatlas::AddMesh` / `Generate`
   - Writes OBJ with per-face UV indices
   - Verification: round-trip against the Python `xatlas` binding
2. **Native triangle rasterizer** — `cuda/hy3d_paint/cuda_raster_kernels.h`
   - Ports `rasterizeTriangleCPU` + the `rasterizeImage` GPU wrapper as an
     NVRTC-compiled kernel taking `CUdeviceptr pos/tri/out`
   - First output: 512×512 depth + triangle-index buffer for a single view
   - Verification: compare against PyTorch `custom_rasterizer.rasterize`
3. **Per-view normal + position maps** — use the rasterizer + barycentric
   interpolation to produce `[V, H, W, 3]` normal maps and position maps
4. **DINOv2-giant encoder** — reuse the existing `cuda/hy3d` DINOv2-L
   runner with:
   - `hidden_size = 1536`, `num_heads = 24`, `num_layers = 40`,
     `intermediate_size = 4096` (dinov2-giant specifics)
   - Same attention / LayerScale / GELU kernels, just different weight
     sizes
5. **Multiview SD-2.1 UNet** — the big one. SD-2.1 backbone (unet_2d
   condition) with custom multiview cross-attention. Architecture is
   similar to flux2 / qwen-image in this repo, but with different block
   types and an extra multiview attention layer. Plan: stand up a new
   runner `cuda/hy3d_paint/cuda_paint_unet.*` iteratively, one block
   type at a time, using `run_texturegen.py --trace-dir` for per-step
   reference.
6. **SD VAE decoder** — small, similar to the shape-gen ShapeVAE.
   Probably reusable from an existing SD runner (qimg / flux2).
7. **Real-ESRGAN x4** — RRDBNet. Small enough to port as its own runner.
8. **Back-projection bake** — native CUDA kernel consuming the rasterized
   UV maps + RGB views and scattering colors into the UV texture.
9. **Texture inpaint** — port `mesh_inpaint_processor.cpp` to a native
   CUDA kernel; it's a simple diffusion-based fill over the uncovered
   pixels in the UV atlas.

The Python runner continues to work in parallel throughout; each CUDA
step swaps in one piece at a time and is diffed against the trace
dump from `run_texturegen.py --trace-dir`.

## File layout

```
common/
  xatlas.h                       [vendored, MIT]
  xatlas.cc                      [vendored, MIT]
  hunyuan3d_paint.h              [CPU API + constants, to write]

cuda/hy3d_paint/
  Makefile
  test_uv_unwrap.cc              [xatlas native smoke test]
  cuda_hy3d_paint_kernels.h      [NVRTC kernels: raster, bake, inpaint]
  cuda_hy3d_paint_ops.h          [op wrappers / ops table]
  cuda_hy3d_paint_runner.h       [public runner API]
  cuda_hy3d_paint_runner.c       [runner impl, stages wired up]
  test_cuda_hy3d_paint.c         [end-to-end CLI]

ref/hy3d/
  run_texturegen.py              [PyTorch reference + --trace-dir, done]

doc/hy3d/
  paint.md                       [this file]
```

## First milestone

Working `test_uv_unwrap` that loads `/tmp/hy3d_ref.obj` (the shape-gen
output, unwelded 233k verts / 78k tris) and emits an OBJ with UV
coords. Success metric: resulting atlas width/height agree with the
Python `xatlas` binding to within the library's own non-determinism,
and the file can be loaded into the Python texturegen pipeline as a
drop-in replacement for its own UV unwrap output.
