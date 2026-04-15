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

Earliest → hardest. Each step has a small, testable slice. Status as of
**2026-04-15**:

1. ✅ **xatlas UV unwrap** — `cuda/hy3d_paint/test_uv_unwrap.cc`. Matches
   Python `xatlas` binding on 2084 charts / 54413 verts.
2. ✅ **Native triangle rasterizer** — `cuda/hy3d_paint/cuda_paint_raster_kernels.h`
   (`rasterize_faces_f32`, `resolve_bary_f32`, `interpolate_attr_f32`).
   `test_raster.c` reports 96086/96087 face-id match vs PyTorch
   `custom_rasterizer.rasterize`. Uses int64 atomicMin packed
   `(depth_quant, face_id+1)` tokens.
3. ✅ **Per-view normal + position maps** — `test_view_maps.c`, 6-view
   render at F32 noise floor on 5/6 views.
4. ✅ **DINOv2-giant encoder** — `test_dinov2_giant.c` (~680 lines).
   `hidden=1536, heads=24, layers=40, ffn_half=4096, seq=257`. Reuses
   shape-gen kernels + new `split_silu_gate_f32` (SwiGLU) from
   `cuda_paint_nn_kernels.h`. Host-side Catmull-Rom bicubic pos_embed
   interpolation 37×37→16×16 (bilinear gave 6.3e-4, bicubic gives
   8.3e-8). Final output mean abs err **1.26e-03** (0.086% relative);
   same accumulation profile as DINOv2-L.
5. ⏳ **Multiview SD-2.1 UNet** — pending (biggest piece). Reference
   trace already available via `run_texturegen.py --trace-dir`.
6. ⏳ **SD VAE decoder** — pending.
7. 🟡 **Real-ESRGAN x4** — PyTorch reference dump landed
   (`ref/hy3d/dump_realesrgan.py`, commit `d817d5e`). Produces
   `/tmp/hy3d_resrgan/resrgan_{input,output,rdb0}.npy` (16.7M params,
   `[1,3,128,128]→[1,3,512,512]`). Native CUDA RRDBNet runner pending —
   needs only `conv2d_f32` + `leaky_relu_f32` kernels on top of the
   existing common ops.
8. ✅ **Back-projection bake** — `test_back_project.c` +
   `back_project_sample_f32` kernel. Round-trip F32 noise floor at
   three Htex/Himg ratios.
9. ✅ **Texture inpaint** — `mesh_vertex_inpaint.h` (header-only with
   `MESH_VERTEX_INPAINT_IMPLEMENTATION`). Methods: `MVI_SMOOTH`
   (1/d² weighted neighbour averaging, 2-pass), `MVI_FORWARD` (BFS).
   `test_vertex_inpaint.cc` reports byte-identical match vs upstream
   pybind `mesh_inpaint_processor`.

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
  test_uv_unwrap.cc                [xatlas native, done]
  cuda_paint_raster_kernels.h      [raster + bary + attr interp + back-project]
  cuda_paint_nn_kernels.h          [SwiGLU split_silu_gate; conv2d/leaky_relu pending]
  mesh_vertex_inpaint.h            [header-only, vertex graph diffusion fill]
  test_raster.c                    [rasterizer vs custom_rasterizer]
  test_view_maps.c                 [6-view normal+position maps]
  test_back_project.c              [UV-space bake]
  test_vertex_inpaint.cc           [vs upstream pybind]
  test_swiglu.c                    [split_silu_gate_f32]
  test_dinov2_giant.c              [DINOv2-giant runner + verify]
  cuda_hy3d_paint_runner.{h,c}     [TODO: end-to-end runner wiring]
  test_cuda_hy3d_paint.c           [TODO: end-to-end CLI]

ref/hy3d/
  run_texturegen.py              [PyTorch reference + --trace-dir, done]

doc/hy3d/
  paint.md                       [this file]
```

## First milestone — ✅ done

`test_uv_unwrap` matches the Python `xatlas` binding on the shape-gen
output mesh.

## Next milestone

Native CUDA RRDBNet runner consuming `/tmp/hy3d_resrgan/resrgan_*.npy`.
After that, the only remaining ML stages are the multiview SD-2.1 UNet
and the SD VAE decoder.
