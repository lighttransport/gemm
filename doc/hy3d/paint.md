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
**2026-05-10**: end-to-end paint chain runs natively on CUDA with the
PyTorch reference only required for development / per-stage diffing.

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
5. ✅ **Multiview SD-2.1 UNet** — `cuda_paint_unet_runner.h` (~2.1k lines)
   + `paint_stage_unet.c`. UniPC scheduler at 30 steps × 3-chunk CFG =
   90 forwards per chain. TC dispatch: BF16 attention (FA2-style
   `flash_attn_bf16_hd64*`), BF16 GEMM (`gemm_bf16_v7`), and native FP8
   e4m3 GEMM (`gemm_fp8_v7_fused` + `_p2`). Per-chunk reference-attention
   cache. Beff=12 (N_pbr=2 × N_gen=6). All bit-identical against the
   PyTorch trace at the chain output (mask_mismatch=4558, max_diff=0.43).
6. ✅ **SD VAE encoder + decoder** — `cuda_paint_vae_runner.h` +
   `paint_stage_vae.c`. Same TC dispatch playbook as UNet (FP8/BF16
   GEMM, FP8 conv via im2col). Encode 13 conditioning views + ref
   image; decode 12 latent → 12×512² RGB views.
7. 🟡 **Real-ESRGAN x4** — PyTorch reference dump landed
   (`ref/hy3d/dump_realesrgan.py`, commit `d817d5e`). Produces
   `/tmp/hy3d_resrgan/resrgan_{input,output,rdb0}.npy` (16.7M params,
   `[1,3,128,128]→[1,3,512,512]`). Native CUDA RRDBNet runner pending —
   needs only `conv2d_f32` + `leaky_relu_f32` kernels on top of the
   existing common ops. Optional for the base pipeline (chain runs
   without it).
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

## End-to-end pipeline runner

`test_paint_pipeline.c` is the unified harness. The `chain` subcommand
runs the full mesh → textured-OBJ flow:

```
./test_paint_pipeline chain <mesh.obj> <dinov2g.safetensors> <dinov2g_input.npy> \
    <unet_wrapper.safetensors> <unet_inputs_dir> \
    <vae.safetensors> <bp_ref_dir> <out_dir>
```

Per-stage wall-clock is printed via `[chain-time] <stage>: %.3fs`.
`CHAIN_STEPS=N` and `CHAIN_CFG={2,3}` env vars override the defaults
(30 steps, 3-chunk CFG).

## Performance status (2026-05-10, RTX 5060 Ti / sm_120)

End-to-end 30-step CFG-3 chain on `fujisan` input, 6 views @ 512²:

| Stage           | Wall (s) | % of total | Notes |
|-----------------|---------:|-----------:|-------|
| view_maps       |     0.11 |       0.1% | trivial |
| dinov2g         |     5.14 |       5.3% | encoder forward |
| vae_encode      |     3.58 |       3.7% | 13 encodes |
| **unet**        | **82.12**|  **84.7%** | 90 forwards (30 × CFG-3) |
| vae_decode      |     5.81 |       6.0% | 12 latents → RGB |
| xatlas+raster   |     0.08 |       0.1% | trivial |
| back_project    |     0.05 |       0.1% | trivial (GPU bake) |
| inpaint+write   |     0.01 |       0.0% | trivial (CPU) |
| **TOTAL**       | **96.91**|     100%   | bit-identical (mask_mismatch=4558) |

PyTorch reference: ~50.6 s end-to-end. Remaining gap: **1.92×**.

The gap is dominated by the UNet (85% of wall). Per-kernel profile
(PAINT_PROFILE=1, 4-step):

| Kernel               | Total ms | Calls  | avg µs | Share |
|----------------------|---------:|-------:|-------:|------:|
| FA2 attention (`fn`) |   3 652  |  7 776 |    470 |  33%  |
| FP8/BF16 GEMM (`f`)  |   2 516  | 16 251 |    155 |  23%  |
| reduce_max_abs_f32   |   1 043  | 11 031 |     94 |   9%  |
| group_norm_f32       |     819  |  8 964 |     91 |   8%  |
| quantize_fp8_e4m3    |     731  | 11 031 |     66 |   7%  |
| im2col_3x3_p1        |     639  |  7 053 |     91 |   6%  |
| (others)             |   1 540  | 32 946 |     ~  |  14%  |

## Tensor-core dispatch toggles

All TC paths are env-gated and default ON. Set to `0` to disable:

| Env var                  | Path                                        | Default |
|--------------------------|---------------------------------------------|--------:|
| `PAINT_BF16_ATTN`        | UNet attention via FA2 BF16 (hd=64)         | 1       |
| `PAINT_BF16_GEMM`        | UNet linears via `gemm_bf16_v7`             | 1       |
| `PAINT_FP8_GEMM`         | UNet BF16-pipe FP8 fallback (`gemm_fp8_mt4`)| 1       |
| `PAINT_FP8_CONV`         | UNet 3×3 conv via im2col + FP8 GEMM         | 1       |
| `PAINT_FP8_V7`           | UNet native e4m3 fused GEMM                 | 1       |
| `PAINT_FP8_V7_P2`        | UNet 2×2 panel variant of v7                | 1       |
| `PAINT_VAE_BF16_GEMM`    | VAE BF16 GEMM                               | 1       |
| `PAINT_VAE_FP8_GEMM`     | VAE BF16-pipe FP8                           | 1       |
| `PAINT_VAE_FP8_V7`       | VAE native e4m3 fused GEMM                  | 1       |
| `PAINT_VAE_FP8_V7_P2`    | VAE 2×2 panel variant of v7                 | 1       |
| `PAINT_VAE_FP8_CONV`     | VAE 3×3 conv via im2col + FP8 GEMM          | 1       |
| `PAINT_VAE_FP8_MT4`      | VAE 4-tile FP8 GEMM dispatch                | 1       |
| `PAINT_PROFILE`          | per-kernel cuEvent timing summary on exit   | 0       |
| `PAINT_FP8_DEBUG`        | per-launch sync + abort-on-error            | 0       |
| `PAINT_FA_BKV64`         | UNet attention BKV=64 variant (scaffold)    | 0       |
| `PAINT_UNET_GRAPH`       | CUDA graph capture for UNet forward         | 0       |

Disabling individual paths is useful for bisecting numeric drift; the
`bf16` and `fp8` paths fall through to each other and finally to the
F32 pipe so any combination is valid.

## Remaining optimization opportunities

The cheap levers are exhausted. Each remaining option is high-effort
relative to its expected wall-clock impact, ordered roughly by ROI:

1. **Warp-specialized FA2 (Plan A).** The attention kernel is the single
   largest cost (33% of UNet kernel time). Plans B (3-stage cp.async at
   BKV=64) and C (register-diet by fused P-pack) were both tried and
   landed bit-identical but performance-neutral (B was actually 29%
   slower). Confirmed the kernel is MMA-throughput-bound, not load- or
   register-bound. The remaining lever is producer/consumer warp split:
   1 warp continuously cp.async-loads K/V while 3 warps run MMA without
   ever blocking on the load. Design notes live in
   `~/.claude/projects/.../memory/project_hy3d_paint_fa2.md`. Estimated
   upside: 15–25% on the attention kernel = 5–8% wall.
2. **CUDA graph capture for the UNet forward**
   (`PAINT_UNET_GRAPH=1`). Wired and tested; gave ~3% wall in earlier
   measurements. Currently OFF by default because the kernels are
   compute-bound and the saving is small relative to the capture
   complexity. Worth re-measuring once Plan A lands (lower kernel cost
   raises launch-overhead share).
3. **Step-count reduction via a faster scheduler.** UniPC at 30 steps
   is the largest single multiplier on UNet wall. DPM++ 2M Karras at
   ~15 steps is a candidate; LCM-LoRA-style distillation (4–8 steps)
   would be much bigger but requires retraining. **Quality constraint:**
   user-visible PSNR drops below 45 dB at half-step counts on this
   pipeline are unacceptable, so any scheduler change must be paired
   with a quality measurement pass.
4. **Drop CFG-3 to CFG-2.** Saves 33% of UNet calls. Risk: changes the
   conditioning structure (uncond / ref / full) → the ref-attention
   cache layout and chain output both shift. Earlier 2-chunk experiment
   landed but was reverted to 3-chunk for production parity.
5. **Stream overlap across CFG chunks.** All 90 forwards are currently
   serialized on stream 0 with a per-chunk RA cache. Issuing the three
   chunks of step `t+1` on separate streams while step `t`'s VAE work
   runs would overlap latency only — the GPU is already 99% busy on
   the kernel mix, so this is unlikely to move the needle.
6. **Shrink reduce_max_abs + quantize_fp8 launch count.** 22k launches /
   1.77 s combined. A grid-cap experiment made the kernel 36% faster in
   isolation but slowed UNet wall by 5 s — shorter critical-path
   kernels reduced overlap with concurrent ops. A safer win would fuse
   the two (single launch, shared-memory max → quantize) but requires a
   grid-level barrier (cooperative groups).
7. **Real-ESRGAN x4 upsample.** Optional stage; currently skipped. If
   added, it becomes a noticeable share of wall (small CNN, 16.7 M
   params per view × 6 views).
8. **Multi-GPU.** Out of scope for the 5060 Ti port.

The FA2 BKV=64 variant (`flash_attn_bf16_hd64_xq_b64`, opt-in via
`PAINT_FA_BKV64=1`) is kept in source as the scaffold for the Plan A
warp-specialized port — it gets the BKV layout closer to FA2's
canonical hd=64 shape but is 5% slower on its own. Don't ship it on
by default.

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

The native pipeline is feature-complete. Future work is the optimization
queue above (Plan A FA2 warp-specialization is highest ROI) plus the
optional Real-ESRGAN x4 upsample for the high-res variant of the chain.
