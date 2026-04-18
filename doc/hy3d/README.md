# Hunyuan3D-2.1 (hy3d) — CUDA Runner Status

Single-image-to-3D pipeline based on Tencent's **Hunyuan3D-2.1**. Runs
end-to-end on a single CUDA device with NVRTC-compiled kernels (no `nvcc`
required). F16 weights on the GPU, F32 compute, optional F32-GEMM escape hatch
for bit-level PyTorch parity.

This document captures the current state of the implementation and the
reproduction / verification procedure against a PyTorch reference.

## Pipeline

| Stage | Module | Input | Output | Notes |
|---|---|---|---|---|
| 1 | Preprocess | RGB uint8 `[h,w,3]` | `[3,518,518]` f32 | resize + ImageNet normalize |
| 2 | DINOv2-L encoder | `[3,518,518]` | `[1370,1024]` | 24 blocks, CLS + 1369 patch tokens |
| 3 | DiT flow matching | `[4096,64]` noise + ctx | `[4096,64]` latent | 21 blocks, MoE on 15–20, U-Net skips 11–20, Euler + CFG |
| 4 | ShapeVAE decoder | `[4096,64]` latent | SDF grid `[G,G,G]` | post-KL proj + 16 blocks + Fourier cross-attn geo decoder |
| 5 | Marching cubes | SDF grid | vertices + triangles | writes OBJ/PLY |

Model constants (from `common/hunyuan3d.h:42`):

- **DINOv2-L**: hidden 1024, heads 16, layers 24, ffn 4096, patch 14, img 518, seq 1370
- **DiT**: hidden 2048, ctx 1024, depth 21, heads 16, latent `[4096, 64]`, ffn 8192
  - timestep token prepended (seq 4096 → 4097, stripped at output)
  - MoE: 8 experts + shared, top-2, blocks 15–20
  - U-Net skip connections: blocks 11–20
- **ShapeVAE**: latent width 1024, 16 blocks, 4096 tokens, 8 Fourier freqs (51-dim embed)

## Weights

Default location used by this doc and the updated `setup_ref.sh`:

```
/mnt/disk01/models/Hunyuan3D-2.1/
├── hunyuan3d-dit-v2-1/
│   ├── config.yaml
│   └── model.fp16.ckpt              # ~6.1 GB, combined DiT + conditioner
├── hunyuan3d-vae-v2-1/
│   ├── config.yaml
│   └── model.fp16.ckpt              # ShapeVAE (NOTE: .ckpt, not .safetensors)
├── conditioner.safetensors           # exported from dit ckpt
├── model.safetensors                 # exported from dit ckpt
└── vae.safetensors                   # exported from vae ckpt
```

**Download (correct HuggingFace repo is `tencent/Hunyuan3D-2.1`, not `Hunyuan3D-2`):**

```bash
hf download tencent/Hunyuan3D-2.1 \
  --local-dir /mnt/disk01/models/Hunyuan3D-2.1 \
  --include "hunyuan3d-dit-v2-1/*"
hf download tencent/Hunyuan3D-2.1 \
  --local-dir /mnt/disk01/models/Hunyuan3D-2.1 \
  --include "hunyuan3d-vae-v2-1/*"
```

Then export the combined `.ckpt` files into per-component `.safetensors`
with `ref/hy3d/export_safetensors.py`.

## File map

| Path | Role |
|---|---|
| `common/hunyuan3d.h` | CPU API surface + constants (inference not implemented) |
| `cpu/hy3d/test_hy3d.c` | weight-load smoke test (CPU) |
| `cuda/hy3d/cuda_hy3d_runner.h` | public runner + verify API |
| `cuda/hy3d/cuda_hy3d_runner.c` | implementation (1904 lines) |
| `cuda/hy3d/cuda_hy3d_kernels.h` | NVRTC kernel source strings |
| `cuda/hy3d/cuda_hy3d_ops.h` | kernel function-pointer table |
| `cuda/hy3d/test_cuda_hy3d.c` | end-to-end CLI: image → OBJ |
| `cuda/hy3d/verify_dinov2.c` | stage 2 verifier vs `ref/hy3d/output/dinov2_*.npy` |
| `cuda/hy3d/verify_dit.c` | stage 3 single-step verifier |
| `cuda/hy3d/verify_vae.c` | stage 4 verifier (blocks + SDF grid) |
| `cuda/hy3d/setup_ref.sh` | environment + weight bootstrap (thin wrapper) |
| `cuda/hy3d/Makefile` | builds `test_cuda_hy3d` + `verify_*` |
| `ref/hy3d/` | standalone PyTorch reference scripts (mirrors `ref/qwen_image/`) |

Key line anchors in `cuda_hy3d_runner.c`:

- `load_dino_weights` @ 453, `load_dit_weights` @ 521, `load_vae_weights` @ 638
- `run_dinov2` @ 740, `run_dit_moe` @ 850, `run_dit_forward` @ 983
- `run_vae_block` @ 1370, `run_shapevae` @ 1437
- pipeline entry `cuda_hy3d_predict` @ 1684

## Build

```bash
cd cuda/hy3d
make                 # builds test_cuda_hy3d
make verify          # builds verify_dinov2, verify_dit, verify_vae
```

Only `gcc`, `libcuda.so`, and `libnvrtc.so` are required. No `nvcc`.

## Run

```bash
./test_cuda_hy3d \
  /mnt/disk01/models/Hunyuan3D-2.1/conditioner.safetensors \
  /mnt/disk01/models/Hunyuan3D-2.1/model.safetensors \
  /mnt/disk01/models/Hunyuan3D-2.1/vae.safetensors \
  -i input.ppm -o output.obj --ply output.ply \
  -s 30 -g 7.5 --grid 256 --seed 42
```

## Reference / verification framework

Layout mirrors `ref/qwen_image/`:

```
ref/hy3d/
├── pyproject.toml              # uv venv (torch, safetensors, transformers, diffusers, trimesh, einops, ...)
├── export_safetensors.py       # split hunyuan3d-dit-v2-1/model.fp16.ckpt → conditioner.safetensors + model.safetensors
├── export_vae_safetensors.py   # convert hunyuan3d-vae-v2-1/model.fp16.ckpt → vae.safetensors
├── dump_dinov2.py              # DINOv2 forward → dinov2_input.npy, dinov2_output.npy, hidden_{0,12,23,24}
├── dump_dit_single_step.py     # DiT single forward @ t=0.5 → dit_input_latents.npy, dit_output.npy, dit_timestep_embed.npy
├── dump_vae.py                 # post-KL + 16 blocks + Fourier SDF → vae_input_latents.npy, vae_post_kl.npy, vae_block_{0,8,15}.npy, vae_decoded_latents.npy, vae_sdf_grid.npy
├── compare.py                  # ref/test .npy comparison (rtol=1e-3 atol=1e-4)
└── output/                     # generated .npy fixtures (gitignored)
```

### Reproduction procedure

```bash
# 1. Create venv (once)
cd ref/hy3d
uv sync

# 2. Export safetensors (once, after download)
uv run python export_safetensors.py \
  --ckpt /mnt/disk01/models/Hunyuan3D-2.1/hunyuan3d-dit-v2-1/model.fp16.ckpt \
  --outdir /mnt/disk01/models/Hunyuan3D-2.1
uv run python export_vae_safetensors.py \
  --ckpt /mnt/disk01/models/Hunyuan3D-2.1/hunyuan3d-vae-v2-1/model.fp16.ckpt \
  --outdir /mnt/disk01/models/Hunyuan3D-2.1

# 3. Dump PyTorch reference tensors
uv run python dump_dinov2.py \
  --ckpt /mnt/disk01/models/Hunyuan3D-2.1/hunyuan3d-dit-v2-1/model.fp16.ckpt \
  --outdir output
uv run python dump_vae.py \
  --vae-path /mnt/disk01/models/Hunyuan3D-2.1/vae.safetensors \
  --grid-res 8 --outdir output
uv run python dump_dit_single_step.py \
  --ckpt /mnt/disk01/models/Hunyuan3D-2.1/hunyuan3d-dit-v2-1/model.fp16.ckpt \
  --outdir output

# 4. Build & run CUDA verifiers
cd ../../cuda/hy3d
make verify
./verify_suite.sh \
  --models-dir /mnt/disk01/models/Hunyuan3D-2.1 \
  --ref-dir ../../ref/hy3d/output \
  --profile both

# 5. Optional: DiT trajectory spot-checks (selected steps)
#    For deterministic parity against a local PyTorch trace, reuse the traced
#    step-0 latents/context instead of relying on CUDA's local RNG path.
./test_cuda_hy3d \
  /mnt/disk01/models/Hunyuan3D-2.1/conditioner.safetensors \
  /mnt/disk01/models/Hunyuan3D-2.1/model.safetensors \
  /mnt/disk01/models/Hunyuan3D-2.1/vae.safetensors \
  -i input.ppm -o /tmp/hy3d_cuda.obj -s 30 \
  --init-trace-dir /tmp/hy3d_trace \
  --dump-latent-steps 1,15,30 \
  --dump-latent-prefix /tmp/hy3d_latent/dit_latent_step \
  --dump-velocity-steps 1,15,30 \
  --dump-velocity-prefix /tmp/hy3d_latent/dit_velocity_step

cd ../../ref/hy3d
uv run python compare_trajectory.py \
  /tmp/hy3d_trace /tmp/hy3d_latent \
  --steps 1,15,30 --cuda-prefix dit_latent_step
uv run python compare_trajectory.py \
  /tmp/hy3d_trace /tmp/hy3d_latent \
  --steps 1,15,30 --cuda-prefix dit_velocity_step --kind velocity
```

`--init-trace-dir` expands to:

- `04_dit_latents_step0.npy` for the initial latent tensor
- `03_dit_context_cfg.npy` for the conditional/unconditional CFG context

Use this mode when you want DiT trajectory parity independent of the currently
non-matching CUDA-vs-PyTorch initial RNG path.

`verify_suite.sh` writes `cuda_output/f16/` and `cuda_output/f32/` and calls
`ref/hy3d/compare.py --manifest ref/hy3d/verify_manifest.json` so pass/fail is
stage-specific (`max_abs` + `mean_abs`) instead of one global `allclose`.

## Completed vs pending

| Stage | CUDA | Verifier | PyTorch ref | Notes |
|---|---|---|---|---|
| Preprocess (resize+normalize) | ✅ | — | — | kernel `resize_normalize` |
| DINOv2-L encoder | ✅ | ✅ `verify_dinov2` | ✅ `dump_dinov2.py` | 24 blocks, layerscale, patch embed |
| DiT forward | ✅ | ✅ `verify_dit` | ✅ `dump_dit_single_step.py` | skip 11–20, MoE 15–20, timestep token |
| DiT flow-matching loop + CFG | ✅ | ⚠️ spot-check | ⚠️ partial | trajectory compare via selected latent dumps |
| ShapeVAE decoder | ✅ | ✅ `verify_vae` | ✅ `dump_vae.py` | post-KL + 16 blocks + Fourier geo decoder |
| Marching cubes | ✅ | — | — | `common/marching_cubes.h` |
| F32 GEMM fallback | ✅ | — | — | `cuda_hy3d_set_f32_gemm` |
| End-to-end mesh diff vs PyTorch | ❌ | — | ❌ | needs `run_full_pipeline.py` |
| CPU reference (`common/hunyuan3d.h`) | ❌ | — | — | weight load only |
| HIP / Vulkan backends | (mirrored) | — | — | out of scope for this doc |

## Known numerical issues & recent fixes

- **QKV head-interleaving** (fixed `bf2a97e`, `c5a5ae3`): DiT self- and
  cross-attention were reshaping the fused QKV tensor in `[heads, 3, head_dim]`
  order while the ckpt weights are `[3, heads, head_dim]`. The fused GEMM +
  buffer aliasing path also tripped over the same issue. Both paths now match
  PyTorch layout.
- **F32 GEMM escape hatch** (`039b988`): `cuda_hy3d_set_f32_gemm(r, 1)` keeps
  all weights in F32 so GEMM rounding doesn't contaminate stage comparisons.
  Use when a verifier reports `FAIL` at `~1e-3` and you want to confirm the
  structure is right before chasing precision.
- **DINOv2/VAE dtype + LayerNorm smem** (`3f28748`): early bug where F16/F32
  path split and LayerNorm shared-memory sizing were wrong; fixed.

## Known issues in `setup_ref.sh` (to fix)

- Uses wrong HuggingFace repo `tencent/Hunyuan3D-2` (has only v2.0 variants).
  Correct repo is **`tencent/Hunyuan3D-2.1`**.
- Assumes VAE ships as `.safetensors`; upstream actually ships `model.fp16.ckpt`.
  Needs an explicit export step for the VAE too.
- Embeds all Python scripts as bash heredocs; should call into `ref/hy3d/` instead.
- `MODELS_DIR` default hardcoded to `/mnt/nvme02/...`; should default to
  `/mnt/disk01/models/Hunyuan3D-2.1`.

## Verification status

Reference regenerated against `/mnt/disk01/models/Hunyuan3D-2.1` on
**2026-04-16** with the repo-committed `ref/hy3d/` scripts and an RTX 5060 Ti
(sm_120), F16 weights + F32 compute.

| Stage | max abs err | mean abs err | target (<1e-4 mean) |
|---|---|---|---|
| DINOv2 final `[1370,1024]` | 2.42e-02 | 6.44e-04 | ❌ accumulation-limited (see below) |
| DiT timestep embed `[2048]` | 5.96e-08 | 1.95e-09 | ✅ |
| DiT single-step output `[4096,64]` | **5.36e-03** | **2.03e-05** | ✅ |
| VAE SDF grid `[8,8,8]` | 1.82e-04 | 4.80e-06 | ✅ |

### Investigation history

Initial numbers before this pass: DINOv2 8.91e-03, DiT 1.53e-01. Root causes:

1. **Tanh-approx GELU vs PyTorch exact GELU** (affects DINOv2 + DiT).
   The shared `gelu_f32` kernel in `cuda/cuda_kernels_common.h` used the
   tanh approximation `0.5x(1+tanh(√(2/π)(x+0.044715x³)))`, but PyTorch's
   `nn.GELU()` (DINOv2's timm `Mlp`, DiT's `nn.GELU()`, diffusers'
   `FeedForward(activation_fn="gelu")`) all default to the exact erf form.
   The per-layer mismatch compounds across 24 blocks. **Fix**: added a
   new `gelu_exact_f32` kernel (`erff(x/√2)`-based) and have hy3d's ops
   table load it into the `gelu` slot instead of `gelu_f32`. No other
   model consumers touched.
2. **Buffer-aliasing corruption in `run_dit_moe`** (affects only DiT MoE
   blocks 15–20). The MoE call site aliased `d_moe_out` (the accumulator
   that receives expert-scaled outputs) with `d_moe_scratch` base
   (`scratch[1]`), and the per-expert fc1 output (`d_exp_h`, 134 MB at
   offset 131 KB) overlapped the first ~33 MB of the accumulator. The
   expert fc1 GEMM therefore *clobbered* the partial sum from previous
   experts, then the host-side read-modify-write re-uploaded corrupted
   data. Per-block diagnostic dumps (below) localised this as a step
   from mean err 2.6e-05 at block 14 → 1.81e-01 at block 15. **Fix**:
   pass `d_attn` (scratch[2], idle during the MLP phase) as
   `d_moe_out` instead of reusing scratch[1]. Documented inline at
   `cuda/hy3d/cuda_hy3d_runner.c:1360`.

After both fixes, DiT per-block error holds at the F32 accumulation floor
(≤4e-5 mean across blocks 0–20), and final output is 2.03e-05 mean.

### Why DINOv2 does not reach <1e-4 mean

DINOv2-L runs 24 LayerNorm + Attention + MLP + LayerScale + residual
blocks over a sequence that develops well-documented register-token
outliers (`|x|` up to ~400 in the last 3 blocks — see
`ref/hy3d/output/dinov2_hidden_23.npy`). Each F32 GEMM / softmax /
LayerNorm reduction uses a summation order that differs from PyTorch's
eager/SDPA kernels, contributing ~1e-7 relative error per op. Over 24
blocks with register-token amplification, the accumulated mean abs err
saturates around 6e-4 regardless of F16 vs F32 weights
(`cuda_hy3d_set_f32_gemm(r, 1)` gave **8.2e-04**, slightly *worse*
than F16 mode at 6.44e-04). Reaching <1e-4 would require reproducing
PyTorch's kernel reduction order bit-for-bit, which is impractical.

Per-hidden-state error progression (after fixes, F16 weights):

| checkpoint | max abs err | mean abs err |
|---|---|---|
| `hidden_0` (post patch+pos+cls) | 1.26e-04 | 6.44e-06 |
| `hidden_12` (after 12 blocks) | 8.57e-04 | 6.03e-05 |
| `hidden_23` (after 23 blocks) | 4.73e-02 | 7.76e-04 |
| `hidden_24` (after 24 blocks)  | 9.00e-02 | 1.23e-03 |
| `output` (post final LN)       | 2.42e-02 | 6.44e-04 |

### Debugging hooks

Both `run_dinov2` and `run_dit_forward` in
`cuda/hy3d/cuda_hy3d_runner.c` call `hy3d_dbg_dump_npy()` at key
checkpoints. These are **no-ops by default** — set `HY3D_DUMP_DIR=/tmp/x`
to enable `.npy` dumps matching the PyTorch reference file names for
`compare.py`.
- VAE internal block compares (`vae_post_kl`, `vae_block_{0,8,15}`,
  `vae_decoded_latents`) are dumped by `ref/hy3d/dump_vae.py` but the
  current `verify_vae.c` only compares the final `vae_sdf_grid`. If we need
  per-block error, extend `verify_vae.c` to load those extra `.npy` files.
- DiT forward-pass error is dominated by F16 GEMM rounding across 21 blocks
  + 6 MoE layers. Max abs err ~2.0 on activations with `|mean|<0.5` is
  consistent with the per-layer rounding budget; mean abs err 0.15 is what
  actually governs downstream flow-matching quality. For tighter bounds,
  rerun with `--f32` (uses `cuda_hy3d_set_f32_gemm(r, 1)`).
- To regenerate the DiT single-step reference, clone the upstream repo at
  `github.com/Tencent-Hunyuan/Hunyuan3D-2.1` and run:

  ```bash
  HY3D_REPO=/path/to/Hunyuan3D-2.1-repo/hy3dshape \
    .venv/bin/python dump_dit_single_step.py \
      --ckpt /mnt/disk01/models/Hunyuan3D-2.1/hunyuan3d-dit-v2-1/model.fp16.ckpt \
      --outdir output
  ```

  `HY3D_REPO` must point at the dir that contains the `hy3dshape` package
  (i.e., the intermediate `hy3dshape/` subdirectory, not the repo root).
  The script stubs `hy3dshape.__init__` so heavy optional deps
  (`pymeshlab`, `gradio`, ...) are not required; only `timm`, `omegaconf`,
  `opencv-python`, and the existing ref/hy3d deps are needed (all already
  in `pyproject.toml`).
