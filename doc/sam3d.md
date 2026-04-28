# SAM 3D Objects — Current State and TODOs

Snapshot: 2026-04-28.

## What is sam-3d-objects?

FAIR's [sam-3d-objects](https://github.com/facebookresearch/sam-3d-objects)
is an image → 3D-object reconstruction model. It takes a single RGBA
image plus a foreground mask and produces a **3D Gaussian Splat**
(`.ply`) and, optionally, a textured mesh (`.glb`).

Architecturally it is **not** a SAM-3 successor — it is a two-stage
latent flow-matching pipeline very close to TRELLIS.2:

1. **DINOv2-L/14+reg** image encoder (RGB and mask channels) +
   **PointPatchEmbed** over a MoGe-derived pointmap, fused by a
   Llama-style SwiGLU **CondEmbedderFuser** into ~2740 conditioning
   tokens.
2. **SS Flow DiT** (24 transformer blocks, multi-modality MOT shape +
   pose streams, cross-attn over cond tokens) integrates a shortcut
   ODE → 8-channel sparse-structure latent on a 16³ grid.
3. **SS-VAE decoder** (3D-conv + groupnorm + 2× pixel-shuffle
   upsample, 16³ → 64³) → 64³ occupancy logits.
4. Voxel-prune (`occ > 0`) → **SLAT Flow DiT** (sparse 3D + full
   sparse self-attention) integrates structured-latent features at
   the active voxels.
5. **SLAT GS decoder** (sparse transformer + per-voxel decode head)
   emits 32 Gaussians per active voxel as
   `(xyz_offset, dc_sh_color, scaling_log, quat_rotation, opacity_logit)`
   in INRIA-PLY layout.
6. PLY writer.

This document mirrors the CUDA port in `cuda/sam3d/`. The CPU sibling
lives at `cpu/sam3d/`; both validate per-stage numerics against the
same `/tmp/sam3d_ref/*.npy` dumps produced by
`ref/sam3d/gen_image_ref.py`.

## Supported backends

Two parallel runners exist; both expose the same per-stage entry
points and accept the same converted safetensors checkpoint.

| Backend | Path                          | Status                                                                 |
|---------|-------------------------------|------------------------------------------------------------------------|
| CPU     | `cpu/sam3d/`                  | All stages green vs pytorch refs. Reference for the CUDA port.         |
| CUDA    | `cuda/sam3d/` (this document) | NVRTC runner (cuew, no nvcc). DINOv2 + CondFuser + SS Flow DiT (forward + shortcut ODE) GPU-resident; SS-VAE decoder GPU-resident; SLAT prune and SLAT transformer stack are GPU-resident; SLAT sparse IO resblocks and SLAT GS decoder still use CPU fallbacks being kernelized stage by stage. |

CUDA precision is configurable at `cuda_sam3d_create` time via
`cfg.precision = "fp16" | "bf16" | "fp32"` (currently the GPU-resident
stages all run fp32; F16 / BF16 tensor-core paths land alongside the
remaining kernelization phases). MoGe depth, SLAT mesh decoder,
texture bake, layout optimization, `.glb` export, and any
server / web integration are deferred — see the TODOs section.

## Pipeline

```
RGBA image + mask
    │
    ├─ preprocess (resize 518² + ImageNet norm; mask channel)
    ├─ pointmap (precomputed from MoGe; supplied via --pointmap *.npy in v1)
    │
    ▼
DINOv2-L/14 + 4 register tokens   (24 blocks · D=1024 · 1374 tokens)
    │   tokens (1374, 1024) for RGB and mask, concatenated/projected
    │
    ▼
PointPatchEmbed + Llama SwiGLU CondEmbedderFuser
    │   cond tokens (N_c≈2740, D=1024)
    │
    ▼
SS Flow DiT  (24 blocks · MOT shape+pose streams · cross-attn over cond)
    │   shortcut ODE → ss_latent (NCDHW [8,16,16,16])
    │
    ▼
SS-VAE decoder  (3D-conv + groupnorm + pixel-shuffle upsample)
    │   occupancy logits 64³
    │
    ▼
Voxel prune (occ > 0) → active voxel coords
    │
    ▼
SLAT Flow DiT  (sparse 3D + full sparse self-attn)
    │   slat tokens (N, out_ch)
    │
    ▼
SLAT GS decoder  (sparse transformer + per-voxel decode head; 32 gaussians/voxel)
    │   per-gaussian rows [N·32, 17] in INRIA PLY layout
    │
    ▼
.ply (gs_ply_writer.h)
```

## Layout

```
cpu/sam3d/                          # CPU port + CLI + verify_*.c
cuda/sam3d/                         # CUDA port (NVRTC) + verify_*.c (this doc)
common/dinov2.h                     # DINOv2-B/L/14+reg ViT (subset of dinov3.h)
common/sam3d_cond_fuser.h           # PointPatchEmbed + Llama SwiGLU fuser
common/sam3d_ss_flow_dit.h          # SS Flow DiT (CPU host reference)
common/sam3d_shortcut_solver.h      # shortcut ODE schedule + Euler step
common/trellis2_ss_decoder.h        # reused SS-VAE 3D-conv decoder
common/sparse3d.h                   # sparse 3D + shift-window attn
common/sam3d_slat_dit.h             # SLAT Flow DiT
common/sam3d_gs_decoder.h           # SLAT GS decoder
common/gs_ply_writer.h              # INRIA gsplat PLY writer
ref/sam3d/gen_image_ref.py          # pytorch ref dump
/mnt/disk01/models/sam3d/safetensors/   # converted ckpt
```

## Status — drift table

Per-stage gates are GREEN. `max_abs` / `mean_abs` are vs the pytorch
ref or CPU host (as noted). Phase 1b (DINOv2), Phase 2b (CondFuser),
Phase 2c (SS Flow DiT), and Phase 4b (SS-VAE decoder) are GPU-resident.
Phases 5 (SLAT DiT) and 6 (SLAT GS) currently route through the CPU
fallback in the runner.

| Stage          | Path | Precision | max_abs (vs ref/CPU) | Status |
|----------------|------|-----------|----------------------|--------|
| dinov2         | GPU  | fp32      | 2.02e-04             | **Phase 1b.8 GREEN** — full GPU forward via `cs3d_dinov2_gpu_forward`; ~48× over CPU at fp32 |
| cond_fuser     | GPU  | fp32      | 1.48e-05             | **Phase 2b.8e GREEN** — PPE on-device + 3× fuser projection on GPU |
| ss_dit (single)| GPU  | fp32      | 1.05e-05             | **Phase 2c.13 GREEN** — `cs3d_ssdit_outer_forward` via runner |
| ss_dit (ODE)   | GPU  | fp32      | 2.19e-05 @ steps=2   | **Phase 2c.16 GREEN** — upstream sampler semantics: rescale_t=3, reversed=0, d=0, add_flag CFG |
| ss_decoder     | GPU  | fp32      | 2.77e-04             | **Phase 4b GREEN** — 3D conv + channel-LN + SiLU + pixel-shuffle GPU forward; ~558 ms verifier path |
| slat_dit       | mix  | fp32      | 4.41e-05             | **Phase 5b.22 GREEN** — hybrid SLAT ODE body plus real-weight CUDA gates for all SLAT IO resblocks; runner sparse IO wiring still pending |
| slat_gs        | CPU  | fp32      | 7.77e-05             | **Phase 6a GREEN** — CPU fallback. Phase 6b kernelization pending |
| End-to-end PLY | mix  | fp32      | visual (GS viewer)   | **Phase 7a GREEN** — full sampled path writes 38016 gaussians in 126.92–128.21 s after 5b.17/5b.18; `--slat-ref` smoke writes 8192 gaussians in 2.77 s |

## Phase summaries

### Phase 0 — scaffold (GREEN)

NVRTC-based runner mirroring `cpu/sam3d/sam3d_runner.h`:
`cuda_sam3d_runner.{h,c}`, `cuda_sam3d_kernels.h`, CPU-fallback TU
`sam3d_cpu.{c,h}` (mirrors `sam3d_body_cpu.c` pattern), CLI
`test_cuda_sam3d.c`, regression gate `verify_dinov2.c`.

### Phase 1a / 2a — DINOv2 + CondFuser CPU fallback (GREEN)

Each stage's runner entrypoint calls into `sam3d_cpu.c` and mirrors
output into device buffers; debug-override hooks short-circuit
readback so a single `verify_*.c` isolates that stage's drift.

### Phase 1b — DINOv2 NVRTC kernelization (GREEN)

Sub-stages, all green:

| # | Sub-stage | Kernel(s) |
|---|-----------|-----------|
| 1b.0 | weight upload | `cs3d_dinov2_gpu_load` (1.16 GiB on device) |
| 1b.1 | patch_embed | `dinov2_patch_embed_f32` (Conv2d ps=14) |
| 1b.2 | prepend + pos_embed | `dinov2_prepend_cls_reg_f32` + `dinov2_add_pos_embed_f32` |
| 1b.3 | LN | `layernorm_token_f32` (foundation, reused everywhere) |
| 1b.4a | gemm foundation | `gemm_f32_bias` |
| 1b.4b | SDPA + qkv split | `qkv_split_f32` + `sdpa_f32` |
| 1b.5 | LayerScale + residual | `layerscale_add_f32` |
| 1b.6 | MLP | `gelu_inplace_f32` (exact erf) + reuse `gemm_f32_bias` |
| 1b.7a | single block | `cs3d_dinov2_block_forward` composed launcher |
| 1b.7b | full forward | `cs3d_dinov2_gpu_forward` (24 blocks + final LN) — **GPU 2.96 s vs CPU 141 s** |
| 1b.8 | runner swap | `cuda_sam3d_run_dinov2` lazy-loads; CPU model retained for preprocessing only |

### Phase 2b — CondEmbedderFuser kernelization (GREEN)

Llama SwiGLU per-modality fuser + heavier PointPatchEmbed (per-pixel
linear with NaN-invalid replacement → window pack → 1× ViT block →
CLS extract + window pos add). New kernels: `silu_mul_f32`,
`ppe_linear3_invalid_f32`, `ppe_window_pack_f32`, `sdpa_batched_f32`,
`ppe_cls_pos_extract_f32`. Composed forwards in
`cuda_sam3d_ppe_forward.h` and `cuda_sam3d_fuser_forward.h`; weights
in `cuda_sam3d_ppe_gpu.h` (10.2 MiB) and `cuda_sam3d_fuser_gpu.h`
(88.0 MiB). End-to-end `verify_cond_fuser` max_abs=1.48e-5 vs ref.

### Phase 2c — SS Flow DiT kernelization (GREEN)

Sub-stages, all green:

| # | Sub-stage | Notes |
|---|-----------|-------|
| 2c.0 | timestep + silu primitives | `timestep_embed_cossin_f32`, `silu_inplace_f32` |
| 2c.1 | t/d-embedder MLP | composed |
| 2c.2 | AdaLN modulation | `silu` + gemm to 6D, contiguous split |
| 2c.3 | modulated LN | `modulated_ln_f32` (LN no-affine + (1+scale)·x + shift) |
| 2c.4 | multi-head RMSNorm | L2 normalize over D_h then γ·sqrt(D_h); supports stride>H·D_h for QKV-packed rows |
| 2c.5 | MOT self-attn | shape-attends-shape; pose-attends-concat([pose,shape]) via DtoD KV concat (no kernel) |
| 2c.6 | MOT cross-attn | new `kv_split_f32` for interleaved [N_c, 2D] |
| 2c.7 | FFN | new `gelu_tanh_inplace_f32` (tanh-approx, distinct from 1b.6's exact erf) |
| 2c.8 | latent in/out projection | composed; pos_emb pre-baked into ckpt |
| 2c.9 | full single-block forward | new `gated_residual_add_f32` (per-feature gate broadcast) |
| 2c.10 | weight loader | `cuda_sam3d_ssdit_gpu.h` — 3.59 GiB on device, round-trip bit-exact |
| 2c.11 | real-weights single-block | `cuda_sam3d_ssdit_forward.h` — block 0 vs CPU host: max_abs ≤ 1.68e-4 |
| 2c.12 | full GPU forward | `cuda_sam3d_ssdit_outer.h` — 24 blocks + 5-modality I/O proj; max_abs=5.72e-6 vs CPU |
| 2c.13 | runner wiring | `cuda_sam3d_debug_ss_dit_forward` calls GPU directly; `verify_ss_dit` max_abs=1.05e-5 |
| 2c.14 | GPU shortcut ODE | byte-identical xorshift64* + Box-Muller; per-step GPU forward + host Euler; SHAPE→NCDHW perm |
| 2c.15 | one-shot cond upload | `cond=NULL` semantic = "already on device"; `cs3d_ssdit_outer_upload_cond` once per ODE; saves N_steps × 11 MiB HtoD |
| 2c.16 | sampler parity | matches upstream pointmap inference: non-reversed rescaled schedule, `d=0`, and zero-cond `add_flag` CFG; `verify_ss_dit_ode` max_abs=2.19e-5 |

### Phase 3a / 4a / 5a / 6a — downstream stages CPU fallback (GREEN)

`cuda_sam3d_run_ss_decode` (3D-conv decoder → 64³ occupancy logits),
`cuda_sam3d_run_slat_dit` (voxel prune → flow_matching loop on slat
latents → SLAT un-norm), `cuda_sam3d_run_slat_gs_decode` (sparse
transformer + APE + 12 windowed self-attn blocks + per-voxel decode
head → `[N·32, 17]` PLY-layout rows). All four mirror the
debug-override pattern so each `verify_*.c` isolates its own drift.

### Phase 4b — SS-VAE decoder kernelization (GREEN)

SS decoder weights are dequantized from the loaded CPU model and
uploaded by `cuda_sam3d_ss_decoder_gpu.h` (281 MiB on device).
`cuda_sam3d_ss_decoder_forward.h` composes `conv3d_k3_pad1_f32`,
`channel_layernorm_3d_f32`, `silu_inplace_f32`,
`pixel_shuffle_3d_f32`, and `residual_add_f32` for the full decoder.
`cuda_sam3d_run_ss_decode` now calls this GPU path and keeps the host
occupancy mirror for downstream CPU fallback and `verify_ss_decoder`.

Current gate: `verify_ss_decoder` max_abs=2.770424e-04,
mean_abs=3.038599e-05 vs `/tmp/sam3d_ref/ss_dec_out.npy`; forward time
on RTX 5060 Ti is ~558 ms. The 5D conv weights require sizing via
`qtensor.n_rows * qtensor.n_cols`; using `qt_numel()` truncates the
fifth dimension and causes device OOB.

### Phase 7a — end-to-end CLI (GREEN)

`test_cuda_sam3d` parses
`--safetensors-dir`, `--pointmap`, `--seed`, `--steps`, `--slat-steps`,
`--cfg`, `-o splat.ply`, `--device`, `--precision`, `-v`, plus
`--slat-ref` for SLAT-GS-decode-only bypass. Bypass smoke writes 8192
gaussians in 2.77 s. Full pipeline on the pytorch ref input now chains
through the GPU SS decoder, reports `pos=1188` occupancy voxels, runs
hybrid SLAT DiT plus GS fallback, and writes 38016 gaussians to
`/tmp/cuda_sam3d_5b17_timed.ply` in 126.92 s.

The full sampled path required matching upstream SS generator sampler
semantics: `rescale_t=3`, `reversed_timestamp=False`, `no_shortcut=True`
(`d=0`), and `unconditional_handling=add_flag`. In this C/CUDA port
`add_flag` is represented by a second zero-conditioned SS DiT forward
inside the CFG interval `[0,500]`, then velocity mixing
`(1+w)*cond - w*uncond`.

## TODOs

### Next phase — Phase 5b

Continue SLAT Flow DiT kernelization beyond the 5b.22 hybrid
CPU-sparse-IO / GPU-transformer ODE body with resident cond and hook
scratch reuse.
The current end-to-end path is correct enough for pipeline/perf smoke
but still uses CPU fallback for SLAT sparse IO resblocks and SLAT GS
decode.

### Phase 5b — SLAT Flow DiT kernelization

Sparse 3D + full sparse self-attn. Phase 5b.0 moves voxel pruning to a
deterministic GPU argwhere over 64³ occupancy logits and passes the
resulting `(b,z,y,x)` coords into the existing CPU ODE body. Next
Phase 5b.1 adds the SLAT absolute positional embedding CUDA primitive
(`slat_ape_add_f32`): N=1188, dim=1024, filled=1020,
max_abs=8.374e-06, mean_abs=3.035e-07, avg=0.0145 ms over 200
launches. Phase 5b.2 adds deterministic sparse downsample factor-2 with
`include_self=True` mean semantics
(`slat_downsample2_mean_include_self_serial_f32`): N=1188, C=128,
out_N=1165, max_abs=0, mean_abs=0, avg=45.5210 ms over 200 launches.
The 5b.2 implementation is intentionally serial to preserve CPU
first-occurrence coordinate ordering; it is a correctness bridge, not
the final parallel compaction path. Phase 5b.3 adds target-order
nearest-neighbor sparse upsample (`slat_upsample2_nearest_f32`):
src_N=1176, target_N=1188, C=2048, max_abs=0, mean_abs=0, avg=0.3862
ms over 200 launches. Phase 5b.4 adds submanifold sparse Conv3d
(`slat_build_coord_index64_i32` + `slat_submconv3x3_f32`) for fixed
output sparse coords and weight layout `[out_C,27,in_C]`: N=1188,
Cin=128, Cout=128, max_abs=1.192e-07, mean_abs=9.126e-09, avg=0.2609
ms over 20 launches. Next sub-stages should peel off a whole SLAT
resblock and then transformer pieces while preserving sparse
coordinate ordering. Phase 5b.5 pins SLAT full sparse self-attention
for batch=1 using `qkv_split_f32` + `sdpa_f32`: N=1188, dim=1024,
H=16, D_h=64, max_abs=1.192e-07, mean_abs=7.983e-09, SDPA avg=17.9921
ms over 20 launches. Phase 5b.6 pins SLAT cross-attention core using
`kv_split_f32` + `sdpa_f32`: N=1188, N_cond=2740, dim=1024, H=16,
D_h=64, max_abs=2.161e-07, mean_abs=6.685e-09, SDPA avg=41.9248 ms
over 10 launches. FFN / modulated-LN reuse Phase 2c kernels; next
Phase 5b.7 composes the no-up/down identity-skip IO SparseResBlock3d
path from existing CUDA primitives: affine LN → SiLU → SubMConv3d →
modulated LN → SiLU → SubMConv3d → residual add. Test shape
N=1188, C=128, max_abs=2.384e-07, mean_abs=1.847e-08, avg=0.5319 ms
over 20 launches. Phase 5b.8 composes the SLAT transformer MLP residual
path: modulated LN → fc1 → tanh-GELU → fc2 → gated residual add.
Correctness uses production `dim=1024, hidden=4096` at N=128 to keep
the scalar CPU reference practical: max_abs=1.073e-06,
mean_abs=1.134e-07, avg=6.3216 ms over 10 launches. Next step is
wiring real-weight IO resblock slices or composing a full single-stream
SLAT transformer block. Phase 5b.9 composes the SLAT self-attention
residual path: modulated LN → QKV projection → Q/K multi-head RMSNorm
→ full self-attn → output projection → gated residual. Correctness uses
production `dim=1024, H=16, D_h=64` at N=128: max_abs=1.192e-07,
mean_abs=9.934e-09, avg=3.3442 ms over 10 launches.
Phase 5b.10 composes the SLAT cross-attention residual path: affine LN
→ Q projection, cond KV projection → KV split → cross SDPA → output
projection → residual add. Correctness uses production `dim=1024,
H=16, D_h=64` at N=128, N_cond=512: max_abs=1.192e-07,
mean_abs=5.959e-09, avg=8.5170 ms over 10 launches.
Phase 5b.11 composes a complete single-stream SLAT transformer block:
self-attn residual → cross-attn residual → MLP residual. Correctness
uses production `dim=1024, H=16, D_h=64, hidden=4096` at N=64,
N_cond=256: max_abs=1.013e-06, mean_abs=1.174e-07, avg=9.1691 ms over
5 launches.
Phase 5b.12 adds a real-weight block-0 verifier
(`verify_slat_transformer_block_realw`) using traced post-APE activations
`c_h_after_ape.npy`, checkpoint weights, traced cond tokens, and traced
`c_h_after_block_0.npy` as reference. Shape N=1007, N_cond=1374,
dim=1024, H=16, D_h=64, hidden=4096; max_abs=1.220703e-04,
mean_abs=3.961771e-06, avg=131.9640 ms over 3 launches. This pins the
real checkpoint block path before hoisting SLAT block state into a
persistent GPU model.
Phase 5b.13 extends that verifier with `--stack`, uploading all real
block weights once and running blocks 0..23 on the same device-resident
activation buffer before comparing to `c_h_after_block_23.npy`. Shape
N=1007, N_cond=1374, dim=1024, H=16, D_h=64, hidden=4096; max_abs=
4.821777e-03, mean_abs=4.799369e-05, avg=3168.8317 ms for one full
24-block stack. The stack gate uses threshold 1e-2 because per-block
CUDA `expf`/GEMM reduction-order differences accumulate across 24
blocks.
Phase 5b.14 extracts the verifier-local transformer block device weight
state into reusable `cuda_sam3d_slat_dit_gpu.h`
(`cs3d_slatdit_gpu_load_transformer` / `cs3d_slatdit_gpu_free`). The
real-weight verifier now consumes this persistent GPU model container
instead of dequantizing/uploading block weights through local structs.
Post-refactor gates: single block max_abs=1.220703e-04,
mean_abs=3.961771e-06, avg=132.0270 ms over 3 launches; 24-block stack
max_abs=4.821777e-03, mean_abs=4.799369e-05, avg=3175.5491 ms over 1
launch.
Phase 5b.15 extracts the verifier-local transformer launch sequence into
reusable `cuda_sam3d_slat_dit_forward.h`. The forward driver provides
`cs3d_slatdit_fns_lookup`, reusable block workspace allocation/free, a
single-block forward, and a multi-block stack forward over the persistent
`cs3d_slatdit_gpu` weights. The real-weight verifier is now a thin
harness around that driver. Post-refactor gates: single block
max_abs=1.220703e-04, mean_abs=3.961771e-06, avg=131.9374 ms over 3
launches; 24-block stack max_abs=4.821777e-03,
mean_abs=4.799369e-05, avg=3167.2253 ms over 1 launch. Existing
`verify_slat_dit` remains green at max_abs=1.668930e-05,
mean_abs=1.609899e-06.
Phase 5b.16 wires the reusable transformer stack into the CUDA runner
through a narrow hook in `common/sam3d_slat_dit.h`: the existing CPU
forward still owns input layer, sparse input/output resblocks, APE,
final LN, and output layer, while the hook uploads post-APE activations
and runs `cs3d_slatdit_stack_forward` over the persistent GPU weights.
`verify_slat_dit` now exercises this hybrid path and remains green:
max_abs=4.410744e-05, mean_abs=4.311766e-06. The standalone 24-block
stack verifier remains unchanged: max_abs=4.821777e-03,
mean_abs=4.799369e-05, avg=3166.6134 ms over 1 launch. Full sampled
CUDA E2E with `--steps 2 --slat-steps 12 --seed 42` writes 38016
gaussians to `/tmp/cuda_sam3d_5b16_timed.ply` in real=128.73 s
(`/usr/bin/time -p`).
Phase 5b.17 reuses the runner-owned `d_cond_tokens` buffer inside the
SLAT transformer hook when the normal E2E path supplies resident
conditioning tokens, avoiding one cond HtoD upload per SLAT ODE step.
Verifier override buffers keep the previous upload fallback. The
numeric gate is unchanged (`verify_slat_dit` max_abs=4.410744e-05,
mean_abs=4.311766e-06). Full sampled CUDA E2E writes 38016 gaussians to
`/tmp/cuda_sam3d_5b17_timed.ply` in real=126.92 s
(`/usr/bin/time -p`).
Phase 5b.18 keeps the SLAT hook activation and timestep device buffers
allocated in `cuda_sam3d_ctx`, growing them only if a later call needs
more capacity. This removes per-step `cuMemAlloc`/`cuMemFree` churn for
`d_x` and `d_t_emb`; cond-token behavior remains 5b.17's resident-buffer
fast path plus verifier upload fallback. Numerics are unchanged:
`verify_slat_dit` max_abs=4.410744e-05, mean_abs=4.311766e-06. Full
sampled CUDA E2E writes 38016 gaussians to
`/tmp/cuda_sam3d_5b18_timed.ply` in real=128.21 s; the wall time is
within run variance and not a proven throughput gain.
Phase 5b.19 adds `verify_slat_input_block0_realw`, a real-checkpoint
CUDA gate for the first SLAT IO SparseResBlock3d. It starts from traced
`c_h_after_input_layer.npy`, uses traced `c_t_emb.npy` and checkpoint
`input_blocks[0]` weights, then compares to
`c_h_after_input_block_0.npy`. This pins the no-up/down, identity-skip
real sparse IO block before moving it into the runner. Gate:
N=1024, C=128, dim=1024, max_abs=5.722046e-06,
mean_abs=2.371230e-07, avg=0.4729 ms over 20 launches. The verifier
also fixes a 5D conv-weight upload hazard by sizing dequantized conv
weights with `qtensor.n_rows * qtensor.n_cols`, not `qt_numel()`.
Phase 5b.20 adds `verify_slat_input_block1_realw`, a real-checkpoint
CUDA gate for the downsampling SLAT input SparseResBlock3d. It starts
from traced `c_h_after_input_block_0.npy` /
`c_coords_after_input_block_0.npy`, applies the deterministic factor-2
mean downsample, runs the real `input_blocks[1]` skip projection and
two submanifold convs, then compares features and coordinates to
`c_h_after_input_block_1.npy` /
`c_coords_after_input_block_1.npy`. Gate against the regenerated current
trace in `/tmp/sam3d_ref_5b20`: N0=1024, N1=1007, C_in=128,
C_out=1024, dim=1024, max_abs=7.390976e-06,
mean_abs=3.324221e-07, coord_bad=0, avg=65.4651 ms over 20 launches.
The older `/tmp/sam3d_ref` copy is stale for this intermediate: coords
match but `c_h_after_input_block_1.npy` differs by max_abs=6.041392.
Phase 5b.21 adds `verify_slat_out_block0_realw`, a real-checkpoint CUDA
gate for the upsampling SLAT output SparseResBlock3d. It starts from
traced `c_h_after_block_23.npy`, concatenates the reverse skip
`c_h_after_input_block_1.npy`, upsamples into
`c_coords_after_input_block_0.npy`, runs real `out_blocks[0]` skip
projection and two submanifold convs, then compares to
`c_h_after_out_block_0.npy`. Gate against `/tmp/sam3d_ref_5b20`:
Nsrc=1007, Ndst=1024, C_in=2048, C_out=128, dim=1024,
max_abs=7.247925e-05, mean_abs=4.278031e-06, avg=5.8301 ms over
20 launches.
Phase 5b.22 adds `verify_slat_out_block1_realw`, the final real-weight
CUDA gate for the SLAT IO SparseResBlock3d set. It starts from traced
`c_h_after_out_block_0.npy`, concatenates the shallow skip
`c_h_after_input_block_0.npy`, runs real `out_blocks[1]` skip
projection and two submanifold convs on
`c_coords_after_input_block_0.npy`, then compares to
`c_h_after_out_block_1.npy`. Gate against `/tmp/sam3d_ref_5b20`:
N=1024, C_in=256, C_out=128, dim=1024, max_abs=2.670288e-05,
mean_abs=1.852996e-06, avg=0.8912 ms over 20 launches. All four SLAT
IO resblocks now have real-weight CUDA verifier coverage; the next
step is wiring those verified launch sequences into the SLAT runner
path.

### Phase 6b — SLAT GS decoder kernelization

Sparse transformer (12 windowed self-attn blocks) + per-voxel decode
head emitting (xyz_offset, dc_sh, scaling_log, quat, opacity_logit).
Hammersley sub-sampling is deterministic and runs once on host.

### v1 deferred

- MoGe depth model — `--pointmap *.npy` is supplied externally.
- SLAT mesh decoder + FlexiCubes `cube2mesh` — needed for `.glb`.
- Texture bake (xatlas + nvdiffrast).
- Layout optimization (open3d plane-fit).
- `.glb` export.
- Server / web integration.

## Reproducing

### 1. Checkpoint

The upstream model lives on gated HF as `facebook/sam-3d-objects` and
must be obtained manually (license acceptance required). After
download, repackage into per-module safetensors via
`cpu/sam3d/convert_ckpt.py` so both runners can load them:

```bash
python cpu/sam3d/convert_ckpt.py \
    --hf-dir <path-to-sam-3d-objects-hf-snapshot> \
    --out-dir /mnt/disk01/models/sam3d/safetensors
# produces sam3d_dinov2.safetensors, sam3d_point_patch_embed.safetensors,
# sam3d_cond_fuser.safetensors, sam3d_ss_dit.safetensors,
# sam3d_ss_decoder.safetensors, sam3d_slat_dit.safetensors,
# sam3d_slat_gs_decoder.safetensors
```

### 2. Pytorch reference dumps (one-time)

Used by every `verify_*.c` to pin numerics. v1 ships the pointmap
externally because MoGe is not yet ported, so any pointmap source is
acceptable as long as it matches the input image:

```bash
cd ref/sam3d && source .venv/bin/activate
python gen_image_ref.py \
    --hf-snapshot <hf-dir> \
    --image fujisan.jpg --mask fujisan_mask.png \
    --outdir /tmp/sam3d_ref --seed 42
# writes input_image.npy, dinov2_tokens.npy, cond_fuser_*.npy,
# ss_dit_in_*.npy / ss_dit_out_*.npy, ss_dec_in.npy / occupancy.npy,
# slat_dit_in_*.npy / slat_dit_out_*.npy, slat_gs_*.npy, pointmap.npy
```

### 3. CPU build + e2e

```bash
cd cpu/sam3d && make all
./test_sam3d --safetensors-dir /mnt/disk01/models/sam3d/safetensors \
    fujisan.jpg fujisan_mask.png \
    --pointmap /tmp/sam3d_ref/pointmap.npy \
    --seed 42 --steps 2 --slat-steps 12 --cfg 2.0 \
    -o /tmp/cpu_sam3d.ply

# Per-stage numerics gates:
make verify-dinov2 verify-cond-fuser verify-ss-dit \
     verify-ss-decoder verify-slat-dit verify-slat-gs
```

### 4. CUDA build + e2e

```bash
cd cuda/sam3d && make all   # builds test_cuda_sam3d + verify_*.c + test_*.c
./test_cuda_sam3d --safetensors-dir /mnt/disk01/models/sam3d/safetensors \
    fujisan.jpg fujisan_mask.png \
    --pointmap /tmp/sam3d_ref/pointmap.npy \
    --seed 42 --steps 2 --slat-steps 12 --cfg 2.0 \
    --precision fp32 -o /tmp/cuda_sam3d.ply

# SLAT-GS-decode-only smoke (skips upstream stages, injects ref tokens):
./test_cuda_sam3d --safetensors-dir /mnt/disk01/models/sam3d/safetensors \
    --slat-ref /tmp/sam3d_ref -o /tmp/cuda_sam3d_slat_ref.ply

# Per-stage numerics gates:
make verify-dinov2 verify-cond-fuser verify-ss-dit \
     verify-ss-decoder verify-slat-dit verify-slat-gs

# Per-kernel microbenches (one binary per kernel; see Makefile
# KERNEL_TESTS for the full list):
./test_dinov2_forward
./test_ssdit_block_forward
./test_pixel_shuffle_3d
# ...
```

### 5. Visual check

Load `/tmp/cuda_sam3d.ply` (or the CPU output) in any Gaussian-splat
viewer — e.g. https://poly.cam/tools/gaussian-splatting-viewer — and
compare against the pytorch-generated reference splat. Numerics on
each stage are already pinned by the per-stage `verify_*` gates;
visual agreement confirms the end-to-end glue.

## Pointers

- Live journal (this doc's source): `cuda/sam3d/PORT.md`.
- CPU port: `cpu/sam3d/PORT.md`, `cpu/sam3d/sam3d_runner.{h,c}`.
- Reference dumps: `/tmp/sam3d_ref/*.npy` ←
  `ref/sam3d/gen_image_ref.py`.
- Checkpoint: `/mnt/disk01/models/sam3d/safetensors/`.
