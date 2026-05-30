# SAM 3D Objects — RDNA4/HIP port journal

Mechanical port of `cuda/sam3d/` to AMD RDNA4 (RX 9070 XT, gfx1201) via
HIPRTC + `rocew`. The CUDA port lands first; the per-stage history,
drift table, and TODOs live in `doc/sam3d.md` and `cuda/sam3d/PORT.md`.

## Port adaptations

- Drop `cuda_hip_compat.h`; include `../rocew.h`,
  `../hip_runner_common.h`, `../hip_kernels_common.h` instead.
- Mechanical `cu*` → `hip*` rename across runner, kernels and tests.
- `hipCtxSynchronize` returns `hipErrorNotSupported` (801) on ROCm
  7.x — the CUDA port's `cuCtxSynchronize` calls were rewritten to
  `hipDeviceSynchronize` across the tree (`test_dinov2_forward` was the
  early canary).
- `MODELS ?= /mnt/disk1/models` (CUDA Makefile points at
  `/mnt/disk01/models` which doesn't exist on this host).
- All GPU-resident stages keep fp32; bf16 follow-up out of scope, same
  as `cuda/sam3d`.

## Verifier status (against CUDA-generated PyTorch dump)

End-to-end `test_hip_sam3d` writes a 135 936-gaussian PLY (~9.2 MiB)
matching the CUDA reference within fp32 noise.

| Verifier (refdir = `/tmp/sam3d_ref`)        | max_abs    | gate     |
|---------------------------------------------|------------|----------|
| `verify_dinov2`                             | 2.52e-04   | 5e-02    |
| `verify_cond_fuser` (smoke)                 | —          | —        |
| `verify_ss_dit` (6drot/scale/translation)   | ≤ 5e-06    | 5e-04    |
| `verify_ss_decoder`                         | 2.06e-04   | 1e-03    |
| `verify_slat_dit`                           | 7.17e-06   | 5e-02    |
| `verify_slat_gs` (transformer/packed_ply)   | ≤ 8.2e-05  | 1e-03    |

Real-weight `_realw` verifiers consume CPU-runner traces produced via
`SLAT_DIT_TRACE=1 SLAT_DIT_TRACE_DIR=/tmp/sam3d_trace
cpu/sam3d/verify_slat_dit` (single forward, not the full pipeline):

| Verifier (refdir = `/tmp/sam3d_trace`)      | max_abs    | gate     |
|---------------------------------------------|------------|----------|
| `verify_slat_input_block0_realw`            | 5.72e-06   | 5e-04    |
| `verify_slat_input_block1_realw`            | 6.68e-06   | 5e-04    |
| `verify_slat_transformer_block_realw`       | 1.45e-04   | 2e-03    |
| `verify_slat_out_block0_realw`              | 8.20e-05   | 5e-04    |
| `verify_slat_out_block1_realw`              | 3.05e-05   | 5e-04    |

The transformer_block verifier additionally needs
`slat_dit_cond.npy`; symlink it from the PyTorch dump:

```
ln -s /tmp/sam3d_ref/slat_dit_cond.npy /tmp/sam3d_trace/slat_dit_cond.npy
```

## e2e correctness fix — PPE NaN under -ffast-math (commit 7fbb648)

Real MoGe pointmaps have invalid (NaN) background pixels. `ppe_linear3_invalid_f32`
used `isfinite()` to substitute `invalid_xyz_token` for them, but HIPRTC builds with
`-ffast-math` (`-ffinite-math-only`) so `isfinite()` folds to constant `true` → NaN
pixels took the valid path → `W·NaN = NaN` → NaN cond tokens → NaN SS-DiT latent →
garbage 262144-voxel occupancy → broken/garbage mesh. The per-stage verifiers all
passed because they feed clean reference inputs; only the real-pointmap e2e hit the
path. Fixed with a bit-pattern non-finite check (exponent==0xFF). cond_fuser now
matches the PyTorch ref (`mean_abs` nan→4.2e-7) and the e2e produces a clean finite
mesh. **Audit other kernels for `isfinite`/`isnan`/`isinf` — same trap applies.**

Validation aids (this host): on-host PyTorch dump at
`/mnt/disk1/models/sam3d/sam3d_ref_new/` (pass as `--refdir`; has `input_image.npy`,
`input_mask.npy`, `pointmap.npy` + every per-stage tensor). Gotchas: serialize e2e
runs (overlapping processes corrupt a shared `-o` file; two models also OOM the 16 GB
card), and consume the `\n` after `end_header` when byte-parsing the binary PLY.

## Speed — BF16 WMMA GEMM (commit 6352c3c)

`gemm_bf16w_bf16a_wmma_t` (ported from rdna4/trellis2, round-to-nearest-even bf16
conversion) + per-call dispatchers, gated by `SAM3D_WMMA` (default on; `=0` = exact
F32). **Now wired across DINOv2 + SS-DiT + SLAT DiT** (commits 6352c3c, b606cae,
af6810e). Coverage: DINOv2 block GEMMs (cs3d_dinov2_gemm), SS-DiT block forward
(ssdit_gemm), SLAT IO/input/final-layer hooks (cs3d_slat_gemm) and SLAT transformer
block forward (slatdit_gemm). Each falls back to gemm_f32_bias when SAM3D_WMMA=0 or
dims aren't 16-aligned. Stream 0 throughout. Attention (sdpa) stays scalar — the
reusable WMMA flash-attn hardcodes head_dim=128 and sam3d uses 64.

Speed (image→splat, 25/25-step, warm, RX 9070 XT): e2e **833.8 → 669.6 s = 1.25×**
(WMMA off vs on). The modest e2e ratio (vs 2.6× on the DINOv2 GEMMs alone) is
because the pipeline is dominated by NON-GEMM work WMMA can't touch — the scalar
`sdpa_f32` self-attention and the CPU-driven SLAT ODE. Those, plus a head_dim=64
WMMA flash-attn, are the real e2e levers; GEMM WMMA was the safe first pass.

Quality (judge by e2e mesh, not the strict per-stage F32 gates — bf16 max_abs ~1e-2
on the diffused latent is expected and over those gates):
  - DINOv2 (conditioning): occupancy 6600→6626, distributions match (2-step).
  - SS-DiT (diffused occupancy var, the dominant shift): full-step occupied-voxel
    IoU 0.82 vs F32, centroids + distributions match, zero nonfinite. The IoU is
    dominated by threshold jitter on the occupancy field; gaussian *values* match.
  - SLAT (refines features on the fixed voxel set): incremental IoU 0.99 vs
    SS-DiT-only — structurally negligible.
  - Full pipeline sign-off (25/25-step, full all-WMMA vs full F32): occupied-voxel
    IoU 0.82 (4246 F32 vs 3935 WMMA voxels, 3686 shared), centroids
    (0.500,0.539,0.514)/(0.503,0.527,0.514), op 5.73/5.68, scale -9.87/-9.79,
    dc -1.747/-1.747, zero nonfinite. Matches the SS-DiT-only full-step figure —
    SLAT adds no further drift, consistent with the 0.99 incremental IoU.

TODO: optional hipBLASLt (rdna4/llm/mm_blaslt_bridge, more accurate than bf16 WMMA)
for the largest GEMMs; carry the PPE NaN fix + WMMA to rdna4/sam3d_body.

## Reproducing the dumps

See `ref/sam3d/README.md` for the CUDA-host workflow. The pytorch3d
Linux wheels are CUDA-only, so the PyTorch reference must be generated
on a CUDA box and `rsync`'d to the AMD host.
