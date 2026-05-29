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
conversion) + a per-call dispatcher, gated by `SAM3D_WMMA` (default on; `=0` = exact
F32). **Wired for DINOv2 only** so far: ~2.6× forward compute, e2e mesh negligibly
changed (occupancy pos 6600→6626, gaussian distributions match) — DINOv2 is diffusion
*conditioning*, so its bf16 error is absorbed. `verify_dinov2` max_abs 0.144 exceeds
the strict 5e-2 F32 gate (expected for bf16), so judge by the e2e mesh, not that gate.

TODO (bigger e2e win, not yet done): route SS-DiT / SLAT-DiT / SLAT-GS GEMMs through
the same dispatcher. These are the *diffused* variable — bf16 error accumulates over
sampler steps — so validate the e2e mesh per stage and keep `SAM3D_WMMA`-gated; only
default-on if the mesh holds. Then optional hipBLASLt (rdna4/llm/mm_blaslt_bridge,
more accurate) for the largest GEMMs, and carry both fixes to rdna4/sam3d_body.

## Reproducing the dumps

See `ref/sam3d/README.md` for the CUDA-host workflow. The pytorch3d
Linux wheels are CUDA-only, so the PyTorch reference must be generated
on a CUDA box and `rsync`'d to the AMD host.
