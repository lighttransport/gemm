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

## Reproducing the dumps

See `ref/sam3d/README.md` for the CUDA-host workflow. The pytorch3d
Linux wheels are CUDA-only, so the PyTorch reference must be generated
on a CUDA box and `rsync`'d to the AMD host.
