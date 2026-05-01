# SAM 3D Body — RDNA4/HIP port journal

Mechanical port of `cuda/sam3d_body/` to AMD RDNA4 (RX 9070 XT, gfx1201)
via HIPRTC + `rocew`. The CUDA port history lives in
`doc/sam3d-body.md`.

## Port adaptations

- Drop `cuda_hip_compat.h`; include `../rocew.h`,
  `../hip_runner_common.h`, `../hip_kernels_common.h` instead.
- Mechanical `cu*` → `hip*` rename across runner, kernels and tests.
- `hipDeviceSynchronize` everywhere — `hipCtxSynchronize` returns
  `hipErrorNotSupported` (801) on ROCm 7.x.
- `MODELS ?= /mnt/disk1/models` (CUDA Makefile points at
  `/mnt/disk01/models` which doesn't exist on this host).
- Backbone precision: bf16 / fp16 only (mirrors CUDA — `fp32` is
  rejected by `sam3d_body_create`).

## End-to-end

```
./test_hip_sam3d_body \
    --safetensors-dir /mnt/disk1/models/sam3d-body/safetensors \
    --mhr-assets      /mnt/disk1/models/sam3d-body/safetensors \
    --image /tmp/sam-3d-objects/notebook/images/human_object/image.png \
    --bbox 200 100 1400 800 \
    -o /tmp/hip_sam3d_body.obj
```

Writes `V=18439 F=36874` (~1.2 MiB OBJ) on the human_object sample.

## Verifier status (refdir = `/tmp/sam3d_body_ref`)

| Verifier              | max_abs    | gate       |
|-----------------------|------------|------------|
| `verify_dinov3`       | 5.43e-01   | 1.5e+00    |
| `verify_ray_cond`     | 1.72e-05   | 2.0e-03    |
| `verify_build_tokens` | 3.81e-06   | 5.0e-04    |
| `verify_decoder_layer`| 6.68e-06   | 2.0e-03    |
| `verify_kp_update`    | 5.96e-06   | 5.0e-04    |
| `verify_mhr_head`     | 6.68e-06   | 1.0e-04    |
| `verify_decoder`      | 8.11e-06   | 5.0e-03    |
| `verify_mhr`          | 3.05e-05   | 5.0e-04    |

`verify_decoder` and `verify_mhr` need `--mhr-assets[-dir]
/mnt/disk1/models/sam3d-body/safetensors` in addition to
`--safetensors-dir`.

### Not green

- `verify_vith` — needs `vith_input.npy`, not in the current
  `/tmp/sam3d_body_ref` dump. `verify_dinov3` already covers the
  backbone path.
- `verify_end_to_end` — scaffold only, mirrors the CUDA status (the
  upstream pipeline isn't wired into the verifier yet).

## Reproducing the dumps

The `/tmp/sam3d_body_ref/` dumps come from a CUDA host running
`ref/sam3d-body/`'s python helpers; same constraint as `sam3d/`
(`pytorch3d`/`xformers`/etc. are CUDA-only Linux wheels). Generate on a
CUDA box and `rsync` the dump to this AMD host.
