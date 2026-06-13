# DA3 Sinusoidal UV Pos-Embed — Coverage vs Official

**Date:** 2026-05-18
**Reference:** `depth_anything_3/model/{dualdpt,dpt,gsdpt}.py`, `_add_pos_embed`
at `ratio=0.1, omega_0=100`. Default DA3-Small config has `pos_embed=True`.

## Summary

Official DA3 applies `_add_pos_embed` at **11 sites** controlled by the
`pos_embed=True` flag. The HIP port now covers all 11 sites:

- 2026-05-17: main DualDPT post-fusion and GSDPT post-merger.
- 2026-05-18: main DualDPT per-stage projections, aux post-fusion, and
  GSDPT per-stage projections.

The per-stage implementation uses a token-major variant of the existing
sinusoidal UV kernel so projection outputs can be adjusted before resize
without adding layout conversions on the ConvTranspose stages.

## Coverage table

| Site                                | File                           | Status     |
|-------------------------------------|--------------------------------|------------|
| DualDPT main: post-fused upsample   | `dualdpt.py:240`               | **✓ HIP post-fusion path** |
| GSDPT: post-merger inject           | `gsdpt.py:111`                 | **✓ HIP added 2026-05-17 (post-merger)** |
| DualDPT main: per-stage projection ×4 | `dualdpt.py:225`             | **✓ HIP added 2026-05-18** |
| DualDPT aux: post-fused upsample    | `dualdpt.py:252`               | **✓ HIP added 2026-05-18** |
| GSDPT: per-stage projection ×4      | `gsdpt.py:94`                  | **✓ HIP added 2026-05-18** |

## What was added 2026-05-17

`hip_da3_runner.c` GSDPT branch, between `kl_add_inplace` (merger inject)
and the `output_conv2` chain (`out_0` Conv3x3+ReLU → `out_2` Conv1x1):

```c
{
    float aspect_gs = (float)gs_fw / (float)gs_fh;
    float diag_gs   = sqrtf(aspect_gs * aspect_gs + 1.0f);
    float span_x_gs = aspect_gs / diag_gs;
    float span_y_gs = 1.0f / diag_gs;
    float ratio_gs  = 0.1f;
    int   total_uv  = gs_feat_half * gs_fh * gs_fw;
    int   grid_uv   = (total_uv + 255) / 256;
    void *uv_args[] = {&r->d_dpt_tmp, &gs_feat_half, &gs_fh, &gs_fw,
                       &span_x_gs, &span_y_gs, &ratio_gs};
    KL(r->fn_sinusoidal_uv_posembed, (unsigned)grid_uv, 1, 1, 256, 1, 1, 0,
       r->stream, uv_args);
}
```

This matches the main DPT site's parameter computation exactly (just with
gs_fw/gs_fh instead of model_w/model_h). Verified by smoke run: `test_hip_da3
--gaussians` completes without error, and the depth output is unchanged
(corners and center values bit-identical to non-gaussians run, since the
GSDPT branch only writes to `result.gaussians`).

## What was added 2026-05-18

- `sinusoidal_uv_posembed_tok`: token-major `[H*W, C]` variant used right
  after each per-stage projection and before `resize_layers`.
- Main DualDPT projection loop: calls token-major UV pos-embed for all 4
  feature levels.
- Aux DPT branch: calls the existing CHW UV pos-embed after final aux
  RefineNet fusion and before the aux output conv chain.
- GSDPT projection loop: calls token-major UV pos-embed for all 4 feature
  levels.

Verification:

```sh
make -C rdna4/da3 test_hip_da3
rdna4/da3/test_hip_da3 /mnt/disk1/models/DA3-Small/model.safetensors \
    -i Brooklyn_Bridge_Manhattan.jpg --npy /tmp/da3_small_depth_posembed.npy -v 1
timeout --kill-after=5s 240s rdna4/da3/test_hip_da3 \
    /mnt/disk1/models/DA3-Giant/model.safetensors --gaussians -v 1
```

Results:

- DA3-Small depth sanity PASS, total inference `120.8 ms`, DPT head `63.2 ms`.
- DA3-Giant synthetic `--gaussians` sanity PASS, GSDPT loaded and ran in
  `356.3 ms`, total inference `1925.0 ms`.
- Local DA3-Small/Base/Large safetensors do not contain `gs_head` tensors;
  GSDPT coverage is therefore verified with `/mnt/disk1/models/DA3-Giant/`.

## Reproducibility

```sh
cd rdna4/da3
make test_hip_da3
# Depth-only sanity:
./test_hip_da3 /mnt/disk1/models/DA3-Small/model.safetensors \
    -i /path/to/image.jpg --npy /tmp/depth.npy
# Gaussians path (requires a model with gs_head tensors, e.g. DA3-Giant):
./test_hip_da3 /mnt/disk1/models/DA3-Giant/model.safetensors \
    --gaussians
```
