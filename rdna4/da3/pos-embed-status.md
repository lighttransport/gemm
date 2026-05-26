# DA3 Sinusoidal UV Pos-Embed — Coverage vs Official

**Date:** 2026-05-17
**Reference:** `depth_anything_3/model/{dualdpt,dpt,gsdpt}.py`, `_add_pos_embed`
at `ratio=0.1, omega_0=100`. Default DA3-Small config has `pos_embed=True`.

## Summary

Official DA3 applies `_add_pos_embed` at **11 sites** controlled by the
`pos_embed=True` flag. The HIP port currently covers **2 sites**, both at
the post-fusion/pre-output_conv2 location of a head (one in the main DualDPT
head, one in GSDPT — added 2026-05-17 to match `gsdpt.py:111-112`).

Missing sites have not measurably hurt the **main depth output**: HIP vs
PyTorch correlation is **0.9994** despite the gap (per `README.md`). The
`pos_embed` contribution at ratio=0.1 perturbs features by ±0.1 vs typical
magnitudes 1-10, so missing per-stage applications shift the output by a
few percent at most, which the bottom-up RefineNet fusion largely smooths.

## Coverage table

| Site                                | File                           | Status     |
|-------------------------------------|--------------------------------|------------|
| DualDPT main: post-fused upsample   | `dualdpt.py:240`               | **✓ HIP `hip_da3_runner.c:2707`** |
| GSDPT: post-merger inject           | `gsdpt.py:111`                 | **✓ HIP added 2026-05-17 (post-merger)** |
| DualDPT main: per-stage projection ×4 | `dualdpt.py:225`             | ✗ missing  |
| DualDPT aux: post-fused upsample    | `dualdpt.py:252`               | ✗ missing  |
| GSDPT: per-stage projection ×4      | `gsdpt.py:94`                  | ✗ missing  |

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

## Why the remaining 9 sites are deferred

- **Effort isn't trivial:** the per-stage sites need pos_embed insertion
  inside the per-feature-level projection loop (`hip_da3_runner.c:~2520`
  per-stage code path), which is currently a tight `kl_gemm` →
  `kl_deconv_gemm_scatter` / `kl_tok_to_chw` sequence. The pos_embed call
  must come after the per-stage projection and before the resize layer, so
  it requires reshaping the projection output back to spatial layout
  in-place, applying pos_embed, then resizing.
- **Impact unclear:** main depth at 0.9994 corr suggests these sites add
  high-frequency detail that the official keeps, but the headline accuracy
  doesn't see. To know whether ports would improve `--gaussians` quality or
  `--rays` quality, we'd need a PyTorch reference run with those modalities
  and a per-pixel error map — not currently set up in `rdna4/da3`.
- **Sequencing:** if a PyTorch comparison harness for `--gaussians` /
  `--rays` lands, the missing sites become measurable targets. Until then,
  porting them is speculative.

## Reproducibility

```sh
cd rdna4/da3
make test_hip_da3
# Depth-only sanity (unchanged by today's edit):
./test_hip_da3 /mnt/disk1/models/DA3-Small/model.safetensors \
    -i /path/to/image.jpg --npy /tmp/depth.npy
# Gaussians path (now includes the post-merger pos_embed):
./test_hip_da3 /mnt/disk1/models/DA3-Small/model.safetensors \
    -i /path/to/image.jpg --gaussians --npy /tmp/gs.npy
```
