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

## Speed — BF16 WMMA GEMM for the encoder (commit 6f9f1c0)

The encoder backbone (DINOv3-H+ / ViT-H, ~9 GEMMs/block × 32 blocks — the
dominant cost) stores F16 weights, so `gemm_f16w_bf16a_wmma_t` is an F16-weight
variant of sam3d's BF16 WMMA kernel (`half_to_float` at SMEM load, RNE bf16 for
both operands, matching the encoder's existing bf16-rounded activation path). An
`sb_enc_gemm()` dispatcher routes all 9 encoder GEMMs through it, gated by
`SAM3D_BODY_WMMA` (default on; `=0` = exact F16-tiled path).

The decoder transformer (6 layers) also routes through WMMA via a sibling
`gemm_bf16w_bf16a_wmma_t` (F32-weight variant — decoder weights are F32),
`sb_dec_gemm()`, same `SAM3D_BODY_WMMA` gate. The cross-attn K/V GEMMs run over
the 1024 encoder-context tokens, which fill the WMMA tile (the 145-token
self-attn underfills but falls back cleanly); measured ~2.0× on the layers
(43.8 → 22.0 ms).

Speed (dancing.jpg, DINOv3 backbone, warm, RX 9070 XT): encoder transformer
blocks 664 → 219 ms (**3.04×**); encoder total 687 → 240 ms (2.86×); decoder
layers 43.8 → 22.0 ms (~2.0×); full pipeline (encoder+decoder WMMA) **e2e
1.72 → 1.27 s = 1.35×**. The remaining e2e floor is CPU MHR skinning (~230 ms)
+ decoder norm_heads — neither is GEMM; MHR-on-GPU is the next lever.

Quality: DINOv3 is the precision floor (verify_dinov3 max_abs 0.54 vs gate 1.5),
but the e2e mesh tolerates bf16 well — full WMMA vs full F32 (V=18439 F=36874,
identical topology) shows mean vertex displacement 0.0017 / max 0.0035 on a 1.725
bbox diagonal (0.1% / 0.20%), zero nonfinite.

The sam3d PPE `-ffast-math`/`isfinite` NaN fix has no analogue here: sam3d_body
masks invalid points via explicit host-passed `int *invalid` flags, not runtime
`isfinite`.

## Reproducing the dumps

The `/tmp/sam3d_body_ref/` dumps come from a CUDA host running
`ref/sam3d-body/`'s python helpers; same constraint as `sam3d/`
(`pytorch3d`/`xformers`/etc. are CUDA-only Linux wheels). Generate on a
CUDA box and `rsync` the dump to this AMD host.
