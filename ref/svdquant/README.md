# ref/svdquant — SVDQuant reference + ground-truth generator

Self-contained PyTorch reference for the `cpu/svdquant` and `cuda/svdquant`
unit tests. Pins the SVDQuant forward (SmoothQuant λ + rank-128 SVD low-rank +
4-bit residual) for **INT4** and **NVFP4**, in both **W4A16** and **W4A4**.

## What SVDQuant computes

For one linear `W[out,in]`, smoothing `λ[in]`, rank-r factors and the 4-bit
residual `R` of the *smoothed* weight `What = W·diag(λ)`:

```
y = act(x/λ) @ R_dec^T  +  (x @ lora_down_emit^T) @ lora_up^T  +  bias
```

- residual term uses the **smoothed** activation `x/λ`; low-rank uses **raw** `x`;
- `lora_down_emit = lora_down/λ` folds smoothing back so the two branches
  recombine to `W@x` (minus the 4-bit residual error);
- `act(·)` = identity (W4A16) or 4-bit per-token-group quant→dequant (W4A4);
- INT4 residual = group-64 signed `[-7,7]`; NVFP4 residual = e2m1 group-16 with
  e4m3 micro-scales + per-row `wcwt`.

## Generate the dumps

```
python3 gen_svdquant_ref.py --out dumps              # synthetic, seeded, model-free
python3 gen_svdquant_ref.py --out dumps --no-smooth  # λ=1 baseline (tightest)
```

Output: `dumps/*.npy` (float32 / int32 / uint8 **only** — the C loader
`common/npy_io.h` reads exactly `f4`/`i4`/`u1`) + `dumps/manifest.json`.

### Tensor contract (per `dumps/`)

Shared: `x`(f32 TOK×IN), `W`(f32 OUT×IN), `bias`(f32 OUT),
`dims`(i32 `[OUT,IN,TOK,RANK]`), `y_fp`(f32 = `x@W^T+b`).

Per case `<fmt>_<scope>_` where fmt∈{int4,nvfp4}, scope∈{w4a16,w4a4}:
`smooth`(f32 IN), `lora_up`(f32 OUT×R), `lora_down`(f32 R×IN, already `/λ`),
`y_svdq`(f32 TOK×OUT — **the implementation target**), plus
- INT4: `qint4`(u8 OUT×IN/2), `wscale`(f32 OUT×IN/64);
- NVFP4: `qw`(i32 OUT×IN/8), `ws`(u8 OUT×IN/16 raw e4m3), `wcwt`(f32 OUT);
- W4A4 (either fmt): `xr_dq`(f32 TOK×IN) = the dequantized residual activation
  the driver used (the CPU consumes this so it need not re-implement e4m3/e2m1
  activation rounding).

`y_svdq` is computed by **decoding the dumped quantized tensors**, so a C/CUDA
reader of the same bytes with the same decode formula matches to f32 rounding.

## Optional

- `--cross-check` — drives deepcompressor's authoritative SVD low-rank branch
  (`deepcompressor.nn.patch.lowrank.LowRankBranch`, exact `torch.linalg.svd`) and
  diffs it against ours (randomized `torch.svd_lowrank`): residual energy,
  effective-weight agreement, forward agreement. Observed: ours is within ~0.03%
  of the exact-SVD residual lower bound, subspaces agree to ~3%. Also tries
  deepcompressor's `simple_quantize` for a 4-bit RTN cross-check, skipped
  gracefully when its CUDA C-extension build is unavailable. Never on the default
  path; the whole check skips cleanly if deepcompressor isn't importable.
- `--real <ckpt.safetensors> --layer <key>` — pull a real dense `[out,in]` weight
  (synthetic activation).
- `--real-nunchaku <dump.safetensors>` — derive a **real DiT-magnitude** weight
  AND **real activation** from a Nunchaku SVDQuant ground-truth dump (e.g.
  `nunchaku_ref_dump.transformer_blocks.0.*.safetensors`). The rank-128 low-rank
  is decoded exactly via the installed Nunchaku packer; the int4-main residual is
  de-swizzled with the in-repo perms (NOTE: that byte-swizzle is
  nunchaku-version-specific, so the reconstructed Ŵ may not bit-match the original
  layer — the driver reports the `Ŵ@(x/smooth)+b` vs real-`y` fidelity). The
  point is realistic non-Gaussian weight statistics + real outlier activations to
  stress SmoothQuant; the CPU/CUDA tests still pass (gates unchanged, quant floor
  rises to ~0.08–0.13 as real weights are harder than Gaussian).

`deepcompressor/` is a plain `git clone`
(`https://github.com/mit-han-lab/deepcompressor`) kept as source-of-truth and
**gitignored** — it is never imported on the default generation path.

## Math vendored from (credit)

- `rdna4/qimg/tools/svdquant_from_bf16.py` — `smoothing_lambda`,
  `svd_lowrank_branch`, `quant_int4_g64`, `pack_nibbles`.
- `nunchaku_fp4_repack_omma.py` — `quant_nvfp4`, `pack_codes_u32`, `_E2M1_THR`,
  `_decode_nvfp4`.
