# cpu/svdquant — portable C SVDQuant forward + validation

Pure-C (double-accumulation) reference for the SVDQuant forward, validated
against the PyTorch ground truth in `ref/svdquant/dumps`.

```
make test          # builds, generates dumps if missing, runs all 4 cases
./test_svdquant ../../ref/svdquant/dumps
```

Covers all four cases — `{int4,nvfp4} x {w4a16,w4a4}`. For each it decodes the
4-bit residual weight, runs `sq_forward`, and gates on
`rel_L2(impl, y_svdq) <= 1e-5` (same scheme as the reference, so only f64-vs-f32
rounding separates them; a real layout/packing bug shows up as rel_L2 ~0.1+).
It also reports `rel_L2` vs the full-precision `y_fp` (the irreducible 4-bit
quant floor, ~0.04–0.07 here).

## `svdquant_cpu.h` (single header, all `static inline`)

- `sq_unpack_int4_residual` — signed `[-8,7]` nibble x group-64 scale.
- `sq_unpack_nvfp4_residual` — e2m1 code x e4m3 micro-scale x per-row `wcwt`
  (`SQ_E2M1` LUT + `sq_ue4m3_decode`, same formulas as `cuda/fp4_w4a4.h`).
- `sq_quant_act_int4_g64` — per-token group-64 INT4 activation quant (`rintf` =
  round-half-to-even, bit-matches `torch.round`; W4A4 path).
- `sq_smooth_div`, `sq_forward` — `y = act(x/λ)@R^T + (x@ld^T)@lu^T + bias`.

## Notes

- W4A4 INT4 runs the CPU's **own** activation quantizer and additionally checks
  its drift vs the reference dequantized activation (observed `0.000e+00`).
- W4A4 NVFP4 consumes the reference's dumped `xr_dq` (the kernel-side e2m1/e4m3
  activation rounding is validated on the GPU in `cuda/svdquant`, not re-derived
  in scalar C).
