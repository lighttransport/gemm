# cuda/svdquant — CUDA SVDQuant forward + validation

Validates the SVDQuant forward on the GPU against the PyTorch ground truth in
`ref/svdquant/dumps`, reusing the host decode helpers from
`cpu/svdquant/svdquant_cpu.h`.

```
make test          # builds, generates dumps if missing, runs all 4 cases
./test_cuda_svdquant ../../ref/svdquant/dumps
```

No CUDA SDK at compile time: `cuew` dlopen's the driver API, `cublasew` dlopen's
cuBLAS, and the NVFP4 kernels in `cuda/fp4_w4a4.h` are NVRTC-compiled at runtime
(sm_120a).

## Paths

- **INT4 (w4a16 + w4a4) and NVFP4 w4a16** — host-decode the 4-bit residual
  weight to f32, then run the forward as **pedantic FP32** cuBLAS GEMMs
  (`cublasew_gemm_f32_pedantic_rowmajor_nt`): residual GEMM + the two rank-128
  low-rank GEMMs, bias added on the host. Validates the device toolchain and the
  GEMM composition. Gate `rel_L2(impl, y_svdq) <= 2e-4` (f32 reduction order).
- **NVFP4 w4a4** — the native HW path: `fp4_w4a4_gemm` runs the sm_120a
  `mma.sync ... mxf4nvf4.block_scale` on the residual, quantizing the activation
  on-device; the low-rank branch + bias are added separately. Gate `<= 1e-2`
  (the kernel re-quantizes activations vs numpy; observed ~2e-7, i.e. the
  kernel's e4m3/e2m1 activation rounding bit-matches the reference within f32
  noise). **Skipped gracefully** (not failed) on a non-sm_120a GPU.

All cases report `rel_L2` vs the full-precision `y_fp` too (the irreducible
4-bit quant floor, ~0.04–0.07).
