# cuda/svdquant — CUDA SVDQuant forward + validation

Validates the SVDQuant forward on the GPU against the PyTorch ground truth in
`ref/svdquant/dumps`. The baseline path reuses the host decode helpers from
`cpu/svdquant/svdquant_cpu.h`; the CUDA MMA paths use `cuda_svdquant.h`.

```
make test          # builds, refreshes stale dumps, runs all cases
./test_cuda_svdquant ../../ref/svdquant/dumps
```

No CUDA SDK at compile time: `cuew` dlopen's the driver API, `cublasew` dlopen's
cuBLAS, and the helper kernels are NVRTC-compiled at runtime. NVFP4 W4A4 still
requires the sm_120a FP4 OMMA kernel from `cuda/fp4_w4a4.h`.

## Paths

- **INT4 (w4a16 + w4a4) and NVFP4 w4a16** — host-decode the 4-bit residual
  weight to f32, then run the forward as **pedantic FP32** cuBLAS GEMMs
  (`cublasew_gemm_f32_pedantic_rowmajor_nt`): residual GEMM + the two rank-128
  low-rank GEMMs, bias added on the host. Validates the device toolchain and the
  GEMM composition. Gate `rel_L2(impl, y_svdq) <= 2e-4` (f32 reduction order).
- **INT4-as-INT8 w4a4** — GPU-decode existing INT4 nibbles to signed int8
  values, GPU-quantize `x / lambda` to the matching group-64 signed int8 domain
  (`[-7,7]`), then run group-wise cuBLAS IMMA (`s8 x s8 -> s32`). This is the
  tensor-core path for GPUs without hardware INT4 MMA. Gate `<= 2e-4`.
- **INT8 w8a8** — true group-64 signed INT8 SVDQuant (`[-127,127]`) for both
  residual weights and activations, encoded on the GPU and run through the same
  IMMA residual path. Gate `<= 2e-4`.
- **NVFP4 w4a4** — the native HW path: `fp4_w4a4_gemm` runs the sm_120a
  `mma.sync ... mxf4nvf4.block_scale` on the residual, quantizing the activation
  on-device; the low-rank branch + bias are added separately. Gate `<= 1e-2`
  (the kernel re-quantizes activations vs numpy; observed ~2e-7, i.e. the
  kernel's e4m3/e2m1 activation rounding bit-matches the reference within f32
  noise). **Skipped gracefully** (not failed) on a non-sm_120a GPU.

All cases report `rel_L2` vs the full-precision `y_fp` too. The 4-bit quant
floor is usually ~0.04-0.07 on the synthetic dump; true W8A8 is much tighter.
