# FP8 GEMM on RTX 5060 Ti

Verified E4M3/E5M2 GEMM benchmarks using the existing kernels in this folder,
the optimized FP8 kernels from `cuda/gemm`, and the shared cuBLAS loader.
See [measured results](RESULTS.md) for comparisons with FP4 and E0M3.

```sh
make -C cuda/fp8 test
make -C cuda/fp8 bench
make -C cuda/fp8 sanitize
```

The comparison requires a CUDA driver, cuBLAS, and NVRTC with SM120a support
(tested with CUDA 13.2 and cuBLAS 13.4.1). Host programs build with GCC and
the existing dynamic loaders; CUDA SDK headers are not required. Compiler and
NVRTC scratch storage uses repository `tmp/fp8`, configurable with
`FP8_TMP_DIR`. CSV, metadata, and logs belong in ignored `build/`.

```sh
make -C cuda/fp8 bench FP8_BENCH_ARGS='--shape square_4k'
cd cuda/fp8
./build/bench_fp8 --shape all --kernel cublas --csv build/cublas.csv
./build/bench_fp8 --m 512 --n 11008 --k 4096 --kernel ptx
./build/bench_fp8 --m 128 --n 256 --k 128 --full-check --iters 2 --rounds 1
./build/bench_fp8 --help
```

The twelve named shapes match `cuda/int4` and `cuda/fp4`. Select one kernel
by its CSV name, or select `all`, `legacy`, `ptx`, `cublas`, or `bf16`.
Defaults are five warmups, 100 launches per timed round, five rounds, and
rotating kernel order. CSV reports median/min/max CUDA-event time and dense
`2MNK/time` TOp/s. A companion `.metadata.txt` records the command, device,
driver, cuBLAS/NVRTC versions, and initial clocks.

All comparison paths compute row-major `C[M,N] = A[M,K] * B[N,K]^T`, with
FP32 accumulation and output. Original PTX kernels receive a separately
prepared transpose of B, matching their original ABI. Inputs are deterministic
uniform FP32 values in `[-2,2]`, quantized with nearest-even rounding and unit
per-tensor scales. BF16 consumes decoded E4M3 inputs, so it computes the same
product as the E4M3 paths. The optimized custom and cuBLAS paths use E4M3;
the original PTX tiles also measure E5M2.

This runner measures packed GEMM. Compilation, allocation, CPU quantization,
transposition, uploads, and validation are outside timing. FP4's K16 block
scales and FP8's unit tensor scales have different numerical behavior; use
`quant_l2` separately from arithmetic `rel_l2` when comparing them. Large
benchmarks check 64 evenly spaced outputs against FP64 sums over decoded
inputs before and after timing. `--full-check` checks every output for small
matrices, capped at 65,536 output elements. Arithmetic relative L2 must be
at most `1e-4`; NaNs fail validation.

Dimensions must be positive and at most 16,384, and K a multiple of 64. Original `tc32`, `tc64`, and `tc128`
tiles require both M and N divisible by their tile size; unsupported shapes
print `SKIP`. The reused v5/v7 kernels support M/N tails; v7 launches a padded
grid for its 4x4 panel swizzle. This is an SM120 benchmark, with no automatic
GPU architecture fallback for the NVRTC paths.

`fp8_gemm.c` remains the original PTX-only runner (`make run`, `--tile
32|64|128`, or `--naive`). Its B layout is `[K,N]`; it rounds tensor-core
dimensions up and computes a full CPU reference. The corrected shared
`fp8_gemm_kernels.h` is used by both runners. The fragment mapping now uses
rows `lane/4` and `lane/4+8`, with the matching A register order and output
layout. The 128 tile now computes all eight 16x8 MMAs per warp; its previous
four MMAs left half the columns unwritten. Verification tolerance is tighter,
and failure returns a nonzero status. Unverified hardware peak estimates
were removed from the original runners. The scalar fallback now checks M/N
bounds and decodes FP8 subnormals correctly, including odd matrix sizes.

`cublas_fp8_gemm.c` remains a standalone BF16-output cuBLASLt smoke test.
Its descriptor attribute IDs, 64-byte algorithm ABI, transpose/layout setup,
conversion helpers, and verification status have been corrected. An unavailable
algorithm is reported for the requested shape/type, without claiming that FP8
is unavailable on the GPU. Use `build/bench_fp8` for the FP32-output performance
comparison. E5M2 conversion now handles nearest-even rounding, subnormals,
signed zero, and infinities; its tests no longer skip the infinity bug.
The cuBLAS descriptor definitions follow the
[cuBLASLt API](https://docs.nvidia.com/cuda/cublas/#cublasltmatmuldescattributes-t).
