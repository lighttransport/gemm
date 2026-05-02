# Vulkan RDNA4 WMMA GEMM

Experimental Vulkan cooperative-matrix GEMM benchmark for RDNA4-style shapes.

The v1 path is FP16 input with FP32 accumulation:

```text
Y[M,N] = X[M,K] * W[N,K]^T
```

This matches the ROCm GEMM benchmark convention where activations are `X[M,K]`
and weights are pre-transposed as `W[N,K]`.

## Build

From the repository root:

```sh
cmake -S vulkan -B vulkan/build
cmake --build vulkan/build -j
```

Or standalone:

```sh
cmake -S vulkan/wmma -B vulkan/wmma/build
cmake --build vulkan/wmma/build -j
```

## Run

From `vulkan/build`:

```sh
./wmma/bench_wmma_gemm --check-caps
./wmma/bench_wmma_gemm --shape mm0 --iters 200 --check
./wmma/bench_wmma_gemm --shape all --iters 200 --check
```

From another working directory, pass the build directory:

```sh
vulkan/build/wmma/bench_wmma_gemm --shader-dir vulkan/build --shape mm0
```

## Notes

- The shader uses `VK_KHR_cooperative_matrix` with 16x16x16 subgroup tiles.
- Shapes are padded internally to `M % 128 == 0`, `N % 64 == 0`, `K % 64 == 0`.
- Verification is sampled by default so large shapes such as `mm0` do not spend
  minutes in a full CPU reference.
- BF16 and FP8 are not implemented in this Vulkan v1 path. The local Vulkan
  headers and `glslc` expose cooperative-matrix component types for FP16/FP32
  and integer formats, but not RDNA4 BF16/FP8 WMMA component types comparable
  to the ROCm builtins.
