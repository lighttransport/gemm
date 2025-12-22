# Vulkan Cooperative Matrix Matmul Benchmark

Self-contained benchmark for matrix multiplication using Vulkan compute shaders, targeting AMD RDNA4 GPUs with `VK_KHR_cooperative_matrix` extension.

## Features

- **Cooperative Matrix Kernel**: Hardware-accelerated 16x16x16 matrix tiles using `VK_KHR_cooperative_matrix`
- **Tiled FP16 Kernel**: Optimized shared memory tiled implementation (fallback)
- **Naive FP32 Kernel**: Baseline for correctness verification
- **Self-contained**: All dependencies bundled in `deps/` directory
- **No link-time Vulkan dependency**: Uses vkew for runtime loading

## Requirements

- CMake 3.16+
- C++17 compiler (GCC 9+, Clang 10+)
- Vulkan SDK with `glslc` shader compiler
- Vulkan 1.3+ capable GPU (RDNA4 recommended for cooperative matrix)

## Build

```bash
# From this directory
mkdir build && cd build
cmake ..
make -j$(nproc)
```

Or from project root:
```bash
mkdir build_vulkan_matmul && cd build_vulkan_matmul
cmake ../matmul/vulkan
make -j$(nproc)
```

## Run

```bash
# Run benchmark (from build directory)
./matmul_vulkan_coopmat

# Shaders are automatically copied to build/shaders/
```

## Kernels

### 1. Cooperative Matrix FP16 (`matmul_coopmat_f16.comp`)

Highly optimized kernel using `VK_KHR_cooperative_matrix` extension:
- 128x64 output tiles per workgroup
- 64-deep K-tiling for L2 cache efficiency
- 2x4 cooperative matrix tiles (16x16) per subgroup
- Vectorized 128-bit loads (f16vec4)
- Bank-conflict-free shared memory layout
- Adaptive to 32/64-lane wave sizes

### 2. Tiled FP16 (`matmul_tiled_f16.comp`)

Shared memory tiled implementation:
- 32x32 tiles
- FP16 compute with FP32 accumulation
- Good baseline for GPUs without cooperative matrix

### 3. Naive FP32 (`matmul_naive_f32.comp`)

Simple per-element kernel for correctness verification.

## Performance (AMD RX 9070 XT / RDNA4)

| Matrix Size | Cooperative Matrix | Tiled FP16 | Speedup |
|-------------|-------------------|------------|---------|
| 2048x2048   | 217 GFLOPS        | 214 GFLOPS | 1.01x   |
| 4096x4096   | 434 GFLOPS        | 155 GFLOPS | 2.80x   |
| 8192x8192   | 582 GFLOPS        | 159 GFLOPS | 3.67x   |

Cooperative matrix performance scales better with larger matrices due to higher arithmetic intensity.

## Directory Structure

```
matmul/vulkan/
├── CMakeLists.txt           # Build configuration
├── README.md                 # This file
├── test_matmul_coopmat.cc   # Benchmark harness
├── deps/                     # Bundled dependencies
│   ├── vkew.h               # Vulkan Extension Wrangler header
│   ├── vkew.cc              # Vulkan Extension Wrangler implementation
│   ├── vulkan-runner.hh     # Compute runner interface
│   └── vulkan-runner.cc     # Compute runner implementation
└── shaders/                  # GLSL compute shaders
    ├── matmul_coopmat_f16.comp
    ├── matmul_tiled_f16.comp
    └── matmul_naive_f32.comp
```

## Dependencies

All dependencies are bundled in `deps/`:

- **vkew**: Vulkan Extension Wrangler - loads Vulkan dynamically at runtime without link-time dependency
- **vulkan-runner**: High-level Vulkan compute pipeline wrapper

## Troubleshooting

### glslc not found
Install Vulkan SDK: https://vulkan.lunarg.com/sdk/home

### Cooperative matrix not supported
Your GPU may not support `VK_KHR_cooperative_matrix`. The benchmark will report this and skip the cooperative matrix kernel. The tiled kernel will still run.

### Verification failures
If cooperative matrix results don't match reference, ensure:
1. GPU supports cooperative matrix properly
2. Shader compiled with `--target-env=vulkan1.3 --target-spv=spv1.6`

## License

MIT License - Copyright 2025 Light Transport Entertainment Inc.
