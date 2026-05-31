# DA3 Depth Estimation - Vulkan

Cross-platform GPU-accelerated [Depth Anything 3](https://github.com/DepthAnything/Depth-Anything-V3) depth estimation using Vulkan compute shaders. Works on any GPU with Vulkan 1.0+ compute support (AMD, NVIDIA, Intel).

Port of the verified HIP/RDNA4 implementation with identical pipeline and accuracy.

## Quick Start

```bash
# Build
cd /path/to/gemm/vulkan
mkdir -p build && cd build
cmake .. && make test_vulkan_da3

# Run
./test_vulkan_da3 /path/to/DA3-Small/model.safetensors \
  -i photo.jpg -o depth.png --npy depth.npy \
  --shader-dir /path/to/gemm/vulkan/build
```

On systems with multiple Vulkan ICDs (e.g., AMDGPU + RADV):
```bash
VK_ICD_FILENAMES=/etc/vulkan/icd.d/radeon_icd_amdgpu.x86_64.json \
  ./test_vulkan_da3 model.safetensors -i photo.jpg -o depth.png --shader-dir .
```

## Requirements

**Build time:**
- CMake 3.16+
- C++17 compiler (g++ or clang++)
- Vulkan SDK (`glslc` shader compiler for SPIR-V compilation)

**Run time:**
- Vulkan-capable GPU with compute support
- Vulkan loader library (`libvulkan.so` / `vulkan-1.dll`)
- No Vulkan SDK needed at runtime (vkew loads dynamically)

**Model:**
- [depth-anything/DA3-Small](https://huggingface.co/depth-anything/DA3-Small) safetensors

## Performance

Tested on AMD RX 9070 XT (RDNA4) via RADV:

| Image Size | Grid | Backbone | DPT Head | Total |
|-----------|------|----------|----------|-------|
| 2100x1400 | 36x24 | 750 ms | 570 ms | ~1.3 s |
| 504x336 | 36x24 | 750 ms | 540 ms | ~1.3 s |
| 504x504 | 36x36 | 850 ms | 600 ms | ~1.5 s |

Currently uses per-dispatch synchronization (each kernel dispatch waits for completion). Performance can be significantly improved by batching dispatches into single command buffers with pipeline barriers.

Comparison with other backends (same image, same model):

| Backend | Time | Speedup |
|---------|------|---------|
| PyTorch CPU (Ryzen 9 3950X) | 5.3 s | 1x |
| **Vulkan (RX 9070 XT)** | **1.3 s** | **4x** |
| HIP (RX 9070 XT) | 0.13 s | 41x |

## Accuracy

Verified against PyTorch official and the HIP implementation:

| Comparison | Correlation | rel_L2 |
|-----------|-------------|--------|
| Vulkan vs PyTorch | 0.9994 | 1.5% |
| Vulkan vs HIP | 0.999999 | <0.01% |
| HIP vs PyTorch | 0.9994 | 1.5% |

The Vulkan version uses F32 GEMM (vs HIP's F16 GEMM), so results are numerically near-identical to HIP.

## Supported Models

Auto-detected from safetensors tensor shapes:

| Model | embed_dim | Blocks | Heads | Status |
|-------|-----------|--------|-------|--------|
| DA3-Small | 384 | 12 | 6 | Verified (all modalities) |
| DA3-Base | 768 | 12 | 12 | Untested (should work) |
| DA3-Large | 1024 | 24 | 16 | Untested (should work) |
| DA3-Giant | 1536 | 40 | 24 | Verified (all modalities, ~39s unoptimized) |

## Output Modalities

| Output | Flag | Status |
|--------|------|--------|
| Depth + Confidence | `DA3_OUTPUT_DEPTH` | Implemented, verified |
| Pose estimation (CameraDec) | `DA3_OUTPUT_POSE` | Implemented, verified vs HIP |
| Rays + Sky Seg (Aux DPT) | `DA3_OUTPUT_RAYS` | Implemented |
| 3D Gaussians (GSDPT) | `DA3_OUTPUT_GAUSSIANS` | Implemented (needs Giant model) |

## CLI Options

```
./test_vulkan_da3 <model.safetensors> [options]

Input:
  -i <path>            Input image (JPEG/PNG/BMP). Default: synthetic gradient
  -d <id>              GPU device ID (default: 0)
  --shader-dir <path>  Path to compiled SPIR-V shaders (default: ".")

Output:
  -o <path>            Falsecolor depth PNG
  --npy <path>         Raw depth as NumPy .npy

Debug:
  --full               Enable all output modalities
  --pose               Enable pose estimation
  --rays               Enable ray + sky segmentation
  --gaussians          Enable 3D gaussian output

Debug:
  -v <level>           Verbosity:
                         0 = quiet
                         1 = timing + model info
                         2 = per-block backbone stats
                         3 = full intermediate tensor dumps
```

## Compute Shaders

18 GLSL compute shaders in `vulkan/shaders/da3/`:

| Shader | Purpose |
|--------|---------|
| `resize_normalize_f32` | Area-based downscale + ImageNet normalize |
| `patch_embed_conv2d_f32` | Conv2d patch embedding (CHW input) |
| `cls_pos_embed_f32` | CLS token + positional embedding |
| `rope_2d_f32` | 2D Rotary Position Embedding |
| `swiglu_f32` | SiLU-gated linear unit |
| `kv_transpose_f32` | Deinterleave K,V from QKV |
| `flash_attn_tiled_f32` | Tiled FlashAttention (BQ=64, BKV=16) |
| `cat_local_global_f32` | Concatenate local/global features |
| `dpt_cls_concat_f32` | Extract patch tokens (skip CLS) |
| `strided_layernorm_f32` | LayerNorm on second half of features |
| `dpt_tok_to_chw_f32` | Token-major to spatial CHW reshape |
| `deconv_scatter_f32` | ConvTranspose2d via GEMM scatter |
| `conv2d_f32` | Generic Conv2d |
| `bilinear_upsample_f32` | Bilinear interpolation |
| `depth_activation_f32` | exp(depth), exp(conf)+1 |
| `sinusoidal_uv_posembed_f32` | UV positional embedding |
| `relu_f32` | In-place ReLU |
| `channel_layernorm_f32` | Per-position channel LayerNorm |
| `silu_f32` | In-place SiLU activation (for GSDPT merger) |

Plus 6 shared shaders: `layernorm_f32`, `matmul_bias_f32`, `gelu_f32`, `add_f32`, `hy3d/layerscale_add_f32`, `hy3d/qk_layernorm_f32`.

## Architecture

Same pipeline as the HIP implementation:

```
Input RGB
  -> area-based resize (longest side 504, round to 14)
  -> patch embedding (Conv2d 14x14, CHW)
  -> CLS token + bicubic-interpolated pos_embed
  -> camera_token injection at rope_start
  -> 12x ViT blocks:
       LayerNorm -> QKV GEMM -> QK-norm -> 2D RoPE (local/global alternating)
       -> KV transpose -> FlashAttention -> output proj -> LayerScale + residual
       -> LayerNorm -> GELU MLP -> LayerScale + residual
       -> cat_token feature save [local_x, backbone_norm(x)]
  -> DualDPT head:
       4-level token projection + spatial alignment (deconv/identity/conv)
       -> adapter Conv3x3 -> bottom-up RefineNet fusion
       -> neck Conv3x3 (no ReLU) -> bilinear upsample -> UV posembed
       -> out_0 Conv3x3 + ReLU -> out_2 Conv1x1
  -> depth activation (exp)
  -> bilinear upsample to original resolution
  -> depth [H, W] + confidence [H, W]
```

## Files

| File | Description |
|------|-------------|
| `vulkan_da3_runner.h` | Public C API |
| `vulkan_da3_runner.cc` | Implementation (~1900 lines) |
| `test_vulkan_da3.cc` | Test harness |
| `stb_impl.c` | stb_image/write implementations |

## Differences from HIP Version

| Aspect | HIP | Vulkan |
|--------|-----|--------|
| GEMM precision | F16 weights, F32 accumulate | F32 weights, F32 accumulate |
| Kernel compilation | HIPRTC at runtime (~30 min first run) | Pre-compiled SPIR-V (instant) |
| GPU support | RDNA4 only (gfx1200/1201) | Any Vulkan compute GPU |
| Dispatch model | Async stream | Per-dispatch sync (optimization TODO) |
| Output modalities | Depth, Pose, Rays, Gaussians | Depth, Pose, Rays, Gaussians |
