# DA3 Depth Estimation - HIP/RDNA4

GPU-accelerated [Depth Anything 3](https://github.com/DepthAnything/Depth-Anything-V3) depth estimation using HIP/ROCm on AMD RDNA4 GPUs.

Compiles with plain `gcc` (no `hipcc` needed). Kernels are compiled at runtime via HIPRTC.

## Quick Start

```bash
# Build
make

# Run (safetensors)
./test_hip_da3 /path/to/DA3-Small/model.safetensors -i photo.jpg -o depth.png --npy depth.npy

# Run (GGUF)
./test_hip_da3 /path/to/da3-small.gguf -i photo.jpg -o depth.png
```

## Requirements

- AMD RDNA4 GPU (RX 9070 XT / gfx1200 / gfx1201)
- ROCm runtime libraries on the system:
  - `libamdhip64.so` (HIP runtime)
  - `libhiprtc.so` (runtime kernel compiler)
- `gcc` (no hipcc or ROCm SDK needed at compile time)
- Model weights: [depth-anything/DA3-Small](https://huggingface.co/depth-anything/DA3-Small) safetensors

## Performance

| Image Size | Grid | Backbone | DPT Head | Total | Notes |
|-----------|------|----------|----------|-------|-------|
| 2100x1400 | 36x24 | 74 ms | 76 ms | ~150 ms | Brooklyn Bridge test image |
| 1050x700 | 36x24 | 74 ms | 76 ms | ~128 ms | Same image, half resolution |
| 518x518 | 37x37 | 85 ms | 98 ms | ~183 ms | Square input |

First run includes HIPRTC kernel compilation (~30-60 min on gfx1201 with ROCm 7.x).

## Accuracy

Verified against the official PyTorch DA3 implementation:

| Comparison | Correlation | rel_L2 |
|-----------|-------------|--------|
| HIP vs PyTorch (matched input pixels) | 0.9994 | 1.3% |
| HIP vs PyTorch (official preprocessing) | 0.9994 | 1.5% |
| HIP vs CPU reference | 0.9999 | 0.5% |

## Supported Models

Auto-detected from safetensors tensor shapes. No config.json required for basic inference.

| Model | embed_dim | Blocks | Heads | Status |
|-------|-----------|--------|-------|--------|
| DA3-Small | 384 | 12 | 6 | Verified |
| DA3-Base | 768 | 12 | 12 | Untested (should work) |
| DA3-Large | 1024 | 24 | 16 | Untested (should work) |
| DA3-Giant | 1536 | 40 | 24 | Untested (should work) |

## Output Modalities

| Output | Flag | Status |
|--------|------|--------|
| Depth + Confidence | `DA3_OUTPUT_DEPTH` | Implemented, verified |
| Pose (CameraDec) | `DA3_OUTPUT_POSE` | Implemented |
| Rays + Sky Seg | `DA3_OUTPUT_RAYS` | Implemented |
| 3D Gaussians (GSDPT) | `DA3_OUTPUT_GAUSSIANS` | Implemented |
| Metric Depth | - | Not implemented |

## CLI Options

```
./test_hip_da3 <model.safetensors|model.gguf> [options]

Input:
  -i <path>          Input image (JPEG/PNG/BMP). Default: synthetic gradient
  -d <id>            GPU device ID (default: 0)

Output:
  -o <path>          Output image (.png falsecolor, .pgm 16-bit, .exr float)
  --npy <path>       Save raw depth as NumPy .npy
  --npy-dir <dir>    Save all outputs as .npy files to directory

Modes:
  --full             Enable all output modalities
  --pose             Enable pose estimation
  --rays             Enable ray + sky segmentation
  --gaussians        Enable 3D gaussian output

Debug:
  -v <level>         Verbosity (0=quiet, 1=timing, 2=per-block stats)
```

## Output Formats

| Format | Content |
|--------|---------|
| `.png` | Falsecolor (turbo colormap) depth visualization |
| `.pgm` | 16-bit normalized depth |
| `.exr` | Raw float32 channels (depth, confidence, rays, gaussians) |
| `.npy` | NumPy float32 array |

## Preprocessing

Matches the official DA3 pipeline:
- Aspect-ratio preserving resize (longest side to 504)
- Each dimension rounded to nearest multiple of 14 (patch size)
- Area-based downscaling (matching `cv2.INTER_AREA`)
- ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- Bicubic position embedding interpolation for non-square grids

## Architecture

```
Input RGB → resize + normalize → patch_embed (Conv2d 14x14)
         → CLS token + pos_embed → camera_token injection
         → 12x ViT blocks (QK-norm, 2D RoPE, FlashAttention, GELU MLP)
         → cat_token feature extraction [local_x, backbone_norm(x)]
         → DualDPT head (4-level spatial fusion + RefineNet)
         → neck conv → upsample → UV posembed → out conv → exp() activation
         → bilinear upsample to original resolution
         → depth [H, W] + confidence [H, W]
```

## Files

| File | Description |
|------|-------------|
| `hip_da3_runner.h` | Public C API |
| `hip_da3_runner.c` | Implementation (~2970 lines, HIPRTC kernels) |
| `test_hip_da3.c` | Test harness with multiple output formats |
| `test_cpu_da3.c` | CPU-only reference for validation |
| `Makefile` | Build (just `make`) |
