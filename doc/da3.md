# DA3 (Depth Anything 3) — Multi-Backend Depth Estimation

Production-ready [Depth Anything 3](https://github.com/DepthAnything/Depth-Anything-V3) monocular depth estimation with CUDA, HIP/ROCm, Vulkan, and CPU backends. All verified against PyTorch official (correlation > 0.999).

## Quick Start

```bash
# CUDA (NVIDIA GPU)
cd cuda/da3 && make
./test_cuda_da3 model.safetensors -i photo.jpg -o depth.png

# HIP (AMD RDNA4)
cd rdna4/da3 && make
./test_hip_da3 model.safetensors -i photo.jpg -o depth.png

# Vulkan (any GPU)
cd vulkan && mkdir -p build && cd build && cmake .. && make test_vulkan_da3
./test_vulkan_da3 model.safetensors -i photo.jpg -o depth.png --shader-dir .

# CPU (no GPU needed)
cd rdna4/da3
gcc -O3 -o test_cpu_da3 test_cpu_da3.c -lm -I../../common
./test_cpu_da3 model.safetensors -i photo.jpg -o depth.npy -t 8
```

## Models

Download from HuggingFace: [depth-anything/DA3-Small](https://huggingface.co/depth-anything/DA3-Small), [DA3-Base](https://huggingface.co/depth-anything/DA3-Base), [DA3-Large](https://huggingface.co/depth-anything/DA3-Large), [DA3-Giant](https://huggingface.co/depth-anything/DA3-Giant)

| Model | Params | File Size | embed_dim | Blocks | Status |
|-------|--------|-----------|-----------|--------|--------|
| DA3-Small | 25M | 131 MB | 384 | 12 | All backends verified |
| DA3-Base | 86M | 517 MB | 768 | 12 | Should work (untested) |
| DA3-Large | 300M | 1.6 GB | 1024 | 24 | Should work (untested) |
| DA3-Giant | 1.2B | 5.1 GB | 1536 | 40 | CUDA/HIP/Vulkan verified |

Models auto-detected from safetensors tensor shapes. Optional `config.json` in the same directory.

## Performance

**DA3-Small** (Brooklyn Bridge 2100×1400, depth-only):

| Backend | Device | Total | Backbone | DPT Head |
|---------|--------|-------|----------|----------|
| CUDA | V100 | ~100 ms | ~40 ms | ~60 ms |
| HIP | RX 9070 XT | ~150 ms | ~74 ms | ~76 ms |
| Vulkan | RX 9070 XT (RADV) | ~390 ms | ~154 ms | ~214 ms |
| CPU | Ryzen 9 3950X (8t) | ~39 s | ~29 s | ~10 s |
| PyTorch CPU | Ryzen 9 3950X | ~5.3 s | — | — |

**DA3-Giant** (Brooklyn Bridge 2100×1400, all outputs):

| Backend | Total | Backbone | DPT | Aux DPT | GSDPT |
|---------|-------|----------|-----|---------|-------|
| CUDA | ~850 ms | ~600 ms | ~250 ms | — | — |
| HIP | ~2.0 s | 750 ms | 310 ms | 314 ms | 549 ms |
| Vulkan | ~2.3 s | 2067 ms | 134 ms | — | — |

## Accuracy vs PyTorch Reference

| Backend | Model | Correlation | rel_L2 |
|---------|-------|-------------|--------|
| CUDA F16 | Small | 0.9992 | 1.1% |
| HIP F16 | Small | 0.9994 | 1.5% |
| Vulkan F16 | Small | 0.9994 | 1.5% |
| CPU F32 | Small | 0.9999 | 0.5% |
| HIP vs Vulkan | Small | 1.0000 | <0.01% |

## Output Modalities

| Output | Channels | CUDA | HIP | Vulkan | CPU |
|--------|----------|------|-----|--------|-----|
| Depth + Confidence | 2 | ✅ | ✅ | ✅ | ✅ |
| Pose (CameraDec) | 9 | ✅ | ✅ | ✅ | ✅ |
| Pose conditioning (CameraEnc) | input | ✅ | ✅ | ✅ | — |
| Rays + Sky Seg (Aux DPT) | 7 | ✅ | ✅ | ✅ | ✅ |
| 3D Gaussians (GSDPT) | 38 | ✅ | ✅ | ✅ | ✅ |
| Metric Depth (Nested) | 1 | ✅ | — | — | — |
| Sky Segmentation (Nested) | 1 | ✅ | — | — | — |

## CLI Reference

All backends share the same CLI interface:

```
<binary> <model.safetensors> [options]

Input:
  -i <path>              Input image (JPEG/PNG/BMP)
  -d <id>                GPU device ID (default: 0)

Output:
  -o <path>              Output image (.png falsecolor, .pgm 16-bit, .exr float)
  --npy <path>           Raw depth as NumPy .npy
  --npy-dir <dir>        All outputs as .npy files

Modes:
  --full                 All output modalities
  --pose                 Pose estimation (CameraDec)
  --rays                 Rays + sky segmentation
  --gaussians            3D Gaussians (GSDPT)
  --pose-in "9 floats"   Pose conditioning (CameraEnc)

Debug:
  -v <level>             0=quiet, 1=timing, 2=per-block, 3=tensor dumps

Vulkan-specific:
  --shader-dir <path>    Path to compiled SPIR-V shaders
```

## Build Requirements

| Backend | Compiler | GPU SDK | Runtime Libs |
|---------|----------|---------|--------------|
| CUDA | gcc | — (NVRTC at runtime) | libcuda.so, libnvrtc.so |
| HIP | gcc | — (HIPRTC at runtime) | libamdhip64.so, libhiprtc.so |
| Vulkan | g++ (C++17) | Vulkan SDK (glslc) | libvulkan.so |
| CPU | gcc | — | — |

No GPU SDK is needed at compile time for CUDA/HIP — kernels are compiled at runtime via NVRTC/HIPRTC. Vulkan needs `glslc` to pre-compile GLSL→SPIR-V shaders.

## Preprocessing Pipeline

All backends implement identical preprocessing matching the official DA3:

1. **Aspect-ratio resize**: Longest side → 504px, preserve aspect ratio
2. **Patch alignment**: Round each dimension to nearest multiple of 14
3. **Area-based downscale**: Matches `cv2.INTER_AREA`
4. **ImageNet normalization**: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
5. **Bicubic pos_embed interpolation**: For non-square grids (DINOv2 style, offset=0.1)

## Architecture

```
RGB → area resize → patch embed (Conv2d 14×14) → CLS + pos_embed
  → camera_token injection at alt_start
  → N× ViT blocks:
      LayerNorm → QKV → QK-norm → 2D RoPE (local/global alternating)
      → FlashAttention → output proj → LayerScale + residual
      → LayerNorm → GELU MLP (or SwiGLU) → LayerScale + residual
  → cat_token features [local_x, backbone_norm(x)]
  → DualDPT head:
      4-level projection → deconv/identity/conv spatial alignment
      → adapter Conv3×3 → bottom-up RefineNet fusion
      → neck Conv3×3 (no ReLU) → bilinear upsample → UV posembed
      → out Conv3×3 + ReLU → Conv1×1 → exp() activation
  → bilinear upsample to original resolution
  → depth [H, W] + confidence [H, W]
```

## Reproducing Reference Comparison

### Setup PyTorch reference

```bash
cd /mnt/disk1 && mkdir -p da3_pytorch_ref && cd da3_pytorch_ref
python3 -m venv .venv && source .venv/bin/activate
pip install torch torchvision  # or with ROCm: --index-url https://download.pytorch.org/whl/rocm6.2.4
pip install "git+https://github.com/ByteDance-Seed/depth-anything-3.git"
pip install safetensors packaging
```

### Run PyTorch reference

```python
import os; os.environ["HF_HOME"] = "/mnt/disk1/hf_cache"
from depth_anything_3.api import DepthAnything3
model = DepthAnything3.from_pretrained("depth-anything/DA3-Small")
model = model.to("cpu").eval()
result = model.inference(["photo.jpg"])
# result.depth: numpy array [H, W]
```

### Run our implementation and compare

```bash
# Run HIP
./test_hip_da3 /mnt/disk1/models/DA3-Small/model.safetensors \
  -i photo.jpg --npy /tmp/hip_depth.npy

# Run Vulkan
VK_ICD_FILENAMES=/etc/vulkan/icd.d/radeon_icd_amdgpu.x86_64.json \
./test_vulkan_da3 /mnt/disk1/models/DA3-Small/model.safetensors \
  -i photo.jpg --npy /tmp/vk_depth.npy --shader-dir /path/to/build

# Compare
python3 -c "
import numpy as np, torch.nn.functional as F, torch
pt = np.load('pytorch_depth.npy')      # from PyTorch
hip = np.load('/tmp/hip_depth.npy')
vk = np.load('/tmp/vk_depth.npy')

# Resize to same shape for comparison
for name, arr in [('HIP', hip), ('VK', vk)]:
    rs = F.interpolate(torch.from_numpy(arr).unsqueeze(0).unsqueeze(0),
                       size=pt.shape, mode='bilinear', align_corners=False)[0,0].numpy()
    corr = np.corrcoef(pt.flat, rs.flat)[0,1]
    rel = np.sqrt(np.sum((pt-rs)**2) / np.sum(pt**2))
    print(f'{name} vs PT: corr={corr:.6f} rel_L2={rel:.4f}')
"
```

## File Layout

```
gemm/
├── common/
│   ├── depth_anything3.h      # CPU reference (~3300 lines, single-header)
│   ├── safetensors.h          # SafeTensors loader
│   └── stb_image.h            # Image I/O
├── cuda/da3/
│   ├── cuda_da3_runner.h/c    # CUDA implementation (NVRTC)
│   ├── test_cuda_da3.c        # Test harness
│   └── Makefile
├── rdna4/da3/
│   ├── hip_da3_runner.h/c     # HIP implementation (HIPRTC)
│   ├── test_hip_da3.c         # Test harness
│   ├── test_cpu_da3.c         # CPU-only test
│   └── Makefile
├── vulkan/da3/
│   ├── vulkan_da3_runner.h/cc # Vulkan implementation (SPIR-V)
│   ├── test_vulkan_da3.cc     # Test harness
│   └── README.md
├── vulkan/shaders/da3/
│   └── *.comp                 # 19 GLSL compute shaders
└── doc/
    └── da3.md                 # This document
```

## Known Limitations

1. **Vulkan per-dispatch sync**: Each dispatch calls `vkQueueWaitIdle` (~2.5ms overhead). Planned fix: DEVICE_LOCAL memory + command buffer batching.
2. **HIP first-run HIPRTC**: ~30-60 min kernel compilation on gfx1201 (cached after).
3. **Metric Depth**: Only CUDA supports nested models (HIP/Vulkan: depth/pose/rays/GS only).
4. **Vulkan GGUF**: SafeTensors only (CUDA/HIP support both).
5. **DA3-Giant on Vulkan**: Uses F32 GEMM (~8GB VRAM). HIP uses F16 (~5GB).
