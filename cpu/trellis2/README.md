# TRELLIS.2 Stage 1 — Structure Generation

CPU implementation of TRELLIS.2 Stage 1 (Microsoft, 1.3B params): generates a 64x64x64
occupancy grid from a single image via DINOv3 conditioning + DiT flow-matching diffusion.

**Pipeline**: Image -> DINOv3 [1029,1024] -> Stage 1 DiT (16^3 latent) -> Decoder -> 64^3 occupancy

## Prerequisites

### Model Weights

Download three safetensors files (total ~3.9 GB):

```bash
# 1. DINOv3 ViT-L/16 encoder (timm, ungated, 1.2 GB)
#    Repo: timm/vit_large_patch16_dinov3.lvd1689m
uvx --from huggingface_hub hf download \
  timm/vit_large_patch16_dinov3.lvd1689m model.safetensors \
  --local-dir /mnt/disk01/models/dinov3-vitl16

# 2. Stage 1 flow model (BF16, 2.5 GB)
#    Repo: microsoft/TRELLIS.2-4B
uvx --from huggingface_hub hf download \
  microsoft/TRELLIS.2-4B ckpts/ss_flow_img_dit_1_3B_64_bf16.safetensors \
  --local-dir /mnt/disk01/models/trellis2-4b

# 3. Structure decoder (F16, 141 MB)
#    Repo: microsoft/TRELLIS-image-large
uvx --from huggingface_hub hf download \
  microsoft/TRELLIS-image-large ckpts/ss_dec_conv3d_16l8_fp16.safetensors \
  --local-dir /mnt/disk01/models/trellis-image-large
```

After downloading, model files should be at:

| Component | Path | Size |
|-----------|------|------|
| DINOv3 ViT-L/16 | `/mnt/disk01/models/dinov3-vitl16/model.safetensors` | 1.2 GB |
| Stage 1 DiT | `/mnt/disk01/models/trellis2-4b/ckpts/ss_flow_img_dit_1_3B_64_bf16.safetensors` | 2.5 GB |
| Decoder | `/mnt/disk01/models/trellis-image-large/ckpts/ss_dec_conv3d_16l8_fp16.safetensors` | 141 MB |

### Build

```bash
cd cpu/trellis2

# Optimized build (default: Zen2)
make

# For your CPU
make ARCH=native

# Debug build with AddressSanitizer
make DEBUG=1
```

## End-to-End Run

### Quick start (full pipeline)

```bash
DINOV3=/mnt/disk01/models/dinov3-vitl16/model.safetensors
STAGE1=/mnt/disk01/models/trellis2-4b/ckpts/ss_flow_img_dit_1_3B_64_bf16.safetensors
DECODER=/mnt/disk01/models/trellis-image-large/ckpts/ss_dec_conv3d_16l8_fp16.safetensors

./test_trellis2 full $DINOV3 $STAGE1 $DECODER input.jpg \
  -t 4 -s 42 -o occupancy.npy
```

This runs the complete pipeline:
1. DINOv3 encodes the image to [1029, 1024] feature tokens
2. Stage 1 DiT generates a [8, 16, 16, 16] latent via 12-step Euler flow sampling with CFG
3. Decoder expands the latent to [64, 64, 64] occupancy logits

Intermediate files are saved automatically:
- `stage1_cond.npy` — DINOv3 features [1029, 1024]
- `stage1_latent.npy` — DiT output [8, 16, 16, 16]
- `occupancy.npy` — final occupancy logits [64, 64, 64]

### Step-by-step run (individual stages)

Each stage can be run independently for debugging or for reusing intermediate results.

**Step 1: DINOv3 encoding**

```bash
./test_trellis2 encode $DINOV3 input.jpg -t 4 -o features.npy
```

Output: `features.npy` — float32 [1029, 1024]

**Step 2: Stage 1 flow sampling**

```bash
./test_trellis2 stage1 $STAGE1 features.npy -t 4 -s 42 -o latent.npy
```

Output: `latent.npy` — float32 [8, 16, 16, 16]

Options:
- `-s <seed>` — random seed for initial noise (default: 42)
- `-t <threads>` — number of CPU threads (default: 4)

**Step 3: Structure decoding**

```bash
./test_trellis2 decode $DECODER latent.npy -t 4 -o occupancy.npy
```

Output: `occupancy.npy` — float32 [64, 64, 64] occupancy logits

### Interpreting the output

The output is a 64x64x64 grid of logits (not probabilities). Positive values indicate
occupied voxels:

```python
import numpy as np
occupancy = np.load("occupancy.npy")
occupied = occupancy > 0.0  # simple threshold
print(f"Occupied: {occupied.sum()}/{occupied.size} ({100*occupied.sum()/occupied.size:.1f}%)")
```

## Performance

Measured on Zen2 (AMD EPYC 7742) with 4 threads:

| Stage | Time | Peak RAM |
|-------|------|----------|
| DINOv3 encode (512x512) | ~3.5 min | ~1.2 GB |
| Stage 1 DiT (12 steps x 2 CFG) | ~9 hours | ~8.4 GB |
| Structure decoder (16^3 -> 64^3) | ~22 min | ~200 MB |

The DiT sampling is the bottleneck (1.25B params, 24 forward passes, 30 blocks each
with large GEMMs at 4096 tokens x 1536 dim). Potential optimizations:

- Tiled F32 GEMM with better cache utilization
- Threaded Conv3d for the decoder
- Reduced CFG steps or fewer Euler steps
- F16 weight retention (avoid BF16->F32 conversion, use F16 GEMM path)

## Architecture

### Stage 1 DiT (SparseStructureFlowModel)

- 30 x ModulatedTransformerCrossBlock
- model_channels=1536, heads=12, head_dim=128, FFN=8192
- adaLN with shared modulation + per-block bias
- QK RMSNorm (per-head gamma [12, 128])
- 3D RoPE: 21 freqs/axis for 16^3 grid, theta=10000
- Cross-attention: Q from tokens, KV from DINOv3 [1029, 1024]
- Flow matching: 12 Euler steps, CFG scale=7.5, rescale_t=5.0

### Structure Decoder

```
input_layer:  Conv3d(8 -> 512, k=3)           16^3
middle_block: 2x ResBlock3d(512)               16^3
blocks.0-1:   2x ResBlock3d(512)               16^3
blocks.2:     Conv3d(512 -> 1024, k=3)         16^3
              pixel_shuffle_3d(factor=2)        -> 32^3 (128 ch)
blocks.3-4:   2x ResBlock3d(128)               32^3
blocks.5:     Conv3d(128 -> 256, k=3)          32^3
              pixel_shuffle_3d(factor=2)        -> 64^3 (32 ch)
blocks.6-7:   2x ResBlock3d(32)                64^3
out_layer:    GroupNorm(32) -> SiLU -> Conv3d(32 -> 1, k=3)
```

### DINOv3 ViT-L/16

- 24 blocks, embed_dim=1024, heads=16, head_dim=64, FFN=4096
- Patch size 16, image size 512x512 -> 32x32 = 1024 patches
- 1 CLS + 4 register tokens + 1024 patches = 1029 tokens
- RoPE only (no learned positional embeddings)

## Source Files

| File | Description |
|------|-------------|
| `common/cpu_compute.h` | Shared CPU primitives (attention, cross-attention, LayerNorm, GELU, RoPE) |
| `common/dinov3.h` | DINOv3 ViT-L/16 encoder |
| `common/trellis2_dit.h` | DiT blocks, F32 GEMM, adaLN, QK RMSNorm, 3D RoPE, safetensors loading |
| `common/trellis2_stage1.h` | Euler flow sampling with CFG, RNG |
| `common/trellis2_ss_decoder.h` | Dense Conv3D, GroupNorm, pixel_shuffle_3d, ResBlock3d, decoder pipeline |
| `cpu/trellis2/test_trellis2.c` | Test harness (full/stage1/decode/encode modes, .npy I/O) |
| `cpu/trellis2/verify_trellis2.py` | Inspect safetensors weights, compare .npy outputs, visualize occupancy |

## Verification

### Inspect model weights

```bash
uvx --with safetensors --with numpy python3 verify_trellis2.py --inspect $STAGE1
uvx --with safetensors --with numpy python3 verify_trellis2.py --inspect $DECODER
```

### Compare .npy outputs

```bash
uvx --with safetensors --with numpy python3 verify_trellis2.py --compare ref.npy c_output.npy
```

### Visualize occupancy

```bash
uvx --with safetensors --with numpy --with matplotlib python3 verify_trellis2.py \
  --visualize occupancy.npy --threshold 0.0
```
